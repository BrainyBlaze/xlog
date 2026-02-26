"""dILP trainer — train_only() entry point.

Integrates valid_candidates, adaptive temperature, entropy regularization,
and multi-start into a single training API. Uses the dense N³ mask path
(alpha); factorized scoring deferred to RC.

See docs/plans/2026-02-26-dilp-hardening-design.md §5.1.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

import pyxlog
from pyxlog.ilp.entropy import entropy_weight_at_step, normalized_entropy
from pyxlog.ilp.exceptions import IlpCandidateError, IlpConfigError, IlpTrainingError
from pyxlog.ilp.temperature import AdaptiveTempController
from pyxlog.ilp.types import (
    CandidateMapEntry,
    LearnedArtifact,
    StepRecord,
    TrainConfig,
    TrainResult,
    TrainTelemetry,
)


def train_only(
    source: str,
    mask_name: str,
    positives: list[tuple[str, list[int]]],
    negatives: list[tuple[str, list[int]]],
    config: TrainConfig = TrainConfig(),
) -> TrainResult:
    """Train a single learnable rule via differentiable ILP.

    Returns a TrainResult with convergence status, discovered rule,
    confidence metrics, and optional telemetry.
    """
    _validate_inputs(source, mask_name, positives, negatives, config)

    if config.seed is not None:
        torch.manual_seed(config.seed)

    attempts: list[_AttemptResult] = []
    global_steps = 0
    numeric_failures = 0

    for attempt_idx in range(config.max_attempts):
        if global_steps >= config.global_step_limit:
            break

        remaining = config.global_step_limit - global_steps
        step_budget = min(config.step_budget_per_attempt, remaining)
        if step_budget <= 0:
            break

        try:
            result = _run_single_attempt(
                source, mask_name, positives, negatives,
                config, attempt_idx, step_budget,
            )
            attempts.append(result)
            global_steps += result.steps_used
        except _NumericFailure:
            numeric_failures += 1
            global_steps += 1  # count as at least 1 step
            if numeric_failures >= config.max_numeric_failures:
                raise IlpTrainingError(
                    "numeric_instability: too many NaN/Inf failures",
                    {"attempt": attempt_idx, "numeric_failures": numeric_failures},
                )

    return _select_winner(attempts, config, global_steps)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_inputs(
    source: str,
    mask_name: str,
    positives: list[tuple[str, list[int]]],
    negatives: list[tuple[str, list[int]]],
    config: TrainConfig,
) -> None:
    if not positives:
        raise IlpConfigError("positives must be non-empty")

    # Check for contradictions: same fact in both positives and negatives
    pos_set = {(r, tuple(v)) for r, v in positives}
    neg_set = {(r, tuple(v)) for r, v in negatives}
    overlap = pos_set & neg_set
    if overlap:
        raise IlpConfigError(
            f"positives and negatives contradict: {overlap}"
        )

    # Alpha scope: recursive candidates are beta-only
    if config.allow_recursive_candidates:
        raise IlpConfigError(
            "allow_recursive_candidates=True is a beta-only feature"
        )


# ---------------------------------------------------------------------------
# Single attempt
# ---------------------------------------------------------------------------

class _NumericFailure(Exception):
    pass


class _AttemptResult:
    __slots__ = (
        "converged", "discovered_rule", "steps_used",
        "precision", "recall", "confidence_margin",
        "top_k_concentration", "soft_probs", "logits",
        "candidate_map", "telemetry_steps", "argmax_ijk",
    )

    def __init__(self) -> None:
        self.converged = False
        self.discovered_rule: str | None = None
        self.steps_used = 0
        self.precision = 0.0
        self.recall = 0.0
        self.confidence_margin = 0.0
        self.top_k_concentration = 0.0
        self.soft_probs: list[float] = []
        self.logits: list[float] = []
        self.candidate_map: list[CandidateMapEntry] = []
        self.telemetry_steps: list[StepRecord] = []
        self.argmax_ijk: tuple[int, int, int] | None = None


def _run_single_attempt(
    source: str,
    mask_name: str,
    positives: list[tuple[str, list[int]]],
    negatives: list[tuple[str, list[int]]],
    config: TrainConfig,
    attempt_idx: int,
    step_budget: int,
) -> _AttemptResult:
    prog = pyxlog.IlpProgramFactory.compile(
        source, device=config.device, memory_mb=config.memory_mb,
        max_active_rules=config.max_active_rules,
    )
    n = prog.ilp_schema_size()
    rel_names = prog.ilp_relation_names()

    # Alpha always passes False for recursive candidates
    candidates = prog.valid_candidates(mask_name, False)
    if not candidates:
        raise IlpCandidateError("No valid candidates for mask")

    C = len(candidates)
    candidate_map = [
        CandidateMapEntry(
            id=c["id"], i=c["i"], j=c["j"], k=c["k"],
            left_name=c["left_name"], right_name=c["right_name"],
            head_name=c["head_name"],
        )
        for c in candidates
    ]

    W = torch.randn((n, n, n), requires_grad=True, device="cuda")
    optimizer = torch.optim.Adam([W], lr=0.1)

    temp_controller = AdaptiveTempController(
        tau_start=config.tau_start,
        tau_floor=config.tau_floor,
        plateau_window=config.plateau_window,
        plateau_threshold=config.plateau_threshold,
        warmup_increment=config.warmup_increment,
        trap_disc_threshold=config.trap_disc_threshold,
        trap_progress_window=config.trap_progress_window,
        total_steps=step_budget,
    )

    prev_argmax: tuple[int, int, int] | None = None
    stable_count = 0
    result = _AttemptResult()
    result.candidate_map = candidate_map

    for step in range(step_budget):
        optimizer.zero_grad()

        M_hard, M_soft = _build_budget_aware_mask(
            W, temp_controller.tau, config.max_active_rules,
        )

        # Check for NaN/Inf
        if torch.isnan(M_soft).any() or torch.isinf(M_soft).any():
            raise _NumericFailure()

        prog.set_rule_mask(
            mask_name,
            M_hard.detach().contiguous().view(-1),
            M_soft.detach().contiguous().view(-1),
            n,
        )
        prog.evaluate()

        # Compute task loss
        loss = _compute_loss(prog, M_soft, positives, negatives, rel_names, n)

        # Entropy regularization
        ent_weight = entropy_weight_at_step(
            step, step_budget,
            start=config.entropy_weight_start,
            end=config.entropy_weight_end,
        )
        if ent_weight > 0 and C > 1:
            # Extract soft probs for valid candidates
            cand_probs = torch.stack([M_soft[c["i"], c["j"], c["k"]] for c in candidates])
            cand_sum = cand_probs.sum()
            if cand_sum > 1e-8:
                cand_probs_norm = cand_probs / cand_sum
                ent = normalized_entropy(cand_probs_norm, C)
                loss = loss - ent_weight * ent

        # Check for NaN/Inf in loss
        if torch.isnan(loss) or torch.isinf(loss):
            raise _NumericFailure()

        if loss.requires_grad:
            loss.backward()
            optimizer.step()

        # Decode argmax
        argmax_ijk = _decode_argmax_ijk(W, n)
        if argmax_ijk == prev_argmax:
            stable_count += 1
        else:
            stable_count = 0
        prev_argmax = argmax_ijk

        # Compute witness coverage for temperature controller
        witness_count = sum(
            1 for rel, vals in positives if prog.fact_exists(rel, vals)
        )
        witness_coverage = witness_count / len(positives)

        disc = M_soft.max(dim=-1)[0].mean().item()
        temp_controller.step(
            loss=loss.item(), disc=disc, witness_coverage=witness_coverage,
        )

        # Telemetry
        if config.telemetry_level >= 1:
            i, j, k = argmax_ijk
            argmax_rule = f"{rel_names[i]}+{rel_names[j]}->{rel_names[k]}"
            # Compute entropy for telemetry
            with torch.no_grad():
                tel_cand_probs = torch.stack(
                    [M_soft[c["i"], c["j"], c["k"]] for c in candidates]
                )
                tel_sum = tel_cand_probs.sum()
                if tel_sum > 1e-8 and C > 1:
                    tel_normed = tel_cand_probs / tel_sum
                    tel_ent = normalized_entropy(tel_normed, C).item()
                else:
                    tel_ent = 0.0
            result.telemetry_steps.append(StepRecord(
                step=step,
                loss=loss.item(),
                argmax_rule=argmax_rule,
                discreteness=disc,
                temperature=temp_controller.tau,
                entropy=tel_ent,
                stable_count=stable_count,
                active_candidates=C,
            ))

        # Convergence check
        if stable_count >= 5:
            converged = _check_convergence(
                prog, W, mask_name, positives, negatives,
                rel_names, n, argmax_ijk,
            )
            if converged:
                result.converged = True
                result.precision = 1.0
                result.recall = 1.0
                i, j, k = argmax_ijk
                result.discovered_rule = _format_rule(
                    rel_names[i], rel_names[j], rel_names[k],
                )
                result.steps_used = step + 1
                result.argmax_ijk = argmax_ijk
                _fill_metrics(result, M_soft, candidates, W, n)
                return result

    # Did not converge — compute partial recall from witness coverage
    result.steps_used = step_budget
    result.recall = witness_coverage
    if prev_argmax is not None:
        i, j, k = prev_argmax
        result.discovered_rule = _format_rule(
            rel_names[i], rel_names[j], rel_names[k],
        )
        result.argmax_ijk = prev_argmax
    _fill_metrics(result, M_soft, candidates, W, n)
    return result


# ---------------------------------------------------------------------------
# Mask building (dense N³, same as showcase)
# ---------------------------------------------------------------------------

def _build_budget_aware_mask(
    W: torch.Tensor, tau: float, budget: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ST-Gumbel-Softmax with global top-k hard selection."""
    M_soft = F.gumbel_softmax(W, tau=tau, hard=False, dim=-1)

    flat = M_soft.view(-1)
    k = min(budget, flat.numel())
    _, topk_idx = flat.topk(k)

    M_hard_flat = torch.zeros_like(flat)
    M_hard_flat[topk_idx] = 1.0
    M_hard = M_hard_flat.view_as(M_soft)

    return M_hard, M_soft


# ---------------------------------------------------------------------------
# Loss computation (per-fact surrogate credit)
# ---------------------------------------------------------------------------

def _compute_loss(
    prog,
    M_soft: torch.Tensor,
    positives: list[tuple[str, list[int]]],
    negatives: list[tuple[str, list[int]]],
    rel_names: list[str],
    n: int,
) -> torch.Tensor:
    """Per-fact surrogate loss with missed-positive penalty."""
    loss = torch.tensor(0.0, device="cuda")

    for rel_name, values in positives:
        contributing = prog.tagged_entries_containing_fact(rel_name, values)
        if contributing:
            credit = sum(M_soft[i, j, k] for (i, j, k) in contributing)
            loss = loss + (-torch.log(credit.clamp(min=1e-8)))
        else:
            k_idx = rel_names.index(rel_name)
            loss = loss + (-M_soft[:, :, k_idx].sum() / (n * n))

    for rel_name, values in negatives:
        contributing = prog.tagged_entries_containing_fact(rel_name, values)
        if contributing:
            credit = sum(M_soft[i, j, k] for (i, j, k) in contributing)
            loss = loss + (-torch.log((1.0 - credit).clamp(min=1e-8)))

    return loss


# ---------------------------------------------------------------------------
# Convergence check
# ---------------------------------------------------------------------------

def _check_convergence(
    prog,
    W: torch.Tensor,
    mask_name: str,
    positives: list[tuple[str, list[int]]],
    negatives: list[tuple[str, list[int]]],
    rel_names: list[str],
    n: int,
    argmax_ijk: tuple[int, int, int],
) -> bool:
    """Three-gated convergence: stable argmax + all derived + argmax-only check."""
    # Gate 1: all positives derived under current mask
    all_derived = all(
        prog.fact_exists(rel, vals) for rel, vals in positives
    )
    if not all_derived:
        return False

    # Gate 2: no negatives derived
    any_neg = any(
        prog.fact_exists(rel, vals) for rel, vals in negatives
    )
    if any_neg:
        return False

    # Gate 3: argmax-only re-evaluation
    i, j, k = argmax_ijk
    M_check = torch.zeros((n, n, n), device="cuda")
    M_check[i, j, k] = 1.0
    flat_check = M_check.contiguous().view(-1)
    prog.set_rule_mask(mask_name, flat_check, flat_check, n)
    prog.evaluate()

    argmax_pos_ok = all(
        prog.fact_exists(rel, vals) for rel, vals in positives
    )
    if not argmax_pos_ok:
        return False

    # Gate 4: argmax-only must also not derive negatives
    argmax_neg = any(
        prog.fact_exists(rel, vals) for rel, vals in negatives
    )
    return not argmax_neg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_argmax_ijk(W: torch.Tensor, n: int) -> tuple[int, int, int]:
    """Decode the argmax of W into (i, j, k) triple."""
    with torch.no_grad():
        flat = W.view(-1)
        idx = flat.argmax().item()
        i = idx // (n * n)
        j = (idx % (n * n)) // n
        k = idx % n
    return (i, j, k)


def _format_rule(left: str, right: str, head: str) -> str:
    """Format a discovered (left, right, head) triple as a Datalog rule string."""
    return f"{head}(X, Y) :- {left}(X, Z), {right}(Z, Y)."


def _fill_metrics(
    result: _AttemptResult,
    M_soft: torch.Tensor,
    candidates: list[dict],
    W: torch.Tensor,
    n: int,
) -> None:
    """Fill confidence metrics on an attempt result."""
    with torch.no_grad():
        # Extract candidate soft probs
        cand_probs = torch.stack(
            [M_soft[c["i"], c["j"], c["k"]] for c in candidates]
        )
        total = cand_probs.sum()
        if total > 1e-8:
            normed = cand_probs / total
            sorted_probs, _ = normed.sort(descending=True)
            top1 = sorted_probs[0].item()
            top2 = sorted_probs[1].item() if len(sorted_probs) > 1 else 0.0
            result.confidence_margin = top1 - top2
        else:
            result.confidence_margin = 0.0

        # Top-k concentration
        k = min(32, len(cand_probs))
        topk_vals, _ = cand_probs.topk(k)
        total_mass = cand_probs.sum().item()
        if total_mass > 1e-8:
            result.top_k_concentration = topk_vals.sum().item() / total_mass
        else:
            result.top_k_concentration = 0.0

        # Store raw values for artifact
        result.soft_probs = cand_probs.cpu().tolist()
        result.logits = [
            W[c["i"], c["j"], c["k"]].item() for c in candidates
        ]


# ---------------------------------------------------------------------------
# Winner selection
# ---------------------------------------------------------------------------

def _select_winner(
    attempts: list[_AttemptResult],
    config: TrainConfig,
    global_steps: int,
) -> TrainResult:
    """Select best attempt via deterministic tie-break chain."""
    if not attempts:
        return TrainResult(total_steps=global_steps, attempt_count=0)

    converged = [a for a in attempts if a.converged]

    if converged:
        # Tie-break: recall → precision → fewer steps → lower candidate id
        winner = max(converged, key=lambda a: (
            a.recall, a.precision, -a.steps_used,
        ))
    else:
        # No convergence — pick attempt with most witness coverage (best recall proxy)
        winner = max(attempts, key=lambda a: (a.recall, -a.steps_used))

    # Compute rule_frequency: fraction of converged attempts finding same rule
    same_rule_count = sum(
        1 for a in attempts
        if a.converged and a.discovered_rule == winner.discovered_rule
    )
    rule_freq = same_rule_count / len(attempts) if attempts else 0.0

    # Compute precision/recall for winner
    precision, recall = _compute_precision_recall(winner, config)

    return TrainResult(
        converged=winner.converged,
        discovered_rule=winner.discovered_rule,
        attempt_count=len(attempts),
        total_steps=global_steps,
        precision=precision,
        recall=recall,
        confidence_margin=winner.confidence_margin,
        top_k_concentration=winner.top_k_concentration,
        rule_frequency=rule_freq,
        single_attempt=len(attempts) == 1,
        artifact=LearnedArtifact(
            candidate_map=winner.candidate_map,
            logits=winner.logits,
            soft_probs=winner.soft_probs,
            selected_hard=[],
            discovered_rule=winner.discovered_rule or "",
            telemetry=TrainTelemetry(steps=winner.telemetry_steps),
        ),
    )


def _compute_precision_recall(
    result: _AttemptResult,
    config: TrainConfig,
) -> tuple[float, float]:
    """Compute precision and recall for the winner's discovered rule.

    Since we validate convergence by checking all positives are derived
    and no negatives are derived, a converged result has recall=1.0 and
    precision=1.0 by construction (on the training set).
    """
    if result.converged:
        return 1.0, 1.0
    return 0.0, 0.0
