"""dILP trainer -- train_only() entry point.

Integrates valid_candidates, adaptive temperature, entropy regularization,
and multi-start into a single training API. Supports dense N^3 mask path
(alpha) and sparse candidate-indexed path (beta default) via MaskBackend.

See docs/plans/2026-02-26-dilp-hardening-design.md.
"""
from __future__ import annotations

import math
import time

import torch

import pyxlog
from pyxlog.ilp.backend import DenseMaskBackend, SparseMaskBackend
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
    *,
    _compute_holdout: bool = True,
) -> TrainResult:
    """Train a single learnable rule via differentiable ILP.

    Returns a TrainResult with convergence status, discovered rule,
    confidence metrics, and optional telemetry.
    """
    _validate_inputs(source, mask_name, positives, negatives, config)

    if config.deterministic:
        _set_deterministic_cuda(config.seed)

    attempts: list[_AttemptResult] = []
    global_steps = 0
    numeric_failures = 0

    for attempt_idx in range(config.max_attempts):
        if global_steps >= config.global_step_limit:
            break

        if config.deterministic:
            _seed_for_attempt(config.seed, attempt_idx)

        remaining = config.global_step_limit - global_steps
        step_budget = min(config.step_budget_per_attempt, remaining)
        if step_budget <= 0:
            break

        try:
            result = _run_single_attempt(
                source,
                mask_name,
                positives,
                negatives,
                config,
                attempt_idx,
                step_budget,
            )
            attempts.append(result)
            global_steps += result.steps_used
        except _NumericFailure:
            numeric_failures += 1
            global_steps += 1  # count as at least 1 step
            if numeric_failures >= config.max_numeric_failures:
                raise IlpTrainingError(
                    "numeric_instability: too many NaN/Inf failures",
                    {
                        "attempt": attempt_idx,
                        "step": None,
                        "C": 0,
                        "k": config.max_active_rules,
                        "device_name": _device_name(config),
                        "allocated_bytes": _allocated_bytes(config.device),
                        "terminal_reason": "numeric_instability",
                    },
                )
        except _AttemptRuntimeFailure as e:
            context = e.context
            raise IlpTrainingError(f"{e.terminal_reason}: {e.cause}", context)

    return _select_winner(
        attempts,
        source,
        mask_name,
        positives,
        negatives,
        config,
        global_steps,
        _compute_holdout=_compute_holdout,
    )


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


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def _set_deterministic_cuda(seed: int | None) -> None:
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends.cudnn, "deterministic"):
            torch.backends.cudnn.deterministic = True


def _seed_for_attempt(seed: int | None, attempt_idx: int) -> None:
    if seed is None:
        return
    attempt_seed = seed + attempt_idx
    torch.manual_seed(attempt_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(attempt_seed)


def _device_name(config: TrainConfig) -> str:
    if not torch.cuda.is_available():
        return "cpu"
    try:
        return torch.cuda.get_device_name(config.device)
    except Exception:
        return f"cuda:{config.device}"


def _allocated_bytes(device_idx: int) -> int:
    if not torch.cuda.is_available():
        return 0
    try:
        return int(torch.cuda.memory_allocated(device_idx))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Single attempt
# ---------------------------------------------------------------------------


class _NumericFailure(Exception):
    pass


class _AttemptRuntimeFailure(Exception):
    def __init__(self, terminal_reason: str, context: dict, cause: Exception):
        super().__init__(terminal_reason)
        self.terminal_reason = terminal_reason
        self.context = context
        self.cause = cause


class _AttemptResult:
    __slots__ = (
        "converged", "discovered_rule", "steps_used",
        "precision", "recall", "confidence_margin",
        "top_k_concentration", "soft_probs", "logits",
        "candidate_map", "telemetry_steps", "argmax_ijk",
        "selected_hard", "telemetry_timings",
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
        self.telemetry_timings: dict[str, float] = {}
        self.argmax_ijk: tuple[int, int, int] | None = None
        self.selected_hard: list[int] = []


def _run_single_attempt(
    source: str,
    mask_name: str,
    positives: list[tuple[str, list[int]]],
    negatives: list[tuple[str, list[int]]],
    config: TrainConfig,
    attempt_idx: int,
    step_budget: int,
) -> _AttemptResult:
    context_base = {
        "attempt": attempt_idx,
        "C": 0,
        "k": config.max_active_rules,
        "device_name": _device_name(config),
        "allocated_bytes": _allocated_bytes(config.device),
    }
    try:
        prog = pyxlog.IlpProgramFactory.compile(
            source, device=config.device, memory_mb=config.memory_mb,
            max_active_rules=config.max_active_rules,
        )
        n = prog.ilp_schema_size()
        rel_names = prog.ilp_relation_names()

        allow_recursive = config.allow_recursive_candidates
        candidates = prog.valid_candidates(mask_name, allow_recursive)
        if not candidates:
            raise IlpCandidateError("No valid candidates for mask")

        C = len(candidates)
        context_base["C"] = C
        candidate_map = [
            CandidateMapEntry(
                id=c["id"], i=c["i"], j=c["j"], k=c["k"],
                left_name=c["left_name"], right_name=c["right_name"],
                head_name=c["head_name"],
            )
            for c in candidates
        ]

        # Build (i,j,k) -> candidate index lookup for loss computation
        ijk_to_cidx: dict[tuple[int, int, int], int] = {}
        for ci, c in enumerate(candidates):
            ijk_to_cidx[(c["i"], c["j"], c["k"])] = ci

        # Select backend based on config
        backend = SparseMaskBackend() if not config.debug_dense_mask else DenseMaskBackend()
        device = f"cuda:{config.device}" if torch.cuda.is_available() else "cpu"
        W = backend.init_weights(C, n, device)
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
        forward_hist_us: list[float] = []
        allocated_hist_bytes: list[float] = []
        result = _AttemptResult()
        result.candidate_map = candidate_map
        witness_coverage = 0.0
        best_witness_coverage = 0.0

        for step in range(step_budget):
            try:
                optimizer.zero_grad()
                prog.reset_d2h_transfer_count()

                cand_probs, selected_hard = backend.apply_mask(
                    prog, mask_name, W, temp_controller.tau,
                    config.max_active_rules, candidates, n,
                    allow_recursive=allow_recursive,
                )
                result.selected_hard = [int(i) for i in selected_hard if int(i) >= 0]

                # Check for NaN/Inf
                if torch.isnan(cand_probs).any() or torch.isinf(cand_probs).any():
                    raise _NumericFailure()

                forward_us = _evaluate_forward_us(prog)
                forward_hist_us.append(forward_us)
                allocated_hist_bytes.append(float(_allocated_bytes(config.device)))
                forward_p95_us = _percentile_95(forward_hist_us)

                # Hard-negative mining: sample false positives every 20 steps
                if config.max_mined_negatives > 0 and step > 0 and step % 20 == 0:
                    head_name = candidates[0]["head_name"]
                    fps = prog.sample_false_positives(
                        head_name, positives, config.max_mined_negatives,
                    )
                    mined_negs = [(head_name, fp) for fp in fps]
                    all_negatives = list(negatives) + mined_negs
                    # Reset D2H counter after mining (download_column_* was called)
                    prog.reset_d2h_transfer_count()
                else:
                    all_negatives = negatives

                # Compute task loss using candidate probs
                loss = _compute_loss_from_candidates(
                    prog, cand_probs, positives, all_negatives, candidates, ijk_to_cidx,
                )

                # Entropy regularization
                ent_weight = entropy_weight_at_step(
                    step, step_budget,
                    start=config.entropy_weight_start,
                    end=config.entropy_weight_end,
                )
                if ent_weight > 0 and C > 1:
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

                # Decode argmax via backend
                argmax_idx = backend.decode_argmax(W, candidates, n)
                argmax_ijk = (
                    candidates[argmax_idx]["i"],
                    candidates[argmax_idx]["j"],
                    candidates[argmax_idx]["k"],
                )
                if argmax_ijk == prev_argmax:
                    stable_count += 1
                else:
                    stable_count = 0
                prev_argmax = argmax_ijk

                # Batch witness coverage -- GPU-side fact membership
                pos_by_rel: dict[str, list[list[int]]] = {}
                pos_indices_by_rel: dict[str, list[int]] = {}
                for idx, (rel, vals) in enumerate(positives):
                    pos_by_rel.setdefault(rel, []).append(vals)
                    pos_indices_by_rel.setdefault(rel, []).append(idx)

                witness_mask = [False] * len(positives)
                for rel_name, facts_list in pos_by_rel.items():
                    mask = prog.batch_fact_membership(rel_name, facts_list)
                    for local_idx, found in enumerate(mask):
                        global_idx = pos_indices_by_rel[rel_name][local_idx]
                        witness_mask[global_idx] = found

                witness_count = sum(witness_mask)
                witness_coverage = witness_count / len(positives)
                best_witness_coverage = max(best_witness_coverage, witness_coverage)

                # Discreteness metric -- max candidate prob
                with torch.no_grad():
                    disc = cand_probs.max().item() if cand_probs.numel() > 0 else 0.0
                temp_controller.step(
                    loss=float(loss.item()), disc=disc, witness_coverage=witness_coverage,
                )

                # Telemetry
                if config.telemetry_level >= 1:
                    i, j, k = argmax_ijk
                    argmax_rule = f"{rel_names[i]}+{rel_names[j]}->{rel_names[k]}"
                    tel_sum = cand_probs.sum()
                    if tel_sum > 1e-8 and C > 1:
                        tel_normed = cand_probs / tel_sum
                        tel_ent = normalized_entropy(tel_normed, C).item()
                    else:
                        tel_ent = 0.0
                    result.telemetry_steps.append(StepRecord(
                        step=step,
                        loss=float(loss.item()),
                        argmax_rule=argmax_rule,
                        discreteness=disc,
                        temperature=temp_controller.tau,
                        entropy=tel_ent,
                        stable_count=stable_count,
                        forward_p95_us=_percentile_95(forward_hist_us),
                        active_candidates=C,
                    ))

                # Convergence check
                if stable_count >= 5:
                    converged = _check_convergence(
                        prog, W, mask_name, positives, negatives,
                        rel_names, n, argmax_ijk,
                        perform_mask_checks=not isinstance(backend, SparseMaskBackend),
                    )
                    if converged:
                        result.converged = True
                        result.precision = 1.0
                        result.recall = 1.0
                        i, j, k = argmax_ijk
                        result.discovered_rule = _format_rule(
                            rel_names[i], rel_names[j], rel_names[k],
                        )
                        result.telemetry_timings = {
                            "forward_p95_us": forward_p95_us,
                            "allocated_bytes_p95": _percentile_95(allocated_hist_bytes),
                            "allocated_bytes_max": max(allocated_hist_bytes),
                        }
                        result.steps_used = step + 1
                        result.argmax_ijk = argmax_ijk
                        _fill_metrics(result, cand_probs, candidates, W, backend)
                        return result

                # --- D2H hard gate: assert zero column downloads in step body ---
                d2h_count = prog.d2h_transfer_count()
                if d2h_count > 0:
                    context = _attempt_context(
                        attempt=attempt_idx,
                        step=step,
                        C=C,
                        config=config,
                        terminal_reason="d2h_gate_violation",
                    )
                    context["d2h_count"] = d2h_count
                    raise _AttemptRuntimeFailure(
                        terminal_reason="d2h_gate_violation",
                        context=context,
                        cause=RuntimeError("d2h transfer observed in hot step loop"),
                    )

            except _NumericFailure:
                raise
            except _AttemptRuntimeFailure:
                raise
            except Exception as e:
                context = _attempt_context(
                    attempt=attempt_idx,
                    step=step,
                    C=C,
                    config=config,
                    terminal_reason=_classify_failure_reason(e),
                )
                context["error"] = str(e)
                raise _AttemptRuntimeFailure(
                    terminal_reason=_classify_failure_reason(e),
                    context=context,
                    cause=e,
                )

        # Did not converge -- compute partial recall from best witness coverage
        result.steps_used = step_budget
        result.recall = best_witness_coverage
        if prev_argmax is not None:
            i, j, k = prev_argmax
            result.discovered_rule = _format_rule(
                rel_names[i], rel_names[j], rel_names[k],
            )
            result.argmax_ijk = prev_argmax
        _fill_metrics(result, cand_probs, candidates, W, backend)
        if forward_hist_us:
            result.telemetry_timings = {
                "forward_p95_us": _percentile_95(forward_hist_us),
                "allocated_bytes_p95": _percentile_95(allocated_hist_bytes),
                "allocated_bytes_max": max(allocated_hist_bytes),
            }
        return result
    except _NumericFailure:
        raise
    except IlpCandidateError:
        raise
    except _AttemptRuntimeFailure:
        raise
    except Exception as e:
        # Wrap compile/early failures
        context = {
            "attempt": attempt_idx,
            "C": C if "C" in locals() else 0,
            "k": config.max_active_rules,
            "device_name": _device_name(config),
            "allocated_bytes": _allocated_bytes(config.device),
            "step": None,
            "terminal_reason": _classify_failure_reason(e),
        }
        raise _AttemptRuntimeFailure(
            terminal_reason=_classify_failure_reason(e),
            context=context,
            cause=e,
        )


def _classify_failure_reason(exc: Exception) -> str:
    msg = str(exc).lower()
    if "out of memory" in msg and "cuda" in msg:
        return "cuda_oom"
    if "runtimeerror" in msg and "cuda" in msg:
        return "cuda_error"
    if "device-side" in msg:
        return "device_side_error"
    if "cudnn" in msg:
        return "cuda_error"
    if "d2h_gate_violation" in msg:
        return "d2h_gate_violation"
    return "runtime_error"


def _attempt_context(
    attempt: int,
    step: int | None,
    C: int,
    config: TrainConfig,
    terminal_reason: str,
) -> dict:
    return {
        "attempt": attempt,
        "step": step,
        "C": C,
        "k": config.max_active_rules,
        "device_name": _device_name(config),
        "allocated_bytes": _allocated_bytes(config.device),
        "terminal_reason": terminal_reason,
    }


def _evaluate_forward_us(prog: object) -> float:
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        prog.evaluate()
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end)) * 1000.0

    t0 = time.perf_counter()
    prog.evaluate()
    return (time.perf_counter() - t0) * 1_000_000.0


def _percentile_95(values: list[float]) -> float:
    """Compute the 95th percentile using nearest-rank semantics."""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sorted_values = sorted(values)
    idx = min(len(sorted_values) - 1, max(0, math.ceil(0.95 * len(sorted_values)) - 1))
    return float(sorted_values[idx])


# ---------------------------------------------------------------------------
# Loss computation (per-fact surrogate credit via candidate probs)
# ---------------------------------------------------------------------------

def _compute_loss_from_candidates(
    prog,
    cand_probs: torch.Tensor,
    positives: list[tuple[str, list[int]]],
    negatives: list[tuple[str, list[int]]],
    candidates: list[dict],
    ijk_to_cidx: dict[tuple[int, int, int], int],
) -> torch.Tensor:
    """Per-fact surrogate loss using candidate-level
    soft probs and a (i,j,k)->candidate_index mapping.

    Works with both dense and sparse backends by using candidate-level
    soft probs and a (i,j,k)->candidate_index mapping.
    """
    device = cand_probs.device
    loss = torch.tensor(0.0, device=device)
    C = len(candidates)

    # Group facts by relation for batch API calls
    if positives:
        pos_by_rel: dict[str, list[list[int]]] = {}
        for rel_name, values in positives:
            pos_by_rel.setdefault(rel_name, []).append(values)

        for rel_name, facts_list in pos_by_rel.items():
            credits = prog.batch_tagged_credit(rel_name, facts_list)
            for fact_idx, contributing in enumerate(credits):
                if contributing:
                    credit = torch.tensor(0.0, device=device)
                    for (i, j, k) in contributing:
                        ci = ijk_to_cidx.get((i, j, k))
                        if ci is not None:
                            credit = credit + cand_probs[ci]
                    loss = loss + (-torch.log(credit.clamp(min=1e-8)))
                else:
                    # No contributing rules -- push all candidates toward
                    # the head relation as a gradient signal
                    loss = loss + (-cand_probs.sum() / max(1, C))

    if negatives:
        neg_by_rel: dict[str, list[list[int]]] = {}
        for rel_name, values in negatives:
            neg_by_rel.setdefault(rel_name, []).append(values)

        for rel_name, facts_list in neg_by_rel.items():
            credits = prog.batch_tagged_credit(rel_name, facts_list)
            for fact_idx, contributing in enumerate(credits):
                if contributing:
                    credit = torch.tensor(0.0, device=device)
                    for (i, j, k) in contributing:
                        ci = ijk_to_cidx.get((i, j, k))
                        if ci is not None:
                            credit = credit + cand_probs[ci]
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
    *,
    perform_mask_checks: bool = True,
) -> bool:
    """Convergence check using training-mask checks plus argmax-only verification."""
    pos_by_rel: dict[str, list[list[int]]] = {}
    for rel, vals in positives:
        pos_by_rel.setdefault(rel, []).append(vals)

    if perform_mask_checks:
        # Gate 1: all positives derived under current mask
        for rel_name, facts_list in pos_by_rel.items():
            mask = prog.batch_fact_membership(rel_name, facts_list)
            if not all(mask):
                return False

        # Gate 2: no negatives derived under current mask
        # Some sparse candidate sets are intentionally smaller than
        # max_active_rules and can remain overactive; argmax-only
        # verification is still sufficient for promotion semantics.
        neg_by_rel: dict[str, list[list[int]]] = {}
        if negatives:
            for rel, vals in negatives:
                neg_by_rel.setdefault(rel, []).append(vals)
            for rel_name, facts_list in neg_by_rel.items():
                mask = prog.batch_fact_membership(rel_name, facts_list)
                if any(mask):
                    return False

    # Gate 3: argmax-only re-evaluation (always uses dense mask for verification)
    i, j, k = argmax_ijk
    M_check = torch.zeros((n, n, n), device=W.device)
    M_check[i, j, k] = 1.0
    flat_check = M_check.contiguous().view(-1)
    prog.set_rule_mask(mask_name, flat_check, flat_check, n)
    prog.evaluate()
    prog.reset_d2h_transfer_count()

    # Gate 3a: argmax-only must derive all positives
    for rel_name, facts_list in pos_by_rel.items():
        mask = prog.batch_fact_membership(rel_name, facts_list)
        if not all(mask):
            return False

    # Gate 4: argmax-only must not derive negatives
    if not negatives:
        return True

    neg_by_rel = {}
    for rel, vals in negatives:
        neg_by_rel.setdefault(rel, []).append(vals)

    if negatives:
        for rel_name, facts_list in neg_by_rel.items():
            mask = prog.batch_fact_membership(rel_name, facts_list)
            if any(mask):
                return False

    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_rule(left: str, right: str, head: str) -> str:
    """Format a discovered (left, right, head) triple as a Datalog rule string."""
    return f"{head}(X, Y) :- {left}(X, Z), {right}(Z, Y)."


def _fill_metrics(
    result: _AttemptResult,
    cand_probs: torch.Tensor,
    candidates: list[dict],
    W: torch.Tensor,
    backend,
) -> None:
    """Fill confidence metrics on an attempt result."""
    with torch.no_grad():
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
        result.soft_probs = cand_probs.detach().cpu().tolist()
        # Logits: for sparse backend W is C-shaped, for dense it's N^3
        if isinstance(backend, SparseMaskBackend):
            result.logits = W.detach().cpu().tolist()
        else:
            result.logits = [
                W[c["i"], c["j"], c["k"]].item() for c in candidates
            ]


def _select_winner(
    attempts: list[_AttemptResult],
    source: str,
    mask_name: str,
    positives: list[tuple[str, list[int]]],
    negatives: list[tuple[str, list[int]]],
    config: TrainConfig,
    global_steps: int,
    *,
    _compute_holdout: bool = True,
) -> TrainResult:
    """Select best attempt via deterministic tie-break chain."""
    if not attempts:
        return TrainResult(total_steps=global_steps, attempt_count=0)

    converged = [a for a in attempts if a.converged]

    if converged:
        # Tie-break: recall -> precision -> fewer steps -> lower candidate id
        winner = max(
            converged,
            key=lambda a: (a.recall, a.precision, -a.steps_used),
        )
    else:
        # No convergence -- pick attempt with most witness coverage (best recall proxy)
        winner = max(attempts, key=lambda a: (a.recall, -a.steps_used))

    holdout_f1 = None
    holdout_variance = 0.0
    if _compute_holdout and winner.converged:
        from pyxlog.ilp.holdout import holdout_f1_and_variance

        holdout_f1, holdout_variance = holdout_f1_and_variance(
            source,
            mask_name,
            positives,
            negatives,
            config,
        )

    # Compute rule_frequency: fraction of converged attempts finding same rule
    same_rule_count = sum(
        1 for a in attempts
        if a.converged and a.discovered_rule == winner.discovered_rule
    )
    rule_freq = same_rule_count / len(attempts) if attempts else 0.0

    return TrainResult(
        converged=winner.converged,
        discovered_rule=winner.discovered_rule,
        attempt_count=len(attempts),
        total_steps=global_steps,
        precision=winner.precision,
        recall=winner.recall,
        holdout_f1=holdout_f1,
        holdout_variance=holdout_variance,
        confidence_margin=winner.confidence_margin,
        top_k_concentration=winner.top_k_concentration,
        rule_frequency=rule_freq,
        single_attempt=len(attempts) == 1,
        artifact=LearnedArtifact(
            candidate_map=winner.candidate_map,
            logits=winner.logits,
            soft_probs=winner.soft_probs,
            selected_hard=winner.selected_hard,
            discovered_rule=winner.discovered_rule or "",
            config_snapshot=config,
            telemetry=TrainTelemetry(
                steps=winner.telemetry_steps,
                step_timings=winner.telemetry_timings,
            ),
        ),
    )
