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
    StrictLearnedArtifact,
    StrictTrainResult,
    TrainConfig,
    TrainResult,
    TrainTelemetry,
)


_TrainOnlyResult = TrainResult | StrictTrainResult
_RelationBatch = dict[str, list[object]]


def train_only(
    source: str,
    mask_name: str,
    positives: list[tuple[str, list[int]]],
    negatives: list[tuple[str, list[int]]],
    config: TrainConfig = TrainConfig(),
    *,
    _compute_holdout: bool = True,
) -> _TrainOnlyResult:
    """Train a single learnable rule via differentiable ILP.

    Returns a compatibility TrainResult when strict_gpu_native is disabled,
    or a StrictTrainResult with explicit export-only host materialization
    when strict_gpu_native is enabled.
    """
    _validate_inputs(source, mask_name, positives, negatives, config)

    if config.deterministic:
        _set_deterministic_cuda(config.seed)

    # Compile once for all attempts
    _compile_t0 = time.perf_counter()
    prog = pyxlog.IlpProgramFactory.compile(
        source, device=config.device, memory_mb=config.memory_mb,
        max_active_rules=config.max_active_rules,
    )
    prog.set_strict_zero_dtoh(config.strict_gpu_native)
    compile_ms = (time.perf_counter() - _compile_t0) * 1000.0

    return _train_on_compiled(
        prog, source, mask_name, positives, negatives, config,
        compile_ms=compile_ms,
        _compute_holdout=_compute_holdout,
    )


def train_on_compiled_relations(
    prog: object,
    mask_name: str,
    positives: _RelationBatch,
    negatives: _RelationBatch,
    config: TrainConfig = TrainConfig(),
) -> StrictTrainResult:
    """Train on a pre-compiled ILP program using relation-native example batches.

    `positives` and `negatives` must be dicts mapping relation name to a
    sequence of DLPack-compatible columns already resident on device.
    This API is strict-only and does not fall back to host-shaped training.
    """
    _validate_relation_inputs(mask_name, positives, negatives, config)

    if config.deterministic:
        _set_deterministic_cuda(config.seed)

    prog.set_strict_zero_dtoh(True)
    return _train_on_compiled_relations_strict(
        prog,
        mask_name,
        positives,
        negatives,
        config,
    )


def _train_on_compiled(
    prog: object,
    source: str,
    mask_name: str,
    positives: list[tuple[str, list[int]]],
    negatives: list[tuple[str, list[int]]],
    config: TrainConfig,
    *,
    compile_ms: float = 0.0,
    _compute_holdout: bool = True,
    _reset_before_first_attempt: bool = False,
) -> _TrainOnlyResult:
    """Run training on an already-compiled CompiledIlpProgram.

    This is the inner training loop extracted from train_only() to support
    compile-once-per-stage reuse in test harnesses. When
    _reset_before_first_attempt is True, reset_runtime() is called before
    the first attempt (needed when reusing a program across seeds).
    """
    prog.set_strict_zero_dtoh(config.strict_gpu_native)
    if config.strict_gpu_native:
        return _train_on_compiled_strict(
            prog,
            source,
            mask_name,
            positives,
            negatives,
            config,
            compile_ms=compile_ms,
            _reset_before_first_attempt=_reset_before_first_attempt,
        )

    attempts: list[_AttemptResult] = []
    global_steps = 0
    numeric_failures = 0
    reset_ms_list: list[float] = []

    for attempt_idx in range(config.max_attempts):
        if global_steps >= config.global_step_limit:
            break

        if config.deterministic:
            _seed_for_attempt(config.seed, attempt_idx)

        remaining = config.global_step_limit - global_steps
        step_budget = min(config.step_budget_per_attempt, remaining)
        if step_budget <= 0:
            break

        # Reset runtime state for attempts after the first
        # (or before the first when reusing a compiled program across seeds)
        if attempt_idx > 0 or _reset_before_first_attempt:
            _reset_t0 = time.perf_counter()
            prog.reset_runtime()
            reset_ms_list.append((time.perf_counter() - _reset_t0) * 1000.0)

        try:
            result = _run_single_attempt(
                prog,
                mask_name,
                positives,
                negatives,
                config,
                attempt_idx,
                step_budget,
            )
            # Attach compile/reset instrumentation
            result.telemetry_timings["compile_ms_once"] = compile_ms
            if reset_ms_list:
                result.telemetry_timings["reset_ms_total"] = sum(reset_ms_list)
                result.telemetry_timings["reset_ms_p95"] = _percentile_95(reset_ms_list)
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
        prog,
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

    _validate_strict_config(config)


def _validate_relation_inputs(
    mask_name: str,
    positives: _RelationBatch,
    negatives: _RelationBatch,
    config: TrainConfig,
) -> None:
    if not mask_name:
        raise IlpConfigError("mask_name must be non-empty")
    if not isinstance(positives, dict):
        raise IlpConfigError("positives must be a dict[str, sequence[dlpack]]")
    if not isinstance(negatives, dict):
        raise IlpConfigError("negatives must be a dict[str, sequence[dlpack]]")
    if not positives:
        raise IlpConfigError("positives must be non-empty")
    if not config.strict_gpu_native:
        raise IlpConfigError(
            "train_on_compiled_relations requires strict_gpu_native=True"
        )
    _validate_strict_config(config)


def _validate_strict_config(config: TrainConfig) -> None:
    if config.strict_gpu_native and config.max_mined_negatives > 0:
        raise IlpConfigError(
            "host negative mining is incompatible with strict_gpu_native; "
            "set max_mined_negatives=0 or disable strict_gpu_native"
        )
    if config.strict_gpu_native and config.telemetry_level > 0:
        raise IlpConfigError(
            "telemetry collection is incompatible with strict_gpu_native; "
            "set telemetry_level=0 or disable strict_gpu_native"
        )
    if config.strict_gpu_native and config.debug_dense_mask:
        raise IlpConfigError(
            "dense mask backend is incompatible with strict_gpu_native; "
            "set debug_dense_mask=False or disable strict_gpu_native"
        )
    if config.strict_gpu_native and config.allow_recursive_candidates:
        raise IlpConfigError(
            "recursive candidates are incompatible with strict_gpu_native; "
            "set allow_recursive_candidates=False or disable strict_gpu_native"
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
        "selected_hard", "telemetry_timings", "_compat_export_state",
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
        self._compat_export_state: dict | None = None


class _StrictAttemptState:
    __slots__ = (
        "candidate_map",
        "telemetry_steps",
        "telemetry_timings",
        "steps_used",
        "W_device",
        "cand_probs_device",
        "score_loss_device",
        "score_top1_device",
        "selected_candidate_ids_device",
        "argmax_candidate_id_device",
        "attempt_id_device",
        "rel_names",
        "candidates",
        "n",
        "allow_recursive",
    )

    def __init__(
        self,
        *,
        candidate_map: list[CandidateMapEntry],
        telemetry_steps: list[StepRecord],
        telemetry_timings: dict[str, float],
        steps_used: int,
        W_device: torch.Tensor,
        cand_probs_device: torch.Tensor,
        score_loss_device: torch.Tensor,
        score_top1_device: torch.Tensor,
        selected_candidate_ids_device: torch.Tensor,
        argmax_candidate_id_device: torch.Tensor,
        attempt_id_device: torch.Tensor,
        rel_names: list[str],
        candidates: list[dict],
        n: int,
        allow_recursive: bool,
    ) -> None:
        self.candidate_map = candidate_map
        self.telemetry_steps = telemetry_steps
        self.telemetry_timings = telemetry_timings
        self.steps_used = steps_used
        self.W_device = W_device
        self.cand_probs_device = cand_probs_device
        self.score_loss_device = score_loss_device
        self.score_top1_device = score_top1_device
        self.selected_candidate_ids_device = selected_candidate_ids_device
        self.argmax_candidate_id_device = argmax_candidate_id_device
        self.attempt_id_device = attempt_id_device
        self.rel_names = rel_names
        self.candidates = candidates
        self.n = n
        self.allow_recursive = allow_recursive


def _run_single_attempt(
    prog: object,
    mask_name: str,
    positives: list[tuple[str, list[int]]],
    negatives: list[tuple[str, list[int]]],
    config: TrainConfig,
    attempt_idx: int,
    step_budget: int,
) -> _AttemptResult | _StrictAttemptState:
    if config.strict_gpu_native:
        return _run_single_attempt_strict(
            prog,
            mask_name,
            positives,
            negatives,
            config,
            attempt_idx,
            step_budget,
        )
    return _run_single_attempt_compat(
        prog,
        mask_name,
        positives,
        negatives,
        config,
        attempt_idx,
        step_budget,
    )


def _train_on_compiled_strict(
    prog: object,
    source: str,
    mask_name: str,
    positives: list[tuple[str, list[int]]],
    negatives: list[tuple[str, list[int]]],
    config: TrainConfig,
    *,
    compile_ms: float = 0.0,
    _reset_before_first_attempt: bool = False,
) -> StrictTrainResult:
    best_state: _StrictAttemptState | None = None
    attempt_selected_candidate_ids: list[torch.Tensor] = []
    attempt_argmax_candidate_ids: list[torch.Tensor] = []
    attempt_host_metadata: dict[int, dict[str, object]] = {}
    global_steps = 0
    numeric_failures = 0
    reset_ms_list: list[float] = []

    for attempt_idx in range(config.max_attempts):
        if global_steps >= config.global_step_limit:
            break

        if config.deterministic:
            _seed_for_attempt(config.seed, attempt_idx)

        remaining = config.global_step_limit - global_steps
        step_budget = min(config.step_budget_per_attempt, remaining)
        if step_budget <= 0:
            break

        if attempt_idx > 0 or _reset_before_first_attempt:
            _reset_t0 = time.perf_counter()
            prog.reset_runtime()
            reset_ms_list.append((time.perf_counter() - _reset_t0) * 1000.0)

        try:
            state = _run_single_attempt_strict(
                prog,
                mask_name,
                positives,
                negatives,
                config,
                attempt_idx,
                step_budget,
            )
            state.telemetry_timings["compile_ms_once"] = compile_ms
            if reset_ms_list:
                state.telemetry_timings["reset_ms_total"] = sum(reset_ms_list)
                state.telemetry_timings["reset_ms_p95"] = _percentile_95(reset_ms_list)
            attempt_selected_candidate_ids.append(state.selected_candidate_ids_device.detach().clone())
            attempt_argmax_candidate_ids.append(state.argmax_candidate_id_device.detach().clone())
            attempt_host_metadata[attempt_idx] = {
                "steps_used": state.steps_used,
                "telemetry_steps": state.telemetry_steps,
                "telemetry_timings": state.telemetry_timings,
            }
            best_state = state if best_state is None else _select_better_strict_attempt(best_state, state)
            global_steps += state.steps_used
        except _NumericFailure:
            numeric_failures += 1
            global_steps += 1
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
            raise IlpTrainingError(f"{e.terminal_reason}: {e.cause}", e.context)

    return _build_strict_train_result(
        best_state=best_state,
        prog=prog,
        source=source,
        mask_name=mask_name,
        positives=positives,
        negatives=negatives,
        config=config,
        global_steps=global_steps,
        attempt_count=len(attempt_selected_candidate_ids),
        attempt_selected_candidate_ids=attempt_selected_candidate_ids,
        attempt_argmax_candidate_ids=attempt_argmax_candidate_ids,
        attempt_host_metadata=attempt_host_metadata,
    )


def _train_on_compiled_relations_strict(
    prog: object,
    mask_name: str,
    positives: _RelationBatch,
    negatives: _RelationBatch,
    config: TrainConfig,
    *,
    compile_ms: float = 0.0,
    _reset_before_first_attempt: bool = False,
) -> StrictTrainResult:
    best_state: _StrictAttemptState | None = None
    global_steps = 0
    numeric_failures = 0
    attempt_count = 0

    for attempt_idx in range(config.max_attempts):
        if global_steps >= config.global_step_limit:
            break

        if config.deterministic:
            _seed_for_attempt(config.seed, attempt_idx)

        remaining = config.global_step_limit - global_steps
        step_budget = min(config.step_budget_per_attempt, remaining)
        if step_budget <= 0:
            break

        if attempt_idx > 0 or _reset_before_first_attempt:
            prog.reset_runtime()

        try:
            state = _run_single_attempt_strict_relations(
                prog,
                mask_name,
                positives,
                negatives,
                config,
                attempt_idx,
                step_budget,
            )
            attempt_count += 1
            if compile_ms:
                state.telemetry_timings["compile_ms_once"] = compile_ms
            best_state = state if best_state is None else _select_better_strict_attempt(best_state, state)
            global_steps += state.steps_used
        except _NumericFailure:
            numeric_failures += 1
            global_steps += 1
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
            raise IlpTrainingError(f"{e.terminal_reason}: {e.cause}", e.context)

    return _build_relation_native_strict_train_result(
        best_state=best_state,
        config=config,
        global_steps=global_steps,
        attempt_count=attempt_count,
    )


def _run_single_attempt_strict_relations(
    prog: object,
    mask_name: str,
    positives: _RelationBatch,
    negatives: _RelationBatch,
    config: TrainConfig,
    attempt_idx: int,
    step_budget: int,
) -> _StrictAttemptState:
    try:
        n = prog.ilp_schema_size()
        rel_names = prog.ilp_relation_names()
        allow_recursive = config.allow_recursive_candidates
        candidates = prog.valid_candidates(mask_name, allow_recursive)
        if not candidates:
            raise IlpCandidateError("No valid candidates for mask")

        candidate_map = [
            CandidateMapEntry(
                id=c["id"], i=c["i"], j=c["j"], k=c["k"],
                left_name=c["left_name"], right_name=c["right_name"],
                head_name=c["head_name"],
            )
            for c in candidates
        ]

        prog.set_candidate_map([(c["i"], c["j"], c["k"]) for c in candidates])

        device = f"cuda:{config.device}" if torch.cuda.is_available() else "cpu"
        W = SparseMaskBackend().init_weights(len(candidates), n, device)
        optimizer = torch.optim.Adam([W], lr=0.1, capturable=True)
        result = _AttemptResult()
        result.candidate_map = candidate_map
        last_cand_probs: torch.Tensor | None = None
        last_logits: torch.Tensor | None = None
        last_loss: torch.Tensor | None = None

        for step in range(step_budget):
            try:
                optimizer.zero_grad()
                prog.reset_d2h_transfer_count()

                tau = _strict_tau_for_step(step, step_budget, config)
                cand_probs = torch.nn.functional.gumbel_softmax(W, tau=tau, hard=False, dim=0)
                effective_budget = min(config.max_active_rules, cand_probs.numel())
                selected = torch.argsort(
                    cand_probs.detach(),
                    descending=True,
                    stable=True,
                )[:effective_budget]
                selected_soft = cand_probs.detach().index_select(0, selected).contiguous()
                prog.set_rule_mask_sparse_selected_device(
                    mask_name,
                    selected.to(dtype=torch.int64).contiguous(),
                    selected_soft,
                    allow_recursive,
                )
                loss_dl, grad_dl = prog.compute_ilp_loss_grad_gpu_relations(
                    positives,
                    negatives,
                    cand_probs.detach(),
                )
                credit_loss = torch.from_dlpack(loss_dl).reshape(())
                credit_grad = torch.from_dlpack(grad_dl)
                last_loss = credit_loss.detach().clone()
                last_cand_probs = cand_probs.detach().clone()
                if step == step_budget - 1:
                    last_logits = W.detach().clone()

                ent_weight = entropy_weight_at_step(
                    step,
                    step_budget,
                    start=config.entropy_weight_start,
                    end=config.entropy_weight_end,
                )
                if ent_weight > 0 and len(candidates) > 1:
                    cand_probs_norm = cand_probs / cand_probs.sum().clamp_min(1e-8)
                    ent = normalized_entropy(cand_probs_norm, len(candidates))
                    ent_grads = torch.autograd.grad(
                        -ent_weight * ent,
                        cand_probs,
                        retain_graph=True,
                    )[0]
                    credit_grad = credit_grad + ent_grads

                cand_probs.backward(credit_grad)
                optimizer.step()

                d2h_count = prog.d2h_transfer_count()
                if d2h_count > 0:
                    context = _attempt_context(
                        attempt=attempt_idx,
                        step=step,
                        C=len(candidates),
                        config=config,
                        terminal_reason="d2h_gate_violation",
                    )
                    context["d2h_count"] = d2h_count
                    raise _AttemptRuntimeFailure(
                        terminal_reason="d2h_gate_violation",
                        context=context,
                        cause=RuntimeError(
                            "d2h transfer observed in strict relation-native hot step loop"
                        ),
                    )
            except _AttemptRuntimeFailure:
                raise
            except Exception as e:
                context = _attempt_context(
                    attempt=attempt_idx,
                    step=step,
                    C=len(candidates),
                    config=config,
                    terminal_reason=_classify_failure_reason(e),
                )
                context["error"] = str(e)
                raise _AttemptRuntimeFailure(
                    terminal_reason=_classify_failure_reason(e),
                    context=context,
                    cause=e,
                )

        if last_cand_probs is None or last_logits is None or last_loss is None:
            raise IlpTrainingError(
                "strict_relation_native_hot_loop executed zero steps",
                {"attempt": attempt_idx},
            )

        return _finalize_strict_attempt(
            config,
            attempt_idx,
            step_budget,
            rel_names,
            candidates,
            n,
            last_logits,
            last_cand_probs,
            last_loss,
            result,
        )
    except IlpCandidateError:
        raise
    except _AttemptRuntimeFailure:
        raise
    except Exception as e:
        context = {
            "attempt": attempt_idx,
            "C": len(candidates) if "candidates" in locals() else 0,
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


def _select_better_strict_attempt(
    best_state: _StrictAttemptState,
    candidate_state: _StrictAttemptState,
) -> _StrictAttemptState:
    choose_candidate = (
        (candidate_state.score_loss_device < best_state.score_loss_device)
        | (
            (candidate_state.score_loss_device == best_state.score_loss_device)
            & (candidate_state.score_top1_device > best_state.score_top1_device)
        )
    ).to(dtype=torch.bool)

    best_state.W_device = torch.where(choose_candidate, candidate_state.W_device, best_state.W_device)
    best_state.cand_probs_device = torch.where(
        choose_candidate,
        candidate_state.cand_probs_device,
        best_state.cand_probs_device,
    )
    best_state.score_loss_device = torch.where(
        choose_candidate,
        candidate_state.score_loss_device,
        best_state.score_loss_device,
    )
    best_state.score_top1_device = torch.where(
        choose_candidate,
        candidate_state.score_top1_device,
        best_state.score_top1_device,
    )
    best_state.selected_candidate_ids_device = torch.where(
        choose_candidate,
        candidate_state.selected_candidate_ids_device,
        best_state.selected_candidate_ids_device,
    )
    best_state.argmax_candidate_id_device = torch.where(
        choose_candidate,
        candidate_state.argmax_candidate_id_device,
        best_state.argmax_candidate_id_device,
    )
    best_state.attempt_id_device = torch.where(
        choose_candidate,
        candidate_state.attempt_id_device,
        best_state.attempt_id_device,
    )
    return best_state


def _run_single_attempt_compat(
    prog: object,
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

        # Upload candidate map to Rust for GPU-resident loss computation
        prog.set_candidate_map([(c["i"], c["j"], c["k"]) for c in candidates])

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
        best_witness_coverage = torch.tensor(0.0, device=W.device)

        # Phase timing accumulators (microseconds per step)
        phase_apply_mask_us: list[float] = []
        phase_loss_credit_us: list[float] = []
        phase_loss_reduce_us: list[float] = []
        phase_backward_step_us: list[float] = []
        phase_membership_us: list[float] = []
        phase_convergence_us: list[float] = []
        pos_by_rel, pos_indices_by_rel = _prepare_relation_groups(positives, W.device)

        for step in range(step_budget):
            try:
                optimizer.zero_grad()
                prog.reset_d2h_transfer_count()

                _t0 = time.perf_counter()
                cand_probs, selected_hard = backend.apply_mask(
                    prog, mask_name, W, temp_controller.tau,
                    config.max_active_rules, candidates, n,
                    allow_recursive=allow_recursive,
                )
                result.selected_hard = [int(i) for i in selected_hard if int(i) >= 0]
                phase_apply_mask_us.append((time.perf_counter() - _t0) * 1_000_000)

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

                # Compute task loss using GPU-resident credit path
                _t0 = time.perf_counter()
                loss_dl, grad_dl = prog.compute_ilp_loss_grad_gpu(
                    positives, all_negatives, cand_probs.detach(),
                )
                credit_loss = torch.from_dlpack(loss_dl).reshape(())
                credit_grad = torch.from_dlpack(grad_dl)
                phase_loss_credit_us.append((time.perf_counter() - _t0) * 1_000_000)

                # Entropy regularization
                _t0 = time.perf_counter()
                ent_weight = entropy_weight_at_step(
                    step, step_budget,
                    start=config.entropy_weight_start,
                    end=config.entropy_weight_end,
                )
                loss_value = credit_loss
                if ent_weight > 0 and C > 1:
                    cand_sum = cand_probs.sum()
                    if cand_sum > 1e-8:
                        cand_probs_norm = cand_probs / cand_sum
                        ent = normalized_entropy(cand_probs_norm, C)
                        loss_value = loss_value - ent_weight * ent
                        ent_grads = torch.autograd.grad(
                            -ent_weight * ent, cand_probs, retain_graph=True,
                        )[0]
                        credit_grad = credit_grad + ent_grads

                # Check for NaN/Inf
                loss_value_scalar = float(loss_value.detach().cpu())
                if math.isnan(loss_value_scalar) or math.isinf(loss_value_scalar):
                    raise _NumericFailure()
                phase_loss_reduce_us.append((time.perf_counter() - _t0) * 1_000_000)

                _t0 = time.perf_counter()
                cand_probs.backward(credit_grad)
                optimizer.step()
                phase_backward_step_us.append((time.perf_counter() - _t0) * 1_000_000)

                # Decode argmax via backend
                argmax_idx = backend.decode_argmax_compat(W, candidates, n)
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
                _t0 = time.perf_counter()
                witness_mask = _compute_grouped_membership_mask_device(
                    prog,
                    pos_by_rel,
                    pos_indices_by_rel,
                    len(positives),
                    W.device,
                )
                witness_count = int(torch.count_nonzero(witness_mask).detach().cpu())
                witness_coverage = witness_count / len(positives)
                best_witness_coverage = torch.maximum(
                    best_witness_coverage,
                    torch.tensor(witness_coverage, device=W.device),
                )
                phase_membership_us.append((time.perf_counter() - _t0) * 1_000_000)

                # Discreteness metric -- max candidate prob
                with torch.no_grad():
                    disc = (
                        float(cand_probs.max().detach().cpu())
                        if cand_probs.numel() > 0
                        else 0.0
                    )
                temp_controller.step(
                    loss=loss_value_scalar, disc=disc, witness_coverage=witness_coverage,
                )

                # Telemetry
                if config.telemetry_level >= 1:
                    i, j, k = argmax_ijk
                    argmax_rule = f"{rel_names[i]}+{rel_names[j]}->{rel_names[k]}"
                    tel_sum = cand_probs.sum()
                    if tel_sum > 1e-8 and C > 1:
                        tel_normed = cand_probs / tel_sum
                        tel_ent = float(normalized_entropy(tel_normed, C).detach().cpu())
                    else:
                        tel_ent = 0.0
                    result.telemetry_steps.append(StepRecord(
                        step=step,
                        loss=loss_value_scalar,
                        argmax_rule=argmax_rule,
                        discreteness=disc,
                        temperature=temp_controller.tau,
                        entropy=tel_ent,
                        stable_count=stable_count,
                        forward_p95_us=_percentile_95(forward_hist_us),
                        active_candidates=C,
                    ))

                # Convergence check
                _t0 = time.perf_counter()
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
                            "apply_mask_p95_us": _percentile_95(phase_apply_mask_us),
                            "loss_credit_p95_us": _percentile_95(phase_loss_credit_us),
                            "loss_reduce_p95_us": _percentile_95(phase_loss_reduce_us),
                            "backward_step_p95_us": _percentile_95(phase_backward_step_us),
                            "membership_p95_us": _percentile_95(phase_membership_us),
                            "convergence_p95_us": _percentile_95(phase_convergence_us),
                            "apply_mask_total_ms": sum(phase_apply_mask_us) / 1000,
                            "loss_credit_total_ms": sum(phase_loss_credit_us) / 1000,
                            "loss_reduce_total_ms": sum(phase_loss_reduce_us) / 1000,
                            "backward_step_total_ms": sum(phase_backward_step_us) / 1000,
                            "membership_total_ms": sum(phase_membership_us) / 1000,
                            "convergence_total_ms": sum(phase_convergence_us) / 1000,
                        }
                        result.steps_used = step + 1
                        result.argmax_ijk = argmax_ijk
                        _materialize_compat_metrics(result, cand_probs, candidates, W, backend)
                        return result
                phase_convergence_us.append((time.perf_counter() - _t0) * 1_000_000)

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
        result.recall = float(best_witness_coverage.detach().cpu())
        if prev_argmax is not None:
            i, j, k = prev_argmax
            result.discovered_rule = _format_rule(
                rel_names[i], rel_names[j], rel_names[k],
            )
            result.argmax_ijk = prev_argmax
        _materialize_compat_metrics(result, cand_probs, candidates, W, backend)
        if forward_hist_us:
            result.telemetry_timings = {
                "forward_p95_us": _percentile_95(forward_hist_us),
                "allocated_bytes_p95": _percentile_95(allocated_hist_bytes),
                "allocated_bytes_max": max(allocated_hist_bytes),
                "apply_mask_p95_us": _percentile_95(phase_apply_mask_us),
                "loss_credit_p95_us": _percentile_95(phase_loss_credit_us),
                "loss_reduce_p95_us": _percentile_95(phase_loss_reduce_us),
                "backward_step_p95_us": _percentile_95(phase_backward_step_us),
                "membership_p95_us": _percentile_95(phase_membership_us),
                "convergence_p95_us": _percentile_95(phase_convergence_us),
                "apply_mask_total_ms": sum(phase_apply_mask_us) / 1000,
                "loss_credit_total_ms": sum(phase_loss_credit_us) / 1000,
                "loss_reduce_total_ms": sum(phase_loss_reduce_us) / 1000,
                "backward_step_total_ms": sum(phase_backward_step_us) / 1000,
                "membership_total_ms": sum(phase_membership_us) / 1000,
                "convergence_total_ms": sum(phase_convergence_us) / 1000,
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


def _run_single_attempt_strict(
    prog: object,
    mask_name: str,
    positives: list[tuple[str, list[int]]],
    negatives: list[tuple[str, list[int]]],
    config: TrainConfig,
    attempt_idx: int,
    step_budget: int,
) -> _StrictAttemptState:
    try:
        n = prog.ilp_schema_size()
        rel_names = prog.ilp_relation_names()
        allow_recursive = config.allow_recursive_candidates
        candidates = prog.valid_candidates(mask_name, allow_recursive)
        if not candidates:
            raise IlpCandidateError("No valid candidates for mask")

        candidate_map = [
            CandidateMapEntry(
                id=c["id"], i=c["i"], j=c["j"], k=c["k"],
                left_name=c["left_name"], right_name=c["right_name"],
                head_name=c["head_name"],
            )
            for c in candidates
        ]

        prog.set_candidate_map([(c["i"], c["j"], c["k"]) for c in candidates])

        backend = SparseMaskBackend()
        device = f"cuda:{config.device}" if torch.cuda.is_available() else "cpu"
        W = backend.init_weights(len(candidates), n, device)
        optimizer = torch.optim.Adam([W], lr=0.1, capturable=True)
        result = _AttemptResult()
        result.candidate_map = candidate_map
        last_cand_probs: torch.Tensor | None = None
        last_logits: torch.Tensor | None = None
        last_loss: torch.Tensor | None = None

        for step in range(step_budget):
            try:
                optimizer.zero_grad()
                prog.reset_d2h_transfer_count()

                tau = _strict_tau_for_step(step, step_budget, config)
                cand_probs = torch.nn.functional.gumbel_softmax(W, tau=tau, hard=False, dim=0)
                effective_budget = min(config.max_active_rules, cand_probs.numel())
                selected = torch.argsort(
                    cand_probs.detach(),
                    descending=True,
                    stable=True,
                )[:effective_budget]
                selected_soft = cand_probs.detach().index_select(0, selected).contiguous()
                prog.set_rule_mask_sparse_selected_device(
                    mask_name,
                    selected.to(dtype=torch.int64).contiguous(),
                    selected_soft,
                    allow_recursive,
                )
                last_cand_probs = cand_probs.detach().clone()
                if step == step_budget - 1:
                    last_logits = W.detach().clone()

                loss_dl, grad_dl = prog.compute_ilp_loss_grad_gpu(
                    positives,
                    negatives,
                    cand_probs.detach(),
                )
                credit_loss = torch.from_dlpack(loss_dl).reshape(())
                credit_grad = torch.from_dlpack(grad_dl)
                last_loss = credit_loss.detach().clone()

                ent_weight = entropy_weight_at_step(
                    step,
                    step_budget,
                    start=config.entropy_weight_start,
                    end=config.entropy_weight_end,
                )
                if ent_weight > 0 and len(candidates) > 1:
                    cand_probs_norm = cand_probs / cand_probs.sum().clamp_min(1e-8)
                    ent = normalized_entropy(cand_probs_norm, len(candidates))
                    ent_grads = torch.autograd.grad(
                        -ent_weight * ent,
                        cand_probs,
                        retain_graph=True,
                    )[0]
                    credit_grad = credit_grad + ent_grads

                cand_probs.backward(credit_grad)
                optimizer.step()

                d2h_count = prog.d2h_transfer_count()
                if d2h_count > 0:
                    context = _attempt_context(
                        attempt=attempt_idx,
                        step=step,
                        C=len(candidates),
                        config=config,
                        terminal_reason="d2h_gate_violation",
                    )
                    context["d2h_count"] = d2h_count
                    raise _AttemptRuntimeFailure(
                        terminal_reason="d2h_gate_violation",
                        context=context,
                        cause=RuntimeError("d2h transfer observed in strict hot step loop"),
                    )
            except _AttemptRuntimeFailure:
                raise
            except Exception as e:
                context = _attempt_context(
                    attempt=attempt_idx,
                    step=step,
                    C=len(candidates),
                    config=config,
                    terminal_reason=_classify_failure_reason(e),
                )
                context["error"] = str(e)
                raise _AttemptRuntimeFailure(
                    terminal_reason=_classify_failure_reason(e),
                    context=context,
                    cause=e,
                )

        if last_cand_probs is None or last_logits is None or last_loss is None:
            raise IlpTrainingError("strict_hot_loop executed zero steps", {"attempt": attempt_idx})

        return _finalize_strict_attempt(
            config,
            attempt_idx,
            step_budget,
            rel_names,
            candidates,
            n,
            last_logits,
            last_cand_probs,
            last_loss,
            result,
        )
    except IlpCandidateError:
        raise
    except _AttemptRuntimeFailure:
        raise
    except Exception as e:
        context = {
            "attempt": attempt_idx,
            "C": len(candidates) if "candidates" in locals() else 0,
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


def _strict_tau_for_step(step: int, step_budget: int, config: TrainConfig) -> float:
    if step_budget <= 1:
        return config.tau_floor
    frac = step / max(1, step_budget - 1)
    tau = config.tau_start + (config.tau_floor - config.tau_start) * frac
    return max(config.tau_floor, min(config.tau_start, tau))


def _materialize_compat_selected_hard(cand_probs: torch.Tensor, budget: int) -> list[int]:
    effective_budget = min(budget, cand_probs.numel())
    selected = torch.argsort(
        cand_probs.detach(),
        descending=True,
        stable=True,
    )[:effective_budget]
    return [int(idx) for idx in selected.cpu().tolist()]


def _materialize_compat_witness_coverage(
    prog,
    positives: list[tuple[str, list[int]]],
    device: torch.device,
) -> float:
    if not positives:
        return 0.0
    pos_by_rel, pos_indices_by_rel = _prepare_relation_groups(positives, device)
    witness_mask = _compute_grouped_membership_mask_device(
        prog,
        pos_by_rel,
        pos_indices_by_rel,
        len(positives),
        device,
    )
    witness_count = int(torch.count_nonzero(witness_mask).detach().cpu())
    return witness_count / len(positives)

def _strict_selected_candidate_ids(cand_probs: torch.Tensor, budget: int) -> torch.Tensor:
    effective_budget = min(budget, cand_probs.numel())
    return torch.argsort(
        cand_probs.detach(),
        descending=True,
        stable=True,
    )[:effective_budget]


def _finalize_strict_attempt(
    config: TrainConfig,
    attempt_idx: int,
    step_budget: int,
    rel_names: list[str],
    candidates: list[dict],
    n: int,
    W: torch.Tensor,
    cand_probs: torch.Tensor,
    loss: torch.Tensor,
    result: _AttemptResult,
) -> _StrictAttemptState:
    selected = _strict_selected_candidate_ids(cand_probs, config.max_active_rules)
    argmax_candidate_id = W.argmax().reshape(1).to(dtype=torch.int64).contiguous()
    score_top1 = (
        cand_probs.max().reshape(()).detach().clone()
        if cand_probs.numel() > 0
        else torch.zeros((), device=cand_probs.device, dtype=cand_probs.dtype)
    )
    return _StrictAttemptState(
        candidate_map=result.candidate_map,
        telemetry_steps=result.telemetry_steps,
        telemetry_timings=result.telemetry_timings,
        steps_used=step_budget,
        W_device=W,
        cand_probs_device=cand_probs,
        score_loss_device=loss,
        score_top1_device=score_top1,
        selected_candidate_ids_device=selected.to(dtype=torch.int64).contiguous(),
        argmax_candidate_id_device=argmax_candidate_id,
        attempt_id_device=torch.tensor([attempt_idx], device=cand_probs.device, dtype=torch.int64),
        rel_names=rel_names,
        candidates=candidates,
        n=n,
        allow_recursive=config.allow_recursive_candidates,
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
    pos_by_rel, pos_indices_by_rel = _prepare_relation_groups(positives, W.device)

    if perform_mask_checks:
        # Gate 1: all positives derived under current mask
        pos_mask = _compute_grouped_membership_mask_device(
            prog,
            pos_by_rel,
            pos_indices_by_rel,
            len(positives),
            W.device,
        )
        if not bool(pos_mask.all().item()):
            return False

        # Gate 2: no negatives derived under current mask
        # Some sparse candidate sets are intentionally smaller than
        # max_active_rules and can remain overactive; argmax-only
        # verification is still sufficient for promotion semantics.
        if negatives:
            neg_by_rel, neg_indices_by_rel = _prepare_relation_groups(negatives, W.device)
            neg_mask = _compute_grouped_membership_mask_device(
                prog,
                neg_by_rel,
                neg_indices_by_rel,
                len(negatives),
                W.device,
            )
            if bool(neg_mask.any().item()):
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
    pos_mask = _compute_grouped_membership_mask_device(
        prog,
        pos_by_rel,
        pos_indices_by_rel,
        len(positives),
        W.device,
    )
    if not bool(pos_mask.all().item()):
        return False

    # Gate 4: argmax-only must not derive negatives
    if not negatives:
        return True

    neg_by_rel, neg_indices_by_rel = _prepare_relation_groups(negatives, W.device)
    neg_mask = _compute_grouped_membership_mask_device(
        prog,
        neg_by_rel,
        neg_indices_by_rel,
        len(negatives),
        W.device,
    )
    if bool(neg_mask.any().item()):
        return False

    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prepare_relation_groups(
    facts: list[tuple[str, list[int]]],
    device: torch.device,
) -> tuple[dict[str, list[list[int]]], dict[str, torch.Tensor]]:
    """Group example tuples by relation and prebuild CUDA index tensors."""
    facts_by_rel: dict[str, list[list[int]]] = {}
    indices_by_rel_host: dict[str, list[int]] = {}
    for idx, (rel_name, values) in enumerate(facts):
        facts_by_rel.setdefault(rel_name, []).append(values)
        indices_by_rel_host.setdefault(rel_name, []).append(idx)
    indices_by_rel = {
        rel_name: torch.tensor(indices, device=device, dtype=torch.long)
        for rel_name, indices in indices_by_rel_host.items()
    }
    return facts_by_rel, indices_by_rel


def _compute_grouped_membership_mask_device(
    prog,
    facts_by_rel: dict[str, list[list[int]]],
    indices_by_rel: dict[str, torch.Tensor],
    total_facts: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute a per-fact CUDA membership mask via device-native pyxlog APIs."""
    witness_mask = torch.zeros(total_facts, dtype=torch.bool, device=device)
    for rel_name, facts_list in facts_by_rel.items():
        mask = torch.from_dlpack(prog.batch_fact_membership_device(rel_name, facts_list))
        witness_mask.index_copy_(0, indices_by_rel[rel_name], mask.to(torch.bool))
    return witness_mask


def _format_rule(left: str, right: str, head: str) -> str:
    """Format a discovered (left, right, head) triple as a Datalog rule string."""
    return f"{head}(X, Y) :- {left}(X, Z), {right}(Z, Y)."


def _materialize_compat_metrics(
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
            result.confidence_margin = None

        # Top-k concentration
        k = min(32, len(cand_probs))
        topk_vals, _ = cand_probs.topk(k)
        total_mass = cand_probs.sum().item()
        if total_mass > 1e-8:
            result.top_k_concentration = topk_vals.sum().item() / total_mass
        else:
            result.top_k_concentration = None

        # Store raw values for artifact
        result.soft_probs = cand_probs.detach().cpu().tolist()
        # Logits: for sparse backend W is C-shaped, for dense it's N^3
        if isinstance(backend, SparseMaskBackend):
            result.logits = W.detach().cpu().tolist()
        else:
            result.logits = [
                W[c["i"], c["j"], c["k"]].item() for c in candidates
            ]


def _pick_winner(attempts: list[_AttemptResult]) -> _AttemptResult:
    converged = [a for a in attempts if a.converged]
    if converged:
        return max(
            converged,
            key=lambda a: (a.recall, a.precision, -a.steps_used),
        )
    return max(attempts, key=lambda a: (a.recall, -a.steps_used))


def _build_compat_train_result(
    attempts: list[_AttemptResult],
    winner: _AttemptResult,
    source: str,
    mask_name: str,
    positives: list[tuple[str, list[int]]],
    negatives: list[tuple[str, list[int]]],
    config: TrainConfig,
    global_steps: int,
    *,
    _compute_holdout: bool = True,
    rule_frequency_override: float | None = None,
    attempt_count_override: int | None = None,
) -> TrainResult:
    holdout_f1 = None
    holdout_variance = 0.0
    if _compute_holdout and winner.converged and not config.strict_gpu_native:
        from pyxlog.ilp.holdout import holdout_f1_and_variance

        holdout_f1, holdout_variance = holdout_f1_and_variance(
            source,
            mask_name,
            positives,
            negatives,
            config,
        )

    attempt_count = attempt_count_override if attempt_count_override is not None else len(attempts)
    if rule_frequency_override is None:
        same_rule_count = sum(
            1 for a in attempts
            if a.converged and a.discovered_rule == winner.discovered_rule
        )
        rule_freq = same_rule_count / len(attempts) if attempts else 0.0
    else:
        rule_freq = rule_frequency_override

    return TrainResult(
        converged=winner.converged,
        discovered_rule=winner.discovered_rule,
        attempt_count=attempt_count,
        total_steps=global_steps,
        precision=winner.precision,
        recall=winner.recall,
        holdout_f1=holdout_f1,
        holdout_variance=holdout_variance,
        confidence_margin=winner.confidence_margin,
        top_k_concentration=winner.top_k_concentration,
        rule_frequency=rule_freq,
        single_attempt=attempt_count == 1,
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
            strict_gpu_native=False,
            compat_materialized=True,
        ),
        strict_gpu_native=False,
        compat_materialized=True,
    )


def _export_compat_attempt(
    prog,
    attempt: _AttemptResult,
    mask_name: str,
    positives: list[tuple[str, list[int]]],
    negatives: list[tuple[str, list[int]]],
    config: TrainConfig,
) -> _AttemptResult:
    state = attempt._compat_export_state
    if state is None:
        return attempt

    prog.reset_runtime()

    backend = SparseMaskBackend()
    W = state["W"]
    cand_probs = state["cand_probs"]
    candidates = state["candidates"]
    rel_names = state["rel_names"]
    n = state["n"]
    allow_recursive = state["allow_recursive"]

    effective_budget = min(config.max_active_rules, cand_probs.numel())
    selected = torch.argsort(
        cand_probs.detach(),
        descending=True,
        stable=True,
    )[:effective_budget]
    selected_soft = cand_probs.detach().index_select(0, selected).contiguous()
    selected_ids = [int(idx) for idx in selected.detach().cpu().tolist()]

    prog.set_rule_mask_sparse_selected(
        mask_name,
        selected_ids,
        selected_soft,
        allow_recursive,
    )
    prog.evaluate()

    argmax_idx = backend.decode_argmax_compat(W, candidates, n)
    argmax_ijk = (
        candidates[argmax_idx]["i"],
        candidates[argmax_idx]["j"],
        candidates[argmax_idx]["k"],
    )

    compat_attempt = _AttemptResult()
    compat_attempt.steps_used = attempt.steps_used
    compat_attempt.candidate_map = attempt.candidate_map
    compat_attempt.telemetry_steps = attempt.telemetry_steps
    compat_attempt.telemetry_timings = attempt.telemetry_timings
    compat_attempt.argmax_ijk = argmax_ijk
    compat_attempt.selected_hard = _materialize_compat_selected_hard(
        cand_probs,
        config.max_active_rules,
    )

    witness_coverage = _materialize_compat_witness_coverage(
        prog,
        positives,
        cand_probs.device,
    )
    converged = _check_convergence(
        prog,
        W,
        mask_name,
        positives,
        negatives,
        rel_names,
        n,
        argmax_ijk,
        perform_mask_checks=False,
    )
    if converged:
        compat_attempt.converged = True
        compat_attempt.precision = 1.0
        compat_attempt.recall = 1.0
    else:
        compat_attempt.recall = witness_coverage

    i, j, k = argmax_ijk
    compat_attempt.discovered_rule = _format_rule(rel_names[i], rel_names[j], rel_names[k])
    _materialize_compat_metrics(compat_attempt, cand_probs, candidates, W, backend)
    return compat_attempt


def _materialize_selected_candidate_ids(selected_candidate_ids_device: torch.Tensor) -> list[int]:
    return [int(idx) for idx in selected_candidate_ids_device.detach().cpu().tolist()]


def _materialize_argmax_candidate_id(argmax_candidate_id_device: torch.Tensor) -> int:
    argmax_values = argmax_candidate_id_device.detach().cpu().reshape(-1).tolist()
    if not argmax_values:
        raise RuntimeError("strict argmax export state was empty")
    return int(argmax_values[0])


def _materialize_strict_attempt_index(attempt_id_device: torch.Tensor) -> int:
    attempt_values = attempt_id_device.detach().cpu().reshape(-1).tolist()
    if not attempt_values:
        raise RuntimeError("strict attempt id export state was empty")
    return int(attempt_values[0])


def _export_compat_attempt_from_strict_state(
    prog,
    state: _StrictAttemptState,
    host_metadata: dict[str, object],
    mask_name: str,
    positives: list[tuple[str, list[int]]],
    negatives: list[tuple[str, list[int]]],
    config: TrainConfig,
) -> _AttemptResult:
    prog.reset_runtime()

    backend = SparseMaskBackend()
    W = state.W_device
    cand_probs = state.cand_probs_device
    selected_device = state.selected_candidate_ids_device.to(dtype=torch.long).contiguous()
    selected_ids = _materialize_selected_candidate_ids(selected_device)
    if selected_device.numel() > 0:
        selected_soft = cand_probs.detach().index_select(0, selected_device).contiguous().double()
    else:
        selected_soft = torch.empty(0, device=cand_probs.device, dtype=torch.float64)

    prog.set_rule_mask_sparse_selected(
        mask_name,
        selected_ids,
        selected_soft,
        state.allow_recursive,
    )
    prog.evaluate()

    argmax_idx = _materialize_argmax_candidate_id(state.argmax_candidate_id_device)
    argmax_ijk = (
        state.candidates[argmax_idx]["i"],
        state.candidates[argmax_idx]["j"],
        state.candidates[argmax_idx]["k"],
    )

    compat_attempt = _AttemptResult()
    compat_attempt.steps_used = int(host_metadata["steps_used"])
    compat_attempt.candidate_map = state.candidate_map
    compat_attempt.telemetry_steps = list(host_metadata["telemetry_steps"])
    compat_attempt.telemetry_timings = dict(host_metadata["telemetry_timings"])
    compat_attempt.argmax_ijk = argmax_ijk
    compat_attempt.selected_hard = list(selected_ids)

    witness_coverage = _materialize_compat_witness_coverage(
        prog,
        positives,
        cand_probs.device,
    )
    converged = _check_convergence(
        prog,
        W,
        mask_name,
        positives,
        negatives,
        state.rel_names,
        state.n,
        argmax_ijk,
        perform_mask_checks=False,
    )
    if converged:
        compat_attempt.converged = True
        compat_attempt.precision = 1.0
        compat_attempt.recall = 1.0
    else:
        compat_attempt.recall = witness_coverage

    i, j, k = argmax_ijk
    compat_attempt.discovered_rule = _format_rule(
        state.rel_names[i],
        state.rel_names[j],
        state.rel_names[k],
    )
    _materialize_compat_metrics(compat_attempt, cand_probs, state.candidates, W, backend)
    return compat_attempt


def _strict_attempt_convergence_summary(
    prog,
    mask_name: str,
    positives: list[tuple[str, list[int]]],
    negatives: list[tuple[str, list[int]]],
    rel_names: list[str],
    candidates: list[dict],
    n: int,
    allow_recursive: bool,
    selected_candidate_ids_device: torch.Tensor,
    argmax_candidate_id_device: torch.Tensor,
) -> tuple[bool, int]:
    prog.reset_runtime()

    selected_ids = _materialize_selected_candidate_ids(selected_candidate_ids_device)
    selected_soft = torch.ones(
        len(selected_ids),
        device=selected_candidate_ids_device.device,
        dtype=torch.float64,
    )
    prog.set_rule_mask_sparse_selected(
        mask_name,
        selected_ids,
        selected_soft,
        allow_recursive,
    )
    prog.evaluate()

    argmax_idx = _materialize_argmax_candidate_id(argmax_candidate_id_device)
    argmax_ijk = (
        candidates[argmax_idx]["i"],
        candidates[argmax_idx]["j"],
        candidates[argmax_idx]["k"],
    )
    reference_tensor = (
        selected_candidate_ids_device
        if selected_candidate_ids_device.numel() > 0
        else argmax_candidate_id_device
    )
    converged = _check_convergence(
        prog,
        reference_tensor,
        mask_name,
        positives,
        negatives,
        rel_names,
        n,
        argmax_ijk,
        perform_mask_checks=False,
    )
    return converged, argmax_idx


def _compute_strict_rule_frequency(
    prog,
    mask_name: str,
    positives: list[tuple[str, list[int]]],
    negatives: list[tuple[str, list[int]]],
    rel_names: list[str],
    candidates: list[dict],
    n: int,
    allow_recursive: bool,
    winner_argmax_candidate_id_device: torch.Tensor,
    attempt_selected_candidate_ids: list[torch.Tensor],
    attempt_argmax_candidate_ids: list[torch.Tensor],
) -> float:
    if not attempt_selected_candidate_ids:
        return 0.0

    winner_argmax_idx = _materialize_argmax_candidate_id(winner_argmax_candidate_id_device)
    same_rule_count = 0
    for selected_candidate_ids_device, argmax_candidate_id_device in zip(
        attempt_selected_candidate_ids,
        attempt_argmax_candidate_ids,
        strict=True,
    ):
        converged, argmax_idx = _strict_attempt_convergence_summary(
            prog,
            mask_name,
            positives,
            negatives,
            rel_names,
            candidates,
            n,
            allow_recursive,
            selected_candidate_ids_device,
            argmax_candidate_id_device,
        )
        if converged and argmax_idx == winner_argmax_idx:
            same_rule_count += 1
    return same_rule_count / len(attempt_selected_candidate_ids)


def _build_strict_train_result(
    *,
    best_state: _StrictAttemptState | None,
    prog,
    source: str,
    mask_name: str,
    positives: list[tuple[str, list[int]]],
    negatives: list[tuple[str, list[int]]],
    config: TrainConfig,
    global_steps: int,
    attempt_count: int,
    attempt_selected_candidate_ids: list[torch.Tensor],
    attempt_argmax_candidate_ids: list[torch.Tensor],
    attempt_host_metadata: dict[int, dict[str, object]],
) -> StrictTrainResult:
    if best_state is None:
        return StrictTrainResult(
            attempt_count=attempt_count,
            total_steps=global_steps,
            single_attempt=attempt_count == 1,
        )

    compat_cache: dict[str, TrainResult] = {}

    def _export_compat_result() -> TrainResult:
        cached = compat_cache.get("result")
        if cached is not None:
            return cached
        prog.set_strict_zero_dtoh(False)
        winner_attempt_idx = _materialize_strict_attempt_index(best_state.attempt_id_device)
        winner_host_metadata = attempt_host_metadata[winner_attempt_idx]
        compat_winner = _export_compat_attempt_from_strict_state(
            prog,
            best_state,
            winner_host_metadata,
            mask_name,
            positives,
            negatives,
            config,
        )
        rule_frequency = (
            _compute_strict_rule_frequency(
                prog,
                mask_name,
                positives,
                negatives,
                best_state.rel_names,
                best_state.candidates,
                best_state.n,
                best_state.allow_recursive,
                best_state.argmax_candidate_id_device,
                attempt_selected_candidate_ids,
                attempt_argmax_candidate_ids,
            )
            if compat_winner.converged
            else 0.0
        )
        compat_result = _build_compat_train_result(
            [compat_winner],
            compat_winner,
            source,
            mask_name,
            positives,
            negatives,
            config,
            global_steps,
            _compute_holdout=False,
            rule_frequency_override=rule_frequency,
            attempt_count_override=attempt_count,
        )
        compat_cache["result"] = compat_result
        return compat_result

    strict_artifact = StrictLearnedArtifact(
        candidate_map=best_state.candidate_map,
        config_snapshot=config,
        telemetry=TrainTelemetry(),
        strict_gpu_native=True,
        compat_materialized=False,
        _compat_exporter=lambda: _export_compat_result().artifact,
    )
    return StrictTrainResult(
        attempt_count=attempt_count,
        total_steps=global_steps,
        single_attempt=attempt_count == 1,
        artifact=strict_artifact,
        strict_gpu_native=True,
        compat_materialized=False,
        _compat_exporter=_export_compat_result,
    )


def _build_relation_native_strict_train_result(
    *,
    best_state: _StrictAttemptState | None,
    config: TrainConfig,
    global_steps: int,
    attempt_count: int,
) -> StrictTrainResult:
    if best_state is None:
        return StrictTrainResult(
            attempt_count=attempt_count,
            total_steps=global_steps,
            single_attempt=attempt_count == 1,
        )

    strict_artifact = StrictLearnedArtifact(
        candidate_map=best_state.candidate_map,
        config_snapshot=config,
        telemetry=TrainTelemetry(
            steps=list(best_state.telemetry_steps),
            step_timings=dict(best_state.telemetry_timings),
        ),
        strict_gpu_native=True,
        compat_materialized=False,
        _compat_exporter=None,
    )
    return StrictTrainResult(
        attempt_count=attempt_count,
        total_steps=global_steps,
        single_attempt=attempt_count == 1,
        artifact=strict_artifact,
        strict_gpu_native=True,
        compat_materialized=False,
        _compat_exporter=None,
    )


def _select_winner(
    attempts: list[_AttemptResult],
    prog,
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
        return TrainResult(
            total_steps=global_steps,
            attempt_count=0,
            strict_gpu_native=config.strict_gpu_native,
            compat_materialized=not config.strict_gpu_native,
        )

    winner = _pick_winner(attempts)
    return _build_compat_train_result(
        attempts,
        winner,
        source,
        mask_name,
        positives,
        negatives,
        config,
        global_steps,
        _compute_holdout=_compute_holdout,
    )
