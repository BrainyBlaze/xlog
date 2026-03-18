# python/tests/test_ilp_robustness.py
"""Robustness tests: input validation, NaN/Inf detection, edge cases."""
import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda
skip_unless_pyxlog_cuda()

from pyxlog.ilp import (
    DenseMaskBackend,
    SparseMaskBackend,
    TrainConfig,
    IlpCandidateError,
    IlpConfigError,
    IlpTrainingError,
    train_only,
)

SOURCE = """
    edge(1, 2). edge(2, 3).
    learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""


def test_empty_positives_raises_config_error():
    with pytest.raises(IlpConfigError, match="positives"):
        train_only(SOURCE, "W", [], [], TrainConfig(max_attempts=1))


def test_contradictory_examples_raises_config_error():
    pos = [("reach", [1, 3])]
    neg = [("reach", [1, 3])]
    with pytest.raises(IlpConfigError, match="contradict"):
        train_only(SOURCE, "W", pos, neg, TrainConfig(max_attempts=1))


def test_recursive_candidates_accepted_in_beta():
    """allow_recursive_candidates=True is now a supported beta feature."""
    config = TrainConfig(
        max_attempts=1,
        allow_recursive_candidates=True,
        strict_gpu_native=False,
    )
    # Should not raise — just run with limited budget
    result = train_only(SOURCE, "W", [("reach", [1, 3])], [], config)
    assert result.total_steps > 0


def test_all_distractors_returns_not_converged():
    """When no useful relations exist, training should not converge but not crash."""
    distractor_source = """
        color(1, 0). color(2, 1).
        learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """
    config = TrainConfig(
        step_budget_per_attempt=20, max_attempts=2,
        tau_start=1.0, tau_floor=0.1, seed=0,
        strict_gpu_native=False,
    )
    result = train_only(
        source=distractor_source, mask_name="W",
        positives=[("reach", [1, 3])], negatives=[], config=config,
    )
    assert result.converged is False, "expected non-convergence with all-distractor relations"
    assert result.total_steps > 0


def test_nan_inf_detection_raises_after_limit():
    """NaN/Inf in logits triggers IlpTrainingError after max_numeric_failures."""
    from unittest.mock import patch
    from pyxlog.ilp.backend import SparseMaskBackend

    original_apply = SparseMaskBackend.apply_mask
    call_count = [0]

    def _inject_nan(self, prog, mask_name, W, tau, budget, candidates, n,
                    allow_recursive=False):
        call_count[0] += 1
        result = original_apply(self, prog, mask_name, W, tau, budget,
                                candidates, n, allow_recursive=allow_recursive)
        # Replace cand_probs with NaN to trigger numeric failure detection
        result[0].fill_(float("nan"))
        return result

    config = TrainConfig(
        step_budget_per_attempt=10, max_attempts=5,
        max_numeric_failures=2, seed=0,
        strict_gpu_native=False,
    )

    with patch.object(SparseMaskBackend, "apply_mask", _inject_nan):
        with pytest.raises(IlpTrainingError, match="numeric_instability"):
            train_only(
                source=SOURCE, mask_name="W",
                positives=[("reach", [1, 3])], negatives=[], config=config,
            )
    assert call_count[0] >= 2


def test_degenerate_candidate_raises_candidate_error():
    """N=1 (degenerate) candidate space should fail fast with IlpCandidateError."""
    source = """
        learnable(W) :: reach(X, Y) :- reach(X, Z), reach(Z, Y).
    """
    config = TrainConfig(max_attempts=1, step_budget_per_attempt=5)

    with pytest.raises(IlpCandidateError, match="No valid candidates"):
        train_only(
            source=source,
            mask_name="W",
            positives=[("reach", [1, 2])],
            negatives=[],
            config=config,
        )


def test_cuda_oom_error_shape_and_context():
    """CUDA OOM errors should be wrapped as IlpTrainingError with terminal_reason."""
    from unittest.mock import patch

    config = TrainConfig(
        step_budget_per_attempt=8,
        max_attempts=1,
        seed=0,
        strict_gpu_native=False,
    )

    original_apply = SparseMaskBackend.apply_mask

    def _raise_oom(*_args, **_kwargs):
        raise RuntimeError("CUDA out of memory while allocating temporary buffer")

    with patch.object(DenseMaskBackend, "apply_mask", _raise_oom), \
         patch.object(SparseMaskBackend, "apply_mask", _raise_oom):
        with pytest.raises(IlpTrainingError) as exc:
            train_only(
                source=SOURCE,
                mask_name="W",
                positives=[("reach", [1, 3])],
                negatives=[],
                config=config,
            )

    assert exc.value.context["terminal_reason"] == "cuda_oom"
