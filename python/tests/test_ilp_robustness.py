# python/tests/test_ilp_robustness.py
"""Robustness tests: input validation, NaN/Inf detection, edge cases."""
import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda
skip_unless_pyxlog_cuda()

from pyxlog.ilp import train_only, TrainConfig, IlpConfigError, IlpTrainingError

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


def test_recursive_candidates_rejected_in_alpha():
    """allow_recursive_candidates=True is a beta-only feature."""
    config = TrainConfig(max_attempts=1, allow_recursive_candidates=True)
    with pytest.raises(IlpConfigError, match="beta"):
        train_only(SOURCE, "W", [("reach", [1, 3])], [], config)


def test_all_distractors_returns_not_converged():
    """When no useful relations exist, training should not converge but not crash."""
    distractor_source = """
        color(1, 0). color(2, 1).
        learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """
    config = TrainConfig(
        step_budget_per_attempt=20, max_attempts=2,
        tau_start=1.0, tau_floor=0.1, seed=0,
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
    import pyxlog.ilp.trainer as trainer_mod

    original_fn = trainer_mod._build_budget_aware_mask

    call_count = [0]

    def _inject_nan(*args, **kwargs):
        call_count[0] += 1
        result = original_fn(*args, **kwargs)
        result[1].fill_(float("nan"))
        return result

    config = TrainConfig(
        step_budget_per_attempt=10, max_attempts=5,
        max_numeric_failures=2, seed=0,
    )

    with patch.object(trainer_mod, "_build_budget_aware_mask", _inject_nan):
        with pytest.raises(IlpTrainingError, match="numeric_instability"):
            train_only(
                source=SOURCE, mask_name="W",
                positives=[("reach", [1, 3])], negatives=[], config=config,
            )
    assert call_count[0] >= 2
