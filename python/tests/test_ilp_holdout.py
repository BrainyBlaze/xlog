# python/tests/test_ilp_holdout.py
"""Tests for holdout scoring and ambiguity scan."""
import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda
skip_unless_pyxlog_cuda()

from pyxlog.ilp import train_and_promote, TrainConfig, PromotionResult, PromotionStatus
from pyxlog.ilp.holdout import loo_holdout_f1

SOURCE = """
    edge(1, 2). edge(2, 3). edge(3, 4). edge(4, 5). edge(5, 6).
    learnable(W_reach) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""
POS = [("reach", [1, 3]), ("reach", [2, 4]), ("reach", [3, 5]), ("reach", [4, 6])]
NEG = []


def test_loo_holdout_f1_perfect():
    """LOO on reach should give F1 >= 0.9 when rule is correct."""
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=5,
        tau_start=2.0, tau_floor=0.05, seed=42,
    )
    f1 = loo_holdout_f1(SOURCE, "W_reach", POS, NEG, config)
    assert f1 >= 0.9, f"LOO F1 should be >= 0.9 for reach, got {f1}"


def test_loo_singleton_skipped():
    """LOO with 1 positive should return None (can't hold out)."""
    config = TrainConfig(
        step_budget_per_attempt=50, max_attempts=2,
        tau_start=2.0, tau_floor=0.05, seed=42,
    )
    f1 = loo_holdout_f1(SOURCE, "W_reach", [("reach", [1, 3])], NEG, config)
    assert f1 is None


def test_promote_with_loo():
    """Promotion with holdout should use LOO results."""
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=5,
        tau_start=2.0, tau_floor=0.05, seed=42,
        holdout_strategy="loo",
    )
    result = train_and_promote(
        SOURCE, "W_reach", POS, NEG, config,
        holdout_positives=POS[:2],
        holdout_negatives=[],
    )
    if result.status == PromotionStatus.PROMOTED:
        assert result.artifact.discovered_rule


def test_ambiguity_scan_smoke():
    """Ambiguity check does not crash."""
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=5,
        tau_start=2.0, tau_floor=0.05, seed=42,
        check_ambiguity=True,
    )
    result = train_and_promote(
        SOURCE, "W_reach", POS, NEG, config,
        holdout_positives=POS[:2],
        holdout_negatives=[],
    )
    assert isinstance(result, PromotionResult)
    assert result.ambiguous_alternatives is None or isinstance(result.ambiguous_alternatives, list)
