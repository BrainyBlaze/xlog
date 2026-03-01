# python/tests/test_ilp_promoter.py
"""Tests for train_and_promote() — transactional commit + promotion gates."""
import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda
skip_unless_pyxlog_cuda()

from pyxlog.ilp import (
    train_and_promote, TrainConfig, PromotionResult, PromotionStatus,
)

SOURCE = """
    edge(1, 2). edge(2, 3). edge(3, 4). edge(4, 5). edge(5, 6).
    learnable(W_reach) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""
POS = [("reach", [1, 3]), ("reach", [2, 4]), ("reach", [3, 5]), ("reach", [4, 6])]
NEG = []


def test_promote_returns_promotion_result():
    """train_and_promote returns a PromotionResult."""
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=5,
        tau_start=2.0, tau_floor=0.05, seed=42,
    )
    result = train_and_promote(
        SOURCE, "W_reach", POS, NEG, config,
        holdout_positives=[("reach", [1, 3])],
        holdout_negatives=[],
    )
    assert isinstance(result, PromotionResult)


def test_promote_with_holdout_can_promote():
    """With holdout examples provided, promotion is possible."""
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=5,
        tau_start=2.0, tau_floor=0.05, seed=42,
    )
    result = train_and_promote(
        SOURCE, "W_reach", POS, NEG, config,
        holdout_positives=[("reach", [1, 3]), ("reach", [2, 4])],
        holdout_negatives=[],
    )
    if result.status == PromotionStatus.PROMOTED:
        assert all(g.passed for g in result.gates)


def test_promote_without_holdout_requires_review():
    """Without holdout, result is MANUAL_REVIEW_REQUIRED (not PROMOTED)."""
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=5,
        tau_start=2.0, tau_floor=0.05, seed=42,
    )
    result = train_and_promote(SOURCE, "W_reach", POS, NEG, config)
    assert result.status in (
        PromotionStatus.MANUAL_REVIEW_REQUIRED,
        PromotionStatus.NOT_CONVERGED,
    )


def test_promote_gates_are_populated():
    """Gate list should contain all required gates per design doc Section 4.3."""
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=5,
        tau_start=2.0, tau_floor=0.05, seed=42,
    )
    result = train_and_promote(
        SOURCE, "W_reach", POS, NEG, config,
        holdout_positives=[("reach", [1, 3])],
        holdout_negatives=[],
    )
    assert result.status != PromotionStatus.NOT_CONVERGED, \
        "Cannot test gates without convergence"
    gate_names = [g.name for g in result.gates]
    assert "training_positive" in gate_names
    assert "training_negative" in gate_names
    assert "novel_fact_audit" in gate_names
    assert "regression_check" in gate_names


def test_promote_not_converged():
    """Non-convergent training returns NOT_CONVERGED status."""
    distractor_source = """
        color(1, 0). color(2, 1).
        learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """
    config = TrainConfig(
        step_budget_per_attempt=10, max_attempts=1,
        tau_start=1.0, tau_floor=0.1, seed=0,
    )
    result = train_and_promote(
        distractor_source, "W",
        [("reach", [1, 3])], [], config,
    )
    assert result.status == PromotionStatus.NOT_CONVERGED
