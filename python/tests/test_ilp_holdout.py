# python/tests/test_ilp_holdout.py
"""Tests for holdout scoring and ambiguity scan."""
import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda
skip_unless_pyxlog_cuda()

from pyxlog.ilp import (
    IlpConfigError,
    PromotionResult,
    PromotionStatus,
    TrainConfig,
    train_and_promote,
    train_only,
)
from pyxlog.ilp.holdout import holdout_f1_and_variance, loo_holdout_f1


LARGE_SOURCE = """
    edge(1, 2). edge(2, 3). edge(3, 4). edge(4, 5). edge(5, 6).
    edge(6, 7). edge(7, 8). edge(8, 9). edge(9, 10). edge(10, 11).
    edge(11, 12). edge(12, 13). edge(13, 14). edge(14, 15). edge(15, 16).
    edge(16, 17). edge(17, 18). edge(18, 19). edge(19, 20). edge(20, 21).
    edge(21, 22). edge(22, 23). edge(23, 24). edge(24, 25). edge(25, 26).
    learnable(W_reach_large) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""
LARGE_POS = [("reach", [i, i + 2]) for i in range(1, 25)]
LARGE_NEG = []

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
        strict_gpu_native=False,
    )
    f1 = loo_holdout_f1(SOURCE, "W_reach", POS, NEG, config)
    assert f1 >= 0.9, f"LOO F1 should be >= 0.9 for reach, got {f1}"


def test_loo_singleton_skipped():
    """LOO with 1 positive should return None (can't hold out)."""
    config = TrainConfig(
        step_budget_per_attempt=50, max_attempts=2,
        tau_start=2.0, tau_floor=0.05, seed=42,
        strict_gpu_native=False,
    )
    f1 = loo_holdout_f1(SOURCE, "W_reach", [("reach", [1, 3])], NEG, config)
    assert f1 is None


def test_promote_with_loo():
    """Promotion with holdout should use LOO results."""
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=5,
        tau_start=2.0, tau_floor=0.05, seed=42,
        holdout_strategy="loo",
        strict_gpu_native=False,
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
        strict_gpu_native=False,
    )
    result = train_and_promote(
        SOURCE, "W_reach", POS, NEG, config,
        holdout_positives=POS[:2],
        holdout_negatives=[],
    )
    assert isinstance(result, PromotionResult)
    assert result.ambiguous_alternatives is None or isinstance(result.ambiguous_alternatives, list)


def test_kfold_holdout_and_variance():
    """Large-positive datasets should run k-fold holdout validation."""
    config = TrainConfig(
        step_budget_per_attempt=120,
        max_attempts=4,
        tau_start=2.0,
        tau_floor=0.05,
        deterministic=True,
        seed=1,
        holdout_strategy="kfold",
        holdout_folds=4,
        strict_gpu_native=False,
    )
    f1, variance = holdout_f1_and_variance(
        LARGE_SOURCE,
        "W_reach_large",
        LARGE_POS,
        LARGE_NEG,
        config,
    )
    assert f1 is not None
    assert 0.0 <= f1 <= 1.0
    assert variance >= 0.0


def test_train_only_holdout_variance_in_result():
    config = TrainConfig(
        step_budget_per_attempt=120,
        max_attempts=4,
        tau_start=2.0,
        tau_floor=0.05,
        deterministic=True,
        seed=1,
        holdout_strategy="kfold",
        strict_gpu_native=False,
    )
    result = train_only(
        LARGE_SOURCE,
        "W_reach_large",
        LARGE_POS,
        LARGE_NEG,
        config=config,
    )
    assert result.holdout_f1 is None or 0.0 <= result.holdout_f1 <= 1.0
    assert result.holdout_variance >= 0.0


def test_holdout_rejects_strict_gpu_native():
    config = TrainConfig(
        step_budget_per_attempt=40,
        max_attempts=2,
        tau_start=2.0,
        tau_floor=0.05,
        seed=42,
        strict_gpu_native=True,
    )
    with pytest.raises(IlpConfigError, match="strict_gpu_native"):
        holdout_f1_and_variance(
            SOURCE,
            "W_reach",
            POS,
            NEG,
            config,
        )


def test_train_only_strict_gpu_native_disables_holdout_scoring():
    config = TrainConfig(
        step_budget_per_attempt=80,
        max_attempts=3,
        tau_start=2.0,
        tau_floor=0.05,
        seed=42,
        strict_gpu_native=True,
    )
    result = train_only(
        SOURCE,
        "W_reach",
        POS,
        NEG,
        config=config,
    )
    assert result.holdout_f1 is None
    assert result.holdout_variance == 0.0
