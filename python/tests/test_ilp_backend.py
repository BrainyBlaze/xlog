# python/tests/test_ilp_backend.py
"""Tests for trainer backend abstraction (dense vs sparse)."""
import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda
skip_unless_pyxlog_cuda()

from pyxlog.ilp import train_only, TrainConfig

SOURCE = """
    edge(1, 2). edge(2, 3). edge(3, 4). edge(4, 5). edge(5, 6).
    learnable(W_reach) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""
POS = [("reach", [1, 3]), ("reach", [2, 4]), ("reach", [3, 5]), ("reach", [4, 6])]
NEG = []


def test_sparse_backend_converges():
    """Strict sparse backend converges; rule text is compat-export only."""
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=5,
        tau_start=2.0, tau_floor=0.05, seed=42,
        debug_dense_mask=False,
    )
    result = train_only(SOURCE, "W_reach", POS, NEG, config)
    compat = result.export_compat_result()
    assert compat.converged
    assert "edge" in compat.discovered_rule


def test_dense_backend_still_works():
    """train_only with debug_dense_mask=True (dense fallback) still works."""
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=5,
        tau_start=2.0, tau_floor=0.05, seed=42,
        debug_dense_mask=True,
        strict_gpu_native=False,
    )
    result = train_only(SOURCE, "W_reach", POS, NEG, config)
    assert result.converged
    assert "edge" in result.discovered_rule


def test_sparse_and_dense_find_same_rule():
    """Both backends converge to the same rule on the same seed."""
    base = dict(
        step_budget_per_attempt=100, max_attempts=5,
        tau_start=2.0, tau_floor=0.05, seed=42,
    )
    r_sparse = train_only(SOURCE, "W_reach", POS, NEG,
                          TrainConfig(**base, debug_dense_mask=False))
    r_dense = train_only(SOURCE, "W_reach", POS, NEG,
                         TrainConfig(**base, debug_dense_mask=True, strict_gpu_native=False))
    r_sparse_compat = r_sparse.export_compat_result()
    if r_sparse_compat.converged and r_dense.converged:
        assert r_sparse_compat.discovered_rule == r_dense.discovered_rule
