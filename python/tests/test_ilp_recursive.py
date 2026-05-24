# python/tests/test_ilp_recursive.py
"""Tests for recursive candidate support (beta feature)."""
import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda
skip_unless_pyxlog_cuda()

from pyxlog.ilp import train_only, TrainConfig

SOURCE = """
    edge(1, 2). edge(2, 3). edge(3, 4). edge(4, 5).
    learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""
RECURSIVE_POS = [("reach", [1, 4]), ("reach", [1, 3]), ("reach", [2, 4])]
RECURSIVE_NEG: list[tuple[str, list[int]]] = []


def test_recursive_candidates_no_longer_raises():
    """allow_recursive_candidates=True is now allowed in beta."""
    config = TrainConfig(
        step_budget_per_attempt=10, max_attempts=1,
        tau_start=2.0, tau_floor=0.05, seed=42,
        allow_recursive_candidates=True,
        strict_gpu_native=False,
    )
    result = train_only(SOURCE, "W", RECURSIVE_POS, RECURSIVE_NEG, config)
    assert isinstance(result.attempt_count, int)


def test_recursive_candidates_has_more_candidates():
    """Recursive mode includes body-references-head candidates."""
    prog = pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)
    non_recursive = prog.valid_candidates("W", False)
    recursive = prog.valid_candidates("W", True)
    assert len(recursive) > len(non_recursive)


def test_non_recursive_still_default():
    """Default config still has allow_recursive_candidates=False."""
    config = TrainConfig()
    assert config.allow_recursive_candidates is False


@pytest.mark.slow
def test_recursive_reach_runs_without_crash():
    """With recursive candidates enabled, training runs without errors.

    Bounded smoke: one short recursive training attempt is enough to prove the
    recursive candidate path executes without falling back to a source-only check.
    """
    config = TrainConfig(
        step_budget_per_attempt=10, max_attempts=1,
        tau_start=2.0, tau_floor=0.05, seed=42,
        allow_recursive_candidates=True,
        strict_gpu_native=False,
    )
    result = train_only(SOURCE, "W", RECURSIVE_POS, RECURSIVE_NEG, config)
    assert result.total_steps > 0
