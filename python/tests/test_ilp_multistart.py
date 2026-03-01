# python/tests/test_ilp_multistart.py
"""Integration tests for structured multi-start with distractor relations."""
import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda
skip_unless_pyxlog_cuda()

from pyxlog.ilp import train_only, TrainConfig

GRANDPARENT_SOURCE = """
    parent(1, 2). parent(2, 3). parent(3, 4). parent(4, 5).
    gender(1, 0). gender(2, 1). gender(3, 0). gender(4, 1).
    sibling(1, 3). sibling(3, 1).
    learnable(W_gp) :: grandparent(X, Y) :- bL(X, Z), bR(Z, Y).
"""
GRANDPARENT_POS = [("grandparent", [1, 3]), ("grandparent", [2, 4]), ("grandparent", [3, 5])]
GRANDPARENT_NEG = [("grandparent", [1, 2]), ("grandparent", [3, 1])]


def test_multistart_converges_with_distractors():
    config = TrainConfig(
        step_budget_per_attempt=60, max_attempts=3,
        tau_start=2.0, tau_floor=0.05, seed=42,
    )
    result = train_only(
        source=GRANDPARENT_SOURCE, mask_name="W_gp",
        positives=GRANDPARENT_POS, negatives=GRANDPARENT_NEG, config=config,
    )
    assert result.converged
    assert "parent" in result.discovered_rule


def test_multistart_reports_attempt_count():
    config = TrainConfig(
        step_budget_per_attempt=60, max_attempts=3,
        tau_start=2.0, tau_floor=0.05, seed=42,
    )
    result = train_only(
        source=GRANDPARENT_SOURCE, mask_name="W_gp",
        positives=GRANDPARENT_POS, negatives=GRANDPARENT_NEG, config=config,
    )
    assert 1 <= result.attempt_count <= 7
