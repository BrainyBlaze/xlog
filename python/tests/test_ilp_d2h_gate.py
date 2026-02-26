# python/tests/test_ilp_d2h_gate.py
"""Tests for D2H transfer counter exposed via PyO3."""
import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda
skip_unless_pyxlog_cuda()

REACH_SOURCE = """
    edge(1, 2). edge(2, 3). edge(3, 4).
    learnable(W_reach) :: reach(X, Y) :- b1(X, Z), b2(Z, Y).
"""


def test_d2h_counter_accessible():
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    assert prog.d2h_transfer_count() == 0


def test_d2h_counter_reset():
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    # fact_exists triggers D2H transfers
    prog.evaluate()
    prog.fact_exists("edge", [1, 2])
    assert prog.d2h_transfer_count() > 0

    prog.reset_d2h_transfer_count()
    assert prog.d2h_transfer_count() == 0
