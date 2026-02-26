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


def test_batch_fact_membership_basic():
    """batch_fact_membership returns correct bool mask."""
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    prog.evaluate()

    # edge relation has: (1,2), (2,3), (3,4)
    facts = [[1, 2], [5, 6], [2, 3]]
    mask = prog.batch_fact_membership("edge", facts)
    assert mask == [True, False, True]


def test_batch_fact_membership_empty_facts():
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    prog.evaluate()
    mask = prog.batch_fact_membership("edge", [])
    assert mask == []


def test_batch_fact_membership_no_d2h_columns():
    """batch_fact_membership must NOT use download_column_*."""
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    prog.evaluate()

    prog.reset_d2h_transfer_count()
    facts = [[1, 2], [5, 6], [2, 3]]
    _ = prog.batch_fact_membership("edge", facts)
    assert prog.d2h_transfer_count() == 0, (
        f"batch_fact_membership triggered {prog.d2h_transfer_count()} column downloads"
    )


def test_batch_tagged_credit_basic():
    """batch_tagged_credit returns per-fact lists of (i,j,k) contributors."""
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    n = prog.ilp_schema_size()
    rel_names = prog.ilp_relation_names()
    k_reach = rel_names.index("reach")
    i_edge = rel_names.index("edge")

    # Set mask so that edge+edge->reach is active
    M_hard = torch.zeros(n, n, n, device="cuda")
    M_soft = torch.zeros(n, n, n, device="cuda")
    M_hard[i_edge, i_edge, k_reach] = 1.0
    M_soft[i_edge, i_edge, k_reach] = 1.0
    prog.set_rule_mask("W_reach", M_hard.view(-1), M_soft.view(-1), n)
    prog.evaluate()

    # reach should derive (1,3), (2,4) via edge(X,Z) join edge(Z,Y)
    facts = [[1, 3], [2, 4], [99, 99]]
    credits = prog.batch_tagged_credit("reach", facts)
    assert len(credits) == 3

    # First two should have at least one contributing entry
    assert len(credits[0]) > 0, f"(1,3) should be derived, got empty credits"
    assert len(credits[1]) > 0, f"(2,4) should be derived, got empty credits"
    # The contributing entry should be (edge_idx, edge_idx, reach_idx)
    assert credits[0][0] == (i_edge, i_edge, k_reach)
    # (99,99) is not derived -- no contributors
    assert credits[2] == []


def test_batch_tagged_credit_no_d2h_columns():
    """batch_tagged_credit must NOT use download_column_*."""
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    n = prog.ilp_schema_size()
    rel_names = prog.ilp_relation_names()
    k_reach = rel_names.index("reach")
    i_edge = rel_names.index("edge")

    M_hard = torch.zeros(n, n, n, device="cuda")
    M_soft = torch.zeros(n, n, n, device="cuda")
    M_hard[i_edge, i_edge, k_reach] = 1.0
    M_soft[i_edge, i_edge, k_reach] = 1.0
    prog.set_rule_mask("W_reach", M_hard.view(-1), M_soft.view(-1), n)
    prog.evaluate()

    prog.reset_d2h_transfer_count()
    facts = [[1, 3], [2, 4], [99, 99]]
    _ = prog.batch_tagged_credit("reach", facts)
    assert prog.d2h_transfer_count() == 0, (
        f"batch_tagged_credit triggered {prog.d2h_transfer_count()} column downloads"
    )
