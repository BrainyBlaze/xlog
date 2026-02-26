"""Tests for set_rule_mask_sparse() dense-parity behavior."""

import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda

skip_unless_pyxlog_cuda()


SOURCE = """
edge(1, 2). edge(2, 3). edge(3, 4).
learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""

POS = [("reach", [1, 3]), ("reach", [2, 4])]


def _compile():
    return pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)


def _dense_set_single(prog, n, cand):
    hard = torch.zeros(n * n * n, device="cuda")
    soft = torch.zeros(n * n * n, device="cuda")
    flat = cand["i"] * n * n + cand["j"] * n + cand["k"]
    hard[flat] = 1.0
    soft[flat] = 1.0
    prog.set_rule_mask("W", hard, soft, n)


def test_sparse_mask_exists():
    prog = _compile()
    cands = prog.valid_candidates("W", False)
    c = len(cands)
    ids = list(range(c))
    soft = [1.0 / c] * c
    prog.set_rule_mask_sparse("W", ids, soft, 32)


def test_sparse_derives_same_as_dense():
    prog_dense = _compile()
    prog_sparse = _compile()
    cands = prog_dense.valid_candidates("W", False)
    n = prog_dense.ilp_schema_size()

    soft = [0.0] * len(cands)
    soft[0] = 1.0

    _dense_set_single(prog_dense, n, cands[0])
    prog_dense.evaluate()
    dense_pos = [prog_dense.fact_exists(r, v) for r, v in POS]

    prog_sparse.set_rule_mask_sparse("W", list(range(len(cands))), soft, 32)
    prog_sparse.evaluate()
    sparse_pos = [prog_sparse.fact_exists(r, v) for r, v in POS]

    assert dense_pos == sparse_pos, f"dense={dense_pos} sparse={sparse_pos}"


def test_sparse_tagged_entries_match_dense():
    prog_dense = _compile()
    prog_sparse = _compile()
    cands = prog_dense.valid_candidates("W", False)
    n = prog_dense.ilp_schema_size()

    soft = [0.0] * len(cands)
    soft[0] = 1.0

    _dense_set_single(prog_dense, n, cands[0])
    prog_dense.evaluate()
    dense_tags = prog_dense.get_tagged_results()

    prog_sparse.set_rule_mask_sparse("W", list(range(len(cands))), soft, 32)
    prog_sparse.evaluate()
    sparse_tags = prog_sparse.get_tagged_results()

    assert set(dense_tags) == set(sparse_tags)


def test_sparse_validates_candidate_count():
    prog = _compile()
    c = len(prog.valid_candidates("W", False))
    with pytest.raises(ValueError, match="candidate"):
        prog.set_rule_mask_sparse("W", list(range(c - 1)), [1.0] * (c - 1), 32)


def test_sparse_validates_soft_length():
    prog = _compile()
    c = len(prog.valid_candidates("W", False))
    with pytest.raises(ValueError, match="length"):
        prog.set_rule_mask_sparse("W", list(range(c)), [1.0] * (c + 1), 32)


def test_sparse_top_k_deterministic_tiebreak():
    prog = _compile()
    cands = prog.valid_candidates("W", False)
    c = len(cands)
    soft = [1.0 / c] * c
    prog.set_rule_mask_sparse("W", list(range(c)), soft, 1)
    prog.evaluate()
    tags = prog.get_tagged_results()
    if tags:
        i, j, k, _ = tags[0]
        assert (i, j, k) == (cands[0]["i"], cands[0]["j"], cands[0]["k"])
