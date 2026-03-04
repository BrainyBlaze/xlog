"""Tests for set_rule_mask_sparse() dense-parity behavior."""

import pytest
import unittest.mock as mock

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")
import pyxlog.ilp.backend as backend_mod

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
    soft = torch.tensor([1.0 / c] * c, device="cuda", dtype=torch.float64)
    prog.set_rule_mask_sparse("W", ids, soft, 32)


def test_sparse_derives_same_as_dense():
    prog_dense = _compile()
    prog_sparse = _compile()
    cands = prog_dense.valid_candidates("W", False)
    n = prog_dense.ilp_schema_size()

    soft_list = [0.0] * len(cands)
    soft_list[0] = 1.0
    soft = torch.tensor(soft_list, device="cuda", dtype=torch.float64)

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

    soft_list = [0.0] * len(cands)
    soft_list[0] = 1.0
    soft = torch.tensor(soft_list, device="cuda", dtype=torch.float64)

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
    soft = torch.tensor([1.0] * (c - 1), device="cuda", dtype=torch.float64)
    with pytest.raises(ValueError, match="candidate"):
        prog.set_rule_mask_sparse("W", list(range(c - 1)), soft, 32)


def test_sparse_validates_soft_length():
    prog = _compile()
    c = len(prog.valid_candidates("W", False))
    soft = torch.tensor([1.0] * (c + 1), device="cuda", dtype=torch.float64)
    with pytest.raises(ValueError, match="length"):
        prog.set_rule_mask_sparse("W", list(range(c)), soft, 32)


def test_sparse_top_k_deterministic_tiebreak():
    prog = _compile()
    cands = prog.valid_candidates("W", False)
    c = len(cands)
    soft = torch.tensor([1.0 / c] * c, device="cuda", dtype=torch.float64)
    prog.set_rule_mask_sparse("W", list(range(c)), soft, 1)
    prog.evaluate()
    tags = prog.get_tagged_results()
    if tags:
        i, j, k, _ = tags[0]
        assert (i, j, k) == (cands[0]["i"], cands[0]["j"], cands[0]["k"])


def test_sparse_backend_respects_max_active_rules_budget():
    """Sparse backend should pass budget unchanged to set_rule_mask_sparse."""
    class _ProgCapture:
        def __init__(self, candidates):
            self.candidates = candidates
            self.captured_budget = None

        def valid_candidates(self, _mask_name, _allow_recursive):
            return self.candidates

        def set_rule_mask_sparse(self, _mask_name, candidate_ids, soft_probs, budget, _allow_recursive):
            self.captured_budget = budget
            # Keep Python-side behavior deterministic for apply_mask return path.
            return None

        def reset_d2h_transfer_count(self):
            return None

    candidates = [
        {"id": 0, "i": 0, "j": 1, "k": 1, "left_name": "edge", "right_name": "edge", "head_name": "reach"},
        {"id": 1, "i": 1, "j": 2, "k": 1, "left_name": "edge", "right_name": "edge", "head_name": "reach"},
        {"id": 2, "i": 2, "j": 3, "k": 1, "left_name": "edge", "right_name": "edge", "head_name": "reach"},
    ]
    prog = _ProgCapture(candidates)
    backend = backend_mod.SparseMaskBackend()
    W = torch.randn(len(candidates), device="cpu")

    backend.apply_mask(
        prog=prog,
        mask_name="W",
        W=W,
        tau=1.0,
        budget=2,
        candidates=candidates,
        n=4,
        allow_recursive=False,
    )

    assert prog.captured_budget == 2


def test_sparse_selected_hard_maps_live_indices_back_to_original():
    """selected_hard should use artifact candidate indices, not live candidate indices."""
    class _ProgCapture:
        def __init__(self, candidates):
            self.candidates = candidates

        def valid_candidates(self, _mask_name, _allow_recursive):
            return self.candidates

        def set_rule_mask_sparse(self, mask_name, candidate_ids, soft_probs, budget, allow_recursive):
            assert mask_name == "W"
            assert candidate_ids == list(range(len(self.candidates)))
            assert budget == 2
            return None

        def reset_d2h_transfer_count(self):
            return None

    original = [
        {"id": 0, "i": 0, "j": 1, "k": 1, "left_name": "edge", "right_name": "edge", "head_name": "reach"},
        {"id": 1, "i": 1, "j": 2, "k": 1, "left_name": "edge", "right_name": "edge", "head_name": "reach"},
    ]
    # Simulate live candidates growing with one extra candidate that has no match.
    live = [
        {"id": 0, "i": 9, "j": 9, "k": 1, "left_name": "edge", "right_name": "edge", "head_name": "reach"},
        original[0],
        original[1],
    ]

    prog = _ProgCapture(live)
    backend = backend_mod.SparseMaskBackend()

    def _det_softmax(logits, tau, hard=False, dim=-1):
        return torch.tensor([0.1, 0.8, 0.2], device=logits.device, dtype=logits.dtype, requires_grad=True)

    with mock.patch("pyxlog.ilp.backend.F.gumbel_softmax", _det_softmax):
        _, selected = backend.apply_mask(
            prog=prog,
            mask_name="W",
            W=torch.tensor([1.0, 2.0], device="cpu", requires_grad=True),
            tau=1.0,
            budget=2,
            candidates=original,
            n=4,
            allow_recursive=False,
        )

    assert selected == [0, 1], f"selected_hard was not mapped to original indices: {selected}"
