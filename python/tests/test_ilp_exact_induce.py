"""M8 Phase 1 — parity contract for induce_exact(backend=...).

The ``python`` backend is the reference implementation. The ``native`` backend
is implemented by the ``xlog-induce`` engine. This file locks the contract:

1. On bounded requests, ``native`` must return the same ordered
   ``ExactInductionResult`` as ``python``.
2. The ``native`` hot loop must not scale host/device transfers with the number
   of candidate pairs — a small constant-size D2H is the only allowed export.

Both tests are the release contract for the wired native backend: they catch
semantic drift against the strict Python reference and host-transfer scaling
regressions in the native scoring path.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda

skip_unless_pyxlog_cuda()

from pyxlog.ilp import induce_exact  # noqa: E402


# ── Helpers ─────────────────────────────────────────────────────────────

_TARGET_SOURCE = """
    pred p_A(u64, u64).
    pred p_B(u64, u64).
    pred p_C(u64, u64).
    pred p_D(u64, u64).
    pred p_E(u64, u64).
    pred p_F(u64, u64).
    learnable(W_chain_p_A)  :: p_A(X, Y) :- bL(X, Z), bR(Z, Y).
    learnable(W_star_p_A)   :: p_A(X, Y) :- bL(X, Y), bR(X, Y).
    learnable(W_fanout_p_A) :: p_A(X, Y) :- bL(X, Z), bR(X, Y).
    learnable(W_fanin_p_A)  :: p_A(X, Y) :- bL(X, Y), bR(Z, Y).
"""


def _build_bounded_request(n_candidates: int = 4):
    """Build a small DTS-shaped ILP program + positive/negative tensors.

    ``n_candidates`` controls how many of ``p_B..p_F`` are passed as
    body candidates (excluding the target ``p_A``).
    """
    assert 2 <= n_candidates <= 5
    prog = pyxlog.IlpProgramFactory.compile(_TARGET_SOURCE, device=0, memory_mb=64)

    dev = torch.device("cuda")
    # Upload some facts per candidate so the scorer has data to score against.
    # chain(p_B, p_C) derives (1, 4) and (2, 5) via Z=2, Z=3.
    prog.put_relation("p_B", [
        torch.tensor([1, 2], dtype=torch.int64, device=dev),
        torch.tensor([2, 3], dtype=torch.int64, device=dev),
    ])
    prog.put_relation("p_C", [
        torch.tensor([2, 3, 4], dtype=torch.int64, device=dev),
        torch.tensor([4, 5, 6], dtype=torch.int64, device=dev),
    ])
    prog.put_relation("p_D", [
        torch.tensor([1, 2], dtype=torch.int64, device=dev),
        torch.tensor([4, 5], dtype=torch.int64, device=dev),
    ])
    prog.put_relation("p_E", [
        torch.tensor([7], dtype=torch.int64, device=dev),
        torch.tensor([8], dtype=torch.int64, device=dev),
    ])
    prog.put_relation("p_F", [
        torch.tensor([9], dtype=torch.int64, device=dev),
        torch.tensor([10], dtype=torch.int64, device=dev),
    ])

    pos_arg0 = torch.tensor([1, 2], dtype=torch.int64, device=dev)
    pos_arg1 = torch.tensor([4, 5], dtype=torch.int64, device=dev)
    # One negative so neg_covered scoring path is exercised.
    neg_arg0 = torch.tensor([7], dtype=torch.int64, device=dev)
    neg_arg1 = torch.tensor([8], dtype=torch.int64, device=dev)

    candidate_relations = [f"p_{chr(ord('B') + i)}" for i in range(n_candidates)]

    kwargs = dict(
        head_relation="p_A",
        candidate_relations=candidate_relations,
        positive_arg0=pos_arg0,
        positive_arg1=pos_arg1,
        negative_arg0=neg_arg0,
        negative_arg1=neg_arg1,
        k_per_topology=2,
        deterministic=True,
    )
    return prog, kwargs


def _run_and_collect_transfer_stats(prog, kwargs):
    """Reset D2H counter, run native backend, return transfer stats snapshot."""
    prog.reset_d2h_transfer_count()
    _ = induce_exact(prog, backend="native", **kwargs)
    return {"dtoh_calls": prog.d2h_transfer_count()}


# ── Parity contract ────────────────────────────────────────────────────

def test_induce_exact_native_matches_python_reference(monkeypatch):
    """Native backend returns the same ordered ExactInductionResult as python reference.

    Python reference uses ``strict_per_topology=True`` so each topology's
    scoring is isolated — matching the native backend's by-construction
    strict semantics. Without this, the Python prototype's stale-mask
    contamination inflates fanout/fanin coverage relative to the native
    kernel (see exact_induce.py:strict_per_topology for details).
    """
    prog, kwargs = _build_bounded_request(n_candidates=3)
    monkeypatch.setenv("XLOG_ALLOW_PYTHON_ILP_REFERENCE", "1")

    py_result = induce_exact(
        prog, backend="python", strict_per_topology=True, **kwargs
    )
    native_result = induce_exact(prog, backend="native", **kwargs)

    # Summary equality
    assert native_result.total_scored == py_result.total_scored
    assert native_result.candidate_count == py_result.candidate_count
    assert native_result.positive_count == py_result.positive_count
    assert native_result.negative_count == py_result.negative_count

    # Ordered candidate equality (topology then rank). The Python reference
    # returns ScoredCandidate dataclass instances; native must yield equivalent
    # tuples (topology, L, R, pos, neg, rank, diagnostics).
    assert len(native_result.candidates) == len(py_result.candidates)
    for n, p in zip(native_result.candidates, py_result.candidates):
        assert n.topology == p.topology
        assert n.head_relation == p.head_relation
        assert n.left_relation == p.left_relation
        assert n.right_relation == p.right_relation
        assert n.positives_covered == p.positives_covered
        assert n.negatives_covered == p.negatives_covered
        assert n.local_rank == p.local_rank


# ── D2H hot-loop gate ──────────────────────────────────────────────────

def test_induce_exact_native_does_not_scale_d2h_with_candidate_pairs():
    """Native backend: transfers must be constant-size, not per-candidate-pair.

    The python reference calls set_rule_mask + evaluate + batch_fact_membership_device
    per (topology, left, right) pair, which scales D2H with N^2 × 4. The
    native engine must keep D2H flat at a small constant (one final result export).
    """
    prog_small, kwargs_small = _build_bounded_request(n_candidates=2)
    small_stats = _run_and_collect_transfer_stats(prog_small, kwargs_small)

    prog_large, kwargs_large = _build_bounded_request(n_candidates=5)
    large_stats = _run_and_collect_transfer_stats(prog_large, kwargs_large)

    # Tolerate a small constant slack (the final tiny result export can grow
    # slightly if K × 4 widens). 2 is the slack budget the plan specifies.
    assert large_stats["dtoh_calls"] <= small_stats["dtoh_calls"] + 2, (
        f"native D2H scaled with candidates: small={small_stats['dtoh_calls']}, "
        f"large={large_stats['dtoh_calls']}"
    )
