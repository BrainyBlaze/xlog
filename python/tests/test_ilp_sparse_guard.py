"""Guard tests: sparse trainer path must not fall back to dense APIs."""

import inspect
import re

import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda

skip_unless_pyxlog_cuda()

from pyxlog.ilp import train_only, TrainConfig
import pyxlog.ilp.backend as backend_mod


SOURCE = """
edge(1, 2). edge(2, 3). edge(3, 4). edge(4, 5).
learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""
POS = [("reach", [1, 3]), ("reach", [2, 4])]
NEG = [("reach", [1, 1])]


def test_sparse_backend_prefers_selected_sparse_api():
    """Sparse backend apply_mask must use the selected sparse API, never dense.

    Two-pronged verification:
    1. Static: inspect SparseMaskBackend.apply_mask source for bare
       set_rule_mask( calls.
    2. Runtime: run a short sparse training to confirm the code path
       executes without error.
    """
    # --- Prong 1: static source inspection ---
    src = inspect.getsource(backend_mod.SparseMaskBackend.apply_mask)

    # Find all set_rule_mask calls; filter out set_rule_mask_sparse.
    # Pattern: .set_rule_mask( NOT followed by _sparse
    bare_calls = re.findall(r'\.set_rule_mask\b(?!_sparse)', src)
    assert len(bare_calls) == 0, (
        f"SparseMaskBackend.apply_mask contains {len(bare_calls)} call(s) "
        f"to set_rule_mask (dense API); expected only sparse APIs"
    )

    legacy_sparse_calls = re.findall(r'\.set_rule_mask_sparse\b(?!_selected)', src)
    assert len(legacy_sparse_calls) == 0, (
        "SparseMaskBackend.apply_mask still calls legacy set_rule_mask_sparse"
    )

    selected_sparse_calls = re.findall(r'\.set_rule_mask_sparse_selected\b', src)
    assert len(selected_sparse_calls) > 0, (
        "SparseMaskBackend.apply_mask does not call set_rule_mask_sparse_selected"
    )

    # --- Prong 2: runtime smoke test ---
    config = TrainConfig(
        step_budget_per_attempt=10,
        max_attempts=1,
        seed=42,
        debug_dense_mask=False,  # sparse backend
    )
    result = train_only(SOURCE, "W", POS, NEG, config)
    assert result.attempt_count >= 1


def test_sparse_backend_apply_mask_selected_path_is_zero_dtoh():
    prog = pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)
    backend = backend_mod.SparseMaskBackend()
    candidates = prog.valid_candidates("W", False)
    W = torch.zeros(len(candidates), device="cuda", requires_grad=True)

    prog.reset_host_transfer_stats()
    backend.apply_mask(
        prog=prog,
        mask_name="W",
        W=W,
        tau=1.0,
        budget=2,
        candidates=candidates,
        n=prog.ilp_schema_size(),
        allow_recursive=False,
    )
    after = prog.host_transfer_stats()

    assert after["dtoh_calls"] == 0
    assert after["dtoh_bytes"] == 0


def test_sparse_d2h_counter_clean_after_mask_setup():
    """set_rule_mask_sparse must not increment the D2H transfer counter.

    The Rust implementation uses download_f64_untracked for the DLPack
    soft-probs import, so the D2H counter should remain at zero.
    """
    prog = pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)
    cands = prog.valid_candidates("W", False)
    c = len(cands)

    soft = torch.tensor([1.0 / c] * c, device="cuda", dtype=torch.float64)
    prog.reset_d2h_transfer_count()
    prog.set_rule_mask_sparse("W", list(range(c)), soft, 32)

    assert prog.d2h_transfer_count() == 0, (
        f"set_rule_mask_sparse incremented D2H counter to {prog.d2h_transfer_count()}"
    )
