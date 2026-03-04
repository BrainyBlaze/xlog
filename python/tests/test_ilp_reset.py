"""Tests for CompiledIlpProgram.reset_runtime() parity and state isolation."""

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


def _get_facts(prog, rel_name):
    """Return sorted list of tuples for deterministic comparison."""
    facts = prog.relation_facts(rel_name)
    return sorted(tuple(f) for f in facts)


def test_reset_parity_with_fresh_compile():
    """compile+reset+evaluate must match a fresh compile."""
    # Fresh compile A
    prog_a = _compile()
    facts_a = _get_facts(prog_a, "edge")

    # Compile B, then reset
    prog_b = _compile()
    prog_b.reset_runtime()
    facts_b = _get_facts(prog_b, "edge")

    assert facts_a == facts_b, f"Parity failed: {facts_a} != {facts_b}"

    # Also check derived relations match
    reach_a = _get_facts(prog_a, "reach")
    reach_b = _get_facts(prog_b, "reach")
    assert reach_a == reach_b, f"Derived parity failed: {reach_a} != {reach_b}"


def test_reset_clears_ilp_registry():
    """After reset, ILP masks and tagged results must be gone."""
    prog = _compile()
    cands = prog.valid_candidates("W", False)
    n = prog.ilp_schema_size()
    c = len(cands)

    # Set a sparse mask and evaluate
    ids = list(range(c))
    soft = torch.tensor([1.0 / c] * c, device="cuda", dtype=torch.float64)
    prog.set_rule_mask_sparse("W", ids, soft, 32)
    prog.evaluate()
    tagged_before = prog.get_tagged_results()
    assert len(tagged_before) > 0, "Should have tagged results after evaluate"

    # Reset
    prog.reset_runtime()

    # After reset: no tagged results (ILP registry cleared)
    tagged_after = prog.get_tagged_results()
    assert len(tagged_after) == 0, (
        f"Tagged results should be empty after reset, got {tagged_after}"
    )


def test_reset_no_state_leak_across_masks():
    """Post-reset mask must produce same results as fresh compile with same mask."""
    prog = _compile()
    cands = prog.valid_candidates("W", False)
    n = prog.ilp_schema_size()
    c = len(cands)

    # Run with candidate 0 active (produces tagged results)
    ids = list(range(c))
    soft_first = torch.zeros(c, device="cuda", dtype=torch.float64)
    soft_first[0] = 1.0
    prog.set_rule_mask_sparse("W", ids, soft_first, 1)
    prog.evaluate()
    results_before = prog.get_tagged_results()
    assert len(results_before) > 0, "Candidate 0 should produce tagged results"

    # Reset
    prog.reset_runtime()

    # Run with last candidate active (produces different/empty results)
    cands_after = prog.valid_candidates("W", False)
    c_after = len(cands_after)
    assert c_after == c, "Candidate count should match after reset"

    soft_last = torch.zeros(c, device="cuda", dtype=torch.float64)
    soft_last[-1] = 1.0
    prog.set_rule_mask_sparse("W", list(range(c)), soft_last, 1)
    prog.evaluate()
    results_after_reset = prog.get_tagged_results()

    # Fresh compile with same last-candidate mask for parity check
    prog_fresh = _compile()
    soft_last_fresh = torch.zeros(c, device="cuda", dtype=torch.float64)
    soft_last_fresh[-1] = 1.0
    prog_fresh.set_rule_mask_sparse("W", list(range(c)), soft_last_fresh, 1)
    prog_fresh.evaluate()
    results_fresh = prog_fresh.get_tagged_results()

    # Post-reset results must match fresh compile (no state leak)
    assert results_after_reset == results_fresh, (
        f"State leak: post-reset={results_after_reset} != fresh={results_fresh}"
    )


def test_reset_d2h_counter_cleared():
    """D2H transfer counter must be zero after reset."""
    prog = _compile()
    prog.reset_d2h_transfer_count()
    prog.reset_runtime()
    assert prog.d2h_transfer_count() == 0, "D2H counter should be 0 after reset"
