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


from pyxlog.ilp import TrainConfig, train_only
from pyxlog.ilp.trainer import _train_on_compiled


def test_train_deterministic_reproducibility():
    """Two train_only() calls with same seed must produce identical results."""
    config = TrainConfig(
        step_budget_per_attempt=150,
        max_attempts=3,
        tau_start=2.0,
        tau_floor=0.05,
        seed=42,
        deterministic=True,
        debug_dense_mask=False,
    )

    result_1 = train_only(
        source=SOURCE,
        mask_name="W",
        positives=POS,
        negatives=[],
        config=config,
        _compute_holdout=False,
    )

    result_2 = train_only(
        source=SOURCE,
        mask_name="W",
        positives=POS,
        negatives=[],
        config=config,
        _compute_holdout=False,
    )

    assert result_1.converged == result_2.converged, (
        f"Determinism: converged mismatch {result_1.converged} vs {result_2.converged}"
    )
    assert result_1.discovered_rule == result_2.discovered_rule, (
        f"Determinism: rule mismatch {result_1.discovered_rule!r} vs {result_2.discovered_rule!r}"
    )


# ---------------------------------------------------------------------------
# Compile-once + reset_runtime reuse parity tests
# ---------------------------------------------------------------------------

# Use the reliability gate's stage 1 (reach) for parity -- representative and fast.
PARITY_SOURCE = """
    edge(1, 2). edge(2, 3). edge(3, 4). edge(4, 5). edge(5, 6).
    learnable(W_reach) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""
PARITY_POS = [("reach", [1, 3]), ("reach", [2, 4]), ("reach", [3, 5]), ("reach", [4, 6])]
PARITY_NEG = []
PARITY_MASK = "W_reach"


def _parity_config(seed: int) -> TrainConfig:
    return TrainConfig(
        step_budget_per_attempt=150,
        max_attempts=7,
        tau_start=2.0,
        tau_floor=0.05,
        seed=seed,
        device=0,
        memory_mb=512,
    )


@pytest.mark.parametrize("seed", [0, 2])
def test_compile_once_reuse_parity(seed):
    """_train_on_compiled with reset matches fresh train_only on convergence,
    attempt_count, and discovered_rule."""
    config = _parity_config(seed)

    # Baseline: fresh compile via train_only
    fresh = train_only(
        source=PARITY_SOURCE,
        mask_name=PARITY_MASK,
        positives=PARITY_POS,
        negatives=PARITY_NEG,
        config=config,
        _compute_holdout=False,
    )

    # Reuse: compile once, then train_on_compiled with reset
    prog = pyxlog.IlpProgramFactory.compile(
        PARITY_SOURCE, device=config.device, memory_mb=config.memory_mb,
    )
    reused = _train_on_compiled(
        prog, PARITY_SOURCE, PARITY_MASK, PARITY_POS, PARITY_NEG, config,
        _compute_holdout=False,
        _reset_before_first_attempt=True,
    )

    assert fresh.converged == reused.converged, (
        f"seed={seed}: converged mismatch fresh={fresh.converged} vs reused={reused.converged}"
    )
    assert fresh.attempt_count == reused.attempt_count, (
        f"seed={seed}: attempt_count mismatch {fresh.attempt_count} vs {reused.attempt_count}"
    )
    assert fresh.discovered_rule == reused.discovered_rule, (
        f"seed={seed}: rule mismatch {fresh.discovered_rule!r} vs {reused.discovered_rule!r}"
    )


def test_compile_once_multi_seed_isolation():
    """Running two seeds on the same compiled program must not leak state."""
    prog = pyxlog.IlpProgramFactory.compile(
        PARITY_SOURCE, device=0, memory_mb=512,
    )

    results = []
    for seed in [0, 1]:
        config = _parity_config(seed)
        r = _train_on_compiled(
            prog, PARITY_SOURCE, PARITY_MASK, PARITY_POS, PARITY_NEG, config,
            _compute_holdout=False,
            _reset_before_first_attempt=True,
        )
        results.append(r)

    # Both must converge (this stage always converges)
    for i, r in enumerate(results):
        assert r.converged, f"seed={i} did not converge: {r.discovered_rule}"

    # Verify seed 0 result matches a fresh train_only with seed 0
    fresh_0 = train_only(
        source=PARITY_SOURCE,
        mask_name=PARITY_MASK,
        positives=PARITY_POS,
        negatives=PARITY_NEG,
        config=_parity_config(0),
        _compute_holdout=False,
    )
    assert results[0].discovered_rule == fresh_0.discovered_rule, (
        f"State leak: reused seed=0 rule={results[0].discovered_rule!r} "
        f"vs fresh={fresh_0.discovered_rule!r}"
    )
