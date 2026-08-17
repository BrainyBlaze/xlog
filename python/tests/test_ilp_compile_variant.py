"""Tests for CompiledIlpProgram.compile_variant() parity with a fresh compile."""

import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda

skip_unless_pyxlog_cuda()

from pyxlog.ilp.holdout import _commit_rule


SOURCE = """
    edge(1, 2). edge(2, 3). edge(3, 4). edge(4, 5). edge(5, 6).
    learnable(W_reach) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""
RULE = "reach(X, Y) :- edge(X, Z), edge(Z, Y)."


def _facts(prog, rel):
    return sorted(tuple(f) for f in prog.relation_facts(rel))


def _compile(source):
    return pyxlog.IlpProgramFactory.compile(source, device=0, memory_mb=512)


def test_variant_matches_fresh_compile():
    """A variant must derive exactly what a fresh compile of the same source does."""
    trial_source = _commit_rule(SOURCE, "W_reach", RULE)

    fresh = _compile(trial_source)
    base = _compile(SOURCE)
    variant = base.compile_variant(trial_source)

    assert _facts(variant, "edge") == _facts(fresh, "edge")
    assert _facts(variant, "reach") == _facts(fresh, "reach")
    assert _facts(variant, "reach") == [(1, 3), (2, 4), (3, 5), (4, 6)]
    for r in [(1, 3), (2, 4), (3, 5), (4, 6)]:
        assert variant.fact_exists("reach", list(r))
    assert not variant.fact_exists("reach", [1, 2])


def test_variant_reuses_facts_and_skips_provider():
    """The variant's timing must show no provider phase; a fresh compile must."""
    base = _compile(SOURCE)
    assert "provider" in base.compile_timing_ms()
    variant = base.compile_variant(_commit_rule(SOURCE, "W_reach", RULE))
    timing = variant.compile_timing_ms()
    assert "provider" not in timing
    for key in ("frontend", "facts", "edb_snapshot", "execute"):
        assert key in timing


def test_variant_is_independent_of_base():
    """Mutating the variant's store must not leak into the base and vice versa."""
    base = _compile(SOURCE)
    variant = base.compile_variant(_commit_rule(SOURCE, "W_reach", RULE))
    base_reach_before = _facts(base, "reach")

    variant.evaluate()  # re-executes the variant from its own facts
    assert _facts(variant, "reach") == [(1, 3), (2, 4), (3, 5), (4, 6)]
    assert _facts(base, "reach") == base_reach_before

    base.reset_runtime()
    assert _facts(variant, "reach") == [(1, 3), (2, 4), (3, 5), (4, 6)]


def test_variant_with_changed_facts_falls_back_to_reload():
    """Facts that differ from the base are loaded fresh, not taken from the snapshot."""
    base = _compile(SOURCE)
    changed = _commit_rule(SOURCE + "\n    edge(6, 7).\n", "W_reach", RULE)
    variant = base.compile_variant(changed)
    fresh = _compile(changed)
    assert _facts(variant, "edge") == _facts(fresh, "edge")
    assert _facts(variant, "reach") == _facts(fresh, "reach")
    assert (5, 7) in _facts(variant, "reach")


def test_variant_of_variant():
    """A variant can itself serve as a base (its own snapshot is taken)."""
    base = _compile(SOURCE)
    v1 = base.compile_variant(_commit_rule(SOURCE, "W_reach", RULE))
    v2 = v1.compile_variant(_commit_rule(SOURCE, "W_reach", RULE))
    assert _facts(v2, "reach") == _facts(v1, "reach")


def test_evaluate_after_fresh_compile_is_a_noop():
    """compile() already executes the plan; evaluate() on a fresh program changes nothing."""
    trial_source = _commit_rule(SOURCE, "W_reach", RULE)
    a = _compile(trial_source)
    reach_a = _facts(a, "reach")
    b = _compile(trial_source)
    b.evaluate()
    assert _facts(b, "reach") == reach_a
    v = _compile(SOURCE).compile_variant(trial_source)
    reach_v = _facts(v, "reach")
    v.evaluate()
    assert _facts(v, "reach") == reach_v == reach_a
