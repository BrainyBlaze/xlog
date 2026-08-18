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
EXPECTED_REACH = [(1, 3), (2, 4), (3, 5), (4, 6)]


def _facts(prog, rel):
    return sorted(tuple(f) for f in prog.relation_facts(rel))


def _compile(source, **kwargs):
    return pyxlog.IlpProgramFactory.compile(source, device=0, memory_mb=512, **kwargs)


def test_variant_matches_fresh_compile():
    """A variant must derive exactly what a fresh compile of the same source does."""
    trial_source = _commit_rule(SOURCE, "W_reach", RULE)

    fresh = _compile(trial_source)
    base = _compile(SOURCE)
    variant = base.compile_variant(trial_source)

    assert _facts(variant, "edge") == _facts(fresh, "edge")
    assert _facts(variant, "reach") == _facts(fresh, "reach") == EXPECTED_REACH
    for r in EXPECTED_REACH:
        assert variant.fact_exists("reach", list(r))
    assert not variant.fact_exists("reach", [1, 2])


def test_variant_skips_the_provider_phase():
    """A fresh compile pays for a CUDA provider; a variant does not."""
    base = _compile(SOURCE)
    assert base.compile_timing_ms()["provider"] > 0.0
    variant = base.compile_variant(_commit_rule(SOURCE, "W_reach", RULE))
    timing = variant.compile_timing_ms()
    assert "provider" not in timing
    for key in ("frontend", "facts", "execute"):
        assert key in timing


def test_variant_is_independent_of_base():
    """Mutating the variant's store must not leak into the base and vice versa."""
    base = _compile(SOURCE)
    variant = base.compile_variant(_commit_rule(SOURCE, "W_reach", RULE))
    base_reach_before = _facts(base, "reach")

    variant.evaluate()  # re-executes the variant from its own facts
    assert _facts(variant, "reach") == EXPECTED_REACH
    assert _facts(base, "reach") == base_reach_before

    base.reset_runtime()
    assert _facts(variant, "reach") == EXPECTED_REACH


def test_variant_with_changed_facts():
    """A variant may add facts; it loads its own source, not the base's."""
    base = _compile(SOURCE)
    changed = _commit_rule(SOURCE + "\n    edge(6, 7).\n", "W_reach", RULE)
    variant = base.compile_variant(changed)
    fresh = _compile(changed)
    assert _facts(variant, "edge") == _facts(fresh, "edge")
    assert _facts(variant, "reach") == _facts(fresh, "reach")
    assert (5, 7) in _facts(variant, "reach")


def test_variant_with_fewer_facts():
    """Dropping a fact in the variant must drop the derivations that needed it."""
    base = _compile(SOURCE)
    fewer = _commit_rule(SOURCE.replace("edge(5, 6). ", "").replace("edge(5, 6).", ""), "W_reach", RULE)
    variant = base.compile_variant(fewer)
    assert (4, 6) not in _facts(variant, "reach")
    assert _facts(variant, "reach") == _facts(_compile(fewer), "reach")


def test_variant_with_changed_schema():
    """A variant may change a predicate's arity; it must match a fresh compile."""
    base = _compile(SOURCE)
    wider = """
    edge(1, 2, 9). edge(2, 3, 9). edge(3, 4, 9).
    reach(X, Y) :- edge(X, Z, _), edge(Z, Y, _).
    """
    variant = base.compile_variant(wider)
    fresh = _compile(wider)
    assert _facts(variant, "edge") == _facts(fresh, "edge")
    assert _facts(variant, "reach") == _facts(fresh, "reach") == [(1, 3), (2, 4)]


def test_variant_with_predicate_that_is_both_fact_and_derived():
    """A predicate carrying both base facts and rule output stays consistent."""
    src = """
    edge(1, 2). edge(2, 3). edge(3, 4).
    reach(9, 9).
    learnable(W_r) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """
    trial = _commit_rule(src, "W_r", RULE)
    base = _compile(src)
    variant = base.compile_variant(trial)
    fresh = _compile(trial)
    assert _facts(variant, "reach") == _facts(fresh, "reach")
    assert (9, 9) in _facts(variant, "reach")
    assert _facts(variant, "reach").count((9, 9)) == 1
    variant.evaluate()
    assert _facts(variant, "reach") == _facts(fresh, "reach")


def test_variant_of_variant():
    """A variant can itself serve as a base."""
    base = _compile(SOURCE)
    v1 = base.compile_variant(_commit_rule(SOURCE, "W_reach", RULE))
    v2 = v1.compile_variant(_commit_rule(SOURCE, "W_reach", RULE))
    assert _facts(v2, "reach") == _facts(v1, "reach") == EXPECTED_REACH


def test_variant_after_commit_induced_rule_on_base():
    """A base mutated by commit_induced_rule still produces correct variants."""
    base = _compile(SOURCE)
    base.commit_induced_rule("edge(6, 7).\n" + RULE)
    trial_source = _commit_rule(SOURCE + "\n    edge(6, 7).\n", "W_reach", RULE)
    variant = base.compile_variant(trial_source)
    assert (5, 7) in _facts(variant, "reach")
    assert _facts(variant, "reach") == _facts(_compile(trial_source), "reach")


def test_variant_of_a_base_with_non_default_max_active_rules():
    """A variant compiles with the base program's max_active_rules and stays correct."""
    trial_source = _commit_rule(SOURCE, "W_reach", RULE)
    base = _compile(SOURCE, max_active_rules=64)
    variant = base.compile_variant(trial_source)
    fresh = _compile(trial_source, max_active_rules=64)
    assert _facts(variant, "reach") == _facts(fresh, "reach") == EXPECTED_REACH


def test_invalid_source_raises_value_error():
    """The frontend runs before the provider, so a parse error is a ValueError."""
    base = _compile(SOURCE)
    with pytest.raises(ValueError):
        base.compile_variant("this is not xlog(")
    with pytest.raises(ValueError):
        _compile("this is not xlog(")
    # The base survives a failed variant.
    assert _facts(base, "edge") == [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]


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
