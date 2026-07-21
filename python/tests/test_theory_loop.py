"""Unit tests for `theory_loop.py` (task S5a) -- CPU, no pkl, no engine,
no torch. `induce_theory` is pure control logic wrapping two caller-supplied
closures (`select_once`, `predict_clause`); every fake here is plain Python,
following the style of `test_caviar_scorer.py`/`test_caviar_detector_probe.py`.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "caviar_woled"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from theory_loop import induce_theory  # noqa: E402

ITERATION_KEYS = {
    "rule", "reason", "margin", "n_residual_pos_before", "n_newly_covered",
}


@dataclass
class FakeSelection:
    """Stands in for `pyxlog.ilp.neural_credit.HoldoutSelection`: only
    `.rule` is required by `induce_theory`; `.margin` is read via
    `getattr(..., "margin", None)`, so tests both with and without it are
    meaningful."""
    rule: object
    margin: float = 0.0


def predict_membership(rule, fact):
    """A rule IS the frozenset of facts it covers -- the simplest possible
    `predict_clause`, used throughout this file."""
    return fact in rule


def make_incremental_selector(cover_size, margin=0.5):
    """A `select_once` that always proposes a rule covering exactly
    `cover_size` of the CURRENT residual's positives (the first `cover_size`
    it sees, in residual order) -- or fewer, honestly, if fewer than
    `cover_size` residual positives remain. Records every call's arguments
    (as copies, since `induce_theory` must never hand a closure a list it
    later mutates in place) on `.calls` for inspection."""
    calls = []

    def select_once(residual_facts, residual_is_positive):
        calls.append((list(residual_facts), list(residual_is_positive)))
        positives = [f for f, p in zip(residual_facts, residual_is_positive) if p]
        rule = frozenset(positives[:cover_size])
        return FakeSelection(rule=rule, margin=margin)

    select_once.calls = calls
    return select_once


# ---------------------------------------------------------------------------
# Stop condition 1: max_clauses reached
# ---------------------------------------------------------------------------


def test_stops_at_max_clauses_without_an_extra_select_once_call():
    facts = list(range(100))
    is_positive = [True] * 100
    select_once = make_incremental_selector(cover_size=3)

    result = induce_theory(
        select_once, predict_membership, facts, is_positive,
        max_clauses=4, min_new_covered=3,
    )

    assert result["stop_reason"] == "max_clauses reached"
    assert len(result["clauses"]) == 4
    assert len(result["iterations"]) == 4
    assert len(select_once.calls) == 4  # no 5th call spent past the cap
    assert all(it["reason"] == "committed" for it in result["iterations"])
    # Each commit covers exactly 3 new positives (cover_size=3), disjoint.
    assert [it["n_newly_covered"] for it in result["iterations"]] == [3, 3, 3, 3]
    assert [it["n_residual_pos_before"] for it in result["iterations"]] == [
        100, 97, 94, 91,
    ]


# ---------------------------------------------------------------------------
# Stop condition 2: no positives remain in the residual
# ---------------------------------------------------------------------------


def test_stops_when_no_positives_remain_without_a_second_select_once_call():
    facts = list(range(5))
    is_positive = [True] * 5
    select_once = make_incremental_selector(cover_size=5)  # covers everything at once

    result = induce_theory(
        select_once, predict_membership, facts, is_positive,
        max_clauses=10, min_new_covered=1,
    )

    assert result["stop_reason"] == "no positives remain in the residual"
    assert len(result["clauses"]) == 1
    assert len(result["iterations"]) == 1
    assert len(select_once.calls) == 1  # the "no positives left" check pre-empts call 2
    assert result["iterations"][0]["reason"] == "committed"
    assert result["iterations"][0]["n_newly_covered"] == 5


# ---------------------------------------------------------------------------
# Stop condition 3: select_once abstains
# ---------------------------------------------------------------------------


def test_stops_immediately_on_an_honest_abstain():
    def always_abstains(residual_facts, residual_is_positive):
        return FakeSelection(rule=None, margin=0.0)

    facts = [0, 1, 2]
    is_positive = [True, True, False]

    result = induce_theory(
        always_abstains, predict_membership, facts, is_positive,
        max_clauses=4, min_new_covered=1,
    )

    assert result["stop_reason"] == "select_once abstained"
    assert result["clauses"] == []
    assert len(result["iterations"]) == 1
    it = result["iterations"][0]
    assert it["rule"] is None
    assert it["reason"] == "select_once abstained"
    assert it["n_newly_covered"] == 0
    assert it["n_residual_pos_before"] == 2


def test_abstain_margin_falls_back_to_none_when_selection_has_no_margin_attribute():
    class BareSelection:
        def __init__(self):
            self.rule = None

    def bare_abstain(residual_facts, residual_is_positive):
        return BareSelection()

    result = induce_theory(
        bare_abstain, predict_membership, [0], [True], max_clauses=1, min_new_covered=1,
    )
    assert result["iterations"][0]["margin"] is None


# ---------------------------------------------------------------------------
# Stop condition 4: proposed clause rejected for insufficient new coverage
# ---------------------------------------------------------------------------


def test_rejects_and_stops_when_newly_covered_is_below_min_new_covered():
    facts = list(range(10))
    is_positive = [True] * 10
    # cover_size=2 < min_new_covered=3 -> every proposal is rejected, never committed.
    select_once = make_incremental_selector(cover_size=2)

    result = induce_theory(
        select_once, predict_membership, facts, is_positive,
        max_clauses=4, min_new_covered=3,
    )

    assert result["stop_reason"] == "insufficient new coverage"
    assert result["clauses"] == []  # rejected, never committed
    assert len(result["iterations"]) == 1
    it = result["iterations"][0]
    assert it["reason"] == "rejected: newly covered positives below min_new_covered"
    assert it["n_newly_covered"] == 2
    assert it["rule"] == frozenset({0, 1})


def test_a_clause_can_commit_then_a_later_one_can_be_rejected_and_stop():
    """First clause covers plenty (committed); once the residual shrinks
    below cover_size, the next proposal covers fewer than min_new_covered
    and the loop stops there -- clauses list keeps the earlier commit."""
    facts = list(range(7))
    is_positive = [True] * 7
    select_once = make_incremental_selector(cover_size=5)

    result = induce_theory(
        select_once, predict_membership, facts, is_positive,
        max_clauses=4, min_new_covered=3,
    )
    # iter 1: 7 residual positives, covers 5 -> committed, 2 remain.
    # iter 2: 2 residual positives, cover_size=5 clamps to 2 < min_new_covered=3 -> rejected.
    assert len(result["clauses"]) == 1
    assert result["stop_reason"] == "insufficient new coverage"
    assert len(result["iterations"]) == 2
    assert result["iterations"][0]["reason"] == "committed"
    assert result["iterations"][1]["reason"] == (
        "rejected: newly covered positives below min_new_covered"
    )
    assert result["iterations"][1]["n_newly_covered"] == 2


# ---------------------------------------------------------------------------
# Residual bookkeeping: only covered POSITIVES are removed; negatives, and
# uncovered positives, always survive into the next iteration's residual.
# ---------------------------------------------------------------------------


def test_residual_removes_only_covered_positives_negatives_always_survive():
    # facts 0,1,2,3 positive; 4,5 negative. Clause A covers {0, 1, 4} --
    # two real positives AND one negative (a false positive for A).
    facts = [0, 1, 2, 3, 4, 5]
    is_positive = [True, True, True, True, False, False]
    rule_a = frozenset({0, 1, 4})
    rule_b = frozenset({2, 3})

    calls = []

    def select_once(residual_facts, residual_is_positive):
        calls.append((list(residual_facts), list(residual_is_positive)))
        return FakeSelection(rule=rule_a if not calls[:-1] else rule_b, margin=0.9)

    result = induce_theory(
        select_once, predict_membership, facts, is_positive,
        max_clauses=4, min_new_covered=1,
    )

    # Call 2 must see residual = [2, 3, 4, 5] with is_positive [T, T, F, F]:
    # 0 and 1 (covered positives) are dropped; 4 (a covered NEGATIVE) stays;
    # 5 (never covered by anything) stays; 2, 3 (uncovered positives) stay.
    assert len(calls) == 2
    second_call_facts, second_call_pos = calls[1]
    assert second_call_facts == [2, 3, 4, 5]
    assert second_call_pos == [True, True, False, False]

    # Clause B then covers the two remaining positives -> no positives left.
    assert result["clauses"] == [rule_a, rule_b]
    assert result["stop_reason"] == "no positives remain in the residual"
    assert result["iterations"][0]["n_newly_covered"] == 2  # 0, 1 -- not 4
    assert result["iterations"][1]["n_newly_covered"] == 2  # 2, 3


def test_facts_and_is_positive_inputs_are_never_mutated_in_place():
    facts = [0, 1, 2]
    is_positive = [True, True, False]
    facts_copy, is_positive_copy = list(facts), list(is_positive)
    select_once = make_incremental_selector(cover_size=1)

    induce_theory(select_once, predict_membership, facts, is_positive, max_clauses=1, min_new_covered=1)

    assert facts == facts_copy
    assert is_positive == is_positive_copy


# ---------------------------------------------------------------------------
# Union scoring / iteration log shape
# ---------------------------------------------------------------------------


def test_every_iteration_entry_has_exactly_the_documented_keys():
    facts = list(range(6))
    is_positive = [True] * 6
    select_once = make_incremental_selector(cover_size=2)

    result = induce_theory(
        select_once, predict_membership, facts, is_positive,
        max_clauses=2, min_new_covered=1,
    )
    assert len(result["iterations"]) >= 1
    for it in result["iterations"]:
        assert set(it) == ITERATION_KEYS


def test_result_top_level_shape():
    facts = [0, 1]
    is_positive = [True, False]
    select_once = make_incremental_selector(cover_size=1)
    result = induce_theory(select_once, predict_membership, facts, is_positive)
    assert set(result) == {"clauses", "iterations", "stop_reason"}
    assert isinstance(result["clauses"], list)
    assert isinstance(result["iterations"], list)
    assert isinstance(result["stop_reason"], str)


def test_max_clauses_zero_stops_immediately_with_no_select_once_call():
    select_once = make_incremental_selector(cover_size=1)
    result = induce_theory(
        select_once, predict_membership, [0], [True], max_clauses=0, min_new_covered=1,
    )
    assert result == {
        "clauses": [], "iterations": [], "stop_reason": "max_clauses reached",
    }
    assert select_once.calls == []


def test_no_positives_at_all_stops_immediately_with_no_select_once_call():
    select_once = make_incremental_selector(cover_size=1)
    result = induce_theory(
        select_once, predict_membership, [0, 1], [False, False],
        max_clauses=4, min_new_covered=1,
    )
    assert result["stop_reason"] == "no positives remain in the residual"
    assert result["clauses"] == []
    assert result["iterations"] == []
    assert select_once.calls == []
