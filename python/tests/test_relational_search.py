"""Unit tests for `relational_search.py` -- CPU, no CUDA, no `pyxlog` engine
(only `torch` on CPU and `pyxlog.ilp.discovery.select_rule`, both plain
Python/CPU-only imports). Follows the style of `test_theory_loop.py`
(hand-built fakes) and `test_caviar_convert.py` (hand-computable fixtures).
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "caviar_woled"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from relational_search import (  # noqa: E402
    body_cover,
    enumerate_bodies,
    induce_relational_theory,
    kfold_scores,
    make_predict_clause,
    select_body,
)

# ---------------------------------------------------------------------------
# enumerate_bodies: counts, canonical order, empty-cover skip accounting
# ---------------------------------------------------------------------------

# a, b, c overlap pairwise and as a triple; d is disjoint from everything, so
# every combination touching d has an empty cover.
_ENUM_RELATIONS = {
    "a": [(0, 1), (1, 1), (2, 1)],
    "b": [(1, 1), (2, 1), (3, 1)],
    "c": [(2, 1), (3, 1), (4, 1)],
    "d": [(9, 1)],
}


def test_enumerate_bodies_counts_2_and_3_literal_with_skips():
    bodies, skipped = enumerate_bodies(_ENUM_RELATIONS, max_literals=3)

    # C(4,2) = 6 total 2-literal combinations: ab, ac, bc survive (nonempty
    # cover); ad, bd, cd are empty (d is disjoint from a/b/c).
    assert skipped[2] == 3
    # C(4,3) = 4 total 3-literal combinations: abc survives; abd, acd, bcd
    # all include d and are empty.
    assert skipped[3] == 3

    assert set(bodies) == {
        ("a", "b"), ("a", "c"), ("b", "c"), ("a", "b", "c"),
    }
    # Canonical order: every body's own literals are sorted, no mirrors.
    for body in bodies:
        assert tuple(sorted(body)) == body
    # No duplicates.
    assert len(bodies) == len(set(bodies))


def test_enumerate_bodies_max_literals_2_only_returns_pairs():
    bodies, skipped = enumerate_bodies(_ENUM_RELATIONS, max_literals=2)
    assert all(len(b) == 2 for b in bodies)
    assert skipped == {2: 3}


def test_enumerate_bodies_rejects_bad_max_literals():
    with pytest.raises(ValueError):
        enumerate_bodies(_ENUM_RELATIONS, max_literals=4)
    with pytest.raises(ValueError):
        enumerate_bodies(_ENUM_RELATIONS, max_literals=1)


# ---------------------------------------------------------------------------
# body_cover: exact set intersection
# ---------------------------------------------------------------------------


def test_body_cover_is_exact_intersection():
    assert body_cover(("a", "b"), _ENUM_RELATIONS) == {(1, 1), (2, 1)}
    assert body_cover(("a", "b", "c"), _ENUM_RELATIONS) == {(2, 1)}
    assert body_cover(("a", "d"), _ENUM_RELATIONS) == set()


def test_make_predict_clause_matches_body_cover():
    predict = make_predict_clause(_ENUM_RELATIONS)
    cover = body_cover(("a", "b", "c"), _ENUM_RELATIONS)
    for pt in range(10):
        fact = (pt, 1)
        assert predict(("a", "b", "c"), fact) == (fact in cover)
    # 2-literal rule: predict behaves identically to the existing
    # run_caviar_theory._predict_clause_relational reading.
    cover_ab = body_cover(("a", "b"), _ENUM_RELATIONS)
    for pt in range(10):
        fact = (pt, 1)
        assert predict(("a", "b"), fact) == (fact in cover_ab)


# ---------------------------------------------------------------------------
# kfold_scores: fold-assignment convention cross-check against
# pyxlog.ilp.neural_credit.kfold_select's own inlined derivation (reproduced
# here verbatim, since it is not a separately importable helper -- see
# kfold_scores's own docstring).
# ---------------------------------------------------------------------------


def test_kfold_scores_fold_assignment_matches_neural_credit_convention():
    facts = [(i, 1) for i in range(12)]
    labels = [i % 4 == 0 for i in range(12)]  # positives at 0, 4, 8
    seed = 5
    folds = 3

    # A body whose cover gets exactly ONE fact wrong (index 5, a false
    # positive) so the per-fold-averaged score is sensitive to exactly which
    # fold that fact lands in -- a strong cross-check of the fold grouping,
    # not just its cardinality.
    relations = {
        "a": [facts[i] for i in [0, 4, 5, 8]],
        "b": [facts[i] for i in range(12)],  # superset: a & b == a
    }
    body = ("a", "b")
    covers = {body: body_cover(body, relations)}

    # Reproduce kfold_select's own fold-assignment lines verbatim (see
    # pyxlog/ilp/neural_credit.py's kfold_select).
    rng = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(facts), generator=rng).tolist()
    fold_of = {f_idx: i % folds for i, f_idx in enumerate(order)}

    cover = covers[body]
    expected_sum = 0.0
    for fold in range(folds):
        held_ids = [i for i in range(len(facts)) if fold_of[i] == fold]
        correct = sum(1 for i in held_ids if (facts[i] in cover) == labels[i])
        expected_sum += correct / len(held_ids)
    expected = expected_sum / folds

    got = kfold_scores([body], relations, facts, labels, folds, seed)[body]
    assert got == pytest.approx(expected)


def test_kfold_scores_rejects_folds_out_of_range():
    facts = [(i, 1) for i in range(3)]
    labels = [True, False, True]
    relations = {"a": facts, "b": facts}
    with pytest.raises(ValueError):
        kfold_scores([("a", "b")], relations, facts, labels, folds=5, seed=0)


def test_kfold_scores_rejects_mismatched_labels_length():
    facts = [(i, 1) for i in range(4)]
    relations = {"a": facts, "b": facts}
    with pytest.raises(ValueError):
        kfold_scores([("a", "b")], relations, facts, [True, False], folds=2, seed=0)


# ---------------------------------------------------------------------------
# select_body: fit gate, tie/abstain, Occam narrowing
# ---------------------------------------------------------------------------


def test_select_body_fit_gate_abstains_when_nothing_reaches_min_fit():
    scores = {("a", "b"): 0.5, ("a", "c"): 0.6}
    covers = {("a", "b"): {1}, ("a", "c"): {2}}
    sel = select_body(scores, covers, min_fit=0.75, tie_tolerance=0.01)
    assert sel.rule is None
    assert "fit gate" in sel.reason


def test_select_body_picks_the_clear_winner():
    scores = {("a", "b"): 0.95, ("a", "c"): 0.80}
    covers = {("a", "b"): {1, 2}, ("a", "c"): {3, 4}}
    sel = select_body(scores, covers, min_fit=0.75, tie_tolerance=0.01)
    assert sel.rule == ("a", "b")
    assert sel.decided


def test_select_body_abstains_on_a_tie_with_genuinely_different_covers():
    scores = {("a", "b"): 0.90, ("p", "q"): 0.90}
    covers = {("a", "b"): {1}, ("p", "q"): {2}}
    sel = select_body(scores, covers, min_fit=0.75, tie_tolerance=0.05)
    assert sel.rule is None
    assert "genuinely different covers" in sel.reason


def test_select_body_occam_narrows_a_tie_with_identical_covers():
    # ("a", "b") and ("a", "b", "z") predict the SAME fact set on this data
    # (z is redundant here) -- Occam should keep the SHORTER body.
    scores = {("a", "b"): 0.90, ("a", "b", "z"): 0.90}
    identical_cover = {1, 2, 3}
    covers = {("a", "b"): identical_cover, ("a", "b", "z"): identical_cover}
    sel = select_body(scores, covers, min_fit=0.75, tie_tolerance=0.05)
    assert sel.rule == ("a", "b")
    assert "Occam" in sel.reason
    assert "IDENTICAL" in sel.reason


def test_select_body_occam_lexicographic_tiebreak_among_equal_length_identical_covers():
    identical_cover = {1, 2}
    scores = {("a", "z"): 0.90, ("a", "b"): 0.90}
    covers = {("a", "z"): identical_cover, ("a", "b"): identical_cover}
    sel = select_body(scores, covers, min_fit=0.75, tie_tolerance=0.05)
    assert sel.rule == ("a", "b")  # lexicographically first among equal length


def test_select_body_rejects_ampersand_in_relation_names():
    scores = {("a&b", "c"): 0.9}
    covers = {("a&b", "c"): {1}}
    with pytest.raises(ValueError):
        select_body(scores, covers, min_fit=0.75, tie_tolerance=0.01)


def test_select_body_rejects_non_positive_tie_tolerance():
    scores = {("a", "b"): 0.9}
    covers = {("a", "b"): {1}}
    with pytest.raises(ValueError):
        select_body(scores, covers, min_fit=0.75, tie_tolerance=0.0)


# ---------------------------------------------------------------------------
# induce_relational_theory: end-to-end on a synthetic dataset with a KNOWN
# 3-literal rule that NO 2-literal body can reproduce -- the search must
# find it (with max_literals=3) and must NOT be able to with max_literals=2
# (the exact expressiveness gap this module exists to close).
#
# Construction: T = {0..9} (10 positives). A = T | X | Y, B = T | X | Z,
# C = T | Y | Z, with X, Y, Z pairwise-disjoint 10-element chunks disjoint
# from T. Then A&B&C == T exactly (perfect fit), while every PAIR (A&B,
# A&C, B&C) covers T plus a different 10-element false-positive chunk --
# same accuracy, but a genuinely DIFFERENT cover each, so a 2-literal-only
# pool cannot even narrow the tie via Occam.
# ---------------------------------------------------------------------------


def _three_literal_world():
    facts = [(pt, 1) for pt in range(60)]
    is_positive = [pt < 10 for pt in range(60)]
    relations = {
        "A": [(pt, 1) for pt in range(30)],                              # T | X | Y
        "B": [(pt, 1) for pt in list(range(20)) + list(range(30, 40))],  # T | X | Z
        "C": [(pt, 1) for pt in list(range(10)) + list(range(20, 40))],  # T | Y | Z
    }
    return relations, facts, is_positive


def test_induce_relational_theory_finds_the_3_literal_rule():
    relations, facts, is_positive = _three_literal_world()
    result = induce_relational_theory(
        relations, facts, is_positive,
        max_literals=3, folds=4, seed=7, min_new_covered=2,
    )
    assert result["clauses"] == [("A", "B", "C")]
    assert result["stop_reason"] == "no positives remain in the residual"
    assert result["iterations"][0]["reason"] == "committed"
    assert result["iterations"][0]["n_newly_covered"] == 10
    assert result["pool"]["bodies_by_size"] == {2: 3, 3: 1}
    assert result["pool"]["skipped_empty_cover"] == {2: 0, 3: 0}


def test_induce_relational_theory_cannot_reach_the_rule_with_2_literals_only():
    relations, facts, is_positive = _three_literal_world()
    result = induce_relational_theory(
        relations, facts, is_positive,
        max_literals=2, folds=4, seed=7, min_new_covered=2,
    )
    # The three 2-literal candidates (A&B, A&C, B&C) tie at equal accuracy
    # with genuinely DIFFERENT covers -- an honest abstain, not a guess.
    assert result["clauses"] == []
    assert result["stop_reason"] == "select_once abstained"


def test_induce_relational_theory_rejects_bad_max_literals():
    relations, facts, is_positive = _three_literal_world()
    with pytest.raises(ValueError):
        induce_relational_theory(relations, facts, is_positive, max_literals=5)
