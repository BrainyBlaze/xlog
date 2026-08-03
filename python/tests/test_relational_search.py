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
    holdout_fold_assignment,
    induce_relational_theory,
    kfold_scores,
    make_predict_clause,
    permutation_null_threshold,
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
# pyxlog.ilp.neural_credit's SHARED `holdout_fold_assignment` helper (both
# `kfold_scores` and `pyxlog.ilp.neural_credit.kfold_select` call the same
# function now -- see kfold_scores's own docstring).
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

    # The SAME shared function kfold_scores itself calls internally (see
    # pyxlog.ilp.neural_credit.holdout_fold_assignment) -- this is now a
    # same-function cross-check, not two independent reproductions.
    from pyxlog.ilp.neural_credit import holdout_fold_assignment

    fold_of = holdout_fold_assignment(len(facts), folds, seed)

    cover = covers[body]
    expected_sum = 0.0
    for fold in range(folds):
        held_ids = [i for i in range(len(facts)) if fold_of[i] == fold]
        correct = sum(1 for i in held_ids if (facts[i] in cover) == labels[i])
        expected_sum += correct / len(held_ids)
    expected = expected_sum / folds

    got = kfold_scores([body], relations, facts, labels, folds, seed)[body]
    assert got == pytest.approx(expected)


# ---------------------------------------------------------------------------
# kfold_scores(score="f1"): recall-aware per-fold F1, byte-identical
# accuracy default, zero-positive-fold skip rule, and the ranking-inversion
# regression this option exists to fix (see kfold_scores's own docstring for
# the score="f1" semantics being pinned here).
# ---------------------------------------------------------------------------


def test_kfold_scores_score_accuracy_explicit_matches_default():
    facts = [(i, 1) for i in range(12)]
    labels = [i % 4 == 0 for i in range(12)]
    relations = {
        "a": [facts[i] for i in [0, 4, 5, 8]],
        "b": [facts[i] for i in range(12)],
    }
    body = ("a", "b")
    default_scores = kfold_scores([body], relations, facts, labels, folds=3, seed=5)
    explicit_scores = kfold_scores(
        [body], relations, facts, labels, folds=3, seed=5, score="accuracy",
    )
    assert default_scores == explicit_scores


def test_kfold_scores_rejects_unknown_score_value():
    facts = [(i, 1) for i in range(4)]
    labels = [True, False, True, False]
    relations = {"a": facts, "b": facts}
    with pytest.raises(ValueError):
        kfold_scores([("a", "b")], relations, facts, labels, folds=2, seed=0, score="bogus")


def test_kfold_scores_f1_matches_manual_prf1_per_fold():
    # Independent recomputation (via scorer.prf1 directly, per fold, with
    # the SAME zero-positive-fold skip this function documents) cross-checked
    # against kfold_scores's own score="f1" output -- not just "some number
    # changed", but the EXACT arithmetic.
    from pyxlog.ilp.neural_credit import holdout_fold_assignment
    from scorer import prf1

    n_facts = 20
    folds = 4
    seed = 3
    facts = [(i, 1) for i in range(n_facts)]
    positive_idx = {0, 1, 2, 3, 10}
    labels = [i in positive_idx for i in range(n_facts)]

    # Body's cover: true positives at 0, 1, 10; false positives at 5, 15;
    # false negative at 2, 3 -- a genuinely imperfect detector, not a
    # perfect-fit toy.
    cover_idx = {0, 1, 5, 10, 15}
    relations = {
        "a": [facts[i] for i in cover_idx],
        "b": [facts[i] for i in range(n_facts)],
    }
    body = ("a", "b")
    covers = {body: body_cover(body, relations)}

    fold_of = holdout_fold_assignment(n_facts, folds, seed)
    per_fold_f1 = []
    for fold in range(folds):
        held = [i for i in range(n_facts) if fold_of[i] == fold]
        gold = [labels[i] for i in held]
        if not any(gold):
            continue  # zero-positive fold: recall undefined, skipped -- see docstring
        pred = [i in cover_idx for i in held]
        per_fold_f1.append(prf1(pred, gold)["f1"])
    expected = sum(per_fold_f1) / len(per_fold_f1)

    got = kfold_scores(
        [body], relations, facts, labels, folds, seed, covers=covers, score="f1",
    )[body]
    assert got == pytest.approx(expected)


def test_kfold_scores_f1_skips_a_zero_positive_fold_rather_than_scoring_it_zero():
    # Fold assignment discovered from the SAME shared function kfold_scores
    # itself calls (holdout_fold_assignment), not hard-coded -- then labels
    # are built so fold 0 holds ZERO positives (every fold-0 fact is
    # negative) while folds 1 and 2 each hold exactly one. A body whose
    # cover is EXACTLY the positive set has perfect precision/recall (F1
    # 1.0) on every fold that can measure it; the zero-positive fold 0
    # contributes NOTHING to the mean (documented skip, not a 0.0 tax).
    n_facts = 12
    folds = 3
    seed = 5
    facts = [(i, 1) for i in range(n_facts)]

    from pyxlog.ilp.neural_credit import holdout_fold_assignment

    fold_of = holdout_fold_assignment(n_facts, folds, seed)

    labels = [False] * n_facts
    for fold in (1, 2):
        held = sorted(i for i in range(n_facts) if fold_of[i] == fold)
        labels[held[0]] = True

    fold0_held = [i for i in range(n_facts) if fold_of[i] == 0]
    assert not any(labels[i] for i in fold0_held)  # sanity: fold 0 really is zero-positive

    positive_facts = [facts[i] for i in range(n_facts) if labels[i]]
    relations = {
        "a": positive_facts,
        "b": [facts[i] for i in range(n_facts)],
    }
    body = ("a", "b")
    covers = {body: body_cover(body, relations)}

    got = kfold_scores(
        [body], relations, facts, labels, folds, seed, covers=covers, score="f1",
    )[body]
    assert got == pytest.approx(1.0)


def test_ranking_inversion_accuracy_vs_f1_on_rare_positive_holdout():
    """THE POINT of score="f1" (see kfold_scores's own docstring and
    docs/experiments/caviar/README.md section D): on a rare-positive-class
    holdout, accuracy can rank an empty predictor ABOVE the best real
    detector. This reproduces exactly that inversion on a small, hand-built
    dataset and pins that score="f1" resolves it -- the whole reason this
    module gained a score parameter."""
    n_facts = 1000
    facts = [(i, 1) for i in range(n_facts)]
    positive_idx = set(range(10))  # 10 positives, indices 0..9
    labels = [i in positive_idx for i in range(n_facts)]

    # "real": a genuine (if imperfect) detector -- true positives at 0, 1, 2
    # (recall 3/10) plus 8 false positives elsewhere (indices 500..507):
    # 7 misses + 8 false positives = 15 mistakes.
    real_cover_idx = {0, 1, 2} | set(range(500, 508))
    # "empty": predicts nothing, ever -- misses all 10 positives, 0 false
    # positives = 10 mistakes (FEWER raw mistakes than "real", hence the
    # accuracy inversion below).
    relations = {
        "real_a": [facts[i] for i in real_cover_idx],
        "real_b": [facts[i] for i in range(n_facts)],  # superset: real_a & real_b == real_a
        "empty_a": [],
        "empty_b": [facts[i] for i in range(n_facts)],  # empty_a & empty_b == empty
    }
    real_body = ("real_a", "real_b")
    empty_body = ("empty_a", "empty_b")
    covers = {
        real_body: body_cover(real_body, relations),
        empty_body: body_cover(empty_body, relations),
    }
    assert covers[real_body] == {facts[i] for i in real_cover_idx}
    assert covers[empty_body] == set()

    acc = kfold_scores(
        [real_body, empty_body], relations, facts, labels, folds=4, seed=7,
        covers=covers, score="accuracy",
    )
    f1 = kfold_scores(
        [real_body, empty_body], relations, facts, labels, folds=4, seed=7,
        covers=covers, score="f1",
    )

    # THE INVERSION: under accuracy, the empty predictor outranks the real
    # detector.
    assert acc[empty_body] > acc[real_body]
    # THE FIX: under F1, the real detector outranks the empty predictor, and
    # the empty predictor's F1 is exactly 0 (degenerate: no predicted
    # positives on any fold).
    assert f1[real_body] > f1[empty_body]
    assert f1[empty_body] == pytest.approx(0.0)


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


def test_induce_relational_theory_forwards_holdout_score_to_kfold_scores(monkeypatch):
    # Direct plumbing check: induce_relational_theory's own select_once must
    # pass its holdout_score parameter straight through to every
    # kfold_scores call it makes (not silently default to "accuracy"
    # regardless of what the caller asked for).
    seen_scores = []
    real_kfold_scores = kfold_scores

    def spy(*args, **kwargs):
        seen_scores.append(kwargs.get("score"))
        return real_kfold_scores(*args, **kwargs)

    monkeypatch.setattr("relational_search.kfold_scores", spy)

    relations, facts, is_positive = _three_literal_world()
    induce_relational_theory(
        relations, facts, is_positive,
        max_literals=3, folds=4, seed=7, min_new_covered=2, holdout_score="f1",
    )
    assert seen_scores  # at least one select_once call happened
    assert all(s == "f1" for s in seen_scores)


def test_induce_relational_theory_default_holdout_score_is_accuracy(monkeypatch):
    seen_scores = []
    real_kfold_scores = kfold_scores

    def spy(*args, **kwargs):
        seen_scores.append(kwargs.get("score"))
        return real_kfold_scores(*args, **kwargs)

    monkeypatch.setattr("relational_search.kfold_scores", spy)

    relations, facts, is_positive = _three_literal_world()
    induce_relational_theory(
        relations, facts, is_positive,
        max_literals=3, folds=4, seed=7, min_new_covered=2,
    )
    assert seen_scores
    assert all(s == "accuracy" for s in seen_scores)


# ---------------------------------------------------------------------------
# permutation_null_threshold: exact arithmetic on a tiny hand-built case,
# permutation-invariance of the fold split/covers, and the quantile
# convention's own edge cases.
# ---------------------------------------------------------------------------


def test_permutation_null_threshold_exact_on_tiny_hand_built_case():
    # Independent reference (NOT relational_search's own set-intersection
    # shortcut): reshuffle the label array literally, per permutation, then
    # recompute each fold's F1 via scorer.prf1 directly -- the same
    # cross-check style test_kfold_scores_f1_matches_manual_prf1_per_fold
    # uses for kfold_scores itself.
    from pyxlog.ilp.neural_credit import holdout_fold_assignment
    from scorer import prf1

    n_facts = 12
    facts = [(i, 1) for i in range(n_facts)]
    labels = [i in {0, 4, 8} for i in range(n_facts)]  # 3 positives
    relations = {
        "a": [facts[i] for i in [0, 1, 4, 9]],
        "b": [facts[i] for i in range(n_facts)],
        "c": [facts[i] for i in [2, 4, 6, 8, 10]],
        "d": [facts[i] for i in range(n_facts)],
    }
    body1 = ("a", "b")
    body2 = ("c", "d")
    bodies = [body1, body2]
    folds = 3
    seed = 5
    n_permutations = 4
    quantile = 0.75
    perm_seed = 7

    covers = {body1: body_cover(body1, relations), body2: body_cover(body2, relations)}
    fold_of = holdout_fold_assignment(n_facts, folds, seed)

    rng = torch.Generator().manual_seed(perm_seed)
    pool_max_samples = []
    for _ in range(n_permutations):
        perm = torch.randperm(n_facts, generator=rng).tolist()
        labels_perm = [labels[perm[i]] for i in range(n_facts)]
        body_means = []
        for body in bodies:
            cover = covers[body]
            per_fold_f1 = []
            for fold in range(folds):
                held = [i for i in range(n_facts) if fold_of[i] == fold]
                gold = [labels_perm[i] for i in held]
                if not any(gold):
                    continue  # zero-positive fold under THIS permutation: skipped
                pred = [facts[i] in cover for i in held]
                per_fold_f1.append(prf1(pred, gold)["f1"])
            body_means.append(
                sum(per_fold_f1) / len(per_fold_f1) if per_fold_f1 else 0.0
            )
        pool_max_samples.append(max(body_means))

    sorted_samples = sorted(pool_max_samples)
    idx = quantile * (len(sorted_samples) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_samples) - 1)
    frac = idx - lo
    expected_threshold = (
        sorted_samples[lo] + (sorted_samples[hi] - sorted_samples[lo]) * frac
    )

    result = permutation_null_threshold(
        bodies, relations, facts, labels, folds=folds, seed=seed,
        n_permutations=n_permutations, quantile=quantile, perm_seed=perm_seed,
    )

    assert result["threshold"] == pytest.approx(expected_threshold)
    assert result["n_permutations"] == n_permutations
    assert result["quantile"] == quantile
    assert result["perm_seed"] == perm_seed
    assert result["pool_max_samples_summary"]["min"] == pytest.approx(min(pool_max_samples))
    assert result["pool_max_samples_summary"]["max"] == pytest.approx(max(pool_max_samples))


def test_permutation_null_threshold_computes_fold_split_and_covers_once(monkeypatch):
    # The fold split and every body's cover are permutation-INVARIANT (see
    # the function's own docstring) -- pin that they are derived exactly
    # once each, never once per permutation, by counting calls to the
    # shared helpers.
    calls = {"fold_assignment": 0, "body_cover": 0}
    real_fold_assignment = holdout_fold_assignment
    real_body_cover = body_cover

    def spy_fold_assignment(*a, **kw):
        calls["fold_assignment"] += 1
        return real_fold_assignment(*a, **kw)

    def spy_body_cover(*a, **kw):
        calls["body_cover"] += 1
        return real_body_cover(*a, **kw)

    monkeypatch.setattr("relational_search.holdout_fold_assignment", spy_fold_assignment)
    monkeypatch.setattr("relational_search.body_cover", spy_body_cover)

    relations, facts, is_positive = _three_literal_world()
    bodies, _ = enumerate_bodies(relations, max_literals=3)

    permutation_null_threshold(
        bodies, relations, facts, is_positive, folds=4, seed=7,
        n_permutations=50, quantile=0.95, perm_seed=7,
    )

    assert calls["fold_assignment"] == 1
    assert calls["body_cover"] == len(bodies)


def test_permutation_null_threshold_accepts_precomputed_covers_without_recomputing(monkeypatch):
    relations, facts, is_positive = _three_literal_world()
    bodies, _ = enumerate_bodies(relations, max_literals=3)
    covers = {b: body_cover(b, relations) for b in bodies}

    calls = {"body_cover": 0}
    real_body_cover = body_cover

    def spy(*a, **kw):
        calls["body_cover"] += 1
        return real_body_cover(*a, **kw)

    monkeypatch.setattr("relational_search.body_cover", spy)
    permutation_null_threshold(
        bodies, relations, facts, is_positive, folds=4, seed=7,
        n_permutations=10, quantile=0.95, perm_seed=7, covers=covers,
    )

    assert calls["body_cover"] == 0


def test_permutation_null_threshold_single_permutation_threshold_is_that_sample():
    relations, facts, is_positive = _three_literal_world()
    bodies, _ = enumerate_bodies(relations, max_literals=3)
    result = permutation_null_threshold(
        bodies, relations, facts, is_positive, folds=4, seed=7,
        n_permutations=1, quantile=0.95, perm_seed=3,
    )
    summary = result["pool_max_samples_summary"]
    assert summary["min"] == summary["max"] == summary["median"] == summary["p95"]
    assert result["threshold"] == pytest.approx(summary["max"])


def test_permutation_null_threshold_p95_summary_is_independent_of_quantile_param():
    # pool_max_samples_summary's own "p95" is ALWAYS the literal 95th
    # percentile, regardless of the caller's chosen `quantile` -- see the
    # function's own docstring.
    relations, facts, is_positive = _three_literal_world()
    bodies, _ = enumerate_bodies(relations, max_literals=3)
    at_default = permutation_null_threshold(
        bodies, relations, facts, is_positive, folds=4, seed=7,
        n_permutations=30, quantile=0.95, perm_seed=7,
    )
    at_custom = permutation_null_threshold(
        bodies, relations, facts, is_positive, folds=4, seed=7,
        n_permutations=30, quantile=0.5, perm_seed=7,
    )
    assert at_default["pool_max_samples_summary"] == at_custom["pool_max_samples_summary"]
    assert at_default["quantile"] == 0.95
    assert at_custom["quantile"] == 0.5
    # threshold tracks the CALLER's own quantile, unlike the fixed p95 summary
    assert at_default["threshold"] == pytest.approx(
        at_default["pool_max_samples_summary"]["p95"]
    )


def test_permutation_null_threshold_rejects_folds_out_of_range():
    facts = [(i, 1) for i in range(3)]
    labels = [True, False, True]
    relations = {"a": facts, "b": facts}
    with pytest.raises(ValueError):
        permutation_null_threshold(
            [("a", "b")], relations, facts, labels, folds=5, seed=0,
            n_permutations=2,
        )


def test_permutation_null_threshold_rejects_mismatched_labels_length():
    facts = [(i, 1) for i in range(4)]
    relations = {"a": facts, "b": facts}
    with pytest.raises(ValueError):
        permutation_null_threshold(
            [("a", "b")], relations, facts, [True, False], folds=2, seed=0,
            n_permutations=2,
        )


def test_permutation_null_threshold_rejects_non_positive_permutation_count():
    relations, facts, is_positive = _three_literal_world()
    bodies, _ = enumerate_bodies(relations, max_literals=3)
    with pytest.raises(ValueError):
        permutation_null_threshold(
            bodies, relations, facts, is_positive, folds=4, seed=7,
            n_permutations=0,
        )


def test_permutation_null_threshold_rejects_quantile_out_of_range():
    relations, facts, is_positive = _three_literal_world()
    bodies, _ = enumerate_bodies(relations, max_literals=3)
    with pytest.raises(ValueError):
        permutation_null_threshold(
            bodies, relations, facts, is_positive, folds=4, seed=7,
            n_permutations=5, quantile=0.0,
        )
    with pytest.raises(ValueError):
        permutation_null_threshold(
            bodies, relations, facts, is_positive, folds=4, seed=7,
            n_permutations=5, quantile=1.5,
        )
