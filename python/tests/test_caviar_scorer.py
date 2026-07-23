"""Unit tests for `scorer.py` -- CPU, no pkl, no engine.

Hand-built tiny `relations` dicts (the same `dict[str, list[(pt, 1)]]` shape
`caviar_convert.convert_split` produces) so every number below is hand-
computable, following the style of `test_caviar_convert.py`.
"""
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "caviar_woled"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from scorer import (  # noqa: E402
    DEFAULT_BASELINE_PAIRS,
    baseline_report,
    pr_curve,
    prf1,
    rule_predictions,
    theory_predictions,
)

# num_pt = 5 pair-times, hand-picked memberships:
#   both_active = {0, 1, 3}
#   close       = {0, 2, 3}
#   -> intersection (star cover) = {0, 3}
RELATIONS = {
    "both_active": [(0, 1), (1, 1), (3, 1)],
    "close": [(0, 1), (2, 1), (3, 1)],
    "far": [(1, 1), (4, 1)],
}
NUM_PT = 5


def test_rule_predictions_is_exact_set_intersection_of_the_two_relations():
    pred = rule_predictions("both_active", "close", RELATIONS, NUM_PT)
    assert pred == [True, False, False, True, False]


def test_rule_predictions_is_order_independent_in_the_two_names():
    a = rule_predictions("both_active", "close", RELATIONS, NUM_PT)
    b = rule_predictions("close", "both_active", RELATIONS, NUM_PT)
    assert a == b


def test_rule_predictions_missing_relation_name_raises_keyerror():
    with pytest.raises(KeyError):
        rule_predictions("both_active", "does_not_exist", RELATIONS, NUM_PT)


def test_prf1_hand_computed_case():
    # pred = star cover {0, 3} (from the test above); gold: meeting at {0, 1, 3}
    pred = [True, False, False, True, False]
    gold = [True, True, False, True, False]
    out = prf1(pred, gold)
    # tp: 0, 3 (both true+true) = 2; fp: none; fn: 1 (pred False, gold True); tn: 2, 4
    assert out["tp"] == 2
    assert out["fp"] == 0
    assert out["fn"] == 1
    assert out["tn"] == 2
    assert out["precision"] == pytest.approx(1.0)          # 2 / (2 + 0)
    assert out["recall"] == pytest.approx(2 / 3)            # 2 / (2 + 1)
    assert out["f1"] == pytest.approx(2 * 1.0 * (2 / 3) / (1.0 + 2 / 3))
    assert out["degenerate"] is False


def test_prf1_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        prf1([True, False], [True])


def test_prf1_degenerate_on_empty_gold_and_empty_pred():
    out = prf1([], [])
    assert out == {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
        "degenerate": True,
    }


def test_prf1_degenerate_when_no_predicted_positives():
    # pred is all False -> tp + fp == 0 -> precision forced to 0.0, degenerate
    out = prf1([False, False, False], [True, False, True])
    assert out["precision"] == 0.0
    assert out["degenerate"] is True


def test_prf1_degenerate_when_no_actual_positives():
    # gold is all False -> tp + fn == 0 -> recall forced to 0.0, degenerate
    out = prf1([True, False, True], [False, False, False])
    assert out["recall"] == 0.0
    assert out["degenerate"] is True


def test_prf1_not_degenerate_when_perfectly_separated():
    out = prf1([True, False], [True, False])
    assert out["precision"] == pytest.approx(1.0)
    assert out["recall"] == pytest.approx(1.0)
    assert out["f1"] == pytest.approx(1.0)
    assert out["degenerate"] is False


def test_baseline_report_default_pairs_plus_all_positive_are_all_present():
    gold = [True, True, False, True, False]
    out = baseline_report(RELATIONS, gold, NUM_PT, pairs=[("both_active", "close")])
    assert set(out) == {"both_active|close", "all_positive"}
    # both_active & close cover {0, 3} -- same as the hand-computed case above.
    assert out["both_active|close"]["tp"] == 2
    # all_positive predicts True everywhere: tp = count(gold), fp = count(not gold)
    assert out["all_positive"]["tp"] == 3
    assert out["all_positive"]["fp"] == 2
    assert out["all_positive"]["fn"] == 0
    assert out["all_positive"]["recall"] == pytest.approx(1.0)


def test_baseline_report_default_pairs_constant_matches_the_documented_defaults():
    assert DEFAULT_BASELINE_PAIRS == [
        ("both_active", "close"),
        ("both_inactive", "close"),
        ("mixed_active_walking", "close"),
    ]


def test_baseline_report_uses_default_pairs_when_none_given():
    relations = {
        "both_active": [(0, 1)],
        "both_inactive": [(1, 1)],
        "mixed_active_walking": [(2, 1)],
        "close": [(0, 1), (1, 1), (2, 1)],
    }
    gold = [True, False, True]
    out = baseline_report(relations, gold, num_pt=3)
    expected_keys = {f"{l}|{r}" for l, r in DEFAULT_BASELINE_PAIRS} | {"all_positive"}
    assert set(out) == expected_keys


# ---------------------------------------------------------------------------
# theory_predictions: union of clause predictions
# ---------------------------------------------------------------------------


def test_theory_predictions_is_the_exact_union_over_clauses():
    # 4 pair-times. Clause A covers {0, 1}; clause B covers {1, 2}.
    # Union: {0, 1, 2}; pt 3 covered by neither.
    covers = {"A": {0, 1}, "B": {1, 2}}

    def predict_clause(rule, fact):
        pt, label = fact
        assert label == 1  # star convention: fixed label column
        return pt in covers[rule]

    out = theory_predictions(["A", "B"], predict_clause, num_pt=4)
    assert out == [True, True, True, False]


def test_theory_predictions_empty_clause_list_is_all_false():
    def predict_clause(rule, fact):
        raise AssertionError("predict_clause must never be called for an empty theory")

    out = theory_predictions([], predict_clause, num_pt=3)
    assert out == [False, False, False]


def test_theory_predictions_single_clause_matches_that_clauses_own_predictions():
    covers = {"A": {0, 2}}

    def predict_clause(rule, fact):
        return fact[0] in covers[rule]

    out = theory_predictions(["A"], predict_clause, num_pt=3)
    assert out == [True, False, True]


# ---------------------------------------------------------------------------
# pr_curve: soft-scoring PR sweep
# ---------------------------------------------------------------------------


def test_pr_curve_exact_tiny_case():
    # scores=[0.9, 0.1], gold=[True, False], n_points=3 -> thresholds 0.0, 0.5, 1.0.
    scores = [0.9, 0.1]
    gold = [True, False]
    curve = pr_curve(scores, gold, n_points=3)

    assert [pt["threshold"] for pt in curve] == pytest.approx([0.0, 0.5, 1.0])

    # threshold 0.0: pred = [True, True] (0.9 > 0, 0.1 > 0) -> tp=1, fp=1, fn=0.
    assert curve[0]["precision"] == pytest.approx(0.5)
    assert curve[0]["recall"] == pytest.approx(1.0)
    assert curve[0]["f1"] == pytest.approx(2 * 0.5 * 1.0 / (0.5 + 1.0))

    # threshold 0.5: pred = [True, False] -- perfectly separated.
    assert curve[1]["precision"] == pytest.approx(1.0)
    assert curve[1]["recall"] == pytest.approx(1.0)
    assert curve[1]["f1"] == pytest.approx(1.0)

    # threshold 1.0: pred = [False, False] -- nothing predicted positive.
    assert curve[2]["precision"] == pytest.approx(0.0)
    assert curve[2]["recall"] == pytest.approx(0.0)
    assert curve[2]["f1"] == pytest.approx(0.0)


def test_pr_curve_recall_is_monotone_non_increasing_in_threshold():
    scores = [0.95, 0.8, 0.6, 0.4, 0.2, 0.05]
    gold = [True, True, False, True, False, False]
    curve = pr_curve(scores, gold, n_points=25)

    recalls = [pt["recall"] for pt in curve]
    assert all(recalls[i] >= recalls[i + 1] - 1e-12 for i in range(len(recalls) - 1))
    # Thresholds themselves are strictly increasing, evenly spaced over [0, 1].
    thresholds = [pt["threshold"] for pt in curve]
    assert thresholds[0] == pytest.approx(0.0)
    assert thresholds[-1] == pytest.approx(1.0)
    assert all(thresholds[i] < thresholds[i + 1] for i in range(len(thresholds) - 1))


def test_pr_curve_refuses_fewer_than_two_points():
    with pytest.raises(ValueError, match="n_points"):
        pr_curve([0.5], [True], n_points=1)
