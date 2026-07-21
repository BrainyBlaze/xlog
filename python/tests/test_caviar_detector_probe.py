"""Unit tests for `detector_probe.py` (task S4a) -- CPU, no pkl, no engine,
no torch required (every case here uses plain Python lists; the module
itself never imports torch, so these tests would pass even without torch
installed -- verified by NOT importing torch anywhere in this file).

Hand-built tiny cases, following the style of `test_caviar_scorer.py`:
every number below is hand-computable and checked exactly.
"""
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "caviar_woled"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from detector_probe import (  # noqa: E402
    DIST_BIN_EDGES,
    assign_bin,
    bin_labels,
    monotone_decay_report,
    probe_detector,
)

# ---------------------------------------------------------------------------
# assign_bin / bin_labels
# ---------------------------------------------------------------------------


def test_bin_labels_are_the_eleven_documented_buckets():
    assert bin_labels() == [
        "0-5", "5-10", "10-15", "15-20", "20-25", "25-30",
        "30-35", "35-40", "40-45", "45-50", "50+",
    ]
    assert len(DIST_BIN_EDGES) == 11


@pytest.mark.parametrize(
    "dist, expected_bin",
    [
        (0.0, 0),
        (4.999, 0),
        (5.0, 1),
        (9.999, 1),
        (24.999, 4),
        (25.0, 5),          # CAVIAR's close_threshold boundary
        (29.999, 5),
        (49.999, 9),
        (50.0, 10),          # open-ended "50+" bin
        (1000.0, 10),
    ],
)
def test_assign_bin_boundaries_are_half_open_and_exact(dist, expected_bin):
    assert assign_bin(dist) == expected_bin


def test_assign_bin_refuses_a_negative_distance():
    with pytest.raises(ValueError, match="negative"):
        assign_bin(-1.0)


# ---------------------------------------------------------------------------
# probe_detector: hand-computed accuracy/prf1/bins
# ---------------------------------------------------------------------------

# pt0: score 0.9 (pred close), dist 2.0  (bin "0-5"),  gold close  -> tp
# pt1: score 0.1 (pred far),   dist 3.0  (bin "0-5"),  gold close  -> fn
# pt2: score 0.6 (pred close), dist 27.0 (bin "25-30"),gold far    -> fp
# pt3: score 0.4 (pred far),   dist 60.0 (bin "50+"),  gold far    -> tn
SCORES = [0.9, 0.1, 0.6, 0.4]
DISTS = [2.0, 3.0, 27.0, 60.0]
CLOSE_ROWS = {0, 1}


def test_probe_detector_hand_computed_accuracy_and_prf1():
    result = probe_detector(SCORES, CLOSE_ROWS, DISTS)

    assert result["num_rows"] == 4
    assert result["num_excluded"] == 0
    assert result["no_close_rows"] is False
    assert result["accuracy"] == pytest.approx(0.5)          # 2/4 correct
    assert result["prf1"]["tp"] == 1
    assert result["prf1"]["fp"] == 1
    assert result["prf1"]["fn"] == 1
    assert result["prf1"]["tn"] == 1
    assert result["prf1"]["precision"] == pytest.approx(0.5)
    assert result["prf1"]["recall"] == pytest.approx(0.5)
    assert result["prf1"]["f1"] == pytest.approx(0.5)
    assert result["prf1"]["degenerate"] is False


def test_probe_detector_hand_computed_bins():
    result = probe_detector(SCORES, CLOSE_ROWS, DISTS)
    by_label = {b["label"]: b for b in result["bins"]}

    assert by_label["0-5"]["count"] == 2
    assert by_label["0-5"]["mean_score"] == pytest.approx((0.9 + 0.1) / 2)
    assert by_label["25-30"]["count"] == 1
    assert by_label["25-30"]["mean_score"] == pytest.approx(0.6)
    assert by_label["50+"]["count"] == 1
    assert by_label["50+"]["mean_score"] == pytest.approx(0.4)
    # Untouched bins: count 0, mean_score None (never a NaN or a divide-by-zero).
    assert by_label["10-15"]["count"] == 0
    assert by_label["10-15"]["mean_score"] is None


def test_probe_detector_exclude_rows_drops_them_from_everything():
    """Excluding pt1 (coords_missing surrogate) removes it from BOTH the
    accuracy/prf1 computation and the bin table -- hand-recomputed with
    only pt0, pt2, pt3 remaining."""
    result = probe_detector(SCORES, CLOSE_ROWS, DISTS, exclude_rows={1})

    assert result["num_rows"] == 3
    assert result["num_excluded"] == 1
    # pt0: pred True gold True (tp); pt2: pred True gold False (fp);
    # pt3: pred False gold False (tn).
    assert result["prf1"]["tp"] == 1
    assert result["prf1"]["fp"] == 1
    assert result["prf1"]["fn"] == 0
    assert result["prf1"]["tn"] == 1
    assert result["accuracy"] == pytest.approx(2 / 3)
    by_label = {b["label"]: b for b in result["bins"]}
    assert by_label["0-5"]["count"] == 1              # only pt0 remains
    assert by_label["0-5"]["mean_score"] == pytest.approx(0.9)


def test_probe_detector_no_close_rows_is_flagged_not_crashed():
    """Degenerate case (per the brief): no ground-truth close rows at all
    must not crash -- flagged via `no_close_rows`, and `prf1`'s own
    `degenerate` flag fires (zero actual positives)."""
    scores = [0.9, 0.2]
    dists = [1.0, 40.0]
    result = probe_detector(scores, close_rows=set(), dists=dists)

    assert result["no_close_rows"] is True
    assert result["prf1"]["degenerate"] is True
    assert result["prf1"]["fn"] == 0
    assert result["prf1"]["tp"] == 0
    assert result["accuracy"] == pytest.approx(0.5)   # pt0 wrong, pt1 right


def test_probe_detector_refuses_zero_rows():
    with pytest.raises(ValueError, match="at least one row"):
        probe_detector([], set(), [])


def test_probe_detector_refuses_mismatched_scores_and_dists_length():
    with pytest.raises(ValueError, match="aligned"):
        probe_detector([0.1, 0.2], set(), [1.0])


def test_probe_detector_refuses_when_every_row_is_excluded():
    with pytest.raises(ValueError, match="excluded"):
        probe_detector([0.1], {0}, [1.0], exclude_rows={0})


# ---------------------------------------------------------------------------
# monotone_decay_report
# ---------------------------------------------------------------------------


def _bins(mean_scores_by_label: dict[str, float | None]) -> list[dict]:
    return [
        {"label": label, "count": (0 if v is None else 1), "mean_score": v}
        for label, v in mean_scores_by_label.items()
    ]


def test_monotone_decay_detects_a_clean_decreasing_sequence():
    bins = _bins({"0-5": 0.9, "5-10": 0.8, "10-15": None, "15-20": 0.5})
    report = monotone_decay_report(bins)

    assert report["monotone_non_increasing"] is True
    # Largest drop is between "5-10" (0.8) and "15-20" (0.5) -- the "10-15"
    # bin has no data and is skipped, not treated as a break in the sequence.
    assert report["knee_label"] == "5-10->15-20"
    assert report["knee_drop"] == pytest.approx(0.3)


def test_monotone_decay_flags_a_non_monotone_sequence():
    bins = _bins({"0-5": 0.5, "5-10": 0.9})
    report = monotone_decay_report(bins)

    assert report["monotone_non_increasing"] is False
    assert report["knee_label"] == "0-5->5-10"
    assert report["knee_drop"] == pytest.approx(-0.4)


def test_monotone_decay_with_fewer_than_two_populated_bins_is_undefined_not_crashed():
    bins = _bins({"0-5": 0.9, "5-10": None, "10-15": None})
    report = monotone_decay_report(bins)

    assert report["monotone_non_increasing"] is None
    assert report["knee_label"] is None
    assert report["knee_drop"] == 0.0
    assert "undefined" in report["reason"]

    # Zero populated bins is the same code path, also not a crash.
    empty_bins = _bins({"0-5": None, "5-10": None})
    report2 = monotone_decay_report(empty_bins)
    assert report2["monotone_non_increasing"] is None
