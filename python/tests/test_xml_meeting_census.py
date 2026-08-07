"""Unit tests for `xml_meeting_census.py` -- section G's diagnostic
census tool. CPU, no CUDA, hand-built two-video corpus: one video where
the reference clause covers part of the meeting positives, one where
meeting is annotated but the clause (and even `close` alone) never holds
on a positive frame -- the lb1gt/rffgt shape the census exists to expose.
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "caviar_woled"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import xml_meeting_census  # noqa: E402


def _video(name, activity_label, id1_xy_by_frame, meeting_frames, n_frames=4):
    tracked = set()
    activity = {}
    coords = {}
    timestamps = []
    for f in range(n_frames):
        t = f * 40
        timestamps.append(t)
        for pid in ("id1", "id2"):
            tracked.add((pid, t))
            activity[(pid, t)] = activity_label
        coords[("id1", t)] = id1_xy_by_frame[f]
        coords[("id2", t)] = (0, 0)  # id2 pinned at the origin
    holds = {
        "meeting": {
            (p1, p2, f * 40)
            for f in meeting_frames
            for p1, p2 in (("id1", "id2"), ("id2", "id1"))
        }
    } if meeting_frames else {}
    return {
        "video": name, "frame_ms": 40, "time_offset": 0,
        "timestamps": timestamps, "persons": ["id1", "id2"],
        "tracked": tracked, "activity": activity, "coords": coords,
        "holds": holds,
    }


def test_census_by_video_counts_clause_and_close_coverage_separately():
    # near.xml: both inactive, id1 at distance 10 (close) on frames 0-1 and
    # distance 100 (far) on frames 2-3; meeting annotated on frames 0-2.
    # -> 3 positives, clause covers 2 (frames 0-1), close covers 2.
    near = _video(
        "near.xml", "inactive",
        {0: (10, 0), 1: (10, 0), 2: (100, 0), 3: (100, 0)},
        meeting_frames=[0, 1, 2],
    )
    # apart.xml: meeting annotated while the pair is 60px apart and
    # WALKING -- neither the clause nor close ever holds on a positive.
    apart = _video(
        "apart.xml", "walking",
        {0: (60, 0), 1: (60, 0), 2: (60, 0), 3: (60, 0)},
        meeting_frames=[0, 1],
    )
    census = xml_meeting_census.census_by_video([near, apart])
    assert census == [
        {"video": "near.xml", "n_meeting_positive_pairframes": 3,
         "n_covered_by_clause": 2, "n_close_on_positives": 2},
        {"video": "apart.xml", "n_meeting_positive_pairframes": 2,
         "n_covered_by_clause": 0, "n_close_on_positives": 0},
    ]


def test_fold_scoring_scores_test_side_and_counts_train_side_coverage():
    near = _video(
        "near.xml", "inactive",
        {0: (10, 0), 1: (10, 0), 2: (100, 0), 3: (100, 0)},
        meeting_frames=[0, 1, 2],
    )
    apart = _video(
        "apart.xml", "walking",
        {0: (60, 0), 1: (60, 0), 2: (60, 0), 3: (60, 0)},
        meeting_frames=[0, 1],
    )
    # Fold 0 holds out near.xml; apart.xml trains.
    got = xml_meeting_census.fold_scoring([near, apart], [0, 1], 0)
    assert got["test_videos"] == ["near.xml"]
    s = got["clause_applied_to_test_fold"]
    # Clause fires on frames 0-1 (both positive): tp=2, fp=0, fn=1.
    assert (s["tp"], s["fp"], s["fn"]) == (2, 0, 1)
    assert s["precision"] == 1.0
    # Train side (apart.xml): clause never fires at all.
    assert got["train_side"] == {
        "n_positives_total": 2,
        "n_positives_covered_by_clause": 0,
        "n_negatives_covered_by_clause": 0,
    }


def test_main_records_canonical_close_threshold(tmp_path, monkeypatch):
    """Every number the census reports depends on the `close` threshold,
    so the artifact must record it -- the recording convention the CV
    artifacts follow (`close_threshold`, resolved through
    `CANONICAL_CLOSE_THRESHOLDS`)."""
    import json

    import caviar_xml_corpus
    import run_caviar_cv

    # A synthetic two-video corpus whose first video carries the wk1gt
    # stem `main` locates the wk fold by.
    wk = _video(
        "wk1gt.xml", "inactive",
        {0: (10, 0), 1: (10, 0), 2: (100, 0), 3: (100, 0)},
        meeting_frames=[0, 1, 2],
    )
    apart = _video(
        "apart.xml", "walking",
        {0: (60, 0), 1: (60, 0), 2: (60, 0), 3: (60, 0)},
        meeting_frames=[0, 1],
    )
    monkeypatch.setattr(caviar_xml_corpus, "load_xml_corpus", lambda d: [wk, apart])
    monkeypatch.setattr(
        run_caviar_cv, "_xml_family_fold_assignment",
        lambda videos, fluent, folds, seed: [0, 1],
    )
    out = tmp_path / "census.json"
    rc = xml_meeting_census.main(
        ["--xml-dir", "ignored", "--folds", "2", "--seed", "7", "--out", str(out)]
    )
    assert rc == 0
    result = json.loads(out.read_text())
    assert result["close_threshold"] == \
        caviar_xml_corpus.CANONICAL_CLOSE_THRESHOLDS["meeting"]
