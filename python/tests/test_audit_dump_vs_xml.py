"""Unit tests for `audit_dump_vs_xml.py` -- the reproducible form of
README section F's dump-vs-XML meeting-transition audit. CPU, no CUDA, no
real data: a hand-built three-video synthetic corpus exercises every
classification the audit exists to make -- a real event, a train/test
duplicate (the same source video shipped in both dump files), and a
splice phantom (a termination created only by the dump's segment-joining
rule bridging two different videos).
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "caviar_woled"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import audit_dump_vs_xml  # noqa: E402


# ---------------------------------------------------------------------------
# pair_transition_events -- the shared transition walk.
# ---------------------------------------------------------------------------


def test_pair_transition_events_first_frame_is_never_an_event():
    # Holds at the pair's very first co-visible frame: no earlier frame
    # exists to show a transition, so no init is emitted for it.
    events = audit_dump_vs_xml.pair_transition_events([0, 40, 80], {0, 40})
    assert events == [(80, "term")]


def test_pair_transition_events_init_and_term_at_the_flip_frames():
    events = audit_dump_vs_xml.pair_transition_events(
        [0, 40, 80, 120, 160], {40, 80},
    )
    assert events == [(40, "init"), (120, "term")]


# ---------------------------------------------------------------------------
# The synthetic corpus: three XML videos, three dump segments.
#
#   video a.xml: meeting frames 2..5  -> XML events init@2, term@6
#   video b.xml: meeting frames 6..9  -> XML event  init@6 (ends positive:
#                no termination inside the video's own frame sequence)
#   video c.xml: no meeting anywhere
#
#   dump train#0 = a.xml at offset 1000          -> init, term   (real)
#   dump train#1 = b.xml at 2000 SPLICED with c.xml at 2400 (the exactly-
#                  one-frame-step glue): the b-side meeting runs to b's
#                  last frame, so the dump's bridged timeline flips
#                  positive->negative at c's first frame -> a termination
#                  that exists in NEITHER video's own ground truth
#   dump test#0  = a.xml AGAIN at offset 5000    -> both events duplicated
# ---------------------------------------------------------------------------

_FRAMES = range(10)


def _xml_video(name, id1_base, id2_base, meeting_frames):
    tracked = set()
    activity = {}
    coords = {}
    timestamps = []
    for f in _FRAMES:
        t = f * 40
        timestamps.append(t)
        for pid, base in (("id1", id1_base), ("id2", id2_base)):
            tracked.add((pid, t))
            activity[(pid, t)] = "walking"
            coords[(pid, t)] = (base + f, base)
    holds = {}
    if meeting_frames:
        holds["meeting"] = {
            (p1, p2, f * 40)
            for f in meeting_frames
            for p1, p2 in (("id1", "id2"), ("id2", "id1"))
        }
    return {
        "video": name, "frame_ms": 40, "time_offset": 0,
        "timestamps": timestamps, "persons": ["id1", "id2"],
        "tracked": tracked, "activity": activity, "coords": coords,
        "holds": holds,
    }


def _dump_segment_from(videos_at_offsets, meeting_canonical_ts):
    """One `load_continuous`-shaped dump segment: each (video, offset)
    contributes its frames at t = frame*40 + offset -- more than one
    entry builds a SPLICED segment, exactly the dump defect under test."""
    timestamps = []
    activity = {}
    coords = {}
    for video, offset in videos_at_offsets:
        for f in _FRAMES:
            t = f * 40 + offset
            timestamps.append(t)
            for pid in ("id1", "id2"):
                activity[(pid, t)] = "walking"
                coords[(pid, t)] = video["coords"][(pid, f * 40)]
    return {
        "timestamps": sorted(timestamps),
        "persons": ["id1", "id2"],
        "activity": activity,
        "coords": coords,
        "meeting": {("id1", "id2", t) for t in meeting_canonical_ts},
    }


def _build_synthetic():
    video_a = _xml_video("a.xml", 10, 30, meeting_frames=range(2, 6))
    video_b = _xml_video("b.xml", 100, 130, meeting_frames=range(6, 10))
    video_c = _xml_video("c.xml", 200, 230, meeting_frames=())
    videos = [video_a, video_b, video_c]

    train_segments = [
        _dump_segment_from([(video_a, 1000)], {f * 40 + 1000 for f in range(2, 6)}),
        _dump_segment_from(
            [(video_b, 2000), (video_c, 2400)], {f * 40 + 2000 for f in range(6, 10)},
        ),
    ]
    test_segments = [
        _dump_segment_from([(video_a, 5000)], {f * 40 + 5000 for f in range(2, 6)}),
    ]
    return videos, train_segments, test_segments


def _run_audit():
    videos, train_segments, test_segments = _build_synthetic()
    traj, index = audit_dump_vs_xml.build_xml_xy_index(videos)
    seg_matches = {
        "train": audit_dump_vs_xml.match_segments(train_segments, videos, traj, index),
        "test": audit_dump_vs_xml.match_segments(test_segments, videos, traj, index),
    }
    dump_events = (
        audit_dump_vs_xml.dump_meeting_events(train_segments, "train")
        + audit_dump_vs_xml.dump_meeting_events(test_segments, "test")
    )
    xml_events = audit_dump_vs_xml.xml_meeting_events(videos)
    audit = audit_dump_vs_xml.classify_events(dump_events, xml_events, seg_matches, videos)
    return videos, seg_matches, dump_events, xml_events, audit


def test_event_derivation_counts_on_both_sides():
    _videos, _seg_matches, dump_events, xml_events, _audit = _run_audit()
    # Dump: 3 inits (train#0, train#1, test#0) + 3 terms (train#0, the
    # train#1 splice phantom, test#0). XML: a init/term + b init.
    assert sorted((e["kind"] for e in dump_events)) == ["init", "init", "init", "term", "term", "term"]
    assert sorted((e["kind"] for e in xml_events)) == ["init", "init", "term"]


def test_classification_real_duplicate_and_splice_phantom():
    _videos, _seg_matches, _dump_events, _xml_events, audit = _run_audit()
    assert audit["classification_counts"] == {"real": 3, "duplicate": 2, "splice_artifact": 1}
    # Every XML ground-truth event is claimed by some dump event -- the
    # dump misses nothing in this corpus, it only over-counts.
    assert audit["xml_events_unclaimed"] == []

    by_class = {}
    for r in audit["dump_events"]:
        by_class.setdefault(r["classification"], []).append(r)

    # The phantom is train#1's termination at the splice boundary: mapped
    # into c.xml's own first frame, where c's ground truth has no event.
    (phantom,) = by_class["splice_artifact"]
    assert phantom["seg_label"] == "train#1"
    assert phantom["kind"] == "term"
    assert phantom["t"] == 2400
    assert phantom["mapped"]["video"] == "c.xml"
    assert phantom["mapped"]["frame"] == 0

    # The duplicates are BOTH of test#0's events (train#0, visited first
    # under the deterministic train-before-test order, claimed a.xml's
    # real events).
    assert {(d["seg_label"], d["kind"]) for d in by_class["duplicate"]} == {
        ("test#0", "init"), ("test#0", "term"),
    }
    assert {(r["seg_label"], r["kind"]) for r in by_class["real"]} == {
        ("train#0", "init"), ("train#0", "term"), ("train#1", "init"),
    }


def test_duplicated_gold_pairframes_counts_the_double_shipped_video():
    videos, seg_matches, _dump_events, _xml_events, _audit = _run_audit()
    got = audit_dump_vs_xml.duplicated_gold_pairframes(videos, seg_matches)
    # Only a.xml is matched from both dump files; its gold meeting mass is
    # 4 unordered pair-frames (frames 2..5).
    assert got == {"a.xml": 4}


def test_splice_resolves_into_two_clusters_for_the_same_segment():
    _videos, seg_matches, _dump_events, _xml_events, _audit = _run_audit()
    spliced = seg_matches["train"][1]["clusters"]
    matched_videos = {v_idx for (v_idx, _offset) in spliced}
    assert matched_videos == {1, 2}  # b.xml AND c.xml -- the splice is visible
