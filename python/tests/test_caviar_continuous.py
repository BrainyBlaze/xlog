"""Unit tests for `caviar_continuous.py` -- CPU, no real data file. Every
fixture below is a small hand-built line-JSON string modeling the real
`caviar-train.json`/`caviar-test.json` shape (one JSON doc per line, keys
`_id`/`time`/`narrative`/`annotation`, atoms exactly as the real dump spells
them), following the style of `test_caviar_convert.py`/`test_caviar_ec.py`.
"""
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "caviar_woled"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from caviar_continuous import (  # noqa: E402
    TRANSITION_RELATION_NAMES,
    convert_continuous,
    derive_ec_masks_continuous,
    derive_ec_targets_continuous,
    load_continuous,
    reconstruct_holds_continuous,
)
from caviar_convert import convert_split  # noqa: E402


def _write_docs(path: Path, docs: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")


# ---------------------------------------------------------------------------
# Fixture: two video segments (a 40ms jump splices them), doc-level
# duplication (doc 0 repeated verbatim as doc 1) and a shared boundary frame
# (doc 2 re-states t=40, already present in doc 0/1) -- the two dedup quirks
# `load_continuous` must fold away. Segment 1 spans t=0..120 (4 frames);
# segment 2 spans t=1000..1040 (2 frames, after the jump).
# ---------------------------------------------------------------------------

def _two_segment_docs():
    doc_a = {
        "_id": 0, "time": "0",
        "narrative": [
            "happensAt(active(id0),0)", "coords(id0,0,0,0)",
            "happensAt(active(id1),0)", "coords(id1,3,4,0)",
            "happensAt(active(id0),40)", "coords(id0,0,0,40)",
            "happensAt(active(id1),40)", "coords(id1,3,4,40)",
        ],
        "annotation": [
            "holdsAt(meeting(id0,id1),40)", "holdsAt(meeting(id1,id0),40)",
        ],
    }
    doc_a_dup = dict(doc_a, _id=1)  # byte-identical duplicate, different _id
    doc_b = {
        "_id": 2, "time": "40",
        "narrative": [
            "happensAt(active(id0),40)", "coords(id0,0,0,40)",       # shared boundary frame
            "happensAt(active(id1),40)", "coords(id1,3,4,40)",
            "happensAt(inactive(id0),80)", "coords(id0,0,0,80)",
            "happensAt(walking(id1),80)", "coords(id1,100,0,80)",
            "happensAt(inactive(id0),120)", "coords(id0,0,0,120)",
            "happensAt(walking(id1),120)", "coords(id1,100,0,120)",
        ],
        "annotation": [
            "holdsAt(meeting(id0,id1),40)", "holdsAt(meeting(id1,id0),40)",
        ],
    }
    doc_c = {
        "_id": 3, "time": "1000",  # non-40ms jump from 120 -> 1000: new segment
        "narrative": [
            "happensAt(active(id0),1000)", "coords(id0,5,5,1000)",
            "happensAt(active(id1),1000)", "coords(id1,5,5,1000)",
            "happensAt(active(id0),1040)", "coords(id0,5,5,1040)",
            "happensAt(active(id1),1040)", "coords(id1,5,5,1040)",
        ],
        "annotation": [
            "holdsAt(meeting(id0,id1),1040)", "holdsAt(meeting(id1,id0),1040)",
        ],
    }
    return [doc_a, doc_a_dup, doc_b, doc_c]


@pytest.fixture()
def two_segment_file(tmp_path):
    path = tmp_path / "caviar-mini.json"
    _write_docs(path, _two_segment_docs())
    return path


# ---------------------------------------------------------------------------
# load_continuous: segmentation, dedup, drift guards
# ---------------------------------------------------------------------------


def test_load_continuous_splits_on_the_non_40ms_jump(two_segment_file):
    segments = load_continuous(str(two_segment_file))
    assert len(segments) == 2
    assert segments[0]["timestamps"] == [0, 40, 80, 120]
    assert segments[1]["timestamps"] == [1000, 1040]


def test_load_continuous_dedupes_duplicate_docs_and_shared_boundary_frame(two_segment_file):
    segments = load_continuous(str(two_segment_file))
    # t=40 is asserted by doc_a (twice, duplicate) AND doc_b (boundary
    # overlap) -- it must appear exactly once in segment 0's own timestamps.
    assert segments[0]["timestamps"].count(40) == 1
    # id0's activity at t=40 is asserted 3 times across the 3 docs touching
    # it (doc_a, its duplicate, doc_b) -- exactly one entry survives.
    assert segments[0]["activity"][("id0", 40)] == "active"


def test_load_continuous_persons_sorted_by_numeric_id(two_segment_file):
    segments = load_continuous(str(two_segment_file))
    assert segments[0]["persons"] == ["id0", "id1"]


def test_load_continuous_refuses_on_timepoint_count_drift(two_segment_file):
    with pytest.raises(ValueError, match="unique narrative timepoints"):
        load_continuous(str(two_segment_file), expected_num_timepoints=999)


def test_load_continuous_refuses_on_segment_count_drift(two_segment_file):
    with pytest.raises(ValueError, match="video segments"):
        load_continuous(str(two_segment_file), expected_num_segments=999)


def test_load_continuous_accepts_matching_expected_counts(two_segment_file):
    segments = load_continuous(
        str(two_segment_file), expected_num_timepoints=6, expected_num_segments=2,
    )
    assert len(segments) == 2


# ---------------------------------------------------------------------------
# Annotation offset alignment: holdsAt(meeting(p1,p2),T)'s own T is used
# AS-IS to key the gold label at narrative timestep T (no shift) -- verified
# empirically (see the module docstring); this fixture pins that reading:
# t=40's meeting annotation lands on t=40's own pair-time row, not on t=0's
# or t=80's.
# ---------------------------------------------------------------------------


def test_meeting_annotation_timestamp_used_as_is_no_shift(two_segment_file):
    segments = load_continuous(str(two_segment_file))
    converted = convert_continuous(segments)
    # segment 0 rows in time order: t=0 (pt0), t=40 (pt1), t=80 (pt2), t=120 (pt3)
    assert converted["is_positive"][0] is False   # t=0: not yet meeting
    assert converted["is_positive"][1] is True    # t=40: meeting (as annotated)
    assert converted["is_positive"][2] is False   # t=80: no longer meeting
    assert converted["is_positive"][3] is False


# ---------------------------------------------------------------------------
# convert_continuous: pair co-visibility, relations, features, contract shape
# ---------------------------------------------------------------------------


def test_convert_continuous_num_pt_and_facts(two_segment_file):
    segments = load_continuous(str(two_segment_file))
    out = convert_continuous(segments)
    # 4 frames in segment 0 (one pair each) + 2 frames in segment 1 = 6 rows
    assert out["num_pt"] == 6
    assert out["facts"] == [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]


def test_convert_continuous_segment_of_pt_never_crosses_a_boundary(two_segment_file):
    segments = load_continuous(str(two_segment_file))
    out = convert_continuous(segments)
    assert out["segment_of_pt"] == [0, 0, 0, 0, 1, 1]


def test_convert_continuous_relations_both_active_and_far_close(two_segment_file):
    segments = load_continuous(str(two_segment_file))
    out = convert_continuous(segments)
    rel = out["relations"]
    # t=0 (pt0) and t=40 (pt1): both active, close (dist=5)
    assert (0, 1) in rel["both_active"] and (1, 1) in rel["both_active"]
    assert (0, 1) in rel["close"] and (1, 1) in rel["close"]
    # t=80 (pt2), t=120 (pt3): id0 inactive, id1 walking -- none of the four
    # activity relations fire; coords are far (dist=100)
    for name in ("both_active", "both_walking", "both_inactive", "mixed_active_walking"):
        assert (2, 1) not in rel[name] and (3, 1) not in rel[name]
    assert (2, 1) in rel["far"] and (3, 1) in rel["far"]
    # segment 1: both frames both-active and close (dist=0)
    assert (4, 1) in rel["both_active"] and (5, 1) in rel["both_active"]
    assert (4, 1) in rel["close"] and (5, 1) in rel["close"]


def test_convert_continuous_excludes_a_non_co_visible_person(tmp_path):
    # id2 has no activity event at all -- it must never appear in any pair row.
    docs = [{
        "_id": 0, "time": "0",
        "narrative": [
            "happensAt(active(id0),0)", "coords(id0,0,0,0)",
            "happensAt(active(id1),0)", "coords(id1,0,0,0)",
            "coords(id2,0,0,0)",  # coords only, no happensAt -- not co-visible
        ],
        "annotation": [],
    }]
    path = tmp_path / "mini.json"
    _write_docs(path, docs)
    segments = load_continuous(str(path))
    out = convert_continuous(segments)
    assert out["num_pt"] == 1  # only the (id0, id1) pair


def test_convert_continuous_missing_coords_row(tmp_path):
    docs = [{
        "_id": 0, "time": "0",
        "narrative": [
            "happensAt(active(id0),0)", "coords(id0,0,0,0)",
            "happensAt(active(id1),0)",  # id1 has no coords at t=0
        ],
        "annotation": [],
    }]
    path = tmp_path / "mini.json"
    _write_docs(path, docs)
    segments = load_continuous(str(path))
    out = convert_continuous(segments)
    assert out["n_coords_missing"] == 1
    assert out["relations"]["coords_missing"] == [(0, 1)]
    assert (0, 1) not in out["relations"]["close"]
    assert (0, 1) not in out["relations"]["far"]
    assert out["features"][0].tolist() == pytest.approx([0.0, 0.0])


def test_convert_continuous_output_keys_match_convert_split_plus_segment_of_pt(two_segment_file):
    segments = load_continuous(str(two_segment_file))
    continuous_out = convert_continuous(segments)
    split_out = convert_split([{
        "p1_labels": [0], "p2_labels": [0], "complex_labels": [0],
        "atoms": "coords(p1,0,0,0). coords(p2,0,0,0).",
    }])
    assert set(continuous_out) == set(split_out) | {"segment_of_pt", "transition_relations"}


# ---------------------------------------------------------------------------
# derive_ec_targets_continuous: real transitions only by default
# ---------------------------------------------------------------------------


def test_derive_ec_targets_continuous_real_transitions(two_segment_file):
    segments = load_continuous(str(two_segment_file))
    converted = convert_continuous(segments)
    ec = derive_ec_targets_continuous(segments, converted)
    # segment 0: t=0 False, t=40 True (real init), t=80 False (real term), t=120 False
    # segment 1: t=1000 False, t=1040 True (real init, no term: still holding
    # at the pair's own last observed pair-time)
    assert ec["is_init"] == [False, True, False, False, False, True]
    assert ec["is_term"] == [False, False, True, False, False, False]
    assert ec["n_init"] == 2
    assert ec["n_term"] == 1


def test_derive_ec_targets_continuous_first_observed_holding_excluded_by_default(tmp_path):
    # Pair is ALREADY meeting at its very first observed co-visible frame in
    # the segment -- the default reading must NOT count this as an init.
    docs = [{
        "_id": 0, "time": "0",
        "narrative": [
            "happensAt(active(id0),0)", "coords(id0,0,0,0)",
            "happensAt(active(id1),0)", "coords(id1,0,0,0)",
            "happensAt(active(id0),40)", "coords(id0,0,0,40)",
            "happensAt(active(id1),40)", "coords(id1,0,0,40)",
        ],
        "annotation": [
            "holdsAt(meeting(id0,id1),0)", "holdsAt(meeting(id1,id0),0)",
            "holdsAt(meeting(id0,id1),40)", "holdsAt(meeting(id1,id0),40)",
        ],
    }]
    path = tmp_path / "mini.json"
    _write_docs(path, docs)
    segments = load_continuous(str(path))
    converted = convert_continuous(segments)

    ec_default = derive_ec_targets_continuous(segments, converted)
    assert ec_default["is_init"] == [False, False]
    assert ec_default["n_init"] == 0
    assert ec_default["n_init_real_transitions_only"] == 0
    assert ec_default["n_init_including_first_observed_holding"] == 1

    ec_incl = derive_ec_targets_continuous(segments, converted, treat_first_observed_as_init=True)
    assert ec_incl["is_init"] == [True, False]
    assert ec_incl["n_init"] == 1


def test_derive_ec_targets_continuous_rejects_mismatched_is_positive_length(two_segment_file):
    segments = load_continuous(str(two_segment_file))
    converted = convert_continuous(segments)
    bad = dict(converted, is_positive=converted["is_positive"][:-1])
    with pytest.raises(ValueError, match="is_positive"):
        derive_ec_targets_continuous(segments, bad)


def test_derive_ec_targets_continuous_rejects_segments_not_matching_converted(two_segment_file):
    segments = load_continuous(str(two_segment_file))
    converted = convert_continuous(segments)
    with pytest.raises(ValueError, match="segments/converted mismatch"):
        derive_ec_targets_continuous(segments[:1], converted)  # dropped segment 1's rows


# ---------------------------------------------------------------------------
# reconstruct_holds_continuous
# ---------------------------------------------------------------------------


def test_reconstruct_holds_continuous_matches_gold(two_segment_file):
    segments = load_continuous(str(two_segment_file))
    converted = convert_continuous(segments)
    ec = derive_ec_targets_continuous(segments, converted)
    holds = reconstruct_holds_continuous(ec["is_init"], ec["is_term"], segments, converted)
    assert holds == converted["is_positive"]


def test_reconstruct_holds_continuous_resets_state_per_segment(tmp_path):
    # Segment 0 ends HOLDING (init, never terminated within the segment);
    # segment 1 has no init at all -- it must not inherit segment 0's state.
    docs = [
        {
            "_id": 0, "time": "0",
            "narrative": [
                "happensAt(active(id0),0)", "coords(id0,0,0,0)",
                "happensAt(active(id1),0)", "coords(id1,0,0,0)",
                "happensAt(active(id0),40)", "coords(id0,0,0,40)",
                "happensAt(active(id1),40)", "coords(id1,0,0,40)",
            ],
            "annotation": [
                "holdsAt(meeting(id0,id1),40)", "holdsAt(meeting(id1,id0),40)",
            ],
        },
        {
            "_id": 1, "time": "1000",
            "narrative": [
                "happensAt(active(id0),1000)", "coords(id0,0,0,1000)",
                "happensAt(active(id1),1000)", "coords(id1,0,0,1000)",
            ],
            "annotation": [],
        },
    ]
    path = tmp_path / "mini.json"
    _write_docs(path, docs)
    segments = load_continuous(str(path))
    converted = convert_continuous(segments)
    ec = derive_ec_targets_continuous(segments, converted)
    holds = reconstruct_holds_continuous(ec["is_init"], ec["is_term"], segments, converted)
    # pt0 (seg0,t0)=F, pt1 (seg0,t40)=T; pt2 (seg1,t1000) must be F, not
    # inherited True from segment 0's own final state.
    assert holds == [False, True, False]


def test_reconstruct_holds_continuous_rejects_mismatched_init_length(two_segment_file):
    segments = load_continuous(str(two_segment_file))
    converted = convert_continuous(segments)
    with pytest.raises(ValueError):
        reconstruct_holds_continuous([True, False], [False] * 6, segments, converted)


def test_reconstruct_holds_continuous_rejects_mismatched_term_length(two_segment_file):
    segments = load_continuous(str(two_segment_file))
    converted = convert_continuous(segments)
    with pytest.raises(ValueError):
        reconstruct_holds_continuous([False] * 6, [True, False], segments, converted)


# ---------------------------------------------------------------------------
# transition_relations: frame-difference vocabulary, ec-mode only (kept out
# of "relations" -- see convert_continuous's own docstring paragraph on why).
# One pair (id0, id1), id1 held fixed at "inactive" throughout so every fired
# relation below is caused by id0's own activity alone, isolating each of the
# four relations to its own pair-time.
# ---------------------------------------------------------------------------


@pytest.fixture()
def transition_fixture_file(tmp_path):
    docs = [{
        "_id": 0, "time": "0",
        "narrative": [
            "happensAt(inactive(id0),0)", "coords(id0,0,0,0)",
            "happensAt(inactive(id1),0)", "coords(id1,0,0,0)",
            "happensAt(active(id0),40)", "coords(id0,0,0,40)",
            "happensAt(inactive(id1),40)", "coords(id1,0,0,40)",
            "happensAt(inactive(id0),80)", "coords(id0,0,0,80)",
            "happensAt(inactive(id1),80)", "coords(id1,0,0,80)",
            "happensAt(walking(id0),120)", "coords(id0,0,0,120)",
            "happensAt(inactive(id1),120)", "coords(id1,0,0,120)",
            "happensAt(running(id0),160)", "coords(id0,0,0,160)",
            "happensAt(inactive(id1),160)", "coords(id1,0,0,160)",
        ],
        "annotation": [],
    }]
    path = tmp_path / "mini.json"
    _write_docs(path, docs)
    return path


def test_transition_relations_exact_contents(transition_fixture_file):
    segments = load_continuous(str(transition_fixture_file))
    out = convert_continuous(segments)
    tr = out["transition_relations"]
    # pt0=t0 (first observed), pt1=t40 (inactive->active), pt2=t80
    # (active->inactive), pt3=t120 (inactive->walking), pt4=t160
    # (walking->running).
    assert tr["any_became_active"] == [(1, 1)]
    assert tr["any_became_inactive"] == [(2, 1)]
    assert tr["any_became_walking"] == [(3, 1)]
    assert tr["any_stopped_walking"] == [(4, 1)]


def test_transition_relations_first_observed_pt_belongs_to_none(transition_fixture_file):
    segments = load_continuous(str(transition_fixture_file))
    out = convert_continuous(segments)
    tr = out["transition_relations"]
    for name in TRANSITION_RELATION_NAMES:
        assert (0, 1) not in tr[name]


def test_transition_relations_are_never_merged_into_relations(transition_fixture_file):
    segments = load_continuous(str(transition_fixture_file))
    out = convert_continuous(segments)
    for name in TRANSITION_RELATION_NAMES:
        assert name not in out["relations"]
    assert set(out["transition_relations"]) == set(TRANSITION_RELATION_NAMES)


def test_transition_relations_direct_protocol_vocabulary_guard(two_segment_file):
    # Any --data continuous run (direct OR ec protocol) shares this same
    # convert_continuous output; a direct-protocol vocabulary builder reads
    # only "relations" (run_caviar_theory.py's _filtered_relation_names /
    # ACTIVITY_RELATIONS), never "transition_relations" -- pinning that the
    # two dicts never overlap is what keeps that guarantee true regardless
    # of which fixture produced them.
    segments = load_continuous(str(two_segment_file))
    out = convert_continuous(segments)
    assert set(out["relations"]) & set(TRANSITION_RELATION_NAMES) == set()


# ---------------------------------------------------------------------------
# derive_ec_masks_continuous: don't-care truth table (task-e2-review.md's F3)
# ---------------------------------------------------------------------------


def test_derive_ec_masks_continuous_matches_two_segment_fixture(two_segment_file):
    segments = load_continuous(str(two_segment_file))
    converted = convert_continuous(segments)
    masks = derive_ec_masks_continuous(segments, converted)
    # pt0/pt4: first-observed, not holding -> term don't-care, init negative.
    # pt1/pt5: real init transitions -> neither dontcare.
    # pt2: real term transition -> neither dontcare.
    # pt3: not-holds-and-not-held-before -> term don't-care, init negative.
    assert masks["init_dontcare"] == [False, False, False, False, False, False]
    assert masks["term_dontcare"] == [True, False, False, True, True, False]
    assert masks["n_init_dontcare"] == 0
    assert masks["n_term_dontcare"] == 3
    assert masks["num_pt"] == 6


@pytest.fixture()
def mid_interval_file(tmp_path):
    # One pair, one segment, 5 frames: not-meeting, meet, meet, meet,
    # not-meeting -- exercises the "pos and prev" (init don't-care / term
    # negative) cell at the two MIDDLE meeting frames.
    docs = [{
        "_id": 0, "time": "0",
        "narrative": [
            f"happensAt(active(id0),{t})" for t in (0, 40, 80, 120, 160)
        ] + [
            f"coords(id0,0,0,{t})" for t in (0, 40, 80, 120, 160)
        ] + [
            f"happensAt(active(id1),{t})" for t in (0, 40, 80, 120, 160)
        ] + [
            f"coords(id1,0,0,{t})" for t in (0, 40, 80, 120, 160)
        ],
        "annotation": [
            f"holdsAt(meeting(id0,id1),{t})" for t in (40, 80, 120)
        ] + [
            f"holdsAt(meeting(id1,id0),{t})" for t in (40, 80, 120)
        ],
    }]
    path = tmp_path / "mini.json"
    _write_docs(path, docs)
    return path


def test_derive_ec_masks_continuous_mid_interval_cells(mid_interval_file):
    segments = load_continuous(str(mid_interval_file))
    converted = convert_continuous(segments)
    assert converted["is_positive"] == [False, True, True, True, False]
    ec = derive_ec_targets_continuous(segments, converted)
    assert ec["is_init"] == [False, True, False, False, False]
    assert ec["is_term"] == [False, False, False, False, True]

    masks = derive_ec_masks_continuous(segments, converted)
    # pt2, pt3: holds now AND held before -> init don't-care, term negative.
    assert masks["init_dontcare"] == [False, False, True, True, False]
    assert masks["term_dontcare"] == [True, False, False, False, False]
    assert masks["n_init_dontcare"] == 2
    assert masks["n_term_dontcare"] == 1


def test_derive_ec_masks_continuous_first_observed_holding(tmp_path):
    # Same fixture as derive_ec_targets_continuous's own first-observed test:
    # the pair is already meeting at its very first observed pair-time.
    docs = [{
        "_id": 0, "time": "0",
        "narrative": [
            "happensAt(active(id0),0)", "coords(id0,0,0,0)",
            "happensAt(active(id1),0)", "coords(id1,0,0,0)",
            "happensAt(active(id0),40)", "coords(id0,0,0,40)",
            "happensAt(active(id1),40)", "coords(id1,0,0,40)",
        ],
        "annotation": [
            "holdsAt(meeting(id0,id1),0)", "holdsAt(meeting(id1,id0),0)",
            "holdsAt(meeting(id0,id1),40)", "holdsAt(meeting(id1,id0),40)",
        ],
    }]
    path = tmp_path / "mini.json"
    _write_docs(path, docs)
    segments = load_continuous(str(path))
    converted = convert_continuous(segments)

    masks_default = derive_ec_masks_continuous(segments, converted)
    # pt0: first-observed AND holding -> unobservable initiation -> init
    # don't-care (term stays negative: the fluent demonstrably holds).
    # pt1: holds AND held before (pt0) -> also init don't-care.
    assert masks_default["init_dontcare"] == [True, True]
    assert masks_default["term_dontcare"] == [False, False]

    masks_incl = derive_ec_masks_continuous(
        segments, converted, treat_first_observed_as_init=True
    )
    # With the flag, pt0 becomes the real is_init positive instead --
    # dropped from init_dontcare; pt1's own mid-interval cell is unaffected.
    assert masks_incl["init_dontcare"] == [False, True]
    assert masks_incl["term_dontcare"] == [False, False]


def test_derive_ec_masks_continuous_rejects_mismatched_is_positive_length(two_segment_file):
    segments = load_continuous(str(two_segment_file))
    converted = convert_continuous(segments)
    bad = dict(converted, is_positive=converted["is_positive"][:-1])
    with pytest.raises(ValueError, match="is_positive"):
        derive_ec_masks_continuous(segments, bad)


def test_derive_ec_masks_continuous_rejects_segments_not_matching_converted(two_segment_file):
    segments = load_continuous(str(two_segment_file))
    converted = convert_continuous(segments)
    with pytest.raises(ValueError, match="segments/converted mismatch"):
        derive_ec_masks_continuous(segments[:1], converted)
