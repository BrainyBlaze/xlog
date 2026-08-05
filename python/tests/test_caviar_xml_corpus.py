"""Unit tests for `caviar_xml_corpus.py` -- CPU, small hand-built XML
strings, following the style of `test_caviar_continuous.py`. A second group
of tests (below the "REAL DATA" marker) is DATA-GATED: they run only when
the real 30-file CAVIAR ground-truth XML directory (``CAVIAR_XML_DIR``) --
and, for the dump cross-check, additionally the real OLED/WOLED continuous
dump directory (``CAVIAR_CONTINUOUS_DIR``) -- are present locally, since
neither dataset is shipped in this repo (same gating style as
`test_caviar_alignment_evidence.py`).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "caviar_woled"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from caviar_xml_corpus import (  # noqa: E402
    HLE_MAPPING,
    TRANSITION_RELATION_NAMES,
    convert_xml_corpus,
    count_fluent_transitions,
    load_xml_corpus,
    parse_video,
)


# ---------------------------------------------------------------------------
# Tiny synthetic-XML builders (plain string templates -- no need for a real
# XML library here, only to *produce* well-formed fixtures the module's own
# `xml.etree.ElementTree`-based parser will consume).
# ---------------------------------------------------------------------------


def _obj_xml(objid, xc, yc, *, movement="walking", appearance="visible",
             orientation=0, w=10, h=10, with_hypothesis=True):
    hyp = ""
    if with_hypothesis:
        hyp = f"""
            <hypothesislist>
                <hypothesis id="1" evaluation="1.0">
                    <movement evaluation="1.0">{movement}</movement>
                    <role evaluation="1.0">walker</role>
                    <context evaluation="1.0">walking</context>
                    <situation evaluation="1.0">moving</situation>
                </hypothesis>
            </hypothesislist>"""
    return f"""
        <object id="{objid}">
            <orientation>{orientation}</orientation>
            <box xc="{xc}" yc="{yc}" w="{w}" h="{h}"/>
            <appearance>{appearance}</appearance>{hyp}
        </object>"""


def _grp_xml(grpid, members, situation):
    members_str = ",".join(str(m) for m in members)
    return f"""
        <group id="{grpid}">
            <members>{members_str}</members>
            <appearance>visible</appearance>
            <hypothesislist>
                <hypothesis id="1" evaluation="1.0">
                    <movement evaluation="1.0">movement</movement>
                    <role evaluation="1.0">walkers</role>
                    <context evaluation="1.0">meeting</context>
                    <situation evaluation="1.0">{situation}</situation>
                </hypothesis>
            </hypothesislist>
        </group>"""


def _frame_xml(number, objects_xml="", groups_xml=""):
    return f"""
    <frame number="{number}">
        <objectlist>{objects_xml}
        </objectlist>
        <grouplist>{groups_xml}
        </grouplist>
    </frame>"""


def _dataset_xml(frames_xml):
    return f'<?xml version="1.0" encoding="UTF-8"?><dataset name="Test">{frames_xml}\n</dataset>'


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _canon(p1: str, p2: str) -> tuple[str, str]:
    return tuple(sorted((p1, p2), key=lambda pid: int(pid[2:])))


# ---------------------------------------------------------------------------
# load_xml_corpus: file discovery (.xml AND .xml_), empty-dir guard
# ---------------------------------------------------------------------------


def test_load_xml_corpus_catches_xml_and_xml_underscore_extensions(tmp_path):
    content = _dataset_xml(_frame_xml(0, _obj_xml(0, 10, 10) + _obj_xml(1, 20, 20)))
    _write(tmp_path / "a.xml", content)
    _write(tmp_path / "b.xml_", content)
    videos = load_xml_corpus(str(tmp_path))
    assert [v["video"] for v in videos] == ["a.xml", "b.xml_"]


def test_load_xml_corpus_raises_on_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="no CAVIAR"):
        load_xml_corpus(str(tmp_path))


# ---------------------------------------------------------------------------
# Frame -> timestamp: frame*40ms default, per-video independence, offset
# ---------------------------------------------------------------------------


def test_parse_video_frame_to_timestamp_is_frame_times_40ms(tmp_path):
    frames = _frame_xml(0, _obj_xml(0, 10, 10) + _obj_xml(1, 20, 20)) + _frame_xml(
        1, _obj_xml(0, 11, 10) + _obj_xml(1, 21, 20)
    )
    path = tmp_path / "v.xml"
    _write(path, _dataset_xml(frames))
    video = parse_video(str(path))
    assert video["timestamps"] == [0, 40]
    assert video["persons"] == ["id0", "id1"]


def test_parse_video_time_offset_shifts_the_whole_timeline(tmp_path):
    # The per-video offset (used to align one video's own local clock onto
    # an external one, e.g. the OLED dump's) is a CONSTANT added to every
    # frame*40ms timestamp in the video -- never a per-atom or per-fluent
    # shift.
    frames = _frame_xml(0, _obj_xml(0, 10, 10)) + _frame_xml(1, _obj_xml(0, 11, 10))
    path = tmp_path / "v.xml"
    _write(path, _dataset_xml(frames))
    video = parse_video(str(path), time_offset=24440)
    assert video["timestamps"] == [24440, 24480]


def test_load_xml_corpus_keeps_each_video_as_its_own_independent_segment(tmp_path):
    pytest.importorskip("torch")  # convert_xml_corpus builds a feature tensor
    one_frame = _dataset_xml(
        _frame_xml(0, _obj_xml(0, 0, 0) + _obj_xml(1, 10, 10), _grp_xml(0, [0, 1], "interacting"))
    )
    _write(tmp_path / "a.xml", one_frame)
    _write(tmp_path / "b.xml", one_frame)
    videos = load_xml_corpus(str(tmp_path))
    # Both videos independently start their own timeline at frame 0 -- no
    # splicing/continuation of one video's clock into the next's.
    assert videos[0]["timestamps"] == [0]
    assert videos[1]["timestamps"] == [0]

    out = convert_xml_corpus(videos, fluent="meeting")
    assert out["segment_of_pt"] == [0, 1]
    assert out["is_positive"] == [True, True]
    assert out["num_pt"] == 2


def test_parse_video_strips_whitespace_padded_object_ids(tmp_path):
    # Group <members> entries are stripped when parsed; the <object id=...>
    # attribute must be stripped the same way, or a whitespace-padded id
    # (<object id=" 1">) silently splits one person into two identities and
    # orphans every group annotation naming that person.
    objs = _obj_xml(0, 0, 0) + _obj_xml(" 1", 10, 10)
    frames = _frame_xml(0, objs, _grp_xml(0, [0, 1], "interacting"))
    path = tmp_path / "v.xml"
    _write(path, _dataset_xml(frames))
    video = parse_video(str(path))
    assert video["persons"] == ["id0", "id1"]
    assert ("id1", 0) in video["tracked"]
    assert video["holds"]["meeting"] == {("id0", "id1", 0), ("id1", "id0", 0)}


def test_parse_video_object_without_id_raises_naming_file_and_frame(tmp_path):
    frames = _frame_xml(3, "<object><appearance>visible</appearance></object>")
    path = tmp_path / "v.xml"
    _write(path, _dataset_xml(frames))
    with pytest.raises(ValueError, match=r"v\.xml.*frame 3"):
        parse_video(str(path))


# ---------------------------------------------------------------------------
# box-center computation: box xc/yc read directly as the center pixel
# ---------------------------------------------------------------------------


def test_parse_video_box_center_reads_xc_yc_directly(tmp_path):
    frames = _frame_xml(0, _obj_xml(0, "199.0", "224.0", w=46, h=19))
    path = tmp_path / "v.xml"
    _write(path, _dataset_xml(frames))
    video = parse_video(str(path))
    assert video["coords"][("id0", 0)] == (199, 224)


# ---------------------------------------------------------------------------
# Co-visibility: a TRACKED object entry, not "has a movement label"
# ---------------------------------------------------------------------------


def test_tracked_entry_present_even_without_a_movement_label(tmp_path):
    pytest.importorskip("torch")  # convert_xml_corpus builds a feature tensor
    obj_no_hypothesis = _obj_xml(0, 10, 10, with_hypothesis=False)
    obj_with_hypothesis = _obj_xml(1, 20, 20)
    frames = _frame_xml(0, obj_no_hypothesis + obj_with_hypothesis)
    path = tmp_path / "v.xml"
    _write(path, _dataset_xml(frames))
    video = parse_video(str(path))

    assert ("id0", 0) in video["tracked"]
    assert ("id0", 0) not in video["activity"]

    # Co-visibility (via convert_xml_corpus's pair-row walk) still admits
    # this pair, because "tracked", not "has activity", is the criterion.
    out = convert_xml_corpus([video], fluent="meeting")
    assert out["num_pt"] == 1


def test_a_person_absent_from_the_whole_video_is_never_a_pair_row(tmp_path):
    pytest.importorskip("torch")  # convert_xml_corpus builds a feature tensor
    # id2 never has a single <object> entry anywhere in the file.
    frames = _frame_xml(0, _obj_xml(0, 0, 0) + _obj_xml(1, 10, 10))
    path = tmp_path / "v.xml"
    _write(path, _dataset_xml(frames))
    videos = load_xml_corpus(str(tmp_path))
    assert videos[0]["persons"] == ["id0", "id1"]
    out = convert_xml_corpus(videos, fluent="meeting")
    assert out["num_pt"] == 1


# ---------------------------------------------------------------------------
# HLE (group situation) -> fluent mapping, verbatim from OLED's
# ParseCAVIAR.scala, and its pair-generation rules.
# ---------------------------------------------------------------------------


def test_hle_mapping_matches_oleds_parsecaviar_verbatim():
    assert HLE_MAPPING == {
        "moving": "moving",
        "fighting": "fighting",
        "leaving object": "leavingObject",
        "interacting": "meeting",
    }


def test_interacting_maps_to_meeting_both_ordered_directions(tmp_path):
    objs = _obj_xml(0, 0, 0) + _obj_xml(1, 10, 10)
    frames = _frame_xml(0, objs, _grp_xml(0, [0, 1], "interacting"))
    path = tmp_path / "v.xml"
    _write(path, _dataset_xml(frames))
    video = parse_video(str(path))
    assert video["holds"]["meeting"] == {("id0", "id1", 0), ("id1", "id0", 0)}


def test_moving_and_fighting_map_straight_through_both_directions(tmp_path):
    objs = _obj_xml(0, 0, 0) + _obj_xml(1, 10, 10)
    frames = _frame_xml(0, objs, _grp_xml(0, [0, 1], "moving")) + _frame_xml(
        1, objs, _grp_xml(0, [0, 1], "fighting")
    )
    path = tmp_path / "v.xml"
    _write(path, _dataset_xml(frames))
    video = parse_video(str(path))
    assert video["holds"]["moving"] == {("id0", "id1", 0), ("id1", "id0", 0)}
    assert video["holds"]["fighting"] == {("id0", "id1", 40), ("id1", "id0", 40)}


def test_moving_group_with_more_than_two_members_emits_all_ordered_2subsets(tmp_path):
    objs = _obj_xml(0, 0, 0) + _obj_xml(1, 10, 10) + _obj_xml(2, 20, 20)
    frames = _frame_xml(0, objs, _grp_xml(0, [0, 1, 2], "moving"))
    path = tmp_path / "v.xml"
    _write(path, _dataset_xml(frames))
    video = parse_video(str(path))
    assert video["holds"]["moving"] == {
        ("id0", "id1", 0), ("id1", "id0", 0),
        ("id0", "id2", 0), ("id2", "id0", 0),
        ("id1", "id2", 0), ("id2", "id1", 0),
    }


def test_leaving_object_uses_only_the_first_two_members_one_ordered_pair(tmp_path):
    objs = _obj_xml(0, 0, 0) + _obj_xml(1, 10, 10) + _obj_xml(2, 20, 20)
    # members order is "1,2,0" -- the special case must read head/second
    # member ONLY (members[0]=1, members[1]=2), ignoring the third entirely,
    # and emit exactly ONE direction (never the reverse pair too).
    grp = _grp_xml(0, [1, 2, 0], "leaving object")
    frames = _frame_xml(0, objs, grp)
    path = tmp_path / "v.xml"
    _write(path, _dataset_xml(frames))
    video = parse_video(str(path))
    assert video["holds"]["leavingObject"] == {("id1", "id2", 0)}


@pytest.mark.parametrize("dropped_situation", ["joining", "split up", "leaving victim"])
def test_situations_absent_from_hle_mapping_are_silently_dropped(tmp_path, dropped_situation):
    objs = _obj_xml(0, 0, 0) + _obj_xml(1, 10, 10)
    frames = _frame_xml(0, objs, _grp_xml(0, [0, 1], dropped_situation))
    path = tmp_path / "v.xml"
    _write(path, _dataset_xml(frames))
    video = parse_video(str(path))
    assert video["holds"] == {}


# ---------------------------------------------------------------------------
# count_fluent_transitions: interval/init/term/pair-frame bookkeeping,
# "first-observed-holding is an interval but never an initiation"
# ---------------------------------------------------------------------------


def test_count_fluent_transitions_basic_init_and_term(tmp_path):
    objs = _obj_xml(0, 0, 0) + _obj_xml(1, 10, 10)
    frames = (
        _frame_xml(0, objs)
        + _frame_xml(1, objs, _grp_xml(0, [0, 1], "interacting"))
        + _frame_xml(2, objs, _grp_xml(0, [0, 1], "interacting"))
        + _frame_xml(3, objs)
    )
    path = tmp_path / "v.xml"
    _write(path, _dataset_xml(frames))
    videos = load_xml_corpus(str(tmp_path))
    counts = count_fluent_transitions(videos, "meeting")
    assert counts == {"n_intervals": 1, "n_init": 1, "n_term": 1, "n_pairframes": 2}


def test_count_fluent_transitions_first_observed_holding_is_not_an_init(tmp_path):
    objs = _obj_xml(0, 0, 0) + _obj_xml(1, 10, 10)
    frames = (
        _frame_xml(0, objs, _grp_xml(0, [0, 1], "interacting"))
        + _frame_xml(1, objs, _grp_xml(0, [0, 1], "interacting"))
        + _frame_xml(2, objs)
    )
    path = tmp_path / "v.xml"
    _write(path, _dataset_xml(frames))
    videos = load_xml_corpus(str(tmp_path))
    counts = count_fluent_transitions(videos, "meeting")
    # Held already at the pair's very first co-visible frame -> counts
    # toward n_intervals but NOT n_init (no earlier frame to show the
    # transition happening); the drop back to not-holding at frame 2 is a
    # genuine, observed termination.
    assert counts == {"n_intervals": 1, "n_init": 0, "n_term": 1, "n_pairframes": 2}


def test_count_fluent_transitions_never_crosses_a_video_boundary(tmp_path):
    # Two videos, each independently already-meeting at their own first
    # frame -- neither contributes an init, and each video's own pair-walk
    # is entirely self-contained (no state carried from one file into another).
    one_frame = _dataset_xml(
        _frame_xml(0, _obj_xml(0, 0, 0) + _obj_xml(1, 10, 10), _grp_xml(0, [0, 1], "interacting"))
    )
    _write(tmp_path / "a.xml", one_frame)
    _write(tmp_path / "b.xml", one_frame)
    videos = load_xml_corpus(str(tmp_path))
    counts = count_fluent_transitions(videos, "meeting")
    assert counts == {"n_intervals": 2, "n_init": 0, "n_term": 0, "n_pairframes": 2}


def test_count_fluent_transitions_rejects_unmapped_fluent(tmp_path):
    frames = _frame_xml(0, _obj_xml(0, 0, 0) + _obj_xml(1, 10, 10))
    path = tmp_path / "v.xml"
    _write(path, _dataset_xml(frames))
    videos = load_xml_corpus(str(tmp_path))
    with pytest.raises(ValueError, match="fluent"):
        count_fluent_transitions(videos, "joining")


# ---------------------------------------------------------------------------
# convert_xml_corpus: output contract, fluent selection, guard rails
# ---------------------------------------------------------------------------


def test_convert_xml_corpus_output_keys_match_convert_continuous_contract(tmp_path):
    pytest.importorskip("torch")  # convert_xml_corpus builds a feature tensor
    frames = _frame_xml(0, _obj_xml(0, 0, 0) + _obj_xml(1, 10, 10), _grp_xml(0, [0, 1], "interacting"))
    path = tmp_path / "v.xml"
    _write(path, _dataset_xml(frames))
    videos = load_xml_corpus(str(tmp_path))
    out = convert_xml_corpus(videos, fluent="meeting")
    assert set(out) == {
        "facts", "is_positive", "relations", "features", "num_pt",
        "n_coords_missing", "segment_of_pt", "transition_relations",
    }
    assert set(out["transition_relations"]) == set(TRANSITION_RELATION_NAMES)
    assert out["features"][0].tolist() == pytest.approx([-10.0 / 100.0, -10.0 / 100.0])


def test_convert_xml_corpus_fluent_parameter_switches_is_positive(tmp_path):
    pytest.importorskip("torch")  # convert_xml_corpus builds a feature tensor
    objs = _obj_xml(0, 0, 0) + _obj_xml(1, 10, 10)
    frames = _frame_xml(0, objs, _grp_xml(0, [0, 1], "moving"))
    path = tmp_path / "v.xml"
    _write(path, _dataset_xml(frames))
    videos = load_xml_corpus(str(tmp_path))
    meeting_out = convert_xml_corpus(videos, fluent="meeting")
    moving_out = convert_xml_corpus(videos, fluent="moving")
    assert meeting_out["is_positive"] == [False]
    assert moving_out["is_positive"] == [True]


def test_convert_xml_corpus_rejects_unsupported_fluent(tmp_path):
    frames = _frame_xml(0, _obj_xml(0, 0, 0) + _obj_xml(1, 10, 10))
    path = tmp_path / "v.xml"
    _write(path, _dataset_xml(frames))
    videos = load_xml_corpus(str(tmp_path))
    with pytest.raises(ValueError, match="fluent"):
        convert_xml_corpus(videos, fluent="fighting")


def test_convert_xml_corpus_missing_coords_row(tmp_path):
    pytest.importorskip("torch")  # convert_xml_corpus builds a feature tensor
    obj0 = _obj_xml(0, 0, 0)
    # id1 tracked (appearance present) but WITHOUT a <box> at all.
    obj1 = """
        <object id="1">
            <orientation>0</orientation>
            <appearance>occluded</appearance>
        </object>"""
    frames = _frame_xml(0, obj0 + obj1)
    path = tmp_path / "v.xml"
    _write(path, _dataset_xml(frames))
    videos = load_xml_corpus(str(tmp_path))
    out = convert_xml_corpus(videos, fluent="meeting")
    assert out["n_coords_missing"] == 1
    assert out["relations"]["coords_missing"] == [(0, 1)]
    assert out["features"][0].tolist() == pytest.approx([0.0, 0.0])


# ===========================================================================
# REAL DATA -- everything below is DATA-GATED (see module docstring).
# ===========================================================================

CAVIAR_XML_DIR = os.environ.get("CAVIAR_XML_DIR")
CAVIAR_CONTINUOUS_DIR = os.environ.get("CAVIAR_CONTINUOUS_DIR")

_xml_dir_ok = bool(CAVIAR_XML_DIR) and Path(CAVIAR_XML_DIR).is_dir()
_continuous_dir_ok = bool(CAVIAR_CONTINUOUS_DIR) and Path(CAVIAR_CONTINUOUS_DIR).is_dir()

requires_xml_dir = pytest.mark.skipif(
    not _xml_dir_ok,
    reason=(
        "CAVIAR_XML_DIR not set to a real directory holding the 30 CAVIAR "
        "ground-truth ISO XML files (*.xml/*.xml_) -- this dataset is not "
        "shipped in this repo."
    ),
)

requires_xml_and_continuous_dirs = pytest.mark.skipif(
    not (_xml_dir_ok and _continuous_dir_ok),
    reason=(
        "needs BOTH CAVIAR_XML_DIR (ground-truth XML) and "
        "CAVIAR_CONTINUOUS_DIR (caviar-train.json/caviar-test.json) set to "
        "real data directories -- neither dataset is shipped in this repo."
    ),
)


@requires_xml_dir
class TestRealCorpusFluentCounts:
    """Corpus-level interval/init/term/pair-frame counts over all 30 real
    CAVIAR ground-truth videos (co-visible = both objects tracked; a
    transition is a real 0->1/1->0 change between consecutive co-visible
    frames within one video; a fluent already holding at a pair's first
    co-visible frame counts as an interval but not an initiation).

    Evidence status, stated honestly: only the MEETING counts are
    corroborated by an independent real-data cross-check (the wk2gt
    splice test below reproduces the OLED dump's own atoms from first
    principles). The MOVING and FIGHTING literals were first produced by
    `count_fluent_transitions` itself on this corpus -- they are
    REGRESSION PINS (the numbers freeze current behavior so silent
    parser/counting drift is caught), not independently-verified
    canonical counts. Anyone may re-derive them from the public XML by
    the definition above, but nobody has yet done so by a route that
    does not go through this module."""

    @pytest.fixture(scope="class")
    def videos(self):
        return load_xml_corpus(CAVIAR_XML_DIR)

    def test_meeting_counts(self, videos):
        """Corroborated: the wk2gt cross-check below independently ties
        this module's meeting derivation to the published OLED dump."""
        assert count_fluent_transitions(videos, "meeting") == {
            "n_intervals": 12, "n_init": 11, "n_term": 10, "n_pairframes": 1812,
        }

    def test_moving_counts(self, videos):
        """Regression pin: self-derived literals (see class docstring)."""
        assert count_fluent_transitions(videos, "moving") == {
            "n_intervals": 18, "n_init": 5, "n_term": 8, "n_pairframes": 3136,
        }

    def test_fighting_counts(self, videos):
        """Regression pin: self-derived literals (see class docstring)."""
        assert count_fluent_transitions(videos, "fighting") == {
            "n_intervals": 6, "n_init": 6, "n_term": 6, "n_pairframes": 630,
        }


@requires_xml_and_continuous_dirs
def test_wk2gt_derived_meeting_atoms_reproduce_the_dump_train_segment():
    """`wk2gt.xml`, read at its own proven splice offset (+24440ms onto the
    OLED dump's continuous train timeline), must derive EXACTLY the same
    `holdsAt(meeting(...))` atoms (as unordered, canonical pairs) the real
    `caviar-train.json` logs over that same time range -- the decisive,
    from-first-principles cross-check that this module's HLE mapping/pair
    generation/time-base agree with the published dump, not merely with
    each other."""
    from caviar_continuous import load_continuous

    video = parse_video(str(Path(CAVIAR_XML_DIR) / "wk2gt.xml"), time_offset=24440)
    derived = {_canon(p1, p2) + (t,) for (p1, p2, t) in video["holds"].get("meeting", ())}

    train_path = Path(CAVIAR_CONTINUOUS_DIR) / "caviar-train.json"
    dump_meeting: set[tuple[str, str, int]] = set()
    for seg in load_continuous(str(train_path)):
        dump_meeting |= seg["meeting"]

    wk2gt_ts = set(video["timestamps"])
    dump_meeting_in_range = {(p1, p2, t) for (p1, p2, t) in dump_meeting if t in wk2gt_ts}

    assert dump_meeting_in_range == derived
    assert len(derived) == 855
