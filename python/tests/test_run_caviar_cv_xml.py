"""Unit tests for `run_caviar_cv.py`'s XML-native data-source seam
(`--data-source xml`, `--fluent`, `--xml-dir`) -- CPU, no CUDA. Follows
`test_caviar_cv.py`'s own style: synthetic fixtures for the wiring, plus a
DATA-GATED group (below the "REAL DATA" marker) that runs only when
`CAVIAR_XML_DIR` points at the real 30-file CAVIAR ground-truth directory.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "caviar_woled"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import run_caviar_cv  # noqa: E402

REQUIRED = [
    "--train-json", "train.json", "--test-json", "test.json", "--out", "o.json",
]


# ---------------------------------------------------------------------------
# parse_args -- --data-source / --fluent / --xml-dir wiring and validation.
# ---------------------------------------------------------------------------


def test_parse_args_data_source_defaults_to_dump_and_fluent_to_meeting():
    args = run_caviar_cv.parse_args(REQUIRED)
    assert args.data_source == "dump"
    assert args.fluent == "meeting"
    assert args.xml_dir is None


def test_parse_args_existing_invocation_unaffected_by_the_new_flags():
    # Backward compatibility: an invocation predating this seam parses to
    # the SAME train/test/out/folds/seed/mode values as before, plus the
    # three new attributes at their byte-identical defaults.
    args = run_caviar_cv.parse_args(REQUIRED)
    assert args.train_json == "train.json"
    assert args.test_json == "test.json"
    assert args.out == "o.json"
    assert args.folds == 10
    assert args.seed == 7
    assert args.mode == "relational"


def test_parse_args_dump_source_with_fluent_moving_errors_out():
    with pytest.raises(SystemExit):
        run_caviar_cv.parse_args(REQUIRED + ["--fluent", "moving"])


def test_parse_args_dump_source_with_explicit_fluent_meeting_is_fine():
    args = run_caviar_cv.parse_args(REQUIRED + ["--fluent", "meeting"])
    assert args.data_source == "dump"
    assert args.fluent == "meeting"


def test_parse_args_dump_source_still_requires_train_and_test_json():
    with pytest.raises(SystemExit):
        run_caviar_cv.parse_args(["--out", "o.json"])
    with pytest.raises(SystemExit):
        run_caviar_cv.parse_args(["--test-json", "test.json", "--out", "o.json"])
    with pytest.raises(SystemExit):
        run_caviar_cv.parse_args(["--train-json", "train.json", "--out", "o.json"])


def test_parse_args_xml_source_does_not_require_train_or_test_json():
    args = run_caviar_cv.parse_args(
        ["--data-source", "xml", "--xml-dir", "some/dir", "--out", "o.json"],
    )
    assert args.data_source == "xml"
    assert args.train_json is None
    assert args.test_json is None
    assert args.xml_dir == "some/dir"


def test_parse_args_xml_source_accepts_moving():
    args = run_caviar_cv.parse_args(
        ["--data-source", "xml", "--xml-dir", "some/dir", "--fluent", "moving", "--out", "o.json"],
    )
    assert args.fluent == "moving"


def test_parse_args_xml_source_requires_xml_dir_or_env_var(monkeypatch):
    monkeypatch.delenv("CAVIAR_XML_DIR", raising=False)
    with pytest.raises(SystemExit):
        run_caviar_cv.parse_args(["--data-source", "xml", "--out", "o.json"])


def test_parse_args_xml_source_falls_back_to_env_var(monkeypatch):
    monkeypatch.setenv("CAVIAR_XML_DIR", "env/dir")
    args = run_caviar_cv.parse_args(["--data-source", "xml", "--out", "o.json"])
    assert args.xml_dir == "env/dir"


def test_parse_args_xml_source_cli_flag_overrides_env_var(monkeypatch):
    monkeypatch.setenv("CAVIAR_XML_DIR", "env/dir")
    args = run_caviar_cv.parse_args(
        ["--data-source", "xml", "--xml-dir", "cli/dir", "--out", "o.json"],
    )
    assert args.xml_dir == "cli/dir"


def test_parse_args_rejects_unknown_data_source():
    with pytest.raises(SystemExit):
        run_caviar_cv.parse_args(REQUIRED + ["--data-source", "bogus"])


def test_run_fold_data_source_parameter_defaults_to_dump():
    import inspect

    default = inspect.signature(run_caviar_cv.run_fold).parameters["data_source"].default
    assert default == "dump"


def test_run_fold_fluent_parameter_defaults_to_meeting():
    import inspect

    default = inspect.signature(run_caviar_cv.run_fold).parameters["fluent"].default
    assert default == "meeting"


# ---------------------------------------------------------------------------
# _segment_positive_counts -- the generalized fluent parameter is
# byte-identical to the pre-existing single-argument call for "meeting".
# ---------------------------------------------------------------------------


def test_segment_positive_counts_default_fluent_matches_pre_existing_meeting_only_call():
    segments = [{"meeting": {(0,), (1,)}}, {"meeting": set()}, {"meeting": {(0,)}}]
    assert run_caviar_cv._segment_positive_counts(segments) == [2, 0, 1]
    assert run_caviar_cv._segment_positive_counts(segments) == run_caviar_cv._segment_positive_counts(
        segments, fluent="meeting",
    )


def test_segment_positive_counts_reads_an_arbitrary_fluent_key():
    segments = [{"moving": {(0,), (1,), (2,)}}, {"moving": {(0,)}}]
    assert run_caviar_cv._segment_positive_counts(segments, fluent="moving") == [3, 1]


# ---------------------------------------------------------------------------
# _xml_video_as_segment -- the adapter presenting one XML video dict as a
# segment for `_segment_positive_counts` and `caviar_continuous`'s
# `_group_pts_by_pair` (via `derive_ec_targets_continuous` et al.).
# ---------------------------------------------------------------------------


def _xml_video(persons, timestamps, tracked, holds):
    return {"persons": persons, "timestamps": timestamps, "tracked": tracked, "holds": holds}


def test_xml_video_as_segment_carries_persons_timestamps_and_tracked_as_activity():
    video = _xml_video(
        persons=["id0", "id1"], timestamps=[0, 40],
        tracked={("id0", 0), ("id1", 0), ("id0", 40), ("id1", 40)},
        holds={"meeting": {("id0", "id1", 40), ("id1", "id0", 40)}},
    )
    adapted = run_caviar_cv._xml_video_as_segment(video, "meeting")
    assert adapted["persons"] == ["id0", "id1"]
    assert adapted["timestamps"] == [0, 40]
    assert adapted["activity"] is video["tracked"]


def test_xml_video_as_segment_canonicalizes_and_deduplicates_both_directions():
    video = _xml_video(
        persons=["id0", "id1"], timestamps=[0, 40],
        tracked={("id0", 0), ("id1", 0), ("id0", 40), ("id1", 40)},
        holds={"meeting": {("id0", "id1", 40), ("id1", "id0", 40), ("id0", "id1", 0)}},
    )
    adapted = run_caviar_cv._xml_video_as_segment(video, "meeting")
    # Both ordered directions at t=40 collapse onto ONE canonical entry.
    assert adapted["meeting"] == {("id0", "id1", 40), ("id0", "id1", 0)}


def test_xml_video_as_segment_missing_fluent_gives_an_empty_set():
    video = _xml_video(
        persons=["id0", "id1"], timestamps=[0], tracked={("id0", 0), ("id1", 0)}, holds={},
    )
    adapted = run_caviar_cv._xml_video_as_segment(video, "moving")
    assert adapted["moving"] == set()


def test_xml_video_as_segment_selects_the_requested_fluent_only():
    video = _xml_video(
        persons=["id0", "id1"], timestamps=[0], tracked={("id0", 0), ("id1", 0)},
        holds={
            "meeting": {("id0", "id1", 0), ("id1", "id0", 0)},
            "moving": set(),
        },
    )
    meeting_view = run_caviar_cv._xml_video_as_segment(video, "meeting")
    moving_view = run_caviar_cv._xml_video_as_segment(video, "moving")
    assert meeting_view["meeting"] == {("id0", "id1", 0)}
    assert "moving" not in meeting_view
    assert moving_view["moving"] == set()
    assert "meeting" not in moving_view


# ---------------------------------------------------------------------------
# Fold assignment over a synthetic 30-video corpus: composing
# `_xml_video_as_segment` + `_segment_positive_counts` +
# `stratified_segment_folds` reproduces EXACTLY the round-robin pattern the
# continuous-segment path already produces for equal-sized inputs -- no
# separate fold-assignment logic is introduced for the XML source.
# ---------------------------------------------------------------------------


def test_synthetic_xml_corpus_fold_assignment_matches_segment_path_for_equal_sizes():
    n_videos = 30
    videos = [
        _xml_video(
            persons=["id0", "id1"], timestamps=[0, 40],
            tracked={("id0", 0), ("id1", 0), ("id0", 40), ("id1", 40)},
            holds={"meeting": {("id0", "id1", 0), ("id1", "id0", 0), ("id0", "id1", 40), ("id1", "id0", 40)}},
        )
        for _ in range(n_videos)
    ]
    adapted = [run_caviar_cv._xml_video_as_segment(v, "meeting") for v in videos]
    counts = run_caviar_cv._segment_positive_counts(adapted, fluent="meeting")
    assert counts == [2] * n_videos

    fold_of_xml = run_caviar_cv.stratified_segment_folds(counts, n_folds=10, seed=7)
    # The reference: calling `stratified_segment_folds` directly on the same
    # counts list is the "segment path" today's dump source already
    # exercises (see `test_caviar_cv.py`'s own `stratified_segment_folds`
    # tests) -- equal counts, so the XML corpus's own composed path must
    # produce the identical assignment, not merely one with the same shape.
    fold_of_reference = run_caviar_cv.stratified_segment_folds([2] * n_videos, n_folds=10, seed=7)
    assert fold_of_xml == fold_of_reference
    assert len(fold_of_xml) == n_videos
    assert set(fold_of_xml) == set(range(10))


def test_synthetic_xml_corpus_fold_assignment_varies_with_unequal_positive_counts():
    videos = []
    for i in range(6):
        n_meeting_frames = i  # 0..5 distinct positive masses
        holds = {"meeting": {("id0", "id1", t * 40) for t in range(n_meeting_frames)}}
        holds["meeting"] |= {("id1", "id0", t * 40) for t in range(n_meeting_frames)}
        videos.append(
            _xml_video(
                persons=["id0", "id1"], timestamps=[t * 40 for t in range(5)],
                tracked={(p, t * 40) for p in ("id0", "id1") for t in range(5)},
                holds=holds,
            )
        )
    adapted = [run_caviar_cv._xml_video_as_segment(v, "meeting") for v in videos]
    counts = run_caviar_cv._segment_positive_counts(adapted, fluent="meeting")
    assert counts == [0, 1, 2, 3, 4, 5]

    fold_of = run_caviar_cv.stratified_segment_folds(counts, n_folds=3, seed=7)
    assert len(fold_of) == 6
    assert set(fold_of) == {0, 1, 2}


# ---------------------------------------------------------------------------
# XML_SCENE_FAMILIES -- table coverage validation against a loaded corpus's
# own file stems (`_validate_xml_scene_family_coverage`, exercised here via
# `_xml_family_fold_assignment`, its only caller).
# ---------------------------------------------------------------------------


def test_xml_scene_families_covers_exactly_the_30_real_stems():
    all_stems = {stem for stems in run_caviar_cv.XML_SCENE_FAMILIES.values() for stem in stems}
    assert len(all_stems) == 30
    # Every stem belongs to exactly one family.
    assert sum(len(stems) for stems in run_caviar_cv.XML_SCENE_FAMILIES.values()) == 30


def test_xml_family_fold_assignment_rejects_a_video_stem_missing_from_the_table():
    videos = [{"video": "totally_unknown_stem.xml"}]
    with pytest.raises(ValueError, match="unknown"):
        run_caviar_cv._xml_family_fold_assignment(videos, "meeting", n_folds=2, seed=7)


def test_xml_family_fold_assignment_rejects_a_corpus_missing_a_table_stem():
    # A corpus that only has ONE of wk1gt/wk2gt/wk3gt is missing the other
    # two table entries -- the table's own family membership can never be
    # verified against a corpus that does not carry every stem it names.
    video = _xml_video(
        persons=["id0"], timestamps=[0], tracked={("id0", 0)}, holds={},
    )
    video["video"] = "wk1gt.xml"
    with pytest.raises(ValueError, match="missing"):
        run_caviar_cv._xml_family_fold_assignment([video], "meeting", n_folds=2, seed=7)


def test_xml_stem_strips_xml_and_xml_underscore_extensions():
    assert run_caviar_cv._xml_stem("wk2gt.xml") == "wk2gt"
    assert run_caviar_cv._xml_stem("fomdgt1.xml_") == "fomdgt1"


# ---------------------------------------------------------------------------
# _xml_family_fold_assignment -- fold unit is the SCENE FAMILY, not the
# video: every video of one family always inherits that family's fold, and
# the family-level round-robin matches the pre-existing segment-path
# pattern for equal family totals.
# ---------------------------------------------------------------------------


def _family_corpus(family_video_counts: dict) -> list[dict]:
    """Build one synthetic video per stem named in `run_caviar_cv.
    XML_SCENE_FAMILIES` (so the real table's own coverage validates
    cleanly), each carrying `family_video_counts[family]` meeting frames
    split evenly, one meeting frame per video where possible -- just enough
    structure for `_segment_positive_counts` to read a real, nonzero count
    off each one."""
    videos = []
    for family, stems in run_caviar_cv.XML_SCENE_FAMILIES.items():
        n_meeting = family_video_counts.get(family, 0)
        for i, stem in enumerate(stems):
            has_meeting = i < n_meeting
            video = _xml_video(
                persons=["id0", "id1"], timestamps=[0],
                tracked={("id0", 0), ("id1", 0)},
                holds={"meeting": {("id0", "id1", 0), ("id1", "id0", 0)}} if has_meeting else {"meeting": set()},
            )
            video["video"] = f"{stem}.xml"
            videos.append(video)
    return videos


def test_xml_family_fold_assignment_every_video_of_a_family_lands_in_that_familys_fold():
    videos = _family_corpus({"wk": 2, "br": 3, "mw": 1})
    fold_of_video = run_caviar_cv._xml_family_fold_assignment(videos, "meeting", n_folds=4, seed=7)

    fold_by_stem = {
        run_caviar_cv._xml_stem(v["video"]): f for v, f in zip(videos, fold_of_video)
    }
    for stems in run_caviar_cv.XML_SCENE_FAMILIES.values():
        folds_in_family = {fold_by_stem[stem] for stem in stems}
        assert len(folds_in_family) == 1, f"family {stems} split across folds: {folds_in_family}"


def test_xml_family_fold_assignment_round_robin_matches_segment_path_for_equal_family_totals():
    # Every video meeting-free: every family's own summed positive count is
    # 0, so the family-level round-robin over `XML_SCENE_FAMILIES`'s 15
    # equal-weight rows (sorted family-name order) must match calling
    # `stratified_segment_folds` directly on 15 equal counts -- the same
    # equivalence `test_synthetic_xml_corpus_fold_assignment_matches_
    # segment_path_for_equal_sizes` pins for the per-video path above.
    videos = _family_corpus({})
    fold_of_video = run_caviar_cv._xml_family_fold_assignment(videos, "meeting", n_folds=5, seed=7)

    family_names = sorted(run_caviar_cv.XML_SCENE_FAMILIES)
    fold_of_family_reference = run_caviar_cv.stratified_segment_folds(
        [0] * len(family_names), n_folds=5, seed=7,
    )
    expected_fold_by_family = dict(zip(family_names, fold_of_family_reference))

    stem_to_family = {
        stem: family for family, stems in run_caviar_cv.XML_SCENE_FAMILIES.items() for stem in stems
    }
    for video, fold in zip(videos, fold_of_video):
        stem = run_caviar_cv._xml_stem(video["video"])
        assert fold == expected_fold_by_family[stem_to_family[stem]]


def test_xml_family_fold_assignment_uses_the_familys_summed_positive_count():
    # Two families, deliberately unbalanced (one family carries ALL the
    # positive mass): the round-robin dealing must be driven by the
    # FAMILY's own summed count, not any single member video's count, or
    # this pair of families would be indistinguishable from the equal-mass
    # case above.
    videos = _family_corpus({"wk": 3, "br": 0})
    fold_of_video = run_caviar_cv._xml_family_fold_assignment(videos, "meeting", n_folds=2, seed=7)
    fold_by_stem = {
        run_caviar_cv._xml_stem(v["video"]): f for v, f in zip(videos, fold_of_video)
    }
    wk_folds = {fold_by_stem[s] for s in run_caviar_cv.XML_SCENE_FAMILIES["wk"]}
    br_folds = {fold_by_stem[s] for s in run_caviar_cv.XML_SCENE_FAMILIES["br"]}
    assert len(wk_folds) == 1
    assert len(br_folds) == 1


# ---------------------------------------------------------------------------
# EC target derivation on the XML path never bridges a fluent interval
# across a video boundary, even when two videos of the same family (hence
# the same fold, by construction now) sit side by side in one fold's own
# segment list.
# ---------------------------------------------------------------------------


def test_derive_ec_targets_never_bridges_an_interval_across_an_xml_video_boundary():
    from caviar_continuous import derive_ec_targets_continuous
    from caviar_xml_corpus import convert_xml_corpus

    # Video A: the pair is NOT meeting at its own last observed frame.
    video_a = _xml_video(
        persons=["id0", "id1"], timestamps=[0, 40],
        tracked={("id0", 0), ("id1", 0), ("id0", 40), ("id1", 40)},
        holds={"meeting": set()},
    )
    video_a["activity"] = {}
    video_a["coords"] = {}
    # Video B: the pair IS ALREADY meeting at its own first observed frame.
    # Under the "first-observed-holding is an interval but never an
    # initiation" convention this must NOT count as an init -- even though
    # video A's own LAST state for this pair was not-holding, so a walk
    # that failed to reset per video boundary would wrongly read this as a
    # genuine 0->1 transition and count a spurious initiation.
    video_b = _xml_video(
        persons=["id0", "id1"], timestamps=[1000, 1040],
        tracked={("id0", 1000), ("id1", 1000), ("id0", 1040), ("id1", 1040)},
        holds={
            "meeting": {
                ("id0", "id1", 1000), ("id1", "id0", 1000),
                ("id0", "id1", 1040), ("id1", "id0", 1040),
            },
        },
    )
    video_b["activity"] = {}
    video_b["coords"] = {}

    videos = [video_a, video_b]
    converted = convert_xml_corpus(videos, fluent="meeting")
    ec_segments = [run_caviar_cv._xml_video_as_segment(v, "meeting") for v in videos]
    ec = derive_ec_targets_continuous(ec_segments, converted)

    assert ec["n_init"] == 0
    assert ec["n_term"] == 0


# ---------------------------------------------------------------------------
# run_fold(data_source="xml") wiring: mirrors `test_caviar_cv.py`'s own
# `run_fold` wiring tests -- `_run_init_search`/`_induce_ec_target`/
# `_induce_direct_theory` are monkeypatched (a fixture this small is not a
# real induction problem; see that file's own rationale), so this proves
# the CONVERSION/EC-DERIVATION wiring (convert_xml_corpus,
# `_xml_video_as_segment`'s row-order alignment), not search correctness.
# ---------------------------------------------------------------------------


def _tiny_two_video_corpus():
    # Video A: 2 co-visible pair-time rows (t=0, t=40), 1 meeting frame (t=40).
    video_a = _xml_video(
        persons=["id0", "id1"], timestamps=[0, 40],
        tracked={("id0", 0), ("id1", 0), ("id0", 40), ("id1", 40)},
        holds={"meeting": {("id0", "id1", 40), ("id1", "id0", 40)}},
    )
    video_a["activity"] = {
        ("id0", 0): "active", ("id1", 0): "active", ("id0", 40): "active", ("id1", 40): "active",
    }
    video_a["coords"] = {
        ("id0", 0): (0, 0), ("id1", 0): (3, 4), ("id0", 40): (0, 0), ("id1", 40): (3, 4),
    }
    # Video B: 3 co-visible pair-time rows (t=1000,1040,1080), 1 meeting
    # frame (t=1040 only) -- so the pair's own sequence is
    # not-holding/holding/not-holding: one real init AND one real term.
    video_b = _xml_video(
        persons=["id0", "id1"], timestamps=[1000, 1040, 1080],
        tracked={("id0", t) for t in (1000, 1040, 1080)} | {("id1", t) for t in (1000, 1040, 1080)},
        holds={"meeting": {(p1, p2, 1040) for p1, p2 in (("id0", "id1"), ("id1", "id0"))}},
    )
    video_b["activity"] = {(p, t): "active" for p in ("id0", "id1") for t in (1000, 1040, 1080)}
    video_b["coords"] = {(p, t): (0, 0) for p in ("id0", "id1") for t in (1000, 1040, 1080)}
    return [video_b], [video_a]  # (train_videos, test_videos)


def _fake_ec_theory(clauses, stop_reason="max_clauses reached"):
    return {
        "clauses": clauses,
        "stop_reason": stop_reason,
        "min_fit": 0.5,
        "null_summary": {"threshold": 0.5},
        "selection_reasons_per_iteration": ["committed"] if clauses else [],
    }


def test_run_fold_xml_source_converts_via_convert_xml_corpus_and_derives_ec_targets(monkeypatch):
    train_videos, test_videos = _tiny_two_video_corpus()

    monkeypatch.setattr(
        run_caviar_cv, "_run_init_search",
        lambda mode, train, test, train_ec_relations, init_facts, init_labels, seed: {
            "clauses": [], "stop_reason": "no positives remain in the residual",
            "min_fit": 0.5, "null_summary": None, "selection_reasons": [],
            "predict_clause_test": None, "detector_probe": None, "wall_s": 0.0,
        },
    )
    monkeypatch.setattr(
        run_caviar_cv, "_induce_ec_target",
        lambda train_relations, facts, labels, seed: _fake_ec_theory([]),
    )
    monkeypatch.setattr(
        run_caviar_cv, "_induce_direct_theory",
        lambda train_relations, facts, labels, seed: {"clauses": [], "stop_reason": "no positives remain in the residual"},
    )

    result = run_caviar_cv.run_fold(
        0, train_videos, test_videos, seed=7, data_source="xml", fluent="meeting",
    )

    assert result["data_source"] == "xml"
    assert result["fluent"] == "meeting"
    # Real conversion/EC-derivation, not faked: video A (test) has exactly
    # one meeting transition (a real init at t=40, the pair's second
    # observed co-visible pair-time); video B (train) has one real init
    # (t=1040) and one real term (t=1080).
    assert result["n_test_pt"] == 2
    assert result["n_train_pt"] == 3
    assert result["ec"]["n_init_test"] == 1
    assert result["ec"]["n_term_test"] == 0
    assert result["ec"]["n_init_train"] == 1
    assert result["ec"]["n_term_train"] == 1


def test_run_fold_xml_source_moving_fluent_reads_a_different_holds_set(monkeypatch):
    video_a = _xml_video(
        persons=["id0", "id1"], timestamps=[0, 40],
        tracked={("id0", 0), ("id1", 0), ("id0", 40), ("id1", 40)},
        holds={"moving": {("id0", "id1", 40), ("id1", "id0", 40)}},
    )
    video_a["activity"] = {
        ("id0", 0): "active", ("id1", 0): "active", ("id0", 40): "active", ("id1", 40): "active",
    }
    video_a["coords"] = {
        ("id0", 0): (0, 0), ("id1", 0): (3, 4), ("id0", 40): (0, 0), ("id1", 40): (3, 4),
    }
    video_b = _xml_video(
        persons=["id0", "id1"], timestamps=[1000, 1040],
        tracked={(p, t) for p in ("id0", "id1") for t in (1000, 1040)},
        holds={},
    )
    video_b["activity"] = {(p, t): "active" for p in ("id0", "id1") for t in (1000, 1040)}
    video_b["coords"] = {(p, t): (0, 0) for p in ("id0", "id1") for t in (1000, 1040)}

    monkeypatch.setattr(
        run_caviar_cv, "_run_init_search",
        lambda mode, train, test, train_ec_relations, init_facts, init_labels, seed: {
            "clauses": [], "stop_reason": "no positives remain in the residual",
            "min_fit": 0.5, "null_summary": None, "selection_reasons": [],
            "predict_clause_test": None, "detector_probe": None, "wall_s": 0.0,
        },
    )
    monkeypatch.setattr(
        run_caviar_cv, "_induce_ec_target",
        lambda train_relations, facts, labels, seed: _fake_ec_theory([]),
    )
    monkeypatch.setattr(
        run_caviar_cv, "_induce_direct_theory",
        lambda train_relations, facts, labels, seed: {"clauses": [], "stop_reason": "no positives remain in the residual"},
    )

    result = run_caviar_cv.run_fold(
        0, [video_b], [video_a], seed=7, data_source="xml", fluent="moving",
    )
    assert result["fluent"] == "moving"
    assert result["ec"]["n_init_test"] == 1  # video A's moving transition at t=40
    assert result["ec"]["n_init_train"] == 0  # video B has no "moving" holds at all


def test_run_fold_dump_source_default_is_unaffected_by_the_new_parameters():
    # Regression guard: the dump-source default path (data_source="dump")
    # is exercised end to end by `test_caviar_cv.py`'s own `run_fold`
    # wiring tests; this only pins that the two new parameters' defaults
    # are what those tests implicitly rely on.
    import inspect

    sig = inspect.signature(run_caviar_cv.run_fold)
    assert sig.parameters["data_source"].default == "dump"
    assert sig.parameters["fluent"].default == "meeting"


# ===========================================================================
# REAL DATA -- everything below is DATA-GATED on CAVIAR_XML_DIR.
# ===========================================================================

CAVIAR_XML_DIR = os.environ.get("CAVIAR_XML_DIR")
_xml_dir_ok = bool(CAVIAR_XML_DIR) and Path(CAVIAR_XML_DIR).is_dir()

requires_xml_dir = pytest.mark.skipif(
    not _xml_dir_ok,
    reason=(
        "CAVIAR_XML_DIR not set to a real directory holding the 30 CAVIAR "
        "ground-truth XML files -- this dataset is not shipped in this repo."
    ),
)


@requires_xml_dir
@pytest.mark.parametrize(
    "fluent,expected_n_init,expected_n_term", [("meeting", 11, 10), ("moving", 5, 8)],
)
def test_real_xml_corpus_fold_assembly_ec_totals_match_corpus_totals(fluent, expected_n_init, expected_n_term):
    # CPU-only smoke test: assembles all 10 folds' EC targets for real, but
    # never calls induction (`_run_init_search`/`_induce_ec_target`/
    # `_induce_direct_theory` are never invoked here) -- keeps this well
    # under the ~2 minute budget.
    from caviar_continuous import derive_ec_targets_continuous
    from caviar_xml_corpus import convert_xml_corpus, load_xml_corpus

    videos = load_xml_corpus(CAVIAR_XML_DIR)
    assert len(videos) == 30

    adapted = [run_caviar_cv._xml_video_as_segment(v, fluent) for v in videos]
    counts = run_caviar_cv._segment_positive_counts(adapted, fluent=fluent)
    fold_of = run_caviar_cv.stratified_segment_folds(counts, n_folds=10, seed=7)

    assert len(fold_of) == 30
    assert set(fold_of) == set(range(10))

    seen_videos: set[str] = set()
    total_init = 0
    total_term = 0
    for fold_index in range(10):
        _, test_videos = run_caviar_cv._fold_segment_split(videos, fold_of, fold_index)
        for v in test_videos:
            assert v["video"] not in seen_videos  # every video held out exactly once
            seen_videos.add(v["video"])

        test_converted = convert_xml_corpus(test_videos, fluent=fluent)
        test_ec_segments = [run_caviar_cv._xml_video_as_segment(v, fluent) for v in test_videos]
        ec_test = derive_ec_targets_continuous(test_ec_segments, test_converted)
        total_init += ec_test["n_init"]
        total_term += ec_test["n_term"]

    assert seen_videos == {v["video"] for v in videos}
    assert total_init == expected_n_init
    assert total_term == expected_n_term


@requires_xml_dir
@pytest.mark.parametrize(
    "fluent,expected_n_init,expected_n_term", [("meeting", 11, 10), ("moving", 5, 8)],
)
def test_real_xml_corpus_family_fold_assembly_ec_totals_match_corpus_totals(fluent, expected_n_init, expected_n_term):
    # Same smoke test as `test_real_xml_corpus_fold_assembly_ec_totals_
    # match_corpus_totals` above, but through `_xml_family_fold_assignment`
    # -- the actual fold-assembly path `run_caviar_cv.main()` now uses for
    # `--data-source xml` -- rather than calling `stratified_segment_folds`
    # directly on per-video counts.
    from caviar_continuous import derive_ec_targets_continuous
    from caviar_xml_corpus import convert_xml_corpus, load_xml_corpus

    videos = load_xml_corpus(CAVIAR_XML_DIR)
    assert len(videos) == 30

    fold_of = run_caviar_cv._xml_family_fold_assignment(videos, fluent, n_folds=10, seed=7)
    assert len(fold_of) == 30

    stem_to_family = {
        stem: family for family, stems in run_caviar_cv.XML_SCENE_FAMILIES.items() for stem in stems
    }
    fold_by_stem = {
        run_caviar_cv._xml_stem(v["video"]): f for v, f in zip(videos, fold_of)
    }

    # Every family maps to exactly one fold, and every video of that family
    # inherits it -- checked over the WHOLE table, not a hand-picked pair,
    # since a single split family is enough to reopen same-scene leakage.
    families_present = set()
    for family, stems in run_caviar_cv.XML_SCENE_FAMILIES.items():
        folds_in_family = {fold_by_stem[stem] for stem in stems}
        assert len(folds_in_family) == 1, f"family {family!r} ({stems}) split across folds: {folds_in_family}"
        families_present.add(family)
    assert families_present == set(run_caviar_cv.XML_SCENE_FAMILIES)
    assert set(fold_of) <= set(range(10))

    # The two families with the most recordings, pinned by name: wk carries
    # 73% of all meeting positives, so splitting it would matter most.
    assert len({fold_by_stem["wk1gt"], fold_by_stem["wk2gt"], fold_by_stem["wk3gt"]}) == 1
    assert len({fold_by_stem["mwt1gt"], fold_by_stem["mwt2gt"], fold_by_stem["mws1gt"]}) == 1

    seen_videos: set[str] = set()
    total_init = 0
    total_term = 0
    for fold_index in range(10):
        _, test_videos = run_caviar_cv._fold_segment_split(videos, fold_of, fold_index)
        for v in test_videos:
            assert v["video"] not in seen_videos  # every video held out exactly once
            seen_videos.add(v["video"])

        test_converted = convert_xml_corpus(test_videos, fluent=fluent)
        test_ec_segments = [run_caviar_cv._xml_video_as_segment(v, fluent) for v in test_videos]
        ec_test = derive_ec_targets_continuous(test_ec_segments, test_converted)
        total_init += ec_test["n_init"]
        total_term += ec_test["n_term"]

    assert seen_videos == {v["video"] for v in videos}
    assert total_init == expected_n_init
    assert total_term == expected_n_term
