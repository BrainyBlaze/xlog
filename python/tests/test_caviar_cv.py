"""Unit tests for `run_caviar_cv.py` -- CPU, no CUDA, no real data file (the
real-data run is a separate, manual step; see `docs/experiments/caviar/
README.md`). Follows the style of `test_run_caviar_theory_args.py` (CLI
parse tests, torch-free-import guard) and `test_caviar_continuous.py`
(hand-built segment fixtures).
"""
import random
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
# parse_args -- CLI parsing, torch-free at import time
# ---------------------------------------------------------------------------


def test_run_caviar_cv_module_does_not_bind_torch_at_import_time():
    assert not hasattr(run_caviar_cv, "torch")


def test_parse_args_defaults():
    args = run_caviar_cv.parse_args(REQUIRED)
    assert args.train_json == "train.json"
    assert args.test_json == "test.json"
    assert args.out == "o.json"
    assert args.folds == 10
    assert args.seed == 7


def test_parse_args_overrides_folds_and_seed():
    args = run_caviar_cv.parse_args(REQUIRED + ["--folds", "5", "--seed", "3"])
    assert args.folds == 5
    assert args.seed == 3


def test_parse_args_requires_train_json():
    with pytest.raises(SystemExit):
        run_caviar_cv.parse_args(["--test-json", "test.json", "--out", "o.json"])


def test_parse_args_requires_test_json():
    with pytest.raises(SystemExit):
        run_caviar_cv.parse_args(["--train-json", "train.json", "--out", "o.json"])


def test_parse_args_requires_out():
    with pytest.raises(SystemExit):
        run_caviar_cv.parse_args(["--train-json", "train.json", "--test-json", "test.json"])


# ---------------------------------------------------------------------------
# stratified_segment_folds -- determinism + stratification property
# ---------------------------------------------------------------------------


def test_stratified_segment_folds_is_deterministic():
    counts = [50, 3, 12, 0, 7, 22, 1, 40, 5, 9, 2, 33]
    a = run_caviar_cv.stratified_segment_folds(counts, n_folds=4, seed=7)
    b = run_caviar_cv.stratified_segment_folds(counts, n_folds=4, seed=7)
    assert a == b


def test_stratified_segment_folds_different_seed_can_differ():
    counts = [50, 3, 12, 0, 7, 22, 1, 40, 5, 9, 2, 33]
    a = run_caviar_cv.stratified_segment_folds(counts, n_folds=4, seed=7)
    b = run_caviar_cv.stratified_segment_folds(counts, n_folds=4, seed=99)
    # Not a hard guarantee of the algorithm in general, but true for this
    # hand-picked case -- pins that seed actually reaches the fold_sequence draw.
    assert a != b


def test_stratified_segment_folds_assigns_every_segment_exactly_once():
    counts = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    fold_of = run_caviar_cv.stratified_segment_folds(counts, n_folds=3, seed=7)
    assert len(fold_of) == len(counts)
    assert set(fold_of) == {0, 1, 2}


def test_stratified_segment_folds_matches_the_documented_algorithm_by_hand():
    # 6 segments, 3 folds: sorted-descending order is [0(30),1(20),2(20),3(10),4(5),5(0)]
    # (ties at 20 broken by original index: seg1 before seg2).
    counts = [30, 20, 20, 10, 5, 0]
    fold_of = run_caviar_cv.stratified_segment_folds(counts, n_folds=3, seed=7)

    rng = torch.Generator().manual_seed(7)
    fold_sequence = torch.randperm(3, generator=rng).tolist()
    order = [0, 1, 2, 3, 4, 5]  # already sorted descending with index tiebreak
    expected = [0] * 6
    for rank, seg_idx in enumerate(order):
        expected[seg_idx] = fold_sequence[rank % 3]
    assert fold_of == expected


def test_stratified_segment_folds_rejects_too_few_folds():
    with pytest.raises(ValueError, match="n_folds must be >= 2"):
        run_caviar_cv.stratified_segment_folds([1, 2, 3], n_folds=1, seed=7)


def test_stratified_segment_folds_rejects_more_folds_than_segments():
    with pytest.raises(ValueError, match="fewer than n_folds"):
        run_caviar_cv.stratified_segment_folds([1, 2], n_folds=3, seed=7)


def test_stratified_segment_folds_rejects_negative_counts():
    with pytest.raises(ValueError, match="non-negative"):
        run_caviar_cv.stratified_segment_folds([1, -1, 2], n_folds=2, seed=7)


def _fold_sums(counts, fold_of, n_folds):
    sums = [0] * n_folds
    for c, f in zip(counts, fold_of):
        sums[f] += c
    return sums


def test_stratified_segment_folds_positive_mass_spread_bound_hand_case():
    counts = [100, 50, 1, 1]
    fold_of = run_caviar_cv.stratified_segment_folds(counts, n_folds=3, seed=7)
    sums = _fold_sums(counts, fold_of, 3)
    assert max(sums) - min(sums) <= max(counts)


def test_stratified_segment_folds_positive_mass_spread_bound_property():
    # Randomized property check (fixed RNG seed for a reproducible test run,
    # not the module's own seeding): for many random (counts, n_folds, seed)
    # combinations, the spread between any two folds' summed positive counts
    # never exceeds the single largest segment's own count -- the bound
    # `stratified_segment_folds` documents and proves by telescoping.
    rnd = random.Random(12345)
    for _ in range(200):
        n_folds = rnd.randint(2, 8)
        n = rnd.randint(n_folds, n_folds * 5)
        counts = [rnd.randint(0, 40) for _ in range(n)]
        seed = rnd.randint(0, 10_000)
        fold_of = run_caviar_cv.stratified_segment_folds(counts, n_folds=n_folds, seed=seed)
        sums = _fold_sums(counts, fold_of, n_folds)
        assert max(sums) - min(sums) <= max(counts), (n_folds, counts, seed)


def test_stratified_segment_folds_matches_real_caviar_scale():
    # 26 segments (21 train + 5 test), 10 folds -- the real corpus's own
    # shape; every fold must get at least one segment and the assignment
    # must still respect the spread bound.
    rnd = random.Random(7)
    counts = [rnd.randint(0, 5) for _ in range(26)]
    fold_of = run_caviar_cv.stratified_segment_folds(counts, n_folds=10, seed=7)
    assert len(fold_of) == 26
    for f in range(10):
        assert f in fold_of  # every fold gets at least one segment
    sums = _fold_sums(counts, fold_of, 10)
    assert max(sums) - min(sums) <= max(counts)


# ---------------------------------------------------------------------------
# _micro_prf1 -- exact arithmetic on a hand case
# ---------------------------------------------------------------------------


def test_micro_prf1_sums_counts_then_computes_prf1_not_a_mean_of_fold_f1s():
    # Fold A: tp=1, fp=0, fn=0 -> F1 = 1.0
    # Fold B: tp=0, fp=0, fn=9 -> F1 = 0.0
    # A per-fold-F1 MEAN would give 0.5; micro-aggregation sums first:
    # tp=1, fp=0, fn=9 -> precision=1.0, recall=0.1, F1=2*1*0.1/1.1=0.181818...
    counts = [
        {"tp": 1, "fp": 0, "fn": 0},
        {"tp": 0, "fp": 0, "fn": 9},
    ]
    got = run_caviar_cv._micro_prf1(counts)
    assert got["tp"] == 1
    assert got["fp"] == 0
    assert got["fn"] == 9
    assert got["precision"] == pytest.approx(1.0)
    assert got["recall"] == pytest.approx(0.1)
    assert got["f1"] == pytest.approx(2 * 1.0 * 0.1 / 1.1)
    assert got["f1"] != pytest.approx(0.5)  # would be the (wrong) mean-of-F1s reading


def test_micro_prf1_multi_fold_exact_hand_case():
    counts = [
        {"tp": 10, "fp": 2, "fn": 3},
        {"tp": 5, "fp": 1, "fn": 0},
        {"tp": 0, "fp": 4, "fn": 6},
    ]
    got = run_caviar_cv._micro_prf1(counts)
    tp, fp, fn = 15, 7, 9
    assert got["tp"] == tp
    assert got["fp"] == fp
    assert got["fn"] == fn
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    assert got["precision"] == pytest.approx(precision)
    assert got["recall"] == pytest.approx(recall)
    assert got["f1"] == pytest.approx(f1)


def test_micro_prf1_zero_division_reports_zero_not_nan_or_raise():
    counts = [{"tp": 0, "fp": 0, "fn": 0}]
    got = run_caviar_cv._micro_prf1(counts)
    assert got == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0}


def test_micro_prf1_empty_fold_list_is_the_all_zero_case():
    got = run_caviar_cv._micro_prf1([])
    assert got == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0}


# ---------------------------------------------------------------------------
# _exclude_dontcare -- same semantics as run_caviar_theory's own (pinned
# independently since this script reimplements it rather than importing it).
# ---------------------------------------------------------------------------


def test_exclude_dontcare_drops_flagged_rows():
    facts = [(0, 1), (1, 1), (2, 1), (3, 1)]
    labels = [False, True, False, True]
    dontcare = [True, False, True, False]
    kept_facts, kept_labels = run_caviar_cv._exclude_dontcare(facts, labels, dontcare)
    assert kept_facts == [(1, 1), (3, 1)]
    assert kept_labels == [True, True]


# ---------------------------------------------------------------------------
# Fold isolation: a synthetic two-segment corpus. Training on the OTHER
# fold's segment must never see the held-out segment's own pair-time rows --
# pinned via exact pt/positive counts (a leak would inflate train's own
# num_pt/positive count by exactly the held-out segment's contribution).
# ---------------------------------------------------------------------------


def _segment(persons, activity_by_t, coords_by_t, meeting_ts):
    """Build a `caviar_continuous.load_continuous`-shaped segment dict by
    hand: `activity_by_t`/`coords_by_t` are `{t: {person: value}}`, applied
    identically to every listed person; `meeting_ts` are the (single-pair)
    timestamps annotated as a meeting."""
    timestamps = sorted(activity_by_t)
    activity = {
        (p, t): act
        for t, by_person in activity_by_t.items()
        for p, act in by_person.items()
    }
    coords = {
        (p, t): xy
        for t, by_person in coords_by_t.items()
        for p, xy in by_person.items()
    }
    meeting = {(persons[0], persons[1], t) for t in meeting_ts}
    return {
        "timestamps": timestamps,
        "persons": sorted(persons),
        "activity": activity,
        "coords": coords,
        "meeting": meeting,
    }


def test_fold_isolation_train_conversion_never_sees_held_out_segment_rows():
    from caviar_continuous import convert_continuous

    # Segment A: 2 co-visible pair-time rows (t=0, t=40), 1 meeting frame (t=40).
    seg_a = _segment(
        ("id0", "id1"),
        activity_by_t={0: {"id0": "active", "id1": "active"}, 40: {"id0": "active", "id1": "active"}},
        coords_by_t={0: {"id0": (0, 0), "id1": (3, 4)}, 40: {"id0": (0, 0), "id1": (3, 4)}},
        meeting_ts=[40],
    )
    # Segment B: 3 co-visible pair-time rows (t=1000,1040,1080), 2 meeting frames.
    seg_b = _segment(
        ("id0", "id1"),
        activity_by_t={
            1000: {"id0": "active", "id1": "active"},
            1040: {"id0": "active", "id1": "active"},
            1080: {"id0": "active", "id1": "active"},
        },
        coords_by_t={
            1000: {"id0": (0, 0), "id1": (0, 0)},
            1040: {"id0": (0, 0), "id1": (0, 0)},
            1080: {"id0": (0, 0), "id1": (0, 0)},
        },
        meeting_ts=[1040, 1080],
    )
    segments = [seg_a, seg_b]

    # Fold 0 holds out segment A; fold 1 holds out segment B (assigned by
    # hand here, not via stratified_segment_folds -- this test is about
    # _fold_segment_split + convert_continuous isolation, not the
    # stratification algorithm, which has its own tests above).
    fold_of_segment = [0, 1]

    train_segments, test_segments = run_caviar_cv._fold_segment_split(segments, fold_of_segment, fold_index=0)
    assert train_segments == [seg_b]
    assert test_segments == [seg_a]

    train = convert_continuous(train_segments)
    test = convert_continuous(test_segments)

    # PIN VIA COUNTS: train sees ONLY segment B's 3 rows/2 positives, test
    # ONLY segment A's 2 rows/1 positive -- never segment A's rows leaking
    # into train's num_pt/positive count (a leak would make train num_pt=5).
    assert train["num_pt"] == 3
    assert sum(train["is_positive"]) == 2
    assert test["num_pt"] == 2
    assert sum(test["is_positive"]) == 1
    # No row lost or double-counted across the fold boundary either.
    assert train["num_pt"] + test["num_pt"] == 5
    assert sum(train["is_positive"]) + sum(test["is_positive"]) == 3

    # The symmetric split (fold 1 holds out segment B) is the mirror image.
    train_segments_2, test_segments_2 = run_caviar_cv._fold_segment_split(segments, fold_of_segment, fold_index=1)
    assert train_segments_2 == [seg_a]
    assert test_segments_2 == [seg_b]
    train_2 = convert_continuous(train_segments_2)
    test_2 = convert_continuous(test_segments_2)
    assert train_2["num_pt"] == 2
    assert sum(train_2["is_positive"]) == 1
    assert test_2["num_pt"] == 3
    assert sum(test_2["is_positive"]) == 2


def test_segment_positive_counts_reads_meeting_set_length():
    seg_a = _segment(
        ("id0", "id1"),
        activity_by_t={0: {"id0": "active", "id1": "active"}},
        coords_by_t={0: {"id0": (0, 0), "id1": (0, 0)}},
        meeting_ts=[0],
    )
    seg_b = _segment(
        ("id0", "id1"),
        activity_by_t={0: {"id0": "active", "id1": "active"}, 40: {"id0": "active", "id1": "active"}},
        coords_by_t={0: {"id0": (0, 0), "id1": (0, 0)}, 40: {"id0": (0, 0), "id1": (0, 0)}},
        meeting_ts=[0, 40],
    )
    assert run_caviar_cv._segment_positive_counts([seg_a, seg_b]) == [1, 2]


# ---------------------------------------------------------------------------
# _ec_relations / _direct_relations: vocabulary filtering
# ---------------------------------------------------------------------------


def test_ec_relations_merges_transitions_and_excludes_coords_missing():
    converted = {
        "relations": {"close": [(0, 1)], "coords_missing": [(1, 1)]},
        "transition_relations": {"any_became_active": [(0, 1)]},
    }
    got = run_caviar_cv._ec_relations(converted)
    assert got == {"close": [(0, 1)], "any_became_active": [(0, 1)]}


def test_direct_relations_excludes_transitions_and_coords_missing():
    converted = {
        "relations": {"close": [(0, 1)], "coords_missing": [(1, 1)]},
        "transition_relations": {"any_became_active": [(0, 1)]},
    }
    got = run_caviar_cv._direct_relations(converted)
    assert got == {"close": [(0, 1)]}


# ---------------------------------------------------------------------------
# --mode -- the flag defaults to 'relational' (byte-identical to every
# behavior that predates it), accepts only 'relational'/'neural', and
# leaves every other default untouched.
# ---------------------------------------------------------------------------


def test_parse_args_mode_defaults_to_relational():
    args = run_caviar_cv.parse_args(REQUIRED)
    assert args.mode == "relational"


def test_parse_args_mode_accepts_neural():
    args = run_caviar_cv.parse_args(REQUIRED + ["--mode", "neural"])
    assert args.mode == "neural"


def test_parse_args_mode_rejects_an_unknown_value():
    with pytest.raises(SystemExit):
        run_caviar_cv.parse_args(REQUIRED + ["--mode", "bogus"])


def test_omitting_mode_leaves_every_other_default_unchanged():
    omitted = run_caviar_cv.parse_args(REQUIRED)
    explicit = run_caviar_cv.parse_args(REQUIRED + ["--mode", "relational"])
    assert vars(omitted) == vars(explicit)


def test_run_fold_mode_parameter_defaults_to_relational():
    import inspect

    default = inspect.signature(run_caviar_cv.run_fold).parameters["mode"].default
    assert default == "relational"


# ---------------------------------------------------------------------------
# `_neural_init_vocab` -- --mode neural's restricted initiation candidate
# pool: the 4 activity relations + the 4 ACTIVITY-based transition
# relations only, never close/far/became_far/distance_increasing.
# ---------------------------------------------------------------------------


def test_neural_init_vocab_selects_activities_and_activity_transitions_only():
    converted = {
        "relations": {
            "both_active": [(0, 1)], "both_walking": [(1, 1)],
            "both_inactive": [(2, 1)], "mixed_active_walking": [(3, 1)],
            "close": [(4, 1)], "far": [(5, 1)], "coords_missing": [(6, 1)],
        },
        "transition_relations": {
            "any_became_active": [(0, 1)], "any_became_inactive": [(1, 1)],
            "any_became_walking": [(2, 1)], "any_stopped_walking": [(3, 1)],
            "became_far": [(4, 1)], "distance_increasing": [(5, 1)],
        },
    }
    got = run_caviar_cv._neural_init_vocab(converted)
    assert set(got) == {
        "both_active", "both_walking", "both_inactive", "mixed_active_walking",
        "any_became_active", "any_became_inactive", "any_became_walking", "any_stopped_walking",
    }
    assert "close" not in got
    assert "far" not in got
    assert "coords_missing" not in got
    assert "became_far" not in got
    assert "distance_increasing" not in got
    assert got["both_active"] == [(0, 1)]
    assert got["any_became_active"] == [(0, 1)]


def test_neural_init_vocab_raises_on_missing_transition_relation():
    converted = {
        "relations": {"both_active": [(0, 1)]},
        "transition_relations": {"any_became_active": [(0, 1)]},  # missing the other 3
    }
    with pytest.raises(KeyError, match="transition_relations is missing"):
        run_caviar_cv._neural_init_vocab(converted)


# ---------------------------------------------------------------------------
# `_neural_init_fit_gate` -- wires enumerate_bodies/body_cover into
# relational_search.permutation_null_threshold over the given vocab,
# exactly as `_induce_ec_target` already does for the relational search.
# ---------------------------------------------------------------------------


def test_neural_init_fit_gate_matches_a_direct_permutation_null_threshold_call():
    from relational_search import body_cover, enumerate_bodies, permutation_null_threshold

    vocab = {
        "both_active": [(0, 1), (1, 1), (4, 1), (5, 1)],
        "any_became_active": [(2, 1), (4, 1), (6, 1), (7, 1)],
    }
    facts = [(i, 1) for i in range(8)]
    labels = [True, True, False, False, True, False, True, False]

    got = run_caviar_cv._neural_init_fit_gate(vocab, facts, labels, seed=7)

    bodies, _ = enumerate_bodies(vocab, max_literals=run_caviar_cv.MAX_LITERALS)
    covers = {b: body_cover(b, vocab) for b in bodies}
    expected = permutation_null_threshold(
        bodies, vocab, facts, labels,
        folds=run_caviar_cv.INNER_FOLDS, seed=7,
        n_permutations=run_caviar_cv.NULL_PERMUTATIONS,
        quantile=run_caviar_cv.NULL_QUANTILE, perm_seed=7, covers=covers,
    )
    assert got == expected


# ---------------------------------------------------------------------------
# `_run_init_search` -- mode dispatch for the initiation search only. The
# behavior this guards: mode="relational" is a byte-identical pass-through
# to `_induce_ec_target` (the SAME call this function replaced, inline, at
# `run_fold`'s own call site) -- proven here by comparing against a direct
# `_induce_ec_target` call on the SAME inputs/seed (both are deterministic
# pure functions of their inputs, so two calls agree exactly).
# ---------------------------------------------------------------------------


def test_run_init_search_relational_mode_matches_induce_ec_target_directly():
    relations = {
        "both_active": [(0, 1), (1, 1), (4, 1), (5, 1)],
        "close": [(0, 1), (2, 1), (4, 1), (6, 1)],
    }
    facts = [(i, 1) for i in range(8)]
    labels = [True, True, False, False, True, False, False, True]

    result = run_caviar_cv._run_init_search(
        "relational", train={}, test={}, train_ec_relations=relations,
        init_facts=facts, init_labels=labels, seed=7,
    )
    direct = run_caviar_cv._induce_ec_target(relations, facts, labels, seed=7)

    assert result["clauses"] == direct["clauses"]
    assert result["stop_reason"] == direct["stop_reason"]
    assert result["min_fit"] == direct["min_fit"]
    assert result["null_summary"] == direct["null_summary"]
    assert result["selection_reasons"] == direct["selection_reasons_per_iteration"]
    assert result["predict_clause_test"] is None
    assert result["detector_probe"] is None


# ---------------------------------------------------------------------------
# `run_fold` wiring -- `mode` reaches `_run_init_search` unchanged, and the
# returned `predict_clause_test`/`detector_probe` are threaded into the
# fold's own result dict (and, for a non-None `predict_clause_test`, ACTUALLY
# used for init scoring instead of the shared relational closure). Both
# `_run_init_search` and the term/direct searches are monkeypatched with
# deterministic fakes here -- the small 2-segment fixture below is too tiny
# for the REAL permutation-null search (folds=4 needs >=4 facts), which is
# exactly why: this test is about run_fold's OWN wiring, not search
# correctness (see test_run_caviar_theory_neural_wiring.py for the engine
# dependency-injection tests, and the module docstring for why an
# integration-level run_fold test needs a much larger synthetic corpus this
# suite does not attempt to build).
# ---------------------------------------------------------------------------


def _tiny_two_row_segments():
    seg_a = _segment(
        ("id0", "id1"),
        activity_by_t={0: {"id0": "active", "id1": "active"}, 40: {"id0": "active", "id1": "active"}},
        coords_by_t={0: {"id0": (0, 0), "id1": (3, 4)}, 40: {"id0": (0, 0), "id1": (3, 4)}},
        meeting_ts=[40],
    )
    seg_b = _segment(
        ("id0", "id1"),
        activity_by_t={
            1000: {"id0": "active", "id1": "active"},
            1040: {"id0": "active", "id1": "active"},
            1080: {"id0": "active", "id1": "active"},
        },
        coords_by_t={
            1000: {"id0": (0, 0), "id1": (0, 0)},
            1040: {"id0": (0, 0), "id1": (0, 0)},
            1080: {"id0": (0, 0), "id1": (0, 0)},
        },
        meeting_ts=[1040, 1080],
    )
    return [seg_b], [seg_a]  # (train_segments, test_segments)


def _fake_ec_theory(clauses, stop_reason="max_clauses reached"):
    return {
        "clauses": clauses,
        "stop_reason": stop_reason,
        "min_fit": 0.5,
        "null_summary": {"threshold": 0.5},
        "selection_reasons_per_iteration": ["committed"] if clauses else [],
    }


def test_run_fold_forwards_mode_to_run_init_search_and_defaults_to_relational(monkeypatch):
    train_segments, test_segments = _tiny_two_row_segments()
    calls = []

    def fake_run_init_search(mode, train, test, train_ec_relations, init_facts, init_labels, seed):
        calls.append(mode)
        return {
            "clauses": [], "stop_reason": "no positives remain in the residual",
            "min_fit": 0.5, "null_summary": None, "selection_reasons": [],
            "predict_clause_test": None, "detector_probe": None, "wall_s": 0.0,
        }

    monkeypatch.setattr(run_caviar_cv, "_run_init_search", fake_run_init_search)
    monkeypatch.setattr(
        run_caviar_cv, "_induce_ec_target",
        lambda train_relations, facts, labels, seed: _fake_ec_theory([]),
    )
    monkeypatch.setattr(
        run_caviar_cv, "_induce_direct_theory",
        lambda train_relations, facts, labels, seed: {"clauses": [], "stop_reason": "no positives remain in the residual"},
    )

    result = run_caviar_cv.run_fold(0, train_segments, test_segments, seed=7)
    assert calls == ["relational"]
    assert result["mode"] == "relational"
    assert result["ec"]["init_clauses"] == []
    assert result["ec"]["init_detector_probe"] is None


def test_run_fold_neural_mode_uses_its_own_predict_clause_and_surfaces_detector_probe(monkeypatch):
    train_segments, test_segments = _tiny_two_row_segments()
    used_neural_predict_clause = {"called": False}

    def neural_predict_clause(rule, fact):
        used_neural_predict_clause["called"] = True
        return True

    def fake_run_init_search(mode, train, test, train_ec_relations, init_facts, init_labels, seed):
        assert mode == "neural"
        return {
            "clauses": [("both_active", "close_nn")],
            "stop_reason": "max_clauses reached",
            "min_fit": 0.3,
            "null_summary": {"threshold": 0.3},
            "selection_reasons": ["committed"],
            "predict_clause_test": neural_predict_clause,
            "detector_probe": {
                "clause0:both_active|close_nn": {
                    "probe_test": {"accuracy": 1.0}, "pair_swap_asymmetry": 0.0,
                },
            },
            "wall_s": 0.02,
        }

    monkeypatch.setattr(run_caviar_cv, "_run_init_search", fake_run_init_search)
    monkeypatch.setattr(
        run_caviar_cv, "_induce_ec_target",
        lambda train_relations, facts, labels, seed: _fake_ec_theory([]),
    )
    monkeypatch.setattr(
        run_caviar_cv, "_induce_direct_theory",
        lambda train_relations, facts, labels, seed: {"clauses": [], "stop_reason": "no positives remain in the residual"},
    )

    result = run_caviar_cv.run_fold(0, train_segments, test_segments, seed=7, mode="neural")
    assert result["mode"] == "neural"
    assert result["ec"]["init_clauses"] == [["both_active", "close_nn"]]
    assert result["ec"]["init_min_fit"] == 0.3
    assert result["ec"]["init_detector_probe"] == {
        "clause0:both_active|close_nn": {"probe_test": {"accuracy": 1.0}, "pair_swap_asymmetry": 0.0},
    }
    # The RETURNED predict_clause_test, not the shared relational closure,
    # must be the one actually used for init scoring.
    assert used_neural_predict_clause["called"] is True
