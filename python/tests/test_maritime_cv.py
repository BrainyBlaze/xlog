"""Unit tests for `examples/maritime_woled/run_maritime_cv.py` -- CPU, no
real archives (the real-data run is a separate, manual job; see
`docs/experiments/maritime/README.md`'s pre-registration). Follows the
style of `test_caviar_cv.py` (CLI parse tests, deterministic fold
assignment) and `test_maritime_convert.py` (synthetic tar/zip mini-archive
fixtures)."""

import io
import json
import os
import sys
import tarfile
import zipfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "examples", "maritime_woled"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "examples", "caviar_woled"))

import run_maritime_cv  # noqa: E402


# ---------------------------------------------------------------------------
# Fold assignment: pairs are atoms, LPT stratification, deterministic
# ---------------------------------------------------------------------------


def test_stratified_pair_folds_returns_one_fold_per_pair():
    # 6 positive pairs, 6 negative pairs, 3 folds.
    pos_counts = [100, 40, 30, 20, 10, 5]
    pt_counts = [200, 90, 70, 50, 30, 15, 400, 300, 200, 100, 50, 25]
    folds = run_maritime_cv.stratified_pair_folds(pos_counts, pt_counts, 6, 3)
    assert len(folds) == 12
    assert all(f in (0, 1, 2) for f in folds)


def test_stratified_pair_folds_every_fold_has_positive_and_negative_pairs():
    pos_counts = [100, 40, 30, 20, 10, 5]
    pt_counts = [200, 90, 70, 50, 30, 15, 400, 300, 200, 100, 50, 25]
    folds = run_maritime_cv.stratified_pair_folds(pos_counts, pt_counts, 6, 3)
    for fold in range(3):
        members = [i for i, f in enumerate(folds) if f == fold]
        assert any(i < 6 for i in members), f"fold {fold} has no positive pair"
        assert any(i >= 6 for i in members), f"fold {fold} has no negative pair"


def test_stratified_pair_folds_positive_balance_bounded_by_largest_pair():
    # Extreme concentration mirroring the real corpus: top pair carries ~33%.
    pos_counts = [1194, 370, 320, 271, 150, 120, 100, 90, 80, 70, 60, 50]
    pt_counts = list(pos_counts) + [500, 400, 300, 200, 100, 50]
    n_pos = len(pos_counts)
    folds = run_maritime_cv.stratified_pair_folds(pos_counts, pt_counts, n_pos, 5)
    sums = [0] * 5
    for i, f in enumerate(folds[:n_pos]):
        sums[f] += pos_counts[i]
    assert max(sums) - min(sums) <= max(pos_counts)


def test_stratified_pair_folds_negative_balance_bounded_by_largest_pair():
    pos_counts = [10, 9, 8, 7, 6, 5]
    neg_counts = [1652, 900, 400, 300, 200, 100, 50, 25, 10]
    pt_counts = [20, 18, 16, 14, 12, 10] + neg_counts
    folds = run_maritime_cv.stratified_pair_folds(pos_counts, pt_counts, 6, 3)
    sums = [0] * 3
    for j, f in enumerate(folds[6:]):
        sums[f] += neg_counts[j]
    assert max(sums) - min(sums) <= max(neg_counts)


def test_stratified_pair_folds_deterministic_and_exact_lpt():
    pos_counts = [100, 40, 30, 20, 10, 5]
    pt_counts = [0, 0, 0, 0, 0, 0, 400, 300, 200, 100, 50, 25]
    a = run_maritime_cv.stratified_pair_folds(pos_counts, pt_counts, 6, 3)
    b = run_maritime_cv.stratified_pair_folds(pos_counts, pt_counts, 6, 3)
    assert a == b
    # Hand-traced LPT: positives sorted desc 100,40,30,20,10,5 ->
    # folds 0,1,2, then loads (100,40,30): 20->fold2 (50), 10->fold1 (50),
    # 5->fold1? loads now (100,50,50) -> tie between folds 1,2 -> lowest
    # index wins: 5->fold1 (55).
    assert a[:6] == [0, 1, 2, 2, 1, 1]
    # Negatives sorted desc 400,300,200,100,50,25 -> folds 0,1,2, loads
    # (400,300,200): 100->fold2 (300), 50->fold1? loads (400,300,300) ->
    # tie 1,2 -> fold1 (350), 25->fold2 (325).
    assert a[6:] == [0, 1, 2, 2, 1, 2]


def test_stratified_pair_folds_rejects_too_few_pairs_or_folds():
    with pytest.raises(ValueError):
        run_maritime_cv.stratified_pair_folds([5, 5], [5, 5, 9, 9], 2, 1)
    with pytest.raises(ValueError):
        # only 2 positive pairs for 3 folds
        run_maritime_cv.stratified_pair_folds([5, 5], [5, 5, 9, 9, 9], 2, 3)
    with pytest.raises(ValueError):
        # only 2 negative pairs for 3 folds
        run_maritime_cv.stratified_pair_folds([5, 5, 5], [5, 5, 5, 9, 9], 3, 3)


def test_pair_counts_from_converted_corpus():
    converted = {
        "pt_pair_index": [0, 0, 0, 1, 1, 2, 2, 2, 2],
        "is_positive": [True, False, True, False, False, False, True, False, False],
    }
    pos_counts, pt_counts = run_maritime_cv.pair_counts(converted, n_pairs=3)
    assert pt_counts == [3, 2, 4]
    assert pos_counts == [2, 0, 1]


# ---------------------------------------------------------------------------
# Synthetic mini-corpus (shape of `maritime_convert.convert`'s output):
# planted rule is_positive == r1 AND r2, perfectly reconstructible --
# positive pairs fire both at t % 6 == 0, negative pairs never fire r2.
# ---------------------------------------------------------------------------


def _synthetic_converted(n_pos=6, n_neg=6, pts_per_pair=30):
    pairs = [(f"P{i:02d}a", f"P{i:02d}b") for i in range(n_pos)] + [
        (f"N{j:02d}a", f"N{j:02d}b") for j in range(n_neg)
    ]
    pt_pair_index, pt_time, is_positive, segments = [], [], [], []
    relations = {"r1": [], "r2": [], "r3": []}
    for pair_idx in range(n_pos + n_neg):
        lo = len(pt_time)
        for t in range(pts_per_pair):
            pt = len(pt_time)
            pt_pair_index.append(pair_idx)
            pt_time.append(1000 * pair_idx + t)
            r1 = t % 2 == 0
            r2 = (t % 3 == 0) and pair_idx < n_pos
            if r1:
                relations["r1"].append(pt)
            if r2:
                relations["r2"].append(pt)
            if t % 5 == 0:
                relations["r3"].append(pt)
            is_positive.append(r1 and r2)
        segments.append((lo, len(pt_time)))
    return {
        "pairs": pairs,
        "pt_pair_index": pt_pair_index,
        "pt_time": pt_time,
        "segments": segments,
        "relations": relations,
        "is_positive": is_positive,
        "counts": {
            "n_positive_pairs": n_pos,
            "n_negative_pairs": n_neg,
            "n_pairs": n_pos + n_neg,
            "n_pt": len(pt_time),
        },
    }


# ---------------------------------------------------------------------------
# Fold slicing: pt-level partition induced by the pair-level assignment
# ---------------------------------------------------------------------------


def test_fold_pt_indices_partition_and_pair_atomicity():
    conv = _synthetic_converted()
    pos_counts, pt_counts = run_maritime_cv.pair_counts(conv, len(conv["pairs"]))
    folds = run_maritime_cv.stratified_pair_folds(
        pos_counts[: conv["counts"]["n_positive_pairs"]], pt_counts,
        conv["counts"]["n_positive_pairs"], 3,
    )
    seen = set()
    for fold in range(3):
        train_pts, test_pts = run_maritime_cv.fold_pt_indices(conv, folds, fold)
        assert not (set(train_pts) & set(test_pts))
        assert sorted(set(train_pts) | set(test_pts)) == list(range(conv["counts"]["n_pt"]))
        # pair atomicity: a pair's rows are entirely on one side
        test_pairs = {conv["pt_pair_index"][i] for i in test_pts}
        train_pairs = {conv["pt_pair_index"][i] for i in train_pts}
        assert not (test_pairs & train_pairs)
        seen |= set(test_pts)
    assert seen == set(range(conv["counts"]["n_pt"]))


def test_restrict_relations_keeps_all_names_and_only_member_pts():
    relations = {"a": [0, 1, 5], "b": [], "c": [2, 5]}
    out = run_maritime_cv.restrict_relations(relations, {0, 2, 5})
    assert sorted(out) == ["a", "b", "c"]
    assert out["a"] == [0, 5]
    assert out["b"] == []
    assert out["c"] == [2, 5]


# ---------------------------------------------------------------------------
# Interval-level scoring (pre-registered definition: maximal runs within a
# segment, overlap >= 1 pt row)
# ---------------------------------------------------------------------------


def test_interval_prf1_run_overlap_matching():
    #                0      1      2     3     4      5      6      7     8     9
    gold = [False, False, True, True, True, False, False, True, False, False]
    pred = [False, False, False, True, False, False, False, False, True, True]
    out = run_maritime_cv.interval_prf1(pred, gold, [(0, 10)])
    # gold runs: [2,5), [7,8); pred runs: [3,4), [8,10)
    assert out["n_gold_intervals"] == 2
    assert out["n_pred_intervals"] == 2
    assert out["n_matched_gold"] == 1
    assert out["n_matched_pred"] == 1
    assert out["precision"] == pytest.approx(0.5)
    assert out["recall"] == pytest.approx(0.5)
    assert out["f1"] == pytest.approx(0.5)


def test_interval_prf1_never_bridges_segments():
    gold = [False, False, True, True, False, False]
    pred = list(gold)
    out = run_maritime_cv.interval_prf1(pred, gold, [(0, 3), (3, 6)])
    # the True run at rows 2,3 straddles the segment boundary: two intervals
    assert out["n_gold_intervals"] == 2
    assert out["n_pred_intervals"] == 2
    assert out["f1"] == pytest.approx(1.0)


def test_interval_prf1_empty_predictions_degenerate():
    out = run_maritime_cv.interval_prf1([False] * 4, [False, True, True, False], [(0, 4)])
    assert out["n_pred_intervals"] == 0
    assert out["precision"] == 0.0
    assert out["recall"] == 0.0
    assert out["f1"] == 0.0


# ---------------------------------------------------------------------------
# run_fold: permutation-null gate + relational search + point/interval
# scoring on the held-out fold
# ---------------------------------------------------------------------------


def test_run_fold_recovers_planted_rule_and_scores_heldout():
    pytest.importorskip("torch")
    conv = _synthetic_converted()
    n_pos = conv["counts"]["n_positive_pairs"]
    pos_counts, pt_counts = run_maritime_cv.pair_counts(conv, len(conv["pairs"]))
    folds = run_maritime_cv.stratified_pair_folds(pos_counts[:n_pos], pt_counts, n_pos, 3)
    record = run_maritime_cv.run_fold(0, conv, folds, seed=7, min_new_covered=2, tie_tolerance=None)

    assert record["fold"] == 0
    assert ["r1", "r2"] in record["clauses"]
    assert record["scoring"]["point"]["f1"] == pytest.approx(1.0)
    assert record["scoring"]["interval"]["f1"] == pytest.approx(1.0)
    # provenance inside the fold record
    assert record["min_fit"] == record["null_summary"]["threshold"]
    assert record["n_test_pt"] + record["n_train_pt"] == conv["counts"]["n_pt"]
    assert record["test_pairs"], "fold record must name its held-out pairs"
    assert record["stop_reason"]
    assert record["iterations"]


# ---------------------------------------------------------------------------
# Verifier smoke: pinned md5 + pinned counts + hard invariants
# ---------------------------------------------------------------------------

HLE_LINES = "\n".join([
    "rendezVous|B|A|true|1000|2000",
    "lowSpeed|A| |true|900|2100",
    "lowSpeed|B| |true|900|1500",
    "stopped|B| |farFromPorts|1500|2100",
])

LLE_LINES = "\n".join([
    "proximity|2200|900|2200|true|B|A",
    "proximity|9000|100|9000|true|C|D",
    "proximity|9000|100|9000|true|E|F",
])


def _tar(tmp_path):
    p = tmp_path / "hle.tar.gz"
    data = HLE_LINES.encode()
    with tarfile.open(p, "w:gz") as tf:
        info = tarfile.TarInfo("Maritime Composite Events/CEs/recognised_CEs.csv")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    return str(p)


def _zip(tmp_path):
    p = tmp_path / "lle.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("brest_critical.csv", LLE_LINES)
    return str(p)


def test_verify_smoke_rejects_synthetic_archives(tmp_path):
    from maritime_convert import convert

    tar_path, zip_path = _tar(tmp_path), _zip(tmp_path)
    converted = convert(tar_path, zip_path)
    report = run_maritime_cv.verify_smoke(converted, tar_path, zip_path)
    assert report["ok"] is False
    assert report["md5"]["ok"] is False
    assert report["counts"]["ok"] is False
    # structural invariants still hold on this tiny, well-formed corpus
    assert report["hard_invariants"]["pair_contiguity_ok"] is True


# ---------------------------------------------------------------------------
# main(): end-to-end cycle on synthetic archives (deep-review finding #7) —
# the verify-gate abort, the --skip-verify warning stamp, --smoke, and the
# micro aggregation, all previously untested.
# ---------------------------------------------------------------------------


def _multi_pair_archives(tmp_path, n_pos=6, n_neg=6):
    """Synthetic tar/zip with enough pairs for real fold assignment:
    n_pos positive pairs (rendezVous + both vessels lowSpeed + proximity)
    and n_neg negative pairs (proximity only)."""
    hle_lines, lle_lines = [], []
    for i in range(n_pos):
        a, b = f"P{i:02d}a", f"P{i:02d}b"
        hle_lines += [
            f"rendezVous|{a}|{b}|true|1000|2000",
            f"lowSpeed|{a}| |true|900|2100",
            f"lowSpeed|{b}| |true|900|2100",
        ]
        lle_lines.append(f"proximity|2200|900|2200|true|{a}|{b}")
    for j in range(n_neg):
        a, b = f"N{j:02d}a", f"N{j:02d}b"
        lle_lines.append(f"proximity|2200|900|2200|true|{a}|{b}")

    tar_p = tmp_path / "multi.tar.gz"
    data = "\n".join(hle_lines).encode()
    with tarfile.open(tar_p, "w:gz") as tf:
        info = tarfile.TarInfo("Maritime Composite Events/CEs/recognised_CEs.csv")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    zip_p = tmp_path / "multi.zip"
    with zipfile.ZipFile(zip_p, "w") as z:
        z.writestr("brest_critical.csv", "\n".join(lle_lines))
    return str(tar_p), str(zip_p)


def test_main_aborts_when_verify_gate_fails(tmp_path):
    # Without --skip-verify the pre-run gate must reject synthetic archives
    # (md5 pins cannot match) BEFORE any search runs: exit 2, no output JSON.
    tar_p, zip_p = _multi_pair_archives(tmp_path)
    out = tmp_path / "out.json"
    rc = run_maritime_cv.main(["--tar", tar_p, "--zip", zip_p, "--out", str(out)])
    assert rc == 2
    assert not out.exists()


def test_main_end_to_end_smoke_on_synthetic_archives(tmp_path):
    pytest.importorskip("torch")
    tar_p, zip_p = _multi_pair_archives(tmp_path)
    out = tmp_path / "out.json"
    rc = run_maritime_cv.main([
        "--tar", tar_p, "--zip", zip_p, "--out", str(out),
        "--smoke", "--skip-verify", "--folds", "3",
    ])
    assert rc == 0

    result = json.loads(out.read_text(encoding="utf-8"))
    # the skip stamp can never masquerade as a passed gate
    assert result["verify_smoke"]["skipped"] is True
    assert "warning" in result["verify_smoke"]
    assert result["params"]["smoke"] is True
    assert len(result["folds"]) == 3

    # micro aggregation: summed per-fold counts ARE the micro counts, and
    # micro P/R/F1 recompute from those sums
    for key in ("tp", "fp", "fn"):
        assert result["micro"]["point"][key] == sum(
            f["scoring"]["point"][key] for f in result["folds"]
        )
    tp = result["micro"]["point"]["tp"]
    fp = result["micro"]["point"]["fp"]
    fn = result["micro"]["point"]["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    assert result["micro"]["point"]["precision"] == pytest.approx(precision)
    assert result["micro"]["point"]["recall"] == pytest.approx(recall)
    assert result["micro"]["point"]["f1"] == pytest.approx(f1)

    # interval micro aggregation sums the per-fold interval counts the same way
    for key in ("n_matched_gold", "n_gold_intervals", "n_matched_pred", "n_pred_intervals"):
        assert result["micro"]["interval"][key] == sum(
            f["scoring"]["interval"][key] for f in result["folds"]
        )

    # per-fold spread block mirrors the fold records
    fold_f1s = [f["scoring"]["point"]["f1"] for f in result["folds"]]
    assert result["per_fold_point_f1"]["values"] == fold_f1s
    assert result["per_fold_point_f1"]["min"] == min(fold_f1s)
    assert result["per_fold_point_f1"]["max"] == max(fold_f1s)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

REQUIRED_ARGS = ["--tar", "a.tar.gz", "--zip", "b.zip", "--out", "o.json"]


def test_parse_args_defaults_match_pre_registration():
    args = run_maritime_cv.parse_args(REQUIRED_ARGS)
    assert args.folds == 5
    assert args.seed == 7
    assert args.min_new_covered == 2
    assert args.tie_tolerance is None
    assert args.smoke is False
    assert args.skip_verify is False


def test_parse_args_requires_tar_zip_out():
    with pytest.raises(SystemExit):
        run_maritime_cv.parse_args(["--zip", "b.zip", "--out", "o.json"])
    with pytest.raises(SystemExit):
        run_maritime_cv.parse_args(["--tar", "a.tar.gz", "--out", "o.json"])
    with pytest.raises(SystemExit):
        run_maritime_cv.parse_args(["--tar", "a.tar.gz", "--zip", "b.zip"])


def test_module_import_is_torch_free():
    assert not hasattr(run_maritime_cv, "torch")
