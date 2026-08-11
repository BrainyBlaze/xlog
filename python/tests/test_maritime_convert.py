import tarfile, zipfile, io, os, sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "examples", "maritime_woled"))

from maritime_convert import (
    EPISODE_GAP_S, PAD_S, parse_hle_archive, parse_lle_proximity,
    select_pairs, build_pair_timeline, convert,
)

HLE_LINES = "\n".join([
    "rendezVous|B|A|true|1000|2000",            # non-canonical order on purpose
    "lowSpeed|A| |true|900|2100",
    "lowSpeed|B| |true|900|1500",
    "stopped|B| |farFromPorts|1500|2100",
    "withinArea|A|nearPorts|true|5000|6000",
    "garbage line without pipes",
])

LLE_LINES = "\n".join([
    "proximity|2200|900|2200|true|B|A",
    "proximity|2500|2200|2500|true|A|B",        # adjacent -> must coalesce with prev
    "coord|900|900|A|-4.3|48.1",                # ignored kind
    "proximity|9|bad|row|true|A|B",             # malformed -> counted, not raised
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


def test_hle_parser_canonicalizes_pairs_and_counts_bad_lines(tmp_path):
    out = parse_hle_archive(_tar(tmp_path))
    assert out["intervals"]["rendezVous"][("A", "B")] == [(1000, 2000)]
    assert out["intervals"]["stopped=farFromPorts"][("B",)] == [(1500, 2100)]
    assert "withinArea=true" not in out["intervals"]  # withinArea keys by (mmsi, area)
    assert ("A", "nearPorts") in out["intervals"]["withinArea"]
    assert sum(out["bad_lines"].values()) == 1


def test_lle_proximity_coalesces_adjacent_intervals(tmp_path):
    out = parse_lle_proximity(_zip(tmp_path))
    assert out["proximity"][("A", "B")] == [(900, 2500)]
    assert out["bad_proximity_lines"] == 1


def test_direct_target_uses_clopen_intervals(tmp_path):
    conv = convert(_tar(tmp_path), _zip(tmp_path))
    times = conv["pt_time"]
    pos = {t for t, y in zip(times, conv["is_positive"]) if y}
    assert 1000 in times and 2000 in times
    assert 1000 in pos          # st inclusive
    assert 2000 not in pos      # et exclusive


def test_ec_targets_mark_boundaries_and_dontcare(tmp_path):
    conv = convert(_tar(tmp_path), _zip(tmp_path))
    ec, times = conv["ec"], conv["pt_time"]
    i_start = times.index(1000)
    i_end = times.index(2000)
    assert ec["init_labels"][i_start] is True
    assert ec["term_labels"][i_end] is True
    inside = [i for i, t in enumerate(times) if 1000 < t < 2000]
    assert inside, "there must be interior timepoints (1500 at least)"
    assert all(ec["init_dontcare"][i] for i in inside)
    assert all(ec["term_labels"][i] is False and not ec["term_dontcare"][i] for i in inside)


def test_vocabulary_conjunctions(tmp_path):
    conv = convert(_tar(tmp_path), _zip(tmp_path))
    times, rel = conv["pt_time"], conv["relations"]
    def holds(name, t):
        return times.index(t) in set(rel[name])
    assert holds("both_low_or_stopped", 1000)   # A: lowSpeed, B: lowSpeed
    assert holds("both_low_or_stopped", 1500)   # A: lowSpeed, B: stopped=far
    assert holds("proximity", 1000)
    assert not holds("proximity", 2500)         # et exclusive
    assert holds("became_far", 2500)


def test_episode_split_and_pad(tmp_path):
    # withinArea boundary at 5000/6000 is > PAD_S past the last proximity et (2500)
    conv = convert(_tar(tmp_path), _zip(tmp_path))
    assert 5000 not in conv["pt_time"]
    assert conv["counts"]["n_dropped_outside_pad"] >= 1
    for lo, hi in conv["segments"]:
        seg_times = conv["pt_time"][lo:hi]
        assert all(b - a <= EPISODE_GAP_S for a, b in zip(seg_times, seg_times[1:]))


# ---------------------------------------------------------------------------
# ceiling_probe: committed derivation of the vocabulary ceiling (deep-review
# finding #4) — definitional-body pointwise stats, FP decomposition, and the
# archive-A nested/overlap interval census. Torch-free, synthetic only.
# ---------------------------------------------------------------------------

import ceiling_probe


def _probe_converted():
    """Hand-built converted-corpus shape: one positive pair (rows 0-4, one
    segment) + one negative pair (rows 5-6, one segment). The definitional
    body covers rows 0,1 (a 100 s run), 3,4 (a 300 s run) and 5,6 (the
    negative pair); row 2 is gold-positive but uncovered (an FN)."""
    body_rows = [0, 1, 3, 4, 5, 6]
    return {
        "pairs": [("P1a", "P1b"), ("N1a", "N1b")],
        "pt_pair_index": [0, 0, 0, 0, 0, 1, 1],
        "pt_time": [0, 100, 200, 1000, 1300, 0, 50],
        "segments": [(0, 5), (5, 7)],
        "relations": {
            "proximity": body_rows + [2],
            "both_low_or_stopped": list(body_rows),
            "both_open_sea": list(body_rows),
        },
        "is_positive": [True, False, True, True, False, False, False],
        "counts": {"n_positive_pairs": 1, "n_negative_pairs": 1, "n_pairs": 2, "n_pt": 7},
    }


def test_ceiling_probe_pointwise_stats_hand_computed():
    out = ceiling_probe.body_pointwise_stats(_probe_converted())
    assert out["tp"] == 2 and out["fp"] == 4 and out["fn"] == 1
    assert out["precision"] == 2 / 6
    assert out["recall"] == 2 / 3
    assert out["body"] == ["proximity", "both_low_or_stopped", "both_open_sea"]


def test_ceiling_probe_fp_decomposition_by_cause():
    out = ceiling_probe.fp_decomposition(_probe_converted())
    # row 1: FP inside the 100 s (< 240 s) predicted run of a positive pair;
    # row 4: FP inside the 300 s run (other); rows 5,6: FPs of the negative pair.
    assert out["fp_negative_pairs"] == 2
    assert out["fp_positive_pairs_short_run"] == 1
    assert out["fp_positive_pairs_other"] == 1
    assert out["fp_total"] == 4
    assert out["n_predicted_runs"] == 3
    assert out["n_short_runs"] == 2  # the 100 s run and the negative pair's 50 s run


def test_ceiling_probe_census_flags_nested_interval():
    hle = {"intervals": {"rendezVous": {("A", "B"): [(0, 100), (10, 20)]}}}
    out = ceiling_probe.nested_interval_census(hle)
    key = out["per_key"]["rendezVous"]
    assert key["n_strictly_nested"] == 1
    assert key["n_overlapping"] == 2
    assert key["n_duplicates"] == 0
    assert key["n_entities_with_nested"] == 1
    assert key["nested_examples"] == [{"entity": ["A", "B"], "st": 10, "et": 20}]
    assert out["totals"]["n_strictly_nested"] == 1


def test_ceiling_probe_census_same_start_and_duplicates_and_clean():
    hle = {"intervals": {"lowSpeed": {
        ("V1",): [(0, 100), (0, 50)],     # same st, shorter et: strictly nested
        ("V2",): [(0, 10), (0, 10)],      # identical duplicates: NOT nested
        ("V3",): [(0, 10), (10, 20)],     # adjacent half-open: no overlap at all
    }}}
    key = ceiling_probe.nested_interval_census(hle)["per_key"]["lowSpeed"]
    assert key["n_strictly_nested"] == 1
    assert key["n_duplicates"] == 1
    assert key["n_overlapping"] == 4    # V1's two + V2's two; V3 contributes none
    assert key["n_entities_with_nested"] == 1


def test_ceiling_probe_main_end_to_end_on_synthetic_archives(tmp_path):
    # Hand-derived on the module fixtures: kept timepoints for pair (A, B)
    # are 900/1000/1500/2000/2100/2500; the body covers 900-2000 (proximity
    # holds to 2100-exclusive, both_low_or_stopped to 2100-exclusive, open
    # sea everywhere), gold rendezVous covers 1000/1500 -> tp=2, the body's
    # 900/2000 rows are fp, fn=0; the single predicted run spans 1100 s
    # (>= 240), and the fixture archive has no nested/overlapping HLE
    # intervals.
    out_path = tmp_path / "probe.json"
    rc = ceiling_probe.main([
        "--tar", _tar(tmp_path), "--zip", _zip(tmp_path), "--out", str(out_path),
    ])
    assert rc == 0
    import json
    report = json.loads(out_path.read_text(encoding="utf-8"))
    p = report["pointwise"]
    assert (p["tp"], p["fp"], p["fn"]) == (2, 2, 0)
    d = report["fp_decomposition"]
    assert d["fp_total"] == 2
    assert d["fp_positive_pairs_other"] == 2
    assert d["fp_positive_pairs_short_run"] == 0
    assert d["fp_negative_pairs"] == 0
    assert d["n_predicted_runs"] == 1
    assert report["nested_interval_census"]["totals"]["n_strictly_nested"] == 0
    assert report["nested_interval_census"]["totals"]["n_overlapping"] == 0
    assert "rendezVous" in report["nested_interval_census"]["per_key"]


def test_negative_pair_subsample_is_deterministic():
    hle = {"intervals": {"rendezVous": {("A", "B"): [(0, 10)]}}}
    pool_pairs = {(f"N{i:03d}", f"M{i:03d}"): [(0, 5)] for i in range(1500)}
    prox = {"proximity": {("A", "B"): [(0, 10)], **pool_pairs}}
    sel1 = select_pairs(hle, prox)
    sel2 = select_pairs(hle, prox)
    assert sel1 == sel2
    assert len(sel1["negative_pairs"]) <= 604
    assert sel1["n_negative_pool"] == 1500
