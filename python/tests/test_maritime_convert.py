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


def test_negative_pair_subsample_is_deterministic():
    hle = {"intervals": {"rendezVous": {("A", "B"): [(0, 10)]}}}
    pool_pairs = {(f"N{i:03d}", f"M{i:03d}"): [(0, 5)] for i in range(1500)}
    prox = {"proximity": {("A", "B"): [(0, 10)], **pool_pairs}}
    sel1 = select_pairs(hle, prox)
    sel2 = select_pairs(hle, prox)
    assert sel1 == sel2
    assert len(sel1["negative_pairs"]) <= 604
    assert sel1["n_negative_pool"] == 1500
