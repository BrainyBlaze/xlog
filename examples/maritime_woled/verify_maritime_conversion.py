"""Real-data verification for `maritime_convert.convert` against the actual
Brest AIS exports: archive A (the HLE/composite-event `.tar.gz`) and archive
B (the LLE/critical-event `.zip`, a ~1GB proximity stream).

This is not a unit test with synthetic fixtures -- it runs the full
converter on the real archives and checks the invariants that only real
data can exercise: archive integrity (md5), episode/segment/pair structural
integrity, EC label bookkeeping, and -- the analogue of a frame-by-frame
correctness proof -- how well the two supervision-relevant relations
(`proximity`, `both_low_or_stopped`) actually cover the positive
(rendezVous-holding) pair-times. The gold `rendezVous` label was produced by
an RTEC rule intersecting exactly those two conditions
(`(lowSpeed ∪ stopped=farFromPorts) both ∩ proximity`), so if the
converter's pair-time sampling and relation derivation are faithful, both
coverage fractions should sit near 1.0; anything less says either the
converter has a bug or the timepoint sampling loses some pts at interval
boundaries, and either way it belongs in the report, not swept under an
`assert x == 1.0`.

Usage:
    py -3.13 examples/maritime_woled/verify_maritime_conversion.py \\
        --tar <path to MaritimeCompositeEvents.tar.gz> \\
        --zip <path to brest_critical.zip> \\
        --out MARITIME_VERIFY.json

Exit code is 0 iff every HARD invariant holds; SOFT numbers (the alignment
fractions, totals) are reported either way and never affect the exit code
by themselves, except that `alignment_warning` is set (and printed loudly)
when either fraction falls below the 0.95 threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from bisect import bisect_left, bisect_right
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from maritime_convert import convert, parse_hle_archive  # noqa: E402

EXPECTED_TAR_MD5 = "05e23621b87fcec211a7ff4ed4397b94"
EXPECTED_ZIP_MD5 = "0b239b8eb212dc433e67de6a599b2d10"
ALIGNMENT_WARNING_THRESHOLD = 0.95
MAX_VIOLATIONS_REPORTED = 20
N_WORST_INTERVALS = 10


def _md5(path: str, chunk_size: int = 1 << 20) -> str:
    """Streamed md5 -- archive B is ~1GB, never read whole into memory."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _pair_range(pt_pair_index: list[int], pair_idx: int) -> tuple[int, int]:
    """[lo, hi) global row range for one pair index. `pt_pair_index` is
    non-decreasing by construction (pairs are appended in a fixed order,
    never interleaved), so both ends are a binary search rather than a
    linear scan."""
    lo = bisect_left(pt_pair_index, pair_idx)
    hi = bisect_right(pt_pair_index, pair_idx)
    return lo, hi


def _first_non_monotonic(seq: list[int]) -> int | None:
    """Index `i` where `seq[i] < seq[i - 1]`, else None."""
    for i in range(1, len(seq)):
        if seq[i] < seq[i - 1]:
            return i
    return None


def _check_pair_contiguity(pt_pair_index: list[int]) -> dict:
    """HARD invariant 3a: every pair's pt rows form one contiguous block.
    Equivalent to `pt_pair_index` being non-decreasing end to end."""
    bad_i = _first_non_monotonic(pt_pair_index)
    if bad_i is None:
        return {"ok": True, "first_violation": None}
    return {
        "ok": False,
        "first_violation": {
            "index": bad_i,
            "pair_idx_at_index": pt_pair_index[bad_i],
            "pair_idx_before": pt_pair_index[bad_i - 1],
        },
    }


def _check_segment_pair_boundaries(segments: list[tuple[int, int]], pt_pair_index: list[int]) -> list[dict]:
    """HARD invariant 3b: no segment (episode) spans two pairs. Given pair
    contiguity already holds, checking a segment's first and last row share
    a pair index is sufficient (a non-decreasing sequence equal at both
    ends of a range cannot change value inside that range)."""
    violations = []
    for seg_id, (lo, hi) in enumerate(segments):
        if hi <= lo:
            continue
        first_pair = pt_pair_index[lo]
        last_pair = pt_pair_index[hi - 1]
        if first_pair != last_pair:
            violations.append({
                "segment": seg_id, "lo": lo, "hi": hi,
                "first_pair_idx": first_pair, "last_pair_idx": last_pair,
            })
            if len(violations) >= MAX_VIOLATIONS_REPORTED:
                break
    return violations


def _check_ec_label_consistency(segments: list[tuple[int, int]], ec: dict) -> list[dict]:
    """HARD invariant 4: per segment, count(init_labels) - count(term_labels)
    is 0 (every initiation matched by a termination) or 1 (the segment ends
    mid-interval, i.e. the last initiation has no termination yet)."""
    init, term = ec["init_labels"], ec["term_labels"]
    violations = []
    for seg_id, (lo, hi) in enumerate(segments):
        n_init = sum(init[lo:hi])
        n_term = sum(term[lo:hi])
        diff = n_init - n_term
        if diff not in (0, 1):
            violations.append({
                "segment": seg_id, "lo": lo, "hi": hi,
                "n_init": n_init, "n_term": n_term, "diff": diff,
            })
            if len(violations) >= MAX_VIOLATIONS_REPORTED:
                break
    return violations


def _segment_id_of(segment_starts: list[int], idx: int) -> int:
    """Which segment (by position in `segments`) owns global row `idx`.
    `segment_starts` is sorted and segments partition [0, n_pt) with no
    gaps, so a single bisect over the starts suffices."""
    return bisect_right(segment_starts, idx) - 1


def _check_episode_boundaries_and_alignment(
    pairs: list[tuple[str, str]],
    n_positive_pairs: int,
    pt_pair_index: list[int],
    pt_time: list[int],
    segments: list[tuple[int, int]],
    is_prox: list[bool],
    is_blos: list[bool],
    hle_rendez: dict,
) -> tuple[list[dict], list[dict]]:
    """HARD invariant 2 + the per-interval data behind the SOFT alignment
    worst-offenders list, computed together since both need, for every gold
    `rendezVous` interval of every positive pair, the exact global row range
    `[g_lo, g_hi)` that interval's own kept pair-timepoints occupy.

    Returns (boundary_violations, all_interval_records). A violation means
    the interval's own timepoints land in more than one segment -- i.e. the
    gold interval spans an episode boundary, which would make it impossible
    to score under a single per-episode EC state machine."""
    segment_starts = [lo for lo, _hi in segments]
    boundary_violations: list[dict] = []
    all_records: list[dict] = []

    for p in range(n_positive_pairs):
        pair = pairs[p]
        lo_p, hi_p = _pair_range(pt_pair_index, p)
        local_times = pt_time[lo_p:hi_p]
        rendez_ivs = hle_rendez.get(pair, [])

        for st, et in rendez_ivs:
            i_st = bisect_left(local_times, st)
            i_et = bisect_left(local_times, et)
            g_lo, g_hi = lo_p + i_st, lo_p + i_et
            n_pts = g_hi - g_lo
            if n_pts <= 0:
                # No kept timepoint of this pair falls in [st, et) -- can
                # only mean the interval boundaries themselves were not
                # matched exactly; note it as a boundary violation rather
                # than silently skipping.
                boundary_violations.append({
                    "pair": list(pair), "st": st, "et": et,
                    "reason": "no pt rows found in interval (boundary timepoint mismatch)",
                })
                continue

            seg_lo = _segment_id_of(segment_starts, g_lo)
            seg_hi = _segment_id_of(segment_starts, g_hi - 1)
            if seg_lo != seg_hi:
                boundary_violations.append({
                    "pair": list(pair), "st": st, "et": et,
                    "segment_of_start": seg_lo, "segment_of_end": seg_hi,
                })

            n_prox = sum(1 for i in range(g_lo, g_hi) if is_prox[i])
            n_blos = sum(1 for i in range(g_lo, g_hi) if is_blos[i])
            all_records.append({
                "pair": list(pair), "st": st, "et": et, "n_pts": n_pts,
                "proximity_fraction": n_prox / n_pts,
                "both_low_or_stopped_fraction": n_blos / n_pts,
            })

    return boundary_violations, all_records


def _worst(records: list[dict], key: str, n: int) -> list[dict]:
    ranked = sorted(records, key=lambda r: r[key])
    return [
        {"pair": r["pair"], "st": r["st"], "et": r["et"], "n_pts": r["n_pts"], "fraction": r[key]}
        for r in ranked[:n]
    ]


def run(tar_path: str, zip_path: str) -> dict:
    tar_md5 = _md5(tar_path)
    zip_md5 = _md5(zip_path)
    tar_md5_ok = tar_md5 == EXPECTED_TAR_MD5
    zip_md5_ok = zip_md5 == EXPECTED_ZIP_MD5

    t0 = time.perf_counter()
    result = convert(tar_path, zip_path)
    convert_wall_time_s = time.perf_counter() - t0

    t1 = time.perf_counter()

    pairs = result["pairs"]
    pt_pair_index = result["pt_pair_index"]
    pt_time = result["pt_time"]
    segments = result["segments"]
    relations = result["relations"]
    is_positive = result["is_positive"]
    ec = result["ec"]
    counts = result["counts"]
    n_pt = len(pt_time)
    n_positive_pairs = counts["n_positive_pairs"]

    # Boolean covers for the two alignment relations, built once (O(n_pt))
    # rather than repeated set-membership tests per query.
    is_prox = [False] * n_pt
    for i in relations["proximity"]:
        is_prox[i] = True
    is_blos = [False] * n_pt
    for i in relations["both_low_or_stopped"]:
        is_blos[i] = True

    contiguity = _check_pair_contiguity(pt_pair_index)
    segment_pair_violations = (
        _check_segment_pair_boundaries(segments, pt_pair_index) if contiguity["ok"] else []
    )
    ec_violations = _check_ec_label_consistency(segments, ec)

    hle_rendez = parse_hle_archive(tar_path)["intervals"].get("rendezVous", {})
    boundary_violations, interval_records = _check_episode_boundaries_and_alignment(
        pairs, n_positive_pairs, pt_pair_index, pt_time, segments,
        is_prox, is_blos, hle_rendez,
    )

    n_positive_pts = sum(1 for p in is_positive if p)
    prox_covered = sum(1 for pos, pr in zip(is_positive, is_prox) if pos and pr)
    blos_covered = sum(1 for pos, bl in zip(is_positive, is_blos) if pos and bl)
    # An empty denominator reports 0.0, NEVER a vacuous 1.0: a zero-positive
    # conversion is a hard failure (positive_support below), and its
    # alignment must look broken, not perfect.
    proximity_fraction = prox_covered / n_positive_pts if n_positive_pts else 0.0
    both_low_or_stopped_fraction = blos_covered / n_positive_pts if n_positive_pts else 0.0

    # HARD invariant 5 (positive support): the corpus this verifier exists
    # for HAS positives; a conversion with none means an upstream
    # regression (e.g. a fluent-key change in parse_hle_archive), and every
    # positive-side check above passed vacuously. Fail loudly by name.
    positive_support_ok = n_positive_pts > 0 and n_positive_pairs > 0
    positive_support = {
        "n_positive_pts": n_positive_pts,
        "n_positive_pairs": n_positive_pairs,
        "ok": positive_support_ok,
        "reason": None if positive_support_ok else (
            f"zero-positive conversion: n_positive_pts={n_positive_pts}, "
            f"n_positive_pairs={n_positive_pairs} — every positive-side check "
            "(alignment fractions, episode boundaries) passed vacuously, which "
            "verifies nothing; an upstream parsing/labeling regression is the "
            "likely cause"
        ),
    }

    alignment_warning = None
    if proximity_fraction < ALIGNMENT_WARNING_THRESHOLD or both_low_or_stopped_fraction < ALIGNMENT_WARNING_THRESHOLD:
        alignment_warning = (
            f"alignment below {ALIGNMENT_WARNING_THRESHOLD}: "
            f"proximity_fraction={proximity_fraction:.4f}, "
            f"both_low_or_stopped_fraction={both_low_or_stopped_fraction:.4f}"
        )

    verify_wall_time_s = time.perf_counter() - t1

    hard = {
        "md5_ok": tar_md5_ok and zip_md5_ok,
        "episode_boundary_ok": len(boundary_violations) == 0,
        "episode_boundary_violations": boundary_violations[:MAX_VIOLATIONS_REPORTED],
        "n_episode_boundary_violations": len(boundary_violations),
        "pair_contiguity_ok": contiguity["ok"],
        "pair_contiguity_first_violation": contiguity["first_violation"],
        "segment_pair_boundary_ok": len(segment_pair_violations) == 0,
        "segment_pair_boundary_violations": segment_pair_violations,
        "ec_label_consistency_ok": len(ec_violations) == 0,
        "ec_label_consistency_violations": ec_violations,
        "positive_support_ok": positive_support_ok,
    }
    hard_ok = (
        hard["md5_ok"]
        and hard["episode_boundary_ok"]
        and hard["pair_contiguity_ok"]
        and hard["segment_pair_boundary_ok"]
        and hard["ec_label_consistency_ok"]
        and hard["positive_support_ok"]
    )

    report = {
        "ok": hard_ok,
        "archives": {
            "tar_path": tar_path,
            "tar_md5": tar_md5,
            "tar_md5_expected": EXPECTED_TAR_MD5,
            "tar_md5_ok": tar_md5_ok,
            "zip_path": zip_path,
            "zip_md5": zip_md5,
            "zip_md5_expected": EXPECTED_ZIP_MD5,
            "zip_md5_ok": zip_md5_ok,
        },
        "hard_invariants": hard,
        "positive_support": positive_support,
        "alignment": {
            "n_positive_pts": n_positive_pts,
            "proximity_fraction": proximity_fraction,
            "both_low_or_stopped_fraction": both_low_or_stopped_fraction,
            "n_gold_intervals": len(interval_records),
            "worst_intervals_by_proximity": _worst(interval_records, "proximity_fraction", N_WORST_INTERVALS),
            "worst_intervals_by_both_low_or_stopped": _worst(
                interval_records, "both_low_or_stopped_fraction", N_WORST_INTERVALS
            ),
        },
        "alignment_warning": alignment_warning,
        "totals": {
            "n_pt": n_pt,
            "n_pairs": len(pairs),
            "n_positive_pairs": n_positive_pairs,
            "n_negative_pairs": counts["n_negative_pairs"],
            "n_segments": len(segments),
            "relation_cover_sizes": {name: len(idxs) for name, idxs in relations.items()},
            "converter_counts": counts,
            "convert_wall_time_seconds": convert_wall_time_s,
            "verify_wall_time_seconds": verify_wall_time_s,
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify maritime_convert.convert against the real Brest AIS archives."
    )
    parser.add_argument("--tar", required=True, help="path to the HLE composite-events .tar.gz (archive A)")
    parser.add_argument("--zip", required=True, help="path to the LLE critical-events .zip (archive B)")
    parser.add_argument("--out", default="MARITIME_VERIFY.json", help="path to write the JSON report")
    args = parser.parse_args()

    report = run(args.tar, args.zip)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    a = report["alignment"]
    print(f"wrote {out_path}")
    print(f"HARD invariants ok: {report['ok']}")
    print(
        f"alignment: proximity_fraction={a['proximity_fraction']:.4f} "
        f"both_low_or_stopped_fraction={a['both_low_or_stopped_fraction']:.4f} "
        f"(n_positive_pts={a['n_positive_pts']})"
    )
    if report["alignment_warning"]:
        print(f"ALIGNMENT WARNING: {report['alignment_warning']}")
    if not report["positive_support"]["ok"]:
        print(f"POSITIVE SUPPORT FAILED: {report['positive_support']['reason']}")
    print(
        f"totals: n_pt={report['totals']['n_pt']} n_pairs={report['totals']['n_pairs']} "
        f"n_segments={report['totals']['n_segments']} "
        f"convert_wall_time_s={report['totals']['convert_wall_time_seconds']:.2f}"
    )

    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
