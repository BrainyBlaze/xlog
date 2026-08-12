"""Committed derivation of the pre-registered vocabulary ceiling for the
maritime rendezVous corpus, plus the archive-A interval-shape census.

WHY THIS EXISTS. The pre-registration in
`docs/experiments/maritime/README.md` declares a ~0.66 pointwise-F1
ceiling from "the exact definitional body achieves recall 1.0 but
precision ~0.49 (TP 3,579 / FP 3,689)" — numbers that, until this tool,
were computed by nothing committed (deep-review finding #4): the cited
source was a non-repo review document. This module makes the headline
claim auditable: it evaluates the exact definitional body on the
converted corpus and emits every number the ceiling rests on, plus the
FP decomposition finding #5 needs and the nested-interval census
finding #8 needs. It is run on the real archives as a separate job; the
unit tests exercise it end to end on the same synthetic mini-archives
the converter tests use.

WHAT IT COMPUTES, given archives A (HLE tar) and B (LLE zip):

(a) `body_pointwise_stats` — pointwise TP/FP/FN and P/R/F1 of the
    definitional body ``proximity AND both_low_or_stopped AND
    both_open_sea`` against ``is_positive``, over ALL pt rows of the
    converted corpus (no folds: the ceiling is a property of the corpus,
    not of a split).

(b) `fp_decomposition` — the same body's false positives split into the
    three causes the pre-registration names:
      - ``fp_negative_pairs``: FP rows belonging to negative pairs (no
        gold rendezVous interval anywhere in the pair);
      - ``fp_positive_pairs_short_run``: FP rows in positive pairs that
        lie inside a predicted run shorter than the gold rule's 240 s
        minimum-duration threshold. A PREDICTED RUN is a maximal run of
        consecutive body-predicted pt rows within one converter segment
        (runs never bridge segments); its DURATION is
        ``pt_time[last] - pt_time[first]`` over the run's own rows —
        i.e. the time span between the first and last predicted
        timepoint of the run, 0 for a single-row run. ``< 240`` on that
        span is the pre-registered reading of "sub-threshold episode".
      - ``fp_positive_pairs_other``: every remaining FP row.
    Categories are assigned in that order, so a short run inside a
    negative pair counts as ``fp_negative_pairs``, never twice.

(c) `nested_interval_census` — for every (fluent key, entity) interval
    list parsed from archive A (the real corpus carries exactly 8 HLE
    fluent keys: rendezVous / tugging / proximity / pilotBoarding /
    lowSpeed / stopped=farFromPorts / stopped=nearPorts / withinArea),
    how many intervals are STRICTLY NESTED inside another interval of
    the same key and entity, and how many OVERLAP another one. An
    interval i is strictly nested in j iff ``st_j <= st_i and
    et_i <= et_j`` and the two are not byte-identical (identical
    duplicates are counted separately, not as nesting); i and j overlap
    iff ``st_i < et_j and st_j < et_i`` (half-open). Strict nesting is
    exactly the shape `maritime_convert._covers` mishandles on unmerged
    lists (finding #8): a zero census on the real archives bounds that
    finding's real-corpus impact to zero; a non-zero census makes the
    impact a documented deviation to be measured, never a silent one.

USAGE (real archives — a 25-30 minute job, conversion dominates):

    py -3.13 examples/maritime_woled/ceiling_probe.py \\
        --tar <MaritimeCompositeEvents.tar.gz> \\
        --zip <brest_critical.zip> \\
        --out ceiling_probe.json

The unit tests call the same `main` on synthetic tar/zip mini-archives
(the `test_maritime_convert.py` fixture format) — that is the synthetic
mode; there is no separate flag, the tool is a pure function of its two
input archives. Torch-free by construction: the only project import is
`maritime_convert`.
"""

from __future__ import annotations

import argparse
import heapq
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from maritime_convert import convert, parse_hle_archive  # noqa: E402

DEFINITIONAL_BODY = ("proximity", "both_low_or_stopped", "both_open_sea")
DURATION_THRESHOLD_S = 240
MAX_NESTED_EXAMPLES = 10


def _predicted_rows(converted: dict) -> set[int]:
    """Global pt indices covered by the definitional body: the
    intersection of its three relations' memberships."""
    relations = converted["relations"]
    covered = set(relations[DEFINITIONAL_BODY[0]])
    for name in DEFINITIONAL_BODY[1:]:
        covered &= set(relations[name])
    return covered


def body_pointwise_stats(converted: dict) -> dict:
    """(a) Pointwise TP/FP/FN + P/R/F1 of the definitional body against
    ``is_positive`` over the whole corpus. Degenerate ratios are 0.0,
    never NaN (the `scorer.prf1` policy)."""
    predicted = _predicted_rows(converted)
    is_positive = converted["is_positive"]
    tp = sum(1 for i in predicted if is_positive[i])
    fp = len(predicted) - tp
    fn = sum(1 for i, pos in enumerate(is_positive) if pos and i not in predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "body": list(DEFINITIONAL_BODY),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def fp_decomposition(converted: dict, duration_threshold_s: int = DURATION_THRESHOLD_S) -> dict:
    """(b) Split the definitional body's FP rows by cause; see the module
    docstring for the run/duration definitions and the assignment order
    (negative pair first, then sub-threshold run, then other)."""
    predicted = _predicted_rows(converted)
    is_positive = converted["is_positive"]
    pt_time = converted["pt_time"]
    pt_pair_index = converted["pt_pair_index"]
    n_positive_pairs = converted["counts"]["n_positive_pairs"]

    # Duration of the maximal predicted run each predicted row sits in,
    # computed segment by segment (runs never bridge segments).
    run_duration_of: dict[int, int] = {}
    n_predicted_runs = 0
    n_short_runs = 0
    for lo, hi in converted["segments"]:
        start = None
        for i in range(lo, hi + 1):
            in_run = i < hi and i in predicted
            if in_run and start is None:
                start = i
            elif not in_run and start is not None:
                duration = pt_time[i - 1] - pt_time[start]
                n_predicted_runs += 1
                if duration < duration_threshold_s:
                    n_short_runs += 1
                for j in range(start, i):
                    run_duration_of[j] = duration
                start = None

    fp_negative_pairs = 0
    fp_positive_pairs_short_run = 0
    fp_positive_pairs_other = 0
    for i in predicted:
        if is_positive[i]:
            continue
        if pt_pair_index[i] >= n_positive_pairs:
            fp_negative_pairs += 1
        elif run_duration_of[i] < duration_threshold_s:
            fp_positive_pairs_short_run += 1
        else:
            fp_positive_pairs_other += 1

    return {
        "duration_threshold_s": duration_threshold_s,
        "fp_negative_pairs": fp_negative_pairs,
        "fp_positive_pairs_short_run": fp_positive_pairs_short_run,
        "fp_positive_pairs_other": fp_positive_pairs_other,
        "fp_total": fp_negative_pairs + fp_positive_pairs_short_run + fp_positive_pairs_other,
        "n_predicted_runs": n_predicted_runs,
        "n_short_runs": n_short_runs,
    }


def _census_one_list(ivs: list[tuple[int, int]]) -> tuple[int, int, int, list[tuple[int, int]]]:
    """(n_strictly_nested, n_overlapping, n_duplicates, nested_examples)
    for one same-key same-entity interval list. Definitions per the
    module docstring. One sweep in (st asc, et desc) order: strict
    nesting of i needs an earlier j with st_j <= st_i and et_j >= et_i
    that is not identical to i, so it suffices to track the max et among
    intervals with st strictly below st_i (>= et_i nests) and among
    already-seen intervals sharing st_i (> et_i nests); overlap flags
    come from a min-heap of still-active earlier ets (et > st_i means
    overlap, since every earlier st_j <= st_i < et_i). The active set
    stays empty on non-overlapping data, so the sweep is O(n log n) plus
    O(actual overlap pairs)."""
    n = len(ivs)
    if n < 2:
        return 0, 0, 0, []
    order = sorted(range(n), key=lambda k: (ivs[k][0], -ivs[k][1]))

    nested = [False] * n
    overlapping = [False] * n
    n_duplicates = 0
    nested_examples: list[tuple[int, int]] = []

    max_et_before = None          # max et over st < current st
    group_st = None               # st of the current equal-st group
    group_max_et = None           # max et over earlier members of that group
    active: list[tuple[int, int]] = []  # (et, idx) heap of not-yet-expired intervals
    prev_iv = None
    for k in order:
        st, et = ivs[k]
        if group_st != st:
            if group_max_et is not None:
                max_et_before = max(max_et_before, group_max_et) if max_et_before is not None else group_max_et
            group_st, group_max_et = st, None

        if (max_et_before is not None and max_et_before >= et) or (
            group_max_et is not None and group_max_et > et
        ):
            nested[k] = True
            if len(nested_examples) < MAX_NESTED_EXAMPLES:
                nested_examples.append((st, et))
        if prev_iv == (st, et):
            n_duplicates += 1

        while active and active[0][0] <= st:
            heapq.heappop(active)
        for other_et, j in active:
            overlapping[j] = True
            overlapping[k] = True
        heapq.heappush(active, (et, k))

        group_max_et = et if group_max_et is None else max(group_max_et, et)
        prev_iv = (st, et)

    return sum(nested), sum(overlapping), n_duplicates, nested_examples


def nested_interval_census(hle: dict) -> dict:
    """(c) Per-fluent-key census of strictly-nested / overlapping /
    duplicate intervals across every entity's list in archive A, plus
    grand totals. ``per_key[key]`` reports the entity count, interval
    count, the three shape counters, how many entities have >= 1 nested
    interval, and up to ``MAX_NESTED_EXAMPLES`` example (st, et) pairs."""
    per_key: dict[str, dict] = {}
    totals = {"n_intervals": 0, "n_strictly_nested": 0, "n_overlapping": 0, "n_duplicates": 0}

    for fluent_key, by_entity in sorted(hle["intervals"].items()):
        key_stats = {
            "n_entities": len(by_entity),
            "n_intervals": 0,
            "n_strictly_nested": 0,
            "n_overlapping": 0,
            "n_duplicates": 0,
            "n_entities_with_nested": 0,
            "nested_examples": [],
        }
        for entity, ivs in sorted(by_entity.items()):
            n_nested, n_overlap, n_dup, examples = _census_one_list(ivs)
            key_stats["n_intervals"] += len(ivs)
            key_stats["n_strictly_nested"] += n_nested
            key_stats["n_overlapping"] += n_overlap
            key_stats["n_duplicates"] += n_dup
            if n_nested:
                key_stats["n_entities_with_nested"] += 1
                for st, et in examples:
                    if len(key_stats["nested_examples"]) < MAX_NESTED_EXAMPLES:
                        key_stats["nested_examples"].append(
                            {"entity": list(entity), "st": st, "et": et}
                        )
        per_key[fluent_key] = key_stats
        totals["n_intervals"] += key_stats["n_intervals"]
        totals["n_strictly_nested"] += key_stats["n_strictly_nested"]
        totals["n_overlapping"] += key_stats["n_overlapping"]
        totals["n_duplicates"] += key_stats["n_duplicates"]

    return {"per_key": per_key, "totals": totals}


def probe(tar_path: str, zip_path: str) -> dict:
    """Full probe: one conversion, one archive-A reparse for the census,
    all three result blocks plus the converter's own counters (so the
    report is self-describing about the corpus it measured)."""
    converted = convert(tar_path, zip_path)
    hle = parse_hle_archive(tar_path)
    return {
        "archives": {"tar": tar_path, "zip": zip_path},
        "pointwise": body_pointwise_stats(converted),
        "fp_decomposition": fp_decomposition(converted),
        "nested_interval_census": nested_interval_census(hle),
        "converter_counts": converted["counts"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tar", required=True, help="MaritimeCompositeEvents.tar.gz (HLE, archive A)")
    parser.add_argument("--zip", required=True, help="brest_critical.zip (LLE, archive B)")
    parser.add_argument("--out", required=True, help="result JSON path")
    args = parser.parse_args(argv)

    report = probe(args.tar, args.zip)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    p = report["pointwise"]
    d = report["fp_decomposition"]
    c = report["nested_interval_census"]["totals"]
    print(f"wrote {out}")
    print(f"definitional body: tp={p['tp']} fp={p['fp']} fn={p['fn']} "
          f"P={p['precision']:.4f} R={p['recall']:.4f} F1={p['f1']:.4f}")
    print(f"fp decomposition: negative_pairs={d['fp_negative_pairs']} "
          f"short_run={d['fp_positive_pairs_short_run']} other={d['fp_positive_pairs_other']}")
    print(f"census: intervals={c['n_intervals']} nested={c['n_strictly_nested']} "
          f"overlapping={c['n_overlapping']} duplicates={c['n_duplicates']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
