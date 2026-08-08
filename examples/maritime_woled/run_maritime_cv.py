"""Pre-registered cross-validation over the converted Brest AIS rendezVous
corpus (`maritime_convert.convert`) -- the maritime counterpart of
`examples/caviar_woled/run_caviar_cv.py`, under the protocol fixed IN
ADVANCE by `docs/experiments/maritime/README.md`'s Pre-registration
section. This module implements EXACTLY that protocol, nothing else; every
constant below is named there.

FOLD ASSIGNMENT (`stratified_pair_folds`). Fold atoms are VESSEL PAIRS --
a pair's pt rows are never split across folds (the top-1 pair carries
33.4% of all positive pt, top-4 carry 60.2%; any finer split leaks the
dominant scene across the train/test boundary). Deterministic greedy LPT,
no RNG anywhere: positive pairs sorted by (positive-pt count DESC, pair
index ASC) are each dealt to the fold with the smallest running
positive-pt sum (ties: lowest fold index); negative pairs then sorted by
(pt-row count DESC, pair index ASC) are dealt the same way against the
folds' running NEGATIVE pt-row sums. Guaranteed property (tested): for
either load, ``max fold sum - min fold sum <= largest single pair's own
count`` -- when the eventual max fold received its LAST pair, it was the
least-loaded fold, so its pre-deal load was <= every other fold's FINAL
load (loads only grow); adding that one pair's count therefore lifts it
above the eventual minimum by at most that count.

ONE CONVERSION PER RUN. Unlike `run_caviar_cv.py` (which re-converts per
fold -- CAVIAR conversion is cheap), `maritime_convert.convert` costs ~31
CPU-minutes, so it runs ONCE and every fold is a PAIR-LEVEL SLICE of that
single corpus. Fold isolation is structural, not conventional: every pt
row belongs to exactly one pair (the verifier's pair-contiguity
invariant), every relation membership is derived by the converter from
that pair's own vessels/episodes alone (proven pair-local by the task-2
adversarial review's mini-archive equivalence, V3-A), and segments never
span pairs -- so restricting rows and relation memberships to a fold's
own pairs cannot move information across the train/test boundary.

SEARCH PER FOLD. The pre-registered direct-target protocol
(`docs/experiments/maritime/README.md`, point f): facts are the training
side's global pt indices, labels its `is_positive` rows; `min_fit` is
`relational_search.permutation_null_threshold`'s 0.95 quantile of 1000
label-permutation pool-max per-fold-F1 samples (`perm_seed = seed`),
derived fresh per fold from that fold's own training facts/labels; the
search is `relational_search.induce_relational_theory` (sequential
covering via `theory_loop`) with `holdout_score="f1"`, inner
`folds=INNER_FOLDS`, `min_new_covered` and `tie_tolerance` from the CLI
(defaults 2 and the derived per-iteration tolerance).

SCORING PER FOLD. Pointwise P/R/F1 on the positive class over the
held-out fold's rows (`scorer.prf1`), PLUS the pre-registered
interval-level aggregate (`interval_prf1`): maximal runs of consecutive
rows within a converter segment, a predicted interval matched iff it
shares >= 1 row with some gold interval and vice versa. Aggregation:
micro (tp/fp/fn summed across folds, then one P/R/F1 -- and the interval
counterpart over summed matched/total interval counts) AND the median +
min/max of the per-fold F1s, per the pre-registration's requirement that
the micro number never stands alone (fold positive masses are necessarily
unequal: the top pair alone carries ~33% of positives).

VERIFIER SMOKE (`verify_smoke`). Before any CV work on real data, this
runner re-checks -- on the corpus it JUST converted, without a second
31-minute conversion -- everything the committed verifier can check
without re-parsing: both archives' md5 against
`verify_maritime_conversion`'s own pinned constants, that module's own
hard structural invariants (pair contiguity, segment/pair boundaries, EC
label-count consistency -- the same committed functions, imported, not
reimplemented), and exact equality of the converted counts against the
pre-registered `EXPECTED_COUNTS`. Any failure ABORTS the run.
`--skip-verify` exists for unit tests on synthetic archives ONLY; it
stamps `{"skipped": true, "warning": ...}` into the result JSON so a
skipped check can never masquerade as a passed one.

CPU-ONLY, TORCH-FREE IMPORT. Module top level never imports torch /
relational_search / scorer / maritime_convert -- every such import is
deferred into the function that needs it (`run_caviar_cv.py`'s own
discipline), so `--help` and the fold-assignment helpers stay
dependency-free.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
CAVIAR_DIR = EXAMPLE_DIR.parent / "caviar_woled"
for _dir in (str(EXAMPLE_DIR), str(CAVIAR_DIR)):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

# Pre-registered search constants -- docs/experiments/maritime/README.md,
# point (f); identical to run_caviar_cv.py's conventions.
INNER_FOLDS = 4
MAX_LITERALS = 3
MAX_CLAUSES = 4
NULL_PERMUTATIONS = 1000
NULL_QUANTILE = 0.95

# Pre-registered converted-corpus constants (README "Corpus and
# provenance"; adversarially verified in the task-2 review). verify_smoke
# refuses to let a run proceed when the fresh conversion disagrees.
EXPECTED_COUNTS = {
    "n_pt": 454858,
    "n_pairs": 806,
    "n_positive_pairs": 302,
    "n_negative_pairs": 504,
    "n_negative_pool": 2014,
    "n_positive_pts": 3579,
}

# --smoke: the deterministic tiny subset (first N positive + first N
# negative pairs in the converter's own sorted pair order) used to exercise
# the full fold->gate->search->score->JSON cycle quickly in tests.
SMOKE_N_POS = 6
SMOKE_N_NEG = 6


def stratified_pair_folds(
    positive_pt_counts: list[int],
    pt_counts: list[int],
    n_positive_pairs: int,
    n_folds: int,
) -> list[int]:
    """One fold index (0..n_folds-1) per pair, pairs ordered exactly as
    `maritime_convert.convert`'s own ``"pairs"`` list (positives first,
    then negatives -- ``n_positive_pairs`` marks the boundary).

    ``positive_pt_counts`` has one entry per POSITIVE pair (its own
    positive-pt row count); ``pt_counts`` has one entry per pair, positives
    AND negatives (its own total pt row count). See the module docstring
    for the LPT mechanism and its balance guarantee.

    Raises ``ValueError`` if ``n_folds < 2``, if either side has fewer
    pairs than folds (every fold must hold >= 1 positive and >= 1 negative
    pair -- a fold without positives cannot be scored on the positive
    class, a fold without negatives cannot expose false positives), if
    ``len(positive_pt_counts) != n_positive_pairs``, or if any count is
    negative."""
    n_pairs = len(pt_counts)
    n_negative_pairs = n_pairs - n_positive_pairs
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2 (got {n_folds!r}).")
    if len(positive_pt_counts) != n_positive_pairs:
        raise ValueError(
            f"positive_pt_counts has {len(positive_pt_counts)} entries, "
            f"expected n_positive_pairs={n_positive_pairs}."
        )
    if n_positive_pairs < n_folds:
        raise ValueError(
            f"{n_positive_pairs} positive pairs is fewer than "
            f"n_folds={n_folds}: every fold needs >= 1 positive pair to "
            "score the positive class on."
        )
    if n_negative_pairs < n_folds:
        raise ValueError(
            f"{n_negative_pairs} negative pairs is fewer than "
            f"n_folds={n_folds}: every fold needs >= 1 negative pair to "
            "expose false positives."
        )
    if any(c < 0 for c in positive_pt_counts) or any(c < 0 for c in pt_counts):
        raise ValueError("pair counts must all be non-negative.")

    fold_of_pair = [0] * n_pairs

    def _deal(indices: list[int], counts_of: dict[int, int]) -> None:
        load = [0] * n_folds
        order = sorted(indices, key=lambda i: (-counts_of[i], i))
        for i in order:
            fold = min(range(n_folds), key=lambda f: (load[f], f))
            fold_of_pair[i] = fold
            load[fold] += counts_of[i]

    _deal(
        list(range(n_positive_pairs)),
        {i: positive_pt_counts[i] for i in range(n_positive_pairs)},
    )
    _deal(
        list(range(n_positive_pairs, n_pairs)),
        {i: pt_counts[i] for i in range(n_positive_pairs, n_pairs)},
    )
    return fold_of_pair


def pair_counts(converted: dict, n_pairs: int) -> tuple[list[int], list[int]]:
    """Per-pair stratification masses read off one converted corpus:
    ``(positive_pt_counts, pt_counts)``, each one entry per pair index
    0..n_pairs-1 -- ``positive_pt_counts[p]`` is pair ``p``'s own
    positive-pt row count (0 for negative pairs, by construction),
    ``pt_counts[p]`` its total pt row count. Reads only
    ``converted["pt_pair_index"]`` and ``converted["is_positive"]``."""
    positive_pt_counts = [0] * n_pairs
    pt_counts = [0] * n_pairs
    for pair_idx, pos in zip(converted["pt_pair_index"], converted["is_positive"]):
        pt_counts[pair_idx] += 1
        if pos:
            positive_pt_counts[pair_idx] += 1
    return positive_pt_counts, pt_counts


def fold_pt_indices(converted: dict, fold_of_pair: list[int], fold: int) -> tuple[list[int], list[int]]:
    """``(train_pts, test_pts)``: the pt-level partition induced by the
    pair-level assignment -- a pair's rows land entirely on one side, so
    the split cannot leak a vessel pair across the boundary. Both lists
    ascend (pt order = the converter's own row order)."""
    train_pts: list[int] = []
    test_pts: list[int] = []
    for pt, pair_idx in enumerate(converted["pt_pair_index"]):
        (test_pts if fold_of_pair[pair_idx] == fold else train_pts).append(pt)
    return train_pts, test_pts


def restrict_relations(relations: dict[str, list], pt_set: set) -> dict[str, list]:
    """Every relation NAME survives (the candidate vocabulary must not
    shrink with the fold -- an empty membership is information, a missing
    name is a silent cap); memberships are filtered to ``pt_set``."""
    return {name: [pt for pt in members if pt in pt_set] for name, members in relations.items()}


def _runs(flags: list[bool], segments: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Maximal [start, end) runs of True WITHIN each segment -- a run never
    bridges a segment boundary (segments are episode/pair-local by the
    converter's contract, and an interval spanning them would be
    fictional)."""
    runs: list[tuple[int, int]] = []
    for lo, hi in segments:
        start = None
        for i in range(lo, hi):
            if flags[i] and start is None:
                start = i
            elif not flags[i] and start is not None:
                runs.append((start, i))
                start = None
        if start is not None:
            runs.append((start, hi))
    return runs


def interval_prf1(pred: list[bool], gold: list[bool], segments: list[tuple[int, int]]) -> dict:
    """The pre-registered interval-level aggregate: maximal True-runs
    within a segment; a gold interval is matched iff it shares >= 1 row
    with some predicted interval, and vice versa. Degenerate ratios are
    0.0, never NaN (same policy as `scorer.prf1`)."""
    gold_runs = _runs(gold, segments)
    pred_runs = _runs(pred, segments)

    def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
        return a[0] < b[1] and b[0] < a[1]

    n_matched_gold = sum(1 for g in gold_runs if any(_overlaps(g, p) for p in pred_runs))
    n_matched_pred = sum(1 for p in pred_runs if any(_overlaps(p, g) for g in gold_runs))
    precision = n_matched_pred / len(pred_runs) if pred_runs else 0.0
    recall = n_matched_gold / len(gold_runs) if gold_runs else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "n_gold_intervals": len(gold_runs),
        "n_pred_intervals": len(pred_runs),
        "n_matched_gold": n_matched_gold,
        "n_matched_pred": n_matched_pred,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _test_segment_bounds(converted: dict, test_set: set) -> list[tuple[int, int]]:
    """Global converter segments that lie in the test fold, re-expressed in
    the coordinates of the SORTED test-row array. Segments never span
    pairs (verifier invariant), so membership of a segment's first row
    decides the whole segment."""
    bounds: list[tuple[int, int]] = []
    offset = 0
    for lo, hi in converted["segments"]:
        if lo in test_set:
            bounds.append((offset, offset + (hi - lo)))
            offset += hi - lo
    return bounds


def run_fold(
    fold: int,
    converted: dict,
    fold_of_pair: list[int],
    *,
    seed: int,
    min_new_covered: int,
    tie_tolerance: float | None,
) -> dict:
    """One held-out fold, end to end: derive the permutation-null gate from
    THIS fold's training labels, run the sequential-covering relational
    search under it (holdout_score=f1 -- pre-registration point f), then
    score the found theory on the held-out rows point-wise AND
    interval-wise. Everything the record reports is recomputable from the
    record itself."""
    from relational_search import (
        body_cover,
        enumerate_bodies,
        induce_relational_theory,
        permutation_null_threshold,
    )
    from scorer import prf1

    train_pts, test_pts = fold_pt_indices(converted, fold_of_pair, fold)
    train_set, test_set = set(train_pts), set(test_pts)
    train_relations = restrict_relations(converted["relations"], train_set)
    train_labels = [converted["is_positive"][pt] for pt in train_pts]

    bodies, skipped = enumerate_bodies(train_relations, max_literals=MAX_LITERALS)
    covers = {body: body_cover(body, train_relations) for body in bodies}
    null = permutation_null_threshold(
        bodies, train_relations, train_pts, train_labels,
        folds=INNER_FOLDS, seed=seed,
        n_permutations=NULL_PERMUTATIONS, quantile=NULL_QUANTILE,
        perm_seed=seed, covers=covers,
    )
    theory = induce_relational_theory(
        train_relations, train_pts, train_labels,
        max_literals=MAX_LITERALS, folds=INNER_FOLDS, seed=seed,
        min_fit=null["threshold"], tie_tolerance=tie_tolerance,
        max_clauses=MAX_CLAUSES, min_new_covered=min_new_covered,
        holdout_score="f1",
    )

    test_relations = restrict_relations(converted["relations"], test_set)
    predicted: set = set()
    for clause in theory["clauses"]:
        predicted |= body_cover(tuple(clause), test_relations)
    pred = [pt in predicted for pt in test_pts]
    gold = [converted["is_positive"][pt] for pt in test_pts]
    point = prf1(pred, gold)
    interval = interval_prf1(pred, gold, _test_segment_bounds(converted, test_set))

    return {
        "fold": fold,
        "clauses": [list(c) for c in theory["clauses"]],
        "stop_reason": theory["stop_reason"],
        "iterations": [
            {**it, "rule": (list(it["rule"]) if it.get("rule") is not None else None)}
            for it in theory["iterations"]
        ],
        "min_fit": null["threshold"],
        "null_summary": null,
        "n_bodies": len(bodies),
        "n_bodies_skipped_empty": skipped,
        "n_train_pt": len(train_pts),
        "n_test_pt": len(test_pts),
        "test_pairs": [list(converted["pairs"][p]) for p, f in enumerate(fold_of_pair) if f == fold],
        "scoring": {"point": point, "interval": interval},
    }


def verify_smoke(converted: dict, tar_path: str, zip_path: str) -> dict:
    """The pre-registered pre-run gate, WITHOUT a second 31-minute
    conversion: both archives' md5 against the committed verifier's own
    pinned constants, that verifier's structural invariants (imported, not
    reimplemented) on the corpus just converted, and exact equality of the
    converted counts against the pre-registered `EXPECTED_COUNTS`."""
    import verify_maritime_conversion as vm

    tar_md5 = vm._md5(tar_path)
    zip_md5 = vm._md5(zip_path)
    md5_report = {
        "tar_md5": tar_md5,
        "zip_md5": zip_md5,
        "ok": tar_md5 == vm.EXPECTED_TAR_MD5 and zip_md5 == vm.EXPECTED_ZIP_MD5,
    }

    contiguity = vm._check_pair_contiguity(converted["pt_pair_index"])
    seg_violations = vm._check_segment_pair_boundaries(
        converted["segments"], converted["pt_pair_index"],
    )
    hard = {
        "pair_contiguity_ok": bool(contiguity.get("ok", False)),
        "segment_pair_boundary_ok": not seg_violations,
    }
    if "ec" in converted:
        ec_violations = vm._check_ec_label_consistency(converted["segments"], converted["ec"])
        hard["ec_label_consistency_ok"] = not ec_violations

    actual_counts = dict(converted.get("counts", {}))
    actual_counts.setdefault("n_positive_pts", sum(bool(x) for x in converted["is_positive"]))
    mismatches = {
        key: {"expected": expected, "actual": actual_counts.get(key)}
        for key, expected in EXPECTED_COUNTS.items()
        if actual_counts.get(key) != expected
    }
    counts_report = {"ok": not mismatches, "mismatches": mismatches}

    return {
        "ok": md5_report["ok"] and counts_report["ok"] and all(hard.values()),
        "md5": md5_report,
        "counts": counts_report,
        "hard_invariants": hard,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tar", required=True, help="MaritimeCompositeEvents.tar.gz (HLE)")
    parser.add_argument("--zip", required=True, help="brest_critical.zip (LLE)")
    parser.add_argument("--out", required=True, help="result JSON path")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-new-covered", type=int, default=2, dest="min_new_covered")
    parser.add_argument("--tie-tolerance", type=float, default=None, dest="tie_tolerance")
    parser.add_argument("--smoke", action="store_true",
                        help=f"first {SMOKE_N_POS}+{SMOKE_N_NEG} pairs only (tests)")
    parser.add_argument("--skip-verify", action="store_true", dest="skip_verify",
                        help="unit tests on synthetic archives ONLY; stamped into the JSON")
    return parser.parse_args(argv)


def _smoke_subset(converted: dict) -> dict:
    """The deterministic tiny subset: first SMOKE_N_POS positive + first
    SMOKE_N_NEG negative pairs in the converter's own pair order, with pt
    indices REMAPPED to a dense 0..n range (relations/segments/ec sliced
    accordingly is out of scope for smoke: relations are filtered, ec
    dropped)."""
    n_pos_keep = min(SMOKE_N_POS, converted["counts"]["n_positive_pairs"])
    keep_pairs = set(range(n_pos_keep)) | {
        converted["counts"]["n_positive_pairs"] + j
        for j in range(min(SMOKE_N_NEG, converted["counts"]["n_negative_pairs"]))
    }
    old_to_new_pair = {old: new for new, old in enumerate(sorted(keep_pairs))}
    keep_pts = [pt for pt, pair in enumerate(converted["pt_pair_index"]) if pair in keep_pairs]
    old_to_new_pt = {old: new for new, old in enumerate(keep_pts)}
    keep_set = set(keep_pts)
    segments = [
        (old_to_new_pt[lo], old_to_new_pt[hi - 1] + 1)
        for lo, hi in converted["segments"] if lo in keep_set
    ]
    return {
        "pairs": [converted["pairs"][old] for old in sorted(keep_pairs)],
        "pt_pair_index": [old_to_new_pair[converted["pt_pair_index"][pt]] for pt in keep_pts],
        "pt_time": [converted["pt_time"][pt] for pt in keep_pts],
        "segments": segments,
        "relations": {
            name: [old_to_new_pt[pt] for pt in members if pt in keep_set]
            for name, members in converted["relations"].items()
        },
        "is_positive": [converted["is_positive"][pt] for pt in keep_pts],
        "counts": {
            "n_positive_pairs": n_pos_keep,
            "n_negative_pairs": len(keep_pairs) - n_pos_keep,
            "n_pairs": len(keep_pairs),
            "n_pt": len(keep_pts),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from maritime_convert import convert

    t0 = time.monotonic()
    converted = convert(args.tar, args.zip)
    convert_s = time.monotonic() - t0

    if args.skip_verify:
        verify = {"skipped": True, "warning": "verify_smoke SKIPPED (--skip-verify): "
                  "synthetic-archive unit runs only; a real run must not carry this stamp."}
    else:
        verify = verify_smoke(converted, args.tar, args.zip)
        if not verify["ok"]:
            print(f"verify_smoke FAILED: {json.dumps(verify, indent=2)[:2000]}")
            return 2

    if args.smoke:
        converted = _smoke_subset(converted)

    n_pairs = converted["counts"]["n_pairs"]
    n_pos = converted["counts"]["n_positive_pairs"]
    pos_counts, pt_counts = pair_counts(converted, n_pairs)
    fold_of_pair = stratified_pair_folds(pos_counts[:n_pos], pt_counts, n_pos, args.folds)

    fold_records = []
    for fold in range(args.folds):
        t1 = time.monotonic()
        record = run_fold(
            fold, converted, fold_of_pair,
            seed=args.seed, min_new_covered=args.min_new_covered,
            tie_tolerance=args.tie_tolerance,
        )
        record["wall_s"] = time.monotonic() - t1
        fold_records.append(record)
        point = record["scoring"]["point"]
        print(f"fold {fold}: clauses={record['clauses']} point_f1={point['f1']:.4f}")

    tp = sum(r["scoring"]["point"]["tp"] for r in fold_records)
    fp = sum(r["scoring"]["point"]["fp"] for r in fold_records)
    fn = sum(r["scoring"]["point"]["fn"] for r in fold_records)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    micro_point = {
        "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
        "f1": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
    }
    mg = sum(r["scoring"]["interval"]["n_matched_gold"] for r in fold_records)
    ng = sum(r["scoring"]["interval"]["n_gold_intervals"] for r in fold_records)
    mp = sum(r["scoring"]["interval"]["n_matched_pred"] for r in fold_records)
    np_ = sum(r["scoring"]["interval"]["n_pred_intervals"] for r in fold_records)
    ip = mp / np_ if np_ else 0.0
    ir = mg / ng if ng else 0.0
    micro_interval = {
        "n_matched_gold": mg, "n_gold_intervals": ng,
        "n_matched_pred": mp, "n_pred_intervals": np_,
        "precision": ip, "recall": ir,
        "f1": (2 * ip * ir / (ip + ir)) if ip + ir else 0.0,
    }
    fold_f1s = [r["scoring"]["point"]["f1"] for r in fold_records]

    result = {
        "protocol": "maritime rendezVous direct-target CV "
                    "(docs/experiments/maritime/README.md pre-registration)",
        "vocabulary_ceiling_note": "point-F1 vocabulary ceiling ~0.66 (pre-registered); "
                                   "no comparison with published 0.98 (different grid)",
        "archives": {"tar": args.tar, "zip": args.zip},
        "verify_smoke": verify,
        "params": {
            "folds": args.folds, "seed": args.seed,
            "min_new_covered": args.min_new_covered,
            "tie_tolerance": args.tie_tolerance,
            "holdout_score": "f1", "inner_folds": INNER_FOLDS,
            "max_literals": MAX_LITERALS, "max_clauses": MAX_CLAUSES,
            "null_permutations": NULL_PERMUTATIONS, "null_quantile": NULL_QUANTILE,
            "smoke": args.smoke,
        },
        "candidate_vocabulary": sorted(converted["relations"]),
        "fold_of_pair": fold_of_pair,
        "convert_wall_s": convert_s,
        "folds": fold_records,
        "micro": {"point": micro_point, "interval": micro_interval},
        "per_fold_point_f1": {
            "values": fold_f1s,
            "median": statistics.median(fold_f1s),
            "min": min(fold_f1s),
            "max": max(fold_f1s),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out}  micro point F1={micro_point['f1']:.4f}  "
          f"median fold F1={result['per_fold_point_f1']['median']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
