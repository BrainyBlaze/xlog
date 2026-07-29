"""10-fold cross-validation over the CAVIAR continuous corpus, under the
OLED/WOLED papers' OWN evaluation protocol: stratified folds over VIDEO
SEGMENTS (never split mid-video), train-on-9/test-on-1, tp/fp/fn accumulated
ACROSS folds and MICRO-averaged into one P/R/F1 -- the aggregation those
papers themselves report (e.g. OLED Table 1(b): 0.678/0.953/0.792), so this
script's numbers sit in the SAME table under the SAME protocol, not merely a
comparable one.

PRE-REGISTERED PROTOCOL (fixed before any real-data run; this script
implements EXACTLY this, nothing else):

* Corpus: `caviar-train.json` + `caviar-test.json` COMBINED into one
  continuous corpus -- every video segment from both files
  (`caviar_continuous.load_continuous` on each, concatenated). The papers'
  own CV protocol cross-validates over the whole dataset, not a fixed
  train/test split, so the two files' segments are pooled before folding.
* Folds: `--folds` (default 10), assigned over VIDEO SEGMENTS by
  `stratified_segment_folds` (below) -- deterministic, seeded by `--seed`
  (default 7). A segment is NEVER split across folds.
* Per fold i: train on the other folds' segments, evaluate on fold i's
  segments. `caviar_continuous.convert_continuous`/`derive_ec_targets_
  continuous`/`derive_ec_masks_continuous` are called ONCE PER FOLD PER SIDE
  (the fold's own train segment list, the fold's own test segment list) --
  see FOLD ISOLATION below for why this rules out any cross-fold leakage
  structurally, not just by convention.
* Training per fold, EC protocol (relational, `relational_search.
  induce_relational_theory`, `max_literals=3`): each target's (init, term)
  `min_fit` is `relational_search.permutation_null_threshold`'s 95th
  percentile of 1000 label-permutation pool-max per-fold-F1 samples
  (`perm_seed=7`), derived FRESH per target PER FOLD from that fold's own
  (don't-care-excluded) training facts/labels -- never shared across
  targets or folds. Inner holdout `folds=4`, `seed=7` (== `--seed` here, the
  same value threads through fold assignment, inner holdout, and the
  permutation RNG, matching every seed the pre-registration names as "7").
  `min_new_covered=2`; `holdout_score="f1"` (recall-aware: see
  `relational_search.kfold_scores`'s own docstring for why plain accuracy
  cannot be trusted on a rare-positive-class holdout); tie tolerance at its
  own derived default (`induce_relational_theory`'s
  `max(0.01, 1 / n_residual_facts)`, never a caller-supplied constant).
* Direct-protocol reference theory, per fold (reported for context, not the
  paper-comparison number): the SAME relational search machinery over the
  SAME per-fold vocabulary, against the don't-care-free per-timestep
  `is_positive` target, `max_literals=3`, `folds=4`, `seed=7`,
  `min_new_covered=2` -- but `min_fit` at `induce_relational_theory`'s own
  fixed 0.75 default (no permutation-null derivation: the direct target has
  thousands of positives per fold, not ~1-2, so accuracy-era gating is not
  the failure mode the permutation-null gate exists to fix) and
  `holdout_score` likewise left at its own default (`"accuracy"`). Only
  `min_fit`/tie are explicitly re-pinned to their un-derived defaults by the
  pre-registration ("min_fit default 0.75, tie default"); every other inner
  setting (k, seed, min_new_covered, max_literals) is shared with the EC
  search.
* Scoring: per fold, `caviar_continuous.reconstruct_holds_continuous`'s
  inertia closure over the held-out segments' predicted init/term events
  (NOT `ec_scorer.reconstruct_holds`'s fixed-`num_windows*T` reading -- a
  fold's held-out segments have no single shared window length, the exact
  reason `run_caviar_theory.py --data continuous` itself switches
  reconstruction functions; see that script's own `reconstruct_fn`
  docstring), scored via `scorer.prf1` against the held-out frames'
  `is_positive` gold. Every fold's tp/fp/fn are accumulated and
  MICRO-averaged: summed counts, THEN one P/R/F1 computed from the totals
  -- the papers' own aggregation, not a mean of per-fold F1s (which would
  weight a 3-positive fold the same as a 300-positive one). See
  `_micro_prf1`.

FOLD ISOLATION. Each fold's train/test conversion is entirely independent:
`convert_continuous` is called once on THAT fold's own segment list, so its
`pt` numbering (and every relation/fact list derived from it) never
overlaps, numerically or semantically, with any other fold's. A held-out
segment's own rows are simply never present in the array `convert_
continuous` builds for the other 9 folds' training call -- there is no
filtering step capable of leaking one fold's row into another's, because
the row is never created for the wrong fold's conversion in the first
place.

CPU-ONLY, TORCH-FREE `--help`. `parse_args` and every name bound at this
module's top level never touch `torch`/`relational_search`/
`caviar_continuous` -- every such import is deferred into the function that
actually needs it (mirrors `run_caviar_theory.py`'s identical discipline),
even though nothing in this script ever needs CUDA (`relational_search.py`'s
whole reason to exist is that a relational-only search needs none -- see
its own module docstring).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

INNER_FOLDS = 4
MIN_NEW_COVERED = 2
MAX_LITERALS = 3
NULL_PERMUTATIONS = 1000
NULL_QUANTILE = 0.95


def stratified_segment_folds(
    segment_positive_counts: list[int], n_folds: int, seed: int,
) -> list[int]:
    """Deterministic, seeded fold assignment, one index (0..n_folds-1) per
    SEGMENT (never a pair-time row -- this operates entirely above
    `convert_continuous`, on one integer positive-frame count per video
    segment), by descending-stratified round-robin dealing:

    1. Sort segment INDICES by `segment_positive_counts` DESCENDING, ties
       broken by the segment's own original index ascending -- a stable
       total order: ``sorted(range(n), key=lambda i: (-counts[i], i))``.
    2. Draw ONE random permutation of ``range(n_folds)`` from a single
       ``torch.Generator().manual_seed(seed)``
       (``torch.randperm(n_folds, generator=rng)``) -- this is "fold order
       shuffled once": the sequence of ACTUAL fold ids a dealing pass
       visits, drawn once, never reshuffled again for a later round.
    3. Deal: the segment at rank ``r`` in the sorted order (step 1) is
       assigned to ``fold_sequence[r % n_folds]``.

    This is an ordinary "deal cards to n_folds players, largest card first"
    round-robin, with the PLAYER SEQUENCE (not each round independently)
    drawn once from the seed -- relabeling which physical fold id sits at
    each position in the cycle changes nothing about the BALANCE the
    dealing produces (every fold still receives one card per full round, in
    strictly the same relative rank order), only which fold id ends up
    holding which segments.

    PROPERTY (checked in the test suite): for any two folds' own summed
    positive counts, ``max_fold_sum - min_fold_sum <=
    max(segment_positive_counts)`` -- the largest single segment's own
    count. Sketch: for two dealing SLOTS (positions within a round)
    ``p < q``, round ``r``'s pair of items at global ranks
    ``r*n_folds + p`` and ``r*n_folds + q`` satisfies
    ``value(rank r*n_folds+p) >= value(rank r*n_folds+q)`` (the values are
    sorted descending), and that difference is itself the sum of the
    (non-negative, since sorted descending) consecutive-rank gaps strictly
    BETWEEN those two ranks. Summed over every round, these per-round
    partial sums cover DISJOINT, non-overlapping rank ranges, so their total
    can never exceed the sum of EVERY consecutive gap across the WHOLE
    sorted sequence -- which telescopes exactly to
    ``value(rank 0) - value(rank n-1) <= value(rank 0)`` (counts are
    non-negative). This bounds any two slots' summed difference, hence the
    max-minus-min over all ``n_folds`` slots, by the largest single count --
    independent of which physical fold id `fold_sequence` maps each slot to.

    Raises ``ValueError`` if ``n_folds < 2``, if there are fewer segments
    than folds (a fold with zero segments makes "train on the other folds,
    test on this one" meaningless), or if any count is negative.
    """
    import torch

    n = len(segment_positive_counts)
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2 (got {n_folds!r}).")
    if n < n_folds:
        raise ValueError(
            f"{n} segments is fewer than n_folds={n_folds}: every fold "
            "needs at least one segment to test on."
        )
    if any(c < 0 for c in segment_positive_counts):
        raise ValueError("segment_positive_counts must all be non-negative.")

    order = sorted(range(n), key=lambda i: (-segment_positive_counts[i], i))
    rng = torch.Generator().manual_seed(seed)
    fold_sequence = torch.randperm(n_folds, generator=rng).tolist()

    fold_of_segment = [0] * n
    for rank, seg_idx in enumerate(order):
        fold_of_segment[seg_idx] = fold_sequence[rank % n_folds]
    return fold_of_segment


def _segment_positive_counts(segments: list[dict]) -> list[int]:
    """One count per segment: its own annotated meeting-frame total
    (``len(segment["meeting"])``) -- the ground-truth positive mass
    stratification balances over, read directly off `load_continuous`'s own
    per-segment dict, before any pair-time conversion happens (so
    stratification never depends on which pairs end up co-visible, only on
    the raw annotation)."""
    return [len(seg["meeting"]) for seg in segments]


def _fold_segment_split(
    segments: list[dict], fold_of_segment: list[int], fold_index: int,
) -> tuple[list[dict], list[dict]]:
    """Partition ``segments`` into (train, test) for one fold: every segment
    NOT assigned ``fold_index`` trains, every segment assigned it tests --
    each list is a plain sub-list of the ORIGINAL segment dicts (never
    copied/mutated), so a caller's later `convert_continuous` call sees
    exactly, and only, the segments that belong to its own side."""
    train_segments = [s for s, f in zip(segments, fold_of_segment) if f != fold_index]
    test_segments = [s for s, f in zip(segments, fold_of_segment) if f == fold_index]
    return train_segments, test_segments


def _exclude_dontcare(facts, labels, dontcare):
    """Drop every ``(fact, label)`` pair flagged don't-care -- same
    semantics as `run_caviar_theory._exclude_dontcare` (reimplemented here,
    not imported: this script's only dependencies are the module-level
    pieces named in its own docstring, never `run_caviar_theory.py`'s own
    CLI/CUDA-branching orchestration, which this script does not need)."""
    kept_facts = []
    kept_labels = []
    for f, y, dc in zip(facts, labels, dontcare):
        if not dc:
            kept_facts.append(f)
            kept_labels.append(y)
    return kept_facts, kept_labels


def _ec_relations(converted: dict) -> dict:
    """EC-search candidate vocabulary for one fold's side (train or test):
    every base relation EXCEPT the data-quality flag ``coords_missing``,
    PLUS the frame-difference transition relations -- mirrors
    `run_caviar_theory._ec_relations_with_transitions`'s merge, then
    `_filtered_relation_names`'s exclusion, done in one step (this script
    never runs the direct protocol through this vocabulary, so there is no
    separate un-merged reading to keep byte-identical here, unlike that
    module)."""
    merged = {**converted["relations"], **converted["transition_relations"]}
    return {n: v for n, v in merged.items() if n != "coords_missing"}


def _direct_relations(converted: dict) -> dict:
    """Direct-protocol reference vocabulary for one fold's side: base
    relations only, EXCLUDING ``coords_missing`` -- mirrors
    `run_caviar_theory._filtered_relation_names`. The transition relations
    are scoped to the EC search only (see
    `caviar_continuous.convert_continuous`'s own docstring for why)."""
    return {n: v for n, v in converted["relations"].items() if n != "coords_missing"}


def _micro_prf1(counts: list[dict]) -> dict:
    """Micro-averaged precision/recall/F1: SUM tp/fp/fn across every fold's
    own count dict FIRST, then compute one P/R/F1 from the totals -- the
    papers' own aggregation (e.g. OLED Table 1(b)), NOT a mean of each
    fold's own F1 (which would weight a fold with 3 held-out positives the
    same as one with 300). Same zero-division-to-0.0 substitution
    `scorer.prf1` uses (precision/recall/F1 all report 0.0, never raise or
    NaN, when their own denominator is 0), applied to the SUMMED counts
    instead of a per-row comparison."""
    tp = sum(c["tp"] for c in counts)
    fp = sum(c["fp"] for c in counts)
    fn = sum(c["fn"] for c in counts)
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _induce_ec_target(train_relations: dict, facts, labels, seed: int):
    """One target's (init or term) EC theory for one fold: derive its own
    permutation-null `min_fit` from THIS fold's THIS target's own
    (don't-care-excluded) training facts/labels, then search under it --
    mirrors `run_caviar_theory.py`'s own `resolve_min_fit`/`induce_for`
    pairing for ``--ec-fit-mode permutation-null``, reimplemented directly
    against `relational_search`'s public functions (this script does not
    import `run_caviar_theory.py` at all -- see the module docstring)."""
    from relational_search import (
        body_cover,
        enumerate_bodies,
        induce_relational_theory,
        permutation_null_threshold,
    )

    bodies, _ = enumerate_bodies(train_relations, max_literals=MAX_LITERALS)
    covers = {b: body_cover(b, train_relations) for b in bodies}
    null = permutation_null_threshold(
        bodies, train_relations, facts, labels,
        folds=INNER_FOLDS, seed=seed, n_permutations=NULL_PERMUTATIONS,
        quantile=NULL_QUANTILE, perm_seed=seed, covers=covers,
    )
    result = induce_relational_theory(
        train_relations, facts, labels,
        max_literals=MAX_LITERALS, folds=INNER_FOLDS, seed=seed, tie_tolerance=None,
        max_clauses=4, min_new_covered=MIN_NEW_COVERED, holdout_score="f1",
        min_fit=null["threshold"],
    )
    result["min_fit"] = null["threshold"]
    result["null_summary"] = null
    return result


def _induce_direct_theory(train_relations: dict, facts, labels, seed: int):
    """The direct-protocol reference theory for one fold -- see the module
    docstring's "Direct-protocol reference theory" paragraph for exactly
    which settings are shared with the EC search and which are pinned to
    their own un-derived defaults."""
    from relational_search import induce_relational_theory

    return induce_relational_theory(
        train_relations, facts, labels,
        max_literals=MAX_LITERALS, folds=INNER_FOLDS, seed=seed, tie_tolerance=None,
        max_clauses=4, min_new_covered=MIN_NEW_COVERED, holdout_score="accuracy",
        min_fit=0.75,
    )


def run_fold(fold_index: int, train_segments: list[dict], test_segments: list[dict], seed: int) -> dict:
    """Run one fold of the pre-registered protocol end to end: fold-local
    conversion (see FOLD ISOLATION in the module docstring), EC init/term
    theory induction + holdsAt reconstruction + scoring on the held-out
    segments, and the direct-protocol reference theory on the same fold."""
    from caviar_continuous import (
        convert_continuous,
        derive_ec_masks_continuous,
        derive_ec_targets_continuous,
        reconstruct_holds_continuous,
    )
    from relational_search import make_predict_clause
    from scorer import prf1, theory_predictions

    t0 = time.perf_counter()
    train = convert_continuous(train_segments)
    test = convert_continuous(test_segments)
    wall_convert = time.perf_counter() - t0

    ec_train = derive_ec_targets_continuous(train_segments, train)
    ec_test = derive_ec_targets_continuous(test_segments, test)
    ec_masks = derive_ec_masks_continuous(train_segments, train)

    train_ec_relations = _ec_relations(train)
    test_ec_relations = _ec_relations(test)
    train_direct_relations = _direct_relations(train)
    test_direct_relations = _direct_relations(test)

    init_facts, init_labels = _exclude_dontcare(
        train["facts"], ec_train["is_init"], ec_masks["init_dontcare"],
    )
    term_facts, term_labels = _exclude_dontcare(
        train["facts"], ec_train["is_term"], ec_masks["term_dontcare"],
    )

    t0 = time.perf_counter()
    init_theory = _induce_ec_target(train_ec_relations, init_facts, init_labels, seed)
    wall_init = time.perf_counter() - t0
    t0 = time.perf_counter()
    term_theory = _induce_ec_target(train_ec_relations, term_facts, term_labels, seed)
    wall_term = time.perf_counter() - t0

    predict_clause_test_ec = make_predict_clause(test_ec_relations)
    init_pred_test = theory_predictions(init_theory["clauses"], predict_clause_test_ec, test["num_pt"])
    term_pred_test = theory_predictions(term_theory["clauses"], predict_clause_test_ec, test["num_pt"])
    holds_pred_test = reconstruct_holds_continuous(init_pred_test, term_pred_test, test_segments, test)
    ec_scoring = prf1(holds_pred_test, test["is_positive"])

    t0 = time.perf_counter()
    direct_theory = _induce_direct_theory(train_direct_relations, train["facts"], train["is_positive"], seed)
    wall_direct = time.perf_counter() - t0
    predict_clause_test_direct = make_predict_clause(test_direct_relations)
    direct_pred_test = theory_predictions(direct_theory["clauses"], predict_clause_test_direct, test["num_pt"])
    direct_scoring = prf1(direct_pred_test, test["is_positive"])

    return {
        "fold": fold_index,
        "n_train_segments": len(train_segments),
        "n_test_segments": len(test_segments),
        "n_train_pt": train["num_pt"],
        "n_test_pt": test["num_pt"],
        "ec": {
            "n_init_train": ec_train["n_init"],
            "n_term_train": ec_train["n_term"],
            "n_init_test": ec_test["n_init"],
            "n_term_test": ec_test["n_term"],
            "init_min_fit": init_theory["min_fit"],
            "term_min_fit": term_theory["min_fit"],
            "init_null_summary": init_theory["null_summary"],
            "term_null_summary": term_theory["null_summary"],
            "init_clauses": [list(c) for c in init_theory["clauses"]],
            "term_clauses": [list(c) for c in term_theory["clauses"]],
            "init_stop_reason": init_theory["stop_reason"],
            "term_stop_reason": term_theory["stop_reason"],
            "init_selection_reasons": init_theory["selection_reasons_per_iteration"],
            "term_selection_reasons": term_theory["selection_reasons_per_iteration"],
            "scoring": ec_scoring,
        },
        "direct": {
            "clauses": [list(c) for c in direct_theory["clauses"]],
            "stop_reason": direct_theory["stop_reason"],
            "scoring": direct_scoring,
        },
        "wall_clock_s": {
            "convert": wall_convert,
            "init_search": wall_init,
            "term_search": wall_term,
            "direct_search": wall_direct,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="10-fold cross-validation over the combined CAVIAR "
        "continuous corpus, under the pre-registered EC/direct protocol "
        "(see this module's own docstring) -- CPU-only, no CUDA needed.",
    )
    p.add_argument("--train-json", required=True, help="path to caviar-train.json")
    p.add_argument("--test-json", required=True, help="path to caviar-test.json")
    p.add_argument("--folds", type=int, default=10, help="number of CV folds (default: 10)")
    p.add_argument("--seed", type=int, default=7, help="seed for fold assignment, inner holdout, and the permutation RNG (default: 7)")
    p.add_argument("--out", required=True, help="path to write RESULT.json")
    return p.parse_args(argv)


def _prepare_out_path(out: str) -> Path:
    """Create --out's parent directory and write a tiny probe file BEFORE
    any expensive work starts (mirrors `run_caviar_theory.py`'s identical
    fail-fast fix)."""
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("started\n")
    return out_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_path = _prepare_out_path(args.out)

    from caviar_continuous import load_continuous

    t_start = time.perf_counter()
    train_file_segments = load_continuous(args.train_json)
    test_file_segments = load_continuous(args.test_json)
    all_segments = train_file_segments + test_file_segments

    counts = _segment_positive_counts(all_segments)
    fold_of_segment = stratified_segment_folds(counts, args.folds, args.seed)

    print(
        f"CAVIAR CV: {len(all_segments)} segments "
        f"({len(train_file_segments)} from --train-json, {len(test_file_segments)} from --test-json), "
        f"{args.folds} folds, seed={args.seed}",
        flush=True,
    )

    fold_results = []
    for fold_index in range(args.folds):
        train_segments, test_segments = _fold_segment_split(all_segments, fold_of_segment, fold_index)
        print(
            f"[fold {fold_index}] train_segments={len(train_segments)} "
            f"test_segments={len(test_segments)}",
            flush=True,
        )
        result = run_fold(fold_index, train_segments, test_segments, args.seed)
        fold_results.append(result)
        ec_s, direct_s = result["ec"]["scoring"], result["direct"]["scoring"]
        print(
            f"[fold {fold_index}] ec P/R/F1={ec_s['precision']:.4f}/{ec_s['recall']:.4f}/{ec_s['f1']:.4f} "
            f"(tp={ec_s['tp']} fp={ec_s['fp']} fn={ec_s['fn']})  "
            f"direct P/R/F1={direct_s['precision']:.4f}/{direct_s['recall']:.4f}/{direct_s['f1']:.4f} "
            f"(tp={direct_s['tp']} fp={direct_s['fp']} fn={direct_s['fn']})",
            flush=True,
        )

    ec_micro = _micro_prf1([r["ec"]["scoring"] for r in fold_results])
    direct_micro = _micro_prf1([r["direct"]["scoring"] for r in fold_results])
    total_wall = time.perf_counter() - t_start

    result = {
        "train_json": args.train_json,
        "test_json": args.test_json,
        "folds": args.folds,
        "seed": args.seed,
        "n_segments_total": len(all_segments),
        "fold_of_segment": fold_of_segment,
        "segment_positive_counts": counts,
        "fold_results": fold_results,
        "micro": {"ec": ec_micro, "direct": direct_micro},
        "wall_clock_s_total": total_wall,
    }
    out_path.write_text(json.dumps(result, indent=2))

    print()
    print(
        f"=== MICRO-AVERAGED (EC, holdsAt reconstruction, {args.folds} folds): "
        f"P={ec_micro['precision']:.4f} R={ec_micro['recall']:.4f} F1={ec_micro['f1']:.4f} "
        f"(tp={ec_micro['tp']} fp={ec_micro['fp']} fn={ec_micro['fn']}) ==="
    )
    print(
        f"=== MICRO-AVERAGED (direct-protocol reference, {args.folds} folds): "
        f"P={direct_micro['precision']:.4f} R={direct_micro['recall']:.4f} F1={direct_micro['f1']:.4f} "
        f"(tp={direct_micro['tp']} fp={direct_micro['fp']} fn={direct_micro['fn']}) ==="
    )
    print(f"wall clock total: {total_wall:.2f}s")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
