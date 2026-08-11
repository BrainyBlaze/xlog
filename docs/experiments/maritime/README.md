# Maritime (Brest AIS) rendezVous: cross-validated rule reconstruction

Status: **pre-registered; the run has completed.** The pre-registration
below was committed BEFORE `examples/maritime_woled/run_maritime_cv.py`
executed on the real corpus (commit `90d44107`); every parameter, metric,
and interpretive constraint was fixed in advance and none has been
amended since. The run executed on 2026-08-08 with zero protocol
deviations; results are in the "Results" section at the end of this
document, and the shipped artifact is
`results/baseline_cv5/MARITIME_CV_BASELINE.json`.

## Corpus and provenance

- Archive A (HLE composite events): `MaritimeCompositeEvents.tar.gz`,
  md5 `05e23621b87fcec211a7ff4ed4397b94`.
- Archive B (LLE critical events): `brest_critical.zip`,
  md5 `0b239b8eb212dc433e67de6a599b2d10`.
- Converter: `examples/maritime_woled/maritime_convert.py` (commit
  `e16defed`). Verifier: `examples/maritime_woled/
  verify_maritime_conversion.py` (commit `0fbf449a`) — the exact revision
  that produced the real-data verification report shipped at
  [`results/verification/MARITIME_VERIFY.json`](results/verification/MARITIME_VERIFY.json)
  (md5 of the committed bytes: `c17b82672bc5144a61eeddf66b9ca21c`).
- Converted-corpus constants (verified, adversarially re-derived by an
  independent rebuild — see the task-2 adversarial review): **454,858**
  pair-time rows; **806** vessel pairs = **302** positive (>=1 rendezVous
  interval) + **504** negative (from a pool of **2,014** proximity-only
  pairs, deterministic stride-4 subsample); **3,548** gold rendezVous
  intervals; **3,579** positive pair-time rows (~0.79% of all rows).
- Relation vocabulary (11 unary pt-sets, fixed by the converter):
  `proximity`, `far`, `both_lowspeed`, `both_stopped_far`,
  `both_low_or_stopped`, `either_low_or_stopped`, `any_near_ports`,
  `both_open_sea`, `became_far`, `became_proximate`, `any_slow_ended`.
  The vocabulary does NOT contain the gold rule's two remaining
  discriminators: the 240 s minimum-duration threshold and the
  tug/pilot pair exclusion (see "Expected ceiling" below).

## Pre-registration

### (a) Metric and expected ceiling

- Primary metric: **pointwise F1 on the positive class** (per pair-time
  row), per held-out fold. Never accuracy: positives are 0.79% of rows, so
  accuracy is saturated by the empty predictor.
- **Declared ceiling: ~0.66 pointwise F1.** The exact definitional body of
  the gold rule (`proximity AND both_low_or_stopped AND both_open_sea`)
  achieves recall 1.0 but pointwise precision ~0.49 on this corpus
  (TP 3,579 / FP 3,689 — adversarial-review finding V4-2), because the two
  remaining gold discriminators (240 s duration threshold, tug/pilot pair
  exclusion) are absent from the relation vocabulary. Perfect
  reconstruction of gold inside this vocabulary is IMPOSSIBLE by
  construction. Success is therefore defined as **approaching the
  vocabulary ceiling (~0.66)**, not approaching 1.0.
- **Comparison with the published WOLED figure (F1 = 0.98) is forbidden.**
  That number was computed on a different, dense critical-point grid; this
  corpus's grid has ~1.009 positive pt per gold interval (3,579 pt /
  3,548 intervals), with the covering point almost always the interval's
  own `st` boundary. The two numbers do not live on the same axis and must
  never appear in the same table without this caveat.
- Grid consequence, declared up front: at ~1.009 pt/interval,
  **pointwise F1 ~= interval F1** on this grid — long intervals get no
  extra weight, but the per-frame reading of the published protocol is NOT
  reproduced. The runner additionally reports an **interval-level
  aggregate** (definition below), and any future direct-vs-EC comparison
  MUST be scored interval-based: within a gold interval there are almost
  no interior points for inertia to fill, so a pointwise direct-vs-EC
  delta on this grid would be grid noise, not evidence.
- Interval-level scoring (fixed definition): within each converter
  segment of the held-out fold, gold intervals are maximal runs of
  consecutive gold-positive pt rows, predicted intervals are maximal runs
  of consecutive predicted-positive pt rows; a predicted interval is
  matched iff it shares >=1 pt row with some gold interval, and vice
  versa; interval precision = matched predicted / total predicted,
  interval recall = matched gold / total gold, interval F1 harmonic.

### (b) Fold grouping: strictly by vessel pair, stratified by positive mass

- Fold atoms are **vessel pairs** — a pair's pt rows are NEVER split
  across folds. Rationale (adversarial review V6): the top-1 pair
  (`227226450|227369960`) carries **33.4%** of all positive pt, the top-4
  pairs carry **60.2%**; any pt- or interval-level split leaks the
  dominant scene across the train/test boundary.
- Assignment (deterministic greedy LPT, no RNG):
  1. positive pairs sorted by (positive-pt count descending, pair id
     ascending), each dealt to the fold with the smallest running
     positive-pt sum (ties: lowest fold index);
  2. negative pairs then sorted by (pt-row count descending, pair id
     ascending), each dealt to the fold with the smallest running
     negative-pt-row sum (ties: lowest fold index).
  Guaranteed property (tested): for both loads, max fold sum − min fold
  sum <= the largest single pair's own count — i.e. the top pair dominates
  its fold only by the unavoidable amount, never through assignment skew.
- Reporting: because fold positive masses are necessarily unequal (the
  top-pair fold alone holds ~33% of positives), results are reported as
  **per-fold P/R/F1 PLUS the median and spread (min/max) of per-fold F1**,
  alongside the micro-average (tp/fp/fn summed across folds, then one
  P/R/F1) — never the micro number alone.

### (c) Determinism and seeds

- The converter and the pair scope contain **no RNG at all** (verified by
  the adversarial review, V7): negative-pair selection is a deterministic
  stride over a lexicographically sorted pool; `convert` is a pure
  function of the two md5-pinned archives. **No seed exists or is needed
  for the corpus**, stated here explicitly.
- The fold assignment above is likewise deterministic (greedy LPT, no
  RNG) — no seed enters the outer split.
- The ONLY randomized components are inside the search: the inner holdout
  fold assignment and the permutation-null RNG. Both are governed by the
  single published seed **7** (`--seed 7`), threaded exactly as in
  `run_caviar_cv.py`.
- The factual negative-pair count is **504** (not the documented cap of
  604 — stride arithmetic, adversarial-review finding V4-1); 504 is the
  pre-registered number.

### (d) Synthetic supervision caveat

The gold `rendezVous` labels are the output of a hand-crafted RTEC rule
evaluated over the SAME interval streams from which the relation
vocabulary is derived (archive A fluents; archive B proximity is the same
pipeline's input export). The verifier's alignment fractions of 1.0 are
therefore **conversion fidelity, not label validation**: they prove the
converter's interval arithmetic is faithful, and prove nothing about
annotation quality or task difficulty (adversarial review V1). The task
this experiment measures is **reconstruction of a known hand-crafted rule
under an incomplete vocabulary** (missing the 240 s threshold and the
tug/pilot exclusion) — nothing more.

### (e) Provenance requirements (all must hold BEFORE the real run)

1. Verifier committed: `0fbf449a` (done).
2. The CV runner and any format adapter (`run_maritime_cv.py`) committed
   before the real-data run; the run must use the committed revision.
3. Immediately before CV, the runner re-executes a **verifier smoke**: md5
   of both archives against the verifier's pinned constants, the
   verifier's own hard invariants (pair contiguity, segment/pair
   boundaries, EC label-count consistency, reusing the committed
   verifier's functions) on the freshly converted corpus, and exact
   equality of the converted counts against the pre-registered constants
   (n_pt 454,858; pairs 806 = 302 + 504; negative pool 2,014; positive pt
   3,579). Any mismatch aborts the run. `--skip-verify` exists for unit
   tests on synthetic archives ONLY and stamps an explicit warning into
   the result JSON.
4. The result JSON records: both archive md5s, every parameter in (f),
   the full fold assignment (pair -> fold), the actual relation
   vocabulary searched, per-fold and aggregate scores, and the verifier
   smoke report (or the skip warning).

### (f) Fixed run parameters

| Parameter | Value | Note |
|---|---|---|
| Outer folds | 5 | pair-atom folds; 5 (not CAVIAR's 10) because positive mass is concentrated (top pair 33.4%) and each fold must retain non-top positive pairs |
| Seed | 7 | inner holdout + permutation RNG only (see (c)) |
| Inner holdout folds | 4 | `run_caviar_cv.INNER_FOLDS` convention |
| `min_new_covered` | 2 | `run_caviar_cv.MIN_NEW_COVERED` convention |
| `tie_tolerance` | derived default | `None` -> `max(0.01, 1 / n_residual_facts)` per iteration, never a caller constant |
| `holdout_score` | `f1` | rare-positive holdout; accuracy plateau cannot rank candidates |
| `max_literals` | 3 | body arity cap, as CAVIAR |
| `max_clauses` | 4 | sequential-covering cap, as CAVIAR |
| Fit gate | permutation-null | `permutation_null_threshold`: 1000 permutations, threshold = 0.95 quantile of the pool-max per-fold-mean F1, `perm_seed = seed`, derived fresh per fold from that fold's own training facts/labels |
| Target | direct `is_positive` | one relational search per fold over the 11-relation vocabulary; EC induction is out of scope for this run (see (a) for the interval-scoring requirement any future EC comparison must meet) |
| Conversion | once per run | one `maritime_convert.convert` call; folds are pair-level slices of that single corpus. Fold isolation is structural: every pt row belongs to exactly one pair (pair-contiguity invariant), every relation is derived per pair/per episode by the converter, and segments never span pairs — so slicing by pair cannot leak rows or relation memberships across folds |

## Planned run

```
py -3.13 examples/maritime_woled/run_maritime_cv.py \
    --tar <MaritimeCompositeEvents.tar.gz> --zip <brest_critical.zip> \
    --folds 5 --seed 7 --min-new-covered 2 \
    --out maritime-cv5-relational.json
```

Results will be appended to this document only after the pre-registered
run completes; any deviation from the protocol above must be recorded as
a deviation, not silently amended.

## Results (pre-registered run of 2026-08-08)

Shipped artifact: [`results/baseline_cv5/MARITIME_CV_BASELINE.json`](results/baseline_cv5/MARITIME_CV_BASELINE.json)
(md5 of the committed bytes: `ac650dc5eb293d1f9f55df4bec6ae1bf`).
EOL provenance: the Windows runner wrote the file with CRLF line
endings and the previous pin (`1d3b7fcc56b0032083759100793b743e`) was
computed on that CRLF original; git normalized the line endings to LF
at commit, so the committed bytes hash differently while the content is
identical — the reviewer's diagnosis, confirmed by reconstructing the
CRLF bytes and matching the old pin exactly. Shipped artifacts under
`results/` are now marked `-text` in `.gitattributes`, so future
artifacts are committed byte-exact and this failure class is closed.

### Provenance

- Code revision: `73a6a041` (this branch's tip at run time; converter
  `e16defed` and verifier `0fbf449a` as pre-registered above). The
  working tree was clean.
- Invocation: `run_maritime_cv.main(["--tar", <A>, "--zip", <B>,
  "--out", <artifact>])` — the CLI defaults for every remaining flag
  equal the section-(f) values, and the artifact's `params` block
  records the resolved configuration (`folds=5, seed=7,
  min_new_covered=2, tie_tolerance=null, holdout_score="f1",
  inner_folds=4, max_literals=3, max_clauses=4,
  null_permutations=1000, null_quantile=0.95, smoke=false`), which
  matches section (f) field-for-field.
- `verify_smoke` gate: passed before the search ran (both archive md5s
  matched the pins above; pair-contiguity, segment-boundary and
  EC-label invariants all held; converted-corpus counts matched the
  pre-registered constants).
- Wall time: conversion 1,437 s; search ~2.2 min per fold (CPU only).
- **Protocol deviations: none.**
- **Provenance correction (2026-08-11):** as of commit `10b180d6` this
  document called the `MARITIME_VERIFY.json` real-data report
  "committed". That was false — the file had never been committed in any
  revision and existed only on the runner's machine, while carrying the
  sole backing for the alignment fractions of 1.0 and the zero
  episode-boundary violations cited in §(d) and §(e). The report is
  shipped by the commit that adds this note, byte-exact as produced by
  the run (md5 `c17b82672bc5144a61eeddf66b9ca21c`), at
  [`results/verification/MARITIME_VERIFY.json`](results/verification/MARITIME_VERIFY.json).

### Headline numbers

| Metric (5-fold CV, pair-atom folds) | Value |
|---|---|
| micro pointwise F1 (positive class) | **0.6746** |
| micro pointwise precision / recall | 0.5109 / 0.9925 |
| micro pointwise tp / fp / fn | 3,552 / 3,400 / 27 |
| micro interval F1 | 0.6772 (3,545 of 3,548 gold intervals matched) |
| per-fold pointwise F1 | 0.9933 / 0.4690 / 0.6685 / 0.5699 / 0.6596 |
| per-fold median (min–max) | **0.6596** (0.4690–0.9933) |

### Reconstructed theory

Sequential covering reconstructed the gold rule's definitional body on
**every** fold, decomposed into its disjunctive branches as separate
clauses (all bodies conjoin `proximity` and `both_open_sea` with a
speed/stopped condition):

- `both_open_sea AND both_stopped_far AND proximity` — committed on all
  5 folds (the "stopped" branch);
- `both_lowspeed AND both_open_sea AND proximity` — committed on all
  5 folds (the "lowSpeed" branch);
- folds 0 and 2 additionally committed
  `both_low_or_stopped AND both_open_sea AND proximity` as a residual
  mop-up clause (the union relation subsumes both branches);
- fold 4 committed a transition-flavoured variant of the stopped
  branch first (`became_proximate AND both_open_sea AND
  both_stopped_far`), then the two canonical branches.

Stop reasons were the honest ones: "no positives remain in the
residual" (folds 0, 2) and `select_once` tie-abstention on the residual
tail (folds 1, 3, 4) — no clause was ever committed below its fold's
permutation-null gate.

### Reading against the pre-registered ceiling

Section (a) declared, before the run, a vocabulary ceiling of ~0.66
pointwise F1 with the mechanism "precision ~0.49 at recall ~1.0,
because the 240 s duration threshold and the tug/pilot exclusion are
absent from the vocabulary". Measured: precision 0.5109, recall
0.9925, F1 0.6746 — the aggregate operating point landed where the
pre-declared ceiling mechanism predicted, and only 27 of 3,579
positive pair-times were missed. That agreement is aggregate-level
evidence only: whether the residual 3,400 false positives decompose
into the two pre-identified inexpressible discriminators is measured
component-wise by
[`examples/maritime_woled/ceiling_probe.py`](../../../examples/maritime_woled/ceiling_probe.py)
(its real-corpus JSON lands here after the probe's separate run), not
asserted from the aggregate. Note also that the learned theory is NOT
literally the definitional body: its micro-precision 0.5109 differs
from the exact body's predicted 0.4924 — the per-branch decomposition
plus the folds' mop-up clauses yield a measurably different operator,
so the ceiling reading is about the mechanism, not about operator
identity. Per section (a), this number must NOT be compared with the
published dense-grid figure (0.98). The fold spread (0.469–0.993) is
the declared pair-concentration effect: the fold holding the top pair
(33.4% of positive mass, long clean encounters) scores 0.993, while
folds dominated by many small pairs concentrate the
duration-threshold false positives.

Any improvement past this baseline must come from added
expressiveness (duration discriminators in the vocabulary, or
soft/weighted clauses), not from tuning the search — that is the
pre-registered interpretation of "ceiling reached".
