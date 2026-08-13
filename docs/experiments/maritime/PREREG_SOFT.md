# Maritime (Brest AIS) rendezVous: soft-credit column and duration-vocabulary arm

Status: **pre-registered, not yet run.** This document is committed BEFORE
any code that executes these runs on the real archives; every parameter,
mechanism, hypothesis and interpretive constraint below is fixed in
advance. Results will be appended to this document only after the
pre-registered runs complete; any deviation from the protocol below must
be recorded as a deviation, not silently amended.

The baseline this document extends is the hard-search column of
`README.md` (micro pointwise F1 0.6746, shipped artifact
`results/baseline_cv5/MARITIME_CV_BASELINE.json`).

### (a) Three pre-registered columns

All three columns run on the byte-identical protocol of the baseline:
5-fold pair-atom CV, seed 7, the same md5-pinned corpus, the same
metrics (pointwise + interval, per-fold P/R/F1 + median/min/max of
per-fold F1 + micro):

- **A-soft**: the 11-relation vocabulary (as the baseline), soft weights;
- **B-hard**: the vocabulary + `sustained_240`, hard search (the
  byte-same `run_maritime_cv`, only the vocabulary wider);
- **B-soft**: the vocabulary + `sustained_240`, soft weights.

### (b) Mechanism of A-soft / B-soft

The body pool is every conjunction of 1..3 literals over the vocabulary
that passed THE SAME per-fold permutation-null gate (1000 permutations,
p95 of the pool-max per-fold F1, `perm_seed = seed`) the hard search
uses; one weight per body; the prediction is

    score(pt) = 1 - PROD_c (1 - sigmoid(w_c) * cover_c(pt))

(noisy-OR — the same semantics as the engine's relational
real-credit columns, `pyxlog.ilp.neural_credit.credit_nll`). Training is
BCE, Adam, **steps = 300, lr = 0.05, seed = 7**,
`torch.use_deterministic_algorithms(True)`, CPU; decision threshold 0.5.
Initialization w = -2.0 (every clause "off" — a sparse start).

### (c) `sustained_240` (Arm B)

For each pair: the intersection of the CONTINUOUS intervals
`proximity ∩ both_low_or_stopped ∩ both_open_sea` (interval algebra on
the converter's merged interval lists); a pt row receives the relation
if it lies in an intersection component whose duration is
`et - st >= 240` seconds. The threshold 240 is the gold generator's
constant (`rendezvousTime`) — LANGUAGE parity with RTEC, not a peek into
the data; the tie `== 240` is included (gold: the minimal interval
duration is 241 s by census — the defining body's recall is preserved).

### (d) Hypotheses (fixed before any run)

- **H-A**: A-soft does not substantially exceed the 0.66 ceiling (the
  ceiling is a property of the vocabulary; expectation: micro F1 within
  [0.66, 0.70]); a negative result is published.
- **H-B-ceiling**: the new Arm-B ceiling, derived by the extended
  `ceiling_probe` BEFORE any CV run, is expected to be approximately
  P 3,579/(3,579+733) = 0.830 / R 1.0 / F1 ≈ 0.907 (the short-run 2,956
  FP are cut off; the pair-level 733 remain inexpressible). The canon is
  the number in the committed probe JSON, not this expectation.
- **H-B-delta** (PRIMARY, the WOLED delta): B-soft >= B-hard on the
  per-fold median; the direction is declared, the magnitude is a
  measurement.

### (e) Prohibitions

Comparison with the published 0.98 is forbidden (a different grid);
column comparisons are internal only, on identical folds.

### (f) Provenance

The code is committed before any real run; `verify_smoke` gates every
run; results are byte-exact JSONs in `results/` plus an append to the
README.

## Ceiling canon for Arm B (derived and committed BEFORE any CV run)

Per section (d), the canonical Arm-B ceiling is the committed output of
`ceiling_probe.py --vocab duration` on the pinned archives:
`results/ceiling_probe_duration/CEILING_PROBE_DURATION.json`.

**Canon: TP 3,579 / FP 22 / FN 0 -> P 0.9939 / R 1.0 / F1 0.9969.**

This deviates (upward) from the pre-declared EXPECTATION of ~0.830/0.907:
the expectation assumed `sustained_240` removes only the 2,956 short-run
false positives and leaves the 733 negative-pair ones intact. Measured:
the duration discriminator also eliminates at least 711 of the 733
negative-pair
false positives ("at least": the artifact decomposes the BASE body's
false positives; the duration body's residual 22 are not themselves
decomposed, and a positive-pair FP with a short grid run can still sit
inside a >=240 s continuous component) — the excluded pairs' body-condition episodes are
themselves almost always shorter than 240 s on the continuous streams —
leaving 22. Two consequences, recorded before the CV runs:

1. the pair-exclusion discriminator is largely REDUNDANT with duration on
   this corpus (a finding, not an assumption);
2. Arm B operates near saturation (ceiling 0.9969), which mirrors the
   published maritime setting (all published systems ~0.98 on their
   grid); the non-saturated arena for the soft-vs-hard comparison is
   therefore Arm A (base vocabulary, ceiling ~0.66), and H-B-delta must
   be read with saturation in mind. Hypotheses H-A, H-B-delta and all
   run parameters are unchanged.

## Results (pre-registered runs of 2026-08-12, zero protocol deviations)

Shipped artifacts (byte-exact runner outputs, `-text` attribute):
`results/asoft_cv5/MARITIME_CV_ASOFT.json`,
`results/bhard_cv5/MARITIME_CV_BHARD.json`,
`results/bsoft_cv5/MARITIME_CV_BSOFT.json`.
All three ran with the pre-registered parameters (recorded in each
artifact's `params` block); verify_smoke gated each run.

Committed-bytes md5 pins (all four stored verbatim under the
`docs/experiments/maritime/results/** -text` attribute):

| artifact | md5 |
|---|---|
| `results/asoft_cv5/MARITIME_CV_ASOFT.json` | `6712569558ebf81a203708f1415407a3` |
| `results/bhard_cv5/MARITIME_CV_BHARD.json` | `bed0fc9588d7c2fc20ebd9bc83526152` |
| `results/bsoft_cv5/MARITIME_CV_BSOFT.json` | `3160c5567e3758cc41ef3d6c237af600` |
| `results/ceiling_probe_duration/CEILING_PROBE_DURATION.json` | `0ca0666a64bb1cac346d71ff7211798e` |

| column (5-fold CV, pair-atom) | micro point F1 | P / R | interval F1 | per-fold median |
|---|---|---|---|---|
| hard baseline (base vocab, shipped earlier) | 0.6746 | 0.5109 / 0.9925 | 0.6772 | 0.6596 |
| **A-soft** (base vocab, weighted clauses) | **0.7398** | 0.8715 / 0.6426 | 0.7435 | 0.6928 |
| **B-hard** (duration vocab, crisp) | **0.9968** | 0.9936 / 1.0 | 0.9970 | 0.9942 |
| **B-soft** (duration vocab, weighted) | **0.9968** | 0.9936 / 1.0 | 0.9970 | 0.9942 |

Section (f) also requires an append to `README.md`; that append was
initially omitted and is added with this fix — recorded here per this
document's own no-silent-amendment rule, not slipped in.

### Hypothesis verdicts

- **H-A: REJECTED, in the favorable direction.** The expectation band
  ([0.66, 0.70]) is exceeded: A-soft scores micro 0.7398, +0.0652 over
  the crisp baseline on the SAME vocabulary, same folds, same gate.
  The explanation is an expectation error, not a protocol one: the
  "~0.66 vocabulary ceiling" is the F1 of the definitional-body
  OPERATING POINT (crisp reconstruction at recall ~1.0). Weighted
  clauses are not confined to that point — training moved the operating
  point to precision 0.8715 / recall 0.6426, which carries higher F1.
  Weights over the same clauses beating the crisp selection is exactly
  the OLED-to-WOLED published delta, here measured on a corpus with
  3,548 gold intervals. Honest per-fold nuance: A-soft wins the median
  (0.6928 vs 0.6596) and the micro, but wins only 2 of 5 individual
  folds (0.8000 vs 0.6685; 0.6928 vs 0.5699) — the crisp column keeps
  folds 0, 1, 4. The claim is therefore "weights beat crisp on
  aggregate and median", not "on every fold".
- **H-B-ceiling: CONFIRMED.** B-hard lands at 0.9968 against the
  pre-committed canon 0.9969 (fp 23 vs the canon's 22 — per-fold
  gating admits one extra false positive that the global definitional
  body does not).
- **H-B-delta: holds trivially at saturation.** B-soft equals B-hard on
  every fold — with `sustained_240` in the vocabulary both columns
  recover gold nearly perfectly, so there is no headroom for weights to
  add anything. This is the saturation consequence recorded in the
  ceiling-canon section before the runs, and it mirrors the published
  maritime setting (~0.98 for every published system on their grid).
  The informative weights-vs-crisp comparison on this corpus is Arm A.

Interpretation caveats carried over: no comparison with the published
0.98 (different grid); soft weights are shared among extensionally
equivalent bodies (identical-coverage conjunctions split the weight of
a clause — read `weights_top10` as evidence of which coverage matters,
not as a unique clause ranking).
