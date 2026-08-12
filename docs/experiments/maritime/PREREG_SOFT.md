# Maritime (Brest AIS) rendezVous: soft-credit column and duration-vocabulary arm

Status: **pre-registered, not yet run.** This document is committed BEFORE
any code that executes these runs on the real archives; every parameter,
mechanism, hypothesis and interpretive constraint below is fixed in
advance. Results will be appended to this document only after the
pre-registered runs complete; any deviation from the protocol below must
be recorded as a deviation, not silently amended.

The baseline this document extends is the hard-search column of
[`README.md`](README.md) (micro pointwise F1 0.6746, shipped artifact
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
[`results/ceiling_probe_duration/CEILING_PROBE_DURATION.json`](results/ceiling_probe_duration/CEILING_PROBE_DURATION.json).

**Canon: TP 3,579 / FP 22 / FN 0 -> P 0.9939 / R 1.0 / F1 0.9969.**

This deviates (upward) from the pre-declared EXPECTATION of ~0.830/0.907:
the expectation assumed `sustained_240` removes only the 2,956 short-run
false positives and leaves the 733 negative-pair ones intact. Measured:
the duration discriminator also eliminates 711 of the 733 negative-pair
false positives — the excluded pairs' body-condition episodes are
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
