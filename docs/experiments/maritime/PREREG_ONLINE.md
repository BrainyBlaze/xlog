# Maritime (Brest AIS) rendezVous: single-pass online weights column

Status: **pre-registered, not yet run.** This document is committed BEFORE
any code that executes these runs on the real archives; every parameter,
mechanism, hypothesis and interpretive constraint below is fixed in
advance. Results will be appended to this document only after the
pre-registered runs complete; any deviation from the protocol below must
be recorded as a deviation, not silently amended.

The baseline this document extends is the batch soft-credit column
A-soft of `PREREG_SOFT.md` (micro pointwise F1 0.7398,
shipped artifact `results/asoft_cv5/MARITIME_CV_ASOFT.json`). The column
below is the last element of the OLED/WOLED framing on this corpus:
their published numbers are F1 on held-out CV folds, and "online" in
those systems refers to the TRAINING REGIME (a single pass over the
training data), not to the evaluation protocol — this column reproduces
exactly that reading.

### 1. Column O-online

The base vocabulary (11 relations), 5-fold pair-atom CV, seed 7, the
same md5-pinned corpus, the same per-fold permutation-null gate over the
body pool (1000 permutations, p95) the batch columns use. The gate is
computed on the train fold — a train-side quantity; this makes the frame
"semi-online", which we state outright: online here is the training
regime of the WEIGHTS, while the pool is fixed exactly as in the batch
column, so that EXACTLY ONE variable is isolated (batch weights vs
single-pass weights).

### 2. Training

One pass over the train fold in ascending `pt_time` order (global time,
all the fold's pairs interleaved — as in a real stream), mini-batch =
1,000 pt rows (the last batch is the remainder), one step of the
hand-written Adam from `soft_weights` per batch (lr = 0.05 — the batch
column's parameter, NOT WOLED's AdaGrad: our comparison is the internal
online-vs-batch one, and changing the optimizer would mix two variables;
this decision is declared), init w = -2.0, decision threshold 0.5. The
run itself has no RNG (the order is fixed by the data); the seed is
needed only by the gate (7).

### 3. Metrics

The same pointwise/interval P/R/F1 on the held-out fold, per-fold +
median + micro; plus (a) the **prequential curve** of the train pass
(windowed error) — a diagnostic, with NO summary number and no claims;
(b) the wall-time of the pass — reported honestly; speed is NOT compared
with the published times (different hardware and grid).

### 4. Hypotheses (fixed before any run)

- **H-O1 (primary)**: single-pass weights lose little against batch —
  micro F1 of O-online lands in [0.70; 0.75] (degradation <= 0.04 from
  0.7398); if it falls below 0.70, that is an honest negative and we
  publish it.
- **H-O2 (order)**: the reverse-chronological pass (the diagnostic
  `--stream-order reverse`) differs by micro F1 < 0.02 — sensitivity to
  the order is small; no direction is declared.
- **H-O3**: the prequential curve decreases along the stream (learning
  actually happens).

### 5. Prohibitions

Comparison with the published 0.98 is forbidden, as before; comparing
speed with the published times is forbidden; the duration vocabulary
(`sustained_240`) in the online column is OUT OF SCOPE because of the
future leak (the relation at time t uses the FUTURE duration of its
interval — declared and explained), so the online iteration runs ONLY
on the base vocabulary, where the leak is excluded by construction.

### 6. Provenance

The code is committed before any real run; `verify_smoke` gates every
run; results are byte-exact JSONs plus an append to this document under
its own rule.

## Results (pre-registered runs of 2026-08-12, zero protocol deviations)

Shipped artifacts (byte-exact runner outputs):
`results/online_cv5/MARITIME_CV_ONLINE.json`,
`results/online_reverse_cv5/MARITIME_CV_ONLINE_REVERSE.json`.
Both ran with the pre-registered parameters (window 1,000, lr 0.05, base
vocabulary, one pass); verify_smoke gated each run.

Committed-bytes md5 pins (stored verbatim under the
`docs/experiments/maritime/results/** -text` attribute):

| artifact | md5 |
|---|---|
| `results/online_cv5/MARITIME_CV_ONLINE.json` | `f9ccbc50a4db8c1a301995f4bf235907` |
| `results/online_reverse_cv5/MARITIME_CV_ONLINE_REVERSE.json` | `d7e1c4381c6445e1a72e8745a44cfb8c` |

| column (5-fold pair-atom CV) | micro point F1 | P / R | per-fold median |
|---|---|---|---|
| A-soft, batch weights (shipped earlier) | 0.7398 | 0.8715 / 0.6426 | 0.6928 |
| **O-online, chrono single pass** | **0.7398** | 0.8715 / 0.6426 | 0.6928 |
| O-online, reverse pass (diagnostic) | 0.7398 | 0.8715 / 0.6426 | 0.6928 |

### Hypothesis verdicts

- **H-O1: CONFIRMED at the favorable edge — measured degradation is
  ZERO.** The single-pass column reproduces the batch column's
  thresholded predictions exactly (identical tp/fp/fn on every fold:
  micro 2,300 / 339 / 1,279).
- **H-O2: CONFIRMED** — |chrono − reverse| = 0.0000 < 0.02.
- **H-O3: CONFIRMED** — the prequential error rate decreases on all
  5 folds (first-half vs second-half window means, e.g. fold 0:
  0.0078 -> 0.0036).

### The identity of predictions is a finding, not a failure to run

The first suspicion any reviewer should have — "did the online path
actually run, or did it fall through to the batch path?" — is answered
by the shipped artifacts themselves: the learned WEIGHTS differ
substantially between the three regimes (fold 0's top body carries
sigma(w) = 0.6476 in batch, 0.5880 in the chrono pass, 0.8398 in the
reverse pass), the online fold records carry the stream provenance
(`stream_order`, `stream_windows`, `prequential_curve`, `wall_s_pass`),
and the pass wall-times are recorded (155.9 s across all five folds).
Three different trainings landed in the same 0.5-threshold decision
region over the same gated pool — the thresholded predictor is robust
to the training regime on this corpus. The honest scope of the claim:
prediction-identity holds for THIS pool, corpus and threshold; it is
not a general theorem.

With this column, the setting parity with the published systems is
complete on this corpus: enumerated rule search, weighted clauses, and
single-pass online training, all pre-registered. Comparison caveats are
unchanged (no comparison with the published dense-grid numbers).
