# Maritime (Brest AIS) rendezVous: single-pass online weights column

Status: **pre-registered, not yet run.** This document is committed BEFORE
any code that executes these runs on the real archives; every parameter,
mechanism, hypothesis and interpretive constraint below is fixed in
advance. Results will be appended to this document only after the
pre-registered runs complete; any deviation from the protocol below must
be recorded as a deviation, not silently amended.

The baseline this document extends is the batch soft-credit column
A-soft of [`PREREG_SOFT.md`](PREREG_SOFT.md) (micro pointwise F1 0.7398,
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
