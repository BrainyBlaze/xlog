# CAVIAR experiments: canonical results

Symbolic event-rule induction on the CAVIAR video-surveillance benchmark,
with perception (a `close_nn` proximity detector over raw pair coordinates)
learned jointly through the logic credit — no distance labels ever shown to
the network. Reproduction entrypoints live in `examples/caviar_woled/`;
every number below was produced by a run whose full JSON output is in
`results/` — reproduction commands are below.

## Data provenance

| Dataset | Source | md5 |
|---|---|---|
| Windowed folds (`caviar_folds.pkl`) | github.com/nkatzz/caviar-deeppblg | `6aa3cf0f89b595db74430f12bc64f0b3` |
| Continuous narrative, train (`caviar-train.json`) | users.iit.demokritos.gr/~nkatz/oled/caviar-data.zip (the OLED paper's own data) | `5ba64bf80f135e5a874c3bac2fd0af73` |
| Continuous narrative, test (`caviar-test.json`) | same archive | `08cba3f04c2f528356cd70dd23360d5b` |

Hardware: NVIDIA A40 (RunPod), seed 7, k=4 holdout folds throughout.

**Continuous-narrative loader evidence.** `caviar_continuous.py`'s own
module docstring documents, in prose, exactly how the annotation timestamp
is aligned to the narrative grid (no shift applied) and the per-pair
init/term transition counts that verification rests on. Those counts are
cross-checked independently (own regex/segmentation pass, not a call into
`caviar_continuous.py`) by `python/tests/test_caviar_alignment_evidence.py`
— a data-gated test, skipped unless the real `caviar-train.json`/
`caviar-test.json` are available locally (see `CAVIAR_CONTINUOUS_DIR` in
that test's own docstring). Recorded counts, data md5s, and a provenance
note: `alignment_evidence.json` (this directory).

## A. Windowed folds, direct protocol (per-timestep holdsAt target)

The same two-clause theory was found on every fold in both modes and the
loop then honestly abstained:

```
holdsAt_meeting(PP,T) :- both_inactive(PP,T), close*(PP,T).
holdsAt_meeting(PP,T) :- both_active(PP,T),   close*(PP,T).
```

| test F1 | fold1 | fold2 | fold3 | mean |
|---|---|---|---|---|
| relational (precomputed close, threshold 25) | 0.9215 | 0.6804 | 0.7695 | 0.7905 |
| neural (learned close_nn, symmetrized) | 0.9215 | **0.7918** | 0.7695 | **0.8276** |

The learned detector matches ground-truth geometry on folds 1/3 and beats
the hard threshold on fold 2 (+0.11), where the soft learned boundary
survives the train/test geometry shift better. Files:
`results/caviar-s6-*.json`.

## B. Continuous OLED split, direct protocol

Single train/test split — the OLED paper's own data files.

| configuration | clauses | test F1 |
|---|---|---|
| relational, default tie floor (0.01) | 1 | 0.5995 |
| relational, tie tolerance 0.001 (re-run, see note below) | 2 | 0.7565 |
| neural (learned close_nn) | 2 — found both without tie tuning | 0.7253 |

The default tie floor was calibrated on ~10^4-fact data and swallows a
genuinely leading second clause here (margin 0.0068). The 0.001 tolerance
was pre-declared for the re-run: the value was fixed after the default run
recorded its result, triggered by the default run's train-side selection
margin (0.0068 < 0.01), before the re-run was executed. The neural mode's
soft covers clear the floor on their own. Files:
`results/caviar-e3-cont_*_direct.json`,
`results/caviar-e5-cont_rel_direct_tie001.json`.

## C. Published numbers for context (verbatim from the papers)

Frame-level holdsAt F1 on CAVIAR "meeting", whole dataset:

| system | P | R | F1 | source |
|---|---|---|---|---|
| OLED | 0.678 | 0.953 | 0.792 | OLED paper, arXiv:1608.00100v1, CAVIAR results table (Table 1 in the LaTeX source; rendered as Table 4 in the arXiv PDF), part (b) |
| OLED (as re-run in the WOLED paper) | — | — | 0.782 | WOLED paper, arXiv:2104.00158v1, Table 2 |
| WOLED-ASP | — | — | 0.887 | ibid. |
| Hand-crafted rules | — | — | 0.735 | ibid. |

**Verified, not trusted from either label alone (the "Table 1 vs Table 4"
discrepancy above).** The downloaded PDF (`oled_1608.00100.pdf`, page 10)
was inspected directly: it shows the caption "Table 4. Experimental results
from the CAVIAR dataset" over the exact same table the LaTeX source labels
`\label{table:results}` (the first — and only — table carrying that label,
"Table 1" in the source's own table-environment order; three unlabeled
table-like displays for the EC axioms/example data/action-dispatching
scheme precede it and consume table numbers 1–3 in the rendered PDF, which
is why the LaTeX-source and PDF-rendered numbers diverge). The part (b)
Meeting/OLED row in that same rendered table reads Precision 0.678, Recall
0.953, F1 0.792 — matching, digit-for-digit, the LaTeX source and the
number quoted above; this is what was checked, not assumed.

**Comparability caveats (read before comparing).** The published scores
are 10-fold cross-validation micro-averages under the papers' own best
hyper-parameter settings ("the best among several other parameter settings
that we tried" — OLED, Sect. 5); ours are a single fixed train/test split
with settings chosen before looking at results and no tuning sweep. Their
rule language allows longer bodies (bottom clauses averaging 15 literals —
OLED paper, Sec. 5 "Experimental Evaluation", `src_oled/iclp-2016.tex` line
460, the paragraph immediately after the results table); ours is capped at
2-literal conjunctions in these runs. Their F1 is computed on holdsAt
inferred from learned initiatedAt/terminatedAt via inertia; rows A/B above
use the direct per-timestep target. Numbers from the paper's
positives-only fragment regime (e.g. OLED 0.836) and from synthetic
noise-free annotation (~0.95) are NOT comparable and are deliberately
omitted. Verbatim extracts with exact table locations are in the
"Verbatim sources" section below.

## Verbatim sources

Downloaded artifacts (not shipped in this repo — arXiv e-prints/PDFs):

| Paper | arXiv id (version) | Source tarball md5 | PDF md5 |
|---|---|---|---|
| OLED: Katzouris, Artikis, Paliouras, "Online Learning of Event Definitions" (TPLP 2016) | 1608.00100v1 | `b72a3bac70afc29ea1fc05f028e37c73` (`oled_1608.00100_src.tar.gz`, extracted as `src_oled/iclp-2016.tex`) | `3efb3d5b9e9755873f3d8fedcbdf1044` (`oled_1608.00100.pdf`) |
| WOLED: Katzouris, Artikis, "Online Learning Probabilistic Event Calculus Theories in Answer Set Programming" (TPLP 2021) | 2104.00158v1 | `7ff29cdb54da3edb0eb9e73a58bb43c0` (`woled_2104.00158_src.tar.gz`, extracted as `src_woled/tplp-2020.tex`) | `ae4d1ca69538297ed8d96d9cec00b859` (`woled_2104.00158.pdf`) |

**OLED paper, results table (`\label{table:results}`, "Table 1" in the
LaTeX source's own table order; rendered as "Table 4" in the arXiv PDF —
see the verification note above), meeting rows:**

Table 1(a) — CAVIAR fragment, Sec. 5.1, verbatim LaTeX
(`src_oled/iclp-2016.tex` lines 426–429):

```
~ & \emph{Meeting} & $\mathsf{EC_{crisp}}$ & 0.687 & 0.855 & 0.762 & 23  & --\\
~ & ~ & $\mathsf{EC_{MM}}$ & 0.919 & 0.813 & \textbf{0.863} & 23  & 1133\\
~ & ~ & \xhail &  0.804 & \textbf{0.927}  & 0.861 & \textbf{15}  & 7248  \\
~ & ~ & $\mathsf{OLED}$ &  \textbf{0.943} & 0.750 & 0.836 & 29  & \textbf{23 }\\
```

Table 1(b) — whole CAVIAR dataset, Sec. 5.2, verbatim LaTeX
(`src_oled/iclp-2016.tex` lines 437–438):

```
~ & ~ & $\mathsf{EC_{crisp}}$ & 0.644 & 0.855 & 0.735 & 23 & --\\
~ & \emph{Meeting} & $\mathsf{OLED}$ & \textbf{0.678} & \textbf{0.953} & \textbf{0.792} & 30  & 107\\
```

**WOLED paper, Table 2 (`\label{table:results}`, Sec. 6.2, "Online
structure & weight learning results"), meeting rows, verbatim LaTeX
(`src_woled/tplp-2020.tex` lines 658–666):**

```
\emph{Meeting} & \tiny \textsf{WOLED-ASP} & \textbf{0.887} & 34 & \textbf{12} & -- & 82 \\
~& \tiny \textsf{WOLED-MLN} & 0.841 & 56 & 134 & 12 & 145 \\
~& \tiny \textsf{OLED} & 0.782 & 42 & 10 & -- & 36 \\
~& \tiny \textsf{HandCrafted} & 0.735 & \textbf{23} & -- & -- & -- \\
~& \tiny \textsf{HandCrafted-WL} & 0.753 & \textbf{23} & 13 & -- & \textbf{31} \\
```

**Bottom-clause size claim** ("bottom clauses averaging 15 literals",
referenced in the comparability caveats above) — verbatim LaTeX
(`src_oled/iclp-2016.tex` line 460, Sec. 5 "Experimental Evaluation",
the paragraph immediately following the results table):

```
The size of the search space (clause subsumption lattice) is determined by the size of bottom clauses, which in these experiments consisted on average of 15 literals each.
```

## D. Event-Calculus protocol (initiatedAt/terminatedAt + inertia): honest abstentions

On the continuous data the EC search abstains in every tested
configuration — 2-literal bodies, don't-care-corrected supervision,
transition vocabulary, and 3-literal bodies alike (frame F1 0.0, empty
theories). With 3-literal bodies the top initiation candidate is
`any_became_active & both_active & close` — the canonical rule shape from
the literature — at holdout score 0.9996. The two abstentions have
DIFFERENT root causes, both recorded rather than tuned away:

* **Termination**: genuine data sparsity — three single-row covers each
  catching a different one of 11 termination events tie under any metric.
* **Initiation**: the holdout METRIC, not the data — with 10 positives
  against ~23,000 rows, held-out accuracy sits on the all-false base-rate
  plateau (0.9996), and provably inverts the quality ranking: the best
  real detector by train F1 (`both_active & close`, tp 3/10) scores BELOW
  the empty predictor. The identified fix — a pre-registered
  recall-aware holdout score (per-fold F1 instead of accuracy) — would
  spread the field on the same data, and is future work, to be
  pre-registered before any run.

Files: `results/caviar-e3-cont_rel_ec_*.json`,
`results/caviar-e5-cont_*_ec_*.json`.

### D.1 Recall-aware holdout + permutation-null fit gate: the first EC theory

Both follow-ups above were then implemented under pre-registered
protocols (each configuration declared before its run; no other
configurations were executed on the real data):

1. **Per-fold-F1 holdout** (`--holdout-score f1`) removes the base-rate
   plateau: the initiation field spreads (top `both_active & close` at
   F1 0.238 with a real 0.138 margin) — but every body still fails the
   accuracy-era fit gate (0.75), which under F1 semantics on 10 positives
   demands near-perfection.
2. **Permutation-null fit gate** (`--ec-fit-mode permutation-null`,
   1000 label permutations, permutation seed 7, 95th percentile of the
   POOL-MAXIMUM mean per-fold F1, per target, same fold split): derived
   gates init 0.0444 / term 0.1250. Under these statistically-grounded
   gates:
   * **Initiation: the first non-abstaining EC theory** —
     `initiatedAt_meeting :- both_active & close` (F1 0.238 = 5.4x its
     own null gate); a second clause honestly abstains on a genuinely
     different-cover tie.
   * **Termination: still abstains** (top 0.100 < its 0.125 gate) — the
     negative is now statistically grounded, not metric-artifactual.
   * **Reconstructed holdsAt on test: P 0.984 / R 0.904 / frame F1
     0.942** (tp 442, fp 7, fn 47). Inertia is load-bearing: the same
     clause scored per-frame with NO inertia gives F1 0.341.

**Read this number with its structural caveat.** The 0.942 frame F1 is a
single-split result whose termination theory is empty (once initiated,
the fluent persists to the end of each co-visible pair-run by inertia):
this is benign here only because 96% of positive frames lie in one
interval that ends by the pair leaving co-visibility rather than by an
observed termination, and the sole interior-terminating interval never
triggers the initiation clause (had it fired there, precision would fall
to ~0.50 over its 455 post-meeting frames). The score is therefore not
evidence that terminations are learned, and is comparable to the
published OLED/WOLED numbers only if their evaluation shares this frame
universe (co-visible pair-frames) and tolerates a persistence-only
termination model. Files: `results/e9_permutation_null/*.json`.

## E. 10-fold cross-validation over the whole corpus (protocol-matched)

The combined corpus (train + test dumps, 26 video segments, 32,360
pair-frames, 1,833 gold meeting frames) was cross-validated 10-fold BY
VIDEO SEGMENT (seeded descending round-robin balance), with every gate
re-derived inside each fold: per-fold permutation-null fit thresholds
(1000 permutations, seed 7, p95 pool-max, per target, train-side only),
F1 holdout, inner k=4, all pre-registered. Micro-aggregation over
accumulated tp/fp/fn, as in the papers.

| protocol (10-fold CV micro) | P | R | F1 |
|---|---|---|---|
| EC + inertia (this work) | 0.658 | 0.827 | **0.733** |
| direct relational reference (pre-registered defaults) | 0.686 | 0.127 | 0.214 |
| OLED (published, their CV regime) | 0.678 | 0.953 | 0.792 |
| hand-crafted rules (published) | — | — | 0.735 |
| WOLED-ASP (published) | — | — | 0.887 |

The same initiation clause `both_active & close` was selected on ALL ten
folds (with its own per-fold null gate each time); the termination theory
is empty on all ten. The single-split 0.942 from section D.1 is hereby
SUPERSEDED as a headline: cross-validation exposes exactly the structural
shield D.1's caveat described — 98% of the 788 false positives come from
three folds where detected interior terminations flood unterminated
persistence, dropping precision from 0.984 (single split) to 0.658.

**Comparability caveat.** 0.733 micro-F1 (10-fold CV over video segments,
pre-registered gates/seeds, single 2-literal initiation clause with an
EMPTY termination theory — the fluent persists to the end of each
co-visible pair-run, costing 788 of 2,304 predicted-positive frames as
false positives on folds with detected interior terminations) is
comparable to OLED's 0.792 only up to protocol deltas: OLED
cross-validates over windowed interpretations with subsampled negatives,
learns full initiation AND termination programs in a richer rule
language, and reports tuned rather than pre-registered settings. The
direct-protocol reference of 0.214 reflects pre-registered defaults
(accuracy holdout, fit gate 0.75, default tie floor — dominated by one
fold's total abstain), NOT that protocol's ceiling; sections A/B show
what the direct protocol reaches under its own studied settings.

Files: `results/e10_cv/caviar-e10-cv10.json`.

## Reproduction

```
# windowed folds, direct protocol
python examples/caviar_woled/run_caviar_theory.py --mode neural --protocol direct \
  --pkl caviar_folds.pkl --fold fold1 --k 4 --seed 7 --steps 400 --hidden 16 \
  --max-clauses 4 --out RESULT.json

# continuous OLED split, direct protocol, tie tolerance pre-declared for the re-run (see section B)
python examples/caviar_woled/run_caviar_theory.py --mode relational --protocol direct \
  --data continuous --pkl caviar-train.json --test-json caviar-test.json \
  --k 4 --seed 7 --steps 400 --max-clauses 4 --tie-tolerance 0.001 --out RESULT.json

# continuous, EC protocol, 3-literal relational search (CPU-only)
python examples/caviar_woled/run_caviar_theory.py --mode relational --protocol ec \
  --data continuous --pkl caviar-train.json --test-json caviar-test.json \
  --k 4 --seed 7 --min-new-covered 2 --max-body-literals 3 --out RESULT.json

# continuous, EC protocol, F1 holdout + permutation-null fit gate (section D.1; CPU-only)
python examples/caviar_woled/run_caviar_theory.py --mode relational --protocol ec \
  --data continuous --pkl caviar-train.json --test-json caviar-test.json \
  --k 4 --seed 7 --min-new-covered 2 --max-body-literals 3 \
  --holdout-score f1 --ec-fit-mode permutation-null --out RESULT.json
```

GPU runs need CUDA (see each script's docstring); the 3-literal EC search
runs on CPU.

# 10-fold CV over the whole corpus, EC protocol (section E; CPU-only)
python examples/caviar_woled/run_caviar_cv.py   --train-json caviar-train.json --test-json caviar-test.json   --folds 10 --seed 7 --out RESULT.json
