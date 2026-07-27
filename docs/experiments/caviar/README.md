# CAVIAR experiments: canonical results

Symbolic event-rule induction on the CAVIAR video-surveillance benchmark,
with perception (a `close_nn` proximity detector over raw pair coordinates)
learned jointly through the logic credit — no distance labels ever shown to
the network. Reproduction entrypoints live in `examples/caviar_woled/`;
every number below was produced by a run whose full JSON output is in
`results/` and was predicted by an independent CPU simulation before the
GPU run.

## Data provenance

| Dataset | Source | md5 |
|---|---|---|
| Windowed folds (`caviar_folds.pkl`) | github.com/nkatzz/caviar-deeppblg | `6aa3cf0f89b595db74430f12bc64f0b3` |
| Continuous narrative, train (`caviar-train.json`) | users.iit.demokritos.gr/~nkatz/oled/caviar-data.zip (the OLED paper's own data) | `5ba64bf80f135e5a874c3bac2fd0af73` |
| Continuous narrative, test (`caviar-test.json`) | same archive | `08cba3f04c2f528356cd70dd23360d5b` |

Hardware: NVIDIA A40 (RunPod), seed 7, k=4 holdout folds throughout.

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
| relational, pre-registered tie tolerance 0.001 | 2 | 0.7565 |
| neural (learned close_nn) | 2 — found both without tie tuning | 0.7253 |

The default tie floor was calibrated on ~10^4-fact data and swallows a
genuinely leading second clause here (margin 0.0068); the 0.001 value was
pre-registered before the run. The neural mode's soft covers clear the
floor on their own. Files: `results/caviar-e3-cont_*_direct.json`,
`results/caviar-e5-cont_rel_direct_tie001.json`.

## C. Published numbers for context (verbatim from the papers)

Frame-level holdsAt F1 on CAVIAR "meeting", whole dataset:

| system | P | R | F1 | source |
|---|---|---|---|---|
| OLED | 0.678 | 0.953 | 0.792 | OLED paper (arXiv 1608.00100), Table 1(b) |
| OLED (as re-run in the WOLED paper) | — | — | 0.782 | WOLED paper (arXiv 2104.00158), Table 2 |
| WOLED-ASP | — | — | 0.887 | ibid. |
| Hand-crafted rules | — | — | 0.735 | ibid. |

**Comparability caveats (read before comparing).** The published scores
are 10-fold cross-validation micro-averages under the papers' own best
hyper-parameter settings ("the best among several other parameter settings
that we tried" — OLED, Sect. 5); ours are a single fixed train/test split
with pre-registered settings and no tuning sweep. Their rule language
allows longer bodies (bottom clauses averaging 15 literals); ours is capped
at 2-literal conjunctions in these runs. Their F1 is computed on holdsAt
inferred from learned initiatedAt/terminatedAt via inertia; rows A/B above
use the direct per-timestep target. Numbers from the paper's
positives-only fragment regime (e.g. OLED 0.836) and from synthetic
noise-free annotation (~0.95) are NOT comparable and are deliberately
omitted. Verbatim extracts with table locations: the PR that added this
directory records them; sources are public arXiv ids above.

## D. Event-Calculus protocol (initiatedAt/terminatedAt + inertia): honest abstentions

On the continuous data the EC search abstains in every tested
configuration — 2-literal bodies, don't-care-corrected supervision,
transition vocabulary, and 3-literal bodies alike (frame F1 0.0, empty
theories). With 3-literal bodies the top initiation candidate is
`any_became_active & both_active & close` — the canonical rule shape from
the literature — at holdout score 0.9996, but ~10 true initiation events
in the whole training split cannot single out one winner among dozens of
near-tied distinct covers, and the arbiter refuses to guess. This is
recorded as a data-sparsity boundary (and, at 2 literals, an
expressiveness boundary), not tuned away. Files:
`results/caviar-e3-cont_rel_ec_*.json`, `results/caviar-e5-cont_*_ec_*.json`.

## Reproduction

```
# windowed folds, direct protocol
python examples/caviar_woled/run_caviar_theory.py --mode neural --protocol direct \
  --pkl caviar_folds.pkl --fold fold1 --k 4 --seed 7 --steps 400 --hidden 16 \
  --max-clauses 4 --out RESULT.json

# continuous OLED split, direct protocol, pre-registered tie tolerance
python examples/caviar_woled/run_caviar_theory.py --mode relational --protocol direct \
  --data continuous --pkl caviar-train.json --test-json caviar-test.json \
  --k 4 --seed 7 --steps 400 --max-clauses 4 --tie-tolerance 0.001 --out RESULT.json

# continuous, EC protocol, 3-literal relational search (CPU-only)
python examples/caviar_woled/run_caviar_theory.py --mode relational --protocol ec \
  --data continuous --pkl caviar-train.json --test-json caviar-test.json \
  --k 4 --seed 7 --min-new-covered 2 --max-body-literals 3 --out RESULT.json
```

GPU runs need CUDA (see each script's docstring); the 3-literal EC search
runs on CPU.
