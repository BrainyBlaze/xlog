# CAVIAR: Event-Calculus rule induction with perception learned through the logic

`examples/plasticity_incircuit/` and `examples/neural_join_discovery/` plant a world and
recover the rule from it. This track runs against a **real, published benchmark** — the
CAVIAR video-surveillance corpus, the same one the OLED and WOLED papers evaluate on —
and it inherits everything that makes real data hard: a defective distributed corpus, a
double-digit count of the events you actually want to learn from, and a target that turns
out to be two disjoint regimes wearing one label.

Two things are demonstrated.

**Perception through logic credit.** A small MLP (`close_nn`) learns to read raw
`(dx, dy)` pair coordinates and decide whether two people are close, from the rule
search's gradient alone. `close`/`far`/`coords_missing` — the precomputed ground-truth
geometry — are never declared in the compiled program, never `put_relation`'d, and never
in the candidate pool. They are read once, after all training has finished, only to probe
what the network learned. On windowed fold 1 that probe scores F1 0.9865 and 0.9914
against the geometry it was never shown.

**Rules over change, and honest refusal.** The Event-Calculus protocol induces two
theories — what makes a situation start (`initiatedAt`), what makes it stop
(`terminatedAt`) — and reconstructs a per-frame answer from them by inertia. Each search
is gated by a **permutation-null fit threshold** derived per fold from 1000 label
shufflings, so a candidate must beat chance before it is reported. On the deduplicated,
leakage-free corpus it does not, and the harness returns **no rule**. That abstention is
the designed behaviour and it is the point, not a bug.

> **Read the honest scope before quoting any number.** The 0.7782 meeting F1 is measured
> on a corpus with a documented duplication defect. On the clean protocol the result is
> 0.0 — an abstention at 11 meeting / 5 moving initiations. **No ranking against OLED,
> WOLED-ASP, or hand-crafted rules is claimed or supported.** The neural detector under
> 10-fold EC cross-validation on the dump corpus is a **negative result** (F1 0.0011).
> All three are documented in full in the evidence package linked below.

## The runners

Each writes a complete `RESULT.json`; `--out` is required on all four. Every result quoted
anywhere came from one of these.

| script | what it does | device |
|---|---|---|
| `run_caviar_theory.py` | The main entrypoint. Multi-clause **theory** induction by sequential covering, in either vocabulary (`--mode relational` uses precomputed geometry, `--mode neural` trains `close_nn` instead) and under either protocol (`--protocol direct` for a per-timestep target, `--protocol ec` for `initiatedAt`/`terminatedAt` + inertia). Also `--data continuous` for the OLED narrative split, and `--max-body-literals 3` for the CPU-only 3-literal search. | CUDA, except `--max-body-literals 3` |
| `run_caviar_cv.py` | 10-fold cross-validation over a whole corpus, with every gate re-derived inside each fold. `--data-source dump` folds over video segments; `--data-source xml` rebuilds the corpus from the 30 ground-truth XML files and folds over **scene families**, so no two recordings of one staged scene are split across train and test. `--fluent {meeting,moving}`. | CPU in `--mode relational`; CUDA in `--mode neural` |
| `run_caviar_star.py` | The original probe: one **single** star clause, all-relational vocabulary. Kept as the baseline the theory loop improves on. | CUDA |
| `run_caviar_neural.py` | One single star clause, but with `close_nn` trained in place of the geometry. The narrowest demonstration that the detector learns without labels. | CUDA |
| `verify_conversion.py` | Manual eyeball check of the pkl→relations conversion. **No argparse** — positional only: `python verify_conversion.py <pkl> [fold] [split] [seed] [n]`. Prints nothing to disk. | CPU |

Several flags are **scoped and enforced at argument-parse time** — a mis-scoped
combination exits with a usage error before any work starts. `--max-body-literals 3`
requires `--mode relational --protocol ec`; `--holdout-score` and `--ec-fit-mode` require
`--max-body-literals 3`; `--ec-fit-mode permutation-null` additionally requires
`--holdout-score f1`; `--min-fit` requires `--ec-fit-mode fixed`; `--data continuous`
requires `--test-json`. On `run_caviar_cv.py`, `--train-json`/`--test-json` belong to
`--data-source dump` and `--xml-dir` to `--data-source xml`; crossing them is refused
rather than silently ignored, and `--fluent moving` needs the XML source.

## The supporting modules

All CPU-testable, none of them importing the engine:

- `caviar_convert.py` — windowed pkl → pair-time relations, plus `derive_ec_targets`
  (`is_init`/`is_term`) in the same indexing.
- `caviar_continuous.py` — the OLED/WOLED line-JSON narrative dump, which has no fixed
  window length.
- `caviar_xml_corpus.py` — the original CAVIAR ground-truth XML, per fluent, with real
  per-video boundaries and no splices.
- `theory_loop.py` — engine-agnostic sequential covering. Pure Python, no torch, no
  pyxlog.
- `relational_search.py` — CPU-only up-to-3-literal body search with the same
  tie/fit/abstain semantics as the engine's holdout arbiter.
- `scorer.py`, `ec_scorer.py` — exact set-intersection scoring, and inertia
  reconstruction of `holdsAt` from predicted events.
- `detector_probe.py` — post-hoc evidence that `close_nn` learned a distance detector,
  computed after training against relations the training never saw.

## Data

Not redistributed. You supply `caviar_folds.pkl`, the OLED `caviar-train.json` /
`caviar-test.json` pair, or the 30 CAVIAR ground-truth XML files. Sources and md5s are in
the **Data provenance** table at the top of the evidence package; `run_caviar_cv.py
--data-source xml` also reads `$CAVIAR_XML_DIR` when `--xml-dir` is omitted.

## Where to read the results

- **[The user guide](../../docs/neural/event-calculus-induction.mdx)** — start here. What
  this is for, the smallest runnable command with its real output, how the protocols and
  the fit gate work, and the limits.
- **[The evidence package](../../docs/experiments/caviar/README.md)** — the complete
  record: every protocol A–H, every committed result JSON, the corpus integrity audit,
  verbatim extracts from the OLED and WOLED papers, and the claims summary stating exactly
  what may and may not be concluded.

Result JSONs live under `docs/experiments/caviar/results/`. Reproduction commands for
every one of them are in the evidence package's final section.
