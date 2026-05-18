# v0.9.0 G090_CLOSE Audit

Date: 2026-05-18

Goal node: `G090_CLOSE - Closure Proposal After v0.8.0 Rebase`

Branch: `feat/v090-epistemic-solver-semantics`

Audit scope: current semantic-oracle branch after the GPU-native correction.

## Objective Audit

The corrected goal document makes v0.9.0 closeable only after accepted epistemic
execution is fully GPU-native after parsing/planning. CPU-only or fixture-only
execution is allowed as semantic-oracle scaffolding, but it is not an acceptable
release path.

The current branch therefore cannot produce a closure proposal. It is blocked on
two independent requirements:

- `G090_GPU`: production GPU-native epistemic execution, WCOJ-backed reductions
  where eligible, GPU-resident world-view/candidate/rejection buffers, and zero
  CPU fallback counters.
- `G090_CLOSE`: rebase or merge after v0.8.0 lands, followed by v0.8
  compatibility and v0.9 certification reruns.

## Ref Evidence

| Ref | SHA |
|---|---|
| `main` | `656a8c6232f4611caf6c571eb0bcf1282e9a7339` |
| `origin/main` | `c41f9701971beb698c53beba8eb09603bb48cdf6` |
| `feat/v080-dts-ml-python-productization` | `63ef029891cc2f435cb45e524541002687ec39ee` |
| `feat/v090-epistemic-solver-semantics` | this file's containing commit after the GPU-native correction |

Earlier ref checks after `git fetch origin --prune` showed:

| Check | Result | Interpretation |
|---|---|---|
| `git merge-base --is-ancestor feat/v080-dts-ml-python-productization origin/main` | exit `1` | v0.8 branch had not landed on `origin/main`. |
| `git merge-base --is-ancestor feat/v080-dts-ml-python-productization HEAD` | exit `1` | v0.9 branch was not rebased/merged on top of v0.8. |
| `git merge-base --is-ancestor origin/main HEAD` | exit `0` | v0.9 contained current `origin/main`, but not v0.8. |

## Corrected Gate Table

| Goal | Current Status | Evidence |
|---|---|---|
| G090_PRE | PASS for inventory | Preflight evidence committed. |
| G090_EIR | PARTIAL | EIR is explicit and executable-plan lowering reaches reduced production runtime plans, but accepted epistemic forms still lack production GPU runtime dispatch. |
| G090_G91 | PASS for semantic oracle | Compatibility fixtures pass, but GPU parity remains unproven. |
| G090_FAEEL | PASS for semantic oracle | Foundedness fixtures pass, but GPU parity remains unproven. |
| G090_GPT | PARTIAL | CPU trace fixtures pass; GPU-resident candidate generation, propagation staging, candidate-buffer validation, model-membership staging, bounded world-view validation staging, accepted-candidate materialization staging, and final-result flag staging exist, but actual reduced-runtime stable-model membership population is missing. |
| G090_SPLIT | PARTIAL | CPU split/recompose fixtures pass; GPU split execution is missing. |
| G090_GPU | BLOCKED | GPU-plan, reduced-runtime-plan, workspace allocation/reset, bounded candidate-generation, propagation, candidate-validation, model-membership, world-view-validation, accepted-candidate materialization, final-result flag kernels, and hot-path transfer-budget trace with CUDA-event elapsed timing/runtime-preflight/counter-guard/reduced-plan trace contracts exist, but no production epistemic stable-model membership population/full final tuple materialization dispatch or full semantic kernel buffer use exists. |
| G090_SOLVER | BLOCKED | `SolverService` is CPU fixture enumeration; GPU-native SAT/MaxSAT/portfolio execution is not wired to epistemic candidates. |
| G090_PROB | BLOCKED | Accepted-world-view evidence fixtures exist, but accepted probabilistic epistemic execution is not proven on the GPU-native exact path. |
| G090_CERT | BLOCKED | Missing complete accepted-execution kernel timing, WCOJ evidence, zero CPU fallback counters, and post-v0.8 rerun. |
| G090_DOC | PARTIAL | Guide documents semantic oracle and blockers; production GPU/WCOJ path is not implemented. |
| G090_CLOSE | BLOCKED | Requires G090_GPU/G090_SOLVER/G090_PROB/G090_CERT plus v0.8 integration/rebase. |

## Current Semantic-Oracle Evidence

The branch contains useful scaffolding:

- explicit EIR and typed lowering boundary;
- GPU execution plan contract with required phases, buffer categories, WCOJ
  planner obligations, and zero fallback counters;
- executable lowering contract whose reduced ordinary program uses the normal
  compiler pipeline and can promote WCOJ-eligible reductions to
  `RirNode::MultiWayJoin`, including Goal-038-B K-clique planner, layout, and
  helper-splitting metadata when statistics are supplied;
- runtime GPU workspace layout/allocation API for candidate, world-view,
  model-membership, and rejection-reason buffers;
- device-side workspace reset trace with four `memset_zeros` operations and
  zero host writes;
- bounded candidate-assumption generation kernel with one launch and zero host
  writes;
- bounded propagation staging kernel with one launch and zero host writes for
  world-view/rejection buffers;
- bounded candidate-buffer validation kernel with one launch and zero host
  writes for staged candidate/world-view invariants;
- bounded model-membership staging kernel with one launch and zero host writes
  for candidate-scoped model-membership bytes;
- bounded world-view validation staging kernel with one launch and zero host
  writes for staged model-membership/world-view rejection checks;
- bounded materialization staging kernel with one launch and zero host writes
  for accepted-candidate world-view slots;
- bounded final-result flag staging kernel with one launch, one device
  row-count read from reduced output metadata, and zero host writes for
  world-view result slots;
- runtime preflight that rejects nonzero CPU fallback counters and records
  WCOJ/K-clique/helper route metadata before launch;
- runtime counter guard that refuses to certify WCOJ evidence from preflight
  metadata unless production WCOJ counters advance;
- hot-path transfer-budget trace that rejects tracked data-plane H2D/D2H
  deltas without resetting shared provider telemetry;
- reduced-plan execution trace API that wraps `execute_plan` with before/after
  production counter snapshots;
- G91 and FAEEL fixture evaluators;
- Generate-Propagate-Test phase traces;
- world-view operator fixtures for `know`, `possible`, and `not know`;
- bounded solver-service lifecycle fixtures;
- accepted-world-view probabilistic evidence fixtures.

This evidence should be retained as oracle coverage for the required GPU-native
implementation, but it cannot be used as release-close evidence.

## Missing GPU-Native Evidence

Closure remains blocked until certification includes all of the following:

- nonzero GPU launch counts and kernel timings for actual stable-model
  membership population and full final tuple materialization, plus the existing
  staging timing;
- final-result transfer accounting for complete accepted execution;
- GPU-resident candidate, world-view, model-membership, and rejection buffers;
- zero CPU fallback counters for candidate enumeration and world-view
  validation;
- at least one WCOJ-eligible epistemic reduction proving WCOJ planner/runtime
  dispatch, including layout and helper-splitting evidence where applicable;
- GPU-native SAT/MaxSAT/portfolio assumption lifecycle evidence with distinct
  SAT, UNSAT, UNKNOWN, and TIMEOUT handling;
- accepted-world-view evidence flowing into GPU-native exact/provenance
  execution with zero CPU-only probability recomputation;
- post-v0.8 rebase compatibility evidence.

## Release Hygiene

No push, tag, release-board update, merge to `main`, or v0.8-owned pyxlog API
change was performed.

## Decision

Release decision: `HOLD_FOR_GPU_NATIVE_AND_V080_REBASE`.

The current branch is semantic scaffolding. The next closing work must implement
the corrected `G090_GPU` production runtime/WCOJ/GPU path and then rerun the
full certification set after v0.8.0 integration.
