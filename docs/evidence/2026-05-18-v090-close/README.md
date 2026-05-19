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
| G090_GPT | PARTIAL | CPU trace fixtures pass; GPU-resident candidate generation, propagation staging, candidate-buffer validation, arity 0-3 tuple-source model-membership staging with fixed arity-one/two/three row-scoped ground key comparison, generic arity-N variable-bound tuple matching, bounded world-view validation staging, accepted-candidate materialization staging, final-result flag staging, final-row map construction, and final tuple materialization exist; unary/binary/multi-membership final-row filtering fixtures pass, but broader semantic parity remains missing. |
| G090_SPLIT | PARTIAL | CPU split/recompose fixtures pass; GPU split execution is missing. |
| G090_GPU | BLOCKED | GPU-plan, reduced-runtime-plan, workspace allocation/reset, bounded candidate-generation, propagation, candidate-validation, arity 0-3 tuple-source model-membership staging with fixed arity-one/two/three row-scoped ground key comparison over existing relation buffers, generic arity-N variable-bound tuple matching, world-view-validation, accepted-candidate materialization, final-result flag, final-row map/final tuple materialization kernels, accepted K5 WCOJ dispatch, and hot-path transfer-budget trace with CUDA-event elapsed timing/runtime-preflight/fail-closed WCOJ gate/reduced-plan trace contracts exist, but full semantic kernel-buffer parity, broader solver learned-clause/status lifecycle wiring, probability wiring, and broader fixture coverage remain missing. |
| G090_SOLVER | BLOCKED | Accepted GPU runtime evidence can gate GPU CDCL SAT/UNSAT, reusable workspace-backed UNSAT, one bounded push/solve/retract lifecycle, bounded MaxSAT candidate solving, and bounded SAT/MaxSAT portfolio dispatch, but broader learned-clause lifecycle traces and status-aware UNKNOWN/TIMEOUT portfolio handling are not wired to epistemic candidates. |
| G090_PROB | BLOCKED | Accepted GPU runtime evidence can gate source/program exact compilation, bounded compile/evaluate, PIR/CNF encoding, and query/gradient evaluation through the existing GPU-native path, but broader probabilistic knowledge-compilation execution on accepted world views is incomplete. |
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
- bounded model-membership staging kernel with one launch, one reduced-output
  row-count device read, and zero host writes for candidate-scoped
  model-membership bytes;
- bounded world-view validation staging kernel with one launch and zero host
  writes for staged model-membership/world-view rejection checks;
- bounded materialization staging kernel with one launch and zero host writes
  for accepted-candidate world-view slots;
- bounded final-result flag staging kernel with one launch, one device
  row-count read from reduced output metadata, and zero host writes for
  world-view result slots;
- bounded final tuple materialization kernel with device row-count read/write
  metadata, zero host writes, and a device-resident final-output `CudaBuffer`;
- bounded final-row map construction that filters accepted unary, binary, and
  multi-membership nonzero-arity output rows by bound tuple-key membership on
  device;
- runtime preflight that rejects nonzero CPU fallback counters and records
  WCOJ/K-clique/helper route metadata before launch;
- runtime counter guard that refuses to certify WCOJ evidence from preflight
  metadata unless production WCOJ counters advance, while accepted K5 evidence
  also requires sorted-layout and helper-split preflight metadata before
  model-membership/world-view staging;
- hot-path transfer-budget trace that rejects tracked data-plane H2D/D2H
  deltas without resetting shared provider telemetry;
- reduced-plan execution trace API that wraps `execute_plan` with before/after
  production counter snapshots;
- accepted solver production adapters that gate GPU CDCL SAT/UNSAT, reusable
  workspace-backed UNSAT, bounded push/solve/retract lifecycle, bounded MaxSAT
  candidate solving, and bounded SAT/MaxSAT portfolio dispatch on accepted GPU
  runtime evidence;
- G91 and FAEEL fixture evaluators;
- Generate-Propagate-Test phase traces;
- world-view operator fixtures for `know`, `possible`, and `not know`;
- bounded solver-service lifecycle fixtures;
- accepted-world-view probabilistic evidence fixtures and production adapter
  gates for accepted source/program exact compilation, bounded
  compile/evaluate, PIR/CNF encoding, query evaluation, and gradient
  evaluation.

This evidence should be retained as oracle coverage for the required GPU-native
implementation, but it cannot be used as release-close evidence.

## Missing GPU-Native Evidence

Closure remains blocked until certification includes all of the following:

- broader nonzero GPU launch counts and kernel timings for actual stable-model
  tuple membership population beyond the current unary/binary/multi-membership
  accepted fixtures;
- final-result transfer accounting for complete accepted execution;
- GPU-resident candidate, world-view, model-membership, and rejection buffers;
- zero CPU fallback counters for candidate enumeration and world-view
  validation;
- broader WCOJ-eligible epistemic reductions proving successful planner/runtime
  dispatch beyond the current accepted K5 fixture, including layout and
  helper-splitting evidence where applicable;
- broader accepted SAT/UNSAT learned-clause lifecycle traces with distinct SAT,
  UNSAT, UNKNOWN, and TIMEOUT handling across lifecycle and portfolio paths;
- accepted-world-view evidence flowing through broader GPU-native
  knowledge-compilation evaluation with zero CPU-only probability recomputation;
- post-v0.8 rebase compatibility evidence.

## Release Hygiene

No push, tag, release-board update, merge to `main`, or v0.8-owned pyxlog API
change was performed.

## Decision

Release decision: `HOLD_FOR_GPU_NATIVE_AND_V080_REBASE`.

The current branch is still incomplete. The next closing work must complete the
corrected `G090_GPU` production runtime/WCOJ/GPU path and then rerun the full
certification set after v0.8.0 integration.
