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
| G090_G91 | PASS for semantic oracle plus one accepted runtime fixture | Compatibility fixtures pass and explicit self-supported `possible` reaches accepted GPU runtime execution with mode-aware oracle trace/candidate-index parity, but full GPU parity remains unproven. |
| G090_FAEEL | PASS for semantic oracle plus executable guard | Foundedness fixtures pass, default FAEEL executable-plan lowering rejects an unsupported self-supported `possible` rule before runtime dispatch, independently founded self-`possible` reaches accepted GPU runtime execution, and explicit G91 compatibility remains allowed through accepted runtime execution. Full GPU parity remains unproven. |
| G090_GPT | PARTIAL | CPU trace fixtures pass; GPU-resident candidate generation, propagation staging, candidate-buffer validation, arity 0-3 tuple-source model-membership staging with fixed arity-one/two/three row-scoped ground key comparison, generic arity-N variable-bound tuple matching, explicit operator metrics, negated binding polarity, candidate-assumption-aware bounded world-view validation staging, accepted-candidate materialization staging, final-result flag staging, final-row map construction with row-filter polarity counts, and final tuple materialization exist; unary/possible/not-possible/binary/multi-membership, missing-required multi-membership rejection, negated final-row filtering, split possible-vs-not-known world-view parity, and one bounded GPU-vs-GPT oracle trace parity fixture with accepted candidate index parity pass, but broader semantic parity remains missing. |
| G090_SPLIT | PARTIAL | CPU split/recompose fixtures pass, valid split components lower through GPU executable subplans, and accepted split components execute through `execute_epistemic_gpu_execution_batch` while matching simple component output oracles and the absent `possible` vs true `not know` world-view oracle with zero CPU candidate/world-view fallback counters; full accepted-runtime semantic parity is still missing. |
| G090_GPU | BLOCKED | GPU-plan, reduced-runtime-plan, workspace allocation/reset, bounded candidate-generation, propagation, candidate-validation, arity 0-3 tuple-source model-membership staging with fixed arity-one/two/three row-scoped ground key comparison over existing relation buffers, generic arity-N variable-bound tuple matching, explicit `know`/`possible`/`not know`/`not possible` preflight metrics, negated binding polarity, all-required-membership world-view-validation over GPU candidate-assumption and model-membership buffers, accepted-candidate materialization, final-result flag, final-row map/final tuple materialization kernels with `row_filter_count` and `negated_row_filter_count`, device-derived semantic trace counts with accepted/rejected candidate indices and typed rejection reasons, bounded FAEEL and G91 GPU-vs-GPT oracle trace parity fixtures, split possible-vs-not-known world-view parity, accepted K5/K6/K7/K8 WCOJ dispatch, K5 dispatch-certified edge-permutation/stream-group/skew-scheduled-helper/sorted-layout/helper-split/helper-rule/WCOJ helper input trace metrics, helper metadata-only preflight rejection, WCOJ dispatch certification that fails closed without required layout sort or layout fast-path evidence, K6 G38-B skew-scheduled helper/histogram metadata-build trace metrics, K7/K8 K-clique planner preflight reuse including stream-group metadata, hot-path transfer-budget trace, final-result transfer accounting, CUDA-event elapsed timing/runtime-preflight/fail-closed WCOJ gate/reduced-plan trace contracts, two-record bounded weighted MaxSAT selection encoding/search, and heterogeneous MaxSAT scheduler reuse through existing GPU CNF/CDCL paths exist, but full semantic kernel-buffer parity, probability wiring, and broader fixture coverage remain missing. |
| G090_SOLVER | BLOCKED | Accepted GPU runtime evidence can gate GPU CDCL SAT/UNSAT, reusable workspace-backed UNSAT, one-record and two-record bounded push/solve/retract lifecycles, lifecycle UNKNOWN/TIMEOUT propagation, learned-clause arena publication, same-device-CNF learned-clause import/reuse, two-record learned-clause reuse, distinct-CNF learned-clause import rejection, bounded single- and multi-candidate MaxSAT solving, single-result and two-record MaxSAT search pruning, single-result and two-record weighted soft-clause selection encoding through existing GPU CNF/CDCL paths, heterogeneous MaxSAT scheduling, and single-result plus two-record bounded SAT/MaxSAT portfolio dispatch with UNKNOWN/TIMEOUT status propagation, but broader solver semantic integration and post-v0.8 certification remain incomplete. |
| G090_PROB | BLOCKED | Accepted GPU runtime evidence can gate source/program exact compilation, source/program bounded compile/evaluate, two-record accepted source/program batch compile/evaluate, source/program zero-arity and concrete nonzero-arity true/false evidence conditioning with negative-evidence trace counters, two-record positive and negative conditioned source query batches, two-record conditioned program query batches, conditioned source/program gradient evaluation, single-record plus two-record PIR/CNF encoding, and single-record plus two-record query/gradient evaluation through the existing GPU-native path, but broader probabilistic coverage on accepted world views is incomplete. |
| G090_CERT | BLOCKED | Missing complete accepted-execution kernel timing, broader WCOJ runtime evidence, zero CPU fallback counters, and post-v0.8 rerun. |
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
  helper-splitting metadata when statistics are supplied, rejects default
  FAEEL unsupported self-supported `possible` rules before reduced runtime
  dispatch, and permits independently founded self-`possible` fixtures;
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
  writes for staged candidate-assumption, model-membership, and world-view
  rejection checks;
- bounded materialization staging kernel with one launch and zero host writes
  for accepted-candidate world-view slots;
- bounded final-result flag staging kernel with one launch, one device
  row-count read from reduced output metadata, and zero host writes for
  world-view result slots;
- bounded final tuple materialization kernel with device row-count read/write
  metadata, zero host writes, and a device-resident final-output `CudaBuffer`;
- bounded final-row map construction that filters accepted unary, possible,
  not-possible, binary, multi-membership, and `not know` nonzero-arity output
  rows by bound tuple-key membership on device, with explicit operator counts
  in preflight and final-row polarity counts in the materialization trace;
- negative multi-membership evidence that rejects every candidate before final
  tuple materialization when one required epistemic membership has no
  tuple-source support;
- device-derived semantic trace accounting that reads bounded rejection-reason
  metadata after the hot-path budget and records generated, propagated, tested,
  accepted, rejected, accepted/rejected candidate indices, and typed
  rejection-reason counts with zero CPU candidate/world-view fallback counters,
  including bounded FAEEL and G91 GPU-vs-GPT oracle trace parity fixtures;
- runtime preflight that rejects nonzero CPU fallback counters and records
  WCOJ/K-clique/helper route metadata before launch, including max K-clique
  arity, live edge-permutation counts, distinct stream-group scheduling
  counts, skew-scheduled helper-plan counts, helper-split specs, and
  production helper relation rule/scan counts;
- runtime counter guard that refuses to certify WCOJ evidence from preflight
  metadata unless production WCOJ counters advance and required layout evidence
  records a layout sort or layout fast-path event, while helper-split metadata
  fails closed unless compiler-created helper relation rules and WCOJ input
  scans are present, and helper scans outside WCOJ do not satisfy that gate;
  accepted K5/K6/K7/K8 evidence observes production dispatch counters
  before model-membership and world-view staging; K5 certifies
  edge-permutation, stream-group, skew-scheduled helper, sorted-layout,
  helper-split, helper-rule, and WCOJ helper input counts inside the
  dispatch-certified trace, and K6 certifies G38-B skew-scheduled helper-split
  plus runtime histogram metadata-build counters;
- K7/K8 preflight evidence, plus K7/K8 runtime evidence, that generated
  epistemic reductions reuse the G39 K-clique template planner metadata, with
  complete 21/28 edge-permutation counts, stream-group counts, and zero
  planned-hash/CPU-fallback counters;
- hot-path transfer-budget trace that rejects tracked data-plane H2D/D2H
  deltas without resetting shared provider telemetry;
- post-hot-path final-result transfer accounting that records final output
  rows, columns, payload bytes, row-count metadata reads, and zero accepted-path
  data-plane D2H calls;
- reduced-plan execution trace API that wraps `execute_plan` with before/after
  production counter snapshots;
- accepted solver production adapters that gate GPU CDCL SAT/UNSAT, reusable
  workspace-backed UNSAT, bounded push/solve/retract lifecycle,
  learned-clause arena publication, same-device-CNF learned-clause
  import/reuse, distinct-CNF learned-clause import rejection, bounded single-
  and multi-candidate MaxSAT solving, single-result and two-record MaxSAT
  search pruning, bounded
  single-result plus two-record SAT/MaxSAT portfolio dispatch, and
  UNKNOWN/TIMEOUT portfolio status propagation on accepted GPU runtime
  evidence;
- G91 and FAEEL fixture evaluators plus explicit G91 self-supported `possible`
  accepted runtime execution, a default FAEEL executable-plan foundedness guard,
  and an independently founded self-`possible` accepted GPU runtime fixture;
- Generate-Propagate-Test phase traces;
- world-view operator fixtures for `know`, `possible`, `not know`, and
  `not possible`;
- bounded solver-service lifecycle fixtures;
- accepted-world-view probabilistic evidence fixtures and production adapter
  gates for accepted source/program exact compilation, source/program bounded
  compile/evaluate, source/program zero-arity and concrete nonzero-arity
  true/false evidence conditioning with negative-evidence trace counters,
  two-record positive and negative conditioned source query batches,
  two-record conditioned program query batches, single-record plus two-record
  PIR/CNF encoding, query evaluation, and gradient evaluation.
- bounded executable split components that reuse the existing epistemic GPU
  executable-plan path and a batch adapter over the existing single-plan GPU
  runtime execution path rather than a split-only WCOJ or tuple-store engine.

This evidence should be retained as oracle coverage for the required GPU-native
implementation, but it cannot be used as release-close evidence.

## Missing GPU-Native Evidence

Closure remains blocked until certification includes all of the following:

- broader nonzero GPU launch counts and kernel timings for actual stable-model
  tuple membership population beyond the current unary/possible/not-possible/
  binary/multi-membership/missing-required and `not know` accepted fixtures;
- GPU-resident candidate, world-view, model-membership, and rejection buffers;
- zero CPU fallback counters for candidate enumeration and world-view
  validation;
- broader WCOJ-eligible epistemic reductions proving successful runtime
  dispatch beyond the current accepted K5/K6/K7/K8 fixtures, including layout,
  skew-scheduling, and helper-splitting evidence where applicable;
- broader accepted solver semantic integration beyond the current bounded
  scheduler and portfolio fixtures;
- broader accepted-world-view probabilistic coverage beyond the bounded
  conditioned query/gradient and PIR/CNF GPU-native knowledge-compilation
  fixtures, with zero CPU-only probability recomputation;
- post-v0.8 rebase compatibility evidence.

## Release Hygiene

No push, tag, release-board update, merge to `main`, or v0.8-owned pyxlog API
change was performed.

## Decision

Release decision: `HOLD_FOR_GPU_NATIVE_AND_V080_REBASE`.

The current branch is still incomplete. The next closing work must complete the
corrected `G090_GPU` production runtime/WCOJ/GPU path and then rerun the full
certification set after v0.8.0 integration.
