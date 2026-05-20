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

The current branch still cannot produce a closure proposal. The
v0.7.0/v0.8.0/v0.8.5/v0.8.6 bundle has now been merged into this feature branch and
the v0.8.6 compatibility validator has passed, but closure remains blocked on
the incomplete v0.9 GPU-native certification scope:

- `G090_GPU`: production GPU-native epistemic execution, WCOJ-backed reductions
  where eligible, GPU-resident world-view/candidate/rejection buffers, and zero
  CPU fallback counters.
- `G090_CLOSE`: final closure still requires the remaining v0.9 certification
  reruns and an approved closure proposal after the GPU-native blockers clear.

2026-05-20 update: the same-rule all-operator accepted GPU fixture now gates
solver lifecycle, learned-clause reuse, MaxSAT, portfolio, probabilistic
source conditioning, parsed-program conditioning, gradients, and parsed-program
PIR/CNF production adapters with zero CPU search/recomputation counters. This
narrows the production-reuse gap, but it is not a closure proposal. Default
FAEEL lowering also now rejects nonzero-arity self-`possible` rules unless
tuple-level foundedness can be proven.

2026-05-20 follow-up: single-result quaternary `possible fact4/4` and
`not know fact4/4` accepted GPU results now reach the existing solver SAT gate
and probabilistic conditioned source gate with arity-four counters and zero CPU
search/recomputation. The split-batch quaternary `possible fact4/4` plus
`not know fact4/4` fixture now reaches the existing GPU CDCL lifecycle adapter
and probabilistic conditioned source batch gate with accepted batch/component
counters, arity-four tuple/evidence counters, exact-query counters, balanced
lifecycle pushes/retractions, and zero CPU search/recomputation. The same
split-batch evidence now also reaches existing solver learned-clause reuse and
bounded MaxSAT candidate paths with two arena publications/imports/reused
solves, four GPU CDCL candidate solves, two MaxSAT optima, and zero CPU search
or learned-clause transfers. The same possible/not-know batch now also reaches
probabilistic source/program gradients, source/program PIR/CNF, and
already-compiled exact query/gradient evaluation with arity-four source/program
evidence counters and zero CPU probability recomputation. `G090_SOLVER`,
`G090_PROB`, and `G090_CLOSE` remain blocked.

## Ref Evidence

| Ref | SHA |
|---|---|
| `main` | `bd45229d` / tag `v0.8.6` |
| `origin/main` | `bd45229d` / tag `v0.8.6` |
| `feat/v080-dts-ml-python-productization` | ancestor of `main` |
| `feat/v090-epistemic-solver-semantics` | this file's containing commit after the GPU-native correction |

Current ref checks on 2026-05-19 showed:

| Check | Result | Interpretation |
|---|---|---|
| `git merge-base --is-ancestor v0.7.0 main` | exit `0` | v0.7.0 is merged into `main`. |
| `git merge-base --is-ancestor v0.8.0 main` | exit `0` | v0.8.0 is merged into `main`. |
| `git merge-base --is-ancestor v0.8.5 main` | exit `0` | v0.8.5 is merged into `main`. |
| `git merge-base --is-ancestor v0.8.6 main` | exit `0` | v0.8.6 is merged into `main`. |
| `git merge-base --is-ancestor main HEAD` | exit `0` after the integration merge | v0.9 branch now contains the v0.7.0/v0.8.0/v0.8.5/v0.8.6 bundle. |
| `git merge-base --is-ancestor origin/main HEAD` | exit `0` after the integration merge | v0.9 branch now contains `origin/main` at `bd45229d`. |

## v0.7.0/v0.8.0/v0.8.5/v0.8.6 Bundle Integration

A non-destructive preflight on 2026-05-19 used a throwaway detached worktree at
`HEAD` and ran:

```bash
git merge --no-commit --no-ff main
```

The preflight did not move this branch. It exited `1` with 11 content conflicts:

| Conflict file | Conflict-marker lines |
|---|---:|
| `crates/xlog-integration/src/bin/xlog_run.rs` | 3 |
| `crates/xlog-logic/src/ast.rs` | 12 |
| `crates/xlog-logic/src/grammar.pest` | 6 |
| `crates/xlog-logic/src/lib.rs` | 3 |
| `crates/xlog-logic/src/lower.rs` | 9 |
| `crates/xlog-logic/src/parser.rs` | 14 |
| `crates/xlog-logic/src/stratify.rs` | 3 |
| `crates/xlog-prob/src/mc/buffers.rs` | 6 |
| `crates/xlog-prob/src/mc/mod.rs` | 3 |
| `crates/xlog-prob/src/provenance.rs` | 3 |
| `crates/xlog-runtime/src/lib.rs` | 3 |

The conflicted surface is the expected reuse seam between the v0.7.0/v0.8.0/v0.8.5/v0.8.6
bundle and v0.9.0: CLI execution, logic AST/parser/grammar/lowering/stratifier,
probabilistic Monte Carlo/provenance APIs, and runtime exports. Any real
integration needed to resolve these conflicts by preserving the v0.7.0 WCOJ
architecture/runtime expansion plus the v0.8.6 language, runtime, pyxlog, and
packaging changes while reapplying the v0.9.0 epistemic EIR/GPU/probability/solver
hooks.

The real feature-branch integration then merged `main` carrying the
v0.7.0/v0.8.0/v0.8.5/v0.8.6 bundle into
`feat/v090-epistemic-solver-semantics` and resolved the same 11 files by:

- preserving v0.7.0 WCOJ `MultiWayJoin`/4-cycle/general-arity runtime surfaces
  plus v0.8.5/v0.8.6 `univ`, list/meta, magic-set, approximate
  probabilistic, runtime-delta, pyxlog, example, and validation surfaces;
- preserving v0.9.0 epistemic AST/parser/grammar, typed RIR boundary,
  stratification, runtime trace/export, probabilistic provenance, and MC
  schema hooks;
- extending EIR conversion and GPU tuple-key matching diagnostics for the new
  v0.8.5 term forms;
- removing the stale upfront MC predicate-declaration inference loop that
  regressed `06_prob_aggregate_mc` with
  `Compilation("Inconsistent predicate types for edge")`.

Post-merge compatibility validation:

| Command | Result |
|---|---|
| `cargo check -p xlog-logic -p xlog-prob -p xlog-runtime -p xlog-integration` | PASS |
| `cargo test -p xlog-logic --tests` | PASS |
| `cargo test -p xlog-prob --tests` | PASS |
| `cargo test -p xlog-runtime --test test_epistemic_gpu_workspace -- --nocapture` | PASS, 54 passed |
| `cargo test -p xlog-solve --test gpu_solver_production_reuse` | PASS, 3 passed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_epistemic_v070_4cycle_execution_certifies_production_wcoj_dispatch -- --exact --nocapture` | PASS, 1 passed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_all_binary_operator_batch_gates_probabilistic_pir_cnf_and_exact_evaluation_paths -- --exact --nocapture` | PASS, 1 passed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution split_multi_membership_modal_coupling_rejects_gpu_batching -- --exact --nocapture` | PASS, 1 passed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_mixed_memberships_match_gpt_oracle_parity -- --exact --nocapture` | PASS, 1 passed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_negated_mixed_memberships_match_gpt_oracle_parity -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_all_operator_mixed_memberships_match_gpt_oracle_parity -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_all_operator_mixed_membership_gates_solver_lifecycle_path -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_all_operator_mixed_membership_conditions_probabilistic_evidence -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_all_operator_mixed_membership_gates_solver_reuse_maxsat_and_portfolio_paths -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_all_operator_mixed_membership_gates_probabilistic_program_gradient_and_pir_paths -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution rejects_unrecorded_candidate_generation_timing -- --nocapture` | PASS, 2 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution aggregate_timing_requires_every_component_phase_to_be_recorded -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_quaternary_possible_and_not_know_memberships_match_gpt_oracle_parity -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_quaternary_possible_and_not_know_results_gate_solver_and_probabilistic_paths -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_quaternary_possible_and_not_know_batch_gates_solver_and_probabilistic_paths -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_quaternary_possible_and_not_know_batch_gates_solver_reuse_and_maxsat_paths -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_quaternary_possible_and_not_know_batch_gates_probabilistic_gradient_pir_cnf_and_exact_evaluation_paths -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution -- --nocapture` | PASS, 122 passed, 0 failed |
| `python scripts/validate_v086_examples.py --output /tmp/v090-v086-compat-validation.json` | PASS, consumer certification PASS, examples PASS |

## Corrected Gate Table

| Goal | Current Status | Evidence |
|---|---|---|
| G090_PRE | PASS for inventory | Preflight evidence committed. |
| G090_EIR | PARTIAL | EIR is explicit and executable-plan lowering reaches reduced production runtime plans, but accepted epistemic forms still lack production GPU runtime dispatch. |
| G090_G91 | PASS for semantic oracle plus one accepted runtime fixture | Compatibility fixtures pass and explicit self-supported `possible` reaches accepted GPU runtime execution with mode-aware oracle trace/candidate-index parity, but full GPU parity remains unproven. |
| G090_FAEEL | PASS for semantic oracle plus executable guard | Foundedness fixtures pass, default FAEEL executable-plan lowering rejects unsupported self-supported `possible` before runtime dispatch, rejects nonzero-arity self-`possible` without tuple-level foundedness proof, independently founded zero-arity self-`possible` reaches accepted GPU runtime execution with default-oracle trace/candidate-index parity, and explicit G91 compatibility remains allowed through accepted runtime execution. Full GPU parity remains unproven. |
| G090_GPT | PARTIAL | CPU trace fixtures pass; GPU-resident candidate generation, propagation staging, candidate-buffer validation, arity 0-3 tuple-source model-membership staging with fixed arity-one/two/three row-scoped ground key comparison, generic arity-N variable-bound tuple matching, explicit operator metrics, negated binding polarity, candidate-assumption-aware bounded world-view validation staging, accepted-candidate materialization staging, final-result flag staging, final-row map construction with row-filter polarity counts, and final tuple materialization exist; unary/possible/not-possible/binary/ternary-specialized/quaternary-all-operator/multi-membership, mixed `know`/`possible` same-rule membership, negated `not know`/`not possible` same-rule membership, same-rule all-operator mixed membership, missing-required multi-membership rejection, negated final-row filtering, split possible-vs-not-known world-view parity, a bounded single-literal GPU-vs-GPT oracle trace parity fixture, unary operator-level `possible`/`not possible`/`not know` GPU-vs-GPT trace/candidate-index parity fixtures, binary operator-level `know`/`possible`/`not possible`/`not know` GPU-vs-GPT trace/candidate-index parity fixtures, a ternary `know fact3(A, B, C)` specialized arity-three GPU-vs-GPT trace/candidate-index parity fixture, quaternary `know fact4/4`, `possible fact4/4`, `not know fact4/4`, and `not possible fact4/4` generic arity-N GPU-vs-GPT trace/candidate-index parity fixtures, all-`know`, mixed, negated mixed, and all-operator multi-membership GPU-vs-GPT trace/candidate-index parity fixtures, a four-component split binary `know`/`possible`/`not possible`/`not know` GPU-vs-GPT trace/candidate-index parity fixture, and a two-component split quaternary `know fact4/4` plus `not possible fact4/4` GPU-vs-GPT trace/candidate-index parity fixture pass, but broader semantic parity remains missing. |
| G090_SPLIT | PARTIAL | CPU split/recompose fixtures pass, valid split components lower through GPU executable subplans, unsafe multi-membership modal coupling rejects before GPU split batching, and accepted split components execute through `execute_epistemic_gpu_execution_batch_with_trace` while matching simple component output oracles, per-component GPT trace/candidate-index oracles, four-component all-binary-operator output and polarity oracles, two-component arity-four `know fact4/4` plus `not possible fact4/4` output and polarity oracles, aggregate zero CPU recomposition/per-candidate-host-round-trip counters, aggregate split-batch CUDA-event timing, and the absent `possible` vs true `not know` world-view oracle with zero CPU candidate/world-view fallback counters; full accepted-runtime semantic parity is still missing. |
| G090_GPU | BLOCKED | GPU-plan, reduced-runtime-plan, workspace allocation/reset, bounded candidate-generation, propagation, candidate-validation, arity 0-3 tuple-source model-membership staging with fixed arity-one/two/three row-scoped ground key comparison over existing relation buffers, generic arity-N variable-bound tuple matching, explicit `know`/`possible`/`not know`/`not possible` preflight metrics, negated binding polarity, all-required-membership world-view-validation over GPU candidate-assumption and model-membership buffers, accepted-candidate materialization, final-result flag, final-row map/final tuple materialization kernels with `row_filter_count` and `negated_row_filter_count`, device-derived semantic trace counts with accepted/rejected candidate indices and typed rejection reasons, bounded FAEEL, G91, unary operator, binary all-operator, ternary specialized-arity, quaternary all-operator generic-arity, all-`know`, mixed `know`/`possible`, negated `not know`/`not possible`, and all-operator multi-membership, split-component, all-binary-operator split, and split-quaternary-operator GPU-vs-GPT oracle trace parity fixtures, split batch zero CPU recomposition/per-candidate-host-round-trip counters plus aggregate CUDA-event timing that fails closed on partial component timing, split possible-vs-not-known world-view parity, accepted v0.7.0 4-cycle and K5/K6/K7/K8 WCOJ dispatch, K5 dispatch-certified edge-permutation/stream-group/skew-scheduled-helper/sorted-layout/helper-split/helper-rule/WCOJ helper input trace metrics, helper metadata-only preflight rejection, WCOJ dispatch certification that fails closed without required non-hash `MultiWayJoin` dispatch, layout sort, or layout fast-path evidence, K6 G38-B skew-scheduled helper/histogram metadata-build trace metrics, K7/K8 K-clique planner preflight reuse including stream-group metadata, hot-path transfer-budget trace, final-result transfer accounting, CUDA-event elapsed timing/runtime-preflight/fail-closed WCOJ gate/reduced-plan trace contracts, two-record and accepted split-batch bounded weighted MaxSAT selection encoding/search, and heterogeneous plus accepted split-batch MaxSAT scheduler reuse through existing GPU CNF/CDCL paths exist, but full semantic kernel-buffer parity, probability wiring, and broader fixture coverage remain missing. |
| G090_SOLVER | BLOCKED | Accepted GPU runtime evidence can gate GPU CDCL SAT/UNSAT, reusable workspace-backed UNSAT, one-record, two-record, accepted split-batch, and mixed five-record bounded push/solve/retract lifecycles, single-result, two-record, and accepted split-batch combined lifecycle-plus-MaxSAT, fail-closed empty MaxSAT lifecycle rejection before lifecycle trace mutation, fail-closed all-UNSAT MaxSAT search rejection before solver trace mutation, fail-closed all-UNSAT encoded MaxSAT rejection before accepted-evidence or encode trace mutation, fail-closed invalid encoded MaxSAT scheduler rejection before accepted-batch evidence, scheduler, encode, or solver trace mutation, fail-closed split-batch solver gating when aggregate CUDA-event timing is absent or any component phase is untimed, fail-closed single-result solver gating when candidate-generation CUDA-event timing is absent, accepted G91/default FAEEL mode-specific solver trace counters, accepted operator-family solver trace counters, accepted nonzero-arity tuple-key evidence counters including single-result quaternary `possible`/`not know fact4/4`, accepted split-batch/component counters, all-binary-operator split-batch lifecycle plus all-binary split-batch learned-clause reuse and MaxSAT with accepted `know`/`possible`/`not possible`/`not know` solver evidence counters, split-batch quaternary `know fact4/4` plus `not possible fact4/4` lifecycle, learned-clause reuse, and MaxSAT evidence with nonzero-arity tuple-key counters, mixed unary and binary `possible`/`not possible` plus binary `not know` operator-result lifecycles, lifecycle UNKNOWN/TIMEOUT propagation, learned-clause arena publication, same-device-CNF learned-clause import/reuse, two-record and accepted split-batch learned-clause reuse, distinct-CNF learned-clause import rejection, bounded single-, multi-candidate, and accepted split-batch MaxSAT solving, single-result, two-record, and accepted split-batch MaxSAT search pruning, single-result, two-record, and accepted split-batch weighted soft-clause selection encoding through existing GPU CNF/CDCL paths, heterogeneous and accepted split-batch MaxSAT scheduling, and single-result, two-record, plus split-batch bounded SAT/MaxSAT portfolio dispatch with UNKNOWN/TIMEOUT status propagation, but broader solver semantic integration and post-v0.7.0/v0.8.0/v0.8.5/v0.8.6 certification remain incomplete. |
| G090_PROB | BLOCKED | Accepted GPU runtime evidence can gate source/program exact compilation, two-record and accepted split-batch direct source/program exact compilation, source/program bounded compile/evaluate with source/program-specific exact-query counters, two-record accepted source/program batch compile/evaluate, accepted split-batch source/program compile/evaluate plus conditioned source/program query and gradient evaluation with accepted batch/component counters and aggregate CUDA-event timing validation that fails closed on partial component timing, fail-closed single-result probabilistic gating when candidate-generation CUDA-event timing is absent, all-binary-operator split-batch conditioned source and parsed-program query plus source and parsed-program gradient evidence with true/false `know`/`possible` operator assumptions, split-batch quaternary parsed-program query plus source/program gradient, PIR-CNF, and already-compiled exact query/gradient evidence with one `know fact4/4` component and one `not possible fact4/4` component, split-batch quaternary possible/not-know source/program gradient, PIR-CNF, and already-compiled exact query/gradient evidence, all-binary split-batch source/program PIR-CNF plus already-compiled exact query/gradient evaluation, source/program zero-arity and concrete nonzero-arity true/false evidence conditioning with negative-evidence, source/program-specific, aggregate/source/program nonzero-arity evidence, aggregate/source/program max-arity, aggregate operator-specific, and source/program-specific operator-conditioned trace counters including true `know`, true `possible`, false `possible`/`not possible`, false `know`/`not know`, and two-record quaternary `possible`/`not know fact4/4` source evidence, mode-specific accepted G91/FAEEL production trace counters, two-record positive and negative conditioned source query batches, two-record conditioned program query batches, conditioned source/program gradient evaluation with source/program-specific gradient counters, ternary source, quaternary source, negated quaternary source, quaternary parsed-program, negated quaternary parsed-program, and split-batch quaternary parsed-program/source-gradient/program-gradient accepted probabilistic evidence arity trace coverage, single-record, two-record, and accepted split-batch PIR/CNF encoding with source/program-specific PIR/CNF counters, and single-record, two-record, and accepted split-batch query/gradient evaluation through the existing GPU-native path, but broader probabilistic coverage on accepted world views is incomplete. |
| G090_CERT | BLOCKED | v0.7.0/v0.8.0/v0.8.5/v0.8.6 compatibility reruns passed after the integration merge, and split-batch aggregate CUDA-event timing now exists and fails closed on partial component timing, but the complete accepted-execution kernel-timing matrix, broader semantic/probabilistic coverage, and final GPU-native certification remain missing. |
| G090_DOC | PARTIAL | Guide documents the semantic oracle, partial accepted GPU/WCOJ runtime path, solver/probability production adapters, and remaining blockers; full release documentation is still blocked by broader semantic parity and post-v0.7.0/v0.8.0/v0.8.5/v0.8.6 certification. |
| G090_CLOSE | BLOCKED | The v0.7.0/v0.8.0/v0.8.5/v0.8.6 bundle has been merged into the feature branch and compatibility validation passed, but closure still requires G090_GPU/G090_SOLVER/G090_PROB/G090_CERT completion plus an approved closure proposal. |

2026-05-20 split-batch production-reuse addendum: `G090_SPLIT` now also has a
two-component split quaternary `possible fact4/4` plus `not know fact4/4`
accepted runtime fixture with zero CPU recomposition and batch/component
operator counters; `G090_SOLVER` now has the same split-batch evidence routed
through the existing GPU CDCL lifecycle adapter with nonzero-arity tuple-key
counters plus learned-clause reuse and bounded MaxSAT candidate solving; and
`G090_PROB` now has the same split-batch evidence routed through the existing
conditioned source exact-query path with source-conditioned arity-four evidence
counters, plus source/program gradients, source/program PIR/CNF, and exact
query/gradient evaluation with zero CPU probability recomputation. These are
bounded production-reuse additions only; the BLOCKED/PARTIAL statuses above are
unchanged.

## Current Semantic-Oracle Evidence

The branch contains useful scaffolding:

- explicit EIR and typed lowering boundary;
- GPU execution plan contract with required phases, buffer categories, WCOJ
  planner obligations, and zero fallback counters;
- executable lowering contract whose reduced ordinary program uses the normal
  compiler pipeline and can promote WCOJ-eligible reductions to the v0.7.0
  `RirNode::MultiWayJoin` and deterministic 4-cycle dispatch route, including
  Goal-038-B K-clique planner, layout, and helper-splitting metadata when statistics are supplied, rejects default
  FAEEL unsupported self-supported `possible` rules before reduced runtime
  dispatch, rejects nonzero-arity self-`possible` without tuple-level
  foundedness proof, and permits independently founded zero-arity
  self-`possible` fixtures;
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
  not-possible, binary `know`, binary `possible`, binary `not possible`,
  binary `not know`,
  ternary specialized-arity `know`, quaternary all-operator generic-arity,
  multi-membership, and `not know` nonzero-arity output
  rows by bound tuple-key membership on device, with explicit operator counts
  in preflight and final-row polarity counts in the materialization trace;
- negative multi-membership evidence that rejects every candidate before final
  tuple materialization when one required epistemic membership has no
  tuple-source support;
- device-derived semantic trace accounting that reads bounded rejection-reason
  metadata after the hot-path budget and records generated, propagated, tested,
  accepted, rejected, accepted/rejected candidate indices, and typed
  rejection-reason counts with zero CPU candidate/world-view fallback counters,
  including bounded FAEEL, G91, unary operator, binary all-operator,
  ternary specialized-arity, quaternary all-operator generic-arity, multi-membership,
  split-component, all-binary-operator split, and split-quaternary-operator
  GPU-vs-GPT oracle trace parity fixtures;
- runtime preflight that rejects nonzero CPU fallback counters and records
  WCOJ/K-clique/helper route metadata before launch, including max K-clique
  arity, live edge-permutation counts, distinct stream-group scheduling
  counts, skew-scheduled helper-plan counts, helper-split specs, and
  production helper relation rule/scan counts;
- runtime counter guard that refuses to certify WCOJ evidence from preflight
  metadata unless production WCOJ counters advance for required non-hash
  `MultiWayJoin` reductions and required layout evidence records a layout sort
  or layout fast-path event, while helper-split metadata fails closed unless
  compiler-created helper relation rules and WCOJ input scans are present, and
  helper scans outside WCOJ do not satisfy that gate; accepted v0.7.0 4-cycle
  plus K5/K6/K7/K8 evidence observes production dispatch counters
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
  workspace-backed UNSAT, bounded unary/binary operator push/solve/retract lifecycle,
  ternary nonzero-arity tuple-key evidence tracing at the SAT boundary,
  single-result, two-record, and split-batch combined lifecycle-plus-MaxSAT,
  fail-closed empty MaxSAT lifecycle rejection before lifecycle trace mutation,
  fail-closed all-UNSAT MaxSAT search rejection before solver trace mutation,
  fail-closed all-UNSAT encoded MaxSAT rejection before accepted-evidence or encode trace mutation,
  fail-closed invalid encoded MaxSAT scheduler rejection before accepted-batch
  evidence, scheduler, encode, or solver trace mutation,
  accepted split-batch solver lifecycle, all-binary-operator split-batch
  solver lifecycle plus learned-clause reuse and MaxSAT with accepted
  `know`/`possible`/`not possible`/`not know` component evidence counters,
  split-batch quaternary `know fact4/4` plus `not possible fact4/4`
  solver lifecycle, learned-clause reuse, and MaxSAT evidence with
  nonzero-arity tuple-key counters,
  split-batch quaternary `possible fact4/4` plus `not know fact4/4`
  solver lifecycle, learned-clause reuse, and MaxSAT evidence with
  nonzero-arity tuple-key counters,
  learned-clause reuse, MaxSAT,
  weighted MaxSAT encoding/search, generalized MaxSAT scheduling, and portfolio dispatch with batch/component counters,
  accepted G91/default FAEEL mode-specific solver trace counters,
  learned-clause arena publication, same-device-CNF learned-clause
  import/reuse, two-record and accepted split-batch learned-clause reuse,
  distinct-CNF learned-clause import rejection, bounded single-, split-batch,
  and multi-candidate MaxSAT solving, single-result, two-record, and
  split-batch MaxSAT search pruning plus weighted MaxSAT encoding/search,
  bounded
  single-result plus two-record SAT/MaxSAT portfolio dispatch, and
  UNKNOWN/TIMEOUT portfolio status propagation on accepted GPU runtime
  evidence;
- G91 and FAEEL fixture evaluators plus explicit G91 self-supported `possible`
  accepted runtime execution, a default FAEEL executable-plan foundedness guard,
  an independently founded zero-arity self-`possible` accepted GPU runtime
  fixture, and a nonzero-arity self-`possible` fail-closed executable guard;
- Generate-Propagate-Test phase traces;
- world-view operator fixtures for `know`, `possible`, `not know`, and
  `not possible`;
- bounded solver-service lifecycle fixtures;
- accepted-world-view probabilistic evidence fixtures and production adapter
  gates for accepted source/program exact compilation, two-record and accepted
  split-batch direct source/program exact compilation, source/program bounded
  compile/evaluate, source/program zero-arity and concrete nonzero-arity
  true/false unary and binary operator evidence conditioning with negative and
  aggregate/source/program nonzero-arity evidence, max-arity, and
  quaternary source, possible/not-know source, parsed-program, negated, and split-batch
  parsed-program nonzero-arity max-arity evidence,
  source/program-specific operator-conditioned trace counters,
  split-batch conditioned source/program query and gradient evaluation with accepted batch/component
  counters, including all-binary-operator split-batch conditioned
  source/program query and gradient fixtures, quaternary split-batch
  possible/not-know conditioned source query evidence plus source/program
  gradient, PIR/CNF, and exact query/gradient fixtures,
  mode-specific accepted G91/FAEEL production trace counters,
  two-record positive and negative conditioned source query batches,
  two-record conditioned program query batches, single-record, two-record, and
  accepted split-batch PIR/CNF encoding with source/program-specific PIR/CNF
  counters, single-record, two-record, and accepted split-batch query
  evaluation, and single-record, two-record, and accepted split-batch gradient
  evaluation.
- bounded executable split components that reuse the existing epistemic GPU
  executable-plan path and a traced batch adapter over the existing single-plan
  GPU runtime execution path, with zero CPU recomposition counters, rather than
  a split-only WCOJ or tuple-store engine.

This evidence should be retained as oracle coverage for the required GPU-native
implementation, but it cannot be used as release-close evidence.

## Missing GPU-Native Evidence

Closure remains blocked until certification includes all of the following:

- broader nonzero GPU launch counts and kernel timings for actual stable-model
  tuple membership population beyond the current unary/possible/not-possible/
  binary/ternary-specialized/quaternary-generic/multi-membership/missing-required and `not know`
  accepted fixtures;
- GPU-resident candidate, world-view, model-membership, and rejection buffers;
- zero CPU fallback counters for candidate enumeration and world-view
  validation;
- broader WCOJ-eligible epistemic reductions proving successful runtime
  dispatch beyond the current accepted v0.7.0 4-cycle and K5/K6/K7/K8 fixtures, including layout,
  skew-scheduling, and helper-splitting evidence where applicable;
- broader accepted solver semantic integration beyond the current single/multi/split
  combined lifecycle, all-binary and quaternary split-batch learned-clause/MaxSAT evidence,
  bounded scheduler, split-batch scheduler, and portfolio fixtures;
- broader accepted-world-view probabilistic coverage beyond the bounded
  split-batch conditioned source/program query/gradient, split-batch PIR/CNF,
  exact query/gradient, and PIR/CNF GPU-native knowledge-compilation fixtures,
  with zero CPU-only probability recomputation;
- broader final v0.9 certification evidence beyond the post-v0.7.0/v0.8.0/v0.8.5/v0.8.6 merge and
  compatibility validator.

## Release Hygiene

No push, tag, release-board update, merge to `main`, or v0.8-owned pyxlog API
change was performed.

## Decision

Release decision: `HOLD_FOR_GPU_NATIVE_CERTIFICATION`.

The current branch is still incomplete. The next closing work must complete the
corrected `G090_GPU` production runtime/WCOJ/GPU path and then rerun the full
v0.9 certification set before any closure proposal.
