# v0.9.0 GPU-Native Gate Correction

Date: 2026-05-18

Goal document: `docs/plans/2026-05-18-agent-v090-epistemic-solver-goal.md`

Branch: `feat/v090-epistemic-solver-semantics`

## Correction Summary

The corrected goal document makes fully GPU-native accepted epistemic execution
mandatory for v0.9.0. The current branch has valuable CPU-side semantic oracle
fixtures, but those fixtures are incomplete scaffolding and cannot close the GPU
release gate.

2026-05-20 delta: the same-rule all-operator accepted GPU fixture is now
threaded through solver lifecycle, learned-clause reuse, MaxSAT, portfolio,
probabilistic source conditioning, parsed-program conditioning, gradients, and
parsed-program PIR/CNF production adapters, proving that `know`, `possible`,
`not know`, and `not possible` evidence from one accepted runtime result is
reused without CPU solver search or CPU probability recomputation. Aggregate
single-result and split-batch timing also fails closed when any component
hot-path phase lacks CUDA-event timing. Default FAEEL executable lowering now
also rejects nonzero-arity self-`possible` rules unless tuple-level
foundedness can be proven.

2026-05-20 follow-up delta: single-result quaternary `possible fact4/4` and
`not know fact4/4` accepted GPU results now route through the existing solver
SAT adapter and the existing probabilistic conditioned source adapter, recording
arity-four tuple/evidence counters, one accepted `possible` counter, one
accepted `not know` counter, exact-query counters, and zero CPU
search/recomputation. The split-batch quaternary `possible fact4/4` plus
`not know fact4/4` fixture now routes the same accepted component evidence
through the existing GPU CDCL lifecycle adapter and probabilistic conditioned
source adapter with batch/component counters, arity-four tuple/evidence
counters, balanced lifecycle pushes/retractions, exact-query counters, and zero
CPU search/recomputation. This narrows `G090_SOLVER` and `G090_PROB` but does
not close either node.

## Current Branch Classification

| Area | Current branch state | Release status |
|---|---|---|
| EIR/GPU plan | Epistemic syntax is represented explicitly; `EpistemicGpuPlan` records the GPU contract; `EpistemicExecutablePlan` carries the reduced production runtime plan plus relation IDs for accepted runtime registration. | PARTIAL until accepted semantic parity covers the required fixture matrix. |
| FAEEL/G91 foundedness boundary | Default FAEEL executable-plan lowering rejects unsupported self-supported `possible` rules before reduced runtime compilation, allows zero-arity self-`possible` with independent ordinary founded support, rejects nonzero-arity self-`possible` without tuple-level foundedness proof, and explicit G91 compatibility mode lowers the self-support fixture through accepted GPU runtime execution over the existing reduced fact-buffer path; both accepted FAEEL and G91 zero-arity self-`possible` runtime fixtures now carry oracle parity for counts and accepted/rejected candidate indices. | PARTIAL until full accepted-runtime FAEEL/G91 parity is proven. |
| GPU workspace | `EpistemicGpuWorkspace` maps required buffer categories to runtime `TrackedCudaSlice` handles; `EpistemicGpuWorkspaceResetTrace` records device-side reset with zero host writes; `EpistemicGpuCandidateGenerationTrace` records bounded candidate-assumption kernel launches with CUDA-event elapsed timing; `EpistemicGpuPropagationTrace` records bounded propagation staging launches with CUDA-event elapsed timing; `EpistemicGpuCandidateValidationTrace` records bounded candidate-buffer validation launches with CUDA-event elapsed timing; `EpistemicGpuModelMembershipTrace` records tuple-source model-membership launches with named reduced relation row-count device reads, tuple-key column device reads, output row-count device reads for bound-variable tuple keys, encoded ground tuple-key expectations, reduced-output column metadata for variable-bound tuple keys, binding polarity for negated membership, specialized arity-one/two/three plus generic arity-N kernels, `StableModelTupleBuffer` source, CUDA-event elapsed timing, and zero host writes while retaining row-count-only staging as a negative fixture; `EpistemicGpuWorldViewValidationTrace` records bounded candidate-assumption-aware model-membership/world-view validation launches with CUDA-event elapsed timing and rejects candidates unless every generated epistemic literal assumption has tuple-source support; `EpistemicGpuMaterializationTrace` records bounded accepted-candidate materialization launches with CUDA-event elapsed timing; `EpistemicGpuFinalResultMaterializationTrace` records final-result flag launches from reduced-output device row-count metadata with CUDA-event elapsed timing; `EpistemicGpuFinalTupleMaterializationTrace` records final-output tuple buffer launches gated by GPU model-membership/world-view buffers with device row-count read/write metadata, device row-map filtering for all bound tuple-key filters including `row_filter_count` and `negated_row_filter_count`, CUDA-event elapsed timing, and zero host writes; `EpistemicGpuSemanticTrace` decodes device rejection codes into `EpistemicGpuRejectionReason` and records accepted/rejected candidate indices; `EpistemicGpuTransferBudgetTrace` records zero tracked hot-path host transfers; `EpistemicGpuFinalResultTransferTrace` accounts post-hot-path final rows, columns, payload bytes, row-count metadata reads, and zero accepted-path data-plane D2H calls; `EpistemicGpuBatchExecutionTrace` aggregates split component executions with zero CPU recomposition, zero per-candidate host round trips, binary `possible`/`not possible` component parity, all-binary-operator component parity, split quaternary operator component parity, aggregate `know`/`possible`/`not know`/`not possible` operator counts, and aggregate CUDA-event timing; `EpistemicGpuRuntimePreflight` consumes executable plans, certifies tuple-membership bindings, records WCOJ/helper route metadata including K-clique max arity, edge-permutation counts, stream-group counts, helper relation rule and WCOJ input scan counts, and explicit `know`/`possible`/`not know`/`not possible` operator counts, and rejects helper-split specs without compiler-created helper relation rewrites consumed by WCOJ; `EpistemicGpuRuntimeWcojCertification` rejects WCOJ evidence unless runtime counters advance and, for sorted-layout obligations, a layout sort or layout fast-path counter advances, then carries certified edge-permutation, stream-group, sorted-layout, helper-split, helper-rule, WCOJ helper input, layout, metadata-build counts, and metadata-build nanoseconds for the dispatched plan; `EpistemicGpuRuntimeTrace` records reduced-plan counter deltas; accepted K5/K6/K7/K8 integration evidence certifies production WCOJ dispatch and final row materialization; K5 certifies planner/scheduler/layout/helper metrics plus helper relation rewrites inside the dispatch trace; K6 certifies the G38-B helper-split plus runtime histogram metadata-build count and timing path; K7/K8 preflight evidence proves G39 K-clique planner-surface reuse including stream-group metadata; accepted unary, possible, not possible, binary `know`, binary `possible`, binary `not possible`, binary `not know`, ternary specialized-arity `know`, quaternary all-operator generic-arity fixtures, all-`know` multi-membership, mixed `know`/`possible` multi-membership, negated `not know`/`not possible` multi-membership, all-operator mixed-membership, missing-required multi-membership, `not know`, split possible-vs-not-known nonzero-arity evidence, split all-binary-operator evidence, split quaternary operator evidence, and bounded GPT-oracle parity checks filter, reject, or account rows by bound tuple key. | PARTIAL until accepted semantic parity is proven across the required modes. |
| World views | `EpistemicWorldView` fixtures test `know`, `possible`, and `not know`; accepted GPU world-view validation now rejects partial multi-literal support before final-row filtering, and split GPU runtime distinguishes absent `possible` from true `not know` over the same absent tuple source. | PARTIAL until all world-view modes are generated/validated on GPU. |
| GPT | CPU fixture records guesses, reduced models, accepted world views, accepted/rejected candidate indices, rejection reasons, default FAEEL outcomes, and explicit G91 mode outcomes; accepted GPU runtime evidence now records generated, guess, propagated, pruned, tested, reduced-model-slot, accepted, rejected, accepted/rejected candidate indices, and typed rejection-reason counts from device rejection metadata, including complete-membership rejection for multi-literal candidates, a bounded `know edge(X)` parity check against `run_generate_propagate_test`, unary nonzero-arity `possible edge(X)`, `not possible edge(X)`, and `not know edge(X)` operator parity checks against bounded oracle candidate-index vectors, binary `know edge(X, Y)`, `possible edge(X, Y)`, `not possible edge(X, Y)`, and `not know edge(X, Y)` operator parity checks against bounded oracle candidate-index vectors, a ternary `know fact3(A, B, C)` specialized arity-three parity check against bounded oracle candidate-index vectors, quaternary `know fact4(A, B, C, D)`, `possible fact4(A, B, C, D)`, `not know fact4(A, B, C, D)`, and `not possible fact4(A, B, C, D)` generic arity-N parity checks against bounded oracle candidate-index vectors, a two-component split quaternary `know fact4/4` plus `not possible fact4/4` parity check against bounded oracle candidate-index vectors, two-literal all-`know` `know edge(X), know color(X)`, mixed `know edge(X), possible alt(X)`, negated mixed `not know edge(X), not possible blocked(X)`, and all-operator `know edge(X), possible alt(X), not know hidden(X), not possible blocked(X)` multi-membership parity checks against oracle candidate-index vectors, independently founded FAEEL self-`possible p()` parity against the default oracle, and a G91 self-supported `possible p()` parity check against `run_generate_propagate_test_with_mode`. | PARTIAL: candidate generation, propagation staging, candidate-buffer validation, tuple-source model-membership staging with specialized arity-one/two/three and generic arity-N row-scoped ground-key comparison plus generic arity-N variable-bound comparison, bounded candidate-assumption-aware world-view validation staging, accepted-candidate materialization staging, final-result flag staging, final-row map construction, membership-gated final tuple materialization, and semantic trace accounting use GPU-resident buffers; broader accepted semantic parity remains missing. |
| Splitting | CPU split/recompose fixtures pass, valid split components lower through GPU executable subplans that reuse the existing epistemic executable path, and accepted split components now execute through a traced batch adapter that delegates to the existing single-plan GPU runtime path while matching simple component output oracles, binary `possible`/`not possible` component output oracles, all-binary-operator component output oracles, quaternary `know fact4/4` plus `not possible fact4/4` component output oracles, per-component GPT trace/candidate-index oracles, aggregate zero CPU recomposition/per-candidate-host-round-trip counters, aggregate epistemic operator counters, and the possible-vs-not-known world-view oracle. | PARTIAL until full accepted-runtime semantic parity is covered for split programs. |
| Solver | `SolverService` is a CPU fixture facade with SAT/UNSAT/UNKNOWN/TIMEOUT/Optimal statuses; `GpuSolverProductionAdapter` is a thin adapter over the existing `GpuCdclSolver` production path with accepted-runtime SAT/UNSAT, reusable workspace-backed UNSAT, single-, multi-candidate, split-batch, and combined lifecycle-plus-MaxSAT bounded lifecycle through `GpuSolverProductionBatchExecutionEvidence`, accepted split-batch combined lifecycle-plus-MaxSAT, fail-closed empty MaxSAT lifecycle input rejection before lifecycle trace mutation, fail-closed all-UNSAT MaxSAT search rejection before solver trace mutation, fail-closed all-UNSAT encoded MaxSAT rejection before accepted-evidence or encode trace mutation, fail-closed invalid encoded MaxSAT scheduler rejection before accepted-batch evidence, scheduler, encode, or solver trace mutation, fail-closed rejection when split-batch evidence lacks aggregate CUDA-event timing, fail-closed rejection when single-result evidence lacks candidate-generation CUDA-event timing, accepted split-batch learned-clause reuse, accepted split-batch MaxSAT, accepted split-batch MaxSAT search pruning, accepted split-batch weighted MaxSAT encoding/search, accepted split-batch generalized MaxSAT scheduling, accepted split-batch/component counters, all-binary-operator split-batch lifecycle plus all-binary split-batch learned-clause reuse and MaxSAT with accepted `know`/`possible`/`not possible`/`not know` solver evidence counters, accepted G91/default FAEEL mode-specific solver trace counters, accepted nonzero-arity tuple-key evidence counters, mixed unary and binary `possible`/`not possible` plus binary `not know` operator-result lifecycle, lifecycle UNKNOWN/TIMEOUT propagation, learned-clause arena publication, same-device-CNF learned-clause import/reuse, two-record learned-clause reuse, distinct-CNF learned-clause import rejection, bounded single- and multi-candidate MaxSAT, single-result, two-record, and split-batch MaxSAT search pruning through GPU CDCL UNSAT, single-result, two-record, and split-batch weighted MaxSAT selection encoding/search through existing GPU CNF/CDCL paths, heterogeneous and split-batch MaxSAT scheduler reuse, single-result, two-record, and split-batch bounded SAT/MaxSAT portfolio gates, UNKNOWN/TIMEOUT scheduler/portfolio status propagation, and zero CPU search counters; `production_capabilities` reports those GPU-backed adapters available while disallowing the CPU oracle for production metrics; `GpuSolverProductionTrace::require_production_metric_eligibility` rejects traces without accepted GPU candidate evidence, status-only UNKNOWN/TIMEOUT traces, traces without an existing GPU CDCL/MaxSAT/scheduler/portfolio production-path counter, or traces with CPU search counters. | PARTIAL for accepted-runtime SAT, UNSAT, workspace-backed UNSAT, bounded lifecycle, two-record candidate lifecycle, accepted ternary and quaternary nonzero-arity SAT evidence tracing, single-result, two-record, and split-batch combined lifecycle-plus-MaxSAT plus empty-candidate, all-UNSAT search, all-UNSAT encoded-search, and invalid encoded-scheduler fail-closed rejection, accepted split-batch lifecycle, all-binary-operator accepted split-batch lifecycle plus all-binary split-batch learned-clause reuse and MaxSAT, learned-clause reuse, MaxSAT, MaxSAT search pruning, weighted MaxSAT encoding/search, generalized MaxSAT scheduling, and portfolio dispatch, accepted G91/default FAEEL mode-specific solver trace counters, mixed unary and binary `possible`/`not possible` plus binary `not know` operator-result lifecycle, lifecycle UNKNOWN/TIMEOUT propagation, learned-clause arena publication, same-device-CNF learned-clause import/reuse, two-record and accepted split-batch learned-clause reuse, distinct-CNF learned-clause import rejection, bounded single-, multi-candidate, and split-batch MaxSAT, single-result, two-record, and split-batch MaxSAT search pruning, single-result, two-record, and split-batch weighted MaxSAT encoding/search, heterogeneous and split-batch MaxSAT scheduler reuse, and single-result plus two-record status-aware portfolio production reuse; BLOCKED until broader solver semantic integration and post-v0.7.0/v0.8.0/v0.8.5/v0.8.6 bundle certification are complete. |
| Probabilistic | `AcceptedWorldViewEvidence` guards evidence conditioning in fixtures; `EpistemicProbProductionAdapter` can construct evidence from an accepted `EpistemicGpuExecutionResult`, preserve accepted G91/default FAEEL runtime modes in production trace counters, route source and parsed programs into the existing `ExactDdnnfProgram` GPU exact/provenance compile path, source/program bounded compile/evaluate paths, two-record accepted source/program batch compile/evaluate, split-batch conditioned source/program query and gradient evaluation through `EpistemicProbGpuBatchExecutionEvidence` with accepted batch/component counters and fail-closed aggregate CUDA-event timing validation, fail-closed single-result candidate-generation CUDA-event timing validation, all-binary-operator split-batch conditioned source and parsed-program query plus source and parsed-program gradient evidence with true/false `know`/`possible` operator assumptions, all-binary split-batch source/program PIR-CNF plus already-compiled exact query/gradient evaluation, source/program zero-arity and concrete nonzero-arity true/false conditioned evaluation through parsed `Evidence` AST entries with negative-evidence, source/program-specific conditioned-evidence, aggregate/source/program nonzero-arity and max-arity evidence including ternary source, quaternary source, and quaternary parsed-program fixtures, aggregate operator-specific, and source/program-specific operator-conditioned trace counters including true `know`, true `possible`, false `possible`/`not possible`, and false `know`/`not know` operator results, two-record positive and negative conditioned source query batches, two-record conditioned program query batches, conditioned source/program gradient evaluation with source/program-specific gradient counters, single-record and two-record `GpuPirGraph`/`GpuPirRoots` upload plus `encode_cnf_gpu` with source/program-specific PIR/CNF counters, and single-record plus two-record query/gradient-evaluation paths, and record zero CPU recompute counters; `EpistemicProbProductionTrace::require_production_metric_eligibility` rejects traces without accepted world-view evidence, conditioned evidence facts alone, traces without an aggregate or source/program-specific GPU exact/provenance/PIR/CNF/knowledge-compilation path counter, or traces with CPU/fixture recomputation counters. | PARTIAL for production exact compile/PIR-CNF/evaluation reuse; BLOCKED until broader probabilistic coverage over accepted runtime world views exists. |
| Certification | Semantic-oracle, GPU-plan contract, accepted v0.7.0 4-cycle WCOJ execution, accepted K5/K6/K7/K8 WCOJ execution, K6 G38-B timed helper/histogram reuse, and K7/K8 K-clique preflight reuse tests can pass locally. | BLOCKED until full accepted-execution GPU timing, solver/probability traces, semantic parity, and zero CPU fallback counters exist. |

Solver metric-gate delta: lifecycle UNKNOWN/TIMEOUT propagation remains
failure-mode evidence, but it does not by itself satisfy production solver
reuse metrics. `GpuSolverProductionTrace::require_production_metric_eligibility`
also requires a GPU CDCL, MaxSAT, scheduler, or portfolio production-path
counter.

## Explicit Non-Closure Items

The v0.7.0 reuse correction is specifically about general WCOJ coverage:
accepted epistemic 4-cycle execution now certifies the production
`wcoj_4cycle_dispatch_count` path, and runtime WCOJ certification fails closed
for required non-hash `MultiWayJoin` reductions even when no K-clique metadata
is present.

The following corrected goal nodes remain unclosed:

- `G090_GPU`
- `G090_SOLVER`
- `G090_PROB`
- `G090_CERT`
- `G090_CLOSE`

`G090_GPT` and `G090_SPLIT` are also only partial because their broader
GPU-residency and semantic-parity metrics are not complete.

## Required Next Implementation Slice

The next production slice should start at the lowering/runtime boundary:

1. Define an epistemic executable-plan representation that preserves the
   `EpistemicWorldView` contract and attaches zero-fallback counters. DONE for
   the plan contract in `EpistemicGpuPlan`; runtime execution remains open.
2. Map plan buffer categories to runtime GPU workspace allocations. DONE for
   layout, `TrackedCudaSlice` handles, and device-side reset; accepted semantic
   parity remains open.
3. Lower accepted EIR into production runtime plans instead of the current
   `UnsupportedEpistemicConstruct` boundary. DONE for
   `compile_epistemic_gpu_execution`, its stats-aware variant, relation-ID
   registration metadata, accepted v0.7.0 4-cycle and K5/K6/K7/K8 WCOJ dispatch fixtures,
   fail-closed layout sort/fast-path certification, K6 G38-B
   helper/histogram metadata count and timing reuse, and K7/K8 K-clique
   preflight reuse of G39 planner metadata.
4. Add GPU-resident candidate/world-view/rejection buffer population and launch
   telemetry. PARTIAL for bounded candidate-assumption generation, propagation
   staging, candidate-buffer validation, tuple-source model-membership staging
   with specialized arity-one/two/three and generic arity-N row-scoped ground
   key comparison plus generic arity-N variable-bound comparison, bounded
   candidate-assumption-aware world-view validation staging,
   accepted-candidate materialization staging,
   final-result flag staging, membership-gated final tuple materialization,
   CUDA-event elapsed timing for those staging launches, and hot-path
   transfer-budget tracing, final-result transfer accounting, and accepted
   unary/possible/not-possible/not-know/binary/ternary-specialized/quaternary-generic/all-`know` multi-membership/mixed `know`-`possible` multi-membership/negated `not know`-`not possible` multi-membership variable-bound final-row
   filtering fixtures plus missing-required multi-membership rejection before
   final-row filtering, negated `not know` absent-key filtering, operator
   metrics, and final-row polarity counts;
   accepted rejection-reason
   semantic population and
   broader semantic parity remain open.
5. Route WCOJ-eligible reductions through existing planner/layout/dispatch
   machinery, including helper-splitting evidence where applicable. PARTIAL for
   accepted v0.7.0 4-cycle dispatch, accepted K5/K6/K7/K8 dispatch plus K5 certified helper/layout metrics,
   fail-closed layout sort/fast-path certification, K6 G38-B
   helper/histogram metadata count and timing reuse, and K7/K8
   planner/preflight reuse; broader helper/skew runtime coverage remains open.
6. Replace CPU solver fixture search in accepted execution with GPU-native
   SAT/MaxSAT/portfolio services or a documented GPU-backed adapter. PARTIAL
   for accepted-runtime SAT, UNSAT, reusable workspace-backed UNSAT,
   bounded unary/binary operator lifecycle, accepted split-batch lifecycle,
   all-binary-operator accepted split-batch lifecycle plus all-binary
   split-batch learned-clause reuse and MaxSAT,
   accepted ternary and quaternary nonzero-arity SAT evidence tracing,
   single-result quaternary `possible`/`not know fact4/4` SAT evidence,
   split-batch quaternary `possible`/`not know fact4/4` lifecycle evidence,
   single-result, two-record, and split-batch combined lifecycle-plus-MaxSAT,
   learned-clause reuse, split-batch MaxSAT, split-batch MaxSAT search pruning, and split-batch portfolio dispatch,
   learned-clause arena publication, same-device-CNF
   learned-clause import/reuse, two-record and accepted split-batch learned-clause reuse,
   distinct-CNF learned-clause import rejection, bounded single- and
   multi-candidate MaxSAT, single-result, two-record, and split-batch MaxSAT search
   pruning, and single-result plus two-record bounded status-aware portfolio reuse through
   `GpuSolverProductionAdapter`, `solve_expect_sat_with_gpu_execution_result`,
   `solve_expect_unsat_with_gpu_execution_result`, and
   `solve_expect_unsat_with_branch_limit_ws_with_gpu_execution_result` plus
   `solve_assumption_lifecycle_with_gpu_execution_result`,
   `solve_multi_candidate_assumption_lifecycle_with_gpu_execution_results`,
   `solve_assumption_lifecycle_with_gpu_batch_execution_result`,
   `solve_maxsat_lifecycle_with_gpu_execution_result`,
   `solve_multi_candidate_maxsat_lifecycle_with_gpu_execution_results`,
   `solve_maxsat_lifecycle_with_gpu_batch_execution_result`,
   `solve_unsat_and_publish_learned_clause_arena_with_gpu_execution_result`,
   `solve_unsat_then_reuse_learned_clauses_with_gpu_execution_result`,
   `solve_multi_candidate_learned_clause_reuse_with_gpu_execution_results`,
   `solve_learned_clause_reuse_with_gpu_batch_execution_result`,
   `solve_weighted_maxsat_candidates_with_gpu_execution_result`,
   `solve_multi_candidate_weighted_maxsat_with_gpu_execution_results`,
   `solve_weighted_maxsat_candidates_with_gpu_batch_execution_result`,
   `solve_weighted_maxsat_search_with_gpu_execution_result`,
   `solve_multi_candidate_weighted_maxsat_search_with_gpu_execution_results`,
   `solve_weighted_maxsat_search_with_gpu_batch_execution_result`, and
   `solve_weighted_maxsat_encoded_search_with_gpu_execution_result`,
   `solve_multi_candidate_weighted_maxsat_encoded_search_with_gpu_execution_results`,
   `solve_weighted_maxsat_encoded_search_with_gpu_batch_execution_result`,
   `solve_maxsat_schedule_with_gpu_execution_results`,
   `solve_maxsat_schedule_with_gpu_batch_execution_result`, and
   `solve_portfolio_with_gpu_execution_result`,
   `solve_multi_candidate_portfolio_with_gpu_execution_results`, and
   `solve_portfolio_with_gpu_batch_execution_result`; accepted G91/default
   FAEEL mode-specific solver evidence counters, accepted operator-family
   solver evidence counters, all-binary-operator split-batch lifecycle
   counters, and accepted nonzero-arity tuple-key evidence
   counters are recorded, but broader accepted solver semantic integration
   remains open.
7. Feed accepted world-view evidence into the existing GPU-native
   exact/provenance/PIR/CNF paths and report zero CPU-only probability
   recomputation.
   PARTIAL through `EpistemicProbProductionAdapter`,
   `compile_source_with_gpu_execution_result`,
   `compile_program_with_gpu_execution_result`, and
   `compile_and_evaluate_source_with_gpu_execution_result`,
   `compile_and_evaluate_source_for_gpu_execution_results`,
   `compile_and_evaluate_source_for_gpu_batch_execution_result`,
   `compile_and_evaluate_program_for_gpu_execution_results`,
   `compile_and_evaluate_program_for_gpu_batch_execution_result`,
   `compile_and_evaluate_conditioned_source_with_gpu_execution_result`,
   `compile_and_evaluate_conditioned_source_for_gpu_execution_results`,
   `compile_and_evaluate_conditioned_source_for_gpu_batch_execution_result`,
   `compile_and_evaluate_conditioned_program_with_gpu_execution_result`,
   `compile_and_evaluate_conditioned_program_for_gpu_execution_results`,
   `compile_and_evaluate_conditioned_program_for_gpu_batch_execution_result`,
   `compile_and_evaluate_conditioned_source_with_grads_with_gpu_execution_result`,
   `compile_and_evaluate_conditioned_source_with_grads_for_gpu_execution_results`,
   `compile_and_evaluate_conditioned_source_with_grads_for_gpu_batch_execution_result`,
   `compile_and_evaluate_conditioned_program_with_grads_with_gpu_execution_result`,
   `compile_and_evaluate_conditioned_program_with_grads_for_gpu_execution_results`,
   `compile_and_evaluate_conditioned_program_with_grads_for_gpu_batch_execution_result`,
   `compile_and_evaluate_program_with_gpu_execution_result`,
   `encode_source_pir_cnf_with_gpu_execution_result`,
   `encode_program_pir_cnf_with_gpu_execution_result`,
   `encode_source_pir_cnf_for_gpu_execution_results`,
   `encode_program_pir_cnf_for_gpu_execution_results`,
   `evaluate_with_gpu_execution_result`,
   `evaluate_for_gpu_execution_results`,
   `evaluate_gpu_with_grads_with_gpu_execution_result`, and
   `evaluate_gpu_with_grads_for_gpu_execution_results`; conditioned exact
   evidence preserves true and false operator assumptions with negative,
   accepted split-batch/component evidence,
   source/program-specific conditioned-evidence, source/program-specific
   operator-conditioned evidence, source/program-specific conditioned-gradient,
   operator-specific, two-record quaternary `possible`/`not know fact4/4`
   conditioned source evidence, split-batch quaternary `possible`/`not know
   fact4/4` conditioned source evidence, all-binary-operator split-batch conditioned
   source/program query and gradient evidence, and accepted G91/default FAEEL
   mode-specific trace counters,
   and broader probabilistic coverage remains open.

## Validation Status

| Command | Result |
|---|---|
| `git diff --check` | PASS |
| `cargo fmt --check` | PASS |
| `cargo test -p xlog-logic --test test_epistemic_gpu_plan` | PASS, 8 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_executable_plan` | PASS, 8 passed, 0 failed |
| `cargo test -p xlog-runtime --test test_epistemic_gpu_workspace` | PASS, 54 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_results_gate_generalized_maxsat_scheduler -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_quaternary_membership_matches_gpt_oracle_parity -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_binary_possible_membership_matches_gpt_oracle_parity -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_binary_not_possible_membership_matches_gpt_oracle_parity -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_binary_not_know_membership_matches_gpt_oracle_parity -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_probabilistic_source_and_program_end_to_end_paths -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_probabilistic_conditioned_source_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_all_binary_operator_batch_conditions_probabilistic_evidence -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution rejects_unrecorded_aggregate_kernel_timing -- --nocapture` | PASS, 2 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution rejects_unrecorded_candidate_generation_timing -- --nocapture` | PASS, 2 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution aggregate_timing_requires_every_component_phase_to_be_recorded -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_probabilistic_conditioned_program_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_probabilistic_conditioned_source_gradients -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_probabilistic_conditioned_program_gradients -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_quaternary_source_probabilistic_evidence_records_nonzero_arity_trace -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_quaternary_parsed_program_probabilistic_evidence_records_nonzero_arity_trace -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_solver_lifecycle_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_all_binary_operator_batch_gates_solver_lifecycle_path -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_all_binary_operator_batch_gates_solver_reuse_and_maxsat_paths -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_rejects_empty_maxsat_lifecycle_before_lifecycle_work -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_rejects_all_unsat_maxsat_search_before_solver_work -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_rejects_all_unsat_encoded_maxsat_before_encoding_work -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_solver_maxsat_lifecycle_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_solver_learned_clause_reuse_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_solver_maxsat_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_solver_maxsat_search_pruning -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_solver_encoded_maxsat_and_scheduler_paths -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_solver_portfolio_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_binary_operator_components_match_gpt_oracles -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_all_binary_operators_match_gpt_oracles -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_quaternary_not_possible_batch_matches_gpt_oracles -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_quaternary_possible_and_not_know_memberships_match_gpt_oracle_parity -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_quaternary_possible_and_not_know_results_gate_solver_and_probabilistic_paths -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_quaternary_possible_and_not_know_batch_gates_solver_and_probabilistic_paths -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_ternary_gpu_execution_result_records_solver_nonzero_arity_evidence_trace -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_quaternary_gpu_execution_result_records_solver_nonzero_arity_evidence_trace -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_epistemic_v070_4cycle_execution_certifies_production_wcoj_dispatch -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_epistemic_k5_execution_certifies_production_wcoj_dispatch -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_all_binary_operator_batch_gates_probabilistic_program_and_gradient_paths -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_all_binary_operator_batch_gates_probabilistic_pir_cnf_and_exact_evaluation_paths -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_all_operator_mixed_memberships_match_gpt_oracle_parity -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_all_operator_mixed_membership_gates_solver_lifecycle_path -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_all_operator_mixed_membership_conditions_probabilistic_evidence -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_all_operator_mixed_membership_gates_solver_reuse_maxsat_and_portfolio_paths -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_all_operator_mixed_membership_gates_probabilistic_program_gradient_and_pir_paths -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution -- --nocapture` | PASS, 122 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_eir --test test_epistemic_g91 --test test_epistemic_faeel --test test_epistemic_gpt --test test_epistemic_split --test test_epistemic_world_view --test test_epistemic_examples` | PASS, 25 passed, 0 failed |
| `cargo test -p xlog-solve --test gpu_solver_production_reuse` | PASS, 3 passed, 0 failed |
| `cargo test -p xlog-solve --test solver_service_semantics` | PASS, 5 passed, 0 failed |
| `cargo test -p xlog-solve --test no_dtoh_in_gpu_cdcl` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-prob --test epistemic_prob_production_reuse` | PASS, 3 passed, 0 failed |
| `cargo test -p xlog-prob --test epistemic_prob` | PASS, 5 passed, 0 failed |
| `cargo test -p xlog-prob --test no_cpu_d4_in_exact` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-prob --test no_dtoh_in_gpu_exact_path` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-runtime --lib` | PASS, 128 passed, 0 failed |
| `cargo test -p xlog-logic --lib` | PASS, 238 passed, 0 failed |
| `cargo test -p xlog-solve --lib` | PASS, 111 passed, 0 failed |
| `cargo test -p xlog-prob --lib` | PASS, 56 passed, 0 failed |
| `cargo check -p xlog-logic -p xlog-ir -p xlog-solve -p xlog-prob` | PASS |
| `cargo check -p xlog-prob --features host-io` | PASS |
| `cargo check -p xlog-cuda -p xlog-runtime -p xlog-logic -p xlog-ir` | PASS |
| `cargo check -p pyxlog` | PASS |

These are semantic-oracle and workspace-health checks only. They do not satisfy
the corrected GPU-native release gate.
