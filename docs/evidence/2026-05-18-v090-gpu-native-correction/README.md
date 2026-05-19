# v0.9.0 GPU-Native Gate Correction

Date: 2026-05-18

Goal document: `docs/plans/2026-05-18-agent-v090-epistemic-solver-goal.md`

Branch: `feat/v090-epistemic-solver-semantics`

## Correction Summary

The corrected goal document makes fully GPU-native accepted epistemic execution
mandatory for v0.9.0. The current branch has valuable CPU-side semantic oracle
fixtures, but those fixtures are incomplete scaffolding and cannot close the GPU
release gate.

## Current Branch Classification

| Area | Current branch state | Release status |
|---|---|---|
| EIR/GPU plan | Epistemic syntax is represented explicitly; `EpistemicGpuPlan` records the GPU contract; `EpistemicExecutablePlan` carries the reduced production runtime plan plus relation IDs for accepted runtime registration. | PARTIAL until accepted semantic parity covers the required fixture matrix. |
| FAEEL/G91 foundedness boundary | Default FAEEL executable-plan lowering rejects unsupported self-supported `possible` rules before reduced runtime compilation, allows self-`possible` with independent ordinary founded support, and explicit G91 compatibility mode lowers the self-support fixture through accepted GPU runtime execution over the existing reduced fact-buffer path. | PARTIAL until full accepted-runtime FAEEL/G91 parity is proven. |
| GPU workspace | `EpistemicGpuWorkspace` maps required buffer categories to runtime `TrackedCudaSlice` handles; `EpistemicGpuWorkspaceResetTrace` records device-side reset with zero host writes; `EpistemicGpuCandidateGenerationTrace` records bounded candidate-assumption kernel launches with CUDA-event elapsed timing; `EpistemicGpuPropagationTrace` records bounded propagation staging launches with CUDA-event elapsed timing; `EpistemicGpuCandidateValidationTrace` records bounded candidate-buffer validation launches with CUDA-event elapsed timing; `EpistemicGpuModelMembershipTrace` records tuple-source model-membership launches with named reduced relation row-count device reads, tuple-key column device reads, output row-count device reads for bound-variable tuple keys, encoded ground tuple-key expectations, reduced-output column metadata for variable-bound tuple keys, binding polarity for negated membership, specialized arity-one/two/three plus generic arity-N kernels, `StableModelTupleBuffer` source, CUDA-event elapsed timing, and zero host writes while retaining row-count-only staging as a negative fixture; `EpistemicGpuWorldViewValidationTrace` records bounded candidate-assumption-aware model-membership/world-view validation launches with CUDA-event elapsed timing and rejects candidates unless every generated epistemic literal assumption has tuple-source support; `EpistemicGpuMaterializationTrace` records bounded accepted-candidate materialization launches with CUDA-event elapsed timing; `EpistemicGpuFinalResultMaterializationTrace` records final-result flag launches from reduced-output device row-count metadata with CUDA-event elapsed timing; `EpistemicGpuFinalTupleMaterializationTrace` records final-output tuple buffer launches gated by GPU model-membership/world-view buffers with device row-count read/write metadata, device row-map filtering for all bound tuple-key filters including `row_filter_count` and `negated_row_filter_count`, CUDA-event elapsed timing, and zero host writes; `EpistemicGpuSemanticTrace` decodes device rejection codes into `EpistemicGpuRejectionReason`; `EpistemicGpuTransferBudgetTrace` records zero tracked hot-path host transfers; `EpistemicGpuFinalResultTransferTrace` accounts post-hot-path final rows, columns, payload bytes, row-count metadata reads, and zero accepted-path data-plane D2H calls; `EpistemicGpuRuntimePreflight` consumes executable plans, certifies tuple-membership bindings, records WCOJ/helper route metadata including K-clique max arity, edge-permutation counts, stream-group counts, helper relation rule and WCOJ input scan counts, and explicit `know`/`possible`/`not know`/`not possible` operator counts, and rejects helper-split specs without compiler-created helper relation rewrites consumed by WCOJ; `EpistemicGpuRuntimeWcojCertification` rejects WCOJ evidence unless runtime counters advance and, for sorted-layout obligations, a layout sort or layout fast-path counter advances, then carries certified edge-permutation, stream-group, sorted-layout, helper-split, helper-rule, WCOJ helper input, layout, metadata-build counts, and metadata-build nanoseconds for the dispatched plan; `EpistemicGpuRuntimeTrace` records reduced-plan counter deltas; accepted K5/K6/K7/K8 integration evidence certifies production WCOJ dispatch and final row materialization; K5 certifies planner/scheduler/layout/helper metrics plus helper relation rewrites inside the dispatch trace; K6 certifies the G38-B helper-split plus runtime histogram metadata-build count and timing path; K7/K8 preflight evidence proves G39 K-clique planner-surface reuse including stream-group metadata; accepted unary, possible, not possible, binary, multi-membership, missing-required multi-membership, `not know`, split possible-vs-not-known nonzero-arity evidence, and a bounded GPT-oracle parity check filter, reject, or account rows by bound tuple key. | PARTIAL until accepted semantic parity is proven across the required modes. |
| World views | `EpistemicWorldView` fixtures test `know`, `possible`, and `not know`; accepted GPU world-view validation now rejects partial multi-literal support before final-row filtering, and split GPU runtime distinguishes absent `possible` from true `not know` over the same absent tuple source. | PARTIAL until all world-view modes are generated/validated on GPU. |
| GPT | CPU fixture records guesses, reduced models, accepted world views, and rejection reasons; accepted GPU runtime evidence now records generated, guess, propagated, pruned, tested, reduced-model-slot, accepted, rejected, and typed rejection-reason counts from device rejection metadata, including complete-membership rejection for multi-literal candidates and a bounded `know edge(X)` parity check against `run_generate_propagate_test`. | PARTIAL: candidate generation, propagation staging, candidate-buffer validation, tuple-source model-membership staging with specialized arity-one/two/three and generic arity-N row-scoped ground-key comparison plus generic arity-N variable-bound comparison, bounded candidate-assumption-aware world-view validation staging, accepted-candidate materialization staging, final-result flag staging, final-row map construction, membership-gated final tuple materialization, and semantic trace accounting use GPU-resident buffers; broader accepted semantic parity remains missing. |
| Splitting | CPU split/recompose fixtures pass, valid split components lower through GPU executable subplans that reuse the existing epistemic executable path, and accepted split components now execute through a batch adapter that delegates to the existing single-plan GPU runtime path while matching simple component output oracles plus the possible-vs-not-known world-view oracle. | PARTIAL until full accepted-runtime semantic parity is covered for split programs. |
| Solver | `SolverService` is a CPU fixture facade with SAT/UNSAT/UNKNOWN/TIMEOUT/Optimal statuses; `GpuSolverProductionAdapter` is a thin adapter over the existing `GpuCdclSolver` production path with accepted-runtime SAT/UNSAT, reusable workspace-backed UNSAT, single- and multi-candidate bounded lifecycle, lifecycle UNKNOWN/TIMEOUT propagation, learned-clause arena publication, same-device-CNF learned-clause import/reuse, two-record learned-clause reuse, distinct-CNF learned-clause import rejection, bounded single- and multi-candidate MaxSAT, single-result and two-record MaxSAT search pruning through GPU CDCL UNSAT, single-result and two-record weighted MaxSAT selection encoding/search through existing GPU CNF/CDCL paths, heterogeneous MaxSAT scheduler reuse, single-result and two-record bounded SAT/MaxSAT portfolio gates, UNKNOWN/TIMEOUT scheduler/portfolio status propagation, and zero CPU search counters; `production_capabilities` reports those GPU-backed adapters available while disallowing the CPU oracle for production metrics; `GpuSolverProductionTrace::require_production_metric_eligibility` rejects CPU-oracle-only traces. | PARTIAL for accepted-runtime SAT, UNSAT, workspace-backed UNSAT, bounded lifecycle, two-record candidate lifecycle, lifecycle UNKNOWN/TIMEOUT propagation, learned-clause arena publication, same-device-CNF learned-clause import/reuse, two-record learned-clause reuse, distinct-CNF learned-clause import rejection, bounded single- and multi-candidate MaxSAT, single-result and two-record MaxSAT search pruning, single-result and two-record weighted MaxSAT encoding/search, heterogeneous MaxSAT scheduler reuse, and single-result plus two-record status-aware portfolio production reuse; BLOCKED until broader solver semantic integration and post-v0.8 certification are complete. |
| Probabilistic | `AcceptedWorldViewEvidence` guards evidence conditioning in fixtures; `EpistemicProbProductionAdapter` can construct evidence from an accepted `EpistemicGpuExecutionResult`, route source and parsed programs into the existing `ExactDdnnfProgram` GPU exact/provenance compile path, source/program bounded compile/evaluate paths, two-record accepted source/program batch compile/evaluate, source/program zero-arity and concrete nonzero-arity true/false conditioned evaluation through parsed `Evidence` AST entries with negative-evidence trace counters, two-record positive and negative conditioned source query batches, two-record conditioned program query batches, conditioned source/program gradient evaluation, single-record and two-record `GpuPirGraph`/`GpuPirRoots` upload plus `encode_cnf_gpu`, and single-record plus two-record query/gradient-evaluation paths, and record zero CPU recompute counters; `EpistemicProbProductionTrace::require_production_metric_eligibility` rejects fixture-only metric traces. | PARTIAL for production exact compile/PIR-CNF/evaluation reuse; BLOCKED until broader probabilistic coverage over accepted runtime world views exists. |
| Certification | Semantic-oracle, GPU-plan contract, accepted K5/K6/K7/K8 WCOJ execution, K6 G38-B timed helper/histogram reuse, and K7/K8 K-clique preflight reuse tests can pass locally. | BLOCKED until full accepted-execution GPU timing, solver/probability traces, semantic parity, and zero CPU fallback counters exist. |

## Explicit Non-Closure Items

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
   registration metadata, accepted K5/K6/K7/K8 WCOJ dispatch fixtures,
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
   unary/possible/not-possible/binary/multi-membership variable-bound final-row
   filtering fixtures plus missing-required multi-membership rejection before
   final-row filtering, negated `not know` absent-key filtering, operator
   metrics, and final-row polarity counts;
   accepted rejection-reason
   semantic population and
   broader semantic parity remain open.
5. Route WCOJ-eligible reductions through existing planner/layout/dispatch
   machinery, including helper-splitting evidence where applicable. PARTIAL for
   accepted K5/K6/K7/K8 dispatch plus K5 certified helper/layout metrics,
   fail-closed layout sort/fast-path certification, K6 G38-B
   helper/histogram metadata count and timing reuse, and K7/K8
   planner/preflight reuse; broader helper/skew runtime coverage remains open.
6. Replace CPU solver fixture search in accepted execution with GPU-native
   SAT/MaxSAT/portfolio services or a documented GPU-backed adapter. PARTIAL
   for accepted-runtime SAT, UNSAT, reusable workspace-backed UNSAT,
   bounded lifecycle, learned-clause arena publication, same-device-CNF
   learned-clause import/reuse, two-record learned-clause reuse,
   distinct-CNF learned-clause import rejection, bounded single- and
   multi-candidate MaxSAT, single-result and two-record MaxSAT search
   pruning, and single-result plus two-record bounded status-aware portfolio reuse through
   `GpuSolverProductionAdapter`, `solve_expect_sat_with_gpu_execution_result`,
   `solve_expect_unsat_with_gpu_execution_result`, and
   `solve_expect_unsat_with_branch_limit_ws_with_gpu_execution_result` plus
   `solve_assumption_lifecycle_with_gpu_execution_result`,
   `solve_multi_candidate_assumption_lifecycle_with_gpu_execution_results`,
   `solve_unsat_and_publish_learned_clause_arena_with_gpu_execution_result`,
   `solve_unsat_then_reuse_learned_clauses_with_gpu_execution_result`,
   `solve_multi_candidate_learned_clause_reuse_with_gpu_execution_results`,
   `solve_weighted_maxsat_candidates_with_gpu_execution_result`,
   `solve_multi_candidate_weighted_maxsat_with_gpu_execution_results`,
   `solve_weighted_maxsat_search_with_gpu_execution_result`,
   `solve_multi_candidate_weighted_maxsat_search_with_gpu_execution_results`, and
   `solve_weighted_maxsat_encoded_search_with_gpu_execution_result`,
   `solve_multi_candidate_weighted_maxsat_encoded_search_with_gpu_execution_results`,
   `solve_maxsat_schedule_with_gpu_execution_results`, and
   `solve_portfolio_with_gpu_execution_result` and
   `solve_multi_candidate_portfolio_with_gpu_execution_results`; broader accepted solver
   semantic integration remains open.
7. Feed accepted world-view evidence into the existing GPU-native
   exact/provenance/PIR/CNF paths and report zero CPU-only probability
   recomputation.
   PARTIAL through `EpistemicProbProductionAdapter`,
   `compile_source_with_gpu_execution_result`,
   `compile_program_with_gpu_execution_result`, and
   `compile_and_evaluate_source_with_gpu_execution_result`,
   `compile_and_evaluate_source_for_gpu_execution_results`,
   `compile_and_evaluate_program_for_gpu_execution_results`,
   `compile_and_evaluate_conditioned_source_with_gpu_execution_result`,
   `compile_and_evaluate_conditioned_source_for_gpu_execution_results`,
   `compile_and_evaluate_conditioned_program_with_gpu_execution_result`,
   `compile_and_evaluate_conditioned_program_for_gpu_execution_results`,
   `compile_and_evaluate_conditioned_source_with_grads_with_gpu_execution_result`,
   `compile_and_evaluate_conditioned_source_with_grads_for_gpu_execution_results`,
   `compile_and_evaluate_conditioned_program_with_grads_with_gpu_execution_result`,
   `compile_and_evaluate_conditioned_program_with_grads_for_gpu_execution_results`,
   `compile_and_evaluate_program_with_gpu_execution_result`,
   `encode_source_pir_cnf_with_gpu_execution_result`,
   `encode_program_pir_cnf_with_gpu_execution_result`,
   `encode_source_pir_cnf_for_gpu_execution_results`,
   `encode_program_pir_cnf_for_gpu_execution_results`,
   `evaluate_with_gpu_execution_result`,
   `evaluate_for_gpu_execution_results`,
   `evaluate_gpu_with_grads_with_gpu_execution_result`, and
   `evaluate_gpu_with_grads_for_gpu_execution_results`; conditioned exact
   evidence preserves false `not know` assumptions with a negative-evidence
   trace counter, and broader probabilistic coverage remains open.

## Validation Status

| Command | Result |
|---|---|
| `git diff --check` | PASS |
| `cargo fmt --check` | PASS |
| `cargo test -p xlog-logic --test test_epistemic_gpu_plan` | PASS, 8 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_executable_plan` | PASS, 6 passed, 0 failed |
| `cargo test -p xlog-runtime --test test_epistemic_gpu_workspace` | PASS, 53 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_results_gate_generalized_maxsat_scheduler -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution -- --nocapture` | PASS, 59 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_eir --test test_epistemic_g91 --test test_epistemic_faeel --test test_epistemic_gpt --test test_epistemic_split --test test_epistemic_world_view --test test_epistemic_examples` | PASS, 24 passed, 0 failed |
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
