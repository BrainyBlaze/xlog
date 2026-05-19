# v0.9.0 Production Reuse Source Audit

Date: 2026-05-18

Goal node: `G090_CERT - Certification And Regression Gates`

Metric: `M090_CERT.13 no parallel engines`

Branch: `feat/v090-epistemic-solver-semantics`

## Scope

This artifact is machine-checkable source-audit evidence for the v0.9.0
requirement that accepted epistemic execution reuses existing production paths.
It is not a closure claim. The current branch still has runtime, solver, and
probability blockers before `G090_CLOSE`.

The audit follows the same closure-evidence discipline used by the prior G38,
G38-B, and G39 chains: production path reuse must be proven by concrete source
markers, counters, and explicit blocker status, while board, merge, push, and
tag gates remain separate user-authorized actions.

## Prior Evidence Reuse

| Prior chain | Reused rule | v0.9.0 application |
|---|---|---|
| G38 completion audit | Production reuse claims must name the single production launch or dispatch path, as in the cached HG u32 triangle audit. | The v0.9.0 runtime audit names the existing RIR runtime dispatch, `MultiwayPlan::WcojWithPlan`, and `CudaBuffer` relation columns instead of accepting a parallel epistemic WCOJ path. |
| G38-B integration audit | Hypergraph planner and K-clique work is production evidence only when source/routing audits prove the production path is used. | The v0.9.0 audit requires WCOJ/K-clique/helper-split metadata from the existing runtime plan and keeps successful accepted dispatch as a remaining blocker until launch counters prove it. |
| G39 completion audit | Local artifact completion is separate from user-gated board, merge, push, and tag actions. | This artifact updates certification evidence only; it does not mark `G090_CERT` or `G090_CLOSE` complete and does not imply a board or release action. |

## Accepted-Path Audit

| Area | Reused production path | Audit evidence | Remaining blocker |
|---|---|---|---|
| Runtime/WCOJ lowering | Existing logic compiler and runtime RIR execution | `compile_epistemic_gpu_execution` lowers reduced ordinary programs through `compile_program_with_stats_snapshot`; `compile_epistemic_gpu_split_execution` lowers valid split components through the same executable path; runtime calls `self.execute_plan(&executable.reduced_runtime_plan)`, records `EpistemicGpuRuntimePreflight`, certifies accepted K5, K6, K7, and K8 WCOJ dispatch from production counter deltas, records post-hot-path final-result transfer accounting through `EpistemicGpuFinalResultTransferTrace`, records device-derived GPT counts through `EpistemicGpuSemanticTrace`, and keeps K6 G38-B helper/histogram plus K7/K8 G39 preflight coverage through the production K-clique surfaces. Split execution uses `execute_epistemic_gpu_execution_batch`, which delegates each component executable to the existing single-plan GPU runtime path instead of introducing a split-only engine. | Broader accepted WCOJ runtime dispatch and semantic parity coverage are still missing. |
| WCOJ layout and scheduling | Existing WCOJ, K-clique, planned-hash, sorted-layout, helper-split, and runtime-histogram route metadata | `summarize_runtime_routes` inspects `MultiwayPlan::WcojWithPlan`, counts `kclique_wcoj_max_arity`, `kclique_wcoj_edge_permutation_count`, `helper_split_spec_count`, `planned_hash_route_count`, sorted-layout requirements, and runtime WCOJ dispatch deltas; `EpistemicGpuRuntimeWcojCertification::Certified` carries `certified_edge_permutation_slots`, `certified_sorted_layout_requirements`, `certified_helper_split_specs`, and observed metadata builds only after production K-clique dispatch counters advance. The accepted K5 fixture asserts `10` edge slots, `1` sorted-layout requirement, and `1` helper-split spec inside the certified dispatch trace; accepted K6 asserts `15` edge slots, nonzero sorted-layout/helper-split counts, `wcoj_clique6_dispatch_count >= 1`, and `kclique_metadata_build_count >= 1`; accepted K7 and K8 fixtures supply production `wcoj_clique7_dispatch_count` and `wcoj_clique8_dispatch_count` evidence with edge-permutation metadata, and `epistemic_k7_k8_reductions_reuse_g39_kclique_planner_preflight_surface` proves K7/K8 epistemic reductions carry full production edge-permutation metadata through preflight. | Broader helper/skew runtime coverage remains missing beyond the accepted K5/K6 certified helper/histogram traces. |
| Tuple membership | Existing relation layouts and device buffers | Membership staging reads existing `CudaBuffer` relation columns via `source_relation.column(...)`, compares against reduced output columns via `output.column(bound_col_index)`, and carries the certified binding polarity; final row filtering flattens per-membership row-count pointer/key-offset/polarity metadata over the same relation and reduced-output device buffers; accepted unary, binary, multi-membership, and `not know` fixtures filter final rows by tuple key on device. | Broader semantic parity fixtures for nonzero-arity accepted world views are still missing. |
| Solver | Existing `xlog-solve` GPU CNF/CDCL production path | `GpuSolverProductionAdapter` wraps `GpuCdclSolver::new`, validates accepted `EpistemicGpuExecutionResult` evidence through `solve_expect_sat_with_gpu_execution_result`, `solve_expect_unsat_with_gpu_execution_result`, `solve_expect_unsat_with_branch_limit_ws_with_gpu_execution_result`, `solve_assumption_lifecycle_with_gpu_execution_result`, `solve_multi_candidate_assumption_lifecycle_with_gpu_execution_results`, `solve_unsat_and_publish_learned_clause_arena_with_gpu_execution_result`, `solve_unsat_then_reuse_learned_clauses_with_gpu_execution_result`, `solve_multi_candidate_learned_clause_reuse_with_gpu_execution_results`, `solve_weighted_maxsat_candidates_with_gpu_execution_result`, `solve_multi_candidate_weighted_maxsat_with_gpu_execution_results`, `solve_weighted_maxsat_search_with_gpu_execution_result`, `solve_multi_candidate_weighted_maxsat_search_with_gpu_execution_results`, `solve_weighted_maxsat_encoded_search_with_gpu_execution_result`, `solve_multi_candidate_weighted_maxsat_encoded_search_with_gpu_execution_results`, and `solve_portfolio_with_gpu_execution_result`, then calls existing SAT/UNSAT APIs, including reusable workspace-backed UNSAT, bounded lifecycle steps with `candidate_evidence_records`, `gpu_assumption_pushes`, `gpu_assumption_retractions`, `gpu_lifecycle_workspace_reuses`, `gpu_lifecycle_unknown_status_steps`, and `gpu_lifecycle_timeout_status_steps`, learned-clause arena publication plus same-device-CNF import/reuse with `gpu_learned_clause_*` counters, two-record learned-clause reuse with `candidate_evidence_records`, distinct-CNF import rejection with `gpu_learned_clause_reuse_rejections`, bounded single- and two-record MaxSAT candidate solves with `candidate_evidence_records`, `gpu_maxsat_candidate_solves`, `gpu_maxsat_optima`, single-result and two-record MaxSAT search pruning with `gpu_maxsat_unsat_candidate_prunes`, single-result and two-record weighted soft-clause selection encoding through `GpuCnf::from_host` with `gpu_maxsat_candidate_encodes` and `gpu_cdcl_candidate_encodes`, bounded SAT/MaxSAT portfolio jobs with `gpu_portfolio_*` counters, and UNKNOWN/TIMEOUT status propagation counters. `GpuSolverProductionTrace::require_production_metric_eligibility` rejects traces without accepted GPU candidate evidence or with CPU assignment, MaxSAT, or learned-clause transfer counters. | Full generalized MaxSAT scheduler coverage is still missing beyond the bounded UNSAT-prune search and weighted selection encoding fixtures. |
| Probability | Existing `xlog-prob` GPU exact/provenance/PIR/CNF paths | `EpistemicProbProductionAdapter` calls `compile_source_with_gpu_execution_result`, `compile_program_with_gpu_execution_result`, `compile_and_evaluate_source_with_gpu_execution_result`, `compile_and_evaluate_source_for_gpu_execution_results`, `compile_and_evaluate_conditioned_source_with_gpu_execution_result`, `compile_and_evaluate_conditioned_source_for_gpu_execution_results`, `compile_and_evaluate_conditioned_program_with_gpu_execution_result`, `compile_and_evaluate_conditioned_program_for_gpu_execution_results`, `compile_and_evaluate_conditioned_source_with_grads_with_gpu_execution_result`, `compile_and_evaluate_conditioned_source_with_grads_for_gpu_execution_results`, `compile_and_evaluate_conditioned_program_with_grads_with_gpu_execution_result`, `compile_and_evaluate_conditioned_program_with_grads_for_gpu_execution_results`, `compile_and_evaluate_program_with_gpu_execution_result`, `encode_source_pir_cnf_with_gpu_execution_result`, `encode_program_pir_cnf_with_gpu_execution_result`, `evaluate_with_gpu_execution_result`, and `evaluate_gpu_with_grads_with_gpu_execution_result`, validates accepted `EpistemicGpuExecutionResult` evidence through `AcceptedWorldViewEvidence::from_gpu_execution_result` and `EpistemicProbGpuExecutionEvidence`, then calls `ExactDdnnfProgram::compile_source_with_gpu`, `ExactDdnnfProgram::compile_from_program`, `GpuPirGraph::from_host`, `GpuPirRoots::from_host`, `encode_cnf_gpu`, `ExactDdnnfProgram::evaluate`, and `evaluate_gpu_with_grads`, with `gpu_pir_graph_uploads`, `gpu_cnf_encodes`, aggregate/source/program knowledge-compilation counters, accepted-assumption/evidence-fact counters, `gpu_conditioned_negative_evidence_facts`, conditioned-gradient counters, and zero CPU-only probability recomputation counters. Conditioned source and parsed-program evaluation appends parsed `Evidence` AST entries for zero-arity and concrete nonzero-arity tuple assumptions, preserving false `not know` assumptions as false exact evidence before calling the existing exact path, and the source/program query and gradient batch adapters validate all accepted world-view boundaries before per-record compile/evaluate with different accepted evidence per result. `EpistemicProbProductionCapabilities` disallows fixture circuits for production metrics, and `EpistemicProbProductionTrace::require_production_metric_eligibility` rejects traces without accepted world-view evidence, GPU exact/provenance/PIR/CNF counters, or zero CPU/fixture recomputation counters. | Broader probabilistic semantic coverage remains incomplete. |

## Forbidden Parallel Engines

The accepted paths audited above must not introduce epistemic-only replacements
for existing engines. The source audit checks for the absence of new accepted
path declarations or calls for:

- epistemic-only WCOJ planners or dispatch engines;
- epistemic-only relation or tuple stores;
- epistemic-only solver-search engines;
- epistemic-only probability-inference engines;
- CPU oracle solver or bounded fixture probability calls on accepted paths.

This is the specific `M090_CERT.13` claim: zero new epistemic-only WCOJ,
solver-search, probability-inference, or tuple-store engines are present in the
accepted paths audited by the source test.

## Validation

| Command | Expected result |
|---|---|
| `cargo test -p xlog-runtime --test test_epistemic_production_reuse_audit -- --nocapture` | PASS after this artifact exists and accepted-path source markers are present. |
| `cargo fmt --check` | PASS |
| `git diff --check` | PASS |

## Non-Closure Notes

- This audit strengthens `M090_CERT.13`; it does not close `G090_CERT`.
- `M090_GPU.11`, `M090_SOLVER.9`, `M090_PROB.8`, and broader accepted
  execution traces remain incomplete.
- CPU semantic-oracle fixtures remain scaffolding evidence only and cannot
  satisfy release gates.
- No closure-board edit, merge, push, or tag is implied.
