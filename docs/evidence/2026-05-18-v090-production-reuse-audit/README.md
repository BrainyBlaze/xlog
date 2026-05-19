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
| Runtime/WCOJ lowering | Existing logic compiler and runtime RIR execution | `compile_epistemic_gpu_execution` lowers reduced ordinary programs through `compile_program_with_stats_snapshot`; `compile_epistemic_gpu_split_execution` lowers valid split components through the same executable path; runtime calls `self.execute_plan(&executable.reduced_runtime_plan)`, records `EpistemicGpuRuntimePreflight`, certifies one accepted K5 WCOJ dispatch from production counter deltas while requiring sorted-layout and helper-split preflight metadata, records post-hot-path final-result transfer accounting through `EpistemicGpuFinalResultTransferTrace`, and adds K7/K8 preflight coverage through the G39 W6.4 K-clique template surface. | Broader accepted WCOJ runtime dispatch and semantic parity coverage are still missing. |
| WCOJ layout and scheduling | Existing WCOJ, K-clique, planned-hash, sorted-layout, and helper-split route metadata | `summarize_runtime_routes` inspects `MultiwayPlan::WcojWithPlan`, counts `kclique_wcoj_max_arity`, `kclique_wcoj_edge_permutation_count`, `helper_split_spec_count`, `planned_hash_route_count`, sorted-layout requirements, and runtime WCOJ dispatch deltas; the accepted K5 fixture supplies a passing launch trace, and `epistemic_k7_k8_reductions_reuse_g39_kclique_planner_preflight_surface` proves K7/K8 epistemic reductions carry full production edge-permutation metadata through preflight. | K7/K8 runtime dispatch counters and broader helper/skew runtime coverage remain missing. |
| Tuple membership | Existing relation layouts and device buffers | Membership staging reads existing `CudaBuffer` relation columns via `source_relation.column(...)` and compares against reduced output columns via `output.column(bound_col_index)`; final row filtering flattens per-membership row-count pointer/key-offset metadata over the same relation and reduced-output device buffers; accepted unary, binary, and multi-membership fixtures filter final rows by tuple key on device. | Broader semantic parity fixtures for nonzero-arity accepted world views are still missing. |
| Solver | Existing `xlog-solve` GPU CNF/CDCL production path | `GpuSolverProductionAdapter` wraps `GpuCdclSolver::new`, validates accepted `EpistemicGpuExecutionResult` evidence through `solve_expect_sat_with_gpu_execution_result`, `solve_expect_unsat_with_gpu_execution_result`, `solve_expect_unsat_with_branch_limit_ws_with_gpu_execution_result`, `solve_assumption_lifecycle_with_gpu_execution_result`, `solve_unsat_and_publish_learned_clause_arena_with_gpu_execution_result`, `solve_unsat_then_reuse_learned_clauses_with_gpu_execution_result`, `solve_weighted_maxsat_candidates_with_gpu_execution_result`, and `solve_portfolio_with_gpu_execution_result`, then calls existing SAT/UNSAT APIs, including reusable workspace-backed UNSAT, bounded lifecycle steps with `gpu_assumption_pushes`, `gpu_assumption_retractions`, and `gpu_lifecycle_workspace_reuses`, learned-clause arena publication plus same-device-CNF import/reuse with `gpu_learned_clause_*` counters, bounded MaxSAT candidate solves with `gpu_maxsat_candidate_solves`, bounded SAT/MaxSAT portfolio jobs with `gpu_portfolio_*` counters, and UNKNOWN/TIMEOUT status propagation counters, while exposing zero CPU assignment, learned-clause transfer, and MaxSAT enumeration counters. | Broader distinct-candidate learned-clause validity is still missing. |
| Probability | Existing `xlog-prob` GPU exact/provenance/PIR/CNF paths | `EpistemicProbProductionAdapter` calls `compile_source_with_gpu_execution_result`, `compile_program_with_gpu_execution_result`, `compile_and_evaluate_source_with_gpu_execution_result`, `compile_and_evaluate_program_with_gpu_execution_result`, `encode_source_pir_cnf_with_gpu_execution_result`, `encode_program_pir_cnf_with_gpu_execution_result`, `evaluate_with_gpu_execution_result`, and `evaluate_gpu_with_grads_with_gpu_execution_result`, validates accepted `EpistemicGpuExecutionResult` evidence through `AcceptedWorldViewEvidence::from_gpu_execution_result`, then calls `ExactDdnnfProgram::compile_source_with_gpu`, `ExactDdnnfProgram::compile_from_program`, `GpuPirGraph::from_host`, `GpuPirRoots::from_host`, `encode_cnf_gpu`, `ExactDdnnfProgram::evaluate`, and `evaluate_gpu_with_grads`, with `gpu_pir_graph_uploads`, `gpu_cnf_encodes`, aggregate/source/program knowledge-compilation counters, and zero CPU-only probability recomputation counters. | Broader probabilistic knowledge-compilation coverage over accepted runtime world views is still missing. |

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
