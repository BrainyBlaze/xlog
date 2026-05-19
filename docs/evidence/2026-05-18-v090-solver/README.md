# v0.9.0 G090_SOLVER Semantic And Production-Reuse Evidence

Date: 2026-05-18

Goal node: `G090_SOLVER - Solver-Service Integration`

Branch: `feat/v090-epistemic-solver-semantics`

## Implementation Summary

The current branch contains two solver layers:

- `SolverService`, a bounded CPU fixture facade for semantic-oracle tests. It is
  useful for assumption lifecycle, learned-clause transfer, MaxSAT scoring, and
  failure-mode oracle tests.
- `GpuSolverProductionAdapter`, a thin production-path adapter over the existing
  `GpuCdclSolver` SAT/UNSAT verifier. It provides source-level evidence that
  epistemic-facing SAT/UNSAT, bounded MaxSAT candidate, and SAT/MaxSAT
  portfolio work can route through existing GPU CDCL APIs without using the CPU
  fixture service. Its accepted-runtime SAT/UNSAT and workspace-backed UNSAT
  gates, bounded push/solve/retract lifecycle gate, bounded MaxSAT candidate
  gate, two-record multi-candidate MaxSAT gate, single-result and two-record
  MaxSAT search-pruning gates, single-result and two-record weighted MaxSAT
  encoding/search gates, generalized heterogeneous MaxSAT scheduler gate,
  bounded single-result and two-record status-aware portfolio gates,
  learned-clause arena publication gate, two-record multi-candidate
  push/solve/retract lifecycle gate, mixed `possible`/`not possible`
  operator-result lifecycle gate, two-record multi-candidate learned-clause
  reuse gate, same-device-CNF
  learned-clause reuse gate, and distinct-CNF
  learned-clause import rejection guard consume accepted
  `EpistemicGpuExecutionResult` before dispatching to GPU CDCL or rejecting
  unsafe reuse.
- `production_capabilities`, a source-level capability report that marks GPU
  CDCL SAT/UNSAT plus bounded GPU-backed MaxSAT and SAT/MaxSAT portfolio
  adapters available while keeping the CPU oracle disallowed for production
  metrics.

This remains partial evidence. The branch now proves same-device-CNF
learned-clause import/reuse and records a distinct-CNF rejection reason without
CPU learned-clause transfers, and it proves a two accepted-result lifecycle with
balanced GPU push/retract counters plus bounded UNKNOWN/TIMEOUT lifecycle
propagation plus two-record learned-clause publication/import reuse, a mixed
operator-result lifecycle path, two-record/two-CNF MaxSAT candidate-set
execution, and GPU-CDCL UNSAT pruning inside a bounded MaxSAT search candidate
set plus single-result and two-record
weighted soft-clause selection encoding into GPU CNF candidates plus a
two-record heterogeneous MaxSAT scheduler over candidate-set, search-prune,
encoded-search, UNKNOWN, and TIMEOUT jobs plus two-record status-aware
portfolio dispatch over the existing GPU CDCL adapter, but full solver release
closure remains blocked by broader semantic integration and post-v0.8
certification.

| Requirement | Evidence |
|---|---|
| Solver service interface | `SolverService` exposes bounded SAT/MaxSAT solves, assumptions, retraction, learned-clause transfer, trace, and GPU portfolio status. |
| GPU production adapter | `GpuSolverProductionAdapter` exports SAT/UNSAT calls over `GpuCdclSolver`, including workspace-backed UNSAT reuse, learned-clause arena publication, bounded same-device-CNF learned-clause import/reuse, two-record multi-candidate learned-clause reuse, distinct-CNF learned-clause import rejection, bounded MaxSAT candidate solving, two-record multi-candidate MaxSAT solving, single-result and two-record MaxSAT search pruning of UNSAT candidates, single-result and two-record weighted soft-clause selection encoding into GPU CNF search candidates, two-record heterogeneous MaxSAT scheduling over candidate-set/search/encoded-search/status jobs, single-result and two-record SAT/MaxSAT portfolio dispatch, and UNKNOWN/TIMEOUT portfolio status propagation. |
| Accepted runtime SAT/UNSAT gates | `solve_expect_sat_with_gpu_execution_result`, `solve_expect_unsat_with_gpu_execution_result`, and `solve_expect_unsat_with_branch_limit_ws_with_gpu_execution_result` validate stable tuple-source membership, GPU kernel traces, zero hot-path transfers, and non-empty final device output before calling `GpuCdclSolver` SAT/UNSAT paths, including reusable workspace-backed UNSAT. |
| Accepted runtime lifecycle gate | `solve_assumption_lifecycle_with_gpu_execution_result` validates accepted GPU runtime evidence, then records GPU assumption pushes/retractions while dispatching SAT and UNSAT lifecycle steps through existing GPU CDCL calls and the provided reusable workspace. It also propagates bounded UNKNOWN/TIMEOUT lifecycle statuses with diagnostic/budget checks and zero CPU search counters. |
| Accepted runtime multi-candidate lifecycle gate | `solve_multi_candidate_assumption_lifecycle_with_gpu_execution_results` validates two accepted GPU runtime evidence records, runs the same SAT/UNSAT lifecycle for each through existing GPU CDCL calls, reports `candidate_evidence_records == 2`, records balanced push/retract counters, and reuses the provided GPU CDCL workspace for UNSAT steps. `accepted_operator_gpu_execution_results_gate_solver_lifecycle_path` proves the same lifecycle gate consumes mixed `possible` and `not possible` accepted operator results with zero CPU search counters. |
| Accepted runtime learned-clause arena gate | `solve_unsat_and_publish_learned_clause_arena_with_gpu_execution_result` validates accepted GPU runtime evidence, then runs workspace-backed GPU CDCL UNSAT and publishes the existing device learned-clause/proof arena plus learned-count buffer with zero CPU learned-clause transfers. |
| Accepted runtime learned-clause reuse gate | `solve_unsat_then_reuse_learned_clauses_with_gpu_execution_result` validates accepted GPU runtime evidence, publishes the existing GPU CDCL learned-clause/proof arena from one UNSAT solve, then imports the same device arena into a second workspace-backed UNSAT solve over the same GPU CNF with zero CPU learned-clause transfers; distinct candidate CNFs are rejected before import, incrementing `gpu_learned_clause_reuse_rejections` with zero CPU learned-clause transfers. |
| Accepted runtime multi-candidate learned-clause gate | `solve_multi_candidate_learned_clause_reuse_with_gpu_execution_results` validates two accepted GPU runtime evidence records up front, then performs same-device-CNF learned-clause publication/import once per accepted result through the existing GPU CDCL workspace, recording `candidate_evidence_records == 2`, four workspace-backed UNSAT solves, two arena publications, two imports, two reused solves, and zero CPU learned-clause transfers. |
| Accepted runtime MaxSAT gate | `solve_weighted_maxsat_candidates_with_gpu_execution_result` validates accepted GPU runtime evidence, then certifies bounded weighted MaxSAT candidate CNFs through `GpuCdclSolver::solve_expect_sat_with_branch_limit` and records `candidate_evidence_records`, `gpu_maxsat_candidate_solves`, and `gpu_maxsat_optima` with zero CPU MaxSAT enumerations. |
| Accepted runtime multi-candidate MaxSAT gate | `solve_multi_candidate_weighted_maxsat_with_gpu_execution_results` validates two accepted GPU runtime evidence records up front, then certifies a two-CNF weighted MaxSAT candidate set once per accepted result through the existing GPU CDCL path, recording `candidate_evidence_records == 2`, `candidates_checked == 4`, four GPU CDCL candidate solves, two optima, and zero CPU MaxSAT enumerations. |
| Accepted runtime MaxSAT search-pruning gate | `solve_weighted_maxsat_search_with_gpu_execution_result` validates accepted GPU runtime evidence, then scores satisfiable candidates through `GpuCdclSolver::solve_expect_sat_with_branch_limit` and prunes UNSAT candidates through workspace-backed `solve_expect_unsat_with_branch_limit_ws`, recording `satisfiable_candidates`, `unsat_candidates_pruned`, `gpu_maxsat_unsat_candidate_prunes`, and zero CPU MaxSAT enumerations. |
| Accepted runtime multi-candidate MaxSAT search-pruning gate | `solve_multi_candidate_weighted_maxsat_search_with_gpu_execution_results` validates two accepted GPU runtime evidence records up front, then repeats bounded MaxSAT search pruning once per accepted result through the existing GPU CDCL SAT and workspace-backed UNSAT paths, recording `candidate_evidence_records == 2`, four candidate solves, two UNSAT prunes, two optima, and zero CPU MaxSAT enumerations. |
| Accepted runtime weighted MaxSAT encoding gate | `solve_weighted_maxsat_encoded_search_with_gpu_execution_result` validates accepted GPU runtime evidence, converts caller-declared weighted soft-clause selections into satisfaction CNFs, uploads them through the existing GPU CNF layout, then reuses bounded MaxSAT search pruning through GPU CDCL while recording `gpu_maxsat_candidate_encodes`, `gpu_cdcl_candidate_encodes`, candidate solves, UNSAT prunes, and zero CPU MaxSAT enumerations. |
| Accepted runtime multi-candidate weighted MaxSAT encoding gate | `solve_multi_candidate_weighted_maxsat_encoded_search_with_gpu_execution_results` validates two accepted GPU runtime evidence records up front, converts caller-declared weighted soft-clause selections into satisfaction CNFs once per accepted result through the existing GPU CNF layout, then reuses bounded MaxSAT search pruning through GPU CDCL while recording `candidate_evidence_records == 2`, four GPU CNF candidate encodes, four candidate solves, two UNSAT prunes, two optima, and zero CPU MaxSAT enumerations. |
| Accepted runtime MaxSAT scheduler gate | `solve_maxsat_schedule_with_gpu_execution_results` validates two accepted GPU runtime evidence records up front, then dispatches a heterogeneous schedule of candidate-set, search-pruning, weighted encoded-search, UNKNOWN, and TIMEOUT jobs through the existing GPU CNF/CDCL helpers, recording `candidate_evidence_records == 2`, ten scheduled jobs, twelve GPU CDCL candidate solves, four GPU CNF candidate encodes, four UNSAT prunes, six optima, and zero CPU MaxSAT enumerations. |
| Accepted runtime portfolio gate | `solve_portfolio_with_gpu_execution_result` validates accepted GPU runtime evidence, then dispatches SAT jobs and bounded MaxSAT candidate jobs through the same GPU CDCL adapter while propagating UNKNOWN/TIMEOUT portfolio statuses without CPU search and recording `gpu_portfolio_*` counters. `solve_multi_candidate_portfolio_with_gpu_execution_results` validates two accepted GPU runtime evidence records up front, repeats the same status-aware portfolio once per record, reports `candidate_evidence_records == 2`, and records doubled GPU portfolio counters with zero CPU search. |
| Production capability report | `production_capabilities` reports GPU CDCL SAT/UNSAT, bounded MaxSAT, and bounded portfolio adapters available, with CPU oracle disallowed. |
| CPU solver-search isolation | `GpuSolverProductionTrace` records zero CPU assignment, MaxSAT, and learned-clause transfer counters plus GPU learned-clause, rejection, MaxSAT, and portfolio counters; the production adapter source test rejects `SolverService` use. |
| Incremental SAT assumptions | `solver_service_semantics.rs` verifies assumption add/retract changes SAT status without stale learned contradictions. |
| Learned-clause transfer | `transfer_learned_clauses_to` transfers scoped learned clauses and increments `SolverServiceTrace::learned_clause_transfers`. |
| MaxSAT soft constraints | `SolveInstance::with_weights` fixture returns `SolverServiceStatus::Optimal(5)`. |
| GPU portfolio status | `gpu_portfolio_status` reports that GPU portfolio solving is not implemented in the semantic-oracle facade. |
| Failure modes | Fixtures distinguish `Unsat`, `Unknown`, and `Timeout`. |

## Validation

| Command | Result |
|---|---|
| `cargo fmt --check` | PASS |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_gates_solver_cdcl_sat_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_gates_solver_cdcl_unsat_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_gates_solver_workspace_unsat_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_gates_solver_assumption_lifecycle_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_gates_status_aware_solver_lifecycle_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_results_gate_multi_candidate_solver_lifecycle_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_operator_gpu_execution_results_gate_solver_lifecycle_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_gates_solver_learned_clause_arena_publication -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_gates_solver_same_cnf_learned_clause_reuse -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_results_gate_multi_candidate_learned_clause_reuse -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_rejects_distinct_cnf_learned_clause_reuse -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_gates_solver_maxsat_and_portfolio_paths -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_results_gate_multi_candidate_maxsat_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_prunes_unsat_maxsat_search_candidates -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_results_gate_multi_candidate_maxsat_search_pruning -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_encodes_weighted_maxsat_search_candidates -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_results_gate_multi_candidate_weighted_maxsat_encoded_search -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_results_gate_generalized_maxsat_scheduler -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_results_gate_multi_candidate_portfolio_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-solve --test gpu_solver_production_reuse` | PASS, 3 passed, 0 failed |
| `cargo test -p xlog-solve --test gpu_solver_production_reuse production_solver_metric_gate_rejects_cpu_oracle_only_traces -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-solve --test solver_service_semantics` | PASS, 5 passed, 0 failed |
| `cargo test -p xlog-solve --test no_dtoh_in_gpu_cdcl` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-solve --lib` | PASS, 111 passed, 0 failed |
| `cargo check -p xlog-logic -p xlog-ir -p xlog-solve -p xlog-prob` | PASS |
| `cargo check -p pyxlog` | PASS |

## Metric Status

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M090_SOLVER.1 interface | trait/API documented and tested | PARTIAL | CPU facade API exists, and `GpuSolverProductionAdapter` exposes GPU CDCL SAT/UNSAT, accepted-runtime SAT/UNSAT and workspace-backed UNSAT gates, bounded single- and multi-candidate accepted lifecycle APIs, bounded single- and multi-candidate MaxSAT candidate APIs, single-result and two-record MaxSAT search-pruning APIs, single-result and two-record weighted MaxSAT encoding/search APIs, heterogeneous MaxSAT scheduler APIs, and single-result plus two-record SAT/MaxSAT/UNKNOWN/TIMEOUT portfolio APIs; broader solver coverage is still missing. |
| M090_SOLVER.2 incremental SAT | add/retract assumption fixtures pass on GPU-native path | PARTIAL | CPU oracle fixture exists, and accepted GPU runtime evidence can gate single- and two-record SAT/UNSAT push/solve/retract sequences through `GpuCdclSolver`, including mixed `possible`/`not possible` accepted operator results, plus bounded UNKNOWN/TIMEOUT lifecycle propagation; broader incremental epistemic candidate assumptions are not fully wired. |
| M090_SOLVER.3 learned clauses | transfer observable in GPU trace or test double | PARTIAL | CPU test double observes transfer, and the accepted production adapter now publishes GPU CDCL learned-clause/proof arenas plus learned-count device buffers after accepted runtime evidence, imports the same device arena into a second same-GPU-CNF UNSAT solve with zero CPU learned-clause transfers, repeats that publication/import path for two accepted GPU evidence records, and rejects distinct candidate CNF imports with `gpu_learned_clause_reuse_rejections`. Broader learned-clause coverage beyond same-device-CNF reuse is still missing. |
| M090_SOLVER.4 MaxSAT | soft-constraint fixture returns expected optimum on GPU-native path | PARTIAL | CPU oracle fixture exists; accepted GPU runtime evidence now gates bounded weighted MaxSAT candidate fixtures through `GpuCdclSolver::solve_expect_sat_with_branch_limit`, returns optimum scores from single-result and two-record/two-CNF candidate-set paths, prunes UNSAT candidates through workspace-backed GPU CDCL UNSAT in the single-result and two-record bounded search paths, encodes weighted soft-clause selections into GPU CNF search candidates for one accepted result and two accepted results, and dispatches a two-record heterogeneous MaxSAT schedule with candidate-set, search-pruning, encoded-search, UNKNOWN, and TIMEOUT jobs while recording zero CPU MaxSAT enumerations. Broader MaxSAT semantic integration remains missing. |
| M090_SOLVER.5 GPU portfolio | portfolio dispatch executes on GPU or GPU-backed adapter with measured launch evidence | PARTIAL | Accepted GPU runtime evidence now gates single-result and two-record bounded SAT/MaxSAT/UNKNOWN/TIMEOUT portfolios through `GpuCdclSolver` plus status propagation, and lifecycle-level UNKNOWN/TIMEOUT propagation records dedicated counters while keeping CPU search counters at zero. Broader portfolio scheduling coverage is still missing. |
| M090_SOLVER.6 failure modes | UNSAT/UNKNOWN/TIMEOUT represented distinctly | PASS for oracle | CPU oracle fixtures distinguish the states. |
| M090_SOLVER.7 assumption lifecycle | push, solve, retract, and reuse trace proves no assumption leak between candidates | PARTIAL | CPU lifecycle fixture exists; accepted GPU candidate evidence now gates one-record and two-record SAT/UNSAT push/solve/retract sequences, records `candidate_evidence_records`, balanced pushes/retractions, workspace reuse, mixed `possible`/`not possible` operator-result lifecycle evidence, bounded UNKNOWN/TIMEOUT lifecycle status counters, learned-clause arena publication, bounded same-device-CNF learned-clause import/reuse, two-record learned-clause reuse, distinct-CNF learned-clause import rejection, and two-record MaxSAT candidate-set solving. Full MaxSAT lifecycle coverage is still missing. |
| M090_SOLVER.8 CPU search ban | accepted solver path records zero CPU exhaustive assignment enumeration | PARTIAL | Accepted runtime SAT/UNSAT, workspace-backed UNSAT, single- and multi-candidate bounded lifecycle, mixed `possible`/`not possible` operator-result lifecycle, lifecycle UNKNOWN/TIMEOUT propagation, learned-clause arena, same-device-CNF learned-clause reuse, two-record learned-clause reuse, distinct-CNF learned-clause rejection, bounded single- and multi-candidate MaxSAT, single-result and two-record MaxSAT search pruning, single-result and two-record weighted MaxSAT encoding/search, heterogeneous MaxSAT scheduling, and single-result plus two-record bounded portfolio gates route GPU evidence into GPU CDCL or a fail-closed/status propagation path and record zero CPU assignment/MaxSAT enumeration and learned-clause transfer counters; broader solver coverage remains missing. |
| M090_SOLVER.9 production solver reuse | accepted SAT/MaxSAT fixtures execute through existing GPU CNF/CDCL/solver production APIs or thin adapters over them | PARTIAL | Accepted SAT/UNSAT/lifecycle/status-aware-lifecycle/multi-candidate-lifecycle/operator-result-lifecycle/learned-arena/reuse/multi-candidate-learned-reuse/rejection/MaxSAT/multi-candidate-MaxSAT/MaxSAT-search-prune/multi-candidate-MaxSAT-search-prune/weighted-MaxSAT-encoding/multi-candidate-weighted-MaxSAT-encoding/MaxSAT-scheduler/single-result-portfolio/two-record-portfolio fixtures call `GpuCdclSolver::new`, `GpuCnf::from_host`, `solve_expect_sat`, `solve_expect_sat_with_branch_limit`, `solve_expect_unsat`, `solve_expect_unsat_with_branch_limit_ws`, and `solve_expect_unsat_with_branch_limit_ws_importing_learned`; accepted lifecycle, scheduler, and portfolio fixtures also propagate UNKNOWN/TIMEOUT without CPU search. Broader solver release coverage remains missing. |
| M090_SOLVER.10 fixture isolation | CPU semantic-oracle solver facade is gated so it cannot satisfy closure metrics | PARTIAL | Evidence docs mark `SolverService` as oracle-only, the production adapter source test rejects `SolverService`, `production_capabilities` disallows the CPU oracle for production metrics, and `GpuSolverProductionTrace::require_production_metric_eligibility` rejects traces without accepted GPU candidate evidence or with CPU search counters. Distinct-CNF reuse rejection also records zero CPU learned-clause transfers. Broader solver coverage is still missing, so this is not a G090_SOLVER close. |

## Coordination Notes

- This file is not release-close evidence for `G090_SOLVER`.
- The production adapter is partial accepted-runtime SAT, UNSAT,
  workspace-backed UNSAT, bounded lifecycle, mixed operator-result lifecycle,
  learned-clause arena publication,
  same-device-CNF learned-clause import/reuse, two-record learned-clause reuse,
  distinct-CNF learned-clause import rejection, bounded single- and
  multi-candidate MaxSAT, single-result and two-record MaxSAT search pruning,
  single-result and two-record weighted MaxSAT encoding/search, heterogeneous
  MaxSAT scheduler, and single-result plus two-record status-aware portfolio
  reuse evidence only.
- Broader solver release coverage and post-v0.8 compatibility certification
  remain required before v0.9.0 can close.
- No pyxlog public API signatures were changed.
- No push, tag, release-board update, or merge was performed.
