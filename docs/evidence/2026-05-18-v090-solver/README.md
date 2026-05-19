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
  gate, bounded status-aware portfolio gate, learned-clause arena publication
  gate, and same-device-CNF learned-clause reuse gate consume an accepted
  `EpistemicGpuExecutionResult` before dispatching to GPU CDCL.
- `production_capabilities`, a source-level capability report that marks GPU
  CDCL SAT/UNSAT plus bounded GPU-backed MaxSAT and SAT/MaxSAT portfolio
  adapters available while keeping the CPU oracle disallowed for production
  metrics.

This remains partial evidence. The branch still lacks broader multi-candidate
assumption lifecycle traces that prove learned-clause validity across distinct
candidate CNFs, complete status-aware SAT/UNSAT/UNKNOWN/TIMEOUT lifecycle
coverage beyond the bounded portfolio adapter, and full solver release closure.

| Requirement | Evidence |
|---|---|
| Solver service interface | `SolverService` exposes bounded SAT/MaxSAT solves, assumptions, retraction, learned-clause transfer, trace, and GPU portfolio status. |
| GPU production adapter | `GpuSolverProductionAdapter` exports SAT/UNSAT calls over `GpuCdclSolver`, including workspace-backed UNSAT reuse, learned-clause arena publication, bounded same-device-CNF learned-clause import/reuse, bounded MaxSAT candidate solving, bounded SAT/MaxSAT portfolio dispatch, and UNKNOWN/TIMEOUT portfolio status propagation. |
| Accepted runtime SAT/UNSAT gates | `solve_expect_sat_with_gpu_execution_result`, `solve_expect_unsat_with_gpu_execution_result`, and `solve_expect_unsat_with_branch_limit_ws_with_gpu_execution_result` validate stable tuple-source membership, GPU kernel traces, zero hot-path transfers, and non-empty final device output before calling `GpuCdclSolver` SAT/UNSAT paths, including reusable workspace-backed UNSAT. |
| Accepted runtime lifecycle gate | `solve_assumption_lifecycle_with_gpu_execution_result` validates accepted GPU runtime evidence, then records GPU assumption pushes/retractions while dispatching SAT and UNSAT lifecycle steps through existing GPU CDCL calls and the provided reusable workspace. |
| Accepted runtime learned-clause arena gate | `solve_unsat_and_publish_learned_clause_arena_with_gpu_execution_result` validates accepted GPU runtime evidence, then runs workspace-backed GPU CDCL UNSAT and publishes the existing device learned-clause/proof arena plus learned-count buffer with zero CPU learned-clause transfers. |
| Accepted runtime learned-clause reuse gate | `solve_unsat_then_reuse_learned_clauses_with_gpu_execution_result` validates accepted GPU runtime evidence, publishes the existing GPU CDCL learned-clause/proof arena from one UNSAT solve, then imports the same device arena into a second workspace-backed UNSAT solve over the same GPU CNF with zero CPU learned-clause transfers. |
| Accepted runtime MaxSAT gate | `solve_weighted_maxsat_candidates_with_gpu_execution_result` validates accepted GPU runtime evidence, then certifies bounded weighted MaxSAT candidate CNFs through `GpuCdclSolver::solve_expect_sat_with_branch_limit` and records `gpu_maxsat_candidate_solves`/`gpu_maxsat_optima` with zero CPU MaxSAT enumerations. |
| Accepted runtime portfolio gate | `solve_portfolio_with_gpu_execution_result` validates accepted GPU runtime evidence, then dispatches SAT jobs and bounded MaxSAT candidate jobs through the same GPU CDCL adapter while propagating UNKNOWN/TIMEOUT portfolio statuses without CPU search and recording `gpu_portfolio_*` counters. |
| Production capability report | `production_capabilities` reports GPU CDCL SAT/UNSAT, bounded MaxSAT, and bounded portfolio adapters available, with CPU oracle disallowed. |
| CPU solver-search isolation | `GpuSolverProductionTrace` records zero CPU assignment, MaxSAT, and learned-clause transfer counters plus GPU learned-clause, MaxSAT, and portfolio counters; the production adapter source test rejects `SolverService` use. |
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
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_gates_solver_learned_clause_arena_publication -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_gates_solver_same_cnf_learned_clause_reuse -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_gates_solver_maxsat_and_portfolio_paths -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-solve --test gpu_solver_production_reuse` | PASS, 2 passed, 0 failed |
| `cargo test -p xlog-solve --test solver_service_semantics` | PASS, 5 passed, 0 failed |
| `cargo test -p xlog-solve --test no_dtoh_in_gpu_cdcl` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-solve --lib` | PASS, 111 passed, 0 failed |
| `cargo check -p xlog-logic -p xlog-ir -p xlog-solve -p xlog-prob` | PASS |
| `cargo check -p pyxlog` | PASS |

## Metric Status

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M090_SOLVER.1 interface | trait/API documented and tested | PARTIAL | CPU facade API exists, and `GpuSolverProductionAdapter` exposes GPU CDCL SAT/UNSAT, accepted-runtime SAT/UNSAT and workspace-backed UNSAT gates, bounded accepted lifecycle API, bounded MaxSAT candidate API, and bounded SAT/MaxSAT/UNKNOWN/TIMEOUT portfolio API; broader candidate lifecycle APIs are still missing. |
| M090_SOLVER.2 incremental SAT | add/retract assumption fixtures pass on GPU-native path | PARTIAL | CPU oracle fixture exists, and accepted GPU runtime evidence can gate a SAT/UNSAT push/solve/retract sequence through `GpuCdclSolver`; broader incremental epistemic candidate assumptions are not fully wired. |
| M090_SOLVER.3 learned clauses | transfer observable in GPU trace or test double | PARTIAL | CPU test double observes transfer, and the accepted production adapter now publishes GPU CDCL learned-clause/proof arenas plus learned-count device buffers after accepted runtime evidence, then imports the same device arena into a second same-GPU-CNF UNSAT solve with zero CPU learned-clause transfers; broader distinct-candidate learned-clause validity semantics are missing. |
| M090_SOLVER.4 MaxSAT | soft-constraint fixture returns expected optimum on GPU-native path | PARTIAL | CPU oracle fixture exists; accepted GPU runtime evidence now gates a bounded weighted MaxSAT candidate fixture through `GpuCdclSolver::solve_expect_sat_with_branch_limit`, returns optimum score `5`, and records zero CPU MaxSAT enumerations. Broader MaxSAT encoding/search coverage is still missing. |
| M090_SOLVER.5 GPU portfolio | portfolio dispatch executes on GPU or GPU-backed adapter with measured launch evidence | PARTIAL | Accepted GPU runtime evidence now gates a bounded SAT/MaxSAT/UNKNOWN/TIMEOUT portfolio through `GpuCdclSolver` plus status propagation, records SAT, MaxSAT, UNKNOWN, and TIMEOUT portfolio counters, and keeps CPU search counters at zero. Broader portfolio scheduling coverage is still missing. |
| M090_SOLVER.6 failure modes | UNSAT/UNKNOWN/TIMEOUT represented distinctly | PASS for oracle | CPU oracle fixtures distinguish the states. |
| M090_SOLVER.7 assumption lifecycle | push, solve, retract, and reuse trace proves no assumption leak between candidates | PARTIAL | CPU lifecycle fixture exists; accepted GPU candidate evidence now gates one SAT/UNSAT push/solve/retract sequence and records balanced pushes/retractions, workspace reuse, learned-clause arena publication, and bounded same-device-CNF learned-clause import/reuse. Broader multi-candidate lifecycle traces are missing. |
| M090_SOLVER.8 CPU search ban | accepted solver path records zero CPU exhaustive assignment enumeration | PARTIAL | Accepted runtime SAT/UNSAT, workspace-backed UNSAT, bounded lifecycle, learned-clause arena, same-device-CNF learned-clause reuse, bounded MaxSAT, and bounded portfolio gates route GPU evidence into GPU CDCL and record zero CPU assignment/MaxSAT enumeration and learned-clause transfer counters; broader lifecycle traces are missing. |
| M090_SOLVER.9 production solver reuse | accepted SAT/MaxSAT fixtures execute through existing GPU CNF/CDCL/solver production APIs or thin adapters over them | PARTIAL | Accepted SAT/UNSAT/lifecycle/learned-arena/reuse/MaxSAT/portfolio fixtures call `GpuCdclSolver::new`, `solve_expect_sat`, `solve_expect_sat_with_branch_limit`, `solve_expect_unsat`, `solve_expect_unsat_with_branch_limit_ws`, and `solve_expect_unsat_with_branch_limit_ws_importing_learned`; the accepted portfolio fixture also propagates UNKNOWN/TIMEOUT without CPU search. Broader distinct-candidate learned-clause validity coverage remains missing. |
| M090_SOLVER.10 fixture isolation | CPU semantic-oracle solver facade is gated so it cannot satisfy closure metrics | PARTIAL | Evidence docs mark `SolverService` as oracle-only, the production adapter source test rejects `SolverService`, and `production_capabilities` disallows the CPU oracle for production metrics; accepted-path closure automation is still missing. |

## Coordination Notes

- This file is not release-close evidence for `G090_SOLVER`.
- The production adapter is partial accepted-runtime SAT, UNSAT,
  workspace-backed UNSAT, bounded lifecycle, learned-clause arena publication,
  same-device-CNF learned-clause import/reuse, bounded MaxSAT, and bounded
  status-aware portfolio reuse evidence only.
- Broader multi-candidate lifecycle traces, distinct-candidate learned-clause
  validity coverage, and MaxSAT coverage remain required before v0.9.0 can
  close.
- No pyxlog public API signatures were changed.
- No push, tag, release-board update, or merge was performed.
