# v0.9.0 G090_SOLVER Semantic-Oracle Evidence

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
  epistemic-facing SAT/UNSAT work can route through existing GPU CDCL APIs
  without using the CPU fixture service. Its accepted-runtime SAT gate consumes
  an accepted `EpistemicGpuExecutionResult` before dispatching to GPU CDCL.
- `production_capabilities`, a source-level capability report that marks GPU
  CDCL SAT/UNSAT available and keeps GPU-native MaxSAT and SAT/MaxSAT portfolio
  execution blocked until existing production paths exist.

This remains partial evidence. The branch still lacks GPU-native MaxSAT,
portfolio solving, incremental assumption lifecycle traces over accepted
epistemic candidates, and full solver release closure.

| Requirement | Evidence |
|---|---|
| Solver service interface | `SolverService` exposes bounded SAT/MaxSAT solves, assumptions, retraction, learned-clause transfer, trace, and GPU portfolio status. |
| GPU production adapter | `GpuSolverProductionAdapter` exports SAT/UNSAT calls over `GpuCdclSolver`, including workspace-backed UNSAT reuse. |
| Accepted runtime SAT gate | `solve_expect_sat_with_gpu_execution_result` validates stable tuple-source membership, GPU kernel traces, zero hot-path transfers, and non-empty final device output before calling `solve_expect_sat`. |
| Production capability report | `production_capabilities` reports GPU CDCL SAT/UNSAT available, MaxSAT blocked, portfolio blocked, and CPU oracle disallowed. |
| CPU solver-search isolation | `GpuSolverProductionTrace` records zero CPU assignment and MaxSAT enumerations; the production adapter source test rejects `SolverService` use. |
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
| `cargo test -p xlog-solve --test gpu_solver_production_reuse` | PASS, 2 passed, 0 failed |
| `cargo test -p xlog-solve --test solver_service_semantics` | PASS, 5 passed, 0 failed |
| `cargo test -p xlog-solve --test no_dtoh_in_gpu_cdcl` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-solve --lib` | PASS, 111 passed, 0 failed |
| `cargo check -p xlog-logic -p xlog-ir -p xlog-solve -p xlog-prob` | PASS |
| `cargo check -p pyxlog` | PASS |

## Metric Status

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M090_SOLVER.1 interface | trait/API documented and tested | PARTIAL | CPU facade API exists, and `GpuSolverProductionAdapter` exposes GPU CDCL SAT/UNSAT plus an accepted-runtime SAT gate; full incremental accepted-candidate API is missing. |
| M090_SOLVER.2 incremental SAT | add/retract assumption fixtures pass on GPU-native path | PARTIAL | CPU oracle fixture exists, and accepted GPU runtime evidence can gate one SAT dispatch through `GpuCdclSolver`; add/retract incremental epistemic assumptions are not wired to the GPU path. |
| M090_SOLVER.3 learned clauses | transfer observable in GPU trace or test double | PARTIAL | CPU test double observes transfer, and the production adapter can allocate reusable GPU CDCL workspaces; learned transfer trace from epistemic candidates is missing. |
| M090_SOLVER.4 MaxSAT | soft-constraint fixture returns expected optimum on GPU-native path | BLOCKED | CPU oracle fixture exists; `production_capabilities` explicitly reports GPU-native MaxSAT production execution as blocked. |
| M090_SOLVER.5 GPU portfolio | portfolio dispatch executes on GPU or GPU-backed adapter with measured launch evidence | BLOCKED | `production_capabilities` explicitly reports GPU SAT/MaxSAT portfolio execution as blocked; no CPU fallback is allowed. |
| M090_SOLVER.6 failure modes | UNSAT/UNKNOWN/TIMEOUT represented distinctly | PASS for oracle | CPU oracle fixtures distinguish the states. |
| M090_SOLVER.7 assumption lifecycle | push, solve, retract, and reuse trace proves no assumption leak between candidates | PARTIAL | CPU lifecycle fixture exists, and accepted GPU candidate evidence is counted before SAT dispatch; push/retract/reuse traces across multiple accepted candidates are missing. |
| M090_SOLVER.8 CPU search ban | accepted solver path records zero CPU exhaustive assignment enumeration | PARTIAL | `solve_expect_sat_with_gpu_execution_result` routes accepted GPU runtime evidence into GPU CDCL SAT and records zero CPU assignment/MaxSAT enumeration counters; MaxSAT/portfolio and multi-candidate lifecycle traces are missing. |
| M090_SOLVER.9 production solver reuse | accepted SAT/MaxSAT fixtures execute through existing GPU CNF/CDCL/solver production APIs or thin adapters over them | PARTIAL | Accepted SAT fixture calls `GpuCdclSolver::new` and `solve_expect_sat`; UNSAT adapter calls existing `GpuCdclSolver` paths; capability report blocks MaxSAT and portfolio. |
| M090_SOLVER.10 fixture isolation | CPU semantic-oracle solver facade is gated so it cannot satisfy closure metrics | PARTIAL | Evidence docs mark `SolverService` as oracle-only, the production adapter source test rejects `SolverService`, and `production_capabilities` disallows the CPU oracle for production metrics; accepted-path closure automation is still missing. |

## Coordination Notes

- This file is not release-close evidence for `G090_SOLVER`.
- The production adapter is partial accepted-runtime SAT and UNSAT reuse
  evidence only.
- GPU-native MaxSAT, portfolio execution, and full solver lifecycle traces
  remain required before v0.9.0 can close.
- No pyxlog public API signatures were changed.
- No push, tag, release-board update, or merge was performed.
