# v0.9.0 G090_SOLVER Semantic-Oracle Evidence

Date: 2026-05-18

Goal node: `G090_SOLVER - Solver-Service Integration`

Branch: `feat/v090-epistemic-solver-semantics`

## Implementation Summary

The current branch contains a bounded CPU fixture facade for solver semantics.
It is useful for assumption lifecycle, learned-clause transfer, MaxSAT scoring,
and failure-mode oracle tests. It is not the GPU-native SAT/MaxSAT/portfolio
service required for v0.9.0 release closure.

| Requirement | Evidence |
|---|---|
| Solver service interface | `SolverService` exposes bounded SAT/MaxSAT solves, assumptions, retraction, learned-clause transfer, trace, and GPU portfolio status. |
| Incremental SAT assumptions | `solver_service_semantics.rs` verifies assumption add/retract changes SAT status without stale learned contradictions. |
| Learned-clause transfer | `transfer_learned_clauses_to` transfers scoped learned clauses and increments `SolverServiceTrace::learned_clause_transfers`. |
| MaxSAT soft constraints | `SolveInstance::with_weights` fixture returns `SolverServiceStatus::Optimal(5)`. |
| GPU portfolio status | `gpu_portfolio_status` reports that GPU portfolio solving is not implemented in the semantic-oracle facade. |
| Failure modes | Fixtures distinguish `Unsat`, `Unknown`, and `Timeout`. |

## Validation

| Command | Result |
|---|---|
| `cargo fmt --check` | PASS |
| `cargo test -p xlog-solve --test solver_service_semantics` | PASS, 5 passed, 0 failed |
| `cargo test -p xlog-solve --lib` | PASS, 111 passed, 0 failed |
| `cargo check -p xlog-logic -p xlog-ir -p xlog-solve -p xlog-prob` | PASS |
| `cargo check -p pyxlog` | PASS |

## Metric Status

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M090_SOLVER.1 interface | trait/API documented and tested | PARTIAL | CPU facade API exists; GPU-native interface to epistemic candidates is missing. |
| M090_SOLVER.2 incremental SAT | add/retract assumption fixtures pass on GPU-native path | PARTIAL | CPU oracle fixture exists; GPU-native path is missing. |
| M090_SOLVER.3 learned clauses | transfer observable in GPU trace or test double | PARTIAL | CPU test double observes transfer; GPU trace is missing. |
| M090_SOLVER.4 MaxSAT | soft-constraint fixture returns expected optimum on GPU-native path | PARTIAL | CPU oracle fixture exists; GPU-native MaxSAT is missing. |
| M090_SOLVER.5 GPU portfolio | portfolio dispatch executes on GPU or GPU-backed adapter with measured launch evidence | BLOCKED | Current status reports not implemented. |
| M090_SOLVER.6 failure modes | UNSAT/UNKNOWN/TIMEOUT represented distinctly | PASS for oracle | CPU oracle fixtures distinguish the states. |
| M090_SOLVER.7 assumption lifecycle | push, solve, retract, and reuse trace proves no assumption leak between candidates | PARTIAL | CPU lifecycle fixture exists; GPU candidate trace is missing. |
| M090_SOLVER.8 CPU search ban | accepted solver path records zero CPU exhaustive assignment enumeration | BLOCKED | Current accepted fixture service enumerates assignments on CPU. |

## Coordination Notes

- This file is not release-close evidence for `G090_SOLVER`.
- GPU-native SAT/MaxSAT/portfolio execution remains required before v0.9.0 can
  close.
- No pyxlog public API signatures were changed.
- No push, tag, release-board update, or merge was performed.
