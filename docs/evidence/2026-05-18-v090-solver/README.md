# v0.9.0 G090_SOLVER Evidence

Date: 2026-05-18

Goal node: `G090_SOLVER - SAT/MaxSAT Solver Services`

Branch: `feat/v090-epistemic-solver-semantics`

Predecessor evidence:

- `docs/evidence/2026-05-18-v090-pre/README.md`
- `docs/evidence/2026-05-18-v090-eir/README.md`
- `docs/evidence/2026-05-18-v090-g91/README.md`
- `docs/evidence/2026-05-18-v090-faeel/README.md`
- `docs/evidence/2026-05-18-v090-gpt/README.md`
- `docs/evidence/2026-05-18-v090-split/README.md`

## Implementation Summary

| Requirement | Evidence |
|---|---|
| Solver service interface | `SolverService` exposes bounded SAT/MaxSAT solves, assumptions, retraction, learned-clause transfer, trace, and GPU portfolio status. |
| Incremental SAT assumptions | `solver_service_semantics.rs` verifies assumption add/retract changes SAT status without stale learned contradictions. |
| Learned-clause transfer | `transfer_learned_clauses_to` transfers scoped learned clauses and increments `SolverServiceTrace::learned_clause_transfers`. |
| MaxSAT soft constraints | `SolveInstance::with_weights` fixture returns `SolverServiceStatus::Optimal(5)`. |
| GPU portfolio scope | `gpu_portfolio_status` returns an explicit `Deferred` status with rationale. |
| Failure modes | Fixtures distinguish `Unsat`, `Unknown`, and `Timeout`. |
| Documentation | `docs/architecture/solver-services.md` documents the bounded service contract and GPU portfolio deferral. |

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
| M090_SOLVER.1 service interface | public bounded service API | PASS | `SolverService`, `SolverServiceBudget`, `SolverServiceStatus`, `SolverPortfolioStatus`. |
| M090_SOLVER.2 incremental SAT | assumption add/retract fixture | PASS | `incremental_assumptions_can_be_added_and_retracted`. |
| M090_SOLVER.3 learned clauses | observable transfer | PASS | `learned_clause_transfer_is_observable`. |
| M090_SOLVER.4 MaxSAT | weighted soft-constraint optimum | PASS | `maxsat_soft_constraints_return_expected_optimum`. |
| M090_SOLVER.5 GPU portfolio | explicit deferral with rationale | PASS | `gpu_portfolio_scope_is_explicitly_deferred`. |
| M090_SOLVER.6 failure modes | distinct UNSAT/UNKNOWN/TIMEOUT | PASS | `failure_modes_distinguish_unsat_unknown_and_timeout`. |

## Coordination Notes

- This is a bounded service facade for semantic fixtures, not the production GPU verifier path.
- Learned clauses derived under active assumptions remain scoped to those assumptions.
- GPU portfolio solving is explicitly deferred; no GPU dispatch path was added.
- No pyxlog public API signatures were changed.
- No push, tag, release-board update, or merge was performed.
