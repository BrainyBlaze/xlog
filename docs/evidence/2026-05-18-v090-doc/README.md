# v0.9.0 G090_DOC Evidence

Date: 2026-05-18

Goal node: `G090_DOC - Documentation`

Branch: `feat/v090-epistemic-solver-semantics`

Predecessor evidence:

- `docs/evidence/2026-05-18-v090-pre/README.md`
- `docs/evidence/2026-05-18-v090-eir/README.md`
- `docs/evidence/2026-05-18-v090-g91/README.md`
- `docs/evidence/2026-05-18-v090-faeel/README.md`
- `docs/evidence/2026-05-18-v090-gpt/README.md`
- `docs/evidence/2026-05-18-v090-split/README.md`
- `docs/evidence/2026-05-18-v090-solver/README.md`
- `docs/evidence/2026-05-18-v090-prob/README.md`
- `docs/evidence/2026-05-18-v090-cert/README.md`

## Documentation Scope

| Requirement | Evidence |
|---|---|
| Epistemic guide | `docs/epistemic-solver-semantics-guide.md` explains EIR, G91, FAEEL, GPT, splitting, world-view fixtures, partial accepted GPU runtime evidence, and the corrected GPU-native blocker. |
| Solver guide | The same guide explains assumptions, incremental SAT, learned transfer, MaxSAT, portfolio/status propagation, the accepted GPU CDCL production adapter slices including G91/FAEEL mode-specific solver trace counters and binary `possible`/`not possible`/`not know` operator evidence, and why they still do not close `G090_SOLVER`. |
| GPU/WCOJ guide | The same guide now distinguishes the implemented bounded K5/K6/K7/K8 WCOJ dispatch, GPU buffer, tuple-membership, solver, and probabilistic production-reuse slices, including mode-specific accepted G91/FAEEL solver/probability trace evidence and operator-specific probabilistic evidence counters, from the still-missing release-wide semantic parity and post-v0.8 certification gates. |
| Runnable examples | `examples/epistemic/` has five `.xlog` fixtures run by `test_epistemic_examples`. |
| Roadmap sync | No ROADMAP or release-board rows were edited in this slice. |

## Validation

| Command | Result |
|---|---|
| `cargo fmt --check` | PASS |
| `cargo test -p xlog-logic --test test_epistemic_examples` | PASS, 5 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_binary_possible_membership_matches_gpt_oracle_parity -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_binary_not_possible_membership_matches_gpt_oracle_parity -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_binary_not_know_membership_matches_gpt_oracle_parity -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_quaternary_membership_matches_gpt_oracle_parity -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_g91_and_faeel_modes_gate_solver_production_trace -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_g91_and_faeel_modes_gate_probabilistic_production_trace -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_operator_conditions_record_probabilistic_operator_trace_counters -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution -- --nocapture` | PASS, 72 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_gpu_plan` | PASS, 8 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_eir --test test_epistemic_g91 --test test_epistemic_faeel --test test_epistemic_gpt --test test_epistemic_split --test test_epistemic_world_view --test test_epistemic_examples` | PASS, 24 passed, 0 failed |
| `cargo test -p xlog-solve --test solver_service_semantics` | PASS, 5 passed, 0 failed |
| `cargo test -p xlog-prob --test epistemic_prob` | PASS, 5 passed, 0 failed |
| `cargo check -p xlog-logic -p xlog-ir -p xlog-solve -p xlog-prob` | PASS |
| `cargo check -p pyxlog` | PASS |

## Metric Status

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M090_DOC.1 epistemic guide | guide explains EIR, G91, FAEEL, GPT, splitting | PASS for oracle | `docs/epistemic-solver-semantics-guide.md`. |
| M090_DOC.2 solver guide | guide explains GPU-native assumptions, incremental SAT, MaxSAT, portfolio dispatch, and failure states | PARTIAL | Guide documents the CPU oracle facade, accepted GPU CDCL production adapter slices, accepted G91/default FAEEL mode-specific solver trace counters, mixed unary and binary operator-result lifecycle evidence including `possible`, `not possible`, and binary `not know`, and the remaining solver semantic-integration blocker. |
| M090_DOC.3 examples | at least one runnable example per implemented major semantic mode | PASS for oracle | `examples/epistemic/` and `test_epistemic_examples`: 5/5 passed. |
| M090_DOC.4 roadmap sync | ROADMAP v0.9.0 rows updated only at closure, not prematurely marked done | PASS | No `ROADMAP.md` or board edits in this slice. |
| M090_DOC.5 GPU/WCOJ execution | guide documents the production GPU-native and WCOJ-backed epistemic execution path | PARTIAL | Guide documents the current bounded accepted GPU/WCOJ runtime path, K5/K6/K7/K8 dispatch evidence, GPU tuple-membership/final-materialization path, solver/probability production adapter slices including binary `possible`/`not possible`/`not know` evidence, true `know` probability evidence, accepted G91/default FAEEL mode-specific solver/probability trace counters, operator-specific probabilistic evidence counters, and remaining semantic-parity/rebase blockers. |

## Coordination Notes

- Epistemic examples are run by the fixture harness, not production `xlog run`.
- This documentation evidence is not a release-close claim.
- The guide repeats that the v0.8 compatibility rerun remains pending until rebase.
- No push, tag, release-board update, or merge was performed.
