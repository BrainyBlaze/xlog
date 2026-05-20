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
| Epistemic guide | `docs/epistemic-solver-semantics-guide.md` explains EIR, G91, FAEEL, GPT, splitting including aggregate split-batch zero CPU recomposition tracing, split binary operator GPT parity, split all-binary-operator GPT parity, mixed same-rule `know`/`possible`, negated `not know`/`not possible`, and all-operator membership GPU-vs-GPT parity, fail-closed split multi-membership modal-coupling rejection before GPU batching, all-binary-operator split-batch probabilistic source/program query and gradient conditioning evidence, all-binary split-batch source/program PIR-CNF plus already-compiled exact query/gradient evaluation, world-view fixtures, partial accepted GPU runtime evidence, and the corrected GPU-native blocker. |
| Solver guide | The same guide explains assumptions, incremental SAT, learned transfer, MaxSAT, portfolio/status propagation, the accepted GPU CDCL production adapter slices including split-batch lifecycle, all-binary-operator split-batch lifecycle plus all-binary split-batch learned-clause reuse and MaxSAT, learned-clause reuse, MaxSAT, MaxSAT search pruning, weighted MaxSAT encoding/search, generalized MaxSAT scheduling, portfolio evidence, quaternary solver nonzero-arity SAT evidence, G91/FAEEL mode-specific solver trace counters, accepted operator-family solver trace counters, and binary `possible`/`not possible`/`not know` operator evidence, and why they still do not close `G090_SOLVER`. |
| GPU/WCOJ guide | The same guide now distinguishes the implemented bounded v0.7.0 4-cycle plus K5/K6/K7/K8 WCOJ dispatch, GPU buffer, split-batch trace including binary operator parity and all-binary-operator parity, mixed `know`/`possible`, negated `not know`/`not possible`, and all-operator membership parity, fail-closed split modal-coupling pre-batch rejection, tuple-membership, solver, and probabilistic production-reuse slices, including accepted split-batch solver lifecycle, all-binary-operator split-batch solver lifecycle plus all-binary split-batch learned-clause reuse and MaxSAT, learned-clause reuse, MaxSAT, MaxSAT search pruning, weighted MaxSAT encoding/search, generalized MaxSAT scheduling, portfolio evidence, quaternary solver nonzero-arity SAT evidence, mode-specific accepted G91/FAEEL solver/probability trace evidence plus accepted split-batch source/program compile/evaluate, all-binary-operator conditioned source/program query and gradient paths, all-binary split-batch source/program PIR-CNF plus already-compiled exact query/gradient evaluation, and conditioned source/program query and gradient counters, source/program-specific PIR/CNF, source/program-specific exact-query, source/program-specific conditioned-gradient, source/program-specific conditioned-evidence, source/program-specific operator-conditioned evidence, source and parsed-program quaternary nonzero-arity evidence, and operator-specific probabilistic evidence counters, from the still-missing release-wide semantic parity and post-v0.7.0/v0.8.0/v0.8.5/v0.8.6 certification gates. |
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
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_all_operator_mixed_membership_conditions_probabilistic_evidence -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_all_operator_mixed_membership_gates_probabilistic_program_gradient_and_pir_paths -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_all_operator_mixed_membership_gates_probabilistic_source_pir_and_exact_evaluation_paths -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution conditioned_probabilistic_evidence_records_source_and_program_trace_counters -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_quaternary_source_probabilistic_evidence_records_nonzero_arity_trace -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_quaternary_parsed_program_probabilistic_evidence_records_nonzero_arity_trace -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution conditioned_probabilistic_gradients_record_source_and_program_trace_counters -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution probabilistic_end_to_end_records_source_and_program_query_trace_counters -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution probabilistic_pir_cnf_records_source_and_program_trace_counters -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_components_execute_gpu_runtime_and_match_component_oracles -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_binary_operator_components_match_gpt_oracles -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_all_binary_operators_match_gpt_oracles -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution split_multi_membership_modal_coupling_rejects_gpu_batching -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_mixed_memberships_match_gpt_oracle_parity -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_negated_mixed_memberships_match_gpt_oracle_parity -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_all_operator_mixed_memberships_match_gpt_oracle_parity -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_probabilistic_source_and_program_end_to_end_paths -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_probabilistic_conditioned_source_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_all_binary_operator_batch_conditions_probabilistic_evidence -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_all_binary_operator_batch_gates_probabilistic_program_and_gradient_paths -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_all_binary_operator_batch_gates_probabilistic_pir_cnf_and_exact_evaluation_paths -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_probabilistic_conditioned_program_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_probabilistic_conditioned_source_gradients -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_probabilistic_conditioned_program_gradients -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_solver_lifecycle_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_all_binary_operator_batch_gates_solver_lifecycle_path -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_all_binary_operator_batch_gates_solver_reuse_and_maxsat_paths -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_solver_learned_clause_reuse_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_solver_maxsat_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_solver_maxsat_search_pruning -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_solver_portfolio_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_all_operator_mixed_membership_gates_solver_lifecycle_path -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_all_operator_mixed_membership_gates_solver_reuse_maxsat_and_portfolio_paths -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_quaternary_gpu_execution_result_records_solver_nonzero_arity_evidence_trace -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_epistemic_v070_4cycle_execution_certifies_production_wcoj_dispatch -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution -- --nocapture` | PASS, 122 passed, 0 failed |
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
| M090_DOC.2 solver guide | guide explains GPU-native assumptions, incremental SAT, MaxSAT, portfolio dispatch, and failure states | PARTIAL | Guide documents the CPU oracle facade, accepted GPU CDCL production adapter slices, accepted split-batch lifecycle, all-binary-operator split-batch lifecycle plus all-binary split-batch learned-clause reuse and MaxSAT, learned-clause reuse, MaxSAT, MaxSAT search pruning, weighted MaxSAT encoding/search, generalized MaxSAT scheduling, portfolio evidence with batch/component counters, quaternary solver nonzero-arity SAT evidence, accepted G91/default FAEEL mode-specific solver trace counters, mixed unary and binary operator-result lifecycle evidence including `possible`, `not possible`, and binary `not know`, and the remaining solver semantic-integration blocker. |
| M090_DOC.3 examples | at least one runnable example per implemented major semantic mode | PASS for oracle | `examples/epistemic/` and `test_epistemic_examples`: 5/5 passed. |
| M090_DOC.4 roadmap sync | ROADMAP v0.9.0 rows updated only at closure, not prematurely marked done | PASS | No `ROADMAP.md` or board edits in this slice. |
| M090_DOC.5 GPU/WCOJ execution | guide documents the production GPU-native and WCOJ-backed epistemic execution path | PARTIAL | Guide documents the current bounded accepted GPU/WCOJ runtime path, v0.7.0 4-cycle and K5/K6/K7/K8 dispatch evidence, aggregate split-batch zero CPU recomposition, zero per-candidate host-round-trip trace evidence, split binary operator GPT parity, split all-binary-operator GPT parity, mixed same-rule `know`/`possible`, negated `not know`/`not possible`, and all-operator membership GPU-vs-GPT parity, fail-closed split multi-membership modal-coupling rejection before GPU batching, GPU tuple-membership/final-materialization path, solver/probability production adapter slices including accepted split-batch solver lifecycle, all-binary-operator split-batch solver lifecycle plus all-binary split-batch learned-clause reuse and MaxSAT, learned-clause reuse, MaxSAT, MaxSAT search pruning, weighted MaxSAT encoding/search, generalized MaxSAT scheduling, portfolio dispatch, and quaternary solver nonzero-arity SAT evidence, binary `possible`/`not possible`/`not know` evidence, true `know` probability evidence, accepted split-batch source/program compile/evaluate, all-binary-operator conditioned source/program query and gradient evidence, all-binary split-batch source/program PIR-CNF plus already-compiled exact query/gradient evaluation, and conditioned source/program query and gradient evidence, accepted G91/default FAEEL mode-specific solver/probability trace counters, source/program-specific PIR/CNF, source/program-specific exact-query, source/program-specific conditioned-gradient, source/program-specific conditioned-evidence, source/program-specific operator-conditioned evidence, source and parsed-program quaternary nonzero-arity evidence, and operator-specific probabilistic evidence counters, and remaining semantic-parity/rebase blockers. |

## Coordination Notes

- Epistemic examples are run by the fixture harness, not production `xlog run`.
- This documentation evidence is not a release-close claim.
- The guide keeps the v0.7.0/v0.8.0/v0.8.5/v0.8.6 compatibility gate
  separate from release closure.
- No push, tag, release-board update, or merge was performed.
