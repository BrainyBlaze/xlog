# v0.9.0 G090_PROB Semantic And Production-Reuse Evidence

Date: 2026-05-18

Goal node: `G090_PROB - Probabilistic And Circuit Integration`

Branch: `feat/v090-epistemic-solver-semantics`

## Implementation Summary

The current branch contains fixture-level probabilistic integration for accepted
world-view evidence plus a thin production adapter. The fixture layer proves
that probabilistic evidence is gated by an accepted `EpistemicWorldView`.
`EpistemicProbProductionAdapter` then proves that accepted evidence can gate
calls into the existing GPU-native `ExactDdnnfProgram` exact/provenance path
without using the bounded fixture circuit.

This remains partial evidence. The accepted epistemic runtime can now feed a
validated `EpistemicGpuExecutionResult` into source and parsed-program
production exact compilation, GPU PIR upload/CNF encoding, and GPU exact query
and gradient evaluation. It also gates bounded source and parsed-program
compile-plus-query-evaluation paths end to end through the same
exact/provenance machinery, including a two-record accepted-runtime batch over
the source compile/evaluate path. Bounded source and parsed-program
conditioning paths now compile accepted zero-arity and concrete nonzero-arity
epistemic assumptions, including true `know`, true `possible`, false
`possible`/`not possible`, and false `know`/`not know` operator evidence, into exact
`evidence(atom(...), value)` statements before evaluating through the GPU exact
path, including two-record conditioned source and parsed-program batches whose
query probabilities differ by accepted world-view evidence plus a two-record
conditioned source batch whose false tuple assumptions drive the corresponding
query probabilities to zero and record two negative evidence facts.
Conditioned source and parsed-program gradient APIs now run through the same
accepted-world-view evidence boundary, including a two-record parsed-program
gradient batch. Accepted GPU runtime evidence also preserves the runtime
epistemic mode at the probabilistic boundary, and the production trace records
mode-specific accepted G91 and default FAEEL evidence consumptions. The
conditioned exact trace now also separates true `know`, true `possible`, false
`know` (`not know`), and false `possible` (`not possible`) evidence counters
for a four-record accepted operator batch.
Broader probabilistic coverage and release certification remain incomplete.

| Requirement | Evidence |
|---|---|
| Accepted world-view evidence | `AcceptedWorldViewEvidence` is constructed from a non-empty `EpistemicWorldView`. |
| Semantic contract | `xlog_prob::epistemic` represents accepted epistemic assumptions as probabilistic evidence conditions. |
| Production exact adapter | `EpistemicProbProductionAdapter` gates on `AcceptedWorldViewEvidence` and compiles through `ExactDdnnfProgram::compile_source_with_gpu` or `ExactDdnnfProgram::compile_from_program`. |
| Accepted runtime exact gates | `compile_source_with_gpu_execution_result` and `compile_program_with_gpu_execution_result` construct `AcceptedWorldViewEvidence` from an accepted `EpistemicGpuExecutionResult`, require stable-model tuple membership, GPU kernel traces, zero hot-path transfers, and non-empty final device output before exact compilation. |
| Accepted runtime end-to-end gate | `compile_and_evaluate_source_with_gpu_execution_result` and `compile_and_evaluate_program_with_gpu_execution_result` consume accepted runtime evidence once before compiling through `ExactDdnnfProgram` and evaluating queries from that compiled GPU exact state, with separate source and parsed-program trace counters. |
| Accepted runtime batch gate | `compile_and_evaluate_source_for_gpu_execution_results` and `compile_and_evaluate_program_for_gpu_execution_results` validate two accepted GPU runtime evidence records before running each source or parsed-program compile/evaluate through `ExactDdnnfProgram`, recording two accepted evidence consumptions, source/program compile counters, two query evaluations, source/program knowledge-compilation counters, and zero CPU recomputations. |
| Accepted runtime conditioning gate | `compile_and_evaluate_conditioned_source_with_gpu_execution_result`, `compile_and_evaluate_conditioned_program_with_gpu_execution_result`, `compile_and_evaluate_conditioned_source_for_gpu_execution_results`, and `compile_and_evaluate_conditioned_program_for_gpu_execution_results` construct accepted GPU evidence, append zero-arity and concrete nonzero-arity tuple assumptions as parsed `Evidence` AST entries, preserve true `know`, true `possible`, false `know`/`not know`, and false `possible`/`not possible` operator evidence, and evaluate through `ExactDdnnfProgram` with `accepted_evidence_assumptions_consumed`, `gpu_conditioned_evidence_facts`, `gpu_conditioned_negative_evidence_facts`, operator-specific `gpu_conditioned_*_evidence_facts`, mode-specific `accepted_g91_world_view_evidence_consumed`/`accepted_faeel_world_view_evidence_consumed`, and source/program end-to-end counters. The two-record negative source batch records `gpu_conditioned_negative_evidence_facts == 2`; the four-record operator batch records one true `know`, one true `possible`, one false `know`, and one false `possible` evidence fact while keeping zero CPU recomputation. |
| Accepted runtime conditioned-gradient gate | `compile_and_evaluate_conditioned_source_with_grads_with_gpu_execution_result`, `compile_and_evaluate_conditioned_program_with_grads_with_gpu_execution_result`, `compile_and_evaluate_conditioned_source_with_grads_for_gpu_execution_results`, and `compile_and_evaluate_conditioned_program_with_grads_for_gpu_execution_results` reuse the same parsed exact-evidence conditioning boundary before calling `ExactDdnnfProgram::evaluate_gpu_with_grads`, recording conditioned evidence facts, false-evidence facts, GPU gradient evaluations, source/program end-to-end counters, and zero CPU recomputations. |
| Accepted runtime PIR/CNF gate | `encode_source_pir_cnf_with_gpu_execution_result`, `encode_program_pir_cnf_with_gpu_execution_result`, `encode_source_pir_cnf_for_gpu_execution_results`, and `encode_program_pir_cnf_for_gpu_execution_results` reconstruct accepted evidence before calling `GpuPirGraph::from_host`, `GpuPirRoots::from_host`, and `encode_cnf_gpu`, including two-record source and parsed-program PIR/CNF batches. |
| Accepted runtime evaluation gates | `evaluate_with_gpu_execution_result`, `evaluate_for_gpu_execution_results`, `evaluate_gpu_with_grads_with_gpu_execution_result`, and `evaluate_gpu_with_grads_for_gpu_execution_results` reconstruct accepted evidence from accepted GPU runtime results before calling the existing `ExactDdnnfProgram::evaluate` and `ExactDdnnfProgram::evaluate_gpu_with_grads` paths, including two-record query and gradient batches over one already-compiled exact program. |
| CPU probability isolation | `EpistemicProbProductionTrace` records zero CPU-only probability recomputation and zero fixture-circuit evaluations; the source guard rejects `EpistemicCircuit::compile` in the production adapter. |
| Incremental circuit fixture | `EpistemicCircuit::apply_accepted_world_view` updates active evidence without changing the circuit fingerprint when the adapter supports incremental evidence. |
| Compiler adapter | `KnowledgeCompilerAdapter::external_ddnnf_text` records an alternative Decision-DNNF text adapter design. |
| Numerical stability | `conditional_probability_from_logs` normalizes conditional probabilities with `EPISTEMIC_PROBABILITY_TOLERANCE = 1e-12`. |

## Validation

| Command | Result |
|---|---|
| `cargo fmt --check` | PASS |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_gates_probabilistic_exact_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_gates_probabilistic_program_compile_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_gates_probabilistic_end_to_end_knowledge_compilation_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_results_gate_batched_probabilistic_knowledge_compilation_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_results_gate_batched_probabilistic_program_knowledge_compilation_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_conditions_zero_arity_probabilistic_evidence -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_conditions_nonzero_arity_probabilistic_evidence -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_conditions_negative_nonzero_arity_probabilistic_evidence -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_possible_operator_conditions_probabilistic_evidence -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_not_possible_operator_conditions_negative_probabilistic_evidence -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_binary_possible_operator_conditions_probabilistic_evidence -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_binary_not_possible_operator_conditions_negative_probabilistic_evidence -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_binary_not_know_operator_conditions_negative_probabilistic_evidence -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_g91_and_faeel_modes_gate_probabilistic_production_trace -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_operator_conditions_record_probabilistic_operator_trace_counters -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_conditions_parsed_program_probabilistic_evidence -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_conditions_negative_parsed_program_probabilistic_evidence -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_results_gate_batched_conditioned_probabilistic_queries -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_results_gate_batched_negative_conditioned_probabilistic_queries -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_results_gate_batched_conditioned_parsed_program_queries -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_conditions_probabilistic_gradient_evidence -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_results_gate_batched_conditioned_parsed_program_gradients -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_gates_probabilistic_program_end_to_end_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_gates_probabilistic_pir_cnf_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_results_gate_batched_probabilistic_source_pir_cnf_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_results_gate_batched_probabilistic_program_pir_cnf_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_gates_probabilistic_query_evaluation_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_results_gate_batched_probabilistic_query_evaluations -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_gates_probabilistic_gradient_evaluation_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_results_gate_batched_probabilistic_gradient_evaluations -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-prob --test epistemic_prob_production_reuse` | PASS, 3 passed, 0 failed |
| `cargo test -p xlog-prob --test epistemic_prob_production_reuse production_prob_capabilities_disallow_fixture_circuit_metrics -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-prob --test epistemic_prob_production_reuse production_prob_metric_gate_rejects_fixture_only_traces -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-prob --test epistemic_prob` | PASS, 5 passed, 0 failed |
| `cargo test -p xlog-prob --test no_cpu_d4_in_exact` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-prob --test no_dtoh_in_gpu_exact_path` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-prob --lib` | PASS, 56 passed, 0 failed |
| `cargo check -p xlog-prob --features host-io` | PASS |
| `cargo check -p xlog-logic -p xlog-ir -p xlog-solve -p xlog-prob` | PASS |
| `cargo check -p pyxlog` | PASS |

## Metric Status

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M090_PROB.1 semantic contract | documented interaction between epistemic and probabilistic layers | PASS for oracle | `AcceptedWorldViewEvidence` and architecture docs. |
| M090_PROB.2 incremental circuit fixture | changed assumption updates circuit without full rebuild where supported | PASS for oracle | `evidence_conditioning_consumes_accepted_world_view`. |
| M090_PROB.3 compiler adapter | at least one alternative compiler adapter design or implementation | PASS for oracle | `external_ddnnf_text_compiler_adapter_is_explicitly_represented`. |
| M090_PROB.4 numerical stability | deterministic fixture within documented tolerance | PASS for oracle | `log_space_conditional_probability_is_tolerance_bounded`. |
| M090_PROB.5 evidence conditioning | probabilistic integration consumes accepted world views, not raw unvalidated guesses | PARTIAL | `AcceptedWorldViewEvidence` requires an `EpistemicWorldView` for oracle fixtures and can be constructed from one or more accepted GPU runtime results after stable tuple-source, kernel-trace, transfer-budget, non-empty final-output, and runtime epistemic-mode checks. The conditioned exact path consumes accepted zero-arity and concrete nonzero-arity tuple assumptions, including true `know`, true `possible`, false `possible`/`not possible`, and false `know`/`not know` operator evidence, as parsed exact evidence for source, parsed-program, two-record positive source-batch, two-record negative source-batch, two-record parsed-program-batch, conditioned source-gradient, conditioned parsed-program-gradient, and four-record operator-trace inputs and records accepted-assumption, total evidence-fact, negative evidence-fact, operator-specific, and mode-specific G91/FAEEL evidence counters. |
| M090_PROB.6 GPU exact integration | accepted world-view evidence updates the GPU-native exact/provenance path | PARTIAL | Accepted GPU runtime evidence gates `ExactDdnnfProgram::compile_source_with_gpu`, `ExactDdnnfProgram::compile_from_program`, `evaluate`, `evaluate_gpu_with_grads`, source plus parsed-program compile-plus-query-evaluation through the same exact state, two-record source and parsed-program batch compile/evaluate, source plus parsed-program zero-arity/concrete nonzero-arity true and false exact evidence conditioning including true `know`, true `possible`, false `possible`/`not possible`, and false `know`/`not know` operator results, two-record positive and negative conditioned source query batches, two-record conditioned parsed-program query batches, conditioned source plus parsed-program gradient evaluation, mode-specific accepted G91/FAEEL production trace accounting, operator-specific conditioned evidence counters, and two-record query/gradient evaluation batches over one already-compiled exact program. Broader probabilistic semantic coverage is still missing. |
| M090_PROB.7 CPU recompute ban | accepted probabilistic epistemic path records zero CPU-only probability recomputation | PARTIAL | Production trace records zero CPU-only recomputation and zero fixture-circuit counters for accepted runtime source-compile, parsed-program compile, PIR/CNF encoding, query-evaluation, gradient-evaluation, source plus parsed-program end-to-end compile/evaluate paths, two-record source and parsed-program batch compile/evaluate, two-record source/program PIR/CNF encoding, source plus parsed-program zero-arity/concrete nonzero-arity true and false conditioned evaluation including true `know`, true `possible`, false `possible`/`not possible`, and false `know`/`not know` operator-result conditioning, four-record operator-specific trace accounting, two-record positive and negative conditioned source query batches, two-record conditioned parsed-program query batches, conditioned source plus parsed-program gradient evaluation, mode-specific accepted G91/FAEEL evidence accounting, and two-record exact query/gradient evaluation batches; full probabilistic execution traces are missing. |
| M090_PROB.8 production prob reuse | accepted probabilistic fixtures execute through existing GPU exact/provenance/PIR/knowledge-compilation APIs | PARTIAL | Source guard and integration fixtures prove accepted GPU runtime evidence compiles source and parsed programs, performs source and parsed-program bounded compile/evaluate knowledge-compilation through `ExactDdnnfProgram` with distinct trace counters, performs two-record accepted source and parsed-program batch compile/evaluate, conditions zero-arity and concrete nonzero-arity true and false evidence via parsed `Evidence` AST entries for source, parsed-program, operator-level true `know`, true `possible`, false `possible`/`not possible`, false `know`/`not know`, two-record positive and negative source-batch, two-record parsed-program-batch, conditioned source-gradient, conditioned parsed-program-gradient, and four-record operator-trace inputs, records separate accepted G91 and FAEEL evidence counters plus operator-specific evidence counters, encodes single-record and two-record source/program PIR/CNF through `GpuPirGraph` and `encode_cnf_gpu`, evaluates single-record and two-record query probabilities, and evaluates single-record and two-record gradients through the existing exact/provenance path. Broader probabilistic coverage is still missing. |
| M090_PROB.9 fixture isolation | bounded epistemic probability fixtures are marked oracle-only and cannot satisfy closure metrics | PARTIAL | Evidence docs separate `EpistemicCircuit` fixtures from `EpistemicProbProductionAdapter`; `EpistemicProbProductionCapabilities` disallows fixture circuits for production metrics; `EpistemicProbProductionTrace::require_production_metric_eligibility` rejects traces without accepted world-view evidence, without existing GPU exact/provenance/PIR/CNF counters, or with CPU/fixture recomputation counters. Full probabilistic coverage is still missing, so this is not a G090_PROB close. |

## Coordination Notes

- This file is not release-close evidence for `G090_PROB`.
- Production WFS/provenance still rejects direct epistemic literals.
- The production adapter is partial source/program exact-compile,
  single-record and two-record PIR/CNF, single-record and two-record query-evaluation, single-record and two-record gradient-evaluation, zero-arity and concrete
  nonzero-arity true and false conditioned source/program evaluation,
  operator-level true know/possible plus false not-possible/not-know conditioned source evaluation,
  four-record operator-specific exact trace counters,
  mode-specific accepted G91/FAEEL production trace counters,
  source/program bounded compile/evaluate reuse, two-record accepted
  source/program batch reuse, two-record negative conditioned source batch reuse, and
  two-record conditioned source/parsed-program query and gradient batch
  evidence only.
- The external Decision-DNNF adapter is a design contract, not a dispatch path.
- No pyxlog public API signatures were changed.
- No push, tag, release-board update, or merge was performed.
