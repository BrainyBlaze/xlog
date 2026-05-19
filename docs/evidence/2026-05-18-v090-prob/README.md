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
exact/provenance machinery, but broader probabilistic knowledge-compilation
coverage over accepted world views is not complete.

| Requirement | Evidence |
|---|---|
| Accepted world-view evidence | `AcceptedWorldViewEvidence` is constructed from a non-empty `EpistemicWorldView`. |
| Semantic contract | `xlog_prob::epistemic` represents accepted epistemic assumptions as probabilistic evidence conditions. |
| Production exact adapter | `EpistemicProbProductionAdapter` gates on `AcceptedWorldViewEvidence` and compiles through `ExactDdnnfProgram::compile_source_with_gpu` or `ExactDdnnfProgram::compile_from_program`. |
| Accepted runtime exact gates | `compile_source_with_gpu_execution_result` and `compile_program_with_gpu_execution_result` construct `AcceptedWorldViewEvidence` from an accepted `EpistemicGpuExecutionResult`, require stable-model tuple membership, GPU kernel traces, zero hot-path transfers, and non-empty final device output before exact compilation. |
| Accepted runtime end-to-end gate | `compile_and_evaluate_source_with_gpu_execution_result` and `compile_and_evaluate_program_with_gpu_execution_result` consume accepted runtime evidence once before compiling through `ExactDdnnfProgram` and evaluating queries from that compiled GPU exact state, with separate source and parsed-program trace counters. |
| Accepted runtime PIR/CNF gate | `encode_source_pir_cnf_with_gpu_execution_result` and `encode_program_pir_cnf_with_gpu_execution_result` reconstruct accepted evidence before calling `GpuPirGraph::from_host`, `GpuPirRoots::from_host`, and `encode_cnf_gpu`. |
| Accepted runtime evaluation gates | `evaluate_with_gpu_execution_result` and `evaluate_gpu_with_grads_with_gpu_execution_result` reconstruct accepted evidence from the GPU runtime result before calling the existing `ExactDdnnfProgram::evaluate` and `ExactDdnnfProgram::evaluate_gpu_with_grads` paths. |
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
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_gates_probabilistic_program_end_to_end_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_gates_probabilistic_pir_cnf_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_gates_probabilistic_query_evaluation_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_gates_probabilistic_gradient_evaluation_path -- --nocapture` | PASS, 1 passed, 0 failed |
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
| M090_PROB.5 evidence conditioning | probabilistic integration consumes accepted world views, not raw unvalidated guesses | PARTIAL | `AcceptedWorldViewEvidence` requires an `EpistemicWorldView` for oracle fixtures and can be constructed from an accepted GPU runtime result after stable tuple-source, kernel-trace, transfer-budget, and non-empty final-output checks. |
| M090_PROB.6 GPU exact integration | accepted world-view evidence updates the GPU-native exact/provenance path | PARTIAL | Accepted GPU runtime evidence gates `ExactDdnnfProgram::compile_source_with_gpu`, `ExactDdnnfProgram::compile_from_program`, `evaluate`, `evaluate_gpu_with_grads`, and source plus parsed-program compile-plus-query-evaluation through the same exact state; broader query-conditioning coverage is still missing. |
| M090_PROB.7 CPU recompute ban | accepted probabilistic epistemic path records zero CPU-only probability recomputation | PARTIAL | Production trace records zero CPU-only recomputation and zero fixture-circuit counters for accepted runtime source-compile, parsed-program compile, PIR/CNF encoding, query-evaluation, gradient-evaluation, and source plus parsed-program end-to-end compile/evaluate paths; broader probabilistic execution traces are missing. |
| M090_PROB.8 production prob reuse | accepted probabilistic fixtures execute through existing GPU exact/provenance/PIR/knowledge-compilation APIs | PARTIAL | Source guard and integration fixtures prove accepted GPU runtime evidence compiles source and parsed programs, performs source and parsed-program bounded compile/evaluate knowledge-compilation through `ExactDdnnfProgram` with distinct trace counters, encodes PIR/CNF through `GpuPirGraph` and `encode_cnf_gpu`, evaluates query probabilities, and evaluates gradients through the existing exact/provenance path; broader knowledge-compilation coverage is missing. |
| M090_PROB.9 fixture isolation | bounded epistemic probability fixtures are marked oracle-only and cannot satisfy closure metrics | PARTIAL | Evidence docs separate `EpistemicCircuit` fixtures from `EpistemicProbProductionAdapter`; `EpistemicProbProductionCapabilities` disallows fixture circuits for production metrics; `EpistemicProbProductionTrace::require_production_metric_eligibility` rejects traces without accepted world-view evidence, without existing GPU exact/provenance/PIR/CNF counters, or with CPU/fixture recomputation counters. Broader probabilistic coverage is still missing, so this is not a G090_PROB close. |

## Coordination Notes

- This file is not release-close evidence for `G090_PROB`.
- Production WFS/provenance still rejects direct epistemic literals.
- The production adapter is partial source/program exact-compile,
  PIR/CNF, query-evaluation, gradient-evaluation, and source/program bounded
  compile/evaluate reuse evidence only.
- The external Decision-DNNF adapter is a design contract, not a dispatch path.
- No pyxlog public API signatures were changed.
- No push, tag, release-board update, or merge was performed.
