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
and gradient evaluation, including two-record and accepted split-batch direct
source and parsed-program exact compilation. It also gates bounded source and parsed-program
compile-plus-query-evaluation paths end to end through the same
exact/provenance machinery, including two-record accepted-runtime batches and
accepted split-batch runtime evidence over the source and parsed-program
compile/evaluate paths. Bounded source and parsed-program
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
for a four-record accepted operator batch, and the source-vs-program trace
separates those operator-family evidence counters by source and parsed-program
conditioned paths. Ternary and quaternary source-conditioned evidence,
negated quaternary `not possible` source-conditioned evidence, quaternary
parsed-program accepted probabilistic evidence, and negated quaternary
`not possible` parsed-program accepted probabilistic evidence now record
aggregate and source/program-specific nonzero-arity evidence counts plus
maximum conditioned evidence arity in the production trace. A two-record
quaternary `possible fact4/4` plus `not know fact4/4` conditioned source batch
now proves the same accepted runtime results drive one query probability to
true and one query probability to false while recording source-conditioned
arity-four, operator-family, negative-evidence, exact-query, and zero CPU
recompute counters.
The same two-record accepted runtime source batch now also conditions source
gradient evaluation, recording source-conditioned arity-four `possible` and
`not know` evidence counters, one negative evidence fact, source conditioned
gradient counters, and zero CPU probability recomputation.
A single-result quaternary `know fact4/4` accepted runtime fixture now also
gates source and parsed-program PIR/CNF encoding plus already-compiled exact
query and gradient evaluation through the existing GPU exact/provenance APIs,
recording source/program PIR-CNF counters, accepted-assumption accounting, and
zero CPU probability recomputation.
The same accepted runtime evidence now also conditions source and
parsed-program gradient evaluation, recording source/program conditioned
arity-four evidence counters, source/program conditioned-gradient counters, and
zero CPU probability recomputation.
A single-result quaternary `not possible fact4/4` accepted runtime fixture now
also conditions source and parsed-program gradient evaluation, recording
source/program conditioned negative arity-four evidence counters,
source/program conditioned-gradient counters, and zero CPU probability
recomputation.
The same-rule all-operator accepted runtime fixture now also conditions source
and parsed-program queries and gradients from one accepted GPU execution result
with four assumptions, and it gates parsed-program PIR/CNF encoding, proving
`know`, `possible`, `not know`, and `not possible` evidence facts are consumed
together without CPU recomputation.
Accepted split-batch runtime evidence can now gate unconditioned source and
parsed-program compile/evaluate plus conditioned source and
parsed-program query and gradient evaluation through
`EpistemicProbGpuBatchExecutionEvidence`, validating the aggregate
`EpistemicGpuBatchExecutionTrace` before each component's accepted world-view is
routed through the existing exact path or each component's accepted assumptions
are appended as parsed evidence, with
`accepted_gpu_batch_evidence_consumed`,
`accepted_gpu_batch_component_evidence_consumed`, and source/program-specific
conditioned gradient trace counters. A four-component all-binary-operator split
batch now conditions source and parsed-program queries plus source and
parsed-program gradients with one true `know`, one true `possible`, one false
`possible`/`not possible`, and one false `know`/`not know` accepted assumption
while preserving aggregate split-batch zero CPU recomposition and zero
probability recomputation counters. A two-component quaternary split batch now
conditions parsed-program queries with one true `know fact4/4` component and
one false `possible`/`not possible fact4/4` component while recording
program-conditioned arity-four evidence, one negative evidence fact, and zero
CPU probability recomputation. A two-component quaternary `possible fact4/4`
plus `not know fact4/4` split batch now conditions source queries with one true
possible assumption and one false known assumption while recording arity-four
source-conditioned evidence, one negative evidence fact, and zero CPU
probability recomputation. The same possible/not-know split batch now also gates
conditioned source and parsed-program gradients, source and parsed-program
PIR/CNF encoding, and already-compiled exact query/gradient evaluation with
arity-four accepted assumptions, recording batch/component evidence counters and
zero CPU recomputation. The same accepted split-batch
evidence also gates source and parsed-program PIR/CNF encoding plus
already-compiled exact query and gradient evaluation for the all-binary split
batch through the existing GPU exact/provenance APIs.
All accepted probabilistic split-batch entrypoints now share the single
`accepted_world_views_from_gpu_batch_execution_evidence` validator before any
source/program compile, conditioned query, conditioned gradient, PIR/CNF,
already-compiled query, or already-compiled gradient work can run. The source
audit locks this with `production_prob_batch_paths_use_single_gpu_batch_gate`,
so missing component counts, CPU recomposition/fallback counters, tracked D2H
calls, per-candidate host round trips, or incomplete aggregate CUDA-event
timing cannot drift independently across probability batch APIs.
Broader probabilistic coverage and release certification remain incomplete.

| Requirement | Evidence |
|---|---|
| Accepted world-view evidence | `AcceptedWorldViewEvidence` is constructed from a non-empty `EpistemicWorldView`. |
| Semantic contract | `xlog_prob::epistemic` represents accepted epistemic assumptions as probabilistic evidence conditions. |
| Production exact adapter | `EpistemicProbProductionAdapter` gates on `AcceptedWorldViewEvidence` and compiles through `ExactDdnnfProgram::compile_source_with_gpu` or `ExactDdnnfProgram::compile_from_program`. |
| Accepted runtime exact gates | `compile_source_with_gpu_execution_result`, `compile_program_with_gpu_execution_result`, `compile_source_for_gpu_execution_results`, `compile_program_for_gpu_execution_results`, `compile_source_for_gpu_batch_execution_result`, and `compile_program_for_gpu_batch_execution_result` construct `AcceptedWorldViewEvidence` from accepted `EpistemicGpuExecutionResult` or accepted `EpistemicProbGpuBatchExecutionEvidence` records, require stable-model tuple membership, timed GPU candidate-generation/propagation/validation/model-membership/world-view/final-materialization traces, zero hot-path transfers, non-empty final device output, and aggregate split-batch zero CPU recomposition/fallback/host-round-trip counters plus aggregate CUDA-event timing that fails closed on partial component timing before exact compilation. |
| Accepted runtime end-to-end gate | `compile_and_evaluate_source_with_gpu_execution_result` and `compile_and_evaluate_program_with_gpu_execution_result` consume accepted runtime evidence once before compiling through `ExactDdnnfProgram` and evaluating queries from that compiled GPU exact state, with separate source and parsed-program compile, exact-query, and end-to-end trace counters. |
| Accepted runtime batch gate | `compile_and_evaluate_source_for_gpu_execution_results` and `compile_and_evaluate_program_for_gpu_execution_results` validate two accepted GPU runtime evidence records before running each source or parsed-program compile/evaluate through `ExactDdnnfProgram`, recording two accepted evidence consumptions, source/program compile counters, two query evaluations, source/program knowledge-compilation counters, and zero CPU recomputations. `compile_and_evaluate_source_for_gpu_batch_execution_result` and `compile_and_evaluate_program_for_gpu_batch_execution_result` consume `EpistemicProbGpuBatchExecutionEvidence`, validate aggregate split-batch zero CPU recomposition/fallback/host-round-trip counters plus aggregate CUDA-event timing that fails closed on partial component timing, and route each accepted component through the same source or parsed-program compile/evaluate exact path while recording batch and component evidence counters. |
| Accepted runtime conditioning gate | `compile_and_evaluate_conditioned_source_with_gpu_execution_result`, `compile_and_evaluate_conditioned_program_with_gpu_execution_result`, `compile_and_evaluate_conditioned_source_for_gpu_execution_results`, and `compile_and_evaluate_conditioned_program_for_gpu_execution_results` construct accepted GPU evidence, append zero-arity and concrete nonzero-arity tuple assumptions as parsed `Evidence` AST entries, preserve true `know`, true `possible`, false `know`/`not know`, and false `possible`/`not possible` operator evidence, and evaluate through `ExactDdnnfProgram` with `accepted_evidence_assumptions_consumed`, `gpu_conditioned_evidence_facts`, `gpu_conditioned_nonzero_arity_evidence_facts`, `gpu_source_conditioned_nonzero_arity_evidence_facts`, `gpu_program_conditioned_nonzero_arity_evidence_facts`, `gpu_conditioned_max_evidence_arity`, `gpu_source_conditioned_max_evidence_arity`, `gpu_program_conditioned_max_evidence_arity`, `gpu_conditioned_negative_evidence_facts`, source/program-specific conditioned evidence counters, operator-specific `gpu_conditioned_*_evidence_facts`, source/program-specific operator-conditioned evidence counters, mode-specific `accepted_g91_world_view_evidence_consumed`/`accepted_faeel_world_view_evidence_consumed`, and source/program end-to-end counters. The two-record negative source batch records `gpu_conditioned_negative_evidence_facts == 2`; the ternary source fixture records one source nonzero-arity evidence fact and maximum evidence arity `3`; the quaternary source fixture records one source nonzero-arity evidence fact and maximum evidence arity `4`; `accepted_quaternary_not_possible_probabilistic_evidence_records_negative_nonzero_arity_trace` records one source nonzero-arity evidence fact, maximum evidence arity `4`, one negative evidence fact, and one source-conditioned `not possible` evidence fact while driving the queried tuple probability to zero; `accepted_quaternary_possible_and_not_know_results_gate_solver_and_probabilistic_paths` records two source nonzero-arity evidence facts, maximum evidence arity `4`, one negative evidence fact, one source-conditioned `possible` evidence fact, and one source-conditioned `not know` evidence fact while driving the corresponding source queries to true and false; `accepted_quaternary_possible_and_not_know_results_gate_source_conditioned_probabilistic_gradients` records the same two accepted source-conditioned arity-four facts through gradient evaluation with source-conditioned `possible` and `not know` counters; `accepted_quaternary_possible_and_not_know_results_gate_parsed_program_probabilistic_paths` records the same two accepted arity-four facts through parsed-program query and gradient conditioning with parsed-program `possible` and `not know` counters; the quaternary parsed-program fixture records one parsed-program nonzero-arity evidence fact and maximum evidence arity `4`; the source-vs-program trace fixture records one source-conditioned `know` evidence fact and one parsed-program-conditioned `not know` negative evidence fact; the four-record operator batch records one true `know`, one true `possible`, one false `know`, and one false `possible` evidence fact while keeping zero CPU recomputation. |
| Accepted split-batch conditioning gate | `compile_and_evaluate_conditioned_source_for_gpu_batch_execution_result`, `compile_and_evaluate_conditioned_program_for_gpu_batch_execution_result`, `compile_and_evaluate_conditioned_source_with_grads_for_gpu_batch_execution_result`, and `compile_and_evaluate_conditioned_program_with_grads_for_gpu_batch_execution_result` consume `EpistemicProbGpuBatchExecutionEvidence` through the shared `accepted_world_views_from_gpu_batch_execution_evidence` gate, validate the aggregate split-batch trace for one GPU runtime execution per component plus zero CPU recomposition, zero CPU candidate/world-view fallback, zero tracked D2H calls, zero per-candidate host-round-trip counters, and aggregate CUDA-event timing that fails closed on partial component timing, then evaluate each component's accepted assumptions through the existing conditioned source or parsed-program `ExactDdnnfProgram` query/gradient path while recording batch, component evidence, exact query, and conditioned gradient counters; `accepted_split_all_binary_operator_batch_conditions_probabilistic_evidence` proves a four-component `know`/`possible`/`not possible`/`not know` split batch conditions source query probabilities with two positive and two negative evidence facts, `accepted_split_all_binary_operator_batch_gates_probabilistic_program_and_gradient_paths` proves the same all-binary batch conditions parsed-program queries plus source and parsed-program gradients while preserving source/program-specific operator counters and zero CPU probability recomputation, `accepted_split_quaternary_not_possible_batch_conditions_parsed_program_probabilistic_evidence` proves the two-component quaternary split batch conditions parsed-program queries with arity-four positive and negative assumptions, `accepted_split_quaternary_possible_and_not_know_batch_gates_solver_and_probabilistic_paths` proves a two-component quaternary `possible fact4/4` plus `not know fact4/4` split batch conditions source queries with arity-four source evidence, one negative evidence fact, one source-conditioned `possible` counter, one source-conditioned `not know` counter, and zero CPU recomputation, `accepted_split_quaternary_not_possible_batch_gates_probabilistic_gradient_pir_cnf_and_exact_evaluation_paths` proves the quaternary `know` plus `not possible fact4/4` split batch conditions source and parsed-program gradients while preserving arity-four source/program-specific evidence counters and zero CPU probability recomputation, and `accepted_split_quaternary_possible_and_not_know_batch_gates_probabilistic_gradient_pir_cnf_and_exact_evaluation_paths` proves the quaternary `possible` plus `not know fact4/4` split batch gates source/program gradients, source/program PIR/CNF, and already-compiled exact query/gradient evaluation while preserving arity-four source/program-specific evidence counters and zero CPU probability recomputation. |
| Accepted runtime conditioned-gradient gate | `compile_and_evaluate_conditioned_source_with_grads_with_gpu_execution_result`, `compile_and_evaluate_conditioned_program_with_grads_with_gpu_execution_result`, `compile_and_evaluate_conditioned_source_with_grads_for_gpu_execution_results`, and `compile_and_evaluate_conditioned_program_with_grads_for_gpu_execution_results` reuse the same parsed exact-evidence conditioning boundary before calling `ExactDdnnfProgram::evaluate_gpu_with_grads`, recording conditioned evidence facts, false-evidence facts, GPU gradient evaluations, source/program-specific conditioned gradient counters, source/program end-to-end counters, same-rule all-operator mixed-membership source gradients, single-result quaternary `know fact4/4` source/program conditioned gradients, single-result quaternary `not possible fact4/4` source/program conditioned gradients, two-record quaternary `possible fact4/4` plus `not know fact4/4` source conditioned gradients, and zero CPU recomputations. |
| Accepted runtime PIR/CNF gate | `encode_source_pir_cnf_with_gpu_execution_result`, `encode_program_pir_cnf_with_gpu_execution_result`, `encode_source_pir_cnf_for_gpu_execution_results`, `encode_program_pir_cnf_for_gpu_execution_results`, `encode_source_pir_cnf_for_gpu_batch_execution_result`, and `encode_program_pir_cnf_for_gpu_batch_execution_result` reconstruct accepted evidence before calling `GpuPirGraph::from_host`, `GpuPirRoots::from_host`, and `encode_cnf_gpu`, including source/program-specific PIR/CNF trace counters, two-record source and parsed-program PIR/CNF batches, accepted split-batch source/program PIR/CNF gates, same-rule all-operator mixed-membership source PIR/CNF, all-binary split-batch source/program PIR-CNF, single-result quaternary `know fact4/4` source/program PIR-CNF, single-result quaternary `not possible fact4/4` source/program PIR-CNF, single-result quaternary `possible fact4/4` plus `not know fact4/4` source/program PIR-CNF, and quaternary `know`/`not possible fact4/4` plus `possible`/`not know fact4/4` split-batch source/program PIR-CNF. |
| Accepted runtime evaluation gates | `evaluate_with_gpu_execution_result`, `evaluate_for_gpu_execution_results`, `evaluate_for_gpu_batch_execution_result`, `evaluate_gpu_with_grads_with_gpu_execution_result`, `evaluate_gpu_with_grads_for_gpu_execution_results`, and `evaluate_gpu_with_grads_for_gpu_batch_execution_result` reconstruct accepted evidence from accepted GPU runtime results or accepted split-batch runtime evidence before calling the existing `ExactDdnnfProgram::evaluate` and `ExactDdnnfProgram::evaluate_gpu_with_grads` paths, including two-record, accepted split-batch, same-rule all-operator mixed-membership source and parsed-program exact query/gradient evaluation, single-result quaternary `know fact4/4`, single-result quaternary `not possible fact4/4`, and single-result quaternary `possible fact4/4` plus `not know fact4/4` source/program query/gradient batches over one already-compiled exact program, all-binary split-batch, and quaternary `know`/`not possible` plus `possible`/`not know` split-batch query and gradient batches over one already-compiled exact program. |
| CPU probability isolation | `EpistemicProbProductionTrace` records zero CPU-only probability recomputation and zero fixture-circuit evaluations; the source guard rejects `EpistemicCircuit::compile` in the production adapter. |
| Production metric gate | `EpistemicProbProductionTrace::require_production_metric_eligibility` rejects accepted-evidence traces that only record conditioned evidence facts without an aggregate or source/program-specific GPU exact/provenance/PIR/CNF/knowledge-compilation path counter. |
| Incremental circuit fixture | `EpistemicCircuit::apply_accepted_world_view` updates active evidence without changing the circuit fingerprint when the adapter supports incremental evidence, including replacement of stale active evidence for changed accepted assumptions. |
| Compiler adapter | `KnowledgeCompilerAdapter::external_ddnnf_text`, `KnowledgeCompilerAdapter::external_c2d`, and `KnowledgeCompilerAdapter::external_mini_c2d` record alternative Decision-DNNF adapter designs for generic d-DNNF text, c2d, and miniC2D. |
| Numerical stability | `conditional_probability_from_logs` normalizes conditional probabilities with `EPISTEMIC_PROBABILITY_TOLERANCE = 1e-12`. |

## Validation

| Command | Result |
|---|---|
| `cargo fmt --check` | PASS |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_gates_probabilistic_exact_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_gates_probabilistic_program_compile_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_batches_gate_probabilistic_exact_compile_paths -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_gates_probabilistic_end_to_end_knowledge_compilation_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution probabilistic_end_to_end_records_source_and_program_query_trace_counters -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_results_gate_batched_probabilistic_knowledge_compilation_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_results_gate_batched_probabilistic_program_knowledge_compilation_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_conditions_zero_arity_probabilistic_evidence -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_conditions_nonzero_arity_probabilistic_evidence -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_ternary_probabilistic_evidence_records_nonzero_arity_trace -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_quaternary_source_probabilistic_evidence_records_nonzero_arity_trace -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_quaternary_parsed_program_probabilistic_evidence_records_nonzero_arity_trace -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_quaternary_gpu_execution_result_conditions_source_and_program_probabilistic_gradients -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_quaternary_gpu_execution_result_gates_source_and_program_pir_cnf_and_exact_evaluation_paths -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_quaternary_not_possible_probabilistic_evidence_records_negative_nonzero_arity_trace -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_quaternary_not_possible_parsed_program_probabilistic_evidence_records_negative_nonzero_arity_trace -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_quaternary_not_possible_conditions_source_and_program_probabilistic_gradients -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_quaternary_not_possible_gates_source_and_program_pir_cnf_and_exact_evaluation_paths -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_quaternary_possible_and_not_know_results_gate_solver_and_probabilistic_paths -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_quaternary_possible_and_not_know_results_gate_source_conditioned_probabilistic_gradients -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_quaternary_possible_and_not_know_results_gate_parsed_program_probabilistic_paths -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_quaternary_possible_and_not_know_results_gate_source_pir_cnf_and_exact_evaluation_paths -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_quaternary_possible_and_not_know_results_gate_parsed_program_pir_cnf_and_exact_evaluation_paths -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_quaternary_not_possible_batch_conditions_parsed_program_probabilistic_evidence -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_quaternary_possible_and_not_know_batch_gates_solver_and_probabilistic_paths -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_quaternary_possible_and_not_know_batch_gates_probabilistic_gradient_pir_cnf_and_exact_evaluation_paths -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_quaternary_not_possible_batch_gates_probabilistic_gradient_pir_cnf_and_exact_evaluation_paths -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_conditions_negative_nonzero_arity_probabilistic_evidence -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_possible_operator_conditions_probabilistic_evidence -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_not_possible_operator_conditions_negative_probabilistic_evidence -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_binary_possible_operator_conditions_probabilistic_evidence -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_binary_not_possible_operator_conditions_negative_probabilistic_evidence -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_binary_not_know_operator_conditions_negative_probabilistic_evidence -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_g91_and_faeel_modes_gate_probabilistic_production_trace -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_operator_conditions_record_probabilistic_operator_trace_counters -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_all_operator_mixed_membership_conditions_probabilistic_evidence -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_all_operator_mixed_membership_gates_probabilistic_program_gradient_and_pir_paths -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_all_operator_mixed_membership_gates_probabilistic_source_pir_and_exact_evaluation_paths -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_all_operator_mixed_membership_gates_probabilistic_program_exact_evaluation_paths -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution conditioned_probabilistic_evidence_records_source_and_program_trace_counters -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_conditions_parsed_program_probabilistic_evidence -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_conditions_negative_parsed_program_probabilistic_evidence -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_results_gate_batched_conditioned_probabilistic_queries -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_results_gate_batched_negative_conditioned_probabilistic_queries -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_probabilistic_source_and_program_end_to_end_paths -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_probabilistic_conditioned_source_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_all_binary_operator_batch_conditions_probabilistic_evidence -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution rejects_unrecorded_aggregate_kernel_timing -- --nocapture` | PASS, 2 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution rejects_unrecorded_candidate_generation_timing -- --nocapture` | PASS, 2 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution aggregate_timing_requires_every_component_phase_to_be_recorded -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_all_binary_operator_batch_gates_probabilistic_program_and_gradient_paths -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_all_binary_operator_batch_gates_probabilistic_pir_cnf_and_exact_evaluation_paths -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_probabilistic_conditioned_program_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_probabilistic_conditioned_source_gradients -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_probabilistic_conditioned_program_gradients -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_results_gate_batched_conditioned_parsed_program_queries -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_conditions_probabilistic_gradient_evidence -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_results_gate_batched_conditioned_parsed_program_gradients -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution conditioned_probabilistic_gradients_record_source_and_program_trace_counters -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_gates_probabilistic_program_end_to_end_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_gates_probabilistic_pir_cnf_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution probabilistic_pir_cnf_records_source_and_program_trace_counters -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_results_gate_batched_probabilistic_source_pir_cnf_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_results_gate_batched_probabilistic_program_pir_cnf_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_gates_probabilistic_query_evaluation_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_results_gate_batched_probabilistic_query_evaluations -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_gates_probabilistic_gradient_evaluation_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_results_gate_batched_probabilistic_gradient_evaluations -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_probabilistic_pir_cnf_and_exact_evaluation_paths -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-prob --test epistemic_prob_production_reuse` | PASS, 4 passed, 0 failed |
| `cargo test -p xlog-prob --test epistemic_prob_production_reuse production_prob_capabilities_disallow_fixture_circuit_metrics -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-prob --test epistemic_prob_production_reuse production_prob_batch_paths_use_single_gpu_batch_gate -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-prob --test epistemic_prob_production_reuse production_prob_metric_gate_rejects_fixture_only_traces -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-prob --test epistemic_prob c2d_and_minic2d_compiler_adapters_are_explicitly_represented -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-prob --test epistemic_prob changed_assumption_replaces_active_evidence_without_rebuilding_circuit -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-prob --test epistemic_prob` | PASS, 7 passed, 0 failed |
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
| M090_PROB.2 incremental circuit fixture | changed assumption updates circuit without full rebuild where supported | PASS for oracle | `evidence_conditioning_consumes_accepted_world_view` and `changed_assumption_replaces_active_evidence_without_rebuilding_circuit`. |
| M090_PROB.3 compiler adapter | at least one alternative compiler adapter design or implementation | PASS for oracle | `external_ddnnf_text_compiler_adapter_is_explicitly_represented` and `c2d_and_minic2d_compiler_adapters_are_explicitly_represented`. |
| M090_PROB.4 numerical stability | deterministic fixture within documented tolerance | PASS for oracle | `log_space_conditional_probability_is_tolerance_bounded`. |
| M090_PROB.5 evidence conditioning | probabilistic integration consumes accepted world views, not raw unvalidated guesses | PARTIAL | `AcceptedWorldViewEvidence` requires an `EpistemicWorldView` for oracle fixtures and can be constructed from one or more accepted GPU runtime results after stable tuple-source, kernel-trace, transfer-budget, non-empty final-output, and runtime epistemic-mode checks. The conditioned exact path consumes accepted zero-arity and concrete nonzero-arity tuple assumptions, including true `know`, true `possible`, false `possible`/`not possible`, and false `know`/`not know` operator evidence, as parsed exact evidence for source, parsed-program, two-record positive source-batch, two-record negative source-batch, two-record parsed-program-batch, split-batch source/program compile-evaluate inputs, split-batch conditioned source and parsed-program query/gradient inputs, four-component all-binary-operator split-batch conditioned source, parsed-program, source-gradient, and parsed-program-gradient inputs, split-batch quaternary `not possible` parsed-program query plus source-gradient and parsed-program-gradient inputs, split-batch quaternary `possible`/`not know` source query plus source-gradient and parsed-program-gradient inputs, source-vs-parsed-program trace, conditioned source-gradient, conditioned parsed-program-gradient, source-vs-parsed-program conditioned-gradient trace, ternary and quaternary nonzero-arity source evidence, negated quaternary `not possible` source evidence, two-record quaternary `possible`/`not know` source evidence, two-record quaternary `possible`/`not know` parsed-program query/gradient evidence, quaternary nonzero-arity parsed-program evidence, negated quaternary `not possible` parsed-program evidence, and four-record operator-trace inputs and records accepted-assumption, total evidence-fact, negative evidence-fact, nonzero-arity evidence-fact, maximum evidence-arity, split-batch evidence, split-batch component evidence, source/program-specific conditioned-gradient, source/program-specific conditioned-evidence, aggregate operator-specific, source/program-specific operator-conditioned, and mode-specific G91/FAEEL evidence counters. |
| M090_PROB.6 GPU exact integration | accepted world-view evidence updates the GPU-native exact/provenance path | PARTIAL | Accepted GPU runtime evidence gates `ExactDdnnfProgram::compile_source_with_gpu`, `ExactDdnnfProgram::compile_from_program`, `evaluate`, `evaluate_gpu_with_grads`, two-record and accepted split-batch direct source/program exact compilation, source plus parsed-program compile-plus-query-evaluation through the same exact state with source/program-specific query counters, two-record source and parsed-program batch compile/evaluate, accepted split-batch source and parsed-program compile/evaluate after aggregate GPU batch trace validation, source plus parsed-program zero-arity/concrete nonzero-arity true and false exact evidence conditioning including ternary arity-three source evidence, quaternary arity-four source evidence, negated quaternary arity-four source evidence, two-record quaternary `possible`/`not know` arity-four source evidence, two-record quaternary `possible`/`not know` arity-four parsed-program query/gradient evidence, single-result quaternary `not possible` arity-four source/program PIR/CNF plus already-compiled exact query/gradient gates, two-record quaternary `possible`/`not know` arity-four source/program PIR/CNF plus already-compiled exact query/gradient gates, split-batch quaternary `possible`/`not know` arity-four source evidence, quaternary arity-four parsed-program evidence, negated quaternary arity-four parsed-program evidence, and split-batch negated quaternary arity-four parsed-program/source-gradient/parsed-program-gradient evidence, true `know`, true `possible`, false `possible`/`not possible`, and false `know`/`not know` operator results, split-batch conditioned source and parsed-program query/gradient evaluation after aggregate GPU batch trace validation, all-binary-operator split-batch conditioned source and parsed-program query plus source and parsed-program gradient evaluation after aggregate GPU batch trace validation, quaternary `know`/`not possible fact4/4` and `possible`/`not know fact4/4` split-batch source and parsed-program gradient evaluation after aggregate GPU batch trace validation, source/program-specific PIR/CNF upload and encode counters, accepted split-batch source/program PIR/CNF encoding after aggregate GPU batch trace validation, quaternary `know`/`not possible` and `possible`/`not know` split-batch source/program PIR-CNF encoding, two-record positive and negative conditioned source query batches, two-record conditioned parsed-program query batches, conditioned source plus parsed-program gradient evaluation with source/program-specific gradient counters, mode-specific accepted G91/FAEEL production trace accounting, source/program-specific conditioned evidence counters, aggregate/source/program nonzero-arity evidence counters and max evidence arity, aggregate and source/program-specific operator-conditioned evidence counters, two-record query/gradient evaluation batches over one already-compiled exact program, and accepted split-batch plus quaternary `know`/`not possible` and `possible`/`not know` split-batch query/gradient evaluation over one already-compiled exact program. Broader probabilistic semantic coverage is still missing. |
| M090_PROB.7 CPU recompute ban | accepted probabilistic epistemic path records zero CPU-only probability recomputation | PARTIAL | Production trace records zero CPU-only recomputation and zero fixture-circuit counters for accepted runtime source-compile, parsed-program compile, two-record and accepted split-batch direct source/program exact compile, PIR/CNF encoding, query-evaluation, gradient-evaluation, source plus parsed-program end-to-end compile/evaluate paths with source/program-specific exact-query counters, two-record source and parsed-program batch compile/evaluate, accepted split-batch source and parsed-program compile/evaluate with aggregate batch/component counters, split-batch conditioned source and parsed-program query/gradient evaluation with aggregate batch/component counters, all-binary-operator split-batch conditioned source and parsed-program query plus source and parsed-program gradient evaluation with aggregate batch/component counters, split-batch quaternary `not possible` parsed-program query plus source/program gradient evaluation with aggregate batch/component counters, split-batch quaternary `possible`/`not know` source query plus source/program gradient evaluation with aggregate batch/component counters, source/program-specific PIR/CNF upload and encode accounting, two-record and accepted split-batch source/program PIR/CNF encoding including single-result quaternary `not possible` source/program PIR-CNF, two-record quaternary `possible`/`not know` source/program PIR-CNF, and quaternary `know`/`not possible` and `possible`/`not know` split-batch source/program PIR-CNF, source plus parsed-program zero-arity/concrete nonzero-arity true and false conditioned evaluation including arity-three source, arity-four source, negated arity-four source, two-record quaternary `possible`/`not know` source evidence, two-record quaternary `possible`/`not know` parsed-program query/gradient evidence, split-batch quaternary `possible`/`not know` source evidence, arity-four parsed-program, and negated arity-four parsed-program trace accounting, true `know`, true `possible`, false `possible`/`not possible`, and false `know`/`not know` operator-result conditioning, source/program-specific conditioned evidence accounting, aggregate/source/program nonzero-arity evidence and max-arity accounting, source/program-specific operator-conditioned evidence accounting, source/program-specific conditioned gradient accounting, four-record operator-specific trace accounting, two-record positive and negative conditioned source query batches, two-record conditioned parsed-program query batches, conditioned source plus parsed-program gradient evaluation, mode-specific accepted G91/FAEEL evidence accounting, and two-record plus accepted split-batch exact query/gradient evaluation batches including single-result quaternary `not possible` source/program exact query/gradient evaluation, two-record quaternary `possible`/`not know` source/program exact query/gradient evaluation, and quaternary `know`/`not possible` and `possible`/`not know` split-batch exact query/gradient evaluation; full probabilistic execution traces are missing. |
| M090_PROB.8 production prob reuse | accepted probabilistic fixtures execute through existing GPU exact/provenance/PIR/knowledge-compilation APIs | PARTIAL | Source guard and integration fixtures prove accepted GPU runtime evidence compiles source and parsed programs, performs two-record and accepted split-batch direct source/program exact compilation, performs source and parsed-program bounded compile/evaluate knowledge-compilation through `ExactDdnnfProgram` with distinct trace counters including source/program-specific exact-query counters, performs two-record accepted source and parsed-program batch compile/evaluate, validates accepted split-batch execution evidence through the single `accepted_world_views_from_gpu_batch_execution_evidence` gate before routing each component through unconditioned source and parsed-program exact compile, compile/evaluate, conditioned source and parsed-program query/gradient evaluation, all-binary-operator conditioned source and parsed-program query plus source and parsed-program gradient evaluation, split-batch quaternary `not possible` parsed-program query plus source/program gradient evaluation, split-batch quaternary `possible`/`not know` source query plus source/program gradient evaluation, source/program PIR/CNF encoding including single-result quaternary `not possible` source/program PIR-CNF, two-record quaternary `possible`/`not know` source/program PIR-CNF, and quaternary `know`/`not possible` and `possible`/`not know` split-batch source/program PIR-CNF, and already-compiled exact query/gradient evaluation including single-result quaternary `not possible` source/program exact query/gradient evaluation, two-record quaternary `possible`/`not know` source/program exact query/gradient evaluation, and quaternary `know`/`not possible` plus `possible`/`not know` split-batch exact query/gradient evaluation, conditions zero-arity and concrete nonzero-arity true and false evidence via parsed `Evidence` AST entries for source, parsed-program, ternary arity-three source evidence, quaternary arity-four source evidence, negated quaternary arity-four source evidence, two-record quaternary `possible`/`not know` source evidence, two-record quaternary `possible`/`not know` parsed-program query/gradient evidence, split-batch quaternary `possible`/`not know` source evidence, quaternary arity-four parsed-program evidence, negated quaternary arity-four parsed-program evidence, split-batch negated quaternary parsed-program/source-gradient/parsed-program-gradient evidence, split-batch possible/not-know source-gradient/parsed-program-gradient evidence, operator-level true `know`, true `possible`, false `possible`/`not possible`, false `know`/`not know`, two-record positive and negative source-batch, two-record parsed-program-batch, split-batch conditioned source and parsed-program, four-component all-binary-operator split-batch conditioned source, parsed-program, source-gradient, and parsed-program-gradient inputs, source-vs-parsed-program trace, conditioned source-gradient, conditioned parsed-program-gradient, source-vs-parsed-program conditioned-gradient trace, and four-record operator-trace inputs, records separate accepted G91 and FAEEL evidence counters plus split-batch, split-batch component, source/program-specific conditioned-gradient, source/program-specific conditioned-evidence, aggregate/source/program nonzero-arity evidence counters, aggregate/source/program max evidence arity, aggregate operator-specific, and source/program-specific operator-conditioned evidence counters, encodes single-record, two-record, and accepted split-batch source/program PIR/CNF through `GpuPirGraph` and `encode_cnf_gpu` with source/program-specific PIR/CNF counters, evaluates single-record, two-record, and accepted split-batch query probabilities, and evaluates single-record, two-record, and accepted split-batch gradients through the existing exact/provenance path. Broader probabilistic coverage is still missing. |
| M090_PROB.9 fixture isolation | bounded epistemic probability fixtures are marked oracle-only and cannot satisfy closure metrics | PARTIAL | Evidence docs separate `EpistemicCircuit` fixtures from `EpistemicProbProductionAdapter`; `EpistemicProbProductionCapabilities` disallows fixture circuits for production metrics; `EpistemicProbProductionTrace::require_production_metric_eligibility` rejects traces without accepted world-view evidence, without existing aggregate or source/program-specific GPU exact/provenance/PIR/CNF/knowledge-compilation counters, with conditioned evidence facts alone, or with CPU/fixture recomputation counters. Full probabilistic coverage is still missing, so this is not a G090_PROB close. |

## Coordination Notes

- This file is not release-close evidence for `G090_PROB`.
- Production WFS/provenance still rejects direct epistemic literals.
- The production adapter is partial source/program exact-compile,
  single-record, two-record, and accepted split-batch exact compile,
  single-record, two-record, and accepted split-batch PIR/CNF with
  source/program-specific PIR/CNF trace counters, source/program-specific
  exact-query trace counters, single-record, two-record, and accepted
  split-batch query-evaluation, single-record, two-record, and accepted
  split-batch gradient-evaluation, zero-arity and concrete
  nonzero-arity true and false conditioned source/program evaluation,
  source/program-specific conditioned exact trace counters,
  aggregate/source/program nonzero-arity evidence and max-arity trace counters,
  single-result quaternary `know fact4/4` source/program PIR-CNF and exact
  query/gradient evaluation,
  split-batch quaternary parsed-program not-possible evidence reuse plus
	  source/program gradients, PIR/CNF, and exact query/gradient evaluation,
	  two-record quaternary possible/not-know conditioned source evidence plus
	  parsed-program query/gradient evidence,
	  split-batch quaternary possible/not-know conditioned source evidence plus
	  source/program gradients, PIR/CNF, and exact query/gradient evaluation,
	  source/program-specific operator-conditioned exact trace counters,
  source/program-specific conditioned gradient trace counters,
  all-binary-operator split-batch conditioned source/program query and gradient reuse,
  operator-level true know/possible plus false not-possible/not-know conditioned source evaluation,
  four-record operator-specific exact trace counters,
  mode-specific accepted G91/FAEEL production trace counters,
  source/program bounded compile/evaluate reuse, two-record accepted
  source/program batch reuse, split-batch source/program compile/evaluate
  reuse, split-batch conditioned source/program query and gradient reuse,
  two-record negative conditioned source batch reuse, and
  two-record conditioned source/parsed-program query and gradient batch
  evidence only.
- The external Decision-DNNF, c2d, and miniC2D adapters are design contracts,
  not dispatch paths.
- No pyxlog public API signatures were changed.
- No push, tag, release-board update, or merge was performed.
