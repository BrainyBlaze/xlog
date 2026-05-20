# v0.9.0 G090_GPT Evidence

Date: 2026-05-18

Goal node: `G090_GPT - Generate-Propagate-Test Execution`

Branch: `feat/v090-epistemic-solver-semantics`

Predecessor evidence:

- `docs/evidence/2026-05-18-v090-pre/README.md`
- `docs/evidence/2026-05-18-v090-eir/README.md`
- `docs/evidence/2026-05-18-v090-g91/README.md`
- `docs/evidence/2026-05-18-v090-faeel/README.md`

## Implementation Summary

| Requirement | Evidence |
|---|---|
| Phase separation | `run_generate_propagate_test` and `run_generate_propagate_test_with_mode` have explicit generate, propagate, and test phases. |
| Trace output | `GeneratePropagateTestTrace` reports generated, guesses, propagated, pruned, reduced-program models, tested, accepted, accepted world views, rejected, and rejection reasons; `GeneratePropagateTestOutcome` records accepted and rejected candidate indices. |
| Correctness fixtures | `test_epistemic_gpt.rs` covers one accepted candidate, one FAEEL-rejected candidate, one propagation-pruned contradiction, and one explicit G91 compatibility-mode candidate set. |
| Bounded behavior | `GeneratePropagateTestConfig::max_candidates` rejects oversized candidate sets with `XlogError::ResourceExhausted`. |
| GPU candidate generation | `epistemic_generate_candidate_assumptions_u8` populates bounded candidate-assumption bitsets in the runtime workspace. |
| GPU propagation staging | `epistemic_propagate_candidates_u8` stages generated candidates into world-view/rejection buffers in the runtime workspace. |
| GPU candidate validation | `epistemic_validate_candidate_bits_u8` validates staged candidate bitsets and world-view activity in the runtime workspace. |
| GPU model-membership staging | Tuple-source kernels write candidate-scoped model-membership bytes from named reduced stable-model relation row counts, compare encoded ground tuple-key expectations against current model-slot relation key-column bytes, and compare variable-bound tuple keys against reduced-output `CudaBuffer` columns in the generic arity-N runtime path. |
| GPU world-view validation staging | `epistemic_validate_world_views_u8` checks staged candidate-assumption and model-membership bytes against active world-view slots, requires every generated candidate assumption for the reduced rule's epistemic literals to have matching tuple-source support, and updates rejection codes. |
| GPU materialization staging | `epistemic_materialize_accepted_candidates_u8` writes accepted-candidate flags from rejection codes into world-view slots. |
| GPU final-result flag staging | `epistemic_materialize_final_result_flags_u8` writes final-result flags from reduced output device row-count metadata and rejection codes into world-view slots. |
| GPU final tuple materialization | `epistemic_materialize_final_tuple_column_u8` writes a device-resident final-output tuple buffer and final row-count metadata from reduced output columns after checking GPU model-membership and world-view buffers for an accepted membership. |
| GPU semantic summary trace | `EpistemicGpuSemanticTrace::from_device_rejection_reasons` reads the bounded device rejection-reason metadata after the hot-path transfer-budget window, decodes nonzero device codes through `EpistemicGpuRejectionReason`, and reports generated, guess, propagated, pruned, tested, reduced-model-slot, accepted-candidate, accepted-world-view, rejected-candidate, accepted/rejected candidate indices, and rejection-reason counts with zero CPU candidate enumeration/world-view validation counters. |
| GPU residency | Candidate-assumption generation, propagation staging, candidate-buffer validation, tuple-source model-membership staging with fixed arity-one/two/three and generic arity-N row-scoped ground-key comparison plus generic arity-N variable-bound comparison, bounded candidate-assumption-aware world-view validation staging, accepted-candidate materialization staging, final-result flag staging, membership-gated final tuple materialization, and post-hot-path semantic trace accounting are backed by GPU workspace/output buffers; the accepted `fact3/3` fixture exercises the specialized arity-three tuple-source path and the accepted `fact4/4` fixture exercises the wider generic arity-N bound-output path, but accepted semantic parity remains a runtime gap. |
| Documentation | `docs/architecture/epistemic-semantics.md` documents the GPT phase contract and guard. |

## Validation

| Command | Result |
|---|---|
| `cargo fmt --check` | PASS |
| `cargo test -p xlog-runtime --test test_epistemic_gpu_workspace candidate_generation` | PASS, 2 passed, 0 failed |
| `cargo test -p xlog-runtime --test test_epistemic_gpu_workspace propagation` | PASS, 3 passed, 0 failed |
| `cargo test -p xlog-runtime --test test_epistemic_gpu_workspace candidate_validation` | PASS, 2 passed, 0 failed |
| `cargo test -p xlog-runtime --test test_epistemic_gpu_workspace validation` | PASS, 7 passed, 0 failed |
| `cargo test -p xlog-runtime --test test_epistemic_gpu_workspace materialization` | PASS, 5 passed, 0 failed |
| `cargo test -p xlog-runtime --test test_epistemic_gpu_workspace` | PASS, 53 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_records_device_semantic_trace_counts -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_semantic_trace_matches_gpt_oracle_rejection_reason -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_multiple_memberships_filter_final_rows_by_all_bound_tuple_keys -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_mixed_memberships_match_gpt_oracle_parity -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_negated_mixed_memberships_match_gpt_oracle_parity -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_all_operator_mixed_memberships_match_gpt_oracle_parity -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_possible_nonzero_arity_membership_records_operator_metrics -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_not_possible_nonzero_arity_membership_records_operator_and_polarity_metrics -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_ternary_membership_matches_gpt_oracle_parity -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_quaternary_membership_matches_gpt_oracle_parity -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_binary_possible_membership_matches_gpt_oracle_parity -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_binary_not_possible_membership_matches_gpt_oracle_parity -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_binary_not_know_membership_matches_gpt_oracle_parity -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_not_know_nonzero_arity_membership_filters_final_rows_by_absent_bound_tuple_key -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution world_view_validation_rejects_candidates_missing_one_required_membership -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_binary_operator_components_match_gpt_oracles -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_all_binary_operators_match_gpt_oracles -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_quaternary_not_possible_batch_matches_gpt_oracles -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution split_multi_membership_modal_coupling_rejects_gpu_batching -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution split_gpu_world_view_distinguishes_absent_possible_from_not_known -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_gpt` | PASS, 3 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_faeel` | PASS, 3 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_g91` | PASS, 3 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_eir` | PASS, 4 passed, 0 failed |
| `cargo test -p xlog-logic --lib` | PASS, 238 passed, 0 failed |
| `cargo check -p xlog-logic -p xlog-ir -p xlog-solve -p xlog-prob` | PASS |
| `cargo check -p pyxlog` | PASS |

## Metric Status

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M090_GPT.1 phase separation | generate, propagate, test boundaries visible in code | PASS for oracle | `run_generate_propagate_test` and mode-aware `run_generate_propagate_test_with_mode` implementation and fixtures. |
| M090_GPT.2 trace output | debug/trace mode reports phase counts and GPU launch counters | PARTIAL | CPU trace counts are asserted; candidate-generation, propagation, candidate-validation, tuple-source model-membership staging with fixed arity-one/two/three and generic arity-N row-scoped ground-key comparison plus generic arity-N variable-bound comparison, candidate-assumption-aware world-view validation, accepted-candidate materialization, final-result flag, and membership-gated final tuple traces each record GPU launches with CUDA-event elapsed timing; `EpistemicGpuSemanticTrace` now reports generated/guess/propagated/pruned/tested/reduced-model-slot/accepted/rejected counts, exact accepted/rejected candidate indices, and typed rejection reasons from the device rejection buffer for accepted runtime fixtures. Full semantic parity coverage is still missing. |
| M090_GPT.3 correctness fixtures | accepted/rejected candidate fixtures pass | PASS for oracle | `test_epistemic_gpt`: 3/3 passed, including G91 compatibility mode. |
| M090_GPT.4 bounded behavior | candidate explosion guard implemented or explicitly scoped | PASS for oracle | `ResourceExhausted` guard fixture. |
| M090_GPT.5 world-view validation | trace records guess count, reduced-program model count, accepted world-view count, and rejection reasons | PARTIAL | CPU trace fields and accepted/rejected candidate indices are asserted in `test_epistemic_gpt`, including explicit G91 compatibility mode; accepted GPU runtime evidence now asserts device-derived generated/guess/tested/reduced-model-slot/accepted-world-view/rejected counts, exact accepted/rejected candidate indices, and typed `UnsatisfiedMembership` rejection reasons for rejected candidates, including a bounded `know edge(X)` parity check against `run_generate_propagate_test`, independently founded FAEEL self-`possible p()` parity against the default oracle, a G91 self-supported `possible p()` parity check against `run_generate_propagate_test_with_mode`, unary nonzero-arity `possible edge(X)`, `not possible edge(X)`, and `not know edge(X)` operator fixtures whose generated/propagated/tested/accepted/rejected counts plus accepted/rejected candidate-index vectors match bounded GPT oracles, binary `possible edge(X, Y)`, `not possible edge(X, Y)`, and `not know edge(X, Y)` operator fixtures whose trace/candidate-index vectors match bounded GPT oracles, a ternary `know fact3(A, B, C)` fixture whose accepted specialized arity-three trace/candidate-index vectors match the bounded GPT oracle, a quaternary `know fact4(A, B, C, D)` fixture whose accepted generic arity-N trace/candidate-index vectors match the bounded GPT oracle, a two-literal `know edge(X), know color(X)` multi-membership fixture whose generated/propagated/tested/accepted/rejected counts plus accepted/rejected candidate-index vectors match the bounded GPT oracle, a mixed `know edge(X), possible alt(X)` multi-membership fixture whose GPU trace/candidate-index vectors match the bounded GPT oracle while recording one `know` and one `possible` operator, a negated mixed `not know edge(X), not possible blocked(X)` multi-membership fixture whose GPU trace/candidate-index vectors match the bounded GPT oracle while recording one `not know`, one `not possible`, and two negated row filters, and an all-operator `know edge(X), possible alt(X), not know hidden(X), not possible blocked(X)` multi-membership fixture whose 16-candidate GPU trace/candidate-index vectors match the bounded GPT oracle while recording one operator from each family and two negated row filters. The missing-required multi-literal fixture still proves candidates missing one required membership are rejected instead of being hidden by final-row filtering. Split GPU runtime evidence also proves independent binary `possible edge(X, Y)` and `not possible blocked(X, Y)` split components, a four-component split batch covering binary `know edge(X, Y)`, `possible alt(X, Y)`, `not possible blocked(X, Y)`, and `not know seen(X, Y)`, and a two-component quaternary split batch covering `know fact4(A, B, C, D)` plus `not possible fact4(A, B, C, D)`, against GPT oracle candidate-index vectors, and fail-closed split lowering rejects unsafe multi-membership modal coupling before GPU batch execution; the world-view distinction where absent `possible edge(X)` rejects while `not know edge(X)` accepts over the same absent tuple source remains covered. Broader semantic parity fixtures are still missing. |
| M090_GPT.6 GPU residency | candidate generation, propagation, and world-view validation hot path uses GPU-resident buffers | PARTIAL | Candidate-assumption generation, propagation staging, candidate-buffer validation, tuple-source model-membership staging with fixed arity-one/two/three and generic arity-N row-scoped ground-key comparison plus generic arity-N variable-bound comparison, bounded candidate-assumption-aware world-view validation staging, accepted-candidate materialization staging, final-result flag staging, and membership-gated final tuple materialization use GPU-resident workspace/output buffers through CUDA kernels; semantic trace counts are derived from bounded device rejection metadata after the hot-path transfer budget. Accepted semantic parity is still not fully wired. |

## Coordination Notes

- Candidate generation, propagation staging, candidate-buffer validation, tuple-source model-membership staging with fixed arity-one/two/three and generic arity-N row-scoped ground-key comparison plus generic arity-N variable-bound comparison, bounded candidate-assumption-aware world-view validation staging, accepted-candidate materialization staging, final-result flag staging, membership-gated final tuple materialization, typed device-derived semantic trace accounting with accepted/rejected candidate indices, and bounded FAEEL/G91/unary-operator/binary-all-operator/ternary-specialized-arity/quaternary-generic-arity/multi-membership/positive-negated-and-all-operator-mixed-membership/all-operator-mixed-membership/split-component/split-binary-operator/all-binary-split-operator/split-quaternary-operator GPU-vs-GPT oracle parity checks are now available for bounded GPU workspace/output paths. Arbitrary EIR world enumeration, broader semantic parity, and full reduced-runtime stable-model test phases remain required production scope.
- No pyxlog public API signatures were changed.
- No push, tag, release-board update, or merge was performed.
