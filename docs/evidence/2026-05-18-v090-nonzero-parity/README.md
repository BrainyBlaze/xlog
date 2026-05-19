# v0.9.0 Nonzero-Arity Membership Parity Evidence

Date: 2026-05-19

Goal node: `G090_GPU - GPU-Native Runtime And WCOJ Execution`

Metrics: `M090_GPU.9`, `M090_GPU.10`, `M090_CERT.10`

Branch: `feat/v090-epistemic-solver-semantics`

## Scope

This artifact records accepted runtime fixtures proving `know`, `possible`,
`not possible`, multi-membership, missing-required multi-membership, and
negated `not know` variable-bound nonzero-arity tuple membership can filter or
reject final output rows on device. The current slice broadens binary negated
operator parity with `not know edge(X, Y)` so the accepted GPU path now covers
all four binary epistemic operators against bounded GPT trace/candidate-index
oracles. It is not a closure claim for `G090_GPU`, `G090_CERT`, or
`G090_CLOSE`.

## Implementation Evidence

| Requirement | Evidence |
|---|---|
| Bound tuple-key plan metadata | `EpistemicTupleMembershipBinding::bound_output_columns` records the reduced-output head column for each variable tuple-key term and fails closed when variable terms lack a column binding. |
| Device model-membership comparison | `Executor::populate_epistemic_gpu_model_membership_from_tuple_sources` uses the bound output column metadata for generic arity-N tuple-key comparison against existing `CudaBuffer` columns and carries binding polarity for negated epistemic membership. |
| Operator metrics | `EpistemicGpuRuntimePreflight` reports `know_operator_count`, `possible_operator_count`, `not_know_operator_count`, and `not_possible_operator_count` from the accepted executable GPU plan. |
| Device final-row filtering | `epistemic_build_final_tuple_row_map_u8` builds a device row map from accepted model-membership/world-view buffers, tuple-source key comparison, and row-filter polarity before `epistemic_materialize_final_tuple_column_u8` compacts final output columns. `EpistemicGpuFinalTupleMaterializationTrace` records `row_filter_count` and `negated_row_filter_count`. |
| Accepted unary fixture | `test_epistemic_gpu_wcoj_execution::accepted_nonzero_arity_membership_filters_final_rows_by_bound_tuple_key` runs `accepted(X) :- node(X), know edge(X)` with `node = [1, 2]` and `edge = [1]`; the final device output downloads as `[1]`. |
| Accepted possible fixture | `test_epistemic_gpu_wcoj_execution::accepted_possible_nonzero_arity_membership_records_operator_metrics` runs `accepted(X) :- node(X), possible edge(X)` with `node = [1, 2, 3]` and `edge = [2, 3]`; the final device output downloads as `[2, 3]`, preflight records `possible_operator_count == 1`, and the GPU trace/candidate-index fields match a bounded GPT oracle. |
| Accepted not-possible fixture | `test_epistemic_gpu_wcoj_execution::accepted_not_possible_nonzero_arity_membership_records_operator_and_polarity_metrics` runs `accepted(X) :- node(X), not possible edge(X)` with `node = [1, 2, 3]` and `edge = [2]`; the final device output downloads as `[1, 3]`, preflight records `not_possible_operator_count == 1`, final tuple materialization records one negated row filter, and the GPU trace/candidate-index fields match a bounded GPT oracle. |
| Accepted binary fixture | `test_epistemic_gpu_wcoj_execution::accepted_binary_membership_filters_final_rows_by_bound_tuple_key` runs `accepted(X, Y) :- pair(X, Y), know edge(X, Y)` with `pair = [(1, 2), (2, 3)]` and `edge = [(1, 2)]`; the final device output downloads as `[(1, 2)]`, preflight records `know_operator_count == 1`, and the GPU trace/candidate-index fields match a bounded GPT oracle for `edge/2`. |
| Accepted quaternary fixture | `test_epistemic_gpu_wcoj_execution::accepted_quaternary_membership_matches_gpt_oracle_parity` runs `accepted(A, B, C, D) :- tuple4(A, B, C, D), know fact4(A, B, C, D)` with `tuple4 = [(1, 2, 3, 4), (2, 3, 4, 5), (9, 9, 9, 9)]` and `fact4 = [(2, 3, 4, 5)]`; the final device output downloads as `[(2, 3, 4, 5)]`, the tuple-membership binding records arity 4 with all four bound output columns, preflight records `know_operator_count == 1`, and the GPU trace/candidate-index fields match a bounded GPT oracle for `fact4/4` with zero CPU candidate/world-view fallback counters and zero tracked hot-path D2H calls. |
| Accepted binary possible fixture | `test_epistemic_gpu_wcoj_execution::accepted_binary_possible_membership_matches_gpt_oracle_parity` runs `accepted(X, Y) :- pair(X, Y), possible edge(X, Y)` with `pair = [(1, 2), (2, 3), (3, 4)]` and `edge = [(1, 2), (3, 4)]`; the final device output downloads as `[(1, 2), (3, 4)]`, preflight records `possible_operator_count == 1`, and the GPU trace/candidate-index fields match a bounded GPT oracle for `edge/2`. |
| Accepted binary not-possible fixture | `test_epistemic_gpu_wcoj_execution::accepted_binary_not_possible_membership_matches_gpt_oracle_parity` runs `accepted(X, Y) :- pair(X, Y), not possible edge(X, Y)` with `pair = [(1, 2), (2, 3), (3, 4)]` and `edge = [(2, 3)]`; the final device output downloads as `[(1, 2), (3, 4)]`, preflight records `not_possible_operator_count == 1`, final tuple materialization records one negated row filter, and the GPU trace/candidate-index fields match a bounded GPT oracle for `edge/2`. |
| Accepted binary not-know fixture | `test_epistemic_gpu_wcoj_execution::accepted_binary_not_know_membership_matches_gpt_oracle_parity` runs `accepted(X, Y) :- pair(X, Y), not know edge(X, Y)` with `pair = [(1, 2), (2, 3), (3, 4)]` and `edge = [(2, 3)]`; the final device output downloads as `[(1, 2), (3, 4)]`, preflight records `not_know_operator_count == 1`, final tuple materialization records one negated row filter, and the GPU trace/candidate-index fields match a bounded GPT oracle for `edge/2` with zero CPU candidate/world-view fallback counters and zero tracked hot-path D2H calls. |
| Accepted multi-membership fixture | `test_epistemic_gpu_wcoj_execution::accepted_multiple_memberships_filter_final_rows_by_all_bound_tuple_keys` runs `accepted(X) :- node(X), know edge(X), know color(X)`, returns `[2]`, and compares generated/propagated/tested/accepted/rejected counts plus accepted/rejected candidate indices against the bounded GPT oracle for the four-candidate two-literal matrix. |
| Missing-required multi-membership fixture | `test_epistemic_gpu_wcoj_execution::world_view_validation_rejects_candidates_missing_one_required_membership` runs `accepted(X) :- node(X), know edge(X), know color(X)` with no `color` tuple support, rejects all four candidates at the world-view boundary, and leaves final output empty. |
| Accepted negated unary fixture | `test_epistemic_gpu_wcoj_execution::accepted_not_know_nonzero_arity_membership_filters_final_rows_by_absent_bound_tuple_key` runs `accepted(X) :- node(X), not know edge(X)` with `node = [1, 2, 3]` and `edge = [1, 3]`; the final device output downloads as `[2]`, and the GPU trace/candidate-index fields match a bounded GPT oracle. |
| Split possible-vs-not-known fixture | `test_epistemic_gpu_wcoj_execution::split_gpu_world_view_distinguishes_absent_possible_from_not_known` executes split components over `node = [1, 2, 3]` and an empty `edge` tuple source; `possible edge(X)` returns `[]` while `not know edge(X)` returns `[1, 2, 3]` through the accepted GPU runtime path. |
| Existing relation reuse | The fixture registers `EpistemicExecutablePlan::relation_ids`, seeds ordinary runtime relations, executes the reduced production runtime plan, and reads final output from the runtime-owned device buffer. |

## Validation

| Command | Result |
|---|---|
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_nonzero_arity_membership_filters_final_rows_by_bound_tuple_key -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_possible_nonzero_arity_membership_records_operator_metrics -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_not_possible_nonzero_arity_membership_records_operator_and_polarity_metrics -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_binary_membership_filters_final_rows_by_bound_tuple_key -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_quaternary_membership_matches_gpt_oracle_parity -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_binary_possible_membership_matches_gpt_oracle_parity -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_binary_not_possible_membership_matches_gpt_oracle_parity -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_binary_not_know_membership_matches_gpt_oracle_parity -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_multiple_memberships_filter_final_rows_by_all_bound_tuple_keys -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_not_know_nonzero_arity_membership_filters_final_rows_by_absent_bound_tuple_key -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution split_gpu_world_view_distinguishes_absent_possible_from_not_known -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution -- --nocapture` | PASS, 69 passed, 0 failed |
| `cargo test -p xlog-runtime --test test_epistemic_gpu_workspace -- --nocapture` | PASS, 53 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_executable_plan -- --nocapture` | PASS, 6 passed, 0 failed |

## Non-Closure Notes

- This covers unary, binary, and quaternary variable-bound `know` membership fixtures,
  unary and binary variable-bound `possible`, `not possible`, and `not know`
  membership metrics with operator-level GPT trace/candidate-index oracle parity, generic
  arity-N bound-output tuple-key comparison through an accepted `fact4/4`
  fixture, multi-membership
  acceptance with two-literal GPT trace/candidate-index oracle parity,
  missing-required rejection, and unary/binary variable-bound `not know` absent-key
  filtering with operator-level GPT trace/candidate-index oracle parity plus
  split possible-vs-not-known output parity over the same absent tuple source.
- It does not prove the full G91, FAEEL, GPT, splitting, solver, or
  probabilistic parity matrix.
- Multiple-epistemic-literal final-row filters now have focused positive and
  missing-required fixtures, but broader semantic parity still requires more
  coverage.
- No closure-board edit, merge, push, or tag is implied.
