# v0.9.0 Nonzero-Arity Membership Parity Evidence

Date: 2026-05-19

Goal node: `G090_GPU - GPU-Native Runtime And WCOJ Execution`

Metrics: `M090_GPU.9`, `M090_GPU.10`, `M090_CERT.10`

Branch: `feat/v090-epistemic-solver-semantics`

## Scope

This artifact records accepted runtime fixtures proving unary, binary, and
negated unary variable-bound nonzero-arity tuple membership can filter final
output rows on device. It is not a closure claim for `G090_GPU`, `G090_CERT`,
or `G090_CLOSE`.

## Implementation Evidence

| Requirement | Evidence |
|---|---|
| Bound tuple-key plan metadata | `EpistemicTupleMembershipBinding::bound_output_columns` records the reduced-output head column for each variable tuple-key term and fails closed when variable terms lack a column binding. |
| Device model-membership comparison | `Executor::populate_epistemic_gpu_model_membership_from_tuple_sources` uses the bound output column metadata for generic arity-N tuple-key comparison against existing `CudaBuffer` columns and carries binding polarity for `not know`. |
| Device final-row filtering | `epistemic_build_final_tuple_row_map_u8` builds a device row map from accepted model-membership/world-view buffers, tuple-source key comparison, and row-filter polarity before `epistemic_materialize_final_tuple_column_u8` compacts final output columns. |
| Accepted unary fixture | `test_epistemic_gpu_wcoj_execution::accepted_nonzero_arity_membership_filters_final_rows_by_bound_tuple_key` runs `accepted(X) :- node(X), know edge(X)` with `node = [1, 2]` and `edge = [1]`; the final device output downloads as `[1]`. |
| Accepted binary fixture | `test_epistemic_gpu_wcoj_execution::accepted_binary_membership_filters_final_rows_by_bound_tuple_key` runs `accepted(X, Y) :- pair(X, Y), know edge(X, Y)` with `pair = [(1, 2), (2, 3)]` and `edge = [(1, 2)]`; the final device output downloads as `[(1, 2)]`. |
| Accepted negated unary fixture | `test_epistemic_gpu_wcoj_execution::accepted_not_know_nonzero_arity_membership_filters_final_rows_by_absent_bound_tuple_key` runs `accepted(X) :- node(X), not know edge(X)` with `node = [1, 2, 3]` and `edge = [1, 3]`; the final device output downloads as `[2]`. |
| Existing relation reuse | The fixture registers `EpistemicExecutablePlan::relation_ids`, seeds ordinary runtime relations, executes the reduced production runtime plan, and reads final output from the runtime-owned device buffer. |

## Validation

| Command | Result |
|---|---|
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_nonzero_arity_membership_filters_final_rows_by_bound_tuple_key -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_binary_membership_filters_final_rows_by_bound_tuple_key -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_not_know_nonzero_arity_membership_filters_final_rows_by_absent_bound_tuple_key -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-runtime --test test_epistemic_gpu_workspace -- --nocapture` | PASS, 47 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_executable_plan -- --nocapture` | PASS, 3 passed, 0 failed |

## Non-Closure Notes

- This covers unary and binary variable-bound `know` membership fixtures plus
  unary variable-bound `not know` absent-key filtering.
- It does not prove the full G91, FAEEL, GPT, splitting, solver, or
  probabilistic parity matrix.
- Multiple-epistemic-literal final-row filters have a focused positive fixture,
  but broader semantic parity still requires more coverage.
- No closure-board edit, merge, push, or tag is implied.
