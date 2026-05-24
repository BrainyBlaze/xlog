# v0.9.0 G090_SPLIT Evidence

Date: 2026-05-18

Goal node: `G090_SPLIT - Epistemic Splitting`

Branch: `feat/v090-epistemic-solver-semantics`

Predecessor evidence:

- `docs/evidence/2026-05-18-v090-pre/README.md`
- `docs/evidence/2026-05-18-v090-eir/README.md`
- `docs/evidence/2026-05-18-v090-g91/README.md`
- `docs/evidence/2026-05-18-v090-faeel/README.md`
- `docs/evidence/2026-05-18-v090-gpt/README.md`

## Implementation Summary

| Requirement | Evidence |
|---|---|
| Deterministic graph | `build_epistemic_dependency_graph` builds sorted connected predicate components with source rule indices. Component predicate sets include rule heads, ordinary positive/negated body predicates, epistemic body predicates, and integrity-constraint body predicates, while connectivity is limited to rule-head dependencies and constraints that mention component heads. |
| Valid split fixture | `test_epistemic_split.rs` verifies two independent epistemic rules split into two deterministic components. |
| Shared extensional input reuse | `shared_extensional_inputs_do_not_coalesce_epistemic_split_components` proves independent epistemic rules that both read `node/1` stay split while each component retains its own `node` input predicate. |
| Relational dependency guard | `relational_dependency_coalesces_epistemic_split_components` proves an ordinary dependency between epistemic rules keeps them in one component instead of producing an unsafe independent split certificate. |
| Constraint dependency guard | `constraint_dependency_coalesces_epistemic_split_components` proves an integrity constraint that mentions two split heads coalesces those rules into one component. |
| Constraint ownership | `split_component_constraints_stay_with_own_component` proves split executable lowering keeps an `a`-only integrity constraint with the `a/p` component and does not copy it into the independent `b/q` component. |
| Invalid split rejection | A rule coupling `know p()` and `possible q()` returns `UnsupportedEpistemicConstruct { construct: "epistemic splitting" }`, and `split_multi_membership_modal_coupling_rejects_gpu_batching` proves a multi-membership rule coupling `know edge(X)` and `know color(X)` rejects before unsafe split GPU batching. |
| Recomposition | `EpistemicSplitPlan::recomposed_rule_indices()` sorts component rule indices and equals the unsplit source order in fixtures. |
| GPU split execution | `compile_epistemic_gpu_split_execution` compiles each epistemic split component through the existing `compile_epistemic_gpu_execution_with_stats_snapshot` path, preserving zero fallback counters, component-owned integrity constraints, and reduced production runtime subplans; accepted split integration executes component plans through `execute_epistemic_gpu_execution_batch`, `accepted_split_binary_operator_components_match_gpt_oracles` covers independent binary `possible` and `not possible` split components against GPT oracles, `accepted_split_all_binary_operators_match_gpt_oracles` covers independent binary `know`, `possible`, `not possible`, and `not know` split components in one batch, and `accepted_split_quaternary_not_possible_batch_matches_gpt_oracles` covers independent arity-four `know fact4/4` and `not possible fact4/4` split components against GPT oracles; the traced wrapper `execute_epistemic_gpu_execution_batch_with_trace` aggregates `EpistemicGpuBatchExecutionTrace` counters proving each component delegated to the existing single-plan GPU runtime path with zero CPU recomposition, zero CPU candidate/world-view fallbacks, zero per-candidate host round trips, aggregate `know`/`possible`/`not know`/`not possible` operator counts, and aggregate CUDA-event timing from every component hot-path kernel. |
| Binary operator GPU split fixture | `test_epistemic_gpu_wcoj_execution::accepted_split_binary_operator_components_match_gpt_oracles` executes independent binary `possible edge(X, Y)` and `not possible blocked(X, Y)` split components over shared `pair/2`, maps results by source rule index, compares each component's semantic trace and candidate-index vectors against bounded GPT oracles, returns `[(1, 2), (3, 4)]` for the possible component and `[(1, 2), (2, 3)]` for the not-possible component, records row-filter polarity counts, and keeps aggregate zero CPU recomposition, zero fallback, zero tracked D2H, and zero per-candidate host round trips. |
| All binary operators GPU split fixture | `test_epistemic_gpu_wcoj_execution::accepted_split_all_binary_operators_match_gpt_oracles` executes four independent split components over shared `pair/2`, maps results by source rule index, compares every component's semantic trace and candidate-index vectors against bounded GPT oracles, returns `[(1, 2), (3, 4)]` for `know edge`, `[(2, 3)]` for `possible alt`, `[(1, 2), (2, 3)]` for `not possible blocked`, and `[(2, 3), (3, 4)]` for `not know seen`, records row-filter polarity counts for positive and negated operators, and keeps aggregate `know == 1`, `possible == 1`, `not possible == 1`, `not know == 1`, zero CPU recomposition, zero fallback, zero tracked D2H, zero per-candidate host round trips, and 32 aggregate CUDA-event timing pairs across the four component hot paths. |
| Quaternary operator GPU split fixture | `test_epistemic_gpu_wcoj_execution::accepted_split_quaternary_not_possible_batch_matches_gpt_oracles` executes independent arity-four `know fact4(A, B, C, D)` and `not possible fact4(A, B, C, D)` split components over shared `tuple4/4`, maps results by source rule index, compares every component's semantic trace and candidate-index vectors against bounded GPT oracles, returns `[(2, 3, 4, 5)]` for `know fact4/4` and `[(1, 2, 3, 4), (9, 9, 9, 9)]` for `not possible fact4/4`, records four tuple-key column reads per component, records row-filter polarity counts for positive and negated operators, and keeps aggregate `know == 1`, `not_possible == 1`, zero CPU recomposition, zero fallback, zero tracked D2H, zero per-candidate host round trips, and 16 aggregate CUDA-event timing pairs across the two component hot paths. |
| Unsafe modal-coupling GPU split guard | `test_epistemic_gpu_wcoj_execution::split_multi_membership_modal_coupling_rejects_gpu_batching` pins the split-lowering boundary by rejecting the multi-membership `know edge(X), know color(X)` body in a mixed program that also contains an independent `possible alt(X), not possible blocked(X)` rule; the typed rejection occurs before component batching can execute on the GPU path. |
| Possible-vs-not-known GPU split fixture | `test_epistemic_gpu_wcoj_execution::split_gpu_world_view_distinguishes_absent_possible_from_not_known` executes split `possible edge(X)` and `not know edge(X)` components over the same absent tuple source, returning empty `possible_edge` output and `[1, 2, 3]` `not_known_edge` output with zero CPU candidate/world-view fallback counters and aggregate batch operator counts of `possible == 1` and `not know == 1`. |
| Documentation | `docs/architecture/epistemic-semantics.md` documents the bounded split graph, rejection, and recomposition contract. |

## Validation

| Command | Result |
|---|---|
| `cargo fmt --check` | PASS |
| `cargo test -p xlog-logic --test test_epistemic_split` | PASS, 8 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_gpt` | PASS, 2 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_faeel` | PASS, 3 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_g91` | PASS, 3 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_eir` | PASS, 4 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_components_execute_gpu_runtime_and_match_component_oracles -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_binary_operator_components_match_gpt_oracles -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_all_binary_operators_match_gpt_oracles -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_quaternary_not_possible_batch_matches_gpt_oracles -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution split_multi_membership_modal_coupling_rejects_gpu_batching -- --exact --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution split_gpu_world_view_distinguishes_absent_possible_from_not_known -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-logic --lib` | PASS, 238 passed, 0 failed |
| `cargo check -p xlog-logic -p xlog-ir -p xlog-solve -p xlog-prob` | PASS |
| `cargo check -p pyxlog` | PASS |

## Metric Status

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M090_SPLIT.1 graph construction | deterministic dependency graph | PASS for oracle | Sorted component fixture plus shared-extensional-input reuse, relational dependency coalescing, and integrity-constraint coalescing fixtures. |
| M090_SPLIT.2 valid split fixtures | 100 percent pass | PASS for oracle | `test_epistemic_split`: 8/8 passed. |
| M090_SPLIT.3 invalid split fixtures | typed rejection | PASS for oracle | `UnsupportedEpistemicConstruct` invalid same-rule coupling fixtures, including the multi-membership split GPU batching guard. |
| M090_SPLIT.4 recomposition | recomposed output equals unsplit output on fixtures | PASS for oracle | `recomposed_rule_indices() == [0, 1]` fixtures, including coalesced relational and constraint-dependent components. |
| M090_SPLIT.5 modal coupling guard | fixture with cross-component epistemic dependency is not split | PASS for oracle | Invalid same-rule coupling rejects split, the multi-membership split GPU batching guard rejects unsafe same-rule `know edge(X), know color(X)` coupling before component batching, ordinary dependency between epistemic rules coalesces into one component, and integrity constraints that span split heads coalesce those components. |
| M090_SPLIT.6 GPU split execution | valid split components execute through GPU-native subplans, not CPU-only recomposition | SUPERSEDED | Valid split components now compile through GPU executable subplans using the existing epistemic GPU contract and reduced runtime compiler path while retaining only component-owned integrity constraints; accepted split integration executes two simple `know` components, two binary operator components, a four-component all-binary-operator split batch, and a two-component quaternary `know`/`not possible` split batch through `execute_epistemic_gpu_execution_batch_with_trace`, matches component output rows against tuple-intersection or binary/quaternary operator oracles, compares per-component GPT trace/candidate-index fields, and records aggregate batch traces with zero CPU recomposition steps, zero CPU candidate/world-view fallback counters, zero per-candidate host round trips, aggregate operator counts, and aggregate CUDA-event timing. It proves binary split `possible edge(X, Y)` and `not possible blocked(X, Y)` components over shared extensional input, a shared-`pair/2` four-component batch with aggregate `know_operator_count == 1`, `possible_operator_count == 1`, `not_possible_operator_count == 1`, `not_know_operator_count == 1`, and `aggregate_kernel_timing.cuda_event_pairs == 32`, a shared-`tuple4/4` two-component batch with aggregate `know_operator_count == 1`, `not_possible_operator_count == 1`, four tuple-key column reads per component, and `aggregate_kernel_timing.cuda_event_pairs == 16`, and fail-closed rejection before GPU split batching for unsafe multi-membership modal coupling; it also preserves the world-view distinction where absent `possible edge(X)` yields no rows while `not know edge(X)` yields every node row. Full accepted-runtime semantic parity remains under `G090_GPU`. |

## Coordination Notes

- This is bounded split planning plus executable-subplan lowering and
  accepted-runtime component execution evidence with aggregate zero CPU
  recomposition tracing plus aggregate split-batch CUDA-event timing, not full
  GPU semantic parity for arbitrary split programs.
- No pyxlog public API signatures were changed.
- No push, tag, release-board update, or merge was performed.
