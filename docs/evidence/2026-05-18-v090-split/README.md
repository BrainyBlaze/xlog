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
| Deterministic graph | `build_epistemic_dependency_graph` builds sorted predicate components with source rule indices. |
| Valid split fixture | `test_epistemic_split.rs` verifies two independent epistemic rules split into two deterministic components. |
| Invalid split rejection | A rule coupling `know p()` and `possible q()` returns `UnsupportedEpistemicConstruct { construct: "epistemic splitting" }`. |
| Recomposition | `EpistemicSplitPlan::recomposed_rule_indices()` sorts component rule indices and equals the unsplit source order in fixtures. |
| GPU split execution | `compile_epistemic_gpu_split_execution` compiles each epistemic split component through the existing `compile_epistemic_gpu_execution_with_stats_snapshot` path, preserving zero fallback counters and reduced production runtime subplans; accepted split integration executes component plans through `execute_epistemic_gpu_execution_batch`, and the traced wrapper `execute_epistemic_gpu_execution_batch_with_trace` aggregates `EpistemicGpuBatchExecutionTrace` counters proving each component delegated to the existing single-plan GPU runtime path with zero CPU recomposition, zero CPU candidate/world-view fallbacks, zero per-candidate host round trips, and aggregate `know`/`possible`/`not know`/`not possible` operator counts. |
| Possible-vs-not-known GPU split fixture | `test_epistemic_gpu_wcoj_execution::split_gpu_world_view_distinguishes_absent_possible_from_not_known` executes split `possible edge(X)` and `not know edge(X)` components over the same absent tuple source, returning empty `possible_edge` output and `[1, 2, 3]` `not_known_edge` output with zero CPU candidate/world-view fallback counters and aggregate batch operator counts of `possible == 1` and `not know == 1`. |
| Documentation | `docs/architecture/epistemic-semantics.md` documents the bounded split graph, rejection, and recomposition contract. |

## Validation

| Command | Result |
|---|---|
| `cargo fmt --check` | PASS |
| `cargo test -p xlog-logic --test test_epistemic_split` | PASS, 4 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_gpt` | PASS, 2 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_faeel` | PASS, 3 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_g91` | PASS, 3 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_eir` | PASS, 4 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_components_execute_gpu_runtime_and_match_component_oracles -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution split_gpu_world_view_distinguishes_absent_possible_from_not_known -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-logic --lib` | PASS, 238 passed, 0 failed |
| `cargo check -p xlog-logic -p xlog-ir -p xlog-solve -p xlog-prob` | PASS |
| `cargo check -p pyxlog` | PASS |

## Metric Status

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M090_SPLIT.1 graph construction | deterministic dependency graph | PASS for oracle | Sorted component fixture. |
| M090_SPLIT.2 valid split fixtures | 100 percent pass | PASS for oracle | `test_epistemic_split`: 4/4 passed. |
| M090_SPLIT.3 invalid split fixtures | typed rejection | PASS for oracle | `UnsupportedEpistemicConstruct` invalid coupling fixture. |
| M090_SPLIT.4 recomposition | recomposed output equals unsplit output on fixtures | PASS for oracle | `recomposed_rule_indices() == [0, 1]` fixture. |
| M090_SPLIT.5 modal coupling guard | fixture with cross-component epistemic dependency is not split | PASS for oracle | Invalid coupling fixture rejects split. |
| M090_SPLIT.6 GPU split execution | valid split components execute through GPU-native subplans, not CPU-only recomposition | PARTIAL | Valid split components now compile through GPU executable subplans using the existing epistemic GPU contract and reduced runtime compiler path; accepted split integration executes two components through `execute_epistemic_gpu_execution_batch_with_trace`, matches component output rows against simple tuple-intersection oracles, compares per-component GPT trace/candidate-index fields, and records an aggregate batch trace with two GPU runtime component executions, zero CPU recomposition steps, zero CPU candidate/world-view fallback counters, zero per-candidate host round trips, and aggregate operator counts. It also proves the world-view distinction where absent `possible edge(X)` yields no rows while `not know edge(X)` yields every node row, and the batch trace records `possible_operator_count == 1` plus `not_know_operator_count == 1`. Full accepted-runtime semantic parity remains under `G090_GPU`. |

## Coordination Notes

- This is bounded split planning plus executable-subplan lowering and
  accepted-runtime component execution evidence with aggregate zero CPU
  recomposition tracing, not full GPU semantic parity for arbitrary split
  programs.
- No pyxlog public API signatures were changed.
- No push, tag, release-board update, or merge was performed.
