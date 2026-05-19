# v0.9.0 Accepted WCOJ Execution Evidence

Date: 2026-05-18

Goal node: `G090_GPU - GPU-Native Runtime And WCOJ Execution`

Metrics: `M090_GPU.2`, `M090_GPU.9`, `M090_GPU.11`, `M090_CERT.9`,
`M090_CERT.10`

Branch: `feat/v090-epistemic-solver-semantics`

## Scope

This artifact records accepted epistemic runtime fixtures that execute through
existing production runtime paths plus planner-surface reuse checks for broader
K-clique coverage. The K5, K6, K7, and K8 fixtures prove WCOJ dispatch through
production counters; the K5/K6 fixtures additionally prove G38-B K-clique
skew-scheduled helper-split evidence, and K6 proves runtime-histogram metadata count and timing under
accepted epistemic execution; unary, binary, quaternary generic-arity, and multi-membership fixtures
prove variable-bound nonzero-arity membership filters final rows on device; a
`possible` fixture records accepted preflight operator metrics; the negated
unary and binary fixtures prove `not know` and `not possible` filter by absent bound tuple
keys on the same device row-map path, with final-row polarity counts recorded
in the materialization trace; split batch execution now exposes aggregate
zero-CPU-recomposition and zero per-candidate-host-round-trip counters while
delegating every component to the existing single-plan GPU runtime path. This is
not a closure claim for `G090_GPU`,
`G090_CERT`, or `G090_CLOSE`.

## Implementation Evidence

| Requirement | Evidence |
|---|---|
| Executable relation registration | `EpistemicExecutablePlan` now carries `relation_ids`, the predicate-to-`RelId` map produced by the reduced production compiler. Runtime callers can register input buffers against the same IDs used by the reduced plan. |
| Reduced output boundary | `EpistemicReductionPlan` now carries `head_predicate`, and `Executor::execute_epistemic_gpu_execution` materializes from the named relation stored by `execute_plan` rather than from `execute_plan`'s empty sentinel return buffer. |
| Production WCOJ dispatch | `test_epistemic_gpu_wcoj_execution::accepted_epistemic_k5_execution_certifies_production_wcoj_dispatch`, `accepted_epistemic_k6_execution_certifies_g38b_helper_histogram_path`, `accepted_epistemic_k7_execution_certifies_production_wcoj_dispatch`, and `accepted_epistemic_k8_execution_certifies_production_wcoj_dispatch` compile epistemic K5/K6/K7/K8 fixtures with `know gate()`, register returned relation IDs, execute the reduced production plans, assert `EpistemicGpuRuntimeWcojCertification::Certified`, and observe `wcoj_clique5_dispatch_count >= 1`, `wcoj_clique6_dispatch_count >= 1`, `wcoj_clique7_dispatch_count >= 1`, and `wcoj_clique8_dispatch_count >= 1`. The K5 certified trace carries `certified_edge_permutation_slots == 10`, `certified_stream_groups == 1`, `certified_skew_scheduled_plans == 1`, `certified_sorted_layout_requirements == 1`, `certified_helper_split_specs == 1`, `certified_helper_relation_rules == 1`, and `certified_helper_relation_scans == 1`; the K6 certified trace carries `certified_edge_permutation_slots == 15`, nonzero stream-group/skew-scheduled/sorted-layout/helper-split/helper-relation counts, `observed_metadata_builds >= 1`, and `observed_metadata_build_nanos >= 1`. Runtime preflight evidence now fails closed when sorted-layout obligations exist but neither a layout sort nor a layout fast-path event is observed, when helper-split specs exist without production helper relation rules and WCOJ input scans, and when helper scans occur outside the WCOJ plan. |
| Existing K-clique path reuse | The accepted K5 fixture requires `kclique_wcoj_plan_count == 1`, `kclique_stream_group_count == 1`, `kclique_skew_scheduled_plan_count == 1`, `sorted_layout_requirement_count == 1`, `helper_split_spec_count == 1`, `helper_relation_rule_count == 1`, `helper_relation_scan_count == 1`, stable tuple-source model membership, and the same planner/layout/helper/skew-scheduler metrics inside the dispatch-certified WCOJ trace. The runtime WCOJ certificate now rejects dispatch-only evidence without layout sort or layout fast-path counters when sorted-layout requirements are present, and preflight rejects helper metadata without the compiler-created helper relation rewrite. The accepted K6 fixture uses a buried-skew stats snapshot and requires the G38-B skew-scheduled helper-split specs, production helper relation rules and WCOJ input scans, plus runtime K-clique histogram metadata-build counts and nanoseconds to be observed inside the accepted epistemic execution trace. The accepted K7 and K8 fixtures require `kclique_wcoj_max_arity == 7/8`, full edge-permutation counts `21/28`, `kclique_stream_group_count == 1`, nonzero sorted-layout requirements, certified runtime WCOJ counters, and final tuple materialization. All four use the existing K-clique relation layouts, stream groups, stats snapshots, WCOJ planner metadata, and runtime counter paths from the G38-B/G39 production evidence chain. |
| K7/K8 planner-surface reuse | `test_epistemic_gpu_wcoj_execution::epistemic_k7_k8_reductions_reuse_g39_kclique_planner_preflight_surface` compiles generated epistemic K7 and K8 reductions with `know gate()`, supplies production-style stats snapshots, and requires `kclique_wcoj_max_arity == 7/8`, full `kclique_wcoj_edge_permutation_count` of 21/28, `kclique_stream_group_count == 1`, nonzero sorted-layout requirements, and zero planned-hash/CPU-fallback counters. The accepted K7 and K8 runtime fixtures now prove those generated reductions also reach production dispatch counters. |
| Accepted final tuple materialization | The final epistemic output row count is read from device metadata and must be `1`, proving that the accepted world-view path materializes the production WCOJ row rather than only proving preflight metadata. |
| Accepted nonzero-arity membership | Unary fixture `accepted(X) :- node(X), know edge(X)` returns `[1]`; possible fixture `accepted(X) :- node(X), possible edge(X)` returns `[2, 3]` and records `possible_operator_count == 1`; not-possible fixture `accepted(X) :- node(X), not possible edge(X)` returns `[1, 3]`, records `not_possible_operator_count == 1`, and records `row_filter_count == 1` plus `negated_row_filter_count == 1`; binary fixture `accepted(X, Y) :- pair(X, Y), know edge(X, Y)` returns `[(1, 2)]`; quaternary fixture `accepted(A, B, C, D) :- tuple4(A, B, C, D), know fact4(A, B, C, D)` returns `[(2, 3, 4, 5)]`; multi-membership fixture `accepted(X) :- node(X), know edge(X), know color(X)` returns `[2]` and exactly one accepted world view, proving bound tuple-key membership filters rows by all variable-bound tuple keys rather than accepting every reduced tuple. Negative multi-membership fixture `world_view_validation_rejects_candidates_missing_one_required_membership` leaves final output empty and rejects all candidates when one required membership has no tuple-source support. |
| Accepted `not know` nonzero-arity membership | Negated unary fixture `accepted(X) :- node(X), not know edge(X)` with `node = [1, 2, 3]` and `edge = [1, 3]` returns `[2]`; binary fixture `accepted(X, Y) :- pair(X, Y), not know edge(X, Y)` with `edge = [(2, 3)]` returns `[(1, 2), (3, 4)]`, proving the final device row-map carries binding polarity and keeps rows whose bound tuple key is absent from the stable-model tuple source. |
| Split batch runtime trace | `test_epistemic_gpu_wcoj_execution::accepted_split_components_execute_gpu_runtime_and_match_component_oracles` uses `execute_epistemic_gpu_execution_batch_with_trace` and `EpistemicGpuBatchExecutionTrace` to prove two split components executed through the existing single-plan GPU runtime path, not CPU recomposition, while preserving per-component GPT trace/candidate-index parity, zero CPU candidate/world-view fallbacks, zero tracked hot-path D2H calls, and zero per-candidate host round trips. |

## Validation

| Command | Result |
|---|---|
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_binary_not_know_membership_matches_gpt_oracle_parity -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution -- --nocapture` | PASS, 76 passed, 0 failed |

## Non-Closure Notes

- This satisfies accepted WCOJ dispatch evidence for K5, K6, K7, and K8 epistemic
  reductions.
- It adds accepted K5/K6 evidence for G38-B skew-scheduled helper-split reuse,
  production helper relation rewrites, and K6 timed runtime histogram metadata
  path.
- It retains K7/K8 planner/preflight reuse evidence for the G39 W6.4 K-clique
  template surface and now pairs K8 metadata with accepted runtime dispatch.
- It satisfies accepted unary, possible-operator, not-possible-operator,
  binary all-operator, quaternary generic-arity, multi-membership,
  missing-required multi-membership, and negated `not know` variable-bound
  nonzero-arity membership fixtures.
- It adds batch-level split execution trace counters, but does not close the
  full splitting semantic parity matrix.
- It does not prove the full G91, FAEEL, GPT, and splitting semantic parity
  matrix.
- It does not close broader solver semantic integration or broader accepted
  probabilistic integration.
- No closure-board edit, merge, push, or tag is implied.
