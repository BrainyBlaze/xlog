# v0.9.0 Accepted WCOJ Execution Evidence

Date: 2026-05-18

Goal node: `G090_GPU - GPU-Native Runtime And WCOJ Execution`

Metrics: `M090_GPU.2`, `M090_GPU.9`, `M090_GPU.11`, `M090_CERT.9`,
`M090_CERT.10`

Branch: `feat/v090-epistemic-solver-semantics`

## Scope

This artifact records accepted epistemic runtime fixtures that execute through
existing production runtime paths plus planner-surface reuse checks for broader
K-clique coverage. The K5, K7, and K8 fixtures prove WCOJ dispatch through
production counters; unary, binary, and multi-membership fixtures
prove variable-bound nonzero-arity membership filters final rows on device; the
negated unary fixture proves `not know` filters by absent bound tuple keys on
the same device row-map path. This is not a closure claim for `G090_GPU`,
`G090_CERT`, or `G090_CLOSE`.

## Implementation Evidence

| Requirement | Evidence |
|---|---|
| Executable relation registration | `EpistemicExecutablePlan` now carries `relation_ids`, the predicate-to-`RelId` map produced by the reduced production compiler. Runtime callers can register input buffers against the same IDs used by the reduced plan. |
| Reduced output boundary | `EpistemicReductionPlan` now carries `head_predicate`, and `Executor::execute_epistemic_gpu_execution` materializes from the named relation stored by `execute_plan` rather than from `execute_plan`'s empty sentinel return buffer. |
| Production WCOJ dispatch | `test_epistemic_gpu_wcoj_execution::accepted_epistemic_k5_execution_certifies_production_wcoj_dispatch`, `accepted_epistemic_k7_execution_certifies_production_wcoj_dispatch`, and `accepted_epistemic_k8_execution_certifies_production_wcoj_dispatch` compile epistemic K5/K7/K8 fixtures with `know gate()`, register returned relation IDs, execute the reduced production plans, assert `EpistemicGpuRuntimeWcojCertification::Certified`, and observe `wcoj_clique5_dispatch_count >= 1`, `wcoj_clique7_dispatch_count >= 1`, and `wcoj_clique8_dispatch_count >= 1`. |
| Existing K-clique path reuse | The accepted K5 fixture requires `kclique_wcoj_plan_count == 1`, `sorted_layout_requirement_count == 1`, `helper_split_spec_count == 1`, and stable tuple-source model membership. The accepted K7 and K8 fixtures require `kclique_wcoj_max_arity == 7/8`, full edge-permutation counts `21/28`, nonzero sorted-layout requirements, certified runtime WCOJ counters, and final tuple materialization. All three use the existing K-clique relation layouts, stats snapshots, WCOJ planner metadata, and runtime counter paths from the G38-B/G39 production evidence chain. |
| K7/K8 planner-surface reuse | `test_epistemic_gpu_wcoj_execution::epistemic_k7_k8_reductions_reuse_g39_kclique_planner_preflight_surface` compiles generated epistemic K7 and K8 reductions with `know gate()`, supplies production-style stats snapshots, and requires `kclique_wcoj_max_arity == 7/8`, full `kclique_wcoj_edge_permutation_count` of 21/28, nonzero sorted-layout requirements, and zero planned-hash/CPU-fallback counters. The accepted K7 and K8 runtime fixtures now prove those generated reductions also reach production dispatch counters. |
| Accepted final tuple materialization | The final epistemic output row count is read from device metadata and must be `1`, proving that the accepted world-view path materializes the production WCOJ row rather than only proving preflight metadata. |
| Accepted nonzero-arity membership | Unary fixture `accepted(X) :- node(X), know edge(X)` returns `[1]`; binary fixture `accepted(X, Y) :- pair(X, Y), know edge(X, Y)` returns `[(1, 2)]`; multi-membership fixture `accepted(X) :- node(X), know edge(X), know color(X)` returns `[2]`, proving bound tuple-key membership filters rows by all variable-bound tuple keys rather than accepting every reduced tuple. |
| Accepted `not know` nonzero-arity membership | Negated unary fixture `accepted(X) :- node(X), not know edge(X)` with `node = [1, 2, 3]` and `edge = [1, 3]` returns `[2]`, proving the final device row-map carries binding polarity and keeps rows whose bound tuple key is absent from the stable-model tuple source. |

## Validation

| Command | Result |
|---|---|
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution -- --nocapture` | PASS, 32 passed, 0 failed |

## Non-Closure Notes

- This satisfies accepted WCOJ dispatch evidence for K5, K7, and K8 epistemic
  reductions.
- It retains K7/K8 planner/preflight reuse evidence for the G39 W6.4 K-clique
  template surface and now pairs K8 metadata with accepted runtime dispatch.
- It satisfies accepted unary, binary, multi-membership, and negated `not know`
  variable-bound nonzero-arity membership fixtures.
- It does not prove the full G91, FAEEL, GPT, and splitting semantic parity
  matrix.
- It does not close full MaxSAT encoding/search coverage or broader accepted
  query-conditioned probabilistic integration.
- No closure-board edit, merge, push, or tag is implied.
