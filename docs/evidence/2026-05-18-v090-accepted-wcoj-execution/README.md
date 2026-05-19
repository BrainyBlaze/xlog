# v0.9.0 Accepted WCOJ Execution Evidence

Date: 2026-05-18

Goal node: `G090_GPU - GPU-Native Runtime And WCOJ Execution`

Metrics: `M090_GPU.2`, `M090_GPU.9`, `M090_GPU.11`, `M090_CERT.9`,
`M090_CERT.10`

Branch: `feat/v090-epistemic-solver-semantics`

## Scope

This artifact records accepted epistemic runtime fixtures that execute through
existing production runtime paths plus preflight-only reuse checks for broader
K-clique planner coverage. The K5 fixture proves WCOJ dispatch; the K7/K8
fixture proves reduced epistemic programs reuse the G39 K-clique planner
surface through runtime preflight; unary, binary, and multi-membership fixtures
prove variable-bound nonzero-arity membership filters final rows on device; the
negated unary fixture proves `not know` filters by absent bound tuple keys on
the same device row-map path. This is not a closure claim for `G090_GPU`,
`G090_CERT`, or `G090_CLOSE`.

## Implementation Evidence

| Requirement | Evidence |
|---|---|
| Executable relation registration | `EpistemicExecutablePlan` now carries `relation_ids`, the predicate-to-`RelId` map produced by the reduced production compiler. Runtime callers can register input buffers against the same IDs used by the reduced plan. |
| Reduced output boundary | `EpistemicReductionPlan` now carries `head_predicate`, and `Executor::execute_epistemic_gpu_execution` materializes from the named relation stored by `execute_plan` rather than from `execute_plan`'s empty sentinel return buffer. |
| Production WCOJ dispatch | `test_epistemic_gpu_wcoj_execution::accepted_epistemic_k5_execution_certifies_production_wcoj_dispatch` compiles an epistemic K5 fixture with `know gate()`, registers the returned relation IDs, executes the reduced production plan, and asserts `EpistemicGpuRuntimeWcojCertification::Certified`. |
| Existing K-clique path reuse | The accepted fixture requires `wcoj_clique5_dispatch_count >= 1`, `kclique_wcoj_plan_count == 1`, `sorted_layout_requirement_count == 1`, `helper_split_spec_count == 1`, and stable tuple-source model membership. It uses the same K5 relation layouts, stats snapshot, WCOJ planner metadata, helper-split metadata, and runtime counter path as the G38-B/G39 production K-clique evidence. |
| K7/K8 planner-surface reuse | `test_epistemic_gpu_wcoj_execution::epistemic_k7_k8_reductions_reuse_g39_kclique_planner_preflight_surface` compiles generated epistemic K7 and K8 reductions with `know gate()`, supplies production-style stats snapshots, and requires `kclique_wcoj_max_arity == 7/8`, full `kclique_wcoj_edge_permutation_count` of 21/28, nonzero sorted-layout requirements, and zero planned-hash/CPU-fallback counters. This is compile/preflight evidence only, not accepted K7/K8 runtime dispatch. |
| Accepted final tuple materialization | The final epistemic output row count is read from device metadata and must be `1`, proving that the accepted world-view path materializes the production WCOJ row rather than only proving preflight metadata. |
| Accepted nonzero-arity membership | Unary fixture `accepted(X) :- node(X), know edge(X)` returns `[1]`; binary fixture `accepted(X, Y) :- pair(X, Y), know edge(X, Y)` returns `[(1, 2)]`; multi-membership fixture `accepted(X) :- node(X), know edge(X), know color(X)` returns `[2]`, proving bound tuple-key membership filters rows by all variable-bound tuple keys rather than accepting every reduced tuple. |
| Accepted `not know` nonzero-arity membership | Negated unary fixture `accepted(X) :- node(X), not know edge(X)` with `node = [1, 2, 3]` and `edge = [1, 3]` returns `[2]`, proving the final device row-map carries binding polarity and keeps rows whose bound tuple key is absent from the stable-model tuple source. |

## Validation

| Command | Result |
|---|---|
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution -- --nocapture` | PASS, 22 passed, 0 failed |

## Non-Closure Notes

- This satisfies the specific WCOJ dispatch-evidence gap for one accepted
  WCOJ-eligible epistemic reduction.
- It adds K7/K8 planner/preflight reuse evidence for the G39 W6.4 K-clique
  template surface, without claiming K7/K8 accepted runtime dispatch.
- It satisfies accepted unary, binary, multi-membership, and negated `not know`
  variable-bound nonzero-arity membership fixtures.
- It does not prove the full G91, FAEEL, GPT, and splitting semantic parity
  matrix.
- It does not close broader multi-candidate solver lifecycle or broader
  accepted probabilistic knowledge-compilation integration.
- No closure-board edit, merge, push, or tag is implied.
