# v0.9.0 Accepted WCOJ Execution Evidence

Date: 2026-05-18

Goal node: `G090_GPU - GPU-Native Runtime And WCOJ Execution`

Metrics: `M090_GPU.2`, `M090_GPU.9`, `M090_GPU.11`, `M090_CERT.9`,
`M090_CERT.10`

Branch: `feat/v090-epistemic-solver-semantics`

## Scope

This artifact records accepted epistemic runtime fixtures that execute through
existing production runtime paths. The K5 fixture proves WCOJ dispatch; unary
and binary fixtures prove variable-bound nonzero-arity membership filters final
rows on device. This is not a closure claim for `G090_GPU`, `G090_CERT`, or
`G090_CLOSE`.

## Implementation Evidence

| Requirement | Evidence |
|---|---|
| Executable relation registration | `EpistemicExecutablePlan` now carries `relation_ids`, the predicate-to-`RelId` map produced by the reduced production compiler. Runtime callers can register input buffers against the same IDs used by the reduced plan. |
| Reduced output boundary | `EpistemicReductionPlan` now carries `head_predicate`, and `Executor::execute_epistemic_gpu_execution` materializes from the named relation stored by `execute_plan` rather than from `execute_plan`'s empty sentinel return buffer. |
| Production WCOJ dispatch | `test_epistemic_gpu_wcoj_execution::accepted_epistemic_k5_execution_certifies_production_wcoj_dispatch` compiles an epistemic K5 fixture with `know gate()`, registers the returned relation IDs, executes the reduced production plan, and asserts `EpistemicGpuRuntimeWcojCertification::Certified`. |
| Existing K-clique path reuse | The accepted fixture requires `wcoj_clique5_dispatch_count >= 1`, `kclique_wcoj_plan_count == 1`, and stable tuple-source model membership. It uses the same K5 relation layouts, stats snapshot, WCOJ planner metadata, and runtime counter path as the G38-B/G39 production K-clique evidence. |
| Accepted final tuple materialization | The final epistemic output row count is read from device metadata and must be `1`, proving that the accepted world-view path materializes the production WCOJ row rather than only proving preflight metadata. |
| Accepted nonzero-arity membership | Unary fixture `accepted(X) :- node(X), know edge(X)` returns `[1]`; binary fixture `accepted(X, Y) :- pair(X, Y), know edge(X, Y)` returns `[(1, 2)]`, proving bound tuple-key membership filters rows rather than accepting every reduced tuple. |

## Validation

| Command | Result |
|---|---|
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution -- --nocapture` | PASS, 9 passed, 0 failed |

## Non-Closure Notes

- This satisfies the specific WCOJ dispatch-evidence gap for one accepted
  WCOJ-eligible epistemic reduction.
- It satisfies accepted unary and binary variable-bound nonzero-arity
  membership fixtures.
- It does not prove the full G91, FAEEL, GPT, and splitting semantic parity
  matrix.
- It does not close GPU-native MaxSAT/portfolio, full solver lifecycle traces,
  or broader accepted probabilistic PIR/knowledge-compilation integration.
- No closure-board edit, merge, push, or tag is implied.
