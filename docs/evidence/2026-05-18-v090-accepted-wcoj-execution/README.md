# v0.9.0 Accepted WCOJ Execution Evidence

Date: 2026-05-18

Goal node: `G090_GPU - GPU-Native Runtime And WCOJ Execution`

Metrics: `M090_GPU.2`, `M090_GPU.11`, `M090_CERT.9`

Branch: `feat/v090-epistemic-solver-semantics`

## Scope

This artifact records the first accepted epistemic runtime fixture that executes
an eligible reduced rule through the existing production WCOJ path. It is not a
closure claim for `G090_GPU`, `G090_CERT`, or `G090_CLOSE`.

## Implementation Evidence

| Requirement | Evidence |
|---|---|
| Executable relation registration | `EpistemicExecutablePlan` now carries `relation_ids`, the predicate-to-`RelId` map produced by the reduced production compiler. Runtime callers can register input buffers against the same IDs used by the reduced plan. |
| Reduced output boundary | `EpistemicReductionPlan` now carries `head_predicate`, and `Executor::execute_epistemic_gpu_execution` materializes from the named relation stored by `execute_plan` rather than from `execute_plan`'s empty sentinel return buffer. |
| Production WCOJ dispatch | `test_epistemic_gpu_wcoj_execution::accepted_epistemic_k5_execution_certifies_production_wcoj_dispatch` compiles an epistemic K5 fixture with `know gate()`, registers the returned relation IDs, executes the reduced production plan, and asserts `EpistemicGpuRuntimeWcojCertification::Certified`. |
| Existing K-clique path reuse | The accepted fixture requires `wcoj_clique5_dispatch_count >= 1`, `kclique_wcoj_plan_count == 1`, and stable tuple-source model membership. It uses the same K5 relation layouts, stats snapshot, WCOJ planner metadata, and runtime counter path as the G38-B/G39 production K-clique evidence. |
| Accepted final tuple materialization | The final epistemic output row count is read from device metadata and must be `1`, proving that the accepted world-view path materializes the production WCOJ row rather than only proving preflight metadata. |

## Validation

| Command | Result |
|---|---|
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution -- --nocapture` | PASS, 1 passed, 0 failed |

## Non-Closure Notes

- This satisfies the specific WCOJ dispatch-evidence gap for one accepted
  WCOJ-eligible epistemic reduction.
- It does not prove the full G91, FAEEL, GPT, and splitting semantic parity
  matrix.
- It does not close GPU-native MaxSAT/portfolio or accepted probabilistic
  runtime integration.
- No closure-board edit, merge, push, or tag is implied.
