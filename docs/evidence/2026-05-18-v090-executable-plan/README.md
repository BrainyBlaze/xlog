# v0.9.0 G090_GPU Executable-Plan Evidence

Date: 2026-05-18

Goal node: `G090_GPU - GPU-Native Runtime And WCOJ Execution`

Branch: `feat/v090-epistemic-solver-semantics`

## Scope

This slice added the first production-facing lowering route after the explicit
epistemic semantic contract was built. Later runtime evidence adds bounded
candidate-generation, propagation, candidate-buffer validation, and
materialization-staging kernels; this file alone is not a GPU execution claim
and does not close `G090_GPU`.

## Reuse Audit

The prior closure/evidence docs were inspected before finalizing this slice:

| Prior goal | Reuse decision |
|---|---|
| G38 | Reuse the prompt-to-artifact audit style and avoid treating stale proxy gates as closure evidence. |
| G38-B | Reuse the production WCOJ planner surface: `StatsSnapshot`, `MultiwayPlan`, `KCliqueVariableOrder`, sorted-layout requirements, and helper-splitting specs. Do not create a parallel epistemic WCOJ route. |
| G39 | Preserve user-gated board/tag/merge discipline and explicit PASS/PENDING/MISSING audit language. |

## Implementation Summary

| Requirement | Evidence |
|---|---|
| Explicit semantic boundary first | `compile_epistemic_gpu_execution` calls `plan_epistemic_gpu_execution` before any reduced ordinary program is compiled. |
| Production runtime plan route | The reduced ordinary program is compiled through `Compiler::compile_program_with_stats_snapshot`. |
| WCOJ planner reuse | `test_epistemic_executable_plan` proves a WCOJ-eligible reduced body reaches `RirNode::MultiWayJoin`. |
| 38-B K-clique reuse | The stats-aware K5 fixture proves `MultiwayPlan::WcojWithPlan`, `KCliqueVariableOrder`, sorted-layout requirements, and helper-splitting specs are preserved for epistemic reductions. |

## Validation

| Command | Result |
|---|---|
| `git diff --check` | PASS |
| `cargo fmt --check` | PASS |
| `cargo test -p xlog-logic --test test_epistemic_executable_plan` | PASS, 3 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_gpu_plan` | PASS, 3 passed, 0 failed |
| `cargo test -p xlog-runtime --test test_epistemic_gpu_workspace` | PASS, 2 passed, 0 failed |
| `cargo test -p xlog-ir --lib` | PASS, 14 passed, 0 failed |
| `cargo test -p xlog-logic --lib` | PASS, 238 passed, 0 failed |
| `cargo test -p xlog-runtime --lib` | PASS, 125 passed, 0 failed |
| `cargo check -p xlog-logic -p xlog-ir -p xlog-runtime` | PASS |
| `cargo check -p pyxlog` | PASS |

## Metric Status

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M090_EIR.6 production route | accepted epistemic forms have a production lowering route | PARTIAL | Executable route exists after the semantic contract; direct `xlog run` lowering still rejects epistemic literals. |
| M090_GPU.1 production lowering | accepted epistemic fixture runs through production runtime dispatch | PARTIAL | Reduced runtime plan is produced through the production compiler; later runtime evidence launches candidate generation/propagation/candidate validation/model-membership/materialization staging before reduced-plan dispatch, but stable-model validation/final materialization dispatch is not implemented. |
| M090_GPU.2 WCOJ eligibility | at least one epistemic reduction uses the WCOJ planner/path where eligible | PARTIAL | Reduced fixtures reach `RirNode::MultiWayJoin` and K-clique `MultiwayPlan`; no runtime dispatch evidence yet. |
| M090_GPU.4 kernel coverage | kernels cover GPT hot paths | PARTIAL | Later runtime evidence launches candidate-generation, propagation-staging, candidate-buffer validation, model-membership staging, and materialization-staging kernels; stable-model validation/final materialization kernels are missing. |
| M090_GPU.6 launch evidence | nonzero GPU launches and timing | PARTIAL | Later runtime traces record candidate-generation, propagation, candidate-validation, model-membership, and materialization launches with CUDA-event elapsed timing; stable-model validation/final materialization timing is missing. |

## Remaining Blocker

The next runtime slice must add stable-model world-view validation and final
query-result materialization kernels or GPU-backed adapters, then report launch
counters/timings plus zero CPU fallback counters.
