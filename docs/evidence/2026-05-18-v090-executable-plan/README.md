# v0.9.0 G090_GPU Executable-Plan Evidence

Date: 2026-05-18

Goal node: `G090_GPU - GPU-Native Runtime And WCOJ Execution`

Branch: `feat/v090-epistemic-solver-semantics`

## Scope

This slice added the first production-facing lowering route after the explicit
epistemic semantic contract was built. Later runtime evidence adds bounded
candidate-generation, propagation, candidate-buffer validation,
model-membership, world-view-validation, materialization-staging, and
final-result flag plus final tuple materialization kernels; this file alone is
not a GPU execution claim and does not close `G090_GPU`.

## Reuse Audit

The prior closure/evidence docs were inspected before finalizing this slice:

| Prior goal | Reuse decision |
|---|---|
| G38 | Reuse the prompt-to-artifact audit style and avoid treating stale proxy gates or superseded evidence as closure evidence. |
| G38-B | Reuse the production WCOJ planner surface: `StatsSnapshot`, `MultiwayPlan`, `KCliqueVariableOrder`, sorted-layout requirements, runtime histogram metadata, cost-gated hash routing, and helper-splitting specs. Do not create a parallel epistemic WCOJ route. |
| G39 | Reuse the existing production substrate for chain dispatch, K7/K8 templates, sort labels, DLPack/zero-transfer discipline, CUDA Graphs, and DTS replay certification only when the v0.9 epistemic path actually touches those surfaces. Preserve user-gated board/tag/merge discipline and explicit PASS/PENDING/MISSING audit language. |

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
| `cargo test -p xlog-logic --test test_epistemic_gpu_plan` | PASS, 8 passed, 0 failed |
| `cargo test -p xlog-runtime --test test_epistemic_gpu_workspace` | PASS, 44 passed, 0 failed |
| `cargo test -p xlog-ir --lib` | PASS, 14 passed, 0 failed |
| `cargo test -p xlog-logic --lib` | PASS, 238 passed, 0 failed |
| `cargo test -p xlog-runtime --lib` | PASS, 128 passed, 0 failed |
| `cargo check -p xlog-logic -p xlog-ir -p xlog-runtime` | PASS |
| `cargo check -p pyxlog` | PASS |

## Metric Status

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M090_EIR.6 production route | accepted epistemic forms have a production lowering route | PARTIAL | Executable route exists after the semantic contract; direct `xlog run` lowering still rejects epistemic literals. |
| M090_GPU.1 production lowering | accepted epistemic fixture runs through production runtime dispatch | PARTIAL | Reduced runtime plan is produced through the production compiler; later runtime evidence launches candidate generation/propagation/candidate validation before reduced-plan dispatch, then arity 0-3 tuple-source model-membership staging with row-scoped ground key comparison for arity one/two/three, world-view-validation/materialization, final-result flag staging, and final tuple materialization; bound-variable tuple-key matching, arbitrary arity, and full accepted semantics remain missing. |
| M090_GPU.2 WCOJ eligibility | at least one epistemic reduction uses the WCOJ planner/path where eligible | PARTIAL | Reduced fixtures reach `RirNode::MultiWayJoin` and K-clique `MultiwayPlan`; later runtime evidence fails closed when a required K-clique WCOJ plan lacks counter deltas, but no certified successful dispatch evidence exists yet. |
| M090_GPU.4 kernel coverage | kernels cover GPT hot paths | PARTIAL | Later runtime evidence launches candidate-generation, propagation-staging, candidate-buffer validation, arity 0-3 tuple-source model-membership staging with fixed arity-one/two/three row-scoped ground key comparison, world-view-validation staging, materialization-staging, final-result flag, and final tuple materialization kernels; bound-variable and arbitrary-arity tuple matching remain missing. |
| M090_GPU.6 launch evidence | nonzero GPU launches and timing | PARTIAL | Later runtime traces record candidate-generation, propagation, candidate-validation, arity 0-3 tuple-source model-membership staging with fixed arity-one/two/three row-scoped ground key comparison, world-view-validation, accepted-candidate materialization, final-result flag, and final tuple materialization launches with CUDA-event elapsed timing; accepted semantic parity timing is missing. |

## Remaining Blocker

The next runtime slice must extend fixed arity 0-3 ground-key matching to
bound-value tuple keys and arbitrary arity, semantically gate final query
results with that membership, then report launch counters/timings plus zero CPU
fallback counters.
