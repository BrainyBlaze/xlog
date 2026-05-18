# v0.9.0 G090_GPU Runtime Preflight, Workspace Reset, Counter-Guard, And Trace Evidence

Date: 2026-05-18

Goal node: `G090_GPU - GPU-Native Runtime And WCOJ Execution`

Branch: `feat/v090-epistemic-solver-semantics`

## Scope

This slice connects `EpistemicExecutablePlan` to `xlog-runtime` preflight, adds
a device-side workspace reset trace, adds bounded GPU candidate generation,
adds bounded GPU propagation staging, adds a certification guard tying WCOJ
evidence to actual production counter deltas, and exposes a reduced-plan
execution trace around `execute_plan`. It is still incomplete for the epistemic
hot path and does not close `G090_GPU`.

## Implementation Summary

| Requirement | Evidence |
|---|---|
| Runtime consumes executable plan | `EpistemicGpuRuntimePreflight::for_executable_plan` accepts `EpistemicExecutablePlan`. |
| Workspace layout tied to executable plan | Preflight computes `EpistemicGpuWorkspaceLayout` from the GPU contract and capacity limits. |
| CPU fallback ban starts at runtime boundary | Preflight rejects nonzero forbidden CPU fallback counters with typed `UnsupportedEpistemicConstruct`. |
| WCOJ route metadata inspected | Preflight records reduced rule count, `MultiWayJoin` count, K-clique WCOJ plan count, planned-hash count, sorted-layout requirement count, and helper-split spec count. |
| Runtime prepare API | `Executor::prepare_epistemic_gpu_execution` pairs preflight with GPU workspace allocation and reset. |
| Device-side workspace reset | `Executor::reset_epistemic_gpu_workspace` submits `memset_zeros` for candidate assumptions, world views, model membership, and rejection reasons. |
| Workspace reset trace | `EpistemicGpuWorkspaceResetTrace` records zeroed bytes, `device_zero_ops = 4`, and `host_write_ops = 0`. |
| Candidate generation kernel | `epistemic_generate_candidate_assumptions_u8` writes bounded candidate-assumption bitsets into the GPU workspace. |
| Candidate generation trace | `EpistemicGpuCandidateGenerationTrace` records literal count, generated candidates, candidate bytes, `kernel_launches = 1`, and `host_write_ops = 0`. |
| Propagation staging kernel | `epistemic_propagate_candidates_u8` stages generated candidates into GPU world-view/rejection buffers. |
| Propagation staging trace | `EpistemicGpuPropagationTrace` records propagated candidates, world-view bytes, rejection-reason slots, `kernel_launches = 1`, and `host_write_ops = 0`. |
| Runtime WCOJ counter snapshot | `Executor::epistemic_gpu_runtime_counters` snapshots existing production WCOJ, layout-sort, and K-clique metadata counters. |
| Preflight-only WCOJ evidence rejected | `EpistemicGpuRuntimeWcojCertification` reports `MissingRequiredWcojDispatch` when a K-clique WCOJ plan exists but runtime WCOJ counters do not advance. |
| Reduced-plan execution trace | `Executor::execute_epistemic_gpu_execution` prepares workspace, launches candidate generation and propagation, executes the reduced production runtime plan with `execute_plan`, and captures before/after counter deltas in `EpistemicGpuRuntimeTrace`. |

## Validation

| Command | Result |
|---|---|
| `cargo fmt --check` | PASS |
| `git diff --check` | PASS |
| `cargo test -p xlog-runtime --test test_epistemic_gpu_workspace` | PASS, 14 passed, 0 failed |
| `cargo test -p xlog-cuda --test build_script_tests -- --nocapture` | PASS, 4 passed, 0 failed |
| `cargo test -p xlog-runtime --lib` | PASS, 125 passed, 0 failed |
| `cargo check -p xlog-cuda -p xlog-runtime -p xlog-logic -p xlog-ir` | PASS |
| `cargo check -p pyxlog` | PASS |

## Metric Status

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M090_GPU.1 production lowering | accepted epistemic fixture runs through production runtime dispatch | PARTIAL | Runtime API launches candidate generation and propagation before reduced production-plan execution with counter tracing; validation/materialization dispatch is still missing. |
| M090_GPU.2 WCOJ eligibility | at least one epistemic reduction uses the WCOJ planner/path where eligible | PARTIAL | Preflight records WCOJ/K-clique/helper metadata, and the counter guard rejects preflight-only evidence; no dispatch launch evidence yet. |
| M090_GPU.3 GPU buffers | candidate, world-view, and rejection state have GPU-resident representations | PARTIAL | Prepare API combines preflight with workspace allocation and device-side reset; candidate and propagation staging buffers can be populated by bounded CUDA kernels; model-membership and semantic validation population are still missing. |
| M090_GPU.4 kernel coverage | GPU kernels cover candidate generation, propagation, validation, and materialization hot paths | PARTIAL | Candidate generation has `epistemic_generate_candidate_assumptions_u8`; propagation staging has `epistemic_propagate_candidates_u8`; validation and materialization kernels are missing. |
| M090_GPU.5 CPU fallback ban | accepted trace records zero CPU candidate/world-view fallbacks | PARTIAL | Preflight rejects nonzero fallback counters, and candidate/propagation traces record zero host writes; validation fallback evidence remains missing. |
| M090_GPU.6 launch evidence | nonzero GPU launch counts and timings | PARTIAL | Candidate-generation and propagation traces record nonzero launches; timing evidence is still missing. |

## Remaining Blocker

The next slice must move from allocation/reset/candidate generation/propagation
staging/counter tracing to actual epistemic runtime dispatch: validate world
views, materialize accepted results, and emit full launch counters/timing plus
zero CPU fallback counters.
