# v0.9.0 G090_GPU Runtime Preflight, Workspace Reset, Counter-Guard, And Trace Evidence

Date: 2026-05-18

Goal node: `G090_GPU - GPU-Native Runtime And WCOJ Execution`

Branch: `feat/v090-epistemic-solver-semantics`

## Scope

This slice connects `EpistemicExecutablePlan` to `xlog-runtime` preflight, adds
a device-side workspace reset trace, adds bounded GPU candidate generation,
adds bounded GPU propagation staging, adds bounded candidate-buffer validation,
adds bounded world-view validation/materialization staging, adds final-result
flag staging from reduced-output device row-count metadata, adds a
certification guard tying WCOJ evidence to actual production counter deltas, and
exposes a reduced-plan execution trace around `execute_plan`. It is still
incomplete for the epistemic hot path and does not close `G090_GPU`.

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
| Candidate generation trace | `EpistemicGpuCandidateGenerationTrace` records literal count, generated candidates, candidate bytes, `kernel_launches = 1`, CUDA-event elapsed timing, and `host_write_ops = 0`. |
| Propagation staging kernel | `epistemic_propagate_candidates_u8` stages generated candidates into GPU world-view/rejection buffers. |
| Propagation staging trace | `EpistemicGpuPropagationTrace` records propagated candidates, world-view bytes, rejection-reason slots, `kernel_launches = 1`, CUDA-event elapsed timing, and `host_write_ops = 0`. |
| Candidate validation kernel | `epistemic_validate_candidate_bits_u8` checks staged candidate bits and world-view activity in GPU buffers. |
| Candidate validation trace | `EpistemicGpuCandidateValidationTrace` records validated candidates, checked bytes, rejection-reason slots, `kernel_launches = 1`, CUDA-event elapsed timing, and `host_write_ops = 0`. |
| Model-membership staging kernel | `epistemic_populate_model_membership_u8` writes candidate-scoped model-membership bytes from candidate assumptions, world-view activity, and rejection codes. |
| Model-membership staging trace | `EpistemicGpuModelMembershipTrace` records checked candidates, reductions, models per reduction, model-membership bytes, rejection slots, `kernel_launches = 1`, CUDA-event elapsed timing, and `host_write_ops = 0`. |
| World-view validation staging kernel | `epistemic_validate_world_views_u8` checks staged model-membership bytes against active world-view slots and updates rejection codes. |
| World-view validation staging trace | `EpistemicGpuWorldViewValidationTrace` records checked candidates, reductions, models per reduction, membership bytes, world-view slots, rejection slots, `kernel_launches = 1`, CUDA-event elapsed timing, and `host_write_ops = 0`. |
| Materialization staging kernel | `epistemic_materialize_accepted_candidates_u8` writes accepted-candidate flags from rejection codes into GPU world-view slots. |
| Materialization staging trace | `EpistemicGpuMaterializationTrace` records materialized candidates, world-view slots, `kernel_launches = 1`, CUDA-event elapsed timing, and `host_write_ops = 0`. |
| Final-result flag staging kernel | `epistemic_materialize_final_result_flags_u8` reads `output.num_rows_device()` plus rejection codes and writes final-result flags into GPU world-view slots. |
| Final-result flag staging trace | `EpistemicGpuFinalResultMaterializationTrace` records materialized candidates, one output row-count device read, world-view slots, `kernel_launches = 1`, CUDA-event elapsed timing, and `host_write_ops = 0`. |
| Runtime WCOJ counter snapshot | `Executor::epistemic_gpu_runtime_counters` snapshots existing production WCOJ, layout-sort, and K-clique metadata counters. |
| Preflight-only WCOJ evidence rejected | `EpistemicGpuRuntimeWcojCertification` reports `MissingRequiredWcojDispatch` when a K-clique WCOJ plan exists but runtime WCOJ counters do not advance. |
| Reduced-plan execution trace | `Executor::execute_epistemic_gpu_execution` prepares workspace, launches candidate generation, propagation, and candidate validation, executes the reduced production runtime plan with `execute_plan`, captures before/after counter deltas in `EpistemicGpuRuntimeTrace`, then launches model-membership, world-view validation, accepted-candidate materialization, and final-result flag staging. |
| Hot-path transfer budget | `EpistemicGpuTransferBudgetTrace` snapshots provider host-transfer counters around the GPU hot path and rejects tracked H2D/D2H deltas without resetting shared stats. |

## Validation

| Command | Result |
|---|---|
| `cargo fmt --check` | PASS |
| `git diff --check` | PASS |
| `cargo test -p xlog-runtime --test test_epistemic_gpu_workspace` | PASS, 32 passed, 0 failed |
| `cargo test -p xlog-cuda --test build_script_tests -- --nocapture` | PASS, 4 passed, 0 failed |
| `cargo test -p xlog-runtime --lib` | PASS, 125 passed, 0 failed |
| `cargo check -p xlog-cuda -p xlog-runtime -p xlog-logic -p xlog-ir` | PASS |
| `cargo check -p pyxlog` | PASS |

## Metric Status

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M090_GPU.1 production lowering | accepted epistemic fixture runs through production runtime dispatch | PARTIAL | Runtime API launches candidate generation, propagation, and candidate validation before reduced production-plan execution with counter tracing, then launches model-membership, world-view validation, accepted-candidate materialization, and final-result flag staging; actual stable-model membership population/full final tuple materialization dispatch is still missing. |
| M090_GPU.2 WCOJ eligibility | at least one epistemic reduction uses the WCOJ planner/path where eligible | PARTIAL | Preflight records WCOJ/K-clique/helper metadata, and the counter guard rejects preflight-only evidence; no dispatch launch evidence yet. |
| M090_GPU.3 GPU buffers | candidate, world-view, and rejection state have GPU-resident representations | PARTIAL | Prepare API combines preflight with workspace allocation and device-side reset; candidate, propagation, candidate-validation, model-membership, bounded world-view-validation, accepted-candidate materialization, and final-result flag buffers can be populated or checked by CUDA kernels; actual stable-model membership population is still missing. |
| M090_GPU.4 kernel coverage | GPU kernels cover candidate generation, propagation, validation, and materialization hot paths | PARTIAL | Candidate generation has `epistemic_generate_candidate_assumptions_u8`; propagation staging has `epistemic_propagate_candidates_u8`; candidate-buffer validation has `epistemic_validate_candidate_bits_u8`; model-membership staging has `epistemic_populate_model_membership_u8`; bounded world-view validation has `epistemic_validate_world_views_u8`; materialization staging has `epistemic_materialize_accepted_candidates_u8`; final-result flag staging has `epistemic_materialize_final_result_flags_u8`; full final tuple materialization kernels are missing. |
| M090_GPU.5 CPU fallback ban | accepted trace records zero CPU candidate/world-view fallbacks | PARTIAL | Preflight rejects nonzero fallback counters, and candidate/propagation/validation/model-membership/world-view-validation/materialization/final-result traces record zero host writes; actual stable-model membership population evidence remains missing. |
| M090_GPU.6 launch evidence | nonzero GPU launch counts and timings | PARTIAL | Candidate-generation, propagation, candidate-validation, model-membership, world-view-validation, accepted-candidate materialization, and final-result flag traces record nonzero launches with CUDA-event elapsed timing; full final tuple materialization timing is still missing. |
| M090_GPU.8 transfer budget | host-device transfers are bounded and reported; no per-candidate host round trip in hot path | PARTIAL | Hot-path provider transfer snapshots reject tracked data-plane H2D/D2H deltas; final result transfer accounting for complete accepted execution is still missing. |

## Remaining Blocker

The next slice must populate model-membership from actual reduced-runtime
stable-model output, materialize full final query tuple results, and emit full
accepted-execution timing, final transfer accounting, and zero CPU fallback
counters.
