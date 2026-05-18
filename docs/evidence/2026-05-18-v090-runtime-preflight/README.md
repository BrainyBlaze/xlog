# v0.9.0 G090_GPU Runtime Preflight, Workspace Reset, Counter-Guard, And Trace Evidence

Date: 2026-05-18

Goal node: `G090_GPU - GPU-Native Runtime And WCOJ Execution`

Branch: `feat/v090-epistemic-solver-semantics`

## Scope

This slice connects `EpistemicExecutablePlan` to `xlog-runtime` preflight, adds
a device-side workspace reset trace, adds bounded GPU candidate generation,
adds bounded GPU propagation staging, adds bounded candidate-buffer validation,
adds bounded world-view validation/materialization staging, adds final-result
flag staging from reduced-output device row-count metadata, adds final tuple
materialization into a device-resident output buffer, adds a
certification guard tying WCOJ evidence to actual production counter deltas, and
exposes a reduced-plan execution trace around `execute_plan`. It is still
incomplete for the epistemic hot path and does not close `G090_GPU`.

## Implementation Summary

| Requirement | Evidence |
|---|---|
| Runtime consumes executable plan | `EpistemicGpuRuntimePreflight::for_executable_plan` accepts `EpistemicExecutablePlan`. |
| Workspace layout tied to executable plan | Preflight computes `EpistemicGpuWorkspaceLayout` from the GPU contract and capacity limits. |
| CPU fallback ban starts at runtime boundary | Preflight rejects nonzero forbidden CPU fallback counters with typed `UnsupportedEpistemicConstruct`. |
| Tuple-membership bindings are certified | Preflight calls `EpistemicGpuPlan::validate_tuple_membership_bindings` and records the tuple-membership binding count. |
| WCOJ route metadata inspected | Preflight records reduced rule count, `MultiWayJoin` count, K-clique WCOJ plan count, planned-hash count, sorted-layout requirement count, helper-split spec count, and tuple-membership binding count. |
| Runtime prepare API | `Executor::prepare_epistemic_gpu_execution` pairs preflight with GPU workspace allocation and reset. |
| Device-side workspace reset | `Executor::reset_epistemic_gpu_workspace` submits `memset_zeros` for candidate assumptions, world views, model membership, and rejection reasons. |
| Workspace reset trace | `EpistemicGpuWorkspaceResetTrace` records zeroed bytes, `device_zero_ops = 4`, and `host_write_ops = 0`. |
| Candidate generation kernel | `epistemic_generate_candidate_assumptions_u8` writes bounded candidate-assumption bitsets into the GPU workspace. |
| Candidate generation trace | `EpistemicGpuCandidateGenerationTrace` records literal count, generated candidates, candidate bytes, `kernel_launches = 1`, CUDA-event elapsed timing, and `host_write_ops = 0`. |
| Propagation staging kernel | `epistemic_propagate_candidates_u8` stages generated candidates into GPU world-view/rejection buffers. |
| Propagation staging trace | `EpistemicGpuPropagationTrace` records propagated candidates, world-view bytes, rejection-reason slots, `kernel_launches = 1`, CUDA-event elapsed timing, and `host_write_ops = 0`. |
| Candidate validation kernel | `epistemic_validate_candidate_bits_u8` checks staged candidate bits and world-view activity in GPU buffers. |
| Candidate validation trace | `EpistemicGpuCandidateValidationTrace` records validated candidates, checked bytes, rejection-reason slots, `kernel_launches = 1`, CUDA-event elapsed timing, and `host_write_ops = 0`. |
| Model-membership tuple-source kernels | `epistemic_populate_model_membership_from_tuple_source_u8`, `epistemic_populate_model_membership_from_tuple_source_arity1_u8`, and `epistemic_populate_model_membership_from_tuple_source_arity2_u8` write candidate-scoped model-membership bytes from candidate assumptions, world-view activity, rejection codes, and named reduced stable-model tuple-source relations; arity-one/two kernels compare encoded ground key bits against current model-slot relation-cell bytes on device. |
| Model-membership tuple-source trace | `EpistemicGpuModelMembershipTrace` records checked candidates, reductions, models per reduction, model-membership bytes, zero output row-count device reads, tuple-source row-count device reads, tuple-key column device reads, `membership_source = StableModelTupleBuffer`, CUDA-event elapsed timing, and `host_write_ops = 0`. |
| Stable-model membership gate | `EpistemicGpuModelMembershipTrace::require_stable_model_tuple_source` still rejects row-count-only staging; the accepted runtime path now uses stable tuple-source traces for arity 0-2 bindings and fails closed for tuple keys that need bound value buffers. |
| World-view validation staging kernel | `epistemic_validate_world_views_u8` checks staged model-membership bytes against active world-view slots and updates rejection codes. |
| World-view validation staging trace | `EpistemicGpuWorldViewValidationTrace` records checked candidates, reductions, models per reduction, membership bytes, world-view slots, rejection slots, `kernel_launches = 1`, CUDA-event elapsed timing, and `host_write_ops = 0`. |
| Materialization staging kernel | `epistemic_materialize_accepted_candidates_u8` writes accepted-candidate flags from rejection codes into GPU world-view slots. |
| Materialization staging trace | `EpistemicGpuMaterializationTrace` records materialized candidates, world-view slots, `kernel_launches = 1`, CUDA-event elapsed timing, and `host_write_ops = 0`. |
| Final-result flag staging kernel | `epistemic_materialize_final_result_flags_u8` reads `output.num_rows_device()` plus rejection codes and writes final-result flags into GPU world-view slots. |
| Final-result flag staging trace | `EpistemicGpuFinalResultMaterializationTrace` records materialized candidates, one output row-count device read, world-view slots, `kernel_launches = 1`, CUDA-event elapsed timing, and `host_write_ops = 0`. |
| Final tuple materialization kernel | `epistemic_materialize_final_tuple_column_u8` copies reduced-output tuple columns into the final-output buffer and writes final row-count metadata on device. |
| Final tuple materialization trace | `EpistemicGpuFinalTupleMaterializationTrace` records output column count, row capacity, covered tuple bytes, one output row-count device read, one final row-count device write, kernel launches, CUDA-event elapsed timing, and `host_write_ops = 0`. |
| Runtime WCOJ counter snapshot | `Executor::epistemic_gpu_runtime_counters` snapshots existing production WCOJ, layout-sort, and K-clique metadata counters. |
| Preflight-only WCOJ evidence rejected | `EpistemicGpuRuntimeWcojCertification` reports `MissingRequiredWcojDispatch` when a K-clique WCOJ plan exists but runtime WCOJ counters do not advance. |
| Runtime WCOJ gate | `EpistemicGpuRuntimeTrace::require_wcoj_certification` returns a typed `UnsupportedEpistemicConstruct` error for required K-clique WCOJ plans with zero observed WCOJ dispatches. |
| Reduced-plan execution trace | `Executor::execute_epistemic_gpu_execution` prepares workspace, launches candidate generation, propagation, and candidate validation, executes the reduced production runtime plan with `execute_plan`, captures before/after counter deltas in `EpistemicGpuRuntimeTrace`, requires WCOJ certification, then launches model-membership, world-view validation, accepted-candidate materialization, final-result flag staging, and final tuple materialization. |
| Hot-path transfer budget | `EpistemicGpuTransferBudgetTrace` snapshots provider host-transfer counters around the GPU hot path and rejects tracked H2D/D2H deltas without resetting shared stats. |

## Validation

| Command | Result |
|---|---|
| `cargo fmt --check` | PASS |
| `git diff --check` | PASS |
| `cargo test -p xlog-runtime --test test_epistemic_gpu_workspace` | PASS, 43 passed, 0 failed |
| `cargo test -p xlog-cuda --test build_script_tests -- --nocapture` | PASS, 4 passed, 0 failed |
| `cargo test -p xlog-runtime --lib` | PASS, 128 passed, 0 failed |
| `cargo check -p xlog-cuda -p xlog-runtime -p xlog-logic -p xlog-ir` | PASS |
| `cargo check -p pyxlog` | PASS |

## Metric Status

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M090_GPU.1 production lowering | accepted epistemic fixture runs through production runtime dispatch | PARTIAL | Runtime API launches candidate generation, propagation, and candidate validation before reduced production-plan execution with counter tracing, then launches arity 0-2 tuple-source-backed model-membership staging with row-scoped ground key comparison for arity one/two, world-view validation, accepted-candidate materialization, final-result flag staging, and final tuple materialization; bound-variable tuple-key matching, arbitrary arity, and full accepted semantics remain missing. |
| M090_GPU.2 WCOJ eligibility | at least one epistemic reduction uses the WCOJ planner/path where eligible | PARTIAL | Preflight records WCOJ/K-clique/helper metadata, and the runtime WCOJ gate rejects preflight-only evidence before model-membership/world-view staging; no certified successful dispatch evidence yet. |
| M090_GPU.3 GPU buffers | candidate, world-view, and rejection state have GPU-resident representations | PARTIAL | Prepare API combines preflight with workspace allocation and device-side reset; candidate, propagation, candidate-validation, arity 0-2 tuple-source model-membership staging with encoded ground key expectations, bounded world-view-validation, accepted-candidate materialization, and final-result flag buffers can be populated or checked by CUDA kernels; bound-variable and arbitrary-arity stable-model tuple matching are still missing. |
| M090_GPU.4 kernel coverage | GPU kernels cover candidate generation, propagation, validation, and materialization hot paths | PARTIAL | Candidate generation has `epistemic_generate_candidate_assumptions_u8`; propagation staging has `epistemic_propagate_candidates_u8`; candidate-buffer validation has `epistemic_validate_candidate_bits_u8`; arity 0-2 tuple-source model membership has fixed kernels over existing relation columns and row-scoped ground-key byte comparison for arity one/two; bounded world-view validation has `epistemic_validate_world_views_u8`; materialization staging has `epistemic_materialize_accepted_candidates_u8`; final-result flag staging has `epistemic_materialize_final_result_flags_u8`; final tuple materialization has `epistemic_materialize_final_tuple_column_u8`; bound-variable and arbitrary-arity tuple matching remain missing. |
| M090_GPU.5 CPU fallback ban | accepted trace records zero CPU candidate/world-view fallbacks | PARTIAL | Preflight rejects nonzero fallback counters, and candidate/propagation/validation/model-membership/world-view-validation/materialization/final-result traces record zero host writes; arity 0-2 tuple-source staging reads existing device relation buffers and compares row-scoped ground keys on device, while bound-value tuple matching remains missing. |
| M090_GPU.6 launch evidence | nonzero GPU launch counts and timings | PARTIAL | Candidate-generation, propagation, candidate-validation, arity 0-2 tuple-source model-membership staging with fixed arity-one/two row-scoped ground key comparison, world-view-validation, accepted-candidate materialization, final-result flag, and final tuple traces record nonzero launches with CUDA-event elapsed timing; accepted semantic parity timing evidence is still missing. |
| M090_GPU.9 nonzero-arity membership | at least two fixtures with arity >= 1 check stable-model tuple membership on GPU over existing relation layouts | PARTIAL | Plan/runtime tests now require identity key-column metadata, source tuple terms, encoded expected tuple-key bits/type codes, and arity-one/arity-two tuple-source kernels over existing `CudaBuffer` columns; semantic oracle parity fixtures are still missing. |
| M090_GPU.10 row-count guard | nonzero-arity membership fails closed if only row-count metadata is available | PARTIAL | `EpistemicTupleMembershipBinding::key_columns` validation rejects invalid key metadata and row-count-only certification remains a negative fixture; complete accepted-execution fixture coverage is still missing. |
| M090_GPU.8 transfer budget | host-device transfers are bounded and reported; no per-candidate host round trip in hot path | PARTIAL | Hot-path provider transfer snapshots reject tracked data-plane H2D/D2H deltas; final result transfer accounting for complete accepted execution is still missing. |

## Remaining Blocker

The next slice must extend fixed arity 0-2 ground-key matching to bound-value
tuple keys and arbitrary arity, use that membership when producing final query
results, and emit full accepted-execution timing, final transfer accounting,
semantic parity fixtures, and zero CPU fallback counters.
