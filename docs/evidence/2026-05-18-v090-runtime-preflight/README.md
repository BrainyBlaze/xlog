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
incomplete for the full epistemic hot path and does not close `G090_GPU`.

## Implementation Summary

| Requirement | Evidence |
|---|---|
| Runtime consumes executable plan | `EpistemicGpuRuntimePreflight::for_executable_plan` accepts `EpistemicExecutablePlan`. |
| Executable registration metadata | `EpistemicExecutablePlan` carries reduced production compiler relation IDs for runtime registration. |
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
| Model-membership tuple-source kernels | `epistemic_populate_model_membership_from_tuple_source_u8`, `epistemic_populate_model_membership_from_tuple_source_arity1_u8`, `epistemic_populate_model_membership_from_tuple_source_arity2_u8`, and `epistemic_populate_model_membership_from_tuple_source_arity3_u8` write candidate-scoped model-membership bytes from candidate assumptions, world-view activity, rejection codes, and named reduced stable-model tuple-source relations; arity-one/two/three kernels compare encoded ground key bits against current model-slot relation-cell bytes on device. |
| Model-membership tuple-source trace | `EpistemicGpuModelMembershipTrace` records checked candidates, reductions, models per reduction, model-membership bytes, zero output row-count device reads, tuple-source row-count device reads, tuple-key column device reads, `membership_source = StableModelTupleBuffer`, CUDA-event elapsed timing, and `host_write_ops = 0`. |
| Stable-model membership gate | `EpistemicGpuModelMembershipTrace::require_stable_model_tuple_source` still rejects row-count-only staging; the accepted runtime path now uses stable tuple-source traces for arity 0-3 bindings and fails closed for tuple keys that need bound value buffers. |
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
| Reduced-plan execution trace | `Executor::execute_epistemic_gpu_execution` prepares workspace, launches candidate generation, propagation, and candidate validation, executes the reduced production runtime plan with `execute_plan`, captures before/after counter deltas in `EpistemicGpuRuntimeTrace`, requires WCOJ certification, clones the named reduced output relation, then launches model-membership, world-view validation, accepted-candidate materialization, final-result flag staging, and final tuple materialization. |
| Accepted WCOJ dispatch | `test_epistemic_gpu_wcoj_execution` proves one accepted K5 epistemic reduction reaches certified production WCOJ dispatch and final row materialization. |
| Hot-path transfer budget | `EpistemicGpuTransferBudgetTrace` snapshots provider host-transfer counters around the GPU hot path and rejects tracked H2D/D2H deltas without resetting shared stats. |

## Validation

| Command | Result |
|---|---|
| `cargo fmt --check` | PASS |
| `git diff --check` | PASS |
| `cargo test -p xlog-runtime --test test_epistemic_gpu_workspace` | PASS, 47 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-cuda --test build_script_tests -- --nocapture` | PASS, 4 passed, 0 failed |
| `cargo test -p xlog-runtime --lib` | PASS, 128 passed, 0 failed |
| `cargo check -p xlog-cuda -p xlog-runtime -p xlog-logic -p xlog-ir` | PASS |
| `cargo check -p pyxlog` | PASS |

## Metric Status

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M090_GPU.1 production lowering | accepted epistemic fixture runs through production runtime dispatch | PARTIAL | Runtime API launches candidate generation, propagation, and candidate validation before reduced production-plan execution with counter tracing, then launches arity 0-3 and generic arity-N tuple-source-backed model-membership staging with row-scoped ground and variable-bound key comparison, world-view validation, accepted-candidate materialization, final-result flag staging, final-row map construction, and final tuple materialization; broader full accepted semantics remain missing. |
| M090_GPU.2 WCOJ eligibility | at least one epistemic reduction uses the WCOJ planner/path where eligible | PASS | Accepted K5 fixture records WCOJ/K-clique/helper metadata, passes the runtime WCOJ certification gate, observes production K5 dispatch counters, and materializes one final accepted row. |
| M090_GPU.3 GPU buffers | candidate, world-view, and rejection state have GPU-resident representations | PARTIAL | Prepare API combines preflight with workspace allocation and device-side reset; candidate, propagation, candidate-validation, arity 0-3 and generic arity-N tuple-source model-membership staging with encoded ground key expectations and bound-output column metadata, bounded world-view-validation, accepted-candidate materialization, final-result flag buffers, and final-row maps can be populated or checked by CUDA kernels; broader semantic parity remains missing. |
| M090_GPU.4 kernel coverage | GPU kernels cover candidate generation, propagation, validation, and materialization hot paths | PARTIAL | Candidate generation has `epistemic_generate_candidate_assumptions_u8`; propagation staging has `epistemic_propagate_candidates_u8`; candidate-buffer validation has `epistemic_validate_candidate_bits_u8`; tuple-source model membership has fixed arity-one/two/three kernels plus generic arity-N ground and variable-bound comparison over existing relation columns; bounded world-view validation has `epistemic_validate_world_views_u8`; materialization staging has `epistemic_materialize_accepted_candidates_u8`; final-result flag staging has `epistemic_materialize_final_result_flags_u8`; final tuple materialization has `epistemic_build_final_tuple_row_map_u8` and `epistemic_materialize_final_tuple_column_u8`. |
| M090_GPU.5 CPU fallback ban | accepted trace records zero CPU candidate/world-view fallbacks | PARTIAL | Preflight rejects nonzero fallback counters, and candidate/propagation/validation/model-membership/world-view-validation/materialization/final-result/final-row traces record zero host writes; tuple-source staging reads existing device relation buffers and compares row-scoped ground and variable-bound keys on device. |
| M090_GPU.6 launch evidence | nonzero GPU launch counts and timings | PARTIAL | Candidate-generation, propagation, candidate-validation, tuple-source model-membership staging with fixed arity-one/two/three and generic arity-N row-scoped comparison, world-view-validation, accepted-candidate materialization, final-result flag, final-row map, and final tuple traces record nonzero launches with CUDA-event elapsed timing; accepted semantic parity timing evidence is still incomplete. |
| M090_GPU.9 nonzero-arity membership | at least two fixtures with arity >= 1 check stable-model tuple membership on GPU over existing relation layouts | PARTIAL | Plan/runtime tests require identity key-column metadata, source tuple terms, encoded expected tuple-key bits/type codes, arity-one/arity-two/arity-three and generic arity-N tuple-source kernels over existing `CudaBuffer` columns, and one accepted unary variable-bound final-row filtering fixture; broader semantic parity fixtures are still missing. |
| M090_GPU.10 row-count guard | nonzero-arity membership fails closed if only row-count metadata is available | PARTIAL | `EpistemicTupleMembershipBinding::key_columns` and `bound_output_columns` validation rejects invalid key metadata, row-count-only certification remains a negative fixture, and the accepted unary fixture proves final output is filtered by tuple key rather than row count alone; complete accepted-execution fixture coverage is still missing. |
| M090_GPU.8 transfer budget | host-device transfers are bounded and reported; no per-candidate host round trip in hot path | PARTIAL | Hot-path provider transfer snapshots reject tracked data-plane H2D/D2H deltas; final result transfer accounting for complete accepted execution is still missing. |

## Remaining Blocker

The next slice must broaden accepted-execution parity beyond the K5 WCOJ
fixture, extend solver/probability accepted-runtime traces, and complete zero
CPU fallback certification.
