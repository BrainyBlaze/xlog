# v0.9.0 G090_GPU Workspace Evidence

Date: 2026-05-18

Goal node: `G090_GPU - GPU-Native Runtime And WCOJ Execution`

Branch: `feat/v090-epistemic-solver-semantics`

## Scope

This slice maps `EpistemicGpuPlan` buffer requirements to runtime workspace
layout, allocatable device-buffer handles, device-side workspace reset, bounded
GPU candidate generation, propagation staging, candidate-buffer validation, and
bounded world-view validation/materialization staging, including final-result
flag staging from reduced-output device row-count metadata and final tuple
materialization into a device-resident output buffer. It does not close
`G090_GPU`.

## Implementation Summary

| Requirement | Evidence |
|---|---|
| Runtime workspace layout | `EpistemicGpuWorkspaceLayout::for_plan` computes candidate, world-view, candidate-scoped model-membership, and rejection-reason buffer sizes. |
| Device-resident handle types | `EpistemicGpuWorkspace` stores required buffers as `TrackedCudaSlice<u8>` and `TrackedCudaSlice<u32>`. |
| Runtime allocation API | `Executor::allocate_epistemic_gpu_workspace` allocates all required buffers from the CUDA memory manager. |
| Runtime reset API | `Executor::reset_epistemic_gpu_workspace` zeroes all required buffers with device `memset_zeros`. |
| Reset trace | `EpistemicGpuWorkspaceResetTrace` records candidate/world/model/rejection bytes, `device_zero_ops = 4`, and `host_write_ops = 0`. |
| Candidate generation API | `Executor::generate_epistemic_gpu_candidates` launches `epistemic_generate_candidate_assumptions_u8` into the candidate workspace. |
| Candidate generation trace | `EpistemicGpuCandidateGenerationTrace` records bounded generated candidates, candidate bytes, `kernel_launches = 1`, CUDA-event elapsed timing, and `host_write_ops = 0`. |
| Propagation staging API | `Executor::propagate_epistemic_gpu_candidates` launches `epistemic_propagate_candidates_u8` against generated candidate rows. |
| Propagation staging trace | `EpistemicGpuPropagationTrace` records propagated candidates, world-view bytes, rejection-reason slots, `kernel_launches = 1`, CUDA-event elapsed timing, and `host_write_ops = 0`. |
| Candidate validation API | `Executor::validate_epistemic_gpu_candidates` launches `epistemic_validate_candidate_bits_u8` against staged candidate buffers. |
| Candidate validation trace | `EpistemicGpuCandidateValidationTrace` records validated candidates, candidate/world-view bytes checked, rejection-reason slots, `kernel_launches = 1`, CUDA-event elapsed timing, and `host_write_ops = 0`. |
| Model-membership tuple-source API | `Executor::populate_epistemic_gpu_model_membership_from_tuple_sources` launches zero-arity plus arity-one/arity-two/arity-three tuple-source kernels once per certified binding, reads named reduced stable-model relation row-count scalars and key columns on device, and passes encoded ground tuple-key expectations to the arity-one/two/three kernels. |
| Model-membership tuple-source trace | `EpistemicGpuModelMembershipTrace` records checked candidates, reductions, models per reduction, model-membership bytes, zero output row-count device reads, tuple-source row-count device reads, tuple-key column device reads, `membership_source = StableModelTupleBuffer`, CUDA-event elapsed timing, and `host_write_ops = 0`. |
| Stable-model membership gate | `EpistemicGpuModelMembershipTrace::require_stable_model_tuple_source` rejects row-count-only staging; the runtime path uses stable tuple-source traces for arity 0-3 bindings, compares row-scoped ground key bytes on device for arity one/two/three, and fails closed for bound-variable keys and beyond the fixed-arity kernels. |
| World-view validation staging API | `Executor::validate_epistemic_gpu_world_views` launches `epistemic_validate_world_views_u8` against model-membership and world-view buffers. |
| World-view validation staging trace | `EpistemicGpuWorldViewValidationTrace` records checked candidates, reductions, models per reduction, membership bytes, world-view slots, rejection slots, `kernel_launches = 1`, CUDA-event elapsed timing, and `host_write_ops = 0`. |
| Materialization staging API | `Executor::materialize_epistemic_gpu_candidates` launches `epistemic_materialize_accepted_candidates_u8` from rejection codes into world-view slots. |
| Materialization staging trace | `EpistemicGpuMaterializationTrace` records materialized candidates, world-view slots, `kernel_launches = 1`, CUDA-event elapsed timing, and `host_write_ops = 0`. |
| Final-result flag staging API | `Executor::materialize_epistemic_gpu_final_results` launches `epistemic_materialize_final_result_flags_u8` from `output.num_rows_device()` and rejection codes into world-view slots. |
| Final-result flag staging trace | `EpistemicGpuFinalResultMaterializationTrace` records materialized candidates, one output row-count device read, world-view slots, `kernel_launches = 1`, CUDA-event elapsed timing, and `host_write_ops = 0`. |
| Final tuple materialization API | `Executor::materialize_epistemic_gpu_final_tuples` launches `epistemic_materialize_final_tuple_column_u8` to populate a final-output `CudaBuffer` from reduced-output columns on device. |
| Final tuple materialization trace | `EpistemicGpuFinalTupleMaterializationTrace` records output column count, row capacity, covered tuple bytes, one output row-count device read, one final row-count device write, kernel launches, CUDA-event elapsed timing, and `host_write_ops = 0`. |
| Runtime preflight | `EpistemicGpuRuntimePreflight::for_executable_plan` consumes `EpistemicExecutablePlan`, computes workspace layout, rejects nonzero CPU fallback counters, and records WCOJ/helper route metadata. |
| Tuple-membership preflight gate | `EpistemicGpuPlan::validate_tuple_membership_bindings` requires one matching reduced stable-model tuple binding per epistemic literal before runtime workspace preparation. |
| Runtime counter guard | `EpistemicGpuRuntimeWcojCertification` requires actual WCOJ counter deltas before WCOJ evidence can certify a K-clique epistemic reduction. |
| Reduced-plan execution trace | `Executor::execute_epistemic_gpu_execution` launches candidate generation, propagation, and candidate validation before the reduced production runtime plan, captures `EpistemicGpuRuntimeTrace` counter deltas, requires WCOJ certification, then launches model-membership, world-view validation, accepted-candidate materialization, final-result flag staging, and final tuple materialization. |
| Hot-path transfer budget | `EpistemicGpuTransferBudgetTrace` snapshots provider host-transfer counters around the GPU hot path and rejects tracked H2D/D2H deltas without resetting shared stats. |
| Capacity guard | Zero candidate/world/model capacities are rejected with typed `ResourceExhausted` errors. |

## Validation

| Command | Result |
|---|---|
| `cargo fmt` | PASS |
| `cargo test -p xlog-runtime --test test_epistemic_gpu_workspace` | PASS, 44 passed, 0 failed |
| `cargo test -p xlog-cuda --test build_script_tests -- --nocapture` | PASS, 4 passed, 0 failed |
| `cargo test -p xlog-runtime --lib` | PASS, 128 passed, 0 failed |
| `cargo check -p xlog-cuda -p xlog-runtime -p xlog-logic -p xlog-ir` | PASS |
| `cargo check -p pyxlog` | PASS |

## Metric Status

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M090_GPU.1 production lowering | accepted epistemic fixture runs through production runtime dispatch | PARTIAL | Runtime API launches candidate generation, propagation, and candidate validation before reduced production-plan execution with counter tracing, then launches arity 0-3 tuple-source-backed model-membership staging with row-scoped ground key comparison for arity one/two/three, world-view validation, accepted-candidate materialization, final-result flag staging, and device-side final tuple materialization; bound-variable tuple-key matching, arbitrary arity, and full accepted semantics remain missing. |
| M090_GPU.2 WCOJ eligibility | at least one epistemic reduction uses the WCOJ planner/path where eligible | PARTIAL | Preflight records WCOJ/K-clique/helper route metadata, and the runtime entry point now fails closed when a WCOJ-required K-clique reduction lacks production counter deltas; certified successful dispatch evidence is still missing. |
| M090_GPU.3 GPU buffers | candidate, world-view, and rejection state have GPU-resident representations | PARTIAL | Runtime workspace uses `TrackedCudaSlice` handles, device-side reset, bounded candidate-assumption kernel writes, propagation staging writes, candidate validation writes, arity 0-3 tuple-source model-membership staging from existing relation buffers with encoded ground key expectations, bounded world-view validation reads/writes, accepted-candidate materialization writes, final-result flag writes, and device-resident final-output tuple buffers; bound-variable and arbitrary-arity stable-model tuple matching are missing. |
| M090_GPU.4 kernel coverage | GPU kernels cover candidate generation, propagation, validation, and materialization hot paths | PARTIAL | Candidate generation, propagation staging, candidate-buffer validation, arity 0-3 tuple-source model-membership staging with fixed arity-one/two/three row-scoped ground key comparison, bounded world-view validation staging, accepted-candidate materialization staging, final-result flag staging, and final tuple materialization have CUDA kernels; bound-variable and arbitrary-arity tuple matching remain missing. |
| M090_GPU.5 CPU fallback ban | accepted execution trace records zero CPU candidate enumeration/world-view validation fallbacks | PARTIAL | Runtime preflight rejects nonzero forbidden CPU fallback counters, and candidate/propagation/validation/model-membership/world-view-validation/materialization/final-result traces record zero host writes; arity 0-3 tuple-source staging reads device relation columns and compares row-scoped ground keys on device, while bound-value tuple matching remains missing. |
| M090_GPU.6 launch evidence | certification logs include nonzero GPU launch counts and kernel timing for epistemic execution | PARTIAL | Candidate-generation, propagation, candidate-validation, arity 0-3 tuple-source model-membership staging with fixed arity-one/two/three row-scoped ground key comparison, world-view-validation, accepted-candidate materialization, final-result flag, and final tuple traces each record kernel launches with CUDA-event elapsed timing; accepted semantic parity timing evidence is missing. |
| M090_GPU.9 nonzero-arity membership | at least two fixtures with arity >= 1 check stable-model tuple membership on GPU over existing relation layouts | PARTIAL | Source and trace tests require arity-one, arity-two, and arity-three tuple-source kernels over existing relation columns, encoded expected tuple-key bits/type codes, and device cell-byte comparison for ground keys; semantic GPU parity fixtures are still missing. |
| M090_GPU.10 row-count guard | nonzero-arity membership fails closed if only row-count metadata is available | PARTIAL | Invalid tuple-key metadata is rejected and row-count-only membership remains a negative certification fixture; accepted end-to-end coverage is still missing. |
| M090_GPU.7 parity | GPU output matches semantic oracle on all G91, FAEEL, GPT, and splitting fixtures | BLOCKED | Bounded final-output device buffers exist, but semantic oracle-matched GPU output from accepted stable-model membership is not proven yet. |
| M090_GPU.8 transfer budget | host-device transfers are bounded and reported; no per-candidate host round trip in hot path | PARTIAL | `EpistemicGpuTransferBudgetTrace` records zero tracked data-plane H2D/D2H calls in the bounded hot path and rejects tracked deltas; full accepted-execution final-result transfer accounting is still missing. |

## Remaining Blocker

The next slice must extend fixed arity 0-3 ground-key matching to bound-value
tuple keys and arbitrary arity, use that membership when producing final query
results, then produce a complete accepted-execution trace with semantic parity,
WCOJ dispatch evidence, and zero CPU fallback counters.
