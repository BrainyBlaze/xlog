# v0.9.0 G090_GPU Workspace Evidence

Date: 2026-05-18

Goal node: `G090_GPU - GPU-Native Runtime And WCOJ Execution`

Branch: `feat/v090-epistemic-solver-semantics`

## Scope

This slice maps `EpistemicGpuPlan` buffer requirements to runtime workspace
layout, allocatable device-buffer handles, device-side workspace reset, bounded
GPU candidate generation, propagation staging, candidate-buffer validation, and
bounded world-view validation/materialization staging, including final-result
flag staging from reduced-output device row-count metadata. It does not close
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
| Model-membership staging API | `Executor::populate_epistemic_gpu_model_membership` launches `epistemic_populate_model_membership_u8` into the candidate-scoped model-membership workspace and gates writes on `output.num_rows_device()`. |
| Model-membership staging trace | `EpistemicGpuModelMembershipTrace` records checked candidates, reductions, models per reduction, model-membership bytes, one output row-count device read, rejection slots, `kernel_launches = 1`, CUDA-event elapsed timing, and `host_write_ops = 0`. |
| World-view validation staging API | `Executor::validate_epistemic_gpu_world_views` launches `epistemic_validate_world_views_u8` against model-membership and world-view buffers. |
| World-view validation staging trace | `EpistemicGpuWorldViewValidationTrace` records checked candidates, reductions, models per reduction, membership bytes, world-view slots, rejection slots, `kernel_launches = 1`, CUDA-event elapsed timing, and `host_write_ops = 0`. |
| Materialization staging API | `Executor::materialize_epistemic_gpu_candidates` launches `epistemic_materialize_accepted_candidates_u8` from rejection codes into world-view slots. |
| Materialization staging trace | `EpistemicGpuMaterializationTrace` records materialized candidates, world-view slots, `kernel_launches = 1`, CUDA-event elapsed timing, and `host_write_ops = 0`. |
| Final-result flag staging API | `Executor::materialize_epistemic_gpu_final_results` launches `epistemic_materialize_final_result_flags_u8` from `output.num_rows_device()` and rejection codes into world-view slots. |
| Final-result flag staging trace | `EpistemicGpuFinalResultMaterializationTrace` records materialized candidates, one output row-count device read, world-view slots, `kernel_launches = 1`, CUDA-event elapsed timing, and `host_write_ops = 0`. |
| Runtime preflight | `EpistemicGpuRuntimePreflight::for_executable_plan` consumes `EpistemicExecutablePlan`, computes workspace layout, rejects nonzero CPU fallback counters, and records WCOJ/helper route metadata. |
| Runtime counter guard | `EpistemicGpuRuntimeWcojCertification` requires actual WCOJ counter deltas before WCOJ evidence can certify a K-clique epistemic reduction. |
| Reduced-plan execution trace | `Executor::execute_epistemic_gpu_execution` launches candidate generation, propagation, and candidate validation before the reduced production runtime plan, captures `EpistemicGpuRuntimeTrace` counter deltas, requires WCOJ certification, then launches model-membership, world-view validation, accepted-candidate materialization, and final-result flag staging. |
| Hot-path transfer budget | `EpistemicGpuTransferBudgetTrace` snapshots provider host-transfer counters around the GPU hot path and rejects tracked H2D/D2H deltas without resetting shared stats. |
| Capacity guard | Zero candidate/world/model capacities are rejected with typed `ResourceExhausted` errors. |

## Validation

| Command | Result |
|---|---|
| `cargo fmt` | PASS |
| `cargo test -p xlog-runtime --test test_epistemic_gpu_workspace` | PASS, 33 passed, 0 failed |
| `cargo test -p xlog-cuda --test build_script_tests -- --nocapture` | PASS, 4 passed, 0 failed |
| `cargo test -p xlog-runtime --lib` | PASS, 125 passed, 0 failed |
| `cargo check -p xlog-cuda -p xlog-runtime -p xlog-logic -p xlog-ir` | PASS |
| `cargo check -p pyxlog` | PASS |

## Metric Status

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M090_GPU.1 production lowering | accepted epistemic fixture runs through production runtime dispatch | PARTIAL | Runtime API launches candidate generation, propagation, and candidate validation before reduced production-plan execution with counter tracing, then launches row-count-gated model-membership, world-view validation, accepted-candidate materialization, and final-result flag staging; actual reduced stable-model tuple membership population and full final tuple materialization are still missing. |
| M090_GPU.2 WCOJ eligibility | at least one epistemic reduction uses the WCOJ planner/path where eligible | PARTIAL | Preflight records WCOJ/K-clique/helper route metadata, and the runtime entry point now fails closed when a WCOJ-required K-clique reduction lacks production counter deltas; certified successful dispatch evidence is still missing. |
| M090_GPU.3 GPU buffers | candidate, world-view, and rejection state have GPU-resident representations | PARTIAL | Runtime workspace uses `TrackedCudaSlice` handles, device-side reset, bounded candidate-assumption kernel writes, propagation staging writes, candidate validation writes, row-count-gated candidate-scoped model-membership writes, bounded world-view validation reads/writes, accepted-candidate materialization writes, and final-result flag writes; actual stable-model tuple membership population is missing. |
| M090_GPU.4 kernel coverage | GPU kernels cover candidate generation, propagation, validation, and materialization hot paths | PARTIAL | Candidate generation, propagation staging, candidate-buffer validation, model-membership staging, bounded world-view validation staging, accepted-candidate materialization staging, and final-result flag staging have CUDA kernels; full final tuple/query-result materialization kernels are missing. |
| M090_GPU.5 CPU fallback ban | accepted execution trace records zero CPU candidate enumeration/world-view validation fallbacks | PARTIAL | Runtime preflight rejects nonzero forbidden CPU fallback counters, and candidate/propagation/validation/model-membership/world-view-validation/materialization/final-result traces record zero host writes; actual stable-model tuple membership population evidence remains missing. |
| M090_GPU.6 launch evidence | certification logs include nonzero GPU launch counts and kernel timing for epistemic execution | PARTIAL | Candidate-generation, propagation, candidate-validation, model-membership, world-view-validation, accepted-candidate materialization, and final-result flag traces each record a kernel launch with CUDA-event elapsed timing; full final tuple materialization timing evidence is missing. |
| M090_GPU.7 parity | GPU output matches semantic oracle on all G91, FAEEL, GPT, and splitting fixtures | BLOCKED | No GPU output exists yet. |
| M090_GPU.8 transfer budget | host-device transfers are bounded and reported; no per-candidate host round trip in hot path | PARTIAL | `EpistemicGpuTransferBudgetTrace` records zero tracked data-plane H2D/D2H calls in the bounded hot path and rejects tracked deltas; full accepted-execution final-result transfer accounting is still missing. |

## Remaining Blocker

The next slice must map actual reduced-runtime stable-model tuples into
model-membership bytes and attach full final query-result tuple materialization
kernels or GPU-backed adapters to this initialized workspace, then produce a
complete accepted-execution trace with final tuple materialization timings,
WCOJ dispatch evidence, and zero CPU fallback counters.
