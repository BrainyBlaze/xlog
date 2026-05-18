# v0.9.0 G090_GPU Workspace Evidence

Date: 2026-05-18

Goal node: `G090_GPU - GPU-Native Runtime And WCOJ Execution`

Branch: `feat/v090-epistemic-solver-semantics`

## Scope

This slice maps `EpistemicGpuPlan` buffer requirements to runtime workspace
layout, allocatable device-buffer handles, device-side workspace reset, and
bounded GPU candidate generation plus propagation staging. It does not close
`G090_GPU`.

## Implementation Summary

| Requirement | Evidence |
|---|---|
| Runtime workspace layout | `EpistemicGpuWorkspaceLayout::for_plan` computes candidate, world-view, model-membership, and rejection-reason buffer sizes. |
| Device-resident handle types | `EpistemicGpuWorkspace` stores required buffers as `TrackedCudaSlice<u8>` and `TrackedCudaSlice<u32>`. |
| Runtime allocation API | `Executor::allocate_epistemic_gpu_workspace` allocates all required buffers from the CUDA memory manager. |
| Runtime reset API | `Executor::reset_epistemic_gpu_workspace` zeroes all required buffers with device `memset_zeros`. |
| Reset trace | `EpistemicGpuWorkspaceResetTrace` records candidate/world/model/rejection bytes, `device_zero_ops = 4`, and `host_write_ops = 0`. |
| Candidate generation API | `Executor::generate_epistemic_gpu_candidates` launches `epistemic_generate_candidate_assumptions_u8` into the candidate workspace. |
| Candidate generation trace | `EpistemicGpuCandidateGenerationTrace` records bounded generated candidates, candidate bytes, `kernel_launches = 1`, and `host_write_ops = 0`. |
| Propagation staging API | `Executor::propagate_epistemic_gpu_candidates` launches `epistemic_propagate_candidates_u8` against generated candidate rows. |
| Propagation staging trace | `EpistemicGpuPropagationTrace` records propagated candidates, world-view bytes, rejection-reason slots, `kernel_launches = 1`, and `host_write_ops = 0`. |
| Runtime preflight | `EpistemicGpuRuntimePreflight::for_executable_plan` consumes `EpistemicExecutablePlan`, computes workspace layout, rejects nonzero CPU fallback counters, and records WCOJ/helper route metadata. |
| Runtime counter guard | `EpistemicGpuRuntimeWcojCertification` requires actual WCOJ counter deltas before WCOJ evidence can certify a K-clique epistemic reduction. |
| Reduced-plan execution trace | `Executor::execute_epistemic_gpu_execution` launches candidate generation and propagation before the reduced production runtime plan and captures `EpistemicGpuRuntimeTrace` counter deltas. |
| Capacity guard | Zero candidate/world/model capacities are rejected with typed `ResourceExhausted` errors. |

## Validation

| Command | Result |
|---|---|
| `cargo fmt` | PASS |
| `cargo test -p xlog-runtime --test test_epistemic_gpu_workspace` | PASS, 14 passed, 0 failed |
| `cargo test -p xlog-cuda --test build_script_tests -- --nocapture` | PASS, 4 passed, 0 failed |
| `cargo test -p xlog-runtime --lib` | PASS, 125 passed, 0 failed |
| `cargo check -p xlog-cuda -p xlog-runtime -p xlog-logic -p xlog-ir` | PASS |
| `cargo check -p pyxlog` | PASS |

## Metric Status

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M090_GPU.1 production lowering | accepted epistemic fixture runs through production runtime dispatch | PARTIAL | Runtime API launches candidate generation and propagation before reduced production-plan execution with counter tracing; validation/materialization dispatch is still missing. |
| M090_GPU.2 WCOJ eligibility | at least one epistemic reduction uses the WCOJ planner/path where eligible | PARTIAL | Preflight records WCOJ/K-clique/helper route metadata and the counter guard rejects metadata-only evidence; runtime dispatch evidence is missing. |
| M090_GPU.3 GPU buffers | candidate, world-view, and rejection state have GPU-resident representations | PARTIAL | Runtime workspace uses `TrackedCudaSlice` handles, device-side reset, bounded candidate-assumption kernel writes, and propagation staging writes for world-view/rejection buffers; model-membership and semantic validation population are missing. |
| M090_GPU.4 kernel coverage | GPU kernels cover candidate generation, propagation, validation, and materialization hot paths | PARTIAL | Candidate generation and propagation staging have CUDA kernels; validation and materialization kernels are missing. |
| M090_GPU.5 CPU fallback ban | accepted execution trace records zero CPU candidate enumeration/world-view validation fallbacks | PARTIAL | Runtime preflight rejects nonzero forbidden CPU fallback counters, and candidate/propagation traces record zero host writes; world-view validation fallback evidence is still missing. |
| M090_GPU.6 launch evidence | certification logs include nonzero GPU launch counts and kernel timing for epistemic execution | PARTIAL | Candidate-generation and propagation traces each record a kernel launch; timing evidence is missing. |
| M090_GPU.7 parity | GPU output matches semantic oracle on all G91, FAEEL, GPT, and splitting fixtures | BLOCKED | No GPU output exists yet. |
| M090_GPU.8 transfer budget | host-device transfers are bounded and reported; no per-candidate host round trip in hot path | BLOCKED | No execution transfer trace exists yet. |

## Remaining Blocker

The next slice must attach world-view validation and materialization kernels or
GPU-backed adapters to this initialized workspace and produce a measured
execution trace with launch counts, kernel timings, WCOJ dispatch evidence, and
zero CPU fallback counters.
