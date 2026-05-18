# v0.9.0 G090_GPU Workspace Evidence

Date: 2026-05-18

Goal node: `G090_GPU - GPU-Native Runtime And WCOJ Execution`

Branch: `feat/v090-epistemic-solver-semantics`

## Scope

This slice maps `EpistemicGpuPlan` buffer requirements to runtime workspace
layout and allocatable device-buffer handles. It is still pre-kernel plumbing
and does not close `G090_GPU`.

## Implementation Summary

| Requirement | Evidence |
|---|---|
| Runtime workspace layout | `EpistemicGpuWorkspaceLayout::for_plan` computes candidate, world-view, model-membership, and rejection-reason buffer sizes. |
| Device-resident handle types | `EpistemicGpuWorkspace` stores required buffers as `TrackedCudaSlice<u8>` and `TrackedCudaSlice<u32>`. |
| Runtime allocation API | `Executor::allocate_epistemic_gpu_workspace` allocates all required buffers from the CUDA memory manager. |
| Runtime preflight | `EpistemicGpuRuntimePreflight::for_executable_plan` consumes `EpistemicExecutablePlan`, computes workspace layout, rejects nonzero CPU fallback counters, and records WCOJ/helper route metadata. |
| Capacity guard | Zero candidate/world/model capacities are rejected with typed `ResourceExhausted` errors. |

## Validation

| Command | Result |
|---|---|
| `cargo fmt` | PASS |
| `cargo test -p xlog-runtime --test test_epistemic_gpu_workspace` | PASS, 4 passed, 0 failed |
| `cargo test -p xlog-runtime --lib` | PASS, 125 passed, 0 failed |
| `cargo check -p xlog-runtime -p xlog-logic -p xlog-ir` | PASS |
| `cargo check -p pyxlog` | PASS |

## Metric Status

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M090_GPU.1 production lowering | accepted epistemic fixture runs through production runtime dispatch | PARTIAL | Runtime preflight consumes executable plans; dispatch is still missing. |
| M090_GPU.2 WCOJ eligibility | at least one epistemic reduction uses the WCOJ planner/path where eligible | PARTIAL | Preflight records WCOJ/K-clique/helper route metadata; runtime dispatch evidence is missing. |
| M090_GPU.3 GPU buffers | candidate, world-view, and rejection state have GPU-resident representations | PARTIAL | Runtime workspace uses `TrackedCudaSlice` handles; kernels do not populate them yet. |
| M090_GPU.4 kernel coverage | GPU kernels cover candidate generation, propagation, validation, and materialization hot paths | BLOCKED | No epistemic kernels are launched yet. |
| M090_GPU.5 CPU fallback ban | accepted execution trace records zero CPU candidate enumeration/world-view validation fallbacks | PARTIAL | Runtime preflight rejects nonzero forbidden CPU fallback counters; accepted execution trace is missing. |
| M090_GPU.6 launch evidence | certification logs include nonzero GPU launch counts and kernel timing for epistemic execution | BLOCKED | No launch or timing evidence exists yet. |
| M090_GPU.7 parity | GPU output matches semantic oracle on all G91, FAEEL, GPT, and splitting fixtures | BLOCKED | No GPU output exists yet. |
| M090_GPU.8 transfer budget | host-device transfers are bounded and reported; no per-candidate host round trip in hot path | BLOCKED | No execution transfer trace exists yet. |

## Remaining Blocker

The next slice must attach kernels or GPU-backed adapters to this workspace and
produce a measured execution trace with launch counts, kernel timings, WCOJ
dispatch evidence, and zero CPU fallback counters.
