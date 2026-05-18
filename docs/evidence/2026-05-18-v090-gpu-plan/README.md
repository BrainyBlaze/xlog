# v0.9.0 G090_GPU Plan-Contract Evidence

Date: 2026-05-18

Goal node: `G090_GPU - GPU-Native Runtime And WCOJ Execution`

Branch: `feat/v090-epistemic-solver-semantics`

## Scope

This slice adds a production-facing GPU execution plan contract. It is not GPU
execution and does not close `G090_GPU`.

## Implementation Summary

| Requirement | Evidence |
|---|---|
| Accepted EIR reaches a production-facing plan contract | `plan_epistemic_gpu_execution` builds `EpistemicGpuPlan` from parsed AST/EIR. |
| GPU hot-path phases are explicit | `EpistemicGpuHotPathPhase` requires candidate generation, propagation, world-view validation, and result materialization. |
| GPU-resident buffers are explicit | `EpistemicGpuBufferKind` requires candidate assumptions, world views, model membership, and rejection reasons. |
| CPU fallback counters are tracked | `EpistemicCpuFallbackCounters::is_zero()` covers candidate enumeration, world-view validation, solver search, and probabilistic recomputation. |
| WCOJ planner obligation is visible | Multi-relation reduced bodies are marked `RequiresPlannerEligibility` rather than bypassing the WCOJ planner. |
| Non-epistemic programs stay out of the epistemic plan path | Non-epistemic source returns a typed `UnsupportedEpistemicConstruct` error for this API. |

## Validation

| Command | Result |
|---|---|
| `cargo fmt` | PASS |
| `cargo test -p xlog-logic --test test_epistemic_gpu_plan` | PASS, 3 passed, 0 failed |
| `cargo test -p xlog-ir --lib` | PASS, 14 passed, 0 failed |
| `cargo test -p xlog-logic --lib` | PASS, 238 passed, 0 failed |
| `cargo check -p xlog-logic -p xlog-ir -p xlog-solve -p xlog-prob` | PASS |
| `cargo check -p pyxlog` | PASS |

## Metric Status

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M090_GPU.1 production lowering | accepted epistemic fixture runs through production runtime dispatch | PARTIAL | Plan contract exists; runtime dispatch is still missing. |
| M090_GPU.2 WCOJ eligibility | at least one epistemic reduction uses the WCOJ planner/path where eligible | PARTIAL | Reduced body is marked `RequiresPlannerEligibility`; planner/runtime dispatch is not executed yet. |
| M090_GPU.3 GPU buffers | candidate, world-view, and rejection state have GPU-resident representations | PARTIAL | Buffer categories are explicit; runtime allocation/use is missing. |
| M090_GPU.4 kernel coverage | GPU kernels cover candidate generation, propagation, validation, and materialization hot paths | BLOCKED | No epistemic kernels are launched yet. |
| M090_GPU.5 CPU fallback ban | accepted execution trace records zero CPU candidate enumeration/world-view validation fallbacks | PARTIAL | Plan counters initialize to zero; accepted execution trace is missing. |
| M090_GPU.6 launch evidence | certification logs include nonzero GPU launch counts and kernel timing for epistemic execution | BLOCKED | No launch or timing evidence exists yet. |
| M090_GPU.7 parity | GPU output matches semantic oracle on all G91, FAEEL, GPT, and splitting fixtures | BLOCKED | No GPU output exists yet. |
| M090_GPU.8 transfer budget | host-device transfers are bounded and reported; no per-candidate host round trip in hot path | BLOCKED | No execution transfer trace exists yet. |

## Remaining Blocker

The next slice must attach this contract to runtime/CUDA execution: device buffer
allocation, kernel launch telemetry, WCOJ planner dispatch for eligible
reductions, and a real accepted execution trace with zero CPU fallback counters.
