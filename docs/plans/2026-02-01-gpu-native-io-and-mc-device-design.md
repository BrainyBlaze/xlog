# GPU-Native IO and MC Device Results Design
**Date:** February 1, 2026  
**Status:** Proposed  
**Scope:** GPU-native interop (Arrow device import), MC device-only results, stricter host-io gating

---

## Overview

This design formalizes two GPU-native improvements aligned with XLOG’s zero host-read contract:

1) **Experimental Arrow C Device import** (zero-copy) alongside existing Arrow IPC host-copy paths.
2) **Device-only Monte Carlo outputs** as the primary GPU-native API surface, with host probability
   computation gated behind `host-io`.

The goal is to keep the production path purely GPU-resident while preserving opt-in convenience
paths for users who need host-side results.

---

## Goals

- **Zero-copy Arrow device import** for CUDA-resident Arrow C Device arrays.
- **Device-only MC results** for GPU-native probabilistic inference.
- **Strict host-io gating** for any host-read convenience APIs.
- Maintain backward compatibility by keeping existing host-copy imports and host outputs behind
  feature flags.

## Non-Goals

- Multi-GPU Arrow import orchestration.
- Arrow IPC zero-copy import (out of scope; Arrow IPC remains host-copy).
- CLI device output mode (deferred).

---

## Architecture

The system separates **GPU-native core APIs** from **host-io convenience APIs**:

- GPU-native core: no device->host reads for data-plane results.
- Host-io convenience: opt-in via feature flag and explicit errors when disabled.

This preserves the “control-plane only” host role while still supporting ergonomic workflows
when host-io is enabled.

---

## Components and APIs

### 1) Arrow Device Import (Experimental)

Add a feature-flagged API to `xlog-cuda`:

- `CudaKernelProvider::from_arrow_device_record_batch(...) -> Result<CudaBuffer>`

Requirements:
- Device type must be CUDA (`ARROW_DEVICE_CUDA`).
- Device id must match the provider device.
- Schema must be compatible with XLOG scalar types.
- Null buffers must be rejected (no nullability support in device import).

Lifetime management:
- Wrap Arrow device buffers as `CudaColumn` without copies.
- Keep a device-side keepalive handle (mirroring the export path’s ownership model).

Python exposure (experimental):
- `pyxlog.import_arrow_device(...)` or equivalent capsule-based wrapper, feature-gated and clearly
  marked unstable.

### 2) MC Device Results (Primary GPU-Native API)

Promote and stabilize device-only MC evaluation:

- `McProgram::evaluate_gpu_device(...) -> McDeviceResult`

Python:
- `Program.evaluate_device(...)` returning DLPack buffers for `query_counts` and `evidence_count`
  plus metadata (samples, seed, confidence, nonmonotone stats).

Host probability computation remains:
- `McProgram::evaluate(...)` and `Program.evaluate(...)` behind `host-io`.

### 3) Host-IO Gating

Strictly gate host-read convenience APIs behind `#[cfg(feature = "host-io")]` in Rust and
explicit checks in Python/CLI. If disabled, return a clear error:

"Host output is disabled. Use device-resident APIs or rebuild with --features host-io."

---

## Data Flow

### Arrow Device Import
1) Validate device type/id and schema.
2) Wrap CUDA buffers into `CudaColumn` without copies.
3) Construct `CudaBuffer` with schema + device row counts.

### MC Device Results
1) GPU sampling produces a device-resident sample matrix.
2) GPU evaluation kernels compute query/evidence counts entirely on device.
3) `McDeviceResult` returns device buffers; Python exposes them via DLPack.
4) Host-io path (opt-in) downloads counts and computes probabilities/confidence intervals.

---

## Error Handling

- Arrow device import errors: unsupported type, schema mismatch, device mismatch, null buffers.
- MC device results: return counts even if evidence is unsatisfied; host-io path raises an error
  if evidence count is zero.
- Host-io disabled: explicit error messages in Python and CLI.

---

## Testing

### Arrow Device Import
- Round-trip: export -> import -> schema + row count check.
- Negative tests: device mismatch, unsupported type, null buffers.
- CUDA availability: tests skip cleanly when runtime is unavailable.

### MC Device Results
- Validate device counts match expected values.
- Keep `no_dtoh` guard tests to ensure GPU-native path remains clean.
- Python tests verifying DLPack consumption in Torch/CuPy.

### Host-IO Gating
- Source-scan tests to confirm `host-io` guards are present.
- Python/CLI error-path tests for host-io disabled builds.

---

## Rollout Plan

1) Add Arrow device import under `arrow-device-import` feature.
2) Expose Python `evaluate_device` for MC and DLPack counts.
3) Tighten host-io gating across Rust and Python.
4) Update docs:
   - `docs/architecture/cudf-interop.md` (device import, experimental)
   - `docs/architecture/python-bindings.md` (device-only MC)
   - `docs/design/2026-01-28-gpu-native-gaps.md` (close resolved gaps)

---

## Open Questions

- Should device import be gated in `xlog-cuda` only or also in top-level workspace features?
- Best Python API shape for Arrow device import (capsule vs direct pointer wrapper)?
- When to add CLI support for device outputs?
