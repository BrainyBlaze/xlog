# GPU-Native Gaps Report (XLOG)
**Date:** January 28, 2026  
**Status:** Updated (many gaps closed)  
**Scope:** GPU-native knowledge compilation, SAT verification, and execution paths

---

## Executive Summary

This document originally recorded the remaining gaps preventing XLOG from being a fully GPU-native system.
As of the February 2026 iteration, most items below have been closed in code (GPU-native exact compile,
GPU-native MC device evaluation, SAT/CDCL certification categories, and strict host-io gating).

The remaining constraints are primarily around **interop**: Arrow IPC remains host-copy and Arrow C Device
import is experimental and limited in supported types (no nulls; no bit-packed bool import yet).

---

## Gap Inventory (Evidence + Impact)

### 1) Exact D4 path still uses CPU D4 and host CNF/DDNNF I/O
**Status:** Resolved.  
**Evidence:** `crates/xlog-prob/src/exact.rs` routes compilation through GPU CNF encoding + GPU D4 compile
and verification (`encode_cnf_gpu`, `compile_gpu_d4_and_verify_cached`).  
**Impact:** Exact compilation is GPU-native in the steady-state path. Host output remains opt-in via `host-io`.

### 2) GPU CNF encoder is not wired into the default exact pipeline
**Status:** Resolved.  
**Evidence:** `crates/xlog-prob/src/exact.rs` uses `encode_cnf_gpu` in the default GPU compilation pipeline.

### 3) SAT/CDCL certification category missing in `xlog-cuda-tests`
**Status:** Resolved.  
**Evidence:** `crates/xlog-cuda-tests/src/categories/g07_sat_cdcl.rs` exists and covers SAT/CDCL certification.

### 4) Monte Carlo path is not GPU-native
**Status:** Resolved (device-native primary path).  
**Evidence:** `crates/xlog-prob/src/mc.rs` exposes `evaluate_gpu_device_with_provider` returning
`McDeviceResult` with device-resident `query_counts` and `evidence_count`. Host probability computation is
behind `host-io`.  
**Impact:** GPU-native MC workflows can consume counts without DTOH reads. Python exposes this as
`CompiledProgram.evaluate_device(...)` returning DLPack counts.

### 5) Arrow interop still requires host copies
**Status:** Partially resolved.  
**Evidence:** `crates/xlog-cuda/src/provider.rs` supports zero-copy Arrow C Device export
(`to_arrow_device_record_batch`) and experimental zero-copy import behind `--features arrow-device-import`
(`from_arrow_device_record_batch`). Arrow IPC remains host-copy.  
**Impact:** GPU-native Arrow interop is available for supported types, but the production-recommended
interop path remains DLPack (broadest support and simplest invariants).

### 6) Host-read convenience APIs remain in GPU evaluation
**Status:** Resolved via strict gating.  
**Evidence:** `crates/xlog-prob/src/gpu.rs` host-read convenience APIs are gated behind `#[cfg(feature = "host-io")]`,
and device-only APIs exist (`eval_log_wmc_device_into`, `eval_grads_inplace`, etc).  
**Impact:** GPU-native call sites can enforce "no host reads" by leaving `host-io` disabled.

---

## Notes on Scope

- This report focuses on **GPU-native compilation and execution** gaps. It does not enumerate multi-GPU or
out-of-core features unless they block GPU-native correctness.
- Gaps are listed as **concrete, verifiable** constraints. Each item is tied to a specific file or documented
behavior.

---

## Summary

XLOG’s GPU-native compiler, verifier, and MC device evaluation are in place and feature-gated host output
APIs prevent accidental DTOH reads in production GPU-native paths. Remaining gaps are concentrated in the
interop layer (Arrow IPC host-copy, and experimental/limited Arrow C Device import support).
