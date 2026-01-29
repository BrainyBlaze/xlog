# GPU-Native Gaps Report (XLOG)
**Date:** January 28, 2026  
**Status:** Current gaps in 100% GPU-native execution  
**Scope:** GPU-native knowledge compilation, SAT verification, and execution paths

---

## Executive Summary

This document records the remaining gaps that prevent XLOG from being a fully GPU-native system across
compilation, verification, and execution. Each gap is backed by a concrete code or documentation reference.
The gaps below are **actionable constraints** that still force host-side work or limit GPU-native coverage.

---

## Gap Inventory (Evidence + Impact)

### 1) Exact D4 path still uses CPU D4 and host CNF/DDNNF I/O
**Evidence:** `crates/xlog-prob/src/exact.rs` writes DIMACS CNF to disk, calls the vendored D4 binary, and
parses DDNNF on host (`TempDirGuard`, `fs::write`, `D4Compiler::detect`, `compile_ddnnf`).  
**Impact:** The default exact pipeline is not GPU-native and requires host serialization + CPU compilation.  
**Required resolution:** Replace the CPU D4 invocation in `ExactDdnnfProgram::compile_provenance` with
GPU-native compilation (`encode_cnf_gpu` + `compile_gpu_d4_and_verify`) and device-resident circuit emission.

### 2) GPU CNF encoder is not wired into the default exact pipeline
**Evidence:** `docs/architecture/xlog-prob.md` states `encode_cnf_gpu` exists but is not wired into the default
`ExactDdnnfProgram`. The code path in `exact.rs` uses `encode_cnf` (host) rather than `encode_cnf_gpu`.  
**Impact:** The GPU-native CNF pipeline is available but unused in production, blocking end-to-end GPU-native
compilation.  
**Required resolution:** Route `ExactDdnnfProgram` through the GPU CNF encoder and GPU D4 verifier path.

### 3) SAT/CDCL certification category missing in `xlog-cuda-tests`
**Evidence:** `docs/architecture/solver-services.md` calls for SAT certification categories; `docs/ROADMAP.md`
tracks the need for SAT/CDCL certification coverage; `crates/xlog-cuda-tests` currently has no SAT category.  
**Impact:** The GPU CDCL solver lacks kernel-level certification coverage, weakening production guarantees.  
**Required resolution:** Add SAT/CDCL certification categories to `xlog-cuda-tests` with deterministic,
GPU-only SAT/UNSAT correctness, proof validation, and failure-mode coverage.

### 4) Monte Carlo path is not GPU-native
**Evidence:** `crates/xlog-prob/src/mc.rs` downloads the Bernoulli sample matrix to host
(`sample_bernoulli_matrix` returns `Vec<u8>`) and performs evaluation on CPU.  
**Impact:** The MC pipeline violates the “zero host transfers” GPU-native goal for probabilistic inference.  
**Required resolution:** Implement GPU-resident MC evaluation (sampling, evidence filtering, query counts) and
return device-resident results for optional host readout only.

### 5) Arrow interop still requires host copies
**Evidence:** `docs/architecture/cudf-interop.md` states Arrow export/import performs device↔host copies; only
DLPack paths are zero-copy.  
**Impact:** Arrow-based interop breaks GPU-native data residency for workloads that rely on Arrow I/O.  
**Required resolution:** Provide a zero-copy Arrow device memory extension path or explicitly constrain
GPU-native workflows to DLPack-only interop.

### 6) Host-read convenience APIs remain in GPU evaluation
**Evidence:** `crates/xlog-prob/src/gpu.rs` exposes `eval_log_wmc` and `eval_log_wmc_and_grads` which download
root value and gradients to host.  
**Impact:** These APIs enable host reads in evaluation, so GPU-native call sites must avoid them or use
device-only variants.  
**Required resolution:** Ensure production GPU-native paths exclusively use device-resident evaluation APIs
(`eval_log_wmc_device_into`, `eval_grads_inplace`, and CUDA tensor outputs) and avoid DTOH reads.

---

## Notes on Scope

- This report focuses on **GPU-native compilation and execution** gaps. It does not enumerate multi-GPU or
out-of-core features unless they block GPU-native correctness.
- Gaps are listed as **concrete, verifiable** constraints. Each item is tied to a specific file or documented
behavior.

---

## Summary

XLOG’s GPU-native compiler and verifier core are in place, but the default exact path still traverses CPU D4
and host I/O, the MC path remains host-resident, Arrow interop is not zero-copy, and solver certification gaps
remain. Addressing these items is required to claim a fully GPU-native, zero-transfer execution pipeline.
