# GPU Provenance + GPU PIR + GPU CNF Design (Phase 3)
**Date:** January 27, 2026
**Status:** Draft for implementation

## Goal
Deliver a full GPU-native provenance pipeline: GPU execution produces device-resident provenance (PIR), GPU PIR is interned deterministically, and GPU CNF is encoded on device for the D4/CDCL pipeline. No device-to-host data-plane transfers are allowed.

## Architecture Overview
Phase 3 extends the existing GPU runtime (xlog-runtime + xlog-cuda) with an optional provenance column (`prov_id: u32`) tracked alongside each relation. The execution plan (RIR) remains unchanged; provenance is computed in a runtime mode that appends and maintains `prov_id` using GPU kernels. PIR node creation is batched and interned on device using stable sort + dedup, producing a canonical `GpuPirGraph` (SoA) and root list on device. The GPU CNF encoder then performs a device-side Tseitin encoding over the `GpuPirGraph` and outputs `GpuCnf` + var tables for weights/evidence. All operations are deterministic and memory-bounded with fail-fast overflow semantics.

## Core Components
1. **GPU Provenance Column**
   - Extend runtime buffers with an optional `prov_id` column.
   - Join/AND: `prov_out = AND(prov_left, prov_right, ...)`
   - Union/Dedup/OR: merge duplicate tuples with OR of provenance.
   - Negation/WFS: compute provenance for non-monotone SCCs on GPU using WFS loop.

2. **GPU PIR Interner**
   - Device-side interning of PIR node keys using stable radix sort + dedup.
   - Keys are canonical: AND/OR children are sorted and deduped deterministically.
   - Outputs canonical `PirNodeId` and appends to `GpuPirGraph` pools.

3. **GPU CNF Encoder**
   - Device-side Tseitin encoding mirroring `crates/xlog-prob/src/cnf.rs`.
   - CSR sizing via device prefix-sum; no host reads.
   - Outputs `GpuCnf` + var tables (`node_var`, `leaf_var`, `choice_var`).

## Determinism and Failure Semantics
- Stable radix sort (existing kernels) + deterministic key packing.
- No nondeterministic atomics for ID assignment.
- Fixed-capacity pools; overflow => device trap.
- No CPU fallback, no UNKNOWN.

## CUDA Certification Requirements
Add certification coverage for:
- PIR interning determinism
- GPU CNF encoder correctness (counts + SAT checks via GPU CDCL)
- WFS non-monotone correctness on GPU
- Zero host reads in GPU provenance/CNF path
- Overflow/limit enforcement

## Integration Points
- Runtime: provenance mode for Executor (xlog-runtime)
- Probabilistic compilation: new GPU PIR/CNF modules in xlog-prob
- Verifier: reuse existing GPU CDCL verification

