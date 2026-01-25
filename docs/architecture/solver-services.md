# Solver Services (xlog-solve)

This document describes XLOG's SAT solver services. The **production correctness path is GPU-native**: SAT/UNSAT is decided on device with a **complete CDCL solver**, and results are returned as **device-resident buffers**.

## Why This Exists

XLOG uses SAT solving in multiple subsystems:

- **Knowledge compilation verification** (`xlog-prob`): prove `φ ≡ C` by checking two UNSAT queries on GPU:
  - `UNSAT(φ ∧ ¬C)`
  - `UNSAT(C ∧ ¬φ)`
- **D4-style compilation** (GPU D4): unit propagation, decomposition, and (optionally) SAT calls during compilation.
- **ASP/ELP-style workflows** (future): candidate model checks and brave/cautious consequence checks.

The verifier must be **complete**. Heuristic solvers (CLS, local search) are allowed only as *optional accelerators*, never as the final authority.

## Zero CPU Transfers (Data-Plane Contract)

In the GPU-native path:

- **CNF inputs are device-resident** (e.g., already on GPU from PIR/CNF building or imported via DLPack).
- **Solver state is device-resident** (assignments, trail, learned clauses).
- The host may launch kernels and synchronize streams (**control-plane**) but does not copy CNF/circuit/state back and forth (**data-plane**).

The solver may optionally export a final SAT model or UNSAT certificate, but the default verifier integration consumes device-resident outputs directly.

## GPU CDCL Verifier (Required)

### Interface

`xlog-solve` exposes a GPU solver that accepts **GPU CNF** and produces a SAT/UNSAT decision:

```rust
pub struct GpuCnf {
    pub num_vars: u32,
    pub num_clauses: u32,
    pub clause_offsets: TrackedCudaSlice<u32>, // len = num_clauses + 1
    pub literals: TrackedCudaSlice<i32>,       // signed DIMACS: ±var_id (1-based)
}

pub struct GpuCdclSolver {
    pub provider: Arc<CudaKernelProvider>,
    pub config: GpuCdclConfig,
}

pub enum GpuSolveStatus { Sat, Unsat }

pub struct GpuCdclResult {
    pub status: GpuSolveStatus,
    pub assignment: TrackedCudaSlice<i32>, // len = num_vars + 1, values {-1,0,1}
}
```

**Note:** `assignment` stays on device; higher-level verifiers can keep it device-resident or export it.

### Core Data Structures (Device)

The CDCL implementation uses a fixed-capacity arena on GPU:

- **CNF (CSR):** `clause_offsets[]` and `literals[]`.
- **Assignment:** `assign[var] ∈ {-1, 0, +1}` for false/unassigned/true.
- **Trail:** `trail_vars[]`, `trail_len`.
- **Decision levels:** `level[var]`, `level_start[level]`.
- **Reasons:** `reason[var] = clause_id` for implied assignments, `-1` for decisions.
- **Learned clause arena:** append-only `(offsets, lits)` with configurable capacity.

The verifier prioritizes deterministic correctness over aggressive micro-optimizations:

- **BCP:** scan-based unit propagation to a fixpoint (deterministic, massively parallel).
- **Conflict analysis:** 1-UIP clause learning and non-chronological backjumping.
- **Heuristic:** deterministic variable selection (upgradeable to deterministic VSIDS-style scoring).

### Determinism Contract

To keep verification reproducible, the solver must be deterministic for a fixed input CNF:

- Conflict clause selection uses a deterministic reducer (e.g., min clause id).
- When multiple clauses imply the same literal, reason selection uses a deterministic rule (e.g., min clause id).
- No random restarts; if restarts are used, they must be schedule-based and deterministic.

### CUDA Kernels

The solver is implemented as a GPU kernel entrypoint loaded by `xlog-cuda`:

- `sat_cdcl_solve`: executes the CDCL loop on device and writes:
  - `out_status` (SAT/UNSAT)
  - `assign[]` (model when SAT)
  - optional debugging counters and conflict stats

The host launches this kernel and waits for completion. The solver’s internal loop does not require host-device memcpy.

## Continuous Local Search (Optional, Non-Verifying)

`xlog-solve` also contains a Continuous Local Search (CLS) solver (FastFourierSAT-inspired) for:

- fast best-effort SAT guesses
- MaxSAT approximations

CLS is **not complete** and must never be used as the verifier. It may be used to seed CDCL (future) by providing an initial assignment on GPU.

## Integration Notes

### xlog-prob (GPU-Native Compilation)

The verifier integration uses `GpuCdclSolver` to solve the two UNSAT queries for equivalence. If a query is SAT, the solver returns a **device-resident counterexample assignment** for debugging (no silent fallback).

### SAT Subsystem Scope

The SAT subsystem in XLOG is defined as:

- GPU-resident CNF encoding
- GPU CDCL solver (complete SAT/UNSAT)
- GPU equivalence checking helpers used by compilation and future non-monotone logic

## Testing Strategy (GPU-Only)

- Unit tests: small SAT/UNSAT instances, determinism checks.
- Property tests: random CNFs where results are cross-checked against a tiny CPU reference *in tests only*.
- Kernel certification: add a SAT category to `xlog-cuda-tests` once the kernel API stabilizes.

## See Also

- `docs/design/2026-01-22-gpu-native-compilation-design.md`
- `docs/research/2026-01-22-architecture-validation-and-refinement.md`
- `crates/xlog-prob/src/compilation/` (GPU compilation + verification entrypoints)

