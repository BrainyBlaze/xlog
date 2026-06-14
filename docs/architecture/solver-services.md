# Solver Services (xlog-solve)

This document describes XLOG's SAT solver services. The **production correctness path is GPU-native**: SAT/UNSAT is decided on device with a **complete CDCL solver**, and results are returned as **device-resident buffers**.

## Why This Exists

XLOG uses SAT solving in multiple subsystems:

- **Knowledge compilation verification** (`xlog-prob`): prove `φ ≡ C` by checking two UNSAT queries on GPU:
  - `UNSAT(φ ∧ ¬C)`
  - `UNSAT(C ∧ ¬φ)`
- **Decision-DNNF-style compilation** (GPU Decision-DNNF): unit propagation,
  decomposition, and (optionally) SAT calls during compilation.
- **ASP/ELP-style workflows** (future): candidate model checks and brave/cautious consequence checks.

The verifier must be **complete**. Heuristic solvers (CLS, local search) are allowed only as *optional accelerators*, never as the final authority.

## Zero CPU Transfers (Data-Plane Contract)

In the GPU-native path:

- **CNF inputs are device-resident** (e.g., already on GPU from PIR/CNF building or imported via DLPack).
- Device-resident CNF sources include `xlog_prob::compilation::encode_cnf_gpu` (PIR→CNF) and the XGCF helpers
  in `kernels/sat.cu` (`sat_xgcf_cnf_*`), both of which keep exact sizes on device.
- **Solver state is device-resident** (assignments, trail, learned clauses).
- The host may launch kernels and synchronize streams (**control-plane**) but does not copy CNF/circuit/state back and forth (**data-plane**).

Verifier-grade integrations additionally enforce **zero device->host reads**: even the SAT/UNSAT status is not copied back. The host observes only CUDA success/failure while GPU-side assertion kernels validate the outcome and trap on mismatch.

## GPU CDCL Verifier (Required)

### Interface

`xlog-solve` exposes a GPU solver that accepts **GPU CNF**. To support fully GPU-native construction (where exact sizes are computed on device), CNF size metadata is also device-resident:

```rust
pub struct GpuCnf {
    // Host-known capacities (buffers are allocated to these sizes).
    pub var_cap: u32,
    pub clause_cap: u32,
    pub lit_cap: u32,

    // Device-resident exact counts (len = 1 each).
    pub num_vars: TrackedCudaSlice<u32>,
    pub num_clauses: TrackedCudaSlice<u32>,
    pub num_lits: TrackedCudaSlice<u32>,

    // CSR buffers sized by capacity.
    pub clause_offsets: TrackedCudaSlice<u32>, // len = clause_cap + 1
    pub literals: TrackedCudaSlice<i32>,       // len = lit_cap, signed DIMACS: ±var_id (1-based)
}

pub struct GpuCdclSolver {
    /* internal fields */
}

impl GpuCdclSolver {
    pub fn new(provider: Arc<CudaKernelProvider>, config: GpuCdclConfig) -> Self;
    pub fn solve_expect_sat(&self, cnf: &GpuCnf) -> xlog_core::Result<TrackedCudaSlice<i8>>;
    pub fn solve_expect_unsat(&self, cnf: &GpuCnf) -> xlog_core::Result<()>;
}
```

**Verifier semantics (zero host reads):**
- `solve_expect_sat`: runs CDCL, asserts SAT on GPU, runs GPU model check, and returns the device-resident assignment.
- `solve_expect_unsat`: runs CDCL, asserts UNSAT on GPU, runs GPU proof check, and returns `Ok(())`.

If the solver returns the wrong status or produces an invalid model/proof, GPU-side assertion kernels trap so the host cannot continue silently.

### Core Data Structures (Device)

The CDCL implementation uses a fixed-capacity arena on GPU:

- **CNF (CSR):** `clause_offsets[]` and `literals[]`.
- **Assignment:** `assign[var] ∈ {-1, 0, +1}` for false/unassigned/true.
- **Trail:** `trail_vars[]`, `trail_len`.
- **Decision levels:** `level[var]`, `level_start[level]`.
- **Reasons:** `reason[var] = clause_id` for implied assignments, `-1` for decisions.
- **Learned clause arena:** append-only `(offsets, lits)` with configurable capacity.

The verifier prioritizes deterministic correctness over aggressive micro-optimizations:

- **BCP:** watched literals with per-literal watch lists (deterministic traversal order).
- **Conflict analysis:** 1-UIP clause learning and non-chronological backjumping.
- **Heuristic:** deterministic variable selection (deterministic VSIDS-style scoring).
- **SAT validation:** every SAT result must pass an on-GPU model check (`sat_check_model`).
- **UNSAT validation:** every UNSAT result must pass an on-GPU proof check (`sat_proof_check`) using a solver-emitted
  resolution-trace certificate (no CPU proof checking).

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

The SAT PTX module also includes verifier helper kernels used by `xlog-solve` and `xlog-prob`:

- Assertion + validation: `sat_assert_status`, `sat_assert_ok`, `sat_check_model`, `sat_proof_check`
- XGCF→CNF construction: `sat_xgcf_cnf_counts`, `sat_xgcf_cnf_emit`, `sat_xgcf_cnf_capture_last_counts`,
  `sat_xgcf_cnf_compute_totals`, `sat_cnf_write_terminator`
- Equivalence query construction: `sat_cnf_copy_into`, `sat_xgcf_write_root_unit_clause`, `sat_not_phi_counts`,
  `sat_emit_not_phi`

## Production Adapter For Epistemic Callers

`xlog_solve::GpuSolverProductionAdapter` is the shipped GPU-native solver
service for epistemic callers: a thin adapter over the existing
`GpuCdclSolver` that makes solver production-path reuse auditable. It is the
production solver path, with the CPU oracle gated off for production metrics.

The adapter:

- constructs and owns `GpuCdclSolver::new`;
- dispatches SAT through `solve_expect_sat`;
- dispatches UNSAT through `solve_expect_unsat`;
- dispatches workspace-backed UNSAT through
  `solve_expect_unsat_with_branch_limit_ws`;
- dispatches bounded single-result and multi-candidate MaxSAT and SAT/MaxSAT
  portfolio jobs through the same GPU CNF/CDCL adapter;
- exposes `GpuSolverProductionTrace` counters for GPU CDCL SAT/UNSAT calls;
- exposes `production_capabilities`, where GPU CDCL SAT/UNSAT, bounded
  MaxSAT, and bounded SAT/MaxSAT portfolio adapters are `Available` while the
  CPU oracle remains disallowed for production metrics;
- exposes hard-zero CPU search counters:
  `cpu_assignment_enumerations` and `cpu_maxsat_enumerations`.

This is not a separate solver engine. It does not call `SolverService`, does not
enumerate assignments on CPU, and does not introduce an epistemic-only search
path. It is a **bounded** surface: GPU CDCL SAT/UNSAT, bounded MaxSAT, and
bounded SAT/MaxSAT portfolio jobs are wired and tested, but the surface is not
full coverage of all MaxSAT, portfolio, and weighted forms.
Broader solver semantic integration and a dedicated Solver IR (SIR) remain
future work.

The capability report is intentionally fail-closed. The CPU semantic-oracle
service and CPU CLS solver are not counted toward production metrics; accepted
solver work routes through the GPU-backed adapter and reports zero CPU search
counters.

## Semantic-Oracle Solver Service Semantics

XLOG also ships a CPU-side service facade for bounded semantic fixtures. It is
not the production verifier, it is not GPU-native epistemic solving, and it does
not dispatch epistemic solving to a GPU portfolio. This facade is an
oracle/reference surface used for testing and fixture-scale semantics; the
GPU-backed `GpuSolverProductionAdapter` is the production solver path.

`SolverService` owns a `SolveInstance` and exposes:

- incremental SAT assumptions through `assume` and `retract_assumption`
- learned-clause transfer observability through `transfer_learned_clauses_to` and `SolverServiceTrace`
- exact fixture-scale MaxSAT scoring for `SolveInstance::with_weights`
- explicit service statuses: `Sat`, `Unsat`, `Unknown`, `Timeout`, and `Optimal`
- an explicit GPU portfolio unimplemented status through `gpu_portfolio_status`

Incremental assumptions are scoped. Clauses learned while temporary assumptions are active are only applied while those same assumption literals remain active, so retracting an assumption cannot leave behind an unconditional contradiction. Transfer records the number of learned clauses delivered to another service and preserves their scope.

MaxSAT support is deliberately fixture-scale: `SolverService` enumerates assignments for bounded tests, treats weighted clauses as soft constraints, and returns the best integer score as `Optimal(score)`. The service separates non-search (`Unknown`) from exhausted UNSAT (`Unsat`) and zero-budget bounded search (`Timeout`) so callers can test failure-mode routing without relying on GPU availability.

GPU portfolio solving is not implemented in this facade. `gpu_portfolio_status`
returns a `Deferred` status with this rationale:

```text
GPU portfolio solving is not implemented in the semantic-oracle facade; use the GPU-backed production adapter
```

This CPU assignment-enumeration path is a test/reference surface only. The
production path for accepted epistemic execution is the GPU-backed
`GpuSolverProductionAdapter`, which routes SAT/MaxSAT/portfolio jobs through GPU
CDCL with the CPU oracle gated off (`cpu_oracle_solver_allowed` defaults to
false) and reports zero CPU solver-search fallback counters.

## Continuous Local Search (Optional, Non-Verifying)

`xlog-solve` also contains a Continuous Local Search (CLS) solver (FastFourierSAT-inspired) for:

- fast best-effort SAT guesses
- MaxSAT approximations

CLS is **not complete** and must never be used as the verifier. It may be used to seed CDCL (future) by providing an initial assignment on GPU.

## Integration Notes

### xlog-prob (GPU-Native Compilation)

The verifier integration solves the two UNSAT queries for equivalence via `GpuCdclSolver::solve_expect_unsat`. A SAT result indicates a compiler bug and is handled as **fail-fast** (GPU trap / CUDA error), without copying any status or counters to the host.

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

- [`../whitepaper/main.pdf`](../whitepaper/main.pdf) — whitepaper discussion of the GPU CDCL verifier in the compiled reasoning pipeline
- [`xlog-prob.md`](xlog-prob.md) — probabilistic tier that consumes the verifier
- `crates/xlog-prob/src/compilation/` — GPU compilation and verification entrypoints
