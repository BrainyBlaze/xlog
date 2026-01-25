# GPU CDCL Verifier Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Update (Jan 25, 2026):** This work is implemented on branch `gpu-cdcl-solver`, and the verifier contract is now stricter than this original plan: the verifier path performs **zero device->host reads** (even scalar SAT/UNSAT status). See `docs/plans/2026-01-25-zero-host-reads-gpu-verifier.md` for the finalized contract and implementation plan.

**Goal:** Make `xlog-solve` provide a deterministic, complete **GPU-native CDCL SAT solver** (SAT/UNSAT) suitable as the verifier required by `docs/design/2026-01-22-gpu-native-compilation-design.md`.

**Architecture:** Add a new CUDA PTX module (`kernels/sat.ptx`) loaded by `xlog-cuda` and expose a `GpuCdclSolver` in `xlog-solve` that solves **device-resident CNF** with no data-plane host transfers during solve (host is control-plane only).

**Tech Stack:** Rust (workspace crates), CUDA C -> PTX (nvcc), `cudarc` (kernel launch + device memory), `xlog-cuda::{CudaKernelProvider,GpuMemoryManager}`.

---

## Notes / Constraints (Non-Negotiable)

- **100% GPU-native data-plane:** CNF, circuit, assignments, learned clauses stay on device during solve.
- **Deterministic:** no randomized validation; no nondeterministic “first-wins” atomics for reason selection.
- **Complete:** must return `SAT` with a model or `UNSAT` (no “maybe UNSAT”).
- **Control-plane allowed:** host may launch kernels / synchronize streams; avoid host memcpys except optional final result export.

---

### Task 1: Fix Rust Builds In This Sandbox (EXDEV rename)

**Files:**
- Create: `.cargo/config.toml`
- Create: `scripts/rustc-wrapper.sh`
- Create: `tools/exdev-shim/exdev_shim.c`
- Create: `tools/exdev-shim/build.sh`
- Modify: `.gitignore`

**Step 1: Add the EXDEV rename shim**

Create `tools/exdev-shim/exdev_shim.c` implementing an `LD_PRELOAD` override for `rename()` that falls back to copy+rename-in-dest-dir on `errno==EXDEV`.

**Step 2: Add shim build script**

Create `tools/exdev-shim/build.sh` that compiles `libexdev_shim.so` via:

```bash
cc -shared -fPIC -O2 -Wall -Wextra -o tools/exdev-shim/libexdev_shim.so tools/exdev-shim/exdev_shim.c -ldl
```

**Step 3: Wire shim into Cargo via rustc wrapper**

Create `scripts/rustc-wrapper.sh` and `.cargo/config.toml`:

```toml
[build]
rustc-wrapper = "scripts/rustc-wrapper.sh"
```

Wrapper sets `LD_PRELOAD=tools/exdev-shim/libexdev_shim.so` and execs the real rustc.

**Step 4: Verify build works**

Run:
- `cargo build -p xlog-core`

Expected: succeeds without `Invalid cross-device link (os error 18)`.

**Step 5: Commit**

```bash
git add .cargo/config.toml scripts/rustc-wrapper.sh tools/exdev-shim/exdev_shim.c tools/exdev-shim/build.sh .gitignore
git commit -m "build: work around EXDEV rename by preloading shim for rustc"
```

---

### Task 2: Rewrite Solver Services Doc To Match GPU-Native CDCL Verifier Reality

**Files:**
- Modify: `docs/architecture/solver-services.md`

**Step 1: Write doc changes**

Update the document to:
- Clearly separate **heuristic CLS** (optional) from **complete GPU CDCL verifier** (required).
- Remove implied CPU fallback language for UNSAT verification (CDCL is on GPU).
- Define the solver service API in terms of **device-resident CNF** and **device-resident model**.

**Step 2: Commit**

```bash
git add docs/architecture/solver-services.md
git commit -m "docs: update solver services to GPU CDCL verifier + optional CLS"
```

---

### Task 3: Add SAT/PTX Module To xlog-cuda (GPU CDCL + Proof Checking)

**Files:**
- Modify: `crates/xlog-cuda/build.rs`
- Modify: `crates/xlog-cuda/src/provider.rs`
- Create: `kernels/sat.cu`

**Step 1: Add production SAT kernels (no stubs / no placeholders)**

Create `kernels/sat.cu` implementing these GPU entrypoints:

- `sat_cdcl_solve`: complete, deterministic CDCL with watched literals + 1-UIP learning + restarts + clause DB management.
- `sat_check_model`: checks CNF satisfaction for a device-resident assignment.
- `sat_proof_check`: checks UNSAT soundness by verifying the solver-emitted **resolution-trace proof** on GPU.

**Step 2: Ensure build.rs compiles sat.cu**

Add `"sat"` to the `kernels` list in `crates/xlog-cuda/build.rs`.

**Step 3: Load SAT module in provider**

In `crates/xlog-cuda/src/provider.rs`:
- Add `SAT_PTX` include_str.
- Add `SAT_MODULE` const and kernel name constants:
  - `sat_kernels::SAT_CDCL_SOLVE`
  - `sat_kernels::SAT_CHECK_MODEL`
  - `sat_kernels::SAT_PROOF_CHECK`
- Load module in `CudaKernelProvider::new`.

**Step 4: Add PTX presence test (no CUDA required)**

Add a small `#[test]` in `crates/xlog-cuda/src/provider.rs` (or a new test file) that:
- Asserts `SAT_PTX.contains("sat_cdcl_solve")`
- Asserts `SAT_PTX.contains("sat_check_model")`
- Asserts `SAT_PTX.contains("sat_proof_check")`

**Step 5: Run tests**

Run:
- `cargo test -p xlog-cuda --lib`

Expected: pass (skips if no CUDA device, if needed).

**Step 6: Commit**

```bash
git add crates/xlog-cuda/build.rs crates/xlog-cuda/src/provider.rs kernels/sat.cu
git commit -m "cuda: add GPU CDCL SAT solver + on-GPU proof checking"
```

---

### Task 4: Add GPU CNF Representation In xlog-solve

**Files:**
- Create: `crates/xlog-solve/src/gpu_cnf.rs`
- Modify: `crates/xlog-solve/src/lib.rs`

**Step 1: Implement `GpuCnf`**

Create:
- `GpuCnf` with host-known capacities and device-resident exact sizes:
  - `var_cap`, `clause_cap`, `lit_cap`
  - `num_vars`, `num_clauses`, `num_lits` as `TrackedCudaSlice<u32>` (len=1)
  - `clause_offsets` sized to `clause_cap + 1`
  - `literals` sized to `lit_cap`
- `GpuCnf::from_host(instance, &CudaKernelProvider) -> Result<GpuCnf>` (host->device copy for tests/tools)

**Step 2: Export from crate**

Re-export `GpuCnf` from `crates/xlog-solve/src/lib.rs`.

**Step 3: Add unit tests**

Add `crates/xlog-solve/tests/gpu_cnf_tests.rs` that:
- Builds a tiny CNF
- Uploads to `GpuCnf`
- Downloads buffers and asserts offsets/literals match the DIMACS encoding

**Step 4: Run tests**

Run:
- `cargo test -p xlog-solve --tests gpu_cnf_tests -- --nocapture`

Expected: pass or skip when no CUDA device.

**Step 5: Commit**

```bash
git add crates/xlog-solve/src/gpu_cnf.rs crates/xlog-solve/src/lib.rs crates/xlog-solve/tests/gpu_cnf_tests.rs
git commit -m "solve: add GPU CNF CSR representation"
```

---

### Task 5: Implement GPU CDCL Solver (Watched Literals + Proof-Checked UNSAT)

**Files:**
- Modify: `kernels/sat.cu`
- Modify: `crates/xlog-cuda/src/provider.rs`
- Create: `crates/xlog-solve/src/gpu_cdcl.rs`
- Modify: `crates/xlog-solve/src/lib.rs`

**Step 1: Write failing tests (GPU SAT + UNSAT, fully verified)**

Create `crates/xlog-solve/tests/gpu_cdcl_tests.rs` with two tests:
- SAT: `(x1)` should be SAT with `x1=true`
- UNSAT: `(x1) ∧ (¬x1)` should be UNSAT

Additionally:
- SAT must pass `sat_check_model` on GPU.
- UNSAT must pass `sat_proof_check` on GPU.

Expected initially: FAIL because solver is not implemented.

**Step 2: Implement production CDCL (as per design doc)**

Implement the CDCL core exactly as required by `docs/design/2026-01-22-gpu-native-compilation-design.md`:
- watched literals + per-literal watch lists
- 1-UIP conflict analysis + clause learning
- deterministic restarts
- bounded, deterministic clause database management (eviction + watcher removal)
- proof trace emission suitable for `sat_proof_check`

**Step 3: Wire kernel into provider**

Add `sat_kernels::{SAT_CDCL_SOLVE,SAT_CHECK_MODEL,SAT_PROOF_CHECK}` and load them in `CudaKernelProvider::new`.

**Step 4: Implement Rust orchestrator**

In `crates/xlog-solve/src/gpu_cdcl.rs`:
- `GpuCdclConfig` (caps for learned clauses/lits, optional conflict limit)
- Expectation-based verifier API (no device->host reads):
  - `GpuCdclSolver::solve_expect_sat(&self, cnf: &GpuCnf) -> Result<TrackedCudaSlice<i8>>`
  - `GpuCdclSolver::solve_expect_unsat(&self, cnf: &GpuCnf) -> Result<()>`
  - Uses GPU-side assertion kernels + trap semantics so the host observes only CUDA success/failure.

**Step 5: Make tests pass**

In tests, allow downloading the assignment only for assertion purposes.

**Step 6: Run tests**

Run:
- `cargo test -p xlog-solve --test gpu_cdcl_tests -- --nocapture`

Expected: PASS (or skip if no CUDA device).

**Step 7: Commit**

```bash
git add kernels/sat.cu crates/xlog-cuda/src/provider.rs crates/xlog-solve/src/gpu_cdcl.rs crates/xlog-solve/src/lib.rs crates/xlog-solve/tests/gpu_cdcl_tests.rs
git commit -m "solve: add GPU CDCL SAT solver kernel + Rust API"
```

---

### Task 6: Integrate GPU CDCL Verifier Hook In xlog-prob (No CPU Validation)

**Files:**
- Modify: `crates/xlog-prob/src/compilation/validation.rs`
- Modify: `crates/xlog-prob/Cargo.toml`

**Step 1: Replace placeholder validation**

Change `validate_against_cpu_evaluator` into a GPU-equivalence-check entrypoint:
- `validate_equivalence_gpu(phi: &GpuCnf, circuit: &GpuXgcf, provider: &CudaKernelProvider) -> Result<()>`
- Internally builds the two CNFs `phi ∧ ¬C` and `C ∧ ¬phi` on GPU and calls `GpuCdclSolver`.

**Step 2: Add dependency**

Add `xlog-solve = { path = \"../xlog-solve\" }` to `crates/xlog-prob/Cargo.toml`.

**Step 3: Add a small integration test (skippable)**

Create `crates/xlog-prob/tests/gpu_equivalence_smoke.rs`:
- Build a tiny φ and an equivalent trivial circuit C
- Assert verification returns OK.

**Step 4: Run tests**

Run:
- `cargo test -p xlog-prob --tests gpu_equivalence_smoke -- --nocapture`

**Step 5: Commit**

```bash
git add crates/xlog-prob/src/compilation/validation.rs crates/xlog-prob/Cargo.toml crates/xlog-prob/tests/gpu_equivalence_smoke.rs
git commit -m "prob: replace CPU validation stub with GPU CDCL equivalence verifier hook"
```

---

### Task 7: Repo-Wide Verification

**Step 1: Build**

Run:
- `cargo build --workspace`

**Step 2: Test**

Run:
- `cargo test --workspace`

**Step 3: Document updates**

Ensure:
- `docs/architecture/solver-services.md` matches implementation
- `docs/design/2026-01-22-gpu-native-compilation-design.md` references the GPU CDCL verifier entrypoint correctly.
