# Zero Host Reads GPU Verifier Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

> **Status (Jan 25, 2026):** Implemented on branch `gpu-cdcl-solver` with an additional robustness requirement:
> equivalence query construction must treat `GpuCnf::{var_cap,clause_cap,lit_cap}` as *capacities* only and must use
> device-resident `GpuCnf::{num_vars,num_clauses,num_lits}` for all index math (supports GPU-native builders where
> capacity > exact size).

**Goal:** Make the GPU CDCL verifier + GPU equivalence validation run with **zero device→host memory transfers** (no `dtoh` reads), while preserving production-grade correctness guarantees (SAT model check, UNSAT proof check) fully on GPU.

**Architecture:** Replace host-side branching/inspection with a GPU-only validation chain:
1) CDCL solve writes device-resident `status/error/learned_count`.
2) GPU kernels validate the result (model/proof) and **trap** on contract violation.
3) The host observes only CUDA success/failure (control-plane), never reads device memory.

**Tech Stack:** CUDA C kernels compiled to PTX (nvcc), Rust orchestrators in `xlog-solve`/`xlog-prob`, `cudarc` kernel launch, `xlog-cuda::{CudaKernelProvider,GpuMemoryManager}`.

---

## Notes / Constraints (Non-Negotiable)

- **No host reads:** no device→host copies (`dtoh_sync_copy_into`, `copy_to_host`, etc.) in production GPU verifier paths.
- **Deterministic + complete:** CDCL remains deterministic and must decide SAT/UNSAT (no “unknown”).
- **GPU-side validation only:** SAT must pass GPU model check; UNSAT must pass GPU resolution-trace proof check.
- **Fail-fast:** If an equivalence UNSAT query returns SAT or invalid proof, GPU must trap (CUDA error) so the host cannot continue.
- **No stubs / TODOs / placeholders.**

---

### Task 1: Add GPU Assertion Kernels (Fail-Fast, No Host Reads)

**Files:**
- Modify: `kernels/sat.cu`
- Modify: `crates/xlog-cuda/src/provider.rs`
- Modify: `crates/xlog-cuda/build.rs` (only if needed)

**Step 1: Add GPU trap helper + assertion kernels**

In `kernels/sat.cu`, add:
- `sat_assert_status(expected_status)`: reads `out_status/out_error` and traps unless `out_error==0 && out_status==expected_status`.
- `sat_assert_ok`: reads `out_ok` and traps unless `out_ok==1`.

**Step 2: Make proof checker accept learned_count from device**

Update `sat_proof_check` signature so it does **not** require host-provided `learned_count`.
Pass `const uint32_t* learned_count` (len=1) and read it on device.

**Step 3: Update provider kernel name constants**

In `crates/xlog-cuda/src/provider.rs`:
- Add `sat_kernels::SAT_ASSERT_STATUS`
- Add `sat_kernels::SAT_ASSERT_OK`
- Ensure PTX presence tests assert these symbols exist in `SAT_PTX`.

**Step 4: Verify**

Run:
- `cargo test -p xlog-cuda --lib`

Expected: PASS.

**Step 5: Commit**

```bash
git add kernels/sat.cu crates/xlog-cuda/src/provider.rs
git commit -m "cuda: add SAT verifier assertion kernels (zero host reads)"
```

---

### Task 2: Add Device-Counted CNF Meta (No dtoh sizing reads)

**Files:**
- Modify: `kernels/sat.cu`
- Modify: `crates/xlog-cuda/src/provider.rs`

**Step 1: Add XGCF CNF totals kernel**

In `kernels/sat.cu`, add kernels that:
- Capture the last pre-scan counts into device scalars.
- After in-place exclusive scans, compute `internal_total/clause_total/lit_total` on device, validate against capacities, and write:
  - `out_num_vars` (base_num_vars + internal_total)
  - `out_num_clauses` (clause_total)
  - `out_num_lits` (lit_total)

Also remove host-provided `total_clauses/total_lits` dependency from `sat_xgcf_cnf_emit` by moving CSR terminator write into a small finalize kernel that reads device totals.

**Step 2: Update provider kernel constants and PTX tests**

Add kernel names to `sat_kernels` and to the embedded PTX symbol assertions.

**Step 3: Verify**

Run:
- `cargo test -p xlog-cuda --lib`

Expected: PASS.

**Step 4: Commit**

```bash
git add kernels/sat.cu crates/xlog-cuda/src/provider.rs
git commit -m "cuda: compute XGCF->CNF totals on GPU (no host sizing reads)"
```

---

### Task 3: Zero-Host-Read GPU CDCL API in xlog-solve (Expectation-Based)

**Files:**
- Modify: `crates/xlog-solve/src/gpu_cdcl.rs`
- Modify: `crates/xlog-solve/src/lib.rs`
- Modify: `crates/xlog-solve/tests/gpu_cdcl_tests.rs`

**Step 1: Replace host-reading `solve()` with GPU-validated expectation APIs**

Implement:
- `solve_expect_sat(&self, cnf: &GpuCnf) -> Result<TrackedCudaSlice<i8>>`
  - launches `sat_cdcl_solve`
  - launches `sat_assert_status(SAT)`
  - launches `sat_check_model`
  - launches `sat_assert_ok`
  - returns the device-resident assignment

- `solve_expect_unsat(&self, cnf: &GpuCnf) -> Result<()>`
  - launches `sat_cdcl_solve`
  - launches `sat_assert_status(UNSAT)`
  - launches `sat_proof_check` (learned_count read on device)
  - launches `sat_assert_ok`

Remove all `dtoh_sync_copy_into` from the production solver path.

**Step 2: Update tests**

Update `crates/xlog-solve/tests/gpu_cdcl_tests.rs`:
- SAT test calls `solve_expect_sat` and then (test-only) downloads the assignment to assert `x1=true`.
- UNSAT test calls `solve_expect_unsat` and asserts it returns `Ok(())`.

**Step 3: Verify**

Run:
- `cargo test -p xlog-solve --test gpu_cdcl_tests -- --nocapture`

Expected: PASS (or skip if CUDA runtime unavailable).

**Step 4: Commit**

```bash
git add crates/xlog-solve/src/gpu_cdcl.rs crates/xlog-solve/tests/gpu_cdcl_tests.rs crates/xlog-solve/src/lib.rs
git commit -m "solve: enforce SAT/UNSAT on GPU with zero host reads"
```

---

### Task 4: Zero-Host-Read GPU Equivalence Verification in xlog-prob

**Files:**
- Modify: `crates/xlog-prob/src/compilation/validation.rs`
- Modify: `crates/xlog-prob/tests/gpu_equivalence_smoke.rs`

**Step 1: Remove all dtoh reads from CNF construction**

Rewrite circuit CNF construction to:
- allocate capacities from host-known bounds (`num_nodes`, `child_indices.len()`)
- compute exact totals on GPU via Task 2 kernels
- use device-resident totals for downstream kernel indices (unit clause placement, shift bases, etc.)

**Additional requirement (hardening):** The verifier must support `GpuCnf` inputs where capacity > exact size.
This requires GPU-side CNF concatenation and GPU-side size math for `¬φ`:

- Add `sat_cnf_copy_into`: copy a CSR CNF into an output CNF at device-resident clause/lit bases.
- Add `sat_not_phi_counts`: compute exact `¬φ` size contributions from device-resident `phi.num_*`.
- Update `sat_xgcf_write_root_unit_clause` to take device-resident `clause_base/lit_base` and `extra_num_*` (len=1 each).
- Add regression test: `crates/xlog-prob/tests/gpu_equivalence_padded_phi.rs` (phi cap > exact).

**Step 2: Make `validate_equivalence_gpu` fail-fast without inspecting status on host**

`validate_equivalence_gpu` should:
- build both query CNFs on GPU
- call `GpuCdclSolver::solve_expect_unsat` on both
- return `Ok(())` if both succeed; otherwise return the CUDA failure as `Err(...)`

No host-side SAT/UNSAT branching.

**Step 3: Update smoke test**

`gpu_equivalence_smoke` should:
- call `validate_equivalence_gpu` and assert `Ok(())` (skips if CUDA unavailable).

**Step 4: Verify**

Run:
- `cargo test -p xlog-prob --test gpu_equivalence_smoke -- --nocapture`

Expected: PASS (or skip if CUDA runtime unavailable).

**Step 5: Commit**

```bash
git add crates/xlog-prob/src/compilation/validation.rs crates/xlog-prob/tests/gpu_equivalence_smoke.rs
git commit -m "prob: GPU equivalence verifier with zero host reads"
```

---

### Task 5: Enforce “No dtoh” in GPU Verifier Paths (Regression Guard)

**Files:**
- Create: `crates/xlog-solve/tests/no_dtoh_in_gpu_cdcl.rs`
- Create: `crates/xlog-prob/tests/no_dtoh_in_gpu_equivalence.rs`

**Step 1: Add tests that reject dtoh usage in production verifier modules**

Implement tests that read the Rust source files and assert they do **not** contain:
- `dtoh_sync_copy_into`
- `copy_to_host`
- `dtoh`

Scope:
- `crates/xlog-solve/src/gpu_cdcl.rs`
- `crates/xlog-prob/src/compilation/validation.rs`

**Step 2: Verify**

Run:
- `cargo test -p xlog-solve --tests no_dtoh_in_gpu_cdcl -- --nocapture`
- `cargo test -p xlog-prob --tests no_dtoh_in_gpu_equivalence -- --nocapture`

Expected: PASS.

**Step 3: Commit**

```bash
git add crates/xlog-solve/tests/no_dtoh_in_gpu_cdcl.rs crates/xlog-prob/tests/no_dtoh_in_gpu_equivalence.rs
git commit -m "test: enforce zero host reads in GPU verifier paths"
```

---

### Task 6: Documentation Alignment (Absolute “Zero Host Reads” Contract)

**Files:**
- Modify: `docs/design/2026-01-22-gpu-native-compilation-design.md`
- Modify: `docs/architecture/solver-services.md`

**Step 1: Update docs to define the stronger contract**

Clarify that the verifier path must not perform device→host transfers, including scalar status reads.
Document the fail-fast behavior (GPU trap / CUDA error) and the expectation-based solver API.

**Step 2: Verify**

Run:
- `cargo test --workspace`

Expected: PASS.

**Step 3: Commit**

```bash
git add docs/design/2026-01-22-gpu-native-compilation-design.md docs/architecture/solver-services.md
git commit -m "docs: require zero host reads for GPU CDCL verifier path"
```

---

### Task 7: Repo-Wide Verification

**Step 1: Build**

Run:
- `cargo build --workspace`

**Step 2: Test**

Run:
- `cargo test --workspace`

Expected: PASS.
