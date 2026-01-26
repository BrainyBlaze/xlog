# GPU D4 Post-Processing (Task 6) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Implement GPU-native smoothing and free-variable correction with strict memory caps and zero device→host data-plane reads.

**Architecture:** Add GPU smoothing as a count+emit pass over device-resident XGCF using random-var bitsets and a prebuilt tautology table. Compute free-var masks on device and apply correction on GPU during evaluation (logZ + gradients) using deterministic reduction.

**Tech Stack:** Rust (xlog-prob/xlog-cuda), CUDA C++ kernels compiled to PTX via `nvcc` and loaded by `cudarc`.

---

### Task 1: Add post-processing caps + plumbing

**Files:**
- Modify: `crates/xlog-prob/src/compilation/gpu_d4.rs`
- Modify: `crates/xlog-prob/src/compilation/gpu_d4.rs` tests

**Step 1: Write failing test**

Add a compile-time test that constructs `GpuCompileConfig` with required smoothing caps and asserts the new fields exist.

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-prob --tests gpu_d4_compile_phase1_unit_clause_is_equivalent_via_gpu_cdcl -- --nocapture --test-threads=1`
Expected: FAIL because new config fields are missing.

**Step 3: Implement minimal code**

Add to `GpuCompileConfig`:
- `smooth_node_cap: u32`
- `smooth_edge_cap: u32`

Update all config construction sites in `crates/xlog-prob/src/compilation/gpu_d4.rs` tests.

**Step 4: Run test to verify it passes**

Run: same as Step 2. Expected: PASS.

**Step 5: Commit**

```bash
git add crates/xlog-prob/src/compilation/gpu_d4.rs
git commit -m "feat(d4): add smoothing caps to compile config"
```

---

### Task 2: GPU smoothing kernels (bitset support + count/emit)

**Files:**
- Modify: `kernels/d4.cu`
- Modify: `crates/xlog-cuda/src/provider.rs`

**Step 1: Write failing tests**

Add GPU tests in `crates/xlog-prob/tests/gpu_xgcf.rs`:
- `test_gpu_xgcf_smoothing_matches_cpu_gradients`
  - Build unsmoothed `Xgcf` (from a small Decision-DNNF with random vars).
  - CPU: `smooth_random_vars`, evaluate logZ+grads.
  - GPU: upload unsmoothed circuit, run GPU smoothing with random_var_list, eval logZ+grads.
  - Assert CPU smoothed == GPU smoothed (logZ + grads).

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-prob test_gpu_xgcf_smoothing_matches_cpu_gradients -- --nocapture --test-threads=1`
Expected: FAIL (GPU smoothing API missing).

**Step 3: Implement minimal code**

Add smoothing kernels to `kernels/d4.cu`:
- `d4_support_level` (compute random-var support bitsets per level).
- `d4_smooth_count` (per-node wrapper counts + wrapper edge counts + orig edge counts).
- `d4_smooth_emit` (emit smoothed circuit, tautology table, wrappers, updated child indices, node levels).
- `d4_smooth_finalize` (fill unused nodes and set child_offsets sentinel deterministically).

Update `crates/xlog-cuda/src/provider.rs` with new kernel symbols in `d4_kernels` and module load list.

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-prob test_gpu_xgcf_smoothing_matches_cpu_gradients -- --nocapture --test-threads=1`
Expected: PASS.

**Step 5: Commit**

```bash
git add kernels/d4.cu crates/xlog-cuda/src/provider.rs
git commit -m "feat(d4): add gpu smoothing count+emit kernels"
```

---

### Task 3: GPU free-var mask computation (device-side)

**Files:**
- Modify: `kernels/d4.cu`
- Modify: `crates/xlog-cuda/src/provider.rs`
- Modify: `crates/xlog-prob/src/compilation/gpu_d4.rs` (wrapper API)

**Step 1: Write failing tests**

Add a GPU test in `crates/xlog-prob/tests/gpu_xgcf.rs`:
- `test_gpu_free_var_correction_matches_cpu`
  - Circuit where var2 is absent from CNF/circuit.
  - Set free_var mask on GPU and run eval_log_wmc_and_grads.
  - Assert logZ and grads match CPU correction (manual or CPU eval + correction).

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-prob test_gpu_free_var_correction_matches_cpu -- --nocapture --test-threads=1`
Expected: FAIL (free-var mask + correction missing).

**Step 3: Implement minimal code**

Add kernels in `kernels/d4.cu`:
- `d4_mark_vars_in_clauses`
- `d4_mark_vars_in_circuit`
- `d4_build_free_var_mask` (trap if var in clauses but missing in circuit)

Expose symbols in `d4_kernels` and provider.

Add Rust helper in `crates/xlog-prob/src/compilation/gpu_d4.rs`:
- `compute_free_var_mask_gpu(cnf: &GpuCnf, circuit: &GpuXgcf, provider: &CudaKernelProvider) -> TrackedCudaSlice<u8>`

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-prob test_gpu_free_var_correction_matches_cpu -- --nocapture --test-threads=1`
Expected: PASS.

**Step 5: Commit**

```bash
git add kernels/d4.cu crates/xlog-cuda/src/provider.rs crates/xlog-prob/src/compilation/gpu_d4.rs crates/xlog-prob/tests/gpu_xgcf.rs
git commit -m "feat(d4): add gpu free-var mask computation"
```

---

### Task 4: GPU free-var correction during evaluation

**Files:**
- Modify: `kernels/circuit.cu`
- Modify: `crates/xlog-cuda/src/provider.rs`
- Modify: `crates/xlog-prob/src/gpu.rs`

**Step 1: Write failing test**

Reuse `test_gpu_free_var_correction_matches_cpu` from Task 3 (should still fail until correction applied).

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-prob test_gpu_free_var_correction_matches_cpu -- --nocapture --test-threads=1`
Expected: FAIL (GPU eval ignores free-var correction).

**Step 3: Implement minimal code**

Add kernels to `kernels/circuit.cu`:
- `xgcf_free_var_apply_grad` (per-var gradient correction on device)
- `xgcf_free_var_reduce_stage` (deterministic reduction for logZ correction)
- `xgcf_add_scalar` (add correction to root value)

Update provider `circuit_kernels` and module load list.

In `crates/xlog-prob/src/gpu.rs`:
- Add `free_var_mask: Option<TrackedCudaSlice<u8>>` to `GpuXgcf`.
- Add `set_free_var_mask_device(...)` and `set_free_var_mask_from_host(...)` helpers.
- Add private `apply_free_var_correction(...)` called from:
  - `eval_log_wmc`
  - `eval_log_wmc_and_grads`
  - `eval_grads_inplace`

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-prob test_gpu_free_var_correction_matches_cpu -- --nocapture --test-threads=1`
Expected: PASS.

**Step 5: Commit**

```bash
git add kernels/circuit.cu crates/xlog-cuda/src/provider.rs crates/xlog-prob/src/gpu.rs
git commit -m "feat(gpu): apply free-var correction on device"
```

---

### Task 5: End-to-end smoothing + correction integration

**Files:**
- Modify: `crates/xlog-prob/src/gpu.rs`
- Modify: `crates/xlog-prob/src/compilation/gpu_d4.rs`
- Modify: `crates/xlog-prob/tests/gpu_xgcf.rs`

**Step 1: Write failing test**

Reuse `test_gpu_xgcf_smoothing_matches_cpu_gradients` to ensure smoothing + gradients still match CPU after free-var corrections are wired.

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-prob test_gpu_xgcf_smoothing_matches_cpu_gradients -- --nocapture --test-threads=1`
Expected: FAIL if smoothing not wired into GPU API.

**Step 3: Implement minimal code**

- Add `GpuXgcf::smooth_random_vars_device(...) -> Result<GpuXgcf>` that uses the D4 smoothing kernels.
- Wire smoothing to use `GpuCompileConfig::smooth_node_cap/smooth_edge_cap`.

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-prob test_gpu_xgcf_smoothing_matches_cpu_gradients -- --nocapture --test-threads=1`
Expected: PASS.

**Step 5: Commit**

```bash
git add crates/xlog-prob/src/gpu.rs crates/xlog-prob/src/compilation/gpu_d4.rs crates/xlog-prob/tests/gpu_xgcf.rs
git commit -m "feat(gpu): add smoothing pass for random vars"
```

---

### Task 6: Verification

**Step 1: Run targeted tests**

```bash
cargo test -p xlog-prob test_gpu_xgcf_smoothing_matches_cpu_gradients -- --nocapture --test-threads=1
cargo test -p xlog-prob test_gpu_free_var_correction_matches_cpu -- --nocapture --test-threads=1
```

**Step 2: Run full crate tests**

```bash
cargo test -p xlog-prob --tests -- --nocapture --test-threads=1
```

**Step 3: Commit (if anything left)**

```bash
git status -sb
```
