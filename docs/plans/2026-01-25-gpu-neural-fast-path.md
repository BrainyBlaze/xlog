# GPU Neural Fast-Path (100% GPU-Native AD Weights + Gradients) Implementation Plan

> **For Codex:** REQUIRED SUB-SKILLS: `superpowers:executing-plans` + `superpowers:test-driven-development`

**Goal:** Make `pyxlog` training’s “GPU neural fast-path” fully GPU-native (no CPU data-plane transfers) by computing
annotated-disjunction (AD) conditional-chain weights and probability gradients on GPU from device-resident tensors.

**Architecture:** Keep the compiled XGCF circuit cached, then per step: (1) read neural softmax outputs via DLPack
(device pointers), (2) fill XGCF `var_log_true/var_log_false` on GPU using the exact AD conditional-chain mapping,
(3) run XGCF forward+backward on GPU, (4) scatter chain-rule gradients back into per-output tensors on GPU and call
`output.backward(grad)` (no `.tolist()`, no host weight/grad uploads, no device->host reads required).

**Tech Stack:** Rust, CUDA (`nvcc` PTX via `crates/xlog-cuda/build.rs`), `cudarc`, DLPack (`xlog-cuda::dlpack`),
PyO3 (`crates/pyxlog`), Torch (Python side).

---

### Task 1: Add Neural PTX Module + Provider Wiring (RED → GREEN)

**Files:**
- Create: `kernels/neural.cu`
- Modify: `crates/xlog-cuda/build.rs`
- Modify: `crates/xlog-cuda/src/provider.rs`
- Test: `crates/xlog-cuda/tests/ptx_validation.rs` (or add a new focused test)

**Step 1: Write failing provider test (RED)**

Add assertions that `NEURAL_PTX` is embedded and contains the new entrypoints:
- `neural_fill_ad_chain_f32`
- `neural_scatter_ad_chain_grads_f32`

Run: `cargo test -p xlog-cuda ptx_validation -- --nocapture`
Expected: FAIL because `NEURAL_PTX`/module/kernels don’t exist yet.

**Step 2: Implement `kernels/neural.cu` (GREEN)**

Implement production-grade CUDA kernels:

1) **Weight fill (correct AD chain):**
`neural_fill_ad_chain_f32(const float* p, uint32_t n, const uint32_t* slot_var, double eps, double min_p, double* var_log_true, double* var_log_false)`

- Compute `p_out[i] = (1-eps) * max(p[i], min_p)`
- Compute conditional chain:
  - `remaining = 1 - sum_{k<i} p_out[k]`
  - `q[i] = p_out[i] / remaining` (clamp to `(min_p, 1-min_p)` for stability)
  - Write `var_log_true[var_i] = ln(q[i])`, `var_log_false[var_i] = ln(1-q[i])`

2) **Gradient scatter (correct chain rule; uses both grads):**
`neural_scatter_ad_chain_grads_f32(const float* p, uint32_t n, const uint32_t* slot_var, double eps, double min_p, const double* grad_true, const double* grad_false, uint8_t mode, float* out)`

- Recompute `p_out`, `remaining`, `q`.
- For each i: `dlogZ/dq[i] = gT*(1/q) + gF*(-1/(1-q))`
- Backward sweep:
  - `S = 0`
  - for i from n-1..0:
    - `dlogZ/dp_out[i] = dlogZ/dq[i] * (1/remaining[i]) + S`
    - `S += dlogZ/dq[i] * (p_out[i]/(remaining[i]^2))`
- Convert to `dlogZ/dp = (1-eps) * dlogZ/dp_out[i]`
- `mode=0`: `out[i] = (float)dlogZ/dp`
- `mode=1`: `out[i] -= (float)dlogZ/dp` (used for NLL: base minus query)

**Step 3: Wire kernel compilation + provider module (GREEN)**

- Add `"neural"` to `crates/xlog-cuda/build.rs` kernel list so PTX is generated.
- In `crates/xlog-cuda/src/provider.rs`:
  - `const NEURAL_PTX: &str = include_str!(\"../../../kernels/neural.ptx\");`
  - `pub const NEURAL_MODULE: &str = \"xlog_neural\";`
  - `pub mod neural_kernels { ... }` with the function names
  - Load the PTX in `CudaKernelProvider::new(...)` alongside other modules

**Step 4: Re-run provider test**

Run: `cargo test -p xlog-cuda ptx_validation -- --nocapture`
Expected: PASS.

**Step 5: Commit**

```bash
git add kernels/neural.cu crates/xlog-cuda/build.rs crates/xlog-cuda/src/provider.rs crates/xlog-cuda/tests/ptx_validation.rs
git commit -m "cuda(neural): add PTX module for AD weight fill and gradient scatter"
```

---

### Task 2: GPU-Only XGCF “Eval+Grads In-Place” API (No Host Reads) (RED → GREEN)

**Files:**
- Modify: `crates/xlog-prob/src/gpu.rs`
- Test: `crates/xlog-prob/tests/gpu_xgcf.rs` (add a new focused test)

**Step 1: Write failing test (RED)**

Add a test that:
- uploads a tiny XGCF circuit,
- sets weights on-device (no host upload in eval),
- runs forward+backward,
- asserts it does not call `dtoh_sync_copy_into` anywhere in the new API (use a source-scan test like existing no-dtoh tests).

Run: `cargo test -p xlog-prob gpu_xgcf -- --nocapture`
Expected: FAIL (new API doesn’t exist).

**Step 2: Implement GPU-only eval+grads method (GREEN)**

In `GpuXgcf` add:
- `pub fn set_base_weights(&mut self, provider: &CudaKernelProvider, weights: &[(f64,f64)]) -> Result<()>`
  - one-time host->device upload of `var_log_true/var_log_false` (template init)
- `pub fn eval_grads_inplace(&mut self, provider: &CudaKernelProvider) -> Result<()>`
  - forward levels (no root download)
  - zero adj/grad buffers
  - set `adj[root]=1.0` using `arith_fill_const_f64` on a 1-element device view (no htod copy)
  - backward kernels per level (populate `grad_true/grad_false` on device)
- `pub fn grad_true(&self) -> &TrackedCudaSlice<f64>` + `pub fn grad_false(&self) -> &TrackedCudaSlice<f64>`
- `pub fn var_log_true_mut(&mut self) -> &mut TrackedCudaSlice<f64>` + `pub fn var_log_false_mut(&mut self) -> &mut TrackedCudaSlice<f64>`

**Step 3: Re-run test**

Run: `cargo test -p xlog-prob gpu_xgcf -- --nocapture`
Expected: PASS.

**Step 4: Commit**

```bash
git add crates/xlog-prob/src/gpu.rs crates/xlog-prob/tests/gpu_xgcf.rs
git commit -m "prob(gpu): add XGCF eval+grads in-place API (no host reads)"
```

---

### Task 3: Implement `GpuWeightSlots` (Device Slot Map) (RED → GREEN)

**Files:**
- Create: `crates/xlog-prob/src/neural_fast_path.rs`
- Modify: `crates/xlog-prob/src/lib.rs`
- Test: `crates/xlog-prob/tests/neural_fast_path.rs`

**Step 1: Write failing test (RED)**

Test that `GpuWeightSlots::upload(...)` produces correct device buffers (roundtrip download allowed in test).

Run: `cargo test -p xlog-prob neural_fast_path -- --nocapture`
Expected: FAIL.

**Step 2: Implement `GpuWeightSlots` (GREEN)**

Implement:
- `pub struct GpuWeightSlots { group_offsets: TrackedCudaSlice<u32>, slot_cnf_var: TrackedCudaSlice<u32>, num_groups: u32, total_slots: u32 }`
- `pub fn upload(provider: &CudaKernelProvider, groups: &[Vec<u32>]) -> Result<Self>`
  - Flatten vars, build offsets, upload to device
- Accessors for per-group `CudaView` slices

**Step 3: Re-run test**

Run: `cargo test -p xlog-prob neural_fast_path -- --nocapture`
Expected: PASS.

**Step 4: Commit**

```bash
git add crates/xlog-prob/src/neural_fast_path.rs crates/xlog-prob/src/lib.rs crates/xlog-prob/tests/neural_fast_path.rs
git commit -m "prob(neural): add device-resident weight slot map for neural fast-path"
```

---

### Task 4: GPU Neural Fast-Path: Fill Weights + Base/Query Scatter (RED → GREEN)

**Files:**
- Modify: `crates/xlog-prob/src/exact.rs`
- Modify: `crates/xlog-prob/src/neural_fast_path.rs`
- Test: `crates/xlog-prob/tests/neural_fast_path.rs`

**Step 1: Write failing end-to-end test (RED)**

Create a tiny program with one AD (e.g., 3 labels + implicit none) and one query.

Test should:
- Compile program
- Build `GpuWeightSlots`
- Create device probability tensor via DLPack roundtrip (use `xlog-cuda` provider import helpers)
- Run `ExactDdnnfProgram::neural_backward_nll(...)`
- Download the produced `out_grad` tensor and compare to CPU reference chain-rule computation.

Run: `cargo test -p xlog-prob neural_fast_path -- --nocapture`
Expected: FAIL.

**Step 2: Implement the production fast-path (GREEN)**

In `ExactDdnnfProgram`, add:
- `pub fn query_var(&self, idx: usize) -> Option<u32>`
- `pub fn ensure_gpu_state(&self) -> Result<()>` (or expose provider via a new method)

In `GpuExactState::new(...)`:
- Initialize GPU circuit once, and upload `evidence_log_weights` into `GpuXgcf` base weight tables.

In `neural_fast_path.rs`, implement:
- `pub struct NeuralFastPathConfig { eps: f64, min_p: f64 }`
- `pub fn backward_nll(&self, program: &ExactDdnnfProgram, slots: &GpuWeightSlots, query_idx: usize, probs: &[DlpackManagedTensor], out_grads: &[DlpackManagedTensor], cfg: NeuralFastPathConfig) -> Result<()>`
  - Validate arity: `probs.len()==num_groups==out_grads.len()`
  - For each group:
    - import `probs[g]` as contiguous 1D F32 (len = labels)
    - launch `neural_fill_ad_chain_f32` into `GpuXgcf.var_log_true/false` for that group’s vars
  - Base run:
    - `GpuXgcf.eval_grads_inplace()`
    - For each group: launch `neural_scatter_ad_chain_grads_f32(mode=0, out=grad_out[g])`
  - Query run:
    - force query var true by setting `var_log_false[query_var] = -INF` using `arith_fill_const_f64` on a 1-element view
    - `GpuXgcf.eval_grads_inplace()`
    - For each group: launch `neural_scatter_ad_chain_grads_f32(mode=1, out=grad_out[g])`  // out -= dlogZ_query/dp
    - restore query var false log-weight to original from `evidence_log_weights[query_var].1` (also via `arith_fill_const_f64` on 1-element view)
  - After this, `grad_out[g]` contains `dL/dp = dlogZ_base/dp - dlogZ_query/dp` (NLL).

**Step 3: Re-run test**

Run: `cargo test -p xlog-prob neural_fast_path -- --nocapture`
Expected: PASS.

**Step 4: Commit**

```bash
git add crates/xlog-prob/src/exact.rs crates/xlog-prob/src/neural_fast_path.rs crates/xlog-prob/tests/neural_fast_path.rs
git commit -m "prob(neural): GPU-native AD weight fill + chain-rule gradient scatter (NLL)"
```

---

### Task 5: Upgrade `pyxlog` Training Path to GPU-Native (No `.tolist()`, No Host Weight/Grad) (RED → GREEN)

**Files:**
- Modify: `crates/pyxlog/src/lib.rs`
- Update tests: `python/tests/test_circuit_cache.py`, `python/tests/test_minimal_example.py` (CUDA-only, skip if unavailable)

**Step 1: Write failing python test (RED)**

Update `python/tests/test_circuit_cache.py` to:
- move tensors + model to CUDA if available
- assert `forward_backward(...)` still works
- ensure no `.tolist()` is used internally by switching to GPU tensors (old code would force CPU sync/copy)

Run (manual): `pytest python/tests/test_circuit_cache.py -v`
Expected: FAIL until `pyxlog` is updated.

**Step 2: Implement GPU-native `forward_backward_complex` (GREEN)**

In `crates/pyxlog/src/lib.rs`:
- Replace probability extraction `.tolist()` with DLPack import of `output_squeezed` (device tensor)
- On cache miss: compile template circuit using **dummy uniform probabilities** (no host readback of live network outputs)
- Build `GpuWeightSlots` and cache it alongside `ExactDdnnfProgram`
- For each group:
  - allocate `grad_tensor = torch.empty_like(output_squeezed)` (device)
  - pass `output_squeezed.__dlpack__()` and `grad_tensor.__dlpack__()` into `xlog_prob::neural_fast_path::backward_nll`
  - call `output_squeezed.backward(grad_tensor)` (no host copies)

**Step 3: Run python tests**

Run: `pytest python/tests/test_circuit_cache.py -v`
Expected: PASS (skips if CUDA/torch unavailable).

**Step 4: Commit**

```bash
git add crates/pyxlog/src/lib.rs python/tests/test_circuit_cache.py python/tests/test_minimal_example.py
git commit -m "pyxlog(neural): GPU-native forward_backward via DLPack + AD chain kernels"
```

---

### Task 6: Guardrails: Enforce “No Host Reads” in Neural Fast-Path (RED → GREEN)

**Files:**
- Add test: `crates/xlog-prob/tests/no_dtoh_in_gpu_neural_fast_path.rs`

**Step 1: Write failing test (RED)**

Add a source-scan test similar to existing:
- Fail if `crates/xlog-prob/src/neural_fast_path.rs` contains `dtoh` or `tolist` strings.

Run: `cargo test -p xlog-prob no_dtoh_in_gpu_neural_fast_path -- --nocapture`
Expected: FAIL until code is clean.

**Step 2: Fix violations (GREEN)**

Remove any remaining host reads/copies from the fast-path implementation.

**Step 3: Commit**

```bash
git add crates/xlog-prob/tests/no_dtoh_in_gpu_neural_fast_path.rs
git commit -m "test: enforce no host reads in GPU neural fast-path"
```

---

### Task 7: Documentation Updates (Align With GPU-Native Design)

**Files:**
- Modify: `docs/design/2026-01-22-gpu-native-compilation-design.md` (mark 5.3 as implemented; document APIs)
- Modify: `docs/plans/2026-01-21-circuit-caching-design.md` (supersede CPU `.tolist()` path)
- Modify: `docs/ARCHITECTURE.md` / `docs/architecture/xlog-prob.md` if they list kernel modules

**Step 1: Update design doc and architecture docs**

Document:
- new `kernels/neural.cu` module and entrypoints
- how DLPack tensors are used for p and gradients
- exact AD chain semantics and gradient mapping (both grads used)
- zero CPU transfers guarantee for the fast-path

**Step 2: Commit**

```bash
git add docs/design/2026-01-22-gpu-native-compilation-design.md docs/plans/2026-01-21-circuit-caching-design.md docs/ARCHITECTURE.md docs/architecture/xlog-prob.md
git commit -m "docs: align neural fast-path docs with GPU-native AD kernels"
```

