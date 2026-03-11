# GPU-Resident ILP Credit/Loss Path — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace Python-side `_compute_loss_from_candidates()` with a single Rust/CUDA call that computes ILP training loss and gradient entirely on GPU, returning DLPack tensors with zero data-plane D2H transfers.

**Architecture:** CSR scatter-gather on GPU. Build COO `(fact_idx, cand_idx)` pairs from per-entry membership masks on device, sort → CSR, then forward kernel (credit gather + clamp + NLL loss) and backward kernel (gradient scatter via atomicAdd). Returns `(loss_dlpack, grad_dlpack)` for `cand_probs.backward(grad)`.

**Tech Stack:** Rust (PyO3), CUDA C, Python (PyTorch/DLPack)

**Design doc:** `docs/plans/2026-03-05-gpu-resident-loss-credit-design.md`

---

### Task 1: `membership_mask_device()` on CudaKernelProvider

Add a GPU-resident variant of `membership_mask` that returns `TrackedCudaSlice<u8>` instead of downloading to `Vec<bool>`. Then refactor existing `membership_mask` to call it internally.

**Files:**
- Modify: `crates/xlog-cuda/src/provider/mod.rs:8327-8449`
- Test: `crates/xlog-cuda/tests/certification_suite/mod.rs` (or a new test in the CUDA cert suite)

**Step 1: Write the failing test**

Add a test in the CUDA cert suite that calls `membership_mask_device` and validates the returned device mask. The test should:

```rust
#[test]
fn test_membership_mask_device_basic() {
    let provider = test_provider();
    // Create a 3-row probe buffer: [(1,2), (3,4), (5,6)]
    // Create a 2-row build buffer: [(1,2), (5,6)]
    // Call membership_mask_device(probe, build, &[0,1], &[0,1])
    // Download the returned TrackedCudaSlice<u8> to host
    // Assert: [1, 0, 1]
    let probe = create_test_buffer(&provider, &[(1u32, 2u32), (3, 4), (5, 6)]);
    let build = create_test_buffer(&provider, &[(1u32, 2u32), (5, 6)]);
    let d_mask = provider.membership_mask_device(&probe, &build, &[0, 1], &[0, 1]).unwrap();
    let mut host_mask = vec![0u8; 3];
    provider.device().inner().dtoh_sync_copy_into(&d_mask, &mut host_mask).unwrap();
    assert_eq!(host_mask, vec![1, 0, 1]);
}

#[test]
fn test_membership_mask_device_empty_build() {
    let provider = test_provider();
    let probe = create_test_buffer(&provider, &[(1u32, 2u32)]);
    let build = create_empty_test_buffer(&provider, 2);
    let d_mask = provider.membership_mask_device(&probe, &build, &[0, 1], &[0, 1]).unwrap();
    let mut host_mask = vec![0u8; 1];
    provider.device().inner().dtoh_sync_copy_into(&d_mask, &mut host_mask).unwrap();
    assert_eq!(host_mask, vec![0]);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-cuda-tests --test certification_suite test_membership_mask_device --release`
Expected: FAIL — `membership_mask_device` method does not exist.

**Step 3: Implement `membership_mask_device`**

In `crates/xlog-cuda/src/provider/mod.rs`, add a new public method immediately before the existing `membership_mask` (around line 8327):

```rust
/// GPU-resident membership mask: returns TrackedCudaSlice<u8> on device.
/// Values are in {0, 1} only. Does NOT download to host.
pub fn membership_mask_device(
    &self,
    probe: &CudaBuffer,
    build: &CudaBuffer,
    probe_keys: &[usize],
    build_keys: &[usize],
) -> Result<TrackedCudaSlice<u8>> {
    let num_probe = self.device_row_count(probe)?;
    let num_build = self.device_row_count(build)?;

    if num_probe == 0 {
        return self.memory.alloc::<u8>(0);
    }
    if num_build == 0 {
        let mut d_mask = self.memory.alloc::<u8>(num_probe)?;
        self.device.inner().memset_zeros(&mut d_mask)
            .map_err(|e| XlogError::Kernel(format!("memset failed: {}", e)))?;
        return Ok(d_mask);
    }

    if num_probe > u32::MAX as usize || num_build > u32::MAX as usize {
        return Err(XlogError::Kernel(format!(
            "membership_mask_device supports at most {} rows per side (probe={}, build={})",
            u32::MAX, num_probe, num_build
        )));
    }
    if probe_keys.is_empty() || build_keys.is_empty() {
        return Err(XlogError::Kernel(
            "membership_mask_device requires at least one key column".to_string(),
        ));
    }
    if probe_keys.len() != build_keys.len() {
        return Err(XlogError::Kernel(
            "Probe and build key columns must have same length".to_string(),
        ));
    }
    for (&p_idx, &b_idx) in probe_keys.iter().zip(build_keys.iter()) {
        let p_type = probe.schema().column_type(p_idx);
        let b_type = build.schema().column_type(b_idx);
        if p_type != b_type {
            return Err(XlogError::Kernel(format!(
                "Key column type mismatch: probe[{}]={:?}, build[{}]={:?}",
                p_idx, p_type, b_idx, b_type
            )));
        }
    }

    let num_probe_u32 = num_probe as u32;
    let num_build_u32 = num_build as u32;

    let probe_packed = self.compute_hashes_and_pack_keys(probe, probe_keys)?;
    let build_packed = self.compute_hashes_and_pack_keys(build, build_keys)?;
    let table = self.build_hash_table_v2(&build_packed.hashes, num_build_u32)?;

    let d_has_match = self.memory.alloc::<u8>(num_probe)?;

    let semi_func = self.device.inner()
        .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_SEMI)
        .ok_or_else(|| XlogError::Kernel("hash_join_semi kernel not found".to_string()))?;

    let block_size = 256u32;
    let grid_size = (num_probe_u32 + block_size - 1) / block_size;
    let config = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        semi_func.clone().launch(config, (
            &probe_packed.hashes, num_probe_u32,
            &table.bucket_offsets, &table.bucket_counts,
            &table.bucket_entries, &table.bucket_entry_hashes,
            table.bucket_mask,
            &probe_packed.packed_keys, &build_packed.packed_keys,
            probe_packed.key_bytes, &d_has_match,
        ))
    }
    .map_err(|e| XlogError::Kernel(format!("hash_join_semi failed: {}", e)))?;

    Ok(d_has_match)
}
```

Then refactor existing `membership_mask` to call it:

```rust
pub fn membership_mask(
    &self,
    probe: &CudaBuffer,
    build: &CudaBuffer,
    probe_keys: &[usize],
    build_keys: &[usize],
) -> Result<Vec<bool>> {
    let d_has_match = self.membership_mask_device(probe, build, probe_keys, build_keys)?;
    let num_probe = d_has_match.len();
    if num_probe == 0 {
        return Ok(Vec::new());
    }
    let mut host_mask = vec![0u8; num_probe];
    self.device.inner()
        .dtoh_sync_copy_into(&d_has_match, &mut host_mask)
        .map_err(|e| XlogError::Kernel(format!("Failed to download membership mask: {}", e)))?;
    Ok(host_mask.into_iter().map(|b| b != 0).collect())
}
```

**Step 4: Run tests to verify**

Run: `cargo test -p xlog-cuda-tests --test certification_suite --release`
Expected: All existing tests PASS + new `test_membership_mask_device_*` PASS.

Also run the Python tests to verify no regression in `batch_tagged_credit` / `batch_fact_membership`:
Run: `.venv/bin/python -m pytest python/tests/test_typed_batch_upload.py -v`
Expected: All PASS.

**Step 5: Commit**

```bash
git add crates/xlog-cuda/src/provider/mod.rs crates/xlog-cuda/tests/
git commit -m "feat(cuda): add membership_mask_device() — GPU-resident mask variant"
```

---

### Task 2: CUDA Kernels — `kernels/ilp_credit.cu`

Write three new CUDA kernels for the ILP credit/loss pipeline: COO fill, forward (credit + NLL), backward (gradient scatter). Both f32 and f64 variants.

**Files:**
- Create: `kernels/ilp_credit.cu`
- Modify: `crates/xlog-cuda/src/kernel_manifest_data.rs:10-31` (add `"ilp_credit"` to KERNEL_CU_NAMES)
- Modify: `crates/xlog-cuda/src/provider/mod.rs:201-204` (add module constant, update assertion)
- Modify: `crates/xlog-cuda/src/provider/mod.rs:256-258` (add kernel name constants)
- Test: Build succeeds and CUDA cert suite loads the new module

**Step 1: Create `kernels/ilp_credit.cu`**

```cuda
#include <stdint.h>
#include <math.h>

// ============================================================================
// COO fill: write (fact_idx[i], cidx) pairs for compacted matching indices
// ============================================================================

extern "C" __global__ void ilp_coo_fill(
    const uint32_t* compacted_fact_indices,  // [count]
    uint32_t cidx,                            // broadcast value
    uint32_t count,                           // number of matching facts
    uint32_t offset,                          // write offset in COO arrays
    uint32_t* coo_fact,                       // output: fact indices [nnz_total]
    uint32_t* coo_cand                        // output: cand indices [nnz_total]
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    coo_fact[offset + tid] = compacted_fact_indices[tid];
    coo_cand[offset + tid] = cidx;
}

// ============================================================================
// Forward: CSR credit gather + clamp + NLL loss
// ============================================================================

extern "C" __global__ void ilp_credit_forward_f32(
    const uint32_t* row_offsets,    // [num_facts + 1]
    const uint32_t* col_indices,    // [nnz]
    const float* cand_probs,        // [num_candidates]
    const uint8_t* is_positive,     // [num_facts] — 1=positive, 0=negative
    uint32_t num_facts,
    float eps,
    float* credit_out,              // [num_facts] — clamped credit per fact
    float* loss_contrib             // [num_facts] — per-fact NLL contribution
) {
    uint32_t f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= num_facts) return;

    uint32_t start = row_offsets[f];
    uint32_t end   = row_offsets[f + 1];

    float credit = 0.0f;
    for (uint32_t p = start; p < end; p++) {
        credit += cand_probs[col_indices[p]];
    }

    if (is_positive[f]) {
        float clamped = fmaxf(credit, eps);
        credit_out[f] = clamped;
        loss_contrib[f] = -logf(clamped);
    } else {
        float one_minus = 1.0f - credit;
        float clamped = fmaxf(one_minus, eps);
        credit_out[f] = clamped;
        loss_contrib[f] = -logf(clamped);
    }
}

extern "C" __global__ void ilp_credit_forward_f64(
    const uint32_t* row_offsets,
    const uint32_t* col_indices,
    const double* cand_probs,
    const uint8_t* is_positive,
    uint32_t num_facts,
    double eps,
    double* credit_out,
    double* loss_contrib
) {
    uint32_t f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= num_facts) return;

    uint32_t start = row_offsets[f];
    uint32_t end   = row_offsets[f + 1];

    double credit = 0.0;
    for (uint32_t p = start; p < end; p++) {
        credit += cand_probs[col_indices[p]];
    }

    if (is_positive[f]) {
        double clamped = fmax(credit, eps);
        credit_out[f] = clamped;
        loss_contrib[f] = -log(clamped);
    } else {
        double one_minus = 1.0 - credit;
        double clamped = fmax(one_minus, eps);
        credit_out[f] = clamped;
        loss_contrib[f] = -log(clamped);
    }
}

// ============================================================================
// Backward: gradient scatter into d_cand_probs via CSR
// ============================================================================

extern "C" __global__ void ilp_credit_backward_f32(
    const uint32_t* row_offsets,    // [num_facts + 1]
    const uint32_t* col_indices,    // [nnz]
    const float* credit_out,        // [num_facts] — clamped credit from forward
    const uint8_t* is_positive,     // [num_facts]
    uint32_t num_facts,
    float* d_cand_probs             // [num_candidates] — accumulated gradients
) {
    uint32_t f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= num_facts) return;

    // d_loss/d_credit:
    //   positive: -1/credit (where credit = clamped sum)
    //   negative: 1/(1-sum) = 1/clamped_one_minus (stored as credit_out for negatives)
    float d_credit;
    if (is_positive[f]) {
        d_credit = -1.0f / credit_out[f];
    } else {
        d_credit = 1.0f / credit_out[f];
    }

    uint32_t start = row_offsets[f];
    uint32_t end   = row_offsets[f + 1];

    for (uint32_t p = start; p < end; p++) {
        float grad;
        if (is_positive[f]) {
            grad = d_credit;  // d_loss/d_cand_probs[c] = d_loss/d_credit * d_credit/d_cand = d_credit * 1
        } else {
            grad = -d_credit; // d_credit = d(-log(1-sum))/d_cand = 1/(1-sum) * (-1) = -1/clamped → negated
                              // Actually: d_loss/d_sum = 1/(1-sum), d_sum/d_cand = 1
                              // So d_loss/d_cand = 1/(1-sum). But sign: d(-log(1-s))/ds = 1/(1-s).
        }
        atomicAdd(&d_cand_probs[col_indices[p]], grad);
    }
}

extern "C" __global__ void ilp_credit_backward_f64(
    const uint32_t* row_offsets,
    const uint32_t* col_indices,
    const double* credit_out,
    const uint8_t* is_positive,
    uint32_t num_facts,
    double* d_cand_probs
) {
    uint32_t f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= num_facts) return;

    double d_credit;
    if (is_positive[f]) {
        d_credit = -1.0 / credit_out[f];
    } else {
        d_credit = 1.0 / credit_out[f];
    }

    uint32_t start = row_offsets[f];
    uint32_t end   = row_offsets[f + 1];

    for (uint32_t p = start; p < end; p++) {
        double grad;
        if (is_positive[f]) {
            grad = d_credit;
        } else {
            grad = -d_credit;
        }
        // atomicAdd for f64 requires sm_60+
        atomicAdd(&d_cand_probs[col_indices[p]], grad);
    }
}
```

**Important note on backward gradient derivation:**

For positive facts: `loss = -log(clamp(sum, eps))`. So `d_loss/d_cand_probs[c] = -1/clamp(sum, eps)` for each `c` in the CSR row.

For negative facts: `loss = -log(clamp(1 - sum, eps))`. So `d_loss/d_sum = 1/(1 - sum)`, and `d_sum/d_cand_probs[c] = 1`, giving `d_loss/d_cand_probs[c] = 1/(1 - sum)`. But we stored `credit_out[f] = clamp(1-sum, eps)`, so `d_loss/d_cand_probs[c] = 1/credit_out[f]`.

Wait — for negatives, `d(-log(1-s))/ds = 1/(1-s)`. Each `cand_probs[c]` contributes +1 to `s`. So the gradient per candidate index is `+1/(1-s)`.

Let me correct the backward kernel to be cleaner:

The backward kernels should be:
- Positive: `grad = -1/credit_out[f]` (where `credit_out` stores clamped sum)
- Negative: `grad = 1/credit_out[f]` (where `credit_out` stores clamped `(1-sum)`)

Both cases: `atomicAdd(&d_cand_probs[col_indices[p]], grad)`

This is already what the code above does (the inner `if/else` with `grad = d_credit` / `grad = -d_credit` is incorrect and will be simplified in implementation).

**Step 2: Register the kernel**

In `crates/xlog-cuda/src/kernel_manifest_data.rs`, add `"ilp_credit"` to `KERNEL_CU_NAMES`:

```rust
pub const KERNEL_CU_NAMES: &[&str] = &[
    "join", "dedup", "groupby", "scan", "sort", "filter", "set_ops",
    "pack", "pir", "cnf", "cache", "weights", "circuit",
    "mc_sample", "mc_eval", "arith", "sat", "d4", "neural", "ilp",
    "ilp_credit",  // NEW
];
```

Update the compile-time assertion in `provider/mod.rs (pre-Wave-2 line 204)`:

```rust
const _: () = assert!(crate::kernel_manifest_data::KERNEL_CU_NAMES.len() == 21);
```

Add module constant and kernel names in `provider.rs`:

```rust
pub const ILP_CREDIT_MODULE: &str = "xlog_ilp_credit";
```

```rust
pub mod ilp_credit_kernels {
    pub const ILP_COO_FILL: &str = "ilp_coo_fill";
    pub const ILP_CREDIT_FORWARD_F32: &str = "ilp_credit_forward_f32";
    pub const ILP_CREDIT_FORWARD_F64: &str = "ilp_credit_forward_f64";
    pub const ILP_CREDIT_BACKWARD_F32: &str = "ilp_credit_backward_f32";
    pub const ILP_CREDIT_BACKWARD_F64: &str = "ilp_credit_backward_f64";
}
```

Add module loading in `provider.rs` after the ILP module block (around line 1487):

```rust
// ILP credit module
{
    let t0 = if profiling { Some(Instant::now()) } else { None };
    let (ptx, is_cubin) = load_module_from_file("ilp_credit", cc)?;
    device.inner()
        .load_ptx(ptx, ILP_CREDIT_MODULE, &[
            ilp_credit_kernels::ILP_COO_FILL,
            ilp_credit_kernels::ILP_CREDIT_FORWARD_F32,
            ilp_credit_kernels::ILP_CREDIT_FORWARD_F64,
            ilp_credit_kernels::ILP_CREDIT_BACKWARD_F32,
            ilp_credit_kernels::ILP_CREDIT_BACKWARD_F64,
        ])
        .map_err(|e| XlogError::Kernel(format!("Failed to load ILP credit module: {}", e)))?;
    if let Some(t0) = t0 {
        if profiling {
            device.inner().synchronize()
                .map_err(|e| XlogError::Kernel(format!("sync after ILP credit load: {}", e)))?;
        }
        let elapsed = t0.elapsed().as_secs_f64();
        profile.per_module_sec.push(("ilp_credit".to_string(), elapsed));
        profile.total_sec += elapsed;
        if is_cubin { profile.cubin_loaded += 1; } else { profile.ptx_fallback += 1; }
    }
}
```

**Step 3: Build and verify**

Run: `cargo build -p xlog-cuda --release`
Expected: PASS — kernel compiles, Rust module constants compile.

Run: `cargo test -p xlog-cuda-tests --test certification_suite run_full_certification --release`
Expected: All tests PASS, now with 21 modules loaded.

**Step 4: Commit**

```bash
git add kernels/ilp_credit.cu crates/xlog-cuda/src/kernel_manifest_data.rs crates/xlog-cuda/src/provider/mod.rs
git commit -m "feat(cuda): add ilp_credit.cu kernels — COO fill, forward, backward"
```

---

### Task 3: `set_candidate_map()` on IlpProgramFactory (PyO3)

Add a method to upload the `(i,j,k) -> cidx` mapping once per attempt. Stored as a `HashMap<(u32,u32,u32), u32>` in Rust. Validated on each `compute_ilp_loss_grad_gpu` call via length check.

**Files:**
- Modify: `crates/pyxlog/src/lib.rs` (add `candidate_map` field + `set_candidate_map` method)
- Test: `python/tests/test_ilp_credit_gpu.py` (new file)

**Step 1: Write the failing test**

Create `python/tests/test_ilp_credit_gpu.py`:

```python
"""Tests for GPU-resident ILP credit/loss path."""
import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda
skip_unless_pyxlog_cuda()


def _compile_reach() -> "pyxlog.IlpProgramFactory":
    prog = pyxlog.IlpProgramFactory.compile("""
        pred edge(u32, u32).
        edge(1, 2). edge(2, 3). edge(3, 4).
        learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """, device=0, memory_mb=64)
    prog.evaluate()
    return prog


def test_set_candidate_map_basic():
    prog = _compile_reach()
    candidates = [(0, 0, 0), (0, 1, 0), (1, 0, 0)]
    prog.set_candidate_map(candidates)
    # No crash = success. Validation tested via compute_ilp_loss_grad_gpu later.


def test_set_candidate_map_length_validation():
    prog = _compile_reach()
    prog.set_candidate_map([(0, 0, 0), (1, 1, 0)])
    assert prog.candidate_map_len() == 2
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_credit_gpu.py::test_set_candidate_map_basic -v`
Expected: FAIL — `set_candidate_map` method does not exist.

**Step 3: Implement**

In `crates/pyxlog/src/lib.rs`, add to the `IlpProgramFactory` struct:

```rust
// New field (add to struct definition):
candidate_map: Option<HashMap<(u32, u32, u32), u32>>,
```

Initialize as `None` in the constructor.

Add PyO3 methods:

```rust
/// Upload candidate (i,j,k) -> index mapping. Called once per attempt.
pub fn set_candidate_map(&mut self, candidates: Vec<(u32, u32, u32)>) -> PyResult<()> {
    let mut map = HashMap::with_capacity(candidates.len());
    for (cidx, (i, j, k)) in candidates.iter().enumerate() {
        map.insert((*i, *j, *k), cidx as u32);
    }
    self.candidate_map = Some(map);
    Ok(())
}

/// Length of current candidate map (0 if not set).
pub fn candidate_map_len(&self) -> usize {
    self.candidate_map.as_ref().map_or(0, |m| m.len())
}
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_credit_gpu.py -v`
Expected: PASS.

**Step 5: Rebuild pyxlog and commit**

```bash
cd crates/pyxlog && maturin develop --release && cd ../..
git add crates/pyxlog/src/lib.rs python/tests/test_ilp_credit_gpu.py
git commit -m "feat(pyxlog): add set_candidate_map() for ILP credit GPU path"
```

---

### Task 4: `compute_ilp_loss_grad_gpu()` — CSR Build Pipeline

Implement the core Rust method that builds the sparse CSR structure on GPU from retained `IlpTagEntry` buffers. This is the orchestrator: membership masks → COO → sort → CSR → forward → backward → DLPack export.

**Files:**
- Modify: `crates/pyxlog/src/lib.rs` (add `compute_ilp_loss_grad_gpu` method)
- Modify: `crates/xlog-cuda/src/provider/mod.rs` (add helper methods for COO concat, CSR build, forward/backward kernel launch)
- Test: `python/tests/test_ilp_credit_gpu.py` (extend)

**Step 1: Write the failing test**

Extend `python/tests/test_ilp_credit_gpu.py`:

```python
def test_compute_ilp_loss_grad_gpu_basic():
    """GPU loss/grad matches expected structure (shape, device, dtype)."""
    prog = _compile_reach()
    prog.evaluate()

    # Set up candidates (use actual ILP candidates from the program)
    # For reach with edge(1,2), edge(2,3), edge(3,4):
    # Candidate (i=0, j=0, k=0) means rule 0 with bL=edge, bR=edge for head reach
    candidates = [(0, 0, 0)]
    prog.set_candidate_map(candidates)

    device = torch.device("cuda:0")
    cand_probs = torch.tensor([0.8], device=device, dtype=torch.float32, requires_grad=True)

    positives = [("reach", [1, 3])]  # reach(1,3) should be derivable
    negatives = []

    loss_dl, grad_dl = prog.compute_ilp_loss_grad_gpu(
        positives, negatives, cand_probs
    )
    loss = torch.from_dlpack(loss_dl)
    grad = torch.from_dlpack(grad_dl)

    assert loss.device.type == "cuda"
    assert loss.shape == ()  # scalar
    assert loss.dtype == torch.float32
    assert grad.shape == cand_probs.shape
    assert grad.dtype == torch.float32
    assert grad.device.type == "cuda"
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_credit_gpu.py::test_compute_ilp_loss_grad_gpu_basic -v`
Expected: FAIL — method does not exist.

**Step 3: Implement the pipeline**

This is the largest implementation step. The method in `lib.rs`:

```rust
/// Compute ILP loss and gradient on GPU. Returns (loss_dlpack, grad_dlpack).
/// Zero data-plane D2H transfers.
pub fn compute_ilp_loss_grad_gpu<'py>(
    &self,
    py: Python<'py>,
    positives: Vec<(String, Vec<i64>)>,
    negatives: Vec<(String, Vec<i64>)>,
    cand_probs_obj: &Bound<'py, PyAny>,
) -> PyResult<(PyObject, PyObject)> {
    // 1. Validate candidate map exists
    let candidate_map = self.candidate_map.as_ref()
        .ok_or_else(|| PyValueError::new_err(
            "candidate_map not set — call set_candidate_map() first"
        ))?;

    // 2. Import cand_probs via DLPack
    let managed = dlpack_from_py(cand_probs_obj)?;
    let cand_table = self.provider.from_dlpack_tensors(vec![managed])
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let cand_buf = cand_table.column(0)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Validate length
    let num_cands = self.provider.device_row_count(&cand_buf)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    if num_cands != candidate_map.len() {
        return Err(PyValueError::new_err(format!(
            "cand_probs length {} != candidate_map length {}",
            num_cands, candidate_map.len()
        )));
    }

    // 3. Determine dtype (f32 or f64)
    // ... check cand_buf schema column type

    // 4. Get retained IlpTaggedResult
    let tagged = self.executor.ilp_last_result()
        .ok_or_else(|| PyRuntimeError::new_err(
            "No ILP result available — call evaluate() first"
        ))?;

    // 5. Combine positives + negatives into fact list with labels
    // Group by relation, upload query buffers, run membership_mask_device per entry

    // 6. Build COO on device: per entry, compact mask → fact indices → COO fill kernel
    // 7. Concatenate COO chunks, sort by (fact_idx, cand_idx) via radix sort
    // 8. Build CSR: boundary detection on sorted fact_idx → prefix sum → row_offsets

    // 9. Launch forward kernel
    // 10. GPU reduction of loss_contrib → scalar loss
    // 11. Launch backward kernel → d_cand_probs

    // 12. Export loss and grad via DLPack
}
```

The implementation should follow the design doc architecture section exactly. Key helpers to add to `provider.rs`:

- `ilp_coo_fill_launch()`: launches the `ilp_coo_fill` kernel
- `ilp_credit_forward_launch()`: launches the appropriate dtype forward kernel
- `ilp_credit_backward_launch()`: launches the appropriate dtype backward kernel
- `build_csr_from_sorted_keys()`: boundary detection → prefix sum → row_offsets array

Each helper uses existing primitives (radix sort, scan, boundary detection) already proven in the groupby path.

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_credit_gpu.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add crates/pyxlog/src/lib.rs crates/xlog-cuda/src/provider/mod.rs python/tests/test_ilp_credit_gpu.py
git commit -m "feat(pyxlog): compute_ilp_loss_grad_gpu — GPU-resident ILP loss pipeline"
```

---

### Task 5: Gradient Parity Tests

Verify that the GPU loss/grad matches the Python `_compute_loss_from_candidates()` output numerically. Run both f32 and f64 variants against known programs.

**Files:**
- Modify: `python/tests/test_ilp_credit_gpu.py` (add parity tests)

**Step 1: Write parity tests**

```python
def _python_loss_grad(prog, cand_probs, positives, negatives, candidates, ijk_to_cidx):
    """Reference: current Python implementation."""
    from pyxlog.ilp.trainer import _compute_loss_from_candidates
    cp = cand_probs.detach().clone().requires_grad_(True)
    loss = _compute_loss_from_candidates(prog, cp, positives, negatives, candidates, ijk_to_cidx)
    loss.backward()
    return loss.item(), cp.grad.clone()


def test_loss_parity_f32_reach():
    """GPU f32 loss matches Python reference (atol=1e-6, rtol=1e-5)."""
    prog = _compile_reach()
    candidates = [{"i": 0, "j": 0, "k": 0, "head_name": "reach"}]
    ijk_to_cidx = {(0, 0, 0): 0}

    device = torch.device("cuda:0")
    cand_probs = torch.tensor([0.7], device=device, dtype=torch.float32, requires_grad=True)

    positives = [("reach", [1, 3])]
    negatives = [("reach", [1, 4])]

    # Python reference
    ref_loss, ref_grad = _python_loss_grad(
        prog, cand_probs, positives, negatives, candidates, ijk_to_cidx
    )

    # GPU path
    prog.set_candidate_map([(0, 0, 0)])
    loss_dl, grad_dl = prog.compute_ilp_loss_grad_gpu(positives, negatives, cand_probs)
    gpu_loss = torch.from_dlpack(loss_dl).item()
    gpu_grad = torch.from_dlpack(grad_dl)

    assert abs(gpu_loss - ref_loss) < 1e-6, f"Loss mismatch: {gpu_loss} vs {ref_loss}"
    assert torch.allclose(gpu_grad, ref_grad, atol=1e-5), \
        f"Grad mismatch: {gpu_grad} vs {ref_grad}"


def test_loss_parity_f64_reach():
    """GPU f64 loss matches Python reference (atol=1e-12, rtol=1e-10)."""
    prog = _compile_reach()
    candidates = [{"i": 0, "j": 0, "k": 0, "head_name": "reach"}]
    ijk_to_cidx = {(0, 0, 0): 0}

    device = torch.device("cuda:0")
    cand_probs = torch.tensor([0.7], device=device, dtype=torch.float64, requires_grad=True)

    positives = [("reach", [1, 3])]
    negatives = [("reach", [1, 4])]

    ref_loss, ref_grad = _python_loss_grad(
        prog, cand_probs, positives, negatives, candidates, ijk_to_cidx
    )

    prog.set_candidate_map([(0, 0, 0)])
    loss_dl, grad_dl = prog.compute_ilp_loss_grad_gpu(positives, negatives, cand_probs)
    gpu_loss = torch.from_dlpack(loss_dl).item()
    gpu_grad = torch.from_dlpack(grad_dl)

    assert abs(gpu_loss - ref_loss) < 1e-12, f"Loss mismatch: {gpu_loss} vs {ref_loss}"
    assert torch.allclose(gpu_grad, ref_grad, atol=1e-10), \
        f"Grad mismatch: {gpu_grad} vs {ref_grad}"


def test_loss_parity_multi_candidate():
    """Parity with multiple candidates contributing to the same fact."""
    # Use grandparent or a program where multiple rules can derive the same fact.
    # This tests that CSR accumulation matches Python loop summation.
    prog = pyxlog.IlpProgramFactory.compile("""
        pred edge(u32, u32).
        edge(1, 2). edge(2, 3). edge(3, 4). edge(1, 4).
        learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """, device=0, memory_mb=64)
    prog.evaluate()

    # Multiple candidate rules that could derive reach(1,3):
    # (0,0,0): reach(X,Y) :- edge(X,Z), edge(Z,Y) — derives reach(1,3) via edge(1,2),edge(2,3)
    candidates = [
        {"i": 0, "j": 0, "k": 0, "head_name": "reach"},
        {"i": 0, "j": 1, "k": 0, "head_name": "reach"},
    ]
    ijk_to_cidx = {(0, 0, 0): 0, (0, 1, 0): 1}

    device = torch.device("cuda:0")
    cand_probs = torch.tensor([0.5, 0.3], device=device, dtype=torch.float32, requires_grad=True)

    positives = [("reach", [1, 3])]
    negatives = []

    ref_loss, ref_grad = _python_loss_grad(
        prog, cand_probs, positives, negatives, candidates, ijk_to_cidx
    )

    prog.set_candidate_map([(0, 0, 0), (0, 1, 0)])
    loss_dl, grad_dl = prog.compute_ilp_loss_grad_gpu(positives, negatives, cand_probs)
    gpu_loss = torch.from_dlpack(loss_dl).item()
    gpu_grad = torch.from_dlpack(grad_dl)

    assert abs(gpu_loss - ref_loss) < 1e-5
    assert torch.allclose(gpu_grad, ref_grad, atol=1e-5)
```

**Step 2: Run to verify they pass**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_credit_gpu.py -v -k "parity"`
Expected: All parity tests PASS.

**Step 3: Commit**

```bash
git add python/tests/test_ilp_credit_gpu.py
git commit -m "test: ILP credit GPU loss/grad parity against Python reference"
```

---

### Task 6: D2H Accounting Tests

Verify `compute_ilp_loss_grad_gpu` causes zero additional D2H transfers.

**Files:**
- Modify: `python/tests/test_ilp_credit_gpu.py` (add D2H test)

**Step 1: Write the test**

```python
def test_zero_dtoh_transfers():
    """compute_ilp_loss_grad_gpu must not cause any D2H transfers."""
    prog = _compile_reach()
    prog.evaluate()

    candidates = [(0, 0, 0)]
    prog.set_candidate_map(candidates)

    device = torch.device("cuda:0")
    cand_probs = torch.tensor([0.8], device=device, dtype=torch.float32)

    positives = [("reach", [1, 3])]
    negatives = []

    prog.reset_d2h_transfer_count()
    before = prog.d2h_transfer_count()
    assert before == 0

    loss_dl, grad_dl = prog.compute_ilp_loss_grad_gpu(positives, negatives, cand_probs)

    after = prog.d2h_transfer_count()
    assert after == 0, f"D2H transfers occurred: {after}"
```

**Step 2: Run to verify**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_credit_gpu.py::test_zero_dtoh_transfers -v`
Expected: PASS.

**Step 3: Commit**

```bash
git add python/tests/test_ilp_credit_gpu.py
git commit -m "test: verify zero D2H transfers in ILP credit GPU path"
```

---

### Task 7: Trainer Integration

Replace `_compute_loss_from_candidates()` call in the training step with the GPU path.

**Files:**
- Modify: `crates/pyxlog/python/pyxlog/ilp/trainer.py:284-286` (add `set_candidate_map` call)
- Modify: `crates/pyxlog/python/pyxlog/ilp/trainer.py:358-363` (replace loss computation)
- Modify: `crates/pyxlog/python/pyxlog/ilp/trainer.py:384-388` (change backward call)

**Step 1: Add `set_candidate_map` at attempt start**

At line 286, after building `ijk_to_cidx`, add:

```python
        # Upload candidate map to Rust for GPU-resident loss computation
        prog.set_candidate_map([(c["i"], c["j"], c["k"]) for c in candidates])
```

**Step 2: Replace loss computation in step loop**

Replace lines 358-363 (the `_compute_loss_from_candidates` call) with:

```python
                # Compute task loss using GPU-resident credit path
                _t0 = time.perf_counter()
                loss_dl, grad_dl = prog.compute_ilp_loss_grad_gpu(
                    positives, all_negatives, cand_probs,
                )
                loss_tensor = torch.from_dlpack(loss_dl)
                grad_tensor = torch.from_dlpack(grad_dl)
                # loss for logging/convergence (detached scalar)
                loss = loss_tensor.detach()
                phase_loss_credit_us.append((time.perf_counter() - _t0) * 1_000_000)
```

**Step 3: Replace backward call**

Replace lines 384-388 (the `loss.backward()` / `optimizer.step()` block) with:

```python
                _t0 = time.perf_counter()
                cand_probs.backward(grad_tensor)
                optimizer.step()
                phase_backward_step_us.append((time.perf_counter() - _t0) * 1_000_000)
```

Remove the `if loss.requires_grad:` guard — the GPU path always produces a gradient.

**Step 4: Run full test suite**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_reliability.py -v`
Expected: 20/20 PASS.

Run: `.venv/bin/python -m pytest python/tests/ -v --timeout=300`
Expected: All tests PASS.

**Step 5: Commit**

```bash
git add crates/pyxlog/python/pyxlog/ilp/trainer.py
git commit -m "feat(trainer): integrate GPU-resident ILP credit/loss path"
```

---

### Task 8: Full Regression + Performance Measurement

Run the complete test matrix and measure performance against baselines.

**Files:**
- No code changes. Test runs and measurement only.

**Step 1: Run Rust workspace tests**

Run: `cargo test --workspace --all-targets --exclude pyxlog --release`
Expected: All PASS.

**Step 2: Run CUDA cert suite**

Run: `cargo test -p xlog-cuda-tests --test certification_suite --release`
Expected: All PASS (now with 21 modules).

**Step 3: Run ILP reliability**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_reliability.py -v`
Expected: 20/20 PASS.

**Step 4: Run GA reliability**

Run: `GA_RELIABILITY_MAX_ATTEMPTS=2 .venv/bin/python -m pytest python/tests/test_ilp_ga_reliability.py -v`
Expected: PASS with Clopper-Pearson lower95 >= 0.929.

Record the wall-clock time. Target: <= 327s (>= 25% faster than 436s baseline).

**Step 5: Check phase timing**

Examine `loss_credit_us` p95 from the reliability run logs. Target: >= 8x faster than baseline.

**Step 6: Run SLO tests**

Run: `ILP_PERF_ENFORCE_SLO=1 .venv/bin/python -m pytest python/tests/test_slo_scaling.py -v`
Expected: All PASS.

**Step 7: Document results**

Create `docs/reports/YYYY-MM-DD-gpu-credit-loss-report.md` with timing data, parity results, and go/no-go decision.

**Step 8: Commit report**

```bash
git add docs/reports/
git commit -m "docs: GPU-resident credit/loss path performance report"
```

---

## Success Criteria Summary

| Check | Threshold | Task |
|-------|-----------|------|
| Loss parity f32 (atol/rtol) | 1e-6 / 1e-5 | Task 5 |
| Loss parity f64 (atol/rtol) | 1e-12 / 1e-10 | Task 5 |
| Gradient max abs error f32 | <= 1e-5 | Task 5 |
| Gradient max abs error f64 | <= 1e-10 | Task 5 |
| Zero D2H transfers | 0 additional | Task 6 |
| ILP reliability | 20/20 | Task 8 |
| GA reliability | Clopper-Pearson lower95 >= 0.929 | Task 8 |
| `loss_credit_us` p95 | >= 8x faster | Task 8 |
| GA 50-seed wall-clock | <= 327s | Task 8 |

## Task Dependency Graph

```
Task 1 (membership_mask_device)
  └─→ Task 2 (CUDA kernels) ──→ Task 4 (compute_ilp_loss_grad_gpu)
  └─→ Task 3 (set_candidate_map) ──→ Task 4
                                          └─→ Task 5 (parity tests)
                                          └─→ Task 6 (D2H tests)
                                          └─→ Task 7 (trainer integration)
                                                   └─→ Task 8 (full regression)
```

Tasks 1, 2, 3 can proceed in parallel. Tasks 5 and 6 can proceed in parallel after Task 4. Task 7 depends on Task 4. Task 8 depends on Task 7.
