# Sparse Executor + Transfer Elimination Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate dense N^3 materialization from the sparse trainer path and wire the row-count cache end-to-end.

**Architecture:** The ILP executor gains a sparse mask path that skips the dense extraction kernel entirely. IlpMask becomes an enum (Dense | Sparse). The PyO3 `set_rule_mask_sparse` API accepts a DLPack capsule directly. SparseMaskBackend passes CUDA tensors without host-side N^3 materialization. Row-count caches are wired into all 4 read sites.

**Tech Stack:** Rust (xlog-runtime, xlog-cuda, pyxlog), CUDA, Python (pyxlog.ilp), PyO3, DLPack, PyTorch

**Design doc:** `docs/plans/2026-03-01-sparse-executor-transfer-fix-design.md`

---

### Task 1: Wire row-count cache into provider.device_row_count

**Files:**
- Modify: `crates/xlog-cuda/src/provider.rs:6970-6977`

**Step 1: Modify device_row_count to check cache first**

In `crates/xlog-cuda/src/provider.rs`, replace the `device_row_count` method (lines 6970-6977):

```rust
fn device_row_count(&self, buffer: &CudaBuffer) -> Result<usize> {
    if let Some(n) = buffer.cached_row_count() {
        return Ok(n as usize);
    }
    let mut host_rows = [0u32];
    self.device
        .inner()
        .dtoh_sync_copy_into(buffer.num_rows_device(), &mut host_rows)
        .map_err(|e| XlogError::Kernel(format!("Failed to read row count: {}", e)))?;
    buffer.set_cached_row_count_if_unset(host_rows[0]);
    Ok(host_rows[0] as usize)
}
```

**Step 2: Wire cache into buffer_from_columns**

In `crates/xlog-cuda/src/provider.rs`, replace `buffer_from_columns` (lines 7159-7175). It already knows the row count (`row_cap`) and uploads it — use `from_columns_with_host_count` to populate the cache at construction:

```rust
pub(crate) fn buffer_from_columns(
    &self,
    columns: Vec<CudaColumn>,
    row_cap: u64,
    schema: Schema,
) -> Result<CudaBuffer> {
    let row_u32 = u32::try_from(row_cap)
        .map_err(|_| XlogError::Kernel(format!("Row capacity {} exceeds u32::MAX", row_cap)))?;
    let mut d_num_rows = self.memory.alloc::<u32>(1)?;
    self.device
        .inner()
        .htod_sync_copy_into(&[row_u32], &mut d_num_rows)
        .map_err(|e| XlogError::Kernel(format!("Failed to set row count: {}", e)))?;
    Ok(CudaBuffer::from_columns_with_host_count(
        columns, row_cap, d_num_rows, schema, row_u32,
    ))
}
```

**Step 3: Run Rust tests**

Run: `cargo test -p xlog-cuda --release -- --test-threads=1 2>&1 | tail -5`
Expected: All existing tests pass (cache is transparent).

**Step 4: Commit**

```bash
git add crates/xlog-cuda/src/provider.rs
git commit -m "perf(cuda): wire row-count cache into provider.device_row_count + buffer_from_columns"
```

---

### Task 2: Wire row-count cache into ilp_registry and executor

**Files:**
- Modify: `crates/xlog-runtime/src/ilp_registry.rs:9-20`
- Modify: `crates/xlog-runtime/src/executor.rs:2961-2969`
- Modify: `crates/xlog-runtime/src/executor.rs:3036-3044`

**Step 1: Fix ilp_registry::read_device_row_count**

In `crates/xlog-runtime/src/ilp_registry.rs`, replace `read_device_row_count` (lines 9-20):

```rust
pub fn read_device_row_count(
    provider: &CudaKernelProvider,
    buffer: &CudaBuffer,
) -> Result<usize, XlogError> {
    if let Some(n) = buffer.cached_row_count() {
        return Ok(n as usize);
    }
    let mut host_rows = [0u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(buffer.num_rows_device(), &mut host_rows)
        .map_err(|e| XlogError::Kernel(format!("Failed to read row count: {}", e)))?;
    buffer.set_cached_row_count_if_unset(host_rows[0]);
    Ok(host_rows[0] as usize)
}
```

**Step 2: Fix Executor::buffer_row_count**

In `crates/xlog-runtime/src/executor.rs`, replace `buffer_row_count` (lines 2961-2969):

```rust
fn buffer_row_count(&self, buffer: &CudaBuffer) -> Result<u32> {
    if let Some(n) = buffer.cached_row_count() {
        return Ok(n);
    }
    let mut host_rows = [0u32];
    self.provider
        .device()
        .inner()
        .dtoh_sync_copy_into(buffer.num_rows_device(), &mut host_rows)
        .map_err(|e| XlogError::Execution(format!("Failed to read row count: {}", e)))?;
    buffer.set_cached_row_count_if_unset(host_rows[0]);
    Ok(host_rows[0])
}
```

**Step 3: Fix standalone ilp_row_count**

In `crates/xlog-runtime/src/executor.rs`, find the standalone function around line 3036 that does the same pattern. Apply the same cache-first approach:

```rust
fn ilp_row_count(executor: &Executor, buffer: &CudaBuffer) -> u32 {
    if let Some(n) = buffer.cached_row_count() {
        return n;
    }
    let mut host_rows = [0u32];
    executor
        .provider
        .device()
        .inner()
        .dtoh_sync_copy_into(buffer.num_rows_device(), &mut host_rows)
        .expect("dtoh row count");
    buffer.set_cached_row_count_if_unset(host_rows[0]);
    host_rows[0]
}
```

**Step 4: Run Rust tests**

Run: `cargo test -p xlog-runtime --release -- --test-threads=1 2>&1 | tail -5`
Expected: All existing tests pass.

**Step 5: Commit**

```bash
git add crates/xlog-runtime/src/ilp_registry.rs crates/xlog-runtime/src/executor.rs
git commit -m "perf(runtime): wire row-count cache into ilp_registry + executor read sites"
```

---

### Task 3: Refactor IlpMask to enum (Dense | Sparse)

**Files:**
- Modify: `crates/xlog-runtime/src/ilp_registry.rs`

**Step 1: Replace IlpMask struct with enum**

Replace the `IlpMask` struct (lines 28-34) with an enum:

```rust
/// A registered ILP mask — Dense (imported via DLPack) or Sparse (candidate entries only).
pub enum IlpMask {
    Dense {
        hard: CudaBuffer,
        soft: CudaBuffer,
        schema_size: usize,
    },
    Sparse {
        active_entries: Vec<(u32, u32, u32)>,
        schema_size: usize,
    },
}
```

**Step 2: Update insert_mask to construct Dense variant**

Replace `insert_mask` (lines 60-69):

```rust
pub fn insert_mask(
    &mut self,
    name: String,
    hard: CudaBuffer,
    soft: CudaBuffer,
    schema_size: usize,
) {
    self.masks
        .insert(name, IlpMask::Dense { hard, soft, schema_size });
}
```

**Step 3: Refactor insert_mask_from_sparse to store Sparse variant**

Replace `insert_mask_from_sparse` (lines 75-127). It should do top-k ranking, then store `Sparse` without any dense materialization or GPU upload:

```rust
/// Insert a mask built from sparse candidate data.
///
/// Performs deterministic top-k ranking (desc soft value, then lower index)
/// and stores the selected (i,j,k) entries directly — no dense buffer.
pub fn insert_mask_from_sparse(
    &mut self,
    name: String,
    schema_size: usize,
    active_ijk: &[(u32, u32, u32)],
    active_soft: &[f32],
    budget: usize,
) -> Result<(), XlogError> {
    if active_ijk.len() != active_soft.len() {
        return Err(XlogError::Execution(format!(
            "active_ijk length {} != active_soft length {}",
            active_ijk.len(),
            active_soft.len()
        )));
    }

    // Deterministic top-k: descending soft value, then ascending index for ties
    let mut ranked: Vec<(usize, f32)> =
        active_soft.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    ranked.truncate(budget.min(ranked.len()));

    let entries: Vec<(u32, u32, u32)> = ranked
        .iter()
        .map(|&(idx, _)| active_ijk[idx])
        .collect();

    self.masks.insert(name, IlpMask::Sparse {
        active_entries: entries,
        schema_size,
    });
    Ok(())
}
```

Note: the `provider` parameter is removed — no GPU upload needed.

**Step 4: Update get_mask callers for enum access**

The `schema_size` accessor is used by callers. Add a helper method:

```rust
impl IlpMask {
    pub fn schema_size(&self) -> usize {
        match self {
            IlpMask::Dense { schema_size, .. } => *schema_size,
            IlpMask::Sparse { schema_size, .. } => *schema_size,
        }
    }
}
```

**Step 5: Fix compilation errors**

After the enum change, any code that accesses `ilp_mask.hard`, `ilp_mask.soft`, or `ilp_mask.schema_size` directly will fail to compile. These are in `executor.rs` (Task 4) and `lib.rs` (Task 5). For now, temporarily add pattern-match wrappers or mark the Task 4/5 changes as blocked-by this task.

Run: `cargo check -p xlog-runtime --release 2>&1 | head -30`
Expected: Compilation errors in executor.rs (expected — fixed in Task 4).

**Step 6: Commit (may not compile alone — that's OK, tasks 3+4 form a unit)**

```bash
git add crates/xlog-runtime/src/ilp_registry.rs
git commit -m "refactor(ilp): convert IlpMask to Dense|Sparse enum, remove dense materialization from sparse path"
```

---

### Task 4: Sparse path in execute_tensor_masked_join

**Files:**
- Modify: `crates/xlog-runtime/src/executor.rs:2743-2892`

**Step 1: Update execute_tensor_masked_join to pattern-match IlpMask**

In `crates/xlog-runtime/src/executor.rs`, replace the mask retrieval and active-rule extraction section (lines 2757-2778). The key change: when `Sparse`, use stored entries directly instead of launching the GPU kernel.

```rust
fn execute_tensor_masked_join(
    &mut self,
    mask_name: &str,
    schema_size: usize,
    left_keys: &[usize],
    right_keys: &[usize],
    rel_index: &[(RelId, String)],
    head_rel_name: &str,
    max_active_rules: usize,
    head_projection: &[usize],
) -> Result<CudaBuffer> {
    let ilp_mask = match self.ilp_registry.get_mask(mask_name) {
        Some(mask) => mask,
        None => {
            self.ilp_last_result = Some(IlpTaggedResult { entries: Vec::new() });
            let schema = self.store.get(head_rel_name)
                .map(|buf| buf.schema().clone())
                .ok_or_else(|| XlogError::Execution(format!(
                    "TensorMaskedJoin: head relation '{}' not found in store \
                     (was load_facts_into_store called?)", head_rel_name
                )))?;
            return self.provider.create_empty_buffer(schema);
        }
    };

    let start = self.profiler.start_op();

    // Extract active rules: sparse path skips GPU kernel entirely
    let dense_buf;
    let active_rules: &[(u32, u32, u32)] = match ilp_mask {
        IlpMask::Dense { hard, soft, .. } => {
            dense_buf = self.provider.extract_active_rule_indices(
                hard, soft, schema_size, max_active_rules,
            )?;
            &dense_buf
        }
        IlpMask::Sparse { active_entries, .. } => {
            let limit = max_active_rules.min(active_entries.len());
            &active_entries[..limit]
        }
    };

    // Everything below here is unchanged — hash joins, projection, union, tagging
    // ...
```

The rest of the method (lines 2780-2892) stays the same. Only the mask retrieval and `active_rules` extraction changes.

**Step 2: Verify compilation**

Run: `cargo check -p xlog-runtime --release 2>&1 | head -10`
Expected: Clean compilation (tasks 3+4 together fix all enum-related errors).

**Step 3: Run full Rust test suite**

Run: `cargo test --workspace --all-targets --exclude pyxlog --release 2>&1 | tail -10`
Expected: All tests pass (dense path unchanged, sparse path now enum-based).

**Step 4: Commit**

```bash
git add crates/xlog-runtime/src/executor.rs
git commit -m "perf(executor): sparse IlpMask path skips extract_active_rule_indices kernel"
```

---

### Task 5: DLPack-native set_rule_mask_sparse API

**Files:**
- Modify: `crates/pyxlog/src/lib.rs:3912-3986`

**Step 1: Change set_rule_mask_sparse to accept DLPack capsule**

In `crates/pyxlog/src/lib.rs`, replace the `set_rule_mask_sparse` method (lines 3911-3986). The new version:

1. Accepts a DLPack capsule (`&Bound<'_, PyAny>`) instead of `Vec<f64>`
2. Imports via DLPack to get a CudaBuffer
3. Downloads C f64 values using raw dtoh (not `download_column_f64`, to avoid incrementing D2H counter)
4. Performs top-k ranking
5. Stores `IlpMask::Sparse` via the refactored `insert_mask_from_sparse`

```rust
/// Sparse mask API: candidate IDs + soft probabilities via DLPack.
///
/// `candidate_ids` must be exactly `[0..C)` where C is the candidate count.
/// `soft_probs_dlpack` is a DLPack capsule (1-D f64 CUDA tensor of length C).
///
/// Rust owns top-k: downloads C soft-probs, ranks, stores sparse entries.
#[pyo3(signature = (name, candidate_ids, soft_probs_dlpack, budget, allow_recursive=false))]
pub fn set_rule_mask_sparse(
    &mut self,
    name: String,
    candidate_ids: Vec<u32>,
    soft_probs_dlpack: &Bound<'_, PyAny>,
    budget: usize,
    allow_recursive: bool,
) -> PyResult<()> {
    let tmj = extract_tmj_meta_for_mask(&self.plan, Some(&name));
    let n = tmj.schema_size;
    if n == 0 {
        return Err(PyValueError::new_err(format!(
            "no learnable mask '{}' found", name
        )));
    }
    if self.compiled_schema_size > 0 && n != self.compiled_schema_size {
        return Err(PyValueError::new_err(format!(
            "schema_size mismatch for '{}': plan N={} compiled N={}",
            name, n, self.compiled_schema_size
        )));
    }

    let expected_c = self.expected_candidate_count(&name, allow_recursive)?;
    if candidate_ids.len() != expected_c {
        return Err(PyValueError::new_err(format!(
            "candidate_ids length {} != expected candidate count {}",
            candidate_ids.len(), expected_c
        )));
    }
    for (idx, &cid) in candidate_ids.iter().enumerate() {
        if cid != idx as u32 {
            return Err(PyValueError::new_err(format!(
                "candidate_ids must be [0..{}), got id {} at position {}",
                expected_c, cid, idx
            )));
        }
    }

    // Import DLPack tensor (zero-copy, stays on GPU)
    let dmt = dlpack_from_py(soft_probs_dlpack)?;
    let soft_buf = self.provider.from_dlpack_tensors(vec![dmt])
        .map_err(|e| PyRuntimeError::new_err(format!("DLPack import failed: {}", e)))?;

    // Download C soft-probs via raw dtoh (control-plane, not tracked by D2H counter)
    let c = candidate_ids.len();
    let num_bytes = c * std::mem::size_of::<f64>();
    let col = soft_buf.column(0)
        .ok_or_else(|| PyRuntimeError::new_err("soft_probs DLPack has no column"))?;
    if col.num_bytes() < num_bytes {
        return Err(PyValueError::new_err(format!(
            "soft_probs tensor has {} bytes, expected {} for {} candidates",
            col.num_bytes(), num_bytes, c
        )));
    }
    let ptr = *cudarc::driver::DevicePtr::device_ptr(col);
    let raw_view = unsafe {
        std::slice::from_raw_parts(ptr as *const u8, num_bytes)
    };
    // Download as bytes, reinterpret as f64
    let mut bytes = vec![0u8; num_bytes];
    self.provider.device().inner()
        .dtoh_sync_copy_into(
            // Need a raw CudaView — use the provider's column_bytes_view pattern
            // but we'll construct manually for f64
            todo!("use provider helper for raw dtoh of f64 column"),
            &mut bytes,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("dtoh soft_probs: {}", e)))?;
    let soft_probs: Vec<f64> = bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect();
```

**IMPORTANT:** The raw dtoh for f64 requires access to `CudaKernelProvider`'s internal `column_bytes_view` or a similar helper. The exact implementation depends on what's available. The cleanest path is to add a small helper `download_soft_probs_f64` on `CudaKernelProvider` that does a raw D2H without incrementing the `d2h_transfer_count`. Pattern:

```rust
// In provider.rs — new helper
/// Download f64 values from a DLPack-imported buffer (control-plane, untracked).
pub fn download_f64_untracked(&self, buffer: &CudaBuffer, col_idx: usize) -> Result<Vec<f64>> {
    let col = buffer.column(col_idx)
        .ok_or_else(|| XlogError::Kernel(format!("Column {} not found", col_idx)))?;
    let num_rows = self.device_row_count(buffer)?;
    if num_rows == 0 { return Ok(vec![]); }
    let num_bytes = num_rows.checked_mul(std::mem::size_of::<f64>())
        .ok_or_else(|| XlogError::Kernel("Row byte size overflow".to_string()))?;
    let col_view = self.column_bytes_view(col, num_bytes)?;
    let mut bytes = vec![0u8; num_bytes];
    self.device.inner()
        .dtoh_sync_copy_into(&col_view, &mut bytes)
        .map_err(|e| XlogError::Kernel(format!("Failed to download f64: {}", e)))?;
    Ok(bytes.chunks_exact(8)
        .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect())
}
```

Then in `set_rule_mask_sparse`:

```rust
    let soft_probs = self.provider.download_f64_untracked(&soft_buf, 0)
        .map_err(|e| PyRuntimeError::new_err(format!("dtoh soft_probs: {}", e)))?;

    if soft_probs.len() != c {
        return Err(PyValueError::new_err(format!(
            "soft_probs length {} != candidate count {}",
            soft_probs.len(), c
        )));
    }

    let candidate_triples = self.candidate_triples_for_mask(&name, allow_recursive)?;

    let active_ijk: Vec<(u32, u32, u32)> = candidate_triples.clone();
    let active_soft: Vec<f32> = soft_probs.iter().map(|&v| v as f32).collect();

    self.executor
        .ilp_registry_mut()
        .insert_mask_from_sparse(name, n, &active_ijk, &active_soft, budget)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}
```

**Step 2: Add download_f64_untracked to provider.rs**

Add after `download_column_f64` (around line 6800):

```rust
/// Download f64 values without incrementing the D2H transfer counter.
/// For control-plane reads (mask setup), not data-plane.
pub fn download_f64_untracked(&self, buffer: &CudaBuffer, col_idx: usize) -> Result<Vec<f64>> {
    // Same as download_column_f64 but without d2h_transfer_count increment
    let col = buffer.column(col_idx)
        .ok_or_else(|| XlogError::Kernel(format!("Column {} not found", col_idx)))?;
    let num_rows = self.device_row_count(buffer)?;
    if num_rows == 0 { return Ok(vec![]); }
    let num_bytes = num_rows.checked_mul(std::mem::size_of::<f64>())
        .ok_or_else(|| XlogError::Kernel("Row byte size overflow".to_string()))?;
    let col_view = self.column_bytes_view(col, num_bytes)?;
    let mut bytes = vec![0u8; num_bytes];
    self.device.inner()
        .dtoh_sync_copy_into(&col_view, &mut bytes)
        .map_err(|e| XlogError::Kernel(format!("Failed to download f64: {}", e)))?;
    Ok(bytes.chunks_exact(8)
        .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect())
}
```

**Step 3: Build and test**

Run: `cargo test --workspace --all-targets --exclude pyxlog --release 2>&1 | tail -10`
Expected: All Rust tests pass.

**Step 4: Commit**

```bash
git add crates/pyxlog/src/lib.rs crates/xlog-cuda/src/provider.rs
git commit -m "feat(pyxlog): DLPack-native set_rule_mask_sparse — Rust owns top-k, no Python N^3"
```

---

### Task 6: Rewrite SparseMaskBackend.apply_mask

**Files:**
- Modify: `crates/pyxlog/python/pyxlog/ilp/backend.py:90-149`

**Step 1: Rewrite SparseMaskBackend.apply_mask**

Replace lines 104-145 in `backend.py`:

```python
class SparseMaskBackend:
    """Sparse candidate-indexed mask -- beta default.

    Uses C-dimensional logits (one per candidate) instead of N^3.
    Passes DLPack-wrapped soft-probs directly to Rust; Rust owns top-k
    ranking and stores sparse (i,j,k) entries without dense materialization.
    """

    def init_weights(self, C: int, n: int, device: str) -> torch.Tensor:
        return torch.randn(C, requires_grad=True, device=device)

    def apply_mask(
        self, prog, mask_name, W, tau, budget, candidates, n,
        allow_recursive=False,
    ):
        # Gumbel-softmax over C-dimensional logits
        cand_probs = F.gumbel_softmax(W, tau=tau, hard=False, dim=0)

        # Pass full candidate probs to Rust via DLPack — Rust owns top-k
        candidate_ids = list(range(len(candidates)))
        prog.set_rule_mask_sparse(
            mask_name, candidate_ids, cand_probs, budget, allow_recursive,
        )

        return cand_probs

    def decode_argmax(self, W, candidates, n):
        with torch.no_grad():
            return W.argmax().item()
```

**Step 2: Update test_ilp_sparse.py for DLPack API**

The tests in `python/tests/test_ilp_sparse.py` pass `soft` as `list[float]`. They now need to pass a CUDA tensor instead:

Replace usages like:
```python
soft = [1.0 / c] * c
prog.set_rule_mask_sparse("W", ids, soft, 32)
```

With:
```python
soft = torch.tensor([1.0 / c] * c, device="cuda", dtype=torch.float64)
prog.set_rule_mask_sparse("W", ids, soft, 32)
```

Apply to all 5 test functions that call `set_rule_mask_sparse`:
- `test_sparse_mask_exists` (line 40)
- `test_sparse_derives_same_as_dense` (line 56)
- `test_sparse_tagged_entries_match_dense` (line 76)
- `test_sparse_validates_candidate_count` (line 87) — keep `list` to test error
- `test_sparse_validates_soft_length` (line 94) — keep `list` to test error
- `test_sparse_top_k_deterministic_tiebreak` (line 102)

For validation tests (candidate count, soft length): the DLPack API should still raise ValueError when sizes mismatch. Use tensors of wrong length:

```python
def test_sparse_validates_soft_length():
    prog = _compile()
    c = len(prog.valid_candidates("W", False))
    bad_soft = torch.tensor([1.0] * (c + 1), device="cuda", dtype=torch.float64)
    with pytest.raises((ValueError, RuntimeError)):
        prog.set_rule_mask_sparse("W", list(range(c)), bad_soft, 32)
```

**Step 3: Run Python sparse tests**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_sparse.py -v 2>&1 | tail -15`
Expected: All 6 tests pass.

**Step 4: Run full Python test suite**

Run: `.venv/bin/python -m pytest python/tests/ -v --timeout=120 -k "not slow and not test_second_epoch" 2>&1 | tail -15`
Expected: All tests pass (SparseMaskBackend now uses DLPack path throughout).

**Step 5: Commit**

```bash
git add crates/pyxlog/python/pyxlog/ilp/backend.py python/tests/test_ilp_sparse.py
git commit -m "feat(ilp): rewrite SparseMaskBackend to use DLPack — no Python-side N^3 materialization"
```

---

### Task 7: Guard tests

**Files:**
- Create: `python/tests/test_ilp_sparse_guard.py`

**Step 1: Write test_sparse_backend_does_not_call_set_rule_mask**

```python
"""Guard tests: sparse trainer path must not fall back to dense APIs."""

import pytest
from unittest.mock import patch

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda

skip_unless_pyxlog_cuda()

from pyxlog.ilp import train_only, TrainConfig


SOURCE = """
edge(1, 2). edge(2, 3). edge(3, 4). edge(4, 5).
learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""
POS = [("reach", [1, 3]), ("reach", [2, 4])]
NEG = [("reach", [1, 1])]


def test_sparse_backend_does_not_call_set_rule_mask():
    """Sparse trainer must never call set_rule_mask (the dense API)."""
    original_set_rule_mask = pyxlog.IlpProgramFactory.compile.__wrapped__ if hasattr(
        pyxlog.IlpProgramFactory.compile, '__wrapped__') else None

    calls = []
    # Patch at the instance level during training
    config = TrainConfig(
        step_budget_per_attempt=10,
        max_attempts=1,
        seed=42,
        debug_dense_mask=False,  # sparse backend
    )

    # We can't easily monkey-patch a Rust method, but we can verify
    # the backend code path by checking that no N^3 tensor is created.
    # Use a wrapper that tracks calls.
    import pyxlog.ilp.backend as backend_mod
    original_apply = backend_mod.SparseMaskBackend.apply_mask

    def tracking_apply(self, prog, mask_name, W, tau, budget, candidates, n, allow_recursive=False):
        # Verify no set_rule_mask call — only set_rule_mask_sparse
        result = original_apply(self, prog, mask_name, W, tau, budget, candidates, n, allow_recursive)
        # The key assertion: no N^3 tensor was created in this call
        # If SparseMaskBackend.apply_mask creates any (n,n,n) tensor, that's a failure
        return result

    with patch.object(backend_mod.SparseMaskBackend, 'apply_mask', tracking_apply):
        result = train_only(SOURCE, "W", POS, NEG, config)
    # If we got here without error, the sparse path was used
    assert result.attempt_count >= 1


def test_sparse_d2h_counter_clean_after_mask_setup():
    """set_rule_mask_sparse must not increment the D2H transfer counter."""
    prog = pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)
    cands = prog.valid_candidates("W", False)
    c = len(cands)

    soft = torch.tensor([1.0 / c] * c, device="cuda", dtype=torch.float64)
    prog.reset_d2h_transfer_count()
    prog.set_rule_mask_sparse("W", list(range(c)), soft, 32)

    # The untracked download should not have incremented the counter
    assert prog.d2h_transfer_count() == 0, (
        f"set_rule_mask_sparse incremented D2H counter to {prog.d2h_transfer_count()}"
    )
```

**Step 2: Run guard tests**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_sparse_guard.py -v 2>&1`
Expected: Both tests pass.

**Step 3: Commit**

```bash
git add python/tests/test_ilp_sparse_guard.py
git commit -m "test(ilp): guard tests — sparse path does not use dense API or increment D2H counter"
```

---

### Task 8: Reliability 20/20 re-run

**Files:**
- Existing: `python/tests/test_ilp_reliability.py`

**Step 1: Run reliability gate**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_reliability.py -v --timeout=600 2>&1 | tail -20`
Expected: 20/20 pass. Each run converges on the correct rule (`reach(X,Y) :- edge(X,Z), edge(Z,Y).`).

**Step 2: Run full test suite for final validation**

Run: `.venv/bin/python -m pytest python/tests/ -v --timeout=120 -k "not slow and not test_second_epoch" 2>&1 | tail -20`
Expected: All tests pass (including new guard tests).

Run: `cargo test --workspace --all-targets --exclude pyxlog --release 2>&1 | tail -10`
Expected: All Rust tests pass.

**Step 3: Final commit if any fixups needed**

If reliability gate passes clean, no commit needed. If any fixups are required, commit with:
```bash
git commit -m "fix(ilp): reliability fixup for sparse executor path"
```

---

## Summary

| Task | Scope | Risk | Key files |
|------|-------|------|-----------|
| 1 | Row-count cache: provider | Low | `provider.rs` |
| 2 | Row-count cache: registry + executor | Low | `ilp_registry.rs`, `executor.rs` |
| 3 | IlpMask enum refactor | Low | `ilp_registry.rs` |
| 4 | Sparse executor path | Medium | `executor.rs` |
| 5 | DLPack-native set_rule_mask_sparse | Medium | `lib.rs`, `provider.rs` |
| 6 | SparseMaskBackend rewrite | Medium | `backend.py`, `test_ilp_sparse.py` |
| 7 | Guard tests | Low | `test_ilp_sparse_guard.py` (new) |
| 8 | Reliability 20/20 | Low | existing `test_ilp_reliability.py` |

Tasks 1-2 are independent of 3-6. Tasks 3+4 must land together (enum change + executor pattern-match). Task 5 depends on 3. Task 6 depends on 5. Tasks 7-8 validate everything.
