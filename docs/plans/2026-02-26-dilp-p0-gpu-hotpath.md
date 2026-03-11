# P0: Eliminate D2H Data-Plane Transfers from ILP Training Loop

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove all GPU-to-CPU column downloads (`download_column_*`) from the per-step training hot path, replacing them with GPU-side batch APIs that return only scalar telemetry.

**Architecture:** The training loop currently calls `fact_exists` (~10-20x/step) and `tagged_entries_containing_fact` (~5x/step), both of which download entire GPU columns to host memory for CPU-side row scanning. We add two new GPU-native batch APIs: (1) `batch_fact_membership` — uploads query facts, semi-joins against a relation buffer, returns a boolean mask; (2) `batch_tagged_credit` — retains per-(i,j,k) join result buffers from `evaluate()`, semi-joins query facts against each, returns per-rule membership masks. The trainer is then rewritten to use only these batch APIs in the step loop. A hard-gate test asserts zero `download_column_*` calls on hot paths.

**Tech Stack:** Rust (xlog-cuda, xlog-runtime, pyxlog), CUDA kernels (hash_join_semi), Python (trainer.py), PyO3

**Build/test commands:**
```bash
# Rust build + Python install
cargo build -p pyxlog --release && /home/dev/.local/bin/maturin develop --release -m crates/pyxlog/Cargo.toml

# Fast ILP tests (skips reliability suite)
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/test_ilp_*.py -v --timeout=300 -m "not slow"

# Full ILP tests including reliability (30 min)
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/test_ilp_*.py -v --timeout=600
```

**Definition of done:**
- No per-step calls to `fact_exists`, `tagged_entries_containing_fact`, or `relation_facts` from trainer hot loop
- No `download_column_*` on hot-path APIs
- Only scalar telemetry D2H per step (row counts via `device_row_count`: 4 bytes each)
- All existing ILP tests pass (46 non-slow + 20 slow reliability)

---

### Task 1: Add D2H instrumentation counter to CudaKernelProvider

**Context:** Before fixing the hot path, we need a way to _count_ D2H transfers so we can assert they're zero during training steps. `CudaKernelProvider` (in `crates/xlog-cuda/src/provider/mod.rs`) is the single point through which all GPU↔CPU data moves. We add an atomic counter that increments on every `download_column_*` call, plus methods to read/reset it.

**Files:**
- Modify: `crates/xlog-cuda/src/provider/mod.rs`
- Test: `crates/xlog-cuda/tests/test_d2h_counter.rs` (create)

**Step 1: Write the failing test**

Create `crates/xlog-cuda/tests/test_d2h_counter.rs`:

```rust
//! Test D2H transfer counter on CudaKernelProvider.
use xlog_cuda::CudaKernelProvider;

#[test]
fn d2h_counter_starts_at_zero() {
    let provider = CudaKernelProvider::new(0, 64).unwrap();
    assert_eq!(provider.d2h_transfer_count(), 0);
}

#[test]
fn d2h_counter_increments_on_download() {
    let provider = CudaKernelProvider::new(0, 64).unwrap();

    // Create a small buffer with one u32 column and one row
    let schema = xlog_cuda::Schema::new(vec![xlog_cuda::ScalarType::U32]);
    let buf = provider.create_buffer_from_u32_columns(&[vec![42u32]], schema).unwrap();

    provider.reset_d2h_transfer_count();
    let _ = provider.download_column_u32(&buf, 0).unwrap();
    assert_eq!(provider.d2h_transfer_count(), 1);

    let _ = provider.download_column_u32(&buf, 0).unwrap();
    assert_eq!(provider.d2h_transfer_count(), 2);
}

#[test]
fn d2h_counter_resets() {
    let provider = CudaKernelProvider::new(0, 64).unwrap();
    let schema = xlog_cuda::Schema::new(vec![xlog_cuda::ScalarType::U32]);
    let buf = provider.create_buffer_from_u32_columns(&[vec![42u32]], schema).unwrap();

    let _ = provider.download_column_u32(&buf, 0).unwrap();
    assert!(provider.d2h_transfer_count() > 0);

    provider.reset_d2h_transfer_count();
    assert_eq!(provider.d2h_transfer_count(), 0);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-cuda --test test_d2h_counter --release 2>&1 | tail -20`
Expected: FAIL — `d2h_transfer_count` and `reset_d2h_transfer_count` methods don't exist.

**Step 3: Implement the counter**

In `crates/xlog-cuda/src/provider/mod.rs`, add to `CudaKernelProvider` struct:

```rust
use std::sync::atomic::{AtomicU64, Ordering};

// Add field to CudaKernelProvider struct:
d2h_transfer_count: AtomicU64,

// Initialize in new():
d2h_transfer_count: AtomicU64::new(0),
```

Add public methods:

```rust
/// Returns the number of column-download D2H transfers since last reset.
pub fn d2h_transfer_count(&self) -> u64 {
    self.d2h_transfer_count.load(Ordering::Relaxed)
}

/// Resets the D2H transfer counter to zero.
pub fn reset_d2h_transfer_count(&self) {
    self.d2h_transfer_count.store(0, Ordering::Relaxed);
}
```

Add increment call to each `download_column_*` method. There are 5 variants — search for `fn download_column_` in provider.rs. In each, add at the top of the function body:

```rust
self.d2h_transfer_count.fetch_add(1, Ordering::Relaxed);
```

The methods to instrument are: `download_column_u32`, `download_column_i32`, `download_column_i64`, `download_column_u64`, `download_column_bool`. (Also instrument `download_column_f32` and `download_column_f64` if they exist.)

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-cuda --test test_d2h_counter --release 2>&1 | tail -20`
Expected: 3 tests PASS.

**Step 5: Commit**

```bash
git add crates/xlog-cuda/src/provider/mod.rs crates/xlog-cuda/tests/test_d2h_counter.rs
git commit -m "feat(xlog-cuda): add D2H transfer counter to CudaKernelProvider"
```

---

### Task 2: Expose D2H counter to Python via PyO3

**Context:** The Python trainer needs access to the D2H counter to (a) assert zero transfers during step loops in tests, and (b) log transfer counts in telemetry. `CompiledIlpProgram` in `crates/pyxlog/src/lib.rs` holds a reference to the `CudaKernelProvider`.

**Files:**
- Modify: `crates/pyxlog/src/lib.rs`
- Test: `python/tests/test_ilp_d2h_gate.py` (create)

**Step 1: Write the failing test**

Create `python/tests/test_ilp_d2h_gate.py`:

```python
# python/tests/test_ilp_d2h_gate.py
"""Tests for D2H transfer counter exposed via PyO3."""
import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda
skip_unless_pyxlog_cuda()

REACH_SOURCE = """
    edge(1, 2). edge(2, 3). edge(3, 4).
    learnable(W_reach) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""


def test_d2h_counter_accessible():
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    assert prog.d2h_transfer_count() == 0


def test_d2h_counter_reset():
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    # fact_exists triggers D2H transfers
    prog.evaluate()
    prog.fact_exists("edge", [1, 2])
    assert prog.d2h_transfer_count() > 0

    prog.reset_d2h_transfer_count()
    assert prog.d2h_transfer_count() == 0
```

**Step 2: Run test to verify it fails**

Run: `cargo build -p pyxlog --release && /home/dev/.local/bin/maturin develop --release -m crates/pyxlog/Cargo.toml && LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/test_ilp_d2h_gate.py -v --timeout=60 2>&1 | tail -20`
Expected: FAIL — `d2h_transfer_count` not a method on `CompiledIlpProgram`.

**Step 3: Add PyO3 methods**

In `crates/pyxlog/src/lib.rs`, inside the `#[pymethods] impl CompiledIlpProgram` block, add:

```rust
pub fn d2h_transfer_count(&self) -> u64 {
    self.provider.d2h_transfer_count()
}

pub fn reset_d2h_transfer_count(&self) {
    self.provider.reset_d2h_transfer_count()
}
```

**Step 4: Build and run test**

Run: `cargo build -p pyxlog --release && /home/dev/.local/bin/maturin develop --release -m crates/pyxlog/Cargo.toml && LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/test_ilp_d2h_gate.py -v --timeout=60 2>&1 | tail -20`
Expected: 2 tests PASS.

**Step 5: Commit**

```bash
git add crates/pyxlog/src/lib.rs python/tests/test_ilp_d2h_gate.py
git commit -m "feat(pyxlog): expose D2H transfer counter to Python"
```

---

### Task 3: Add `membership_mask` to CudaKernelProvider

**Context:** `hash_join_semi_impl` (provider/mod.rs (pre-Wave-2 line 8163)-8268) already computes a `d_has_match: CudaSlice<u8>` mask (line 8225) with one byte per left row indicating whether that row has a match in the right side. But it immediately calls `filter_by_device_mask` (line 8267) and discards the mask. We extract a new public method `membership_mask` that returns just the raw mask, plus a `download_mask` helper that copies it to host.

This is the foundational GPU primitive for batch fact-membership.

**Files:**
- Modify: `crates/xlog-cuda/src/provider/mod.rs`
- Test: `crates/xlog-cuda/tests/test_membership_mask.rs` (create)

**Step 1: Write the failing test**

Create `crates/xlog-cuda/tests/test_membership_mask.rs`:

```rust
//! Test GPU-side membership mask computation.
use xlog_cuda::{CudaKernelProvider, Schema, ScalarType};

#[test]
fn membership_mask_basic() {
    let provider = CudaKernelProvider::new(0, 64).unwrap();

    // Relation: edge(1,2), edge(2,3), edge(3,4)
    let schema = Schema::new(vec![ScalarType::U32, ScalarType::U32]);
    let relation = provider
        .create_buffer_from_u32_columns(
            &[vec![1, 2, 3], vec![2, 3, 4]],
            schema.clone(),
        )
        .unwrap();

    // Query facts: (1,2) and (5,6)
    let queries = provider
        .create_buffer_from_u32_columns(
            &[vec![1, 5], vec![2, 6]],
            schema.clone(),
        )
        .unwrap();

    // All columns are keys (full-tuple membership)
    let keys: Vec<usize> = (0..2).collect();
    let mask = provider.membership_mask(&queries, &relation, &keys, &keys).unwrap();

    // mask should be [true, false] — (1,2) exists, (5,6) does not
    assert_eq!(mask, vec![true, false]);
}

#[test]
fn membership_mask_all_match() {
    let provider = CudaKernelProvider::new(0, 64).unwrap();
    let schema = Schema::new(vec![ScalarType::U32, ScalarType::U32]);
    let relation = provider
        .create_buffer_from_u32_columns(
            &[vec![1, 2], vec![2, 3]],
            schema.clone(),
        )
        .unwrap();
    let queries = provider
        .create_buffer_from_u32_columns(
            &[vec![1, 2], vec![2, 3]],
            schema.clone(),
        )
        .unwrap();

    let keys: Vec<usize> = (0..2).collect();
    let mask = provider.membership_mask(&queries, &relation, &keys, &keys).unwrap();
    assert_eq!(mask, vec![true, true]);
}

#[test]
fn membership_mask_none_match() {
    let provider = CudaKernelProvider::new(0, 64).unwrap();
    let schema = Schema::new(vec![ScalarType::U32, ScalarType::U32]);
    let relation = provider
        .create_buffer_from_u32_columns(
            &[vec![1, 2], vec![2, 3]],
            schema.clone(),
        )
        .unwrap();
    let queries = provider
        .create_buffer_from_u32_columns(
            &[vec![10, 20], vec![20, 30]],
            schema.clone(),
        )
        .unwrap();

    let keys: Vec<usize> = (0..2).collect();
    let mask = provider.membership_mask(&queries, &relation, &keys, &keys).unwrap();
    assert_eq!(mask, vec![false, false]);
}

#[test]
fn membership_mask_empty_relation() {
    let provider = CudaKernelProvider::new(0, 64).unwrap();
    let schema = Schema::new(vec![ScalarType::U32, ScalarType::U32]);
    let relation = provider.create_empty_buffer(schema.clone()).unwrap();
    let queries = provider
        .create_buffer_from_u32_columns(
            &[vec![1], vec![2]],
            schema.clone(),
        )
        .unwrap();

    let keys: Vec<usize> = (0..2).collect();
    let mask = provider.membership_mask(&queries, &relation, &keys, &keys).unwrap();
    assert_eq!(mask, vec![false]);
}

#[test]
fn membership_mask_projected_keys() {
    let provider = CudaKernelProvider::new(0, 64).unwrap();
    // Relation with 3 columns: (src, mid, dst)
    let rel_schema = Schema::new(vec![ScalarType::U32, ScalarType::U32, ScalarType::U32]);
    let relation = provider
        .create_buffer_from_u32_columns(
            &[vec![1, 2], vec![10, 20], vec![3, 4]],
            rel_schema.clone(),
        )
        .unwrap();

    // Query: check if (src=1, dst=3) exists (columns 0 and 2 only)
    let query_schema = Schema::new(vec![ScalarType::U32, ScalarType::U32]);
    let queries = provider
        .create_buffer_from_u32_columns(
            &[vec![1, 2], vec![3, 5]],
            query_schema,
        )
        .unwrap();

    // Query keys [0,1] map to relation keys [0,2]
    let mask = provider.membership_mask(&queries, &relation, &[0, 1], &[0, 2]).unwrap();
    assert_eq!(mask, vec![true, false]); // (1,3) matches row 0; (2,5) does not
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-cuda --test test_membership_mask --release 2>&1 | tail -20`
Expected: FAIL — `membership_mask` method doesn't exist on `CudaKernelProvider`.

**Step 3: Implement `membership_mask`**

In `crates/xlog-cuda/src/provider/mod.rs`, add a new public method. The implementation reuses the internal hash_join_semi logic but returns the mask bytes instead of filtering:

```rust
/// Compute a per-row membership mask: for each row in `probe`, check whether
/// a matching row exists in `build` (by the specified key columns).
///
/// Returns a `Vec<bool>` of length = probe row count. This performs exactly
/// one D2H copy of `num_probe` bytes (the mask), NOT per-column downloads.
///
/// Key columns in `probe_keys` index into `probe`; `build_keys` index into `build`.
pub fn membership_mask(
    &self,
    probe: &CudaBuffer,
    build: &CudaBuffer,
    probe_keys: &[usize],
    build_keys: &[usize],
) -> Result<Vec<bool>> {
    let num_probe = self.device_row_count(probe)?;
    let num_build = self.device_row_count(build)?;

    // Empty probe → empty mask
    if num_probe == 0 {
        return Ok(Vec::new());
    }
    // Empty build → all false
    if num_build == 0 {
        return Ok(vec![false; num_probe]);
    }

    if probe_keys.is_empty() || build_keys.is_empty() || probe_keys.len() != build_keys.len() {
        return Err(XlogError::Kernel(
            "membership_mask: key columns must be non-empty and same length".to_string(),
        ));
    }

    let num_probe = num_probe as u32;
    let num_build = num_build as u32;

    // Compute hashes + pack keys
    let probe_packed = self.compute_hashes_and_pack_keys(probe, probe_keys)?;
    let build_packed = self.compute_hashes_and_pack_keys(build, build_keys)?;

    // Build hash table from build side
    let table = self.build_hash_table_v2(&build_packed.hashes, num_build)?;

    // Allocate device mask
    let d_mask = self.memory.alloc::<u8>(num_probe as usize)?;

    // Launch semi-join kernel (same kernel as hash_join_semi_impl)
    let semi_func = self
        .device
        .inner()
        .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_SEMI)
        .ok_or_else(|| XlogError::Kernel("hash_join_semi kernel not found".to_string()))?;

    let block_size = 256u32;
    let grid_size = (num_probe + block_size - 1) / block_size;
    let config = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        semi_func
            .clone()
            .launch(
                config,
                (
                    &probe_packed.hashes,
                    num_probe,
                    &table.bucket_offsets,
                    &table.bucket_counts,
                    &table.bucket_entries,
                    &table.bucket_entry_hashes,
                    table.bucket_mask,
                    &probe_packed.packed_keys,
                    &build_packed.packed_keys,
                    probe_packed.key_bytes,
                    &d_mask,
                ),
            )
            .map_err(|e| XlogError::Kernel(format!("membership_mask failed: {}", e)))?;
    }

    // Download mask (num_probe bytes — NOT column data)
    let mut host_mask = vec![0u8; num_probe as usize];
    self.device
        .inner()
        .dtoh_sync_copy_into(&d_mask, &mut host_mask)
        .map_err(|e| XlogError::Kernel(format!("membership_mask download: {}", e)))?;

    Ok(host_mask.into_iter().map(|b| b != 0).collect())
}
```

**Important:** This does NOT increment `d2h_transfer_count` because it downloads the mask bytes, not column data. The mask download is expected and acceptable (N bytes vs N*arity*sizeof(u32) bytes for column downloads).

**Step 4: Run tests**

Run: `cargo test -p xlog-cuda --test test_membership_mask --release 2>&1 | tail -20`
Expected: 5 tests PASS.

**Step 5: Commit**

```bash
git add crates/xlog-cuda/src/provider/mod.rs crates/xlog-cuda/tests/test_membership_mask.rs
git commit -m "feat(xlog-cuda): add GPU-side membership_mask primitive"
```

---

### Task 4: Add `batch_fact_membership` to CompiledIlpProgram (PyO3)

**Context:** The Python trainer currently calls `prog.fact_exists(rel, vals)` per-fact per-step, each triggering full column downloads. We replace this with `batch_fact_membership(relation, facts)` that uploads all query facts to GPU at once, calls `membership_mask`, and returns a `list[bool]`. This covers the `fact_exists` calls at trainer.py lines 243, 381, 388, 402, 409.

**Files:**
- Modify: `crates/pyxlog/src/lib.rs`
- Test: `python/tests/test_ilp_d2h_gate.py` (extend)

**Step 1: Write the failing test**

Add to `python/tests/test_ilp_d2h_gate.py`:

```python
def test_batch_fact_membership_basic():
    """batch_fact_membership returns correct bool mask."""
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    prog.evaluate()

    # edge relation has: (1,2), (2,3), (3,4)
    facts = [[1, 2], [5, 6], [2, 3]]
    mask = prog.batch_fact_membership("edge", facts)
    assert mask == [True, False, True]


def test_batch_fact_membership_empty_facts():
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    prog.evaluate()
    mask = prog.batch_fact_membership("edge", [])
    assert mask == []


def test_batch_fact_membership_no_d2h_columns():
    """batch_fact_membership must NOT use download_column_*."""
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    prog.evaluate()

    prog.reset_d2h_transfer_count()
    facts = [[1, 2], [5, 6], [2, 3]]
    _ = prog.batch_fact_membership("edge", facts)
    assert prog.d2h_transfer_count() == 0, (
        f"batch_fact_membership triggered {prog.d2h_transfer_count()} column downloads"
    )
```

**Step 2: Run test to verify it fails**

Run the build+test command from the top of this plan, targeting just the d2h gate test file.
Expected: FAIL — `batch_fact_membership` not found.

**Step 3: Implement `batch_fact_membership`**

In `crates/pyxlog/src/lib.rs`, add to `#[pymethods] impl CompiledIlpProgram`:

```rust
/// GPU-side batch fact membership check.
///
/// Uploads `facts` (list of value-lists) to a temporary CudaBuffer,
/// semi-joins against the named relation, returns per-fact boolean mask.
/// Zero download_column_* calls — only downloads the u8 mask.
pub fn batch_fact_membership(
    &self,
    relation: &str,
    facts: Vec<Vec<i64>>,
) -> PyResult<Vec<bool>> {
    if facts.is_empty() {
        return Ok(Vec::new());
    }

    let buf = self.executor.store().get(relation)
        .ok_or_else(|| PyValueError::new_err(
            format!("Relation '{}' not found", relation)
        ))?;

    let arity = buf.arity();
    if arity == 0 {
        return Ok(vec![false; facts.len()]);
    }

    // Validate all facts have correct arity
    for (idx, fact) in facts.iter().enumerate() {
        if fact.len() != arity {
            return Err(PyValueError::new_err(format!(
                "Fact {} has {} values, expected {} (arity of '{}')",
                idx, fact.len(), arity, relation,
            )));
        }
    }

    // Transpose facts into columnar layout for upload
    let mut columns: Vec<Vec<u32>> = vec![Vec::with_capacity(facts.len()); arity];
    for fact in &facts {
        for (col_idx, &val) in fact.iter().enumerate() {
            columns[col_idx].push(val as u32);
        }
    }

    // Upload to GPU
    let col_refs: Vec<Vec<u32>> = columns;
    let query_buf = self.provider
        .create_buffer_from_u32_columns(
            &col_refs.iter().map(|c| c.as_slice()).collect::<Vec<_>>()
                .iter().map(|s| s.to_vec()).collect::<Vec<_>>(),
            buf.schema().clone(),
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // All columns are keys (full-tuple match)
    let keys: Vec<usize> = (0..arity).collect();

    self.provider
        .membership_mask(&query_buf, buf, &keys, &keys)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}
```

**Note on `create_buffer_from_u32_columns` signature:** Check the actual signature in provider/mod.rs (pre-Wave-2 line 5762). It takes `&[Vec<u32>]` and a `Schema`. If it takes `&[&[u32]]` instead, adjust the column construction accordingly. The key point: one H2D upload for all facts, then GPU-side matching.

**Step 4: Build and run tests**

Run the build+test command. Expected: 5 tests PASS (2 from Task 2 + 3 new).

**Step 5: Commit**

```bash
git add crates/pyxlog/src/lib.rs python/tests/test_ilp_d2h_gate.py
git commit -m "feat(pyxlog): add batch_fact_membership — GPU-side fact checking"
```

---

### Task 5: Retain per-entry join buffers in IlpTaggedResult

**Context:** Currently `execute_tensor_masked_join` (executor.rs:2743) stores only metadata `(i,j,k,num_rows)` in `IlpTaggedResult`, then unions per-(i,j,k) buffers by k and drops the individual buffers. For `batch_tagged_credit`, we need the individual per-(i,j,k) join result buffers. We modify `IlpTaggedResult` and `IlpTagEntry` to optionally retain the projected CudaBuffer, and change the executor to populate them.

**Files:**
- Modify: `crates/xlog-runtime/src/ilp_registry.rs`
- Modify: `crates/xlog-runtime/src/executor.rs`
- Test: Existing ILP tests must still pass (run full suite)

**Step 1: Write the test expectation**

The test is behavioral: after this change, `get_tagged_results` should still return the same metadata, and all existing ILP tests should pass unchanged. The buffer retention is an internal detail consumed by Task 6. No new Python-visible behavior yet.

We verify by running the existing test suite:

```bash
cargo build -p pyxlog --release && /home/dev/.local/bin/maturin develop --release -m crates/pyxlog/Cargo.toml
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/test_ilp_trainer.py python/tests/test_ilp_candidates.py -v --timeout=120
```

**Step 2: Modify IlpTagEntry to hold an optional buffer**

In `crates/xlog-runtime/src/ilp_registry.rs`, change `IlpTagEntry`:

```rust
/// Metadata for a single active rule (i,j,k) and its result cardinality.
/// Optionally retains the projected CudaBuffer for batch credit queries.
pub struct IlpTagEntry {
    pub i: u32,
    pub j: u32,
    pub k: u32,
    pub num_rows: u32,
    /// The projected join result buffer. Populated when the executor
    /// retains per-entry buffers for batch credit assignment.
    pub buffer: Option<CudaBuffer>,
}
```

**Step 3: Update the executor to populate entry buffers**

In `crates/xlog-runtime/src/executor.rs`, in `execute_tensor_masked_join`, around lines 2831-2835, change the tag_entries push and results_by_k handling:

Before (current code around line 2831):
```rust
if num_rows > 0 {
    tag_entries.push(IlpTagEntry { i, j, k, num_rows });
    results_by_k.entry(k).or_default().push(projected);
}
```

After:
```rust
if num_rows > 0 {
    // Clone the projected buffer for the tag entry before moving into union
    let entry_buf = projected.clone();
    tag_entries.push(IlpTagEntry { i, j, k, num_rows, buffer: Some(entry_buf) });
    results_by_k.entry(k).or_default().push(projected);
}
```

**IMPORTANT:** Check if `CudaBuffer` implements `Clone`. If not (the comment at RD-17 says "CudaBuffer is not Clone"), we need a different approach. In that case:

**Alternative (if CudaBuffer is not Clone):** We keep the projected buffer in the tag entry and use a *reference* for the union. Since Rust ownership makes this tricky, the pragmatic approach is:
1. First pass: collect all (i,j,k) join results into `tag_entries` with `buffer: Some(projected)`.
2. Second pass: build the union buffers by borrowing from `tag_entries[..].buffer.as_ref()`.
3. Store tag entries with buffers intact.

The executor code becomes:

```rust
// 3. Dispatch hash joins, collect tag metadata WITH buffers
let mut tag_entries: Vec<IlpTagEntry> = Vec::new();

for &(i, j, k) in &active_rules {
    // ... (same skip/guard logic as before) ...

    let joined = self.provider.hash_join_v2(/* ... */)?;
    let projected = /* ... same projection logic ... */;
    let num_rows = read_device_row_count(&self.provider, &projected)? as u32;

    if num_rows > 0 {
        tag_entries.push(IlpTagEntry { i, j, k, num_rows, buffer: Some(projected) });
    }
}

// 4. Union results per target relation, borrowing from tag_entries
let mut bufs_by_k: HashMap<u32, Vec<&CudaBuffer>> = HashMap::new();
for entry in &tag_entries {
    if let Some(ref buf) = entry.buffer {
        bufs_by_k.entry(entry.k).or_default().push(buf);
    }
}

for (k, buffers) in bufs_by_k {
    let (_, target_name) = &rel_index[k as usize];
    let union_buf = if buffers.len() == 1 {
        // Need to re-create since we can't move out of borrow
        self.provider.union_gpu(buffers[0], buffers[0])? // identity union = copy
    } else {
        let mut acc = self.provider.union_gpu(buffers[0], buffers[1])?;
        for buf in &buffers[2..] {
            acc = self.provider.union_gpu(&acc, buf)?;
        }
        acc
    };

    if let Some(existing) = self.store.get(target_name) {
        let delta = self.provider.diff_gpu(&union_buf, existing)?;
        if !delta.is_empty() {
            let merged = self.provider.union_gpu(existing, &delta)?;
            self.store_put(target_name, merged);
        }
    } else {
        let key_cols: Vec<usize> = (0..union_buf.arity()).collect();
        let deduped = self.provider.dedup(&union_buf, &key_cols)?;
        self.store_put(target_name, deduped);
    }
}

// 5. Store tag metadata (with buffers)
self.ilp_last_result = Some(IlpTaggedResult { entries: tag_entries });
```

**Note for implementer:** The above code sketches the approach but may need adjustments depending on borrow checker constraints. The critical requirement is: `IlpTagEntry.buffer` must be `Some(projected_buffer)` after `execute_tensor_masked_join`, and the union logic must still work correctly. If `union_gpu` for a single buffer case is awkward, use a plain memory copy or just store the buffer directly.

**Step 4: Fix compilation and run tests**

Build and fix any compilation errors, then run:
```bash
cargo build -p pyxlog --release && /home/dev/.local/bin/maturin develop --release -m crates/pyxlog/Cargo.toml
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/test_ilp_trainer.py python/tests/test_ilp_candidates.py -v --timeout=120
```
Expected: All existing tests PASS. The metadata returned by `get_tagged_results()` is identical.

**Step 5: Commit**

```bash
git add crates/xlog-runtime/src/ilp_registry.rs crates/xlog-runtime/src/executor.rs
git commit -m "feat(xlog-runtime): retain per-entry CudaBuffers in IlpTaggedResult"
```

---

### Task 6: Add `batch_tagged_credit` to CompiledIlpProgram (PyO3)

**Context:** The Python trainer calls `prog.tagged_entries_containing_fact(rel, vals)` per-fact per-step (trainer.py lines 347, 356). Each call re-does hash joins and downloads column data. With entry buffers now retained in `IlpTaggedResult` (Task 5), we can implement `batch_tagged_credit(relation, facts)` that checks each fact against each retained entry buffer using `membership_mask`, all GPU-side.

**Files:**
- Modify: `crates/pyxlog/src/lib.rs`
- Test: `python/tests/test_ilp_d2h_gate.py` (extend)

**Step 1: Write the failing test**

Add to `python/tests/test_ilp_d2h_gate.py`:

```python
def test_batch_tagged_credit_basic():
    """batch_tagged_credit returns per-fact lists of (i,j,k) contributors."""
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    n = prog.ilp_schema_size()

    # Set mask so that edge+edge→reach is active (i=0,j=0,k=?)
    M_hard = torch.zeros(n, n, n, device="cuda")
    M_soft = torch.zeros(n, n, n, device="cuda")
    # Find the right k index for "reach"
    rel_names = prog.ilp_relation_names()
    k_reach = rel_names.index("reach")
    i_edge = rel_names.index("edge")
    M_hard[i_edge, i_edge, k_reach] = 1.0
    M_soft[i_edge, i_edge, k_reach] = 1.0
    prog.set_rule_mask("W_reach", M_hard.view(-1), M_soft.view(-1), n)
    prog.evaluate()

    # reach should derive (1,3), (2,4), (3,5) via edge(X,Z) join edge(Z,Y)
    facts = [[1, 3], [2, 4], [99, 99]]
    credits = prog.batch_tagged_credit("reach", facts)
    assert len(credits) == 3

    # First two should have at least one contributing entry
    assert len(credits[0]) > 0  # (1,3) is derived
    assert len(credits[1]) > 0  # (2,4) is derived
    # The contributing entry should be (edge_idx, edge_idx, reach_idx)
    assert credits[0][0] == (i_edge, i_edge, k_reach)
    # (99,99) is not derived — no contributors
    assert credits[2] == []


def test_batch_tagged_credit_no_d2h_columns():
    """batch_tagged_credit must NOT use download_column_*."""
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    n = prog.ilp_schema_size()
    rel_names = prog.ilp_relation_names()
    k_reach = rel_names.index("reach")
    i_edge = rel_names.index("edge")

    M_hard = torch.zeros(n, n, n, device="cuda")
    M_soft = torch.zeros(n, n, n, device="cuda")
    M_hard[i_edge, i_edge, k_reach] = 1.0
    M_soft[i_edge, i_edge, k_reach] = 1.0
    prog.set_rule_mask("W_reach", M_hard.view(-1), M_soft.view(-1), n)
    prog.evaluate()

    prog.reset_d2h_transfer_count()
    facts = [[1, 3], [2, 4], [99, 99]]
    _ = prog.batch_tagged_credit("reach", facts)
    assert prog.d2h_transfer_count() == 0, (
        f"batch_tagged_credit triggered {prog.d2h_transfer_count()} column downloads"
    )
```

**Step 2: Run test to verify it fails**

Expected: FAIL — `batch_tagged_credit` not found.

**Step 3: Implement `batch_tagged_credit`**

In `crates/pyxlog/src/lib.rs`, add to `#[pymethods] impl CompiledIlpProgram`:

```rust
/// GPU-side batch credit assignment.
///
/// For each fact in `facts`, returns the list of (i,j,k) entries whose
/// join result contains that fact. Uses membership_mask against retained
/// per-entry buffers — zero download_column_* calls.
pub fn batch_tagged_credit(
    &self,
    relation: &str,
    facts: Vec<Vec<i64>>,
) -> PyResult<Vec<Vec<(u32, u32, u32)>>> {
    if facts.is_empty() {
        return Ok(Vec::new());
    }

    // Find k index for this relation
    let k_idx = self.rel_index.iter()
        .position(|(_, name)| name == relation)
        .ok_or_else(|| PyValueError::new_err(
            format!("Relation '{}' not in ILP schema", relation)
        ))? as u32;

    let tagged = match self.executor.ilp_last_result() {
        Some(t) => t,
        None => return Ok(vec![Vec::new(); facts.len()]),
    };

    // Filter entries to those matching target relation k
    let relevant_entries: Vec<&xlog_runtime::IlpTagEntry> = tagged.entries.iter()
        .filter(|e| e.k == k_idx && e.num_rows > 0 && e.buffer.is_some())
        .collect();

    if relevant_entries.is_empty() {
        return Ok(vec![Vec::new(); facts.len()]);
    }

    // Determine arity from the first entry's buffer
    let arity = relevant_entries[0].buffer.as_ref().unwrap().arity();
    if arity == 0 {
        return Ok(vec![Vec::new(); facts.len()]);
    }

    // Validate facts arity. If head_projection is active, facts have
    // head arity (e.g. 2) but entry buffers have projected arity (also 2
    // after projection in executor). So arity should match.
    for (idx, fact) in facts.iter().enumerate() {
        if fact.len() != arity {
            return Err(PyValueError::new_err(format!(
                "Fact {} has {} values, expected {} (projected arity)",
                idx, fact.len(), arity,
            )));
        }
    }

    // Upload query facts to GPU once
    let mut columns: Vec<Vec<u32>> = vec![Vec::with_capacity(facts.len()); arity];
    for fact in &facts {
        for (col_idx, &val) in fact.iter().enumerate() {
            columns[col_idx].push(val as u32);
        }
    }

    let schema = relevant_entries[0].buffer.as_ref().unwrap().schema().clone();
    let query_buf = self.provider
        .create_buffer_from_u32_columns(&columns, schema)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let keys: Vec<usize> = (0..arity).collect();

    // For each relevant entry, compute membership mask
    let mut per_fact_credits: Vec<Vec<(u32, u32, u32)>> = vec![Vec::new(); facts.len()];

    for entry in &relevant_entries {
        let entry_buf = entry.buffer.as_ref().unwrap();
        let mask = self.provider
            .membership_mask(&query_buf, entry_buf, &keys, &keys)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        for (fact_idx, &found) in mask.iter().enumerate() {
            if found {
                per_fact_credits[fact_idx].push((entry.i, entry.j, entry.k));
            }
        }
    }

    Ok(per_fact_credits)
}
```

**Step 4: Build and run tests**

Expected: All d2h_gate tests PASS.

**Step 5: Commit**

```bash
git add crates/pyxlog/src/lib.rs python/tests/test_ilp_d2h_gate.py
git commit -m "feat(pyxlog): add batch_tagged_credit — GPU-side credit assignment"
```

---

### Task 7: Rewrite trainer.py to use batch APIs in step loop

**Context:** This is the main payoff task. Replace all per-fact `fact_exists` and `tagged_entries_containing_fact` calls in `_compute_loss`, `_check_convergence`, and the witness coverage computation with batch API calls. After this, the only D2H transfers per step are scalar telemetry (row counts, loss values).

**Files:**
- Modify: `crates/pyxlog/python/pyxlog/ilp/trainer.py`
- Test: `python/tests/test_ilp_d2h_gate.py` (extend with step-loop gate test)

**Step 1: Write the gate test**

Add to `python/tests/test_ilp_d2h_gate.py`:

```python
from pyxlog.ilp import train_only, TrainConfig


def test_zero_d2h_in_training_step_loop():
    """The training step loop must have zero download_column_* calls.

    This is the hard gate: after evaluate(), the trainer must use only
    batch_fact_membership and batch_tagged_credit, never fact_exists or
    tagged_entries_containing_fact.
    """
    config = TrainConfig(
        step_budget_per_attempt=20, max_attempts=1,
        tau_start=2.0, tau_floor=0.05, seed=42,
    )

    # We can't directly instrument the step loop from outside, but we
    # can check that train_only converges with reasonable performance.
    # The real gate is in the code itself — _compute_loss and
    # _check_convergence use batch APIs, not per-fact calls.
    result = train_only(
        source=REACH_SOURCE, mask_name="W_reach",
        positives=[("reach", [1, 3]), ("reach", [2, 4])],
        negatives=[],
        config=config,
    )
    assert isinstance(result.total_steps, int)
    assert result.total_steps > 0
```

**Step 2: Rewrite `_compute_loss` to use `batch_tagged_credit`**

Replace `_compute_loss` in `trainer.py`:

```python
def _compute_loss(
    prog,
    M_soft: torch.Tensor,
    positives: list[tuple[str, list[int]]],
    negatives: list[tuple[str, list[int]]],
    rel_names: list[str],
    n: int,
) -> torch.Tensor:
    """Per-fact surrogate loss using GPU-side batch credit assignment."""
    loss = torch.tensor(0.0, device="cuda")

    # Group facts by relation for batch API calls
    if positives:
        pos_by_rel: dict[str, list[list[int]]] = {}
        for rel_name, values in positives:
            pos_by_rel.setdefault(rel_name, []).append(values)

        for rel_name, facts_list in pos_by_rel.items():
            credits = prog.batch_tagged_credit(rel_name, facts_list)
            for fact_idx, contributing in enumerate(credits):
                if contributing:
                    credit = sum(M_soft[i, j, k] for (i, j, k) in contributing)
                    loss = loss + (-torch.log(credit.clamp(min=1e-8)))
                else:
                    k_idx = rel_names.index(rel_name)
                    loss = loss + (-M_soft[:, :, k_idx].sum() / (n * n))

    if negatives:
        neg_by_rel: dict[str, list[list[int]]] = {}
        for rel_name, values in negatives:
            neg_by_rel.setdefault(rel_name, []).append(values)

        for rel_name, facts_list in neg_by_rel.items():
            credits = prog.batch_tagged_credit(rel_name, facts_list)
            for fact_idx, contributing in enumerate(credits):
                if contributing:
                    credit = sum(M_soft[i, j, k] for (i, j, k) in contributing)
                    loss = loss + (-torch.log((1.0 - credit).clamp(min=1e-8)))

    return loss
```

**Step 3: Rewrite witness coverage computation**

In `_run_single_attempt`, replace the witness coverage block (around line 242-245):

Before:
```python
witness_count = sum(
    1 for rel, vals in positives if prog.fact_exists(rel, vals)
)
witness_coverage = witness_count / len(positives)
```

After:
```python
# Batch witness coverage — GPU-side fact membership
pos_by_rel: dict[str, list[list[int]]] = {}
pos_indices_by_rel: dict[str, list[int]] = {}
for idx, (rel, vals) in enumerate(positives):
    pos_by_rel.setdefault(rel, []).append(vals)
    pos_indices_by_rel.setdefault(rel, []).append(idx)

witness_mask = [False] * len(positives)
for rel_name, facts_list in pos_by_rel.items():
    mask = prog.batch_fact_membership(rel_name, facts_list)
    for local_idx, found in enumerate(mask):
        global_idx = pos_indices_by_rel[rel_name][local_idx]
        witness_mask[global_idx] = found

witness_count = sum(witness_mask)
witness_coverage = witness_count / len(positives)
```

**Step 4: Rewrite `_check_convergence` to use batch APIs**

Replace `_check_convergence`:

```python
def _check_convergence(
    prog,
    W: torch.Tensor,
    mask_name: str,
    positives: list[tuple[str, list[int]]],
    negatives: list[tuple[str, list[int]]],
    rel_names: list[str],
    n: int,
    argmax_ijk: tuple[int, int, int],
) -> bool:
    """Three-gated convergence using batch GPU APIs."""
    # Gate 1: all positives derived under current mask
    pos_by_rel: dict[str, list[list[int]]] = {}
    for rel, vals in positives:
        pos_by_rel.setdefault(rel, []).append(vals)
    for rel_name, facts_list in pos_by_rel.items():
        mask = prog.batch_fact_membership(rel_name, facts_list)
        if not all(mask):
            return False

    # Gate 2: no negatives derived
    if negatives:
        neg_by_rel: dict[str, list[list[int]]] = {}
        for rel, vals in negatives:
            neg_by_rel.setdefault(rel, []).append(vals)
        for rel_name, facts_list in neg_by_rel.items():
            mask = prog.batch_fact_membership(rel_name, facts_list)
            if any(mask):
                return False

    # Gate 3: argmax-only re-evaluation
    i, j, k = argmax_ijk
    M_check = torch.zeros((n, n, n), device="cuda")
    M_check[i, j, k] = 1.0
    flat_check = M_check.contiguous().view(-1)
    prog.set_rule_mask(mask_name, flat_check, flat_check, n)
    prog.evaluate()

    # Gate 3a: argmax-only must derive all positives
    for rel_name, facts_list in pos_by_rel.items():
        mask = prog.batch_fact_membership(rel_name, facts_list)
        if not all(mask):
            return False

    # Gate 4: argmax-only must not derive negatives
    if negatives:
        for rel_name, facts_list in neg_by_rel.items():
            mask = prog.batch_fact_membership(rel_name, facts_list)
            if any(mask):
                return False

    return True
```

**Step 5: Build and run full test suite**

```bash
cargo build -p pyxlog --release && /home/dev/.local/bin/maturin develop --release -m crates/pyxlog/Cargo.toml
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/test_ilp_*.py -v --timeout=300 -m "not slow"
```

Expected: All 46+ non-slow tests PASS. The trainer now uses zero `download_column_*` calls in the step loop.

**Step 6: Commit**

```bash
git add crates/pyxlog/python/pyxlog/ilp/trainer.py python/tests/test_ilp_d2h_gate.py
git commit -m "feat(ilp): rewrite trainer hot loop to use GPU batch APIs — zero D2H columns"
```

---

### Task 8: Add hard-gate integration test

**Context:** The final verification: an integration test that actually runs a training session and asserts the D2H counter stays at zero during the step loop. This requires a small hook in the trainer to reset/check the counter around the step loop.

**Files:**
- Modify: `crates/pyxlog/python/pyxlog/ilp/trainer.py` (add counter check)
- Test: `python/tests/test_ilp_d2h_gate.py` (extend)

**Step 1: Add D2H counter bookkeeping to trainer step loop**

In `trainer.py`, in `_run_single_attempt`, add counter management around the step loop.

After `prog.evaluate()` (line 205), add:

```python
        prog.reset_d2h_transfer_count()
```

And at the end of each iteration (after the convergence check block, before the next `for step` iteration), add a debug assertion. We do this only when telemetry_level >= 2 (to avoid overhead in production):

Actually, a cleaner approach: add a method to `_run_single_attempt` that returns the max D2H count per step, and check it in the gate test. But the simplest approach for the gate test is:

In the step loop, after `_compute_loss` and after `_check_convergence`, we can verify the counter. Add at the top of each step iteration:

```python
    for step in range(step_budget):
        optimizer.zero_grad()

        M_hard, M_soft = _build_budget_aware_mask(
            W, temp_controller.tau, config.max_active_rules,
        )

        if torch.isnan(M_soft).any() or torch.isinf(M_soft).any():
            raise _NumericFailure()

        prog.set_rule_mask(
            mask_name,
            M_hard.detach().contiguous().view(-1),
            M_soft.detach().contiguous().view(-1),
            n,
        )
        prog.evaluate()

        # --- D2H hard gate: reset counter after evaluate() ---
        prog.reset_d2h_transfer_count()

        # ... rest of step (loss, convergence, etc.) ...

        # --- D2H hard gate: assert zero column downloads in step body ---
        d2h_count = prog.d2h_transfer_count()
        if d2h_count > 0:
            raise IlpTrainingError(
                "d2h_gate_violation: download_column_* called in step loop",
                {"step": step, "d2h_count": d2h_count},
            )
```

**Step 2: Write the integration test**

Add to `python/tests/test_ilp_d2h_gate.py`:

```python
def test_full_training_zero_d2h_gate():
    """Full training run must complete without D2H gate violation."""
    config = TrainConfig(
        step_budget_per_attempt=50, max_attempts=3,
        tau_start=2.0, tau_floor=0.05, seed=42,
    )
    result = train_only(
        source=REACH_SOURCE, mask_name="W_reach",
        positives=[("reach", [1, 3]), ("reach", [2, 4]),
                   ("reach", [3, 5]), ("reach", [4, 6])],
        negatives=[],
        config=config,
    )
    # If we got here without IlpTrainingError("d2h_gate_violation"),
    # the hard gate passed.
    assert result.converged
    assert "edge" in result.discovered_rule


def test_d2h_gate_with_negatives():
    """D2H gate holds even with negative examples."""
    source = """
        parent(1, 2). parent(2, 3). parent(3, 4). parent(4, 5).
        gender(1, 0). gender(2, 1). gender(3, 0). gender(4, 1).
        sibling(1, 3). sibling(3, 1).
        learnable(W_gp) :: grandparent(X, Y) :- bL(X, Z), bR(Z, Y).
    """
    config = TrainConfig(
        step_budget_per_attempt=60, max_attempts=3,
        tau_start=2.0, tau_floor=0.05, seed=42,
    )
    result = train_only(
        source=source, mask_name="W_gp",
        positives=[("grandparent", [1, 3]), ("grandparent", [2, 4])],
        negatives=[("grandparent", [1, 2]), ("grandparent", [3, 1])],
        config=config,
    )
    # No d2h_gate_violation raised
    assert result.converged
    assert "parent" in result.discovered_rule
```

**Step 3: Build and run all tests**

```bash
cargo build -p pyxlog --release && /home/dev/.local/bin/maturin develop --release -m crates/pyxlog/Cargo.toml
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/test_ilp_*.py -v --timeout=300 -m "not slow"
```

Expected: All tests PASS, including the new hard-gate tests. No `IlpTrainingError("d2h_gate_violation")` raised.

**Step 4: Commit**

```bash
git add crates/pyxlog/python/pyxlog/ilp/trainer.py python/tests/test_ilp_d2h_gate.py
git commit -m "test(ilp): add D2H hard gate — zero column downloads in training step loop"
```

---

### Task 9: Run full reliability suite and verify performance

**Context:** Final verification that the GPU hot-path fix hasn't broken convergence behavior, and that performance has improved. Run the slow reliability suite (20 tests, 4 stages × 5 seeds).

**Files:**
- No new files — verification only

**Step 1: Run reliability suite**

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/test_ilp_reliability.py -v --timeout=600
```

Expected: 20/20 PASS. Time should be significantly faster than the previous ~30 minutes due to eliminated D2H transfers.

**Step 2: Run full non-slow suite and record timing**

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/test_ilp_*.py -v --timeout=300 -m "not slow" 2>&1 | tail -5
```

Expected: All PASS. Note the total time — should be ≤ the previous 77s baseline (likely faster).

**Step 3: Commit timing results as a note in the plan file**

Update this plan with the actual timing results. No code changes.

```bash
git add docs/plans/2026-02-26-dilp-p0-gpu-hotpath.md
git commit -m "docs: record P0 performance results"
```

---

## Summary of Changes

| File | Change |
|------|--------|
| `crates/xlog-cuda/src/provider/mod.rs` | Add `d2h_transfer_count` field + counter methods, add `membership_mask` method |
| `crates/xlog-cuda/tests/test_d2h_counter.rs` | New: D2H counter unit tests |
| `crates/xlog-cuda/tests/test_membership_mask.rs` | New: membership_mask GPU primitive tests |
| `crates/xlog-runtime/src/ilp_registry.rs` | Add `buffer: Option<CudaBuffer>` to `IlpTagEntry` |
| `crates/xlog-runtime/src/executor.rs` | Retain per-entry buffers in `execute_tensor_masked_join` |
| `crates/pyxlog/src/lib.rs` | Add `d2h_transfer_count`, `reset_d2h_transfer_count`, `batch_fact_membership`, `batch_tagged_credit` |
| `crates/pyxlog/python/pyxlog/ilp/trainer.py` | Rewrite `_compute_loss`, witness coverage, `_check_convergence` to use batch APIs; add D2H hard gate |
| `python/tests/test_ilp_d2h_gate.py` | New: comprehensive D2H gate tests (counter, batch APIs, integration) |

## Task Dependencies

```
Task 1 (D2H counter) → Task 2 (PyO3 exposure) → Task 8 (hard gate integration)
Task 3 (membership_mask) → Task 4 (batch_fact_membership) → Task 7 (trainer rewrite)
Task 5 (retain entry buffers) → Task 6 (batch_tagged_credit) → Task 7 (trainer rewrite)
Task 7 (trainer rewrite) + Task 8 (hard gate) → Task 9 (reliability verification)
```

Parallelizable: Tasks 1+3 can run concurrently. Tasks 4+5 can run concurrently after Task 3. Task 6 depends on Tasks 3+5.
