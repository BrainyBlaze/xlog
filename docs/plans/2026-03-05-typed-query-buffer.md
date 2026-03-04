# Typed Query-Buffer Builder Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix `batch_fact_membership` and `batch_tagged_credit` to use schema-aware typed packing instead of blind `as u32` coercion.

**Architecture:** A private helper `pack_i64_columns_typed` dispatches on `schema.column_type(col_idx)` to produce correctly-encoded byte columns. Both batch functions call this helper, then upload via `create_buffer_from_slices`. TDD: tests written first in Python against the PyO3 API.

**Tech Stack:** Rust (PyO3 bindings in `crates/pyxlog/src/lib.rs`), Python tests in `python/tests/`, CUDA via `CudaKernelProvider`.

---

### Task 1: Write typed packing tests (positive cases)

**Files:**
- Create: `python/tests/test_typed_batch_upload.py`

**Step 1: Write the failing test file**

```python
"""Tests for typed batch upload in batch_fact_membership / batch_tagged_credit."""
import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda
skip_unless_pyxlog_cuda()


def _compile_typed(source: str) -> "pyxlog.IlpProgramFactory":
    prog = pyxlog.IlpProgramFactory.compile(source, device=0, memory_mb=64)
    prog.evaluate()
    return prog


def test_membership_u32_columns():
    """U32 columns (default) still work correctly."""
    prog = _compile_typed("""
        pred edge(u32, u32).
        edge(1, 2). edge(2, 3). edge(3, 4).
        learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """)
    mask = prog.batch_fact_membership("edge", [[1, 2], [99, 99], [3, 4]])
    assert mask == [True, False, True]


def test_membership_i32_columns():
    """I32 columns with negative values must round-trip correctly."""
    prog = _compile_typed("""
        pred data(i32, i32).
        data(-10, 20). data(30, -40). data(0, 0).
        learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """)
    mask = prog.batch_fact_membership("data", [[-10, 20], [30, -40], [1, 1]])
    assert mask == [True, True, False]


def test_membership_i64_columns():
    """I64 columns with large values must round-trip correctly."""
    big = 2**40  # exceeds u32 range
    prog = _compile_typed(f"""
        pred data(i64, i64).
        data({big}, 1). data(2, {big}).
        learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """)
    mask = prog.batch_fact_membership("data", [[big, 1], [2, big], [0, 0]])
    assert mask == [True, True, False]


def test_membership_bool_columns():
    """Bool columns accept 0/1 only."""
    prog = _compile_typed("""
        pred flag(u32, bool).
        flag(1, true). flag(2, false).
        learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """)
    mask = prog.batch_fact_membership("flag", [[1, 1], [2, 0], [1, 0]])
    assert mask == [True, True, False]
```

**Step 2: Run tests to verify they fail**

Run: `LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/test_typed_batch_upload.py -v --timeout=120`
Expected: FAIL — I32 negative values, I64 large values, and Bool will produce wrong results due to `as u32` truncation.

**Step 3: Commit failing tests**

```bash
git add python/tests/test_typed_batch_upload.py
git commit -m "test: add typed batch upload tests (expected failures)"
```

---

### Task 2: Implement `pack_i64_columns_typed` helper

**Files:**
- Modify: `crates/pyxlog/src/lib.rs` — add helper function near line 3662 (after `push_term_bytes`, before `load_facts_into_store`)

**Step 1: Add the helper function**

Insert after `push_term_bytes` (after line 3662):

```rust
/// Pack `i64` fact values into typed byte columns according to schema.
/// Returns one `Vec<u8>` per column with correctly-encoded LE bytes.
/// Rejects F32/F64 columns (not supported in batch APIs).
fn pack_i64_columns_typed(
    relation: &str,
    facts: &[Vec<i64>],
    schema: &Schema,
) -> PyResult<Vec<Vec<u8>>> {
    use xlog_core::types::ScalarType;

    let arity = schema.arity();
    // Defensive arity check
    for (idx, fact) in facts.iter().enumerate() {
        if fact.len() != arity {
            return Err(PyValueError::new_err(format!(
                "Relation '{}': fact {} has {} values, expected {}",
                relation, idx, fact.len(), arity,
            )));
        }
    }

    let mut columns: Vec<Vec<u8>> = (0..arity)
        .map(|_| Vec::with_capacity(facts.len() * 8))
        .collect();

    for (row_idx, fact) in facts.iter().enumerate() {
        for (col_idx, &val) in fact.iter().enumerate() {
            let col_type = schema.column_type(col_idx);
            let col = &mut columns[col_idx];
            match col_type {
                ScalarType::U32 => {
                    let v = u32::try_from(val).map_err(|_| PyValueError::new_err(format!(
                        "Relation '{}' column {} (U32): value {} out of range [0, {}]",
                        relation, col_idx, val, u32::MAX,
                    )))?;
                    col.extend_from_slice(&v.to_le_bytes());
                }
                ScalarType::I32 => {
                    let v = i32::try_from(val).map_err(|_| PyValueError::new_err(format!(
                        "Relation '{}' column {} (I32): value {} out of range [{}, {}]",
                        relation, col_idx, val, i32::MIN, i32::MAX,
                    )))?;
                    col.extend_from_slice(&v.to_le_bytes());
                }
                ScalarType::U64 => {
                    let v = u64::try_from(val).map_err(|_| PyValueError::new_err(format!(
                        "Relation '{}' column {} (U64): value {} out of range [0, {}]",
                        relation, col_idx, val, i64::MAX,
                    )))?;
                    col.extend_from_slice(&v.to_le_bytes());
                }
                ScalarType::I64 => {
                    col.extend_from_slice(&val.to_le_bytes());
                }
                ScalarType::Bool => {
                    match val {
                        0 => col.push(0u8),
                        1 => col.push(1u8),
                        _ => return Err(PyValueError::new_err(format!(
                            "Relation '{}' column {} (Bool): value {} not in {{0, 1}}",
                            relation, col_idx, val,
                        ))),
                    }
                }
                ScalarType::Symbol => {
                    let v = u32::try_from(val).map_err(|_| PyValueError::new_err(format!(
                        "Relation '{}' column {} (Symbol): value {} out of range [0, {}]",
                        relation, col_idx, val, u32::MAX,
                    )))?;
                    col.extend_from_slice(&v.to_le_bytes());
                }
                ScalarType::F32 => {
                    return Err(PyValueError::new_err(format!(
                        "Relation '{}' column {} (F32): float columns not supported in batch APIs",
                        relation, col_idx,
                    )));
                }
                ScalarType::F64 => {
                    return Err(PyValueError::new_err(format!(
                        "Relation '{}' column {} (F64): float columns not supported in batch APIs",
                        relation, col_idx,
                    )));
                }
            }
        }
    }

    Ok(columns)
}
```

**Step 2: Build to verify it compiles**

Run: `cargo build -p pyxlog --release 2>&1 | tail -5`
Expected: Compiles (helper is unused so far, but Rust allows dead code in non-library targets; add `#[allow(dead_code)]` if needed).

**Step 3: Commit helper**

```bash
git add crates/pyxlog/src/lib.rs
git commit -m "feat(pyxlog): add pack_i64_columns_typed helper for schema-aware batch upload"
```

---

### Task 3: Wire helper into `batch_fact_membership`

**Files:**
- Modify: `crates/pyxlog/src/lib.rs:4425-4440` — replace u32 transpose + upload block

**Step 1: Replace the upload block in `batch_fact_membership`**

Replace lines 4425-4440 (the `// Transpose facts into columnar layout` through `create_buffer_from_u32_columns` block) with:

```rust
        // Schema-aware typed upload
        let col_bytes = pack_i64_columns_typed(relation, &facts, buf.schema())?;
        let col_slices: Vec<&[u8]> = col_bytes.iter().map(|c| c.as_slice()).collect();
        let query_buf = self.provider
            .create_buffer_from_slices(
                &col_slices,
                buf.schema().clone(),
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
```

Also remove the now-redundant arity check at lines 4416-4423 (the helper does its own).

**Step 2: Build**

Run: `cargo build -p pyxlog --release 2>&1 | tail -5`
Expected: Compiles successfully.

**Step 3: Run positive tests**

Run: `LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/test_typed_batch_upload.py -v --timeout=120`
Expected: All `test_membership_*` tests PASS.

**Step 4: Run existing d2h gate tests**

Run: `LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/test_ilp_d2h_gate.py -v --timeout=120`
Expected: All PASS (regression check).

**Step 5: Commit**

```bash
git add crates/pyxlog/src/lib.rs
git commit -m "fix(pyxlog): use typed packing in batch_fact_membership (no more as u32)"
```

---

### Task 4: Wire helper into `batch_tagged_credit`

**Files:**
- Modify: `crates/pyxlog/src/lib.rs:4502-4514` — replace u32 transpose + upload block

**Step 1: Replace the upload block in `batch_tagged_credit`**

Replace lines 4502-4514 (the `// Upload query facts to GPU once` through `create_buffer_from_u32_columns` block) with:

```rust
        // Schema-aware typed upload
        let schema = first_buf.schema().clone();
        let col_bytes = pack_i64_columns_typed(relation, &facts, &schema)?;
        let col_slices: Vec<&[u8]> = col_bytes.iter().map(|c| c.as_slice()).collect();
        let query_buf = self.provider
            .create_buffer_from_slices(&col_slices, schema)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
```

Also remove the now-redundant arity check at lines 4492-4500 (the helper does its own).

**Step 2: Build and test**

Run: `cargo build -p pyxlog --release && LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/test_ilp_d2h_gate.py python/tests/test_typed_batch_upload.py -v --timeout=120`
Expected: All PASS.

**Step 3: Commit**

```bash
git add crates/pyxlog/src/lib.rs
git commit -m "fix(pyxlog): use typed packing in batch_tagged_credit (no more as u32)"
```

---

### Task 5: Add negative tests (error cases)

**Files:**
- Modify: `python/tests/test_typed_batch_upload.py` — add error tests

**Step 1: Add negative tests**

Append to `test_typed_batch_upload.py`:

```python
def test_membership_u32_overflow_error():
    """U32 column rejects values > u32::MAX."""
    prog = _compile_typed("""
        pred data(u32, u32).
        data(1, 2).
        learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """)
    with pytest.raises(ValueError, match=r"column 0 \(U32\).*out of range"):
        prog.batch_fact_membership("data", [[2**33, 1]])


def test_membership_u32_negative_error():
    """U32 column rejects negative values."""
    prog = _compile_typed("""
        pred data(u32, u32).
        data(1, 2).
        learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """)
    with pytest.raises(ValueError, match=r"column 0 \(U32\).*out of range"):
        prog.batch_fact_membership("data", [[-1, 1]])


def test_membership_u64_negative_error():
    """U64 column rejects negative values."""
    prog = _compile_typed("""
        pred data(u64).
        data(1).
        learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """)
    with pytest.raises(ValueError, match=r"column 0 \(U64\).*out of range"):
        prog.batch_fact_membership("data", [[-5]])


def test_membership_bool_invalid_error():
    """Bool column rejects values not in {0, 1}."""
    prog = _compile_typed("""
        pred data(u32, bool).
        data(1, true).
        learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """)
    with pytest.raises(ValueError, match=r"column 1 \(Bool\).*not in"):
        prog.batch_fact_membership("data", [[1, 2]])


def test_membership_f64_rejected():
    """F64 columns are explicitly rejected in batch APIs."""
    prog = _compile_typed("""
        pred data(f64).
        data(1.5).
        learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """)
    with pytest.raises(ValueError, match=r"column 0 \(F64\).*not supported"):
        prog.batch_fact_membership("data", [[1]])
```

**Step 2: Run all typed tests**

Run: `LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/test_typed_batch_upload.py -v --timeout=120`
Expected: All PASS.

**Step 3: Commit**

```bash
git add python/tests/test_typed_batch_upload.py
git commit -m "test: add negative tests for typed batch upload error cases"
```

---

### Task 6: Full regression suite

**Files:** (none modified — verification only)

**Step 1: Run core test suites**

Run: `LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/test_ilp_trainer.py python/tests/test_ilp_sparse.py python/tests/test_ilp_reset.py python/tests/test_ilp_d2h_gate.py python/tests/test_ilp_performance.py python/tests/test_typed_batch_upload.py -v --timeout=300 -k "not slow"`
Expected: All PASS.

**Step 2: Run Rust workspace tests**

Run: `cargo test --workspace --all-targets --exclude pyxlog --release 2>&1 | tail -5`
Expected: All PASS.

**Step 3: Verify no remaining `as u32` in batch upload paths**

Run: `grep -n "val as u32" crates/pyxlog/src/lib.rs`
Expected: Only non-batch coercions remain (lines ~3957, ~4232, ~4469, ~4586, ~4602-3 — all index/loop-counter casts, not fact-value uploads).

**Step 4: Update CHANGELOG**

Add to `[Unreleased]` > `### Fixed`:
```markdown
- **Typed batch upload**: `batch_fact_membership` and `batch_tagged_credit` now use
  schema-aware typed packing for all column types (I32, I64, U64, Bool, Symbol).
  Previously, all values were blindly cast to `u32`, corrupting non-U32 columns.
  F32/F64 columns are explicitly rejected with a clear error message.
```

**Step 5: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: record typed batch upload fix in changelog"
```
