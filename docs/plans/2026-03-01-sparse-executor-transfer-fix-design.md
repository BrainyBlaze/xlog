# Design: Sparse Executor + Transfer Elimination Fix

**Date:** 2026-03-01
**Status:** Approved
**Addresses:** Two implementation gaps found in dILP beta audit

## Problem Statement

The dILP beta documentation claims sparse-by-default mask application and GPU-resident training
hot loop, but the code has two structural gaps:

1. **SparseMaskBackend still materializes dense N^3** — `backend.py:111-143` builds N^3 Python
   tensors and calls `set_rule_mask()`. The Rust `set_rule_mask_sparse()` exists but is test-only.
   Even the Rust sparse path (`insert_mask_from_sparse`) builds dense N^3 host arrays internally.

2. **Row-count cache unwired** — `CudaBuffer.cached_row_count` field exists with getter/setter
   but all 4 call sites (`provider/mod.rs (pre-Wave-2 line 6970)`, `ilp_registry.rs:17`, `executor.rs:2961`,
   `executor.rs:3036`) do synchronous D2H transfers. `buffer_from_columns` does not populate
   the cache at construction time.

## Design

### 1. Row-Count Cache Wiring

**Scope:** `provider.rs`, `ilp_registry.rs`, `executor.rs`, `memory.rs`

Each row-count read site gains a cache-first pattern:

```rust
fn device_row_count(&self, buffer: &CudaBuffer) -> Result<usize> {
    if let Some(n) = buffer.cached_row_count() {
        return Ok(n as usize);
    }
    let mut host_rows = [0u32];
    self.device.inner()
        .dtoh_sync_copy_into(buffer.num_rows_device(), &mut host_rows)
        .map_err(|e| XlogError::Kernel(format!("Failed to read row count: {}", e)))?;
    let n = host_rows[0];
    buffer.set_cached_row_count_if_unset(n);
    Ok(n as usize)
}
```

Apply to all 4 call sites:
- `CudaKernelProvider::device_row_count()` (provider/mod.rs (pre-Wave-2 line 6970))
- `ilp_registry::read_device_row_count()` (ilp_registry.rs:9-20)
- `Executor::buffer_row_count()` (executor.rs:2961)
- Standalone `ilp_row_count()` (executor.rs:3036)

Additionally, `buffer_from_columns()` (provider/mod.rs (pre-Wave-2 line 7159)) already knows the row count — switch to
`CudaBuffer::from_columns_with_host_count()` to pre-populate the cache at construction.

**Safety:** Cache is write-once (CAS from u32::MAX sentinel). Mutable buffers rebuilt each
iteration don't use cached values.

### 2. Sparse-Native Mask Storage (IlpRegistry)

**Scope:** `ilp_registry.rs`

Replace single struct with enum:

```rust
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

- `insert_mask()` stores `Dense` variant (backward-compatible, used by dense backend and
  convergence verification)
- `insert_mask_from_sparse()` refactored: does top-k ranking, stores `Sparse` variant directly.
  No dense host array materialization, no GPU upload for mask buffers.
- `get_mask()` returns `&IlpMask`, callers pattern-match

### 3. Sparse Path in execute_tensor_masked_join

**Scope:** `executor.rs`

When `IlpMask::Sparse`, skip the `extract_active_rule_indices` GPU kernel entirely:

```rust
let active_rules: &[(u32, u32, u32)] = match ilp_mask {
    IlpMask::Dense { hard, soft, .. } => {
        dense_buf = self.provider.extract_active_rule_indices(hard, soft, schema_size, max_active_rules)?;
        &dense_buf
    }
    IlpMask::Sparse { active_entries, schema_size: _ } => {
        let limit = max_active_rules.min(active_entries.len());
        &active_entries[..limit]
    }
};
```

Hardening:
- Enforce `max_active_rules` defensively via slice, even though upstream already truncates
- Iterate by reference (no clone)

**Eliminated for sparse path:** N^3 device allocation (5 buffers), kernel launch, 5 D2H transfers.

### 4. DLPack-Native set_rule_mask_sparse API

**Scope:** `lib.rs` (PyO3), `backend.py`

Change `set_rule_mask_sparse` to accept a DLPack capsule for soft_probs instead of `Vec<f64>`:

```rust
#[pyo3(signature = (name, candidate_ids, soft_probs_dlpack, budget, allow_recursive=false))]
pub fn set_rule_mask_sparse(
    &mut self,
    name: String,
    candidate_ids: Vec<u32>,
    soft_probs_dlpack: &Bound<'_, PyAny>,  // DLPack capsule
    budget: usize,
    allow_recursive: bool,
) -> PyResult<()>
```

Rust downloads the C-element f64 tensor internally (O(C) bytes, C < 100), performs top-k ranking
on host, and stores `IlpMask::Sparse` directly. This eliminates `cand_probs.detach().cpu().tolist()`
from the Python hot loop.

**Top-k ownership:** Rust owns top-k. Python passes full candidate probs + budget.

### 5. SparseMaskBackend Rewrite

**Scope:** `backend.py`

```python
class SparseMaskBackend:
    def apply_mask(self, prog, mask_name, W, tau, budget, candidates, n,
                   allow_recursive=False):
        cand_probs = F.gumbel_softmax(W, tau=tau, hard=False, dim=0)
        candidate_ids = list(range(len(candidates)))
        prog.set_rule_mask_sparse(
            mask_name, candidate_ids, cand_probs, budget, allow_recursive,
        )
        return cand_probs
```

No N^3 tensor. No `.cpu().tolist()`. DLPack capsule passed directly from CUDA tensor.

`_check_convergence()` (trainer.py:440-444) still uses single-cell dense mask for argmax-only
verification — acceptable (one-shot, not hot-loop).

### 6. Guard Tests

1. **`test_sparse_backend_does_not_call_set_rule_mask`** — monkey-patch `set_rule_mask` to raise,
   run `train_only()` with sparse backend, assert no raise.
2. **`test_sparse_executor_skips_dense_kernel`** — verify D2H transfer count in sparse path
   does not include extract_active_rule_indices transfers.
3. **Reliability 20/20 re-run** — unchanged test, re-verify convergence with new code path.

## Data Flow Comparison

### Before (Python-side N^3)

```
Python: W(C) → Gumbel-Softmax → build N^3 tensor (Python) → flatten → DLPack → set_rule_mask()
Rust:   DLPack import → IlpMask::Dense { hard: N^3 CudaBuffer, soft: N^3 CudaBuffer }
Exec:   extract_active_rule_indices kernel (N^3 scan) → 5x D2H → Vec<(i,j,k)> → joins
```

### After (Sparse)

```
Python: W(C) → Gumbel-Softmax → DLPack capsule(C) → set_rule_mask_sparse()
Rust:   DLPack download (C floats) → top-k ranking → IlpMask::Sparse { entries: Vec<(i,j,k)> }
Exec:   entries already known → skip kernel → &[(i,j,k)] → joins
```

## Risk Assessment

| Change | Risk | Mitigation |
|--------|------|------------|
| Row-count cache wiring | Low | Write-once CAS, no mutation hazard |
| IlpMask enum | Low | Dense path unchanged, backward-compatible |
| Sparse executor branch | Medium | Existing test_ilp_sparse.py validates parity |
| DLPack API change | Medium | Tests must update, but only 1 caller (SparseMaskBackend) |
| SparseMaskBackend rewrite | Medium | 20/20 reliability gate as final validation |

## Implementation Order

1. Row-count cache wiring (independent, zero risk to existing tests)
2. IlpMask enum + sparse storage in IlpRegistry
3. Sparse path in execute_tensor_masked_join
4. DLPack-native set_rule_mask_sparse API
5. SparseMaskBackend rewrite + convergence verification removal of N^3
6. Guard tests
7. Reliability 20/20 re-run
