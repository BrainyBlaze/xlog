# v0.5.0 Phase 1 Implementation Plan — Zero Data-Plane D2H + Artifact Schema

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate the last data-plane D2H transfer from `compute_ilp_loss_grad_gpu` (P0), add artifact schema migration with telemetry persistence (P1), and validate release gates (P4).

**Scope:** This plan covers P0, P1, and P4 of the v0.5.0 design. P2a (term embeddings), P2b (extended training controls), and P3 (incremental verifier) are separate follow-up plans — they depend on P0 completing first (see design doc sequencing).

**Architecture:** Replace the host-side sentinel-filter merge (D2H + H2D round-trip) with a two-pass GPU-only COO merge algorithm: pass 1 counts NNZ per task into a device array, pass 2 fills COO at pre-computed offsets. A new `coo_chunk_budget` parameter bounds per-chunk temp allocations (masks, prefix sums, scratch) while allowing the final exact-NNZ output buffer to exceed it. Note: per-task `d_mask` buffers are still fully retained before the chunking branch (`lib.rs:4218`); "bounded-memory" refers to the COO merge scratch only, not the total compute-path footprint. A `dtoh_scalar_untracked` helper reads the 4-byte total_nnz metadata without touching D2H accounting.

**Tech Stack:** Rust (PyO3), CUDA C, Python (pytest)

**Design doc:** `docs/plans/2026-03-05-v050-execution-design.md`

---

## Task Overview

| Task | Tier | Description |
|------|------|-------------|
| 1 | P0 | Rename `coo_memory_cap` → `coo_chunk_budget` |
| 2 | P0 | Add `count_mask_into_slot` provider method |
| 3 | P0 | Add `dtoh_scalar_untracked` provider helper |
| 4 | P0 | Implement two-pass GPU-only chunk merge |
| 5 | P0 | Update strict D2H test + add chunked-passes-strict test |
| 6 | P1 | Artifact schema migration beta-v1 → beta-v2 |
| 7 | P1 | Bounded telemetry persistence |
| 8 | P4 | Release gate validation |

---

### Task 1: Rename `coo_memory_cap` → `coo_chunk_budget`

**Files:**
- Modify: `crates/pyxlog/src/lib.rs:3987-3994` (struct field + doc)
- Modify: `crates/pyxlog/src/lib.rs:3964` (default value in `compile`)
- Modify: `crates/pyxlog/src/lib.rs:4013-4018` (setter method)
- Modify: `crates/pyxlog/src/lib.rs:4244-4254` (usage in chunking check)
- Modify: `crates/pyxlog/src/lib.rs:4361` (usage in chunk size calc)
- Modify: `python/tests/test_ilp_credit_gpu.py:512` (test uses `set_coo_memory_cap`)

**Step 1: Rename struct field and update doc comment**

In `crates/pyxlog/src/lib.rs`, change the field and its doc comment:

```rust
// At line 3987-3990, change:
    /// Maximum COO buffer size in bytes. If the COO allocation for a loss
    /// computation would exceed this cap, candidates are processed in chunks.
    /// Default: 16 MB.
    coo_memory_cap: u64,
// To:
    /// Maximum bytes for per-chunk temp allocations (masks, prefix sums,
    /// chunk-local COO scratch). The final merged COO buffer is exact-NNZ
    /// sized and may exceed this budget. Default: 16 MB.
    coo_chunk_budget: u64,
```

**Step 2: Update the default value assignment**

In `crates/pyxlog/src/lib.rs` line 3964, change:
```rust
coo_memory_cap: 16 * 1024 * 1024,
// To:
coo_chunk_budget: 16 * 1024 * 1024,
```

**Step 3: Add new setter, keep old as deprecated alias**

In `crates/pyxlog/src/lib.rs`, replace the `set_coo_memory_cap` method (lines 4013-4018):

```rust
    /// Set the per-chunk temp allocation budget in bytes. The final merged
    /// COO buffer is exact-NNZ sized and may exceed this budget. Default: 16 MB.
    pub fn set_coo_chunk_budget(&mut self, bytes: u64) {
        self.coo_chunk_budget = bytes;
    }

    /// Deprecated: use `set_coo_chunk_budget`. Kept for one release cycle.
    #[allow(deprecated)]
    pub fn set_coo_memory_cap(&mut self, bytes: u64) {
        self.coo_chunk_budget = bytes;
    }
```

**Step 4: Update all usages of `self.coo_memory_cap` → `self.coo_chunk_budget`**

In `crates/pyxlog/src/lib.rs`:

- Line 4247: `let coo_bytes = (upper_bound as u64) * 8;` → keep
- Line 4247: `self.coo_memory_cap` → `self.coo_chunk_budget`
- Line 4253: `self.coo_memory_cap` → `self.coo_chunk_budget`
- Line 4361: `self.coo_memory_cap` → `self.coo_chunk_budget`

Use find-and-replace across the file: `self.coo_memory_cap` → `self.coo_chunk_budget`.

**Step 5: Verify compile**

Run: `cargo build -p pyxlog --release 2>&1 | tail -5`
Expected: Compiles successfully (no errors)

**Step 6: Commit**

```bash
git add crates/pyxlog/src/lib.rs
git commit -m "refactor(ilp): rename coo_memory_cap → coo_chunk_budget

New semantics: controls per-chunk temp allocation budget, not a hard
ceiling on the final COO buffer. set_coo_memory_cap kept as deprecated
alias for one release cycle."
```

---

### Task 2: Add `count_mask_into_slot` provider method

**Files:**
- Modify: `crates/xlog-cuda/src/provider/mod.rs:4484-4525` (add method near `count_mask_device`)

**Context:** The existing `count_mask_device` (provider/mod.rs (pre-Wave-2 line 4484)) allocates a fresh 1-element `TrackedCudaSlice<u32>` per call, zero-inits it, launches the `count_mask` kernel, syncs, and returns the slice. For pass 1 of the two-pass merge, we need to call this N times (once per task) but write results into a single pre-allocated `task_counts` array. The new method avoids N allocations.

The `count_mask` kernel (scan.cu:105) writes via `atomicAdd(count, block_count)` — the `count` pointer is just a `uint32_t*`. We can point it at `task_counts[idx]` by offsetting the pointer. The slot must be pre-zeroed by the caller.

**Step 1: Write the failing test**

We test via Rust unit tests. However, this is a provider method that requires GPU hardware. We'll test it indirectly through the Python integration test in Task 5. For now, add the method and verify compilation.

**Step 2: Add `count_mask_into_slot` method**

In `crates/xlog-cuda/src/provider/mod.rs`, immediately after `count_mask_device` (after line 4525), add:

```rust
    /// Count 1-bits in `mask[0..n]` and write the result into
    /// `task_counts[slot_idx]` via the existing `count_mask` kernel.
    ///
    /// The caller MUST ensure `task_counts[slot_idx]` is zero before
    /// calling (e.g. by zeroing the whole array once).
    ///
    /// This avoids allocating a fresh 1-element device buffer per call,
    /// which matters when iterating over hundreds of tasks.
    pub fn count_mask_into_slot(
        &self,
        mask: &crate::memory::TrackedCudaSlice<u8>,
        n: u32,
        task_counts: &mut crate::memory::TrackedCudaSlice<u32>,
        slot_idx: usize,
    ) -> Result<()> {
        if n == 0 {
            // Slot is already zero (caller pre-zeroed); nothing to do.
            return Ok(());
        }
        if slot_idx >= task_counts.len() {
            return Err(XlogError::Kernel(format!(
                "count_mask_into_slot: slot_idx={} >= len={}",
                slot_idx,
                task_counts.len()
            )));
        }

        let device = self.device.inner();
        let block_size = 256u32;
        let grid_size = (n + block_size - 1) / block_size;

        let count_fn = device
            .get_func(SCAN_MODULE, scan_kernels::COUNT_MASK)
            .ok_or_else(|| {
                XlogError::Kernel("count_mask kernel not found".to_string())
            })?;

        // Get a mutable sub-slice pointing at task_counts[slot_idx..slot_idx+1].
        let mut slot = task_counts.try_slice_mut(slot_idx..slot_idx + 1)
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                    "count_mask_into_slot: slice failed at idx={}",
                    slot_idx
                ))
            })?;

        // SAFETY: count_mask(mask, n, count) — writes atomicAdd into count ptr.
        // The slot was pre-zeroed by the caller.
        unsafe {
            count_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (mask, n, &mut slot),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("count_mask_into_slot kernel failed: {}", e)))?;

        Ok(())
    }
```

**Step 3: Verify compile**

Run: `cargo build -p xlog-cuda --release 2>&1 | tail -5`
Expected: Compiles successfully

**Step 4: Commit**

```bash
git add crates/xlog-cuda/src/provider/mod.rs
git commit -m "feat(cuda): add count_mask_into_slot provider method

Writes count_mask result directly into a pre-allocated task_counts
array slot, avoiding per-call 1-element device allocation churn.
Reuses existing count_mask kernel (atomicAdd to output pointer)."
```

---

### Task 3: Add `dtoh_scalar_untracked` provider helper

**Files:**
- Modify: `crates/xlog-cuda/src/provider/mod.rs` (add method near transfer tracker section, ~line 1615)

**Context:** The two-pass merge needs to read one `u32` (total_nnz) from device to host after pass 1's exclusive scan. This is 4 bytes of metadata, not data-plane. The helper reads via `device().inner()` without touching `record_dtoh`, making the "metadata ≠ data-plane" contract explicit. Existing strict D2H tests (`dtoh_calls == 0`, `dtoh_bytes == 0`) continue to pass.

**Step 1: Add `dtoh_scalar_untracked` method**

In `crates/xlog-cuda/src/provider/mod.rs`, after `dtoh_sync_copy_into_tracked` (after line ~1628), add:

```rust
    /// Read a single scalar from device to host WITHOUT updating the
    /// D2H transfer tracker. Use ONLY for metadata reads (e.g. total_nnz
    /// after an exclusive scan), never for data-plane transfers.
    ///
    /// This makes the "metadata ≠ data-plane" contract explicit and
    /// auditable: callers that bypass tracking must call this method
    /// (which is grep-able) rather than reaching for device().inner().
    pub fn dtoh_scalar_untracked<T: cudarc::driver::DeviceRepr + Default + Copy>(
        &self,
        src: &crate::memory::TrackedCudaSlice<T>,
        index: usize,
    ) -> Result<T> {
        if index >= src.len() {
            return Err(XlogError::Kernel(format!(
                "dtoh_scalar_untracked: index={} >= len={}",
                index,
                src.len()
            )));
        }
        let slice = src.try_slice(index..index + 1)
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                    "dtoh_scalar_untracked: slice failed at index={}",
                    index
                ))
            })?;
        let mut buf = [T::default()];
        self.device
            .inner()
            .dtoh_sync_copy_into(&slice, &mut buf)
            .map_err(|e| XlogError::Kernel(format!(
                "dtoh_scalar_untracked: copy failed: {}", e
            )))?;
        Ok(buf[0])
    }
```

**Step 2: Verify compile**

Run: `cargo build -p xlog-cuda --release 2>&1 | tail -5`
Expected: Compiles successfully

**Step 3: Commit**

```bash
git add crates/xlog-cuda/src/provider/mod.rs
git commit -m "feat(cuda): add dtoh_scalar_untracked for metadata-only reads

Explicit helper for reading single scalars from device without
incrementing the D2H transfer tracker. Documents the metadata vs
data-plane contract. Used by two-pass chunk merge for total_nnz."
```

---

### Task 4: Implement two-pass GPU-only chunk merge

**Files:**
- Modify: `crates/pyxlog/src/lib.rs:4249-4255` (remove strict rejection of chunking)
- Modify: `crates/pyxlog/src/lib.rs:4353-4514` (replace chunked path)

**Context:** This is the core P0 change. The current chunked path (lib.rs:4353-4514) does D2H of COO arrays per chunk, host-side sentinel filtering, then H2D upload of merged result. The new algorithm:

- **Pass 1**: For each chunk of tasks, call `count_mask_into_slot(mask, num_query, task_counts, task_idx)`. After all chunks: `task_counts` is populated on device. Compute `exclusive_scan_u32_inplace(task_counts, num_tasks+1) → task_offsets`. Read `total_nnz = task_offsets[num_tasks]` via `dtoh_scalar_untracked`.
- **Allocation**: Allocate `d_global_facts[total_nnz]` + `d_global_cands[total_nnz]` (exact-NNZ, no sentinels).
- **Pass 2**: Re-iterate tasks. For each: `scan_u8_mask_device` + `ilp_coo_fill_from_mask_launch` with `offset_idx = task_idx` into the global `task_offsets` array.

**Step 1: Remove strict rejection of chunking**

In `crates/pyxlog/src/lib.rs` lines 4249-4255, remove or comment out the strict check:

```rust
// REMOVE these lines:
        if needs_chunking && self.strict_zero_dtoh {
            return Err(PyRuntimeError::new_err(format!(
                "strict_zero_dtoh: COO allocation {} bytes exceeds cap {} bytes; \
                 chunked fallback would require D2H transfers",
                coo_bytes, self.coo_chunk_budget
            )));
        }
```

The chunked path is now GPU-only, so strict mode no longer needs to reject it.

**Step 2: Replace the chunked path (lines 4353-4514)**

Replace the entire `else` branch (from `} else {` at line 4353 through `(d_coo_facts, d_coo_cands, actual_nnz)` at line 4514) with the two-pass algorithm:

```rust
        } else {
            // ── Phase C (chunked): Two-pass GPU-only bounded-memory merge ──
            //
            // Pass 1: Count NNZ per task on device (bounded temp per chunk).
            // Pass 2: Fill COO at pre-computed offsets (bounded temp per chunk).
            // The final COO buffer is exact-NNZ sized and may exceed
            // coo_chunk_budget — this is safe because actual NNZ << upper_bound.

            let max_queries_per_chunk = (self.coo_chunk_budget / 8).max(1) as u32;

            // Allocate task_counts array (num_tasks + 1 for exclusive scan).
            let tc_len = num_tasks + 1;
            let mut d_task_counts = self
                .provider
                .memory()
                .alloc::<u32>(tc_len)
                .map_err(|e| PyRuntimeError::new_err(format!("alloc task_counts: {}", e)))?;
            // Zero-init the entire array (counts default to 0, last slot = 0 for scan).
            {
                let zeros = vec![0u32; tc_len];
                self.provider
                    .device()
                    .inner()
                    .htod_sync_copy_into(&zeros, &mut d_task_counts)
                    .map_err(|e| PyRuntimeError::new_err(format!("zero task_counts: {}", e)))?;
            }

            // ── Pass 1: Count NNZ per task ──
            {
                let mut chunk_start = 0usize;
                while chunk_start < tasks.len() {
                    let mut chunk_end = chunk_start;
                    let mut chunk_sum = 0u32;
                    while chunk_end < tasks.len() {
                        let nq = tasks[chunk_end].num_query;
                        if chunk_sum + nq > max_queries_per_chunk && chunk_sum > 0 {
                            break;
                        }
                        chunk_sum += nq;
                        chunk_end += 1;
                    }
                    if chunk_end == chunk_start {
                        chunk_end = chunk_start + 1;
                    }

                    for task_idx in chunk_start..chunk_end {
                        let task = &tasks[task_idx];
                        self.provider
                            .count_mask_into_slot(
                                &task.d_mask,
                                task.num_query,
                                &mut d_task_counts,
                                task_idx,
                            )
                            .map_err(|e| PyRuntimeError::new_err(format!(
                                "count_mask_into_slot: {}", e
                            )))?;
                    }

                    chunk_start = chunk_end;
                }
            }

            // Synchronize to ensure all count kernels have completed.
            self.provider.device().synchronize()
                .map_err(|e| PyRuntimeError::new_err(format!("sync pass1: {}", e)))?;

            // Compute per-task write offsets via exclusive scan.
            // d_task_counts[0..num_tasks] has counts; after scan,
            // d_task_counts[i] = sum of counts[0..i] = write offset for task i.
            // d_task_counts[num_tasks] = total_nnz.
            self.provider
                .exclusive_scan_u32_inplace(&mut d_task_counts, tc_len as u32)
                .map_err(|e| PyRuntimeError::new_err(format!("scan task_counts: {}", e)))?;

            // Read total_nnz from the last element (metadata-only, untracked).
            let total_nnz: u32 = self.provider
                .dtoh_scalar_untracked(&d_task_counts, num_tasks)
                .map_err(|e| PyRuntimeError::new_err(format!("read total_nnz: {}", e)))?;

            if total_nnz == 0 {
                return self.build_loss_grad_empty_coo(
                    py,
                    &is_positive_host,
                    num_facts,
                    num_cands,
                    is_f64,
                );
            }

            // Allocate exact-NNZ global COO buffers (may exceed coo_chunk_budget).
            let mut d_coo_facts = self
                .provider
                .memory()
                .alloc::<u32>(total_nnz as usize)
                .map_err(|e| PyRuntimeError::new_err(format!("alloc global coo_facts: {}", e)))?;
            let mut d_coo_cands = self
                .provider
                .memory()
                .alloc::<u32>(total_nnz as usize)
                .map_err(|e| PyRuntimeError::new_err(format!("alloc global coo_cands: {}", e)))?;

            // ── Pass 2: Fill COO at pre-computed offsets ──
            // d_task_counts now contains offsets (after exclusive scan).
            // ilp_coo_fill_from_mask_launch reads d_offsets[offset_idx] for the
            // write base position. We pass d_task_counts as d_offsets with
            // offset_idx = task_idx.
            {
                let mut chunk_start = 0usize;
                while chunk_start < tasks.len() {
                    let mut chunk_end = chunk_start;
                    let mut chunk_sum = 0u32;
                    while chunk_end < tasks.len() {
                        let nq = tasks[chunk_end].num_query;
                        if chunk_sum + nq > max_queries_per_chunk && chunk_sum > 0 {
                            break;
                        }
                        chunk_sum += nq;
                        chunk_end += 1;
                    }
                    if chunk_end == chunk_start {
                        chunk_end = chunk_start + 1;
                    }

                    for task_idx in chunk_start..chunk_end {
                        let task = &tasks[task_idx];

                        let d_prefix = self
                            .provider
                            .scan_u8_mask_device(&task.d_mask, task.num_query)
                            .map_err(|e| PyRuntimeError::new_err(format!("scan mask: {}", e)))?;

                        self.provider
                            .ilp_coo_fill_from_mask_launch(
                                &task.d_mask,
                                &d_prefix,
                                &fact_indices_buffers[task.fact_indices_idx],
                                task_idx as u32,
                                task.cidx,
                                task.num_query,
                                &d_task_counts,  // offsets after scan
                                &mut d_coo_facts,
                                &mut d_coo_cands,
                            )
                            .map_err(|e| PyRuntimeError::new_err(format!("coo fill: {}", e)))?;
                        // d_prefix dropped here (bounded temp).
                    }

                    chunk_start = chunk_end;
                }
            }

            (d_coo_facts, d_coo_cands, total_nnz)
        };
```

**Step 3: Update Phase D comment**

After line 4515 (now after the new closing `};`), update the comment at `Phase D`:

```rust
        // ── Phase D: Sort COO + device-side CSR build ──
        //
        // Sort all entries by fact index. In the non-chunked path, sentinels
        // (fact = num_facts) sort to the end and are ignored by the histogram
        // kernel's f < num_facts guard. In the chunked path, the COO is
        // exact-NNZ (no sentinels) from the two-pass GPU-only merge.
```

**Step 4: Verify compile**

Run: `cargo build -p pyxlog --release 2>&1 | tail -5`
Expected: Compiles successfully

**Step 5: Run existing tests (expect pass — non-chunked path unchanged)**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_credit_gpu.py -v -k "not strict_zero_dtoh_rejects" 2>&1 | tail -20`
Expected: All tests pass except the strict rejection test (which we'll update in Task 5)

**Step 6: Commit**

```bash
git add crates/pyxlog/src/lib.rs
git commit -m "feat(ilp): two-pass GPU-only chunk merge (zero data-plane D2H)

Replace host-side sentinel-filter merge with two-pass streaming:
- Pass 1: count_mask_into_slot per task → exclusive scan → total_nnz
- Pass 2: fill COO at pre-computed per-task offsets
- 4-byte metadata read (total_nnz) via dtoh_scalar_untracked (untracked)

Chunked COO merge now has zero data-plane D2H transfers.
strict_zero_dtoh no longer needs to reject chunking."
```

---

### Task 5: Update strict D2H test + add chunked-passes-strict test

**Files:**
- Modify: `python/tests/test_ilp_credit_gpu.py:486-526`

**Step 1: Replace `test_strict_zero_dtoh_rejects_chunking`**

The old test verifies that strict mode raises when chunking is needed. Now chunking is GPU-only and should succeed under strict mode. Replace the test:

```python
def test_strict_zero_dtoh_chunked_passes():
    """Chunked path now GPU-only — must pass under strict_zero_dtoh."""
    prog = pyxlog.IlpProgramFactory.compile("""
        pred edge(u32, u32). pred reach(u32, u32).
        edge(1, 2). edge(2, 3).
        learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """, device=0, memory_mb=64)
    prog.evaluate()

    N = prog.ilp_schema_size()
    names = prog.ilp_relation_names()
    edge_idx = names.index("edge")
    reach_idx = names.index("reach")

    cand = (edge_idx, edge_idx, reach_idx)
    device = torch.device("cuda:0")
    mask = torch.zeros(N ** 3, device=device, dtype=torch.float32)
    mask[edge_idx * N * N + edge_idx * N + reach_idx] = 1.0
    prog.set_rule_mask("W", mask, mask, N)
    prog.evaluate()

    prog.set_candidate_map([cand])
    cand_probs = torch.tensor([0.5], device=device, dtype=torch.float32)

    # Enable strict mode + tiny budget to force chunking
    prog.set_strict_zero_dtoh(True)
    prog.set_coo_chunk_budget(1)  # Force chunking

    # Reset stats, run loss/grad, verify zero tracked D2H
    prog.reset_host_transfer_stats()
    loss_dl, grad_dl = prog.compute_ilp_loss_grad_gpu(
        [("reach", [1, 3])], [("reach", [1, 2])], cand_probs
    )
    loss = torch.from_dlpack(loss_dl)
    assert torch.isfinite(loss).item(), f"Loss not finite: {loss.item()}"

    stats = prog.host_transfer_stats()
    assert stats['dtoh_calls'] == 0, (
        f"Chunked path caused {stats['dtoh_calls']} tracked D2H calls; expected 0"
    )
    assert stats['dtoh_bytes'] == 0, (
        f"Chunked path transferred {stats['dtoh_bytes']} tracked D2H bytes; expected 0"
    )


def test_deprecated_set_coo_memory_cap():
    """Deprecated alias still works."""
    prog = pyxlog.IlpProgramFactory.compile("""
        pred edge(u32, u32). pred reach(u32, u32).
        edge(1, 2). edge(2, 3).
        learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """, device=0, memory_mb=64)
    # Should not raise — alias forwards to set_coo_chunk_budget
    prog.set_coo_memory_cap(1024)
```

**Step 2: Run the new tests**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_credit_gpu.py::test_strict_zero_dtoh_chunked_passes -v 2>&1 | tail -10`
Expected: PASS

Run: `.venv/bin/python -m pytest python/tests/test_ilp_credit_gpu.py::test_deprecated_set_coo_memory_cap -v 2>&1 | tail -10`
Expected: PASS

**Step 3: Run the full credit GPU test suite**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_credit_gpu.py -v 2>&1 | tail -20`
Expected: All tests pass

**Step 4: Commit**

```bash
git add python/tests/test_ilp_credit_gpu.py
git commit -m "test(ilp): verify chunked path passes under strict_zero_dtoh

Replace test_strict_zero_dtoh_rejects_chunking with
test_strict_zero_dtoh_chunked_passes: tiny budget forces chunking,
strict mode enabled, verify zero tracked D2H calls/bytes.

Add test_deprecated_set_coo_memory_cap for backward compat alias."
```

---

### Task 6: Artifact schema migration beta-v1 → beta-v2

**Files:**
- Modify: `crates/pyxlog/python/pyxlog/ilp/types.py:133-240`
- Modify: `python/tests/test_ilp_artifact.py`

**Context:** Currently `save()` writes `"schema_version": "beta-v1"` and `load()` rejects anything except `beta-v1`. We need v2 to support a `telemetry_snapshot` field (Task 7). This task adds schema versioning: `save()` writes `beta-v2`, `load()` accepts both v1 and v2.

**Step 1: Write the failing test**

Add to `python/tests/test_ilp_artifact.py`:

```python
def test_schema_v1_loads_in_v2_code(tmp_path):
    """A beta-v1 artifact must load cleanly in v2 code."""
    import json
    v1_data = {
        "schema_version": "beta-v1",
        "discovered_rule": "reach(X,Y) :- edge(X,Z), edge(Z,Y).",
        "candidate_map": [
            {"id": 0, "i": 0, "j": 0, "k": 1,
             "left_name": "edge", "right_name": "edge", "head_name": "reach"}
        ],
        "logits": [1.5],
        "soft_probs": [0.82],
        "selected_hard": [0],
        "metadata": {
            "pyxlog_version": "0.4.0",
            "cuda_version": "12.6",
            "device_name": "test",
            "candidate_map_hash": "",
            "config_hash": "",
            "timestamp_utc": "2026-01-01T00:00:00Z",
        },
        "config_snapshot": None,
    }
    path = tmp_path / "v1_artifact.json"
    with open(path, "w") as f:
        json.dump(v1_data, f)

    from pyxlog.ilp.types import LearnedArtifact
    art = LearnedArtifact.load(path)
    assert art.discovered_rule == "reach(X,Y) :- edge(X,Z), edge(Z,Y)."
    assert len(art.candidate_map) == 1
    # Telemetry should be empty (v1 has no telemetry_snapshot)
    assert len(art.telemetry.steps) == 0


def test_schema_v2_roundtrip(tmp_path):
    """Save as v2, load back, verify fields."""
    from pyxlog.ilp.types import (
        LearnedArtifact, CandidateMapEntry, ArtifactMetadata
    )
    art = LearnedArtifact(
        candidate_map=[CandidateMapEntry(
            id=0, i=0, j=0, k=1,
            left_name="edge", right_name="edge", head_name="reach"
        )],
        logits=[2.0],
        soft_probs=[0.88],
        selected_hard=[0],
        discovered_rule="test rule",
        metadata=ArtifactMetadata(pyxlog_version="0.5.0"),
    )
    path = tmp_path / "v2_artifact.json"
    art.save(path)

    import json
    with open(path) as f:
        data = json.load(f)
    assert data["schema_version"] == "beta-v2"

    art2 = LearnedArtifact.load(path)
    assert art2.discovered_rule == "test rule"
    assert art2.logits == [2.0]
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_artifact.py::test_schema_v1_loads_in_v2_code -v 2>&1 | tail -10`
Expected: FAIL (load rejects v1 because save writes v1, but load's version check `!= "beta-v1"` may pass — the v2 roundtrip test should fail because save still writes v1)

Run: `.venv/bin/python -m pytest python/tests/test_ilp_artifact.py::test_schema_v2_roundtrip -v 2>&1 | tail -10`
Expected: FAIL (`assert data["schema_version"] == "beta-v2"`)

**Step 3: Update `save()` to write beta-v2**

In `crates/pyxlog/python/pyxlog/ilp/types.py`, line 156:

```python
# Change:
            "schema_version": "beta-v1",
# To:
            "schema_version": "beta-v2",
```

**Step 4: Update `load()` to accept both v1 and v2**

In `crates/pyxlog/python/pyxlog/ilp/types.py`, lines 200-204:

```python
# Change:
        stored_version = data.get("schema_version", "")
        if stored_version and stored_version != "beta-v1":
            raise ValueError(
                f"Incompatible schema version: {stored_version} (expected beta-v1)"
            )
# To:
        stored_version = data.get("schema_version", "")
        _SUPPORTED_VERSIONS = {"beta-v1", "beta-v2"}
        if stored_version and stored_version not in _SUPPORTED_VERSIONS:
            raise ValueError(
                f"Incompatible schema version: {stored_version} "
                f"(supported: {sorted(_SUPPORTED_VERSIONS)})"
            )
```

**Step 5: Run tests**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_artifact.py::test_schema_v1_loads_in_v2_code python/tests/test_ilp_artifact.py::test_schema_v2_roundtrip -v 2>&1 | tail -10`
Expected: PASS

**Step 6: Commit**

```bash
git add crates/pyxlog/python/pyxlog/ilp/types.py python/tests/test_ilp_artifact.py
git commit -m "feat(ilp): artifact schema migration beta-v1 → beta-v2

save() now writes beta-v2. load() accepts both v1 and v2.
v2 adds telemetry_snapshot field (next task). Backward compat tested."
```

---

### Task 7: Bounded telemetry persistence

**Files:**
- Modify: `crates/pyxlog/python/pyxlog/ilp/types.py` (TrainConfig, save, load)
- Modify: `python/tests/test_ilp_artifact.py`

**Context:** `TrainConfig` gets two new fields: `persist_telemetry: bool = False` and `telemetry_persist_limit: int = 100`. When `persist_telemetry=True`, `save()` includes a bounded snapshot of telemetry. `load()` restores it.

**Step 1: Write the failing test**

Add to `python/tests/test_ilp_artifact.py`:

```python
def test_telemetry_persistence_roundtrip(tmp_path):
    """Telemetry snapshot persists and restores when enabled."""
    from pyxlog.ilp.types import (
        LearnedArtifact, CandidateMapEntry, ArtifactMetadata,
        TrainConfig, TrainTelemetry, StepRecord,
    )
    config = TrainConfig(persist_telemetry=True, telemetry_persist_limit=5)
    steps = [
        StepRecord(step=i, loss=1.0 / (i + 1), argmax_rule="r",
                   discreteness=0.9, temperature=1.0, entropy=0.5,
                   stable_count=i)
        for i in range(10)
    ]
    art = LearnedArtifact(
        candidate_map=[CandidateMapEntry(
            id=0, i=0, j=0, k=1,
            left_name="e", right_name="e", head_name="r"
        )],
        discovered_rule="test",
        config_snapshot=config,
        telemetry=TrainTelemetry(steps=steps, step_timings={"total": 1.5}),
        metadata=ArtifactMetadata(pyxlog_version="0.5.0"),
    )
    path = tmp_path / "telem_artifact.json"
    art.save(path)

    art2 = LearnedArtifact.load(path)
    # Should have last 5 steps only (truncated by persist_limit)
    assert len(art2.telemetry.steps) == 5
    assert art2.telemetry.steps[0].step == 5  # last 5 of 0..9
    assert art2.telemetry.step_timings == {"total": 1.5}


def test_telemetry_not_persisted_by_default(tmp_path):
    """Telemetry is NOT saved when persist_telemetry=False (default)."""
    from pyxlog.ilp.types import (
        LearnedArtifact, CandidateMapEntry, ArtifactMetadata,
        TrainConfig, TrainTelemetry, StepRecord,
    )
    config = TrainConfig()  # persist_telemetry defaults to False
    steps = [StepRecord(step=0, loss=1.0, argmax_rule="r",
                        discreteness=0.9, temperature=1.0, entropy=0.5,
                        stable_count=0)]
    art = LearnedArtifact(
        candidate_map=[CandidateMapEntry(
            id=0, i=0, j=0, k=1,
            left_name="e", right_name="e", head_name="r"
        )],
        discovered_rule="test",
        config_snapshot=config,
        telemetry=TrainTelemetry(steps=steps),
        metadata=ArtifactMetadata(pyxlog_version="0.5.0"),
    )
    path = tmp_path / "no_telem.json"
    art.save(path)

    import json
    with open(path) as f:
        data = json.load(f)
    assert data.get("telemetry_snapshot") is None

    art2 = LearnedArtifact.load(path)
    assert len(art2.telemetry.steps) == 0


def test_telemetry_persist_limit_truncation(tmp_path):
    """persist_limit truncates to last N steps."""
    from pyxlog.ilp.types import (
        LearnedArtifact, CandidateMapEntry, ArtifactMetadata,
        TrainConfig, TrainTelemetry, StepRecord,
    )
    config = TrainConfig(persist_telemetry=True, telemetry_persist_limit=3)
    steps = [
        StepRecord(step=i, loss=float(i), argmax_rule="r",
                   discreteness=0.9, temperature=1.0, entropy=0.5,
                   stable_count=0)
        for i in range(100)
    ]
    art = LearnedArtifact(
        candidate_map=[CandidateMapEntry(
            id=0, i=0, j=0, k=1,
            left_name="e", right_name="e", head_name="r"
        )],
        discovered_rule="test",
        config_snapshot=config,
        telemetry=TrainTelemetry(steps=steps),
        metadata=ArtifactMetadata(pyxlog_version="0.5.0"),
    )
    path = tmp_path / "truncated.json"
    art.save(path)

    art2 = LearnedArtifact.load(path)
    assert len(art2.telemetry.steps) == 3
    assert art2.telemetry.steps[0].step == 97  # last 3 of 0..99
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_artifact.py::test_telemetry_persistence_roundtrip -v 2>&1 | tail -10`
Expected: FAIL (persist_telemetry field doesn't exist)

**Step 3: Add fields to `TrainConfig`**

In `crates/pyxlog/python/pyxlog/ilp/types.py`, after line 67 (`telemetry_sink: Path | None = None`), add:

```python
    persist_telemetry: bool = False
    telemetry_persist_limit: int = 100
```

**Step 4: Update `save()` to include telemetry_snapshot**

In `crates/pyxlog/python/pyxlog/ilp/types.py`, in the `save()` method, after the `"config_snapshot"` line (line 172), add the `telemetry_snapshot` field:

```python
        # Bounded telemetry snapshot (only when config says persist)
        telemetry_snapshot = None
        if (self.config_snapshot
                and self.config_snapshot.persist_telemetry
                and self.telemetry.steps):
            limit = self.config_snapshot.telemetry_persist_limit
            truncated_steps = self.telemetry.steps[-limit:]
            telemetry_snapshot = {
                "steps": [dataclasses.asdict(s) for s in truncated_steps],
                "step_timings": self.telemetry.step_timings,
            }

        data = {
            ...  # existing fields unchanged
            "telemetry_snapshot": telemetry_snapshot,
        }
```

The full `data` dict should now be:

```python
        data = {
            "schema_version": "beta-v2",
            "discovered_rule": self.discovered_rule,
            "candidate_map": map_data,
            "logits": self.logits,
            "soft_probs": self.soft_probs,
            "selected_hard": self.selected_hard,
            "metadata": {
                "pyxlog_version": self.metadata.pyxlog_version,
                "git_sha": self.metadata.git_sha,
                "cuda_version": self.metadata.cuda_version,
                "device_name": self.metadata.device_name,
                "candidate_map_hash": candidate_map_hash,
                "config_hash": config_hash,
                "timestamp_utc": self.metadata.timestamp_utc,
            },
            "config_snapshot": dataclasses.asdict(self.config_snapshot) if self.config_snapshot else None,
            "telemetry_snapshot": telemetry_snapshot,
        }
```

**Step 5: Update `load()` to restore telemetry from snapshot**

In `crates/pyxlog/python/pyxlog/ilp/types.py`, in the `load()` method, before the `return cls(...)` (around line 232), add telemetry restoration:

```python
        # Restore telemetry from snapshot if present (v2+)
        telemetry = TrainTelemetry()
        raw_telemetry = data.get("telemetry_snapshot")
        if raw_telemetry is not None:
            restored_steps = [
                StepRecord(**s) for s in raw_telemetry.get("steps", [])
            ]
            telemetry = TrainTelemetry(
                steps=restored_steps,
                step_timings=raw_telemetry.get("step_timings"),
            )
```

And update the `return cls(...)` to include `telemetry=telemetry`:

```python
        return cls(
            candidate_map=candidate_map,
            logits=data.get("logits", []),
            soft_probs=data.get("soft_probs", []),
            selected_hard=data.get("selected_hard", []),
            discovered_rule=data.get("discovered_rule", ""),
            config_snapshot=config_snapshot,
            telemetry=telemetry,
            metadata=metadata,
        )
```

**Step 6: Run tests**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_artifact.py -v 2>&1 | tail -20`
Expected: All artifact tests pass (including new telemetry tests)

**Step 7: Commit**

```bash
git add crates/pyxlog/python/pyxlog/ilp/types.py python/tests/test_ilp_artifact.py
git commit -m "feat(ilp): bounded telemetry persistence in beta-v2 artifacts

TrainConfig gains persist_telemetry (default False) and
telemetry_persist_limit (default 100). When enabled, save() writes
last N StepRecords + step_timings to telemetry_snapshot. load()
restores telemetry from snapshot if present, else empty."
```

---

### Task 8: Release gate validation

**Files:** None modified (validation only)

**Context:** P4 release gates from the approved design doc (design.md:285-299). The D2H gate applies to `compute_ilp_loss_grad_gpu` only, not the full training step. Three mandatory gates plus one Rust workspace check.

**Step 1: Gate 1 — Zero data-plane D2H on all 4 ILP stages**

This gate requires `strict_zero_dtoh=True` + tiny `coo_chunk_budget` to pass on all 4 stages (reach, grandparent, colleague, plus2). The existing `test_ilp_reliability.py` stages are marked `@pytest.mark.slow` so we run a direct verification script that enables strict mode and checks transfer counters.

Run the existing credit-GPU D2H tests first (reach coverage):
```bash
.venv/bin/python -m pytest python/tests/test_ilp_credit_gpu.py -v -k "dtoh" 2>&1 | tail -20
```
Expected: All D2H tests pass, including `test_strict_zero_dtoh_chunked_passes`

Then verify all 4 stages with strict mode + tiny budget + transfer counter checks:
```bash
cd /home/dev/projects/xlog && .venv/bin/python -c "
import sys; sys.path.insert(0, 'python/tests')
import torch, pyxlog
from test_ilp_reliability import STAGES
from pyxlog.ilp import TrainConfig, train_only

for name, source, pos, neg, mask in STAGES:
    print(f'=== Stage: {name} ===')
    # Compile, evaluate, set up candidates
    prog = pyxlog.IlpProgramFactory.compile(source, device=0, memory_mb=128)
    prog.evaluate()
    N = prog.ilp_schema_size()
    names = prog.ilp_relation_names()

    # Use train_only to confirm convergence
    result = train_only(source, mask, pos, neg, TrainConfig(
        step_budget_per_attempt=80, max_attempts=3, seed=42, deterministic=True))
    assert result.converged, f'{name} did not converge'

    # Now test strict zero data-plane D2H with tiny budget (forces chunking)
    prog2 = pyxlog.IlpProgramFactory.compile(source, device=0, memory_mb=128)
    prog2.evaluate()
    prog2.set_strict_zero_dtoh(True)
    prog2.set_coo_chunk_budget(1)  # tiny = force chunking

    # Set up a single candidate from valid_candidates (stage-aware)
    device = torch.device('cuda:0')
    cands = prog2.valid_candidates(mask, False)
    assert len(cands) > 0, f'{name}: no valid candidates for mask {mask}'
    c = cands[0]  # dict with keys: id, i, j, k, left_name, right_name, head_name
    cand = (c['i'], c['j'], c['k'])
    mask_t = torch.zeros(N**3, device=device, dtype=torch.float32)
    mask_t[cand[0]*N*N + cand[1]*N + cand[2]] = 1.0
    prog2.set_rule_mask(mask, mask_t, mask_t, N)
    prog2.evaluate()
    prog2.set_candidate_map([cand])
    cand_probs = torch.tensor([0.5], device=device, dtype=torch.float32)

    prog2.reset_host_transfer_stats()
    loss_dl, grad_dl = prog2.compute_ilp_loss_grad_gpu(pos, neg, cand_probs)
    loss = torch.from_dlpack(loss_dl)
    assert torch.isfinite(loss).item(), f'{name}: loss not finite'

    stats = prog2.host_transfer_stats()
    assert stats['dtoh_calls'] == 0, f'{name}: {stats[\"dtoh_calls\"]} tracked D2H calls'
    assert stats['dtoh_bytes'] == 0, f'{name}: {stats[\"dtoh_bytes\"]} tracked D2H bytes'
    print(f'  {name}: converged=True, strict_dtoh=PASS')

print('All 4 stages pass strict zero data-plane D2H gate.')
"
```
Expected: All 4 stages converge AND have zero tracked D2H calls/bytes under strict mode with chunking forced.

**Step 2: Gate 2 — SLO harness (enforced)**

The SLO assertions in `test_ilp_performance.py:134` are gated behind `ILP_PERF_ENFORCE_SLO=1`. Run with the env var set:

```bash
ILP_PERF_ENFORCE_SLO=1 .venv/bin/python -m pytest python/tests/test_ilp_performance.py -v 2>&1 | tail -15
```
Expected: All SLO tests pass with assertions enforced.

**Step 3: Gate 3 — 50-seed GA reliability**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_ga_reliability.py -v 2>&1 | tail -15`
Expected: 200/200 pass. Runtime criterion: ≤600s on native Linux (baseline from
50-seed optimization report). WSL2 adds ~70% overhead; ≤1100s accepted on WSL.

Note: This test is slow (~400-600s native, ~900-1100s WSL). If running in CI, ensure adequate timeout.

**Step 4: ILP regression suite (curated, blocking)**

Run the ILP-specific test files that are on the P0/P1 critical path:
```bash
.venv/bin/python -m pytest \
    python/tests/test_ilp_credit_gpu.py \
    python/tests/test_ilp_artifact.py \
    python/tests/test_ilp_d2h_gate.py \
    python/tests/test_ilp_backend.py \
    python/tests/test_ilp_sparse.py \
    python/tests/test_ilp_sparse_guard.py \
    python/tests/test_ilp_beta_gate.py \
    python/tests/test_ilp_performance.py \
    python/tests/test_ilp_reliability.py \
    python/tests/test_ilp_reset.py \
    -v 2>&1 | tail -30
```
Expected: All pass

**Step 5: Rust workspace tests**

Run: `cargo test --workspace --all-targets --exclude pyxlog --release 2>&1 | tail -10`
Expected: All pass

**Step 7: Commit**

```bash
git add docs/plans/2026-03-05-v050-implementation.md docs/plans/2026-03-05-v050-execution-design.md
git commit -m "docs: v0.5.0 phase 1 (P0+P1) implementation complete — all P4 gates pass

Verified: 4-stage strict zero data-plane D2H, SLO harness, 50-seed GA
reliability (200/200), ILP regression suite, Rust workspace tests."
```

---

## Sequencing Summary

```
Task 1 (rename) → Task 2 (count_mask_into_slot) → Task 3 (dtoh_scalar_untracked)
    → Task 4 (two-pass merge) → Task 5 (tests)

Task 6 (schema migration) — independent, can run after Task 1
Task 7 (telemetry persistence) — depends on Task 6

Task 8 (release gates) — after all above
```

Tasks 1-5 are sequential (each builds on the previous). Tasks 6-7 are independent of 1-5 but must be sequential with each other. Task 8 is the final validation gate.

## Out of Scope (follow-up plans)

The following tiers from the v0.5.0 design are **not** covered by this plan:

- **P2a: Term embeddings** — depends on P0 completing (touches `lib.rs`). Separate plan.
- **P2b: Extended training controls** — depends on P0 completing (touches `lib.rs`). Separate plan.
- **P3: Incremental verifier interface** — file-independent but architecturally complex. Separate plan.
