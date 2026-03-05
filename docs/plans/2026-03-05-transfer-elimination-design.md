# Sparse Executor + Transfer Elimination — Design

**Date:** 2026-03-05
**Status:** Approved
**Prerequisite:** Phase 1 GPU credit/loss path (commit `6b0bc327`)

---

## Goal

Eliminate all three D2H transfers from `compute_ilp_loss_grad_gpu` so the loss/gradient path makes zero device-to-host copies. The acceptance gate asserts `host_transfer_stats().dtoh_calls == 0` and `host_transfer_stats().dtoh_bytes == 0`.

## Current D2H Sites

| # | Location | Data | Size | Purpose |
|---|----------|------|------|---------|
| 1 | `lib.rs:4173` | `u8[num_query]` per candidate | O(C × Q) bytes | Membership mask download for COO assembly |
| 2 | `lib.rs:4234` | `u32[nnz]` | O(nnz × 4) bytes | Sorted COO facts for CSR row_offsets build |
| 3 | `lib.rs:4304/4339` | `f32/f64[num_facts]` | O(F × 4-8) bytes | Loss contributions for host-side summation |

All three use raw `device.inner().dtoh_sync_copy_into()`, bypassing both the `d2h_transfer_count` gate and the `HostTransferTracker`.

## Elimination Strategy

### Transfer #1: Mask D2H → Device COO Assembly

**Current flow:** Download mask → scan for nonzeros on host → push to host COO vecs.

**New flow per candidate:**
1. `membership_mask_device()` → `d_mask[num_query]` (already on device)
2. `scan_u8_mask_device(d_mask)` → `d_prefix[num_query]` (existing primitive)
3. `compact_u32_by_mask(fact_indices, d_mask, d_prefix, num_query, d_compacted)` (existing kernel in `filter.cu`)
4. Read compacted count from device prefix-sum: `count = d_prefix[num_query-1] + d_mask[num_query-1]` — but reading this scalar is a D2H. Instead, use the **over-allocation strategy** below.

**Over-allocation strategy:** Pre-allocate COO arrays at `upper_bound = num_candidates × num_query`. Each candidate writes its compacted entries at a device-computed offset. A device-side prefix-sum over per-candidate counts produces the offset array. Grid sizes use `num_query` (the upper bound per candidate) with mask-based bounds checking.

**Per-candidate kernel pipeline:**
1. `scan_u8_mask_device(d_mask)` → `d_prefix`
2. New kernel `ilp_coo_fill_from_mask(d_mask, d_prefix, d_fact_indices, cidx, num_query, d_offsets, coo_fact, coo_cand)` — reads `d_offsets[cidx]` from device, writes compacted entries at that offset. Grid = ceil(num_query / 256). Each thread checks `mask[tid]`; if set, writes at `d_offsets[cidx] + d_prefix[tid]`.

**Offset computation (between pass 1 and pass 2):**
1. Pass 1: For each candidate, launch `count_mask(d_mask)` → `d_counts[cidx]` (existing kernel, writes to device)
2. `exclusive_scan_u32_inplace(d_counts, num_candidates)` → `d_offsets` (existing primitive)
3. Total nnz = `d_offsets[last] + d_counts[last]` — stays on device. Host uses `upper_bound` for allocation.

Pass 2 then fills COO using offsets from device.

**Memory cap + chunking (constraint #1):** If `upper_bound` exceeds a threshold (default 16 MB, configurable), split candidates into chunks. Each chunk computes its own COO segment. Merge by concatenating device buffers. The chunking path downloads only the per-chunk nnz (one `u32` scalar per chunk) to size the next allocation. This fallback trades a small number of D2H scalar reads for bounded memory use. Under the memory cap, zero D2H.

### Transfer #2: Sorted COO D2H → Device CSR Build

**Current flow:** Download sorted fact indices → histogram + prefix-sum on host → upload row_offsets.

**New flow:**
1. Radix sort COO on device (already done)
2. New kernel `ilp_csr_histogram(d_sorted_facts, nnz, num_facts, d_hist)` — each thread atomicAdds `d_hist[d_sorted_facts[tid]]`. Grid = ceil(nnz / 256).
3. `exclusive_scan_u32_inplace(d_hist, num_facts + 1)` → `d_row_offsets` (existing primitive)

The histogram kernel ignores any padding entries from over-allocation by bounds-checking `d_sorted_facts[tid] < num_facts`.

This replaces the host-side histogram loop (`lib.rs:4244-4253`) and the `row_offsets` H2D upload (`lib.rs:4256-4265`).

### Transfer #3: Loss D2H → Device Reduction

**Current flow:** Download `loss_contrib[num_facts]` → sum on host → upload scalar.

**New flow:**
1. New kernel `ilp_reduce_sum_f32(d_loss_contrib, num_facts, d_total_loss)` — standard tree reduction in shared memory, single block for num_facts < 1024, two-pass for larger.
2. Same for `ilp_reduce_sum_f64`.
3. `d_total_loss` stays on device. `export_loss_grad_*` reads from device buffer directly into DLPack capsule.

### Accounting Fix

After these changes, `compute_ilp_loss_grad_gpu` contains zero `dtoh_sync_copy_into` calls. The H2D uploads (`is_positive`, query buffers) remain and are expected.

The acceptance test uses `host_transfer_stats()` (byte+call level tracking) rather than the coarser `d2h_transfer_count` (column-level only). This catches any residual raw `dtoh_sync_copy_into` calls that might bypass the column gate.

## New CUDA Kernels

| Kernel | File | Purpose |
|--------|------|---------|
| `ilp_coo_fill_from_mask` | `kernels/ilp_credit.cu` | Scatter COO entries by device mask + prefix-sum, reading offset from device array |
| `ilp_csr_histogram` | `kernels/ilp_credit.cu` | Count fact_idx occurrences for CSR row_offsets |
| `ilp_reduce_sum_f32` | `kernels/ilp_credit.cu` | Parallel reduction of f32 array to scalar |
| `ilp_reduce_sum_f64` | `kernels/ilp_credit.cu` | Parallel reduction of f64 array to scalar |

## Existing Primitives Reused

| Primitive | Location | Use |
|-----------|----------|-----|
| `membership_mask_device` | `provider.rs` | Device-resident membership mask (already called) |
| `scan_u8_mask_device` | `provider.rs` | Mask → exclusive prefix-sum on device |
| `count_mask` | `scan.cu` | Per-candidate nnz to device scalar |
| `exclusive_scan_u32_inplace` | `provider.rs` | CSR row_offsets from histogram; candidate offset array |
| `radix_sort_u32_pairs` | `provider.rs` | COO sort (already used) |

## Acceptance Gate

```python
prog.reset_host_transfer_stats()
prog.compute_ilp_loss_grad_gpu(positives, negatives, cand_probs)
stats = prog.host_transfer_stats()
assert stats['dtoh_calls'] == 0, f"D2H calls: {stats['dtoh_calls']}"
assert stats['dtoh_bytes'] == 0, f"D2H bytes: {stats['dtoh_bytes']}"
```

Plus: grep audit confirms no `device.inner().dtoh_sync_copy_into` remains in `compute_ilp_loss_grad_gpu` or its helper functions.

## Constraints

1. **Memory cap + chunking**: Over-allocation bounded by configurable threshold (default 16 MB). Exceeding it triggers chunked processing with per-chunk scalar D2H for sizing.
2. **Minimal new kernels**: Four new kernels (`ilp_coo_fill_from_mask`, `ilp_csr_histogram`, `ilp_reduce_sum_f32/f64`). Existing `compact_u32_by_mask`, `scan_u8_mask_device`, `exclusive_scan_u32_inplace`, and `count_mask` handle the rest.
3. **Strict accounting**: `host_transfer_stats()` assertion (dtoh_calls=0, dtoh_bytes=0). No raw `dtoh_sync_copy_into` in the loss path.

## Out of Scope

- Extending `d2h_transfer_count` (column-level gate) to cover `dtoh_sync_copy_into` — the `host_transfer_stats` API already tracks this at finer granularity.
- PyTorch `autograd.Function` wrapper — Phase 3.
- Changing the chunking threshold dynamically based on GPU memory pressure.
