# GPU-Resident ILP Credit/Loss — Performance Report

**Date:** 2026-03-05
**Commit:** `87f689cb` (fix(ilp): eliminate D2H from empty-COO and zero-loss paths)
**Plan:** `docs/plans/2026-03-05-sparse-executor-transfer-elimination-impl.md`

---

## Summary

`compute_ilp_loss_grad_gpu` is a single Rust/CUDA call that replaces the Python-side `_compute_loss_from_candidates()` loop. The GPU path builds a COO→CSR sparse matrix on-device, runs forward (credit gather + NLL loss) and backward (gradient scatter via atomicAdd) CUDA kernels, reduces loss on-device, and returns loss + grad as DLPack tensors. **Zero D2H transfers** occur in the non-chunked path, confirmed by strict byte-level accounting (`host_transfer_stats(): dtoh_calls=0, dtoh_bytes=0`).

## Test Results

| Suite | Result | Time |
|-------|--------|------|
| GPU credit tests (`test_ilp_credit_gpu.py`) | **15/15 PASS** | 1.2s |
| Gradient parity (GPU vs Python reference) | **3/3 PASS** (f32, f64, multi-candidate) | included above |
| D2H transfer accounting (coarse + strict) | **2/2 PASS** (0 transfers, 0 bytes) | included above |
| Memory cap / chunked fallback | **1/1 PASS** | included above |
| Rust workspace (`cargo test --workspace`) | **PASS** | — |
| CUDA certification suite | **PASS** | — |
| ILP 20/20 reliability | **20/20 PASS** | — |

## D2H Transfer Accounting

**Phase 2 complete: zero D2H transfers.**

Two independent gates confirm zero host transfers during `compute_ilp_loss_grad_gpu`:

1. **Coarse gate** (`d2h_transfer_count()`): 0 column-level D2H transfers.
2. **Strict gate** (`host_transfer_stats()`): `dtoh_calls=0`, `dtoh_bytes=0`.

All five D2H transfer sites have been eliminated from non-chunked paths:

| Transfer | Old path | New path |
|----------|----------|----------|
| Membership mask download | `dtoh_sync_copy` per candidate | `scan_u8_mask_device` + `ilp_coo_fill_from_mask` kernel on-device |
| COO facts D2H for CSR build | `dtoh_sync_copy` of sorted facts | `ilp_csr_histogram` kernel + GPU prefix-sum |
| Loss reduction D2H (main path) | `dtoh_sync_copy` of loss array | `ilp_reduce_sum_{f32,f64}` kernel → device-resident scalar |
| Loss reduction D2H (empty-COO) | `dtoh_sync_copy_into` + host sum | `ilp_reduce_sum_{f32,f64}_launch` on device |
| Zero-loss scalar upload | host `0.0` → `htod_sync_copy` | `alloc` + `memset_zeros` (device-only) |

**Chunked fallback path** (activated when COO allocation exceeds `coo_memory_cap`, default 16 MB): Uses bounded D2H transfers per chunk, merges on host, then H2D uploads merged COO. This is a correctness fallback for large candidate sets; the default path is fully zero-D2H.

## New CUDA Kernels

| Kernel | File | Purpose |
|--------|------|---------|
| `ilp_coo_fill_from_mask` | `kernels/ilp.cu` | Fill COO arrays from device-side mask + prefix-sum |
| `ilp_csr_histogram` | `kernels/ilp.cu` | Histogram of fact indices for CSR row_offsets |
| `ilp_reduce_sum_f32` | `kernels/ilp.cu` | Block-level f32 sum reduction |
| `ilp_reduce_sum_f64` | `kernels/ilp.cu` | Block-level f64 sum reduction |

## Gradient Parity

Three parity tests verify GPU kernel output matches a pure-PyTorch reference implementation:

- **f32 single candidate** (pos + neg facts): loss atol=1e-6, grad atol=1e-5
- **f64 single candidate**: loss atol=1e-12, grad atol=1e-10
- **f32 multi-candidate** (2 candidates, shared derivation): loss atol=1e-6, grad atol=1e-5

All pass, confirming the CSR forward/backward kernels compute the correct surrogate NLL loss and gradients.

## Architecture

```
Python trainer
  └─ prog.compute_ilp_loss_grad_gpu(pos, neg, cand_probs)
       └─ Rust (lib.rs)
            ├─ Phase A: Parse pos/neg facts, resolve to compacted fact indices
            ├─ Phase B: For each candidate: build task (d_mask, fact_indices, cidx)
            ├─ Phase C: COO fill — scan + fill kernel per task (fully on-device)
            ├─ Phase D: Sort COO → CSR via radix sort + histogram kernel + prefix-sum
            ├─ Phase E: Forward kernel → credit + loss_contrib
            │           Reduce kernel → device-resident total loss
            │           Backward kernel → device-resident grad
            └─ Return (loss, grad) as DLPack capsules (zero-copy GPU→PyTorch)
```

CUDA kernels: `kernels/ilp_credit.cu` (forward/backward f32/f64), `kernels/ilp.cu` (COO fill, CSR histogram, reduce sum).

## Go/No-Go

**GO for Phase 2 (true zero-D2H).** All correctness gates pass: 15/15 tests, 3/3 gradient parity, strict byte-level D2H accounting (0 calls, 0 bytes), 20/20 reliability. The GPU path is functionally complete with zero host transfers in the default (non-chunked) path.

## Known Limitations

- Benchmark at toy scale only; real-world ILP training benchmarks deferred to v0.5.0 GA
- Chunked fallback path uses bounded D2H transfers (correctness-safe but not zero-D2H)
- `ilp_reduce_sum` uses `atomicAdd` to global memory — sufficient for current block counts but could use hierarchical reduction for very large arrays

## Completed Follow-Up Tasks

1. ~~Move COO/CSR construction fully to GPU~~ — Done: `ilp_coo_fill_from_mask` kernel + `ilp_csr_histogram` kernel
2. ~~Route all D2H/H2D through tracked wrappers~~ — Done: `host_transfer_stats()` strict accounting
3. ~~Memory cap + chunked fallback~~ — Done: `coo_memory_cap` field with bounded-memory chunked COO assembly
4. ~~Eliminate D2H from empty-COO and zero-loss paths~~ — Done: device-side reduce + `memset_zeros`; legacy `export_loss_grad_f32/f64` helpers removed
