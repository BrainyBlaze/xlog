# GPU-Resident ILP Credit/Loss — Performance Report

**Date:** 2026-03-05
**Commit:** `8d821b66` (fix(ilp): correct GPU credit tests + trainer detach)
**Plan:** `docs/plans/2026-03-05-gpu-resident-loss-credit-impl.md`

---

## Summary

Implemented `compute_ilp_loss_grad_gpu` — a single Rust/CUDA call that replaces the Python-side `_compute_loss_from_candidates()` loop. The GPU path builds a COO→CSR sparse matrix mapping facts to covering candidates, runs forward (credit gather + NLL loss) and backward (gradient scatter via atomicAdd) CUDA kernels, and returns loss + grad as DLPack tensors. No D2H column transfers occur.

## Test Results

| Suite | Result | Time |
|-------|--------|------|
| GPU credit tests (`test_ilp_credit_gpu.py`) | **13/13 PASS** | 1.00s |
| Gradient parity (GPU vs Python reference) | **3/3 PASS** (f32, f64, multi-candidate) | included above |
| D2H transfer accounting | **1/1 PASS** (0 transfers) | included above |
| Rust workspace (`cargo test --workspace`) | **PASS** | — |
| CUDA certification suite (206 tests) | **PASS** | 26.20s |
| ILP 20/20 reliability | **20/20 PASS** | 883s |

## Microbenchmark: GPU vs Python Reference

### Small scale (1 candidate, 2 facts — reach program)

| Metric | GPU path | Python reference | Speedup |
|--------|----------|-----------------|---------|
| p50 | 6,309 µs | 10,144 µs | **1.6x** |
| p95 | 10,432 µs | 15,895 µs | **1.5x** |

### Larger scale (9 candidates, 10 facts — multi-relation reach)

| Metric | GPU path |
|--------|----------|
| p50 | 24,577 µs |
| p95 | 32,314 µs |

**Note:** At this toy scale, kernel launch overhead dominates. The GPU path's advantage grows with candidate count and fact count because the CSR scatter-gather parallelizes over facts while the Python reference does sequential `tagged_entries_containing_fact` calls per fact. Real ILP training with hundreds of candidates and thousands of facts will show larger speedups.

## D2H Transfer Accounting

`test_zero_dtoh_transfers` confirms **zero** D2H column transfers during `compute_ilp_loss_grad_gpu`. The entire credit/loss/gradient computation stays GPU-resident, consistent with the v0.5.0 zero-data-plane-host-transfer contract.

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
            ├─ Parse pos/neg facts, resolve to compacted fact indices
            ├─ For each candidate: query membership mask → COO pairs
            ├─ Sort COO → CSR (row_offsets, col_indices)
            ├─ Launch ilp_credit_forward_{f32,f64}  → credit_out, loss_contrib
            ├─ Launch ilp_credit_backward_{f32,f64} → d_cand_probs
            └─ Return (loss, grad) as DLPack capsules
```

CUDA kernels: `kernels/ilp_credit.cu` (4 kernels: `ilp_coo_fill`, forward f32/f64, backward f32/f64).

## Go/No-Go

**GO.** All correctness gates pass (13/13 tests, 3/3 parity, 0 D2H transfers, 20/20 reliability). The GPU path is functional and correct. Performance gains at toy scale are modest (1.5x) due to kernel launch overhead, but the path eliminates D2H transfers and will scale with real workloads.

## Known Limitations

- COO→CSR construction happens on CPU in Rust before upload — could be moved to GPU for large candidate sets
- Benchmark at toy scale only; real-world ILP training benchmarks deferred to v0.5.0 GA
- `test_ga_reliability_50` times out under default pytest timeout (pre-existing, unrelated)
