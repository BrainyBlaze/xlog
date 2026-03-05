# GPU-Resident ILP Credit/Loss Path — v0.5 Phase 1 Design

**Date:** 2026-03-05
**Author:** Claude (brainstorming session with user)
**Status:** Approved
**Supersedes:** Python-side `_compute_loss_from_candidates()` in `trainer.py:643-698`

---

## Goal

Replace the Python per-fact credit aggregation loop with a single Rust/CUDA call that computes ILP training loss and its gradient entirely on GPU. The new method returns GPU-resident tensors via DLPack — zero data-plane device-to-host transfers.

## Scope

**In scope (Phase 1):**
- New PyO3 method: `compute_ilp_loss_grad_gpu(positives, negatives, cand_probs_dlpack) -> (loss_dlpack, grad_dlpack)`
- GPU-resident sparse (CSR) credit structure built from retained `IlpTagEntry` buffers
- Forward kernel: sparse credit gather + clamp + NLL loss reduction
- Backward kernel: gradient scatter into `d_cand_probs`
- Candidate `(i,j,k) -> cidx` mapping uploaded once per attempt
- Gradient parity tests against current Python implementation
- Trainer integration: drop-in replacement for `_compute_loss_from_candidates`

**Out of scope:**
- Witness coverage (`batch_fact_membership` — O(20 bytes) D2H, acceptable)
- Convergence detection, temperature scheduling (control plane, stays Python-side)
- PyTorch `autograd.Function` wrapper (optional Phase 2)

---

## Current Pipeline (What We Replace)

The training step calls `_compute_loss_from_candidates()`, which:

1. Calls `prog.batch_tagged_credit(rel, facts)` per relation — Rust runs GPU semi-joins per retained entry, downloads boolean masks to host, builds `List[List[(i,j,k)]]`
2. Loops over each fact's `(i,j,k)` triples in Python
3. Looks up `ci = ijk_to_cidx[(i,j,k)]`, sums `cand_probs[ci]` (one autograd node per index op)
4. Computes `loss += -log(credit)` for positives, `loss += -log(1 - credit)` for negatives

This creates thousands of small Python tensor ops per step, each adding autograd graph nodes.

---

## New Architecture

### Gradient Strategy

Rust-side full backward + DLPack export, matching the proven neural training pattern (`neural_backward_nll_buffers_with_device_loss`). Python calls `cand_probs.backward(grad_tensor)` to propagate gradients through the existing autograd graph to `W`.

### Data Flow (One Training Step)

```
Python trainer:
  1. backend.apply_mask(prog, ..., cand_probs)       # existing
  2. prog.evaluate()                                   # existing, retains IlpTaggedResult
  3. (loss_dl, grad_dl) = prog.compute_ilp_loss_grad_gpu(
         positives, negatives, cand_probs_dlpack)      # NEW
  4. loss = torch.from_dlpack(loss_dl)                  # CUDA scalar
  5. grad = torch.from_dlpack(grad_dl)                  # CUDA tensor (C,)
  6. cand_probs.backward(grad)                          # PyTorch backward to W
  7. optimizer.step()                                   # existing
```

### Inside `compute_ilp_loss_grad_gpu` (Rust)

**Host control plane** (metadata only):
- Import `cand_probs` via DLPack, validate `len == candidate_map.len()` + hash
- Iterate `IlpTagEntry` list (host loop, ~20 entries): look up `cidx` per entry
- Compute kernel launch parameters (sizes, offsets)

**GPU data plane** (zero D2H):

1. **Upload query facts** to GPU via `pack_i64_columns_typed` (existing)
2. **Per entry**: `membership_mask_device(query_buf, entry.buffer)` returns `CudaBuffer<u8>` mask on GPU (new variant — does not download)
3. **Stream-compact** each mask to `matching_fact_indices` (GPU filter+compact, existing primitives)
4. **Fill COO chunk**: pair `(matching_fact_indices, broadcast cidx)` on GPU
5. **Concatenate** all COO chunks on GPU
6. **Sort** COO by `(fact_idx, cand_idx)` for deterministic accumulation (existing radix sort)
7. **Build CSR** `row_offsets` via GPU boundary detection + prefix sum (existing kernels)
8. **Forward kernel** (`ilp_credit_forward`):
   - Per row `f`: `credit[f] = sum(cand_probs[col_indices[ptr]])` for `ptr` in `row_offsets[f]..row_offsets[f+1]`
   - `credit[f] = max(credit[f], eps)` where `eps = 1e-8`
   - Positives: `loss_contrib[f] = -log(credit[f])`
   - Negatives: `loss_contrib[f] = -log(1 - credit[f])`
   - `loss = sum(loss_contrib)` via GPU reduction
9. **Backward kernel** (`ilp_credit_backward`):
   - Per row `f`: `d_credit[f] = -1/credit[f]` (positives) or `1/(1 - credit[f])` (negatives)
   - For each `(f, c)` in CSR: `atomicAdd(d_cand_probs[c], d_credit[f])`
10. **Export** `(loss_scalar, d_cand_probs)` via DLPack

### New CUDA Kernel File: `kernels/ilp_credit.cu`

Three kernels:
- `ilp_coo_fill`: Given a compacted index array and a scalar `cidx`, write `(fact_idx, cidx)` pairs
- `ilp_credit_forward`: CSR gather → credit sum → clamp → NLL → reduction
- `ilp_credit_backward`: CSR traverse → d_credit → atomic scatter into `d_cand_probs`

### New Rust Primitives

- `membership_mask_device()` on `CudaKernelProvider`: variant of `membership_mask` that returns `CudaBuffer<u8>` instead of downloading to `Vec<bool>`. The existing `membership_mask` can call this internally + download.
- `compute_ilp_loss_grad_gpu()` on `IlpProgramFactory` (PyO3): orchestrates the full pipeline

### Candidate Mapping

- `set_candidate_map(candidates: List[(i,j,k)])` called once per attempt start
- Stored as `HashMap<(u32,u32,u32), u32>` in Rust + GPU-uploaded `u32` lookup array
- Validated each call: `len(cand_probs) == candidate_map.len()`
- Stable hash for consistency check across resets

---

## Contracts

### Tensor Contract
- `cand_probs` input: f32 (primary) or f64, CUDA device
- `grad_dlpack` output: same shape, device, and dtype as `cand_probs`
- `loss_dlpack` output: scalar, same dtype as `cand_probs`

### Dtype Policy
- f32: default computation dtype
- f64: full f64 kernels (no cast), matching tolerances

### Numerical Stability
- Forward clamp: `credit[f] = max(credit[f], 1e-8)` — matches current Python `clamp(min=1e-8)`
- Same clamp applies to `(1 - credit[f])` for negatives
- Backward uses clamped values for derivative

### Membership Mask Device
- Output: `CudaBuffer<u8>`, values in `{0, 1}` only
- Debug assertion validates no values outside `{0, 1}`
- Deterministic ordering in compact output

### Determinism
- COO sorted by `(fact_idx, cand_idx)` before CSR build
- Accumulation order in forward kernel follows CSR column order (deterministic)
- Backward atomic adds are non-deterministic in f32; document tolerance. For reproducibility tests, use f64.

---

## Success Criteria

### Correctness (must-pass)

| Check | f32 threshold | f64 threshold |
|-------|--------------|--------------|
| Loss parity (atol) | 1e-6 | 1e-12 |
| Loss parity (rtol) | 1e-5 | 1e-10 |
| Gradient max abs error | 1e-5 | 1e-10 |

- Semantic parity: same discovered rule for each seed on reach/grandparent mini-suite

### Data-Plane Transfer (must-pass)

- `compute_ilp_loss_grad_gpu` causes 0 additional `dtoh_bytes` and `dtoh_calls` in `host_transfer_stats()`
- Control metadata excluded by contract

### Performance (go/no-go)

| Metric | Target | Baseline |
|--------|--------|----------|
| `loss_credit_us` p95 | >= 8x faster | Current Python loop baseline |
| GA 50-seed wall-clock | <= 327s | 436s (>= 25% faster) |

### No Regression (must-pass)

- `test_ilp_reliability.py`: 20/20 pass
- `test_ilp_ga_reliability.py`: Clopper-Pearson lower95 >= 0.929

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Atomic float add precision in backward | Medium | Low | Sort COO for deterministic order; use f64 for parity tests; document tolerance |
| CSR build overhead exceeds Python loop savings | Low | Medium | Profile CSR build separately; typical size is hundreds of nonzeros |
| `membership_mask_device` refactor breaks existing callers | Low | High | Implement as new method; existing `membership_mask` calls it internally + downloads |
| DLPack lifetime/ownership issues with cand_probs | Medium | Medium | Follow existing neural path pattern exactly; borrow cand_probs read-only |

---

## Phase 2 (Future, Not in Scope)

- PyTorch `autograd.Function` wrapper for cleaner API
- GPU-resident witness coverage (move `batch_fact_membership` D2H to device scalar)
- Fused semi-join + COO build kernel (eliminate per-entry membership_mask calls)
- Control-plane migration (convergence detection, temperature scheduling)
