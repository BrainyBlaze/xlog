# Circuit Caching for 100% GPU Training (GPU Neural Fast-Path)

## Overview

Eliminate repeated D4 compilation during training by caching compiled circuits by **template signature** and running a
GPU-native weight+gradient update path per step.

**Status (Jan 25, 2026): Implemented** for the neural training fast-path:
- Cached circuits are keyed by `{predicate}:{input_count}:{num_labels}`.
- Neural outputs and gradients are exchanged as **CUDA tensors via DLPack** (no `.tolist()`, no host-side weight tables).
- Annotated-disjunction conditional-chain weights and probability gradients are computed on GPU (see design §5.3).

This plan covers the caching layer only; replacing CPU D4 with GPU D4 + GPU CDCL verifier is tracked separately in
`docs/design/2026-01-22-gpu-native-compilation-design.md`.

---

## Problem

Without caching, `forward_backward_complex()` would invoke D4 on every call. Training loops typically call
`forward_backward()` dozens of times per epoch, making D4 the dominant cost.

---

## Solution

Cache compiled circuits by **template signature**. A template is defined by:
- query predicate (e.g., `addition`)
- number of neural inputs in the query (e.g., `2`)
- neural label count (e.g., `10` for MNIST digits)

At runtime:
- neural probabilities arrive as device-resident tensors (Torch CUDA)
- a cached device slot-map tells XLOG which CNF vars correspond to each probability slot
- the GPU kernels fill log-weights and scatter probability gradients directly into device grad tensors

---

## Cache Key

**v0.4.0-alpha scope:** single network per program.

```
{predicate}:{input_count}:{num_labels}
```

Example:
```
addition:2:10
```

---

## Data Structures

```rust
pub struct CachedCircuit {
    pub program: ExactDdnnfProgram,
    pub slots: GpuWeightSlots,
    pub num_labels: usize,
}
```

Where:
- `program` owns the compiled circuit (GPU-resident once initialized).
- `slots` is a **device-resident** mapping from neural output slots → CNF var ids (DIMACS, 1-based).
- `num_labels` is used for template query indexing (e.g., sum range for `addition`).

---

## Algorithm (Runtime)

```
forward_backward_complex(query):
  1) Parse query → (predicate, input_indices, target_value)
  2) Run neural network for each input index → output_squeezed (CUDA, 1D, f32)
  3) cache_key = {predicate}:{len(input_indices)}:{num_labels}

  4) On cache miss:
       a) Build template-expanded source with placeholder inputs 0..num_inputs-1
       b) Use uniform dummy probabilities (sum < 1.0) to force an implicit "none" branch
       c) Emit a query for each possible target value (so runtime selects by index)
       d) Compile once with D4 and build/upload `GpuWeightSlots`

  5) Allocate grad tensors `zeros_like(output_squeezed)` (CUDA)
  6) Call:
       ExactDdnnfProgram::neural_backward_nll_buffers_with_device_loss(
         slots, query_idx, probs_dlpack, grads_dlpack
       )
     This:
       - fills AD-chain weights on GPU from device probabilities
       - runs GPU forward+backward twice (base + query-forced)
       - scatters probability-space gradients on GPU into grad tensors
       - returns device-resident scalar NLL loss (no device->host reads required)
  7) Call `output_squeezed.backward(grad_tensor)` for each input
```

Notes:
- No host-side `.tolist()` extraction.
- No host-side construction of weight tables or gradient maps.
- The legacy `forward_backward(...) -> f64` API may still read back a single scalar loss for convenience, but the
  strict GPU-native path returns a CUDA tensor loss via `forward_backward_tensor(...)`.

---

## Files

Implemented in:
- `crates/pyxlog/src/lib.rs`
- `crates/xlog-prob/src/exact.rs`
- `crates/xlog-prob/src/neural_fast_path.rs`
- `kernels/neural.cu` (+ provider wiring)

---

## Testing

- Python:
  - `python/tests/test_circuit_cache.py` is CUDA-only and forbids `.tolist()` to enforce GPU-native behavior.
  - `python/tests/test_gpu_native_forward_backward_returns_tensor.py` is CUDA-only and forbids `.tolist()` and
    `.item()` to enforce the strict GPU-native API: `forward_backward_tensor(...) -> torch.Tensor`.
- Rust:
  - `crates/xlog-prob/tests/neural_fast_path.rs` checks GPU fast-path gradients and loss on a tiny AD program.
  - `crates/xlog-prob/tests/no_dtoh_in_gpu_neural_fast_path.rs` guards against accidental device→host reads in the
    slot-map module.
  - `crates/xlog-prob/tests/no_dtoh_in_neural_backward_nll.rs` guards against accidental device→host reads in the
    end-to-end neural fast-path entrypoint.
