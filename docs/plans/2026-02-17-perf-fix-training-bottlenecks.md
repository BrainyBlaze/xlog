# Fix Training Performance Bottlenecks Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce per-query training latency from ~16 seconds to <100ms by eliminating CPU-side bottlenecks in the XGCF backward pass, removing unnecessary device synchronizations, and scaling GPU memory defaults.

**Architecture:** The backward loop in `gpu_cache.rs:eval_grads_inplace` iterates `level_cap` (65,536) times launching 3 kernels per level from the CPU, even though actual circuits have ~20-50 levels. We replace this with a single fused backward kernel (matching the forward path's `xgcf_eval_all_levels_cached` pattern). We also remove the per-query `device().synchronize()` calls in the neural fast-path and scale memory/cache defaults.

**Tech Stack:** Rust (xlog-prob, xlog-cuda, pyxlog), CUDA C++ (kernels/circuit.cu), PyO3

**Root Cause Evidence:**
- `gpu_cache.rs:1456-1457`: backward loops `level_cap` not actual levels → ~393K kernel launches/query
- `exact.rs:578`: `device().synchronize()` per query → full pipeline drain
- `gpu_cache.rs:1573`: second `synchronize()` inside `eval_grads_inplace` → another drain
- `exact.rs:80`: hardcoded 1GB memory default
- `exact.rs:1240`: `num_slots: 1` cache

---

## Task 0: Cherry-pick main-only commits into worktree

**Context:** 7 commits landed on `main` that belong in `.worktrees/v0.4.0-alpha-integrated`. These must be ported before performance work begins, since the performance fixes target the worktree branch.

**Commits to port:**
- `aca906ed` — `train_model_tensor` + `clear_circuit_cache` + test suite
- `1572f483` — train.py updates (engine flag, seed, eval)
- `52cd912b` — stdout flush fix
- `a375b75e` — cache ablation benchmark script
- `c88e07f6` — Track A collection script + comparison data
- `5ad1b173` — Track A seed 7 results
- `626416ce` — Track A seed 42 results

**Step 1: Cherry-pick commits into worktree**

```bash
cd /home/dev/projects/xlog/.worktrees/v0.4.0-alpha-integrated
git cherry-pick aca906ed 1572f483 52cd912b a375b75e 5ad1b173 626416ce c88e07f6
```

Resolve any conflicts (most likely in `crates/pyxlog/src/lib.rs` — the worktree's `lib.rs` has different neural registry structure).

**Step 2: Verify build**

```bash
cargo build --release -p pyxlog
```

**Step 3: Run existing tests**

```bash
cargo test -p xlog-prob
```

**Step 4: Commit if cherry-pick required conflict resolution**

---

## Task 1: Write fused backward CUDA kernel

**Files:**
- Modify: `kernels/circuit.cu` (append new kernel)
- Modify: `crates/xlog-cuda/src/provider.rs:421-427` (register new kernel name)

**Context:** The forward path already has `xgcf_eval_all_levels_cached` (circuit.cu:585-703) which runs all levels in a single kernel using `meta_num_levels[slot]`. The backward path has NO equivalent — it launches 3 separate kernels per level from the CPU. We need `xgcf_backward_all_levels_cached` that does the same level loop internally.

**Step 1: Write the failing test**

Create test: `crates/xlog-prob/tests/gpu_backward_fused_parity.rs`

```rust
//! Verify fused backward kernel produces identical gradients to the per-level backward.
//!
//! Strategy: compile a small circuit, run eval_grads_inplace (existing per-level),
//! save grad_true/grad_false. Then run eval_grads_inplace_fused (new), compare.

#[cfg(feature = "gpu-tests")]
mod fused_backward {
    use xlog_prob::ExactDdnnfProgram;

    #[test]
    fn fused_backward_matches_per_level() {
        let source = r#"
            0.3::a. 0.7::b.
            q :- a, b.
            query(q).
        "#;

        let prog = ExactDdnnfProgram::compile_source_with_gpu(
            source,
            xlog_prob::GpuConfig::default(),
        )
        .expect("compile");

        // Run existing per-level backward
        let result_per_level = prog.evaluate_gpu_with_grads().expect("per-level");

        // Run new fused backward
        let result_fused = prog.evaluate_gpu_with_grads_fused().expect("fused");

        // Gradients must match within f64 epsilon
        for (a, b) in result_per_level.query_grads.iter()
            .zip(result_fused.query_grads.iter())
        {
            assert!(
                (a.probability - b.probability).abs() < 1e-10,
                "probability mismatch: {} vs {}",
                a.probability, b.probability
            );
        }
    }
}
```

**Step 2: Run test to verify it fails**

```bash
cargo test -p xlog-prob fused_backward_matches_per_level -- --nocapture
```

Expected: FAIL — `evaluate_gpu_with_grads_fused` does not exist.

**Step 3: Write the fused CUDA kernel**

Append to `kernels/circuit.cu` (after `xgcf_backward_level_lit_grad_cached` at ~line 920):

```cuda
/**
 * Fused backward pass through all levels of a cached XGCF circuit.
 *
 * Mirrors xgcf_eval_all_levels_cached but for the backward direction:
 * iterates levels in reverse using meta_num_levels[slot], performing
 * propagate + decision_grad + lit_grad per level within a single kernel.
 *
 * This eliminates ~3 * level_cap host-side kernel launches per evaluation,
 * replacing them with a single launch that loops internally.
 *
 * Block size: 256 threads. Single block (blockIdx.x == 0 only).
 */
extern "C" __global__ void xgcf_backward_all_levels_cached(
    const uint32_t* __restrict__ active_slot,
    uint32_t node_cap,
    uint32_t edge_cap,
    uint32_t level_cap,
    uint32_t var_cap,
    const uint8_t* __restrict__ node_type,
    const uint32_t* __restrict__ child_offsets,
    const uint32_t* __restrict__ child_indices,
    const uint32_t* __restrict__ decision_var,
    const uint32_t* __restrict__ decision_child_false,
    const uint32_t* __restrict__ decision_child_true,
    const int32_t* __restrict__ lit,
    const uint32_t* __restrict__ level_nodes,
    const uint32_t* __restrict__ level_offsets,
    const double* __restrict__ var_log_true,
    const double* __restrict__ var_log_false,
    const double* __restrict__ values,
    double* __restrict__ adj,
    double* __restrict__ grad_true,
    double* __restrict__ grad_false,
    const uint32_t* __restrict__ meta_num_levels
) {
    if (blockIdx.x != 0) return;

    uint32_t slot = active_slot[0];
    uint32_t num_levels = meta_num_levels[slot];
    if (num_levels == 0) return;

    size_t node_base = slot_offset(slot, node_cap);
    size_t edge_base = slot_offset(slot, edge_cap);
    size_t offset_base = slot_offset(slot, node_cap + 1u);
    size_t level_base = slot_offset(slot, node_cap);
    size_t level_offset_base = slot_offset(slot, level_cap + 1u);
    size_t var_base = slot_offset(slot, var_cap + 1u);

    const uint8_t*  nt  = node_type + node_base;
    const uint32_t* co  = child_offsets + offset_base;
    const uint32_t* ci  = child_indices + edge_base;
    const uint32_t* dv  = decision_var + node_base;
    const uint32_t* dcf = decision_child_false + node_base;
    const uint32_t* dct = decision_child_true + node_base;
    const int32_t*  lt  = lit + node_base;
    const uint32_t* ln  = level_nodes + level_base;
    const uint32_t* lo  = level_offsets + level_offset_base;
    const double*   vlt = var_log_true + var_base;
    const double*   vlf = var_log_false + var_base;
    const double*   val = values + node_base;
    double*         ad  = adj + node_base;
    double*         gt  = grad_true + var_base;
    double*         gf  = grad_false + var_base;

    for (int32_t level = (int32_t)num_levels - 1; level >= 0; --level) {
        uint32_t loff = lo[level];
        uint32_t num_nodes = lo[level + 1] - loff;

        // Phase 1: Propagate adjoint from parent to children
        for (uint32_t idx = threadIdx.x; idx < num_nodes; idx += blockDim.x) {
            uint32_t node = ln[loff + idx];
            uint8_t ty = nt[node];
            double a = ad[node];

            if (ty == XGCF_AND) {
                uint32_t c0 = co[node];
                uint32_t c1 = co[node + 1];
                for (uint32_t i = c0; i < c1; i++) {
                    atomicAdd(&ad[ci[i]], a);
                }
            } else if (ty == XGCF_OR) {
                uint32_t c0 = co[node];
                uint32_t c1 = co[node + 1];
                double parent_val = val[node];
                for (uint32_t i = c0; i < c1; i++) {
                    uint32_t child = ci[i];
                    double w = exp(val[child] - parent_val);
                    atomicAdd(&ad[child], a * w);
                }
            } else if (ty == XGCF_DECISION) {
                uint32_t var = dv[node];
                uint32_t child_f = dcf[node];
                uint32_t child_t = dct[node];
                double t = vlt[var];
                double f = vlf[var];
                double parent_val = val[node];
                double wt = exp(t + val[child_t] - parent_val);
                double wf = exp(f + val[child_f] - parent_val);
                atomicAdd(&ad[child_t], a * wt);
                atomicAdd(&ad[child_f], a * wf);
                // Decision variable gradients
                atomicAdd(&gt[var], a * wt * val[child_t]);
                atomicAdd(&gf[var], a * wf * val[child_f]);
            }
        }
        __syncthreads();

        // Phase 2: Literal gradients (accumulate into var grad buffers)
        for (uint32_t idx = threadIdx.x; idx < num_nodes; idx += blockDim.x) {
            uint32_t node = ln[loff + idx];
            if (nt[node] == XGCF_LIT) {
                int32_t l = lt[node];
                uint32_t var = (l > 0) ? (uint32_t)l : (uint32_t)(-l);
                double a = ad[node];
                if (l > 0) {
                    atomicAdd(&gt[var], a);
                } else {
                    atomicAdd(&gf[var], a);
                }
            }
        }
        __syncthreads();
    }
}
```

**IMPORTANT:** This kernel must be verified against the existing per-level kernels for correctness. The decision grad computation differs slightly from the per-level version — compare carefully with `xgcf_backward_level_decision_grad_cached` (circuit.cu:~830) and `xgcf_backward_level_propagate_cached` (circuit.cu:~721) to ensure identical semantics. The above is a reference structure; the exact gradient formulas must match.

**Step 4: Recompile PTX**

```bash
nvcc -ptx -arch=sm_80 kernels/circuit.cu -o kernels/circuit.ptx
```

**Step 5: Register kernel name**

In `crates/xlog-cuda/src/provider.rs`, add after line 427:

```rust
pub const XGCF_BACKWARD_ALL_LEVELS_CACHED: &str = "xgcf_backward_all_levels_cached";
```

And register it in the PTX loading section (~line 938) alongside the other circuit kernels.

**Step 6: Commit**

```bash
git add kernels/circuit.cu kernels/circuit.ptx crates/xlog-cuda/src/provider.rs
git commit -m "feat(gpu): add fused backward XGCF kernel (xgcf_backward_all_levels_cached)"
```

---

## Task 2: Replace per-level backward loop with fused kernel call

**Files:**
- Modify: `crates/xlog-prob/src/compilation/gpu_cache.rs:1272-1575` (`eval_grads_inplace`)

**Context:** Currently `eval_grads_inplace` does:
1. Launch `xgcf_eval_all_levels_cached` (forward, 1 kernel) ✓ good
2. Zero adj, grad_true, grad_false (3 kernel launches) — keep
3. Set root adjoint to 1.0 (1 kernel launch) — keep
4. **Loop `level_cap` times launching 3 kernels each** ← REPLACE
5. `apply_free_var_correction_cached` — keep
6. `synchronize()` at line 1573 — REMOVE (Task 3)

**Step 1: Add `eval_grads_inplace_fused` method**

Add a new method to `GpuCircuitCache` that replaces the per-level loop (lines 1457-1570) with a single call to `xgcf_backward_all_levels_cached`:

```rust
// In gpu_cache.rs, after eval_grads_inplace

pub fn eval_grads_inplace_fused(&mut self, handle: &GpuCircuitCacheHandle) -> Result<()> {
    let device = self.provider.device().inner();

    // 1. Forward: same single-kernel eval
    let eval_all = device
        .get_func(
            xlog_cuda::CIRCUIT_MODULE,
            xlog_cuda::circuit_kernels::XGCF_EVAL_ALL_LEVELS_CACHED,
        )
        .ok_or_else(|| XlogError::Kernel("xgcf_eval_all_levels_cached not found".into()))?;

    let block_size: u32 = 256;
    // ... (same forward launch as existing eval_grads_inplace lines 1282-1313)

    // 2. Zero adj, grad_true, grad_false: same 3 store launches
    // ... (same as existing lines 1316-1396)

    // 3. Set root adjoint
    // ... (same as existing lines 1398-1422)

    // 4. FUSED backward: single kernel replaces the level loop
    let backward_all = device
        .get_func(
            xlog_cuda::CIRCUIT_MODULE,
            xlog_cuda::circuit_kernels::XGCF_BACKWARD_ALL_LEVELS_CACHED,
        )
        .ok_or_else(|| XlogError::Kernel("xgcf_backward_all_levels_cached not found".into()))?;

    unsafe {
        backward_all.clone().launch(
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                handle.slot_device(),
                self.node_cap,
                self.edge_cap,
                self.level_cap,
                self.var_cap,
                &self.node_type,
                &self.child_offsets,
                &self.child_indices,
                &self.decision_var,
                &self.decision_child_false,
                &self.decision_child_true,
                &self.lit,
                &self.level_nodes,
                &self.level_offsets,
                &self.var_log_true,
                &self.var_log_false,
                &self.values,
                &mut self.adj,
                &mut self.grad_true,
                &mut self.grad_false,
                &self.meta_num_levels,
            ),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("xgcf_backward_all_levels_cached failed: {}", e)))?;

    // 5. Free-var correction (unchanged)
    self.apply_free_var_correction_cached(handle, true, true)?;

    // NO synchronize() here — caller decides when to sync
    Ok(())
}
```

**Step 2: Run parity test**

```bash
cargo test -p xlog-prob fused_backward_matches_per_level -- --nocapture
```

Expected: PASS — gradients match between per-level and fused.

**Step 3: Wire fused path into neural fast-path**

In `crates/xlog-prob/src/exact.rs`, change both calls to `eval_grads_inplace`:
- Line 423: `cache.eval_grads_inplace(state.handle())?;` → `cache.eval_grads_inplace_fused(state.handle())?;`
- Line 513: `cache.eval_grads_inplace(state.handle())?;` → `cache.eval_grads_inplace_fused(state.handle())?;`

**Step 4: Run all xlog-prob tests**

```bash
cargo test -p xlog-prob
```

**Step 5: Commit**

```bash
git add crates/xlog-prob/src/compilation/gpu_cache.rs crates/xlog-prob/src/exact.rs
git commit -m "perf(gpu): replace per-level backward loop with fused kernel

Eliminates ~196K kernel launches per query (3 × level_cap where level_cap=65536).
Replaced with single xgcf_backward_all_levels_cached launch that loops internally
using meta_num_levels[slot] (actual level count, typically 20-50)."
```

---

## Task 3: Remove per-query device().synchronize() calls

**Files:**
- Modify: `crates/xlog-prob/src/exact.rs:578`
- Modify: `crates/xlog-prob/src/compilation/gpu_cache.rs:1573`

**Context:** After Task 2, the fused path has no `synchronize()` inside `eval_grads_inplace_fused`. But `exact.rs:578` still has a global sync at the end of `neural_backward_nll_buffers_inner`. This sync is unnecessary because:
- The loss tensor stays on GPU (returned as `TrackedCudaSlice<f64>`)
- Gradient buffers stay on GPU (written via DLPack zero-copy)
- PyTorch's `.backward()` operates on CUDA tensors (doesn't need host sync)
- The only host read happens at `.item()` per batch (in `train_epoch_tensor_internal`), which triggers its own implicit sync

**Step 1: Write guardrail test**

Create test: `crates/xlog-prob/tests/no_synchronize_in_neural_backward.rs`

```rust
//! Guardrail: neural_backward_nll_buffers_inner must not call device().synchronize().
//! The caller (pyxlog training loop) is responsible for any needed sync points.

use std::path::PathBuf;

#[test]
fn no_device_synchronize_in_neural_backward_inner() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("src");
    path.push("exact.rs");

    let text = std::fs::read_to_string(&path).expect("read exact.rs");

    // Find the function body of neural_backward_nll_buffers_inner
    let fn_start = text
        .find("fn neural_backward_nll_buffers_inner")
        .expect("function must exist");
    // Find the next top-level function after it
    let fn_body = &text[fn_start..];
    let fn_end = fn_body[1..]
        .find("\n    pub fn ")
        .or_else(|| fn_body[1..].find("\n    #[cfg"))
        .unwrap_or(fn_body.len() - 1);
    let fn_text = &fn_body[..fn_end];

    assert!(
        !fn_text.contains(".synchronize()"),
        "neural_backward_nll_buffers_inner must not call device().synchronize(). \
         Remove the sync — callers handle sync at batch boundaries."
    );
}
```

**Step 2: Run test to verify it fails**

```bash
cargo test -p xlog-prob no_device_synchronize_in_neural_backward_inner -- --nocapture
```

Expected: FAIL — `exact.rs:578` still has `state.provider.device().synchronize()?;`

**Step 3: Remove the synchronize call**

In `crates/xlog-prob/src/exact.rs`, line 578:
```rust
// REMOVE this line:
state.provider.device().synchronize()?;
```

Also remove the `synchronize()` from the OLD `eval_grads_inplace` if it's still reachable:
`crates/xlog-prob/src/compilation/gpu_cache.rs:1573`:
```rust
// REMOVE this line:
self.provider.device().synchronize()?;
```

**Step 4: Run guardrail test + all tests**

```bash
cargo test -p xlog-prob no_device_synchronize_in_neural_backward_inner -- --nocapture
cargo test -p xlog-prob
```

**Step 5: Commit**

```bash
git add crates/xlog-prob/src/exact.rs crates/xlog-prob/src/compilation/gpu_cache.rs \
    crates/xlog-prob/tests/no_synchronize_in_neural_backward.rs
git commit -m "perf(gpu): remove per-query device().synchronize() from neural fast-path

The synchronize at exact.rs:578 drained the entire GPU pipeline after every
query evaluation. Since loss and gradients stay device-resident, no host sync
is needed until the training loop reads the batch loss via .item().

Adds guardrail test to prevent regression."
```

---

## Task 4: Scale GPU memory and cache defaults

**Files:**
- Modify: `crates/xlog-prob/src/exact.rs:76-82` (GpuConfig default)
- Modify: `crates/xlog-prob/src/exact.rs:1239-1246` (cache config)
- Modify: `crates/pyxlog/src/lib.rs:430` (Python API default)

**Step 1: Write test for new defaults**

Create test: `crates/xlog-prob/tests/gpu_config_defaults.rs`

```rust
#[test]
fn default_memory_uses_available_gpu_when_possible() {
    let config = xlog_prob::GpuConfig::default();
    // Default should be large enough for serious training (at least 4GB)
    assert!(
        config.memory_bytes >= 4 * 1024 * 1024 * 1024,
        "Default memory_bytes {} is too small for training",
        config.memory_bytes
    );
}
```

**Step 2: Run test to verify it fails**

```bash
cargo test -p xlog-prob default_memory_uses_available_gpu -- --nocapture
```

Expected: FAIL — current default is 1GB.

**Step 3: Update defaults**

In `crates/xlog-prob/src/exact.rs`, update `GpuConfig::default()`:

```rust
impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            device_ordinal: 0,
            // Use ~80% of GPU memory by default for training workloads.
            // Previous 1GB default starved D4 compiler and circuit cache.
            memory_bytes: 32 * 1024 * 1024 * 1024, // 32GB
        }
    }
}
```

In `crates/xlog-prob/src/exact.rs`, update `default_cache_config`:

```rust
pub(crate) fn default_cache_config(
    cnf: &xlog_solve::GpuCnf,
    compile: &GpuCompileConfig,
) -> Result<GpuCircuitCacheConfig> {
    if compile.smooth_node_cap == 0 || compile.smooth_edge_cap == 0 {
        return Err(XlogError::Compilation(
            "GPU cache config requires non-zero smoothing caps".to_string(),
        ));
    }
    Ok(GpuCircuitCacheConfig {
        num_slots: 4,         // Was 1. Allow 4 circuit templates in cache.
        table_size: 8,        // Was 1. Power-of-2 hash table for 4 slots.
        node_cap: compile.smooth_node_cap,
        edge_cap: compile.smooth_edge_cap,
        level_cap: compile.smooth_node_cap,
        var_cap: cnf.var_cap,
    })
}
```

In `crates/pyxlog/src/lib.rs`, update the Python API defaults. Find all `memory_mb=1024` signature defaults and change to `memory_mb=32768`:

```rust
#[pyo3(signature = (source, device=0, memory_mb=32768, prob_engine=None))]
```

Apply the same change to all `memory_mb` defaults in `Program::compile`, `LogicProgram::compile`, and the `dlpack_roundtrip` function.

**Step 4: Run tests**

```bash
cargo test -p xlog-prob
cargo test -p pyxlog
```

**Step 5: Commit**

```bash
git add crates/xlog-prob/src/exact.rs crates/pyxlog/src/lib.rs \
    crates/xlog-prob/tests/gpu_config_defaults.rs
git commit -m "perf: scale GPU memory default to 32GB, cache slots to 4

Previous 1GB default used 2.5% of a 40GB GPU. Previous num_slots=1 cache
evicted the circuit on every new query shape.

Now defaults to 32GB (safe on A100/A6000; clamped to available memory at
runtime by GpuMemoryManager). Cache holds 4 templates simultaneously."
```

---

## Task 5: Add GIL release around GPU work in pyxlog

**Files:**
- Modify: `crates/pyxlog/src/lib.rs` (training loop functions)

**Context:** The Python GIL is held during the entire neural fast-path GPU execution. While this doesn't block GPU kernels, it:
- Prevents Python-side parallelism (data loading, etc.)
- Adds unnecessary PyO3 FFI overhead for the long-running Rust/GPU work
- Blocks Python's garbage collector and signal handling

The fix is to wrap the Rust-only GPU orchestration in `py.allow_threads(...)`. We can only do this for code sections that don't call back into Python.

**Step 1: Identify GIL-releasable sections**

In `forward_backward_complex_tensor` (lib.rs:1591-1837):
- Lines 1780-1790: The `neural_backward_nll_buffers_with_device_loss` call is pure Rust/CUDA — no Python callbacks. This is the hot section.
- Lines 1793-1802: `htod_sync_copy_into` is pure CUDA.

We CANNOT release GIL around:
- PyTorch forward passes (lines 1662-1713) — these call Python
- DLPack import/export (calls into Python torch)
- PyTorch `.backward()` calls (lines 1831-1834)

**Step 2: Wrap GPU-heavy section**

In `forward_backward_complex_tensor`, replace lines ~1780-1815 with:

```rust
// Release GIL during pure Rust/CUDA neural fast-path execution.
let (loss_dev_bytes, loss_dev_len) = py.allow_threads(|| -> Result<_> {
    let cfg = NeuralFastPathConfig::default();
    let loss_dev = cached
        .program
        .neural_backward_nll_buffers_with_device_loss(
            &cached.slots,
            query_idx,
            &prob_bufs,
            &mut grad_bufs,
            cfg,
        )?;

    // Prepare the loss buffer metadata while still GIL-free
    Ok((loss_dev.into_bytes(), 1usize))
}).map_err(|e| PyRuntimeError::new_err(format!("Neural fast-path error: {}", e)))?;
```

Then reconstruct the `CudaBuffer` after reacquiring the GIL.

**NOTE:** This requires that `prob_bufs`, `grad_bufs`, `cached` all are `Send`. Check that `CudaBuffer` and `ExactDdnnfProgram` implement `Send`. If not, this task may need to be deferred or implemented differently (e.g., using `unsafe impl Send`).

**Step 3: Test**

```bash
cargo build --release -p pyxlog
# Run the Python test suite
cd /home/dev/projects/xlog && python -m pytest python/tests/ -v
```

**Step 4: Commit**

```bash
git add crates/pyxlog/src/lib.rs
git commit -m "perf(pyxlog): release GIL during GPU neural fast-path execution"
```

---

## Task 6: End-to-end benchmark validation

**Files:**
- Modify: `examples/neural/01_minimal/train.py` (add timing instrumentation)

**Step 1: Run baseline timing (before fixes)**

Before applying ANY fixes, run a short training benchmark to establish baseline:

```bash
cd /home/dev/projects/xlog
python examples/neural/01_minimal/train.py \
    --epochs 1 --batch-size 4 --train-limit 8 --seed 42 \
    2>&1 | tee /tmp/baseline_timing.txt
```

Record: epoch time, per-query latency, GPU utilization (`nvidia-smi`).

**Step 2: Run post-fix timing**

After applying Tasks 1-5:

```bash
python examples/neural/01_minimal/train.py \
    --epochs 1 --batch-size 4 --train-limit 8 --seed 42 \
    2>&1 | tee /tmp/postfix_timing.txt
```

**Step 3: Compare**

Expected improvements:
- Per-query latency: ~16s → <1s (conservative) or <100ms (optimistic)
- GPU memory: 1GB → 32GB allocated
- Kernel launches per query: ~393K → ~20

**Step 4: Run cache ablation**

```bash
python scripts/cache_ablation.py --epochs 2 --train-limit 16 --seed 42
```

**Step 5: Commit timing results**

```bash
git add examples/neural/results/
git commit -m "evidence: post-perf-fix timing results (fused backward + no sync)"
```

---

## Execution Order & Dependencies

```
Task 0 (cherry-pick) ─── must be first
    │
    ├── Task 1 (fused kernel) ─── Task 2 (wire fused path)
    │                                  │
    │                              Task 3 (remove sync)
    │
    ├── Task 4 (memory defaults) ─── independent
    │
    ├── Task 5 (GIL release) ─── independent, may defer if Send issues
    │
    └── Task 6 (benchmark) ─── must be last
```

Tasks 1→2→3 are sequential (each depends on the previous).
Tasks 4 and 5 are independent of each other and of the kernel work.
Task 6 validates everything.

---

## Risk Notes

1. **Fused backward kernel correctness** — The gradient formulas in the fused kernel MUST exactly match the existing per-level kernels. The reference code in Task 1 is a structural guide; verify every gradient computation against `xgcf_backward_level_propagate_cached`, `xgcf_backward_level_decision_grad_cached`, and `xgcf_backward_level_lit_grad_cached` in `circuit.cu`. Mismatches here will produce wrong gradients (silent correctness bug).

2. **atomicAdd precision** — The fused kernel uses `atomicAdd` for f64 on GPU. This requires `sm_60+` (Pascal or later). Verify target architecture supports native f64 atomics.

3. **Memory default change** — 32GB default may fail on GPUs with <32GB. The `GpuMemoryManager` should clamp to available memory, but verify this code path works. May need to query device memory and use 80% of available.

4. **Send bounds for GIL release** — `CudaBuffer` and related types may not implement `Send`. If so, Task 5 needs a different approach (e.g., restructure to move data before/after the GIL release boundary).

5. **Cherry-pick conflicts** — The worktree's `lib.rs` has different neural registry code. Expect merge conflicts in `crates/pyxlog/src/lib.rs` during Task 0.
