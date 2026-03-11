# GPU-Native Zero Host-IO Phase 4 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate all host reads in the GPU-native exact + MC paths by moving dynamic metadata, random-var discovery, smoothing checks, and query/evidence evaluation fully onto the GPU, and by feature-gating host-output APIs.

**Architecture:** Keep device-resident metadata for D4 compile, smoothing, and cache evaluation; move random-var list construction to GPU compaction; and route MC query/evidence truth evaluation entirely through device kernels. Host read APIs are behind a `host-io` feature that is OFF by default, so production GPU-native paths remain device-only.

**Tech Stack:** Rust (xlog-prob/xlog-cuda/xlog-cli/pyxlog), CUDA kernels (`kernels/*.cu` -> `.ptx`), cudarc.

---

### Task 1: Add failing “no DTOH” regression tests for GPU-native compile + MC

**Files:**
- Create: `crates/xlog-prob/tests/no_dtoh_gpu_native.rs`

**Step 1: Write the failing test**

```rust
use std::sync::Arc;
use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_prob::compilation::{compile_gpu_d4_and_verify, GpuCompileConfig};
use xlog_prob::mc::{McEvalConfig, McProgram};
use xlog_solve::{Clause, GpuCnf, Literal, SolveInstance};

fn try_provider() -> Option<Arc<CudaKernelProvider>> {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {}", e);
            return None;
        }
    };
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), MemoryBudget::with_limit(1 << 30)));
    match CudaKernelProvider::new(device, memory) {
        Ok(p) => Some(Arc::new(p)),
        Err(e) => {
            eprintln!("Skipping test: failed to create CUDA kernel provider: {}", e);
            None
        }
    }
}

#[test]
fn no_dtoh_in_gpu_d4_compile_and_smooth() {
    let Some(provider) = try_provider() else { return; };
    let instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let cnf = GpuCnf::from_host(&instance, &provider).expect("GpuCnf upload");
    provider.reset_host_transfer_stats();

    let config = GpuCompileConfig {
        frontier_depth: 0,
        max_frontier_items: 8,
        max_depth: 32,
        smooth_node_cap: 1024,
        smooth_edge_cap: 4096,
        cdcl_restart_interval: 32,
        cdcl_learned_bytes: 4 * 1024 * 1024,
        cdcl_conflict_budget: None,
    };

    let _ = compile_gpu_d4_and_verify(&cnf, &provider, &config, &[1]).expect("compile must succeed");
    let stats = provider.host_transfer_stats();
    assert_eq!(stats.dtoh_calls, 0, "DTOH calls must be 0 in GPU D4 compile path");
    assert_eq!(stats.dtoh_bytes, 0, "DTOH bytes must be 0 in GPU D4 compile path");
}

#[test]
fn no_dtoh_in_mc_gpu_device_eval() {
    let source = r#"
0.3::rain().
dry() :- not rain().
query(dry()).
"#;
    let program = match McProgram::compile_source_with_gpu(source, Default::default()) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Skipping test: MC compile failed: {}", e);
            return;
        }
    };

    let cfg = McEvalConfig { samples: 128, seed: 1, confidence: 0.95, max_nonmonotone_iterations: 128 };
    let provider = program.provider().expect("provider");
    provider.reset_host_transfer_stats();
    let _ = program.evaluate_gpu_device(cfg).expect("gpu mc device eval");
    let stats = provider.host_transfer_stats();
    assert_eq!(stats.dtoh_calls, 0, "DTOH calls must be 0 in MC device eval");
    assert_eq!(stats.dtoh_bytes, 0, "DTOH bytes must be 0 in MC device eval");
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-prob no_dtoh_gpu_native -- --nocapture`  
Expected: FAIL (DTOH calls/bytes > 0).

**Step 3: Commit the failing test**

```bash
git add crates/xlog-prob/tests/no_dtoh_gpu_native.rs
git commit -m "test(gpu): add no-dtoh guardrails for native compile and mc eval"
```

---

### Task 2: Remove DTOH from GPU D4 compile metadata capture

**Files:**
- Modify: `kernels/d4.cu`
- Modify: `kernels/d4.ptx` (regenerate)
- Modify: `crates/xlog-cuda/src/provider/mod.rs`
- Modify: `crates/xlog-prob/src/compilation/gpu_d4.rs`
- Modify: `crates/xlog-prob/src/gpu.rs` (GpuCircuitLayout / meta usage)

**Step 1: Write a failing unit test for device-only meta**

Add a small test to `crates/xlog-prob/tests/no_dtoh_gpu_native.rs` to assert that the D4 compile path no longer touches any `dtoh_sync_copy_into` sites in `gpu_d4.rs`:

```rust
#[test]
fn d4_compile_no_dtoh_calls_in_source() {
    let text = std::fs::read_to_string("crates/xlog-prob/src/compilation/gpu_d4.rs")
        .expect("read gpu_d4.rs");
    assert!(!text.contains("dtoh_sync_copy_into"), "gpu_d4.rs still contains DTOH calls");
}
```

Run: `cargo test -p xlog-prob d4_compile_no_dtoh_calls_in_source -- --nocapture`  
Expected: FAIL.

**Step 2: Add device-side meta capture kernel**

In `kernels/d4.cu`, add a kernel that reads the final offsets/counts and frontier size, computes totals, and:
- writes `num_nodes`, `num_edges`, `frontier_size` into device meta buffers
- validates caps and traps on overflow using `asm("trap;")`

Example kernel signature:

```cuda
extern "C" __global__ void d4_capture_emit_meta(
    const uint32_t* __restrict__ node_offsets,
    const uint32_t* __restrict__ node_counts,
    const uint32_t* __restrict__ edge_offsets,
    const uint32_t* __restrict__ edge_counts,
    uint32_t last_idx,
    const uint32_t* __restrict__ frontier_size,
    uint32_t node_cap,
    uint32_t edge_cap,
    uint32_t* __restrict__ out_num_nodes,
    uint32_t* __restrict__ out_num_edges,
    uint32_t* __restrict__ out_frontier
);
```

**Step 3: Wire kernel in provider**

In `crates/xlog-cuda/src/provider/mod.rs`:
- Add a `d4_kernels::D4_CAPTURE_EMIT_META` constant.
- Include it in the kernel resolution list for the D4 module.

**Step 4: Use device meta in `gpu_d4.rs`**

In `crates/xlog-prob/src/compilation/gpu_d4.rs`:
- Allocate device meta buffers for `num_nodes`, `num_edges`, `frontier_size`.
- Replace DTOH reads of `node_offsets/node_counts/edge_offsets/edge_counts/frontier.size_device` with the new kernel.
- Remove host-side `actual_nodes/actual_edges` checks; rely on device trap.
- Pass device meta into the layout/`GpuXgcf` construction path (see Task 4).

**Step 5: Run tests**

Run: `cargo test -p xlog-prob d4_compile_no_dtoh_calls_in_source -- --nocapture`  
Expected: PASS.

Run: `cargo test -p xlog-prob no_dtoh_in_gpu_d4_compile_and_smooth -- --nocapture`  
Expected: still FAIL (smoothing + random-var list still DTOH; will be fixed later).

**Step 6: Commit**

```bash
git add kernels/d4.cu kernels/d4.ptx crates/xlog-cuda/src/provider/mod.rs crates/xlog-prob/src/compilation/gpu_d4.rs
git commit -m "feat(d4): capture compile metadata on device and remove dtoh reads"
```

---

### Task 3: Remove DTOH from GPU smoothing (wrapper counts + edge caps)

**Files:**
- Modify: `kernels/d4.cu`
- Modify: `kernels/d4.ptx` (regenerate)
- Modify: `crates/xlog-prob/src/gpu.rs`

**Step 1: Write a failing test**

Extend `crates/xlog-prob/tests/no_dtoh_gpu_native.rs`:

```rust
#[test]
fn smoothing_no_dtoh_calls_in_source() {
    let text = std::fs::read_to_string("crates/xlog-prob/src/gpu.rs").expect("read gpu.rs");
    assert!(!text.contains("wrap_counts_host"), "gpu.rs still uses host wrap_counts");
    assert!(!text.contains("Failed to read smoothed edges"), "gpu.rs still reads smoothed edges");
}
```

Run: `cargo test -p xlog-prob smoothing_no_dtoh_calls_in_source -- --nocapture`  
Expected: FAIL.

**Step 2: Make smoothing checks device-only**

In `kernels/d4.cu`:
- Extend `d4_smooth_check_edge_cap` (or add a new kernel) to:
  - read `out_child_offsets[smooth_node_cap]` (total edges),
  - trap if `total_edges == 0` or `total_edges > smooth_edge_cap`,
  - optionally write `total_edges` into a device meta buffer if needed later.
- Avoid any host reads of `wrap_counts` or `total_edges`.

**Step 3: Update `smooth_random_vars_device`**

In `crates/xlog-prob/src/gpu.rs`:
- Remove DTOH reads of `wrap_counts` and `total_edges`.
- Use `smooth_node_cap` to index the final prefix-scan entry (device-side check handles bounds).
- Keep `wrapper_base = base_node + num_nodes` from host inputs (control-plane OK).

**Step 4: Run tests**

Run: `cargo test -p xlog-prob smoothing_no_dtoh_calls_in_source -- --nocapture`  
Expected: PASS.

**Step 5: Commit**

```bash
git add kernels/d4.cu kernels/d4.ptx crates/xlog-prob/src/gpu.rs
git commit -m "feat(d4): make GPU smoothing cap checks device-only"
```

---

### Task 4: Device-only random-var list + device meta layout

**Files:**
- Modify: `kernels/filter.cu` (or new CUDA util file)
- Modify: `kernels/filter.ptx` (regenerate)
- Modify: `crates/xlog-cuda/src/provider/mod.rs`
- Modify: `crates/xlog-prob/src/exact.rs`
- Modify: `crates/xlog-prob/src/gpu.rs`
- Modify: `crates/xlog-prob/src/compilation/gpu_cache.rs`

**Step 1: Write a failing test**

Add to `crates/xlog-prob/tests/no_dtoh_gpu_native.rs`:

```rust
#[test]
fn random_var_collection_no_dtoh_in_source() {
    let text = std::fs::read_to_string("crates/xlog-prob/src/exact.rs").expect("read exact.rs");
    assert!(!text.contains("collect_random_vars_host"), "host random-var collection still present");
}
```

Run: `cargo test -p xlog-prob random_var_collection_no_dtoh_in_source -- --nocapture`  
Expected: FAIL.

**Step 2: Add device kernels for random-var list**

In `kernels/filter.cu` (or `kernels/d4.cu` if you prefer module locality), add:
- `fill_u32_iota(out, n, base)` to build a [1..max_var] index array
- `mark_random_vars(leaf_var, choice_var, leaf_len, choice_len, out_mask, mask_len)` to set mask[var]=1

Use existing scan + `compact_u32_by_mask` to emit the device `random_var_list`.

**Step 3: Wire kernel names in provider**

Add new kernel name constants and module resolution in `crates/xlog-cuda/src/provider/mod.rs`.

**Step 4: Replace `collect_random_vars_host`**

In `crates/xlog-prob/src/exact.rs`:
- Replace `collect_random_vars_host` with `collect_random_vars_device` returning `TrackedCudaSlice<u32>` plus a device count (or a struct with `list` + `count`).
- Store the device list in `ExactDdnnfProgram` (e.g., `random_vars_device`) instead of `Vec<u32>`.

**Step 5: Update smoothing API to accept device list**

In `crates/xlog-prob/src/gpu.rs`:
- Update `GpuXgcf::smooth_random_vars_device` to accept `(random_var_list_device, random_var_count)` and to build the random-var-to-bit mapping on GPU.
- Remove any host-side validation that requires downloading the list.

**Step 6: Update cache store/eval to rely on device meta**

In `crates/xlog-prob/src/compilation/gpu_cache.rs`:
- Add a new path that uses device meta (num_nodes/num_levels/root/max_var) instead of host scalars.
- Use `xgcf_eval_all_levels_cached` and `xgcf_copy_root_cached_meta` in evaluation paths.
- For backward propagation, loop `level_cap` on host and make kernels guard on `level < meta_num_levels[slot]` to avoid host meta reads.

**Step 7: Run tests**

Run: `cargo test -p xlog-prob random_var_collection_no_dtoh_in_source -- --nocapture`  
Expected: PASS.

**Step 8: Commit**

```bash
git add kernels/filter.cu kernels/filter.ptx crates/xlog-cuda/src/provider/mod.rs \
  crates/xlog-prob/src/exact.rs crates/xlog-prob/src/gpu.rs crates/xlog-prob/src/compilation/gpu_cache.rs
git commit -m "feat(gpu): device-only random-var list and cache meta evaluation"
```

---

### Task 5: GPU-native MC defaults (device-only query/evidence evaluation)

**Files:**
- Modify: `kernels/mc_eval.cu`
- Modify: `kernels/mc_eval.ptx` (regenerate)
- Modify: `crates/xlog-cuda/src/provider/mod.rs`
- Modify: `crates/xlog-prob/src/mc.rs`

**Step 1: Write a failing test**

Extend `crates/xlog-prob/tests/no_dtoh_gpu_native.rs`:

```rust
#[test]
fn mc_device_eval_no_host_query_truth() {
    let text = std::fs::read_to_string("crates/xlog-prob/src/mc.rs").expect("read mc.rs");
    assert!(!text.contains("query_truth"), "mc.rs still builds host query_truth for device eval");
}
```

Run: `cargo test -p xlog-prob mc_device_eval_no_host_query_truth -- --nocapture`  
Expected: FAIL.

**Step 2: Add device kernels for query/evidence truth**

In `kernels/mc_eval.cu`, add a kernel that:
- Accepts arrays of device row-count pointers for query relations and evidence relations,
- Writes `query_truth` (u8) and `evidence_ok` (u8) purely on GPU.

Example signature:

```cuda
extern "C" __global__ void mc_eval_query_evidence_truth(
    const uint32_t** __restrict__ query_counts_ptrs,
    uint32_t query_count,
    const uint32_t** __restrict__ evidence_counts_ptrs,
    const uint8_t* __restrict__ evidence_expected,
    uint32_t evidence_count,
    uint8_t* __restrict__ out_query_truth,
    uint8_t* __restrict__ out_evidence_ok
);
```

**Step 3: Wire kernel in provider**

Add kernel name and include in module resolution list.

**Step 4: Update `evaluate_gpu_device` path**

In `crates/xlog-prob/src/mc.rs`:
- Prebuild device arrays of `d_num_rows` pointers for query and evidence relations.
- Replace host `query_truth`/`evidence_ok` logic with the new kernel.
- Update `evaluate_gpu_device` and `evaluate_gpu_counts_with` to accumulate counts using only device buffers.

**Step 5: Run tests**

Run: `cargo test -p xlog-prob mc_device_eval_no_host_query_truth -- --nocapture`  
Expected: PASS.

Run: `cargo test -p xlog-prob no_dtoh_in_mc_gpu_device_eval -- --nocapture`  
Expected: PASS.

**Step 6: Commit**

```bash
git add kernels/mc_eval.cu kernels/mc_eval.ptx crates/xlog-cuda/src/provider/mod.rs crates/xlog-prob/src/mc.rs
git commit -m "feat(mc): device-only query/evidence evaluation for gpu-native MC"
```

---

### Task 6: Feature-gate host-output APIs (`host-io`) and update call sites

**Files:**
- Modify: `crates/xlog-prob/Cargo.toml`
- Modify: `crates/xlog-prob/src/exact.rs`
- Modify: `crates/xlog-prob/src/gpu.rs`
- Modify: `crates/xlog-prob/src/mc.rs`
- Modify: `crates/xlog-cli/Cargo.toml`
- Modify: `crates/xlog-cli/src/main.rs`
- Modify: `crates/pyxlog/Cargo.toml`
- Modify: `crates/pyxlog/src/lib.rs`

**Step 1: Write a failing compile-time guard test**

Add to `crates/xlog-prob/tests/no_dtoh_gpu_native.rs`:

```rust
#[test]
fn host_io_feature_required_for_dtoh_apis() {
    let text = std::fs::read_to_string("crates/xlog-prob/src/gpu.rs").expect("read gpu.rs");
    assert!(text.contains("cfg(feature = \"host-io\")"), "host-io feature gates missing for host DTOH APIs");
}
```

Run: `cargo test -p xlog-prob host_io_feature_required_for_dtoh_apis -- --nocapture`  
Expected: FAIL.

**Step 2: Add `host-io` feature**

In `crates/xlog-prob/Cargo.toml`:

```toml
[features]
host-io = []
```

**Step 3: Gate host-read APIs**

In `crates/xlog-prob/src/gpu.rs` and `crates/xlog-prob/src/exact.rs`:
- Gate host-read functions (`eval_log_wmc`, `eval_log_wmc_and_grads`, `ExactDdnnfProgram::evaluate`, etc.) with `#[cfg(feature = "host-io")]`.
- Introduce device-only alternatives that return device buffers or device results (no DTOH).

In `crates/xlog-prob/src/mc.rs`:
- Gate `evaluate` and `evaluate_gpu` host-returning paths behind `host-io`.
- Keep `evaluate_gpu_device` as the default device-only API.

**Step 4: Update CLI + pyxlog call sites**

In `crates/xlog-cli` and `crates/pyxlog`:
- If `host-io` feature is ON, keep current host-output behavior.
- If `host-io` is OFF, return an explicit error indicating host output is disabled and suggest using device-resident APIs (DLPack paths).

**Step 5: Run tests**

Run: `cargo test -p xlog-prob host_io_feature_required_for_dtoh_apis -- --nocapture`  
Expected: PASS.

**Step 6: Commit**

```bash
git add crates/xlog-prob/Cargo.toml crates/xlog-prob/src/exact.rs crates/xlog-prob/src/gpu.rs crates/xlog-prob/src/mc.rs \
  crates/xlog-cli/Cargo.toml crates/xlog-cli/src/main.rs crates/pyxlog/Cargo.toml crates/pyxlog/src/lib.rs
git commit -m "feat(gpu): gate host-read APIs behind host-io and default to device-only paths"
```

---

### Task 7: Regenerate PTX + run GPU-native certification smoke tests

**Files:**
- Modify (if kernels changed): `kernels/*.ptx`
- Modify (if needed): `crates/xlog-cuda-tests` category files for new kernels

**Step 1: Regenerate PTX**

Run: `cargo build -p xlog-cuda`  
Expected: nvcc rebuilds updated `.ptx` files.

**Step 2: Run targeted tests**

Run: `cargo test -p xlog-prob no_dtoh_gpu_native -- --nocapture`  
Run: `cargo test -p xlog-cuda-tests ptx_validation -- --nocapture`

**Step 3: Commit regenerated PTX and any new test updates**

```bash
git add kernels/*.ptx crates/xlog-cuda-tests
git commit -m "test(cuda): refresh PTX and validate new GPU-native kernels"
```

---

### Task 8: Final verification sweep (no host reads)

**Files:**
- None (verification only)

**Step 1: Run full GPU-native test set**

Run:
```bash
cargo test -p xlog-prob gpu_d4_compile_and_verify gpu_cache_compile_and_verify no_dtoh_gpu_native -- --nocapture
cargo test -p xlog-cuda-tests -- --nocapture
```

**Step 2: Verify DTOH counters remain zero**

Confirm `no_dtoh_gpu_native` tests pass and `HostTransferStats.dtoh_*` remain zero.

**Step 3: Commit if any final fixes were needed**

```bash
git add crates/xlog-prob crates/xlog-cuda kernels
git commit -m "chore(gpu): finalize zero-host-io GPU-native path"
```

---

## Execution Handoff

Plan complete and saved to `docs/plans/2026-01-29-gpu-native-zero-host-io.md`.

Two execution options:

1. **Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration  
2. **Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
