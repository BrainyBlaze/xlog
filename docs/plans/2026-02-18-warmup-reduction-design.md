# Warmup Reduction Design: First-Epoch from ~72s to <20s

**Date:** 2026-02-18
**Branch:** v0.4.0-alpha-integrated
**Status:** Approved design, pending implementation plan

## Problem

After enabling batched circuit evaluation, steady-state per-query cost dropped from
10.3ms to 0.97ms (XLOG now 1.5x faster than Scallop). But total training time is still
dominated by first-epoch warmup: ~72s vs Scallop's 0.72s. The warmup has two suspected
components — CUDA PTX JIT compilation and d-DNNF circuit compilation — but no
instrumentation exists to attribute time to either.

**Target:** Cut first-epoch warmup from ~72s to <20s while preserving current
steady-state performance (0.248s/epoch).

## Approach: Measurement-Gated Optimization

Four phases, with a decision gate after Phase 0 to prioritize based on data.

```
Phase 0: Instrument warmup (cold vs warm attribution)
    |
    v
Decision gate: which component is dominant?
    |
    +-- JIT dominant (>20s) --> Phase 1 first (cubin)
    +-- D4 dominant (>20s)  --> Phase 2 first (circuit cache)
    +-- Both substantial    --> larger first
    |
    v
Phase 1: Cubin support (eliminate PTX JIT)
Phase 2: Circuit artifact cache (skip d-DNNF recompilation)
Phase 3: Metrics changes (split reporting)
```

---

## Phase 0: Warmup Instrumentation

### Goal

Produce cold-start vs warm-start timing breakdown that cleanly separates PTX JIT from
D4/verify/smooth/store. Run both cold (after clearing `~/.nv/ComputeCache/`) and warm
to isolate driver JIT cost.

### Instrumentation Sites

**1. PTX/kernel load timing — `crates/xlog-cuda/src/provider.rs`**

Wrap the 19 `load_ptx()` calls in `CudaKernelProvider::new()` (lines 667-1116) with
`Instant`-based timing. Record total `ptx.total_sec` and per-module breakdown.

Gate: runtime env flag `XLOG_WARMUP_PROFILE=1` only. No `cfg(debug_assertions)`.
When profiling is off, the default path is unchanged — no synchronization overhead.

**2. Circuit compilation timing — `crates/xlog-prob/src/compilation/mod.rs`**

Instrument each stage of `compile_gpu_d4_and_verify_cached()` (lines 96-275):
- `cnf_hash_sec` — `hash_cnf_gpu()`
- `d4_compile_sec` — `compile_gpu_d4_gated()`
- `verify_sec` — `validate_equivalence_gpu_gated()`
- `smooth_sec` — `smooth_random_vars_device()`
- `cache_store_sec` — `store_from_xgcf()` + free-var mask

Add `device().synchronize()` before/after each stage **only when profiling flag is on**.
Default path retains current behavior (no extra syncs).

Include cache-hit/miss labels:
- `gpu_cache_hit: bool` — from `lookup_or_insert_device` result
- At pyxlog level: `template_cache_hit: bool` — whether `circuit_cache` had the entry

**3. Template-level timing — `crates/pyxlog/src/lib.rs`**

In `compile_circuit_for_template()` (lines 2721-2780): wrap with timing on first cache
miss per template shape.

### Cold vs Warm Protocol

Run benchmark script twice:
1. Cold: `rm -rf ~/.nv/ComputeCache/ && XLOG_WARMUP_PROFILE=1 python ...`
2. Warm: `XLOG_WARMUP_PROFILE=1 python ...` (immediately after)

The delta in `ptx.total_sec` is pure driver JIT. The warm residual is module loading
overhead.

### Output Format

```json
"warmup_breakdown": {
  "run_metadata": {
    "driver_version": "570.86.10",
    "sm": "sm_120",
    "cuda_runtime": "13.1",
    "compute_cache": "cold"
  },
  "ptx": {
    "total_sec": 35.2,
    "cubin_loaded": 0,
    "ptx_fallback": 19,
    "per_module_sec": {
      "xlog_circuit": 4.1,
      "xlog_sat": 3.8,
      "xlog_d4": 3.5
    }
  },
  "circuit": {
    "template_cache_hit": false,
    "gpu_cache_hit": false,
    "disk_cache_hit": false,
    "d4_compile_sec": 28.1,
    "verify_sec": 5.3,
    "smooth_sec": 1.2,
    "cache_store_sec": 0.8
  }
}
```

### Decision Gate

After Phase 0 produces cold/warm numbers:
- If `ptx.total_sec(cold) - ptx.total_sec(warm) > 20s` → Phase 1 (cubin) next
- If `circuit.d4_compile_sec > 20s` → Phase 2 (circuit cache) next
- If both substantial → start with whichever is larger

---

## Phase 1: Cubin Support

### Goal

Eliminate CUDA driver JIT by loading pre-compiled cubin when device SM matches.
Fall back to PTX (portable) on unknown GPUs.

### Pre-step: Fix Kernel Manifest Source of Truth

Currently the kernel list is duplicated:
- `build.rs` (line 17): 18 kernels — **missing `cnf`**
- `provider.rs` (lines 663-1116): 19 `load_ptx()` calls with inline kernel name lists

SM arch is inconsistent:
- `build.rs:61`: hardcoded `sm_70`
- `bin/nvrtc_ptx.rs:27`: hardcoded `sm_70`
- `provider.rs:20,648,10268`: comments/references to `sm_70`
- Checked-in PTX files: actually target `sm_120`

**Fix:** Create `src/kernel_manifest.rs` with a single const array of
`(module_name, ptx_filename, &[kernel_entry_points])`. Both `build.rs` and
`provider.rs` consume this manifest. Clean up all stale `sm_70` references.

### Build-Time (`build.rs`)

For each of the 19 modules in the manifest, for each arch in `XLOG_CUBIN_ARCHS`
(default `sm_120`):

```
nvcc --cubin -arch=sm_{cc} -o $OUT_DIR/{name}.sm_{cc}.cubin kernels/{name}.cu
```

Separately (once per module, **outside** the per-arch loop), generate one portable PTX
targeting a baseline SM as the fallback for unknown GPUs:

```
nvcc --ptx -arch=sm_70 -o $OUT_DIR/{name}.portable.ptx kernels/{name}.cu
```

This replaces the current repo-embedded PTX which targets `sm_120` and would fail on
non-Blackwell hardware. The portable PTX is generated only once per module regardless
of how many architectures are listed in `XLOG_CUBIN_ARCHS`.

Env vars:
- `XLOG_NO_CUBIN=1` — skip cubin generation (dev builds without nvcc)
- `XLOG_CUBIN_ARCHS=sm_120,sm_89` — comma-separated SM targets

Rebuild triggers:
- `cargo:rerun-if-env-changed=XLOG_NO_CUBIN`
- `cargo:rerun-if-env-changed=XLOG_CUBIN_ARCHS`
- `cargo:rerun-if-changed=kernels/{name}.cu` for each module

### Runtime Loader (`provider.rs`)

1. Detect device compute capability:
   ```rust
   let major = device.inner().attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
   let minor = device.inner().attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
   let cc = major * 10 + minor;  // e.g. 120, 89
   ```

2. Module resolution order per module:
   1. `$XLOG_CUBIN_DIR/{name}.sm_{cc}.cubin` (explicit override for packaged installs)
   2. `$OUT_DIR/{name}.sm_{cc}.cubin` (build output)
   3. `$XLOG_CUBIN_DIR/{name}.portable.ptx` or `$OUT_DIR/{name}.portable.ptx`
      (portable baseline PTX, built targeting `sm_70`)

   **Note:** The current embedded PTX (`include_str!()` in provider.rs) targets
   `sm_120` and is NOT portable. Phase 1 replaces this with file-based loading from
   build outputs. The embedded PTX constants become dead code and should be removed.

3. Load cubin via `Ptx::from_file(path)` (cudarc 0.12 supports this for module files).
   PTX fallback uses `Ptx::from_file()` for the portable PTX, same resolution strategy.

4. When profiling flag is on, log per-module: cubin vs PTX, load time.

### Rationale: `XLOG_CUBIN_DIR` Override

Loading from `OUT_DIR` is fragile for packaged installs (maturin wheels, pip install).
The `XLOG_CUBIN_DIR` env var lets packaged distributions place cubins at a known path.
PTX fallback ensures unknown GPUs or missing cubins never cause hard failures.

---

## Phase 2: Circuit Artifact Cache

### Goal

Cache compiled circuit topology on disk so d-DNNF compilation is skipped on warm
starts. Cache topology and metadata only — query-specific weights are computed per-query
at runtime.

### Free-Var Mask: Per-Slot Metadata (Prerequisite)

The current `has_free_var_mask` flag is **global** on `GpuCircuitCache`
(`gpu_cache.rs:61`), not per-slot. With 4 cache slots, a global flag is unsafe:
compiling/restoring a zero-mask circuit into one slot and clearing the flag disables
free-var correction for another slot that genuinely needs it.

**Required fix (prerequisite for Phase 2, recommended in Phase 1):** Make
`has_free_var_mask` **per-slot metadata**, stored alongside existing per-slot fields
(`meta_num_nodes`, `meta_num_edges`, etc.) in `GpuCircuitCache`.

Concretely:
1. Replace `has_free_var_mask: bool` (global) with `has_free_var_mask: [bool; NUM_SLOTS]`
2. In `store_from_xgcf()` and the new disk-cache restore path:
   a. Zero the `free_var_mask` region for the target slot (`memset_zeros`)
   b. Store topology arrays
   c. If mask is non-zero: H→D copy mask, set `has_free_var_mask[slot] = true`
   d. If mask is all-zero: `has_free_var_mask[slot] = false` (region is clean zeros)
3. In `apply_free_var_correction_cached()`: check `has_free_var_mask[slot]` instead of
   the global flag. The slot index is already available from the LRU lookup.
4. In `eval_grads_inplace_fused_batched()`: same per-slot check — the batch path uses
   a single slot per batch, so the slot index is known.

This is a small change (one bool → array of bools, two check-sites updated) and
eliminates the cross-slot interference that makes the global flag unsafe.

### What Gets Cached

Everything from `store_from_xgcf()` except weights:
- `node_type`, `child_offsets`, `child_indices`, `lit`, `decision_var`,
  `decision_child_false`, `decision_child_true`, `level_nodes`, `level_offsets`
- Metadata: `num_nodes`, `num_edges`, `num_levels`, `root`, `max_var`
- `free_var_mask` — **always stored** in the cache artifact (even if all-zero), so the
  restore path can write it to the slot and set `has_free_var_mask[slot]` correctly

Weights (`var_log_true`, `var_log_false`) are NOT cached — they change per query.

### Cache Key

`(cnf_hash: u64, compile_config_hash: u64, random_vars_hash: u64, sm: u32, format_version: u32)`

- `cnf_hash`: existing GPU-computed hash (D→H copy, 8 bytes)
- `compile_config_hash`: hash of `GpuCompileConfig` fields (caps, smoothing config)
- `random_vars_hash`: hash of random variable list count + contents — smoothing in
  `compile_gpu_d4_and_verify_cached()` (mod.rs:120) depends on this
- `sm`: device compute capability (circuit layout may differ by arch)
- `format_version`: monotonic integer — bump on any layout change; stale entries ignored

### Storage

**Location:** `~/.cache/xlog/circuits/{key_hex}.bin`

**File format:** Flat binary, no serde:
```
[magic: u32 = 0x584C4743]  "XLGC"
[format_version: u32]
[cnf_hash: u64]
[config_hash: u64]
[random_vars_hash: u64]
[sm: u32]
[num_nodes: u32]
[num_edges: u32]
[num_levels: u32]
[root: u32]
[max_var: u32]
[has_free_var_mask: u8]
[padding: 3 bytes]
--- arrays in fixed order, raw bytes ---
[node_type: u8 x num_nodes]
[child_offsets: u32 x (num_nodes+1)]
[child_indices: u32 x num_edges]
[lit: i32 x num_nodes]
[decision_var: u32 x num_nodes]
[decision_child_false: u32 x num_nodes]
[decision_child_true: u32 x num_nodes]
[level_nodes: u32 x num_nodes]
[level_offsets: u32 x (num_levels+1)]
[free_var_mask: u8 x (max_var+1)]  -- always present (may be all-zero)
```

### Write Path (after successful compilation)

1. D→H copy all topology arrays (one-time cost — acceptable after seconds of D4 work)
2. Write to `$TMPDIR/{random}.tmp`
3. Atomic rename to `~/.cache/xlog/circuits/{key_hex}.bin`

### Read Path (before D4 compilation)

Integration point: between `lookup_or_insert_device` (mod.rs:146) and
`compile_gpu_d4_gated` (mod.rs:151).

1. After `lookup_or_insert_device`, D→H copy `compile_needed` (4 bytes)
2. If `compile_needed == 0` (GPU cache hit): skip disk cache entirely
3. If `compile_needed == 1`:
   a. D→H copy `cnf_hash` (8 bytes), compute full cache key
   b. Check if `~/.cache/xlog/circuits/{key_hex}.bin` exists
   c. If hit: read file, validate header (magic + version + key match),
      H→D copy arrays into GPU cache slot, skip D4/verify/smooth/store
   d. If miss or stale: proceed with normal D4 compilation, then write cache entry

### Eviction

Max cache size: `XLOG_CIRCUIT_CACHE_MAX_MB` (default 512MB). On write, if total
directory size exceeds limit, delete oldest entries by mtime.

---

## Phase 3: Metrics Changes

### Existing Fields (preserved for compatibility)

`compile_api_sec`, `first_epoch_sec`, `steady_epoch_sec_mean`, `warmup_sec`,
`per_query_ms`, `epoch_sec`, `total_train_sec` — all unchanged.

### New Optional Fields

When `XLOG_WARMUP_PROFILE=1`:
- `warmup_breakdown.run_metadata` — driver version, SM, CUDA runtime, cache state
- `warmup_breakdown.ptx` — total load time, per-module breakdown, cubin vs PTX counts
- `warmup_breakdown.circuit` — per-stage compile times, cache hit/miss labels

### Runner Changes

**`scripts/track_b_runner.py` and `scripts/track_a_runner.py`:**
- Update frozen-key merge logic to **preserve** extra diagnostic fields (currently
  merges only frozen keys and drops extras)
- Add `warmup_breakdown` to the set of preserved-but-optional keys

**Comparison JSON:**
- Add `jit_sec` and `circuit_compile_sec` alongside existing fields when available

---

## Key Invariants

- **Steady-state unchanged:** Phases 1-2 affect only initialization/first-query path.
  Per-query evaluation uses the same cached GPU circuit and batched kernels.
- **Correctness preserved:** Disk cache stores topology computed by the same D4 +
  verification pipeline. Cache key includes all inputs that affect output. Version
  stamp invalidates stale entries. Per-slot `has_free_var_mask` metadata ensures
  free-var correction is applied only to slots that need it, with mask regions
  zeroed on every compile/restore to prevent stale data from prior slot occupants.
- **Graceful degradation:** Cubin miss → PTX fallback. Disk cache miss → full
  recompilation. No hard failure modes.
- **No new build dependencies for default path:** `XLOG_NO_CUBIN=1` skips nvcc
  requirement. Disk cache is opportunistic.

## Files to Modify

### Phase 0
- `crates/xlog-cuda/src/provider.rs` — PTX load timing
- `crates/xlog-prob/src/compilation/mod.rs` — per-stage compile timing
- `crates/pyxlog/src/lib.rs` — template-level timing, profiling flag plumbing

### Phase 1
- `crates/xlog-cuda/src/kernel_manifest.rs` — new, single source of truth
- `crates/xlog-cuda/build.rs` — cubin generation, all 19 modules, env var wiring
- `crates/xlog-cuda/src/provider.rs` — SM detection, cubin/PTX loader
- `crates/xlog-cuda/src/bin/nvrtc_ptx.rs` — clean stale sm_70

### Phase 2
- `crates/xlog-prob/src/compilation/mod.rs` — disk cache check/write integration
- `crates/xlog-prob/src/compilation/disk_cache.rs` — new, serialization/restore logic
- `crates/xlog-prob/src/compilation/gpu_cache.rs` — restore-from-host-arrays method

### Phase 3
- `scripts/track_b_runner.py` — preserve optional diagnostic fields
- `scripts/track_a_runner.py` — same
- `scripts/neural_training.py` — emit breakdown fields
