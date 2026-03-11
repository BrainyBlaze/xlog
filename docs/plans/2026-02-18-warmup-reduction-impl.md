# Warmup Reduction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Cut first-epoch warmup from ~72s to <20s while preserving 0.248s/epoch steady-state.

**Architecture:** Measurement-gated optimization: instrument warmup (Phase 0), then eliminate
CUDA PTX JIT via precompiled cubins (Phase 1), skip d-DNNF recompilation via disk cache (Phase 2),
and update metric reporting (Phase 3). Decision gate after Phase 0 determines Phase 1 vs 2 priority.

**Tech Stack:** Rust (cudarc 0.12, xlog-cuda, xlog-prob), Python (scripts), CUDA (nvcc cubins).

**Design doc:** `docs/plans/2026-02-18-warmup-reduction-design.md`

---

## Task 1: Per-Slot `has_free_var_mask` (Prerequisite)

The current `has_free_var_mask: bool` on `GpuCircuitCache` is global. With 4 slots, a
global flag is unsafe: clearing it for a zero-mask slot disables correction for a
different slot that genuinely needs it. This is a prerequisite for Phase 2 (disk cache
restore path needs per-slot mask state) and a correctness fix for Phase 1.

**Files:**
- Modify: `crates/xlog-prob/src/compilation/gpu_cache.rs`

### Step 1: Add host-side `slot_host` to `GpuCircuitCacheHandle`

The slot index is assigned at lookup time (off the hot path) but currently only exists
on device. Add a host-resident copy so hot-path code never needs a D→H sync.

In `gpu_cache.rs:96-104`, add `slot_host: u32`:

```rust
pub struct GpuCircuitCacheHandle {
    provider: Arc<CudaKernelProvider>,
    slot: TrackedCudaSlice<u32>,
    compile_needed: TrackedCudaSlice<u32>,
    slot_host: u32,           // <-- new: host-side slot index
    num_nodes: u32,
    num_levels: u32,
    root: u32,
    max_var: u32,
}
```

Add accessor:

```rust
    pub fn slot_index(&self) -> u32 {
        self.slot_host
    }
```

### Step 2: Populate `slot_host` during `into_handle()`

In `GpuCacheLookup::into_handle()` (gpu_cache.rs:83-93), do one D→H copy of the slot
index. This happens at lookup time (once per compilation, off the eval hot path):

```rust
    pub fn into_handle(self) -> Result<GpuCircuitCacheHandle> {
        let slot_host_vec: Vec<u32> = self.provider.device().inner()
            .dtoh_sync_copy(&self.slot)
            .map_err(|e| XlogError::Kernel(format!("dtoh slot index: {}", e)))?;
        Ok(GpuCircuitCacheHandle {
            provider: self.provider,
            slot: self.slot,
            compile_needed: self.compile_needed,
            slot_host: slot_host_vec[0],
            num_nodes: 0,
            num_levels: 0,
            root: 0,
            max_var: 0,
        })
    }
```

**Note:** `into_handle()` return type changes from `GpuCircuitCacheHandle` to
`Result<GpuCircuitCacheHandle>`. Update all callers (`lookup.into_handle()` →
`lookup.into_handle()?`).

### Step 3: Change `has_free_var_mask` field from `bool` to `Vec<bool>`

In `gpu_cache.rs:61`, change:

```rust
has_free_var_mask: bool,
```

to:

```rust
has_free_var_mask: Vec<bool>,
```

### Step 4: Initialize per-slot in `new()` constructor

In `gpu_cache.rs:802`, change:

```rust
            has_free_var_mask: false,
```

to:

```rust
            has_free_var_mask: vec![false; config.num_slots as usize],
```

### Step 5: Update `has_free_var_mask()` accessor

In `gpu_cache.rs:225-227`, change:

```rust
    pub fn has_free_var_mask(&self) -> bool {
        self.has_free_var_mask
    }
```

to:

```rust
    pub fn has_any_free_var_mask(&self) -> bool {
        self.has_free_var_mask.iter().any(|&v| v)
    }

    pub fn has_free_var_mask_for_slot(&self, slot: u32) -> bool {
        self.has_free_var_mask
            .get(slot as usize)
            .copied()
            .unwrap_or(false)
    }
```

### Step 6: Update `store_free_var_mask()` to be per-slot

In `gpu_cache.rs:1486`, change:

```rust
        self.has_free_var_mask = true;
```

to:

```rust
        let slot_idx = handle.slot_index() as usize;
        if slot_idx < self.has_free_var_mask.len() {
            self.has_free_var_mask[slot_idx] = true;
        }
```

No D→H sync here — uses the host-side slot index from the handle.

### Step 7: Update `apply_free_var_correction_cached()` to check per-slot

In `gpu_cache.rs:2105`, change:

```rust
        if !self.has_free_var_mask {
            return Ok(());
        }
```

to:

```rust
        if !self.has_free_var_mask_for_slot(handle.slot_index()) {
            return Ok(());
        }
```

No D→H sync — uses the host-side slot index from the handle.

### Step 8: Update `eval_grads_inplace_fused_batched()` to check per-slot

In `gpu_cache.rs:327-331`, change:

```rust
        if self.has_free_var_mask {
            return Err(XlogError::Execution(
                "Batched fused eval currently does not support free-var correction".to_string(),
            ));
        }
```

to:

```rust
        if self.has_free_var_mask_for_slot(handle.slot_index()) {
            return Err(XlogError::Execution(
                "Batched fused eval currently does not support free-var correction".to_string(),
            ));
        }
```

No D→H sync — uses the host-side slot index from the handle.

### Step 9: Fix all callers of the old `has_free_var_mask()` accessor and `into_handle()`

Two search-and-fix passes:

1. `rg 'has_free_var_mask\(\)' crates/` — rename to `has_any_free_var_mask()` or
   `has_free_var_mask_for_slot(slot)` depending on context.
2. `rg 'into_handle\(\)' crates/` — add `?` after each call since it now returns
   `Result<GpuCircuitCacheHandle>`.

### Step 10: Compile and test

Run: `cargo check -p xlog-prob`
Expected: compiles clean

Run: `cargo test -p xlog-prob`
Expected: all tests pass

### Step 11: Commit

```bash
git add crates/xlog-prob/src/compilation/gpu_cache.rs
git commit -m "fix(gpu-cache): make has_free_var_mask per-slot instead of global

A global flag was unsafe with 4 slots: clearing it for a zero-mask
circuit could disable free-var correction for another slot that
genuinely needs it. Now each slot tracks its own mask state.

Host-side slot index stored in GpuCircuitCacheHandle at lookup time
to avoid D→H syncs on the per-query eval hot path."
```

---

## Task 2: Phase 0 — PTX Load Timing in `provider.rs`

Instrument the 19 `load_ptx()` calls with `Instant`-based timing, gated by
`XLOG_WARMUP_PROFILE=1` env var at runtime.

**Files:**
- Modify: `crates/xlog-cuda/src/provider/mod.rs`

### Step 1: Add profiling data structure and env check

At the top of `provider.rs`, after the imports (around line 18), add:

```rust
use std::time::Instant;

/// Per-module PTX load timing (populated only when XLOG_WARMUP_PROFILE=1).
#[derive(Debug, Clone, Default)]
pub struct PtxLoadProfile {
    pub total_sec: f64,
    pub per_module_sec: Vec<(String, f64)>,
    pub cubin_loaded: u32,
    pub ptx_fallback: u32,
}

fn warmup_profiling_enabled() -> bool {
    std::env::var("XLOG_WARMUP_PROFILE").map(|v| v == "1").unwrap_or(false)
}
```

### Step 2: Add profiling field to `CudaKernelProvider`

Find the `CudaKernelProvider` struct definition and add:

```rust
    ptx_load_profile: Option<PtxLoadProfile>,
```

### Step 3: Wrap each `load_ptx()` call with timing

In `CudaKernelProvider::new()` (lines 663-1116), refactor the 19 `load_ptx` calls
into a helper or wrap each with timing. The pattern for each module becomes:

```rust
        let profiling = warmup_profiling_enabled();
        let mut profile = PtxLoadProfile::default();

        // For each module, wrap in timing:
        {
            let t0 = if profiling { Some(Instant::now()) } else { None };
            device.inner().load_ptx(
                Ptx::from_src(JOIN_PTX),
                JOIN_MODULE,
                &[/* kernel names */],
            ).map_err(|e| XlogError::Kernel(format!("Failed to load join PTX: {}", e)))?;
            if let Some(t0) = t0 {
                if profiling {
                    device.inner().synchronize()
                        .map_err(|e| XlogError::Kernel(format!("sync after join load: {}", e)))?;
                }
                let elapsed = t0.elapsed().as_secs_f64();
                profile.per_module_sec.push(("join".to_string(), elapsed));
                profile.total_sec += elapsed;
                profile.ptx_fallback += 1;
            }
        }
```

Apply this pattern to all 19 modules. The `synchronize()` after each load is
**profiling-only** — it forces the driver JIT to complete so timing is accurate.
When profiling is off, no syncs are added.

### Step 4: Store profile in constructor return

Change the constructor to store the profile:

```rust
        Ok(Self {
            device,
            memory,
            transfer_tracker: HostTransferTracker::default(),
            ptx_load_profile: if profiling { Some(profile) } else { None },
        })
```

### Step 5: Add accessor

```rust
    pub fn ptx_load_profile(&self) -> Option<&PtxLoadProfile> {
        self.ptx_load_profile.as_ref()
    }
```

### Step 6: Compile

Run: `cargo check -p xlog-cuda`
Expected: compiles clean

### Step 7: Commit

```bash
git add crates/xlog-cuda/src/provider/mod.rs
git commit -m "feat(warmup): instrument PTX load timing in CudaKernelProvider

Gated by XLOG_WARMUP_PROFILE=1. Records per-module load time with
device synchronize after each load_ptx() to capture JIT cost.
Default path unchanged — no extra syncs when profiling is off."
```

---

## Task 3: Phase 0 — Circuit Compilation Timing in `mod.rs`

Instrument each stage of `compile_gpu_d4_and_verify_cached()` with per-stage timing.

**Files:**
- Modify: `crates/xlog-prob/src/compilation/mod.rs`

### Step 1: Add profiling data structure

At the top of `mod.rs`, add:

```rust
use std::time::Instant;

/// Per-stage compilation timing (populated only when XLOG_WARMUP_PROFILE=1).
#[derive(Debug, Clone, Default)]
pub struct CircuitCompileProfile {
    pub cnf_hash_sec: f64,
    pub d4_compile_sec: f64,
    pub verify_sec: f64,
    pub smooth_sec: f64,
    pub cache_store_sec: f64,
    pub free_var_mask_sec: f64,
    pub gpu_cache_hit: bool,
}

fn warmup_profiling_enabled() -> bool {
    std::env::var("XLOG_WARMUP_PROFILE").map(|v| v == "1").unwrap_or(false)
}
```

### Step 2: Instrument `compile_gpu_d4_and_verify_cached()`

Add timing around each stage in the function (lines 120-275). The function signature
gains an optional output parameter. Rather than changing the public API, return the
profile as a second value:

Change return type from `Result<GpuCircuitCacheHandle>` to
`Result<(GpuCircuitCacheHandle, Option<CircuitCompileProfile>)>`.

Before each stage, capture `Instant::now()` if profiling. After each stage, if profiling,
call `device().synchronize()` and record elapsed time.

Example for the D4 compile stage (lines 150-159):

```rust
    let t_d4 = if profiling { Some(Instant::now()) } else { None };
    let circuit_base =
        gpu_d4::compile_gpu_d4_gated(cnf, provider, &d4_config, handle.compile_needed_device())?;
    if let Some(t0) = t_d4 {
        provider.device().synchronize()
            .map_err(|e| XlogError::Kernel(format!("sync after d4 compile: {}", e)))?;
        profile.d4_compile_sec = t0.elapsed().as_secs_f64();
    }
```

Apply the same pattern for: hash_cnf_gpu, validate_equivalence_gpu_gated,
smooth_random_vars_device, store_from_xgcf + store_free_var_mask.

Set `profile.gpu_cache_hit` based on a D→H read of `compile_needed` when profiling.

### Step 3: Update all callers

Search for callers of `compile_gpu_d4_and_verify_cached` and update them to handle
the new return type. The profile can be ignored (`.0`) when not needed.

Run: `rg 'compile_gpu_d4_and_verify_cached' crates/` to find callers.

### Step 4: Compile and test

Run: `cargo check -p xlog-prob`
Expected: compiles clean

Run: `cargo test -p xlog-prob`
Expected: all tests pass

### Step 5: Commit

```bash
git add crates/xlog-prob/src/compilation/mod.rs
git commit -m "feat(warmup): instrument per-stage circuit compilation timing

Gated by XLOG_WARMUP_PROFILE=1. Records cnf_hash, d4_compile,
verify, smooth, cache_store, and free_var_mask durations with
device synchronize between stages for accurate attribution."
```

---

## Task 4: Phase 0 — Template-Level Timing and JSON Output in `pyxlog`

Surface warmup profile data through pyxlog to the Python training scripts.

**Files:**
- Modify: `crates/pyxlog/src/lib.rs`

### Step 1: Expose PTX profile through pyxlog

**Mutability strategy:** `compile_circuit_for_template()` takes `&self` (line 2722),
but its callers (`forward_backward_complex_tensor` at line 1767,
`forward_backward_batch_complex_tensor` at line 1910) take `&mut self` and are
responsible for inserting into `circuit_cache` and incrementing
`template_compile_count`. Store the latest `CircuitCompileProfile` on
`CompiledProgram` as a new field:

```rust
    /// Latest circuit compilation profile (populated on cache miss when profiling).
    last_compile_profile: Option<CircuitCompileProfile>,
```

In the `&mut self` callers (lines 1767-1772 and 1910-1915), after calling
`compile_circuit_for_template()` and receiving the profile from
`compile_gpu_d4_and_verify_cached()`, store it in `self.last_compile_profile`.
No interior mutability needed — the mutation happens in the `&mut self` methods.

The `PtxLoadProfile` is stored on `CudaKernelProvider` (from Task 2) and accessed
via `self.output_provider.ptx_load_profile()` — this is already `&self`-accessible
through `Arc`.

Add a Python-accessible method:

```rust
    /// Return warmup profiling data as a Python dict (or None if profiling disabled).
    fn warmup_breakdown(&self) -> PyResult<Option<PyObject>> { ... }
```

This method reads the `PtxLoadProfile` from the kernel provider and
`last_compile_profile` from the struct, builds a Python dict matching the JSON
format in the design doc, and returns it.

### Step 2: Wire into Python training scripts

In `scripts/neural_training.py` (or the training example that writes metrics.json),
after training completes, call `engine.warmup_breakdown()` and if non-None, include
it in the metrics dict under `warmup_breakdown`.

### Step 3: Compile and test

Run: `maturin develop -m crates/pyxlog/Cargo.toml --release --features host-io`
Expected: builds clean

Run: `XLOG_WARMUP_PROFILE=1 python examples/neural/01_minimal/train.py --engine xlog --epochs 1 --batch-size 64 --seed 42 --train-limit 512 --data-path examples/neural/01_minimal/data/mnist --metrics-path /tmp/test_metrics.json`
Expected: `/tmp/test_metrics.json` contains `warmup_breakdown` with timing data

### Step 4: Commit

```bash
git add crates/pyxlog/src/lib.rs scripts/neural_training.py
git commit -m "feat(warmup): expose warmup profiling breakdown in pyxlog

When XLOG_WARMUP_PROFILE=1, engine.warmup_breakdown() returns a dict
with PTX load timing, per-module breakdown, and circuit compilation
per-stage timing. Training scripts include this in metrics.json."
```

---

## Task 5: Phase 0 — Run Cold/Warm Profiling and Decision Gate

Execute the cold/warm protocol from the design doc. This is a manual step — the
engineer runs the benchmarks and records the numbers.

### Step 1: Run cold benchmark

```bash
rm -rf ~/.nv/ComputeCache/
XLOG_WARMUP_PROFILE=1 LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH \
  python examples/neural/01_minimal/train.py \
    --engine xlog --epochs 2 --batch-size 64 --seed 42 \
    --train-limit 512 \
    --data-path examples/neural/01_minimal/data/mnist \
    --metrics-path /tmp/cold_metrics.json
```

Record `ptx.total_sec` and `circuit.d4_compile_sec` from `/tmp/cold_metrics.json`.

### Step 2: Run warm benchmark (immediately after)

```bash
XLOG_WARMUP_PROFILE=1 LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH \
  python examples/neural/01_minimal/train.py \
    --engine xlog --epochs 2 --batch-size 64 --seed 42 \
    --train-limit 512 \
    --data-path examples/neural/01_minimal/data/mnist \
    --metrics-path /tmp/warm_metrics.json
```

### Step 3: Decision gate

Compare cold vs warm `ptx.total_sec`:
- Delta = cold - warm = pure driver JIT cost
- If delta > 20s → Phase 1 (cubin) next
- If `circuit.d4_compile_sec` > 20s → Phase 2 (disk cache) next
- If both substantial → larger first

Record decision and proceed to Task 6 or Task 8 accordingly.

### Step 4: Commit benchmark results

Save both JSON files under `docs/plans/warmup-profile-results/` and commit.

---

## Task 6: Phase 1 — Kernel Manifest (Single Source of Truth)

Create a shared kernel manifest consumed by both `build.rs` and `provider.rs`.

**Files:**
- Create: `crates/xlog-cuda/src/kernel_manifest_data.rs`
- Modify: `crates/xlog-cuda/src/lib.rs` (add `pub mod kernel_manifest_data;`)
- Modify: `crates/xlog-cuda/build.rs` (add `include!()` for manifest)

### Step 1: Create shared manifest data file

Create `crates/xlog-cuda/src/kernel_manifest_data.rs` as a **plain data file** that
can be consumed by both `build.rs` (via `include!()`) and `provider.rs` (via `mod`).

The file must use only types and expressions valid in both contexts. Use simple
`&[&str]` arrays:

```rust
// Single source of truth for CUDA kernel modules.
//
// This file is consumed by both `build.rs` (via include!()) and `provider.rs`
// (via mod) to avoid the kernel list being duplicated in two places.
// NOTE: Use regular comments (//), NOT inner doc comments (//!), because
// include!() in build.rs would interpret //! as documenting the wrong item.

/// Module names matching the .cu filenames (without extension).
/// Order matches provider.rs load order. All 19 modules listed.
pub const KERNEL_CU_NAMES: &[&str] = &[
    "join", "dedup", "groupby", "scan", "sort", "filter", "set_ops",
    "pack", "pir", "cnf", "cache", "weights", "circuit",
    "mc_sample", "mc_eval", "arith", "sat", "d4", "neural",
];
```

The entry-point lists stay in `provider.rs` (they reference module-specific kernel
name constants that are only available there). The manifest's job is only to
enumerate modules and enforce the 19-count invariant.

### Step 2: Wire into both consumers

**`crates/xlog-cuda/src/lib.rs`:** Add `pub mod kernel_manifest_data;`

**`crates/xlog-cuda/build.rs`:** Replace the hardcoded 18-kernel list with:

```rust
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/kernel_manifest_data.rs"));
// Now KERNEL_CU_NAMES is available — use it for the module loop.
```

This ensures both `build.rs` and `provider.rs` use the same module list.

### Step 3: Add compile-time count assertion in provider.rs

After loading all 19 modules, add:

```rust
const _: () = assert!(kernel_manifest_data::KERNEL_CU_NAMES.len() == 19);
```

This catches any drift between the manifest and the actual load calls at compile time.

### Step 3: Compile

Run: `cargo check -p xlog-cuda`
Expected: compiles clean

### Step 4: Commit

```bash
git add crates/xlog-cuda/src/kernel_manifest_data.rs crates/xlog-cuda/src/lib.rs crates/xlog-cuda/build.rs
git commit -m "feat(cuda): add kernel_manifest_data.rs as single source of truth

Lists all 19 CUDA module names. Consumed by build.rs (via include!())
and provider.rs (via mod) to eliminate kernel list duplication.
Compile-time assertion enforces 19-count invariant."
```

---

## Task 7: Phase 1 — Build-Time Cubin Generation in `build.rs`

Update `build.rs` to generate cubins for specified SM architectures and one portable
PTX per module.

**Files:**
- Modify: `crates/xlog-cuda/build.rs`

### Step 1: Rewrite build.rs

Replace the current build.rs (which compiles 18 kernels to PTX targeting sm_70) with:

1. Read `XLOG_NO_CUBIN` — if `1`, **skip cubin generation only** (step 3a), but
   **still generate portable PTX** (step 3b). Portable PTX is always required as the
   runtime fallback — without it, loading fails when `include_str!()` is removed.
2. Read `XLOG_CUBIN_ARCHS` — default `sm_120`, comma-separated
3. For each module in `KERNEL_CU_NAMES` (from the included manifest):
   a. If `XLOG_NO_CUBIN` is not set, for each arch: `nvcc --cubin -arch={arch} -O3 -o $OUT_DIR/{name}.{arch}.cubin kernels/{name}.cu`
   b. Always, once per module (outside arch loop): `nvcc --ptx -arch=sm_70 -O3 -o $OUT_DIR/{name}.portable.ptx kernels/{name}.cu`
4. Write `cargo:rerun-if-env-changed` for both env vars
5. Write `cargo:rerun-if-changed` for each `.cu` file

**Note:** `build.rs` uses `include!()` to import `KERNEL_CU_NAMES` from the shared
manifest (see Task 6). No duplication — the 19-module list is defined once.

### Step 2: Handle nvcc not found

If nvcc is not found at all, **emit a hard build error** (`panic!` or
`compile_error!`). Since Phase 1 removes all embedded `include_str!()` PTX
constants, there is no fallback — a successful build without nvcc would
produce a binary that fails at runtime on every module load. The error message
should say: `"nvcc not found — required to generate portable PTX/cubin build
artifacts. Install the CUDA toolkit and ensure nvcc is on PATH."`.

If `XLOG_NO_CUBIN=1`, skip only cubin generation — portable PTX is still built
(and still requires nvcc).

### Step 3: Test build

Run: `cargo build -p xlog-cuda`
Expected: generates cubins in `$OUT_DIR` (check with `ls target/*/build/xlog-cuda-*/out/*.cubin`)

### Step 4: Commit

```bash
git add crates/xlog-cuda/build.rs
git commit -m "feat(cuda): generate cubins per SM arch + portable PTX in build.rs

Generates {name}.sm_{cc}.cubin for each arch in XLOG_CUBIN_ARCHS
and {name}.portable.ptx (sm_70 baseline) for each of the 19 modules.
XLOG_NO_CUBIN=1 skips cubin generation but still builds portable PTX."
```

---

## Task 8: Phase 1 — Runtime Cubin/PTX Loader in `provider.rs`

Replace `include_str!()` + `Ptx::from_src()` with file-based loading that tries cubin
first, falls back to portable PTX.

**Files:**
- Modify: `crates/xlog-cuda/src/provider/mod.rs`

### Step 1: Add SM detection helper

```rust
fn detect_compute_capability(device: &Arc<CudaDevice>) -> Result<u32> {
    let major = device.inner()
        .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .map_err(|e| XlogError::Kernel(format!("Failed to query SM major: {}", e)))?;
    let minor = device.inner()
        .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .map_err(|e| XlogError::Kernel(format!("Failed to query SM minor: {}", e)))?;
    Ok((major as u32) * 10 + (minor as u32))
}
```

### Step 2: Add module resolution helper

```rust
fn resolve_module_path(name: &str, cc: u32) -> Option<std::path::PathBuf> {
    let cubin_dir = std::env::var("XLOG_CUBIN_DIR").ok();
    let out_dir = option_env!("OUT_DIR");

    // 1. XLOG_CUBIN_DIR/{name}.sm_{cc}.cubin
    if let Some(dir) = &cubin_dir {
        let p = std::path::PathBuf::from(dir).join(format!("{}.sm_{}.cubin", name, cc));
        if p.exists() { return Some(p); }
    }
    // 2. OUT_DIR/{name}.sm_{cc}.cubin
    if let Some(dir) = out_dir {
        let p = std::path::PathBuf::from(dir).join(format!("{}.sm_{}.cubin", name, cc));
        if p.exists() { return Some(p); }
    }
    // 3. Portable PTX fallback
    if let Some(dir) = &cubin_dir {
        let p = std::path::PathBuf::from(dir).join(format!("{}.portable.ptx", name));
        if p.exists() { return Some(p); }
    }
    if let Some(dir) = out_dir {
        let p = std::path::PathBuf::from(dir).join(format!("{}.portable.ptx", name));
        if p.exists() { return Some(p); }
    }
    None
}
```

### Step 3: Replace `include_str!()` loads with file-based loading

For each module in `CudaKernelProvider::new()`, replace:

```rust
device.inner().load_ptx(Ptx::from_src(JOIN_PTX), JOIN_MODULE, &[...])
```

with:

```rust
let ptx = resolve_module_path("join", cc)
    .map(Ptx::from_file)
    .ok_or_else(|| XlogError::Kernel(
        "join: no cubin or portable PTX found (set XLOG_CUBIN_DIR or rebuild with nvcc)".into()
    ))?;
device.inner().load_ptx(ptx, JOIN_MODULE, &[...])
```

**Remove all `include_str!()` PTX constants** (`JOIN_PTX`, `DEDUP_PTX`, ... lines 21-39).
The embedded PTX targets `sm_120` and is NOT portable — keeping it as a fallback
silently masks the failure to build portable PTX and would break on non-Blackwell
hardware. File-based loading from build outputs is now the only path.

### Step 4: Remove stale `sm_70` references

Update comment at line 20 (`sm_70` → `sm_120` or remove), line 648 (same).

### Step 5: Migrate tests that depend on embedded PTX constants

Tests at `provider/mod.rs (pre-Wave-2 line 10110)-10288` reference the removed `include_str!()`
constants (`JOIN_PTX`, `DEDUP_PTX`, `CIRCUIT_PTX`, etc.) for inline module
loading. These will fail to compile after Step 3 removes the constants.

For each affected test:
- If it loads a module via `Ptx::from_src(JOIN_PTX)`, replace with
  `Ptx::from_file(resolve_module_path("join", cc).expect("test requires built PTX"))`.
- If it just checks that the PTX string is non-empty (a smoke test for the
  `include_str!()` embedding), **delete it** — it tests the old embedding
  mechanism which no longer exists.
- If it tests kernel launch/correctness, keep the test but switch to
  file-based loading. These tests already require a CUDA device, so requiring
  build artifacts is not an additional constraint.

Verify after migration: `cargo test -p xlog-cuda` passes.

### Step 6: Compile and test

Run: `cargo check -p xlog-cuda`
Expected: compiles clean

Run: `cargo test -p xlog-cuda`
Expected: all tests pass (including migrated tests from Step 5)

### Step 7: Commit

```bash
git add crates/xlog-cuda/src/provider/mod.rs
git commit -m "feat(cuda): load cubin with portable PTX fallback in provider

Detects device SM, tries cubin first (from XLOG_CUBIN_DIR or
OUT_DIR), then portable PTX. Removes all include_str!() embedded
PTX constants and migrates tests to file-based loading.
Eliminates driver JIT when cubins are available."
```

---

## Task 9: Phase 1 — Clean Up `nvrtc_ptx.rs`

**Files:**
- Modify: `crates/xlog-cuda/src/bin/nvrtc_ptx.rs`

### Step 1: Update stale sm_70 reference

Find the hardcoded `sm_70` around line 27 and update to use the correct arch
or make it configurable via env var.

### Step 2: Commit

```bash
git add crates/xlog-cuda/src/bin/nvrtc_ptx.rs
git commit -m "fix(cuda): update stale sm_70 in nvrtc_ptx.rs"
```

---

## Task 10: Phase 2 — Disk Cache Serialization (`disk_cache.rs`)

Create the disk cache module for reading/writing circuit artifacts.

**Files:**
- Create: `crates/xlog-prob/src/compilation/disk_cache.rs`
- Modify: `crates/xlog-prob/src/compilation/mod.rs` (add `pub mod disk_cache;`)

### Step 1: Define cache key and header

```rust
//! On-disk circuit artifact cache.
//!
//! Caches compiled circuit topology so d-DNNF compilation is skipped on warm starts.
//! Stores topology and metadata only — weights change per query.

use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;

use xlog_core::{Result, XlogError};

const MAGIC: u32 = 0x584C4743; // "XLGC"
const FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct CircuitCacheKey {
    pub cnf_hash: u64,
    pub config_hash: u64,
    pub random_vars_hash: u64,
    pub sm: u32,
}

#[derive(Debug)]
pub struct CircuitArtifact {
    pub num_nodes: u32,
    pub num_edges: u32,
    pub num_levels: u32,
    pub root: u32,
    pub max_var: u32,
    pub has_free_var_mask: bool,
    pub node_type: Vec<u8>,
    pub child_offsets: Vec<u32>,
    pub child_indices: Vec<u32>,
    pub lit: Vec<i32>,
    pub decision_var: Vec<u32>,
    pub decision_child_false: Vec<u32>,
    pub decision_child_true: Vec<u32>,
    pub level_nodes: Vec<u32>,
    pub level_offsets: Vec<u32>,
    pub free_var_mask: Vec<u8>,
}
```

### Step 2: Implement cache directory resolution

```rust
pub fn cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("XLOG_CIRCUIT_CACHE_DIR") {
        PathBuf::from(dir)
    } else if let Some(cache) = dirs::cache_dir() {
        cache.join("xlog").join("circuits")
    } else {
        PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join(".cache").join("xlog").join("circuits")
    }
}

fn cache_path(key: &CircuitCacheKey) -> PathBuf {
    let hex = format!(
        "{:016x}_{:016x}_{:016x}_{:08x}_{:08x}",
        key.cnf_hash, key.config_hash, key.random_vars_hash, key.sm, FORMAT_VERSION,
    );
    cache_dir().join(format!("{}.bin", hex))
}
```

### Step 3: Implement write path

```rust
pub fn write_artifact(key: &CircuitCacheKey, artifact: &CircuitArtifact) -> Result<()> {
    // Helper: XlogError has no From<std::io::Error> impl, so every IO op
    // must be explicitly mapped.
    fn io(e: std::io::Error) -> XlogError {
        XlogError::Compilation(format!("circuit cache IO error: {}", e))
    }

    let dir = cache_dir();
    fs::create_dir_all(&dir).map_err(io)?;

    let path = cache_path(key);
    let tmp = path.with_extension("tmp");
    let mut f = fs::File::create(&tmp).map_err(io)?;

    // Write header
    f.write_all(&MAGIC.to_le_bytes()).map_err(io)?;
    f.write_all(&FORMAT_VERSION.to_le_bytes()).map_err(io)?;
    f.write_all(&key.cnf_hash.to_le_bytes()).map_err(io)?;
    f.write_all(&key.config_hash.to_le_bytes()).map_err(io)?;
    f.write_all(&key.random_vars_hash.to_le_bytes()).map_err(io)?;
    f.write_all(&key.sm.to_le_bytes()).map_err(io)?;
    f.write_all(&artifact.num_nodes.to_le_bytes()).map_err(io)?;
    f.write_all(&artifact.num_edges.to_le_bytes()).map_err(io)?;
    f.write_all(&artifact.num_levels.to_le_bytes()).map_err(io)?;
    f.write_all(&artifact.root.to_le_bytes()).map_err(io)?;
    f.write_all(&artifact.max_var.to_le_bytes()).map_err(io)?;
    f.write_all(&[artifact.has_free_var_mask as u8]).map_err(io)?;
    f.write_all(&[0u8; 3]).map_err(io)?; // padding

    // Write arrays as raw bytes
    f.write_all(&artifact.node_type).map_err(io)?;
    f.write_all(bytemuck::cast_slice(&artifact.child_offsets)).map_err(io)?;
    f.write_all(bytemuck::cast_slice(&artifact.child_indices)).map_err(io)?;
    f.write_all(bytemuck::cast_slice(&artifact.lit)).map_err(io)?;
    f.write_all(bytemuck::cast_slice(&artifact.decision_var)).map_err(io)?;
    f.write_all(bytemuck::cast_slice(&artifact.decision_child_false)).map_err(io)?;
    f.write_all(bytemuck::cast_slice(&artifact.decision_child_true)).map_err(io)?;
    f.write_all(bytemuck::cast_slice(&artifact.level_nodes)).map_err(io)?;
    f.write_all(bytemuck::cast_slice(&artifact.level_offsets)).map_err(io)?;
    f.write_all(&artifact.free_var_mask).map_err(io)?;

    drop(f);
    fs::rename(&tmp, &path).map_err(io)?;

    evict_if_needed()?;
    Ok(())
}
```

### Step 4: Implement read path

```rust
pub fn read_artifact(key: &CircuitCacheKey) -> Result<Option<CircuitArtifact>> {
    let path = cache_path(key);
    if !path.exists() {
        return Ok(None);
    }

    let data = fs::read(&path).map_err(|e| {
        XlogError::Compilation(format!("Failed to read cache file: {}", e))
    })?;

    // Validate header (magic, version, key match)
    // Parse arrays from raw bytes
    // Return Some(artifact) or None on validation failure
    // ...
}
```

### Step 5: Implement eviction

```rust
fn evict_if_needed() -> Result<()> {
    let max_mb: u64 = std::env::var("XLOG_CIRCUIT_CACHE_MAX_MB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);

    let dir = cache_dir();
    // Read dir, sum sizes, if > max_mb, delete oldest by mtime
    // ...
    Ok(())
}
```

### Step 6: Add dependencies for disk-cache helpers

Check `crates/xlog-prob/Cargo.toml` for both `bytemuck` and `dirs`.
If either is missing, add it.
`bytemuck` is used for `cast_slice` calls and `dirs` is used by
`dirs::cache_dir()` in `cache_dir()`.
Alternatively, if you want to avoid adding `dirs`, rewrite `cache_dir()` using
only `std::env` path resolution.

### Step 7: Compile and unit test

Write a unit test that creates an artifact, writes it, reads it back, and verifies
all fields match. Place in the same file.

Run: `cargo test -p xlog-prob disk_cache`
Expected: passes

### Step 8: Commit

```bash
git add crates/xlog-prob/src/compilation/disk_cache.rs crates/xlog-prob/src/compilation/mod.rs
git commit -m "feat(disk-cache): add circuit artifact disk cache module

Flat binary format with magic/version/key header. Write uses atomic
rename. Read validates header before returning. Eviction by mtime
when total size exceeds XLOG_CIRCUIT_CACHE_MAX_MB (default 512MB)."
```

---

## Task 11: Phase 2 — GPU Cache `restore_from_host()` Method

Add a method to `GpuCircuitCache` that populates a slot from host-resident arrays
(the disk cache read path).

**Files:**
- Modify: `crates/xlog-prob/src/compilation/gpu_cache.rs`

### Step 1: Add `restore_from_host_arrays()` method

This mirrors `store_from_xgcf()` but takes host-resident `Vec`s instead of a
device-resident `GpuXgcf`. It:

1. Zeros the slot's free_var_mask region (memset_zeros on the slot's mask slice)
2. H→D copies each topology array into the slot using `cache_store_*` kernels
3. H→D copies the free_var_mask
4. Sets `has_free_var_mask[slot]` based on whether mask has non-zero entries
5. Stores metadata via `cache_store_meta` kernel
6. Sets handle metadata (num_nodes, num_levels, root, max_var)

```rust
    pub fn restore_from_host_arrays(
        &mut self,
        handle: &mut GpuCircuitCacheHandle,
        artifact: &disk_cache::CircuitArtifact,
    ) -> Result<()> {
        // Validate sizes against cache caps
        // Upload arrays using the same cache_store_* kernels as store_from_xgcf
        // Zero + conditionally write free_var_mask
        // Set per-slot has_free_var_mask flag
    }
```

### Step 2: Compile and test

Run: `cargo check -p xlog-prob`
Expected: compiles clean

### Step 3: Commit

```bash
git add crates/xlog-prob/src/compilation/gpu_cache.rs
git commit -m "feat(gpu-cache): add restore_from_host_arrays for disk cache path

Populates a GPU cache slot from host-resident arrays. Zeros mask
region before writing, sets per-slot has_free_var_mask flag."
```

---

## Task 12: Phase 2 — Integration in `compile_gpu_d4_and_verify_cached()`

Wire the disk cache into the compilation pipeline.

**Files:**
- Modify: `crates/xlog-prob/src/compilation/mod.rs`

### Step 1: Add disk cache check between lookup and D4

After `lookup_or_insert_device` (line 146) and before `compile_gpu_d4_gated` (line 152):

```rust
    // Disk cache check (only when compile_needed == 1)
    if warmup_profiling_enabled() || true {  // always check disk cache when available
        let compile_needed_host: Vec<u32> = provider.device().inner()
            .dtoh_sync_copy(handle.compile_needed_device())
            .map_err(|e| XlogError::Kernel(format!("dtoh compile_needed: {}", e)))?;
        if compile_needed_host[0] == 1 {
            let cnf_hash_host: Vec<u64> = provider.device().inner()
                .dtoh_sync_copy(&key)
                .map_err(|e| XlogError::Kernel(format!("dtoh cnf_hash: {}", e)))?;

            let cc = detect_compute_capability(provider)?;
            let cache_key = disk_cache::CircuitCacheKey {
                cnf_hash: cnf_hash_host[0],
                config_hash: hash_compile_config(config),
                random_vars_hash: hash_random_vars(random_vars),
                sm: cc,
            };

            if let Some(artifact) = disk_cache::read_artifact(&cache_key)? {
                cache.restore_from_host_arrays(&mut handle, &artifact)?;
                // Skip D4/verify/smooth/store — topology restored from disk
                return Ok((handle, profile));
            }
        }
    }
```

### Step 2: Add disk cache write after successful compilation

After `store_from_xgcf` and `store_free_var_mask`, D→H copy topology arrays and write:

```rust
    // Write to disk cache for next warm start
    if let Ok(artifact) = build_artifact_from_device(cache, &handle, provider) {
        let _ = disk_cache::write_artifact(&cache_key, &artifact);
        // Ignore write errors — disk cache is opportunistic
    }
```

### Step 3: Add helper functions

```rust
fn hash_compile_config(config: &GpuCompileConfig) -> u64 { ... }
fn hash_random_vars(random_vars: &DeviceRandomVarList) -> u64 { ... }
fn detect_compute_capability(provider: &Arc<CudaKernelProvider>) -> Result<u32> { ... }
fn build_artifact_from_device(...) -> Result<CircuitArtifact> { ... }
```

### Step 4: Compile and test

Run: `cargo check -p xlog-prob`
Expected: compiles clean

Run: `cargo test -p xlog-prob`
Expected: all tests pass

### Step 5: Integration test

Run training twice:
1. First run: disk cache miss, full compilation, writes cache file
2. Second run: disk cache hit, skips D4/verify/smooth

Verify both produce identical loss values.

### Step 6: Commit

```bash
git add crates/xlog-prob/src/compilation/mod.rs
git commit -m "feat(disk-cache): wire circuit disk cache into compilation pipeline

Checks disk cache after GPU cache miss, before D4 compilation.
On hit: restores topology from disk, skips D4/verify/smooth.
On miss: compiles normally, writes artifact for next warm start."
```

---

## Task 13: Phase 3 — Runner Merge Logic Updates

Update track runners to preserve optional diagnostic fields.

**Files:**
- Modify: `scripts/track_b_runner.py`
- Modify: `scripts/track_a_runner.py`

### Step 1: Add warmup_breakdown to preserved keys

In `track_b_runner.py`, after the frozen-key merge (line 336), add:

```python
            # Preserve optional diagnostic fields
            for opt_key in ("warmup_breakdown",):
                if opt_key in ext:
                    metrics[opt_key] = ext[opt_key]
```

### Step 2: Same change in `track_a_runner.py`

Apply identical change.

### Step 3: Test

Run: `python scripts/track_b_runner.py --dry-run` (if available) or manual review.

### Step 4: Commit

```bash
git add scripts/track_b_runner.py scripts/track_a_runner.py
git commit -m "feat(runners): preserve warmup_breakdown in metric merge

Optional diagnostic fields from training scripts are now carried
through to the final metrics.json instead of being dropped."
```

---

## Task 14: End-to-End Validation

Run full Track B with all changes and verify:

1. Steady-state performance preserved (0.248s/epoch ± 10%)
2. Warmup profiling produces valid breakdown JSON
3. Cubin loading works (if Phase 1 completed)
4. Disk cache hit on second run (if Phase 2 completed)
5. Loss convergence identical to baseline

### Step 1: Full Track B run

```bash
cd /home/dev/projects/xlog/.worktrees/v0.4.0-alpha-integrated
LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH \
  python scripts/track_b_runner.py
```

### Step 2: Verify results

Check that all 3 seeds pass schema gate and metrics are sane.

### Step 3: Update comparison JSON if numbers changed

Rebuild `mnist_xlog_vs_scallop.json` with new numbers.

### Step 4: Commit results

```bash
git add examples/neural/results/
git commit -m "bench: track B results with warmup reduction changes"
```

---

## Dependencies

```
Task 1 (per-slot mask) ─── must complete before ───> Task 10-12 (Phase 2)
Task 2 (PTX timing) ──┐
Task 3 (circuit timing)├── must complete before ───> Task 4 (pyxlog surface)
                       │
Task 4 ────────────────┘── must complete before ───> Task 5 (decision gate)
Task 5 (decision gate) ──> determines order of Tasks 6-9 vs 10-12
Task 6 (manifest) ────────> Task 7 (build.rs) ─────> Task 8 (loader)
Task 10 (disk_cache.rs) ──> Task 11 (restore) ─────> Task 12 (integration)
Task 13 (runners) ── independent, can run in parallel
Task 14 (validation) ── must be last
```

## Parallelizable Groups

- **Group A (independent):** Tasks 1, 2, 3, 13 — can all be done in parallel
- **Group B (after Group A):** Task 4 (needs 2+3)
- **Group C (after decision gate):** Tasks 6→7→8→9 (Phase 1) OR Tasks 10→11→12 (Phase 2)
- **Group D:** Task 14 (after everything)
