# GPU Native Phase 4 (Cache + Integration) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a fully device-resident circuit cache and integrate the GPU-native exact compiler path (no CPU D4, no host reads) per `docs/design/2026-01-22-gpu-native-compilation-design.md` Phase 4.

**Architecture:** Add a GPU-resident cache keyed by a device-computed CNF hash, with device-side lookup/insert and LRU eviction. Integrate GPU D4 + GPU CDCL verification with cache slots, and route exact inference (pyxlog + CLI) through the GPU-native pipeline with device-only evaluation kernels.

**Tech Stack:** Rust, CUDA (PTX via `nvcc`), `cudarc`, `xlog-cuda` provider, GPU D4 kernels (`kernels/d4.cu`), circuit kernels (`kernels/circuit.cu`), SAT verifier (`kernels/sat.cu`).

---

## Task 1: GPU CNF Hash Kernel + Rust Wrapper

**Files:**
- Create: `kernels/cache.cu`
- Modify: `crates/xlog-cuda/build.rs`
- Modify: `crates/xlog-cuda/src/provider.rs`
- Create: `crates/xlog-prob/src/compilation/gpu_cache.rs`
- Create: `crates/xlog-prob/tests/gpu_cnf_hash.rs`

**Step 1: Write failing test (GPU hash vs CPU reference)**

```rust
// crates/xlog-prob/tests/gpu_cnf_hash.rs
use std::sync::Arc;

use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_solve::{Clause, GpuCnf, Literal, SolveInstance};
use xlog_prob::compilation::gpu_cache::hash_cnf_gpu;

fn cpu_hash_u64(vals: &[u64]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut h = FNV_OFFSET;
    for &v in vals {
        h ^= v;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

#[test]
fn gpu_cnf_hash_matches_cpu_reference() {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {}", e);
            return;
        }
    };
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), MemoryBudget::with_limit(1 << 30)));
    let provider = Arc::new(CudaKernelProvider::new(device, memory).expect("provider"));

    let clauses = vec![
        Clause::new(vec![Literal::new(1, true), Literal::new(2, false)]),
        Clause::new(vec![Literal::new(2, true)]),
    ];
    let instance = SolveInstance::new(2, clauses).expect("instance");
    let cnf = GpuCnf::from_host(&instance, &provider).expect("GpuCnf");

    let hashes = hash_cnf_gpu(&cnf, &provider).expect("hash");

    // Build a simple CPU reference input stream: num_vars, num_clauses, offsets, literals.
    let mut host: Vec<u32> = Vec::new();
    let mut tmp = vec![0u32; 1];
    provider.device().inner().dtoh_sync_copy_into(&cnf.num_vars, &mut tmp).unwrap();
    host.push(tmp[0]);
    provider.device().inner().dtoh_sync_copy_into(&cnf.num_clauses, &mut tmp).unwrap();
    host.push(tmp[0]);

    let mut offsets = vec![0u32; cnf.clause_offsets.len()];
    let mut lits = vec![0i32; cnf.literals.len()];
    provider.device().inner().dtoh_sync_copy_into(&cnf.clause_offsets, &mut offsets).unwrap();
    provider.device().inner().dtoh_sync_copy_into(&cnf.literals, &mut lits).unwrap();

    for &v in &offsets {
        host.push(v);
    }
    for &v in &lits {
        host.push(v as u32);
    }

    let cpu_hash = cpu_hash_u64(&host.iter().map(|&v| v as u64).collect::<Vec<_>>());

    let mut gpu_hash = vec![0u64; 1];
    provider.device().inner().dtoh_sync_copy_into(&hashes, &mut gpu_hash).unwrap();

    assert_eq!(gpu_hash[0], cpu_hash);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-prob gpu_cnf_hash -- --nocapture`

Expected: FAIL (missing `hash_cnf_gpu` and kernel).

**Step 3: Implement cache hash kernel and wrapper**

- Add a sequential GPU hash kernel in `kernels/cache.cu` (single block/thread) computing FNV-1a over the CNF metadata, offsets, and literals.
- Export as `cache_cnf_hash`.
- Add module and kernel names in `crates/xlog-cuda/src/provider.rs`.
- Add kernel to build list in `crates/xlog-cuda/build.rs`.
- Implement `hash_cnf_gpu` in `crates/xlog-prob/src/compilation/gpu_cache.rs`.

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-prob gpu_cnf_hash -- --nocapture`

Expected: PASS.

**Step 5: Commit**

```bash
git add kernels/cache.cu kernels/cache.ptx crates/xlog-cuda/build.rs crates/xlog-cuda/src/provider.rs \
  crates/xlog-prob/src/compilation/gpu_cache.rs crates/xlog-prob/tests/gpu_cnf_hash.rs

git commit -m "feat(gpu-cache): add CNF hash kernel and wrapper"
```

---

## Task 2: GPU Cache Table + LRU Kernels

**Files:**
- Modify: `kernels/cache.cu`
- Modify: `crates/xlog-cuda/src/provider.rs`
- Modify: `crates/xlog-prob/src/compilation/gpu_cache.rs`
- Create: `crates/xlog-prob/tests/gpu_circuit_cache.rs`

**Step 1: Write failing cache behavior tests**

```rust
// crates/xlog-prob/tests/gpu_circuit_cache.rs
use std::sync::Arc;

use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_prob::compilation::gpu_cache::GpuCircuitCache;

#[test]
fn gpu_cache_hit_miss_and_eviction() {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {}", e);
            return;
        }
    };
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), MemoryBudget::with_limit(1 << 30)));
    let provider = Arc::new(CudaKernelProvider::new(device, memory).expect("provider"));

    let mut cache = GpuCircuitCache::new(&provider, 2, 4).expect("cache"); // 2 slots, table size 4

    let k1 = 0x1111u64;
    let k2 = 0x2222u64;
    let k3 = 0x3333u64;

    let h1 = cache.lookup_or_insert(k1).expect("lookup k1");
    assert!(h1.compile_needed_host());
    let h2 = cache.lookup_or_insert(k1).expect("lookup k1 again");
    assert!(!h2.compile_needed_host());

    let h3 = cache.lookup_or_insert(k2).expect("lookup k2");
    assert!(h3.compile_needed_host());

    // Cache full; insert k3 should evict least-recently-used key (k1).
    let h4 = cache.lookup_or_insert(k3).expect("lookup k3");
    assert!(h4.compile_needed_host());

    let h5 = cache.lookup_or_insert(k1).expect("lookup k1 post-evict");
    assert!(h5.compile_needed_host());
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-prob gpu_circuit_cache -- --nocapture`

Expected: FAIL (missing cache API, kernels).

**Step 3: Implement cache table + LRU**

- In `kernels/cache.cu`, implement:
  - `cache_lookup_or_insert` (open addressing; updates `last_used` and returns slot + compile_needed).
  - `cache_evict_lru` (scan `last_used` for min and evict deterministically).
- In `gpu_cache.rs`, implement `GpuCircuitCache` that owns:
  - `keys: DeviceSlice<u64>`
  - `slots: DeviceSlice<u32>`
  - `last_used: DeviceSlice<u64>`
  - `state: DeviceSlice<u32>`
  - `clock: DeviceSlice<u64>`
  - `active_slot: DeviceSlice<u32>`
  - `compile_needed: DeviceSlice<u32>`
- Provide a `lookup_or_insert` wrapper that launches the kernel and (in tests only) copies `compile_needed` to host for assertions.

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-prob gpu_circuit_cache -- --nocapture`

Expected: PASS.

**Step 5: Commit**

```bash
git add kernels/cache.cu kernels/cache.ptx crates/xlog-cuda/src/provider.rs \
  crates/xlog-prob/src/compilation/gpu_cache.rs crates/xlog-prob/tests/gpu_circuit_cache.rs

git commit -m "feat(gpu-cache): add device-resident cache table + LRU"
```

---

## Task 3: Cache-Aware XGCF Evaluation Kernels

**Files:**
- Modify: `kernels/circuit.cu`
- Modify: `crates/xlog-cuda/src/provider.rs`
- Modify: `crates/xlog-prob/src/gpu.rs`
- Create: `crates/xlog-prob/tests/gpu_xgcf_cached.rs`

**Step 1: Write failing cached-eval test**

```rust
// crates/xlog-prob/tests/gpu_xgcf_cached.rs
use std::sync::Arc;

use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_prob::gpu::{GpuXgcf};
use xlog_prob::compilation::gpu_cache::GpuCircuitCache;
use xlog_prob::xgcf::Xgcf;

#[test]
fn cached_eval_matches_direct_eval() {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {}", e);
            return;
        }
    };
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), MemoryBudget::with_limit(1 << 30)));
    let provider = Arc::new(CudaKernelProvider::new(device, memory).expect("provider"));

    let circuit = Xgcf::lit(1).expect("circuit");
    let mut direct = GpuXgcf::upload(&provider, &circuit).expect("upload");

    let mut cache = GpuCircuitCache::new(&provider, 1, 4).expect("cache");
    let handle = cache.claim_slot(0xabcdefu64).expect("claim");
    cache.store_from_xgcf(&handle, &direct).expect("store");

    let mut out_direct = provider.memory().alloc::<f64>(1).unwrap();
    let mut out_cached = provider.memory().alloc::<f64>(1).unwrap();

    direct
        .eval_log_wmc_device_inplace(&provider, &mut out_direct)
        .expect("direct eval");
    cache
        .eval_log_wmc_device_inplace(&handle, &mut out_cached)
        .expect("cached eval");

    let mut hd = vec![0.0f64; 1];
    let mut hc = vec![0.0f64; 1];
    provider.device().inner().dtoh_sync_copy_into(&out_direct, &mut hd).unwrap();
    provider.device().inner().dtoh_sync_copy_into(&out_cached, &mut hc).unwrap();

    assert_eq!(hd[0].to_bits(), hc[0].to_bits());
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-prob gpu_xgcf_cached -- --nocapture`

Expected: FAIL (missing cached kernels + cache eval API).

**Step 3: Implement cached kernels + API**

- Add cached variants to `kernels/circuit.cu`:
  - `xgcf_forward_level_cached`
  - `xgcf_backward_level_*_cached`
  - `xgcf_free_var_*_cached`
  - `xgcf_add_scalar_cached`
- Extend `circuit_kernels` list and `provider.rs` load list.
- In `gpu_cache.rs`, add `eval_log_wmc_device_inplace` using cached kernels; compute base offsets from device `active_slot` inside kernels.

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-prob gpu_xgcf_cached -- --nocapture`

Expected: PASS.

**Step 5: Commit**

```bash
git add kernels/circuit.cu kernels/circuit.ptx crates/xlog-cuda/src/provider.rs \
  crates/xlog-prob/src/gpu.rs crates/xlog-prob/src/compilation/gpu_cache.rs \
  crates/xlog-prob/tests/gpu_xgcf_cached.rs

git commit -m "feat(gpu-cache): add cached XGCF eval kernels"
```

---

## Task 4: Cache Slot Storage + Metadata

**Files:**
- Modify: `kernels/cache.cu`
- Modify: `crates/xlog-prob/src/compilation/gpu_cache.rs`
- Create: `crates/xlog-prob/tests/gpu_cache_store.rs`

**Step 1: Write failing cache-store test**

```rust
// crates/xlog-prob/tests/gpu_cache_store.rs
use std::sync::Arc;

use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_prob::compilation::gpu_cache::GpuCircuitCache;
use xlog_prob::gpu::GpuXgcf;
use xlog_prob::xgcf::Xgcf;

#[test]
fn cache_store_writes_metadata() {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {}", e);
            return;
        }
    };
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), MemoryBudget::with_limit(1 << 30)));
    let provider = Arc::new(CudaKernelProvider::new(device, memory).expect("provider"));

    let circuit = Xgcf::lit(1).expect("circuit");
    let direct = GpuXgcf::upload(&provider, &circuit).expect("upload");

    let mut cache = GpuCircuitCache::new(&provider, 1, 4).expect("cache");
    let handle = cache.claim_slot(0xabcdu64).expect("claim");
    cache.store_from_xgcf(&handle, &direct).expect("store");

    let meta = cache.read_meta_host(&handle).expect("meta");
    assert!(meta.num_nodes > 0);
    assert!(meta.num_levels > 0);
    assert!(meta.max_var > 0);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-prob gpu_cache_store -- --nocapture`

Expected: FAIL (missing store + metadata API).

**Step 3: Implement cache store kernels + metadata**

- Add `cache_store_xgcf` kernel to copy circuit buffers into slot arrays.
- Add `cache_store_meta` kernel to write `num_nodes`, `num_levels`, `root`, `max_var` per slot.
- Extend `GpuCircuitCache` with `store_from_xgcf` and `read_meta_host` (test-only host read).

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-prob gpu_cache_store -- --nocapture`

Expected: PASS.

**Step 5: Commit**

```bash
git add kernels/cache.cu kernels/cache.ptx crates/xlog-prob/src/compilation/gpu_cache.rs \
  crates/xlog-prob/tests/gpu_cache_store.rs

git commit -m "feat(gpu-cache): add cache store + metadata"
```

---

## Task 5: GPU D4 + GPU CDCL Cache Integration

**Files:**
- Modify: `crates/xlog-prob/src/compilation/gpu_d4.rs`
- Modify: `crates/xlog-prob/src/compilation/validation.rs`
- Modify: `crates/xlog-prob/src/compilation/mod.rs`
- Modify: `crates/xlog-solve/src/gpu_cdcl.rs`
- Modify: `kernels/d4.cu`
- Modify: `kernels/sat.cu`
- Modify: `crates/xlog-cuda/src/provider.rs`
- Create: `crates/xlog-prob/tests/gpu_cache_compile_and_verify.rs`

**Step 1: Write failing compile+cache test**

```rust
// crates/xlog-prob/tests/gpu_cache_compile_and_verify.rs
use std::sync::Arc;

use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_prob::compilation::{compile_gpu_d4_and_verify_cached, GpuCompileConfig};
use xlog_prob::compilation::gpu_cache::GpuCircuitCache;
use xlog_solve::{Clause, GpuCnf, Literal, SolveInstance};

#[test]
fn gpu_cache_compile_reuses_slot() {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {}", e);
            return;
        }
    };
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), MemoryBudget::with_limit(1 << 30)));
    let provider = Arc::new(CudaKernelProvider::new(device, memory).expect("provider"));

    let clauses = vec![Clause::new(vec![Literal::new(1, true)])];
    let instance = SolveInstance::new(1, clauses).unwrap();
    let cnf = GpuCnf::from_host(&instance, &provider).unwrap();

    let mut cache = GpuCircuitCache::new(&provider, 1, 4).unwrap();
    let config = GpuCompileConfig::default();

    let h1 = compile_gpu_d4_and_verify_cached(&cnf, &provider, &config, &mut cache)
        .expect("compile 1");
    let h2 = compile_gpu_d4_and_verify_cached(&cnf, &provider, &config, &mut cache)
        .expect("compile 2");

    assert_eq!(h1.slot_host(), h2.slot_host());
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-prob gpu_cache_compile_and_verify -- --nocapture`

Expected: FAIL (missing cached compile path).

**Step 3: Implement cached compile + verification**

- Add `compile_gpu_d4_and_verify_cached` in `compilation/mod.rs` that:
  1) hashes CNF on GPU,
  2) does cache lookup/insert,
  3) compiles into temp `GpuXgcf` (current D4),
  4) stores into cache slot (conditional on `compile_needed`),
  5) validates equivalence with GPU CDCL (conditional on `compile_needed`).
- Add compile gating in `kernels/d4.cu` (early-exit if `compile_needed == 0`).
- Add verifier gating in `kernels/sat.cu` + `gpu_cdcl.rs` so `solve_expect_*` can skip work when `compile_needed == 0`.
- Store free-var mask into cache slot after compile (reuse `compute_free_var_mask_gpu`).

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-prob gpu_cache_compile_and_verify -- --nocapture`

Expected: PASS.

**Step 5: Commit**

```bash
git add kernels/d4.cu kernels/d4.ptx kernels/sat.cu kernels/sat.ptx \
  crates/xlog-solve/src/gpu_cdcl.rs crates/xlog-prob/src/compilation/gpu_d4.rs \
  crates/xlog-prob/src/compilation/validation.rs crates/xlog-prob/src/compilation/mod.rs \
  crates/xlog-prob/tests/gpu_cache_compile_and_verify.rs

git commit -m "feat(gpu-cache): integrate D4 + CDCL with cache slots"
```

---

## Task 6: Replace CPU D4 in Exact Path (GPU-Only)

**Files:**
- Modify: `crates/xlog-prob/src/exact.rs`
- Modify: `crates/pyxlog/src/lib.rs`
- Modify: `crates/xlog-cli/src/main.rs`
- Create: `crates/xlog-prob/tests/no_cpu_d4_in_exact.rs`

**Step 1: Write failing guardrail test**

```rust
// crates/xlog-prob/tests/no_cpu_d4_in_exact.rs
use std::path::PathBuf;

#[test]
fn exact_path_uses_gpu_only() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("src");
    path.push("exact.rs");

    let text = std::fs::read_to_string(&path).expect("read exact.rs");
    assert!(!text.contains("D4Compiler"));
    assert!(!text.contains("TempDirGuard"));
    assert!(!text.contains("in.cnf"));
    assert!(!text.contains("out.nnf"));
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-prob no_cpu_d4_in_exact -- --nocapture`

Expected: FAIL (CPU D4 still referenced).

**Step 3: Implement GPU-native exact path + cache**

- Replace `compile_provenance` CPU D4 + file IO with GPU pipeline:
  - build GPU PIR/roots,
  - `encode_cnf_gpu`,
  - `compile_gpu_d4_and_verify_cached`,
  - store device weights using GPU kernels (no host reads).
- Update `GpuExactState` to use a `GpuCircuitCacheHandle` instead of a host `Xgcf`.
- Ensure evaluation uses cache-aware device kernels.
- Update pyxlog template cache to store cache handles, not host `ExactDdnnfProgram`.
- Update CLI to route to GPU-native compilation path.

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-prob no_cpu_d4_in_exact -- --nocapture`

Expected: PASS.

**Step 5: Commit**

```bash
git add crates/xlog-prob/src/exact.rs crates/pyxlog/src/lib.rs crates/xlog-cli/src/main.rs \
  crates/xlog-prob/tests/no_cpu_d4_in_exact.rs

git commit -m "feat(exact): replace CPU D4 with GPU-native cache path"
```

---

## Task 7: GPU Cache Integration Tests + DTOH Guardrails

**Files:**
- Create: `crates/xlog-prob/tests/gpu_exact_cache_integration.rs`
- Modify: `crates/xlog-prob/tests/no_dtoh_in_gpu_equivalence.rs` (include cache modules)

**Step 1: Write failing integration test**

```rust
// crates/xlog-prob/tests/gpu_exact_cache_integration.rs
use std::sync::Arc;

use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_prob::exact::{ExactDdnnfProgram, GpuConfig};

#[test]
fn exact_gpu_cache_hit_reuses_circuit() {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {}", e);
            return;
        }
    };
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), MemoryBudget::with_limit(1 << 30)));
    let provider = CudaKernelProvider::new(device, memory).expect("provider");

    let source = "0.5::a. query(a).";
    let config = GpuConfig { device_ordinal: 0, memory_bytes: 1 << 30 };

    let prog = ExactDdnnfProgram::compile_source_with_gpu(source, config).expect("compile");
    let r1 = prog.evaluate().expect("eval 1");
    let r2 = prog.evaluate().expect("eval 2");

    assert_eq!(r1.query_probs.len(), r2.query_probs.len());
    assert_eq!(r1.query_probs[0].prob.to_bits(), r2.query_probs[0].prob.to_bits());
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-prob gpu_exact_cache_integration -- --nocapture`

Expected: FAIL until cache path is wired.

**Step 3: Update guardrails**

- Extend `no_dtoh_in_gpu_equivalence.rs` and add a new `no_dtoh_in_gpu_cache.rs` to ensure cache modules do not call `dtoh_sync_copy_into` or `copy_to_host`.

**Step 4: Run tests to verify they pass**

Run: `cargo test -p xlog-prob gpu_exact_cache_integration no_dtoh_in_gpu_cache -- --nocapture`

Expected: PASS.

**Step 5: Commit**

```bash
git add crates/xlog-prob/tests/gpu_exact_cache_integration.rs \
  crates/xlog-prob/tests/no_dtoh_in_gpu_equivalence.rs crates/xlog-prob/tests/no_dtoh_in_gpu_cache.rs

git commit -m "test(gpu-cache): add integration + dtoh guardrails"
```

---

## Task 8: Documentation + Certification Updates

**Files:**
- Modify: `docs/design/2026-01-22-gpu-native-compilation-design.md`
- Modify: `docs/architecture/xlog-prob.md`
- Modify: `docs/architecture/cuda-certification.md`
- Modify: `docs/roadmap.md`
- Modify: `docs/CHANGELOG.md`

**Step 1: Update design doc Phase 4 status**

Document that Phase 4 cache + integration is implemented and list new cache module, kernels, and APIs.

**Step 2: Update architecture docs**

Add `cache.ptx` to kernel lists and describe cache module layout + device-only lookup.

**Step 3: Update roadmap/changelog**

Record Phase 4 completion, new cache, and GPU-only exact inference.

**Step 4: Verify docs build**

Run: `rg -n "Phase 4" docs/design/2026-01-22-gpu-native-compilation-design.md`

Expected: updated status and new kernel references.

**Step 5: Commit**

```bash
git add docs/design/2026-01-22-gpu-native-compilation-design.md docs/architecture/xlog-prob.md \
  docs/architecture/cuda-certification.md docs/roadmap.md docs/CHANGELOG.md

git commit -m "docs: record Phase 4 GPU cache + integration"
```

---

## Execution

Plan complete and saved to `docs/plans/2026-01-28-gpu-native-phase4-cache-integration.md`.

Two execution options:

1. **Subagent-Driven (this session)** — I dispatch fresh subagent per task, review between tasks, fast iteration
2. **Parallel Session (separate)** — Open new session with executing-plans, batch execution with checkpoints

Which approach?
