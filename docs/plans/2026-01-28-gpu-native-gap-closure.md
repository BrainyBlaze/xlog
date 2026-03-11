# GPU-Native Gap Closure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete Phase 4 requirements and close all remaining GPU-native gaps (no device-to-host transfers on the production GPU path, certification coverage, GPU-native MC, and zero-copy interop).

**Architecture:** Keep compilation, cache, and evaluation fully device-resident by moving remaining control/metadata reads to device and by providing device-only evaluation entrypoints. Close solver and interop gaps by extending CUDA certification and adding GPU-native Monte Carlo and Arrow zero-copy paths.

**Tech Stack:** Rust, CUDA (cudarc), xlog-cuda kernels, xlog-prob/xlog-solve/xlog-runtime, Arrow C Data Interface, DLPack.

---

### Task 1: Guardrails for GPU-native exact path (no DTOH in GPU-only modules)

**Files:**
- Create: `crates/xlog-prob/tests/no_dtoh_in_gpu_exact_path.rs`
- Modify: `crates/xlog-prob/src/exact_gpu.rs` (new module)
- Modify: `crates/xlog-prob/src/lib.rs`

**Step 1: Write the failing test**

```rust
// crates/xlog-prob/tests/no_dtoh_in_gpu_exact_path.rs
use std::fs;
use std::path::Path;

#[test]
fn no_device_to_host_reads_in_exact_gpu_module() {
    let path = Path::new("crates/xlog-prob/src/exact_gpu.rs");
    let text = fs::read_to_string(path).expect("read exact_gpu.rs");
    assert!(
        !text.contains("dtoh_sync_copy_into"),
        "exact_gpu.rs must not use dtoh_sync_copy_into"
    );
    assert!(
        !text.contains("copy_to_host"),
        "exact_gpu.rs must not use copy_to_host"
    );
    assert!(
        !text.contains("dtoh"),
        "exact_gpu.rs must not reference dtoh transfers"
    );
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-prob --test no_dtoh_in_gpu_exact_path`
Expected: FAIL because `exact_gpu.rs` does not exist yet.

**Step 3: Write minimal implementation**

Create `crates/xlog-prob/src/exact_gpu.rs` and move GPU-only compilation/evaluation entrypoints into it.
All GPU-only functions must avoid DTOH reads. Start with a minimal module that re-exports a new
GPU-only compile entrypoint and keeps any DTOH work in `exact.rs` (host APIs).

```rust
// crates/xlog-prob/src/exact_gpu.rs
use std::sync::Arc;
use xlog_core::{Result, XlogError};
use xlog_cuda::CudaKernelProvider;
use xlog_solve::GpuCnf;

use crate::compilation::{
    compile_gpu_d4_and_verify_cached, encode_cnf_gpu, GpuCompileConfig, GpuPirGraph, GpuPirRoots,
};
use crate::compilation::gpu_cache::{GpuCircuitCache, GpuCircuitCacheHandle, GpuCircuitCacheConfig};
use crate::compilation::gpu_weights::{
    build_evidence_by_var_gpu, build_weights_gpu, map_nodes_to_vars_gpu, upload_weights_from_host,
    GpuWeights,
};
use crate::pir::Provenance;

pub struct ExactGpuState {
    pub provider: Arc<CudaKernelProvider>,
    pub cache: GpuCircuitCache,
    pub handle: GpuCircuitCacheHandle,
    pub weights: GpuWeights,
    pub max_var: u32,
    pub query_vars_device: Option<xlog_cuda::memory::TrackedCudaSlice<u32>>,
}

pub fn compile_provenance_gpu_only(
    provenance: &Provenance,
    config: GpuConfig,
) -> Result<ExactGpuState> {
    let device = cudarc::driver::CudaDevice::new(config.device_ordinal)?;
    let provider = Arc::new(CudaKernelProvider::new(device, config.memory_bytes)?);

    let (roots, queries, random_vars, evidence_formulas) = crate::exact::collect_roots_queries(provenance)?;

    let gpu_pir = GpuPirGraph::from_host(&provenance.pir, &provider)?;
    let gpu_roots = GpuPirRoots::from_host(&roots, &provider)?;
    let encoding = encode_cnf_gpu(&gpu_pir, &gpu_roots, &provider)?;
    if encoding.vars.max_var != encoding.cnf.var_cap {
        return Err(XlogError::Compilation("CNF var_cap != max_var".to_string()));
    }

    let (leaf_probs_host, choice_true_host, choice_false_host) =
        crate::exact::build_weight_sources(provenance)?;
    let leaf_probs = upload_weights_from_host(&provider, &leaf_probs_host)?;
    let choice_true = upload_weights_from_host(&provider, &choice_true_host)?;
    let choice_false = upload_weights_from_host(&provider, &choice_false_host)?;

    let evidence_by_var = crate::exact::build_evidence_table_gpu(
        &provider,
        &encoding.vars,
        &evidence_formulas,
    )?;
    let weights = build_weights_gpu(
        &encoding.vars,
        &leaf_probs,
        &choice_true,
        &choice_false,
        &evidence_by_var,
        &provider,
    )?;

    let compile_config = crate::exact::default_compile_config(&encoding.cnf, config.memory_bytes)?;
    let cache_config = crate::exact::default_cache_config(&encoding.cnf, &compile_config)?;
    let mut cache = GpuCircuitCache::new(&provider, cache_config)?;
    let handle = compile_gpu_d4_and_verify_cached(
        &encoding.cnf,
        &provider,
        &compile_config,
        &mut cache,
        &random_vars,
    )?;

    cache.store_weights(&handle, &weights.log_true, &weights.log_false)?;

    let query_vars_device = if queries.is_empty() {
        None
    } else {
        let node_ids = crate::exact::queries_to_node_ids(&queries);
        let node_ids_device = crate::compilation::gpu_weights::upload_u32(&provider, &node_ids)?;
        Some(map_nodes_to_vars_gpu(
            &encoding.vars.node_var,
            &node_ids_device,
            encoding.vars.max_var,
            &provider,
        )?)
    };

    Ok(ExactGpuState {
        provider,
        cache,
        handle,
        weights,
        max_var: encoding.vars.max_var,
        query_vars_device,
    })
}

#[derive(Clone, Copy)]
pub struct GpuConfig {
    pub device_ordinal: u32,
    pub memory_bytes: usize,
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-prob --test no_dtoh_in_gpu_exact_path`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/xlog-prob/src/exact_gpu.rs crates/xlog-prob/src/lib.rs crates/xlog-prob/tests/no_dtoh_in_gpu_exact_path.rs
git commit -m "feat: add GPU-only exact compilation module and DTOH guard"
```

---

### Task 2: Remove host reads in cache hit/miss flow (device-only meta)

**Files:**
- Modify: `crates/xlog-prob/src/compilation/mod.rs`
- Modify: `crates/xlog-prob/src/compilation/gpu_cache.rs`
- Modify: `crates/xlog-prob/src/compilation/gpu_d4.rs`
- Modify: `crates/xlog-prob/src/gpu.rs`
- Test: `crates/xlog-prob/tests/no_dtoh_in_gpu_cache_path.rs`

**Step 1: Write the failing test**

```rust
// crates/xlog-prob/tests/no_dtoh_in_gpu_cache_path.rs
use std::fs;
use std::path::Path;

#[test]
fn no_device_to_host_reads_in_cache_compile_path() {
    let paths = [
        "crates/xlog-prob/src/compilation/mod.rs",
        "crates/xlog-prob/src/compilation/gpu_cache.rs",
    ];
    for path in paths {
        let text = fs::read_to_string(Path::new(path)).expect("read source");
        assert!(
            !text.contains("dtoh_sync_copy_into"),
            "cache compile path must not use dtoh_sync_copy_into: {}",
            path
        );
        assert!(
            !text.contains("copy_to_host"),
            "cache compile path must not use copy_to_host: {}",
            path
        );
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-prob --test no_dtoh_in_gpu_cache_path`
Expected: FAIL (current cache compile path reads compile_needed + meta on host).

**Step 3: Write minimal implementation**

- Remove `compile_needed` host read by always invoking gated kernels.
- Replace `load_cache_handle_meta` with a device-resident handle that carries device meta slices.
- Update evaluation kernels to read device meta (`num_nodes`, `num_levels`, `root`, `max_var`) on GPU.

```rust
// crates/xlog-prob/src/compilation/mod.rs (sketch)
let lookup = cache.lookup_or_insert_device(&key)?;
let handle = lookup.into_handle();

// Always run gated compilation and verification
let d4_config = d4_config_for_smoothing(config, random_vars)?;
let circuit = gpu_d4::compile_gpu_d4_gated(cnf, provider, &d4_config, handle.compile_needed_device())?;
let circuit = if random_vars.is_empty() { circuit } else {
    circuit.smooth_random_vars_device(provider, random_vars, config.smooth_node_cap, config.smooth_edge_cap)?
};
cache.store_from_xgcf(&handle, &circuit)?;
let free_var_mask = gpu_d4::compute_free_var_mask_gpu_gated(cnf, &circuit, provider, handle.compile_needed_device())?;
cache.store_free_var_mask(&handle, &free_var_mask)?;
validate_equivalence_gpu_gated(cnf, &circuit, provider, GpuEquivalenceConfig { cdcl }, handle.compile_needed_device())?;
Ok(handle)
```

```rust
// crates/xlog-prob/src/compilation/gpu_cache.rs (sketch)
pub struct GpuCircuitCacheHandle {
    slot: TrackedCudaSlice<u32>,
    compile_needed: TrackedCudaSlice<u32>,
    meta_num_nodes: TrackedCudaSlice<u32>,
    meta_num_levels: TrackedCudaSlice<u32>,
    meta_root: TrackedCudaSlice<u32>,
    meta_max_var: TrackedCudaSlice<u32>,
}

impl GpuCircuitCacheHandle {
    pub fn meta_num_nodes_device(&self) -> &TrackedCudaSlice<u32> { &self.meta_num_nodes }
    pub fn meta_num_levels_device(&self) -> &TrackedCudaSlice<u32> { &self.meta_num_levels }
    pub fn meta_root_device(&self) -> &TrackedCudaSlice<u32> { &self.meta_root }
    pub fn meta_max_var_device(&self) -> &TrackedCudaSlice<u32> { &self.meta_max_var }
}
```

```rust
// crates/xlog-prob/src/compilation/gpu_cache.rs (handle init sketch)
let slot = lookup.slot_device().clone();
let compile_needed = lookup.compile_needed_device().clone();
let meta_num_nodes = cache.meta_num_nodes_device().slice(slot_idx..slot_idx + 1).into();
// same for num_levels/root/max_var
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-prob --test no_dtoh_in_gpu_cache_path`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/xlog-prob/src/compilation/mod.rs crates/xlog-prob/src/compilation/gpu_cache.rs crates/xlog-prob/tests/no_dtoh_in_gpu_cache_path.rs
git commit -m "feat: remove host reads from GPU cache compile path"
```

---

### Task 3: Device-only evaluation kernels (no host-per-level loops)

**Files:**
- Modify: `kernels/circuit.cu`
- Modify: `crates/xlog-cuda/src/kernels.rs`
- Modify: `crates/xlog-prob/src/gpu.rs`
- Modify: `crates/xlog-prob/src/compilation/gpu_cache.rs`
- Test: `crates/xlog-prob/tests/gpu_eval_device_only.rs`

**Step 1: Write the failing test**

```rust
// crates/xlog-prob/tests/gpu_eval_device_only.rs
use xlog_prob::tests::small_xgcf_fixture;

#[test]
fn gpu_eval_device_only_matches_host_eval() {
    let (provider, xgcf, weights) = small_xgcf_fixture();
    let mut cache = xgcf.to_cache(&provider).unwrap();
    cache.store_weights(cache.handle(), &weights.log_true, &weights.log_false).unwrap();

    let mut out = provider.memory().alloc::<f64>(1).unwrap();
    cache.eval_log_wmc_device_only(cache.handle(), &mut out).unwrap();

    let host = cache.eval_log_wmc_host(cache.handle()).unwrap();
    let mut host_out = [0.0_f64];
    provider.device().inner().dtoh_sync_copy_into(&out, &mut host_out).unwrap();
    assert!((host_out[0] - host).abs() < 1e-9);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-prob --test gpu_eval_device_only`
Expected: FAIL (missing `eval_log_wmc_device_only`).

**Step 3: Write minimal implementation**

Add a single-kernel forward and backward evaluation that loops levels on device using device meta.

```cuda
// kernels/circuit.cu (sketch)
extern "C" __global__ void xgcf_eval_all_levels(
    const uint8_t* node_type,
    const int32_t* lit,
    const uint32_t* level_nodes,
    const uint32_t* level_offsets,
    const uint32_t* adj,
    const double* log_true,
    const double* log_false,
    const uint32_t* meta_num_levels,
    const uint32_t* meta_root,
    double* values
) {
    uint32_t num_levels = meta_num_levels[0];
    for (uint32_t level = 0; level < num_levels; ++level) {
        // existing per-level kernel body (inlined) with bounds
        __syncthreads();
    }
}
```

```rust
// crates/xlog-prob/src/compilation/gpu_cache.rs (sketch)
pub fn eval_log_wmc_device_only(
    &mut self,
    handle: &GpuCircuitCacheHandle,
    out: &mut TrackedCudaSlice<f64>,
) -> Result<()> {
    let device = self.provider().device().inner();
    let func = device.get_func(CIRCUIT_MODULE, circuit_kernels::XGCF_EVAL_ALL_LEVELS)
        .ok_or_else(|| XlogError::Kernel("xgcf_eval_all_levels kernel not found".to_string()))?;
    unsafe {
        func.clone().launch(LaunchConfig::for_num_elems(self.level_nodes.len() as u32), (
            self.node_type(),
            self.lit(),
            self.level_nodes(),
            self.level_offsets(),
            self.adj(),
            self.var_log_true(),
            self.var_log_false(),
            handle.meta_num_levels_device(),
            handle.meta_root_device(),
            self.values_mut(),
        ))
    }?;
    // read root value into `out` on device (no DTOH)
    self.read_root_into_device(handle, out)?;
    Ok(())
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-prob --test gpu_eval_device_only`
Expected: PASS

**Step 5: Commit**

```bash
git add kernels/circuit.cu crates/xlog-cuda/src/kernels.rs crates/xlog-prob/src/gpu.rs crates/xlog-prob/src/compilation/gpu_cache.rs crates/xlog-prob/tests/gpu_eval_device_only.rs
git commit -m "feat: add device-only XGCF evaluation kernels"
```

---

### Task 4: GPU-resident query mapping (no DTOH for query vars)

**Files:**
- Modify: `crates/xlog-prob/src/compilation/gpu_weights.rs`
- Modify: `kernels/weights.cu`
- Modify: `crates/xlog-prob/src/exact_gpu.rs`
- Test: `crates/xlog-prob/tests/gpu_query_vars_device.rs`

**Step 1: Write the failing test**

```rust
// crates/xlog-prob/tests/gpu_query_vars_device.rs
use xlog_prob::tests::small_query_fixture;

#[test]
fn query_var_mapping_device_only() {
    let (provider, vars, query_nodes) = small_query_fixture();
    let node_ids = xlog_prob::compilation::gpu_weights::upload_u32(&provider, &query_nodes).unwrap();
    let vars_device = xlog_prob::compilation::gpu_weights::map_nodes_to_vars_gpu(
        &vars.node_var,
        &node_ids,
        vars.max_var,
        &provider,
    ).unwrap();

    // DTOH is allowed in test to validate output.
    let mut host = vec![0u32; vars_device.len()];
    provider.device().inner().dtoh_sync_copy_into(&vars_device, &mut host).unwrap();
    assert_eq!(host.len(), query_nodes.len());
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-prob --test gpu_query_vars_device`
Expected: FAIL until query mapping is wired into GPU-only path.

**Step 3: Write minimal implementation**

- Ensure GPU-only compilation uses device query vars and never reads them back on host.
- Add a kernel to apply query restriction directly from device query var IDs.

```cuda
// kernels/weights.cu (sketch)
extern "C" __global__ void weights_apply_query_vars(
    const uint32_t* query_vars,
    uint32_t num_queries,
    double* log_false,
    double* saved
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_queries) return;
    uint32_t var = query_vars[i];
    if (var == 0) return;
    saved[i] = log_false[var];
    log_false[var] = -INFINITY; // force false
}
```

```rust
// crates/xlog-prob/src/exact_gpu.rs (sketch)
if let Some(vars_device) = &state.query_vars_device {
    apply_query_vars_device(
        &state.provider,
        vars_device,
        cache.var_log_false_mut(),
        &mut restore_buf,
    )?;
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-prob --test gpu_query_vars_device`
Expected: PASS

**Step 5: Commit**

```bash
git add kernels/weights.cu crates/xlog-prob/src/compilation/gpu_weights.rs crates/xlog-prob/src/exact_gpu.rs crates/xlog-prob/tests/gpu_query_vars_device.rs
git commit -m "feat: map query vars on device for GPU-only exact path"
```

---

### Task 5: SAT/CDCL certification category (G07)

**Files:**
- Create: `crates/xlog-cuda-tests/src/categories/g07_sat_cdcl.rs`
- Modify: `crates/xlog-cuda-tests/src/categories/mod.rs`
- Modify: `crates/xlog-cuda-tests/tests/certification_suite.rs`
- Modify: `crates/xlog-cuda-tests/Cargo.toml`
- Modify: `Cargo.lock`

**Step 1: Write the failing test**

```rust
// crates/xlog-cuda-tests/tests/certification_suite.rs (add)
println!("Running G07: SAT/CDCL...");
results.add_category(categories::g07_sat_cdcl::run_all(&ctx));
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-cuda-tests --test certification_suite --release -- --nocapture`
Expected: FAIL because category module is missing.

**Step 3: Write minimal implementation**

```rust
// crates/xlog-cuda-tests/src/categories/g07_sat_cdcl.rs
use xlog_cuda_tests::{Category, CategoryId, TestResult};
use xlog_solve::{GpuCdclConfig, GpuCdclSolver, GpuCnf, Literal, Clause, SolveInstance};
use xlog_cuda::CudaKernelProvider;
use std::sync::Arc;

pub fn category() -> Category {
    Category::new(CategoryId::new("g07"), "sat_cdcl", vec![
        test_sat_small(),
        test_unsat_small(),
        test_model_check(),
        test_proof_check(),
    ])
}

fn test_sat_small() -> TestResult {
    // CNF: (x1) AND (x2 OR ~x1)
    let cnf = GpuCnf::from_clauses(2, vec![
        Clause::new(vec![Literal::pos(1)]),
        Clause::new(vec![Literal::pos(2), Literal::neg(1)]),
    ]).unwrap();
    let provider = Arc::new(CudaKernelProvider::new_default().unwrap());
    let solver = GpuCdclSolver::new(provider, GpuCdclConfig::default());
    solver.solve_expect_sat(&cnf).unwrap();
    TestResult::pass()
}

// Similar UNSAT, model check, proof check cases.
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-cuda-tests --test certification_suite --release -- --nocapture`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/xlog-cuda-tests/src/categories/g07_sat_cdcl.rs crates/xlog-cuda-tests/src/categories/mod.rs crates/xlog-cuda-tests/Cargo.toml Cargo.lock crates/xlog-cuda-tests/tests/certification_suite.rs
git commit -m "feat: add SAT/CDCL certification category"
```

---

### Task 6: GPU-native Monte Carlo pipeline

**Files:**
- Modify: `crates/xlog-prob/src/mc.rs`
- Modify: `crates/xlog-prob/src/lib.rs`
- Modify: `crates/xlog-runtime/src/executor.rs`
- Modify: `crates/xlog-cuda/src/provider/mod.rs`
- Create: `kernels/mc_eval.cu`
- Test: `crates/xlog-prob/tests/gpu_mc_vs_cpu.rs`

**Step 1: Write the failing test**

```rust
// crates/xlog-prob/tests/gpu_mc_vs_cpu.rs
use xlog_prob::{McProgram, McEvalConfig};

#[test]
fn gpu_mc_matches_cpu_on_small_program() {
    let prog = McProgram::compile_source("p(1) :- coin(0.3).");
    let cpu = prog.evaluate(McEvalConfig::default()).unwrap();
    let gpu = prog.evaluate_gpu(McEvalConfig::default()).unwrap();
    assert!((cpu.query_probs[0] - gpu.query_probs[0]).abs() < 0.02);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-prob --test gpu_mc_vs_cpu`
Expected: FAIL (no `evaluate_gpu`).

**Step 3: Write minimal implementation**

- Add `CudaKernelProvider::sample_bernoulli_matrix_device` returning `TrackedCudaSlice<u8>`.
- Add GPU MC execution: build per-sample fact tables on GPU, execute GPU runtime per batch,
  and reduce query counts on device.

```rust
// crates/xlog-prob/src/mc.rs (sketch)
pub fn evaluate_gpu(&self, cfg: McEvalConfig) -> Result<McResult> {
    let provider = self.provider()?;
    let samples = provider.sample_bernoulli_matrix_device(&self.bernoulli_probs, cfg.samples, cfg.seed)?;
    let mut executor = xlog_runtime::Executor::new(provider.clone());
    let counts = execute_mc_on_gpu(&mut executor, &samples, &self.plan, &self.queries)?;
    Ok(counts.to_result(cfg))
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-prob --test gpu_mc_vs_cpu`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/xlog-prob/src/mc.rs crates/xlog-runtime/src/executor.rs crates/xlog-cuda/src/provider/mod.rs kernels/mc_eval.cu crates/xlog-prob/tests/gpu_mc_vs_cpu.rs
git commit -m "feat: add GPU-native Monte Carlo evaluation"
```

---

### Task 7: Zero-copy Arrow device interop

**Files:**
- Modify: `crates/xlog-cuda/src/interop/arrow.rs`
- Modify: `crates/xlog-cuda/src/lib.rs`
- Test: `crates/xlog-cuda-tests/tests/arrow_device_zero_copy.rs`

**Step 1: Write the failing test**

```rust
// crates/xlog-cuda-tests/tests/arrow_device_zero_copy.rs
use xlog_cuda::interop::arrow::to_arrow_device_record_batch;

#[test]
fn arrow_device_export_is_zero_copy() {
    let (provider, buffer) = xlog_cuda_tests::fixtures::small_table();
    let batch = to_arrow_device_record_batch(&provider, &buffer).unwrap();
    // Validate device pointer export metadata (no host copy).
    assert!(batch.schema().metadata().contains_key("cuda_ptr"));
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-cuda-tests --test arrow_device_zero_copy --release -- --nocapture`
Expected: FAIL (API missing).

**Step 3: Write minimal implementation**

- Add Arrow C Data Interface export for device buffers.
- Encode CUDA device pointer in Arrow metadata and ensure buffers are not copied.

```rust
// crates/xlog-cuda/src/interop/arrow.rs (sketch)
pub fn to_arrow_device_record_batch(
    provider: &CudaKernelProvider,
    buffer: &CudaBuffer,
) -> Result<arrow::record_batch::RecordBatch> {
    // Use arrow::ffi to construct arrays with device pointer buffers and metadata.
    // No device->host copy allowed.
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-cuda-tests --test arrow_device_zero_copy --release -- --nocapture`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/xlog-cuda/src/interop/arrow.rs crates/xlog-cuda/src/lib.rs crates/xlog-cuda-tests/tests/arrow_device_zero_copy.rs
git commit -m "feat: add zero-copy Arrow device export"
```

---

### Task 8: Documentation updates (gaps closed)

**Files:**
- Modify: `docs/design/2026-01-28-gpu-native-gaps.md`
- Modify: `docs/architecture/xlog-prob.md`
- Modify: `docs/architecture/cuda-certification.md`
- Modify: `docs/architecture/cudf-interop.md`
- Modify: `docs/ROADMAP.md`
- Modify: `CHANGELOG.md`

**Step 1: Write the failing test**

No test required.

**Step 2: Update docs**

- Mark closed gaps with dates and references to commits/files.
- Update CUDA certification section to include SAT/CDCL category.
- Update MC and Arrow interop to reflect GPU-native paths.

**Step 3: Commit**

```bash
git add docs/design/2026-01-28-gpu-native-gaps.md docs/architecture/xlog-prob.md docs/architecture/cuda-certification.md docs/architecture/cudf-interop.md docs/ROADMAP.md CHANGELOG.md
git commit -m "docs: update GPU-native gap status and certifications"
```

---

Plan complete and saved to `docs/plans/2026-01-28-gpu-native-gap-closure.md`. Two execution options:

1. Subagent-Driven (this session) - I dispatch fresh subagent per task, review between tasks, fast iteration
2. Parallel Session (separate) - Open new session with executing-plans, batch execution with checkpoints

Which approach?
