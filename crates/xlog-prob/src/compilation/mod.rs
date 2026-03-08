//! GPU-native knowledge compilation.
//!
//! This module is the home of GPU-native compilation + verification utilities.
//!
//! Production correctness requires the GPU CDCL equivalence verifier (see `validation`).

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::DeviceSlice;
use xlog_core::{Result, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::CudaKernelProvider;
use xlog_solve::{GpuCdclConfig, GpuCnf};

use crate::compilation::gpu_cache::{GpuCircuitCache, GpuCircuitCacheHandle};
use crate::gpu::GpuXgcf;

pub mod disk_cache;
pub mod gpu_cache;
pub mod gpu_cnf;
pub mod gpu_d4;
pub mod gpu_pir;
pub mod gpu_pir_intern;
pub mod gpu_weights;
pub mod sparse_matrix;
pub mod validation;

pub use gpu_cnf::{encode_cnf_gpu, GpuCnfEncoding, GpuCnfVarTables};
pub use gpu_d4::GpuCompileConfig;
pub use gpu_pir::{
    GpuPirGraph, GpuPirRoots, PIR_AND, PIR_CONST, PIR_DECISION, PIR_LIT, PIR_NEG_LIT, PIR_OR,
};
pub use gpu_pir_intern::{GpuPirInterner, PirBatch};
pub use gpu_weights::{
    apply_query_vars_device, build_evidence_by_var_gpu, build_weights_gpu, map_nodes_to_vars_gpu,
    restore_query_vars_device, GpuWeights,
};
pub use sparse_matrix::GpuCsrCnf;
pub use validation::{
    build_equivalence_queries_gpu, check_equivalence_gpu, check_equivalence_gpu_gated,
    validate_equivalence_gpu, validate_equivalence_gpu_gated, GpuEquivalenceConfig,
    GpuEquivalenceQueries,
};

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
    pub disk_cache_hit: bool,
}

fn warmup_profiling_enabled() -> bool {
    std::env::var("XLOG_WARMUP_PROFILE")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Device-resident random-variable list for GPU smoothing.
pub struct DeviceRandomVarList {
    list: TrackedCudaSlice<u32>,
    count: u32,
}

impl DeviceRandomVarList {
    pub fn from_device(list: TrackedCudaSlice<u32>, count: u32) -> Result<Self> {
        let len = u32::try_from(list.len()).map_err(|_| {
            XlogError::Compilation("DeviceRandomVarList: list length exceeds u32".to_string())
        })?;
        if count > len {
            return Err(XlogError::Compilation(format!(
                "DeviceRandomVarList: count {} exceeds list len {}",
                count, len
            )));
        }
        Ok(Self { list, count })
    }

    pub fn from_host(provider: &CudaKernelProvider, host: &[u32]) -> Result<Self> {
        let memory = provider.memory();
        let mut list = memory.alloc::<u32>(host.len())?;
        if !host.is_empty() {
            provider
                .device()
                .inner()
                .htod_sync_copy_into(host, &mut list)
                .map_err(|e| {
                    XlogError::Kernel(format!("DeviceRandomVarList upload failed: {}", e))
                })?;
        }
        let count = u32::try_from(host.len()).map_err(|_| {
            XlogError::Compilation("DeviceRandomVarList: host len exceeds u32".to_string())
        })?;
        Ok(Self { list, count })
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn count(&self) -> u32 {
        self.count
    }

    pub fn list(&self) -> &TrackedCudaSlice<u32> {
        &self.list
    }
}

/// Compile CNF on GPU, then verify equivalence with GPU CDCL.
pub fn compile_gpu_d4_and_verify(
    cnf: &GpuCnf,
    decision_var_limit: &TrackedCudaSlice<u32>,
    provider: &Arc<CudaKernelProvider>,
    config: &GpuCompileConfig,
) -> Result<GpuXgcf> {
    if config.cdcl_conflict_budget.is_some() {
        return Err(XlogError::Compilation(
            "cdcl_conflict_budget is not supported by the GPU CDCL verifier".to_string(),
        ));
    }
    let circuit = gpu_d4::compile_gpu_d4(cnf, provider, config)?;
    let cdcl = cdcl_config_from_compile(config)?;
    validate_equivalence_gpu(
        cnf,
        decision_var_limit,
        &circuit,
        provider,
        GpuEquivalenceConfig { cdcl, reuse_workspace: config.incremental_verify },
    )?;
    Ok(circuit)
}

/// Compile CNF on GPU, cache the circuit, then verify equivalence with GPU CDCL.
///
/// `canonical_cnf_hash`: a process-independent hash of the PIR structure, used as
/// the `cnf_hash` in the disk cache key. Computed via [`crate::cnf::canonical_pir_hash`].
/// If `None`, disk caching is skipped.
pub fn compile_gpu_d4_and_verify_cached(
    cnf: &GpuCnf,
    decision_var_limit: &TrackedCudaSlice<u32>,
    provider: &Arc<CudaKernelProvider>,
    config: &GpuCompileConfig,
    cache: &mut GpuCircuitCache,
    random_vars: &DeviceRandomVarList,
    canonical_cnf_hash: Option<u64>,
) -> Result<(GpuCircuitCacheHandle, Option<CircuitCompileProfile>)> {
    if config.cdcl_conflict_budget.is_some() {
        return Err(XlogError::Compilation(
            "cdcl_conflict_budget is not supported by the GPU CDCL verifier".to_string(),
        ));
    }

    let profiling = warmup_profiling_enabled();
    let mut profile = CircuitCompileProfile::default();

    // --- CNF hash stage ---
    #[cfg(debug_assertions)]
    eprintln!("[xlog-prob] compile_gpu_d4_and_verify_cached: hash_cnf_gpu");
    let t_hash = if profiling { Some(Instant::now()) } else { None };
    let key = gpu_cache::hash_cnf_gpu(cnf, provider)?;
    if let Some(t0) = t_hash {
        provider
            .device()
            .synchronize()
            .map_err(|e| XlogError::Kernel(format!("sync after hash_cnf_gpu: {}", e)))?;
        profile.cnf_hash_sec = t0.elapsed().as_secs_f64();
    }
    #[cfg(debug_assertions)]
    {
        if !profiling {
            provider
                .device()
                .synchronize()
                .map_err(|e| {
                    XlogError::Kernel(format!("sync after hash_cnf_gpu failed: {}", e))
                })?;
        }
    }
    #[cfg(debug_assertions)]
    eprintln!("[xlog-prob] compile_gpu_d4_and_verify_cached: lookup_or_insert_device");
    let lookup = cache.lookup_or_insert_device(&key)?;
    let mut handle = lookup.into_handle()?;

    // --- Disk cache check (only on GPU cache miss) ---
    //
    // D→H copy compile_needed to decide whether we need to compile at all.
    // If compile_needed == 0, the GPU cache already has the circuit (GPU cache hit).
    // If compile_needed == 1, we check the disk cache before falling through to D4.
    let compile_needed_host: Vec<u32> = provider
        .device()
        .inner()
        .dtoh_sync_copy(handle.compile_needed_device())
        .map_err(|e| XlogError::Kernel(format!("dtoh compile_needed: {}", e)))?;
    let compile_needed = compile_needed_host[0];

    // GPU cache hit — short-circuit the entire compile pipeline.
    if compile_needed == 0 {
        profile.gpu_cache_hit = true;
        let out_profile = if profiling { Some(profile) } else { None };
        return Ok((handle, out_profile));
    }

    // Build the disk cache key (we know compile_needed == 1 at this point).
    // Uses the caller-supplied canonical PIR hash (process-independent) instead of the
    // GPU CNF hash (which varies per process due to PirNodeId non-determinism).
    let cache_key = if compile_needed == 1 {
        if let Some(cnf_hash) = canonical_cnf_hash {
            let config_hash = hash_compile_config(config);
            let random_vars_hash = hash_random_vars(random_vars, provider)?;
            let sm = detect_compute_capability(provider)?;
            Some(disk_cache::CircuitCacheKey {
                cnf_hash,
                config_hash,
                random_vars_hash,
                sm,
            })
        } else {
            None
        }
    } else {
        None
    };

    // Check disk cache on GPU cache miss
    if let Some(ref disk_key) = cache_key {
        #[cfg(debug_assertions)]
        eprintln!("[xlog-prob] compile_gpu_d4_and_verify_cached: checking disk cache");
        if let Ok(Some(artifact)) = disk_cache::read_artifact(disk_key) {
            #[cfg(debug_assertions)]
            eprintln!("[xlog-prob] compile_gpu_d4_and_verify_cached: disk cache hit");
            cache.restore_from_host_arrays(&mut handle, &artifact)?;
            provider
                .device()
                .synchronize()
                .map_err(|e| {
                    XlogError::Kernel(format!("sync after disk cache restore: {}", e))
                })?;
            profile.disk_cache_hit = true;
            let out_profile = if profiling { Some(profile) } else { None };
            return Ok((handle, out_profile));
        }
        #[cfg(debug_assertions)]
        eprintln!("[xlog-prob] compile_gpu_d4_and_verify_cached: disk cache miss");
    }

    let d4_config = d4_config_for_smoothing(config, random_vars.count())?;

    // --- D4 compile stage ---
    #[cfg(debug_assertions)]
    eprintln!("[xlog-prob] compile_gpu_d4_and_verify_cached: compile_gpu_d4_gated");
    let t_d4 = if profiling { Some(Instant::now()) } else { None };
    let circuit_base =
        gpu_d4::compile_gpu_d4_gated(cnf, provider, &d4_config, handle.compile_needed_device())?;
    if let Some(t0) = t_d4 {
        provider
            .device()
            .synchronize()
            .map_err(|e| XlogError::Kernel(format!("sync after d4 compile: {}", e)))?;
        profile.d4_compile_sec = t0.elapsed().as_secs_f64();
    }
    #[cfg(debug_assertions)]
    {
        if !profiling {
            provider.device().synchronize().map_err(|e| {
                XlogError::Kernel(format!("sync after compile_gpu_d4_gated failed: {}", e))
            })?;
        }
    }
    if circuit_base.num_nodes() == 0 || circuit_base.num_levels() == 0 {
        // Defensive: D4 returned an empty circuit (the primary GPU cache hit is handled
        // by the compile_needed == 0 early return above; this catches degenerate CNFs).
        let out_profile = if profiling { Some(profile) } else { None };
        return Ok((handle, out_profile));
    }

    // --- Verify equivalence stage ---
    //
    // Verify equivalence on the *base* circuit (pre-smoothing) to keep the verifier CNFs minimal.
    //
    // `encode_cnf_gpu` sets `decision_var_limit` to the end of the leaf+choice var range. For
    // deterministic programs with no probabilistic vars, this range is empty (limit=0). In that
    // case, the verifier must still be able to branch, so fall back to `cnf.num_vars` (all CNF
    // vars are semantically meaningful when there is no probabilistic decision set).
    let verifier_decision_var_limit = if random_vars.is_empty() {
        &cnf.num_vars
    } else {
        decision_var_limit
    };
    let cdcl = cdcl_config_from_compile(config)?;
    #[cfg(debug_assertions)]
    eprintln!("[xlog-prob] compile_gpu_d4_and_verify_cached: validate_equivalence_gpu_gated");
    let t_verify = if profiling { Some(Instant::now()) } else { None };
    validate_equivalence_gpu_gated(
        cnf,
        verifier_decision_var_limit,
        &circuit_base,
        provider,
        GpuEquivalenceConfig { cdcl, reuse_workspace: config.incremental_verify },
        handle.compile_needed_device(),
    )?;
    if let Some(t0) = t_verify {
        provider
            .device()
            .synchronize()
            .map_err(|e| XlogError::Kernel(format!("sync after verify: {}", e)))?;
        profile.verify_sec = t0.elapsed().as_secs_f64();
    }
    #[cfg(debug_assertions)]
    {
        if !profiling {
            provider.device().synchronize().map_err(|e| {
                XlogError::Kernel(format!(
                    "sync after validate_equivalence_gpu_gated failed: {}",
                    e
                ))
            })?;
        }
    }

    // --- Smoothing stage ---
    //
    // Smoothing is evaluation-only (WMC/grad correctness); it is semantics-preserving and does not
    // need to participate in the equivalence check.
    let t_smooth = if profiling { Some(Instant::now()) } else { None };
    let circuit_eval = if random_vars.is_empty() {
        circuit_base
    } else {
        #[cfg(debug_assertions)]
        eprintln!("[xlog-prob] compile_gpu_d4_and_verify_cached: smooth_random_vars_device");
        let smoothed = circuit_base.smooth_random_vars_device(
            provider,
            random_vars.list(),
            random_vars.count(),
            config.smooth_node_cap,
            config.smooth_edge_cap,
        )?;
        #[cfg(debug_assertions)]
        {
            if !profiling {
                provider.device().synchronize().map_err(|e| {
                    XlogError::Kernel(format!(
                        "sync after smooth_random_vars_device failed: {}",
                        e
                    ))
                })?;
            }
        }
        smoothed
    };
    if let Some(t0) = t_smooth {
        provider
            .device()
            .synchronize()
            .map_err(|e| XlogError::Kernel(format!("sync after smooth: {}", e)))?;
        profile.smooth_sec = t0.elapsed().as_secs_f64();
    }

    // --- Cache store stage ---
    #[cfg(debug_assertions)]
    eprintln!("[xlog-prob] compile_gpu_d4_and_verify_cached: store_from_xgcf");
    let t_store = if profiling { Some(Instant::now()) } else { None };
    cache.store_from_xgcf(&mut handle, &circuit_eval)?;
    if let Some(t0) = t_store {
        provider
            .device()
            .synchronize()
            .map_err(|e| XlogError::Kernel(format!("sync after cache store: {}", e)))?;
        profile.cache_store_sec = t0.elapsed().as_secs_f64();
    }
    #[cfg(debug_assertions)]
    {
        if !profiling {
            provider
                .device()
                .synchronize()
                .map_err(|e| {
                    XlogError::Kernel(format!("sync after store_from_xgcf failed: {}", e))
                })?;
        }
    }

    // --- Free-var mask stage ---
    #[cfg(debug_assertions)]
    eprintln!("[xlog-prob] compile_gpu_d4_and_verify_cached: compute_free_var_mask_gpu_gated");
    let t_fvm = if profiling { Some(Instant::now()) } else { None };
    let free_var_mask = gpu_d4::compute_free_var_mask_gpu_gated(
        cnf,
        &circuit_eval,
        provider,
        handle.compile_needed_device(),
    )?;
    #[cfg(debug_assertions)]
    {
        if !profiling {
            provider.device().synchronize().map_err(|e| {
                XlogError::Kernel(format!(
                    "sync after compute_free_var_mask_gpu_gated failed: {}",
                    e
                ))
            })?;
        }
    }
    // Only enable free-var correction if there are actual free variables.
    // When the mask is all-zero (common for smoothed d-DNNF circuits),
    // skipping this keeps has_free_var_mask[slot]=false, which avoids unnecessary
    // free-var correction kernel launches on every subsequent eval.
    let mask_host: Vec<u8> = provider
        .device()
        .inner()
        .dtoh_sync_copy(&free_var_mask)
        .map_err(|e| XlogError::Kernel(format!("Failed to read free_var_mask: {}", e)))?;
    let has_free_vars = mask_host.iter().any(|&b| b != 0);
    #[cfg(debug_assertions)]
    eprintln!(
        "[xlog-prob] free_var_mask: {} free vars, batched eval {}",
        if has_free_vars { "has" } else { "no" },
        if has_free_vars { "DISABLED" } else { "ENABLED" },
    );
    if has_free_vars {
        cache.store_free_var_mask(&mut handle, &free_var_mask)?;
        #[cfg(debug_assertions)]
        {
            if !profiling {
                provider.device().synchronize().map_err(|e| {
                    XlogError::Kernel(format!("sync after store_free_var_mask failed: {}", e))
                })?;
            }
        }
    }
    if let Some(t0) = t_fvm {
        provider
            .device()
            .synchronize()
            .map_err(|e| XlogError::Kernel(format!("sync after free_var_mask: {}", e)))?;
        profile.free_var_mask_sec = t0.elapsed().as_secs_f64();
    }

    // --- Disk cache write (opportunistic) ---
    //
    // After a successful compilation, write the artifact to disk for next warm start.
    // Errors are silently ignored — the disk cache is best-effort.
    if let Some(ref disk_key) = cache_key {
        if let Ok(artifact) = cache.build_artifact_from_device(&handle, provider) {
            let _ = disk_cache::write_artifact(disk_key, &artifact);
            #[cfg(debug_assertions)]
            eprintln!("[xlog-prob] compile_gpu_d4_and_verify_cached: wrote disk cache artifact");
        }
    }

    let out_profile = if profiling { Some(profile) } else { None };
    Ok((handle, out_profile))
}

fn d4_config_for_smoothing(
    config: &GpuCompileConfig,
    random_var_count: u32,
) -> Result<GpuCompileConfig> {
    if random_var_count == 0 {
        return Ok(*config);
    }
    let headroom = 2u32
        .checked_add(random_var_count)
        .ok_or_else(|| XlogError::Compilation("smooth headroom overflow".to_string()))?;
    if config.smooth_node_cap <= headroom {
        return Err(XlogError::Compilation(format!(
            "GpuCompileConfig smooth_node_cap {} too small for smoothing headroom {}",
            config.smooth_node_cap, headroom
        )));
    }
    let base_cap = config
        .smooth_node_cap
        .checked_sub(headroom)
        .ok_or_else(|| XlogError::Compilation("smooth node cap underflow".to_string()))?;
    if base_cap < 3 {
        return Err(XlogError::Compilation(
            "GpuCompileConfig smooth_node_cap leaves <3 base nodes".to_string(),
        ));
    }
    let mut out = *config;
    out.smooth_node_cap = base_cap;
    Ok(out)
}

fn cdcl_config_from_compile(config: &GpuCompileConfig) -> Result<GpuCdclConfig> {
    if config.cdcl_restart_interval == 0 {
        return Err(XlogError::Compilation(
            "cdcl_restart_interval must be > 0".to_string(),
        ));
    }
    if config.cdcl_learned_bytes == 0 {
        return Err(XlogError::Compilation(
            "cdcl_learned_bytes must be > 0".to_string(),
        ));
    }

    // Deterministic sizing: assume average learned clause length = 4.
    const AVG_LEN: u64 = 4;
    const META_BYTES_PER_CLAUSE: u64 = 24; // offsets + lbd + activity + flags + proof offsets (rounded up)
    const PROOF_BYTES_PER_CLAUSE: u64 = 8 + (8 * AVG_LEN); // (conflict, steps) + 2*u32 per lit
    const LIT_BYTES_PER_CLAUSE: u64 = 4 * AVG_LEN;

    let bytes_per_clause = META_BYTES_PER_CLAUSE
        .checked_add(PROOF_BYTES_PER_CLAUSE)
        .and_then(|v| v.checked_add(LIT_BYTES_PER_CLAUSE))
        .ok_or_else(|| XlogError::Compilation("cdcl bytes per clause overflow".to_string()))?;

    let max_clauses = config
        .cdcl_learned_bytes
        .checked_div(bytes_per_clause)
        .ok_or_else(|| XlogError::Compilation("cdcl_learned_bytes div overflow".to_string()))?;
    if max_clauses == 0 {
        return Err(XlogError::Compilation(
            "cdcl_learned_bytes too small for learned clause arena".to_string(),
        ));
    }

    let max_lits = max_clauses
        .checked_mul(AVG_LEN)
        .ok_or_else(|| XlogError::Compilation("max_learned_lits overflow".to_string()))?;
    let max_proof_u32 = max_clauses
        .checked_mul(2 + 2 * AVG_LEN)
        .ok_or_else(|| XlogError::Compilation("max_proof_u32 overflow".to_string()))?;

    let max_learned_clauses = u32::try_from(max_clauses)
        .map_err(|_| XlogError::Compilation("max_learned_clauses exceeds u32::MAX".to_string()))?;
    let max_learned_lits = u32::try_from(max_lits)
        .map_err(|_| XlogError::Compilation("max_learned_lits exceeds u32::MAX".to_string()))?;
    let max_proof_u32 = u32::try_from(max_proof_u32)
        .map_err(|_| XlogError::Compilation("max_proof_u32 exceeds u32::MAX".to_string()))?;

    let reduce_interval = config
        .cdcl_restart_interval
        .checked_mul(20)
        .ok_or_else(|| XlogError::Compilation("cdcl reduce_interval overflow".to_string()))?;

    Ok(GpuCdclConfig {
        max_learned_clauses,
        max_learned_lits,
        max_proof_u32,
        restart_base: config.cdcl_restart_interval,
        reduce_interval,
    })
}

// ---------------------------------------------------------------------------
// Disk cache helpers
// ---------------------------------------------------------------------------

/// FNV-1a 64-bit hash — deterministic across processes and Rust versions.
/// Matches the FNV-1a algorithm used in the GPU hash kernel (kernels/cache.cu).
fn fnv1a_u64(bytes: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Hash the compile config fields that affect circuit topology output.
fn hash_compile_config(config: &GpuCompileConfig) -> u64 {
    let mut buf = Vec::new();
    buf.extend_from_slice(&config.frontier_depth.to_le_bytes());
    buf.extend_from_slice(&config.max_frontier_items.to_le_bytes());
    buf.extend_from_slice(&config.max_depth.to_le_bytes());
    buf.extend_from_slice(&config.smooth_node_cap.to_le_bytes());
    buf.extend_from_slice(&config.smooth_edge_cap.to_le_bytes());
    // CDCL verifier params do not affect the compiled circuit topology,
    // but we include them for safety so a verifier config change invalidates the cache.
    buf.extend_from_slice(&config.cdcl_restart_interval.to_le_bytes());
    buf.extend_from_slice(&config.cdcl_learned_bytes.to_le_bytes());
    fnv1a_u64(&buf)
}

/// Hash the random variable list (D→H copy + hash).
fn hash_random_vars(
    random_vars: &DeviceRandomVarList,
    provider: &Arc<CudaKernelProvider>,
) -> Result<u64> {
    let count = random_vars.count();
    let mut buf = Vec::new();
    buf.extend_from_slice(&count.to_le_bytes());
    if count > 0 {
        let host: Vec<u32> = provider
            .device()
            .inner()
            .dtoh_sync_copy(random_vars.list())
            .map_err(|e| {
                XlogError::Kernel(format!("dtoh random_vars for disk cache hash: {}", e))
            })?;
        // Hash only the valid elements (count may be less than the allocation).
        for &v in &host[..count as usize] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    Ok(fnv1a_u64(&buf))
}

/// Query the device compute capability and encode as `major * 10 + minor` (e.g. 89 for sm_89).
fn detect_compute_capability(provider: &Arc<CudaKernelProvider>) -> Result<u32> {
    use cudarc::driver::sys::CUdevice_attribute;

    let device = provider.device().inner();
    let major = device
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .map_err(|e| {
            XlogError::Kernel(format!("Failed to query compute capability major: {}", e))
        })?;
    let minor = device
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .map_err(|e| {
            XlogError::Kernel(format!("Failed to query compute capability minor: {}", e))
        })?;
    let major_u32: u32 = major.try_into().map_err(|_| {
        XlogError::Kernel(format!(
            "compute capability major {} cannot be converted to u32",
            major
        ))
    })?;
    let minor_u32: u32 = minor.try_into().map_err(|_| {
        XlogError::Kernel(format!(
            "compute capability minor {} cannot be converted to u32",
            minor
        ))
    })?;
    Ok(major_u32 * 10 + minor_u32)
}
