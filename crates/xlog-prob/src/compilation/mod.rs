//! GPU-native knowledge compilation.
//!
//! This module is the home of GPU-native compilation + verification utilities.
//!
//! Production correctness requires the GPU CDCL equivalence verifier (see `validation`).

use std::sync::Arc;

use xlog_core::{Result, XlogError};
use xlog_cuda::CudaKernelProvider;
use xlog_solve::{GpuCdclConfig, GpuCnf};

use crate::gpu::GpuXgcf;
use crate::compilation::gpu_cache::{GpuCircuitCache, GpuCircuitCacheHandle};

pub mod gpu_d4;
pub mod gpu_cache;
pub mod gpu_cnf;
pub mod gpu_pir;
pub mod gpu_pir_intern;
pub mod gpu_weights;
pub mod sparse_matrix;
pub mod validation;

pub use gpu_d4::GpuCompileConfig;
pub use gpu_cnf::{encode_cnf_gpu, GpuCnfEncoding, GpuCnfVarTables};
pub use gpu_pir::{
    GpuPirGraph, GpuPirRoots, PIR_AND, PIR_CONST, PIR_DECISION, PIR_LIT, PIR_NEG_LIT, PIR_OR,
};
pub use gpu_pir_intern::{GpuPirInterner, PirBatch};
pub use gpu_weights::{build_evidence_by_var_gpu, build_weights_gpu, map_nodes_to_vars_gpu, GpuWeights};
pub use sparse_matrix::GpuCsrCnf;
pub use validation::{
    check_equivalence_gpu, check_equivalence_gpu_gated, validate_equivalence_gpu,
    validate_equivalence_gpu_gated, GpuEquivalenceConfig,
};

/// Compile CNF on GPU, then verify equivalence with GPU CDCL.
pub fn compile_gpu_d4_and_verify(
    cnf: &GpuCnf,
    provider: &Arc<CudaKernelProvider>,
    config: &GpuCompileConfig,
    random_vars: &[u32],
) -> Result<GpuXgcf> {
    if config.cdcl_conflict_budget.is_some() {
        return Err(XlogError::Compilation(
            "cdcl_conflict_budget is not supported by the GPU CDCL verifier".to_string(),
        ));
    }
    let d4_config = d4_config_for_smoothing(config, random_vars)?;
    let mut circuit = gpu_d4::compile_gpu_d4(cnf, provider, &d4_config)?;
    if !random_vars.is_empty() {
        circuit = circuit.smooth_random_vars_device(
            provider,
            random_vars,
            config.smooth_node_cap,
            config.smooth_edge_cap,
        )?;
    }
    let cdcl = cdcl_config_from_compile(config)?;
    validate_equivalence_gpu(cnf, &circuit, provider, GpuEquivalenceConfig { cdcl })?;
    Ok(circuit)
}

/// Compile CNF on GPU, cache the circuit, then verify equivalence with GPU CDCL.
pub fn compile_gpu_d4_and_verify_cached(
    cnf: &GpuCnf,
    provider: &Arc<CudaKernelProvider>,
    config: &GpuCompileConfig,
    cache: &mut GpuCircuitCache,
    random_vars: &[u32],
) -> Result<GpuCircuitCacheHandle> {
    if config.cdcl_conflict_budget.is_some() {
        return Err(XlogError::Compilation(
            "cdcl_conflict_budget is not supported by the GPU CDCL verifier".to_string(),
        ));
    }

    let key = gpu_cache::hash_cnf_gpu(cnf, provider)?;
    let lookup = cache.lookup_or_insert_device(&key)?;
    let mut handle = lookup.into_handle();

    let mut compile_needed_host = [0u32; 1];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(handle.compile_needed_device(), &mut compile_needed_host)
        .map_err(|e| XlogError::Kernel(format!("Failed to read compile_needed: {}", e)))?;
    if compile_needed_host[0] == 0 {
        load_cache_handle_meta(cache, &mut handle)?;
        return Ok(handle);
    }

    let d4_config = d4_config_for_smoothing(config, random_vars)?;
    let circuit = gpu_d4::compile_gpu_d4_gated(
        cnf,
        provider,
        &d4_config,
        handle.compile_needed_device(),
    )?;
    let circuit = if random_vars.is_empty() {
        circuit
    } else {
        let smoothed = circuit.smooth_random_vars_device(
            provider,
            random_vars,
            config.smooth_node_cap,
            config.smooth_edge_cap,
        )?;
        smoothed
    };
    cache.store_from_xgcf(&mut handle, &circuit)?;

    let free_var_mask =
        gpu_d4::compute_free_var_mask_gpu_gated(cnf, &circuit, provider, handle.compile_needed_device())?;
    cache.store_free_var_mask(&mut handle, &free_var_mask)?;

    let cdcl = cdcl_config_from_compile(config)?;
    validate_equivalence_gpu_gated(
        cnf,
        &circuit,
        provider,
        GpuEquivalenceConfig { cdcl },
        handle.compile_needed_device(),
    )?;
    Ok(handle)
}

fn load_cache_handle_meta(
    cache: &GpuCircuitCache,
    handle: &mut GpuCircuitCacheHandle,
) -> Result<()> {
    let device = cache.provider().device().inner();
    let mut slot_host = [0u32; 1];
    device
        .dtoh_sync_copy_into(handle.slot_device(), &mut slot_host)
        .map_err(|e| XlogError::Kernel(format!("Failed to read cache slot: {}", e)))?;
    let slot = slot_host[0] as usize;
    if slot >= cache.num_slots() as usize {
        return Err(XlogError::Compilation(format!(
            "GpuCircuitCache meta slot {} out of bounds (num_slots={})",
            slot,
            cache.num_slots()
        )));
    }

    let mut num_nodes = [0u32; 1];
    let mut num_levels = [0u32; 1];
    let mut root = [0u32; 1];
    let mut max_var = [0u32; 1];

    let nodes_view = cache.meta_num_nodes_device().slice(slot..slot + 1);
    let levels_view = cache.meta_num_levels_device().slice(slot..slot + 1);
    let root_view = cache.meta_root_device().slice(slot..slot + 1);
    let var_view = cache.meta_max_var_device().slice(slot..slot + 1);

    device
        .dtoh_sync_copy_into(&nodes_view, &mut num_nodes)
        .map_err(|e| XlogError::Kernel(format!("Failed to read cache num_nodes: {}", e)))?;
    device
        .dtoh_sync_copy_into(&levels_view, &mut num_levels)
        .map_err(|e| XlogError::Kernel(format!("Failed to read cache num_levels: {}", e)))?;
    device
        .dtoh_sync_copy_into(&root_view, &mut root)
        .map_err(|e| XlogError::Kernel(format!("Failed to read cache root: {}", e)))?;
    device
        .dtoh_sync_copy_into(&var_view, &mut max_var)
        .map_err(|e| XlogError::Kernel(format!("Failed to read cache max_var: {}", e)))?;

    if num_nodes[0] == 0 || num_levels[0] == 0 {
        return Err(XlogError::Compilation(
            "GpuCircuitCache meta missing for slot".to_string(),
        ));
    }

    handle.set_meta(num_nodes[0], num_levels[0], root[0], max_var[0]);
    Ok(())
}

fn d4_config_for_smoothing(
    config: &GpuCompileConfig,
    random_vars: &[u32],
) -> Result<GpuCompileConfig> {
    if random_vars.is_empty() {
        return Ok(*config);
    }
    let headroom = 2u32
        .checked_add(random_vars.len() as u32)
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
