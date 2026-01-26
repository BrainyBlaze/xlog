//! GPU-native D4 compilation (Phase 1).
//!
//! This module provides the configuration and kernel-facing utilities needed
//! to compile a device-resident CNF into a device-resident XGCF circuit.
//!
//! Primary spec: docs/design/2026-01-22-gpu-native-compilation-design.md (Section 5.2.4).

use cudarc::driver::{LaunchAsync, LaunchConfig};
use xlog_core::{Result, XlogError};
use xlog_cuda::provider::{d4_kernels, D4_MODULE};
use xlog_cuda::CudaKernelProvider;
use xlog_solve::GpuCnf;

/// Configuration for GPU D4 + GPU CDCL.
///
/// This is the public control-plane contract for the GPU-native compilation pipeline.
#[derive(Debug, Clone, Copy)]
pub struct GpuCompileConfig {
    /// BFS expansion depth before handing each frontier item to a per-block DFS worker.
    pub frontier_depth: u16,
    /// Hard cap on the number of frontier work items (overflow is a hard error).
    pub max_frontier_items: u32,
    /// Absolute depth cap (defensive); exceeding this is a hard error (no UNKNOWN).
    pub max_depth: u16,

    /// CDCL restart cadence (deterministic).
    pub cdcl_restart_interval: u32,
    /// Learned clause arena size (bytes) for the verifier instance.
    pub cdcl_learned_bytes: u64,
    /// Optional conflict budget for debug/profiling only; production must be unbounded.
    pub cdcl_conflict_budget: Option<u64>,
}

/// Validate `GpuCnf` CSR invariants on the GPU (fail-fast trap on invalid input).
///
/// This is used as a mandatory invariant check for GPU-native compilation paths where
/// the host cannot safely "peek" into device-resident CNF buffers.
pub fn validate_cnf_gpu(cnf: &GpuCnf, provider: &CudaKernelProvider) -> Result<()> {
    let device = provider.device().inner();
    let validate = device
        .get_func(D4_MODULE, d4_kernels::D4_VALIDATE_CNF)
        .ok_or_else(|| XlogError::Kernel("d4_validate_cnf kernel not found".to_string()))?;

    let var_cap = cnf.var_cap;
    let clause_cap = cnf.clause_cap;
    let lit_cap = cnf.lit_cap;

    // SAFETY: d4_validate_cnf(var_cap, clause_cap, lit_cap, num_vars*, num_clauses*, num_lits*, offsets*, lits*)
    unsafe {
        validate.clone().launch(
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                var_cap,
                clause_cap,
                lit_cap,
                &cnf.num_vars,
                &cnf.num_clauses,
                &cnf.num_lits,
                &cnf.clause_offsets,
                &cnf.literals,
            ),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("d4_validate_cnf failed: {}", e)))?;

    Ok(())
}

