//! GPU-native Decision-DNNF knowledge compilation.
//!
//! This module provides the configuration and kernel-facing utilities needed
//! to compile a device-resident CNF into a device-resident XGCF circuit.
//!
//! Primary spec: docs/design/2026-01-22-gpu-native-compilation-design.md (Section 5.2.4).

use std::sync::Arc;

use cudarc::driver::{DeviceSlice, LaunchConfig};
use xlog_core::{Result, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::provider::{d4_kernels, scan_kernels, D4_MODULE, SCAN_MODULE};
use xlog_cuda::{CudaKernelProvider, LaunchAsync};
use xlog_solve::GpuCnf;

use crate::gpu::GpuXgcf;

pub(crate) mod build;
pub(crate) mod frontier;

// Re-export used by test code in build.rs (via super::super::).
#[cfg(test)]
pub(crate) use frontier::build_frontier_bitset;

pub(super) fn alloc_compile_gate(
    provider: &CudaKernelProvider,
    value: u32,
) -> Result<TrackedCudaSlice<u32>> {
    let memory = provider.memory();
    let mut gate = memory.alloc::<u32>(1)?;
    provider
        .htod_launch_metadata_sync_copy_into(&[value], &mut gate)
        .map_err(|e| XlogError::Kernel(format!("compile gate upload failed: {}", e)))?;
    Ok(gate)
}

pub(super) fn memset_u8_sync(dst: &mut TrackedCudaSlice<u8>, value: u8) -> Result<()> {
    dst.stream()
        .context()
        .bind_to_thread()
        .map_err(|e| XlogError::Kernel(format!("bind_to_thread failed: {}", e)))?;
    let dptr = dst.device_ptr_value();
    unsafe { cudarc::driver::result::memset_d8_sync(dptr, value, dst.len()) }
        .map_err(|e| XlogError::Kernel(format!("memset_d8_sync failed: {}", e)))?;
    Ok(())
}

/// Configuration for GPU-native Decision-DNNF knowledge compilation plus GPU CDCL verification.
///
/// This is the public control-plane contract for the GPU-native compilation pipeline.
/// Use [`GpuCompileConfig::default()`] for conservative static defaults, or
/// [`crate::exact::default_compile_config`] for dynamic sizing from a CNF.
#[derive(Debug, Clone, Copy)]
pub struct GpuCompileConfig {
    /// BFS expansion depth before handing each frontier item to a per-block DFS worker.
    pub frontier_depth: u16,
    /// Hard cap on the number of frontier work items (overflow is a hard error).
    pub max_frontier_items: u32,
    /// Absolute depth cap (defensive); exceeding this is a hard error (no UNKNOWN).
    pub max_depth: u16,
    /// Hard cap on nodes emitted by the GPU smoothing pass.
    pub smooth_node_cap: u32,
    /// Hard cap on edges emitted by the GPU smoothing pass.
    pub smooth_edge_cap: u32,

    /// CDCL restart cadence (deterministic).
    pub cdcl_restart_interval: u32,
    /// Learned clause arena size (bytes) for the verifier instance.
    pub cdcl_learned_bytes: u64,
    /// Optional conflict budget for debug/profiling only; production must be unbounded.
    pub cdcl_conflict_budget: Option<u64>,

    /// Enable workspace reuse in the equivalence verifier (amortizes arena allocation).
    pub incremental_verify: bool,
}

impl Default for GpuCompileConfig {
    /// Conservative defaults suitable for small-to-medium CNFs.
    ///
    /// Production callers should use [`crate::exact::default_compile_config`] which
    /// sizes arenas dynamically from the CNF and memory budget.
    fn default() -> Self {
        Self {
            frontier_depth: 6,
            max_frontier_items: 64,
            max_depth: 128,
            smooth_node_cap: 65_536,
            smooth_edge_cap: 131_072,
            cdcl_restart_interval: 64,
            cdcl_learned_bytes: 4 * 1024 * 1024,
            cdcl_conflict_budget: None,
            incremental_verify: false,
        }
    }
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

    // `d4_validate_cnf` uses a device-side trap for fail-fast invariants. If it trips, we must
    // surface that error here; otherwise it will show up later as a panicking `CudaSlice` drop.
    provider
        .device()
        .synchronize()
        .map_err(|e| XlogError::Kernel(format!("sync after d4_validate_cnf failed: {}", e)))?;

    Ok(())
}

/// Compute free-variable mask on device (vars absent from CNF and circuit).
///
/// Returns a device-resident u8 mask of length var_cap+1 where mask[v]=1 means
/// var v is free. Traps on device if a variable appears in CNF clauses but is
/// missing from the circuit.
pub fn compute_free_var_mask_gpu(
    cnf: &GpuCnf,
    circuit: &GpuXgcf,
    provider: &CudaKernelProvider,
) -> Result<TrackedCudaSlice<u8>> {
    let compile_needed = alloc_compile_gate(provider, 1)?;
    compute_free_var_mask_gpu_gated(cnf, circuit, provider, &compile_needed)
}

pub(crate) fn compute_free_var_mask_gpu_gated(
    cnf: &GpuCnf,
    circuit: &GpuXgcf,
    provider: &CudaKernelProvider,
    compile_needed: &TrackedCudaSlice<u32>,
) -> Result<TrackedCudaSlice<u8>> {
    if cnf.var_cap == 0 {
        return Err(XlogError::Compilation(
            "compute_free_var_mask_gpu requires var_cap > 0".to_string(),
        ));
    }
    if circuit.max_var() > cnf.var_cap {
        return Err(XlogError::Compilation(format!(
            "compute_free_var_mask_gpu: circuit max_var {} exceeds CNF var_cap {}",
            circuit.max_var(),
            cnf.var_cap
        )));
    }

    let num_nodes = u32::try_from(circuit.num_nodes()).map_err(|_| {
        XlogError::Compilation("compute_free_var_mask_gpu: num_nodes overflow".to_string())
    })?;

    let mask_len = (cnf.var_cap as u64)
        .checked_add(1)
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| {
            XlogError::Compilation("compute_free_var_mask_gpu: mask length overflow".to_string())
        })?;

    let memory = provider.memory();
    let device = provider.device().inner();

    let mut vars_in_clauses = memory.alloc::<u32>(mask_len)?;
    let mut vars_in_circuit = memory.alloc::<u32>(mask_len)?;
    let mut free_var_mask = memory.alloc::<u8>(mask_len)?;

    device.memset_zeros(&mut vars_in_clauses).map_err(|e| {
        XlogError::Kernel(format!(
            "compute_free_var_mask_gpu: zero vars_in_clauses: {}",
            e
        ))
    })?;
    device.memset_zeros(&mut vars_in_circuit).map_err(|e| {
        XlogError::Kernel(format!(
            "compute_free_var_mask_gpu: zero vars_in_circuit: {}",
            e
        ))
    })?;
    device.memset_zeros(&mut free_var_mask).map_err(|e| {
        XlogError::Kernel(format!(
            "compute_free_var_mask_gpu: zero free_var_mask: {}",
            e
        ))
    })?;

    let block_dim = 256u32;

    if cnf.lit_cap > 0 {
        let grid_dim = cnf.lit_cap.div_ceil(block_dim);
        let mark_clauses = device
            .get_func(D4_MODULE, d4_kernels::D4_MARK_VARS_IN_CLAUSES)
            .ok_or_else(|| {
                XlogError::Kernel("d4_mark_vars_in_clauses kernel not found".to_string())
            })?;
        unsafe {
            mark_clauses.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_dim, 1, 1),
                    block_dim: (block_dim, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    compile_needed,
                    cnf.var_cap,
                    cnf.lit_cap,
                    &cnf.num_vars,
                    &cnf.num_lits,
                    &cnf.literals,
                    &mut vars_in_clauses,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("d4_mark_vars_in_clauses failed: {}", e)))?;
    }

    if num_nodes > 0 {
        let grid_dim = num_nodes.div_ceil(block_dim);
        let mark_circuit = device
            .get_func(D4_MODULE, d4_kernels::D4_MARK_VARS_IN_CIRCUIT)
            .ok_or_else(|| {
                XlogError::Kernel("d4_mark_vars_in_circuit kernel not found".to_string())
            })?;
        unsafe {
            mark_circuit.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_dim, 1, 1),
                    block_dim: (block_dim, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    compile_needed,
                    circuit.node_type(),
                    circuit.lit(),
                    circuit.decision_var(),
                    circuit.num_nodes_device(),
                    &cnf.num_vars,
                    cnf.var_cap,
                    &mut vars_in_circuit,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("d4_mark_vars_in_circuit failed: {}", e)))?;
    }

    let mask_len_u32 = cnf.var_cap.checked_add(1).ok_or_else(|| {
        XlogError::Compilation("compute_free_var_mask_gpu: mask length overflow".to_string())
    })?;
    let grid_dim = mask_len_u32.div_ceil(block_dim);
    let build_mask = device
        .get_func(D4_MODULE, d4_kernels::D4_BUILD_FREE_VAR_MASK)
        .ok_or_else(|| XlogError::Kernel("d4_build_free_var_mask kernel not found".to_string()))?;
    unsafe {
        build_mask.clone().launch(
            LaunchConfig {
                grid_dim: (grid_dim, 1, 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                compile_needed,
                &cnf.num_vars,
                cnf.var_cap,
                &vars_in_clauses,
                &vars_in_circuit,
                &mut free_var_mask,
            ),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("d4_build_free_var_mask failed: {}", e)))?;

    Ok(free_var_mask)
}

pub(super) fn bitset_words_per_item(var_cap: u32) -> Result<u32> {
    // Bit 0 is unused (DIMACS vars are 1-based), so we allocate var_cap+1 bits.
    let bits = var_cap
        .checked_add(1)
        .ok_or_else(|| XlogError::Kernel("bitset var_cap+1 overflow".to_string()))?;
    Ok(bits.div_ceil(32))
}

pub(super) fn checked_pool_len_u32(max_items: u32, stride: u32, context: &str) -> Result<u32> {
    let len = (max_items as u64)
        .checked_mul(stride as u64)
        .ok_or_else(|| XlogError::Kernel(format!("{} pool length overflow", context)))?;
    if len > (u32::MAX as u64) {
        return Err(XlogError::Kernel(format!(
            "{} pool length {} exceeds u32::MAX",
            context, len
        )));
    }
    Ok(len as u32)
}

pub(super) fn checked_pool_len_usize(max_items: u32, stride: u32, context: &str) -> Result<usize> {
    let len_u32 = checked_pool_len_u32(max_items, stride, context)?;
    Ok(len_u32 as usize)
}

pub(crate) fn exclusive_scan_u32_inplace(
    provider: &CudaKernelProvider,
    data: &mut TrackedCudaSlice<u32>,
    n: u32,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }

    let device = provider.device().inner();
    let block_size = 256u32;

    if n <= block_size {
        let phase2 = device
            .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_PHASE2)
            .ok_or_else(|| {
                XlogError::Kernel("multiblock_scan_phase2 kernel not found".to_string())
            })?;
        unsafe {
            phase2.clone().launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut *data, n),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("multiblock_scan_phase2 failed: {}", e)))?;
        return Ok(());
    }

    let num_blocks = n.div_ceil(block_size);
    let memory = provider.memory();
    let mut block_sums = memory.alloc::<u32>(num_blocks as usize)?;

    let phase1 = device
        .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_U32_PHASE1)
        .ok_or_else(|| {
            XlogError::Kernel("multiblock_scan_u32_phase1 kernel not found".to_string())
        })?;
    unsafe {
        phase1.clone().launch(
            LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            },
            (&mut *data, &mut block_sums, n),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("multiblock_scan_u32_phase1 failed: {}", e)))?;

    if num_blocks > 1 {
        exclusive_scan_u32_inplace(provider, &mut block_sums, num_blocks)?;
    }

    let phase3 = device
        .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_PHASE3)
        .ok_or_else(|| XlogError::Kernel("multiblock_scan_phase3 kernel not found".to_string()))?;
    unsafe {
        phase3.clone().launch(
            LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            },
            (&mut *data, &block_sums, n),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("multiblock_scan_phase3 failed: {}", e)))?;

    Ok(())
}

/// Compile a device-resident CNF into a device-resident XGCF circuit.
pub(crate) fn compile_gpu_d4(
    cnf: &GpuCnf,
    provider: &Arc<CudaKernelProvider>,
    config: &GpuCompileConfig,
) -> Result<GpuXgcf> {
    let compile_needed = alloc_compile_gate(provider, 1)?;
    Ok(build::compile_gpu_d4_with_gate(cnf, provider, config, &compile_needed)?.0)
}

/// Compile a device-resident CNF into a device-resident XGCF circuit,
/// skipping work on the device when `compile_needed` is 0.
pub fn compile_gpu_d4_gated(
    cnf: &GpuCnf,
    provider: &Arc<CudaKernelProvider>,
    config: &GpuCompileConfig,
    compile_needed: &TrackedCudaSlice<u32>,
) -> Result<GpuXgcf> {
    Ok(build::compile_gpu_d4_with_gate(cnf, provider, config, compile_needed)?.0)
}

/// Gated compile that also returns the BFS frontier item count for profiling
/// (0 unless `XLOG_WARMUP_PROFILE=1`).
pub(crate) fn compile_gpu_d4_gated_with_stats(
    cnf: &GpuCnf,
    provider: &Arc<CudaKernelProvider>,
    config: &GpuCompileConfig,
    compile_needed: &TrackedCudaSlice<u32>,
) -> Result<(GpuXgcf, u32)> {
    build::compile_gpu_d4_with_gate(cnf, provider, config, compile_needed)
}

#[cfg(test)]
mod tests {
    #[test]
    fn gpu_d4_compile_config_requires_smoothing_caps() {
        let config = super::GpuCompileConfig {
            frontier_depth: 0,
            max_frontier_items: 1,
            max_depth: 8,
            cdcl_restart_interval: 128,
            cdcl_learned_bytes: 1 << 20,
            cdcl_conflict_budget: None,
            smooth_node_cap: 256,
            smooth_edge_cap: 512,
            incremental_verify: false,
        };
        assert!(config.smooth_node_cap > 0);
        assert!(config.smooth_edge_cap > 0);
    }
}
