//! GPU-native D4 compilation (Phase 1).
//!
//! This module provides the configuration and kernel-facing utilities needed
//! to compile a device-resident CNF into a device-resident XGCF circuit.
//!
//! Primary spec: docs/design/2026-01-22-gpu-native-compilation-design.md (Section 5.2.4).

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{DevicePtr, DeviceRepr, DeviceSlice, LaunchAsync, LaunchConfig};
use xlog_core::{Result, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::provider::{d4_kernels, scan_kernels, D4_MODULE, SCAN_MODULE};
use xlog_cuda::CudaKernelProvider;
use xlog_solve::GpuCnf;

use crate::gpu::{GpuCircuitBuilder, GpuCircuitLayout, GpuXgcf};

pub(crate) mod build;
pub(crate) mod frontier;


pub use frontier::{
    build_frontier_bitset, build_frontier_dense, GpuFrontierBitset, GpuFrontierDense,
};

pub(super) fn alloc_compile_gate(provider: &CudaKernelProvider, value: u32) -> Result<TrackedCudaSlice<u32>> {
    let memory = provider.memory();
    let mut gate = memory.alloc::<u32>(1)?;
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&[value], &mut gate)
        .map_err(|e| XlogError::Kernel(format!("compile gate upload failed: {}", e)))?;
    Ok(gate)
}

fn memset_u8_sync(
    device: &Arc<cudarc::driver::CudaDevice>,
    dst: &mut TrackedCudaSlice<u8>,
    value: u8,
) -> Result<()> {
    device
        .bind_to_thread()
        .map_err(|e| XlogError::Kernel(format!("bind_to_thread failed: {}", e)))?;
    let dptr = *DevicePtr::device_ptr(&*dst);
    unsafe { cudarc::driver::result::memset_d8_sync(dptr, value, dst.len()) }
        .map_err(|e| XlogError::Kernel(format!("memset_d8_sync failed: {}", e)))?;
    Ok(())
}

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

pub fn compute_free_var_mask_gpu_gated(
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
        let grid_dim = (cnf.lit_cap + block_dim - 1) / block_dim;
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
        let grid_dim = (num_nodes + block_dim - 1) / block_dim;
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
    let grid_dim = (mask_len_u32 + block_dim - 1) / block_dim;
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
    Ok((bits + 31) / 32)
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

    let num_blocks = (n + block_size - 1) / block_size;
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

fn alloc_component_scratch(
    provider: &CudaKernelProvider,
    max_items: u32,
    var_cap: u32,
) -> Result<(
    TrackedCudaSlice<u32>,
    TrackedCudaSlice<u32>,
    TrackedCudaSlice<u32>,
    u32,
)> {
    let uf_stride = var_cap
        .checked_add(1)
        .ok_or_else(|| XlogError::Kernel("component scratch var_cap+1 overflow".to_string()))?;
    let uf_len = checked_pool_len_usize(max_items, uf_stride, "component scratch")?;
    let memory = provider.memory();
    let device = provider.device().inner();

    let mut uf_parent = memory.alloc::<u32>(uf_len)?;
    let mut uf_aux = memory.alloc::<u32>(uf_len)?;
    let mut comp_list = memory.alloc::<u32>(uf_len)?;
    device.memset_zeros(&mut uf_parent).map_err(|e| {
        XlogError::Kernel(format!("Failed to zero component scratch uf_parent: {}", e))
    })?;
    device.memset_zeros(&mut uf_aux).map_err(|e| {
        XlogError::Kernel(format!("Failed to zero component scratch uf_aux: {}", e))
    })?;
    device.memset_zeros(&mut comp_list).map_err(|e| {
        XlogError::Kernel(format!("Failed to zero component scratch comp_list: {}", e))
    })?;
    Ok((uf_parent, uf_aux, comp_list, uf_stride))
}

/// Compile a device-resident CNF into a device-resident XGCF circuit (Phase 1).
pub fn compile_gpu_d4(
    cnf: &GpuCnf,
    provider: &Arc<CudaKernelProvider>,
    config: &GpuCompileConfig,
) -> Result<GpuXgcf> {
    let compile_needed = alloc_compile_gate(provider, 1)?;
    compile_gpu_d4_with_gate(cnf, provider, config, &compile_needed)
}

/// Compile a device-resident CNF into a device-resident XGCF circuit (Phase 1),
/// skipping work on the device when `compile_needed` is 0.
pub fn compile_gpu_d4_gated(
    cnf: &GpuCnf,
    provider: &Arc<CudaKernelProvider>,
    config: &GpuCompileConfig,
    compile_needed: &TrackedCudaSlice<u32>,
) -> Result<GpuXgcf> {
    compile_gpu_d4_with_gate(cnf, provider, config, compile_needed)
}

fn compile_gpu_d4_with_gate(
    cnf: &GpuCnf,
    provider: &Arc<CudaKernelProvider>,
    config: &GpuCompileConfig,
    compile_needed: &TrackedCudaSlice<u32>,
) -> Result<GpuXgcf> {
    if config.max_frontier_items == 0 {
        return Err(XlogError::Compilation(
            "GpuCompileConfig.max_frontier_items must be > 0".to_string(),
        ));
    }
    if config.frontier_depth > config.max_depth {
        return Err(XlogError::Compilation(
            "GpuCompileConfig.frontier_depth must be <= max_depth".to_string(),
        ));
    }
    if config.smooth_node_cap == 0 || config.smooth_edge_cap == 0 {
        return Err(XlogError::Compilation(
            "GpuCompileConfig smooth_node_cap/smooth_edge_cap must be > 0".to_string(),
        ));
    }
    if cnf.var_cap == 0 {
        return Err(XlogError::Compilation(
            "compile_gpu_d4 requires var_cap > 0".to_string(),
        ));
    }

    #[cfg(debug_assertions)]
    eprintln!("[xlog-prob] gpu_d4: validate_cnf_gpu");
    validate_cnf_gpu(cnf, provider)?;

    #[cfg(debug_assertions)]
    eprintln!("[xlog-prob] gpu_d4: build_frontier_bitset");
    let frontier = build_frontier_bitset(cnf, provider, config, compile_needed)?;
    // No device synchronize after build_frontier: words_per_item is a host-side
    // struct field and all subsequent device ops use same-stream ordering.

    let max_items = config.max_frontier_items;
    let words_per_item = frontier.words_per_item();
    let node_cap = config.smooth_node_cap;
    let edge_cap = config.smooth_edge_cap;

    if node_cap < 3 {
        return Err(XlogError::Compilation(
            "GpuCompileConfig.smooth_node_cap must be >= 3".to_string(),
        ));
    }
    if edge_cap < max_items {
        return Err(XlogError::Compilation(format!(
            "GpuCompileConfig.smooth_edge_cap {} < max_frontier_items {}",
            edge_cap, max_items
        )));
    }

    let trail_stride = cnf
        .var_cap
        .checked_add(1)
        .ok_or_else(|| XlogError::Kernel("trail stride var_cap+1 overflow".to_string()))?;
    let trail_len = checked_pool_len_usize(max_items, trail_stride, "trail")?;

    let memory = provider.memory();
    let device = provider.device().inner();

    let mut scratch_trail = memory.alloc::<i32>(trail_len)?;
    device
        .memset_zeros(&mut scratch_trail)
        .map_err(|e| XlogError::Kernel(format!("Failed to zero scratch_trail: {}", e)))?;
    let (mut uf_parent, mut uf_aux, mut comp_list, uf_stride) =
        alloc_component_scratch(provider, max_items, cnf.var_cap)?;

    let mut node_counts = memory.alloc::<u32>(max_items as usize)?;
    let mut edge_counts = memory.alloc::<u32>(max_items as usize)?;
    device
        .memset_zeros(&mut node_counts)
        .map_err(|e| XlogError::Kernel(format!("Failed to zero node_counts: {}", e)))?;
    device
        .memset_zeros(&mut edge_counts)
        .map_err(|e| XlogError::Kernel(format!("Failed to zero edge_counts: {}", e)))?;

    let compile_count = device
        .get_func(D4_MODULE, d4_kernels::D4_COMPILE_COUNT)
        .ok_or_else(|| XlogError::Kernel("d4_compile_count kernel not found".to_string()))?;

    let max_depth = config.max_depth as u32;
    let max_items_u32 = max_items;
    let words_per_item_u32 = words_per_item;
    let trail_stride_u32 = trail_stride;
    let uf_stride_u32 = uf_stride;

    let mut count_params: Vec<*mut c_void> = vec![
        compile_needed.as_kernel_param(),
        max_depth.as_kernel_param(),
        (&cnf.clause_offsets).as_kernel_param(),
        (&cnf.literals).as_kernel_param(),
        (&cnf.num_vars).as_kernel_param(),
        (&cnf.num_clauses).as_kernel_param(),
        (&frontier.items).as_kernel_param(),
        frontier.size_device().as_kernel_param(),
        frontier.true_bits_device().as_kernel_param(),
        frontier.false_bits_device().as_kernel_param(),
        words_per_item_u32.as_kernel_param(),
        (&mut scratch_trail).as_kernel_param(),
        trail_stride_u32.as_kernel_param(),
        (&mut uf_parent).as_kernel_param(),
        (&mut uf_aux).as_kernel_param(),
        (&mut comp_list).as_kernel_param(),
        uf_stride_u32.as_kernel_param(),
        (&mut node_counts).as_kernel_param(),
        (&mut edge_counts).as_kernel_param(),
    ];

    #[cfg(debug_assertions)]
    eprintln!("[xlog-prob] gpu_d4: launch d4_compile_count");
    unsafe {
        compile_count.clone().launch(
            LaunchConfig {
                grid_dim: (max_items_u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            },
            &mut count_params,
        )
    }
    .map_err(|e| XlogError::Kernel(format!("d4_compile_count failed: {}", e)))?;
    // No device synchronize after d4_compile_count: next ops are alloc + dtod_copy
    // (both device-ordered) with no host reads of device data.

    let mut node_offsets = memory.alloc::<u32>(max_items as usize)?;
    let mut edge_offsets = memory.alloc::<u32>(max_items as usize)?;
    device
        .dtod_copy(&node_counts, &mut node_offsets)
        .map_err(|e| XlogError::Kernel(format!("dtod_copy(node_counts) failed: {}", e)))?;
    device
        .dtod_copy(&edge_counts, &mut edge_offsets)
        .map_err(|e| XlogError::Kernel(format!("dtod_copy(edge_counts) failed: {}", e)))?;
    #[cfg(debug_assertions)]
    eprintln!("[xlog-prob] gpu_d4: scan node_offsets/edge_offsets");
    exclusive_scan_u32_inplace(provider, &mut node_offsets, max_items_u32)?;
    exclusive_scan_u32_inplace(provider, &mut edge_offsets, max_items_u32)?;
    // No device synchronize after scans: next ops are kernel launches on same stream.

    let node_cap_usize = usize::try_from(node_cap)
        .map_err(|_| XlogError::Compilation("smooth_node_cap exceeds usize::MAX".to_string()))?;
    let edge_cap_usize = usize::try_from(edge_cap)
        .map_err(|_| XlogError::Compilation("smooth_edge_cap exceeds usize::MAX".to_string()))?;
    let offsets_len = node_cap_usize
        .checked_add(1)
        .ok_or_else(|| XlogError::Compilation("child_offsets length overflow".to_string()))?;

    let mut node_type = memory.alloc::<u8>(node_cap_usize)?;
    let mut child_offsets = memory.alloc::<u32>(offsets_len)?;
    let mut child_indices = memory.alloc::<u32>(edge_cap_usize)?;
    let mut lit = memory.alloc::<i32>(node_cap_usize)?;
    let mut decision_var = memory.alloc::<u32>(node_cap_usize)?;
    let mut decision_child_false = memory.alloc::<u32>(node_cap_usize)?;
    let mut decision_child_true = memory.alloc::<u32>(node_cap_usize)?;
    let mut node_level = memory.alloc::<u32>(node_cap_usize)?;

    // Important: unused nodes in the capacity tail must not contribute clauses in the equivalence
    // verifier. We default them to `LIT` so SAT CNF encoding skips them; `d4_compile_emit`
    // overwrites the prefix [0..meta_num_nodes) with real tags.
    const XGCF_LIT: u8 = 2;
    memset_u8_sync(device, &mut node_type, XGCF_LIT)?;
    device
        .memset_zeros(&mut child_offsets)
        .map_err(|e| XlogError::Kernel(format!("Failed to zero child_offsets: {}", e)))?;
    device
        .memset_zeros(&mut child_indices)
        .map_err(|e| XlogError::Kernel(format!("Failed to zero child_indices: {}", e)))?;
    device
        .memset_zeros(&mut lit)
        .map_err(|e| XlogError::Kernel(format!("Failed to zero lit: {}", e)))?;
    device
        .memset_zeros(&mut decision_var)
        .map_err(|e| XlogError::Kernel(format!("Failed to zero decision_var: {}", e)))?;
    device
        .memset_zeros(&mut decision_child_false)
        .map_err(|e| XlogError::Kernel(format!("Failed to zero decision_child_false: {}", e)))?;
    device
        .memset_zeros(&mut decision_child_true)
        .map_err(|e| XlogError::Kernel(format!("Failed to zero decision_child_true: {}", e)))?;
    device
        .memset_zeros(&mut node_level)
        .map_err(|e| XlogError::Kernel(format!("Failed to zero node_level: {}", e)))?;

    let compile_emit = device
        .get_func(D4_MODULE, d4_kernels::D4_COMPILE_EMIT)
        .ok_or_else(|| XlogError::Kernel("d4_compile_emit kernel not found".to_string()))?;
    let capture_meta = device
        .get_func(D4_MODULE, d4_kernels::D4_CAPTURE_EMIT_META)
        .ok_or_else(|| XlogError::Kernel("d4_capture_emit_meta kernel not found".to_string()))?;

    let max_items_u32 = max_items;
    let node_cap_u32 = node_cap;
    let edge_cap_u32 = edge_cap;
    let mut meta_num_nodes = memory.alloc::<u32>(1)?;
    let mut meta_num_edges = memory.alloc::<u32>(1)?;
    let mut meta_frontier = memory.alloc::<u32>(1)?;
    device
        .memset_zeros(&mut meta_num_nodes)
        .map_err(|e| XlogError::Kernel(format!("Failed to zero meta_num_nodes: {}", e)))?;
    device
        .memset_zeros(&mut meta_num_edges)
        .map_err(|e| XlogError::Kernel(format!("Failed to zero meta_num_edges: {}", e)))?;
    device
        .memset_zeros(&mut meta_frontier)
        .map_err(|e| XlogError::Kernel(format!("Failed to zero meta_frontier: {}", e)))?;
    let mut emit_params: Vec<*mut c_void> = vec![
        compile_needed.as_kernel_param(),
        max_depth.as_kernel_param(),
        max_items_u32.as_kernel_param(),
        node_cap_u32.as_kernel_param(),
        edge_cap_u32.as_kernel_param(),
        (&cnf.clause_offsets).as_kernel_param(),
        (&cnf.literals).as_kernel_param(),
        (&cnf.num_vars).as_kernel_param(),
        (&cnf.num_clauses).as_kernel_param(),
        (&frontier.items).as_kernel_param(),
        frontier.size_device().as_kernel_param(),
        frontier.true_bits_device().as_kernel_param(),
        frontier.false_bits_device().as_kernel_param(),
        words_per_item_u32.as_kernel_param(),
        (&mut scratch_trail).as_kernel_param(),
        trail_stride_u32.as_kernel_param(),
        (&mut uf_parent).as_kernel_param(),
        (&mut uf_aux).as_kernel_param(),
        (&mut comp_list).as_kernel_param(),
        uf_stride_u32.as_kernel_param(),
        (&node_counts).as_kernel_param(),
        (&edge_counts).as_kernel_param(),
        (&node_offsets).as_kernel_param(),
        (&edge_offsets).as_kernel_param(),
        (&mut node_type).as_kernel_param(),
        (&mut child_offsets).as_kernel_param(),
        (&mut child_indices).as_kernel_param(),
        (&mut lit).as_kernel_param(),
        (&mut decision_var).as_kernel_param(),
        (&mut decision_child_false).as_kernel_param(),
        (&mut decision_child_true).as_kernel_param(),
        (&mut node_level).as_kernel_param(),
    ];
    #[cfg(debug_assertions)]
    eprintln!("[xlog-prob] gpu_d4: launch d4_compile_emit");
    unsafe {
        compile_emit.clone().launch(
            LaunchConfig {
                grid_dim: (max_items_u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            },
            &mut emit_params,
        )
    }
    .map_err(|e| XlogError::Kernel(format!("d4_compile_emit failed: {}", e)))?;
    // No device synchronize after d4_compile_emit: next op is capture_meta launch
    // on same stream.
    #[cfg(debug_assertions)]
    eprintln!("[xlog-prob] gpu_d4: launch d4_capture_emit_meta");
    unsafe {
        capture_meta.clone().launch(
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                compile_needed,
                max_items_u32,
                &node_offsets,
                &node_counts,
                &edge_offsets,
                &edge_counts,
                frontier.size_device(),
                node_cap_u32,
                edge_cap_u32,
                &mut meta_num_nodes,
                &mut meta_num_edges,
                &mut meta_frontier,
            ),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("d4_capture_emit_meta failed: {}", e)))?;
    // No device synchronize after d4_capture_emit_meta: next ops are host-only
    // arithmetic and memory allocations (no device data reads).

    let num_levels = (max_depth as u32)
        .checked_mul(2)
        .and_then(|v| v.checked_add(8))
        .ok_or_else(|| XlogError::Compilation("num_levels overflow".to_string()))?;

    let mut level_counts = memory.alloc::<u32>(num_levels as usize)?;
    let mut level_offsets = memory.alloc::<u32>((num_levels as usize) + 1)?;
    let mut level_nodes = memory.alloc::<u32>(node_cap_usize)?;
    let mut level_cursors = memory.alloc::<u32>(num_levels as usize)?;
    device
        .memset_zeros(&mut level_counts)
        .map_err(|e| XlogError::Kernel(format!("Failed to zero level_counts: {}", e)))?;
    device
        .memset_zeros(&mut level_offsets)
        .map_err(|e| XlogError::Kernel(format!("Failed to zero level_offsets: {}", e)))?;
    device
        .memset_zeros(&mut level_nodes)
        .map_err(|e| XlogError::Kernel(format!("Failed to zero level_nodes: {}", e)))?;
    device
        .memset_zeros(&mut level_cursors)
        .map_err(|e| XlogError::Kernel(format!("Failed to zero level_cursors: {}", e)))?;

    let lvl_counts = device
        .get_func(D4_MODULE, d4_kernels::D4_LEVELIZE_COUNTS)
        .ok_or_else(|| XlogError::Kernel("d4_levelize_counts kernel not found".to_string()))?;
    let mut grid = (node_cap_u32 + 255) / 256;
    if grid == 0 {
        grid = 1;
    }
    #[cfg(debug_assertions)]
    eprintln!("[xlog-prob] gpu_d4: launch d4_levelize_counts");
    unsafe {
        lvl_counts.clone().launch(
            LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                compile_needed,
                &node_level,
                &meta_num_nodes,
                num_levels,
                &mut level_counts,
            ),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("d4_levelize_counts failed: {}", e)))?;

    device
        .dtod_copy(
            &level_counts,
            &mut level_offsets.slice_mut(0..num_levels as usize),
        )
        .map_err(|e| XlogError::Kernel(format!("dtod_copy(level_counts) failed: {}", e)))?;
    exclusive_scan_u32_inplace(provider, &mut level_offsets, num_levels + 1)?;

    let lvl_emit = device
        .get_func(D4_MODULE, d4_kernels::D4_LEVELIZE_EMIT)
        .ok_or_else(|| XlogError::Kernel("d4_levelize_emit kernel not found".to_string()))?;
    unsafe {
        lvl_emit.clone().launch(
            LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                compile_needed,
                &node_level,
                &meta_num_nodes,
                num_levels,
                &level_offsets,
                &mut level_cursors,
                &mut level_nodes,
            ),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("d4_levelize_emit failed: {}", e)))?;
    // No device synchronize after d4_levelize_emit: builder wraps device slices
    // consumed by subsequent GPU ops on same stream.

    let builder = GpuCircuitBuilder {
        node_type,
        child_offsets,
        child_indices,
        lit,
        decision_var,
        decision_child_false,
        decision_child_true,
    };
    let layout = GpuCircuitLayout {
        num_nodes: node_cap_u32,
        num_edges: edge_cap_u32,
        num_levels,
        level_offsets,
        level_nodes,
        root: 2,
        max_var: cnf.var_cap,
        num_nodes_device: Some(meta_num_nodes),
        num_edges_device: Some(meta_num_edges),
    };

    GpuXgcf::from_device(builder, layout, provider)
}


#[cfg(test)]
mod tests {
    use std::ffi::c_void;
    use std::sync::Arc;

    use cudarc::driver::{DeviceRepr, LaunchAsync, LaunchConfig};
    use xlog_core::MemoryBudget;
    use xlog_cuda::provider::{d4_kernels, D4_MODULE};
    use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
    use xlog_solve::{Clause, GpuCnf, Literal, SolveInstance};

    fn try_provider() -> Option<Arc<CudaKernelProvider>> {
        let device = match CudaDevice::new(0) {
            Ok(d) => Arc::new(d),
            Err(e) => {
                eprintln!("Skipping test: CUDA runtime unavailable: {}", e);
                return None;
            }
        };
        let budget = MemoryBudget::with_limit(1024 * 1024 * 1024); // 1 GiB
        let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
        match CudaKernelProvider::new(device, memory) {
            Ok(p) => Some(Arc::new(p)),
            Err(e) => {
                eprintln!(
                    "Skipping test: failed to create CUDA kernel provider: {}",
                    e
                );
                None
            }
        }
    }

    fn alloc_component_scratch(
        device: &Arc<cudarc::driver::CudaDevice>,
        memory: &Arc<GpuMemoryManager>,
        max_items: u32,
        var_cap: u32,
    ) -> (
        xlog_cuda::memory::TrackedCudaSlice<u32>,
        xlog_cuda::memory::TrackedCudaSlice<u32>,
        xlog_cuda::memory::TrackedCudaSlice<u32>,
        u32,
    ) {
        let uf_stride = var_cap
            .checked_add(1)
            .expect("var_cap+1 overflow for uf_stride");
        let uf_len = (max_items as usize) * (uf_stride as usize);
        let mut uf_parent = memory.alloc::<u32>(uf_len).unwrap();
        let mut uf_aux = memory.alloc::<u32>(uf_len).unwrap();
        let mut comp_list = memory.alloc::<u32>(uf_len).unwrap();
        device.memset_zeros(&mut uf_parent).unwrap();
        device.memset_zeros(&mut uf_aux).unwrap();
        device.memset_zeros(&mut comp_list).unwrap();
        (uf_parent, uf_aux, comp_list, uf_stride)
    }

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


    #[test]
    fn gpu_d4_compile_phase1_unit_clause_is_equivalent_via_gpu_cdcl() {
        let Some(provider) = try_provider() else {
            return;
        };
        let device = provider.device().inner();
        let memory = provider.memory();

        // φ = (x1)
        let instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
        let phi = GpuCnf::from_host(&instance, &provider).expect("GpuCnf upload");

        let cfg = super::GpuCompileConfig {
            frontier_depth: 0,
            max_frontier_items: 8,
            max_depth: 32,
            smooth_node_cap: 1024,
            smooth_edge_cap: 4096,
            cdcl_restart_interval: 32,
            cdcl_learned_bytes: 4 * 1024 * 1024,
            cdcl_conflict_budget: None,
            incremental_verify: false,
        };

        super::validate_cnf_gpu(&phi, &provider).expect("CNF validation must succeed");

        // One frontier item (root) with empty assignment; DFS must handle unit propagation.
        let compile_needed = super::alloc_compile_gate(&provider, 1).expect("compile gate");
        let frontier = super::build_frontier_bitset(&phi, &provider, &cfg, &compile_needed)
            .expect("build_frontier_bitset must succeed");
        provider
            .device()
            .synchronize()
            .expect("sync after build_frontier_bitset must succeed");
        let max_items = cfg.max_frontier_items;
        let max_depth_u32 = cfg.max_depth as u32;
        let wpi = frontier.words_per_item;

        // Circuit pool sizing for this test (small CNFs only).
        let node_cap: u32 = 256;
        let edge_cap: u32 = 512;
        let num_levels: u32 = max_depth_u32 * 2u32 + 8u32;

        let mut node_type = memory.alloc::<u8>(node_cap as usize).unwrap();
        let mut child_offsets = memory.alloc::<u32>((node_cap as usize) + 1).unwrap();
        let mut child_indices = memory.alloc::<u32>(edge_cap as usize).unwrap();
        let mut lit = memory.alloc::<i32>(node_cap as usize).unwrap();
        let mut decision_var = memory.alloc::<u32>(node_cap as usize).unwrap();
        let mut decision_child_false = memory.alloc::<u32>(node_cap as usize).unwrap();
        let mut decision_child_true = memory.alloc::<u32>(node_cap as usize).unwrap();
        let mut node_level = memory.alloc::<u32>(node_cap as usize).unwrap();

        device.memset_zeros(&mut node_type).unwrap();
        device.memset_zeros(&mut child_offsets).unwrap();
        device.memset_zeros(&mut child_indices).unwrap();
        device.memset_zeros(&mut lit).unwrap();
        device.memset_zeros(&mut decision_var).unwrap();
        device.memset_zeros(&mut decision_child_false).unwrap();
        device.memset_zeros(&mut decision_child_true).unwrap();
        device.memset_zeros(&mut node_level).unwrap();

        let mut node_counts = memory.alloc::<u32>(max_items as usize).unwrap();
        let mut edge_counts = memory.alloc::<u32>(max_items as usize).unwrap();
        device.memset_zeros(&mut node_counts).unwrap();
        device.memset_zeros(&mut edge_counts).unwrap();

        let trail_stride: u32 = phi
            .var_cap
            .checked_add(1)
            .expect("var_cap+1 overflow in test");
        let trail_len = (max_items as usize) * (trail_stride as usize);
        let mut scratch_trail = memory.alloc::<i32>(trail_len).unwrap();
        device.memset_zeros(&mut scratch_trail).unwrap();
        let (mut uf_parent, mut uf_aux, mut comp_list, uf_stride) =
            alloc_component_scratch(device, &memory, max_items, phi.var_cap);

        let compile_count = device
            .get_func(D4_MODULE, "d4_compile_count")
            .expect("d4_compile_count must exist");
        let mut count_params: Vec<*mut c_void> = vec![
            (&compile_needed).as_kernel_param(),
            max_depth_u32.as_kernel_param(),
            (&phi.clause_offsets).as_kernel_param(),
            (&phi.literals).as_kernel_param(),
            (&phi.num_vars).as_kernel_param(),
            (&phi.num_clauses).as_kernel_param(),
            (&frontier.items).as_kernel_param(),
            frontier.size_device().as_kernel_param(),
            frontier.true_bits_device().as_kernel_param(),
            frontier.false_bits_device().as_kernel_param(),
            wpi.as_kernel_param(),
            (&mut scratch_trail).as_kernel_param(),
            trail_stride.as_kernel_param(),
            (&mut uf_parent).as_kernel_param(),
            (&mut uf_aux).as_kernel_param(),
            (&mut comp_list).as_kernel_param(),
            uf_stride.as_kernel_param(),
            (&mut node_counts).as_kernel_param(),
            (&mut edge_counts).as_kernel_param(),
        ];
        unsafe {
            compile_count
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (max_items, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut count_params,
                )
                .unwrap();
        }
        provider.device().synchronize().unwrap();

        // Compute exclusive scans for per-item node/edge regions (no host reads).
        let mut node_offsets = memory.alloc::<u32>(max_items as usize).unwrap();
        let mut edge_offsets = memory.alloc::<u32>(max_items as usize).unwrap();
        device.dtod_copy(&node_counts, &mut node_offsets).unwrap();
        device.dtod_copy(&edge_counts, &mut edge_offsets).unwrap();
        super::exclusive_scan_u32_inplace(&provider, &mut node_offsets, max_items)
            .expect("node scan must succeed");
        super::exclusive_scan_u32_inplace(&provider, &mut edge_offsets, max_items)
            .expect("edge scan must succeed");

        let compile_emit = device
            .get_func(D4_MODULE, "d4_compile_emit")
            .expect("d4_compile_emit must exist");
        let mut emit_params: Vec<*mut c_void> = vec![
            (&compile_needed).as_kernel_param(),
            max_depth_u32.as_kernel_param(),
            max_items.as_kernel_param(),
            node_cap.as_kernel_param(),
            edge_cap.as_kernel_param(),
            (&phi.clause_offsets).as_kernel_param(),
            (&phi.literals).as_kernel_param(),
            (&phi.num_vars).as_kernel_param(),
            (&phi.num_clauses).as_kernel_param(),
            (&frontier.items).as_kernel_param(),
            frontier.size_device().as_kernel_param(),
            frontier.true_bits_device().as_kernel_param(),
            frontier.false_bits_device().as_kernel_param(),
            wpi.as_kernel_param(),
            (&mut scratch_trail).as_kernel_param(),
            trail_stride.as_kernel_param(),
            (&mut uf_parent).as_kernel_param(),
            (&mut uf_aux).as_kernel_param(),
            (&mut comp_list).as_kernel_param(),
            uf_stride.as_kernel_param(),
            (&node_counts).as_kernel_param(),
            (&edge_counts).as_kernel_param(),
            (&node_offsets).as_kernel_param(),
            (&edge_offsets).as_kernel_param(),
            (&mut node_type).as_kernel_param(),
            (&mut child_offsets).as_kernel_param(),
            (&mut child_indices).as_kernel_param(),
            (&mut lit).as_kernel_param(),
            (&mut decision_var).as_kernel_param(),
            (&mut decision_child_false).as_kernel_param(),
            (&mut decision_child_true).as_kernel_param(),
            (&mut node_level).as_kernel_param(),
        ];
        unsafe {
            compile_emit
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (max_items, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut emit_params,
                )
                .unwrap();
        }

        // Levelize the circuit using a fixed `num_levels` bound derived from `max_depth`.
        let mut level_counts = memory.alloc::<u32>(num_levels as usize).unwrap();
        let mut level_offsets = memory.alloc::<u32>((num_levels as usize) + 1).unwrap();
        let mut level_nodes = memory.alloc::<u32>(node_cap as usize).unwrap();
        let mut level_cursors = memory.alloc::<u32>(num_levels as usize).unwrap();
        device.memset_zeros(&mut level_counts).unwrap();
        device.memset_zeros(&mut level_offsets).unwrap();
        device.memset_zeros(&mut level_nodes).unwrap();
        device.memset_zeros(&mut level_cursors).unwrap();
        let mut num_nodes_device = memory.alloc::<u32>(1).unwrap();
        device
            .htod_sync_copy_into(&[node_cap], &mut num_nodes_device)
            .unwrap();

        let lvl_counts = device
            .get_func(D4_MODULE, d4_kernels::D4_LEVELIZE_COUNTS)
            .expect("d4_levelize_counts must exist");
        unsafe {
            lvl_counts
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: ((node_cap + 255) / 256, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &compile_needed,
                        &node_level,
                        &num_nodes_device,
                        num_levels,
                        &mut level_counts,
                    ),
                )
                .unwrap();
        }
        device
            .dtod_copy(
                &level_counts,
                &mut level_offsets.slice_mut(0..num_levels as usize),
            )
            .unwrap();
        super::exclusive_scan_u32_inplace(&provider, &mut level_offsets, num_levels + 1)
            .expect("level scan must succeed");

        let lvl_emit = device
            .get_func(D4_MODULE, d4_kernels::D4_LEVELIZE_EMIT)
            .expect("d4_levelize_emit must exist");
        unsafe {
            lvl_emit
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: ((node_cap + 255) / 256, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &compile_needed,
                        &node_level,
                        &num_nodes_device,
                        num_levels,
                        &level_offsets,
                        &mut level_cursors,
                        &mut level_nodes,
                    ),
                )
                .unwrap();
        }
        provider
            .device()
            .synchronize()
            .expect("sync after d4_levelize_emit must succeed");

        let builder = crate::gpu::GpuCircuitBuilder {
            node_type,
            child_offsets,
            child_indices,
            lit,
            decision_var,
            decision_child_false,
            decision_child_true,
        };
        let layout = crate::gpu::GpuCircuitLayout {
            num_nodes: node_cap,
            num_edges: edge_cap,
            num_levels,
            level_offsets,
            level_nodes,
            root: 2, // reserved root OR (see d4_compile_emit contract)
            max_var: phi.var_cap,
            num_nodes_device: None,
            num_edges_device: None,
        };

        let circuit = crate::gpu::GpuXgcf::from_device(builder, layout, &provider)
            .expect("GpuXgcf::from_device must succeed");

        crate::compilation::validate_equivalence_gpu(
            &phi,
            &phi.num_vars,
            &circuit,
            &provider,
            crate::compilation::GpuEquivalenceConfig::default(),
        )
        .expect("equivalence must hold");
    }

    #[test]
    fn gpu_d4_compile_phase1_binary_clause_branches_and_is_equivalent_via_gpu_cdcl() {
        let Some(provider) = try_provider() else {
            return;
        };
        let device = provider.device().inner();
        let memory = provider.memory();

        // φ = (x1 ∨ x2)
        let instance = SolveInstance::new(
            2,
            vec![Clause::new(vec![
                Literal::positive(0),
                Literal::positive(1),
            ])],
        );
        let phi = GpuCnf::from_host(&instance, &provider).expect("GpuCnf upload");

        let cfg = super::GpuCompileConfig {
            frontier_depth: 0,
            max_frontier_items: 8,
            max_depth: 32,
            smooth_node_cap: 1024,
            smooth_edge_cap: 4096,
            cdcl_restart_interval: 32,
            cdcl_learned_bytes: 4 * 1024 * 1024,
            cdcl_conflict_budget: None,
            incremental_verify: false,
        };

        super::validate_cnf_gpu(&phi, &provider).expect("CNF validation must succeed");
        let compile_needed = super::alloc_compile_gate(&provider, 1).expect("compile gate");
        let frontier = super::build_frontier_bitset(&phi, &provider, &cfg, &compile_needed)
            .expect("build_frontier_bitset must succeed");
        let max_items = cfg.max_frontier_items;
        let max_depth_u32 = cfg.max_depth as u32;
        let wpi = frontier.words_per_item;

        let node_cap: u32 = 512;
        let edge_cap: u32 = 2048;
        let num_levels: u32 = max_depth_u32 * 2u32 + 8u32;

        let mut node_type = memory.alloc::<u8>(node_cap as usize).unwrap();
        let mut child_offsets = memory.alloc::<u32>((node_cap as usize) + 1).unwrap();
        let mut child_indices = memory.alloc::<u32>(edge_cap as usize).unwrap();
        let mut lit = memory.alloc::<i32>(node_cap as usize).unwrap();
        let mut decision_var = memory.alloc::<u32>(node_cap as usize).unwrap();
        let mut decision_child_false = memory.alloc::<u32>(node_cap as usize).unwrap();
        let mut decision_child_true = memory.alloc::<u32>(node_cap as usize).unwrap();
        let mut node_level = memory.alloc::<u32>(node_cap as usize).unwrap();

        device.memset_zeros(&mut node_type).unwrap();
        device.memset_zeros(&mut child_offsets).unwrap();
        device.memset_zeros(&mut child_indices).unwrap();
        device.memset_zeros(&mut lit).unwrap();
        device.memset_zeros(&mut decision_var).unwrap();
        device.memset_zeros(&mut decision_child_false).unwrap();
        device.memset_zeros(&mut decision_child_true).unwrap();
        device.memset_zeros(&mut node_level).unwrap();

        let mut node_counts = memory.alloc::<u32>(max_items as usize).unwrap();
        let mut edge_counts = memory.alloc::<u32>(max_items as usize).unwrap();
        device.memset_zeros(&mut node_counts).unwrap();
        device.memset_zeros(&mut edge_counts).unwrap();

        let trail_stride: u32 = phi
            .var_cap
            .checked_add(1)
            .expect("var_cap+1 overflow in test");
        let trail_len = (max_items as usize) * (trail_stride as usize);
        let mut scratch_trail = memory.alloc::<i32>(trail_len).unwrap();
        device.memset_zeros(&mut scratch_trail).unwrap();
        let (mut uf_parent, mut uf_aux, mut comp_list, uf_stride) =
            alloc_component_scratch(device, &memory, max_items, phi.var_cap);

        let compile_count = device
            .get_func(D4_MODULE, d4_kernels::D4_COMPILE_COUNT)
            .expect("d4_compile_count must exist");
        let mut count_params: Vec<*mut c_void> = vec![
            (&compile_needed).as_kernel_param(),
            max_depth_u32.as_kernel_param(),
            (&phi.clause_offsets).as_kernel_param(),
            (&phi.literals).as_kernel_param(),
            (&phi.num_vars).as_kernel_param(),
            (&phi.num_clauses).as_kernel_param(),
            (&frontier.items).as_kernel_param(),
            frontier.size_device().as_kernel_param(),
            frontier.true_bits_device().as_kernel_param(),
            frontier.false_bits_device().as_kernel_param(),
            wpi.as_kernel_param(),
            (&mut scratch_trail).as_kernel_param(),
            trail_stride.as_kernel_param(),
            (&mut uf_parent).as_kernel_param(),
            (&mut uf_aux).as_kernel_param(),
            (&mut comp_list).as_kernel_param(),
            uf_stride.as_kernel_param(),
            (&mut node_counts).as_kernel_param(),
            (&mut edge_counts).as_kernel_param(),
        ];
        unsafe {
            compile_count
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (max_items, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut count_params,
                )
                .unwrap();
        }
        provider
            .device()
            .synchronize()
            .expect("sync after d4_compile_count must succeed");

        let mut node_offsets = memory.alloc::<u32>(max_items as usize).unwrap();
        let mut edge_offsets = memory.alloc::<u32>(max_items as usize).unwrap();
        device.dtod_copy(&node_counts, &mut node_offsets).unwrap();
        device.dtod_copy(&edge_counts, &mut edge_offsets).unwrap();
        super::exclusive_scan_u32_inplace(&provider, &mut node_offsets, max_items).unwrap();
        super::exclusive_scan_u32_inplace(&provider, &mut edge_offsets, max_items).unwrap();
        provider
            .device()
            .synchronize()
            .expect("sync after scans must succeed");
        provider.device().synchronize().unwrap();

        let compile_emit = device
            .get_func(D4_MODULE, d4_kernels::D4_COMPILE_EMIT)
            .expect("d4_compile_emit must exist");
        let mut emit_params: Vec<*mut c_void> = vec![
            (&compile_needed).as_kernel_param(),
            max_depth_u32.as_kernel_param(),
            max_items.as_kernel_param(),
            node_cap.as_kernel_param(),
            edge_cap.as_kernel_param(),
            (&phi.clause_offsets).as_kernel_param(),
            (&phi.literals).as_kernel_param(),
            (&phi.num_vars).as_kernel_param(),
            (&phi.num_clauses).as_kernel_param(),
            (&frontier.items).as_kernel_param(),
            frontier.size_device().as_kernel_param(),
            frontier.true_bits_device().as_kernel_param(),
            frontier.false_bits_device().as_kernel_param(),
            wpi.as_kernel_param(),
            (&mut scratch_trail).as_kernel_param(),
            trail_stride.as_kernel_param(),
            (&mut uf_parent).as_kernel_param(),
            (&mut uf_aux).as_kernel_param(),
            (&mut comp_list).as_kernel_param(),
            uf_stride.as_kernel_param(),
            (&node_counts).as_kernel_param(),
            (&edge_counts).as_kernel_param(),
            (&node_offsets).as_kernel_param(),
            (&edge_offsets).as_kernel_param(),
            (&mut node_type).as_kernel_param(),
            (&mut child_offsets).as_kernel_param(),
            (&mut child_indices).as_kernel_param(),
            (&mut lit).as_kernel_param(),
            (&mut decision_var).as_kernel_param(),
            (&mut decision_child_false).as_kernel_param(),
            (&mut decision_child_true).as_kernel_param(),
            (&mut node_level).as_kernel_param(),
        ];
        unsafe {
            compile_emit
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (max_items, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut emit_params,
                )
                .unwrap();
        }
        provider
            .device()
            .synchronize()
            .expect("sync after d4_compile_emit must succeed");
        provider.device().synchronize().unwrap();

        let mut level_counts = memory.alloc::<u32>(num_levels as usize).unwrap();
        let mut level_offsets = memory.alloc::<u32>((num_levels as usize) + 1).unwrap();
        let mut level_nodes = memory.alloc::<u32>(node_cap as usize).unwrap();
        let mut level_cursors = memory.alloc::<u32>(num_levels as usize).unwrap();
        device.memset_zeros(&mut level_counts).unwrap();
        device.memset_zeros(&mut level_offsets).unwrap();
        device.memset_zeros(&mut level_nodes).unwrap();
        device.memset_zeros(&mut level_cursors).unwrap();
        let mut num_nodes_device = memory.alloc::<u32>(1).unwrap();
        device
            .htod_sync_copy_into(&[node_cap], &mut num_nodes_device)
            .unwrap();

        let lvl_counts = device
            .get_func(D4_MODULE, d4_kernels::D4_LEVELIZE_COUNTS)
            .expect("d4_levelize_counts must exist");
        unsafe {
            lvl_counts
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: ((node_cap + 255) / 256, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &compile_needed,
                        &node_level,
                        &num_nodes_device,
                        num_levels,
                        &mut level_counts,
                    ),
                )
                .unwrap();
        }
        provider.device().synchronize().unwrap();
        device
            .dtod_copy(
                &level_counts,
                &mut level_offsets.slice_mut(0..num_levels as usize),
            )
            .unwrap();
        super::exclusive_scan_u32_inplace(&provider, &mut level_offsets, num_levels + 1).unwrap();

        let lvl_emit = device
            .get_func(D4_MODULE, d4_kernels::D4_LEVELIZE_EMIT)
            .expect("d4_levelize_emit must exist");
        unsafe {
            lvl_emit
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: ((node_cap + 255) / 256, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &compile_needed,
                        &node_level,
                        &num_nodes_device,
                        num_levels,
                        &level_offsets,
                        &mut level_cursors,
                        &mut level_nodes,
                    ),
                )
                .unwrap();
        }
        provider.device().synchronize().unwrap();

        let builder = crate::gpu::GpuCircuitBuilder {
            node_type,
            child_offsets,
            child_indices,
            lit,
            decision_var,
            decision_child_false,
            decision_child_true,
        };
        let layout = crate::gpu::GpuCircuitLayout {
            num_nodes: node_cap,
            num_edges: edge_cap,
            num_levels,
            level_offsets,
            level_nodes,
            root: 2,
            max_var: phi.var_cap,
            num_nodes_device: None,
            num_edges_device: None,
        };
        let circuit = crate::gpu::GpuXgcf::from_device(builder, layout, &provider).unwrap();

        crate::compilation::validate_equivalence_gpu(
            &phi,
            &phi.num_vars,
            &circuit,
            &provider,
            crate::compilation::GpuEquivalenceConfig::default(),
        )
        .expect("equivalence must hold");
    }

    #[test]
    fn gpu_d4_compile_phase1_unsat_is_equivalent_via_gpu_cdcl() {
        let Some(provider) = try_provider() else {
            return;
        };
        let device = provider.device().inner();
        let memory = provider.memory();

        // φ = (x1) ∧ (¬x1)
        let instance = SolveInstance::new(
            1,
            vec![
                Clause::new(vec![Literal::positive(0)]),
                Clause::new(vec![Literal::negative(0)]),
            ],
        );
        let phi = GpuCnf::from_host(&instance, &provider).expect("GpuCnf upload");

        let cfg = super::GpuCompileConfig {
            frontier_depth: 0,
            max_frontier_items: 8,
            max_depth: 32,
            smooth_node_cap: 1024,
            smooth_edge_cap: 4096,
            cdcl_restart_interval: 32,
            cdcl_learned_bytes: 4 * 1024 * 1024,
            cdcl_conflict_budget: None,
            incremental_verify: false,
        };

        super::validate_cnf_gpu(&phi, &provider).expect("CNF validation must succeed");
        let compile_needed = super::alloc_compile_gate(&provider, 1).expect("compile gate");
        let frontier = super::build_frontier_bitset(&phi, &provider, &cfg, &compile_needed)
            .expect("build_frontier_bitset must succeed");
        let max_items = cfg.max_frontier_items;
        let max_depth_u32 = cfg.max_depth as u32;
        let wpi = frontier.words_per_item;

        let node_cap: u32 = 128;
        let edge_cap: u32 = 512;
        let num_levels: u32 = max_depth_u32 * 2u32 + 8u32;

        let mut node_type = memory.alloc::<u8>(node_cap as usize).unwrap();
        let mut child_offsets = memory.alloc::<u32>((node_cap as usize) + 1).unwrap();
        let mut child_indices = memory.alloc::<u32>(edge_cap as usize).unwrap();
        let mut lit = memory.alloc::<i32>(node_cap as usize).unwrap();
        let mut decision_var = memory.alloc::<u32>(node_cap as usize).unwrap();
        let mut decision_child_false = memory.alloc::<u32>(node_cap as usize).unwrap();
        let mut decision_child_true = memory.alloc::<u32>(node_cap as usize).unwrap();
        let mut node_level = memory.alloc::<u32>(node_cap as usize).unwrap();

        device.memset_zeros(&mut node_type).unwrap();
        device.memset_zeros(&mut child_offsets).unwrap();
        device.memset_zeros(&mut child_indices).unwrap();
        device.memset_zeros(&mut lit).unwrap();
        device.memset_zeros(&mut decision_var).unwrap();
        device.memset_zeros(&mut decision_child_false).unwrap();
        device.memset_zeros(&mut decision_child_true).unwrap();
        device.memset_zeros(&mut node_level).unwrap();

        let mut node_counts = memory.alloc::<u32>(max_items as usize).unwrap();
        let mut edge_counts = memory.alloc::<u32>(max_items as usize).unwrap();
        device.memset_zeros(&mut node_counts).unwrap();
        device.memset_zeros(&mut edge_counts).unwrap();

        let trail_stride: u32 = phi
            .var_cap
            .checked_add(1)
            .expect("var_cap+1 overflow in test");
        let trail_len = (max_items as usize) * (trail_stride as usize);
        let mut scratch_trail = memory.alloc::<i32>(trail_len).unwrap();
        device.memset_zeros(&mut scratch_trail).unwrap();
        let (mut uf_parent, mut uf_aux, mut comp_list, uf_stride) =
            alloc_component_scratch(device, &memory, max_items, phi.var_cap);

        let compile_count = device
            .get_func(D4_MODULE, d4_kernels::D4_COMPILE_COUNT)
            .expect("d4_compile_count must exist");
        let mut count_params: Vec<*mut c_void> = vec![
            (&compile_needed).as_kernel_param(),
            max_depth_u32.as_kernel_param(),
            (&phi.clause_offsets).as_kernel_param(),
            (&phi.literals).as_kernel_param(),
            (&phi.num_vars).as_kernel_param(),
            (&phi.num_clauses).as_kernel_param(),
            (&frontier.items).as_kernel_param(),
            frontier.size_device().as_kernel_param(),
            frontier.true_bits_device().as_kernel_param(),
            frontier.false_bits_device().as_kernel_param(),
            wpi.as_kernel_param(),
            (&mut scratch_trail).as_kernel_param(),
            trail_stride.as_kernel_param(),
            (&mut uf_parent).as_kernel_param(),
            (&mut uf_aux).as_kernel_param(),
            (&mut comp_list).as_kernel_param(),
            uf_stride.as_kernel_param(),
            (&mut node_counts).as_kernel_param(),
            (&mut edge_counts).as_kernel_param(),
        ];
        unsafe {
            compile_count
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (max_items, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut count_params,
                )
                .unwrap();
        }

        let mut node_offsets = memory.alloc::<u32>(max_items as usize).unwrap();
        let mut edge_offsets = memory.alloc::<u32>(max_items as usize).unwrap();
        device.dtod_copy(&node_counts, &mut node_offsets).unwrap();
        device.dtod_copy(&edge_counts, &mut edge_offsets).unwrap();
        super::exclusive_scan_u32_inplace(&provider, &mut node_offsets, max_items).unwrap();
        super::exclusive_scan_u32_inplace(&provider, &mut edge_offsets, max_items).unwrap();

        let compile_emit = device
            .get_func(D4_MODULE, d4_kernels::D4_COMPILE_EMIT)
            .expect("d4_compile_emit must exist");
        let mut emit_params: Vec<*mut c_void> = vec![
            (&compile_needed).as_kernel_param(),
            max_depth_u32.as_kernel_param(),
            max_items.as_kernel_param(),
            node_cap.as_kernel_param(),
            edge_cap.as_kernel_param(),
            (&phi.clause_offsets).as_kernel_param(),
            (&phi.literals).as_kernel_param(),
            (&phi.num_vars).as_kernel_param(),
            (&phi.num_clauses).as_kernel_param(),
            (&frontier.items).as_kernel_param(),
            frontier.size_device().as_kernel_param(),
            frontier.true_bits_device().as_kernel_param(),
            frontier.false_bits_device().as_kernel_param(),
            wpi.as_kernel_param(),
            (&mut scratch_trail).as_kernel_param(),
            trail_stride.as_kernel_param(),
            (&mut uf_parent).as_kernel_param(),
            (&mut uf_aux).as_kernel_param(),
            (&mut comp_list).as_kernel_param(),
            uf_stride.as_kernel_param(),
            (&node_counts).as_kernel_param(),
            (&edge_counts).as_kernel_param(),
            (&node_offsets).as_kernel_param(),
            (&edge_offsets).as_kernel_param(),
            (&mut node_type).as_kernel_param(),
            (&mut child_offsets).as_kernel_param(),
            (&mut child_indices).as_kernel_param(),
            (&mut lit).as_kernel_param(),
            (&mut decision_var).as_kernel_param(),
            (&mut decision_child_false).as_kernel_param(),
            (&mut decision_child_true).as_kernel_param(),
            (&mut node_level).as_kernel_param(),
        ];
        unsafe {
            compile_emit
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (max_items, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut emit_params,
                )
                .unwrap();
        }
        let mut level_counts = memory.alloc::<u32>(num_levels as usize).unwrap();
        let mut level_offsets = memory.alloc::<u32>((num_levels as usize) + 1).unwrap();
        let mut level_nodes = memory.alloc::<u32>(node_cap as usize).unwrap();
        let mut level_cursors = memory.alloc::<u32>(num_levels as usize).unwrap();
        device.memset_zeros(&mut level_counts).unwrap();
        device.memset_zeros(&mut level_offsets).unwrap();
        device.memset_zeros(&mut level_nodes).unwrap();
        device.memset_zeros(&mut level_cursors).unwrap();
        let mut num_nodes_device = memory.alloc::<u32>(1).unwrap();
        device
            .htod_sync_copy_into(&[node_cap], &mut num_nodes_device)
            .unwrap();

        let lvl_counts = device
            .get_func(D4_MODULE, d4_kernels::D4_LEVELIZE_COUNTS)
            .expect("d4_levelize_counts must exist");
        unsafe {
            lvl_counts
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: ((node_cap + 255) / 256, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &compile_needed,
                        &node_level,
                        &num_nodes_device,
                        num_levels,
                        &mut level_counts,
                    ),
                )
                .unwrap();
        }
        provider
            .device()
            .synchronize()
            .expect("sync after d4_levelize_counts must succeed");
        device
            .dtod_copy(
                &level_counts,
                &mut level_offsets.slice_mut(0..num_levels as usize),
            )
            .unwrap();
        super::exclusive_scan_u32_inplace(&provider, &mut level_offsets, num_levels + 1).unwrap();
        provider
            .device()
            .synchronize()
            .expect("sync after level scan must succeed");

        let lvl_emit = device
            .get_func(D4_MODULE, d4_kernels::D4_LEVELIZE_EMIT)
            .expect("d4_levelize_emit must exist");
        unsafe {
            lvl_emit
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: ((node_cap + 255) / 256, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &compile_needed,
                        &node_level,
                        &num_nodes_device,
                        num_levels,
                        &level_offsets,
                        &mut level_cursors,
                        &mut level_nodes,
                    ),
                )
                .unwrap();
        }
        provider
            .device()
            .synchronize()
            .expect("sync after d4_levelize_emit must succeed");

        let builder = crate::gpu::GpuCircuitBuilder {
            node_type,
            child_offsets,
            child_indices,
            lit,
            decision_var,
            decision_child_false,
            decision_child_true,
        };
        let layout = crate::gpu::GpuCircuitLayout {
            num_nodes: node_cap,
            num_edges: edge_cap,
            num_levels,
            level_offsets,
            level_nodes,
            root: 2,
            max_var: phi.var_cap,
            num_nodes_device: None,
            num_edges_device: None,
        };
        let circuit = crate::gpu::GpuXgcf::from_device(builder, layout, &provider).unwrap();

        crate::compilation::validate_equivalence_gpu(
            &phi,
            &phi.num_vars,
            &circuit,
            &provider,
            crate::compilation::GpuEquivalenceConfig::default(),
        )
        .expect("equivalence must hold");
    }

    #[test]
    fn gpu_d4_compile_phase1_frontier_depth2_multi_leaf_is_equivalent_via_gpu_cdcl() {
        let Some(provider) = try_provider() else {
            return;
        };
        let device = provider.device().inner();
        let memory = provider.memory();

        // φ = (x1 ∨ x2) ∧ (x3 ∨ x4)
        let instance = SolveInstance::new(
            4,
            vec![
                Clause::new(vec![Literal::positive(0), Literal::positive(1)]),
                Clause::new(vec![Literal::positive(2), Literal::positive(3)]),
            ],
        );
        let phi = GpuCnf::from_host(&instance, &provider).expect("GpuCnf upload");

        let cfg = super::GpuCompileConfig {
            frontier_depth: 2,
            max_frontier_items: 8,
            max_depth: 32,
            smooth_node_cap: 1024,
            smooth_edge_cap: 4096,
            cdcl_restart_interval: 32,
            cdcl_learned_bytes: 4 * 1024 * 1024,
            cdcl_conflict_budget: None,
            incremental_verify: false,
        };

        super::validate_cnf_gpu(&phi, &provider).expect("CNF validation must succeed");
        let compile_needed = super::alloc_compile_gate(&provider, 1).expect("compile gate");
        let frontier = super::build_frontier_bitset(&phi, &provider, &cfg, &compile_needed)
            .expect("build_frontier_bitset must succeed");
        let max_items = cfg.max_frontier_items;
        let max_depth_u32 = cfg.max_depth as u32;
        let wpi = frontier.words_per_item;

        let node_cap: u32 = 4096;
        let edge_cap: u32 = 16384;
        let num_levels: u32 = max_depth_u32 * 2u32 + 8u32;

        let mut node_type = memory.alloc::<u8>(node_cap as usize).unwrap();
        let mut child_offsets = memory.alloc::<u32>((node_cap as usize) + 1).unwrap();
        let mut child_indices = memory.alloc::<u32>(edge_cap as usize).unwrap();
        let mut lit = memory.alloc::<i32>(node_cap as usize).unwrap();
        let mut decision_var = memory.alloc::<u32>(node_cap as usize).unwrap();
        let mut decision_child_false = memory.alloc::<u32>(node_cap as usize).unwrap();
        let mut decision_child_true = memory.alloc::<u32>(node_cap as usize).unwrap();
        let mut node_level = memory.alloc::<u32>(node_cap as usize).unwrap();

        device.memset_zeros(&mut node_type).unwrap();
        device.memset_zeros(&mut child_offsets).unwrap();
        device.memset_zeros(&mut child_indices).unwrap();
        device.memset_zeros(&mut lit).unwrap();
        device.memset_zeros(&mut decision_var).unwrap();
        device.memset_zeros(&mut decision_child_false).unwrap();
        device.memset_zeros(&mut decision_child_true).unwrap();
        device.memset_zeros(&mut node_level).unwrap();

        let mut node_counts = memory.alloc::<u32>(max_items as usize).unwrap();
        let mut edge_counts = memory.alloc::<u32>(max_items as usize).unwrap();
        device.memset_zeros(&mut node_counts).unwrap();
        device.memset_zeros(&mut edge_counts).unwrap();

        let trail_stride: u32 = phi
            .var_cap
            .checked_add(1)
            .expect("var_cap+1 overflow in test");
        let trail_len = (max_items as usize) * (trail_stride as usize);
        let mut scratch_trail = memory.alloc::<i32>(trail_len).unwrap();
        device.memset_zeros(&mut scratch_trail).unwrap();
        let (mut uf_parent, mut uf_aux, mut comp_list, uf_stride) =
            alloc_component_scratch(device, &memory, max_items, phi.var_cap);

        let compile_count = device
            .get_func(D4_MODULE, d4_kernels::D4_COMPILE_COUNT)
            .expect("d4_compile_count must exist");
        let mut count_params: Vec<*mut c_void> = vec![
            (&compile_needed).as_kernel_param(),
            max_depth_u32.as_kernel_param(),
            (&phi.clause_offsets).as_kernel_param(),
            (&phi.literals).as_kernel_param(),
            (&phi.num_vars).as_kernel_param(),
            (&phi.num_clauses).as_kernel_param(),
            (&frontier.items).as_kernel_param(),
            frontier.size_device().as_kernel_param(),
            frontier.true_bits_device().as_kernel_param(),
            frontier.false_bits_device().as_kernel_param(),
            wpi.as_kernel_param(),
            (&mut scratch_trail).as_kernel_param(),
            trail_stride.as_kernel_param(),
            (&mut uf_parent).as_kernel_param(),
            (&mut uf_aux).as_kernel_param(),
            (&mut comp_list).as_kernel_param(),
            uf_stride.as_kernel_param(),
            (&mut node_counts).as_kernel_param(),
            (&mut edge_counts).as_kernel_param(),
        ];
        unsafe {
            compile_count
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (max_items, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut count_params,
                )
                .unwrap();
        }

        let mut node_offsets = memory.alloc::<u32>(max_items as usize).unwrap();
        let mut edge_offsets = memory.alloc::<u32>(max_items as usize).unwrap();
        device.dtod_copy(&node_counts, &mut node_offsets).unwrap();
        device.dtod_copy(&edge_counts, &mut edge_offsets).unwrap();
        super::exclusive_scan_u32_inplace(&provider, &mut node_offsets, max_items).unwrap();
        super::exclusive_scan_u32_inplace(&provider, &mut edge_offsets, max_items).unwrap();

        let compile_emit = device
            .get_func(D4_MODULE, d4_kernels::D4_COMPILE_EMIT)
            .expect("d4_compile_emit must exist");
        let mut emit_params: Vec<*mut c_void> = vec![
            (&compile_needed).as_kernel_param(),
            max_depth_u32.as_kernel_param(),
            max_items.as_kernel_param(),
            node_cap.as_kernel_param(),
            edge_cap.as_kernel_param(),
            (&phi.clause_offsets).as_kernel_param(),
            (&phi.literals).as_kernel_param(),
            (&phi.num_vars).as_kernel_param(),
            (&phi.num_clauses).as_kernel_param(),
            (&frontier.items).as_kernel_param(),
            frontier.size_device().as_kernel_param(),
            frontier.true_bits_device().as_kernel_param(),
            frontier.false_bits_device().as_kernel_param(),
            wpi.as_kernel_param(),
            (&mut scratch_trail).as_kernel_param(),
            trail_stride.as_kernel_param(),
            (&mut uf_parent).as_kernel_param(),
            (&mut uf_aux).as_kernel_param(),
            (&mut comp_list).as_kernel_param(),
            uf_stride.as_kernel_param(),
            (&node_counts).as_kernel_param(),
            (&edge_counts).as_kernel_param(),
            (&node_offsets).as_kernel_param(),
            (&edge_offsets).as_kernel_param(),
            (&mut node_type).as_kernel_param(),
            (&mut child_offsets).as_kernel_param(),
            (&mut child_indices).as_kernel_param(),
            (&mut lit).as_kernel_param(),
            (&mut decision_var).as_kernel_param(),
            (&mut decision_child_false).as_kernel_param(),
            (&mut decision_child_true).as_kernel_param(),
            (&mut node_level).as_kernel_param(),
        ];
        unsafe {
            compile_emit
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (max_items, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut emit_params,
                )
                .unwrap();
        }

        let mut level_counts = memory.alloc::<u32>(num_levels as usize).unwrap();
        let mut level_offsets = memory.alloc::<u32>((num_levels as usize) + 1).unwrap();
        let mut level_nodes = memory.alloc::<u32>(node_cap as usize).unwrap();
        let mut level_cursors = memory.alloc::<u32>(num_levels as usize).unwrap();
        device.memset_zeros(&mut level_counts).unwrap();
        device.memset_zeros(&mut level_offsets).unwrap();
        device.memset_zeros(&mut level_nodes).unwrap();
        device.memset_zeros(&mut level_cursors).unwrap();
        let mut num_nodes_device = memory.alloc::<u32>(1).unwrap();
        device
            .htod_sync_copy_into(&[node_cap], &mut num_nodes_device)
            .unwrap();

        let lvl_counts = device
            .get_func(D4_MODULE, d4_kernels::D4_LEVELIZE_COUNTS)
            .expect("d4_levelize_counts must exist");
        unsafe {
            lvl_counts
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: ((node_cap + 255) / 256, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &compile_needed,
                        &node_level,
                        &num_nodes_device,
                        num_levels,
                        &mut level_counts,
                    ),
                )
                .unwrap();
        }
        device
            .dtod_copy(
                &level_counts,
                &mut level_offsets.slice_mut(0..num_levels as usize),
            )
            .unwrap();
        super::exclusive_scan_u32_inplace(&provider, &mut level_offsets, num_levels + 1).unwrap();

        let lvl_emit = device
            .get_func(D4_MODULE, d4_kernels::D4_LEVELIZE_EMIT)
            .expect("d4_levelize_emit must exist");
        unsafe {
            lvl_emit
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: ((node_cap + 255) / 256, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &compile_needed,
                        &node_level,
                        &num_nodes_device,
                        num_levels,
                        &level_offsets,
                        &mut level_cursors,
                        &mut level_nodes,
                    ),
                )
                .unwrap();
        }

        let builder = crate::gpu::GpuCircuitBuilder {
            node_type,
            child_offsets,
            child_indices,
            lit,
            decision_var,
            decision_child_false,
            decision_child_true,
        };
        let layout = crate::gpu::GpuCircuitLayout {
            num_nodes: node_cap,
            num_edges: edge_cap,
            num_levels,
            level_offsets,
            level_nodes,
            root: 2,
            max_var: phi.var_cap,
            num_nodes_device: None,
            num_edges_device: None,
        };
        let circuit = crate::gpu::GpuXgcf::from_device(builder, layout, &provider).unwrap();
        provider
            .device()
            .synchronize()
            .expect("sync after GpuXgcf::from_device must succeed");

        crate::compilation::validate_equivalence_gpu(
            &phi,
            &phi.num_vars,
            &circuit,
            &provider,
            crate::compilation::GpuEquivalenceConfig::default(),
        )
        .expect("equivalence must hold");
        provider
            .device()
            .synchronize()
            .expect("sync after validate_equivalence_gpu must succeed");
    }

    #[test]
    fn gpu_d4_compile_phase1_component_decomposition_emits_and_root() {
        let Some(provider) = try_provider() else {
            return;
        };
        let device = provider.device().inner();
        let memory = provider.memory();

        // Two independent components: (x1 ∨ x2) ∧ (x3 ∨ x4).
        let instance = SolveInstance::new(
            4,
            vec![
                Clause::new(vec![Literal::positive(0), Literal::positive(1)]),
                Clause::new(vec![Literal::positive(2), Literal::positive(3)]),
            ],
        );
        let phi = GpuCnf::from_host(&instance, &provider).expect("GpuCnf upload");

        let cfg = super::GpuCompileConfig {
            frontier_depth: 0,
            max_frontier_items: 1,
            max_depth: 32,
            smooth_node_cap: 1024,
            smooth_edge_cap: 4096,
            cdcl_restart_interval: 32,
            cdcl_learned_bytes: 4 * 1024 * 1024,
            cdcl_conflict_budget: None,
            incremental_verify: false,
        };

        super::validate_cnf_gpu(&phi, &provider).expect("CNF validation must succeed");
        let compile_needed = super::alloc_compile_gate(&provider, 1).expect("compile gate");
        let frontier = super::build_frontier_bitset(&phi, &provider, &cfg, &compile_needed)
            .expect("build_frontier_bitset must succeed");
        provider
            .device()
            .synchronize()
            .expect("sync after build_frontier_bitset must succeed");
        let max_items = cfg.max_frontier_items;
        let max_depth_u32 = cfg.max_depth as u32;
        let wpi = frontier.words_per_item;

        let node_cap: u32 = 4096;
        let edge_cap: u32 = 16384;

        let mut node_type = memory.alloc::<u8>(node_cap as usize).unwrap();
        let mut child_offsets = memory.alloc::<u32>((node_cap as usize) + 1).unwrap();
        let mut child_indices = memory.alloc::<u32>(edge_cap as usize).unwrap();
        let mut lit = memory.alloc::<i32>(node_cap as usize).unwrap();
        let mut decision_var = memory.alloc::<u32>(node_cap as usize).unwrap();
        let mut decision_child_false = memory.alloc::<u32>(node_cap as usize).unwrap();
        let mut decision_child_true = memory.alloc::<u32>(node_cap as usize).unwrap();
        let mut node_level = memory.alloc::<u32>(node_cap as usize).unwrap();

        device.memset_zeros(&mut node_type).unwrap();
        device.memset_zeros(&mut child_offsets).unwrap();
        device.memset_zeros(&mut child_indices).unwrap();
        device.memset_zeros(&mut lit).unwrap();
        device.memset_zeros(&mut decision_var).unwrap();
        device.memset_zeros(&mut decision_child_false).unwrap();
        device.memset_zeros(&mut decision_child_true).unwrap();
        device.memset_zeros(&mut node_level).unwrap();

        let mut node_counts = memory.alloc::<u32>(max_items as usize).unwrap();
        let mut edge_counts = memory.alloc::<u32>(max_items as usize).unwrap();
        device.memset_zeros(&mut node_counts).unwrap();
        device.memset_zeros(&mut edge_counts).unwrap();

        let trail_stride: u32 = phi
            .var_cap
            .checked_add(1)
            .expect("var_cap+1 overflow in test");
        let trail_len = (max_items as usize) * (trail_stride as usize);
        let mut scratch_trail = memory.alloc::<i32>(trail_len).unwrap();
        device.memset_zeros(&mut scratch_trail).unwrap();
        let (mut uf_parent, mut uf_aux, mut comp_list, uf_stride) =
            alloc_component_scratch(device, &memory, max_items, phi.var_cap);

        let compile_count = device
            .get_func(D4_MODULE, d4_kernels::D4_COMPILE_COUNT)
            .expect("d4_compile_count must exist");
        let mut count_params: Vec<*mut c_void> = vec![
            (&compile_needed).as_kernel_param(),
            max_depth_u32.as_kernel_param(),
            (&phi.clause_offsets).as_kernel_param(),
            (&phi.literals).as_kernel_param(),
            (&phi.num_vars).as_kernel_param(),
            (&phi.num_clauses).as_kernel_param(),
            (&frontier.items).as_kernel_param(),
            frontier.size_device().as_kernel_param(),
            frontier.true_bits_device().as_kernel_param(),
            frontier.false_bits_device().as_kernel_param(),
            wpi.as_kernel_param(),
            (&mut scratch_trail).as_kernel_param(),
            trail_stride.as_kernel_param(),
            (&mut uf_parent).as_kernel_param(),
            (&mut uf_aux).as_kernel_param(),
            (&mut comp_list).as_kernel_param(),
            uf_stride.as_kernel_param(),
            (&mut node_counts).as_kernel_param(),
            (&mut edge_counts).as_kernel_param(),
        ];
        unsafe {
            compile_count
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (max_items, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut count_params,
                )
                .unwrap();
        }
        provider
            .device()
            .synchronize()
            .expect("sync after d4_compile_count must succeed");

        let mut node_offsets = memory.alloc::<u32>(max_items as usize).unwrap();
        let mut edge_offsets = memory.alloc::<u32>(max_items as usize).unwrap();
        device.dtod_copy(&node_counts, &mut node_offsets).unwrap();
        device.dtod_copy(&edge_counts, &mut edge_offsets).unwrap();
        super::exclusive_scan_u32_inplace(&provider, &mut node_offsets, max_items).unwrap();
        super::exclusive_scan_u32_inplace(&provider, &mut edge_offsets, max_items).unwrap();
        provider
            .device()
            .synchronize()
            .expect("sync after scans must succeed");

        let compile_emit = device
            .get_func(D4_MODULE, d4_kernels::D4_COMPILE_EMIT)
            .expect("d4_compile_emit must exist");
        let mut emit_params: Vec<*mut c_void> = vec![
            (&compile_needed).as_kernel_param(),
            max_depth_u32.as_kernel_param(),
            max_items.as_kernel_param(),
            node_cap.as_kernel_param(),
            edge_cap.as_kernel_param(),
            (&phi.clause_offsets).as_kernel_param(),
            (&phi.literals).as_kernel_param(),
            (&phi.num_vars).as_kernel_param(),
            (&phi.num_clauses).as_kernel_param(),
            (&frontier.items).as_kernel_param(),
            frontier.size_device().as_kernel_param(),
            frontier.true_bits_device().as_kernel_param(),
            frontier.false_bits_device().as_kernel_param(),
            wpi.as_kernel_param(),
            (&mut scratch_trail).as_kernel_param(),
            trail_stride.as_kernel_param(),
            (&mut uf_parent).as_kernel_param(),
            (&mut uf_aux).as_kernel_param(),
            (&mut comp_list).as_kernel_param(),
            uf_stride.as_kernel_param(),
            (&node_counts).as_kernel_param(),
            (&edge_counts).as_kernel_param(),
            (&node_offsets).as_kernel_param(),
            (&edge_offsets).as_kernel_param(),
            (&mut node_type).as_kernel_param(),
            (&mut child_offsets).as_kernel_param(),
            (&mut child_indices).as_kernel_param(),
            (&mut lit).as_kernel_param(),
            (&mut decision_var).as_kernel_param(),
            (&mut decision_child_false).as_kernel_param(),
            (&mut decision_child_true).as_kernel_param(),
            (&mut node_level).as_kernel_param(),
        ];
        unsafe {
            compile_emit
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (max_items, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut emit_params,
                )
                .unwrap();
        }
        provider
            .device()
            .synchronize()
            .expect("sync after d4_compile_emit must succeed");

        // Root OR (node 2) should point to a leaf root that is an AND over 2 component circuits.
        let assert_and = device
            .get_func(D4_MODULE, d4_kernels::D4_ASSERT_LEAF_ROOT_AND_DEGREE)
            .expect("d4_assert_leaf_root_and_degree must exist");
        unsafe {
            assert_and
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &node_type,
                        &child_offsets,
                        &child_indices,
                        node_cap,
                        0u32, // leaf index in root OR
                        2u32, // expect AND of 2 components
                    ),
                )
                .unwrap();
        }
        provider
            .device()
            .synchronize()
            .expect("sync after component decomposition assert must succeed");
    }
}
