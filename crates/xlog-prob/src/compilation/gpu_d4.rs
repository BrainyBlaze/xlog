//! GPU-native D4 compilation (Phase 1).
//!
//! This module provides the configuration and kernel-facing utilities needed
//! to compile a device-resident CNF into a device-resident XGCF circuit.
//!
//! Primary spec: docs/design/2026-01-22-gpu-native-compilation-design.md (Section 5.2.4).

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{DeviceRepr, DeviceSlice, LaunchAsync, LaunchConfig};
use xlog_core::{Result, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::provider::{d4_kernels, scan_kernels, D4_MODULE, SCAN_MODULE};
use xlog_cuda::CudaKernelProvider;
use xlog_solve::GpuCnf;

use crate::gpu::{GpuCircuitBuilder, GpuCircuitLayout, GpuXgcf};

fn alloc_compile_gate(provider: &CudaKernelProvider, value: u32) -> Result<TrackedCudaSlice<u32>> {
    let memory = provider.memory();
    let mut gate = memory.alloc::<u32>(1)?;
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&[value], &mut gate)
        .map_err(|e| XlogError::Kernel(format!("compile gate upload failed: {}", e)))?;
    Ok(gate)
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
                    num_nodes,
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

// ---------------------------------------------------------------------------
// Frontier work queues (Phase 1) - device-resident BFS expansion
// ---------------------------------------------------------------------------

/// Must match `struct D4WorkItem` in `kernels/d4.cu`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct D4WorkItem {
    subproblem_id: u32,
    parent_node: u32,
    branch: u8,
    depth: u16,
    assignment_offset: u32,
}

unsafe impl DeviceRepr for D4WorkItem {}

fn bitset_words_per_item(var_cap: u32) -> Result<u32> {
    // Bit 0 is unused (DIMACS vars are 1-based), so we allocate var_cap+1 bits.
    let bits = var_cap
        .checked_add(1)
        .ok_or_else(|| XlogError::Kernel("bitset var_cap+1 overflow".to_string()))?;
    Ok((bits + 31) / 32)
}

fn checked_pool_len_u32(max_items: u32, stride: u32, context: &str) -> Result<u32> {
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

fn checked_pool_len_usize(max_items: u32, stride: u32, context: &str) -> Result<usize> {
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

/// Device-resident frontier state using compressed bitset assignments.
pub struct GpuFrontierBitset {
    items: TrackedCudaSlice<D4WorkItem>,
    size: TrackedCudaSlice<u32>,
    true_bits: TrackedCudaSlice<u32>,
    false_bits: TrackedCudaSlice<u32>,
    words_per_item: u32,
    max_frontier_items: u32,
}

impl GpuFrontierBitset {
    pub fn size_device(&self) -> &TrackedCudaSlice<u32> {
        &self.size
    }

    pub fn true_bits_device(&self) -> &TrackedCudaSlice<u32> {
        &self.true_bits
    }

    pub fn false_bits_device(&self) -> &TrackedCudaSlice<u32> {
        &self.false_bits
    }

    pub fn words_per_item(&self) -> u32 {
        self.words_per_item
    }

    pub fn max_frontier_items(&self) -> u32 {
        self.max_frontier_items
    }

    pub fn items_len(&self) -> usize {
        self.items.len()
    }
}

/// Device-resident frontier state using dense tri-state assignments.
pub struct GpuFrontierDense {
    items: TrackedCudaSlice<D4WorkItem>,
    size: TrackedCudaSlice<u32>,
    assignments: TrackedCudaSlice<u8>,
    stride_bytes: u32,
    max_frontier_items: u32,
}

impl GpuFrontierDense {
    pub fn size_device(&self) -> &TrackedCudaSlice<u32> {
        &self.size
    }

    pub fn assignments_device(&self) -> &TrackedCudaSlice<u8> {
        &self.assignments
    }

    pub fn stride_bytes(&self) -> u32 {
        self.stride_bytes
    }

    pub fn max_frontier_items(&self) -> u32 {
        self.max_frontier_items
    }

    pub fn items_len(&self) -> usize {
        self.items.len()
    }
}

/// Build a BFS frontier using compressed bitsets (default).
///
/// This is an internal building block for Phase 1 GPU D4 compilation.
pub fn build_frontier_bitset(
    cnf: &GpuCnf,
    provider: &CudaKernelProvider,
    config: &GpuCompileConfig,
    compile_needed: &TrackedCudaSlice<u32>,
) -> Result<GpuFrontierBitset> {
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

    let max_frontier_items = config.max_frontier_items;
    let words_per_item = bitset_words_per_item(cnf.var_cap)?;
    let pool_len = checked_pool_len_usize(max_frontier_items, words_per_item, "bitset")?;

    let memory = provider.memory();
    let device = provider.device().inner();

    let mut cur_items = memory.alloc::<D4WorkItem>(max_frontier_items as usize)?;
    let mut next_items = memory.alloc::<D4WorkItem>(max_frontier_items as usize)?;
    let mut cur_size = memory.alloc::<u32>(1)?;
    let mut next_size = memory.alloc::<u32>(1)?;

    let mut cur_true = memory.alloc::<u32>(pool_len)?;
    let mut cur_false = memory.alloc::<u32>(pool_len)?;
    let mut next_true = memory.alloc::<u32>(pool_len)?;
    let mut next_false = memory.alloc::<u32>(pool_len)?;

    let mut counts = memory.alloc::<u32>(max_frontier_items as usize)?;
    let mut prefix = memory.alloc::<u32>(max_frontier_items as usize)?;
    let mut pick_var = memory.alloc::<u32>(max_frontier_items as usize)?;

    // Initialize: one root work item at index 0 with an empty assignment.
    device
        .memset_zeros(&mut cur_true)
        .map_err(|e| XlogError::Kernel(format!("Failed to zero bitset true pool: {}", e)))?;
    device
        .memset_zeros(&mut cur_false)
        .map_err(|e| XlogError::Kernel(format!("Failed to zero bitset false pool: {}", e)))?;
    device
        .memset_zeros(&mut next_true)
        .map_err(|e| XlogError::Kernel(format!("Failed to zero bitset next_true pool: {}", e)))?;
    device
        .memset_zeros(&mut next_false)
        .map_err(|e| XlogError::Kernel(format!("Failed to zero bitset next_false pool: {}", e)))?;
    device
        .memset_zeros(&mut counts)
        .map_err(|e| XlogError::Kernel(format!("Failed to zero frontier counts: {}", e)))?;
    device
        .memset_zeros(&mut pick_var)
        .map_err(|e| XlogError::Kernel(format!("Failed to zero frontier pick_var: {}", e)))?;

    // Provide a defined default layout for all slots; only `cur_size` items are considered active.
    let mut host_items: Vec<D4WorkItem> = Vec::with_capacity(max_frontier_items as usize);
    for i in 0..max_frontier_items {
        let off = i
            .checked_mul(words_per_item)
            .ok_or_else(|| XlogError::Kernel("bitset assignment_offset overflow".to_string()))?;
        host_items.push(D4WorkItem {
            subproblem_id: i,
            parent_node: u32::MAX,
            branch: 0,
            depth: 0,
            assignment_offset: off,
        });
    }
    device
        .htod_sync_copy_into(&host_items, &mut cur_items)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload frontier items: {}", e)))?;
    device
        .htod_sync_copy_into(&[1u32], &mut cur_size)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload frontier size: {}", e)))?;
    device
        .memset_zeros(&mut next_size)
        .map_err(|e| XlogError::Kernel(format!("Failed to zero next_size: {}", e)))?;

    let prep = device
        .get_func(D4_MODULE, d4_kernels::D4_FRONTIER_PREPARE)
        .ok_or_else(|| XlogError::Kernel("d4_frontier_prepare kernel not found".to_string()))?;
    let expand = device
        .get_func(D4_MODULE, d4_kernels::D4_FRONTIER_EXPAND)
        .ok_or_else(|| XlogError::Kernel("d4_frontier_expand kernel not found".to_string()))?;

    let frontier_depth_u32 = config.frontier_depth as u32;
    let max_depth_u32 = config.max_depth as u32;
    let words_per_item_u32 = words_per_item;
    let max_frontier_items_u32 = max_frontier_items;

    // Expand exactly `frontier_depth` steps. Terminal items carry forward unchanged.
    for _step in 0..frontier_depth_u32 {
        let mut prep_params: Vec<*mut c_void> = vec![
            compile_needed.as_kernel_param(),
            frontier_depth_u32.as_kernel_param(),
            max_depth_u32.as_kernel_param(),
            (&cnf.clause_offsets).as_kernel_param(),
            (&cnf.literals).as_kernel_param(),
            (&cnf.num_vars).as_kernel_param(),
            (&cnf.num_clauses).as_kernel_param(),
            (&cur_items).as_kernel_param(),
            (&cur_size).as_kernel_param(),
            (&mut cur_true).as_kernel_param(),
            (&mut cur_false).as_kernel_param(),
            words_per_item_u32.as_kernel_param(),
            (&mut counts).as_kernel_param(),
            (&mut pick_var).as_kernel_param(),
        ];
        unsafe {
            prep.clone().launch(
                LaunchConfig {
                    grid_dim: ((max_frontier_items_u32 + 127) / 128, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                },
                &mut prep_params,
            )
        }
        .map_err(|e| XlogError::Kernel(format!("d4_frontier_prepare failed: {}", e)))?;

        // Build exclusive prefix offsets in `prefix` without host reads.
        device
            .dtod_copy(&counts, &mut prefix)
            .map_err(|e| XlogError::Kernel(format!("dtod_copy(counts->prefix) failed: {}", e)))?;
        exclusive_scan_u32_inplace(provider, &mut prefix, max_frontier_items_u32)?;

        let mut expand_params: Vec<*mut c_void> = vec![
            compile_needed.as_kernel_param(),
            (&cur_size).as_kernel_param(),
            (&cur_items).as_kernel_param(),
            (&counts).as_kernel_param(),
            (&prefix).as_kernel_param(),
            (&pick_var).as_kernel_param(),
            (&cur_true).as_kernel_param(),
            (&cur_false).as_kernel_param(),
            words_per_item_u32.as_kernel_param(),
            (&mut next_items).as_kernel_param(),
            (&mut next_size).as_kernel_param(),
            (&mut next_true).as_kernel_param(),
            (&mut next_false).as_kernel_param(),
            max_frontier_items_u32.as_kernel_param(),
        ];
        unsafe {
            expand.clone().launch(
                LaunchConfig {
                    grid_dim: ((max_frontier_items_u32 + 127) / 128, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                },
                &mut expand_params,
            )
        }
        .map_err(|e| XlogError::Kernel(format!("d4_frontier_expand failed: {}", e)))?;

        std::mem::swap(&mut cur_items, &mut next_items);
        std::mem::swap(&mut cur_size, &mut next_size);
        std::mem::swap(&mut cur_true, &mut next_true);
        std::mem::swap(&mut cur_false, &mut next_false);
    }

    Ok(GpuFrontierBitset {
        items: cur_items,
        size: cur_size,
        true_bits: cur_true,
        false_bits: cur_false,
        words_per_item,
        max_frontier_items,
    })
}

/// Build a BFS frontier using dense tri-state assignments.
pub fn build_frontier_dense(
    cnf: &GpuCnf,
    provider: &CudaKernelProvider,
    config: &GpuCompileConfig,
    compile_needed: &TrackedCudaSlice<u32>,
) -> Result<GpuFrontierDense> {
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

    let max_frontier_items = config.max_frontier_items;
    let stride_bytes = cnf
        .var_cap
        .checked_add(1)
        .ok_or_else(|| XlogError::Kernel("dense stride var_cap+1 overflow".to_string()))?;
    let pool_len = checked_pool_len_usize(max_frontier_items, stride_bytes, "dense")?;

    let memory = provider.memory();
    let device = provider.device().inner();

    let mut cur_items = memory.alloc::<D4WorkItem>(max_frontier_items as usize)?;
    let mut next_items = memory.alloc::<D4WorkItem>(max_frontier_items as usize)?;
    let mut cur_size = memory.alloc::<u32>(1)?;
    let mut next_size = memory.alloc::<u32>(1)?;

    let mut cur_assign = memory.alloc::<u8>(pool_len)?;
    let mut next_assign = memory.alloc::<u8>(pool_len)?;

    let mut counts = memory.alloc::<u32>(max_frontier_items as usize)?;
    let mut prefix = memory.alloc::<u32>(max_frontier_items as usize)?;
    let mut pick_var = memory.alloc::<u32>(max_frontier_items as usize)?;

    device
        .memset_zeros(&mut cur_assign)
        .map_err(|e| XlogError::Kernel(format!("Failed to zero dense assignment pool: {}", e)))?;
    device.memset_zeros(&mut next_assign).map_err(|e| {
        XlogError::Kernel(format!("Failed to zero dense next assignment pool: {}", e))
    })?;
    device
        .memset_zeros(&mut counts)
        .map_err(|e| XlogError::Kernel(format!("Failed to zero frontier counts: {}", e)))?;
    device
        .memset_zeros(&mut pick_var)
        .map_err(|e| XlogError::Kernel(format!("Failed to zero frontier pick_var: {}", e)))?;

    let mut host_items: Vec<D4WorkItem> = Vec::with_capacity(max_frontier_items as usize);
    for i in 0..max_frontier_items {
        let off = i
            .checked_mul(stride_bytes)
            .ok_or_else(|| XlogError::Kernel("dense assignment_offset overflow".to_string()))?;
        host_items.push(D4WorkItem {
            subproblem_id: i,
            parent_node: u32::MAX,
            branch: 0,
            depth: 0,
            assignment_offset: off,
        });
    }
    device
        .htod_sync_copy_into(&host_items, &mut cur_items)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload frontier items: {}", e)))?;
    device
        .htod_sync_copy_into(&[1u32], &mut cur_size)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload frontier size: {}", e)))?;
    device
        .memset_zeros(&mut next_size)
        .map_err(|e| XlogError::Kernel(format!("Failed to zero next_size: {}", e)))?;

    let prep = device
        .get_func(D4_MODULE, d4_kernels::D4_FRONTIER_PREPARE_DENSE)
        .ok_or_else(|| {
            XlogError::Kernel("d4_frontier_prepare_dense kernel not found".to_string())
        })?;
    let expand = device
        .get_func(D4_MODULE, d4_kernels::D4_FRONTIER_EXPAND_DENSE)
        .ok_or_else(|| {
            XlogError::Kernel("d4_frontier_expand_dense kernel not found".to_string())
        })?;

    let frontier_depth_u32 = config.frontier_depth as u32;
    let max_depth_u32 = config.max_depth as u32;
    let stride_u32 = stride_bytes;
    let max_frontier_items_u32 = max_frontier_items;

    for _step in 0..frontier_depth_u32 {
        let mut prep_params: Vec<*mut c_void> = vec![
            compile_needed.as_kernel_param(),
            frontier_depth_u32.as_kernel_param(),
            max_depth_u32.as_kernel_param(),
            (&cnf.clause_offsets).as_kernel_param(),
            (&cnf.literals).as_kernel_param(),
            (&cnf.num_vars).as_kernel_param(),
            (&cnf.num_clauses).as_kernel_param(),
            (&cur_items).as_kernel_param(),
            (&cur_size).as_kernel_param(),
            (&mut cur_assign).as_kernel_param(),
            stride_u32.as_kernel_param(),
            (&mut counts).as_kernel_param(),
            (&mut pick_var).as_kernel_param(),
        ];
        unsafe {
            prep.clone().launch(
                LaunchConfig {
                    grid_dim: ((max_frontier_items_u32 + 127) / 128, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                },
                &mut prep_params,
            )
        }
        .map_err(|e| XlogError::Kernel(format!("d4_frontier_prepare_dense failed: {}", e)))?;

        device
            .dtod_copy(&counts, &mut prefix)
            .map_err(|e| XlogError::Kernel(format!("dtod_copy(counts->prefix) failed: {}", e)))?;
        exclusive_scan_u32_inplace(provider, &mut prefix, max_frontier_items_u32)?;

        let mut expand_params: Vec<*mut c_void> = vec![
            compile_needed.as_kernel_param(),
            (&cur_size).as_kernel_param(),
            (&cur_items).as_kernel_param(),
            (&counts).as_kernel_param(),
            (&prefix).as_kernel_param(),
            (&pick_var).as_kernel_param(),
            (&cur_assign).as_kernel_param(),
            stride_u32.as_kernel_param(),
            (&mut next_items).as_kernel_param(),
            (&mut next_size).as_kernel_param(),
            (&mut next_assign).as_kernel_param(),
            max_frontier_items_u32.as_kernel_param(),
        ];
        unsafe {
            expand.clone().launch(
                LaunchConfig {
                    grid_dim: ((max_frontier_items_u32 + 127) / 128, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                },
                &mut expand_params,
            )
        }
        .map_err(|e| XlogError::Kernel(format!("d4_frontier_expand_dense failed: {}", e)))?;

        std::mem::swap(&mut cur_items, &mut next_items);
        std::mem::swap(&mut cur_size, &mut next_size);
        std::mem::swap(&mut cur_assign, &mut next_assign);
    }

    Ok(GpuFrontierDense {
        items: cur_items,
        size: cur_size,
        assignments: cur_assign,
        stride_bytes,
        max_frontier_items,
    })
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

    validate_cnf_gpu(cnf, provider)?;

    let frontier = build_frontier_bitset(cnf, provider, config, compile_needed)?;
    provider
        .device()
        .synchronize()
        .map_err(|e| XlogError::Kernel(format!("sync after build_frontier failed: {}", e)))?;

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
    provider
        .device()
        .synchronize()
        .map_err(|e| XlogError::Kernel(format!("sync after d4_compile_count failed: {}", e)))?;

    let mut node_offsets = memory.alloc::<u32>(max_items as usize)?;
    let mut edge_offsets = memory.alloc::<u32>(max_items as usize)?;
    device
        .dtod_copy(&node_counts, &mut node_offsets)
        .map_err(|e| XlogError::Kernel(format!("dtod_copy(node_counts) failed: {}", e)))?;
    device
        .dtod_copy(&edge_counts, &mut edge_offsets)
        .map_err(|e| XlogError::Kernel(format!("dtod_copy(edge_counts) failed: {}", e)))?;
    exclusive_scan_u32_inplace(provider, &mut node_offsets, max_items_u32)?;
    exclusive_scan_u32_inplace(provider, &mut edge_offsets, max_items_u32)?;

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

    device
        .memset_zeros(&mut node_type)
        .map_err(|e| XlogError::Kernel(format!("Failed to zero node_type: {}", e)))?;
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

    let max_items_u32 = max_items;
    let node_cap_u32 = node_cap;
    let edge_cap_u32 = edge_cap;
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
    provider
        .device()
        .synchronize()
        .map_err(|e| XlogError::Kernel(format!("sync after d4_compile_emit failed: {}", e)))?;

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
                node_cap_u32,
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
                node_cap_u32,
                num_levels,
                &level_offsets,
                &mut level_cursors,
                &mut level_nodes,
            ),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("d4_levelize_emit failed: {}", e)))?;
    provider
        .device()
        .synchronize()
        .map_err(|e| XlogError::Kernel(format!("sync after d4_levelize_emit failed: {}", e)))?;

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
        num_levels,
        level_offsets,
        level_nodes,
        root: 2,
        max_var: cnf.var_cap,
    };

    GpuXgcf::from_device(builder, layout, provider)
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;
    use std::sync::Arc;

    use super::D4WorkItem;
    use cudarc::driver::{DeviceRepr, LaunchAsync, LaunchConfig};
    use xlog_core::MemoryBudget;
    use xlog_cuda::provider::{d4_kernels, scan_kernels, D4_MODULE, SCAN_MODULE};
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

    fn words_per_item(var_cap: u32) -> u32 {
        // Bit 0 is unused (DIMACS vars are 1-based), so we allocate var_cap+1 bits.
        let bits = var_cap.checked_add(1).expect("var_cap+1 overflow in test");
        (bits + 31) / 32
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
        };
        assert!(config.smooth_node_cap > 0);
        assert!(config.smooth_edge_cap > 0);
    }

    #[test]
    fn gpu_d4_frontier_prepare_and_expand_unit_clause_prunes_branching() {
        let Some(provider) = try_provider() else {
            return;
        };
        let device = provider.device().inner();
        let memory = provider.memory();

        // CNF: (x1). Unit propagation should assign x1=true and mark the instance satisfied.
        let instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
        let cnf = GpuCnf::from_host(&instance, &provider).expect("GpuCnf upload");

        let max_frontier_items: u32 = 8;
        let wpi = words_per_item(cnf.var_cap);

        let mut cur_items = memory
            .alloc::<D4WorkItem>(max_frontier_items as usize)
            .unwrap();
        let mut next_items = memory
            .alloc::<D4WorkItem>(max_frontier_items as usize)
            .unwrap();
        let mut cur_size = memory.alloc::<u32>(1).unwrap();
        let mut next_size = memory.alloc::<u32>(1).unwrap();

        // Bitset assignment pools: (true_bits, false_bits) per work item.
        let pool_len = (max_frontier_items as usize) * (wpi as usize);
        let mut cur_true = memory.alloc::<u32>(pool_len).unwrap();
        let mut cur_false = memory.alloc::<u32>(pool_len).unwrap();
        let mut next_true = memory.alloc::<u32>(pool_len).unwrap();
        let mut next_false = memory.alloc::<u32>(pool_len).unwrap();

        // Per-item frontier metadata.
        let mut counts = memory.alloc::<u32>(max_frontier_items as usize).unwrap();
        let mut pick_var = memory.alloc::<u32>(max_frontier_items as usize).unwrap();

        // Initialize: 1 work item with empty assignment.
        device.memset_zeros(&mut cur_true).unwrap();
        device.memset_zeros(&mut cur_false).unwrap();
        device.memset_zeros(&mut next_true).unwrap();
        device.memset_zeros(&mut next_false).unwrap();
        device.memset_zeros(&mut counts).unwrap();
        device.memset_zeros(&mut pick_var).unwrap();
        let host_items = vec![
            D4WorkItem {
                subproblem_id: 0,
                parent_node: u32::MAX,
                branch: 0,
                depth: 0,
                assignment_offset: 0,
            };
            max_frontier_items as usize
        ];
        device
            .htod_sync_copy_into(&host_items, &mut cur_items)
            .unwrap();
        device.htod_sync_copy_into(&[1u32], &mut cur_size).unwrap();
        device.memset_zeros(&mut next_size).unwrap();

        let prep = device
            .get_func(D4_MODULE, "d4_frontier_prepare")
            .expect("d4_frontier_prepare must exist");
        let frontier_depth_u32 = 4u32;
        let max_depth_u32 = 32u32;
        let wpi_u32 = wpi;
        let mut prep_params: Vec<*mut c_void> = vec![
            frontier_depth_u32.as_kernel_param(),
            max_depth_u32.as_kernel_param(),
            (&cnf.clause_offsets).as_kernel_param(),
            (&cnf.literals).as_kernel_param(),
            (&cnf.num_vars).as_kernel_param(),
            (&cnf.num_clauses).as_kernel_param(),
            (&cur_items).as_kernel_param(),
            (&cur_size).as_kernel_param(),
            (&mut cur_true).as_kernel_param(),
            (&mut cur_false).as_kernel_param(),
            wpi_u32.as_kernel_param(),
            (&mut counts).as_kernel_param(),
            (&mut pick_var).as_kernel_param(),
        ];
        unsafe {
            prep.clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (64, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut prep_params,
                )
                .unwrap();
        }

        // Compute exclusive prefix offsets for the output queue.
        let mut prefix = memory.alloc::<u32>(max_frontier_items as usize).unwrap();
        device.dtod_copy(&counts, &mut prefix).unwrap();

        // multiblock_scan_phase2 performs an *exclusive* scan in place.
        let scan = device
            .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_PHASE2)
            .expect("multiblock_scan_phase2 must exist");
        unsafe {
            scan.clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&mut prefix, 1u32),
                )
                .unwrap();
        }

        let expand = device
            .get_func(D4_MODULE, "d4_frontier_expand")
            .expect("d4_frontier_expand must exist");
        let max_frontier_items_u32 = max_frontier_items;
        let mut expand_params: Vec<*mut c_void> = vec![
            (&cur_size).as_kernel_param(),
            (&cur_items).as_kernel_param(),
            (&counts).as_kernel_param(), // per-item counts (1 or 2)
            (&prefix).as_kernel_param(), // exclusive prefix offsets
            (&pick_var).as_kernel_param(),
            (&cur_true).as_kernel_param(),
            (&cur_false).as_kernel_param(),
            wpi_u32.as_kernel_param(),
            (&mut next_items).as_kernel_param(),
            (&mut next_size).as_kernel_param(),
            (&mut next_true).as_kernel_param(),
            (&mut next_false).as_kernel_param(),
            max_frontier_items_u32.as_kernel_param(),
        ];
        unsafe {
            expand
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (64, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut expand_params,
                )
                .unwrap();
        }

        // GPU-only checks: next_size == 1 and x1 is true for work item 0.
        let assert_u32 = device
            .get_func(D4_MODULE, "d4_assert_u32_eq")
            .expect("d4_assert_u32_eq must exist");
        unsafe {
            assert_u32
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&next_size, 1u32),
                )
                .unwrap();
        }

        let assert_var = device
            .get_func(D4_MODULE, "d4_assert_bitset_var")
            .expect("d4_assert_bitset_var must exist");
        unsafe {
            assert_var
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &next_true,
                        &next_false,
                        wpi,
                        0u32, // work_id
                        1u32, // var id
                        1u32, // expected_state: 1=true, 2=false, 0=unassigned
                    ),
                )
                .unwrap();
        }
        provider.device().synchronize().unwrap();
    }

    #[test]
    fn gpu_d4_frontier_prepare_and_expand_branches_on_first_clause_min_var() {
        let Some(provider) = try_provider() else {
            return;
        };
        let device = provider.device().inner();
        let memory = provider.memory();

        // CNF: (x1 ∨ x2). No unit propagation initially; branch on x1 deterministically.
        let instance = SolveInstance::new(
            2,
            vec![Clause::new(vec![
                Literal::positive(0),
                Literal::positive(1),
            ])],
        );
        let cnf = GpuCnf::from_host(&instance, &provider).expect("GpuCnf upload");

        let max_frontier_items: u32 = 8;
        let wpi = words_per_item(cnf.var_cap);

        let mut cur_items = memory
            .alloc::<D4WorkItem>(max_frontier_items as usize)
            .unwrap();
        let mut next_items = memory
            .alloc::<D4WorkItem>(max_frontier_items as usize)
            .unwrap();
        let mut cur_size = memory.alloc::<u32>(1).unwrap();
        let mut next_size = memory.alloc::<u32>(1).unwrap();

        let pool_len = (max_frontier_items as usize) * (wpi as usize);
        let mut cur_true = memory.alloc::<u32>(pool_len).unwrap();
        let mut cur_false = memory.alloc::<u32>(pool_len).unwrap();
        let mut next_true = memory.alloc::<u32>(pool_len).unwrap();
        let mut next_false = memory.alloc::<u32>(pool_len).unwrap();

        let mut counts = memory.alloc::<u32>(max_frontier_items as usize).unwrap();
        let mut pick_var = memory.alloc::<u32>(max_frontier_items as usize).unwrap();

        device.memset_zeros(&mut cur_true).unwrap();
        device.memset_zeros(&mut cur_false).unwrap();
        device.memset_zeros(&mut next_true).unwrap();
        device.memset_zeros(&mut next_false).unwrap();
        device.memset_zeros(&mut counts).unwrap();
        device.memset_zeros(&mut pick_var).unwrap();
        let host_items = vec![
            D4WorkItem {
                subproblem_id: 0,
                parent_node: u32::MAX,
                branch: 0,
                depth: 0,
                assignment_offset: 0,
            };
            max_frontier_items as usize
        ];
        device
            .htod_sync_copy_into(&host_items, &mut cur_items)
            .unwrap();
        device.htod_sync_copy_into(&[1u32], &mut cur_size).unwrap();
        device.memset_zeros(&mut next_size).unwrap();

        let prep = device
            .get_func(D4_MODULE, "d4_frontier_prepare")
            .expect("d4_frontier_prepare must exist");
        let frontier_depth_u32 = 4u32;
        let max_depth_u32 = 32u32;
        let wpi_u32 = wpi;
        let mut prep_params: Vec<*mut c_void> = vec![
            frontier_depth_u32.as_kernel_param(),
            max_depth_u32.as_kernel_param(),
            (&cnf.clause_offsets).as_kernel_param(),
            (&cnf.literals).as_kernel_param(),
            (&cnf.num_vars).as_kernel_param(),
            (&cnf.num_clauses).as_kernel_param(),
            (&cur_items).as_kernel_param(),
            (&cur_size).as_kernel_param(),
            (&mut cur_true).as_kernel_param(),
            (&mut cur_false).as_kernel_param(),
            wpi_u32.as_kernel_param(),
            (&mut counts).as_kernel_param(),
            (&mut pick_var).as_kernel_param(),
        ];
        unsafe {
            prep.clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (64, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut prep_params,
                )
                .unwrap();
        }

        // Compute exclusive prefix offsets for the output queue.
        let mut prefix = memory.alloc::<u32>(max_frontier_items as usize).unwrap();
        device.dtod_copy(&counts, &mut prefix).unwrap();

        let scan = device
            .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_PHASE2)
            .expect("multiblock_scan_phase2 must exist");
        unsafe {
            scan.clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&mut prefix, 1u32),
                )
                .unwrap();
        }

        let expand = device
            .get_func(D4_MODULE, "d4_frontier_expand")
            .expect("d4_frontier_expand must exist");
        let max_frontier_items_u32 = max_frontier_items;
        let mut expand_params: Vec<*mut c_void> = vec![
            (&cur_size).as_kernel_param(),
            (&cur_items).as_kernel_param(),
            (&counts).as_kernel_param(), // per-item counts (1 or 2)
            (&prefix).as_kernel_param(), // exclusive prefix offsets
            (&pick_var).as_kernel_param(),
            (&cur_true).as_kernel_param(),
            (&cur_false).as_kernel_param(),
            wpi_u32.as_kernel_param(),
            (&mut next_items).as_kernel_param(),
            (&mut next_size).as_kernel_param(),
            (&mut next_true).as_kernel_param(),
            (&mut next_false).as_kernel_param(),
            max_frontier_items_u32.as_kernel_param(),
        ];
        unsafe {
            expand
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (64, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut expand_params,
                )
                .unwrap();
        }

        let assert_u32 = device
            .get_func(D4_MODULE, "d4_assert_u32_eq")
            .expect("d4_assert_u32_eq must exist");
        unsafe {
            assert_u32
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&next_size, 2u32),
                )
                .unwrap();
        }

        // Work item 0: x1 = false, Work item 1: x1 = true.
        let assert_var = device
            .get_func(D4_MODULE, "d4_assert_bitset_var")
            .expect("d4_assert_bitset_var must exist");
        unsafe {
            assert_var
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&next_true, &next_false, wpi, 0u32, 1u32, 2u32),
                )
                .unwrap();
        }
        unsafe {
            assert_var
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&next_true, &next_false, wpi, 1u32, 1u32, 1u32),
                )
                .unwrap();
        }
        provider.device().synchronize().unwrap();
    }

    #[test]
    fn gpu_d4_build_frontier_dense_depth2_sets_expected_assignments() {
        let Some(provider) = try_provider() else {
            return;
        };
        let device = provider.device().inner();

        // CNF: (x1 ∨ x2) ∧ (x3 ∨ x4). Branch on x1, then on x3 deterministically.
        let instance = SolveInstance::new(
            4,
            vec![
                Clause::new(vec![Literal::positive(0), Literal::positive(1)]),
                Clause::new(vec![Literal::positive(2), Literal::positive(3)]),
            ],
        );
        let cnf = GpuCnf::from_host(&instance, &provider).expect("GpuCnf upload");

        let config = super::GpuCompileConfig {
            frontier_depth: 2,
            max_frontier_items: 8,
            max_depth: 32,
            smooth_node_cap: 1024,
            smooth_edge_cap: 4096,
            cdcl_restart_interval: 64,
            cdcl_learned_bytes: 1 << 20,
            cdcl_conflict_budget: None,
        };

        let compile_needed = super::alloc_compile_gate(&provider, 1).expect("compile gate");
        let frontier = super::build_frontier_dense(&cnf, &provider, &config, &compile_needed)
            .expect("build_frontier_dense");

        let assert_u32 = device
            .get_func(D4_MODULE, "d4_assert_u32_eq")
            .expect("d4_assert_u32_eq must exist");
        unsafe {
            assert_u32
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&frontier.size, 4u32),
                )
                .unwrap();
        }

        let assert_dense = device
            .get_func(D4_MODULE, "d4_assert_dense_var")
            .expect("d4_assert_dense_var must exist");

        // Work item ordering is deterministic: children emitted in parent order, false then true.
        // Work 0: x1=false, x2=true (unit), x3=false.
        unsafe {
            assert_dense
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &frontier.assignments,
                        frontier.stride_bytes,
                        0u32,
                        1u32,
                        2u32,
                    ),
                )
                .unwrap();
        }
        unsafe {
            assert_dense
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &frontier.assignments,
                        frontier.stride_bytes,
                        0u32,
                        2u32,
                        1u32,
                    ),
                )
                .unwrap();
        }
        unsafe {
            assert_dense
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &frontier.assignments,
                        frontier.stride_bytes,
                        0u32,
                        3u32,
                        2u32,
                    ),
                )
                .unwrap();
        }

        // Work 1: x1=false, x2=true, x3=true.
        unsafe {
            assert_dense
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &frontier.assignments,
                        frontier.stride_bytes,
                        1u32,
                        1u32,
                        2u32,
                    ),
                )
                .unwrap();
        }
        unsafe {
            assert_dense
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &frontier.assignments,
                        frontier.stride_bytes,
                        1u32,
                        2u32,
                        1u32,
                    ),
                )
                .unwrap();
        }
        unsafe {
            assert_dense
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &frontier.assignments,
                        frontier.stride_bytes,
                        1u32,
                        3u32,
                        1u32,
                    ),
                )
                .unwrap();
        }

        // Work 2: x1=true, x2=unassigned, x3=false.
        unsafe {
            assert_dense
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &frontier.assignments,
                        frontier.stride_bytes,
                        2u32,
                        1u32,
                        1u32,
                    ),
                )
                .unwrap();
        }
        unsafe {
            assert_dense
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &frontier.assignments,
                        frontier.stride_bytes,
                        2u32,
                        2u32,
                        0u32,
                    ),
                )
                .unwrap();
        }
        unsafe {
            assert_dense
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &frontier.assignments,
                        frontier.stride_bytes,
                        2u32,
                        3u32,
                        2u32,
                    ),
                )
                .unwrap();
        }

        // Work 3: x1=true, x2=unassigned, x3=true.
        unsafe {
            assert_dense
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &frontier.assignments,
                        frontier.stride_bytes,
                        3u32,
                        1u32,
                        1u32,
                    ),
                )
                .unwrap();
        }
        unsafe {
            assert_dense
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &frontier.assignments,
                        frontier.stride_bytes,
                        3u32,
                        2u32,
                        0u32,
                    ),
                )
                .unwrap();
        }
        unsafe {
            assert_dense
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &frontier.assignments,
                        frontier.stride_bytes,
                        3u32,
                        3u32,
                        1u32,
                    ),
                )
                .unwrap();
        }

        provider.device().synchronize().unwrap();
    }

    #[test]
    fn gpu_d4_build_frontier_bitset_depth2_sets_expected_assignments() {
        let Some(provider) = try_provider() else {
            return;
        };
        let device = provider.device().inner();

        // CNF: (x1 ∨ x2) ∧ (x3 ∨ x4). Branch on x1, then on x3 deterministically.
        let instance = SolveInstance::new(
            4,
            vec![
                Clause::new(vec![Literal::positive(0), Literal::positive(1)]),
                Clause::new(vec![Literal::positive(2), Literal::positive(3)]),
            ],
        );
        let cnf = GpuCnf::from_host(&instance, &provider).expect("GpuCnf upload");

        let config = super::GpuCompileConfig {
            frontier_depth: 2,
            max_frontier_items: 8,
            max_depth: 32,
            smooth_node_cap: 1024,
            smooth_edge_cap: 4096,
            cdcl_restart_interval: 64,
            cdcl_learned_bytes: 1 << 20,
            cdcl_conflict_budget: None,
        };

        let compile_needed = super::alloc_compile_gate(&provider, 1).expect("compile gate");
        let frontier = super::build_frontier_bitset(&cnf, &provider, &config, &compile_needed)
            .expect("build_frontier_bitset");

        let assert_u32 = device
            .get_func(D4_MODULE, "d4_assert_u32_eq")
            .expect("d4_assert_u32_eq must exist");
        unsafe {
            assert_u32
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (frontier.size_device(), 4u32),
                )
                .unwrap();
        }

        let wpi = frontier.words_per_item();
        let assert_var = device
            .get_func(D4_MODULE, "d4_assert_bitset_var")
            .expect("d4_assert_bitset_var must exist");

        // Work 0: x1=false, x2=true (unit), x3=false.
        unsafe {
            assert_var
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        frontier.true_bits_device(),
                        frontier.false_bits_device(),
                        wpi,
                        0u32,
                        1u32,
                        2u32,
                    ),
                )
                .unwrap();
        }
        unsafe {
            assert_var
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        frontier.true_bits_device(),
                        frontier.false_bits_device(),
                        wpi,
                        0u32,
                        2u32,
                        1u32,
                    ),
                )
                .unwrap();
        }
        unsafe {
            assert_var
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        frontier.true_bits_device(),
                        frontier.false_bits_device(),
                        wpi,
                        0u32,
                        3u32,
                        2u32,
                    ),
                )
                .unwrap();
        }

        // Work 1: x1=false, x2=true, x3=true.
        unsafe {
            assert_var
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        frontier.true_bits_device(),
                        frontier.false_bits_device(),
                        wpi,
                        1u32,
                        1u32,
                        2u32,
                    ),
                )
                .unwrap();
        }
        unsafe {
            assert_var
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        frontier.true_bits_device(),
                        frontier.false_bits_device(),
                        wpi,
                        1u32,
                        2u32,
                        1u32,
                    ),
                )
                .unwrap();
        }
        unsafe {
            assert_var
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        frontier.true_bits_device(),
                        frontier.false_bits_device(),
                        wpi,
                        1u32,
                        3u32,
                        1u32,
                    ),
                )
                .unwrap();
        }

        // Work 2: x1=true, x2=unassigned, x3=false.
        unsafe {
            assert_var
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        frontier.true_bits_device(),
                        frontier.false_bits_device(),
                        wpi,
                        2u32,
                        1u32,
                        1u32,
                    ),
                )
                .unwrap();
        }
        unsafe {
            assert_var
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        frontier.true_bits_device(),
                        frontier.false_bits_device(),
                        wpi,
                        2u32,
                        2u32,
                        0u32,
                    ),
                )
                .unwrap();
        }
        unsafe {
            assert_var
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        frontier.true_bits_device(),
                        frontier.false_bits_device(),
                        wpi,
                        2u32,
                        3u32,
                        2u32,
                    ),
                )
                .unwrap();
        }

        // Work 3: x1=true, x2=unassigned, x3=true.
        unsafe {
            assert_var
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        frontier.true_bits_device(),
                        frontier.false_bits_device(),
                        wpi,
                        3u32,
                        1u32,
                        1u32,
                    ),
                )
                .unwrap();
        }
        unsafe {
            assert_var
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        frontier.true_bits_device(),
                        frontier.false_bits_device(),
                        wpi,
                        3u32,
                        2u32,
                        0u32,
                    ),
                )
                .unwrap();
        }
        unsafe {
            assert_var
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        frontier.true_bits_device(),
                        frontier.false_bits_device(),
                        wpi,
                        3u32,
                        3u32,
                        1u32,
                    ),
                )
                .unwrap();
        }

        provider.device().synchronize().unwrap();
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
                    (&node_level, node_cap, num_levels, &mut level_counts),
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
                        &node_level,
                        node_cap,
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
            num_levels,
            level_offsets,
            level_nodes,
            root: 2, // reserved root OR (see d4_compile_emit contract)
            max_var: phi.var_cap,
        };

        let circuit = crate::gpu::GpuXgcf::from_device(builder, layout, &provider)
            .expect("GpuXgcf::from_device must succeed");

        crate::compilation::validate_equivalence_gpu(
            &phi,
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
                    (&node_level, node_cap, num_levels, &mut level_counts),
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
                        &node_level,
                        node_cap,
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
            num_levels,
            level_offsets,
            level_nodes,
            root: 2,
            max_var: phi.var_cap,
        };
        let circuit = crate::gpu::GpuXgcf::from_device(builder, layout, &provider).unwrap();

        crate::compilation::validate_equivalence_gpu(
            &phi,
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
                    (&node_level, node_cap, num_levels, &mut level_counts),
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
                        &node_level,
                        node_cap,
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
            num_levels,
            level_offsets,
            level_nodes,
            root: 2,
            max_var: phi.var_cap,
        };
        let circuit = crate::gpu::GpuXgcf::from_device(builder, layout, &provider).unwrap();

        crate::compilation::validate_equivalence_gpu(
            &phi,
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
                    (&node_level, node_cap, num_levels, &mut level_counts),
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
                        &node_level,
                        node_cap,
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
            num_levels,
            level_offsets,
            level_nodes,
            root: 2,
            max_var: phi.var_cap,
        };
        let circuit = crate::gpu::GpuXgcf::from_device(builder, layout, &provider).unwrap();
        provider
            .device()
            .synchronize()
            .expect("sync after GpuXgcf::from_device must succeed");

        crate::compilation::validate_equivalence_gpu(
            &phi,
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
