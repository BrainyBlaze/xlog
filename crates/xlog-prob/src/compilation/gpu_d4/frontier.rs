//! Frontier expansion for GPU D4 compilation.

use std::ffi::c_void;

use cudarc::driver::{DeviceRepr, DeviceSlice, LaunchAsync, LaunchConfig};
use xlog_core::{Result, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::provider::{d4_kernels, D4_MODULE};
use xlog_cuda::CudaKernelProvider;
use xlog_solve::GpuCnf;

use super::{
    bitset_words_per_item, checked_pool_len_usize, exclusive_scan_u32_inplace, GpuCompileConfig,
};

// ---------------------------------------------------------------------------
// Frontier work queues (Phase 1) - device-resident BFS expansion
// ---------------------------------------------------------------------------

/// Must match `struct D4WorkItem` in `kernels/d4.cu`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct D4WorkItem {
    subproblem_id: u32,
    parent_node: u32,
    branch: u8,
    depth: u16,
    assignment_offset: u32,
}

unsafe impl DeviceRepr for D4WorkItem {}

/// Device-resident frontier state using compressed bitset assignments.
pub(crate) struct GpuFrontierBitset {
    pub(super) items: TrackedCudaSlice<D4WorkItem>,
    size: TrackedCudaSlice<u32>,
    true_bits: TrackedCudaSlice<u32>,
    false_bits: TrackedCudaSlice<u32>,
    pub(super) words_per_item: u32,
    #[allow(dead_code)] // retained for diagnostics / future sizing checks
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

    #[allow(dead_code)] // diagnostic accessor
    pub fn max_frontier_items(&self) -> u32 {
        self.max_frontier_items
    }

    #[allow(dead_code)] // diagnostic accessor
    pub fn items_len(&self) -> usize {
        self.items.len()
    }
}

/// Device-resident frontier state using dense tri-state assignments.
#[allow(dead_code)] // reserved: dense frontier is an alternative to bitset (not yet wired)
pub(crate) struct GpuFrontierDense {
    pub(super) items: TrackedCudaSlice<D4WorkItem>,
    size: TrackedCudaSlice<u32>,
    assignments: TrackedCudaSlice<u8>,
    stride_bytes: u32,
    max_frontier_items: u32,
}

#[allow(dead_code)] // reserved: dense frontier is an alternative to bitset (not yet wired)
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
pub(crate) fn build_frontier_bitset(
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
            (&compile_needed).as_kernel_param(),
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
            (&compile_needed).as_kernel_param(),
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
#[allow(dead_code)] // reserved: dense frontier is an alternative to bitset (not yet wired)
pub(crate) fn build_frontier_dense(
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
            (&compile_needed).as_kernel_param(),
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
            (&compile_needed).as_kernel_param(),
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

#[cfg(test)]
mod tests {
    use std::ffi::c_void;
    use std::sync::Arc;

    use super::D4WorkItem;
    use cudarc::driver::{DeviceRepr, LaunchAsync, LaunchConfig};
    use xlog_core::MemoryBudget;
    use xlog_cuda::provider::{scan_kernels, D4_MODULE, SCAN_MODULE};
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
        let compile_needed = super::super::alloc_compile_gate(&provider, 1).expect("compile gate");

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
            (&compile_needed).as_kernel_param(),
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
            (&compile_needed).as_kernel_param(),
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
        let compile_needed = super::super::alloc_compile_gate(&provider, 1).expect("compile gate");

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
            (&compile_needed).as_kernel_param(),
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
            (&compile_needed).as_kernel_param(),
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

        let config = super::super::GpuCompileConfig {
            frontier_depth: 2,
            max_frontier_items: 8,
            max_depth: 32,
            smooth_node_cap: 1024,
            smooth_edge_cap: 4096,
            cdcl_restart_interval: 64,
            cdcl_learned_bytes: 1 << 20,
            cdcl_conflict_budget: None,
            incremental_verify: false,
        };

        let compile_needed = super::super::alloc_compile_gate(&provider, 1).expect("compile gate");
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

        let config = super::super::GpuCompileConfig {
            frontier_depth: 2,
            max_frontier_items: 8,
            max_depth: 32,
            smooth_node_cap: 1024,
            smooth_edge_cap: 4096,
            cdcl_restart_interval: 64,
            cdcl_learned_bytes: 1 << 20,
            cdcl_conflict_budget: None,
            incremental_verify: false,
        };

        let compile_needed = super::super::alloc_compile_gate(&provider, 1).expect("compile gate");
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
}
