//! XGCF circuit construction from frontier items.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::LaunchConfig;
use xlog_core::{Result, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::provider::{d4_kernels, D4_MODULE};
use xlog_cuda::{AsKernelParam, CudaKernelProvider, LaunchAsync};
use xlog_solve::GpuCnf;

use super::frontier::build_frontier_bitset;
use super::{
    checked_pool_len_usize, exclusive_scan_u32_inplace, memset_u8_sync, validate_cnf_gpu,
    GpuCompileConfig,
};
use crate::gpu::{GpuCircuitBuilder, GpuCircuitLayout, GpuXgcf};

type ComponentScratch = (
    TrackedCudaSlice<u32>,
    TrackedCudaSlice<u32>,
    TrackedCudaSlice<u32>,
    u32,
);

fn alloc_component_scratch(
    provider: &CudaKernelProvider,
    max_items: u32,
    var_cap: u32,
) -> Result<ComponentScratch> {
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

pub(super) fn compile_gpu_d4_with_gate(
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
    let (uf_parent, uf_aux, comp_list, uf_stride) =
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
        (&scratch_trail).as_kernel_param(),
        trail_stride_u32.as_kernel_param(),
        (&uf_parent).as_kernel_param(),
        (&uf_aux).as_kernel_param(),
        (&comp_list).as_kernel_param(),
        uf_stride_u32.as_kernel_param(),
        (&node_counts).as_kernel_param(),
        (&edge_counts).as_kernel_param(),
    ];

    #[cfg(debug_assertions)]
    eprintln!("[xlog-prob] gpu_d4: launch d4_compile_count");
    // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
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
    memset_u8_sync(&mut node_type, XGCF_LIT)?;
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
        (&scratch_trail).as_kernel_param(),
        trail_stride_u32.as_kernel_param(),
        (&uf_parent).as_kernel_param(),
        (&uf_aux).as_kernel_param(),
        (&comp_list).as_kernel_param(),
        uf_stride_u32.as_kernel_param(),
        (&node_counts).as_kernel_param(),
        (&edge_counts).as_kernel_param(),
        (&node_offsets).as_kernel_param(),
        (&edge_offsets).as_kernel_param(),
        (&node_type).as_kernel_param(),
        (&child_offsets).as_kernel_param(),
        (&child_indices).as_kernel_param(),
        (&lit).as_kernel_param(),
        (&decision_var).as_kernel_param(),
        (&decision_child_false).as_kernel_param(),
        (&decision_child_true).as_kernel_param(),
        (&node_level).as_kernel_param(),
    ];
    #[cfg(debug_assertions)]
    eprintln!("[xlog-prob] gpu_d4: launch d4_compile_emit");
    // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
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
    // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
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

    let num_levels = max_depth
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
    let mut grid = node_cap_u32.div_ceil(256);
    if grid == 0 {
        grid = 1;
    }
    #[cfg(debug_assertions)]
    eprintln!("[xlog-prob] gpu_d4: launch d4_levelize_counts");
    // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
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
    // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
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

    use xlog_core::MemoryBudget;
    use xlog_cuda::provider::{d4_kernels, D4_MODULE};
    use xlog_cuda::{
        AsKernelParam, CudaDevice, CudaKernelProvider, GpuMemoryManager, LaunchAsync, LaunchConfig,
    };
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
        provider: &Arc<CudaKernelProvider>,
        memory: &Arc<GpuMemoryManager>,
        max_items: u32,
        var_cap: u32,
    ) -> (
        xlog_cuda::memory::TrackedCudaSlice<u32>,
        xlog_cuda::memory::TrackedCudaSlice<u32>,
        xlog_cuda::memory::TrackedCudaSlice<u32>,
        u32,
    ) {
        let device = provider.device().inner();
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
    fn gpu_d4_compile_phase1_unit_clause_is_equivalent_via_gpu_cdcl() {
        let Some(provider) = try_provider() else {
            return;
        };
        let device = provider.device().inner();
        let memory = provider.memory();

        // φ = (x1)
        let instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
        let phi = GpuCnf::from_host(&instance, &provider).expect("GpuCnf upload");

        let cfg = super::super::GpuCompileConfig {
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

        super::super::validate_cnf_gpu(&phi, &provider).expect("CNF validation must succeed");

        // One frontier item (root) with empty assignment; DFS must handle unit propagation.
        let compile_needed = super::super::alloc_compile_gate(&provider, 1).expect("compile gate");
        let frontier = super::super::build_frontier_bitset(&phi, &provider, &cfg, &compile_needed)
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
        let (uf_parent, uf_aux, comp_list, uf_stride) =
            alloc_component_scratch(&provider, memory, max_items, phi.var_cap);

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
            (&scratch_trail).as_kernel_param(),
            trail_stride.as_kernel_param(),
            (&uf_parent).as_kernel_param(),
            (&uf_aux).as_kernel_param(),
            (&comp_list).as_kernel_param(),
            uf_stride.as_kernel_param(),
            (&node_counts).as_kernel_param(),
            (&edge_counts).as_kernel_param(),
        ];
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
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
        super::super::exclusive_scan_u32_inplace(&provider, &mut node_offsets, max_items)
            .expect("node scan must succeed");
        super::super::exclusive_scan_u32_inplace(&provider, &mut edge_offsets, max_items)
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
            (&scratch_trail).as_kernel_param(),
            trail_stride.as_kernel_param(),
            (&uf_parent).as_kernel_param(),
            (&uf_aux).as_kernel_param(),
            (&comp_list).as_kernel_param(),
            uf_stride.as_kernel_param(),
            (&node_counts).as_kernel_param(),
            (&edge_counts).as_kernel_param(),
            (&node_offsets).as_kernel_param(),
            (&edge_offsets).as_kernel_param(),
            (&node_type).as_kernel_param(),
            (&child_offsets).as_kernel_param(),
            (&child_indices).as_kernel_param(),
            (&lit).as_kernel_param(),
            (&decision_var).as_kernel_param(),
            (&decision_child_false).as_kernel_param(),
            (&decision_child_true).as_kernel_param(),
            (&node_level).as_kernel_param(),
        ];
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
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
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            lvl_counts
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (node_cap.div_ceil(256), 1, 1),
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
        super::super::exclusive_scan_u32_inplace(&provider, &mut level_offsets, num_levels + 1)
            .expect("level scan must succeed");

        let lvl_emit = device
            .get_func(D4_MODULE, d4_kernels::D4_LEVELIZE_EMIT)
            .expect("d4_levelize_emit must exist");
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            lvl_emit
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (node_cap.div_ceil(256), 1, 1),
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

        let cfg = super::super::GpuCompileConfig {
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

        super::super::validate_cnf_gpu(&phi, &provider).expect("CNF validation must succeed");
        let compile_needed = super::super::alloc_compile_gate(&provider, 1).expect("compile gate");
        let frontier = super::super::build_frontier_bitset(&phi, &provider, &cfg, &compile_needed)
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
        let (uf_parent, uf_aux, comp_list, uf_stride) =
            alloc_component_scratch(&provider, memory, max_items, phi.var_cap);

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
            (&scratch_trail).as_kernel_param(),
            trail_stride.as_kernel_param(),
            (&uf_parent).as_kernel_param(),
            (&uf_aux).as_kernel_param(),
            (&comp_list).as_kernel_param(),
            uf_stride.as_kernel_param(),
            (&node_counts).as_kernel_param(),
            (&edge_counts).as_kernel_param(),
        ];
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
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
        super::super::exclusive_scan_u32_inplace(&provider, &mut node_offsets, max_items).unwrap();
        super::super::exclusive_scan_u32_inplace(&provider, &mut edge_offsets, max_items).unwrap();
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
            (&scratch_trail).as_kernel_param(),
            trail_stride.as_kernel_param(),
            (&uf_parent).as_kernel_param(),
            (&uf_aux).as_kernel_param(),
            (&comp_list).as_kernel_param(),
            uf_stride.as_kernel_param(),
            (&node_counts).as_kernel_param(),
            (&edge_counts).as_kernel_param(),
            (&node_offsets).as_kernel_param(),
            (&edge_offsets).as_kernel_param(),
            (&node_type).as_kernel_param(),
            (&child_offsets).as_kernel_param(),
            (&child_indices).as_kernel_param(),
            (&lit).as_kernel_param(),
            (&decision_var).as_kernel_param(),
            (&decision_child_false).as_kernel_param(),
            (&decision_child_true).as_kernel_param(),
            (&node_level).as_kernel_param(),
        ];
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
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
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            lvl_counts
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (node_cap.div_ceil(256), 1, 1),
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
        super::super::exclusive_scan_u32_inplace(&provider, &mut level_offsets, num_levels + 1)
            .unwrap();

        let lvl_emit = device
            .get_func(D4_MODULE, d4_kernels::D4_LEVELIZE_EMIT)
            .expect("d4_levelize_emit must exist");
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            lvl_emit
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (node_cap.div_ceil(256), 1, 1),
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

        let cfg = super::super::GpuCompileConfig {
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

        super::super::validate_cnf_gpu(&phi, &provider).expect("CNF validation must succeed");
        let compile_needed = super::super::alloc_compile_gate(&provider, 1).expect("compile gate");
        let frontier = super::super::build_frontier_bitset(&phi, &provider, &cfg, &compile_needed)
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
        let (uf_parent, uf_aux, comp_list, uf_stride) =
            alloc_component_scratch(&provider, memory, max_items, phi.var_cap);

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
            (&scratch_trail).as_kernel_param(),
            trail_stride.as_kernel_param(),
            (&uf_parent).as_kernel_param(),
            (&uf_aux).as_kernel_param(),
            (&comp_list).as_kernel_param(),
            uf_stride.as_kernel_param(),
            (&node_counts).as_kernel_param(),
            (&edge_counts).as_kernel_param(),
        ];
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
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
        super::super::exclusive_scan_u32_inplace(&provider, &mut node_offsets, max_items).unwrap();
        super::super::exclusive_scan_u32_inplace(&provider, &mut edge_offsets, max_items).unwrap();

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
            (&scratch_trail).as_kernel_param(),
            trail_stride.as_kernel_param(),
            (&uf_parent).as_kernel_param(),
            (&uf_aux).as_kernel_param(),
            (&comp_list).as_kernel_param(),
            uf_stride.as_kernel_param(),
            (&node_counts).as_kernel_param(),
            (&edge_counts).as_kernel_param(),
            (&node_offsets).as_kernel_param(),
            (&edge_offsets).as_kernel_param(),
            (&node_type).as_kernel_param(),
            (&child_offsets).as_kernel_param(),
            (&child_indices).as_kernel_param(),
            (&lit).as_kernel_param(),
            (&decision_var).as_kernel_param(),
            (&decision_child_false).as_kernel_param(),
            (&decision_child_true).as_kernel_param(),
            (&node_level).as_kernel_param(),
        ];
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
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
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            lvl_counts
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (node_cap.div_ceil(256), 1, 1),
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
        super::super::exclusive_scan_u32_inplace(&provider, &mut level_offsets, num_levels + 1)
            .unwrap();
        provider
            .device()
            .synchronize()
            .expect("sync after level scan must succeed");

        let lvl_emit = device
            .get_func(D4_MODULE, d4_kernels::D4_LEVELIZE_EMIT)
            .expect("d4_levelize_emit must exist");
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            lvl_emit
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (node_cap.div_ceil(256), 1, 1),
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

        let cfg = super::super::GpuCompileConfig {
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

        super::super::validate_cnf_gpu(&phi, &provider).expect("CNF validation must succeed");
        let compile_needed = super::super::alloc_compile_gate(&provider, 1).expect("compile gate");
        let frontier = super::super::build_frontier_bitset(&phi, &provider, &cfg, &compile_needed)
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
        let (uf_parent, uf_aux, comp_list, uf_stride) =
            alloc_component_scratch(&provider, memory, max_items, phi.var_cap);

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
            (&scratch_trail).as_kernel_param(),
            trail_stride.as_kernel_param(),
            (&uf_parent).as_kernel_param(),
            (&uf_aux).as_kernel_param(),
            (&comp_list).as_kernel_param(),
            uf_stride.as_kernel_param(),
            (&node_counts).as_kernel_param(),
            (&edge_counts).as_kernel_param(),
        ];
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
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
        super::super::exclusive_scan_u32_inplace(&provider, &mut node_offsets, max_items).unwrap();
        super::super::exclusive_scan_u32_inplace(&provider, &mut edge_offsets, max_items).unwrap();

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
            (&scratch_trail).as_kernel_param(),
            trail_stride.as_kernel_param(),
            (&uf_parent).as_kernel_param(),
            (&uf_aux).as_kernel_param(),
            (&comp_list).as_kernel_param(),
            uf_stride.as_kernel_param(),
            (&node_counts).as_kernel_param(),
            (&edge_counts).as_kernel_param(),
            (&node_offsets).as_kernel_param(),
            (&edge_offsets).as_kernel_param(),
            (&node_type).as_kernel_param(),
            (&child_offsets).as_kernel_param(),
            (&child_indices).as_kernel_param(),
            (&lit).as_kernel_param(),
            (&decision_var).as_kernel_param(),
            (&decision_child_false).as_kernel_param(),
            (&decision_child_true).as_kernel_param(),
            (&node_level).as_kernel_param(),
        ];
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
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
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            lvl_counts
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (node_cap.div_ceil(256), 1, 1),
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
        super::super::exclusive_scan_u32_inplace(&provider, &mut level_offsets, num_levels + 1)
            .unwrap();

        let lvl_emit = device
            .get_func(D4_MODULE, d4_kernels::D4_LEVELIZE_EMIT)
            .expect("d4_levelize_emit must exist");
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            lvl_emit
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (node_cap.div_ceil(256), 1, 1),
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

        let cfg = super::super::GpuCompileConfig {
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

        super::super::validate_cnf_gpu(&phi, &provider).expect("CNF validation must succeed");
        let compile_needed = super::super::alloc_compile_gate(&provider, 1).expect("compile gate");
        let frontier = super::super::build_frontier_bitset(&phi, &provider, &cfg, &compile_needed)
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
        let (uf_parent, uf_aux, comp_list, uf_stride) =
            alloc_component_scratch(&provider, memory, max_items, phi.var_cap);

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
            (&scratch_trail).as_kernel_param(),
            trail_stride.as_kernel_param(),
            (&uf_parent).as_kernel_param(),
            (&uf_aux).as_kernel_param(),
            (&comp_list).as_kernel_param(),
            uf_stride.as_kernel_param(),
            (&node_counts).as_kernel_param(),
            (&edge_counts).as_kernel_param(),
        ];
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
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
        super::super::exclusive_scan_u32_inplace(&provider, &mut node_offsets, max_items).unwrap();
        super::super::exclusive_scan_u32_inplace(&provider, &mut edge_offsets, max_items).unwrap();
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
            (&scratch_trail).as_kernel_param(),
            trail_stride.as_kernel_param(),
            (&uf_parent).as_kernel_param(),
            (&uf_aux).as_kernel_param(),
            (&comp_list).as_kernel_param(),
            uf_stride.as_kernel_param(),
            (&node_counts).as_kernel_param(),
            (&edge_counts).as_kernel_param(),
            (&node_offsets).as_kernel_param(),
            (&edge_offsets).as_kernel_param(),
            (&node_type).as_kernel_param(),
            (&child_offsets).as_kernel_param(),
            (&child_indices).as_kernel_param(),
            (&lit).as_kernel_param(),
            (&decision_var).as_kernel_param(),
            (&decision_child_false).as_kernel_param(),
            (&decision_child_true).as_kernel_param(),
            (&node_level).as_kernel_param(),
        ];
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
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
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
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
