//! GPU evaluator for XGCF circuits (CUDA).

use std::ffi::c_void;

use cudarc::driver::{DeviceRepr, DeviceSlice, LaunchAsync, LaunchConfig};
use xlog_core::{Result, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::provider::{
    arith_kernels, d4_kernels, filter_kernels, ARITH_MODULE, D4_MODULE, FILTER_MODULE,
};
use xlog_cuda::{circuit_kernels, CudaKernelProvider, CIRCUIT_MODULE};

use crate::compilation::gpu_d4::exclusive_scan_u32_inplace;
use crate::xgcf::{Xgcf, XgcfNodeType};

/// Device-resident circuit buffers produced by the GPU compiler.
///
/// This matches the XGCF node layout used by `kernels/circuit.cu` and the SAT verifier CNF encoder
/// in `kernels/sat.cu`.
pub struct GpuCircuitBuilder {
    pub node_type: TrackedCudaSlice<u8>,
    pub child_offsets: TrackedCudaSlice<u32>,
    pub child_indices: TrackedCudaSlice<u32>,
    pub lit: TrackedCudaSlice<i32>,
    pub decision_var: TrackedCudaSlice<u32>,
    pub decision_child_false: TrackedCudaSlice<u32>,
    pub decision_child_true: TrackedCudaSlice<u32>,
}

/// Device layout metadata for XGCF construction.
pub struct GpuCircuitLayout {
    pub num_nodes: u32,
    pub num_edges: u32,
    pub num_levels: u32,
    pub level_offsets: TrackedCudaSlice<u32>,
    pub level_nodes: TrackedCudaSlice<u32>,
    pub root: u32,
    pub max_var: u32,
    pub num_nodes_device: Option<TrackedCudaSlice<u32>>,
    pub num_edges_device: Option<TrackedCudaSlice<u32>>,
}

pub struct GpuXgcf {
    node_type: TrackedCudaSlice<u8>,
    child_offsets: TrackedCudaSlice<u32>,
    child_indices: TrackedCudaSlice<u32>,
    lit: TrackedCudaSlice<i32>,
    decision_var: TrackedCudaSlice<u32>,
    decision_child_false: TrackedCudaSlice<u32>,
    decision_child_true: TrackedCudaSlice<u32>,
    level_nodes: TrackedCudaSlice<u32>,
    level_offsets: TrackedCudaSlice<u32>,
    /// Optional host mirror for efficient per-level launch sizing when the circuit was uploaded
    /// from host (`GpuXgcf::upload`). GPU-native compilation paths do not populate this.
    level_offsets_host: Option<Vec<u32>>,
    node_cap: u32,
    edge_cap: u32,
    num_levels: u32,
    root: u32,
    max_var: u32,
    meta_num_nodes: TrackedCudaSlice<u32>,
    meta_num_edges: TrackedCudaSlice<u32>,
    var_log_true: TrackedCudaSlice<f64>,
    var_log_false: TrackedCudaSlice<f64>,
    values: TrackedCudaSlice<f64>,
    adj: TrackedCudaSlice<f64>,
    grad_true: TrackedCudaSlice<f64>,
    grad_false: TrackedCudaSlice<f64>,
    free_var_mask: Option<TrackedCudaSlice<u8>>,
}

impl GpuXgcf {
    pub fn from_device(
        builder: GpuCircuitBuilder,
        layout: GpuCircuitLayout,
        provider: &CudaKernelProvider,
    ) -> Result<GpuXgcf> {
        if layout.num_nodes == 0 {
            return Err(XlogError::Compilation(
                "GpuXgcf::from_device requires num_nodes > 0".to_string(),
            ));
        }
        if layout.root >= layout.num_nodes {
            return Err(XlogError::Compilation(format!(
                "GpuXgcf::from_device: root {} out of bounds (num_nodes={})",
                layout.root, layout.num_nodes
            )));
        }
        if layout.num_levels == 0 {
            return Err(XlogError::Compilation(
                "GpuXgcf::from_device requires num_levels > 0".to_string(),
            ));
        }

        let num_nodes = layout.num_nodes as usize;
        let num_edges = layout.num_edges as usize;
        let node_cap = builder.node_type.len();
        if num_nodes == 0 || num_nodes > node_cap {
            return Err(XlogError::Compilation(
                "GpuXgcf::from_device: num_nodes out of bounds".to_string(),
            ));
        }
        if builder.child_offsets.len() != node_cap + 1
            || builder.lit.len() != node_cap
            || builder.decision_var.len() != node_cap
            || builder.decision_child_false.len() != node_cap
            || builder.decision_child_true.len() != node_cap
        {
            return Err(XlogError::Compilation(
                "GpuXgcf::from_device: circuit buffer length mismatch".to_string(),
            ));
        }
        if num_edges > builder.child_indices.len() {
            return Err(XlogError::Compilation(
                "GpuXgcf::from_device: num_edges out of bounds".to_string(),
            ));
        }

        let num_levels = layout.num_levels as usize;
        if layout.level_offsets.len() != num_levels + 1 {
            return Err(XlogError::Compilation(format!(
                "GpuXgcf::from_device: level_offsets len {} != num_levels+1 ({})",
                layout.level_offsets.len(),
                num_levels + 1
            )));
        }
        if layout.level_nodes.len() < num_nodes {
            return Err(XlogError::Compilation(format!(
                "GpuXgcf::from_device: level_nodes len {} < num_nodes ({})",
                layout.level_nodes.len(),
                num_nodes
            )));
        }

        let memory = provider.memory();

        let weights_len = (layout.max_var as usize) + 1;
        let var_log_true = memory.alloc::<f64>(weights_len)?;
        let var_log_false = memory.alloc::<f64>(weights_len)?;
        let values = memory.alloc::<f64>(num_nodes)?;
        let adj = memory.alloc::<f64>(num_nodes)?;
        let grad_true = memory.alloc::<f64>(weights_len)?;
        let grad_false = memory.alloc::<f64>(weights_len)?;

        let meta_num_nodes = match layout.num_nodes_device {
            Some(meta) => meta,
            None => {
                let mut meta = memory.alloc::<u32>(1)?;
                provider
                    .device()
                    .inner()
                    .htod_sync_copy_into(&[layout.num_nodes], &mut meta)
                    .map_err(|e| {
                        XlogError::Kernel(format!("Failed to upload num_nodes meta: {}", e))
                    })?;
                meta
            }
        };
        let meta_num_edges = match layout.num_edges_device {
            Some(meta) => meta,
            None => {
                let mut meta = memory.alloc::<u32>(1)?;
                provider
                    .device()
                    .inner()
                    .htod_sync_copy_into(&[layout.num_edges], &mut meta)
                    .map_err(|e| {
                        XlogError::Kernel(format!("Failed to upload num_edges meta: {}", e))
                    })?;
                meta
            }
        };

        Ok(Self {
            node_type: builder.node_type,
            child_offsets: builder.child_offsets,
            child_indices: builder.child_indices,
            lit: builder.lit,
            decision_var: builder.decision_var,
            decision_child_false: builder.decision_child_false,
            decision_child_true: builder.decision_child_true,
            level_nodes: layout.level_nodes,
            level_offsets: layout.level_offsets,
            level_offsets_host: None,
            node_cap: layout.num_nodes,
            edge_cap: layout.num_edges,
            num_levels: layout.num_levels,
            root: layout.root,
            max_var: layout.max_var,
            meta_num_nodes,
            meta_num_edges,
            var_log_true,
            var_log_false,
            values,
            adj,
            grad_true,
            grad_false,
            free_var_mask: None,
        })
    }

    /// GPU-native smoothing pass for random variables.
    ///
    /// Returns a new device-resident circuit that is smooth w.r.t. `random_var_list`.
    /// This method performs no device->host data-plane transfers and traps on capacity overflow.
    pub fn smooth_random_vars_device(
        &self,
        provider: &CudaKernelProvider,
        random_var_list: &TrackedCudaSlice<u32>,
        random_var_count: u32,
        smooth_node_cap: u32,
        smooth_edge_cap: u32,
    ) -> Result<GpuXgcf> {
        if smooth_node_cap == 0 || smooth_edge_cap == 0 {
            return Err(XlogError::Compilation(
                "GPU smoothing requires non-zero node/edge caps".to_string(),
            ));
        }

        let num_nodes = self.node_cap;
        if num_nodes == 0 {
            return Err(XlogError::Compilation(
                "GPU smoothing: num_nodes must be > 0".to_string(),
            ));
        }
        if self.child_offsets.len() < (num_nodes as usize + 1) {
            return Err(XlogError::Compilation(
                "GPU smoothing: child_offsets len mismatch".to_string(),
            ));
        }
        let num_edges = self.edge_cap;
        if num_edges == 0 {
            return Err(XlogError::Compilation(
                "GPU smoothing: num_edges must be > 0".to_string(),
            ));
        }

        let list_len = u32::try_from(random_var_list.len()).map_err(|_| {
            XlogError::Compilation("GPU smoothing: random var list len exceeds u32".to_string())
        })?;
        let num_random_vars = random_var_count;
        if num_random_vars > list_len {
            return Err(XlogError::Compilation(format!(
                "GPU smoothing: random var count {} exceeds list len {}",
                num_random_vars, list_len
            )));
        }

        let base_node = 2u32.checked_add(num_random_vars).ok_or_else(|| {
            XlogError::Compilation("GPU smoothing: base node overflow".to_string())
        })?;
        let base_nodes = (base_node as u64)
            .checked_add(num_nodes as u64)
            .ok_or_else(|| {
                XlogError::Compilation("GPU smoothing: base node overflow".to_string())
            })?;
        if base_nodes > smooth_node_cap as u64 {
            return Err(XlogError::Compilation(format!(
                "GPU smoothing: base nodes {} exceed smooth_node_cap {}",
                base_nodes, smooth_node_cap
            )));
        }

        let words_per_support = ((num_random_vars + 31) / 32).max(1);

        let support_len = (num_nodes as u64)
            .checked_mul(words_per_support as u64)
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| {
                XlogError::Compilation("GPU smoothing: support size overflow".to_string())
            })?;

        let dec_entries = (num_nodes as u64)
            .checked_mul(2)
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| {
                XlogError::Compilation("GPU smoothing: decision array overflow".to_string())
            })?;
        let dec_entries_u32 = u32::try_from(dec_entries).map_err(|_| {
            XlogError::Compilation("GPU smoothing: decision entries exceed u32".to_string())
        })?;

        let device = provider.device().inner();
        let memory = provider.memory();
        let block_size: u32 = 256;

        let map_len = (self.max_var as usize)
            .checked_add(1)
            .ok_or_else(|| XlogError::Compilation("GPU smoothing: max_var overflow".to_string()))?;
        let map_len_u32 = u32::try_from(map_len).map_err(|_| {
            XlogError::Compilation("GPU smoothing: random map len exceeds u32".to_string())
        })?;
        let mut d_random_map = memory.alloc::<u32>(map_len)?;
        if map_len > 0 {
            let fill_const = device
                .get_func(FILTER_MODULE, filter_kernels::FILL_U32_CONST)
                .ok_or_else(|| XlogError::Kernel("fill_u32_const kernel not found".to_string()))?;
            let grid = (map_len_u32 + block_size - 1) / block_size;
            unsafe {
                fill_const.clone().launch(
                    LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&mut d_random_map, map_len_u32, u32::MAX),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("fill_u32_const failed: {}", e)))?;
        }
        if num_random_vars > 0 {
            let map_kernel = device
                .get_func(FILTER_MODULE, filter_kernels::RANDOM_VAR_TO_BIT_FROM_LIST)
                .ok_or_else(|| {
                    XlogError::Kernel("random_var_to_bit_from_list kernel not found".to_string())
                })?;
            let grid = (num_random_vars + block_size - 1) / block_size;
            unsafe {
                map_kernel.clone().launch(
                    LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        random_var_list,
                        num_random_vars,
                        map_len_u32,
                        &mut d_random_map,
                    ),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("random_var_to_bit_from_list failed: {}", e)))?;
        }

        let mut support = memory.alloc::<u32>(support_len)?;
        device
            .memset_zeros(&mut support)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero support: {}", e)))?;

        let support_kernel = device
            .get_func(D4_MODULE, d4_kernels::D4_SUPPORT_LEVEL)
            .ok_or_else(|| XlogError::Kernel("d4_support_level kernel not found".to_string()))?;

        let num_levels = self.num_levels as usize;
        let random_map_len = map_len_u32;
        for level in 0..num_levels {
            let num_level_nodes: usize = match self.level_offsets_host.as_ref() {
                Some(off) => (off[level + 1].saturating_sub(off[level])) as usize,
                None => self.level_nodes.len(),
            };
            if num_level_nodes == 0 {
                continue;
            }
            let num_blocks = ((num_level_nodes as u32) + block_size - 1) / block_size;
            let config = LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            let level_u32 = level as u32;
            let mut params: Vec<*mut c_void> = vec![
                (&self.node_type).as_kernel_param(),
                (&self.child_offsets).as_kernel_param(),
                (&self.child_indices).as_kernel_param(),
                (&self.lit).as_kernel_param(),
                (&self.decision_var).as_kernel_param(),
                (&self.decision_child_false).as_kernel_param(),
                (&self.decision_child_true).as_kernel_param(),
                (&self.level_nodes).as_kernel_param(),
                (&self.level_offsets).as_kernel_param(),
                level_u32.as_kernel_param(),
                (&d_random_map).as_kernel_param(),
                random_map_len.as_kernel_param(),
                words_per_support.as_kernel_param(),
                (&mut support).as_kernel_param(),
            ];
            unsafe { support_kernel.clone().launch(config, &mut params) }
                .map_err(|e| XlogError::Kernel(format!("d4_support_level failed: {}", e)))?;
        }

        if num_random_vars > 0 {
            let root_kernel = device
                .get_func(D4_MODULE, d4_kernels::D4_SUPPORT_SET_ROOT_BITS)
                .ok_or_else(|| {
                    XlogError::Kernel("d4_support_set_root_bits kernel not found".to_string())
                })?;
            let num_words = (num_random_vars + 31) / 32;
            let grid = (num_words + block_size - 1) / block_size;
            unsafe {
                root_kernel.clone().launch(
                    LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (self.root, num_random_vars, words_per_support, &mut support),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("d4_support_set_root_bits failed: {}", e)))?;
        }

        let mut wrap_prefix_or = memory.alloc::<u32>(num_edges as usize)?;
        let mut wrap_missing_or = memory.alloc::<u32>(num_edges as usize)?;
        let mut wrap_prefix_dec = memory.alloc::<u32>(dec_entries)?;
        let mut wrap_missing_dec = memory.alloc::<u32>(dec_entries)?;

        device
            .memset_zeros(&mut wrap_prefix_or)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero wrap_prefix_or: {}", e)))?;
        device
            .memset_zeros(&mut wrap_missing_or)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero wrap_missing_or: {}", e)))?;
        device
            .memset_zeros(&mut wrap_prefix_dec)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero wrap_prefix_dec: {}", e)))?;
        device
            .memset_zeros(&mut wrap_missing_dec)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero wrap_missing_dec: {}", e)))?;

        let mut out_edge_counts = memory.alloc::<u32>(smooth_node_cap as usize)?;
        device
            .memset_zeros(&mut out_edge_counts)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero edge_counts: {}", e)))?;

        let count_kernel = device
            .get_func(D4_MODULE, d4_kernels::D4_SMOOTH_COUNT)
            .ok_or_else(|| XlogError::Kernel("d4_smooth_count kernel not found".to_string()))?;
        let num_blocks = (num_nodes + block_size - 1) / block_size;
        let mut params: Vec<*mut c_void> = vec![
            (&self.node_type).as_kernel_param(),
            (&self.child_offsets).as_kernel_param(),
            (&self.child_indices).as_kernel_param(),
            (&self.decision_var).as_kernel_param(),
            (&self.decision_child_false).as_kernel_param(),
            (&self.decision_child_true).as_kernel_param(),
            (&self.meta_num_nodes).as_kernel_param(),
            (&support).as_kernel_param(),
            words_per_support.as_kernel_param(),
            (&d_random_map).as_kernel_param(),
            random_map_len.as_kernel_param(),
            (&mut wrap_prefix_or).as_kernel_param(),
            (&mut wrap_missing_or).as_kernel_param(),
            (&mut wrap_prefix_dec).as_kernel_param(),
            (&mut wrap_missing_dec).as_kernel_param(),
            (&mut out_edge_counts).as_kernel_param(),
            base_node.as_kernel_param(),
            smooth_node_cap.as_kernel_param(),
        ];
        unsafe {
            count_kernel.clone().launch(
                LaunchConfig {
                    grid_dim: (num_blocks, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                &mut params,
            )
        }
        .map_err(|e| XlogError::Kernel(format!("d4_smooth_count failed: {}", e)))?;

        exclusive_scan_u32_inplace(provider, &mut wrap_prefix_or, num_edges)?;
        exclusive_scan_u32_inplace(provider, &mut wrap_prefix_dec, dec_entries_u32)?;

        let mut wrap_counts = memory.alloc::<u32>(3)?;
        device
            .memset_zeros(&mut wrap_counts)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero wrap_counts: {}", e)))?;

        let counts_kernel = device
            .get_func(D4_MODULE, d4_kernels::D4_SMOOTH_WRAPPER_COUNTS)
            .ok_or_else(|| {
                XlogError::Kernel("d4_smooth_wrapper_counts kernel not found".to_string())
            })?;
        unsafe {
            counts_kernel.clone().launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &wrap_prefix_or,
                    &wrap_missing_or,
                    num_edges,
                    &wrap_prefix_dec,
                    &wrap_missing_dec,
                    dec_entries_u32,
                    base_node,
                    &self.meta_num_nodes,
                    u32::MAX,
                    &mut wrap_counts,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("d4_smooth_wrapper_counts failed: {}", e)))?;

        let wrap_or_kernel = device
            .get_func(D4_MODULE, d4_kernels::D4_SMOOTH_WRAPPER_EDGE_COUNTS_OR)
            .ok_or_else(|| {
                XlogError::Kernel("d4_smooth_wrapper_edge_counts_or kernel not found".to_string())
            })?;
        if num_edges > 0 {
            let num_blocks = (num_edges + block_size - 1) / block_size;
            unsafe {
                wrap_or_kernel.clone().launch(
                    LaunchConfig {
                        grid_dim: (num_blocks, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &wrap_prefix_or,
                        &wrap_missing_or,
                        num_edges,
                        base_node,
                        &self.meta_num_nodes,
                        smooth_node_cap,
                        &mut out_edge_counts,
                    ),
                )
            }
            .map_err(|e| {
                XlogError::Kernel(format!("d4_smooth_wrapper_edge_counts_or failed: {}", e))
            })?;
        }

        let wrap_dec_kernel = device
            .get_func(D4_MODULE, d4_kernels::D4_SMOOTH_WRAPPER_EDGE_COUNTS_DEC)
            .ok_or_else(|| {
                XlogError::Kernel("d4_smooth_wrapper_edge_counts_dec kernel not found".to_string())
            })?;
        if dec_entries > 0 {
            let num_blocks = (dec_entries_u32 + block_size - 1) / block_size;
            unsafe {
                wrap_dec_kernel.clone().launch(
                    LaunchConfig {
                        grid_dim: (num_blocks, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &wrap_prefix_dec,
                        &wrap_missing_dec,
                        dec_entries_u32,
                        base_node,
                        &self.meta_num_nodes,
                        &wrap_counts,
                        smooth_node_cap,
                        &mut out_edge_counts,
                    ),
                )
            }
            .map_err(|e| {
                XlogError::Kernel(format!("d4_smooth_wrapper_edge_counts_dec failed: {}", e))
            })?;
        }

        let mut out_child_offsets = memory.alloc::<u32>((smooth_node_cap as usize) + 1)?;
        device
            .memset_zeros(&mut out_child_offsets)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero child_offsets: {}", e)))?;
        if smooth_node_cap > 0 {
            device
                .dtod_copy(
                    &out_edge_counts,
                    &mut out_child_offsets.slice_mut(0..smooth_node_cap as usize),
                )
                .map_err(|e| XlogError::Kernel(format!("Failed to copy edge_counts: {}", e)))?;
        }
        let child_scan_len = smooth_node_cap.checked_add(1).ok_or_else(|| {
            XlogError::Compilation("GPU smoothing: child offset scan overflow".to_string())
        })?;
        exclusive_scan_u32_inplace(provider, &mut out_child_offsets, child_scan_len)?;

        let edge_cap_check = device
            .get_func(D4_MODULE, d4_kernels::D4_SMOOTH_CHECK_EDGE_CAP)
            .ok_or_else(|| {
                XlogError::Kernel("d4_smooth_check_edge_cap kernel not found".to_string())
            })?;
        let mut meta_num_nodes = memory.alloc::<u32>(1)?;
        let mut meta_num_edges = memory.alloc::<u32>(1)?;
        device
            .memset_zeros(&mut meta_num_nodes)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero smooth num_nodes: {}", e)))?;
        device
            .memset_zeros(&mut meta_num_edges)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero smooth num_edges: {}", e)))?;
        unsafe {
            edge_cap_check.clone().launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &out_child_offsets,
                    smooth_node_cap,
                    smooth_edge_cap,
                    &wrap_counts,
                    &mut meta_num_nodes,
                    &mut meta_num_edges,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("d4_smooth_check_edge_cap failed: {}", e)))?;

        let mut out_node_type = memory.alloc::<u8>(smooth_node_cap as usize)?;
        let mut out_child_indices = memory.alloc::<u32>(smooth_edge_cap as usize)?;
        let mut out_lit = memory.alloc::<i32>(smooth_node_cap as usize)?;
        let mut out_decision_var = memory.alloc::<u32>(smooth_node_cap as usize)?;
        let mut out_decision_child_false = memory.alloc::<u32>(smooth_node_cap as usize)?;
        let mut out_decision_child_true = memory.alloc::<u32>(smooth_node_cap as usize)?;
        let mut out_node_level = memory.alloc::<u32>(smooth_node_cap as usize)?;

        device
            .memset_zeros(&mut out_node_type)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero node_type: {}", e)))?;
        device
            .memset_zeros(&mut out_child_indices)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero child_indices: {}", e)))?;
        device
            .memset_zeros(&mut out_lit)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero lit: {}", e)))?;
        device
            .memset_zeros(&mut out_decision_var)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero decision_var: {}", e)))?;
        device
            .memset_zeros(&mut out_decision_child_false)
            .map_err(|e| {
                XlogError::Kernel(format!("Failed to zero decision_child_false: {}", e))
            })?;
        device
            .memset_zeros(&mut out_decision_child_true)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero decision_child_true: {}", e)))?;
        device
            .memset_zeros(&mut out_node_level)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero node_level: {}", e)))?;

        let init_kernel = device
            .get_func(D4_MODULE, d4_kernels::D4_SMOOTH_INIT_NODES)
            .ok_or_else(|| {
                XlogError::Kernel("d4_smooth_init_nodes kernel not found".to_string())
            })?;
        let init_blocks = ((num_random_vars.max(1)) + block_size - 1) / block_size;
        unsafe {
            init_kernel.clone().launch(
                LaunchConfig {
                    grid_dim: (init_blocks, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    random_var_list,
                    num_random_vars,
                    smooth_node_cap,
                    &mut out_node_type,
                    &mut out_lit,
                    &mut out_decision_var,
                    &mut out_decision_child_false,
                    &mut out_decision_child_true,
                    &mut out_node_level,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("d4_smooth_init_nodes failed: {}", e)))?;

        let num_levels_out = self.num_levels.saturating_mul(2).saturating_add(4);

        let emit_kernel = device
            .get_func(D4_MODULE, d4_kernels::D4_SMOOTH_EMIT_LEVEL)
            .ok_or_else(|| {
                XlogError::Kernel("d4_smooth_emit_level kernel not found".to_string())
            })?;
        for level in 0..num_levels {
            let num_level_nodes: usize = match self.level_offsets_host.as_ref() {
                Some(off) => (off[level + 1].saturating_sub(off[level])) as usize,
                None => self.level_nodes.len(),
            };
            if num_level_nodes == 0 {
                continue;
            }
            let num_blocks = ((num_level_nodes as u32) + block_size - 1) / block_size;
            let level_u32 = level as u32;
            let mut params: Vec<*mut c_void> = vec![
                (&self.node_type).as_kernel_param(),
                (&self.child_offsets).as_kernel_param(),
                (&self.child_indices).as_kernel_param(),
                (&self.lit).as_kernel_param(),
                (&self.decision_var).as_kernel_param(),
                (&self.decision_child_false).as_kernel_param(),
                (&self.decision_child_true).as_kernel_param(),
                (&self.level_nodes).as_kernel_param(),
                (&self.level_offsets).as_kernel_param(),
                level_u32.as_kernel_param(),
                (&support).as_kernel_param(),
                words_per_support.as_kernel_param(),
                (&wrap_prefix_or).as_kernel_param(),
                (&wrap_missing_or).as_kernel_param(),
                (&wrap_prefix_dec).as_kernel_param(),
                (&wrap_missing_dec).as_kernel_param(),
                base_node.as_kernel_param(),
                (&self.meta_num_nodes).as_kernel_param(),
                (&wrap_counts).as_kernel_param(),
                num_random_vars.as_kernel_param(),
                num_levels_out.as_kernel_param(),
                (&mut out_node_type).as_kernel_param(),
                (&out_child_offsets).as_kernel_param(),
                (&mut out_child_indices).as_kernel_param(),
                (&mut out_lit).as_kernel_param(),
                (&mut out_decision_var).as_kernel_param(),
                (&mut out_decision_child_false).as_kernel_param(),
                (&mut out_decision_child_true).as_kernel_param(),
                (&mut out_node_level).as_kernel_param(),
            ];
            unsafe {
                emit_kernel.clone().launch(
                    LaunchConfig {
                        grid_dim: (num_blocks, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut params,
                )
            }
            .map_err(|e| XlogError::Kernel(format!("d4_smooth_emit_level failed: {}", e)))?;
        }

        let mut level_counts = memory.alloc::<u32>(num_levels_out as usize)?;
        let mut level_offsets = memory.alloc::<u32>((num_levels_out as usize) + 1)?;
        let mut level_cursors = memory.alloc::<u32>(num_levels_out as usize)?;
        let mut level_nodes = memory.alloc::<u32>(smooth_node_cap as usize)?;

        device
            .memset_zeros(&mut level_counts)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero level_counts: {}", e)))?;
        device
            .memset_zeros(&mut level_offsets)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero level_offsets: {}", e)))?;
        device
            .memset_zeros(&mut level_cursors)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero level_cursors: {}", e)))?;
        device
            .memset_zeros(&mut level_nodes)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero level_nodes: {}", e)))?;

        let mut compile_needed = memory.alloc::<u32>(1)?;
        device
            .htod_sync_copy_into(&[1u32], &mut compile_needed)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload compile_needed: {}", e)))?;

        let levelize_counts = device
            .get_func(D4_MODULE, d4_kernels::D4_LEVELIZE_COUNTS)
            .ok_or_else(|| XlogError::Kernel("d4_levelize_counts kernel not found".to_string()))?;
        let num_blocks = (smooth_node_cap + block_size - 1) / block_size;
        unsafe {
            levelize_counts.clone().launch(
                LaunchConfig {
                    grid_dim: (num_blocks, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &compile_needed,
                    &out_node_level,
                    &meta_num_nodes,
                    num_levels_out,
                    &mut level_counts,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("d4_levelize_counts failed: {}", e)))?;

        device
            .dtod_copy(
                &level_counts,
                &mut level_offsets.slice_mut(0..num_levels_out as usize),
            )
            .map_err(|e| XlogError::Kernel(format!("Failed to copy level_counts: {}", e)))?;
        let level_scan_len = num_levels_out.checked_add(1).ok_or_else(|| {
            XlogError::Compilation("GPU smoothing: level offset scan overflow".to_string())
        })?;
        exclusive_scan_u32_inplace(provider, &mut level_offsets, level_scan_len)?;

        let levelize_emit = device
            .get_func(D4_MODULE, d4_kernels::D4_LEVELIZE_EMIT)
            .ok_or_else(|| XlogError::Kernel("d4_levelize_emit kernel not found".to_string()))?;
        unsafe {
            levelize_emit.clone().launch(
                LaunchConfig {
                    grid_dim: (num_blocks, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &compile_needed,
                    &out_node_level,
                    &meta_num_nodes,
                    num_levels_out,
                    &level_offsets,
                    &mut level_cursors,
                    &mut level_nodes,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("d4_levelize_emit failed: {}", e)))?;

        // No device synchronize: result is device buffers used by subsequent GPU ops.
        let builder = GpuCircuitBuilder {
            node_type: out_node_type,
            child_offsets: out_child_offsets,
            child_indices: out_child_indices,
            lit: out_lit,
            decision_var: out_decision_var,
            decision_child_false: out_decision_child_false,
            decision_child_true: out_decision_child_true,
        };
        let layout = GpuCircuitLayout {
            num_nodes: smooth_node_cap,
            num_edges: smooth_edge_cap,
            num_levels: num_levels_out,
            level_offsets,
            level_nodes,
            root: base_node + self.root,
            max_var: self.max_var,
            num_nodes_device: Some(meta_num_nodes),
            num_edges_device: Some(meta_num_edges),
        };

        GpuXgcf::from_device(builder, layout, provider)
    }

    pub fn upload(provider: &CudaKernelProvider, circuit: &Xgcf) -> Result<Self> {
        if circuit.roots.len() != 1 {
            return Err(XlogError::Compilation(format!(
                "GPU XGCF eval expects exactly 1 root, got {}",
                circuit.roots.len()
            )));
        }

        let memory = provider.memory().clone();
        let device = provider.device().inner();

        let n = circuit.node_type.len();
        let mut host_node_type: Vec<u8> = Vec::with_capacity(n);
        for &ty in &circuit.node_type {
            host_node_type.push(ty as u8);
        }

        let mut max_var: u32 = 0;
        for (&ty, &lit) in circuit.node_type.iter().zip(circuit.lit.iter()) {
            if ty == XgcfNodeType::Lit && lit != 0 {
                max_var = max_var.max(lit.unsigned_abs());
            }
        }
        for &var in &circuit.decision_var {
            max_var = max_var.max(var);
        }

        let mut d_node_type = memory.alloc::<u8>(n)?;
        device
            .htod_sync_copy_into(&host_node_type, &mut d_node_type)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload circuit node_type: {}", e)))?;

        let mut d_child_offsets = memory.alloc::<u32>(circuit.child_offsets.len())?;
        device
            .htod_sync_copy_into(&circuit.child_offsets, &mut d_child_offsets)
            .map_err(|e| {
                XlogError::Kernel(format!("Failed to upload circuit child_offsets: {}", e))
            })?;

        let mut d_child_indices = memory.alloc::<u32>(circuit.child_indices.len())?;
        device
            .htod_sync_copy_into(&circuit.child_indices, &mut d_child_indices)
            .map_err(|e| {
                XlogError::Kernel(format!("Failed to upload circuit child_indices: {}", e))
            })?;

        let mut d_lit = memory.alloc::<i32>(circuit.lit.len())?;
        device
            .htod_sync_copy_into(&circuit.lit, &mut d_lit)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload circuit lit: {}", e)))?;

        let mut d_decision_var = memory.alloc::<u32>(circuit.decision_var.len())?;
        device
            .htod_sync_copy_into(&circuit.decision_var, &mut d_decision_var)
            .map_err(|e| {
                XlogError::Kernel(format!("Failed to upload circuit decision_var: {}", e))
            })?;

        let mut d_decision_child_false = memory.alloc::<u32>(circuit.decision_child_false.len())?;
        device
            .htod_sync_copy_into(&circuit.decision_child_false, &mut d_decision_child_false)
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "Failed to upload circuit decision_child_false: {}",
                    e
                ))
            })?;

        let mut d_decision_child_true = memory.alloc::<u32>(circuit.decision_child_true.len())?;
        device
            .htod_sync_copy_into(&circuit.decision_child_true, &mut d_decision_child_true)
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "Failed to upload circuit decision_child_true: {}",
                    e
                ))
            })?;

        let mut d_level_nodes = memory.alloc::<u32>(circuit.level_nodes.len())?;
        device
            .htod_sync_copy_into(&circuit.level_nodes, &mut d_level_nodes)
            .map_err(|e| {
                XlogError::Kernel(format!("Failed to upload circuit level_nodes: {}", e))
            })?;

        let mut d_level_offsets = memory.alloc::<u32>(circuit.level_offsets.len())?;
        device
            .htod_sync_copy_into(&circuit.level_offsets, &mut d_level_offsets)
            .map_err(|e| {
                XlogError::Kernel(format!("Failed to upload circuit level_offsets: {}", e))
            })?;

        let weights_len = (max_var as usize) + 1;
        let var_log_true = memory.alloc::<f64>(weights_len)?;
        let var_log_false = memory.alloc::<f64>(weights_len)?;
        let values = memory.alloc::<f64>(n)?;
        let adj = memory.alloc::<f64>(n)?;
        let grad_true = memory.alloc::<f64>(weights_len)?;
        let grad_false = memory.alloc::<f64>(weights_len)?;
        let mut meta_num_nodes = memory.alloc::<u32>(1)?;
        device
            .htod_sync_copy_into(&[n as u32], &mut meta_num_nodes)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload num_nodes meta: {}", e)))?;
        let mut meta_num_edges = memory.alloc::<u32>(1)?;
        device
            .htod_sync_copy_into(&[circuit.child_indices.len() as u32], &mut meta_num_edges)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload num_edges meta: {}", e)))?;

        Ok(Self {
            node_type: d_node_type,
            child_offsets: d_child_offsets,
            child_indices: d_child_indices,
            lit: d_lit,
            decision_var: d_decision_var,
            decision_child_false: d_decision_child_false,
            decision_child_true: d_decision_child_true,
            level_nodes: d_level_nodes,
            level_offsets: d_level_offsets,
            level_offsets_host: Some(circuit.level_offsets.clone()),
            node_cap: n as u32,
            edge_cap: circuit.child_indices.len() as u32,
            num_levels: (circuit.level_offsets.len().saturating_sub(1)) as u32,
            root: circuit.roots[0],
            max_var,
            meta_num_nodes,
            meta_num_edges,
            var_log_true,
            var_log_false,
            values,
            adj,
            grad_true,
            grad_false,
            free_var_mask: None,
        })
    }

    pub fn max_var(&self) -> u32 {
        self.max_var
    }

    /// Root node id of the circuit (XGCF requires exactly one root for evaluation/verification).
    pub fn root(&self) -> u32 {
        self.root
    }

    /// Capacity (upper bound) for XGCF nodes in the circuit buffers.
    pub fn num_nodes(&self) -> usize {
        self.node_cap as usize
    }

    /// Capacity (upper bound) for XGCF edges in the circuit buffers.
    pub fn num_edges(&self) -> usize {
        self.edge_cap as usize
    }

    /// Number of topological levels in the circuit.
    pub fn num_levels(&self) -> u32 {
        self.num_levels
    }

    /// Device-resident actual node count (len = 1).
    pub fn num_nodes_device(&self) -> &TrackedCudaSlice<u32> {
        &self.meta_num_nodes
    }

    /// Device-resident actual edge count (len = 1).
    pub fn num_edges_device(&self) -> &TrackedCudaSlice<u32> {
        &self.meta_num_edges
    }

    /// Device-resident level -> node index mapping (len = num_nodes).
    pub fn level_nodes(&self) -> &TrackedCudaSlice<u32> {
        &self.level_nodes
    }

    /// Device-resident offsets for each level (len = num_levels + 1).
    pub fn level_offsets(&self) -> &TrackedCudaSlice<u32> {
        &self.level_offsets
    }

    /// Device-resident node type tags (see `XgcfNodeType`).
    pub fn node_type(&self) -> &TrackedCudaSlice<u8> {
        &self.node_type
    }

    /// Device-resident CSR child offsets for AND/OR nodes (len = num_nodes + 1).
    pub fn child_offsets(&self) -> &TrackedCudaSlice<u32> {
        &self.child_offsets
    }

    /// Device-resident CSR child indices for AND/OR nodes.
    pub fn child_indices(&self) -> &TrackedCudaSlice<u32> {
        &self.child_indices
    }

    /// Device-resident literals for LIT nodes (signed DIMACS, 1-based var ids).
    pub fn lit(&self) -> &TrackedCudaSlice<i32> {
        &self.lit
    }

    /// Device-resident decision var ids for DECISION nodes (0 for non-decision).
    pub fn decision_var(&self) -> &TrackedCudaSlice<u32> {
        &self.decision_var
    }

    pub fn decision_child_false(&self) -> &TrackedCudaSlice<u32> {
        &self.decision_child_false
    }

    pub fn decision_child_true(&self) -> &TrackedCudaSlice<u32> {
        &self.decision_child_true
    }

    /// Device-resident per-node values buffer (log-space). Written by forward pass.
    pub fn values(&self) -> &TrackedCudaSlice<f64> {
        &self.values
    }

    /// Device-resident gradient buffer for ln(true-weight) per CNF variable.
    pub fn grad_true(&self) -> &TrackedCudaSlice<f64> {
        &self.grad_true
    }

    /// Device-resident gradient buffer for ln(false-weight) per CNF variable.
    pub fn grad_false(&self) -> &TrackedCudaSlice<f64> {
        &self.grad_false
    }

    /// Device-resident log(true-weight) table.
    pub fn var_log_true(&self) -> &TrackedCudaSlice<f64> {
        &self.var_log_true
    }

    /// Device-resident log(false-weight) table.
    pub fn var_log_false(&self) -> &TrackedCudaSlice<f64> {
        &self.var_log_false
    }

    /// Mutable access to device-resident log(true-weight) table.
    pub(crate) fn var_log_true_mut(&mut self) -> &mut TrackedCudaSlice<f64> {
        &mut self.var_log_true
    }

    /// Mutable access to device-resident log(false-weight) table.
    pub(crate) fn var_log_false_mut(&mut self) -> &mut TrackedCudaSlice<f64> {
        &mut self.var_log_false
    }

    /// Mutable access to both log-weight tables (true/false) at once.
    ///
    /// This is useful when passing both slices to a single CUDA kernel launch, avoiding
    /// overlapping mutable borrows of `self`.
    pub fn var_log_weights_mut(
        &mut self,
    ) -> (&mut TrackedCudaSlice<f64>, &mut TrackedCudaSlice<f64>) {
        (&mut self.var_log_true, &mut self.var_log_false)
    }

    /// Attach a device-resident free-variable mask (length = max_var + 1).
    pub fn set_free_var_mask_device(&mut self, mask: TrackedCudaSlice<u8>) -> Result<()> {
        if mask.len() != self.var_log_true.len() {
            return Err(XlogError::Compilation(format!(
                "GPU free-var mask len {} != weights len {}",
                mask.len(),
                self.var_log_true.len()
            )));
        }
        self.free_var_mask = Some(mask);
        Ok(())
    }

    /// Upload a host free-variable mask (length = max_var + 1).
    pub(crate) fn set_free_var_mask_from_host(
        &mut self,
        provider: &CudaKernelProvider,
        mask: &[u8],
    ) -> Result<()> {
        if mask.len() != self.var_log_true.len() {
            return Err(XlogError::Compilation(format!(
                "GPU free-var mask len {} != weights len {}",
                mask.len(),
                self.var_log_true.len()
            )));
        }
        let memory = provider.memory();
        let mut d_mask = memory.alloc::<u8>(mask.len())?;
        provider
            .device()
            .inner()
            .htod_sync_copy_into(mask, &mut d_mask)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload free_var_mask: {}", e)))?;
        self.free_var_mask = Some(d_mask);
        Ok(())
    }

    /// Upload a host weight table into the device-resident `var_log_true/var_log_false` buffers.
    ///
    /// This is intended for one-time initialization of static weights (evidence + non-neural facts).
    /// Neural fast-path updates should overwrite only the relevant subset on GPU.
    pub fn set_base_weights(
        &mut self,
        provider: &CudaKernelProvider,
        var_log_weights: &[(f64, f64)],
    ) -> Result<()> {
        let device = provider.device().inner();

        let weights_len = (self.max_var as usize) + 1;
        if var_log_weights.len() < weights_len {
            return Err(XlogError::Compilation(format!(
                "GPU XGCF weights init expects weight table len >= {}, got {}",
                weights_len,
                var_log_weights.len()
            )));
        }

        let mut host_true: Vec<f64> = Vec::with_capacity(weights_len);
        let mut host_false: Vec<f64> = Vec::with_capacity(weights_len);
        for &(t, f) in &var_log_weights[..weights_len] {
            host_true.push(t);
            host_false.push(f);
        }

        device
            .htod_sync_copy_into(&host_true, &mut self.var_log_true)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload log_true weights: {}", e)))?;
        device
            .htod_sync_copy_into(&host_false, &mut self.var_log_false)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload log_false weights: {}", e)))?;

        Ok(())
    }

    /// Evaluate logZ on the device using the currently loaded weights and write it into `out_log_z`.
    ///
    /// This method performs no device->host transfers.
    pub fn eval_log_wmc_device_inplace(
        &mut self,
        provider: &CudaKernelProvider,
        out_log_z: &mut TrackedCudaSlice<f64>,
    ) -> Result<()> {
        if out_log_z.len() != 1 {
            return Err(XlogError::Compilation(format!(
                "GPU device logZ output len {} != 1",
                out_log_z.len()
            )));
        }

        let device = provider.device().inner();
        let func = device
            .get_func(CIRCUIT_MODULE, circuit_kernels::XGCF_FORWARD_LEVEL)
            .ok_or_else(|| XlogError::Kernel("xgcf_forward_level kernel not found".to_string()))?;

        let block_size: u32 = 256;
        let num_levels: usize = self.num_levels as usize;
        for level in 0..num_levels {
            let num_level_nodes: usize = match self.level_offsets_host.as_ref() {
                Some(off) => (off[level + 1].saturating_sub(off[level])) as usize,
                None => self.level_nodes.len(),
            };
            if num_level_nodes == 0 {
                continue;
            }

            let num_blocks = ((num_level_nodes as u32) + block_size - 1) / block_size;
            let config = LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            let level_u32: u32 = level as u32;

            let mut params: Vec<*mut c_void> = vec![
                (&self.node_type).as_kernel_param(),
                (&self.child_offsets).as_kernel_param(),
                (&self.child_indices).as_kernel_param(),
                (&self.lit).as_kernel_param(),
                (&self.decision_var).as_kernel_param(),
                (&self.decision_child_false).as_kernel_param(),
                (&self.decision_child_true).as_kernel_param(),
                (&self.level_nodes).as_kernel_param(),
                (&self.level_offsets).as_kernel_param(),
                level_u32.as_kernel_param(),
                (&self.var_log_true).as_kernel_param(),
                (&self.var_log_false).as_kernel_param(),
                (&mut self.values).as_kernel_param(),
            ];

            // SAFETY: xgcf_forward_level(...) writes values for the provided level nodes.
            unsafe { func.clone().launch(config, &mut params) }
                .map_err(|e| XlogError::Kernel(format!("xgcf_forward_level failed: {}", e)))?;
        }

        self.apply_free_var_correction(provider, true, false)?;

        let root_idx = self.root as usize;
        let root_view = self.values.slice(root_idx..(root_idx + 1));
        device
            .dtod_copy(&root_view, out_log_z)
            .map_err(|e| XlogError::Kernel(format!("Failed to copy device logZ: {}", e)))?;

        // No device synchronize: callers read back with a synchronous host copy
        // or pass the result to subsequent GPU operations (same-stream ordering).
        Ok(())
    }

    /// Evaluate logZ on the device and write it into `out_log_z` (uploads weights from host).
    pub fn eval_log_wmc_device_into(
        &mut self,
        provider: &CudaKernelProvider,
        var_log_weights: &[(f64, f64)],
        out_log_z: &mut TrackedCudaSlice<f64>,
    ) -> Result<()> {
        self.set_base_weights(provider, var_log_weights)?;
        self.eval_log_wmc_device_inplace(provider, out_log_z)
    }

    /// Evaluate logZ on the device and return a device-resident scalar (uploads weights from host).
    pub fn eval_log_wmc_device(
        &mut self,
        provider: &CudaKernelProvider,
        var_log_weights: &[(f64, f64)],
    ) -> Result<TrackedCudaSlice<f64>> {
        let memory = provider.memory();
        let mut out_log_z = memory.alloc::<f64>(1)?;
        self.eval_log_wmc_device_into(provider, var_log_weights, &mut out_log_z)?;
        Ok(out_log_z)
    }

    fn apply_free_var_correction(
        &mut self,
        provider: &CudaKernelProvider,
        apply_log_z: bool,
        apply_grads: bool,
    ) -> Result<()> {
        let Some(mask) = self.free_var_mask.as_ref() else {
            return Ok(());
        };

        if mask.len() != self.var_log_true.len() {
            return Err(XlogError::Compilation(format!(
                "GPU free-var mask len {} != weights len {}",
                mask.len(),
                self.var_log_true.len()
            )));
        }

        let n = u32::try_from(mask.len())
            .map_err(|_| XlogError::Compilation("GPU free-var mask length overflow".to_string()))?;
        if n == 0 {
            return Ok(());
        }

        let device = provider.device().inner();
        let block_dim = 256u32;
        let grid_dim = (n + block_dim - 1) / block_dim;

        if apply_grads {
            let apply_grad = device
                .get_func(CIRCUIT_MODULE, circuit_kernels::XGCF_FREE_VAR_APPLY_GRAD)
                .ok_or_else(|| {
                    XlogError::Kernel("xgcf_free_var_apply_grad kernel not found".to_string())
                })?;
            unsafe {
                apply_grad.clone().launch(
                    LaunchConfig {
                        grid_dim: (grid_dim, 1, 1),
                        block_dim: (block_dim, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        mask,
                        &self.var_log_true,
                        &self.var_log_false,
                        n,
                        &mut self.grad_true,
                        &mut self.grad_false,
                    ),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("xgcf_free_var_apply_grad failed: {}", e)))?;
        }

        if apply_log_z {
            let reduce_stage = device
                .get_func(CIRCUIT_MODULE, circuit_kernels::XGCF_FREE_VAR_REDUCE_STAGE)
                .ok_or_else(|| {
                    XlogError::Kernel("xgcf_free_var_reduce_stage kernel not found".to_string())
                })?;
            let add_scalar = device
                .get_func(CIRCUIT_MODULE, circuit_kernels::XGCF_ADD_SCALAR)
                .ok_or_else(|| XlogError::Kernel("xgcf_add_scalar kernel not found".to_string()))?;

            let memory = provider.memory();
            let mut buf_a = memory.alloc::<f64>(mask.len())?;
            let mut buf_b = memory.alloc::<f64>(mask.len())?;

            let mut stage_n = n;
            let mut stage0 = true;
            let mut output_is_a = true;
            loop {
                let out_len = (stage_n + 1) / 2;
                let stage_grid = (out_len + block_dim - 1) / block_dim;

                let (in_buf, out_buf): (&TrackedCudaSlice<f64>, &mut TrackedCudaSlice<f64>) =
                    if output_is_a {
                        (&buf_b, &mut buf_a)
                    } else {
                        (&buf_a, &mut buf_b)
                    };
                let mode = if stage0 { 0u32 } else { 1u32 };

                unsafe {
                    reduce_stage.clone().launch(
                        LaunchConfig {
                            grid_dim: (stage_grid, 1, 1),
                            block_dim: (block_dim, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            mask,
                            &self.var_log_true,
                            &self.var_log_false,
                            in_buf,
                            stage_n,
                            mode,
                            out_buf,
                        ),
                    )
                }
                .map_err(|e| {
                    XlogError::Kernel(format!("xgcf_free_var_reduce_stage failed: {}", e))
                })?;

                if out_len == 1 {
                    let result_buf = if output_is_a { &buf_a } else { &buf_b };
                    unsafe {
                        add_scalar.clone().launch(
                            LaunchConfig {
                                grid_dim: (1, 1, 1),
                                block_dim: (1, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (&mut self.values, self.root, result_buf),
                        )
                    }
                    .map_err(|e| XlogError::Kernel(format!("xgcf_add_scalar failed: {}", e)))?;
                    break;
                }

                stage_n = out_len;
                stage0 = false;
                output_is_a = !output_is_a;
            }
        }

        Ok(())
    }

    /// Evaluate the circuit and populate `grad_true/grad_false` on the device (no host reads).
    ///
    /// Preconditions:
    /// - `var_log_true/var_log_false` contain the current weights on device.
    /// - Caller may read back results for testing/debugging, but this API performs no dtoh transfers.
    pub fn eval_grads_inplace(&mut self, provider: &CudaKernelProvider) -> Result<()> {
        let device = provider.device().inner();

        // Forward pass (identical to eval_log_wmc, minus weight upload and root readback).
        let func = device
            .get_func(CIRCUIT_MODULE, circuit_kernels::XGCF_FORWARD_LEVEL)
            .ok_or_else(|| XlogError::Kernel("xgcf_forward_level kernel not found".to_string()))?;

        let block_size: u32 = 256;
        let num_levels: usize = self.num_levels as usize;
        for level in 0..num_levels {
            let num_level_nodes: usize = match self.level_offsets_host.as_ref() {
                Some(off) => (off[level + 1].saturating_sub(off[level])) as usize,
                None => self.level_nodes.len(),
            };
            if num_level_nodes == 0 {
                continue;
            }

            let num_blocks = ((num_level_nodes as u32) + block_size - 1) / block_size;
            let config = LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            let level_u32: u32 = level as u32;

            let mut params: Vec<*mut c_void> = vec![
                (&self.node_type).as_kernel_param(),
                (&self.child_offsets).as_kernel_param(),
                (&self.child_indices).as_kernel_param(),
                (&self.lit).as_kernel_param(),
                (&self.decision_var).as_kernel_param(),
                (&self.decision_child_false).as_kernel_param(),
                (&self.decision_child_true).as_kernel_param(),
                (&self.level_nodes).as_kernel_param(),
                (&self.level_offsets).as_kernel_param(),
                level_u32.as_kernel_param(),
                (&self.var_log_true).as_kernel_param(),
                (&self.var_log_false).as_kernel_param(),
                (&mut self.values).as_kernel_param(),
            ];

            // SAFETY: xgcf_forward_level(...) writes values for the provided level nodes.
            unsafe { func.clone().launch(config, &mut params) }
                .map_err(|e| XlogError::Kernel(format!("xgcf_forward_level failed: {}", e)))?;
        }

        // Backward pass buffers.
        device
            .memset_zeros(&mut self.adj)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero adj buffer: {}", e)))?;
        device
            .memset_zeros(&mut self.grad_true)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero grad_true buffer: {}", e)))?;
        device
            .memset_zeros(&mut self.grad_false)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero grad_false buffer: {}", e)))?;

        // Set root adjoint to 1.0 via GPU kernel (avoid host copy).
        let root_idx = self.root as usize;
        let mut root_adj_view = self.adj.slice_mut(root_idx..(root_idx + 1));
        let fill_const = device
            .get_func(ARITH_MODULE, arith_kernels::ARITH_FILL_CONST_F64)
            .ok_or_else(|| {
                XlogError::Kernel("arith_fill_const_f64 kernel not found".to_string())
            })?;
        unsafe {
            fill_const.clone().launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                (1.0_f64, 1u32, &mut root_adj_view),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("arith_fill_const_f64 failed: {}", e)))?;

        let propagate = device
            .get_func(
                CIRCUIT_MODULE,
                circuit_kernels::XGCF_BACKWARD_LEVEL_PROPAGATE,
            )
            .ok_or_else(|| {
                XlogError::Kernel("xgcf_backward_level_propagate kernel not found".to_string())
            })?;
        let decision_grad = device
            .get_func(
                CIRCUIT_MODULE,
                circuit_kernels::XGCF_BACKWARD_LEVEL_DECISION_GRAD,
            )
            .ok_or_else(|| {
                XlogError::Kernel("xgcf_backward_level_decision_grad kernel not found".to_string())
            })?;
        let lit_grad = device
            .get_func(
                CIRCUIT_MODULE,
                circuit_kernels::XGCF_BACKWARD_LEVEL_LIT_GRAD,
            )
            .ok_or_else(|| {
                XlogError::Kernel("xgcf_backward_level_lit_grad kernel not found".to_string())
            })?;

        let num_levels: usize = self.num_levels as usize;
        for level in (0..num_levels).rev() {
            let num_level_nodes: usize = match self.level_offsets_host.as_ref() {
                Some(off) => (off[level + 1].saturating_sub(off[level])) as usize,
                None => self.level_nodes.len(),
            };
            if num_level_nodes == 0 {
                continue;
            }

            let num_blocks = ((num_level_nodes as u32) + block_size - 1) / block_size;
            let config = LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            let level_u32: u32 = level as u32;

            let mut params: Vec<*mut c_void> = vec![
                (&self.node_type).as_kernel_param(),
                (&self.child_offsets).as_kernel_param(),
                (&self.child_indices).as_kernel_param(),
                (&self.decision_var).as_kernel_param(),
                (&self.decision_child_false).as_kernel_param(),
                (&self.decision_child_true).as_kernel_param(),
                (&self.level_nodes).as_kernel_param(),
                (&self.level_offsets).as_kernel_param(),
                level_u32.as_kernel_param(),
                (&self.var_log_true).as_kernel_param(),
                (&self.var_log_false).as_kernel_param(),
                (&self.values).as_kernel_param(),
                (&mut self.adj).as_kernel_param(),
            ];

            unsafe { propagate.clone().launch(config, &mut params) }.map_err(|e| {
                XlogError::Kernel(format!("xgcf_backward_level_propagate failed: {}", e))
            })?;

            let mut params: Vec<*mut c_void> = vec![
                (&self.node_type).as_kernel_param(),
                (&self.decision_var).as_kernel_param(),
                (&self.decision_child_false).as_kernel_param(),
                (&self.decision_child_true).as_kernel_param(),
                (&self.level_nodes).as_kernel_param(),
                (&self.level_offsets).as_kernel_param(),
                level_u32.as_kernel_param(),
                (&self.var_log_true).as_kernel_param(),
                (&self.var_log_false).as_kernel_param(),
                (&self.values).as_kernel_param(),
                (&self.adj).as_kernel_param(),
                (&mut self.grad_true).as_kernel_param(),
                (&mut self.grad_false).as_kernel_param(),
            ];

            unsafe { decision_grad.clone().launch(config, &mut params) }.map_err(|e| {
                XlogError::Kernel(format!("xgcf_backward_level_decision_grad failed: {}", e))
            })?;

            unsafe {
                lit_grad.clone().launch(
                    config,
                    (
                        &self.node_type,
                        &self.lit,
                        &self.level_nodes,
                        &self.level_offsets,
                        level_u32,
                        &self.adj,
                        &self.grad_true,
                        &self.grad_false,
                    ),
                )
            }
            .map_err(|e| {
                XlogError::Kernel(format!("xgcf_backward_level_lit_grad failed: {}", e))
            })?;
        }

        self.apply_free_var_correction(provider, true, true)?;
        // No device synchronize: callers batch multiple eval/backward calls
        // before syncing at the query boundary.
        Ok(())
    }

    #[cfg(feature = "host-io")]
    pub fn eval_log_wmc(
        &mut self,
        provider: &CudaKernelProvider,
        var_log_weights: &[(f64, f64)],
    ) -> Result<f64> {
        let device = provider.device().inner();
        let mut out_log_z = provider.memory().alloc::<f64>(1)?;
        self.eval_log_wmc_device_into(provider, var_log_weights, &mut out_log_z)?;

        let mut host = [0.0_f64];
        device
            .dtoh_sync_copy_into(&out_log_z, &mut host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read circuit root value: {}", e)))?;
        Ok(host[0])
    }

    #[cfg(feature = "host-io")]
    pub fn eval_log_wmc_and_grads(
        &mut self,
        provider: &CudaKernelProvider,
        var_log_weights: &[(f64, f64)],
    ) -> Result<(f64, Vec<f64>, Vec<f64>)> {
        self.set_base_weights(provider, var_log_weights)?;
        self.eval_grads_inplace(provider)?;

        let device = provider.device().inner();

        let weights_len = (self.max_var as usize) + 1;
        let mut host_grad_true: Vec<f64> = vec![0.0; weights_len];
        let mut host_grad_false: Vec<f64> = vec![0.0; weights_len];

        let root_idx = self.root as usize;
        let root_view = self.values.slice(root_idx..(root_idx + 1));
        let mut log_z = [0.0_f64];
        device
            .dtoh_sync_copy_into(&root_view, &mut log_z)
            .map_err(|e| XlogError::Kernel(format!("Failed to read circuit root value: {}", e)))?;

        device
            .dtoh_sync_copy_into(&self.grad_true, &mut host_grad_true)
            .map_err(|e| XlogError::Kernel(format!("Failed to download grad_true: {}", e)))?;
        device
            .dtoh_sync_copy_into(&self.grad_false, &mut host_grad_false)
            .map_err(|e| XlogError::Kernel(format!("Failed to download grad_false: {}", e)))?;

        Ok((log_z[0], host_grad_true, host_grad_false))
    }
}
