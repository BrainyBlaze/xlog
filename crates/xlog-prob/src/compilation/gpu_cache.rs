//! GPU-resident circuit cache helpers.

use std::sync::Arc;

use cudarc::driver::{DeviceRepr, DeviceSlice, LaunchAsync, LaunchConfig};
use xlog_core::{Result, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::provider::{cache_kernels, CACHE_MODULE};
use xlog_cuda::CudaKernelProvider;
use xlog_solve::GpuCnf;

use crate::gpu::GpuXgcf;

#[derive(Debug, Clone, Copy)]
pub struct GpuCircuitCacheConfig {
    pub num_slots: u32,
    pub table_size: u32,
    pub node_cap: u32,
    pub edge_cap: u32,
    pub level_cap: u32,
    pub var_cap: u32,
}

pub struct GpuCircuitCache {
    provider: Arc<CudaKernelProvider>,
    table_size: u32,
    num_slots: u32,
    node_cap: u32,
    edge_cap: u32,
    level_cap: u32,
    var_cap: u32,
    keys: TrackedCudaSlice<u64>,
    slots: TrackedCudaSlice<u32>,
    state: TrackedCudaSlice<u32>,
    last_used: TrackedCudaSlice<u64>,
    slot_states: TrackedCudaSlice<u32>,
    clock: TrackedCudaSlice<u64>,
    node_type: TrackedCudaSlice<u8>,
    child_offsets: TrackedCudaSlice<u32>,
    child_indices: TrackedCudaSlice<u32>,
    lit: TrackedCudaSlice<i32>,
    decision_var: TrackedCudaSlice<u32>,
    decision_child_false: TrackedCudaSlice<u32>,
    decision_child_true: TrackedCudaSlice<u32>,
    level_nodes: TrackedCudaSlice<u32>,
    level_offsets: TrackedCudaSlice<u32>,
    var_log_true: TrackedCudaSlice<f64>,
    var_log_false: TrackedCudaSlice<f64>,
    values: TrackedCudaSlice<f64>,
    adj: TrackedCudaSlice<f64>,
    grad_true: TrackedCudaSlice<f64>,
    grad_false: TrackedCudaSlice<f64>,
    meta_num_nodes: TrackedCudaSlice<u32>,
    meta_num_levels: TrackedCudaSlice<u32>,
    meta_root: TrackedCudaSlice<u32>,
    meta_max_var: TrackedCudaSlice<u32>,
    always_on: TrackedCudaSlice<u32>,
    zero_f64: TrackedCudaSlice<f64>,
    one_f64: TrackedCudaSlice<f64>,
    free_var_mask: TrackedCudaSlice<u8>,
    has_free_var_mask: bool,
}

pub struct GpuCacheLookup {
    provider: Arc<CudaKernelProvider>,
    slot: TrackedCudaSlice<u32>,
    compile_needed: TrackedCudaSlice<u32>,
}

impl GpuCacheLookup {
    pub fn slot_device(&self) -> &TrackedCudaSlice<u32> {
        &self.slot
    }

    pub fn compile_needed_device(&self) -> &TrackedCudaSlice<u32> {
        &self.compile_needed
    }

    pub fn provider(&self) -> &Arc<CudaKernelProvider> {
        &self.provider
    }

    pub fn into_handle(self) -> GpuCircuitCacheHandle {
        GpuCircuitCacheHandle {
            provider: self.provider,
            slot: self.slot,
            compile_needed: self.compile_needed,
            num_nodes: 0,
            num_levels: 0,
            root: 0,
            max_var: 0,
        }
    }
}

pub struct GpuCircuitCacheHandle {
    provider: Arc<CudaKernelProvider>,
    slot: TrackedCudaSlice<u32>,
    compile_needed: TrackedCudaSlice<u32>,
    num_nodes: u32,
    num_levels: u32,
    root: u32,
    max_var: u32,
}

impl GpuCircuitCacheHandle {
    pub fn slot_device(&self) -> &TrackedCudaSlice<u32> {
        &self.slot
    }

    pub fn compile_needed_device(&self) -> &TrackedCudaSlice<u32> {
        &self.compile_needed
    }

    pub fn provider(&self) -> &Arc<CudaKernelProvider> {
        &self.provider
    }

    pub fn num_nodes(&self) -> u32 {
        self.num_nodes
    }

    pub fn num_levels(&self) -> u32 {
        self.num_levels
    }

    pub fn root(&self) -> u32 {
        self.root
    }

    pub fn max_var(&self) -> u32 {
        self.max_var
    }

    pub fn set_meta(&mut self, num_nodes: u32, num_levels: u32, root: u32, max_var: u32) {
        self.num_nodes = num_nodes;
        self.num_levels = num_levels;
        self.root = root;
        self.max_var = max_var;
    }
}

/// Compute a deterministic CNF hash on the GPU.
///
/// Hash input order matches the cache kernel: num_vars, num_clauses, num_lits,
/// clause_offsets[0..num_clauses], literals[0..num_lits-1].
pub fn hash_cnf_gpu(
    cnf: &GpuCnf,
    provider: &Arc<CudaKernelProvider>,
) -> Result<TrackedCudaSlice<u64>> {
    let memory = provider.memory();
    let mut out_hash = memory.alloc::<u64>(1)?;

    let func = provider
        .device()
        .inner()
        .get_func(CACHE_MODULE, cache_kernels::CACHE_CNF_HASH)
        .ok_or_else(|| XlogError::Kernel("cache_cnf_hash kernel not found".to_string()))?;

    unsafe {
        func.clone().launch(
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                &cnf.num_vars,
                &cnf.num_clauses,
                &cnf.num_lits,
                &cnf.clause_offsets,
                &cnf.literals,
                &mut out_hash,
            ),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("cache_cnf_hash launch failed: {}", e)))?;
    // No device synchronize: hash stays device-resident for lookup kernel; same-stream ordering suffices.
    Ok(out_hash)
}

impl GpuCircuitCache {
    pub fn provider(&self) -> &Arc<CudaKernelProvider> {
        &self.provider
    }

    pub fn var_log_weights_mut(
        &mut self,
    ) -> (&mut TrackedCudaSlice<f64>, &mut TrackedCudaSlice<f64>) {
        (&mut self.var_log_true, &mut self.var_log_false)
    }

    pub fn grad_true(&self) -> &TrackedCudaSlice<f64> {
        &self.grad_true
    }

    pub fn grad_false(&self) -> &TrackedCudaSlice<f64> {
        &self.grad_false
    }

    pub fn values(&self) -> &TrackedCudaSlice<f64> {
        &self.values
    }

    pub fn meta_num_nodes_device(&self) -> &TrackedCudaSlice<u32> {
        &self.meta_num_nodes
    }

    pub fn meta_num_levels_device(&self) -> &TrackedCudaSlice<u32> {
        &self.meta_num_levels
    }

    pub fn meta_root_device(&self) -> &TrackedCudaSlice<u32> {
        &self.meta_root
    }

    pub fn meta_max_var_device(&self) -> &TrackedCudaSlice<u32> {
        &self.meta_max_var
    }

    pub fn num_slots(&self) -> u32 {
        self.num_slots
    }

    pub fn has_free_var_mask(&self) -> bool {
        self.has_free_var_mask
    }

    pub fn var_stride(&self) -> Result<u32> {
        self.var_cap
            .checked_add(1)
            .ok_or_else(|| XlogError::Compilation("GpuCircuitCache var_cap overflow".to_string()))
    }

    pub fn node_stride(&self) -> u32 {
        self.node_cap
    }

    pub fn copy_slot_weights_to_batch(
        &mut self,
        handle: &GpuCircuitCacheHandle,
        out_true_batch: &mut TrackedCudaSlice<f64>,
        out_false_batch: &mut TrackedCudaSlice<f64>,
        batch_size: u32,
    ) -> Result<()> {
        if batch_size == 0 {
            return Ok(());
        }
        let var_stride = self.var_stride()?;
        let expected = (batch_size as usize)
            .checked_mul(var_stride as usize)
            .ok_or_else(|| {
                XlogError::Compilation("GpuCircuitCache batch weight size overflow".to_string())
            })?;
        if out_true_batch.len() != expected || out_false_batch.len() != expected {
            return Err(XlogError::Compilation(format!(
                "GpuCircuitCache batched weight buffers must both have len {}, got {} and {}",
                expected,
                out_true_batch.len(),
                out_false_batch.len()
            )));
        }

        let device = self.provider.device().inner();
        let func = device
            .get_func(
                xlog_cuda::provider::WEIGHTS_MODULE,
                xlog_cuda::provider::weights_kernels::WEIGHTS_COPY_SLOT_TO_BATCH,
            )
            .ok_or_else(|| {
                XlogError::Kernel("weights_copy_slot_to_batch kernel not found".to_string())
            })?;

        let block_dim = 256u32;
        let total = (batch_size as u64)
            .checked_mul(var_stride as u64)
            .ok_or_else(|| {
                XlogError::Compilation("GpuCircuitCache batch copy overflow".to_string())
            })?;
        let grid_dim = if total == 0 {
            0
        } else {
            ((total + (block_dim as u64) - 1) / (block_dim as u64)) as u32
        };
        if grid_dim == 0 {
            return Ok(());
        }

        unsafe {
            func.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_dim, 1, 1),
                    block_dim: (block_dim, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    handle.slot_device(),
                    self.var_cap,
                    &self.var_log_true,
                    &self.var_log_false,
                    out_true_batch,
                    out_false_batch,
                    var_stride,
                    batch_size,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("weights_copy_slot_to_batch failed: {}", e)))?;

        Ok(())
    }

    pub fn eval_grads_inplace_fused_batched(
        &mut self,
        handle: &GpuCircuitCacheHandle,
        var_log_true_batch: &TrackedCudaSlice<f64>,
        var_log_false_batch: &TrackedCudaSlice<f64>,
        values_batch: &mut TrackedCudaSlice<f64>,
        adj_batch: &mut TrackedCudaSlice<f64>,
        grad_true_batch: &mut TrackedCudaSlice<f64>,
        grad_false_batch: &mut TrackedCudaSlice<f64>,
        batch_size: u32,
    ) -> Result<()> {
        if batch_size == 0 {
            return Ok(());
        }
        if self.has_free_var_mask {
            return Err(XlogError::Execution(
                "Batched fused eval currently does not support free-var correction".to_string(),
            ));
        }

        let var_stride = self.var_stride()?;
        let node_stride = self.node_stride();
        let expected_var = (batch_size as usize)
            .checked_mul(var_stride as usize)
            .ok_or_else(|| {
                XlogError::Compilation("GpuCircuitCache batched var buffer overflow".to_string())
            })?;
        let expected_node = (batch_size as usize)
            .checked_mul(node_stride as usize)
            .ok_or_else(|| {
                XlogError::Compilation("GpuCircuitCache batched node buffer overflow".to_string())
            })?;

        if var_log_true_batch.len() != expected_var
            || var_log_false_batch.len() != expected_var
            || grad_true_batch.len() != expected_var
            || grad_false_batch.len() != expected_var
        {
            return Err(XlogError::Compilation(format!(
                "GpuCircuitCache batched var buffers must have len {}",
                expected_var
            )));
        }
        if values_batch.len() != expected_node || adj_batch.len() != expected_node {
            return Err(XlogError::Compilation(format!(
                "GpuCircuitCache batched node buffers must have len {}",
                expected_node
            )));
        }

        let device = self.provider.device().inner();
        device
            .memset_zeros(adj_batch)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero batched adj: {}", e)))?;
        device
            .memset_zeros(grad_true_batch)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero batched grad_true: {}", e)))?;
        device
            .memset_zeros(grad_false_batch)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero batched grad_false: {}", e)))?;

        let eval_all = device
            .get_func(
                xlog_cuda::CIRCUIT_MODULE,
                xlog_cuda::circuit_kernels::XGCF_EVAL_ALL_LEVELS_CACHED_BATCHED,
            )
            .ok_or_else(|| {
                XlogError::Kernel("xgcf_eval_all_levels_cached_batched not found".to_string())
            })?;
        let set_root_adj = device
            .get_func(
                xlog_cuda::CIRCUIT_MODULE,
                xlog_cuda::circuit_kernels::XGCF_SET_ROOT_ADJ_CACHED_BATCHED,
            )
            .ok_or_else(|| {
                XlogError::Kernel("xgcf_set_root_adj_cached_batched not found".to_string())
            })?;
        let backward_all = device
            .get_func(
                xlog_cuda::CIRCUIT_MODULE,
                xlog_cuda::circuit_kernels::XGCF_BACKWARD_ALL_LEVELS_CACHED_BATCHED,
            )
            .ok_or_else(|| {
                XlogError::Kernel("xgcf_backward_all_levels_cached_batched not found".to_string())
            })?;

        let block_size = 256u32;
        let mut eval_params: Vec<*mut std::ffi::c_void> = vec![
            handle.slot_device().as_kernel_param(),
            self.node_cap.as_kernel_param(),
            self.edge_cap.as_kernel_param(),
            self.level_cap.as_kernel_param(),
            self.var_cap.as_kernel_param(),
            (&self.node_type).as_kernel_param(),
            (&self.child_offsets).as_kernel_param(),
            (&self.child_indices).as_kernel_param(),
            (&self.lit).as_kernel_param(),
            (&self.decision_var).as_kernel_param(),
            (&self.decision_child_false).as_kernel_param(),
            (&self.decision_child_true).as_kernel_param(),
            (&self.level_nodes).as_kernel_param(),
            (&self.level_offsets).as_kernel_param(),
            (&self.meta_num_levels).as_kernel_param(),
            var_log_true_batch.as_kernel_param(),
            var_log_false_batch.as_kernel_param(),
            var_stride.as_kernel_param(),
            values_batch.as_kernel_param(),
            node_stride.as_kernel_param(),
            batch_size.as_kernel_param(),
        ];
        unsafe {
            eval_all.clone().launch(
                LaunchConfig {
                    grid_dim: (batch_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                &mut eval_params,
            )
        }
        .map_err(|e| {
            XlogError::Kernel(format!("xgcf_eval_all_levels_cached_batched failed: {}", e))
        })?;

        unsafe {
            set_root_adj.clone().launch(
                LaunchConfig {
                    grid_dim: (batch_size, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    handle.slot_device(),
                    self.node_cap,
                    &self.meta_root,
                    &mut *adj_batch,
                    node_stride,
                    batch_size,
                ),
            )
        }
        .map_err(|e| {
            XlogError::Kernel(format!("xgcf_set_root_adj_cached_batched failed: {}", e))
        })?;

        let mut backward_params: Vec<*mut std::ffi::c_void> = vec![
            handle.slot_device().as_kernel_param(),
            self.node_cap.as_kernel_param(),
            self.edge_cap.as_kernel_param(),
            self.level_cap.as_kernel_param(),
            self.var_cap.as_kernel_param(),
            (&self.node_type).as_kernel_param(),
            (&self.child_offsets).as_kernel_param(),
            (&self.child_indices).as_kernel_param(),
            (&self.decision_var).as_kernel_param(),
            (&self.decision_child_false).as_kernel_param(),
            (&self.decision_child_true).as_kernel_param(),
            (&self.lit).as_kernel_param(),
            (&self.level_nodes).as_kernel_param(),
            (&self.level_offsets).as_kernel_param(),
            (&self.meta_num_levels).as_kernel_param(),
            var_log_true_batch.as_kernel_param(),
            var_log_false_batch.as_kernel_param(),
            var_stride.as_kernel_param(),
            values_batch.as_kernel_param(),
            node_stride.as_kernel_param(),
            adj_batch.as_kernel_param(),
            node_stride.as_kernel_param(),
            grad_true_batch.as_kernel_param(),
            grad_false_batch.as_kernel_param(),
            var_stride.as_kernel_param(),
            batch_size.as_kernel_param(),
        ];
        unsafe {
            backward_all.clone().launch(
                LaunchConfig {
                    grid_dim: (batch_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                &mut backward_params,
            )
        }
        .map_err(|e| {
            XlogError::Kernel(format!(
                "xgcf_backward_all_levels_cached_batched failed: {}",
                e
            ))
        })?;

        Ok(())
    }

    pub fn copy_root_batched_from_values(
        &self,
        handle: &GpuCircuitCacheHandle,
        values_batch: &TrackedCudaSlice<f64>,
        out_roots: &mut TrackedCudaSlice<f64>,
        batch_size: u32,
    ) -> Result<()> {
        if batch_size == 0 {
            return Ok(());
        }
        let node_stride = self.node_stride();
        let expected_values = (batch_size as usize)
            .checked_mul(node_stride as usize)
            .ok_or_else(|| {
                XlogError::Compilation("GpuCircuitCache batched values overflow".to_string())
            })?;
        if values_batch.len() != expected_values || out_roots.len() != batch_size as usize {
            return Err(XlogError::Compilation(format!(
                "GpuCircuitCache root copy expects values len {} and roots len {}, got {} and {}",
                expected_values,
                batch_size,
                values_batch.len(),
                out_roots.len()
            )));
        }

        let device = self.provider.device().inner();
        let copy_root = device
            .get_func(
                xlog_cuda::CIRCUIT_MODULE,
                xlog_cuda::circuit_kernels::XGCF_COPY_ROOT_CACHED_META_BATCHED,
            )
            .ok_or_else(|| {
                XlogError::Kernel("xgcf_copy_root_cached_meta_batched not found".to_string())
            })?;
        unsafe {
            copy_root.clone().launch(
                LaunchConfig {
                    grid_dim: (batch_size, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    handle.slot_device(),
                    self.node_cap,
                    &self.meta_root,
                    values_batch,
                    node_stride,
                    out_roots,
                    batch_size,
                ),
            )
        }
        .map_err(|e| {
            XlogError::Kernel(format!("xgcf_copy_root_cached_meta_batched failed: {}", e))
        })?;
        Ok(())
    }

    pub fn new(provider: &Arc<CudaKernelProvider>, config: GpuCircuitCacheConfig) -> Result<Self> {
        if config.num_slots == 0 {
            return Err(XlogError::Compilation(
                "GpuCircuitCache requires num_slots > 0".to_string(),
            ));
        }
        if config.table_size == 0 {
            return Err(XlogError::Compilation(
                "GpuCircuitCache requires table_size > 0".to_string(),
            ));
        }
        if config.table_size < config.num_slots {
            return Err(XlogError::Compilation(format!(
                "GpuCircuitCache table_size {} < num_slots {}",
                config.table_size, config.num_slots
            )));
        }
        if config.node_cap == 0
            || config.edge_cap == 0
            || config.level_cap == 0
            || config.var_cap == 0
        {
            return Err(XlogError::Compilation(
                "GpuCircuitCache requires non-zero caps".to_string(),
            ));
        }

        let memory = provider.memory();
        let device = provider.device().inner();

        let table_len = usize::try_from(config.table_size).map_err(|_| {
            XlogError::Compilation("GpuCircuitCache table_size overflow".to_string())
        })?;
        let slot_len = usize::try_from(config.num_slots).map_err(|_| {
            XlogError::Compilation("GpuCircuitCache num_slots overflow".to_string())
        })?;

        let node_cap = usize::try_from(config.node_cap)
            .map_err(|_| XlogError::Compilation("GpuCircuitCache node_cap overflow".to_string()))?;
        let edge_cap = usize::try_from(config.edge_cap)
            .map_err(|_| XlogError::Compilation("GpuCircuitCache edge_cap overflow".to_string()))?;
        let level_cap = usize::try_from(config.level_cap).map_err(|_| {
            XlogError::Compilation("GpuCircuitCache level_cap overflow".to_string())
        })?;
        let var_cap = usize::try_from(config.var_cap)
            .map_err(|_| XlogError::Compilation("GpuCircuitCache var_cap overflow".to_string()))?;

        let node_slots = slot_len.checked_mul(node_cap).ok_or_else(|| {
            XlogError::Compilation("GpuCircuitCache node slots overflow".to_string())
        })?;
        let edge_slots = slot_len.checked_mul(edge_cap).ok_or_else(|| {
            XlogError::Compilation("GpuCircuitCache edge slots overflow".to_string())
        })?;
        let var_slots = slot_len.checked_mul(var_cap + 1).ok_or_else(|| {
            XlogError::Compilation("GpuCircuitCache var slots overflow".to_string())
        })?;
        let node_offsets = slot_len.checked_mul(node_cap + 1).ok_or_else(|| {
            XlogError::Compilation("GpuCircuitCache offset slots overflow".to_string())
        })?;
        let level_offsets = slot_len.checked_mul(level_cap + 1).ok_or_else(|| {
            XlogError::Compilation("GpuCircuitCache level offsets overflow".to_string())
        })?;

        let mut keys = memory.alloc::<u64>(table_len)?;
        let mut slots = memory.alloc::<u32>(table_len)?;
        let mut state = memory.alloc::<u32>(table_len)?;
        let mut last_used = memory.alloc::<u64>(table_len)?;
        let mut slot_states = memory.alloc::<u32>(slot_len)?;
        let mut clock = memory.alloc::<u64>(1)?;

        let mut node_type = memory.alloc::<u8>(node_slots)?;
        let mut child_offsets = memory.alloc::<u32>(node_offsets)?;
        let mut child_indices = memory.alloc::<u32>(edge_slots)?;
        let mut lit = memory.alloc::<i32>(node_slots)?;
        let mut decision_var = memory.alloc::<u32>(node_slots)?;
        let mut decision_child_false = memory.alloc::<u32>(node_slots)?;
        let mut decision_child_true = memory.alloc::<u32>(node_slots)?;
        let mut level_nodes = memory.alloc::<u32>(node_slots)?;
        let mut level_offsets = memory.alloc::<u32>(level_offsets)?;

        let mut var_log_true = memory.alloc::<f64>(var_slots)?;
        let mut var_log_false = memory.alloc::<f64>(var_slots)?;
        let mut values = memory.alloc::<f64>(node_slots)?;
        let mut adj = memory.alloc::<f64>(node_slots)?;
        let mut grad_true = memory.alloc::<f64>(var_slots)?;
        let mut grad_false = memory.alloc::<f64>(var_slots)?;
        let mut free_var_mask = memory.alloc::<u8>(var_slots)?;
        let mut meta_num_nodes = memory.alloc::<u32>(slot_len)?;
        let mut meta_num_levels = memory.alloc::<u32>(slot_len)?;
        let mut meta_root = memory.alloc::<u32>(slot_len)?;
        let mut meta_max_var = memory.alloc::<u32>(slot_len)?;
        let mut always_on = memory.alloc::<u32>(1)?;
        let zero_len = node_cap.max(var_cap + 1);
        let mut zero_f64 = memory.alloc::<f64>(zero_len)?;
        let mut one_f64 = memory.alloc::<f64>(1)?;

        device
            .memset_zeros(&mut keys)
            .map_err(|e| XlogError::Kernel(format!("GpuCircuitCache zero keys failed: {}", e)))?;
        device
            .memset_zeros(&mut slots)
            .map_err(|e| XlogError::Kernel(format!("GpuCircuitCache zero slots failed: {}", e)))?;
        device
            .memset_zeros(&mut state)
            .map_err(|e| XlogError::Kernel(format!("GpuCircuitCache zero state failed: {}", e)))?;
        device.memset_zeros(&mut last_used).map_err(|e| {
            XlogError::Kernel(format!("GpuCircuitCache zero last_used failed: {}", e))
        })?;
        device.memset_zeros(&mut slot_states).map_err(|e| {
            XlogError::Kernel(format!("GpuCircuitCache zero slot_states failed: {}", e))
        })?;
        device
            .memset_zeros(&mut clock)
            .map_err(|e| XlogError::Kernel(format!("GpuCircuitCache zero clock failed: {}", e)))?;

        device.memset_zeros(&mut node_type).map_err(|e| {
            XlogError::Kernel(format!("GpuCircuitCache zero node_type failed: {}", e))
        })?;
        device.memset_zeros(&mut child_offsets).map_err(|e| {
            XlogError::Kernel(format!("GpuCircuitCache zero child_offsets failed: {}", e))
        })?;
        device.memset_zeros(&mut child_indices).map_err(|e| {
            XlogError::Kernel(format!("GpuCircuitCache zero child_indices failed: {}", e))
        })?;
        device
            .memset_zeros(&mut lit)
            .map_err(|e| XlogError::Kernel(format!("GpuCircuitCache zero lit failed: {}", e)))?;
        device.memset_zeros(&mut decision_var).map_err(|e| {
            XlogError::Kernel(format!("GpuCircuitCache zero decision_var failed: {}", e))
        })?;
        device
            .memset_zeros(&mut decision_child_false)
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "GpuCircuitCache zero decision_child_false failed: {}",
                    e
                ))
            })?;
        device.memset_zeros(&mut decision_child_true).map_err(|e| {
            XlogError::Kernel(format!(
                "GpuCircuitCache zero decision_child_true failed: {}",
                e
            ))
        })?;
        device.memset_zeros(&mut level_nodes).map_err(|e| {
            XlogError::Kernel(format!("GpuCircuitCache zero level_nodes failed: {}", e))
        })?;
        device.memset_zeros(&mut level_offsets).map_err(|e| {
            XlogError::Kernel(format!("GpuCircuitCache zero level_offsets failed: {}", e))
        })?;
        device.memset_zeros(&mut var_log_true).map_err(|e| {
            XlogError::Kernel(format!("GpuCircuitCache zero var_log_true failed: {}", e))
        })?;
        device.memset_zeros(&mut var_log_false).map_err(|e| {
            XlogError::Kernel(format!("GpuCircuitCache zero var_log_false failed: {}", e))
        })?;
        device
            .memset_zeros(&mut values)
            .map_err(|e| XlogError::Kernel(format!("GpuCircuitCache zero values failed: {}", e)))?;
        device
            .memset_zeros(&mut adj)
            .map_err(|e| XlogError::Kernel(format!("GpuCircuitCache zero adj failed: {}", e)))?;
        device.memset_zeros(&mut grad_true).map_err(|e| {
            XlogError::Kernel(format!("GpuCircuitCache zero grad_true failed: {}", e))
        })?;
        device.memset_zeros(&mut grad_false).map_err(|e| {
            XlogError::Kernel(format!("GpuCircuitCache zero grad_false failed: {}", e))
        })?;
        device.memset_zeros(&mut free_var_mask).map_err(|e| {
            XlogError::Kernel(format!("GpuCircuitCache zero free_var_mask failed: {}", e))
        })?;
        device.memset_zeros(&mut meta_num_nodes).map_err(|e| {
            XlogError::Kernel(format!("GpuCircuitCache zero meta_num_nodes failed: {}", e))
        })?;
        device.memset_zeros(&mut meta_num_levels).map_err(|e| {
            XlogError::Kernel(format!(
                "GpuCircuitCache zero meta_num_levels failed: {}",
                e
            ))
        })?;
        device.memset_zeros(&mut meta_root).map_err(|e| {
            XlogError::Kernel(format!("GpuCircuitCache zero meta_root failed: {}", e))
        })?;
        device.memset_zeros(&mut meta_max_var).map_err(|e| {
            XlogError::Kernel(format!("GpuCircuitCache zero meta_max_var failed: {}", e))
        })?;
        device.memset_zeros(&mut zero_f64).map_err(|e| {
            XlogError::Kernel(format!("GpuCircuitCache zero zero_f64 failed: {}", e))
        })?;
        device
            .htod_sync_copy_into(&[1u32], &mut always_on)
            .map_err(|e| {
                XlogError::Kernel(format!("GpuCircuitCache init always_on failed: {}", e))
            })?;
        device
            .htod_sync_copy_into(&[1.0f64], &mut one_f64)
            .map_err(|e| {
                XlogError::Kernel(format!("GpuCircuitCache init one_f64 failed: {}", e))
            })?;

        Ok(Self {
            provider: provider.clone(),
            table_size: config.table_size,
            num_slots: config.num_slots,
            node_cap: config.node_cap,
            edge_cap: config.edge_cap,
            level_cap: config.level_cap,
            var_cap: config.var_cap,
            keys,
            slots,
            state,
            last_used,
            slot_states,
            clock,
            node_type,
            child_offsets,
            child_indices,
            lit,
            decision_var,
            decision_child_false,
            decision_child_true,
            level_nodes,
            level_offsets,
            var_log_true,
            var_log_false,
            values,
            adj,
            grad_true,
            grad_false,
            meta_num_nodes,
            meta_num_levels,
            meta_root,
            meta_max_var,
            always_on,
            zero_f64,
            one_f64,
            free_var_mask,
            has_free_var_mask: false,
        })
    }

    pub fn lookup_or_insert(&mut self, key: u64) -> Result<GpuCacheLookup> {
        let memory = self.provider.memory();
        let mut key_device = memory.alloc::<u64>(1)?;
        self.provider
            .device()
            .inner()
            .htod_sync_copy_into(&[key], &mut key_device)
            .map_err(|e| XlogError::Kernel(format!("cache upload key failed: {}", e)))?;
        self.lookup_or_insert_device(&key_device)
    }

    pub fn lookup_or_insert_device(
        &mut self,
        key_device: &TrackedCudaSlice<u64>,
    ) -> Result<GpuCacheLookup> {
        let memory = self.provider.memory();
        let mut out_slot = memory.alloc::<u32>(1)?;
        let mut out_compile_needed = memory.alloc::<u32>(1)?;

        let func = self
            .provider
            .device()
            .inner()
            .get_func(CACHE_MODULE, cache_kernels::CACHE_LOOKUP_OR_INSERT)
            .ok_or_else(|| {
                XlogError::Kernel("cache_lookup_or_insert kernel not found".to_string())
            })?;

        unsafe {
            func.clone().launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    key_device,
                    self.table_size,
                    self.num_slots,
                    &mut self.keys,
                    &mut self.slots,
                    &mut self.state,
                    &mut self.last_used,
                    &mut self.slot_states,
                    &mut self.clock,
                    &mut out_slot,
                    &mut out_compile_needed,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("cache_lookup_or_insert failed: {}", e)))?;
        // No device synchronize: slot and compile_needed stay device-resident; same-stream ordering suffices.
        Ok(GpuCacheLookup {
            provider: self.provider.clone(),
            slot: out_slot,
            compile_needed: out_compile_needed,
        })
    }

    pub fn claim_slot(&mut self, key: u64) -> Result<GpuCircuitCacheHandle> {
        let lookup = self.lookup_or_insert(key)?;
        Ok(lookup.into_handle())
    }

    pub fn store_from_xgcf(
        &mut self,
        handle: &mut GpuCircuitCacheHandle,
        xgcf: &GpuXgcf,
    ) -> Result<()> {
        let num_nodes = u32::try_from(xgcf.num_nodes()).map_err(|_| {
            XlogError::Compilation("GpuCircuitCache store: num_nodes overflow".to_string())
        })?;
        if num_nodes == 0 {
            return Err(XlogError::Compilation(
                "GpuCircuitCache store: num_nodes must be > 0".to_string(),
            ));
        }
        if num_nodes > self.node_cap {
            return Err(XlogError::Compilation(format!(
                "GpuCircuitCache store: num_nodes {} exceeds node_cap {}",
                num_nodes, self.node_cap
            )));
        }

        let num_edges = u32::try_from(xgcf.num_edges()).map_err(|_| {
            XlogError::Compilation("GpuCircuitCache store: num_edges overflow".to_string())
        })?;
        if num_edges > self.edge_cap {
            return Err(XlogError::Compilation(format!(
                "GpuCircuitCache store: num_edges {} exceeds edge_cap {}",
                num_edges, self.edge_cap
            )));
        }

        let num_levels = xgcf.num_levels();
        if num_levels == 0 {
            return Err(XlogError::Compilation(
                "GpuCircuitCache store: num_levels must be > 0".to_string(),
            ));
        }
        if num_levels > self.level_cap {
            return Err(XlogError::Compilation(format!(
                "GpuCircuitCache store: num_levels {} exceeds level_cap {}",
                num_levels, self.level_cap
            )));
        }

        let root = xgcf.root();
        if root >= num_nodes {
            return Err(XlogError::Compilation(format!(
                "GpuCircuitCache store: root {} out of bounds (num_nodes={})",
                root, num_nodes
            )));
        }

        let max_var = xgcf.max_var();
        if max_var > self.var_cap {
            return Err(XlogError::Compilation(format!(
                "GpuCircuitCache store: max_var {} exceeds var_cap {}",
                max_var, self.var_cap
            )));
        }

        let expected_child_offsets = (num_nodes as usize) + 1;
        if xgcf.child_offsets().len() < expected_child_offsets {
            return Err(XlogError::Compilation(format!(
                "GpuCircuitCache store: child_offsets len {} < num_nodes+1 {}",
                xgcf.child_offsets().len(),
                expected_child_offsets
            )));
        }
        if xgcf.level_nodes().len() < num_nodes as usize {
            return Err(XlogError::Compilation(format!(
                "GpuCircuitCache store: level_nodes len {} < num_nodes {}",
                xgcf.level_nodes().len(),
                num_nodes
            )));
        }
        let expected_level_offsets = (num_levels as usize) + 1;
        if xgcf.level_offsets().len() != expected_level_offsets {
            return Err(XlogError::Compilation(format!(
                "GpuCircuitCache store: level_offsets len {} != num_levels+1 {}",
                xgcf.level_offsets().len(),
                expected_level_offsets
            )));
        }

        handle.num_nodes = num_nodes;
        handle.num_levels = num_levels;
        handle.root = root;
        handle.max_var = max_var;

        let device = self.provider.device().inner();
        let store_u8 = device
            .get_func(CACHE_MODULE, cache_kernels::CACHE_STORE_U8)
            .ok_or_else(|| XlogError::Kernel("cache_store_u8 kernel not found".to_string()))?;
        let store_u32 = device
            .get_func(CACHE_MODULE, cache_kernels::CACHE_STORE_U32)
            .ok_or_else(|| XlogError::Kernel("cache_store_u32 kernel not found".to_string()))?;
        let store_i32 = device
            .get_func(CACHE_MODULE, cache_kernels::CACHE_STORE_I32)
            .ok_or_else(|| XlogError::Kernel("cache_store_i32 kernel not found".to_string()))?;
        let store_f64 = device
            .get_func(CACHE_MODULE, cache_kernels::CACHE_STORE_F64)
            .ok_or_else(|| XlogError::Kernel("cache_store_f64 kernel not found".to_string()))?;
        let store_meta = device
            .get_func(CACHE_MODULE, cache_kernels::CACHE_STORE_META)
            .ok_or_else(|| XlogError::Kernel("cache_store_meta kernel not found".to_string()))?;

        let block_dim = 256u32;
        let grid_for = |count: u32| -> u32 {
            if count == 0 {
                0
            } else {
                (count + block_dim - 1) / block_dim
            }
        };

        let node_stride = self.node_cap;
        let offset_stride = self.node_cap.checked_add(1).ok_or_else(|| {
            XlogError::Compilation("GpuCircuitCache store: node_cap overflow".to_string())
        })?;
        let level_offset_stride = self.level_cap.checked_add(1).ok_or_else(|| {
            XlogError::Compilation("GpuCircuitCache store: level_cap overflow".to_string())
        })?;
        let var_stride = self.var_cap.checked_add(1).ok_or_else(|| {
            XlogError::Compilation("GpuCircuitCache store: var_cap overflow".to_string())
        })?;

        let num_nodes_plus1 = num_nodes.checked_add(1).ok_or_else(|| {
            XlogError::Compilation("GpuCircuitCache store: num_nodes overflow".to_string())
        })?;
        let num_levels_plus1 = num_levels.checked_add(1).ok_or_else(|| {
            XlogError::Compilation("GpuCircuitCache store: num_levels overflow".to_string())
        })?;
        let weights_len = max_var.checked_add(1).ok_or_else(|| {
            XlogError::Compilation("GpuCircuitCache store: max_var overflow".to_string())
        })?;

        let grid_nodes = grid_for(num_nodes);
        if grid_nodes != 0 {
            unsafe {
                store_u8.clone().launch(
                    LaunchConfig {
                        grid_dim: (grid_nodes, 1, 1),
                        block_dim: (block_dim, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        handle.slot_device(),
                        handle.compile_needed_device(),
                        node_stride,
                        xgcf.node_type(),
                        &mut self.node_type,
                        num_nodes,
                    ),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("cache_store_u8 failed: {}", e)))?;
        }

        let grid_offsets = grid_for(num_nodes_plus1);
        if grid_offsets != 0 {
            unsafe {
                store_u32.clone().launch(
                    LaunchConfig {
                        grid_dim: (grid_offsets, 1, 1),
                        block_dim: (block_dim, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        handle.slot_device(),
                        handle.compile_needed_device(),
                        offset_stride,
                        xgcf.child_offsets(),
                        &mut self.child_offsets,
                        num_nodes_plus1,
                    ),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("cache_store_child_offsets failed: {}", e)))?;
        }

        let grid_edges = grid_for(num_edges);
        if grid_edges != 0 {
            unsafe {
                store_u32.clone().launch(
                    LaunchConfig {
                        grid_dim: (grid_edges, 1, 1),
                        block_dim: (block_dim, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        handle.slot_device(),
                        handle.compile_needed_device(),
                        self.edge_cap,
                        xgcf.child_indices(),
                        &mut self.child_indices,
                        num_edges,
                    ),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("cache_store_child_indices failed: {}", e)))?;
        }

        if grid_nodes != 0 {
            unsafe {
                store_i32.clone().launch(
                    LaunchConfig {
                        grid_dim: (grid_nodes, 1, 1),
                        block_dim: (block_dim, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        handle.slot_device(),
                        handle.compile_needed_device(),
                        node_stride,
                        xgcf.lit(),
                        &mut self.lit,
                        num_nodes,
                    ),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("cache_store_lit failed: {}", e)))?;

            unsafe {
                store_u32.clone().launch(
                    LaunchConfig {
                        grid_dim: (grid_nodes, 1, 1),
                        block_dim: (block_dim, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        handle.slot_device(),
                        handle.compile_needed_device(),
                        node_stride,
                        xgcf.decision_var(),
                        &mut self.decision_var,
                        num_nodes,
                    ),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("cache_store_decision_var failed: {}", e)))?;

            unsafe {
                store_u32.clone().launch(
                    LaunchConfig {
                        grid_dim: (grid_nodes, 1, 1),
                        block_dim: (block_dim, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        handle.slot_device(),
                        handle.compile_needed_device(),
                        node_stride,
                        xgcf.decision_child_false(),
                        &mut self.decision_child_false,
                        num_nodes,
                    ),
                )
            }
            .map_err(|e| {
                XlogError::Kernel(format!("cache_store_decision_child_false failed: {}", e))
            })?;

            unsafe {
                store_u32.clone().launch(
                    LaunchConfig {
                        grid_dim: (grid_nodes, 1, 1),
                        block_dim: (block_dim, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        handle.slot_device(),
                        handle.compile_needed_device(),
                        node_stride,
                        xgcf.decision_child_true(),
                        &mut self.decision_child_true,
                        num_nodes,
                    ),
                )
            }
            .map_err(|e| {
                XlogError::Kernel(format!("cache_store_decision_child_true failed: {}", e))
            })?;

            unsafe {
                store_u32.clone().launch(
                    LaunchConfig {
                        grid_dim: (grid_nodes, 1, 1),
                        block_dim: (block_dim, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        handle.slot_device(),
                        handle.compile_needed_device(),
                        node_stride,
                        xgcf.level_nodes(),
                        &mut self.level_nodes,
                        num_nodes,
                    ),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("cache_store_level_nodes failed: {}", e)))?;
        }

        let grid_levels = grid_for(num_levels_plus1);
        if grid_levels != 0 {
            unsafe {
                store_u32.clone().launch(
                    LaunchConfig {
                        grid_dim: (grid_levels, 1, 1),
                        block_dim: (block_dim, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        handle.slot_device(),
                        handle.compile_needed_device(),
                        level_offset_stride,
                        xgcf.level_offsets(),
                        &mut self.level_offsets,
                        num_levels_plus1,
                    ),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("cache_store_level_offsets failed: {}", e)))?;
        }

        let grid_weights = grid_for(weights_len);
        if grid_weights != 0 {
            unsafe {
                store_f64.clone().launch(
                    LaunchConfig {
                        grid_dim: (grid_weights, 1, 1),
                        block_dim: (block_dim, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        handle.slot_device(),
                        handle.compile_needed_device(),
                        var_stride,
                        xgcf.var_log_true(),
                        &mut self.var_log_true,
                        weights_len,
                    ),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("cache_store_var_log_true failed: {}", e)))?;

            unsafe {
                store_f64.clone().launch(
                    LaunchConfig {
                        grid_dim: (grid_weights, 1, 1),
                        block_dim: (block_dim, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        handle.slot_device(),
                        handle.compile_needed_device(),
                        var_stride,
                        xgcf.var_log_false(),
                        &mut self.var_log_false,
                        weights_len,
                    ),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("cache_store_var_log_false failed: {}", e)))?;
        }

        unsafe {
            store_meta.clone().launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    handle.slot_device(),
                    handle.compile_needed_device(),
                    self.num_slots,
                    num_nodes,
                    num_levels,
                    root,
                    max_var,
                    &mut self.meta_num_nodes,
                    &mut self.meta_num_levels,
                    &mut self.meta_root,
                    &mut self.meta_max_var,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("cache_store_meta failed: {}", e)))?;

        // No device synchronize needed: all stores are GPU-to-GPU on the same stream.
        // Same-stream ordering guarantees subsequent kernels see the stored data.
        Ok(())
    }

    pub fn store_weights(
        &mut self,
        handle: &GpuCircuitCacheHandle,
        weights_true: &TrackedCudaSlice<f64>,
        weights_false: &TrackedCudaSlice<f64>,
    ) -> Result<()> {
        let weights_len = handle.max_var.checked_add(1).ok_or_else(|| {
            XlogError::Compilation("GpuCircuitCache store_weights max_var overflow".to_string())
        })?;
        let weights_len_usize = usize::try_from(weights_len).map_err(|_| {
            XlogError::Compilation("GpuCircuitCache store_weights len overflow".to_string())
        })?;
        if weights_true.len() < weights_len_usize || weights_false.len() < weights_len_usize {
            return Err(XlogError::Compilation(format!(
                "GpuCircuitCache store_weights requires weights len >= {}, got true={} false={}",
                weights_len,
                weights_true.len(),
                weights_false.len()
            )));
        }

        let device = self.provider.device().inner();
        let store_f64 = device
            .get_func(CACHE_MODULE, cache_kernels::CACHE_STORE_F64)
            .ok_or_else(|| XlogError::Kernel("cache_store_f64 kernel not found".to_string()))?;

        let block_dim = 256u32;
        let grid_dim = if weights_len == 0 {
            0
        } else {
            (weights_len + block_dim - 1) / block_dim
        };
        if grid_dim != 0 {
            let var_stride = self.var_cap.checked_add(1).ok_or_else(|| {
                XlogError::Compilation("GpuCircuitCache store_weights var_cap overflow".to_string())
            })?;
            unsafe {
                store_f64.clone().launch(
                    LaunchConfig {
                        grid_dim: (grid_dim, 1, 1),
                        block_dim: (block_dim, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        handle.slot_device(),
                        handle.compile_needed_device(),
                        var_stride,
                        weights_true,
                        &mut self.var_log_true,
                        weights_len,
                    ),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("cache_store_weights_true failed: {}", e)))?;

            unsafe {
                store_f64.clone().launch(
                    LaunchConfig {
                        grid_dim: (grid_dim, 1, 1),
                        block_dim: (block_dim, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        handle.slot_device(),
                        handle.compile_needed_device(),
                        var_stride,
                        weights_false,
                        &mut self.var_log_false,
                        weights_len,
                    ),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("cache_store_weights_false failed: {}", e)))?;
        }

        // No device synchronize: same-stream ordering guarantees visibility.
        Ok(())
    }

    pub fn overwrite_weights(
        &mut self,
        handle: &GpuCircuitCacheHandle,
        weights_true: &TrackedCudaSlice<f64>,
        weights_false: &TrackedCudaSlice<f64>,
    ) -> Result<()> {
        let weights_len = handle.max_var.checked_add(1).ok_or_else(|| {
            XlogError::Compilation("GpuCircuitCache overwrite_weights max_var overflow".to_string())
        })?;
        let weights_len_usize = usize::try_from(weights_len).map_err(|_| {
            XlogError::Compilation("GpuCircuitCache overwrite_weights len overflow".to_string())
        })?;
        if weights_true.len() < weights_len_usize || weights_false.len() < weights_len_usize {
            return Err(XlogError::Compilation(format!(
                "GpuCircuitCache overwrite_weights requires weights len >= {}, got true={} false={}",
                weights_len,
                weights_true.len(),
                weights_false.len()
            )));
        }

        let device = self.provider.device().inner();
        let store_f64 = device
            .get_func(CACHE_MODULE, cache_kernels::CACHE_STORE_F64)
            .ok_or_else(|| XlogError::Kernel("cache_store_f64 kernel not found".to_string()))?;

        let block_dim = 256u32;
        let grid_dim = if weights_len == 0 {
            0
        } else {
            (weights_len + block_dim - 1) / block_dim
        };
        if grid_dim != 0 {
            let var_stride = self.var_cap.checked_add(1).ok_or_else(|| {
                XlogError::Compilation(
                    "GpuCircuitCache overwrite_weights var_cap overflow".to_string(),
                )
            })?;
            unsafe {
                store_f64.clone().launch(
                    LaunchConfig {
                        grid_dim: (grid_dim, 1, 1),
                        block_dim: (block_dim, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        handle.slot_device(),
                        &self.always_on,
                        var_stride,
                        weights_true,
                        &mut self.var_log_true,
                        weights_len,
                    ),
                )
            }
            .map_err(|e| {
                XlogError::Kernel(format!("cache_overwrite_weights_true failed: {}", e))
            })?;

            unsafe {
                store_f64.clone().launch(
                    LaunchConfig {
                        grid_dim: (grid_dim, 1, 1),
                        block_dim: (block_dim, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        handle.slot_device(),
                        &self.always_on,
                        var_stride,
                        weights_false,
                        &mut self.var_log_false,
                        weights_len,
                    ),
                )
            }
            .map_err(|e| {
                XlogError::Kernel(format!("cache_overwrite_weights_false failed: {}", e))
            })?;
        }

        // No device synchronize: same-stream ordering guarantees visibility.
        Ok(())
    }

    pub fn store_free_var_mask(
        &mut self,
        handle: &GpuCircuitCacheHandle,
        mask: &TrackedCudaSlice<u8>,
    ) -> Result<()> {
        let mask_len = u32::try_from(mask.len()).map_err(|_| {
            XlogError::Compilation("GpuCircuitCache free_var_mask len overflow".to_string())
        })?;
        let expected_len = handle.max_var.checked_add(1).ok_or_else(|| {
            XlogError::Compilation("GpuCircuitCache free_var_mask max_var overflow".to_string())
        })?;
        if mask_len != expected_len {
            return Err(XlogError::Compilation(format!(
                "GpuCircuitCache free_var_mask len {} != expected {}",
                mask_len, expected_len
            )));
        }
        let var_stride = self.var_cap.checked_add(1).ok_or_else(|| {
            XlogError::Compilation("GpuCircuitCache free_var_mask var_cap overflow".to_string())
        })?;
        if expected_len > var_stride {
            return Err(XlogError::Compilation(format!(
                "GpuCircuitCache free_var_mask len {} exceeds var_cap+1 {}",
                expected_len, var_stride
            )));
        }

        let device = self.provider.device().inner();
        let store_u8 = device
            .get_func(CACHE_MODULE, cache_kernels::CACHE_STORE_U8)
            .ok_or_else(|| XlogError::Kernel("cache_store_u8 kernel not found".to_string()))?;

        let block_dim = 256u32;
        let grid_dim = (mask_len + block_dim - 1) / block_dim;
        if grid_dim == 0 {
            return Ok(());
        }

        unsafe {
            store_u8.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_dim, 1, 1),
                    block_dim: (block_dim, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    handle.slot_device(),
                    handle.compile_needed_device(),
                    var_stride,
                    mask,
                    &mut self.free_var_mask,
                    mask_len,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("cache_store_free_var_mask failed: {}", e)))?;

        // No device synchronize: same-stream ordering guarantees visibility.
        self.has_free_var_mask = true;
        Ok(())
    }

    pub fn eval_log_wmc_device_inplace(
        &mut self,
        handle: &GpuCircuitCacheHandle,
        out_log_z: &mut TrackedCudaSlice<f64>,
    ) -> Result<()> {
        self.eval_log_wmc_device_only(handle, out_log_z)
    }

    pub fn eval_log_wmc_device_only(
        &mut self,
        handle: &GpuCircuitCacheHandle,
        out_log_z: &mut TrackedCudaSlice<f64>,
    ) -> Result<()> {
        if out_log_z.len() != 1 {
            return Err(XlogError::Compilation(format!(
                "GPU cache logZ output len {} != 1",
                out_log_z.len()
            )));
        }

        {
            let device = self.provider.device().inner();
            let eval_all = device
                .get_func(
                    xlog_cuda::CIRCUIT_MODULE,
                    xlog_cuda::circuit_kernels::XGCF_EVAL_ALL_LEVELS_CACHED,
                )
                .ok_or_else(|| {
                    XlogError::Kernel("xgcf_eval_all_levels_cached kernel not found".to_string())
                })?;

            let block_size: u32 = 256;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                handle.slot_device().as_kernel_param(),
                self.node_cap.as_kernel_param(),
                self.edge_cap.as_kernel_param(),
                self.level_cap.as_kernel_param(),
                self.var_cap.as_kernel_param(),
                (&self.node_type).as_kernel_param(),
                (&self.child_offsets).as_kernel_param(),
                (&self.child_indices).as_kernel_param(),
                (&self.lit).as_kernel_param(),
                (&self.decision_var).as_kernel_param(),
                (&self.decision_child_false).as_kernel_param(),
                (&self.decision_child_true).as_kernel_param(),
                (&self.level_nodes).as_kernel_param(),
                (&self.level_offsets).as_kernel_param(),
                (&self.var_log_true).as_kernel_param(),
                (&self.var_log_false).as_kernel_param(),
                (&mut self.values).as_kernel_param(),
                (&self.meta_num_levels).as_kernel_param(),
            ];
            unsafe {
                eval_all.clone().launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut params,
                )
            }
            .map_err(|e| XlogError::Kernel(format!("xgcf_eval_all_levels_cached failed: {}", e)))?;
        }

        self.apply_free_var_correction_cached(handle, true, false)?;

        let device = self.provider.device().inner();
        let copy_root = device
            .get_func(
                xlog_cuda::CIRCUIT_MODULE,
                xlog_cuda::circuit_kernels::XGCF_COPY_ROOT_CACHED_META,
            )
            .ok_or_else(|| {
                XlogError::Kernel("xgcf_copy_root_cached_meta kernel not found".to_string())
            })?;
        unsafe {
            copy_root.clone().launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    handle.slot_device(),
                    self.node_cap,
                    &self.values,
                    &self.meta_root,
                    out_log_z,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("xgcf_copy_root_cached_meta failed: {}", e)))?;

        // No device synchronize: callers read back with a synchronous host copy
        // or pass the result to subsequent GPU operations (same-stream ordering).
        Ok(())
    }

    pub fn eval_grads_inplace(&mut self, handle: &GpuCircuitCacheHandle) -> Result<()> {
        let device = self.provider.device().inner();
        let eval_all = device
            .get_func(
                xlog_cuda::CIRCUIT_MODULE,
                xlog_cuda::circuit_kernels::XGCF_EVAL_ALL_LEVELS_CACHED,
            )
            .ok_or_else(|| {
                XlogError::Kernel("xgcf_eval_all_levels_cached kernel not found".to_string())
            })?;
        let block_size: u32 = 256;
        let mut params: Vec<*mut std::ffi::c_void> = vec![
            handle.slot_device().as_kernel_param(),
            self.node_cap.as_kernel_param(),
            self.edge_cap.as_kernel_param(),
            self.level_cap.as_kernel_param(),
            self.var_cap.as_kernel_param(),
            (&self.node_type).as_kernel_param(),
            (&self.child_offsets).as_kernel_param(),
            (&self.child_indices).as_kernel_param(),
            (&self.lit).as_kernel_param(),
            (&self.decision_var).as_kernel_param(),
            (&self.decision_child_false).as_kernel_param(),
            (&self.decision_child_true).as_kernel_param(),
            (&self.level_nodes).as_kernel_param(),
            (&self.level_offsets).as_kernel_param(),
            (&self.var_log_true).as_kernel_param(),
            (&self.var_log_false).as_kernel_param(),
            (&mut self.values).as_kernel_param(),
            (&self.meta_num_levels).as_kernel_param(),
        ];
        unsafe {
            eval_all.clone().launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                &mut params,
            )
        }
        .map_err(|e| XlogError::Kernel(format!("xgcf_eval_all_levels_cached failed: {}", e)))?;

        let device = self.provider.device().inner();
        let store_f64 = device
            .get_func(CACHE_MODULE, cache_kernels::CACHE_STORE_F64)
            .ok_or_else(|| XlogError::Kernel("cache_store_f64 kernel not found".to_string()))?;

        let grid_for = |count: u32| -> u32 {
            if count == 0 {
                0
            } else {
                (count + block_size - 1) / block_size
            }
        };
        let node_stride = self.node_cap;
        let var_stride = self.var_cap.checked_add(1).ok_or_else(|| {
            XlogError::Compilation("GPU cache eval_grads var_cap overflow".to_string())
        })?;
        let weights_len = self.var_cap.checked_add(1).ok_or_else(|| {
            XlogError::Compilation("GPU cache eval_grads var_cap overflow".to_string())
        })?;

        let grid_nodes = grid_for(self.node_cap);
        if grid_nodes != 0 {
            unsafe {
                store_f64.clone().launch(
                    LaunchConfig {
                        grid_dim: (grid_nodes, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        handle.slot_device(),
                        &self.always_on,
                        node_stride,
                        &self.zero_f64,
                        &mut self.adj,
                        self.node_cap,
                    ),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("cache zero adj failed: {}", e)))?;
        }

        let grid_weights = grid_for(weights_len);
        if grid_weights != 0 {
            unsafe {
                store_f64.clone().launch(
                    LaunchConfig {
                        grid_dim: (grid_weights, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        handle.slot_device(),
                        &self.always_on,
                        var_stride,
                        &self.zero_f64,
                        &mut self.grad_true,
                        weights_len,
                    ),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("cache zero grad_true failed: {}", e)))?;

            unsafe {
                store_f64.clone().launch(
                    LaunchConfig {
                        grid_dim: (grid_weights, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        handle.slot_device(),
                        &self.always_on,
                        var_stride,
                        &self.zero_f64,
                        &mut self.grad_false,
                        weights_len,
                    ),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("cache zero grad_false failed: {}", e)))?;
        }

        let add_scalar = device
            .get_func(
                xlog_cuda::CIRCUIT_MODULE,
                xlog_cuda::circuit_kernels::XGCF_ADD_SCALAR_CACHED,
            )
            .ok_or_else(|| {
                XlogError::Kernel("xgcf_add_scalar_cached kernel not found".to_string())
            })?;
        unsafe {
            add_scalar.clone().launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    handle.slot_device(),
                    self.node_cap,
                    &mut self.adj,
                    &self.meta_root,
                    &self.one_f64,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("xgcf_add_scalar_cached (adj) failed: {}", e)))?;

        let propagate = device
            .get_func(
                xlog_cuda::CIRCUIT_MODULE,
                xlog_cuda::circuit_kernels::XGCF_BACKWARD_LEVEL_PROPAGATE_CACHED,
            )
            .ok_or_else(|| {
                XlogError::Kernel(
                    "xgcf_backward_level_propagate_cached kernel not found".to_string(),
                )
            })?;
        let decision_grad = device
            .get_func(
                xlog_cuda::CIRCUIT_MODULE,
                xlog_cuda::circuit_kernels::XGCF_BACKWARD_LEVEL_DECISION_GRAD_CACHED,
            )
            .ok_or_else(|| {
                XlogError::Kernel(
                    "xgcf_backward_level_decision_grad_cached kernel not found".to_string(),
                )
            })?;
        let lit_grad = device
            .get_func(
                xlog_cuda::CIRCUIT_MODULE,
                xlog_cuda::circuit_kernels::XGCF_BACKWARD_LEVEL_LIT_GRAD_CACHED,
            )
            .ok_or_else(|| {
                XlogError::Kernel(
                    "xgcf_backward_level_lit_grad_cached kernel not found".to_string(),
                )
            })?;

        let num_blocks = (self.node_cap + block_size - 1) / block_size;
        let num_levels = self.level_cap;
        for level in (0..num_levels).rev() {
            if num_blocks == 0 {
                continue;
            }
            let level_u32: u32 = level;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                handle.slot_device().as_kernel_param(),
                self.node_cap.as_kernel_param(),
                self.edge_cap.as_kernel_param(),
                self.level_cap.as_kernel_param(),
                self.var_cap.as_kernel_param(),
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
                (&self.meta_num_levels).as_kernel_param(),
            ];

            unsafe {
                propagate.clone().launch(
                    LaunchConfig {
                        grid_dim: (num_blocks, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut params,
                )
            }
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "xgcf_backward_level_propagate_cached failed: {}",
                    e
                ))
            })?;

            let mut params: Vec<*mut std::ffi::c_void> = vec![
                handle.slot_device().as_kernel_param(),
                self.node_cap.as_kernel_param(),
                self.edge_cap.as_kernel_param(),
                self.level_cap.as_kernel_param(),
                self.var_cap.as_kernel_param(),
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
                (&self.meta_num_levels).as_kernel_param(),
            ];

            unsafe {
                decision_grad.clone().launch(
                    LaunchConfig {
                        grid_dim: (num_blocks, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut params,
                )
            }
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "xgcf_backward_level_decision_grad_cached failed: {}",
                    e
                ))
            })?;

            let mut params: Vec<*mut std::ffi::c_void> = vec![
                handle.slot_device().as_kernel_param(),
                self.node_cap.as_kernel_param(),
                self.edge_cap.as_kernel_param(),
                self.level_cap.as_kernel_param(),
                self.var_cap.as_kernel_param(),
                (&self.node_type).as_kernel_param(),
                (&self.lit).as_kernel_param(),
                (&self.level_nodes).as_kernel_param(),
                (&self.level_offsets).as_kernel_param(),
                level_u32.as_kernel_param(),
                (&self.adj).as_kernel_param(),
                (&self.grad_true).as_kernel_param(),
                (&self.grad_false).as_kernel_param(),
                (&self.meta_num_levels).as_kernel_param(),
            ];

            unsafe {
                lit_grad.clone().launch(
                    LaunchConfig {
                        grid_dim: (num_blocks, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut params,
                )
            }
            .map_err(|e| {
                XlogError::Kernel(format!("xgcf_backward_level_lit_grad_cached failed: {}", e))
            })?;
        }

        self.apply_free_var_correction_cached(handle, true, true)?;
        // No device synchronize: callers batch multiple eval/backward calls
        // before syncing at the query boundary.
        Ok(())
    }

    /// Like [`eval_grads_inplace`] but replaces the per-level backward loop
    /// with a single launch of `xgcf_backward_all_levels_cached`, and omits the
    /// trailing `device().synchronize()` so that the caller can batch multiple
    /// queries before syncing.
    pub fn eval_grads_inplace_fused(&mut self, handle: &GpuCircuitCacheHandle) -> Result<()> {
        let device = self.provider.device().inner();
        let eval_all = device
            .get_func(
                xlog_cuda::CIRCUIT_MODULE,
                xlog_cuda::circuit_kernels::XGCF_EVAL_ALL_LEVELS_CACHED,
            )
            .ok_or_else(|| {
                XlogError::Kernel("xgcf_eval_all_levels_cached kernel not found".to_string())
            })?;
        let block_size: u32 = 256;
        let mut params: Vec<*mut std::ffi::c_void> = vec![
            handle.slot_device().as_kernel_param(),
            self.node_cap.as_kernel_param(),
            self.edge_cap.as_kernel_param(),
            self.level_cap.as_kernel_param(),
            self.var_cap.as_kernel_param(),
            (&self.node_type).as_kernel_param(),
            (&self.child_offsets).as_kernel_param(),
            (&self.child_indices).as_kernel_param(),
            (&self.lit).as_kernel_param(),
            (&self.decision_var).as_kernel_param(),
            (&self.decision_child_false).as_kernel_param(),
            (&self.decision_child_true).as_kernel_param(),
            (&self.level_nodes).as_kernel_param(),
            (&self.level_offsets).as_kernel_param(),
            (&self.var_log_true).as_kernel_param(),
            (&self.var_log_false).as_kernel_param(),
            (&mut self.values).as_kernel_param(),
            (&self.meta_num_levels).as_kernel_param(),
        ];
        unsafe {
            eval_all.clone().launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                &mut params,
            )
        }
        .map_err(|e| XlogError::Kernel(format!("xgcf_eval_all_levels_cached failed: {}", e)))?;

        let device = self.provider.device().inner();
        let store_f64 = device
            .get_func(CACHE_MODULE, cache_kernels::CACHE_STORE_F64)
            .ok_or_else(|| XlogError::Kernel("cache_store_f64 kernel not found".to_string()))?;

        let grid_for = |count: u32| -> u32 {
            if count == 0 {
                0
            } else {
                (count + block_size - 1) / block_size
            }
        };
        let node_stride = self.node_cap;
        let var_stride = self.var_cap.checked_add(1).ok_or_else(|| {
            XlogError::Compilation("GPU cache eval_grads var_cap overflow".to_string())
        })?;
        let weights_len = self.var_cap.checked_add(1).ok_or_else(|| {
            XlogError::Compilation("GPU cache eval_grads var_cap overflow".to_string())
        })?;

        let grid_nodes = grid_for(self.node_cap);
        if grid_nodes != 0 {
            unsafe {
                store_f64.clone().launch(
                    LaunchConfig {
                        grid_dim: (grid_nodes, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        handle.slot_device(),
                        &self.always_on,
                        node_stride,
                        &self.zero_f64,
                        &mut self.adj,
                        self.node_cap,
                    ),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("cache zero adj failed: {}", e)))?;
        }

        let grid_weights = grid_for(weights_len);
        if grid_weights != 0 {
            unsafe {
                store_f64.clone().launch(
                    LaunchConfig {
                        grid_dim: (grid_weights, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        handle.slot_device(),
                        &self.always_on,
                        var_stride,
                        &self.zero_f64,
                        &mut self.grad_true,
                        weights_len,
                    ),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("cache zero grad_true failed: {}", e)))?;

            unsafe {
                store_f64.clone().launch(
                    LaunchConfig {
                        grid_dim: (grid_weights, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        handle.slot_device(),
                        &self.always_on,
                        var_stride,
                        &self.zero_f64,
                        &mut self.grad_false,
                        weights_len,
                    ),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("cache zero grad_false failed: {}", e)))?;
        }

        let add_scalar = device
            .get_func(
                xlog_cuda::CIRCUIT_MODULE,
                xlog_cuda::circuit_kernels::XGCF_ADD_SCALAR_CACHED,
            )
            .ok_or_else(|| {
                XlogError::Kernel("xgcf_add_scalar_cached kernel not found".to_string())
            })?;
        unsafe {
            add_scalar.clone().launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    handle.slot_device(),
                    self.node_cap,
                    &mut self.adj,
                    &self.meta_root,
                    &self.one_f64,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("xgcf_add_scalar_cached (adj) failed: {}", e)))?;

        // Fused backward: single kernel replaces the per-level loop.
        let backward_all = device
            .get_func(
                xlog_cuda::CIRCUIT_MODULE,
                xlog_cuda::circuit_kernels::XGCF_BACKWARD_ALL_LEVELS_CACHED,
            )
            .ok_or_else(|| XlogError::Kernel("xgcf_backward_all_levels_cached not found".into()))?;

        let mut params: Vec<*mut std::ffi::c_void> = vec![
            handle.slot_device().as_kernel_param(),
            self.node_cap.as_kernel_param(),
            self.edge_cap.as_kernel_param(),
            self.level_cap.as_kernel_param(),
            self.var_cap.as_kernel_param(),
            (&self.node_type).as_kernel_param(),
            (&self.child_offsets).as_kernel_param(),
            (&self.child_indices).as_kernel_param(),
            (&self.decision_var).as_kernel_param(),
            (&self.decision_child_false).as_kernel_param(),
            (&self.decision_child_true).as_kernel_param(),
            (&self.lit).as_kernel_param(),
            (&self.level_nodes).as_kernel_param(),
            (&self.level_offsets).as_kernel_param(),
            (&self.var_log_true).as_kernel_param(),
            (&self.var_log_false).as_kernel_param(),
            (&self.values).as_kernel_param(),
            (&mut self.adj).as_kernel_param(),
            (&mut self.grad_true).as_kernel_param(),
            (&mut self.grad_false).as_kernel_param(),
            (&self.meta_num_levels).as_kernel_param(),
        ];

        unsafe {
            backward_all.clone().launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                &mut params,
            )
        }
        .map_err(|e| XlogError::Kernel(format!("xgcf_backward_all_levels_cached failed: {}", e)))?;

        self.apply_free_var_correction_cached(handle, true, true)?;
        Ok(())
    }

    fn apply_free_var_correction_cached(
        &mut self,
        handle: &GpuCircuitCacheHandle,
        apply_log_z: bool,
        apply_grads: bool,
    ) -> Result<()> {
        if !self.has_free_var_mask {
            return Ok(());
        }
        let n = self
            .var_cap
            .checked_add(1)
            .ok_or_else(|| XlogError::Compilation("GPU cache free-var overflow".to_string()))?;
        if n == 0 {
            return Ok(());
        }

        let device = self.provider.device().inner();
        let block_dim = 256u32;
        let grid_dim = (n + block_dim - 1) / block_dim;

        if apply_grads {
            let apply_grad = device
                .get_func(
                    xlog_cuda::CIRCUIT_MODULE,
                    xlog_cuda::circuit_kernels::XGCF_FREE_VAR_APPLY_GRAD_CACHED,
                )
                .ok_or_else(|| {
                    XlogError::Kernel(
                        "xgcf_free_var_apply_grad_cached kernel not found".to_string(),
                    )
                })?;
            unsafe {
                apply_grad.clone().launch(
                    LaunchConfig {
                        grid_dim: (grid_dim, 1, 1),
                        block_dim: (block_dim, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        handle.slot_device(),
                        self.var_cap,
                        &self.free_var_mask,
                        &self.var_log_true,
                        &self.var_log_false,
                        n,
                        &mut self.grad_true,
                        &mut self.grad_false,
                    ),
                )
            }
            .map_err(|e| {
                XlogError::Kernel(format!("xgcf_free_var_apply_grad_cached failed: {}", e))
            })?;
        }

        if apply_log_z {
            let reduce_stage = device
                .get_func(
                    xlog_cuda::CIRCUIT_MODULE,
                    xlog_cuda::circuit_kernels::XGCF_FREE_VAR_REDUCE_STAGE_CACHED,
                )
                .ok_or_else(|| {
                    XlogError::Kernel(
                        "xgcf_free_var_reduce_stage_cached kernel not found".to_string(),
                    )
                })?;
            let add_scalar = device
                .get_func(
                    xlog_cuda::CIRCUIT_MODULE,
                    xlog_cuda::circuit_kernels::XGCF_ADD_SCALAR_CACHED,
                )
                .ok_or_else(|| {
                    XlogError::Kernel("xgcf_add_scalar_cached kernel not found".to_string())
                })?;

            let memory = self.provider.memory();
            let mut buf_a = memory.alloc::<f64>(n as usize)?;
            let mut buf_b = memory.alloc::<f64>(n as usize)?;

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
                            handle.slot_device(),
                            self.var_cap,
                            &self.free_var_mask,
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
                    XlogError::Kernel(format!("xgcf_free_var_reduce_stage_cached failed: {}", e))
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
                            (
                                handle.slot_device(),
                                self.node_cap,
                                &mut self.values,
                                &self.meta_root,
                                result_buf,
                            ),
                        )
                    }
                    .map_err(|e| {
                        XlogError::Kernel(format!("xgcf_add_scalar_cached failed: {}", e))
                    })?;
                    break;
                }

                stage_n = out_len;
                stage0 = false;
                output_is_a = !output_is_a;
            }
        }

        Ok(())
    }
}
