//! GPU evaluator for XGCF circuits (CUDA).

use std::ffi::c_void;

use cudarc::driver::{DeviceRepr, DeviceSlice, LaunchAsync, LaunchConfig};
use xlog_core::{Result, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::provider::{arith_kernels, ARITH_MODULE};
use xlog_cuda::{circuit_kernels, CudaKernelProvider, CIRCUIT_MODULE};

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
    pub num_levels: u32,
    pub level_offsets: TrackedCudaSlice<u32>,
    pub level_nodes: TrackedCudaSlice<u32>,
    pub root: u32,
    pub max_var: u32,
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
    num_levels: u32,
    root: u32,
    max_var: u32,
    var_log_true: TrackedCudaSlice<f64>,
    var_log_false: TrackedCudaSlice<f64>,
    values: TrackedCudaSlice<f64>,
    adj: TrackedCudaSlice<f64>,
    grad_true: TrackedCudaSlice<f64>,
    grad_false: TrackedCudaSlice<f64>,
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
        if builder.node_type.len() != num_nodes
            || builder.child_offsets.len() != num_nodes + 1
            || builder.lit.len() != num_nodes
            || builder.decision_var.len() != num_nodes
            || builder.decision_child_false.len() != num_nodes
            || builder.decision_child_true.len() != num_nodes
        {
            return Err(XlogError::Compilation(
                "GpuXgcf::from_device: circuit buffer length mismatch".to_string(),
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
        if layout.level_nodes.len() != num_nodes {
            return Err(XlogError::Compilation(format!(
                "GpuXgcf::from_device: level_nodes len {} != num_nodes ({})",
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
            num_levels: layout.num_levels,
            root: layout.root,
            max_var: layout.max_var,
            var_log_true,
            var_log_false,
            values,
            adj,
            grad_true,
            grad_false,
        })
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
            num_levels: (circuit.level_offsets.len().saturating_sub(1)) as u32,
            root: circuit.roots[0],
            max_var,
            var_log_true,
            var_log_false,
            values,
            adj,
            grad_true,
            grad_false,
        })
    }

    pub fn max_var(&self) -> u32 {
        self.max_var
    }

    /// Root node id of the circuit (XGCF requires exactly one root for evaluation/verification).
    pub fn root(&self) -> u32 {
        self.root
    }

    /// Number of XGCF nodes in the circuit.
    pub fn num_nodes(&self) -> usize {
        self.node_type.len()
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

    /// Mutable access to device-resident log(true-weight) table.
    pub fn var_log_true_mut(&mut self) -> &mut TrackedCudaSlice<f64> {
        &mut self.var_log_true
    }

    /// Mutable access to device-resident log(false-weight) table.
    pub fn var_log_false_mut(&mut self) -> &mut TrackedCudaSlice<f64> {
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

        provider.device().synchronize()?;
        Ok(())
    }

    pub fn eval_log_wmc(
        &mut self,
        provider: &CudaKernelProvider,
        var_log_weights: &[(f64, f64)],
    ) -> Result<f64> {
        let device = provider.device().inner();

        let weights_len = (self.max_var as usize) + 1;
        if var_log_weights.len() < weights_len {
            return Err(XlogError::Compilation(format!(
                "GPU XGCF eval expects weight table len >= {}, got {}",
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

        provider.device().synchronize()?;

        let root_idx = self.root as usize;
        let root_view = self.values.slice(root_idx..(root_idx + 1));
        let mut out = [0.0_f64];
        device
            .dtoh_sync_copy_into(&root_view, &mut out)
            .map_err(|e| XlogError::Kernel(format!("Failed to read circuit root value: {}", e)))?;
        Ok(out[0])
    }

    pub fn eval_log_wmc_and_grads(
        &mut self,
        provider: &CudaKernelProvider,
        var_log_weights: &[(f64, f64)],
    ) -> Result<(f64, Vec<f64>, Vec<f64>)> {
        let log_z = self.eval_log_wmc(provider, var_log_weights)?;

        let device = provider.device().inner();

        device
            .memset_zeros(&mut self.adj)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero adj buffer: {}", e)))?;
        device
            .memset_zeros(&mut self.grad_true)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero grad_true buffer: {}", e)))?;
        device
            .memset_zeros(&mut self.grad_false)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero grad_false buffer: {}", e)))?;

        let root_idx = self.root as usize;
        let mut root_adj_view = self.adj.slice_mut(root_idx..(root_idx + 1));
        device
            .htod_sync_copy_into(&[1.0_f64], &mut root_adj_view)
            .map_err(|e| XlogError::Kernel(format!("Failed to set root adjoint: {}", e)))?;

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

        let block_size: u32 = 256;
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

        provider.device().synchronize()?;

        let weights_len = (self.max_var as usize) + 1;
        let mut host_grad_true: Vec<f64> = vec![0.0; weights_len];
        let mut host_grad_false: Vec<f64> = vec![0.0; weights_len];

        device
            .dtoh_sync_copy_into(&self.grad_true, &mut host_grad_true)
            .map_err(|e| XlogError::Kernel(format!("Failed to download grad_true: {}", e)))?;
        device
            .dtoh_sync_copy_into(&self.grad_false, &mut host_grad_false)
            .map_err(|e| XlogError::Kernel(format!("Failed to download grad_false: {}", e)))?;

        Ok((log_z, host_grad_true, host_grad_false))
    }
}
