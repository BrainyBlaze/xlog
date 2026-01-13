//! GPU evaluator for XGCF circuits (CUDA).

use cudarc::driver::{LaunchAsync, LaunchConfig};
use xlog_core::{Result, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::{circuit_kernels, CudaKernelProvider, CIRCUIT_MODULE};

use crate::xgcf::{Xgcf, XgcfNodeType};

pub struct GpuXgcf {
    node_type: TrackedCudaSlice<u8>,
    child_offsets: TrackedCudaSlice<u32>,
    child_indices: TrackedCudaSlice<u32>,
    lit: TrackedCudaSlice<i32>,
    decision_var: TrackedCudaSlice<u32>,
    decision_child_false: TrackedCudaSlice<u32>,
    decision_child_true: TrackedCudaSlice<u32>,
    level_nodes: TrackedCudaSlice<u32>,
    level_offsets: Vec<u32>,
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
        device.htod_sync_copy_into(&circuit.child_offsets, &mut d_child_offsets).map_err(|e| {
            XlogError::Kernel(format!("Failed to upload circuit child_offsets: {}", e))
        })?;

        let mut d_child_indices = memory.alloc::<u32>(circuit.child_indices.len())?;
        device.htod_sync_copy_into(&circuit.child_indices, &mut d_child_indices).map_err(|e| {
            XlogError::Kernel(format!("Failed to upload circuit child_indices: {}", e))
        })?;

        let mut d_lit = memory.alloc::<i32>(circuit.lit.len())?;
        device
            .htod_sync_copy_into(&circuit.lit, &mut d_lit)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload circuit lit: {}", e)))?;

        let mut d_decision_var = memory.alloc::<u32>(circuit.decision_var.len())?;
        device.htod_sync_copy_into(&circuit.decision_var, &mut d_decision_var).map_err(|e| {
            XlogError::Kernel(format!("Failed to upload circuit decision_var: {}", e))
        })?;

        let mut d_decision_child_false = memory.alloc::<u32>(circuit.decision_child_false.len())?;
        device
            .htod_sync_copy_into(&circuit.decision_child_false, &mut d_decision_child_false)
            .map_err(|e| {
                XlogError::Kernel(format!("Failed to upload circuit decision_child_false: {}", e))
            })?;

        let mut d_decision_child_true = memory.alloc::<u32>(circuit.decision_child_true.len())?;
        device
            .htod_sync_copy_into(&circuit.decision_child_true, &mut d_decision_child_true)
            .map_err(|e| {
                XlogError::Kernel(format!("Failed to upload circuit decision_child_true: {}", e))
            })?;

        let mut d_level_nodes = memory.alloc::<u32>(circuit.level_nodes.len())?;
        device
            .htod_sync_copy_into(&circuit.level_nodes, &mut d_level_nodes)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload circuit level_nodes: {}", e)))?;

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
            level_offsets: circuit.level_offsets.clone(),
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
        let num_levels = self.level_offsets.len().saturating_sub(1);
        for level in 0..num_levels {
            let start = self.level_offsets[level] as usize;
            let end = self.level_offsets[level + 1] as usize;
            let num_level_nodes = end.saturating_sub(start);
            if num_level_nodes == 0 {
                continue;
            }

            let num_blocks = ((num_level_nodes as u32) + block_size - 1) / block_size;
            let config = LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };

            let level_range = ((num_level_nodes as u64) << 32) | (start as u64);

            // SAFETY: xgcf_forward_level(...) writes values for the provided level nodes.
            unsafe {
                func.clone().launch(
                    config,
                    (
                        &self.node_type,
                        &self.child_offsets,
                        &self.child_indices,
                        &self.lit,
                        &self.decision_var,
                        &self.decision_child_false,
                        &self.decision_child_true,
                        &self.level_nodes,
                        level_range,
                        &self.var_log_true,
                        &self.var_log_false,
                        &self.values,
                    ),
                )
            }
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
            .get_func(CIRCUIT_MODULE, circuit_kernels::XGCF_BACKWARD_LEVEL_PROPAGATE)
            .ok_or_else(|| {
                XlogError::Kernel("xgcf_backward_level_propagate kernel not found".to_string())
            })?;
        let decision_grad = device
            .get_func(CIRCUIT_MODULE, circuit_kernels::XGCF_BACKWARD_LEVEL_DECISION_GRAD)
            .ok_or_else(|| {
                XlogError::Kernel(
                    "xgcf_backward_level_decision_grad kernel not found".to_string(),
                )
            })?;
        let lit_grad = device
            .get_func(CIRCUIT_MODULE, circuit_kernels::XGCF_BACKWARD_LEVEL_LIT_GRAD)
            .ok_or_else(|| {
                XlogError::Kernel("xgcf_backward_level_lit_grad kernel not found".to_string())
            })?;

        let block_size: u32 = 256;
        let num_levels = self.level_offsets.len().saturating_sub(1);
        for level in (0..num_levels).rev() {
            let start = self.level_offsets[level] as usize;
            let end = self.level_offsets[level + 1] as usize;
            let num_level_nodes = end.saturating_sub(start);
            if num_level_nodes == 0 {
                continue;
            }

            let num_blocks = ((num_level_nodes as u32) + block_size - 1) / block_size;
            let config = LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            let level_range = ((num_level_nodes as u64) << 32) | (start as u64);

            unsafe {
                propagate.clone().launch(
                    config,
                    (
                        &self.node_type,
                        &self.child_offsets,
                        &self.child_indices,
                        &self.decision_var,
                        &self.decision_child_false,
                        &self.decision_child_true,
                        &self.level_nodes,
                        level_range,
                        &self.var_log_true,
                        &self.var_log_false,
                        &self.values,
                        &self.adj,
                    ),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("xgcf_backward_level_propagate failed: {}", e)))?;

            unsafe {
                decision_grad.clone().launch(
                    config,
                    (
                        &self.node_type,
                        &self.decision_var,
                        &self.decision_child_false,
                        &self.decision_child_true,
                        &self.level_nodes,
                        level_range,
                        &self.var_log_true,
                        &self.var_log_false,
                        &self.values,
                        &self.adj,
                        &self.grad_true,
                        &self.grad_false,
                    ),
                )
            }
            .map_err(|e| {
                XlogError::Kernel(format!("xgcf_backward_level_decision_grad failed: {}", e))
            })?;

            unsafe {
                lit_grad.clone().launch(
                    config,
                    (
                        &self.node_type,
                        &self.lit,
                        &self.level_nodes,
                        level_range,
                        &self.adj,
                        &self.grad_true,
                        &self.grad_false,
                    ),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("xgcf_backward_level_lit_grad failed: {}", e)))?;
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
