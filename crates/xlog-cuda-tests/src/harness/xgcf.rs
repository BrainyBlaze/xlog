//! Helpers for testing XGCF circuit CUDA kernels.

use std::ffi::c_void;

use cudarc::driver::{CudaFunction, DeviceRepr, DeviceSlice, LaunchAsync, LaunchConfig};
use xlog_core::{Result, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::{circuit_kernels, CIRCUIT_MODULE};

use super::TestContext;

#[derive(Debug, Clone)]
pub struct TinyXgcfSpec {
    pub num_nodes: usize,
    pub num_vars: usize,
    pub root: u32,
    pub node_type: Vec<u8>,
    pub child_offsets: Vec<u32>,
    pub child_indices: Vec<u32>,
    pub lit: Vec<i32>,
    pub decision_var: Vec<u32>,
    pub decision_child_false: Vec<u32>,
    pub decision_child_true: Vec<u32>,
    pub level_nodes: Vec<u32>,
    pub levels: Vec<(u32, u32)>,
    pub var_log_true: Vec<f64>,
    pub var_log_false: Vec<f64>,
    pub expected_values: Vec<f64>,
    pub expected_grad_true: Vec<f64>,
    pub expected_grad_false: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct TinyXgcfRun {
    pub values: Vec<f64>,
    pub adj: Vec<f64>,
    pub grad_true: Vec<f64>,
    pub grad_false: Vec<f64>,
}

/// Device-resident XGCF circuit + reusable buffers.
///
/// This is used by certification categories that validate *transfer efficiency* and *circuit reuse*.
/// The key property: circuit structure is uploaded once; repeated evaluations reuse device buffers.
pub struct TinyXgcfDevice {
    pub num_nodes: usize,
    pub num_vars: usize,
    pub root: u32,
    levels: Vec<(u32, u32)>,

    // Cached kernel handles for performance-sensitive certification categories.
    forward_fn: CudaFunction,
    backward_propagate_fn: CudaFunction,
    backward_decision_grad_fn: CudaFunction,
    backward_lit_grad_fn: CudaFunction,

    // Circuit structure (device-resident).
    d_node_type: TrackedCudaSlice<u8>,
    d_child_offsets: TrackedCudaSlice<u32>,
    d_child_indices: TrackedCudaSlice<u32>,
    d_lit: TrackedCudaSlice<i32>,
    d_decision_var: TrackedCudaSlice<u32>,
    d_decision_child_false: TrackedCudaSlice<u32>,
    d_decision_child_true: TrackedCudaSlice<u32>,
    d_level_nodes: TrackedCudaSlice<u32>,
    d_level_offsets: TrackedCudaSlice<u32>,

    // Per-evaluation inputs (device-resident).
    d_var_log_true: TrackedCudaSlice<f64>,
    d_var_log_false: TrackedCudaSlice<f64>,

    // Per-evaluation outputs / scratch (device-resident).
    d_values: TrackedCudaSlice<f64>,
    d_adj: TrackedCudaSlice<f64>,
    d_grad_true: TrackedCudaSlice<f64>,
    d_grad_false: TrackedCudaSlice<f64>,
}

impl TinyXgcfDevice {
    pub fn upload(ctx: &TestContext, spec: &TinyXgcfSpec) -> Result<Self> {
        let device = ctx.device.inner();

        let forward_fn = device
            .get_func(CIRCUIT_MODULE, circuit_kernels::XGCF_FORWARD_LEVEL)
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                    "Kernel {} not found in {}",
                    circuit_kernels::XGCF_FORWARD_LEVEL,
                    CIRCUIT_MODULE
                ))
            })?;
        let backward_propagate_fn = device
            .get_func(
                CIRCUIT_MODULE,
                circuit_kernels::XGCF_BACKWARD_LEVEL_PROPAGATE,
            )
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                    "Kernel {} not found in {}",
                    circuit_kernels::XGCF_BACKWARD_LEVEL_PROPAGATE,
                    CIRCUIT_MODULE
                ))
            })?;
        let backward_decision_grad_fn = device
            .get_func(
                CIRCUIT_MODULE,
                circuit_kernels::XGCF_BACKWARD_LEVEL_DECISION_GRAD,
            )
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                    "Kernel {} not found in {}",
                    circuit_kernels::XGCF_BACKWARD_LEVEL_DECISION_GRAD,
                    CIRCUIT_MODULE
                ))
            })?;
        let backward_lit_grad_fn = device
            .get_func(
                CIRCUIT_MODULE,
                circuit_kernels::XGCF_BACKWARD_LEVEL_LIT_GRAD,
            )
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                    "Kernel {} not found in {}",
                    circuit_kernels::XGCF_BACKWARD_LEVEL_LIT_GRAD,
                    CIRCUIT_MODULE
                ))
            })?;

        let mut d_node_type = ctx.memory.alloc::<u8>(spec.node_type.len())?;
        ctx.htod_sync_copy_into(&spec.node_type, &mut d_node_type)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload node_type: {}", e)))?;

        let mut d_child_offsets = ctx.memory.alloc::<u32>(spec.child_offsets.len())?;
        ctx.htod_sync_copy_into(&spec.child_offsets, &mut d_child_offsets)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload child_offsets: {}", e)))?;

        let mut d_child_indices = ctx.memory.alloc::<u32>(spec.child_indices.len())?;
        ctx.htod_sync_copy_into(&spec.child_indices, &mut d_child_indices)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload child_indices: {}", e)))?;

        let mut d_lit = ctx.memory.alloc::<i32>(spec.lit.len())?;
        ctx.htod_sync_copy_into(&spec.lit, &mut d_lit)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload lit: {}", e)))?;

        let mut d_decision_var = ctx.memory.alloc::<u32>(spec.decision_var.len())?;
        ctx.htod_sync_copy_into(&spec.decision_var, &mut d_decision_var)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload decision_var: {}", e)))?;

        let mut d_decision_child_false =
            ctx.memory.alloc::<u32>(spec.decision_child_false.len())?;
        ctx.htod_sync_copy_into(&spec.decision_child_false, &mut d_decision_child_false)
            .map_err(|e| {
                XlogError::Kernel(format!("Failed to upload decision_child_false: {}", e))
            })?;

        let mut d_decision_child_true = ctx.memory.alloc::<u32>(spec.decision_child_true.len())?;
        ctx.htod_sync_copy_into(&spec.decision_child_true, &mut d_decision_child_true)
            .map_err(|e| {
                XlogError::Kernel(format!("Failed to upload decision_child_true: {}", e))
            })?;

        let mut d_level_nodes = ctx.memory.alloc::<u32>(spec.level_nodes.len())?;
        ctx.htod_sync_copy_into(&spec.level_nodes, &mut d_level_nodes)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload level_nodes: {}", e)))?;

        // Device-resident level offsets (len = num_levels + 1) for level-aware kernels.
        if spec.levels.is_empty() {
            return Err(XlogError::Kernel(
                "TinyXgcfSpec requires non-empty levels".to_string(),
            ));
        }
        let mut level_offsets: Vec<u32> = Vec::with_capacity(spec.levels.len() + 1);
        for &(offset, _len) in &spec.levels {
            level_offsets.push(offset);
        }
        let (last_offset, last_len) = *spec.levels.last().unwrap();
        level_offsets.push(last_offset + last_len);

        if level_offsets[0] != 0 {
            return Err(XlogError::Kernel(
                "TinyXgcfSpec level_offsets must start at 0".to_string(),
            ));
        }
        for (i, &(offset, len)) in spec.levels.iter().enumerate() {
            let expected_next = offset + len;
            if level_offsets[i] != offset || level_offsets[i + 1] != expected_next {
                return Err(XlogError::Kernel(
                    "TinyXgcfSpec levels must be contiguous and match offsets".to_string(),
                ));
            }
        }
        let total = *level_offsets.last().unwrap() as usize;
        if total != spec.level_nodes.len() {
            return Err(XlogError::Kernel(format!(
                "TinyXgcfSpec level_nodes len {} != level_offsets.last {}",
                spec.level_nodes.len(),
                total
            )));
        }

        let mut d_level_offsets = ctx.memory.alloc::<u32>(level_offsets.len())?;
        ctx.htod_sync_copy_into(&level_offsets, &mut d_level_offsets)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload level_offsets: {}", e)))?;

        let mut d_var_log_true = ctx.memory.alloc::<f64>(spec.var_log_true.len())?;
        ctx.htod_sync_copy_into(&spec.var_log_true, &mut d_var_log_true)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload var_log_true: {}", e)))?;

        let mut d_var_log_false = ctx.memory.alloc::<f64>(spec.var_log_false.len())?;
        ctx.htod_sync_copy_into(&spec.var_log_false, &mut d_var_log_false)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload var_log_false: {}", e)))?;

        let d_values = ctx.memory.alloc::<f64>(spec.num_nodes)?;
        let d_adj = ctx.memory.alloc::<f64>(spec.num_nodes)?;
        let d_grad_true = ctx.memory.alloc::<f64>(spec.num_vars + 1)?;
        let d_grad_false = ctx.memory.alloc::<f64>(spec.num_vars + 1)?;

        Ok(Self {
            num_nodes: spec.num_nodes,
            num_vars: spec.num_vars,
            root: spec.root,
            levels: spec.levels.clone(),
            forward_fn,
            backward_propagate_fn,
            backward_decision_grad_fn,
            backward_lit_grad_fn,
            d_node_type,
            d_child_offsets,
            d_child_indices,
            d_lit,
            d_decision_var,
            d_decision_child_false,
            d_decision_child_true,
            d_level_nodes,
            d_level_offsets,
            d_var_log_true,
            d_var_log_false,
            d_values,
            d_adj,
            d_grad_true,
            d_grad_false,
        })
    }

    fn launch_level_cached(
        kernel: &CudaFunction,
        num_level_nodes: u32,
        params: &mut Vec<*mut c_void>,
    ) -> Result<()> {
        if num_level_nodes == 0 {
            return Ok(());
        }
        let block_size = 256u32;
        let num_blocks = (num_level_nodes + block_size - 1) / block_size;
        let config = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { kernel.clone().launch(config, params) }
            .map_err(|e| XlogError::Kernel(format!("Failed to launch level kernel: {}", e)))?;
        Ok(())
    }

    pub fn set_weights(
        &mut self,
        ctx: &TestContext,
        log_true: &[f64],
        log_false: &[f64],
    ) -> Result<()> {
        if log_true.len() != self.d_var_log_true.len()
            || log_false.len() != self.d_var_log_false.len()
        {
            return Err(XlogError::Kernel(format!(
                "Weight length mismatch: got (true={}, false={}), expected (true={}, false={})",
                log_true.len(),
                log_false.len(),
                self.d_var_log_true.len(),
                self.d_var_log_false.len()
            )));
        }
        ctx.htod_sync_copy_into(log_true, &mut self.d_var_log_true)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload var_log_true: {}", e)))?;
        ctx.htod_sync_copy_into(log_false, &mut self.d_var_log_false)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload var_log_false: {}", e)))?;
        Ok(())
    }

    /// Launch forward kernels (no sync, no host transfers).
    pub fn forward_launch(&mut self, _ctx: &TestContext) -> Result<()> {
        for (level, &(_offset, len)) in self.levels.iter().enumerate() {
            let level_u32 = level as u32;
            let mut params: Vec<*mut c_void> = vec![
                (&self.d_node_type).as_kernel_param(),
                (&self.d_child_offsets).as_kernel_param(),
                (&self.d_child_indices).as_kernel_param(),
                (&self.d_lit).as_kernel_param(),
                (&self.d_decision_var).as_kernel_param(),
                (&self.d_decision_child_false).as_kernel_param(),
                (&self.d_decision_child_true).as_kernel_param(),
                (&self.d_level_nodes).as_kernel_param(),
                (&self.d_level_offsets).as_kernel_param(),
                level_u32.as_kernel_param(),
                (&self.d_var_log_true).as_kernel_param(),
                (&self.d_var_log_false).as_kernel_param(),
                (&mut self.d_values).as_kernel_param(),
            ];
            Self::launch_level_cached(&self.forward_fn, len, &mut params)?;
        }
        Ok(())
    }

    pub fn forward_download_values(&mut self, ctx: &TestContext) -> Result<Vec<f64>> {
        self.forward_launch(ctx)?;
        ctx.sync_and_check()?;
        ctx.dtoh_sync_copy(&self.d_values)
            .map_err(|e| XlogError::Kernel(format!("Failed to download values: {}", e)))
    }

    pub fn forward_download_root(&mut self, ctx: &TestContext) -> Result<f64> {
        self.forward_launch(ctx)?;
        ctx.sync_and_check()?;
        let root_idx: usize = self.root as usize;
        if root_idx >= self.num_nodes {
            return Err(XlogError::Kernel(format!(
                "Root {} out of bounds for num_nodes {}",
                self.root, self.num_nodes
            )));
        }
        let root_view = self.d_values.slice(root_idx..(root_idx + 1));
        let mut root_host = [0.0f64];
        ctx.dtoh_sync_copy_into(&root_view, &mut root_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to download root value: {}", e)))?;
        Ok(root_host[0])
    }

    /// Launch backward kernels using existing `d_values` (no sync, no host transfers).
    pub fn backward_only_launch(&mut self, ctx: &TestContext) -> Result<()> {
        let device = ctx.device.inner();
        device
            .memset_zeros(&mut self.d_adj)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero adj: {}", e)))?;
        device
            .memset_zeros(&mut self.d_grad_true)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero grad_true: {}", e)))?;
        device
            .memset_zeros(&mut self.d_grad_false)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero grad_false: {}", e)))?;

        let root_idx: usize = self.root as usize;
        if root_idx >= self.num_nodes {
            return Err(XlogError::Kernel(format!(
                "Root {} out of bounds for num_nodes {}",
                self.root, self.num_nodes
            )));
        }
        let mut root_view = self.d_adj.slice_mut(root_idx..(root_idx + 1));
        ctx.htod_sync_copy_into(&[1.0f64], &mut root_view)
            .map_err(|e| XlogError::Kernel(format!("Failed to set root adjoint: {}", e)))?;

        for (level, &(_offset, len)) in self.levels.iter().enumerate().rev() {
            let level_u32 = level as u32;
            let mut params: Vec<*mut c_void> = vec![
                (&self.d_node_type).as_kernel_param(),
                (&self.d_child_offsets).as_kernel_param(),
                (&self.d_child_indices).as_kernel_param(),
                (&self.d_decision_var).as_kernel_param(),
                (&self.d_decision_child_false).as_kernel_param(),
                (&self.d_decision_child_true).as_kernel_param(),
                (&self.d_level_nodes).as_kernel_param(),
                (&self.d_level_offsets).as_kernel_param(),
                level_u32.as_kernel_param(),
                (&self.d_var_log_true).as_kernel_param(),
                (&self.d_var_log_false).as_kernel_param(),
                (&self.d_values).as_kernel_param(),
                (&mut self.d_adj).as_kernel_param(),
            ];
            Self::launch_level_cached(&self.backward_propagate_fn, len, &mut params)?;
        }

        for (level, &(_offset, len)) in self.levels.iter().enumerate().rev() {
            let level_u32 = level as u32;
            let mut params: Vec<*mut c_void> = vec![
                (&self.d_node_type).as_kernel_param(),
                (&self.d_decision_var).as_kernel_param(),
                (&self.d_decision_child_false).as_kernel_param(),
                (&self.d_decision_child_true).as_kernel_param(),
                (&self.d_level_nodes).as_kernel_param(),
                (&self.d_level_offsets).as_kernel_param(),
                level_u32.as_kernel_param(),
                (&self.d_var_log_true).as_kernel_param(),
                (&self.d_var_log_false).as_kernel_param(),
                (&self.d_values).as_kernel_param(),
                (&self.d_adj).as_kernel_param(),
                (&mut self.d_grad_true).as_kernel_param(),
                (&mut self.d_grad_false).as_kernel_param(),
            ];
            Self::launch_level_cached(&self.backward_decision_grad_fn, len, &mut params)?;
        }

        for (level, &(_offset, len)) in self.levels.iter().enumerate().rev() {
            let level_u32 = level as u32;
            let mut params: Vec<*mut c_void> = vec![
                (&self.d_node_type).as_kernel_param(),
                (&self.d_lit).as_kernel_param(),
                (&self.d_level_nodes).as_kernel_param(),
                (&self.d_level_offsets).as_kernel_param(),
                level_u32.as_kernel_param(),
                (&self.d_adj).as_kernel_param(),
                (&mut self.d_grad_true).as_kernel_param(),
                (&mut self.d_grad_false).as_kernel_param(),
            ];
            Self::launch_level_cached(&self.backward_lit_grad_fn, len, &mut params)?;
        }

        Ok(())
    }

    /// Convenience helper: forward + backward in one launch sequence (no sync, no host transfers).
    pub fn forward_then_backward_launch(&mut self, ctx: &TestContext) -> Result<()> {
        self.forward_launch(ctx)?;
        self.backward_only_launch(ctx)
    }
}

fn logsumexp2(a: f64, b: f64) -> f64 {
    let m = a.max(b);
    if m.is_infinite() && m.is_sign_negative() {
        return m;
    }
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// Tiny Decision-DNNF-shaped XGCF circuit that exercises CONST/LIT/AND/OR/DECISION nodes.
pub fn tiny_xgcf_spec() -> TinyXgcfSpec {
    const CONST0: u8 = 0;
    const CONST1: u8 = 1;
    const LIT: u8 = 2;
    const AND: u8 = 3;
    const OR: u8 = 4;
    const DECISION: u8 = 5;

    // Node indices:
    // 0: CONST1
    // 1: LIT(+1)
    // 2: LIT(-2)
    // 3: AND(1,2)
    // 4: DECISION(var3, child_f=0, child_t=3)
    // 5: CONST0
    // 6: OR(4,5)   (root)
    let num_nodes = 7;
    let root = 6u32;

    let node_type: Vec<u8> = vec![CONST1, LIT, LIT, AND, DECISION, CONST0, OR];
    let lit: Vec<i32> = vec![0, 1, -2, 0, 0, 0, 0];
    let decision_var: Vec<u32> = vec![0, 0, 0, 0, 3, 0, 0];
    let decision_child_false: Vec<u32> = vec![0, 0, 0, 0, 0, 0, 0];
    let decision_child_true: Vec<u32> = vec![0, 0, 0, 0, 3, 0, 0];

    let child_offsets: Vec<u32> = vec![0, 0, 0, 0, 2, 2, 2, 4];
    let child_indices: Vec<u32> = vec![1, 2, 4, 5];

    // Levels: [0,1,2,5], [3], [4], [6]
    let level_nodes: Vec<u32> = vec![0, 1, 2, 5, 3, 4, 6];
    let levels: Vec<(u32, u32)> = vec![(0, 4), (4, 1), (5, 1), (6, 1)];

    let num_vars = 3usize;
    let var_log_true: Vec<f64> = vec![
        0.0,
        0.7f64.ln(), // var1
        0.2f64.ln(), // var2
        0.6f64.ln(), // var3
    ];
    let var_log_false: Vec<f64> = vec![
        0.0,
        0.3f64.ln(), // var1
        0.8f64.ln(), // var2
        0.4f64.ln(), // var3
    ];

    let v0 = 0.0;
    let v1 = var_log_true[1];
    let v2 = var_log_false[2];
    let v3 = v1 + v2;
    let v4 = logsumexp2(var_log_false[3] + v0, var_log_true[3] + v3);
    let v5 = f64::NEG_INFINITY;
    let v6 = logsumexp2(v4, v5);

    let expected_values: Vec<f64> = vec![v0, v1, v2, v3, v4, v5, v6];

    let p_false = (var_log_false[3] + v0 - v4).exp();
    let p_true = (var_log_true[3] + v3 - v4).exp();

    let mut expected_grad_true = vec![0.0f64; num_vars + 1];
    let mut expected_grad_false = vec![0.0f64; num_vars + 1];
    expected_grad_true[1] = p_true; // LIT(+1)
    expected_grad_false[2] = p_true; // LIT(-2)
    expected_grad_true[3] = p_true; // DECISION var3
    expected_grad_false[3] = p_false;

    TinyXgcfSpec {
        num_nodes,
        num_vars,
        root,
        node_type,
        child_offsets,
        child_indices,
        lit,
        decision_var,
        decision_child_false,
        decision_child_true,
        level_nodes,
        levels,
        var_log_true,
        var_log_false,
        expected_values,
        expected_grad_true,
        expected_grad_false,
    }
}

fn launch_level(
    ctx: &TestContext,
    kernel_name: &str,
    num_level_nodes: u32,
    params: &mut Vec<*mut c_void>,
) -> Result<()> {
    if num_level_nodes == 0 {
        return Ok(());
    }
    let device = ctx.device.inner();
    let kernel = device
        .get_func(CIRCUIT_MODULE, kernel_name)
        .ok_or_else(|| {
            XlogError::Kernel(format!(
                "Kernel {} not found in {}",
                kernel_name, CIRCUIT_MODULE
            ))
        })?;

    let block_size = 256u32;
    let num_blocks = (num_level_nodes + block_size - 1) / block_size;
    let config = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe { kernel.clone().launch(config, params) }
        .map_err(|e| XlogError::Kernel(format!("Failed to launch {}: {}", kernel_name, e)))?;
    Ok(())
}

pub fn run_tiny_xgcf_forward(ctx: &TestContext, spec: &TinyXgcfSpec) -> Result<Vec<f64>> {
    let mut d_node_type = ctx.memory.alloc::<u8>(spec.node_type.len())?;
    ctx
        .htod_sync_copy_into(&spec.node_type, &mut d_node_type)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload node_type: {}", e)))?;

    let mut d_child_offsets = ctx.memory.alloc::<u32>(spec.child_offsets.len())?;
    ctx
        .htod_sync_copy_into(&spec.child_offsets, &mut d_child_offsets)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload child_offsets: {}", e)))?;

    let mut d_child_indices = ctx.memory.alloc::<u32>(spec.child_indices.len())?;
    ctx
        .htod_sync_copy_into(&spec.child_indices, &mut d_child_indices)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload child_indices: {}", e)))?;

    let mut d_lit = ctx.memory.alloc::<i32>(spec.lit.len())?;
    ctx
        .htod_sync_copy_into(&spec.lit, &mut d_lit)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload lit: {}", e)))?;

    let mut d_decision_var = ctx.memory.alloc::<u32>(spec.decision_var.len())?;
    ctx
        .htod_sync_copy_into(&spec.decision_var, &mut d_decision_var)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload decision_var: {}", e)))?;

    let mut d_decision_child_false = ctx.memory.alloc::<u32>(spec.decision_child_false.len())?;
    ctx
        .htod_sync_copy_into(&spec.decision_child_false, &mut d_decision_child_false)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload decision_child_false: {}", e)))?;

    let mut d_decision_child_true = ctx.memory.alloc::<u32>(spec.decision_child_true.len())?;
    ctx
        .htod_sync_copy_into(&spec.decision_child_true, &mut d_decision_child_true)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload decision_child_true: {}", e)))?;

    let mut d_level_nodes = ctx.memory.alloc::<u32>(spec.level_nodes.len())?;
    ctx
        .htod_sync_copy_into(&spec.level_nodes, &mut d_level_nodes)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload level_nodes: {}", e)))?;

    let mut level_offsets: Vec<u32> = Vec::with_capacity(spec.levels.len() + 1);
    for &(offset, _len) in &spec.levels {
        level_offsets.push(offset);
    }
    let (last_offset, last_len) = *spec
        .levels
        .last()
        .ok_or_else(|| XlogError::Kernel("TinyXgcfSpec requires non-empty levels".to_string()))?;
    level_offsets.push(last_offset + last_len);
    if level_offsets[0] != 0 {
        return Err(XlogError::Kernel(
            "TinyXgcfSpec level_offsets must start at 0".to_string(),
        ));
    }
    let mut d_level_offsets = ctx.memory.alloc::<u32>(level_offsets.len())?;
    ctx
        .htod_sync_copy_into(&level_offsets, &mut d_level_offsets)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload level_offsets: {}", e)))?;

    let mut level_offsets: Vec<u32> = Vec::with_capacity(spec.levels.len() + 1);
    for &(offset, _len) in &spec.levels {
        level_offsets.push(offset);
    }
    let (last_offset, last_len) = *spec
        .levels
        .last()
        .ok_or_else(|| XlogError::Kernel("TinyXgcfSpec requires non-empty levels".to_string()))?;
    level_offsets.push(last_offset + last_len);
    if level_offsets[0] != 0 {
        return Err(XlogError::Kernel(
            "TinyXgcfSpec level_offsets must start at 0".to_string(),
        ));
    }
    let mut d_level_offsets = ctx.memory.alloc::<u32>(level_offsets.len())?;
    ctx
        .htod_sync_copy_into(&level_offsets, &mut d_level_offsets)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload level_offsets: {}", e)))?;

    let mut d_var_log_true = ctx.memory.alloc::<f64>(spec.var_log_true.len())?;
    ctx
        .htod_sync_copy_into(&spec.var_log_true, &mut d_var_log_true)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload var_log_true: {}", e)))?;

    let mut d_var_log_false = ctx.memory.alloc::<f64>(spec.var_log_false.len())?;
    ctx
        .htod_sync_copy_into(&spec.var_log_false, &mut d_var_log_false)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload var_log_false: {}", e)))?;

    let mut d_values = ctx.memory.alloc::<f64>(spec.num_nodes)?;
    let init_values = vec![0.0f64; spec.num_nodes];
    ctx
        .htod_sync_copy_into(&init_values, &mut d_values)
        .map_err(|e| XlogError::Kernel(format!("Failed to init values: {}", e)))?;

    for (level, &(_offset, len)) in spec.levels.iter().enumerate() {
        let level_u32 = level as u32;
        let mut params: Vec<*mut c_void> = vec![
            (&d_node_type).as_kernel_param(),
            (&d_child_offsets).as_kernel_param(),
            (&d_child_indices).as_kernel_param(),
            (&d_lit).as_kernel_param(),
            (&d_decision_var).as_kernel_param(),
            (&d_decision_child_false).as_kernel_param(),
            (&d_decision_child_true).as_kernel_param(),
            (&d_level_nodes).as_kernel_param(),
            (&d_level_offsets).as_kernel_param(),
            level_u32.as_kernel_param(),
            (&d_var_log_true).as_kernel_param(),
            (&d_var_log_false).as_kernel_param(),
            (&mut d_values).as_kernel_param(),
        ];
        launch_level(ctx, circuit_kernels::XGCF_FORWARD_LEVEL, len, &mut params)?;
    }

    ctx.sync_and_check()?;

    ctx
        .dtoh_sync_copy(&d_values)
        .map_err(|e| XlogError::Kernel(format!("Failed to download values: {}", e)))
}

pub fn run_tiny_xgcf_backward(ctx: &TestContext, spec: &TinyXgcfSpec) -> Result<TinyXgcfRun> {
    let mut d_node_type = ctx.memory.alloc::<u8>(spec.node_type.len())?;
    ctx
        .htod_sync_copy_into(&spec.node_type, &mut d_node_type)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload node_type: {}", e)))?;

    let mut d_child_offsets = ctx.memory.alloc::<u32>(spec.child_offsets.len())?;
    ctx
        .htod_sync_copy_into(&spec.child_offsets, &mut d_child_offsets)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload child_offsets: {}", e)))?;

    let mut d_child_indices = ctx.memory.alloc::<u32>(spec.child_indices.len())?;
    ctx
        .htod_sync_copy_into(&spec.child_indices, &mut d_child_indices)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload child_indices: {}", e)))?;

    let mut d_lit = ctx.memory.alloc::<i32>(spec.lit.len())?;
    ctx
        .htod_sync_copy_into(&spec.lit, &mut d_lit)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload lit: {}", e)))?;

    let mut d_decision_var = ctx.memory.alloc::<u32>(spec.decision_var.len())?;
    ctx
        .htod_sync_copy_into(&spec.decision_var, &mut d_decision_var)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload decision_var: {}", e)))?;

    let mut d_decision_child_false = ctx.memory.alloc::<u32>(spec.decision_child_false.len())?;
    ctx
        .htod_sync_copy_into(&spec.decision_child_false, &mut d_decision_child_false)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload decision_child_false: {}", e)))?;

    let mut d_decision_child_true = ctx.memory.alloc::<u32>(spec.decision_child_true.len())?;
    ctx
        .htod_sync_copy_into(&spec.decision_child_true, &mut d_decision_child_true)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload decision_child_true: {}", e)))?;

    let mut d_level_nodes = ctx.memory.alloc::<u32>(spec.level_nodes.len())?;
    ctx
        .htod_sync_copy_into(&spec.level_nodes, &mut d_level_nodes)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload level_nodes: {}", e)))?;

    let mut level_offsets: Vec<u32> = Vec::with_capacity(spec.levels.len() + 1);
    for &(offset, _len) in &spec.levels {
        level_offsets.push(offset);
    }
    let (last_offset, last_len) = *spec
        .levels
        .last()
        .ok_or_else(|| XlogError::Kernel("TinyXgcfSpec requires non-empty levels".to_string()))?;
    level_offsets.push(last_offset + last_len);
    if level_offsets[0] != 0 {
        return Err(XlogError::Kernel(
            "TinyXgcfSpec level_offsets must start at 0".to_string(),
        ));
    }
    let mut d_level_offsets = ctx.memory.alloc::<u32>(level_offsets.len())?;
    ctx
        .htod_sync_copy_into(&level_offsets, &mut d_level_offsets)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload level_offsets: {}", e)))?;

    let mut d_var_log_true = ctx.memory.alloc::<f64>(spec.var_log_true.len())?;
    ctx
        .htod_sync_copy_into(&spec.var_log_true, &mut d_var_log_true)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload var_log_true: {}", e)))?;

    let mut d_var_log_false = ctx.memory.alloc::<f64>(spec.var_log_false.len())?;
    ctx
        .htod_sync_copy_into(&spec.var_log_false, &mut d_var_log_false)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload var_log_false: {}", e)))?;

    let mut d_values = ctx.memory.alloc::<f64>(spec.num_nodes)?;
    let init_values = vec![0.0f64; spec.num_nodes];
    ctx
        .htod_sync_copy_into(&init_values, &mut d_values)
        .map_err(|e| XlogError::Kernel(format!("Failed to init values: {}", e)))?;

    for (level, &(_offset, len)) in spec.levels.iter().enumerate() {
        let level_u32 = level as u32;
        let mut params: Vec<*mut c_void> = vec![
            (&d_node_type).as_kernel_param(),
            (&d_child_offsets).as_kernel_param(),
            (&d_child_indices).as_kernel_param(),
            (&d_lit).as_kernel_param(),
            (&d_decision_var).as_kernel_param(),
            (&d_decision_child_false).as_kernel_param(),
            (&d_decision_child_true).as_kernel_param(),
            (&d_level_nodes).as_kernel_param(),
            (&d_level_offsets).as_kernel_param(),
            level_u32.as_kernel_param(),
            (&d_var_log_true).as_kernel_param(),
            (&d_var_log_false).as_kernel_param(),
            (&mut d_values).as_kernel_param(),
        ];
        launch_level(ctx, circuit_kernels::XGCF_FORWARD_LEVEL, len, &mut params)?;
    }

    // adj[root] = 1, others 0
    let mut adj_init = vec![0.0f64; spec.num_nodes];
    let root_idx: usize = spec.root as usize;
    if root_idx >= adj_init.len() {
        return Err(XlogError::Kernel(format!(
            "Root {} out of bounds for num_nodes {}",
            spec.root, spec.num_nodes
        )));
    }
    adj_init[root_idx] = 1.0;
    let mut d_adj = ctx.memory.alloc::<f64>(spec.num_nodes)?;
    ctx
        .htod_sync_copy_into(&adj_init, &mut d_adj)
        .map_err(|e| XlogError::Kernel(format!("Failed to init adj: {}", e)))?;

    let mut d_grad_true = ctx.memory.alloc::<f64>(spec.num_vars + 1)?;
    let mut d_grad_false = ctx.memory.alloc::<f64>(spec.num_vars + 1)?;
    let grad_init = vec![0.0f64; spec.num_vars + 1];
    ctx
        .htod_sync_copy_into(&grad_init, &mut d_grad_true)
        .map_err(|e| XlogError::Kernel(format!("Failed to init grad_true: {}", e)))?;
    ctx
        .htod_sync_copy_into(&grad_init, &mut d_grad_false)
        .map_err(|e| XlogError::Kernel(format!("Failed to init grad_false: {}", e)))?;

    for (level, &(_offset, len)) in spec.levels.iter().enumerate().rev() {
        let level_u32 = level as u32;
        let mut params: Vec<*mut c_void> = vec![
            (&d_node_type).as_kernel_param(),
            (&d_child_offsets).as_kernel_param(),
            (&d_child_indices).as_kernel_param(),
            (&d_decision_var).as_kernel_param(),
            (&d_decision_child_false).as_kernel_param(),
            (&d_decision_child_true).as_kernel_param(),
            (&d_level_nodes).as_kernel_param(),
            (&d_level_offsets).as_kernel_param(),
            level_u32.as_kernel_param(),
            (&d_var_log_true).as_kernel_param(),
            (&d_var_log_false).as_kernel_param(),
            (&d_values).as_kernel_param(),
            (&mut d_adj).as_kernel_param(),
        ];
        launch_level(
            ctx,
            circuit_kernels::XGCF_BACKWARD_LEVEL_PROPAGATE,
            len,
            &mut params,
        )?;
    }

    for (level, &(_offset, len)) in spec.levels.iter().enumerate().rev() {
        let level_u32 = level as u32;
        let mut params: Vec<*mut c_void> = vec![
            (&d_node_type).as_kernel_param(),
            (&d_decision_var).as_kernel_param(),
            (&d_decision_child_false).as_kernel_param(),
            (&d_decision_child_true).as_kernel_param(),
            (&d_level_nodes).as_kernel_param(),
            (&d_level_offsets).as_kernel_param(),
            level_u32.as_kernel_param(),
            (&d_var_log_true).as_kernel_param(),
            (&d_var_log_false).as_kernel_param(),
            (&d_values).as_kernel_param(),
            (&d_adj).as_kernel_param(),
            (&mut d_grad_true).as_kernel_param(),
            (&mut d_grad_false).as_kernel_param(),
        ];
        launch_level(
            ctx,
            circuit_kernels::XGCF_BACKWARD_LEVEL_DECISION_GRAD,
            len,
            &mut params,
        )?;
    }

    for (level, &(_offset, len)) in spec.levels.iter().enumerate().rev() {
        let level_u32 = level as u32;
        let mut params: Vec<*mut c_void> = vec![
            (&d_node_type).as_kernel_param(),
            (&d_lit).as_kernel_param(),
            (&d_level_nodes).as_kernel_param(),
            (&d_level_offsets).as_kernel_param(),
            level_u32.as_kernel_param(),
            (&d_adj).as_kernel_param(),
            (&mut d_grad_true).as_kernel_param(),
            (&mut d_grad_false).as_kernel_param(),
        ];
        launch_level(
            ctx,
            circuit_kernels::XGCF_BACKWARD_LEVEL_LIT_GRAD,
            len,
            &mut params,
        )?;
    }

    ctx.sync_and_check()?;

    let values = ctx
        .dtoh_sync_copy(&d_values)
        .map_err(|e| XlogError::Kernel(format!("Failed to download values: {}", e)))?;
    let adj = ctx
        .dtoh_sync_copy(&d_adj)
        .map_err(|e| XlogError::Kernel(format!("Failed to download adj: {}", e)))?;
    let grad_true = ctx
        .dtoh_sync_copy(&d_grad_true)
        .map_err(|e| XlogError::Kernel(format!("Failed to download grad_true: {}", e)))?;
    let grad_false = ctx
        .dtoh_sync_copy(&d_grad_false)
        .map_err(|e| XlogError::Kernel(format!("Failed to download grad_false: {}", e)))?;

    Ok(TinyXgcfRun {
        values,
        adj,
        grad_true,
        grad_false,
    })
}

/// Generate a single-literal circuit: root = Lit(+var)
pub fn gen_single_lit_circuit(var: u32) -> TinyXgcfSpec {
    const LIT: u8 = 2;

    let num_nodes = 1;
    let num_vars = var as usize;
    let root = 0;

    let node_type = vec![LIT];
    let child_offsets = vec![0, 0];
    let child_indices = vec![];
    let lit = vec![var as i32];
    let decision_var = vec![0];
    let decision_child_false = vec![0];
    let decision_child_true = vec![0];
    let level_nodes = vec![0];
    let levels = vec![(0, 1)];

    let mut var_log_true = vec![0.0; num_vars + 1];
    let mut var_log_false = vec![0.0; num_vars + 1];
    var_log_true[var as usize] = 0.7_f64.ln();
    var_log_false[var as usize] = 0.3_f64.ln();

    let expected_values = vec![var_log_true[var as usize]];
    let mut expected_grad_true = vec![0.0; num_vars + 1];
    let expected_grad_false = vec![0.0; num_vars + 1];
    expected_grad_true[var as usize] = 1.0;

    TinyXgcfSpec {
        num_nodes,
        num_vars,
        root,
        node_type,
        child_offsets,
        child_indices,
        lit,
        decision_var,
        decision_child_false,
        decision_child_true,
        level_nodes,
        levels,
        var_log_true,
        var_log_false,
        expected_values,
        expected_grad_true,
        expected_grad_false,
    }
}

/// Generate an AND circuit: root = AND(Lit(+1), Lit(+2))
pub fn gen_and_circuit() -> TinyXgcfSpec {
    const LIT: u8 = 2;
    const AND: u8 = 3;

    let num_nodes = 3;
    let num_vars = 2;
    let root = 2;

    let node_type = vec![LIT, LIT, AND];
    let child_offsets = vec![0, 0, 0, 2];
    let child_indices = vec![0, 1];
    let lit = vec![1, 2, 0];
    let decision_var = vec![0, 0, 0];
    let decision_child_false = vec![0, 0, 0];
    let decision_child_true = vec![0, 0, 0];
    let level_nodes = vec![0, 1, 2];
    let levels = vec![(0, 2), (2, 1)];

    let p1 = 0.7_f64;
    let p2 = 0.6_f64;
    let var_log_true = vec![0.0, p1.ln(), p2.ln()];
    let var_log_false = vec![0.0, (1.0 - p1).ln(), (1.0 - p2).ln()];

    let v0 = var_log_true[1];
    let v1 = var_log_true[2];
    let v2 = v0 + v1;
    let expected_values = vec![v0, v1, v2];
    let expected_grad_true = vec![0.0, 1.0, 1.0];
    let expected_grad_false = vec![0.0, 0.0, 0.0];

    TinyXgcfSpec {
        num_nodes,
        num_vars,
        root,
        node_type,
        child_offsets,
        child_indices,
        lit,
        decision_var,
        decision_child_false,
        decision_child_true,
        level_nodes,
        levels,
        var_log_true,
        var_log_false,
        expected_values,
        expected_grad_true,
        expected_grad_false,
    }
}

/// Generate an OR circuit: root = OR(Lit(+1), Lit(+2))
pub fn gen_or_circuit() -> TinyXgcfSpec {
    const LIT: u8 = 2;
    const OR: u8 = 4;

    let num_nodes = 3;
    let num_vars = 2;
    let root = 2;

    let node_type = vec![LIT, LIT, OR];
    let child_offsets = vec![0, 0, 0, 2];
    let child_indices = vec![0, 1];
    let lit = vec![1, 2, 0];
    let decision_var = vec![0, 0, 0];
    let decision_child_false = vec![0, 0, 0];
    let decision_child_true = vec![0, 0, 0];
    let level_nodes = vec![0, 1, 2];
    let levels = vec![(0, 2), (2, 1)];

    let p1 = 0.7_f64;
    let p2 = 0.6_f64;
    let var_log_true = vec![0.0, p1.ln(), p2.ln()];
    let var_log_false = vec![0.0, (1.0 - p1).ln(), (1.0 - p2).ln()];

    let v0 = var_log_true[1];
    let v1 = var_log_true[2];
    let v2 = logsumexp2(v0, v1);
    let expected_values = vec![v0, v1, v2];

    let p_child0 = (v0 - v2).exp();
    let p_child1 = (v1 - v2).exp();
    let expected_grad_true = vec![0.0, p_child0, p_child1];
    let expected_grad_false = vec![0.0, 0.0, 0.0];

    TinyXgcfSpec {
        num_nodes,
        num_vars,
        root,
        node_type,
        child_offsets,
        child_indices,
        lit,
        decision_var,
        decision_child_false,
        decision_child_true,
        level_nodes,
        levels,
        var_log_true,
        var_log_false,
        expected_values,
        expected_grad_true,
        expected_grad_false,
    }
}

/// Generate a Decision circuit: root = Decision(var, false_child=Const1, true_child=Lit(+1))
pub fn gen_decision_circuit() -> TinyXgcfSpec {
    const CONST1: u8 = 1;
    const LIT: u8 = 2;
    const DECISION: u8 = 5;

    let num_nodes = 3;
    let num_vars = 2;
    let root = 2;

    let node_type = vec![CONST1, LIT, DECISION];
    let child_offsets = vec![0, 0, 0, 0];
    let child_indices = vec![];
    let lit = vec![0, 1, 0];
    let decision_var = vec![0, 0, 2];
    let decision_child_false = vec![0, 0, 0];
    let decision_child_true = vec![0, 0, 1];
    let level_nodes = vec![0, 1, 2];
    let levels = vec![(0, 2), (2, 1)];

    let p1 = 0.7_f64;
    let p2 = 0.6_f64;
    let var_log_true = vec![0.0, p1.ln(), p2.ln()];
    let var_log_false = vec![0.0, (1.0 - p1).ln(), (1.0 - p2).ln()];

    let v0 = 0.0;
    let v1 = var_log_true[1];
    let v2 = logsumexp2(var_log_false[2] + v0, var_log_true[2] + v1);
    let expected_values = vec![v0, v1, v2];

    let p_false = (var_log_false[2] + v0 - v2).exp();
    let p_true = (var_log_true[2] + v1 - v2).exp();

    let expected_grad_true = vec![0.0, p_true, p_true];
    let expected_grad_false = vec![0.0, 0.0, p_false];

    TinyXgcfSpec {
        num_nodes,
        num_vars,
        root,
        node_type,
        child_offsets,
        child_indices,
        lit,
        decision_var,
        decision_child_false,
        decision_child_true,
        level_nodes,
        levels,
        var_log_true,
        var_log_false,
        expected_values,
        expected_grad_true,
        expected_grad_false,
    }
}

/// Generate a large circuit with N parallel literals under an OR node
pub fn gen_large_or_circuit(num_vars: usize) -> TinyXgcfSpec {
    const LIT: u8 = 2;
    const OR: u8 = 4;

    let num_nodes = num_vars + 1;
    let root = num_vars as u32;

    let mut node_type = vec![LIT; num_vars];
    node_type.push(OR);

    let mut child_offsets: Vec<u32> = (0..=num_vars).map(|_| 0).collect();
    child_offsets.push(num_vars as u32);

    let child_indices: Vec<u32> = (0..num_vars as u32).collect();

    let mut lit: Vec<i32> = (1..=num_vars as i32).collect();
    lit.push(0);

    let decision_var = vec![0; num_nodes];
    let decision_child_false = vec![0; num_nodes];
    let decision_child_true = vec![0; num_nodes];

    let mut level_nodes: Vec<u32> = (0..num_vars as u32).collect();
    level_nodes.push(root);
    let levels = vec![(0, num_vars as u32), (num_vars as u32, 1)];

    let p = 0.5_f64;
    let mut var_log_true = vec![0.0; num_vars + 1];
    let mut var_log_false = vec![0.0; num_vars + 1];
    for i in 1..=num_vars {
        var_log_true[i] = p.ln();
        var_log_false[i] = (1.0 - p).ln();
    }

    let lit_val = p.ln();
    let mut expected_values = vec![lit_val; num_vars];
    let or_val = lit_val + (num_vars as f64).ln();
    expected_values.push(or_val);

    let grad_per_lit = 1.0 / num_vars as f64;
    let mut expected_grad_true = vec![0.0; num_vars + 1];
    for i in 1..=num_vars {
        expected_grad_true[i] = grad_per_lit;
    }
    let expected_grad_false = vec![0.0; num_vars + 1];

    TinyXgcfSpec {
        num_nodes,
        num_vars,
        root,
        node_type,
        child_offsets,
        child_indices,
        lit,
        decision_var,
        decision_child_false,
        decision_child_true,
        level_nodes,
        levels,
        var_log_true,
        var_log_false,
        expected_values,
        expected_grad_true,
        expected_grad_false,
    }
}

/// Generate a deep chain circuit: AND(AND(AND(...Lit(1)...)))
pub fn gen_deep_chain_circuit(depth: usize) -> TinyXgcfSpec {
    const LIT: u8 = 2;
    const AND: u8 = 3;

    let num_nodes = depth + 1;
    let num_vars = 1;
    let root = depth as u32;

    let mut node_type = vec![LIT];
    for _ in 0..depth {
        node_type.push(AND);
    }

    let mut child_offsets: Vec<u32> = vec![0];
    let mut child_indices: Vec<u32> = vec![];
    for i in 0..depth {
        child_offsets.push(child_indices.len() as u32);
        child_indices.push(i as u32);
    }
    child_offsets.push(child_indices.len() as u32);

    let mut lit = vec![1i32];
    lit.extend(vec![0i32; depth]);

    let decision_var = vec![0; num_nodes];
    let decision_child_false = vec![0; num_nodes];
    let decision_child_true = vec![0; num_nodes];

    let level_nodes: Vec<u32> = (0..num_nodes as u32).collect();
    let levels: Vec<(u32, u32)> = (0..num_nodes).map(|i| (i as u32, 1)).collect();

    let p = 0.7_f64;
    let var_log_true = vec![0.0, p.ln()];
    let var_log_false = vec![0.0, (1.0 - p).ln()];

    let lit_val = p.ln();
    let expected_values = vec![lit_val; num_nodes];

    let expected_grad_true = vec![0.0, 1.0];
    let expected_grad_false = vec![0.0, 0.0];

    TinyXgcfSpec {
        num_nodes,
        num_vars,
        root,
        node_type,
        child_offsets,
        child_indices,
        lit,
        decision_var,
        decision_child_false,
        decision_child_true,
        level_nodes,
        levels,
        var_log_true,
        var_log_false,
        expected_values,
        expected_grad_true,
        expected_grad_false,
    }
}

/// Compute numerical gradient for verification
pub fn numerical_gradient(
    ctx: &TestContext,
    spec: &TinyXgcfSpec,
    var: usize,
    eps: f64,
) -> xlog_core::Result<(f64, f64)> {
    let mut spec_plus = spec.clone();
    let mut spec_minus = spec.clone();
    spec_plus.var_log_true[var] += eps;
    spec_minus.var_log_true[var] -= eps;

    let values_plus = run_tiny_xgcf_forward(ctx, &spec_plus)?;
    let values_minus = run_tiny_xgcf_forward(ctx, &spec_minus)?;

    let grad_true =
        (values_plus[spec.root as usize] - values_minus[spec.root as usize]) / (2.0 * eps);

    let mut spec_plus = spec.clone();
    let mut spec_minus = spec.clone();
    spec_plus.var_log_false[var] += eps;
    spec_minus.var_log_false[var] -= eps;

    let values_plus = run_tiny_xgcf_forward(ctx, &spec_plus)?;
    let values_minus = run_tiny_xgcf_forward(ctx, &spec_minus)?;

    let grad_false =
        (values_plus[spec.root as usize] - values_minus[spec.root as usize]) / (2.0 * eps);

    Ok((grad_true, grad_false))
}
