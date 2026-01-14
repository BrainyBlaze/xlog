//! Helpers for testing XGCF circuit CUDA kernels.

use cudarc::driver::{CudaFunction, LaunchAsync, LaunchConfig};
use xlog_core::{Result, XlogError};
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

fn launch_level<A>(
    ctx: &TestContext,
    kernel_name: &str,
    num_level_nodes: u32,
    args: A,
) -> Result<()>
where
    CudaFunction: LaunchAsync<A>,
{
    if num_level_nodes == 0 {
        return Ok(());
    }
    let device = ctx.device.inner();
    let kernel = device
        .get_func(CIRCUIT_MODULE, kernel_name)
        .ok_or_else(|| XlogError::Kernel(format!("Kernel {} not found in {}", kernel_name, CIRCUIT_MODULE)))?;

    let block_size = 256u32;
    let num_blocks = (num_level_nodes + block_size - 1) / block_size;
    let config = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe { kernel.clone().launch(config, args) }
        .map_err(|e| XlogError::Kernel(format!("Failed to launch {}: {}", kernel_name, e)))?;
    Ok(())
}

pub fn run_tiny_xgcf_forward(ctx: &TestContext, spec: &TinyXgcfSpec) -> Result<Vec<f64>> {
    let device = ctx.device.inner();

    let mut d_node_type = ctx.memory.alloc::<u8>(spec.node_type.len())?;
    device
        .htod_sync_copy_into(&spec.node_type, &mut d_node_type)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload node_type: {}", e)))?;

    let mut d_child_offsets = ctx.memory.alloc::<u32>(spec.child_offsets.len())?;
    device
        .htod_sync_copy_into(&spec.child_offsets, &mut d_child_offsets)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload child_offsets: {}", e)))?;

    let mut d_child_indices = ctx.memory.alloc::<u32>(spec.child_indices.len())?;
    device
        .htod_sync_copy_into(&spec.child_indices, &mut d_child_indices)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload child_indices: {}", e)))?;

    let mut d_lit = ctx.memory.alloc::<i32>(spec.lit.len())?;
    device
        .htod_sync_copy_into(&spec.lit, &mut d_lit)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload lit: {}", e)))?;

    let mut d_decision_var = ctx.memory.alloc::<u32>(spec.decision_var.len())?;
    device
        .htod_sync_copy_into(&spec.decision_var, &mut d_decision_var)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload decision_var: {}", e)))?;

    let mut d_decision_child_false = ctx.memory.alloc::<u32>(spec.decision_child_false.len())?;
    device
        .htod_sync_copy_into(&spec.decision_child_false, &mut d_decision_child_false)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload decision_child_false: {}", e)))?;

    let mut d_decision_child_true = ctx.memory.alloc::<u32>(spec.decision_child_true.len())?;
    device
        .htod_sync_copy_into(&spec.decision_child_true, &mut d_decision_child_true)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload decision_child_true: {}", e)))?;

    let mut d_level_nodes = ctx.memory.alloc::<u32>(spec.level_nodes.len())?;
    device
        .htod_sync_copy_into(&spec.level_nodes, &mut d_level_nodes)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload level_nodes: {}", e)))?;

    let mut d_var_log_true = ctx.memory.alloc::<f64>(spec.var_log_true.len())?;
    device
        .htod_sync_copy_into(&spec.var_log_true, &mut d_var_log_true)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload var_log_true: {}", e)))?;

    let mut d_var_log_false = ctx.memory.alloc::<f64>(spec.var_log_false.len())?;
    device
        .htod_sync_copy_into(&spec.var_log_false, &mut d_var_log_false)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload var_log_false: {}", e)))?;

    let mut d_values = ctx.memory.alloc::<f64>(spec.num_nodes)?;
    let init_values = vec![0.0f64; spec.num_nodes];
    device
        .htod_sync_copy_into(&init_values, &mut d_values)
        .map_err(|e| XlogError::Kernel(format!("Failed to init values: {}", e)))?;

    for &(offset, len) in &spec.levels {
        let level_range: u64 = ((len as u64) << 32) | (offset as u64);
        launch_level(
            ctx,
            circuit_kernels::XGCF_FORWARD_LEVEL,
            len,
            (
                &d_node_type,
                &d_child_offsets,
                &d_child_indices,
                &d_lit,
                &d_decision_var,
                &d_decision_child_false,
                &d_decision_child_true,
                &d_level_nodes,
                level_range,
                &d_var_log_true,
                &d_var_log_false,
                &mut d_values,
            ),
        )?;
    }

    ctx.sync_and_check()?;

    device
        .dtoh_sync_copy(&d_values)
        .map_err(|e| XlogError::Kernel(format!("Failed to download values: {}", e)))
}

pub fn run_tiny_xgcf_backward(ctx: &TestContext, spec: &TinyXgcfSpec) -> Result<TinyXgcfRun> {
    let device = ctx.device.inner();

    let mut d_node_type = ctx.memory.alloc::<u8>(spec.node_type.len())?;
    device
        .htod_sync_copy_into(&spec.node_type, &mut d_node_type)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload node_type: {}", e)))?;

    let mut d_child_offsets = ctx.memory.alloc::<u32>(spec.child_offsets.len())?;
    device
        .htod_sync_copy_into(&spec.child_offsets, &mut d_child_offsets)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload child_offsets: {}", e)))?;

    let mut d_child_indices = ctx.memory.alloc::<u32>(spec.child_indices.len())?;
    device
        .htod_sync_copy_into(&spec.child_indices, &mut d_child_indices)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload child_indices: {}", e)))?;

    let mut d_lit = ctx.memory.alloc::<i32>(spec.lit.len())?;
    device
        .htod_sync_copy_into(&spec.lit, &mut d_lit)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload lit: {}", e)))?;

    let mut d_decision_var = ctx.memory.alloc::<u32>(spec.decision_var.len())?;
    device
        .htod_sync_copy_into(&spec.decision_var, &mut d_decision_var)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload decision_var: {}", e)))?;

    let mut d_decision_child_false = ctx.memory.alloc::<u32>(spec.decision_child_false.len())?;
    device
        .htod_sync_copy_into(&spec.decision_child_false, &mut d_decision_child_false)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload decision_child_false: {}", e)))?;

    let mut d_decision_child_true = ctx.memory.alloc::<u32>(spec.decision_child_true.len())?;
    device
        .htod_sync_copy_into(&spec.decision_child_true, &mut d_decision_child_true)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload decision_child_true: {}", e)))?;

    let mut d_level_nodes = ctx.memory.alloc::<u32>(spec.level_nodes.len())?;
    device
        .htod_sync_copy_into(&spec.level_nodes, &mut d_level_nodes)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload level_nodes: {}", e)))?;

    let mut d_var_log_true = ctx.memory.alloc::<f64>(spec.var_log_true.len())?;
    device
        .htod_sync_copy_into(&spec.var_log_true, &mut d_var_log_true)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload var_log_true: {}", e)))?;

    let mut d_var_log_false = ctx.memory.alloc::<f64>(spec.var_log_false.len())?;
    device
        .htod_sync_copy_into(&spec.var_log_false, &mut d_var_log_false)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload var_log_false: {}", e)))?;

    let mut d_values = ctx.memory.alloc::<f64>(spec.num_nodes)?;
    let init_values = vec![0.0f64; spec.num_nodes];
    device
        .htod_sync_copy_into(&init_values, &mut d_values)
        .map_err(|e| XlogError::Kernel(format!("Failed to init values: {}", e)))?;

    for &(offset, len) in &spec.levels {
        let level_range: u64 = ((len as u64) << 32) | (offset as u64);
        launch_level(
            ctx,
            circuit_kernels::XGCF_FORWARD_LEVEL,
            len,
            (
                &d_node_type,
                &d_child_offsets,
                &d_child_indices,
                &d_lit,
                &d_decision_var,
                &d_decision_child_false,
                &d_decision_child_true,
                &d_level_nodes,
                level_range,
                &d_var_log_true,
                &d_var_log_false,
                &mut d_values,
            ),
        )?;
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
    device
        .htod_sync_copy_into(&adj_init, &mut d_adj)
        .map_err(|e| XlogError::Kernel(format!("Failed to init adj: {}", e)))?;

    let mut d_grad_true = ctx.memory.alloc::<f64>(spec.num_vars + 1)?;
    let mut d_grad_false = ctx.memory.alloc::<f64>(spec.num_vars + 1)?;
    let grad_init = vec![0.0f64; spec.num_vars + 1];
    device
        .htod_sync_copy_into(&grad_init, &mut d_grad_true)
        .map_err(|e| XlogError::Kernel(format!("Failed to init grad_true: {}", e)))?;
    device
        .htod_sync_copy_into(&grad_init, &mut d_grad_false)
        .map_err(|e| XlogError::Kernel(format!("Failed to init grad_false: {}", e)))?;

    for &(offset, len) in spec.levels.iter().rev() {
        let level_range: u64 = ((len as u64) << 32) | (offset as u64);
        launch_level(
            ctx,
            circuit_kernels::XGCF_BACKWARD_LEVEL_PROPAGATE,
            len,
            (
                &d_node_type,
                &d_child_offsets,
                &d_child_indices,
                &d_decision_var,
                &d_decision_child_false,
                &d_decision_child_true,
                &d_level_nodes,
                level_range,
                &d_var_log_true,
                &d_var_log_false,
                &d_values,
                &mut d_adj,
            ),
        )?;
    }

    for &(offset, len) in spec.levels.iter().rev() {
        let level_range: u64 = ((len as u64) << 32) | (offset as u64);
        launch_level(
            ctx,
            circuit_kernels::XGCF_BACKWARD_LEVEL_DECISION_GRAD,
            len,
            (
                &d_node_type,
                &d_decision_var,
                &d_decision_child_false,
                &d_decision_child_true,
                &d_level_nodes,
                level_range,
                &d_var_log_true,
                &d_var_log_false,
                &d_values,
                &d_adj,
                &mut d_grad_true,
                &mut d_grad_false,
            ),
        )?;
    }

    for &(offset, len) in spec.levels.iter().rev() {
        let level_range: u64 = ((len as u64) << 32) | (offset as u64);
        launch_level(
            ctx,
            circuit_kernels::XGCF_BACKWARD_LEVEL_LIT_GRAD,
            len,
            (
                &d_node_type,
                &d_lit,
                &d_level_nodes,
                level_range,
                &d_adj,
                &mut d_grad_true,
                &mut d_grad_false,
            ),
        )?;
    }

    ctx.sync_and_check()?;

    let values = device
        .dtoh_sync_copy(&d_values)
        .map_err(|e| XlogError::Kernel(format!("Failed to download values: {}", e)))?;
    let adj = device
        .dtoh_sync_copy(&d_adj)
        .map_err(|e| XlogError::Kernel(format!("Failed to download adj: {}", e)))?;
    let grad_true = device
        .dtoh_sync_copy(&d_grad_true)
        .map_err(|e| XlogError::Kernel(format!("Failed to download grad_true: {}", e)))?;
    let grad_false = device
        .dtoh_sync_copy(&d_grad_false)
        .map_err(|e| XlogError::Kernel(format!("Failed to download grad_false: {}", e)))?;

    Ok(TinyXgcfRun {
        values,
        adj,
        grad_true,
        grad_false,
    })
}
