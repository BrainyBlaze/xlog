//! GPU-native weight table builders for exact inference.

use std::sync::Arc;

use cudarc::driver::{DeviceSlice, LaunchAsync, LaunchConfig};
use xlog_core::{Result, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::provider::{weights_kernels, WEIGHTS_MODULE};
use xlog_cuda::CudaKernelProvider;

use crate::compilation::gpu_cnf::GpuCnfVarTables;

pub struct GpuWeights {
    pub log_true: TrackedCudaSlice<f64>,
    pub log_false: TrackedCudaSlice<f64>,
}

fn grid_for(count: u32, block: u32) -> u32 {
    if count == 0 {
        0
    } else {
        (count + block - 1) / block
    }
}

pub fn build_evidence_by_var_gpu(
    node_var: &TrackedCudaSlice<u32>,
    evidence_nodes: &TrackedCudaSlice<u32>,
    evidence_vals: &TrackedCudaSlice<u8>,
    var_cap: u32,
    provider: &Arc<CudaKernelProvider>,
) -> Result<TrackedCudaSlice<u8>> {
    if evidence_nodes.len() != evidence_vals.len() {
        return Err(XlogError::Compilation(format!(
            "GPU evidence nodes len {} != vals len {}",
            evidence_nodes.len(),
            evidence_vals.len()
        )));
    }
    let len = (var_cap as usize)
        .checked_add(1)
        .ok_or_else(|| XlogError::Compilation("evidence var_cap overflow".to_string()))?;

    let memory = provider.memory();
    let device = provider.device().inner();
    let mut evidence_by_var = memory.alloc::<u8>(len)?;
    device
        .memset_zeros(&mut evidence_by_var)
        .map_err(|e| XlogError::Kernel(format!("Failed to zero evidence buffer: {}", e)))?;

    let count = evidence_nodes.len();
    if count == 0 {
        return Ok(evidence_by_var);
    }

    let func = device
        .get_func(
            WEIGHTS_MODULE,
            weights_kernels::WEIGHTS_SET_EVIDENCE_FROM_NODES,
        )
        .ok_or_else(|| {
            XlogError::Kernel("weights_set_evidence_from_nodes kernel not found".to_string())
        })?;
    let block = 256u32;
    let grid = grid_for(count as u32, block);
    // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
    unsafe {
        func.clone().launch(
            LaunchConfig {
                grid_dim: (grid.max(1), 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                node_var,
                evidence_nodes,
                evidence_vals,
                count as u32,
                var_cap,
                &mut evidence_by_var,
            ),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("weights_set_evidence_from_nodes failed: {}", e)))?;
    // No device synchronize: returns device-resident slice; same-stream ordering suffices.
    Ok(evidence_by_var)
}

pub fn map_nodes_to_vars_gpu(
    node_var: &TrackedCudaSlice<u32>,
    node_ids: &TrackedCudaSlice<u32>,
    var_cap: u32,
    provider: &Arc<CudaKernelProvider>,
) -> Result<TrackedCudaSlice<u32>> {
    let memory = provider.memory();
    let device = provider.device().inner();
    let mut out = memory.alloc::<u32>(node_ids.len())?;
    let count = node_ids.len();
    if count == 0 {
        return Ok(out);
    }

    let func = device
        .get_func(WEIGHTS_MODULE, weights_kernels::WEIGHTS_MAP_NODES_TO_VARS)
        .ok_or_else(|| {
            XlogError::Kernel("weights_map_nodes_to_vars kernel not found".to_string())
        })?;

    let block = 256u32;
    let grid = grid_for(count as u32, block);
    // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
    unsafe {
        func.clone().launch(
            LaunchConfig {
                grid_dim: (grid.max(1), 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            },
            (node_var, node_ids, count as u32, var_cap, &mut out),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("weights_map_nodes_to_vars failed: {}", e)))?;
    // No device synchronize: returns device-resident slice; same-stream ordering suffices.
    Ok(out)
}

pub fn apply_query_vars_device(
    provider: &Arc<CudaKernelProvider>,
    query_vars: &TrackedCudaSlice<u32>,
    var_cap: u32,
    log_false: &mut TrackedCudaSlice<f64>,
    saved: &mut TrackedCudaSlice<f64>,
) -> Result<()> {
    let count = query_vars.len();
    if saved.len() < count {
        return Err(XlogError::Compilation(format!(
            "query restore buffer len {} < query vars len {}",
            saved.len(),
            count
        )));
    }
    let weights_len = (var_cap as usize)
        .checked_add(1)
        .ok_or_else(|| XlogError::Compilation("query var_cap overflow".to_string()))?;
    if log_false.len() < weights_len {
        return Err(XlogError::Compilation(format!(
            "log_false len {} < var_cap+1 {}",
            log_false.len(),
            weights_len
        )));
    }
    if count == 0 {
        return Ok(());
    }

    let device = provider.device().inner();
    let func = device
        .get_func(WEIGHTS_MODULE, weights_kernels::WEIGHTS_APPLY_QUERY_VARS)
        .ok_or_else(|| {
            XlogError::Kernel("weights_apply_query_vars kernel not found".to_string())
        })?;

    let block = 256u32;
    let grid = grid_for(count as u32, block);
    // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
    unsafe {
        func.clone().launch(
            LaunchConfig {
                grid_dim: (grid.max(1), 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            },
            (query_vars, count as u32, var_cap, log_false, saved),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("weights_apply_query_vars failed: {}", e)))?;
    // No device synchronize: same-stream ordering guarantees visibility to subsequent kernels.
    Ok(())
}

pub fn restore_query_vars_device(
    provider: &Arc<CudaKernelProvider>,
    query_vars: &TrackedCudaSlice<u32>,
    var_cap: u32,
    log_false: &mut TrackedCudaSlice<f64>,
    saved: &TrackedCudaSlice<f64>,
) -> Result<()> {
    let count = query_vars.len();
    if saved.len() < count {
        return Err(XlogError::Compilation(format!(
            "query restore buffer len {} < query vars len {}",
            saved.len(),
            count
        )));
    }
    let weights_len = (var_cap as usize)
        .checked_add(1)
        .ok_or_else(|| XlogError::Compilation("query var_cap overflow".to_string()))?;
    if log_false.len() < weights_len {
        return Err(XlogError::Compilation(format!(
            "log_false len {} < var_cap+1 {}",
            log_false.len(),
            weights_len
        )));
    }
    if count == 0 {
        return Ok(());
    }

    let device = provider.device().inner();
    let func = device
        .get_func(WEIGHTS_MODULE, weights_kernels::WEIGHTS_RESTORE_QUERY_VARS)
        .ok_or_else(|| {
            XlogError::Kernel("weights_restore_query_vars kernel not found".to_string())
        })?;

    let block = 256u32;
    let grid = grid_for(count as u32, block);
    // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
    unsafe {
        func.clone().launch(
            LaunchConfig {
                grid_dim: (grid.max(1), 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            },
            (query_vars, count as u32, var_cap, log_false, saved),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("weights_restore_query_vars failed: {}", e)))?;
    // No device synchronize: same-stream ordering guarantees visibility to subsequent kernels.
    Ok(())
}

pub fn build_weights_gpu(
    vars: &GpuCnfVarTables,
    leaf_probs: &TrackedCudaSlice<f64>,
    choice_true: &TrackedCudaSlice<f64>,
    choice_false: &TrackedCudaSlice<f64>,
    evidence_by_var: &TrackedCudaSlice<u8>,
    provider: &Arc<CudaKernelProvider>,
) -> Result<GpuWeights> {
    let var_cap = vars.max_var;
    let weights_len = (var_cap as usize)
        .checked_add(1)
        .ok_or_else(|| XlogError::Compilation("weight table size overflow".to_string()))?;

    if vars.leaf_var.len() < leaf_probs.len() {
        return Err(XlogError::Compilation(format!(
            "leaf_probs len {} exceeds leaf_var len {}",
            leaf_probs.len(),
            vars.leaf_var.len()
        )));
    }
    if vars.choice_var.len() < choice_true.len() {
        return Err(XlogError::Compilation(format!(
            "choice_true len {} exceeds choice_var len {}",
            choice_true.len(),
            vars.choice_var.len()
        )));
    }
    if choice_true.len() != choice_false.len() {
        return Err(XlogError::Compilation(format!(
            "choice_true len {} != choice_false len {}",
            choice_true.len(),
            choice_false.len()
        )));
    }
    if evidence_by_var.len() != weights_len {
        return Err(XlogError::Compilation(format!(
            "evidence_by_var len {} != weights len {}",
            evidence_by_var.len(),
            weights_len
        )));
    }

    let memory = provider.memory();
    let device = provider.device().inner();
    let mut log_true = memory.alloc::<f64>(weights_len)?;
    let mut log_false = memory.alloc::<f64>(weights_len)?;

    // Initialize to 0.0
    device
        .memset_zeros(&mut log_true)
        .map_err(|e| XlogError::Kernel(format!("Failed to zero log_true weights: {}", e)))?;
    device
        .memset_zeros(&mut log_false)
        .map_err(|e| XlogError::Kernel(format!("Failed to zero log_false weights: {}", e)))?;

    let block = 256u32;

    if leaf_probs.len() > 0 {
        let func = device
            .get_func(WEIGHTS_MODULE, weights_kernels::WEIGHTS_FILL_LEAF)
            .ok_or_else(|| XlogError::Kernel("weights_fill_leaf kernel not found".to_string()))?;
        let grid = grid_for(leaf_probs.len() as u32, block);
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            func.clone().launch(
                LaunchConfig {
                    grid_dim: (grid.max(1), 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &vars.leaf_var,
                    leaf_probs,
                    leaf_probs.len() as u32,
                    var_cap,
                    &mut log_true,
                    &mut log_false,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("weights_fill_leaf failed: {}", e)))?;
    }

    if choice_true.len() > 0 {
        let func = device
            .get_func(WEIGHTS_MODULE, weights_kernels::WEIGHTS_FILL_CHOICE)
            .ok_or_else(|| XlogError::Kernel("weights_fill_choice kernel not found".to_string()))?;
        let grid = grid_for(choice_true.len() as u32, block);
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            func.clone().launch(
                LaunchConfig {
                    grid_dim: (grid.max(1), 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &vars.choice_var,
                    choice_true,
                    choice_false,
                    choice_true.len() as u32,
                    var_cap,
                    &mut log_true,
                    &mut log_false,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("weights_fill_choice failed: {}", e)))?;
    }

    if evidence_by_var.len() > 0 {
        let func = device
            .get_func(WEIGHTS_MODULE, weights_kernels::WEIGHTS_APPLY_EVIDENCE)
            .ok_or_else(|| {
                XlogError::Kernel("weights_apply_evidence kernel not found".to_string())
            })?;
        let grid = grid_for(var_cap + 1, block);
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            func.clone().launch(
                LaunchConfig {
                    grid_dim: (grid.max(1), 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: 0,
                },
                (evidence_by_var, var_cap, &mut log_true, &mut log_false),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("weights_apply_evidence failed: {}", e)))?;
    }
    // No device synchronize: returns device-resident weights; same-stream ordering suffices.
    Ok(GpuWeights {
        log_true,
        log_false,
    })
}

#[allow(dead_code)] // reserved: host-side weight upload path for testing/diagnostics
pub(crate) fn upload_weights_from_host(
    provider: &Arc<CudaKernelProvider>,
    weights: &[(f64, f64)],
) -> Result<GpuWeights> {
    let weights_len = weights.len();
    let mut host_true: Vec<f64> = Vec::with_capacity(weights_len);
    let mut host_false: Vec<f64> = Vec::with_capacity(weights_len);
    for &(t, f) in weights {
        host_true.push(t);
        host_false.push(f);
    }

    let memory = provider.memory();
    let device = provider.device().inner();
    let mut log_true = memory.alloc::<f64>(weights_len)?;
    let mut log_false = memory.alloc::<f64>(weights_len)?;
    device
        .htod_sync_copy_into(&host_true, &mut log_true)
        .map_err(|e| XlogError::Kernel(format!("Upload log_true weights failed: {}", e)))?;
    device
        .htod_sync_copy_into(&host_false, &mut log_false)
        .map_err(|e| XlogError::Kernel(format!("Upload log_false weights failed: {}", e)))?;

    Ok(GpuWeights {
        log_true,
        log_false,
    })
}
