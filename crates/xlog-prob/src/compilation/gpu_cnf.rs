//! GPU-native Tseitin CNF encoding for PIR graphs.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{DeviceRepr, DeviceSlice, LaunchAsync, LaunchConfig};
use xlog_core::{Result, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::provider::{cnf_kernels, CNF_MODULE};
use xlog_cuda::CudaKernelProvider;
use xlog_solve::GpuCnf;

use crate::compilation::gpu_pir::GpuPirGraph;
use crate::compilation::gpu_pir::GpuPirRoots;

/// GPU-resident CNF variable tables for PIR ids.
pub struct GpuCnfVarTables {
    pub node_var: TrackedCudaSlice<u32>,
    pub leaf_var: TrackedCudaSlice<u32>,
    pub choice_var: TrackedCudaSlice<u32>,
    pub max_var: u32,
}

/// GPU-resident CNF encoding bundle (CNF + var tables).
pub struct GpuCnfEncoding {
    pub cnf: GpuCnf,
    pub vars: GpuCnfVarTables,
    /// Largest variable id that is semantically meaningful and should be eligible for branching
    /// in the GPU CDCL verifier (len = 1, device-resident).
    ///
    /// For PIR Tseitin encodings this is the end of the leaf+choice var range, excluding internal
    /// node Tseitin vars which are propagation-only.
    pub decision_var_limit: TrackedCudaSlice<u32>,
}

fn grid_dim(n: u32, block: u32) -> u32 {
    let mut grid = (n + block - 1) / block;
    if grid == 0 {
        grid = 1;
    }
    if grid > 65_535 {
        grid = 65_535;
    }
    grid
}

/// Encode a GPU PIR graph into GPU-resident Tseitin CNF.
pub fn encode_cnf_gpu(
    pir: &GpuPirGraph,
    roots: &GpuPirRoots,
    provider: &Arc<CudaKernelProvider>,
) -> Result<GpuCnfEncoding> {
    if roots.roots.len() == 0 {
        return Err(XlogError::Compilation(
            "Cannot encode CNF for empty PIR root set".to_string(),
        ));
    }
    let num_nodes = pir.node_type.len();
    if num_nodes == 0 {
        return Err(XlogError::Compilation(
            "Cannot encode CNF for empty PIR graph".to_string(),
        ));
    }

    let num_nodes_u32 = u32::try_from(num_nodes)
        .map_err(|_| XlogError::Compilation("PIR node count overflow".to_string()))?;
    let num_roots_u32 = u32::try_from(roots.roots.len())
        .map_err(|_| XlogError::Compilation("PIR root count exceeds u32::MAX".to_string()))?;

    let num_edges = pir.children.len();
    let n64 = num_nodes as u64;
    let e64 = num_edges as u64;

    let var_cap = u32::try_from(
        n64.checked_mul(3)
            .ok_or_else(|| XlogError::Kernel("CNF var capacity overflow".to_string()))?,
    )
    .map_err(|_| XlogError::Kernel("CNF var capacity exceeds u32::MAX".to_string()))?;
    let clause_cap = u32::try_from(
        e64.checked_add(
            n64.checked_mul(4)
                .ok_or_else(|| XlogError::Kernel("CNF clause capacity overflow".to_string()))?,
        )
        .ok_or_else(|| XlogError::Kernel("CNF clause capacity overflow".to_string()))?,
    )
    .map_err(|_| XlogError::Kernel("CNF clause capacity exceeds u32::MAX".to_string()))?;
    let lit_cap =
        u32::try_from(
            e64.checked_mul(3)
                .ok_or_else(|| XlogError::Kernel("CNF literal capacity overflow".to_string()))?
                .checked_add(n64.checked_mul(12).ok_or_else(|| {
                    XlogError::Kernel("CNF literal capacity overflow".to_string())
                })?)
                .ok_or_else(|| XlogError::Kernel("CNF literal capacity overflow".to_string()))?,
        )
        .map_err(|_| XlogError::Kernel("CNF literal capacity exceeds u32::MAX".to_string()))?;

    let leaf_cap = num_nodes_u32;
    let choice_cap = num_nodes_u32;

    let memory = provider.memory();
    let device = provider.device().inner();

    let mut reachable = memory.alloc::<u32>(num_nodes)?;
    let mut queue = memory.alloc::<u32>(num_nodes)?;
    let mut queue_ready = memory.alloc::<u32>(num_nodes)?;
    let mut head = memory.alloc::<u32>(1)?;
    let mut tail = memory.alloc::<u32>(1)?;
    let mut in_flight = memory.alloc::<u32>(1)?;

    let mut leaf_used = memory.alloc::<u32>(leaf_cap as usize)?;
    let mut choice_used = memory.alloc::<u32>(choice_cap as usize)?;
    let mut leaf_var = memory.alloc::<u32>(leaf_cap as usize)?;
    let mut choice_var = memory.alloc::<u32>(choice_cap as usize)?;

    let mut node_needs_var = memory.alloc::<u32>(num_nodes)?;
    let mut node_var = memory.alloc::<u32>(num_nodes)?;

    let mut clause_counts = memory.alloc::<u32>(num_nodes)?;
    let mut lit_counts = memory.alloc::<u32>(num_nodes)?;

    let mut leaf_prefix = memory.alloc::<u32>(leaf_cap as usize)?;
    let mut choice_prefix = memory.alloc::<u32>(choice_cap as usize)?;

    let mut node_last = memory.alloc::<u32>(1)?;
    let mut clause_last = memory.alloc::<u32>(1)?;
    let mut lit_last = memory.alloc::<u32>(1)?;

    let mut num_leaf = memory.alloc::<u32>(1)?;
    let mut num_choice = memory.alloc::<u32>(1)?;
    let mut base_choice = memory.alloc::<u32>(1)?;
    let mut base_node = memory.alloc::<u32>(1)?;
    let mut decision_var_limit = memory.alloc::<u32>(1)?;

    let mut d_num_vars = memory.alloc::<u32>(1)?;
    let mut d_num_clauses = memory.alloc::<u32>(1)?;
    let mut d_num_lits = memory.alloc::<u32>(1)?;

    let mut d_offsets = memory.alloc::<u32>((clause_cap as usize) + 1)?;
    let mut d_lits = memory.alloc::<i32>(lit_cap as usize)?;

    device
        .memset_zeros(&mut reachable)
        .map_err(|e| XlogError::Kernel(format!("zero reachable: {}", e)))?;
    device
        .memset_zeros(&mut queue)
        .map_err(|e| XlogError::Kernel(format!("zero queue: {}", e)))?;
    device
        .memset_zeros(&mut queue_ready)
        .map_err(|e| XlogError::Kernel(format!("zero queue_ready: {}", e)))?;
    device
        .memset_zeros(&mut head)
        .map_err(|e| XlogError::Kernel(format!("zero head: {}", e)))?;
    device
        .memset_zeros(&mut tail)
        .map_err(|e| XlogError::Kernel(format!("zero tail: {}", e)))?;
    device
        .memset_zeros(&mut in_flight)
        .map_err(|e| XlogError::Kernel(format!("zero in_flight: {}", e)))?;
    device
        .memset_zeros(&mut leaf_used)
        .map_err(|e| XlogError::Kernel(format!("zero leaf_used: {}", e)))?;
    device
        .memset_zeros(&mut choice_used)
        .map_err(|e| XlogError::Kernel(format!("zero choice_used: {}", e)))?;
    device
        .memset_zeros(&mut leaf_var)
        .map_err(|e| XlogError::Kernel(format!("zero leaf_var: {}", e)))?;
    device
        .memset_zeros(&mut choice_var)
        .map_err(|e| XlogError::Kernel(format!("zero choice_var: {}", e)))?;
    device
        .memset_zeros(&mut node_needs_var)
        .map_err(|e| XlogError::Kernel(format!("zero node_needs_var: {}", e)))?;
    device
        .memset_zeros(&mut node_var)
        .map_err(|e| XlogError::Kernel(format!("zero node_var: {}", e)))?;
    device
        .memset_zeros(&mut clause_counts)
        .map_err(|e| XlogError::Kernel(format!("zero clause_counts: {}", e)))?;
    device
        .memset_zeros(&mut lit_counts)
        .map_err(|e| XlogError::Kernel(format!("zero lit_counts: {}", e)))?;

    let reach_init_fn = device
        .get_func(CNF_MODULE, cnf_kernels::CNF_REACHABILITY_INIT)
        .ok_or_else(|| XlogError::Kernel("cnf_reachability_init kernel not found".to_string()))?;
    let reach_bfs_fn = device
        .get_func(CNF_MODULE, cnf_kernels::CNF_REACHABILITY_BFS)
        .ok_or_else(|| XlogError::Kernel("cnf_reachability_bfs kernel not found".to_string()))?;
    let mark_leaf_choice_fn = device
        .get_func(CNF_MODULE, cnf_kernels::CNF_MARK_LEAF_CHOICE)
        .ok_or_else(|| XlogError::Kernel("cnf_mark_leaf_choice kernel not found".to_string()))?;
    let assign_leaf_var_fn = device
        .get_func(CNF_MODULE, cnf_kernels::CNF_ASSIGN_LEAF_VAR)
        .ok_or_else(|| XlogError::Kernel("cnf_assign_leaf_var kernel not found".to_string()))?;
    let assign_choice_var_fn = device
        .get_func(CNF_MODULE, cnf_kernels::CNF_ASSIGN_CHOICE_VAR)
        .ok_or_else(|| XlogError::Kernel("cnf_assign_choice_var kernel not found".to_string()))?;
    let mark_node_vars_fn = device
        .get_func(CNF_MODULE, cnf_kernels::CNF_MARK_NODE_VARS)
        .ok_or_else(|| XlogError::Kernel("cnf_mark_node_vars kernel not found".to_string()))?;
    let count_clauses_fn = device
        .get_func(CNF_MODULE, cnf_kernels::CNF_COUNT_CLAUSES)
        .ok_or_else(|| XlogError::Kernel("cnf_count_clauses kernel not found".to_string()))?;
    let capture_last_fn = device
        .get_func(CNF_MODULE, cnf_kernels::CNF_CAPTURE_LAST_COUNTS)
        .ok_or_else(|| XlogError::Kernel("cnf_capture_last_counts kernel not found".to_string()))?;
    let leaf_choice_totals_fn = device
        .get_func(CNF_MODULE, cnf_kernels::CNF_COMPUTE_LEAF_CHOICE_TOTALS)
        .ok_or_else(|| {
            XlogError::Kernel("cnf_compute_leaf_choice_totals kernel not found".to_string())
        })?;
    let compute_totals_fn = device
        .get_func(CNF_MODULE, cnf_kernels::CNF_COMPUTE_TOTALS)
        .ok_or_else(|| XlogError::Kernel("cnf_compute_totals kernel not found".to_string()))?;
    let assign_node_var_fn = device
        .get_func(CNF_MODULE, cnf_kernels::CNF_ASSIGN_NODE_VAR)
        .ok_or_else(|| XlogError::Kernel("cnf_assign_node_var kernel not found".to_string()))?;
    let emit_clauses_fn = device
        .get_func(CNF_MODULE, cnf_kernels::CNF_EMIT_CLAUSES)
        .ok_or_else(|| XlogError::Kernel("cnf_emit_clauses kernel not found".to_string()))?;
    let set_clause_end_fn = device
        .get_func(CNF_MODULE, cnf_kernels::CNF_SET_CLAUSE_END)
        .ok_or_else(|| XlogError::Kernel("cnf_set_clause_end kernel not found".to_string()))?;

    let block = 256u32;

    let grid_roots = grid_dim(num_roots_u32, block);
    unsafe {
        reach_init_fn.clone().launch(
            LaunchConfig {
                grid_dim: (grid_roots, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                &roots.roots,
                num_roots_u32,
                num_nodes_u32,
                &mut reachable,
                &mut queue,
                &mut queue_ready,
                &mut head,
                &mut tail,
                &mut in_flight,
            ),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("cnf_reachability_init failed: {}", e)))?;

    let grid_nodes = grid_dim(num_nodes_u32, block);
    unsafe {
        reach_bfs_fn.clone().launch(
            LaunchConfig {
                grid_dim: (grid_nodes, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                &pir.node_type,
                &pir.child_offsets,
                &pir.children,
                &pir.decision_child_false,
                &pir.decision_child_true,
                num_nodes_u32,
                &mut reachable,
                &mut queue,
                &mut queue_ready,
                &mut head,
                &mut tail,
                &mut in_flight,
            ),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("cnf_reachability_bfs failed: {}", e)))?;

    unsafe {
        mark_leaf_choice_fn.clone().launch(
            LaunchConfig {
                grid_dim: (grid_nodes, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                &pir.node_type,
                &pir.leaf_id,
                &pir.decision_var,
                &reachable,
                num_nodes_u32,
                leaf_cap,
                choice_cap,
                &mut leaf_used,
                &mut choice_used,
            ),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("cnf_mark_leaf_choice failed: {}", e)))?;

    if leaf_cap > 0 {
        device
            .dtod_copy(&leaf_used, &mut leaf_prefix)
            .map_err(|e| XlogError::Kernel(format!("copy leaf_used: {}", e)))?;
        provider.exclusive_scan_u32_inplace(&mut leaf_prefix, leaf_cap)?;
    }
    if choice_cap > 0 {
        device
            .dtod_copy(&choice_used, &mut choice_prefix)
            .map_err(|e| XlogError::Kernel(format!("copy choice_used: {}", e)))?;
        provider.exclusive_scan_u32_inplace(&mut choice_prefix, choice_cap)?;
    }

    unsafe {
        leaf_choice_totals_fn.clone().launch(
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                &leaf_prefix,
                &leaf_used,
                leaf_cap,
                &choice_prefix,
                &choice_used,
                choice_cap,
                &mut num_leaf,
                &mut num_choice,
                &mut base_choice,
                &mut base_node,
                &mut decision_var_limit,
            ),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("cnf_compute_leaf_choice_totals failed: {}", e)))?;

    if leaf_cap > 0 {
        let grid_leaf = grid_dim(leaf_cap, block);
        unsafe {
            assign_leaf_var_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_leaf, 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&leaf_used, &leaf_prefix, leaf_cap, &mut leaf_var),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("cnf_assign_leaf_var failed: {}", e)))?;
    }
    if choice_cap > 0 {
        let grid_choice = grid_dim(choice_cap, block);
        unsafe {
            assign_choice_var_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_choice, 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &choice_used,
                    &choice_prefix,
                    choice_cap,
                    &base_choice,
                    &mut choice_var,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("cnf_assign_choice_var failed: {}", e)))?;
    }

    unsafe {
        mark_node_vars_fn.clone().launch(
            LaunchConfig {
                grid_dim: (grid_nodes, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                &pir.node_type,
                &reachable,
                num_nodes_u32,
                &mut node_needs_var,
            ),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("cnf_mark_node_vars failed: {}", e)))?;

    unsafe {
        count_clauses_fn.clone().launch(
            LaunchConfig {
                grid_dim: (grid_nodes, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                &pir.node_type,
                &pir.child_offsets,
                &reachable,
                num_nodes_u32,
                &mut clause_counts,
                &mut lit_counts,
            ),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("cnf_count_clauses failed: {}", e)))?;

    unsafe {
        capture_last_fn.clone().launch(
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                &node_needs_var,
                &clause_counts,
                &lit_counts,
                num_nodes_u32,
                &mut node_last,
                &mut clause_last,
                &mut lit_last,
            ),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("cnf_capture_last_counts failed: {}", e)))?;

    provider.exclusive_scan_u32_inplace(&mut node_needs_var, num_nodes_u32)?;
    provider.exclusive_scan_u32_inplace(&mut clause_counts, num_nodes_u32)?;
    provider.exclusive_scan_u32_inplace(&mut lit_counts, num_nodes_u32)?;

    let mut totals_params: Vec<*mut c_void> = vec![
        (&node_needs_var).as_kernel_param(),
        (&clause_counts).as_kernel_param(),
        (&lit_counts).as_kernel_param(),
        (&node_last).as_kernel_param(),
        (&clause_last).as_kernel_param(),
        (&lit_last).as_kernel_param(),
        num_nodes_u32.as_kernel_param(),
        (&base_node).as_kernel_param(),
        var_cap.as_kernel_param(),
        clause_cap.as_kernel_param(),
        lit_cap.as_kernel_param(),
        (&mut d_num_vars).as_kernel_param(),
        (&mut d_num_clauses).as_kernel_param(),
        (&mut d_num_lits).as_kernel_param(),
    ];
    unsafe {
        compute_totals_fn.clone().launch(
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            },
            &mut totals_params,
        )
    }
    .map_err(|e| XlogError::Kernel(format!("cnf_compute_totals failed: {}", e)))?;

    unsafe {
        assign_node_var_fn.clone().launch(
            LaunchConfig {
                grid_dim: (grid_nodes, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                &pir.node_type,
                &pir.leaf_id,
                &reachable,
                &node_needs_var,
                &base_node,
                num_nodes_u32,
                leaf_cap,
                &leaf_var,
                &mut node_var,
            ),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("cnf_assign_node_var failed: {}", e)))?;

    let mut emit_params: Vec<*mut c_void> = vec![
        (&pir.node_type).as_kernel_param(),
        (&pir.child_offsets).as_kernel_param(),
        (&pir.children).as_kernel_param(),
        (&pir.leaf_id).as_kernel_param(),
        (&pir.decision_var).as_kernel_param(),
        (&pir.decision_child_false).as_kernel_param(),
        (&pir.decision_child_true).as_kernel_param(),
        (&reachable).as_kernel_param(),
        (&node_var).as_kernel_param(),
        (&leaf_var).as_kernel_param(),
        (&choice_var).as_kernel_param(),
        (&clause_counts).as_kernel_param(),
        (&lit_counts).as_kernel_param(),
        num_nodes_u32.as_kernel_param(),
        leaf_cap.as_kernel_param(),
        choice_cap.as_kernel_param(),
        (&mut d_offsets).as_kernel_param(),
        (&mut d_lits).as_kernel_param(),
    ];

    unsafe {
        emit_clauses_fn.clone().launch(
            LaunchConfig {
                grid_dim: (grid_nodes, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            },
            &mut emit_params,
        )
    }
    .map_err(|e| XlogError::Kernel(format!("cnf_emit_clauses failed: {}", e)))?;

    unsafe {
        set_clause_end_fn.clone().launch(
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            },
            (&mut d_offsets, &d_num_clauses, &d_num_lits),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("cnf_set_clause_end failed: {}", e)))?;

    provider.device().synchronize()?;

    Ok(GpuCnfEncoding {
        cnf: GpuCnf {
            var_cap,
            clause_cap,
            lit_cap,
            num_vars: d_num_vars,
            num_clauses: d_num_clauses,
            num_lits: d_num_lits,
            clause_offsets: d_offsets,
            literals: d_lits,
        },
        vars: GpuCnfVarTables {
            node_var,
            leaf_var,
            choice_var,
            max_var: var_cap,
        },
        decision_var_limit,
    })
}
