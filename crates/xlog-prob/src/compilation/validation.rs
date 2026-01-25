//! GPU-native equivalence validation (φ ≡ C) using the GPU CDCL verifier.

use std::sync::Arc;

use std::ffi::c_void;

use cudarc::driver::{DeviceRepr, DeviceSlice, LaunchAsync, LaunchConfig};
use xlog_core::{Result, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::provider::sat_kernels;
use xlog_cuda::provider::SAT_MODULE;
use xlog_cuda::CudaKernelProvider;
use xlog_solve::{GpuCdclConfig, GpuCdclSolver, GpuCnf};

use crate::gpu::GpuXgcf;

#[derive(Debug, Clone, Copy)]
pub struct GpuEquivalenceConfig {
    pub cdcl: GpuCdclConfig,
}

impl Default for GpuEquivalenceConfig {
    fn default() -> Self {
        Self {
            cdcl: GpuCdclConfig::default(),
        }
    }
}

struct CircuitCnf {
    cnf: GpuCnf,
    /// Exclusive prefix sum over `is_internal(node)` (len = num_nodes).
    /// Used to map internal node ids -> Tseitin vars in kernels.
    internal_prefix: TrackedCudaSlice<u32>,
}

fn build_circuit_cnf(provider: &Arc<CudaKernelProvider>, circuit: &GpuXgcf, base_num_vars: u32) -> Result<CircuitCnf> {
    if base_num_vars == 0 {
        return Err(XlogError::Compilation(
            "GPU equivalence verifier requires base_num_vars > 0".to_string(),
        ));
    }
    if circuit.max_var() > base_num_vars {
        return Err(XlogError::Compilation(format!(
            "Circuit references var {} but base CNF has only {} vars",
            circuit.max_var(),
            base_num_vars
        )));
    }

    let num_nodes = circuit.num_nodes();
    if num_nodes == 0 {
        return Err(XlogError::Compilation(
            "GPU equivalence verifier requires circuit with num_nodes > 0".to_string(),
        ));
    }
    if circuit.root() as usize >= num_nodes {
        return Err(XlogError::Compilation(format!(
            "GPU equivalence verifier: circuit root {} out of bounds (num_nodes={})",
            circuit.root(),
            num_nodes
        )));
    }

    let num_nodes_u32 = u32::try_from(num_nodes).map_err(|_| {
        XlogError::Compilation(format!(
            "GPU equivalence verifier: circuit num_nodes {} exceeds u32::MAX",
            num_nodes
        ))
    })?;

    // Safe, host-known upper bounds (no device->host reads required).
    let num_edges = circuit.child_indices().len();
    let n64 = num_nodes as u64;
    let e64 = num_edges as u64;

    let var_cap = u32::try_from((base_num_vars as u64).saturating_add(n64)).map_err(|_| {
        XlogError::Kernel("Circuit CNF var capacity exceeds u32::MAX".to_string())
    })?;
    let clause_cap = u32::try_from(
        e64.checked_add(4u64.saturating_mul(n64))
            .ok_or_else(|| XlogError::Kernel("Circuit CNF clause capacity overflow".to_string()))?,
    )
    .map_err(|_| XlogError::Kernel("Circuit CNF clause capacity exceeds u32::MAX".to_string()))?;
    let lit_cap = u32::try_from(
        (3u64.saturating_mul(e64))
            .checked_add(12u64.saturating_mul(n64))
            .ok_or_else(|| XlogError::Kernel("Circuit CNF literal capacity overflow".to_string()))?,
    )
    .map_err(|_| XlogError::Kernel("Circuit CNF literal capacity exceeds u32::MAX".to_string()))?;

    let memory = provider.memory();
    let device = provider.device().inner();

    // Per-node count arrays (len = num_nodes) used for exclusive scans.
    let mut internal_prefix = memory.alloc::<u32>(num_nodes)?;
    let mut clause_base = memory.alloc::<u32>(num_nodes)?;
    let mut lit_base = memory.alloc::<u32>(num_nodes)?;

    let counts_fn = device
        .get_func(SAT_MODULE, sat_kernels::SAT_XGCF_CNF_COUNTS)
        .ok_or_else(|| XlogError::Kernel("sat_xgcf_cnf_counts kernel not found".to_string()))?;

    let block = 256u32;
    let mut grid = (num_nodes_u32 + block - 1) / block;
    if grid == 0 {
        grid = 1;
    }
    if grid > 65_535 {
        grid = 65_535;
    }

    // SAFETY: sat_xgcf_cnf_counts(node_type, child_offsets, num_nodes, internal_counts, clause_counts, lit_counts)
    unsafe {
        counts_fn.clone().launch(
            LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                circuit.node_type(),
                circuit.child_offsets(),
                num_nodes_u32,
                &mut internal_prefix,
                &mut clause_base,
                &mut lit_base,
            ),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("sat_xgcf_cnf_counts failed: {}", e)))?;

    // Capture last elements before scans overwrite them.
    let mut internal_last = memory.alloc::<u32>(1)?;
    let mut clause_last = memory.alloc::<u32>(1)?;
    let mut lit_last = memory.alloc::<u32>(1)?;

    let capture_last_fn = device
        .get_func(SAT_MODULE, sat_kernels::SAT_XGCF_CNF_CAPTURE_LAST_COUNTS)
        .ok_or_else(|| {
            XlogError::Kernel("sat_xgcf_cnf_capture_last_counts kernel not found".to_string())
        })?;
    // SAFETY: sat_xgcf_cnf_capture_last_counts(internal_counts, clause_counts, lit_counts, num_nodes, out_internal_last, out_clause_last, out_lit_last)
    unsafe {
        capture_last_fn
            .clone()
            .launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &internal_prefix,
                    &clause_base,
                    &lit_base,
                    num_nodes_u32,
                    &mut internal_last,
                    &mut clause_last,
                    &mut lit_last,
                ),
            )
    }
    .map_err(|e| XlogError::Kernel(format!("sat_xgcf_cnf_capture_last_counts failed: {}", e)))?;

    provider.exclusive_scan_u32_inplace(&mut internal_prefix, num_nodes_u32)?;
    provider.exclusive_scan_u32_inplace(&mut clause_base, num_nodes_u32)?;
    provider.exclusive_scan_u32_inplace(&mut lit_base, num_nodes_u32)?;
    provider.device().synchronize()?;

    // Output CNF buffers + device-resident meta.
    let mut d_num_vars = memory.alloc::<u32>(1)?;
    let mut d_num_clauses = memory.alloc::<u32>(1)?;
    let mut d_num_lits = memory.alloc::<u32>(1)?;
    let mut d_offsets = memory.alloc::<u32>((clause_cap as usize) + 1)?;
    let mut d_lits = memory.alloc::<i32>(lit_cap as usize)?;

    let totals_fn = device
        .get_func(SAT_MODULE, sat_kernels::SAT_XGCF_CNF_COMPUTE_TOTALS)
        .ok_or_else(|| XlogError::Kernel("sat_xgcf_cnf_compute_totals kernel not found".to_string()))?;
    // SAFETY: sat_xgcf_cnf_compute_totals(internal_prefix, clause_base, lit_base, internal_last*, clause_last*, lit_last*, num_nodes, base_num_vars, clause_cap, lit_cap, out_num_vars*, out_num_clauses*, out_num_lits*)
    let mut totals_params: Vec<*mut c_void> = vec![
        (&internal_prefix).as_kernel_param(),
        (&clause_base).as_kernel_param(),
        (&lit_base).as_kernel_param(),
        (&internal_last).as_kernel_param(),
        (&clause_last).as_kernel_param(),
        (&lit_last).as_kernel_param(),
        num_nodes_u32.as_kernel_param(),
        base_num_vars.as_kernel_param(),
        clause_cap.as_kernel_param(),
        lit_cap.as_kernel_param(),
        (&mut d_num_vars).as_kernel_param(),
        (&mut d_num_clauses).as_kernel_param(),
        (&mut d_num_lits).as_kernel_param(),
    ];
    unsafe {
        totals_fn.clone().launch(
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            },
            &mut totals_params,
        )
    }
    .map_err(|e| XlogError::Kernel(format!("sat_xgcf_cnf_compute_totals failed: {}", e)))?;

    let emit_fn = device
        .get_func(SAT_MODULE, sat_kernels::SAT_XGCF_CNF_EMIT)
        .ok_or_else(|| XlogError::Kernel("sat_xgcf_cnf_emit kernel not found".to_string()))?;

    // sat_xgcf_cnf_emit(...) exceeds cudarc's tuple-arity impls for LaunchAsync, so launch with
    // an explicit parameter list.
    let mut params: Vec<*mut c_void> = vec![
        circuit.node_type().as_kernel_param(),
        circuit.child_offsets().as_kernel_param(),
        circuit.child_indices().as_kernel_param(),
        circuit.lit().as_kernel_param(),
        circuit.decision_var().as_kernel_param(),
        circuit.decision_child_false().as_kernel_param(),
        circuit.decision_child_true().as_kernel_param(),
        (&internal_prefix).as_kernel_param(),
        (&clause_base).as_kernel_param(),
        (&lit_base).as_kernel_param(),
        base_num_vars.as_kernel_param(),
        num_nodes_u32.as_kernel_param(),
        (&mut d_offsets).as_kernel_param(),
        (&mut d_lits).as_kernel_param(),
    ];

    unsafe {
        emit_fn.clone().launch(
            LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            },
            &mut params,
        )
    }
    .map_err(|e| XlogError::Kernel(format!("sat_xgcf_cnf_emit failed: {}", e)))?;

    // sat_xgcf_cnf_emit does not write the CSR terminator; finalize deterministically on device.
    let term_fn = device
        .get_func(SAT_MODULE, sat_kernels::SAT_CNF_WRITE_TERMINATOR)
        .ok_or_else(|| XlogError::Kernel("sat_cnf_write_terminator kernel not found".to_string()))?;
    // SAFETY: sat_cnf_write_terminator(out_offsets, num_clauses*, num_lits*)
    unsafe {
        term_fn.clone().launch(
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            },
            (&mut d_offsets, &d_num_clauses, &d_num_lits),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("sat_cnf_write_terminator failed: {}", e)))?;
    provider.device().synchronize()?;

    Ok(CircuitCnf {
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
        internal_prefix,
    })
}

fn build_phi_and_not_c(
    provider: &Arc<CudaKernelProvider>,
    phi: &GpuCnf,
    circuit: &GpuXgcf,
    circuit_cnf: &CircuitCnf,
) -> Result<GpuCnf> {
    let device = provider.device().inner();
    let memory = provider.memory();

    let phi_clauses = phi.clause_cap;
    let phi_lits = phi.lit_cap;

    let clause_cap = u32::try_from(
        (phi_clauses as u64)
            .checked_add(circuit_cnf.cnf.clause_cap as u64)
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| XlogError::Kernel("phi ∧ ¬C clause capacity overflow".to_string()))?,
    )
    .map_err(|_| XlogError::Kernel("phi ∧ ¬C clause capacity exceeds u32::MAX".to_string()))?;
    let lit_cap = u32::try_from(
        (phi_lits as u64)
            .checked_add(circuit_cnf.cnf.lit_cap as u64)
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| XlogError::Kernel("phi ∧ ¬C literal capacity overflow".to_string()))?,
    )
    .map_err(|_| XlogError::Kernel("phi ∧ ¬C literal capacity exceeds u32::MAX".to_string()))?;

    let var_cap = circuit_cnf.cnf.var_cap;

    let mut out_num_vars = memory.alloc::<u32>(1)?;
    let mut out_num_clauses = memory.alloc::<u32>(1)?;
    let mut out_num_lits = memory.alloc::<u32>(1)?;
    let mut d_unused0 = memory.alloc::<u32>(1)?;
    let mut d_unused1 = memory.alloc::<u32>(1)?;
    let mut d_unused2 = memory.alloc::<u32>(1)?;

    let mut out_offsets = memory.alloc::<u32>((clause_cap as usize) + 1)?;
    let mut out_lits = memory.alloc::<i32>(lit_cap as usize)?;

    // Copy phi into the front.
    {
        let mut dst = out_offsets.slice_mut(0..((phi_clauses as usize) + 1));
        device
            .dtod_copy(&phi.clause_offsets, &mut dst)
            .map_err(|e| XlogError::Kernel(format!("Failed to copy phi offsets: {}", e)))?;
    }
    if phi_lits > 0 {
        let mut dst = out_lits.slice_mut(0..(phi_lits as usize));
        device
            .dtod_copy(&phi.literals, &mut dst)
            .map_err(|e| XlogError::Kernel(format!("Failed to copy phi lits: {}", e)))?;
    }

    // Copy CNF(C) literals after phi (capacity copy; solver reads only up to device-resident totals).
    if circuit_cnf.cnf.lit_cap > 0 {
        let start = phi_lits as usize;
        let end = start + (circuit_cnf.cnf.lit_cap as usize);
        let mut dst = out_lits.slice_mut(start..end);
        device
            .dtod_copy(&circuit_cnf.cnf.literals, &mut dst)
            .map_err(|e| XlogError::Kernel(format!("Failed to copy CNF(C) lits: {}", e)))?;
    }

    // Shift and write CNF(C) offsets after phi (dynamic length via device-resident c_num_clauses).
    let shift_fn = device
        .get_func(SAT_MODULE, sat_kernels::SAT_SHIFT_OFFSETS)
        .ok_or_else(|| XlogError::Kernel("sat_shift_offsets kernel not found".to_string()))?;
    let add = phi_lits;
    let dst_base = phi_clauses;
    let n_cap = circuit_cnf.cnf.clause_cap.saturating_add(1);

    let block = 256u32;
    let mut grid = (n_cap + block - 1) / block;
    if grid == 0 {
        grid = 1;
    }
    if grid > 65_535 {
        grid = 65_535;
    }

    // SAFETY: sat_shift_offsets(src_offsets, src_num_clauses*, add, dst_base, dst_offsets)
    unsafe {
        shift_fn
            .clone()
            .launch(
                LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &circuit_cnf.cnf.clause_offsets,
                    &circuit_cnf.cnf.num_clauses,
                    add,
                    dst_base,
                    &mut out_offsets,
                ),
            )
    }
    .map_err(|e| XlogError::Kernel(format!("sat_shift_offsets failed: {}", e)))?;

    // Finalize: append unit clause forcing root false + write device-resident totals for the combined query.
    let unit_fn = device
        .get_func(SAT_MODULE, sat_kernels::SAT_XGCF_WRITE_ROOT_UNIT_CLAUSE)
        .ok_or_else(|| {
            XlogError::Kernel("sat_xgcf_write_root_unit_clause kernel not found".to_string())
        })?;
    let mut params: Vec<*mut c_void> = vec![
        circuit.node_type().as_kernel_param(),
        circuit.lit().as_kernel_param(),
        (&circuit_cnf.internal_prefix).as_kernel_param(),
        phi.var_cap.as_kernel_param(),
        circuit.root().as_kernel_param(),
        0i32.as_kernel_param(), // force_false
        phi_clauses.as_kernel_param(),
        phi_lits.as_kernel_param(),
        (&circuit_cnf.cnf.num_vars).as_kernel_param(),
        (&circuit_cnf.cnf.num_clauses).as_kernel_param(),
        (&circuit_cnf.cnf.num_lits).as_kernel_param(),
        0u32.as_kernel_param(), // extra_num_vars
        0u32.as_kernel_param(), // extra_num_clauses
        0u32.as_kernel_param(), // extra_num_lits
        var_cap.as_kernel_param(),
        clause_cap.as_kernel_param(),
        lit_cap.as_kernel_param(),
        (&mut out_num_vars).as_kernel_param(),
        (&mut out_num_clauses).as_kernel_param(),
        (&mut out_num_lits).as_kernel_param(),
        (&mut d_unused0).as_kernel_param(),
        (&mut d_unused1).as_kernel_param(),
        (&mut d_unused2).as_kernel_param(),
        (&mut out_offsets).as_kernel_param(),
        (&mut out_lits).as_kernel_param(),
    ];

    unsafe {
        unit_fn
            .clone()
            .launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                &mut params,
            )
    }
    .map_err(|e| XlogError::Kernel(format!("sat_xgcf_write_root_unit_clause failed: {}", e)))?;

    provider.device().synchronize()?;

    Ok(GpuCnf {
        var_cap,
        clause_cap,
        lit_cap,
        num_vars: out_num_vars,
        num_clauses: out_num_clauses,
        num_lits: out_num_lits,
        clause_offsets: out_offsets,
        literals: out_lits,
    })
}

fn build_c_and_not_phi(
    provider: &Arc<CudaKernelProvider>,
    phi: &GpuCnf,
    circuit: &GpuXgcf,
    circuit_cnf: &CircuitCnf,
) -> Result<GpuCnf> {
    let device = provider.device().inner();
    let memory = provider.memory();

    let m = phi.clause_cap;
    let l = phi.lit_cap;

    // ¬phi encoding:
    // clauses_notphi = sum(len_j + 1) + 1 = L + m + 1
    // lits_notphi = sum(3*len_j + 1) + m = 3L + 2m
    let notphi_clauses = u32::try_from(
        (l as u64)
            .checked_add(m as u64)
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| XlogError::Kernel("¬phi clause count overflow".to_string()))?,
    )
    .map_err(|_| XlogError::Kernel("¬phi clause count exceeds u32::MAX".to_string()))?;
    let notphi_lits = u32::try_from(
        (l as u64)
            .checked_mul(3)
            .and_then(|v| v.checked_add(2u64.saturating_mul(m as u64)))
            .ok_or_else(|| XlogError::Kernel("¬phi literal count overflow".to_string()))?,
    )
    .map_err(|_| XlogError::Kernel("¬phi literal count exceeds u32::MAX".to_string()))?;

    let var_cap = circuit_cnf
        .cnf
        .var_cap
        .checked_add(m)
        .ok_or_else(|| XlogError::Kernel("C ∧ ¬phi var capacity overflow".to_string()))?;
    let clause_cap = u32::try_from(
        (circuit_cnf.cnf.clause_cap as u64)
            .checked_add(1)
            .and_then(|v| v.checked_add(notphi_clauses as u64))
            .ok_or_else(|| XlogError::Kernel("C ∧ ¬phi clause capacity overflow".to_string()))?,
    )
    .map_err(|_| XlogError::Kernel("C ∧ ¬phi clause capacity exceeds u32::MAX".to_string()))?;
    let lit_cap = u32::try_from(
        (circuit_cnf.cnf.lit_cap as u64)
            .checked_add(1)
            .and_then(|v| v.checked_add(notphi_lits as u64))
            .ok_or_else(|| XlogError::Kernel("C ∧ ¬phi literal capacity overflow".to_string()))?,
    )
    .map_err(|_| XlogError::Kernel("C ∧ ¬phi literal capacity exceeds u32::MAX".to_string()))?;

    let mut out_num_vars = memory.alloc::<u32>(1)?;
    let mut out_num_clauses = memory.alloc::<u32>(1)?;
    let mut out_num_lits = memory.alloc::<u32>(1)?;

    let mut d_unsat_var_base = memory.alloc::<u32>(1)?;
    let mut d_notphi_clause_base = memory.alloc::<u32>(1)?;
    let mut d_notphi_lit_base = memory.alloc::<u32>(1)?;

    let mut out_offsets = memory.alloc::<u32>((clause_cap as usize) + 1)?;
    let mut out_lits = memory.alloc::<i32>(lit_cap as usize)?;

    // Copy CNF(C) into the front (capacity copy; all query indices are computed from device-resident totals).
    if circuit_cnf.cnf.clause_cap > 0 {
        let mut dst = out_offsets.slice_mut(0..((circuit_cnf.cnf.clause_cap as usize) + 1));
        device
            .dtod_copy(&circuit_cnf.cnf.clause_offsets, &mut dst)
            .map_err(|e| XlogError::Kernel(format!("Failed to copy CNF(C) offsets: {}", e)))?;
    }
    if circuit_cnf.cnf.lit_cap > 0 {
        let mut dst = out_lits.slice_mut(0..(circuit_cnf.cnf.lit_cap as usize));
        device
            .dtod_copy(&circuit_cnf.cnf.literals, &mut dst)
            .map_err(|e| XlogError::Kernel(format!("Failed to copy CNF(C) lits: {}", e)))?;
    }

    // Prepare: insert unit clause forcing root true and compute device-resident totals / bases.
    let unit_fn = device
        .get_func(SAT_MODULE, sat_kernels::SAT_XGCF_WRITE_ROOT_UNIT_CLAUSE)
        .ok_or_else(|| {
            XlogError::Kernel("sat_xgcf_write_root_unit_clause kernel not found".to_string())
        })?;

    let mut params: Vec<*mut c_void> = vec![
        circuit.node_type().as_kernel_param(),
        circuit.lit().as_kernel_param(),
        (&circuit_cnf.internal_prefix).as_kernel_param(),
        phi.var_cap.as_kernel_param(),
        circuit.root().as_kernel_param(),
        1i32.as_kernel_param(), // force_true
        0u32.as_kernel_param(), // clause_base
        0u32.as_kernel_param(), // lit_base
        (&circuit_cnf.cnf.num_vars).as_kernel_param(),
        (&circuit_cnf.cnf.num_clauses).as_kernel_param(),
        (&circuit_cnf.cnf.num_lits).as_kernel_param(),
        m.as_kernel_param(),              // extra_num_vars (u_j vars)
        notphi_clauses.as_kernel_param(), // extra_num_clauses
        notphi_lits.as_kernel_param(),    // extra_num_lits
        var_cap.as_kernel_param(),
        clause_cap.as_kernel_param(),
        lit_cap.as_kernel_param(),
        (&mut out_num_vars).as_kernel_param(),
        (&mut out_num_clauses).as_kernel_param(),
        (&mut out_num_lits).as_kernel_param(),
        (&mut d_unsat_var_base).as_kernel_param(),
        (&mut d_notphi_clause_base).as_kernel_param(),
        (&mut d_notphi_lit_base).as_kernel_param(),
        (&mut out_offsets).as_kernel_param(),
        (&mut out_lits).as_kernel_param(),
    ];

    unsafe {
        unit_fn
            .clone()
            .launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                &mut params,
            )
    }
    .map_err(|e| XlogError::Kernel(format!("sat_xgcf_write_root_unit_clause failed: {}", e)))?;

    // Emit ¬phi encoding after CNF(C) + unit using device-resident base indices.
    let not_phi_fn = device
        .get_func(SAT_MODULE, sat_kernels::SAT_EMIT_NOT_PHI)
        .ok_or_else(|| XlogError::Kernel("sat_emit_not_phi kernel not found".to_string()))?;

    let block = 256u32;
    let mut grid = (m + block - 1) / block;
    if grid == 0 {
        grid = 1;
    }
    if grid > 65_535 {
        grid = 65_535;
    }

    // SAFETY: sat_emit_not_phi(phi_offsets, phi_lits, phi_num_clauses*, unsat_var_base*, out_clause_base*, out_lit_base*, out_offsets, out_lits)
    unsafe {
        not_phi_fn
            .clone()
            .launch(
                LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &phi.clause_offsets,
                    &phi.literals,
                    &phi.num_clauses,
                    &d_unsat_var_base,
                    &d_notphi_clause_base,
                    &d_notphi_lit_base,
                    &mut out_offsets,
                    &mut out_lits,
                ),
            )
    }
    .map_err(|e| XlogError::Kernel(format!("sat_emit_not_phi failed: {}", e)))?;

    provider.device().synchronize()?;

    Ok(GpuCnf {
        var_cap,
        clause_cap,
        lit_cap,
        num_vars: out_num_vars,
        num_clauses: out_num_clauses,
        num_lits: out_num_lits,
        clause_offsets: out_offsets,
        literals: out_lits,
    })
}

pub fn check_equivalence_gpu(
    phi: &GpuCnf,
    circuit: &GpuXgcf,
    provider: &Arc<CudaKernelProvider>,
    config: GpuEquivalenceConfig,
) -> Result<()> {
    let circuit_cnf = build_circuit_cnf(provider, circuit, phi.var_cap)?;

    let q1 = build_phi_and_not_c(provider, phi, circuit, &circuit_cnf)?;
    let q2 = build_c_and_not_phi(provider, phi, circuit, &circuit_cnf)?;

    let solver = GpuCdclSolver::new(provider.clone(), config.cdcl);
    solver.solve_expect_unsat(&q1)?;
    solver.solve_expect_unsat(&q2)?;
    Ok(())
}

pub fn validate_equivalence_gpu(
    phi: &GpuCnf,
    circuit: &GpuXgcf,
    provider: &Arc<CudaKernelProvider>,
    config: GpuEquivalenceConfig,
) -> Result<()> {
    check_equivalence_gpu(phi, circuit, provider, config)
}
