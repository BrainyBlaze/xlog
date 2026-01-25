//! GPU-native equivalence validation (φ ≡ C) using the GPU CDCL verifier.

use std::sync::Arc;

use std::ffi::c_void;

use cudarc::driver::{DeviceRepr, DeviceSlice, LaunchAsync, LaunchConfig};
use xlog_core::{Result, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::provider::sat_kernels;
use xlog_cuda::provider::SAT_MODULE;
use xlog_cuda::CudaKernelProvider;
use xlog_solve::{GpuCdclConfig, GpuCdclResult, GpuCdclSolver, GpuCnf, GpuSolveStatus};

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

#[derive(Debug)]
pub struct GpuEquivalenceCheck {
    pub phi_and_not_c: GpuCdclResult,
    pub c_and_not_phi: GpuCdclResult,
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

    let memory = provider.memory();
    let device = provider.device().inner();

    let mut internal_counts = memory.alloc::<u32>(num_nodes)?;
    let mut clause_counts = memory.alloc::<u32>(num_nodes)?;
    let mut lit_counts = memory.alloc::<u32>(num_nodes)?;

    let counts_fn = device
        .get_func(SAT_MODULE, sat_kernels::SAT_XGCF_CNF_COUNTS)
        .ok_or_else(|| XlogError::Kernel("sat_xgcf_cnf_counts kernel not found".to_string()))?;

    let block = 256u32;
    let mut grid = ((num_nodes as u32) + block - 1) / block;
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
                num_nodes as u32,
                &mut internal_counts,
                &mut clause_counts,
                &mut lit_counts,
            ),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("sat_xgcf_cnf_counts failed: {}", e)))?;
    provider.device().synchronize()?;

    let last_idx = (num_nodes - 1)..num_nodes;

    let mut internal_last = [0u32];
    let mut clause_last = [0u32];
    let mut lit_last = [0u32];
    device
        .dtoh_sync_copy_into(&internal_counts.slice(last_idx.clone()), &mut internal_last)
        .map_err(|e| XlogError::Kernel(format!("Failed to read internal_counts last: {}", e)))?;
    device
        .dtoh_sync_copy_into(&clause_counts.slice(last_idx.clone()), &mut clause_last)
        .map_err(|e| XlogError::Kernel(format!("Failed to read clause_counts last: {}", e)))?;
    device
        .dtoh_sync_copy_into(&lit_counts.slice(last_idx.clone()), &mut lit_last)
        .map_err(|e| XlogError::Kernel(format!("Failed to read lit_counts last: {}", e)))?;

    provider.exclusive_scan_u32_inplace(&mut internal_counts, num_nodes as u32)?;
    provider.exclusive_scan_u32_inplace(&mut clause_counts, num_nodes as u32)?;
    provider.exclusive_scan_u32_inplace(&mut lit_counts, num_nodes as u32)?;
    provider.device().synchronize()?;

    let mut internal_prefix_last = [0u32];
    let mut clause_prefix_last = [0u32];
    let mut lit_prefix_last = [0u32];
    device
        .dtoh_sync_copy_into(
            &internal_counts.slice(last_idx.clone()),
            &mut internal_prefix_last,
        )
        .map_err(|e| XlogError::Kernel(format!("Failed to read internal_prefix last: {}", e)))?;
    device
        .dtoh_sync_copy_into(&clause_counts.slice(last_idx.clone()), &mut clause_prefix_last)
        .map_err(|e| XlogError::Kernel(format!("Failed to read clause_prefix last: {}", e)))?;
    device
        .dtoh_sync_copy_into(&lit_counts.slice(last_idx.clone()), &mut lit_prefix_last)
        .map_err(|e| XlogError::Kernel(format!("Failed to read lit_prefix last: {}", e)))?;

    let internal_total = internal_prefix_last[0]
        .checked_add(internal_last[0])
        .ok_or_else(|| XlogError::Kernel("Circuit internal var count overflow".to_string()))?;
    let clause_total = clause_prefix_last[0]
        .checked_add(clause_last[0])
        .ok_or_else(|| XlogError::Kernel("Circuit CNF clause count overflow".to_string()))?;
    let lit_total = lit_prefix_last[0]
        .checked_add(lit_last[0])
        .ok_or_else(|| XlogError::Kernel("Circuit CNF literal count overflow".to_string()))?;

    let num_vars = base_num_vars
        .checked_add(internal_total)
        .ok_or_else(|| XlogError::Kernel("Circuit CNF num_vars overflow".to_string()))?;

    let offsets_len = (clause_total as usize)
        .checked_add(1)
        .ok_or_else(|| XlogError::Kernel("Circuit CNF offsets length overflow".to_string()))?;
    let mut d_offsets = memory.alloc::<u32>(offsets_len)?;
    let mut d_lits = memory.alloc::<i32>(lit_total as usize)?;

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
        (&internal_counts).as_kernel_param(),
        (&clause_counts).as_kernel_param(),
        (&lit_counts).as_kernel_param(),
        base_num_vars.as_kernel_param(),
        (num_nodes as u32).as_kernel_param(),
        clause_total.as_kernel_param(),
        lit_total.as_kernel_param(),
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
    provider.device().synchronize()?;

    Ok(CircuitCnf {
        cnf: GpuCnf {
            num_vars,
            num_clauses: clause_total,
            clause_offsets: d_offsets,
            clause_lits: d_lits,
        },
        internal_prefix: internal_counts,
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

    let phi_clauses = phi.num_clauses as usize;
    let phi_lits = phi.clause_lits.len();

    let c_clauses = circuit_cnf.cnf.num_clauses as usize;
    let c_lits = circuit_cnf.cnf.clause_lits.len();

    let total_clauses = phi_clauses
        .checked_add(c_clauses)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| XlogError::Kernel("phi ∧ ¬C clause count overflow".to_string()))?;
    let total_lits = phi_lits
        .checked_add(c_lits)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| XlogError::Kernel("phi ∧ ¬C literal count overflow".to_string()))?;

    if total_clauses > u32::MAX as usize || total_lits > u32::MAX as usize {
        return Err(XlogError::Kernel(
            "phi ∧ ¬C exceeds u32::MAX address space".to_string(),
        ));
    }

    let mut out_offsets = memory.alloc::<u32>(total_clauses + 1)?;
    let mut out_lits = memory.alloc::<i32>(total_lits)?;

    // Copy phi into the front.
    {
        let mut dst = out_offsets.slice_mut(0..(phi_clauses + 1));
        device
            .dtod_copy(&phi.clause_offsets, &mut dst)
            .map_err(|e| XlogError::Kernel(format!("Failed to copy phi offsets: {}", e)))?;
    }
    if phi_lits > 0 {
        let mut dst = out_lits.slice_mut(0..phi_lits);
        device
            .dtod_copy(&phi.clause_lits, &mut dst)
            .map_err(|e| XlogError::Kernel(format!("Failed to copy phi lits: {}", e)))?;
    }

    // Copy CNF(C) literals after phi.
    if c_lits > 0 {
        let mut dst = out_lits.slice_mut(phi_lits..(phi_lits + c_lits));
        device
            .dtod_copy(&circuit_cnf.cnf.clause_lits, &mut dst)
            .map_err(|e| XlogError::Kernel(format!("Failed to copy CNF(C) lits: {}", e)))?;
    }

    // Shift and write CNF(C) offsets after phi.
    let shift_fn = device
        .get_func(SAT_MODULE, sat_kernels::SAT_SHIFT_OFFSETS)
        .ok_or_else(|| XlogError::Kernel("sat_shift_offsets kernel not found".to_string()))?;
    let n = (c_clauses as u32) + 1;
    let add = phi_lits as u32;
    let dst_base = phi_clauses as u32;

    let block = 256u32;
    let mut grid = (n + block - 1) / block;
    if grid == 0 {
        grid = 1;
    }
    if grid > 65_535 {
        grid = 65_535;
    }

    // SAFETY: sat_shift_offsets(src_offsets, n, add, dst_base, dst_offsets)
    unsafe {
        shift_fn.clone().launch(
            LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                &circuit_cnf.cnf.clause_offsets,
                n,
                add,
                dst_base,
                &mut out_offsets,
            ),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("sat_shift_offsets failed: {}", e)))?;

    // Append unit clause forcing root false.
    let unit_fn = device
        .get_func(SAT_MODULE, sat_kernels::SAT_XGCF_WRITE_ROOT_UNIT_CLAUSE)
        .ok_or_else(|| {
            XlogError::Kernel("sat_xgcf_write_root_unit_clause kernel not found".to_string())
        })?;
    let unit_clause_idx = (phi_clauses + c_clauses) as u32;
    let unit_lit_idx = (phi_lits + c_lits) as u32;

    // SAFETY: sat_xgcf_write_root_unit_clause(...)
    unsafe {
        unit_fn.clone().launch(
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                circuit.node_type(),
                circuit.lit(),
                &circuit_cnf.internal_prefix,
                phi.num_vars,
                circuit.root(),
                0i32, // force_false
                unit_clause_idx,
                unit_lit_idx,
                &mut out_offsets,
                &mut out_lits,
            ),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("sat_xgcf_write_root_unit_clause failed: {}", e)))?;

    provider.device().synchronize()?;

    Ok(GpuCnf {
        num_vars: circuit_cnf.cnf.num_vars,
        num_clauses: total_clauses as u32,
        clause_offsets: out_offsets,
        clause_lits: out_lits,
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

    let c_clauses = circuit_cnf.cnf.num_clauses as usize;
    let c_lits = circuit_cnf.cnf.clause_lits.len();

    let m = phi.num_clauses as usize;
    let l = phi.clause_lits.len();

    // ¬phi encoding:
    // clauses_notphi = sum(len_j + 1) + 1 = L + m + 1
    // lits_notphi = sum(3*len_j + 1) + m = 3L + 2m
    let notphi_clauses = l
        .checked_add(m)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| XlogError::Kernel("¬phi clause count overflow".to_string()))?;
    let notphi_lits = l
        .checked_mul(3)
        .and_then(|v| v.checked_add(2 * m))
        .ok_or_else(|| XlogError::Kernel("¬phi literal count overflow".to_string()))?;

    let total_clauses = c_clauses
        .checked_add(1)
        .and_then(|v| v.checked_add(notphi_clauses))
        .ok_or_else(|| XlogError::Kernel("C ∧ ¬phi clause count overflow".to_string()))?;
    let total_lits = c_lits
        .checked_add(1)
        .and_then(|v| v.checked_add(notphi_lits))
        .ok_or_else(|| XlogError::Kernel("C ∧ ¬phi literal count overflow".to_string()))?;

    if total_clauses > u32::MAX as usize || total_lits > u32::MAX as usize {
        return Err(XlogError::Kernel(
            "C ∧ ¬phi exceeds u32::MAX address space".to_string(),
        ));
    }

    let num_vars = circuit_cnf
        .cnf
        .num_vars
        .checked_add(phi.num_clauses)
        .ok_or_else(|| XlogError::Kernel("C ∧ ¬phi num_vars overflow".to_string()))?;

    let mut out_offsets = memory.alloc::<u32>(total_clauses + 1)?;
    let mut out_lits = memory.alloc::<i32>(total_lits)?;

    // Copy CNF(C) into the front.
    {
        let mut dst = out_offsets.slice_mut(0..(c_clauses + 1));
        device
            .dtod_copy(&circuit_cnf.cnf.clause_offsets, &mut dst)
            .map_err(|e| XlogError::Kernel(format!("Failed to copy CNF(C) offsets: {}", e)))?;
    }
    if c_lits > 0 {
        let mut dst = out_lits.slice_mut(0..c_lits);
        device
            .dtod_copy(&circuit_cnf.cnf.clause_lits, &mut dst)
            .map_err(|e| XlogError::Kernel(format!("Failed to copy CNF(C) lits: {}", e)))?;
    }

    // Insert unit clause forcing root true.
    let unit_fn = device
        .get_func(SAT_MODULE, sat_kernels::SAT_XGCF_WRITE_ROOT_UNIT_CLAUSE)
        .ok_or_else(|| {
            XlogError::Kernel("sat_xgcf_write_root_unit_clause kernel not found".to_string())
        })?;

    // SAFETY: sat_xgcf_write_root_unit_clause(...)
    unsafe {
        unit_fn.clone().launch(
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                circuit.node_type(),
                circuit.lit(),
                &circuit_cnf.internal_prefix,
                phi.num_vars,
                circuit.root(),
                1i32, // force_true
                c_clauses as u32,
                c_lits as u32,
                &mut out_offsets,
                &mut out_lits,
            ),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("sat_xgcf_write_root_unit_clause failed: {}", e)))?;

    // Emit ¬phi encoding after CNF(C) + unit.
    let not_phi_fn = device
        .get_func(SAT_MODULE, sat_kernels::SAT_EMIT_NOT_PHI)
        .ok_or_else(|| XlogError::Kernel("sat_emit_not_phi kernel not found".to_string()))?;

    let unsat_var_base = circuit_cnf
        .cnf
        .num_vars
        .checked_add(1)
        .ok_or_else(|| XlogError::Kernel("unsat_var_base overflow".to_string()))?;

    let out_clause_base = (c_clauses as u32)
        .checked_add(1)
        .ok_or_else(|| XlogError::Kernel("out_clause_base overflow".to_string()))?;
    let out_lit_base = (c_lits as u32)
        .checked_add(1)
        .ok_or_else(|| XlogError::Kernel("out_lit_base overflow".to_string()))?;

    let block = 256u32;
    let mut grid = ((phi.num_clauses) + block - 1) / block;
    if grid == 0 {
        grid = 1;
    }
    if grid > 65_535 {
        grid = 65_535;
    }

    // SAFETY: sat_emit_not_phi(phi_offsets, phi_lits, num_clauses, unsat_var_base, out_clause_base, out_lit_base, out_offsets, out_lits)
    unsafe {
        not_phi_fn.clone().launch(
            LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                &phi.clause_offsets,
                &phi.clause_lits,
                phi.num_clauses,
                unsat_var_base,
                out_clause_base,
                out_lit_base,
                &mut out_offsets,
                &mut out_lits,
            ),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("sat_emit_not_phi failed: {}", e)))?;

    provider.device().synchronize()?;

    Ok(GpuCnf {
        num_vars,
        num_clauses: total_clauses as u32,
        clause_offsets: out_offsets,
        clause_lits: out_lits,
    })
}

pub fn check_equivalence_gpu(
    phi: &GpuCnf,
    circuit: &GpuXgcf,
    provider: &Arc<CudaKernelProvider>,
    config: GpuEquivalenceConfig,
) -> Result<GpuEquivalenceCheck> {
    let circuit_cnf = build_circuit_cnf(provider, circuit, phi.num_vars)?;

    let q1 = build_phi_and_not_c(provider, phi, circuit, &circuit_cnf)?;
    let q2 = build_c_and_not_phi(provider, phi, circuit, &circuit_cnf)?;

    let solver = GpuCdclSolver::new(provider.clone(), config.cdcl);
    let r1 = solver.solve(&q1)?;
    let r2 = solver.solve(&q2)?;

    Ok(GpuEquivalenceCheck {
        phi_and_not_c: r1,
        c_and_not_phi: r2,
    })
}

pub fn validate_equivalence_gpu(
    phi: &GpuCnf,
    circuit: &GpuXgcf,
    provider: &Arc<CudaKernelProvider>,
    config: GpuEquivalenceConfig,
) -> Result<()> {
    let res = check_equivalence_gpu(phi, circuit, provider, config)?;

    if res.phi_and_not_c.status != GpuSolveStatus::Unsat {
        return Err(XlogError::Compilation(
            "Equivalence check failed: SAT(phi ∧ ¬C) returned SAT".to_string(),
        ));
    }
    if res.c_and_not_phi.status != GpuSolveStatus::Unsat {
        return Err(XlogError::Compilation(
            "Equivalence check failed: SAT(C ∧ ¬phi) returned SAT".to_string(),
        ));
    }
    Ok(())
}
