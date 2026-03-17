//! GPU-native equivalence validation (φ ≡ C) using the GPU CDCL verifier.

use std::sync::Arc;

use std::ffi::c_void;

use cudarc::driver::{DeviceRepr, LaunchAsync, LaunchConfig};
use xlog_core::{Result, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::provider::sat_kernels;
use xlog_cuda::provider::SAT_MODULE;
use xlog_cuda::CudaKernelProvider;
use xlog_solve::{GpuCdclConfig, GpuCdclSolver, GpuCnf};

#[cfg(debug_assertions)]
use crate::compilation::gpu_d4::validate_cnf_gpu;

use crate::gpu::GpuXgcf;

/// Configuration for GPU-native equivalence verification (phi equiv C).
///
/// Controls the CDCL solver parameters and whether to reuse the solver
/// workspace across multiple equivalence checks. Workspace reuse amortizes
/// device-memory allocation when verifying many circuits in sequence (e.g.,
/// during incremental compilation).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct GpuEquivalenceConfig {
    /// CDCL solver configuration for the equivalence verifier.
    pub cdcl: GpuCdclConfig,
    /// Reuse the CDCL workspace across successive verifier invocations.
    pub reuse_workspace: bool,
}

impl Default for GpuEquivalenceConfig {
    fn default() -> Self {
        Self {
            cdcl: GpuCdclConfig::default(),
            reuse_workspace: false,
        }
    }
}

/// GPU-resident equivalence queries + device metadata required to solve them without host reads.
pub struct GpuEquivalenceQueries {
    pub q1: GpuCnf,
    pub q2: GpuCnf,
    /// Base variable id for the ¬phi selector vars in q2 (len=1, device-resident).
    pub q2_unsat_var_base: TrackedCudaSlice<u32>,
}

struct CircuitCnf {
    cnf: GpuCnf,
    /// Exclusive prefix sum over `is_internal(node)` (len = num_nodes).
    /// Used to map internal node ids -> Tseitin vars in kernels.
    internal_prefix: TrackedCudaSlice<u32>,
}

fn build_circuit_cnf(
    provider: &Arc<CudaKernelProvider>,
    circuit: &GpuXgcf,
    base_num_vars: &TrackedCudaSlice<u32>,
    base_var_cap: u32,
    compile_needed: &TrackedCudaSlice<u32>,
) -> Result<CircuitCnf> {
    if base_var_cap == 0 {
        return Err(XlogError::Compilation(
            "GPU equivalence verifier requires base_var_cap > 0".to_string(),
        ));
    }
    if circuit.max_var() > base_var_cap {
        return Err(XlogError::Compilation(format!(
            "Circuit references var {} but base CNF has only {} vars",
            circuit.max_var(),
            base_var_cap
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
    let num_edges = circuit.num_edges();
    let n64 = num_nodes as u64;
    let e64 = num_edges as u64;

    let var_cap = u32::try_from((base_var_cap as u64).saturating_add(n64))
        .map_err(|_| XlogError::Kernel("Circuit CNF var capacity exceeds u32::MAX".to_string()))?;
    let clause_cap =
        u32::try_from(e64.checked_add(4u64.saturating_mul(n64)).ok_or_else(|| {
            XlogError::Kernel("Circuit CNF clause capacity overflow".to_string())
        })?)
        .map_err(|_| {
            XlogError::Kernel("Circuit CNF clause capacity exceeds u32::MAX".to_string())
        })?;
    let lit_cap = u32::try_from(
        (3u64.saturating_mul(e64))
            .checked_add(12u64.saturating_mul(n64))
            .ok_or_else(|| {
                XlogError::Kernel("Circuit CNF literal capacity overflow".to_string())
            })?,
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

    // SAFETY: sat_xgcf_cnf_counts(compile_needed, node_type, child_offsets, num_nodes, internal_counts, clause_counts, lit_counts)
    unsafe {
        counts_fn.clone().launch(
            LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                compile_needed,
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
        capture_last_fn.clone().launch(
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
    // No device synchronize: next ops are alloc + kernel launches on same stream.

    // Output CNF buffers + device-resident meta.
    let mut d_num_vars = memory.alloc::<u32>(1)?;
    let mut d_num_clauses = memory.alloc::<u32>(1)?;
    let mut d_num_lits = memory.alloc::<u32>(1)?;
    let mut d_offsets = memory.alloc::<u32>((clause_cap as usize) + 1)?;
    let mut d_lits = memory.alloc::<i32>(lit_cap as usize)?;

    let totals_fn = device
        .get_func(SAT_MODULE, sat_kernels::SAT_XGCF_CNF_COMPUTE_TOTALS)
        .ok_or_else(|| {
            XlogError::Kernel("sat_xgcf_cnf_compute_totals kernel not found".to_string())
        })?;
    // SAFETY: sat_xgcf_cnf_compute_totals(internal_prefix, clause_base, lit_base, internal_last*, clause_last*, lit_last*, num_nodes, base_num_vars, clause_cap, lit_cap, out_num_vars*, out_num_clauses*, out_num_lits*)
    let mut totals_params: Vec<*mut c_void> = vec![
        (&internal_prefix).as_kernel_param(),
        (&clause_base).as_kernel_param(),
        (&lit_base).as_kernel_param(),
        (&internal_last).as_kernel_param(),
        (&clause_last).as_kernel_param(),
        (&lit_last).as_kernel_param(),
        num_nodes_u32.as_kernel_param(),
        (base_num_vars).as_kernel_param(),
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
        compile_needed.as_kernel_param(),
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
        (base_num_vars).as_kernel_param(),
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
        .ok_or_else(|| {
            XlogError::Kernel("sat_cnf_write_terminator kernel not found".to_string())
        })?;
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
    // No device synchronize: returns device-resident CNF; same-stream ordering suffices.

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
    compile_needed: &TrackedCudaSlice<u32>,
) -> Result<GpuCnf> {
    let device = provider.device().inner();
    let memory = provider.memory();

    let phi_clause_cap = phi.clause_cap;
    let phi_lit_cap = phi.lit_cap;

    let clause_cap = u32::try_from(
        (phi_clause_cap as u64)
            .checked_add(circuit_cnf.cnf.clause_cap as u64)
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| XlogError::Kernel("phi ∧ ¬C clause capacity overflow".to_string()))?,
    )
    .map_err(|_| XlogError::Kernel("phi ∧ ¬C clause capacity exceeds u32::MAX".to_string()))?;
    let lit_cap = u32::try_from(
        (phi_lit_cap as u64)
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

    let mut d_zero = memory.alloc::<u32>(1)?;
    device
        .htod_sync_copy_into(&[0u32], &mut d_zero)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload zero: {}", e)))?;

    let mut out_offsets = memory.alloc::<u32>((clause_cap as usize) + 1)?;
    let mut out_lits = memory.alloc::<i32>(lit_cap as usize)?;

    let copy_fn = device
        .get_func(SAT_MODULE, sat_kernels::SAT_CNF_COPY_INTO)
        .ok_or_else(|| XlogError::Kernel("sat_cnf_copy_into kernel not found".to_string()))?;

    let block = 256u32;
    let mut grid = (phi_clause_cap.saturating_add(1).max(phi_lit_cap) + block - 1) / block;
    if grid == 0 {
        grid = 1;
    }
    if grid > 65_535 {
        grid = 65_535;
    }

    // Copy phi (exact sizes) into the front.
    // sat_cnf_copy_into(src_offsets, src_lits, src_num_clauses*, src_num_lits*, src_clause_cap, src_lit_cap,
    //                  dst_clause_base*, dst_lit_base*, dst_clause_cap, dst_lit_cap, dst_offsets, dst_lits)
    unsafe {
        copy_fn.clone().launch(
            LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                &phi.clause_offsets,
                &phi.literals,
                &phi.num_clauses,
                &phi.num_lits,
                phi.clause_cap,
                phi.lit_cap,
                &d_zero,
                &d_zero,
                clause_cap,
                lit_cap,
                &mut out_offsets,
                &mut out_lits,
            ),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("sat_cnf_copy_into(phi) failed: {}", e)))?;

    // Copy CNF(C) after phi using device-resident bases (phi.num_clauses/phi.num_lits).
    let mut grid_c = (circuit_cnf
        .cnf
        .clause_cap
        .saturating_add(1)
        .max(circuit_cnf.cnf.lit_cap)
        + block
        - 1)
        / block;
    if grid_c == 0 {
        grid_c = 1;
    }
    if grid_c > 65_535 {
        grid_c = 65_535;
    }
    unsafe {
        copy_fn.clone().launch(
            LaunchConfig {
                grid_dim: (grid_c, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                &circuit_cnf.cnf.clause_offsets,
                &circuit_cnf.cnf.literals,
                &circuit_cnf.cnf.num_clauses,
                &circuit_cnf.cnf.num_lits,
                circuit_cnf.cnf.clause_cap,
                circuit_cnf.cnf.lit_cap,
                &phi.num_clauses,
                &phi.num_lits,
                clause_cap,
                lit_cap,
                &mut out_offsets,
                &mut out_lits,
            ),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("sat_cnf_copy_into(C) failed: {}", e)))?;

    // Finalize: append unit clause forcing root false + write device-resident totals for the combined query.
    let unit_fn = device
        .get_func(SAT_MODULE, sat_kernels::SAT_XGCF_WRITE_ROOT_UNIT_CLAUSE)
        .ok_or_else(|| {
            XlogError::Kernel("sat_xgcf_write_root_unit_clause kernel not found".to_string())
        })?;

    // IMPORTANT: When launching with an explicit `Vec<*mut c_void>` parameter list, scalar kernel
    // arguments MUST be backed by stable host storage until `cuLaunchKernel` copies them. Do not
    // pass temporaries like `circuit.root().as_kernel_param()` or `0i32.as_kernel_param()`.
    let root = circuit.root();
    let force_true: i32 = 0;
    let out_var_cap = var_cap;
    let out_clause_cap = clause_cap;
    let out_lit_cap = lit_cap;

    let mut params: Vec<*mut c_void> = vec![
        compile_needed.as_kernel_param(),
        circuit.node_type().as_kernel_param(),
        circuit.lit().as_kernel_param(),
        (&circuit_cnf.internal_prefix).as_kernel_param(),
        (&phi.num_vars).as_kernel_param(),
        root.as_kernel_param(),
        force_true.as_kernel_param(), // force_false
        (&phi.num_clauses).as_kernel_param(),
        (&phi.num_lits).as_kernel_param(),
        (&circuit_cnf.cnf.num_vars).as_kernel_param(),
        (&circuit_cnf.cnf.num_clauses).as_kernel_param(),
        (&circuit_cnf.cnf.num_lits).as_kernel_param(),
        (&d_zero).as_kernel_param(), // extra_num_vars
        (&d_zero).as_kernel_param(), // extra_num_clauses
        (&d_zero).as_kernel_param(), // extra_num_lits
        out_var_cap.as_kernel_param(),
        out_clause_cap.as_kernel_param(),
        out_lit_cap.as_kernel_param(),
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
        unit_fn.clone().launch(
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            },
            &mut params,
        )
    }
    .map_err(|e| XlogError::Kernel(format!("sat_xgcf_write_root_unit_clause failed: {}", e)))?;
    // No device synchronize: returns device-resident CNF; same-stream ordering suffices.

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
    compile_needed: &TrackedCudaSlice<u32>,
) -> Result<(GpuCnf, TrackedCudaSlice<u32>)> {
    let device = provider.device().inner();
    let memory = provider.memory();

    let phi_clause_cap = phi.clause_cap;
    let phi_lit_cap = phi.lit_cap;

    // ¬phi encoding:
    // clauses_notphi = sum(len_j + 1) + 1 = L + m + 1
    // lits_notphi = sum(3*len_j + 1) + m = 3L + 2m
    let notphi_clause_cap = u32::try_from(
        (phi_lit_cap as u64)
            .checked_add(phi_clause_cap as u64)
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| XlogError::Kernel("¬phi clause count overflow".to_string()))?,
    )
    .map_err(|_| XlogError::Kernel("¬phi clause count exceeds u32::MAX".to_string()))?;
    let notphi_lit_cap = u32::try_from(
        (phi_lit_cap as u64)
            .checked_mul(3)
            .and_then(|v| v.checked_add(2u64.saturating_mul(phi_clause_cap as u64)))
            .ok_or_else(|| XlogError::Kernel("¬phi literal count overflow".to_string()))?,
    )
    .map_err(|_| XlogError::Kernel("¬phi literal count exceeds u32::MAX".to_string()))?;

    let var_cap = circuit_cnf
        .cnf
        .var_cap
        .checked_add(phi_clause_cap)
        .ok_or_else(|| XlogError::Kernel("C ∧ ¬phi var capacity overflow".to_string()))?;
    let clause_cap = u32::try_from(
        (circuit_cnf.cnf.clause_cap as u64)
            .checked_add(1)
            .and_then(|v| v.checked_add(notphi_clause_cap as u64))
            .ok_or_else(|| XlogError::Kernel("C ∧ ¬phi clause capacity overflow".to_string()))?,
    )
    .map_err(|_| XlogError::Kernel("C ∧ ¬phi clause capacity exceeds u32::MAX".to_string()))?;
    let lit_cap = u32::try_from(
        (circuit_cnf.cnf.lit_cap as u64)
            .checked_add(1)
            .and_then(|v| v.checked_add(notphi_lit_cap as u64))
            .ok_or_else(|| XlogError::Kernel("C ∧ ¬phi literal capacity overflow".to_string()))?,
    )
    .map_err(|_| XlogError::Kernel("C ∧ ¬phi literal capacity exceeds u32::MAX".to_string()))?;

    let mut out_num_vars = memory.alloc::<u32>(1)?;
    let mut out_num_clauses = memory.alloc::<u32>(1)?;
    let mut out_num_lits = memory.alloc::<u32>(1)?;

    let mut d_zero = memory.alloc::<u32>(1)?;
    device
        .htod_sync_copy_into(&[0u32], &mut d_zero)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload zero: {}", e)))?;

    // Device-resident exact extras for ¬phi (computed from phi.num_*).
    let mut d_extra_num_vars = memory.alloc::<u32>(1)?;
    let mut d_extra_num_clauses = memory.alloc::<u32>(1)?;
    let mut d_extra_num_lits = memory.alloc::<u32>(1)?;

    let mut d_unsat_var_base = memory.alloc::<u32>(1)?;
    let mut d_notphi_clause_base = memory.alloc::<u32>(1)?;
    let mut d_notphi_lit_base = memory.alloc::<u32>(1)?;

    let mut out_offsets = memory.alloc::<u32>((clause_cap as usize) + 1)?;
    let mut out_lits = memory.alloc::<i32>(lit_cap as usize)?;

    let copy_fn = device
        .get_func(SAT_MODULE, sat_kernels::SAT_CNF_COPY_INTO)
        .ok_or_else(|| XlogError::Kernel("sat_cnf_copy_into kernel not found".to_string()))?;

    // Copy CNF(C) into the front (exact sizes).
    let block = 256u32;
    let mut grid = (circuit_cnf
        .cnf
        .clause_cap
        .saturating_add(1)
        .max(circuit_cnf.cnf.lit_cap)
        + block
        - 1)
        / block;
    if grid == 0 {
        grid = 1;
    }
    if grid > 65_535 {
        grid = 65_535;
    }
    // sat_cnf_copy_into(...)
    unsafe {
        copy_fn.clone().launch(
            LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                &circuit_cnf.cnf.clause_offsets,
                &circuit_cnf.cnf.literals,
                &circuit_cnf.cnf.num_clauses,
                &circuit_cnf.cnf.num_lits,
                circuit_cnf.cnf.clause_cap,
                circuit_cnf.cnf.lit_cap,
                &d_zero,
                &d_zero,
                clause_cap,
                lit_cap,
                &mut out_offsets,
                &mut out_lits,
            ),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("sat_cnf_copy_into(C) failed: {}", e)))?;

    // Compute exact ¬phi size contributions on GPU.
    let notphi_counts_fn = device
        .get_func(SAT_MODULE, sat_kernels::SAT_NOT_PHI_COUNTS)
        .ok_or_else(|| XlogError::Kernel("sat_not_phi_counts kernel not found".to_string()))?;
    // SAFETY: sat_not_phi_counts(phi_num_clauses*, phi_num_lits*, out_extra_num_vars*, out_extra_num_clauses*, out_extra_num_lits*)
    unsafe {
        notphi_counts_fn.clone().launch(
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                compile_needed,
                &phi.num_clauses,
                &phi.num_lits,
                &mut d_extra_num_vars,
                &mut d_extra_num_clauses,
                &mut d_extra_num_lits,
            ),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("sat_not_phi_counts failed: {}", e)))?;

    // Prepare: insert unit clause forcing root true and compute device-resident totals / bases.
    let unit_fn = device
        .get_func(SAT_MODULE, sat_kernels::SAT_XGCF_WRITE_ROOT_UNIT_CLAUSE)
        .ok_or_else(|| {
            XlogError::Kernel("sat_xgcf_write_root_unit_clause kernel not found".to_string())
        })?;

    // IMPORTANT: See note in build_phi_and_not_c about stable scalar kernel parameters.
    let root = circuit.root();
    let force_true: i32 = 1;
    let out_var_cap = var_cap;
    let out_clause_cap = clause_cap;
    let out_lit_cap = lit_cap;

    let mut params: Vec<*mut c_void> = vec![
        compile_needed.as_kernel_param(),
        circuit.node_type().as_kernel_param(),
        circuit.lit().as_kernel_param(),
        (&circuit_cnf.internal_prefix).as_kernel_param(),
        (&phi.num_vars).as_kernel_param(),
        root.as_kernel_param(),
        force_true.as_kernel_param(), // force_true
        (&d_zero).as_kernel_param(),  // clause_base
        (&d_zero).as_kernel_param(),  // lit_base
        (&circuit_cnf.cnf.num_vars).as_kernel_param(),
        (&circuit_cnf.cnf.num_clauses).as_kernel_param(),
        (&circuit_cnf.cnf.num_lits).as_kernel_param(),
        (&d_extra_num_vars).as_kernel_param(), // extra_num_vars (u_j vars)
        (&d_extra_num_clauses).as_kernel_param(), // extra_num_clauses
        (&d_extra_num_lits).as_kernel_param(), // extra_num_lits
        out_var_cap.as_kernel_param(),
        out_clause_cap.as_kernel_param(),
        out_lit_cap.as_kernel_param(),
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
        unit_fn.clone().launch(
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
    let mut grid = (phi_clause_cap + block - 1) / block;
    if grid == 0 {
        grid = 1;
    }
    if grid > 65_535 {
        grid = 65_535;
    }

    // SAFETY: sat_emit_not_phi(phi_offsets, phi_lits, phi_num_clauses*, unsat_var_base*, out_clause_base*, out_lit_base*, out_offsets, out_lits)
    unsafe {
        not_phi_fn.clone().launch(
            LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                compile_needed,
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
    // No device synchronize: returns device-resident CNF; same-stream ordering suffices.

    Ok((
        GpuCnf {
            var_cap,
            clause_cap,
            lit_cap,
            num_vars: out_num_vars,
            num_clauses: out_num_clauses,
            num_lits: out_num_lits,
            clause_offsets: out_offsets,
            literals: out_lits,
        },
        d_unsat_var_base,
    ))
}

pub(crate) fn check_equivalence_gpu(
    phi: &GpuCnf,
    phi_decision_var_limit: &TrackedCudaSlice<u32>,
    circuit: &GpuXgcf,
    provider: &Arc<CudaKernelProvider>,
    config: GpuEquivalenceConfig,
) -> Result<()> {
    let queries = build_equivalence_queries_gpu(phi, circuit, provider)?;

    #[cfg(debug_assertions)]
    {
        // Fail-fast: if query CNFs are malformed, the solver may hang or misbehave.
        validate_cnf_gpu(&queries.q1, provider.as_ref())?;
        validate_cnf_gpu(&queries.q2, provider.as_ref())?;
    }

    let solver = GpuCdclSolver::new(provider.clone(), config.cdcl);
    if config.reuse_workspace {
        let max_var_cap = std::cmp::max(queries.q1.var_cap, queries.q2.var_cap);
        let max_clause_cap = std::cmp::max(queries.q1.clause_cap, queries.q2.clause_cap);
        let mut ws = solver.new_workspace(max_var_cap, max_clause_cap)?;
        // q1: decisions only on semantically meaningful phi vars (exclude internal/Tseitin vars).
        solver.solve_expect_unsat_with_branch_limit_ws(
            &mut ws,
            &queries.q1,
            phi_decision_var_limit,
        )?;
        // q2: decisions on semantically meaningful phi vars + ¬phi selector vars.
        solver.solve_expect_unsat_with_decision_ranges_ws(
            &mut ws,
            &queries.q2,
            phi_decision_var_limit,
            &queries.q2_unsat_var_base,
            &phi.num_clauses,
        )?;
    } else {
        // q1: decisions only on semantically meaningful phi vars (exclude internal/Tseitin vars).
        solver.solve_expect_unsat_with_branch_limit(&queries.q1, phi_decision_var_limit)?;
        // q2: decisions on semantically meaningful phi vars + ¬phi selector vars.
        solver.solve_expect_unsat_with_decision_ranges(
            &queries.q2,
            phi_decision_var_limit,
            &queries.q2_unsat_var_base,
            &phi.num_clauses,
        )?;
    }
    Ok(())
}

/// Build the two equivalence-check queries on GPU:
/// - q1 = φ ∧ ¬C
/// - q2 = C ∧ ¬φ
///
/// This helper exists so tests and tooling can inspect query CNFs without duplicating kernel
/// orchestration logic.
pub fn build_equivalence_queries_gpu(
    phi: &GpuCnf,
    circuit: &GpuXgcf,
    provider: &Arc<CudaKernelProvider>,
) -> Result<GpuEquivalenceQueries> {
    // Non-gated path: force compilation/verification on.
    let memory = provider.memory();
    let device = provider.device().inner();
    let mut compile_needed = memory.alloc::<u32>(1)?;
    device
        .htod_sync_copy_into(&[1u32], &mut compile_needed)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload compile_needed=1: {}", e)))?;

    let circuit_cnf = build_circuit_cnf(
        provider,
        circuit,
        &phi.num_vars,
        phi.var_cap,
        &compile_needed,
    )?;
    let q1 = build_phi_and_not_c(provider, phi, circuit, &circuit_cnf, &compile_needed)?;
    let (q2, q2_unsat_var_base) =
        build_c_and_not_phi(provider, phi, circuit, &circuit_cnf, &compile_needed)?;
    Ok(GpuEquivalenceQueries {
        q1,
        q2,
        q2_unsat_var_base,
    })
}

pub(crate) fn check_equivalence_gpu_gated(
    phi: &GpuCnf,
    phi_decision_var_limit: &TrackedCudaSlice<u32>,
    circuit: &GpuXgcf,
    provider: &Arc<CudaKernelProvider>,
    config: GpuEquivalenceConfig,
    compile_needed: &TrackedCudaSlice<u32>,
) -> Result<()> {
    #[cfg(debug_assertions)]
    eprintln!("[xlog-prob] equivalence: build_circuit_cnf");
    let circuit_cnf = build_circuit_cnf(
        provider,
        circuit,
        &phi.num_vars,
        phi.var_cap,
        compile_needed,
    )?;
    #[cfg(debug_assertions)]
    {
        provider.device().synchronize().map_err(|e| {
            XlogError::Kernel(format!("sync after build_circuit_cnf failed: {}", e))
        })?;
        eprintln!("[xlog-prob] equivalence: build_phi_and_not_c");
    }

    let q1 = build_phi_and_not_c(provider, phi, circuit, &circuit_cnf, compile_needed)?;
    #[cfg(debug_assertions)]
    {
        provider.device().synchronize().map_err(|e| {
            XlogError::Kernel(format!("sync after build_phi_and_not_c failed: {}", e))
        })?;
        eprintln!("[xlog-prob] equivalence: build_c_and_not_phi");
    }
    let (q2, q2_unsat_var_base) =
        build_c_and_not_phi(provider, phi, circuit, &circuit_cnf, compile_needed)?;
    #[cfg(debug_assertions)]
    {
        provider.device().synchronize().map_err(|e| {
            XlogError::Kernel(format!("sync after build_c_and_not_phi failed: {}", e))
        })?;
        eprintln!(
            "[xlog-prob] equivalence: caps: phi(v={} c={} l={}) circuit_cnf(v={} c={} l={}) q1(v={} c={} l={}) q2(v={} c={} l={})",
            phi.var_cap,
            phi.clause_cap,
            phi.lit_cap,
            circuit_cnf.cnf.var_cap,
            circuit_cnf.cnf.clause_cap,
            circuit_cnf.cnf.lit_cap,
            q1.var_cap,
            q1.clause_cap,
            q1.lit_cap,
            q2.var_cap,
            q2.clause_cap,
            q2.lit_cap,
        );
        eprintln!("[xlog-prob] equivalence: solve_expect_unsat q1");
    }

    #[cfg(debug_assertions)]
    {
        validate_cnf_gpu(&q1, provider.as_ref())?;
        validate_cnf_gpu(&q2, provider.as_ref())?;
    }

    let solver = GpuCdclSolver::new(provider.clone(), config.cdcl);
    if config.reuse_workspace {
        let max_var_cap = std::cmp::max(q1.var_cap, q2.var_cap);
        let max_clause_cap = std::cmp::max(q1.clause_cap, q2.clause_cap);
        let mut ws = solver.new_workspace(max_var_cap, max_clause_cap)?;
        solver.solve_expect_unsat_with_branch_limit_gated_ws(
            &mut ws,
            &q1,
            compile_needed,
            phi_decision_var_limit,
        )?;
        #[cfg(debug_assertions)]
        {
            provider.device().synchronize().map_err(|e| {
                XlogError::Kernel(format!("sync after solve_expect_unsat(q1) failed: {}", e))
            })?;
            eprintln!("[xlog-prob] equivalence: solve_expect_unsat q2");
        }
        solver.solve_expect_unsat_with_decision_ranges_gated_ws(
            &mut ws,
            &q2,
            compile_needed,
            phi_decision_var_limit,
            &q2_unsat_var_base,
            &phi.num_clauses,
        )?;
    } else {
        solver.solve_expect_unsat_with_branch_limit_gated(
            &q1,
            compile_needed,
            phi_decision_var_limit,
        )?;
        #[cfg(debug_assertions)]
        {
            provider.device().synchronize().map_err(|e| {
                XlogError::Kernel(format!("sync after solve_expect_unsat(q1) failed: {}", e))
            })?;
            eprintln!("[xlog-prob] equivalence: solve_expect_unsat q2");
        }
        solver.solve_expect_unsat_with_decision_ranges_gated(
            &q2,
            compile_needed,
            phi_decision_var_limit,
            &q2_unsat_var_base,
            &phi.num_clauses,
        )?;
    }
    #[cfg(debug_assertions)]
    {
        provider.device().synchronize().map_err(|e| {
            XlogError::Kernel(format!("sync after solve_expect_unsat(q2) failed: {}", e))
        })?;
        eprintln!("[xlog-prob] equivalence: done");
    }
    Ok(())
}

pub fn validate_equivalence_gpu(
    phi: &GpuCnf,
    phi_decision_var_limit: &TrackedCudaSlice<u32>,
    circuit: &GpuXgcf,
    provider: &Arc<CudaKernelProvider>,
    config: GpuEquivalenceConfig,
) -> Result<()> {
    check_equivalence_gpu(phi, phi_decision_var_limit, circuit, provider, config)
}

pub fn validate_equivalence_gpu_gated(
    phi: &GpuCnf,
    phi_decision_var_limit: &TrackedCudaSlice<u32>,
    circuit: &GpuXgcf,
    provider: &Arc<CudaKernelProvider>,
    config: GpuEquivalenceConfig,
    compile_needed: &TrackedCudaSlice<u32>,
) -> Result<()> {
    check_equivalence_gpu_gated(
        phi,
        phi_decision_var_limit,
        circuit,
        provider,
        config,
        compile_needed,
    )
}
