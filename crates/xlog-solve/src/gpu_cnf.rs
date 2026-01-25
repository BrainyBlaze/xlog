use std::sync::Arc;

use cudarc::driver::DeviceSlice;
use xlog_core::{Result, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::CudaKernelProvider;

use crate::instance::SolveInstance;

/// GPU-resident CNF in CSR form (DIMACS literals, 1-based variable ids).
///
/// This is the solver-facing CNF representation used by the GPU CDCL verifier.
pub struct GpuCnf {
    /// Variable capacity (>= num_vars).
    pub var_cap: u32,
    /// Clause capacity (>= num_clauses).
    pub clause_cap: u32,
    /// Literal capacity (>= num_lits).
    pub lit_cap: u32,
    /// Device-resident num_vars (len = 1).
    pub num_vars: TrackedCudaSlice<u32>,
    /// Device-resident num_clauses (len = 1).
    pub num_clauses: TrackedCudaSlice<u32>,
    /// Device-resident num_lits (len = 1).
    pub num_lits: TrackedCudaSlice<u32>,
    /// CSR offsets (len = clause_cap + 1).
    pub clause_offsets: TrackedCudaSlice<u32>,
    /// Flattened CSR literal array (len = lit_cap).
    pub literals: TrackedCudaSlice<i32>,
}

impl GpuCnf {
    #[inline]
    pub fn offsets_len(&self) -> usize {
        self.clause_offsets.len()
    }

    #[inline]
    pub fn num_literals_cap(&self) -> usize {
        self.lit_cap as usize
    }

    /// Host -> device upload helper for tests and tooling.
    ///
    /// Production GPU-native paths should build `GpuCnf` directly on device.
    pub fn from_host(instance: &SolveInstance, provider: &Arc<CudaKernelProvider>) -> Result<Self> {
        if instance.objective != crate::Objective::Satisfaction {
            return Err(XlogError::Compilation(format!(
                "GpuCnf::from_host only supports Objective::Satisfaction, got {:?}",
                instance.objective
            )));
        }
        if instance.num_vars == 0 {
            return Err(XlogError::Compilation(
                "GpuCnf::from_host requires num_vars > 0".to_string(),
            ));
        }

        let num_vars = instance.num_vars;
        let num_clauses = instance.clauses.len() as u32;

        // Build CSR on host.
        let mut clause_offsets: Vec<u32> = Vec::with_capacity(instance.clauses.len() + 1);
        clause_offsets.push(0);

        let mut literals: Vec<i32> = Vec::new();
        for clause in &instance.clauses {
            let start = *clause_offsets.last().unwrap();
            let len = clause.literals.len() as u32;
            let end = start
                .checked_add(len)
                .ok_or_else(|| XlogError::Compilation("CNF literal count overflow".to_string()))?;
            clause_offsets.push(end);

            for &lit in &clause.literals {
                let dimacs = lit.to_dimacs();
                if dimacs == 0 {
                    return Err(XlogError::Compilation(
                        "CNF contains DIMACS 0 literal".to_string(),
                    ));
                }
                literals.push(dimacs);
            }
        }

        if clause_offsets.len() != (num_clauses as usize + 1) {
            return Err(XlogError::Kernel(
                "GpuCnf::from_host internal error: offsets length mismatch".to_string(),
            ));
        }

        let memory = provider.memory();

        // Device scalars (len=1 each).
        let mut d_num_vars = memory.alloc::<u32>(1)?;
        let mut d_num_clauses = memory.alloc::<u32>(1)?;
        let mut d_num_lits = memory.alloc::<u32>(1)?;

        provider
            .device()
            .inner()
            .htod_sync_copy_into(&[num_vars], &mut d_num_vars)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload CNF num_vars: {}", e)))?;
        provider
            .device()
            .inner()
            .htod_sync_copy_into(&[num_clauses], &mut d_num_clauses)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload CNF num_clauses: {}", e)))?;

        let mut d_offsets = memory.alloc::<u32>(clause_offsets.len())?;
        let mut d_lits = memory.alloc::<i32>(literals.len())?;

        provider
            .device()
            .inner()
            .htod_sync_copy_into(&clause_offsets, &mut d_offsets)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload CNF offsets: {}", e)))?;
        provider
            .device()
            .inner()
            .htod_sync_copy_into(&literals, &mut d_lits)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload CNF lits: {}", e)))?;

        provider
            .device()
            .inner()
            .htod_sync_copy_into(&[literals.len() as u32], &mut d_num_lits)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload CNF num_lits: {}", e)))?;

        Ok(Self {
            var_cap: num_vars,
            clause_cap: num_clauses,
            lit_cap: literals.len() as u32,
            num_vars: d_num_vars,
            num_clauses: d_num_clauses,
            num_lits: d_num_lits,
            clause_offsets: d_offsets,
            literals: d_lits,
        })
    }
}
