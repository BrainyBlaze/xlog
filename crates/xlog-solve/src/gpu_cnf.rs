use std::sync::Arc;

use cudarc::driver::DeviceSlice;
use xlog_core::{Result, XlogError};
use xlog_cuda::CudaKernelProvider;
use xlog_cuda::memory::TrackedCudaSlice;

use crate::instance::SolveInstance;

/// GPU-resident CNF in CSR form (DIMACS literals, 1-based variable ids).
///
/// This is the solver-facing CNF representation used by the GPU CDCL verifier.
pub struct GpuCnf {
    pub num_vars: u32,
    pub num_clauses: u32,
    /// CSR offsets (len = num_clauses + 1).
    pub clause_offsets: TrackedCudaSlice<u32>,
    /// Flattened CSR literal array (len = num_literals).
    pub clause_lits: TrackedCudaSlice<i32>,
}

impl GpuCnf {
    #[inline]
    pub fn num_literals(&self) -> usize {
        self.clause_lits.len()
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

        let mut clause_lits: Vec<i32> = Vec::new();
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
                    return Err(XlogError::Compilation("CNF contains DIMACS 0 literal".to_string()));
                }
                clause_lits.push(dimacs);
            }
        }

        if clause_offsets.len() != (num_clauses as usize + 1) {
            return Err(XlogError::Kernel(
                "GpuCnf::from_host internal error: offsets length mismatch".to_string(),
            ));
        }

        let memory = provider.memory();
        let mut d_offsets = memory.alloc::<u32>(clause_offsets.len())?;
        let mut d_lits = memory.alloc::<i32>(clause_lits.len())?;

        provider
            .device()
            .inner()
            .htod_sync_copy_into(&clause_offsets, &mut d_offsets)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload CNF offsets: {}", e)))?;
        provider
            .device()
            .inner()
            .htod_sync_copy_into(&clause_lits, &mut d_lits)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload CNF lits: {}", e)))?;

        Ok(Self {
            num_vars,
            num_clauses,
            clause_offsets: d_offsets,
            clause_lits: d_lits,
        })
    }
}
