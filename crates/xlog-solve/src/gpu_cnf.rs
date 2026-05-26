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
    pub(crate) fn require_provider_memory(
        &self,
        provider: &CudaKernelProvider,
        context: &'static str,
    ) -> Result<()> {
        let expected_memory = Arc::as_ptr(provider.memory()) as usize;
        self.require_slice_provider_memory(
            context,
            "num_vars",
            self.num_vars.memory_manager_ptr_value(),
            expected_memory,
        )?;
        self.require_slice_provider_memory(
            context,
            "num_clauses",
            self.num_clauses.memory_manager_ptr_value(),
            expected_memory,
        )?;
        self.require_slice_provider_memory(
            context,
            "num_lits",
            self.num_lits.memory_manager_ptr_value(),
            expected_memory,
        )?;
        self.require_slice_provider_memory(
            context,
            "clause_offsets",
            self.clause_offsets.memory_manager_ptr_value(),
            expected_memory,
        )?;
        self.require_slice_provider_memory(
            context,
            "literals",
            self.literals.memory_manager_ptr_value(),
            expected_memory,
        )?;

        if self.num_vars.len() != 1 || self.num_clauses.len() != 1 || self.num_lits.len() != 1 {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: context.to_string(),
                context: format!(
                    "GPU CNF scalar buffers must have len=1, got num_vars={} num_clauses={} num_lits={}",
                    self.num_vars.len(),
                    self.num_clauses.len(),
                    self.num_lits.len()
                ),
            });
        }
        let expected_offsets = (self.clause_cap as usize).checked_add(1).ok_or_else(|| {
            XlogError::UnsupportedEpistemicConstruct {
                construct: context.to_string(),
                context: "GPU CNF clause offset length overflowed".to_string(),
            }
        })?;
        if self.clause_offsets.len() != expected_offsets
            || self.literals.len() != self.lit_cap as usize
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: context.to_string(),
                context: format!(
                    "GPU CNF buffer lengths must match capacities, got offsets={}/{} literals={}/{}",
                    self.clause_offsets.len(),
                    expected_offsets,
                    self.literals.len(),
                    self.lit_cap
                ),
            });
        }
        Ok(())
    }

    fn require_slice_provider_memory(
        &self,
        context: &'static str,
        name: &'static str,
        actual_memory: usize,
        expected_memory: usize,
    ) -> Result<()> {
        if actual_memory != expected_memory {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: context.to_string(),
                context: format!(
                    "GPU CNF buffer {name} belongs to memory manager {actual_memory}, expected {expected_memory}"
                ),
            });
        }
        Ok(())
    }

    #[inline]
    #[allow(dead_code)] // diagnostic accessor, retained for debugging
    pub(crate) fn offsets_len(&self) -> usize {
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
        if instance.num_vars > i32::MAX as u32 {
            return Err(XlogError::Compilation(
                "GpuCnf::from_host requires DIMACS variables to fit i32".to_string(),
            ));
        }
        if !instance.validate() {
            return Err(XlogError::Compilation(
                "GpuCnf::from_host saw a literal variable outside num_vars".to_string(),
            ));
        }

        let num_vars = instance.num_vars;
        let num_clauses = u32::try_from(instance.clauses.len()).map_err(|_| {
            XlogError::Compilation("GpuCnf::from_host clause count exceeds u32".to_string())
        })?;
        let offsets_len = instance.clauses.len().checked_add(1).ok_or_else(|| {
            XlogError::Compilation("GpuCnf::from_host clause offset count overflow".to_string())
        })?;
        let total_literals = instance.clauses.iter().try_fold(0usize, |acc, clause| {
            acc.checked_add(clause.literals.len()).ok_or_else(|| {
                XlogError::Compilation("GpuCnf::from_host literal count overflow".to_string())
            })
        })?;
        let lit_cap = u32::try_from(total_literals).map_err(|_| {
            XlogError::Compilation("GpuCnf::from_host literal count exceeds u32".to_string())
        })?;

        // Build CSR on host.
        let mut clause_offsets: Vec<u32> = Vec::with_capacity(offsets_len);
        clause_offsets.push(0);

        let mut literals: Vec<i32> = Vec::with_capacity(total_literals);
        for clause in &instance.clauses {
            let start = clause_offsets.last().copied().ok_or_else(|| {
                XlogError::Kernel(
                    "GpuCnf::from_host internal error: missing initial clause offset".to_string(),
                )
            })?;
            let len = u32::try_from(clause.literals.len()).map_err(|_| {
                XlogError::Compilation(
                    "GpuCnf::from_host clause literal count exceeds u32".to_string(),
                )
            })?;
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

        if clause_offsets.len() != offsets_len {
            return Err(XlogError::Kernel(
                "GpuCnf::from_host internal error: offsets length mismatch".to_string(),
            ));
        }
        if literals.len() != total_literals {
            return Err(XlogError::Kernel(
                "GpuCnf::from_host internal error: literal length mismatch".to_string(),
            ));
        }

        let memory = provider.memory();

        // Device scalars (len=1 each).
        let mut d_num_vars = memory.alloc::<u32>(1)?;
        let mut d_num_clauses = memory.alloc::<u32>(1)?;
        let mut d_num_lits = memory.alloc::<u32>(1)?;

        provider
            .htod_launch_metadata_sync_copy_into(&[num_vars], &mut d_num_vars)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload CNF num_vars: {}", e)))?;
        provider
            .htod_launch_metadata_sync_copy_into(&[num_clauses], &mut d_num_clauses)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload CNF num_clauses: {}", e)))?;

        let mut d_offsets = memory.alloc::<u32>(clause_offsets.len())?;
        let mut d_lits = memory.alloc::<i32>(literals.len())?;

        provider
            .htod_sync_copy_into_tracked(&clause_offsets, &mut d_offsets)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload CNF offsets: {}", e)))?;
        provider
            .htod_sync_copy_into_tracked(&literals, &mut d_lits)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload CNF lits: {}", e)))?;

        provider
            .htod_launch_metadata_sync_copy_into(&[lit_cap], &mut d_num_lits)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload CNF num_lits: {}", e)))?;

        Ok(Self {
            var_cap: num_vars,
            clause_cap: num_clauses,
            lit_cap,
            num_vars: d_num_vars,
            num_clauses: d_num_clauses,
            num_lits: d_num_lits,
            clause_offsets: d_offsets,
            literals: d_lits,
        })
    }
}
