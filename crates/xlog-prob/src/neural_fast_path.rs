//! GPU neural fast-path helpers (device slot mapping + AD-chain glue).
//!
//! This module contains GPU-resident tables used to map neural predicate outputs
//! (probability vectors) to CNF variable ids in the compiled circuit.

use cudarc::driver::{CudaView, DeviceSlice};
use xlog_core::{Result, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::CudaKernelProvider;

#[derive(Debug, Clone, Copy)]
pub struct NeuralFastPathConfig {
    /// Probability mass reserved for the implicit "none" outcome.
    pub eps: f64,
    /// Minimum probability clamp used for numerical stability.
    pub min_p: f64,
}

impl Default for NeuralFastPathConfig {
    fn default() -> Self {
        Self {
            eps: 1e-7,
            min_p: 1e-12,
        }
    }
}

/// Device-resident mapping from neural output slots to CNF variable ids.
///
/// Slots are grouped (one group per neural predicate instance). Each slot is a
/// CNF var id (DIMACS, 1-based) whose log-weights should be updated from the
/// group’s probability vector.
pub struct GpuWeightSlots {
    group_offsets_host: Vec<u32>,
    group_offsets: TrackedCudaSlice<u32>, // len = num_groups + 1
    slot_cnf_var: TrackedCudaSlice<u32>,  // len = total_slots
}

impl GpuWeightSlots {
    /// Upload a slot mapping from host vectors.
    ///
    /// `groups[g][i]` is the CNF variable id corresponding to label/slot `i` of group `g`.
    pub fn upload(provider: &CudaKernelProvider, groups: &[Vec<u32>]) -> Result<Self> {
        let mut offsets: Vec<u32> = Vec::with_capacity(groups.len().saturating_add(1));
        offsets.push(0);

        let mut flat: Vec<u32> = Vec::new();
        for g in groups {
            flat.extend_from_slice(g);
            offsets.push(flat.len() as u32);
        }

        let memory = provider.memory().clone();
        let device = provider.device().inner();

        let mut d_offsets = memory.alloc::<u32>(offsets.len())?;
        device
            .htod_sync_copy_into(&offsets, &mut d_offsets)
            .map_err(|e| {
                XlogError::Kernel(format!("Failed to upload weight slot offsets: {}", e))
            })?;

        let mut d_vars = memory.alloc::<u32>(flat.len())?;
        device
            .htod_sync_copy_into(&flat, &mut d_vars)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload weight slot vars: {}", e)))?;

        Ok(Self {
            group_offsets_host: offsets,
            group_offsets: d_offsets,
            slot_cnf_var: d_vars,
        })
    }

    pub fn num_groups(&self) -> u32 {
        self.group_offsets_host
            .len()
            .saturating_sub(1)
            .try_into()
            .unwrap_or(0)
    }

    pub fn total_slots(&self) -> u32 {
        self.group_offsets_host.last().copied().unwrap_or(0)
    }

    pub fn group_offsets(&self) -> &TrackedCudaSlice<u32> {
        &self.group_offsets
    }

    pub fn slot_cnf_var(&self) -> &TrackedCudaSlice<u32> {
        &self.slot_cnf_var
    }

    /// Device view over `slot_cnf_var` for a single group.
    pub fn group_slot_cnf_var(&self, group_idx: usize) -> Result<CudaView<'_, u32>> {
        let start = *self
            .group_offsets_host
            .get(group_idx)
            .ok_or_else(|| XlogError::Compilation("Group index out of bounds".to_string()))?
            as usize;
        let end = *self
            .group_offsets_host
            .get(group_idx + 1)
            .ok_or_else(|| XlogError::Compilation("Group index out of bounds".to_string()))?
            as usize;
        if end < start || end > self.slot_cnf_var.len() {
            return Err(XlogError::Compilation(
                "Invalid group slot range in GpuWeightSlots".to_string(),
            ));
        }
        Ok(self.slot_cnf_var.slice(start..end))
    }
}
