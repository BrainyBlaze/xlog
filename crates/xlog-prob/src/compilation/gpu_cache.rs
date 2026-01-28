//! GPU-resident circuit cache helpers.

use std::sync::Arc;

use cudarc::driver::{LaunchAsync, LaunchConfig};
use xlog_core::{Result, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::provider::{cache_kernels, CACHE_MODULE};
use xlog_cuda::CudaKernelProvider;
use xlog_solve::GpuCnf;

pub struct GpuCircuitCache {
    provider: Arc<CudaKernelProvider>,
    table_size: u32,
    num_slots: u32,
    keys: TrackedCudaSlice<u64>,
    slots: TrackedCudaSlice<u32>,
    state: TrackedCudaSlice<u32>,
    last_used: TrackedCudaSlice<u64>,
    slot_states: TrackedCudaSlice<u32>,
    clock: TrackedCudaSlice<u64>,
}

pub struct GpuCacheLookup {
    provider: Arc<CudaKernelProvider>,
    slot: TrackedCudaSlice<u32>,
    compile_needed: TrackedCudaSlice<u32>,
}

impl GpuCacheLookup {
    pub fn slot_device(&self) -> &TrackedCudaSlice<u32> {
        &self.slot
    }

    pub fn compile_needed_device(&self) -> &TrackedCudaSlice<u32> {
        &self.compile_needed
    }

    pub fn provider(&self) -> &Arc<CudaKernelProvider> {
        &self.provider
    }
}

/// Compute a deterministic CNF hash on the GPU.
///
/// Hash input order matches the cache kernel: num_vars, num_clauses, num_lits,
/// clause_offsets[0..num_clauses], literals[0..num_lits-1].
pub fn hash_cnf_gpu(
    cnf: &GpuCnf,
    provider: &Arc<CudaKernelProvider>,
) -> Result<TrackedCudaSlice<u64>> {
    let memory = provider.memory();
    let mut out_hash = memory.alloc::<u64>(1)?;

    let func = provider
        .device()
        .inner()
        .get_func(CACHE_MODULE, cache_kernels::CACHE_CNF_HASH)
        .ok_or_else(|| XlogError::Kernel("cache_cnf_hash kernel not found".to_string()))?;

    unsafe {
        func.clone().launch(
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                &cnf.num_vars,
                &cnf.num_clauses,
                &cnf.num_lits,
                &cnf.clause_offsets,
                &cnf.literals,
                &mut out_hash,
            ),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("cache_cnf_hash launch failed: {}", e)))?;

    provider
        .device()
        .synchronize()
        .map_err(|e| XlogError::Kernel(format!("cache_cnf_hash sync failed: {}", e)))?;

    Ok(out_hash)
}

impl GpuCircuitCache {
    pub fn new(
        provider: &Arc<CudaKernelProvider>,
        num_slots: u32,
        table_size: u32,
    ) -> Result<Self> {
        if num_slots == 0 {
            return Err(XlogError::Compilation(
                "GpuCircuitCache requires num_slots > 0".to_string(),
            ));
        }
        if table_size == 0 {
            return Err(XlogError::Compilation(
                "GpuCircuitCache requires table_size > 0".to_string(),
            ));
        }
        if table_size < num_slots {
            return Err(XlogError::Compilation(format!(
                "GpuCircuitCache table_size {} < num_slots {}",
                table_size, num_slots
            )));
        }

        let memory = provider.memory();
        let device = provider.device().inner();

        let mut keys = memory.alloc::<u64>(table_size as usize)?;
        let mut slots = memory.alloc::<u32>(table_size as usize)?;
        let mut state = memory.alloc::<u32>(table_size as usize)?;
        let mut last_used = memory.alloc::<u64>(table_size as usize)?;
        let mut slot_states = memory.alloc::<u32>(num_slots as usize)?;
        let mut clock = memory.alloc::<u64>(1)?;

        device.memset_zeros(&mut keys).map_err(|e| {
            XlogError::Kernel(format!("GpuCircuitCache zero keys failed: {}", e))
        })?;
        device.memset_zeros(&mut slots).map_err(|e| {
            XlogError::Kernel(format!("GpuCircuitCache zero slots failed: {}", e))
        })?;
        device.memset_zeros(&mut state).map_err(|e| {
            XlogError::Kernel(format!("GpuCircuitCache zero state failed: {}", e))
        })?;
        device.memset_zeros(&mut last_used).map_err(|e| {
            XlogError::Kernel(format!("GpuCircuitCache zero last_used failed: {}", e))
        })?;
        device.memset_zeros(&mut slot_states).map_err(|e| {
            XlogError::Kernel(format!("GpuCircuitCache zero slot_states failed: {}", e))
        })?;
        device.memset_zeros(&mut clock).map_err(|e| {
            XlogError::Kernel(format!("GpuCircuitCache zero clock failed: {}", e))
        })?;

        Ok(Self {
            provider: provider.clone(),
            table_size,
            num_slots,
            keys,
            slots,
            state,
            last_used,
            slot_states,
            clock,
        })
    }

    pub fn lookup_or_insert(&mut self, key: u64) -> Result<GpuCacheLookup> {
        let memory = self.provider.memory();
        let mut out_slot = memory.alloc::<u32>(1)?;
        let mut out_compile_needed = memory.alloc::<u32>(1)?;

        let func = self
            .provider
            .device()
            .inner()
            .get_func(CACHE_MODULE, cache_kernels::CACHE_LOOKUP_OR_INSERT)
            .ok_or_else(|| {
                XlogError::Kernel("cache_lookup_or_insert kernel not found".to_string())
            })?;

        unsafe {
            func.clone().launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    key,
                    self.table_size,
                    self.num_slots,
                    &mut self.keys,
                    &mut self.slots,
                    &mut self.state,
                    &mut self.last_used,
                    &mut self.slot_states,
                    &mut self.clock,
                    &mut out_slot,
                    &mut out_compile_needed,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("cache_lookup_or_insert failed: {}", e)))?;

        self.provider
            .device()
            .synchronize()
            .map_err(|e| XlogError::Kernel(format!("cache_lookup_or_insert sync failed: {}", e)))?;

        Ok(GpuCacheLookup {
            provider: self.provider.clone(),
            slot: out_slot,
            compile_needed: out_compile_needed,
        })
    }
}
