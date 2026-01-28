//! GPU-resident circuit cache helpers.

use std::sync::Arc;

use cudarc::driver::{LaunchAsync, LaunchConfig};
use xlog_core::{Result, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::provider::{cache_kernels, CACHE_MODULE};
use xlog_cuda::CudaKernelProvider;
use xlog_solve::GpuCnf;

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
