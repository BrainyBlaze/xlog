//! Stream-ordered prefix scans used by native compilation and circuit evaluation.

use cudarc::driver::LaunchConfig;
use xlog_core::{Result, XlogError};

use crate::memory::TrackedCudaSlice;
use crate::provider::{scan_kernels, SCAN_MODULE};
use crate::{CudaKernelProvider, LaunchAsync};

/// Enqueues an in-place exclusive scan without a host or device synchronization.
/// Recursive scratch storage remains owned until its queued uses have completed.
pub fn exclusive_scan_u32_inplace(
    provider: &CudaKernelProvider,
    data: &mut TrackedCudaSlice<u32>,
    n: u32,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }

    let device = provider.device().inner();
    let block_size = 256u32;

    if n <= block_size {
        let phase2 = device
            .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_PHASE2)
            .ok_or_else(|| {
                XlogError::Kernel("multiblock_scan_phase2 kernel not found".to_string())
            })?;
        unsafe {
            phase2.clone().launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut *data, n),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("multiblock_scan_phase2 failed: {}", e)))?;
        return Ok(());
    }

    let num_blocks = n.div_ceil(block_size);
    let memory = provider.memory();
    let mut block_sums = memory.alloc::<u32>(num_blocks as usize)?;

    let phase1 = device
        .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_U32_PHASE1)
        .ok_or_else(|| {
            XlogError::Kernel("multiblock_scan_u32_phase1 kernel not found".to_string())
        })?;
    unsafe {
        phase1.clone().launch(
            LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            },
            (&mut *data, &mut block_sums, n),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("multiblock_scan_u32_phase1 failed: {}", e)))?;

    if num_blocks > 1 {
        exclusive_scan_u32_inplace(provider, &mut block_sums, num_blocks)?;
    }

    let phase3 = device
        .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_PHASE3)
        .ok_or_else(|| XlogError::Kernel("multiblock_scan_phase3 kernel not found".to_string()))?;
    unsafe {
        phase3.clone().launch(
            LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            },
            (&mut *data, &block_sums, n),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("multiblock_scan_phase3 failed: {}", e)))?;

    Ok(())
}
