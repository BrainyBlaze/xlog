//! Probabilistic operations: Monte Carlo sampling (Bernoulli matrix).

use crate::{CudaView, LaunchAsync, LaunchConfig};
use xlog_core::{Result, XlogError};

use super::{mc_sample_kernels, MC_SAMPLE_MODULE};
use crate::memory::TrackedCudaSlice;

impl super::CudaKernelProvider {
    /// Sample independent Bernoulli variables on the GPU.
    ///
    /// Returns a row-major `(sample, var)` matrix as a flat `Vec<u8>` of length
    /// `num_samples * probs.len()`, where each entry is 0/1.
    pub fn sample_bernoulli_matrix(
        &self,
        probs: &[f32],
        num_samples: usize,
        seed: u64,
        force_mask: &CudaView<u8>,
        forced_value: &CudaView<u8>,
    ) -> Result<Vec<u8>> {
        if probs.is_empty() || num_samples == 0 {
            return Ok(Vec::new());
        }

        let num_vars_u32: u32 = probs.len().try_into().map_err(|_| {
            XlogError::Kernel(format!(
                "sample_bernoulli_matrix: num_vars {} exceeds u32::MAX",
                probs.len()
            ))
        })?;
        let num_samples_u32: u32 = num_samples.try_into().map_err(|_| {
            XlogError::Kernel(format!(
                "sample_bernoulli_matrix: num_samples {} exceeds u32::MAX",
                num_samples
            ))
        })?;

        let total = probs.len().checked_mul(num_samples).ok_or_else(|| {
            XlogError::Kernel("sample_bernoulli_matrix: size overflow".to_string())
        })?;

        let device = self.device.inner();

        let mut d_probs = self.memory.alloc::<f32>(probs.len())?;
        self.htod_sync_copy_into_tracked(probs, &mut d_probs)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload Bernoulli probs: {}", e)))?;

        let mut d_out = self.memory.alloc::<u8>(total)?;

        let block_size = 256u32;
        let total_u32: u32 = total.try_into().map_err(|_| {
            XlogError::Kernel(format!(
                "sample_bernoulli_matrix: total {} exceeds u32::MAX",
                total
            ))
        })?;
        let num_blocks = total_u32.div_ceil(block_size);
        let config = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let kernel = device
            .get_func(MC_SAMPLE_MODULE, mc_sample_kernels::MC_SAMPLE_BERNOULLI)
            .ok_or_else(|| XlogError::Kernel("mc_sample_bernoulli kernel not found".to_string()))?;

        // SAFETY: mc_sample_bernoulli(out, probs, force_mask, forced_value, num_vars, num_samples, seed)
        unsafe {
            kernel.clone().launch(
                config,
                (
                    &mut d_out,
                    &d_probs,
                    force_mask,
                    forced_value,
                    num_vars_u32,
                    num_samples_u32,
                    seed,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("Failed to launch mc_sample_bernoulli: {}", e)))?;

        let mut host: Vec<u8> = vec![0u8; total];
        device.dtoh_sync_copy_into(&d_out, &mut host).map_err(|e| {
            XlogError::Kernel(format!("Failed to download Bernoulli samples: {}", e))
        })?;

        Ok(host)
    }

    /// Sample Bernoulli matrix on GPU and return device-resident output.
    ///
    /// Returns a row-major [num_samples][num_vars] matrix of 0/1 bytes on device.
    pub fn sample_bernoulli_matrix_device(
        &self,
        probs: &[f32],
        num_samples: usize,
        seed: u64,
        force_mask: &CudaView<u8>,
        forced_value: &CudaView<u8>,
    ) -> Result<TrackedCudaSlice<u8>> {
        if probs.is_empty() || num_samples == 0 {
            return self.memory.alloc::<u8>(0).map_err(|e| {
                XlogError::Kernel(format!("Failed to allocate empty sample matrix: {}", e))
            });
        }

        let num_vars_u32: u32 = probs.len().try_into().map_err(|_| {
            XlogError::Kernel(format!(
                "sample_bernoulli_matrix_device: num_vars {} exceeds u32::MAX",
                probs.len()
            ))
        })?;
        let num_samples_u32: u32 = num_samples.try_into().map_err(|_| {
            XlogError::Kernel(format!(
                "sample_bernoulli_matrix_device: num_samples {} exceeds u32::MAX",
                num_samples
            ))
        })?;

        let total = probs.len().saturating_mul(num_samples);
        let device = self.device.inner();

        let mut d_probs = self.memory.alloc::<f32>(probs.len())?;
        self.htod_sync_copy_into_tracked(probs, &mut d_probs)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload Bernoulli probs: {}", e)))?;

        let mut d_out = self.memory.alloc::<u8>(total)?;

        let block_size = 256u32;
        let total_u32: u32 = total.try_into().map_err(|_| {
            XlogError::Kernel(format!(
                "sample_bernoulli_matrix_device: total {} exceeds u32::MAX",
                total
            ))
        })?;
        let num_blocks = total_u32.div_ceil(block_size);
        let config = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let kernel = device
            .get_func(MC_SAMPLE_MODULE, mc_sample_kernels::MC_SAMPLE_BERNOULLI)
            .ok_or_else(|| XlogError::Kernel("mc_sample_bernoulli kernel not found".to_string()))?;

        // SAFETY: mc_sample_bernoulli(out, probs, force_mask, forced_value, num_vars, num_samples, seed)
        unsafe {
            kernel.clone().launch(
                config,
                (
                    &mut d_out,
                    &d_probs,
                    force_mask,
                    forced_value,
                    num_vars_u32,
                    num_samples_u32,
                    seed,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("Failed to launch mc_sample_bernoulli: {}", e)))?;

        Ok(d_out)
    }
}
