//! Data-driven kernel module loading.
//!
//! Replaces the 812-line hand-unrolled `new()` constructor with a loop over
//! `KERNEL_MODULES`, reducing ~800 LOC to ~30.

use std::sync::Arc;
use std::time::Instant;

use xlog_core::{Result, XlogError};

use super::{CudaKernelProvider, PtxLoadProfile};
use crate::kernel_manifest_data::KERNEL_MODULES;
use crate::CudaDevice;

impl CudaKernelProvider {
    /// Load every kernel module listed in `KERNEL_MODULES` into `device`.
    ///
    /// Returns `Some(PtxLoadProfile)` when `profiling` is true, `None` otherwise.
    pub(crate) fn load_all_kernel_modules(
        device: &Arc<CudaDevice>,
        profiling: bool,
    ) -> Result<Option<PtxLoadProfile>> {
        let cc = super::detect_compute_capability(device)?;
        let mut profile = PtxLoadProfile::default();

        for spec in KERNEL_MODULES {
            let t0 = if profiling {
                Some(Instant::now())
            } else {
                None
            };

            let (ptx, is_cubin) = super::load_module_from_file(spec.cu_name, cc)?;

            device
                .inner()
                .load_ptx(ptx, spec.module_name, spec.kernels)
                .map_err(|e| {
                    XlogError::Kernel(format!("Failed to load {} module: {}", spec.cu_name, e))
                })?;

            if let Some(t0) = t0 {
                if profiling {
                    device.inner().synchronize().map_err(|e| {
                        XlogError::Kernel(format!("sync after {} load: {}", spec.cu_name, e))
                    })?;
                }
                let elapsed = t0.elapsed().as_secs_f64();
                profile
                    .per_module_sec
                    .push((spec.cu_name.to_string(), elapsed));
                profile.total_sec += elapsed;
                if is_cubin {
                    profile.cubin_loaded += 1;
                } else {
                    profile.ptx_fallback += 1;
                }
            }
        }

        Ok(if profiling { Some(profile) } else { None })
    }
}
