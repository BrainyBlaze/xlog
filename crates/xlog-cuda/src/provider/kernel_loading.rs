//! Data-driven kernel module loading.
//!
//! Replaces the 812-line hand-unrolled `new()` constructor with a loop over
//! `KERNEL_MODULES`, reducing ~800 LOC to ~30.

use std::sync::Arc;
use std::time::Instant;

use cudarc::nvrtc::Ptx;
use xlog_core::{Result, XlogError};

use super::{CudaKernelProvider, KernelModuleSource, PtxLoadProfile};
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

            let sources = super::load_module_sources(spec.cu_name, cc)?;
            let mut loaded_from_cubin = false;
            let mut load_errors = Vec::new();

            for source in sources {
                match source {
                    KernelModuleSource::File { path, is_cubin } => {
                        match device
                            .inner()
                            .load_file(&path, spec.module_name, spec.kernels)
                        {
                            Ok(()) => {
                                loaded_from_cubin = is_cubin;
                                load_errors.clear();
                                break;
                            }
                            Err(e) => load_errors.push(format!(
                                "{} from {}: {}",
                                if is_cubin { "cubin" } else { "portable PTX" },
                                path.display(),
                                e
                            )),
                        }
                    }
                    KernelModuleSource::EmbeddedPortablePtx { ptx } => {
                        match device.inner().load_ptx(
                            Ptx::from_src(ptx),
                            spec.module_name,
                            spec.kernels,
                        ) {
                            Ok(()) => {
                                loaded_from_cubin = false;
                                load_errors.clear();
                                break;
                            }
                            Err(e) => {
                                load_errors.push(format!("embedded portable PTX: {}", e));
                            }
                        }
                    }
                }
            }
            if !load_errors.is_empty() {
                return Err(XlogError::Kernel(format!(
                    "Failed to load {} module after {} attempt(s): {}",
                    spec.cu_name,
                    load_errors.len(),
                    load_errors.join("; ")
                )));
            }

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
                if loaded_from_cubin {
                    profile.cubin_loaded += 1;
                } else {
                    profile.ptx_fallback += 1;
                }
            }
        }

        Ok(if profiling { Some(profile) } else { None })
    }
}
