//! CUDA kernel provider implementation
//!
//! This module provides the `CudaKernelProvider` which manages pre-compiled
//! PTX kernels for GPU execution of relational operations (join, dedup, groupby).

use std::sync::Arc;

use cudarc::nvrtc::Ptx;
use xlog_core::{Result, XlogError};

use crate::{CudaDevice, GpuMemoryManager};

// Embedded PTX sources (pre-compiled from .cu files with nvcc -ptx --gpu-architecture=sm_90)
const JOIN_PTX: &str = include_str!("../../../kernels/join.ptx");
const DEDUP_PTX: &str = include_str!("../../../kernels/dedup.ptx");
const GROUPBY_PTX: &str = include_str!("../../../kernels/groupby.ptx");

/// Module names for loaded PTX modules
pub const JOIN_MODULE: &str = "xlog_join";
pub const DEDUP_MODULE: &str = "xlog_dedup";
pub const GROUPBY_MODULE: &str = "xlog_groupby";

/// Kernel function names in the join module
pub mod join_kernels {
    pub const HASH_JOIN_BUILD: &str = "hash_join_build";
    pub const HASH_JOIN_PROBE: &str = "hash_join_probe";
}

/// Kernel function names in the dedup module
pub mod dedup_kernels {
    pub const MARK_DUPLICATES: &str = "mark_duplicates";
    pub const COMPACT_ROWS: &str = "compact_rows";
}

/// Kernel function names in the groupby module
pub mod groupby_kernels {
    pub const DETECT_GROUP_BOUNDARIES: &str = "detect_group_boundaries";
    pub const GROUPBY_COUNT: &str = "groupby_count";
    pub const GROUPBY_SUM: &str = "groupby_sum";
    pub const GROUPBY_MIN: &str = "groupby_min";
    pub const GROUPBY_MAX: &str = "groupby_max";
}

/// CUDA kernel provider for xlog GPU operations
///
/// Manages pre-compiled PTX modules for relational operations:
/// - **Join**: Hash join with build/probe phases
/// - **Dedup**: Sort-based deduplication with prefix-sum compaction
/// - **GroupBy**: Sorted-input group aggregation (count, sum, min, max)
///
/// PTX modules are loaded at construction time and stored in the CUDA device.
/// Kernel functions can be retrieved using `device.get_func()`.
///
/// # Example
/// ```ignore
/// use std::sync::Arc;
/// use xlog_cuda::{CudaDevice, GpuMemoryManager, CudaKernelProvider};
/// use xlog_core::MemoryBudget;
///
/// let device = Arc::new(CudaDevice::new(0)?);
/// let memory = Arc::new(GpuMemoryManager::new(device.clone(), MemoryBudget::default()));
/// let provider = CudaKernelProvider::new(device, memory)?;
/// ```
pub struct CudaKernelProvider {
    /// The CUDA device with loaded PTX modules
    device: Arc<CudaDevice>,
    /// GPU memory manager for kernel allocations
    memory: Arc<GpuMemoryManager>,
}

impl CudaKernelProvider {
    /// Create a new CUDA kernel provider
    ///
    /// Loads all PTX modules (join, dedup, groupby) into the CUDA device.
    /// The modules are compiled for sm_90 (H200/Hopper architecture).
    ///
    /// # Arguments
    /// * `device` - The CUDA device to load modules into
    /// * `memory` - The GPU memory manager for kernel allocations
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if PTX loading fails
    ///
    /// # Example
    /// ```ignore
    /// let device = Arc::new(CudaDevice::new(0)?);
    /// let memory = Arc::new(GpuMemoryManager::new(device.clone(), MemoryBudget::default()));
    /// let provider = CudaKernelProvider::new(device, memory)?;
    /// ```
    pub fn new(device: Arc<CudaDevice>, memory: Arc<GpuMemoryManager>) -> Result<Self> {
        // Load join module
        device
            .inner()
            .load_ptx(
                Ptx::from_src(JOIN_PTX),
                JOIN_MODULE,
                &[join_kernels::HASH_JOIN_BUILD, join_kernels::HASH_JOIN_PROBE],
            )
            .map_err(|e| XlogError::Kernel(format!("Failed to load join PTX: {}", e)))?;

        // Load dedup module
        device
            .inner()
            .load_ptx(
                Ptx::from_src(DEDUP_PTX),
                DEDUP_MODULE,
                &[dedup_kernels::MARK_DUPLICATES, dedup_kernels::COMPACT_ROWS],
            )
            .map_err(|e| XlogError::Kernel(format!("Failed to load dedup PTX: {}", e)))?;

        // Load groupby module
        device
            .inner()
            .load_ptx(
                Ptx::from_src(GROUPBY_PTX),
                GROUPBY_MODULE,
                &[
                    groupby_kernels::DETECT_GROUP_BOUNDARIES,
                    groupby_kernels::GROUPBY_COUNT,
                    groupby_kernels::GROUPBY_SUM,
                    groupby_kernels::GROUPBY_MIN,
                    groupby_kernels::GROUPBY_MAX,
                ],
            )
            .map_err(|e| XlogError::Kernel(format!("Failed to load groupby PTX: {}", e)))?;

        Ok(Self { device, memory })
    }

    /// Get the CUDA device
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Get the GPU memory manager
    pub fn memory(&self) -> &Arc<GpuMemoryManager> {
        &self.memory
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlog_core::MemoryBudget;

    fn has_cuda_device() -> bool {
        cudarc::driver::CudaDevice::count().unwrap_or(0) > 0
    }

    #[test]
    fn test_ptx_embedded() {
        // Verify PTX sources are embedded and non-empty
        assert!(!JOIN_PTX.is_empty(), "JOIN_PTX should not be empty");
        assert!(!DEDUP_PTX.is_empty(), "DEDUP_PTX should not be empty");
        assert!(!GROUPBY_PTX.is_empty(), "GROUPBY_PTX should not be empty");

        // Verify PTX contains expected kernel names
        assert!(
            JOIN_PTX.contains("hash_join_build"),
            "JOIN_PTX should contain hash_join_build"
        );
        assert!(
            JOIN_PTX.contains("hash_join_probe"),
            "JOIN_PTX should contain hash_join_probe"
        );
        assert!(
            DEDUP_PTX.contains("mark_duplicates"),
            "DEDUP_PTX should contain mark_duplicates"
        );
        assert!(
            DEDUP_PTX.contains("compact_rows"),
            "DEDUP_PTX should contain compact_rows"
        );
        assert!(
            GROUPBY_PTX.contains("detect_group_boundaries"),
            "GROUPBY_PTX should contain detect_group_boundaries"
        );
        assert!(
            GROUPBY_PTX.contains("groupby_count"),
            "GROUPBY_PTX should contain groupby_count"
        );
        assert!(
            GROUPBY_PTX.contains("groupby_sum"),
            "GROUPBY_PTX should contain groupby_sum"
        );
        assert!(
            GROUPBY_PTX.contains("groupby_min"),
            "GROUPBY_PTX should contain groupby_min"
        );
        assert!(
            GROUPBY_PTX.contains("groupby_max"),
            "GROUPBY_PTX should contain groupby_max"
        );
    }

    #[test]
    fn test_ptx_target_architecture() {
        // Verify PTX is compiled for sm_90
        assert!(
            JOIN_PTX.contains(".target sm_90"),
            "JOIN_PTX should target sm_90"
        );
        assert!(
            DEDUP_PTX.contains(".target sm_90"),
            "DEDUP_PTX should target sm_90"
        );
        assert!(
            GROUPBY_PTX.contains(".target sm_90"),
            "GROUPBY_PTX should target sm_90"
        );
    }

    #[test]
    fn test_kernel_provider_creation() {
        if !has_cuda_device() {
            eprintln!("Skipping test: no CUDA device available");
            return;
        }

        let device = Arc::new(CudaDevice::new(0).expect("Failed to create device"));
        let budget = MemoryBudget::with_limit(1024 * 1024 * 1024); // 1 GB
        let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));

        let provider = CudaKernelProvider::new(device.clone(), memory.clone());
        assert!(
            provider.is_ok(),
            "Failed to create kernel provider: {:?}",
            provider.err()
        );

        let provider = provider.unwrap();
        assert!(Arc::ptr_eq(provider.device(), &device));
        assert!(Arc::ptr_eq(provider.memory(), &memory));
    }

    #[test]
    fn test_kernel_functions_accessible() {
        if !has_cuda_device() {
            eprintln!("Skipping test: no CUDA device available");
            return;
        }

        let device = Arc::new(CudaDevice::new(0).expect("Failed to create device"));
        let budget = MemoryBudget::with_limit(1024 * 1024 * 1024);
        let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));

        let _provider = CudaKernelProvider::new(device.clone(), memory).expect("Failed to create provider");

        // Verify all kernel functions can be retrieved
        let inner = device.inner();

        // Join kernels
        let build_fn = inner.get_func(JOIN_MODULE, join_kernels::HASH_JOIN_BUILD);
        assert!(
            build_fn.is_some(),
            "hash_join_build function should be accessible"
        );

        let probe_fn = inner.get_func(JOIN_MODULE, join_kernels::HASH_JOIN_PROBE);
        assert!(
            probe_fn.is_some(),
            "hash_join_probe function should be accessible"
        );

        // Dedup kernels
        let mark_fn = inner.get_func(DEDUP_MODULE, dedup_kernels::MARK_DUPLICATES);
        assert!(
            mark_fn.is_some(),
            "mark_duplicates function should be accessible"
        );

        let compact_fn = inner.get_func(DEDUP_MODULE, dedup_kernels::COMPACT_ROWS);
        assert!(
            compact_fn.is_some(),
            "compact_rows function should be accessible"
        );

        // GroupBy kernels
        let boundaries_fn = inner.get_func(GROUPBY_MODULE, groupby_kernels::DETECT_GROUP_BOUNDARIES);
        assert!(
            boundaries_fn.is_some(),
            "detect_group_boundaries function should be accessible"
        );

        let count_fn = inner.get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_COUNT);
        assert!(
            count_fn.is_some(),
            "groupby_count function should be accessible"
        );

        let sum_fn = inner.get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_SUM);
        assert!(sum_fn.is_some(), "groupby_sum function should be accessible");

        let min_fn = inner.get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_MIN);
        assert!(min_fn.is_some(), "groupby_min function should be accessible");

        let max_fn = inner.get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_MAX);
        assert!(max_fn.is_some(), "groupby_max function should be accessible");
    }

    #[test]
    fn test_module_names_unique() {
        // Ensure module names don't collide
        assert_ne!(JOIN_MODULE, DEDUP_MODULE);
        assert_ne!(JOIN_MODULE, GROUPBY_MODULE);
        assert_ne!(DEDUP_MODULE, GROUPBY_MODULE);
    }
}
