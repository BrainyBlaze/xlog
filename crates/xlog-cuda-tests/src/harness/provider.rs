//! Test context and provider setup for CUDA certification tests.

use std::sync::Arc;
use xlog_core::{MemoryBudget, Result, XlogError};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

/// Test context providing CUDA resources for certification tests.
pub struct TestContext {
    pub provider: CudaKernelProvider,
    pub device: Arc<CudaDevice>,
    pub memory: Arc<GpuMemoryManager>,
}

impl TestContext {
    /// Create test context with specific memory budget in bytes.
    pub fn with_budget(budget_bytes: usize) -> Result<Self> {
        let device_count = cudarc::driver::CudaDevice::count()
            .map_err(|e| XlogError::Kernel(format!("Failed to get device count: {}", e)))?;

        if device_count == 0 {
            return Err(XlogError::Kernel("No CUDA devices available".to_string()));
        }

        let device = Arc::new(CudaDevice::new(0)?);
        let budget = MemoryBudget::with_limit(budget_bytes as u64);
        let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
        let provider = CudaKernelProvider::new(device.clone(), memory.clone())?;

        Ok(Self { provider, device, memory })
    }

    /// Create test context with default 1GB memory budget.
    pub fn new() -> Result<Self> {
        Self::with_budget(1024 * 1024 * 1024)
    }

    /// Create test context with large 4GB memory budget for stress tests.
    pub fn large() -> Result<Self> {
        Self::with_budget(4 * 1024 * 1024 * 1024)
    }

    /// Force device synchronization and check for async errors.
    pub fn sync_and_check(&self) -> Result<()> {
        self.device.inner().synchronize()
            .map_err(|e| XlogError::Kernel(format!("Sync failed: {}", e)))?;
        Ok(())
    }

    /// Get current memory usage in bytes.
    pub fn memory_used(&self) -> usize {
        self.memory.allocated_bytes() as usize
    }

    /// Get memory budget in bytes.
    pub fn memory_budget(&self) -> usize {
        self.memory.budget().device_bytes as usize
    }

    /// Check if multi-GPU is available.
    pub fn multi_gpu_available(&self) -> bool {
        cudarc::driver::CudaDevice::count().unwrap_or(0) > 1
    }

    /// Get compute capability of current device.
    pub fn compute_capability(&self) -> (u32, u32) {
        // Default to sm_70 (Volta) as minimum supported
        (7, 0)
    }
}

/// Macro for tests requiring CUDA device - panics if not available.
#[macro_export]
macro_rules! gpu_test {
    ($name:ident, $body:expr) => {
        #[test]
        fn $name() {
            let ctx = $crate::harness::TestContext::new()
                .expect("CUDA device required for this test");
            $body(&ctx);
        }
    };
}

/// Macro for tests requiring CUDA device - skips if not available.
#[macro_export]
macro_rules! gpu_test_skip {
    ($name:ident, $body:expr) => {
        #[test]
        fn $name() {
            let ctx = match $crate::harness::TestContext::new() {
                Ok(ctx) => ctx,
                Err(_) => {
                    eprintln!("Skipping {}: no CUDA device", stringify!($name));
                    return;
                }
            };
            $body(&ctx);
        }
    };
}
