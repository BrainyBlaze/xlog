//! Test context and provider setup for CUDA certification tests.

use std::sync::{Arc, Mutex, OnceLock};
use xlog_core::{MemoryBudget, Result, XlogError};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

fn gpu_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// Test context providing CUDA resources for certification tests.
pub struct TestContext {
    // Hold a process-wide mutex for the lifetime of the context so GPU tests run serially.
    // This prevents timing-based certification tests from producing false positives under
    // parallel `cargo test` execution.
    _lock: std::sync::MutexGuard<'static, ()>,
    pub provider: CudaKernelProvider,
    pub device: Arc<CudaDevice>,
    pub memory: Arc<GpuMemoryManager>,
}

impl TestContext {
    /// Create test context with specific memory budget in bytes.
    pub fn with_budget(budget_bytes: usize) -> Result<Self> {
        let lock = gpu_test_lock();

        // cudarc may panic on driver init failures in restricted containers; treat as a normal error.
        let device_count = std::panic::catch_unwind(|| cudarc::driver::CudaDevice::count())
            .map_err(|_| {
                XlogError::Kernel(
                    "Failed to get device count: cudarc panicked during driver initialization"
                        .to_string(),
                )
            })?
            .map_err(|e| XlogError::Kernel(format!("Failed to get device count: {}", e)))?;

        if device_count == 0 {
            return Err(XlogError::Kernel("No CUDA devices available".to_string()));
        }

        let device = Arc::new(CudaDevice::new(0)?);
        let budget = MemoryBudget::with_limit(budget_bytes as u64);
        let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
        let provider = CudaKernelProvider::new(device.clone(), memory.clone())?;

        Ok(Self {
            _lock: lock,
            provider,
            device,
            memory,
        })
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
        self.device
            .inner()
            .synchronize()
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
        match std::panic::catch_unwind(|| cudarc::driver::CudaDevice::count()) {
            Ok(Ok(n)) => n > 1,
            _ => false,
        }
    }

    /// Get compute capability of current device.
    pub fn compute_capability(&self) -> Result<(u32, u32)> {
        use cudarc::driver::sys::CUdevice_attribute;

        let major = self
            .device
            .inner()
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| {
                XlogError::Kernel(format!("Failed to query compute capability major: {}", e))
            })?;
        let minor = self
            .device
            .inner()
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(|e| {
                XlogError::Kernel(format!("Failed to query compute capability minor: {}", e))
            })?;

        let major_u32: u32 = major.try_into().map_err(|_| {
            XlogError::Kernel(format!(
                "Failed to convert compute capability major {} to u32",
                major
            ))
        })?;
        let minor_u32: u32 = minor.try_into().map_err(|_| {
            XlogError::Kernel(format!(
                "Failed to convert compute capability minor {} to u32",
                minor
            ))
        })?;

        Ok((major_u32, minor_u32))
    }
}

/// Macro for tests requiring CUDA device - panics if not available.
#[macro_export]
macro_rules! gpu_test {
    ($name:ident, $body:expr) => {
        #[test]
        fn $name() {
            let ctx =
                $crate::harness::TestContext::new().expect("CUDA device required for this test");
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
