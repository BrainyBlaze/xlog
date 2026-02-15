//! Test context and provider setup for CUDA certification tests.

use cudarc::driver::{result, sys};
use cudarc::driver::{DevicePtr, DevicePtrMut, DeviceRepr};
use fs2::FileExt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::{fs::OpenOptions, path::Path};
use xlog_core::{MemoryBudget, Result, XlogError};
use xlog_cuda::{CudaBuffer, CudaDevice, CudaKernelProvider, GpuMemoryManager};

#[derive(Default)]
struct TransferCounters {
    htod_bytes: AtomicU64,
    dtoh_bytes: AtomicU64,
}

impl TransferCounters {
    fn reset(&self) {
        self.htod_bytes.store(0, Ordering::SeqCst);
        self.dtoh_bytes.store(0, Ordering::SeqCst);
    }

    fn add_htod(&self, bytes: u64) {
        self.htod_bytes.fetch_add(bytes, Ordering::SeqCst);
    }

    fn add_dtoh(&self, bytes: u64) {
        self.dtoh_bytes.fetch_add(bytes, Ordering::SeqCst);
    }

    fn snapshot(&self) -> (u64, u64) {
        (
            self.htod_bytes.load(Ordering::SeqCst),
            self.dtoh_bytes.load(Ordering::SeqCst),
        )
    }
}

const GPU_TEST_LOCK_PATH: &str = "/tmp/xlog-gpu-tests.lock";

fn gpu_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

fn gpu_test_lock_file() -> Result<std::fs::File> {
    let path = Path::new(GPU_TEST_LOCK_PATH);
    let file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(path)
        .map_err(|e| XlogError::Kernel(format!("Failed to open GPU test lock file: {}", e)))?;
    file.lock_exclusive()
        .map_err(|e| XlogError::Kernel(format!("Failed to acquire GPU test lock file: {}", e)))?;
    Ok(file)
}

/// Test context providing CUDA resources for certification tests.
pub struct TestContext {
    // Hold a process-wide mutex for the lifetime of the context so GPU tests run serially.
    // This prevents timing-based certification tests from producing false positives under
    // parallel `cargo test` execution.
    _lock: std::sync::MutexGuard<'static, ()>,
    _file_lock: std::fs::File,
    pub provider: CudaKernelProvider,
    pub device: Arc<CudaDevice>,
    pub memory: Arc<GpuMemoryManager>,
    transfer: Arc<TransferCounters>,
}

impl TestContext {
    /// Create test context with specific memory budget in bytes.
    pub fn with_budget(budget_bytes: usize) -> Result<Self> {
        let lock = gpu_test_lock();
        let file_lock = gpu_test_lock_file()?;

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
        let transfer = Arc::new(TransferCounters::default());

        Ok(Self {
            _lock: lock,
            _file_lock: file_lock,
            provider,
            device,
            memory,
            transfer,
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

    /// Reset host/device transfer counters (in bytes).
    pub fn reset_transfer_counters(&self) {
        self.transfer.reset();
    }

    /// Snapshot transfer counters (htod_bytes, dtoh_bytes).
    pub fn transfer_counters(&self) -> (u64, u64) {
        self.transfer.snapshot()
    }

    /// Track a synchronous host-to-device copy.
    pub fn htod_sync_copy_into<T: DeviceRepr, Dst: DevicePtrMut<T>>(
        &self,
        src: &[T],
        dst: &mut Dst,
    ) -> Result<()> {
        let bytes = std::mem::size_of::<T>()
            .checked_mul(src.len())
            .ok_or_else(|| XlogError::Kernel("htod byte count overflow".to_string()))?;
        self.transfer.add_htod(bytes as u64);
        self.device
            .inner()
            .htod_sync_copy_into(src, dst)
            .map_err(|e| XlogError::Kernel(format!("Failed htod copy: {}", e)))?;
        Ok(())
    }

    /// Track a synchronous device-to-host copy returning a Vec.
    pub fn dtoh_sync_copy<T: DeviceRepr, Src: DevicePtr<T>>(&self, src: &Src) -> Result<Vec<T>> {
        let bytes = std::mem::size_of::<T>()
            .checked_mul(src.len())
            .ok_or_else(|| XlogError::Kernel("dtoh byte count overflow".to_string()))?;
        self.transfer.add_dtoh(bytes as u64);
        self.device
            .inner()
            .dtoh_sync_copy(src)
            .map_err(|e| XlogError::Kernel(format!("Failed dtoh copy: {}", e)))
    }

    /// Track a synchronous device-to-host copy into a slice.
    pub fn dtoh_sync_copy_into<T: DeviceRepr, Src: DevicePtr<T>>(
        &self,
        src: &Src,
        dst: &mut [T],
    ) -> Result<()> {
        let bytes = std::mem::size_of::<T>()
            .checked_mul(dst.len())
            .ok_or_else(|| XlogError::Kernel("dtoh byte count overflow".to_string()))?;
        self.transfer.add_dtoh(bytes as u64);
        self.device
            .inner()
            .dtoh_sync_copy_into(src, dst)
            .map_err(|e| XlogError::Kernel(format!("Failed dtoh copy: {}", e)))?;
        Ok(())
    }

    /// Measure GPU time (in milliseconds) for work enqueued on the device stream.
    pub fn measure_gpu_ms<F>(&self, f: F) -> Result<f32>
    where
        F: FnOnce() -> Result<()>,
    {
        let device = self.device.inner();
        let stream = *device.cu_stream();
        let start = result::event::create(sys::CUevent_flags::CU_EVENT_DEFAULT)
            .map_err(|e| XlogError::Kernel(format!("Failed to create start event: {}", e)))?;
        let end = result::event::create(sys::CUevent_flags::CU_EVENT_DEFAULT)
            .map_err(|e| XlogError::Kernel(format!("Failed to create end event: {}", e)))?;

        unsafe {
            result::event::record(start, stream)
                .map_err(|e| XlogError::Kernel(format!("Failed to record start event: {}", e)))?;
        }
        f()?;
        unsafe {
            result::event::record(end, stream)
                .map_err(|e| XlogError::Kernel(format!("Failed to record end event: {}", e)))?;
        }
        self.sync_and_check()?;
        let elapsed = unsafe { result::event::elapsed(start, end) }.map_err(|e| {
            XlogError::Kernel(format!("Failed to measure event elapsed time: {}", e))
        })?;
        unsafe {
            result::event::destroy(start)
                .map_err(|e| XlogError::Kernel(format!("Failed to destroy start event: {}", e)))?;
            result::event::destroy(end)
                .map_err(|e| XlogError::Kernel(format!("Failed to destroy end event: {}", e)))?;
        }
        Ok(elapsed)
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

    /// Read device-resident row count for a buffer.
    ///
    /// Panics if the count cannot be read; certification tests treat this as fatal.
    pub fn device_row_count(&self, buffer: &CudaBuffer) -> u64 {
        let mut host_rows = [0u32];
        self.dtoh_sync_copy_into(buffer.num_rows_device(), &mut host_rows)
            .unwrap_or_else(|e| panic!("Failed to read device row count: {}", e));
        host_rows[0] as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    #[test]
    fn test_gpu_test_lock_file_is_exclusive() {
        let ctx = match TestContext::new() {
            Ok(ctx) => ctx,
            Err(_) => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let status = Command::new("flock")
            .arg("-n")
            .arg(GPU_TEST_LOCK_PATH)
            .arg("-c")
            .arg("true")
            .status();

        match status {
            Ok(status) => assert!(
                !status.success(),
                "GPU test lock file should be held exclusively while TestContext is alive"
            ),
            Err(_) => {
                eprintln!("Skipping test: flock command not available");
            }
        }

        drop(ctx);
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
