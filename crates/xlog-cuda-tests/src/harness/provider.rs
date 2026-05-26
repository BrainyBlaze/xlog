//! Test context and provider setup for CUDA certification tests.

use cudarc::driver::sys;
use cudarc::driver::{DevicePtr, DevicePtrMut, DeviceRepr};
use fs2::FileExt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::{fs::OpenOptions, path::Path};
use xlog_core::{MemoryBudget, Result, XlogError};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
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
        .truncate(false)
        .read(true)
        .write(true)
        .open(path)
        .map_err(|e| XlogError::Kernel(format!("Failed to open GPU test lock file: {}", e)))?;
    file.lock_exclusive()
        .map_err(|e| XlogError::Kernel(format!("Failed to acquire GPU test lock file: {}", e)))?;
    Ok(file)
}

/// Sink that drops every log record. Used by the runtime-backed
/// `TestContext` mode: the cert harness has no need for a live
/// log stream during kernel testing, but `LoggingResource`
/// requires a sink. Errors from the sink would propagate as
/// allocation failures, which would mask kernel issues — drop
/// silently here.
struct DiscardSink;

impl LoggingSink for DiscardSink {
    fn emit(&self, _record: LogRecord) -> std::result::Result<(), SinkError> {
        Ok(())
    }
}

/// Backend the test context uses for allocation + recorded-launch
/// dispatch. Selectable via `XLOG_USE_DEVICE_RUNTIME=1` at process
/// start; default is `Legacy` so the existing cert-suite evidence
/// is unperturbed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TestRuntimeBackend {
    /// Legacy cudarc-backed `GpuMemoryManager::new`. The
    /// env-var dispatchers in `provider::sort` / `filter_by_mask`
    /// / `hash_join_v2` / etc. fall through to the legacy
    /// kernels because `provider.memory.runtime()` is `None`.
    Legacy,
    /// `XlogDeviceRuntime` with the production decorator stack
    /// (`AsyncCudaResource → LoggingResource → GlobalDeviceBudget`).
    /// `XLOG_USE_RECORDED_OPS=1` (or per-operator
    /// `XLOG_USE_RECORDED_*=1`) routes operator calls through
    /// the recorded path so cert categories that use
    /// `provider.sort` / `provider.filter_by_mask` etc. exercise
    /// the prepare/finish stream-dependency manager landed in
    /// PR #72.
    DeviceRuntime,
}

impl TestRuntimeBackend {
    /// Read `XLOG_USE_DEVICE_RUNTIME` at context construction;
    /// any non-empty truthy value selects the runtime stack.
    /// Cached per-process via `OnceLock` so every `TestContext`
    /// in a single test binary agrees on the backend.
    fn from_env() -> Self {
        static CACHED: OnceLock<TestRuntimeBackend> = OnceLock::new();
        *CACHED.get_or_init(|| match std::env::var("XLOG_USE_DEVICE_RUNTIME") {
            Ok(v) if matches!(v.as_str(), "1" | "true" | "TRUE" | "True") => {
                TestRuntimeBackend::DeviceRuntime
            }
            _ => TestRuntimeBackend::Legacy,
        })
    }
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
    /// Backend selected for this context — informational, used
    /// by tests that want to assert which stack is active.
    backend: TestRuntimeBackend,
    /// Held so the `XlogDeviceRuntime`'s stream pool outlives
    /// the context when the runtime backend is active.
    /// `None` in legacy mode.
    _runtime: Option<Arc<XlogDeviceRuntime>>,
    transfer: Arc<TransferCounters>,
}

impl TestContext {
    /// Create test context with specific memory budget in bytes.
    /// Backend is chosen by `XLOG_USE_DEVICE_RUNTIME` (default
    /// legacy).
    pub fn with_budget(budget_bytes: usize) -> Result<Self> {
        let lock = gpu_test_lock();
        let file_lock = gpu_test_lock_file()?;

        // cudarc may panic on driver init failures in restricted containers; treat as a normal error.
        let device_count = std::panic::catch_unwind(CudaDevice::count)
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
        let backend = TestRuntimeBackend::from_env();
        let transfer = Arc::new(TransferCounters::default());

        let (memory, provider, runtime_arc) = match backend {
            TestRuntimeBackend::Legacy => {
                let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
                let provider = CudaKernelProvider::new(device.clone(), memory.clone())?;
                (memory, provider, None)
            }
            TestRuntimeBackend::DeviceRuntime => {
                // Build the same decorator stack production callers
                // (and the integration suite under
                // XLOG_USE_DEVICE_RUNTIME=1) use, so cert
                // categories that touch `provider.sort` /
                // `filter_by_mask` etc. exercise the recorded
                // launch discipline once `XLOG_USE_RECORDED_*`
                // is also set.
                let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
                let async_resource: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(
                    AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool)),
                );
                let logging: Box<dyn DeviceMemoryResource + Send + Sync> =
                    Box::new(LoggingResource::new(
                        async_resource,
                        Arc::new(DiscardSink) as Arc<dyn LoggingSink>,
                    ));
                let budget_resource: Box<dyn DeviceMemoryResource + Send + Sync> =
                    Box::new(GlobalDeviceBudget::new(logging, budget_bytes));
                let runtime = Arc::new(XlogDeviceRuntime::with_resource(
                    Arc::clone(&device),
                    0,
                    Arc::clone(&pool),
                    budget_resource,
                ));
                let memory = Arc::new(GpuMemoryManager::with_runtime(
                    Arc::clone(&device),
                    budget,
                    Arc::clone(&runtime),
                ));
                let provider =
                    CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory))?;
                (memory, provider, Some(runtime))
            }
        };

        Ok(Self {
            _lock: lock,
            _file_lock: file_lock,
            provider,
            device,
            memory,
            backend,
            _runtime: runtime_arc,
            transfer,
        })
    }

    /// Return whether this context was built against the
    /// runtime-backed allocator stack (vs the legacy cudarc-only
    /// stack). Cert categories can use this to gate behavior or
    /// diagnostics.
    pub fn uses_device_runtime(&self) -> bool {
        self.backend == TestRuntimeBackend::DeviceRuntime
    }

    /// Drain pending async frees on the runtime allocator, if
    /// any. No-op for the legacy backend. Cert harnesses call
    /// this between categories so the `GlobalDeviceBudget`
    /// reservation bookkeeping releases bytes that have been
    /// freed via `cuMemFreeAsync` but whose owning stream has
    /// not yet been synchronized — without this, a long
    /// sequence of small allocate-then-drop tests piles up
    /// pending frees and exhausts the reservation pool even
    /// though the GPU has plenty of free memory.
    pub fn reap_pending(&self) {
        if let Some(rt) = &self._runtime {
            // Best-effort: a transient driver error during reap
            // is recoverable on the next iteration, and the cert
            // suite already runs categories sequentially under
            // the harness lock — propagating an error here would
            // tear down the entire suite for what is a
            // bookkeeping issue, not a kernel correctness one.
            let _ = rt.reap_pending();
        }
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
        let stream = self.device.inner().stream();
        let start = stream
            .context()
            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| XlogError::Kernel(format!("Failed to create start event: {}", e)))?;
        let end = stream
            .context()
            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| XlogError::Kernel(format!("Failed to create end event: {}", e)))?;

        start
            .record(stream)
            .map_err(|e| XlogError::Kernel(format!("Failed to record start event: {}", e)))?;
        f()?;
        end.record(stream)
            .map_err(|e| XlogError::Kernel(format!("Failed to record end event: {}", e)))?;
        self.sync_and_check()?;
        let elapsed = start.elapsed_ms(&end).map_err(|e| {
            XlogError::Kernel(format!("Failed to measure event elapsed time: {}", e))
        })?;
        Ok(elapsed)
    }

    /// Check if multi-GPU is available.
    pub fn multi_gpu_available(&self) -> bool {
        match std::panic::catch_unwind(CudaDevice::count) {
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
