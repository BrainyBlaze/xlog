use std::sync::Arc;
use xlog_core::MemoryBudget;
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, InMemorySink, LoggingResource,
    LoggingSink, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

/// Canonical CUDA provider for tests. Returns None if CUDA is unavailable.
#[allow(dead_code)] // not all integration test binaries use this fixture
pub fn setup_provider() -> Option<Arc<CudaKernelProvider>> {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping: CUDA runtime unavailable: {}", e);
            return None;
        }
    };
    let memory = Arc::new(GpuMemoryManager::new(
        device.clone(),
        MemoryBudget::with_limit(1024 * 1024 * 1024),
    ));
    CudaKernelProvider::new(device, memory).ok().map(Arc::new)
}

/// Handles produced by [`setup_provider_with_runtime`]. Exposes the
/// provider plus the underlying [`XlogDeviceRuntime`] and the
/// [`InMemorySink`] that captured every alloc/dealloc/reap record,
/// so tests can both run real provider operations and inspect the
/// resulting routing through the v0.6 stack.
#[allow(dead_code)] // not all integration test binaries use every field
pub struct RuntimeProviderHandles {
    pub provider: Arc<CudaKernelProvider>,
    pub memory: Arc<GpuMemoryManager>,
    pub runtime: Arc<XlogDeviceRuntime>,
    pub sink: Arc<InMemorySink>,
}

/// Opt-in v0.6 variant of [`setup_provider`].
///
/// Constructs the canonical recommended runtime stack —
/// `GlobalDeviceBudget(LoggingResource(AsyncCudaResource))` — wires
/// it into a [`GpuMemoryManager`] via
/// [`GpuMemoryManager::with_runtime`], then builds the provider via
/// [`CudaKernelProvider::with_runtime`] (the opt-in constructor
/// that requires a runtime-attached manager).
///
/// [`setup_provider`] remains the legacy default; existing tests
/// that do not need to observe runtime routing are unaffected.
/// Tests that opt into this fixture get the same
/// `Arc<CudaKernelProvider>` shape they are used to, plus the
/// additional handles required to assert on runtime budget /
/// logging behavior.
///
/// Returns `None` when CUDA is unavailable, mirroring
/// [`setup_provider`].
#[allow(dead_code)] // exercised by tests in other binaries
pub fn setup_provider_with_runtime() -> Option<RuntimeProviderHandles> {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping: CUDA runtime unavailable: {}", e);
            return None;
        }
    };
    let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
    let sink: Arc<InMemorySink> = Arc::new(InMemorySink::new());

    let async_resource: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(
        AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool)),
    );
    let logging: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(LoggingResource::new(
        async_resource,
        sink.clone() as Arc<dyn LoggingSink>,
    ));
    // Generous runtime budget so the fixture is reusable across
    // tests with varied allocation sizes; tighter budgets belong
    // in tests that specifically exercise rejection.
    let budget: Box<dyn DeviceMemoryResource + Send + Sync> =
        Box::new(GlobalDeviceBudget::new(logging, 1024 * 1024 * 1024));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));

    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(1024 * 1024 * 1024),
        Arc::clone(&runtime),
    ));

    let provider = match CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory))
    {
        Ok(p) => Arc::new(p),
        Err(e) => {
            eprintln!("Skipping: provider with_runtime construction failed: {}", e);
            return None;
        }
    };

    Some(RuntimeProviderHandles {
        provider,
        memory,
        runtime,
        sink,
    })
}
