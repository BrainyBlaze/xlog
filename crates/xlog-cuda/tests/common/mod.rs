use std::sync::Arc;
use xlog_core::MemoryBudget;
use xlog_cuda::device_runtime::{InMemorySink, LoggingSink, XlogDeviceRuntime};
use xlog_cuda::{CudaKernelProvider, CudaProviderBuilder, GpuMemoryManager};

/// Canonical CUDA provider for tests. Returns None if CUDA is unavailable.
#[allow(dead_code)] // not all integration test binaries use this fixture
pub fn setup_provider() -> Option<Arc<CudaKernelProvider>> {
    let result = CudaProviderBuilder::new(0, MemoryBudget::with_limit(1024 * 1024 * 1024))
        .build()
        .map(Arc::new);

    match result {
        Ok(provider) => Some(provider),
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA provider construction failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping: CUDA provider unavailable: {error}");
            None
        }
    }
}

/// Handles produced by [`setup_provider_with_runtime`]. Exposes the
/// provider plus the underlying [`XlogDeviceRuntime`] and the
/// [`InMemorySink`] that captured every alloc/dealloc/reap record,
/// so tests can both run real provider operations and inspect the
/// resulting routing through the runtime-attached allocator stack.
#[allow(dead_code)] // not all integration test binaries use every field
pub struct RuntimeProviderHandles {
    pub provider: Arc<CudaKernelProvider>,
    pub memory: Arc<GpuMemoryManager>,
    pub runtime: Arc<XlogDeviceRuntime>,
    pub sink: Arc<InMemorySink>,
}

/// Runtime-attached variant of [`setup_provider`].
///
/// Constructs the canonical recommended runtime stack —
/// `GlobalDeviceBudget(LoggingResource(AsyncCudaResource))` — wires
/// it into a [`GpuMemoryManager`] via
/// [`GpuMemoryManager::with_runtime`], then builds the provider via
/// [`CudaKernelProvider::with_runtime`] (the opt-in constructor
/// that requires a runtime-attached manager).
///
/// [`setup_provider`] remains the default; existing tests
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
    let sink: Arc<InMemorySink> = Arc::new(InMemorySink::new());
    let logging_sink: Arc<dyn LoggingSink> = sink.clone();
    let provider = match CudaProviderBuilder::new(0, MemoryBudget::with_limit(1024 * 1024 * 1024))
        .with_logging_sink(logging_sink)
        .build()
    {
        Ok(provider) => Arc::new(provider),
        Err(e) => {
            eprintln!("Skipping: canonical provider construction failed: {e}");
            return None;
        }
    };
    let memory = Arc::clone(provider.memory());
    let runtime = Arc::clone(
        memory
            .runtime()
            .expect("canonical provider must own a runtime"),
    );

    Some(RuntimeProviderHandles {
        provider,
        memory,
        runtime,
        sink,
    })
}
