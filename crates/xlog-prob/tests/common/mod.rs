use std::sync::Arc;
use xlog_core::MemoryBudget;
use xlog_cuda::{CudaKernelProvider, CudaProviderBuilder};

/// Canonical CUDA provider for tests. Returns None if CUDA is unavailable.
pub fn setup_provider() -> Option<Arc<CudaKernelProvider>> {
    CudaProviderBuilder::new(0, MemoryBudget::with_limit(1024 * 1024 * 1024))
        .build()
        .ok()
        .map(Arc::new)
}
