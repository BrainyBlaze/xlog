use std::sync::Arc;

use xlog_core::{MemoryBudget, Result, XlogError};

use super::CudaKernelProvider;
use crate::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LoggingResource, LoggingSink,
    StreamPool, XlogDeviceRuntime,
};
use crate::{CudaDevice, GpuMemoryManager};

/// Builds the complete production CUDA provider ownership graph.
///
/// Every provider created here owns one device handle, one stream pool, one
/// asynchronous resource stack, one runtime, and one memory manager. The same
/// handles are shared through that graph; callers cannot inject mismatched
/// allocator or device components.
pub struct CudaProviderBuilder {
    device_ordinal: usize,
    memory_budget: MemoryBudget,
    runtime_budget_bytes: Option<u64>,
    logging_sink: Option<Arc<dyn LoggingSink>>,
    stream_capacity: Option<usize>,
}

impl CudaProviderBuilder {
    /// Select a CUDA device and the byte budget enforced by both the memory
    /// manager and the runtime resource stack.
    pub fn new(device_ordinal: usize, memory_budget: MemoryBudget) -> Self {
        Self {
            device_ordinal,
            memory_budget,
            runtime_budget_bytes: None,
            logging_sink: None,
            stream_capacity: None,
        }
    }

    /// Record resource lifecycle events in `sink` without changing allocation
    /// or budget semantics.
    pub fn with_logging_sink(mut self, sink: Arc<dyn LoggingSink>) -> Self {
        self.logging_sink = Some(sink);
        self
    }

    /// Set the global runtime-resource byte limit independently of the memory
    /// manager limit. By default both layers use `memory_budget.device_bytes`.
    pub fn with_runtime_budget_limit(mut self, bytes: u64) -> Self {
        self.runtime_budget_bytes = Some(bytes);
        self
    }

    /// Set the maximum number of pooled streams for workloads that need more
    /// concurrent lanes than the production default.
    pub fn with_stream_capacity(mut self, stream_capacity: usize) -> Self {
        self.stream_capacity = Some(stream_capacity);
        self
    }

    /// Construct and validate the complete provider-owned runtime.
    pub fn build(self) -> Result<CudaKernelProvider> {
        let runtime_ordinal =
            u32::try_from(self.device_ordinal).map_err(|_| XlogError::Configuration {
                name: "device_ordinal".to_string(),
                value: self.device_ordinal.to_string(),
                expected: "an integer representable as u32",
            })?;
        let runtime_budget_limit = checked_runtime_budget_limit(self.runtime_budget_bytes())?;

        let device = Arc::new(CudaDevice::new(self.device_ordinal)?);
        let stream_pool = Arc::new(match self.stream_capacity {
            Some(stream_capacity) => StreamPool::new(Arc::clone(&device), stream_capacity),
            None => StreamPool::with_defaults(Arc::clone(&device)),
        });
        let asynchronous = AsyncCudaResource::new(
            Arc::clone(&device),
            runtime_ordinal,
            Arc::clone(&stream_pool),
        );
        debug_assert!(Arc::ptr_eq(asynchronous.stream_pool(), &stream_pool));

        let mut resource: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(asynchronous);
        if let Some(sink) = self.logging_sink {
            resource = Box::new(LoggingResource::new(resource, sink));
        }
        resource = Box::new(GlobalDeviceBudget::new(resource, runtime_budget_limit));

        let runtime = Arc::new(XlogDeviceRuntime::with_resource(
            Arc::clone(&device),
            runtime_ordinal,
            stream_pool,
            resource,
        ));
        let memory = Arc::new(GpuMemoryManager::with_runtime(
            Arc::clone(&device),
            self.memory_budget,
            runtime,
        ));

        CudaKernelProvider::from_runtime_parts(device, memory)
    }

    fn runtime_budget_bytes(&self) -> u64 {
        self.runtime_budget_bytes
            .unwrap_or(self.memory_budget.device_bytes)
    }
}

fn checked_runtime_budget_limit(bytes: u64) -> Result<usize> {
    checked_runtime_budget_limit_for_platform(bytes, usize::MAX as u64)
}

fn checked_runtime_budget_limit_for_platform(bytes: u64, platform_max: u64) -> Result<usize> {
    if bytes > platform_max {
        return Err(XlogError::Configuration {
            name: "memory_budget.device_bytes".to_string(),
            value: bytes.to_string(),
            expected: "a byte count representable as usize",
        });
    }
    usize::try_from(bytes).map_err(|_| XlogError::Configuration {
        name: "memory_budget.device_bytes".to_string(),
        value: bytes.to_string(),
        expected: "a byte count representable as usize",
    })
}

#[cfg(test)]
mod tests {
    use xlog_core::MemoryBudget;

    use super::{checked_runtime_budget_limit_for_platform, CudaProviderBuilder};

    #[test]
    fn byte_budget_conversion_does_not_saturate() {
        let error = checked_runtime_budget_limit_for_platform(u32::MAX as u64 + 1, u32::MAX as u64)
            .expect_err("out-of-range byte budgets must be rejected");
        assert!(error.to_string().contains("memory_budget.device_bytes"));
    }

    #[test]
    fn runtime_budget_defaults_to_manager_budget_and_can_be_set_independently() {
        let default = CudaProviderBuilder::new(0, MemoryBudget::with_limit(128));
        assert_eq!(default.runtime_budget_bytes(), 128);

        let independent = default.with_runtime_budget_limit(512);
        assert_eq!(independent.runtime_budget_bytes(), 512);
    }
}
