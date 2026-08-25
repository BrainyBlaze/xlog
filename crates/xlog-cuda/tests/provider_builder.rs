use std::sync::Arc;

use xlog_core::MemoryBudget;
use xlog_cuda::device_runtime::{AllocTag, InMemorySink, LoggingSink};
use xlog_cuda::CudaProviderBuilder;

fn provider_or_skip(
    budget_bytes: u64,
    sink: Arc<InMemorySink>,
) -> Option<xlog_cuda::CudaKernelProvider> {
    let sink: Arc<dyn LoggingSink> = sink;
    match CudaProviderBuilder::new(0, MemoryBudget::with_limit(budget_bytes))
        .with_logging_sink(sink)
        .build()
    {
        Ok(provider) => Some(provider),
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but canonical provider construction failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping canonical provider test: {error}");
            None
        }
    }
}

#[test]
fn canonical_builder_owns_one_device_pool_runtime_and_budget() {
    let sink = Arc::new(InMemorySink::new());
    let Some(provider) = provider_or_skip(8 * 1024, Arc::clone(&sink)) else {
        return;
    };
    let memory = provider.memory();
    let runtime = memory
        .runtime()
        .expect("canonical provider must expose its owned runtime");

    assert!(Arc::ptr_eq(provider.device(), memory.device()));
    assert!(Arc::ptr_eq(provider.device(), runtime.device()));
    assert!(runtime.supports_block_use_tracking());

    let stream = runtime
        .stream_pool()
        .acquire()
        .expect("canonical stream pool must provide a non-default stream");
    let block = runtime
        .allocate(4 * 1024, stream, AllocTag("provider-builder-test"))
        .expect("allocation on the runtime-owned stream must succeed");
    runtime
        .record_block_use(&block, stream)
        .expect("the async resource must resolve the runtime-owned stream");

    let error = runtime
        .allocate(
            4 * 1024 + 1,
            stream,
            AllocTag("provider-builder-over-budget"),
        )
        .expect_err("the global byte budget must reject cumulative pressure");
    assert!(error.to_string().contains("budget"), "{error}");

    runtime
        .deallocate(block)
        .expect("deallocation must succeed");
    runtime.reap_pending().expect("pending frees must reap");
    assert!(
        sink.snapshot().len() >= 4,
        "optional logging must observe allocation, rejection, deallocation, and reap"
    );
}

#[cfg(target_pointer_width = "64")]
#[test]
fn canonical_builder_rejects_an_ordinal_that_cannot_fit_the_runtime() {
    let ordinal = u32::MAX as usize + 1;
    let error = match CudaProviderBuilder::new(ordinal, MemoryBudget::with_limit(1)).build() {
        Ok(_) => panic!("out-of-range ordinal unexpectedly built a provider"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("u32"), "{error}");
}
