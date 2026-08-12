use std::sync::Arc;

use xlog_core::{MemoryBudget, XlogError};
use xlog_cuda::device_runtime::{
    AllocTag, DeviceMemoryResource, DirectCudaResource, GlobalDeviceBudget, StreamPool,
    XlogDeviceRuntime,
};
use xlog_cuda::{CudaDevice, GpuMemoryManager};

fn try_device() -> Option<Arc<CudaDevice>> {
    match CudaDevice::new(0) {
        Ok(device) => Some(Arc::new(device)),
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {error}");
            None
        }
    }
}

fn assert_pressure(
    error: XlogError,
    expected_context: &str,
    expected_required: u64,
    expected_budget: u64,
) {
    match error {
        XlogError::ResourceExhausted {
            context,
            estimated_bytes,
            budget_bytes,
        } => {
            assert_eq!(context, expected_context);
            assert_eq!(estimated_bytes, expected_required);
            assert_eq!(budget_bytes, expected_budget);
        }
        other => panic!("expected ResourceExhausted, got {other:?}"),
    }
}

#[test]
fn allocation_refusal_reports_exact_pressure() {
    let Some(device) = try_device() else {
        return;
    };
    let manager = Arc::new(GpuMemoryManager::new(
        device,
        MemoryBudget::with_limit(4096),
    ));
    let baseline = manager.alloc::<u8>(1024).expect("baseline allocation");

    let error = match manager.alloc::<u8>(4096) {
        Err(error) => error,
        Ok(_) => panic!("cumulative allocation must exceed the configured budget"),
    };

    assert_pressure(
        error,
        "GPU memory pressure: layer=manager_alloc current_bytes=1024 requested_bytes=4096 required_bytes=5120 required_u64_overflow=false budget_bytes=4096 prior_peak_bytes=1024",
        5120,
        4096,
    );
    assert_eq!(manager.allocated_bytes(), 1024);
    assert_eq!(manager.peak_bytes(), 1024);
    drop(baseline);
}

#[test]
fn runtime_refusal_preserves_manager_peak() {
    let Some(device) = try_device() else {
        return;
    };
    let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
    let direct: Box<dyn DeviceMemoryResource + Send + Sync> =
        Box::new(DirectCudaResource::new(Arc::clone(&device), 0));
    let runtime_budget: Box<dyn DeviceMemoryResource + Send + Sync> =
        Box::new(GlobalDeviceBudget::new(direct, 4096));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        pool,
        runtime_budget,
    ));
    let manager = Arc::new(GpuMemoryManager::with_runtime(
        device,
        MemoryBudget::with_limit(8192),
        runtime,
    ));
    let baseline = manager
        .alloc_raw(1024, AllocTag::UNTAGGED)
        .expect("baseline allocation");
    let prior_peak = manager.peak_bytes();

    let error = manager
        .alloc_raw(4096, AllocTag::UNTAGGED)
        .expect_err("runtime budget must reject cumulative pressure");

    assert_pressure(
        error,
        "GPU memory pressure: layer=device_runtime current_bytes=1024 requested_bytes=4096 required_bytes=5120 required_u64_overflow=false budget_bytes=4096 prior_peak_bytes=1024",
        5120,
        4096,
    );
    assert_eq!(manager.allocated_bytes(), 1024);
    assert_eq!(manager.peak_bytes(), prior_peak);
    drop(baseline);
}
