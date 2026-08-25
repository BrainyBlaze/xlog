use std::sync::Arc;

use xlog_core::{MemoryBudget, XlogError};
use xlog_cuda::CudaProviderBuilder;

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
    let provider = match CudaProviderBuilder::new(0, MemoryBudget::with_limit(4096)).build() {
        Ok(provider) => provider,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {error}");
            return;
        }
    };
    let manager = Arc::clone(provider.memory());
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
