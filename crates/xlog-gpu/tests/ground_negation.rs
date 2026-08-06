#![allow(clippy::arc_with_non_send_sync)]

use std::collections::HashMap;
use std::sync::Arc;

use xlog_core::{MemoryBudget, Result};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

fn create_test_provider() -> Option<Arc<CudaKernelProvider>> {
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let budget = MemoryBudget::with_limit(1024 * 1024 * 1024);
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
    Some(Arc::new(CudaKernelProvider::new(device, memory).ok()?))
}

fn evaluate_unary_query(source: &str) -> Result<Option<Vec<u32>>> {
    let Some(provider) = create_test_provider() else {
        if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed");
        }
        eprintln!("Skipping: no CUDA device");
        return Ok(None);
    };

    let program = xlog_gpu::logic::LogicProgram::compile(source)?;
    let result = program.evaluate(provider.clone(), HashMap::new())?;
    let mut values = provider.download_column::<u32>(&result.queries[0].buffer, 0)?;
    values.sort_unstable();
    Ok(Some(values))
}

#[test]
fn ground_negation_keeps_input_rows_when_the_ground_atom_is_absent() -> Result<()> {
    let source = r#"
        pred x(u32).
        pred p(u32).
        pred ok(u32).

        x(1).
        x(2).
        p(4).
        ok(X) :- x(X), not p(3).

        ?- ok(X).
    "#;

    let Some(values) = evaluate_unary_query(source)? else {
        return Ok(());
    };
    assert_eq!(values, vec![1, 2]);
    Ok(())
}

#[test]
fn ground_negation_removes_input_rows_when_the_ground_atom_is_present() -> Result<()> {
    let source = r#"
        pred x(u32).
        pred p(u32).
        pred ok(u32).

        x(1).
        x(2).
        p(3).
        ok(X) :- x(X), not p(3).

        ?- ok(X).
    "#;

    let Some(values) = evaluate_unary_query(source)? else {
        return Ok(());
    };
    assert!(values.is_empty());
    Ok(())
}
