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

fn read_u32_col(
    provider: &CudaKernelProvider,
    buffer: &xlog_cuda::CudaBuffer,
    col: usize,
) -> Vec<u32> {
    provider
        .download_column::<u32>(buffer, col)
        .unwrap_or_default()
}

fn run_query(
    source: &str,
) -> Result<Option<(Arc<CudaKernelProvider>, xlog_gpu::logic::LogicEvalResult)>> {
    let Some(provider) = create_test_provider() else {
        eprintln!("Skipping: no CUDA device");
        return Ok(None);
    };
    let program = xlog_gpu::logic::LogicProgram::compile(source)?;
    let result = program.evaluate(provider.clone(), HashMap::new())?;
    Ok(Some((provider, result)))
}

#[test]
fn v085_meta_builtins_execute_through_gpu_relations() -> Result<()> {
    let source = r#"
        pred edge(src: u32, dst: u32).
        pred good(value: u32).
        pred next(input: u32, output: u32).
        pred termbag(t: term).

        edge(1, 2).
        edge(1, 3).
        good(1).
        good(2).
        next(1, 10).
        next(2, 20).
        termbag(point(1, 2)).

        out_functor(A) :- functor(point(1, 2), point, A).
        out_term_functor(A) :- termbag(T), functor(T, point, A).
        out_univ(N) :- point(1, 2) =.. Parts, length(Parts, N).
        out_findall(N) :- findall(X, edge(1, X), L), length(L, N).
        out_maplist(N) :- maplist(next, [1, 2], L), length(L, N), nth(0, L, 10), nth(1, L, 20).
        out_all_good(1) :- maplist(good, [1, 2]).

        ?- out_functor(A).
        ?- out_term_functor(A).
        ?- out_univ(N).
        ?- out_findall(N).
        ?- out_maplist(N).
        ?- out_all_good(Flag).
    "#;

    let Some((provider, result)) = run_query(source)? else {
        return Ok(());
    };

    assert_eq!(result.queries.len(), 6);
    assert_eq!(
        read_u32_col(&provider, &result.queries[0].buffer, 0),
        vec![2]
    );
    assert_eq!(
        read_u32_col(&provider, &result.queries[1].buffer, 0),
        vec![2]
    );
    assert_eq!(
        read_u32_col(&provider, &result.queries[2].buffer, 0),
        vec![3]
    );
    assert_eq!(
        read_u32_col(&provider, &result.queries[3].buffer, 0),
        vec![2]
    );
    assert_eq!(
        read_u32_col(&provider, &result.queries[4].buffer, 0),
        vec![2]
    );
    assert_eq!(
        read_u32_col(&provider, &result.queries[5].buffer, 0),
        vec![1]
    );

    Ok(())
}
