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

fn read_pairs(provider: &CudaKernelProvider, buffer: &xlog_cuda::CudaBuffer) -> Vec<(u32, u32)> {
    let c0 = read_u32_col(provider, buffer, 0);
    let c1 = read_u32_col(provider, buffer, 1);
    let mut rows: Vec<_> = c0.into_iter().zip(c1).collect();
    rows.sort_unstable();
    rows.dedup();
    rows
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
fn v085_list_builtins_execute_through_gpu_relations() -> Result<()> {
    let source = r#"
        pred bag(id: u32, xs: list<u32>).
        bag(1, [10, 20, 10]).
        bag(2, [30]).

        out_member(Id, X) :- bag(Id, L), member(X, L).
        out_memberchk(Id, X) :- bag(Id, L), memberchk(X, L).
        out_length(Id, N) :- bag(Id, L), length(L, N).
        out_nth(Id, X) :- bag(Id, L), nth(1, L, X).
        out_head(Id, H) :- bag(Id, [H | T]).
        out_append_len(N) :- append([10], [20], L), length(L, N).
        out_sort_member(X) :- sort([20, 10, 10], L), member(X, L).
        out_msort_len(N) :- msort([20, 10, 10], L), length(L, N).
        out_set_len(N) :- list_to_set([20, 10, 20], L), length(L, N).
        out_is_list(1) :- is_list([10, 20]).

        ?- out_member(Id, X).
        ?- out_memberchk(Id, X).
        ?- out_length(Id, N).
        ?- out_nth(Id, X).
        ?- out_head(Id, H).
        ?- out_append_len(N).
        ?- out_sort_member(X).
        ?- out_msort_len(N).
        ?- out_set_len(N).
        ?- out_is_list(Flag).
    "#;

    let Some((provider, result)) = run_query(source)? else {
        return Ok(());
    };

    assert_eq!(result.queries.len(), 10);

    assert_eq!(
        read_pairs(&provider, &result.queries[0].buffer),
        vec![(1, 10), (1, 20), (2, 30)]
    );
    assert_eq!(
        read_pairs(&provider, &result.queries[1].buffer),
        vec![(1, 10), (1, 20), (2, 30)]
    );
    assert_eq!(
        read_pairs(&provider, &result.queries[2].buffer),
        vec![(1, 3), (2, 1)]
    );
    assert_eq!(
        read_pairs(&provider, &result.queries[3].buffer),
        vec![(1, 20)]
    );
    assert_eq!(
        read_pairs(&provider, &result.queries[4].buffer),
        vec![(1, 10), (2, 30)]
    );
    assert_eq!(
        read_u32_col(&provider, &result.queries[5].buffer, 0),
        vec![2]
    );
    assert_eq!(
        read_u32_col(&provider, &result.queries[6].buffer, 0),
        vec![10, 20]
    );
    assert_eq!(
        read_u32_col(&provider, &result.queries[7].buffer, 0),
        vec![3]
    );
    assert_eq!(
        read_u32_col(&provider, &result.queries[8].buffer, 0),
        vec![2]
    );
    assert_eq!(
        read_u32_col(&provider, &result.queries[9].buffer, 0),
        vec![1]
    );

    Ok(())
}
