#![allow(clippy::arc_with_non_send_sync)]
use std::sync::Arc;

use xlog_core::{MemoryBudget, Result, ScalarType, Schema};
use xlog_cuda::{CudaBuffer, CudaDevice, CudaKernelProvider, GpuMemoryManager};

fn create_test_provider() -> Option<Arc<CudaKernelProvider>> {
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let budget = MemoryBudget::with_limit(1024 * 1024 * 1024);
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
    Some(Arc::new(CudaKernelProvider::new(device, memory).ok()?))
}

fn create_edge_buffer(provider: &CudaKernelProvider, edges: &[(u32, u32)]) -> Result<CudaBuffer> {
    let schema = Schema::new(vec![
        ("c0".to_string(), ScalarType::U32),
        ("c1".to_string(), ScalarType::U32),
    ]);

    if edges.is_empty() {
        return provider.create_empty_buffer(schema);
    }

    let col0: Vec<u8> = edges.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
    let col1: Vec<u8> = edges.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
    provider.create_buffer_from_slices(&[&col0, &col1], schema)
}

fn read_pairs(provider: &CudaKernelProvider, buffer: &CudaBuffer) -> Vec<(u32, u32)> {
    let c0 = provider.download_column_u32(buffer, 0).unwrap_or_default();
    let c1 = provider.download_column_u32(buffer, 1).unwrap_or_default();
    c0.into_iter().zip(c1).collect()
}

#[test]
fn test_logic_program_runs_with_gpu_inputs() -> Result<()> {
    let Some(provider) = create_test_provider() else {
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    let source = r#"
        pred edge(u32, u32).
        pred reach(u32, u32).

        reach(X, Y) :- edge(X, Y).
        reach(X, Z) :- reach(X, Y), edge(Y, Z).

        ?- reach(X, Y).
    "#;

    let program = xlog_gpu::logic::LogicProgram::compile(source)?;

    let edge_buf = create_edge_buffer(&provider, &[(1, 2), (2, 3)])?;
    let mut inputs = std::collections::HashMap::new();
    inputs.insert("edge".to_string(), edge_buf);

    let result = program.evaluate(provider.clone(), inputs)?;
    let query0 = &result.queries[0];
    let pairs = read_pairs(&provider, &query0.buffer);

    let mut got = pairs;
    got.sort_unstable();
    got.dedup();

    let mut expected = vec![(1, 2), (1, 3), (2, 3)];
    expected.sort_unstable();

    assert_eq!(got, expected);
    Ok(())
}
