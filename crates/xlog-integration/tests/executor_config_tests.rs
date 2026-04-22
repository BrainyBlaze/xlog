#![allow(clippy::arc_with_non_send_sync)]
use std::sync::Arc;

use xlog_core::{symbol, MemoryBudget, RuntimeConfig, ScalarType, Schema};
use xlog_cuda::{CudaBuffer, CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_ir::{CompareOp, ConstValue, Expr};
use xlog_logic::Compiler;
use xlog_runtime::Executor;

fn has_cuda_device() -> bool {
    CudaDevice::new(0).is_ok()
}

fn create_executor_with_config(
    config: RuntimeConfig,
) -> Option<(Executor, Arc<CudaKernelProvider>)> {
    if !has_cuda_device() {
        return None;
    }

    let device = Arc::new(CudaDevice::new(0).ok()?);
    let budget = MemoryBudget::with_limit(1024 * 1024 * 1024);
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
    let provider = Arc::new(CudaKernelProvider::new(device, memory).ok()?);
    let executor = Executor::new_with_config(provider.clone(), config);

    Some((executor, provider))
}

fn device_row_count(
    provider: &CudaKernelProvider,
    rows: u64,
) -> xlog_cuda::memory::TrackedCudaSlice<u32> {
    let rows_u32 = u32::try_from(rows).expect("row count fits u32");
    let mut d_num_rows = provider.memory().alloc::<u32>(1).expect("alloc");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&[rows_u32], &mut d_num_rows)
        .expect("htod");
    d_num_rows
}

fn create_edge_buffer(provider: &CudaKernelProvider, edges: &[(u32, u32)]) -> CudaBuffer {
    let schema = Schema::new(vec![
        ("c0".to_string(), ScalarType::U32),
        ("c1".to_string(), ScalarType::U32),
    ]);

    if edges.is_empty() {
        let col0 = provider.memory().alloc::<u8>(0).expect("alloc");
        let col1 = provider.memory().alloc::<u8>(0).expect("alloc");
        let d_num_rows = device_row_count(provider, 0);
        return CudaBuffer::from_columns(vec![col0.into(), col1.into()], 0, d_num_rows, schema);
    }

    let col0_bytes: Vec<u8> = edges
        .iter()
        .flat_map(|(from, _)| from.to_le_bytes())
        .collect();
    let col1_bytes: Vec<u8> = edges.iter().flat_map(|(_, to)| to.to_le_bytes()).collect();

    let mut col0 = provider
        .memory()
        .alloc::<u8>(col0_bytes.len())
        .expect("alloc");
    let mut col1 = provider
        .memory()
        .alloc::<u8>(col1_bytes.len())
        .expect("alloc");

    provider
        .device()
        .inner()
        .htod_sync_copy_into(&col0_bytes, &mut col0)
        .expect("htod");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&col1_bytes, &mut col1)
        .expect("htod");

    let rows = edges.len() as u64;
    let d_num_rows = device_row_count(provider, rows);
    CudaBuffer::from_columns(vec![col0.into(), col1.into()], rows, d_num_rows, schema)
}

fn setup_executor_with_facts(
    executor: &mut Executor,
    compiler: &Compiler,
    facts: Vec<(&str, CudaBuffer)>,
) {
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }

    for (name, buffer) in facts {
        executor.store_mut().put(name, buffer);
    }
}

#[test]
fn test_executor_respects_max_iterations() {
    let config = {
        let mut config = RuntimeConfig::default();
        config.max_iterations = 1;
        config
    };

    let (mut executor, provider) = match create_executor_with_config(config) {
        Some(e) => e,
        None => {
            eprintln!("Skipping test: no CUDA device available");
            return;
        }
    };

    let source = r#"
        edge(1, 2).
        edge(2, 3).
        edge(3, 4).
        reach(X, Y) :- edge(X, Y).
        reach(X, Z) :- reach(X, Y), edge(Y, Z).
    "#;

    let mut compiler = Compiler::new();
    let plan = compiler.compile(source).expect("Compilation failed");
    assert!(plan.has_recursion(), "Expected recursive plan");

    let edge_buffer = create_edge_buffer(&provider, &[(1, 2), (2, 3), (3, 4)]);
    setup_executor_with_facts(&mut executor, &compiler, vec![("edge", edge_buffer)]);

    let err = match executor.execute_plan(&plan) {
        Ok(_) => panic!("expected iteration cap error"),
        Err(err) => err,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("iteration limit (1)"),
        "unexpected error message: {msg}"
    );
}

#[test]
fn test_executor_filter_with_column_column_compare_and_symbol() {
    let (executor, provider) = match create_executor_with_config(RuntimeConfig::default()) {
        Some(e) => e,
        None => {
            eprintln!("Skipping test: no CUDA device available");
            return;
        }
    };

    let symbol_hash = symbol::intern("sym");

    let schema = Schema::new(vec![
        ("a".to_string(), ScalarType::U32),
        ("b".to_string(), ScalarType::U32),
        ("s".to_string(), ScalarType::Symbol),
    ]);

    let buf = provider
        .create_buffer_from_u32_columns(
            &[&[1, 2, 3], &[1, 9, 3], &[symbol_hash, 7, symbol_hash]],
            schema,
        )
        .unwrap();

    let predicate = Expr::And(vec![
        Expr::Compare {
            left: Box::new(Expr::Column(0)),
            op: CompareOp::Eq,
            right: Box::new(Expr::Column(1)),
        },
        Expr::Compare {
            left: Box::new(Expr::Column(2)),
            op: CompareOp::Eq,
            right: Box::new(Expr::Const(ConstValue::Symbol("sym".to_string()))),
        },
    ]);

    let filtered = executor.execute_filter(&buf, &predicate).unwrap();
    let vals = provider.download_column::<u32>(&filtered, 0).unwrap();
    assert_eq!(vals, vec![1, 3]);
}
