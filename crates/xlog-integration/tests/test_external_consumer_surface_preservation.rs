use std::sync::Arc;

use xlog_core::{MemoryBudget, Result, RuntimeConfig, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::{CudaBuffer, CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_gpu::logic::{LogicEvalResult, LogicProgram};
use xlog_logic::Compiler;
use xlog_runtime::{Executor, RelationDelta};

#[allow(dead_code)]
#[path = "../benches/fixtures/paper_class.rs"]
mod paper_class;

struct DiscardSink;

impl LoggingSink for DiscardSink {
    fn emit(&self, _record: LogRecord) -> std::result::Result<(), SinkError> {
        Ok(())
    }
}

fn test_provider() -> Option<Arc<CudaKernelProvider>> {
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let budget = MemoryBudget::with_limit(1024 * 1024 * 1024);
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
    Some(Arc::new(CudaKernelProvider::new(device, memory).ok()?))
}

fn test_runtime_provider() -> Option<Arc<CudaKernelProvider>> {
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let pool = Arc::new(StreamPool::with_defaults(device.clone()));
    let async_resource: Box<dyn DeviceMemoryResource + Send + Sync> =
        Box::new(AsyncCudaResource::new(device.clone(), 0, pool.clone()));
    let logging: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(LoggingResource::new(
        async_resource,
        Arc::new(DiscardSink) as Arc<dyn LoggingSink>,
    ));
    let budget: Box<dyn DeviceMemoryResource + Send + Sync> =
        Box::new(GlobalDeviceBudget::new(logging, 256 * 1024 * 1024));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        device.clone(),
        0,
        pool,
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        device.clone(),
        MemoryBudget::with_limit(256 * 1024 * 1024),
        runtime,
    ));
    Some(Arc::new(
        CudaKernelProvider::with_runtime(device, memory).ok()?,
    ))
}

fn u32_buffer(provider: &CudaKernelProvider, rows: &[u32]) -> CudaBuffer {
    let schema = Schema::new(vec![("id".to_string(), ScalarType::U32)]);
    let bytes: Vec<u8> = rows.iter().flat_map(|value| value.to_le_bytes()).collect();
    let mut column = provider
        .memory()
        .alloc::<u8>(bytes.len())
        .expect("alloc column");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&bytes, &mut column)
        .expect("upload rows");

    let mut row_count_device = provider.memory().alloc::<u32>(1).expect("alloc rows");
    let row_count = rows.len() as u32;
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&[row_count], &mut row_count_device)
        .expect("upload row count");

    CudaBuffer::from_columns(
        vec![column.into()],
        rows.len() as u64,
        row_count_device,
        schema,
    )
}

fn u32_pair_buffer(provider: &CudaKernelProvider, rows: &[(u32, u32)]) -> CudaBuffer {
    let row_count = rows.len() as u32;
    let bytes_per_col = rows.len() * std::mem::size_of::<u32>();
    let mut col0 = provider
        .memory()
        .alloc::<u8>(bytes_per_col)
        .expect("alloc col0");
    let mut col1 = provider
        .memory()
        .alloc::<u8>(bytes_per_col)
        .expect("alloc col1");
    let mut row_count_device = provider.memory().alloc::<u32>(1).expect("alloc rows");

    if !rows.is_empty() {
        let col0_bytes: Vec<u8> = rows.iter().flat_map(|(x, _)| x.to_le_bytes()).collect();
        let col1_bytes: Vec<u8> = rows.iter().flat_map(|(_, y)| y.to_le_bytes()).collect();
        provider
            .device()
            .inner()
            .htod_sync_copy_into(&col0_bytes, &mut col0)
            .expect("upload col0");
        provider
            .device()
            .inner()
            .htod_sync_copy_into(&col1_bytes, &mut col1)
            .expect("upload col1");
    }
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&[row_count], &mut row_count_device)
        .expect("upload row count");

    let schema = Schema::new(vec![
        ("x".to_string(), ScalarType::U32),
        ("y".to_string(), ScalarType::U32),
    ]);
    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into()],
        row_count as u64,
        row_count_device,
        schema,
        row_count,
    )
}

fn sorted_query_rows(provider: &CudaKernelProvider, result: &LogicEvalResult) -> Vec<u32> {
    let mut rows = provider
        .download_column::<u32>(&result.queries[0].buffer, 0)
        .expect("download query rows");
    rows.sort_unstable();
    rows
}

fn sorted_triples(provider: &CudaKernelProvider, buffer: &CudaBuffer) -> Vec<(u32, u32, u32)> {
    assert_eq!(buffer.arity(), 3);
    let col0 = provider.download_column::<u32>(buffer, 0).expect("col0");
    let col1 = provider.download_column::<u32>(buffer, 1).expect("col1");
    let col2 = provider.download_column::<u32>(buffer, 2).expect("col2");
    assert_eq!(col0.len(), col1.len());
    assert_eq!(col0.len(), col2.len());

    let mut rows: Vec<(u32, u32, u32)> = col0
        .into_iter()
        .zip(col1)
        .zip(col2)
        .map(|((x, y), z)| (x, y, z))
        .collect();
    rows.sort_unstable();
    rows.dedup();
    rows
}

struct ExternalConsumerTriangleExecution {
    rows: Vec<(u32, u32, u32)>,
    wcoj_dispatches: u64,
    dtoh_calls_before_result_download: u64,
    dtoh_bytes_before_result_download: u64,
    d2h_count_before_result_download: u64,
}

fn run_external_consumer_triangle_fixture(
    provider: Arc<CudaKernelProvider>,
    fixture: &paper_class::TriangleFixture,
    config: RuntimeConfig,
) -> Result<ExternalConsumerTriangleExecution> {
    let mut compiler = Compiler::new();
    let plan = compiler.compile(
        r#"
        pred e_xy(u32, u32).
        pred e_yz(u32, u32).
        pred e_xz(u32, u32).
        pred tri(u32, u32, u32).

        tri(X, Y, Z) :- e_xy(X, Y), e_yz(Y, Z), e_xz(X, Z).
        "#,
    )?;
    let mut executor = Executor::new_with_config(provider.clone(), config);
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("e_xy", u32_pair_buffer(&provider, &fixture.e_xy));
    executor.put_relation("e_yz", u32_pair_buffer(&provider, &fixture.e_yz));
    executor.put_relation("e_xz", u32_pair_buffer(&provider, &fixture.e_xz));

    provider.reset_host_transfer_stats();
    provider.reset_d2h_transfer_count();
    executor.execute_plan(&plan)?;
    let transfer_stats = provider.host_transfer_stats();
    let d2h_count = provider.d2h_transfer_count();
    let wcoj_dispatches = executor.wcoj_triangle_dispatch_count();
    let rows = sorted_triples(
        &provider,
        executor.store().get("tri").expect("tri relation"),
    );

    Ok(ExternalConsumerTriangleExecution {
        rows,
        wcoj_dispatches,
        dtoh_calls_before_result_download: transfer_stats.dtoh_calls,
        dtoh_bytes_before_result_download: transfer_stats.dtoh_bytes,
        d2h_count_before_result_download: d2h_count,
    })
}

#[test]
fn logic_session_delta_runtime_preserves_external_consumer_surface_values() -> Result<()> {
    let Some(provider) = test_provider() else {
        eprintln!("Skipping external consumer surface preservation: CUDA unavailable");
        return Ok(());
    };

    let source = r#"
        pred committed_input(u32).
        pred tensor_support(u32).

        tensor_support(X) :- committed_input(X).

        ?- tensor_support(X).
    "#;
    let program = LogicProgram::compile(source)?;
    let mut session_store = program.create_relation_store(provider.clone())?;
    let mut session_cache = None;

    provider.reset_host_transfer_stats();
    provider.reset_d2h_transfer_count();

    let report = program.apply_relation_delta_batch(
        provider.clone(),
        &mut session_store,
        &mut session_cache,
        vec![
            (
                "committed_input".to_string(),
                RelationDelta::new(Some(u32_buffer(&provider, &[1, 2, 3])), None),
            ),
            (
                "committed_input".to_string(),
                RelationDelta::new(None, Some(u32_buffer(&provider, &[2]))),
            ),
            (
                "committed_input".to_string(),
                RelationDelta::new(Some(u32_buffer(&provider, &[4])), None),
            ),
        ],
    )?;

    let transfer_stats = provider.host_transfer_stats();
    assert_eq!(report.input_delta_count, 3);
    assert_eq!(report.changed_relations, 1);
    assert_eq!(report.insert_rows, 3);
    assert_eq!(report.delete_rows, 0);
    assert_eq!(report.coalesced_insert_rows, 3);
    assert_eq!(report.coalesced_delete_rows, 0);
    assert_eq!(report.canceled_rows, 1);
    assert_eq!(transfer_stats.dtoh_calls, 0);
    assert_eq!(transfer_stats.dtoh_bytes, 0);
    assert_eq!(provider.d2h_transfer_count(), 0);

    let session_result = program.evaluate_cached_relation_store(
        provider.clone(),
        session_cache
            .as_ref()
            .expect("cached store after external consumer-style delta batch"),
    )?;
    let session_rows = sorted_query_rows(&provider, &session_result);

    let mut replacement_store = program.create_relation_store(provider.clone())?;
    replacement_store.put("committed_input", u32_buffer(&provider, &[1, 3, 4]));
    let replacement_result =
        program.evaluate_with_relation_store(provider.clone(), &replacement_store, false)?;
    let replacement_rows = sorted_query_rows(&provider, &replacement_result);

    let mut retained_runtime =
        program.create_session_runtime(provider.clone(), &session_store, true)?;
    let (retained_result, _) =
        program.evaluate_with_session_runtime(provider.clone(), &mut retained_runtime)?;
    let retained_rows = sorted_query_rows(&provider, &retained_result);
    let cache_stats = retained_runtime.join_index_cache_stats();

    assert_eq!(session_rows, vec![1, 3, 4]);
    assert_eq!(session_rows, replacement_rows);
    assert_eq!(session_rows, retained_rows);
    assert!(cache_stats.lookups >= cache_stats.hits);
    Ok(())
}

#[test]
fn external_consumer_analog_fixture_runs_through_wcoj_runtime_path() -> Result<()> {
    let Some(provider) = test_runtime_provider() else {
        eprintln!("Skipping external consumer WCOJ surface preservation: CUDA unavailable");
        return Ok(());
    };

    let fixtures = paper_class::paper_class_fixtures(64);
    assert_eq!(
        fixtures.len(),
        paper_class::paper_class_expected_fixture_count()
    );
    let external_consumer = fixtures
        .iter()
        .find(|fixture| fixture.name == "external_consumer_analog")
        .expect("external_consumer_analog fixture is registered");

    assert!(external_consumer.recursive);
    assert!(external_consumer.total_rows() > 0);
    assert!(!external_consumer.e_xy.is_empty());
    assert!(!external_consumer.e_yz.is_empty());
    assert!(!external_consumer.e_xz.is_empty());
    assert!(external_consumer.e_yz.len() >= external_consumer.e_xy.len());

    let binary = run_external_consumer_triangle_fixture(
        provider.clone(),
        external_consumer,
        RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(false)),
    )?;
    assert_eq!(binary.wcoj_dispatches, 0);
    assert!(!binary.rows.is_empty());

    let wcoj = run_external_consumer_triangle_fixture(
        provider,
        external_consumer,
        RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(true)),
    )?;
    assert!(
        wcoj.wcoj_dispatches > 0,
        "forced WCOJ run must dispatch through the production runtime path"
    );
    assert_eq!(wcoj.rows, binary.rows);
    assert_eq!(wcoj.dtoh_calls_before_result_download, 0);
    assert_eq!(wcoj.dtoh_bytes_before_result_download, 0);
    assert_eq!(wcoj.d2h_count_before_result_download, 0);

    Ok(())
}
