#![allow(dead_code)]

//! External consumer analog fixture: registration metadata AND real execution.
//!
//! The original test only asserted that `bundle_path_status` (a hardcoded
//! `&'static str` on the fixture) contained "PASS" markers — a metadata
//! string check that certified nothing about behavior. It now also executes
//! the fixture's triangle workload through the real Compiler + Executor
//! pipeline and checks the produced row set against a host brute-force
//! oracle, so the fixture's data path is value-verified.

#[path = "../benches/fixtures/paper_class.rs"]
mod paper_class;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use cudarc::driver::sys;
use xlog_core::{MemoryBudget, RuntimeConfig, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::CudaBuffer;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_logic::Compiler;
use xlog_runtime::Executor;

struct DiscardSink;
impl LoggingSink for DiscardSink {
    fn emit(&self, _record: LogRecord) -> Result<(), SinkError> {
        Ok(())
    }
}

struct RuntimeFixture {
    device: Arc<CudaDevice>,
    runtime: Arc<XlogDeviceRuntime>,
    memory: Arc<GpuMemoryManager>,
    provider: Arc<CudaKernelProvider>,
    pool: Arc<StreamPool>,
}

fn make_runtime_fixture() -> Option<RuntimeFixture> {
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
    let async_resource: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(
        AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool)),
    );
    let logging: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(LoggingResource::new(
        async_resource,
        Arc::new(DiscardSink) as Arc<dyn LoggingSink>,
    ));
    let budget: Box<dyn DeviceMemoryResource + Send + Sync> =
        Box::new(GlobalDeviceBudget::new(logging, 256 * 1024 * 1024));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(256 * 1024 * 1024),
        Arc::clone(&runtime),
    ));
    let provider =
        Arc::new(CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory)).ok()?);
    Some(RuntimeFixture {
        device,
        runtime,
        memory,
        provider,
        pool,
    })
}

fn upload_binary_u32(memory: &Arc<GpuMemoryManager>, rows: &[(u32, u32)]) -> CudaBuffer {
    let n = rows.len() as u32;
    let col0_host: Vec<u32> = rows.iter().map(|(a, _)| *a).collect();
    let col1_host: Vec<u32> = rows.iter().map(|(_, b)| *b).collect();
    let bytes_per_col = (n as usize) * std::mem::size_of::<u32>();
    let mut col0 = memory.alloc::<u8>(bytes_per_col).expect("alloc col0");
    let mut col1 = memory.alloc::<u8>(bytes_per_col).expect("alloc col1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let device = memory.device().inner();
    if !col0_host.is_empty() {
        let col0_bytes: Vec<u8> = col0_host.iter().flat_map(|v| v.to_le_bytes()).collect();
        let col1_bytes: Vec<u8> = col1_host.iter().flat_map(|v| v.to_le_bytes()).collect();
        device
            .htod_sync_copy_into(&col0_bytes, &mut col0)
            .expect("htod col0");
        device
            .htod_sync_copy_into(&col1_bytes, &mut col1)
            .expect("htod col1");
    }
    device
        .htod_sync_copy_into(&[n], &mut d_num_rows)
        .expect("htod d_num_rows");
    let schema = Schema::new(vec![
        ("col0".to_string(), ScalarType::U32),
        ("col1".to_string(), ScalarType::U32),
    ]);
    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into()],
        n as u64,
        d_num_rows,
        schema,
        n,
    )
}

fn download_triples(buf: &CudaBuffer) -> Vec<(u32, u32, u32)> {
    let n = match buf.cached_row_count() {
        Some(c) => c as usize,
        None => {
            let mut count_host = [0u32; 1];
            unsafe {
                let res = sys::cuMemcpyDtoH_v2(
                    count_host.as_mut_ptr() as *mut _,
                    *buf.num_rows_device().device_ptr(),
                    std::mem::size_of::<u32>(),
                );
                assert_eq!(res, sys::cudaError_enum::CUDA_SUCCESS);
            }
            count_host[0] as usize
        }
    };
    if n == 0 {
        return Vec::new();
    }
    assert_eq!(buf.arity(), 3, "expected 3-column triangle output");
    let mut cols: Vec<Vec<u8>> = (0..3).map(|_| vec![0u8; n * 4]).collect();
    for (idx, col) in cols.iter_mut().enumerate() {
        unsafe {
            let res = sys::cuMemcpyDtoH_v2(
                col.as_mut_ptr() as *mut _,
                *buf.column(idx).unwrap().device_ptr(),
                col.len(),
            );
            assert_eq!(res, sys::cudaError_enum::CUDA_SUCCESS);
        }
    }
    let mut out: Vec<(u32, u32, u32)> = (0..n)
        .map(|i| {
            (
                u32::from_le_bytes(cols[0][i * 4..i * 4 + 4].try_into().unwrap()),
                u32::from_le_bytes(cols[1][i * 4..i * 4 + 4].try_into().unwrap()),
                u32::from_le_bytes(cols[2][i * 4..i * 4 + 4].try_into().unwrap()),
            )
        })
        .collect();
    out.sort();
    out.dedup();
    out
}

/// Host brute-force triangle oracle over the fixture's three relations.
fn brute_force_triangles(
    e_xy: &[(u32, u32)],
    e_yz: &[(u32, u32)],
    e_xz: &[(u32, u32)],
) -> Vec<(u32, u32, u32)> {
    let xz_set: BTreeSet<(u32, u32)> = e_xz.iter().copied().collect();
    let mut yz_by_y: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for (y, z) in e_yz {
        yz_by_y.entry(*y).or_default().push(*z);
    }
    let mut out = BTreeSet::new();
    for (x, y) in e_xy {
        if let Some(zs) = yz_by_y.get(y) {
            for z in zs {
                if xz_set.contains(&(*x, *z)) {
                    out.insert((*x, *y, *z));
                }
            }
        }
    }
    out.into_iter().collect()
}

#[test]
fn external_consumer_analog_fixture_is_registered_with_paper_class_harness() {
    let fixtures = paper_class::paper_class_fixtures(128);
    assert_eq!(
        fixtures.len(),
        4,
        "fixture registry extends the three paper-class fixtures with one external consumer analog"
    );

    let external_consumer = fixtures
        .iter()
        .find(|fixture| fixture.name == "external_consumer_analog")
        .expect("external_consumer_analog fixture is registered");

    assert!(
        external_consumer.recursive,
        "external consumer analog must exercise recursive set maintenance"
    );
    assert!(
        !external_consumer.e_xy.is_empty()
            && !external_consumer.e_yz.is_empty()
            && !external_consumer.e_xz.is_empty(),
        "external consumer analog must populate every relation in the triangle harness"
    );
    assert!(
        external_consumer.e_yz.len() >= external_consumer.e_xy.len(),
        "middle-key fanout should model external consumer two-hop support expansion"
    );
    // Registration metadata only — `bundle_path_status` is a hardcoded
    // fixture label, NOT execution evidence. Behavioral coverage is locked
    // by `external_consumer_analog_fixture_executes_with_oracle_parity` below.
    assert!(
        external_consumer
            .bundle_path_status
            .contains("cuda_graph=PASS")
            && external_consumer.bundle_path_status.contains("invoked=7/7"),
        "external consumer analog registration metadata must keep its bundle-path label"
    );
}

#[test]
fn external_consumer_analog_fixture_executes_with_oracle_parity() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let fixtures = paper_class::paper_class_fixtures(128);
    let external_consumer = fixtures
        .iter()
        .find(|fixture| fixture.name == "external_consumer_analog")
        .expect("external_consumer_analog fixture is registered");

    let expected = brute_force_triangles(
        &external_consumer.e_xy,
        &external_consumer.e_yz,
        &external_consumer.e_xz,
    );
    assert!(
        !expected.is_empty(),
        "external consumer analog must contain at least one triangle or the workload is vacuous"
    );

    let source = "tri(X, Y, Z) :- e_xy(X, Y), e_yz(Y, Z), e_xz(X, Z).";
    let mut compiler = Compiler::new();
    let plan = compiler.compile(source).expect("compile triangle rule");
    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation(
        "e_xy",
        upload_binary_u32(&fix.memory, &external_consumer.e_xy),
    );
    executor.put_relation(
        "e_yz",
        upload_binary_u32(&fix.memory, &external_consumer.e_yz),
    );
    executor.put_relation(
        "e_xz",
        upload_binary_u32(&fix.memory, &external_consumer.e_xz),
    );
    executor.execute_plan(&plan).expect("execute triangle plan");

    let tri = executor
        .store()
        .get("tri")
        .expect("tri relation materialized");
    let got = download_triples(tri);
    assert_eq!(
        got, expected,
        "GPU triangle row set over the external consumer analog must match the host brute-force oracle"
    );
    assert_eq!(
        executor.wcoj_error_decline_count(),
        0,
        "healthy execution must not consume WCOJ error declines"
    );
}
