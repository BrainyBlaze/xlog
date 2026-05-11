//! W5.2 skewed multiway benchmark skeleton.
//!
//! Step 2 establishes the shared provider-direct harness and the
//! Criterion group name. Workload-specific 4-cycle, 5-clique, and
//! pivot-heavy K5 benchmark groups are added in later commits.

#![allow(dead_code)]

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamId, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::CudaBuffer;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

const BENCH_GROUP: &str = "w52_skewed_multiway";
const DEVICE_BUDGET_BYTES: u64 = 8 * 1024 * 1024 * 1024;

struct DiscardSink;
impl LoggingSink for DiscardSink {
    fn emit(&self, _record: LogRecord) -> Result<(), SinkError> {
        Ok(())
    }
}

struct Provider {
    _device: Arc<CudaDevice>,
    _runtime: Arc<XlogDeviceRuntime>,
    memory: Arc<GpuMemoryManager>,
    provider: Arc<CudaKernelProvider>,
    pool: Arc<StreamPool>,
    launch_stream: StreamId,
}

fn make_provider() -> Option<Provider> {
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let pool = Arc::new(StreamPool::new(Arc::clone(&device), 1024));
    let async_resource: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(
        AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool)),
    );
    let logging: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(LoggingResource::new(
        async_resource,
        Arc::new(DiscardSink) as Arc<dyn LoggingSink>,
    ));
    let budget: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(GlobalDeviceBudget::new(
        logging,
        DEVICE_BUDGET_BYTES as usize,
    ));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(DEVICE_BUDGET_BYTES),
        Arc::clone(&runtime),
    ));
    let provider =
        Arc::new(CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory)).ok()?);
    let launch_stream = pool.acquire().ok()?;
    Some(Provider {
        _device: device,
        _runtime: runtime,
        memory,
        provider,
        pool,
        launch_stream,
    })
}

fn sync_launch_stream(prov: &Provider) {
    prov.pool
        .resolve(prov.launch_stream)
        .expect("resolve WCOJ launch stream")
        .synchronize()
        .expect("sync WCOJ launch stream");
}

fn upload_2col_u32(
    memory: &Arc<GpuMemoryManager>,
    col0_name: &str,
    col1_name: &str,
    rows: &[(u32, u32)],
) -> CudaBuffer {
    let n = rows.len() as u32;
    let bytes_per_col = (n as usize) * std::mem::size_of::<u32>();
    let mut col0 = memory.alloc::<u8>(bytes_per_col).expect("alloc col0");
    let mut col1 = memory.alloc::<u8>(bytes_per_col).expect("alloc col1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let dev = memory.device().inner();
    if n > 0 {
        let b0: Vec<u8> = rows.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
        let b1: Vec<u8> = rows.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
        dev.htod_sync_copy_into(&b0, &mut col0).expect("htod c0");
        dev.htod_sync_copy_into(&b1, &mut col1).expect("htod c1");
    }
    dev.htod_sync_copy_into(&[n], &mut d_num_rows)
        .expect("htod n");
    let schema = Schema::new(vec![
        (col0_name.to_string(), ScalarType::U32),
        (col1_name.to_string(), ScalarType::U32),
    ]);
    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into()],
        n as u64,
        d_num_rows,
        schema,
        n,
    )
}

fn download_u32_rows(
    buf: &CudaBuffer,
    prov: &CudaKernelProvider,
    arity: usize,
) -> BTreeSet<Vec<u32>> {
    assert_eq!(buf.arity(), arity, "download arity mismatch");
    let n = prov.device_row_count(buf).expect("row count");
    let cols: Vec<Vec<u32>> = (0..arity)
        .map(|i| {
            prov.download_column_untracked::<u32>(buf, i)
                .expect("download col")
        })
        .collect();
    (0..n)
        .map(|row| (0..arity).map(|col| cols[col][row]).collect())
        .collect()
}

fn bench_w52_skewed_multiway(c: &mut Criterion) {
    let Some(prov) = make_provider() else {
        eprintln!("Skipping W5.2 skewed multiway bench: CUDA runtime unavailable");
        return;
    };

    let rows = [(1_u32, 1_u32), (2, 2), (3, 3)];
    let uploaded = upload_2col_u32(&prov.memory, "left", "right", &rows);
    let observed = download_u32_rows(&uploaded, &prov.provider, 2);
    let expected: BTreeSet<Vec<u32>> = rows.iter().map(|(a, b)| vec![*a, *b]).collect();
    assert_eq!(observed, expected, "skeleton upload/download parity");

    let mut group = c.benchmark_group(BENCH_GROUP);
    group.bench_function("skeleton/provider_ready", |b| {
        b.iter(|| black_box(uploaded.cached_row_count()))
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(50)
        .measurement_time(Duration::from_secs(8))
        .warm_up_time(Duration::from_secs(1));
    targets = bench_w52_skewed_multiway
}
criterion_main!(benches);
