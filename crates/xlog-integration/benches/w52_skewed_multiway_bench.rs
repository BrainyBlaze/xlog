//! W5.2 skewed multiway benchmark skeleton.
//!
//! Step 2 establishes the shared provider-direct harness and the
//! Criterion group name. Workload-specific 4-cycle, 5-clique, and
//! pivot-heavy K5 benchmark groups are added in later commits.

#![allow(dead_code)]

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamId, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::CudaBuffer;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager, JoinType};

const BENCH_GROUP: &str = "w52_skewed_multiway";
const DEVICE_BUDGET_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const FOUR_CYCLE_CELLS: &[u32] = &[50, 250, 1000, 2000];

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

fn head_schema_4cycle() -> Schema {
    Schema::new(vec![
        ("w".to_string(), ScalarType::U32),
        ("x".to_string(), ScalarType::U32),
        ("y".to_string(), ScalarType::U32),
        ("z".to_string(), ScalarType::U32),
    ])
}

fn hub_filtered_4cycle(n: u32) -> [Vec<(u32, u32)>; 4] {
    let e1: Vec<(u32, u32)> = (0..n).map(|i| (i, 0)).collect();
    let e2: Vec<(u32, u32)> = (0..n).map(|i| (0, i)).collect();
    let e3: Vec<(u32, u32)> = (0..n).map(|i| (i, i)).collect();
    let e4: Vec<(u32, u32)> = (0..n).map(|i| (i, i)).collect();
    [e1, e2, e3, e4]
}

fn upload_4cycle_fixture(prov: &Provider, rows: &[Vec<(u32, u32)>; 4]) -> [CudaBuffer; 4] {
    [
        upload_2col_u32(&prov.memory, "w", "x", &rows[0]),
        upload_2col_u32(&prov.memory, "x", "y", &rows[1]),
        upload_2col_u32(&prov.memory, "y", "z", &rows[2]),
        upload_2col_u32(&prov.memory, "z", "w", &rows[3]),
    ]
}

fn gpu_wcoj_4cycle_path(prov: &Provider, inputs: &[CudaBuffer; 4]) -> CudaBuffer {
    let e1 = prov
        .provider
        .wcoj_layout_u32_recorded(&inputs[0], prov.launch_stream)
        .expect("layout e1");
    let e2 = prov
        .provider
        .wcoj_layout_u32_recorded(&inputs[1], prov.launch_stream)
        .expect("layout e2");
    let e3 = prov
        .provider
        .wcoj_layout_u32_recorded(&inputs[2], prov.launch_stream)
        .expect("layout e3");
    let e4 = prov
        .provider
        .wcoj_layout_u32_recorded(&inputs[3], prov.launch_stream)
        .expect("layout e4");
    let out = prov
        .provider
        .wcoj_4cycle_u32_recorded(&e1, &e2, &e3, &e4, prov.launch_stream)
        .expect("wcoj 4cycle");
    sync_launch_stream(prov);
    out
}

fn hash_4cycle_chain_path(prov: &Provider, inputs: &[CudaBuffer; 4]) -> CudaBuffer {
    let j12 = prov
        .provider
        .hash_join_v2(&inputs[0], &inputs[1], &[1], &[0], JoinType::Inner)
        .expect("hash e1_e2");
    let j123 = prov
        .provider
        .hash_join_v2(&j12, &inputs[2], &[3], &[0], JoinType::Inner)
        .expect("hash e1_e2_e3");
    let j1234 = prov
        .provider
        .hash_join_v2(&j123, &inputs[3], &[5, 0], &[0, 1], JoinType::Inner)
        .expect("hash e1_e2_e3_e4");
    let out = prov
        .provider
        .wcoj_project_output_columns_recorded(
            &j1234,
            &[0, 1, 3, 5],
            head_schema_4cycle(),
            prov.launch_stream,
        )
        .expect("project hash output to WXYZ");
    sync_launch_stream(prov);
    out
}

fn assert_4cycle_parity(prov: &Provider, inputs: &[CudaBuffer; 4], n: u32) {
    let wcoj = gpu_wcoj_4cycle_path(prov, inputs);
    let hash = hash_4cycle_chain_path(prov, inputs);
    let wcoj_rows = download_u32_rows(&wcoj, &prov.provider, 4);
    let hash_rows = download_u32_rows(&hash, &prov.provider, 4);
    assert_eq!(wcoj_rows, hash_rows, "4-cycle WCOJ/hash parity at N={n}");
    assert_eq!(wcoj_rows.len(), n as usize, "4-cycle final row count");
    eprintln!(
        "  [parity] workload=4cycle N={n:>4} final_rows={:>4} binary_intermediate={}",
        wcoj_rows.len(),
        (n as u64) * (n as u64)
    );
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

    for &n in FOUR_CYCLE_CELLS {
        let rows = hub_filtered_4cycle(n);
        let inputs = upload_4cycle_fixture(&prov, &rows);
        assert_4cycle_parity(&prov, &inputs, n);
        let cell = format!("4cycle_N{n}");

        group.bench_with_input(BenchmarkId::new("gpu_wcoj", &cell), &n, |b, _| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let out = gpu_wcoj_4cycle_path(&prov, &inputs);
                    black_box(out.cached_row_count());
                }
                start.elapsed()
            })
        });

        group.bench_with_input(BenchmarkId::new("hash_chain", &cell), &n, |b, _| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let out = hash_4cycle_chain_path(&prov, &inputs);
                    black_box(out.cached_row_count());
                }
                start.elapsed()
            })
        });
    }

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
