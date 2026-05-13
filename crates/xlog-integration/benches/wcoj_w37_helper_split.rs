use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamId, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::CudaBuffer;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager, JoinType};

const DEVICE_BUDGET_BYTES: u64 = 8 * 1024 * 1024 * 1024;

#[derive(Clone)]
struct DeepSkewFixture {
    label: &'static str,
    outer_rows: u32,
    inner_fanout: u32,
    f_buckets: u32,
    r_ab: Vec<(u32, u32)>,
    r_bc: Vec<(u32, u32)>,
    r_cd: Vec<(u32, u32)>,
    r_de: Vec<(u32, u32)>,
    r_ef: Vec<(u32, u32)>,
    r_af: Vec<(u32, u32)>,
}

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

struct UploadedFixture {
    r_ab: CudaBuffer,
    r_bc: CudaBuffer,
    r_cd: CudaBuffer,
    r_de: CudaBuffer,
    r_ef: CudaBuffer,
    r_af: CudaBuffer,
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
        .expect("resolve launch stream")
        .synchronize()
        .expect("sync launch stream");
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

fn upload_fixture(prov: &Provider, fixture: &DeepSkewFixture) -> UploadedFixture {
    UploadedFixture {
        r_ab: upload_2col_u32(&prov.memory, "a", "b", &fixture.r_ab),
        r_bc: upload_2col_u32(&prov.memory, "b", "c", &fixture.r_bc),
        r_cd: upload_2col_u32(&prov.memory, "c", "d", &fixture.r_cd),
        r_de: upload_2col_u32(&prov.memory, "d", "e", &fixture.r_de),
        r_ef: upload_2col_u32(&prov.memory, "e", "f", &fixture.r_ef),
        r_af: upload_2col_u32(&prov.memory, "a", "f", &fixture.r_af),
    }
}

fn head_schema() -> Schema {
    Schema::new(vec![
        ("a".to_string(), ScalarType::U32),
        ("b".to_string(), ScalarType::U32),
        ("c".to_string(), ScalarType::U32),
        ("d".to_string(), ScalarType::U32),
        ("f".to_string(), ScalarType::U32),
    ])
}

fn helper_schema() -> Schema {
    Schema::new(vec![
        ("d".to_string(), ScalarType::U32),
        ("f".to_string(), ScalarType::U32),
    ])
}

fn make_fixture(
    label: &'static str,
    outer_rows: u32,
    inner_fanout: u32,
    f_buckets: u32,
) -> DeepSkewFixture {
    let r_ab: Vec<(u32, u32)> = (0..outer_rows).map(|a| (a, a)).collect();
    let r_bc: Vec<(u32, u32)> = (0..outer_rows).map(|b| (b, b)).collect();
    let r_cd: Vec<(u32, u32)> = (0..outer_rows).map(|c| (c, 0)).collect();
    let r_de: Vec<(u32, u32)> = (0..inner_fanout).map(|e| (0, e)).collect();
    let r_ef: Vec<(u32, u32)> = (0..inner_fanout).map(|e| (e, e % f_buckets)).collect();
    let r_af: Vec<(u32, u32)> = (0..outer_rows).map(|a| (a, a % f_buckets)).collect();
    DeepSkewFixture {
        label,
        outer_rows,
        inner_fanout,
        f_buckets,
        r_ab,
        r_bc,
        r_cd,
        r_de,
        r_ef,
        r_af,
    }
}

fn download_rows(buf: &CudaBuffer, prov: &CudaKernelProvider, arity: usize) -> BTreeSet<Vec<u32>> {
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

fn dedup_recorded(prov: &Provider, input: &CudaBuffer) -> CudaBuffer {
    let out = prov
        .provider
        .dedup_full_row_recorded(input, prov.launch_stream)
        .expect("dedup full row");
    sync_launch_stream(prov);
    out
}

fn build_helper(prov: &Provider, input: &UploadedFixture) -> CudaBuffer {
    let de_ef = prov
        .provider
        .hash_join_v2(&input.r_de, &input.r_ef, &[1], &[0], JoinType::Inner)
        .expect("join de ef");
    let projected = prov
        .provider
        .wcoj_project_output_columns_recorded(&de_ef, &[0, 3], helper_schema(), prov.launch_stream)
        .expect("project helper d f");
    sync_launch_stream(prov);
    dedup_recorded(prov, &projected)
}

fn run_unsplit(prov: &Provider, input: &UploadedFixture) -> CudaBuffer {
    let ab_bc = prov
        .provider
        .hash_join_v2(&input.r_ab, &input.r_bc, &[1], &[0], JoinType::Inner)
        .expect("join ab bc");
    let with_cd = prov
        .provider
        .hash_join_v2(&ab_bc, &input.r_cd, &[3], &[0], JoinType::Inner)
        .expect("join cd");
    let with_de = prov
        .provider
        .hash_join_v2(&with_cd, &input.r_de, &[5], &[0], JoinType::Inner)
        .expect("join de");
    let with_ef = prov
        .provider
        .hash_join_v2(&with_de, &input.r_ef, &[7], &[0], JoinType::Inner)
        .expect("join ef");
    let with_af = prov
        .provider
        .hash_join_v2(&with_ef, &input.r_af, &[0, 9], &[0, 1], JoinType::Inner)
        .expect("join af");
    let projected = prov
        .provider
        .wcoj_project_output_columns_recorded(
            &with_af,
            &[0, 1, 3, 5, 9],
            head_schema(),
            prov.launch_stream,
        )
        .expect("project unsplit");
    sync_launch_stream(prov);
    dedup_recorded(prov, &projected)
}

fn run_hand_split(prov: &Provider, input: &UploadedFixture) -> CudaBuffer {
    let helper = build_helper(prov, input);
    let ab_bc = prov
        .provider
        .hash_join_v2(&input.r_ab, &input.r_bc, &[1], &[0], JoinType::Inner)
        .expect("join ab bc");
    let with_cd = prov
        .provider
        .hash_join_v2(&ab_bc, &input.r_cd, &[3], &[0], JoinType::Inner)
        .expect("join cd");
    let with_helper = prov
        .provider
        .hash_join_v2(&with_cd, &helper, &[5], &[0], JoinType::Inner)
        .expect("join helper");
    let with_af = prov
        .provider
        .hash_join_v2(&with_helper, &input.r_af, &[0, 7], &[0, 1], JoinType::Inner)
        .expect("join af");
    let projected = prov
        .provider
        .wcoj_project_output_columns_recorded(
            &with_af,
            &[0, 1, 3, 5, 7],
            head_schema(),
            prov.launch_stream,
        )
        .expect("project hand split");
    sync_launch_stream(prov);
    dedup_recorded(prov, &projected)
}

fn assert_parity(prov: &Provider, fixture: &DeepSkewFixture, input: &UploadedFixture) {
    let unsplit = run_unsplit(prov, input);
    let split = run_hand_split(prov, input);
    let unsplit_rows = download_rows(&unsplit, &prov.provider, 5);
    let split_rows = download_rows(&split, &prov.provider, 5);
    assert_eq!(unsplit_rows, split_rows, "row equality {}", fixture.label);
    assert_eq!(unsplit_rows.len(), fixture.outer_rows as usize);
    eprintln!(
        "W37_ROW_EQUALITY {} PASS rows={} inner_intermediate_rows={} helper_rows={}",
        fixture.label,
        unsplit_rows.len(),
        (fixture.outer_rows as u64) * (fixture.inner_fanout as u64),
        fixture.f_buckets
    );
}

fn measure_unsplit(prov: &Provider, input: &UploadedFixture, iters: u64) -> Duration {
    let mut measured = Duration::ZERO;
    for _ in 0..iters {
        let start = Instant::now();
        let out = run_unsplit(prov, input);
        measured += start.elapsed();
        black_box(out.cached_row_count());
        let split = run_hand_split(prov, input);
        black_box(split.cached_row_count());
    }
    measured
}

fn measure_hand_split(prov: &Provider, input: &UploadedFixture, iters: u64) -> Duration {
    let mut measured = Duration::ZERO;
    for _ in 0..iters {
        let unsplit = run_unsplit(prov, input);
        black_box(unsplit.cached_row_count());
        let start = Instant::now();
        let out = run_hand_split(prov, input);
        measured += start.elapsed();
        black_box(out.cached_row_count());
    }
    measured
}

fn bench_fixture(c: &mut Criterion, prov: &Provider, fixture: DeepSkewFixture) {
    let uploaded = upload_fixture(prov, &fixture);
    assert_parity(prov, &fixture, &uploaded);
    eprintln!(
        "W37_FIXTURE {} outer_rows={} inner_fanout={} f_buckets={} inner_intermediate_rows={}",
        fixture.label,
        fixture.outer_rows,
        fixture.inner_fanout,
        fixture.f_buckets,
        (fixture.outer_rows as u64) * (fixture.inner_fanout as u64)
    );

    let mut group = c.benchmark_group("wcoj_w37_helper_split");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(1));
    group.throughput(Throughput::Elements(
        (fixture.outer_rows as u64) * (fixture.inner_fanout as u64),
    ));
    group.bench_with_input(
        BenchmarkId::new("unsplit_no_rewrite", fixture.label),
        &(),
        |b, _| b.iter_custom(|iters| measure_unsplit(prov, &uploaded, iters)),
    );
    group.bench_with_input(
        BenchmarkId::new("hand_split_helper", fixture.label),
        &(),
        |b, _| b.iter_custom(|iters| measure_hand_split(prov, &uploaded, iters)),
    );
    group.finish();
}

fn bench_w37_helper_split(c: &mut Criterion) {
    let Some(prov) = make_provider() else {
        eprintln!("Skipping wcoj_w37_helper_split: CUDA runtime unavailable");
        return;
    };
    bench_fixture(
        c,
        &prov,
        make_fixture("callgraph-inner-skew", 4096, 1024, 4),
    );
    bench_fixture(
        c,
        &prov,
        make_fixture("heapalloc-inner-skew", 8192, 4096, 1),
    );
}

criterion_group! {
    name = wcoj_w37_helper_split;
    config = Criterion::default();
    targets = bench_w37_helper_split
}
criterion_main!(wcoj_w37_helper_split);
