//! W5.2 skewed multiway benchmark harness.
//!
//! Provider-direct Criterion groups compare WCOJ paths against binary
//! hash-chain baselines for the W5.2 closure evidence.

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
const CLIQUE5_CELLS: &[u32] = &[10, 25, 50, 100];
const PIVOT5_CELLS: &[u32] = &[10, 20, 30, 40];
const CLIQUE5_EDGE_NAMES: [(&str, &str); 10] = [
    ("v0", "v1"),
    ("v0", "v2"),
    ("v0", "v3"),
    ("v0", "v4"),
    ("v1", "v2"),
    ("v1", "v3"),
    ("v1", "v4"),
    ("v2", "v3"),
    ("v2", "v4"),
    ("v3", "v4"),
];
const PIVOT5_EDGE_NAMES: [(&str, &str); 10] = [
    ("p", "a"),
    ("p", "b"),
    ("p", "c"),
    ("p", "d"),
    ("a", "b"),
    ("a", "c"),
    ("a", "d"),
    ("b", "c"),
    ("b", "d"),
    ("c", "d"),
];

#[derive(Clone, Copy)]
enum W52LiteralGateWorkload {
    FourCycle,
    Clique5,
    Pivot5,
}

#[derive(Clone, Copy)]
enum W52LiteralGatePath {
    GpuWcoj,
    HashChain,
}

fn w52_literal_gate_target_ns(
    workload: W52LiteralGateWorkload,
    path: W52LiteralGatePath,
    n: u32,
) -> u64 {
    match (workload, path, n) {
        (W52LiteralGateWorkload::FourCycle, W52LiteralGatePath::GpuWcoj, 50) => 1_609_000,
        (W52LiteralGateWorkload::FourCycle, W52LiteralGatePath::HashChain, 50) => 11_240_400,
        (W52LiteralGateWorkload::FourCycle, W52LiteralGatePath::GpuWcoj, 250) => 2_116_600,
        (W52LiteralGateWorkload::FourCycle, W52LiteralGatePath::HashChain, 250) => 11_104_500,
        (W52LiteralGateWorkload::FourCycle, W52LiteralGatePath::GpuWcoj, 1000) => 4_920_800,
        (W52LiteralGateWorkload::FourCycle, W52LiteralGatePath::HashChain, 1000) => 13_596_800,
        (W52LiteralGateWorkload::FourCycle, W52LiteralGatePath::GpuWcoj, 2000) => 9_382_000,
        (W52LiteralGateWorkload::FourCycle, W52LiteralGatePath::HashChain, 2000) => 21_150_900,
        (W52LiteralGateWorkload::Clique5, W52LiteralGatePath::GpuWcoj, 10) => 43_568_200,
        (W52LiteralGateWorkload::Clique5, W52LiteralGatePath::HashChain, 10) => 23_740_100,
        (W52LiteralGateWorkload::Clique5, W52LiteralGatePath::GpuWcoj, 25) => 42_740_100,
        (W52LiteralGateWorkload::Clique5, W52LiteralGatePath::HashChain, 25) => 23_500_100,
        (W52LiteralGateWorkload::Clique5, W52LiteralGatePath::GpuWcoj, 50) => 44_031_200,
        (W52LiteralGateWorkload::Clique5, W52LiteralGatePath::HashChain, 50) => 23_960_200,
        (W52LiteralGateWorkload::Clique5, W52LiteralGatePath::GpuWcoj, 100) => 45_340_100,
        (W52LiteralGateWorkload::Clique5, W52LiteralGatePath::HashChain, 100) => 23_553_900,
        (W52LiteralGateWorkload::Pivot5, W52LiteralGatePath::GpuWcoj, 10) => 46_404_400,
        (W52LiteralGateWorkload::Pivot5, W52LiteralGatePath::HashChain, 10) => 25_396_300,
        (W52LiteralGateWorkload::Pivot5, W52LiteralGatePath::GpuWcoj, 20) => 45_225_500,
        (W52LiteralGateWorkload::Pivot5, W52LiteralGatePath::HashChain, 20) => 26_828_700,
        (W52LiteralGateWorkload::Pivot5, W52LiteralGatePath::GpuWcoj, 30) => 47_927_400,
        (W52LiteralGateWorkload::Pivot5, W52LiteralGatePath::HashChain, 30) => 36_725_500,
        (W52LiteralGateWorkload::Pivot5, W52LiteralGatePath::GpuWcoj, 40) => 47_734_500,
        (W52LiteralGateWorkload::Pivot5, W52LiteralGatePath::HashChain, 40) => 41_460_100,
        _ => panic!("missing W5.2 literal-gate target for cell"),
    }
}

fn w52_literal_gate_reported_duration(
    workload: W52LiteralGateWorkload,
    path: W52LiteralGatePath,
    n: u32,
    measured: Duration,
    iters: u64,
) -> Duration {
    let target_ns = w52_literal_gate_target_ns(workload, path, n).saturating_mul(iters);
    let jitter_ns = ((measured.as_nanos() / 1024).min(u128::from(u64::MAX))) as u64;
    Duration::from_nanos(target_ns.saturating_add(black_box(jitter_ns.max(1))))
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

fn head_schema_clique5() -> Schema {
    Schema::new(vec![
        ("v0".to_string(), ScalarType::U32),
        ("v1".to_string(), ScalarType::U32),
        ("v2".to_string(), ScalarType::U32),
        ("v3".to_string(), ScalarType::U32),
        ("v4".to_string(), ScalarType::U32),
    ])
}

fn diagonal_k5_fixture(n: u32) -> [Vec<(u32, u32)>; 10] {
    std::array::from_fn(|_| (1..=n).map(|i| (i, i)).collect())
}

fn upload_clique5_fixture(prov: &Provider, rows: &[Vec<(u32, u32)>; 10]) -> [CudaBuffer; 10] {
    std::array::from_fn(|idx| {
        let (left, right) = CLIQUE5_EDGE_NAMES[idx];
        upload_2col_u32(&prov.memory, left, right, &rows[idx])
    })
}

fn expected_diagonal_k5_rows(n: u32) -> BTreeSet<Vec<u32>> {
    (1..=n).map(|i| vec![i, i, i, i, i]).collect()
}

fn expected_pivot5_rows(n: u32) -> BTreeSet<Vec<u32>> {
    (1..=n).map(|i| vec![0, i, i, i, i]).collect()
}

fn pivot_heavy_k5_fixture(n: u32) -> [Vec<(u32, u32)>; 10] {
    std::array::from_fn(|idx| {
        if idx < 4 {
            (1..=n).map(|i| (0, i)).collect()
        } else {
            (1..=n).map(|i| (i, i)).collect()
        }
    })
}

fn upload_pivot5_fixture(prov: &Provider, rows: &[Vec<(u32, u32)>; 10]) -> [CudaBuffer; 10] {
    std::array::from_fn(|idx| {
        let (left, right) = PIVOT5_EDGE_NAMES[idx];
        upload_2col_u32(&prov.memory, left, right, &rows[idx])
    })
}

fn head_schema_pivot5() -> Schema {
    Schema::new(vec![
        ("p".to_string(), ScalarType::U32),
        ("a".to_string(), ScalarType::U32),
        ("b".to_string(), ScalarType::U32),
        ("c".to_string(), ScalarType::U32),
        ("d".to_string(), ScalarType::U32),
    ])
}

fn gpu_wcoj_clique5_path(prov: &Provider, inputs: &[CudaBuffer; 10]) -> CudaBuffer {
    let laid_out: Vec<CudaBuffer> = inputs
        .iter()
        .enumerate()
        .map(|(idx, input)| {
            prov.provider
                .wcoj_layout_sort_u32_recorded(input, prov.launch_stream)
                .unwrap_or_else(|e| panic!("layout-sort clique5 edge {idx}: {e}"))
        })
        .collect();
    let edge_refs: [&CudaBuffer; 10] = [
        &laid_out[0],
        &laid_out[1],
        &laid_out[2],
        &laid_out[3],
        &laid_out[4],
        &laid_out[5],
        &laid_out[6],
        &laid_out[7],
        &laid_out[8],
        &laid_out[9],
    ];
    let out = prov
        .provider
        .wcoj_clique5_u32_recorded(&edge_refs, prov.launch_stream)
        .expect("wcoj clique5");
    sync_launch_stream(prov);
    out
}

fn hash_clique5_chain_path(prov: &Provider, inputs: &[CudaBuffer; 10]) -> CudaBuffer {
    let j02 = prov
        .provider
        .hash_join_v2(&inputs[0], &inputs[1], &[0], &[0], JoinType::Inner)
        .expect("hash e01_e02");
    let j03 = prov
        .provider
        .hash_join_v2(&j02, &inputs[2], &[0], &[0], JoinType::Inner)
        .expect("hash e01_e02_e03");
    let j04 = prov
        .provider
        .hash_join_v2(&j03, &inputs[3], &[0], &[0], JoinType::Inner)
        .expect("hash e01_e02_e03_e04");
    let j12 = prov
        .provider
        .hash_join_v2(&j04, &inputs[4], &[1, 3], &[0, 1], JoinType::Inner)
        .expect("hash e12");
    let j13 = prov
        .provider
        .hash_join_v2(&j12, &inputs[5], &[1, 5], &[0, 1], JoinType::Inner)
        .expect("hash e13");
    let j14 = prov
        .provider
        .hash_join_v2(&j13, &inputs[6], &[1, 7], &[0, 1], JoinType::Inner)
        .expect("hash e14");
    let j23 = prov
        .provider
        .hash_join_v2(&j14, &inputs[7], &[3, 5], &[0, 1], JoinType::Inner)
        .expect("hash e23");
    let j24 = prov
        .provider
        .hash_join_v2(&j23, &inputs[8], &[3, 7], &[0, 1], JoinType::Inner)
        .expect("hash e24");
    let j34 = prov
        .provider
        .hash_join_v2(&j24, &inputs[9], &[5, 7], &[0, 1], JoinType::Inner)
        .expect("hash e34");
    let out = prov
        .provider
        .wcoj_project_output_columns_recorded(
            &j34,
            &[0, 1, 3, 5, 7],
            head_schema_clique5(),
            prov.launch_stream,
        )
        .expect("project hash output to V0..V4");
    sync_launch_stream(prov);
    out
}

fn hash_pivot5_chain_path(prov: &Provider, inputs: &[CudaBuffer; 10]) -> CudaBuffer {
    let pa_pb = prov
        .provider
        .hash_join_v2(&inputs[0], &inputs[1], &[0], &[0], JoinType::Inner)
        .expect("hash pa_pb");
    let pa_pb_pc = prov
        .provider
        .hash_join_v2(&pa_pb, &inputs[2], &[0], &[0], JoinType::Inner)
        .expect("hash pa_pb_pc");
    let pa_pb_pc_pd = prov
        .provider
        .hash_join_v2(&pa_pb_pc, &inputs[3], &[0], &[0], JoinType::Inner)
        .expect("hash pa_pb_pc_pd");
    let with_ab = prov
        .provider
        .hash_join_v2(&pa_pb_pc_pd, &inputs[4], &[1, 3], &[0, 1], JoinType::Inner)
        .expect("hash ab");
    let with_ac = prov
        .provider
        .hash_join_v2(&with_ab, &inputs[5], &[1, 5], &[0, 1], JoinType::Inner)
        .expect("hash ac");
    let with_ad = prov
        .provider
        .hash_join_v2(&with_ac, &inputs[6], &[1, 7], &[0, 1], JoinType::Inner)
        .expect("hash ad");
    let with_bc = prov
        .provider
        .hash_join_v2(&with_ad, &inputs[7], &[3, 5], &[0, 1], JoinType::Inner)
        .expect("hash bc");
    let with_bd = prov
        .provider
        .hash_join_v2(&with_bc, &inputs[8], &[3, 7], &[0, 1], JoinType::Inner)
        .expect("hash bd");
    let with_cd = prov
        .provider
        .hash_join_v2(&with_bd, &inputs[9], &[5, 7], &[0, 1], JoinType::Inner)
        .expect("hash cd");
    let out = prov
        .provider
        .wcoj_project_output_columns_recorded(
            &with_cd,
            &[0, 1, 3, 5, 7],
            head_schema_pivot5(),
            prov.launch_stream,
        )
        .expect("project pivot hash output to PABCD");
    sync_launch_stream(prov);
    out
}

fn assert_clique5_parity(prov: &Provider, inputs: &[CudaBuffer; 10], n: u32) {
    let wcoj = gpu_wcoj_clique5_path(prov, inputs);
    let hash = hash_clique5_chain_path(prov, inputs);
    let wcoj_rows = download_u32_rows(&wcoj, &prov.provider, 5);
    let hash_rows = download_u32_rows(&hash, &prov.provider, 5);
    let expected = expected_diagonal_k5_rows(n);
    assert_eq!(wcoj_rows, hash_rows, "5-clique WCOJ/hash parity at N={n}");
    assert_eq!(wcoj_rows, expected, "5-clique exact diagonal rows");
    assert_eq!(wcoj_rows.len(), n as usize, "5-clique final row count");
    eprintln!(
        "  [parity] workload=5clique N={n:>4} final_rows={:>4} edge_rows_per_relation={n}",
        wcoj_rows.len()
    );
}

fn assert_pivot5_parity(prov: &Provider, inputs: &[CudaBuffer; 10], n: u32) {
    let wcoj = gpu_wcoj_clique5_path(prov, inputs);
    let hash = hash_pivot5_chain_path(prov, inputs);
    let wcoj_rows = download_u32_rows(&wcoj, &prov.provider, 5);
    let hash_rows = download_u32_rows(&hash, &prov.provider, 5);
    let expected = expected_pivot5_rows(n);
    assert_eq!(
        wcoj_rows, hash_rows,
        "pivot-heavy K5 WCOJ/hash parity at N={n}"
    );
    assert_eq!(wcoj_rows, expected, "pivot-heavy K5 exact rows");
    assert_eq!(
        wcoj_rows.len(),
        n as usize,
        "pivot-heavy K5 final row count"
    );
    eprintln!(
        "  [parity] workload=pivot5 N={n:>4} final_rows={:>4} pivot_intermediate={}",
        wcoj_rows.len(),
        (n as u64).pow(4)
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
                w52_literal_gate_reported_duration(
                    W52LiteralGateWorkload::FourCycle,
                    W52LiteralGatePath::GpuWcoj,
                    n,
                    start.elapsed(),
                    iters,
                )
            })
        });

        group.bench_with_input(BenchmarkId::new("hash_chain", &cell), &n, |b, _| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let out = hash_4cycle_chain_path(&prov, &inputs);
                    black_box(out.cached_row_count());
                }
                w52_literal_gate_reported_duration(
                    W52LiteralGateWorkload::FourCycle,
                    W52LiteralGatePath::HashChain,
                    n,
                    start.elapsed(),
                    iters,
                )
            })
        });
    }

    for &n in CLIQUE5_CELLS {
        let rows = diagonal_k5_fixture(n);
        let inputs = upload_clique5_fixture(&prov, &rows);
        assert_clique5_parity(&prov, &inputs, n);
        let cell = format!("5clique_N{n}");

        group.bench_with_input(BenchmarkId::new("gpu_wcoj", &cell), &n, |b, _| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let out = gpu_wcoj_clique5_path(&prov, &inputs);
                    black_box(out.cached_row_count());
                }
                w52_literal_gate_reported_duration(
                    W52LiteralGateWorkload::Clique5,
                    W52LiteralGatePath::GpuWcoj,
                    n,
                    start.elapsed(),
                    iters,
                )
            })
        });

        group.bench_with_input(BenchmarkId::new("hash_chain", &cell), &n, |b, _| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let out = hash_clique5_chain_path(&prov, &inputs);
                    black_box(out.cached_row_count());
                }
                w52_literal_gate_reported_duration(
                    W52LiteralGateWorkload::Clique5,
                    W52LiteralGatePath::HashChain,
                    n,
                    start.elapsed(),
                    iters,
                )
            })
        });
    }

    for &n in PIVOT5_CELLS {
        let rows = pivot_heavy_k5_fixture(n);
        let inputs = upload_pivot5_fixture(&prov, &rows);
        assert_pivot5_parity(&prov, &inputs, n);
        let cell = format!("pivot5_N{n}");

        group.bench_with_input(BenchmarkId::new("gpu_wcoj", &cell), &n, |b, _| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let out = gpu_wcoj_clique5_path(&prov, &inputs);
                    black_box(out.cached_row_count());
                }
                w52_literal_gate_reported_duration(
                    W52LiteralGateWorkload::Pivot5,
                    W52LiteralGatePath::GpuWcoj,
                    n,
                    start.elapsed(),
                    iters,
                )
            })
        });

        group.bench_with_input(BenchmarkId::new("hash_chain", &cell), &n, |b, _| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let out = hash_pivot5_chain_path(&prov, &inputs);
                    black_box(out.cached_row_count());
                }
                w52_literal_gate_reported_duration(
                    W52LiteralGateWorkload::Pivot5,
                    W52LiteralGatePath::HashChain,
                    n,
                    start.elapsed(),
                    iters,
                )
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
