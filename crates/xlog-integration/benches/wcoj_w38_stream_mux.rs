use std::collections::BTreeSet;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamId, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::CudaBuffer;
use xlog_cuda::wcoj_metadata::WcojTriangleHgWorkPlanU32;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

const DEVICE_BUDGET_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const ROWS: u32 = 2_000_000;
const BLOCK_WORK_UNIT: u32 = 65_536;
const RULES: usize = 4;

#[derive(Clone)]
struct HostTriangle {
    xy: Vec<(u32, u32)>,
    yz: Vec<(u32, u32)>,
    xz: Vec<(u32, u32)>,
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
    streams: Vec<StreamId>,
}

struct RuleInput {
    xy: CudaBuffer,
    yz: CudaBuffer,
    xz: CudaBuffer,
    plan: WcojTriangleHgWorkPlanU32,
    stream: StreamId,
    input_rows: u64,
}

fn lcg_next(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    *state
}

fn dedup_pairs(mut rows: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    rows.sort();
    rows.dedup();
    rows
}

fn superhub_pairs_xy(seed: u64, rows: u32, key_range: u32, hub_y: u32) -> Vec<(u32, u32)> {
    let mut state = seed;
    (0..rows)
        .map(|i| {
            let x = (lcg_next(&mut state) % key_range as u64) as u32;
            let y = if i.is_multiple_of(2) {
                hub_y
            } else {
                (lcg_next(&mut state) % key_range as u64) as u32
            };
            (x, y)
        })
        .collect()
}

fn superhub_pairs_first(seed: u64, rows: u32, key_range: u32, hub_first: u32) -> Vec<(u32, u32)> {
    let mut state = seed;
    (0..rows)
        .map(|i| {
            let first = if i.is_multiple_of(2) {
                hub_first
            } else {
                (lcg_next(&mut state) % key_range as u64) as u32
            };
            let second = (lcg_next(&mut state) % key_range as u64) as u32;
            (first, second)
        })
        .collect()
}

fn make_rule_fixture(rule_idx: usize) -> HostTriangle {
    let key_range = (ROWS / 8).max(1000);
    let hub_y = 17 + rule_idx as u32 * 31;
    let hub_x = 23 + rule_idx as u32 * 37;
    let seed = 1000 + rule_idx as u64 * 100;
    HostTriangle {
        xy: dedup_pairs(superhub_pairs_xy(seed + 1, ROWS, key_range, hub_y)),
        yz: dedup_pairs(superhub_pairs_first(seed + 2, ROWS, key_range, hub_y)),
        xz: dedup_pairs(superhub_pairs_first(seed + 3, ROWS, key_range, hub_x)),
    }
}

fn make_provider() -> Option<Provider> {
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let pool = Arc::new(StreamPool::new(Arc::clone(&device), 1024));
    let streams = (0..RULES)
        .map(|_| pool.acquire().ok())
        .collect::<Option<Vec<_>>>()?;
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
    Some(Provider {
        _device: device,
        _runtime: runtime,
        memory,
        provider,
        pool,
        streams,
    })
}

fn sync_stream(prov: &Provider, stream: StreamId) {
    prov.pool
        .resolve(stream)
        .expect("resolve stream")
        .synchronize()
        .expect("sync stream");
}

fn upload_binary(memory: &Arc<GpuMemoryManager>, rows: &[(u32, u32)]) -> CudaBuffer {
    let n = rows.len() as u32;
    let bytes_per_col = (n as usize) * std::mem::size_of::<u32>();
    let mut col0 = memory.alloc::<u8>(bytes_per_col).expect("alloc col0");
    let mut col1 = memory.alloc::<u8>(bytes_per_col).expect("alloc col1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let device = memory.device().inner();
    if n > 0 {
        let c0: Vec<u8> = rows.iter().flat_map(|(x, _)| x.to_le_bytes()).collect();
        let c1: Vec<u8> = rows.iter().flat_map(|(_, y)| y.to_le_bytes()).collect();
        device.htod_sync_copy_into(&c0, &mut col0).expect("htod c0");
        device.htod_sync_copy_into(&c1, &mut col1).expect("htod c1");
    }
    device
        .htod_sync_copy_into(&[n], &mut d_num_rows)
        .expect("htod rows");
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

fn upload_rule(prov: &Provider, host: &HostTriangle, stream: StreamId) -> RuleInput {
    let xy_raw = upload_binary(&prov.memory, &host.xy);
    let yz_raw = upload_binary(&prov.memory, &host.yz);
    let xz_raw = upload_binary(&prov.memory, &host.xz);
    let xy = prov
        .provider
        .wcoj_layout_u32_recorded(&xy_raw, stream)
        .expect("layout xy");
    let yz = prov
        .provider
        .wcoj_layout_u32_recorded(&yz_raw, stream)
        .expect("layout yz");
    let xz = prov
        .provider
        .wcoj_layout_u32_recorded(&xz_raw, stream)
        .expect("layout xz");
    let plan = prov
        .provider
        .wcoj_triangle_hg_work_plan_u32_recorded(&xy, &yz, &xz, BLOCK_WORK_UNIT, stream)
        .expect("work plan");
    sync_stream(prov, stream);
    RuleInput {
        xy,
        yz,
        xz,
        plan,
        stream,
        input_rows: (host.xy.len() + host.yz.len() + host.xz.len()) as u64,
    }
}

fn run_rule(prov: &Provider, rule: &RuleInput) -> CudaBuffer {
    let count = prov
        .provider
        .wcoj_triangle_hg_count_phase_u32_recorded(
            &rule.xy,
            &rule.yz,
            &rule.xz,
            &rule.plan,
            rule.stream,
        )
        .expect("count phase");
    prov.provider
        .wcoj_triangle_hg_materialize_phase_u32_recorded(
            &rule.xy,
            &rule.yz,
            &rule.xz,
            &rule.plan,
            count,
            rule.stream,
        )
        .expect("materialize phase")
}

fn run_public_reference(prov: &Provider, rule: &RuleInput) -> CudaBuffer {
    prov.provider
        .wcoj_triangle_u32_recorded(&rule.xy, &rule.yz, &rule.xz, rule.stream)
        .expect("public reference")
}

fn download_triples(prov: &Provider, buf: &CudaBuffer) -> BTreeSet<(u32, u32, u32)> {
    assert_eq!(buf.arity(), 3);
    let n = prov.provider.device_row_count(buf).expect("row count");
    let x = prov
        .provider
        .download_column_untracked::<u32>(buf, 0)
        .expect("download x");
    let y = prov
        .provider
        .download_column_untracked::<u32>(buf, 1)
        .expect("download y");
    let z = prov
        .provider
        .download_column_untracked::<u32>(buf, 2)
        .expect("download z");
    (0..n).map(|idx| (x[idx], y[idx], z[idx])).collect()
}

fn assert_sets_equal(
    label: &str,
    expected: &BTreeSet<(u32, u32, u32)>,
    actual: &BTreeSet<(u32, u32, u32)>,
) {
    if expected == actual {
        return;
    }
    let missing: Vec<_> = expected.difference(actual).take(8).copied().collect();
    let extra: Vec<_> = actual.difference(expected).take(8).copied().collect();
    panic!(
        "{label}: expected_rows={} actual_rows={} missing_sample={missing:?} extra_sample={extra:?}",
        expected.len(),
        actual.len()
    );
}

fn assert_row_equality(prov: &Provider, rules: &[RuleInput]) {
    for (rule_idx, rule) in rules.iter().enumerate() {
        let seq = run_rule(prov, rule);
        let reference = run_public_reference(prov, rule);
        let mux = run_rule(prov, rule);
        let reference_rows = download_triples(prov, &reference);
        let seq_rows = download_triples(prov, &seq);
        let mux_rows = download_triples(prov, &mux);
        assert_sets_equal(
            &format!("public vs phase sequential rule {rule_idx}"),
            &reference_rows,
            &seq_rows,
        );
        assert_sets_equal(
            &format!("phase sequential vs mux rule {rule_idx}"),
            &seq_rows,
            &mux_rows,
        );
        eprintln!(
            "W38_ROW_EQUALITY rule={} PASS rows={} total_work={}",
            rule_idx,
            seq_rows.len(),
            rule.plan.total_work
        );
    }
}

fn measure_sequential(prov: &Provider, rules: &[RuleInput], iters: u64) -> Duration {
    let mut measured = Duration::ZERO;
    for _ in 0..iters {
        let start = Instant::now();
        for rule in rules {
            let out = run_rule(prov, rule);
            black_box(out.cached_row_count());
        }
        measured += start.elapsed();
    }
    measured
}

fn measure_mux(prov: &Provider, rules: &[RuleInput], iters: u64) -> Duration {
    let start = Instant::now();
    let barrier = Arc::new(Barrier::new(rules.len()));
    thread::scope(|scope| {
        for rule in rules {
            let barrier = Arc::clone(&barrier);
            scope.spawn(move || {
                for _ in 0..iters {
                    let count = prov
                        .provider
                        .wcoj_triangle_hg_count_phase_u32_recorded(
                            &rule.xy,
                            &rule.yz,
                            &rule.xz,
                            &rule.plan,
                            rule.stream,
                        )
                        .expect("count phase");
                    barrier.wait();
                    let out = prov
                        .provider
                        .wcoj_triangle_hg_materialize_phase_u32_recorded(
                            &rule.xy,
                            &rule.yz,
                            &rule.xz,
                            &rule.plan,
                            count,
                            rule.stream,
                        )
                        .expect("materialize phase");
                    black_box(out.cached_row_count());
                    barrier.wait();
                }
            });
        }
    });
    start.elapsed()
}

fn measure_stream_scheduler(prov: &Provider, rules: &[RuleInput], iters: u64) -> Duration {
    if rules.len() <= 1 {
        measure_sequential(prov, rules, iters)
    } else {
        measure_mux(prov, rules, iters)
    }
}

fn direct_measure(prov: &Provider, rules: &[RuleInput]) {
    let seq = measure_sequential(prov, rules, 5);
    let mux = measure_mux(prov, rules, 5);
    let ratio = seq.as_secs_f64() / mux.as_secs_f64();
    let sum_single: f64 = rules
        .iter()
        .map(|rule| {
            let start = Instant::now();
            let out = run_rule(prov, rule);
            black_box(out.cached_row_count());
            start.elapsed().as_secs_f64()
        })
        .sum();
    let wall = measure_mux(prov, rules, 1).as_secs_f64();
    let concurrency = sum_single / wall;
    eprintln!(
        "W38_DIRECT_MEASURE sequential_us={:.3} mux_us={:.3} speedup={:.3} concurrency={:.3}",
        seq.as_secs_f64() * 1.0e6,
        mux.as_secs_f64() * 1.0e6,
        ratio,
        concurrency
    );
}

fn direct_measure_single_rule(prov: &Provider, rule: &RuleInput) {
    let one_rule = std::slice::from_ref(rule);
    let seq = measure_sequential(prov, one_rule, 5);
    let scheduled = measure_stream_scheduler(prov, one_rule, 5);
    let ratio = seq.as_secs_f64() / scheduled.as_secs_f64();
    eprintln!(
        "W38_SINGLE_RULE_MEASURE sequential_us={:.3} scheduled_us={:.3} ratio={:.3}",
        seq.as_secs_f64() * 1.0e6,
        scheduled.as_secs_f64() * 1.0e6,
        ratio
    );
}

fn bench_w38_stream_mux(c: &mut Criterion) {
    let Some(prov) = make_provider() else {
        eprintln!("Skipping wcoj_w38_stream_mux: CUDA runtime unavailable");
        return;
    };
    let hosts: Vec<_> = (0..RULES).map(make_rule_fixture).collect();
    let rules: Vec<_> = hosts
        .iter()
        .zip(prov.streams.iter().copied())
        .map(|(host, stream)| upload_rule(&prov, host, stream))
        .collect();
    assert_row_equality(&prov, &rules);
    direct_measure(&prov, &rules);
    direct_measure_single_rule(&prov, &rules[0]);

    let total_input_rows: u64 = rules.iter().map(|rule| rule.input_rows).sum();
    let mut group = c.benchmark_group("wcoj_w38_stream_mux");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(6));
    group.warm_up_time(Duration::from_secs(1));
    group.throughput(Throughput::Elements(total_input_rows));
    group.bench_with_input(
        BenchmarkId::new("sequential_dispatch", "four_rule_superhub_2M_sparse_grid"),
        &(),
        |b, _| b.iter_custom(|iters| measure_sequential(&prov, &rules, iters)),
    );
    group.bench_with_input(
        BenchmarkId::new("hand_stream_mux", "four_rule_superhub_2M_sparse_grid"),
        &(),
        |b, _| b.iter_custom(|iters| measure_mux(&prov, &rules, iters)),
    );
    group.finish();

    let single_rule = &rules[0..1];
    let mut single_group = c.benchmark_group("wcoj_w38_stream_mux_single_rule");
    single_group.sample_size(20);
    single_group.measurement_time(Duration::from_secs(6));
    single_group.warm_up_time(Duration::from_secs(1));
    single_group.throughput(Throughput::Elements(single_rule[0].input_rows));
    single_group.bench_with_input(
        BenchmarkId::new("sequential_reference", "one_rule_superhub_2M"),
        &(),
        |b, _| b.iter_custom(|iters| measure_sequential(&prov, single_rule, iters)),
    );
    single_group.bench_with_input(
        BenchmarkId::new("stream_scheduler", "one_rule_superhub_2M"),
        &(),
        |b, _| b.iter_custom(|iters| measure_stream_scheduler(&prov, single_rule, iters)),
    );
    single_group.finish();
}

criterion_group! {
    name = wcoj_w38_stream_mux;
    config = Criterion::default();
    targets = bench_w38_stream_mux
}
criterion_main!(wcoj_w38_stream_mux);
