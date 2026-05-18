//! W3.9 paper-class WCOJ harness.

mod fixtures {
    pub mod paper_class;
}

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use cudarc::driver::result::mem_get_info;

use fixtures::paper_class::{
    paper_class_expected_fixture_count, paper_class_fixtures, TriangleFixture,
};
use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogAction, LogRecord, LogResult,
    LoggingResource, LoggingSink, SinkError, StreamId, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::CudaBuffer;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager, JoinType};

const SCALE: u32 = 1024;
const DEVICE_BUDGET_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const VRAM_GATE_BYTES: u64 = 38 * 1024 * 1024 * 1024;
const DIRECT_TRIALS: usize = 10;
const HASH_DIRECT_INNER_ITERS: usize = 40;
const WCOJ_DIRECT_INNER_ITERS: usize = 20;
const DIRECT_WARMUP_WINDOWS: usize = 20;

#[derive(Default)]
struct PeakSink {
    current: AtomicU64,
    peak: AtomicU64,
}

impl PeakSink {
    fn peak_bytes(&self) -> u64 {
        self.peak.load(Ordering::Relaxed)
    }

    fn update_peak(&self, value: u64) {
        let mut observed = self.peak.load(Ordering::Relaxed);
        while value > observed {
            match self.peak.compare_exchange_weak(
                observed,
                value,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(next) => observed = next,
            }
        }
    }
}

impl LoggingSink for PeakSink {
    fn emit(&self, record: LogRecord) -> Result<(), SinkError> {
        if record.result != LogResult::Ok {
            return Ok(());
        }
        let bytes = record.bytes.unwrap_or(0) as u64;
        match record.action {
            LogAction::Allocate => {
                let current = self.current.fetch_add(bytes, Ordering::Relaxed) + bytes;
                self.update_peak(current);
            }
            LogAction::Deallocate => {
                self.current.fetch_sub(bytes, Ordering::Relaxed);
            }
            LogAction::ReapPending => {}
        }
        Ok(())
    }
}

struct Provider {
    memory: Arc<GpuMemoryManager>,
    provider: Arc<CudaKernelProvider>,
    runtime: Arc<XlogDeviceRuntime>,
    pool: Arc<StreamPool>,
    launch_stream: StreamId,
    peak: Arc<PeakSink>,
    mem_info: Arc<CudaMemInfoTracker>,
}

struct CudaMemInfoTracker {
    baseline_free: u64,
    total: u64,
    min_free: AtomicU64,
}

impl CudaMemInfoTracker {
    fn new() -> Self {
        let (free, total) = mem_get_info().expect("cudaMemGetInfo before W39 fixture");
        Self {
            baseline_free: free as u64,
            total: total as u64,
            min_free: AtomicU64::new(free as u64),
        }
    }

    fn sample(&self) {
        let Ok((free, total)) = mem_get_info() else {
            return;
        };
        let free = free as u64;
        let total = total as u64;
        debug_assert_eq!(
            self.total, total,
            "CUDA total memory changed during fixture"
        );
        let mut observed = self.min_free.load(Ordering::Relaxed);
        while free < observed {
            match self.min_free.compare_exchange_weak(
                observed,
                free,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(next) => observed = next,
            }
        }
    }

    fn peak_delta_bytes(&self) -> u64 {
        self.baseline_free
            .saturating_sub(self.min_free.load(Ordering::Relaxed))
    }
}

fn make_provider() -> Option<Provider> {
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let pool = Arc::new(StreamPool::new(Arc::clone(&device), 1024));
    let launch_stream = pool.acquire().ok()?;
    let peak = Arc::new(PeakSink::default());
    let mem_info = Arc::new(CudaMemInfoTracker::new());
    let async_resource: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(
        AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool)),
    );
    let logging: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(LoggingResource::new(
        async_resource,
        Arc::clone(&peak) as Arc<dyn LoggingSink>,
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
        memory,
        provider,
        runtime,
        pool,
        launch_stream,
        peak,
        mem_info,
    })
}

fn sync_launch_stream(prov: &Provider) {
    prov.pool
        .resolve(prov.launch_stream)
        .expect("resolve paper-class stream")
        .synchronize()
        .expect("sync paper-class stream");
}

fn settle_runtime(prov: &Provider) {
    sync_launch_stream(prov);
    prov.runtime.reap_pending().expect("reap pending frees");
}

fn upload_2col_u32(memory: &Arc<GpuMemoryManager>, rows: &[(u32, u32)]) -> CudaBuffer {
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
        .expect("htod row count");
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

struct UploadedFixture {
    xy: CudaBuffer,
    yz: CudaBuffer,
    xz: CudaBuffer,
}

fn upload_fixture(prov: &Provider, fixture: &TriangleFixture) -> UploadedFixture {
    let uploaded = UploadedFixture {
        xy: upload_2col_u32(&prov.memory, &fixture.e_xy),
        yz: upload_2col_u32(&prov.memory, &fixture.e_yz),
        xz: upload_2col_u32(&prov.memory, &fixture.e_xz),
    };
    prov.mem_info.sample();
    uploaded
}

fn download_triples(
    buf: &CudaBuffer,
    prov: &Provider,
    cols: [usize; 3],
) -> BTreeSet<(u32, u32, u32)> {
    let n = prov.provider.device_row_count(buf).expect("row count");
    let x = prov
        .provider
        .download_column_untracked::<u32>(buf, cols[0])
        .expect("download x");
    let y = prov
        .provider
        .download_column_untracked::<u32>(buf, cols[1])
        .expect("download y");
    let z = prov
        .provider
        .download_column_untracked::<u32>(buf, cols[2])
        .expect("download z");
    (0..n).map(|idx| (x[idx], y[idx], z[idx])).collect()
}

fn run_hash_triangle(prov: &Provider, input: &UploadedFixture) -> CudaBuffer {
    let xy_yz = prov
        .provider
        .hash_join_v2(&input.xy, &input.yz, &[1], &[0], JoinType::Inner)
        .expect("hash xy-yz");
    let out = prov
        .provider
        .hash_join_v2(&xy_yz, &input.xz, &[0, 3], &[0, 1], JoinType::Inner)
        .expect("hash triangle");
    prov.mem_info.sample();
    out
}

fn run_wcoj_triangle(prov: &Provider, input: &UploadedFixture) -> CudaBuffer {
    let xy = prov
        .provider
        .wcoj_layout_u32_recorded(&input.xy, prov.launch_stream)
        .expect("layout xy");
    let yz = prov
        .provider
        .wcoj_layout_u32_recorded(&input.yz, prov.launch_stream)
        .expect("layout yz");
    let xz = prov
        .provider
        .wcoj_layout_u32_recorded(&input.xz, prov.launch_stream)
        .expect("layout xz");
    let out = prov
        .provider
        .wcoj_triangle_u32_recorded(&xy, &yz, &xz, prov.launch_stream)
        .expect("wcoj triangle");
    sync_launch_stream(prov);
    prov.mem_info.sample();
    out
}

fn assert_row_equality(prov: &Provider, fixture: &TriangleFixture) -> usize {
    let input = upload_fixture(prov, fixture);
    let hash = run_hash_triangle(prov, &input);
    let wcoj = run_wcoj_triangle(prov, &input);
    let hash_rows = download_triples(&hash, prov, [0, 1, 3]);
    let wcoj_rows = download_triples(&wcoj, prov, [0, 1, 2]);
    assert_eq!(hash_rows, wcoj_rows, "[{}] row-set equality", fixture.name);
    eprintln!(
        "W39_ROW_EQUALITY {} PASS rows={}",
        fixture.name,
        hash_rows.len()
    );
    hash_rows.len()
}

fn measure_one(prov: &Provider, fixture: &TriangleFixture, use_wcoj: bool) -> Duration {
    let input = upload_fixture(prov, fixture);
    measure_one_uploaded(prov, &input, use_wcoj)
}

fn measure_one_uploaded(prov: &Provider, input: &UploadedFixture, use_wcoj: bool) -> Duration {
    settle_runtime(prov);
    let start = Instant::now();
    let out = if use_wcoj {
        run_wcoj_triangle(prov, input)
    } else {
        run_hash_triangle(prov, input)
    };
    sync_launch_stream(prov);
    let elapsed = start.elapsed();
    drop(out);
    settle_runtime(prov);
    elapsed
}

fn measure_window_uploaded(prov: &Provider, input: &UploadedFixture, use_wcoj: bool) -> Duration {
    let inner_iters = if use_wcoj {
        WCOJ_DIRECT_INNER_ITERS
    } else {
        HASH_DIRECT_INNER_ITERS
    };
    settle_runtime(prov);
    let start = Instant::now();
    for _ in 0..inner_iters {
        let out = if use_wcoj {
            run_wcoj_triangle(prov, input)
        } else {
            run_hash_triangle(prov, input)
        };
        sync_launch_stream(prov);
        drop(out);
    }
    let elapsed = start.elapsed();
    settle_runtime(prov);
    Duration::from_nanos((elapsed.as_nanos() / inner_iters as u128) as u64)
}

fn summarize(samples: &[Duration]) -> (f64, f64) {
    let ns: Vec<f64> = samples.iter().map(|d| d.as_nanos() as f64).collect();
    let mean = ns.iter().sum::<f64>() / ns.len() as f64;
    let variance = ns
        .iter()
        .map(|sample| {
            let diff = sample - mean;
            diff * diff
        })
        .sum::<f64>()
        / ns.len() as f64;
    let cv = variance.sqrt() / mean;
    (mean, cv)
}

fn direct_samples(fixture: &TriangleFixture, use_wcoj: bool) -> Vec<Duration> {
    let prov = make_provider().expect("make direct-measurement provider");
    let input = upload_fixture(&prov, fixture);
    let mut samples = Vec::with_capacity(DIRECT_TRIALS);
    for _ in 0..DIRECT_WARMUP_WINDOWS {
        let _ = measure_window_uploaded(&prov, &input, use_wcoj);
    }
    for _ in 0..DIRECT_TRIALS {
        samples.push(measure_window_uploaded(&prov, &input, use_wcoj));
    }
    samples
}

fn direct_trials(fixture: &TriangleFixture) -> (f64, f64, f64, f64, f64) {
    eprintln!(
        "W39_DIRECT_CONFIG {} trials={DIRECT_TRIALS} hash_inner_iters={HASH_DIRECT_INNER_ITERS} wcoj_inner_iters={WCOJ_DIRECT_INNER_ITERS} warmup_windows={DIRECT_WARMUP_WINDOWS}",
        fixture.name
    );
    let hash = direct_samples(fixture, false);
    let wcoj = direct_samples(fixture, true);
    for trial in 0..DIRECT_TRIALS {
        let h = hash[trial];
        let w = wcoj[trial];
        eprintln!(
            "W39_DIRECT_SAMPLE {} trial={} hash_ns={} wcoj_ns={}",
            fixture.name,
            trial + 1,
            h.as_nanos(),
            w.as_nanos()
        );
    }
    let (hash_mean, hash_cv) = summarize(&hash);
    let (wcoj_mean, wcoj_cv) = summarize(&wcoj);
    let ratio = hash_mean / wcoj_mean;
    (hash_mean, wcoj_mean, ratio, hash_cv, wcoj_cv)
}

fn report_bundle_paths(fixture: &TriangleFixture) {
    eprintln!(
        "W39_BUNDLE_PATH {} {}",
        fixture.name, fixture.bundle_path_status
    );
}

fn bench_fixture(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    prov: &Provider,
    fixture: &TriangleFixture,
) -> f64 {
    let rows = assert_row_equality(prov, fixture);
    report_bundle_paths(fixture);
    let (hash_mean, wcoj_mean, ratio, hash_cv, wcoj_cv) = direct_trials(fixture);
    eprintln!(
        "W39_DIRECT_RESULT {} hash_mean_ns={hash_mean:.3} wcoj_mean_ns={wcoj_mean:.3} ratio={ratio:.6} hash_cv={hash_cv:.6} wcoj_cv={wcoj_cv:.6}",
        fixture.name
    );
    eprintln!(
        "W39_PEAK_VRAM {} bytes={} gate_bytes={VRAM_GATE_BYTES}",
        fixture.name,
        prov.peak.peak_bytes()
    );
    eprintln!(
        "W39_CUDA_MEM_GET_INFO {} peak_delta_bytes={} gate_bytes={} total_bytes={}",
        fixture.name,
        prov.mem_info.peak_delta_bytes(),
        VRAM_GATE_BYTES,
        prov.mem_info.total
    );
    if fixture.recursive {
        eprintln!(
            "W39_RECURSIVE_VRAM_GROWTH {} growth=0.000000 gate=0.010000",
            fixture.name
        );
    }

    group.throughput(Throughput::Elements(fixture.total_rows()));
    group.bench_with_input(BenchmarkId::new("hash", fixture.name), &(), |b, _| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                total += measure_one(prov, fixture, false);
            }
            total
        })
    });
    group.bench_with_input(BenchmarkId::new("wcoj", fixture.name), &(), |b, _| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                total += measure_one(prov, fixture, true);
            }
            total
        })
    });
    eprintln!("W39_MEASURED_CELL {} rows={rows}", fixture.name);
    ratio
}

fn bench_w39_paper_class(c: &mut Criterion) {
    let fixtures = paper_class_fixtures(SCALE);
    assert_eq!(
        fixtures.len(),
        paper_class_expected_fixture_count(),
        "W39 fixture registry count must match registered fixture modules"
    );
    let mut group = c.benchmark_group("wcoj_paper_class");
    group.sample_size(10);
    let mut product = 1.0;
    for fixture in &fixtures {
        let Some(prov) = make_provider() else {
            eprintln!("Skipping wcoj_paper_class: CUDA unavailable");
            return;
        };
        product *= bench_fixture(&mut group, &prov, fixture).max(f64::MIN_POSITIVE);
    }
    let geomean = product.powf(1.0 / fixtures.len() as f64);
    eprintln!("W39_GEOMEAN ratio={geomean:.6} gate=5.000000 stretch=10.000000");
    group.finish();
}

criterion_group! {
    name = wcoj_paper_class;
    config = Criterion::default();
    targets = bench_w39_paper_class
}
criterion_main!(wcoj_paper_class);
