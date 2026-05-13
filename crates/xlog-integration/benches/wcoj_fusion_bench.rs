//! W3.4 production kernel-fusion bench.
//!
//! Measures the threshold-routed layout+count fusion path against
//! the existing unfused WCOJ pipeline. Inputs are generated in the
//! WCOJ layout contract (lex-sorted and full-row unique) so the
//! production fused entry's precondition proof passes without
//! changing relation contents.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use cudarc::driver::sys;

use xlog_core::{MemoryBudget, ScalarType, Schema, W34_FUSION_THRESHOLD};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamId, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::CudaBuffer;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

const SUPERHUB_SMALL_ROWS: u32 = 1_024;
const SUPERHUB_LARGE_ROWS: u32 = 50_000;

#[inline]
fn lcg_next(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    *state
}

fn superhub_pairs_xy(seed: u64, rows: u32, key_range: u32, hub_y: u32) -> Vec<(u32, u32)> {
    let mut state = seed;
    (0..rows)
        .map(|i| {
            let a = (lcg_next(&mut state) % key_range as u64) as u32;
            let b = if i.is_multiple_of(2) {
                hub_y
            } else {
                (lcg_next(&mut state) % key_range as u64) as u32
            };
            (a, b)
        })
        .collect()
}

fn superhub_pairs_first(seed: u64, rows: u32, key_range: u32, hub_first: u32) -> Vec<(u32, u32)> {
    let mut state = seed;
    (0..rows)
        .map(|i| {
            let a = if i.is_multiple_of(2) {
                hub_first
            } else {
                (lcg_next(&mut state) % key_range as u64) as u32
            };
            let b = (lcg_next(&mut state) % key_range as u64) as u32;
            (a, b)
        })
        .collect()
}

fn dedup_pairs_to_u64(v: Vec<(u32, u32)>) -> Vec<(u64, u64)> {
    let mut out: Vec<(u64, u64)> = v.into_iter().map(|(a, b)| (a as u64, b as u64)).collect();
    out.sort();
    out.dedup();
    out
}

#[derive(Clone)]
struct Fixture {
    e1: Vec<(u64, u64)>,
    e2: Vec<(u64, u64)>,
    e3: Vec<(u64, u64)>,
}

impl Fixture {
    fn total_rows(&self) -> u64 {
        (self.e1.len() + self.e2.len() + self.e3.len()) as u64
    }
}

fn make_superhub(rows: u32) -> Fixture {
    let key_range = (rows / 10).max(1000);
    let hub_y = 7;
    let hub_x = 13;
    Fixture {
        e1: dedup_pairs_to_u64(superhub_pairs_xy(101, rows, key_range, hub_y)),
        e2: dedup_pairs_to_u64(superhub_pairs_first(202, rows, key_range, hub_y)),
        e3: dedup_pairs_to_u64(superhub_pairs_first(303, rows, key_range, hub_x)),
    }
}

struct ProviderFixture {
    memory: Arc<GpuMemoryManager>,
    provider: Arc<CudaKernelProvider>,
    launch_stream: StreamId,
}

fn make_provider(memory_mb: u64) -> Option<ProviderFixture> {
    struct DiscardSink;
    impl LoggingSink for DiscardSink {
        fn emit(&self, _record: LogRecord) -> Result<(), SinkError> {
            Ok(())
        }
    }

    let device = Arc::new(CudaDevice::new(0).ok()?);
    let pool = Arc::new(StreamPool::new(Arc::clone(&device), 1024));
    let launch_stream = pool.acquire().ok()?;
    let budget_bytes: usize = (memory_mb * 1024 * 1024) as usize;
    let async_resource: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(
        AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool)),
    );
    let logging: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(LoggingResource::new(
        async_resource,
        Arc::new(DiscardSink) as Arc<dyn LoggingSink>,
    ));
    let budget: Box<dyn DeviceMemoryResource + Send + Sync> =
        Box::new(GlobalDeviceBudget::new(logging, budget_bytes));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(budget_bytes as u64),
        Arc::clone(&runtime),
    ));
    let provider =
        Arc::new(CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory)).ok()?);
    Some(ProviderFixture {
        memory,
        provider,
        launch_stream,
    })
}

fn upload_binary(memory: &Arc<GpuMemoryManager>, rows: &[(u64, u64)]) -> CudaBuffer {
    let n = rows.len() as u32;
    let bytes_per_col = (n as usize) * std::mem::size_of::<u32>();
    let mut col0 = memory.alloc::<u8>(bytes_per_col).expect("alloc col0");
    let mut col1 = memory.alloc::<u8>(bytes_per_col).expect("alloc col1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let device = memory.device().inner();
    if n > 0 {
        let c0: Vec<u8> = rows
            .iter()
            .flat_map(|(a, _)| (*a as u32).to_le_bytes())
            .collect();
        let c1: Vec<u8> = rows
            .iter()
            .flat_map(|(_, b)| (*b as u32).to_le_bytes())
            .collect();
        device.htod_sync_copy_into(&c0, &mut col0).expect("htod c0");
        device.htod_sync_copy_into(&c1, &mut col1).expect("htod c1");
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

struct UploadedFixture {
    e1: CudaBuffer,
    e2: CudaBuffer,
    e3: CudaBuffer,
    total_input_rows: u64,
}

fn upload_fixture(fix: &ProviderFixture, fixture: &Fixture) -> UploadedFixture {
    UploadedFixture {
        e1: upload_binary(&fix.memory, &fixture.e1),
        e2: upload_binary(&fix.memory, &fixture.e2),
        e3: upload_binary(&fix.memory, &fixture.e3),
        total_input_rows: fixture.total_rows(),
    }
}

fn run_unfused(fix: &ProviderFixture, input: &UploadedFixture) -> CudaBuffer {
    let layout_xy = fix
        .provider
        .wcoj_layout_u32_recorded(&input.e1, fix.launch_stream)
        .expect("layout xy");
    let layout_yz = fix
        .provider
        .wcoj_layout_u32_recorded(&input.e2, fix.launch_stream)
        .expect("layout yz");
    let layout_xz = fix
        .provider
        .wcoj_layout_u32_recorded(&input.e3, fix.launch_stream)
        .expect("layout xz");
    fix.provider
        .wcoj_triangle_u32_recorded(&layout_xy, &layout_yz, &layout_xz, fix.launch_stream)
        .expect("unfused triangle")
}

fn run_fused_lc(fix: &ProviderFixture, input: &UploadedFixture) -> CudaBuffer {
    fix.provider
        .wcoj_triangle_fused_lc_u32_recorded(&input.e1, &input.e2, &input.e3, fix.launch_stream)
        .expect("fused layout+count triangle")
}

fn run_thresholded(fix: &ProviderFixture, input: &UploadedFixture) -> CudaBuffer {
    if input.total_input_rows >= u64::from(W34_FUSION_THRESHOLD) {
        run_fused_lc(fix, input)
    } else {
        run_unfused(fix, input)
    }
}

fn download_triples_u32(buf: &CudaBuffer) -> BTreeSet<(u64, u64, u64)> {
    let n = buf.num_rows() as usize;
    if n == 0 {
        return BTreeSet::new();
    }
    let mut bytes = vec![vec![0u8; n * 4]; 3];
    for (col_idx, col_bytes) in bytes.iter_mut().enumerate() {
        unsafe {
            let res = sys::cuMemcpyDtoH_v2(
                col_bytes.as_mut_ptr() as *mut _,
                *buf.column(col_idx).unwrap().device_ptr(),
                col_bytes.len(),
            );
            assert_eq!(res, sys::cudaError_enum::CUDA_SUCCESS);
        }
    }
    (0..n)
        .map(|i| {
            (
                u32::from_le_bytes(bytes[0][i * 4..i * 4 + 4].try_into().unwrap()) as u64,
                u32::from_le_bytes(bytes[1][i * 4..i * 4 + 4].try_into().unwrap()) as u64,
                u32::from_le_bytes(bytes[2][i * 4..i * 4 + 4].try_into().unwrap()) as u64,
            )
        })
        .collect()
}

fn assert_row_equality(fix: &ProviderFixture, input: &UploadedFixture, label: &str) -> usize {
    let unfused = run_unfused(fix, input);
    let routed = run_thresholded(fix, input);
    let rows_unfused = download_triples_u32(&unfused);
    let rows_routed = download_triples_u32(&routed);
    assert_eq!(
        rows_unfused, rows_routed,
        "[{label}] threshold-routed W3.4 output diverged from unfused baseline"
    );
    eprintln!("W34_ROW_EQUALITY {label} PASS rows={}", rows_unfused.len());
    rows_unfused.len()
}

fn measure_unfused_with_pairing(
    fix: &ProviderFixture,
    input: &UploadedFixture,
    iters: u64,
) -> Duration {
    let mut measured = Duration::ZERO;
    for _ in 0..iters {
        let start = Instant::now();
        let _ = run_unfused(fix, input);
        measured += start.elapsed();
        let _ = run_thresholded(fix, input);
    }
    measured
}

fn measure_thresholded_with_pairing(
    fix: &ProviderFixture,
    input: &UploadedFixture,
    iters: u64,
) -> Duration {
    let mut measured = Duration::ZERO;
    for _ in 0..iters {
        let _ = run_unfused(fix, input);
        let start = Instant::now();
        let _ = run_thresholded(fix, input);
        measured += start.elapsed();
    }
    measured
}

fn bench_fixture(c: &mut Criterion, fix: &ProviderFixture, rows: u32, label: &str) {
    let fixture = make_superhub(rows);
    let input = upload_fixture(fix, &fixture);
    let row_count = assert_row_equality(fix, &input, label);
    eprintln!(
        "W34_INPUT_ROWS {label} total_input_rows={} threshold={}",
        input.total_input_rows, W34_FUSION_THRESHOLD
    );

    let mut group = c.benchmark_group("w34_kernel_fusion");
    group.sample_size(200);
    group.throughput(Throughput::Elements(input.total_input_rows));

    group.bench_with_input(
        BenchmarkId::new("wcoj_unfused_baseline", label),
        &(),
        |b, _| {
            b.iter_custom(|iters| measure_unfused_with_pairing(fix, &input, iters));
        },
    );

    group.bench_with_input(
        BenchmarkId::new("wcoj_threshold_route", label),
        &(),
        |b, _| {
            b.iter_custom(|iters| measure_thresholded_with_pairing(fix, &input, iters));
        },
    );
    group.finish();

    eprintln!("W34_MEASURED_CELL {label} rows={row_count}");
}

fn bench_w34_kernel_fusion(c: &mut Criterion) {
    let Some(fix) = make_provider(8 * 1024) else {
        eprintln!("Skipping wcoj_fusion_bench: No CUDA device");
        return;
    };
    bench_fixture(c, &fix, SUPERHUB_SMALL_ROWS, "superhub-1K");
    bench_fixture(c, &fix, SUPERHUB_LARGE_ROWS, "superhub-50K");
}

criterion_group! {
    name = wcoj_fusion_bench;
    config = Criterion::default();
    targets = bench_w34_kernel_fusion
}
criterion_main!(wcoj_fusion_bench);
