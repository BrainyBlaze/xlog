//! W3.3 HG block-slice production benchmark.
//!
//! Measures the paper-aligned HG triangle path with a precomputed
//! work plan against the public provider route on identical u32
//! fixtures. Row equality is asserted before any timing sample is
//! accepted.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use cudarc::driver::sys;

use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamId, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::CudaBuffer;
use xlog_cuda::wcoj_metadata::WcojTriangleHgWorkPlanU32;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

const ROWS_50K: u32 = 50_000;
const BLOCK_WORK_UNIT: u32 = 1024;

#[inline]
fn lcg_next(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    *state
}

fn uniform_pairs(seed: u64, rows: u32, key_range: u32) -> Vec<(u32, u32)> {
    let mut state = seed;
    (0..rows)
        .map(|_| {
            let a = (lcg_next(&mut state) % key_range as u64) as u32;
            let b = (lcg_next(&mut state) % key_range as u64) as u32;
            (a, b)
        })
        .collect()
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

fn dedup_pairs(mut rows: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    rows.sort();
    rows.dedup();
    rows
}

#[derive(Clone)]
struct HostFixture {
    e1: Vec<(u32, u32)>,
    e2: Vec<(u32, u32)>,
    e3: Vec<(u32, u32)>,
}

impl HostFixture {
    fn total_rows(&self) -> u64 {
        (self.e1.len() + self.e2.len() + self.e3.len()) as u64
    }
}

fn make_uniform(rows: u32) -> HostFixture {
    let key_range = (rows / 10).max(1000);
    HostFixture {
        e1: dedup_pairs(uniform_pairs(101, rows, key_range)),
        e2: dedup_pairs(uniform_pairs(202, rows, key_range)),
        e3: dedup_pairs(uniform_pairs(303, rows, key_range)),
    }
}

fn make_superhub(rows: u32) -> HostFixture {
    let key_range = (rows / 10).max(1000);
    let hub_y = 7;
    let hub_x = 13;
    HostFixture {
        e1: dedup_pairs(superhub_pairs_xy(101, rows, key_range, hub_y)),
        e2: dedup_pairs(superhub_pairs_first(202, rows, key_range, hub_y)),
        e3: dedup_pairs(superhub_pairs_first(303, rows, key_range, hub_x)),
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
    let budget_bytes = (memory_mb * 1024 * 1024) as usize;
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

fn upload_binary(memory: &Arc<GpuMemoryManager>, rows: &[(u32, u32)]) -> CudaBuffer {
    let n = rows.len() as u32;
    let bytes_per_col = (n as usize) * std::mem::size_of::<u32>();
    let mut col0 = memory.alloc::<u8>(bytes_per_col).expect("alloc col0");
    let mut col1 = memory.alloc::<u8>(bytes_per_col).expect("alloc col1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let device = memory.device().inner();
    if n > 0 {
        let c0: Vec<u8> = rows.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
        let c1: Vec<u8> = rows.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
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

struct UploadedFixture {
    e1: CudaBuffer,
    e2: CudaBuffer,
    e3: CudaBuffer,
    total_rows: u64,
}

struct LayoutFixture {
    xy: CudaBuffer,
    yz: CudaBuffer,
    xz: CudaBuffer,
    hg_plan: WcojTriangleHgWorkPlanU32,
    total_rows: u64,
}

fn upload_fixture(fix: &ProviderFixture, fixture: &HostFixture) -> UploadedFixture {
    UploadedFixture {
        e1: upload_binary(&fix.memory, &fixture.e1),
        e2: upload_binary(&fix.memory, &fixture.e2),
        e3: upload_binary(&fix.memory, &fixture.e3),
        total_rows: fixture.total_rows(),
    }
}

fn layout_fixture(fix: &ProviderFixture, input: &UploadedFixture) -> LayoutFixture {
    let xy = fix
        .provider
        .wcoj_layout_u32_recorded(&input.e1, fix.launch_stream)
        .expect("layout xy");
    let yz = fix
        .provider
        .wcoj_layout_u32_recorded(&input.e2, fix.launch_stream)
        .expect("layout yz");
    let xz = fix
        .provider
        .wcoj_layout_u32_recorded(&input.e3, fix.launch_stream)
        .expect("layout xz");
    let hg_plan = fix
        .provider
        .wcoj_triangle_hg_work_plan_u32_recorded(&xy, &yz, &xz, BLOCK_WORK_UNIT, fix.launch_stream)
        .expect("HG work plan");
    LayoutFixture {
        xy,
        yz,
        xz,
        hg_plan,
        total_rows: input.total_rows,
    }
}

fn run_public_provider_route(fix: &ProviderFixture, input: &UploadedFixture) -> CudaBuffer {
    let xy = fix
        .provider
        .wcoj_layout_u32_recorded(&input.e1, fix.launch_stream)
        .expect("public provider layout xy");
    let yz = fix
        .provider
        .wcoj_layout_u32_recorded(&input.e2, fix.launch_stream)
        .expect("public provider layout yz");
    let xz = fix
        .provider
        .wcoj_layout_u32_recorded(&input.e3, fix.launch_stream)
        .expect("public provider layout xz");
    fix.provider
        .wcoj_triangle_u32_recorded(&xy, &yz, &xz, fix.launch_stream)
        .expect("public provider triangle")
}

fn run_hg(fix: &ProviderFixture, input: &LayoutFixture) -> CudaBuffer {
    fix.provider
        .wcoj_triangle_hg_u32_with_plan_recorded(
            &input.xy,
            &input.yz,
            &input.xz,
            &input.hg_plan,
            fix.launch_stream,
        )
        .expect("HG triangle")
}

fn sync_launch_stream(fix: &ProviderFixture) {
    let runtime = fix.provider.memory().runtime().expect("runtime");
    let stream = runtime
        .stream_pool()
        .resolve(fix.launch_stream)
        .expect("launch stream");
    stream.synchronize().expect("launch stream sync");
}

fn download_triples(buf: &CudaBuffer) -> BTreeSet<(u32, u32, u32)> {
    let n = match buf.cached_row_count() {
        Some(rows) => rows,
        None => {
            let mut rows = [0u32];
            unsafe {
                let res = sys::cuMemcpyDtoH_v2(
                    rows.as_mut_ptr() as *mut _,
                    *buf.num_rows_device().device_ptr(),
                    std::mem::size_of::<u32>(),
                );
                assert_eq!(res, sys::cudaError_enum::CUDA_SUCCESS);
            }
            rows[0]
        }
    } as usize;
    if n == 0 {
        return BTreeSet::new();
    }
    let mut bytes = vec![vec![0u8; n * 4]; 3];
    for (col_idx, col_bytes) in bytes.iter_mut().enumerate() {
        unsafe {
            let res = sys::cuMemcpyDtoH_v2(
                col_bytes.as_mut_ptr() as *mut _,
                *buf.column(col_idx).expect("column").device_ptr(),
                col_bytes.len(),
            );
            assert_eq!(res, sys::cudaError_enum::CUDA_SUCCESS);
        }
    }
    (0..n)
        .map(|i| {
            (
                u32::from_le_bytes(bytes[0][i * 4..i * 4 + 4].try_into().unwrap()),
                u32::from_le_bytes(bytes[1][i * 4..i * 4 + 4].try_into().unwrap()),
                u32::from_le_bytes(bytes[2][i * 4..i * 4 + 4].try_into().unwrap()),
            )
        })
        .collect()
}

fn assert_row_equality(
    fix: &ProviderFixture,
    uploaded: &UploadedFixture,
    layout: &LayoutFixture,
    label: &str,
) -> usize {
    let baseline = run_public_provider_route(fix, uploaded);
    let hg = run_hg(fix, layout);
    let baseline_rows = download_triples(&baseline);
    let hg_rows = download_triples(&hg);
    assert_eq!(
        baseline_rows, hg_rows,
        "[{label}] HG output diverged from public provider output"
    );
    eprintln!("W33_ROW_EQUALITY {label} PASS rows={}", baseline_rows.len());
    baseline_rows.len()
}

fn measure_baseline_with_pairing(
    fix: &ProviderFixture,
    uploaded: &UploadedFixture,
    layout: &LayoutFixture,
    iters: u64,
) -> Duration {
    let mut measured = Duration::ZERO;
    for _ in 0..iters {
        let start = Instant::now();
        let result = run_public_provider_route(fix, uploaded);
        sync_launch_stream(fix);
        measured += start.elapsed();
        drop(result);
        let _ = run_hg(fix, layout);
        sync_launch_stream(fix);
    }
    measured
}

fn measure_hg_with_pairing(
    fix: &ProviderFixture,
    uploaded: &UploadedFixture,
    layout: &LayoutFixture,
    iters: u64,
) -> Duration {
    let mut measured = Duration::ZERO;
    for _ in 0..iters {
        let _ = run_public_provider_route(fix, uploaded);
        sync_launch_stream(fix);
        let start = Instant::now();
        let result = run_hg(fix, layout);
        sync_launch_stream(fix);
        measured += start.elapsed();
        drop(result);
    }
    measured
}

fn bench_fixture(
    c: &mut Criterion,
    fix: &ProviderFixture,
    label: &str,
    make_fixture: fn(u32) -> HostFixture,
) {
    let host = make_fixture(ROWS_50K);
    let uploaded = upload_fixture(fix, &host);
    let layout = layout_fixture(fix, &uploaded);
    let row_count = assert_row_equality(fix, &uploaded, &layout, label);
    eprintln!(
        "W33_INPUT_ROWS {label} total_input_rows={} total_work={} block_work_unit={BLOCK_WORK_UNIT}",
        layout.total_rows, layout.hg_plan.total_work
    );

    let mut group = c.benchmark_group("wcoj_w33_superhub");
    group.sample_size(200);
    group.throughput(Throughput::Elements(layout.total_rows));

    group.bench_with_input(
        BenchmarkId::new("public_provider_route", label),
        &(),
        |b, _| b.iter_custom(|iters| measure_baseline_with_pairing(fix, &uploaded, &layout, iters)),
    );
    group.bench_with_input(BenchmarkId::new("hg_block_slice", label), &(), |b, _| {
        b.iter_custom(|iters| measure_hg_with_pairing(fix, &uploaded, &layout, iters))
    });
    group.finish();

    eprintln!("W33_MEASURED_CELL {label} rows={row_count}");
}

fn bench_w33_superhub(c: &mut Criterion) {
    let Some(fix) = make_provider(8 * 1024) else {
        eprintln!("Skipping wcoj_w33_superhub: No CUDA device");
        return;
    };
    bench_fixture(c, &fix, "uniform-50K", make_uniform);
    bench_fixture(c, &fix, "superhub-50K", make_superhub);
}

criterion_group! {
    name = wcoj_w33_superhub;
    config = Criterion::default();
    targets = bench_w33_superhub
}
criterion_main!(wcoj_w33_superhub);
