//! W4.2 production-kernel benchmark — nested-loop vs hash on the
//! production eligibility envelope.
//!
//! Differs from the bench-spike at `bench-spike/w42-nested-loop`
//! in two structural ways:
//!
//!   1. **Multi-col arity** — 3-col buffers (key at col 0, two
//!      payload columns) match production traffic shape. The
//!      spike was 1-col-no-payload to isolate kernel cost; this
//!      bench exercises the full provider path including the
//!      `gather_buffer_by_indices` materialization step.
//!
//!   2. **Eligible cells get nested-loop benched; above-threshold
//!      cells are hash-only** — the production provider Errs on
//!      `num_left * num_right > NESTED_LOOP_TOTAL_THRESHOLD`, so
//!      benching nested-loop above the threshold would fail.
//!      Above-threshold hash measurements establish hash's
//!      scaling baseline for the evidence README.
//!
//! Methodology (provider-direct envelope-parity):
//!   * Build provider once. 8 GiB device budget. 1024-stream pool.
//!   * Per cell: upload buffers; pre-run row-set parity (NL vs
//!     hash, both produce 6-col `combine_schemas` output) outside
//!     the timed region.
//!   * Timed region: `provider.nested_loop_join_v2_inner_u32_1key`
//!     or `provider.hash_join_v2` only. Same uploaded buffers
//!     across both paths within a cell.
//!
//! D7 acceptance criterion #6: nested-loop must win ≥ 2× vs hash
//! on the eligible cells.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::CudaBuffer;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager, JoinType};

// ---------------------------------------------------------------
// Provider setup.
// ---------------------------------------------------------------

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
    _pool: Arc<StreamPool>,
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
    let budget: Box<dyn DeviceMemoryResource + Send + Sync> =
        Box::new(GlobalDeviceBudget::new(logging, 8 * 1024 * 1024 * 1024));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(8 * 1024 * 1024 * 1024),
        Arc::clone(&runtime),
    ));
    let provider =
        Arc::new(CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory)).ok()?);
    Some(Provider {
        _device: device,
        _runtime: runtime,
        memory,
        provider,
        _pool: pool,
    })
}

// ---------------------------------------------------------------
// 3-col U32 fixture upload + 6-col parity download.
// ---------------------------------------------------------------

fn upload_3col_u32(memory: &Arc<GpuMemoryManager>, rows: &[(u32, u32, u32)]) -> CudaBuffer {
    let n = rows.len() as u32;
    let bytes_per_col = (n as usize) * 4;
    let mut col0 = memory.alloc::<u8>(bytes_per_col).expect("alloc col0");
    let mut col1 = memory.alloc::<u8>(bytes_per_col).expect("alloc col1");
    let mut col2 = memory.alloc::<u8>(bytes_per_col).expect("alloc col2");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let dev = memory.device().inner();
    if n > 0 {
        let b0: Vec<u8> = rows.iter().flat_map(|(a, _, _)| a.to_le_bytes()).collect();
        let b1: Vec<u8> = rows.iter().flat_map(|(_, a, _)| a.to_le_bytes()).collect();
        let b2: Vec<u8> = rows.iter().flat_map(|(_, _, a)| a.to_le_bytes()).collect();
        dev.htod_sync_copy_into(&b0, &mut col0).expect("htod c0");
        dev.htod_sync_copy_into(&b1, &mut col1).expect("htod c1");
        dev.htod_sync_copy_into(&b2, &mut col2).expect("htod c2");
    }
    dev.htod_sync_copy_into(&[n], &mut d_num_rows)
        .expect("htod n");
    let schema = Schema::new(vec![
        ("k".to_string(), ScalarType::U32),
        ("p1".to_string(), ScalarType::U32),
        ("p2".to_string(), ScalarType::U32),
    ]);
    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into(), col2.into()],
        n as u64,
        d_num_rows,
        schema,
        n,
    )
}

/// Download a 6-col U32 buffer (the `combine_schemas` output of
/// 3-col left ⋈ 3-col right). Returns `Vec<[u32; 6]>` for
/// `BTreeSet`-friendly comparison.
fn download_6col(buf: &CudaBuffer, prov: &CudaKernelProvider) -> Vec<[u32; 6]> {
    let n = prov.device_row_count(buf).expect("row count");
    if n == 0 {
        return Vec::new();
    }
    let cols: Vec<Vec<u32>> = (0..6)
        .map(|i| {
            prov.download_column_untracked::<u32>(buf, i)
                .expect("download col")
        })
        .collect();
    (0..n)
        .map(|i| {
            [
                cols[0][i], cols[1][i], cols[2][i], cols[3][i], cols[4][i], cols[5][i],
            ]
        })
        .collect()
}

// ---------------------------------------------------------------
// Bench matrix (per W4.2 plan iter-4 Step 12).
// ---------------------------------------------------------------

/// Eligible cells: nested-loop AND hash benched. Matrix covers
/// `(L, R)` ∈ {(100,100), (500,500), (1000,1000), (2000,2000)} —
/// all within the 4M Cartesian threshold.
const ELIGIBLE_MATRIX: &[(u32, u32)] = &[(100, 100), (500, 500), (1000, 1000), (2000, 2000)];

/// Above-threshold cells: hash only. Establishes hash's scaling
/// baseline for the evidence README. The W4.2 dispatcher routes
/// these to hash (they exceed 4M Cartesian).
const ABOVE_THRESHOLD_MATRIX: &[(u32, u32)] = &[
    (5000, 5000),  // 25M Cartesian
    (10000, 1000), // 10M Cartesian (asymmetric)
];

/// Fixture: 3-col U32 buffers with key at col 0. left covers
/// keys `[0..num_left)`, right covers keys
/// `[num_left/2..num_left/2 + num_right)` so output has
/// `min(num_left/2 + num_right, num_left) - num_left/2`
/// matched rows. For symmetric `(N, N)` cells this gives
/// `N/2 + 1` matches (50% match rate).
fn fixture_3col(num_left: u32, num_right: u32) -> (Vec<(u32, u32, u32)>, Vec<(u32, u32, u32)>) {
    let left: Vec<(u32, u32, u32)> = (0..num_left)
        .map(|i| (i, 1_000_000 + i, 2_000_000 + i))
        .collect();
    let offset = num_left / 2;
    let right: Vec<(u32, u32, u32)> = (offset..offset.saturating_add(num_right))
        .map(|i| (i, 3_000_000 + i, 4_000_000 + i))
        .collect();
    (left, right)
}

fn bench_w42_production(c: &mut Criterion) {
    let prov_holder = match make_provider() {
        Some(p) => p,
        None => {
            eprintln!("Skipping w42 production bench: CUDA runtime unavailable");
            return;
        }
    };
    let memory = Arc::clone(&prov_holder.memory);
    let prov = Arc::clone(&prov_holder.provider);

    let mut group = c.benchmark_group("w42_production_nested_loop_vs_hash");

    // Eligible cells: bench both paths after parity-checking they
    // produce identical row sets.
    for &(num_left, num_right) in ELIGIBLE_MATRIX {
        let (left_rows, right_rows) = fixture_3col(num_left, num_right);
        let left_buf = upload_3col_u32(&memory, &left_rows);
        let right_buf = upload_3col_u32(&memory, &right_rows);

        // Pre-cell parity check (outside timed region).
        let nl_out = prov
            .nested_loop_join_v2_inner_u32_1key(&left_buf, &right_buf, 0, 0)
            .expect("parity nl");
        let hash_out = prov
            .hash_join_v2(&left_buf, &right_buf, &[0], &[0], JoinType::Inner)
            .expect("parity hash");
        let nl_set: BTreeSet<[u32; 6]> = download_6col(&nl_out, &prov).into_iter().collect();
        let hash_set: BTreeSet<[u32; 6]> = download_6col(&hash_out, &prov).into_iter().collect();
        assert_eq!(
            nl_set, hash_set,
            "row-set parity FAILED at L={} R={}",
            num_left, num_right
        );
        eprintln!(
            "  [parity] L={:>5} R={:>5} matches={:>5} OK",
            num_left,
            num_right,
            nl_set.len()
        );

        group.bench_with_input(
            BenchmarkId::new("nested_loop", format!("L{}xR{}", num_left, num_right)),
            &(),
            |b, _| {
                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        let _ = prov
                            .nested_loop_join_v2_inner_u32_1key(&left_buf, &right_buf, 0, 0)
                            .expect("nl bench");
                    }
                    start.elapsed()
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("hash_v2", format!("L{}xR{}", num_left, num_right)),
            &(),
            |b, _| {
                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        let _ = prov
                            .hash_join_v2(&left_buf, &right_buf, &[0], &[0], JoinType::Inner)
                            .expect("hash bench");
                    }
                    start.elapsed()
                })
            },
        );
    }

    // Above-threshold cells: hash only (nested-loop provider Errs
    // above the 4M Cartesian threshold). These rows establish
    // hash's scaling baseline outside the W4.2 envelope.
    for &(num_left, num_right) in ABOVE_THRESHOLD_MATRIX {
        let (left_rows, right_rows) = fixture_3col(num_left, num_right);
        let left_buf = upload_3col_u32(&memory, &left_rows);
        let right_buf = upload_3col_u32(&memory, &right_rows);
        eprintln!(
            "  [hash-only above-threshold] L={:>5} R={:>5} cartesian={}",
            num_left,
            num_right,
            (num_left as u64) * (num_right as u64)
        );
        group.bench_with_input(
            BenchmarkId::new("hash_v2", format!("L{}xR{}", num_left, num_right)),
            &(),
            |b, _| {
                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        let _ = prov
                            .hash_join_v2(&left_buf, &right_buf, &[0], &[0], JoinType::Inner)
                            .expect("hash bench");
                    }
                    start.elapsed()
                })
            },
        );
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .measurement_time(Duration::from_secs(3))
        .warm_up_time(Duration::from_millis(500));
    targets = bench_w42_production
}
criterion_main!(benches);
