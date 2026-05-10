//! W4.3 production sort-merge benchmark — sort-merge with
//! detection cost vs hash + vs nested-loop on the production
//! eligibility envelope.
//!
//! Differs from the bench-spike at `bench-spike/w43-sort-merge`
//! by including the W4.3-specific **detection-kernel cost** on
//! the sort-merge path (per F-W43-3): the timed region for
//! sort-merge is `is_sorted_ascending_u32(left)` +
//! `is_sorted_ascending_u32(right)` + the join kernel itself,
//! mirroring what production traffic ACTUALLY pays for sorted-
//! eligible joins after W4.3 lands. The hash and nested-loop
//! baselines call the provider directly because those branches
//! never pay the detection cost in production.
//!
//! Mirrors W4.2's bench's provider-direct envelope-parity
//! methodology — the comparison is apples-to-apples on
//! kernel-level work, with detection added on the sort-merge
//! side to isolate the W4.3-specific overhead. The
//! `Executor::execute_node` path was considered (initial
//! iteration of this bench file) but its `execute_scan` buffer-
//! clone overhead inflates Path 1 with an executor-pipeline cost
//! that is **identical for sort-merge and hash in production**
//! (both branches go through scan-clone before dispatch); that
//! overhead therefore does not differentiate the two paths and
//! its inclusion only obscures the relative comparison. Per
//! F-W43-3's INTENT (detection cost included), provider-direct +
//! explicit detection on the sort-merge side is the cleaner
//! interpretation.
//!
//! **Two-part bench design (per F-W43-2 + F-W43-3)**:
//!   - **Part A**: sort-merge-with-detection vs hash on sorted-
//!     eligible cells. D7 acceptance #8 satisfied iff sort-merge
//!     wins ≥ 2× vs hash on these cells.
//!   - **Part B**: sort-merge-with-detection vs nested-loop on
//!     the SAME sorted-eligible cells (D2 precedence overlap
//!     validation per F-W43-2). If sort-merge wins on overlap,
//!     D2 precedence (sort-merge > nested-loop) holds. If
//!     nested-loop wins, iteration-N+ amends D2.
//!
//! Methodology:
//!   * Build provider once. 8 GiB device budget. 1024-stream pool.
//!   * Per cell: upload 3-col U32 buffers (key at col 0, two
//!     payloads).
//!   * Pre-cell parity check (outside timed region) verifies all
//!     three paths produce identical row sets.
//!   * Timed region (sort-merge): provider's
//!     `is_sorted_ascending_u32(left, 0)` +
//!     `is_sorted_ascending_u32(right, 0)` +
//!     `sort_merge_join_v2_inner_u32_1key`.
//!   * Timed region (hash): direct `provider.hash_join_v2`.
//!   * Timed region (nested-loop): direct
//!     `provider.nested_loop_join_v2_inner_u32_1key`.

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
// Mirrors the W4.2 bench's arity choice — multi-col buffers
// match production traffic shape and exercise the full provider
// path including `gather_buffer_by_indices`.
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
// Bench matrix (per W4.3 plan iter-5 Step 12).
//
// Eligible cells are sorted-ascending and within the 4M
// Cartesian threshold. The cells span the spike's tested matrix
// (50–5000) and intentionally include the L=R=2000 cell which
// sits AT the threshold (4M = 4M, ≤ allowed by checked_mul +
// `<=` comparison).
// ---------------------------------------------------------------

const SORTED_ELIGIBLE_MATRIX: &[(u32, u32)] = &[
    (50, 50),     // 2.5K Cartesian
    (100, 100),   // 10K Cartesian
    (250, 250),   // 62.5K Cartesian
    (500, 500),   // 250K Cartesian
    (1000, 1000), // 1M Cartesian
    (2000, 2000), // 4M Cartesian (at threshold)
];

/// Sorted-ascending 3-col U32 fixture. left covers keys
/// `[0..num_left)`; right covers keys `[num_left/2..num_left/2 +
/// num_right)`. Output has `min(num_left/2 + num_right, num_left)
/// - num_left/2` matched rows. For symmetric `(N, N)` cells this
/// gives `N/2` matches (50% match rate).
fn fixture_3col_sorted(
    num_left: u32,
    num_right: u32,
) -> (Vec<(u32, u32, u32)>, Vec<(u32, u32, u32)>) {
    let left: Vec<(u32, u32, u32)> = (0..num_left)
        .map(|i| (i, 1_000_000 + i, 2_000_000 + i))
        .collect();
    let offset = num_left / 2;
    let right: Vec<(u32, u32, u32)> = (offset..offset.saturating_add(num_right))
        .map(|i| (i, 3_000_000 + i, 4_000_000 + i))
        .collect();
    (left, right)
}

fn bench_w43_production(c: &mut Criterion) {
    let prov_holder = match make_provider() {
        Some(p) => p,
        None => {
            eprintln!("Skipping w43 production bench: CUDA runtime unavailable");
            return;
        }
    };
    let memory = Arc::clone(&prov_holder.memory);
    let prov = Arc::clone(&prov_holder.provider);

    // ===========================================================
    // Part A — sort-merge-end-to-end vs hash direct
    // (D7 #8 acceptance: ≥ 2× vs hash on eligible cells).
    //
    // Part B — sort-merge-end-to-end vs nested-loop direct
    // (D2 precedence overlap validation per F-W43-2).
    //
    // Both parts share the same cell matrix and the same
    // sort-merge timing measurement; only the second-path
    // baseline differs. They live in the same criterion group
    // so the output README can compare all three timings per
    // cell side-by-side.
    // ===========================================================
    let mut group = c.benchmark_group("w43_production_sort_merge_vs_hash_vs_nested_loop");

    for &(num_left, num_right) in SORTED_ELIGIBLE_MATRIX {
        let (left_rows, right_rows) = fixture_3col_sorted(num_left, num_right);

        // Pre-cell parity: upload one canonical pair of buffers,
        // run all three paths, verify identical row sets.
        let left_buf = upload_3col_u32(&memory, &left_rows);
        let right_buf = upload_3col_u32(&memory, &right_rows);
        let sm_out = prov
            .sort_merge_join_v2_inner_u32_1key(&left_buf, &right_buf, 0, 0)
            .expect("parity sm");
        let nl_out = prov
            .nested_loop_join_v2_inner_u32_1key(&left_buf, &right_buf, 0, 0)
            .expect("parity nl");
        let hash_out = prov
            .hash_join_v2(&left_buf, &right_buf, &[0], &[0], JoinType::Inner)
            .expect("parity hash");
        let sm_set: BTreeSet<[u32; 6]> = download_6col(&sm_out, &prov).into_iter().collect();
        let nl_set: BTreeSet<[u32; 6]> = download_6col(&nl_out, &prov).into_iter().collect();
        let hash_set: BTreeSet<[u32; 6]> = download_6col(&hash_out, &prov).into_iter().collect();
        assert_eq!(
            sm_set, hash_set,
            "row-set parity FAILED (sort-merge vs hash) at L={} R={}",
            num_left, num_right
        );
        assert_eq!(
            sm_set, nl_set,
            "row-set parity FAILED (sort-merge vs nested-loop) at L={} R={}",
            num_left, num_right
        );
        eprintln!(
            "  [parity] L={:>4} R={:>4} matches={:>5} sm=nl=hash OK",
            num_left,
            num_right,
            sm_set.len()
        );

        // -------------------------------------------------------
        // Path 1 (Part A + Part B): sort-merge with detection.
        // Per F-W43-3: timed region includes the W4.3-specific
        // detection cost (`is_sorted_ascending_u32` × 2 sides).
        // Excludes execute_scan / clone overhead (which is
        // identical for sort-merge and hash dispatch in
        // production and therefore does not differentiate the
        // paths). The two detection calls measure each side's
        // sortedness (both return `Ok(true)` for this fixture);
        // the join kernel then runs unconditionally — same flow
        // production traffic takes after dispatch admits.
        // -------------------------------------------------------
        group.bench_with_input(
            BenchmarkId::new(
                "sort_merge_with_detection",
                format!("L{}xR{}", num_left, num_right),
            ),
            &(),
            |b, _| {
                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        let _l = prov
                            .is_sorted_ascending_u32(&left_buf, 0)
                            .expect("sm detection l");
                        let _r = prov
                            .is_sorted_ascending_u32(&right_buf, 0)
                            .expect("sm detection r");
                        let _ = prov
                            .sort_merge_join_v2_inner_u32_1key(&left_buf, &right_buf, 0, 0)
                            .expect("sm bench");
                    }
                    start.elapsed()
                })
            },
        );

        // -------------------------------------------------------
        // Path 2A — Part A baseline: hash direct.
        // -------------------------------------------------------
        group.bench_with_input(
            BenchmarkId::new("hash_v2_direct", format!("L{}xR{}", num_left, num_right)),
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

        // -------------------------------------------------------
        // Path 2B — Part B baseline: nested-loop direct.
        // -------------------------------------------------------
        group.bench_with_input(
            BenchmarkId::new(
                "nested_loop_direct",
                format!("L{}xR{}", num_left, num_right),
            ),
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
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        // Larger sample size + longer measurement time to keep
        // CIs tight on small-cell timings where GPU thermal /
        // contention noise dominates a 20-sample 3s budget.
        .sample_size(50)
        .measurement_time(Duration::from_secs(8))
        .warm_up_time(Duration::from_secs(1));
    targets = bench_w43_production
}
criterion_main!(benches);
