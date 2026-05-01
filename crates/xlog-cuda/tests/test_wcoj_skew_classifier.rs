// crates/xlog-cuda/tests/test_wcoj_skew_classifier.rs
//! v0.6.2 commit A — WCOJ adaptive-dispatch classifier.
//!
//! RED-first tests for:
//!
//!   * `CudaKernelProvider::dtoh_small_metadata_untracked<T>` —
//!     a sibling of `dtoh_scalar_untracked` for small metadata
//!     vectors (≤ 4096 bytes). Whitelisted by the strict
//!     deterministic-D2H gate. NOT a general data-plane D2H
//!     escape hatch.
//!   * `CudaKernelProvider::wcoj_triangle_skew_score_u32` and
//!     `wcoj_triangle_skew_score_u64` — single-launch combined
//!     histogram + host-side max-bucket-fraction across the
//!     three triangle join-key columns (e1.col1, e2.col0,
//!     e3.col0). Returns `Ok(Some(score))` on success or
//!     `Ok(None)` on any classifier-side failure (silent
//!     fallback per the slice spec — classifier is an
//!     optimization, not a correctness primitive).
//!
//! Hard scope (commit A of 3):
//!   * Provider/kernel surface only — no executor wiring.
//!   * No `RuntimeConfig::wcoj_triangle_dispatch_adaptive` —
//!     that field lands in commit B with the dispatcher branch.
//!   * No bench cells — that's commit C.
//!
//! Score semantics (locked from
//! `docs/evidence/2026-05-01-wcoj-bench-baseline/` probe):
//!
//!   bucket_u32(key) = (key * 2654435761u32) >> 26   // high 6 bits, 64 buckets
//!   bucket_u64(key) = SplitMix64(key) >> 58         // high 6 bits, 64 buckets
//!   score_i = max_bucket_count(col_i) / row_count(col_i)
//!   score   = max(score_e1, score_e2, score_e3)
//!
//! Threshold convention (consumed in commit B): score >= 0.10
//! → dispatch WCOJ; else fall back. Probe shows uniform/empty
//! ≈ 0.018–0.020 (margin 5×) and super-hub ≈ 0.180 (margin
//! 1.8×) at 64 buckets with the multiplicative mixer.

use std::sync::Arc;

use xlog_core::{MemoryBudget, ScalarType, Schema, XlogError};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamId, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::CudaBuffer;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

// ---------------------------------------------------------------
// Fixture helpers (mirror the bench's LCG + bucketing semantics)
// ---------------------------------------------------------------

struct DiscardSink;
impl LoggingSink for DiscardSink {
    fn emit(&self, _record: LogRecord) -> Result<(), SinkError> {
        Ok(())
    }
}

#[allow(dead_code)]
struct Fix {
    device: Arc<CudaDevice>,
    runtime: Arc<XlogDeviceRuntime>,
    memory: Arc<GpuMemoryManager>,
    provider: CudaKernelProvider,
    pool: Arc<StreamPool>,
}

fn make_fix() -> Option<Fix> {
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
    let limit_bytes: usize = 256 * 1024 * 1024;
    let async_resource: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(
        AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool)),
    );
    let logging: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(LoggingResource::new(
        async_resource,
        Arc::new(DiscardSink) as Arc<dyn LoggingSink>,
    ));
    let budget: Box<dyn DeviceMemoryResource + Send + Sync> =
        Box::new(GlobalDeviceBudget::new(logging, limit_bytes));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(limit_bytes as u64),
        Arc::clone(&runtime),
    ));
    let provider =
        CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory)).ok()?;
    Some(Fix {
        device,
        runtime,
        memory,
        provider,
        pool,
    })
}

#[inline]
fn lcg_next(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    *state
}

fn dedup_pairs(mut v: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    v.sort();
    v.dedup();
    v
}

/// Uniform Erdős-Rényi over `[0, key_range)` — same generator
/// as `bench_uniform`. Drives the score below threshold.
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

/// Super-hub: half the rows have the second column == hub_y.
fn superhub_pairs_xy(seed: u64, rows: u32, key_range: u32, hub_y: u32) -> Vec<(u32, u32)> {
    let mut state = seed;
    (0..rows)
        .map(|i| {
            let a = (lcg_next(&mut state) % key_range as u64) as u32;
            let b = if i % 2 == 0 {
                hub_y
            } else {
                (lcg_next(&mut state) % key_range as u64) as u32
            };
            (a, b)
        })
        .collect()
}

/// Super-hub: half the rows have the first column == hub_first.
fn superhub_pairs_first(seed: u64, rows: u32, key_range: u32, hub_first: u32) -> Vec<(u32, u32)> {
    let mut state = seed;
    (0..rows)
        .map(|i| {
            let a = if i % 2 == 0 {
                hub_first
            } else {
                (lcg_next(&mut state) % key_range as u64) as u32
            };
            let b = (lcg_next(&mut state) % key_range as u64) as u32;
            (a, b)
        })
        .collect()
}

/// Disjoint key range fixture (matches bench's `make_empty`).
fn disjoint_pairs(seed: u64, rows: u32, base: u32, range: u32) -> Vec<(u32, u32)> {
    let mut state = seed;
    (0..rows)
        .map(|_| {
            let a = base + (lcg_next(&mut state) % range as u64) as u32;
            let b = base + (lcg_next(&mut state) % range as u64) as u32;
            (a, b)
        })
        .collect()
}

fn upload_binary_u32(memory: &Arc<GpuMemoryManager>, rows: &[(u32, u32)]) -> CudaBuffer {
    let n = rows.len() as u32;
    let bpc = (n as usize) * 4;
    let mut col0 = memory.alloc::<u8>(bpc).expect("alloc c0");
    let mut col1 = memory.alloc::<u8>(bpc).expect("alloc c1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc nr");
    let device = memory.device().inner();
    if n > 0 {
        let c0: Vec<u8> = rows.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
        let c1: Vec<u8> = rows.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
        device.htod_sync_copy_into(&c0, &mut col0).expect("htod c0");
        device.htod_sync_copy_into(&c1, &mut col1).expect("htod c1");
    }
    device
        .htod_sync_copy_into(&[n], &mut d_num_rows)
        .expect("htod nr");
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

fn upload_binary_u64(memory: &Arc<GpuMemoryManager>, rows: &[(u64, u64)]) -> CudaBuffer {
    let n = rows.len() as u32;
    let bpc = (n as usize) * 8;
    let mut col0 = memory.alloc::<u8>(bpc).expect("alloc c0");
    let mut col1 = memory.alloc::<u8>(bpc).expect("alloc c1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc nr");
    let device = memory.device().inner();
    if n > 0 {
        let c0: Vec<u8> = rows.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
        let c1: Vec<u8> = rows.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
        device.htod_sync_copy_into(&c0, &mut col0).expect("htod c0");
        device.htod_sync_copy_into(&c1, &mut col1).expect("htod c1");
    }
    device
        .htod_sync_copy_into(&[n], &mut d_num_rows)
        .expect("htod nr");
    let schema = Schema::new(vec![
        ("col0".to_string(), ScalarType::U64),
        ("col1".to_string(), ScalarType::U64),
    ]);
    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into()],
        n as u64,
        d_num_rows,
        schema,
        n,
    )
}

// =================================================================
// dtoh_small_metadata_untracked
// =================================================================

#[test]
fn dtoh_small_metadata_untracked_reads_192_u32_histogram_payload() {
    // 192 u32s = 768 bytes — the actual classifier histogram size.
    // Lock that the helper can read at least the classifier's
    // working set in a single call.
    let Some(fix) = make_fix() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let host_in: Vec<u32> = (0..192u32).collect();
    let mut device_buf = fix
        .memory
        .alloc::<u32>(host_in.len())
        .expect("alloc device");
    fix.device
        .inner()
        .htod_sync_copy_into(&host_in, &mut device_buf)
        .expect("htod");
    let host_out = fix
        .provider
        .dtoh_small_metadata_untracked::<u32>(&device_buf, host_in.len())
        .expect("dtoh_small_metadata_untracked must accept 768-byte read");
    assert_eq!(host_out, host_in);
}

#[test]
fn dtoh_small_metadata_untracked_rejects_above_4096_bytes() {
    // Cap is the contract — the helper is metadata-only, NOT a
    // general vector D2H escape hatch. Anything > 4096 bytes
    // must fail with a clear error so future callers don't
    // sneak a column-sized download through this path.
    let Some(fix) = make_fix() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    // 1025 u32s = 4100 bytes > 4096.
    let n = 1025usize;
    let device_buf = fix.memory.alloc::<u32>(n).expect("alloc device");
    let result = fix
        .provider
        .dtoh_small_metadata_untracked::<u32>(&device_buf, n);
    let err = match result {
        Ok(_) => panic!("dtoh_small_metadata_untracked must reject > 4096 bytes"),
        Err(e) => e,
    };
    let msg = format!("{:?}", err);
    assert!(
        msg.contains("4096") || msg.contains("metadata") || msg.contains("cap"),
        "error must mention the size cap; got: {}",
        msg
    );
}

#[test]
fn dtoh_small_metadata_untracked_no_violation_under_strict_d2h_gate() {
    // The classifier path runs the histogram D2H under the strict
    // deterministic-D2H gate. The new helper must be whitelisted
    // (mirrors `dtoh_scalar_untracked`'s gate semantics) so the
    // gate is not tripped.
    let Some(fix) = make_fix() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let host_in: Vec<u32> = vec![42; 192];
    let mut device_buf = fix.memory.alloc::<u32>(192).expect("alloc");
    fix.device
        .inner()
        .htod_sync_copy_into(&host_in, &mut device_buf)
        .expect("htod");
    fix.provider.reset_deterministic_d2h_violations();
    fix.provider.enable_strict_deterministic_d2h();
    let result = fix
        .provider
        .dtoh_small_metadata_untracked::<u32>(&device_buf, 192);
    fix.provider.disable_strict_deterministic_d2h();
    let v = result.expect("must succeed under strict gate");
    assert_eq!(v, host_in);
    let violations = fix.provider.deterministic_d2h_violation_count();
    assert_eq!(
        violations, 0,
        "dtoh_small_metadata_untracked must NOT trip the strict deterministic-D2H gate; got {violations} violations"
    );
}

// =================================================================
// wcoj_triangle_skew_score_u32 — the classifier provider entry
// =================================================================

const SKEW_THRESHOLD: f64 = 0.10;

fn legacy_provider() -> Option<(Arc<CudaDevice>, Arc<GpuMemoryManager>, CudaKernelProvider)> {
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let memory = Arc::new(GpuMemoryManager::new(
        Arc::clone(&device),
        MemoryBudget::with_limit(64 * 1024 * 1024),
    ));
    let provider = CudaKernelProvider::new(Arc::clone(&device), Arc::clone(&memory)).ok()?;
    Some((device, memory, provider))
}

#[test]
fn skew_score_u32_legacy_manager_returns_none() {
    // Classifier is an optimization, not correctness — when the
    // manager has no runtime (and thus the recorded histogram
    // pipeline can't execute), the entry must return Ok(None)
    // so callers can silently fall back to binary join.
    let Some((_device, memory, provider)) = legacy_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };
    let buf = upload_binary_u32(&memory, &[(1, 2), (3, 4)]);
    let result = provider.wcoj_triangle_skew_score_u32(&buf, &buf, &buf, StreamId::DEFAULT);
    match result {
        Ok(None) => {}
        Ok(Some(s)) => panic!("legacy manager must yield Ok(None); got Ok(Some({s}))"),
        Err(e) => panic!("legacy manager must yield Ok(None); got Err({e:?})"),
    }
}

#[test]
fn skew_score_u32_uniform_below_threshold() {
    let Some(fix) = make_fix() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let rows = 5_000u32;
    let kr = 1000u32;
    let e1 = dedup_pairs(uniform_pairs(1, rows, kr));
    let e2 = dedup_pairs(uniform_pairs(2, rows, kr));
    let e3 = dedup_pairs(uniform_pairs(3, rows, kr));
    let buf_e1 = upload_binary_u32(&fix.memory, &e1);
    let buf_e2 = upload_binary_u32(&fix.memory, &e2);
    let buf_e3 = upload_binary_u32(&fix.memory, &e3);
    let stream = fix.pool.acquire().expect("stream");
    let score = fix
        .provider
        .wcoj_triangle_skew_score_u32(&buf_e1, &buf_e2, &buf_e3, stream)
        .expect("classifier OK")
        .expect("classifier returned Some");
    eprintln!("uniform u32 score = {score:.4}");
    assert!(
        score < SKEW_THRESHOLD,
        "uniform fixture must score below {SKEW_THRESHOLD} threshold; got {score:.4}"
    );
    // Probe predicts ~0.020 — assert generously to allow for
    // bucket-count, mixer, and small-fixture noise.
    assert!(
        score < 0.05,
        "uniform score should be near 0.020; got {score:.4}"
    );
}

#[test]
fn skew_score_u32_superhub_above_threshold() {
    let Some(fix) = make_fix() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let rows = 5_000u32;
    let kr = 1000u32;
    let hub_y = 7u32;
    let hub_x = 13u32;
    let e1 = dedup_pairs(superhub_pairs_xy(101, rows, kr, hub_y));
    let e2 = dedup_pairs(superhub_pairs_first(202, rows, kr, hub_y));
    let e3 = dedup_pairs(superhub_pairs_first(303, rows, kr, hub_x));
    let buf_e1 = upload_binary_u32(&fix.memory, &e1);
    let buf_e2 = upload_binary_u32(&fix.memory, &e2);
    let buf_e3 = upload_binary_u32(&fix.memory, &e3);
    let stream = fix.pool.acquire().expect("stream");
    let score = fix
        .provider
        .wcoj_triangle_skew_score_u32(&buf_e1, &buf_e2, &buf_e3, stream)
        .expect("classifier OK")
        .expect("classifier returned Some");
    eprintln!("superhub u32 score = {score:.4}");
    assert!(
        score >= SKEW_THRESHOLD,
        "superhub fixture must score >= {SKEW_THRESHOLD} threshold; got {score:.4}"
    );
    assert!(
        score >= 0.15,
        "superhub score should be ~0.18 (probe-locked); got {score:.4}"
    );
}

#[test]
fn skew_score_u32_empty_below_threshold() {
    // Empty here means "three relations whose join produces no
    // triangles" — same as the bench's empty fixture (disjoint
    // key ranges). The COLUMN distributions are still uniform-
    // ish; classifier must keep them below threshold so binary
    // join's fast 1-2 ms early-exit handles them.
    let Some(fix) = make_fix() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let rows = 5_000u32;
    let range = 1000u32;
    let e1 = dedup_pairs(disjoint_pairs(11, rows, 0, range));
    let e2 = dedup_pairs(disjoint_pairs(22, rows, 1_000_000, range));
    let e3 = dedup_pairs(disjoint_pairs(33, rows, 2_000_000, range));
    let buf_e1 = upload_binary_u32(&fix.memory, &e1);
    let buf_e2 = upload_binary_u32(&fix.memory, &e2);
    let buf_e3 = upload_binary_u32(&fix.memory, &e3);
    let stream = fix.pool.acquire().expect("stream");
    let score = fix
        .provider
        .wcoj_triangle_skew_score_u32(&buf_e1, &buf_e2, &buf_e3, stream)
        .expect("classifier OK")
        .expect("classifier returned Some");
    eprintln!("empty u32 score = {score:.4}");
    assert!(
        score < SKEW_THRESHOLD,
        "empty fixture must score below {SKEW_THRESHOLD}; got {score:.4}"
    );
}

#[test]
fn skew_score_u32_empty_input_returns_none() {
    // Zero rows on any input → classifier can't compute (division
    // by zero on the score formula). Return Ok(None); caller falls
    // back to binary join (which itself handles empty trivially).
    let Some(fix) = make_fix() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let buf_empty = upload_binary_u32(&fix.memory, &[]);
    let buf_some = upload_binary_u32(&fix.memory, &[(1, 2), (3, 4)]);
    let stream = fix.pool.acquire().expect("stream");
    let result = fix
        .provider
        .wcoj_triangle_skew_score_u32(&buf_empty, &buf_some, &buf_some, stream)
        .expect("must not error on empty input");
    assert!(
        result.is_none(),
        "any zero-row input must yield Ok(None); got {result:?}"
    );
}

#[test]
fn skew_score_u32_no_d2h_violation_under_strict_gate() {
    // The classifier's ONLY device→host transfer is the 768-byte
    // histogram via `dtoh_small_metadata_untracked`. Under the
    // strict deterministic-D2H gate, this must succeed with zero
    // violations. Any future regression that routes the histogram
    // through `download_column_*` or another tracked path would
    // trip the gate and fail this test.
    let Some(fix) = make_fix() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let rows = 1_000u32;
    let e1 = dedup_pairs(superhub_pairs_xy(101, rows, 200, 7));
    let e2 = dedup_pairs(superhub_pairs_first(202, rows, 200, 7));
    let e3 = dedup_pairs(superhub_pairs_first(303, rows, 200, 13));
    let buf_e1 = upload_binary_u32(&fix.memory, &e1);
    let buf_e2 = upload_binary_u32(&fix.memory, &e2);
    let buf_e3 = upload_binary_u32(&fix.memory, &e3);
    fix.provider.reset_deterministic_d2h_violations();
    fix.provider.enable_strict_deterministic_d2h();
    let stream = fix.pool.acquire().expect("stream");
    let result = fix
        .provider
        .wcoj_triangle_skew_score_u32(&buf_e1, &buf_e2, &buf_e3, stream);
    fix.provider.disable_strict_deterministic_d2h();
    let _ = result.expect("classifier must succeed under strict gate");
    let v = fix.provider.deterministic_d2h_violation_count();
    assert_eq!(
        v, 0,
        "skew classifier must not trigger deterministic-D2H gate violations; got {v}"
    );
}

// =================================================================
// wcoj_triangle_skew_score_u64 — parallel path
// =================================================================

#[test]
fn skew_score_u64_uniform_below_threshold() {
    let Some(fix) = make_fix() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let rows = 5_000u32;
    let kr = 1000u32;
    let to_u64 = |v: Vec<(u32, u32)>| -> Vec<(u64, u64)> {
        v.into_iter().map(|(a, b)| (a as u64, b as u64)).collect()
    };
    let e1 = to_u64(dedup_pairs(uniform_pairs(1, rows, kr)));
    let e2 = to_u64(dedup_pairs(uniform_pairs(2, rows, kr)));
    let e3 = to_u64(dedup_pairs(uniform_pairs(3, rows, kr)));
    let buf_e1 = upload_binary_u64(&fix.memory, &e1);
    let buf_e2 = upload_binary_u64(&fix.memory, &e2);
    let buf_e3 = upload_binary_u64(&fix.memory, &e3);
    let stream = fix.pool.acquire().expect("stream");
    let score = fix
        .provider
        .wcoj_triangle_skew_score_u64(&buf_e1, &buf_e2, &buf_e3, stream)
        .expect("classifier OK")
        .expect("classifier returned Some");
    eprintln!("uniform u64 score = {score:.4}");
    assert!(
        score < SKEW_THRESHOLD,
        "uniform u64 must score below {SKEW_THRESHOLD}; got {score:.4}"
    );
}

#[test]
fn skew_score_u64_superhub_above_threshold() {
    let Some(fix) = make_fix() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let rows = 5_000u32;
    let kr = 1000u32;
    let to_u64 = |v: Vec<(u32, u32)>| -> Vec<(u64, u64)> {
        v.into_iter().map(|(a, b)| (a as u64, b as u64)).collect()
    };
    let e1 = to_u64(dedup_pairs(superhub_pairs_xy(101, rows, kr, 7)));
    let e2 = to_u64(dedup_pairs(superhub_pairs_first(202, rows, kr, 7)));
    let e3 = to_u64(dedup_pairs(superhub_pairs_first(303, rows, kr, 13)));
    let buf_e1 = upload_binary_u64(&fix.memory, &e1);
    let buf_e2 = upload_binary_u64(&fix.memory, &e2);
    let buf_e3 = upload_binary_u64(&fix.memory, &e3);
    let stream = fix.pool.acquire().expect("stream");
    let score = fix
        .provider
        .wcoj_triangle_skew_score_u64(&buf_e1, &buf_e2, &buf_e3, stream)
        .expect("classifier OK")
        .expect("classifier returned Some");
    eprintln!("superhub u64 score = {score:.4}");
    assert!(
        score >= SKEW_THRESHOLD,
        "superhub u64 must score >= {SKEW_THRESHOLD}; got {score:.4}"
    );
    assert!(
        score >= 0.15,
        "superhub u64 score should be ~0.18; got {score:.4}"
    );
}

#[allow(dead_code)]
fn _silence_unused_xlog_error_import() -> XlogError {
    XlogError::Kernel("never called".to_string())
}
