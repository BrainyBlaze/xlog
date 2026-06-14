// crates/xlog-integration/tests/test_sort_merge_dispatch.rs
//! Sort-merge join — operator-level provider parity tests.
//!
//! The sort-merge operator is implemented at the provider layer
//! (`provider.sort_merge_join_v2_inner_u32_1key`) but is NOT wired
//! into the executor's dispatch decision tree. The production
//! benchmark rejected the original executor-dispatch assumptions:
//! sort-merge did not win by at least 2× against hash join on any
//! tested cell, and nested-loop join won by 1.25×–2.46× across the
//! shared sort-merge/nested-loop eligibility envelope. The resulting
//! rewrite removed executor dispatch and kept this suite at the
//! provider/operator layer.
//!
//! The four tests in this file verify the operator
//! surface directly via `provider.sort_merge_join_v2_inner_u32_1key`
//! and `provider.is_sorted_ascending_u32`:
//!   * Sorted 100-row 1-key U32 fixture parity vs hash.
//!   * Sorted Symbol-typed parity vs hash.
//!   * Duplicate-key 250 × 4 → 4000 output rows + tuple
//!     distinctness + parity vs hash (run-length emit path).
//!   * Empty-input layered short-circuit
//!     (sortedness probe `n < 2 → Ok(true)` + operator empty
//!     fast path + parity vs hash empty fast path).
//!
//! Earlier dispatch-shape tests for unsorted fall-through,
//! above-threshold joins, multi-column joins, and semi joins are
//! retired: asserting `sort_merge_dispatch_count == 0` on fall-through
//! fixtures is vacuous in the absence of an executor dispatch path.
//! The nested-loop dispatch suite at `test_nested_loop_dispatch.rs`
//! already covers the same fixture shapes for the production-routing
//! guard.

use std::collections::BTreeSet;
use std::sync::Arc;

use cudarc::driver::sys;
use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::CudaBuffer;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager, JoinType};

// ---------------------------------------------------------------
// Fixture helpers (provider-level — no executor scaffolding).
// ---------------------------------------------------------------

struct DiscardSink;
impl LoggingSink for DiscardSink {
    fn emit(&self, _record: LogRecord) -> Result<(), SinkError> {
        Ok(())
    }
}

#[allow(dead_code)]
struct ProviderFixture {
    device: Arc<CudaDevice>,
    runtime: Arc<XlogDeviceRuntime>,
    memory: Arc<GpuMemoryManager>,
    provider: Arc<CudaKernelProvider>,
    pool: Arc<StreamPool>,
}

fn make_provider_fixture() -> Option<ProviderFixture> {
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
    let async_resource: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(
        AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool)),
    );
    let logging: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(LoggingResource::new(
        async_resource,
        Arc::new(DiscardSink) as Arc<dyn LoggingSink>,
    ));
    let budget: Box<dyn DeviceMemoryResource + Send + Sync> =
        Box::new(GlobalDeviceBudget::new(logging, 64 * 1024 * 1024));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(64 * 1024 * 1024),
        Arc::clone(&runtime),
    ));
    let provider =
        Arc::new(CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory)).ok()?);
    Some(ProviderFixture {
        device,
        runtime,
        memory,
        provider,
        pool,
    })
}

/// Upload a 2-col U32 buffer (column 0 = key, column 1 = payload).
fn upload_binary_u32(memory: &Arc<GpuMemoryManager>, rows: &[(u32, u32)]) -> CudaBuffer {
    let n = rows.len() as u32;
    let bytes_per_col = (n as usize) * std::mem::size_of::<u32>();
    let mut col0 = memory.alloc::<u8>(bytes_per_col).expect("alloc col0");
    let mut col1 = memory.alloc::<u8>(bytes_per_col).expect("alloc col1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let device = memory.device().inner();
    if n > 0 {
        let col0_bytes: Vec<u8> = rows.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
        let col1_bytes: Vec<u8> = rows.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
        device
            .htod_sync_copy_into(&col0_bytes, &mut col0)
            .expect("htod col0");
        device
            .htod_sync_copy_into(&col1_bytes, &mut col1)
            .expect("htod col1");
    }
    device
        .htod_sync_copy_into(&[n], &mut d_num_rows)
        .expect("htod d_num_rows");
    let schema = Schema::new(vec![
        ("k".to_string(), ScalarType::U32),
        ("p".to_string(), ScalarType::U32),
    ]);
    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into()],
        n as u64,
        d_num_rows,
        schema,
        n,
    )
}

/// Upload a 2-col Symbol-keyed buffer (col 0 = `ScalarType::Symbol`,
/// col 1 = `ScalarType::U32`). Byte layout identical to
/// `upload_binary_u32` — Symbol is u32 at the byte level. Only
/// the schema's column-type label differs.
fn upload_symbol_keyed(memory: &Arc<GpuMemoryManager>, rows: &[(u32, u32)]) -> CudaBuffer {
    let n = rows.len() as u32;
    let bytes_per_col = (n as usize) * std::mem::size_of::<u32>();
    let mut col0 = memory.alloc::<u8>(bytes_per_col).expect("alloc col0");
    let mut col1 = memory.alloc::<u8>(bytes_per_col).expect("alloc col1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let device = memory.device().inner();
    if n > 0 {
        let col0_bytes: Vec<u8> = rows.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
        let col1_bytes: Vec<u8> = rows.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
        device
            .htod_sync_copy_into(&col0_bytes, &mut col0)
            .expect("htod col0");
        device
            .htod_sync_copy_into(&col1_bytes, &mut col1)
            .expect("htod col1");
    }
    device
        .htod_sync_copy_into(&[n], &mut d_num_rows)
        .expect("htod d_num_rows");
    let schema = Schema::new(vec![
        ("sym".to_string(), ScalarType::Symbol),
        ("p".to_string(), ScalarType::U32),
    ]);
    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into()],
        n as u64,
        d_num_rows,
        schema,
        n,
    )
}

/// Download a 4-col U32 buffer — the `combine_schemas` output of
/// 2-col left ⋈ 2-col right is `[left_k, left_p, right_k, right_p]`.
fn download_quads(buf: &CudaBuffer) -> Vec<(u32, u32, u32, u32)> {
    let n = match buf.cached_row_count() {
        Some(c) => c as usize,
        None => {
            let mut count_host = [0u32; 1];
            unsafe {
                sys::cuMemcpyDtoH_v2(
                    count_host.as_mut_ptr() as *mut _,
                    *buf.num_rows_device().device_ptr(),
                    std::mem::size_of::<u32>(),
                );
            }
            count_host[0] as usize
        }
    };
    if n == 0 {
        return Vec::new();
    }
    assert_eq!(buf.arity(), 4, "download_quads expects arity 4");
    let mut cols = [
        vec![0u8; n * 4],
        vec![0u8; n * 4],
        vec![0u8; n * 4],
        vec![0u8; n * 4],
    ];
    for (i, col_bytes) in cols.iter_mut().enumerate() {
        unsafe {
            sys::cuMemcpyDtoH_v2(
                col_bytes.as_mut_ptr() as *mut _,
                *buf.column(i).unwrap().device_ptr(),
                col_bytes.len(),
            );
        }
    }
    (0..n)
        .map(|i| {
            let off = i * 4;
            (
                u32::from_le_bytes(cols[0][off..off + 4].try_into().unwrap()),
                u32::from_le_bytes(cols[1][off..off + 4].try_into().unwrap()),
                u32::from_le_bytes(cols[2][off..off + 4].try_into().unwrap()),
                u32::from_le_bytes(cols[3][off..off + 4].try_into().unwrap()),
            )
        })
        .collect()
}

// ---------------------------------------------------------------
// Sorted-key operator parity.
//
// `provider.sort_merge_join_v2_inner_u32_1key` on a sorted
// 100-row 1-key U32 fixture produces row-set parity vs
// `provider.hash_join_v2 Inner`. Pure operator parity — no
// executor, no dispatch counter, no selectivity feedback (those
// were original executor-dispatch concerns superseded by the
// provider-only rewrite).
// ---------------------------------------------------------------

#[test]
fn sort_merge_operator_parity_sorted_unique_u32() {
    let Some(fix) = make_provider_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    // Fixture: 100 unique-keyed rows on each side, sorted
    // ascending. All 100 keys match → 100 join output rows.
    let left_rows: Vec<(u32, u32)> = (0..100u32).map(|i| (i, 1000 + i)).collect();
    let right_rows: Vec<(u32, u32)> = (0..100u32).map(|i| (i, 2000 + i)).collect();

    let left_buf = upload_binary_u32(&fix.memory, &left_rows);
    let right_buf = upload_binary_u32(&fix.memory, &right_rows);

    // Hash reference (4-col [lk, lp, rk, rp]).
    let hash_buf = fix
        .provider
        .hash_join_v2(&left_buf, &right_buf, &[0], &[0], JoinType::Inner)
        .expect("hash_join_v2 reference");
    let hash_set: BTreeSet<(u32, u32, u32, u32)> = download_quads(&hash_buf).into_iter().collect();
    assert_eq!(
        hash_set.len(),
        100,
        "hash reference should produce exactly 100 matched rows"
    );

    // Sort-merge operator (4-col [lk, lp, rk, rp]).
    let sm_buf = fix
        .provider
        .sort_merge_join_v2_inner_u32_1key(&left_buf, &right_buf, 0, 0)
        .expect("sort_merge_join_v2_inner_u32_1key");
    let sm_set: BTreeSet<(u32, u32, u32, u32)> = download_quads(&sm_buf).into_iter().collect();
    assert_eq!(
        sm_set.len(),
        100,
        "sort-merge operator should produce exactly 100 matched rows"
    );

    assert_eq!(
        sm_set, hash_set,
        "sort-merge operator row set must equal hash_join_v2 reference"
    );
}

// ---------------------------------------------------------------
// Symbol-typed operator parity.
//
// Same shape as the sorted-key parity test but with `ScalarType::Symbol` on the key
// column. Symbol is byte-identical to U32 at the kernel level,
// so the same operator applies.
// ---------------------------------------------------------------

#[test]
fn sort_merge_operator_parity_sorted_unique_symbol() {
    let Some(fix) = make_provider_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let left_rows: Vec<(u32, u32)> = (0..100u32).map(|i| (i, 1000 + i)).collect();
    let right_rows: Vec<(u32, u32)> = (0..100u32).map(|i| (i, 2000 + i)).collect();

    let left_buf = upload_symbol_keyed(&fix.memory, &left_rows);
    let right_buf = upload_symbol_keyed(&fix.memory, &right_rows);

    let hash_buf = fix
        .provider
        .hash_join_v2(&left_buf, &right_buf, &[0], &[0], JoinType::Inner)
        .expect("hash_join_v2 reference (Symbol)");
    let hash_set: BTreeSet<(u32, u32, u32, u32)> = download_quads(&hash_buf).into_iter().collect();
    assert_eq!(hash_set.len(), 100);

    let sm_buf = fix
        .provider
        .sort_merge_join_v2_inner_u32_1key(&left_buf, &right_buf, 0, 0)
        .expect("sort_merge_join_v2_inner_u32_1key (Symbol)");
    let sm_set: BTreeSet<(u32, u32, u32, u32)> = download_quads(&sm_buf).into_iter().collect();

    assert_eq!(
        sm_set, hash_set,
        "sort-merge operator on Symbol-typed buffers must equal hash reference"
    );
}

// ---------------------------------------------------------------
// Duplicate-key operator parity (run-length emit path).
//
// 250 unique keys × 4 dups each side → 1000 rows each side →
// 4000 output rows (250 keys × 4×4 per-key matches). Mirrors
// the duplicate-heavy benchmark shape. Exercises the kernel's per-thread
// `lower_bound`/`upper_bound` run-length emit path that the
// unique-key parity test does not cover.
// ---------------------------------------------------------------

#[test]
fn sort_merge_operator_parity_duplicate_key() {
    let Some(fix) = make_provider_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    // 1000 rows each side with 250 unique keys × 4 dups, sorted
    // ascending. Payload = running counter so each duplicated row
    // is distinct in (k, p) space (avoids parity-check ambiguity
    // on the multiplicity oracle).
    let left_rows: Vec<(u32, u32)> = (0..1000u32).map(|i| (i / 4, 1000 + i)).collect();
    let right_rows: Vec<(u32, u32)> = (0..1000u32).map(|i| (i / 4, 2000 + i)).collect();

    let left_buf = upload_binary_u32(&fix.memory, &left_rows);
    let right_buf = upload_binary_u32(&fix.memory, &right_rows);

    let hash_buf = fix
        .provider
        .hash_join_v2(&left_buf, &right_buf, &[0], &[0], JoinType::Inner)
        .expect("hash_join_v2 reference (duplicate-key)");
    let hash_set: BTreeSet<(u32, u32, u32, u32)> = download_quads(&hash_buf).into_iter().collect();
    assert_eq!(
        hash_set.len(),
        4000,
        "hash reference should produce exactly 4000 matched rows \
         (250 keys × 4×4 per-key matches)"
    );

    let sm_buf = fix
        .provider
        .sort_merge_join_v2_inner_u32_1key(&left_buf, &right_buf, 0, 0)
        .expect("sort_merge_join_v2_inner_u32_1key (duplicate-key)");
    let sm_quads = download_quads(&sm_buf);
    assert_eq!(
        sm_quads.len(),
        4000,
        "sort-merge operator must emit 4000 rows on duplicate-key fixture \
         (proves run-length kernel emits the right multiplicity); got {}",
        sm_quads.len()
    );
    let sm_set: BTreeSet<(u32, u32, u32, u32)> = sm_quads.into_iter().collect();
    assert_eq!(
        sm_set.len(),
        4000,
        "all 4000 (lk, lp, rk, rp) tuples must be distinct \
         (running-counter payload design); got {} distinct tuples",
        sm_set.len()
    );
    assert_eq!(
        sm_set, hash_set,
        "sort-merge operator row set must equal hash reference on duplicate-key fixture"
    );
}

// ---------------------------------------------------------------
// Empty-input layered short-circuit.
//
// Two subcases (`num_left == 0`, `num_right == 0`). Each verifies
// the layered short-circuit contract end-to-end at the provider layer:
//   1. `provider.is_sorted_ascending_u32` returns `Ok(true)` on
//      `n == 0` via its `n < 2` internal short-circuit (BEFORE
//      allocation/launch — the kernel grid `(0+255)/256 = 0`
//      hazard is avoided).
//   2. `provider.sort_merge_join_v2_inner_u32_1key` produces
//      an empty `combine_schemas` buffer via its own empty
//      fast path. No kernel-launch crash.
//   3. Row-set parity vs `provider.hash_join_v2` (which has its
//      own empty fast path at `relational.rs:3165-3170`).
// ---------------------------------------------------------------

#[test]
fn sort_merge_operator_empty_input_layered_short_circuit() {
    let Some(fix) = make_provider_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    // ===========================================================
    // Subcase: empty L, populated sorted R.
    // ===========================================================
    {
        let left_rows: Vec<(u32, u32)> = Vec::new();
        let right_rows: Vec<(u32, u32)> = (0..50u32).map(|i| (i, 2000 + i)).collect();
        let left_buf = upload_binary_u32(&fix.memory, &left_rows);
        let right_buf = upload_binary_u32(&fix.memory, &right_rows);

        // Layer 1: sortedness probe `n < 2 → Ok(true)` short-circuit.
        // For empty L (n = 0) and sorted R (n = 50), both return Ok(true).
        let left_sorted = fix
            .provider
            .is_sorted_ascending_u32(&left_buf, 0)
            .expect("is_sorted_ascending_u32 must not error on empty L");
        let right_sorted = fix
            .provider
            .is_sorted_ascending_u32(&right_buf, 0)
            .expect("is_sorted_ascending_u32 must not error on sorted R");
        assert!(
            left_sorted,
            "empty-left layer 1: empty L must short-circuit to Ok(true) (n < 2 fast path)"
        );
        assert!(
            right_sorted,
            "empty-left layer 1: sorted R must return Ok(true)"
        );

        // Layer 2 + 3: operator empty fast path + parity.
        let sm_buf = fix
            .provider
            .sort_merge_join_v2_inner_u32_1key(&left_buf, &right_buf, 0, 0)
            .expect("empty-left layer 2: sort-merge operator must not crash on empty L");
        let sm_set: BTreeSet<(u32, u32, u32, u32)> = download_quads(&sm_buf).into_iter().collect();
        assert!(
            sm_set.is_empty(),
            "empty-left layer 2: sort-merge operator must produce empty output on empty L"
        );

        let hash_buf = fix
            .provider
            .hash_join_v2(&left_buf, &right_buf, &[0], &[0], JoinType::Inner)
            .expect("empty-left layer 3: hash reference on empty L");
        let hash_set: BTreeSet<(u32, u32, u32, u32)> =
            download_quads(&hash_buf).into_iter().collect();
        assert!(hash_set.is_empty());
        assert_eq!(
            sm_set, hash_set,
            "empty-left layer 3: sort-merge row set must equal hash empty-fast-path output (both empty)"
        );
    }

    // ===========================================================
    // Subcase: populated sorted L, empty R.
    // ===========================================================
    {
        let left_rows: Vec<(u32, u32)> = (0..50u32).map(|i| (i, 1000 + i)).collect();
        let right_rows: Vec<(u32, u32)> = Vec::new();
        let left_buf = upload_binary_u32(&fix.memory, &left_rows);
        let right_buf = upload_binary_u32(&fix.memory, &right_rows);

        let left_sorted = fix
            .provider
            .is_sorted_ascending_u32(&left_buf, 0)
            .expect("is_sorted_ascending_u32 must not error on sorted L");
        let right_sorted = fix
            .provider
            .is_sorted_ascending_u32(&right_buf, 0)
            .expect("is_sorted_ascending_u32 must not error on empty R");
        assert!(
            left_sorted,
            "empty-right layer 1: sorted L must return Ok(true)"
        );
        assert!(
            right_sorted,
            "empty-right layer 1: empty R must short-circuit to Ok(true) (n < 2 fast path)"
        );

        let sm_buf = fix
            .provider
            .sort_merge_join_v2_inner_u32_1key(&left_buf, &right_buf, 0, 0)
            .expect("empty-right layer 2: sort-merge operator must not crash on empty R");
        let sm_set: BTreeSet<(u32, u32, u32, u32)> = download_quads(&sm_buf).into_iter().collect();
        assert!(
            sm_set.is_empty(),
            "empty-right layer 2: sort-merge operator must produce empty output on empty R"
        );

        let hash_buf = fix
            .provider
            .hash_join_v2(&left_buf, &right_buf, &[0], &[0], JoinType::Inner)
            .expect("empty-right layer 3: hash reference on empty R");
        let hash_set: BTreeSet<(u32, u32, u32, u32)> =
            download_quads(&hash_buf).into_iter().collect();
        assert!(hash_set.is_empty());
        assert_eq!(
            sm_set, hash_set,
            "empty-right layer 3: sort-merge row set must equal hash empty-fast-path output (both empty)"
        );
    }
}
