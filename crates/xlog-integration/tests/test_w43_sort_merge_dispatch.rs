// crates/xlog-integration/tests/test_w43_sort_merge_dispatch.rs
//! W4.3 sort-merge join dispatch + parity certs.
//!
//! Cert A — pre-sorted small × small dispatch routes through the
//! W4.3 `sort_merge_join_v2_inner_u32_1key` provider entry point,
//! produces a row-set bit-identical to `hash_join_v2`'s reference
//! output, AND wires `record_join_result` feedback into
//! `StatsManager` (the same D6 invariant the W4.2 cert pins for
//! the nested-loop path).
//!
//! Subsequent certs (B / C / D / D' / E / F / G) will land in
//! follow-up commits per the W4.3 plan iteration-4 Steps 7 / 8 /
//! 9 / 10.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use cudarc::driver::sys;
use xlog_core::{MemoryBudget, RuntimeConfig, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::CudaBuffer;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager, JoinType};
use xlog_logic::Compiler;
use xlog_runtime::Executor;

// ---------------------------------------------------------------
// Fixture helpers (mirrors W4.2 cert pattern at
// crates/xlog-integration/tests/test_w42_nested_loop_dispatch.rs;
// duplicated here per the existing tests/ convention of
// self-contained per-file helpers).
// ---------------------------------------------------------------

struct DiscardSink;
impl LoggingSink for DiscardSink {
    fn emit(&self, _record: LogRecord) -> Result<(), SinkError> {
        Ok(())
    }
}

#[allow(dead_code)]
struct RuntimeBackedFixture {
    device: Arc<CudaDevice>,
    runtime: Arc<XlogDeviceRuntime>,
    memory: Arc<GpuMemoryManager>,
    provider: Arc<CudaKernelProvider>,
    pool: Arc<StreamPool>,
}

fn make_runtime_backed_fixture() -> Option<RuntimeBackedFixture> {
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
    Some(RuntimeBackedFixture {
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

/// Download a 3-col U32 buffer (used for `result(K, A, B)`).
fn download_triples(buf: &CudaBuffer) -> Vec<(u32, u32, u32)> {
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
    assert_eq!(buf.arity(), 3, "download_triples expects arity 3");
    let mut col0_bytes = vec![0u8; n * 4];
    let mut col1_bytes = vec![0u8; n * 4];
    let mut col2_bytes = vec![0u8; n * 4];
    unsafe {
        sys::cuMemcpyDtoH_v2(
            col0_bytes.as_mut_ptr() as *mut _,
            *buf.column(0).unwrap().device_ptr(),
            col0_bytes.len(),
        );
        sys::cuMemcpyDtoH_v2(
            col1_bytes.as_mut_ptr() as *mut _,
            *buf.column(1).unwrap().device_ptr(),
            col1_bytes.len(),
        );
        sys::cuMemcpyDtoH_v2(
            col2_bytes.as_mut_ptr() as *mut _,
            *buf.column(2).unwrap().device_ptr(),
            col2_bytes.len(),
        );
    }
    (0..n)
        .map(|i| {
            let off = i * 4;
            (
                u32::from_le_bytes(col0_bytes[off..off + 4].try_into().unwrap()),
                u32::from_le_bytes(col1_bytes[off..off + 4].try_into().unwrap()),
                u32::from_le_bytes(col2_bytes[off..off + 4].try_into().unwrap()),
            )
        })
        .collect()
}

/// Download a 4-col U32 buffer (used for the direct-provider hash
/// reference, which produces `[left_k, left_p, right_k, right_p]`
/// per `combine_schemas`).
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
// Cert A — pre-sorted small × small dispatches sort-merge,
// matches hash, and records join feedback.
// ---------------------------------------------------------------

/// Datalog program with a single inner binary join. The lowerer
/// produces a `Join` RIR node followed by a `Project` for the
/// head's (K, A, B) shape. The join node has:
///   * `JoinType::Inner`.
///   * 1 key column (k) on each side.
///   * U32 key type on each side.
/// Combined with row counts in the eligibility envelope (100×100
/// = 10_000 ≤ NESTED_LOOP_TOTAL_THRESHOLD = 4_000_000) AND
/// sorted-ascending fixtures on both sides, this routes through
/// the W4.3 sort-merge provider entry point per D2 precedence
/// (sort-merge > nested-loop > hash).
const SMALL_INNER_JOIN_PROGRAM: &str = r#"
    pred left_rel(u32, u32).
    pred right_rel(u32, u32).
    pred result(u32, u32, u32).
    result(K, A, B) :- left_rel(K, A), right_rel(K, B).
"#;

#[test]
fn pre_sorted_small_cartesian_dispatches_sort_merge_and_matches_hash() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    // Fixture: 100 unique-keyed rows on each side, **sorted
    // ascending** on the key column. With L=R=100 and
    // 100×100 = 10_000 ≤ 4_000_000 threshold, both sides
    // pass `is_sorted_ascending_u32` (Ok(true)) and the
    // Cartesian-product test → W4.3 sort-merge dispatch.
    // 100 unique keys on each side → 100 join output rows.
    let left_rows: Vec<(u32, u32)> = (0..100u32).map(|i| (i, 1000 + i)).collect();
    let right_rows: Vec<(u32, u32)> = (0..100u32).map(|i| (i, 2000 + i)).collect();

    // -----------------------------------------------------------
    // Reference row set: direct provider call to hash_join_v2 on
    // the same uploaded buffers. Bypasses the executor's dispatch
    // path so it cannot be confused with the W4.3 path. Output
    // schema is [left_k, left_p, right_k, right_p] via
    // `combine_schemas`.
    // -----------------------------------------------------------
    let left_buf = upload_binary_u32(&fix.memory, &left_rows);
    let right_buf = upload_binary_u32(&fix.memory, &right_rows);
    let hash_quads_buf = fix
        .provider
        .hash_join_v2(&left_buf, &right_buf, &[0], &[0], JoinType::Inner)
        .expect("hash_join_v2 reference");
    let hash_quads = download_quads(&hash_quads_buf);
    // Project to (K, A, B) — drop the duplicate K from right side.
    // For our unique-key fixture, left_k == right_k for every
    // matched row, so projecting either is equivalent.
    let reference_set: BTreeSet<(u32, u32, u32)> = hash_quads
        .into_iter()
        .map(|(lk, lp, _rk, rp)| (lk, lp, rp))
        .collect();
    assert_eq!(
        reference_set.len(),
        100,
        "hash reference should produce exactly 100 matched rows"
    );

    // -----------------------------------------------------------
    // Dispatched run: Executor::execute_plan goes through the
    // W4.3 dispatch wiring at execute_join. Build a fresh executor
    // so both dispatch counters start at 0.
    // -----------------------------------------------------------
    let mut compiler = Compiler::new();
    let plan = compiler.compile(SMALL_INNER_JOIN_PROGRAM).expect("compile");
    let rel_ids = compiler.rel_ids().clone();
    let left_rel = *rel_ids.get("left_rel").expect("left_rel rel_id");
    let right_rel = *rel_ids.get("right_rel").expect("right_rel rel_id");

    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in &rel_ids {
        executor.register_relation(*rel_id, name);
    }
    let mut inputs: BTreeMap<&str, Vec<(u32, u32)>> = BTreeMap::new();
    inputs.insert("left_rel", left_rows.clone());
    inputs.insert("right_rel", right_rows.clone());
    for (name, rows) in &inputs {
        let buf = upload_binary_u32(&fix.memory, rows);
        executor.put_relation(name, buf);
    }

    // Pre-execute invariants: both counters at 0, no selectivity
    // feedback yet.
    assert_eq!(
        executor.sort_merge_dispatch_count(),
        0,
        "sort-merge dispatch counter must be zero before execute_plan"
    );
    assert_eq!(
        executor.nested_loop_dispatch_count(),
        0,
        "nested-loop dispatch counter must be zero before execute_plan"
    );
    assert!(
        executor
            .stats()
            .get_join_selectivity(left_rel, right_rel)
            .is_none(),
        "no selectivity feedback should exist before execute_plan"
    );

    // Run.
    executor.execute_plan(&plan).expect("execute_plan");

    // -----------------------------------------------------------
    // Post-execute assertions:
    //   1. W4.3 sort-merge dispatch counter == 1 (proves the W4.3
    //      path fired exactly once — exact equality, not `>= 1`,
    //      so the cert catches double-dispatch / re-execution
    //      regressions instead of silently masking them on a
    //      single-join fixture).
    //   2. W4.2 nested-loop counter unchanged (proves D2 precedence
    //      — sort-merge took priority over nested-loop on this
    //      sorted-eligible fixture).
    //   3. record_join_result feedback was wired into stats
    //      (D6 invariant — proves the W4.3 dispatch branch
    //      reaches the shared record_join_result block at the
    //      bottom of execute_join).
    //   4. Row-set parity vs hash reference (proves correctness).
    // -----------------------------------------------------------
    assert_eq!(
        executor.sort_merge_dispatch_count(),
        1,
        "W4.3 sort-merge dispatch must have fired exactly once for this \
         single-join program; got counter {}",
        executor.sort_merge_dispatch_count()
    );
    assert_eq!(
        executor.nested_loop_dispatch_count(),
        0,
        "W4.2 nested-loop dispatch must NOT have fired (D2 precedence: \
         sort-merge takes priority on sorted-eligible fixtures); got counter {}",
        executor.nested_loop_dispatch_count()
    );

    assert!(
        executor
            .stats()
            .get_join_selectivity(left_rel, right_rel)
            .is_some(),
        "record_join_result must have been called for the W4.3-dispatched join \
         (left_rel={:?}, right_rel={:?}); selectivity should transition None → Some",
        left_rel,
        right_rel
    );

    let result_buf = executor
        .store()
        .get("result")
        .expect("result relation must exist post-execute");
    let dispatched_set: BTreeSet<(u32, u32, u32)> =
        download_triples(result_buf).into_iter().collect();
    assert_eq!(
        dispatched_set.len(),
        100,
        "W4.3-dispatched result should produce exactly 100 matched rows; got {}",
        dispatched_set.len()
    );
    assert_eq!(
        dispatched_set, reference_set,
        "W4.3 sort-merge row set must equal the hash_join_v2 reference"
    );
}
