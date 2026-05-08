// crates/xlog-integration/tests/test_w42_nested_loop_dispatch.rs
//! W4.2 dispatch + parity certs.
//!
//! Cert A — small × small dispatch routes through the W4.2
//! `nested_loop_join_v2_inner_u32_1key` provider entry point,
//! produces a row-set bit-identical to `hash_join_v2`'s reference
//! output, AND wires `record_join_result` feedback into
//! `StatsManager` (the same contract the W2.4 cert pins for the
//! WCOJ path).
//!
//! Subsequent certs (B / C / C' / E) will land in follow-up
//! commits per the W4.2 plan iteration-4 Steps 7 / 8 / 10.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use cudarc::driver::sys;
use xlog_core::RelId;
use xlog_core::{MemoryBudget, RuntimeConfig, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::CudaBuffer;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager, JoinType};
use xlog_ir::{JoinType as IrJoinType, RirNode};
use xlog_logic::Compiler;
use xlog_runtime::Executor;

// ---------------------------------------------------------------
// Fixture helpers (mirrors W2.4 cert pattern at
// crates/xlog-integration/tests/test_wcoj_record_join_result_feedback.rs).
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
// Cert A — small×small dispatches nested-loop, matches hash, and
// records join feedback.
// ---------------------------------------------------------------

/// Datalog program with a single inner binary join. The lowerer
/// produces a `Join` RIR node followed by a `Project` for the
/// head's (K, A, B) shape. The join node has:
///   * `JoinType::Inner`.
///   * 1 key column (k) on each side.
///   * U32 key type on each side.
/// Combined with row counts in the eligibility envelope (100×100
/// = 10_000 ≤ NESTED_LOOP_TOTAL_THRESHOLD = 4_000_000), this
/// routes through the W4.2 nested-loop provider entry point.
const SMALL_INNER_JOIN_PROGRAM: &str = r#"
    pred left_rel(u32, u32).
    pred right_rel(u32, u32).
    pred result(u32, u32, u32).
    result(K, A, B) :- left_rel(K, A), right_rel(K, B).
"#;

#[test]
fn small_small_dispatches_nested_loop_and_matches_hash() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    // Fixture: 100 unique-keyed rows on each side. All 100 keys
    // match → 100 join output rows. 100×100 = 10_000 Cartesian
    // ≤ 4_000_000 threshold → eligible for W4.2 dispatch.
    let left_rows: Vec<(u32, u32)> = (0..100u32).map(|i| (i, 1000 + i)).collect();
    let right_rows: Vec<(u32, u32)> = (0..100u32).map(|i| (i, 2000 + i)).collect();

    // -----------------------------------------------------------
    // Reference row set: direct provider call to hash_join_v2 on
    // the same uploaded buffers. Bypasses the executor's dispatch
    // path so it cannot be confused with the W4.2 path. Output
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
    // Dispatched run: Executor::execute_plan goes through the W4.2
    // dispatch wiring at execute_join. Build a fresh executor so
    // the dispatch counter starts at 0.
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

    // Pre-execute invariants (no `wcoj_*` references per the W4.2
    // plan iter-4 F-W42-6 — wcoj counters are unrelated):
    assert_eq!(
        executor.nested_loop_dispatch_count(),
        0,
        "dispatch counter must be zero before execute_plan"
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
    //   1. Dispatch counter incremented (proves W4.2 path fired).
    //   2. record_join_result feedback was wired into stats
    //      (proves the Step-5 patch routes through the shared
    //      record_join_result block — directly addresses the bug
    //      that motivated the patch).
    //   3. Row-set parity vs hash reference (proves correctness).
    // -----------------------------------------------------------
    assert!(
        executor.nested_loop_dispatch_count() >= 1,
        "W4.2 nested-loop dispatch must have fired at least once; got counter {}",
        executor.nested_loop_dispatch_count()
    );

    assert!(
        executor
            .stats()
            .get_join_selectivity(left_rel, right_rel)
            .is_some(),
        "record_join_result must have been called for the W4.2-dispatched join \
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
        "W4.2-dispatched result should produce exactly 100 matched rows; got {}",
        dispatched_set.len()
    );
    assert_eq!(
        dispatched_set, reference_set,
        "W4.2 nested-loop row set must equal the hash_join_v2 reference"
    );
}

// ---------------------------------------------------------------
// Cert B — large × small Cartesian product falls back to hash.
//
// Per W4.2 plan iter-4 F-W42-3: matches the board's "large × small
// picks hash" acceptance line. Asymmetric `L=50_000, R=100` →
// Cartesian = 5_000_000 > NESTED_LOOP_TOTAL_THRESHOLD = 4_000_000
// → ineligible. The eligibility predicate's individual checks
// (Inner, 1-key, U32, type equality) ALL pass; only the
// Cartesian-product threshold rejects this join. Confirms the
// `left * right` semantic of the threshold (a naive `right_rows
// < 1000` semantic would admit this since R=100 < 1000).
// ---------------------------------------------------------------

#[test]
fn large_times_small_falls_back_to_hash_above_threshold() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    // Fixture: L=50_000 unique keys [0..50_000); R=100 unique
    // keys [0..100). Cartesian = 5_000_000 > 4_000_000 threshold.
    //
    // Bounded matches: right keys ⊆ left keys; each right key
    // matches exactly one left row. Output = 100 rows — small
    // enough for `BTreeSet<Row>` parity comparison.
    let left_rows: Vec<(u32, u32)> = (0..50_000u32).map(|i| (i, 1000 + i)).collect();
    let right_rows: Vec<(u32, u32)> = (0..100u32).map(|i| (i, 2_000_000 + i)).collect();

    // Reference row set: direct provider.hash_join_v2 on the
    // same uploaded buffers. Output schema is 4-col
    // [left_k, left_p, right_k, right_p]; project to 3-col
    // (K, A, B) via dropping the duplicate K from the right.
    let left_buf = upload_binary_u32(&fix.memory, &left_rows);
    let right_buf = upload_binary_u32(&fix.memory, &right_rows);
    let hash_quads_buf = fix
        .provider
        .hash_join_v2(&left_buf, &right_buf, &[0], &[0], JoinType::Inner)
        .expect("hash_join_v2 reference");
    let reference_set: BTreeSet<(u32, u32, u32)> = download_quads(&hash_quads_buf)
        .into_iter()
        .map(|(lk, lp, _rk, rp)| (lk, lp, rp))
        .collect();
    assert_eq!(
        reference_set.len(),
        100,
        "hash reference should produce exactly 100 matched rows"
    );

    // Dispatched run: Executor::execute_plan must route through
    // hash because the threshold predicate refuses the W4.2
    // dispatch.
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

    assert_eq!(
        executor.nested_loop_dispatch_count(),
        0,
        "dispatch counter must be zero before execute_plan"
    );

    executor.execute_plan(&plan).expect("execute_plan");

    // -----------------------------------------------------------
    // Assertions:
    //   1. **Load-bearing**: `nested_loop_dispatch_count() == 0`
    //      — the eligibility predicate refused to dispatch
    //      W4.2 because the Cartesian product (5M) exceeds the
    //      threshold (4M). Strict equality, NOT `<= some bound`.
    //   2. Row-set parity vs hash reference (correctness witness
    //      for the hash fallback path).
    //   3. `record_join_result` feedback wired even on the hash
    //      fallback path (drop-in contract: the existing W2.4
    //      contract holds regardless of which dispatch path
    //      execute_join chose).
    // -----------------------------------------------------------
    assert_eq!(
        executor.nested_loop_dispatch_count(),
        0,
        "above-threshold join MUST NOT dispatch nested-loop; \
         got dispatch counter {}",
        executor.nested_loop_dispatch_count()
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
        "above-threshold result should produce exactly 100 matched rows; got {}",
        dispatched_set.len()
    );
    assert_eq!(
        dispatched_set, reference_set,
        "hash-fallback row set must equal the direct provider.hash_join_v2 reference"
    );

    assert!(
        executor
            .stats()
            .get_join_selectivity(left_rel, right_rel)
            .is_some(),
        "record_join_result must fire on the hash fallback path too \
         (the W4.2 Step-5 patch routed all three dispatch paths through \
          the shared feedback block); pre-execute None → post-execute \
          Some(_) is the W2.4 contract that W4.2 must preserve"
    );
}

// ---------------------------------------------------------------
// Helpers for Certs C and C' (manual RirNode construction).
// ---------------------------------------------------------------

/// Download a 2-col U32 buffer (used for Semi-join output, which
/// preserves the left schema of an arity-2 left input).
fn download_pairs(buf: &CudaBuffer) -> Vec<(u32, u32)> {
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
    assert_eq!(buf.arity(), 2, "download_pairs expects arity 2");
    let mut col0_bytes = vec![0u8; n * 4];
    let mut col1_bytes = vec![0u8; n * 4];
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
    }
    (0..n)
        .map(|i| {
            let off = i * 4;
            (
                u32::from_le_bytes(col0_bytes[off..off + 4].try_into().unwrap()),
                u32::from_le_bytes(col1_bytes[off..off + 4].try_into().unwrap()),
            )
        })
        .collect()
}

/// Build an executor + register two relations under manual
/// RelIds + upload buffers. Returns the executor and the
/// RelIds (`lhs`, `rhs`) so the test can construct an
/// `RirNode::Join` referencing them.
fn build_executor_with_two_relations(
    fix: &RuntimeBackedFixture,
    left_rows: &[(u32, u32)],
    right_rows: &[(u32, u32)],
) -> (Executor, RelId, RelId) {
    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    let lhs = RelId(1000);
    let rhs = RelId(1001);
    executor.register_relation(lhs, "lhs");
    executor.register_relation(rhs, "rhs");
    executor.put_relation("lhs", upload_binary_u32(&fix.memory, left_rows));
    executor.put_relation("rhs", upload_binary_u32(&fix.memory, right_rows));
    (executor, lhs, rhs)
}

// ---------------------------------------------------------------
// Cert C — multi-col composite key inner join falls back to hash.
//
// Per W4.2 plan iter-4 D1 + D5: the eligibility predicate's
// `left_keys.len() != 1 || right_keys.len() != 1` check rejects
// composite-key joins regardless of size. Fixture is small
// (100 × 100 = 10K, well below the 4M threshold) so SIZE is
// NOT the disqualifying property — only the key arity is. This
// isolates the multi-key rejection.
// ---------------------------------------------------------------

#[test]
fn multi_col_key_falls_back_to_hash() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    // Fixture: 2-col arity each side (both columns are keys; no
    // payload). 100 rows each side; key tuples (i, i+1000). All
    // 100 tuples match exactly. Cartesian = 10_000 ≪ 4_000_000.
    // Size is admissible; only the multi-key shape disqualifies.
    let left_rows: Vec<(u32, u32)> = (0..100u32).map(|i| (i, i + 1000)).collect();
    let right_rows: Vec<(u32, u32)> = (0..100u32).map(|i| (i, i + 1000)).collect();

    // Reference: provider.hash_join_v2 with composite key cols
    // [0, 1] on each side. Output schema is 4-col
    // [left_k1, left_k2, right_k1, right_k2] (combine_schemas).
    let left_buf = upload_binary_u32(&fix.memory, &left_rows);
    let right_buf = upload_binary_u32(&fix.memory, &right_rows);
    let hash_buf = fix
        .provider
        .hash_join_v2(&left_buf, &right_buf, &[0, 1], &[0, 1], JoinType::Inner)
        .expect("hash_join_v2 reference (multi-key)");
    let reference_set: BTreeSet<(u32, u32, u32, u32)> =
        download_quads(&hash_buf).into_iter().collect();
    assert_eq!(
        reference_set.len(),
        100,
        "multi-key reference should produce exactly 100 matched rows"
    );

    // Dispatched: Executor::execute_node on a manually-built
    // RirNode::Join { left_keys: [0, 1], right_keys: [0, 1],
    // join_type: Inner }. The compiler+lowerer would produce
    // this same shape from a Datalog rule with a 2-col
    // composite key, but we construct it directly to keep the
    // test focused on the dispatch decision.
    let (mut executor, lhs, rhs) = build_executor_with_two_relations(&fix, &left_rows, &right_rows);
    let join = RirNode::Join {
        left: Box::new(RirNode::Scan { rel: lhs }),
        right: Box::new(RirNode::Scan { rel: rhs }),
        left_keys: vec![0, 1],
        right_keys: vec![0, 1],
        join_type: IrJoinType::Inner,
    };
    let result = executor.execute_node(&join).expect("execute_node");

    // -----------------------------------------------------------
    // Load-bearing: dispatch counter MUST be exactly zero.
    // The eligibility predicate's
    //   `left_keys.len() == 1 && right_keys.len() == 1`
    // check rejected this join despite small size + Inner +
    // matching U32 types.
    // -----------------------------------------------------------
    assert_eq!(
        executor.nested_loop_dispatch_count(),
        0,
        "multi-key inner join MUST NOT dispatch nested-loop \
         (eligibility predicate's 1-key gate); got dispatch counter {}",
        executor.nested_loop_dispatch_count()
    );

    // Row-set parity vs hash reference (correctness witness for
    // the hash fallback path on multi-key inputs).
    let dispatched_set: BTreeSet<(u32, u32, u32, u32)> =
        download_quads(&result).into_iter().collect();
    assert_eq!(
        dispatched_set, reference_set,
        "multi-key hash-fallback row set must equal the direct provider.hash_join_v2 reference"
    );
}

// ---------------------------------------------------------------
// Cert C' — Semi-join falls back to hash, semi-join row-set
// semantics preserved.
//
// Per W4.2 plan iter-4 D1 + D5: the eligibility predicate's
// `join_type == JoinType::Inner` check rejects Semi/Anti/LeftOuter
// regardless of size + key shape. Fixture is small (100 × 100 =
// 10K, well below threshold) and uses 1-key U32 — only the
// non-Inner join type disqualifies.
//
// Semi-join semantics: output = subset of left rows whose key
// matches at least one right key. Output schema = left schema.
// For the reference computation we filter host-side and compare
// to the executor's output as `BTreeSet<(left_col0, left_col1)>`.
// ---------------------------------------------------------------

#[test]
fn semi_join_falls_back_to_hash() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    // Fixture: 2-col arity each side. left has 100 rows with
    // keys [0..100); right has 50 rows with keys [25..75) (a
    // proper subset of left's keys). Semi-join should produce
    // exactly the 50 left rows whose keys are in [25..75).
    let left_rows: Vec<(u32, u32)> = (0..100u32).map(|i| (i, 1000 + i)).collect();
    let right_rows: Vec<(u32, u32)> = (25..75u32).map(|i| (i, 9999)).collect();

    // Host-computed semi-join reference: left rows whose key is
    // in right's key set.
    let right_keys: BTreeSet<u32> = right_rows.iter().map(|(k, _)| *k).collect();
    let reference_set: BTreeSet<(u32, u32)> = left_rows
        .iter()
        .filter(|(k, _)| right_keys.contains(k))
        .copied()
        .collect();
    assert_eq!(
        reference_set.len(),
        50,
        "semi-join reference should produce exactly 50 left rows with matching keys"
    );

    // Dispatched: Executor::execute_node on a manually-built
    // RirNode::Join { join_type: Semi, ... }.
    let (mut executor, lhs, rhs) = build_executor_with_two_relations(&fix, &left_rows, &right_rows);
    let join = RirNode::Join {
        left: Box::new(RirNode::Scan { rel: lhs }),
        right: Box::new(RirNode::Scan { rel: rhs }),
        left_keys: vec![0],
        right_keys: vec![0],
        join_type: IrJoinType::Semi,
    };
    let result = executor.execute_node(&join).expect("execute_node");

    // -----------------------------------------------------------
    // Load-bearing: dispatch counter MUST be exactly zero. The
    // eligibility predicate's `join_type == JoinType::Inner`
    // check rejected this join despite small size + 1-key + U32.
    // -----------------------------------------------------------
    assert_eq!(
        executor.nested_loop_dispatch_count(),
        0,
        "Semi join MUST NOT dispatch nested-loop \
         (eligibility predicate's Inner-only gate); got dispatch counter {}",
        executor.nested_loop_dispatch_count()
    );

    // Semi-join row-set semantics: output schema = left schema
    // (arity 2). Each output row is a left row whose key has at
    // least one match in right.
    assert_eq!(
        result.arity(),
        2,
        "Semi-join output schema must equal left's schema (arity 2)"
    );
    let dispatched_set: BTreeSet<(u32, u32)> = download_pairs(&result).into_iter().collect();
    assert_eq!(
        dispatched_set, reference_set,
        "Semi-join row set must match the host-computed semi reference \
         (left rows whose keys appear in right)"
    );
}
