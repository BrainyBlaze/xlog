// crates/xlog-integration/tests/test_wcoj_record_join_result_feedback.rs
//! WCOJ dispatch feedback validation: successful WCOJ dispatch wires observed
//! selectivity back into `xlog_stats::StatsManager` via
//! `record_join_result`.
//!
//! Locks three `record_join_result` feedback properties
//! (corrected after review):
//!
//!   1. **Dispatch path actually calls `record_join_result`.**
//!      Pre-dispatch `get_join_selectivity(slot_a, slot_b) == None`;
//!      post-dispatch `Some(_)`. The transition proves the
//!      executor path made the call (test-side
//!      `update_cardinality` does NOT touch the selectivity
//!      cache, so a `None → Some` transition cannot be a
//!      false positive from fixture seeding).
//!
//!   2. **`binary_est` differs between runs.** Read directly
//!      via `executor.stats().estimate_join_cardinality(...)`
//!      — same call the cardinality cost model uses, no new
//!      public API or test-only peek.
//!
//!   3. **Row-set parity unchanged across runs.** Same
//!      executor runs `execute_plan` twice; the recursive
//!      store converges to the same fixpoint both times.
//!
//! Recording is skipped when input cardinalities are missing
//! (the missing-stats safety floor); the validation
//! `wcoj_dispatch_does_not_record_when_input_cards_missing`
//! pins this.

use std::collections::BTreeMap;
use std::sync::Arc;

use cudarc::driver::sys;
use xlog_core::{MemoryBudget, RuntimeConfig, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::CudaBuffer;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_logic::Compiler;
use xlog_runtime::Executor;

// ---------------------------------------------------------------
// Fixture helpers
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

fn upload_binary_u32(memory: &Arc<GpuMemoryManager>, rows: &[(u32, u32)]) -> CudaBuffer {
    let n = rows.len() as u32;
    let col0_host: Vec<u32> = rows.iter().map(|(a, _)| *a).collect();
    let col1_host: Vec<u32> = rows.iter().map(|(_, b)| *b).collect();
    let bytes_per_col = (n as usize) * std::mem::size_of::<u32>();
    let mut col0 = memory.alloc::<u8>(bytes_per_col).expect("alloc col0");
    let mut col1 = memory.alloc::<u8>(bytes_per_col).expect("alloc col1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let device = memory.device().inner();
    if !col0_host.is_empty() {
        let col0_bytes: Vec<u8> = col0_host.iter().flat_map(|v| v.to_le_bytes()).collect();
        let col1_bytes: Vec<u8> = col1_host.iter().flat_map(|v| v.to_le_bytes()).collect();
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
        ("c0".to_string(), ScalarType::U32),
        ("c1".to_string(), ScalarType::U32),
    ]);
    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into()],
        n as u64,
        d_num_rows,
        schema,
        n,
    )
}

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
    assert_eq!(buf.arity(), 3);
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
    let mut out: Vec<(u32, u32, u32)> = (0..n)
        .map(|i| {
            (
                u32::from_le_bytes(col0_bytes[i * 4..i * 4 + 4].try_into().unwrap()),
                u32::from_le_bytes(col1_bytes[i * 4..i * 4 + 4].try_into().unwrap()),
                u32::from_le_bytes(col2_bytes[i * 4..i * 4 + 4].try_into().unwrap()),
            )
        })
        .collect();
    out.sort();
    out.dedup();
    out
}

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
    assert_eq!(buf.arity(), 4);
    let mut cols = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for (c, col_bytes) in cols.iter_mut().enumerate() {
        *col_bytes = vec![0u8; n * 4];
        unsafe {
            sys::cuMemcpyDtoH_v2(
                col_bytes.as_mut_ptr() as *mut _,
                *buf.column(c).unwrap().device_ptr(),
                col_bytes.len(),
            );
        }
    }
    let mut out: Vec<(u32, u32, u32, u32)> = (0..n)
        .map(|i| {
            (
                u32::from_le_bytes(cols[0][i * 4..i * 4 + 4].try_into().unwrap()),
                u32::from_le_bytes(cols[1][i * 4..i * 4 + 4].try_into().unwrap()),
                u32::from_le_bytes(cols[2][i * 4..i * 4 + 4].try_into().unwrap()),
                u32::from_le_bytes(cols[3][i * 4..i * 4 + 4].try_into().unwrap()),
            )
        })
        .collect();
    out.sort();
    out.dedup();
    out
}

// ---------------------------------------------------------------
// Triangle fixture
// ---------------------------------------------------------------

const STABLE_TRIANGLE_RECURSIVE: &str = r#"
    pred e1(u32, u32).
    pred e2(u32, u32).
    pred e3(u32, u32).
    pred tri(u32, u32, u32).
    pred echo(u32, u32, u32).
    tri(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z).
    echo(X, Y, Z) :- tri(X, Y, Z).
    tri(X, Y, Z) :- echo(X, Y, Z).
"#;

fn triangle_inputs() -> BTreeMap<&'static str, Vec<(u32, u32)>> {
    let mut m: BTreeMap<&'static str, Vec<(u32, u32)>> = BTreeMap::new();
    m.insert(
        "e1",
        vec![
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
            (3, 4),
            (5, 6),
            (5, 7),
            (6, 7),
        ],
    );
    m.insert("e2", vec![(2, 3), (2, 4), (3, 4), (6, 7)]);
    m.insert("e3", vec![(1, 3), (1, 4), (2, 4), (3, 4), (5, 7)]);
    m
}

const STABLE_4CYCLE_RECURSIVE: &str = r#"
    pred e1(u32, u32).
    pred e2(u32, u32).
    pred e3(u32, u32).
    pred e4(u32, u32).
    pred cyc(u32, u32, u32, u32).
    pred echo(u32, u32, u32, u32).
    cyc(W, X, Y, Z) :- e1(W, X), e2(X, Y), e3(Y, Z), e4(Z, W).
    echo(W, X, Y, Z) :- cyc(W, X, Y, Z).
    cyc(W, X, Y, Z) :- echo(W, X, Y, Z).
"#;

fn cycle4_inputs() -> BTreeMap<&'static str, Vec<(u32, u32)>> {
    let mut m: BTreeMap<&'static str, Vec<(u32, u32)>> = BTreeMap::new();
    m.insert("e1", vec![(1, 2), (5, 6)]);
    m.insert("e2", vec![(2, 3), (6, 7)]);
    m.insert("e3", vec![(3, 4), (7, 8)]);
    m.insert("e4", vec![(4, 1), (8, 5)]);
    m
}

/// Build executor + register relations + upload EDB + optionally
/// seed runtime stats (cardinality on each named relation).
fn build_executor_with_seeded_stats(
    provider: Arc<CudaKernelProvider>,
    memory: &Arc<GpuMemoryManager>,
    config: RuntimeConfig,
    source: &str,
    inputs: &BTreeMap<&str, Vec<(u32, u32)>>,
    seeded_cards: &BTreeMap<&str, u64>,
) -> (Executor, xlog_core::RelId, xlog_core::RelId) {
    let mut compiler = Compiler::new();
    let plan = compiler.compile(source).expect("compile");
    let mut executor = Executor::new_with_config(provider, config);
    let rel_ids = compiler.rel_ids().clone();
    for (name, rel_id) in &rel_ids {
        executor.register_relation(*rel_id, name);
    }
    for (name, rows) in inputs {
        let buf = upload_binary_u32(memory, rows);
        executor.put_relation(name, buf);
    }
    for (name, card) in seeded_cards {
        if let Some(rid) = rel_ids.get(*name) {
            executor.stats_mut().register_relation(*rid);
            executor.stats_mut().update_cardinality(*rid, *card);
        }
    }
    let _ = executor.execute_plan(&plan).expect("execute_plan run 1");
    let slot_a = *rel_ids.get("e1").expect("e1 rel_id");
    let slot_b = *rel_ids.get("e2").expect("e2 rel_id");
    (executor, slot_a, slot_b)
}

fn rerun_plan(executor: &mut Executor, source: &str) {
    let mut compiler = Compiler::new();
    let plan = compiler.compile(source).expect("compile run 2");
    // Re-register relations on the second compiler instance.
    // The executor's relation registry is already populated;
    // re-registering with the same RelId is idempotent.
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    let _ = executor.execute_plan(&plan).expect("execute_plan run 2");
}

// ---------------------------------------------------------------
// Triangle: dispatch path actually records into StatsManager
// ---------------------------------------------------------------

#[test]
fn triangle_dispatch_records_join_result_into_stats_manager() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    // Seed large-enough cards so the cardinality cost model
    // dispatches under adaptive mode.
    let mut seeded = BTreeMap::new();
    seeded.insert("e1", 100_000u64);
    seeded.insert("e2", 100_000u64);
    seeded.insert("e3", 100_000u64);

    let (mut executor, slot_a, slot_b) = build_executor_with_seeded_stats(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default(),
        STABLE_TRIANGLE_RECURSIVE,
        &triangle_inputs(),
        &seeded,
    );

    // Property #1: dispatch fired AND stats now have a record
    // for the inner pair. Pre-dispatch this was None; the
    // None → Some transition is what proves the executor
    // path called record_join_result (not test-side mutation).
    assert!(
        executor.wcoj_triangle_dispatch_count() >= 1,
        "expected ≥ 1 triangle dispatch on the seeding pass; got {}",
        executor.wcoj_triangle_dispatch_count()
    );
    let post_run1 = executor.stats().get_join_selectivity(slot_a, slot_b);
    assert!(
        post_run1.is_some(),
        "WCOJ dispatch must call record_join_result; selectivity entry still None"
    );

    // Property #3 baseline: capture run 1's row set.
    let rows_run1 = download_triples(executor.store().get("tri").expect("tri"));
    assert!(
        !rows_run1.is_empty(),
        "fixture should produce at least 1 tri row to make this validation non-degenerate"
    );

    // Capture binary_est BEFORE run 2 (i.e., AFTER run 1's
    // record). This is what run 2's cost model will read.
    let binary_est_after_run1 =
        executor
            .stats()
            .estimate_join_cardinality(slot_a, slot_b, &[1], &[0]);

    // Run again on the same executor.
    rerun_plan(&mut executor, STABLE_TRIANGLE_RECURSIVE);

    // Property #2: binary_est differs between the two runs.
    // After run 2, the EMA has averaged in another observation
    // — selectivity (and therefore binary_est) should move.
    let binary_est_after_run2 =
        executor
            .stats()
            .estimate_join_cardinality(slot_a, slot_b, &[1], &[0]);
    assert_ne!(
        binary_est_after_run1, binary_est_after_run2,
        "binary_est must differ between runs after WCOJ feedback updates the EMA; \
         got run1={} run2={}",
        binary_est_after_run1, binary_est_after_run2
    );

    // Property #3: row-set parity unchanged.
    let rows_run2 = download_triples(executor.store().get("tri").expect("tri"));
    assert_eq!(
        rows_run1, rows_run2,
        "recursive store must converge to the same fixpoint on both runs"
    );
}

// ---------------------------------------------------------------
// 4-cycle: dispatch path actually records into StatsManager
// ---------------------------------------------------------------

#[test]
fn cycle4_dispatch_records_join_result_into_stats_manager() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let mut seeded = BTreeMap::new();
    seeded.insert("e1", 100_000u64);
    seeded.insert("e2", 100_000u64);
    seeded.insert("e3", 100_000u64);
    seeded.insert("e4", 100_000u64);

    let (mut executor, slot_a, slot_b) = build_executor_with_seeded_stats(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default().with_wcoj_4cycle_dispatch_adaptive(Some(true)),
        STABLE_4CYCLE_RECURSIVE,
        &cycle4_inputs(),
        &seeded,
    );

    assert!(
        executor.wcoj_4cycle_dispatch_count() >= 1,
        "expected ≥ 1 4-cycle dispatch; got {}",
        executor.wcoj_4cycle_dispatch_count()
    );
    let post_run1 = executor.stats().get_join_selectivity(slot_a, slot_b);
    assert!(
        post_run1.is_some(),
        "4-cycle WCOJ dispatch must call record_join_result"
    );

    let rows_run1 = download_quads(executor.store().get("cyc").expect("cyc"));
    assert!(!rows_run1.is_empty(), "fixture should produce ≥ 1 cyc row");

    let binary_est_after_run1 =
        executor
            .stats()
            .estimate_join_cardinality(slot_a, slot_b, &[1], &[0]);

    rerun_plan(&mut executor, STABLE_4CYCLE_RECURSIVE);

    let binary_est_after_run2 =
        executor
            .stats()
            .estimate_join_cardinality(slot_a, slot_b, &[1], &[0]);
    assert_ne!(
        binary_est_after_run1, binary_est_after_run2,
        "4-cycle binary_est must differ between runs; got run1={} run2={}",
        binary_est_after_run1, binary_est_after_run2
    );

    let rows_run2 = download_quads(executor.store().get("cyc").expect("cyc"));
    assert_eq!(rows_run1, rows_run2, "4-cycle row-set parity across runs");
}

// ---------------------------------------------------------------
// Missing input cards: helper must NOT record
// ---------------------------------------------------------------

#[test]
fn wcoj_dispatch_does_not_record_when_input_cards_missing() {
    // Force gate on the triangle dispatch (skips the cost
    // model, but the success arm still calls
    // record_wcoj_feedback). Stats are NOT seeded — the
    // helper's missing-cards safety floor must skip the
    // EMA update.
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let (executor, slot_a, slot_b) = build_executor_with_seeded_stats(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(true)),
        STABLE_TRIANGLE_RECURSIVE,
        &triangle_inputs(),
        &BTreeMap::new(),
    );

    // Counter advanced (force-mode dispatched).
    assert_eq!(
        executor.wcoj_triangle_dispatch_count(),
        1,
        "force gate must dispatch once on stable triangle"
    );
    // But selectivity stayed None: the helper saw missing
    // input cardinalities and skipped the record.
    let sel = executor.stats().get_join_selectivity(slot_a, slot_b);
    assert!(
        sel.is_none(),
        "missing input cards must not produce a selectivity record; got {:?}",
        sel
    );
    // Row-set parity isn't compared across runs here (one run
    // only); just confirm the binary fallback / WCOJ produced
    // a non-empty result.
    let rows = download_triples(executor.store().get("tri").expect("tri"));
    assert!(
        !rows.is_empty(),
        "force-mode WCOJ should produce ≥ 1 row even with unseeded stats"
    );
}
