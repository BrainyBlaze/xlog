// crates/xlog-integration/tests/test_wcoj_cardinality_cost_model.rs
//! v0.6.5 slice 5 — `CardinalityAwareCostModel` cert.
//!
//! Locks the contract for the slice 5 opt-in cost model:
//!
//!   * **Default config** (no env, no field) keeps slice 4
//!     dispatch counts bit-identical — the factory falls
//!     through to `SkewClassifierCostModel`.
//!   * **Opt-in via config field** with populated runtime
//!     stats: large `binary_est` triggers the asymptotic
//!     clause and dispatches.
//!   * **Opt-in via config field** with small populated
//!     stats: `binary_est` below `MIN_CARDINALITY_BINARY_INTERMEDIATE`
//!     keeps the binary-join path (counter == 0).
//!   * **Opt-in WITHOUT seeded stats**: cardinality model
//!     delegates to `SkewClassifierCostModel`, so the dispatch
//!     decision matches the legacy default.
//!   * **Env var ↔ config field parity**: `XLOG_WCOJ_COST_MODEL=cardinality`
//!     produces the same selection as the config builder.
//!
//! ## Runtime-stats seeding
//!
//! The cost model reads `Executor::stats` at dispatch time, NOT
//! compile-time-inferred stats. EDB uploads via `put_relation`
//! do not auto-populate `StatsManager`; tests must call
//! `executor.stats_mut().update_cardinality(rel_id, count)` AFTER
//! `register_relation` + `put_relation` to seed cardinality
//! before `execute_plan` runs the WCOJ dispatch site. This
//! pattern is documented for future cardinality-driven cert
//! authors.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::{Mutex, OnceLock};

use cudarc::driver::sys;
use xlog_core::{CostModelKind, MemoryBudget, RuntimeConfig, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::CudaBuffer;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_logic::Compiler;
use xlog_runtime::Executor;

// ---------------------------------------------------------------
// Fixture helpers (mirror test_wcoj_recursive_dispatch.rs)
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
    for c in 0..4 {
        cols[c] = vec![0u8; n * 4];
        unsafe {
            sys::cuMemcpyDtoH_v2(
                cols[c].as_mut_ptr() as *mut _,
                *buf.column(c).unwrap().device_ptr(),
                cols[c].len(),
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

/// Triangle program with `pred` declarations to anchor U32
/// schemas, identical to slice 4's stable-triangle fixture.
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

/// 4-cycle program with `pred` declarations, mirrors slice 4's
/// stable-4-cycle fixture. Adaptive mode for 4-cycle is opt-in
/// (not default-on like triangle), so the tests below set
/// `with_wcoj_4cycle_dispatch_adaptive(Some(true))` explicitly.
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

/// Compile + execute. Optionally seed runtime stats with the
/// supplied (relation-name, cardinality) pairs after relations
/// are registered + populated. Empty `seeded_cards` means
/// "stats not seeded" — the cardinality model's missing-stats
/// safety floor delegates to the skew model.
fn run_with_optional_stats(
    provider: Arc<CudaKernelProvider>,
    memory: &Arc<GpuMemoryManager>,
    config: RuntimeConfig,
    source: &str,
    inputs: &BTreeMap<&str, Vec<(u32, u32)>>,
    seeded_cards: &BTreeMap<&str, u64>,
) -> Executor {
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
    // v0.6.5 slice 5 — seed runtime stats AFTER register +
    // upload. The cost model reads `Executor::stats` at
    // dispatch time; without explicit seeding the cardinality
    // model delegates to the skew classifier (safety floor).
    for (name, card) in seeded_cards {
        if let Some(rid) = rel_ids.get(*name) {
            executor.stats_mut().register_relation(*rid);
            executor.stats_mut().update_cardinality(*rid, *card);
        }
    }
    let _ = executor.execute_plan(&plan).expect("execute_plan");
    executor
}

// ---------------------------------------------------------------
// Env-lock for XLOG_WCOJ_COST_MODEL
// ---------------------------------------------------------------

fn cost_model_env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

struct CostModelEnvSnapshot(Option<String>);

impl CostModelEnvSnapshot {
    fn capture_and_clear() -> Self {
        let prior = std::env::var("XLOG_WCOJ_COST_MODEL").ok();
        // SAFETY: caller holds `cost_model_env_lock`.
        unsafe {
            std::env::remove_var("XLOG_WCOJ_COST_MODEL");
        }
        Self(prior)
    }
}

impl Drop for CostModelEnvSnapshot {
    fn drop(&mut self) {
        unsafe {
            match self.0.take() {
                Some(v) => std::env::set_var("XLOG_WCOJ_COST_MODEL", v),
                None => std::env::remove_var("XLOG_WCOJ_COST_MODEL"),
            }
        }
    }
}

fn with_cost_model_env<R>(f: impl FnOnce() -> R) -> R {
    let _guard = cost_model_env_lock()
        .lock()
        .expect("cost-model env lock poisoned");
    let _snap = CostModelEnvSnapshot::capture_and_clear();
    f()
}

// ---------------------------------------------------------------
// Tests
// ---------------------------------------------------------------

#[test]
fn cardinality_default_off_keeps_slice4_dispatch_counts() {
    // No env, no config field → factory selects
    // SkewClassifierCostModel. Slice 4's stable-triangle test
    // baseline (counter == 1 with force gate on) must hold
    // bit-identical.
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    with_cost_model_env(|| {
        let executor = run_with_optional_stats(
            Arc::clone(&fix.provider),
            &fix.memory,
            RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(true)),
            STABLE_TRIANGLE_RECURSIVE,
            &triangle_inputs(),
            &BTreeMap::new(),
        );
        assert_eq!(
            executor.wcoj_triangle_dispatch_count(),
            1,
            "default cost model must preserve slice 4 stable-triangle counter"
        );
    });
}

#[test]
fn cardinality_opt_in_with_seeded_large_cards_dispatches_via_adaptive() {
    // Opt-in cardinality model. Seed runtime stats large enough
    // that binary_est >= LARGE_CARDINALITY_BINARY_INTERMEDIATE
    // (1M). 100K * 100K * 0.1 default selectivity ≈ 1B → above
    // threshold. Use adaptive mode (default-on for triangle)
    // so the cost model is consulted, not bypassed.
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    with_cost_model_env(|| {
        let mut seeded = BTreeMap::new();
        seeded.insert("e1", 100_000u64);
        seeded.insert("e2", 100_000u64);
        seeded.insert("e3", 100_000u64);
        let executor = run_with_optional_stats(
            Arc::clone(&fix.provider),
            &fix.memory,
            RuntimeConfig::default().with_wcoj_cost_model(Some(CostModelKind::Cardinality)),
            STABLE_TRIANGLE_RECURSIVE,
            &triangle_inputs(),
            &seeded,
        );
        assert!(
            executor.wcoj_triangle_dispatch_count() >= 1,
            "cardinality model + huge binary_est must dispatch even on uniform inputs; got counter {}",
            executor.wcoj_triangle_dispatch_count()
        );
    });
}

#[test]
fn cardinality_opt_in_with_small_cards_falls_back_to_binary() {
    // Opt-in cardinality model, but seeded stats are tiny:
    // 5 * 5 * 0.1 = 2.5 → binary_est = 1 (clamped to >= 1) <
    // MIN_CARDINALITY_BINARY_INTERMEDIATE (4096). Even with
    // adaptive mode and a high score, the cardinality model
    // says "kernel not worth the launch" → no dispatch.
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    with_cost_model_env(|| {
        let mut seeded = BTreeMap::new();
        seeded.insert("e1", 5u64);
        seeded.insert("e2", 5u64);
        seeded.insert("e3", 5u64);
        let executor = run_with_optional_stats(
            Arc::clone(&fix.provider),
            &fix.memory,
            RuntimeConfig::default().with_wcoj_cost_model(Some(CostModelKind::Cardinality)),
            STABLE_TRIANGLE_RECURSIVE,
            &triangle_inputs(),
            &seeded,
        );
        assert_eq!(
            executor.wcoj_triangle_dispatch_count(),
            0,
            "cardinality model + small binary_est must fall back; got counter {}",
            executor.wcoj_triangle_dispatch_count()
        );
    });
}

#[test]
fn cardinality_opt_in_without_seeded_stats_delegates_to_skew_model() {
    // Opt-in cardinality model with NO stats seeded → the
    // missing-stats safety floor MUST delegate to
    // SkewClassifierCostModel. We prove that by running two
    // adaptive-mode executions on the same fixture and
    // asserting counter + row set parity:
    //
    //   1. Default config (SkewClassifier).
    //   2. Cardinality opt-in (no stats seeded → delegates).
    //
    // Force-gate on a triangle would bypass the cost model
    // entirely, so this test stays in adaptive mode (default-on
    // for triangle) so the cost model IS consulted. If the
    // delegation logic broke, the cardinality model would
    // produce a different decision and the assertion would
    // catch it.
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    with_cost_model_env(|| {
        // Run 1: default cost model (skew classifier), adaptive.
        let baseline = run_with_optional_stats(
            Arc::clone(&fix.provider),
            &fix.memory,
            RuntimeConfig::default(),
            STABLE_TRIANGLE_RECURSIVE,
            &triangle_inputs(),
            &BTreeMap::new(),
        );
        let baseline_counter = baseline.wcoj_triangle_dispatch_count();
        let baseline_rows = download_triples(baseline.store().get("tri").expect("tri"));

        // Run 2: cardinality model, NO stats seeded → delegate.
        let delegated = run_with_optional_stats(
            Arc::clone(&fix.provider),
            &fix.memory,
            RuntimeConfig::default().with_wcoj_cost_model(Some(CostModelKind::Cardinality)),
            STABLE_TRIANGLE_RECURSIVE,
            &triangle_inputs(),
            &BTreeMap::new(),
        );

        assert_eq!(
            delegated.wcoj_triangle_dispatch_count(),
            baseline_counter,
            "cardinality model with missing stats must delegate to \
             SkewClassifier; counter {} ≠ skew baseline {}",
            delegated.wcoj_triangle_dispatch_count(),
            baseline_counter,
        );
        let delegated_rows = download_triples(delegated.store().get("tri").expect("tri"));
        assert_eq!(
            delegated_rows, baseline_rows,
            "delegation must produce the same row set as the legacy default"
        );
    });
}

#[test]
fn cardinality_4cycle_opt_in_with_seeded_large_cards_dispatches() {
    // 4-cycle counterpart of the large-binary triangle test.
    // 4-cycle adaptive is opt-in (not default-on), so enable
    // it explicitly. With cardinality model + seeded large
    // cards, binary_est >> LARGE threshold → dispatch.
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    with_cost_model_env(|| {
        let mut seeded = BTreeMap::new();
        seeded.insert("e1", 100_000u64);
        seeded.insert("e2", 100_000u64);
        seeded.insert("e3", 100_000u64);
        seeded.insert("e4", 100_000u64);
        let executor = run_with_optional_stats(
            Arc::clone(&fix.provider),
            &fix.memory,
            RuntimeConfig::default()
                .with_wcoj_cost_model(Some(CostModelKind::Cardinality))
                .with_wcoj_4cycle_dispatch_adaptive(Some(true)),
            STABLE_4CYCLE_RECURSIVE,
            &cycle4_inputs(),
            &seeded,
        );
        assert!(
            executor.wcoj_4cycle_dispatch_count() >= 1,
            "cardinality model + huge binary_est must dispatch on 4-cycle; got counter {}",
            executor.wcoj_4cycle_dispatch_count()
        );
    });
}

#[test]
fn cardinality_4cycle_opt_in_with_small_cards_falls_back_to_binary() {
    // 4-cycle counterpart of the small-binary triangle test.
    // Seeded stats are tiny → binary_est below MIN threshold →
    // no dispatch (binary-join handles).
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    with_cost_model_env(|| {
        let mut seeded = BTreeMap::new();
        seeded.insert("e1", 5u64);
        seeded.insert("e2", 5u64);
        seeded.insert("e3", 5u64);
        seeded.insert("e4", 5u64);
        let executor = run_with_optional_stats(
            Arc::clone(&fix.provider),
            &fix.memory,
            RuntimeConfig::default()
                .with_wcoj_cost_model(Some(CostModelKind::Cardinality))
                .with_wcoj_4cycle_dispatch_adaptive(Some(true)),
            STABLE_4CYCLE_RECURSIVE,
            &cycle4_inputs(),
            &seeded,
        );
        assert_eq!(
            executor.wcoj_4cycle_dispatch_count(),
            0,
            "cardinality model + small binary_est must fall back on 4-cycle; got counter {}",
            executor.wcoj_4cycle_dispatch_count()
        );
        // Confirm the program still produced the correct row
        // set via the binary-join path.
        let rows = download_quads(executor.store().get("cyc").expect("cyc"));
        assert!(
            !rows.is_empty(),
            "binary-join fallback must still produce 4-cycle rows"
        );
    });
}

#[test]
fn cardinality_opt_in_via_env_var_matches_config_field() {
    // Set env var to `cardinality`, no config-field override —
    // expect identical behavior to the config-field-set test
    // above.
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    with_cost_model_env(|| {
        // SAFETY: caller holds `cost_model_env_lock`.
        unsafe {
            std::env::set_var("XLOG_WCOJ_COST_MODEL", "cardinality");
        }
        let mut seeded = BTreeMap::new();
        seeded.insert("e1", 100_000u64);
        seeded.insert("e2", 100_000u64);
        seeded.insert("e3", 100_000u64);
        let executor = run_with_optional_stats(
            Arc::clone(&fix.provider),
            &fix.memory,
            RuntimeConfig::default(), // no config-field override
            STABLE_TRIANGLE_RECURSIVE,
            &triangle_inputs(),
            &seeded,
        );
        assert!(
            executor.wcoj_triangle_dispatch_count() >= 1,
            "env-var opt-in with seeded large cards must dispatch; got counter {}",
            executor.wcoj_triangle_dispatch_count()
        );
    });
}
