// crates/xlog-integration/tests/test_wcoj_adaptive_default_on.rs
//! v0.6.2 default-on adaptive dispatch + kill-switch contract.
//!
//! Locks the post-A2-lite default behavior:
//!
//!   * `RuntimeConfig::default()` (no overrides) routes
//!     non-recursive triangle rules through the adaptive
//!     classifier. Super-hub-shaped inputs dispatch WCOJ;
//!     uniform/empty fall back to binary join. Force /
//!     explicit-off still beat the default.
//!   * **Hard kill switch** `wcoj_triangle_dispatch_disabled` /
//!     env `XLOG_DISABLE_WCOJ_TRIANGLE=1` beats EVERY other
//!     dispatch flag — config force, env force, config adaptive,
//!     env adaptive, and the default-on. Use case: ops emergency
//!     to pin all WCOJ dispatch off without touching application
//!     code or other env vars.
//!
//! Precedence (highest → lowest):
//!   1. `wcoj_triangle_dispatch_disabled = Some(true)` /
//!      `XLOG_DISABLE_WCOJ_TRIANGLE=1` → no dispatch.
//!   2. `wcoj_triangle_dispatch = Some(true)` /
//!      `XLOG_USE_WCOJ_TRIANGLE_U32=1` → force WCOJ; classifier
//!      bypassed.
//!   3. `wcoj_triangle_dispatch = Some(false)` → explicit off.
//!   4. `wcoj_triangle_dispatch_adaptive = Some(true)` /
//!      `XLOG_USE_WCOJ_TRIANGLE_ADAPTIVE=1` → run classifier.
//!   5. **Default**: classifier runs (post-flip).
//!
//! Hard scope:
//!   * No bench changes.
//!   * No env-mutating tests (they'd race the bench process).
//!     Env precedence is exercised only via the resolver unit
//!     tests in xlog-runtime.

use std::collections::BTreeMap;
use std::sync::Arc;

use xlog_core::{MemoryBudget, RuntimeConfig, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::CudaBuffer;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_logic::Compiler;
use xlog_runtime::Executor;

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
    provider: Arc<CudaKernelProvider>,
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
        Arc::new(CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory)).ok()?);
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
            let b = if i % 2 == 0 {
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

const SOURCE: &str = "tri(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z).";

fn build_uniform_inputs(memory: &Arc<GpuMemoryManager>) -> [CudaBuffer; 3] {
    let rows = 5_000u32;
    let kr = 1000u32;
    let e1 = dedup_pairs(uniform_pairs(1, rows, kr));
    let e2 = dedup_pairs(uniform_pairs(2, rows, kr));
    let e3 = dedup_pairs(uniform_pairs(3, rows, kr));
    [
        upload_binary_u32(memory, &e1),
        upload_binary_u32(memory, &e2),
        upload_binary_u32(memory, &e3),
    ]
}

fn build_superhub_inputs(memory: &Arc<GpuMemoryManager>) -> [CudaBuffer; 3] {
    let rows = 5_000u32;
    let kr = 1000u32;
    let e1 = dedup_pairs(superhub_pairs_xy(101, rows, kr, 7));
    let e2 = dedup_pairs(superhub_pairs_first(202, rows, kr, 7));
    let e3 = dedup_pairs(superhub_pairs_first(303, rows, kr, 13));
    [
        upload_binary_u32(memory, &e1),
        upload_binary_u32(memory, &e2),
        upload_binary_u32(memory, &e3),
    ]
}

fn run_with_config_and_cards(
    fix: &Fix,
    config: RuntimeConfig,
    inputs: [CudaBuffer; 3],
    seeded_cards: Option<[u64; 3]>,
) -> Executor {
    let mut compiler = Compiler::new();
    let plan = compiler.compile(SOURCE).expect("compile");
    let mut executor = Executor::new_with_config(Arc::clone(&fix.provider), config);
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    let [b1, b2, b3] = inputs;
    executor.put_relation("e1", b1);
    executor.put_relation("e2", b2);
    executor.put_relation("e3", b3);
    if let Some(cards) = seeded_cards {
        for (idx, name) in ["e1", "e2", "e3"].iter().enumerate() {
            if let Some(rel_id) = compiler.rel_ids().get(*name) {
                executor.stats_mut().register_relation(*rel_id);
                executor.stats_mut().update_cardinality(*rel_id, cards[idx]);
            }
        }
    }
    let _ = executor.execute_plan(&plan).expect("execute_plan");
    executor
}

fn run_with_config(fix: &Fix, config: RuntimeConfig, inputs: [CudaBuffer; 3]) -> Executor {
    run_with_config_and_cards(fix, config, inputs, None)
}

// =================================================================
// Default-on: bare RuntimeConfig::default() routes triangles
// through the classifier.
// =================================================================

#[test]
fn default_runtime_dispatches_superhub() {
    // No overrides at all — RuntimeConfig::default() should
    // produce adaptive-on behavior. Super-hub fixture clears
    // the classifier threshold → counter == 1.
    let Some(fix) = make_fix() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let exec = run_with_config_and_cards(
        &fix,
        RuntimeConfig::default(),
        build_superhub_inputs(&fix.memory),
        Some([100_000; 3]),
    );
    assert_eq!(
        exec.wcoj_triangle_dispatch_count(),
        1,
        "default RuntimeConfig must dispatch on super-hub (post-flip); got counter {}",
        exec.wcoj_triangle_dispatch_count()
    );
}

#[test]
fn default_runtime_falls_back_uniform() {
    // Uniform fixture — classifier rejects, binary-join handles
    // the rule. Counter stays 0.
    let Some(fix) = make_fix() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let exec = run_with_config(
        &fix,
        RuntimeConfig::default(),
        build_uniform_inputs(&fix.memory),
    );
    assert_eq!(
        exec.wcoj_triangle_dispatch_count(),
        0,
        "default RuntimeConfig on uniform must fall back to binary; got counter {}",
        exec.wcoj_triangle_dispatch_count()
    );
}

// =================================================================
// Existing explicit-off remains explicit-off.
// =================================================================

#[test]
fn explicit_off_beats_default_on_superhub() {
    // wcoj_triangle_dispatch=Some(false) is the existing
    // bench/test "off" knob. Default-on flip must NOT regress
    // its semantics — counter must stay 0 even on super-hub.
    let Some(fix) = make_fix() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let config = RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(false));
    let exec = run_with_config_and_cards(
        &fix,
        config,
        build_superhub_inputs(&fix.memory),
        Some([100_000; 3]),
    );
    assert_eq!(
        exec.wcoj_triangle_dispatch_count(),
        0,
        "explicit force-off must beat default-on; got counter {}",
        exec.wcoj_triangle_dispatch_count()
    );
}

// =================================================================
// Existing force-on still bypasses the classifier.
// =================================================================

#[test]
fn explicit_force_on_uniform_dispatches() {
    // wcoj_triangle_dispatch=Some(true) bypasses the classifier
    // — even uniform triangles dispatch. Pre-existing test
    // contract; locking it survives the default-on flip.
    let Some(fix) = make_fix() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let config = RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(true));
    let exec = run_with_config(&fix, config, build_uniform_inputs(&fix.memory));
    assert_eq!(
        exec.wcoj_triangle_dispatch_count(),
        1,
        "force-WCOJ must dispatch even on uniform; got counter {}",
        exec.wcoj_triangle_dispatch_count()
    );
}

// =================================================================
// Hard kill switch — beats every other flag.
// =================================================================

#[test]
fn disable_beats_default_on_superhub() {
    let Some(fix) = make_fix() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let config = RuntimeConfig::default().with_wcoj_triangle_dispatch_disabled(Some(true));
    let exec = run_with_config(&fix, config, build_superhub_inputs(&fix.memory));
    assert_eq!(
        exec.wcoj_triangle_dispatch_count(),
        0,
        "kill switch must beat default-on; got counter {}",
        exec.wcoj_triangle_dispatch_count()
    );
}

#[test]
fn disable_beats_force_on_superhub() {
    // The kill switch's headline use case: ops can pin WCOJ
    // dispatch off without touching app code or other env vars,
    // even when force-WCOJ is set.
    let Some(fix) = make_fix() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let config = RuntimeConfig::default()
        .with_wcoj_triangle_dispatch_disabled(Some(true))
        .with_wcoj_triangle_dispatch(Some(true));
    let exec = run_with_config(&fix, config, build_superhub_inputs(&fix.memory));
    assert_eq!(
        exec.wcoj_triangle_dispatch_count(),
        0,
        "kill switch must beat force-WCOJ; got counter {}",
        exec.wcoj_triangle_dispatch_count()
    );
}

#[test]
fn disable_beats_explicit_adaptive_on_superhub() {
    let Some(fix) = make_fix() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let config = RuntimeConfig::default()
        .with_wcoj_triangle_dispatch_disabled(Some(true))
        .with_wcoj_triangle_dispatch_adaptive(Some(true));
    let exec = run_with_config(&fix, config, build_superhub_inputs(&fix.memory));
    assert_eq!(
        exec.wcoj_triangle_dispatch_count(),
        0,
        "kill switch must beat explicit adaptive=Some(true); got counter {}",
        exec.wcoj_triangle_dispatch_count()
    );
}

#[test]
fn disable_some_false_does_not_disable() {
    // Some(false) on the disabled flag means "do NOT engage the
    // kill switch" — must NOT inhibit dispatch. Distinct from
    // Some(true). Default behavior (None) is also not kill, so
    // super-hub still dispatches via default-on adaptive.
    let Some(fix) = make_fix() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let config = RuntimeConfig::default().with_wcoj_triangle_dispatch_disabled(Some(false));
    let exec = run_with_config_and_cards(
        &fix,
        config,
        build_superhub_inputs(&fix.memory),
        Some([100_000; 3]),
    );
    assert_eq!(
        exec.wcoj_triangle_dispatch_count(),
        1,
        "disabled=Some(false) must NOT inhibit; got counter {}",
        exec.wcoj_triangle_dispatch_count()
    );
}

// =================================================================
// Adaptive=Some(true) is still respected (explicit opt-in path).
// =================================================================

#[test]
fn explicit_adaptive_on_uniform_falls_back() {
    // After default-on flip, explicit adaptive=Some(true) is
    // semantically equivalent to default — but it should still
    // be a valid no-op opt-in for users who want to be explicit.
    let Some(fix) = make_fix() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let config = RuntimeConfig::default().with_wcoj_triangle_dispatch_adaptive(Some(true));
    let exec = run_with_config(&fix, config, build_uniform_inputs(&fix.memory));
    assert_eq!(
        exec.wcoj_triangle_dispatch_count(),
        0,
        "explicit adaptive=Some(true) must reject uniform; got counter {}",
        exec.wcoj_triangle_dispatch_count()
    );
}

#[test]
fn explicit_adaptive_off_disables_default() {
    // Adaptive=Some(false) is "explicit-off for the adaptive
    // path". After default-on, this should turn the adaptive
    // path back off — counter == 0 even on super-hub.
    let Some(fix) = make_fix() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let config = RuntimeConfig::default().with_wcoj_triangle_dispatch_adaptive(Some(false));
    let exec = run_with_config(&fix, config, build_superhub_inputs(&fix.memory));
    assert_eq!(
        exec.wcoj_triangle_dispatch_count(),
        0,
        "adaptive=Some(false) must override default-on; got counter {}",
        exec.wcoj_triangle_dispatch_count()
    );
}

#[test]
fn _silence_unused_imports() {
    let _: BTreeMap<&str, ()> = BTreeMap::new();
}
