//! ChainJoin production route fallback identity and timing checks.
//!
//! The production route emits `RirNode::ChainJoin`. This test
//! proves the end-to-end path has the required fallback identity:
//! default-on chain dispatch and env-disabled fallback produce the same
//! rows, while the dispatch counter distinguishes the paths.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use cudarc::driver::sys;
use xlog_core::{MemoryBudget, RuntimeConfig, ScalarType, Schema};
use xlog_cuda::device_runtime::{LogRecord, LoggingSink, SinkError, StreamPool, XlogDeviceRuntime};
use xlog_cuda::memory::CudaBuffer;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_ir::{ExecutionPlan, ProjectExpr, RirNode};
use xlog_logic::Compiler;
use xlog_runtime::Executor;

struct DiscardSink;
impl LoggingSink for DiscardSink {
    fn emit(&self, _record: LogRecord) -> Result<(), SinkError> {
        Ok(())
    }
}

struct RuntimeBackedFixture {
    _device: Arc<CudaDevice>,
    _runtime: Arc<XlogDeviceRuntime>,
    memory: Arc<GpuMemoryManager>,
    provider: Arc<CudaKernelProvider>,
    _pool: Arc<StreamPool>,
}

fn make_runtime_backed_fixture() -> Option<RuntimeBackedFixture> {
    make_runtime_backed_fixture_with_budget(64 * 1024 * 1024)
}

fn make_runtime_backed_fixture_with_budget(budget_bytes: usize) -> Option<RuntimeBackedFixture> {
    let provider = Arc::new(
        xlog_cuda::CudaProviderBuilder::new(0, MemoryBudget::with_limit(budget_bytes as u64))
            .with_logging_sink(Arc::new(DiscardSink) as Arc<dyn LoggingSink>)
            .build()
            .ok()?,
    );
    let device = Arc::clone(provider.device());
    let memory = Arc::clone(provider.memory());
    let runtime = Arc::clone(memory.runtime()?);
    let pool = Arc::clone(runtime.stream_pool());
    Some(RuntimeBackedFixture {
        _device: device,
        _runtime: runtime,
        memory,
        provider,
        _pool: pool,
    })
}

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
    assert_eq!(buf.arity(), 2);
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
    let mut out: Vec<(u32, u32)> = (0..n)
        .map(|i| {
            (
                u32::from_le_bytes(col0_bytes[i * 4..i * 4 + 4].try_into().unwrap()),
                u32::from_le_bytes(col1_bytes[i * 4..i * 4 + 4].try_into().unwrap()),
            )
        })
        .collect();
    out.sort();
    out.dedup();
    out
}

const CHAIN_SOURCE: &str = r#"
    pred a(u32, u32).
    pred b(u32, u32).
    pred out(u32, u32).
    out(X, Y) :- a(X, Z), b(Z, Y).
"#;

fn chain_fixture() -> BTreeMap<&'static str, Vec<(u32, u32)>> {
    let mut m = BTreeMap::new();
    m.insert("a", (0..128u32).map(|i| (10_000 + i, i)).collect());
    m.insert("b", (0..128u32).map(|i| (i, 20_000 + i)).collect());
    m
}

fn chain_fixture_n(n: u32) -> BTreeMap<&'static str, Vec<(u32, u32)>> {
    let mut m = BTreeMap::new();
    m.insert("a", (0..n).map(|i| (10_000_000 + i, i)).collect());
    m.insert("b", (0..n).map(|i| (i, 20_000_000 + i)).collect());
    m
}

fn chain_profiler_trace_path() -> PathBuf {
    let evidence_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/evidence");
    for dir_entry in fs::read_dir(&evidence_root).expect("read evidence root") {
        let dir = dir_entry.expect("read evidence directory entry").path();
        if !dir.is_dir() {
            continue;
        }
        let Some(dir_name) = dir.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !dir_name.ends_with("pre-profiler-trace") {
            continue;
        }
        for file_entry in fs::read_dir(&dir).expect("read profiler trace directory") {
            let file = file_entry.expect("read profiler trace entry").path();
            let Some(file_name) = file.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if file_name.ends_with("trace-50.jsonl") {
                return file;
            }
        }
    }
    panic!(
        "chain profiler trace fixture not found under {}",
        evidence_root.display()
    );
}

fn load_chain_profiler_trace_subset(limit: usize) -> (Vec<u32>, u128) {
    let text = fs::read_to_string(chain_profiler_trace_path()).expect("read profiler trace");
    let mut rows = Vec::with_capacity(limit);
    let mut baseline_ns = 0u128;
    for line in text.lines() {
        let value: serde_json::Value =
            serde_json::from_str(line).expect("parse profiler JSONL row");
        if value.get("kind").and_then(|v| v.as_str()) != Some("xlog_evaluate_step") {
            continue;
        }
        if value.get("max_body_len").and_then(|v| v.as_u64()) != Some(2) {
            continue;
        }
        let committed_rows = value
            .get("committed_rows")
            .and_then(|v| v.as_u64())
            .expect("committed_rows") as u32;
        let evaluate_ns = value
            .get("evaluate_ns")
            .and_then(|v| v.as_u64())
            .expect("evaluate_ns") as u128;
        rows.push(committed_rows);
        baseline_ns += evaluate_ns;
        if rows.len() == limit {
            break;
        }
    }
    assert!(
        rows.len() >= limit,
        "profiler trace must contain at least {limit} chain-shaped invocations"
    );
    (rows, baseline_ns)
}

fn run_chain(
    provider: Arc<CudaKernelProvider>,
    memory: &Arc<GpuMemoryManager>,
    inputs: &BTreeMap<&str, Vec<(u32, u32)>>,
) -> Executor {
    let (plan, mut executor) = prepare_chain_executor(provider, memory, inputs);
    executor.execute_plan(&plan).expect("execute chain");
    executor
}

fn prepare_chain_executor(
    provider: Arc<CudaKernelProvider>,
    memory: &Arc<GpuMemoryManager>,
    inputs: &BTreeMap<&str, Vec<(u32, u32)>>,
) -> (ExecutionPlan, Executor) {
    let mut compiler = Compiler::new();
    let plan = compiler.compile(CHAIN_SOURCE).expect("compile chain");
    let mut executor = Executor::new_with_config(provider, RuntimeConfig::default());
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    for (name, rows) in inputs {
        executor.put_relation(name, upload_binary_u32(memory, rows));
    }
    (plan, executor)
}

fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn invalidate_specialized_chain_projection(plan: &mut ExecutionPlan) {
    let mut matches = 0;
    for rule in plan.rules_by_scc.iter_mut().flatten() {
        if let RirNode::ChainJoin { output_columns, .. } = &mut rule.body {
            matches += 1;
            output_columns[0] = ProjectExpr::Column(usize::MAX);
        }
    }
    assert_eq!(
        matches, 1,
        "fixture must contain exactly one promoted ChainJoin"
    );
}

fn profile_op_count(executor: &Executor, op_name: &str, output_rows: u64) -> usize {
    executor
        .execution_stats(output_rows)
        .strata
        .iter()
        .flat_map(|stratum| &stratum.ops)
        .filter(|op| op.op_name == op_name)
        .count()
}

fn restore_env(name: &str, value: Option<String>) {
    unsafe {
        match value {
            Some(value) => std::env::set_var(name, value),
            None => std::env::remove_var(name),
        }
    }
}

#[test]
fn chain_dispatch_default_on_matches_env_disabled_fallback() {
    let _guard = env_lock().lock().expect("chain env lock poisoned");
    let old = std::env::var("XLOG_WCOJ_CHAIN_ENABLE").ok();
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = chain_fixture();

    unsafe {
        std::env::set_var("XLOG_WCOJ_CHAIN_ENABLE", "0");
    }
    let fallback = run_chain(Arc::clone(&fix.provider), &fix.memory, &inputs);
    let fallback_rows: BTreeSet<(u32, u32)> = download_pairs(
        fallback
            .store()
            .get("out")
            .expect("fallback out relation must exist"),
    )
    .into_iter()
    .collect();
    assert_eq!(fallback.chain_dispatch_count(), 0);
    let fallback_profile = fallback.execution_stats(fallback_rows.len() as u64);
    assert_eq!(fallback_profile.chain_fallback_scan_equivalents, 0);
    assert_eq!(fallback_profile.chain_fallback_filter_equivalents, 0);

    unsafe {
        std::env::remove_var("XLOG_WCOJ_CHAIN_ENABLE");
    }
    let dispatched = run_chain(Arc::clone(&fix.provider), &fix.memory, &inputs);
    let dispatched_rows: BTreeSet<(u32, u32)> = download_pairs(
        dispatched
            .store()
            .get("out")
            .expect("dispatched out relation must exist"),
    )
    .into_iter()
    .collect();

    unsafe {
        match old {
            Some(v) => std::env::set_var("XLOG_WCOJ_CHAIN_ENABLE", v),
            None => std::env::remove_var("XLOG_WCOJ_CHAIN_ENABLE"),
        }
    }

    assert_eq!(dispatched.chain_dispatch_count(), 1);
    let dispatched_profile = dispatched.execution_stats(dispatched_rows.len() as u64);
    assert_eq!(dispatched_profile.chain_fallback_scan_equivalents, 2);
    assert_eq!(dispatched_profile.chain_fallback_filter_equivalents, 0);
    assert_eq!(dispatched_rows.len(), 128);
    assert_eq!(dispatched_rows, fallback_rows);
}

#[test]
fn matched_chain_projection_error_declines_to_physical_fallback_without_equivalents() {
    let _guard = env_lock().lock().expect("chain env lock poisoned");
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let old_chain = std::env::var("XLOG_WCOJ_CHAIN_ENABLE").ok();
    let old_strict = std::env::var("XLOG_WCOJ_STRICT").ok();
    unsafe {
        std::env::remove_var("XLOG_WCOJ_CHAIN_ENABLE");
        std::env::remove_var("XLOG_WCOJ_STRICT");
    }

    let inputs = chain_fixture();
    let (mut plan, mut executor) =
        prepare_chain_executor(Arc::clone(&fix.provider), &fix.memory, &inputs);
    invalidate_specialized_chain_projection(&mut plan);
    executor.set_profiling(true);
    let result = executor.execute_plan(&plan);

    restore_env("XLOG_WCOJ_CHAIN_ENABLE", old_chain);
    restore_env("XLOG_WCOJ_STRICT", old_strict);
    result.expect("non-strict specialization error must execute embedded fallback");

    let rows: BTreeSet<(u32, u32)> = download_pairs(
        executor
            .store()
            .get("out")
            .expect("fallback out relation must exist"),
    )
    .into_iter()
    .collect();
    let expected: BTreeSet<(u32, u32)> = (0..128u32).map(|i| (10_000 + i, 20_000 + i)).collect();
    assert_eq!(rows, expected);
    assert_eq!(executor.chain_dispatch_count(), 0);
    assert_eq!(executor.wcoj_error_decline_count(), 1);
    let profile = executor.execution_stats(rows.len() as u64);
    assert_eq!(profile.chain_fallback_scan_equivalents, 0);
    assert_eq!(profile.chain_fallback_filter_equivalents, 0);
    assert_eq!(profile_op_count(&executor, "scan", rows.len() as u64), 2);
    assert_eq!(profile_op_count(&executor, "join", rows.len() as u64), 1);
}

#[test]
fn matched_chain_projection_error_propagates_in_strict_mode_without_equivalents() {
    let _guard = env_lock().lock().expect("chain env lock poisoned");
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let old_chain = std::env::var("XLOG_WCOJ_CHAIN_ENABLE").ok();
    let old_strict = std::env::var("XLOG_WCOJ_STRICT").ok();
    unsafe {
        std::env::remove_var("XLOG_WCOJ_CHAIN_ENABLE");
        std::env::set_var("XLOG_WCOJ_STRICT", "1");
    }

    let inputs = chain_fixture();
    let (mut plan, mut executor) =
        prepare_chain_executor(Arc::clone(&fix.provider), &fix.memory, &inputs);
    invalidate_specialized_chain_projection(&mut plan);
    let result = executor.execute_plan(&plan);

    restore_env("XLOG_WCOJ_CHAIN_ENABLE", old_chain);
    restore_env("XLOG_WCOJ_STRICT", old_strict);
    assert!(
        result.is_err(),
        "strict specialization error must propagate"
    );
    assert_eq!(executor.chain_dispatch_count(), 0);
    assert_eq!(executor.wcoj_error_decline_count(), 1);
    let profile = executor.execution_stats(0);
    assert_eq!(profile.chain_fallback_scan_equivalents, 0);
    assert_eq!(profile.chain_fallback_filter_equivalents, 0);
}

fn timed_loaded_chain_runs(
    provider: Arc<CudaKernelProvider>,
    memory: &Arc<GpuMemoryManager>,
    inputs: &BTreeMap<&str, Vec<(u32, u32)>>,
    iterations: u32,
) -> (Duration, u64) {
    let (plan, mut executor) = prepare_chain_executor(provider, memory, inputs);
    let start_dispatches = executor.chain_dispatch_count();
    let start = Instant::now();
    for _ in 0..iterations {
        executor.store_mut().remove("out");
        executor.execute_plan(&plan).expect("execute loaded chain");
    }
    (
        start.elapsed(),
        executor
            .chain_dispatch_count()
            .saturating_sub(start_dispatches),
    )
}

#[test]
#[ignore = "performance smoke; run manually for chain timing evidence"]
fn chain_dispatch_timing_smoke_sorted_threshold_cell() {
    let _guard = env_lock().lock().expect("chain env lock poisoned");
    let old = std::env::var("XLOG_WCOJ_CHAIN_ENABLE").ok();
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = chain_fixture_n(2_000);
    let iterations = 20;

    unsafe {
        std::env::set_var("XLOG_WCOJ_CHAIN_ENABLE", "0");
    }
    let (fallback_elapsed, fallback_dispatches) =
        timed_loaded_chain_runs(Arc::clone(&fix.provider), &fix.memory, &inputs, iterations);

    unsafe {
        std::env::remove_var("XLOG_WCOJ_CHAIN_ENABLE");
    }
    let (chain_elapsed, chain_dispatches) =
        timed_loaded_chain_runs(Arc::clone(&fix.provider), &fix.memory, &inputs, iterations);

    unsafe {
        match old {
            Some(v) => std::env::set_var("XLOG_WCOJ_CHAIN_ENABLE", v),
            None => std::env::remove_var("XLOG_WCOJ_CHAIN_ENABLE"),
        }
    }

    let ratio = fallback_elapsed.as_secs_f64() / chain_elapsed.as_secs_f64();
    eprintln!(
        "CHAIN_DISPATCH_TIMING sorted_threshold n=2000 iterations={} fallback_ms={:.3} chain_ms={:.3} ratio={:.6} fallback_dispatches={} chain_dispatches={}",
        iterations,
        fallback_elapsed.as_secs_f64() * 1000.0,
        chain_elapsed.as_secs_f64() * 1000.0,
        ratio,
        fallback_dispatches,
        chain_dispatches
    );
    assert_eq!(fallback_dispatches, 0);
    assert_eq!(chain_dispatches, iterations as u64);
}

#[test]
#[ignore = "acceptance timing; run manually for synthetic large-chain gate"]
fn chain_dispatch_timing_synthetic_977k() {
    let _guard = env_lock().lock().expect("chain env lock poisoned");
    let old = std::env::var("XLOG_WCOJ_CHAIN_ENABLE").ok();
    let Some(fix) = make_runtime_backed_fixture_with_budget(512 * 1024 * 1024) else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = chain_fixture_n(977_000);
    let iterations = 3;

    unsafe {
        std::env::set_var("XLOG_WCOJ_CHAIN_ENABLE", "0");
    }
    let (fallback_elapsed, fallback_dispatches) =
        timed_loaded_chain_runs(Arc::clone(&fix.provider), &fix.memory, &inputs, iterations);

    unsafe {
        std::env::remove_var("XLOG_WCOJ_CHAIN_ENABLE");
    }
    let (chain_elapsed, chain_dispatches) =
        timed_loaded_chain_runs(Arc::clone(&fix.provider), &fix.memory, &inputs, iterations);

    unsafe {
        match old {
            Some(v) => std::env::set_var("XLOG_WCOJ_CHAIN_ENABLE", v),
            None => std::env::remove_var("XLOG_WCOJ_CHAIN_ENABLE"),
        }
    }

    let ratio = fallback_elapsed.as_secs_f64() / chain_elapsed.as_secs_f64();
    eprintln!(
        "CHAIN_DISPATCH_TIMING synthetic_977k n=977000 iterations={} fallback_ms={:.3} chain_ms={:.3} ratio={:.6} fallback_dispatches={} chain_dispatches={}",
        iterations,
        fallback_elapsed.as_secs_f64() * 1000.0,
        chain_elapsed.as_secs_f64() * 1000.0,
        ratio,
        fallback_dispatches,
        chain_dispatches
    );
    assert_eq!(fallback_dispatches, 0);
    assert_eq!(chain_dispatches, iterations as u64);
    assert!(
        ratio >= 1.5,
        "synthetic large-chain gate requires ratio >= 1.5x, got {ratio:.6}x"
    );
}

#[test]
#[ignore = "acceptance timing; run manually for profiler-trace gate"]
fn chain_dispatch_timing_profiler_trace_subset_128() {
    let _guard = env_lock().lock().expect("chain env lock poisoned");
    let old = std::env::var("XLOG_WCOJ_CHAIN_ENABLE").ok();
    let Some(fix) = make_runtime_backed_fixture_with_budget(256 * 1024 * 1024) else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let (rows, baseline_ns) = load_chain_profiler_trace_subset(128);

    unsafe {
        std::env::remove_var("XLOG_WCOJ_CHAIN_ENABLE");
    }
    let mut chain_elapsed = Duration::from_nanos(0);
    let mut dispatches = 0u64;
    let mut output_rows = 0u64;
    for n in rows {
        let inputs = chain_fixture_n(n);
        let (plan, mut executor) =
            prepare_chain_executor(Arc::clone(&fix.provider), &fix.memory, &inputs);
        let start = Instant::now();
        executor.execute_plan(&plan).expect("execute trace chain");
        chain_elapsed += start.elapsed();
        dispatches += executor.chain_dispatch_count();
        output_rows += fix
            .provider
            .device_row_count(executor.store().get("out").expect("out relation"))
            .expect("out row count") as u64;
    }

    unsafe {
        match old {
            Some(v) => std::env::set_var("XLOG_WCOJ_CHAIN_ENABLE", v),
            None => std::env::remove_var("XLOG_WCOJ_CHAIN_ENABLE"),
        }
    }

    let baseline_ms = baseline_ns as f64 / 1_000_000.0;
    let chain_ms = chain_elapsed.as_secs_f64() * 1000.0;
    let ratio = baseline_ms / chain_ms;
    eprintln!(
        "CHAIN_DISPATCH_TIMING profiler_trace_subset invocations=128 baseline_ms={:.3} chain_ms={:.3} ratio={:.6} dispatches={} output_rows={}",
        baseline_ms, chain_ms, ratio, dispatches, output_rows
    );
    assert_eq!(dispatches, 128);
    assert!(
        ratio >= 2.0,
        "profiler-trace gate requires ratio >= 2.0x, got {ratio:.6}x"
    );
}
