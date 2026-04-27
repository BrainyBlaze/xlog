#![allow(clippy::arc_with_non_send_sync)]
use std::sync::Arc;

use xlog_core::{symbol, MemoryBudget, RuntimeConfig, ScalarType, Schema};
use xlog_cuda::{CudaBuffer, CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_ir::{CompareOp, ConstValue, Expr};
use xlog_logic::Compiler;
use xlog_runtime::Executor;

fn has_cuda_device() -> bool {
    CudaDevice::new(0).is_ok()
}

fn create_executor_with_config(
    config: RuntimeConfig,
) -> Option<(Executor, Arc<CudaKernelProvider>)> {
    if !has_cuda_device() {
        return None;
    }

    let device = Arc::new(CudaDevice::new(0).ok()?);
    let budget = MemoryBudget::with_limit(1024 * 1024 * 1024);
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
    let provider = Arc::new(CudaKernelProvider::new(device, memory).ok()?);
    let executor = Executor::new_with_config(provider.clone(), config);

    Some((executor, provider))
}

fn device_row_count(
    provider: &CudaKernelProvider,
    rows: u64,
) -> xlog_cuda::memory::TrackedCudaSlice<u32> {
    let rows_u32 = u32::try_from(rows).expect("row count fits u32");
    let mut d_num_rows = provider.memory().alloc::<u32>(1).expect("alloc");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&[rows_u32], &mut d_num_rows)
        .expect("htod");
    d_num_rows
}

fn create_edge_buffer(provider: &CudaKernelProvider, edges: &[(u32, u32)]) -> CudaBuffer {
    let schema = Schema::new(vec![
        ("c0".to_string(), ScalarType::U32),
        ("c1".to_string(), ScalarType::U32),
    ]);

    if edges.is_empty() {
        let col0 = provider.memory().alloc::<u8>(0).expect("alloc");
        let col1 = provider.memory().alloc::<u8>(0).expect("alloc");
        let d_num_rows = device_row_count(provider, 0);
        return CudaBuffer::from_columns(vec![col0.into(), col1.into()], 0, d_num_rows, schema);
    }

    let col0_bytes: Vec<u8> = edges
        .iter()
        .flat_map(|(from, _)| from.to_le_bytes())
        .collect();
    let col1_bytes: Vec<u8> = edges.iter().flat_map(|(_, to)| to.to_le_bytes()).collect();

    let mut col0 = provider
        .memory()
        .alloc::<u8>(col0_bytes.len())
        .expect("alloc");
    let mut col1 = provider
        .memory()
        .alloc::<u8>(col1_bytes.len())
        .expect("alloc");

    provider
        .device()
        .inner()
        .htod_sync_copy_into(&col0_bytes, &mut col0)
        .expect("htod");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&col1_bytes, &mut col1)
        .expect("htod");

    let rows = edges.len() as u64;
    let d_num_rows = device_row_count(provider, rows);
    CudaBuffer::from_columns(vec![col0.into(), col1.into()], rows, d_num_rows, schema)
}

fn setup_executor_with_facts(
    executor: &mut Executor,
    compiler: &Compiler,
    facts: Vec<(&str, CudaBuffer)>,
) {
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }

    for (name, buffer) in facts {
        executor.store_mut().put(name, buffer);
    }
}

#[test]
fn test_executor_respects_max_iterations() {
    let config = {
        let mut config = RuntimeConfig::default();
        config.max_iterations = 1;
        config
    };

    let (mut executor, provider) = match create_executor_with_config(config) {
        Some(e) => e,
        None => {
            eprintln!("Skipping test: no CUDA device available");
            return;
        }
    };

    let source = r#"
        edge(1, 2).
        edge(2, 3).
        edge(3, 4).
        reach(X, Y) :- edge(X, Y).
        reach(X, Z) :- reach(X, Y), edge(Y, Z).
    "#;

    let mut compiler = Compiler::new();
    let plan = compiler.compile(source).expect("Compilation failed");
    assert!(plan.has_recursion(), "Expected recursive plan");

    let edge_buffer = create_edge_buffer(&provider, &[(1, 2), (2, 3), (3, 4)]);
    setup_executor_with_facts(&mut executor, &compiler, vec![("edge", edge_buffer)]);

    let err = match executor.execute_plan(&plan) {
        Ok(_) => panic!("expected iteration cap error"),
        Err(err) => err,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("iteration limit (1)"),
        "unexpected error message: {msg}"
    );
}

#[test]
fn test_executor_filter_with_column_column_compare_and_symbol() {
    let (executor, provider) = match create_executor_with_config(RuntimeConfig::default()) {
        Some(e) => e,
        None => {
            eprintln!("Skipping test: no CUDA device available");
            return;
        }
    };

    let symbol_hash = symbol::intern("sym");

    let schema = Schema::new(vec![
        ("a".to_string(), ScalarType::U32),
        ("b".to_string(), ScalarType::U32),
        ("s".to_string(), ScalarType::Symbol),
    ]);

    let buf = provider
        .create_buffer_from_u32_columns(
            &[&[1, 2, 3], &[1, 9, 3], &[symbol_hash, 7, symbol_hash]],
            schema,
        )
        .unwrap();

    let predicate = Expr::And(vec![
        Expr::Compare {
            left: Box::new(Expr::Column(0)),
            op: CompareOp::Eq,
            right: Box::new(Expr::Column(1)),
        },
        Expr::Compare {
            left: Box::new(Expr::Column(2)),
            op: CompareOp::Eq,
            right: Box::new(Expr::Const(ConstValue::Symbol("sym".to_string()))),
        },
    ]);

    let filtered = executor.execute_filter(&buf, &predicate).unwrap();
    let vals = provider.download_column::<u32>(&filtered, 0).unwrap();
    assert_eq!(vals, vec![1, 3]);
}

/// Clean-path coverage for the strict deterministic-Datalog D2H gate.
///
/// The v0.5.5 runtime is the *target* of the guard: most non-trivial
/// deterministic paths still fall back to host-side set algebra (this is
/// why the guard ships opt-in for now). The clean path that is provably
/// D2H-free today is a facts-only program: no rules, no queries, no
/// fixpoint iteration. `execute_plan` must:
///
///   * Engage the gate on entry (config flag is `true`).
///   * Run all strata without issuing a tracked D2H transfer.
///   * Restore the gate to its prior state on exit (RAII guard).
///   * Leave the violation counter at zero.
///
/// Once the GPU-native dedup/diff and deterministic join-materialize
/// kernels land, the broader Datalog surface should also satisfy this
/// contract; this test is the foothold that lets later PRs widen the
/// clean-path coverage.
#[test]
fn strict_deterministic_d2h_clean_path() {
    let mut config = RuntimeConfig::default();
    config.strict_deterministic_d2h = true;

    let (mut executor, provider) = match create_executor_with_config(config) {
        Some(e) => e,
        None => {
            eprintln!("Skipping test: no CUDA device available");
            return;
        }
    };

    // Single-column facts-only program. Multi-column EDB ingestion still
    // routes through a host-side dedup helper today; single-column
    // ingestion uses the GPU dedup path and is provably D2H-free.
    let source = r#"
        node(1).
        node(2).
        node(3).
    "#;

    let mut compiler = Compiler::new();
    let plan = compiler.compile(source).expect("Compilation failed");

    let node_schema = Schema::new(vec![("c0".to_string(), ScalarType::U32)]);
    let node_buffer = provider
        .create_buffer_from_u32_columns(&[&[1u32, 2, 3]], node_schema)
        .unwrap();
    setup_executor_with_facts(&mut executor, &compiler, vec![("node", node_buffer)]);

    let prior_gate = provider.strict_deterministic_d2h_enabled();
    provider.reset_deterministic_d2h_violations();
    executor
        .execute_plan(&plan)
        .expect("facts-only plan must succeed under strict deterministic D2H gate");

    // Gate must have been restored to its prior state by the RAII guard.
    assert_eq!(
        provider.strict_deterministic_d2h_enabled(),
        prior_gate,
        "gate guard did not restore the provider's prior state"
    );
    assert_eq!(
        provider.deterministic_d2h_violation_count(),
        0,
        "facts-only plan tripped the gate; this likely means a new runtime \
         path now issues a host fallback we did not previously have"
    );

    // Sanity: the EDB relation is intact and downloadable once the gate is off.
    let mut vals = provider
        .download_column::<u32>(executor.store().get("node").unwrap(), 0)
        .unwrap();
    vals.sort_unstable();
    assert_eq!(vals, vec![1, 2, 3]);
}

/// `execute_plan` must only reset the provider's
/// `deterministic_d2h_violation_count` when *this* call is the one that
/// engages the gate. If a caller has manually enabled the gate to
/// accumulate violations across a broader strict section,
/// `execute_plan` running with `RuntimeConfig::strict_deterministic_d2h
/// = true` must preserve that accumulated count rather than clobbering it.
#[test]
fn execute_plan_preserves_externally_engaged_violation_counter() {
    let mut config = RuntimeConfig::default();
    config.strict_deterministic_d2h = true;

    let (mut executor, provider) = match create_executor_with_config(config) {
        Some(e) => e,
        None => {
            eprintln!("Skipping test: no CUDA device available");
            return;
        }
    };

    // Pre-accumulate a violation outside any execute_plan call.
    provider.reset_deterministic_d2h_violations();
    provider.enable_strict_deterministic_d2h();
    let schema = Schema::new(vec![("c0".to_string(), ScalarType::U32)]);
    let probe = provider
        .create_buffer_from_u32_columns(&[&[1u32, 2]], schema)
        .unwrap();
    let _ = provider.download_column::<u32>(&probe, 0); // intentional violation
    let pre_count = provider.deterministic_d2h_violation_count();
    assert_eq!(pre_count, 1);

    // Run a clean-path program. With the previous always-reset behavior,
    // `pre_count` would be wiped to 0; with the transition-only reset,
    // the counter is preserved across the call.
    let source = r#"
        node(1).
        node(2).
    "#;
    let mut compiler = Compiler::new();
    let plan = compiler.compile(source).expect("compile");
    let node_schema = Schema::new(vec![("c0".to_string(), ScalarType::U32)]);
    let node_buffer = provider
        .create_buffer_from_u32_columns(&[&[1u32, 2]], node_schema)
        .unwrap();
    setup_executor_with_facts(&mut executor, &compiler, vec![("node", node_buffer)]);

    executor
        .execute_plan(&plan)
        .expect("clean plan must succeed under strict gate");
    assert!(
        provider.strict_deterministic_d2h_enabled(),
        "externally engaged gate must remain on after execute_plan returns"
    );
    assert_eq!(
        provider.deterministic_d2h_violation_count(),
        pre_count,
        "execute_plan clobbered an externally accumulated violation count"
    );

    provider.disable_strict_deterministic_d2h();
}

/// The gate must default to off — `RuntimeConfig::default()` does not
/// engage it, and a program that *would* violate while strict therefore
/// continues to succeed unchanged.
#[test]
fn default_runtime_does_not_engage_gate() {
    let (mut executor, provider) = match create_executor_with_config(RuntimeConfig::default()) {
        Some(e) => e,
        None => {
            eprintln!("Skipping test: no CUDA device available");
            return;
        }
    };

    let source = r#"
        edge(1, 2).
        edge(2, 3).
        node(1). node(2). node(3).
        has_in(X) :- edge(_, X).
        no_in(X) :- node(X), !has_in(X).
        ?- no_in(X).
    "#;

    let mut compiler = Compiler::new();
    let plan = match compiler.compile(source) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Skipping: source did not compile: {}", e);
            return;
        }
    };

    let edge_buffer = create_edge_buffer(&provider, &[(1, 2), (2, 3)]);
    let node_schema = Schema::new(vec![("c0".to_string(), ScalarType::U32)]);
    let node_buffer = provider
        .create_buffer_from_u32_columns(&[&[1u32, 2, 3]], node_schema)
        .unwrap();
    setup_executor_with_facts(
        &mut executor,
        &compiler,
        vec![("edge", edge_buffer), ("node", node_buffer)],
    );

    provider.reset_deterministic_d2h_violations();
    executor
        .execute_plan(&plan)
        .expect("default config must let known-violator programs run");
    assert_eq!(provider.deterministic_d2h_violation_count(), 0);
    assert!(!provider.strict_deterministic_d2h_enabled());
}

/// Negative-direction coverage: a program that the v0.5.5 runtime still
/// services via host fallback (stratified negation routes through `diff`)
/// must surface a violation when the strict gate is enabled. This pins the
/// gate as a regression detector — if a future change moves negation onto
/// a GPU-native diff, this test should be updated alongside that work, not
/// silently passed.
#[test]
fn strict_deterministic_d2h_known_violator_is_detected() {
    let mut config = RuntimeConfig::default();
    config.strict_deterministic_d2h = true;

    let (mut executor, provider) = match create_executor_with_config(config) {
        Some(e) => e,
        None => {
            eprintln!("Skipping test: no CUDA device available");
            return;
        }
    };

    // Stratified negation lowers to set difference (`diff`) in the runtime,
    // and `diff` still falls back to host-side set algebra.
    let source = r#"
        edge(1, 2).
        edge(2, 3).
        edge(3, 4).
        node(1). node(2). node(3). node(4).
        has_in(X) :- edge(_, X).
        no_in(X) :- node(X), !has_in(X).
        ?- no_in(X).
    "#;

    let mut compiler = Compiler::new();
    let plan = match compiler.compile(source) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Skipping: source did not compile: {}", e);
            return;
        }
    };

    let edge_buffer = create_edge_buffer(&provider, &[(1, 2), (2, 3), (3, 4)]);
    let node_schema = Schema::new(vec![("c0".to_string(), ScalarType::U32)]);
    let node_buffer = provider
        .create_buffer_from_u32_columns(&[&[1u32, 2, 3, 4]], node_schema)
        .unwrap();
    setup_executor_with_facts(
        &mut executor,
        &compiler,
        vec![("edge", edge_buffer), ("node", node_buffer)],
    );

    provider.reset_deterministic_d2h_violations();
    let res = executor.execute_plan(&plan);
    assert!(
        res.is_err(),
        "expected gate to surface a violation for negation-via-diff"
    );
    assert!(
        provider.deterministic_d2h_violation_count() >= 1,
        "violation counter did not increment for known violator"
    );
}
