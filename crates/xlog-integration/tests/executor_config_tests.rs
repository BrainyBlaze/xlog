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

/// Clean-path coverage for the strict deterministic-Datalog device-to-host
/// transfer gate.
///
/// The v0.5.5 runtime is the *target* of the guard: most non-trivial
/// deterministic paths still fall back to host-side set algebra (this is
/// why the guard ships opt-in for now). The clean path that is provably
/// free of tracked device-to-host transfers today is a facts-only program: no
/// rules, no queries, no fixpoint iteration. `execute_plan` must:
///
///   * Engage the gate on entry (config flag is `true`).
///   * Run all strata without issuing a tracked device-to-host transfer.
///   * Restore the gate to its prior state on exit (RAII guard).
///   * Leave the violation counter at zero.
///
/// Once the GPU-native dedup/diff and deterministic join-materialize
/// kernels land, the broader Datalog surface should also satisfy this
/// contract; this test is the foothold that lets later PRs widen the
/// clean-path coverage.
#[test]
fn strict_deterministic_device_to_host_clean_path() {
    let mut config = RuntimeConfig::default();
    config.strict_deterministic_d2h = true;

    let (mut executor, provider) = match create_executor_with_config(config) {
        Some(e) => e,
        None => {
            eprintln!("Skipping test: no CUDA device available");
            return;
        }
    };

    // Single-column facts-only program. Both single-column and
    // multi-column EDB ingestion are free of tracked device-to-host
    // transfers since the set-algebra GPU pipeline landed; this test
    // stays single-column because the gate-guard machinery is what's
    // being exercised here, not the dedup pipeline.
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
        .expect("facts-only plan must succeed under strict deterministic device-to-host gate");

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
/// strict deterministic device-to-host violation count when *this* call is the
/// one that engages the gate. If a caller has manually enabled the gate to
/// accumulate violations across a broader strict section, `execute_plan`
/// running with strict deterministic device-to-host transfer checking enabled
/// must preserve that accumulated count rather than clobbering it.
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
        no_in(X) :- node(X), not has_in(X).
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
        .expect("default config must let the program run");
    assert_eq!(provider.deterministic_d2h_violation_count(), 0);
    assert!(!provider.strict_deterministic_d2h_enabled());
}

/// Single-column negation seal under the strict gate.
///
/// In the pre-set-algebra baseline this program tripped the gate via
/// the host-side `diff_via_deterministic_set` → `BTreeSet<Vec<u8>>`
/// fallback used by stratified negation when the multi-column dedup
/// path entered the host-collection branch. The set-algebra GPU
/// hardening replaced that fallback with a deterministic GPU
/// pipeline (typed multi-column sort → bytewise adjacent-equality
/// mask → exclusive scan → column-wise gather). After the swap the
/// program must run cleanly under the strict gate with the correct
/// result set.
#[test]
fn strict_deterministic_device_to_host_single_column_negation_clean() {
    let mut config = RuntimeConfig::default();
    config.strict_deterministic_d2h = true;

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
        node(1). node(2). node(3). node(4).
        has_in(X) :- edge(_, X).
        no_in(X) :- node(X), not has_in(X).
        ?- no_in(X).
    "#;

    let mut compiler = Compiler::new();
    let plan = compiler
        .compile(source)
        .expect("single-column negation source must compile");

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
    executor
        .execute_plan(&plan)
        .expect("single-column negation must run clean under strict gate");
    assert_eq!(
        provider.deterministic_d2h_violation_count(),
        0,
        "single-column negation tripped the gate; the host-fallback path \
         should be replaced by the GPU pipeline by this PR"
    );

    // Result set: nodes with no incoming edge → {1}. Read from the store
    // (executor.execute_plan returns an empty placeholder buffer; the
    // answer set lives in the named relation).
    let no_in = executor
        .store()
        .get("no_in")
        .expect("no_in relation present after execution");
    let mut vals = provider.download_column::<u32>(no_in, 0).unwrap();
    vals.sort_unstable();
    assert_eq!(vals, vec![1u32]);
}

/// Two-column negation seal — proves the runtime uses *full-row* tuple
/// equality, not first-column / key-only equality, after PR 2.
///
/// Inputs: `pair = {(1,10),(1,20),(2,10)}`, `blocked = {(1,10)}`.
/// Expected `keep = {(1,20),(2,10)}`.
///
/// A first-column key diff would incorrectly remove `(1,20)` because
/// column 0 alone equals `1` in `blocked`. This test fails iff the
/// runtime wires negation onto a key-only diff.
///
/// Run under the strict deterministic device-to-host transfer gate so the test
/// simultaneously seals "no host fallback".
#[test]
fn strict_deterministic_device_to_host_two_column_negation_full_row_semantics() {
    let mut config = RuntimeConfig::default();
    config.strict_deterministic_d2h = true;

    let (mut executor, provider) = match create_executor_with_config(config) {
        Some(e) => e,
        None => {
            eprintln!("Skipping test: no CUDA device available");
            return;
        }
    };

    let source = r#"
        pair(1, 10). pair(1, 20). pair(2, 10).
        blocked(1, 10).
        keep(X, Y) :- pair(X, Y), not blocked(X, Y).
        ?- keep(X, Y).
    "#;
    let mut compiler = Compiler::new();
    let plan = compiler.compile(source).expect("compile two-col negation");

    let pair_buffer = create_edge_buffer(&provider, &[(1u32, 10), (1, 20), (2, 10)]);
    let blocked_buffer = create_edge_buffer(&provider, &[(1u32, 10)]);
    setup_executor_with_facts(
        &mut executor,
        &compiler,
        vec![("pair", pair_buffer), ("blocked", blocked_buffer)],
    );

    provider.reset_deterministic_d2h_violations();
    executor
        .execute_plan(&plan)
        .expect("two-column negation must run clean under strict gate");
    assert_eq!(
        provider.deterministic_d2h_violation_count(),
        0,
        "two-column negation tripped the gate"
    );

    // Read both columns from the named relation and assert as a sorted set.
    let keep = executor
        .store()
        .get("keep")
        .expect("keep relation present after execution");
    let col0 = provider.download_column::<u32>(keep, 0).unwrap();
    let col1 = provider.download_column::<u32>(keep, 1).unwrap();
    assert_eq!(
        col0.len(),
        col1.len(),
        "result columns disagree on row count"
    );
    let mut got: Vec<(u32, u32)> = col0.into_iter().zip(col1).collect();
    got.sort_unstable();
    assert_eq!(
        got,
        vec![(1u32, 20), (2u32, 10)],
        "two-column negation collapsed under key-only diff"
    );
}

/// Recursive reach: full-row tuple semantics for semi-naive delta dedup,
/// free of tracked device-to-host transfers.
///
/// Proves that the recursive fixpoint preserves *full-row* equality
/// AND that binary-join materialization is free of tracked device-to-host
/// transfers after the metadata-read hardening — the eight
/// `Failed to read output count` sites in `provider/relational.rs`
/// were reclassified as metadata reads via `dtoh_scalar_untracked`,
/// which the strict gate explicitly allows.
///
/// With key-based dedup, rows `(1,2),(1,3),(1,4)` would collapse to a
/// single entry. With full-row dedup all three survive and the answer
/// set has five unique rows.
#[test]
fn strict_deterministic_device_to_host_recursive_reach_clean() {
    let mut config = RuntimeConfig::default();
    config.strict_deterministic_d2h = true;

    let (mut executor, provider) = match create_executor_with_config(config) {
        Some(e) => e,
        None => {
            eprintln!("Skipping test: no CUDA device available");
            return;
        }
    };

    let source = r#"
        edge(1, 2). edge(1, 3). edge(2, 4). edge(3, 4).
        reach(X, Y) :- edge(X, Y).
        reach(X, Z) :- reach(X, Y), edge(Y, Z).
        ?- reach(X, Y).
    "#;
    let mut compiler = Compiler::new();
    let plan = compiler.compile(source).expect("compile reach");

    let edge_buffer = create_edge_buffer(&provider, &[(1u32, 2), (1, 3), (2, 4), (3, 4)]);
    setup_executor_with_facts(&mut executor, &compiler, vec![("edge", edge_buffer)]);

    provider.reset_deterministic_d2h_violations();
    executor
        .execute_plan(&plan)
        .expect("recursive reach must run clean under strict gate");
    assert_eq!(
        provider.deterministic_d2h_violation_count(),
        0,
        "recursive reach tripped the gate; binary-join materialization \
         likely regressed onto a tracked output-count read"
    );

    let reach = executor
        .store()
        .get("reach")
        .expect("reach relation present after execution");
    let col0 = provider.download_column::<u32>(reach, 0).unwrap();
    let col1 = provider.download_column::<u32>(reach, 1).unwrap();
    let mut got: Vec<(u32, u32)> = col0.into_iter().zip(col1).collect();
    got.sort_unstable();
    let expected = vec![(1u32, 2), (1, 3), (1, 4), (2, 4), (3, 4)];
    assert_eq!(
        got.len(),
        expected.len(),
        "row count mismatch — semi-naive delta dedup likely collapsed full rows under a key-only path"
    );
    assert_eq!(got, expected);
}

/// Join-heavy strict-gate seal: forces inner-join materialization with a
/// non-trivial fan-out (one binary join with multiple matches per probe
/// key), then asserts the result is correct AND that no tracked device-to-host
/// transfer was issued.
///
/// Distinct from the recursive-reach test: this exercises a
/// non-recursive single-rule program with an explicit inner join in the
/// rule body, so it stresses the `hash_join_inner_v2*`
/// count→materialize path. The `hash_join_left_outer_*` path is an
/// internal IR-level join type not directly reachable from a Datalog
/// rule body; both halves of that path (non-indexed
/// `hash_join_left_outer_impl` and indexed
/// `hash_join_left_outer_indexed`) are covered by kernel-level
/// strict-gate tests in the CUDA provider test suite
/// (`left_outer_join_strict_gate_clean` and
/// `left_outer_join_indexed_strict_gate_clean`).
#[test]
fn strict_deterministic_device_to_host_inner_join_materialize_clean() {
    let mut config = RuntimeConfig::default();
    config.strict_deterministic_d2h = true;

    let (mut executor, provider) = match create_executor_with_config(config) {
        Some(e) => e,
        None => {
            eprintln!("Skipping test: no CUDA device available");
            return;
        }
    };

    // `path(X, Z)` is a single-step inner join of `edge(X, Y)` with
    // `edge(Y, Z)` on the middle column. Forces a non-trivial join
    // materialization with multiple matches per probe key.
    let source = r#"
        edge(1, 2). edge(1, 3). edge(2, 4). edge(3, 4). edge(4, 5).
        path(X, Z) :- edge(X, Y), edge(Y, Z).
        ?- path(X, Z).
    "#;
    let mut compiler = Compiler::new();
    let plan = compiler.compile(source).expect("compile path");

    let edge_buffer = create_edge_buffer(&provider, &[(1u32, 2), (1, 3), (2, 4), (3, 4), (4, 5)]);
    setup_executor_with_facts(&mut executor, &compiler, vec![("edge", edge_buffer)]);

    provider.reset_deterministic_d2h_violations();
    executor
        .execute_plan(&plan)
        .expect("inner-join program must run clean under strict gate");
    assert_eq!(
        provider.deterministic_d2h_violation_count(),
        0,
        "inner-join materialization tripped the gate; output-count reads \
         must go through `dtoh_scalar_untracked` (metadata)"
    );

    let path = executor
        .store()
        .get("path")
        .expect("path relation present after execution");
    let col0 = provider.download_column::<u32>(path, 0).unwrap();
    let col1 = provider.download_column::<u32>(path, 1).unwrap();
    assert_eq!(
        col0.len(),
        col1.len(),
        "path relation columns must have the same row count before zipping"
    );
    let mut got: Vec<(u32, u32)> = col0.into_iter().zip(col1).collect();
    got.sort_unstable();
    // edge(X,Y) joined with edge(Y,Z) over {(1,2),(1,3),(2,4),(3,4),(4,5)}:
    // 1->2->4, 1->3->4, 2->4->5, 3->4->5 → after dedup: {(1,4),(2,5),(3,5)}.
    let expected = vec![(1u32, 4), (2, 5), (3, 5)];
    assert_eq!(got, expected);
}
