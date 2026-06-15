// crates/xlog-integration/tests/test_wcoj_clique_dispatch.rs
//! Runtime dispatch tests for k=5..k=8 clique WCOJ.
//!
//! Counter/parity cells:
//!   1. clique5 counter advances + row set matches MultiWayJoin.fallback.
//!   2. clique6 same at k=6.
//!   3. clique7 same at k=7.
//!   4. clique8 same at k=8.
//!
//! Dispatcher-decline cells:
//!   5. clique5 dispatcher decline does NOT advance counter +
//!      row set matches fallback (malformed-schema dispatch path).
//!   6. clique6 same.
//!
//! Tests 1 + 2 build a small K-clique rule via the compiler, run
//! under default config, assert
//! `executor.wcoj_clique{5,6,7,8}_dispatch_count() >= 1` AND row set
//! equals the body that would result from `MultiWayJoin.fallback`
//! (built via a test-only RIR rewrite helper that substitutes
//! MultiWayJoin nodes with their fallback field). NO new
//! force/kill/adaptive runtime knobs.
//!
//! Tests 3 + 4 engineer an internal dispatcher decline by
//! uploading ONE of the clique's edge buffers with a
//! [`xlog_core::ScalarType::I64`] schema (8-byte signed integer
//! — outside both FourByte (`U32`/`Symbol`) and EightByte (`U64`)
//! width-classes). The promoter
//! still validates structure and emits `MultiWayJoin`; the
//! dispatcher's per-edge width-class check (via
//! `wcoj_layout_sort_u32_recorded`) rejects the I64 column,
//! returns `Ok(None)`, and the executor falls through to
//! `MultiWayJoin.fallback`. The fallback's binary-join tree also
//! errors at the type-mismatch boundary (hash-join key column
//! types must match), so both the dispatch and the
//! `replace_multiway_with_fallback` reference paths leave the
//! head store empty — proving observable parity at the row-set
//! level. Each test asserts:
//!   (a) compiled plan contains a `MultiWayJoin` with
//!       `inputs.len() == C(k, 2)` (catches promotion regression
//!       — without this, both paths empty out and the counter
//!       check could silently false-pass);
//!   (b) `executor.wcoj_clique{5,6}_dispatch_count() == 0` after
//!       the run;
//!   (c) `decline_rows == fallback_reference_rows` (both empty
//!       under malformed-schema, but the equality is what's
//!       contractually checked).

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::sys;
use xlog_core::{MemoryBudget, RelId, RuntimeConfig, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::CudaBuffer;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_ir::{ExecutionPlan, RirNode};
use xlog_logic::Compiler;
use xlog_runtime::Executor;
use xlog_stats::{
    ColumnStats, JoinSelectivity, KeyHeatStats, PrefixDegreeStats, RelationStats, StatsSnapshot,
};

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
    let async_r: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(AsyncCudaResource::new(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
    ));
    let logging: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(LoggingResource::new(
        async_r,
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
    let bpc = (n as usize).max(1) * 4;
    let mut col0 = memory.alloc::<u8>(bpc).expect("alloc col0");
    let mut col1 = memory.alloc::<u8>(bpc).expect("alloc col1");
    let mut d_n = memory.alloc::<u32>(1).expect("alloc d_n");
    let dev = memory.device().inner();
    if n > 0 {
        let c0: Vec<u8> = rows.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
        let c1: Vec<u8> = rows.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
        dev.htod_sync_copy_into(&c0, &mut col0).unwrap();
        dev.htod_sync_copy_into(&c1, &mut col1).unwrap();
    }
    dev.htod_sync_copy_into(&[n], &mut d_n).unwrap();
    let schema = Schema::new(vec![
        ("c0".to_string(), ScalarType::U32),
        ("c1".to_string(), ScalarType::U32),
    ]);
    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into()],
        n as u64,
        d_n,
        schema,
        n,
    )
}

/// Upload `(u32, u32)` rows widened to 8-byte cells with
/// `ScalarType::I64` schema. Used to malform a single edge
/// buffer for the dispatcher-decline tests: I64 falls outside
/// both the FourByte (`U32`/`Symbol`) and EightByte (`U64`)
/// width-classes, so the dispatcher's
/// `wcoj_layout_sort_u32_recorded` per-edge check rejects with
/// `Err`, and the dispatcher silently returns `Ok(None)` — which
/// is the contract this test file pins.
fn upload_binary_i64(memory: &Arc<GpuMemoryManager>, rows: &[(u32, u32)]) -> CudaBuffer {
    let n = rows.len() as u32;
    let bpc = (n as usize).max(1) * 8;
    let mut col0 = memory.alloc::<u8>(bpc).expect("alloc col0 i64");
    let mut col1 = memory.alloc::<u8>(bpc).expect("alloc col1 i64");
    let mut d_n = memory.alloc::<u32>(1).expect("alloc d_n");
    let dev = memory.device().inner();
    if n > 0 {
        let c0: Vec<u8> = rows
            .iter()
            .flat_map(|(a, _)| (*a as i64).to_le_bytes())
            .collect();
        let c1: Vec<u8> = rows
            .iter()
            .flat_map(|(_, b)| (*b as i64).to_le_bytes())
            .collect();
        dev.htod_sync_copy_into(&c0, &mut col0).unwrap();
        dev.htod_sync_copy_into(&c1, &mut col1).unwrap();
    }
    dev.htod_sync_copy_into(&[n], &mut d_n).unwrap();
    let schema = Schema::new(vec![
        ("c0".to_string(), ScalarType::I64),
        ("c1".to_string(), ScalarType::I64),
    ]);
    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into()],
        n as u64,
        d_n,
        schema,
        n,
    )
}

/// XLOG source for K_5 clique evaluation. 10 edges in canonical
/// (i, j) order: e01, e02, e03, e04, e12, e13, e14, e23, e24, e34.
const CLIQUE5_SRC: &str = r#"
    pred e01(u32, u32). pred e02(u32, u32). pred e03(u32, u32). pred e04(u32, u32).
    pred e12(u32, u32). pred e13(u32, u32). pred e14(u32, u32).
    pred e23(u32, u32). pred e24(u32, u32).
    pred e34(u32, u32).
    pred clique5(u32, u32, u32, u32, u32).
    clique5(V0, V1, V2, V3, V4) :-
        e01(V0, V1), e02(V0, V2), e03(V0, V3), e04(V0, V4),
        e12(V1, V2), e13(V1, V3), e14(V1, V4),
        e23(V2, V3), e24(V2, V4),
        e34(V3, V4).
"#;

/// XLOG source for K_6 clique. 15 edges.
const CLIQUE6_SRC: &str = r#"
    pred e01(u32, u32). pred e02(u32, u32). pred e03(u32, u32).
    pred e04(u32, u32). pred e05(u32, u32).
    pred e12(u32, u32). pred e13(u32, u32). pred e14(u32, u32). pred e15(u32, u32).
    pred e23(u32, u32). pred e24(u32, u32). pred e25(u32, u32).
    pred e34(u32, u32). pred e35(u32, u32).
    pred e45(u32, u32).
    pred clique6(u32, u32, u32, u32, u32, u32).
    clique6(V0, V1, V2, V3, V4, V5) :-
        e01(V0, V1), e02(V0, V2), e03(V0, V3), e04(V0, V4), e05(V0, V5),
        e12(V1, V2), e13(V1, V3), e14(V1, V4), e15(V1, V5),
        e23(V2, V3), e24(V2, V4), e25(V2, V5),
        e34(V3, V4), e35(V3, V5),
        e45(V4, V5).
"#;

/// XLOG source for K_7 clique. 21 edges.
const CLIQUE7_SRC: &str = r#"
    pred e01(u32, u32). pred e02(u32, u32). pred e03(u32, u32).
    pred e04(u32, u32). pred e05(u32, u32). pred e06(u32, u32).
    pred e12(u32, u32). pred e13(u32, u32). pred e14(u32, u32). pred e15(u32, u32). pred e16(u32, u32).
    pred e23(u32, u32). pred e24(u32, u32). pred e25(u32, u32). pred e26(u32, u32).
    pred e34(u32, u32). pred e35(u32, u32). pred e36(u32, u32).
    pred e45(u32, u32). pred e46(u32, u32).
    pred e56(u32, u32).
    pred clique7(u32, u32, u32, u32, u32, u32, u32).
    clique7(V0, V1, V2, V3, V4, V5, V6) :-
        e01(V0, V1), e02(V0, V2), e03(V0, V3), e04(V0, V4), e05(V0, V5), e06(V0, V6),
        e12(V1, V2), e13(V1, V3), e14(V1, V4), e15(V1, V5), e16(V1, V6),
        e23(V2, V3), e24(V2, V4), e25(V2, V5), e26(V2, V6),
        e34(V3, V4), e35(V3, V5), e36(V3, V6),
        e45(V4, V5), e46(V4, V6),
        e56(V5, V6).
"#;

/// XLOG source for K_8 clique. 28 edges.
const CLIQUE8_SRC: &str = r#"
    pred e01(u32, u32). pred e02(u32, u32). pred e03(u32, u32).
    pred e04(u32, u32). pred e05(u32, u32). pred e06(u32, u32). pred e07(u32, u32).
    pred e12(u32, u32). pred e13(u32, u32). pred e14(u32, u32). pred e15(u32, u32). pred e16(u32, u32). pred e17(u32, u32).
    pred e23(u32, u32). pred e24(u32, u32). pred e25(u32, u32). pred e26(u32, u32). pred e27(u32, u32).
    pred e34(u32, u32). pred e35(u32, u32). pred e36(u32, u32). pred e37(u32, u32).
    pred e45(u32, u32). pred e46(u32, u32). pred e47(u32, u32).
    pred e56(u32, u32). pred e57(u32, u32).
    pred e67(u32, u32).
    pred clique8(u32, u32, u32, u32, u32, u32, u32, u32).
    clique8(V0, V1, V2, V3, V4, V5, V6, V7) :-
        e01(V0, V1), e02(V0, V2), e03(V0, V3), e04(V0, V4), e05(V0, V5), e06(V0, V6), e07(V0, V7),
        e12(V1, V2), e13(V1, V3), e14(V1, V4), e15(V1, V5), e16(V1, V6), e17(V1, V7),
        e23(V2, V3), e24(V2, V4), e25(V2, V5), e26(V2, V6), e27(V2, V7),
        e34(V3, V4), e35(V3, V5), e36(V3, V6), e37(V3, V7),
        e45(V4, V5), e46(V4, V6), e47(V4, V7),
        e56(V5, V6), e57(V5, V7),
        e67(V6, V7).
"#;

/// Build a complete-K_K fixture on K vertices. Returns
/// `[(edge_name, rows)]`.
fn k_clique_inputs(k: usize) -> BTreeMap<String, Vec<(u32, u32)>> {
    let mut m: BTreeMap<String, Vec<(u32, u32)>> = BTreeMap::new();
    for i in 0u32..(k as u32) {
        for j in (i + 1)..(k as u32) {
            // Edge name e{i}{j} carries the single tuple (i+1, j+1).
            let name = format!("e{}{}", i, j);
            m.insert(name, vec![(i + 1, j + 1)]);
        }
    }
    m
}

/// Test-only RIR rewrite helper: walk the plan tree, detect
/// `RirNode::MultiWayJoin` nodes, and substitute each with its
/// `fallback` field. Used to build the binary-join reference
/// row set without introducing new force/kill/adaptive knobs.
fn replace_multiway_with_fallback(mut plan: ExecutionPlan) -> ExecutionPlan {
    fn rewrite(node: &RirNode) -> RirNode {
        match node {
            RirNode::MultiWayJoin { fallback, .. } => rewrite(fallback),
            RirNode::Project { input, columns } => RirNode::Project {
                input: Box::new(rewrite(input)),
                columns: columns.clone(),
            },
            RirNode::Filter { input, predicate } => RirNode::Filter {
                input: Box::new(rewrite(input)),
                predicate: predicate.clone(),
            },
            RirNode::Join {
                left,
                right,
                left_keys,
                right_keys,
                join_type,
            } => RirNode::Join {
                left: Box::new(rewrite(left)),
                right: Box::new(rewrite(right)),
                left_keys: left_keys.clone(),
                right_keys: right_keys.clone(),
                join_type: *join_type,
            },
            RirNode::Union { inputs } => RirNode::Union {
                inputs: inputs.iter().map(rewrite).collect(),
            },
            RirNode::Diff { left, right } => RirNode::Diff {
                left: Box::new(rewrite(left)),
                right: Box::new(rewrite(right)),
            },
            RirNode::Distinct { input, key_cols } => RirNode::Distinct {
                input: Box::new(rewrite(input)),
                key_cols: key_cols.clone(),
            },
            other => other.clone(),
        }
    }
    for rules in plan.rules_by_scc.iter_mut() {
        for rule in rules.iter_mut() {
            rule.body = rewrite(&rule.body);
        }
    }
    plan
}

/// Walk `plan.rules_by_scc` → `CompiledRule.body` and return
/// true iff at least one `RirNode::MultiWayJoin` node has
/// `inputs.len() == arity`. Used by the dispatcher-decline
/// tests to assert that promotion actually emitted a MultiWayJoin
/// at the expected `C(k, 2)` shape — so if the clique promoter
/// regresses, the tests fail
/// loudly rather than passing on incidental empty outputs.
fn plan_contains_multiway_with_arity(plan: &ExecutionPlan, arity: usize) -> bool {
    fn walk(node: &RirNode, target: usize) -> bool {
        match node {
            RirNode::MultiWayJoin {
                inputs, fallback, ..
            } => inputs.len() == target || walk(fallback, target),
            RirNode::Project { input, .. } => walk(input, target),
            RirNode::Filter { input, .. } => walk(input, target),
            RirNode::Join { left, right, .. } => walk(left, target) || walk(right, target),
            RirNode::Union { inputs } => inputs.iter().any(|n| walk(n, target)),
            RirNode::Diff { left, right } => walk(left, target) || walk(right, target),
            RirNode::Distinct { input, .. } => walk(input, target),
            _ => false,
        }
    }
    plan.rules_by_scc
        .iter()
        .any(|rules| rules.iter().any(|rule| walk(&rule.body, arity)))
}

fn plan_contains_multiway_scan(plan: &ExecutionPlan, arity: usize, target_rel: RelId) -> bool {
    fn contains_scan(node: &RirNode, target_rel: RelId) -> bool {
        match node {
            RirNode::Scan { rel } => *rel == target_rel,
            RirNode::Project { input, .. }
            | RirNode::Filter { input, .. }
            | RirNode::Distinct { input, .. }
            | RirNode::GroupBy { input, .. } => contains_scan(input, target_rel),
            RirNode::Join { left, right, .. } | RirNode::Diff { left, right } => {
                contains_scan(left, target_rel) || contains_scan(right, target_rel)
            }
            RirNode::Union { inputs } => {
                inputs.iter().any(|input| contains_scan(input, target_rel))
            }
            RirNode::MultiWayJoin { inputs, .. } => {
                inputs.iter().any(|input| contains_scan(input, target_rel))
            }
            _ => false,
        }
    }

    fn walk(node: &RirNode, arity: usize, target_rel: RelId) -> bool {
        match node {
            RirNode::MultiWayJoin { inputs, .. } if inputs.len() == arity => {
                inputs.iter().any(|input| contains_scan(input, target_rel))
            }
            RirNode::MultiWayJoin {
                inputs, fallback, ..
            } => {
                inputs.iter().any(|input| walk(input, arity, target_rel))
                    || walk(fallback, arity, target_rel)
            }
            RirNode::Project { input, .. }
            | RirNode::Filter { input, .. }
            | RirNode::Distinct { input, .. }
            | RirNode::GroupBy { input, .. } => walk(input, arity, target_rel),
            RirNode::Join { left, right, .. } | RirNode::Diff { left, right } => {
                walk(left, arity, target_rel) || walk(right, arity, target_rel)
            }
            RirNode::Union { inputs } => inputs.iter().any(|input| walk(input, arity, target_rel)),
            _ => false,
        }
    }

    plan.rules_by_scc
        .iter()
        .any(|rules| rules.iter().any(|rule| walk(&rule.body, arity, target_rel)))
}

fn download_k_row_set(buf: &CudaBuffer, k: usize) -> std::collections::BTreeSet<Vec<u32>> {
    let n = match buf.cached_row_count() {
        Some(c) => c as usize,
        None => {
            let mut count = [0u32; 1];
            unsafe {
                sys::cuMemcpyDtoH_v2(
                    count.as_mut_ptr() as *mut _,
                    *buf.num_rows_device().device_ptr(),
                    4,
                );
            }
            count[0] as usize
        }
    };
    if n == 0 {
        return std::collections::BTreeSet::new();
    }
    let mut cols: Vec<Vec<u8>> = (0..k).map(|_| vec![0u8; n * 4]).collect();
    for (c, col_bytes) in cols.iter_mut().enumerate().take(k) {
        unsafe {
            sys::cuMemcpyDtoH_v2(
                col_bytes.as_mut_ptr() as *mut _,
                *buf.column(c).unwrap().device_ptr(),
                col_bytes.len(),
            );
        }
    }
    (0..n)
        .map(|r| {
            (0..k)
                .map(|c| {
                    let off = r * 4;
                    u32::from_le_bytes([
                        cols[c][off],
                        cols[c][off + 1],
                        cols[c][off + 2],
                        cols[c][off + 3],
                    ])
                })
                .collect()
        })
        .collect()
}

/// Run a clique-K test: compile, build two executors (dispatch
/// path + fallback-only path), assert counter ≥ 1 on dispatch
/// AND row-set equality between dispatch output and
/// `replace_multiway_with_fallback` reference.
fn run_counter_advance_test(
    fix: &RuntimeBackedFixture,
    src: &str,
    head_name: &str,
    k: usize,
    check_counter: fn(&Executor) -> u64,
) {
    let inputs = k_clique_inputs(k);

    // 1. Dispatch path: compile + run under default dispatch.
    let mut compiler = Compiler::new();
    let snapshot = named_clique_stats(k as u8);
    let plan = compiler
        .compile_with_stats_snapshot(src, Some(&snapshot))
        .expect("compile");
    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rid) in compiler.rel_ids().clone() {
        executor.register_relation(rid, &name);
    }
    for (name, rows) in &inputs {
        executor.put_relation(name, upload_binary_u32(&fix.memory, rows));
    }
    let _ = executor.execute_plan(&plan).expect("execute dispatch");
    let dispatch_rows = download_k_row_set(executor.store().get(head_name).expect("head"), k);
    assert!(
        check_counter(&executor) >= 1,
        "expected ≥ 1 clique dispatch; got {}",
        check_counter(&executor)
    );

    // 2. Fallback reference: rewrite the plan to replace
    //    MultiWayJoin nodes with their fallback bodies, then
    //    run on a fresh executor with the same inputs. This
    //    exercises the binary-join path without any new
    //    force/kill/adaptive knobs.
    let mut compiler_ref = Compiler::new();
    let plan_ref = compiler_ref
        .compile_with_stats_snapshot(src, Some(&snapshot))
        .expect("compile ref");
    let fallback_plan = replace_multiway_with_fallback(plan_ref);
    let mut executor_ref =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rid) in compiler_ref.rel_ids().clone() {
        executor_ref.register_relation(rid, &name);
    }
    for (name, rows) in &inputs {
        executor_ref.put_relation(name, upload_binary_u32(&fix.memory, rows));
    }
    let _ = executor_ref
        .execute_plan(&fallback_plan)
        .expect("execute fallback");
    let fallback_rows =
        download_k_row_set(executor_ref.store().get(head_name).expect("head ref"), k);

    // 3. Row-set parity: dispatch output == fallback output.
    assert_eq!(
        dispatch_rows, fallback_rows,
        "K={} dispatch row set must equal MultiWayJoin.fallback reference",
        k
    );
}

fn named_clique_stats(k: u8) -> StatsSnapshot {
    named_clique_stats_with_hot_variable(k, None)
}

fn named_clique_stats_with_hot_variable(k: u8, hot: Option<(u8, f64)>) -> StatsSnapshot {
    let mut snapshot = StatsSnapshot::default();
    let mut edges = Vec::new();
    let mut rel_id = 1u32;

    for i in 0..k {
        for j in (i + 1)..k {
            let rel = RelId(rel_id);
            rel_id += 1;
            snapshot.rel_names.push((rel, format!("e{i}{j}")));
            edges.push((rel, i, j));

            let mut stats = RelationStats::new(rel);
            stats.update_cardinality(2_000 + u64::from(k));
            for (col_idx, variable) in [(0usize, i), (1usize, j)] {
                let mut col = ColumnStats::new(col_idx, ScalarType::U32);
                col.update_distinct(1_000 + u64::from(k));
                stats.add_column(col);
                stats.add_prefix_degree(PrefixDegreeStats::new(col_idx, 2.0, 2.5));
                let heat = match hot {
                    Some((hot_variable, hot_heat)) if hot_variable == variable => hot_heat,
                    _ => 0.75,
                };
                stats.add_key_heat(KeyHeatStats::new(col_idx, heat, heat));
            }
            snapshot.relations.push(stats);
        }
    }

    for (left_idx, (left_rel, left_i, left_j)) in edges.iter().enumerate() {
        for (right_rel, right_i, right_j) in edges.iter().skip(left_idx + 1) {
            if left_i == right_i || left_i == right_j || left_j == right_i || left_j == right_j {
                let mut sel = JoinSelectivity::new(*left_rel, *right_rel);
                sel.set_keys(vec![0], vec![0]);
                sel.set_selectivity(0.001);
                snapshot.join_selectivities.push(sel);
            }
        }
    }

    snapshot
}

#[test]
fn helper_split_k5_matches_direct_kclique_and_refreshes_metadata() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = k_clique_inputs(5);

    let mut helper_compiler = Compiler::new();
    let helper_snapshot = named_clique_stats_with_hot_variable(5, Some((3, 5.0)));
    let helper_plan = helper_compiler
        .compile_with_stats_snapshot(CLIQUE5_SRC, Some(&helper_snapshot))
        .expect("compile helper-split K5");
    let helpers: Vec<_> = helper_compiler
        .rel_ids()
        .iter()
        .filter(|(name, _)| name.starts_with("__kclique_helper_"))
        .map(|(name, rel)| (name.clone(), *rel))
        .collect();
    assert_eq!(
        helpers.len(),
        1,
        "buried-skew K5 must allocate exactly one helper relation"
    );
    assert!(
        plan_contains_multiway_scan(&helper_plan, 10, helpers[0].1),
        "outer K5 MultiWayJoin must consume the helper relation"
    );

    fix.provider.reset_kclique_metadata_build_metrics();
    let mut helper_executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rid) in helper_compiler.rel_ids().clone() {
        helper_executor.register_relation(rid, &name);
    }
    for (name, rows) in &inputs {
        helper_executor.put_relation(name, upload_binary_u32(&fix.memory, rows));
    }
    helper_executor
        .execute_plan(&helper_plan)
        .expect("execute helper-split K5");
    let helper_rows = download_k_row_set(
        helper_executor
            .store()
            .get("clique5")
            .expect("helper clique5"),
        5,
    );
    let metadata_build_count = fix.provider.kclique_metadata_build_count();
    let metadata_build_nanos = fix.provider.kclique_metadata_build_nanos();

    let mut direct_compiler = Compiler::new();
    let direct_plan = direct_compiler
        .compile_with_stats_snapshot(CLIQUE5_SRC, Some(&named_clique_stats(5)))
        .expect("compile direct K5");
    assert!(
        !direct_compiler
            .rel_ids()
            .keys()
            .any(|name| name.starts_with("__kclique_helper_")),
        "uniform K5 reference must keep the direct K-clique path"
    );
    let mut direct_executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rid) in direct_compiler.rel_ids().clone() {
        direct_executor.register_relation(rid, &name);
    }
    for (name, rows) in &inputs {
        direct_executor.put_relation(name, upload_binary_u32(&fix.memory, rows));
    }
    direct_executor
        .execute_plan(&direct_plan)
        .expect("execute direct K5");
    let direct_rows = download_k_row_set(
        direct_executor
            .store()
            .get("clique5")
            .expect("direct clique5"),
        5,
    );

    assert_eq!(
        helper_rows, direct_rows,
        "helper-split K5 row set must equal direct K-clique row set"
    );
    assert!(
        helper_executor.wcoj_clique5_dispatch_count() >= 1,
        "helper-split K5 must dispatch through the K-clique kernel"
    );
    assert!(
        metadata_build_count >= 1,
        "post-split helper K5 path must build K-clique metadata"
    );
    eprintln!(
        "KCLIQUE_HELPER_SPLIT helper K5: helper_relations={} dispatch_count={} metadata_build_count={} metadata_build_nanos={} rows={}",
        helpers.len(),
        helper_executor.wcoj_clique5_dispatch_count(),
        metadata_build_count,
        metadata_build_nanos,
        helper_rows.len()
    );
}

const RECURSIVE_K5_HISTOGRAM_SRC: &str = r#"
    pred seed01(u32, u32).
    pred path01(u32, u32).
    pred e02(u32, u32). pred e03(u32, u32). pred e04(u32, u32).
    pred e12(u32, u32). pred e13(u32, u32). pred e14(u32, u32).
    pred e23(u32, u32). pred e24(u32, u32).
    pred e34(u32, u32).
    pred clique5(u32, u32, u32, u32, u32).

    path01(A, B) :- seed01(A, B).
    clique5(A, B, C, D, E) :-
        path01(A, B), e02(A, C), e03(A, D), e04(A, E),
        e12(B, C), e13(B, D), e14(B, E),
        e23(C, D), e24(C, E),
        e34(D, E).
    path01(A, C) :- clique5(A, B, C, D, E).
"#;

fn recursive_k5_inputs() -> BTreeMap<String, Vec<(u32, u32)>> {
    BTreeMap::from([
        ("seed01".to_string(), vec![(1, 2)]),
        ("e02".to_string(), vec![(1, 3), (1, 4), (1, 5)]),
        ("e03".to_string(), vec![(1, 4), (1, 5), (1, 6)]),
        ("e04".to_string(), vec![(1, 5), (1, 6), (1, 7)]),
        ("e12".to_string(), vec![(2, 3), (3, 4), (4, 5)]),
        ("e13".to_string(), vec![(2, 4), (3, 5), (4, 6)]),
        ("e14".to_string(), vec![(2, 5), (3, 6), (4, 7)]),
        ("e23".to_string(), vec![(3, 4), (4, 5), (5, 6)]),
        ("e24".to_string(), vec![(3, 5), (4, 6), (5, 7)]),
        ("e34".to_string(), vec![(4, 5), (5, 6), (6, 7)]),
    ])
}

fn recursive_k5_stats(rel_ids: &BTreeMap<String, RelId>) -> StatsSnapshot {
    let mut snapshot = StatsSnapshot::default();
    let canonical_edges = [
        ("path01", 0u8, 1u8),
        ("e02", 0, 2),
        ("e03", 0, 3),
        ("e04", 0, 4),
        ("e12", 1, 2),
        ("e13", 1, 3),
        ("e14", 1, 4),
        ("e23", 2, 3),
        ("e24", 2, 4),
        ("e34", 3, 4),
    ];
    for name in [
        "seed01", "path01", "e02", "e03", "e04", "e12", "e13", "e14", "e23", "e24", "e34",
    ] {
        let rel = *rel_ids.get(name).expect("relation id");
        snapshot.rel_names.push((rel, name.to_string()));
        let mut stats = RelationStats::new(rel);
        stats.update_cardinality(if name == "seed01" { 1 } else { 2_005 });
        for col_idx in [0usize, 1usize] {
            let mut col = ColumnStats::new(col_idx, ScalarType::U32);
            col.update_distinct(1_005);
            stats.add_column(col);
            stats.add_prefix_degree(PrefixDegreeStats::new(col_idx, 2.0, 2.5));
            stats.add_key_heat(KeyHeatStats::new(col_idx, 0.75, 0.75));
        }
        snapshot.relations.push(stats);
    }
    for (left_idx, (left_name, left_i, left_j)) in canonical_edges.iter().enumerate() {
        let left_rel = *rel_ids.get(*left_name).expect("left relation id");
        for (right_name, right_i, right_j) in canonical_edges.iter().skip(left_idx + 1) {
            if left_i == right_i || left_i == right_j || left_j == right_i || left_j == right_j {
                let right_rel = *rel_ids.get(*right_name).expect("right relation id");
                let mut sel = JoinSelectivity::new(left_rel, right_rel);
                sel.set_keys(vec![0], vec![0]);
                sel.set_selectivity(0.001);
                snapshot.join_selectivities.push(sel);
            }
        }
    }
    snapshot
}

fn run_recursive_k5(
    fix: &RuntimeBackedFixture,
    fallback: bool,
) -> (Executor, std::collections::BTreeSet<Vec<u32>>) {
    let inputs = recursive_k5_inputs();
    let mut id_compiler = Compiler::new();
    let _ = id_compiler
        .compile(RECURSIVE_K5_HISTOGRAM_SRC)
        .expect("compile recursive k5 ids");
    let rel_ids: BTreeMap<String, RelId> = id_compiler
        .rel_ids()
        .iter()
        .map(|(name, rel)| (name.clone(), *rel))
        .collect();
    let snapshot = recursive_k5_stats(&rel_ids);
    let mut compiler = Compiler::new();
    let plan = compiler
        .compile_with_stats_snapshot(RECURSIVE_K5_HISTOGRAM_SRC, Some(&snapshot))
        .expect("compile recursive k5");
    assert!(
        plan_contains_multiway_with_arity(&plan, 10),
        "recursive K5 plan must contain a 10-input MultiWayJoin"
    );
    let plan = if fallback {
        replace_multiway_with_fallback(plan)
    } else {
        plan
    };
    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rid) in compiler.rel_ids().clone() {
        executor.register_relation(rid, &name);
    }
    for (name, rows) in &inputs {
        executor.put_relation(name, upload_binary_u32(&fix.memory, rows));
    }
    executor.execute_plan(&plan).expect("execute recursive k5");
    let rows = download_k_row_set(executor.store().get("path01").expect("path01"), 2);
    (executor, rows)
}

#[test]
fn recursive_k5_refreshes_histogram_metadata_and_matches_fallback() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    fix.provider.reset_kclique_metadata_build_metrics();
    let wall_start = Instant::now();
    let (dispatched, dispatch_rows) = run_recursive_k5(&fix, false);
    let dispatch_wall = wall_start.elapsed();
    let metadata_build_count = fix.provider.kclique_metadata_build_count();
    let metadata_build_nanos = fix.provider.kclique_metadata_build_nanos();
    let (_fallback, fallback_rows) = run_recursive_k5(&fix, true);
    assert_eq!(
        dispatch_rows, fallback_rows,
        "recursive K5 metadata-refresh path must match fallback fixpoint output"
    );
    assert!(
        dispatched.wcoj_clique5_dispatch_count() >= 2,
        "recursive K5 must dispatch on seeding and at least one semi-naive variant; got {}",
        dispatched.wcoj_clique5_dispatch_count()
    );
    assert!(
        dispatched.kclique_histogram_refresh_count() >= 1,
        "recursive Merge phase must mark at least one K-clique histogram refresh"
    );
    let metadata_ratio = metadata_build_nanos as f64 / (dispatch_wall.as_nanos() as f64).max(1.0);
    eprintln!(
        "KCLIQUE_HISTOGRAM_REFRESH recursive K5: dispatch_count={} refresh_count={} metadata_build_count={} metadata_build_nanos={} wall_nanos={} metadata_ratio={:.6}",
        dispatched.wcoj_clique5_dispatch_count(),
        dispatched.kclique_histogram_refresh_count(),
        metadata_build_count,
        metadata_build_nanos,
        dispatch_wall.as_nanos(),
        metadata_ratio
    );
    assert!(
        metadata_ratio <= 0.05,
        "metadata build cost ratio must stay <= 5%; got {:.6}",
        metadata_ratio
    );
    assert_eq!(
        dispatch_rows.len(),
        4,
        "recursive K5 fixture should derive the seed plus three transitive path01 rows"
    );
}

#[test]
fn clique5_dispatch_counter_advances_and_row_set_matches_fallback_body() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    fix.provider.reset_wcoj_layout_sort_invocation_count();
    run_counter_advance_test(&fix, CLIQUE5_SRC, "clique5", 5, |e| {
        e.wcoj_clique5_dispatch_count()
    });
    let sort_count = fix.provider.wcoj_layout_sort_invocation_count();
    assert!(
        sort_count < 10,
        "K5 planned dispatch must use fewer than the old 10 layout-sort calls; got {sort_count}"
    );
}

#[test]
fn clique6_dispatch_counter_advances_and_row_set_matches_fallback_body() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    run_counter_advance_test(&fix, CLIQUE6_SRC, "clique6", 6, |e| {
        e.wcoj_clique6_dispatch_count()
    });
}

#[test]
fn clique7_dispatch_counter_advances_and_row_set_matches_fallback_body() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    run_counter_advance_test(&fix, CLIQUE7_SRC, "clique7", 7, |e| {
        e.wcoj_clique7_dispatch_count()
    });
}

#[test]
fn clique8_dispatch_counter_advances_and_row_set_matches_fallback_body() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    run_counter_advance_test(&fix, CLIQUE8_SRC, "clique8", 8, |e| {
        e.wcoj_clique8_dispatch_count()
    });
}

/// Run a clique-K dispatcher-decline test under the malformed-schema
/// contract:
///
/// One of the K*(K-1)/2 edge buffers is uploaded with a
/// `ScalarType::I64` schema (8-byte signed integer — outside
/// both the FourByte (`U32`/`Symbol`) and EightByte (`U64`)
/// width-classes). The promoter validates structure independently
/// of edge schemas and emits a `MultiWayJoin`. The dispatcher
/// then layout-sorts each edge through
/// `wcoj_layout_sort_u32_recorded`, which rejects the I64 column
/// with `Err`, and the dispatcher silently returns `Ok(None)` —
/// the documented decline path.
///
/// On dispatch decline the executor falls through to
/// `MultiWayJoin.fallback`. That binary-join tree's hash-join
/// also rejects at the type-mismatch boundary between the I64
/// edge and adjacent U32 edges, so the head store remains empty.
/// The `replace_multiway_with_fallback` reference path runs the
/// same binary-join tree against the same malformed fixture and
/// errors at the same boundary, leaving the head equally empty.
/// Row-set parity is therefore the operational guarantee: both
/// observable outputs are equal (both empty) under malformed-
/// schema input.
///
/// Asserts:
///   (a) `plan_contains_multiway_with_arity(&plan, C(k, 2))` —
///       proves promotion ran (catches regression).
///   (b) `check_counter(&executor) == 0` — dispatcher declined.
///   (c) `decline_rows == fallback_rows` — observable parity.
fn run_dispatcher_decline_test(
    fix: &RuntimeBackedFixture,
    src: &str,
    head_name: &str,
    k: usize,
    malformed_edge_name: &str,
    check_counter: fn(&Executor) -> u64,
) {
    let inputs = k_clique_inputs(k);
    let expected_arity = k * (k - 1) / 2;

    // 1. Compile + assert MultiWayJoin emitted at the K-clique
    //    arity. Without this guard, a regression in
    //    `try_promote_clique_k` would silently route through the
    //    binary-join tree on both sides, both heads would be
    //    empty, and the counter==0 + parity asserts would still
    //    "pass" — false-pass risk per user iteration finding.
    let mut compiler = Compiler::new();
    let plan = compiler.compile(src).expect("compile");
    assert!(
        plan_contains_multiway_with_arity(&plan, expected_arity),
        "K={} compiled plan must contain a MultiWayJoin with {} inputs \
         (proves promoter emitted the clique node before dispatcher decline)",
        k,
        expected_arity
    );

    // 2. Dispatch executor: all edges valid U32 except
    //    `malformed_edge_name` which is uploaded with I64
    //    schema. Layout-sort rejects → dispatcher Ok(None) →
    //    counter stays 0 → fallback runs but errors at
    //    type-mismatch boundary.
    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rid) in compiler.rel_ids().clone() {
        executor.register_relation(rid, &name);
    }
    for (name, rows) in &inputs {
        let buf = if name == malformed_edge_name {
            upload_binary_i64(&fix.memory, rows)
        } else {
            upload_binary_u32(&fix.memory, rows)
        };
        executor.put_relation(name, buf);
    }
    let _ = executor.execute_plan(&plan);
    assert_eq!(
        check_counter(&executor),
        0,
        "K={} dispatcher decline must NOT advance the counter; got {}",
        k,
        check_counter(&executor)
    );

    // 3. Fallback reference executor: same malformed fixture,
    //    plan rewritten via `replace_multiway_with_fallback` to
    //    bypass MultiWayJoin → pure binary-join.
    let mut compiler_ref = Compiler::new();
    let plan_ref = compiler_ref.compile(src).expect("compile ref");
    let fallback_plan = replace_multiway_with_fallback(plan_ref);
    let mut executor_ref =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rid) in compiler_ref.rel_ids().clone() {
        executor_ref.register_relation(rid, &name);
    }
    for (name, rows) in &inputs {
        let buf = if name == malformed_edge_name {
            upload_binary_i64(&fix.memory, rows)
        } else {
            upload_binary_u32(&fix.memory, rows)
        };
        executor_ref.put_relation(name, buf);
    }
    let _ = executor_ref.execute_plan(&fallback_plan);

    // 4. Row-set parity: both paths see the same malformed
    //    schema and produce the same (empty) result. This is the
    //    plan's locked equivalence between the dispatcher
    //    silent-decline behavior and the `MultiWayJoin.fallback`
    //    body.
    let decline_rows = executor
        .store()
        .get(head_name)
        .map(|b| download_k_row_set(b, k))
        .unwrap_or_default();
    let fallback_rows = executor_ref
        .store()
        .get(head_name)
        .map(|b| download_k_row_set(b, k))
        .unwrap_or_default();
    assert_eq!(
        decline_rows, fallback_rows,
        "K={} dispatcher-decline row set must equal MultiWayJoin.fallback reference \
         (malformed edge: {})",
        k, malformed_edge_name
    );
}

#[test]
fn clique5_dispatcher_decline_does_not_advance_counter_and_row_set_matches_fallback() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    run_dispatcher_decline_test(&fix, CLIQUE5_SRC, "clique5", 5, "e34", |e| {
        e.wcoj_clique5_dispatch_count()
    });
}

#[test]
fn clique6_dispatcher_decline_does_not_advance_counter_and_row_set_matches_fallback() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    run_dispatcher_decline_test(&fix, CLIQUE6_SRC, "clique6", 6, "e45", |e| {
        e.wcoj_clique6_dispatch_count()
    });
}
