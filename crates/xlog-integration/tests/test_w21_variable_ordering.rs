// crates/xlog-integration/tests/test_w21_variable_ordering.rs
//! W2.1 step 7 — Part C, D, E acceptance gate (11 tests).
//!
//! End-to-end + IR-level acceptance for the variable-ordering
//! cost model:
//!
//! * Part C (7) — end-to-end row-set parity. Each test compiles
//!   a triangle/4-cycle fixture with stats favoring a target
//!   leader, runs both with force-WCOJ + LeaderCardinality and
//!   force-binary-join, and asserts:
//!     - WCOJ dispatch counter ≥ 1 (kernel actually ran).
//!     - WCOJ row set equals the binary-join reference (W2.1
//!       reordering preserves rule semantics).
//!
//! * Part D (2) — stats-driven divergence. Same source, two
//!   distinct stats snapshots → different `var_order.leader_idx`
//!   on the compiled plans.
//!
//! * Part E (2) — threshold gate cert. Ratio at 0.6 (above 0.5)
//!   leaves `var_order = None`; ratio at 0.3 fires.

use std::collections::BTreeMap;
use std::sync::Arc;

use cudarc::driver::sys;
use xlog_core::{MemoryBudget, RelId, RuntimeConfig, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::CudaBuffer;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_ir::rir::VariableOrder;
use xlog_ir::RirNode;
use xlog_logic::compile::Compiler;
use xlog_logic::compiler_config::{CompilerConfig, WcojVarOrderingKind};
use xlog_runtime::Executor;
use xlog_stats::{RelationStats, StatsSnapshot};

// ---------------------------------------------------------------
// Fixture infrastructure (mirrors W2.2 cert conventions).
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

fn make_runtime_fixture() -> Option<RuntimeBackedFixture> {
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
    let bytes_per_col = (n as usize) * std::mem::size_of::<u32>();
    let mut col0 = memory.alloc::<u8>(bytes_per_col).expect("alloc col0");
    let mut col1 = memory.alloc::<u8>(bytes_per_col).expect("alloc col1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let device = memory.device().inner();
    if !rows.is_empty() {
        let bs0: Vec<u8> = rows.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
        let bs1: Vec<u8> = rows.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
        device.htod_sync_copy_into(&bs0, &mut col0).unwrap();
        device.htod_sync_copy_into(&bs1, &mut col1).unwrap();
    }
    device.htod_sync_copy_into(&[n], &mut d_num_rows).unwrap();
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
    let mut col0 = vec![0u8; n * 4];
    let mut col1 = vec![0u8; n * 4];
    let mut col2 = vec![0u8; n * 4];
    unsafe {
        sys::cuMemcpyDtoH_v2(
            col0.as_mut_ptr() as *mut _,
            *buf.column(0).unwrap().device_ptr(),
            col0.len(),
        );
        sys::cuMemcpyDtoH_v2(
            col1.as_mut_ptr() as *mut _,
            *buf.column(1).unwrap().device_ptr(),
            col1.len(),
        );
        sys::cuMemcpyDtoH_v2(
            col2.as_mut_ptr() as *mut _,
            *buf.column(2).unwrap().device_ptr(),
            col2.len(),
        );
    }
    let mut out: Vec<(u32, u32, u32)> = (0..n)
        .map(|i| {
            (
                u32::from_le_bytes(col0[i * 4..i * 4 + 4].try_into().unwrap()),
                u32::from_le_bytes(col1[i * 4..i * 4 + 4].try_into().unwrap()),
                u32::from_le_bytes(col2[i * 4..i * 4 + 4].try_into().unwrap()),
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
    let mut cols = [
        vec![0u8; n * 4],
        vec![0u8; n * 4],
        vec![0u8; n * 4],
        vec![0u8; n * 4],
    ];
    for (c, col_bytes) in cols.iter_mut().enumerate() {
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

const TRIANGLE_SRC: &str = r#"
    pred e1(u32, u32).
    pred e2(u32, u32).
    pred e3(u32, u32).
    pred tri(u32, u32, u32).
    tri(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z).
"#;

const CYCLE4_SRC: &str = r#"
    pred e1(u32, u32).
    pred e2(u32, u32).
    pred e3(u32, u32).
    pred e4(u32, u32).
    pred cyc(u32, u32, u32, u32).
    cyc(W, X, Y, Z) :- e1(W, X), e2(X, Y), e3(Y, Z), e4(Z, W).
"#;

fn triangle_inputs() -> BTreeMap<&'static str, Vec<(u32, u32)>> {
    let mut m: BTreeMap<&'static str, Vec<(u32, u32)>> = BTreeMap::new();
    m.insert("e1", vec![(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]);
    m.insert("e2", vec![(2, 3), (2, 4), (3, 4)]);
    m.insert("e3", vec![(1, 3), (1, 4), (2, 4), (3, 4)]);
    m
}

fn cycle4_inputs() -> BTreeMap<&'static str, Vec<(u32, u32)>> {
    let mut m: BTreeMap<&'static str, Vec<(u32, u32)>> = BTreeMap::new();
    // 2 disjoint 4-cycles: (1→2→3→4→1) and (5→6→7→8→5).
    m.insert("e1", vec![(1, 2), (5, 6)]);
    m.insert("e2", vec![(2, 3), (6, 7)]);
    m.insert("e3", vec![(3, 4), (7, 8)]);
    m.insert("e4", vec![(4, 1), (8, 5)]);
    m
}

fn make_snapshot(seeded: &[(&str, u64)]) -> StatsSnapshot {
    let relations: Vec<RelationStats> = seeded
        .iter()
        .enumerate()
        .map(|(i, (_, card))| {
            let mut s = RelationStats::new(RelId(i as u32));
            s.cardinality = *card;
            s
        })
        .collect();
    let rel_names: Vec<(RelId, String)> = seeded
        .iter()
        .enumerate()
        .map(|(i, (name, _))| (RelId(i as u32), (*name).to_string()))
        .collect();
    StatsSnapshot {
        relations,
        join_selectivities: vec![],
        rel_names,
    }
}

fn run_w21(
    fix: &RuntimeBackedFixture,
    runtime_config: RuntimeConfig,
    source: &str,
    inputs: &BTreeMap<&str, Vec<(u32, u32)>>,
    snapshot: &StatsSnapshot,
    compiler_config: &CompilerConfig,
) -> Executor {
    let mut compiler = Compiler::new();
    let plan = compiler
        .compile_with_config_and_stats_snapshot(source, compiler_config, Some(snapshot))
        .expect("compile_with_config_and_stats_snapshot");
    let mut executor = Executor::new_with_config(Arc::clone(&fix.provider), runtime_config);
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    for (name, rows) in inputs {
        let buf = upload_binary_u32(&fix.memory, rows);
        executor.put_relation(name, buf);
    }
    executor.execute_plan(&plan).expect("execute_plan");
    executor
}

fn force_wcoj_cfg() -> RuntimeConfig {
    let mut c = RuntimeConfig::default();
    c.wcoj_triangle_dispatch = Some(true);
    c.wcoj_4cycle_dispatch = Some(true);
    c
}

fn no_wcoj_cfg() -> RuntimeConfig {
    let mut c = RuntimeConfig::default();
    c.wcoj_triangle_dispatch = Some(false);
    c.wcoj_4cycle_dispatch = Some(false);
    c
}

fn w21_compiler_config(kind: WcojVarOrderingKind) -> CompilerConfig {
    CompilerConfig {
        wcoj_variable_ordering: kind,
        ..CompilerConfig::default()
    }
}

fn first_var_order(plan: &xlog_ir::ExecutionPlan) -> Option<VariableOrder> {
    fn find(node: &RirNode) -> Option<VariableOrder> {
        match node {
            RirNode::MultiWayJoin { var_order, .. } => var_order.clone(),
            RirNode::Filter { input, .. }
            | RirNode::Project { input, .. }
            | RirNode::GroupBy { input, .. }
            | RirNode::Distinct { input, .. } => find(input),
            RirNode::Join { left, right, .. } | RirNode::Diff { left, right } => {
                find(left).or_else(|| find(right))
            }
            RirNode::Union { inputs } => inputs.iter().find_map(find),
            RirNode::Fixpoint {
                base, recursive, ..
            } => find(base).or_else(|| find(recursive)),
            _ => None,
        }
    }
    plan.rules_by_scc
        .iter()
        .flatten()
        .find_map(|r| find(&r.body))
}

// ===============================================================
// Part C — End-to-end row-set parity (7 tests).
//
// For each leader, run two compiles:
//   * WCOJ ON  + LeaderCardinality config — kernel rotation path.
//   * WCOJ OFF + Disabled config — binary-join reference.
// Assert dispatch counter ≥ 1 (W2.1 path actually ran) and the
// WCOJ row set equals the binary reference.
// ===============================================================

fn assert_triangle_w21_matches_binary_reference(
    fix: &RuntimeBackedFixture,
    snapshot: &StatsSnapshot,
) {
    let inputs = triangle_inputs();

    // Reference: binary-join.
    let exec_ref = run_w21(
        fix,
        no_wcoj_cfg(),
        TRIANGLE_SRC,
        &inputs,
        snapshot,
        &w21_compiler_config(WcojVarOrderingKind::Disabled),
    );
    let ref_rows = match exec_ref.store().get("tri") {
        Some(buf) => download_triples(buf),
        None => Vec::new(),
    };

    // W2.1 path: force-WCOJ + LeaderCardinality.
    let exec_w21 = run_w21(
        fix,
        force_wcoj_cfg(),
        TRIANGLE_SRC,
        &inputs,
        snapshot,
        &w21_compiler_config(WcojVarOrderingKind::LeaderCardinality),
    );
    assert!(
        exec_w21.wcoj_triangle_dispatch_count() >= 1,
        "WCOJ triangle dispatch must have fired ≥ 1 times"
    );
    let w21_rows = match exec_w21.store().get("tri") {
        Some(buf) => download_triples(buf),
        None => Vec::new(),
    };
    assert_eq!(
        w21_rows, ref_rows,
        "W2.1 triangle row set must match binary-join reference"
    );
}

#[test]
fn part_c_triangle_default_leader_e_xy() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    // Default leader is already e1 (e_xy). With LeaderCardinality
    // config, cost model returns None — slice 1 path runs. Row
    // set must still match binary-join reference.
    let snap = make_snapshot(&[("e1", 100), ("e2", 1000), ("e3", 1000)]);
    assert_triangle_w21_matches_binary_reference(&fix, &snap);
}

#[test]
fn part_c_triangle_leader_e_yz() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let snap = make_snapshot(&[("e1", 1000), ("e2", 50), ("e3", 1000)]);
    assert_triangle_w21_matches_binary_reference(&fix, &snap);
}

#[test]
fn part_c_triangle_leader_e_xz() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let snap = make_snapshot(&[("e1", 1000), ("e2", 1000), ("e3", 50)]);
    assert_triangle_w21_matches_binary_reference(&fix, &snap);
}

fn assert_cycle4_w21_matches_binary_reference(
    fix: &RuntimeBackedFixture,
    snapshot: &StatsSnapshot,
) {
    let inputs = cycle4_inputs();
    let exec_ref = run_w21(
        fix,
        no_wcoj_cfg(),
        CYCLE4_SRC,
        &inputs,
        snapshot,
        &w21_compiler_config(WcojVarOrderingKind::Disabled),
    );
    let ref_rows = match exec_ref.store().get("cyc") {
        Some(buf) => download_quads(buf),
        None => Vec::new(),
    };
    let exec_w21 = run_w21(
        fix,
        force_wcoj_cfg(),
        CYCLE4_SRC,
        &inputs,
        snapshot,
        &w21_compiler_config(WcojVarOrderingKind::LeaderCardinality),
    );
    assert!(
        exec_w21.wcoj_4cycle_dispatch_count() >= 1,
        "WCOJ 4-cycle dispatch must have fired ≥ 1 times"
    );
    let w21_rows = match exec_w21.store().get("cyc") {
        Some(buf) => download_quads(buf),
        None => Vec::new(),
    };
    assert_eq!(
        w21_rows, ref_rows,
        "W2.1 4-cycle row set must match binary-join reference"
    );
}

#[test]
fn part_c_cycle4_default_leader_e_wx() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let snap = make_snapshot(&[("e1", 100), ("e2", 1000), ("e3", 1000), ("e4", 1000)]);
    assert_cycle4_w21_matches_binary_reference(&fix, &snap);
}

#[test]
fn part_c_cycle4_leader_e_xy() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let snap = make_snapshot(&[("e1", 1000), ("e2", 50), ("e3", 1000), ("e4", 1000)]);
    assert_cycle4_w21_matches_binary_reference(&fix, &snap);
}

#[test]
fn part_c_cycle4_leader_e_yz() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let snap = make_snapshot(&[("e1", 1000), ("e2", 1000), ("e3", 50), ("e4", 1000)]);
    assert_cycle4_w21_matches_binary_reference(&fix, &snap);
}

#[test]
fn part_c_cycle4_leader_e_zw() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let snap = make_snapshot(&[("e1", 1000), ("e2", 1000), ("e3", 1000), ("e4", 50)]);
    assert_cycle4_w21_matches_binary_reference(&fix, &snap);
}

// ===============================================================
// Part D — Stats-driven divergence (2 tests). Two snapshots
// favoring different leaders → different `var_order.leader_idx`.
// ===============================================================

#[test]
fn part_d_triangle_two_snapshots_produce_different_leader_idx() {
    let snap_a = make_snapshot(&[("e1", 1000), ("e2", 50), ("e3", 1000)]);
    let snap_b = make_snapshot(&[("e1", 1000), ("e2", 1000), ("e3", 50)]);
    let mut compiler_a = Compiler::new();
    let mut compiler_b = Compiler::new();
    let cfg = w21_compiler_config(WcojVarOrderingKind::LeaderCardinality);
    let plan_a = compiler_a
        .compile_with_config_and_stats_snapshot(TRIANGLE_SRC, &cfg, Some(&snap_a))
        .expect("compile a");
    let plan_b = compiler_b
        .compile_with_config_and_stats_snapshot(TRIANGLE_SRC, &cfg, Some(&snap_b))
        .expect("compile b");
    let vo_a = first_var_order(&plan_a).expect("plan_a must set var_order");
    let vo_b = first_var_order(&plan_b).expect("plan_b must set var_order");
    assert_ne!(
        vo_a.leader_idx, vo_b.leader_idx,
        "stats favoring different leaders must produce different leader_idx"
    );
}

#[test]
fn part_d_cycle4_two_snapshots_produce_different_leader_idx() {
    let snap_a = make_snapshot(&[("e1", 1000), ("e2", 50), ("e3", 1000), ("e4", 1000)]);
    let snap_b = make_snapshot(&[("e1", 1000), ("e2", 1000), ("e3", 1000), ("e4", 50)]);
    let mut compiler_a = Compiler::new();
    let mut compiler_b = Compiler::new();
    let cfg = w21_compiler_config(WcojVarOrderingKind::LeaderCardinality);
    let plan_a = compiler_a
        .compile_with_config_and_stats_snapshot(CYCLE4_SRC, &cfg, Some(&snap_a))
        .expect("compile a");
    let plan_b = compiler_b
        .compile_with_config_and_stats_snapshot(CYCLE4_SRC, &cfg, Some(&snap_b))
        .expect("compile b");
    let vo_a = first_var_order(&plan_a).expect("plan_a must set var_order");
    let vo_b = first_var_order(&plan_b).expect("plan_b must set var_order");
    assert_ne!(vo_a.leader_idx, vo_b.leader_idx);
}

// ===============================================================
// Part E — Threshold gate cert (2 tests). Pin the 0.5 ratio
// boundary policy. Marginal cases above 0.5 must NOT trigger
// var_order; clear wins below 0.5 must trigger.
// ===============================================================

#[test]
fn part_e_marginal_leader_cardinality_does_not_trigger_var_order() {
    // ratio = 600 / 1000 = 0.6 → above threshold → None.
    // Triangle, e_yz candidate.
    let snap = make_snapshot(&[("e1", 1000), ("e2", 600), ("e3", 1000)]);
    let mut compiler = Compiler::new();
    let cfg = w21_compiler_config(WcojVarOrderingKind::LeaderCardinality);
    let plan = compiler
        .compile_with_config_and_stats_snapshot(TRIANGLE_SRC, &cfg, Some(&snap))
        .expect("compile");
    let vo = first_var_order(&plan);
    assert!(
        vo.is_none(),
        "ratio = 0.6 (above 0.5) must leave var_order = None, got {:?}",
        vo
    );
}

#[test]
fn part_e_clear_win_leader_cardinality_triggers_var_order() {
    // ratio = 300 / 1000 = 0.3 → at or below threshold → Some.
    // Triangle, e_yz leader (canonical idx 1).
    let snap = make_snapshot(&[("e1", 1000), ("e2", 300), ("e3", 1000)]);
    let mut compiler = Compiler::new();
    let cfg = w21_compiler_config(WcojVarOrderingKind::LeaderCardinality);
    let plan = compiler
        .compile_with_config_and_stats_snapshot(TRIANGLE_SRC, &cfg, Some(&snap))
        .expect("compile");
    let vo = first_var_order(&plan).expect("ratio = 0.3 must trigger var_order = Some");
    assert_eq!(vo.leader_idx, 1, "e_yz leader is canonical idx 1");
}
