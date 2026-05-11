//! W2.6 step 7 Parts C/D/E — real-runtime end-to-end certs
//! locking the heat-aware leader selection contract.
//!
//! * Part C (3 tests): real runtime-observed signals
//!   (`record_join_result` selectivity + `record_access` heat)
//!   captured via `Executor::stats_snapshot()` drive a HeatAware
//!   leader change vs the LeaderCardinality baseline on the same
//!   snapshot. Row-set parity vs binary-join reference holds.
//! * Part D (2 tests): default `CompilerConfig::default()` preserves
//!   row-set parity across the W2.3 skew baseline and W2.5 cardinality
//!   runtime default; W2.4 feedback's canonical `(slot_rels[0],
//!   slot_rels[1])` pair with `[1]/[0]` keys is preserved when
//!   `var_order = None`.
//! * Part E (1 test): when HeatAware emits a non-default leader
//!   on triangle (idx 2), W2.6's `feedback_pair_from_var_order`
//!   reroute records selectivity on the **rotated** pair
//!   (canonicalized) with `[1]/[1]` keys — proving the W2.6
//!   step-5 contract end-to-end.

use std::collections::BTreeMap;
use std::sync::Arc;

use cudarc::driver::sys;
use xlog_core::{CostModelKind, MemoryBudget, RelId, RuntimeConfig, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::CudaBuffer;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_ir::rir::VariableOrder;
use xlog_ir::RirNode;
use xlog_logic::compiler_config::{CompilerConfig, WcojVarOrderingKind};
use xlog_logic::Compiler;
use xlog_runtime::Executor;
use xlog_stats::{JoinSelectivity, RelationStats, StatsSnapshot};

// ---------------------------------------------------------------
// CUDA fixture (mirror of W2.4 `test_wcoj_record_join_result_feedback`)
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

// ---------------------------------------------------------------
// Plan inspection helpers
// ---------------------------------------------------------------

/// Walk the plan tree to find the first `MultiWayJoin.var_order`.
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

/// Mirrors `StatsManager::canonical_join_key`: smaller RelId on left.
fn canonical_pair(a: RelId, b: RelId) -> (RelId, RelId) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

// ---------------------------------------------------------------
// Triangle non-recursive fixture (Part C.1, D.2, E.1)
// ---------------------------------------------------------------

const TRI_NONREC_SRC: &str = r#"
    pred e1(u32, u32). pred e2(u32, u32). pred e3(u32, u32).
    pred tri(u32, u32, u32).
    tri(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z).
"#;

fn tri_nonrec_inputs() -> BTreeMap<&'static str, Vec<(u32, u32)>> {
    // 5 rows each EDB; only (1,2)+(2,3)+(1,3) joins → 1 triangle row.
    let mut m: BTreeMap<&'static str, Vec<(u32, u32)>> = BTreeMap::new();
    m.insert("e1", vec![(1, 2), (10, 99), (20, 98), (30, 97), (40, 96)]);
    m.insert("e2", vec![(2, 3), (50, 51), (60, 61), (70, 71), (80, 81)]);
    m.insert("e3", vec![(1, 3), (50, 52), (60, 62), (70, 72), (80, 82)]);
    m
}

// ---------------------------------------------------------------
// 4-cycle non-recursive fixture (Part C.3)
// ---------------------------------------------------------------

const CYC4_NONREC_SRC: &str = r#"
    pred e1(u32, u32). pred e2(u32, u32). pred e3(u32, u32). pred e4(u32, u32).
    pred cyc(u32, u32, u32, u32).
    cyc(W, X, Y, Z) :- e1(W, X), e2(X, Y), e3(Y, Z), e4(Z, W).
"#;

fn cyc4_nonrec_inputs() -> BTreeMap<&'static str, Vec<(u32, u32)>> {
    // 5 rows each EDB; only (1,2)+(2,3)+(3,4)+(4,1) joins → 1 cyc row.
    let mut m: BTreeMap<&'static str, Vec<(u32, u32)>> = BTreeMap::new();
    m.insert("e1", vec![(1, 2), (10, 99), (20, 98), (30, 97), (40, 96)]);
    m.insert("e2", vec![(2, 3), (50, 51), (60, 61), (70, 71), (80, 81)]);
    m.insert("e3", vec![(3, 4), (50, 52), (60, 62), (70, 72), (80, 82)]);
    m.insert("e4", vec![(4, 1), (50, 53), (60, 63), (70, 73), (80, 83)]);
    m
}

// ---------------------------------------------------------------
// Slice-4 anchor (Part D.1) — copied from
// `test_wcoj_recursive_dispatch::LINEAR_REC_TRIANGLE`.
// ---------------------------------------------------------------

const LINEAR_REC_TRIANGLE: &str = r#"
    pred e1_seed(u32, u32).
    pred e1(u32, u32).
    pred e2(u32, u32).
    pred e3(u32, u32).
    pred tri(u32, u32, u32).
    e1(X, Y) :- e1_seed(X, Y).
    e1(X, Y) :- tri(X, Z, Y).
    tri(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z).
"#;

fn linear_rec_triangle_inputs() -> BTreeMap<&'static str, Vec<(u32, u32)>> {
    let mut m: BTreeMap<&'static str, Vec<(u32, u32)>> = BTreeMap::new();
    m.insert("e1_seed", vec![(1, 2)]);
    m.insert("e2", vec![(2, 3), (3, 4)]);
    m.insert("e3", vec![(1, 3), (1, 4)]);
    m
}

// ---------------------------------------------------------------
// Common helpers for compile + execute
// ---------------------------------------------------------------

/// Build executor, register predicates from the compiler, upload
/// EDBs, and seed cardinalities (W2.4 missing-cards safety floor
/// requires explicit `update_cardinality` for `record_join_result`
/// to fire — `put_relation` alone does NOT seed `StatsManager`).
fn build_executor(
    provider: Arc<CudaKernelProvider>,
    memory: &Arc<GpuMemoryManager>,
    runtime_config: RuntimeConfig,
    compiler: &Compiler,
    inputs: &BTreeMap<&str, Vec<(u32, u32)>>,
    seeded_cards: &BTreeMap<&str, u64>,
) -> Executor {
    let mut executor = Executor::new_with_config(provider, runtime_config);
    for (name, rid) in compiler.rel_ids().clone() {
        executor.register_relation(rid, &name);
    }
    for (name, rows) in inputs {
        let buf = upload_binary_u32(memory, rows);
        executor.put_relation(name, buf);
    }
    for (name, card) in seeded_cards {
        if let Some(rid) = compiler.rel_ids().get(*name) {
            executor.stats_mut().register_relation(*rid);
            executor.stats_mut().update_cardinality(*rid, *card);
        }
    }
    executor
}

// ===============================================================
// Part C.1 — Real selectivity drives leader for triangle
// ===============================================================

#[test]
fn triangle_real_observed_selectivity_drives_heat_aware_leader_to_idx_2() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    // Phase 1: warm-up under default config + force-WCOJ-on.
    // 4 sequential execute_plan calls feed `record_join_result`
    // 4× via the W2.4 EMA path. Cards equal at 5 (no recursion).
    let mut compiler = Compiler::new();
    let cfg_default = CompilerConfig::default();
    let plan_default = compiler
        .compile_with_config_and_stats_snapshot(TRI_NONREC_SRC, &cfg_default, None)
        .expect("compile default");

    let inputs = tri_nonrec_inputs();
    // Seed card=5 to match plan iteration 7 — actual EDB size.
    // Slice-1 promoter's right-deep normalizer (W2.6) handles
    // the lowerer's bushy DP choice of right-deep trees at this
    // small scale.
    let mut seeded = BTreeMap::new();
    seeded.insert("e1", 5u64);
    seeded.insert("e2", 5u64);
    seeded.insert("e3", 5u64);
    let mut executor = build_executor(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(true)),
        &compiler,
        &inputs,
        &seeded,
    );
    for _ in 0..4 {
        let _ = executor.execute_plan(&plan_default).expect("execute_plan");
    }
    assert_eq!(
        executor.wcoj_triangle_dispatch_count(),
        4,
        "4 execute_plan calls on non-recursive plan must yield 4 dispatches"
    );

    // Phase 2: snapshot pins equal-card invariant + EMA selectivity
    // converged to ~0.270 on the canonical (rel_xy, rel_yz) pair.
    // Math: input_rows = 5*5 = 25; output_rows = 1;
    // observed_sel = 0.04. EMA(0.7-old, 0.3-new) for 4 steps
    // converges to ~0.270.
    let snap = executor.stats_snapshot();
    let rel_xy = *compiler.rel_ids().get("e1").expect("e1 rel_id");
    let rel_yz = *compiler.rel_ids().get("e2").expect("e2 rel_id");
    let rel_xz = *compiler.rel_ids().get("e3").expect("e3 rel_id");
    for (label, rid) in [("rel_xy", rel_xy), ("rel_yz", rel_yz), ("rel_xz", rel_xz)] {
        let card = snap
            .relations
            .iter()
            .find(|r| r.rel_id == rid)
            .map(|r| r.cardinality)
            .unwrap_or(0);
        assert_eq!(
            card, 5,
            "{} card must remain at seeded 5 (force-WCOJ-on bypasses execute_scan auto-update); got {}",
            label, card
        );
    }
    let canon_xy_yz = canonical_pair(rel_xy, rel_yz);
    let entries: Vec<&JoinSelectivity> = snap
        .join_selectivities
        .iter()
        .filter(|js| (js.left_rel, js.right_rel) == canon_xy_yz)
        .collect();
    assert_eq!(
        entries.len(),
        1,
        "exactly one canonical (rel_xy, rel_yz) selectivity entry expected"
    );
    let sel_xy_yz = entries[0].selectivity;
    assert!(
        (0.25..=0.30).contains(&sel_xy_yz),
        "EMA selectivity after 4 dispatches at card=5 must be in [0.25, 0.30]; got {}",
        sel_xy_yz
    );

    // Phase 3: re-compile under HeatAware + this snapshot.
    // Score: rel_xy ~ 23.50, rel_yz ~ 23.50, rel_xz ~ 10.00.
    // argmin = idx 2 (rel_xz). Ratio 10/23.5 ≈ 0.425 ≤ 0.5 → Some(2).
    let mut compiler_heat = Compiler::new();
    let cfg_heat = CompilerConfig {
        wcoj_variable_ordering: WcojVarOrderingKind::HeatAware,
        ..CompilerConfig::default()
    };
    let plan_heat = compiler_heat
        .compile_with_config_and_stats_snapshot(TRI_NONREC_SRC, &cfg_heat, Some(&snap))
        .expect("compile HeatAware");
    let vo_heat = first_var_order(&plan_heat).expect("HeatAware must set var_order");
    assert_eq!(
        vo_heat.leader_idx, 2,
        "HeatAware on selectivity-biased snapshot must pick leader_idx = 2"
    );

    // Phase 4: same snapshot under LeaderCardinality → None
    // (cards equal, W2.1 short-circuits). This proves the leader
    // change is selectivity-driven, NOT cardinality-driven.
    let mut compiler_card = Compiler::new();
    let cfg_card = CompilerConfig {
        wcoj_variable_ordering: WcojVarOrderingKind::LeaderCardinality,
        ..CompilerConfig::default()
    };
    let plan_card = compiler_card
        .compile_with_config_and_stats_snapshot(TRI_NONREC_SRC, &cfg_card, Some(&snap))
        .expect("compile LeaderCardinality");
    assert!(
        first_var_order(&plan_card).is_none(),
        "LeaderCardinality on equal-card snapshot must return None"
    );

    // Phase 5: row-set parity vs binary-join reference.
    let mut compiler_ref = Compiler::new();
    let plan_ref = compiler_ref
        .compile_with_config_and_stats_snapshot(TRI_NONREC_SRC, &cfg_default, None)
        .expect("compile reference");
    let mut executor_ref = build_executor(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default().with_wcoj_triangle_dispatch_disabled(Some(true)),
        &compiler_ref,
        &inputs,
        &seeded,
    );
    let _ = executor_ref
        .execute_plan(&plan_ref)
        .expect("execute reference");
    let rows_ref = download_triples(executor_ref.store().get("tri").expect("tri ref"));

    let mut executor_heat = build_executor(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(true)),
        &compiler_heat,
        &inputs,
        &seeded,
    );
    let _ = executor_heat
        .execute_plan(&plan_heat)
        .expect("execute HeatAware");
    let rows_heat = download_triples(executor_heat.store().get("tri").expect("tri heat"));
    assert_eq!(
        rows_ref, rows_heat,
        "HeatAware leader_idx=2 must produce same row set as binary-join reference"
    );
    assert_eq!(rows_heat, vec![(1, 2, 3)], "expected exactly {{(1,2,3)}}");
}

// ===============================================================
// Part C.2 — Real heat drives leader for triangle
// ===============================================================

/// Heater-only source — `dummy_e1` projects e1, no tri rule.
/// The binary-join path auto-records `record_join_result`
/// after every hash join (`node_dispatch.rs:343`), so any rule
/// containing `Join` would create a selectivity entry that
/// perturbs the heat-only signal. Single-Scan rule keeps the
/// snapshot's `join_selectivities` empty.
const TRI_HEAT_HEATER_SRC: &str = r#"
    pred e1(u32, u32).
    pred e2(u32, u32).
    pred e3(u32, u32).
    pred dummy_e1(u32).
    dummy_e1(X) :- e1(X, _).
"#;

#[test]
fn triangle_real_observed_heat_drives_heat_aware_leader_to_idx_1() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    // Same Compiler instance across all phases preserves rel_id
    // assignments for shared predicates (e1, e2, e3).
    let mut compiler = Compiler::new();
    let cfg_default = CompilerConfig::default();

    // Phase A+B (merged): heater-only source `dummy_e1(X) :- e1(X, _).`,
    // 11 sequential `execute_plan` calls under triangle-WCOJ kill
    // switch. Each call scans e1 once (record_access advances heat
    // EMA). e2/e3 are NEVER scanned in this rule, so their heat
    // stays at the initial 0.0 — the differential signal Phase D
    // needs. The earlier "combined dummy_e1 + tri" Phase A would
    // bake a `record_join_result` selectivity entry from the
    // binary-join `tri` rule (`node_dispatch.rs:343` calls
    // `record_join_result` after every hash join), perturbing the
    // intended heat-only signal — splitting it out keeps the cert
    // purely heat-driven.
    let plan_heater = compiler
        .compile_with_config_and_stats_snapshot(TRI_HEAT_HEATER_SRC, &cfg_default, None)
        .expect("compile heater");
    let inputs = tri_nonrec_inputs();
    let mut seeded = BTreeMap::new();
    seeded.insert("e1", 5u64);
    seeded.insert("e2", 5u64);
    seeded.insert("e3", 5u64);
    let mut executor = build_executor(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default().with_wcoj_triangle_dispatch_disabled(Some(true)),
        &compiler,
        &inputs,
        &seeded,
    );
    for _ in 0..11 {
        let _ = executor.execute_plan(&plan_heater).expect("execute heater");
    }

    let rel_e1 = *compiler.rel_ids().get("e1").expect("e1 rel_id");
    let rel_e2 = *compiler.rel_ids().get("e2").expect("e2 rel_id");
    let rel_e3 = *compiler.rel_ids().get("e3").expect("e3 rel_id");

    // Refresh e1's card back to 5 — execute_scan auto-updates to
    // actual buffer rows (5) which already matches our seed; e2/e3
    // keep their seeded value since they're never scanned.
    executor.stats_mut().update_cardinality(rel_e1, 5);

    // Phase C: pin heat differential — e1 heated by 11 EMA steps
    // (1 - 0.9^11 ≈ 0.686), e2/e3 untouched at 0.
    let snap = executor.stats_snapshot();
    let heat_e1 = snap
        .relations
        .iter()
        .find(|r| r.rel_id == rel_e1)
        .map(|r| r.heat)
        .unwrap_or(0.0);
    let heat_e2 = snap
        .relations
        .iter()
        .find(|r| r.rel_id == rel_e2)
        .map(|r| r.heat)
        .unwrap_or(0.0);
    let heat_e3 = snap
        .relations
        .iter()
        .find(|r| r.rel_id == rel_e3)
        .map(|r| r.heat)
        .unwrap_or(0.0);
    assert!(
        heat_e1 >= 0.6,
        "11 heater calls must drive e1 heat ≥ 0.6; got {}",
        heat_e1
    );
    assert!(
        heat_e2 <= 0.05,
        "e2 heat must remain near 0 (never scanned); got {}",
        heat_e2
    );
    assert!(
        heat_e3 <= 0.05,
        "e3 heat must remain near 0 (never scanned); got {}",
        heat_e3
    );
    assert!(
        snap.join_selectivities.is_empty(),
        "heater-only warm-up must not write any join_selectivities; got {:?}",
        snap.join_selectivities
    );

    // Phase D: re-compile triangle-only under HeatAware + snapshot.
    // No selectivity records (heater-only warm-up). Penalty per
    // rel = 1 + 1 = 2. Heat factors: e1 = 1+4*0.686 = 3.744;
    // e2/e3 = 1.0. Cards uniform 5.
    //   score(e1) = 5 * 3.744 * 2 = 37.44
    //   score(e2) = score(e3) = 5 * 1.0 * 2 = 10
    // argmin = idx 1 (e2, first-hit ties). Ratio 10/37.44 ≈ 0.267
    // ≤ 0.5 → Some(1).
    let mut compiler_heat = Compiler::new();
    let cfg_heat = CompilerConfig {
        wcoj_variable_ordering: WcojVarOrderingKind::HeatAware,
        ..CompilerConfig::default()
    };
    let plan_heat = compiler_heat
        .compile_with_config_and_stats_snapshot(TRI_NONREC_SRC, &cfg_heat, Some(&snap))
        .expect("compile HeatAware");
    let vo_heat = first_var_order(&plan_heat).expect("HeatAware must set var_order");
    assert_eq!(
        vo_heat.leader_idx, 1,
        "HeatAware on heat-biased snapshot must pick leader_idx = 1"
    );

    // Same snapshot under LeaderCardinality → None (cards equal).
    let mut compiler_card = Compiler::new();
    let cfg_card = CompilerConfig {
        wcoj_variable_ordering: WcojVarOrderingKind::LeaderCardinality,
        ..CompilerConfig::default()
    };
    let plan_card = compiler_card
        .compile_with_config_and_stats_snapshot(TRI_NONREC_SRC, &cfg_card, Some(&snap))
        .expect("compile LeaderCardinality");
    assert!(
        first_var_order(&plan_card).is_none(),
        "LeaderCardinality on equal-card snapshot must return None"
    );

    // Phase E: row-set parity vs binary-join reference.
    let mut compiler_ref = Compiler::new();
    let plan_ref = compiler_ref
        .compile_with_config_and_stats_snapshot(TRI_NONREC_SRC, &cfg_default, None)
        .expect("compile reference");
    let mut executor_ref = build_executor(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default().with_wcoj_triangle_dispatch_disabled(Some(true)),
        &compiler_ref,
        &inputs,
        &seeded,
    );
    let _ = executor_ref
        .execute_plan(&plan_ref)
        .expect("execute reference");
    let rows_ref = download_triples(executor_ref.store().get("tri").expect("tri ref"));

    let mut executor_heat = build_executor(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(true)),
        &compiler_heat,
        &inputs,
        &seeded,
    );
    let _ = executor_heat
        .execute_plan(&plan_heat)
        .expect("execute HeatAware");
    let rows_heat = download_triples(executor_heat.store().get("tri").expect("tri heat"));
    assert_eq!(
        rows_ref, rows_heat,
        "HeatAware leader_idx=1 must produce same row set as binary-join reference"
    );
    assert_eq!(rows_heat, vec![(1, 2, 3)], "expected exactly {{(1,2,3)}}");
}

// ===============================================================
// Part C.4 — Real heat drives leader; non-zero cold-baseline
// ===============================================================
//
// Plan iteration 7 originally specified Phase A with combined
// `dummy_e1 + tri` source so each rel got a baseline scan giving
// `e2/e3.heat ≈ 0.1`. Implementing that would create a
// `record_join_result` selectivity entry from the binary-join
// `tri` rule (`node_dispatch.rs:343` calls record_join_result
// after every hash join), perturbing the heat-only signal.
//
// The triple-dummy source `dummy_e1 + dummy_e2 + dummy_e3` (no
// joins, three single-Scan rules) achieves the plan's intent —
// non-zero baseline heat for every rel — without any
// `record_join_result` side effect.

const TRI_HEAT_TRIPLE_DUMMY_SRC: &str = r#"
    pred e1(u32, u32). pred e2(u32, u32). pred e3(u32, u32).
    pred dummy_e1(u32). pred dummy_e2(u32). pred dummy_e3(u32).
    dummy_e1(X) :- e1(X, _).
    dummy_e2(X) :- e2(X, _).
    dummy_e3(X) :- e3(X, _).
"#;

#[test]
fn triangle_real_observed_heat_with_baseline_drives_heat_aware_leader_to_idx_1() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    // Same Compiler instance across all phases preserves rel_id
    // assignments for shared predicates.
    let mut compiler = Compiler::new();
    let cfg_default = CompilerConfig::default();

    // Phase A (baseline): triple-dummy source, 1 execute_plan
    // call → each rel scanned exactly once (one record_access
    // per scan). e1.heat = e2.heat = e3.heat = 0.1.
    let plan_baseline = compiler
        .compile_with_config_and_stats_snapshot(TRI_HEAT_TRIPLE_DUMMY_SRC, &cfg_default, None)
        .expect("compile baseline");
    let inputs = tri_nonrec_inputs();
    let mut seeded = BTreeMap::new();
    seeded.insert("e1", 5u64);
    seeded.insert("e2", 5u64);
    seeded.insert("e3", 5u64);
    let mut executor = build_executor(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default().with_wcoj_triangle_dispatch_disabled(Some(true)),
        &compiler,
        &inputs,
        &seeded,
    );
    let _ = executor
        .execute_plan(&plan_baseline)
        .expect("execute baseline");

    let rel_e1 = *compiler.rel_ids().get("e1").expect("e1 rel_id");
    let rel_e2 = *compiler.rel_ids().get("e2").expect("e2 rel_id");
    let rel_e3 = *compiler.rel_ids().get("e3").expect("e3 rel_id");

    // Phase B (heater): dummy_e1 only, 11 calls. e1 heat
    // advances from 0.1 to 1 - 0.9^12 ≈ 0.7176; e2 / e3 stay
    // at 0.1.
    let plan_heater = compiler
        .compile_with_config_and_stats_snapshot(TRI_HEAT_HEATER_SRC, &cfg_default, None)
        .expect("compile heater");
    for _ in 0..11 {
        let _ = executor.execute_plan(&plan_heater).expect("execute heater");
    }

    // execute_scan auto-updates cards to actual buffer rows (5)
    // each call — already matches our card=5 seed for e1; e2 / e3
    // were updated in Phase A and stay at 5.

    // Phase C: pin heat differential — non-zero baseline.
    let snap = executor.stats_snapshot();
    let heat_e1 = snap
        .relations
        .iter()
        .find(|r| r.rel_id == rel_e1)
        .map(|r| r.heat)
        .unwrap_or(0.0);
    let heat_e2 = snap
        .relations
        .iter()
        .find(|r| r.rel_id == rel_e2)
        .map(|r| r.heat)
        .unwrap_or(0.0);
    let heat_e3 = snap
        .relations
        .iter()
        .find(|r| r.rel_id == rel_e3)
        .map(|r| r.heat)
        .unwrap_or(0.0);
    assert!(
        heat_e1 >= 0.6,
        "Phase A+B must drive e1 heat ≥ 0.6; got {}",
        heat_e1
    );
    // e2/e3 picked up exactly one scan in Phase A → heat = 0.1.
    // Band [0.05, 0.15] keeps the cert robust to ± single
    // additional scan in case of optimizer quirks.
    assert!(
        (0.05..=0.15).contains(&heat_e2),
        "e2 heat must reflect Phase-A baseline scan (≈ 0.1, band [0.05, 0.15]); got {}",
        heat_e2
    );
    assert!(
        (0.05..=0.15).contains(&heat_e3),
        "e3 heat must reflect Phase-A baseline scan (≈ 0.1, band [0.05, 0.15]); got {}",
        heat_e3
    );
    assert!(
        snap.join_selectivities.is_empty(),
        "triple-dummy + heater warm-up must not write any join_selectivities; got {:?}",
        snap.join_selectivities
    );

    // Phase D: re-compile triangle-only under HeatAware + snapshot.
    // Score (card=5):
    //   factor(e1) = 1 + 4*0.7176 ≈ 3.870
    //   factor(e2) = factor(e3) = 1 + 4*0.1 = 1.4
    //   penalty per rel = 2 (no selectivity records)
    //   score(e1) = 5 * 3.870 * 2 ≈ 38.70
    //   score(e2) = score(e3) = 5 * 1.4 * 2 = 14
    // argmin = idx 1 (e2, first-hit ties). Ratio 14/38.70 ≈ 0.362
    // ≤ 0.5 → Some(1). The non-zero-baseline case the plan
    // intended.
    let mut compiler_heat = Compiler::new();
    let cfg_heat = CompilerConfig {
        wcoj_variable_ordering: WcojVarOrderingKind::HeatAware,
        ..CompilerConfig::default()
    };
    let plan_heat = compiler_heat
        .compile_with_config_and_stats_snapshot(TRI_NONREC_SRC, &cfg_heat, Some(&snap))
        .expect("compile HeatAware");
    let vo_heat = first_var_order(&plan_heat).expect("HeatAware must set var_order");
    assert_eq!(
        vo_heat.leader_idx, 1,
        "HeatAware on heat-biased snapshot (non-zero cold baseline) must pick leader_idx = 1"
    );

    let mut compiler_card = Compiler::new();
    let cfg_card = CompilerConfig {
        wcoj_variable_ordering: WcojVarOrderingKind::LeaderCardinality,
        ..CompilerConfig::default()
    };
    let plan_card = compiler_card
        .compile_with_config_and_stats_snapshot(TRI_NONREC_SRC, &cfg_card, Some(&snap))
        .expect("compile LeaderCardinality");
    assert!(
        first_var_order(&plan_card).is_none(),
        "LeaderCardinality on equal-card snapshot must return None"
    );

    // Phase E: row-set parity vs binary-join reference.
    let mut compiler_ref = Compiler::new();
    let plan_ref = compiler_ref
        .compile_with_config_and_stats_snapshot(TRI_NONREC_SRC, &cfg_default, None)
        .expect("compile reference");
    let mut executor_ref = build_executor(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default().with_wcoj_triangle_dispatch_disabled(Some(true)),
        &compiler_ref,
        &inputs,
        &seeded,
    );
    let _ = executor_ref
        .execute_plan(&plan_ref)
        .expect("execute reference");
    let rows_ref = download_triples(executor_ref.store().get("tri").expect("tri ref"));

    let mut executor_heat = build_executor(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(true)),
        &compiler_heat,
        &inputs,
        &seeded,
    );
    let _ = executor_heat
        .execute_plan(&plan_heat)
        .expect("execute HeatAware");
    let rows_heat = download_triples(executor_heat.store().get("tri").expect("tri heat"));
    assert_eq!(
        rows_ref, rows_heat,
        "non-zero-baseline HeatAware leader_idx=1 must produce same row set as binary-join reference"
    );
    assert_eq!(rows_heat, vec![(1, 2, 3)], "expected exactly {{(1,2,3)}}");
}

// ===============================================================
// Part C.3 — Real selectivity drives leader for 4-cycle
// ===============================================================

#[test]
fn cycle4_real_observed_selectivity_drives_heat_aware_leader_to_idx_2() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    // Phase 1: warm-up under default config + force-4cycle-on.
    let mut compiler = Compiler::new();
    let cfg_default = CompilerConfig::default();
    let plan_default = compiler
        .compile_with_config_and_stats_snapshot(CYC4_NONREC_SRC, &cfg_default, None)
        .expect("compile default");
    let inputs = cyc4_nonrec_inputs();
    // Seed card=5 (plan iteration 7 — actual EDB size).
    let mut seeded = BTreeMap::new();
    seeded.insert("e1", 5u64);
    seeded.insert("e2", 5u64);
    seeded.insert("e3", 5u64);
    seeded.insert("e4", 5u64);
    let mut executor = build_executor(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default().with_wcoj_4cycle_dispatch(Some(true)),
        &compiler,
        &inputs,
        &seeded,
    );
    for _ in 0..4 {
        let _ = executor.execute_plan(&plan_default).expect("execute_plan");
    }
    assert_eq!(
        executor.wcoj_4cycle_dispatch_count(),
        4,
        "4 execute_plan calls must yield 4 4-cycle dispatches"
    );

    let snap = executor.stats_snapshot();
    let rel_e1 = *compiler.rel_ids().get("e1").expect("e1 rel_id");
    let rel_e2 = *compiler.rel_ids().get("e2").expect("e2 rel_id");
    let rel_e3 = *compiler.rel_ids().get("e3").expect("e3 rel_id");
    let rel_e4 = *compiler.rel_ids().get("e4").expect("e4 rel_id");
    for (label, rid) in [
        ("e1", rel_e1),
        ("e2", rel_e2),
        ("e3", rel_e3),
        ("e4", rel_e4),
    ] {
        let card = snap
            .relations
            .iter()
            .find(|r| r.rel_id == rid)
            .map(|r| r.cardinality)
            .unwrap_or(0);
        assert_eq!(
            card, 5,
            "{} card must remain at seeded 5; got {}",
            label, card
        );
    }
    let canon_e1_e2 = canonical_pair(rel_e1, rel_e2);
    let entries: Vec<&JoinSelectivity> = snap
        .join_selectivities
        .iter()
        .filter(|js| (js.left_rel, js.right_rel) == canon_e1_e2)
        .collect();
    assert_eq!(
        entries.len(),
        1,
        "exactly one canonical (e1, e2) selectivity entry expected"
    );
    let sel_e1_e2 = entries[0].selectivity;
    assert!(
        (0.25..=0.30).contains(&sel_e1_e2),
        "4-cycle EMA selectivity after 4 dispatches at card=5 must be in [0.25, 0.30]; got {}",
        sel_e1_e2
    );

    // Phase 3: re-compile under HeatAware + snapshot.
    // Score: e1 ~ 23.50, e2 ~ 23.50, e3 ~ 10.00, e4 ~ 10.00.
    // argmin = idx 2 (e3, first-hit ties). Ratio 0.425 ≤ 0.5 → Some(2).
    let mut compiler_heat = Compiler::new();
    let cfg_heat = CompilerConfig {
        wcoj_variable_ordering: WcojVarOrderingKind::HeatAware,
        ..CompilerConfig::default()
    };
    let plan_heat = compiler_heat
        .compile_with_config_and_stats_snapshot(CYC4_NONREC_SRC, &cfg_heat, Some(&snap))
        .expect("compile HeatAware");
    let vo_heat = first_var_order(&plan_heat).expect("HeatAware must set var_order");
    assert_eq!(
        vo_heat.leader_idx, 2,
        "HeatAware on selectivity-biased 4-cycle snapshot must pick leader_idx = 2"
    );

    let mut compiler_card = Compiler::new();
    let cfg_card = CompilerConfig {
        wcoj_variable_ordering: WcojVarOrderingKind::LeaderCardinality,
        ..CompilerConfig::default()
    };
    let plan_card = compiler_card
        .compile_with_config_and_stats_snapshot(CYC4_NONREC_SRC, &cfg_card, Some(&snap))
        .expect("compile LeaderCardinality");
    assert!(
        first_var_order(&plan_card).is_none(),
        "LeaderCardinality on equal-card 4-cycle snapshot must return None"
    );

    // Row-set parity vs binary-join reference.
    let mut compiler_ref = Compiler::new();
    let plan_ref = compiler_ref
        .compile_with_config_and_stats_snapshot(CYC4_NONREC_SRC, &cfg_default, None)
        .expect("compile reference");
    let mut executor_ref = build_executor(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default().with_wcoj_4cycle_dispatch(Some(false)),
        &compiler_ref,
        &inputs,
        &seeded,
    );
    let _ = executor_ref
        .execute_plan(&plan_ref)
        .expect("execute reference");
    let rows_ref = download_quads(executor_ref.store().get("cyc").expect("cyc ref"));

    let mut executor_heat = build_executor(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default().with_wcoj_4cycle_dispatch(Some(true)),
        &compiler_heat,
        &inputs,
        &seeded,
    );
    let _ = executor_heat
        .execute_plan(&plan_heat)
        .expect("execute HeatAware");
    let rows_heat = download_quads(executor_heat.store().get("cyc").expect("cyc heat"));
    assert_eq!(
        rows_ref, rows_heat,
        "4-cycle HeatAware leader_idx=2 must produce same row set as reference"
    );
    assert_eq!(
        rows_heat,
        vec![(1, 2, 3, 4)],
        "expected exactly {{(1,2,3,4)}}"
    );
}

// ===============================================================
// Part D.1 — default compiler config row parity across runtime cost models
// ===============================================================

#[test]
fn default_compiler_config_preserves_rows_across_runtime_cost_models() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    // Reference: gate-OFF (binary join only).
    let mut compiler_ref = Compiler::new();
    let plan_ref = compiler_ref
        .compile_with_config_and_stats_snapshot(
            LINEAR_REC_TRIANGLE,
            &CompilerConfig::default(),
            None,
        )
        .expect("compile reference");
    let inputs = linear_rec_triangle_inputs();
    let mut executor_ref = build_executor(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default().with_wcoj_triangle_dispatch_disabled(Some(true)),
        &compiler_ref,
        &inputs,
        &BTreeMap::new(),
    );
    let _ = executor_ref
        .execute_plan(&plan_ref)
        .expect("execute reference");
    let rows_ref = download_triples(executor_ref.store().get("tri").expect("tri ref"));
    assert_eq!(
        executor_ref.wcoj_triangle_dispatch_count(),
        0,
        "gate-OFF must not dispatch"
    );

    // Explicit legacy skew runtime: W2.3 baseline counter remains pinned.
    let mut compiler_skew = Compiler::new();
    let plan_skew = compiler_skew
        .compile_with_config_and_stats_snapshot(
            LINEAR_REC_TRIANGLE,
            &CompilerConfig::default(),
            None,
        )
        .expect("compile explicit skew");
    let mut executor_skew = build_executor(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default().with_wcoj_cost_model(Some(CostModelKind::SkewClassifier)),
        &compiler_skew,
        &inputs,
        &BTreeMap::new(),
    );
    let _ = executor_skew
        .execute_plan(&plan_skew)
        .expect("execute explicit skew");
    let rows_skew = download_triples(executor_skew.store().get("tri").expect("tri skew"));
    // Slice-4 anchor: 1 seeding + 1 e1_delta(1,3) + 1 e1_delta(1,4)
    // = 3 dispatches (last iter has empty delta, skips).
    assert_eq!(
        executor_skew.wcoj_triangle_dispatch_count(),
        3,
        "explicit skew slice-4 baseline counter must remain at 3"
    );
    assert_eq!(
        rows_skew, rows_ref,
        "explicit skew row set must match binary-join reference"
    );

    // Bare W2.5 runtime default: CardinalityAwareCostModel sees the
    // small scan-populated cards in this linear-recursive fixture and
    // keeps the binary path, while preserving row-set parity.
    let mut compiler_def = Compiler::new();
    let plan_def = compiler_def
        .compile_with_config_and_stats_snapshot(
            LINEAR_REC_TRIANGLE,
            &CompilerConfig::default(),
            None,
        )
        .expect("compile bare default");
    let mut executor_def = build_executor(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default(),
        &compiler_def,
        &inputs,
        &BTreeMap::new(),
    );
    let _ = executor_def
        .execute_plan(&plan_def)
        .expect("execute bare default");
    let rows_def = download_triples(executor_def.store().get("tri").expect("tri def"));
    assert_eq!(
        executor_def.wcoj_triangle_dispatch_count(),
        0,
        "bare W2.5 default must keep the binary path on this small-cardinality fixture"
    );
    assert_eq!(
        rows_def, rows_ref,
        "bare W2.5 default row set must match binary-join reference"
    );
}

// ===============================================================
// Part D.2 — var_order=None pair unchanged (pre-W2.6 baseline)
// ===============================================================

#[test]
fn record_wcoj_feedback_var_order_none_pair_unchanged() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    // Default compiler config → var_order = None for the
    // promoted MultiWayJoin → feedback uses canonical
    // (slot_rels[0], slot_rels[1]) with [1]/[0] keys.
    let mut compiler = Compiler::new();
    let plan = compiler
        .compile_with_config_and_stats_snapshot(TRI_NONREC_SRC, &CompilerConfig::default(), None)
        .expect("compile");
    let inputs = tri_nonrec_inputs();
    let mut seeded = BTreeMap::new();
    seeded.insert("e1", 5u64);
    seeded.insert("e2", 5u64);
    seeded.insert("e3", 5u64);
    let mut executor = build_executor(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(true)),
        &compiler,
        &inputs,
        &seeded,
    );
    let _ = executor.execute_plan(&plan).expect("execute_plan");
    assert_eq!(executor.wcoj_triangle_dispatch_count(), 1);

    let snap = executor.stats_snapshot();
    let rel_xy = *compiler.rel_ids().get("e1").expect("e1");
    let rel_yz = *compiler.rel_ids().get("e2").expect("e2");
    let canon = canonical_pair(rel_xy, rel_yz);
    // Keys swap with rels under canonical_join_key. For triangle
    // default leader (idx 0), record_wcoj_feedback called with
    // (rel_xy, rel_yz, [1], [0]) — if rel_xy < rel_yz in canonical
    // order, keys stay [1]/[0]; else swap to [0]/[1].
    let (expect_lk, expect_rk) = if rel_xy <= rel_yz {
        (vec![1usize], vec![0usize])
    } else {
        (vec![0usize], vec![1usize])
    };
    let matches: Vec<&JoinSelectivity> = snap
        .join_selectivities
        .iter()
        .filter(|js| {
            (js.left_rel, js.right_rel) == canon
                && js.left_keys == expect_lk
                && js.right_keys == expect_rk
        })
        .collect();
    assert_eq!(
        matches.len(),
        1,
        "var_order=None must record on canonical(rel_xy, rel_yz) keys [1]/[0]; \
         snapshot.join_selectivities = {:?}",
        snap.join_selectivities
    );
}

// ===============================================================
// Part E.1 — var_order=Some rotated-feedback cert
// ===============================================================

/// Build a hand-built triangle snapshot with HEAT-only bias that
/// pushes HeatAware leader to idx 2. `join_selectivities` is left
/// empty so the post-execution cert can prove the rotated entry
/// was created by W2.6's `feedback_pair_from_var_order` reroute,
/// not pre-existing.
fn make_triangle_heat_idx2_snapshot() -> StatsSnapshot {
    // RelIds 0,1,2 map to e1,e2,e3 via rel_names. The compiler
    // remaps to its own RelIds via the rel_names path.
    let mk_rel = |id: u32, card: u64, heat: f32| -> RelationStats {
        let mut r = RelationStats::new(RelId(id));
        r.cardinality = card;
        r.heat = heat;
        r
    };
    StatsSnapshot {
        relations: vec![
            mk_rel(0, 100, 0.5), // e1 / rel_xy: heat-demoted
            mk_rel(1, 100, 0.5), // e2 / rel_yz: heat-demoted
            mk_rel(2, 100, 0.0), // e3 / rel_xz: cold → leader
        ],
        join_selectivities: Vec::new(),
        rel_names: vec![
            (RelId(0), "e1".to_string()),
            (RelId(1), "e2".to_string()),
            (RelId(2), "e3".to_string()),
        ],
    }
}

#[test]
fn heat_aware_rotated_leader_records_feedback_on_rotated_pair() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    // Phase 1: compile under HeatAware + heat-biased snapshot.
    // Score: rel_xy = rel_yz = 100*3*2 = 600; rel_xz = 100*1*2 = 200.
    // argmin = idx 2 (rel_xz). Ratio 200/600 = 0.333 ≤ 0.5 → Some(2).
    let snap_in = make_triangle_heat_idx2_snapshot();
    let mut compiler = Compiler::new();
    let cfg_heat = CompilerConfig {
        wcoj_variable_ordering: WcojVarOrderingKind::HeatAware,
        ..CompilerConfig::default()
    };
    let plan_heat = compiler
        .compile_with_config_and_stats_snapshot(TRI_NONREC_SRC, &cfg_heat, Some(&snap_in))
        .expect("compile HeatAware");
    let vo_heat = first_var_order(&plan_heat).expect("HeatAware must set var_order");
    assert_eq!(
        vo_heat.leader_idx, 2,
        "snapshot with rel_xy.heat=rel_yz.heat=0.5 must drive leader_idx = 2"
    );

    // Phase 2: fresh executor; seeded card pre-condition;
    // join_selectivities must start empty.
    let inputs = tri_nonrec_inputs();
    let mut seeded = BTreeMap::new();
    seeded.insert("e1", 5u64);
    seeded.insert("e2", 5u64);
    seeded.insert("e3", 5u64);
    let mut executor = build_executor(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(true)),
        &compiler,
        &inputs,
        &seeded,
    );
    let pre = executor.stats_snapshot();
    assert!(
        pre.join_selectivities.is_empty(),
        "fresh executor's stats must have empty join_selectivities; got {:?}",
        pre.join_selectivities
    );

    // Phase 3: execute — exactly one WCOJ dispatch fires
    // record_wcoj_feedback, which through W2.6's rerouting
    // records on the rotated (slot_rels[0], slot_rels[1]) pair.
    let _ = executor.execute_plan(&plan_heat).expect("execute_plan");
    assert_eq!(executor.wcoj_triangle_dispatch_count(), 1);

    // Phase 4: post-execution snapshot has exactly one entry —
    // the rotated entry on canonical(rel_xz, rel_yz) keys [1]/[1].
    let snap_post = executor.stats_snapshot();
    assert_eq!(
        snap_post.join_selectivities.len(),
        1,
        "exactly one join_selectivities entry after one WCOJ dispatch; got {:?}",
        snap_post.join_selectivities
    );
    let entry = &snap_post.join_selectivities[0];
    let rel_xy = *compiler.rel_ids().get("e1").expect("e1");
    let rel_yz = *compiler.rel_ids().get("e2").expect("e2");
    let rel_xz = *compiler.rel_ids().get("e3").expect("e3");
    let canon_xz_yz = canonical_pair(rel_xz, rel_yz);
    assert_eq!(
        (entry.left_rel, entry.right_rel),
        canon_xz_yz,
        "rotated feedback entry must canonicalize to (rel_xz, rel_yz)"
    );
    // Both keys are [1] regardless of canonicalization swap
    // direction (symmetric).
    assert_eq!(
        entry.left_keys,
        vec![1usize],
        "rotated triangle leader=2 records left_keys = [1]; got {:?}",
        entry.left_keys
    );
    assert_eq!(
        entry.right_keys,
        vec![1usize],
        "rotated triangle leader=2 records right_keys = [1]; got {:?}",
        entry.right_keys
    );

    // Pre-W2.6 canonical pair must NOT have an entry.
    let canon_xy_yz = canonical_pair(rel_xy, rel_yz);
    let pre_w26 = snap_post
        .join_selectivities
        .iter()
        .find(|js| (js.left_rel, js.right_rel) == canon_xy_yz);
    assert!(
        pre_w26.is_none(),
        "canonical (rel_xy, rel_yz) pair must NOT exist when leader rotates to idx 2; \
         found {:?}",
        pre_w26
    );
}
