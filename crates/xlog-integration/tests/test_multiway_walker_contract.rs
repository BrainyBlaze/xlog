// crates/xlog-integration/tests/test_multiway_walker_contract.rs
//! v0.6.5 slice 2 (D2) — `MultiWayJoin` walker contract cross-crate
//! tests.
//!
//! Slice 1 introduced [`xlog_ir::RirNode::MultiWayJoin`] and added
//! explicit walker arms across the workspace. This file pins the
//! observable behavior of those arms under several runtime entry
//! points so slice 2a (4-way kernels) and slice 2b (cost model) can
//! refactor freely without silently regressing the v0.6.2 fallback
//! contract.
//!
//! Test catalogue:
//!
//! * **C1** — `Executor::execute_plan` with WCOJ force-on, triangle
//!   program: WCOJ counter == 1, row set non-empty (slice 1 happy
//!   path).
//! * **C2** — `Executor::execute_node` invoked directly on a
//!   synthesized `MultiWayJoin` body: produces the same row set as
//!   the embedded `fallback`. Locks the safety-net arm in
//!   `node_dispatch.rs`.
//! * **C3** — `Executor::execute_plan` with the WCOJ kill switch
//!   ON: counter == 0, row set still correct (fallback descent in
//!   `recursive::execute_stratum_impl`).
//! * **C4** — `Executor::execute_plan` with `wcoj_triangle_dispatch
//!   = Some(false)` (explicit force-off, mirrors the bench's
//!   `Mode::Off` and the adaptive fall-back outcome): counter == 0,
//!   row set correct.
//! * **C5** — `Executor::execute_non_recursive_scc` invoked directly
//!   with rules carrying a synthesized `MultiWayJoin` body: row set
//!   matches the fallback's row set, no panic. **Load-bearing**:
//!   this is the path `xlog-prob::mc::sampling` uses for monotone
//!   SCCs, which never invokes the WCOJ dispatch hook and relies
//!   entirely on the safety-net arm.
//! * **C6** — Source-contract check: the explicit `MultiWayJoin {
//!   fallback, .. } => walk_tmj(fallback, target_mask)` arm in
//!   `pyxlog/src/ilp.rs` is present. `pyxlog` is `cdylib` with
//!   `test = false`, so we cannot link it; the source string is the
//!   most reliable cross-crate guard.

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
use xlog_ir::rir::ProjectExpr;
use xlog_ir::{CompiledRule, JoinType, RirNode};
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
struct RuntimeFixture {
    device: Arc<CudaDevice>,
    runtime: Arc<XlogDeviceRuntime>,
    memory: Arc<GpuMemoryManager>,
    provider: Arc<CudaKernelProvider>,
    pool: Arc<StreamPool>,
}

fn make_runtime_fixture() -> Option<RuntimeFixture> {
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
    Some(RuntimeFixture {
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

const TRIANGLE_SOURCE: &str = "tri(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z).";

fn triangle_fixture() -> BTreeMap<&'static str, Vec<(u32, u32)>> {
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

fn run_program(
    provider: Arc<CudaKernelProvider>,
    memory: &Arc<GpuMemoryManager>,
    config: RuntimeConfig,
    inputs: &BTreeMap<&str, Vec<(u32, u32)>>,
) -> (Executor, u64) {
    let mut compiler = Compiler::new();
    let plan = compiler.compile(TRIANGLE_SOURCE).expect("compile");
    let mut executor = Executor::new_with_config(provider, config);
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    for (name, rows) in inputs {
        let buf = upload_binary_u32(memory, rows);
        executor.put_relation(name, buf);
    }
    executor.execute_plan(&plan).expect("execute_plan");
    let counter = executor.wcoj_triangle_dispatch_count();
    (executor, counter)
}

/// Compute the canonical-correct row set for the triangle program by
/// running it with the WCOJ gate explicitly OFF — the binary-join
/// chain. Used as the row-set reference that C1/C2/C5 must match
/// exactly. Without this, "fallback descent works" tests would pass
/// even if the walker silently dropped rows (e.g. returned just the
/// first triangle).
fn gate_off_reference_rows(
    provider: Arc<CudaKernelProvider>,
    memory: &Arc<GpuMemoryManager>,
    inputs: &BTreeMap<&str, Vec<(u32, u32)>>,
) -> Vec<(u32, u32, u32)> {
    let config_off = RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(false));
    let (exec_off, counter_off) = run_program(provider, memory, config_off, inputs);
    assert_eq!(
        counter_off, 0,
        "gate-off reference must not dispatch; got counter {counter_off}"
    );
    download_triples(exec_off.store().get("tri").expect("tri present"))
}

/// Build a synthesized `MultiWayJoin` body that wraps a real
/// binary-join `fallback` semantically equivalent to
/// `tri(X,Y,Z) :- e1(X,Y), e2(Y,Z), e3(X,Z)`. The promoter would
/// produce this for the same source under the canonical lowered
/// shape; we build it directly so C2 / C5 can exercise the walker
/// arms without relying on the promoter (which slice 2a may
/// generalize).
fn build_canonical_triangle_body(rel_xy: u32, rel_yz: u32, rel_xz: u32) -> RirNode {
    use xlog_core::RelId;
    let scan_xy = RirNode::Scan { rel: RelId(rel_xy) };
    let scan_yz = RirNode::Scan { rel: RelId(rel_yz) };
    let scan_xz = RirNode::Scan { rel: RelId(rel_xz) };
    let inner = RirNode::Join {
        left: Box::new(scan_xy.clone()),
        right: Box::new(scan_yz.clone()),
        left_keys: vec![1],
        right_keys: vec![0],
        join_type: JoinType::Inner,
    };
    let outer = RirNode::Join {
        left: Box::new(inner),
        right: Box::new(scan_xz.clone()),
        left_keys: vec![0, 3],
        right_keys: vec![0, 1],
        join_type: JoinType::Inner,
    };
    let fallback = RirNode::Project {
        input: Box::new(outer),
        columns: vec![
            ProjectExpr::Column(0),
            ProjectExpr::Column(1),
            ProjectExpr::Column(3),
        ],
    };
    RirNode::MultiWayJoin {
        inputs: vec![scan_xy, scan_yz, scan_xz],
        slot_vars: vec![
            vec![Some(0), Some(1)],
            vec![Some(1), Some(2)],
            vec![Some(0), Some(2)],
        ],
        output_columns: vec![
            ProjectExpr::Column(0),
            ProjectExpr::Column(1),
            ProjectExpr::Column(3),
        ],
        fallback: Box::new(fallback),
        var_order: None,
        plan: None,
    }
}

// ---------------------------------------------------------------
// C1 — slice 1 happy path (force-on)
// ---------------------------------------------------------------

#[test]
fn c1_force_on_dispatches_and_matches_gate_off_row_set() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping C1: CUDA runtime unavailable");
        return;
    };
    let inputs = triangle_fixture();

    // Reference: gate-off binary-join chain produces the canonical row set.
    let reference_rows = gate_off_reference_rows(Arc::clone(&fix.provider), &fix.memory, &inputs);
    assert!(
        !reference_rows.is_empty(),
        "binary-join reference must produce at least one triangle on the K_4 fixture"
    );

    // Under test: force WCOJ. Counter must increment AND the row
    // set must match the binary-join reference exactly.
    let config = RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(true));
    let (executor, counter) = run_program(Arc::clone(&fix.provider), &fix.memory, config, &inputs);
    assert_eq!(
        counter, 1,
        "force-on triangle must dispatch exactly once; got {counter}"
    );
    let dispatch_rows = download_triples(executor.store().get("tri").expect("tri present"));
    assert_eq!(
        dispatch_rows, reference_rows,
        "WCOJ dispatch row set must equal the gate-off binary-join reference"
    );
}

// ---------------------------------------------------------------
// C2 — direct execute_node on a MultiWayJoin → fallback row set
// ---------------------------------------------------------------

#[test]
fn c2_direct_execute_node_descends_to_fallback() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping C2: CUDA runtime unavailable");
        return;
    };
    let inputs = triangle_fixture();

    // Reference row set comes from the binary-join chain.
    let reference_rows = gate_off_reference_rows(Arc::clone(&fix.provider), &fix.memory, &inputs);

    // Set up an executor with WCOJ kill switch ON so that even if
    // a future refactor adds dispatch logic to execute_node, the
    // fallback path is what we exercise.
    let config = RuntimeConfig::default().with_wcoj_triangle_dispatch_disabled(Some(true));
    let mut executor = Executor::new_with_config(Arc::clone(&fix.provider), config);

    // Compile the same triangle program through the regular path
    // to obtain rel_ids and store the inputs in the executor.
    let mut compiler = Compiler::new();
    let _plan = compiler.compile(TRIANGLE_SOURCE).expect("compile");
    let rel_ids = compiler.rel_ids().clone();
    for (name, rel_id) in &rel_ids {
        executor.register_relation(*rel_id, name);
    }
    for (name, rows) in &inputs {
        let buf = upload_binary_u32(&fix.memory, rows);
        executor.put_relation(name, buf);
    }

    // Build the synthesized MultiWayJoin body directly from the
    // rel_ids the compiler assigned.
    let rel_xy = rel_ids.get("e1").expect("e1 rel_id").0;
    let rel_yz = rel_ids.get("e2").expect("e2 rel_id").0;
    let rel_xz = rel_ids.get("e3").expect("e3 rel_id").0;
    let body = build_canonical_triangle_body(rel_xy, rel_yz, rel_xz);

    // Direct execute_node — bypasses execute_plan / dispatch hook.
    let buf = executor
        .execute_node(&body)
        .expect("execute_node must succeed via fallback descent");
    let rows = download_triples(&buf);
    assert_eq!(
        rows, reference_rows,
        "execute_node(MultiWayJoin) must reproduce the binary-join reference \
         row-for-row via the safety-net fallback descent. A walker that \
         silently drops rows (e.g. returns only the first match) would slip \
         through a non-empty assertion but fail this exact comparison."
    );
    assert_eq!(
        executor.wcoj_triangle_dispatch_count(),
        0,
        "execute_node must NOT engage WCOJ dispatch — that is execute_plan's job"
    );
}

// ---------------------------------------------------------------
// C3 — kill switch ON: row set correct, counter == 0
// ---------------------------------------------------------------

#[test]
fn c3_kill_switch_falls_back_with_correct_row_set() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping C3: CUDA runtime unavailable");
        return;
    };
    let inputs = triangle_fixture();

    // Reference: gate explicitly off (binary-join chain).
    let config_off = RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(false));
    let (exec_off, counter_off) =
        run_program(Arc::clone(&fix.provider), &fix.memory, config_off, &inputs);
    assert_eq!(counter_off, 0);
    let reference_rows = download_triples(exec_off.store().get("tri").expect("tri"));

    // Kill switch beats every other flag.
    let config_kill = RuntimeConfig::default()
        .with_wcoj_triangle_dispatch(Some(true)) // would normally force WCOJ
        .with_wcoj_triangle_dispatch_disabled(Some(true)); // …but this wins
    let (exec_kill, counter_kill) =
        run_program(Arc::clone(&fix.provider), &fix.memory, config_kill, &inputs);
    assert_eq!(
        counter_kill, 0,
        "kill switch must override force-on; got counter {counter_kill}"
    );
    let kill_rows = download_triples(exec_kill.store().get("tri").expect("tri"));
    assert_eq!(
        kill_rows, reference_rows,
        "kill-switched fallback row set must equal binary-join reference"
    );
}

// ---------------------------------------------------------------
// C4 — adaptive opt-out fall-back
// ---------------------------------------------------------------

#[test]
fn c4_adaptive_optout_falls_back_with_correct_row_set() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping C4: CUDA runtime unavailable");
        return;
    };
    let inputs = triangle_fixture();

    // Reference: explicit force-off.
    let config_off = RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(false));
    let (exec_off, counter_off) =
        run_program(Arc::clone(&fix.provider), &fix.memory, config_off, &inputs);
    assert_eq!(counter_off, 0);
    let reference_rows = download_triples(exec_off.store().get("tri").expect("tri"));

    // Adaptive opt-out: classifier never runs; force is also None
    // (default). No dispatch path engages → fallback to binary.
    let config_adopt_off =
        RuntimeConfig::default().with_wcoj_triangle_dispatch_adaptive(Some(false));
    let (exec_adopt_off, counter_adopt_off) = run_program(
        Arc::clone(&fix.provider),
        &fix.memory,
        config_adopt_off,
        &inputs,
    );
    assert_eq!(
        counter_adopt_off, 0,
        "adaptive opt-out with no force gate must not dispatch; got {counter_adopt_off}"
    );
    let adopt_off_rows = download_triples(exec_adopt_off.store().get("tri").expect("tri"));
    assert_eq!(
        adopt_off_rows, reference_rows,
        "adaptive-opt-out fallback row set must equal binary-join reference"
    );
}

// ---------------------------------------------------------------
// C5 — execute_non_recursive_scc on synthesized MultiWayJoin (R1)
// ---------------------------------------------------------------

#[test]
fn c5_execute_non_recursive_scc_descends_via_safety_net() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping C5: CUDA runtime unavailable");
        return;
    };
    let inputs = triangle_fixture();

    // Reference row set comes from the binary-join chain.
    let reference_rows = gate_off_reference_rows(Arc::clone(&fix.provider), &fix.memory, &inputs);

    // execute_non_recursive_scc bypasses the WCOJ dispatch hook
    // entirely and relies on execute_node's MultiWayJoin safety-net
    // arm. xlog-prob::mc::sampling uses this path; a regression
    // there would silently break MC sampling on triangle programs.
    let config = RuntimeConfig::default();
    let mut executor = Executor::new_with_config(Arc::clone(&fix.provider), config);

    // Compile to obtain rel_ids; we discard the plan and
    // construct rules with synthesized MultiWayJoin bodies.
    let mut compiler = Compiler::new();
    let _plan = compiler.compile(TRIANGLE_SOURCE).expect("compile");
    let rel_ids = compiler.rel_ids().clone();
    for (name, rel_id) in &rel_ids {
        executor.register_relation(*rel_id, name);
    }
    for (name, rows) in &inputs {
        let buf = upload_binary_u32(&fix.memory, rows);
        executor.put_relation(name, buf);
    }

    let rel_xy = rel_ids.get("e1").expect("e1 rel_id").0;
    let rel_yz = rel_ids.get("e2").expect("e2 rel_id").0;
    let rel_xz = rel_ids.get("e3").expect("e3 rel_id").0;
    let synthesized_body = build_canonical_triangle_body(rel_xy, rel_yz, rel_xz);
    let rule = CompiledRule {
        head: "tri".to_string(),
        body: synthesized_body,
        meta: Default::default(),
    };

    // Public Executor entry point used by xlog-prob::mc::sampling.
    executor
        .execute_non_recursive_scc(std::slice::from_ref(&rule))
        .expect("execute_non_recursive_scc must succeed via safety-net");

    // No WCOJ dispatch fired (the path skips the hook).
    assert_eq!(
        executor.wcoj_triangle_dispatch_count(),
        0,
        "execute_non_recursive_scc must NOT engage WCOJ dispatch"
    );
    // Result installed under the rule's head, row-set identical to
    // the binary-join reference. This is the load-bearing assertion:
    // a walker that silently drops or duplicates rows would slip
    // through a non-empty check but fail row-for-row equality.
    let buf = executor
        .store()
        .get("tri")
        .expect("execute_non_recursive_scc must install head buffer");
    let rows = download_triples(buf);
    assert_eq!(
        rows, reference_rows,
        "execute_non_recursive_scc on a MultiWayJoin body must reproduce the \
         binary-join reference row-for-row via the safety-net fallback descent. \
         This is the path xlog-prob::mc::sampling uses for monotone SCCs; \
         silent row-set divergence here would corrupt MC inference."
    );
}

// ---------------------------------------------------------------
// C6 — pyxlog walk_tmj source-contract (no link, source check)
// ---------------------------------------------------------------

#[test]
fn c6_pyxlog_walk_tmj_has_explicit_multiway_arm() {
    // pyxlog is cdylib + test = false; we cannot link against
    // walk_tmj or any other pyxlog symbol. The source is the most
    // reliable cross-crate guard. If a refactor removes the explicit
    // arm and lets MultiWayJoin fall through to the catch-all
    // `_ => None`, a TMJ wrapped inside a promoted MultiWayJoin's
    // fallback would silently disappear from any pyxlog ILP query.
    let src = include_str!("../../pyxlog/src/ilp.rs");
    let needle = "RirNode::MultiWayJoin { fallback, .. } => walk_tmj(fallback, target_mask)";
    assert!(
        src.contains(needle),
        "pyxlog::ilp::walk_tmj must contain the explicit MultiWayJoin -> walk_tmj(fallback, …) \
         arm. Source-string contract for v0.6.5 slice 2 walker hardening of pyxlog::ilp; \
         if the arm is removed or reformatted, walk_tmj will silently miss any TMJ wrapped \
         inside a promoted MultiWayJoin's fallback (the catch-all `_ => None` would swallow it)."
    );
}
