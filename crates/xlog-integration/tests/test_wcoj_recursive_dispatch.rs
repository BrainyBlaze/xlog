// crates/xlog-integration/tests/test_wcoj_recursive_dispatch.rs
//! v0.6.5 slice 4 — recursive-SCC WCOJ dispatch certification.
//!
//! Locks the contract for `Executor::execute_wcoj_or_fallback_node`
//! at both the seeding pass and the per-variant evaluation in
//! `execute_recursive_scc`. After slice 4:
//!
//!   * A **stable triangle** (zero recursive Scans in body) inside
//!     a recursive SCC is promoted by the slice 4 promoter and
//!     dispatched on the seeding pass — counter == 1.
//!   * A **stable 4-cycle** in a recursive SCC behaves the same
//!     for the 4-cycle counter.
//!   * A **multi-recursive** triangle (≥ 2 in-SCC body Scans) is
//!     NOT promoted and runs the binary-join semi-naive path —
//!     counter == 0; final row set still matches the binary
//!     reference.
//!   * A **linear-recursive triangle / 4-cycle** (exactly one
//!     in-SCC body Scan) IS promoted and dispatches both on the
//!     seeding pass AND on per-variant evaluation. Counter
//!     strictly exceeds the seeding-only case (≥ 2 dispatches).
//!     Final row set matches the binary-join reference, so the
//!     fixed-point convergence + delta correctness are both
//!     verified.
//!
//! Counter semantics: `wcoj_*_dispatch_count` increments per
//! successful WCOJ kernel result — once per (seeding pass,
//! iteration, variant) where the dispatcher returns a buffer.

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
// Fixture helpers (mirror test_wcoj_executor_wiring.rs)
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

/// Upload a 2-column U32 EDB buffer with column names matching
/// the compiler's convention (`c0`, `c1`). Recursive programs
/// require this schema-name compat — the executor unions the
/// runtime-uploaded EDB with compiler-emitted IDB buffers each
/// iteration, and union compares schemas strictly.
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

/// Compile + execute a recursive program. EDB facts are present
/// in `source` (so the compiler infers U32 schemas across the
/// whole program — necessary for the recursive engine's per-
/// iteration union to type-check) AND uploaded as runtime
/// buffers in `inputs` (so the executor's store has actual data
/// when the rules run; the compiler does not auto-load source
/// facts into the GPU store). Schema column names use the
/// compiler convention (`c0/c1`) so the EDB upload unions
/// cleanly with compiler-emitted IDB buffers.
fn run_program(
    provider: Arc<CudaKernelProvider>,
    memory: &Arc<GpuMemoryManager>,
    config: RuntimeConfig,
    source: &str,
    inputs: &BTreeMap<&str, Vec<(u32, u32)>>,
) -> Executor {
    let mut compiler = Compiler::new();
    let plan = compiler.compile(source).expect("compile");
    let mut executor = Executor::new_with_config(provider, config);
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    for (name, rows) in inputs {
        let buf = upload_binary_u32(memory, rows);
        executor.put_relation(name, buf);
    }
    let _ = executor.execute_plan(&plan).expect("execute_plan");
    executor
}

// ---------------------------------------------------------------
// Stable triangle in recursive SCC
// ---------------------------------------------------------------

/// A recursive program where the triangle rule's body uses only
/// extensional relations (e1/e2/e3) — count of in-SCC Scans is 0.
/// Slice 4 promotes it; the seeding pass dispatches WCOJ exactly
/// once. The `echo`+feedback rules force `tri` into a recursive
/// SCC ({tri, echo}) without adding any in-SCC body atoms.
///
/// Explicit `pred` declarations anchor U32 schemas across all
/// predicates so the recursive engine's per-iteration union
/// type-checks against the runtime-uploaded EDB buffers.
/// Inline facts would also work for typing, but they perturb
/// the optimizer's cardinality estimates and can flip the
/// canonical triangle shape from left-deep to right-deep —
/// which the slice 1 promoter doesn't recognize.
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

#[test]
fn stable_triangle_in_recursive_scc_dispatches_wcoj_on_seeding() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = triangle_inputs();

    // Reference: gate off → no WCOJ; binary-join produces the row set.
    let reference = run_program(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(false)),
        STABLE_TRIANGLE_RECURSIVE,
        &inputs,
    );
    assert_eq!(
        reference.wcoj_triangle_dispatch_count(),
        0,
        "gate=off must not dispatch in the recursive arm"
    );
    let reference_rows = download_triples(reference.store().get("tri").expect("tri"));
    assert!(
        !reference_rows.is_empty(),
        "binary-join reference produced empty triangle — fixture is degenerate"
    );

    // Gate on: slice 4 promotes the stable rule → dispatch on
    // seeding pass. Counter == 1 (rule 1 only; the echo + copy
    // rules don't match a WCOJ shape).
    let dispatched = run_program(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(true)),
        STABLE_TRIANGLE_RECURSIVE,
        &inputs,
    );
    assert_eq!(
        dispatched.wcoj_triangle_dispatch_count(),
        1,
        "stable triangle rule in recursive SCC must dispatch WCOJ \
         exactly once on the seeding pass; got counter {}",
        dispatched.wcoj_triangle_dispatch_count()
    );
    let dispatched_rows = download_triples(dispatched.store().get("tri").expect("tri"));
    assert_eq!(
        dispatched_rows, reference_rows,
        "WCOJ dispatch in recursive arm must produce the same row set as binary-join"
    );
}

// ---------------------------------------------------------------
// Stable 4-cycle in recursive SCC
// ---------------------------------------------------------------

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

#[test]
fn stable_4cycle_in_recursive_scc_dispatches_wcoj_on_seeding() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = cycle4_inputs();

    let reference = run_program(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default().with_wcoj_4cycle_dispatch(Some(false)),
        STABLE_4CYCLE_RECURSIVE,
        &inputs,
    );
    assert_eq!(
        reference.wcoj_4cycle_dispatch_count(),
        0,
        "gate=off must not dispatch in the recursive arm"
    );
    let reference_rows = download_quads(reference.store().get("cyc").expect("cyc"));
    assert!(
        !reference_rows.is_empty(),
        "binary-join reference produced empty 4-cycle — fixture is degenerate"
    );

    let dispatched = run_program(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default().with_wcoj_4cycle_dispatch(Some(true)),
        STABLE_4CYCLE_RECURSIVE,
        &inputs,
    );
    assert_eq!(
        dispatched.wcoj_4cycle_dispatch_count(),
        1,
        "stable 4-cycle rule in recursive SCC must dispatch WCOJ \
         exactly once on the seeding pass; got counter {}",
        dispatched.wcoj_4cycle_dispatch_count()
    );
    let dispatched_rows = download_quads(dispatched.store().get("cyc").expect("cyc"));
    assert_eq!(
        dispatched_rows, reference_rows,
        "4-cycle WCOJ dispatch in recursive arm must produce the same row set as binary-join"
    );
}

// ---------------------------------------------------------------
// Multi-recursive triangle: WCOJ dispatched (seeding + variants),
// binary-join parity (W4.1 paper P1).
// ---------------------------------------------------------------

/// W4.1 multi-recursive triangle. Two of the three body Scans
/// (`r1`, `r2`) are recursive — they receive feedback from `tri`.
/// The third (`r3`) is extensional. Per paper P1 (semi-naive
/// occurrence semantics), the W4.1 promoter admits this body
/// because the recursive Scans target DISTINCT predicates — the
/// variant-construction loop in `recursive.rs:455-540` builds one
/// variant per recursive occurrence with a non-empty delta and
/// dispatches WCOJ on each.
///
/// Counter dynamics:
///   * Seeding pass (`recursive.rs:331-347`): triangle rule fires
///     once on its full body — counter += 1.
///   * Iteration 1 (`recursive.rs:455-540`): both `r1_init` and
///     `r2_init` are non-empty, so two variants fire (one with
///     `r1`'s scan rewritten to `r1_delta`, one with `r2`'s scan
///     rewritten to `r2_delta`) — counter += 2.
///
/// Total: counter `>= 2` (in fact `== 3` for this fixture).
const MULTIREC_TRIANGLE: &str = r#"
    pred r1_init(u32, u32).
    pred r2_init(u32, u32).
    pred r3(u32, u32).
    pred r1(u32, u32).
    pred r2(u32, u32).
    pred tri(u32, u32, u32).
    r1(X, Y) :- r1_init(X, Y).
    r1(X, Y) :- tri(X, Y, Z).
    r2(X, Y) :- r2_init(X, Y).
    r2(X, Y) :- tri(Z, X, Y).
    tri(X, Y, Z) :- r1(X, Y), r2(Y, Z), r3(X, Z).
"#;

// ---------------------------------------------------------------
// Adaptive parity: classifier makes same decision in recursive arm
// ---------------------------------------------------------------

/// Hub-heavy fixture: vertex 1 is incident to many edges (skew
/// well above the 0.10 threshold). The same classifier that
/// dispatches in the non-recursive arm must dispatch here too.
fn superhub_inputs() -> BTreeMap<&'static str, Vec<(u32, u32)>> {
    let mut m: BTreeMap<&'static str, Vec<(u32, u32)>> = BTreeMap::new();
    let mut edges = Vec::new();
    for v in 2..=300 {
        edges.push((1u32, v));
        edges.push((v, 1));
    }
    edges.push((2, 3));
    edges.push((3, 4));
    edges.push((4, 2));
    edges.sort();
    edges.dedup();
    m.insert("e1", edges.clone());
    m.insert("e2", edges.clone());
    m.insert("e3", edges);
    m
}

#[test]
fn adaptive_dispatches_in_recursive_scc_on_superhub() {
    // Adaptive default-on (no explicit `with_wcoj_triangle_dispatch`)
    // → classifier runs on the seeding pass; super-hub fixture
    // produces score ≥ 0.10 → dispatch fires. Counter == 1
    // (rule 0 only; the echo + copy rules don't match a triangle
    // shape).
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let executor = run_program(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default(),
        STABLE_TRIANGLE_RECURSIVE,
        &superhub_inputs(),
    );
    assert!(
        executor.wcoj_triangle_dispatch_count() >= 1,
        "adaptive classifier on super-hub fixture must dispatch in \
         the recursive arm; got counter {}",
        executor.wcoj_triangle_dispatch_count()
    );
}

fn multirec_inputs() -> BTreeMap<&'static str, Vec<(u32, u32)>> {
    let mut m: BTreeMap<&'static str, Vec<(u32, u32)>> = BTreeMap::new();
    m.insert("r1_init", vec![(1, 2), (1, 3), (2, 3)]);
    m.insert("r2_init", vec![(2, 3), (3, 4)]);
    m.insert("r3", vec![(1, 3), (2, 4), (1, 4)]);
    m
}

#[test]
fn multirec_triangle_dispatches_wcoj_and_matches_binary_join() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = multirec_inputs();

    // Reference: gate off → binary-join answer.
    let reference = run_program(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(false)),
        MULTIREC_TRIANGLE,
        &inputs,
    );
    assert_eq!(
        reference.wcoj_triangle_dispatch_count(),
        0,
        "gate=off must not dispatch"
    );
    let reference_rows = download_triples(reference.store().get("tri").expect("tri"));
    assert!(
        !reference_rows.is_empty(),
        "fixture must produce at least one tri row so row-set parity is \
         not trivially satisfied by an empty-output bug; got {} rows",
        reference_rows.len()
    );

    // Gate on: W4.1 promoter admits the multi-recursive triangle.
    // Seeding fires WCOJ once on the full body. Iter 1 fires one
    // variant per recursive predicate with a non-empty delta — for
    // this fixture, both `r1_init` and `r2_init` are non-empty, so
    // two variants fire. Total counter `>= 2`. Final row set must
    // equal the binary-join reference.
    let dispatched = run_program(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(true)),
        MULTIREC_TRIANGLE,
        &inputs,
    );
    assert!(
        dispatched.wcoj_triangle_dispatch_count() >= 2,
        "multi-recursive triangle (distinct recursive predicates \
         r1, r2) must dispatch WCOJ on the seeding pass AND ≥ 1 \
         variant in the iteration loop; got counter {}",
        dispatched.wcoj_triangle_dispatch_count()
    );
    let dispatched_rows = download_triples(dispatched.store().get("tri").expect("tri"));
    assert_eq!(
        dispatched_rows, reference_rows,
        "multi-recursive WCOJ row set must equal the binary-join reference"
    );
}

// ---------------------------------------------------------------
// Linear-recursive triangle: WCOJ dispatches on seeding AND per
// variant, counter strictly > 1, row set matches binary-join.
// ---------------------------------------------------------------

/// Linear-recursive triangle. Body has exactly ONE in-SCC Scan
/// (`e1`, fed back from `tri` via the second `e1` rule). The
/// other two body atoms (`e2`, `e3`) are extensional. Slice 4
/// promoter gate: count == 1 → promote.
///
/// Recursive dynamics:
///   1. Seeding pass: triangle rule joins `e1_seed` ⋈ e2 ⋈ e3 →
///      one or more `tri` rows. WCOJ counter increments by 1.
///   2. Iteration 1: `e1`'s recursive rule projects new `e1`
///      rows from `tri` (using `tri(X, Z, Y)` so the projection
///      shifts indices and produces e1 rows that weren't in the
///      seed). `e1_delta` is non-empty.
///      Triangle variant: e1's Scan rewritten to `e1_delta`.
///      Dispatch fires. Counter increments — confirming the
///      per-variant path actually exercises WCOJ on a body with
///      one rewritten slot.
///   3. Iteration 2+: chain may continue or terminate; the
///      counter is asserted as ≥ 2 (seeding + ≥ 1 variant), not
///      a tight number, since iteration count depends on
///      convergence of the chain reaction.
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
    // Chain: seed (1,2) → tri(1,2,3) (via e2(2,3), e3(1,3)) →
    //   recursive e1(1,3) → tri(1,3,4) (via e2(3,4), e3(1,4)).
    m.insert("e1_seed", vec![(1, 2)]);
    m.insert("e2", vec![(2, 3), (3, 4)]);
    m.insert("e3", vec![(1, 3), (1, 4)]);
    m
}

#[test]
fn linear_recursive_triangle_dispatches_on_seeding_and_per_variant() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = linear_rec_triangle_inputs();

    // Reference: gate off. Binary-join semi-naive computes the
    // fixpoint without any WCOJ dispatch.
    let reference = run_program(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(false)),
        LINEAR_REC_TRIANGLE,
        &inputs,
    );
    assert_eq!(
        reference.wcoj_triangle_dispatch_count(),
        0,
        "gate=off must not dispatch"
    );
    let reference_rows = download_triples(reference.store().get("tri").expect("tri"));
    assert!(
        reference_rows.len() >= 2,
        "fixture should produce ≥ 2 tri rows so we know the recursive \
         chain actually fired (seeding + ≥ 1 iteration); got {} rows",
        reference_rows.len()
    );

    // Gate on: slice 4 promotes the linear-recursive triangle.
    // Seeding fires WCOJ once. Each iteration with a non-empty
    // `e1_delta` fires WCOJ on the rewritten variant body. The
    // counter must strictly exceed the seeding-only case.
    let dispatched = run_program(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(true)),
        LINEAR_REC_TRIANGLE,
        &inputs,
    );
    assert!(
        dispatched.wcoj_triangle_dispatch_count() >= 2,
        "linear-recursive triangle must dispatch WCOJ on BOTH the \
         seeding pass AND ≥ 1 variant iteration; got counter {}",
        dispatched.wcoj_triangle_dispatch_count()
    );
    let dispatched_rows = download_triples(dispatched.store().get("tri").expect("tri"));
    assert_eq!(
        dispatched_rows, reference_rows,
        "linear-recursive WCOJ row set must equal the binary-join reference"
    );
}

// ---------------------------------------------------------------
// Linear-recursive 4-cycle: same contract as triangle.
// ---------------------------------------------------------------

/// Linear-recursive 4-cycle. `e1` is recursive (fed back from
/// `cyc(Y, W, X, Z)`); `e2/e3/e4` are extensional. Slice 4
/// promoter sees count == 1 → promote. Same seeding +
/// per-variant dispatch contract as the triangle case above.
const LINEAR_REC_4CYCLE: &str = r#"
    pred e1_seed(u32, u32).
    pred e1(u32, u32).
    pred e2(u32, u32).
    pred e3(u32, u32).
    pred e4(u32, u32).
    pred cyc(u32, u32, u32, u32).
    e1(W, X) :- e1_seed(W, X).
    e1(W, X) :- cyc(Y, W, X, Z).
    cyc(W, X, Y, Z) :- e1(W, X), e2(X, Y), e3(Y, Z), e4(Z, W).
"#;

fn linear_rec_cycle4_inputs() -> BTreeMap<&'static str, Vec<(u32, u32)>> {
    let mut m: BTreeMap<&'static str, Vec<(u32, u32)>> = BTreeMap::new();
    // Seeding: cyc(1,2,3,4) from e1_seed(1,2) ⋈ e2(2,3) ⋈ e3(3,4) ⋈ e4(4,1).
    // Iter 1: e1 recursive rule projects cyc(1,2,3,4)'s (col1, col2) =
    //   (2, 3) into a new e1(2, 3). Triangle variant fires WCOJ on
    //   e1_delta=(2,3) ⋈ e2(3,4) ⋈ e3(4,5) ⋈ e4(5,2), producing
    //   cyc(2,3,4,5). Iter 2 may cascade further; counter ≥ 2.
    m.insert("e1_seed", vec![(1, 2)]);
    m.insert("e2", vec![(2, 3), (3, 4)]);
    m.insert("e3", vec![(3, 4), (4, 5)]);
    m.insert("e4", vec![(4, 1), (5, 2)]);
    m
}

#[test]
fn linear_recursive_4cycle_dispatches_on_seeding_and_per_variant() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = linear_rec_cycle4_inputs();

    let reference = run_program(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default().with_wcoj_4cycle_dispatch(Some(false)),
        LINEAR_REC_4CYCLE,
        &inputs,
    );
    assert_eq!(
        reference.wcoj_4cycle_dispatch_count(),
        0,
        "gate=off must not dispatch"
    );
    let reference_rows = download_quads(reference.store().get("cyc").expect("cyc"));
    assert!(
        reference_rows.len() >= 2,
        "fixture should produce ≥ 2 cyc rows so the recursive chain \
         demonstrably fires (seeding + ≥ 1 iteration); got {} rows",
        reference_rows.len()
    );

    let dispatched = run_program(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default().with_wcoj_4cycle_dispatch(Some(true)),
        LINEAR_REC_4CYCLE,
        &inputs,
    );
    assert!(
        dispatched.wcoj_4cycle_dispatch_count() >= 2,
        "linear-recursive 4-cycle must dispatch WCOJ on BOTH the \
         seeding pass AND ≥ 1 variant iteration; got counter {}",
        dispatched.wcoj_4cycle_dispatch_count()
    );
    let dispatched_rows = download_quads(dispatched.store().get("cyc").expect("cyc"));
    assert_eq!(
        dispatched_rows, reference_rows,
        "linear-recursive 4-cycle WCOJ row set must equal the binary-join reference"
    );
}

// ---------------------------------------------------------------
// Multi-recursive 4-cycle: WCOJ dispatched (seeding + variants),
// binary-join parity (W4.1 paper P1).
// ---------------------------------------------------------------

/// W4.1 multi-recursive 4-cycle. Two of the four body Scans
/// (`r1`, `r2`) are recursive — they receive feedback from `cyc`
/// via SHIFTED projections so iter 1 produces non-empty deltas
/// for BOTH:
///   * `r1(W, X) :- cyc(Y, W, X, Z)` — extracts cyc cols 1,2.
///   * `r2(A, B) :- cyc(W, X, A, B)` — extracts cyc cols 2,3.
/// The other two atoms (`r3`, `r4`) are extensional. Per paper
/// P1 (semi-naive occurrence semantics), the W4.1 promoter
/// admits this body because the recursive Scans target DISTINCT
/// predicates — the variant-construction loop in
/// `recursive.rs:455-540` builds one variant per recursive
/// occurrence with a non-empty delta and dispatches WCOJ on each.
///
/// Counter dynamics:
///   * Seeding pass (`recursive.rs:331-347`): cyc rule fires
///     once on its full body — counter += 1.
///   * Iteration 1 (`recursive.rs:455-540`): both `r1_init` and
///     `r2_init` are non-empty AND the shifted projections of
///     the seeded `cyc` rows lie outside the initial r1/r2, so
///     two variants fire — counter += 2.
///
/// Total: counter `>= 2` (in fact `== 3` for this fixture).
const MULTIREC_4CYCLE: &str = r#"
    pred r1_init(u32, u32).
    pred r2_init(u32, u32).
    pred r3(u32, u32).
    pred r4(u32, u32).
    pred r1(u32, u32).
    pred r2(u32, u32).
    pred cyc(u32, u32, u32, u32).
    r1(W, X) :- r1_init(W, X).
    r1(W, X) :- cyc(Y, W, X, Z).
    r2(X, Y) :- r2_init(X, Y).
    r2(A, B) :- cyc(W, X, A, B).
    cyc(W, X, Y, Z) :- r1(W, X), r2(X, Y), r3(Y, Z), r4(Z, W).
"#;

fn multirec_4cycle_inputs() -> BTreeMap<&'static str, Vec<(u32, u32)>> {
    let mut m: BTreeMap<&'static str, Vec<(u32, u32)>> = BTreeMap::new();
    // Seeding produces two disjoint 4-cycles:
    //   * r1_init(1,2) ⋈ r2_init(2,3) ⋈ r3(3,4) ⋈ r4(4,1) → cyc(1,2,3,4)
    //   * r1_init(5,6) ⋈ r2_init(6,7) ⋈ r3(7,8) ⋈ r4(8,5) → cyc(5,6,7,8)
    // Iter 1 produces non-empty deltas for BOTH recursive
    // predicates via the shifted projections (see MULTIREC_4CYCLE
    // doc), so two variants fire — counter >= 2.
    m.insert("r1_init", vec![(1, 2), (5, 6)]);
    m.insert("r2_init", vec![(2, 3), (6, 7)]);
    m.insert("r3", vec![(3, 4), (7, 8)]);
    m.insert("r4", vec![(4, 1), (8, 5)]);
    m
}

#[test]
fn multirec_4cycle_dispatches_wcoj_and_matches_binary_join() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = multirec_4cycle_inputs();

    // Reference: gate off → binary-join answer.
    let reference = run_program(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default().with_wcoj_4cycle_dispatch(Some(false)),
        MULTIREC_4CYCLE,
        &inputs,
    );
    assert_eq!(
        reference.wcoj_4cycle_dispatch_count(),
        0,
        "gate=off must not dispatch"
    );
    let reference_rows = download_quads(reference.store().get("cyc").expect("cyc"));
    assert!(
        !reference_rows.is_empty(),
        "fixture must produce at least one cyc row so row-set parity \
         is not trivially satisfied by an empty-output bug; got {} rows",
        reference_rows.len()
    );

    // Gate on: W4.1 promoter admits the multi-recursive 4-cycle.
    // Seeding fires WCOJ once on the full body. Iter 1 fires one
    // variant per recursive predicate with a non-empty delta —
    // for this fixture, both r1's and r2's shifted projections
    // produce non-empty deltas, so two variants fire. Total
    // counter >= 2. Final row set must equal the binary-join
    // reference.
    let dispatched = run_program(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default().with_wcoj_4cycle_dispatch(Some(true)),
        MULTIREC_4CYCLE,
        &inputs,
    );
    assert!(
        dispatched.wcoj_4cycle_dispatch_count() >= 2,
        "multi-recursive 4-cycle (distinct recursive predicates \
         r1, r2) must dispatch WCOJ on the seeding pass AND ≥ 1 \
         variant in the iteration loop; got counter {}",
        dispatched.wcoj_4cycle_dispatch_count()
    );
    let dispatched_rows = download_quads(dispatched.store().get("cyc").expect("cyc"));
    assert_eq!(
        dispatched_rows, reference_rows,
        "multi-recursive 4-cycle WCOJ row set must equal the binary-join reference"
    );
}

// ---------------------------------------------------------------
// Self-recursive triangle (same-predicate, paper P1 lock):
// WCOJ dispatched (seeding + variants), binary-join parity.
// ---------------------------------------------------------------

/// W4.1 self-recursive triangle. Rule 3's body has TWO `p` Scans
/// (same predicate, two distinct occurrences). Per paper P1
/// (semi-naïve evaluation reasons over body-clause OCCURRENCES,
/// not predicate names), this body must be admitted by the
/// promoter — every occurrence is a valid Δ-binding site, and
/// each occurrence yields its own per-iteration variant.
///
/// This is the strict test of Step 6's `rewrite_scan_nth` fix:
/// with two same-predicate occurrences, the rewrite walker must
/// stop after replacing the k-th occurrence (not continue and
/// replace the (k+1)-th too) and the inputs/fallback views must
/// be walked symmetrically. Distinct-predicate certs (multirec
/// triangle / 4-cycle) cannot detect a pre-Step-6 walker
/// regression because the second predicate has a different RelId
/// and does not match.
///
/// Counter dynamics:
///   * Seeding pass: rule 3's body is promoted to a MultiWayJoin
///     and `execute_wcoj_or_fallback_node` dispatches WCOJ once.
///     Store-`p` is empty at seeding entry (rule 1 has not yet
///     applied to the store), so the kernel launches with 0
///     rows but counter still increments — counter += 1.
///   * Iteration 1: delta_p (after seeding) = `{(1,2),(2,3)}`,
///     non-empty. Rule 3 has 2 recursive `p` occurrences, so the
///     variant-construction loop builds 2 variants and dispatches
///     WCOJ on each — counter += 2.
///
/// Total: counter `>= 2` (in fact `>= 3` for this fixture).
const SELFREC_TRIANGLE: &str = r#"
    pred p_init(u32, u32).
    pred q(u32, u32).
    pred p(u32, u32).
    pred tri(u32, u32, u32).
    p(X, Y) :- p_init(X, Y).
    p(X, Z) :- tri(X, Y, Z).
    tri(X, Y, Z) :- p(X, Y), p(Y, Z), q(X, Z).
"#;

fn selfrec_triangle_inputs() -> BTreeMap<&'static str, Vec<(u32, u32)>> {
    // F-W41-13 pinned fixture:
    //   p_init = {(1,2),(2,3)} → seeding rule 1 populates p.
    //   q      = {(1,3)}.
    //   Reference tri = {(1,2,3)} non-empty. Iteration 1's two
    //   variants of `p` (occ=0 and occ=1) both produce tri(1,2,3)
    //   (already present); iteration 2 may grow `p` via rule 2
    //   (p(1,3) from tri's projection) but no further tri rows.
    let mut m: BTreeMap<&'static str, Vec<(u32, u32)>> = BTreeMap::new();
    m.insert("p_init", vec![(1, 2), (2, 3)]);
    m.insert("q", vec![(1, 3)]);
    m
}

#[test]
fn selfrec_triangle_dispatches_wcoj_and_matches_binary_join() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = selfrec_triangle_inputs();

    // Reference: gate off → binary-join answer. Body is still
    // promoted to MultiWayJoin (compile-time) but execution uses
    // the fallback path; rewrite_scan_nth still operates on the
    // MultiWayJoin to build per-occurrence variants, so the same
    // Step-6 occurrence-identity contract applies under gate=off.
    let reference = run_program(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(false)),
        SELFREC_TRIANGLE,
        &inputs,
    );
    assert_eq!(
        reference.wcoj_triangle_dispatch_count(),
        0,
        "gate=off must not dispatch"
    );
    let reference_rows = download_triples(reference.store().get("tri").expect("tri"));
    assert_eq!(
        reference_rows,
        vec![(1, 2, 3)],
        "F-W41-13 pinned reference: tri must equal {{(1,2,3)}}; got {:?}",
        reference_rows
    );

    // Gate on: W4.1 promoter admits the same-predicate
    // self-recursive triangle (paper P1). Step 6's
    // `rewrite_scan_nth` fix preserves occurrence identity for
    // both `p` Scans across the inputs and fallback views, so
    // each variant rewrites exactly one `p` occurrence to
    // `delta_p` and dispatches WCOJ. Total counter `>= 2`.
    let dispatched = run_program(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(true)),
        SELFREC_TRIANGLE,
        &inputs,
    );
    assert!(
        dispatched.wcoj_triangle_dispatch_count() >= 2,
        "self-recursive triangle (same-predicate `p` appearing twice) \
         must dispatch WCOJ on the seeding pass AND ≥ 1 variant in \
         the iteration loop; got counter {}",
        dispatched.wcoj_triangle_dispatch_count()
    );
    let dispatched_rows = download_triples(dispatched.store().get("tri").expect("tri"));
    assert_eq!(
        dispatched_rows, reference_rows,
        "self-recursive WCOJ row set must equal the binary-join reference"
    );
}
