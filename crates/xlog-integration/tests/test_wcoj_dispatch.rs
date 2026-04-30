// crates/xlog-integration/tests/test_wcoj_dispatch.rs
//! v0.6.2 minimal env-gated WCOJ triangle dispatch.
//!
//! Locks the dispatch contract for the entry
//! `xlog_integration::wcoj_dispatch::try_wcoj_triangle_u32_dispatch`:
//!
//!   * Env-gated via `XLOG_USE_WCOJ_TRIANGLE_U32`. The dispatch
//!     uses the helper's "inner" form which takes the gate as an
//!     explicit `bool` to keep tests free of process-global env
//!     races.
//!   * Recognizes exactly the
//!     `tri(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z)`-shaped rule:
//!     three positive 2-arity body atoms over distinct head
//!     variables (X, Y, Z) in head-position order, no negation,
//!     no comparison filters, no recursion (head predicate not in
//!     body), all input buffers 2-column u32, and `plan_rule`
//!     returns `RulePlan::MultiwayCandidate`.
//!   * Falls back **silently** (returns `Ok(None)`) for any
//!     mismatched shape — no error, no log noise. Caller is
//!     expected to take the existing binary-join path.
//!   * On match, builds three sorted+deduped layouts via
//!     `wcoj_layout_u32_recorded` and runs
//!     `wcoj_triangle_u32_recorded` on the configured launch
//!     stream.
//!
//! The integration arc:
//!   1. Build the same triangle rule as a `.xlog` source program.
//!   2. Compile + execute it through the existing executor
//!      (binary-join chain via `Executor::execute_plan`).
//!   3. Run the new env-gated dispatch helper on the same fixture.
//!   4. Compare the row sets (column-for-column, sorted lex).
//!
//! Out of scope per spec: recursion, SCC mixed execution, cost
//! model, Symbol / u64, histogram dispatch.

use std::collections::BTreeMap;
use std::sync::Arc;

use cudarc::driver::sys;
use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::CudaBuffer;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

use xlog_logic::ast::{Atom, BodyLiteral, Rule, Term};
use xlog_logic::Compiler;
use xlog_runtime::Executor;

use xlog_integration::wcoj_dispatch::try_wcoj_triangle_u32_dispatch_with_gate;

// ---------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------

struct DiscardSink;
impl LoggingSink for DiscardSink {
    fn emit(&self, _record: LogRecord) -> Result<(), SinkError> {
        Ok(())
    }
}

#[allow(dead_code)] // device/runtime kept alive via Arc clones for cross-stream lifetimes
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

fn var(name: &str) -> Term {
    Term::Variable(name.to_string())
}

fn atom(predicate: &str, terms: Vec<Term>) -> Atom {
    Atom {
        predicate: predicate.to_string(),
        terms,
    }
}

fn pos(predicate: &str, terms: Vec<Term>) -> BodyLiteral {
    BodyLiteral::Positive(atom(predicate, terms))
}

fn rule_with(head: Atom, body: Vec<BodyLiteral>) -> Rule {
    Rule { head, body }
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
                let res = sys::cuMemcpyDtoH_v2(
                    count_host.as_mut_ptr() as *mut _,
                    *buf.num_rows_device().device_ptr(),
                    std::mem::size_of::<u32>(),
                );
                assert_eq!(res, sys::cudaError_enum::CUDA_SUCCESS);
            }
            count_host[0] as usize
        }
    };
    if n == 0 {
        return Vec::new();
    }
    assert_eq!(buf.arity(), 3, "expected 3-column triangle output");
    let mut col0_bytes = vec![0u8; n * 4];
    let mut col1_bytes = vec![0u8; n * 4];
    let mut col2_bytes = vec![0u8; n * 4];
    unsafe {
        let res0 = sys::cuMemcpyDtoH_v2(
            col0_bytes.as_mut_ptr() as *mut _,
            *buf.column(0).unwrap().device_ptr(),
            col0_bytes.len(),
        );
        let res1 = sys::cuMemcpyDtoH_v2(
            col1_bytes.as_mut_ptr() as *mut _,
            *buf.column(1).unwrap().device_ptr(),
            col1_bytes.len(),
        );
        let res2 = sys::cuMemcpyDtoH_v2(
            col2_bytes.as_mut_ptr() as *mut _,
            *buf.column(2).unwrap().device_ptr(),
            col2_bytes.len(),
        );
        assert_eq!(res0, sys::cudaError_enum::CUDA_SUCCESS);
        assert_eq!(res1, sys::cudaError_enum::CUDA_SUCCESS);
        assert_eq!(res2, sys::cudaError_enum::CUDA_SUCCESS);
    }
    let mut out: Vec<(u32, u32, u32)> = Vec::with_capacity(n);
    for i in 0..n {
        let x = u32::from_le_bytes(col0_bytes[i * 4..i * 4 + 4].try_into().unwrap());
        let y = u32::from_le_bytes(col1_bytes[i * 4..i * 4 + 4].try_into().unwrap());
        let z = u32::from_le_bytes(col2_bytes[i * 4..i * 4 + 4].try_into().unwrap());
        out.push((x, y, z));
    }
    out.sort();
    out.dedup();
    out
}

/// Run the same triangle rule through the existing
/// Compiler + Executor pipeline (binary-join chain). Returns
/// the row set of the head predicate.
fn execute_via_compiler(
    fix: &RuntimeFixture,
    source: &str,
    head_predicate: &str,
    inputs: &BTreeMap<&str, Vec<(u32, u32)>>,
) -> Vec<(u32, u32, u32)> {
    let mut compiler = Compiler::new();
    let plan = compiler.compile(source).expect("compile");

    let mut executor = Executor::new(Arc::clone(&fix.provider));
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    for (name, rows) in inputs {
        let buf = upload_binary_u32(&fix.memory, rows);
        executor.put_relation(name, buf);
    }
    let _ = executor.execute_plan(&plan).expect("execute_plan");
    let head_buf = executor
        .store()
        .get(head_predicate)
        .expect("head predicate present");
    download_triples(head_buf)
}

// ---------------------------------------------------------------
// Tests
// ---------------------------------------------------------------

#[test]
fn dispatch_gate_off_returns_none() {
    // With the gate explicitly off, the helper must return Ok(None)
    // even when the rule + inputs match the WCOJ shape — locking
    // the "no surprise opt-in" contract. Caller takes the existing
    // path silently.
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let r = rule_with(
        atom("tri", vec![var("X"), var("Y"), var("Z")]),
        vec![
            pos("e1", vec![var("X"), var("Y")]),
            pos("e2", vec![var("Y"), var("Z")]),
            pos("e3", vec![var("X"), var("Z")]),
        ],
    );
    let mut inputs: BTreeMap<String, CudaBuffer> = BTreeMap::new();
    inputs.insert("e1".into(), upload_binary_u32(&fix.memory, &[(1, 2)]));
    inputs.insert("e2".into(), upload_binary_u32(&fix.memory, &[(2, 3)]));
    inputs.insert("e3".into(), upload_binary_u32(&fix.memory, &[(1, 3)]));
    let stream = fix.pool.acquire().expect("stream");
    let result =
        try_wcoj_triangle_u32_dispatch_with_gate(false, &r, &inputs, &fix.provider, stream)
            .expect("must not error on gate-off path");
    assert!(
        result.is_none(),
        "gate-off must return None even on matching shape, got {:?}",
        result.is_some()
    );
}

#[test]
fn dispatch_gate_on_matching_triangle_matches_executor_output() {
    // Gate on + matching shape → dispatch returns Some(buffer);
    // row set must equal the existing executor's output for the
    // same triangle rule on the same inputs.
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let raw_e1: Vec<(u32, u32)> = vec![
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
        (5, 6),
        (5, 7),
        (6, 7),
    ];
    let raw_e2: Vec<(u32, u32)> = vec![(2, 3), (2, 4), (3, 4), (6, 7)];
    let raw_e3: Vec<(u32, u32)> = vec![(1, 3), (1, 4), (2, 4), (3, 4), (5, 7)];

    // ----- existing executor path (binary-join chain) -----
    let source = "
        tri(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z).
    ";
    let mut named_inputs: BTreeMap<&str, Vec<(u32, u32)>> = BTreeMap::new();
    named_inputs.insert("e1", raw_e1.clone());
    named_inputs.insert("e2", raw_e2.clone());
    named_inputs.insert("e3", raw_e3.clone());
    let mut executor_rows = execute_via_compiler(&fix, source, "tri", &named_inputs);
    executor_rows.sort();
    executor_rows.dedup();

    // ----- new dispatch path -----
    let r = rule_with(
        atom("tri", vec![var("X"), var("Y"), var("Z")]),
        vec![
            pos("e1", vec![var("X"), var("Y")]),
            pos("e2", vec![var("Y"), var("Z")]),
            pos("e3", vec![var("X"), var("Z")]),
        ],
    );
    let mut inputs: BTreeMap<String, CudaBuffer> = BTreeMap::new();
    inputs.insert("e1".into(), upload_binary_u32(&fix.memory, &raw_e1));
    inputs.insert("e2".into(), upload_binary_u32(&fix.memory, &raw_e2));
    inputs.insert("e3".into(), upload_binary_u32(&fix.memory, &raw_e3));
    let stream = fix.pool.acquire().expect("stream");
    let dispatched =
        try_wcoj_triangle_u32_dispatch_with_gate(true, &r, &inputs, &fix.provider, stream)
            .expect("must succeed under gate-on with matching shape")
            .expect("must return Some on matching shape");
    let dispatched_rows = download_triples(&dispatched);

    assert_eq!(
        dispatched_rows, executor_rows,
        "WCOJ dispatch output must equal existing executor output \
         row-for-row on the same fixture"
    );
}

#[test]
fn dispatch_gate_on_two_atoms_falls_back_silently() {
    // Body has 2 atoms, not 3 → triangle shape mismatch → Ok(None).
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let r = rule_with(
        atom("path", vec![var("X"), var("Z")]),
        vec![
            pos("e1", vec![var("X"), var("Y")]),
            pos("e2", vec![var("Y"), var("Z")]),
        ],
    );
    let mut inputs: BTreeMap<String, CudaBuffer> = BTreeMap::new();
    inputs.insert("e1".into(), upload_binary_u32(&fix.memory, &[(1, 2)]));
    inputs.insert("e2".into(), upload_binary_u32(&fix.memory, &[(2, 3)]));
    let stream = fix.pool.acquire().expect("stream");
    let result = try_wcoj_triangle_u32_dispatch_with_gate(true, &r, &inputs, &fix.provider, stream)
        .expect("must not error on shape mismatch");
    assert!(
        result.is_none(),
        "2-atom rule must fall back silently; got Some(_)"
    );
}

#[test]
fn dispatch_gate_on_recursive_rule_falls_back_silently() {
    // tri's head predicate appears in the body (recursive) → not
    // dispatchable. Caller path must take over.
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let r = rule_with(
        atom("tri", vec![var("X"), var("Y"), var("Z")]),
        vec![
            pos("e1", vec![var("X"), var("Y")]),
            pos("e2", vec![var("Y"), var("Z")]),
            pos("tri", vec![var("X"), var("Y"), var("Z")]),
        ],
    );
    let mut inputs: BTreeMap<String, CudaBuffer> = BTreeMap::new();
    inputs.insert("e1".into(), upload_binary_u32(&fix.memory, &[(1, 2)]));
    inputs.insert("e2".into(), upload_binary_u32(&fix.memory, &[(2, 3)]));
    inputs.insert("tri".into(), upload_binary_u32(&fix.memory, &[(1, 2)]));
    let stream = fix.pool.acquire().expect("stream");
    let result = try_wcoj_triangle_u32_dispatch_with_gate(true, &r, &inputs, &fix.provider, stream)
        .expect("must not error on recursive shape");
    assert!(
        result.is_none(),
        "recursive rule must fall back silently; got Some(_)"
    );
}

#[test]
fn dispatch_gate_on_atom_args_reversed_falls_back_silently() {
    // Triangle topology but one atom's args are NOT in head-order
    // — e1 is (Y, X) instead of (X, Y). v1 dispatch is strict on
    // head-order alignment; reversed-arg atoms fall back.
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let r = rule_with(
        atom("tri", vec![var("X"), var("Y"), var("Z")]),
        vec![
            pos("e1", vec![var("Y"), var("X")]), // reversed
            pos("e2", vec![var("Y"), var("Z")]),
            pos("e3", vec![var("X"), var("Z")]),
        ],
    );
    let mut inputs: BTreeMap<String, CudaBuffer> = BTreeMap::new();
    inputs.insert("e1".into(), upload_binary_u32(&fix.memory, &[(1, 2)]));
    inputs.insert("e2".into(), upload_binary_u32(&fix.memory, &[(2, 3)]));
    inputs.insert("e3".into(), upload_binary_u32(&fix.memory, &[(1, 3)]));
    let stream = fix.pool.acquire().expect("stream");
    let result = try_wcoj_triangle_u32_dispatch_with_gate(true, &r, &inputs, &fix.provider, stream)
        .expect("must not error on reversed-arg shape");
    assert!(
        result.is_none(),
        "atom args not in head order must fall back silently; got Some(_)"
    );
}

#[test]
fn dispatch_gate_on_non_u32_buffer_falls_back_silently() {
    // Triangle topology but one input buffer has I32 schema
    // instead of U32. The dispatch helper checks input schemas
    // BEFORE calling into the WCOJ kernel and returns None on
    // mismatch — caller path takes over.
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let r = rule_with(
        atom("tri", vec![var("X"), var("Y"), var("Z")]),
        vec![
            pos("e1", vec![var("X"), var("Y")]),
            pos("e2", vec![var("Y"), var("Z")]),
            pos("e3", vec![var("X"), var("Z")]),
        ],
    );
    // Build an I32-schema buffer for e1.
    let i32_schema = Schema::new(vec![
        ("col0".to_string(), ScalarType::I32),
        ("col1".to_string(), ScalarType::I32),
    ]);
    let bytes = std::mem::size_of::<i32>() * 2;
    let mut col0 = fix.memory.alloc::<u8>(bytes).expect("alloc");
    let mut col1 = fix.memory.alloc::<u8>(bytes).expect("alloc");
    let mut d_num_rows = fix.memory.alloc::<u32>(1).expect("alloc");
    let device = fix.memory.device().inner();
    let pair: [i32; 2] = [1, 2];
    let pair_bytes: Vec<u8> = pair.iter().flat_map(|v| v.to_le_bytes()).collect();
    device
        .htod_sync_copy_into(&pair_bytes, &mut col0)
        .expect("htod");
    device
        .htod_sync_copy_into(&pair_bytes, &mut col1)
        .expect("htod");
    device
        .htod_sync_copy_into(&[1u32], &mut d_num_rows)
        .expect("htod count");
    let i32_buf = CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into()],
        1,
        d_num_rows,
        i32_schema,
        1,
    );

    let mut inputs: BTreeMap<String, CudaBuffer> = BTreeMap::new();
    inputs.insert("e1".into(), i32_buf);
    inputs.insert("e2".into(), upload_binary_u32(&fix.memory, &[(2, 3)]));
    inputs.insert("e3".into(), upload_binary_u32(&fix.memory, &[(1, 3)]));
    let stream = fix.pool.acquire().expect("stream");
    let result = try_wcoj_triangle_u32_dispatch_with_gate(true, &r, &inputs, &fix.provider, stream)
        .expect("must not error on non-u32 shape");
    assert!(
        result.is_none(),
        "non-u32 input must fall back silently; got Some(_)"
    );
}

// ---------------------------------------------------------------
// U64 + mixed-width dispatch surface (commit 3 of v0.6.2 u64 slice).
//
// The dispatch helper widens the shape gate to accept either
// 4-byte (U32 / Symbol) or 8-byte (U64) inputs uniformly across
// all three slots; mixed-width triangles must return Ok(None) so
// the existing binary-join chain handles them.
// ---------------------------------------------------------------

/// Upload a host-side `Vec<(u64, u64)>` to a 2-column U64
/// `CudaBuffer`. Mirror of `upload_binary_u32`.
fn upload_binary_u64(memory: &Arc<GpuMemoryManager>, rows: &[(u64, u64)]) -> CudaBuffer {
    let n = rows.len() as u32;
    let col0_host: Vec<u64> = rows.iter().map(|(a, _)| *a).collect();
    let col1_host: Vec<u64> = rows.iter().map(|(_, b)| *b).collect();
    let bytes_per_col = (n as usize) * std::mem::size_of::<u64>();
    let mut col0 = memory.alloc::<u8>(bytes_per_col).expect("alloc col0");
    let mut col1 = memory.alloc::<u8>(bytes_per_col).expect("alloc col1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let device = memory.device().inner();
    if !col0_host.is_empty() {
        let c0: Vec<u8> = col0_host.iter().flat_map(|v| v.to_le_bytes()).collect();
        let c1: Vec<u8> = col1_host.iter().flat_map(|v| v.to_le_bytes()).collect();
        device.htod_sync_copy_into(&c0, &mut col0).expect("htod c0");
        device.htod_sync_copy_into(&c1, &mut col1).expect("htod c1");
    }
    device
        .htod_sync_copy_into(&[n], &mut d_num_rows)
        .expect("htod d_num_rows");
    let schema = Schema::new(vec![
        ("col0".to_string(), ScalarType::U64),
        ("col1".to_string(), ScalarType::U64),
    ]);
    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into()],
        n as u64,
        d_num_rows,
        schema,
        n,
    )
}

#[test]
fn dispatch_gate_on_matching_u64_triangle_returns_some() {
    // Gate on, U64-typed inputs across all three slots, valid
    // triangle shape → dispatch must produce Some(buffer) with
    // U64 output schema preserved per column.
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    // Shift the existing multi-triangle fixture into hi-half u64
    // space so a buggy width-truncating dispatch (e.g. routing
    // U64 to the U32 entry) would visibly fail.
    let big = (u32::MAX as u64) + 1;
    let raw_e1: Vec<(u64, u64)> = vec![
        (big + 1, big + 2),
        (big + 1, big + 3),
        (big + 1, big + 4),
        (big + 2, big + 3),
        (big + 2, big + 4),
        (big + 3, big + 4),
        (big + 5, big + 6),
        (big + 5, big + 7),
        (big + 6, big + 7),
    ];
    let raw_e2: Vec<(u64, u64)> = vec![
        (big + 2, big + 3),
        (big + 2, big + 4),
        (big + 3, big + 4),
        (big + 6, big + 7),
    ];
    let raw_e3: Vec<(u64, u64)> = vec![
        (big + 1, big + 3),
        (big + 1, big + 4),
        (big + 2, big + 4),
        (big + 3, big + 4),
        (big + 5, big + 7),
    ];
    let r = rule_with(
        atom("tri", vec![var("X"), var("Y"), var("Z")]),
        vec![
            pos("e1", vec![var("X"), var("Y")]),
            pos("e2", vec![var("Y"), var("Z")]),
            pos("e3", vec![var("X"), var("Z")]),
        ],
    );
    let mut inputs: BTreeMap<String, CudaBuffer> = BTreeMap::new();
    inputs.insert("e1".into(), upload_binary_u64(&fix.memory, &raw_e1));
    inputs.insert("e2".into(), upload_binary_u64(&fix.memory, &raw_e2));
    inputs.insert("e3".into(), upload_binary_u64(&fix.memory, &raw_e3));
    let stream = fix.pool.acquire().expect("stream");
    let dispatched =
        try_wcoj_triangle_u32_dispatch_with_gate(true, &r, &inputs, &fix.provider, stream)
            .expect("must not error on U64 triangle")
            .expect("must dispatch on U64 triangle");
    // Output schema must be U64 × 3 (provider preserves it).
    assert_eq!(dispatched.schema.column_type(0), Some(ScalarType::U64));
    assert_eq!(dispatched.schema.column_type(1), Some(ScalarType::U64));
    assert_eq!(dispatched.schema.column_type(2), Some(ScalarType::U64));
    assert_eq!(
        dispatched.num_rows() as usize,
        5,
        "U64 dispatch must produce same 5 triangles as U32 dispatch on identical-shape fixture"
    );
}

#[test]
fn dispatch_gate_on_mixed_u32_u64_triangle_falls_back_silently() {
    // Gate on, valid triangle shape, but slot widths are mixed:
    // e1 + e3 are U32, e2 is U64. Cross-relation type
    // compatibility would have been rejected upstream by the
    // planner via `analyze_typed`; this test locks that the
    // dispatch helper itself also bails (returns Ok(None))
    // rather than picking a width arbitrarily and running
    // bit-equality joins across U32/U64 buffers.
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let r = rule_with(
        atom("tri", vec![var("X"), var("Y"), var("Z")]),
        vec![
            pos("e1", vec![var("X"), var("Y")]),
            pos("e2", vec![var("Y"), var("Z")]),
            pos("e3", vec![var("X"), var("Z")]),
        ],
    );
    let mut inputs: BTreeMap<String, CudaBuffer> = BTreeMap::new();
    inputs.insert("e1".into(), upload_binary_u32(&fix.memory, &[(1, 2)]));
    inputs.insert("e2".into(), upload_binary_u64(&fix.memory, &[(2, 3)]));
    inputs.insert("e3".into(), upload_binary_u32(&fix.memory, &[(1, 3)]));
    let stream = fix.pool.acquire().expect("stream");
    let result = try_wcoj_triangle_u32_dispatch_with_gate(true, &r, &inputs, &fix.provider, stream)
        .expect("must not error on mixed-width inputs");
    assert!(
        result.is_none(),
        "mixed-width triangle (U32 + U64 slots) must fall back silently; got Some(_)"
    );
}
