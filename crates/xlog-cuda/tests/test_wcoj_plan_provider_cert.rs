// crates/xlog-cuda/tests/test_wcoj_plan_provider_cert.rs
//! v0.6.2 planner-to-provider certification (pure-Rust seam).
//!
//! Locks the contract that the **hypergraph planner** (PRs 5–9,
//! pure Rust) and the **GPU WCOJ provider** (PRs 1–3 of the
//! GPU stack: triangle kernel, device scan, sorted-layout
//! construction) agree on the same fixture.
//!
//! Test surface:
//!   1. Triangle rule plans as `MultiwayCandidate`, the GPU
//!      provider's `wcoj_layout_u32_recorded` →
//!      `wcoj_triangle_u32_recorded` pipeline produces row-for-row
//!      identical output to `evaluate_rule_typed` on the same
//!      logical fixture.
//!   2. Unsupported key type (I32 — outside
//!      `WCOJ_SUPPORTED_KEY_TYPES`) plans as `BinaryFallback` with
//!      a structured `UnsupportedKeyType` boundary; the GPU path
//!      is never invoked.
//!   3. >4 distinct join-key variables plans as `BinaryFallback`
//!      with `JoinKeysExceedBinaryFallbackLimit`; the GPU path is
//!      never invoked.
//!
//! Hard boundaries (per slice spec):
//!   * No automatic dispatch from planner to GPU — the test
//!     manually orchestrates the two layers.
//!   * No runtime executor wiring.
//!   * No new kernels.
//!   * No new public xlog-cuda API.
//!
//! This test is the seam every future GPU dispatcher must clear:
//! before any executor wires `RulePlan::MultiwayCandidate` into a
//! GPU launch, the row sets must match the CPU oracle on the
//! same fixture. The integration test in
//! `test_wcoj_layout_u32.rs` (`wcoj_layout_then_triangle_u32_matches_cpu_oracle`)
//! covers the GPU pipeline alone; this test wraps the planner
//! around it to certify the dispatch contract end-to-end.

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
use xlog_logic::hypergraph::{
    evaluate_rule_typed, plan_rule, AppearanceOrder, Boundary, RefRelation as LogicRefRelation,
    RefRelationStore, RefValue, RulePlan,
};

// ---------------------------------------------------------------
// Fixture helpers (mirror the conventions of the WCOJ test files)
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
    provider: CudaKernelProvider,
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
        CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory)).ok()?;
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

/// Build a 2-column U32 [`LogicRefRelation`] from a host-side
/// `Vec<(u32, u32)>`. Used as the planner's view of the same
/// pairs the GPU sees.
fn logic_u32_relation(rows: &[(u32, u32)]) -> LogicRefRelation {
    LogicRefRelation {
        schema: vec![ScalarType::U32, ScalarType::U32],
        rows: rows
            .iter()
            .map(|(a, b)| vec![RefValue::U32(*a), RefValue::U32(*b)])
            .collect(),
    }
}

/// Upload a host-side `Vec<(u32, u32)>` to a 2-column u32
/// `CudaBuffer`. The GPU layout pass will sort + dedup it.
fn upload_binary_u32_unsorted(memory: &Arc<GpuMemoryManager>, rows: &[(u32, u32)]) -> CudaBuffer {
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
    out
}

fn oracle_rows_to_triples(rows: Vec<Vec<RefValue>>) -> Vec<(u32, u32, u32)> {
    let mut out: Vec<(u32, u32, u32)> = rows
        .into_iter()
        .map(|r| match (&r[0], &r[1], &r[2]) {
            (RefValue::U32(a), RefValue::U32(b), RefValue::U32(c)) => (*a, *b, *c),
            other => panic!("unexpected oracle row shape: {other:?}"),
        })
        .collect();
    out.sort();
    out
}

// ---------------------------------------------------------------
// Test 1: planner accepts the triangle rule + GPU result matches
//          CPU oracle row-for-row.
// ---------------------------------------------------------------

#[test]
fn plan_provider_cert_triangle_matches_cpu_oracle() {
    // Logical rule: tri(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z)
    let r = rule_with(
        atom("tri", vec![var("X"), var("Y"), var("Z")]),
        vec![
            pos("e1", vec![var("X"), var("Y")]),
            pos("e2", vec![var("Y"), var("Z")]),
            pos("e3", vec![var("X"), var("Z")]),
        ],
    );

    // Same K_4 + small triangle fixture as the WCOJ tests.
    // Intentionally unsorted + duplicated so the layout pass has
    // real work; the planner sees the dedup'd CPU view.
    let raw_e1: Vec<(u32, u32)> = vec![
        (3, 4),
        (1, 2),
        (1, 3),
        (1, 2), // dup
        (2, 3),
        (5, 6),
        (1, 4),
        (5, 7),
        (2, 4),
        (6, 7),
        (3, 4), // dup
    ];
    let raw_e2: Vec<(u32, u32)> = vec![
        (3, 4),
        (2, 3),
        (6, 7),
        (2, 4),
        (3, 4), // dup
    ];
    let raw_e3: Vec<(u32, u32)> = vec![
        (1, 4),
        (2, 4),
        (3, 4),
        (1, 3),
        (5, 7),
        (1, 4), // dup
    ];

    // Planner view: sorted+deduped (the planner is set-semantic
    // so the rows-as-set matters, not insertion order).
    let mut e1_sorted = raw_e1.clone();
    e1_sorted.sort();
    e1_sorted.dedup();
    let mut e2_sorted = raw_e2.clone();
    e2_sorted.sort();
    e2_sorted.dedup();
    let mut e3_sorted = raw_e3.clone();
    e3_sorted.sort();
    e3_sorted.dedup();

    let mut store: RefRelationStore = BTreeMap::new();
    store.insert("e1".to_string(), logic_u32_relation(&e1_sorted));
    store.insert("e2".to_string(), logic_u32_relation(&e2_sorted));
    store.insert("e3".to_string(), logic_u32_relation(&e3_sorted));

    // ----- planner layer -----
    let plan = plan_rule(&r, &store).expect("plan must succeed for U32 triangle");
    match &plan {
        RulePlan::MultiwayCandidate {
            head_predicate,
            hypergraph,
            variable_order,
        } => {
            assert_eq!(head_predicate, "tri");
            assert_eq!(hypergraph.hyperedge_count(), 3);
            assert_eq!(variable_order.len(), 3);
        }
        other => panic!(
            "expected MultiwayCandidate for U32 triangle, got {other:?} — \
             planner and provider would disagree on this fixture"
        ),
    }

    // ----- CPU oracle -----
    let oracle_rows = evaluate_rule_typed(&r, &store, &AppearanceOrder)
        .expect("evaluate_rule_typed must succeed for the same fixture the planner accepted");
    let cpu_triples = oracle_rows_to_triples(oracle_rows);

    // ----- GPU layer -----
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping GPU comparison: CUDA runtime unavailable");
        // Planner verdict alone is enough to lock the
        // pure-Rust contract; the GPU-vs-CPU equality below is
        // the additional cert that requires a device.
        return;
    };
    let buf_e1_raw = upload_binary_u32_unsorted(&fix.memory, &raw_e1);
    let buf_e2_raw = upload_binary_u32_unsorted(&fix.memory, &raw_e2);
    let buf_e3_raw = upload_binary_u32_unsorted(&fix.memory, &raw_e3);

    let layout_stream = fix.pool.acquire().expect("layout stream");
    let buf_e1 = fix
        .provider
        .wcoj_layout_u32_recorded(&buf_e1_raw, layout_stream)
        .expect("layout e1");
    let buf_e2 = fix
        .provider
        .wcoj_layout_u32_recorded(&buf_e2_raw, layout_stream)
        .expect("layout e2");
    let buf_e3 = fix
        .provider
        .wcoj_layout_u32_recorded(&buf_e3_raw, layout_stream)
        .expect("layout e3");

    let triangle_stream = fix.pool.acquire().expect("triangle stream");
    let result = fix
        .provider
        .wcoj_triangle_u32_recorded(&buf_e1, &buf_e2, &buf_e3, triangle_stream)
        .expect("triangle");
    let gpu_triples = download_triples(&result);

    // The seam: planner says multiway, oracle says X, GPU says Y.
    // X must equal Y exactly.
    assert_eq!(
        gpu_triples, cpu_triples,
        "planner accepted MultiwayCandidate but GPU output disagrees with CPU oracle"
    );
}

// ---------------------------------------------------------------
// Test 2: unsupported key type stays off the GPU WCOJ path at the
//          plan layer.
// ---------------------------------------------------------------

#[test]
fn plan_provider_cert_unsupported_key_type_falls_back_at_planner() {
    // Same triangle rule shape, but the relation schemas use I32
    // (outside `WCOJ_SUPPORTED_KEY_TYPES = {U32, U64, Symbol}`).
    // The planner must reject this as a multiway candidate.
    let r = rule_with(
        atom("tri", vec![var("X"), var("Y"), var("Z")]),
        vec![
            pos("e1", vec![var("X"), var("Y")]),
            pos("e2", vec![var("Y"), var("Z")]),
            pos("e3", vec![var("X"), var("Z")]),
        ],
    );

    let i32_relation = LogicRefRelation {
        schema: vec![ScalarType::I32, ScalarType::I32],
        rows: vec![vec![RefValue::I32(1), RefValue::I32(2)]],
    };
    let mut store: RefRelationStore = BTreeMap::new();
    store.insert("e1".to_string(), i32_relation.clone());
    store.insert("e2".to_string(), i32_relation.clone());
    store.insert("e3".to_string(), i32_relation);

    let plan = plan_rule(&r, &store).expect("plan must classify rather than error");
    match &plan {
        RulePlan::BinaryFallback {
            head_predicate,
            boundaries,
        } => {
            assert_eq!(head_predicate, "tri");
            // I32 join key (Y in this rule's join graph) must
            // surface as UnsupportedKeyType. Other join-key
            // variables (X, Z) likewise — but locking on at
            // least one I32 boundary is enough to certify the
            // fallback verdict.
            let has_i32 = boundaries.iter().any(|b| {
                matches!(
                    b,
                    Boundary::UnsupportedKeyType {
                        ty: ScalarType::I32,
                        ..
                    }
                )
            });
            assert!(
                has_i32,
                "expected UnsupportedKeyType I32 boundary for I32 join key, got {boundaries:?}"
            );
        }
        other => panic!(
            "expected BinaryFallback for I32 triangle (unsupported key type), got {other:?} — \
             dispatcher would route this through the GPU WCOJ path incorrectly"
        ),
    }

    // GPU path must not be invoked on this fixture. We assert
    // by *not* invoking it — any future dispatcher that ignores
    // the planner's BinaryFallback verdict would surface as a
    // separate test failure, not in this one.
}

// ---------------------------------------------------------------
// Test 3: >4 distinct join-key variables stays off the GPU WCOJ
//          path at the plan layer.
// ---------------------------------------------------------------

#[test]
fn plan_provider_cert_exceeds_join_key_limit_falls_back_at_planner() {
    // 5 distinct join-key variables (each shared by ≥2 atoms) →
    // exceeds `BINARY_FALLBACK_KEY_LIMIT = 4`. The planner must
    // emit `JoinKeysExceedBinaryFallbackLimit`.
    //
    // Construction: chain of 6 binary atoms sharing 5 variables.
    //   q(A, F) :- e(A, B), e(B, C), e(C, D), e(D, E), e(E, F), e(A, F)
    // Variables: A, B, C, D, E, F. Each of B, C, D, E, F, A appears
    // in 2 atoms → 6 join keys (above the 4-key limit).
    let r = rule_with(
        atom("q", vec![var("A"), var("F")]),
        vec![
            pos("e", vec![var("A"), var("B")]),
            pos("e", vec![var("B"), var("C")]),
            pos("e", vec![var("C"), var("D")]),
            pos("e", vec![var("D"), var("E")]),
            pos("e", vec![var("E"), var("F")]),
            pos("e", vec![var("A"), var("F")]),
        ],
    );

    let mut store: RefRelationStore = BTreeMap::new();
    store.insert(
        "e".to_string(),
        logic_u32_relation(&[(1u32, 2u32), (2, 3), (3, 4), (4, 5), (5, 6), (1, 6)]),
    );

    let plan = plan_rule(&r, &store).expect("plan must classify rather than error");
    match &plan {
        RulePlan::BinaryFallback {
            head_predicate,
            boundaries,
        } => {
            assert_eq!(head_predicate, "q");
            let has_key_limit = boundaries
                .iter()
                .any(|b| matches!(b, Boundary::JoinKeysExceedBinaryFallbackLimit { .. }));
            assert!(
                has_key_limit,
                "expected JoinKeysExceedBinaryFallbackLimit boundary for 5+ join keys, \
                 got {boundaries:?}"
            );
        }
        other => panic!(
            "expected BinaryFallback for >4 join keys, got {other:?} — \
             dispatcher would route this through the GPU WCOJ path incorrectly"
        ),
    }
}
