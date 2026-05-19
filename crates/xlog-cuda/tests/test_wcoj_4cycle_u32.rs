// crates/xlog-cuda/tests/test_wcoj_4cycle_u32.rs
//! v0.6.5 slice 2 — 4-cycle WCOJ kernel (u32) provider tests.
//!
//! Locks the provider entry
//! `CudaKernelProvider::wcoj_4cycle_u32_recorded(e1, e2, e3, e4, launch_stream)`
//! against the rule
//!
//!   cycle4(W, X, Y, Z) :- e1(W, X), e2(X, Y), e3(Y, Z), e4(Z, W)
//!
//! Inputs (caller-supplied — sorted+deduped 2-column u32 buffers):
//!   * e1: lex-sorted by (W, X)
//!   * e2: lex-sorted by (X, Y)
//!   * e3: lex-sorted by (Y, Z)
//!   * e4: lex-sorted by (Z, W)
//!
//! Test surface mirrors `test_wcoj_triangle_u32.rs`:
//!   1. correctness with multiple 4-cycles vs CPU reference
//!   2. empty result on any empty input
//!   3. no false positives when the closing edge is missing
//!   4. deterministic lex output across two runs
//!
//! Hard boundaries (per slice spec):
//!   * U32 only.
//!   * Count → scan → materialize, deterministic output sorted (W, X, Y, Z).
//!   * Strict LaunchRecorder discipline; runtime-backed manager required.
//!   * No executor integration, no planner dispatch, no recursion.

use std::collections::BTreeSet;
use std::sync::Arc;

use cudarc::driver::sys;
use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::CudaBuffer;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

// ---------------------------------------------------------------
// Shared helpers (mirror test_wcoj_triangle_u32.rs)
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

/// Download a 4-column u32 result `CudaBuffer` to a host
/// `Vec<(u32, u32, u32, u32)>` for assertion. Sorted lex.
fn download_quads(buf: &CudaBuffer) -> Vec<(u32, u32, u32, u32)> {
    let n = buf.num_rows() as usize;
    if n == 0 {
        return Vec::new();
    }
    assert_eq!(buf.arity(), 4, "expected 4-column 4-cycle output");
    let mut bytes = [
        vec![0u8; n * 4],
        vec![0u8; n * 4],
        vec![0u8; n * 4],
        vec![0u8; n * 4],
    ];
    for i in 0..4 {
        unsafe {
            let res = sys::cuMemcpyDtoH_v2(
                bytes[i].as_mut_ptr() as *mut _,
                *buf.column(i).unwrap().device_ptr(),
                bytes[i].len(),
            );
            assert_eq!(res, sys::cudaError_enum::CUDA_SUCCESS);
        }
    }
    let mut out: Vec<(u32, u32, u32, u32)> = Vec::with_capacity(n);
    for i in 0..n {
        let w = u32::from_le_bytes(bytes[0][i * 4..i * 4 + 4].try_into().unwrap());
        let x = u32::from_le_bytes(bytes[1][i * 4..i * 4 + 4].try_into().unwrap());
        let y = u32::from_le_bytes(bytes[2][i * 4..i * 4 + 4].try_into().unwrap());
        let z = u32::from_le_bytes(bytes[3][i * 4..i * 4 + 4].try_into().unwrap());
        out.push((w, x, y, z));
    }
    out
}

/// CPU reference: enumerate all (W, X, Y, Z) such that
/// (W, X) ∈ e1 ∧ (X, Y) ∈ e2 ∧ (Y, Z) ∈ e3 ∧ (Z, W) ∈ e4.
/// Returns sorted, deduped.
fn cpu_4cycle_reference(
    e1: &[(u32, u32)],
    e2: &[(u32, u32)],
    e3: &[(u32, u32)],
    e4: &[(u32, u32)],
) -> Vec<(u32, u32, u32, u32)> {
    let e2_set: BTreeSet<(u32, u32)> = e2.iter().copied().collect();
    let e3_set: BTreeSet<(u32, u32)> = e3.iter().copied().collect();
    let e4_set: BTreeSet<(u32, u32)> = e4.iter().copied().collect();
    let mut out: BTreeSet<(u32, u32, u32, u32)> = BTreeSet::new();
    for &(w, x) in e1 {
        for &(x2, y) in e2 {
            if x2 != x {
                continue;
            }
            for &(y2, z) in e3 {
                if y2 != y {
                    continue;
                }
                if e4_set.contains(&(z, w)) && e2_set.contains(&(x, y)) && e3_set.contains(&(y, z))
                {
                    out.insert((w, x, y, z));
                }
            }
        }
    }
    out.into_iter().collect()
}

// ---------------------------------------------------------------
// Tests
// ---------------------------------------------------------------

#[test]
fn wcoj_4cycle_u32_matches_cpu_reference_perfect_square() {
    // Perfect 4-cycle on vertices {1, 2, 3, 4}: edges 1→2→3→4→1.
    // Each of e1, e2, e3, e4 carries the same edge set, so every
    // rotation of the cycle produces a closing 4-cycle:
    //   (1,2,3,4), (2,3,4,1), (3,4,1,2), (4,1,2,3)
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let edges: Vec<(u32, u32)> = vec![(1, 2), (2, 3), (3, 4), (4, 1)];
    let buf_e1 = upload_binary_u32(&fix.memory, &edges);
    let buf_e2 = upload_binary_u32(&fix.memory, &edges);
    let buf_e3 = upload_binary_u32(&fix.memory, &edges);
    let buf_e4 = upload_binary_u32(&fix.memory, &edges);

    let launch_stream = fix.pool.acquire().expect("acquire launch_stream");
    let result = fix
        .provider
        .wcoj_4cycle_u32_recorded(&buf_e1, &buf_e2, &buf_e3, &buf_e4, launch_stream)
        .expect("wcoj_4cycle_u32_recorded");

    let host_result = download_quads(&result);
    let expected = cpu_4cycle_reference(&edges, &edges, &edges, &edges);
    assert_eq!(
        host_result, expected,
        "GPU 4-cycle output must match CPU reference"
    );
    assert_eq!(
        host_result.len(),
        4,
        "perfect 4-cycle fixture must produce 4 quads (one per starting vertex)"
    );
    assert!(host_result.contains(&(1, 2, 3, 4)));
    assert!(host_result.contains(&(2, 3, 4, 1)));
    assert!(host_result.contains(&(3, 4, 1, 2)));
    assert!(host_result.contains(&(4, 1, 2, 3)));
}

#[test]
fn wcoj_4cycle_u32_matches_cpu_reference_with_chord() {
    // Square + diagonal chord: 1→2→3→4→1 plus 1→3 in some
    // relations to see whether the kernel generalizes beyond
    // the trivial-rotation case.
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let mut edges: Vec<(u32, u32)> = vec![(1, 2), (1, 3), (2, 3), (3, 4), (4, 1)];
    edges.sort();
    let buf_e1 = upload_binary_u32(&fix.memory, &edges);
    let buf_e2 = upload_binary_u32(&fix.memory, &edges);
    let buf_e3 = upload_binary_u32(&fix.memory, &edges);
    let buf_e4 = upload_binary_u32(&fix.memory, &edges);

    let launch_stream = fix.pool.acquire().expect("acquire launch_stream");
    let result = fix
        .provider
        .wcoj_4cycle_u32_recorded(&buf_e1, &buf_e2, &buf_e3, &buf_e4, launch_stream)
        .expect("wcoj_4cycle_u32_recorded");

    let host_result = download_quads(&result);
    let expected = cpu_4cycle_reference(&edges, &edges, &edges, &edges);
    assert_eq!(
        host_result, expected,
        "GPU 4-cycle output must match CPU reference (chord fixture)"
    );
}

#[test]
fn wcoj_4cycle_u32_empty_inputs_produce_empty_output() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let empty: Vec<(u32, u32)> = Vec::new();
    let buf = upload_binary_u32(&fix.memory, &empty);
    let launch_stream = fix.pool.acquire().expect("acquire launch_stream");
    let result = fix
        .provider
        .wcoj_4cycle_u32_recorded(&buf, &buf, &buf, &buf, launch_stream)
        .expect("empty inputs must succeed");
    assert_eq!(result.num_rows(), 0);
    assert_eq!(result.arity(), 4);
}

#[test]
fn wcoj_4cycle_u32_no_false_positives_on_open_chain() {
    // Open chain: e1, e2, e3 all carry the edge sequence 1→2→3→4
    // but e4 LACKS the closing edge (4, 1). No 4-cycle should emit.
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let chain: Vec<(u32, u32)> = vec![(1, 2), (2, 3), (3, 4)];
    let buf_e1 = upload_binary_u32(&fix.memory, &chain);
    let buf_e2 = upload_binary_u32(&fix.memory, &chain);
    let buf_e3 = upload_binary_u32(&fix.memory, &chain);
    // e4 is empty — no closing edges anywhere.
    let buf_e4 = upload_binary_u32(&fix.memory, &Vec::new());

    let launch_stream = fix.pool.acquire().expect("acquire launch_stream");
    let result = fix
        .provider
        .wcoj_4cycle_u32_recorded(&buf_e1, &buf_e2, &buf_e3, &buf_e4, launch_stream)
        .expect("wcoj_4cycle_u32_recorded");
    assert_eq!(
        result.num_rows(),
        0,
        "open chain (no closing edge in e4) must NOT emit any 4-cycle"
    );
}

#[test]
fn wcoj_4cycle_u32_deterministic_lex_output() {
    // Same fixture as `matches_cpu_reference_perfect_square`,
    // run twice; the row-set and row-order must agree.
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let edges: Vec<(u32, u32)> = vec![(1, 2), (2, 3), (3, 4), (4, 1)];

    let run = || -> Vec<(u32, u32, u32, u32)> {
        let buf_e1 = upload_binary_u32(&fix.memory, &edges);
        let buf_e2 = upload_binary_u32(&fix.memory, &edges);
        let buf_e3 = upload_binary_u32(&fix.memory, &edges);
        let buf_e4 = upload_binary_u32(&fix.memory, &edges);
        let launch_stream = fix.pool.acquire().expect("acquire launch_stream");
        let result = fix
            .provider
            .wcoj_4cycle_u32_recorded(&buf_e1, &buf_e2, &buf_e3, &buf_e4, launch_stream)
            .expect("wcoj_4cycle_u32_recorded");
        download_quads(&result)
    };
    let first = run();
    let second = run();
    assert_eq!(first, second, "kernel output must be deterministic");
}
