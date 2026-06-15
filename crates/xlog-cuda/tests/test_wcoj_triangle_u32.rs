// crates/xlog-cuda/tests/test_wcoj_triangle_u32.rs
//! Tests for the GPU 3-way WCOJ triangle kernel for u32 inputs.
//!
//! Locks the provider-only entry
//! `CudaKernelProvider::wcoj_triangle_u32_recorded(e_xy, e_yz, e_xz, launch_stream)`
//! against the SRDatalog (Sun et al., arXiv 2604.20073) WCOJ
//! shape adapted to a single triangle rule
//!
//!   tri(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z)
//!
//! Inputs (caller-supplied — no physical layout construction in
//! this provider-only test):
//!   * e_xy: 2-column u32 CudaBuffer, sorted+deduped by (X, Y)
//!   * e_yz: 2-column u32 CudaBuffer, sorted+deduped by (Y, Z)
//!   * e_xz: 2-column u32 CudaBuffer, sorted+deduped by (X, Z)
//!
//! Execution boundaries:
//!   * U32 only, two-column relations only.
//!   * Count → scan → materialize, deterministic output sorted (X, Y, Z).
//!   * Strict `LaunchRecorder` discipline: every input read,
//!     scratch write, and output write recorded; runtime-backed
//!     manager required.
//!   * No executor integration, no planner dispatch, no recursion,
//!     no Symbol/u64.
//!
//! Test surface:
//!   1. correctness with multiple triangles (matches CPU reference)
//!   2. empty result (any of the three relations empty, or no
//!      mutual triangle exists)
//!   3. no false positives from partial wedges (X-Y and Y-Z but
//!      no X-Z closing edge)
//!   4. deterministic lexicographic output across two runs
//!   5. duplicate-input policy (caller responsibility — kernel
//!      assumes deduped)
//!   6. runtime-backed drop+reuse safety
//!   7. legacy-manager rejection (no runtime → kernel error)

use std::collections::BTreeSet;
use std::sync::Arc;

use cudarc::driver::sys;
use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamId, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::CudaBuffer;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

// ---------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------

struct DiscardSink;
impl LoggingSink for DiscardSink {
    fn emit(&self, _record: LogRecord) -> Result<(), SinkError> {
        Ok(())
    }
}

#[allow(dead_code)] // device/runtime kept alive via Arc clones; required for cross-stream tracking lifetimes.
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

/// Upload a host-side `Vec<(u32, u32)>` to a 2-column u32
/// `CudaBuffer`. Caller guarantees the input is already sorted
/// and deduplicated in the column order required by the kernel.
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

/// Download a 3-column u32 result `CudaBuffer` to a host
/// `Vec<(u32, u32, u32)>` for assertion.
fn download_triples(buf: &CudaBuffer) -> Vec<(u32, u32, u32)> {
    let n = buf.num_rows() as usize;
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
    out
}

/// CPU reference: enumerate all (X, Y, Z) such that
/// (X, Y) ∈ e_xy ∧ (Y, Z) ∈ e_yz ∧ (X, Z) ∈ e_xz.
/// Output sorted lex (X, Y, Z) and deduplicated.
fn cpu_triangle_reference(
    e_xy: &[(u32, u32)],
    e_yz: &[(u32, u32)],
    e_xz: &[(u32, u32)],
) -> Vec<(u32, u32, u32)> {
    let yz_set: BTreeSet<(u32, u32)> = e_yz.iter().copied().collect();
    let xz_set: BTreeSet<(u32, u32)> = e_xz.iter().copied().collect();
    let mut out: BTreeSet<(u32, u32, u32)> = BTreeSet::new();
    for &(x, y) in e_xy {
        // Enumerate Z from yz where (y, z) ∈ yz, then check xz.
        for &(y2, z) in e_yz {
            if y2 != y {
                continue;
            }
            if xz_set.contains(&(x, z)) && yz_set.contains(&(y, z)) {
                out.insert((x, y, z));
            }
        }
    }
    out.into_iter().collect()
}

unsafe fn drain_raw(stream: sys::CUstream) {
    let res = sys::cuStreamSynchronize(stream);
    assert_eq!(res, sys::cudaError_enum::CUDA_SUCCESS);
}

// ---------------------------------------------------------------
// Tests
// ---------------------------------------------------------------

#[test]
fn wcoj_triangle_u32_matches_cpu_reference_with_multiple_triangles() {
    // Directed graph fragments. We pick three edge sets that form
    // five directed triangles tri(X,Y,Z) :- e_xy(X,Y), e_yz(Y,Z), e_xz(X,Z):
    //
    // Triangles expected (hand-derived):
    //   K_4 component on {1,2,3,4}: every pair has all three relations
    //   present at the corresponding (col0, col1) — gives
    //     (1,2,3), (1,2,4), (1,3,4), (2,3,4)
    //   Smaller cycle on {5,6,7}: (5,6,7).
    //
    // Sanity: cpu_triangle_reference is the source of truth; we
    // also assert the row count (5).
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let e_xy: Vec<(u32, u32)> = vec![
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
    // For tri(X,Y,Z), e_yz must contain every (Y, Z) of the
    // triangles above. From the K_4: Y ∈ {2,3,4}, Z must satisfy
    // both e_yz(Y,Z) and e_xz(X,Z). We use the same edge set
    // labelled differently (each triangle requires (Y, Z) ∈ e_yz).
    let e_yz: Vec<(u32, u32)> = vec![(2, 3), (2, 4), (3, 4), (6, 7)];
    // e_xz must close each triangle's (X, Z) pair.
    let e_xz: Vec<(u32, u32)> = vec![(1, 3), (1, 4), (2, 4), (3, 4), (5, 7)];

    let buf_xy = upload_binary_u32(&fix.memory, &e_xy);
    let buf_yz = upload_binary_u32(&fix.memory, &e_yz);
    let buf_xz = upload_binary_u32(&fix.memory, &e_xz);

    let launch_stream = fix.pool.acquire().expect("acquire launch_stream");
    let result = fix
        .provider
        .wcoj_triangle_u32_recorded(&buf_xy, &buf_yz, &buf_xz, launch_stream)
        .expect("wcoj_triangle_u32_recorded");

    let host_result = download_triples(&result);
    let expected = cpu_triangle_reference(&e_xy, &e_yz, &e_xz);
    assert_eq!(
        host_result, expected,
        "GPU triangle output must match CPU reference"
    );
    // Hand-verified count for the fixture above.
    assert_eq!(host_result.len(), 5);
}

#[test]
fn wcoj_triangle_u32_empty_inputs_produce_empty_output() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let empty: Vec<(u32, u32)> = Vec::new();
    let buf_xy = upload_binary_u32(&fix.memory, &empty);
    let buf_yz = upload_binary_u32(&fix.memory, &empty);
    let buf_xz = upload_binary_u32(&fix.memory, &empty);
    let launch_stream = fix.pool.acquire().expect("acquire launch_stream");
    let result = fix
        .provider
        .wcoj_triangle_u32_recorded(&buf_xy, &buf_yz, &buf_xz, launch_stream)
        .expect("empty inputs must succeed");
    assert_eq!(result.num_rows(), 0);
    assert_eq!(result.arity(), 3);
}

#[test]
fn wcoj_triangle_u32_no_false_positives_on_open_wedge() {
    // Open wedge: (X=1, Y=2) ∈ e_xy, (Y=2, Z=3) ∈ e_yz, but the
    // closing edge (X=1, Z=3) is NOT in e_xz. Must NOT emit
    // (1, 2, 3).
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let e_xy: Vec<(u32, u32)> = vec![(1, 2)];
    let e_yz: Vec<(u32, u32)> = vec![(2, 3)];
    let e_xz: Vec<(u32, u32)> = vec![(2, 3)]; // unrelated edge; (1,3) absent
    let buf_xy = upload_binary_u32(&fix.memory, &e_xy);
    let buf_yz = upload_binary_u32(&fix.memory, &e_yz);
    let buf_xz = upload_binary_u32(&fix.memory, &e_xz);
    let launch_stream = fix.pool.acquire().expect("launch_stream");
    let result = fix
        .provider
        .wcoj_triangle_u32_recorded(&buf_xy, &buf_yz, &buf_xz, launch_stream)
        .expect("must succeed");
    let host_result = download_triples(&result);
    assert!(
        host_result.is_empty(),
        "open wedge must not produce triangle, got {:?}",
        host_result
    );
}

#[test]
fn wcoj_triangle_u32_output_is_lex_sorted_and_deterministic() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    // Arrange triangles with intentionally interleaved key sets
    // so a non-deterministic kernel could emit in different
    // orders; we require deterministic lex (X, Y, Z) sort.
    let e_xy: Vec<(u32, u32)> = vec![(1, 2), (1, 3), (2, 3), (5, 6), (5, 7), (6, 7)];
    let e_yz: Vec<(u32, u32)> = vec![(2, 3), (3, 4), (6, 7)];
    let e_xz: Vec<(u32, u32)> = vec![(1, 3), (1, 4), (2, 3), (2, 4), (5, 7)];
    let buf_xy = upload_binary_u32(&fix.memory, &e_xy);
    let buf_yz = upload_binary_u32(&fix.memory, &e_yz);
    let buf_xz = upload_binary_u32(&fix.memory, &e_xz);

    let stream_a = fix.pool.acquire().expect("stream_a");
    let result_a = fix
        .provider
        .wcoj_triangle_u32_recorded(&buf_xy, &buf_yz, &buf_xz, stream_a)
        .expect("run a");
    let host_a = download_triples(&result_a);

    let stream_b = fix.pool.acquire().expect("stream_b");
    let result_b = fix
        .provider
        .wcoj_triangle_u32_recorded(&buf_xy, &buf_yz, &buf_xz, stream_b)
        .expect("run b");
    let host_b = download_triples(&result_b);

    assert_eq!(host_a, host_b, "two runs of same input must be identical");
    let mut sorted = host_a.clone();
    sorted.sort();
    assert_eq!(host_a, sorted, "output must already be lex-sorted");
}

#[test]
fn wcoj_triangle_u32_duplicate_input_policy_documented() {
    // The kernel assumes inputs are already deduped. This test
    // documents the contract: feeding a duplicated row in any
    // input means the kernel may either emit duplicate output
    // rows or undefined behavior — caller responsibility.
    //
    // We assert the kernel does NOT panic / corrupt; result row
    // count may exceed the deduped reference. Locks the
    // "garbage-in → defined-failure-mode garbage-out" contract.
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    // Duplicated (2, 3) in e_yz. Sorted but not deduped — caller
    // contract violation.
    let e_xy: Vec<(u32, u32)> = vec![(1, 2)];
    let e_yz: Vec<(u32, u32)> = vec![(2, 3), (2, 3)];
    let e_xz: Vec<(u32, u32)> = vec![(1, 3)];
    let buf_xy = upload_binary_u32(&fix.memory, &e_xy);
    let buf_yz = upload_binary_u32(&fix.memory, &e_yz);
    let buf_xz = upload_binary_u32(&fix.memory, &e_xz);
    let launch_stream = fix.pool.acquire().expect("launch_stream");
    let result = fix
        .provider
        .wcoj_triangle_u32_recorded(&buf_xy, &buf_yz, &buf_xz, launch_stream)
        .expect("duplicate input must not crash; behavior is caller's responsibility");
    let host_result = download_triples(&result);
    // We don't assert exact rows — the contract is only that the
    // kernel returns SOMETHING without UB. Check that every row
    // it emits is a real triangle (no spurious rows).
    let valid_set: BTreeSet<(u32, u32, u32)> = cpu_triangle_reference(
        &e_xy,
        // Pass deduped yz to reference for the validity check.
        &[(2, 3)],
        &e_xz,
    )
    .into_iter()
    .collect();
    for row in &host_result {
        assert!(
            valid_set.contains(row),
            "kernel emitted row {:?} that is not a valid triangle in deduped semantics",
            row
        );
    }
}

#[test]
fn wcoj_triangle_u32_survives_drop_and_reuse_of_input_buffers() {
    // Mirrors the drop+reuse safety pattern from
    // test_provider_launch_recorder.rs: build inputs, run kernel,
    // drop the input CudaBuffers without host sync, allocate
    // fresh buffers (which may reuse the same device addresses),
    // then drain. The recorded launch must have ordered the
    // input reads before the deallocations so the new buffers
    // are not corrupted.
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let e_xy: Vec<(u32, u32)> = vec![(1, 2)];
    let e_yz: Vec<(u32, u32)> = vec![(2, 3)];
    let e_xz: Vec<(u32, u32)> = vec![(1, 3)];

    let launch_stream = fix.pool.acquire().expect("launch_stream");
    let launch_handle = fix.pool.resolve(launch_stream).expect("resolve");

    // Run + drop without sync.
    {
        let buf_xy = upload_binary_u32(&fix.memory, &e_xy);
        let buf_yz = upload_binary_u32(&fix.memory, &e_yz);
        let buf_xz = upload_binary_u32(&fix.memory, &e_xz);
        let result = fix
            .provider
            .wcoj_triangle_u32_recorded(&buf_xy, &buf_yz, &buf_xz, launch_stream)
            .expect("kernel");
        // Read before drop to confirm correctness on this run.
        let host_result = download_triples(&result);
        assert_eq!(host_result, vec![(1, 2, 3)]);
        // buf_xy, buf_yz, buf_xz, result all drop here.
    }

    // Allocate a different-shape relation to provoke address
    // reuse. If the recorded launch failed to chain its input
    // reads ahead of the dealloc, this allocation could land
    // before the kernel's read of the prior buffer finishes.
    let _reuse: Vec<_> = (0..16u32)
        .map(|i| upload_binary_u32(&fix.memory, &[(i, i + 1)]))
        .collect();
    unsafe {
        drain_raw(launch_handle.cu_stream());
    }
    // No assertion needed — survival without abort is the
    // contract. UB on that path would have shown as a memory
    // checker failure or silent wrong answer in the prior run.
}

#[test]
fn wcoj_triangle_u32_legacy_manager_rejected() {
    // The kernel must reject a legacy GpuMemoryManager (no
    // runtime attached) with a clear error rather than running
    // unsafe.
    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let memory = Arc::new(GpuMemoryManager::new(
        Arc::clone(&device),
        MemoryBudget::with_limit(16 * 1024 * 1024),
    ));
    let provider = CudaKernelProvider::new(Arc::clone(&device), Arc::clone(&memory))
        .expect("legacy provider construction");
    let e: Vec<(u32, u32)> = vec![(1, 2)];
    let buf_xy = upload_binary_u32(&memory, &e);
    let buf_yz = upload_binary_u32(&memory, &e);
    let buf_xz = upload_binary_u32(&memory, &e);
    // No runtime → no real launch_stream. We use DEFAULT here;
    // the validation should fire on the manager check, not on
    // stream resolution.
    let result = provider.wcoj_triangle_u32_recorded(&buf_xy, &buf_yz, &buf_xz, StreamId::DEFAULT);
    let err = match result {
        Ok(_) => panic!("legacy manager must be rejected, but kernel returned Ok"),
        Err(e) => e,
    };
    let msg = format!("{:?}", err);
    assert!(
        msg.contains("runtime") || msg.contains("with_runtime"),
        "error must mention runtime requirement, got: {}",
        msg
    );
}

#[test]
fn wcoj_triangle_u32_no_count_vector_d2h_under_strict_gate() {
    // Locks the device-scan transfer contract: with the strict
    // deterministic-D2H gate enabled, `wcoj_triangle_u32_recorded`
    // must succeed and trip zero violations. The previous host-scan
    // path used a raw `cuMemcpyDtoH_v2` over the count vector
    // that bypassed the gate's chokepoints. The device-scan path
    // replaces that with a recorded on-stream prefix-sum +
    // a single 4-byte `dtoh_scalar_untracked` of the inclusive
    // total — the latter is the sanctioned metadata-read path
    // the gate explicitly whitelists.
    //
    // Future regressions that route a column-sized D2H back
    // through `download_column_*` or `dtoh_sync_copy_into_tracked`
    // will trip the gate and fail this test.
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    // Same fixture shape as the multi-triangle correctness test
    // so the kernel does meaningful count + scan + materialize
    // work, not the empty-input early return.
    let e_xy: Vec<(u32, u32)> = vec![
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
    let e_yz: Vec<(u32, u32)> = vec![(2, 3), (2, 4), (3, 4), (6, 7)];
    let e_xz: Vec<(u32, u32)> = vec![(1, 3), (1, 4), (2, 4), (3, 4), (5, 7)];

    let buf_xy = upload_binary_u32(&fix.memory, &e_xy);
    let buf_yz = upload_binary_u32(&fix.memory, &e_yz);
    let buf_xz = upload_binary_u32(&fix.memory, &e_xz);

    fix.provider.reset_deterministic_d2h_violations();
    fix.provider.enable_strict_deterministic_d2h();
    let launch_stream = fix.pool.acquire().expect("launch_stream");
    let result = fix
        .provider
        .wcoj_triangle_u32_recorded(&buf_xy, &buf_yz, &buf_xz, launch_stream);
    fix.provider.disable_strict_deterministic_d2h();

    let buf = result.expect(
        "wcoj_triangle_u32_recorded must succeed under strict deterministic-D2H gate \
         (the previous count-vector D2H path is gone)",
    );
    assert_eq!(buf.num_rows() as usize, 5, "expected 5 triangles");
    let violations = fix.provider.deterministic_d2h_violation_count();
    assert_eq!(
        violations, 0,
        "WCOJ device-scan path must not trigger any deterministic-D2H gate violations; got {}",
        violations
    );
}
