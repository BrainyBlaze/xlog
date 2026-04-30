// crates/xlog-cuda/tests/test_wcoj_triangle_u64.rs
//! Tests for the v0.6.2 GPU 3-way WCOJ triangle kernel — u64 variant.
//!
//! Locks the provider entry
//! `CudaKernelProvider::wcoj_triangle_u64_recorded(e_xy, e_yz, e_xz, launch_stream)`
//! against the same SRDatalog two-phase contract as the u32 path,
//! widened to 64-bit keys via parallel kernels (`wcoj_triangle_count_u64`,
//! `wcoj_triangle_materialize_u64`) and the shared `wcoj_compute_total`
//! reducer (counters stay u32 since they're bounded by `u32::MAX`).
//!
//! Hard scope (commit 2 of 3):
//!   * Tests provider entry only — no AST/RIR dispatch (commit 3).
//!   * Caller-supplied sorted+deduped inputs (layout tests cover that).
//!   * No mixed-width admission — all three relations must be U64.

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

/// Upload `Vec<(u64, u64)>` to a 2-column U64 [`CudaBuffer`].
/// Caller guarantees the input is already sorted + deduplicated.
fn upload_binary_u64(memory: &Arc<GpuMemoryManager>, rows: &[(u64, u64)]) -> CudaBuffer {
    let n = rows.len() as u32;
    let col0_host: Vec<u64> = rows.iter().map(|(a, _)| *a).collect();
    let col1_host: Vec<u64> = rows.iter().map(|(_, b)| *b).collect();
    let bytes_per_col = (n as usize) * std::mem::size_of::<u64>();
    let mut col0 = memory.alloc::<u8>(bytes_per_col).expect("alloc col0");
    let mut col1 = memory.alloc::<u8>(bytes_per_col).expect("alloc col1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let device = memory.device().inner();
    if n > 0 {
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

/// Download a 3-column U64 result `CudaBuffer` to host triples.
fn download_triples_u64(buf: &CudaBuffer) -> Vec<(u64, u64, u64)> {
    let n = buf.num_rows() as usize;
    if n == 0 {
        return Vec::new();
    }
    assert_eq!(buf.arity(), 3, "expected 3-column triangle output");
    let mut col0_bytes = vec![0u8; n * 8];
    let mut col1_bytes = vec![0u8; n * 8];
    let mut col2_bytes = vec![0u8; n * 8];
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
    let mut out: Vec<(u64, u64, u64)> = Vec::with_capacity(n);
    for i in 0..n {
        let x = u64::from_le_bytes(col0_bytes[i * 8..i * 8 + 8].try_into().unwrap());
        let y = u64::from_le_bytes(col1_bytes[i * 8..i * 8 + 8].try_into().unwrap());
        let z = u64::from_le_bytes(col2_bytes[i * 8..i * 8 + 8].try_into().unwrap());
        out.push((x, y, z));
    }
    out
}

fn cpu_triangle_reference_u64(
    e_xy: &[(u64, u64)],
    e_yz: &[(u64, u64)],
    e_xz: &[(u64, u64)],
) -> Vec<(u64, u64, u64)> {
    let yz_set: BTreeSet<(u64, u64)> = e_yz.iter().copied().collect();
    let xz_set: BTreeSet<(u64, u64)> = e_xz.iter().copied().collect();
    let mut out: BTreeSet<(u64, u64, u64)> = BTreeSet::new();
    for &(x, y) in e_xy {
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

// ---------------------------------------------------------------
// Tests
// ---------------------------------------------------------------

#[test]
fn wcoj_triangle_u64_matches_cpu_reference_with_multiple_triangles() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    // Same triangle topology as the u32 multi-triangle test, but
    // every key shifted into the upper half of the u64 range.
    // Locks that the kernel actually reads hi-half bits.
    let big = (u32::MAX as u64) + 1;
    let e_xy: Vec<(u64, u64)> = vec![
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
    let e_yz: Vec<(u64, u64)> = vec![
        (big + 2, big + 3),
        (big + 2, big + 4),
        (big + 3, big + 4),
        (big + 6, big + 7),
    ];
    let e_xz: Vec<(u64, u64)> = vec![
        (big + 1, big + 3),
        (big + 1, big + 4),
        (big + 2, big + 4),
        (big + 3, big + 4),
        (big + 5, big + 7),
    ];
    let buf_xy = upload_binary_u64(&fix.memory, &e_xy);
    let buf_yz = upload_binary_u64(&fix.memory, &e_yz);
    let buf_xz = upload_binary_u64(&fix.memory, &e_xz);
    let stream = fix.pool.acquire().expect("stream");
    let result = fix
        .provider
        .wcoj_triangle_u64_recorded(&buf_xy, &buf_yz, &buf_xz, stream)
        .expect("triangle u64");
    // Output schema must preserve U64 per column.
    assert_eq!(result.schema.column_type(0), Some(ScalarType::U64));
    assert_eq!(result.schema.column_type(1), Some(ScalarType::U64));
    assert_eq!(result.schema.column_type(2), Some(ScalarType::U64));
    let host = download_triples_u64(&result);
    let expected = cpu_triangle_reference_u64(&e_xy, &e_yz, &e_xz);
    assert_eq!(host, expected);
    assert_eq!(host.len(), 5, "expected 5 triangles on this fixture");
}

#[test]
fn wcoj_triangle_u64_empty_inputs_produce_empty_output() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let buf_xy = upload_binary_u64(&fix.memory, &[]);
    let buf_yz = upload_binary_u64(&fix.memory, &[(1, 2)]);
    let buf_xz = upload_binary_u64(&fix.memory, &[(3, 4)]);
    let stream = fix.pool.acquire().expect("stream");
    let result = fix
        .provider
        .wcoj_triangle_u64_recorded(&buf_xy, &buf_yz, &buf_xz, stream)
        .expect("triangle u64 empty");
    assert_eq!(download_triples_u64(&result), Vec::<(u64, u64, u64)>::new());
}

#[test]
fn wcoj_triangle_u64_no_false_positives_on_open_wedge() {
    // X→Y and Y→Z exist but no closing X→Z edge → 0 triangles.
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let big = (u32::MAX as u64) + 1;
    let buf_xy = upload_binary_u64(&fix.memory, &[(big + 1, big + 2)]);
    let buf_yz = upload_binary_u64(&fix.memory, &[(big + 2, big + 3)]);
    let buf_xz = upload_binary_u64(&fix.memory, &[(big + 7, big + 8)]); // unrelated
    let stream = fix.pool.acquire().expect("stream");
    let result = fix
        .provider
        .wcoj_triangle_u64_recorded(&buf_xy, &buf_yz, &buf_xz, stream)
        .expect("triangle open wedge u64");
    assert_eq!(download_triples_u64(&result).len(), 0);
}

#[test]
fn wcoj_triangle_u64_legacy_manager_rejected() {
    // Legacy GpuMemoryManager (no runtime) must yield a clear
    // error rather than run unsafe. Construct via
    // `CudaKernelProvider::new`, which accepts non-runtime
    // managers; the entry's runtime check fires inside the call.
    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        eprintln!("Skipping: no CUDA device");
        return;
    };
    let memory = Arc::new(GpuMemoryManager::new(
        Arc::clone(&device),
        MemoryBudget::with_limit(16 * 1024 * 1024),
    ));
    let provider = CudaKernelProvider::new(Arc::clone(&device), Arc::clone(&memory))
        .expect("legacy provider construction");
    let buf_xy = upload_binary_u64(&memory, &[(1, 2)]);
    let buf_yz = upload_binary_u64(&memory, &[(2, 3)]);
    let buf_xz = upload_binary_u64(&memory, &[(1, 3)]);
    let result = provider.wcoj_triangle_u64_recorded(&buf_xy, &buf_yz, &buf_xz, StreamId::DEFAULT);
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
fn wcoj_triangle_u64_no_count_vector_d2h_under_strict_gate() {
    // U64 sibling of the U32 device-scan gate test. Same
    // contract: under the strict deterministic-D2H gate,
    // `wcoj_triangle_u64_recorded` must succeed (the only
    // device→host path is `dtoh_scalar_untracked` for the
    // total scalar, which the gate explicitly whitelists)
    // and trip zero violations. Any future regression that
    // routes a column-sized D2H back through `download_column_*`
    // or `dtoh_sync_copy_into_tracked` will trip the gate
    // and fail this test.
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    // Mirror the multi-triangle correctness fixture, shifted
    // into hi-half u64 space so the kernel does real count +
    // scan + materialize work rather than the empty early-out.
    let big = (u32::MAX as u64) + 1;
    let e_xy: Vec<(u64, u64)> = vec![
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
    let e_yz: Vec<(u64, u64)> = vec![
        (big + 2, big + 3),
        (big + 2, big + 4),
        (big + 3, big + 4),
        (big + 6, big + 7),
    ];
    let e_xz: Vec<(u64, u64)> = vec![
        (big + 1, big + 3),
        (big + 1, big + 4),
        (big + 2, big + 4),
        (big + 3, big + 4),
        (big + 5, big + 7),
    ];

    let buf_xy = upload_binary_u64(&fix.memory, &e_xy);
    let buf_yz = upload_binary_u64(&fix.memory, &e_yz);
    let buf_xz = upload_binary_u64(&fix.memory, &e_xz);

    fix.provider.reset_deterministic_d2h_violations();
    fix.provider.enable_strict_deterministic_d2h();
    let launch_stream = fix.pool.acquire().expect("launch_stream");
    let result = fix
        .provider
        .wcoj_triangle_u64_recorded(&buf_xy, &buf_yz, &buf_xz, launch_stream);
    fix.provider.disable_strict_deterministic_d2h();

    let buf = result.expect(
        "wcoj_triangle_u64_recorded must succeed under strict deterministic-D2H gate \
         (only the dtoh_scalar_untracked total is allowed)",
    );
    assert_eq!(buf.num_rows() as usize, 5, "expected 5 triangles");
    let violations = fix.provider.deterministic_d2h_violation_count();
    assert_eq!(
        violations, 0,
        "WCOJ U64 device-scan path must not trigger any deterministic-D2H gate violations; got {}",
        violations
    );
}

#[test]
fn wcoj_triangle_u64_rejects_mixed_width_inputs() {
    // Negative shape: caller passes a U32-typed buffer to the U64
    // entry. The provider must reject rather than silently
    // mis-interpret bits. Cross-relation type compatibility is
    // the planner's job upstream; this is the provider-level
    // type guard that catches misuse from a hand-written caller.
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let big = (u32::MAX as u64) + 1;
    let buf_xy = upload_binary_u64(&fix.memory, &[(big + 1, big + 2)]);
    let buf_yz = upload_binary_u64(&fix.memory, &[(big + 2, big + 3)]);
    // U32-typed buffer in slot e_xz.
    let n = 1u32;
    let mut col0 = fix.memory.alloc::<u8>(4).expect("alloc col0");
    let mut col1 = fix.memory.alloc::<u8>(4).expect("alloc col1");
    let mut d_num_rows = fix.memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let device = fix.memory.device().inner();
    device
        .htod_sync_copy_into(&1u32.to_le_bytes(), &mut col0)
        .expect("htod col0");
    device
        .htod_sync_copy_into(&3u32.to_le_bytes(), &mut col1)
        .expect("htod col1");
    device
        .htod_sync_copy_into(&[n], &mut d_num_rows)
        .expect("htod d_num_rows");
    let u32_schema = Schema::new(vec![
        ("col0".to_string(), ScalarType::U32),
        ("col1".to_string(), ScalarType::U32),
    ]);
    let buf_xz_u32 = CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into()],
        n as u64,
        d_num_rows,
        u32_schema,
        n,
    );
    let stream = fix.pool.acquire().expect("stream");
    let result = fix
        .provider
        .wcoj_triangle_u64_recorded(&buf_xy, &buf_yz, &buf_xz_u32, stream);
    assert!(
        result.is_err(),
        "wcoj_triangle_u64_recorded must reject U32-typed input slot \
         (got Ok({:?}))",
        result.as_ref().ok().map(|b| b.arity())
    );
}
