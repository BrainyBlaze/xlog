// crates/xlog-cuda/tests/test_wcoj_layout_u32.rs
//! Tests for the v0.6.2 GPU WCOJ sorted-layout construction
//! kernel (u32, slice 1).
//!
//! Locks the provider entry
//! `CudaKernelProvider::wcoj_layout_u32_recorded(input, launch_stream)`
//! against the contract:
//!
//!   * Input: ordinary 2-column u32 [`CudaBuffer`].
//!   * Output: 2-column u32 [`CudaBuffer`], sorted lexicographically
//!     by `(col0, col1)` and deduplicated. The output is suitable
//!     for direct consumption by `wcoj_triangle_u32_recorded`.
//!   * Reuses existing recorded primitives end-to-end (typed sort
//!     + full-row dedup) — no new algorithm.
//!   * Strict `LaunchRecorder` discipline inherited from the
//!     composed primitives; runtime-backed manager required.
//!
//! Test surface:
//!   1. unsorted input with no duplicates is sorted lex
//!   2. duplicates are removed
//!   3. empty input yields empty output
//!   4. already-sorted+deduped input round-trips unchanged
//!   5. drop+reuse safety (recorder ordering chained end-to-end)
//!   6. legacy-manager rejection (no runtime → kernel error)
//!   7. integration: feeding three unsorted buffers through layout
//!      construction into `wcoj_triangle_u32_recorded` produces the
//!      same triangles as a CPU oracle.

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
// Shared helpers (mirror test_wcoj_triangle_u32.rs conventions)
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

/// Upload a host-side `Vec<(u32, u32)>` to a 2-column u32
/// `CudaBuffer`. Caller-supplied row order is preserved (no sort).
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

/// Download a 2-column u32 `CudaBuffer` to a host
/// `Vec<(u32, u32)>` in row order.
///
/// `CudaBuffer::num_rows()` returns the allocation `row_cap` (which
/// may exceed the logical row count for primitives that compact in
/// place — notably `dedup_full_row_recorded`). We use the cached
/// host count when available, otherwise read the logical count
/// from the device-resident `d_num_rows` slot via a 4-byte D2H.
fn download_pairs(buf: &CudaBuffer) -> Vec<(u32, u32)> {
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
    assert_eq!(buf.arity(), 2, "expected 2-column layout output");
    let mut col0_bytes = vec![0u8; n * 4];
    let mut col1_bytes = vec![0u8; n * 4];
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
        assert_eq!(res0, sys::cudaError_enum::CUDA_SUCCESS);
        assert_eq!(res1, sys::cudaError_enum::CUDA_SUCCESS);
    }
    let mut out: Vec<(u32, u32)> = Vec::with_capacity(n);
    for i in 0..n {
        let a = u32::from_le_bytes(col0_bytes[i * 4..i * 4 + 4].try_into().unwrap());
        let b = u32::from_le_bytes(col1_bytes[i * 4..i * 4 + 4].try_into().unwrap());
        out.push((a, b));
    }
    out
}

/// CPU reference (sort + dedup).
fn cpu_sort_dedup(rows: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let set: BTreeSet<(u32, u32)> = rows.iter().copied().collect();
    set.into_iter().collect()
}

/// CPU triangle reference (matches the WCOJ test's reference).
fn cpu_triangle_reference(
    e_xy: &[(u32, u32)],
    e_yz: &[(u32, u32)],
    e_xz: &[(u32, u32)],
) -> Vec<(u32, u32, u32)> {
    let yz_set: BTreeSet<(u32, u32)> = e_yz.iter().copied().collect();
    let xz_set: BTreeSet<(u32, u32)> = e_xz.iter().copied().collect();
    let mut out: BTreeSet<(u32, u32, u32)> = BTreeSet::new();
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
    out.sort();
    out
}

fn sync_stream(fix: &RuntimeFixture, stream: StreamId) {
    fix.pool
        .resolve(stream)
        .expect("resolve stream")
        .synchronize()
        .expect("sync stream");
}

// ---------------------------------------------------------------
// Tests
// ---------------------------------------------------------------

#[test]
fn wcoj_layout_u32_sorts_unsorted_input_lex() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    // Out-of-order input, no duplicates.
    let input: Vec<(u32, u32)> = vec![(3, 1), (1, 5), (2, 0), (1, 3), (3, 0), (1, 4)];
    let buf = upload_binary_u32(&fix.memory, &input);
    let launch_stream = fix.pool.acquire().expect("launch_stream");
    let out = fix
        .provider
        .wcoj_layout_u32_recorded(&buf, launch_stream)
        .expect("layout construction must succeed");
    sync_stream(&fix, launch_stream);
    let host = download_pairs(&out);
    assert_eq!(host, cpu_sort_dedup(&input));
}

#[test]
fn wcoj_layout_u32_removes_duplicates() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    // Mix of duplicates: (1,2) appears 3x, (3,4) appears 2x,
    // (2,3) once.
    let input: Vec<(u32, u32)> = vec![(1, 2), (3, 4), (1, 2), (2, 3), (3, 4), (1, 2)];
    let buf = upload_binary_u32(&fix.memory, &input);
    let launch_stream = fix.pool.acquire().expect("launch_stream");
    let out = fix
        .provider
        .wcoj_layout_u32_recorded(&buf, launch_stream)
        .expect("must succeed");
    sync_stream(&fix, launch_stream);
    let host = download_pairs(&out);
    assert_eq!(host, vec![(1, 2), (2, 3), (3, 4)]);
}

#[test]
fn wcoj_layout_u32_empty_input_produces_empty_output() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let empty: Vec<(u32, u32)> = Vec::new();
    let buf = upload_binary_u32(&fix.memory, &empty);
    let launch_stream = fix.pool.acquire().expect("launch_stream");
    let out = fix
        .provider
        .wcoj_layout_u32_recorded(&buf, launch_stream)
        .expect("empty input must succeed");
    sync_stream(&fix, launch_stream);
    assert_eq!(out.num_rows(), 0);
    assert_eq!(out.arity(), 2);
}

#[test]
fn wcoj_layout_u32_already_sorted_deduped_round_trips() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    // Already sorted+deduped — output must equal input as a set
    // (and as a sorted vec, since input is already lex-sorted).
    let input: Vec<(u32, u32)> = vec![(1, 2), (2, 3), (3, 4), (5, 6)];
    let buf = upload_binary_u32(&fix.memory, &input);
    let launch_stream = fix.pool.acquire().expect("launch_stream");
    let out = fix
        .provider
        .wcoj_layout_u32_recorded(&buf, launch_stream)
        .expect("must succeed");
    sync_stream(&fix, launch_stream);
    let host = download_pairs(&out);
    assert_eq!(host, input);
}

#[test]
fn wcoj_layout_u32_survives_drop_and_reuse() {
    // Mirrors the WCOJ-triangle drop+reuse test. The recorded
    // primitives (sort + dedup) chain reads/writes via
    // LaunchRecorder, so dropping the input buffer immediately
    // after the call must not corrupt subsequent allocations
    // that may reuse the same device addresses.
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let launch_stream = fix.pool.acquire().expect("launch_stream");
    let launch_handle = fix.pool.resolve(launch_stream).expect("resolve");

    {
        let input: Vec<(u32, u32)> = vec![(2, 1), (1, 1), (1, 0)];
        let buf = upload_binary_u32(&fix.memory, &input);
        let out = fix
            .provider
            .wcoj_layout_u32_recorded(&buf, launch_stream)
            .expect("layout");
        sync_stream(&fix, launch_stream);
        let host = download_pairs(&out);
        assert_eq!(host, vec![(1, 0), (1, 1), (2, 1)]);
    }
    // Provoke address reuse with a batch of small allocs.
    let _reuse: Vec<_> = (0..16u32)
        .map(|i| upload_binary_u32(&fix.memory, &[(i, i + 1)]))
        .collect();
    unsafe {
        let res = sys::cuStreamSynchronize(launch_handle.cu_stream());
        assert_eq!(res, sys::cudaError_enum::CUDA_SUCCESS);
    }
}

#[test]
fn wcoj_layout_u32_legacy_manager_rejected() {
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
    let input: Vec<(u32, u32)> = vec![(1, 2)];
    let buf = upload_binary_u32(&memory, &input);
    let result = provider.wcoj_layout_u32_recorded(&buf, StreamId::DEFAULT);
    let err = match result {
        Ok(_) => panic!("legacy manager must be rejected, but layout returned Ok"),
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
fn wcoj_layout_then_triangle_u32_matches_cpu_oracle() {
    // End-to-end: feed three UNSORTED+DUPLICATED buffers through
    // wcoj_layout_u32_recorded to produce sorted+deduped layouts,
    // then through wcoj_triangle_u32_recorded. Compare against
    // the CPU triangle oracle on the deduped input.
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    // K_4-on-{1,2,3,4} + small triangle-on-{5,6,7}, shuffled +
    // duplicated.
    let raw_xy: Vec<(u32, u32)> = vec![
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
    let raw_yz: Vec<(u32, u32)> = vec![
        (3, 4),
        (2, 3),
        (6, 7),
        (2, 4),
        (3, 4), // dup
    ];
    let raw_xz: Vec<(u32, u32)> = vec![
        (1, 4),
        (2, 4),
        (3, 4),
        (1, 3),
        (5, 7),
        (1, 4), // dup
    ];

    let buf_xy_raw = upload_binary_u32(&fix.memory, &raw_xy);
    let buf_yz_raw = upload_binary_u32(&fix.memory, &raw_yz);
    let buf_xz_raw = upload_binary_u32(&fix.memory, &raw_xz);

    let stream = fix.pool.acquire().expect("layout stream");
    let buf_xy = fix
        .provider
        .wcoj_layout_u32_recorded(&buf_xy_raw, stream)
        .expect("layout xy");
    let buf_yz = fix
        .provider
        .wcoj_layout_u32_recorded(&buf_yz_raw, stream)
        .expect("layout yz");
    let buf_xz = fix
        .provider
        .wcoj_layout_u32_recorded(&buf_xz_raw, stream)
        .expect("layout xz");
    sync_stream(&fix, stream);

    let tri_stream = fix.pool.acquire().expect("triangle stream");
    let result = fix
        .provider
        .wcoj_triangle_u32_recorded(&buf_xy, &buf_yz, &buf_xz, tri_stream)
        .expect("triangle");
    sync_stream(&fix, tri_stream);
    let host_result = download_triples(&result);

    // CPU oracle on deduped input.
    let dedup_xy = cpu_sort_dedup(&raw_xy);
    let dedup_yz = cpu_sort_dedup(&raw_yz);
    let dedup_xz = cpu_sort_dedup(&raw_xz);
    let expected = cpu_triangle_reference(&dedup_xy, &dedup_yz, &dedup_xz);

    assert_eq!(
        host_result, expected,
        "layout → triangle pipeline must match CPU oracle on deduped inputs"
    );
}

// ---------------------------------------------------------------
// Symbol support — Symbol shares u32's 4-byte physical layout, so
// the kernel & layout primitives accept Symbol columns unchanged.
// ---------------------------------------------------------------

/// Build a 2-column Symbol [`CudaBuffer`] from interned-id pairs.
/// The on-device byte representation is identical to the U32
/// helper above; only the schema differs.
fn upload_binary_symbol(memory: &Arc<GpuMemoryManager>, rows: &[(u32, u32)]) -> CudaBuffer {
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
        ("col0".to_string(), ScalarType::Symbol),
        ("col1".to_string(), ScalarType::Symbol),
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
fn wcoj_layout_symbol_round_trips_with_correct_schema() {
    // Sort+dedup an unsorted Symbol-typed binary relation. Output
    // must remain Symbol-typed (not silently widened to U32) and
    // contain the same row set as the U32 path on identical bits.
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let input: Vec<(u32, u32)> = vec![
        (3, 1),
        (1, 5),
        (1, 5), // dup
        (2, 0),
        (1, 3),
        (3, 0),
        (3, 1), // dup
    ];
    let buf = upload_binary_symbol(&fix.memory, &input);
    let launch_stream = fix.pool.acquire().expect("launch_stream");
    let out = fix
        .provider
        .wcoj_layout_u32_recorded(&buf, launch_stream)
        .expect("layout must accept Symbol input");
    sync_stream(&fix, launch_stream);
    // Schema preservation: output columns must remain Symbol.
    assert_eq!(
        out.schema.column_type(0),
        Some(ScalarType::Symbol),
        "Symbol input must produce Symbol output column 0"
    );
    assert_eq!(
        out.schema.column_type(1),
        Some(ScalarType::Symbol),
        "Symbol input must produce Symbol output column 1"
    );
    let host = download_pairs(&out);
    assert_eq!(host, cpu_sort_dedup(&input));
}

#[test]
fn wcoj_triangle_symbol_matches_cpu_reference() {
    // Same K_4 + small triangle fixture as the U32 multi-triangle
    // test, but with Symbol schemas. Locks that the kernel reads
    // bits unchanged AND the output schema preserves Symbol.
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
    let e_yz: Vec<(u32, u32)> = vec![(2, 3), (2, 4), (3, 4), (6, 7)];
    let e_xz: Vec<(u32, u32)> = vec![(1, 3), (1, 4), (2, 4), (3, 4), (5, 7)];
    let buf_xy = upload_binary_symbol(&fix.memory, &e_xy);
    let buf_yz = upload_binary_symbol(&fix.memory, &e_yz);
    let buf_xz = upload_binary_symbol(&fix.memory, &e_xz);
    let stream = fix.pool.acquire().expect("stream");
    let result = fix
        .provider
        .wcoj_triangle_u32_recorded(&buf_xy, &buf_yz, &buf_xz, stream)
        .expect("triangle must accept Symbol inputs");
    sync_stream(&fix, stream);
    // Output schema: each column must carry its input scalar
    // type (Symbol here, since all inputs are Symbol).
    assert_eq!(result.schema.column_type(0), Some(ScalarType::Symbol));
    assert_eq!(result.schema.column_type(1), Some(ScalarType::Symbol));
    assert_eq!(result.schema.column_type(2), Some(ScalarType::Symbol));
    let host = download_triples(&result);
    let expected = cpu_triangle_reference(&e_xy, &e_yz, &e_xz);
    assert_eq!(host, expected);
    assert_eq!(host.len(), 5, "expected 5 triangles on this fixture");
}
