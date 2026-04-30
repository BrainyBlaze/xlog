// crates/xlog-cuda/tests/test_wcoj_layout_u64.rs
//! Tests for the v0.6.2 GPU WCOJ sorted-layout construction —
//! u64 variant.
//!
//! Locks the provider entry
//! `CudaKernelProvider::wcoj_layout_u64_recorded(input, launch_stream)`
//! against the same contract as the u32 path, widened to 64-bit
//! keys. Internally delegates to `dedup_full_row_recorded` (which
//! gained U64 admission in commit 1).
//!
//! Hard scope (commit 2 of 3):
//!   * Layout-only — feeds into `wcoj_triangle_u64_recorded`
//!     for the integration test below.
//!   * No AST/RIR dispatch (commit 3).

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

/// Layout outputs compact in place via dedup, so `num_rows()`
/// returns the row_cap (allocation), not the logical row count.
/// Mirror the WCOJ test downloader and prefer the cached host
/// count, falling back to a 4-byte D2H of `d_num_rows`.
fn logical_row_count(buf: &CudaBuffer) -> usize {
    if let Some(c) = buf.cached_row_count() {
        return c as usize;
    }
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

fn download_pairs_u64(buf: &CudaBuffer) -> Vec<(u64, u64)> {
    let n = logical_row_count(buf);
    if n == 0 {
        return Vec::new();
    }
    assert_eq!(buf.arity(), 2, "expected 2-column layout output");
    let mut col0_bytes = vec![0u8; n * 8];
    let mut col1_bytes = vec![0u8; n * 8];
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
    let mut out: Vec<(u64, u64)> = Vec::with_capacity(n);
    for i in 0..n {
        let a = u64::from_le_bytes(col0_bytes[i * 8..i * 8 + 8].try_into().unwrap());
        let b = u64::from_le_bytes(col1_bytes[i * 8..i * 8 + 8].try_into().unwrap());
        out.push((a, b));
    }
    out
}

fn cpu_sort_dedup(rows: &[(u64, u64)]) -> Vec<(u64, u64)> {
    let set: BTreeSet<(u64, u64)> = rows.iter().copied().collect();
    set.into_iter().collect()
}

fn cpu_triangle_reference(
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

fn download_triples_u64(buf: &CudaBuffer) -> Vec<(u64, u64, u64)> {
    let n = buf.num_rows() as usize;
    if n == 0 {
        return Vec::new();
    }
    let mut c0 = vec![0u8; n * 8];
    let mut c1 = vec![0u8; n * 8];
    let mut c2 = vec![0u8; n * 8];
    unsafe {
        let r0 = sys::cuMemcpyDtoH_v2(
            c0.as_mut_ptr() as *mut _,
            *buf.column(0).unwrap().device_ptr(),
            c0.len(),
        );
        let r1 = sys::cuMemcpyDtoH_v2(
            c1.as_mut_ptr() as *mut _,
            *buf.column(1).unwrap().device_ptr(),
            c1.len(),
        );
        let r2 = sys::cuMemcpyDtoH_v2(
            c2.as_mut_ptr() as *mut _,
            *buf.column(2).unwrap().device_ptr(),
            c2.len(),
        );
        assert_eq!(r0, sys::cudaError_enum::CUDA_SUCCESS);
        assert_eq!(r1, sys::cudaError_enum::CUDA_SUCCESS);
        assert_eq!(r2, sys::cudaError_enum::CUDA_SUCCESS);
    }
    (0..n)
        .map(|i| {
            (
                u64::from_le_bytes(c0[i * 8..i * 8 + 8].try_into().unwrap()),
                u64::from_le_bytes(c1[i * 8..i * 8 + 8].try_into().unwrap()),
                u64::from_le_bytes(c2[i * 8..i * 8 + 8].try_into().unwrap()),
            )
        })
        .collect()
}

// ---------------------------------------------------------------
// Tests
// ---------------------------------------------------------------

#[test]
fn wcoj_layout_u64_sorts_unsorted_input_lex() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    // Unsorted input including hi-half-bit keys.
    let big = (u32::MAX as u64) + 1;
    let input: Vec<(u64, u64)> = vec![
        (big + 3, 1),
        (1, big + 5),
        (big + 2, 0),
        (1, 3),
        (big + 3, 0),
    ];
    let buf = upload_binary_u64(&fix.memory, &input);
    let stream = fix.pool.acquire().expect("stream");
    let out = fix
        .provider
        .wcoj_layout_u64_recorded(&buf, stream)
        .expect("layout u64");
    assert_eq!(out.schema.column_type(0), Some(ScalarType::U64));
    assert_eq!(out.schema.column_type(1), Some(ScalarType::U64));
    assert_eq!(download_pairs_u64(&out), cpu_sort_dedup(&input));
}

#[test]
fn wcoj_layout_u64_removes_duplicates() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let big = (u32::MAX as u64) + 1;
    let input: Vec<(u64, u64)> = vec![
        (1, 2),
        (1, 2), // dup
        (big, big + 1),
        (big, big + 1), // dup
        (3, 4),
    ];
    let buf = upload_binary_u64(&fix.memory, &input);
    let stream = fix.pool.acquire().expect("stream");
    let out = fix
        .provider
        .wcoj_layout_u64_recorded(&buf, stream)
        .expect("layout u64 dedup");
    assert_eq!(download_pairs_u64(&out), cpu_sort_dedup(&input));
}

#[test]
fn wcoj_layout_u64_empty_input_produces_empty_output() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let buf = upload_binary_u64(&fix.memory, &[]);
    let stream = fix.pool.acquire().expect("stream");
    let out = fix
        .provider
        .wcoj_layout_u64_recorded(&buf, stream)
        .expect("layout u64 empty");
    assert_eq!(download_pairs_u64(&out), Vec::<(u64, u64)>::new());
}

#[test]
fn wcoj_layout_u64_already_sorted_deduped_round_trips() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let big = (u32::MAX as u64) + 1;
    let input: Vec<(u64, u64)> = vec![(1, 2), (1, 5), (3, 0), (big, 1), (big + 1, 0)];
    let buf = upload_binary_u64(&fix.memory, &input);
    let stream = fix.pool.acquire().expect("stream");
    let out = fix
        .provider
        .wcoj_layout_u64_recorded(&buf, stream)
        .expect("layout u64 sorted/deduped");
    assert_eq!(download_pairs_u64(&out), input);
}

#[test]
fn wcoj_layout_u64_legacy_manager_rejected() {
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
    let buf = upload_binary_u64(&memory, &[(1, 2), (3, 4)]);
    let result = provider.wcoj_layout_u64_recorded(&buf, StreamId::DEFAULT);
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
fn wcoj_layout_then_triangle_u64_matches_cpu_oracle() {
    // Feed three unsorted U64 fixtures through layout construction
    // into wcoj_triangle_u64_recorded and verify the row set
    // matches the CPU oracle. End-to-end provider pipeline cert.
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let big = (u32::MAX as u64) + 1;
    let raw_e_xy: Vec<(u64, u64)> = vec![
        (big + 3, big + 4),
        (big + 1, big + 2),
        (big + 1, big + 3),
        (big + 1, big + 2), // dup
        (big + 2, big + 3),
        (big + 5, big + 6),
        (big + 1, big + 4),
        (big + 5, big + 7),
        (big + 2, big + 4),
        (big + 6, big + 7),
        (big + 3, big + 4), // dup
    ];
    let raw_e_yz: Vec<(u64, u64)> = vec![
        (big + 3, big + 4),
        (big + 2, big + 3),
        (big + 6, big + 7),
        (big + 2, big + 4),
        (big + 3, big + 4), // dup
    ];
    let raw_e_xz: Vec<(u64, u64)> = vec![
        (big + 1, big + 4),
        (big + 2, big + 4),
        (big + 3, big + 4),
        (big + 1, big + 3),
        (big + 5, big + 7),
        (big + 1, big + 4), // dup
    ];
    let buf_xy_raw = upload_binary_u64(&fix.memory, &raw_e_xy);
    let buf_yz_raw = upload_binary_u64(&fix.memory, &raw_e_yz);
    let buf_xz_raw = upload_binary_u64(&fix.memory, &raw_e_xz);

    let stream = fix.pool.acquire().expect("layout stream");
    let buf_xy = fix
        .provider
        .wcoj_layout_u64_recorded(&buf_xy_raw, stream)
        .expect("layout xy");
    let buf_yz = fix
        .provider
        .wcoj_layout_u64_recorded(&buf_yz_raw, stream)
        .expect("layout yz");
    let buf_xz = fix
        .provider
        .wcoj_layout_u64_recorded(&buf_xz_raw, stream)
        .expect("layout xz");

    let tri_stream = fix.pool.acquire().expect("triangle stream");
    let result = fix
        .provider
        .wcoj_triangle_u64_recorded(&buf_xy, &buf_yz, &buf_xz, tri_stream)
        .expect("triangle u64");
    assert_eq!(result.schema.column_type(0), Some(ScalarType::U64));
    assert_eq!(result.schema.column_type(1), Some(ScalarType::U64));
    assert_eq!(result.schema.column_type(2), Some(ScalarType::U64));

    // CPU oracle on dedup'd inputs.
    let mut cpu_xy = raw_e_xy.clone();
    cpu_xy.sort();
    cpu_xy.dedup();
    let mut cpu_yz = raw_e_yz.clone();
    cpu_yz.sort();
    cpu_yz.dedup();
    let mut cpu_xz = raw_e_xz.clone();
    cpu_xz.sort();
    cpu_xz.dedup();
    let expected = cpu_triangle_reference(&cpu_xy, &cpu_yz, &cpu_xz);

    let host = download_triples_u64(&result);
    assert_eq!(host, expected);
    assert_eq!(host.len(), 5, "expected 5 triangles on this fixture");
}
