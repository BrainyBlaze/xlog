// crates/xlog-cuda/tests/test_wcoj_project.rs
//! Owned, recorded WCOJ projection helper coverage for
//! `wcoj_project_2col_swap_recorded` +
//! `wcoj_project_output_columns_recorded`.
//!
//! 11 tests:
//!   * 6 swap (u32, u64, Symbol, empty, schema-swap, row-count parity)
//!   * 5 output projection (u32 triangle perm, u64 4-cycle perm,
//!     Symbol, row-count + identity, empty n=0 with non-identity perm)

use std::sync::Arc;

use cudarc::driver::sys;
use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamId, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::CudaBuffer;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

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

// ---------------------------------------------------------------
// Upload / download helpers — minimal, scoped to projection helper tests.
// ---------------------------------------------------------------

fn upload_binary_typed(
    memory: &Arc<GpuMemoryManager>,
    rows: &[(u32, u32)],
    dtype: ScalarType,
) -> CudaBuffer {
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
        ("col0".to_string(), dtype),
        ("col1".to_string(), dtype),
    ]);
    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into()],
        n as u64,
        d_num_rows,
        schema,
        n,
    )
}

fn upload_binary_u64_typed(memory: &Arc<GpuMemoryManager>, rows: &[(u64, u64)]) -> CudaBuffer {
    let n = rows.len() as u32;
    let col0_host: Vec<u64> = rows.iter().map(|(a, _)| *a).collect();
    let col1_host: Vec<u64> = rows.iter().map(|(_, b)| *b).collect();
    let bytes_per_col = (n as usize) * std::mem::size_of::<u64>();
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

fn download_pairs_u32(buf: &CudaBuffer) -> Vec<(u32, u32)> {
    let n = buf.cached_row_count().unwrap() as usize;
    if n == 0 {
        return Vec::new();
    }
    let mut col0 = vec![0u8; n * 4];
    let mut col1 = vec![0u8; n * 4];
    unsafe {
        let r0 = sys::cuMemcpyDtoH_v2(
            col0.as_mut_ptr() as *mut _,
            *buf.column(0).unwrap().device_ptr(),
            col0.len(),
        );
        let r1 = sys::cuMemcpyDtoH_v2(
            col1.as_mut_ptr() as *mut _,
            *buf.column(1).unwrap().device_ptr(),
            col1.len(),
        );
        assert_eq!(r0, sys::cudaError_enum::CUDA_SUCCESS);
        assert_eq!(r1, sys::cudaError_enum::CUDA_SUCCESS);
    }
    (0..n)
        .map(|i| {
            let a = u32::from_le_bytes(col0[i * 4..i * 4 + 4].try_into().unwrap());
            let b = u32::from_le_bytes(col1[i * 4..i * 4 + 4].try_into().unwrap());
            (a, b)
        })
        .collect()
}

fn download_pairs_u64(buf: &CudaBuffer) -> Vec<(u64, u64)> {
    let n = buf.cached_row_count().unwrap() as usize;
    if n == 0 {
        return Vec::new();
    }
    let mut col0 = vec![0u8; n * 8];
    let mut col1 = vec![0u8; n * 8];
    unsafe {
        let r0 = sys::cuMemcpyDtoH_v2(
            col0.as_mut_ptr() as *mut _,
            *buf.column(0).unwrap().device_ptr(),
            col0.len(),
        );
        let r1 = sys::cuMemcpyDtoH_v2(
            col1.as_mut_ptr() as *mut _,
            *buf.column(1).unwrap().device_ptr(),
            col1.len(),
        );
        assert_eq!(r0, sys::cudaError_enum::CUDA_SUCCESS);
        assert_eq!(r1, sys::cudaError_enum::CUDA_SUCCESS);
    }
    (0..n)
        .map(|i| {
            let a = u64::from_le_bytes(col0[i * 8..i * 8 + 8].try_into().unwrap());
            let b = u64::from_le_bytes(col1[i * 8..i * 8 + 8].try_into().unwrap());
            (a, b)
        })
        .collect()
}

fn download_num_rows_device(buf: &CudaBuffer) -> u32 {
    let mut count_host = [0u32; 1];
    unsafe {
        let res = sys::cuMemcpyDtoH_v2(
            count_host.as_mut_ptr() as *mut _,
            *buf.num_rows_device().device_ptr(),
            std::mem::size_of::<u32>(),
        );
        assert_eq!(res, sys::cudaError_enum::CUDA_SUCCESS);
    }
    count_host[0]
}

fn upload_3col_u32(memory: &Arc<GpuMemoryManager>, rows: &[(u32, u32, u32)]) -> CudaBuffer {
    let n = rows.len() as u32;
    let bytes_per_col = (n as usize) * std::mem::size_of::<u32>();
    let mut c0 = memory.alloc::<u8>(bytes_per_col).expect("alloc c0");
    let mut c1 = memory.alloc::<u8>(bytes_per_col).expect("alloc c1");
    let mut c2 = memory.alloc::<u8>(bytes_per_col).expect("alloc c2");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let device = memory.device().inner();
    if !rows.is_empty() {
        let bs0: Vec<u8> = rows.iter().flat_map(|(a, _, _)| a.to_le_bytes()).collect();
        let bs1: Vec<u8> = rows.iter().flat_map(|(_, b, _)| b.to_le_bytes()).collect();
        let bs2: Vec<u8> = rows.iter().flat_map(|(_, _, c)| c.to_le_bytes()).collect();
        device.htod_sync_copy_into(&bs0, &mut c0).unwrap();
        device.htod_sync_copy_into(&bs1, &mut c1).unwrap();
        device.htod_sync_copy_into(&bs2, &mut c2).unwrap();
    }
    device.htod_sync_copy_into(&[n], &mut d_num_rows).unwrap();
    let schema = Schema::new(vec![
        ("c0".to_string(), ScalarType::U32),
        ("c1".to_string(), ScalarType::U32),
        ("c2".to_string(), ScalarType::U32),
    ]);
    CudaBuffer::from_columns_with_host_count(
        vec![c0.into(), c1.into(), c2.into()],
        n as u64,
        d_num_rows,
        schema,
        n,
    )
}

fn download_triples_u32(buf: &CudaBuffer) -> Vec<(u32, u32, u32)> {
    let n = buf.cached_row_count().unwrap() as usize;
    if n == 0 {
        return Vec::new();
    }
    let mut c0 = vec![0u8; n * 4];
    let mut c1 = vec![0u8; n * 4];
    let mut c2 = vec![0u8; n * 4];
    unsafe {
        for (slice, idx) in [(&mut c0, 0usize), (&mut c1, 1), (&mut c2, 2)] {
            let res = sys::cuMemcpyDtoH_v2(
                slice.as_mut_ptr() as *mut _,
                *buf.column(idx).unwrap().device_ptr(),
                slice.len(),
            );
            assert_eq!(res, sys::cudaError_enum::CUDA_SUCCESS);
        }
    }
    (0..n)
        .map(|i| {
            let a = u32::from_le_bytes(c0[i * 4..i * 4 + 4].try_into().unwrap());
            let b = u32::from_le_bytes(c1[i * 4..i * 4 + 4].try_into().unwrap());
            let c = u32::from_le_bytes(c2[i * 4..i * 4 + 4].try_into().unwrap());
            (a, b, c)
        })
        .collect()
}

fn upload_4col_u64(memory: &Arc<GpuMemoryManager>, rows: &[(u64, u64, u64, u64)]) -> CudaBuffer {
    let n = rows.len() as u32;
    let bytes_per_col = (n as usize) * std::mem::size_of::<u64>();
    let mut c0 = memory.alloc::<u8>(bytes_per_col).expect("alloc c0");
    let mut c1 = memory.alloc::<u8>(bytes_per_col).expect("alloc c1");
    let mut c2 = memory.alloc::<u8>(bytes_per_col).expect("alloc c2");
    let mut c3 = memory.alloc::<u8>(bytes_per_col).expect("alloc c3");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let device = memory.device().inner();
    if !rows.is_empty() {
        let bs0: Vec<u8> = rows.iter().flat_map(|t| t.0.to_le_bytes()).collect();
        let bs1: Vec<u8> = rows.iter().flat_map(|t| t.1.to_le_bytes()).collect();
        let bs2: Vec<u8> = rows.iter().flat_map(|t| t.2.to_le_bytes()).collect();
        let bs3: Vec<u8> = rows.iter().flat_map(|t| t.3.to_le_bytes()).collect();
        device.htod_sync_copy_into(&bs0, &mut c0).unwrap();
        device.htod_sync_copy_into(&bs1, &mut c1).unwrap();
        device.htod_sync_copy_into(&bs2, &mut c2).unwrap();
        device.htod_sync_copy_into(&bs3, &mut c3).unwrap();
    }
    device.htod_sync_copy_into(&[n], &mut d_num_rows).unwrap();
    let schema = Schema::new(vec![
        ("c0".to_string(), ScalarType::U64),
        ("c1".to_string(), ScalarType::U64),
        ("c2".to_string(), ScalarType::U64),
        ("c3".to_string(), ScalarType::U64),
    ]);
    CudaBuffer::from_columns_with_host_count(
        vec![c0.into(), c1.into(), c2.into(), c3.into()],
        n as u64,
        d_num_rows,
        schema,
        n,
    )
}

fn download_quads_u64(buf: &CudaBuffer) -> Vec<(u64, u64, u64, u64)> {
    let n = buf.cached_row_count().unwrap() as usize;
    if n == 0 {
        return Vec::new();
    }
    let mut c = [
        vec![0u8; n * 8],
        vec![0u8; n * 8],
        vec![0u8; n * 8],
        vec![0u8; n * 8],
    ];
    unsafe {
        for (idx, col_bytes) in c.iter_mut().enumerate() {
            let res = sys::cuMemcpyDtoH_v2(
                col_bytes.as_mut_ptr() as *mut _,
                *buf.column(idx).unwrap().device_ptr(),
                col_bytes.len(),
            );
            assert_eq!(res, sys::cudaError_enum::CUDA_SUCCESS);
        }
    }
    (0..n)
        .map(|i| {
            let a = u64::from_le_bytes(c[0][i * 8..i * 8 + 8].try_into().unwrap());
            let b = u64::from_le_bytes(c[1][i * 8..i * 8 + 8].try_into().unwrap());
            let cc = u64::from_le_bytes(c[2][i * 8..i * 8 + 8].try_into().unwrap());
            let d = u64::from_le_bytes(c[3][i * 8..i * 8 + 8].try_into().unwrap());
            (a, b, cc, d)
        })
        .collect()
}

// ===============================================================
// Tests — wcoj_project_2col_swap_recorded (6)
// ===============================================================

#[test]
fn swap_u32_round_trip_equals_original() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let rows: Vec<(u32, u32)> = vec![(1, 10), (2, 20), (3, 30)];
    let src = upload_binary_typed(&fix.memory, &rows, ScalarType::U32);
    let stream = StreamId(0);
    let swapped = fix
        .provider
        .wcoj_project_2col_swap_recorded(&src, stream)
        .expect("swap must succeed");
    let unswapped = fix
        .provider
        .wcoj_project_2col_swap_recorded(&swapped, stream)
        .expect("re-swap must succeed");
    assert_eq!(download_pairs_u32(&unswapped), rows);
}

#[test]
fn swap_u64_round_trip_equals_original() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let rows: Vec<(u64, u64)> = vec![(1, 10), (2, 20), (3, 30)];
    let src = upload_binary_u64_typed(&fix.memory, &rows);
    let stream = StreamId(0);
    let swapped = fix
        .provider
        .wcoj_project_2col_swap_recorded(&src, stream)
        .expect("swap u64 must succeed");
    let unswapped = fix
        .provider
        .wcoj_project_2col_swap_recorded(&swapped, stream)
        .expect("re-swap u64 must succeed");
    assert_eq!(download_pairs_u64(&unswapped), rows);
}

#[test]
fn swap_symbol_round_trip_preserves_dtype() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let rows: Vec<(u32, u32)> = vec![(7, 11), (5, 13)];
    let src = upload_binary_typed(&fix.memory, &rows, ScalarType::Symbol);
    let stream = StreamId(0);
    let swapped = fix
        .provider
        .wcoj_project_2col_swap_recorded(&src, stream)
        .expect("swap Symbol must succeed");
    // Schema preserves Symbol dtype after swap (only positions differ).
    assert_eq!(swapped.schema().column_type(0), Some(ScalarType::Symbol));
    assert_eq!(swapped.schema().column_type(1), Some(ScalarType::Symbol));
    // Bits round-trip the same as u32.
    let unswapped = fix
        .provider
        .wcoj_project_2col_swap_recorded(&swapped, stream)
        .expect("re-swap Symbol must succeed");
    assert_eq!(download_pairs_u32(&unswapped), rows);
}

#[test]
fn swap_empty_buffer_yields_empty_output_with_zero_num_rows() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let rows: Vec<(u32, u32)> = vec![];
    let src = upload_binary_typed(&fix.memory, &rows, ScalarType::U32);
    let stream = StreamId(0);
    let swapped = fix
        .provider
        .wcoj_project_2col_swap_recorded(&src, stream)
        .expect("swap empty must succeed");
    assert_eq!(swapped.cached_row_count(), Some(0));
    // create_empty_buffer initializes num_rows_device to 0.
    assert_eq!(download_num_rows_device(&swapped), 0);
}

#[test]
fn swap_schema_reflects_column_swap_in_names() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let rows: Vec<(u32, u32)> = vec![(1, 10)];
    let src = upload_binary_typed(&fix.memory, &rows, ScalarType::U32);
    let stream = StreamId(0);
    let swapped = fix
        .provider
        .wcoj_project_2col_swap_recorded(&src, stream)
        .unwrap();
    // Source schema = ("col0", U32), ("col1", U32). After swap,
    // schema columns are reordered: position 0 holds the old col1.
    assert_eq!(swapped.schema().columns[0].0, "col1");
    assert_eq!(swapped.schema().columns[1].0, "col0");
}

#[test]
fn swap_carries_cached_row_count_and_num_rows_device() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let rows: Vec<(u32, u32)> = vec![(1, 10), (2, 20), (3, 30), (4, 40)];
    let src = upload_binary_typed(&fix.memory, &rows, ScalarType::U32);
    assert_eq!(src.cached_row_count(), Some(4));
    let stream = StreamId(0);
    let swapped = fix
        .provider
        .wcoj_project_2col_swap_recorded(&src, stream)
        .unwrap();
    assert_eq!(swapped.cached_row_count(), Some(4));
    assert_eq!(download_num_rows_device(&swapped), 4);
}

// ===============================================================
// Tests — wcoj_project_output_columns_recorded (5)
// ===============================================================

#[test]
fn output_proj_u32_triangle_e_yz_perm() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    // Simulate a kernel-direct output where leader was e_yz:
    // kernel emits (Y, Z, X). head_proj = [2, 0, 1] → (X, Y, Z).
    // Use synthetic rows: (Y, Z, X) = (10, 20, 1).
    let rows: Vec<(u32, u32, u32)> = vec![(10, 20, 1), (11, 21, 2), (12, 22, 3)];
    let src = upload_3col_u32(&fix.memory, &rows);
    let head_schema = Schema::new(vec![
        ("X".to_string(), ScalarType::U32),
        ("Y".to_string(), ScalarType::U32),
        ("Z".to_string(), ScalarType::U32),
    ]);
    let stream = StreamId(0);
    let projected = fix
        .provider
        .wcoj_project_output_columns_recorded(&src, &[2, 0, 1], head_schema, stream)
        .expect("triangle perm must succeed");
    let triples = download_triples_u32(&projected);
    // Expected: (X, Y, Z) = (col_2, col_0, col_1) = (1, 10, 20), …
    assert_eq!(triples, vec![(1, 10, 20), (2, 11, 21), (3, 12, 22)]);
}

#[test]
fn output_proj_u64_4cycle_e_xy_perm() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    // Simulate a kernel-direct output where leader was e_xy:
    // kernel emits (X, Y, Z, W). head_proj = [3, 0, 1, 2] → (W, X, Y, Z).
    let rows: Vec<(u64, u64, u64, u64)> = vec![(100, 200, 300, 400), (101, 201, 301, 401)];
    let src = upload_4col_u64(&fix.memory, &rows);
    let head_schema = Schema::new(vec![
        ("W".to_string(), ScalarType::U64),
        ("X".to_string(), ScalarType::U64),
        ("Y".to_string(), ScalarType::U64),
        ("Z".to_string(), ScalarType::U64),
    ]);
    let stream = StreamId(0);
    let projected = fix
        .provider
        .wcoj_project_output_columns_recorded(&src, &[3, 0, 1, 2], head_schema, stream)
        .expect("4-cycle perm must succeed");
    let quads = download_quads_u64(&projected);
    // Expected: (W, X, Y, Z) = (col_3, col_0, col_1, col_2) = (400, 100, 200, 300)
    assert_eq!(quads, vec![(400, 100, 200, 300), (401, 101, 201, 301)]);
}

#[test]
fn output_proj_symbol_round_trip_preserves_dtype() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    // 3-col Symbol input. Apply identity permutation; verify dtype
    // and bits preserved.
    let rows: Vec<(u32, u32, u32)> = vec![(1, 2, 3)];
    let mut src = upload_3col_u32(&fix.memory, &rows);
    // Mutate schema to Symbol so the helper sees a 4-byte Symbol-typed input.
    src.schema = Schema::new(vec![
        ("a".to_string(), ScalarType::Symbol),
        ("b".to_string(), ScalarType::Symbol),
        ("c".to_string(), ScalarType::Symbol),
    ]);
    let head_schema = Schema::new(vec![
        ("c".to_string(), ScalarType::Symbol),
        ("a".to_string(), ScalarType::Symbol),
        ("b".to_string(), ScalarType::Symbol),
    ]);
    let stream = StreamId(0);
    let projected = fix
        .provider
        .wcoj_project_output_columns_recorded(&src, &[2, 0, 1], head_schema, stream)
        .expect("Symbol perm must succeed");
    assert_eq!(projected.schema().column_type(0), Some(ScalarType::Symbol));
    assert_eq!(projected.schema().column_type(1), Some(ScalarType::Symbol));
    assert_eq!(projected.schema().column_type(2), Some(ScalarType::Symbol));
    let triples = download_triples_u32(&projected);
    assert_eq!(triples, vec![(3, 1, 2)]);
}

#[test]
fn output_proj_identity_perm_equals_src_with_carried_row_counts() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let rows: Vec<(u32, u32, u32)> = vec![(1, 2, 3), (4, 5, 6), (7, 8, 9)];
    let src = upload_3col_u32(&fix.memory, &rows);
    assert_eq!(src.cached_row_count(), Some(3));
    let head_schema = src.schema().clone();
    let stream = StreamId(0);
    let projected = fix
        .provider
        .wcoj_project_output_columns_recorded(&src, &[0, 1, 2], head_schema, stream)
        .expect("identity perm must succeed");
    // cached_row_count + num_rows_device device scalar both reflect 3.
    assert_eq!(projected.cached_row_count(), Some(3));
    assert_eq!(download_num_rows_device(&projected), 3);
    // Identity content equals src.
    assert_eq!(download_triples_u32(&projected), rows);
}

#[test]
fn output_proj_empty_n_zero_with_non_identity_perm() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    // n=0 case — WCOJ legitimately produces empty output. The
    // helper must not divide-by-zero or refuse to materialize.
    let rows: Vec<(u32, u32, u32)> = vec![];
    let src = upload_3col_u32(&fix.memory, &rows);
    let head_schema = Schema::new(vec![
        ("X".to_string(), ScalarType::U32),
        ("Y".to_string(), ScalarType::U32),
        ("Z".to_string(), ScalarType::U32),
    ]);
    let stream = StreamId(0);
    let projected = fix
        .provider
        .wcoj_project_output_columns_recorded(&src, &[2, 0, 1], head_schema.clone(), stream)
        .expect("empty perm must succeed");
    // Schema is the requested head_schema (column names from head, not src).
    assert_eq!(projected.schema().columns[0].0, "X");
    assert_eq!(projected.schema().columns[1].0, "Y");
    assert_eq!(projected.schema().columns[2].0, "Z");
    assert_eq!(projected.cached_row_count(), Some(0));
    assert_eq!(download_num_rows_device(&projected), 0);
}
