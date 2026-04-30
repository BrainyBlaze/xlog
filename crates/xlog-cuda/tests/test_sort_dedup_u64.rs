// crates/xlog-cuda/tests/test_sort_dedup_u64.rs
//! v0.6.2 commit-1: extend `sort_recorded` and
//! `dedup_full_row_recorded` to accept U64 key columns.
//!
//! The legacy non-recorded `sort()` already supports U64 via a
//! hi/lo radix-pass strategy (`gather_keys_u64_lo_u32` /
//! `gather_keys_u64_hi_u32`). The recorded variants reject U64
//! today; this slice ports the same strategy into the recorded
//! path so downstream WCOJ U64 layout primitives can compose
//! against it.
//!
//! Hard scope (per slice spec):
//!   * NO new kernels — reuses the existing hi/lo gather pair
//!     from `sort.cu`.
//!   * NO changes to provider/wcoj.rs — that's commit 2.
//!   * NO AST/RIR dispatch changes — that's commit 3.
//!
//! Test surface:
//!   1. `sort_recorded` accepts a single U64 key column and
//!      sorts ascending.
//!   2. `sort_recorded` preserves alignment of a U64 key with a
//!      U32 follower column.
//!   3. `sort_recorded` handles empty U64 input (no kernel
//!      launches).
//!   4. `sort_recorded` round-trips already-sorted U64 input.
//!   5. `sort_recorded` correctly orders U64 keys with hi-half
//!      bits above `u32::MAX` (locks the hi/lo strategy is doing
//!      a true 64-bit sort, not lo-half truncation).
//!   6. `dedup_full_row_recorded` removes U64 duplicates and
//!      returns lex-sorted unique rows.
//!   7. `dedup_full_row_recorded` round-trips a sorted+unique
//!      U64 input unchanged (no false dedup).

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
// Shared helpers (mirror WCOJ test conventions).
// ---------------------------------------------------------------

struct DiscardSink;
impl LoggingSink for DiscardSink {
    fn emit(&self, _record: LogRecord) -> Result<(), SinkError> {
        Ok(())
    }
}

#[allow(dead_code)] // Arc clones keep device/runtime alive across stream lifetimes
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

/// Upload a single-column U64 buffer.
fn upload_unary_u64(memory: &Arc<GpuMemoryManager>, keys: &[u64]) -> CudaBuffer {
    let n = keys.len() as u32;
    let bytes_per_col = (n as usize) * std::mem::size_of::<u64>();
    let mut col0 = memory.alloc::<u8>(bytes_per_col).expect("alloc col0");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let device = memory.device().inner();
    if !keys.is_empty() {
        let bytes: Vec<u8> = keys.iter().flat_map(|v| v.to_le_bytes()).collect();
        device
            .htod_sync_copy_into(&bytes, &mut col0)
            .expect("htod col0");
    }
    device
        .htod_sync_copy_into(&[n], &mut d_num_rows)
        .expect("htod d_num_rows");
    let schema = Schema::new(vec![("key".to_string(), ScalarType::U64)]);
    CudaBuffer::from_columns_with_host_count(vec![col0.into()], n as u64, d_num_rows, schema, n)
}

/// Upload a 2-column buffer with U64 key + U32 value.
fn upload_u64_with_u32_value(memory: &Arc<GpuMemoryManager>, rows: &[(u64, u32)]) -> CudaBuffer {
    let n = rows.len() as u32;
    let key_host: Vec<u64> = rows.iter().map(|(a, _)| *a).collect();
    let val_host: Vec<u32> = rows.iter().map(|(_, b)| *b).collect();
    let key_bytes_per_col = (n as usize) * std::mem::size_of::<u64>();
    let val_bytes_per_col = (n as usize) * std::mem::size_of::<u32>();
    let mut col0 = memory.alloc::<u8>(key_bytes_per_col).expect("alloc col0");
    let mut col1 = memory.alloc::<u8>(val_bytes_per_col).expect("alloc col1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let device = memory.device().inner();
    if n > 0 {
        let key_bytes: Vec<u8> = key_host.iter().flat_map(|v| v.to_le_bytes()).collect();
        let val_bytes: Vec<u8> = val_host.iter().flat_map(|v| v.to_le_bytes()).collect();
        device
            .htod_sync_copy_into(&key_bytes, &mut col0)
            .expect("htod col0");
        device
            .htod_sync_copy_into(&val_bytes, &mut col1)
            .expect("htod col1");
    }
    device
        .htod_sync_copy_into(&[n], &mut d_num_rows)
        .expect("htod d_num_rows");
    let schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U64),
        ("val".to_string(), ScalarType::U32),
    ]);
    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into()],
        n as u64,
        d_num_rows,
        schema,
        n,
    )
}

/// Upload a 2-column U64 buffer (keys × keys, used for full-row dedup).
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

/// Resolve the logical row count: prefer the cached host count,
/// fall back to a 4-byte D2H of `d_num_rows`. Mirrors the WCOJ
/// downloader since dedup compacts in place (`row_cap` ≥ logical).
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

fn download_u64_col(buf: &CudaBuffer, col_idx: usize) -> Vec<u64> {
    let n = logical_row_count(buf);
    if n == 0 {
        return Vec::new();
    }
    let mut bytes = vec![0u8; n * std::mem::size_of::<u64>()];
    unsafe {
        let res = sys::cuMemcpyDtoH_v2(
            bytes.as_mut_ptr() as *mut _,
            *buf.column(col_idx).unwrap().device_ptr(),
            bytes.len(),
        );
        assert_eq!(res, sys::cudaError_enum::CUDA_SUCCESS);
    }
    bytes
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn download_u32_col(buf: &CudaBuffer, col_idx: usize) -> Vec<u32> {
    let n = logical_row_count(buf);
    if n == 0 {
        return Vec::new();
    }
    let mut bytes = vec![0u8; n * std::mem::size_of::<u32>()];
    unsafe {
        let res = sys::cuMemcpyDtoH_v2(
            bytes.as_mut_ptr() as *mut _,
            *buf.column(col_idx).unwrap().device_ptr(),
            bytes.len(),
        );
        assert_eq!(res, sys::cudaError_enum::CUDA_SUCCESS);
    }
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

// ---------------------------------------------------------------
// sort_recorded U64 surface
// ---------------------------------------------------------------

#[test]
fn sort_recorded_u64_single_column_sorts_lex() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let keys: Vec<u64> = vec![5, 2, 8, 1, 9, 3, 7, 4, 6];
    let buf = upload_unary_u64(&fix.memory, &keys);
    let stream = fix.pool.acquire().expect("stream");
    let sorted = fix
        .provider
        .sort_recorded(&buf, &[0], stream)
        .expect("sort_recorded must accept U64 key");
    assert_eq!(
        download_u64_col(&sorted, 0),
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9]
    );
}

#[test]
fn sort_recorded_u64_with_value_column_preserves_alignment() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    // Sort by U64 key; U32 value follows the permutation.
    let rows: Vec<(u64, u32)> = vec![(30, 300), (10, 100), (20, 200), (40, 400)];
    let buf = upload_u64_with_u32_value(&fix.memory, &rows);
    let stream = fix.pool.acquire().expect("stream");
    let sorted = fix
        .provider
        .sort_recorded(&buf, &[0], stream)
        .expect("sort_recorded must accept (U64 key, U32 val)");
    assert_eq!(download_u64_col(&sorted, 0), vec![10, 20, 30, 40]);
    assert_eq!(download_u32_col(&sorted, 1), vec![100, 200, 300, 400]);
}

#[test]
fn sort_recorded_u64_empty_input_yields_empty() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let buf = upload_unary_u64(&fix.memory, &[]);
    let stream = fix.pool.acquire().expect("stream");
    let sorted = fix
        .provider
        .sort_recorded(&buf, &[0], stream)
        .expect("sort_recorded must accept empty U64 input");
    assert_eq!(download_u64_col(&sorted, 0), Vec::<u64>::new());
}

#[test]
fn sort_recorded_u64_already_sorted_round_trips() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let keys: Vec<u64> = vec![1, 2, 3, 5, 8, 13, 21];
    let buf = upload_unary_u64(&fix.memory, &keys);
    let stream = fix.pool.acquire().expect("stream");
    let sorted = fix
        .provider
        .sort_recorded(&buf, &[0], stream)
        .expect("sort_recorded already-sorted U64");
    assert_eq!(download_u64_col(&sorted, 0), keys);
}

#[test]
fn sort_recorded_u64_handles_keys_above_u32_max() {
    // The hi/lo radix strategy must inspect the hi half to order
    // these correctly. A bug that drops the hi half (e.g. only
    // sorting on lo bits) would yield a different ordering than
    // the host-side numeric sort.
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let big = (u32::MAX as u64) + 1;
    let keys: Vec<u64> = vec![
        big + 7,  // hi=1, lo=7
        2,        // hi=0, lo=2
        big,      // hi=1, lo=0
        big + 2,  // hi=1, lo=2
        7,        // hi=0, lo=7
        u64::MAX, // hi=u32::MAX, lo=u32::MAX
        big - 1,  // hi=0, lo=u32::MAX
    ];
    let buf = upload_unary_u64(&fix.memory, &keys);
    let stream = fix.pool.acquire().expect("stream");
    let sorted = fix
        .provider
        .sort_recorded(&buf, &[0], stream)
        .expect("sort_recorded above-u32::MAX U64");
    let mut expected = keys.clone();
    expected.sort();
    assert_eq!(
        download_u64_col(&sorted, 0),
        expected,
        "U64 sort must respect hi-half bits"
    );
}

// ---------------------------------------------------------------
// dedup_full_row_recorded U64 surface
// ---------------------------------------------------------------

#[test]
fn dedup_full_row_recorded_u64_removes_duplicates() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let big = (u32::MAX as u64) + 1;
    let rows: Vec<(u64, u64)> = vec![
        (big, 5),
        (1, 2),
        (1, 2),   // dup
        (big, 5), // dup
        (big, 4),
        (1, 3),
    ];
    let buf = upload_binary_u64(&fix.memory, &rows);
    let stream = fix.pool.acquire().expect("stream");
    let deduped = fix
        .provider
        .dedup_full_row_recorded(&buf, stream)
        .expect("dedup_full_row_recorded must accept U64 columns");
    let mut expected: Vec<(u64, u64)> = rows.clone();
    expected.sort();
    expected.dedup();
    let got_c0 = download_u64_col(&deduped, 0);
    let got_c1 = download_u64_col(&deduped, 1);
    let got: Vec<(u64, u64)> = got_c0.into_iter().zip(got_c1).collect();
    assert_eq!(got, expected);
}

#[test]
fn dedup_full_row_recorded_u64_preserves_unique_input() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let big = (u32::MAX as u64) + 1;
    let rows: Vec<(u64, u64)> = vec![(1, 2), (1, 3), (big, 1), (big, 2)];
    let buf = upload_binary_u64(&fix.memory, &rows);
    let stream = fix.pool.acquire().expect("stream");
    let deduped = fix
        .provider
        .dedup_full_row_recorded(&buf, stream)
        .expect("dedup_full_row_recorded U64 unique input");
    let got_c0 = download_u64_col(&deduped, 0);
    let got_c1 = download_u64_col(&deduped, 1);
    let got: Vec<(u64, u64)> = got_c0.into_iter().zip(got_c1).collect();
    assert_eq!(got, rows, "no duplicates must round-trip in lex order");
}
