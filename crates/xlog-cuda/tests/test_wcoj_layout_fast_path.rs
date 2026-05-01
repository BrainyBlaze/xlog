// crates/xlog-cuda/tests/test_wcoj_layout_fast_path.rs
//! v0.6.2 — WCOJ layout fast-path correctness tests.
//!
//! `wcoj_layout_u32_recorded` / `wcoj_layout_u64_recorded` gain
//! a proof-based pre-check: if the input is already lex-sorted
//! and full-row unique, skip `dedup_full_row_recorded` (sort +
//! mark-unique + compact) and emit a recorded device-side
//! clone. Phase report at v0.6.2 showed layout to be 91-97%
//! of WCOJ adaptive dispatch wall time on the bench's host-
//! deduped fixtures; this slice targets that overhead.
//!
//! No caching, no buffer fingerprint — purely proof-based per
//! invocation. Failures fall through silently to the existing
//! dedup pipeline; correctness is preserved by construction.
//!
//! Tests cover:
//!   * Fast-path hit on already sorted+unique u32 / u64 / Symbol
//!     inputs; output byte-equivalent to input, schema preserved.
//!   * Fall-through on duplicates (output deduplicated).
//!   * Fall-through on unsorted (output sorted+deduped).
//!   * Compacted-buffer correctness: row_cap > logical_count
//!     must check only the logical rows, not the row_cap slice.
//!   * Empty input (n==0): preserve existing empty-buffer
//!     semantics, no fast-path counter increment.
//!   * Single-row input (n==1): trivial fast-path hit.
//!   * Strict deterministic-D2H gate: zero violations on the
//!     1× 4-byte flag read.
//!   * Provider counter `wcoj_layout_fast_path_hit_count`
//!     increments correctly across all the above.

use std::sync::Arc;

use cudarc::driver::sys;
use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
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
struct Fix {
    device: Arc<CudaDevice>,
    runtime: Arc<XlogDeviceRuntime>,
    memory: Arc<GpuMemoryManager>,
    provider: CudaKernelProvider,
    pool: Arc<StreamPool>,
}

fn make_fix() -> Option<Fix> {
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
    let limit_bytes: usize = 256 * 1024 * 1024;
    let async_resource: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(
        AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool)),
    );
    let logging: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(LoggingResource::new(
        async_resource,
        Arc::new(DiscardSink) as Arc<dyn LoggingSink>,
    ));
    let budget: Box<dyn DeviceMemoryResource + Send + Sync> =
        Box::new(GlobalDeviceBudget::new(logging, limit_bytes));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(limit_bytes as u64),
        Arc::clone(&runtime),
    ));
    let provider =
        CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory)).ok()?;
    Some(Fix {
        device,
        runtime,
        memory,
        provider,
        pool,
    })
}

fn upload_binary_typed(
    memory: &Arc<GpuMemoryManager>,
    rows_u32: &[(u32, u32)],
    ty: ScalarType,
) -> CudaBuffer {
    let n = rows_u32.len() as u32;
    let bytes_per_elem = ty.size_bytes();
    let bpc = (n as usize) * bytes_per_elem;
    let mut col0 = memory.alloc::<u8>(bpc).expect("alloc c0");
    let mut col1 = memory.alloc::<u8>(bpc).expect("alloc c1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc nr");
    let device = memory.device().inner();
    if n > 0 {
        match ty {
            ScalarType::U32 | ScalarType::Symbol => {
                let c0: Vec<u8> = rows_u32.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
                let c1: Vec<u8> = rows_u32.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
                device.htod_sync_copy_into(&c0, &mut col0).expect("htod c0");
                device.htod_sync_copy_into(&c1, &mut col1).expect("htod c1");
            }
            ScalarType::U64 => {
                let c0: Vec<u8> = rows_u32
                    .iter()
                    .flat_map(|(a, _)| (*a as u64).to_le_bytes())
                    .collect();
                let c1: Vec<u8> = rows_u32
                    .iter()
                    .flat_map(|(_, b)| (*b as u64).to_le_bytes())
                    .collect();
                device.htod_sync_copy_into(&c0, &mut col0).expect("htod c0");
                device.htod_sync_copy_into(&c1, &mut col1).expect("htod c1");
            }
            _ => panic!("unsupported scalar type for fast-path test fixture: {ty:?}"),
        }
    }
    device
        .htod_sync_copy_into(&[n], &mut d_num_rows)
        .expect("htod nr");
    let schema = Schema::new(vec![("col0".to_string(), ty), ("col1".to_string(), ty)]);
    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into()],
        n as u64,
        d_num_rows,
        schema,
        n,
    )
}

fn download_pairs_u32(buf: &CudaBuffer) -> Vec<(u32, u32)> {
    let n = buf.cached_row_count().unwrap_or_else(|| {
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
    }) as usize;
    if n == 0 {
        return Vec::new();
    }
    let mut c0 = vec![0u8; n * 4];
    let mut c1 = vec![0u8; n * 4];
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
        assert_eq!(r0, sys::cudaError_enum::CUDA_SUCCESS);
        assert_eq!(r1, sys::cudaError_enum::CUDA_SUCCESS);
    }
    (0..n)
        .map(|i| {
            (
                u32::from_le_bytes(c0[i * 4..i * 4 + 4].try_into().unwrap()),
                u32::from_le_bytes(c1[i * 4..i * 4 + 4].try_into().unwrap()),
            )
        })
        .collect()
}

fn download_pairs_u64(buf: &CudaBuffer) -> Vec<(u64, u64)> {
    let n = buf.cached_row_count().unwrap_or_else(|| {
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
    }) as usize;
    if n == 0 {
        return Vec::new();
    }
    let mut c0 = vec![0u8; n * 8];
    let mut c1 = vec![0u8; n * 8];
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
        assert_eq!(r0, sys::cudaError_enum::CUDA_SUCCESS);
        assert_eq!(r1, sys::cudaError_enum::CUDA_SUCCESS);
    }
    (0..n)
        .map(|i| {
            (
                u64::from_le_bytes(c0[i * 8..i * 8 + 8].try_into().unwrap()),
                u64::from_le_bytes(c1[i * 8..i * 8 + 8].try_into().unwrap()),
            )
        })
        .collect()
}

// =================================================================
// Fast-path hits — already sorted+unique input
// =================================================================

#[test]
fn fast_path_u32_sorted_unique_increments_counter() {
    let Some(fix) = make_fix() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let input = vec![(1u32, 2), (1, 5), (2, 0), (3, 1), (3, 4)];
    let buf = upload_binary_typed(&fix.memory, &input, ScalarType::U32);
    fix.provider.reset_wcoj_layout_fast_path_hit_count();
    let stream = fix.pool.acquire().expect("stream");
    let out = fix
        .provider
        .wcoj_layout_u32_recorded(&buf, stream)
        .expect("layout u32");
    assert_eq!(
        fix.provider.wcoj_layout_fast_path_hit_count(),
        1,
        "sorted+unique u32 input must hit fast-path exactly once"
    );
    assert_eq!(download_pairs_u32(&out), input);
    assert_eq!(out.schema.column_type(0), Some(ScalarType::U32));
    assert_eq!(out.schema.column_type(1), Some(ScalarType::U32));
}

#[test]
fn fast_path_u64_sorted_unique_increments_counter() {
    let Some(fix) = make_fix() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let big = (u32::MAX as u64) + 1;
    let input_u32: Vec<(u32, u32)> = vec![(1, 2), (1, 5), (2, 0), (3, 1), (3, 4)];
    let buf = upload_binary_typed(&fix.memory, &input_u32, ScalarType::U64);
    fix.provider.reset_wcoj_layout_fast_path_hit_count();
    let stream = fix.pool.acquire().expect("stream");
    let out = fix
        .provider
        .wcoj_layout_u64_recorded(&buf, stream)
        .expect("layout u64");
    assert_eq!(
        fix.provider.wcoj_layout_fast_path_hit_count(),
        1,
        "sorted+unique u64 must hit fast-path"
    );
    let expected: Vec<(u64, u64)> = input_u32
        .into_iter()
        .map(|(a, b)| (a as u64, b as u64))
        .collect();
    assert_eq!(download_pairs_u64(&out), expected);
    assert_eq!(out.schema.column_type(0), Some(ScalarType::U64));
    assert_eq!(out.schema.column_type(1), Some(ScalarType::U64));
    let _ = big; // silence unused
}

#[test]
fn fast_path_symbol_sorted_unique_increments_counter() {
    // Symbol shares u32's physical layout — same fast-path
    // should fire.
    let Some(fix) = make_fix() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let input = vec![(1u32, 2), (1, 5), (2, 0)];
    let buf = upload_binary_typed(&fix.memory, &input, ScalarType::Symbol);
    fix.provider.reset_wcoj_layout_fast_path_hit_count();
    let stream = fix.pool.acquire().expect("stream");
    let out = fix
        .provider
        .wcoj_layout_u32_recorded(&buf, stream)
        .expect("layout symbol");
    assert_eq!(
        fix.provider.wcoj_layout_fast_path_hit_count(),
        1,
        "Symbol must follow u32 fast-path"
    );
    assert_eq!(download_pairs_u32(&out), input);
    assert_eq!(out.schema.column_type(0), Some(ScalarType::Symbol));
    assert_eq!(out.schema.column_type(1), Some(ScalarType::Symbol));
}

#[test]
fn fast_path_n1_input_hits() {
    let Some(fix) = make_fix() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let input = vec![(42u32, 7)];
    let buf = upload_binary_typed(&fix.memory, &input, ScalarType::U32);
    fix.provider.reset_wcoj_layout_fast_path_hit_count();
    let stream = fix.pool.acquire().expect("stream");
    let out = fix
        .provider
        .wcoj_layout_u32_recorded(&buf, stream)
        .expect("layout n=1");
    assert_eq!(
        fix.provider.wcoj_layout_fast_path_hit_count(),
        1,
        "n=1 input is trivially sorted+unique → fast-path hit"
    );
    assert_eq!(download_pairs_u32(&out), input);
}

// =================================================================
// Fast-path misses — fall through to dedup_full_row_recorded
// =================================================================

#[test]
fn duplicate_input_falls_back() {
    let Some(fix) = make_fix() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    // Sorted but with adjacent duplicate (1,2) — fast-path
    // (strict lex-increase) must reject.
    let input = vec![(1u32, 2), (1, 2), (2, 0)];
    let buf = upload_binary_typed(&fix.memory, &input, ScalarType::U32);
    fix.provider.reset_wcoj_layout_fast_path_hit_count();
    let stream = fix.pool.acquire().expect("stream");
    let out = fix
        .provider
        .wcoj_layout_u32_recorded(&buf, stream)
        .expect("layout dup");
    assert_eq!(
        fix.provider.wcoj_layout_fast_path_hit_count(),
        0,
        "duplicate must NOT hit fast-path"
    );
    let mut expected = input.clone();
    expected.sort();
    expected.dedup();
    assert_eq!(download_pairs_u32(&out), expected);
}

#[test]
fn unsorted_input_falls_back() {
    let Some(fix) = make_fix() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let input = vec![(3u32, 1), (1, 2), (2, 0)];
    let buf = upload_binary_typed(&fix.memory, &input, ScalarType::U32);
    fix.provider.reset_wcoj_layout_fast_path_hit_count();
    let stream = fix.pool.acquire().expect("stream");
    let out = fix
        .provider
        .wcoj_layout_u32_recorded(&buf, stream)
        .expect("layout unsorted");
    assert_eq!(
        fix.provider.wcoj_layout_fast_path_hit_count(),
        0,
        "unsorted must NOT hit fast-path"
    );
    let mut expected = input.clone();
    expected.sort();
    expected.dedup();
    assert_eq!(download_pairs_u32(&out), expected);
}

// =================================================================
// Compacted buffer (row_cap > logical) — must check only logical
// rows. Build a buffer where the host-cached count == 3 but the
// underlying byte allocation has data for more rows after that;
// the checker must not see bytes past the logical count.
// =================================================================

#[test]
fn compacted_buffer_checks_only_logical_rows() {
    let Some(fix) = make_fix() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    // Allocation: 5 rows worth. Logical count: 3. Bytes 4..5
    // contain values that would VIOLATE strict lex if the
    // checker walked past the logical count: (0, 0) after
    // (3, 1) — that's a decrease.
    let n_logical = 3u32;
    let row_cap = 5u32;
    let bytes = (row_cap as usize) * 4;
    let device = fix.memory.device().inner();
    let mut col0 = fix.memory.alloc::<u8>(bytes).expect("alloc c0");
    let mut col1 = fix.memory.alloc::<u8>(bytes).expect("alloc c1");
    let mut d_num_rows = fix.memory.alloc::<u32>(1).expect("alloc nr");

    // Logical (sorted+unique): (1,2), (2,0), (3,1).
    // Past-logical (row_cap-only): (0,0), (0,0) — would fail.
    let c0_vals: [u32; 5] = [1, 2, 3, 0, 0];
    let c1_vals: [u32; 5] = [2, 0, 1, 0, 0];
    let c0_bytes: Vec<u8> = c0_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let c1_bytes: Vec<u8> = c1_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    device
        .htod_sync_copy_into(&c0_bytes, &mut col0)
        .expect("htod c0");
    device
        .htod_sync_copy_into(&c1_bytes, &mut col1)
        .expect("htod c1");
    device
        .htod_sync_copy_into(&[n_logical], &mut d_num_rows)
        .expect("htod nr");

    let schema = Schema::new(vec![
        ("col0".to_string(), ScalarType::U32),
        ("col1".to_string(), ScalarType::U32),
    ]);
    let buf = CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into()],
        row_cap as u64,
        d_num_rows,
        schema,
        n_logical,
    );
    fix.provider.reset_wcoj_layout_fast_path_hit_count();
    let stream = fix.pool.acquire().expect("stream");
    let out = fix
        .provider
        .wcoj_layout_u32_recorded(&buf, stream)
        .expect("layout compacted");
    assert_eq!(
        fix.provider.wcoj_layout_fast_path_hit_count(),
        1,
        "checker must use logical_row_count_u32, not num_rows() / row_cap; \
         past-logical bytes must be ignored"
    );
    let expected = vec![(1u32, 2), (2, 0), (3, 1)];
    assert_eq!(download_pairs_u32(&out), expected);
}

// =================================================================
// n==0 preserves existing empty-buffer semantics. NO fast-path
// counter increment (the zero-row early-out predates this slice).
// =================================================================

#[test]
fn empty_input_preserves_existing_semantics_no_counter() {
    let Some(fix) = make_fix() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let buf = upload_binary_typed(&fix.memory, &[], ScalarType::U32);
    fix.provider.reset_wcoj_layout_fast_path_hit_count();
    let stream = fix.pool.acquire().expect("stream");
    let out = fix
        .provider
        .wcoj_layout_u32_recorded(&buf, stream)
        .expect("layout empty");
    assert_eq!(
        fix.provider.wcoj_layout_fast_path_hit_count(),
        0,
        "n=0 preserves existing empty-buffer path; no fast-path increment"
    );
    assert_eq!(download_pairs_u32(&out), Vec::<(u32, u32)>::new());
}

// =================================================================
// Strict deterministic-D2H gate: zero violations.
// =================================================================

#[test]
fn fast_path_no_d2h_violations_under_strict_gate() {
    let Some(fix) = make_fix() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let input = vec![(1u32, 2), (1, 5), (2, 0), (3, 1)];
    let buf = upload_binary_typed(&fix.memory, &input, ScalarType::U32);
    fix.provider.reset_deterministic_d2h_violations();
    fix.provider.enable_strict_deterministic_d2h();
    fix.provider.reset_wcoj_layout_fast_path_hit_count();
    let stream = fix.pool.acquire().expect("stream");
    let result = fix.provider.wcoj_layout_u32_recorded(&buf, stream);
    fix.provider.disable_strict_deterministic_d2h();
    let _ = result.expect("layout under strict gate");
    let v = fix.provider.deterministic_d2h_violation_count();
    assert_eq!(
        v, 0,
        "fast-path 4-byte flag D2H must be metadata-class, not tracked; got {v} violations"
    );
    assert_eq!(
        fix.provider.wcoj_layout_fast_path_hit_count(),
        1,
        "fast-path must still fire under strict gate"
    );
}

// =================================================================
// Triangle-style: 3 already-sorted+deduped inputs → 3 fast-path
// hits per dispatch. Locks the bench-relevant case where every
// layout call should short-circuit.
// =================================================================

#[test]
fn three_sorted_unique_inputs_yield_three_fast_path_hits() {
    let Some(fix) = make_fix() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let e_xy = vec![(1u32, 2), (1, 3), (2, 3), (3, 4)];
    let e_yz = vec![(2u32, 3), (3, 4)];
    let e_xz = vec![(1u32, 3), (2, 4), (3, 4)];
    let buf_xy = upload_binary_typed(&fix.memory, &e_xy, ScalarType::U32);
    let buf_yz = upload_binary_typed(&fix.memory, &e_yz, ScalarType::U32);
    let buf_xz = upload_binary_typed(&fix.memory, &e_xz, ScalarType::U32);
    fix.provider.reset_wcoj_layout_fast_path_hit_count();
    let stream = fix.pool.acquire().expect("stream");
    let _ = fix
        .provider
        .wcoj_layout_u32_recorded(&buf_xy, stream)
        .expect("xy");
    let _ = fix
        .provider
        .wcoj_layout_u32_recorded(&buf_yz, stream)
        .expect("yz");
    let _ = fix
        .provider
        .wcoj_layout_u32_recorded(&buf_xz, stream)
        .expect("xz");
    assert_eq!(
        fix.provider.wcoj_layout_fast_path_hit_count(),
        3,
        "3 already-sorted+deduped inputs → 3 fast-path hits"
    );
}
