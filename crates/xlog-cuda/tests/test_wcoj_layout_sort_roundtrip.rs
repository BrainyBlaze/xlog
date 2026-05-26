// crates/xlog-cuda/tests/test_wcoj_layout_sort_roundtrip.rs
//! W3.1 — round-trip Part B grid for the new generic accessors
//! `CudaKernelProvider::wcoj_layout_sort_u32_recorded` (4-byte
//! width-class) and `CudaKernelProvider::wcoj_layout_sort_u64_recorded`
//! (8-byte width-class).
//!
//! Three round-trip shapes (empty / already-sorted /
//! unsorted+duplicated) × four width-class fixtures (U32 / U64 /
//! Symbol / mixed-4-byte alternating `(U32, Symbol, ...)`) × six
//! arities {2, 3, 4, 5, 6, 7} = **72 cells**. The arity=7 sentinel
//! proves the implementation does not silently cap at the W3.2
//! k-bound.
//!
//! Per-cell asserts:
//!   1. Output `num_rows()` matches expected (deduped, sorted).
//!   2. Each row is lex-≤ its successor (sortedness).
//!   3. No two consecutive rows are equal (full-row uniqueness).
//!   4. Row set equals input row set modulo dedup.
//!   5. Output schema matches input schema bit-for-bit (no
//!      width-class promotion / Symbol→U32 collapse).
//!
//! Symbol fixtures use **real interner-allocated IDs** via
//! `xlog_core::symbol::intern("sym_<n>")`. The interner is the
//! production allocator for `ScalarType::Symbol` values; using
//! its IDs (rather than raw u32 bit patterns) is the
//! Symbol-parity contract this cert is meant to lock. The seed
//! pattern preserves cell-equality structure: every Symbol-typed
//! cell with seed value `n` is replaced by `intern("sym_n")`,
//! so two cells share an interned ID iff they share the seed
//! value, which matches the U32-path equality structure exactly.

use std::collections::BTreeSet;
use std::sync::Arc;

use cudarc::driver::sys;
use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::{CudaBuffer, CudaColumn};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

// ---------------------------------------------------------------
// Shared fixtures
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

// ---------------------------------------------------------------
// Fixture-builder primitives
// ---------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WidthClass {
    U32,
    U64,
    Symbol,
    Mixed4,
}

impl WidthClass {
    fn col_type(self, col_idx: usize) -> ScalarType {
        match self {
            WidthClass::U32 => ScalarType::U32,
            WidthClass::U64 => ScalarType::U64,
            WidthClass::Symbol => ScalarType::Symbol,
            WidthClass::Mixed4 => {
                if col_idx.is_multiple_of(2) {
                    ScalarType::U32
                } else {
                    ScalarType::Symbol
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    Empty,
    AlreadySorted,
    UnsortedDup,
}

/// Locked seed for the AlreadySorted fixture, parameterized by
/// arity. Returns 4 lex-sorted rows. For 4-byte classes
/// (U32 / Symbol / Mixed4), values are interpreted as u32 bits.
/// For U64 class, values fit in u32 too (Sort/dedup lex order is
/// the same; we just store as u64 in the buffer).
///
/// Pattern: row i (0..4) is `[i+1, i+2, ..., i+arity]`. Strictly
/// monotone per row in lex order:
///   row 0: (1, 2, 3, ...)
///   row 1: (2, 3, 4, ...)
///   row 2: (3, 4, 5, ...)
///   row 3: (4, 5, 6, ...)
fn locked_sorted_rows(arity: usize) -> Vec<Vec<u32>> {
    (0..4u32)
        .map(|i| (0..arity as u32).map(|c| i + 1 + c).collect())
        .collect()
}

/// UnsortedDup variant: the AlreadySorted rows interleaved with
/// reverses + duplicates. Same row set after dedup as
/// AlreadySorted.
fn locked_unsorted_dup_rows(arity: usize) -> Vec<Vec<u32>> {
    let sorted = locked_sorted_rows(arity);
    let mut rows = Vec::with_capacity(sorted.len() * 2 + 2);
    // Insert in reverse order.
    for r in sorted.iter().rev() {
        rows.push(r.clone());
    }
    // Then re-insert in forward order (creates duplicates).
    for r in &sorted {
        rows.push(r.clone());
    }
    // Pin two specific dups at the end.
    rows.push(sorted[0].clone());
    rows.push(sorted[2].clone());
    rows
}

fn rows_for_shape(arity: usize, shape: Shape) -> Vec<Vec<u32>> {
    match shape {
        Shape::Empty => Vec::new(),
        Shape::AlreadySorted => locked_sorted_rows(arity),
        Shape::UnsortedDup => locked_unsorted_dup_rows(arity),
    }
}

/// Map a seed value to a real interner-allocated Symbol ID via
/// `xlog_core::symbol::intern("sym_<n>")`. The interner is
/// global+thread-safe and deterministic for a given string, so
/// repeat calls with the same seed return the same u32 ID — the
/// row equality / cell-sharing structure of the seed pattern is
/// preserved 1:1 in the Symbol space.
fn interned_id(seed: u32) -> u32 {
    xlog_core::symbol::intern(&format!("sym_{}", seed))
}

/// Transform a row's per-column values according to the
/// width-class: cells whose ScalarType is `Symbol` get
/// `interned_id(seed)`, every other cell keeps the raw seed
/// value. Used both at buffer-build time AND at expected-set
/// computation time so input/expected/output stay in lock-step.
fn transform_row_for_wc(row: &[u32], wc: WidthClass) -> Vec<u32> {
    row.iter()
        .enumerate()
        .map(|(col_idx, &v)| {
            if matches!(wc.col_type(col_idx), ScalarType::Symbol) {
                interned_id(v)
            } else {
                v
            }
        })
        .collect()
}

// ---------------------------------------------------------------
// Buffer construction (4-byte-class and U64 paths)
// ---------------------------------------------------------------

/// Build a 4-byte-class buffer (U32 / Symbol / Mixed4 — every
/// column has 4-byte physical layout). Per-column ScalarType
/// driven by `wc`.
fn build_buffer_4byte(
    memory: &Arc<GpuMemoryManager>,
    arity: usize,
    wc: WidthClass,
    rows: &[Vec<u32>],
) -> CudaBuffer {
    assert!(!matches!(wc, WidthClass::U64));
    let n = rows.len() as u32;
    let bytes_per_col = (n as usize) * std::mem::size_of::<u32>();
    let mut col_bufs: Vec<CudaColumn> = Vec::with_capacity(arity);
    let device = memory.device().inner();
    for col_idx in 0..arity {
        // Allocate at least 4 bytes even when n==0 so the
        // buffer is layout-valid; the kernel never reads past
        // d_num_rows.
        let alloc_bytes = bytes_per_col.max(4);
        let mut col = memory.alloc::<u8>(alloc_bytes).expect("alloc col");
        if n > 0 {
            let col_bytes: Vec<u8> = rows.iter().flat_map(|r| r[col_idx].to_le_bytes()).collect();
            device
                .htod_sync_copy_into(&col_bytes, &mut col)
                .expect("htod col");
        }
        col_bufs.push(col.into());
    }
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    device
        .htod_sync_copy_into(&[n], &mut d_num_rows)
        .expect("htod num_rows");
    let schema = Schema::new(
        (0..arity)
            .map(|i| (format!("c{}", i), wc.col_type(i)))
            .collect(),
    );
    CudaBuffer::from_columns_with_host_count(col_bufs, n as u64, d_num_rows, schema, n)
}

/// Build an 8-byte-class buffer (U64). Value `r[col]` (as u32)
/// is widened to u64 so the bit pattern is stable across u32 /
/// u64 fixtures of the same locked seed.
fn build_buffer_u64(memory: &Arc<GpuMemoryManager>, arity: usize, rows: &[Vec<u32>]) -> CudaBuffer {
    let n = rows.len() as u32;
    let bytes_per_col = (n as usize) * std::mem::size_of::<u64>();
    let mut col_bufs: Vec<CudaColumn> = Vec::with_capacity(arity);
    let device = memory.device().inner();
    for col_idx in 0..arity {
        let alloc_bytes = bytes_per_col.max(8);
        let mut col = memory.alloc::<u8>(alloc_bytes).expect("alloc col");
        if n > 0 {
            let col_bytes: Vec<u8> = rows
                .iter()
                .flat_map(|r| (r[col_idx] as u64).to_le_bytes())
                .collect();
            device
                .htod_sync_copy_into(&col_bytes, &mut col)
                .expect("htod col");
        }
        col_bufs.push(col.into());
    }
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    device
        .htod_sync_copy_into(&[n], &mut d_num_rows)
        .expect("htod num_rows");
    let schema = Schema::new(
        (0..arity)
            .map(|i| (format!("c{}", i), ScalarType::U64))
            .collect(),
    );
    CudaBuffer::from_columns_with_host_count(col_bufs, n as u64, d_num_rows, schema, n)
}

// ---------------------------------------------------------------
// Output download (per-width-class)
// ---------------------------------------------------------------

fn cached_or_d2h_count(buf: &CudaBuffer) -> usize {
    match buf.cached_row_count() {
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
    }
}

/// Read 4-byte-class output as `Vec<Vec<u32>>` (rows × cols).
/// Works for U32, Symbol, Mixed4 — they all have 4-byte
/// physical columns, so on-wire bits read identically.
fn download_4byte(buf: &CudaBuffer, arity: usize) -> Vec<Vec<u32>> {
    let n = cached_or_d2h_count(buf);
    if n == 0 {
        return Vec::new();
    }
    let mut col_bytes: Vec<Vec<u8>> = (0..arity).map(|_| vec![0u8; n * 4]).collect();
    for (col_idx, bytes) in col_bytes.iter_mut().enumerate().take(arity) {
        unsafe {
            let res = sys::cuMemcpyDtoH_v2(
                bytes.as_mut_ptr() as *mut _,
                *buf.column(col_idx).unwrap().device_ptr(),
                bytes.len(),
            );
            assert_eq!(res, sys::cudaError_enum::CUDA_SUCCESS);
        }
    }
    (0..n)
        .map(|row_i| {
            (0..arity)
                .map(|col_i| {
                    let off = row_i * 4;
                    u32::from_le_bytes([
                        col_bytes[col_i][off],
                        col_bytes[col_i][off + 1],
                        col_bytes[col_i][off + 2],
                        col_bytes[col_i][off + 3],
                    ])
                })
                .collect()
        })
        .collect()
}

/// Read 8-byte-class (U64) output as `Vec<Vec<u64>>`.
fn download_u64(buf: &CudaBuffer, arity: usize) -> Vec<Vec<u64>> {
    let n = cached_or_d2h_count(buf);
    if n == 0 {
        return Vec::new();
    }
    let mut col_bytes: Vec<Vec<u8>> = (0..arity).map(|_| vec![0u8; n * 8]).collect();
    for (col_idx, bytes) in col_bytes.iter_mut().enumerate().take(arity) {
        unsafe {
            let res = sys::cuMemcpyDtoH_v2(
                bytes.as_mut_ptr() as *mut _,
                *buf.column(col_idx).unwrap().device_ptr(),
                bytes.len(),
            );
            assert_eq!(res, sys::cudaError_enum::CUDA_SUCCESS);
        }
    }
    (0..n)
        .map(|row_i| {
            (0..arity)
                .map(|col_i| {
                    let off = row_i * 8;
                    u64::from_le_bytes([
                        col_bytes[col_i][off],
                        col_bytes[col_i][off + 1],
                        col_bytes[col_i][off + 2],
                        col_bytes[col_i][off + 3],
                        col_bytes[col_i][off + 4],
                        col_bytes[col_i][off + 5],
                        col_bytes[col_i][off + 6],
                        col_bytes[col_i][off + 7],
                    ])
                })
                .collect()
        })
        .collect()
}

// ---------------------------------------------------------------
// Per-cell round-trip check
// ---------------------------------------------------------------

fn run_roundtrip_4byte(arity: usize, wc: WidthClass, shape: Shape) {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!(
            "Skipping arity={} wc={:?} shape={:?}: CUDA runtime unavailable",
            arity, wc, shape
        );
        return;
    };
    let raw_rows = rows_for_shape(arity, shape);
    // Transform raw seed rows: Symbol-typed cells go through the
    // global interner; everything else stays raw u32.
    let input_rows: Vec<Vec<u32>> = raw_rows
        .iter()
        .map(|r| transform_row_for_wc(r, wc))
        .collect();
    let buf = build_buffer_4byte(&fix.memory, arity, wc, &input_rows);
    let stream = fix.pool.acquire().expect("stream");
    let out = fix
        .provider
        .wcoj_layout_sort_u32_recorded(&buf, stream)
        .expect("wcoj_layout_sort_u32_recorded must accept 4-byte-class");

    // 5. Schema preservation.
    assert_eq!(out.arity(), arity);
    for col_idx in 0..arity {
        assert_eq!(
            out.schema.column_type(col_idx),
            Some(wc.col_type(col_idx)),
            "schema preservation: arity={} wc={:?} col={}",
            arity,
            wc,
            col_idx
        );
    }

    let actual = download_4byte(&out, arity);
    assert_roundtrip_props_u32(arity, wc, shape, &input_rows, &actual);
}

fn run_roundtrip_u64(arity: usize, shape: Shape) {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!(
            "Skipping arity={} wc=U64 shape={:?}: CUDA runtime unavailable",
            arity, shape
        );
        return;
    };
    let input_rows = rows_for_shape(arity, shape);
    let buf = build_buffer_u64(&fix.memory, arity, &input_rows);
    let stream = fix.pool.acquire().expect("stream");
    let out = fix
        .provider
        .wcoj_layout_sort_u64_recorded(&buf, stream)
        .expect("wcoj_layout_sort_u64_recorded must accept U64");

    // 5. Schema preservation.
    assert_eq!(out.arity(), arity);
    for col_idx in 0..arity {
        assert_eq!(
            out.schema.column_type(col_idx),
            Some(ScalarType::U64),
            "schema preservation: arity={} U64 col={}",
            arity,
            col_idx
        );
    }

    let actual_u64 = download_u64(&out, arity);
    let actual_u32: Vec<Vec<u32>> = actual_u64
        .into_iter()
        .map(|r| r.into_iter().map(|v| v as u32).collect())
        .collect();
    assert_roundtrip_props_u32(arity, WidthClass::U64, shape, &input_rows, &actual_u32);
}

fn assert_roundtrip_props_u32(
    arity: usize,
    wc: WidthClass,
    shape: Shape,
    input: &[Vec<u32>],
    actual: &[Vec<u32>],
) {
    // 1. Expected row count.
    let expected: BTreeSet<Vec<u32>> = input.iter().cloned().collect();
    assert_eq!(
        actual.len(),
        expected.len(),
        "row-count mismatch: arity={} wc={:?} shape={:?} expected_unique={} actual={}",
        arity,
        wc,
        shape,
        expected.len(),
        actual.len()
    );
    // 2. Sortedness.
    for w in actual.windows(2) {
        assert!(
            w[0] <= w[1],
            "out of order: arity={} wc={:?} shape={:?} prev={:?} next={:?}",
            arity,
            wc,
            shape,
            w[0],
            w[1]
        );
    }
    // 3. Full-row uniqueness.
    for w in actual.windows(2) {
        assert_ne!(
            w[0], w[1],
            "duplicate adjacent rows: arity={} wc={:?} shape={:?} row={:?}",
            arity, wc, shape, w[0]
        );
    }
    // 4. Set equality.
    let actual_set: BTreeSet<Vec<u32>> = actual.iter().cloned().collect();
    assert_eq!(
        actual_set, expected,
        "row set mismatch: arity={} wc={:?} shape={:?}",
        arity, wc, shape
    );
}

// ===============================================================
// 4-byte width-class — U32, Symbol, Mixed4 — 18 cells × 3 = 54
// ===============================================================

// arity 2

#[test]
fn roundtrip_u32_arity2_empty() {
    run_roundtrip_4byte(2, WidthClass::U32, Shape::Empty);
}
#[test]
fn roundtrip_u32_arity2_sorted() {
    run_roundtrip_4byte(2, WidthClass::U32, Shape::AlreadySorted);
}
#[test]
fn roundtrip_u32_arity2_unsorted_dup() {
    run_roundtrip_4byte(2, WidthClass::U32, Shape::UnsortedDup);
}
#[test]
fn roundtrip_symbol_arity2_empty() {
    run_roundtrip_4byte(2, WidthClass::Symbol, Shape::Empty);
}
#[test]
fn roundtrip_symbol_arity2_sorted() {
    run_roundtrip_4byte(2, WidthClass::Symbol, Shape::AlreadySorted);
}
#[test]
fn roundtrip_symbol_arity2_unsorted_dup() {
    run_roundtrip_4byte(2, WidthClass::Symbol, Shape::UnsortedDup);
}
#[test]
fn roundtrip_mixed4_arity2_empty() {
    run_roundtrip_4byte(2, WidthClass::Mixed4, Shape::Empty);
}
#[test]
fn roundtrip_mixed4_arity2_sorted() {
    run_roundtrip_4byte(2, WidthClass::Mixed4, Shape::AlreadySorted);
}
#[test]
fn roundtrip_mixed4_arity2_unsorted_dup() {
    run_roundtrip_4byte(2, WidthClass::Mixed4, Shape::UnsortedDup);
}

// arity 3
#[test]
fn roundtrip_u32_arity3_empty() {
    run_roundtrip_4byte(3, WidthClass::U32, Shape::Empty);
}
#[test]
fn roundtrip_u32_arity3_sorted() {
    run_roundtrip_4byte(3, WidthClass::U32, Shape::AlreadySorted);
}
#[test]
fn roundtrip_u32_arity3_unsorted_dup() {
    run_roundtrip_4byte(3, WidthClass::U32, Shape::UnsortedDup);
}
#[test]
fn roundtrip_symbol_arity3_empty() {
    run_roundtrip_4byte(3, WidthClass::Symbol, Shape::Empty);
}
#[test]
fn roundtrip_symbol_arity3_sorted() {
    run_roundtrip_4byte(3, WidthClass::Symbol, Shape::AlreadySorted);
}
#[test]
fn roundtrip_symbol_arity3_unsorted_dup() {
    run_roundtrip_4byte(3, WidthClass::Symbol, Shape::UnsortedDup);
}
#[test]
fn roundtrip_mixed4_arity3_empty() {
    run_roundtrip_4byte(3, WidthClass::Mixed4, Shape::Empty);
}
#[test]
fn roundtrip_mixed4_arity3_sorted() {
    run_roundtrip_4byte(3, WidthClass::Mixed4, Shape::AlreadySorted);
}
#[test]
fn roundtrip_mixed4_arity3_unsorted_dup() {
    run_roundtrip_4byte(3, WidthClass::Mixed4, Shape::UnsortedDup);
}

// arity 4
#[test]
fn roundtrip_u32_arity4_empty() {
    run_roundtrip_4byte(4, WidthClass::U32, Shape::Empty);
}
#[test]
fn roundtrip_u32_arity4_sorted() {
    run_roundtrip_4byte(4, WidthClass::U32, Shape::AlreadySorted);
}
#[test]
fn roundtrip_u32_arity4_unsorted_dup() {
    run_roundtrip_4byte(4, WidthClass::U32, Shape::UnsortedDup);
}
#[test]
fn roundtrip_symbol_arity4_empty() {
    run_roundtrip_4byte(4, WidthClass::Symbol, Shape::Empty);
}
#[test]
fn roundtrip_symbol_arity4_sorted() {
    run_roundtrip_4byte(4, WidthClass::Symbol, Shape::AlreadySorted);
}
#[test]
fn roundtrip_symbol_arity4_unsorted_dup() {
    run_roundtrip_4byte(4, WidthClass::Symbol, Shape::UnsortedDup);
}
#[test]
fn roundtrip_mixed4_arity4_empty() {
    run_roundtrip_4byte(4, WidthClass::Mixed4, Shape::Empty);
}
#[test]
fn roundtrip_mixed4_arity4_sorted() {
    run_roundtrip_4byte(4, WidthClass::Mixed4, Shape::AlreadySorted);
}
#[test]
fn roundtrip_mixed4_arity4_unsorted_dup() {
    run_roundtrip_4byte(4, WidthClass::Mixed4, Shape::UnsortedDup);
}

// arity 5
#[test]
fn roundtrip_u32_arity5_empty() {
    run_roundtrip_4byte(5, WidthClass::U32, Shape::Empty);
}
#[test]
fn roundtrip_u32_arity5_sorted() {
    run_roundtrip_4byte(5, WidthClass::U32, Shape::AlreadySorted);
}
#[test]
fn roundtrip_u32_arity5_unsorted_dup() {
    run_roundtrip_4byte(5, WidthClass::U32, Shape::UnsortedDup);
}
#[test]
fn roundtrip_symbol_arity5_empty() {
    run_roundtrip_4byte(5, WidthClass::Symbol, Shape::Empty);
}
#[test]
fn roundtrip_symbol_arity5_sorted() {
    run_roundtrip_4byte(5, WidthClass::Symbol, Shape::AlreadySorted);
}
#[test]
fn roundtrip_symbol_arity5_unsorted_dup() {
    run_roundtrip_4byte(5, WidthClass::Symbol, Shape::UnsortedDup);
}
#[test]
fn roundtrip_mixed4_arity5_empty() {
    run_roundtrip_4byte(5, WidthClass::Mixed4, Shape::Empty);
}
#[test]
fn roundtrip_mixed4_arity5_sorted() {
    run_roundtrip_4byte(5, WidthClass::Mixed4, Shape::AlreadySorted);
}
#[test]
fn roundtrip_mixed4_arity5_unsorted_dup() {
    run_roundtrip_4byte(5, WidthClass::Mixed4, Shape::UnsortedDup);
}

// arity 6
#[test]
fn roundtrip_u32_arity6_empty() {
    run_roundtrip_4byte(6, WidthClass::U32, Shape::Empty);
}
#[test]
fn roundtrip_u32_arity6_sorted() {
    run_roundtrip_4byte(6, WidthClass::U32, Shape::AlreadySorted);
}
#[test]
fn roundtrip_u32_arity6_unsorted_dup() {
    run_roundtrip_4byte(6, WidthClass::U32, Shape::UnsortedDup);
}
#[test]
fn roundtrip_symbol_arity6_empty() {
    run_roundtrip_4byte(6, WidthClass::Symbol, Shape::Empty);
}
#[test]
fn roundtrip_symbol_arity6_sorted() {
    run_roundtrip_4byte(6, WidthClass::Symbol, Shape::AlreadySorted);
}
#[test]
fn roundtrip_symbol_arity6_unsorted_dup() {
    run_roundtrip_4byte(6, WidthClass::Symbol, Shape::UnsortedDup);
}
#[test]
fn roundtrip_mixed4_arity6_empty() {
    run_roundtrip_4byte(6, WidthClass::Mixed4, Shape::Empty);
}
#[test]
fn roundtrip_mixed4_arity6_sorted() {
    run_roundtrip_4byte(6, WidthClass::Mixed4, Shape::AlreadySorted);
}
#[test]
fn roundtrip_mixed4_arity6_unsorted_dup() {
    run_roundtrip_4byte(6, WidthClass::Mixed4, Shape::UnsortedDup);
}

// arity 7 — sentinel
#[test]
fn roundtrip_u32_arity7_empty() {
    run_roundtrip_4byte(7, WidthClass::U32, Shape::Empty);
}
#[test]
fn roundtrip_u32_arity7_sorted() {
    run_roundtrip_4byte(7, WidthClass::U32, Shape::AlreadySorted);
}
#[test]
fn roundtrip_u32_arity7_unsorted_dup() {
    run_roundtrip_4byte(7, WidthClass::U32, Shape::UnsortedDup);
}
#[test]
fn roundtrip_symbol_arity7_empty() {
    run_roundtrip_4byte(7, WidthClass::Symbol, Shape::Empty);
}
#[test]
fn roundtrip_symbol_arity7_sorted() {
    run_roundtrip_4byte(7, WidthClass::Symbol, Shape::AlreadySorted);
}
#[test]
fn roundtrip_symbol_arity7_unsorted_dup() {
    run_roundtrip_4byte(7, WidthClass::Symbol, Shape::UnsortedDup);
}
#[test]
fn roundtrip_mixed4_arity7_empty() {
    run_roundtrip_4byte(7, WidthClass::Mixed4, Shape::Empty);
}
#[test]
fn roundtrip_mixed4_arity7_sorted() {
    run_roundtrip_4byte(7, WidthClass::Mixed4, Shape::AlreadySorted);
}
#[test]
fn roundtrip_mixed4_arity7_unsorted_dup() {
    run_roundtrip_4byte(7, WidthClass::Mixed4, Shape::UnsortedDup);
}

// ===============================================================
// 8-byte width-class — U64 — 18 cells
// ===============================================================

#[test]
fn roundtrip_u64_arity2_empty() {
    run_roundtrip_u64(2, Shape::Empty);
}
#[test]
fn roundtrip_u64_arity2_sorted() {
    run_roundtrip_u64(2, Shape::AlreadySorted);
}
#[test]
fn roundtrip_u64_arity2_unsorted_dup() {
    run_roundtrip_u64(2, Shape::UnsortedDup);
}
#[test]
fn roundtrip_u64_arity3_empty() {
    run_roundtrip_u64(3, Shape::Empty);
}
#[test]
fn roundtrip_u64_arity3_sorted() {
    run_roundtrip_u64(3, Shape::AlreadySorted);
}
#[test]
fn roundtrip_u64_arity3_unsorted_dup() {
    run_roundtrip_u64(3, Shape::UnsortedDup);
}
#[test]
fn roundtrip_u64_arity4_empty() {
    run_roundtrip_u64(4, Shape::Empty);
}
#[test]
fn roundtrip_u64_arity4_sorted() {
    run_roundtrip_u64(4, Shape::AlreadySorted);
}
#[test]
fn roundtrip_u64_arity4_unsorted_dup() {
    run_roundtrip_u64(4, Shape::UnsortedDup);
}
#[test]
fn roundtrip_u64_arity5_empty() {
    run_roundtrip_u64(5, Shape::Empty);
}
#[test]
fn roundtrip_u64_arity5_sorted() {
    run_roundtrip_u64(5, Shape::AlreadySorted);
}
#[test]
fn roundtrip_u64_arity5_unsorted_dup() {
    run_roundtrip_u64(5, Shape::UnsortedDup);
}
#[test]
fn roundtrip_u64_arity6_empty() {
    run_roundtrip_u64(6, Shape::Empty);
}
#[test]
fn roundtrip_u64_arity6_sorted() {
    run_roundtrip_u64(6, Shape::AlreadySorted);
}
#[test]
fn roundtrip_u64_arity6_unsorted_dup() {
    run_roundtrip_u64(6, Shape::UnsortedDup);
}
#[test]
fn roundtrip_u64_arity7_empty() {
    run_roundtrip_u64(7, Shape::Empty);
}
#[test]
fn roundtrip_u64_arity7_sorted() {
    run_roundtrip_u64(7, Shape::AlreadySorted);
}
#[test]
fn roundtrip_u64_arity7_unsorted_dup() {
    run_roundtrip_u64(7, Shape::UnsortedDup);
}
