// crates/xlog-cuda/tests/test_wcoj_layout_sort_u64.rs
//! Width-class validation tests for the generic 8-byte-class
//! WCOJ layout-sort entry point
//! `CudaKernelProvider::wcoj_layout_sort_u64_recorded`.
//!
//! Mirror of `test_wcoj_layout_sort_u32.rs` at the 8-byte
//! width-class. Pins:
//!   * Manager must be runtime-backed.
//!   * `input.arity() >= 2`.
//!   * Every column type = `U64`. `U32` / `Symbol` rejected
//!     (use `wcoj_layout_sort_u32_recorded` instead).
//!   * Stream resolution is owned by `dedup_full_row_recorded`.

use std::sync::Arc;

use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamId, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::{CudaBuffer, CudaColumn};
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

fn empty_buf_with_types(memory: &Arc<GpuMemoryManager>, col_types: &[ScalarType]) -> CudaBuffer {
    let mut cols: Vec<CudaColumn> = Vec::with_capacity(col_types.len());
    for ty in col_types {
        let buf = memory.alloc::<u8>(ty.size_bytes()).expect("alloc col");
        cols.push(buf.into());
    }
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    memory
        .device()
        .inner()
        .htod_sync_copy_into(&[0u32; 1], &mut d_num_rows)
        .expect("htod num_rows = 0");
    let schema = Schema::new(
        col_types
            .iter()
            .enumerate()
            .map(|(i, ty)| (format!("c{}", i), *ty))
            .collect(),
    );
    CudaBuffer::from_columns_with_host_count(cols, 0, d_num_rows, schema, 0)
}

fn unwrap_err<T>(result: Result<T, xlog_core::XlogError>, msg: &str) -> xlog_core::XlogError {
    match result {
        Ok(_) => panic!("{}", msg),
        Err(e) => e,
    }
}

// ===============================================================
// Width-class validation — 5 tests
// ===============================================================

#[test]
fn arity_2_rejects_u32_column() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let buf = empty_buf_with_types(&fix.memory, &[ScalarType::U64, ScalarType::U32]);
    let err = unwrap_err(
        fix.provider
            .wcoj_layout_sort_u64_recorded(&buf, StreamId::DEFAULT),
        "U32 column must be rejected by 8-byte entry",
    );
    let msg = format!("{:?}", err);
    assert!(
        msg.contains("must be U64"),
        "error must mention U64; got: {}",
        msg
    );
}

#[test]
fn arity_3_rejects_mixed_8byte_4byte() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let buf = empty_buf_with_types(
        &fix.memory,
        &[ScalarType::U64, ScalarType::U32, ScalarType::U64],
    );
    let err = unwrap_err(
        fix.provider
            .wcoj_layout_sort_u64_recorded(&buf, StreamId::DEFAULT),
        "mixed 8-byte + 4-byte must be rejected",
    );
    let msg = format!("{:?}", err);
    assert!(
        msg.contains("8-byte width-class"),
        "error must mention 8-byte width-class; got: {}",
        msg
    );
}

#[test]
fn arity_4_accepts_uniform_u64() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let buf = empty_buf_with_types(
        &fix.memory,
        &[
            ScalarType::U64,
            ScalarType::U64,
            ScalarType::U64,
            ScalarType::U64,
        ],
    );
    let out = fix
        .provider
        .wcoj_layout_sort_u64_recorded(&buf, StreamId::DEFAULT)
        .expect("uniform U64 at arity 4 must be accepted");
    assert_eq!(out.arity(), 4);
    for col_idx in 0..4 {
        assert_eq!(out.schema.column_type(col_idx), Some(ScalarType::U64));
    }
}

#[test]
fn arity_below_2_rejected() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let buf = empty_buf_with_types(&fix.memory, &[ScalarType::U64]);
    let err = unwrap_err(
        fix.provider
            .wcoj_layout_sort_u64_recorded(&buf, StreamId::DEFAULT),
        "arity-1 must be rejected",
    );
    let msg = format!("{:?}", err);
    assert!(
        msg.contains("arity >= 2"),
        "error must mention arity >= 2; got: {}",
        msg
    );
}

#[test]
fn runtime_backed_required() {
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
    let buf = empty_buf_with_types(&memory, &[ScalarType::U64, ScalarType::U64]);
    let err = unwrap_err(
        provider.wcoj_layout_sort_u64_recorded(&buf, StreamId::DEFAULT),
        "legacy manager must be rejected",
    );
    let msg = format!("{:?}", err);
    assert!(
        msg.contains("with_runtime"),
        "error must mention with_runtime; got: {}",
        msg
    );
}
