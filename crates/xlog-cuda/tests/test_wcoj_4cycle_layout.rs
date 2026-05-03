// crates/xlog-cuda/tests/test_wcoj_4cycle_layout.rs
//! v0.6.5 slice 2 — layout reuse smoke for 4-cycle.
//!
//! `wcoj_layout_u32_recorded` and `wcoj_layout_u64_recorded` were
//! introduced in v0.6.2 to produce sorted+deduped 2-column buffers
//! for WCOJ kernel consumption. Their contract is shape-agnostic:
//! they sort lex `(col0, col1)` and dedup, regardless of which
//! kernel downstream consumes the result.
//!
//! This test pins that contract for 4-cycle: feed unsorted+duplicate
//! 2-column u32 inputs through the layout helper, then through the
//! 4-cycle u32 dispatch, and assert correct row sets. No new layout
//! kernel is required — slice 2 reuses the existing helpers
//! verbatim.

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

fn upload_unsorted_u32(memory: &Arc<GpuMemoryManager>, rows: &[(u32, u32)]) -> CudaBuffer {
    let n = rows.len() as u32;
    let bytes_per_col = (n as usize) * std::mem::size_of::<u32>();
    let mut col0 = memory.alloc::<u8>(bytes_per_col).expect("alloc col0");
    let mut col1 = memory.alloc::<u8>(bytes_per_col).expect("alloc col1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let device = memory.device().inner();
    if n > 0 {
        let c0: Vec<u8> = rows.iter().flat_map(|&(a, _)| a.to_le_bytes()).collect();
        let c1: Vec<u8> = rows.iter().flat_map(|&(_, b)| b.to_le_bytes()).collect();
        device.htod_sync_copy_into(&c0, &mut col0).expect("htod c0");
        device.htod_sync_copy_into(&c1, &mut col1).expect("htod c1");
    }
    device
        .htod_sync_copy_into(&[n], &mut d_num_rows)
        .expect("htod n");
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

fn download_quads(buf: &CudaBuffer) -> Vec<(u32, u32, u32, u32)> {
    let n = buf.num_rows() as usize;
    if n == 0 {
        return Vec::new();
    }
    assert_eq!(buf.arity(), 4);
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
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push((
            u32::from_le_bytes(bytes[0][i * 4..i * 4 + 4].try_into().unwrap()),
            u32::from_le_bytes(bytes[1][i * 4..i * 4 + 4].try_into().unwrap()),
            u32::from_le_bytes(bytes[2][i * 4..i * 4 + 4].try_into().unwrap()),
            u32::from_le_bytes(bytes[3][i * 4..i * 4 + 4].try_into().unwrap()),
        ));
    }
    out
}

#[test]
fn wcoj_layout_then_4cycle_u32_handles_unsorted_dedup_inputs() {
    // Unsorted + duplicated edges: same perfect-square graph as the
    // 4-cycle u32 correctness test, but the rows are scrambled and
    // duplicated. The layout helper must sort+dedup; the 4-cycle
    // kernel then sees clean inputs and emits the same 4 quads.
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    // Scrambled order, with duplicates. The canonical sorted+deduped
    // set is [(1,2), (2,3), (3,4), (4,1)].
    let unsorted: Vec<(u32, u32)> = vec![
        (3, 4),
        (1, 2),
        (4, 1),
        (2, 3),
        (1, 2), // duplicate
        (3, 4), // duplicate
    ];

    // Upload four copies, run each through the layout helper.
    let launch_stream = fix.pool.acquire().expect("acquire launch_stream");

    let raw_e1 = upload_unsorted_u32(&fix.memory, &unsorted);
    let raw_e2 = upload_unsorted_u32(&fix.memory, &unsorted);
    let raw_e3 = upload_unsorted_u32(&fix.memory, &unsorted);
    let raw_e4 = upload_unsorted_u32(&fix.memory, &unsorted);

    let layout_e1 = fix
        .provider
        .wcoj_layout_u32_recorded(&raw_e1, launch_stream)
        .expect("layout e1");
    let layout_e2 = fix
        .provider
        .wcoj_layout_u32_recorded(&raw_e2, launch_stream)
        .expect("layout e2");
    let layout_e3 = fix
        .provider
        .wcoj_layout_u32_recorded(&raw_e3, launch_stream)
        .expect("layout e3");
    let layout_e4 = fix
        .provider
        .wcoj_layout_u32_recorded(&raw_e4, launch_stream)
        .expect("layout e4");

    // Each layout output must dedup to exactly 4 logical rows
    // (the 6 raw rows contain 2 duplicates). `num_rows()` is the
    // allocation cap; the post-dedup logical count lives in the
    // device-resident `d_num_rows` slot, accessed via the cached
    // host count when available or a 4-byte D2H. The 4-cycle
    // kernel reads this internally; here we just sanity-check.
    let logical_rows = |buf: &CudaBuffer| -> u32 {
        match buf.cached_row_count() {
            Some(c) => c as u32,
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
                count_host[0]
            }
        }
    };
    assert_eq!(logical_rows(&layout_e1), 4, "layout must dedup to 4 rows");
    assert_eq!(logical_rows(&layout_e2), 4);
    assert_eq!(logical_rows(&layout_e3), 4);
    assert_eq!(logical_rows(&layout_e4), 4);

    // Feed the layout outputs straight into the 4-cycle kernel.
    let result = fix
        .provider
        .wcoj_4cycle_u32_recorded(
            &layout_e1,
            &layout_e2,
            &layout_e3,
            &layout_e4,
            launch_stream,
        )
        .expect("wcoj_4cycle_u32_recorded over layout outputs");
    let quads: BTreeSet<(u32, u32, u32, u32)> = download_quads(&result).into_iter().collect();
    let expected: BTreeSet<(u32, u32, u32, u32)> =
        [(1, 2, 3, 4), (2, 3, 4, 1), (3, 4, 1, 2), (4, 1, 2, 3)]
            .into_iter()
            .collect();
    assert_eq!(
        quads, expected,
        "layout → 4-cycle pipeline must produce the canonical 4 perfect-square cycles"
    );
}
