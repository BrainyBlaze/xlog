// crates/xlog-cuda/tests/test_wcoj_4cycle_u64.rs
//! v0.6.5 slice 2 — 4-cycle WCOJ kernel (u64) provider tests.
//!
//! Locks the provider entry
//! `CudaKernelProvider::wcoj_4cycle_u64_recorded(e1, e2, e3, e4, launch_stream)`
//! against the rule
//!
//!   cycle4(W, X, Y, Z) :- e1(W, X), e2(X, Y), e3(Y, Z), e4(Z, W)
//!
//! Mirrors `test_wcoj_4cycle_u32.rs` but with U64 inputs and outputs.

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

fn download_quads_u64(buf: &CudaBuffer) -> Vec<(u64, u64, u64, u64)> {
    let n = buf.num_rows() as usize;
    if n == 0 {
        return Vec::new();
    }
    assert_eq!(buf.arity(), 4);
    let mut bytes = [
        vec![0u8; n * 8],
        vec![0u8; n * 8],
        vec![0u8; n * 8],
        vec![0u8; n * 8],
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
        let w = u64::from_le_bytes(bytes[0][i * 8..i * 8 + 8].try_into().unwrap());
        let x = u64::from_le_bytes(bytes[1][i * 8..i * 8 + 8].try_into().unwrap());
        let y = u64::from_le_bytes(bytes[2][i * 8..i * 8 + 8].try_into().unwrap());
        let z = u64::from_le_bytes(bytes[3][i * 8..i * 8 + 8].try_into().unwrap());
        out.push((w, x, y, z));
    }
    out
}

fn cpu_4cycle_reference_u64(
    e1: &[(u64, u64)],
    e2: &[(u64, u64)],
    e3: &[(u64, u64)],
    e4: &[(u64, u64)],
) -> Vec<(u64, u64, u64, u64)> {
    let e2_set: BTreeSet<(u64, u64)> = e2.iter().copied().collect();
    let e3_set: BTreeSet<(u64, u64)> = e3.iter().copied().collect();
    let e4_set: BTreeSet<(u64, u64)> = e4.iter().copied().collect();
    let mut out: BTreeSet<(u64, u64, u64, u64)> = BTreeSet::new();
    for &(w, x) in e1 {
        for &(x2, y) in e2 {
            if x2 != x {
                continue;
            }
            for &(y2, z) in e3 {
                if y2 != y {
                    continue;
                }
                if e4_set.contains(&(z, w)) && e2_set.contains(&(x, y)) && e3_set.contains(&(y, z))
                {
                    out.insert((w, x, y, z));
                }
            }
        }
    }
    out.into_iter().collect()
}

#[test]
fn wcoj_4cycle_u64_matches_cpu_reference_perfect_square() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    // Use values larger than u32::MAX to lock that the path is
    // genuinely 64-bit, not u32 with wide schemas.
    let big = (1u64 << 40) - 1; // > u32::MAX
    let edges: Vec<(u64, u64)> = vec![
        (big, big + 1),
        (big + 1, big + 2),
        (big + 2, big + 3),
        (big + 3, big),
    ];
    let buf_e1 = upload_binary_u64(&fix.memory, &edges);
    let buf_e2 = upload_binary_u64(&fix.memory, &edges);
    let buf_e3 = upload_binary_u64(&fix.memory, &edges);
    let buf_e4 = upload_binary_u64(&fix.memory, &edges);

    let launch_stream = fix.pool.acquire().expect("acquire launch_stream");
    let result = fix
        .provider
        .wcoj_4cycle_u64_recorded(&buf_e1, &buf_e2, &buf_e3, &buf_e4, launch_stream)
        .expect("wcoj_4cycle_u64_recorded");

    assert_eq!(result.schema.column_type(0), Some(ScalarType::U64));
    assert_eq!(result.schema.column_type(1), Some(ScalarType::U64));
    assert_eq!(result.schema.column_type(2), Some(ScalarType::U64));
    assert_eq!(result.schema.column_type(3), Some(ScalarType::U64));

    let host = download_quads_u64(&result);
    let expected = cpu_4cycle_reference_u64(&edges, &edges, &edges, &edges);
    assert_eq!(host, expected);
    assert_eq!(host.len(), 4);
}

#[test]
fn wcoj_4cycle_u64_empty_inputs_produce_empty_output() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let buf = upload_binary_u64(&fix.memory, &[]);
    let launch_stream = fix.pool.acquire().expect("acquire launch_stream");
    let result = fix
        .provider
        .wcoj_4cycle_u64_recorded(&buf, &buf, &buf, &buf, launch_stream)
        .expect("empty inputs must succeed");
    assert_eq!(result.num_rows(), 0);
    assert_eq!(result.arity(), 4);
}

#[test]
fn wcoj_4cycle_u64_no_false_positives_on_open_chain() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let big = 1u64 << 35;
    let chain: Vec<(u64, u64)> = vec![(big, big + 1), (big + 1, big + 2), (big + 2, big + 3)];
    let buf_e1 = upload_binary_u64(&fix.memory, &chain);
    let buf_e2 = upload_binary_u64(&fix.memory, &chain);
    let buf_e3 = upload_binary_u64(&fix.memory, &chain);
    let buf_e4 = upload_binary_u64(&fix.memory, &[]);

    let launch_stream = fix.pool.acquire().expect("acquire launch_stream");
    let result = fix
        .provider
        .wcoj_4cycle_u64_recorded(&buf_e1, &buf_e2, &buf_e3, &buf_e4, launch_stream)
        .expect("wcoj_4cycle_u64_recorded");
    assert_eq!(result.num_rows(), 0);
}

#[test]
fn wcoj_4cycle_u64_rejects_u32_inputs() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    // Build a U32-schema buffer (the u32 helper from the u32 test)
    // — but here we manufacture one inline to keep the test crate
    // boundary clean.
    let memory = Arc::clone(&fix.memory);
    let mk_u32_buf = |rows: &[(u32, u32)]| -> CudaBuffer {
        let n = rows.len() as u32;
        let bytes_per_col = (n as usize) * std::mem::size_of::<u32>();
        let mut col0 = memory.alloc::<u8>(bytes_per_col).expect("alloc col0");
        let mut col1 = memory.alloc::<u8>(bytes_per_col).expect("alloc col1");
        let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
        let dev = memory.device().inner();
        if n > 0 {
            let c0: Vec<u8> = rows.iter().flat_map(|&(a, _)| a.to_le_bytes()).collect();
            let c1: Vec<u8> = rows.iter().flat_map(|&(_, b)| b.to_le_bytes()).collect();
            dev.htod_sync_copy_into(&c0, &mut col0).expect("htod c0");
            dev.htod_sync_copy_into(&c1, &mut col1).expect("htod c1");
        }
        dev.htod_sync_copy_into(&[n], &mut d_num_rows)
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
    };
    let buf_u32 = mk_u32_buf(&[(1, 2)]);
    let launch_stream = fix.pool.acquire().expect("acquire launch_stream");
    let err = fix.provider.wcoj_4cycle_u64_recorded(
        &buf_u32,
        &buf_u32,
        &buf_u32,
        &buf_u32,
        launch_stream,
    );
    assert!(
        err.is_err(),
        "wcoj_4cycle_u64_recorded must reject U32-schema inputs"
    );
}
