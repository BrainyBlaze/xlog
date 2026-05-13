use std::sync::Arc;

use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::{CudaBuffer, CudaColumn};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

struct DiscardSink;
impl LoggingSink for DiscardSink {
    fn emit(&self, _record: LogRecord) -> Result<(), SinkError> {
        Ok(())
    }
}

struct Fixture {
    memory: Arc<GpuMemoryManager>,
    provider: Arc<CudaKernelProvider>,
}

fn make_fixture() -> Option<Fixture> {
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
        Box::new(GlobalDeviceBudget::new(logging, 256 * 1024 * 1024));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(256 * 1024 * 1024),
        runtime,
    ));
    let provider =
        Arc::new(CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory)).ok()?);
    Some(Fixture { memory, provider })
}

fn upload_u32_keys(memory: &Arc<GpuMemoryManager>, keys: &[u32]) -> CudaBuffer {
    let n = keys.len() as u32;
    let mut col0 = memory
        .alloc::<u8>(keys.len() * std::mem::size_of::<u32>())
        .expect("alloc col0");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let bytes: Vec<u8> = keys.iter().flat_map(|v| v.to_le_bytes()).collect();
    let device = memory.device().inner();
    device
        .htod_sync_copy_into(&bytes, &mut col0)
        .expect("htod col0");
    device
        .htod_sync_copy_into(&[n], &mut d_num_rows)
        .expect("htod d_num_rows");
    let schema = Schema::new(vec![("key".to_string(), ScalarType::U32)]);
    CudaBuffer::from_columns_with_host_count(vec![col0.into()], n as u64, d_num_rows, schema, n)
}

fn upload_u64_keys(memory: &Arc<GpuMemoryManager>, keys: &[u64]) -> CudaBuffer {
    let n = keys.len() as u32;
    let mut col0 = memory
        .alloc::<u8>(keys.len() * std::mem::size_of::<u64>())
        .expect("alloc col0");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let bytes: Vec<u8> = keys.iter().flat_map(|v| v.to_le_bytes()).collect();
    let device = memory.device().inner();
    device
        .htod_sync_copy_into(&bytes, &mut col0)
        .expect("htod col0");
    device
        .htod_sync_copy_into(&[n], &mut d_num_rows)
        .expect("htod d_num_rows");
    let schema = Schema::new(vec![("key".to_string(), ScalarType::U64)]);
    CudaBuffer::from_columns_with_host_count(vec![col0.into()], n as u64, d_num_rows, schema, n)
}

fn upload_binary_u32(memory: &Arc<GpuMemoryManager>, rows: &[(u32, u32)]) -> CudaBuffer {
    let n = rows.len() as u32;
    let mut col0 = memory
        .alloc::<u8>(rows.len() * std::mem::size_of::<u32>())
        .expect("alloc col0");
    let mut col1 = memory
        .alloc::<u8>(rows.len() * std::mem::size_of::<u32>())
        .expect("alloc col1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let col0_bytes: Vec<u8> = rows.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
    let col1_bytes: Vec<u8> = rows.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
    let device = memory.device().inner();
    device
        .htod_sync_copy_into(&col0_bytes, &mut col0)
        .expect("htod col0");
    device
        .htod_sync_copy_into(&col1_bytes, &mut col1)
        .expect("htod col1");
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

fn download_count_column(memory: &Arc<GpuMemoryManager>, buffer: &CudaBuffer) -> Vec<u32> {
    let n = buffer.cached_row_count().expect("cached count") as usize;
    let mut bytes = vec![0u8; n * std::mem::size_of::<u32>()];
    let CudaColumn::Owned(col) = buffer.column(0).expect("count column") else {
        panic!("count column must be owned");
    };
    memory
        .device()
        .inner()
        .dtoh_sync_copy_into(col, &mut bytes)
        .expect("dtoh count column");
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("u32 bytes")))
        .collect()
}

fn download_triples(memory: &Arc<GpuMemoryManager>, buffer: &CudaBuffer) -> Vec<(u32, u32, u32)> {
    let n = buffer.cached_row_count().expect("cached count") as usize;
    let mut cols = [Vec::new(), Vec::new(), Vec::new()];
    for (idx, out) in cols.iter_mut().enumerate() {
        out.resize(n * std::mem::size_of::<u32>(), 0);
        let CudaColumn::Owned(col) = buffer.column(idx).expect("output column") else {
            panic!("output column must be owned");
        };
        memory
            .device()
            .inner()
            .dtoh_sync_copy_into(col, out)
            .expect("dtoh output column");
    }
    (0..n)
        .map(|i| {
            let read = |bytes: &[u8]| {
                let start = i * 4;
                u32::from_le_bytes(bytes[start..start + 4].try_into().expect("u32 bytes"))
            };
            (read(&cols[0]), read(&cols[1]), read(&cols[2]))
        })
        .collect()
}

#[test]
fn wcoj_metadata_u32_builds_unique_fanout_prefix() {
    let Some(fix) = make_fixture() else {
        eprintln!("skipping WCOJ metadata u32 test: no CUDA device");
        return;
    };
    let input = upload_u32_keys(&fix.memory, &[1, 1, 1, 2, 4, 4, 9]);
    let metadata = fix
        .provider
        .wcoj_build_metadata_u32_recorded(&input, 0, xlog_cuda::device_runtime::StreamId::DEFAULT)
        .expect("metadata");

    assert_eq!(metadata.key_count, 4);
    assert_eq!(metadata.row_count, 7);
    assert_eq!(metadata.total, 7);

    let mut keys = vec![0u32; metadata.key_count as usize];
    let mut fan_out = vec![0u32; metadata.key_count as usize];
    let mut prefix = vec![0u32; metadata.key_count as usize];
    let device = fix.memory.device().inner();
    device
        .dtoh_sync_copy_into(&metadata.unique_keys, &mut keys)
        .expect("dtoh unique");
    device
        .dtoh_sync_copy_into(&metadata.fan_out, &mut fan_out)
        .expect("dtoh fan_out");
    device
        .dtoh_sync_copy_into(&metadata.prefix_sum, &mut prefix)
        .expect("dtoh prefix");

    assert_eq!(keys, vec![1, 2, 4, 9]);
    assert_eq!(fan_out, vec![3, 1, 2, 1]);
    assert_eq!(prefix, vec![0, 3, 4, 6]);
}

#[test]
fn wcoj_metadata_u64_builds_unique_fanout_prefix() {
    let Some(fix) = make_fixture() else {
        eprintln!("skipping WCOJ metadata u64 test: no CUDA device");
        return;
    };
    let input = upload_u64_keys(&fix.memory, &[10, 10, 11, 13, 13, 13]);
    let metadata = fix
        .provider
        .wcoj_build_metadata_u64_recorded(&input, 0, xlog_cuda::device_runtime::StreamId::DEFAULT)
        .expect("metadata");

    assert_eq!(metadata.key_count, 3);
    assert_eq!(metadata.row_count, 6);
    assert_eq!(metadata.total, 6);

    let mut keys = vec![0u64; metadata.key_count as usize];
    let mut fan_out = vec![0u32; metadata.key_count as usize];
    let mut prefix = vec![0u32; metadata.key_count as usize];
    let device = fix.memory.device().inner();
    device
        .dtoh_sync_copy_into(&metadata.unique_keys, &mut keys)
        .expect("dtoh unique");
    device
        .dtoh_sync_copy_into(&metadata.fan_out, &mut fan_out)
        .expect("dtoh fan_out");
    device
        .dtoh_sync_copy_into(&metadata.prefix_sum, &mut prefix)
        .expect("dtoh prefix");

    assert_eq!(keys, vec![10, 11, 13]);
    assert_eq!(fan_out, vec![2, 1, 3]);
    assert_eq!(prefix, vec![0, 2, 3]);
}

#[test]
fn wcoj_triangle_hg_count_matches_cpu_total() {
    let Some(fix) = make_fixture() else {
        eprintln!("skipping WCOJ HG count test: no CUDA device");
        return;
    };
    let e_xy = upload_binary_u32(&fix.memory, &[(1, 2), (1, 3), (2, 2), (3, 4)]);
    let e_yz = upload_binary_u32(&fix.memory, &[(2, 5), (2, 6), (3, 6), (4, 9)]);
    let e_xz = upload_binary_u32(&fix.memory, &[(1, 5), (1, 6), (2, 5), (3, 9)]);

    let plan = fix
        .provider
        .wcoj_triangle_hg_work_plan_u32_recorded(
            &e_xy,
            &e_yz,
            &e_xz,
            2,
            xlog_cuda::device_runtime::StreamId::DEFAULT,
        )
        .expect("work plan");
    assert_eq!(plan.total_work, 5);
    assert_eq!(plan.row_count, 4);

    let counts = fix
        .provider
        .wcoj_triangle_count_hg_u32_recorded(
            &e_yz,
            &e_xz,
            &plan,
            xlog_cuda::device_runtime::StreamId::DEFAULT,
        )
        .expect("hg count");
    let block_counts = download_count_column(&fix.memory, &counts);
    assert_eq!(block_counts.iter().sum::<u32>(), 5);
}

#[test]
fn wcoj_triangle_hg_materializes_expected_rows() {
    let Some(fix) = make_fixture() else {
        eprintln!("skipping WCOJ HG materialize test: no CUDA device");
        return;
    };
    let e_xy = upload_binary_u32(&fix.memory, &[(1, 2), (1, 3), (2, 2), (3, 4)]);
    let e_yz = upload_binary_u32(&fix.memory, &[(2, 5), (2, 6), (3, 6), (4, 9)]);
    let e_xz = upload_binary_u32(&fix.memory, &[(1, 5), (1, 6), (2, 5), (3, 9)]);

    let output = fix
        .provider
        .wcoj_triangle_hg_u32_recorded(
            &e_xy,
            &e_yz,
            &e_xz,
            2,
            xlog_cuda::device_runtime::StreamId::DEFAULT,
        )
        .expect("hg triangle");
    let mut rows = download_triples(&fix.memory, &output);
    rows.sort();
    assert_eq!(
        rows,
        vec![(1, 2, 5), (1, 2, 6), (1, 3, 6), (2, 2, 5), (3, 4, 9)]
    );
}
