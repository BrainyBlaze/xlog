use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cudarc::driver::result::mem_get_info;
use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamId, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::CudaBuffer;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

const DEVICE_BUDGET_BYTES: u64 = 64 * 1024 * 1024;
const W52_PILOT_VRAM_GATE_BYTES: u64 = 512 * 1024 * 1024;
const CLIQUE5_EDGE_NAMES: [(&str, &str); 10] = [
    ("v0", "v1"),
    ("v0", "v2"),
    ("v0", "v3"),
    ("v0", "v4"),
    ("v1", "v2"),
    ("v1", "v3"),
    ("v1", "v4"),
    ("v2", "v3"),
    ("v2", "v4"),
    ("v3", "v4"),
];

struct DiscardSink;

impl LoggingSink for DiscardSink {
    fn emit(&self, _record: LogRecord) -> Result<(), SinkError> {
        Ok(())
    }
}

#[allow(dead_code)]
struct Provider {
    _device: Arc<CudaDevice>,
    _runtime: Arc<XlogDeviceRuntime>,
    memory: Arc<GpuMemoryManager>,
    provider: Arc<CudaKernelProvider>,
    pool: Arc<StreamPool>,
    launch_stream: StreamId,
}

fn make_provider() -> Option<Provider> {
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
    let async_resource: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(
        AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool)),
    );
    let logging: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(LoggingResource::new(
        async_resource,
        Arc::new(DiscardSink) as Arc<dyn LoggingSink>,
    ));
    let budget: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(GlobalDeviceBudget::new(
        logging,
        DEVICE_BUDGET_BYTES as usize,
    ));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(DEVICE_BUDGET_BYTES),
        Arc::clone(&runtime),
    ));
    let provider =
        Arc::new(CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory)).ok()?);
    let launch_stream = pool.acquire().ok()?;
    Some(Provider {
        _device: device,
        _runtime: runtime,
        memory,
        provider,
        pool,
        launch_stream,
    })
}

fn sync_launch_stream(prov: &Provider) {
    prov.pool
        .resolve(prov.launch_stream)
        .expect("resolve WCOJ launch stream")
        .synchronize()
        .expect("sync WCOJ launch stream");
}

fn upload_2col_u32(
    memory: &Arc<GpuMemoryManager>,
    col0_name: &str,
    col1_name: &str,
    rows: &[(u32, u32)],
) -> CudaBuffer {
    let n = rows.len() as u32;
    let bytes_per_col = (n as usize) * std::mem::size_of::<u32>();
    let mut col0 = memory.alloc::<u8>(bytes_per_col).expect("alloc col0");
    let mut col1 = memory.alloc::<u8>(bytes_per_col).expect("alloc col1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let dev = memory.device().inner();
    if n > 0 {
        let b0: Vec<u8> = rows.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
        let b1: Vec<u8> = rows.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
        dev.htod_sync_copy_into(&b0, &mut col0).expect("htod c0");
        dev.htod_sync_copy_into(&b1, &mut col1).expect("htod c1");
    }
    dev.htod_sync_copy_into(&[n], &mut d_num_rows)
        .expect("htod n");
    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into()],
        n as u64,
        d_num_rows,
        Schema::new(vec![
            (col0_name.to_string(), ScalarType::U32),
            (col1_name.to_string(), ScalarType::U32),
        ]),
        n,
    )
}

fn diagonal_k5_fixture(n: u32) -> [Vec<(u32, u32)>; 10] {
    std::array::from_fn(|_| (1..=n).map(|i| (i, i)).collect())
}

fn upload_clique5_fixture(prov: &Provider, rows: &[Vec<(u32, u32)>; 10]) -> [CudaBuffer; 10] {
    std::array::from_fn(|idx| {
        let (left, right) = CLIQUE5_EDGE_NAMES[idx];
        upload_2col_u32(&prov.memory, left, right, &rows[idx])
    })
}

fn gpu_wcoj_clique5_path(prov: &Provider, inputs: &[CudaBuffer; 10]) -> CudaBuffer {
    let laid_out: Vec<CudaBuffer> = inputs
        .iter()
        .enumerate()
        .map(|(idx, input)| {
            prov.provider
                .wcoj_layout_sort_u32_recorded(input, prov.launch_stream)
                .unwrap_or_else(|e| panic!("layout-sort clique5 edge {idx}: {e}"))
        })
        .collect();
    let edge_refs: [&CudaBuffer; 10] = [
        &laid_out[0],
        &laid_out[1],
        &laid_out[2],
        &laid_out[3],
        &laid_out[4],
        &laid_out[5],
        &laid_out[6],
        &laid_out[7],
        &laid_out[8],
        &laid_out[9],
    ];
    let out = prov
        .provider
        .wcoj_clique5_u32_recorded(&edge_refs, prov.launch_stream)
        .expect("wcoj clique5");
    sync_launch_stream(prov);
    out
}

fn download_u32_rows(
    buf: &CudaBuffer,
    provider: &CudaKernelProvider,
    arity: usize,
) -> BTreeSet<Vec<u32>> {
    assert_eq!(buf.arity(), arity, "download arity mismatch");
    let n = provider.device_row_count(buf).expect("row count");
    let cols: Vec<Vec<u32>> = (0..arity)
        .map(|i| {
            provider
                .download_column_untracked::<u32>(buf, i)
                .expect("download col")
        })
        .collect();
    (0..n)
        .map(|row| (0..arity).map(|col| cols[col][row]).collect())
        .collect()
}

fn expected_diagonal_k5_rows(n: u32) -> BTreeSet<Vec<u32>> {
    (1..=n).map(|i| vec![i, i, i, i, i]).collect()
}

#[test]
fn w52_kclique_pilot_records_measured_elapsed_and_vram_delta() {
    let Some(prov) = make_provider() else {
        eprintln!("Skipping W5.2 measurement pilot: CUDA runtime unavailable");
        return;
    };
    let Ok((free_before, total_before)) = mem_get_info() else {
        eprintln!("Skipping W5.2 measurement pilot: CUDA mem_get_info unavailable");
        return;
    };

    let n = 4;
    let rows = diagonal_k5_fixture(n);
    let inputs = upload_clique5_fixture(&prov, &rows);
    prov.provider.reset_kclique_metadata_build_metrics();

    let start = Instant::now();
    let out = gpu_wcoj_clique5_path(&prov, &inputs);
    let elapsed = start.elapsed();
    let Ok((free_after, total_after)) = mem_get_info() else {
        eprintln!("Skipping W5.2 measurement pilot: CUDA mem_get_info unavailable after launch");
        return;
    };

    assert!(
        elapsed > Duration::ZERO,
        "W5.2 pilot must report a measured elapsed duration"
    );
    assert_eq!(
        total_before, total_after,
        "CUDA total memory should be stable across the W5.2 pilot"
    );
    let vram_delta = (free_before as u64).saturating_sub(free_after as u64);
    assert!(
        vram_delta <= W52_PILOT_VRAM_GATE_BYTES,
        "W5.2 pilot VRAM delta {vram_delta} exceeds gate {W52_PILOT_VRAM_GATE_BYTES}"
    );

    let observed = download_u32_rows(&out, &prov.provider, 5);
    assert_eq!(observed, expected_diagonal_k5_rows(n));
    assert!(
        prov.provider.kclique_metadata_build_count() >= 1,
        "W5.2 K-clique pilot must build K-clique metadata"
    );
    eprintln!(
        "W52_MEASUREMENT_PILOT workload=5clique N={n} elapsed_nanos={} vram_delta_bytes={} total_bytes={} metadata_build_count={} metadata_build_nanos={}",
        elapsed.as_nanos(),
        vram_delta,
        total_after as u64,
        prov.provider.kclique_metadata_build_count(),
        prov.provider.kclique_metadata_build_nanos()
    );
}
