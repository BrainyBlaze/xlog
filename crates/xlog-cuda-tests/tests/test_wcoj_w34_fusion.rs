//! W3.4 production layout+count fusion dispatch certs.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::sys;
use xlog_core::{
    CostModelKind, MemoryBudget, RuntimeConfig, ScalarType, Schema, ENV_WCOJ_W34_THRESHOLD,
    W34_FUSION_THRESHOLD,
};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::CudaBuffer;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_logic::Compiler;
use xlog_runtime::Executor;

const TRIANGLE_SOURCE: &str = "tri(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z).";

struct DiscardSink;
impl LoggingSink for DiscardSink {
    fn emit(&self, _record: LogRecord) -> Result<(), SinkError> {
        Ok(())
    }
}

#[allow(dead_code)]
struct Fixture {
    device: Arc<CudaDevice>,
    runtime: Arc<XlogDeviceRuntime>,
    memory: Arc<GpuMemoryManager>,
    provider: Arc<CudaKernelProvider>,
    pool: Arc<StreamPool>,
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
        Arc::clone(&runtime),
    ));
    let provider =
        Arc::new(CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory)).ok()?);
    Some(Fixture {
        device,
        runtime,
        memory,
        provider,
        pool,
    })
}

#[inline]
fn lcg_next(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    *state
}

fn superhub_pairs_xy(seed: u64, rows: u32, key_range: u32, hub_y: u32) -> Vec<(u32, u32)> {
    let mut state = seed;
    (0..rows)
        .map(|i| {
            let a = (lcg_next(&mut state) % key_range as u64) as u32;
            let b = if i.is_multiple_of(2) {
                hub_y
            } else {
                (lcg_next(&mut state) % key_range as u64) as u32
            };
            (a, b)
        })
        .collect()
}

fn superhub_pairs_first(seed: u64, rows: u32, key_range: u32, hub_first: u32) -> Vec<(u32, u32)> {
    let mut state = seed;
    (0..rows)
        .map(|i| {
            let a = if i.is_multiple_of(2) {
                hub_first
            } else {
                (lcg_next(&mut state) % key_range as u64) as u32
            };
            let b = (lcg_next(&mut state) % key_range as u64) as u32;
            (a, b)
        })
        .collect()
}

fn dedup_pairs(mut rows: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    rows.sort();
    rows.dedup();
    rows
}

fn superhub_fixture(rows: u32) -> BTreeMap<&'static str, Vec<(u32, u32)>> {
    let key_range = (rows / 10).max(1000);
    let hub_y = 7;
    let hub_x = 13;
    let mut m = BTreeMap::new();
    m.insert(
        "e1",
        dedup_pairs(superhub_pairs_xy(101, rows, key_range, hub_y)),
    );
    m.insert(
        "e2",
        dedup_pairs(superhub_pairs_first(202, rows, key_range, hub_y)),
    );
    m.insert(
        "e3",
        dedup_pairs(superhub_pairs_first(303, rows, key_range, hub_x)),
    );
    m
}

fn total_input_rows(inputs: &BTreeMap<&str, Vec<(u32, u32)>>) -> u64 {
    inputs.values().map(|v| v.len() as u64).sum()
}

fn upload_binary_u32(memory: &Arc<GpuMemoryManager>, rows: &[(u32, u32)]) -> CudaBuffer {
    let n = rows.len() as u32;
    let bytes_per_col = (n as usize) * std::mem::size_of::<u32>();
    let mut col0 = memory.alloc::<u8>(bytes_per_col).expect("alloc col0");
    let mut col1 = memory.alloc::<u8>(bytes_per_col).expect("alloc col1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let device = memory.device().inner();
    if n > 0 {
        let col0_bytes: Vec<u8> = rows.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
        let col1_bytes: Vec<u8> = rows.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
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

fn download_triples(buf: &CudaBuffer) -> Vec<(u32, u32, u32)> {
    let n = buf.cached_row_count().unwrap_or(buf.num_rows() as u32) as usize;
    if n == 0 {
        return Vec::new();
    }
    let mut col0_bytes = vec![0u8; n * 4];
    let mut col1_bytes = vec![0u8; n * 4];
    let mut col2_bytes = vec![0u8; n * 4];
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
        let res2 = sys::cuMemcpyDtoH_v2(
            col2_bytes.as_mut_ptr() as *mut _,
            *buf.column(2).unwrap().device_ptr(),
            col2_bytes.len(),
        );
        assert_eq!(res0, sys::cudaError_enum::CUDA_SUCCESS);
        assert_eq!(res1, sys::cudaError_enum::CUDA_SUCCESS);
        assert_eq!(res2, sys::cudaError_enum::CUDA_SUCCESS);
    }
    let mut out: Vec<(u32, u32, u32)> = (0..n)
        .map(|i| {
            (
                u32::from_le_bytes(col0_bytes[i * 4..i * 4 + 4].try_into().unwrap()),
                u32::from_le_bytes(col1_bytes[i * 4..i * 4 + 4].try_into().unwrap()),
                u32::from_le_bytes(col2_bytes[i * 4..i * 4 + 4].try_into().unwrap()),
            )
        })
        .collect();
    out.sort();
    out.dedup();
    out
}

#[derive(Debug)]
struct RunResult {
    rows: Vec<(u32, u32, u32)>,
    fused_count: u64,
    unfused_count: u64,
    triangle_count: u64,
    total_input_rows: u64,
}

fn run_program(
    inputs: &BTreeMap<&str, Vec<(u32, u32)>>,
    config: RuntimeConfig,
) -> Option<RunResult> {
    let fix = make_fixture()?;
    let mut compiler = Compiler::new();
    let plan = compiler.compile(TRIANGLE_SOURCE).expect("compile");
    let mut executor = Executor::new_with_config(Arc::clone(&fix.provider), config);
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    for (name, rows) in inputs {
        executor.put_relation(name, upload_binary_u32(&fix.memory, rows));
    }
    let _ = executor.execute_plan(&plan).expect("execute_plan");
    let tri = executor.store().get("tri").expect("tri present");
    Some(RunResult {
        rows: download_triples(tri),
        fused_count: fix.provider.wcoj_triangle_fused_dispatch_count(),
        unfused_count: fix.provider.wcoj_triangle_unfused_dispatch_count(),
        triangle_count: executor.wcoj_triangle_dispatch_count(),
        total_input_rows: total_input_rows(inputs),
    })
}

fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

struct EnvGuard {
    key: &'static str,
    prev: Option<OsString>,
}

impl EnvGuard {
    fn set(key: &'static str, value: Option<&str>) -> Self {
        let prev = std::env::var_os(key);
        match value {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }
        Self { key, prev }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        match &self.prev {
            Some(v) => std::env::set_var(self.key, v),
            None => std::env::remove_var(self.key),
        }
    }
}

fn with_w34_threshold<T>(value: Option<&str>, f: impl FnOnce() -> T) -> T {
    let _guard = env_lock().lock().expect("env lock");
    let _env = EnvGuard::set(ENV_WCOJ_W34_THRESHOLD, value);
    f()
}

#[test]
fn above_threshold_routes_to_fused_and_matches_reference() {
    let inputs = superhub_fixture(6_000);
    let Some(reference) = with_w34_threshold(Some(&u32::MAX.to_string()), || {
        run_program(
            &inputs,
            RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(true)),
        )
    }) else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let tested = with_w34_threshold(None, || {
        run_program(
            &inputs,
            RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(true)),
        )
        .expect("CUDA runtime available after reference run")
    });
    assert!(
        tested.total_input_rows >= u64::from(W34_FUSION_THRESHOLD),
        "fixture must be above threshold: rows={} threshold={}",
        tested.total_input_rows,
        W34_FUSION_THRESHOLD
    );
    assert_eq!(tested.rows, reference.rows);
    assert_eq!(tested.triangle_count, 1);
    assert_eq!(tested.fused_count, 1);
    assert_eq!(tested.unfused_count, 0);
}

#[test]
fn below_threshold_routes_to_unfused_and_matches_reference() {
    let inputs = superhub_fixture(1_024);
    let Some(reference) = with_w34_threshold(None, || {
        run_program(
            &inputs,
            RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(false)),
        )
    }) else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let tested = with_w34_threshold(None, || {
        run_program(
            &inputs,
            RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(true)),
        )
        .expect("CUDA runtime available after reference run")
    });
    assert!(
        tested.total_input_rows < u64::from(W34_FUSION_THRESHOLD),
        "fixture must be below threshold: rows={} threshold={}",
        tested.total_input_rows,
        W34_FUSION_THRESHOLD
    );
    assert_eq!(tested.rows, reference.rows);
    assert_eq!(tested.triangle_count, 1);
    assert_eq!(tested.fused_count, 0);
    assert_eq!(tested.unfused_count, 1);
}

#[test]
fn env_override_can_force_unfused_on_large_fixture() {
    let inputs = superhub_fixture(6_000);
    let Some(tested) = with_w34_threshold(Some(&u32::MAX.to_string()), || {
        run_program(
            &inputs,
            RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(true)),
        )
    }) else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    assert_eq!(tested.triangle_count, 1);
    assert_eq!(tested.fused_count, 0);
    assert_eq!(tested.unfused_count, 1);
}

#[test]
fn env_override_can_force_fused_on_small_fixture() {
    let inputs = superhub_fixture(1_024);
    let Some(reference) = with_w34_threshold(None, || {
        run_program(
            &inputs,
            RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(false)),
        )
    }) else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let tested = with_w34_threshold(Some("0"), || {
        run_program(
            &inputs,
            RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(true)),
        )
        .expect("CUDA runtime available after reference run")
    });
    assert_eq!(tested.rows, reference.rows);
    assert_eq!(tested.triangle_count, 1);
    assert_eq!(tested.fused_count, 1);
    assert_eq!(tested.unfused_count, 0);
}

#[test]
fn default_threshold_and_w25_prior_stay_locked() {
    assert_eq!(W34_FUSION_THRESHOLD, 4_096);
    assert_eq!(
        RuntimeConfig::default().resolved_wcoj_cost_model(),
        CostModelKind::Cardinality
    );
}
