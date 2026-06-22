// crates/xlog-integration/tests/test_same_generation_gpu.rs
//! Same Generation GPU WCOJ dispatch and output parity.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use cudarc::driver::sys;
use xlog_core::{MemoryBudget, RuntimeConfig, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::CudaBuffer;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_logic::Compiler;
use xlog_runtime::Executor;

struct DiscardSink;
impl LoggingSink for DiscardSink {
    fn emit(&self, _record: LogRecord) -> Result<(), SinkError> {
        Ok(())
    }
}

#[allow(dead_code)]
struct RuntimeBackedFixture {
    device: Arc<CudaDevice>,
    runtime: Arc<XlogDeviceRuntime>,
    memory: Arc<GpuMemoryManager>,
    provider: Arc<CudaKernelProvider>,
    pool: Arc<StreamPool>,
}

fn make_runtime_backed_fixture() -> Option<RuntimeBackedFixture> {
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
        Arc::new(CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory)).ok()?);
    Some(RuntimeBackedFixture {
        device,
        runtime,
        memory,
        provider,
        pool,
    })
}

fn upload_binary_u32(memory: &Arc<GpuMemoryManager>, rows: &[(u32, u32)]) -> CudaBuffer {
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
        ("c0".to_string(), ScalarType::U32),
        ("c1".to_string(), ScalarType::U32),
    ]);
    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into()],
        n as u64,
        d_num_rows,
        schema,
        n,
    )
}

fn download_pairs(buf: &CudaBuffer) -> BTreeSet<(u32, u32)> {
    let n = match buf.cached_row_count() {
        Some(c) => c as usize,
        None => {
            let mut count_host = [0u32; 1];
            unsafe {
                sys::cuMemcpyDtoH_v2(
                    count_host.as_mut_ptr() as *mut _,
                    *buf.num_rows_device().device_ptr(),
                    std::mem::size_of::<u32>(),
                );
            }
            count_host[0] as usize
        }
    };
    assert_eq!(buf.arity(), 2);
    if n == 0 {
        return BTreeSet::new();
    }

    let mut col0_bytes = vec![0u8; n * 4];
    let mut col1_bytes = vec![0u8; n * 4];
    unsafe {
        sys::cuMemcpyDtoH_v2(
            col0_bytes.as_mut_ptr() as *mut _,
            *buf.column(0).unwrap().device_ptr(),
            col0_bytes.len(),
        );
        sys::cuMemcpyDtoH_v2(
            col1_bytes.as_mut_ptr() as *mut _,
            *buf.column(1).unwrap().device_ptr(),
            col1_bytes.len(),
        );
    }

    (0..n)
        .map(|i| {
            (
                u32::from_le_bytes(col0_bytes[i * 4..i * 4 + 4].try_into().unwrap()),
                u32::from_le_bytes(col1_bytes[i * 4..i * 4 + 4].try_into().unwrap()),
            )
        })
        .collect()
}

fn run_program(
    provider: Arc<CudaKernelProvider>,
    memory: &Arc<GpuMemoryManager>,
    config: RuntimeConfig,
    source: &str,
    inputs: &BTreeMap<&str, Vec<(u32, u32)>>,
) -> Executor {
    let mut compiler = Compiler::new();
    let plan = compiler.compile(source).expect("compile");
    let mut executor = Executor::new_with_config(provider, config);
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    for (name, rows) in inputs {
        executor.put_relation(name, upload_binary_u32(memory, rows));
    }
    executor.execute_plan(&plan).expect("execute_plan");
    executor
}

const SAME_GENERATION_GPU_PROGRAM: &str = r#"
    pred parent(u32, u32).
    pred parent_rev(u32, u32).
    pred all_child_pairs(u32, u32).
    pred sg(u32, u32).
    pred sg_witness(u32, u32, u32, u32).
    pred sg_result(u32, u32).

    sg(X, Y) :- parent(X, P), parent(Y, P).
    sg(X, Y) :- parent(X, A), sg(A, B), parent(Y, B).
    sg_result(X, Y) :- parent(X, P), parent(Y, P).
    sg_witness(W, X, Y, Z) :- parent(W, X), sg(X, Y), parent_rev(Y, Z), all_child_pairs(Z, W).
    sg_result(W, Z) :- sg_witness(W, X, Y, Z).
"#;

fn sg_reference(parent_edges: &[(u32, u32)]) -> BTreeSet<(u32, u32)> {
    let mut sg = BTreeSet::new();

    for (x, p_x) in parent_edges {
        for (y, p_y) in parent_edges {
            if p_x == p_y {
                sg.insert((*x, *y));
            }
        }
    }

    loop {
        let snapshot: Vec<(u32, u32)> = sg.iter().copied().collect();
        let before_len = sg.len();
        for (a, b) in &snapshot {
            for (x, p_x) in parent_edges {
                if p_x != a {
                    continue;
                }
                for (y, p_y) in parent_edges {
                    if p_y != b {
                        continue;
                    }
                    sg.insert((*x, *y));
                }
            }
        }
        if sg.len() == before_len {
            break;
        }
    }

    sg
}

fn parent_pairs() -> Vec<(u32, u32)> {
    vec![(1, 10), (2, 10), (11, 1), (12, 2), (13, 1), (14, 12)]
}

fn same_generation_inputs() -> BTreeMap<&'static str, Vec<(u32, u32)>> {
    let parents = parent_pairs();
    let mut child_ids: Vec<u32> = parents.iter().map(|(child, _)| *child).collect();
    child_ids.sort();

    let mut all_child_pairs = Vec::new();
    for z in &child_ids {
        for w in &child_ids {
            all_child_pairs.push((*z, *w));
        }
    }

    let mut inputs = BTreeMap::new();
    inputs.insert("parent", parents.clone());
    inputs.insert(
        "parent_rev",
        parents
            .iter()
            .map(|(child, parent)| (*parent, *child))
            .collect(),
    );
    inputs.insert("all_child_pairs", all_child_pairs);
    inputs
}

#[test]
fn same_generation_gpu_4cycle_witness_matches_cpu_oracle() {
    let cpu_rows = sg_reference(&parent_pairs());
    assert!(
        !cpu_rows.is_empty(),
        "Same Generation CPU oracle must be non-empty"
    );
    assert_eq!(cpu_rows.len(), 14, "locked Same Generation row-set size");

    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = same_generation_inputs();
    let executor = run_program(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default().with_wcoj_4cycle_dispatch(Some(true)),
        SAME_GENERATION_GPU_PROGRAM,
        &inputs,
    );

    let four_counter = executor.wcoj_4cycle_dispatch_count();
    eprintln!("Same Generation GPU measured wcoj_4cycle_dispatch_count={four_counter}");
    assert_eq!(
        four_counter, 1,
        "Same Generation 4-cycle witness must dispatch exactly once"
    );
    assert_eq!(
        executor.wcoj_triangle_dispatch_count(),
        0,
        "Same Generation GPU test must not dispatch the triangle path"
    );

    let gpu_rows = download_pairs(executor.store().get("sg_result").expect("sg_result"));
    eprintln!(
        "Same Generation GPU measured row_set_size={}",
        gpu_rows.len()
    );
    assert_eq!(
        gpu_rows, cpu_rows,
        "GPU Same Generation output must match the CPU oracle"
    );
}
