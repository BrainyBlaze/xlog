// crates/xlog-integration/tests/test_w51_skewed_multiway_gpu_cert.rs
//! W5.1 skewed multiway GPU certification.

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
use xlog_logic::ast::{Atom, BodyLiteral, Rule, Term};
use xlog_logic::hypergraph::{
    evaluate_rule_typed, AppearanceOrder, RefRelation, RefRelationStore, RefValue,
};
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

fn download_triples(buf: &CudaBuffer) -> BTreeSet<(u32, u32, u32)> {
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
    assert_eq!(buf.arity(), 3);
    if n == 0 {
        return BTreeSet::new();
    }

    let mut col0_bytes = vec![0u8; n * 4];
    let mut col1_bytes = vec![0u8; n * 4];
    let mut col2_bytes = vec![0u8; n * 4];
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
        sys::cuMemcpyDtoH_v2(
            col2_bytes.as_mut_ptr() as *mut _,
            *buf.column(2).unwrap().device_ptr(),
            col2_bytes.len(),
        );
    }

    (0..n)
        .map(|i| {
            (
                u32::from_le_bytes(col0_bytes[i * 4..i * 4 + 4].try_into().unwrap()),
                u32::from_le_bytes(col1_bytes[i * 4..i * 4 + 4].try_into().unwrap()),
                u32::from_le_bytes(col2_bytes[i * 4..i * 4 + 4].try_into().unwrap()),
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

const SKEWED_MULTIWAY_GPU_CERT: &str = r#"
    pred big(u32, u32).
    pred small_a(u32, u32).
    pred small_b(u32, u32).
    pred result(u32, u32, u32).

    result(X, Y, Z) :- big(X, Y), small_a(Y, Z), small_b(X, Z).
"#;

fn var(name: &str) -> Term {
    Term::Variable(name.to_string())
}

fn atom(predicate: &str, terms: Vec<Term>) -> Atom {
    Atom {
        predicate: predicate.to_string(),
        terms,
    }
}

fn pos(predicate: &str, terms: Vec<Term>) -> BodyLiteral {
    BodyLiteral::Positive(atom(predicate, terms))
}

fn rule_with(head: Atom, body: Vec<BodyLiteral>) -> Rule {
    Rule { head, body }
}

fn skewed_rule() -> Rule {
    rule_with(
        atom("result", vec![var("X"), var("Y"), var("Z")]),
        vec![
            pos("big", vec![var("X"), var("Y")]),
            pos("small_a", vec![var("Y"), var("Z")]),
            pos("small_b", vec![var("X"), var("Z")]),
        ],
    )
}

fn big_pairs() -> Vec<(u32, u32)> {
    (1u32..=8)
        .flat_map(|x| (1u32..=8).filter(move |y| *y != x).map(move |y| (x, y)))
        .collect()
}

fn small_a_pairs() -> Vec<(u32, u32)> {
    vec![(2, 10), (3, 20), (4, 30), (5, 40)]
}

fn small_b_pairs() -> Vec<(u32, u32)> {
    vec![(1, 10), (2, 20), (3, 30), (4, 40)]
}

fn u32_pair_relation(rows: &[(u32, u32)]) -> RefRelation {
    RefRelation {
        schema: vec![ScalarType::U32, ScalarType::U32],
        rows: rows
            .iter()
            .map(|(a, b)| vec![RefValue::U32(*a), RefValue::U32(*b)])
            .collect(),
    }
}

fn skewed_store() -> RefRelationStore {
    let mut store = BTreeMap::new();
    store.insert("big".to_string(), u32_pair_relation(&big_pairs()));
    store.insert("small_a".to_string(), u32_pair_relation(&small_a_pairs()));
    store.insert("small_b".to_string(), u32_pair_relation(&small_b_pairs()));
    store
}

fn skewed_inputs() -> BTreeMap<&'static str, Vec<(u32, u32)>> {
    let mut inputs = BTreeMap::new();
    inputs.insert("big", big_pairs());
    inputs.insert("small_a", small_a_pairs());
    inputs.insert("small_b", small_b_pairs());
    inputs
}

fn cpu_oracle_rows() -> BTreeSet<(u32, u32, u32)> {
    let rows = evaluate_rule_typed(&skewed_rule(), &skewed_store(), &AppearanceOrder)
        .expect("skewed eval");
    rows.iter()
        .map(|row| match row.as_slice() {
            [RefValue::U32(x), RefValue::U32(y), RefValue::U32(z)] => (*x, *y, *z),
            other => panic!("unexpected CPU oracle row: {other:?}"),
        })
        .collect()
}

#[test]
fn skewed_multiway_gpu_triangle_matches_typed_cpu_oracle() {
    let cpu_rows = cpu_oracle_rows();
    let expected = BTreeSet::from([(1, 2, 10), (2, 3, 20), (3, 4, 30), (4, 5, 40)]);
    assert_eq!(
        cpu_rows, expected,
        "typed CPU oracle must match the locked skewed fixture"
    );
    assert!(
        !cpu_rows.is_empty(),
        "skewed multiway CPU oracle must be non-empty"
    );
    assert_eq!(cpu_rows.len(), 4, "locked skewed multiway row-set size");

    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let inputs = skewed_inputs();
    let executor = run_program(
        Arc::clone(&fix.provider),
        &fix.memory,
        RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(true)),
        SKEWED_MULTIWAY_GPU_CERT,
        &inputs,
    );

    let triangle_counter = executor.wcoj_triangle_dispatch_count();
    let fourcycle_counter = executor.wcoj_4cycle_dispatch_count();
    let clique5_counter = executor.wcoj_clique5_dispatch_count();
    let clique6_counter = executor.wcoj_clique6_dispatch_count();
    eprintln!("Skewed multiway W5.1 measured wcoj_triangle_dispatch_count={triangle_counter}");
    eprintln!(
        "Skewed multiway W5.1 measured wcoj_4cycle_dispatch_count={fourcycle_counter}, \
         wcoj_clique5_dispatch_count={clique5_counter}, \
         wcoj_clique6_dispatch_count={clique6_counter}"
    );
    assert_eq!(
        triangle_counter, 1,
        "skewed multiway triangle cert must dispatch exactly once"
    );
    assert_eq!(
        fourcycle_counter, 0,
        "skewed multiway cert must not dispatch the 4-cycle path"
    );
    assert_eq!(
        clique5_counter, 0,
        "skewed multiway cert must not dispatch the clique5 path"
    );
    assert_eq!(
        clique6_counter, 0,
        "skewed multiway cert must not dispatch the clique6 path"
    );

    let gpu_rows = download_triples(executor.store().get("result").expect("result"));
    eprintln!(
        "Skewed multiway W5.1 measured row_set_size={}",
        gpu_rows.len()
    );
    assert_eq!(
        gpu_rows, cpu_rows,
        "skewed multiway GPU output must match the typed CPU oracle"
    );
}
