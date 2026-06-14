#![allow(clippy::arc_with_non_send_sync)]

//! Cross-mode determinism harness for WCOJ, binary-join fallback, recursive,
//! and dynamic-rule-injection execution on a shared fixture.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::sys;
use xlog_core::{MemoryBudget, RelId, RuntimeConfig, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::{CudaBuffer, CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_ir::ExecutionPlan;
use xlog_logic::Compiler;
use xlog_runtime::Executor;

const FIXED_SEED: u64 = 0x573;
const ITERATIONS: usize = 100;

const CROSS_MODE_SOURCE: &str = r#"
    pred e1(u32, u32).
    pred e2(u32, u32).
    pred e3(u32, u32).
    pred tri(u32, u32, u32).
    pred chain(u32, u32).
    pred path(u32, u32).

    tri(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z).
    chain(X, Z) :- e1(X, Y), e2(Y, Z).
    path(X, Y) :- e1(X, Y).
    path(X, Z) :- path(X, Y), e1(Y, Z).
"#;

const RECURSIVE_TRIANGLE_SOURCE: &str = r#"
    pred e1(u32, u32).
    pred e2(u32, u32).
    pred e3(u32, u32).
    pred tri(u32, u32, u32).
    pred tri_echo(u32, u32, u32).
    pred chain(u32, u32).
    pred path(u32, u32).

    tri(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z).
    tri_echo(X, Y, Z) :- tri(X, Y, Z).
    tri(X, Y, Z) :- tri_echo(X, Y, Z).
    chain(X, Z) :- e1(X, Y), e2(Y, Z).
    path(X, Y) :- e1(X, Y).
    path(X, Z) :- path(X, Y), e1(Y, Z).
"#;

const DYNAMIC_BASE_RULE_SOURCE: &str = r#"
    pred e1(u32, u32).
    pred learned(u32, u32).

    learned(X, Y) :- e1(X, Y).
"#;

const DYNAMIC_TRANSITIVE_RULE: &str = "learned(X, Z) :- learned(X, Y), e1(Y, Z).";

const TRAINING_ROLLBACK_BASE_SOURCE: &str = r#"
    pred e1(u32, u32).
    pred arm_d_path(u32, u32).

    arm_d_path(X, Y) :- e1(X, Y).
"#;

const TRAINING_ROLLBACK_DISCOVERED_RULE: &str = "arm_d_path(X, Z) :- arm_d_path(X, Y), e1(Y, Z).";

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
    provider: Arc<CudaKernelProvider>,
    pool: Arc<StreamPool>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ExecutionSnapshot {
    relations: BTreeMap<String, Vec<Vec<u32>>>,
    wcoj_triangle_dispatches: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct InjectionSnapshot {
    before: Vec<Vec<u32>>,
    after: Vec<Vec<u32>>,
}

struct CompiledFixture {
    plan: ExecutionPlan,
    rel_ids: HashMap<String, RelId>,
}

struct SimulatedTrainResult {
    discovered_rule: &'static str,
}

struct DeterministicEnvGuard {
    prior: Option<String>,
}

impl DeterministicEnvGuard {
    fn set() -> Self {
        let prior = std::env::var("XLOG_DETERMINISTIC").ok();
        std::env::set_var("XLOG_DETERMINISTIC", "1");
        Self { prior }
    }
}

impl Drop for DeterministicEnvGuard {
    fn drop(&mut self) {
        match &self.prior {
            Some(value) => std::env::set_var("XLOG_DETERMINISTIC", value),
            None => std::env::remove_var("XLOG_DETERMINISTIC"),
        }
    }
}

fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn with_deterministic_env<R>(f: impl FnOnce() -> R) -> R {
    let _lock = env_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let _guard = DeterministicEnvGuard::set();
    f()
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
        Box::new(GlobalDeviceBudget::new(logging, 128 * 1024 * 1024));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(128 * 1024 * 1024),
        Arc::clone(&runtime),
    ));
    let provider =
        Arc::new(CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory)).ok()?);
    Some(RuntimeFixture {
        device,
        runtime,
        memory,
        provider,
        pool,
    })
}

fn shared_inputs() -> BTreeMap<&'static str, Vec<(u32, u32)>> {
    assert_eq!(FIXED_SEED, 0x573, "fixture seed must remain pinned");
    BTreeMap::from([
        (
            "e1",
            vec![
                (1, 2),
                (1, 3),
                (1, 4),
                (2, 3),
                (2, 4),
                (3, 4),
                (5, 6),
                (6, 7),
            ],
        ),
        ("e2", vec![(2, 3), (2, 4), (3, 4), (6, 7)]),
        ("e3", vec![(1, 3), (1, 4), (2, 4), (3, 4), (5, 7)]),
    ])
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

    if n > 0 {
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

fn row_count(buf: &CudaBuffer) -> usize {
    if let Some(count) = buf.cached_row_count() {
        return count as usize;
    }
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

fn download_canonical_u32_rows(buf: &CudaBuffer) -> Vec<Vec<u32>> {
    let n = row_count(buf);
    if n == 0 {
        return Vec::new();
    }

    let arity = buf.arity();
    let mut cols = Vec::with_capacity(arity);
    for col_idx in 0..arity {
        let mut bytes = vec![0u8; n * std::mem::size_of::<u32>()];
        unsafe {
            let res = sys::cuMemcpyDtoH_v2(
                bytes.as_mut_ptr() as *mut _,
                *buf.column(col_idx).expect("column").device_ptr(),
                bytes.len(),
            );
            assert_eq!(res, sys::cudaError_enum::CUDA_SUCCESS);
        }
        cols.push(bytes);
    }

    let mut rows: Vec<Vec<u32>> = (0..n)
        .map(|row| {
            (0..arity)
                .map(|col| {
                    let start = row * std::mem::size_of::<u32>();
                    u32::from_le_bytes(cols[col][start..start + 4].try_into().unwrap())
                })
                .collect()
        })
        .collect();
    rows.sort();
    rows
}

fn compile_source(source: &str) -> CompiledFixture {
    let mut compiler = Compiler::new();
    let plan = compiler.compile(source).expect("compile");
    CompiledFixture {
        plan,
        rel_ids: compiler.rel_ids().clone(),
    }
}

fn run_compiled(
    fixture: &RuntimeFixture,
    compiled: &CompiledFixture,
    inputs: &BTreeMap<&str, Vec<(u32, u32)>>,
    config: RuntimeConfig,
    output_relations: &[&str],
) -> ExecutionSnapshot {
    let mut executor = Executor::new_with_config(Arc::clone(&fixture.provider), config);
    for (name, rel_id) in &compiled.rel_ids {
        executor.register_relation(*rel_id, name);
    }
    for (name, rows) in inputs {
        if compiled.rel_ids.contains_key(*name) {
            executor.put_relation(name, upload_binary_u32(&fixture.memory, rows));
        }
    }

    executor.execute_plan(&compiled.plan).expect("execute_plan");

    let mut relations = BTreeMap::new();
    for relation in output_relations {
        let buffer = executor
            .store()
            .get(relation)
            .unwrap_or_else(|| panic!("missing output relation {relation}"));
        relations.insert((*relation).to_string(), download_canonical_u32_rows(buffer));
    }

    ExecutionSnapshot {
        relations,
        wcoj_triangle_dispatches: executor.wcoj_triangle_dispatch_count(),
    }
}

fn source_with_injected_rule(base: &str, rule: &str) -> String {
    format!("{base}\n{rule}\n")
}

fn dynamic_injection_snapshot(
    fixture: &RuntimeFixture,
    inputs: &BTreeMap<&str, Vec<(u32, u32)>>,
    before_program: &CompiledFixture,
    after_program: &CompiledFixture,
) -> InjectionSnapshot {
    let config = RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(false));
    let before = run_compiled(
        fixture,
        before_program,
        inputs,
        config.clone(),
        &["learned"],
    )
    .relations
    .remove("learned")
    .expect("learned before");
    let after = run_compiled(fixture, after_program, inputs, config, &["learned"])
        .relations
        .remove("learned")
        .expect("learned after");
    assert!(
        after.len() > before.len(),
        "dynamic transitive-rule injection must expand learned rows"
    );
    InjectionSnapshot { before, after }
}

fn train_on_compiled_relations_simulator() -> SimulatedTrainResult {
    SimulatedTrainResult {
        discovered_rule: TRAINING_ROLLBACK_DISCOVERED_RULE,
    }
}

fn training_rollback_snapshot(
    fixture: &RuntimeFixture,
    inputs: &BTreeMap<&str, Vec<(u32, u32)>>,
    before_program: &CompiledFixture,
    after_program: &CompiledFixture,
) -> InjectionSnapshot {
    let config = RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(false));
    let before = run_compiled(
        fixture,
        before_program,
        inputs,
        config.clone(),
        &["arm_d_path"],
    )
    .relations
    .remove("arm_d_path")
    .expect("arm_d_path before");
    let after = run_compiled(fixture, after_program, inputs, config, &["arm_d_path"])
        .relations
        .remove("arm_d_path")
        .expect("arm_d_path after");
    assert!(
        after.len() > before.len(),
        "simulated training-discovered rule must expand arm_d_path rows"
    );
    InjectionSnapshot { before, after }
}

#[test]
fn cross_mode_wcoj_binary_recursive_outputs_are_bit_exact_100x() {
    with_deterministic_env(|| {
        let Some(fixture) = make_runtime_fixture() else {
            eprintln!("Skipping cross-mode determinism: CUDA runtime unavailable");
            return;
        };
        let inputs = shared_inputs();
        let outputs = ["tri", "chain", "path"];
        let cross_mode = compile_source(CROSS_MODE_SOURCE);
        let recursive_triangle = compile_source(RECURSIVE_TRIANGLE_SOURCE);

        let binary = run_compiled(
            &fixture,
            &cross_mode,
            &inputs,
            RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(false)),
            &outputs,
        );
        assert_eq!(
            binary.wcoj_triangle_dispatches, 0,
            "binary-join fallback mode must not dispatch triangle WCOJ"
        );

        let wcoj = run_compiled(
            &fixture,
            &cross_mode,
            &inputs,
            RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(true)),
            &outputs,
        );
        assert!(
            wcoj.wcoj_triangle_dispatches > 0,
            "forced WCOJ mode must dispatch at least once"
        );
        assert_eq!(
            wcoj.relations, binary.relations,
            "WCOJ output must match binary-join fallback output on the shared fixture"
        );

        let recursive = run_compiled(
            &fixture,
            &recursive_triangle,
            &inputs,
            RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(true)),
            &outputs,
        );
        assert!(
            recursive.wcoj_triangle_dispatches > 0,
            "recursive SCC mode must still exercise the WCOJ triangle rule"
        );
        assert_eq!(
            recursive.relations, binary.relations,
            "recursive-SCC output must match binary-join fallback output on the shared fixture"
        );
        assert_eq!(
            wcoj.relations, recursive.relations,
            "WCOJ, binary, and recursive outputs must be three-way bit-exact"
        );
        for relation in outputs {
            assert!(
                !binary.relations[relation].is_empty(),
                "{relation} output must be non-empty"
            );
        }

        let expected = wcoj.relations.clone();
        for iter in 0..ITERATIONS {
            let trial = run_compiled(
                &fixture,
                &cross_mode,
                &inputs,
                RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(true)),
                &outputs,
            );
            assert_eq!(
                trial.relations, expected,
                "fixed-seed evaluate run {iter} diverged from the first WCOJ snapshot"
            );
        }

        println!(
            "CROSS_MODE_DETERMINISM fixed_seed={FIXED_SEED} iterations={ITERATIONS} \
             wcoj_dispatches={} binary_dispatches={} recursive_dispatches={} \
             tri_rows={} chain_rows={} path_rows={}",
            wcoj.wcoj_triangle_dispatches,
            binary.wcoj_triangle_dispatches,
            recursive.wcoj_triangle_dispatches,
            expected["tri"].len(),
            expected["chain"].len(),
            expected["path"].len()
        );
    });
}

#[test]
fn dynamic_injection_and_training_rollback_are_bit_exact_100x() {
    with_deterministic_env(|| {
        let Some(fixture) = make_runtime_fixture() else {
            eprintln!("Skipping injection determinism: CUDA runtime unavailable");
            return;
        };
        let inputs = shared_inputs();
        let dynamic_before = compile_source(DYNAMIC_BASE_RULE_SOURCE);
        let dynamic_after_source =
            source_with_injected_rule(DYNAMIC_BASE_RULE_SOURCE, DYNAMIC_TRANSITIVE_RULE);
        let dynamic_after = compile_source(&dynamic_after_source);
        let training_rollback_before = compile_source(TRAINING_ROLLBACK_BASE_SOURCE);
        let strict_train_result = train_on_compiled_relations_simulator();
        let training_rollback_after_source = source_with_injected_rule(
            TRAINING_ROLLBACK_BASE_SOURCE,
            strict_train_result.discovered_rule,
        );
        let training_rollback_after = compile_source(&training_rollback_after_source);

        let expected_dynamic =
            dynamic_injection_snapshot(&fixture, &inputs, &dynamic_before, &dynamic_after);
        let expected_training_rollback = training_rollback_snapshot(
            &fixture,
            &inputs,
            &training_rollback_before,
            &training_rollback_after,
        );

        for iter in 0..ITERATIONS {
            let dynamic =
                dynamic_injection_snapshot(&fixture, &inputs, &dynamic_before, &dynamic_after);
            assert_eq!(
                dynamic, expected_dynamic,
                "dynamic-rule-injection run {iter} diverged"
            );

            let training_rollback = training_rollback_snapshot(
                &fixture,
                &inputs,
                &training_rollback_before,
                &training_rollback_after,
            );
            assert_eq!(
                training_rollback, expected_training_rollback,
                "training rollback injection run {iter} diverged"
            );
        }

        println!(
            "DYNAMIC_INJECTION_DETERMINISM fixed_seed={FIXED_SEED} iterations={ITERATIONS} \
             dynamic_before_rows={} dynamic_after_rows={} \
             training_rollback_before_rows={} training_rollback_after_rows={}",
            expected_dynamic.before.len(),
            expected_dynamic.after.len(),
            expected_training_rollback.before.len(),
            expected_training_rollback.after.len()
        );
    });
}
