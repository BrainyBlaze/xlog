#![cfg(feature = "host-io")]

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::{Arc, Barrier, Mutex, MutexGuard, OnceLock};

use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::{CudaBuffer, CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_logic::epistemic::compile_epistemic_gpu_execution;
use xlog_logic::parse_program;
use xlog_prob::epistemic_production::EpistemicProbProductionAdapter;
use xlog_prob::exact::{ExactResult, ExactResultWithGrads, GpuConfig, ProbVarInfo};
use xlog_runtime::{EpistemicGpuWorkspaceCapacities, Executor};

const PROBABILITY_SOURCE: &str = r#"
0.5::target(9).
0.6::latent(1, 2).
known_gate(1, 2) :- latent(1, 2).
0.3::missing_known_gate(1, 2).
query(target(9)).
query(known_gate(1, 2)).
query(missing_known_gate(1, 2)).
"#;

struct LockedCudaProvider {
    _guard: MutexGuard<'static, ()>,
    provider: Arc<CudaKernelProvider>,
}

struct IsolatedCircuitCache {
    path: PathBuf,
    previous: Option<OsString>,
}

impl IsolatedCircuitCache {
    fn empty() -> Self {
        let path = std::env::temp_dir().join(format!(
            "xlog-conditioned-circuit-cache-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("create isolated empty circuit cache");
        let previous = std::env::var_os("XLOG_CIRCUIT_CACHE_DIR");
        std::env::set_var("XLOG_CIRCUIT_CACHE_DIR", &path);
        Self { path, previous }
    }
}

impl Drop for IsolatedCircuitCache {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.take() {
            std::env::set_var("XLOG_CIRCUIT_CACHE_DIR", previous);
        } else {
            std::env::remove_var("XLOG_CIRCUIT_CACHE_DIR");
        }
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

impl Deref for LockedCudaProvider {
    type Target = Arc<CudaKernelProvider>;

    fn deref(&self) -> &Self::Target {
        &self.provider
    }
}

fn try_provider() -> Option<LockedCudaProvider> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let guard = LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let device = match CudaDevice::new(0) {
        Ok(device) => Arc::new(device),
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA unavailable: {error}");
            return None;
        }
    };
    let memory = Arc::new(GpuMemoryManager::new(
        device.clone(),
        MemoryBudget::with_limit(1024 * 1024 * 1024),
    ));
    match CudaKernelProvider::new(device, memory) {
        Ok(provider) => Some(LockedCudaProvider {
            _guard: guard,
            provider: Arc::new(provider),
        }),
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA provider creation failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA provider unavailable: {error}");
            None
        }
    }
}

#[test]
fn conditioned_program_compiles_once_and_reuses_circuit_for_atomic_prior_updates() {
    let Some(provider) = try_provider() else {
        return;
    };
    let _cache = IsolatedCircuitCache::empty();
    let evidence = execute_accepted_derived_and_negated_evidence(&provider);
    let config = gpu_config(&provider);
    let mut adapter = EpistemicProbProductionAdapter::new(config);
    let prepared = adapter
        .prepare_conditioned_source_with_gpu_execution_result(
            PROBABILITY_SOURCE,
            &provider,
            &evidence,
            Vec::new(),
        )
        .expect("prepare one conditioned exact circuit");
    let prepared_circuit = prepared
        .circuit_witness()
        .expect("read authoritative prepared-circuit witness");
    assert_eq!(prepared_circuit.preparation_compiles, 1);
    assert_eq!(prepared_circuit.materializations, 1);
    assert_eq!(prepared_circuit.disk_cache_restores, 0);
    assert_eq!(prepared_circuit.gpu_cache_hits, 0);
    assert!(prepared_circuit.circuit_generation > 0);

    let map = prepared
        .prob_var_map()
        .expect("read probability variable map");
    let target_var = map
        .iter()
        .enumerate()
        .find_map(|(var, info)| match info {
            ProbVarInfo::Fact { atom, .. } if atom.predicate == "target" => Some(var as u32),
            _ => None,
        })
        .expect("target fact must have a CNF variable");
    let missing_known_gate_var = map
        .iter()
        .enumerate()
        .find_map(|(var, info)| match info {
            ProbVarInfo::Fact { atom, .. } if atom.predicate == "missing_known_gate" => {
                Some(var as u32)
            }
            _ => None,
        })
        .expect("negated-evidence fact must have a CNF variable");

    for (target_prior, evidence_prior) in [(0.5, 0.3), (0.9, 0.3), (0.1, 0.3), (0.1, 0.8)] {
        prepared
            .set_fact_probabilities(&BTreeMap::from([
                (target_var, target_prior),
                (missing_known_gate_var, evidence_prior),
            ]))
            .expect("update query and evidence-fixed independent fact priors");
        assert_eq!(
            prepared
                .circuit_witness()
                .expect("prior update preserves circuit identity"),
            prepared_circuit
        );

        let (actual, trace) = prepared.evaluate().expect("reuse conditioned circuit");
        let (actual_grads, grad_trace) = prepared
            .evaluate_with_grads()
            .expect("reuse conditioned circuit for gradients");
        let fresh_source = PROBABILITY_SOURCE
            .replacen("0.5::target(9).", &format!("{target_prior}::target(9)."), 1)
            .replacen(
                "0.3::missing_known_gate(1, 2).",
                &format!("{evidence_prior}::missing_known_gate(1, 2)."),
                1,
            );
        let mut fresh_adapter = EpistemicProbProductionAdapter::new(config);
        let expected = fresh_adapter
            .compile_and_evaluate_conditioned_source_with_gpu_execution_result(
                &fresh_source,
                &provider,
                &evidence,
                Vec::new(),
            )
            .expect("fresh conditioned compilation");
        let mut fresh_gradient_adapter = EpistemicProbProductionAdapter::new(config);
        let expected_grads = fresh_gradient_adapter
            .compile_and_evaluate_conditioned_source_with_grads_with_gpu_execution_result(
                &fresh_source,
                &provider,
                &evidence,
                Vec::new(),
            )
            .expect("fresh conditioned gradient compilation");

        assert_exact_result_close(&actual, &expected);
        assert_exact_gradients_close(&actual_grads, &expected_grads);
        assert_eq!(trace.gpu_exact_source_compiles, 0);
        assert_eq!(trace.gpu_exact_program_compiles, 0);
        assert_eq!(trace.gpu_conditioned_circuit_reuses, 1);
        assert_eq!(trace.gpu_conditioned_circuit_preparation_compiles, 1);
        assert_eq!(trace.gpu_conditioned_circuit_materializations, 1);
        assert_eq!(trace.gpu_conditioned_circuit_disk_cache_restores, 0);
        assert_eq!(trace.gpu_conditioned_circuit_gpu_cache_hits, 0);
        assert_eq!(
            trace.gpu_conditioned_circuit_generation,
            prepared_circuit.circuit_generation
        );
        assert_eq!(
            trace.gpu_conditioned_circuit_cache_slot,
            u64::from(prepared_circuit.cache_slot)
        );
        assert_eq!(grad_trace.gpu_exact_source_compiles, 0);
        assert_eq!(grad_trace.gpu_exact_program_compiles, 0);
        assert_eq!(grad_trace.gpu_conditioned_circuit_reuses, 1);
        assert_eq!(grad_trace.gpu_conditioned_circuit_preparation_compiles, 1);
        assert_eq!(grad_trace.gpu_conditioned_circuit_materializations, 1);
        assert_eq!(grad_trace.gpu_conditioned_circuit_disk_cache_restores, 0);
        assert_eq!(grad_trace.gpu_conditioned_circuit_gpu_cache_hits, 0);
        assert_eq!(
            grad_trace.gpu_conditioned_circuit_generation,
            prepared_circuit.circuit_generation
        );
        assert_eq!(
            grad_trace.gpu_conditioned_circuit_cache_slot,
            u64::from(prepared_circuit.cache_slot)
        );
        assert_eq!(
            prepared
                .circuit_witness()
                .expect("evaluation preserves circuit identity"),
            prepared_circuit
        );
    }

    let before = prepared
        .prob_var_map()
        .expect("read map before invalid batch");
    let invalid = BTreeMap::from([(target_var, 0.8), (0, 0.2)]);
    let error = prepared
        .set_fact_probabilities(&invalid)
        .expect_err("one invalid variable must reject the complete batch");
    assert!(error.to_string().contains("CNF variable 0"));
    assert_prob_var_maps_equal(
        &prepared.prob_var_map().expect("read map after rejection"),
        &before,
    );
    assert_eq!(
        prepared
            .circuit_witness()
            .expect("rejected update preserves circuit identity"),
        prepared_circuit
    );

    let clone = prepared.clone();
    clone
        .set_fact_probabilities(&BTreeMap::from([(target_var, 0.4)]))
        .expect("clone updates shared state");
    let shared = prepared.prob_var_map().expect("original sees clone update");
    assert!(matches!(
        &shared[target_var as usize],
        ProbVarInfo::Fact { prob, .. } if (prob - 0.4).abs() < 1e-12
    ));
    assert_eq!(
        clone
            .circuit_witness()
            .expect("clone shares circuit identity"),
        prepared_circuit
    );

    let start = Arc::new(Barrier::new(3));
    let workers: Vec<_> = [0.2, 0.8]
        .into_iter()
        .map(|prior| {
            let prepared = prepared.clone();
            let start = start.clone();
            std::thread::spawn(move || {
                start.wait();
                for _ in 0..8 {
                    prepared
                        .set_fact_probabilities(&BTreeMap::from([(target_var, prior)]))
                        .expect("clone update must serialize");
                    let (result, trace) = prepared
                        .evaluate()
                        .expect("clone evaluation must serialize");
                    let target = result
                        .query_probs
                        .iter()
                        .find(|query| query.atom.predicate == "target")
                        .expect("target query remains available");
                    assert!(
                        (target.prob - 0.2).abs() < 1e-9 || (target.prob - 0.8).abs() < 1e-9,
                        "concurrent evaluation observed partial weights: {}",
                        target.prob
                    );
                    assert_eq!(trace.gpu_conditioned_circuit_reuses, 1);
                    assert_eq!(trace.gpu_conditioned_circuit_preparation_compiles, 1);
                    assert_eq!(trace.gpu_conditioned_circuit_materializations, 1);
                    assert_eq!(trace.gpu_conditioned_circuit_disk_cache_restores, 0);
                    assert_eq!(trace.gpu_conditioned_circuit_gpu_cache_hits, 0);
                    assert_eq!(
                        trace.gpu_conditioned_circuit_generation,
                        prepared_circuit.circuit_generation
                    );
                    assert_eq!(
                        trace.gpu_conditioned_circuit_cache_slot,
                        u64::from(prepared_circuit.cache_slot)
                    );
                }
            })
        })
        .collect();
    start.wait();
    for worker in workers {
        worker.join().expect("conditioned clone worker");
    }

    let mut warm_adapter = EpistemicProbProductionAdapter::new(config);
    let warm = warm_adapter
        .prepare_conditioned_source_with_gpu_execution_result(
            PROBABILITY_SOURCE,
            &provider,
            &evidence,
            Vec::new(),
        )
        .expect("restore a second prepared circuit from the verified disk cache");
    let warm_circuit = warm
        .circuit_witness()
        .expect("read warm prepared-circuit witness");
    assert_eq!(warm_circuit.preparation_compiles, 0);
    assert_eq!(warm_circuit.materializations, 1);
    assert_eq!(warm_circuit.disk_cache_restores, 1);
    assert_eq!(warm_circuit.gpu_cache_hits, 0);
    assert_ne!(
        warm_circuit.circuit_generation,
        prepared_circuit.circuit_generation
    );
    assert_ne!(warm_circuit, prepared_circuit);
    let (_, warm_trace) = warm.evaluate().expect("evaluate restored circuit");
    assert_eq!(warm_trace.gpu_exact_source_compiles, 0);
    assert_eq!(warm_trace.gpu_exact_program_compiles, 0);
    assert_eq!(warm_trace.gpu_conditioned_circuit_preparation_compiles, 0);
    assert_eq!(warm_trace.gpu_conditioned_circuit_materializations, 1);
    assert_eq!(warm_trace.gpu_conditioned_circuit_disk_cache_restores, 1);
    assert_eq!(warm_trace.gpu_conditioned_circuit_gpu_cache_hits, 0);
    assert_eq!(
        warm_trace.gpu_conditioned_circuit_generation,
        warm_circuit.circuit_generation
    );
}

fn gpu_config(provider: &CudaKernelProvider) -> GpuConfig {
    let mut config = GpuConfig::default();
    config.device_ordinal = provider.device().ordinal();
    config.memory_bytes = provider.memory().budget().device_bytes;
    config
}

fn execute_accepted_derived_and_negated_evidence(
    provider: &Arc<CudaKernelProvider>,
) -> xlog_runtime::EpistemicGpuExecutionResult {
    let program = parse_program(
        r#"
        pred seed(u32, u32).
        pred known_gate(u32, u32).
        pred missing_known_gate(u32, u32).
        pred out(u32, u32).

        out(X, Y) :- seed(X, Y), know known_gate(X, Y),
                     not know missing_known_gate(X, Y).
        "#,
    )
    .expect("parse epistemic evidence program");
    let executable = compile_epistemic_gpu_execution(&program).expect("compile epistemic plan");
    let mut executor = Executor::new(provider.clone());
    for (name, rows) in [
        ("seed", &[(1, 2)][..]),
        ("known_gate", &[(1, 2)][..]),
        ("missing_known_gate", &[(2, 2)][..]),
    ] {
        let rel = *executable
            .relation_ids
            .get(name)
            .unwrap_or_else(|| panic!("compiled plan must expose relation id for {name}"));
        executor.register_relation(rel, name);
        executor.put_relation(name, upload_binary_u32(provider, rows, "x", "y"));
    }
    executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 8,
                max_worlds: 4,
                max_models_per_reduction: 1,
            },
        )
        .expect("execute accepted epistemic evidence")
}

fn upload_binary_u32(
    provider: &Arc<CudaKernelProvider>,
    rows: &[(u32, u32)],
    first_name: &str,
    second_name: &str,
) -> CudaBuffer {
    let first: Vec<u8> = rows.iter().flat_map(|row| row.0.to_le_bytes()).collect();
    let second: Vec<u8> = rows.iter().flat_map(|row| row.1.to_le_bytes()).collect();
    let mut first_device = provider
        .memory()
        .alloc(first.len())
        .expect("allocate first column");
    let mut second_device = provider
        .memory()
        .alloc(second.len())
        .expect("allocate second column");
    let mut row_count_device = provider.memory().alloc(1).expect("allocate row count");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&first, &mut first_device)
        .expect("upload first column");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&second, &mut second_device)
        .expect("upload second column");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&[rows.len() as u32], &mut row_count_device)
        .expect("upload row count");
    CudaBuffer::from_columns_with_host_count(
        vec![first_device.into(), second_device.into()],
        rows.len() as u64,
        row_count_device,
        Schema::new(vec![
            (first_name.to_string(), ScalarType::U32),
            (second_name.to_string(), ScalarType::U32),
        ]),
        rows.len() as u32,
    )
}

fn assert_exact_result_close(actual: &ExactResult, expected: &ExactResult) {
    assert!((actual.log_z_e - expected.log_z_e).abs() < 1e-9);
    assert_eq!(actual.query_probs.len(), expected.query_probs.len());
    for (actual, expected) in actual.query_probs.iter().zip(&expected.query_probs) {
        assert_eq!(actual.atom, expected.atom);
        assert!((actual.prob - expected.prob).abs() < 1e-9);
        if actual.log_prob.is_finite() || expected.log_prob.is_finite() {
            assert!((actual.log_prob - expected.log_prob).abs() < 1e-9);
        } else {
            assert_eq!(actual.log_prob, expected.log_prob);
        }
    }
}

fn assert_exact_gradients_close(actual: &ExactResultWithGrads, expected: &ExactResultWithGrads) {
    assert!((actual.log_z_e - expected.log_z_e).abs() < 1e-9);
    assert_eq!(actual.query_grads.len(), expected.query_grads.len());
    for (actual, expected) in actual.query_grads.iter().zip(&expected.query_grads) {
        assert_eq!(actual.atom, expected.atom);
        assert!((actual.prob - expected.prob).abs() < 1e-9);
        if actual.log_prob.is_finite() || expected.log_prob.is_finite() {
            assert!((actual.log_prob - expected.log_prob).abs() < 1e-9);
        } else {
            assert_eq!(actual.log_prob, expected.log_prob);
        }
        assert_eq!(actual.grad_true.len(), expected.grad_true.len());
        assert_eq!(actual.grad_false.len(), expected.grad_false.len());
        for (actual, expected) in actual.grad_true.iter().zip(&expected.grad_true) {
            assert!((actual - expected).abs() < 1e-9);
        }
        for (actual, expected) in actual.grad_false.iter().zip(&expected.grad_false) {
            assert!((actual - expected).abs() < 1e-9);
        }
    }
}

fn assert_prob_var_maps_equal(actual: &[ProbVarInfo], expected: &[ProbVarInfo]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        match (actual, expected) {
            (
                ProbVarInfo::Fact {
                    atom: actual_atom,
                    prob: actual_prob,
                },
                ProbVarInfo::Fact {
                    atom: expected_atom,
                    prob: expected_prob,
                },
            ) => {
                assert_eq!(actual_atom, expected_atom);
                assert!((actual_prob - expected_prob).abs() < 1e-12);
            }
            (ProbVarInfo::Choice { .. }, ProbVarInfo::Choice { .. })
            | (ProbVarInfo::Other, ProbVarInfo::Other) => {}
            _ => panic!("probability variable map kind changed"),
        }
    }
}
