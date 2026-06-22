#![cfg(feature = "host-io")]
//! Engine-contract tests for the host-facing MC `evaluate` wrapper.
//!
//! Contract under test: the CPU oracle (`evaluate_cpu`) must never be
//! silently substituted for the GPU-resident engine. A program the resident
//! engine rejects fails closed with the typed rejection unless the caller
//! explicitly opts into the CPU oracle, and every `McResult` is labeled with
//! the engine that produced it.

use xlog_cuda::CudaDevice;
use xlog_prob::mc::{McEngine, McEvalConfig, McProgram};

fn has_cuda_device() -> bool {
    CudaDevice::new(0).is_ok()
}

/// Negation in a rule body is rejected by the resident engine
/// (`ResidentRejectKind::Negation`), making this the canonical
/// resident-rejected fragment.
const REJECTED_SRC: &str = r#"
0.5::flip().
p() :- flip().
q() :- not p().
p() :- not q().
query(p()).
"#;

#[test]
fn test_mc_evaluate_fails_closed_on_resident_rejection() {
    if !has_cuda_device() {
        eprintln!("Skipping: no CUDA device");
        return;
    }

    let program = McProgram::compile_source(REJECTED_SRC).unwrap();
    let cfg = McEvalConfig::default();
    let err = program
        .evaluate(cfg)
        .expect_err("resident-rejected program must fail closed, not silently run the CPU oracle");
    let msg = err.to_string();
    assert!(
        msg.contains("resident MC engine rejected program"),
        "error must carry the typed rejection: {msg}"
    );
}

#[test]
fn test_mc_cpu_oracle_requires_explicit_opt_in_and_is_labeled() {
    if !has_cuda_device() {
        eprintln!("Skipping: no CUDA device");
        return;
    }

    let program = McProgram::compile_source(REJECTED_SRC).unwrap();
    let mut cfg = McEvalConfig::default();
    cfg.samples = 2_000;
    cfg.seed = 999;
    cfg.allow_cpu_oracle_fallback = true;
    let result = program.evaluate(cfg).unwrap();

    assert_eq!(result.engine, McEngine::CpuOracle);
    assert_eq!(result.engine.as_str(), "cpu-oracle");
    // Semantics match the pre-fix oracle behavior: p() holds in every stable
    // world (flip → p; ¬flip → q...¬p — but p :- not q makes p true when q
    // is underivable), so the estimate stays near the flip marginal.
    let p = result
        .query_estimates
        .iter()
        .find(|q| q.atom.predicate == "p")
        .expect("missing query for p()")
        .prob;
    assert!((p - 0.5).abs() < 0.08, "p={p}");
}

#[test]
fn test_mc_resident_engine_result_is_labeled_gpu() {
    if !has_cuda_device() {
        eprintln!("Skipping: no CUDA device");
        return;
    }

    let src = r#"
0.7::rain().
query(rain()).
"#;
    let program = McProgram::compile_source(src).unwrap();
    let result = program.evaluate(McEvalConfig::default()).unwrap();
    assert_eq!(result.engine, McEngine::GpuResident);
    assert_eq!(result.engine.as_str(), "gpu-resident");
}
