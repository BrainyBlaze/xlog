#![cfg(feature = "host-io")]

use xlog_cuda::CudaDevice;
use xlog_prob::mc::{McEvalConfig, McProgram};

fn has_cuda_device() -> bool {
    // cudarc::driver::CudaDevice::count() may panic in restricted containers. Attempt real init instead.
    CudaDevice::new(0).is_ok()
}

fn prob_of_atom(result: &xlog_prob::mc::McResult, predicate: &str) -> f64 {
    result
        .query_estimates
        .iter()
        .find(|q| q.atom.predicate == predicate && q.atom.args.is_empty())
        .unwrap_or_else(|| panic!("missing query for {}()", predicate))
        .prob
}

#[test]
fn test_mc_probabilistic_fact_marginal_is_reasonable() {
    if !has_cuda_device() {
        eprintln!("Skipping test: no CUDA device available");
        return;
    }

    let src = r#"
0.7::rain().
query(rain()).
"#;

    let program = McProgram::compile_source(src).unwrap();
    let cfg = McEvalConfig {
        samples: 50_000,
        seed: 123,
        confidence: 0.95,
        max_nonmonotone_iterations: 128,
    };
    let result = program.evaluate(cfg).unwrap();

    let p = prob_of_atom(&result, "rain");
    assert!((p - 0.7).abs() < 0.02, "p={}", p);
    assert_eq!(result.evidence_samples, result.total_samples);
}

#[test]
fn test_mc_wet_conditioning_close_to_exact() {
    if !has_cuda_device() {
        eprintln!("Skipping test: no CUDA device available");
        return;
    }

    let src = r#"
0.7::rain().
0.2::sprinkler().
wet() :- rain().
wet() :- sprinkler().
evidence(wet(), true).
query(rain()).
query(sprinkler()).
"#;

    let program = McProgram::compile_source(src).unwrap();
    let cfg = McEvalConfig {
        samples: 80_000,
        seed: 7,
        confidence: 0.95,
        max_nonmonotone_iterations: 128,
    };
    let result = program.evaluate(cfg).unwrap();

    let p_wet = 1.0 - (1.0 - 0.7) * (1.0 - 0.2);
    let expected_rain = 0.7 / p_wet;
    let expected_sprinkler = 0.2 / p_wet;

    let got_rain = prob_of_atom(&result, "rain");
    let got_sprinkler = prob_of_atom(&result, "sprinkler");

    assert!(
        (got_rain - expected_rain).abs() < 0.02,
        "got_rain={}",
        got_rain
    );
    assert!(
        (got_sprinkler - expected_sprinkler).abs() < 0.02,
        "got_sprinkler={}",
        got_sprinkler
    );
    assert!(result.evidence_samples > 0);
}

#[test]
fn test_mc_nonmonotone_recursion_runs_and_is_stable() {
    if !has_cuda_device() {
        eprintln!("Skipping test: no CUDA device available");
        return;
    }

    // Non-monotone recursion through negation. This program is stable in each world:
    // - If flip() holds, p() is true and q() is false.
    // - Otherwise, q() is true and p() is false.
    let src = r#"
0.5::flip().
p() :- flip().
q() :- not p().
p() :- not q().
query(p()).
query(flip()).
"#;

    let program = McProgram::compile_source(src).unwrap();
    let cfg = McEvalConfig {
        samples: 50_000,
        seed: 999,
        confidence: 0.95,
        max_nonmonotone_iterations: 256,
    };
    let result = program.evaluate(cfg).unwrap();

    let p_flip = prob_of_atom(&result, "flip");
    let p_p = prob_of_atom(&result, "p");
    assert!((p_flip - 0.5).abs() < 0.02, "p_flip={}", p_flip);
    assert!((p_p - 0.5).abs() < 0.02, "p_p={}", p_p);
    assert!(result.nonmonotone_sccs > 0);
}

#[test]
fn test_mc_annotated_disjunction_is_exclusive_under_evidence() {
    if !has_cuda_device() {
        eprintln!("Skipping test: no CUDA device available");
        return;
    }

    // Annotated disjunction induces a categorical choice (plus an implicit "none" choice when
    // probabilities sum to < 1.0). Under evidence selecting coin(1), coin(2) must be false.
    let src = r#"
0.3::coin(1); 0.3::coin(2).
evidence(coin(1), true).
query(coin(2)).
"#;

    let program = McProgram::compile_source(src).unwrap();
    let cfg = McEvalConfig {
        samples: 50_000,
        seed: 2026,
        confidence: 0.95,
        max_nonmonotone_iterations: 128,
    };
    let result = program.evaluate(cfg).unwrap();

    let p_coin2 = result
        .query_estimates
        .iter()
        .find(|q| {
            q.atom.predicate == "coin"
                && q.atom.args.len() == 1
                && q.atom.args[0] == xlog_prob::provenance::Value::I64(2)
        })
        .unwrap_or_else(|| panic!("missing query for coin(2)"))
        .prob;

    assert_eq!(p_coin2, 0.0);
    assert!(result.evidence_samples > 0);
}
