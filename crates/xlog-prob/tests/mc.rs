#![cfg(feature = "host-io")]

use xlog_cuda::CudaDevice;
use xlog_prob::mc::{ForceabilityReason, McEvalConfig, McProgram, McSamplingMethod};
use xlog_prob::provenance::Value;

fn has_cuda_device() -> bool {
    // cudarc::driver::CudaDevice::count() may panic in restricted containers. Attempt real init instead.
    CudaDevice::new(0).is_ok()
}

fn value_as_symbol_name(v: &Value) -> Option<String> {
    match v {
        Value::Symbol(id) => Some(xlog_core::symbol::resolve(*id)),
        Value::String(s) => Some(s.clone()),
        _ => None,
    }
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
        sampling_method: None,
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
        sampling_method: None,
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
        sampling_method: None,
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
        sampling_method: None,
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

#[test]
fn test_evidence_forcing_prob_fact_true() {
    let src = r#"
0.3::rain().
0.7::sprinkler().
evidence(rain(), true).
query(sprinkler()).
"#;
    let program = McProgram::compile_source(src).unwrap();
    let forcing = program.compile_evidence_forcing().unwrap();
    assert!(forcing.forceable);
    assert_eq!(forcing.reason, ForceabilityReason::AllForceable);
    // rain is var 0, sprinkler is var 1
    assert_eq!(forcing.force_mask[0], 1);
    assert_eq!(forcing.forced_value[0], 1);
    assert_eq!(forcing.force_mask[1], 0);
}

#[test]
fn test_evidence_forcing_prob_fact_false() {
    let src = r#"
0.3::rain().
evidence(rain(), false).
query(rain()).
"#;
    let program = McProgram::compile_source(src).unwrap();
    let forcing = program.compile_evidence_forcing().unwrap();
    assert!(forcing.forceable);
    assert_eq!(forcing.reason, ForceabilityReason::AllForceable);
    assert_eq!(forcing.force_mask[0], 1);
    assert_eq!(forcing.forced_value[0], 0);
}

#[test]
fn test_evidence_forcing_derived_atom_not_forceable() {
    let src = r#"
0.3::rain().
wet() :- rain().
evidence(wet(), true).
query(rain()).
"#;
    let program = McProgram::compile_source(src).unwrap();
    let forcing = program.compile_evidence_forcing().unwrap();
    assert!(!forcing.forceable);
    assert_eq!(forcing.reason, ForceabilityReason::ContainsDerivedEvidence);
}

#[test]
fn test_evidence_forcing_ad_3way_middle_head() {
    // 3 explicit heads with sum < 1.0, so there is an implicit none branch.
    // AD: 0.2::color(red); 0.3::color(blue); 0.4::color(green).
    // has_none = true (sum = 0.9 < 1.0), so 4-way including none
    // decision_vars: [v0, v1, v2] (3 Bernoulli vars for 4-way including none)
    // evidence(color(blue), true) => choice_idx=1 => force v0=0, v1=1
    let src = r#"
0.2::color(red); 0.3::color(blue); 0.4::color(green).
evidence(color(blue), true).
query(color(red)).
query(color(green)).
"#;
    let program = McProgram::compile_source(src).unwrap();
    let forcing = program.compile_evidence_forcing().unwrap();
    assert!(forcing.forceable, "3-way AD positive evidence should be forceable");

    // v0 forced to 0 (not red), v1 forced to 1 (blue selected), v2 not forced
    assert_eq!(forcing.force_mask[0], 1);
    assert_eq!(forcing.forced_value[0], 0);
    assert_eq!(forcing.force_mask[1], 1);
    assert_eq!(forcing.forced_value[1], 1);
    assert_eq!(forcing.force_mask[2], 0); // irrelevant after v1=1
}

#[test]
fn test_evidence_forcing_ad_last_head_no_none() {
    // 2 heads summing to 1.0 → no none branch
    // AD: 0.4::coin(heads); 0.6::coin(tails).
    // decision_vars: [v0] (1 Bernoulli var for 2-way, no none)
    // evidence(coin(tails), true) => last head, no none → force v0=0
    let src = r#"
0.4::coin(heads); 0.6::coin(tails).
evidence(coin(tails), true).
query(coin(heads)).
"#;
    let program = McProgram::compile_source(src).unwrap();
    let forcing = program.compile_evidence_forcing().unwrap();
    assert!(forcing.forceable);
    assert_eq!(forcing.force_mask[0], 1);
    assert_eq!(forcing.forced_value[0], 0); // last head → all decision vars = 0
}

#[test]
fn test_evidence_clamping_prob_fact_true_matches_exact() {
    if !has_cuda_device() {
        eprintln!("Skipping test: no CUDA device available");
        return;
    }
    let src = r#"
0.7::rain().
0.2::sprinkler().
evidence(sprinkler(), true).
query(rain()).
"#;
    let program = McProgram::compile_source(src).unwrap();
    let cfg = McEvalConfig {
        samples: 50_000,
        seed: 42,
        confidence: 0.95,
        max_nonmonotone_iterations: 128,
        sampling_method: None,
    };
    let result = program.evaluate(cfg).unwrap();
    assert_eq!(result.sampling_method, McSamplingMethod::EvidenceClamping);
    assert_eq!(result.evidence_samples, result.total_samples);
    let p = prob_of_atom(&result, "rain");
    assert!((p - 0.7).abs() < 0.02, "p={}", p);
}

#[test]
fn test_evidence_clamping_prob_fact_false_matches_exact() {
    if !has_cuda_device() {
        eprintln!("Skipping test: no CUDA device available");
        return;
    }
    let src = r#"
0.7::rain().
0.2::sprinkler().
evidence(sprinkler(), false).
query(rain()).
"#;
    let program = McProgram::compile_source(src).unwrap();
    let cfg = McEvalConfig {
        samples: 50_000,
        seed: 42,
        confidence: 0.95,
        max_nonmonotone_iterations: 128,
        sampling_method: None,
    };
    let result = program.evaluate(cfg).unwrap();
    assert_eq!(result.sampling_method, McSamplingMethod::EvidenceClamping);
    assert_eq!(result.evidence_samples, result.total_samples);
    let p = prob_of_atom(&result, "rain");
    assert!((p - 0.7).abs() < 0.02, "p={}", p);
}

#[test]
fn test_evidence_clamping_all_samples_count() {
    if !has_cuda_device() {
        eprintln!("Skipping test: no CUDA device available");
        return;
    }
    let src = r#"
0.01::rare().
0.5::other().
evidence(rare(), true).
query(other()).
"#;
    let program = McProgram::compile_source(src).unwrap();
    let cfg = McEvalConfig {
        samples: 1000,
        seed: 7,
        confidence: 0.95,
        max_nonmonotone_iterations: 128,
        sampling_method: None,
    };
    let result = program.evaluate(cfg).unwrap();
    assert_eq!(result.sampling_method, McSamplingMethod::EvidenceClamping);
    assert_eq!(result.evidence_samples, 1000);
    let p = prob_of_atom(&result, "other");
    assert!((p - 0.5).abs() < 0.05, "p={}", p);
}

#[test]
fn test_evidence_clamping_derived_evidence_falls_back() {
    if !has_cuda_device() {
        eprintln!("Skipping: no CUDA device");
        return;
    }

    let src = r#"
0.3::rain().
wet() :- rain().
evidence(wet(), true).
query(rain()).
"#;
    let program = McProgram::compile_source(src).unwrap();
    let cfg = McEvalConfig {
        samples: 50_000,
        seed: 7,
        confidence: 0.95,
        max_nonmonotone_iterations: 128,
        sampling_method: None, // auto-select → should fall back to Rejection
    };
    let result = program.evaluate(cfg).unwrap();
    assert_eq!(result.sampling_method, McSamplingMethod::Rejection);
    // P(rain | wet) = P(rain) / P(wet) = 0.3 / 0.3 = 1.0
    let p = prob_of_atom(&result, "rain");
    assert!((p - 1.0).abs() < 0.01, "p={}", p);
}

#[test]
fn test_evidence_clamping_negative_ad_falls_back() {
    if !has_cuda_device() {
        eprintln!("Skipping: no CUDA device");
        return;
    }

    let src = r#"
0.3::coin(1); 0.3::coin(2).
evidence(coin(1), false).
query(coin(2)).
"#;
    let program = McProgram::compile_source(src).unwrap();
    let cfg = McEvalConfig {
        samples: 50_000,
        seed: 2026,
        confidence: 0.95,
        max_nonmonotone_iterations: 128,
        sampling_method: None,
    };
    let result = program.evaluate(cfg).unwrap();
    assert_eq!(result.sampling_method, McSamplingMethod::Rejection);
}

#[test]
fn test_explicit_clamping_unforceable_evidence_errors() {
    let src = r#"
0.3::rain().
wet() :- rain().
evidence(wet(), true).
query(rain()).
"#;
    let program = McProgram::compile_source(src).unwrap();
    let cfg = McEvalConfig {
        samples: 1000,
        seed: 7,
        confidence: 0.95,
        max_nonmonotone_iterations: 128,
        sampling_method: Some(McSamplingMethod::EvidenceClamping),
    };
    let result = program.evaluate(cfg);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("EvidenceClamping") || err.contains("forceable"),
        "Error should mention clamping: {}",
        err
    );
}

#[test]
fn test_sampling_method_in_result_metadata() {
    if !has_cuda_device() {
        eprintln!("Skipping: no CUDA device");
        return;
    }

    // No evidence → Rejection
    let src_no_ev = r#"
0.5::a().
query(a()).
"#;
    let prog = McProgram::compile_source(src_no_ev).unwrap();
    let cfg = McEvalConfig {
        samples: 100,
        seed: 0,
        confidence: 0.95,
        max_nonmonotone_iterations: 128,
        sampling_method: None,
    };
    let result = prog.evaluate(cfg).unwrap();
    assert_eq!(result.sampling_method, McSamplingMethod::Rejection);

    // Root evidence → EvidenceClamping
    let src_ev = r#"
0.5::a().
0.3::b().
evidence(a(), true).
query(b()).
"#;
    let prog2 = McProgram::compile_source(src_ev).unwrap();
    let cfg2 = McEvalConfig {
        samples: 100,
        seed: 0,
        confidence: 0.95,
        max_nonmonotone_iterations: 128,
        sampling_method: None,
    };
    let result2 = prog2.evaluate(cfg2).unwrap();
    assert_eq!(result2.sampling_method, McSamplingMethod::EvidenceClamping);
}

#[test]
fn test_rejection_unchanged() {
    if !has_cuda_device() {
        eprintln!("Skipping: no CUDA device");
        return;
    }

    // Explicit Rejection with root evidence — should still work (old behavior)
    let src = r#"
0.7::rain().
0.2::sprinkler().
evidence(sprinkler(), true).
query(rain()).
"#;
    let program = McProgram::compile_source(src).unwrap();
    let cfg = McEvalConfig {
        samples: 50_000,
        seed: 7,
        confidence: 0.95,
        max_nonmonotone_iterations: 128,
        sampling_method: Some(McSamplingMethod::Rejection),
    };
    let result = program.evaluate(cfg).unwrap();
    assert_eq!(result.sampling_method, McSamplingMethod::Rejection);
    let p = prob_of_atom(&result, "rain");
    assert!((p - 0.7).abs() < 0.02, "p={}", p);
    // Under rejection, evidence_samples < total_samples (sprinkler satisfied in ~20% of worlds)
    assert!(result.evidence_samples < result.total_samples);
}

#[test]
fn test_evidence_clamping_ad_head_3way() {
    if !has_cuda_device() {
        eprintln!("Skipping: no CUDA device");
        return;
    }

    // 3-head AD + none: 0.2::color(red); 0.3::color(blue); 0.4::color(green).
    // evidence(color(blue), true) -> clamp, every sample has color(blue)=true
    // P(color(red) | color(blue)) = 0  (AD is exclusive)
    // P(color(green) | color(blue)) = 0
    let src = r#"
0.2::color(red); 0.3::color(blue); 0.4::color(green).
evidence(color(blue), true).
query(color(red)).
query(color(green)).
"#;
    let program = McProgram::compile_source(src).unwrap();
    let cfg = McEvalConfig {
        samples: 10_000,
        seed: 42,
        confidence: 0.95,
        max_nonmonotone_iterations: 128,
        sampling_method: None,
    };
    let result = program.evaluate(cfg).unwrap();

    assert_eq!(result.sampling_method, McSamplingMethod::EvidenceClamping);
    assert_eq!(result.evidence_samples, result.total_samples);

    // Under clamped evidence, color(blue) is always true, others always false
    let p_red = result.query_estimates.iter()
        .find(|q| q.atom.predicate == "color" && !q.atom.args.is_empty()
            && value_as_symbol_name(&q.atom.args[0]).as_deref() == Some("red"))
        .expect("missing query for color(red)")
        .prob;
    let p_green = result.query_estimates.iter()
        .find(|q| q.atom.predicate == "color" && !q.atom.args.is_empty()
            && value_as_symbol_name(&q.atom.args[0]).as_deref() == Some("green"))
        .expect("missing query for color(green)")
        .prob;

    assert_eq!(p_red, 0.0, "P(color(red) | color(blue)) should be 0.0");
    assert_eq!(p_green, 0.0, "P(color(green) | color(blue)) should be 0.0");
}
