#![cfg(feature = "host-io")]

use xlog_logic::{parse_program, ProbMethod};
use xlog_prob::mc::{McEvalConfig, McProgram, McSamplingMethod};

fn prob_of(result: &xlog_prob::mc::McResult, predicate: &str) -> f64 {
    result
        .query_estimates
        .iter()
        .find(|q| q.atom.predicate == predicate)
        .unwrap_or_else(|| panic!("missing query estimate for {predicate}"))
        .prob
}

#[test]
fn mc_eval_config_reads_source_pragmas() {
    let program = parse_program(
        r#"
#pragma prob_samples = 256
#pragma prob_seed = 42
#pragma prob_confidence = 0.90
#pragma prob_method = rejection
#pragma prob_max_nonmonotone_iterations = 32
"#,
    )
    .expect("parse MC pragmas");

    let cfg = McEvalConfig::from_directives(&program.directives).expect("config from directives");
    assert_eq!(cfg.samples, 256);
    assert_eq!(cfg.seed, 42);
    assert!((cfg.confidence - 0.90).abs() < 1e-12);
    assert_eq!(cfg.sampling_method, Some(McSamplingMethod::Rejection));
    assert_eq!(cfg.max_nonmonotone_iterations, 32);
}

#[test]
fn mc_eval_config_maps_evidence_clamping_method() {
    let program =
        parse_program("#pragma prob_method = evidence_clamping").expect("parse method pragma");
    assert_eq!(
        program.directives.prob_method,
        Some(ProbMethod::EvidenceClamping)
    );
    let cfg = McEvalConfig::from_directives(&program.directives).expect("config from directives");
    assert_eq!(
        cfg.sampling_method,
        Some(McSamplingMethod::EvidenceClamping)
    );
}

#[test]
fn fixed_seed_mc_replay_is_deterministic() {
    let source = r#"
0.5::rain().
query(rain()).
"#;
    let program = McProgram::compile_source(source).expect("compile MC program");
    let mut cfg = McEvalConfig::default();
    cfg.samples = 512;
    cfg.seed = 85;
    cfg.sampling_method = Some(McSamplingMethod::Rejection);

    let first = program.evaluate(cfg.clone()).expect("first MC run");
    let second = program.evaluate(cfg).expect("second MC run");

    assert_eq!(first.total_samples, second.total_samples);
    assert_eq!(first.evidence_samples, second.evidence_samples);
    assert_eq!(first.query_estimates.len(), second.query_estimates.len());
    assert_eq!(
        first.query_estimates[0].prob,
        second.query_estimates[0].prob
    );
    assert_eq!(
        first.query_estimates[0].ci_low,
        second.query_estimates[0].ci_low
    );
    assert_eq!(
        first.query_estimates[0].ci_high,
        second.query_estimates[0].ci_high
    );
}

#[test]
fn approximate_count_aggregate_fixture_reports_confidence() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/v085-language/approx/aggregate_mc.xlog");
    let source = std::fs::read_to_string(path).expect("read approximate aggregate example");
    let parsed = parse_program(&source).expect("parse approximate aggregate example");
    let program = McProgram::compile_source(&source).expect("compile aggregate MC program");
    let cfg = McEvalConfig::from_directives(&parsed.directives).expect("MC config from pragmas");

    let result = program
        .evaluate(cfg)
        .expect("evaluate aggregate MC program");
    assert_eq!(result.total_samples, 64);
    assert_eq!(result.evidence_samples, 64);
    assert!((result.confidence - 0.90).abs() < 1e-12);
    assert!((prob_of(&result, "out_degree") - 1.0).abs() < 1e-12);
    assert!(result.query_estimates[0].ci_low <= 1.0);
    assert!(result.query_estimates[0].ci_high >= result.query_estimates[0].prob - 1e-12);
}
