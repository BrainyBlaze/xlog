use xlog_logic::{parse_program, ProbEngine, ProbMethod};

#[test]
fn parses_approximate_inference_pragmas() {
    let program = parse_program(
        r#"
#pragma prob_engine = mc
#pragma prob_samples = 128
#pragma prob_seed = 85
#pragma prob_confidence = 0.90
#pragma prob_method = evidence_clamping
#pragma prob_max_nonmonotone_iterations = 64
"#,
    )
    .expect("parse approximate inference pragmas");

    assert_eq!(program.directives.prob_engine, Some(ProbEngine::Mc));
    assert_eq!(program.directives.prob_samples, Some(128));
    assert_eq!(program.directives.prob_seed, Some(85));
    assert_eq!(program.directives.prob_confidence, Some(0.90));
    assert_eq!(
        program.directives.prob_method,
        Some(ProbMethod::EvidenceClamping)
    );
    assert_eq!(program.directives.prob_max_nonmonotone_iterations, Some(64));
}

#[test]
fn rejects_invalid_probability_confidence_pragma() {
    let err = parse_program("#pragma prob_confidence = 1.0")
        .expect_err("confidence must be strictly inside 0..1")
        .to_string();
    assert!(err.contains("prob_confidence"), "err={err}");
    assert!(err.contains("0 < confidence < 1"), "err={err}");
}
