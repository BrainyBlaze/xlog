use xlog_ir::eir::{EirBodyLiteral, EirEpistemicMode, EirEpistemicOp};
use xlog_logic::ast::{BodyLiteral, EpistemicMode};
use xlog_logic::epistemic::{
    evaluate_epistemic_literal, evaluate_faeel_candidate, run_generate_propagate_test,
    split_epistemic_program, EpistemicInterpretation, FaeelCandidateResult,
    GeneratePropagateTestConfig, TruthValue,
};
use xlog_logic::{build_eir, parse_program};

#[test]
fn eir_boundary_example_builds_explicit_epistemic_ir() {
    let program = parse_program(include_str!(
        "../../../examples/epistemic/01-eir-boundary.xlog"
    ))
    .unwrap();

    let eir = build_eir(&program).unwrap();

    assert_eq!(eir.mode, EirEpistemicMode::Faeel);
    let lit = eir
        .rules
        .iter()
        .flat_map(|rule| rule.body.iter())
        .find_map(|lit| match lit {
            EirBodyLiteral::Epistemic(lit) => Some(lit),
            _ => None,
        })
        .expect("example should include an explicit epistemic literal");
    assert_eq!(lit.op, EirEpistemicOp::Know);
    assert_eq!(lit.atom.predicate, "edge");
}

#[test]
fn g91_compatibility_example_accepts_possible_support() {
    let program = parse_program(include_str!(
        "../../../examples/epistemic/02-g91-compatibility.xlog"
    ))
    .unwrap();
    let BodyLiteral::Epistemic(lit) = &program.rules[0].body[0] else {
        panic!("expected epistemic body literal");
    };
    let interpretation = EpistemicInterpretation::new().with_possible("fact", 0);

    assert_eq!(
        evaluate_epistemic_literal(EpistemicMode::G91, lit, &interpretation),
        TruthValue::True
    );
}

#[test]
fn faeel_default_example_accepts_founded_candidate() {
    let program = parse_program(include_str!(
        "../../../examples/epistemic/03-faeel-default.xlog"
    ))
    .unwrap();
    let interpretation = EpistemicInterpretation::new().with_known("fact", 0);

    assert_eq!(
        evaluate_faeel_candidate(&program, &interpretation).unwrap(),
        FaeelCandidateResult::Model
    );
}

#[test]
fn gpt_example_reports_trace_counts() {
    let program = parse_program(include_str!(
        "../../../examples/epistemic/04-gpt-candidate-filter.xlog"
    ))
    .unwrap();
    let candidates = vec![
        EpistemicInterpretation::new().with_known("fact", 0),
        EpistemicInterpretation::new(),
    ];

    let outcome = run_generate_propagate_test(
        &program,
        candidates,
        GeneratePropagateTestConfig { max_candidates: 4 },
    )
    .unwrap();

    assert_eq!(outcome.trace.generated, 2);
    assert_eq!(outcome.trace.accepted, 1);
    assert_eq!(outcome.trace.rejected, 1);
}

#[test]
fn splitting_example_recomposes_source_rule_order() {
    let program = parse_program(include_str!(
        "../../../examples/epistemic/05-splitting.xlog"
    ))
    .unwrap();

    let split = split_epistemic_program(&program).unwrap();

    assert_eq!(split.components.len(), 2);
    assert_eq!(split.recomposed_rule_indices(), vec![0, 1]);
}
