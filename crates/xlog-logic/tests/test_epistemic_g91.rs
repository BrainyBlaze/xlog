use xlog_logic::epistemic::{evaluate_epistemic_literal, EpistemicInterpretation, TruthValue};
use xlog_logic::{parse_program, BodyLiteral, Compiler, EpistemicMode};

#[test]
fn g91_mode_is_selected_explicitly() {
    let program = parse_program(
        r#"
        #pragma epistemic_mode = g91
        believed() :- know fact().
        "#,
    )
    .unwrap();

    assert_eq!(
        program.directives.epistemic_mode_or_default(),
        EpistemicMode::G91
    );
}

#[test]
fn g91_possible_fixture_differs_from_faeel_default() {
    let program = parse_program("believed() :- possible fact().").unwrap();
    let BodyLiteral::Epistemic(lit) = &program.rules[0].body[0] else {
        panic!("expected epistemic literal");
    };

    let interpretation = EpistemicInterpretation::new().with_possible("fact", 0);

    assert_eq!(
        evaluate_epistemic_literal(EpistemicMode::G91, lit, &interpretation),
        TruthValue::True
    );
    assert_eq!(
        evaluate_epistemic_literal(EpistemicMode::Faeel, lit, &interpretation),
        TruthValue::False
    );
}

#[test]
fn g91_mode_does_not_change_non_epistemic_compile_output() {
    let default_plan = Compiler::new().compile("edge(1, 2).").unwrap();
    let g91_plan = Compiler::new()
        .compile(
            r#"
            #pragma epistemic_mode = g91
            edge(1, 2).
            "#,
        )
        .unwrap();

    assert_eq!(format!("{default_plan:?}"), format!("{g91_plan:?}"));
}
