use xlog_logic::epistemic::{
    evaluate_epistemic_literal, plan_epistemic_gpu_execution, EpistemicInterpretation, TruthValue,
};
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
fn g91_accepts_self_support_that_faeel_rejects_on_production_path() {
    // EGB-07 K3: the SAME self-supporting modal program must reject under the
    // default FAEEL foundedness guard but be accepted under explicit g91
    // compatibility mode. G91 behavior must not leak into FAEEL defaults.
    let self_support = "p() :- possible p().";

    let faeel = parse_program(self_support).unwrap();
    assert!(
        plan_epistemic_gpu_execution(&faeel).is_err(),
        "default FAEEL must reject unfounded self-support"
    );

    let g91 = parse_program(&format!("#pragma epistemic_mode = g91\n{self_support}")).unwrap();
    plan_epistemic_gpu_execution(&g91)
        .expect("g91 compatibility mode must accept the same self-support");
}

#[test]
fn g91_accepts_nonzero_self_support_that_faeel_rejects_on_production_path() {
    let self_support = "p(X) :- dom(X), possible p(X).\ndom(1).";

    let faeel = parse_program(self_support).unwrap();
    assert!(
        plan_epistemic_gpu_execution(&faeel).is_err(),
        "default FAEEL must reject unfounded nonzero self-support"
    );

    let g91 = parse_program(&format!("#pragma epistemic_mode = g91\n{self_support}")).unwrap();
    plan_epistemic_gpu_execution(&g91)
        .expect("g91 compatibility mode must accept the same nonzero self-support");
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
