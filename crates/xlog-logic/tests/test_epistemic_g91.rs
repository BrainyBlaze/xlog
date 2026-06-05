use xlog_logic::epistemic::{
    compile_epistemic_gpu_execution, evaluate_epistemic_literal, plan_epistemic_gpu_execution,
    reduce_epistemic_program_to_ordinary, EpistemicInterpretation, TruthValue,
};
use xlog_logic::{parse_program, BodyLiteral, Compiler, EpistemicMode};

/// Number of reduced ordinary rules that found `predicate` from a non-empty body.
fn founding_rule_count(program: &xlog_logic::ast::Program, predicate: &str) -> usize {
    program
        .rules
        .iter()
        .filter(|rule| rule.head.predicate == predicate && !rule.body.is_empty())
        .count()
}

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
fn faeel_excludes_self_support_that_g91_accepts_on_production_path() {
    // EGB-07 K3 / v0.9.2 ITEM B: the SAME self-supporting modal program executes to
    // DIFFERENT results by mode. Under default FAEEL the unfounded head is excluded
    // from the founded model (the circular self-support rule is dropped from the
    // reduced base → EMPTY extension). Under explicit g91 compatibility mode the
    // circular self-support is ACCEPTED (the rule is kept → p is true). This is the
    // exact FAEEL-vs-G91 mode difference (rows:0 vs rows:1 on device:
    // `parsed_faeel_unfounded_zero_arity_self_support_materializes_empty_on_gpu` vs
    // `parsed_g91_self_supported_possible_executes_on_gpu_runtime_values`).
    let self_support = "p() :- possible p().";

    let faeel = parse_program(self_support).unwrap();
    // FAEEL still PLANS (no rejection) — it is a defined empty result.
    plan_epistemic_gpu_execution(&faeel)
        .expect("FAEEL unfounded self-support is a defined empty result, not a rejection");
    compile_epistemic_gpu_execution(&faeel).expect("FAEEL must compile the empty founded model");
    assert_eq!(
        founding_rule_count(&reduce_epistemic_program_to_ordinary(&faeel), "p"),
        0,
        "FAEEL drops the unfounded circular self-support rule (empty founded model)"
    );

    let g91 = parse_program(&format!("#pragma epistemic_mode = g91\n{self_support}")).unwrap();
    plan_epistemic_gpu_execution(&g91)
        .expect("g91 compatibility mode must accept the same self-support");
    assert_eq!(
        founding_rule_count(&reduce_epistemic_program_to_ordinary(&g91), "p"),
        1,
        "G91 KEEPS the self-support rule (circular support accepted → p present)"
    );
}

#[test]
fn faeel_excludes_nonzero_self_support_that_g91_accepts_on_production_path() {
    let self_support = "p(X) :- dom(X), possible p(X).\ndom(1).";

    let faeel = parse_program(self_support).unwrap();
    plan_epistemic_gpu_execution(&faeel)
        .expect("FAEEL nonzero unfounded self-support is a defined empty result, not a rejection");
    assert_eq!(
        founding_rule_count(&reduce_epistemic_program_to_ordinary(&faeel), "p"),
        0,
        "FAEEL drops the unfounded nonzero circular self-support rule (empty founded model)"
    );

    let g91 = parse_program(&format!("#pragma epistemic_mode = g91\n{self_support}")).unwrap();
    plan_epistemic_gpu_execution(&g91)
        .expect("g91 compatibility mode must accept the same nonzero self-support");
    assert_eq!(
        founding_rule_count(&reduce_epistemic_program_to_ordinary(&g91), "p"),
        1,
        "G91 KEEPS the nonzero self-support rule (circular support accepted)"
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
