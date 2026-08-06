use xlog_logic::epistemic::{
    classify_recursive_epistemic_program, evaluate_epistemic_literal, plan_epistemic_gpu_execution,
    prepare_epistemic_program, reduce_epistemic_program_to_ordinary,
    try_prepare_g91_compatibility_reduction, try_reduce_case_a_recursive_epistemic_program,
    EpistemicInterpretation, RecursiveEpistemicClass, TruthValue,
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
    // The SAME self-supporting modal program executes to DIFFERENT results by
    // mode. Under default FAEEL the unfounded head is excluded from the founded
    // model (the circular self-support rule is dropped from the reduced base →
    // EMPTY extension). Under explicit g91 compatibility mode the circular
    // self-support is ACCEPTED (the rule is kept → p is true). This is the exact
    // FAEEL-vs-G91 mode difference. High-level production GPU/CLI coverage asserts
    // the exact rows:0 versus rows:1 results.
    let self_support = "p() :- possible p().";

    let faeel = parse_program(self_support).unwrap();
    assert_eq!(
        classify_recursive_epistemic_program(&faeel).unwrap(),
        RecursiveEpistemicClass::ModalCycle
    );
    assert!(plan_epistemic_gpu_execution(&faeel).is_err());
    try_reduce_case_a_recursive_epistemic_program(&faeel)
        .unwrap()
        .expect("FAEEL self-support must reduce to its founded fixpoint");
    assert_eq!(
        founding_rule_count(
            &reduce_epistemic_program_to_ordinary(&faeel)
                .expect("FAEEL self-support fixture must reduce"),
            "p",
        ),
        0,
        "FAEEL drops the unfounded circular self-support rule (empty founded model)"
    );

    let g91 = parse_program(&format!("#pragma epistemic_mode = g91\n{self_support}")).unwrap();
    assert_eq!(
        classify_recursive_epistemic_program(&g91).unwrap(),
        RecursiveEpistemicClass::ModalCycle
    );
    let prepared = prepare_epistemic_program(&g91).expect("validate G91 self-support source");
    let compatibility = try_prepare_g91_compatibility_reduction(&prepared)
        .expect("prepare G91 compatibility")
        .expect("G91 self-support must select compatibility iteration");
    assert_eq!(
        founding_rule_count(compatibility.upper_bound_program(), "p"),
        1,
        "G91 upper bound keeps the compatibility-supported clause"
    );
    assert_eq!(compatibility.snapshot_relations().len(), 1);
}

#[test]
fn faeel_excludes_nonzero_self_support_that_g91_accepts_on_production_path() {
    let self_support = "p(X) :- dom(X), possible p(X).\ndom(1).";

    let faeel = parse_program(self_support).unwrap();
    assert_eq!(
        classify_recursive_epistemic_program(&faeel).unwrap(),
        RecursiveEpistemicClass::ModalCycle
    );
    assert!(plan_epistemic_gpu_execution(&faeel).is_err());
    assert_eq!(
        founding_rule_count(
            &reduce_epistemic_program_to_ordinary(&faeel)
                .expect("FAEEL nonzero self-support fixture must reduce"),
            "p",
        ),
        0,
        "FAEEL drops the unfounded nonzero circular self-support rule (empty founded model)"
    );

    let g91 = parse_program(&format!("#pragma epistemic_mode = g91\n{self_support}")).unwrap();
    assert_eq!(
        classify_recursive_epistemic_program(&g91).unwrap(),
        RecursiveEpistemicClass::ModalCycle
    );
    let prepared = prepare_epistemic_program(&g91).expect("validate G91 self-support source");
    let compatibility = try_prepare_g91_compatibility_reduction(&prepared)
        .expect("prepare G91 compatibility")
        .expect("G91 nonzero self-support must select compatibility iteration");
    assert_eq!(
        founding_rule_count(compatibility.upper_bound_program(), "p"),
        1,
        "G91 upper bound retains the domain-restricted compatibility clause"
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
