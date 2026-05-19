use xlog_core::XlogError;
use xlog_ir::ExecutionPlan;
use xlog_logic::epistemic::{
    build_epistemic_dependency_graph, compile_epistemic_gpu_split_execution,
    split_epistemic_program,
};
use xlog_logic::parse_program;

#[test]
fn split_graph_builds_deterministic_independent_components() {
    let program = parse_program(
        r#"
        a() :- know p().
        b() :- know q().
        "#,
    )
    .unwrap();

    let graph = build_epistemic_dependency_graph(&program).unwrap();
    let components: Vec<Vec<String>> = graph
        .components
        .iter()
        .map(|component| component.predicates.clone())
        .collect();

    assert_eq!(
        components,
        vec![
            vec!["a".to_string(), "p".to_string()],
            vec!["b".to_string(), "q".to_string()],
        ]
    );
}

#[test]
fn valid_split_recomposes_to_unsplit_rule_order() {
    let program = parse_program(
        r#"
        a() :- know p().
        b() :- know q().
        "#,
    )
    .unwrap();

    let split = split_epistemic_program(&program).unwrap();
    assert_eq!(split.components.len(), 2);
    assert_eq!(split.recomposed_rule_indices(), vec![0, 1]);
}

#[test]
fn relational_dependency_coalesces_epistemic_split_components() {
    let program = parse_program(
        r#"
        a() :- know p().
        b() :- a(), know q().
        "#,
    )
    .unwrap();

    let split = split_epistemic_program(&program).unwrap();

    assert_eq!(split.components.len(), 1);
    assert_eq!(
        split.components[0].predicates,
        vec![
            "a".to_string(),
            "b".to_string(),
            "p".to_string(),
            "q".to_string()
        ]
    );
    assert_eq!(split.recomposed_rule_indices(), vec![0, 1]);
}

#[test]
fn valid_split_components_compile_through_gpu_executable_subplans() {
    let program = parse_program(
        r#"
        a() :- know p().
        b() :- possible q().
        "#,
    )
    .unwrap();

    let split = compile_epistemic_gpu_split_execution(&program).unwrap();

    assert_eq!(split.components.len(), 2);
    assert_eq!(split.recomposed_rule_indices(), vec![0, 1]);
    for component in &split.components {
        assert!(component.executable.gpu_plan.cpu_fallbacks.is_zero());
        assert_eq!(component.executable.gpu_plan.epistemic_literals.len(), 1);
        assert_eq!(
            compiled_rule_count(&component.executable.reduced_runtime_plan),
            1
        );
    }
}

#[test]
fn invalid_cross_component_split_returns_typed_rejection() {
    let program = parse_program("a() :- know p(), possible q().").unwrap();
    let err = split_epistemic_program(&program).unwrap_err();

    match err {
        XlogError::UnsupportedEpistemicConstruct { construct, .. } => {
            assert_eq!(construct, "epistemic splitting");
        }
        other => panic!("expected typed split rejection, got {other:?}"),
    }
}

fn compiled_rule_count(plan: &ExecutionPlan) -> usize {
    plan.rules_by_scc.iter().map(Vec::len).sum()
}
