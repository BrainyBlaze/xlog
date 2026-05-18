use xlog_core::XlogError;
use xlog_logic::epistemic::{build_epistemic_dependency_graph, split_epistemic_program};
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
