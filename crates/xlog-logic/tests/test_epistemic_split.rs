use xlog_core::XlogError;
use xlog_ir::ExecutionPlan;
use xlog_logic::epistemic::{
    build_epistemic_dependency_graph, compile_epistemic_gpu_split_execution,
    split_epistemic_program, EpistemicComponentMergeReason,
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
fn constraint_dependency_coalesces_epistemic_split_components() {
    let program = parse_program(
        r#"
        a() :- know p().
        b() :- know q().
        :- a(), b().
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
fn split_component_constraints_stay_with_own_component() {
    let program = parse_program(
        r#"
        a() :- know p().
        b() :- know q().
        :- a().
        "#,
    )
    .unwrap();

    let split = compile_epistemic_gpu_split_execution(&program).unwrap();

    assert_eq!(split.components.len(), 2);
    assert_eq!(
        compiled_rule_count(&split.components[0].executable.reduced_runtime_plan),
        2,
        "the a/p component owns its integrity constraint"
    );
    assert_eq!(
        compiled_rule_count(&split.components[1].executable.reduced_runtime_plan),
        1,
        "the b/q component must not inherit the a-only constraint"
    );
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
fn shared_extensional_inputs_do_not_coalesce_epistemic_split_components() {
    let program = parse_program(
        r#"
        pred node(u32).
        pred edge(u32).
        pred color(u32).
        pred a(u32).
        pred b(u32).
        a(X) :- node(X), know edge(X).
        b(X) :- node(X), know color(X).
        "#,
    )
    .unwrap();

    let split = compile_epistemic_gpu_split_execution(&program).unwrap();
    let component_predicates: Vec<Vec<String>> = split
        .components
        .iter()
        .map(|component| component.component.predicates.clone())
        .collect();
    let component_rule_indices: Vec<Vec<usize>> = split
        .components
        .iter()
        .map(|component| component.component.rule_indices.clone())
        .collect();

    assert_eq!(split.components.len(), 2);
    assert_eq!(split.recomposed_rule_indices(), vec![0, 1]);
    assert_eq!(component_rule_indices, vec![vec![0], vec![1]]);
    assert_eq!(
        component_predicates,
        vec![
            vec!["a".to_string(), "edge".to_string(), "node".to_string()],
            vec!["b".to_string(), "color".to_string(), "node".to_string()],
        ]
    );
    for component in &split.components {
        assert_eq!(component.executable.gpu_plan.epistemic_literals.len(), 1);
    }
}

#[test]
fn split_executable_components_recompose_in_source_rule_order() {
    let program = parse_program(
        r#"
        pred z_seed(u32).
        pred z_gate(u32).
        pred z_out(u32).
        pred a_seed(u32).
        pred a_gate(u32).
        pred a_out(u32).
        z_out(X) :- z_seed(X), know z_gate(X).
        a_out(X) :- a_seed(X), know a_gate(X).
        "#,
    )
    .unwrap();

    let split = compile_epistemic_gpu_split_execution(&program).unwrap();
    let component_rule_indices: Vec<Vec<usize>> = split
        .components
        .iter()
        .map(|component| component.component.rule_indices.clone())
        .collect();
    assert_eq!(component_rule_indices, vec![vec![1], vec![0]]);

    let recomposed_rule_indices: Vec<Vec<usize>> = split
        .recomposed_components()
        .iter()
        .map(|component| component.component.rule_indices.clone())
        .collect();
    assert_eq!(recomposed_rule_indices, vec![vec![0], vec![1]]);
}

#[test]
fn shared_modal_inputs_coalesce_epistemic_split_components() {
    let program = parse_program(
        r#"
        pred node(u32).
        pred color(u32).
        pred q(u32).
        pred a(u32).
        pred b(u32).
        a(X) :- node(X), know q(X).
        b(X) :- color(X), possible q(X).
        "#,
    )
    .unwrap();

    let split = split_epistemic_program(&program).unwrap();

    assert_eq!(split.components.len(), 1);
    assert_eq!(split.recomposed_rule_indices(), vec![0, 1]);
    assert_eq!(
        split.components[0].predicates,
        vec![
            "a".to_string(),
            "b".to_string(),
            "color".to_string(),
            "node".to_string(),
            "q".to_string(),
        ]
    );
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

#[test]
fn invalid_cross_arity_modal_coupling_returns_typed_rejection() {
    let program = parse_program(
        r#"
        pred a(u32, u32).
        pred p(u32).
        pred p(u32, u32).
        a(X, Y) :- know p(X), possible p(X, Y).
        "#,
    )
    .unwrap();
    let err = split_epistemic_program(&program).unwrap_err();

    match err {
        XlogError::UnsupportedEpistemicConstruct {
            construct, context, ..
        } => {
            assert_eq!(construct, "epistemic splitting");
            assert!(context.contains("p/1"));
            assert!(context.contains("p/2"));
        }
        other => panic!("expected typed split rejection, got {other:?}"),
    }
}

// --- EGB-05 K2 coverage / K3 diagnostics / source-order stability pilots ---

#[test]
fn independent_split_components_carry_no_merge_reasons() {
    let program = parse_program(
        r#"
        a() :- know p().
        b() :- know q().
        "#,
    )
    .unwrap();
    let split = split_epistemic_program(&program).unwrap();
    assert_eq!(split.components.len(), 2);
    for component in &split.components {
        assert!(
            component.merge_reasons.is_empty(),
            "independent split-out component {:?} must record no coalesce reason",
            component.predicates
        );
    }
}

#[test]
fn derived_dependency_coalesce_explains_why_in_merge_reasons() {
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
        split.components[0].merge_reasons,
        vec![EpistemicComponentMergeReason::DerivedPredicate {
            predicate: "a".to_string(),
        }],
        "the b<-a derived dependency must be the explained coalesce reason"
    );
}

#[test]
fn shared_modal_coalesce_explains_modal_predicate_with_arity() {
    let program = parse_program(
        r#"
        pred node(u32).
        pred color(u32).
        pred q(u32).
        pred a(u32).
        pred b(u32).
        a(X) :- node(X), know q(X).
        b(X) :- color(X), possible q(X).
        "#,
    )
    .unwrap();
    let split = split_epistemic_program(&program).unwrap();
    assert_eq!(split.components.len(), 1);
    assert_eq!(
        split.components[0].merge_reasons,
        vec![EpistemicComponentMergeReason::SharedModalPredicate {
            predicate: "q/1".to_string(),
        }],
        "shared modal predicate q/1 must be the explained coalesce reason"
    );
}

#[test]
fn constraint_coalesce_explains_named_heads() {
    let program = parse_program(
        r#"
        a() :- know p().
        b() :- know q().
        :- a(), b().
        "#,
    )
    .unwrap();
    let split = split_epistemic_program(&program).unwrap();
    assert_eq!(split.components.len(), 1);
    let reasons = &split.components[0].merge_reasons;
    let constraint_reason = reasons
        .iter()
        .find_map(|reason| match reason {
            EpistemicComponentMergeReason::Constraint { predicates } => Some(predicates.clone()),
            _ => None,
        })
        .expect("constraint coalesce must be explained");
    assert!(constraint_reason.contains(&"a".to_string()));
    assert!(constraint_reason.contains(&"b".to_string()));
}

#[test]
fn recomposition_covers_each_relevant_rule_exactly_once() {
    let program = parse_program(
        r#"
        pred x_seed(u32).
        pred x_gate(u32).
        pred x_out(u32).
        pred y_seed(u32).
        pred y_gate(u32).
        pred y_out(u32).
        pred z_seed(u32).
        pred z_gate(u32).
        pred z_out(u32).
        x_out(N) :- x_seed(N), know x_gate(N).
        y_out(N) :- y_seed(N), know y_gate(N).
        z_out(N) :- z_seed(N), know z_gate(N).
        "#,
    )
    .unwrap();
    let split = split_epistemic_program(&program).unwrap();
    assert_eq!(split.components.len(), 3);

    let mut flat: Vec<usize> = split
        .components
        .iter()
        .flat_map(|component| component.rule_indices.iter().copied())
        .collect();
    flat.sort_unstable();
    // No omissions and (by union-find construction) no duplicates: the
    // recomposition is an exact permutation of every source rule index.
    assert_eq!(flat, vec![0, 1, 2]);
    assert_eq!(flat.len(), program.rules.len());
    let mut deduped = flat.clone();
    deduped.dedup();
    assert_eq!(deduped, flat, "each source rule must appear exactly once");
    assert_eq!(split.recomposed_rule_indices(), vec![0, 1, 2]);
}

#[test]
fn split_components_are_stable_under_source_rule_permutation() {
    // Same three independent epistemic rules, declared in a permuted source
    // order. Components and recomposition must be identical regardless of the
    // accidental source order (deterministic split, no order-dependence).
    let canonical = parse_program(
        r#"
        pred a_seed(u32).
        pred a_gate(u32).
        pred a_out(u32).
        pred b_seed(u32).
        pred b_gate(u32).
        pred b_out(u32).
        pred c_seed(u32).
        pred c_gate(u32).
        pred c_out(u32).
        a_out(N) :- a_seed(N), know a_gate(N).
        b_out(N) :- b_seed(N), know b_gate(N).
        c_out(N) :- c_seed(N), know c_gate(N).
        "#,
    )
    .unwrap();
    let permuted = parse_program(
        r#"
        pred a_seed(u32).
        pred a_gate(u32).
        pred a_out(u32).
        pred b_seed(u32).
        pred b_gate(u32).
        pred b_out(u32).
        pred c_seed(u32).
        pred c_gate(u32).
        pred c_out(u32).
        c_out(N) :- c_seed(N), know c_gate(N).
        a_out(N) :- a_seed(N), know a_gate(N).
        b_out(N) :- b_seed(N), know b_gate(N).
        "#,
    )
    .unwrap();

    let canonical_split = split_epistemic_program(&canonical).unwrap();
    let permuted_split = split_epistemic_program(&permuted).unwrap();

    // Components are sorted by predicate set, so the predicate-keyed view is
    // order-stable even though the underlying rule indices differ per source.
    let canonical_predicates: Vec<Vec<String>> = canonical_split
        .components
        .iter()
        .map(|component| component.predicates.clone())
        .collect();
    let permuted_predicates: Vec<Vec<String>> = permuted_split
        .components
        .iter()
        .map(|component| component.predicates.clone())
        .collect();
    assert_eq!(canonical_predicates, permuted_predicates);
    assert_eq!(canonical_split.components.len(), 3);
    assert_eq!(permuted_split.components.len(), 3);
    // Recomposition recovers the full rule set in both source orders.
    assert_eq!(canonical_split.recomposed_rule_indices(), vec![0, 1, 2]);
    assert_eq!(permuted_split.recomposed_rule_indices(), vec![0, 1, 2]);
}

#[test]
fn executable_recomposition_covers_only_executed_epistemic_components() {
    // A pure-ordinary independent component and an independent epistemic
    // component. The epistemic split executable materializes one output buffer
    // per epistemic component, so the ordinary component (rule 0) is NOT part of
    // the epistemic execution surface and must be excluded from the executable
    // recomposition view -- otherwise coverage would silently over-report a rule
    // the executable never runs.
    let program = parse_program(
        r#"
        pred base(u32).
        pred ordinary(u32).
        pred seed(u32).
        pred gate(u32).
        pred epi(u32).
        ordinary(X) :- base(X).
        epi(X) :- seed(X), know gate(X).
        "#,
    )
    .unwrap();

    // Planning view: the full dependency graph keeps both independent components.
    let split_plan = split_epistemic_program(&program).unwrap();
    assert_eq!(split_plan.components.len(), 2);
    assert_eq!(split_plan.recomposed_rule_indices(), vec![0, 1]);

    // Executable view: only the epistemic-bearing component (rule 1) is run.
    let exec = compile_epistemic_gpu_split_execution(&program).unwrap();
    assert_eq!(exec.components.len(), 1);
    assert_eq!(exec.components[0].component.rule_indices, vec![1]);
    // Executable recomposition reflects what is executed -- exactly the epistemic
    // rule, no silent over-reporting of the unexecuted ordinary rule.
    assert_eq!(exec.recomposed_rule_indices(), vec![1]);
    // The full planning recomposition view remains available and unchanged.
    assert_eq!(exec.planned_recomposed_rule_indices(), vec![0, 1]);
}

fn compiled_rule_count(plan: &ExecutionPlan) -> usize {
    plan.rules_by_scc.iter().map(Vec::len).sum()
}
