use xlog_core::XlogError;
use xlog_ir::ExecutionPlan;
use xlog_logic::epistemic::{
    build_epistemic_dependency_graph, classify_recursive_epistemic_program,
    compile_epistemic_gpu_split_execution, plan_epistemic_gpu_execution, split_epistemic_program,
    try_reduce_case_a_recursive_epistemic_program, EpistemicComponentMergeReason,
    RecursiveEpistemicClass,
};
use xlog_logic::{parse_program, BodyLiteral};

#[test]
fn non_case_a_recursive_epistemic_program_fails_closed() {
    // Recursion whose modal literal ranges over a NON-INVARIANT relation (here the
    // modal atom `know derived_edge(...)` references a relation that itself depends on
    // the recursive `reach`) is outside the admissible Case-A fragment: its gated
    // extension changes as the recursion iterates, so it must fail closed with a typed
    // diagnostic rather than be silently mis-evaluated.
    let program = parse_program(
        r#"
        pred vertex(u32).
        pred edge(u32, u32).
        pred reach(u32, u32).
        pred derived_edge(u32, u32).
        vertex(1). vertex(2). edge(1, 2).
        derived_edge(X, Y) :- reach(X, Y).
        reach(X, Y) :- vertex(X), vertex(Y), know edge(X, Y).
        reach(X, Z) :- reach(X, Y), vertex(Z), know derived_edge(Y, Z).
        "#,
    )
    .unwrap();
    match classify_recursive_epistemic_program(&program) {
        Err(XlogError::UnsupportedEpistemicConstruct { construct, context }) => {
            assert_eq!(construct, "recursive epistemic program");
            assert!(
                context.contains("not invariant"),
                "expected non-invariant modal diagnostic, got: {context}"
            );
        }
        other => panic!("expected typed non-Case-A recursive rejection, got {other:?}"),
    }
    // The single-pass GPU planner must also reject any ordinary recursion.
    assert!(matches!(
        plan_epistemic_gpu_execution(&program),
        Err(XlogError::UnsupportedEpistemicConstruct { .. })
    ));
}

#[test]
fn negated_modal_recursive_epistemic_program_fails_closed() {
    // A negated modal literal in a recursion-participating rule is not Case A (the
    // gated complement is not invariant) and must fail closed.
    let program = parse_program(
        r#"
        pred vertex(u32).
        pred edge(u32, u32).
        pred reach(u32, u32).
        vertex(1). vertex(2). edge(1, 2).
        reach(X, Y) :- vertex(X), vertex(Y), know edge(X, Y).
        reach(X, Z) :- reach(X, Y), vertex(Z), not know edge(Y, Z).
        "#,
    )
    .unwrap();
    match classify_recursive_epistemic_program(&program) {
        Err(XlogError::UnsupportedEpistemicConstruct { construct, context }) => {
            assert_eq!(construct, "recursive epistemic program");
            assert!(
                context.contains("NEGATED modal"),
                "expected negated-modal diagnostic, got: {context}"
            );
        }
        other => panic!("expected typed negated-modal rejection, got {other:?}"),
    }
}

#[test]
fn recursion_with_non_invariant_modal_in_unrelated_rule_fails_closed() {
    // Soundness guard: the Case-A reduction rewrites EVERY modal literal in the
    // program to a positive atom over its gated relation, so a modal literal in a
    // rule that does NOT itself participate in the recursion must still be invariant.
    // Here `maybe(X) :- node(X), possible choice(X)` ranges over `choice`, which is
    // epistemic-defined (non-invariant), so blanket-rewriting `possible choice(X)`
    // would be unsound. The whole program must therefore fail closed even though the
    // recursive `reach` rules are themselves Case A.
    let program = parse_program(
        r#"
        pred vertex(u32).
        pred edge(u32, u32).
        pred reach(u32, u32).
        pred node(u32).
        pred choice(u32).
        pred maybe(u32).
        pred seed(u32).
        vertex(1). vertex(2). edge(1, 2).
        node(1). seed(1).
        choice(X) :- seed(X), know edge(X, X).
        reach(X, Y) :- vertex(X), vertex(Y), know edge(X, Y).
        reach(X, Z) :- reach(X, Y), vertex(Z), know edge(Y, Z).
        maybe(X) :- node(X), possible choice(X).
        "#,
    )
    .unwrap();
    match classify_recursive_epistemic_program(&program) {
        Err(XlogError::UnsupportedEpistemicConstruct { construct, context }) => {
            assert_eq!(construct, "recursive epistemic program");
            assert!(
                context.contains("not invariant"),
                "expected non-invariant modal diagnostic, got: {context}"
            );
            assert!(
                context.contains("choice"),
                "diagnostic should name the offending modal relation, got: {context}"
            );
        }
        other => panic!("expected typed fail-closed for non-invariant modal, got {other:?}"),
    }
}

#[test]
fn case_a_recursive_epistemic_program_is_accepted_and_reduced() {
    // Case A: ordinary recursion in `reach`, with every recursion-participating modal
    // atom (`know edge(...)`) over the INVARIANT EDB relation `edge`. The program is
    // classified Case A and reduced to an ordinary recursive program whose modal
    // literals are resolved to positive atoms over their gated relation, so the
    // existing recursive/semi-naive engine derives the transitive closure (including
    // multi-hop tuples), NOT a single pass.
    let program = parse_program(
        r#"
        pred vertex(u32).
        pred edge(u32, u32).
        pred reach(u32, u32).
        vertex(1). vertex(2). vertex(3).
        edge(1, 2). edge(2, 3).
        reach(X, Y) :- vertex(X), vertex(Y), know edge(X, Y).
        reach(X, Z) :- reach(X, Y), vertex(Z), know edge(Y, Z).
        ?- reach(X, Y).
        "#,
    )
    .unwrap();

    assert_eq!(
        classify_recursive_epistemic_program(&program).unwrap(),
        RecursiveEpistemicClass::CaseA
    );

    let reduced = try_reduce_case_a_recursive_epistemic_program(&program)
        .unwrap()
        .expect("Case-A program must reduce to an ordinary recursive program");

    // No epistemic literals survive: each `know edge(...)` is resolved to a positive
    // ordinary atom over the invariant `edge` relation.
    let modal_literals = reduced
        .rules
        .iter()
        .flat_map(|rule| rule.body.iter())
        .filter(|lit| matches!(lit, BodyLiteral::Epistemic(_)))
        .count();
    assert_eq!(
        modal_literals, 0,
        "Case-A reduce must remove all modal literals"
    );

    // The recursive rule now joins `reach` against the gated `edge` relation in-loop
    // (positive atom), preserving head arity 2 across both rules — the fix for the
    // single-pass arity mismatch.
    let recursive_rule = reduced
        .rules
        .iter()
        .find(|rule| {
            rule.head.predicate == "reach"
                && rule.body.iter().any(
                    |lit| matches!(lit, BodyLiteral::Positive(atom) if atom.predicate == "reach"),
                )
        })
        .expect("recursive reach rule must survive reduction");
    assert!(recursive_rule
        .body
        .iter()
        .any(|lit| matches!(lit, BodyLiteral::Positive(atom) if atom.predicate == "edge")));
    for rule in reduced.rules.iter().filter(|r| r.head.predicate == "reach") {
        assert_eq!(rule.head.terms.len(), 2, "reach head arity must stay 2");
    }
}

#[test]
fn modal_self_support_is_not_treated_as_ordinary_recursion() {
    // EGB-07: self-support THROUGH a modal literal (`founded() :- possible founded().`)
    // is not ordinary recursion and is handled by FAEEL foundedness; with an
    // independent founded support it must NOT be rejected by the recursion guard.
    let program = parse_program(
        r#"
        pred base().
        pred founded().
        base().
        founded() :- base().
        founded() :- possible founded().
        "#,
    )
    .unwrap();
    plan_epistemic_gpu_execution(&program)
        .expect("modal self-support with independent foundation must not be rejected as recursion");
}

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
fn source_facts_do_not_coalesce_bound_membership_split_components() {
    let program = parse_program(
        r#"
        pred node(u32).
        pred edge(u32).
        pred color(u32).
        pred alt(u32).
        pred blocked(u32).
        pred both_known(u32).
        pred safe_alt(u32).

        node(1). node(2). node(3).
        edge(1). edge(2).
        color(1).
        alt(2).
        blocked(3).

        both_known(X) :- node(X), know edge(X), know color(X).
        safe_alt(X) :- node(X), possible alt(X), not possible blocked(X).
        "#,
    )
    .unwrap();

    let split = compile_epistemic_gpu_split_execution(&program).unwrap();
    let recomposed_components = split.recomposed_components();
    let component_rule_indices: Vec<Vec<usize>> = recomposed_components
        .iter()
        .map(|component| component.component.rule_indices.clone())
        .collect();
    let literal_counts: Vec<usize> = recomposed_components
        .iter()
        .map(|component| component.executable.gpu_plan.epistemic_literals.len())
        .collect();

    assert_eq!(split.components.len(), 2);
    assert_eq!(component_rule_indices, vec![vec![8], vec![9]]);
    assert_eq!(literal_counts, vec![2, 2]);
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
fn cross_component_modal_coupling_is_solved_jointly_in_one_component() {
    // EGB-06: a rule coupling more than one DISTINCT epistemic body predicate is
    // no longer rejected by the split layer. The dependency graph unions all such
    // modal predicates into a single component that the unsplit joint path solves
    // as a full modal conjunction over the candidate world view.
    let program = parse_program("a() :- know p(), possible q().").unwrap();
    let split = split_epistemic_program(&program).expect("multi-predicate rule splits jointly");

    assert_eq!(split.components.len(), 1);
    assert_eq!(split.recomposed_rule_indices(), vec![0]);
    assert_eq!(
        split.components[0].predicates,
        vec!["a".to_string(), "p".to_string(), "q".to_string()]
    );
}

#[test]
fn safe_cross_arity_modal_coupling_splits_into_one_joint_component() {
    // EGB-06: cross-arity coupling of the same predicate (p/1 + p/2) is a valid
    // joint condition when the shared variable is safely bound by a relational
    // body atom; the two arities materialize as distinct relations and the rule
    // is jointly solved in a single component.
    let program = parse_program(
        r#"
        pred a(u32, u32).
        pred seed(u32, u32).
        pred p(u32).
        pred p(u32, u32).
        a(X, Y) :- seed(X, Y), know p(X), possible p(X, Y).
        "#,
    )
    .unwrap();
    let split = split_epistemic_program(&program).expect("safe cross-arity rule splits jointly");

    assert_eq!(split.components.len(), 1);
    assert_eq!(split.recomposed_rule_indices(), vec![0]);
    assert!(split.components[0].predicates.contains(&"a".to_string()));
    assert!(split.components[0].predicates.contains(&"p".to_string()));
    assert!(split.components[0].predicates.contains(&"seed".to_string()));
}

#[test]
fn unsafe_cross_arity_modal_coupling_fails_closed_at_joint_compile() {
    // EGB-06: removing the blanket coupling rejection must NOT make unsound rules
    // pass. An unsafe shared variable (X appears only in modal literals, needs to
    // bind the head) is now caught by the joint-compile layer's safety analysis
    // (relocated diagnostic), not silently accepted.
    let program = parse_program(
        r#"
        pred a(u32, u32).
        pred p(u32).
        pred p(u32, u32).
        a(X, Y) :- know p(X), possible p(X, Y).
        "#,
    )
    .unwrap();

    // The split layer itself accepts (no coupling rejection); the unsafe variable
    // surfaces from the joint-compile path that each component is recompiled
    // through.
    split_epistemic_program(&program).expect("split layer no longer rejects coupling");

    let err = compile_epistemic_gpu_split_execution(&program)
        .expect_err("unsafe cross-arity coupling must fail closed at joint compile");
    match err {
        XlogError::UnsafeVariable(var) => assert_eq!(var, "X"),
        other => panic!("expected relocated unsafe-variable diagnostic, got {other:?}"),
    }
}

#[test]
fn nested_modal_in_joint_coupling_rule_fails_closed_upstream_of_split() {
    // EGB-06 K4: removing the blanket coupling rejection must NOT let a nested-modal
    // construct inside a multi-epistemic-predicate rule slip through. Nested modals
    // are rejected at PARSE time (EGB-03), upstream of the coupling gate, so the
    // joint-coupling rule never reaches `split_epistemic_program`. This confirms the
    // nested-modal boundary is unaffected by the coupling-gate removal.
    let err = parse_program("h(X) :- know p(X), possible know q(X).").unwrap_err();
    match err {
        XlogError::UnsupportedEpistemicConstruct { construct, .. } => {
            assert_eq!(construct, "nested epistemic literal");
        }
        other => panic!("expected nested-modal parse diagnostic, got {other:?}"),
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
