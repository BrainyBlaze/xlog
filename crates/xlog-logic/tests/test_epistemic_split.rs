use xlog_core::XlogError;
use xlog_ir::ExecutionPlan;
use xlog_logic::epistemic::{
    build_epistemic_dependency_graph, classify_recursive_epistemic_program,
    compile_epistemic_gpu_split_execution, plan_epistemic_gpu_execution,
    reduce_epistemic_program_to_ordinary, split_epistemic_program,
    try_reduce_case_a_recursive_epistemic_program, EpistemicComponentMergeReason,
    RecursiveEpistemicClass,
};
use xlog_logic::{parse_program, BodyLiteral};

#[test]
fn positive_modal_over_co_evolving_relation_is_accepted_case_b() {
    // v0.9.2 ITEM A: recursion whose POSITIVE modal `know derived_edge(...)` ranges
    // over a NON-INVARIANT relation that CO-EVOLVES with the recursion (`derived_edge`
    // depends on the recursive `reach`) is an admissible Case-B program: modal truth and
    // ordinary derivation co-evolve to a FOUNDED least fixpoint. The positive modal is
    // resolved into the recursive SCC and iterated on the semi-naive engine (the least
    // model of the resulting positive program IS its founded model). It is NOT rejected.
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
    assert_eq!(
        classify_recursive_epistemic_program(&program).unwrap(),
        RecursiveEpistemicClass::CaseB,
        "positive modal over a co-evolving relation is admissible Case B"
    );
    // Reduces to an ordinary recursive program: every positive modal resolves to a
    // positive ordinary atom (`know derived_edge` -> `derived_edge`), with no residual
    // modal literal, so the semi-naive engine computes the founded fixpoint.
    let reduced = try_reduce_case_a_recursive_epistemic_program(&program)
        .unwrap()
        .expect("admissible Case-B program reduces to an ordinary recursive program");
    let modal_literals = reduced
        .rules
        .iter()
        .flat_map(|rule| rule.body.iter())
        .filter(|lit| matches!(lit, BodyLiteral::Epistemic(_)))
        .count();
    assert_eq!(
        modal_literals, 0,
        "Case-B reduce removes all modal literals"
    );
    assert!(
        reduced.rules.iter().any(|rule| rule.body.iter().any(
            |lit| matches!(lit, BodyLiteral::Positive(atom) if atom.predicate == "derived_edge")
        )),
        "the co-evolving positive modal must resolve to a positive `derived_edge` atom"
    );
    // The SINGLE-PASS GPU planner still rejects any ordinary recursion (Case-B routes
    // through the ordinary recursive engine, not this planner).
    assert!(matches!(
        plan_epistemic_gpu_execution(&program),
        Err(XlogError::UnsupportedEpistemicConstruct { .. })
    ));
}

#[test]
fn negated_modal_over_invariant_in_recursion_is_accepted_case_a() {
    // K3: a NEGATED modal `not know edge(...)` over the INVARIANT EDB relation
    // `edge` in a recursion-participating rule equals ordinary `not edge(...)` (the
    // accepted world view agrees with `edge` on an invariant relation, so the gated
    // complement IS `not edge`). This is cleanly reducible to ordinary negation
    // (an anti-join, NO modal gating), so it is ADMISSIBLE Case A — it must NOT
    // fail closed.
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
    assert_eq!(
        classify_recursive_epistemic_program(&program).unwrap(),
        RecursiveEpistemicClass::CaseA,
        "negated modal over an invariant relation is admissible Case A"
    );
    // The Case-A reduction resolves `not know edge` to an ordinary NEGATED atom
    // (anti-join), with no residual modal literal.
    let reduced = try_reduce_case_a_recursive_epistemic_program(&program)
        .unwrap()
        .expect("admissible Case-A program reduces");
    let recursive_rule = &reduced.rules[reduced.rules.len() - 1];
    assert!(
        recursive_rule
            .body
            .iter()
            .any(|lit| matches!(lit, BodyLiteral::Negated(atom) if atom.predicate == "edge")),
        "negated modal must resolve to an ordinary negated `edge` atom"
    );
    assert!(
        !recursive_rule
            .body
            .iter()
            .any(|lit| matches!(lit, BodyLiteral::Epistemic(_))),
        "no residual modal literal after Case-A reduction"
    );
}

#[test]
fn negated_modal_over_non_invariant_in_recursion_fails_closed() {
    // The complement: a NEGATED modal over a NON-invariant (epistemic-derived)
    // relation in a recursion-participating program is doubly outside the Case-A
    // fragment and must still fail closed. `choice` is epistemic-defined, so
    // `not know choice` has no invariant complement.
    let program = parse_program(
        r#"
        pred vertex(u32).
        pred edge(u32, u32).
        pred reach(u32, u32).
        pred seed(u32, u32).
        pred choice(u32, u32).
        vertex(1). vertex(2). edge(1, 2). seed(1, 2).
        choice(X, Y) :- seed(X, Y), know edge(X, Y).
        reach(X, Y) :- vertex(X), vertex(Y), know edge(X, Y).
        reach(X, Z) :- reach(X, Y), vertex(Z), not know choice(Y, Z).
        "#,
    )
    .unwrap();
    match classify_recursive_epistemic_program(&program) {
        Err(XlogError::UnsupportedEpistemicConstruct { construct, context }) => {
            assert_eq!(construct, "recursive epistemic program");
            assert!(
                context.contains("NEGATED modal") && context.contains("not invariant"),
                "expected negated-non-invariant diagnostic, got: {context}"
            );
        }
        other => panic!("expected typed negated-modal rejection, got {other:?}"),
    }
}

#[test]
fn recursion_with_positive_non_invariant_modal_in_unrelated_rule_is_accepted_case_b() {
    // v0.9.2 ITEM A: a FAEEL program with ordinary recursion (`reach`) AND a POSITIVE
    // `possible choice(X)` over the epistemic-defined (non-invariant) `choice` is an
    // admissible Case-B program. The positive modal is resolved to a positive ordinary
    // atom and the whole program runs as a founded least fixpoint -- `choice` is itself
    // founded (seed + `know edge`), so resolving `possible choice` -> `choice` and
    // iterating computes the founded model. It is NOT rejected.
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
    assert_eq!(
        classify_recursive_epistemic_program(&program).unwrap(),
        RecursiveEpistemicClass::CaseB,
        "positive `possible` over a non-invariant founded relation is admissible Case B under FAEEL"
    );
    let reduced = try_reduce_case_a_recursive_epistemic_program(&program)
        .unwrap()
        .expect("admissible Case-B program reduces");
    assert_eq!(
        reduced
            .rules
            .iter()
            .flat_map(|rule| rule.body.iter())
            .filter(|lit| matches!(lit, BodyLiteral::Epistemic(_)))
            .count(),
        0,
        "Case-B reduce removes all modal literals"
    );
}

#[test]
fn g91_possible_over_co_evolving_relation_fails_closed() {
    // v0.9.2 ITEM A SOUNDNESS FLOOR: under G91 a `possible` modal over a relation that
    // CO-EVOLVES with the recursion is the autoepistemic SELF-FULFILLING fixpoint, which
    // the monotone founded-least-fixpoint reduction cannot express. It must fail closed
    // rather than silently return the FAEEL founded answer under a G91 pragma. (FAEEL
    // `possible` over the same shape is admitted as Case B -- see
    // `recursion_with_positive_non_invariant_modal_in_unrelated_rule_is_accepted_case_b`.)
    let program = parse_program(
        r#"
        #pragma epistemic_mode = g91
        pred vertex(u32).
        pred linked(u32, u32).
        pred reach(u32, u32).
        vertex(1). vertex(2). vertex(3).
        reach(X, Y) :- linked(X, Y).
        reach(X, Z) :- reach(X, Y), linked(Y, Z).
        linked(X, Y) :- vertex(X), vertex(Y), possible reach(X, Y).
        "#,
    )
    .unwrap();
    match classify_recursive_epistemic_program(&program) {
        Err(XlogError::UnsupportedEpistemicConstruct { construct, context }) => {
            assert_eq!(construct, "recursive epistemic program");
            assert!(
                context.contains("G91") && context.contains("self-fulfilling"),
                "expected G91 possible-recursion wall diagnostic, got: {context}"
            );
        }
        other => panic!("expected typed G91 possible-recursion rejection, got {other:?}"),
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
fn bare_modal_self_support_stays_non_recursive_not_case_b() {
    // v0.9.2 ITEM A regression: a bare modal self-support (`p() :- know p()` /
    // `p() :- possible p()`) has NO ordinary recursion edge -- the only self-dependency
    // is through the modal literal, which contributes no recursion edge. It must stay
    // `NonRecursive` (handled by item B's single-pass founded-extension path: rows:0
    // FAEEL / rows:1 G91) and NEVER be rerouted into Case-B by the relaxation.
    for source in ["p() :- know p().", "p() :- possible p()."] {
        let program = parse_program(source).unwrap();
        assert_eq!(
            classify_recursive_epistemic_program(&program).unwrap(),
            RecursiveEpistemicClass::NonRecursive,
            "bare modal self-support `{source}` must stay NonRecursive, not Case-B"
        );
        // try_reduce returns None (no ordinary recursion to route to the engine).
        assert!(
            try_reduce_case_a_recursive_epistemic_program(&program)
                .unwrap()
                .is_none(),
            "bare modal self-support `{source}` must not produce a Case-A/B reduction"
        );
    }
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
fn cross_arity_modal_coupling_over_invariant_relations_resolves_soundly() {
    // v0.9.2 sound consequence of the invariant-resolve: a head variable bound ONLY by
    // positive modal literals (`X` appears only in `know p(X)` / `possible link(X,Y)`)
    // is now SAFELY range-restricted when those modals range over INVARIANT relations,
    // because for an invariant relation `R` the modal `know R`/`possible R` ranges
    // exactly over `R`'s extension and resolves to a positive ordinary join atom. So
    // `a(X,Y) :- know p(X), possible link(X,Y)` reduces to `a(X,Y) :- p(X), link(X,Y)`,
    // which is safe. (A non-invariant / still-modal target is NOT resolved -- its
    // unbound variable correctly fails closed; see the companion assertion below.)
    let program = parse_program(
        r#"
        pred a(u32, u32).
        pred p(u32).
        pred link(u32, u32).
        a(X, Y) :- know p(X), possible link(X, Y).
        "#,
    )
    .unwrap();

    // The reduced program resolves BOTH positive-invariant modals into positive atoms
    // that range-restrict X (and Y), so no unsafe head variable remains.
    let reduced = reduce_epistemic_program_to_ordinary(&program);
    let a_rule = reduced
        .rules
        .iter()
        .find(|rule| rule.head.predicate == "a" && !rule.body.is_empty())
        .expect("reduced program retains the a rule");
    let positive_preds: std::collections::BTreeSet<&str> = a_rule
        .body
        .iter()
        .filter_map(|lit| match lit {
            BodyLiteral::Positive(atom) => Some(atom.predicate.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        positive_preds.contains("p") && positive_preds.contains("link"),
        "both invariant modals must resolve into positive join atoms, got body: {:?}",
        a_rule.body
    );

    // The split layer accepts the coupling, and joint compilation now SUCCEEDS (the
    // formerly-unsafe X is range-restricted by the resolved invariant joins).
    split_epistemic_program(&program).expect("split layer accepts the coupling");
    let exec = compile_epistemic_gpu_split_execution(&program)
        .expect("cross-arity coupling over invariant relations must resolve and compile");
    assert_eq!(exec.components.len(), 1);

    // COMPANION fail-closed gate: if the shared modal-only variable's binder ranges
    // over a NON-invariant (epistemic-derived) relation, it is NOT resolved, so the
    // variable stays unbound and the program fails closed.
    let unsound = parse_program(
        r#"
        pred a(u32).
        pred q(u32).
        pred r(u32).
        r(X) :- know q(X).
        a(X) :- possible r(X).
        "#,
    )
    .unwrap();
    // `r` is epistemic-derived (defined by `know q`), so it is NOT invariant; the modal
    // `possible r(X)` does not resolve and `a` would be unsafe -- BUT this is a chained
    // modal-over-determined-head shape, which the STRATIFIED path owns (Task 2). The
    // split path here is exercised only to confirm the unsound *unstratified* compile
    // does not silently accept: a non-invariant modal target never becomes a positive
    // binding atom in the reduced program.
    let unsound_reduced = reduce_epistemic_program_to_ordinary(&unsound);
    let a_unsound_rule = unsound_reduced
        .rules
        .iter()
        .find(|rule| rule.head.predicate == "a" && !rule.body.is_empty())
        .expect("reduced program retains the unsound a rule");
    assert!(
        !a_unsound_rule.body.iter().any(|lit| matches!(
            lit,
            BodyLiteral::Positive(atom) if atom.predicate == "r"
        )),
        "a modal over a NON-invariant (epistemic-derived) relation must NOT resolve into \
         a positive binding atom, got body: {:?}",
        a_unsound_rule.body
    );
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

// --- v0.9.2 Bundle 3: cross-component epistemic coupling ---

#[test]
fn modal_over_epistemic_derived_head_coalesces_then_fails_closed_at_joint_compile() {
    // CROSS-COMPONENT modal coupling: component B's MODAL literal (`know a()`)
    // references predicate `a`, which is component A's epistemic-DERIVED head
    // (`a() :- know p()`). Local analysis would split `a` and `b` into two
    // independent components, but the modal truth of `a` depends on A's accepted
    // world view, so an INDEPENDENT split is unsound. The dependency graph correctly
    // coalesces the two rules into ONE component (via the derived dependency on
    // `a`), but that single component carries TWO epistemic output heads (`a` and
    // `b`). The single-output-buffer JOINT split path cannot materialize two coupled
    // epistemic outputs, so the SPLIT LAYER fails closed with a coupling-specific
    // diagnostic naming the coupled predicates AND the merge reason -- NOT the
    // misleading "independent epistemic outputs" message.
    //
    // NOTE (v0.9.2 ITEM F): the full `xlog run` path does NOT route this shape
    // through the split layer. Because `a`'s modal (`know p`) ranges over the
    // INVARIANT base `p`, `a` is epistemically DETERMINED, so STRATIFIED execution
    // intercepts the coupling FIRST: it materializes the gated `a` as a lower
    // stratum, then gates `know a` in the higher stratum against the materialized
    // base. The PRODUCTION result is therefore SOLVED, not rejected -- see the e2e
    // equivalence test `test_derived_head_coupling_stratified_equals_per_stratum_reference`
    // in `xlog-gpu/tests/logic_runner.rs`. This test pins the SPLIT-LAYER fail-closed
    // contract that the stratified path relies on as its sound fallback for any shape
    // stratification declines (e.g. a genuinely-cyclic coupling, which stratification
    // returns `None` for and the split layer then correctly rejects).
    let program = parse_program("a() :- know p(). b() :- know a().").unwrap();

    // Coalescing is correct: one component, derived dependency on `a`.
    let split = split_epistemic_program(&program).unwrap();
    assert_eq!(split.components.len(), 1);
    assert_eq!(
        split.components[0].predicates,
        vec!["a".to_string(), "b".to_string(), "p".to_string()]
    );
    assert!(split.components[0].merge_reasons.contains(
        &EpistemicComponentMergeReason::DerivedPredicate {
            predicate: "a".to_string(),
        }
    ));

    // The coupled component must fail closed at joint compile with a precise,
    // coupling-named diagnostic.
    let err = compile_epistemic_gpu_split_execution(&program)
        .expect_err("coupled multi-epistemic-head component must fail closed");
    match err {
        XlogError::UnsupportedEpistemicConstruct { construct, context } => {
            assert_eq!(construct, "cross-component epistemic coupling");
            // Names the coupled epistemic output heads.
            assert!(
                context.contains("a") && context.contains("b"),
                "diagnostic must name coupled epistemic heads a and b, got: {context}"
            );
            // Names the merge reason (the derived dependency on `a`).
            assert!(
                context.contains("DerivedPredicate") || context.contains("derived"),
                "diagnostic must name the merge reason, got: {context}"
            );
            // Must NOT use the misleading "independent" wording.
            assert!(
                !context.contains("independent epistemic outputs"),
                "coupled components must not be reported as independent: {context}"
            );
        }
        other => panic!("expected cross-component coupling diagnostic, got {other:?}"),
    }
}

#[test]
fn shared_modal_two_head_coupling_is_joint_solved_multi_output() {
    // K1/K2/K3: two rules that share a BASE modal predicate (`q/1`) coalesce into
    // one component carrying two epistemic output heads (`a`, `b`). Because `q` is
    // a base/invariant relation (NOT an epistemic-derived head of the component),
    // the accepted world-view set over `q` is determined independently of which
    // head is being materialized. The component is therefore JOINT-SOLVED: ONE
    // candidate enumeration + world-view validation over the COMBINED modal
    // literals (`know q`, `possible q`), then EACH head relation is materialized
    // against the SAME accepted world view. This is the canonical SharedModalPredicate
    // joint-solving target -- it must COMPILE through the split path, not fail closed.
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
    assert!(split.components[0].merge_reasons.contains(
        &EpistemicComponentMergeReason::SharedModalPredicate {
            predicate: "q/1".to_string(),
        }
    ));

    // Joint-solvable: compiles through the split path as ONE coalesced multi-head
    // component whose joint plan carries both output heads.
    let exec = compile_epistemic_gpu_split_execution(&program)
        .expect("shared-modal two-head coupling over a base predicate must joint-solve");
    assert_eq!(
        exec.components.len(),
        1,
        "coupled heads must share ONE jointly-enumerated component"
    );
    // The single joint component's plan carries BOTH epistemic output heads.
    let heads: std::collections::BTreeSet<&str> = exec.components[0]
        .executable
        .gpu_plan
        .reductions
        .iter()
        .map(|reduction| reduction.head_predicate.as_str())
        .collect();
    assert_eq!(
        heads,
        ["a", "b"].into_iter().collect(),
        "joint component must materialize both coupled heads, got {heads:?}"
    );
    // K3: recomposition covers each source rule exactly once.
    assert_eq!(exec.recomposed_rule_indices(), vec![0, 1]);
}

#[test]
fn augmented_multi_head_shared_modal_coupling_joint_solves_with_per_head_projection() {
    // K4 (v0.9.2 SCOPE-LIMIT CLOSED): a coalesced multi-head component whose
    // epistemic rules need OUTPUT PROJECTION (a modal-literal variable not in the
    // head, e.g. `know edge(X, Y)` with head `a(X)`) is now JOINT-SOLVED via per-head
    // augmented projection. Each head is materialized from ITS OWN reduced relation
    // buffer with ITS OWN reduction row-filter, projecting the first
    // `public_head_arity` columns -- so the augmented modal-literal columns appended
    // after the public head terms are dropped per head, and coupled heads (even of
    // differing arity) each materialize their own public tuple shape. This reads only
    // the store/world-view boundary, never a resolved body. It must COMPILE through
    // the split path, not fail closed.
    let program = parse_program(
        r#"
        pred node(u32).
        pred color(u32).
        pred edge(u32, u32).
        pred a(u32).
        pred b(u32).
        a(X) :- node(X), know edge(X, Y).
        b(X) :- color(X), possible edge(X, Y).
        "#,
    )
    .unwrap();

    // Coalesces via the shared modal predicate `edge/2` into one multi-head component.
    let split = split_epistemic_program(&program).unwrap();
    assert_eq!(split.components.len(), 1);
    assert!(split.components[0].merge_reasons.contains(
        &EpistemicComponentMergeReason::SharedModalPredicate {
            predicate: "edge/2".to_string(),
        }
    ));

    let exec = compile_epistemic_gpu_split_execution(&program)
        .expect("augmented multi-head coupling must joint-solve via per-head projection");
    assert_eq!(
        exec.components.len(),
        1,
        "coupled augmented heads must share ONE jointly-enumerated component"
    );
    let plan = &exec.components[0].executable.gpu_plan;
    // Augmentation is present: the joint plan carries a public output projection.
    assert!(
        plan.final_output_columns.is_some(),
        "augmented multi-head plan must record a public output projection"
    );
    // Each coupled head records its OWN public arity (here both `a` and `b` are
    // unary; the differing-arity proof lives in the runtime device test
    // `augmented_multi_head_per_head_projection_materializes_differing_arity_on_device`).
    let arity_by_head: std::collections::BTreeMap<&str, usize> = plan
        .reductions
        .iter()
        .map(|reduction| {
            (
                reduction.head_predicate.as_str(),
                reduction.public_head_arity,
            )
        })
        .collect();
    assert_eq!(arity_by_head.get("a"), Some(&1));
    assert_eq!(arity_by_head.get("b"), Some(&1));
    // K3: recomposition covers each source rule exactly once.
    assert_eq!(exec.recomposed_rule_indices(), vec![0, 1]);
}

#[test]
fn modal_over_transitively_epistemic_derived_predicate_fails_closed_at_split_layer() {
    // K4: a modal literal ranging over an ORDINARY-derived predicate `r` that
    // TRANSITIVELY depends on an epistemic-derived head (`r :- a`, `a :- know p`)
    // is a nested/stratified epistemic dependency. `know r` cannot be evaluated by
    // one shared accepted world view because `r`'s extension depends on the world
    // view chosen for `a`. The JOINT SPLIT path (single shared world-view enumeration)
    // would mis-evaluate it, so the SPLIT LAYER must FAIL CLOSED -- not admit it as if
    // `r` were a base predicate. This guards the transitive case the direct-head check
    // misses.
    //
    // NOTE: the full `xlog run` path does NOT route this shape through the split layer:
    // v0.9.2 STRATIFIED execution intercepts it first (the transitive determined-closure
    // marks the ordinary `r` determined, materializes gated `a` as a lower stratum,
    // computes `r :- a` over the materialized base, and gates `know r` against the base
    // `r`). See example 24 (`24-transitive-determined-modal-stratified.xlog`) and the
    // CLI test `test_xlog_run_transitive_determined_modal_stratifies_accepted`. This test
    // pins the SPLIT-LAYER fail-closed contract that the stratified path relies on as its
    // sound fallback for any shape stratification declines.
    let program = parse_program(
        r#"
        pred node(u32).
        pred p(u32).
        pred a(u32).
        pred r(u32).
        pred b(u32).
        node(1). node(2). p(1).
        a(X) :- node(X), know p(X).
        r(X) :- a(X).
        b(X) :- node(X), know r(X).
        "#,
    )
    .unwrap();

    // The three epistemic-bearing rules coalesce into ONE component (derived
    // dependency on `a`, then on `r`), carrying two epistemic output heads (`a`,
    // `b`); base facts stay in their own independent components.
    let split = split_epistemic_program(&program).unwrap();
    let coupled = split
        .components
        .iter()
        .find(|c| {
            c.predicates.contains(&"a".to_string()) && c.predicates.contains(&"b".to_string())
        })
        .expect("a and b must coalesce into one component");
    assert!(coupled
        .merge_reasons
        .contains(&EpistemicComponentMergeReason::DerivedPredicate {
            predicate: "r".to_string(),
        }));

    let err = compile_epistemic_gpu_split_execution(&program)
        .expect_err("modal over transitively epistemic-derived predicate must fail closed");
    match err {
        XlogError::UnsupportedEpistemicConstruct { construct, context } => {
            assert_eq!(construct, "cross-component epistemic coupling");
            // Names the transitively-tainted modal predicate `r`.
            assert!(
                context.contains("r/1"),
                "diagnostic must name the nested modal predicate r/1, got: {context}"
            );
            assert!(
                context.contains("nested") || context.contains("epistemic-derived"),
                "diagnostic must name the nested/stratified reason, got: {context}"
            );
        }
        other => panic!("expected nested-modal coupling diagnostic, got {other:?}"),
    }
}

#[test]
fn ordinary_consumer_of_epistemic_head_is_accepted_safe_coupling() {
    // SAFE coupling (the task's named EGB-05 case): component B's ORDINARY body
    // consumes component A's epistemic head (`b() :- a()` where `a() :- know p()`).
    // This is a real derived dependency that must coalesce, but it has exactly ONE
    // epistemic output head (`a`), so it is NOT cross-component modal coupling: the
    // joint single-output path materializes the one epistemic relation and the
    // ordinary consumer rides the reduced ordinary pipeline. It must be ACCEPTED.
    let program = parse_program("a() :- know p(). b() :- a().").unwrap();

    let split = split_epistemic_program(&program).unwrap();
    assert_eq!(split.components.len(), 1);
    assert_eq!(
        split.components[0].merge_reasons,
        vec![EpistemicComponentMergeReason::DerivedPredicate {
            predicate: "a".to_string(),
        }]
    );

    // Accepted: compiles through both the split executable path AND the monolithic
    // single-execution path (one epistemic output head exists for both).
    let exec = compile_epistemic_gpu_split_execution(&program)
        .expect("safe single-epistemic-head coupling must compile through split path");
    assert_eq!(exec.components.len(), 1);
    assert_eq!(exec.recomposed_rule_indices(), vec![0, 1]);
}

#[test]
fn shared_extensional_inputs_do_not_force_false_cross_component_coalesce() {
    // K5: two epistemic components that share ONLY an EDB input (`node`) must stay
    // independent -- the shared extensional input must not force a false coalesce.
    // Each component keeps exactly one epistemic output head, so both compile
    // through the split path as independent components.
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

    let split = split_epistemic_program(&program).unwrap();
    assert_eq!(
        split.components.len(),
        2,
        "shared EDB input must not coalesce the two epistemic components"
    );
    for component in &split.components {
        assert!(
            component.merge_reasons.is_empty(),
            "EDB-only-sharing component {:?} must record no coalesce reason",
            component.predicates
        );
    }

    // Both stay independent through the split executable path (no false coupling
    // failure): two components, each a single epistemic output head.
    let exec = compile_epistemic_gpu_split_execution(&program)
        .expect("EDB-only-sharing components must compile as two independent components");
    assert_eq!(exec.components.len(), 2);
}

fn compiled_rule_count(plan: &ExecutionPlan) -> usize {
    plan.rules_by_scc.iter().map(Vec::len).sum()
}

// --- v0.9.2 stratified epistemic execution: chained coupling over a
// DETERMINED epistemic-derived head ---

#[test]
fn chained_modal_over_determined_epistemic_head_plans_stratified() {
    // K1: `b :- know a` where `a :- know p` and `p` is an EDB/invariant relation.
    // `a` is epistemically DETERMINED (its modal gates only over invariant `p`),
    // so its gated extension IS its truth and can be materialized as a base
    // relation. `b`'s modal `know a` then gates against that materialized relation.
    // This must PLAN STRATIFIED (2 strata, `a` strictly below `b`), NOT fail closed.
    let program = parse_program(
        r#"
        pred p(u32).
        pred node(u32).
        pred a(u32).
        pred b(u32).
        p(1). p(3). node(1). node(2). node(3).
        a(X) :- node(X), know p(X).
        b(X) :- node(X), know a(X).
        ?- b(X).
        "#,
    )
    .unwrap();

    let plan = xlog_logic::epistemic::try_plan_stratified_epistemic_program(&program)
        .expect("stratified planning must not error")
        .expect("chained modal over a determined epistemic head must plan stratified");
    assert_eq!(
        plan.strata.len(),
        2,
        "expected exactly two strata (a below b)"
    );
    // Lower stratum materializes `a`; higher stratum materializes `b`.
    assert_eq!(plan.strata[0].head_predicates, vec!["a".to_string()]);
    assert_eq!(plan.strata[1].head_predicates, vec!["b".to_string()]);
    // The lower stratum's sub-program must NOT contain any rule that redefines a
    // lower-stratum head other than its own (it owns `a`).
    // The higher stratum's sub-program must NOT redefine `a` (it is materialized).
    let higher_redefines_a = plan.strata[1]
        .program
        .rules
        .iter()
        .any(|rule| !rule.body.is_empty() && rule.head.predicate == "a");
    assert!(
        !higher_redefines_a,
        "higher stratum must not redefine the lower-stratum head `a`"
    );
}

#[test]
fn modal_over_transitively_determined_ordinary_head_plans_stratified() {
    // v0.9.2 Task 2: `b :- know r` where `r :- a` (ordinary) and `a :- know p` (`p`
    // EDB). The transitive determined-closure marks the ORDINARY `r` determined (its
    // sole rule ranges only over the determined `a`), so `know r` stratifies: `a` is a
    // lower stratum, and `b`'s stratum sits ABOVE `a` (routed through `r`'s epistemic
    // support `a`). The ordinary `r :- a` is DEFERRED to `b`'s stratum (where `a` is a
    // materialized gated base relation), so it is computed once from the gated `a`.
    let program = parse_program(
        r#"
        pred node(u32).
        pred p(u32).
        pred a(u32).
        pred r(u32).
        pred b(u32).
        node(1). node(2). p(1).
        a(X) :- node(X), know p(X).
        r(X) :- a(X).
        b(X) :- node(X), know r(X).
        ?- b(X).
        "#,
    )
    .unwrap();

    let plan = xlog_logic::epistemic::try_plan_stratified_epistemic_program(&program)
        .expect("stratified planning must not error")
        .expect("modal over a transitively-determined ordinary head must plan stratified");
    assert_eq!(
        plan.strata.len(),
        2,
        "expected exactly two strata (a below b)"
    );
    assert_eq!(plan.strata[0].head_predicates, vec!["a".to_string()]);
    assert_eq!(plan.strata[1].head_predicates, vec!["b".to_string()]);

    // The LOWER stratum (a) must NOT contain the ordinary `r :- a` supporting rule:
    // computing `r` there would derive it from the UNGATED candidate `a` and leak the
    // wrong tuples into the store.
    let lower_defines_r = plan.strata[0]
        .program
        .rules
        .iter()
        .any(|rule| !rule.body.is_empty() && rule.head.predicate == "r");
    assert!(
        !lower_defines_r,
        "the determined-ordinary `r :- a` must be DEFERRED out of `a`'s stratum (no \
         double-materialization from ungated `a`)"
    );

    // The HIGHER stratum (b) DOES carry `r :- a` (computed over the materialized base
    // `a`), and must NOT redefine `a`.
    let higher_defines_r = plan.strata[1]
        .program
        .rules
        .iter()
        .any(|rule| !rule.body.is_empty() && rule.head.predicate == "r");
    let higher_redefines_a = plan.strata[1]
        .program
        .rules
        .iter()
        .any(|rule| !rule.body.is_empty() && rule.head.predicate == "a");
    assert!(
        higher_defines_r,
        "`b`'s stratum must compute `r :- a` over the materialized gated base `a`"
    );
    assert!(
        !higher_redefines_a,
        "higher stratum must not redefine the lower-stratum head `a`"
    );
}

#[test]
fn shared_base_modal_does_not_trigger_stratification() {
    // K4 non-regression: example-18 shape. Two heads share a BASE modal `q` (EDB).
    // `q` is NOT an epistemic-derived head, so NO modal ranges over a determined
    // epistemic head -> stratified planning returns None and the existing joint
    // path keeps ownership (-> known={1,2}, maybe={2}).
    let program = parse_program(
        r#"
        pred node(u32).
        pred color(u32).
        pred q(u32).
        pred known(u32).
        pred maybe(u32).
        node(1). node(2). node(3).
        color(2). color(3).
        q(1). q(2).
        known(X) :- node(X), know q(X).
        maybe(X) :- color(X), possible q(X).
        "#,
    )
    .unwrap();
    let plan = xlog_logic::epistemic::try_plan_stratified_epistemic_program(&program)
        .expect("must not error");
    assert!(
        plan.is_none(),
        "shared base modal must NOT trigger stratification (joint path owns it)"
    );
}

#[test]
fn circular_modality_does_not_plan_stratified() {
    // `p() :- possible p()` (self-support through a modal). `p` is NOT epistemically
    // determined (self-reference), so stratification must decline (None), handing the
    // case to the single-pass FAEEL/G91 path. Under FAEEL the unfounded head is then
    // excluded from the founded model (empty extension); under G91 it is accepted.
    let program = parse_program("p() :- possible p().").unwrap();
    let plan = xlog_logic::epistemic::try_plan_stratified_epistemic_program(&program)
        .expect("must not error");
    assert!(
        plan.is_none(),
        "circular modality must not stratify; the single-pass FAEEL/G91 path owns it"
    );
}
