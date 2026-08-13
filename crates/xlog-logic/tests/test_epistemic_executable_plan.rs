use std::collections::BTreeMap;

use xlog_core::{RelId, ScalarType};
use xlog_ir::rir::MultiwayPlan;
use xlog_ir::{
    EirEpistemicOp, EirTerm, EpistemicSolverCapability, EpistemicSolverStatusKind, ExecutionPlan,
    RirNode,
};
use xlog_logic::epistemic::{
    classify_recursive_epistemic_program, compile_epistemic_gpu_execution,
    compile_epistemic_gpu_execution_with_stats_snapshot, plan_epistemic_gpu_execution,
    prepare_epistemic_program, reduce_epistemic_program_to_ordinary,
    try_prepare_g91_compatibility_reduction, try_reduce_case_a_recursive_epistemic_program,
    RecursiveEpistemicClass,
};
use xlog_logic::{parse_program, BodyLiteral, Compiler, Program};
use xlog_stats::{
    ColumnStats, JoinSelectivity, KeyHeatStats, PrefixDegreeStats, RelationStats, StatsSnapshot,
};

fn compile_modal_fixpoint(program: &Program) -> (Program, ExecutionPlan) {
    assert_eq!(
        classify_recursive_epistemic_program(program).unwrap(),
        RecursiveEpistemicClass::ModalCycle
    );
    assert!(
        plan_epistemic_gpu_execution(program).is_err(),
        "modal cycles must not reach the single-pass GPU planner"
    );
    let reduced = try_reduce_case_a_recursive_epistemic_program(program)
        .unwrap()
        .expect("modal cycle must reduce to an ordinary fixpoint");
    assert!(reduced
        .rules
        .iter()
        .flat_map(|rule| &rule.body)
        .all(|literal| !matches!(literal, BodyLiteral::Epistemic(_))));
    let plan = Compiler::new()
        .compile_program(&reduced)
        .expect("modal fixpoint reduction must compile");
    (reduced, plan)
}

#[test]
fn epistemic_executable_plan_lowers_reduced_program_through_runtime_plan() {
    let program = parse_program(
        r#"
        node(1).
        edge(1).
        accepted(X) :- node(X), know edge(X).
        "#,
    )
    .unwrap();
    let executable = compile_epistemic_gpu_execution(&program).unwrap();

    assert!(executable.gpu_plan.cpu_fallbacks.is_zero());
    assert_eq!(executable.gpu_plan.epistemic_literals.len(), 1);
    assert_eq!(compiled_rule_count(&executable.reduced_runtime_plan), 3);
}

#[test]
fn epistemic_gpu_plan_exports_solver_service_contract_for_all_modal_assumptions() {
    let program = parse_program(
        r#"
        #pragma epistemic_mode = g91
        pred seed(u32).
        pred known_gate(u32).
        pred possible_gate(u32).
        pred not_known_gate(u32).
        pred not_possible_gate(u32).
        pred accepted(u32).

        accepted(X) :-
            seed(X),
            know known_gate(X),
            possible possible_gate(X),
            not know not_known_gate(X),
            not possible not_possible_gate(X).
        "#,
    )
    .unwrap();

    let executable = compile_epistemic_gpu_execution(&program)
        .expect("parsed all-operator program should lower into GPU execution plan");
    let contract = &executable.gpu_plan.solver_contract;

    executable
        .gpu_plan
        .validate_solver_contract()
        .expect("parsed lowering must export a valid solver service contract");
    assert_eq!(
        contract.required_capabilities,
        vec![
            EpistemicSolverCapability::IncrementalSat,
            EpistemicSolverCapability::AssumptionLifecycle,
            EpistemicSolverCapability::LearnedClauseTransfer,
            EpistemicSolverCapability::WeightedMaxSat,
            EpistemicSolverCapability::PortfolioSatMaxSat,
        ]
    );
    assert_eq!(
        contract.required_statuses,
        vec![
            EpistemicSolverStatusKind::Sat,
            EpistemicSolverStatusKind::Unsat,
            EpistemicSolverStatusKind::Unknown,
            EpistemicSolverStatusKind::Timeout,
        ]
    );

    let bindings = &contract.assumption_bindings;
    assert_eq!(bindings.len(), 4);
    let expected = [
        ("known_gate", EirEpistemicOp::Know, false),
        ("possible_gate", EirEpistemicOp::Possible, false),
        ("not_known_gate", EirEpistemicOp::Know, true),
        ("not_possible_gate", EirEpistemicOp::Possible, true),
    ];
    for (index, (binding, (predicate, op, negated))) in bindings.iter().zip(expected).enumerate() {
        assert_eq!(binding.literal_index, index);
        assert_eq!(binding.reduction_index, 0);
        assert_eq!(binding.predicate, predicate);
        assert_eq!(binding.arity, 1);
        assert_eq!(binding.terms, vec![EirTerm::Variable("X".to_string())]);
        assert_eq!(binding.op, op);
        assert_eq!(binding.negated, negated);
    }
    assert!(executable.gpu_plan.cpu_fallbacks.is_zero());
}

#[test]
fn wcoj_eligible_epistemic_reduction_reaches_multiway_runtime_plan() {
    let program = parse_program(
        r#"
        edge(1, 2).
        edge(2, 3).
        edge(1, 3).
        choice().

        accepted(X, Y, Z) :-
            edge(X, Y),
            edge(Y, Z),
            edge(X, Z),
            possible choice().
        "#,
    )
    .unwrap();

    let executable = compile_epistemic_gpu_execution(&program).unwrap();

    assert!(executable
        .gpu_plan
        .reductions
        .iter()
        .any(|reduction| reduction.relational_body_atoms == 3));
    assert!(plan_contains_multiway_join(
        &executable.reduced_runtime_plan
    ));
}

#[test]
fn epistemic_kclique_reduction_reuses_planner_layout_and_helper_split_surface() {
    let program = parse_program(EPISTEMIC_K5_SRC).unwrap();
    let rel_ids = rel_ids_for_reduced_k5();
    let snapshot = k5_stats(&rel_ids, Some((3, 5.0)));

    let executable =
        compile_epistemic_gpu_execution_with_stats_snapshot(&program, Some(&snapshot)).unwrap();
    let kclique = find_kclique_multiway(&executable.reduced_runtime_plan)
        .expect("epistemic K5 reduction must reach production K-clique MultiWayJoin");
    let order = kclique
        .var_order
        .as_ref()
        .and_then(|order| order.kclique.as_ref())
        .expect("epistemic K5 reduction must reuse KCliqueVariableOrder");

    assert!(matches!(kclique.plan, Some(MultiwayPlan::WcojWithPlan(_))));
    assert_eq!(order.k, 5);
    assert!(
        !order.sorted_layout_requirements.edge_slots.is_empty(),
        "K-clique epistemic reduction must carry sorted-layout requirements"
    );
    assert_eq!(
        order.helper_split_specs.len(),
        1,
        "buried-skew epistemic reduction must reuse helper-splitting specs"
    );
}

#[test]
fn faeel_modal_fixpoint_excludes_unfounded_self_support_from_founded_base() {
    // `p() :- possible p()` is supported only by circular self-support. Under FAEEL it
    // is UNFOUNDED, so the founded model is EMPTY and the program EXECUTES to that
    // empty extension (it is NOT rejected). The accepted lowering drops the circular
    // self-support rule from the reduced ordinary base, so `p` has no founding rule and
    // is absent from the founded model. High-level production GPU/CLI coverage
    // asserts the exact `rows: 0` result.
    let program = parse_program(
        r#"
        pred p().
        p() :- possible p().
        "#,
    )
    .unwrap();

    let (recursive_reduced, _) = compile_modal_fixpoint(&program);

    // The circular self-support rule is excluded from the reduced founded base, so `p`
    // has no founding (non-fact) rule.
    let reduced = reduce_epistemic_program_to_ordinary(&program)
        .expect("FAEEL unfounded self-support fixture must reduce");
    assert_eq!(recursive_reduced.rules, reduced.rules);
    assert_eq!(
        reduced
            .rules
            .iter()
            .filter(|rule| rule.head.predicate == "p" && !rule.body.is_empty())
            .count(),
        0,
        "unfounded circular self-support must be dropped from the founded base: {:?}",
        reduced.rules
    );
}

/// Assert that a multi-relation modal cycle uses the recursive modal reduction in both
/// semantic modes. FAEEL computes the founded least fixpoint; G91 applies its
/// compatibility treatment while resolving the same cycle.
fn assert_multi_relation_modal_cycle_reduces_both_modes(body: &str) {
    for (mode, prefix) in [("FAEEL", ""), ("G91", "#pragma epistemic_mode = g91\n")] {
        let src = format!("{prefix}{body}");
        let program = parse_program(&src).unwrap();
        assert_eq!(
            classify_recursive_epistemic_program(&program).unwrap(),
            RecursiveEpistemicClass::ModalCycle,
            "{mode}"
        );
        assert!(plan_epistemic_gpu_execution(&program).is_err(), "{mode}");
        if mode == "G91" {
            let prepared = prepare_epistemic_program(&program)
                .unwrap_or_else(|error| panic!("{mode}: source must validate: {error}"));
            let compatibility = try_prepare_g91_compatibility_reduction(&prepared)
                .unwrap_or_else(|error| panic!("{mode}: compatibility must prepare: {error}"))
                .expect("G91 cycle must select tuple compatibility iteration");
            for reduced in [
                compatibility.upper_bound_program(),
                compatibility.refinement_program(),
            ] {
                assert!(reduced
                    .rules
                    .iter()
                    .flat_map(|rule| &rule.body)
                    .all(|literal| !matches!(literal, BodyLiteral::Epistemic(_))));
                Compiler::new()
                    .compile_program(reduced)
                    .unwrap_or_else(|error| {
                        panic!("{mode}: compatibility pass must compile: {error}")
                    });
            }
        } else {
            let reduced = try_reduce_case_a_recursive_epistemic_program(&program)
                .unwrap_or_else(|error| panic!("{mode}: modal cycle must be admitted: {error}"))
                .expect("modal cycle must reduce to an ordinary fixpoint");
            Compiler::new()
                .compile_program(&reduced)
                .unwrap_or_else(|error| panic!("{mode}: modal fixpoint must compile: {error}"));
        }
    }
}

#[test]
fn multi_relation_mutual_possible_cycle_reduces_in_both_modes() {
    assert_multi_relation_modal_cycle_reduces_both_modes(
        "pred p().\npred q().\np() :- possible q().\nq() :- possible p().",
    );
}

#[test]
fn longer_multi_relation_possible_cycle_reduces_in_both_modes() {
    assert_multi_relation_modal_cycle_reduces_both_modes(
        "pred p().\npred q().\npred r().\np() :- possible q().\nq() :- possible r().\nr() :- possible p().",
    );
}

#[test]
fn mixed_multi_relation_modal_cycle_reduces_in_both_modes() {
    assert_multi_relation_modal_cycle_reduces_both_modes(
        "pred p().\npred q().\np() :- know q().\nq() :- possible p().",
    );
}

#[test]
fn faeel_modal_fixpoint_allows_longer_possible_cycle_with_independent_support() {
    let program = parse_program(
        r#"
        pred seed().
        pred p().
        pred q().
        pred r().
        seed().
        p() :- seed().
        q() :- seed().
        r() :- seed().
        p() :- possible q().
        q() :- possible r().
        r() :- possible p().
        "#,
    )
    .unwrap();

    compile_modal_fixpoint(&program);
}

#[test]
fn epistemic_constraint_reaches_typed_gpu_boundary() {
    // Epistemic integrity constraints are no longer rejected at the GPU boundary.
    // They are lowered first-class as world-view constraints whose body literals
    // are `EpistemicGpuPlan::epistemic_literals` entries, and the constraint plan
    // records which literal forms the body conjunction. They must NOT be rewritten
    // into ordinary RIR constraints (no `__xlog_constraint_*` relation), and they
    // must not be silently erased.
    let mut program = parse_program(
        r#"
        accepted() :- know fact().
        :- possible blocked().
        "#,
    )
    .unwrap();
    program.constraints[0].authored_index = Some(3);
    program
        .prepare_authored_constraint_identity(4)
        .expect("prepare sparse authored modal constraint identity");

    let executable = compile_epistemic_gpu_execution(&program)
        .expect("epistemic constraints reach the typed GPU boundary as world-view constraints");
    let gpu_plan = &executable.gpu_plan;

    // The constraint's `possible blocked()` literal is preserved first-class.
    assert_eq!(gpu_plan.constraints.len(), 1);
    let constraint = &gpu_plan.constraints[0];
    assert_eq!(constraint.constraint_index, 3);
    assert_eq!(constraint.literal_indices.len(), 1);
    let body_literal = &gpu_plan.epistemic_literals[constraint.literal_indices[0]];
    assert_eq!(body_literal.atom.predicate, "blocked");
    assert_eq!(body_literal.op, xlog_ir::EirEpistemicOp::Possible);
    assert!(!body_literal.negated);
    gpu_plan
        .validate_constraints()
        .expect("lowered world-view constraint references in-range literals");

    // No ordinary-RIR constraint rewrite: the reduced runtime plan must not
    // contain a compiler-generated `__xlog_constraint_*` relation for the
    // epistemic constraint.
    assert!(
        !executable
            .relation_ids
            .keys()
            .any(|name| name.starts_with("__xlog_constraint")),
        "epistemic constraint must not be rewritten into an ordinary RIR constraint relation: {:?}",
        executable.relation_ids.keys().collect::<Vec<_>>()
    );
    assert!(gpu_plan.cpu_fallbacks.is_zero());
}

#[test]
fn reduced_ordinary_constraint_uses_authored_relation_index() {
    let mut program = parse_program(
        r#"
        pred p(u32).
        pred q(u32).
        pred accepted(u32).
        p(1).
        accepted(X) :- p(X), know p(X).
        :- q(X), know p(X).
        :- p(X).
        "#,
    )
    .unwrap();
    program
        .prepare_authored_constraint_identity(2)
        .expect("assign authored constraint identities");

    let reduced = reduce_epistemic_program_to_ordinary(&program).unwrap();
    assert_eq!(reduced.constraints.len(), 1);
    assert_eq!(reduced.constraints[0].authored_index, Some(1));

    let mut compiler = Compiler::new();
    let plan = compiler.compile_program(&reduced).unwrap();
    let heads = plan
        .rules_by_scc
        .iter()
        .flatten()
        .map(|rule| rule.head.as_str())
        .collect::<Vec<_>>();
    assert!(heads.contains(&"__xlog_constraint_1"), "{:?}", heads);
    assert!(!heads.contains(&"__xlog_constraint_0"));
}

#[test]
fn public_epistemic_reduction_prepares_authored_constraint_identity_before_filtering() {
    let program = parse_program(
        r#"
        pred p(u32). pred q(u32). pred out(u32).
        p(1).
        out(X) :- p(X), know p(X).
        :- q(X), know p(X).
        :- p(X).
        "#,
    )
    .expect("parse unprepared authored program");

    let reduced = reduce_epistemic_program_to_ordinary(&program)
        .expect("public reduction prepares the full authored program");
    assert_eq!(reduced.authored_constraint_source_bound, Some(2));
    assert_eq!(reduced.constraints.len(), 1);
    assert_eq!(reduced.constraints[0].authored_index, Some(1));

    let mut compiler = Compiler::new();
    let plan = compiler
        .compile_prepared_program(&reduced)
        .expect("reduced program compiles without re-enumeration");
    let heads = plan
        .rules_by_scc
        .iter()
        .flatten()
        .map(|rule| rule.head.as_str())
        .collect::<Vec<_>>();
    assert!(heads.contains(&"__xlog_constraint_1"), "{heads:?}");
    assert!(!heads.contains(&"__xlog_constraint_0"));
}

#[test]
fn epistemic_gpu_execution_resolves_modal_only_bound_output_over_invariant_relation() {
    // Modal-only-bound output over an invariant relation is accepted: a single
    // epistemic head whose ONLY binder of an output variable is a POSITIVE modal
    // over an INVARIANT relation is no longer fail-closed. For an invariant
    // relation `q`, `possible q(X)` ranges exactly over `q`'s extension
    // (`possible q == know q == q`), so `p(X) :- possible q(X)` is the
    // well-defined `p = q`. The reduction resolves the positive-invariant modal
    // into a positive ordinary atom that range-restricts `X`, and the GPU
    // tuple-membership filter re-gates against the accepted world view. (Contrast
    // the self-supported `p() :- possible p()` case, where the target is NOT
    // invariant, so the resolve never fires and the FAEEL foundedness guard keeps
    // it fail-closed.)
    let program = parse_program(
        r#"
        pred p(u32).
        pred q(u32).
        q(1). q(2). q(3).
        p(X) :- possible q(X).
        ?- p(X).
        "#,
    )
    .unwrap();

    // The reduced ordinary program resolves `possible q(X)` into a POSITIVE `q(X)`
    // atom, so `X` is range-restricted (no longer an unsafe head variable).
    let reduced = reduce_epistemic_program_to_ordinary(&program)
        .expect("positive invariant-modal fixture must reduce");
    let p_rule = reduced
        .rules
        .iter()
        .find(|rule| rule.head.predicate == "p" && !rule.body.is_empty())
        .expect("reduced program retains the p rule");
    assert!(
        p_rule.body.iter().any(|lit| matches!(
            lit,
            xlog_logic::ast::BodyLiteral::Positive(atom) if atom.predicate == "q"
        )),
        "positive-invariant modal `possible q(X)` must resolve into a positive `q` join \
         atom that binds X, got body: {:?}",
        p_rule.body
    );
    assert!(
        !p_rule
            .body
            .iter()
            .any(|lit| matches!(lit, xlog_logic::ast::BodyLiteral::Epistemic(_))),
        "the resolved modal must not remain an epistemic literal"
    );

    // The full epistemic plan now compiles (no UnsafeVariable), with `p` carrying its
    // single tuple-membership gate over `q` and zero CPU fallbacks. Exact accepted
    // tuples (`p = q = {1,2,3}`) are asserted on device in
    // `single_head_modal_only_bound_over_invariant_materializes_q_extension_on_device`.
    let executable = compile_epistemic_gpu_execution(&program)
        .expect("modal-only-bound output over an invariant relation must now compile");
    assert_eq!(executable.gpu_plan.epistemic_literals.len(), 1);
    assert_eq!(executable.gpu_plan.tuple_membership_bindings.len(), 1);
    assert!(executable.gpu_plan.cpu_fallbacks.is_zero());
}

#[test]
fn faeel_modal_fixpoint_allows_self_possible_with_independent_founded_support() {
    let program = parse_program(
        r#"
        pred seed().
        pred p().
        seed().
        p() :- seed().
        p() :- possible p().
        "#,
    )
    .unwrap();

    let (reduced, _) = compile_modal_fixpoint(&program);
    assert!(reduced.rules.iter().any(|rule| {
        rule.head.predicate == "p"
            && rule
                .body
                .iter()
                .any(|literal| matches!(literal, BodyLiteral::Positive(atom) if atom.predicate == "seed"))
    }));
}

#[test]
fn faeel_modal_fixpoint_allows_nonzero_self_possible_with_tuple_founded_support() {
    let program = parse_program(
        r#"
        pred seed(u32).
        pred p(u32).
        seed(7).
        p(S) :- seed(S).
        p(X) :- seed(X), possible p(X).
        "#,
    )
    .unwrap();

    compile_modal_fixpoint(&program);
}

#[test]
fn faeel_modal_fixpoint_allows_ground_nonzero_self_possible_with_tuple_founded_support() {
    let program = parse_program(
        r#"
        pred seed(u32).
        pred p(u32).
        seed(7).
        p(S) :- seed(S).
        p(7) :- seed(7), possible p(7).
        "#,
    )
    .unwrap();

    compile_modal_fixpoint(&program);
}

#[test]
fn faeel_modal_fixpoint_allows_ground_possible_with_variable_headed_independent_support() {
    let program = parse_program(
        r#"
        pred seed(u32).
        pred p(u32).
        seed(7).
        p(X) :- seed(X).
        p(7) :- possible p(7).
        "#,
    )
    .unwrap();

    compile_modal_fixpoint(&program);
}

#[test]
fn faeel_modal_fixpoint_excludes_unfounded_tuples_keeping_founded_split() {
    // p(X) has an independent founding rule `p(X) :- seed(X)` (founds p(1) via seed)
    // AND a circular self-support rule `p(X) :- node(X), possible p(X)` over the wider
    // node domain {1,2}. p(2) is supported ONLY by self-support (no seed) and is
    // therefore UNFOUNDED: the founded model excludes it while keeping the founded p(1).
    // The reduced base drops the unfounded self-support rule and keeps the founded
    // `p :- seed` rule, so the founded model is exactly {p(1)}. High-level production
    // GPU/CLI coverage asserts the exact tuple {1}.
    let program = parse_program(
        r#"
        pred seed(u32).
        pred node(u32).
        pred p(u32).
        seed(1).
        node(1).
        node(2).
        p(X) :- seed(X).
        p(X) :- node(X), possible p(X).
        "#,
    )
    .unwrap();

    compile_modal_fixpoint(&program);

    let reduced = reduce_epistemic_program_to_ordinary(&program)
        .expect("partial-founded FAEEL fixture must reduce");
    // Founded `p :- seed` rule survives.
    assert!(
        reduced.rules.iter().any(|rule| rule.head.predicate == "p"
            && rule
                .body
                .iter()
                .any(|lit| matches!(lit, xlog_logic::ast::BodyLiteral::Positive(a) if a.predicate == "seed"))),
        "founded p:-seed rule must survive reduction: {:?}",
        reduced.rules
    );
    // Unfounded self-support rule (body has node(X)) is dropped.
    assert!(
        !reduced.rules.iter().any(|rule| rule.head.predicate == "p"
            && rule
                .body
                .iter()
                .any(|lit| matches!(lit, xlog_logic::ast::BodyLiteral::Positive(a) if a.predicate == "node"))),
        "unfounded p:-node,possible p rule must be dropped from the founded base: {:?}",
        reduced.rules
    );
}

#[test]
fn g91_modal_fixpoint_allows_self_supported_possible_compatibility_fixture() {
    let program = parse_program(
        r#"
        #pragma epistemic_mode = g91
        pred p().
        p() :- possible p().
        "#,
    )
    .unwrap();

    let prepared = prepare_epistemic_program(&program).expect("validate G91 source");
    let compatibility = try_prepare_g91_compatibility_reduction(&prepared)
        .expect("prepare G91 compatibility")
        .expect("G91 self-support must select tuple compatibility iteration");
    let plan = Compiler::new()
        .compile_program(compatibility.upper_bound_program())
        .expect("G91 upper bound must compile");
    assert_eq!(compiled_rule_count(&plan), 1);
}

const EPISTEMIC_K5_SRC: &str = r#"
    pred e01(u32, u32). pred e02(u32, u32). pred e03(u32, u32). pred e04(u32, u32).
    pred e12(u32, u32). pred e13(u32, u32). pred e14(u32, u32).
    pred e23(u32, u32). pred e24(u32, u32).
    pred e34(u32, u32).
    pred gate().
    pred clique5(u32, u32, u32, u32, u32).
    gate().
    clique5(V0, V1, V2, V3, V4) :-
        e01(V0, V1), e02(V0, V2), e03(V0, V3), e04(V0, V4),
        e12(V1, V2), e13(V1, V3), e14(V1, V4),
        e23(V2, V3), e24(V2, V4),
        e34(V3, V4),
        know gate().
"#;

const REDUCED_K5_SRC: &str = r#"
    pred e01(u32, u32). pred e02(u32, u32). pred e03(u32, u32). pred e04(u32, u32).
    pred e12(u32, u32). pred e13(u32, u32). pred e14(u32, u32).
    pred e23(u32, u32). pred e24(u32, u32).
    pred e34(u32, u32).
    pred gate().
    pred clique5(u32, u32, u32, u32, u32).
    gate().
    clique5(V0, V1, V2, V3, V4) :-
        e01(V0, V1), e02(V0, V2), e03(V0, V3), e04(V0, V4),
        e12(V1, V2), e13(V1, V3), e14(V1, V4),
        e23(V2, V3), e24(V2, V4),
        e34(V3, V4).
"#;

const K5_EDGES: [(&str, usize, usize); 10] = [
    ("e01", 0, 1),
    ("e02", 0, 2),
    ("e03", 0, 3),
    ("e04", 0, 4),
    ("e12", 1, 2),
    ("e13", 1, 3),
    ("e14", 1, 4),
    ("e23", 2, 3),
    ("e24", 2, 4),
    ("e34", 3, 4),
];

fn compiled_rule_count(plan: &ExecutionPlan) -> usize {
    plan.rules_by_scc.iter().map(Vec::len).sum()
}

fn plan_contains_multiway_join(plan: &ExecutionPlan) -> bool {
    plan.rules_by_scc
        .iter()
        .flatten()
        .any(|rule| node_contains_multiway_join(&rule.body))
}

fn node_contains_multiway_join(node: &RirNode) -> bool {
    match node {
        RirNode::MultiWayJoin { .. } => true,
        RirNode::Filter { input, .. }
        | RirNode::Project { input, .. }
        | RirNode::Distinct { input, .. }
        | RirNode::GroupBy { input, .. } => node_contains_multiway_join(input),
        RirNode::Join { left, right, .. } | RirNode::Diff { left, right } => {
            node_contains_multiway_join(left) || node_contains_multiway_join(right)
        }
        RirNode::Union { inputs } => inputs.iter().any(node_contains_multiway_join),
        RirNode::Fixpoint {
            base, recursive, ..
        } => node_contains_multiway_join(base) || node_contains_multiway_join(recursive),
        RirNode::ChainJoin { fallback, .. } => node_contains_multiway_join(fallback),
        RirNode::TensorMaskedJoin { .. } | RirNode::Scan { .. } | RirNode::Unit => false,
    }
}

struct KcliqueNode<'a> {
    plan: &'a Option<MultiwayPlan>,
    var_order: &'a Option<xlog_ir::rir::VariableOrder>,
}

fn find_kclique_multiway(plan: &ExecutionPlan) -> Option<KcliqueNode<'_>> {
    fn walk(node: &RirNode) -> Option<KcliqueNode<'_>> {
        match node {
            RirNode::MultiWayJoin {
                inputs,
                plan,
                var_order,
                ..
            } if inputs.len() == 10 => Some(KcliqueNode { plan, var_order }),
            RirNode::Filter { input, .. }
            | RirNode::Project { input, .. }
            | RirNode::Distinct { input, .. }
            | RirNode::GroupBy { input, .. } => walk(input),
            RirNode::Join { left, right, .. } | RirNode::Diff { left, right } => {
                walk(left).or_else(|| walk(right))
            }
            RirNode::Union { inputs } => inputs.iter().find_map(walk),
            RirNode::Fixpoint {
                base, recursive, ..
            } => walk(base).or_else(|| walk(recursive)),
            RirNode::ChainJoin { fallback, .. } => walk(fallback),
            RirNode::TensorMaskedJoin { .. } | RirNode::Scan { .. } | RirNode::Unit => None,
            RirNode::MultiWayJoin { fallback, .. } => walk(fallback),
        }
    }

    plan.rules_by_scc
        .iter()
        .flatten()
        .find_map(|rule| walk(&rule.body))
}

fn rel_ids_for_reduced_k5() -> BTreeMap<String, RelId> {
    let mut compiler = Compiler::new();
    let _ = compiler
        .compile(REDUCED_K5_SRC)
        .expect("compile reduced K5");
    compiler
        .rel_ids()
        .iter()
        .map(|(name, rel)| (name.clone(), *rel))
        .collect()
}

fn k5_stats(rel_ids: &BTreeMap<String, RelId>, hot: Option<(usize, f64)>) -> StatsSnapshot {
    let mut snapshot = StatsSnapshot::default();
    for (name, left, right) in K5_EDGES {
        let rel = *rel_ids.get(name).expect("edge rel id");
        snapshot.rel_names.push((rel, name.to_string()));
        let mut stats = RelationStats::new(rel);
        stats.update_cardinality(10_000);
        for (col_idx, variable) in [(0usize, left), (1usize, right)] {
            let mut col = ColumnStats::new(col_idx, ScalarType::U32);
            col.update_distinct(10_000);
            stats.add_column(col);
            stats.add_prefix_degree(PrefixDegreeStats::new(col_idx, 1.0, 1.25));
            let heat = match hot {
                Some((hot_var, hot_heat)) if hot_var == variable => hot_heat,
                _ => 0.25,
            };
            stats.add_key_heat(KeyHeatStats::new(col_idx, heat, heat));
        }
        snapshot.relations.push(stats);
    }

    for (left_idx, (left_name, left_i, left_j)) in K5_EDGES.iter().enumerate() {
        let left_rel = *rel_ids.get(*left_name).expect("left rel id");
        for (right_name, right_i, right_j) in K5_EDGES.iter().skip(left_idx + 1) {
            if left_i == right_i || left_i == right_j || left_j == right_i || left_j == right_j {
                let right_rel = *rel_ids.get(*right_name).expect("right rel id");
                let mut sel = JoinSelectivity::new(left_rel, right_rel);
                sel.set_keys(vec![0], vec![0]);
                sel.set_selectivity(0.001);
                snapshot.join_selectivities.push(sel);
            }
        }
    }

    snapshot
}
