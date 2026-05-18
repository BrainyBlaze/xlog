use xlog_ir::{
    EirEpistemicMode, EirEpistemicOp, EpistemicGpuBufferKind, EpistemicGpuHotPathPhase,
    EpistemicWcojReductionStatus,
};
use xlog_logic::epistemic::plan_epistemic_gpu_execution;
use xlog_logic::parse_program;

#[test]
fn epistemic_gpu_plan_requires_buffers_phases_and_zero_cpu_fallbacks() {
    let program = parse_program(
        r#"
        accepted(X) :- node(X), know edge(X).
        "#,
    )
    .unwrap();

    let plan = plan_epistemic_gpu_execution(&program).unwrap();

    assert_eq!(plan.mode, EirEpistemicMode::Faeel);
    assert_eq!(
        plan.required_phases,
        vec![
            EpistemicGpuHotPathPhase::CandidateGeneration,
            EpistemicGpuHotPathPhase::Propagation,
            EpistemicGpuHotPathPhase::WorldViewValidation,
            EpistemicGpuHotPathPhase::ResultMaterialization,
        ]
    );
    assert_eq!(
        plan.required_buffers,
        vec![
            EpistemicGpuBufferKind::CandidateAssumptions,
            EpistemicGpuBufferKind::WorldViews,
            EpistemicGpuBufferKind::ModelMembership,
            EpistemicGpuBufferKind::RejectionReasons,
        ]
    );
    assert!(plan.cpu_fallbacks.is_zero());
    assert_eq!(plan.epistemic_literals.len(), 1);
    assert_eq!(plan.epistemic_literals[0].op, EirEpistemicOp::Know);
    assert_eq!(plan.reductions[0].rule_index, 0);
    assert_eq!(
        plan.reductions[0].wcoj_status,
        EpistemicWcojReductionStatus::NotWcojCandidate
    );
}

#[test]
fn epistemic_gpu_plan_marks_multi_relation_reductions_for_wcoj_planner() {
    let program = parse_program(
        r#"
        accepted(X, Y, Z) :-
            edge(X, Y),
            edge(Y, Z),
            edge(X, Z),
            possible choice().
        "#,
    )
    .unwrap();

    let plan = plan_epistemic_gpu_execution(&program).unwrap();

    assert_eq!(plan.epistemic_literals[0].op, EirEpistemicOp::Possible);
    assert_eq!(plan.reductions[0].relational_body_atoms, 3);
    assert_eq!(
        plan.reductions[0].wcoj_status,
        EpistemicWcojReductionStatus::RequiresPlannerEligibility
    );
}

#[test]
fn epistemic_gpu_plan_records_tuple_membership_bindings_for_each_literal() {
    let program = parse_program(
        r#"
        accepted(X) :- node(X), know edge(X).
        visible(Y) :- item(Y), possible label(Y).
        "#,
    )
    .unwrap();

    let plan = plan_epistemic_gpu_execution(&program).unwrap();

    assert_eq!(plan.tuple_membership_bindings.len(), 2);
    assert_eq!(plan.tuple_membership_bindings[0].literal_index, 0);
    assert_eq!(plan.tuple_membership_bindings[0].reduction_index, 0);
    assert_eq!(plan.tuple_membership_bindings[0].predicate, "edge");
    assert_eq!(plan.tuple_membership_bindings[0].arity, 1);
    assert_eq!(plan.tuple_membership_bindings[0].op, EirEpistemicOp::Know);
    assert_eq!(plan.tuple_membership_bindings[1].literal_index, 1);
    assert_eq!(plan.tuple_membership_bindings[1].reduction_index, 1);
    assert_eq!(plan.tuple_membership_bindings[1].predicate, "label");
    assert_eq!(plan.tuple_membership_bindings[1].arity, 1);
    assert_eq!(
        plan.tuple_membership_bindings[1].op,
        EirEpistemicOp::Possible
    );
}

#[test]
fn epistemic_gpu_plan_rejects_duplicate_tuple_membership_literal_bindings() {
    let program = parse_program(
        r#"
        accepted(X) :- node(X), know edge(X).
        visible(Y) :- item(Y), possible label(Y).
        "#,
    )
    .unwrap();

    let mut plan = plan_epistemic_gpu_execution(&program).unwrap();
    plan.tuple_membership_bindings[1] = plan.tuple_membership_bindings[0].clone();

    let err = plan
        .validate_tuple_membership_bindings()
        .expect_err("duplicate literal bindings must fail closed");

    match err {
        xlog_core::XlogError::UnsupportedEpistemicConstruct { construct, context } => {
            assert_eq!(construct, "epistemic GPU tuple membership binding");
            assert!(context.contains("duplicate literal_index 0"));
        }
        other => panic!("expected tuple-membership binding error, got {other:?}"),
    }
}

#[test]
fn non_epistemic_program_does_not_create_gpu_epistemic_plan() {
    let program = parse_program("edge(1, 2).").unwrap();
    let err = plan_epistemic_gpu_execution(&program).unwrap_err();

    match err {
        xlog_core::XlogError::UnsupportedEpistemicConstruct { construct, context } => {
            assert_eq!(construct, "epistemic GPU execution plan");
            assert!(context.contains("requires at least one epistemic literal"));
        }
        other => panic!("expected typed epistemic GPU plan error, got {other:?}"),
    }
}
