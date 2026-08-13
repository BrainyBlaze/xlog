use xlog_logic::{parse_program, Compiler};

fn two_constraint_program() -> xlog_logic::Program {
    parse_program(
        r#"
        :- p(0).
        :- p(1).
        "#,
    )
    .expect("parse constraints")
}

#[test]
fn authored_constraint_identity_all_unassigned_is_assigned_densely() {
    let mut program = two_constraint_program();
    assert!(program
        .constraints
        .iter()
        .all(|constraint| constraint.authored_index.is_none()));

    program
        .prepare_authored_constraint_identity_at_root()
        .expect("assign authored constraint identities");

    assert_eq!(program.constraints[0].authored_index, Some(0));
    assert_eq!(program.constraints[1].authored_index, Some(1));
    assert_eq!(program.authored_constraint_source_bound, Some(2));
}

#[test]
fn authored_constraint_identity_unassigned_subset_with_larger_bound_is_rejected() {
    let mut program = two_constraint_program();

    let error = program
        .prepare_authored_constraint_identity(4)
        .expect_err("an unassigned subset must not invent local authored identities");

    assert!(
        error
            .to_string()
            .contains("unassigned constraint count 2 does not match authored source bound 4"),
        "{error}"
    );
    assert!(program
        .constraints
        .iter()
        .all(|constraint| constraint.authored_index.is_none()));
}

#[test]
fn authored_constraint_identity_assigned_sparse_identities_are_preserved() {
    let mut program = two_constraint_program();
    program.constraints[0].authored_index = Some(1);
    program.constraints[1].authored_index = Some(3);

    program
        .prepare_authored_constraint_identity(4)
        .expect("preserve valid sparse identities");

    assert_eq!(program.constraints[0].authored_index, Some(1));
    assert_eq!(program.constraints[1].authored_index, Some(3));
    assert_eq!(program.authored_constraint_source_bound, Some(4));
}

#[test]
fn authored_constraint_identity_duplicate_identity_is_rejected() {
    let mut program = two_constraint_program();
    program.constraints[0].authored_index = Some(1);
    program.constraints[1].authored_index = Some(1);

    let error = program
        .prepare_authored_constraint_identity(2)
        .expect_err("duplicate authored identities must be rejected");

    assert!(error
        .to_string()
        .contains("duplicate authored constraint index 1"));
}

#[test]
fn authored_constraint_identity_out_of_source_bound_is_rejected() {
    let mut program = two_constraint_program();
    program.constraints[0].authored_index = Some(0);
    program.constraints[1].authored_index = Some(2);

    let error = program
        .prepare_authored_constraint_identity(2)
        .expect_err("out-of-source-bound authored identity must be rejected");

    assert!(error
        .to_string()
        .contains("authored constraint index 2 is outside source bound 2"));
}

#[test]
fn authored_constraint_identity_mixed_assignment_is_rejected() {
    let mut program = two_constraint_program();
    program.constraints[0].authored_index = Some(0);

    let error = program
        .prepare_authored_constraint_identity(2)
        .expect_err("mixed authored identity assignment must be rejected");

    assert!(error
        .to_string()
        .contains("mixed assigned and unassigned authored constraint identities"));
}

#[test]
fn authored_constraint_identity_prepared_subset_uses_carried_source_bound() {
    let mut program = parse_program(
        r#"
        p(0). p(1). p(2). p(3).
        :- p(0).
        :- p(1).
        :- p(2).
        :- p(3).
        "#,
    )
    .expect("parse authored program");
    program
        .prepare_authored_constraint_identity_at_root()
        .expect("prepare root authored identities");
    program
        .constraints
        .retain(|constraint| matches!(constraint.authored_index, Some(1 | 3)));

    program
        .validate_prepared_authored_constraint_identity()
        .expect("sparse subset remains valid against the carried source bound");
    Compiler::new()
        .compile_prepared_program(&program)
        .expect("prepared sparse subset compiles without local re-enumeration");
}

#[test]
fn authored_constraint_identity_prepared_compilation_rejects_unassigned_constraints() {
    let program = two_constraint_program();
    let error = Compiler::new()
        .compile_prepared_program(&program)
        .expect_err("prepared compilation must never invent authored identities");

    assert!(
        error
            .to_string()
            .contains("prepared constraint compilation requires authored identities"),
        "{error}"
    );
}
