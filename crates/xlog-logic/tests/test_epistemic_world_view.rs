use xlog_logic::ast::{BodyLiteral, Term};
use xlog_logic::epistemic::{EpistemicWorld, EpistemicWorldView, TruthValue};
use xlog_logic::parse_program;

#[test]
fn world_view_evaluates_know_possible_and_not_known() {
    let program = parse_program(
        r#"
        known() :- know stable().
        possible_choice() :- possible choice().
        not_known_choice() :- not know choice().
        "#,
    )
    .unwrap();
    let world_view = EpistemicWorldView::from_worlds(vec![
        EpistemicWorld::new()
            .with_fact("stable", 0)
            .with_fact("choice", 0),
        EpistemicWorld::new().with_fact("stable", 0),
    ])
    .unwrap();

    assert_eq!(
        world_view.evaluate(body_epistemic_literal(&program, 0)),
        TruthValue::True
    );
    assert_eq!(
        world_view.evaluate(body_epistemic_literal(&program, 1)),
        TruthValue::True
    );
    assert_eq!(
        world_view.evaluate(body_epistemic_literal(&program, 2)),
        TruthValue::True
    );
}

#[test]
fn world_view_requires_known_in_all_worlds_but_possible_in_any_world() {
    let program = parse_program(
        r#"
        known_choice() :- know choice().
        possible_choice() :- possible choice().
        "#,
    )
    .unwrap();
    let world_view = EpistemicWorldView::from_worlds(vec![
        EpistemicWorld::new().with_fact("choice", 0),
        EpistemicWorld::new(),
    ])
    .unwrap();

    assert_eq!(
        world_view.evaluate(body_epistemic_literal(&program, 0)),
        TruthValue::False
    );
    assert_eq!(
        world_view.evaluate(body_epistemic_literal(&program, 1)),
        TruthValue::True
    );
}

#[test]
fn world_view_distinguishes_absent_possible_from_not_known() {
    let program = parse_program(
        r#"
        impossible() :- possible rare().
        not_known_rare() :- not know rare().
        "#,
    )
    .unwrap();
    let world_view = EpistemicWorldView::from_worlds(vec![
        EpistemicWorld::new().with_fact("stable", 0),
        EpistemicWorld::new().with_fact("stable", 0),
    ])
    .unwrap();

    assert_eq!(
        world_view.evaluate(body_epistemic_literal(&program, 0)),
        TruthValue::False
    );
    assert_eq!(
        world_view.evaluate(body_epistemic_literal(&program, 1)),
        TruthValue::True
    );
}

#[test]
fn world_view_matches_nonzero_arity_facts_by_tuple_key() {
    let program = parse_program(
        r#"
        edge_one() :- know edge(1).
        edge_two() :- know edge(2).
        possible_two() :- possible edge(2).
        "#,
    )
    .unwrap();
    let world_view = EpistemicWorldView::from_worlds(vec![EpistemicWorld::new()
        .with_fact_terms("edge", vec![Term::Integer(1)])
        .unwrap()])
    .unwrap();

    assert_eq!(
        world_view.evaluate(body_epistemic_literal(&program, 0)),
        TruthValue::True
    );
    assert_eq!(
        world_view.evaluate(body_epistemic_literal(&program, 1)),
        TruthValue::False
    );
    assert_eq!(
        world_view.evaluate(body_epistemic_literal(&program, 2)),
        TruthValue::False
    );
}

#[test]
fn world_view_boundary_requires_at_least_one_model() {
    assert!(EpistemicWorldView::from_worlds(Vec::new()).is_err());
}

fn body_epistemic_literal(
    program: &xlog_logic::ast::Program,
    rule_index: usize,
) -> &xlog_logic::ast::EpistemicLiteral {
    let BodyLiteral::Epistemic(lit) = &program.rules[rule_index].body[0] else {
        panic!("expected epistemic literal in rule {rule_index}");
    };
    lit
}
