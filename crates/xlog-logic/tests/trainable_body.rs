//! Unit coverage for the `trainable_rule` body-literal classifier used by the
//! neural joint-training template path (Stage-A gates vs Stage-B joins).
//!
//! Pure AST logic — runs on CPU, no CUDA required.

use std::collections::BTreeSet;

use xlog_logic::ast::{Atom, BodyLiteral, CompOp, Comparison, EpistemicLiteral, EpistemicOp, Term};
use xlog_logic::trainable_body::{classify_trainable_body_literal, TrainableBodyClass};

fn var(name: &str) -> Term {
    Term::Variable(name.to_string())
}

fn atom(pred: &str, terms: Vec<Term>) -> Atom {
    Atom {
        predicate: pred.to_string(),
        terms,
    }
}

fn bound(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|s| s.to_string()).collect()
}

fn not_neural(_: &str) -> bool {
    false
}

#[test]
fn comparison_is_builtin() {
    let lit = BodyLiteral::Comparison(Comparison {
        left: var("X"),
        op: CompOp::Lt,
        right: var("Y"),
    });
    assert_eq!(
        classify_trainable_body_literal(&lit, &bound(&[]), not_neural),
        TrainableBodyClass::Builtin
    );
}

#[test]
fn negated_is_negated() {
    let lit = BodyLiteral::Negated(atom("allowed", vec![var("Case")]));
    assert_eq!(
        classify_trainable_body_literal(&lit, &bound(&["Case"]), not_neural),
        TrainableBodyClass::Negated
    );
}

#[test]
fn epistemic_is_epistemic() {
    let lit = BodyLiteral::Epistemic(EpistemicLiteral {
        op: EpistemicOp::Know,
        negated: false,
        atom: atom("secret", vec![var("Case")]),
    });
    assert_eq!(
        classify_trainable_body_literal(&lit, &bound(&["Case"]), not_neural),
        TrainableBodyClass::Epistemic
    );
}

#[test]
fn neural_predicate_is_neural() {
    let lit = BodyLiteral::Positive(atom("neural_root", vec![var("Case"), var("Label")]));
    let is_neural = |p: &str| p == "neural_root";
    assert_eq!(
        classify_trainable_body_literal(&lit, &bound(&["Case"]), is_neural),
        TrainableBodyClass::Neural
    );
}

#[test]
fn fully_bound_relation_is_gate() {
    // root_case(Case) :- neural_root(Case, positive), allowed(Case).
    let lit = BodyLiteral::Positive(atom("allowed", vec![var("Case")]));
    assert_eq!(
        classify_trainable_body_literal(&lit, &bound(&["Case"]), not_neural),
        TrainableBodyClass::Gate
    );
}

#[test]
fn ground_relation_is_gate() {
    let lit = BodyLiteral::Positive(atom("allowed", vec![Term::Integer(5)]));
    assert_eq!(
        classify_trainable_body_literal(&lit, &bound(&[]), not_neural),
        TrainableBodyClass::Gate
    );
}

#[test]
fn anonymous_position_is_still_gate() {
    // allowed(Case, _) — the wildcard binds nothing downstream (existential).
    let lit = BodyLiteral::Positive(atom("allowed", vec![var("Case"), Term::Anonymous]));
    assert_eq!(
        classify_trainable_body_literal(&lit, &bound(&["Case"]), not_neural),
        TrainableBodyClass::Gate
    );
}

#[test]
fn new_named_variable_is_unbound_join() {
    // plastic(Edge, L) :- saliency(Event, L), pre_before_post(Event, Edge).
    // Event is bound (neural input); Edge is new → Stage-B join.
    let lit = BodyLiteral::Positive(atom("pre_before_post", vec![var("Event"), var("Edge")]));
    assert_eq!(
        classify_trainable_body_literal(&lit, &bound(&["Event"]), not_neural),
        TrainableBodyClass::UnboundJoin {
            var: "Edge".to_string()
        }
    );
}

#[test]
fn reports_first_unbound_variable() {
    let lit = BodyLiteral::Positive(atom("foo", vec![var("New1"), var("New2")]));
    assert_eq!(
        classify_trainable_body_literal(&lit, &bound(&[]), not_neural),
        TrainableBodyClass::UnboundJoin {
            var: "New1".to_string()
        }
    );
}
