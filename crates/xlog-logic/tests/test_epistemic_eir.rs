use xlog_core::XlogError;
use xlog_ir::eir::{EirBodyLiteral, EirEpistemicMode, EirEpistemicOp, EirTerm};
use xlog_logic::{build_eir, parse_program, Compiler};

#[test]
fn epistemic_literal_is_explicit_in_eir() {
    let program = parse_program(
        r#"
        #pragma epistemic_mode = g91
        believed(X) :- node(X), know edge(X).
        "#,
    )
    .unwrap();

    let eir = build_eir(&program).unwrap();
    assert_eq!(eir.mode, EirEpistemicMode::G91);
    assert_eq!(eir.rules.len(), 1);

    let lit = match &eir.rules[0].body[1] {
        EirBodyLiteral::Epistemic(lit) => lit,
        other => panic!("expected explicit epistemic literal, got {other:?}"),
    };
    assert_eq!(lit.op, EirEpistemicOp::Know);
    assert!(!lit.negated);
    assert_eq!(lit.atom.predicate, "edge");
    assert_eq!(lit.atom.arity, 1);
}

#[test]
fn epistemic_possible_and_negated_forms_are_explicit_in_eir() {
    let program = parse_program(
        r#"
        reachable(X) :- node(X), possible path(X).
        uncertain(X) :- node(X), not know blocked(X).
        impossible(X) :- node(X), not possible repair(X).
        "#,
    )
    .unwrap();

    let eir = build_eir(&program).unwrap();
    assert_eq!(eir.rules.len(), 3);

    let possible = match &eir.rules[0].body[1] {
        EirBodyLiteral::Epistemic(lit) => lit,
        other => panic!("expected explicit possible literal, got {other:?}"),
    };
    assert_eq!(possible.op, EirEpistemicOp::Possible);
    assert!(!possible.negated);
    assert_eq!(possible.atom.predicate, "path");

    let not_know = match &eir.rules[1].body[1] {
        EirBodyLiteral::Epistemic(lit) => lit,
        other => panic!("expected explicit not-know literal, got {other:?}"),
    };
    assert_eq!(not_know.op, EirEpistemicOp::Know);
    assert!(not_know.negated);
    assert_eq!(not_know.atom.predicate, "blocked");

    let not_possible = match &eir.rules[2].body[1] {
        EirBodyLiteral::Epistemic(lit) => lit,
        other => panic!("expected explicit not-possible literal, got {other:?}"),
    };
    assert_eq!(not_possible.op, EirEpistemicOp::Possible);
    assert!(not_possible.negated);
    assert_eq!(not_possible.atom.predicate, "repair");
}

#[test]
fn epistemic_literal_preserves_tuple_terms_for_gpu_key_matching() {
    let program = parse_program(
        r#"
        believed(X) :- node(X), know edge(X, 42, label).
        "#,
    )
    .unwrap();

    let eir = build_eir(&program).unwrap();

    let lit = match &eir.rules[0].body[1] {
        EirBodyLiteral::Epistemic(lit) => lit,
        other => panic!("expected explicit epistemic literal, got {other:?}"),
    };
    assert_eq!(lit.atom.predicate, "edge");
    assert_eq!(lit.atom.arity, 3);
    assert_eq!(
        lit.atom.terms,
        vec![
            EirTerm::Variable("X".to_string()),
            EirTerm::Integer(42),
            EirTerm::Symbol(xlog_core::symbol::intern("label")),
        ]
    );
}

#[test]
fn nested_epistemic_literal_returns_typed_error() {
    let err = parse_program("bad(X) :- know possible edge(X).").unwrap_err();
    match err {
        XlogError::UnsupportedEpistemicConstruct { construct, .. } => {
            assert_eq!(construct, "nested epistemic literal");
        }
        other => panic!("expected typed epistemic diagnostic, got {other:?}"),
    }
}

#[test]
fn negated_nested_epistemic_literal_returns_typed_error() {
    let err = parse_program("bad(X) :- not know possible edge(X).").unwrap_err();
    match err {
        XlogError::UnsupportedEpistemicConstruct { construct, context } => {
            assert_eq!(construct, "nested epistemic literal");
            assert!(context.contains("not know possible edge(X)"));
        }
        other => panic!("expected typed negated epistemic diagnostic, got {other:?}"),
    }
}

#[test]
fn negated_possible_nested_epistemic_literal_returns_typed_error() {
    let err = parse_program("bad(X) :- not possible know edge(X).").unwrap_err();
    match err {
        XlogError::UnsupportedEpistemicConstruct { construct, context } => {
            assert_eq!(construct, "nested epistemic literal");
            assert!(context.contains("not possible know edge(X)"));
        }
        other => panic!("expected typed negated possible epistemic diagnostic, got {other:?}"),
    }
}

#[test]
fn rir_lowering_rejects_epistemic_literal_with_typed_error() {
    let err = Compiler::new()
        .compile("believed(X) :- node(X), know edge(X).")
        .unwrap_err();
    match err {
        XlogError::UnsupportedEpistemicConstruct { construct, .. } => {
            assert_eq!(construct, "RIR lowering boundary");
        }
        other => panic!("expected typed epistemic diagnostic, got {other:?}"),
    }
}
