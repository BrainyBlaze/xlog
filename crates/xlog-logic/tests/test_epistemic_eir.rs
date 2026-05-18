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
