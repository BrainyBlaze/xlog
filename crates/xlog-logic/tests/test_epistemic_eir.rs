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

/// Helper: parse a one-rule program, build its EIR, and return the (op, negated)
/// of the FIRST epistemic literal in rule 0's body. The EIR is the semantic
/// boundary, so checking the collapsed literal there proves the chain normalizes
/// to an ordinary single-level epistemic literal (no raw-RIR shortcut).
fn first_eir_epistemic(src: &str) -> (EirEpistemicOp, bool, String) {
    let program = parse_program(src).unwrap_or_else(|e| panic!("parse failed for {src:?}: {e:?}"));
    let eir = build_eir(&program).unwrap_or_else(|e| panic!("eir failed for {src:?}: {e:?}"));
    for lit in &eir.rules[0].body {
        if let EirBodyLiteral::Epistemic(e) = lit {
            return (e.op, e.negated, e.atom.predicate.clone());
        }
    }
    panic!("no epistemic literal in EIR for {src:?}");
}

/// A bare modal CHAIN (`know possible p`) is NOT rejected — it collapses via
/// the KD45/S5 equivalence to its innermost (atom-adjacent) operator and
/// normalizes into an ordinary single-level epistemic literal in the EIR.
/// `know possible edge` -> `possible edge`.
#[test]
fn nested_epistemic_chain_collapses_to_inner_operator_in_eir() {
    assert_eq!(
        first_eir_epistemic("bad(X) :- know possible edge(X)."),
        (EirEpistemicOp::Possible, false, "edge".to_string())
    );
    assert_eq!(
        first_eir_epistemic("bad(X) :- possible know edge(X)."),
        (EirEpistemicOp::Know, false, "edge".to_string())
    );
    assert_eq!(
        first_eir_epistemic("bad(X) :- know know edge(X)."),
        (EirEpistemicOp::Know, false, "edge".to_string())
    );
    assert_eq!(
        first_eir_epistemic("bad(X) :- possible possible edge(X)."),
        (EirEpistemicOp::Possible, false, "edge".to_string())
    );
    // 3-deep: innermost still wins.
    assert_eq!(
        first_eir_epistemic("bad(X) :- know possible know edge(X)."),
        (EirEpistemicOp::Know, false, "edge".to_string())
    );
}

/// A bare chain under a single LEADING negation distributes the negation and
/// collapses to a negated single-level epistemic literal. `not know possible
/// edge` -> `not possible edge`.
#[test]
fn negated_nested_epistemic_chain_collapses_in_eir() {
    assert_eq!(
        first_eir_epistemic("bad(X) :- not know possible edge(X)."),
        (EirEpistemicOp::Possible, true, "edge".to_string())
    );
    assert_eq!(
        first_eir_epistemic("bad(X) :- not possible know edge(X)."),
        (EirEpistemicOp::Know, true, "edge".to_string())
    );
}

/// Nested modal forms with interior or atom-adjacent negation dualize to the
/// equivalent single-level modal literal before EIR lowering.
#[test]
fn nested_epistemic_with_interior_or_atom_negation_dualizes_in_eir() {
    let cases = [
        (
            "bad(X) :- know not possible edge(X).",
            (EirEpistemicOp::Possible, true, "edge".to_string()),
        ),
        (
            "bad(X) :- possible not know edge(X).",
            (EirEpistemicOp::Know, true, "edge".to_string()),
        ),
        (
            "bad(X) :- know possible not edge(X).",
            (EirEpistemicOp::Know, true, "edge".to_string()),
        ),
        (
            "bad(X) :- not know not possible edge(X).",
            (EirEpistemicOp::Possible, false, "edge".to_string()),
        ),
    ];
    for (src, expected) in cases {
        assert_eq!(first_eir_epistemic(src), expected);
    }
}

/// Single-level epistemic forms (including single-operator negated forms) MUST
/// continue to parse — the broadened nested rule must not swallow them.
#[test]
fn single_level_epistemic_forms_still_parse() {
    let cases = [
        "good(X) :- node(X), know edge(X).",
        "good(X) :- node(X), possible edge(X).",
        "good(X) :- node(X), not know edge(X).",
        "good(X) :- node(X), not possible edge(X).",
        "good(X) :- a(X), know b(X), not possible c(X).",
    ];
    for src in cases {
        assert!(
            parse_program(src).is_ok(),
            "single-level epistemic form must still parse: {src:?}"
        );
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
