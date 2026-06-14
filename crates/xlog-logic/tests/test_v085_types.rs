use xlog_core::ScalarType;
use xlog_logic::ast::{Term, TypeRef};
use xlog_logic::{parse_program, Compiler};

#[test]
fn parses_named_columns_domain_aliases_and_language_completeness_type_refs() {
    let src = r#"
        domain user_id: u32.
        pred edge(src: user_id, dst: u32).
        pred bag(owner: user_id, values: list<u32>, payload: term, shape: compound, mapper: predref).
    "#;

    let program = parse_program(src).expect("language-completeness type refs should parse");

    assert_eq!(program.domains[0].name, "user_id");
    assert_eq!(program.domains[0].typ, ScalarType::U32);

    let edge = &program.predicates[0];
    assert_eq!(edge.columns[0].name.as_deref(), Some("src"));
    assert_eq!(edge.columns[0].typ, TypeRef::Domain("user_id".to_string()));
    assert_eq!(edge.columns[1].name.as_deref(), Some("dst"));
    assert_eq!(edge.columns[1].typ, TypeRef::Scalar(ScalarType::U32));

    let bag = &program.predicates[1];
    assert_eq!(
        bag.columns
            .iter()
            .map(|c| c.name.as_deref())
            .collect::<Vec<_>>(),
        vec![
            Some("owner"),
            Some("values"),
            Some("payload"),
            Some("shape"),
            Some("mapper")
        ]
    );
    assert_eq!(
        bag.columns[1].typ,
        TypeRef::List(Box::new(TypeRef::Scalar(ScalarType::U32)))
    );
    assert_eq!(bag.columns[2].typ, TypeRef::Term);
    assert_eq!(bag.columns[3].typ, TypeRef::Compound);
    assert_eq!(bag.columns[4].typ, TypeRef::PredRef);
}

#[test]
fn parses_list_cons_and_compound_terms_into_ast() {
    let src = r#"
        value([1, pair(foo, 2), [3]]).
        cons([H | T]) :- tail(T).
    "#;

    let program = parse_program(src).expect("extended terms should parse");

    match &program.rules[0].head.terms[0] {
        Term::List(items) => {
            assert_eq!(items.len(), 3);
            assert_eq!(items[0], Term::Integer(1));
            assert!(matches!(items[1], Term::Compound { ref functor, .. } if functor == "pair"));
            assert!(matches!(items[2], Term::List(_)));
        }
        other => panic!("expected list term, got {other:?}"),
    }

    match &program.rules[1].head.terms[0] {
        Term::Cons { head, tail } => {
            assert_eq!(**head, Term::Variable("H".to_string()));
            assert_eq!(**tail, Term::Variable("T".to_string()));
        }
        other => panic!("expected cons term, got {other:?}"),
    }
}

#[test]
fn named_scalar_and_domain_columns_lower_to_schema_labels() {
    let src = r#"
        domain user_id: u32.
        pred edge(src: user_id, dst: u32).
        edge(1, 2).
        ?- edge(Src, Dst).
    "#;

    let mut compiler = Compiler::new();
    compiler
        .compile(src)
        .expect("scalar domain aliases should lower");
    let schema = compiler.schemas().get("edge").expect("edge schema");

    assert_eq!(schema.column_index("src"), Some(0));
    assert_eq!(schema.column_index("dst"), Some(1));
    assert_eq!(schema.column_type(0), Some(ScalarType::U32));
    assert_eq!(schema.column_type(1), Some(ScalarType::U32));
}

#[test]
fn rejects_remaining_non_lowered_extended_term_forms_with_typed_errors() {
    let invalid = [
        (
            "unknown alias",
            "pred bad(x: unknown_domain).",
            "unknown domain alias",
        ),
        ("cons fact", "bad([H | T]).", "cons"),
        ("compound fact", "bad(pair(1, 2)).", "compound"),
        (
            "compound comparison",
            "bad(X) :- raw(X), X = pair(1, 2).",
            "compound",
        ),
    ];

    for (label, src, needle) in invalid {
        let err = Compiler::new().compile(src).expect_err(label);
        let msg = err.to_string();
        assert!(
            msg.contains(needle),
            "{label}: expected typed extended-term error containing {needle:?}, got {msg}"
        );
    }
}

#[test]
fn scalar_only_programs_remain_compatible() {
    let src = r#"
        pred edge(u32, u32).
        edge(1, 2).
        edge(2, 3).
        reach(X, Y) :- edge(X, Y).
        reach(X, Z) :- reach(X, Y), edge(Y, Z).
        ?- reach(1, N).
    "#;

    let mut compiler = Compiler::new();
    compiler
        .compile(src)
        .expect("existing scalar program should compile unchanged");
    let schema = compiler.schemas().get("edge").expect("edge schema");

    assert_eq!(schema.column_index("c0"), Some(0));
    assert_eq!(schema.column_index("c1"), Some(1));
    assert_eq!(schema.column_type(0), Some(ScalarType::U32));
    assert_eq!(schema.column_type(1), Some(ScalarType::U32));
}
