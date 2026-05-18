use xlog_core::ScalarType;
use xlog_logic::ast::{BodyLiteral, Term, TypeRef};
use xlog_logic::{normalize_v085_lists, parse_program, Compiler};

fn fact_rows(program: &xlog_logic::Program, pred: &str) -> Vec<Vec<Term>> {
    program
        .facts()
        .filter(|rule| rule.head.predicate == pred)
        .map(|rule| rule.head.terms.clone())
        .collect()
}

fn body_predicates(program: &xlog_logic::Program) -> Vec<String> {
    program
        .proper_rules()
        .flat_map(|rule| {
            rule.body.iter().filter_map(|lit| match lit {
                BodyLiteral::Positive(atom) | BodyLiteral::Negated(atom) => {
                    Some(atom.predicate.clone())
                }
                BodyLiteral::Comparison(_) | BodyLiteral::IsExpr(_) => None,
            })
        })
        .collect()
}

#[test]
fn parses_25_list_syntax_fixtures() {
    let accepted = [
        "value([]).",
        "value([1]).",
        "value([1, 2, 3]).",
        "value([[1], [2, 3]]).",
        "value([alice, bob]).",
        "cons([H | T]) :- tail(T).",
        "pred path(symbol, list<symbol>).",
        "pred nested(list<list<u32>>).",
        "ok(X) :- member(X, [1, 2, 3]).",
        "ok(X) :- memberchk(X, [alice, bob]).",
        "ok(N) :- length([1, 2, 3], N).",
        "ok(X) :- nth(0, [7, 8], X).",
        "ok(L) :- append([1], [2], L).",
        "ok(L) :- sort([2, 1, 1], L).",
        "ok(L) :- msort([2, 1, 1], L).",
        "ok(L) :- list_to_set([2, 1, 2], L).",
        "?- member(X, [1, 2]).",
        "?- length([], N).",
        "?- is_list([1, 2]).",
        "bag([H | [T]]).",
    ];

    let rejected = [
        "value([1, 2).",
        "value([| T]).",
        "pred bad(list<>).",
        "pred bad(xs: list<).",
        "ok(X) :- member(X, [1, 2).",
    ];

    assert_eq!(accepted.len() + rejected.len(), 25);

    for src in accepted {
        parse_program(src).unwrap_or_else(|err| panic!("accepted fixture failed: {src}: {err}"));
    }

    for src in rejected {
        assert!(
            parse_program(src).is_err(),
            "rejected fixture unexpectedly parsed: {src}"
        );
    }
}

#[test]
fn compiles_list_typed_columns_literals_cons_and_core_builtins() {
    let src = r#"
        domain node: u32.
        pred bag(id: node, xs: list<u32>).
        bag(1, [10, 20, 10]).
        bag(2, [30]).

        has_member(Id, X) :- bag(Id, L), member(X, L).
        has_memberchk(Id, X) :- bag(Id, L), memberchk(X, L).
        has_len(Id, N) :- bag(Id, L), length(L, N).
        first(Id, X) :- bag(Id, L), nth(0, L, X).
        tail_of(Id, H, T) :- bag(Id, [H | T]).
        appended(L) :- append([10], [20], L).
        sorted(L) :- sort([20, 10, 10], L).
        msorted(L) :- msort([20, 10, 10], L).
        as_set(L) :- list_to_set([20, 10, 20], L).

        ?- has_member(Id, X).
    "#;

    let mut compiler = Compiler::new();
    compiler.compile(src).expect("finite lists should compile");

    let bag = compiler.schemas().get("bag").expect("bag schema");
    assert_eq!(bag.column_index("id"), Some(0));
    assert_eq!(bag.column_index("xs"), Some(1));
    assert_eq!(bag.column_type(0), Some(ScalarType::U32));
    assert_eq!(bag.column_type(1), Some(ScalarType::U64));

    assert!(compiler.schemas().contains_key("__xlog_list_len"));
    assert!(compiler.schemas().contains_key("__xlog_list_item_u32"));
    assert!(compiler.schemas().contains_key("__xlog_list_cons_u32"));
    assert!(compiler.schemas().contains_key("__xlog_list_append_u32"));
    assert!(compiler.schemas().contains_key("__xlog_list_sort_u32"));
    assert!(compiler.schemas().contains_key("__xlog_list_msort_u32"));
    assert!(compiler.schemas().contains_key("__xlog_list_to_set_u32"));
}

#[test]
fn normalizes_lists_to_helper_facts_and_removes_builtin_atoms() {
    let src = r#"
        pred bag(xs: list<u32>).
        bag([2, 1, 2]).
        member_out(X) :- bag(L), member(X, L).
        nth_out(X) :- nth(1, [2, 1, 2], X).
        len_out(N) :- length([2, 1, 2], N).
        append_out(L) :- append([2], [1, 2], L).
        sort_out(L) :- sort([2, 1, 2], L).
        msort_out(L) :- msort([2, 1, 2], L).
        set_out(L) :- list_to_set([2, 1, 2], L).
    "#;

    let program = parse_program(src).expect("parse");
    let normalized = normalize_v085_lists(&program).expect("normalize");

    let body_preds = body_predicates(&normalized);
    for builtin in [
        "member",
        "memberchk",
        "length",
        "nth",
        "append",
        "sort",
        "msort",
        "list_to_set",
        "is_list",
    ] {
        assert!(
            !body_preds.iter().any(|pred| pred == builtin),
            "builtin predicate remained after normalization: {builtin}"
        );
    }

    assert!(!fact_rows(&normalized, "__xlog_list_len").is_empty());
    let bag_list_id = match &fact_rows(&normalized, "bag")[0][0] {
        Term::Integer(id) => *id,
        other => panic!("expected normalized bag list id, got {other:?}"),
    };
    let bag_items: Vec<Vec<Term>> = fact_rows(&normalized, "__xlog_list_item_u32")
        .into_iter()
        .filter(|row| matches!(row.first(), Some(Term::Integer(id)) if *id == bag_list_id))
        .collect();
    assert_eq!(bag_items.len(), 3);
    assert_eq!(bag_items[0][2], Term::Integer(2));
    assert_eq!(bag_items[1][2], Term::Integer(1));
    assert_eq!(bag_items[2][2], Term::Integer(2));
    assert!(!fact_rows(&normalized, "__xlog_list_append_u32").is_empty());
    assert!(!fact_rows(&normalized, "__xlog_list_sort_u32").is_empty());
    assert!(!fact_rows(&normalized, "__xlog_list_msort_u32").is_empty());
    assert!(!fact_rows(&normalized, "__xlog_list_to_set_u32").is_empty());
}

#[test]
fn list_builtins_have_relational_oracle_fixtures() {
    let fixtures = [
        (
            "is_list literal",
            "ok() :- is_list([1, 2]).",
            "__xlog_list_len",
        ),
        (
            "member literal",
            "out(X) :- member(X, [1, 2]).",
            "__xlog_list_item_u32",
        ),
        (
            "memberchk literal",
            "out(X) :- memberchk(X, [1, 2]).",
            "__xlog_list_item_u32",
        ),
        (
            "length literal",
            "out(N) :- length([1, 2], N).",
            "__xlog_list_len",
        ),
        (
            "nth literal",
            "out(X) :- nth(1, [1, 2], X).",
            "__xlog_list_item_u32",
        ),
        (
            "append literal",
            "out(L) :- append([1], [2], L).",
            "__xlog_list_append_u32",
        ),
        (
            "sort literal",
            "out(L) :- sort([2, 1, 2], L).",
            "__xlog_list_sort_u32",
        ),
        (
            "msort literal",
            "out(L) :- msort([2, 1, 2], L).",
            "__xlog_list_msort_u32",
        ),
        (
            "list_to_set literal",
            "out(L) :- list_to_set([2, 1, 2], L).",
            "__xlog_list_to_set_u32",
        ),
        (
            "declared member",
            "pred bag(list<u32>). bag([1, 2]). out(X) :- bag(L), member(X, L).",
            "__xlog_list_item_u32",
        ),
        (
            "cons pattern",
            "pred bag(list<u32>). bag([1, 2]). out(H, T) :- bag([H | T]).",
            "__xlog_list_cons_u32",
        ),
        (
            "symbol list",
            "pred bag(list<symbol>). bag([alice, bob]). out(X) :- bag(L), member(X, L).",
            "__xlog_list_item_symbol",
        ),
    ];

    assert_eq!(fixtures.len(), 12);

    for (label, src, expected_helper) in fixtures {
        let program = parse_program(src).unwrap_or_else(|err| panic!("{label}: parse: {err}"));
        let normalized =
            normalize_v085_lists(&program).unwrap_or_else(|err| panic!("{label}: {err}"));
        let body_preds = body_predicates(&normalized);
        assert!(
            body_preds.iter().any(|pred| pred == expected_helper),
            "{label}: expected helper {expected_helper}, got {body_preds:?}"
        );
        Compiler::new()
            .compile(src)
            .unwrap_or_else(|err| panic!("{label}: compile: {err}"));
    }
}

#[test]
fn rejects_unsafe_or_untyped_list_forms_with_typed_diagnostics() {
    let invalid = [
        (
            "heterogeneous list",
            "bad([1, alice]).",
            "heterogeneous list",
        ),
        (
            "untyped member variable",
            "out(X) :- member(X, L).",
            "known list",
        ),
        (
            "unbounded append split",
            "out(A, B) :- append(A, B, [1, 2]).",
            "unbounded append",
        ),
        (
            "untyped nth variable",
            "out(X) :- nth(0, L, X).",
            "known list",
        ),
        (
            "unsupported pair helper",
            "out(P) :- pair(P, 1, 2).",
            "pair helpers",
        ),
    ];

    for (label, src, needle) in invalid {
        let err = Compiler::new().compile(src).expect_err(label);
        let msg = err.to_string();
        assert!(
            msg.contains("list") && msg.contains(needle),
            "{label}: expected list diagnostic containing {needle:?}, got {msg}"
        );
    }
}

#[test]
fn list_type_refs_preserve_declared_element_types() {
    let program = parse_program(
        r#"
        domain node: u32.
        pred path(src: node, labels: list<symbol>, hops: list<node>).
    "#,
    )
    .expect("parse");

    assert_eq!(
        program.predicates[0].columns[1].typ,
        TypeRef::List(Box::new(TypeRef::Scalar(ScalarType::Symbol)))
    );
    assert_eq!(
        program.predicates[0].columns[2].typ,
        TypeRef::List(Box::new(TypeRef::Domain("node".to_string())))
    );
}

#[test]
fn committed_list_examples_compile() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root");
    for path in [
        "examples/v085-language/lists/membership.xlog",
        "examples/v085-language/lists/transforms.xlog",
        "examples/v085-language/lists/cons_patterns.xlog",
    ] {
        let full_path = repo_root.join(path);
        let src = std::fs::read_to_string(&full_path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", full_path.display()));
        Compiler::new()
            .compile(&src)
            .unwrap_or_else(|err| panic!("{path} failed to compile: {err}"));
    }
}
