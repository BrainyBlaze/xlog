use xlog_core::ScalarType;
use xlog_logic::ast::{BodyLiteral, Term, TypeRef};
use xlog_logic::{normalize_list_builtins, normalize_meta_builtins, parse_program, Compiler};

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
                BodyLiteral::Comparison(_)
                | BodyLiteral::Epistemic(_)
                | BodyLiteral::IsExpr(_)
                | BodyLiteral::Univ(_) => None,
            })
        })
        .collect()
}

#[test]
fn parses_20_positive_and_10_negative_meta_fixtures() {
    let accepted = [
        "ok() :- ground(1).",
        "ok() :- ground(point(1, 2)).",
        "ok() :- ground([1, 2]).",
        "ok(X) :- value(X), ground(X).",
        "ok() :- nonvar(alice).",
        "ok(X) :- value(X), nonvar(X).",
        "ok() :- var(X).",
        "out(F, A) :- functor(point(1, 2), F, A).",
        "ok() :- functor(point(1, 2), point, 2).",
        "parts(P) :- point(1, 2) =.. P.",
        "ok() :- point(1, 2) =.. [point, 1, 2].",
        "xs(L) :- findall(X, edge(1, X), L).",
        "xs(S, L) :- seed(S), findall(D, edge(S, D), L).",
        "xs(L) :- findall(X, missing(X), L).",
        "all_good() :- maplist(good, [1, 2]).",
        "mapped(L) :- maplist(next, [1, 2], L).",
        "pred termbag(term). termbag(point(1, 2)). out(F) :- termbag(T), functor(T, F, 2).",
        "pred plist(list<term>). plist([point(1), point(2)]).",
        "pred pref(predref). pref(good).",
        "ok() :- maplist(good, []).",
    ];

    let rejected = [
        ("runtime maplist predicate", "bad(P) :- maplist(P, [1])."),
        ("maplist arity", "bad() :- maplist(good, [1], [2], extra)."),
        (
            "unsafe findall variable",
            "bad(L) :- findall(X, edge(Y, X), L).",
        ),
        (
            "derived findall goal",
            "derived(X) :- edge(1, X). bad(L) :- findall(X, derived(X), L).",
        ),
        ("unbound functor term", "bad(F, A) :- functor(T, F, A)."),
        ("unbound univ sides", "bad() :- T =.. P."),
        ("dynamic call", "bad() :- call(edge)."),
        ("dynamic assert", "bad() :- assert(edge(1, 2))."),
        ("dynamic retract", "bad() :- retract(edge(1, 2))."),
        ("functor arity", "bad() :- functor(point(1), point)."),
    ];

    assert_eq!(accepted.len(), 20);
    assert_eq!(rejected.len(), 10);

    for src in accepted {
        parse_program(src).unwrap_or_else(|err| panic!("accepted fixture failed: {src}: {err}"));
    }

    for (label, src) in rejected {
        let program = parse_program(src).unwrap_or_else(|err| panic!("{label}: parse: {err}"));
        let err = normalize_meta_builtins(&program).expect_err(label);
        let msg = err.to_string();
        assert!(
            msg.contains("meta normalization error"),
            "{label}: expected typed meta error, got {msg}"
        );
    }
}

#[test]
fn normalizes_meta_predicates_to_relational_helpers() {
    let src = r#"
        pred edge(src: u32, dst: u32).
        pred good(value: u32).
        pred next(input: u32, output: u32).
        edge(1, 2).
        edge(1, 3).
        good(1).
        good(2).
        next(1, 10).
        next(2, 20).

        out_functor(A) :- functor(point(1, 2), point, A).
        out_univ(N) :- point(1, 2) =.. Parts, length(Parts, N).
        out_findall(N) :- findall(X, edge(1, X), L), length(L, N).
        out_maplist(N) :- maplist(next, [1, 2], L), length(L, N).
        out_all_good(1) :- maplist(good, [1, 2]).
        out_ground(1) :- edge(1, X), ground(X), nonvar(X).
        out_var(1) :- var(Unused).
    "#;

    let program = parse_program(src).expect("parse");
    let meta_normalized = normalize_meta_builtins(&program).expect("meta normalize");
    let normalized = normalize_list_builtins(&meta_normalized).expect("list normalize");

    let body_preds = body_predicates(&normalized);
    for builtin in ["ground", "var", "nonvar", "functor", "findall", "maplist"] {
        assert!(
            !body_preds.iter().any(|pred| pred == builtin),
            "meta builtin predicate remained after normalization: {builtin}"
        );
    }

    assert!(!fact_rows(&normalized, "__xlog_meta_functor").is_empty());
    assert!(!fact_rows(&normalized, "__xlog_meta_univ").is_empty());
    assert!(
        body_preds
            .iter()
            .any(|pred| pred.starts_with("__xlog_meta_findall_")),
        "findall helper missing from body predicates: {body_preds:?}"
    );
    assert!(
        body_preds
            .iter()
            .any(|pred| pred.starts_with("__xlog_meta_maplist_")),
        "maplist helper missing from body predicates: {body_preds:?}"
    );
}

#[test]
fn compiles_safe_meta_predicates_and_finite_term_columns() {
    let src = r#"
        pred edge(src: u32, dst: u32).
        pred good(value: u32).
        pred next(input: u32, output: u32).
        pred termbag(t: term).
        pred plist(xs: list<term>).
        pred pref(p: predref).

        edge(1, 2).
        edge(1, 3).
        good(1).
        good(2).
        next(1, 10).
        next(2, 20).
        termbag(point(1, 2)).
        plist([point(1), point(2)]).
        pref(good).

        out_functor(A) :- functor(point(1, 2), point, A).
        out_term_functor(F) :- termbag(T), functor(T, F, 2).
        out_univ(N) :- point(1, 2) =.. Parts, length(Parts, N).
        out_findall(N) :- findall(X, edge(1, X), L), length(L, N).
        out_maplist(N) :- maplist(next, [1, 2], L), length(L, N).
        out_all_good(1) :- maplist(good, [1, 2]).

        ?- out_functor(A).
    "#;

    let mut compiler = Compiler::new();
    compiler.compile(src).expect("safe meta should compile");

    let termbag = compiler.schemas().get("termbag").expect("termbag schema");
    assert_eq!(termbag.column_type(0), Some(ScalarType::U64));
    let plist = compiler.schemas().get("plist").expect("plist schema");
    assert_eq!(plist.column_type(0), Some(ScalarType::U64));
    let pref = compiler.schemas().get("pref").expect("pref schema");
    assert_eq!(pref.column_type(0), Some(ScalarType::U64));

    assert!(compiler.schemas().contains_key("__xlog_meta_functor"));
    assert!(compiler.schemas().contains_key("__xlog_meta_univ"));
}

#[test]
fn rejects_unsafe_meta_forms_with_typed_diagnostics() {
    let invalid = [
        (
            "runtime predicate ref",
            "bad(P) :- maplist(P, [1]).",
            "static predicate",
        ),
        (
            "higher arity maplist",
            "bad() :- maplist(good, [1], [2], extra).",
            "maplist",
        ),
        (
            "unbound findall variable",
            "bad(L) :- findall(X, edge(Y, X), L).",
            "unsafe",
        ),
        (
            "derived findall goal",
            "edge(1, 2). derived(X) :- edge(1, X). bad(L) :- findall(X, derived(X), L).",
            "source facts",
        ),
        (
            "unbound functor term",
            "bad(F, A) :- functor(T, F, A).",
            "known finite term",
        ),
        ("unbound univ", "bad() :- T =.. P.", "known finite term"),
        ("dynamic call", "bad() :- call(edge).", "dynamic call"),
        (
            "dynamic assert",
            "bad() :- assert(edge(1, 2)).",
            "dynamic database",
        ),
        (
            "dynamic retract",
            "bad() :- retract(edge(1, 2)).",
            "dynamic database",
        ),
        (
            "bad functor arity",
            "bad() :- functor(point(1), point).",
            "expects 3",
        ),
    ];

    assert_eq!(invalid.len(), 10);

    for (label, src, needle) in invalid {
        let err = Compiler::new().compile(src).expect_err(label);
        let msg = err.to_string();
        assert!(
            msg.contains("meta normalization error") && msg.contains(needle),
            "{label}: expected meta diagnostic containing {needle:?}, got {msg}"
        );
    }
}

#[test]
fn meta_type_refs_preserve_declared_shapes() {
    let program = parse_program(
        r#"
        pred termbag(t: term).
        pred compounds(c: compound).
        pred refs(p: predref).
        pred parts(xs: list<term>).
    "#,
    )
    .expect("parse");

    assert_eq!(program.predicates[0].columns[0].typ, TypeRef::Term);
    assert_eq!(program.predicates[1].columns[0].typ, TypeRef::Compound);
    assert_eq!(program.predicates[2].columns[0].typ, TypeRef::PredRef);
    assert_eq!(
        program.predicates[3].columns[0].typ,
        TypeRef::List(Box::new(TypeRef::Term))
    );
}

#[test]
fn committed_meta_examples_compile() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root");
    for path in [
        "examples/language-completeness/meta/inspection.xlog",
        "examples/language-completeness/meta/findall.xlog",
        "examples/language-completeness/meta/maplist.xlog",
    ] {
        let full_path = repo_root.join(path);
        let src = std::fs::read_to_string(&full_path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", full_path.display()));
        Compiler::new()
            .compile(&src)
            .unwrap_or_else(|err| panic!("{path} failed to compile: {err}"));
    }
}
