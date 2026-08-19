//! The owned meta/list normalizers move plain facts through untouched and keep
//! rewriting everything the passes are specified to rewrite.

use xlog_logic::{
    normalize_list_builtins, normalize_list_builtins_owned, normalize_meta_builtins,
    normalize_meta_builtins_owned, parse_program, Compiler, Term,
};

fn fact_terms(program: &xlog_logic::Program, pred: &str) -> Vec<Vec<Term>> {
    program
        .rules
        .iter()
        .filter(|rule| rule.is_fact() && rule.head.predicate == pred)
        .map(|rule| rule.head.terms.clone())
        .collect()
}

#[test]
fn meta_pass_keeps_plain_facts_verbatim_and_in_order() {
    let src = r#"
        pred typed(i64, symbol).
        domain id : u32.
        pred dom(id).
        a(1, "x", sym, 2.5, X, _).
        typed(7, seven).
        dom(3).
        b([1, 2], f(g)).
        r(X) :- a(X, _, _, _, _, _).
        a(2, "y", sym2, 3.5, Y, _).
    "#;
    let parsed = parse_program(src).expect("parse");
    let before: Vec<String> = parsed
        .rules
        .iter()
        .filter(|rule| rule.is_fact())
        .map(|rule| format!("{rule:?}"))
        .collect();
    let out = normalize_meta_builtins_owned(parsed.clone()).expect("meta");
    let after: Vec<String> = out
        .rules
        .iter()
        .filter(|rule| rule.is_fact() && !rule.head.predicate.starts_with("__xlog"))
        .map(|rule| format!("{rule:?}"))
        .collect();
    assert_eq!(
        before, after,
        "plain facts must be unchanged and keep their order"
    );
    // the pass still appends its helper declarations as before
    assert!(out
        .predicates
        .iter()
        .any(|pred| pred.name.starts_with("__xlog")));
    // by-ref and owned entry points agree
    let by_ref = normalize_meta_builtins(&parsed).expect("meta by ref");
    assert_eq!(format!("{by_ref:?}"), format!("{out:?}"));
}

#[test]
fn meta_pass_still_rewrites_facts_of_term_valued_predicates() {
    let src = r#"
        pred holds(term).
        holds(f(a, b)).
        holds(c).
        plain(f(a, b)).
    "#;
    let parsed = parse_program(src).expect("parse");
    let out = normalize_meta_builtins_owned(parsed).expect("meta");
    for terms in fact_terms(&out, "holds") {
        assert!(
            matches!(terms.as_slice(), [Term::Integer(_)]),
            "term-valued column must be lowered to a term id, got {terms:?}"
        );
    }
    let plain = fact_terms(&out, "plain");
    assert_eq!(plain.len(), 1);
    assert!(
        matches!(plain[0].as_slice(), [Term::Compound { .. }]),
        "untyped compound stays a compound: {plain:?}"
    );
}

#[test]
fn list_pass_keeps_facts_without_list_literals_verbatim() {
    let src = r#"
        pred xs(list<i64>).
        xs([1, 2, 3]).
        xs(X) :- ys(X).
        plain(1, "s", sym, V, _).
        plain(2, "t", sym2, W, _).
    "#;
    let parsed = parse_program(src).expect("parse");
    let out = normalize_list_builtins_owned(parsed.clone()).expect("list");
    assert_eq!(
        fact_terms(&out, "plain"),
        fact_terms(&parsed, "plain"),
        "facts without list literals are untouched"
    );
    let xs = fact_terms(&out, "xs");
    assert_eq!(xs.len(), 1);
    assert!(
        matches!(xs[0].as_slice(), [Term::Integer(_)]),
        "list literal in a list<i64> column becomes a list id: {xs:?}"
    );
    let by_ref = normalize_list_builtins(&parsed).expect("list by ref");
    assert_eq!(format!("{by_ref:?}"), format!("{out:?}"));
}

#[test]
fn fact_heavy_program_compiles_to_the_same_plan_from_source_and_from_ast() {
    let mut src = String::new();
    for i in 0..500 {
        src.push_str(&format!("event({i}, {}, {}).\n", i % 97, i % 13));
    }
    src.push_str("happens(E, T) :- event(E, T, _).\n");
    src.push_str("pair(A, B) :- event(A, T, K), event(B, T, K).\n");
    let from_source = Compiler::new().compile(&src).expect("compile source");
    let parsed = parse_program(&src).expect("parse");
    let from_ast = Compiler::new()
        .compile_program(&parsed)
        .expect("compile ast");
    assert_eq!(format!("{from_source:?}"), format!("{from_ast:?}"));
}
