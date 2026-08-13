//! Regression tests: the compiler frontend must produce identical concrete
//! output for identical input, run to run.
//!
//! RelId assignment, SCC ids, rule order and stratum predicate lists used to
//! depend on HashMap/HashSet iteration order. `RandomState` reseeds per map
//! instance, so three compiles in one process already diverge — no
//! cross-process harness is needed to catch a regression here: any
//! `for (k, v) in &some_map` reintroduced on the lowering path turns these
//! tests red.

use xlog_logic::compile::Compiler;
use xlog_logic::{
    normalize_list_builtins, normalize_meta_builtins, parse_program, rewrite_magic_sets, stratify,
};

/// Enough distinct predicates that hash-order effects cannot hide, plus a
/// recursive SCC and facts (facts exercise the predicate->SCC lookup path).
fn test_program() -> String {
    let mut s = String::new();
    for i in 0..30 {
        s.push_str(&format!("p{i}(X, Z) :- e{i}(X, Y), f{i}(Y, Z).\n"));
        s.push_str(&format!("e{i}(1, 2).\n"));
        s.push_str(&format!("f{i}(2, 3).\n"));
    }
    s.push_str("reach(X, Y) :- e0(X, Y).\n");
    s.push_str("reach(X, Z) :- reach(X, Y), e0(Y, Z).\n");
    s
}

/// Concrete (not canonicalized) rendering of everything the frontend assigns:
/// RelId numbers, SCC ids and member order, stratum composition, rule order.
/// Only ordered containers are rendered, so equal content implies equal string.
fn concrete_plan_rendering(src: &str) -> String {
    let mut compiler = Compiler::new();
    let plan = compiler.compile(src).expect("compile");
    let mut rel_ids: Vec<(String, u32)> = compiler
        .rel_ids()
        .iter()
        .map(|(name, id)| (name.clone(), id.0))
        .collect();
    rel_ids.sort();
    let sccs: Vec<String> = plan
        .sccs
        .iter()
        .map(|s| format!("{}:{:?}:{}", s.id, s.predicates, s.is_recursive))
        .collect();
    let strata: Vec<Vec<u32>> = plan.strata.iter().map(|s| s.sccs.clone()).collect();
    let rules: Vec<Vec<String>> = plan
        .rules_by_scc
        .iter()
        .map(|scc_rules| scc_rules.iter().map(|r| format!("{r:?}")).collect())
        .collect();
    format!("{rel_ids:?}|{strata:?}|{sccs:?}|{rules:?}")
}

fn stratify_rendering(src: &str) -> String {
    let parsed = parse_program(src).expect("parse");
    let meta = normalize_meta_builtins(&parsed).expect("meta");
    let list = normalize_list_builtins(&meta).expect("list");
    let magic = rewrite_magic_sets(&list).expect("magic");
    let strata = stratify(&magic.program).expect("stratify");
    strata
        .iter()
        .map(|s| format!("{}:{:?}", s.id, s.predicates))
        .collect::<Vec<_>>()
        .join(";")
}

#[test]
fn compile_output_is_deterministic_within_process() {
    let src = test_program();
    let first = concrete_plan_rendering(&src);
    let second = concrete_plan_rendering(&src);
    let third = concrete_plan_rendering(&src);
    assert_eq!(first, second, "compile output differs between runs");
    assert_eq!(second, third, "compile output differs between runs");
}

#[test]
fn stratify_output_is_deterministic_within_process() {
    let src = test_program();
    let first = stratify_rendering(&src);
    let second = stratify_rendering(&src);
    let third = stratify_rendering(&src);
    assert_eq!(first, second, "stratify output differs between runs");
    assert_eq!(second, third, "stratify output differs between runs");
}
