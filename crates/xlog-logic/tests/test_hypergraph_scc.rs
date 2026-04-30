// crates/xlog-logic/tests/test_hypergraph_scc.rs
//! Tests for the v0.6.2 multi-predicate SCC fixpoint evaluator (PR 4).
//!
//! Builds on PR 3's `evaluate_fixpoint` shape but takes a
//! `BTreeMap<String, Vec<Rule>>` (rules grouped by target
//! predicate) and returns a `RefRelationStore` containing the
//! converged relations for every predicate in the SCC.
//!
//! All tests are pure-Rust and construct AST + `RefRelation`
//! values directly. No parser, no GPU.

use std::collections::BTreeMap;
use xlog_core::ScalarType;
use xlog_logic::ast::{Atom, BodyLiteral, Rule, Term};
use xlog_logic::hypergraph::{
    evaluate_scc_fixpoint, AppearanceOrder, Boundary, FixpointConfig, RefEvalError, RefRelation,
    RefRelationStore, RefValue, SccFixpointError,
};

// ---------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------

fn var(name: &str) -> Term {
    Term::Variable(name.to_string())
}

fn atom(predicate: &str, terms: Vec<Term>) -> Atom {
    Atom {
        predicate: predicate.to_string(),
        terms,
    }
}

fn pos(predicate: &str, terms: Vec<Term>) -> BodyLiteral {
    BodyLiteral::Positive(atom(predicate, terms))
}

fn rule_with(head: Atom, body: Vec<BodyLiteral>) -> Rule {
    Rule { head, body }
}

fn u32_relation(rows: &[&[u32]]) -> RefRelation {
    let arity = rows.first().map(|r| r.len()).unwrap_or(0);
    RefRelation {
        schema: vec![ScalarType::U32; arity],
        rows: rows
            .iter()
            .map(|r| r.iter().map(|v| RefValue::U32(*v)).collect())
            .collect(),
    }
}

fn store_with_one(name: &str, rel: RefRelation) -> RefRelationStore {
    let mut s: RefRelationStore = BTreeMap::new();
    s.insert(name.to_string(), rel);
    s
}

fn rules_grouped(pairs: Vec<(&str, Vec<Rule>)>) -> BTreeMap<String, Vec<Rule>> {
    let mut m: BTreeMap<String, Vec<Rule>> = BTreeMap::new();
    for (name, rs) in pairs {
        m.insert(name.to_string(), rs);
    }
    m
}

fn pairs_from_rel(rel: &RefRelation) -> Vec<(u32, u32)> {
    let mut rows = rel.rows.clone();
    rows.sort();
    rows.iter()
        .map(|r| match (&r[0], &r[1]) {
            (RefValue::U32(a), RefValue::U32(b)) => (*a, *b),
            other => panic!("unexpected RefValue shape: {other:?}"),
        })
        .collect()
}

// ---------------------------------------------------------------
// Convergence tests
// ---------------------------------------------------------------

#[test]
fn mutual_recursion_two_predicates_reaches_fixpoint() {
    // Mutual-recursion shape: two predicates `even_path` and
    // `odd_path` over edges, where path length parity flips
    // between the two predicates.
    //
    //   even_path(X, Z) :- edge(X, Y), odd_path(Y, Z).   -- recursive
    //   even_path(X, Z) :- edge(X, M), edge(M, Z).       -- 2-hop seed
    //   odd_path(X, Z)  :- edge(X, Y), even_path(Y, Z).  -- recursive
    //   odd_path(X, Z)  :- edge(X, M), edge(M, Y), edge(Y, Z). -- 3-hop seed
    //
    // Both base rules have ≥2 positive atoms (eligibility floor).
    // Graph: 1→2→3→4→5.
    let even_step = rule_with(
        atom("even_path", vec![var("X"), var("Z")]),
        vec![
            pos("edge", vec![var("X"), var("Y")]),
            pos("odd_path", vec![var("Y"), var("Z")]),
        ],
    );
    let even_seed = rule_with(
        atom("even_path", vec![var("X"), var("Z")]),
        vec![
            pos("edge", vec![var("X"), var("M")]),
            pos("edge", vec![var("M"), var("Z")]),
        ],
    );
    let odd_step = rule_with(
        atom("odd_path", vec![var("X"), var("Z")]),
        vec![
            pos("edge", vec![var("X"), var("Y")]),
            pos("even_path", vec![var("Y"), var("Z")]),
        ],
    );
    let odd_seed = rule_with(
        atom("odd_path", vec![var("X"), var("Z")]),
        vec![
            pos("edge", vec![var("X"), var("M")]),
            pos("edge", vec![var("M"), var("Y")]),
            pos("edge", vec![var("Y"), var("Z")]),
        ],
    );
    let edges = u32_relation(&[&[1, 2], &[2, 3], &[3, 4], &[4, 5]]);
    let store = store_with_one("edge", edges);
    let rules = rules_grouped(vec![
        ("even_path", vec![even_seed, even_step]),
        ("odd_path", vec![odd_seed, odd_step]),
    ]);

    let result =
        evaluate_scc_fixpoint(&rules, &store, &AppearanceOrder, &FixpointConfig::default())
            .expect("scc fixpoint converges");

    // Even-length paths in 1→2→3→4→5: 2-hop and 4-hop.
    //   2-hop: (1,3), (2,4), (3,5).
    //   4-hop: (1,5).
    let mut even_expected = vec![(1u32, 3u32), (2, 4), (3, 5), (1, 5)];
    even_expected.sort();
    let even_actual = pairs_from_rel(
        result
            .get("even_path")
            .expect("even_path present in result"),
    );
    assert_eq!(even_actual, even_expected);

    // Odd-length paths: 1-hop excluded (single-atom seed not
    // eligible), 3-hop only.
    //   3-hop: (1,4), (2,5).
    let mut odd_expected = vec![(1u32, 4u32), (2, 5)];
    odd_expected.sort();
    let odd_actual = pairs_from_rel(result.get("odd_path").expect("odd_path present in result"));
    assert_eq!(odd_actual, odd_expected);
}

#[test]
fn same_generation_multi_rule_scc_reaches_expected_rows() {
    // Single-predicate "SCC" (i.e. the SCC has one predicate) with
    // multiple rules. Exercises the SCC API on the simpler shape
    // PR 3 covered, confirming the multi-predicate evaluator
    // collapses cleanly to the single-predicate case.
    let r_base = rule_with(
        atom("sg", vec![var("X"), var("Y")]),
        vec![
            pos("parent", vec![var("X"), var("P")]),
            pos("parent", vec![var("Y"), var("P")]),
        ],
    );
    let r_step = rule_with(
        atom("sg", vec![var("X"), var("Y")]),
        vec![
            pos("parent", vec![var("X"), var("A")]),
            pos("sg", vec![var("A"), var("B")]),
            pos("parent", vec![var("Y"), var("B")]),
        ],
    );
    let parent = u32_relation(&[&[1, 10], &[2, 10], &[11, 1], &[12, 2]]);
    let store = store_with_one("parent", parent);
    let rules = rules_grouped(vec![("sg", vec![r_base, r_step])]);
    let result =
        evaluate_scc_fixpoint(&rules, &store, &AppearanceOrder, &FixpointConfig::default())
            .expect("single-predicate scc converges");

    // Base sg pairs (sharing parent 10): (1,1),(1,2),(2,1),(2,2);
    // children with unique parents: (11,11),(12,12);
    // step over base: (11,12),(12,11).
    let mut expected = vec![
        (1u32, 1u32),
        (1, 2),
        (2, 1),
        (2, 2),
        (11, 11),
        (11, 12),
        (12, 11),
        (12, 12),
    ];
    expected.sort();
    let actual = pairs_from_rel(result.get("sg").expect("sg present"));
    assert_eq!(actual, expected);
}

// ---------------------------------------------------------------
// Determinism tests
// ---------------------------------------------------------------

#[test]
fn predicate_order_does_not_change_fixpoint() {
    // Same SCC built two ways: rules grouped under one predicate
    // ordering vs the reverse. BTreeMap iteration is sorted by
    // key — the API takes a BTreeMap precisely so predicate
    // iteration is deterministic regardless of insertion order.
    // Locking that semantic.
    let r_a = rule_with(
        atom("p", vec![var("X"), var("Z")]),
        vec![
            pos("edge", vec![var("X"), var("Y")]),
            pos("q", vec![var("Y"), var("Z")]),
        ],
    );
    let r_b = rule_with(
        atom("q", vec![var("X"), var("Z")]),
        vec![
            pos("edge", vec![var("X"), var("Y")]),
            pos("p", vec![var("Y"), var("Z")]),
        ],
    );
    let r_a_seed = rule_with(
        atom("p", vec![var("X"), var("Z")]),
        vec![
            pos("edge", vec![var("X"), var("M")]),
            pos("edge", vec![var("M"), var("Z")]),
        ],
    );
    let r_b_seed = rule_with(
        atom("q", vec![var("X"), var("Z")]),
        vec![
            pos("edge", vec![var("X"), var("M")]),
            pos("edge", vec![var("M"), var("Z")]),
        ],
    );
    let edges = u32_relation(&[&[1, 2], &[2, 3], &[3, 4]]);
    let store = store_with_one("edge", edges);

    // Two BTreeMaps inserted in different orders — should produce
    // identical results because BTreeMap iteration is ordered.
    let mut rules_pq: BTreeMap<String, Vec<Rule>> = BTreeMap::new();
    rules_pq.insert("p".into(), vec![r_a_seed.clone(), r_a.clone()]);
    rules_pq.insert("q".into(), vec![r_b_seed.clone(), r_b.clone()]);
    let mut rules_qp: BTreeMap<String, Vec<Rule>> = BTreeMap::new();
    rules_qp.insert("q".into(), vec![r_b_seed, r_b]);
    rules_qp.insert("p".into(), vec![r_a_seed, r_a]);

    let res_pq = evaluate_scc_fixpoint(
        &rules_pq,
        &store,
        &AppearanceOrder,
        &FixpointConfig::default(),
    )
    .expect("pq order");
    let res_qp = evaluate_scc_fixpoint(
        &rules_qp,
        &store,
        &AppearanceOrder,
        &FixpointConfig::default(),
    )
    .expect("qp order");

    assert_eq!(
        pairs_from_rel(res_pq.get("p").unwrap()),
        pairs_from_rel(res_qp.get("p").unwrap())
    );
    assert_eq!(
        pairs_from_rel(res_pq.get("q").unwrap()),
        pairs_from_rel(res_qp.get("q").unwrap())
    );
}

#[test]
fn rule_order_within_predicate_does_not_change_fixpoint() {
    // Same predicate, two rules in different orders → same result.
    let r_seed = rule_with(
        atom("reach", vec![var("X"), var("Y")]),
        vec![
            pos("edge", vec![var("X"), var("M")]),
            pos("edge", vec![var("M"), var("Y")]),
        ],
    );
    let r_step = rule_with(
        atom("reach", vec![var("X"), var("Z")]),
        vec![
            pos("edge", vec![var("X"), var("Y")]),
            pos("reach", vec![var("Y"), var("Z")]),
        ],
    );
    let edges = u32_relation(&[&[1, 2], &[2, 3], &[3, 4], &[4, 5]]);
    let store = store_with_one("edge", edges);

    let rules_a = rules_grouped(vec![("reach", vec![r_seed.clone(), r_step.clone()])]);
    let rules_b = rules_grouped(vec![("reach", vec![r_step, r_seed])]);
    let res_a = evaluate_scc_fixpoint(
        &rules_a,
        &store,
        &AppearanceOrder,
        &FixpointConfig::default(),
    )
    .expect("a");
    let res_b = evaluate_scc_fixpoint(
        &rules_b,
        &store,
        &AppearanceOrder,
        &FixpointConfig::default(),
    )
    .expect("b");
    assert_eq!(
        pairs_from_rel(res_a.get("reach").unwrap()),
        pairs_from_rel(res_b.get("reach").unwrap())
    );
}

// ---------------------------------------------------------------
// Error path tests
// ---------------------------------------------------------------

#[test]
fn inconsistent_head_arity_in_scc_is_error() {
    // Two rules both head `reach` but with different arities.
    let r_a = rule_with(
        atom("reach", vec![var("X"), var("Y")]),
        vec![
            pos("edge", vec![var("X"), var("M")]),
            pos("edge", vec![var("M"), var("Y")]),
        ],
    );
    let r_b = rule_with(
        atom("reach", vec![var("X"), var("Y"), var("Z")]),
        vec![
            pos("edge", vec![var("X"), var("Y")]),
            pos("edge", vec![var("Y"), var("Z")]),
        ],
    );
    let store = store_with_one("edge", u32_relation(&[&[1, 2]]));
    let rules = rules_grouped(vec![("reach", vec![r_a, r_b])]);
    let err = evaluate_scc_fixpoint(&rules, &store, &AppearanceOrder, &FixpointConfig::default())
        .expect_err("arity mismatch must fail");
    match err {
        SccFixpointError::HeadArityMismatch {
            predicate,
            rule_index,
            observed_arity,
            expected_arity,
        } => {
            assert_eq!(predicate, "reach");
            assert_eq!(rule_index, 1);
            assert_eq!(observed_arity, 3);
            assert_eq!(expected_arity, 2);
        }
        other => panic!("expected HeadArityMismatch, got {other:?}"),
    }
}

#[test]
fn inconsistent_head_type_in_scc_is_error() {
    // Two rules both heading `p`. The first produces (U32, U32)
    // rows. The second produces an (I64, I64) row via head
    // `Term::Integer` projections — drift from the schema frozen
    // by the first iteration's first row.
    //
    // p(X, Y) :- edge(X, Y), edge(Y, _).         -- produces U32 rows
    // p(11, 22).                                  -- ground fact via integer
    //                                                head terms
    //
    // The ground-fact rule is itself ineligible (`GroundFact`
    // boundary), so to exercise drift cleanly we use two recursive
    // rules whose head terms have different types. The simplest
    // construction: one rule projects two U32 variables; the
    // other projects two integer constants.
    //
    // p(X, Y) :- edge(X, M), edge(M, Y).
    // p(99, 100) :- edge(_, _), edge(_, _).      -- both head terms
    //                                                are i64 constants
    let r_u32 = rule_with(
        atom("p", vec![var("X"), var("Y")]),
        vec![
            pos("edge", vec![var("X"), var("M")]),
            pos("edge", vec![var("M"), var("Y")]),
        ],
    );
    let r_i64 = rule_with(
        atom("p", vec![Term::Integer(99), Term::Integer(100)]),
        vec![
            pos("edge", vec![var("A"), var("B")]),
            pos("edge", vec![var("B"), var("C")]),
        ],
    );
    let store = store_with_one("edge", u32_relation(&[&[1, 2], &[2, 3]]));
    let rules = rules_grouped(vec![("p", vec![r_u32, r_i64])]);
    let err = evaluate_scc_fixpoint(&rules, &store, &AppearanceOrder, &FixpointConfig::default())
        .expect_err("type drift must fail");
    match err {
        SccFixpointError::InconsistentHeadValueTypes { predicate, .. } => {
            assert_eq!(predicate, "p");
        }
        other => panic!("expected InconsistentHeadValueTypes, got {other:?}"),
    }
}

#[test]
fn missing_edb_relation_in_scc_is_error() {
    // Rule references `edge`, but base_relations does not contain
    // it. Must surface as MissingRelation through the per-rule
    // RuleEval wrapper.
    let r = rule_with(
        atom("reach", vec![var("X"), var("Y")]),
        vec![
            pos("edge", vec![var("X"), var("M")]),
            pos("edge", vec![var("M"), var("Y")]),
        ],
    );
    let store: RefRelationStore = BTreeMap::new();
    let rules = rules_grouped(vec![("reach", vec![r])]);
    let err = evaluate_scc_fixpoint(&rules, &store, &AppearanceOrder, &FixpointConfig::default())
        .expect_err("missing EDB must fail");
    match err {
        SccFixpointError::RuleEval {
            predicate,
            rule_index,
            source: RefEvalError::MissingRelation(name),
        } => {
            assert_eq!(predicate, "reach");
            assert_eq!(rule_index, 0);
            assert_eq!(name, "edge");
        }
        other => panic!("expected RuleEval::MissingRelation, got {other:?}"),
    }
}

#[test]
fn max_iterations_exceeded_in_scc_is_error() {
    let r_seed = rule_with(
        atom("reach", vec![var("X"), var("Y")]),
        vec![
            pos("edge", vec![var("X"), var("M")]),
            pos("edge", vec![var("M"), var("Y")]),
        ],
    );
    let r_step = rule_with(
        atom("reach", vec![var("X"), var("Z")]),
        vec![
            pos("edge", vec![var("X"), var("Y")]),
            pos("reach", vec![var("Y"), var("Z")]),
        ],
    );
    let edges = u32_relation(&[&[1, 2], &[2, 3], &[3, 4], &[4, 5], &[5, 6]]);
    let store = store_with_one("edge", edges);
    let rules = rules_grouped(vec![("reach", vec![r_seed, r_step])]);
    let cfg = FixpointConfig { max_iterations: 1 };
    let err = evaluate_scc_fixpoint(&rules, &store, &AppearanceOrder, &cfg)
        .expect_err("must exceed iteration cap");
    match err {
        SccFixpointError::MaxIterationsExceeded { limit, .. } => {
            assert_eq!(limit, 1);
        }
        other => panic!("expected MaxIterationsExceeded, got {other:?}"),
    }
}

#[test]
fn ineligible_rule_in_scc_is_error() {
    // Single-atom body → InsufficientPositiveAtoms. SCC evaluator
    // must surface as RuleEval { predicate, rule_index, source }.
    let r_bad = rule_with(
        atom("reach", vec![var("X"), var("Y")]),
        vec![pos("edge", vec![var("X"), var("Y")])],
    );
    let store = store_with_one("edge", u32_relation(&[&[1, 2]]));
    let rules = rules_grouped(vec![("reach", vec![r_bad])]);
    let err = evaluate_scc_fixpoint(&rules, &store, &AppearanceOrder, &FixpointConfig::default())
        .expect_err("ineligible rule must fail");
    match err {
        SccFixpointError::RuleEval {
            predicate,
            rule_index,
            source: RefEvalError::Ineligible(bs),
        } => {
            assert_eq!(predicate, "reach");
            assert_eq!(rule_index, 0);
            assert!(bs.contains(&Boundary::InsufficientPositiveAtoms { positive_count: 1 }));
        }
        other => panic!("expected RuleEval::Ineligible, got {other:?}"),
    }
}

#[test]
fn rule_head_predicate_mismatch_is_error() {
    // Group key `"reach"` contains a rule whose head names `"sg"`.
    // The grouping invariant — every rule's head predicate equals
    // its BTreeMap key — is checked at function entry, before the
    // fixpoint loop runs. Surfaces as a dedicated
    // RuleHeadPredicateMismatch error so caller sees the
    // misgrouping directly rather than silently as "predicate `sg`
    // had no rules" elsewhere.
    let r_misfiled = rule_with(
        atom("sg", vec![var("X"), var("Y")]),
        vec![
            pos("edge", vec![var("X"), var("M")]),
            pos("edge", vec![var("M"), var("Y")]),
        ],
    );
    let store = store_with_one("edge", u32_relation(&[&[1, 2]]));
    let rules = rules_grouped(vec![("reach", vec![r_misfiled])]);
    let err = evaluate_scc_fixpoint(&rules, &store, &AppearanceOrder, &FixpointConfig::default())
        .expect_err("misgrouped rule must fail");
    match err {
        SccFixpointError::RuleHeadPredicateMismatch {
            group_key,
            rule_index,
            observed,
        } => {
            assert_eq!(group_key, "reach");
            assert_eq!(rule_index, 0);
            assert_eq!(observed, "sg");
        }
        other => panic!("expected RuleHeadPredicateMismatch, got {other:?}"),
    }
}

#[test]
fn scc_predicate_in_base_relations_is_error() {
    // `base_relations` pre-seeds the SCC predicate `reach`. The
    // SCC evaluator constructs all SCC predicates from the
    // fixpoint; allowing base_relations to seed any of them would
    // silently shadow the caller's seed on the first iteration.
    // Caller must encode such seeds as base-case rules instead.
    let r = rule_with(
        atom("reach", vec![var("X"), var("Y")]),
        vec![
            pos("edge", vec![var("X"), var("M")]),
            pos("edge", vec![var("M"), var("Y")]),
        ],
    );
    let mut store: RefRelationStore = BTreeMap::new();
    store.insert("edge".into(), u32_relation(&[&[1, 2]]));
    store.insert("reach".into(), u32_relation(&[&[42, 42]]));
    let rules = rules_grouped(vec![("reach", vec![r])]);
    let err = evaluate_scc_fixpoint(&rules, &store, &AppearanceOrder, &FixpointConfig::default())
        .expect_err("shadowed SCC predicate must fail");
    match err {
        SccFixpointError::PredicateInBaseRelations { name } => {
            assert_eq!(name, "reach");
        }
        other => panic!("expected PredicateInBaseRelations, got {other:?}"),
    }
}
