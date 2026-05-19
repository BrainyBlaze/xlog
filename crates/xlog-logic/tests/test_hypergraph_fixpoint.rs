// crates/xlog-logic/tests/test_hypergraph_fixpoint.rs
//! Tests for the v0.6.2 hypergraph fixpoint evaluator (PR 3).
//!
//! Builds on PR 2's `evaluate_rule`. Each iteration of the
//! fixpoint runs every supplied rule once against
//! `base_relations ∪ {target → derived}`, unions new rows into
//! `derived`, and stops when an iteration produces zero new rows.
//!
//! Tests cover:
//!   * Recursive Same Generation reaching the expected closure.
//!   * Transitive reach reaching the expected closure.
//!   * Idempotent rules (duplicate derivations) — fixpoint
//!     unaffected by rules that re-derive existing tuples.
//!   * `MaxIterationsExceeded` when the bound is too small for
//!     the workload.
//!   * Rule order in the input does not affect the result.
//!   * Ineligible rules surface as a structured error.
//!
//! All tests are pure-Rust and construct AST + `RefRelation`
//! values directly. No parser, no GPU.

use std::collections::BTreeMap;
use xlog_core::ScalarType;
use xlog_logic::ast::{Atom, BodyLiteral, Rule, Term};
use xlog_logic::hypergraph::{
    evaluate_fixpoint, AppearanceOrder, Boundary, FixpointConfig, FixpointError, RefEvalError,
    RefRelation, RefRelationStore, RefValue,
};

// ---------------------------------------------------------------
// (Tests grouped under "Tests" header — supplementary tests
// covering the rule-level validation surface live below.)
// ---------------------------------------------------------------

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

fn sorted_rows(rel: &RefRelation) -> Vec<Vec<RefValue>> {
    let mut rows = rel.rows.clone();
    rows.sort();
    rows
}

// ---------------------------------------------------------------
// Tests
// ---------------------------------------------------------------

#[test]
fn same_generation_recursive_reference_reaches_expected_fixpoint() {
    // sg(X, Y) :- parent(X, P), parent(Y, P).            -- base
    // sg(X, Y) :- parent(X, A), sg(A, B), parent(Y, B).  -- recursive
    //
    // Two-generation tree:
    //   gp1 -> p1 -> c1
    //   gp1 -> p2 -> c2
    //   gp2 -> p3 -> c3
    //   gp2 -> p4 -> c4
    //
    // Same Generation pairs:
    //   - peers via shared parent (none here — every parent has 1 child)
    //   - cousins: c1↔c2 share p1/p2 share gp1; c3↔c4 share p3/p4 share gp2.
    //   - parents share grandparents: p1↔p2 share gp1; p3↔p4 share gp2.
    //   - everyone is sg with themselves at every reachable depth.
    //
    // Encoding (all u32 ids):
    //   gp1=10 gp2=20  p1=1 p2=2 p3=3 p4=4  c1=11 c2=12 c3=13 c4=14
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
    let parent = u32_relation(&[
        &[1, 10],
        &[2, 10],
        &[3, 20],
        &[4, 20],
        &[11, 1],
        &[12, 2],
        &[13, 3],
        &[14, 4],
    ]);
    let store = store_with_one("parent", parent);

    let result = evaluate_fixpoint(
        &[r_base, r_step],
        &store,
        "sg",
        &AppearanceOrder,
        &FixpointConfig::default(),
    )
    .expect("fixpoint converges");

    // Expected base sg pairs (siblings + reflexive via shared P):
    //   p1↔p2 share gp1: (1,1), (1,2), (2,1), (2,2).
    //   p3↔p4 share gp2: (3,3), (3,4), (4,3), (4,4).
    //   gp* have no parent → not in sg.
    //   c* each have a unique parent → only reflexive: (11,11), (12,12),
    //   (13,13), (14,14).
    //
    // Recursive step: for each base sg pair (A, B), pair their
    // children (X via parent(X,A), Y via parent(Y,B)).
    //   From (1,1): (11, 11).
    //   From (1,2): (11, 12).
    //   From (2,1): (12, 11).
    //   From (2,2): (12, 12).
    //   From (3,3): (13, 13).
    //   From (3,4): (13, 14).
    //   From (4,3): (14, 13).
    //   From (4,4): (14, 14).
    //   From (11,11), (12,12), etc — no parent rows where X has
    //   parent 11 or 12, so step adds nothing.
    //
    // Full set is the union of all the above (de-duped).
    let mut expected = vec![
        // Base.
        (1u32, 1u32),
        (1, 2),
        (2, 1),
        (2, 2),
        (3, 3),
        (3, 4),
        (4, 3),
        (4, 4),
        (11, 11),
        (12, 12),
        (13, 13),
        (14, 14),
        // Recursive step over base.
        (11, 12),
        (12, 11),
        (13, 14),
        (14, 13),
    ];
    expected.sort();
    let actual_rows = sorted_rows(&result);
    let actual_pairs: Vec<(u32, u32)> = actual_rows
        .iter()
        .map(|r| match (&r[0], &r[1]) {
            (RefValue::U32(a), RefValue::U32(b)) => (*a, *b),
            other => panic!("unexpected RefValue shape: {other:?}"),
        })
        .collect();
    assert_eq!(actual_pairs, expected);
}

#[test]
fn transitive_reach_reference_reaches_expected_fixpoint() {
    // NOTE: This is the eligibility-respecting variant of
    // transitive reach. Classic Datalog reach uses a single-atom
    // base case (`reach(X, Y) :- edge(X, Y).`), but PR 1's
    // eligibility analyzer rejects single-atom rules via
    // `InsufficientPositiveAtoms`. Until / unless the eligibility
    // floor is widened, multiway-routed reach must use a 2-hop
    // base case as below.
    //
    // Both rules have ≥2 positive atoms (eligibility constraint).
    //   reach(X, Y) :- edge(X, M), edge(M, Y).        -- 2-hops base
    //   reach(X, Z) :- edge(X, Y), reach(Y, Z).       -- recursive extension
    //
    // Graph: 1→2→3→4→5.
    // Base 2-hop: (1,3), (2,4), (3,5).
    // Recursive: (1,4) via 1→2 + reach(2,4); (2,5) via 2→3 + reach(3,5);
    //            (1,5) via 1→2 + reach(2,5).
    let r_base = rule_with(
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

    let result = evaluate_fixpoint(
        &[r_base, r_step],
        &store,
        "reach",
        &AppearanceOrder,
        &FixpointConfig::default(),
    )
    .expect("fixpoint converges");

    let mut expected = vec![(1u32, 3u32), (2, 4), (3, 5), (1, 4), (2, 5), (1, 5)];
    expected.sort();
    let actual: Vec<(u32, u32)> = sorted_rows(&result)
        .iter()
        .map(|r| match (&r[0], &r[1]) {
            (RefValue::U32(a), RefValue::U32(b)) => (*a, *b),
            other => panic!("unexpected RefValue: {other:?}"),
        })
        .collect();
    assert_eq!(actual, expected);
}

#[test]
fn duplicate_derivations_do_not_change_fixpoint() {
    // Two rules that derive identical tuples must yield the same
    // fixpoint as one rule alone — set semantics.
    let r_a = rule_with(
        atom("reach", vec![var("X"), var("Y")]),
        vec![
            pos("edge", vec![var("X"), var("M")]),
            pos("edge", vec![var("M"), var("Y")]),
        ],
    );
    let r_b = rule_with(
        atom("reach", vec![var("X"), var("Y")]),
        vec![
            pos("edge", vec![var("X"), var("N")]),
            pos("edge", vec![var("N"), var("Y")]),
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

    let one_rule = evaluate_fixpoint(
        &[r_a.clone(), r_step.clone()],
        &store,
        "reach",
        &AppearanceOrder,
        &FixpointConfig::default(),
    )
    .expect("one-rule fixpoint");
    let two_rules = evaluate_fixpoint(
        &[r_a, r_b, r_step],
        &store,
        "reach",
        &AppearanceOrder,
        &FixpointConfig::default(),
    )
    .expect("two-rule fixpoint");

    assert_eq!(sorted_rows(&one_rule), sorted_rows(&two_rules));
}

#[test]
fn max_iterations_exceeded_returns_error() {
    // Same recursive reach but with max_iterations clamped below
    // the convergence count. Must surface
    // FixpointError::MaxIterationsExceeded.
    let r_base = rule_with(
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
    // Long chain so convergence takes more than 1 iteration.
    let edges = u32_relation(&[&[1, 2], &[2, 3], &[3, 4], &[4, 5], &[5, 6], &[6, 7]]);
    let store = store_with_one("edge", edges);

    let cfg = FixpointConfig { max_iterations: 1 };
    let err = evaluate_fixpoint(&[r_base, r_step], &store, "reach", &AppearanceOrder, &cfg)
        .expect_err("must exceed iteration cap");
    match err {
        FixpointError::MaxIterationsExceeded { limit, .. } => {
            assert_eq!(limit, 1);
        }
        other => panic!("expected MaxIterationsExceeded, got {other:?}"),
    }
}

#[test]
fn rule_order_does_not_change_fixpoint() {
    // Same rules in two orders must produce the same fixpoint.
    let r_base = rule_with(
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

    let order_a = evaluate_fixpoint(
        &[r_base.clone(), r_step.clone()],
        &store,
        "reach",
        &AppearanceOrder,
        &FixpointConfig::default(),
    )
    .expect("a");
    let order_b = evaluate_fixpoint(
        &[r_step, r_base],
        &store,
        "reach",
        &AppearanceOrder,
        &FixpointConfig::default(),
    )
    .expect("b");

    assert_eq!(sorted_rows(&order_a), sorted_rows(&order_b));
}

#[test]
fn ineligible_rule_in_fixpoint_returns_error() {
    // A rule with a single positive atom is Ineligible
    // (InsufficientPositiveAtoms). The fixpoint evaluator must
    // surface this through a structured error pointing at the
    // offending rule index.
    let r_bad = rule_with(
        atom("reach", vec![var("X"), var("Y")]),
        vec![pos("edge", vec![var("X"), var("Y")])],
    );
    let r_step = rule_with(
        atom("reach", vec![var("X"), var("Z")]),
        vec![
            pos("edge", vec![var("X"), var("Y")]),
            pos("reach", vec![var("Y"), var("Z")]),
        ],
    );
    let edges = u32_relation(&[&[1, 2]]);
    let store = store_with_one("edge", edges);

    let err = evaluate_fixpoint(
        &[r_bad, r_step],
        &store,
        "reach",
        &AppearanceOrder,
        &FixpointConfig::default(),
    )
    .expect_err("must reject ineligible rule");
    match err {
        FixpointError::RuleEval {
            rule_index,
            source: RefEvalError::Ineligible(bs),
        } => {
            assert_eq!(rule_index, 0);
            assert!(bs.contains(&Boundary::InsufficientPositiveAtoms { positive_count: 1 }));
        }
        other => panic!("expected RuleEval::Ineligible, got {other:?}"),
    }
}

// ---------------------------------------------------------------
// Supplementary tests for rule-level validation. The fixpoint
// evaluator is part of the WCOJ correctness oracle chain — every
// caller-visible error path needs its own coverage so PR 4+
// kernel comparisons cannot be fooled by silent fall-throughs.
// ---------------------------------------------------------------

#[test]
fn invalid_max_iterations_returns_dedicated_error() {
    let r = rule_with(
        atom("reach", vec![var("X"), var("Y")]),
        vec![
            pos("edge", vec![var("X"), var("M")]),
            pos("edge", vec![var("M"), var("Y")]),
        ],
    );
    let store = store_with_one("edge", u32_relation(&[&[1, 2]]));
    let cfg = FixpointConfig { max_iterations: 0 };
    let err = evaluate_fixpoint(&[r], &store, "reach", &AppearanceOrder, &cfg)
        .expect_err("max_iterations=0 must fail");
    assert!(matches!(err, FixpointError::InvalidMaxIterations));
}

#[test]
fn target_schema_indeterminable_returns_error_for_empty_rules() {
    // Empty rules slice → no rule heads → schema cannot be
    // inferred. The fixpoint must surface
    // TargetSchemaIndeterminable, not silently return an empty
    // relation with arity 0.
    let store: RefRelationStore = BTreeMap::new();
    let err = evaluate_fixpoint(
        &[],
        &store,
        "reach",
        &AppearanceOrder,
        &FixpointConfig::default(),
    )
    .expect_err("empty rules must fail");
    assert!(matches!(err, FixpointError::TargetSchemaIndeterminable));
}

#[test]
fn rule_not_for_target_returns_dedicated_error() {
    // Rule heads `reach` but caller asked for fixpoint over `sg`.
    let r = rule_with(
        atom("reach", vec![var("X"), var("Y")]),
        vec![
            pos("edge", vec![var("X"), var("M")]),
            pos("edge", vec![var("M"), var("Y")]),
        ],
    );
    let store = store_with_one("edge", u32_relation(&[&[1, 2]]));
    let err = evaluate_fixpoint(
        &[r],
        &store,
        "sg",
        &AppearanceOrder,
        &FixpointConfig::default(),
    )
    .expect_err("wrong-target rule must fail");
    match err {
        FixpointError::RuleNotForTarget {
            rule_index,
            observed,
            expected,
        } => {
            assert_eq!(rule_index, 0);
            assert_eq!(observed, "reach");
            assert_eq!(expected, "sg");
        }
        other => panic!("expected RuleNotForTarget, got {other:?}"),
    }
}

#[test]
fn head_arity_mismatch_returns_dedicated_error() {
    // Two rules headed `reach`: one arity-2, one arity-3. The
    // fixpoint must surface HeadArityMismatch BEFORE iteration —
    // not as a downstream RelationRowArityMismatch with row index.
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
    let err = evaluate_fixpoint(
        &[r_a, r_b],
        &store,
        "reach",
        &AppearanceOrder,
        &FixpointConfig::default(),
    )
    .expect_err("arity mismatch must fail");
    match err {
        FixpointError::HeadArityMismatch {
            rule_index,
            observed_arity,
            expected_arity,
        } => {
            assert_eq!(rule_index, 1);
            assert_eq!(observed_arity, 3);
            assert_eq!(expected_arity, 2);
        }
        other => panic!("expected HeadArityMismatch, got {other:?}"),
    }
}

#[test]
fn target_predicate_in_base_relations_returns_dedicated_error() {
    // base_relations pre-seeds `reach` — the fixpoint must reject,
    // not silently overwrite. If a caller wants seed tuples they
    // should encode them as a base-case rule.
    let r = rule_with(
        atom("reach", vec![var("X"), var("Y")]),
        vec![
            pos("edge", vec![var("X"), var("M")]),
            pos("edge", vec![var("M"), var("Y")]),
        ],
    );
    let mut store = store_with_one("edge", u32_relation(&[&[1, 2]]));
    store.insert("reach".to_string(), u32_relation(&[&[99, 99]]));
    let err = evaluate_fixpoint(
        &[r],
        &store,
        "reach",
        &AppearanceOrder,
        &FixpointConfig::default(),
    )
    .expect_err("seeded target must fail");
    match err {
        FixpointError::TargetPredicateInBaseRelations { name } => {
            assert_eq!(name, "reach");
        }
        other => panic!("expected TargetPredicateInBaseRelations, got {other:?}"),
    }
}
