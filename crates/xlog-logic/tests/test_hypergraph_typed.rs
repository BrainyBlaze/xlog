// crates/xlog-logic/tests/test_hypergraph_typed.rs
//! Tests for the v0.6.2 typed oracle gate (PR 5).
//!
//! Locks the typed-gating contract: evaluators consult relation
//! schemas to derive vertex types, run `analyze_typed`, and reject
//! rules whose **join-key vertices** carry types outside
//! [`WCOJ_SUPPORTED_KEY_TYPES`]. Unknown types — vertices anchored
//! only through predicates absent from `base_relations` (e.g. SCC
//! predicates derived during a fixpoint) — do **not** block. That
//! policy is named in module docs and locked by tests here.
//!
//! Cross-atom type conflict surfaces as
//! `RefEvalError::ConflictingVariableType` (new variant introduced
//! by this slice).

use std::collections::BTreeMap;
use xlog_core::ScalarType;
use xlog_logic::ast::{Atom, BodyLiteral, Rule, Term};
use xlog_logic::hypergraph::{
    evaluate_fixpoint_typed, evaluate_rule_typed, evaluate_scc_fixpoint_typed, AppearanceOrder,
    Boundary, FixpointConfig, FixpointError, RefEvalError, RefRelation, RefRelationStore, RefValue,
    SccFixpointError,
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
// evaluate_rule_typed: supported / unsupported / projection-only
// ---------------------------------------------------------------

#[test]
fn evaluate_rule_typed_evaluates_normally_for_supported_join_keys() {
    // Standard 2-hop with U32 edge schema. Y is the only join key
    // (appears in both body atoms); typed gate sees Y: U32 → fine.
    let r = rule_with(
        atom("reach", vec![var("X"), var("Z")]),
        vec![
            pos("edge", vec![var("X"), var("Y")]),
            pos("edge", vec![var("Y"), var("Z")]),
        ],
    );
    let edges = u32_relation(&[&[1, 2], &[2, 3], &[3, 4]]);
    let store = store_with_one("edge", edges);
    let rows = evaluate_rule_typed(&r, &store, &AppearanceOrder).expect("typed gate passes");
    let mut got: Vec<(u32, u32)> = rows
        .iter()
        .map(|r| match (&r[0], &r[1]) {
            (RefValue::U32(a), RefValue::U32(b)) => (*a, *b),
            other => panic!("unexpected: {other:?}"),
        })
        .collect();
    got.sort();
    assert_eq!(got, vec![(1, 3), (2, 4)]);
}

#[test]
fn evaluate_rule_typed_rejects_i32_join_key() {
    // edge schema is I32 (not in WCOJ_SUPPORTED_KEY_TYPES). Y is
    // the join key; typed gate must surface UnsupportedKeyType for
    // var "Y" with ty I32.
    let r = rule_with(
        atom("reach", vec![var("X"), var("Z")]),
        vec![
            pos("edge", vec![var("X"), var("Y")]),
            pos("edge", vec![var("Y"), var("Z")]),
        ],
    );
    let edge_i32 = RefRelation {
        schema: vec![ScalarType::I32, ScalarType::I32],
        rows: vec![
            vec![RefValue::I32(1), RefValue::I32(2)],
            vec![RefValue::I32(2), RefValue::I32(3)],
        ],
    };
    let store = store_with_one("edge", edge_i32);
    let err = evaluate_rule_typed(&r, &store, &AppearanceOrder)
        .expect_err("I32 join key must be rejected");
    match err {
        RefEvalError::Ineligible(bs) => {
            assert!(
                bs.contains(&Boundary::UnsupportedKeyType {
                    var: "Y".to_string(),
                    ty: ScalarType::I32,
                }),
                "expected UnsupportedKeyType for Y:I32, got {bs:?}"
            );
        }
        other => panic!("expected Ineligible, got {other:?}"),
    }
}

#[test]
fn evaluate_rule_typed_rejects_bool_join_key() {
    // Bool is not in WCOJ_SUPPORTED_KEY_TYPES.
    let r = rule_with(
        atom("reach", vec![var("X"), var("Z")]),
        vec![
            pos("edge", vec![var("X"), var("Y")]),
            pos("edge", vec![var("Y"), var("Z")]),
        ],
    );
    let edge_bool = RefRelation {
        schema: vec![ScalarType::Bool, ScalarType::Bool],
        rows: vec![vec![RefValue::Bool(true), RefValue::Bool(false)]],
    };
    let store = store_with_one("edge", edge_bool);
    let err = evaluate_rule_typed(&r, &store, &AppearanceOrder)
        .expect_err("Bool join key must be rejected");
    match err {
        RefEvalError::Ineligible(bs) => {
            assert!(
                bs.contains(&Boundary::UnsupportedKeyType {
                    var: "Y".to_string(),
                    ty: ScalarType::Bool,
                }),
                "expected UnsupportedKeyType for Y:Bool, got {bs:?}"
            );
        }
        other => panic!("expected Ineligible, got {other:?}"),
    }
}

#[test]
fn evaluate_rule_typed_allows_unsupported_type_in_projection_only_column() {
    // Rule: tag(X, Z) :- p(X, Y), q(X, Z)
    //   X is the only join key (appears in both atoms) → U32 from p.col_0
    //   Y is in p only (projection-only) → I32 schema column should NOT be flagged.
    //   Z is in q only.
    // Per analyze_typed, only join-key vertices are checked for
    // unsupported types. Locks the "projection-only unsupported is fine" policy.
    let r = rule_with(
        atom("tag", vec![var("X"), var("Z")]),
        vec![
            pos("p", vec![var("X"), var("Y")]),
            pos("q", vec![var("X"), var("Z")]),
        ],
    );
    let p = RefRelation {
        schema: vec![ScalarType::U32, ScalarType::I32],
        rows: vec![
            vec![RefValue::U32(1), RefValue::I32(-7)],
            vec![RefValue::U32(2), RefValue::I32(9)],
        ],
    };
    let q = u32_relation(&[&[1, 100], &[2, 200]]);
    let mut store: RefRelationStore = BTreeMap::new();
    store.insert("p".into(), p);
    store.insert("q".into(), q);
    let rows = evaluate_rule_typed(&r, &store, &AppearanceOrder)
        .expect("projection-only I32 must pass typed gate");
    let mut got: Vec<(u32, u32)> = rows
        .iter()
        .map(|r| match (&r[0], &r[1]) {
            (RefValue::U32(a), RefValue::U32(b)) => (*a, *b),
            other => panic!("unexpected: {other:?}"),
        })
        .collect();
    got.sort();
    assert_eq!(got, vec![(1, 100), (2, 200)]);
}

#[test]
fn evaluate_rule_typed_conflicts_when_variable_has_incompatible_types() {
    // Variable X appears in p[col 0] = U32 AND q[col 0] = Symbol.
    // Typed gate must surface ConflictingVariableType, not UnsupportedKeyType.
    let r = rule_with(
        atom("tag", vec![var("X")]),
        vec![
            pos("p", vec![var("X"), var("Y")]),
            pos("q", vec![var("X"), var("Z")]),
        ],
    );
    let p = u32_relation(&[&[1, 2]]);
    let q = RefRelation {
        schema: vec![ScalarType::Symbol, ScalarType::U32],
        rows: vec![vec![RefValue::Symbol("alpha".into()), RefValue::U32(7)]],
    };
    let mut store: RefRelationStore = BTreeMap::new();
    store.insert("p".into(), p);
    store.insert("q".into(), q);
    let err =
        evaluate_rule_typed(&r, &store, &AppearanceOrder).expect_err("type conflict must fail");
    match err {
        RefEvalError::ConflictingVariableType {
            var,
            first_predicate,
            first_position,
            first_type,
            second_predicate,
            second_position,
            second_type,
        } => {
            // Locks the source-order walk contract: `p` (atom 0)
            // types X first as U32; `q` (atom 1) is the conflict.
            // A future contributor reordering the walk (e.g. by
            // selectivity) would flip these assertions and surface
            // the contract change.
            assert_eq!(var, "X");
            assert_eq!(first_predicate, "p");
            assert_eq!(first_position, 0);
            assert_eq!(first_type, ScalarType::U32);
            assert_eq!(second_predicate, "q");
            assert_eq!(second_position, 0);
            assert_eq!(second_type, ScalarType::Symbol);
        }
        other => panic!("expected ConflictingVariableType, got {other:?}"),
    }
}

#[test]
fn evaluate_rule_typed_recursive_only_join_key_passes_gate() {
    // Rule body uses `odd` which is NOT in the relation store.
    // Vertices Z (join key in this rule) is anchored only through
    // `odd`, so its type is unknown to the typed gate. Per locked
    // policy, unknown ≠ unsupported: typed gate passes.
    //
    // The underlying evaluator then surfaces MissingRelation, which
    // the test propagates through. The point is that the typed gate
    // does NOT reject for "unknown type."
    let r = rule_with(
        atom("even", vec![var("X"), var("Y")]),
        vec![
            pos("odd", vec![var("X"), var("Z")]),
            pos("odd", vec![var("Z"), var("Y")]),
        ],
    );
    let store: RefRelationStore = BTreeMap::new();
    let err = evaluate_rule_typed(&r, &store, &AppearanceOrder)
        .expect_err("missing relation must surface");
    // The typed gate must pass through to evaluate_rule, which then
    // emits MissingRelation. If typed gate had rejected here, we'd
    // see Ineligible instead.
    match err {
        RefEvalError::MissingRelation(name) => {
            assert_eq!(name, "odd");
        }
        other => {
            panic!("expected MissingRelation (gate must NOT reject unknown types), got {other:?}")
        }
    }
}

#[test]
fn evaluate_rule_typed_propagates_ground_fact_boundary() {
    // Ground fact (empty body) is structurally Ineligible regardless
    // of types. Locks that structural boundaries flow through the
    // typed entry without being swallowed.
    let r = rule_with(atom("fact", vec![Term::Integer(1)]), vec![]);
    let store: RefRelationStore = BTreeMap::new();
    let err = evaluate_rule_typed(&r, &store, &AppearanceOrder)
        .expect_err("ground fact must be Ineligible");
    match err {
        RefEvalError::Ineligible(bs) => {
            assert!(
                bs.contains(&Boundary::GroundFact),
                "expected GroundFact, got {bs:?}"
            );
        }
        other => panic!("expected Ineligible(GroundFact), got {other:?}"),
    }
}

// ---------------------------------------------------------------
// evaluate_fixpoint_typed
// ---------------------------------------------------------------

#[test]
fn evaluate_fixpoint_typed_evaluates_normally_for_supported_keys() {
    // Standard recursive transitive closure with U32 edge.
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
    let edges = u32_relation(&[&[1, 2], &[2, 3], &[3, 4]]);
    let store = store_with_one("edge", edges);
    let rules = vec![r_seed, r_step];
    let result = evaluate_fixpoint_typed(
        &rules,
        &store,
        "reach",
        &AppearanceOrder,
        &FixpointConfig::default(),
    )
    .expect("typed fixpoint converges");
    // 2-hop: (1,3),(2,4); 3-hop: (1,4)
    let mut expected = vec![(1u32, 3u32), (2, 4), (1, 4)];
    expected.sort();
    assert_eq!(pairs_from_rel(&result), expected);
}

#[test]
fn evaluate_fixpoint_typed_rejects_unsupported_join_key() {
    // I32 edge schema → Y (join key in seed) typed I32 → unsupported.
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
    let edge_i32 = RefRelation {
        schema: vec![ScalarType::I32, ScalarType::I32],
        rows: vec![vec![RefValue::I32(1), RefValue::I32(2)]],
    };
    let store = store_with_one("edge", edge_i32);
    let rules = vec![r_seed, r_step];
    let err = evaluate_fixpoint_typed(
        &rules,
        &store,
        "reach",
        &AppearanceOrder,
        &FixpointConfig::default(),
    )
    .expect_err("I32 join key must be rejected by typed fixpoint");
    match err {
        FixpointError::RuleEval {
            rule_index,
            source: RefEvalError::Ineligible(bs),
        } => {
            // Either rule 0 (seed: M is the join key) or rule 1
            // (step: Y is the join key) — both surface I32.
            assert!(
                rule_index == 0 || rule_index == 1,
                "expected rule_index in {{0,1}}, got {rule_index}"
            );
            assert!(
                bs.iter().any(|b| matches!(
                    b,
                    Boundary::UnsupportedKeyType {
                        ty: ScalarType::I32,
                        ..
                    }
                )),
                "expected UnsupportedKeyType I32, got {bs:?}"
            );
        }
        other => panic!("expected RuleEval(Ineligible), got {other:?}"),
    }
}

// ---------------------------------------------------------------
// evaluate_scc_fixpoint_typed
// ---------------------------------------------------------------

#[test]
fn evaluate_scc_fixpoint_typed_evaluates_normally_for_supported_keys() {
    // Mutual recursion with U32 edges.
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
    let edges = u32_relation(&[&[1, 2], &[2, 3], &[3, 4]]);
    let store = store_with_one("edge", edges);
    let mut rules: BTreeMap<String, Vec<Rule>> = BTreeMap::new();
    rules.insert("even_path".into(), vec![even_seed, even_step]);
    rules.insert("odd_path".into(), vec![odd_seed, odd_step]);
    let result =
        evaluate_scc_fixpoint_typed(&rules, &store, &AppearanceOrder, &FixpointConfig::default())
            .expect("typed scc fixpoint converges");
    assert!(result.contains_key("even_path"));
    assert!(result.contains_key("odd_path"));
}

#[test]
fn evaluate_scc_fixpoint_typed_rejects_unsupported_join_key() {
    // Same SCC shape but edges have I32 schema → must fail with
    // SccFixpointError::RuleEval(_, _, Ineligible(UnsupportedKeyType)).
    let even_seed = rule_with(
        atom("even_path", vec![var("X"), var("Z")]),
        vec![
            pos("edge", vec![var("X"), var("M")]),
            pos("edge", vec![var("M"), var("Z")]),
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
    let edge_i32 = RefRelation {
        schema: vec![ScalarType::I32, ScalarType::I32],
        rows: vec![vec![RefValue::I32(1), RefValue::I32(2)]],
    };
    let store = store_with_one("edge", edge_i32);
    let mut rules: BTreeMap<String, Vec<Rule>> = BTreeMap::new();
    rules.insert("even_path".into(), vec![even_seed]);
    rules.insert("odd_path".into(), vec![odd_seed]);
    let err =
        evaluate_scc_fixpoint_typed(&rules, &store, &AppearanceOrder, &FixpointConfig::default())
            .expect_err("I32 join key must be rejected by typed scc");
    match err {
        SccFixpointError::RuleEval {
            source: RefEvalError::Ineligible(bs),
            ..
        } => {
            assert!(
                bs.iter().any(|b| matches!(
                    b,
                    Boundary::UnsupportedKeyType {
                        ty: ScalarType::I32,
                        ..
                    }
                )),
                "expected UnsupportedKeyType I32, got {bs:?}"
            );
        }
        other => panic!("expected RuleEval(Ineligible), got {other:?}"),
    }
}

#[test]
fn evaluate_scc_fixpoint_typed_recursive_only_join_keys_pass_gate() {
    // SCC where one rule's join keys are anchored ONLY through
    // SCC predicates (not in base_relations). Locks the policy:
    // unknown-from-base ≠ unsupported.
    //
    //   even(X, Y) :- odd(X, Z), odd(Z, Y).   -- Z's type unknown to gate
    //   odd(X, Y)  :- edge(X, M), edge(M, Y). -- M typed U32 from edge
    //
    // Both rules must clear the typed gate. The first because Z is
    // untyped from base (recursive-only); the second because M is
    // U32 (supported).
    let even_rule = rule_with(
        atom("even", vec![var("X"), var("Y")]),
        vec![
            pos("odd", vec![var("X"), var("Z")]),
            pos("odd", vec![var("Z"), var("Y")]),
        ],
    );
    let odd_rule = rule_with(
        atom("odd", vec![var("X"), var("Y")]),
        vec![
            pos("edge", vec![var("X"), var("M")]),
            pos("edge", vec![var("M"), var("Y")]),
        ],
    );
    let edges = u32_relation(&[&[1, 2], &[2, 3], &[3, 4], &[4, 5]]);
    let store = store_with_one("edge", edges);
    let mut rules: BTreeMap<String, Vec<Rule>> = BTreeMap::new();
    rules.insert("even".into(), vec![even_rule]);
    rules.insert("odd".into(), vec![odd_rule]);
    let result =
        evaluate_scc_fixpoint_typed(&rules, &store, &AppearanceOrder, &FixpointConfig::default())
            .expect("recursive-only join keys must clear typed gate");
    // odd is 2-hop edge: (1,3),(2,4),(3,5).
    // even is odd⋈odd: (1,5).
    let odd_actual = pairs_from_rel(result.get("odd").expect("odd present"));
    let mut odd_expected = vec![(1u32, 3u32), (2, 4), (3, 5)];
    odd_expected.sort();
    assert_eq!(odd_actual, odd_expected);
    let even_actual = pairs_from_rel(result.get("even").expect("even present"));
    assert_eq!(even_actual, vec![(1u32, 5u32)]);
}
