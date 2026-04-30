// crates/xlog-logic/tests/test_hypergraph_scc_inference.rs
//! Tests for v0.6.2 transitive type inference across SCC predicates (PR 8).
//!
//! Closes the PR 5 policy gap: where a join-key vertex was anchored
//! only through SCC-recursive atoms, the typed gate previously left
//! it untyped under "unknown ≠ unsupported." Constraint propagation
//! through the rule graph (body atoms type variables; head atoms
//! back-propagate to head-predicate columns; iterate to fixpoint)
//! recovers the missing types without changing any user-visible API.
//!
//! Cyclic-only predicates (no base anchor anywhere in the rule
//! graph) remain truly unknowable; the policy narrows to
//! "unknowable-after-inference ≠ unsupported."

use std::collections::BTreeMap;
use xlog_core::ScalarType;
use xlog_logic::ast::{Atom, BodyLiteral, Rule, Term};
use xlog_logic::hypergraph::{
    evaluate_fixpoint_typed, evaluate_scc_fixpoint_typed, infer_scc_predicate_schemas,
    AppearanceOrder, Boundary, FixpointConfig, InferenceError, RefEvalError, RefRelation,
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

// ---------------------------------------------------------------
// Pure inference tests
// ---------------------------------------------------------------

#[test]
fn infer_propagates_u32_through_recursive_predicates() {
    // even(X, Y) :- odd(X, Z), odd(Z, Y).         -- recursive-only
    // odd(X, Y)  :- edge(X, M), edge(M, Y).       -- base-anchored
    //
    // edge: [U32, U32]. Inference order:
    //   1. odd from base — body types X, M, Y as U32; head propagates
    //      odd[0] = U32, odd[1] = U32.
    //   2. even from inferred odd — body types X, Z, Y as U32; head
    //      propagates even[0] = U32, even[1] = U32.
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
    let store = store_with_one("edge", u32_relation(&[&[1, 2]]));
    let rules = rules_grouped(vec![("even", vec![even_rule]), ("odd", vec![odd_rule])]);
    let inferred = infer_scc_predicate_schemas(&rules, &store).expect("inference converges");
    assert_eq!(
        inferred.get("odd").map(|s| s.as_slice()),
        Some([Some(ScalarType::U32), Some(ScalarType::U32)].as_slice())
    );
    assert_eq!(
        inferred.get("even").map(|s| s.as_slice()),
        Some([Some(ScalarType::U32), Some(ScalarType::U32)].as_slice())
    );
}

#[test]
fn infer_propagates_unsupported_type_through_recursive_predicates() {
    // Same shape but base relation is I64 (outside WCOJ_SUPPORTED_KEY_TYPES).
    // Inference must still propagate; the typed-gate layer then rejects.
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
    let edge_i64 = RefRelation {
        schema: vec![ScalarType::I64, ScalarType::I64],
        rows: vec![vec![RefValue::I64(1), RefValue::I64(2)]],
    };
    let store = store_with_one("edge", edge_i64);
    let rules = rules_grouped(vec![("even", vec![even_rule]), ("odd", vec![odd_rule])]);
    let inferred = infer_scc_predicate_schemas(&rules, &store).expect("inference converges");
    assert_eq!(
        inferred.get("odd").map(|s| s.as_slice()),
        Some([Some(ScalarType::I64), Some(ScalarType::I64)].as_slice())
    );
    assert_eq!(
        inferred.get("even").map(|s| s.as_slice()),
        Some([Some(ScalarType::I64), Some(ScalarType::I64)].as_slice())
    );
}

#[test]
fn infer_returns_none_for_cyclic_only_predicate() {
    // cyclic SCC with no base anchor: a(X) :- b(X), b(X) :- a(X).
    // No relation in base_relations is referenced. Inference
    // converges with both predicates' columns as None.
    let r_a = rule_with(
        atom("a", vec![var("X"), var("Y")]),
        vec![
            pos("b", vec![var("X"), var("Z")]),
            pos("b", vec![var("Z"), var("Y")]),
        ],
    );
    let r_b = rule_with(
        atom("b", vec![var("X"), var("Y")]),
        vec![
            pos("a", vec![var("X"), var("Z")]),
            pos("a", vec![var("Z"), var("Y")]),
        ],
    );
    let store: RefRelationStore = BTreeMap::new();
    let rules = rules_grouped(vec![("a", vec![r_a]), ("b", vec![r_b])]);
    let inferred = infer_scc_predicate_schemas(&rules, &store).expect("inference converges");
    assert_eq!(
        inferred.get("a").map(|s| s.as_slice()),
        Some([None, None].as_slice())
    );
    assert_eq!(
        inferred.get("b").map(|s| s.as_slice()),
        Some([None, None].as_slice())
    );
}

#[test]
fn infer_detects_conflicting_predicate_column_types() {
    // Two rules for `p` head it with incompatible types in the same
    // column. Rule 0 → p[0] = U32 (from edge_u32). Rule 1 → p[0] =
    // Symbol (from edge_sym). Inference must surface
    // ConflictingPredicateColumnType.
    let r_u32 = rule_with(
        atom("p", vec![var("X"), var("Y")]),
        vec![
            pos("edge_u32", vec![var("X"), var("M")]),
            pos("edge_u32", vec![var("M"), var("Y")]),
        ],
    );
    let r_sym = rule_with(
        atom("p", vec![var("X"), var("Y")]),
        vec![
            pos("edge_sym", vec![var("X"), var("M")]),
            pos("edge_sym", vec![var("M"), var("Y")]),
        ],
    );
    let mut store: RefRelationStore = BTreeMap::new();
    store.insert("edge_u32".into(), u32_relation(&[&[1, 2]]));
    store.insert(
        "edge_sym".into(),
        RefRelation {
            schema: vec![ScalarType::Symbol, ScalarType::Symbol],
            rows: vec![vec![
                RefValue::Symbol("a".into()),
                RefValue::Symbol("b".into()),
            ]],
        },
    );
    let rules = rules_grouped(vec![("p", vec![r_u32, r_sym])]);
    let err = infer_scc_predicate_schemas(&rules, &store).expect_err("conflict must fail");
    match err {
        InferenceError::ConflictingPredicateColumnType {
            predicate,
            column,
            first_type,
            second_type,
            ..
        } => {
            assert_eq!(predicate, "p");
            assert_eq!(column, 0);
            // Rule 0 wins first, rule 1 conflicts.
            assert_eq!(first_type, ScalarType::U32);
            assert_eq!(second_type, ScalarType::Symbol);
        }
    }
}

#[test]
fn infer_handles_single_predicate_self_recursion() {
    // Single predicate with both base and recursive rule:
    //   reach(X, Y) :- edge(X, M), edge(M, Y).             -- base
    //   reach(X, Z) :- edge(X, Y), reach(Y, Z).            -- step
    //
    // Inference: base rule alone gives reach = [U32, U32]. Step
    // rule re-confirms both columns. Convergence in one pass.
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
    let store = store_with_one("edge", u32_relation(&[&[1, 2]]));
    let rules = rules_grouped(vec![("reach", vec![r_base, r_step])]);
    let inferred = infer_scc_predicate_schemas(&rules, &store).expect("converges");
    assert_eq!(
        inferred.get("reach").map(|s| s.as_slice()),
        Some([Some(ScalarType::U32), Some(ScalarType::U32)].as_slice())
    );
}

// ---------------------------------------------------------------
// End-to-end: typed evaluators use inference
// ---------------------------------------------------------------

#[test]
fn scc_fixpoint_typed_rejects_recursive_only_unsupported_via_inference() {
    // Was the PR 5 gap: even rule's join key Z is anchored only
    // through `odd` body atoms. Without inference, Z stayed
    // untyped and even passed the typed gate even when odd's
    // schema was unsupported. With inference, odd[0] = I64
    // propagates → Z typed I64 → UnsupportedKeyType.
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
    let edge_i64 = RefRelation {
        schema: vec![ScalarType::I64, ScalarType::I64],
        rows: vec![vec![RefValue::I64(1), RefValue::I64(2)]],
    };
    let store = store_with_one("edge", edge_i64);
    let rules = rules_grouped(vec![("even", vec![even_rule]), ("odd", vec![odd_rule])]);
    let err =
        evaluate_scc_fixpoint_typed(&rules, &store, &AppearanceOrder, &FixpointConfig::default())
            .expect_err("recursive-only I64 join key must be rejected via inference");
    match err {
        SccFixpointError::RuleEval {
            source: RefEvalError::Ineligible(bs),
            ..
        } => {
            assert!(
                bs.iter().any(|b| matches!(
                    b,
                    Boundary::UnsupportedKeyType {
                        ty: ScalarType::I64,
                        ..
                    }
                )),
                "expected UnsupportedKeyType I64, got {bs:?}"
            );
        }
        other => panic!("expected RuleEval(Ineligible), got {other:?}"),
    }
}

#[test]
fn scc_fixpoint_typed_evaluates_normally_for_supported_keys_via_inference() {
    // Same shape with U32 base — must converge. The PR 5 test of
    // the same name already covers this; this one re-locks it
    // post-inference (sanity that inference doesn't break the
    // happy path).
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
    let rules = rules_grouped(vec![("even", vec![even_rule]), ("odd", vec![odd_rule])]);
    let result =
        evaluate_scc_fixpoint_typed(&rules, &store, &AppearanceOrder, &FixpointConfig::default())
            .expect("supported keys must converge");
    assert!(result.contains_key("even"));
    assert!(result.contains_key("odd"));
}

#[test]
fn fixpoint_typed_rejects_target_recursive_only_unsupported_via_inference() {
    // Single-target fixpoint: target predicate `reach` has its
    // recursive case `reach(X, Z) :- edge(X, Y), reach(Y, Z)`.
    // Z is anchored only via `reach[1]` (recursive). With
    // inference, target schema is computed from base case →
    // reach = [I64, I64]. Recursive rule's typed gate then sees
    // Z: I64 → UnsupportedKeyType.
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
    let edge_i64 = RefRelation {
        schema: vec![ScalarType::I64, ScalarType::I64],
        rows: vec![vec![RefValue::I64(1), RefValue::I64(2)]],
    };
    let store = store_with_one("edge", edge_i64);
    let rules = vec![r_base, r_step];
    let err = evaluate_fixpoint_typed(
        &rules,
        &store,
        "reach",
        &AppearanceOrder,
        &FixpointConfig::default(),
    )
    .expect_err("target schema with I64 must be rejected via inference");
    // Either rule may surface the verdict — base on its own join
    // key M, step on its recursive join key Y. Just assert the
    // I64 boundary appears.
    match err {
        FixpointError::RuleEval {
            source: RefEvalError::Ineligible(bs),
            ..
        } => {
            assert!(
                bs.iter().any(|b| matches!(
                    b,
                    Boundary::UnsupportedKeyType {
                        ty: ScalarType::I64,
                        ..
                    }
                )),
                "expected UnsupportedKeyType I64, got {bs:?}"
            );
        }
        other => panic!("expected RuleEval(Ineligible), got {other:?}"),
    }
}

#[test]
fn cyclic_only_predicate_still_passes_typed_gate_locked_policy() {
    // Locks the narrowed policy: unknowable-after-inference ≠
    // unsupported. SCC where neither predicate has any base
    // anchor → inference returns None for every column. Typed
    // gate must still pass these rules (even though it cannot
    // confirm join-key types).
    //
    // Using the same a↔b shape as `infer_returns_none_for_cyclic_only_predicate`.
    // We add `&edge` to the body of one rule so inference partially
    // anchors — but we keep one variable purely cyclic.
    //
    // Actually for simplicity, use plain a↔b with no base; since
    // the rules are structurally Eligible (≥2 atoms, no negation),
    // the typed gate should not surface UnsupportedKeyType. Each
    // rule's evaluator will then fail with MissingRelation on
    // first iteration (because a and b have no defined seed),
    // surfaced through SccFixpointError::RuleEval. The point is:
    // typed gate doesn't reject for unknowable types.
    let r_a = rule_with(
        atom("a", vec![var("X"), var("Y")]),
        vec![
            pos("b", vec![var("X"), var("Z")]),
            pos("b", vec![var("Z"), var("Y")]),
        ],
    );
    let r_b = rule_with(
        atom("b", vec![var("X"), var("Y")]),
        vec![
            pos("a", vec![var("X"), var("Z")]),
            pos("a", vec![var("Z"), var("Y")]),
        ],
    );
    let store: RefRelationStore = BTreeMap::new();
    let rules = rules_grouped(vec![("a", vec![r_a]), ("b", vec![r_b])]);
    let result = evaluate_scc_fixpoint_typed(
        &rules,
        &store,
        &AppearanceOrder,
        &FixpointConfig { max_iterations: 2 },
    );
    // The SCC has no seed source, so every iteration produces 0
    // rows. With max_iterations=2 and no-growth convergence,
    // result should be Ok with empty relations — NOT
    // RuleEval(Ineligible(UnsupportedKeyType)). Locks the policy.
    match result {
        Ok(store) => {
            assert!(store.contains_key("a"));
            assert!(store.contains_key("b"));
            assert_eq!(store.get("a").unwrap().rows.len(), 0);
            assert_eq!(store.get("b").unwrap().rows.len(), 0);
        }
        Err(SccFixpointError::RuleEval {
            source: RefEvalError::Ineligible(bs),
            ..
        }) => {
            // If inference somehow flagged unknowable as unsupported,
            // this branch would fire — that's the regression we're
            // locking against.
            panic!("cyclic-only predicate should NOT be rejected by typed gate, got: {bs:?}");
        }
        Err(other) => {
            panic!("unexpected error: {other:?}");
        }
    }
}

// FixpointError needs to be in scope for the fixpoint test.
use xlog_logic::hypergraph::FixpointError;
