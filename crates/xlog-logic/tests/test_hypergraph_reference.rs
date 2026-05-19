// crates/xlog-logic/tests/test_hypergraph_reference.rs
//! Tests for the v0.6.2 hypergraph reference evaluator (PR 2).
//!
//! Coverage strategy:
//!   * Typed analyzer (`analyze_typed`) — emits
//!     `Boundary::UnsupportedKeyType` for join-key vertices outside
//!     the WCOJ supported set (U32 / U64 / Symbol).
//!   * Reference evaluator (`evaluate_rule`) — pure-Rust oracle:
//!     deterministic, deduplicated, sorted output.
//!
//! All tests construct AST `Rule`s directly and supply
//! `RefRelationStore`s by hand to keep the surface under test
//! minimal. No parser involvement.

use std::collections::BTreeMap;
use xlog_core::ScalarType;
use xlog_logic::ast::{Atom, BodyLiteral, CompOp, Comparison, Rule, Term};
use xlog_logic::hypergraph::{
    analyze_typed, evaluate_rule, AppearanceOrder, Boundary, Eligibility, ExecutorContext,
    HypergraphRule, RefEvalError, RefRelation, RefRelationStore, RefValue, VariableOrder, VertexId,
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

fn cmp(left: Term, op: CompOp, right: Term) -> BodyLiteral {
    BodyLiteral::Comparison(Comparison { left, op, right })
}

fn rule_with(head: Atom, body: Vec<BodyLiteral>) -> Rule {
    Rule { head, body }
}

fn int(n: i64) -> Term {
    Term::Integer(n)
}

fn anon() -> Term {
    Term::Anonymous
}

// ---------------------------------------------------------------
// Typed analyzer
// ---------------------------------------------------------------

#[test]
fn unsupported_key_type_boundary_emitted_by_typed_analyzer() {
    // p(X, Z) :- e(X, Y), e(Y, Z).
    // Y is a join key (appears in both atoms). Type Y as F64 → not
    // in the WCOJ supported set {U32, U64, Symbol} → must emit
    // Boundary::UnsupportedKeyType { var: "Y", ty: F64 }.
    let r = rule_with(
        atom("p", vec![var("X"), var("Z")]),
        vec![
            pos("e", vec![var("X"), var("Y")]),
            pos("e", vec![var("Y"), var("Z")]),
        ],
    );
    let hg = HypergraphRule::from_rule(&r);

    let mut types: BTreeMap<String, ScalarType> = BTreeMap::new();
    types.insert("X".to_string(), ScalarType::U32);
    types.insert("Y".to_string(), ScalarType::F64);
    types.insert("Z".to_string(), ScalarType::U32);

    let v = analyze_typed(&hg, &types, ExecutorContext::HashFallback);
    let bs = v.boundaries();
    assert!(
        bs.iter().any(|b| matches!(
            b,
            Boundary::UnsupportedKeyType { var, ty: ScalarType::F64 } if var == "Y"
        )),
        "expected UnsupportedKeyType for Y:F64, got {bs:?}"
    );
    // Typed analyze must also keep the existing structural
    // boundaries — but this rule has none, so the verdict reduces
    // to Ineligible with exactly one boundary.
    assert_eq!(bs.len(), 1, "expected exactly one boundary, got {bs:?}");
    assert!(matches!(v, Eligibility::Ineligible(_)));
}

#[test]
fn typed_analyzer_eligible_when_all_join_keys_are_supported_types() {
    // Same shape, all U32 → no UnsupportedKeyType, no other
    // boundaries → Eligible.
    let r = rule_with(
        atom("p", vec![var("X"), var("Z")]),
        vec![
            pos("e", vec![var("X"), var("Y")]),
            pos("e", vec![var("Y"), var("Z")]),
        ],
    );
    let hg = HypergraphRule::from_rule(&r);

    let mut types: BTreeMap<String, ScalarType> = BTreeMap::new();
    types.insert("X".to_string(), ScalarType::U32);
    types.insert("Y".to_string(), ScalarType::U32);
    types.insert("Z".to_string(), ScalarType::U32);

    let v = analyze_typed(&hg, &types, ExecutorContext::HashFallback);
    assert_eq!(v, Eligibility::Eligible);
}

// ---------------------------------------------------------------
// Reference evaluator
// ---------------------------------------------------------------

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

#[test]
fn triangle_reference_evaluator_returns_expected_rows() {
    // tri(X, Y, Z) :- e(X, Y), e(Y, Z), e(Z, X).
    // e = { (1,2), (2,3), (3,1), (1,3), (4,5) } — one true triangle
    // 1→2→3→1 plus its rotations, no triangle through 4-5.
    //
    // Expected output (sorted, deduplicated as RefValue rows):
    //   (1, 2, 3)
    //   (2, 3, 1)
    //   (3, 1, 2)
    let r = rule_with(
        atom("tri", vec![var("X"), var("Y"), var("Z")]),
        vec![
            pos("e", vec![var("X"), var("Y")]),
            pos("e", vec![var("Y"), var("Z")]),
            pos("e", vec![var("Z"), var("X")]),
        ],
    );
    let edges = u32_relation(&[&[1, 2], &[2, 3], &[3, 1], &[1, 3], &[4, 5]]);
    let store = store_with_one("e", edges);

    let result = evaluate_rule(&r, &store, &AppearanceOrder).expect("evaluate triangle");

    let expected: Vec<Vec<RefValue>> = vec![
        vec![RefValue::U32(1), RefValue::U32(2), RefValue::U32(3)],
        vec![RefValue::U32(2), RefValue::U32(3), RefValue::U32(1)],
        vec![RefValue::U32(3), RefValue::U32(1), RefValue::U32(2)],
    ];
    // Output must be sorted+deduplicated. We sort the expected
    // vector with the same ordering so the comparison is direct.
    let mut expected_sorted = expected;
    expected_sorted.sort();
    assert_eq!(result, expected_sorted);
}

#[test]
fn same_generation_shape_one_step_matches_expected_rows() {
    // sg(X, Y) :- parent(X, P), parent(Y, P). — single-step Same
    // Generation: X and Y are siblings if they share a parent.
    //
    // parent: Alice→Mom, Bob→Mom, Carol→Dad, Dave→Dad.
    // sg(X, Y) for the four people = pairs (incl. self) sharing a parent.
    //
    // Single-rule semantics — NOT recursive fixpoint (per slice doc).
    let r = rule_with(
        atom("sg", vec![var("X"), var("Y")]),
        vec![
            pos("parent", vec![var("X"), var("P")]),
            pos("parent", vec![var("Y"), var("P")]),
        ],
    );
    let parent = u32_relation(&[
        &[1, 100], // Alice → Mom
        &[2, 100], // Bob → Mom
        &[3, 200], // Carol → Dad
        &[4, 200], // Dave → Dad
    ]);
    let store = store_with_one("parent", parent);

    let result = evaluate_rule(&r, &store, &AppearanceOrder).expect("evaluate sg");

    // Expected pairs: every (X, Y) where X and Y share a parent.
    // For Mom: {1,2} × {1,2} = 4 pairs.
    // For Dad: {3,4} × {3,4} = 4 pairs.
    // Total = 8 pairs (including reflexive self-pairs like (1,1)).
    let mut expected = vec![
        vec![RefValue::U32(1), RefValue::U32(1)],
        vec![RefValue::U32(1), RefValue::U32(2)],
        vec![RefValue::U32(2), RefValue::U32(1)],
        vec![RefValue::U32(2), RefValue::U32(2)],
        vec![RefValue::U32(3), RefValue::U32(3)],
        vec![RefValue::U32(3), RefValue::U32(4)],
        vec![RefValue::U32(4), RefValue::U32(3)],
        vec![RefValue::U32(4), RefValue::U32(4)],
    ];
    expected.sort();
    assert_eq!(result, expected);
}

#[test]
fn skewed_multiway_deduplicates_set_output() {
    // p(Z) :- e(X, Z), e(Y, Z), e(W, Z).
    // Z is the only join key. With many (X, Y, W) triples binding
    // to the same Z, the head projects only Z — output must
    // deduplicate to the set of Z values.
    let r = rule_with(
        atom("p", vec![var("Z")]),
        vec![
            pos("e", vec![var("X"), var("Z")]),
            pos("e", vec![var("Y"), var("Z")]),
            pos("e", vec![var("W"), var("Z")]),
        ],
    );
    // Skewed shape: many sources for Z=99, one source for Z=42.
    let edges = u32_relation(&[&[1, 99], &[2, 99], &[3, 99], &[4, 99], &[5, 99], &[7, 42]]);
    let store = store_with_one("e", edges);

    let result = evaluate_rule(&r, &store, &AppearanceOrder).expect("evaluate skewed");

    // Each rule body assignment produces 5*5*5 = 125 rows for Z=99
    // and 1*1*1 = 1 row for Z=42 — but the head is just (Z), so
    // dedup must collapse to two rows.
    let expected = vec![vec![RefValue::U32(42)], vec![RefValue::U32(99)]];
    assert_eq!(result, expected);
}

#[test]
fn comparison_filter_is_applied_after_join() {
    // p(X, Z) :- e(X, Y), e(Y, Z), Y < 3.
    // The filter Y < 3 prunes after Y is bound by the join.
    let r = rule_with(
        atom("p", vec![var("X"), var("Z")]),
        vec![
            pos("e", vec![var("X"), var("Y")]),
            pos("e", vec![var("Y"), var("Z")]),
            cmp(var("Y"), CompOp::Lt, int(3)),
        ],
    );
    // e: 1→1, 1→2, 2→3, 3→4, 4→5.
    // Y bound from first atom can be 1, 2, 3, 4, 5.
    // Y < 3 → Y ∈ {1, 2}.
    // For Y=1: X must be such that e(X, 1) — X=1. Z must be such that e(1, Z) — Z=1, Z=2.
    //   Pairs: (1,1), (1,2).
    // For Y=2: X such that e(X, 2) — X=1. Z such that e(2, Z) — Z=3.
    //   Pairs: (1,3).
    let edges = u32_relation(&[&[1, 1], &[1, 2], &[2, 3], &[3, 4], &[4, 5]]);
    let store = store_with_one("e", edges);

    let result = evaluate_rule(&r, &store, &AppearanceOrder).expect("evaluate filter");

    let mut expected = vec![
        vec![RefValue::U32(1), RefValue::U32(1)],
        vec![RefValue::U32(1), RefValue::U32(2)],
        vec![RefValue::U32(1), RefValue::U32(3)],
    ];
    expected.sort();
    assert_eq!(result, expected);
}

#[test]
fn constants_and_anonymous_wildcards_match_correctly() {
    // p(X) :- e(X, 42, _), e(X, _, _).
    // Constant 42 in position 1 of the first atom must filter rows
    // to those with column 1 = 42. Anonymous wildcards must match
    // any value (including different values across atoms).
    let r = rule_with(
        atom("p", vec![var("X")]),
        vec![
            pos("e", vec![var("X"), int(42), anon()]),
            pos("e", vec![var("X"), anon(), anon()]),
        ],
    );
    // arity-3 relation with U32 columns.
    let edges = RefRelation {
        schema: vec![ScalarType::U32, ScalarType::U32, ScalarType::U32],
        rows: vec![
            vec![RefValue::U32(1), RefValue::U32(42), RefValue::U32(100)],
            vec![RefValue::U32(2), RefValue::U32(99), RefValue::U32(200)],
            vec![RefValue::U32(3), RefValue::U32(42), RefValue::U32(300)],
        ],
    };
    let store = store_with_one("e", edges);

    let result = evaluate_rule(&r, &store, &AppearanceOrder).expect("evaluate const+wildcard");

    // First atom restricts X to {1, 3} (rows where col 1 = 42).
    // Second atom requires e(X, _, _) to exist — both 1 and 3 do.
    let expected = vec![vec![RefValue::U32(1)], vec![RefValue::U32(3)]];
    assert_eq!(result, expected);
}

#[test]
fn ineligible_rule_is_rejected_by_reference_evaluator() {
    // p(X) :- e(X, Y), not f(X, Y). — has negation → Ineligible →
    // evaluate_rule must return RefEvalError::Ineligible.
    let r = rule_with(
        atom("p", vec![var("X")]),
        vec![
            pos("e", vec![var("X"), var("Y")]),
            BodyLiteral::Negated(atom("f", vec![var("X"), var("Y")])),
        ],
    );
    let store: RefRelationStore = BTreeMap::new();
    let err = evaluate_rule(&r, &store, &AppearanceOrder).expect_err("must reject ineligible");
    match err {
        RefEvalError::Ineligible(bs) => {
            assert!(bs.contains(&Boundary::BodyNegation));
        }
        other => panic!("expected Ineligible, got {other:?}"),
    }
}

#[test]
fn symbol_join_keys_evaluate_correctly() {
    // sg(X, Y) :- parent(X, P), parent(Y, P).
    // Same shape as same_generation_shape_one_step_matches_expected_rows
    // but with Symbol-typed columns. Locks the Symbol code path
    // (string equality, distinct from integer comparison) since
    // Symbol is in WCOJ_SUPPORTED_KEY_TYPES.
    let r = rule_with(
        atom("sg", vec![var("X"), var("Y")]),
        vec![
            pos("parent", vec![var("X"), var("P")]),
            pos("parent", vec![var("Y"), var("P")]),
        ],
    );
    let parent = RefRelation {
        schema: vec![ScalarType::Symbol, ScalarType::Symbol],
        rows: vec![
            vec![
                RefValue::Symbol("alice".to_string()),
                RefValue::Symbol("mom".to_string()),
            ],
            vec![
                RefValue::Symbol("bob".to_string()),
                RefValue::Symbol("mom".to_string()),
            ],
            vec![
                RefValue::Symbol("carol".to_string()),
                RefValue::Symbol("dad".to_string()),
            ],
        ],
    };
    let store = store_with_one("parent", parent);
    let result = evaluate_rule(&r, &store, &AppearanceOrder).expect("evaluate symbol sg");

    // alice / bob share mom (both directions + reflexive).
    // carol shares dad only with herself.
    let mut expected = vec![
        vec![
            RefValue::Symbol("alice".to_string()),
            RefValue::Symbol("alice".to_string()),
        ],
        vec![
            RefValue::Symbol("alice".to_string()),
            RefValue::Symbol("bob".to_string()),
        ],
        vec![
            RefValue::Symbol("bob".to_string()),
            RefValue::Symbol("alice".to_string()),
        ],
        vec![
            RefValue::Symbol("bob".to_string()),
            RefValue::Symbol("bob".to_string()),
        ],
        vec![
            RefValue::Symbol("carol".to_string()),
            RefValue::Symbol("carol".to_string()),
        ],
    ];
    expected.sort();
    assert_eq!(result, expected);
}

#[test]
fn variable_variable_comparison_is_evaluated() {
    // p(X, Z) :- e(X, Y), e(Y, Z), X < Z.
    // Two positive atoms (eligibility-clean) plus the direction
    // filter exercising the var<var comparison code path.
    let r = rule_with(
        atom("p", vec![var("X"), var("Z")]),
        vec![
            pos("e", vec![var("X"), var("Y")]),
            pos("e", vec![var("Y"), var("Z")]),
            cmp(var("X"), CompOp::Lt, var("Z")),
        ],
    );
    // e: 1→2, 2→3, 3→1.
    // Two-hop pairs over the same relation:
    //   (1,2)+(2,3) → (X=1,Y=2,Z=3) ⇒ X<Z ⇒ keep (1,3).
    //   (2,3)+(3,1) → (X=2,Y=3,Z=1) ⇒ X<Z ⇒ drop.
    //   (3,1)+(1,2) → (X=3,Y=1,Z=2) ⇒ X<Z ⇒ drop.
    let edges = u32_relation(&[&[1, 2], &[2, 3], &[3, 1]]);
    let store = store_with_one("e", edges);
    let result = evaluate_rule(&r, &store, &AppearanceOrder).expect("var-var cmp");

    let expected = vec![vec![RefValue::U32(1), RefValue::U32(3)]];
    assert_eq!(result, expected);
}

#[test]
fn empty_result_is_ok_and_empty_not_an_error() {
    // tri on a graph with no triangles → Ok(vec![]).
    let r = rule_with(
        atom("tri", vec![var("X"), var("Y"), var("Z")]),
        vec![
            pos("e", vec![var("X"), var("Y")]),
            pos("e", vec![var("Y"), var("Z")]),
            pos("e", vec![var("Z"), var("X")]),
        ],
    );
    // Linear chain: 1→2→3→4 with no closing edge → no triangles.
    let edges = u32_relation(&[&[1, 2], &[2, 3], &[3, 4]]);
    let store = store_with_one("e", edges);
    let result = evaluate_rule(&r, &store, &AppearanceOrder);
    assert!(
        result.is_ok(),
        "no satisfying assignment must be Ok, not Err"
    );
    assert!(result.unwrap().is_empty(), "no triangles → empty result");
}

#[test]
fn appearance_order_does_not_change_result_set() {
    // Build a rule and a relation; evaluate twice with the trivial
    // AppearanceOrder and a hand-rolled ReverseOrder. Result rows
    // (after sort+dedup) must be identical — variable order is an
    // efficiency knob, not a semantic one.
    struct ReverseOrder;
    impl VariableOrder for ReverseOrder {
        fn name(&self) -> &'static str {
            "reverse"
        }
        fn order(&self, hg: &HypergraphRule) -> Vec<VertexId> {
            let mut v: Vec<VertexId> = hg.vertex_ids().collect();
            v.reverse();
            v
        }
    }

    let r = rule_with(
        atom("tri", vec![var("X"), var("Y"), var("Z")]),
        vec![
            pos("e", vec![var("X"), var("Y")]),
            pos("e", vec![var("Y"), var("Z")]),
            pos("e", vec![var("Z"), var("X")]),
        ],
    );
    let edges = u32_relation(&[&[1, 2], &[2, 3], &[3, 1], &[1, 3], &[4, 5]]);
    let store = store_with_one("e", edges);

    let r1 = evaluate_rule(&r, &store, &AppearanceOrder).expect("appearance");
    let r2 = evaluate_rule(&r, &store, &ReverseOrder).expect("reverse");
    assert_eq!(r1, r2, "variable order must not affect result set");
}

#[test]
fn typed_analyzer_ignores_unsupported_types_on_projection_only_variables() {
    // p(A, B, C, D, E) :- r1(A, B, C), r2(C, D, E).
    // Only C is a join key (appears in both atoms). Even if A/B/D/E
    // are typed as F64 (unsupported for WCOJ), they are NOT join
    // keys — typed analyze must not emit UnsupportedKeyType for
    // projection-only variables.
    let r = rule_with(
        atom("p", vec![var("A"), var("B"), var("C"), var("D"), var("E")]),
        vec![
            pos("r1", vec![var("A"), var("B"), var("C")]),
            pos("r2", vec![var("C"), var("D"), var("E")]),
        ],
    );
    let hg = HypergraphRule::from_rule(&r);

    let mut types: BTreeMap<String, ScalarType> = BTreeMap::new();
    types.insert("A".to_string(), ScalarType::F64);
    types.insert("B".to_string(), ScalarType::F64);
    types.insert("C".to_string(), ScalarType::U32);
    types.insert("D".to_string(), ScalarType::F64);
    types.insert("E".to_string(), ScalarType::F64);

    let v = analyze_typed(&hg, &types, ExecutorContext::HashFallback);
    assert_eq!(v, Eligibility::Eligible);
}

// ---------------------------------------------------------------
// Fixture validation — the evaluator is the WCOJ correctness
// oracle, so malformed inputs must surface as structured errors,
// not silent skips or misleading messages.
// ---------------------------------------------------------------

#[test]
fn relation_arity_mismatch_returns_dedicated_error() {
    // Body atom is arity-2; relation schema is arity-3. The
    // evaluator must surface RelationArityMismatch — NOT
    // ConstantTypeMismatch (which previously masked this case).
    let r = rule_with(
        atom("p", vec![var("X"), var("Z")]),
        vec![
            pos("e", vec![var("X"), var("Y")]),
            pos("e", vec![var("Y"), var("Z")]),
        ],
    );
    let edges = RefRelation {
        schema: vec![ScalarType::U32, ScalarType::U32, ScalarType::U32],
        rows: vec![],
    };
    let store = store_with_one("e", edges);
    let err = evaluate_rule(&r, &store, &AppearanceOrder).expect_err("must reject arity mismatch");
    match err {
        RefEvalError::RelationArityMismatch {
            predicate,
            atom_arity,
            relation_arity,
        } => {
            assert_eq!(predicate, "e");
            assert_eq!(atom_arity, 2);
            assert_eq!(relation_arity, 3);
        }
        other => panic!("expected RelationArityMismatch, got {other:?}"),
    }
}

#[test]
fn relation_row_arity_mismatch_returns_dedicated_error() {
    // Row 1 has length 1 against a 2-column schema. Must surface
    // RelationRowArityMismatch — not produce a misleading partial
    // match silently.
    let r = rule_with(
        atom("p", vec![var("X"), var("Z")]),
        vec![
            pos("e", vec![var("X"), var("Y")]),
            pos("e", vec![var("Y"), var("Z")]),
        ],
    );
    let edges = RefRelation {
        schema: vec![ScalarType::U32, ScalarType::U32],
        rows: vec![
            vec![RefValue::U32(1), RefValue::U32(2)],
            vec![RefValue::U32(3)], // malformed
        ],
    };
    let store = store_with_one("e", edges);
    let err = evaluate_rule(&r, &store, &AppearanceOrder).expect_err("must reject row arity");
    match err {
        RefEvalError::RelationRowArityMismatch {
            predicate,
            row_index,
            row_len,
            schema_len,
        } => {
            assert_eq!(predicate, "e");
            assert_eq!(row_index, 1);
            assert_eq!(row_len, 1);
            assert_eq!(schema_len, 2);
        }
        other => panic!("expected RelationRowArityMismatch, got {other:?}"),
    }
}

#[test]
fn relation_value_type_mismatch_returns_dedicated_error() {
    // Row carries a U64 value in a U32 column. Must surface
    // RelationValueTypeMismatch — not silently coerce or skip.
    let r = rule_with(
        atom("p", vec![var("X"), var("Z")]),
        vec![
            pos("e", vec![var("X"), var("Y")]),
            pos("e", vec![var("Y"), var("Z")]),
        ],
    );
    let edges = RefRelation {
        schema: vec![ScalarType::U32, ScalarType::U32],
        rows: vec![
            vec![RefValue::U32(1), RefValue::U32(2)],
            // Column 1 is U32 but value is U64 — drift.
            vec![RefValue::U32(3), RefValue::U64(4)],
        ],
    };
    let store = store_with_one("e", edges);
    let err = evaluate_rule(&r, &store, &AppearanceOrder).expect_err("must reject value type");
    match err {
        RefEvalError::RelationValueTypeMismatch {
            predicate,
            row_index,
            column,
            expected,
            ..
        } => {
            assert_eq!(predicate, "e");
            assert_eq!(row_index, 1);
            assert_eq!(column, 1);
            assert_eq!(expected, ScalarType::U32);
        }
        other => panic!("expected RelationValueTypeMismatch, got {other:?}"),
    }
}
