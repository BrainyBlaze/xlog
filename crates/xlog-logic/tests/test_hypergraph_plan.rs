// crates/xlog-logic/tests/test_hypergraph_plan.rs
//! Tests for the v0.6.2 mixed plan contract (PR 6).
//!
//! Locks the dispatch contract between PR 1's eligibility analyzer
//! and PR 5's typed gate on one side, and the future planner /
//! mixed binary-multiway evaluator on the other. Each rule is
//! either a [`RulePlan::MultiwayCandidate`] (cleared the typed
//! gate, ready for WCOJ) or a [`RulePlan::BinaryFallback`]
//! (carries the boundary list explaining why). Type conflicts —
//! distinct from boundaries — surface as [`PlanError`] and refuse
//! to plan at all.
//!
//! No executor integration, no RIR lowering, no CUDA, no cost
//! model beyond `AppearanceOrder`. Pure-Rust contract.

use std::collections::BTreeMap;
use xlog_core::ScalarType;
use xlog_logic::ast::{AggExpr, AggOp, Atom, BodyLiteral, Rule, Term};
use xlog_logic::hypergraph::{
    explain_plans, plan_rule, plan_rules, Boundary, PlanError, RefRelation, RefRelationStore,
    RefValue, RulePlan,
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

fn neg(predicate: &str, terms: Vec<Term>) -> BodyLiteral {
    BodyLiteral::Negated(atom(predicate, terms))
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

// ---------------------------------------------------------------
// Single-rule plan: eligible cases
// ---------------------------------------------------------------

#[test]
fn plan_rule_eligible_triangle_is_multiway_candidate() {
    // Triangle pattern: every join key appears in 2 atoms, all
    // U32 → typed gate clears, plan must be MultiwayCandidate.
    //
    //   tri(X, Y, Z) :- e(X, Y), e(Y, Z), e(X, Z)
    let r = rule_with(
        atom("tri", vec![var("X"), var("Y"), var("Z")]),
        vec![
            pos("e", vec![var("X"), var("Y")]),
            pos("e", vec![var("Y"), var("Z")]),
            pos("e", vec![var("X"), var("Z")]),
        ],
    );
    let edges = u32_relation(&[&[1, 2], &[2, 3], &[1, 3]]);
    let store = store_with_one("e", edges);
    let plan = plan_rule(&r, &store).expect("triangle must plan");
    match plan {
        RulePlan::MultiwayCandidate {
            head_predicate,
            hypergraph,
            variable_order,
        } => {
            assert_eq!(head_predicate, "tri");
            assert_eq!(hypergraph.hyperedge_count(), 3);
            // All 3 body variables present in the order.
            assert_eq!(variable_order.len(), 3);
            // Order is deterministic per AppearanceOrder.
            let names: Vec<&str> = variable_order
                .iter()
                .map(|vid| hypergraph.vertex(*vid).name.as_str())
                .collect();
            assert_eq!(names, vec!["X", "Y", "Z"]);
        }
        other => panic!("expected MultiwayCandidate, got {other:?}"),
    }
}

// ---------------------------------------------------------------
// Single-rule plan: fallback cases (one boundary each)
// ---------------------------------------------------------------

#[test]
fn plan_rule_unsupported_key_falls_back_with_boundary() {
    // I32 edge schema — Y is the join key → UnsupportedKeyType.
    let r = rule_with(
        atom("reach", vec![var("X"), var("Z")]),
        vec![
            pos("e", vec![var("X"), var("Y")]),
            pos("e", vec![var("Y"), var("Z")]),
        ],
    );
    let edge_i32 = RefRelation {
        schema: vec![ScalarType::I32, ScalarType::I32],
        rows: vec![vec![RefValue::I32(1), RefValue::I32(2)]],
    };
    let store = store_with_one("e", edge_i32);
    let plan = plan_rule(&r, &store).expect("must plan as fallback");
    match plan {
        RulePlan::BinaryFallback {
            head_predicate,
            boundaries,
        } => {
            assert_eq!(head_predicate, "reach");
            assert!(
                boundaries.contains(&Boundary::UnsupportedKeyType {
                    var: "Y".to_string(),
                    ty: ScalarType::I32,
                }),
                "expected UnsupportedKeyType for Y:I32, got {boundaries:?}"
            );
        }
        other => panic!("expected BinaryFallback, got {other:?}"),
    }
}

#[test]
fn plan_rule_negation_falls_back_with_boundary() {
    // Body negation → BodyNegation boundary regardless of types.
    let r = rule_with(
        atom("safe", vec![var("X")]),
        vec![
            pos("e", vec![var("X"), var("Y")]),
            neg("forbidden", vec![var("X")]),
        ],
    );
    let mut store: RefRelationStore = BTreeMap::new();
    store.insert("e".into(), u32_relation(&[&[1, 2]]));
    store.insert(
        "forbidden".into(),
        RefRelation {
            schema: vec![ScalarType::U32],
            rows: vec![],
        },
    );
    let plan = plan_rule(&r, &store).expect("must plan");
    match plan {
        RulePlan::BinaryFallback {
            head_predicate,
            boundaries,
        } => {
            assert_eq!(head_predicate, "safe");
            assert!(
                boundaries.contains(&Boundary::BodyNegation),
                "expected BodyNegation, got {boundaries:?}"
            );
        }
        other => panic!("expected BinaryFallback, got {other:?}"),
    }
}

#[test]
fn plan_rule_aggregation_falls_back_with_boundary() {
    // Head with aggregate term → HeadAggregation boundary.
    let r = rule_with(
        atom(
            "cnt",
            vec![Term::Aggregate(AggExpr {
                op: AggOp::Count,
                variable: "X".into(),
            })],
        ),
        vec![
            pos("e", vec![var("X"), var("Y")]),
            pos("e", vec![var("Y"), var("Z")]),
        ],
    );
    let store = store_with_one("e", u32_relation(&[&[1, 2], &[2, 3]]));
    let plan = plan_rule(&r, &store).expect("must plan");
    match plan {
        RulePlan::BinaryFallback {
            head_predicate,
            boundaries,
        } => {
            assert_eq!(head_predicate, "cnt");
            assert!(
                boundaries.contains(&Boundary::HeadAggregation),
                "expected HeadAggregation, got {boundaries:?}"
            );
        }
        other => panic!("expected BinaryFallback, got {other:?}"),
    }
}

#[test]
fn plan_rule_ground_fact_falls_back_with_boundary() {
    // Empty body (ground fact) → GroundFact boundary.
    let r = rule_with(atom("fact", vec![Term::Integer(1)]), vec![]);
    let store: RefRelationStore = BTreeMap::new();
    let plan = plan_rule(&r, &store).expect("must plan");
    match plan {
        RulePlan::BinaryFallback {
            head_predicate,
            boundaries,
        } => {
            assert_eq!(head_predicate, "fact");
            assert!(
                boundaries.contains(&Boundary::GroundFact),
                "expected GroundFact, got {boundaries:?}"
            );
        }
        other => panic!("expected BinaryFallback, got {other:?}"),
    }
}

#[test]
fn plan_rule_insufficient_atoms_falls_back_with_boundary() {
    // Single-atom body → InsufficientPositiveAtoms{positive_count: 1}.
    let r = rule_with(
        atom("p", vec![var("X")]),
        vec![pos("e", vec![var("X"), var("Y")])],
    );
    let store = store_with_one("e", u32_relation(&[&[1, 2]]));
    let plan = plan_rule(&r, &store).expect("must plan");
    match plan {
        RulePlan::BinaryFallback {
            head_predicate,
            boundaries,
        } => {
            assert_eq!(head_predicate, "p");
            assert!(
                boundaries.contains(&Boundary::InsufficientPositiveAtoms { positive_count: 1 }),
                "expected InsufficientPositiveAtoms, got {boundaries:?}"
            );
        }
        other => panic!("expected BinaryFallback, got {other:?}"),
    }
}

// ---------------------------------------------------------------
// Single-rule plan: multiple boundaries preserved
// ---------------------------------------------------------------

#[test]
fn plan_rule_preserves_all_boundaries_when_multiple_apply() {
    // Body has both negation AND I32 join key. Plan must surface
    // BOTH boundaries so explain output and downstream callers
    // see all reasons, not just the first one.
    let r = rule_with(
        atom("reach", vec![var("X"), var("Z")]),
        vec![
            pos("e", vec![var("X"), var("Y")]),
            pos("e", vec![var("Y"), var("Z")]),
            neg("forbidden", vec![var("X")]),
        ],
    );
    let edge_i32 = RefRelation {
        schema: vec![ScalarType::I32, ScalarType::I32],
        rows: vec![vec![RefValue::I32(1), RefValue::I32(2)]],
    };
    let mut store: RefRelationStore = BTreeMap::new();
    store.insert("e".into(), edge_i32);
    store.insert(
        "forbidden".into(),
        RefRelation {
            schema: vec![ScalarType::I32],
            rows: vec![],
        },
    );
    let plan = plan_rule(&r, &store).expect("must plan");
    match plan {
        RulePlan::BinaryFallback { boundaries, .. } => {
            assert!(
                boundaries.contains(&Boundary::BodyNegation),
                "missing BodyNegation in {boundaries:?}"
            );
            assert!(
                boundaries.iter().any(|b| matches!(
                    b,
                    Boundary::UnsupportedKeyType {
                        ty: ScalarType::I32,
                        ..
                    }
                )),
                "missing UnsupportedKeyType in {boundaries:?}"
            );
        }
        other => panic!("expected BinaryFallback, got {other:?}"),
    }
}

// ---------------------------------------------------------------
// Hard error: type conflict refuses to plan
// ---------------------------------------------------------------

#[test]
fn plan_rule_conflicting_types_returns_hard_error() {
    // X has U32 in p[0], Symbol in q[0]. Type conflict is a
    // fixture error, not a plannable verdict — must surface as
    // PlanError, not as BinaryFallback. Caller must fix the
    // fixture before re-planning.
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
        rows: vec![vec![RefValue::Symbol("a".into()), RefValue::U32(0)]],
    };
    let mut store: RefRelationStore = BTreeMap::new();
    store.insert("p".into(), p);
    store.insert("q".into(), q);
    let err = plan_rule(&r, &store).expect_err("conflict must refuse to plan");
    match err {
        PlanError::ConflictingVariableType {
            var,
            first_predicate,
            first_type,
            second_predicate,
            second_type,
            ..
        } => {
            assert_eq!(var, "X");
            assert_eq!(first_predicate, "p");
            assert_eq!(first_type, ScalarType::U32);
            assert_eq!(second_predicate, "q");
            assert_eq!(second_type, ScalarType::Symbol);
        }
        other => panic!("expected ConflictingVariableType, got {other:?}"),
    }
}

// ---------------------------------------------------------------
// Multi-rule plan: mixed SCC
// ---------------------------------------------------------------

#[test]
fn plan_rules_mixed_scc_preserves_per_rule_verdicts() {
    // True mutually-recursive SCC: predicates `a` and `b` each
    // reference the other in their bodies. One rule is
    // multiway-eligible, the other carries BodyNegation and falls
    // back. Plan list reflects per-rule verdicts in source order.
    //
    //   a(X, Y) :- e(X, M), b(M, Y).             -- multiway
    //   b(X, Y) :- e(X, M), a(M, Y), !block(X).  -- negation, fallback
    let multiway_rule = rule_with(
        atom("a", vec![var("X"), var("Y")]),
        vec![
            pos("e", vec![var("X"), var("M")]),
            pos("b", vec![var("M"), var("Y")]),
        ],
    );
    let fallback_rule = rule_with(
        atom("b", vec![var("X"), var("Y")]),
        vec![
            pos("e", vec![var("X"), var("M")]),
            pos("a", vec![var("M"), var("Y")]),
            neg("block", vec![var("X")]),
        ],
    );
    let mut store: RefRelationStore = BTreeMap::new();
    store.insert("e".into(), u32_relation(&[&[1, 2]]));
    store.insert(
        "block".into(),
        RefRelation {
            schema: vec![ScalarType::U32],
            rows: vec![],
        },
    );
    let plans = plan_rules(&[multiway_rule, fallback_rule], &store).expect("plans");
    assert_eq!(plans.len(), 2);
    match &plans[0] {
        RulePlan::MultiwayCandidate { head_predicate, .. } => {
            assert_eq!(head_predicate, "a");
        }
        other => panic!("expected MultiwayCandidate at index 0, got {other:?}"),
    }
    match &plans[1] {
        RulePlan::BinaryFallback {
            head_predicate,
            boundaries,
        } => {
            assert_eq!(head_predicate, "b");
            assert!(boundaries.contains(&Boundary::BodyNegation));
        }
        other => panic!("expected BinaryFallback at index 1, got {other:?}"),
    }
}

#[test]
fn plan_rules_short_circuits_on_first_plan_error() {
    // Locks the documented contract: plan_rules stops on the first
    // PlanError. Callers wanting best-effort multi-rule planning
    // are documented to call plan_rule per-rule.
    let conflict_rule = rule_with(
        atom("tag", vec![var("X")]),
        vec![
            pos("p", vec![var("X"), var("Y")]),
            pos("q", vec![var("X"), var("Z")]),
        ],
    );
    let ok_rule = rule_with(
        atom("reach", vec![var("X"), var("Z")]),
        vec![
            pos("e", vec![var("X"), var("M")]),
            pos("e", vec![var("M"), var("Z")]),
        ],
    );
    let mut store: RefRelationStore = BTreeMap::new();
    store.insert("p".into(), u32_relation(&[&[1, 2]]));
    store.insert(
        "q".into(),
        RefRelation {
            schema: vec![ScalarType::Symbol, ScalarType::U32],
            rows: vec![vec![RefValue::Symbol("a".into()), RefValue::U32(0)]],
        },
    );
    store.insert("e".into(), u32_relation(&[&[1, 2]]));
    let err =
        plan_rules(&[conflict_rule, ok_rule], &store).expect_err("must short-circuit on conflict");
    match err {
        PlanError::ConflictingVariableType { var, .. } => {
            assert_eq!(var, "X");
        }
        other => panic!("expected ConflictingVariableType, got {other:?}"),
    }
}

// ---------------------------------------------------------------
// Explain: canonical across all input permutations
// ---------------------------------------------------------------

#[test]
fn explain_plans_is_canonical_under_same_head_reorder() {
    // Strict canonical contract: explain_plans must produce
    // identical output for ANY permutation of the input,
    // including reversal of same-head rules. Tie-break by
    // rendered body content (verdict tag, then var/boundary
    // payload), NEVER by input position.
    let r_a_seed = rule_with(
        atom("a", vec![var("X"), var("Z")]),
        vec![
            pos("e", vec![var("X"), var("M")]),
            pos("e", vec![var("M"), var("Z")]),
        ],
    );
    let r_a_neg = rule_with(
        atom("a", vec![var("X"), var("Z")]),
        vec![
            pos("e", vec![var("X"), var("Z")]),
            neg("forbidden", vec![var("X")]),
        ],
    );
    let r_b = rule_with(
        atom("b", vec![var("X"), var("Z")]),
        vec![
            pos("e", vec![var("X"), var("M")]),
            pos("e", vec![var("M"), var("Z")]),
        ],
    );
    let mut store: RefRelationStore = BTreeMap::new();
    store.insert("e".into(), u32_relation(&[&[1, 2]]));
    store.insert(
        "forbidden".into(),
        RefRelation {
            schema: vec![ScalarType::U32],
            rows: vec![],
        },
    );

    // Three orderings: original, b-first (cross-head reorder),
    // a-rules reversed (within-head reorder — the case the prior
    // contract did NOT lock).
    let plans_orig =
        plan_rules(&[r_a_seed.clone(), r_a_neg.clone(), r_b.clone()], &store).expect("plans");
    let plans_b_first =
        plan_rules(&[r_b.clone(), r_a_seed.clone(), r_a_neg.clone()], &store).expect("plans");
    let plans_a_rev =
        plan_rules(&[r_a_neg.clone(), r_a_seed.clone(), r_b.clone()], &store).expect("plans");

    let explain_orig = explain_plans(&plans_orig);
    let explain_b_first = explain_plans(&plans_b_first);
    let explain_a_rev = explain_plans(&plans_a_rev);

    assert_eq!(
        explain_orig, explain_b_first,
        "cross-head reorder must not change output\n  orig:\n{explain_orig}\n  b_first:\n{explain_b_first}"
    );
    assert_eq!(
        explain_orig, explain_a_rev,
        "within-head reorder must not change output\n  orig:\n{explain_orig}\n  a_rev:\n{explain_a_rev}"
    );

    // Spot-check ordering: 'a' comes before 'b'; verdict
    // 'multiway' sorts before 'binary-fallback' within the 'a'
    // cluster (since 'b' < 'm' < 'multiway' lexicographically:
    // 'b' for binary-fallback, 'm' for multiway — wait, 'b' <
    // 'm', so binary-fallback should sort first within a head).
    // Let's just assert structural ordering, not the within-head
    // verdict ordering, because the latter is an emergent
    // property of the body fingerprint.
    let pos_a = explain_orig.find("a/").expect("a present");
    let pos_b = explain_orig.find("b/").expect("b present");
    assert!(
        pos_a < pos_b,
        "expected 'a' before 'b' in explain output:\n{explain_orig}"
    );
    // Both a-rules present (rank 0 and rank 1).
    assert!(
        explain_orig.contains("a/0:"),
        "missing a/0 in:\n{explain_orig}"
    );
    assert!(
        explain_orig.contains("a/1:"),
        "missing a/1 in:\n{explain_orig}"
    );
}
