// crates/xlog-logic/tests/test_hypergraph_pr9_contracts.rs
//! Tests for v0.6.2 PR 9 contract repair: SCC-aware planning and
//! structural-error precedence over inference.
//!
//! Closes two gaps observed during PR 8 deep validation:
//!
//!   1. The planner (`plan_rule` / `plan_rules`) was base-only,
//!      so a recursive-only unsupported-key rule could be marked
//!      `MultiwayCandidate` while `evaluate_scc_fixpoint_typed`
//!      (PR 8 inference-aware) rejected the same SCC. PR 9 adds
//!      `plan_scc_rules` that runs inference first, so plan and
//!      evaluator verdicts agree.
//!
//!   2. Inference ran *before* per-rule head-match validation, so
//!      a misgrouped rule could surface as
//!      `InferenceConflict` (via back-prop into the wrong group
//!      key) before the structural
//!      `RuleHeadPredicateMismatch` / `RuleNotForTarget` fired.
//!      PR 9 reorders: structural per-rule head-match precedence
//!      check first; inference only on a clean group.

use std::collections::BTreeMap;
use xlog_core::ScalarType;
use xlog_logic::ast::{Atom, BodyLiteral, Rule, Term};
use xlog_logic::hypergraph::{
    evaluate_fixpoint_typed, evaluate_scc_fixpoint_typed, plan_scc_rules, AppearanceOrder,
    Boundary, FixpointConfig, FixpointError, PlanError, RefEvalError, RefRelation,
    RefRelationStore, RefValue, RulePlan, SccFixpointError,
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
// Gap 1: SCC-aware planning
// ---------------------------------------------------------------

#[test]
fn plan_scc_rules_rejects_recursive_only_unsupported_via_inference() {
    // The PR 5/8 flagship case:
    //   even(X, Y) :- odd(X, Z), odd(Z, Y).
    //   odd(X, Y)  :- edge(X, M), edge(M, Y).
    // edge: I64. Inference propagates odd = [I64, I64] → even
    // body's Z (the recursive-only join key) is typed I64 →
    // UnsupportedKeyType.
    //
    // Per-rule plan_rule would mark `even` as MultiwayCandidate
    // (no I64 known from base for `even`'s body atoms which all
    // reference `odd` not in base). The SCC-aware plan_scc_rules
    // must reject `even` consistent with evaluate_scc_fixpoint_typed.
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
    let plans = plan_scc_rules(&rules, &store).expect("must plan");

    // even should be BinaryFallback with UnsupportedKeyType{ty: I64}.
    let even_plans = plans.get("even").expect("even present");
    assert_eq!(even_plans.len(), 1);
    match &even_plans[0] {
        RulePlan::BinaryFallback { boundaries, .. } => {
            assert!(
                boundaries.iter().any(|b| matches!(
                    b,
                    Boundary::UnsupportedKeyType {
                        ty: ScalarType::I64,
                        ..
                    }
                )),
                "expected UnsupportedKeyType I64 in even, got {boundaries:?}"
            );
        }
        other => {
            panic!("expected BinaryFallback for even (recursive-only unsupported), got {other:?}")
        }
    }
    // odd should also fall back (its M is I64 directly).
    let odd_plans = plans.get("odd").expect("odd present");
    match &odd_plans[0] {
        RulePlan::BinaryFallback { boundaries, .. } => {
            assert!(
                boundaries.iter().any(|b| matches!(
                    b,
                    Boundary::UnsupportedKeyType {
                        ty: ScalarType::I64,
                        ..
                    }
                )),
                "expected UnsupportedKeyType I64 in odd, got {boundaries:?}"
            );
        }
        other => panic!("expected BinaryFallback for odd, got {other:?}"),
    }
}

#[test]
fn plan_scc_rules_returns_multiway_for_supported_keys() {
    // Same shape with U32 base — both rules must be
    // MultiwayCandidate after inference clears them.
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
    let plans = plan_scc_rules(&rules, &store).expect("must plan");
    for (predicate, group) in &plans {
        for plan in group {
            match plan {
                RulePlan::MultiwayCandidate { head_predicate, .. } => {
                    assert_eq!(head_predicate, predicate);
                }
                other => panic!("expected MultiwayCandidate for {predicate}, got {other:?}"),
            }
        }
    }
}

#[test]
fn plan_scc_rules_surfaces_inference_conflict_as_plan_error() {
    // Two rules for `p` head with incompatible base-relation
    // schemas at column 0 → InferenceError::ConflictingPredicateColumnType
    // → must surface as PlanError::InferenceConflict at the SCC
    // planner.
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
    let err = plan_scc_rules(&rules, &store).expect_err("inference conflict must fail");
    match err {
        PlanError::InferenceConflict {
            predicate,
            column,
            first_type,
            second_type,
            ..
        } => {
            assert_eq!(predicate, "p");
            assert_eq!(column, 0);
            assert_eq!(first_type, ScalarType::U32);
            assert_eq!(second_type, ScalarType::Symbol);
        }
        other => panic!("expected InferenceConflict, got {other:?}"),
    }
}

#[test]
fn plan_scc_rules_within_rule_body_conflict_still_surfaces() {
    // X has U32 in p[0] and Symbol in q[0] within ONE rule's body.
    // This is a within-rule conflict, caught by the existing
    // ConflictingVariableType path — distinct from
    // InferenceConflict.
    let conflict_rule = rule_with(
        atom("tag", vec![var("X")]),
        vec![
            pos("p", vec![var("X"), var("Y")]),
            pos("q", vec![var("X"), var("Z")]),
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
    let rules = rules_grouped(vec![("tag", vec![conflict_rule])]);
    let err = plan_scc_rules(&rules, &store).expect_err("body conflict must fail");
    match err {
        PlanError::ConflictingVariableType { var, .. } => {
            assert_eq!(var, "X");
        }
        other => panic!("expected ConflictingVariableType, got {other:?}"),
    }
}

// (Inference conflict test above already pattern-matches
// PlanError::InferenceConflict explicitly; no exhaustiveness fix
// needed there.)

// ---------------------------------------------------------------
// Gap 2: Structural-error precedence repair
// ---------------------------------------------------------------

#[test]
fn evaluate_scc_fixpoint_typed_misgrouped_rule_wins_over_inference() {
    // Group key "reach" contains a rule whose head names "sg".
    // The rule's body uses I64 edge — without precedence repair,
    // inference could back-propagate into "reach" and surface
    // an InferenceConflict (or unsupported boundary) before the
    // structural RuleHeadPredicateMismatch fires.
    let misgrouped = rule_with(
        atom("sg", vec![var("X"), var("Z")]),
        vec![
            pos("edge", vec![var("X"), var("Y")]),
            pos("edge", vec![var("Y"), var("Z")]),
        ],
    );
    let edge_i64 = RefRelation {
        schema: vec![ScalarType::I64, ScalarType::I64],
        rows: vec![vec![RefValue::I64(1), RefValue::I64(2)]],
    };
    let store = store_with_one("edge", edge_i64);
    let rules = rules_grouped(vec![("reach", vec![misgrouped])]);
    let err =
        evaluate_scc_fixpoint_typed(&rules, &store, &AppearanceOrder, &FixpointConfig::default())
            .expect_err("misgrouped rule must surface structural error");
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
        other => {
            panic!("expected RuleHeadPredicateMismatch (structural precedence), got {other:?}")
        }
    }
}

#[test]
fn evaluate_fixpoint_typed_wrong_target_wins_over_inference() {
    // Single-target fixpoint: target_predicate "reach", rule
    // headed "not_reach" with I64 base. Inference would propagate
    // I64 if it ran; structural RuleNotForTarget must win.
    let wrong_target = rule_with(
        atom("not_reach", vec![var("X"), var("Z")]),
        vec![
            pos("edge", vec![var("X"), var("Y")]),
            pos("edge", vec![var("Y"), var("Z")]),
        ],
    );
    let edge_i64 = RefRelation {
        schema: vec![ScalarType::I64, ScalarType::I64],
        rows: vec![vec![RefValue::I64(1), RefValue::I64(2)]],
    };
    let store = store_with_one("edge", edge_i64);
    let rules = vec![wrong_target];
    let err = evaluate_fixpoint_typed(
        &rules,
        &store,
        "reach",
        &AppearanceOrder,
        &FixpointConfig::default(),
    )
    .expect_err("wrong-target rule must surface structural error");
    match err {
        FixpointError::RuleNotForTarget {
            rule_index,
            observed,
            expected,
        } => {
            assert_eq!(rule_index, 0);
            assert_eq!(observed, "not_reach");
            assert_eq!(expected, "reach");
        }
        other => panic!("expected RuleNotForTarget (structural precedence), got {other:?}"),
    }
}

// ---------------------------------------------------------------
// Plan-evaluator agreement (cross-check)
// ---------------------------------------------------------------

#[test]
fn plan_scc_rules_agrees_with_evaluate_scc_fixpoint_typed_on_unsupported() {
    // For the recursive-only-unsupported fixture, plan_scc_rules
    // returns BinaryFallback with UnsupportedKeyType I64, AND
    // evaluate_scc_fixpoint_typed returns RuleEval(Ineligible)
    // with the same boundary. Verdicts must agree.
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
    let rules = rules_grouped(vec![
        ("even", vec![even_rule.clone()]),
        ("odd", vec![odd_rule.clone()]),
    ]);

    let plans = plan_scc_rules(&rules, &store).expect("plan must succeed");
    let plan_has_unsupported_i64 = plans.values().flat_map(|v| v.iter()).any(|p| match p {
        RulePlan::BinaryFallback { boundaries, .. } => boundaries.iter().any(|b| {
            matches!(
                b,
                Boundary::UnsupportedKeyType {
                    ty: ScalarType::I64,
                    ..
                }
            )
        }),
        _ => false,
    });
    assert!(
        plan_has_unsupported_i64,
        "plan_scc_rules must produce UnsupportedKeyType I64 fallback"
    );

    let eval_err =
        evaluate_scc_fixpoint_typed(&rules, &store, &AppearanceOrder, &FixpointConfig::default())
            .expect_err("evaluator must reject");
    match eval_err {
        SccFixpointError::RuleEval { source, .. } => {
            // Source must mention the same I64 unsupported reason.
            let s = format!("{source:?}");
            assert!(
                s.contains("I64"),
                "evaluator and plan must agree on I64 reason; eval said: {s}"
            );
        }
        other => panic!("expected RuleEval, got {other:?}"),
    }
}

// ---------------------------------------------------------------
// Tightened precedence test: discriminates structural-runs-instead
// of inference-didn't-conflict (advisor item 2)
// ---------------------------------------------------------------

#[test]
fn evaluate_scc_fixpoint_typed_misgrouped_blocks_inference_that_would_conflict() {
    // Fixture where inference would produce a SPECIFIC
    // `InferenceConflict` if it ran: two rules for predicate "p"
    // with incompatible base schemas at column 0 — except one of
    // them is misgrouped under group key "q".
    //
    // Pre-flight precedence repair: structural
    // RuleHeadPredicateMismatch must win, NOT
    // RuleEval(InferenceConflict). The latter would indicate the
    // pre-flight scan didn't run before inference (regression).
    let r_u32 = rule_with(
        atom("p", vec![var("X"), var("Y")]),
        vec![
            pos("edge_u32", vec![var("X"), var("M")]),
            pos("edge_u32", vec![var("M"), var("Y")]),
        ],
    );
    // Misgrouped under "q" — but with a body that, IF inference ran,
    // would conflict with r_u32's typing of column 0 of "p".
    let r_sym_misgrouped = rule_with(
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
    // Group "p" gets one well-grouped rule; group "q" gets the
    // misgrouped rule whose head says "p".
    let rules = rules_grouped(vec![("p", vec![r_u32]), ("q", vec![r_sym_misgrouped])]);
    let err =
        evaluate_scc_fixpoint_typed(&rules, &store, &AppearanceOrder, &FixpointConfig::default())
            .expect_err("must reject");
    match err {
        SccFixpointError::RuleHeadPredicateMismatch {
            group_key,
            rule_index,
            observed,
        } => {
            assert_eq!(group_key, "q");
            assert_eq!(rule_index, 0);
            assert_eq!(observed, "p");
        }
        SccFixpointError::RuleEval {
            source: RefEvalError::InferenceConflict { .. },
            ..
        } => {
            panic!(
                "regression: pre-flight did not block inference, \
                 InferenceConflict surfaced before RuleHeadPredicateMismatch"
            );
        }
        other => panic!("expected RuleHeadPredicateMismatch, got {other:?}"),
    }
}

#[test]
fn plan_scc_rules_misgrouped_returns_rule_head_predicate_mismatch() {
    // Symmetric to the typed-evaluator pre-flight: plan_scc_rules
    // must surface PlanError::RuleHeadPredicateMismatch BEFORE
    // running inference. Same fixture shape as the SCC fixpoint
    // version above.
    let r_u32 = rule_with(
        atom("p", vec![var("X"), var("Y")]),
        vec![
            pos("edge_u32", vec![var("X"), var("M")]),
            pos("edge_u32", vec![var("M"), var("Y")]),
        ],
    );
    let r_sym_misgrouped = rule_with(
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
    let rules = rules_grouped(vec![("p", vec![r_u32]), ("q", vec![r_sym_misgrouped])]);
    let err = plan_scc_rules(&rules, &store).expect_err("must reject");
    match err {
        PlanError::RuleHeadPredicateMismatch {
            group_key,
            rule_index,
            observed,
        } => {
            assert_eq!(group_key, "q");
            assert_eq!(rule_index, 0);
            assert_eq!(observed, "p");
        }
        PlanError::InferenceConflict { .. } => {
            panic!(
                "regression: pre-flight did not block inference in plan_scc_rules, \
                 InferenceConflict surfaced before RuleHeadPredicateMismatch"
            );
        }
        other => panic!("expected RuleHeadPredicateMismatch, got {other:?}"),
    }
}
