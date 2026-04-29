// crates/xlog-logic/tests/test_hypergraph_planner.rs
//! Tests for the v0.6.2 hypergraph planner foundation (PR 1).
//!
//! Coverage:
//!   * IR structural invariants — vertex/hyperedge consistency,
//!     anonymous-wildcard handling, vertex deduplication.
//!   * Eligibility boundary positives + negatives across the full
//!     boundary list.
//!   * Variable-order determinism.
//!   * Snapshot-style assertions on explain output for canonical
//!     plan shapes (eligible triangle, ineligible-by-aggregation,
//!     ineligible-by-negation, ineligible-by-keys-over-4,
//!     ineligible-by-arity-1).
//!
//! All tests are pure-Rust (no parser, no GPU) — they construct
//! [`xlog_logic::ast::Rule`] values directly to keep the surface
//! under test minimal and the failure messages precise.

use xlog_logic::ast::{AggExpr, AggOp, Atom, BodyLiteral, CompOp, Comparison, IsExpr, Rule, Term};
use xlog_logic::hypergraph::{
    analyze, explain, AppearanceOrder, Boundary, Eligibility, HypergraphRule, VariableOrder,
};

// ---------------------------------------------------------------
// Test helpers — direct AST construction without going through the
// parser, so failures locate cleanly to the IR / analyzer / explain.
// ---------------------------------------------------------------

fn var(name: &str) -> Term {
    Term::Variable(name.to_string())
}

fn anon() -> Term {
    Term::Anonymous
}

fn int(n: i64) -> Term {
    Term::Integer(n)
}

fn atom(predicate: &str, terms: Vec<Term>) -> Atom {
    Atom {
        predicate: predicate.to_string(),
        terms,
    }
}

fn rule_with(head: Atom, body: Vec<BodyLiteral>) -> Rule {
    Rule { head, body }
}

fn pos(predicate: &str, terms: Vec<Term>) -> BodyLiteral {
    BodyLiteral::Positive(atom(predicate, terms))
}

fn neg(predicate: &str, terms: Vec<Term>) -> BodyLiteral {
    BodyLiteral::Negated(atom(predicate, terms))
}

fn cmp(left: Term, op: CompOp, right: Term) -> BodyLiteral {
    BodyLiteral::Comparison(Comparison { left, op, right })
}

// ---------------------------------------------------------------
// IR structural invariants
// ---------------------------------------------------------------

#[test]
fn ir_dedups_repeated_variables_into_a_single_vertex() {
    // p(X, Y) :- e(X, Z), e(Z, Y), e(X, Y).
    let r = rule_with(
        atom("p", vec![var("X"), var("Y")]),
        vec![
            pos("e", vec![var("X"), var("Z")]),
            pos("e", vec![var("Z"), var("Y")]),
            pos("e", vec![var("X"), var("Y")]),
        ],
    );
    let hg = HypergraphRule::from_rule(&r);
    // Three distinct variables across the body.
    assert_eq!(hg.vertex_count(), 3);
    let names: Vec<&str> = hg.vertices.iter().map(|v| v.name.as_str()).collect();
    // First-appearance order across the body.
    assert_eq!(names, vec!["X", "Z", "Y"]);
    // Each hyperedge has full vertex assignment.
    assert_eq!(hg.hyperedges.len(), 3);
    for edge in &hg.hyperedges {
        assert!(edge.vertex_positions.iter().all(|p| p.is_some()));
    }
}

#[test]
fn ir_treats_anonymous_wildcards_as_non_vertices() {
    // p(X) :- e(X, _), e(_, X).
    let r = rule_with(
        atom("p", vec![var("X")]),
        vec![
            pos("e", vec![var("X"), anon()]),
            pos("e", vec![anon(), var("X")]),
        ],
    );
    let hg = HypergraphRule::from_rule(&r);
    assert_eq!(hg.vertex_count(), 1);
    assert_eq!(hg.vertices[0].name, "X");
    // First atom: position 0 is X, position 1 is anonymous → None.
    assert!(hg.hyperedges[0].vertex_positions[0].is_some());
    assert!(hg.hyperedges[0].vertex_positions[1].is_none());
    // Second atom: mirror.
    assert!(hg.hyperedges[1].vertex_positions[0].is_none());
    assert!(hg.hyperedges[1].vertex_positions[1].is_some());
}

#[test]
fn ir_treats_constants_as_non_vertices() {
    // p(X) :- e(X, 42).
    let r = rule_with(
        atom("p", vec![var("X")]),
        vec![pos("e", vec![var("X"), int(42)])],
    );
    let hg = HypergraphRule::from_rule(&r);
    assert_eq!(hg.vertex_count(), 1);
    assert!(hg.hyperedges[0].vertex_positions[0].is_some());
    assert!(hg.hyperedges[0].vertex_positions[1].is_none());
}

#[test]
fn ir_records_comparisons_negation_and_isexpr_flags() {
    // p(X) :- e(X, Y), Y < 10, not f(X), Z is X + 1.
    use xlog_logic::ast::ArithExpr;
    let r = rule_with(
        atom("p", vec![var("X")]),
        vec![
            pos("e", vec![var("X"), var("Y")]),
            cmp(var("Y"), CompOp::Lt, int(10)),
            neg("f", vec![var("X")]),
            BodyLiteral::IsExpr(IsExpr {
                target: "Z".to_string(),
                expr: ArithExpr::Add(
                    Box::new(ArithExpr::Variable("X".to_string())),
                    Box::new(ArithExpr::Integer(1)),
                ),
            }),
        ],
    );
    let hg = HypergraphRule::from_rule(&r);
    assert_eq!(hg.comparison_count, 1);
    assert!(hg.has_negation);
    assert!(hg.has_is_expr);
    // Hyperedges count covers ONLY positive atoms.
    assert_eq!(hg.hyperedges.len(), 1);
}

#[test]
fn ir_marks_ground_facts() {
    // edge(1, 2).
    let r = rule_with(atom("edge", vec![int(1), int(2)]), vec![]);
    let hg = HypergraphRule::from_rule(&r);
    assert!(hg.is_fact);
    assert_eq!(hg.hyperedge_count(), 0);
    assert_eq!(hg.vertex_count(), 0);
}

// ---------------------------------------------------------------
// Eligibility boundaries — positives
// ---------------------------------------------------------------

#[test]
fn eligible_triangle_query_is_eligible() {
    // tri(X, Y, Z) :- e(X, Y), e(Y, Z), e(Z, X).
    let r = rule_with(
        atom("tri", vec![var("X"), var("Y"), var("Z")]),
        vec![
            pos("e", vec![var("X"), var("Y")]),
            pos("e", vec![var("Y"), var("Z")]),
            pos("e", vec![var("Z"), var("X")]),
        ],
    );
    let hg = HypergraphRule::from_rule(&r);
    let v = analyze(&hg);
    assert_eq!(v, Eligibility::Eligible);
    assert!(v.is_eligible());
    assert!(v.boundaries().is_empty());
}

#[test]
fn eligible_two_atom_rule_is_eligible_per_pr_doc() {
    // Per Eligibility doc: a 2-atom rule is Eligible because both
    // multiway and binary lowerings are valid; the planner chooses.
    // p(X, Z) :- e(X, Y), e(Y, Z).
    let r = rule_with(
        atom("p", vec![var("X"), var("Z")]),
        vec![
            pos("e", vec![var("X"), var("Y")]),
            pos("e", vec![var("Y"), var("Z")]),
        ],
    );
    let v = analyze(&HypergraphRule::from_rule(&r));
    assert_eq!(v, Eligibility::Eligible);
}

#[test]
fn eligible_rule_with_filters_stays_eligible() {
    // Comparisons are filters, not boundaries.
    // p(X, Z) :- e(X, Y), e(Y, Z), Y < 10.
    let r = rule_with(
        atom("p", vec![var("X"), var("Z")]),
        vec![
            pos("e", vec![var("X"), var("Y")]),
            pos("e", vec![var("Y"), var("Z")]),
            cmp(var("Y"), CompOp::Lt, int(10)),
        ],
    );
    let v = analyze(&HypergraphRule::from_rule(&r));
    assert_eq!(v, Eligibility::Eligible);
}

// ---------------------------------------------------------------
// Eligibility boundaries — negatives, one per Boundary variant
// (UnsupportedKeyType excluded — not produced by analyze() in PR 1).
// ---------------------------------------------------------------

#[test]
fn ground_fact_is_ineligible_with_groundfact_boundary() {
    let r = rule_with(atom("edge", vec![int(1), int(2)]), vec![]);
    let v = analyze(&HypergraphRule::from_rule(&r));
    let bs = v.boundaries();
    assert!(bs.contains(&Boundary::GroundFact));
    // Also has InsufficientPositiveAtoms? — no, the analyzer skips
    // that check for ground facts to avoid double-counting.
    assert!(!bs
        .iter()
        .any(|b| matches!(b, Boundary::InsufficientPositiveAtoms { .. })));
}

#[test]
fn head_aggregation_triggers_headaggregation_boundary() {
    // p(C) :- e(X, Y), C = count(X). — head has count(X).
    let head = Atom {
        predicate: "p".to_string(),
        terms: vec![Term::Aggregate(AggExpr {
            op: AggOp::Count,
            variable: "X".to_string(),
        })],
    };
    let r = rule_with(
        head,
        vec![
            pos("e", vec![var("X"), var("Y")]),
            pos("e", vec![var("Y"), var("Z")]),
        ],
    );
    let v = analyze(&HypergraphRule::from_rule(&r));
    assert!(v.boundaries().contains(&Boundary::HeadAggregation));
}

#[test]
fn body_negation_triggers_bodynegation_boundary() {
    // p(X, Y) :- e(X, Y), not f(X, Y).
    let r = rule_with(
        atom("p", vec![var("X"), var("Y")]),
        vec![
            pos("e", vec![var("X"), var("Y")]),
            neg("f", vec![var("X"), var("Y")]),
        ],
    );
    let v = analyze(&HypergraphRule::from_rule(&r));
    assert!(v.boundaries().contains(&Boundary::BodyNegation));
}

#[test]
fn body_is_expr_triggers_bodyisexpr_boundary() {
    use xlog_logic::ast::ArithExpr;
    // p(X, Z) :- e(X, Y), Z is X + 1, q(Z).
    let r = rule_with(
        atom("p", vec![var("X"), var("Z")]),
        vec![
            pos("e", vec![var("X"), var("Y")]),
            BodyLiteral::IsExpr(IsExpr {
                target: "Z".to_string(),
                expr: ArithExpr::Add(
                    Box::new(ArithExpr::Variable("X".to_string())),
                    Box::new(ArithExpr::Integer(1)),
                ),
            }),
            pos("q", vec![var("Z")]),
        ],
    );
    let v = analyze(&HypergraphRule::from_rule(&r));
    assert!(v.boundaries().contains(&Boundary::BodyIsExpr));
}

#[test]
fn single_atom_body_triggers_insufficientpositiveatoms_boundary() {
    // p(X) :- e(X, Y).
    let r = rule_with(
        atom("p", vec![var("X")]),
        vec![pos("e", vec![var("X"), var("Y")])],
    );
    let v = analyze(&HypergraphRule::from_rule(&r));
    assert!(v
        .boundaries()
        .contains(&Boundary::InsufficientPositiveAtoms { positive_count: 1 }));
}

#[test]
fn comparison_only_body_triggers_insufficientpositiveatoms_boundary() {
    // q :- 1 < 2. — unusual but legal AST: zero positive atoms.
    let r = rule_with(atom("q", vec![]), vec![cmp(int(1), CompOp::Lt, int(2))]);
    let v = analyze(&HypergraphRule::from_rule(&r));
    assert!(v
        .boundaries()
        .contains(&Boundary::InsufficientPositiveAtoms { positive_count: 0 }));
}

#[test]
fn five_join_keys_trigger_joinkeysexceedbinaryfallbacklimit_boundary() {
    // p(A, B, C, D, E) :- r1(A, B), r2(B, C), r3(C, D), r4(D, E), r5(E, A).
    // Each variable appears in exactly two atoms → all 5 are join keys.
    let r = rule_with(
        atom("p", vec![var("A"), var("B"), var("C"), var("D"), var("E")]),
        vec![
            pos("r1", vec![var("A"), var("B")]),
            pos("r2", vec![var("B"), var("C")]),
            pos("r3", vec![var("C"), var("D")]),
            pos("r4", vec![var("D"), var("E")]),
            pos("r5", vec![var("E"), var("A")]),
        ],
    );
    let v = analyze(&HypergraphRule::from_rule(&r));
    let bs = v.boundaries();
    assert!(bs.iter().any(|b| matches!(
        b,
        Boundary::JoinKeysExceedBinaryFallbackLimit { count: 5, limit: 4 }
    )));
}

#[test]
fn four_join_keys_stay_eligible() {
    // p(A, B, C, D) :- r1(A, B), r2(B, C), r3(C, D), r4(D, A).
    // 4 distinct join keys = at the limit, not over → Eligible.
    let r = rule_with(
        atom("p", vec![var("A"), var("B"), var("C"), var("D")]),
        vec![
            pos("r1", vec![var("A"), var("B")]),
            pos("r2", vec![var("B"), var("C")]),
            pos("r3", vec![var("C"), var("D")]),
            pos("r4", vec![var("D"), var("A")]),
        ],
    );
    let v = analyze(&HypergraphRule::from_rule(&r));
    assert_eq!(v, Eligibility::Eligible);
}

#[test]
fn self_join_within_single_atom_counts_as_one_vertex_occurrence() {
    // p(X, Y) :- r(X, X), s(X, Y).
    // Vertex X appears twice within r's edge, but a self-join inside
    // a single atom is NOT a multi-atom join key. The join-key count
    // for this rule must be 1 (X across r and s), not 2.
    let r = rule_with(
        atom("p", vec![var("X"), var("Y")]),
        vec![
            pos("r", vec![var("X"), var("X")]),
            pos("s", vec![var("X"), var("Y")]),
        ],
    );
    let hg = HypergraphRule::from_rule(&r);
    // Hyperedge::vertices() must dedup within an edge.
    let r_edge = &hg.hyperedges[0];
    assert_eq!(
        r_edge.vertices().len(),
        1,
        "self-join within an atom must not double-count the vertex"
    );
    // And the rule analyzes as Eligible: only X is a join key (Y is
    // projection-only). Locks the eligibility invariant against a
    // future "simplification" that drops the dedup.
    let v = analyze(&hg);
    assert_eq!(v, Eligibility::Eligible);
}

#[test]
fn projection_only_variables_do_not_count_as_join_keys() {
    // p(A, B, C, D, E) :- r1(A, B, C), r2(C, D, E).
    // C appears in both → join key. A, B, D, E each appear in only
    // one atom → not join keys. Total join keys = 1 → eligible
    // even though there are 5 distinct variables.
    let r = rule_with(
        atom("p", vec![var("A"), var("B"), var("C"), var("D"), var("E")]),
        vec![
            pos("r1", vec![var("A"), var("B"), var("C")]),
            pos("r2", vec![var("C"), var("D"), var("E")]),
        ],
    );
    let hg = HypergraphRule::from_rule(&r);
    assert_eq!(hg.vertex_count(), 5);
    let v = analyze(&hg);
    assert_eq!(v, Eligibility::Eligible);
}

#[test]
fn multiple_boundaries_are_reported_independently() {
    // p :- not e(X, Y, Z, W, V), W is X + 1.
    // Negation + IsExpr + zero positive atoms — three independent
    // boundaries; the analyzer reports each one.
    use xlog_logic::ast::ArithExpr;
    let r = rule_with(
        atom("p", vec![]),
        vec![
            neg("e", vec![var("X"), var("Y"), var("Z"), var("W"), var("V")]),
            BodyLiteral::IsExpr(IsExpr {
                target: "W".to_string(),
                expr: ArithExpr::Add(
                    Box::new(ArithExpr::Variable("X".to_string())),
                    Box::new(ArithExpr::Integer(1)),
                ),
            }),
        ],
    );
    let v = analyze(&HypergraphRule::from_rule(&r));
    let bs = v.boundaries();
    assert!(bs.contains(&Boundary::BodyNegation));
    assert!(bs.contains(&Boundary::BodyIsExpr));
    assert!(bs
        .iter()
        .any(|b| matches!(b, Boundary::InsufficientPositiveAtoms { .. })));
}

// ---------------------------------------------------------------
// Variable-order determinism
// ---------------------------------------------------------------

#[test]
fn appearance_order_is_deterministic_and_complete() {
    let r = rule_with(
        atom("tri", vec![var("X"), var("Y"), var("Z")]),
        vec![
            pos("e", vec![var("X"), var("Y")]),
            pos("e", vec![var("Y"), var("Z")]),
            pos("e", vec![var("Z"), var("X")]),
        ],
    );
    let hg = HypergraphRule::from_rule(&r);
    let vo = AppearanceOrder;
    let order_a = vo.order(&hg);
    let order_b = vo.order(&hg);
    assert_eq!(order_a, order_b, "deterministic across calls");
    assert_eq!(order_a.len(), hg.vertex_count(), "covers every vertex");
    let names: Vec<&str> = order_a
        .iter()
        .map(|v| hg.vertex(*v).name.as_str())
        .collect();
    assert_eq!(names, vec!["X", "Y", "Z"]);
}

#[test]
fn appearance_order_name_is_stable() {
    assert_eq!(AppearanceOrder.name(), "appearance");
}

// ---------------------------------------------------------------
// Explain output snapshot tests
// ---------------------------------------------------------------

#[test]
fn explain_eligible_triangle_snapshot() {
    let r = rule_with(
        atom("tri", vec![var("X"), var("Y"), var("Z")]),
        vec![
            pos("e", vec![var("X"), var("Y")]),
            pos("e", vec![var("Y"), var("Z")]),
            pos("e", vec![var("Z"), var("X")]),
        ],
    );
    let hg = HypergraphRule::from_rule(&r);
    let v = analyze(&hg);
    let s = explain(&hg, &v, &AppearanceOrder);
    let expected = "\
rule head=tri
  vertices: [X Y Z]
  hyperedges:
    e(?X, ?Y)
    e(?Y, ?Z)
    e(?Z, ?X)
  filters: 0
  eligibility: Eligible
  variable-order(appearance): [X Y Z]
";
    assert_eq!(s, expected);
}

#[test]
fn explain_ineligible_aggregation_snapshot() {
    let head = Atom {
        predicate: "p".to_string(),
        terms: vec![Term::Aggregate(AggExpr {
            op: AggOp::Count,
            variable: "X".to_string(),
        })],
    };
    let r = rule_with(
        head,
        vec![
            pos("e", vec![var("X"), var("Y")]),
            pos("f", vec![var("Y"), var("Z")]),
        ],
    );
    let hg = HypergraphRule::from_rule(&r);
    let v = analyze(&hg);
    let s = explain(&hg, &v, &AppearanceOrder);
    let expected = "\
rule head=p
  vertices: [X Y Z]
  hyperedges:
    e(?X, ?Y)
    f(?Y, ?Z)
  filters: 0
  eligibility: Ineligible
    HeadAggregation
  variable-order(appearance): [X Y Z]
";
    assert_eq!(s, expected);
}

#[test]
fn explain_ineligible_negation_snapshot() {
    let r = rule_with(
        atom("p", vec![var("X"), var("Y")]),
        vec![
            pos("e", vec![var("X"), var("Y")]),
            neg("f", vec![var("X"), var("Y")]),
        ],
    );
    let hg = HypergraphRule::from_rule(&r);
    let v = analyze(&hg);
    let s = explain(&hg, &v, &AppearanceOrder);
    let expected = "\
rule head=p
  vertices: [X Y]
  hyperedges:
    e(?X, ?Y)
  filters: 0
  eligibility: Ineligible
    BodyNegation
    InsufficientPositiveAtoms(positive_count=1)
  variable-order(appearance): [X Y]
";
    assert_eq!(s, expected);
}

#[test]
fn explain_ineligible_keys_over_4_snapshot() {
    let r = rule_with(
        atom("p", vec![var("A"), var("B"), var("C"), var("D"), var("E")]),
        vec![
            pos("r1", vec![var("A"), var("B")]),
            pos("r2", vec![var("B"), var("C")]),
            pos("r3", vec![var("C"), var("D")]),
            pos("r4", vec![var("D"), var("E")]),
            pos("r5", vec![var("E"), var("A")]),
        ],
    );
    let hg = HypergraphRule::from_rule(&r);
    let v = analyze(&hg);
    let s = explain(&hg, &v, &AppearanceOrder);
    let expected = "\
rule head=p
  vertices: [A B C D E]
  hyperedges:
    r1(?A, ?B)
    r2(?B, ?C)
    r3(?C, ?D)
    r4(?D, ?E)
    r5(?E, ?A)
  filters: 0
  eligibility: Ineligible
    JoinKeysExceedBinaryFallbackLimit(count=5, limit=4)
  variable-order(appearance): [A B C D E]
";
    assert_eq!(s, expected);
}

#[test]
fn explain_single_atom_snapshot() {
    let r = rule_with(
        atom("p", vec![var("X")]),
        vec![pos("e", vec![var("X"), var("Y")])],
    );
    let hg = HypergraphRule::from_rule(&r);
    let v = analyze(&hg);
    let s = explain(&hg, &v, &AppearanceOrder);
    let expected = "\
rule head=p
  vertices: [X Y]
  hyperedges:
    e(?X, ?Y)
  filters: 0
  eligibility: Ineligible
    InsufficientPositiveAtoms(positive_count=1)
  variable-order(appearance): [X Y]
";
    assert_eq!(s, expected);
}

#[test]
fn explain_multi_boundary_snapshot_locks_emission_order() {
    // Same construction as multiple_boundaries_are_reported_independently,
    // but rendered through explain so the boundary EMISSION ORDER is
    // pinned. Order is: BodyNegation, BodyIsExpr, InsufficientPositiveAtoms.
    use xlog_logic::ast::ArithExpr;
    let r = rule_with(
        atom("p", vec![]),
        vec![
            neg("e", vec![var("X"), var("Y"), var("Z"), var("W"), var("V")]),
            BodyLiteral::IsExpr(IsExpr {
                target: "W".to_string(),
                expr: ArithExpr::Add(
                    Box::new(ArithExpr::Variable("X".to_string())),
                    Box::new(ArithExpr::Integer(1)),
                ),
            }),
        ],
    );
    let hg = HypergraphRule::from_rule(&r);
    let v = analyze(&hg);
    let s = explain(&hg, &v, &AppearanceOrder);
    let expected = "\
rule head=p
  vertices: []
  hyperedges: <none>
  filters: 0
  eligibility: Ineligible
    BodyNegation
    BodyIsExpr
    InsufficientPositiveAtoms(positive_count=0)
  variable-order(appearance): []
";
    assert_eq!(s, expected);
}

#[test]
fn explain_ground_fact_snapshot() {
    // Pins the GroundFact format-boundary arm.
    let r = rule_with(atom("edge", vec![int(1), int(2)]), vec![]);
    let hg = HypergraphRule::from_rule(&r);
    let v = analyze(&hg);
    let s = explain(&hg, &v, &AppearanceOrder);
    let expected = "\
rule head=edge
  vertices: []
  hyperedges: <none>
  filters: 0
  eligibility: Ineligible
    GroundFact
  variable-order(appearance): []
";
    assert_eq!(s, expected);
}

#[test]
fn explain_body_is_expr_snapshot() {
    // Pins the BodyIsExpr format-boundary arm without negation
    // (the multi-boundary snapshot also covers BodyIsExpr but
    // bundled with other boundaries).
    use xlog_logic::ast::ArithExpr;
    let r = rule_with(
        atom("p", vec![var("X")]),
        vec![
            pos("e", vec![var("X"), var("Y")]),
            BodyLiteral::IsExpr(IsExpr {
                target: "Z".to_string(),
                expr: ArithExpr::Add(
                    Box::new(ArithExpr::Variable("X".to_string())),
                    Box::new(ArithExpr::Integer(1)),
                ),
            }),
        ],
    );
    let hg = HypergraphRule::from_rule(&r);
    let v = analyze(&hg);
    let s = explain(&hg, &v, &AppearanceOrder);
    let expected = "\
rule head=p
  vertices: [X Y]
  hyperedges:
    e(?X, ?Y)
  filters: 0
  eligibility: Ineligible
    BodyIsExpr
    InsufficientPositiveAtoms(positive_count=1)
  variable-order(appearance): [X Y]
";
    assert_eq!(s, expected);
}

#[test]
fn explain_with_filters_and_anonymous_wildcards_snapshot() {
    let r = rule_with(
        atom("p", vec![var("X"), var("Z")]),
        vec![
            pos("e", vec![var("X"), var("Y"), anon()]),
            pos("e", vec![var("Y"), var("Z"), anon()]),
            cmp(var("Y"), CompOp::Lt, int(10)),
        ],
    );
    let hg = HypergraphRule::from_rule(&r);
    let v = analyze(&hg);
    let s = explain(&hg, &v, &AppearanceOrder);
    let expected = "\
rule head=p
  vertices: [X Y Z]
  hyperedges:
    e(?X, ?Y, _)
    e(?Y, ?Z, _)
  filters: 1
  eligibility: Eligible
  variable-order(appearance): [X Y Z]
";
    assert_eq!(s, expected);
}
