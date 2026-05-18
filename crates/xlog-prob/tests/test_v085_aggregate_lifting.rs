use std::collections::BTreeSet;

use xlog_prob::pir::{PirNode, PirNodeId};
use xlog_prob::provenance::{extract_from_source, AggregateLiftStatus, Provenance, Value};

fn finite_formula_prob(prov: &Provenance, root: PirNodeId) -> f64 {
    let leaves: Vec<_> = prov.leaf_probs.keys().copied().collect();
    let leaf_count = leaves.len();
    assert!(
        leaf_count <= 20,
        "test formula evaluator is intentionally finite"
    );

    let mut prob = 0.0;
    for mask in 0usize..(1usize << leaf_count) {
        let mut true_leaves = BTreeSet::new();
        let mut world_prob = 1.0;
        for (idx, leaf) in leaves.iter().enumerate() {
            let p = prov.leaf_probs[leaf];
            if (mask & (1usize << idx)) != 0 {
                true_leaves.insert(*leaf);
                world_prob *= p;
            } else {
                world_prob *= 1.0 - p;
            }
        }
        if eval_formula(prov, root, &true_leaves) {
            prob += world_prob;
        }
    }
    prob
}

fn eval_formula(
    prov: &Provenance,
    node: PirNodeId,
    true_leaves: &BTreeSet<xlog_prob::LeafId>,
) -> bool {
    match prov.pir.node(node).expect("valid PIR node") {
        PirNode::Const(v) => *v,
        PirNode::Lit { leaf } => true_leaves.contains(leaf),
        PirNode::NegLit { leaf } => !true_leaves.contains(leaf),
        PirNode::And { children } => children
            .iter()
            .all(|child| eval_formula(prov, *child, true_leaves)),
        PirNode::Or { children } => children
            .iter()
            .any(|child| eval_formula(prov, *child, true_leaves)),
        PirNode::Decision { .. } => panic!("test fixtures use probabilistic facts only"),
    }
}

fn binomial_probability(n: usize, k: usize, p: f64) -> f64 {
    let combinations = (0..k).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64);
    combinations * p.powi(k as i32) * (1.0 - p).powi((n - k) as i32)
}

#[test]
fn count_lift_fires_above_naive_exact_cap_and_matches_finite_oracle() {
    let mut source = String::new();
    for y in 1..=17 {
        source.push_str(&format!("0.5::edge(1, {}).\n", y));
    }
    source.push_str(
        r#"
out_degree(X, count(Y)) :- edge(X, Y).
query(out_degree(1, 8)).
"#,
    );

    let prov = extract_from_source(&source).expect("count aggregate lift extraction");
    let root = prov
        .query_formula("out_degree", &[Value::I64(1), Value::I64(8)])
        .expect("lifted count formula");
    assert!((finite_formula_prob(&prov, root) - binomial_probability(17, 8, 0.5)).abs() < 1e-12);

    let report = prov
        .aggregate_lifting
        .iter()
        .find(|entry| entry.predicate == "out_degree")
        .expect("aggregate lift report");
    assert_eq!(report.status, AggregateLiftStatus::Fired);
    assert_eq!(report.operator, "count");
    assert_eq!(report.uncertain_rows, 17);
    assert_eq!(report.cap, 64);
    assert!(report.naive_outcomes >= 131_072);
    assert!((report.dynamic_programming_states as u128) * 100 < report.naive_outcomes);
}

#[test]
fn numeric_operators_report_exact_enumeration_fallback_with_parity() {
    let source = r#"
0.5::obs(1, 2).
0.25::obs(1, 3).
score_sum(X, sum(Y)) :- obs(X, Y).
score_min(X, min(Y)) :- obs(X, Y).
score_max(X, max(Y)) :- obs(X, Y).
score_lse(X, logsumexp(Y)) :- obs(X, Y).
query(score_sum(1, 5)).
query(score_min(1, 2)).
query(score_max(1, 3)).
"#;

    let prov = extract_from_source(source).expect("numeric aggregate fallback extraction");
    for (predicate, operator) in [
        ("score_sum", "sum"),
        ("score_min", "min"),
        ("score_max", "max"),
        ("score_lse", "logsumexp"),
    ] {
        let report = prov
            .aggregate_lifting
            .iter()
            .find(|entry| entry.predicate == predicate && entry.operator == operator)
            .unwrap_or_else(|| panic!("missing report for {predicate}/{operator}"));
        assert_eq!(report.status, AggregateLiftStatus::FallbackExactEnumeration);
        assert!(report.reason.contains("exact finite outcome enumeration"));
    }

    let sum_five = prov
        .query_formula("score_sum", &[Value::I64(1), Value::I64(5)])
        .expect("sum=5 formula");
    let min_two = prov
        .query_formula("score_min", &[Value::I64(1), Value::I64(2)])
        .expect("min=2 formula");
    let max_three = prov
        .query_formula("score_max", &[Value::I64(1), Value::I64(3)])
        .expect("max=3 formula");

    assert!((finite_formula_prob(&prov, sum_five) - 0.125).abs() < 1e-12);
    assert!((finite_formula_prob(&prov, min_two) - 0.5).abs() < 1e-12);
    assert!((finite_formula_prob(&prov, max_three) - 0.25).abs() < 1e-12);
}

#[test]
fn count_lift_domain_cap_reports_typed_diagnostic() {
    let mut source = String::new();
    for y in 1..=65 {
        source.push_str(&format!("0.5::edge(1, {}).\n", y));
    }
    source.push_str(
        r#"
out_degree(X, count(Y)) :- edge(X, Y).
query(out_degree(1, 65)).
"#,
    );

    let err = extract_from_source(&source).expect_err("count lift cap should reject 65 rows");
    let msg = err.to_string();
    assert!(msg.contains("v0.8.5 agg_lift error"), "msg={}", msg);
    assert!(msg.contains("count lift finite domain cap"), "msg={}", msg);
    assert!(msg.contains("65 uncertain rows > cap 64"), "msg={}", msg);
}

#[test]
fn committed_aggregate_lift_example_extracts_report() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/v085-language/aggregate_lifting/count_lift.xlog");
    let source = std::fs::read_to_string(path).expect("read committed aggregate lift example");
    let prov = extract_from_source(&source).expect("committed aggregate lift example");
    let report = prov
        .aggregate_lifting
        .iter()
        .find(|entry| entry.predicate == "out_degree" && entry.operator == "count")
        .expect("aggregate lift report");
    assert_eq!(report.status, AggregateLiftStatus::Fired);
    assert_eq!(report.uncertain_rows, 17);
}
