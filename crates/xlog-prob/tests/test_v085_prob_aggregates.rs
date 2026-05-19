use std::collections::BTreeSet;

use xlog_prob::pir::{PirNode, PirNodeId};
use xlog_prob::provenance::{extract_from_source, Provenance, Value};

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

#[test]
fn exact_count_aggregate_provenance_matches_finite_oracle() {
    let source = r#"
0.5::edge(1, 2).
0.25::edge(1, 3).
out_degree(X, count(Y)) :- edge(X, Y).
evidence(out_degree(1, 2), true).
query(out_degree(1, 1)).
query(out_degree(1, 2)).
"#;

    let prov = extract_from_source(source).expect("probabilistic count aggregate extraction");

    let count_one = prov
        .query_formula("out_degree", &[Value::I64(1), Value::I64(1)])
        .expect("count=1 aggregate formula");
    let count_two = prov
        .query_formula("out_degree", &[Value::I64(1), Value::I64(2)])
        .expect("count=2 aggregate formula");

    assert!((finite_formula_prob(&prov, count_one) - 0.5).abs() < 1e-12);
    assert!((finite_formula_prob(&prov, count_two) - 0.125).abs() < 1e-12);
    assert!(
        prov.query_formula("out_degree", &[Value::I64(1), Value::I64(0)])
            .is_none(),
        "empty probabilistic groups do not materialize count=0 tuples"
    );
    assert_eq!(prov.evidence.len(), 1);
    assert_eq!(prov.evidence[0].0.predicate, "out_degree");
}

#[test]
fn exact_numeric_aggregate_provenance_matches_finite_oracles() {
    let source = r#"
0.5::obs(1, 2).
0.25::obs(1, 3).
score_sum(X, sum(Y)) :- obs(X, Y).
score_min(X, min(Y)) :- obs(X, Y).
score_max(X, max(Y)) :- obs(X, Y).
score_lse(X, logsumexp(Y)) :- obs(X, Y).
query(score_sum(1, 5)).
query(score_min(1, 2)).
query(score_min(1, 3)).
query(score_max(1, 3)).
"#;

    let prov = extract_from_source(source).expect("probabilistic numeric aggregate extraction");

    let sum_five = prov
        .query_formula("score_sum", &[Value::I64(1), Value::I64(5)])
        .expect("sum=5 formula");
    let min_two = prov
        .query_formula("score_min", &[Value::I64(1), Value::I64(2)])
        .expect("min=2 formula");
    let min_three = prov
        .query_formula("score_min", &[Value::I64(1), Value::I64(3)])
        .expect("min=3 formula");
    let max_three = prov
        .query_formula("score_max", &[Value::I64(1), Value::I64(3)])
        .expect("max=3 formula");
    let lse_both = 3.0_f64 + ((2.0_f64 - 3.0_f64).exp() + 1.0).ln();
    let logsumexp_both = prov
        .query_formula(
            "score_lse",
            &[Value::I64(1), Value::F64(lse_both.to_bits())],
        )
        .expect("logsumexp(2,3) formula");

    assert!((finite_formula_prob(&prov, sum_five) - 0.125).abs() < 1e-12);
    assert!((finite_formula_prob(&prov, min_two) - 0.5).abs() < 1e-12);
    assert!((finite_formula_prob(&prov, min_three) - 0.125).abs() < 1e-12);
    assert!((finite_formula_prob(&prov, max_three) - 0.25).abs() < 1e-12);
    assert!((finite_formula_prob(&prov, logsumexp_both) - 0.125).abs() < 1e-12);
}

#[test]
fn exact_aggregate_domain_cap_reports_typed_diagnostic() {
    let source = r#"
0.5::obs(1, 1).
0.5::obs(1, 2).
0.5::obs(1, 3).
0.5::obs(1, 4).
0.5::obs(1, 5).
0.5::obs(1, 6).
0.5::obs(1, 7).
0.5::obs(1, 8).
0.5::obs(1, 9).
0.5::obs(1, 10).
0.5::obs(1, 11).
0.5::obs(1, 12).
0.5::obs(1, 13).
0.5::obs(1, 14).
0.5::obs(1, 15).
0.5::obs(1, 16).
0.5::obs(1, 17).
score_sum(X, sum(Y)) :- obs(X, Y).
query(score_sum(1, 153)).
"#;

    let err = extract_from_source(source).expect_err("exact aggregate cap should reject 17 rows");
    let msg = err.to_string();
    assert!(msg.contains("v0.8.5 prob_aggregate error"), "msg={}", msg);
    assert!(msg.contains("exact aggregate domain cap"), "msg={}", msg);
    assert!(msg.contains("prob_engine = mc"), "msg={}", msg);
}

#[test]
fn committed_prob_aggregate_example_extracts_provenance() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/v085-language/prob_aggregates/finite_outcomes.xlog");
    let source = std::fs::read_to_string(path).expect("read committed prob aggregate example");
    let prov = extract_from_source(&source).expect("committed prob aggregate example");
    assert!(prov
        .query_formula("out_degree", &[Value::I64(1), Value::I64(2)])
        .is_some());
    assert!(prov
        .query_formula("score_sum", &[Value::I64(1), Value::I64(5)])
        .is_some());
}

#[cfg(feature = "host-io")]
fn has_cuda_device() -> bool {
    xlog_cuda::CudaDevice::new(0).is_ok()
}

#[cfg(feature = "host-io")]
#[test]
fn exact_gpu_count_aggregate_query_matches_finite_oracle() {
    if !has_cuda_device() {
        eprintln!("Skipping: no CUDA device");
        return;
    }

    let source = r#"
0.5::edge(1, 2).
0.25::edge(1, 3).
out_degree(X, count(Y)) :- edge(X, Y).
query(out_degree(1, 2)).
"#;

    let compiled = xlog_prob::exact::ExactDdnnfProgram::compile_source(source)
        .expect("compile exact aggregate");
    let result = compiled.evaluate().expect("evaluate exact aggregate");
    let got = result
        .query_probs
        .iter()
        .find(|q| q.atom.predicate == "out_degree")
        .expect("query result")
        .prob;

    assert!((got - 0.125).abs() < 1e-9, "got={}", got);
}

#[cfg(feature = "host-io")]
#[test]
fn mc_gpu_count_aggregate_query_matches_finite_oracle() {
    if !has_cuda_device() {
        eprintln!("Skipping: no CUDA device");
        return;
    }

    let source = r#"
1.0::edge(1, 2).
1.0::edge(1, 3).
out_degree(X, count(Y)) :- edge(X, Y).
query(out_degree(1, 2)).
"#;

    let program = xlog_prob::mc::McProgram::compile_source(source).expect("compile MC aggregate");
    let mut cfg = xlog_prob::mc::McEvalConfig::default();
    cfg.samples = 32;
    cfg.seed = 85;
    let result = program.evaluate(cfg).expect("evaluate MC aggregate");
    let estimate = result
        .query_estimates
        .iter()
        .find(|q| q.atom.predicate == "out_degree")
        .expect("query estimate");

    assert!(
        estimate.ci_low <= 1.0
            && 1.0 <= estimate.ci_high + 1e-12
            && (estimate.prob - 1.0).abs() < 1e-12,
        "estimate={:?}",
        estimate
    );
}
