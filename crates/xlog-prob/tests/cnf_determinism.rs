use xlog_prob::cnf::encode_cnf;
use xlog_prob::pir::{LeafId, PirGraph};

/// Canonicalize clauses for comparison: sort literals within each clause,
/// then sort the clause list. This ignores emission order while preserving
/// logical content.
fn canonicalize(clauses: &[Vec<i32>]) -> Vec<Vec<i32>> {
    let mut out: Vec<Vec<i32>> = clauses
        .iter()
        .map(|c| {
            let mut sorted = c.clone();
            sorted.sort();
            sorted
        })
        .collect();
    out.sort();
    out
}

/// Build two PIR graphs with identical structure but different node ID assignments.
/// Verify that encode_cnf() produces byte-identical CNF output.
#[test]
fn encode_cnf_is_independent_of_pir_node_id_order() {
    // PIR A: lit(0), neglit(1), and(lit, neglit) — IDs 0,1,2
    let mut pir_a = PirGraph::new();
    let a_leaf = pir_a.lit(LeafId::new(0));
    let a_neg = pir_a.neg_lit(LeafId::new(1));
    let a_root = pir_a.and(vec![a_leaf, a_neg]);

    // PIR B: same structure, but waste IDs 0,1 with dummy consts
    let mut pir_b = PirGraph::new();
    let _ = pir_b.const_true(); // ID 0 (dummy)
    let _ = pir_b.const_false(); // ID 1 (dummy)
    let b_leaf = pir_b.lit(LeafId::new(0)); // ID 2
    let b_neg = pir_b.neg_lit(LeafId::new(1)); // ID 3
    let b_root = pir_b.and(vec![b_leaf, b_neg]); // ID 4

    let cnf_a = encode_cnf(&pir_a, &[a_root]).unwrap();
    let cnf_b = encode_cnf(&pir_b, &[b_root]).unwrap();

    assert_eq!(
        cnf_a.cnf.num_vars(),
        cnf_b.cnf.num_vars(),
        "num_vars mismatch: PIR A has {} but PIR B has {}",
        cnf_a.cnf.num_vars(),
        cnf_b.cnf.num_vars()
    );
    assert_eq!(
        cnf_a.cnf.clauses(),
        cnf_b.cnf.clauses(),
        "clauses differ between PIR graphs with different ID assignments"
    );
}

/// Reversed child ordering in And/Or produces logically equivalent CNF.
/// Clause emission order may differ, but canonicalized clauses match.
#[test]
fn encode_cnf_is_independent_of_child_order_in_and_or() {
    let mut pir_a = PirGraph::new();
    let a0 = pir_a.lit(LeafId::new(0));
    let a1 = pir_a.lit(LeafId::new(1));
    let a_root = pir_a.and(vec![a0, a1]);

    let mut pir_b = PirGraph::new();
    let b1 = pir_b.lit(LeafId::new(1));
    let b0 = pir_b.lit(LeafId::new(0));
    let b_root = pir_b.and(vec![b1, b0]); // reversed order

    let cnf_a = encode_cnf(&pir_a, &[a_root]).unwrap();
    let cnf_b = encode_cnf(&pir_b, &[b_root]).unwrap();

    assert_eq!(
        cnf_a.cnf.num_vars(),
        cnf_b.cnf.num_vars(),
        "num_vars mismatch with reversed child order"
    );
    assert_eq!(
        canonicalize(cnf_a.cnf.clauses()),
        canonicalize(cnf_b.cnf.clauses()),
        "canonicalized clauses differ with reversed child order in And"
    );
}

/// Or with reversed children should also produce logically equivalent CNF.
#[test]
fn encode_cnf_or_is_independent_of_child_order() {
    let mut pir_a = PirGraph::new();
    let a0 = pir_a.lit(LeafId::new(0));
    let a1 = pir_a.lit(LeafId::new(1));
    let a_root = pir_a.or(vec![a0, a1]);

    let mut pir_b = PirGraph::new();
    let b1 = pir_b.lit(LeafId::new(1));
    let b0 = pir_b.lit(LeafId::new(0));
    let b_root = pir_b.or(vec![b1, b0]);

    let cnf_a = encode_cnf(&pir_a, &[a_root]).unwrap();
    let cnf_b = encode_cnf(&pir_b, &[b_root]).unwrap();

    assert_eq!(cnf_a.cnf.num_vars(), cnf_b.cnf.num_vars());
    assert_eq!(
        canonicalize(cnf_a.cnf.clauses()),
        canonicalize(cnf_b.cnf.clauses()),
    );
}

/// Deeper graph: nested And(Or(lit, lit), lit) with different ID offsets.
#[test]
fn encode_cnf_nested_graph_with_different_id_offsets() {
    // PIR A: no wasted IDs
    let mut pir_a = PirGraph::new();
    let a0 = pir_a.lit(LeafId::new(0));
    let a1 = pir_a.lit(LeafId::new(1));
    let a2 = pir_a.lit(LeafId::new(2));
    let a_or = pir_a.or(vec![a0, a1]);
    let a_root = pir_a.and(vec![a_or, a2]);

    // PIR B: waste 3 IDs with dummy nodes
    let mut pir_b = PirGraph::new();
    let _ = pir_b.const_true();
    let _ = pir_b.const_false();
    let _ = pir_b.const_true();
    let b0 = pir_b.lit(LeafId::new(0));
    let b1 = pir_b.lit(LeafId::new(1));
    let b2 = pir_b.lit(LeafId::new(2));
    let b_or = pir_b.or(vec![b0, b1]);
    let b_root = pir_b.and(vec![b_or, b2]);

    let cnf_a = encode_cnf(&pir_a, &[a_root]).unwrap();
    let cnf_b = encode_cnf(&pir_b, &[b_root]).unwrap();

    assert_eq!(cnf_a.cnf.num_vars(), cnf_b.cnf.num_vars());
    assert_eq!(cnf_a.cnf.clauses(), cnf_b.cnf.clauses());
}
