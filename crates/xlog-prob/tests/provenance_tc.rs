use xlog_prob::provenance::Provenance;
use xlog_prob::pir::{LeafId, PirNode};

#[test]
fn test_recursive_provenance_builds_acyclic_pir() {
    let src = r#"
        0.5::edge(1,2).
        0.6::edge(2,3).

        reach(X, Y) :- edge(X, Y).
        reach(X, Z) :- reach(X, Y), edge(Y, Z).

        query(reach(1,3)).
    "#;

    let prov: Provenance = xlog_prob::provenance::extract_from_source(src).unwrap();

    let q = prov
        .query_formula("reach", &[1_i64.into(), 3_i64.into()])
        .unwrap();
    let levels = prov.pir.levelize(&[q]).unwrap();
    assert!(!levels.is_empty());

    let mut stack = vec![q];
    let mut leaves = std::collections::BTreeSet::new();
    while let Some(id) = stack.pop() {
        let node = prov.pir.node(id).unwrap();
        match node {
            PirNode::Const(_) => {}
            PirNode::Lit { leaf } | PirNode::NegLit { leaf } => {
                leaves.insert(*leaf);
            }
            PirNode::And { children } | PirNode::Or { children } => {
                stack.extend(children.iter().copied());
            }
            PirNode::Decision {
                child_false,
                child_true,
                ..
            } => {
                stack.push(*child_false);
                stack.push(*child_true);
            }
        }
    }

    assert_eq!(
        leaves,
        std::collections::BTreeSet::from([LeafId::new(0), LeafId::new(1)])
    );
}
