use xlog_prob::pir::LeafId;
use xlog_prob::provenance::Provenance;

#[test]
fn leaf_atom_resolves_prob_facts() {
    let src = r#"
        0.3::rain().
        0.7::sprinkler().
        query(rain()).
    "#;
    let prov: Provenance = xlog_prob::provenance::extract_from_source(src).unwrap();

    // Two prob facts → two leaves.
    assert_eq!(prov.leaf_probs.len(), 2);
    assert_eq!(prov.leaf_atoms.len(), 2);

    // LeafId(0) should map to rain, LeafId(1) to sprinkler.
    let atom0 = prov.leaf_atom(LeafId::new(0)).unwrap();
    assert_eq!(atom0.predicate, "rain");
    assert!(atom0.args.is_empty());

    let atom1 = prov.leaf_atom(LeafId::new(1)).unwrap();
    assert_eq!(atom1.predicate, "sprinkler");
    assert!(atom1.args.is_empty());
}

#[test]
fn leaf_atom_returns_none_for_invalid() {
    let src = r#"
        0.5::edge(1, 2).
        query(edge(1, 2)).
    "#;
    let prov: Provenance = xlog_prob::provenance::extract_from_source(src).unwrap();
    assert!(prov.leaf_atom(LeafId::new(9999)).is_none());
}
