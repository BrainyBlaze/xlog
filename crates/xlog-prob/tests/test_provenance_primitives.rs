use xlog_prob::{ChoiceSource, ChoiceVarId, GroundAtom, LeafId, Provenance, Value};

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

#[test]
fn choice_source_resolves_ad_metadata() {
    // 2 explicit heads → 1 choice variable (m-1 = 1).
    let src = r#"
        0.3::color(red) ; 0.7::color(blue).
        query(color(red)).
    "#;
    let prov: Provenance = xlog_prob::provenance::extract_from_source(src).unwrap();

    assert_eq!(prov.choice_sources.len(), prov.choice_probs.len());

    let cs = prov.choice_source(ChoiceVarId::new(0)).unwrap();

    // choices: explicit heads only (no implicit none branch).
    assert_eq!(cs.choices.len(), 2);
    assert_eq!(cs.choices[0].0.predicate, "color");
    assert_eq!(cs.choices[0].1, 0.3);
    assert_eq!(cs.choices[1].0.predicate, "color");
    assert_eq!(cs.choices[1].1, 0.7);

    // choice_index = 0 (first and only decision stage).
    assert_eq!(cs.choice_index, 0);

    // source_id = None in v1.
    assert!(cs.source_id.is_none());
}

#[test]
fn choice_source_returns_none_for_invalid() {
    let src = r#"
        0.3::color(red) ; 0.7::color(blue).
        query(color(red)).
    "#;
    let prov: Provenance = xlog_prob::provenance::extract_from_source(src).unwrap();
    assert!(prov.choice_source(ChoiceVarId::new(9999)).is_none());
}

#[test]
fn atoms_with_formulas_returns_all_formula_atoms() {
    let src = r#"
        wet() :- rain().
        wet() :- sprinkler().
        0.3::rain().
        0.7::sprinkler().
        query(wet()).
    "#;
    let prov: Provenance = xlog_prob::provenance::extract_from_source(src).unwrap();

    let formula_atoms: Vec<_> = prov.atoms_with_formulas().collect();

    // Should contain: rain, sprinkler (prob facts) + wet (derived).
    assert_eq!(formula_atoms.len(), 3);

    // Every atom returned should have a valid PirNodeId.
    for (_atom, node_id) in &formula_atoms {
        assert!(prov.pir.node(*node_id).is_some());
    }

    // Check that specific atoms are present.
    let predicates: Vec<&str> = formula_atoms.iter().map(|(a, _)| a.predicate.as_str()).collect();
    assert!(predicates.contains(&"rain"));
    assert!(predicates.contains(&"sprinkler"));
    assert!(predicates.contains(&"wet"));

    // Verify consistency with query_formula.
    for (atom, node_id) in &formula_atoms {
        let looked_up = prov.query_formula(&atom.predicate, &atom.args).unwrap();
        assert_eq!(looked_up, *node_id);
    }
}

#[test]
fn ground_atom_new_public() {
    let atom = GroundAtom::new("reach", vec![Value::I64(1), Value::I64(3)]);
    assert_eq!(atom.predicate, "reach");
    assert_eq!(atom.args, vec![Value::I64(1), Value::I64(3)]);
}
