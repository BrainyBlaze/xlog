use xlog_induce::{
    InducedRuleProvenance, InductionAlternative, InductionProvenanceRegistry, InductionSupportRow,
    RuleSourceKind,
};

#[test]
fn generated_rules_expose_search_support_rejections_and_falsifications() {
    let provenance = InducedRuleProvenance::new(
        "reach(X,Y) :- edge(X,Z), edge(Z,Y).",
        24,
        vec!["edge/2".to_string(), "reach/2".to_string()],
    )
    .with_support_rows(vec![
        InductionSupportRow::new("edge", 0, "row-a"),
        InductionSupportRow::new("edge", 1, "row-b"),
    ])
    .with_rejected_alternatives(vec![InductionAlternative::new(
        "reach(X,Y) :- edge(Y,X).",
        2,
        5,
    )])
    .with_falsification_count(3);

    assert_eq!(provenance.source_kind, RuleSourceKind::Generated);
    assert_eq!(provenance.search_space_size, 24);
    assert_eq!(provenance.predicate_inventory, ["edge/2", "reach/2"]);
    assert_eq!(provenance.support_rows.len(), 2);
    assert_eq!(provenance.rejected_alternatives.len(), 1);
    assert_eq!(provenance.falsification_count, 3);
    assert!(!provenance.rule_id.is_empty());
    assert!(!provenance.generation_trace_hash.is_empty());
}

#[test]
fn generated_rule_provenance_can_be_registered_for_introspection() {
    let provenance = InducedRuleProvenance::new(
        "falls(X) :- released(X), unsupported(X).",
        7,
        vec!["released/1".to_string(), "unsupported/1".to_string()],
    );
    let rule_id = provenance.rule_id.clone();

    let mut registry = InductionProvenanceRegistry::new();
    registry.register(provenance);

    assert_eq!(registry.rules().len(), 1);
    assert_eq!(registry.rules()[0].rule_id, rule_id);
}
