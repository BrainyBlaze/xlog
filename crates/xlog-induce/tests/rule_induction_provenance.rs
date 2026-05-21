use xlog_induce::{
    InducedRuleProvenance, InducedRuleRegistry, InductionAlternative, InductionSupportRow,
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

    let mut registry = InducedRuleRegistry::new();
    let rule_id = registry.register(provenance.clone());
    assert_eq!(rule_id, provenance.rule_id);
    assert_eq!(registry.len(), 1);
    assert_eq!(registry.rules()[0].support_rows.len(), 2);
}
