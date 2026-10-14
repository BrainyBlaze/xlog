use std::fs;

use tempfile::TempDir;
use xlog_logic::compile::load_modules;
use xlog_logic::resolver::{
    AggregateOperator, EpistemicOperator, ExecutableBodyLiteral, ExecutableFunctionBody,
    ExecutableNeuralLabel, ExecutableScalarType, ExecutableTerm, ExecutableTypeReference,
    RelationDependencyKind, RelationDependencyProducerKind, ResolvedProgramExtractionError,
};

#[test]
fn resolved_program_extraction_builds_the_executable_dependency_graph() {
    let fixture = TempDir::new().expect("create fixture directory");
    let library_dir = fixture.path().join("lib");
    fs::create_dir_all(&library_dir).expect("create library directory");
    fs::write(
        library_dir.join("support.xlog"),
        "pred base(symbol).\nbase(alice).\n",
    )
    .expect("write imported module");

    let entry = fixture.path().join("main.xlog");
    fs::write(
        &entry,
        concat!(
            "#pragma epistemic_mode = g91\n",
            "use lib/support.\n",
            "candidate(X) :- base(X).\n",
            "blocked(X) :- candidate(X), not base(X).\n",
            "visible(X) :- candidate(X), know base(X).\n",
            ":- visible(alice), not blocked(alice).\n",
            "?- visible(X).\n",
        ),
    )
    .expect("write entry module");

    let extraction = load_modules(&entry, vec![])
        .expect("resolve complete program closure")
        .resolved_program_extraction(fixture.path())
        .expect("extract executable dependency graph");

    assert_eq!(
        extraction.schema_version,
        "xlog.resolved-program-extraction.v2"
    );
    assert_eq!(extraction.source_manifest.modules.len(), 2);

    let program = &extraction.executable_program;
    assert_eq!(program.rules.len(), 4);
    assert_eq!(program.constraints.len(), 1);
    assert_eq!(program.queries.len(), 1);

    let imported_fact = program
        .rules
        .iter()
        .find(|rule| rule.head.relation_id == "relation:base/1")
        .expect("imported fact must be executable");
    assert_ne!(
        imported_fact.module_id,
        extraction.source_manifest.entry_module_id
    );
    assert_eq!(
        imported_fact.head.terms,
        vec![ExecutableTerm::Symbol {
            value: "alice".to_string(),
        }]
    );

    let blocked = program
        .rules
        .iter()
        .find(|rule| rule.head.relation_id == "relation:blocked/1")
        .expect("blocked rule must be present");
    assert!(matches!(
        blocked.body.as_slice(),
        [
            ExecutableBodyLiteral::Positive { .. },
            ExecutableBodyLiteral::Negative { .. }
        ]
    ));

    let visible = program
        .rules
        .iter()
        .find(|rule| rule.head.relation_id == "relation:visible/1")
        .expect("visible rule must be present");
    assert!(matches!(
        &visible.body[1],
        ExecutableBodyLiteral::Epistemic {
            operator: EpistemicOperator::Know,
            negated: false,
            ..
        }
    ));

    assert!(program.dependencies.iter().any(|dependency| {
        dependency.producer_id == blocked.rule_id
            && dependency.producer_kind == RelationDependencyProducerKind::Rule
            && dependency.dependency_relation_id == "relation:base/1"
            && dependency.kind == RelationDependencyKind::Negative
    }));
    assert!(program.dependencies.iter().any(|dependency| {
        dependency.producer_id == visible.rule_id
            && dependency.producer_kind == RelationDependencyProducerKind::Rule
            && dependency.dependency_relation_id == "relation:base/1"
            && dependency.kind == RelationDependencyKind::Epistemic
    }));

    let visible_relation = program
        .relations
        .iter()
        .find(|relation| relation.relation_id == "relation:visible/1")
        .expect("visible relation must be indexed");
    assert!(visible_relation.scc_id.is_some());
    assert!(visible_relation.stratum.is_some());

    assert_eq!(program.queries[0].goal.relation_id, "relation:visible/1");
    assert!(matches!(
        program.constraints[0].body.as_slice(),
        [
            ExecutableBodyLiteral::Positive { .. },
            ExecutableBodyLiteral::Negative { .. }
        ]
    ));
}

#[test]
fn resolved_program_extraction_preserves_domains_schemas_and_functions() {
    let fixture = TempDir::new().expect("create fixture directory");
    let entry = fixture.path().join("main.xlog");
    fs::write(
        &entry,
        concat!(
            "domain key : u32.\n",
            "pred edge(key, key).\n",
            "pred answer(i64).\n",
            "func double(X) = if X > 0 then X * 2 else 0.\n",
            "answer(Y) :- Y is double(2), Y >= 4.\n",
        ),
    )
    .expect("write typed entry module");

    let extraction = load_modules(&entry, vec![])
        .expect("resolve typed program")
        .resolved_program_extraction(fixture.path())
        .expect("extract typed executable program");
    let program = &extraction.executable_program;

    assert_eq!(program.domains.len(), 1);
    assert_eq!(program.domains[0].name, "key");
    assert_eq!(program.domains[0].scalar_type, ExecutableScalarType::U32);

    assert_eq!(program.functions.len(), 1);
    assert_eq!(program.functions[0].name, "double");
    assert!(matches!(
        program.functions[0].body,
        ExecutableFunctionBody::Conditional { .. }
    ));

    let edge = program
        .relations
        .iter()
        .find(|relation| relation.relation_id == "relation:edge/2")
        .expect("declared edge relation");
    assert_eq!(
        edge.schema.as_ref().expect("declared schema")[0].type_reference,
        ExecutableTypeReference::Domain {
            name: "key".to_string(),
        }
    );

    let answer = program
        .rules
        .iter()
        .find(|rule| rule.head.relation_id == "relation:answer/1")
        .expect("answer rule");
    assert!(matches!(
        answer.body.as_slice(),
        [
            ExecutableBodyLiteral::IsExpression { .. },
            ExecutableBodyLiteral::Comparison { .. }
        ]
    ));
}

#[test]
fn resolved_program_extraction_preserves_relation_order_and_schema_arities() {
    let fixture = TempDir::new().unwrap();
    let entry = fixture.path().join("main.xlog");
    let mut source = String::from("domain label : symbol.\n");
    for index in (0..18).rev() {
        let arity = index % 9;
        let columns = (0..arity)
            .map(|column| {
                if index % 2 == 0 {
                    "label".to_string()
                } else {
                    format!("column_{column}: label")
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        let values = vec!["value"; arity].join(", ");
        source.push_str(&format!(
            "pred relation_{index:02}({columns}).\nrelation_{index:02}({values}).\n"
        ));
    }
    fs::write(&entry, source).unwrap();
    let resolver = load_modules(&entry, vec![]).unwrap();
    let extraction = resolver
        .resolved_program_extraction(fixture.path())
        .unwrap();
    let program = &extraction.executable_program;

    assert_eq!(program.relations.len(), 18);
    assert_eq!(program.sccs.len(), 18);
    for (index, relation) in program.relations.iter().enumerate() {
        let arity = index % 9;
        assert_eq!(relation.name, format!("relation_{index:02}"));
        assert_eq!(relation.arity, arity);
        assert_eq!(
            relation.relation_id,
            format!("relation:relation_{index:02}/{arity}")
        );
        let schema = relation.schema.as_ref().unwrap();
        assert_eq!(schema.len(), arity);
        for (column, declaration) in schema.iter().enumerate() {
            assert_eq!(
                declaration.name,
                (index % 2 != 0).then(|| format!("column_{column}"))
            );
            assert_eq!(
                declaration.type_reference,
                ExecutableTypeReference::Domain {
                    name: "label".to_string()
                }
            );
        }
        let scc = program
            .sccs
            .iter()
            .find(|scc| Some(&scc.scc_id) == relation.scc_id.as_ref())
            .unwrap();
        assert_eq!(scc.relation_ids, vec![relation.relation_id.clone()]);
        assert_eq!(relation.stratum, Some(0));
        assert_eq!(scc.stratum, Some(0));
        assert!(!relation.non_monotone);
        assert!(!scc.non_monotone);
    }
    let repeated = resolver
        .resolved_program_extraction(fixture.path())
        .unwrap();
    assert_eq!(
        serde_json::to_vec(&extraction).unwrap(),
        serde_json::to_vec(&repeated).unwrap()
    );
}

#[test]
fn resolved_program_extraction_rejects_conflicting_repeated_relation_arities() {
    let fixture = TempDir::new().unwrap();
    let entry = fixture.path().join("main.xlog");
    for source in [
        "pred p(symbol).\n?- p(a, b).\n",
        "p(a).\n?- p(a, b).\n",
        "pred p(symbol).\nevidence(p(a, b), true).\n",
        "?- p(a).\nquery(p(a, b)).\n",
    ] {
        fs::write(&entry, source).unwrap();
        let error = load_modules(&entry, vec![])
            .unwrap()
            .resolved_program_extraction(fixture.path())
            .unwrap_err();
        assert!(matches!(
            error,
            ResolvedProgramExtractionError::RelationArityConflict { name, first: 1, second: 2 }
                if name == "p"
        ));
    }
}

#[test]
fn resolved_program_extraction_preserves_probabilistic_and_learnable_semantics() {
    let fixture = TempDir::new().expect("create fixture directory");
    let entry = fixture.path().join("main.xlog");
    fs::write(
        &entry,
        concat!(
            "0.5::likely(ok).\n",
            "0.4::choice(a); 0.6::choice(b).\n",
            "evidence(likely(ok), true).\n",
            "query(likely(ok)).\n",
            "nn(classifier, [X], Y, [yes, no]) :: neural_label(X, Y).\n",
            "learnable(W) :: inferred(X, Y) :- left(X, Z), not right(Z, Y).\n",
        ),
    )
    .expect("write entry module");

    let extraction = load_modules(&entry, vec![])
        .expect("resolve program")
        .resolved_program_extraction(fixture.path())
        .expect("extract full executable program");
    let program = &extraction.executable_program;

    assert_eq!(program.probabilistic_facts.len(), 1);
    assert_eq!(
        program.probabilistic_facts[0].probability.ieee754_bits,
        0.5_f64.to_bits()
    );
    assert_eq!(
        program.probabilistic_facts[0].atom.relation_id,
        "relation:likely/1"
    );

    assert_eq!(program.annotated_disjunctions.len(), 1);
    assert_eq!(program.annotated_disjunctions[0].choices.len(), 2);
    assert_eq!(
        program.annotated_disjunctions[0].choices[1]
            .probability
            .ieee754_bits,
        0.6_f64.to_bits()
    );

    assert_eq!(program.evidence.len(), 1);
    assert!(program.evidence[0].value);
    assert_eq!(program.probabilistic_queries.len(), 1);
    assert_eq!(program.neural_predicates.len(), 1);
    assert_eq!(
        program.neural_predicates[0].labels,
        Some(vec![
            ExecutableNeuralLabel::Symbol {
                value: "yes".to_string(),
            },
            ExecutableNeuralLabel::Symbol {
                value: "no".to_string(),
            },
        ])
    );

    assert_eq!(program.learnable_rules.len(), 1);
    let learnable = &program.learnable_rules[0];
    assert_eq!(learnable.mask_name, "W");
    assert_eq!(learnable.head.relation_id, "relation:inferred/2");
    assert!(matches!(
        learnable.body.as_slice(),
        [
            ExecutableBodyLiteral::Positive { .. },
            ExecutableBodyLiteral::Negative { .. }
        ]
    ));
    assert!(program.dependencies.iter().any(|dependency| {
        dependency.producer_id == learnable.learnable_rule_id
            && dependency.producer_kind == RelationDependencyProducerKind::LearnableRule
            && dependency.dependency_relation_id == "relation:right/2"
            && dependency.kind == RelationDependencyKind::Negative
    }));

    for relation_id in [
        "relation:likely/1",
        "relation:choice/1",
        "relation:neural_label/2",
        "relation:inferred/2",
    ] {
        assert!(
            program
                .relations
                .iter()
                .any(|relation| relation.relation_id == relation_id),
            "missing executable relation {relation_id}"
        );
    }
}

#[test]
fn resolved_program_extraction_tracks_the_clause_actually_admitted_by_import_merge() {
    let fixture = TempDir::new().expect("create fixture directory");
    fs::write(
        fixture.path().join("first.xlog"),
        concat!(
            "pred shared(symbol).\n",
            "private pred hidden(symbol).\n",
            "shared(same).\n",
            "hidden(secret).\n",
        ),
    )
    .expect("write first module");
    fs::write(
        fixture.path().join("second.xlog"),
        "pred shared(symbol).\nshared(same).\n",
    )
    .expect("write second module");
    let entry = fixture.path().join("main.xlog");
    fs::write(
        &entry,
        "use first::{shared}.\nuse second::{shared}.\n?- shared(X).\n",
    )
    .expect("write entry module");

    let extraction = load_modules(&entry, vec![])
        .expect("resolve selective imports")
        .resolved_program_extraction(fixture.path())
        .expect("extract selectively merged program");
    let first_module_id = extraction
        .source_manifest
        .modules
        .iter()
        .find(|module| module.source_path == "first.xlog")
        .expect("first module in closure")
        .module_id
        .clone();
    let shared_rules = extraction
        .executable_program
        .rules
        .iter()
        .filter(|rule| rule.head.relation_id == "relation:shared/1")
        .collect::<Vec<_>>();

    assert_eq!(shared_rules.len(), 1);
    assert_eq!(shared_rules[0].module_id, first_module_id);
    assert!(!extraction
        .executable_program
        .relations
        .iter()
        .any(|relation| relation.name == "hidden"));

    let json = serde_json::to_value(&extraction).expect("serialize extraction");
    let dispositions = json["source_object_dispositions"]
        .as_array()
        .expect("complete authored-object custody");
    let source_count: usize = extraction
        .source_manifest
        .modules
        .iter()
        .map(|module| module.source_objects.len())
        .sum();
    assert_eq!(dispositions.len(), source_count);
    for module in &extraction.source_manifest.modules {
        for object in &module.source_objects {
            let custody = dispositions
                .iter()
                .find(|row| row["source_object_id"] == object.object_id)
                .expect("every source object has a disposition");
            assert_eq!(custody["module_id"], module.module_id);
            if object.primary_name.as_deref() == Some("hidden") {
                assert_eq!(custody["disposition"], "private");
            } else if object.primary_name.as_deref() == Some("shared") {
                if module.source_path == "second.xlog" {
                    assert_eq!(custody["disposition"], "coalesced");
                    assert_eq!(custody["equivalence"], "ast_equal");
                    if object.kind == xlog_logic::resolver::ResolvedSourceObjectKind::Rule {
                        assert_eq!(
                            custody["executable_source_object_id"],
                            shared_rules[0].rule_id
                        );
                    }
                } else if module.source_path == "first.xlog" {
                    assert_eq!(custody["disposition"], "admitted");
                    assert_eq!(custody["executable_source_object_id"], object.object_id);
                }
            }
        }
    }
}

#[test]
fn resolved_program_extraction_distinguishes_schema_equivalence_from_ast_equality() {
    let fixture = TempDir::new().unwrap();
    fs::write(
        fixture.path().join("nominal.xlog"),
        "domain label : symbol.\npred shared(label).\nshared(same).\n",
    )
    .unwrap();
    fs::write(
        fixture.path().join("scalar.xlog"),
        "pred shared(symbol).\nshared(same).\n",
    )
    .unwrap();
    let entry = fixture.path().join("main.xlog");
    fs::write(&entry, "use nominal.\nuse scalar.\n").unwrap();
    let extraction = load_modules(&entry, vec![])
        .unwrap()
        .resolved_program_extraction(fixture.path())
        .unwrap();
    assert_eq!(extraction.executable_program.rules.len(), 1);
    let scalar = extraction
        .source_manifest
        .modules
        .iter()
        .find(|module| module.source_path == "scalar.xlog")
        .unwrap();
    let declaration = scalar
        .source_objects
        .iter()
        .find(|object| object.kind == xlog_logic::resolver::ResolvedSourceObjectKind::Predicate)
        .unwrap();
    let json = serde_json::to_value(&extraction).unwrap();
    let custody = json["source_object_dispositions"]
        .as_array()
        .expect("complete authored-object custody")
        .iter()
        .find(|row| row["source_object_id"] == declaration.object_id)
        .unwrap();
    assert_eq!(custody["disposition"], "coalesced");
    assert_eq!(custody["equivalence"], "native_schema");
    assert_ne!(
        custody["executable_source_object_id"],
        declaration.object_id
    );

    fs::write(fixture.path().join("scalar.xlog"), "pred shared(i64).\n").unwrap();
    let resolver = load_modules(&entry, vec![]).unwrap();
    assert!(
        resolver
            .resolved_program_extraction(fixture.path())
            .is_err(),
        "same name never establishes schema equivalence"
    );
}

#[test]
fn resolved_program_extraction_keeps_dispositions_scoped_to_each_entry() {
    let fixture = TempDir::new().unwrap();
    fs::write(
        fixture.path().join("library.xlog"),
        concat!(
            "pred left(symbol).\npred right(symbol).\n",
            "left(a).\nright(b).\n?- left(X).\nquery(right(b)).\n",
        ),
    )
    .unwrap();
    let left_entry = fixture.path().join("left_entry.xlog");
    let right_entry = fixture.path().join("right_entry.xlog");
    fs::write(&left_entry, "use library::{left}.\n").unwrap();
    fs::write(&right_entry, "use library::{right}.\n").unwrap();
    let mut retained_id = None;
    for (entry, selected) in [(&left_entry, "left"), (&right_entry, "right")] {
        let extraction = load_modules(entry, vec![])
            .unwrap()
            .resolved_program_extraction(fixture.path())
            .unwrap();
        let module = extraction
            .source_manifest
            .modules
            .iter()
            .find(|module| module.source_path == "library.xlog")
            .unwrap();
        if let Some(id) = &retained_id {
            assert_eq!(id, &module.module_id);
        }
        retained_id = Some(module.module_id.clone());
        let json = serde_json::to_value(&extraction).unwrap();
        let dispositions = json["source_object_dispositions"]
            .as_array()
            .expect("complete authored-object custody");
        for object in &module.source_objects {
            let custody = dispositions
                .iter()
                .find(|row| row["source_object_id"] == object.object_id)
                .unwrap();
            let expected = match object.kind {
                xlog_logic::resolver::ResolvedSourceObjectKind::Query
                | xlog_logic::resolver::ResolvedSourceObjectKind::ProbabilisticQuery => {
                    "not_merged"
                }
                _ if object.primary_name.as_deref() == Some(selected) => "admitted",
                _ => "unselected",
            };
            assert_eq!(custody["disposition"], expected);
        }
        assert_eq!(extraction.executable_program.rules.len(), 1);
        assert_eq!(extraction.executable_program.rules[0].head.name, selected);
        assert!(extraction.executable_program.queries.is_empty());
        assert!(extraction
            .executable_program
            .probabilistic_queries
            .is_empty());
    }
}

#[test]
fn resolved_program_extraction_covers_every_declared_and_referenced_relation() {
    let fixture = TempDir::new().unwrap();
    let entry = fixture.path().join("main.xlog");
    fs::write(
        &entry,
        concat!(
            "#pragma epistemic_mode = g91\n",
            "pred isolated(symbol).\n",
            ":- constraint_only(a), not constraint_negative(a), know constraint_modal(a).\n",
            "?- query_only(X).\n",
            "0.5::probabilistic_only(a).\n",
            "0.4::choice_only(a); 0.6::other_choice(a).\n",
            "evidence(evidence_only(a), true).\nquery(prob_query_only(a)).\n",
            "nn(classifier, [X], Y, [yes, no]) :: neural_only(X, Y).\n",
            "learnable(W) :: learned(X, Y) :- left(X, Z), right(Z, Y).\n",
            "negative(X) :- left(X, X), not positive(X).\npositive(X) :- negative(X).\n",
            "aggregate(count(X)) :- aggregate_source(X).\n",
            "modal(X) :- left(X, X), know modal_source(X).\n",
        ),
    )
    .unwrap();
    let extraction = load_modules(&entry, vec![])
        .unwrap()
        .resolved_program_extraction(fixture.path())
        .unwrap();
    let program = &extraction.executable_program;
    for relation in &program.relations {
        let scc_id = relation
            .scc_id
            .as_ref()
            .unwrap_or_else(|| panic!("missing SCC for {}", relation.name));
        let scc = program
            .sccs
            .iter()
            .find(|scc| &scc.scc_id == scc_id)
            .unwrap();
        assert!(scc.relation_ids.contains(&relation.relation_id));
        assert_eq!(relation.stratum, scc.stratum);
        assert_eq!(relation.non_monotone, scc.non_monotone);
        if matches!(relation.name.as_str(), "negative" | "positive") {
            assert!(relation.non_monotone);
            assert_eq!(relation.stratum, None);
        } else {
            assert!(!relation.non_monotone);
            assert_eq!(
                relation.stratum,
                Some(if matches!(relation.name.as_str(), "aggregate" | "modal") {
                    1
                } else {
                    0
                })
            );
        }
    }
    assert_eq!(program.relations.len(), 20);
    assert_eq!(program.dependencies.len(), 8);
}

#[cfg(unix)]
#[test]
fn resolved_program_extraction_combines_selections_and_source_aliases_without_duplicate_execution()
{
    let fixture = TempDir::new().unwrap();
    fs::write(
        fixture.path().join("library.xlog"),
        "pred left(symbol).\npred right(symbol).\nleft(a).\nright(b).\n",
    )
    .unwrap();
    std::os::unix::fs::symlink("library.xlog", fixture.path().join("alias.xlog")).unwrap();
    fs::write(
        fixture.path().join("first.xlog"),
        "use library::{left}.\nfirst(X) :- left(X).\n",
    )
    .unwrap();
    fs::write(
        fixture.path().join("second.xlog"),
        "use library::{right}.\nsecond(X) :- right(X).\n",
    )
    .unwrap();
    let entry = fixture.path().join("main.xlog");
    fs::write(
        &entry,
        "use first.\nuse second.\nuse alias::{left}.\nuse alias::{right}.\n",
    )
    .unwrap();
    let resolver = load_modules(&entry, vec![]).unwrap();
    let extraction = resolver
        .resolved_program_extraction(fixture.path())
        .unwrap();
    assert_eq!(extraction.source_manifest.modules.len(), 4);
    assert_eq!(extraction.executable_program.rules.len(), 4);
    let json = serde_json::to_value(&extraction).unwrap();
    assert_eq!(
        json,
        serde_json::to_value(
            resolver
                .resolved_program_extraction(fixture.path())
                .unwrap()
        )
        .unwrap()
    );
    let dispositions = json["source_object_dispositions"]
        .as_array()
        .expect("complete authored-object custody");
    let library = extraction
        .source_manifest
        .modules
        .iter()
        .find(|module| module.source_path == "library.xlog")
        .unwrap();
    for object in &library.source_objects {
        let custody = dispositions
            .iter()
            .find(|row| row["source_object_id"] == object.object_id)
            .unwrap();
        assert_eq!(custody["disposition"], "admitted");
        assert_eq!(custody["executable_source_object_id"], object.object_id);
        assert_eq!(
            dispositions
                .iter()
                .filter(|row| row["source_object_id"] == object.object_id)
                .count(),
            1
        );
    }
}

#[test]
fn resolved_program_extraction_preserves_entry_objects_and_import_configuration_scope() {
    let fixture = TempDir::new().unwrap();
    let library = fixture.path().join("library.xlog");
    fs::write(
        &library,
        concat!(
            "#pragma magic_sets = off\n",
            "domain label : symbol.\n",
            "pred shared(symbol).\nprivate pred hidden(symbol).\n",
            "shared(same).\nhidden(secret).\n",
            "func selected(X) = X.\nfunc omitted(X) = X.\nprivate func secret(X) = X.\n",
        ),
    )
    .unwrap();
    let entry = fixture.path().join("main.xlog");
    fs::write(
        &entry,
        concat!(
            "#pragma magic_sets = off\n",
            "use library::{shared, selected}.\n",
            "pred shared(symbol).\nshared(same).\n",
        ),
    )
    .unwrap();
    let extraction = load_modules(&entry, vec![])
        .unwrap()
        .resolved_program_extraction(fixture.path())
        .unwrap();
    assert_eq!(
        extraction.executable_program.rules.len(),
        2,
        "entry clauses are appended under the canonical merge contract"
    );
    assert_eq!(
        extraction.executable_program.domains.len(),
        1,
        "domain imports are independent of selected item names"
    );
    assert_eq!(extraction.executable_program.functions.len(), 1);
    let json = serde_json::to_value(&extraction).unwrap();
    let dispositions = json["source_object_dispositions"].as_array().unwrap();
    for module in &extraction.source_manifest.modules {
        for object in &module.source_objects {
            let custody = dispositions
                .iter()
                .find(|row| row["source_object_id"] == object.object_id)
                .unwrap();
            let expected = match (
                module.source_path.as_str(),
                object.kind,
                object.primary_name.as_deref(),
            ) {
                ("main.xlog", xlog_logic::resolver::ResolvedSourceObjectKind::Import, _) => {
                    "resolution"
                }
                ("main.xlog", xlog_logic::resolver::ResolvedSourceObjectKind::Directive, _) => {
                    "configuration"
                }
                ("library.xlog", xlog_logic::resolver::ResolvedSourceObjectKind::Directive, _) => {
                    "not_merged"
                }
                ("library.xlog", _, Some("hidden" | "secret")) => "private",
                ("library.xlog", _, Some("omitted")) => "unselected",
                (
                    "library.xlog",
                    xlog_logic::resolver::ResolvedSourceObjectKind::Predicate,
                    Some("shared"),
                ) => "coalesced",
                _ => "admitted",
            };
            assert_eq!(custody["disposition"], expected, "{object:?}");
        }
    }
    let local = load_modules(&library, vec![])
        .unwrap()
        .resolved_program_extraction(fixture.path())
        .unwrap();
    assert_eq!(local.executable_program.rules.len(), 2);
    assert_eq!(local.executable_program.functions.len(), 3);
    let json = serde_json::to_value(&local).unwrap();
    assert!(json["source_object_dispositions"]
        .as_array()
        .unwrap()
        .iter()
        .all(|row| row["disposition"] == "admitted" || row["disposition"] == "configuration"));
}

#[test]
fn resolved_program_extraction_does_not_coalesce_distinct_authored_constant_forms() {
    let fixture = TempDir::new().unwrap();
    fs::write(
        fixture.path().join("symbol.xlog"),
        "pred shared(symbol).\nshared(identifier).\n",
    )
    .unwrap();
    fs::write(
        fixture.path().join("string.xlog"),
        "pred shared(symbol).\nshared(\"identifier\").\n",
    )
    .unwrap();
    let entry = fixture.path().join("main.xlog");
    fs::write(&entry, "use symbol.\nuse string.\n").unwrap();
    let extraction = load_modules(&entry, vec![])
        .unwrap()
        .resolved_program_extraction(fixture.path())
        .unwrap();
    assert_eq!(extraction.executable_program.rules.len(), 2);
    let json = serde_json::to_value(&extraction).unwrap();
    for rule in &extraction.executable_program.rules {
        let custody = json["source_object_dispositions"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["source_object_id"] == rule.source_object_id)
            .unwrap();
        assert_eq!(custody["disposition"], "admitted");
    }
}

#[test]
fn resolved_program_extraction_does_not_enable_unsupported_learnable_bodies() {
    let source = "learnable(W) :: inferred(X, Y) :- left(X, Y), not right(X, Y).\n";
    let error = xlog_logic::compile::compile(source).unwrap_err();
    assert!(
        error.to_string().contains("only positive atoms allowed"),
        "{error}"
    );
}

#[test]
fn resolved_program_extraction_preserves_structured_terms_aggregates_and_univ() {
    let fixture = TempDir::new().expect("create fixture directory");
    let entry = fixture.path().join("main.xlog");
    fs::write(
        &entry,
        concat!(
            "value([1, pair(foo, 2), [3]]).\n",
            "cons([H | T]) :- tail(T).\n",
            "out_degree(X, count(Y)) :- edge(X, Y).\n",
            "decomposed(X) :- X =.. [pair, foo, 2].\n",
        ),
    )
    .expect("write structured-term program");

    let extraction = load_modules(&entry, vec![])
        .expect("resolve structured-term program")
        .resolved_program_extraction(fixture.path())
        .expect("extract structured-term program");
    let rules = &extraction.executable_program.rules;

    let value = rules
        .iter()
        .find(|rule| rule.head.name == "value")
        .expect("value fact");
    assert!(matches!(
        &value.head.terms[0],
        ExecutableTerm::List { items }
            if matches!(&items[1], ExecutableTerm::Compound { functor, .. } if functor == "pair")
    ));

    let cons = rules
        .iter()
        .find(|rule| rule.head.name == "cons")
        .expect("cons rule");
    assert!(matches!(&cons.head.terms[0], ExecutableTerm::Cons { .. }));

    let aggregate = rules
        .iter()
        .find(|rule| rule.head.name == "out_degree")
        .expect("aggregate rule");
    assert!(matches!(
        &aggregate.head.terms[1],
        ExecutableTerm::Aggregate {
            operator: AggregateOperator::Count,
            variable,
        } if variable == "Y"
    ));

    let decomposed = rules
        .iter()
        .find(|rule| rule.head.name == "decomposed")
        .expect("univ rule");
    assert!(matches!(
        decomposed.body.as_slice(),
        [ExecutableBodyLiteral::Univ { .. }]
    ));
}
