use std::fs;

use tempfile::TempDir;
use xlog_logic::compile::load_modules;
use xlog_logic::resolver::{
    AggregateOperator, EpistemicOperator, ExecutableBodyLiteral, ExecutableFunctionBody,
    ExecutableNeuralLabel, ExecutableScalarType, ExecutableTerm, ExecutableTypeReference,
    RelationDependencyKind, RelationDependencyProducerKind,
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
        "xlog.resolved-program-extraction.v1"
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
