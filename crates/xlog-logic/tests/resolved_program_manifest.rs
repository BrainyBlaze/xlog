use std::fs;

use sha2::{Digest, Sha256};
use tempfile::TempDir;
use xlog_logic::compile::load_modules;
use xlog_logic::resolver::{
    ModuleResolver, ResolvedProgramManifestError, ResolvedSourceObjectKind,
    ResolvedSourceObjectProvenance,
};

#[test]
fn resolved_program_manifest_is_content_addressed_and_preserves_authored_order() {
    let fixture = TempDir::new().expect("create fixture directory");
    let library_dir = fixture.path().join("lib");
    fs::create_dir_all(&library_dir).expect("create library directory");
    fs::write(
        library_dir.join("support.xlog"),
        "pred support(symbol).\nsupport(ok).\n",
    )
    .expect("write imported module");

    let entry = fixture.path().join("main.xlog");
    fs::write(
        &entry,
        concat!(
            "use lib/support.\n",
            "domain verdict: symbol.\n",
            "pred decision(verdict).\n",
            "decision(X) :- support(X), not blocked(X).\n",
            ":- decision(blocked).\n",
            "?- decision(X).\n",
        ),
    )
    .expect("write entry module");

    let resolver = load_modules(&entry, vec![]).expect("resolve program closure");
    let manifest = resolver
        .resolved_program_manifest(fixture.path())
        .expect("build resolved program manifest");

    assert_eq!(manifest.schema_version, "xlog.resolved-program-manifest.v1");
    assert_eq!(manifest.modules.len(), 2);
    assert_eq!(manifest.imports.len(), 1);
    assert_eq!(
        manifest,
        resolver.resolved_program_manifest(fixture.path()).unwrap()
    );

    let entry_module = manifest
        .modules
        .iter()
        .find(|module| module.module_id == manifest.entry_module_id)
        .expect("entry module must be present");
    assert_eq!(entry_module.source_path, "main.xlog");
    assert_eq!(entry_module.logical_paths, vec!["main"]);
    assert!(entry_module.content_sha256.starts_with("sha256:"));
    assert_eq!(entry_module.content_sha256.len(), 71);

    let authored_kinds = entry_module
        .source_objects
        .iter()
        .map(|object| object.kind)
        .collect::<Vec<_>>();
    assert_eq!(
        authored_kinds,
        vec![
            ResolvedSourceObjectKind::Import,
            ResolvedSourceObjectKind::Domain,
            ResolvedSourceObjectKind::Predicate,
            ResolvedSourceObjectKind::Rule,
            ResolvedSourceObjectKind::Constraint,
            ResolvedSourceObjectKind::Query,
        ]
    );
    assert!(entry_module.source_objects.iter().all(|object| {
        object.content_sha256.starts_with("sha256:")
            && object.span.start < object.span.end
            && object.span.line > 0
            && object.span.column > 0
            && object.provenance == ResolvedSourceObjectProvenance::Authored
    }));

    let import = &manifest.imports[0];
    assert_eq!(import.importer_module_id, manifest.entry_module_id);
    assert_eq!(import.declared_path, vec!["lib", "support"]);
    assert_eq!(import.resolved_path, vec!["lib", "support"]);
    assert_eq!(import.imported_items, None);
    assert_eq!(
        import.source_object_id,
        entry_module.source_objects[0].object_id
    );
    assert!(manifest
        .modules
        .iter()
        .any(|module| module.module_id == import.target_module_id
            && module.source_path == "lib/support.xlog"));
}

#[test]
fn resolved_program_manifest_excludes_modules_unreachable_from_the_current_entry() {
    let fixture = TempDir::new().expect("create fixture directory");
    fs::write(
        fixture.path().join("old_support.xlog"),
        "pred old_support(symbol).\nold_support(old).\n",
    )
    .expect("write old support module");
    let old_entry = fixture.path().join("old_main.xlog");
    fs::write(&old_entry, "use old_support.\n").expect("write old entry");

    fs::write(
        fixture.path().join("current_support.xlog"),
        "pred current_support(symbol).\ncurrent_support(current).\n",
    )
    .expect("write current support module");
    let current_entry = fixture.path().join("current_main.xlog");
    fs::write(&current_entry, "use current_support.\n").expect("write current entry");

    let mut resolver = ModuleResolver::new(vec![]);
    resolver
        .load_entry_file(&old_entry)
        .expect("resolve old entry");
    resolver
        .load_entry_file(&current_entry)
        .expect("resolve current entry");

    let manifest = resolver
        .resolved_program_manifest(fixture.path())
        .expect("build current manifest");
    let source_paths = manifest
        .modules
        .iter()
        .map(|module| module.source_path.as_str())
        .collect::<Vec<_>>();

    assert_eq!(
        source_paths,
        vec!["current_main.xlog", "current_support.xlog"]
    );
}

#[cfg(unix)]
#[test]
fn resolved_program_manifest_excludes_logical_aliases_from_a_previous_entry() {
    let fixture = TempDir::new().expect("create fixture directory");
    let support = fixture.path().join("support.xlog");
    fs::write(&support, "pred support(symbol).\nsupport(ok).\n")
        .expect("write canonical support module");
    std::os::unix::fs::symlink(&support, fixture.path().join("old_alias.xlog"))
        .expect("create old module alias");
    std::os::unix::fs::symlink(&support, fixture.path().join("current_alias.xlog"))
        .expect("create current module alias");

    let old_entry = fixture.path().join("old_main.xlog");
    fs::write(&old_entry, "use old_alias.\n").expect("write old entry");
    let current_entry = fixture.path().join("current_main.xlog");
    fs::write(&current_entry, "use current_alias.\n").expect("write current entry");

    let mut resolver = ModuleResolver::new(vec![]);
    resolver
        .load_entry_file(&old_entry)
        .expect("resolve old entry");
    resolver
        .load_entry_file(&current_entry)
        .expect("resolve current entry");

    let manifest = resolver
        .resolved_program_manifest(fixture.path())
        .expect("build current manifest");
    let support_module = manifest
        .modules
        .iter()
        .find(|module| module.source_path == "support.xlog")
        .expect("canonical support module must be present");

    assert_eq!(support_module.logical_paths, vec!["current_alias"]);
}

#[test]
fn resolved_program_manifest_rejects_a_module_outside_the_declared_source_root() {
    let fixture = TempDir::new().expect("create fixture directory");
    let external = TempDir::new().expect("create external module directory");
    fs::write(
        external.path().join("support.xlog"),
        "pred support(symbol).\nsupport(ok).\n",
    )
    .expect("write external support module");
    let entry = fixture.path().join("main.xlog");
    fs::write(&entry, "use support.\n").expect("write entry module");

    let resolver = load_modules(&entry, vec![external.path().to_path_buf()])
        .expect("resolve module outside source root");
    let error = resolver
        .resolved_program_manifest(fixture.path())
        .expect_err("manifest must reject sources outside source root");

    assert!(matches!(
        error,
        ResolvedProgramManifestError::SourceOutsideRoot { .. }
    ));
}

#[test]
fn resolved_program_manifest_rejects_an_invalid_resolved_import_surface() {
    let fixture = TempDir::new().expect("create fixture directory");
    fs::write(fixture.path().join("library.xlog"), "known(1).\n").expect("write library module");
    let entry = fixture.path().join("main.xlog");
    fs::write(&entry, "use library::{missing}.\n").expect("write invalid entry module");

    let resolver = load_modules(&entry, vec![]).expect("resolve module paths");
    let error = resolver
        .resolved_program_manifest(fixture.path())
        .expect_err("manifest must reject an invalid merged import surface");

    assert!(matches!(
        error,
        ResolvedProgramManifestError::ModuleValidation { .. }
    ));
    assert!(error.to_string().contains("error[E0404]"));
}

#[test]
fn resolved_program_manifest_uses_the_exact_bytes_parsed_by_the_resolver() {
    let fixture = TempDir::new().expect("create fixture directory");
    let entry = fixture.path().join("main.xlog");
    fs::write(&entry, "pred answer(symbol).\nanswer(first).\n")
        .expect("write initial entry module");

    let resolver = load_modules(&entry, vec![]).expect("resolve initial entry module");
    let before = resolver
        .resolved_program_manifest(fixture.path())
        .expect("build initial manifest");

    fs::write(&entry, "pred answer(symbol).\nanswer(second).\n")
        .expect("mutate entry module after resolution");
    let same_resolution = resolver
        .resolved_program_manifest(fixture.path())
        .expect("rebuild manifest from existing resolution");
    assert_eq!(same_resolution, before);

    let reloaded = load_modules(&entry, vec![])
        .expect("resolve mutated entry module")
        .resolved_program_manifest(fixture.path())
        .expect("build manifest from mutated source");
    assert_ne!(
        reloaded.modules[0].content_sha256,
        before.modules[0].content_sha256
    );
}

#[test]
fn resolved_program_manifest_hashes_the_exact_parser_owned_statement_span() {
    let fixture = TempDir::new().expect("create fixture directory");
    let entry = fixture.path().join("main.xlog");
    let source = "#pragma magic_sets = on   \npred answer(symbol).\nanswer(first).\n";
    fs::write(&entry, source).expect("write entry module");

    let manifest = load_modules(&entry, vec![])
        .expect("resolve entry module")
        .resolved_program_manifest(fixture.path())
        .expect("build source manifest");
    let module = &manifest.modules[0];
    let directive = &module.source_objects[0];
    let exact_statement = &source.as_bytes()[directive.span.start..directive.span.end];

    assert_eq!(directive.content_sha256, sha256_prefixed(exact_statement));
}

#[test]
fn resolved_program_manifest_preserves_unique_names_and_arities_across_construct_families() {
    let fixture = TempDir::new().expect("create fixture directory");
    let entry = fixture.path().join("main.xlog");
    fs::write(
        &entry,
        concat!(
            "func twice(X) = X + X.\n",
            "0.5::likely(ok).\n",
            "0.4::choice(a); 0.6::choice(b).\n",
            "evidence(likely(ok), true).\n",
            "query(likely(ok)).\n",
            "nn(classifier, [X], Y, [yes,no]) :: neural_label(X,Y).\n",
            "learnable(W) :: inferred(X,Y) :- left(X,Z), right(Z,Y).\n",
        ),
    )
    .expect("write multi-construct entry module");

    let manifest = load_modules(&entry, vec![])
        .expect("resolve multi-construct entry module")
        .resolved_program_manifest(fixture.path())
        .expect("build multi-construct source manifest");
    let objects = &manifest.modules[0].source_objects;

    assert_object_identity(objects, ResolvedSourceObjectKind::Function, "twice", 1);
    assert_object_identity(
        objects,
        ResolvedSourceObjectKind::ProbabilisticFact,
        "likely",
        1,
    );
    assert_object_identity(objects, ResolvedSourceObjectKind::Evidence, "likely", 1);
    assert_object_identity(
        objects,
        ResolvedSourceObjectKind::ProbabilisticQuery,
        "likely",
        1,
    );
    assert_object_identity(
        objects,
        ResolvedSourceObjectKind::NeuralPredicate,
        "neural_label",
        2,
    );
    assert_object_identity(
        objects,
        ResolvedSourceObjectKind::LearnableRule,
        "inferred",
        2,
    );

    let annotated_disjunction = objects
        .iter()
        .find(|object| object.kind == ResolvedSourceObjectKind::AnnotatedDisjunction)
        .expect("annotated disjunction must be inventoried");
    assert_eq!(annotated_disjunction.primary_name, None);
    assert_eq!(annotated_disjunction.arity, None);
}

fn assert_object_identity(
    objects: &[xlog_logic::resolver::ResolvedSourceObject],
    kind: ResolvedSourceObjectKind,
    expected_name: &str,
    expected_arity: usize,
) {
    let object = objects
        .iter()
        .find(|object| object.kind == kind)
        .unwrap_or_else(|| panic!("{kind:?} must be inventoried"));
    assert_eq!(object.primary_name.as_deref(), Some(expected_name));
    assert_eq!(object.arity, Some(expected_arity));
}

fn sha256_prefixed(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let hex = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("sha256:{hex}")
}
