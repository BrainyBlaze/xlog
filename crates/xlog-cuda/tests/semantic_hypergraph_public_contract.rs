use std::sync::Arc;

use xlog_core::{symbol, MemoryBudget, RelId, ScalarType, Schema};
use xlog_cuda::{
    CudaProviderBuilder, SemanticAdmissionLimits, SemanticAdmissionRecords, SemanticArgument,
    SemanticExtents, SemanticHandleKind, SemanticHypergraphCapacities, SemanticHypergraphError,
    SemanticInsertOutcome, SemanticPolarity, SemanticPredicateRecord, SemanticRecordRole,
    SemanticSupportRecord, SemanticTruth, SemanticTypedRecord, SemanticView,
};

fn typed_records(symbol_id: u32) -> SemanticAdmissionRecords {
    let roles = [
        SemanticRecordRole::Statement,
        SemanticRecordRole::Qualifier,
        SemanticRecordRole::Provenance,
        SemanticRecordRole::Source,
        SemanticRecordRole::Context,
        SemanticRecordRole::Scope,
    ];
    let predicates = roles
        .into_iter()
        .enumerate()
        .map(|(index, role)| SemanticPredicateRecord {
            predicate: RelId(index as u32),
            role,
            schema: if index == 0 {
                Schema::new(vec![
                    ("entity".into(), ScalarType::Symbol),
                    ("value".into(), ScalarType::U64),
                ])
            } else {
                Schema::new(vec![("value".into(), ScalarType::U64)])
            },
        })
        .collect();
    let mut records = vec![SemanticTypedRecord {
        predicate: RelId(0),
        arguments: vec![
            SemanticArgument::Symbol(symbol_id),
            SemanticArgument::U64(19),
        ],
        qualifiers: vec![1, 2],
    }];
    for (predicate, value) in [
        (1, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (2, 7),
        (3, 8),
        (4, 9),
        (5, 10),
    ] {
        records.push(SemanticTypedRecord {
            predicate: RelId(predicate),
            arguments: vec![SemanticArgument::U64(value)],
            qualifiers: vec![],
        });
    }
    SemanticAdmissionRecords {
        predicates,
        records,
        supports: vec![
            SemanticSupportRecord {
                statement: 0,
                polarity: SemanticPolarity::Pro,
                provenance: 3,
                source: 4,
                context: 5,
                scope: 6,
            },
            SemanticSupportRecord {
                statement: 0,
                polarity: SemanticPolarity::Contra,
                provenance: 7,
                source: 8,
                context: 9,
                scope: 10,
            },
        ],
    }
}

fn admission_limits() -> SemanticAdmissionLimits {
    SemanticAdmissionLimits {
        max_records: 32,
        max_terms: 64,
        max_references: 32,
        max_utf8_bytes: 1024,
    }
}

#[test]
fn initial_support_import_binds_session_to_the_populated_root() {
    if std::env::var("XLOG_REQUIRE_CUDA").as_deref() != Ok("1") {
        return;
    }
    let provider = Arc::new(
        CudaProviderBuilder::new(0, MemoryBudget::with_limit(256 * 1024 * 1024))
            .with_stream_capacity(1)
            .build()
            .unwrap(),
    );
    let runtime = Arc::clone(provider.memory().runtime().unwrap());
    let stream_id = runtime.stream_pool().acquire().unwrap();
    let stream = runtime.stream_pool().resolve(stream_id).unwrap();
    let domain = provider
        .bind_resident_execution_domain(runtime, stream_id, stream)
        .unwrap();
    let graph = provider
        .allocate_semantic_hypergraph(
            &domain,
            SemanticHypergraphCapacities::try_new(4, 5, 6, 6).unwrap(),
        )
        .unwrap();
    let empty = graph.empty_root();
    let mut records = typed_records(symbol::intern("initial semantic source"));
    let mut second = records.records[0].clone();
    second.arguments[1] = SemanticArgument::U64(20);
    let second_index = records.records.len() as u32;
    records.records.push(second);
    let mut unselected = records.supports[0].clone();
    unselected.statement = second_index;
    records.supports.push(unselected);
    let mut graph = graph
        .admit_initial_records(records.clone(), &[0, 1, 0], admission_limits())
        .unwrap();
    let admission = graph.admission().unwrap();
    let first_key = admission.statement_key(0).unwrap();
    let second_key = admission.statement_key(second_index).unwrap();
    let base = admission.base();
    let snapshot = *admission.base_snapshot();
    let identity = admission.identity();
    assert_ne!(base, empty);
    assert_eq!(snapshot.extents(), SemanticExtents::new(1, 2, 2));
    assert_eq!(
        graph.truth(SemanticView::Root(base), &first_key).unwrap(),
        SemanticTruth::Both,
    );
    assert_eq!(
        graph.truth(SemanticView::Root(base), &second_key).unwrap(),
        SemanticTruth::Neither,
    );
    assert_eq!(
        graph.snapshot(SemanticView::Root(empty)).unwrap().extents(),
        SemanticExtents::default(),
    );
    assert_eq!(graph.snapshot(SemanticView::Root(base)).unwrap(), snapshot);
    let mut session = xlog_cuda::SemanticTransitionSession::from_hypergraph(graph).unwrap();
    assert_eq!(session.root_snapshot().unwrap(), snapshot);

    let declaration_only = provider
        .allocate_semantic_hypergraph(
            &domain,
            SemanticHypergraphCapacities::try_new(4, 5, 6, 6).unwrap(),
        )
        .unwrap()
        .admit_initial_records(records, &[], admission_limits())
        .unwrap();
    assert_ne!(declaration_only.admission().unwrap().identity(), identity);
    assert_eq!(
        declaration_only
            .admission()
            .unwrap()
            .base_snapshot()
            .extents(),
        SemanticExtents::default(),
    );
}

#[test]
fn cold_admission_retains_typed_content_and_rejects_invalid_dependencies() {
    if std::env::var("XLOG_REQUIRE_CUDA").as_deref() != Ok("1") {
        return;
    }
    let provider = CudaProviderBuilder::new(0, MemoryBudget::with_limit(64 * 1024 * 1024))
        .with_stream_capacity(1)
        .build()
        .unwrap();
    let provider = Arc::new(provider);
    let runtime = Arc::clone(provider.memory().runtime().unwrap());
    let stream_id = runtime.stream_pool().acquire().unwrap();
    let stream = runtime.stream_pool().resolve(stream_id).unwrap();
    let domain = provider
        .bind_resident_execution_domain(runtime, stream_id, stream)
        .unwrap();
    let mut graph = provider
        .allocate_semantic_hypergraph(
            &domain,
            SemanticHypergraphCapacities::try_new(2, 1, 2, 2).unwrap(),
        )
        .unwrap();
    let base = graph.empty_root();
    let id = symbol::intern("accepted semantic text");
    let records = typed_records(id);
    let before = graph.execution_stats();
    let mut invalid = records.clone();
    invalid.records[0].arguments[0] = SemanticArgument::Symbol(u32::MAX);
    assert!(graph
        .admit_records(base, invalid, admission_limits())
        .is_err());
    let mut invalid = records.clone();
    invalid.records[0].arguments[0] = SemanticArgument::U32(id);
    assert!(graph
        .admit_records(base, invalid, admission_limits())
        .is_err());
    let mut invalid = records.clone();
    invalid.records[0].qualifiers = vec![99];
    assert!(graph
        .admit_records(base, invalid, admission_limits())
        .is_err());
    let mut invalid = records.clone();
    invalid.supports[0].source = 3; // provenance is not a source record
    assert!(graph
        .admit_records(base, invalid, admission_limits())
        .is_err());
    let mut invalid = records.clone();
    invalid.records[0].predicate = RelId(99);
    assert!(graph
        .admit_records(base, invalid, admission_limits())
        .is_err());
    let mut invalid = records.clone();
    invalid.records[0].arguments.pop();
    assert!(graph
        .admit_records(base, invalid, admission_limits())
        .is_err());
    let mut invalid = records.clone();
    invalid.records[1].qualifiers.push(2);
    assert!(graph
        .admit_records(base, invalid, admission_limits())
        .is_err());
    let mut invalid = records.clone();
    invalid.predicates[0].schema.key_columns = vec![2];
    assert!(graph
        .admit_records(base, invalid, admission_limits())
        .is_err());
    let mut invalid = records.clone();
    invalid.predicates[0].schema.key_columns = vec![0, 0];
    assert!(graph
        .admit_records(base, invalid, admission_limits())
        .is_err());
    let mut invalid = records.clone();
    invalid.predicates[1].predicate = RelId(0);
    assert!(graph
        .admit_records(base, invalid, admission_limits())
        .is_err());
    for limits in [
        SemanticAdmissionLimits {
            max_records: 18,
            ..admission_limits()
        },
        SemanticAdmissionLimits {
            max_terms: 25,
            ..admission_limits()
        },
        SemanticAdmissionLimits {
            max_references: 11,
            ..admission_limits()
        },
    ] {
        assert!(graph.admit_records(base, records.clone(), limits).is_err());
    }
    let mut limits = admission_limits();
    limits.max_utf8_bytes = 0;
    assert!(graph.admit_records(base, records.clone(), limits).is_err());
    assert!(graph.admission().is_none());
    assert_eq!(
        graph.execution_stats(),
        before,
        "invalid cold input must not launch or mutate CUDA"
    );
    let exact_limits = SemanticAdmissionLimits {
        max_records: 19,
        max_terms: 26,
        max_references: 12,
        // entity/value column names and their default sort labels, plus symbol text.
        max_utf8_bytes: 72 + "accepted semantic text".len(),
    };
    let admission = graph
        .admit_records(base, records.clone(), exact_limits)
        .unwrap();
    assert_eq!(admission.encoding_generation(), 2);
    assert_eq!(admission.base(), base);
    assert_eq!(admission.records(), &records);
    assert_eq!(
        admission.symbols().entries()[0].1.as_ref(),
        "accepted semantic text"
    );
    let key = admission.statement_key(0).unwrap();
    let event = admission.support_event(0).unwrap();
    assert!(
        admission.statement_key(1).is_err(),
        "a qualifier cannot be used as a statement"
    );
    let mut other = provider
        .allocate_semantic_hypergraph(
            &domain,
            SemanticHypergraphCapacities::try_new(2, 1, 2, 2).unwrap(),
        )
        .unwrap();
    let foreign = other
        .admit_records(other.empty_root(), records.clone(), admission_limits())
        .unwrap();
    let foreign_key = foreign.statement_key(0).unwrap();
    let foreign_event = foreign.support_event(0).unwrap();
    assert_eq!(
        foreign_key.identity(),
        key.identity(),
        "physical owner is not semantic identity"
    );
    let before = graph.execution_stats();
    assert!(graph.truth(SemanticView::Root(base), &foreign_key).is_err());
    assert_eq!(graph.execution_stats(), before);
    assert!(
        graph
            .admit_records(base, records, admission_limits())
            .is_err(),
        "admission is immutable"
    );
    let fork = graph.fork(base).unwrap();
    let before = graph.execution_stats();
    assert!(graph.insert_support(fork, &key, &foreign_event).is_err());
    assert_eq!(graph.execution_stats(), before);
    graph.insert_support(fork, &key, &event).unwrap();
    assert_eq!(
        graph.truth(SemanticView::Fork(fork), &key).unwrap(),
        SemanticTruth::True
    );
    graph.discard(fork).unwrap();
}

#[test]
fn symbol_reuse_changes_new_admission_not_retained_meaning() {
    if std::env::var("XLOG_REQUIRE_CUDA").as_deref() != Ok("1") {
        return;
    }
    // Registry clear is intentionally process-global. Isolate that destructive
    // fixture from sibling tests without serializing the CUDA test suite.
    const CHILD: &str = "XLOG_SYMBOL_ADMISSION_CHILD";
    if std::env::var(CHILD).as_deref() != Ok("1") {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "symbol_reuse_changes_new_admission_not_retained_meaning",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "isolated symbol contract failed:\n{}\n{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        return;
    }
    let provider = CudaProviderBuilder::new(0, MemoryBudget::with_limit(64 * 1024 * 1024))
        .with_stream_capacity(1)
        .build()
        .unwrap();
    let provider = Arc::new(provider);
    let runtime = Arc::clone(provider.memory().runtime().unwrap());
    let stream_id = runtime.stream_pool().acquire().unwrap();
    let stream = runtime.stream_pool().resolve(stream_id).unwrap();
    let domain = provider
        .bind_resident_execution_domain(runtime, stream_id, stream)
        .unwrap();
    symbol::clear();
    let id = symbol::intern("old meaning");
    let mut original = provider
        .allocate_semantic_hypergraph(
            &domain,
            SemanticHypergraphCapacities::try_new(2, 1, 2, 2).unwrap(),
        )
        .unwrap();
    let admitted = original
        .admit_records(original.empty_root(), typed_records(id), admission_limits())
        .unwrap();
    let key = admitted.statement_key(0).unwrap();
    let event = admitted.support_event(0).unwrap();
    let seal = admitted.identity();
    symbol::clear();
    assert_eq!(symbol::intern("new meaning"), id);
    let admitted = original.admission().unwrap();
    assert_eq!(admitted.symbols().entries()[0].1.as_ref(), "old meaning");
    assert_eq!(admitted.statement_key(0).unwrap(), key);
    assert_eq!(admitted.identity(), seal);
    let mut replacement = provider
        .allocate_semantic_hypergraph(
            &domain,
            SemanticHypergraphCapacities::try_new(2, 1, 2, 2).unwrap(),
        )
        .unwrap();
    let new = replacement
        .admit_records(
            replacement.empty_root(),
            typed_records(id),
            admission_limits(),
        )
        .unwrap();
    assert_eq!(new.schema_generation(), admitted.schema_generation());
    assert_ne!(new.statement_key(0).unwrap().identity(), key.identity());
    assert_ne!(new.identity(), seal);
    // The same meaning at another process-local ID retains semantic identity.
    let another_id = symbol::intern("old meaning");
    assert_ne!(another_id, id);
    let mut equivalent = provider
        .allocate_semantic_hypergraph(
            &domain,
            SemanticHypergraphCapacities::try_new(2, 1, 2, 2).unwrap(),
        )
        .unwrap();
    let same = equivalent
        .admit_records(
            equivalent.empty_root(),
            typed_records(another_id),
            admission_limits(),
        )
        .unwrap();
    assert_eq!(same.statement_key(0).unwrap().identity(), key.identity());
    let fork = original.fork(original.empty_root()).unwrap();
    original.insert_support(fork, &key, &event).unwrap();
    assert_eq!(
        original.truth(SemanticView::Fork(fork), &key).unwrap(),
        SemanticTruth::True
    );
    original.discard(fork).unwrap();
}

#[test]
fn admitted_support_metadata_binds_to_the_selected_statement() {
    if std::env::var("XLOG_REQUIRE_CUDA").as_deref() != Ok("1") { return; }
    let provider = CudaProviderBuilder::new(0, MemoryBudget::with_limit(64 * 1024 * 1024))
        .with_stream_capacity(1).build().unwrap();
    let provider = Arc::new(provider);
    let runtime = Arc::clone(provider.memory().runtime().unwrap());
    let stream_id = runtime.stream_pool().acquire().unwrap();
    let stream = runtime.stream_pool().resolve(stream_id).unwrap();
    let domain = provider.bind_resident_execution_domain(runtime, stream_id, stream).unwrap();
    let mut graph = provider.allocate_semantic_hypergraph(&domain,
        SemanticHypergraphCapacities::try_new(2, 2, 2, 2).unwrap()).unwrap();
    let mut records = typed_records(symbol::intern("two statement entity"));
    let mut second = records.records[0].clone();
    second.arguments[1] = SemanticArgument::U64(20);
    let second_index = records.records.len() as u32;
    records.records.push(second);
    let admission = graph.admit_records(graph.empty_root(), records, admission_limits()).unwrap();
    let first = admission.statement_key(0).unwrap();
    let second = admission.statement_key(second_index).unwrap();
    let event = admission.support_event(0).unwrap();
    let fork = graph.fork(graph.empty_root()).unwrap();
    let first_support = match graph.insert_support(fork, &first, &event).unwrap() {
        SemanticInsertOutcome::Inserted(value) => value.support().identity(),
        _ => panic!("first support should be inserted"),
    };
    let second_support = match graph.insert_support(fork, &second, &event).unwrap() {
        SemanticInsertOutcome::Inserted(value) => value.support().identity(),
        _ => panic!("selected statement must get its own support identity"),
    };
    assert_ne!(first_support, second_support);
    for key in [first, second] {
        assert_eq!(graph.truth(SemanticView::Fork(fork), &key).unwrap(), SemanticTruth::True);
        assert!(matches!(graph.insert_support(fork, &key, &event).unwrap(), SemanticInsertOutcome::Unchanged(_)));
    }
    graph.discard(fork).unwrap();
}

const EMPTY_ROOT_BYTES: [u8; 64] = [
    0x58, 0x4c, 0x4f, 0x47, 0x53, 0x48, 0x47, 0x31, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x9a, 0xb9, 0x27, 0x84, 0x56, 0xdf, 0x6f, 0xfb,
    0xfe, 0xf9, 0x8b, 0x53, 0xca, 0xb7, 0x63, 0x72, 0xbd, 0xab, 0x2c, 0xd7, 0x41, 0x24, 0x60, 0xb2,
    0x88, 0x55, 0x82, 0x50, 0x93, 0x78, 0xcc, 0xd6, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

fn assert_stale(error: SemanticHypergraphError, expected: SemanticHandleKind) {
    assert!(
        matches!(
            error,
            SemanticHypergraphError::StaleGeneration { kind, .. } if kind == expected
        ),
        "expected stale {expected:?} handle, got {error:?}"
    );
}

#[test]
fn device_resident_statement_support_lifecycle_is_exact() {
    assert_eq!(
        SemanticTruth::from_presence(false, false),
        SemanticTruth::Neither
    );
    assert_eq!(
        SemanticTruth::from_presence(true, false),
        SemanticTruth::True
    );
    assert_eq!(
        SemanticTruth::from_presence(false, true),
        SemanticTruth::False
    );
    assert_eq!(
        SemanticTruth::from_presence(true, true),
        SemanticTruth::Both
    );

    if std::env::var("XLOG_REQUIRE_CUDA").as_deref() != Ok("1") {
        eprintln!("Skipping: set XLOG_REQUIRE_CUDA=1 to run this real-CUDA contract");
        return;
    }

    let provider = CudaProviderBuilder::new(0, MemoryBudget::with_limit(64 * 1024 * 1024))
        .with_stream_capacity(1)
        .build()
        .expect("CUDA provider must be available when XLOG_REQUIRE_CUDA=1");
    let provider = Arc::new(provider);
    let runtime = Arc::clone(
        provider
            .memory()
            .runtime()
            .expect("canonical provider must own a runtime"),
    );
    let stream_id = runtime
        .stream_pool()
        .acquire()
        .expect("semantic hypergraph must acquire its resident stream");
    let stream = runtime
        .stream_pool()
        .resolve(stream_id)
        .expect("resident stream id must resolve through its runtime");
    let domain = provider
        .bind_resident_execution_domain(Arc::clone(&runtime), stream_id, Arc::clone(&stream))
        .expect("provider must bind its exact resident execution domain");

    let capacities = SemanticHypergraphCapacities::try_new(2, 1, 2, 2).unwrap();
    let mut graph = provider
        .allocate_semantic_hypergraph(&domain, capacities)
        .expect("semantic storage initialization must execute on CUDA");
    let data_plane_before = provider.host_transfer_stats();
    let launch_metadata_before = provider.host_launch_metadata_transfer_stats();
    let admission = graph.admit_records(graph.empty_root(),
        typed_records(symbol::intern("lifecycle entity")), admission_limits()).unwrap();
    let key = admission.statement_key(0).unwrap();
    let pro = admission.support_event(0).unwrap();
    let contra = admission.support_event(1).unwrap();

    let base = graph.empty_root();
    let base_before = graph.snapshot(SemanticView::Root(base)).unwrap();
    assert_eq!(base_before.canonical_bytes(), &EMPTY_ROOT_BYTES);
    assert_eq!(base_before.digest().as_bytes(), &EMPTY_ROOT_BYTES[24..56]);
    assert_eq!(base_before.extents(), SemanticExtents::new(0, 0, 0));
    assert_eq!(
        graph.truth(SemanticView::Root(base), &key).unwrap(),
        SemanticTruth::Neither
    );

    let fork_a = graph.fork(base).unwrap();
    let launches_before_a = graph.execution_stats().cuda_kernel_launches();
    let inserted_a = match graph.insert_support(fork_a, &key, &pro).unwrap() {
        SemanticInsertOutcome::Inserted(inserted) => inserted,
        SemanticInsertOutcome::Unchanged(_) => panic!("first PRO support was reported unchanged"),
    };
    assert_eq!(inserted_a.previous_truth(), SemanticTruth::Neither);
    assert!(!inserted_a.changes_defined_truth());
    assert!(graph.execution_stats().cuda_kernel_launches() > launches_before_a);
    assert_eq!(
        graph.truth(SemanticView::Fork(fork_a), &key).unwrap(),
        SemanticTruth::True
    );
    graph.discard(fork_a).unwrap();
    assert_eq!(
        graph.snapshot(SemanticView::Root(base)).unwrap(),
        base_before
    );

    let stats_before_stale = graph.execution_stats();
    let launch_metadata_before_stale = provider.host_launch_metadata_transfer_stats();
    assert_stale(
        graph.snapshot(SemanticView::Fork(fork_a)).unwrap_err(),
        SemanticHandleKind::Fork,
    );
    assert_stale(
        graph
            .inspect_statement(SemanticView::Root(base), inserted_a.statement().handle())
            .unwrap_err(),
        SemanticHandleKind::Statement,
    );
    assert_stale(
        graph
            .inspect_support(SemanticView::Root(base), inserted_a.support().handle())
            .unwrap_err(),
        SemanticHandleKind::Support,
    );
    assert_stale(
        graph
            .inspect_version(SemanticView::Root(base), inserted_a.version().handle())
            .unwrap_err(),
        SemanticHandleKind::Version,
    );
    assert_eq!(graph.execution_stats(), stats_before_stale);
    let launch_metadata_after_stale = provider.host_launch_metadata_transfer_stats();
    assert_eq!(
        launch_metadata_after_stale.htod_calls,
        launch_metadata_before_stale.htod_calls
    );
    assert_eq!(
        launch_metadata_after_stale.htod_bytes,
        launch_metadata_before_stale.htod_bytes
    );

    let fork_b = graph.fork(base).unwrap();
    assert_eq!(fork_b.slot(), fork_a.slot());
    assert_ne!(fork_b.generation(), fork_a.generation());
    let inserted_b = match graph.insert_support(fork_b, &key, &pro).unwrap() {
        SemanticInsertOutcome::Inserted(inserted) => inserted,
        SemanticInsertOutcome::Unchanged(_) => panic!("discarded PRO support remained reachable"),
    };
    assert_eq!(inserted_b.previous_truth(), SemanticTruth::Neither);
    assert!(!inserted_b.changes_defined_truth());
    assert_eq!(
        inserted_b.statement().identity(),
        inserted_a.statement().identity()
    );
    assert_eq!(
        inserted_b.support().identity(),
        inserted_a.support().identity()
    );
    assert_eq!(
        inserted_b.version().identity(),
        inserted_a.version().identity()
    );
    assert_ne!(
        inserted_b.statement().handle().generation(),
        inserted_a.statement().handle().generation()
    );
    assert_ne!(
        inserted_b.support().handle().generation(),
        inserted_a.support().handle().generation()
    );
    assert_ne!(
        inserted_b.version().handle().generation(),
        inserted_a.version().handle().generation()
    );
    assert_eq!(
        graph.truth(SemanticView::Fork(fork_b), &key).unwrap(),
        SemanticTruth::True
    );

    let true_root = graph.seal(fork_b).unwrap();
    let true_before = graph.snapshot(SemanticView::Root(true_root)).unwrap();
    assert_eq!(true_before.extents(), SemanticExtents::new(1, 1, 1));
    assert_eq!(
        graph.snapshot(SemanticView::Root(base)).unwrap(),
        base_before
    );
    assert_eq!(
        graph.truth(SemanticView::Root(true_root), &key).unwrap(),
        SemanticTruth::True
    );

    let replay = graph.fork(true_root).unwrap();
    let replay_before = graph.snapshot(SemanticView::Fork(replay)).unwrap();
    let launches_before_replay = graph.execution_stats().cuda_kernel_launches();
    let unchanged = match graph.insert_support(replay, &key, &pro).unwrap() {
        SemanticInsertOutcome::Unchanged(version) => version,
        SemanticInsertOutcome::Inserted(_) => panic!("reachable PRO support was duplicated"),
    };
    assert!(graph.execution_stats().cuda_kernel_launches() > launches_before_replay);
    assert_eq!(unchanged, inserted_b.version());
    assert_eq!(
        graph.snapshot(SemanticView::Fork(replay)).unwrap(),
        replay_before
    );
    assert_eq!(
        graph.snapshot(SemanticView::Root(true_root)).unwrap(),
        true_before
    );
    graph.discard(replay).unwrap();

    let both_fork = graph.fork(true_root).unwrap();
    let launches_before_contra = graph.execution_stats().cuda_kernel_launches();
    let inserted_contra = match graph.insert_support(both_fork, &key, &contra).unwrap() {
        SemanticInsertOutcome::Inserted(inserted) => inserted,
        SemanticInsertOutcome::Unchanged(_) => panic!("new CONTRA support was reported unchanged"),
    };
    assert_eq!(inserted_contra.previous_truth(), SemanticTruth::True);
    assert!(inserted_contra.changes_defined_truth());
    assert!(graph.execution_stats().cuda_kernel_launches() > launches_before_contra);
    assert_eq!(
        graph.truth(SemanticView::Fork(both_fork), &key).unwrap(),
        SemanticTruth::Both
    );
    assert_eq!(
        graph
            .snapshot(SemanticView::Fork(both_fork))
            .unwrap()
            .extents(),
        SemanticExtents::new(1, 2, 2)
    );
    assert_eq!(
        graph
            .inspect_support(SemanticView::Fork(both_fork), inserted_b.support().handle(),)
            .unwrap(),
        inserted_b.support()
    );
    assert_eq!(
        graph
            .inspect_support(
                SemanticView::Fork(both_fork),
                inserted_contra.support().handle(),
            )
            .unwrap(),
        inserted_contra.support()
    );
    assert_eq!(
        graph.snapshot(SemanticView::Root(true_root)).unwrap(),
        true_before
    );
    assert_eq!(
        graph.truth(SemanticView::Root(true_root), &key).unwrap(),
        SemanticTruth::True
    );
    graph.discard(both_fork).unwrap();

    let data_plane_after = provider.host_transfer_stats();
    assert_eq!(data_plane_after.dtoh_bytes, data_plane_before.dtoh_bytes);
    assert_eq!(data_plane_after.htod_bytes, data_plane_before.htod_bytes);
    assert_eq!(data_plane_after.dtoh_calls, data_plane_before.dtoh_calls);
    assert_eq!(data_plane_after.htod_calls, data_plane_before.htod_calls);
    let launch_metadata_after = provider.host_launch_metadata_transfer_stats();
    assert!(launch_metadata_after.htod_calls > launch_metadata_before.htod_calls);
    assert!(launch_metadata_after.htod_bytes > launch_metadata_before.htod_bytes);
}

#[test]
fn support_edit_effects_distinguish_new_attachments_truth_changes_and_duplicates() {
    if std::env::var("XLOG_REQUIRE_CUDA").as_deref() != Ok("1") {
        return;
    }
    let provider = Arc::new(
        CudaProviderBuilder::new(0, MemoryBudget::with_limit(64 * 1024 * 1024))
            .with_stream_capacity(1)
            .build()
            .unwrap(),
    );
    let runtime = Arc::clone(provider.memory().runtime().unwrap());
    let stream_id = runtime.stream_pool().acquire().unwrap();
    let stream = runtime.stream_pool().resolve(stream_id).unwrap();
    let domain = provider
        .bind_resident_execution_domain(runtime, stream_id, stream)
        .unwrap();
    let mut graph = provider
        .allocate_semantic_hypergraph(
            &domain,
            SemanticHypergraphCapacities::try_new(2, 1, 4, 4).unwrap(),
        )
        .unwrap();
    let mut records = typed_records(symbol::intern("support effect entity"));
    let mut additional_pro = records.supports[0].clone();
    additional_pro.provenance = records.supports[1].provenance;
    records.supports.push(additional_pro);
    let mut additional_contra = records.supports[1].clone();
    additional_contra.provenance = records.supports[0].provenance;
    records.supports.push(additional_contra);
    let admission = graph
        .admit_records(graph.empty_root(), records, admission_limits())
        .unwrap();
    let key = admission.statement_key(0).unwrap();
    let events = [0, 2, 1, 3].map(|index| admission.support_event(index).unwrap());
    let fork = graph.fork(graph.empty_root()).unwrap();
    let mut first_version = None;
    for (index, (event, (previous, resulting, changes_defined_truth))) in events
        .iter()
        .zip([
            (SemanticTruth::Neither, SemanticTruth::True, false),
            (SemanticTruth::True, SemanticTruth::True, false),
            (SemanticTruth::True, SemanticTruth::Both, true),
            (SemanticTruth::Both, SemanticTruth::Both, false),
        ])
        .enumerate()
    {
        let before = graph.execution_stats().cuda_kernel_launches();
        let SemanticInsertOutcome::Inserted(inserted) =
            graph.insert_support(fork, &key, event).unwrap()
        else {
            panic!("distinct support event was reported unchanged");
        };
        assert_eq!(graph.execution_stats().cuda_kernel_launches(), before + 1);
        assert_eq!(inserted.previous_truth(), previous);
        assert_eq!(inserted.version().truth(), resulting);
        assert_eq!(inserted.changes_defined_truth(), changes_defined_truth);
        if index == 0 {
            first_version = Some(inserted.version());
        }
    }
    let snapshot = graph.snapshot(SemanticView::Fork(fork)).unwrap();
    assert_eq!(snapshot.extents(), SemanticExtents::new(1, 4, 4));
    let before = graph.execution_stats().cuda_kernel_launches();
    let duplicate = graph.insert_support(fork, &key, &events[0]).unwrap();
    assert_eq!(graph.execution_stats().cuda_kernel_launches(), before + 1);
    assert_eq!(
        duplicate,
        SemanticInsertOutcome::Unchanged(first_version.unwrap())
    );
    assert_eq!(graph.snapshot(SemanticView::Fork(fork)).unwrap(), snapshot);
    assert_eq!(
        graph.truth(SemanticView::Fork(fork), &key).unwrap(),
        SemanticTruth::Both
    );
    graph.discard(fork).unwrap();
}

#[test]
fn unchanged_seal_reuses_base_without_spending_root_slots() {
    if std::env::var("XLOG_REQUIRE_CUDA").as_deref() != Ok("1") {
        eprintln!("Skipping: set XLOG_REQUIRE_CUDA=1 to run this real-CUDA contract");
        return;
    }
    let provider = CudaProviderBuilder::new(0, MemoryBudget::with_limit(64 * 1024 * 1024))
        .with_stream_capacity(1)
        .build()
        .expect("CUDA provider must be available when XLOG_REQUIRE_CUDA=1");
    let provider = Arc::new(provider);
    let runtime = Arc::clone(provider.memory().runtime().unwrap());
    let stream_id = runtime.stream_pool().acquire().unwrap();
    let stream = runtime.stream_pool().resolve(stream_id).unwrap();
    let domain = provider
        .bind_resident_execution_domain(runtime, stream_id, stream)
        .unwrap();
    let mut graph = provider
        .allocate_semantic_hypergraph(
            &domain,
            SemanticHypergraphCapacities::try_new(2, 1, 1, 1).unwrap(),
        )
        .unwrap();
    let empty = graph.empty_root();
    let empty_snapshot = graph.snapshot(SemanticView::Root(empty)).unwrap();
    let no_edit = graph.fork(empty).unwrap();
    assert_eq!(graph.seal(no_edit).unwrap(), empty);
    assert_eq!(graph.snapshot(SemanticView::Root(empty)).unwrap(), empty_snapshot);
    assert_stale(
        graph.snapshot(SemanticView::Fork(no_edit)).unwrap_err(),
        SemanticHandleKind::Fork,
    );

    let admission = graph.admit_records(empty,
        typed_records(symbol::intern("repeated entity")), admission_limits()).unwrap();
    let key = admission.statement_key(0).unwrap();
    let event = admission.support_event(0).unwrap();
    let changed = graph.fork(empty).unwrap();
    assert!(matches!(
        graph.insert_support(changed, &key, &event).unwrap(),
        SemanticInsertOutcome::Inserted(_)
    ));
    assert!(matches!(
        graph.insert_support(changed, &key, &event).unwrap(),
        SemanticInsertOutcome::Unchanged(_)
    ));
    let root = graph.seal(changed).unwrap();
    assert_ne!(root, empty, "a later duplicate must not erase the first insertion");
    let snapshot = graph.snapshot(SemanticView::Root(root)).unwrap();
    assert_eq!(snapshot.extents(), SemanticExtents::new(1, 1, 1));
    for _ in 0..3 {
        let duplicate = graph.fork(root).unwrap();
        assert!(matches!(
            graph.insert_support(duplicate, &key, &event).unwrap(),
            SemanticInsertOutcome::Unchanged(_)
        ));
        assert_eq!(graph.seal(duplicate).unwrap(), root);
        assert_eq!(graph.snapshot(SemanticView::Root(root)).unwrap(), snapshot);
        assert_eq!(graph.truth(SemanticView::Root(root), &key).unwrap(), SemanticTruth::True);
    }
}
