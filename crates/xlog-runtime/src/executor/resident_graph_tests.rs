use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use xlog_core::{MemoryBudget, RelId, RuntimeConfig, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, InMemorySink, LogAction,
    LoggingResource, LoggingSink, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::{CudaBuffer, CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_ir::{ExecutionPlan, GeneratedQueryRuleProvenance};
use xlog_logic::Compiler;

use super::Executor;
use crate::resident_graph::{
    reset_resident_route_inspection_count, resident_route_inspection_count,
    ResidentGraphCertifiedPlan, ResidentGraphDeclineReason, ResidentGraphDeviceStatus,
    ResidentGraphDeviceStatusTestInjection, ResidentGraphExecutionError,
    ResidentGraphPrepareOptions, ResidentGraphRouteCertificate, ResidentGraphSchemaCatalog,
};

const RUNTIME_BUDGET: usize = 512 * 1024 * 1024;
const LOCAL_BUDGET: u64 = 512 * 1024 * 1024;

struct RuntimeFixture {
    provider: Arc<CudaKernelProvider>,
    runtime: Arc<XlogDeviceRuntime>,
    sink: Arc<InMemorySink>,
}

fn runtime_fixture() -> Option<RuntimeFixture> {
    runtime_fixture_with_local_budget(LOCAL_BUDGET)
}

fn runtime_fixture_with_local_budget(local_budget: u64) -> Option<RuntimeFixture> {
    let sink = Arc::new(InMemorySink::new());
    let mut budget = MemoryBudget::default();
    budget.device_bytes = local_budget;
    let provider = match xlog_cuda::CudaProviderBuilder::new(0, budget)
        .with_runtime_budget_limit(RUNTIME_BUDGET as u64)
        .with_logging_sink(sink.clone() as Arc<dyn LoggingSink>)
        .build()
    {
        Ok(provider) => Arc::new(provider),
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA setup failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA unavailable: {error}");
            return None;
        }
    };
    let runtime = Arc::clone(
        provider
            .memory()
            .runtime()
            .expect("builder provider owns a runtime"),
    );
    Some(RuntimeFixture {
        provider,
        runtime,
        sink,
    })
}

struct AuthoredPlan {
    executor: Executor,
    plan: ExecutionPlan,
    schemas: HashMap<String, Schema>,
    rel_ids: HashMap<String, RelId>,
}

fn compile_authored_plan(
    provider: Arc<CudaKernelProvider>,
    source: &str,
    config: RuntimeConfig,
) -> AuthoredPlan {
    let mut compiler = Compiler::new();
    let plan = compiler
        .compile(source)
        .expect("authored program must compile");
    let schemas = compiler.schemas().clone();
    let rel_ids = compiler.rel_ids().clone();
    let mut executor = Executor::new_with_config(provider, config);
    let mut registrations = rel_ids.iter().collect::<Vec<_>>();
    registrations.sort_by_key(|(name, rel)| (rel.0, name.as_str()));
    for (name, rel) in registrations {
        executor.register_relation(*rel, name);
    }
    AuthoredPlan {
        executor,
        plan,
        schemas,
        rel_ids,
    }
}

fn catalog(plan: &AuthoredPlan) -> ResidentGraphSchemaCatalog {
    ResidentGraphSchemaCatalog::from_named_schemas(plan.rel_ids.iter().filter_map(|(name, rel)| {
        plan.schemas
            .get(name)
            .cloned()
            .map(|schema| (name.clone(), *rel, schema))
    }))
}

fn put_u32_columns(plan: &mut AuthoredPlan, name: &str, columns: &[&[u32]]) {
    let schema = plan
        .schemas
        .get(name)
        .unwrap_or_else(|| panic!("missing schema for {name}"))
        .clone();
    let uploaded = plan
        .executor
        .provider
        .create_buffer_from_u32_columns(columns, schema)
        .unwrap_or_else(|error| panic!("failed to upload {name}: {error}"));
    let full_row_keys = (0..uploaded.schema().arity()).collect::<Vec<_>>();
    let buffer = plan
        .executor
        .provider
        .dedup(&uploaded, &full_row_keys)
        .unwrap_or_else(|error| panic!("failed to certify {name} as a full-row set: {error}"));
    assert!(buffer.canonical_full_row_set_certified());
    plan.executor.put_relation(name, buffer);
}

fn put_uncertified_u32_columns(plan: &mut AuthoredPlan, name: &str, columns: &[&[u32]]) {
    let schema = plan
        .schemas
        .get(name)
        .unwrap_or_else(|| panic!("missing schema for {name}"))
        .clone();
    let buffer = plan
        .executor
        .provider
        .create_buffer_from_u32_columns(columns, schema)
        .unwrap_or_else(|error| panic!("failed to upload {name}: {error}"));
    assert!(!buffer.canonical_full_row_set_certified());
    plan.executor.put_relation(name, buffer);
}

fn recursive_program() -> &'static str {
    r#"
        pred edge(u32, u32).
        pred seed(u32).
        pred gate(u32).
        pred stable(u32).
        pred reach(u32).
        pred selected(u32).
        reach(X) :- seed(X).
        reach(Y) :- reach(X), edge(Y, X).
        selected(X) :- reach(X), gate(X), X > 1.
        ?- selected(X).
    "#
}

fn sequential_filter_program() -> &'static str {
    r#"
        pred input(u32).
        pred positive(u32).
        pred bounded(u32).
        positive(X) :- input(X), X > 0.
        bounded(X) :- positive(X), X < 10.
        ?- bounded(X).
    "#
}

fn wide_intermediate_program() -> &'static str {
    r#"
        pred left(u32, u32, u32, u32, u32, u32, u32, u32).
        pred right(u32, u32, u32, u32, u32, u32, u32, u32, u32).
        pred projected(u32).
        projected(A) :- left(A, B, C, D, E, F, G, H), right(A, I, J, K, L, M, N, O, P).
        ?- projected(A).
    "#
}

fn sibling_filter_join_program() -> &'static str {
    r#"
        pred left(u32, u32).
        pred right(u32, u32).
        pred output(u32).
        output(X) :- left(X, L), L > 0, right(X, R), R < 10.
        ?- output(X).
    "#
}

fn direct_projection_program() -> &'static str {
    r#"
        pred input(u32, u32).
        pred projected(u32).
        projected(X) :- input(X, Y).
        ?- projected(X).
    "#
}

fn seeded_recursive_plan(provider: Arc<CudaKernelProvider>, max_iterations: u32) -> AuthoredPlan {
    seeded_recursive_plan_with_gate(provider, max_iterations, &[2, 3, 4])
}

fn seeded_recursive_plan_with_gate(
    provider: Arc<CudaKernelProvider>,
    max_iterations: u32,
    gate: &[u32],
) -> AuthoredPlan {
    let mut config = RuntimeConfig::default();
    config.max_iterations = max_iterations;
    config.profile = true;
    let mut authored = compile_authored_plan(provider, recursive_program(), config);
    put_u32_columns(&mut authored, "edge", &[&[2, 3, 4], &[1, 2, 3]]);
    put_u32_columns(&mut authored, "seed", &[&[1]]);
    put_u32_columns(&mut authored, "gate", &[gate]);
    authored
}

fn certified(plan: &AuthoredPlan) -> ResidentGraphRouteCertificate {
    ResidentGraphRouteCertificate::inspect(&plan.plan, &catalog(plan))
        .expect("authored route must certify")
}

fn sealed(plan: &AuthoredPlan) -> ResidentGraphCertifiedPlan {
    ResidentGraphCertifiedPlan::inspect(Arc::new(plan.plan.clone()), &catalog(plan))
        .expect("authored route must seal")
}

#[test]
fn sealed_prepare_skips_reinspection_and_public_prepare_validates_once_before_allocation() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let authored = seeded_recursive_plan(fixture.provider.clone(), 16);

    reset_resident_route_inspection_count();
    let certified_plan = sealed(&authored);
    assert_eq!(resident_route_inspection_count(), 1);
    let prepared = authored
        .executor
        .prepare_certified_resident_graph(&certified_plan, ResidentGraphPrepareOptions::default())
        .expect("sealed plan must prepare");
    assert_eq!(resident_route_inspection_count(), 1);
    drop(prepared);
    fixture
        .provider
        .device()
        .synchronize()
        .expect("cleanup sync");
    fixture.runtime.reap_pending().expect("cleanup reap");

    let certificate = certified(&authored);
    let mut mismatched = authored.plan.clone();
    mismatched.est_memory_peak += 1;
    reset_resident_route_inspection_count();
    fixture.provider.memory().reset_alloc_count();
    let result = authored.executor.prepare_resident_graph(
        &mismatched,
        &certificate,
        ResidentGraphPrepareOptions::default(),
    );
    assert!(matches!(
        result,
        Err(ResidentGraphExecutionError::Declined(
            ResidentGraphDeclineReason::WorkspaceUnbounded { .. }
        ))
    ));
    assert_eq!(resident_route_inspection_count(), 1);
    assert_eq!(fixture.provider.memory().alloc_count(), 0);

    let unsupported_catalog = ResidentGraphSchemaCatalog::from_named_schemas(
        authored.rel_ids.iter().map(|(name, relation)| {
            let schema = if name == "edge" {
                Schema::new(
                    (0..18)
                        .map(|index| (format!("column_{index}"), ScalarType::U32))
                        .collect(),
                )
            } else {
                authored.schemas[name].clone()
            };
            (name.clone(), *relation, schema)
        }),
    );
    let unsupported = ResidentGraphRouteCertificate::inspect(&authored.plan, &unsupported_catalog)
        .expect("unsupported route inspection");
    assert!(!unsupported.is_supported());
    reset_resident_route_inspection_count();
    fixture.provider.memory().reset_alloc_count();
    let result = authored.executor.prepare_resident_graph(
        &authored.plan,
        &unsupported,
        ResidentGraphPrepareOptions::default(),
    );
    assert!(matches!(
        result,
        Err(ResidentGraphExecutionError::Declined(_))
    ));
    assert_eq!(resident_route_inspection_count(), 1);
    assert_eq!(fixture.provider.memory().alloc_count(), 0);
}

#[test]
fn sealed_prepare_rechecks_store_provider_and_options_for_each_executor() {
    let Some(first_fixture) = runtime_fixture() else {
        return;
    };
    let Some(second_fixture) = runtime_fixture() else {
        return;
    };
    let first = seeded_recursive_plan(first_fixture.provider.clone(), 1);
    let certified_plan = sealed(&first);
    let first_prepared = first
        .executor
        .prepare_certified_resident_graph(&certified_plan, ResidentGraphPrepareOptions::default())
        .expect("first provider setup");
    let first_capacity = first_prepared.preflight_report().relation_capacity;
    drop(first_prepared);
    first_fixture
        .provider
        .device()
        .synchronize()
        .expect("first cleanup sync");
    first_fixture
        .runtime
        .reap_pending()
        .expect("first cleanup reap");

    let mut config = RuntimeConfig::default();
    config.max_iterations = 1;
    config.profile = true;
    let mut second =
        compile_authored_plan(second_fixture.provider.clone(), recursive_program(), config);
    second_fixture.provider.memory().reset_alloc_count();
    let error = match second
        .executor
        .prepare_certified_resident_graph(&certified_plan, ResidentGraphPrepareOptions::default())
    {
        Ok(_) => panic!("missing dynamic inputs must reject resident preparation"),
        Err(error) => error,
    };
    assert!(error
        .to_string()
        .contains("missing resident source relation"));
    assert_eq!(second_fixture.provider.memory().alloc_count(), 0);

    let destinations = (2..=66).collect::<Vec<_>>();
    let sources = (1..=65).collect::<Vec<_>>();
    put_u32_columns(
        &mut second,
        "edge",
        &[destinations.as_slice(), sources.as_slice()],
    );
    put_u32_columns(&mut second, "seed", &[&[1]]);
    put_u32_columns(&mut second, "gate", &[destinations.as_slice()]);
    first_fixture.provider.memory().reset_alloc_count();
    second_fixture.provider.memory().reset_alloc_count();
    let prepared = second
        .executor
        .prepare_certified_resident_graph(&certified_plan, ResidentGraphPrepareOptions::default())
        .expect("the same sealed plan must readmit current inputs on the current provider");
    assert!(prepared.preflight_report().relation_capacity > first_capacity);
    assert_eq!(first_fixture.provider.memory().alloc_count(), 0);
    assert!(second_fixture.provider.memory().alloc_count() > 0);
    drop(prepared);
    second_fixture
        .provider
        .device()
        .synchronize()
        .expect("cleanup sync");
    second_fixture.runtime.reap_pending().expect("cleanup reap");
}

fn assert_source_set_decline_before_allocation_or_launch(
    fixture: &RuntimeFixture,
    authored: &AuthoredPlan,
    relation: &str,
) {
    reset_independent_recorders(fixture);
    fixture.provider.memory().reset_alloc_count();
    let allocation_before = fixture.provider.memory().allocated_bytes();
    let graph_before = fixture.runtime.conditional_graph_stats();
    let certificate = certified(authored);
    let result = authored.executor.prepare_resident_graph(
        &authored.plan,
        &certificate,
        ResidentGraphPrepareOptions::default(),
    );
    match result {
        Err(ResidentGraphExecutionError::Declined(
            ResidentGraphDeclineReason::SourceSetUncertified {
                relation: declined_relation,
            },
        )) => assert_eq!(declined_relation, relation),
        Err(other) => panic!("expected exact source-set decline, got {other:?}"),
        Ok(_) => panic!("uncertified source unexpectedly prepared a resident graph"),
    }
    assert_eq!(
        fixture.provider.memory().allocated_bytes(),
        allocation_before
    );
    assert_eq!(fixture.provider.memory().alloc_count(), 0);
    let transfers = fixture.provider.host_transfer_stats();
    assert_eq!(transfers.htod_calls, 0);
    assert_eq!(transfers.dtoh_calls, 0);
    assert_eq!(fixture.provider.d2h_transfer_count(), 0);
    assert_eq!(fixture.provider.untracked_metadata_dtoh_count(), 0);
    let graph_after = fixture.runtime.conditional_graph_stats();
    assert_eq!(graph_after.launches, graph_before.launches);
    assert_eq!(
        graph_after.terminal_synchronizations,
        graph_before.terminal_synchronizations
    );
}

#[test]
fn generated_query_provenance_cannot_reclassify_an_authored_rule() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let mut authored = seeded_recursive_plan(fixture.provider.clone(), 8);
    let (scc_index, rule_index, authored_head) = authored
        .plan
        .rules_by_scc
        .iter()
        .enumerate()
        .find_map(|(scc_index, rules)| rules.first().map(|rule| (scc_index, 0, rule.head.clone())))
        .expect("authored plan should contain a rule");
    authored.plan.generated_query_rules = vec![GeneratedQueryRuleProvenance {
        query_index: 0,
        scc_index,
        rule_index,
    }];
    reset_independent_recorders(&fixture);
    fixture.provider.memory().reset_alloc_count();
    let allocation_before = fixture.provider.memory().allocated_bytes();
    let graph_before = fixture.runtime.conditional_graph_stats();
    let result = authored.executor.prepare_resident_graph(
        &authored.plan,
        &certified(&authored),
        ResidentGraphPrepareOptions::default(),
    );
    match result {
        Err(ResidentGraphExecutionError::Declined(
            ResidentGraphDeclineReason::WorkspaceUnbounded { detail },
        )) => assert_eq!(
            detail,
            format!(
                "generated query provenance 0 expects head __xlog_query_0 but references authored head {authored_head}"
            )
        ),
        Err(other) => panic!("expected generated-query provenance decline, got {other:?}"),
        Ok(_) => panic!("authored rule was reclassified as a generated query"),
    }
    assert_eq!(
        fixture.provider.memory().allocated_bytes(),
        allocation_before
    );
    assert_eq!(fixture.provider.memory().alloc_count(), 0);
    assert!(fixture.sink.snapshot().is_empty());
    let graph_after = fixture.runtime.conditional_graph_stats();
    assert_eq!(graph_after.launches, graph_before.launches);
    assert_eq!(
        graph_after.terminal_synchronizations,
        graph_before.terminal_synchronizations
    );
}

fn reset_independent_recorders(fixture: &RuntimeFixture) {
    fixture.provider.reset_host_transfer_stats();
    fixture.provider.reset_d2h_transfer_count();
    fixture.provider.reset_untracked_metadata_dtoh_count();
    fixture.provider.reset_deterministic_d2h_violations();
    fixture.provider.reset_final_observation_transfer_stats();
    fixture.runtime.reset_conditional_graph_stats();
    fixture.sink.clear();
}

#[test]
fn source_domain_bound_explosion_declines_before_graph_launch() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let mut authored = compile_authored_plan(
        fixture.provider.clone(),
        r#"
            pred left(u32, u32).
            pred right(u32, u32).
            pred combined(u32, u32, u32).
            combined(K, X, Y) :- left(K, X), right(K, Y).
            ?- combined(K, X, Y).
        "#,
        RuntimeConfig::default(),
    );
    let keys = vec![1u32; 257];
    let values = (0u32..257).collect::<Vec<_>>();
    put_u32_columns(&mut authored, "left", &[&keys, &values]);
    put_u32_columns(&mut authored, "right", &[&keys, &values]);
    let certificate = certified(&authored);
    let before = authored
        .executor
        .provider
        .memory()
        .runtime()
        .expect("runtime-backed provider")
        .conditional_graph_stats();

    let error = match authored.executor.prepare_resident_graph(
        &authored.plan,
        &certificate,
        ResidentGraphPrepareOptions::default(),
    ) {
        Ok(_) => panic!("an unreservable join bound must decline before graph construction"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        ResidentGraphExecutionError::Declined(
            ResidentGraphDeclineReason::WorkspaceUnbounded { .. }
        )
    ));
    let after = authored
        .executor
        .provider
        .memory()
        .runtime()
        .expect("runtime-backed provider")
        .conditional_graph_stats();
    assert_eq!(after.launches, before.launches);
    assert_eq!(
        after.terminal_synchronizations,
        before.terminal_synchronizations
    );
}

#[test]
fn uncertified_source_set_declines_before_graph_allocation_or_launch() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let mut authored = compile_authored_plan(
        fixture.provider.clone(),
        r#"
            pred input(u32).
            pred output(u32).
            output(X) :- input(X).
            ?- output(X).
        "#,
        RuntimeConfig::default(),
    );
    put_uncertified_u32_columns(&mut authored, "input", &[&[1, 2]]);
    assert_source_set_decline_before_allocation_or_launch(&fixture, &authored, "input");
}

#[test]
fn certified_source_from_another_runtime_declines_before_graph_allocation_or_launch() {
    let Some(local) = runtime_fixture() else {
        return;
    };
    let Some(foreign) = runtime_fixture() else {
        return;
    };
    let mut authored = compile_authored_plan(
        local.provider.clone(),
        r#"
            pred input(u32).
            pred output(u32).
            output(X) :- input(X).
            ?- output(X).
        "#,
        RuntimeConfig::default(),
    );
    let schema = authored.schemas["input"].clone();
    let uploaded = foreign
        .provider
        .create_buffer_from_u32_columns(&[&[1, 2]], schema)
        .expect("foreign upload");
    let foreign_buffer = foreign
        .provider
        .dedup(&uploaded, &[0])
        .expect("foreign full-row dedup");
    assert!(foreign_buffer.canonical_full_row_set_certified());
    authored.executor.put_relation("input", foreign_buffer);

    reset_independent_recorders(&local);
    local.provider.memory().reset_alloc_count();
    let allocation_before = local.provider.memory().allocated_bytes();
    let graph_before = local.runtime.conditional_graph_stats();
    let certificate = certified(&authored);
    let result = authored.executor.prepare_resident_graph(
        &authored.plan,
        &certificate,
        ResidentGraphPrepareOptions::default(),
    );
    match result {
        Err(ResidentGraphExecutionError::Declined(
            ResidentGraphDeclineReason::SourceSetUncertified { relation },
        )) => assert_eq!(relation, "input column 0 belongs to another memory manager"),
        Err(other) => panic!("expected exact foreign-source decline, got {other:?}"),
        Ok(_) => panic!("foreign certified source unexpectedly prepared a resident graph"),
    }
    assert_eq!(local.provider.memory().allocated_bytes(), allocation_before);
    assert_eq!(local.provider.memory().alloc_count(), 0);
    let transfers = local.provider.host_transfer_stats();
    assert_eq!(transfers.htod_calls, 0);
    assert_eq!(transfers.dtoh_calls, 0);
    assert_eq!(local.provider.d2h_transfer_count(), 0);
    assert_eq!(local.provider.untracked_metadata_dtoh_count(), 0);
    let graph_after = local.runtime.conditional_graph_stats();
    assert_eq!(graph_after.launches, graph_before.launches);
    assert_eq!(
        graph_after.terminal_synchronizations,
        graph_before.terminal_synchronizations
    );
}

#[test]
fn public_singleton_constructor_cannot_forge_a_resident_source_set_proof() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let mut authored = compile_authored_plan(
        fixture.provider.clone(),
        r#"
            pred input(u32).
            pred output(u32).
            output(X) :- input(X).
            ?- output(X).
        "#,
        RuntimeConfig::default(),
    );
    let schema = authored.schemas["input"].clone();
    let mut column = fixture.provider.memory().alloc::<u8>(4).expect("column");
    fixture
        .provider
        .device()
        .inner()
        .htod_sync_copy_into(&7u32.to_ne_bytes(), &mut column)
        .expect("column upload");
    let mut device_count = fixture.provider.memory().alloc::<u32>(1).expect("count");
    fixture
        .provider
        .device()
        .inner()
        .htod_sync_copy_into(&[2], &mut device_count)
        .expect("count upload");
    let forged = CudaBuffer::from_columns(vec![column.into()], 1, device_count, schema);
    authored.executor.put_relation("input", forged);

    assert_source_set_decline_before_allocation_or_launch(&fixture, &authored, "input");
}

#[test]
fn adjacency_dedup_of_unsorted_input_cannot_mint_a_resident_source_set_proof() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let mut authored = compile_authored_plan(
        fixture.provider.clone(),
        r#"
            pred input(u32).
            pred output(u32).
            output(X) :- input(X).
            ?- output(X).
        "#,
        RuntimeConfig::default(),
    );
    let schema = authored.schemas["input"].clone();
    let bag = fixture
        .provider
        .create_buffer_from_u32_columns(&[&[2, 1, 2]], schema)
        .expect("bag upload");
    let adjacency_only = fixture
        .provider
        .dedup_sorted(&bag, &[0])
        .expect("adjacency-only dedup");
    authored.executor.put_relation("input", adjacency_only);

    assert_source_set_decline_before_allocation_or_launch(&fixture, &authored, "input");
}

#[test]
fn full_row_dedup_proof_survives_exact_clone_and_public_mutation_invalidates_it() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let mut authored = compile_authored_plan(
        fixture.provider.clone(),
        r#"
            pred input(u32).
            pred output(u32).
            output(X) :- input(X).
            ?- output(X).
        "#,
        RuntimeConfig::default(),
    );
    let schema = authored.schemas["input"].clone();
    let bag = fixture
        .provider
        .create_buffer_from_u32_columns(&[&[2, 1, 2]], schema)
        .expect("bag upload");
    let canonical = fixture.provider.dedup(&bag, &[0]).expect("full-row dedup");
    assert!(canonical.canonical_full_row_set_certified());
    let cloned = fixture
        .provider
        .clone_buffer(&canonical)
        .expect("exact clone");
    assert!(cloned.canonical_full_row_set_certified());
    let mut mutated = fixture
        .provider
        .clone_buffer(&canonical)
        .expect("mutable clone");
    let _ = mutated.columns_mut();
    assert!(!mutated.canonical_full_row_set_certified());
    authored.executor.put_relation("input", cloned);

    authored
        .executor
        .prepare_resident_graph(
            &authored.plan,
            &certified(&authored),
            ResidentGraphPrepareOptions::default(),
        )
        .expect("real full-row set proof must admit")
        .launch()
        .expect("launch")
        .synchronize_core()
        .expect("sync")
        .observe_final_receipt()
        .expect("receipt")
        .commit(&mut authored.executor)
        .expect("commit");
    assert_eq!(
        download_u32_relation(&authored.executor, "output"),
        vec![vec![1, 2]]
    );
}

#[test]
fn clear_and_reinsert_aba_rejects_observed_transaction_before_commit() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let mut authored = seeded_recursive_plan(fixture.provider, 16);
    let observed = authored
        .executor
        .prepare_resident_graph(
            &authored.plan,
            &certified(&authored),
            ResidentGraphPrepareOptions::default(),
        )
        .expect("prepare")
        .launch()
        .expect("launch")
        .synchronize_core()
        .expect("sync")
        .observe_final_receipt()
        .expect("receipt");

    authored.executor.store_mut().clear();
    put_u32_columns(&mut authored, "edge", &[&[1, 2, 3], &[2, 3, 4]]);
    put_u32_columns(&mut authored, "seed", &[&[1]]);
    put_u32_columns(&mut authored, "gate", &[&[2, 3, 4]]);
    let error = observed
        .commit(&mut authored.executor)
        .expect_err("clear and reinsert must invalidate the optimistic transaction");
    assert_eq!(
        error,
        ResidentGraphExecutionError::Runtime(
            "resident transaction became stale before commit".into()
        )
    );
    assert!(!authored.executor.store().contains("reach"));
    assert!(!authored.executor.store().contains("selected"));
}

#[test]
fn stale_source_epoch_is_rejected_before_graph_enqueue() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let authored = seeded_recursive_plan(fixture.provider.clone(), 16);
    let mut prepared = authored
        .executor
        .prepare_resident_graph(
            &authored.plan,
            &certified(&authored),
            ResidentGraphPrepareOptions::default(),
        )
        .expect("prepare");
    let graph_before = fixture.runtime.conditional_graph_stats();
    prepared.invalidate_expected_source_epoch();
    match prepared.launch() {
        Err(ResidentGraphExecutionError::Declined(
            ResidentGraphDeclineReason::SourceSetUncertified { relation },
        )) => assert_eq!(
            relation,
            "relation store changed after resident preparation"
        ),
        Err(other) => panic!("expected stale source-set decline, got {other:?}"),
        Ok(_) => panic!("stale source epoch unexpectedly launched a graph"),
    }
    let graph_after = fixture.runtime.conditional_graph_stats();
    assert_eq!(graph_after.launches, graph_before.launches);
    assert_eq!(
        graph_after.terminal_synchronizations,
        graph_before.terminal_synchronizations
    );
}

fn assert_core_recorders_are_zero(fixture: &RuntimeFixture, allocations_before: u64) {
    let transfers = fixture.provider.host_transfer_stats();
    assert_eq!(transfers.htod_calls, 0);
    assert_eq!(transfers.htod_bytes, 0);
    assert_eq!(transfers.dtoh_calls, 0);
    assert_eq!(transfers.dtoh_bytes, 0);
    assert_eq!(fixture.provider.d2h_transfer_count(), 0);
    assert_eq!(fixture.provider.untracked_metadata_dtoh_count(), 0);
    assert_eq!(fixture.provider.deterministic_d2h_violation_count(), 0);
    assert_eq!(fixture.provider.memory().alloc_count(), allocations_before);
    assert_eq!(
        fixture
            .provider
            .final_observation_transfer_stats()
            .dtoh_calls,
        0
    );
    assert!(
        !fixture
            .sink
            .snapshot()
            .iter()
            .any(|record| record.action == LogAction::Allocate),
        "the conditional core must not allocate through the runtime resource"
    );
    let graph = fixture.runtime.conditional_graph_stats();
    assert_eq!(graph.launches, 1);
    assert_eq!(graph.terminal_synchronizations, 1);
    assert_eq!(graph.host_iterations, 0);
    assert_eq!(graph.host_allocations, 0);
}

fn download_u32_relation(executor: &Executor, name: &str) -> Vec<Vec<u32>> {
    let buffer = executor
        .store()
        .get(name)
        .unwrap_or_else(|| panic!("missing relation {name}"));
    (0..buffer.schema().arity())
        .map(|column| {
            executor
                .provider
                .download_column::<u32>(buffer, column)
                .unwrap_or_else(|error| panic!("download {name}[{column}] failed: {error}"))
        })
        .collect()
}

fn relation_snapshot(
    executor: &Executor,
    names: &[&str],
) -> BTreeMap<String, (u64, Vec<Vec<u32>>)> {
    names
        .iter()
        .map(|name| {
            let version = executor
                .store()
                .version(name)
                .unwrap_or_else(|| panic!("missing version for {name}"));
            (
                (*name).to_string(),
                (version, download_u32_relation(executor, name)),
            )
        })
        .collect()
}

#[test]
fn authored_recursive_executor_uses_zero_transfer_conditional_core_and_one_final_receipt() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    for gate in [&[2, 3, 4][..], &[][..]] {
        let mut baseline = seeded_recursive_plan_with_gate(fixture.provider.clone(), 16, gate);
        baseline
            .executor
            .execute_plan(&baseline.plan)
            .expect("existing GPU executor must run the authored RIR");
        let expected = download_u32_relation(&baseline.executor, "selected");

        let mut authored = seeded_recursive_plan_with_gate(fixture.provider.clone(), 16, gate);
        let certificate = certified(&authored);
        let route_debug = format!("{:#?}", authored.plan);
        for required in ["Scan", "Filter", "Project", "Join"] {
            assert!(
                route_debug.contains(required),
                "authored RIR must contain {required}"
            );
        }
        fixture.sink.clear();
        let prepared = authored
            .executor
            .prepare_resident_graph(
                &authored.plan,
                &certificate,
                ResidentGraphPrepareOptions::default(),
            )
            .expect("resident setup must finish before the measured core");
        let report = prepared.preflight_report();
        assert_eq!(
            report.estimated_required_bytes,
            report.tracked_device_allocation_bytes,
            "the pre-allocation manifest must exactly cover every tracked prepared allocation; private_relations={} scratch_slots={}",
            report.private_relation_slots,
            report.scratch_slots,
        );
        assert_eq!(
            report.estimated_required_bytes,
            report.relation_device_bytes
                + report.filter_descriptor_device_bytes
                + report.filter_scratch_device_bytes
                + report.project_descriptor_device_bytes
                + report.fixed_workspace_device_bytes,
        );
        assert_eq!(
            report.private_relation_slots,
            report.permanent_relation_slots as usize + report.scratch_slots as usize,
        );
        assert!(
            report.logical_relation_values > report.private_relation_slots,
            "sequential sibling values must reuse physical slots"
        );
        assert_eq!(report.filter_scratch_allocations, 1);
        assert!(report.filter_descriptor_device_bytes > 0);
        assert!(
            fixture
                .sink
                .snapshot()
                .iter()
                .any(|record| record.action == LogAction::Allocate),
            "the private graph workspace must be allocated during setup"
        );

        reset_independent_recorders(&fixture);
        fixture.provider.memory().reset_alloc_count();
        let allocations_before = fixture.provider.memory().alloc_count();
        let synchronized = prepared
            .launch()
            .expect("real conditional graph launch")
            .synchronize_core()
            .expect("one terminal synchronization");
        assert_core_recorders_are_zero(&fixture, allocations_before);

        let receipt = synchronized
            .observe_final_receipt()
            .expect("bounded final receipt observation");
        assert_eq!(fixture.provider.host_transfer_stats().dtoh_calls, 0);
        assert_eq!(fixture.provider.d2h_transfer_count(), 0);
        assert_eq!(fixture.provider.untracked_metadata_dtoh_count(), 0);
        let final_observation = fixture.provider.final_observation_transfer_stats();
        assert_eq!(final_observation.dtoh_calls, 1);
        assert_eq!(final_observation.pinned_receipts, 1);
        assert_eq!(final_observation.dtoh_bytes, receipt.encoded_len() as u64);
        assert!(
            receipt.iterations() > 1,
            "authored recursion must exercise convergence"
        );
        assert!(
            receipt.device_elapsed_ns() > 0,
            "resident profiling must use real CUDA-event elapsed time"
        );
        assert!(receipt.device_scan_invocations() > 0);
        assert!(receipt.device_filter_invocations() > 0);

        receipt
            .commit(&mut authored.executor)
            .expect("successful receipt must atomically commit staged results");
        assert_eq!(
            download_u32_relation(&authored.executor, "selected"),
            expected
        );
    }
}

#[test]
fn schema_winner_matches_first_nonempty_installation_order() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    struct Case {
        name: &'static str,
        first_rows: &'static [u32],
        second_rows: &'static [u32],
        existing_rows: Option<&'static [u32]>,
        expected_label: &'static str,
    }
    let cases = [
        Case {
            name: "all empty without an existing head",
            first_rows: &[],
            second_rows: &[],
            existing_rows: None,
            expected_label: "first_value",
        },
        Case {
            name: "all empty with an existing empty head",
            first_rows: &[],
            second_rows: &[],
            existing_rows: Some(&[]),
            expected_label: "existing_value",
        },
        Case {
            name: "existing empty then a nonempty contribution",
            first_rows: &[],
            second_rows: &[2],
            existing_rows: Some(&[]),
            expected_label: "second_value",
        },
        Case {
            name: "existing nonempty precedes later contributions",
            first_rows: &[1],
            second_rows: &[2],
            existing_rows: Some(&[9]),
            expected_label: "existing_value",
        },
        Case {
            name: "first empty then second nonempty",
            first_rows: &[],
            second_rows: &[2],
            existing_rows: None,
            expected_label: "second_value",
        },
        Case {
            name: "first and later contributions are nonempty",
            first_rows: &[1],
            second_rows: &[2],
            existing_rows: None,
            expected_label: "first_value",
        },
    ];

    for case in cases {
        let mut authored = compile_authored_plan(
            fixture.provider.clone(),
            r#"
                pred first(u32).
                pred second(u32).
                pred output(u32).
                output(X) :- first(X).
                output(X) :- second(X).
            "#,
            RuntimeConfig::default(),
        );
        let make_set = |label: &str, rows: &[u32]| {
            let schema = Schema::new(vec![(label.into(), ScalarType::U32)]);
            if rows.is_empty() {
                fixture
                    .provider
                    .create_empty_buffer(schema)
                    .expect("empty schema-winner source")
            } else {
                let uploaded = fixture
                    .provider
                    .create_buffer_from_slice(rows, schema)
                    .expect("schema-winner source upload");
                fixture
                    .provider
                    .dedup(&uploaded, &[0])
                    .expect("schema-winner source certification")
            }
        };
        authored.schemas.insert(
            "first".into(),
            Schema::new(vec![("first_value".into(), ScalarType::U32)]),
        );
        authored.schemas.insert(
            "second".into(),
            Schema::new(vec![("second_value".into(), ScalarType::U32)]),
        );
        authored
            .executor
            .put_relation("first", make_set("first_value", case.first_rows));
        authored
            .executor
            .put_relation("second", make_set("second_value", case.second_rows));
        if let Some(existing_rows) = case.existing_rows {
            authored
                .executor
                .put_relation("output", make_set("existing_value", existing_rows));
        }

        let certificate = certified(&authored);
        let receipt = authored
            .executor
            .prepare_resident_graph(
                &authored.plan,
                &certificate,
                ResidentGraphPrepareOptions::default(),
            )
            .unwrap_or_else(|error| panic!("{} prepare failed: {error:?}", case.name))
            .launch()
            .unwrap_or_else(|error| panic!("{} launch failed: {error:?}", case.name))
            .synchronize_core()
            .unwrap_or_else(|error| panic!("{} synchronization failed: {error:?}", case.name))
            .observe_final_receipt()
            .unwrap_or_else(|error| panic!("{} observation failed: {error:?}", case.name));
        receipt
            .commit(&mut authored.executor)
            .unwrap_or_else(|error| panic!("{} commit failed: {error:?}", case.name));
        assert_eq!(
            authored
                .executor
                .store()
                .get("output")
                .expect("resident output")
                .schema()
                .columns
                .first()
                .map(|(name, _)| name.as_str()),
            Some(case.expected_label),
            "{}",
            case.name,
        );
    }
}

#[test]
fn schema_equations_accept_stable_collapse_and_acyclic_lineage_but_decline_ambiguity() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let source = r#"
            pred seed(item: u32).
            pred path(item: u32, category: u32).
            path(X, 1) :- seed(X).
            path(X, 2) :- path(X, 1).
        "#;
    let config = RuntimeConfig::default();
    let mut authored = compile_authored_plan(fixture.provider.clone(), source, config);
    put_u32_columns(&mut authored, "seed", &[&[7]]);
    let certificate = certified(&authored);
    let receipt = authored
        .executor
        .prepare_resident_graph(
            &authored.plan,
            &certificate,
            ResidentGraphPrepareOptions::default(),
        )
        .expect("stable type-compatible recursive rule metadata must prepare")
        .launch()
        .expect("recursive catalog-schema graph launch")
        .synchronize_core()
        .expect("recursive catalog-schema synchronization")
        .observe_final_receipt()
        .expect("recursive catalog-schema receipt");
    receipt
        .commit(&mut authored.executor)
        .expect("recursive catalog-schema result commit");
    assert_eq!(
        download_u32_relation(&authored.executor, "path"),
        vec![vec![7, 7], vec![1, 2]]
    );
    let installed_schema = authored
        .executor
        .store()
        .get("path")
        .expect("committed recursive relation")
        .schema();
    assert_eq!(
        installed_schema
            .columns
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        vec!["item", "computed_1"]
    );
    assert_eq!(installed_schema.sort_labels(), &["item", "computed_1"]);

    let mut mismatched =
        compile_authored_plan(fixture.provider.clone(), source, RuntimeConfig::default());
    put_u32_columns(&mut mismatched, "seed", &[&[7]]);
    let mismatched_key_catalog = ResidentGraphSchemaCatalog::from_named_schemas(
        mismatched.rel_ids.iter().filter_map(|(name, rel)| {
            mismatched.schemas.get(name).cloned().map(|mut schema| {
                if name == "path" {
                    schema.key_columns = vec![0];
                }
                (name.clone(), *rel, schema)
            })
        }),
    );
    let mismatched_key_certificate =
        ResidentGraphRouteCertificate::inspect(&mismatched.plan, &mismatched_key_catalog)
            .expect("key-mismatched catalog still has a structurally supported route");
    fixture.provider.memory().reset_alloc_count();
    let graph_before = fixture.runtime.conditional_graph_stats();
    match mismatched.executor.prepare_resident_graph(
        &mismatched.plan,
        &mismatched_key_certificate,
        ResidentGraphPrepareOptions::default(),
    ) {
        Err(ResidentGraphExecutionError::Declined(
            ResidentGraphDeclineReason::WorkspaceUnbounded { detail },
        )) => assert!(
            detail.contains("contribution key columns do not equal its catalog candidate"),
            "unexpected recursive key-schema decline: {detail}"
        ),
        Err(other) => panic!("unexpected recursive key-schema error: {other:?}"),
        Ok(_) => panic!("recursive key-schema mismatch must decline before allocation"),
    }
    assert_eq!(fixture.provider.memory().alloc_count(), 0);
    assert_eq!(fixture.runtime.conditional_graph_stats(), graph_before);

    let varying_source = r#"
            pred left(item: u32, claim_left: u32, rule_left: u32).
            pred right(item: u32, padding: u32, claim_right: u32, rule_right: u32).
            pred varying(kind: u32, item: u32, claim: u32, rule: u32).
            pred stable(kind: u32, item: u32).
            varying(10, X, Claim, Rule) :- left(X, Claim, Rule).
            varying(10, X, Claim, Rule) :- right(X, 0, Claim, Rule).
            stable(10, X) :- varying(10, X, Claim, Rule).
        "#;
    let mut collapsed = compile_authored_plan(
        fixture.provider.clone(),
        varying_source,
        RuntimeConfig::default(),
    );
    put_u32_columns(&mut collapsed, "left", &[&[7], &[20], &[30]]);
    put_u32_columns(&mut collapsed, "right", &[&[8], &[0], &[40], &[50]]);
    let collapsed_certificate = certified(&collapsed);
    let receipt = collapsed
        .executor
        .prepare_resident_graph(
            &collapsed.plan,
            &collapsed_certificate,
            ResidentGraphPrepareOptions::default(),
        )
        .expect("ordinary consumer must accept source metadata variants that it erases")
        .launch()
        .expect("stable-collapse graph launch")
        .synchronize_core()
        .expect("stable-collapse graph synchronization")
        .observe_final_receipt()
        .expect("stable-collapse graph receipt");
    receipt
        .commit(&mut collapsed.executor)
        .expect("stable-collapse result commit");
    assert_eq!(
        download_u32_relation(&collapsed.executor, "stable"),
        vec![vec![10, 10], vec![7, 8]]
    );
    let stable_schema = collapsed
        .executor
        .store()
        .get("stable")
        .expect("committed stable relation")
        .schema();
    assert_eq!(
        stable_schema
            .columns
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        vec!["computed_0", "item"]
    );

    let inherited_source = format!(
        r#"{varying_source}
            pred inherited(claim: u32).
            inherited(Claim) :- varying(10, X, Claim, Rule).
            ?- inherited(Claim).
        "#
    );
    for left_wins in [true, false] {
        let seed_inputs = |authored: &mut AuthoredPlan| {
            let left_item = if left_wins { vec![7] } else { Vec::new() };
            let left_claim = if left_wins { vec![20] } else { Vec::new() };
            let left_rule = if left_wins { vec![30] } else { Vec::new() };
            let right_item = if left_wins { Vec::new() } else { vec![8] };
            let right_padding = if left_wins { Vec::new() } else { vec![0] };
            let right_claim = if left_wins { Vec::new() } else { vec![40] };
            let right_rule = if left_wins { Vec::new() } else { vec![50] };
            put_u32_columns(authored, "left", &[&left_item, &left_claim, &left_rule]);
            put_u32_columns(
                authored,
                "right",
                &[&right_item, &right_padding, &right_claim, &right_rule],
            );
        };

        let mut baseline = compile_authored_plan(
            fixture.provider.clone(),
            &inherited_source,
            RuntimeConfig::default(),
        );
        seed_inputs(&mut baseline);
        baseline
            .executor
            .execute_plan(&baseline.plan)
            .expect("legacy executor must resolve inherited metadata lineage");
        let expected = ["varying", "inherited", "__xlog_query_0"]
            .into_iter()
            .map(|name| {
                let relation = baseline
                    .executor
                    .store()
                    .get(name)
                    .unwrap_or_else(|| panic!("legacy executor did not install {name}"));
                (
                    name,
                    (
                        relation.schema().clone(),
                        download_u32_relation(&baseline.executor, name),
                    ),
                )
            })
            .collect::<BTreeMap<_, _>>();

        let mut inherited = compile_authored_plan(
            fixture.provider.clone(),
            &inherited_source,
            RuntimeConfig::default(),
        );
        seed_inputs(&mut inherited);
        let inherited_certificate = certified(&inherited);
        let receipt = inherited
            .executor
            .prepare_resident_graph(
                &inherited.plan,
                &inherited_certificate,
                ResidentGraphPrepareOptions::default(),
            )
            .expect("acyclic ordinary metadata lineage must prepare")
            .launch()
            .expect("acyclic ordinary metadata lineage graph launch")
            .synchronize_core()
            .expect("acyclic ordinary metadata lineage synchronization")
            .observe_final_receipt()
            .expect("acyclic ordinary metadata lineage receipt");
        receipt
            .commit(&mut inherited.executor)
            .expect("acyclic ordinary metadata lineage commit");
        for (name, (schema, rows)) in &expected {
            let relation = inherited
                .executor
                .store()
                .get(*name)
                .unwrap_or_else(|| panic!("resident executor did not install {name}"));
            assert_eq!(relation.schema(), schema, "schema mismatch for {name}");
            assert_eq!(
                download_u32_relation(&inherited.executor, name),
                *rows,
                "row mismatch for {name}"
            );
        }
    }

    let ambiguous_source = r#"
            pred first_left(item: u32, claim_left: u32, rule_left: u32).
            pred first_right(item: u32, padding: u32, claim_right: u32, rule_right: u32).
            pred second_left(item: u32, claim_left: u32, rule_left: u32).
            pred second_right(item: u32, padding: u32, claim_right: u32, rule_right: u32).
            pred first(kind: u32, item: u32, claim: u32, rule: u32).
            pred second(kind: u32, item: u32, claim: u32, rule: u32).
            pred ambiguous(first_claim: u32, second_claim: u32).
            first(10, X, Claim, Rule) :- first_left(X, Claim, Rule).
            first(10, X, Claim, Rule) :- first_right(X, 0, Claim, Rule).
            second(20, X, Claim, Rule) :- second_left(X, Claim, Rule).
            second(20, X, Claim, Rule) :- second_right(X, 0, Claim, Rule).
            ambiguous(First, Second) :-
                first(10, X, First, FirstRule),
                second(20, X, Second, SecondRule).
        "#;
    let mut ambiguous = compile_authored_plan(
        fixture.provider.clone(),
        ambiguous_source,
        RuntimeConfig::default(),
    );
    put_u32_columns(&mut ambiguous, "first_left", &[&[7], &[20], &[30]]);
    put_u32_columns(&mut ambiguous, "first_right", &[&[8], &[0], &[40], &[50]]);
    put_u32_columns(&mut ambiguous, "second_left", &[&[7], &[60], &[70]]);
    put_u32_columns(&mut ambiguous, "second_right", &[&[8], &[0], &[80], &[90]]);
    let ambiguous_certificate = certified(&ambiguous);
    fixture.provider.memory().reset_alloc_count();
    let graph_before = fixture.runtime.conditional_graph_stats();
    match ambiguous.executor.prepare_resident_graph(
        &ambiguous.plan,
        &ambiguous_certificate,
        ResidentGraphPrepareOptions::default(),
    ) {
        Err(ResidentGraphExecutionError::Declined(
            ResidentGraphDeclineReason::WorkspaceUnbounded { detail },
        )) => assert!(
            detail.contains("multiple schema sources"),
            "unexpected ambiguous-lineage schema decline: {detail}"
        ),
        Err(other) => panic!("unexpected ambiguous-lineage schema error: {other:?}"),
        Ok(_) => panic!("ordinary consumer with independent schema sources must decline"),
    }
    assert_eq!(fixture.provider.memory().alloc_count(), 0);
    assert_eq!(fixture.runtime.conditional_graph_stats(), graph_before);
}

#[test]
fn exact_manifest_reservation_is_atomic_at_the_budget_boundary() {
    let Some(probe_fixture) = runtime_fixture() else {
        return;
    };
    let probe = seeded_recursive_plan_with_gate(probe_fixture.provider.clone(), 16, &[2, 3, 4]);
    let probe_certificate = certified(&probe);
    probe_fixture
        .runtime
        .reap_pending()
        .expect("settle source setup before measuring its retained bytes");
    let retained_source_bytes = probe_fixture.provider.memory().allocated_bytes();
    let probe_prepared = probe
        .executor
        .prepare_resident_graph(
            &probe.plan,
            &probe_certificate,
            ResidentGraphPrepareOptions::default(),
        )
        .expect("probe preparation must expose the exact manifest");
    let manifest_bytes = probe_prepared.preflight_report().estimated_required_bytes;
    assert_eq!(
        probe_prepared
            .preflight_report()
            .tracked_device_allocation_bytes,
        manifest_bytes
    );
    drop(probe_prepared);
    drop(probe);
    probe_fixture
        .runtime
        .reap_pending()
        .expect("probe graph allocations must be reclaimable");
    drop(probe_fixture);

    let short_budget = retained_source_bytes
        .checked_add(manifest_bytes)
        .and_then(|bytes| bytes.checked_sub(1))
        .expect("nonzero manifest budget");
    let Some(short_fixture) = runtime_fixture_with_local_budget(short_budget) else {
        return;
    };
    let short = seeded_recursive_plan_with_gate(short_fixture.provider.clone(), 16, &[2, 3, 4]);
    let short_certificate = certified(&short);
    short_fixture
        .runtime
        .reap_pending()
        .expect("settle short-budget source setup");
    assert_eq!(
        short_fixture.provider.memory().allocated_bytes(),
        retained_source_bytes,
        "the discriminator must reach preparation with exactly manifest-1 bytes available"
    );
    short_fixture.sink.clear();
    short_fixture.provider.memory().reset_alloc_count();
    let allocations_before = short_fixture.provider.memory().alloc_count();
    let manager_bytes_before = short_fixture.provider.memory().allocated_bytes();
    let handles_before = short_fixture
        .runtime
        .resident_graph_handle_lifecycle_stats();
    let graph_stats_before = short_fixture.runtime.conditional_graph_stats();
    let error = match short.executor.prepare_resident_graph(
        &short.plan,
        &short_certificate,
        ResidentGraphPrepareOptions::default(),
    ) {
        Ok(_) => panic!("manifest-1 must refuse before partial materialization"),
        Err(error) => error,
    };
    match error {
        ResidentGraphExecutionError::Declined(ResidentGraphDeclineReason::WorkspaceUnbounded {
            detail,
        }) => assert!(
            detail.contains("allocation manifest reservation"),
            "{detail}"
        ),
        other => panic!("unexpected manifest-1 result: {other:?}"),
    }
    assert_eq!(
        short_fixture.provider.memory().alloc_count(),
        allocations_before
    );
    assert_eq!(
        short_fixture.provider.memory().allocated_bytes(),
        manager_bytes_before
    );
    assert!(short_fixture.sink.snapshot().is_empty());
    assert_eq!(
        short_fixture
            .runtime
            .resident_graph_handle_lifecycle_stats(),
        handles_before
    );
    assert_eq!(
        short_fixture.runtime.conditional_graph_stats(),
        graph_stats_before
    );
    drop(short);
    drop(short_fixture);

    let exact_budget = retained_source_bytes
        .checked_add(manifest_bytes)
        .expect("exact manifest budget");
    let Some(exact_fixture) = runtime_fixture_with_local_budget(exact_budget) else {
        return;
    };
    let exact = seeded_recursive_plan_with_gate(exact_fixture.provider.clone(), 16, &[2, 3, 4]);
    let exact_certificate = certified(&exact);
    exact_fixture
        .runtime
        .reap_pending()
        .expect("settle exact-budget source setup");
    let exact_before = exact_fixture.provider.memory().allocated_bytes();
    let exact_prepared = exact
        .executor
        .prepare_resident_graph(
            &exact.plan,
            &exact_certificate,
            ResidentGraphPrepareOptions::default(),
        )
        .expect("the exact manifest budget must materialize successfully");
    let exact_report = exact_prepared.preflight_report();
    assert_eq!(exact_report.estimated_required_bytes, manifest_bytes);
    assert_eq!(exact_report.tracked_device_allocation_bytes, manifest_bytes);
    assert_eq!(
        exact_fixture
            .provider
            .memory()
            .allocated_bytes()
            .checked_sub(exact_before),
        Some(manifest_bytes)
    );
}

#[test]
fn sequential_filters_share_one_mutable_scan_workspace() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let mut authored = compile_authored_plan(
        fixture.provider.clone(),
        sequential_filter_program(),
        RuntimeConfig::default(),
    );
    // Three source rows produce a four-row resident capacity class.
    put_u32_columns(&mut authored, "input", &[&[0, 1, 9]]);
    let certificate = certified(&authored);
    let graph_before = fixture.runtime.conditional_graph_stats();
    let prepared = authored
        .executor
        .prepare_resident_graph(
            &authored.plan,
            &certificate,
            ResidentGraphPrepareOptions::default(),
        )
        .expect("two-filter resident graph must prepare");
    let report = prepared.preflight_report();
    assert_eq!(report.filter_descriptor_device_bytes, 2 * 48);
    assert_eq!(report.filter_scratch_allocations, 1);
    assert_eq!(
        report.estimated_required_bytes,
        report.tracked_device_allocation_bytes
    );
    assert!(report.logical_relation_values > report.private_relation_slots);
    assert_eq!(
        fixture.runtime.conditional_graph_stats().launches,
        graph_before.launches
    );
}

#[test]
fn manifest_accounts_for_a_wider_join_intermediate_than_the_committed_head() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let mut authored = compile_authored_plan(
        fixture.provider.clone(),
        wide_intermediate_program(),
        RuntimeConfig::default(),
    );
    let left = [1u32];
    let right = [1u32];
    let left_columns = vec![&left[..]; 8];
    let right_columns = vec![&right[..]; 9];
    put_u32_columns(&mut authored, "left", &left_columns);
    put_u32_columns(&mut authored, "right", &right_columns);
    let certificate = certified(&authored);
    let prepared = authored
        .executor
        .prepare_resident_graph(
            &authored.plan,
            &certificate,
            ResidentGraphPrepareOptions::default(),
        )
        .expect("arity-seventeen intermediate must prepare");
    let report = prepared.preflight_report();
    assert_eq!(report.max_row_bytes, 17 * 4);
    assert_eq!(
        report.estimated_required_bytes,
        report.tracked_device_allocation_bytes
    );
    assert!(report.relation_device_bytes >= report.relation_capacity as u64 * 17 * 4 + 4);
}

#[test]
fn sibling_filter_branches_remain_live_through_their_join() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let mut authored = compile_authored_plan(
        fixture.provider.clone(),
        sibling_filter_join_program(),
        RuntimeConfig::default(),
    );
    put_u32_columns(&mut authored, "left", &[&[1], &[1]]);
    put_u32_columns(&mut authored, "right", &[&[1], &[1]]);
    let certificate = certified(&authored);
    let prepared = authored
        .executor
        .prepare_resident_graph(
            &authored.plan,
            &certificate,
            ResidentGraphPrepareOptions::default(),
        )
        .expect("filtered sibling branches must prepare");
    let report = prepared.preflight_report();
    assert_eq!(report.filter_descriptor_device_bytes, 2 * 48);
    assert_eq!(report.filter_scratch_allocations, 1);
    // Both two-column filter results overlap at the join. The four-column
    // join and the overlapping one-column set stages force a five-slot peak,
    // while later compatible logical values reuse those generations.
    assert_eq!(report.scratch_slots, 5);
    assert!(
        report.logical_relation_values - report.permanent_relation_slots as usize
            > report.scratch_slots as usize
    );
    assert_eq!(
        report.estimated_required_bytes,
        report.tracked_device_allocation_bytes
    );
}

#[test]
fn direct_projection_accepts_a_source_smaller_than_its_capacity_class() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let mut authored = compile_authored_plan(
        fixture.provider.clone(),
        direct_projection_program(),
        RuntimeConfig::default(),
    );
    put_u32_columns(&mut authored, "input", &[&[1, 2, 3], &[4, 5, 6]]);
    let certificate = certified(&authored);
    let prepared = authored
        .executor
        .prepare_resident_graph(
            &authored.plan,
            &certificate,
            ResidentGraphPrepareOptions::default(),
        )
        .expect("three-row source projection must prepare in a four-row capacity class");
    let report = prepared.preflight_report();
    assert_eq!(report.relation_capacity, 4);
    assert_eq!(
        report.estimated_required_bytes,
        report.tracked_device_allocation_bytes
    );
}

#[test]
fn real_recursive_plan_preserves_exact_iteration_limit_and_context_reuse() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    for (limit, completed) in [(0, 0), (1, 1)] {
        let mut authored = seeded_recursive_plan(fixture.provider.clone(), limit);
        put_u32_columns(&mut authored, "stable", &[&[11, 12]]);
        let before = relation_snapshot(&authored.executor, &["edge", "seed", "gate", "stable"]);
        let certificate = certified(&authored);
        let prepared = authored
            .executor
            .prepare_resident_graph(
                &authored.plan,
                &certificate,
                ResidentGraphPrepareOptions::default(),
            )
            .expect("setup");
        reset_independent_recorders(&fixture);
        fixture.provider.memory().reset_alloc_count();
        let allocations_before = fixture.provider.memory().alloc_count();
        let synchronized = prepared
            .launch()
            .expect("launch")
            .synchronize_core()
            .expect("terminal sync");
        assert_core_recorders_are_zero(&fixture, allocations_before);
        let observed = synchronized.observe_final_receipt().expect("final receipt");
        assert_eq!(
            fixture
                .provider
                .final_observation_transfer_stats()
                .dtoh_calls,
            1
        );
        let error = observed
            .commit(&mut authored.executor)
            .expect_err("iteration limit must reject staged results");
        assert_eq!(
            error,
            ResidentGraphExecutionError::IterationLimit { limit, completed }
        );
        assert_eq!(
            relation_snapshot(&authored.executor, &["edge", "seed", "gate", "stable"]),
            before
        );
        assert!(!authored.executor.store().contains("reach"));
        assert!(!authored.executor.store().contains("selected"));
        assert!(!authored.executor.store().contains("__xlog_query_0"));
        fixture
            .provider
            .device()
            .synchronize()
            .expect("CUDA context must remain usable after limit status");
    }

    let mut succeeding = seeded_recursive_plan(fixture.provider.clone(), 16);
    let certificate = certified(&succeeding);
    succeeding
        .executor
        .prepare_resident_graph(
            &succeeding.plan,
            &certificate,
            ResidentGraphPrepareOptions::default(),
        )
        .expect("setup after limit")
        .launch()
        .expect("launch after limit")
        .synchronize_core()
        .expect("sync after limit")
        .observe_final_receipt()
        .expect("receipt after limit")
        .commit(&mut succeeding.executor)
        .expect("same CUDA context must execute a later successful graph");
}

#[test]
fn device_status_writer_drives_exact_overflow_and_resource_errors_without_store_mutation() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let statuses = [
        (
            ResidentGraphDeviceStatus::CapacityOverflow {
                op_id: 7,
                required: 5,
                capacity: 4,
            },
            ResidentGraphExecutionError::CapacityOverflow {
                op_id: 7,
                required: 5,
                capacity: 4,
            },
        ),
        (
            ResidentGraphDeviceStatus::ResourceExhausted {
                op_id: 9,
                resource: "workspace_slots",
                required: 3,
                capacity: 2,
            },
            ResidentGraphExecutionError::ResourceExhausted {
                op_id: 9,
                resource: "workspace_slots",
                required: 3,
                capacity: 2,
            },
        ),
    ];

    for (device_status, expected) in statuses {
        let mut authored = seeded_recursive_plan(fixture.provider.clone(), 16);
        put_u32_columns(&mut authored, "stable", &[&[21, 34]]);
        let before = relation_snapshot(&authored.executor, &["edge", "seed", "gate", "stable"]);
        let certified_plan = sealed(&authored);
        let mut default_authored = seeded_recursive_plan(fixture.provider.clone(), 16);
        let default_prepared = default_authored
            .executor
            .prepare_certified_resident_graph(
                &certified_plan,
                ResidentGraphPrepareOptions::default(),
            )
            .expect("default setup");
        reset_independent_recorders(&fixture);
        default_prepared
            .launch()
            .expect("default launch")
            .synchronize_core()
            .expect("default terminal sync")
            .observe_final_receipt()
            .expect("default final receipt")
            .commit(&mut default_authored.executor)
            .expect("default commit");
        assert_eq!(
            fixture
                .runtime
                .conditional_graph_stats()
                .device_status_writer_launches,
            0
        );
        drop(default_authored);
        fixture
            .provider
            .device()
            .synchronize()
            .expect("cleanup sync");
        fixture.runtime.reap_pending().expect("cleanup reap");
        let injection =
            ResidentGraphDeviceStatusTestInjection::device_kernel_after_op(1, device_status);
        let prepared = authored
            .executor
            .prepare_certified_resident_graph(
                &certified_plan,
                ResidentGraphPrepareOptions::default().with_test_device_status(injection),
            )
            .expect("setup");
        reset_independent_recorders(&fixture);
        fixture.provider.memory().reset_alloc_count();
        let allocations_before = fixture.provider.memory().alloc_count();
        let synchronized = prepared
            .launch()
            .expect("launch")
            .synchronize_core()
            .expect("terminal sync");
        assert_core_recorders_are_zero(&fixture, allocations_before);
        let graph = fixture.runtime.conditional_graph_stats();
        assert_eq!(graph.device_status_writer_launches, 1);
        assert_eq!(graph.host_status_injections, 0);
        let observed = synchronized
            .observe_final_receipt()
            .expect("device status receipt");
        assert_eq!(
            fixture
                .provider
                .final_observation_transfer_stats()
                .dtoh_calls,
            1
        );
        let error = observed
            .commit(&mut authored.executor)
            .expect_err("device status must reject staged results");
        assert_eq!(error, expected);
        assert_eq!(
            relation_snapshot(&authored.executor, &["edge", "seed", "gate", "stable"]),
            before
        );
        assert!(!authored.executor.store().contains("reach"));
        assert!(!authored.executor.store().contains("selected"));
        fixture
            .provider
            .device()
            .synchronize()
            .expect("status decoding must not poison the CUDA context");
    }
}

#[test]
fn dropping_a_real_inflight_graph_releases_handles_workspace_and_events_without_commit() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let authored = seeded_recursive_plan(fixture.provider.clone(), 16);
    let before = relation_snapshot(&authored.executor, &["edge", "seed", "gate"]);
    let certificate = certified(&authored);
    fixture
        .runtime
        .reap_pending()
        .expect("settle source canonicalization before lifecycle baseline");
    fixture.sink.clear();
    let runtime_bytes_before_prepare = fixture.runtime.bytes_outstanding();
    let events_before_prepare = fixture.runtime.event_lifecycle_stats();
    let handles_before_prepare = fixture.runtime.resident_graph_handle_lifecycle_stats();
    let graph_before_prepare = fixture.runtime.conditional_graph_stats();
    let prepared = authored
        .executor
        .prepare_resident_graph(
            &authored.plan,
            &certificate,
            ResidentGraphPrepareOptions::default(),
        )
        .expect("setup");
    let handles_after_prepare = fixture.runtime.resident_graph_handle_lifecycle_stats();
    assert_eq!(
        handles_after_prepare.live_graphs,
        handles_before_prepare.live_graphs + 1,
        "prepare must retain one real CUDA graph handle"
    );
    assert_eq!(
        handles_after_prepare.live_graph_execs,
        handles_before_prepare.live_graph_execs + 1,
        "prepare must retain one instantiated CUDA graph executable"
    );
    assert_eq!(
        handles_after_prepare.created_graphs,
        handles_before_prepare.created_graphs + 1
    );
    assert_eq!(
        handles_after_prepare.created_graph_execs,
        handles_before_prepare.created_graph_execs + 1
    );

    let mut live_workspace_allocations = BTreeMap::new();
    for record in fixture.sink.snapshot() {
        match (record.action, record.ptr, record.bytes) {
            (LogAction::Allocate, Some(ptr), Some(bytes)) => {
                live_workspace_allocations.insert(ptr, (bytes, record.order_counter));
            }
            (LogAction::Deallocate, Some(ptr), _) => {
                live_workspace_allocations.remove(&ptr);
            }
            _ => {}
        }
    }
    assert!(
        !live_workspace_allocations.is_empty(),
        "prepare must retain private runtime-backed workspace allocations"
    );
    let private_workspace_bytes: usize = live_workspace_allocations
        .values()
        .map(|(bytes, _)| *bytes)
        .sum();
    assert_eq!(
        fixture.runtime.bytes_outstanding(),
        runtime_bytes_before_prepare + private_workspace_bytes,
        "the retained private workspace must account for every added runtime byte"
    );

    let in_flight = prepared.launch().expect("real conditional launch");
    let events_during_launch = fixture.runtime.event_lifecycle_stats();
    assert!(
        events_during_launch.live_events > events_before_prepare.live_events,
        "launch must own a real completion event"
    );
    let handles_during_launch = fixture.runtime.resident_graph_handle_lifecycle_stats();
    assert_eq!(
        handles_during_launch.live_graphs,
        handles_after_prepare.live_graphs
    );
    assert_eq!(
        handles_during_launch.live_graph_execs,
        handles_after_prepare.live_graph_execs
    );
    let graph_during_launch = fixture.runtime.conditional_graph_stats();
    assert_eq!(
        graph_during_launch.launches,
        graph_before_prepare.launches + 1,
        "launch must execute the prepared conditional CUDA graph"
    );
    assert_eq!(
        fixture.runtime.bytes_outstanding(),
        runtime_bytes_before_prepare + private_workspace_bytes,
        "launch must retain the prepared graph's private workspace"
    );

    drop(in_flight);
    let handles_after_drop = fixture.runtime.resident_graph_handle_lifecycle_stats();
    assert_eq!(
        handles_after_drop.live_graphs,
        handles_before_prepare.live_graphs
    );
    assert_eq!(
        handles_after_drop.live_graph_execs,
        handles_before_prepare.live_graph_execs
    );
    assert_eq!(
        fixture.runtime.bytes_outstanding(),
        runtime_bytes_before_prepare + private_workspace_bytes,
        "Drop must keep queued workspace frees accounted until reap"
    );
    fixture
        .runtime
        .reap_pending()
        .expect("reap dropped graph workspace");
    let events_after_reap = fixture.runtime.event_lifecycle_stats();
    assert_eq!(
        events_after_reap.live_events,
        events_before_prepare.live_events
    );
    assert!(events_after_reap.created_events > events_before_prepare.created_events);
    assert_eq!(
        events_after_reap.created_events - events_before_prepare.created_events,
        events_after_reap.destroyed_events - events_before_prepare.destroyed_events
    );
    assert!(events_after_reap.drop_waits > events_before_prepare.drop_waits);
    let handles_after_reap = fixture.runtime.resident_graph_handle_lifecycle_stats();
    assert_eq!(
        handles_after_reap.live_graphs,
        handles_before_prepare.live_graphs
    );
    assert_eq!(
        handles_after_reap.live_graph_execs,
        handles_before_prepare.live_graph_execs
    );
    assert_eq!(
        handles_after_reap.created_graphs - handles_before_prepare.created_graphs,
        handles_after_reap.destroyed_graphs - handles_before_prepare.destroyed_graphs
    );
    assert_eq!(
        handles_after_reap.created_graph_execs - handles_before_prepare.created_graph_execs,
        handles_after_reap.destroyed_graph_execs - handles_before_prepare.destroyed_graph_execs
    );
    assert_eq!(
        fixture.runtime.bytes_outstanding(),
        runtime_bytes_before_prepare,
        "reap must release every private graph-workspace byte"
    );
    let records_after_reap = fixture.sink.snapshot();
    for (ptr, (_, allocation_order)) in &live_workspace_allocations {
        assert!(
            records_after_reap.iter().any(|record| {
                record.action == LogAction::Deallocate
                    && record.ptr == Some(*ptr)
                    && record.order_counter > *allocation_order
            }),
            "reap must log deallocation of private workspace pointer {ptr:#x}"
        );
    }
    assert_eq!(
        relation_snapshot(&authored.executor, &["edge", "seed", "gate"]),
        before
    );
    assert!(!authored.executor.store().contains("reach"));
    assert!(!authored.executor.store().contains("selected"));
    fixture
        .provider
        .device()
        .synchronize()
        .expect("context must remain usable after in-flight Drop");
}
