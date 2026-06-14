#![allow(clippy::arc_with_non_send_sync)]

//! Replay an external consumer widened-frontier promotion shape through xlog and
//! assert deterministic clean output.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use xlog_core::{MemoryBudget, RuntimeConfig, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::{CudaBuffer, CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_logic::Compiler;
use xlog_runtime::Executor;

const EXTERNAL_CONSUMER_SERVING_V2_ENV: &str = "XLOG_EXTERNAL_CONSUMER_SERVING_V2";
const DEFAULT_EXTERNAL_CONSUMER_SERVING_V2: &str =
    "/home/dev/projects/external-consumer/data/serving/v2";
const ITERATIONS: usize = 20;

const REPLAY_SOURCE: &str = r#"
    pred frontier_pred(u32).
    pred widened_pred(u32).
    pred frontier_edge(u32, u32).
    pred blocked_pred(u32).
    pred promoted(u32).
    pred replay_reachable(u32).
    pred rollback_hit(u32).

    promoted(P) :- frontier_pred(P), widened_pred(P).
    replay_reachable(P) :- promoted(P).
    replay_reachable(Q) :- replay_reachable(P), frontier_edge(P, Q), frontier_pred(Q).
    rollback_hit(P) :- replay_reachable(P), blocked_pred(P).
"#;

#[derive(Clone, Debug)]
struct ReplayArtifact {
    source: &'static str,
    allowed_predicates: Vec<u32>,
}

#[derive(Clone, Debug)]
struct ReplayInputs {
    frontier_pred: Vec<u32>,
    widened_pred: Vec<u32>,
    blocked_pred: Vec<u32>,
    frontier_edge: Vec<(u32, u32)>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ReplaySnapshot {
    relations: BTreeMap<String, Vec<Vec<u32>>>,
}

struct DiscardSink;

impl LoggingSink for DiscardSink {
    fn emit(&self, _record: LogRecord) -> Result<(), SinkError> {
        Ok(())
    }
}

#[allow(dead_code)]
struct RuntimeFixture {
    device: Arc<CudaDevice>,
    runtime: Arc<XlogDeviceRuntime>,
    memory: Arc<GpuMemoryManager>,
    provider: Arc<CudaKernelProvider>,
    pool: Arc<StreamPool>,
}

fn make_runtime_fixture() -> Option<RuntimeFixture> {
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
    let async_resource: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(
        AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool)),
    );
    let logging: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(LoggingResource::new(
        async_resource,
        Arc::new(DiscardSink) as Arc<dyn LoggingSink>,
    ));
    let budget: Box<dyn DeviceMemoryResource + Send + Sync> =
        Box::new(GlobalDeviceBudget::new(logging, 128 * 1024 * 1024));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(128 * 1024 * 1024),
        Arc::clone(&runtime),
    ));
    let provider =
        Arc::new(CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory)).ok()?);
    Some(RuntimeFixture {
        device,
        runtime,
        memory,
        provider,
        pool,
    })
}

fn extract_json_array_numbers(doc: &str, field: &str) -> Option<Vec<u32>> {
    let field_pos = doc.find(&format!("\"{field}\""))?;
    let after_field = &doc[field_pos..];
    let open = after_field.find('[')?;
    let close = after_field[open..].find(']')? + open;
    let body = &after_field[open + 1..close];
    let mut out = Vec::new();
    for item in body.split(',') {
        let trimmed = item.trim();
        if trimmed.is_empty() {
            continue;
        }
        out.push(trimmed.parse::<u32>().ok()?);
    }
    Some(out)
}

fn validate_serving_v2(schema: &str, promotion: &str, allowed: &[u32]) {
    assert!(
        promotion.contains("\"verdict\": \"PASS\""),
        "serving/v2 promotion verdict must be PASS"
    );
    assert!(
        promotion.contains("\"version\": 2"),
        "serving/v2 promotion version must be 2"
    );
    assert!(
        schema.contains("\"predicate_names\""),
        "schema must expose predicate_names"
    );
    assert!(
        schema.contains("\"type_constraints\""),
        "schema must expose type_constraints"
    );
    for pred in allowed {
        assert!(
            schema.contains(&format!("\"{pred}\": 2")),
            "predicate {pred} must be binary in serving/v2 arity_map"
        );
    }
}

fn load_replay_artifact() -> ReplayArtifact {
    let artifact_root = std::env::var(EXTERNAL_CONSUMER_SERVING_V2_ENV)
        .unwrap_or_else(|_| DEFAULT_EXTERNAL_CONSUMER_SERVING_V2.to_string());
    let schema_path = Path::new(&artifact_root).join("schema.json");
    let promotion_path = Path::new(&artifact_root).join("promotion.json");
    if let (Ok(schema), Ok(promotion)) = (
        std::fs::read_to_string(&schema_path),
        std::fs::read_to_string(&promotion_path),
    ) {
        let allowed = extract_json_array_numbers(&schema, "allowed_predicates")
            .filter(|v| v.len() >= 5)
            .expect("serving/v2 schema must expose at least five allowed predicates");
        validate_serving_v2(&schema, &promotion, &allowed);
        return ReplayArtifact {
            source: "external-consumer:data/serving/v2",
            allowed_predicates: allowed,
        };
    }

    ReplayArtifact {
        source: "embedded:minimal-widened-frontier",
        allowed_predicates: vec![1, 2, 3, 4, 5],
    }
}

fn replay_inputs(allowed: &[u32]) -> ReplayInputs {
    assert!(
        allowed.len() >= 5,
        "widened-frontier replay representative needs at least five predicate IDs"
    );
    let frontier_pred = allowed[..5].to_vec();
    let widened_pred = vec![allowed[1], allowed[3]];
    let blocked_pred = vec![allowed[4]];
    let frontier_edge = vec![
        (allowed[1], allowed[2]),
        (allowed[2], allowed[4]),
        (allowed[3], allowed[4]),
    ];
    ReplayInputs {
        frontier_pred,
        widened_pred,
        blocked_pred,
        frontier_edge,
    }
}

fn unary_schema() -> Schema {
    Schema::new(vec![("c0".to_string(), ScalarType::U32)])
}

fn binary_schema() -> Schema {
    Schema::new(vec![
        ("c0".to_string(), ScalarType::U32),
        ("c1".to_string(), ScalarType::U32),
    ])
}

fn upload_unary(provider: &CudaKernelProvider, values: &[u32]) -> CudaBuffer {
    provider
        .create_buffer_from_u32_columns(&[values], unary_schema())
        .expect("upload unary")
}

fn upload_binary(provider: &CudaKernelProvider, values: &[(u32, u32)]) -> CudaBuffer {
    let col0: Vec<u32> = values.iter().map(|(a, _)| *a).collect();
    let col1: Vec<u32> = values.iter().map(|(_, b)| *b).collect();
    provider
        .create_buffer_from_u32_columns(&[&col0, &col1], binary_schema())
        .expect("upload binary")
}

fn download_canonical_rows(provider: &CudaKernelProvider, buffer: &CudaBuffer) -> Vec<Vec<u32>> {
    let columns: Vec<Vec<u32>> = (0..buffer.arity())
        .map(|col| {
            provider
                .download_column::<u32>(buffer, col)
                .expect("download u32 column")
        })
        .collect();
    if columns.is_empty() {
        return Vec::new();
    }
    let row_count = columns[0].len();
    let mut rows: Vec<Vec<u32>> = (0..row_count)
        .map(|row| columns.iter().map(|col| col[row]).collect())
        .collect();
    rows.sort();
    rows
}

fn run_replay(fixture: &RuntimeFixture, inputs: &ReplayInputs) -> ReplaySnapshot {
    let mut compiler = Compiler::new();
    let plan = compiler.compile(REPLAY_SOURCE).expect("compile replay");
    let mut executor = Executor::new_with_config(
        Arc::clone(&fixture.provider),
        RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(false)),
    );
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }

    executor.put_relation(
        "frontier_pred",
        upload_unary(&fixture.provider, &inputs.frontier_pred),
    );
    executor.put_relation(
        "widened_pred",
        upload_unary(&fixture.provider, &inputs.widened_pred),
    );
    executor.put_relation(
        "blocked_pred",
        upload_unary(&fixture.provider, &inputs.blocked_pred),
    );
    executor.put_relation(
        "frontier_edge",
        upload_binary(&fixture.provider, &inputs.frontier_edge),
    );

    executor.execute_plan(&plan).expect("execute replay");

    let mut relations = BTreeMap::new();
    for name in ["promoted", "replay_reachable", "rollback_hit"] {
        let buffer = executor
            .store()
            .get(name)
            .unwrap_or_else(|| panic!("missing replay relation {name}"));
        relations.insert(
            name.to_string(),
            download_canonical_rows(&fixture.provider, buffer),
        );
    }
    ReplaySnapshot { relations }
}

#[test]
fn external_consumer_widened_frontier_replay_is_clean_and_deterministic() {
    let Some(fixture) = make_runtime_fixture() else {
        eprintln!("Skipping widened-frontier replay: CUDA runtime unavailable");
        return;
    };
    let artifact = load_replay_artifact();
    let inputs = replay_inputs(&artifact.allowed_predicates);
    let first = run_replay(&fixture, &inputs);

    assert_eq!(first.relations["promoted"].len(), 2);
    assert_eq!(first.relations["replay_reachable"].len(), 4);
    assert_eq!(first.relations["rollback_hit"].len(), 1);

    for iteration in 0..ITERATIONS {
        let current = run_replay(&fixture, &inputs);
        assert_eq!(
            current, first,
            "widened-frontier replay iteration {iteration} diverged"
        );
    }

    println!(
        "WIDENED_FRONTIER_REPLAY PASS artifact_source={} allowed_predicates={} iterations={} \
         promoted_rows={} reachable_rows={} rollback_rows={}",
        artifact.source,
        artifact.allowed_predicates.len(),
        ITERATIONS,
        first.relations["promoted"].len(),
        first.relations["replay_reachable"].len(),
        first.relations["rollback_hit"].len()
    );
}
