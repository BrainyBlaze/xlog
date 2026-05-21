# v0.8.7 Living-World Diagnostics Architecture

v0.8.7 adds the audit surfaces needed to explain living-world logic workloads:
generated rules, source rules, query proofs, biomedical graph ingestion,
relation deltas, validation staging, relation evidence, and neural hot-loop
runtime state can now be inspected through stable Rust, CLI, and pyxlog APIs.

The design goal is observability without changing the production data path. The
new diagnostics describe rule and relation metadata, hashes, counters, and
runtime states. They do not materialize relation rows on the host unless the
caller already selected a host-readable API.

## Scope Map

| Workload gap | Surface | Primary implementation | Public API |
|---|---|---|---|
| XLOG-LWM-001 induced-rule audit | Generated rule provenance | `crates/xlog-induce/src/provenance.rs` | `InducedRuleProvenance`, `InductionProvenanceRegistry` |
| XLOG-LWM-002 rule origin audit | Source/generated rule provenance | `crates/xlog-logic/src/diagnostics.rs` | `xlog explain`, `CompiledLogicProgram.rule_provenance()`, `LogicRelationSession.rule_provenance()`, `CompiledProgram.rule_provenance()` |
| XLOG-LWM-003 delta replay audit | Relation delta debug summaries | `crates/xlog-gpu/src/logic.rs`, `crates/pyxlog/src/logic.rs` | `LogicRelationSession.apply_relation_delta_debug(...)`, `delta_stats()` |
| XLOG-LWM-004 temporal/source audit | Temporal relation metadata | `crates/pyxlog/python/pyxlog/__init__.py` | `LogicRelationSession.put_temporal_relation(...)`, `temporal_provenance(...)` |
| XLOG-LWM-005 query proof audit | Direct query proof traces | `crates/xlog-logic/src/diagnostics.rs` | `xlog explain`, `proof_traces()` on deterministic and probabilistic pyxlog programs |
| XLOG-LWM-006 neural hot-loop audit | Unified nn/4 diagnostics | `crates/pyxlog/src/program.rs` | `CompiledProgram.neural_hot_loop_diagnostics()` |
| XLOG-BIOKG-001 graph ingestion audit | Native streamed biomedical graph rows | `crates/xlog-gpu/src/biokg.rs` | `StreamingGraphRelationLoader`, `GraphInputFormat` |
| XLOG-GENRULE-002 generated-rule row audit | Accepted/rejected row decisions | `crates/xlog-cli/src/main.rs` | `xlog explain --format json` `generated_rule_diagnostics` |
| XLOG-PYXLOG-003 relation evidence audit | Session and relation provenance APIs | `crates/pyxlog/python/pyxlog/__init__.py` | `put_relation_with_provenance(...)`, `evidence(...)`, `RelationEvidence.provenance()` |
| XLOG-NN4-004 nn/4 lineage audit | Checkpoint/split/calibration/influence metadata | `crates/pyxlog/python/pyxlog/__init__.py` | `register_network(...checkpoint_hash=...)`, `nn4_lineage()`, `record_nn4_influence(...)` |
| XLOG-DELTA-005 delta planner audit | Cache/fallback/speedup telemetry | `crates/xlog-gpu/src/logic.rs`, `crates/pyxlog/src/logic.rs` | `DeltaPlannerTelemetry`, `delta_stats()["planner_telemetry"]` |
| XLOG-REPRO-006 validation staging audit | Promote-only-on-PASS evidence staging | `scripts/validation_staging.py` | `ValidationStagingRun` |

## Rule And Proof Diagnostics

`xlog_logic::diagnostics` is the shared source of truth for rule provenance and
query proof traces. It exports:

```rust
pub enum RuleSourceKind {
    Source,
    Generated,
    Imported,
    RuntimeInjected,
}

pub struct RuleProvenance {
    pub rule_id: String,
    pub head: String,
    pub source_kind: RuleSourceKind,
    pub source_span: Option<String>,
    pub generation_trace_hash: Option<String>,
    pub support_relation_ids: Vec<String>,
    pub counterexample_relation_ids: Vec<String>,
}

pub struct QueryProofTrace {
    pub query_id: String,
    pub query: String,
    pub answer_relation: String,
    pub rule_ids: Vec<String>,
    pub source_facts: Vec<String>,
    pub rejected_alternatives: Vec<String>,
}
```

The CLI uses the same records in `xlog explain`. Text output prints compact
`rule_provenance:` and `proof_traces:` sections. JSON output emits full
`rule_provenance` and `proof_traces` arrays with stable snake-case keys.
For generated-rule candidates, JSON output also emits
`generated_rule_diagnostics`. Each entry includes `row_decisions` with:

- `row_key`
- `accepted`
- `failed_predicates`
- `threshold_comparisons` with left/right values and pass/fail status
- `aggregate_inputs`

This is the XLOG-GENRULE-002 surface for explaining both accepted generated-rule
rows and rejected rows whose predicates or threshold checks failed.

pyxlog exposes the same records as Python dictionaries on:

- `CompiledLogicProgram.rule_provenance()` and `.proof_traces()`
- `LogicRelationSession.rule_provenance()` and `.proof_traces()`
- `CompiledProgram.rule_provenance()` and `.proof_traces()`

Generated rewrite rules are included when the caller supplies a generated
program, such as the magic-set explain path. Duplicate generated rules whose
text matches source rules are suppressed so the report names only new generated
rules.

## Induction Provenance

`xlog-induce` now has first-class provenance records for generated/mined rules:

```rust
pub struct InducedRuleProvenance {
    pub rule_id: String,
    pub rule_source: String,
    pub source_kind: RuleSourceKind,
    pub search_space_size: u64,
    pub predicate_inventory: Vec<String>,
    pub support_rows: Vec<InductionSupportRow>,
    pub rejected_alternatives: Vec<InductionAlternative>,
    pub falsification_count: u64,
    pub generation_trace_hash: String,
}
```

`InductionSupportRow` keeps the relation name, row offset, and caller-supplied
row hash for retained positive support. `InductionAlternative` keeps the
candidate rule source, support count, and falsification count for rejected
alternatives. `InductionProvenanceRegistry` is an in-memory registration surface
for callers that promote generated rules and need to retain their audit records.

This does not replace the bounded exact-induction scorer. It adds the missing
audit layer for selected generated rules and their alternatives.

## Biomedical Graph Streaming

`xlog_gpu::biokg` provides the XLOG-BIOKG-001 native ingestion surface for
large typed biomedical graph streams:

```rust
use xlog_gpu::biokg::{GraphInputFormat, StreamingGraphRelationLoader};

let report = StreamingGraphRelationLoader::new(GraphInputFormat::Jsonl)
    .with_chunk_rows(100_000)
    .load_path_with_sink("primekg_edges.jsonl", |edge| {
        // Stream `edge.subject`, `edge.predicate`, `edge.object`, `edge.split`,
        // and `edge.row_hash` into the caller's relation builder.
    })?;
```

The loader supports JSONL, CSV, and N-Triples inputs. It reports total row
counts, edge row counts, per-predicate `relation_histogram`, split provenance
through `split_histogram`, stable `row_hashes`, and `bounded_memory` chunk
telemetry. `load_path_with_sink(...)` emits typed edge rows without retaining the
full graph in memory; `load_path(...)` returns telemetry when the caller only
needs provenance and histograms.

## Relation Delta Debugging

`LogicDeltaReport` now carries the names of changed relations and a metadata-only
trace:

```rust
pub struct LogicDeltaReport {
    pub input_delta_count: usize,
    pub changed_relations: usize,
    pub changed_relation_names: Vec<String>,
    pub insert_rows: u64,
    pub delete_rows: u64,
    pub has_deletes: bool,
    pub affected_sccs: usize,
    pub recomputed_sccs: usize,
    pub incremental_sccs: usize,
    pub coalesced_insert_rows: u64,
    pub coalesced_delete_rows: u64,
    pub canceled_rows: u64,
    pub planner_telemetry: DeltaPlannerTelemetry,
    pub debug_trace: Vec<String>,
}

pub struct DeltaPlannerTelemetry {
    pub cache_reused: bool,
    pub fallback_decision: String,
    pub affected_sccs: usize,
    pub recomputed_sccs: usize,
    pub incremental_sccs: usize,
    pub estimated_delta_speedup: Option<f64>,
    pub measured_delta_speedup: Option<f64>,
    pub planner_advice: Vec<String>,
}
```

pyxlog returns these fields from normal delta APIs and from
`LogicRelationSession.apply_relation_delta_debug(updates, check_equivalence=False)`.
When `check_equivalence=True`, pyxlog re-runs a full recompute and compares query
stores through `LogicProgram::relation_stores_query_equivalent(...)`; the returned
dictionary then sets `equivalent_to_full_recompute` to `True` or `False`. The
default is `False` because the equivalence check intentionally costs an extra
full evaluation.

Relation callbacks receive the same changed-relation names and debug trace inside
their metadata payload. The callback path remains metadata-only.
The nested `planner_telemetry` dictionary is the XLOG-DELTA-005 surface: it
reports affected SCCs, cache reuse, fallback decision, estimated/measured delta
speedup when available, and planner advice for incremental-vs-full recompute
tradeoffs.

## Temporal Relation Metadata

`LogicRelationSession.put_temporal_relation(...)` wraps `put_relation(...)` and
stores temporal/source metadata beside the session id:

```python
session.put_temporal_relation(
    "events",
    [event_id, timestamp],
    timestamp_column="ts",
    dataset_id="bfo-lwm",
    row_hashes=["..."],
    field_hashes={"ts": ["..."]},
    uncertainty={"kind": "interval"},
    stream_id="simulation-1",
    order_column="seq",
    source="sensor-replay",
)
```

`session.temporal_provenance()` returns all stored records. Passing a relation
name returns only that relation's metadata. The store is session-local and
in-memory; it is an audit companion for the relation store, not a persistence
format.

The general XLOG-PYXLOG-003 evidence API is separate from temporal metadata:

```python
session.put_relation_with_provenance(
    "edge",
    columns,
    relation_schema=["subject", "predicate", "object"],
    source_hash="sha256:...",
    row_hashes=["..."],
    accepted_count=10,
    rejected_count=2,
    output_hash="sha256:...",
    decision_counts={"accepted": 10, "rejected": 2},
)
session.evidence()
session.relation("edge").provenance()
```

`Session.evidence()` includes a `program_hash` and per-relation evidence.
`RelationEvidence.provenance()` exposes relation schema, source hashes, row
hashes, accepted/rejected counts, output hashes, and decision counts.

## Neural Hot-Loop Diagnostics

`CompiledProgram.neural_hot_loop_diagnostics()` returns one dictionary for the
nn/4 hot loop:

- `post_load_dtoh_bytes`, `post_load_htod_bytes`, `post_load_dtoh_calls`,
  `post_load_htod_calls`
- `control_plane_bytes_per_iteration` and `control_plane_status`
- `scalar_sync_checks` and `scalar_sync_status`
- nested `cuda_graph` counters
- nested `circuit_cache` counters

When the runtime cannot provide a separate control-plane or scalar-sync counter,
the value is `None` and the matching status field explains that the counter is
unavailable. Unsupported probes must not be reported as fake zeroes.

The XLOG-NN4-004 lineage companion is exposed by the top-level pyxlog wrapper:

```python
program.register_network(
    "net",
    module,
    optimizer,
    checkpoint_hash="sha256:...",
    split_hashes={"train": "sha256:...", "test": "sha256:..."},
    calibration_metrics={"ece": 0.02},
    cuda_device=0,
    influence_audit={"heldout": "test"},
)
program.record_nn4_influence(
    "net",
    query="accepted(a)",
    changed_acceptance=True,
    before=False,
    after=True,
)
program.nn4_lineage()
program.neural_hot_loop_diagnostics()["nn4_lineage"]
```

This carries checkpoint hash, split hashes, calibration metrics, CUDA device,
influence audit metadata, and changed-acceptance evidence alongside the existing
hot-loop counters.

## Validation Staging

`scripts.validation_staging.ValidationStagingRun` is the XLOG-REPRO-006
repo-level staging helper for long-running evidence jobs. Writers put summaries
and artifacts under a fresh staging directory; `promote_if_pass({"status":
"PASS"})` copies staged artifacts into the canonical evidence directory. Failed
or canceled runs append `validation_events.jsonl` records and leave canonical
evidence untouched.

```python
run = ValidationStagingRun(Path("docs/evidence/latest"))
run.write_json("summary.json", {"status": "PASS"})
run.promote_if_pass({"status": "PASS"})
```

## Verification

The scoped v0.8.7 diagnostics checks are:

```bash
pytest -q python/tests/test_v080_delta_source.py \
  python/tests/test_v086_delta_coalescing.py \
  python/tests/test_v086_relation_callbacks.py \
  python/tests/test_v087_lwm_source.py \
  python/tests/test_pyxlog_evidence_api.py \
  python/tests/test_nn4_training_lineage.py \
  python/tests/test_validation_staging.py
cargo test -p xlog-cli --test explain_cli_tests
cargo test -p xlog-cli --test generated_rule_diagnostics
cargo test -p xlog-gpu --test biokg_streaming_relation_loader \
  --test relation_delta_planner_telemetry
cargo test -p xlog-induce
cargo check -p pyxlog --features extension-module,host-io
git diff --check
```

`cargo test -p pyxlog --lib` is not the gate for this cdylib extension crate in
this workspace because direct Rust test linking does not provide the Python
symbols required by PyO3. Use the pyxlog `cargo check` gate above plus the Python
source tests for this surface.
