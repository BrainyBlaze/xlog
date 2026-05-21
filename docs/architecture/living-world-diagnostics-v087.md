# v0.8.7 Living-World Diagnostics Architecture

v0.8.7 adds the audit surfaces needed to explain living-world logic workloads:
generated rules, source rules, query proofs, relation deltas, temporal relation
metadata, and neural hot-loop runtime state can now be inspected through stable
Rust, CLI, and pyxlog APIs.

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
    pub debug_trace: Vec<String>,
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

## Verification

The scoped v0.8.7 diagnostics checks are:

```bash
pytest -q python/tests/test_v080_delta_source.py \
  python/tests/test_v086_delta_coalescing.py \
  python/tests/test_v086_relation_callbacks.py \
  python/tests/test_v087_lwm_source.py
cargo test -p xlog-cli --test explain_cli_tests
cargo test -p xlog-induce
cargo check -p pyxlog --features extension-module,host-io
git diff --check
```

`cargo test -p pyxlog --lib` is not the gate for this cdylib extension crate in
this workspace because direct Rust test linking does not provide the Python
symbols required by PyO3. Use the pyxlog `cargo check` gate above plus the Python
source tests for this surface.
