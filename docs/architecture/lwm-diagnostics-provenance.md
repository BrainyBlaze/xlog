# Living-World Diagnostics and Provenance

This document records the v0.8.8 architecture surface added after the BFO
living-world model example reported six upstream XLOG gaps. The goal is to move
diagnostics, provenance, delta evidence, temporal metadata, and neural hot-loop
audits out of example-specific Python glue and into reusable XLOG/pyxlog APIs.

## Scope

The v0.8.8 work covers these upstream issues:

| Finding | Area | Runtime surface |
| --- | --- | --- |
| `XLOG-LWM-001` | native induction provenance | `InducedRuleProvenance`, `InducedRuleRegistry` |
| `XLOG-LWM-002` | rule provenance introspection | CLI and pyxlog `rule_provenance()` |
| `XLOG-LWM-003` | delta debugging | `LogicRelationSession.apply_relation_delta_debug(...)` |
| `XLOG-LWM-004` | stream temporal provenance | `pyxlog.put_temporal_relation(...)`, `pyxlog.temporal_provenance(...)` |
| `XLOG-LWM-005` | proof traces | CLI and pyxlog `proof_traces()` |
| `XLOG-LWM-006` | neural hot-loop diagnostics | `CompiledProgram.neural_hot_loop_diagnostics()` |

## Native Induction Provenance

`xlog-induce` now exposes a provenance bundle for candidate rules:

- `RuleSourceKind`: `Source`, `Generated`, `Mined`, `Imported`,
  `RuntimeInjected`.
- `InductionSupportRow`: relation name, row index, and row hash for support
  evidence.
- `InductionAlternative`: rejected candidate fragment plus support and
  falsification counts.
- `InducedRuleProvenance`: generated rule id, rule fragment, generation trace
  hash, search-space size, predicate inventory, support rows, rejected
  alternatives, and falsification count.
- `InducedRuleRegistry`: insertion-ordered registry for induced/generated rule
  provenance records.

This is intentionally metadata-only. The registry does not rerun statistical
search; it gives native XLOG consumers a stable place to register and inspect
candidate-rule provenance that the living-world example previously had to carry
in Python artifacts.

## Rule Provenance and Proof Traces

`xlog-logic` owns the shared diagnostic data models:

- `RuleProvenance`: stable `rule_id`, `source_kind`, head, optional source
  span, optional generation trace hash, support relation ids, and
  counterexample relation ids.
- `QueryProofTrace`: query text, internal answer relation, deriving rule ids,
  source facts, and rejected alternatives.

`xlog explain --format json` emits both `rule_provenance` and `proof_traces`.
The CLI surface distinguishes source-authored rules from generated rules such as
magic-set helper predicates. pyxlog exposes the same data through
`CompiledLogicProgram.rule_provenance()`, `LogicRelationSession.rule_provenance()`,
`CompiledProgram.rule_provenance()`, and matching `proof_traces()` methods.

The proof trace is deliberately structural. It names the source facts and rule
ids that can derive a query answer; it does not materialize every answer tuple
or claim full proof enumeration.

## Delta Debugging

`LogicRelationSession.apply_relation_delta_debug(...)` wraps the existing
runtime delta path and returns a metadata dictionary with:

- `changed_relation_names`
- `equivalence_checked`
- `equivalent_to_full_recompute`
- nested `delta_stats`
- compact `debug_trace`

When `check_equivalence=True`, the method compares the session-managed delta
answer against a full recompute shape check. This covers the relation/SCC
diagnostics that the BFO living-world workload had to assemble manually.

## Temporal Provenance Helpers

pyxlog adds two Python helpers:

```python
pyxlog.put_temporal_relation(
    session,
    "observation",
    [entity, timestamp, value],
    timestamp_column="timestamp",
    dataset_id="living-world",
    row_hashes=row_hashes,
    field_hashes=field_hashes,
    uncertainty=uncertainty,
    stream_id="camera-0",
)
meta = pyxlog.temporal_provenance(session, "observation")
```

The helper delegates the actual relation load to `session.put_relation(...)` and
records timestamp, dataset, row hash, field hash, uncertainty, stream id,
process-boundary, and temporal-order metadata for later diagnostics.

## Neural Hot-Loop Diagnostics

`CompiledProgram.neural_hot_loop_diagnostics()` returns one dictionary for the
`nn/4` hot-loop audit:

- `post_load_dtoh_bytes`
- `post_load_htod_bytes`
- `post_load_dtoh_calls`
- `post_load_htod_calls`
- `control_plane_bytes_per_iteration`
- `scalar_sync_checks`
- `cuda_graph`
- `circuit_cache`

This combines provider transfer counters, CUDA graph counters, and circuit-cache
telemetry without requiring example-specific audit glue.

## Regression Coverage

The v0.8.8 regression coverage is intentionally scoped to metadata and compile
surfaces:

- `crates/xlog-induce/tests/rule_induction_provenance.rs`
- `crates/xlog-cli/tests/explain_cli_tests.rs`
- `python/tests/test_v088_lwm_source.py`

Direct smoke checks against the BFO reproducers use:

```bash
./target/debug/xlog explain --format json \
  .worktrees/bfo-living-world-model/examples/BFO/living_world_model/repro/02_rule_provenance_equivalence.xlog
./target/debug/xlog explain --format json \
  .worktrees/bfo-living-world-model/examples/BFO/living_world_model/repro/05_contradiction_without_trace.xlog
```
