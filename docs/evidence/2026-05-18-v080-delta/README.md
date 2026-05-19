# v0.8.0 Persistent Relation Delta Evidence

**Date:** 2026-05-18
**Branch:** `feat/v080-dts-ml-python-productization`
**Scope:** G080_DELTA DLPack-backed insert/delete relation deltas for `LogicRelationSession`.

---

## Artifacts

| Artifact | Purpose |
|----------|---------|
| `runtime_probe.json` | Branch-local pyxlog probe for the `wmir_committed` Stage-4 analog fixture. |
| `python/tests/test_v080_delta_source.py` | Source-surface regression for delta API exposure, runtime routing, and evidence presence. |
| `crates/xlog-runtime/src/executor/rewrite.rs` | Runtime `RelationDelta` / `apply_deltas_and_recompute` path with recompute stats. |
| `crates/xlog-gpu/src/logic.rs` | Cached session-store evaluation and delta application bridge. |
| `crates/pyxlog/src/logic.rs` | `LogicRelationSession.insert_relation`, `delete_relation`, `apply_relation_delta`, and `delta_stats`. |
| `docs/architecture/python-bindings.md` | Public delta API documentation. |

---

## Validation Commands

| Command | Result |
|---------|--------|
| `pytest -q python/tests/test_v080_delta_source.py` | exit 0; 3 passed |
| `cargo check -p pyxlog` | exit 0 |
| `cargo test -p xlog-runtime --lib apply_deltas_and_recompute` | exit 0; 2 passed |
| `/tmp/xlog-v080-cert-venv/bin/python` branch-local delta probe | exit 0; pyxlog `0.7.0` |

The runtime probe used the isolated virtualenv at `/tmp/xlog-v080-cert-venv`
after reinstalling this branch's locally built `pyxlog-0.7.0` wheel.

---

## Metric Status

| Metric | Target | Status | Evidence |
|--------|--------|--------|----------|
| M080_DELTA.1 API coverage | `insert_relation`, `delete_relation`, `apply_relation_delta`, or approved equivalent implemented | PASS | Native methods and stubs expose all three plus `delta_stats()` |
| M080_DELTA.2 equivalence | delta path and full replacement path bit-exact on certified fixtures | PASS | `runtime_probe.json`: insert and delete equivalence booleans are `true` |
| M080_DELTA.3 delete correctness | delete-containing deltas recompute affected SCCs correctly | PASS | `runtime_probe.json`: `delete_stats.recomputed_sccs=3`, `delete_equivalent_to_full_replacement=true` |
| M080_DELTA.4 monotone insert path | monotone insert-only SCC avoids full recompute where plan permits | PASS | `runtime_probe.json`: `insert_stats.incremental_sccs=3`, `insert_stats.recomputed_sccs=0` |
| M080_DELTA.5 DTS fixture | Stage-4 `wmir_committed` delta fixture committed and green | PASS | Probe fixture uses `wmir_committed(i32, i32)` and transitive `wmir_reachable`; delta uploads 2 rows vs. 5 full-replacement rows across two updates |

---

## API Notes

- `put_relation`, `remove_relation`, and `clear_relations` invalidate the cached runtime store.
- The next `session.evaluate()` after a direct store mutation performs a full plan run and caches the runtime store.
- `insert_relation`, `delete_relation`, and `apply_relation_delta` route through `RelationDelta` / `apply_deltas_and_recompute` and update the cached store.
- Insert-only monotone SCCs are updated without clearing prior output; delete-containing deltas clear and recompute affected SCCs.

No push, tag, release-board update, merge, or final v0.8.0 closure claim is authorized by this evidence.
