# v0.8.6 G086_DELTA_COALESCE Evidence

Date: 2026-05-19
Goal node: G086_DELTA_COALESCE - Device-Resident Batch Relation Delta Coalescing
Branch: `feat/v086-runtime-completion`
Worktree: `.worktrees/v086-runtime-completion`
Goal document: `docs/plans/2026-05-19-agent-v086-dts-runtime-completion-goal.md`

## GDSP / GQM Trace

GDSP consumer goal: reduce repeated DTS-DLM Stage-4 and pyxlog persistent
session update overhead while preserving exact runtime results and the
zero-data-plane-D2H hot-path contract.

Existing xlog subsystem reused: `RelationDelta`, `RelationStore`,
`Executor::apply_deltas_and_recompute`, `LogicRelationSession`, and CUDA
provider full-row set operations (`dedup_full_row`, `diff_full_row`,
`union_gpu`). No Python-side delta engine or host-row coalescer was added.

GQM questions answered:

- Q086_DELTA.1: repeated insert/delete updates are coalesced before runtime
  recompute by `LogicProgram::apply_relation_delta_batch`.
- Q086_DELTA.2: coalesced, sequential, and full-replacement execution produce
  identical final `wmir_committed` query rows in the runtime test.
- Q086_DELTA.3: insert/delete cancellation is performed by CUDA provider
  full-row set operations; the coalescing implementation does not download
  data-plane relation rows.
- Q086_DELTA.4: the DTS-shaped `wmir_committed` fixture reduces recompute calls
  from 3 to 1 and delta rows entering recompute from 5 to 3.

## Artifacts

| Artifact | Purpose |
|---|---|
| `crates/xlog-gpu/src/logic.rs` | Runtime batch coalescing entrypoint, device set-operation cancellation, telemetry fields, runtime equivalence tests |
| `crates/pyxlog/src/logic.rs` | pyxlog session `apply_relation_delta_batch` bridge to the runtime batch path |
| `crates/pyxlog/src/lib.rs` | `LogicDeltaStats` fields for input count, coalesced rows, and canceled rows |
| `crates/pyxlog/python/pyxlog/_native.pyi` | Public pyxlog stub for batch delta updates |
| `docs/architecture/python-bindings.md` | Public API and telemetry documentation |
| `python/tests/test_v086_delta_coalescing.py` | Source guard for API/docs/stats and no host-row materialization in coalescing |
| `docs/evidence/2026-05-19-v086-delta-coalesce/measurements.json` | Raw row/update/transfer measurements from the certified fixture |

## Raw Measurements

| Measurement | Value |
|---|---|
| Fixture relation | `wmir_committed(u32)` |
| Input delta entries | `3` |
| Raw input insert rows | `4` |
| Raw input delete rows | `1` |
| Raw input delta rows | `5` |
| Coalesced insert rows | `3` |
| Coalesced delete rows | `0` |
| Canceled rows | `1` |
| Changed relations after coalescing | `1` |
| Coalesced output rows | `[1, 3, 4]` |
| Sequential output rows | `[1, 3, 4]` |
| Full replacement output rows | `[1, 3, 4]` |
| Hot-path `dtoh_bytes` before final output transfer | `0` |
| Hot-path `dtoh_calls` before final output transfer | `0` |
| Hot-path `d2h_transfer_count` before final output transfer | `0` |
| Sequential recompute calls | `3` |
| Coalesced recompute calls | `1` |
| Recompute-call reduction ratio | `3.0x` |
| Delta-row reduction ratio | `1.6666666667x` |

Interpretation: PASS. The final output download used by the test assertion is
excluded under the v0.8.6 host-transfer contract as the requested result
materialization.

## Metric Disposition

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M086_DELTA.1 API | pyxlog exposes `apply_relation_delta_batch` or approved equivalent | PASS | pyxlog native method, stub, and docs expose `apply_relation_delta_batch` |
| M086_DELTA.2 equivalence | coalesced, sequential, and full replacement outputs are byte-identical | PASS | `relation_delta_batch_updates_runtime_store_and_reports_coalesced_counts` compares `[1, 3, 4]` for all three paths |
| M086_DELTA.3 cancellation | insert/delete cancellation validated on device-resident rows | PASS | `coalesce_batch_cancels_insert_delete_pairs_on_device` validates cancellation via CUDA buffers and set ops |
| M086_DELTA.4 transfer budget | `dtoh_bytes=0`, `dtoh_calls=0` for coalescing hot path, excluding final requested output | PASS | runtime test records `dtoh_bytes=0`, `dtoh_calls=0`, `d2h_transfer_count=0` before final output download; Python source guard rejects coalescer D2H calls |
| M086_DELTA.5 performance | repeated-update fixture reduces upload/recompute work by at least 1.5x or removes a measured correctness blocker | PASS | recompute calls reduce `3 -> 1` (`3.0x`); delta rows entering recompute reduce `5 -> 3` (`1.6666666667x`) |
| M086_DELTA.6 telemetry | delta stats report coalesced inserts, coalesced deletes, canceled rows, affected SCCs, recomputed SCCs | PASS | `LogicDeltaStats` and packed pyxlog stats include `input_delta_count`, `coalesced_insert_rows`, `coalesced_delete_rows`, `canceled_rows`, plus existing SCC counters |

## Validation Commands

| Command | Result |
|---|---|
| `cargo test -p xlog-gpu relation_delta_batch_updates_runtime_store_and_reports_coalesced_counts` | exit 0; 1 passed |
| `pytest -q python/tests/test_v086_delta_coalescing.py` | exit 0; 3 passed |
| `cargo fmt --check` | exit 0 |
| `pytest -q python/tests/test_v086_delta_coalescing.py python/tests/test_v080_delta_source.py` | exit 0; 6 passed |
| `git diff --check` | exit 0 |
| `python scripts/validate_package_metadata.py` | exit 0 |
| `cargo test -p xlog-gpu` | exit 0; 2 lib tests, 13 integration tests, and 0 doc tests passed |
| `cargo test -p pyxlog --lib` | exit 0; 7 passed |
| `cargo check --workspace` | exit 0 |
| `cargo test -p xlog-runtime` | exit 0; 125 lib tests passed, 15 integration tests passed across 4 test binaries, 2 doc tests passed, 2 doc tests ignored |

## Next-Step Decision

G086_DELTA_COALESCE is ready to commit as a closed node. Proceed next to
G086_NOTIFY only after this implementation and evidence are committed. This
evidence does not authorize push, merge, tag, or release-board updates.
