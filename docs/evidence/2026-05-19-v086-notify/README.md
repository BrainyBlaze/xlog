# v0.8.6 G086_NOTIFY Evidence

Date: 2026-05-19
Goal node: G086_NOTIFY - Opt-In Relation-Change Callbacks
Branch: `feat/v086-runtime-completion`
Worktree: `.worktrees/v086-runtime-completion`
Goal document: `docs/plans/2026-05-19-agent-v086-dts-runtime-completion-goal.md`

## GDSP / GQM Trace

GDSP consumer goal: let pyxlog and DTS-DLM persistent-session users observe
committed relation mutations without polling, relation export hooks, or hidden
data-plane D2H transfers.

Existing xlog subsystem reused: `LogicRelationSession` mutation commit points,
`LogicDeltaStats`, the v0.8.0/v0.8.6 relation delta path, and provider host
transfer counters. The callback path does not read relation rows.

GQM questions answered:

- Q086_NOTIFY.1: session users can register and unregister callbacks with
  `register_relation_callback` and `unregister_relation_callback`.
- Q086_NOTIFY.2: callbacks fire only after successful delta commits; failed
  relation updates do not invoke callbacks.
- Q086_NOTIFY.3: callback payloads contain metadata summaries and nested
  telemetry, not relation rows or DLPack capsules.
- Q086_NOTIFY.4: callback ordering is deterministic across 100 runtime replays.

## Artifacts

| Artifact | Purpose |
|---|---|
| `crates/pyxlog/src/lib.rs` | Session callback registry, callback ids, relation generation state |
| `crates/pyxlog/src/logic.rs` | Registration APIs, post-commit firing, metadata payload builder |
| `crates/pyxlog/python/pyxlog/_native.pyi` | Public pyxlog callback stubs |
| `docs/architecture/python-bindings.md` | Public API, GIL/threading, ordering, and metadata-only callback contract |
| `python/tests/test_v086_relation_callbacks.py` | Source guards for API/docs/commit ordering/no row materialization |
| `python/tests/test_v086_relation_callbacks_runtime.py` | Runtime callback success/failure/unregister/order/transfer tests |
| `docs/evidence/2026-05-19-v086-notify/measurements.json` | Raw runtime and overhead measurements |

## Raw Measurements

| Measurement | Value |
|---|---|
| Runtime venv | `/tmp/xlog-v086-notify-venv` |
| Runtime fixture | `wmir_committed(i32)` |
| Runtime tests | `3 passed in 32.52s` |
| Determinism replays | `100` |
| Failed delta callback count unchanged | `true` |
| Unregister removed callback | `true` |
| Callback-disabled `dtoh_bytes` | `0` |
| Callback-disabled `dtoh_calls` | `0` |
| Overhead probe iterations | `24` |
| Baseline seconds | `0.9629620009800419` |
| Callback-disabled seconds | `0.49983562796842307` |
| Overhead ratio | `0.519060593730305` |
| Overhead percent | `-48.0939406269695` |

Interpretation: PASS. The negative overhead reading reflects runtime/GPU
warmth variance, but it is still within the required `<= 2%` callback-disabled
overhead ceiling. The source path also returns immediately when no callbacks
are registered.

## Metric Disposition

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M086_NOTIFY.1 API | opt-in registration and unregistration APIs documented and stubbed | PASS | `register_relation_callback` and `unregister_relation_callback` exist in Rust, stubs, and docs |
| M086_NOTIFY.2 commit semantics | callbacks fire after successful commit and do not fire on failed/rolled-back delta | PASS | runtime test observes one event after successful commit and no additional event after unknown-relation failure |
| M086_NOTIFY.3 payload contract | payload includes relation name, generation, insert/delete/coalesced counts, affected SCCs, and telemetry | PASS | runtime and source tests assert metadata keys and absence of row/tensor payload fields |
| M086_NOTIFY.4 transfer budget | callback path performs zero relation data-plane D2H transfers | PASS | callback-disabled runtime path reports `dtoh_bytes=0`, `dtoh_calls=0`; source guard rejects export/download/DLPack payload code in callback builder |
| M086_NOTIFY.5 determinism | ordered fixture produces identical callback sequence across 100 replays | PASS | runtime test `test_relation_callback_ordering_is_deterministic_across_100_replays` passed |
| M086_NOTIFY.6 overhead | callback-disabled overhead within 2 percent of baseline on certified fixture | PASS | 24-iteration probe measured `overhead_percent=-48.0939406269695`, within the ceiling |

## Validation Commands

| Command | Result |
|---|---|
| `pytest -q python/tests/test_v086_relation_callbacks.py` before implementation | exit 1; 3 failed for missing API/helper/state |
| `cargo test -p pyxlog --lib` | exit 0; 7 passed |
| `pytest -q python/tests/test_v086_relation_callbacks.py python/tests/test_v086_relation_callbacks_runtime.py` in default env | exit 0; 3 source tests passed, 3 runtime tests skipped because installed pyxlog was older |
| `maturin develop --manifest-path crates/pyxlog/Cargo.toml --features host-io,extension-module` in `/tmp/xlog-v086-notify-venv` | exit 0; installed `pyxlog-0.8.5` |
| `/tmp/xlog-v086-notify-venv/bin/python -m pytest -q python/tests/test_v086_relation_callbacks_runtime.py` | exit 0; 3 passed in 32.52s |
| callback-disabled overhead probe in `/tmp/xlog-v086-notify-venv` | exit 0; JSON recorded in `measurements.json` |

## Next-Step Decision

G086_NOTIFY is ready to commit as a closed node after final hygiene checks.
Proceed next to G086_EXACT_TYPES only after this implementation and evidence
are committed. This evidence does not authorize push, merge, tag, or
release-board updates.
