# v0.8.6 Closure Evidence

Date: 2026-05-19
Branch: `feat/v086-runtime-completion`
Scope: `G086_CLOSE` evidence rollup and closure proposal.

## Artifacts

| Artifact | Purpose |
|---|---|
| `docs/plans/2026-05-19-v086-closure-proposal.md` | Coordinator-facing proposal with sub-goal table, GQM metric table, verification matrix, and release decision. |
| `closure_summary.json` | Machine-readable closure summary and release decision. |
| `docs/evidence/2026-05-19-v086-int/README.md` | Full integration/regression certification evidence. |
| `docs/evidence/2026-05-19-v086-consumers/validation_summary.json` | Consumer example and compatibility validator summary. |
| `ROADMAP.md` | v0.8.6 section synced to actual implemented state and unclaimed speedup metric. |

## Metric Status

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M086_CLOSE.1 sub-goal table | every G086 node listed with commit SHA and metric status | PASS | closure proposal and `closure_summary.json` |
| M086_CLOSE.2 roadmap sync | v0.8.6 section reflects actual PASS/BLOCKED states | PASS | `ROADMAP.md` persistent-index bullet names recorded background build and unclaimed timing speedup |
| M086_CLOSE.3 unresolved issues | all red/yellow metrics have explicit disposition | PASS | no unresolved scope blocker; persistent-index >=1.5x timing speedup is not claimed |
| M086_CLOSE.4 release decision | recommendation is `MERGE_READY`, `HOLD_FOR_FIXES`, or `SCOPE_AMENDMENT_REQUIRED` | PASS | `MERGE_READY` after final validation |
| M086_CLOSE.5 no implicit release | no push, tag, board update, or merge without authorization | PASS | no release action performed |
| M086_CLOSE.6 methodology audit | every sub-goal evidence file includes GDSP/GQM evidence | PASS | methodology scan and README amendments for G086_ADAPT/G086_INDEX |

## Validation Commands

| Command | Result |
|---|---|
| `python -m json.tool docs/evidence/2026-05-19-v086-close/closure_summary.json` | PASS |
| `python -m json.tool docs/evidence/2026-05-19-v086-int/validation_summary.json` | PASS |
| `cargo test -p xlog-runtime persistent_hash_index -- --nocapture` | PASS; 4 tests |
| `cargo test -p xlog-cuda test_recorded_join_index_build_runs_on_runtime_stream -- --nocapture` | PASS; 1 provider test |
| methodology heading scan over `docs/evidence/2026-05-19-v086-*` | PASS; 12 evidence files |
| `python scripts/validate_package_metadata.py` | PASS |
| `git diff --check` | PASS |

## Release Decision

`MERGE_READY` after final validation.

The previous persistent-index scope blocker has been amended. The branch now
implements and validates generation/schema/device keying, invalidation, budget
eviction, repeated-session reuse, background request/completion/deferred
telemetry, and a runtime-backed recorded provider build path. It still does not
claim a >=1.5x persistent-index timing speedup; the performance evidence is
bounded to observed retained-index reuse.

## Required Coordinator Decision

1. Review final validation output.
2. Authorize any release-board update, merge, push, and tag as separate actions.
3. v0.9.0 should rebase or merge after any accepted v0.8.6 landing.
