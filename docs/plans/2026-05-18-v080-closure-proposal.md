# v0.8.0 DTS-DLM ML/Python Productization Closure Proposal

Date: 2026-05-18
Branch: `feat/v080-dts-ml-python-productization`
Last runtime-code commit: `3c63cdb2`
Integration evidence commit: `861f6a02`
Closure evidence commit: reported by the final G080_CLOSE sub-goal report

## Recommendation

`MERGE_READY`, pending coordinator authorization for the actual merge.

This proposal does not authorize or perform a push, tag, release-board update,
or merge. v0.9.0 work must rebase or merge after v0.8.0 lands because this
branch is the release train that must land first.

## Sub-Goal Table

| Goal | Commit | Status | Evidence |
|------|--------|--------|----------|
| G080_PRE | `6c9a08c7` | PASS | `docs/evidence/2026-05-18-v080-pre/README.md` |
| G080_CERT | `1cde68b5` | PASS | `docs/evidence/2026-05-18-v080-cert/README.md` |
| G080_PYAPI | `f318ce5d` | PASS | `docs/evidence/2026-05-18-v080-pyapi/README.md` |
| G080_DELTA | `9d659d41` | PASS | `docs/evidence/2026-05-18-v080-delta/README.md` |
| G080_BRIDGE | `63ef0298` | PASS | `docs/evidence/2026-05-18-v080-bridge/README.md` |
| G080_EXACT | `3c63cdb2` | PASS | `docs/evidence/2026-05-18-v080-exact/README.md` |
| G080_PROFILE | `bfa20ef5` | PASS | `docs/evidence/2026-05-18-v080-profile/README.md` |
| G080_INT | `861f6a02` | PASS | `docs/evidence/2026-05-18-v080-int/README.md` |
| G080_CLOSE | final close sub-goal report | PASS when committed | `docs/evidence/2026-05-18-v080-close/README.md` |

## GQM Metric Table

| Metric | Status | Raw result |
|--------|--------|------------|
| M080_PRE.1 branch base | PASS | Base `656a8c62`; branch `feat/v080-dts-ml-python-productization` |
| M080_PRE.2 worktree status | PASS | clean before implementation |
| M080_PRE.3 baseline commands | PASS | `cargo fmt --check`, `cargo check -p pyxlog`, `cargo test -p pyxlog --lib`, `cargo test -p xlog-runtime --lib` recorded |
| M080_PRE.4 DTS references | PASS | M37-A+B and pyxlog 0.7 evidence listed |
| M080_CERT.1 API manifest | PASS | `pyxlog_api_manifest.json` committed |
| M080_CERT.2 required symbol coverage | PASS | `17/17` |
| M080_CERT.3 signature drift | PASS | `0` |
| M080_CERT.4 host-transfer delta | PASS | `dtoh_bytes=0`, `dtoh_calls=0`, `htod_bytes=0`, `htod_calls=0` |
| M080_CERT.5 determinism | PASS | `100/100` fixed-fixture replays bit-exact |
| M080_CERT.6 graph telemetry | PASS | CUDA graph counters available |
| M080_PYAPI.1 async API tests | PASS | source/runtime slice green |
| M080_PYAPI.2 streaming tests | PASS | chunks `[3,3,2]` equal non-streaming row set |
| M080_PYAPI.3 memory override tests | PASS | `512 MiB` accepted; `1 MiB` raises `MemoryError` |
| M080_PYAPI.4 progress counters | PASS | progress, memory, graph, host-transfer stats exposed |
| M080_PYAPI.5 docs | PASS | `docs/architecture/python-bindings.md` updated |
| M080_PYAPI.6 compatibility | PASS | old evaluate/session calls covered by integration slice |
| M080_DELTA.1 API coverage | PASS | `insert_relation`, `delete_relation`, `apply_relation_delta`, `delta_stats` |
| M080_DELTA.2 equivalence | PASS | insert/delete delta path equals full replacement |
| M080_DELTA.3 delete correctness | PASS | delete recomputes affected SCCs; `recomputed_sccs=3` |
| M080_DELTA.4 monotone insert path | PASS | insert path `incremental_sccs=3`, `recomputed_sccs=0` |
| M080_DELTA.5 DTS fixture | PASS | `wmir_committed` analog; delta uploads `2` rows vs `5` full replacement rows |
| M080_BRIDGE.1 gradient smoke | PASS | finite loss, `grad_norm=1.5468`, parameter update `0.3094` |
| M080_BRIDGE.2 DTS-shaped module | PASS | LearnedBridge-shaped fixture works |
| M080_BRIDGE.3 Belnap helper tests | PASS | documented formula values match |
| M080_BRIDGE.4 deterministic top-k | PASS | tie input produces `[2,0,1]` |
| M080_BRIDGE.5 neural cache telemetry | PASS | hits `3`, misses `1`, compile count `1` |
| M080_BRIDGE.6 repeated-query speedup | PASS | `1536.0x` |
| M080_EXACT.1 consumer path | PASS | DTS tensorized ILP calls public `induce_exact(..., backend=\"native\")` |
| M080_EXACT.2 liveness | PASS | accepted DTS evidence records `449/449` native liveness |
| M080_EXACT.3 safety gates | PASS | `rollback_rate=0.0`, `quarantine_rate=0.0` |
| M080_EXACT.4 type dispatch | PASS | `U64` retained; `U32` and `Symbol` explicitly deferred |
| M080_EXACT.5 packaging | PASS | no checked-in generated PTX; packaged `ilp_exact.portable.ptx` and cubin present |
| M080_PROFILE.1 profile evidence | PASS | G_PRE + W63 evidence inventoried |
| M080_PROFILE.2 improvement gate | N/A | no new optimizer/index implementation authorized; prior W63 response `933.200479x` |
| M080_PROFILE.3 non-regression | PASS | profile sub-goal docs-only |
| M080_INT.1 formatting | PASS | `cargo fmt --check` |
| M080_INT.2 pyxlog | PASS | Python slice `42 passed`; pyxlog lib `7 passed` |
| M080_INT.3 runtime | PASS | delta `2`, fixpoint `6`, recursive stats `10` passed |
| M080_INT.4 cuda | PASS | exact `3`, build script `4`, source audit `12` passed |
| M080_INT.5 integration | PASS | cert verify `17/17`; M37-A `4`; WCOJ clique `7`; WCOJ recursive `8`; W32 promoter `15` |
| M080_INT.6 docs | PASS | evidence JSON validates; cross-link scan passed |
| M080_INT.7 git hygiene | PASS | clean before INT evidence; no bulk generated artifacts |
| M080_CLOSE.1 sub-goal table | PASS | prior G080 nodes list commit SHAs; G080_CLOSE commit is reported by the final close sub-goal report |
| M080_CLOSE.2 unresolved issues | PASS | deferred items have dispositions below |
| M080_CLOSE.3 release decision | PASS | `MERGE_READY` |
| M080_CLOSE.4 no implicit release | PASS | no push, tag, board update, or merge performed |

## Deferred Or Non-Blocking Items

- `U32` and `Symbol` native exact-induction dispatch remain deferred until a
  downstream tensorized-ILP consumer needs them. DTS native exact evidence uses
  `U64`.
- Batch coalescing and relation-change callbacks are not implemented in this
  branch. G080_DELTA delivered the required insert/delete/apply-delta session
  surface and byte-equivalence fixture.
- Profile-gated optimizer/index work is intentionally not implemented. Current
  DTS profile evidence does not name CSE, adaptive re-optimization, or index
  rebuild cost as a blocker.
- The full DTS 449-doc native exact job was not relaunched in this xlog
  worktree. G080_EXACT anchors to accepted DTS-DLM evidence and reruns the
  xlog native exact parity/D2H gates.

## Remaining Risk

- Runtime probes used the local CUDA environment and branch-local pyxlog wheel.
  A coordinator may still choose to rerun the cert pack in the target DTS-DLM
  environment before merge.
- v0.9.0 branches must rebase or merge this branch after v0.8.0 lands.

## Required Coordinator Actions

1. Review this closure proposal and the evidence directories.
2. Decide whether to authorize merge.
3. If merge is authorized, perform or explicitly delegate the merge/push/tag
   steps. This branch has not performed them.
