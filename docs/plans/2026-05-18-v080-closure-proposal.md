# v0.8.0 DTS-DLM ML/Python Productization Closure Proposal

Date: 2026-05-18
Branch: `feat/v080-dts-ml-python-productization`
Last runtime-code commit: `30995c1e`
Integration evidence commit: `861f6a02`
Closure evidence commit: `8cd6e095`
Post-close examples addendum: `5314d532`

## Recommendation

`MERGE_READY`, pending coordinator authorization for the actual merge and
acceptance of the historical DTS-DLM 449-doc native-liveness evidence noted
under M080_EXACT.2.

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
| G080_BRIDGE | `63ef0298` + reuse remediation `30995c1e` | PASS | `docs/evidence/2026-05-18-v080-bridge/README.md` |
| G080_EXACT | `3c63cdb2` | PASS | `docs/evidence/2026-05-18-v080-exact/README.md` |
| G080_PROFILE | `bfa20ef5` | PASS | `docs/evidence/2026-05-18-v080-profile/README.md` |
| G080_INT | `861f6a02` | PASS | `docs/evidence/2026-05-18-v080-int/README.md` |
| G080_CLOSE | `8cd6e095` | PASS | `docs/evidence/2026-05-18-v080-close/README.md` |
| G080_EXAMPLES | `5314d532` | PASS | `docs/evidence/2026-05-18-v080-examples/README.md` |

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
| M080_BRIDGE.4 deterministic top-k | PASS | tie input produces `[2,0,1]`; registered-network `k`/`det` modes are applied by direct, complex, and batched neural forward/backward paths in `30995c1e` |
| M080_BRIDGE.5 neural cache telemetry | PASS | hits `3`, misses `1`, compile count `1` |
| M080_BRIDGE.6 repeated-query speedup | PASS | `1536.0x` |
| M080_EXACT.1 consumer path | PASS | DTS tensorized ILP calls public `induce_exact(..., backend=\"native\")` |
| M080_EXACT.2 liveness | PASS by accepted-evidence waiver | accepted DTS-DLM evidence records `449/449`; this xlog branch did not rerun the full DTS 449-doc job |
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
| M080_CLOSE.1 sub-goal table | PASS | every G080 node lists a commit SHA, including G080_CLOSE `8cd6e095` |
| M080_CLOSE.2 unresolved issues | PASS | deferred items have dispositions below |
| M080_CLOSE.3 release decision | PASS | `MERGE_READY` |
| M080_CLOSE.4 no implicit release | PASS | no push, tag, board update, or merge performed |
| M080_EXAMPLES.1 suite coverage | PASS | `example_count=5`; all five examples report `PASS` |
| M080_EXAMPLES.2 async/streaming | PASS | async completion true; stream chunks `[3,3,1]` and materialized chunks `[2,2,2,1]` |
| M080_EXAMPLES.3 relation deltas | PASS | insert/delete/mixed delta equivalence true; delta rows `4` vs full replacement rows `7` |
| M080_EXAMPLES.4 neural bridge | PASS | deterministic top-k selected labels `["reject","accept"]`; registered network reports `top_k=2`, `deterministic=true` |
| M080_EXAMPLES.5 native exact induction | PASS | Python/native parity `summary=true`, `ordered_candidates=true`; total scored `36` |
| M080_EXAMPLES.6 diagnostics | PASS | CUDA tensor checks true where applicable; host-transfer diagnostics report `dtoh=0`, `htod=0` where exposed |

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
  xlog native exact parity/D2H gates. If the coordinator requires a fresh
  449-doc DTS-DLM replay rather than accepting the prior evidence, this
  proposal should be treated as `HOLD_FOR_FIXES` until that rerun is complete.

## Prior Goal Reuse Applied

- G38 and Goal-039 closure proposals keep board edits, merge, push, and tag
  actions outside the proposal until explicit approval. This proposal follows
  the same gate split.
- Goal-038-B and Goal-039 use explicit surface-presence / non-production-replay
  language when a downstream replay is separate scope. M080_EXACT.2 uses the
  same convention: it is a PASS only by accepting the historical DTS-DLM
  449-doc evidence, not a claim that this xlog branch freshly replayed 449 docs.
- Goal-039 M37-A evidence records bounded focused probes and notes which
  long-running DTS artifacts were recomputed from canonical evidence instead of
  mutated or rerun. The v0.8.0 runtime probes are treated the same way, with
  `scripts/v080_pyxlog_runtime_probe.py` now committed for the PYAPI probe.
- Goal-039 also preserved the M37-A neural-symbolic training surface as an
  existing downstream substrate. Post-review code audit found that v0.8.0's
  standalone deterministic-top-k helper did not consume the already shipped
  `register_network(..., k=..., det=...)` configuration in neural training
  paths. Commit `30995c1e` fixes that by reusing `NetworkHandle.k` and
  `NetworkHandle.det` inside direct, complex, and batched forward/backward
  probability handling rather than introducing a parallel helper-only mode.

## Runtime Probe Auditability

| Probe | Audit status |
|-------|--------------|
| G080_PYAPI | Reproducible from committed command: `/tmp/xlog-v080-cert-venv/bin/python scripts/v080_pyxlog_runtime_probe.py --probe pyapi --output docs/evidence/2026-05-18-v080-pyapi/runtime_probe.json`. |
| G080_CERT | Manifest generation and verification are committed in `scripts/v080_dts_cert.py`; `runtime_probe.json` remains branch-local evidence consumed by that manifest. |
| G080_DELTA / G080_BRIDGE / G080_EXACT | Raw probe JSON is committed as branch-local evidence. The closure claim relies on committed source/runtime gates for pass/fail status; probe timing or environment-specific values are treated as observed evidence rather than independently scripted release gates. |
| G080_EXAMPLES | Reproducible from committed command: `/tmp/xlog-v080-cert-venv/bin/python scripts/validate_v080_examples.py --output docs/evidence/2026-05-18-v080-examples/validation_summary.json`. |

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
