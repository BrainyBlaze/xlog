# Agent Goal 040 - v0.8.0 DTS-DLM ML/Python Productization

**Agent:** Agent A, v0.8.0 implementation worker.
**Branch:** `feat/v080-dts-ml-python-productization`.
**Worktree:** `.worktrees/v080-dts`.
**Base:** `main` at or after `656a8c62` (`docs(roadmap): focus v080 on dts ml python productization`).
**Integration order:** v0.8.0 merges before v0.9.0. This branch is the release train that must land first.
**Status:** Dispatch-ready goal document. Implementation begins only after the worktree is created and baseline status is recorded.

## 0. Process Model

This goal uses the requested Goal-Driven Software Development Process and GQM framing.

Sources read:

- Goal-Driven Software Development Process, Wikipedia, accessed 2026-05-18:
  https://en.wikipedia.org/wiki/Goal-Driven_Software_Development_Process
- GQM, Wikipedia, accessed 2026-05-18:
  https://en.wikipedia.org/wiki/GQM

Method adaptation:

- GDSP says goals come before requirements and the technical platform feeds back into goal feasibility. This agent must keep top-down DTS-DLM release goals and bottom-up xlog implementation constraints visible at every sub-goal.
- GDSP links every top-level goal to questions that test the software against that goal after each iteration. This document therefore defines questions and measurements for every sub-goal.
- GDSP favors small goal sets and vertical ownership. This agent owns the v0.8.0 DTS-DLM ML/Python productization slice end-to-end and must not absorb v0.9.0 epistemic/solver work.
- GQM is used in three levels: conceptual goals, operational questions, and quantitative metrics. Each sub-goal below follows that shape.
- The execution loop is: plan, define, collect data, interpret. Every branch commit must leave evidence that can be interpreted against its metrics.

## 1. Business Goal

**BG080.** Ship xlog v0.8.0 as the production-grade DTS-DLM ML/Python substrate for the queued M37-A+B arc, while preserving v0.7.0 WCOJ behavior and pyxlog public compatibility.

The release is successful only if DTS-DLM can use xlog for:

- stable pyxlog runtime/session APIs;
- observable memory, CUDA Graph, and host-transfer behavior;
- incremental persistent relation updates;
- M37-A+B neural-symbolic bridge training surfaces;
- native exact-induction downstream consumer integration;
- reproducible DTS-DLM certification fixtures.

## 2. Scope Boundaries

### In Scope

- pyxlog public-surface compatibility manifest and diff check.
- Python runtime/session API: async evaluation, streaming result surfaces, per-call memory limits, progress reporting, and diagnostics.
- Persistent relation delta APIs on `LogicRelationSession`.
- DTS-DLM certification pack for Stage 4 and M37-A+B surfaces.
- Neural-symbolic bridge integration: term embeddings, foreign tensor predicates, neural output caching, deterministic top-k neural mode, Belnap-aware loss helpers, semantic loss variants required by M37-A+B, and circuit-cache telemetry.
- Native exact-induction consumer integration and downstream liveness reproduction.
- Profile-gated optimizer work only when DTS-DLM evidence shows it is a release blocker.

### Out Of Scope

- v0.9.0 epistemic semantics, EIR, G91, FAEEL, solver services, MaxSAT, SAT assumption solving.
- Broad CLI work: REPL, watch mode, explain visualization.
- Broad xlog-logic product work: list syntax, general meta-predicates, NAF, magic sets, aggregate lifting, approximate inference, unless a DTS-DLM cert explicitly blocks on it.
- Any push, tag, release-board update, or merge to `main` without explicit coordinator authorization.

### Coordination Locks

- Do not edit v0.9.0-owned files if Agent B creates them, except through a coordination commit.
- Do not change pyxlog public signatures without an explicit compatibility section explaining why old DTS-DLM call sites remain valid.
- Do not move Belnap pro/contra semantics into Stage-4 structural kernels. Belnap-aware work belongs to Python/ML training helpers and diagnostics.
- Do not fabricate unavailable GPU memory APIs. If the environment cannot report a metric, expose that absence explicitly and gate on documented availability.
- Do not treat local tests as closure. Every sub-goal needs evidence under `docs/evidence/`.

## 3. Roadmap Mapping

| ROADMAP area | v0.8.0 goal node | Agent responsibility |
|---|---|---|
| DTS-DLM Release Gates | G080_CERT | Build cert pack, surface manifest, zero-copy and determinism gates |
| Python Runtime And Session API | G080_PYAPI | Async, streaming, memory limit, progress, diagnostics |
| Persistent Relation Maintenance | G080_DELTA | Relation insert/delete/batch delta APIs and session equivalence fixture |
| Neural-Symbolic Bridge Integration | G080_BRIDGE | M37-A+B pyxlog training substrate and Belnap-aware loss helpers |
| Native Exact Induction Consumer Integration | G080_EXACT | Consumer path, 449/449 liveness, type dispatch, packaging decision |
| Profile-Gated Optimizer Work | G080_PROFILE | Only implement optimizer work proven hot by DTS profiles |
| Deferred Product Backlog | none | Do not implement unless re-authorized |

## 4. Goal Hierarchy

```
BG080 - xlog v0.8.0 DTS-DLM ML/Python substrate
 |
 +-- G080_PRE      Baseline inventory and worktree health
 +-- G080_CERT     DTS-DLM certification pack and pyxlog API manifest
 +-- G080_PYAPI    Python runtime/session API productization
 +-- G080_DELTA    Persistent relation delta maintenance
 +-- G080_BRIDGE   M37-A+B neural-symbolic bridge support
 +-- G080_EXACT    Native exact-induction downstream consumer integration
 +-- G080_PROFILE  Profile-gated optimizer/index work
 +-- G080_INT      Integration, regression, and cross-crate validation
 +-- G080_CLOSE    Evidence, roadmap sync, and closure proposal
```

## 5. GQM Decomposition

### G080_PRE - Baseline Inventory And Worktree Health

**Goal.** Establish a clean, reproducible v0.8.0 worktree for the purpose of implementing DTS-DLM ML/Python productization with respect to existing v0.7.0 behavior, from the viewpoint of a release engineer coordinating parallel v0.8.0 and v0.9.0 work.

**Questions.**

- Q080_PRE.1: Is the worktree created from the intended base commit?
- Q080_PRE.2: What is the current pyxlog and xlog test baseline?
- Q080_PRE.3: Which DTS-DLM fixtures and docs are authoritative for M37-A+B?

**Metrics.**

| Metric | Target |
|---|---|
| M080_PRE.1 branch base | `git merge-base HEAD main` equals the dispatch base or later approved base |
| M080_PRE.2 worktree status | clean before implementation begins |
| M080_PRE.3 baseline commands | `cargo fmt --check`, `cargo check -p pyxlog`, and targeted pyxlog/xlog-runtime tests recorded |
| M080_PRE.4 DTS references | M37-A+B plan, pyxlog 0.7 evidence, and M37-C' closure paths listed in evidence |

**Evidence.** `docs/evidence/<date>-v080-pre/README.md`.

### G080_CERT - DTS-DLM Certification Pack And API Manifest

**Goal.** Build a DTS-DLM certification pack for the purpose of proving xlog v0.8.0 preserves the public surfaces DTS-DLM relies on, with respect to API compatibility, zero-copy behavior, graph telemetry, and deterministic replay.

**Questions.**

- Q080_CERT.1: Does pyxlog expose every DTS-required symbol and signature?
- Q080_CERT.2: Can a bounded Stage-4 DTS-DLM fixture run without tracked hot-path host transfers?
- Q080_CERT.3: Are fixed fixtures bit-exact across repeated runs?
- Q080_CERT.4: Does the cert pack avoid requiring a full DTS pilot by default while still documenting how to run the full pilot?

**Metrics.**

| Metric | Target |
|---|---|
| M080_CERT.1 API manifest | machine-readable manifest committed under `docs/evidence/...` |
| M080_CERT.2 required symbol coverage | 100 percent of DTS-required pyxlog symbols present |
| M080_CERT.3 signature drift | 0 unapproved breaking changes |
| M080_CERT.4 host-transfer delta | `dtoh_bytes=0`, `dtoh_calls=0`, `htod_bytes=0`, `htod_calls=0` on certified hot path |
| M080_CERT.5 determinism | 100/100 fixed-fixture replays bit-exact |
| M080_CERT.6 graph telemetry | graph counters available or explicit unavailable reason recorded |

**Expected targets.**

- New or updated cert scripts under `scripts/` or `crates/xlog-integration/tests/`.
- Evidence under `docs/evidence/<date>-v080-cert/`.
- pyxlog stub updates if new public methods are added.

### G080_PYAPI - Python Runtime And Session API Productization

**Goal.** Extend pyxlog runtime/session ergonomics for the purpose of supporting long-running DTS-DLM pilots and M37-A+B training, with respect to nonblocking calls, bounded memory, progress visibility, and operational diagnostics.

**Questions.**

- Q080_PYAPI.1: Can Python callers start long evaluations without blocking the caller thread?
- Q080_PYAPI.2: Can large query outputs be consumed incrementally without losing DLPack zero-copy semantics?
- Q080_PYAPI.3: Can callers override memory budget per evaluation call?
- Q080_PYAPI.4: Can DTS-DLM log progress, memory, graph, and host-transfer telemetry without extra host transfers on the hot path?

**Metrics.**

| Metric | Target |
|---|---|
| M080_PYAPI.1 async API tests | pass for logic session and program evaluation |
| M080_PYAPI.2 streaming tests | chunked output equals non-streaming output as a row set |
| M080_PYAPI.3 memory override tests | per-call limit accepted and enforced; over-limit failure is typed |
| M080_PYAPI.4 progress counters | deterministic counters exposed for recursive and neural-symbolic long calls |
| M080_PYAPI.5 docs | `docs/architecture/python-bindings.md` updated |
| M080_PYAPI.6 compatibility | old `evaluate(...)` and `session.evaluate()` calls continue passing |

**Expected targets.**

- `crates/pyxlog/src/logic.rs`
- `crates/pyxlog/src/program.rs`
- `crates/pyxlog/python/pyxlog/_native.pyi`
- `docs/architecture/python-bindings.md`

### G080_DELTA - Persistent Relation Delta Maintenance

**Goal.** Add DLPack-backed relation deltas to `LogicRelationSession` for the purpose of reducing repeated full-table reuploads in DTS-DLM Stage 4, with respect to output equivalence and runtime delta correctness.

**Questions.**

- Q080_DELTA.1: Can Python callers insert, delete, and batch-update relation rows?
- Q080_DELTA.2: Does the session route deltas through runtime incremental recomputation where legal?
- Q080_DELTA.3: Does delta evaluation match full replacement evaluation exactly?
- Q080_DELTA.4: Does the fixture show lower full-table upload pressure on the DTS-DLM `wmir_committed` pattern?

**Metrics.**

| Metric | Target |
|---|---|
| M080_DELTA.1 API coverage | `insert_relation`, `delete_relation`, `apply_relation_delta`, or approved equivalent implemented |
| M080_DELTA.2 equivalence | delta path and full replacement path bit-exact on certified fixtures |
| M080_DELTA.3 delete correctness | delete-containing deltas recompute affected SCCs correctly |
| M080_DELTA.4 monotone insert path | monotone insert-only SCC avoids full recompute where plan permits |
| M080_DELTA.5 DTS fixture | Stage-4 `wmir_committed` delta fixture committed and green |

**Expected targets.**

- `crates/pyxlog/src/logic.rs`
- `crates/xlog-runtime/src/executor/rewrite.rs`
- `crates/pyxlog/python/pyxlog/_native.pyi`
- targeted tests in pyxlog and xlog-runtime

### G080_BRIDGE - M37-A+B Neural-Symbolic Bridge Support

**Goal.** Provide pyxlog neural-symbolic support for DTS-DLM M37-A+B for the purpose of making bridge training measurable and reproducible, with respect to gradient flow, Belnap-aware reward diagnostics, and circuit-cache performance.

**Questions.**

- Q080_BRIDGE.1: Does `forward_backward_tensor` remain callable and gradient-carrying for a LearnedBridge-shaped module?
- Q080_BRIDGE.2: Are term embeddings and foreign tensor predicates available where M37-A+B needs GPU-resident features?
- Q080_BRIDGE.3: Can Belnap-aware loss helpers express pro reward, contra penalty, quarantine penalty, and CFR-oriented diagnostics without moving semantics into Stage 4 kernels?
- Q080_BRIDGE.4: Is circuit-cache behavior observable and good enough for repeated M37-A+B queries?

**Metrics.**

| Metric | Target |
|---|---|
| M080_BRIDGE.1 gradient smoke | finite CUDA loss, nonzero gradient, parameter update observed |
| M080_BRIDGE.2 DTS-shaped module | LearnedBridge-shaped fixture works with pyxlog network registration |
| M080_BRIDGE.3 Belnap helper tests | pro/contra/quarantine helper outputs match documented formulas |
| M080_BRIDGE.4 deterministic top-k | fixed seed and tie inputs produce stable ordered results |
| M080_BRIDGE.5 neural cache telemetry | hit/miss counters available |
| M080_BRIDGE.6 repeated-query speedup | at least 50x on cache-hit microbench or RCA + amended target |

**Expected targets.**

- `crates/pyxlog/src/neural.rs`
- `crates/pyxlog/src/program.rs`
- `crates/pyxlog/python/pyxlog/_native.pyi`
- `docs/architecture/python-bindings.md`
- `docs/evidence/<date>-v080-bridge/`

### G080_EXACT - Native Exact-Induction Consumer Integration

**Goal.** Complete native exact-induction downstream integration for the purpose of replacing prototype-only paths in tensorized ILP consumers, with respect to type coverage, DTS-DLM liveness, and packaging policy.

**Questions.**

- Q080_EXACT.1: Can downstream tensorized ILP invoke native exact induction directly?
- Q080_EXACT.2: Does native exact induction reproduce the 449/449 downstream liveness benchmark?
- Q080_EXACT.3: Are `U32` and `Symbol` callers supported when needed?
- Q080_EXACT.4: Is strict-per-topology behavior documented against legacy Python prototype behavior?
- Q080_EXACT.5: Is the `ilp_exact.ptx` packaging decision resolved?

**Metrics.**

| Metric | Target |
|---|---|
| M080_EXACT.1 consumer path | downstream tensorized ILP calls native backend without private hooks |
| M080_EXACT.2 liveness | 449/449 benchmark reproduced |
| M080_EXACT.3 safety gates | rollback and quarantine rates unchanged from accepted baseline |
| M080_EXACT.4 type dispatch | `U64` retained; `U32` and `Symbol` supported or explicitly deferred with evidence |
| M080_EXACT.5 packaging | committed PTX or documented no-PTX policy aligned with ILP-family convention |

**Expected targets.**

- `crates/pyxlog/src/ilp_exact.rs`
- `crates/pyxlog/src/ilp.rs`
- `crates/pyxlog/python/pyxlog/ilp/exact_induce.py`
- `kernels/ilp_exact.cu` and packaging metadata
- evidence under `docs/evidence/<date>-v080-exact/`

### G080_PROFILE - Profile-Gated Optimizer And Index Work

**Goal.** Implement only profile-justified optimizer/index work for the purpose of removing DTS-DLM release blockers without expanding v0.8.0 into a broad optimizer release.

**Questions.**

- Q080_PROFILE.1: Do DTS-DLM fixtures show duplicated subplans that justify CSE?
- Q080_PROFILE.2: Do fixtures show stable runtime mis-planning that justifies adaptive re-optimization?
- Q080_PROFILE.3: Does index rebuild cost block session-delta performance?

**Metrics.**

| Metric | Target |
|---|---|
| M080_PROFILE.1 profile evidence | required before any optimizer/index implementation |
| M080_PROFILE.2 improvement gate | implemented change improves profiled bottleneck by at least 1.2x or removes a correctness blocker |
| M080_PROFILE.3 non-regression | no DTS cert or WCOJ regression |

**Expected targets.** No file target is authorized until profile evidence names the bottleneck.

### G080_INT - Integration And Regression Gate

**Goal.** Validate the composed v0.8.0 branch for the purpose of preparing release integration, with respect to tests, docs, DTS-DLM certification, and v0.7.0 behavior preservation.

**Metrics.**

| Metric | Target |
|---|---|
| M080_INT.1 formatting | `cargo fmt --check` pass |
| M080_INT.2 pyxlog | pyxlog targeted and compatibility tests pass |
| M080_INT.3 runtime | xlog-runtime delta and recursive tests pass |
| M080_INT.4 cuda | xlog-cuda touched-surface tests pass |
| M080_INT.5 integration | DTS certification pack pass |
| M080_INT.6 docs | roadmap, architecture docs, and evidence cross-links consistent |
| M080_INT.7 git hygiene | no unrelated files, no generated bulk artifacts unless documented |

### G080_CLOSE - Evidence And Closure Proposal

**Goal.** Produce a closure proposal for the purpose of letting the coordinator decide whether v0.8.0 can merge, with respect to the full GQM tree and evidence trail.

**Metrics.**

| Metric | Target |
|---|---|
| M080_CLOSE.1 sub-goal table | every G080 node listed with commit SHA and metric status |
| M080_CLOSE.2 unresolved issues | all red/yellow metrics have explicit disposition |
| M080_CLOSE.3 release decision | recommendation is one of MERGE_READY, HOLD_FOR_FIXES, or SCOPE_AMENDMENT_REQUIRED |
| M080_CLOSE.4 no implicit release | no push, tag, board update, or merge without coordinator authorization |

## 6. KPIs

| KPI | Gate |
|---|---|
| KPI080.1 DTS pyxlog API compatibility | 100 percent required symbols present; 0 unapproved breaking signatures |
| KPI080.2 DTS zero-copy hot path | 0 tracked D2H/H2D bytes and calls on certified hot path |
| KPI080.3 Deterministic replay | 100/100 fixed-fixture runs bit-exact |
| KPI080.4 M37-A+B gradient viability | finite CUDA loss, nonzero gradient, parameter update |
| KPI080.5 Circuit-cache performance | at least 50x repeated-query speedup or approved RCA/amendment |
| KPI080.6 Native exact liveness | 449/449 downstream benchmark reproduced |
| KPI080.7 Session delta equivalence | delta path byte-identical to full replacement path |
| KPI080.8 Release hygiene | all evidence committed; no unapproved merge, push, or tag |

## 7. Worktree Setup

Coordinator command:

```bash
cd /home/dev/projects/xlog
git worktree add .worktrees/v080-dts -b feat/v080-dts-ml-python-productization
cd .worktrees/v080-dts
git status --short --branch
```

Agent first actions:

1. Record `git rev-parse HEAD` and `git status --short --branch`.
2. Read this document, `ROADMAP.md` v0.8.0, `docs/architecture/python-bindings.md`, `docs/architecture/bounded-exact-induction.md`, and DTS-DLM `docs/plans/2026-05-19-m37a-plus-b-plan-freeze.md`.
3. Create `docs/evidence/<date>-v080-pre/README.md` with baseline state.
4. Execute G080_PRE before any implementation.

## 8. Reporting Protocol

After each sub-goal, report:

- branch and commit SHA;
- files changed;
- metrics table with PASS/FAIL/BLOCKED;
- exact validation commands and raw numbers;
- next sub-goal or halt condition.

Do not compress multiple sub-goals into a single "done" report unless all metric evidence is present.

## 9. Halt Conditions

Halt and ask the coordinator if:

- a required DTS-DLM surface is missing or incompatible in a way that needs API redesign;
- a fix requires editing v0.9.0-owned semantics/solver files;
- certification requires running a full DTS-DLM pilot that exceeds local resource bounds;
- any change would require push, tag, merge, or release-board mutation;
- a metric needs to be weakened or redefined.

## 10. Final Deliverable

The final output is a v0.8.0 closure proposal with:

- complete GQM metric table;
- evidence links;
- all sub-goal commit SHAs;
- remaining risk summary;
- merge recommendation;
- explicit statement that v0.9.0 must rebase or merge after v0.8.0 lands.
