# Agent Goal 043 - v0.8.6 DTS-DLM Runtime Completion and GPU-Native Optimizer Pack

**Agent:** Agent D, v0.8.6 runtime/optimizer completion worker.
**Branch:** `feat/v086-runtime-completion`.
**Worktree:** `.worktrees/v086-runtime-completion`.
**Base:** `main` after the v0.8.5 release and after commit `9914f9c5`
(`docs(v080): commit DTS ML Python goal doc`).
**Integration order:** v0.8.6 lands after v0.8.5 and before v0.9.0. The
active v0.9.0 epistemic/solver branch must rebase or merge after v0.8.6
because exact induction, persistent sessions, CSE, adaptive planning, and
index reuse are substrate features for fully GPU-native epistemic execution.
**Status:** Dispatch-ready goal document. Implementation begins only after the
worktree is created, baseline status is recorded, and G086_PRE closes.

## 0. Methodology Implementation

This is a methodology-driven goal document, not a bibliography. The requested
GDSP and GQM material is operationalized as enforceable structure:

### 0.1 GDSP Applied To v0.8.6

| GDSP principle | Concrete v0.8.6 rule |
|---|---|
| Start from goals before requirements | BG086 defines the release purpose before any file list or implementation task. Requirements are derived from consumer goals, not from available code paths alone. |
| Collaborative goal identification | DTS-DLM, Mistaber, v0.9.0 epistemic/solver, and pyxlog users are named as consumers. Each feature must map to at least one consumer need. |
| Top-down and bottom-up convergence | Each sub-goal must connect a top-down consumer outcome to an existing bottom-up xlog subsystem. If the existing subsystem cannot support the goal, the agent must amend the goal or halt; it must not create an unreviewed parallel engine. |
| Vertical ownership | Agent D owns the full runtime completion slice end-to-end: runtime, CUDA provider, pyxlog surface, docs, examples, evidence, and closure. It must not hand unresolved cross-layer work to a later release without a BLOCKED metric. |
| Minimize project size | v0.8.6 is limited to exactly seven deferred v0.8.0 items plus integration and closure. v0.9.0 epistemic semantics and v0.10.0 multi-GPU/out-of-core work are excluded. |
| Iterative delivery | Every G086 node closes independently with a commit SHA, evidence, metric interpretation, and next-step decision before the following node starts. |
| Architecture-goal bridge | The architectural contract in §1.2 is a gate: accepted features must flow through production xlog parser/RIR/PIR/optimizer/runtime/provider/pyxlog paths. |

### 0.2 GQM Applied To v0.8.6

Every sub-goal follows the GQM hierarchy:

1. **Conceptual goal:** what object is improved, why, for whom, and under which
   quality constraints.
2. **Operational questions:** what must be true to judge the goal achieved.
3. **Quantitative metrics:** what data answers those questions, with target
   values.
4. **Data collection:** exact commands, probes, counters, transfer budgets,
   benchmarks, and consumer fixtures recorded in evidence.
5. **Interpretation:** each metric becomes `PASS`, `FAIL`, `BLOCKED`,
   `WAIVED_BY_EXPLICIT_AUTHORIZATION`, or `NOT_APPLICABLE_WITH_REASON`.

No metric may pass because prose says it is acceptable. It passes only by
recorded data meeting the stated target, or by explicit coordinator
authorization recorded as a waiver.

### 0.3 Required Methodology Evidence

Each sub-goal evidence README must include:

- the GDSP consumer goal it serves;
- the existing xlog subsystem reused;
- the GQM questions answered;
- the raw measurements collected;
- the interpretation for every metric;
- any unresolved gap, marked `BLOCKED` rather than hidden in notes.

## 1. Business Goal

**BG086.** Ship xlog v0.8.6 as the production-grade completion pack for the
v0.8.0 DTS-DLM runtime substrate by closing every remaining v0.8.0 deferred
runtime/optimizer item with fully GPU-native, consumer-validated, zero
data-plane host-transfer implementation.

The release is successful only if xlog gains:

- device-resident batch update coalescing for repeated session relation deltas;
- opt-in relation-change notification callbacks that do not force data-plane
  host transfers;
- native exact-induction dispatch for `U32` and `Symbol` pair relations in
  addition to `U64`;
- profile-gated shared-memory caching for chain-topology exact induction;
- GPU-native common subexpression elimination for duplicated subplans;
- adaptive query re-optimization driven by runtime telemetry and deterministic
  rollback;
- persistent hash index management with background GPU-resident build,
  invalidation, and budget-aware reuse;
- consumer certification for DTS-DLM, Mistaber `.xlog` workloads, v0.9.0
  epistemic/solver prerequisites, and general pyxlog session users.

## 1.1 Consumers

v0.8.6 has four first-class consumers:

| Consumer | Required value | v0.8.6 responsibility |
|---|---|---|
| DTS-DLM | Stage-4 and M37 follow-up workloads must avoid repeated upload/recompute churn and must keep native exact induction available beyond `U64` when relation schemas require it. | Delta coalescing, callbacks, exact type dispatch, profile-backed optimizer/index gates, certification fixtures. |
| Mistaber | Mistaber must be expressible as `.xlog` programs using scientific/engineering names rather than project-specific terminology, and it must run through production optimizer/runtime paths. | CSE, adaptive re-optimization, persistent index fixtures, consumer examples without terminology leakage. |
| v0.9.0 epistemic/solver | Fully GPU-native epistemic/probabilistic execution needs reusable runtime primitives rather than CPU solver shortcuts or fixture-only paths. | Exact type dispatch, persistent index reuse, CSE/adaptive plan interfaces, zero host-transfer guardrails. |
| General pyxlog users | Long-running persistent sessions need predictable updates, observability, and performance without hidden transfers. | Session delta coalescing, notifications, API docs, compatibility tests. |

## 1.2 Architectural Contract

Accepted v0.8.6 execution must follow production paths:

```text
Python / CLI / .xlog input
  -> typed parser / pyxlog relation schema
  -> RIR / PIR / exact-induction request / runtime plan
  -> production optimizer and runtime/provider dispatch
  -> CUDA kernels over device-resident buffers
  -> final requested result transfer only
```

Allowed CPU responsibilities:

- parsing, type checking, and static planning;
- runtime metadata bookkeeping that does not inspect data-plane relation values;
- launch orchestration and control-plane counters;
- callback dispatch over committed metadata summaries;
- final requested result materialization;
- diagnostics formatting and evidence generation.

Blocked CPU responsibilities:

- evaluating accepted relation deltas by downloading full relation columns;
- coalescing updates by reading data-plane rows on the host;
- exact-induction scoring by converting accepted `U32` or `Symbol` buffers to
  host-side `U64` vectors;
- chain-topology acceleration via CPU precomputation;
- CSE by materializing intermediate relation rows on the host;
- adaptive re-optimization by sampling device relation data through hidden D2H
  transfers;
- persistent index build/rebuild on host-side mirrors of device buffers;
- fixture-only success paths that bypass production runtime/provider code.

Host-transfer accounting:

- **Data-plane D2H/H2D budget:** `0` for accepted hot paths.
- **Control-plane exceptions:** scalar row counts, kernel status, compact
  telemetry, final requested outputs, and explicit diagnostic snapshots.
- **Every exception must be named in evidence** with byte/call counts and the
  reason it is not data-plane execution.

## 1.3 Conceptual And Theoretical Foundations

v0.8.6 must be grounded in the following foundations:

- **Incremental view maintenance:** Relation deltas compose through insert and
  delete algebra. Coalescing must be confluent: applying `D1` then `D2` must
  match applying the coalesced delta `D1+D2`, modulo cancellation of insert/delete
  pairs.
- **Datalog fixed-point semantics:** Delta coalescing, CSE, persistent indexes,
  and adaptive re-optimization must preserve stratified and semi-naive fixpoint
  results exactly.
- **Relational algebra equivalence:** CSE may share subplans only when the
  subplan key, projection, selection, aggregate, negation, and provenance
  dependencies are equivalent under the current rule stratum.
- **GPU memory hierarchy:** Shared-memory caching is accepted only when it
  reduces global-memory pressure on the exact-induction chain topology without
  altering strict per-topology semantics.
- **Adaptive query optimization:** Re-optimization decisions must be driven by
  runtime telemetry, deterministic thresholds, and rollback safety. No plan swap
  may change answers.
- **Materialized access paths:** Persistent indexes are derived data structures
  tied to relation version/generation ids. Invalidation must be complete before
  reuse.
- **Typed physical layout:** `U32` and `Symbol` may share physical width, but
  logical schemas must remain distinct in public APIs, errors, and evidence.

## 2. Scope Boundaries

### In Scope

- Batch relation delta coalescing for repeated session-managed updates.
- Relation-change notification callbacks over explicit opt-in session APIs.
- Native exact-induction type dispatch for `U64`, `U32`, and `Symbol`.
- Profile-gated chain-topology shared-memory caching.
- GPU-native CSE over deterministic, probabilistic, and exact-induction-safe
  subplans where equivalence can be proven.
- Adaptive runtime re-optimization for stable mis-planning signatures.
- Persistent hash index manager with background build, invalidation, reuse,
  and memory budget integration.
- Consumer certification fixtures for DTS-DLM and Mistaber.
- v0.9.0 substrate notes for epistemic/solver agents.
- ROADMAP, CHANGELOG, architecture docs, examples, evidence, and closure
  proposal updates.

### Out Of Scope

- v0.9.0 EIR, G91, FAEEL, world-view semantics, MaxSAT, and epistemic splitting.
- v0.10.0 multi-GPU, out-of-core spilling, distributed execution, and checkpoint
  recovery.
- Arbitrary dynamic database mutation beyond explicit session deltas.
- CPU-only fallback for any accepted v0.8.6 hot path.
- Public terminology from Mistaber that should be replaced by scientific or
  engineering equivalents in xlog examples.
- Push, tag, merge, or release-board updates without coordinator authorization.

### Coordination Locks

- **Reuse existing codebase.** Every sub-goal must name the existing xlog
  subsystem it extends. Reimplementing parser, runtime, optimizer,
  exact-induction, CUDA provider, probabilistic, WCOJ, or pyxlog behavior is a
  blocker unless the closure proposal explicitly deprecates and removes the old
  subsystem.
- Reuse existing production paths from goals 38, 38-B, 39, v0.8.0, and v0.8.5.
- Do not weaken existing zero-D2H, deterministic replay, WCOJ, exact induction,
  or pyxlog compatibility gates.
- Do not treat profile-gated work as optional. If current profiles do not expose
  the bottleneck, add consumer-shaped certification fixtures that do.
- Do not accept a fixture-only implementation. Every feature needs source-level
  tests, runtime tests, and evidence tying it to production dispatch.
- Do not create v0.9.0-only APIs; reusable substrate surfaces must be versioned,
  documented, and compatible with v0.8.5 language contracts.

## 3. Roadmap Mapping

| ROADMAP item | v0.8.6 goal node | Agent responsibility |
|---|---|---|
| Batch update coalescing for repeated `wmir_committed` updates | G086_DELTA_COALESCE | Device-resident coalescing, cancellation, equivalence, telemetry |
| Change notification callbacks | G086_NOTIFY | Opt-in pyxlog callbacks over metadata summaries |
| Exact induction `U32` / `Symbol` dispatch | G086_EXACT_TYPES | Typed native kernels/dispatch, parity, D2H guardrails |
| Chain-topology shared-memory caching | G086_CHAIN_SMEM | Profile gate, kernel optimization, strict semantic parity |
| Common subexpression elimination | G086_CSE | Equivalence analysis, plan sharing, runtime counters |
| Adaptive query re-optimization | G086_ADAPT | Telemetry, deterministic thresholds, rollback, replay |
| Persistent hash index manager | G086_INDEX | Background build, invalidation, budget-aware reuse |

## 4. Baseline Facts To Verify In G086_PRE

Agent D must verify and record these facts before implementation:

- `ROADMAP.md` contains a planned v0.8.6 milestone with seven open items.
- `docs/plans/2026-05-18-agent-v080-dts-ml-python-goal.md` is committed on
  `main`.
- `crates/pyxlog/src/logic.rs` exposes `insert_relation`, `delete_relation`,
  `apply_relation_delta`, and `delta_stats`, but not coalesced delta batches or
  relation-change callbacks.
- `crates/xlog-induce/src/lib.rs` validates exact-induction pair buffers as
  `U64` only.
- `docs/architecture/bounded-exact-induction.md` explicitly defers `U32`,
  `Symbol`, and chain shared-memory caching.
- Current optimizer/runtime code has no v0.8.6 CSE/adaptive/persistent-index
  implementation that satisfies this goal.

## 5. Goal Hierarchy

```text
BG086 - v0.8.6 runtime completion and GPU-native optimizer pack
 |
 +-- G086_PRE             Baseline inventory and worktree health
 +-- G086_DELTA_COALESCE  Device-resident batch relation delta coalescing
 +-- G086_NOTIFY          Opt-in relation-change callbacks
 +-- G086_EXACT_TYPES     Native exact-induction U32/Symbol dispatch
 +-- G086_CHAIN_SMEM      Chain-topology shared-memory exact scorer
 +-- G086_CSE             GPU-native common subexpression elimination
 +-- G086_ADAPT           Adaptive runtime re-optimization
 +-- G086_INDEX           Persistent hash index manager
 +-- G086_CONSUMERS       DTS-DLM, Mistaber, v0.9.0, and pyxlog certification
 +-- G086_INT             Integration, regression, and performance gates
 +-- G086_CLOSE           Evidence, roadmap sync, and closure proposal
```

Execution order is the hierarchy order above unless G086_PRE evidence proves a
different order reduces risk. G086_CONSUMERS must run after all feature nodes
and before G086_INT.

## 6. GQM Decomposition

### G086_PRE - Baseline Inventory And Worktree Health

**Goal.** Establish a clean v0.8.6 worktree for the purpose of closing the
v0.8.0 deferred runtime/optimizer items with respect to current v0.8.5 main,
consumer evidence, and v0.9.0 substrate needs.

**Questions.**

- Q086_PRE.1: Is the branch cut from the intended post-v0.8.5 base?
- Q086_PRE.2: Are the seven v0.8.6 roadmap items present and unmapped from
  closed v0.8.0?
- Q086_PRE.3: Which crates and tests own each feature?
- Q086_PRE.4: Which current profiles and consumer fixtures are authoritative?

**Metrics.**

| Metric | Target |
|---|---|
| M086_PRE.1 branch base | `git merge-base HEAD main` equals the approved base or later approved base |
| M086_PRE.2 worktree status | clean before implementation begins |
| M086_PRE.3 backlog map | all seven items mapped to G086 nodes |
| M086_PRE.4 ownership map | touched crate/file/test ownership table committed |
| M086_PRE.5 baseline commands | `cargo fmt --check`, `cargo check --workspace`, relevant pyxlog/runtime/induce/cuda tests recorded |
| M086_PRE.6 consumer inventory | DTS-DLM, Mistaber, v0.9.0, and pyxlog fixtures listed with paths or explicit missing-fixture blockers |
| M086_PRE.7 reuse map | every sub-goal names existing subsystems to extend and prohibited duplicate paths |

**Definition of Done.**

- Evidence under `docs/evidence/<date>-v086-pre/README.md`.
- No implementation changes before G086_PRE evidence is committed.
- Any missing consumer fixture is a BLOCKED metric, not silently skipped.
- Any sub-goal without an explicit reuse map is BLOCKED before implementation.

**Required reuse.** Reuse the existing repository status, roadmap, architecture
docs, evidence layout, test harnesses, and code ownership boundaries for the
baseline inventory. Do not create a separate tracking system, external project
board, or duplicate evidence format for v0.8.6.

### G086_DELTA_COALESCE - Device-Resident Batch Relation Delta Coalescing

**Goal.** Implement batch update coalescing for session-managed relations for
the purpose of reducing repeated DTS-DLM Stage-4 update overhead, with respect
to exact output equivalence and zero data-plane host transfers.

**Questions.**

- Q086_DELTA.1: Can repeated insert/delete updates be coalesced before recompute?
- Q086_DELTA.2: Does coalesced execution equal sequential delta execution and
  full replacement?
- Q086_DELTA.3: Does insert/delete cancellation happen without downloading
  relation rows?
- Q086_DELTA.4: Does the DTS `wmir_committed` pattern show fewer recompute or
  upload operations?

**Metrics.**

| Metric | Target |
|---|---|
| M086_DELTA.1 API | pyxlog exposes `apply_relation_delta_batch` or approved equivalent |
| M086_DELTA.2 equivalence | coalesced, sequential, and full replacement outputs are byte-identical |
| M086_DELTA.3 cancellation | insert/delete cancellation validated on device-resident rows |
| M086_DELTA.4 transfer budget | `dtoh_bytes=0`, `dtoh_calls=0` for coalescing hot path, excluding final requested output |
| M086_DELTA.5 performance | repeated-update fixture reduces upload/recompute work by at least 1.5x or removes a measured correctness blocker |
| M086_DELTA.6 telemetry | delta stats report coalesced inserts, coalesced deletes, canceled rows, affected SCCs, recomputed SCCs |

**Expected targets.**

- `crates/pyxlog/src/logic.rs`
- `crates/xlog-runtime/src/executor/rewrite.rs`
- `crates/xlog-gpu/src/logic.rs`
- `crates/xlog-cuda/src/provider/*` if device row cancellation needs kernels
- `python/tests/test_v086_delta_coalescing.py`
- `crates/xlog-runtime/tests/*delta*` or equivalent

**Required reuse.** Extend the existing `RelationDelta`,
`apply_deltas_and_recompute`, session relation store, and CUDA provider
buffer/set-operation machinery. Do not add a second Python-side delta engine or
a host-row coalescer.

**Definition of Done.**

- Source tests prove the public API and docs exist.
- Runtime tests prove semantic equivalence and cancellation.
- CUDA/DTOH guards prove no data-plane host transfer.
- DTS-shaped fixture uses `wmir_committed` and records raw row/update counts.

### G086_NOTIFY - Opt-In Relation-Change Callbacks

**Goal.** Add session relation-change callbacks for the purpose of letting
pyxlog consumers observe committed relation mutations without polling or
forcing relation downloads.

**Questions.**

- Q086_NOTIFY.1: Can Python users register callbacks per relation or session?
- Q086_NOTIFY.2: Are callbacks invoked only after a delta is committed?
- Q086_NOTIFY.3: Do callback payloads contain metadata summaries, not relation
  row data?
- Q086_NOTIFY.4: Is callback ordering deterministic under sequential session
  updates?

**Metrics.**

| Metric | Target |
|---|---|
| M086_NOTIFY.1 API | opt-in registration and unregistration APIs documented and stubbed |
| M086_NOTIFY.2 commit semantics | callbacks fire after successful commit and do not fire on failed/rolled-back delta |
| M086_NOTIFY.3 payload contract | payload includes relation name, generation, insert/delete/coalesced counts, affected SCCs, and telemetry |
| M086_NOTIFY.4 transfer budget | callback path performs zero relation data-plane D2H transfers |
| M086_NOTIFY.5 determinism | ordered fixture produces identical callback sequence across 100 replays |
| M086_NOTIFY.6 overhead | callback-disabled overhead within 2 percent of baseline on certified fixture |

**Expected targets.**

- `crates/pyxlog/src/logic.rs`
- `crates/pyxlog/python/pyxlog/_native.pyi`
- `docs/architecture/python-bindings.md`
- `python/tests/test_v086_relation_callbacks.py`

**Required reuse.** Reuse existing session mutation and `LogicDeltaStats`
commit points. Do not add polling loops, relation export hooks, or a callback
path that reads relation rows from device memory.

**Definition of Done.**

- Python tests cover success, failure, unregister, and ordering.
- Callback payload never contains raw relation columns.
- Docs state callback threading and GIL behavior explicitly.

### G086_EXACT_TYPES - Native Exact-Induction U32/Symbol Dispatch

**Goal.** Extend native exact induction beyond `U64` for the purpose of
supporting downstream tensorized ILP schemas that use `U32` and `Symbol`,
with respect to strict per-topology semantics, typed schema preservation, and
zero hidden host transfers.

**Questions.**

- Q086_EXACT_TYPES.1: Can `U32` pair buffers score natively without widening on
  the host?
- Q086_EXACT_TYPES.2: Can `Symbol` pair buffers reuse the correct physical
  layout while preserving logical type identity?
- Q086_EXACT_TYPES.3: Does every type match Python strict-per-topology parity?
- Q086_EXACT_TYPES.4: Are D2H counts constant and bounded independent of
  candidate/query count?

**Metrics.**

| Metric | Target |
|---|---|
| M086_EXACT_TYPES.1 U32 dispatch | native `U32` exact-induction fixture passes parity |
| M086_EXACT_TYPES.2 Symbol dispatch | native `Symbol` exact-induction fixture passes parity and preserves schema names/types |
| M086_EXACT_TYPES.3 U64 non-regression | existing `U64` tests remain green |
| M086_EXACT_TYPES.4 transfer budget | count-array D2H remains bounded exactly as documented; no type-conversion D2H |
| M086_EXACT_TYPES.5 typed diagnostics | mixed/unsupported types fail with explicit typed errors |
| M086_EXACT_TYPES.6 consumer fixture | DTS-DLM or Mistaber typed ILP fixture exercises at least one non-`U64` path |

**Expected targets.**

- `crates/xlog-induce/src/lib.rs`
- `crates/xlog-cuda/src/provider/ilp_exact.rs`
- `kernels/ilp_exact.cu`
- `crates/pyxlog/src/ilp_exact.rs`
- `python/tests/test_ilp_exact_induce.py`
- `docs/architecture/bounded-exact-induction.md`

**Required reuse.** Extend the existing `induce_exact` request validation,
`CudaKernelProvider::ilp_exact_score`, `kernels/ilp_exact.cu`, and pyxlog
`induce_exact_native` bridge. Do not introduce a separate exact-induction
engine for `U32` or `Symbol`.

**Definition of Done.**

- `U64`, `U32`, and `Symbol` all pass parity and determinism tests.
- No implementation narrows `Symbol` silently in public schema or diagnostics.
- Runtime evidence records D2H counts for small and large requests.

### G086_CHAIN_SMEM - Chain-Topology Shared-Memory Exact Scorer

**Goal.** Add shared-memory caching of L rows for chain-topology exact
induction for the purpose of reducing global-memory pressure only when profile
evidence identifies the chain scorer as hot.

**Questions.**

- Q086_CHAIN.1: Do current DTS-DLM or Mistaber profiles justify optimizing
  chain topology?
- Q086_CHAIN.2: Does shared-memory caching preserve exact strict-per-topology
  semantics?
- Q086_CHAIN.3: Does the optimized kernel outperform the baseline on the hot
  profile without hurting small workloads?

**Metrics.**

| Metric | Target |
|---|---|
| M086_CHAIN.1 profile trigger | profile evidence names chain exact scorer as hot, or certified synthetic fixture documents why it is needed |
| M086_CHAIN.2 parity | shared-memory and baseline kernels produce identical coverage arrays |
| M086_CHAIN.3 speedup | median kernel/runtime speedup >= 1.2x on hot fixture |
| M086_CHAIN.4 small-case guard | no >5 percent regression on small certified fixtures |
| M086_CHAIN.5 transfer budget | zero added data-plane D2H/H2D transfers |
| M086_CHAIN.6 fallback | non-chain topologies remain on existing path unless separately justified |

**Expected targets.**

- `kernels/ilp_exact.cu`
- `crates/xlog-cuda/src/provider/ilp_exact.rs`
- `crates/xlog-induce/src/lib.rs`
- CUDA tests and benchmark/evidence scripts

**Required reuse.** Extend the existing exact-induction chain topology kernel
and provider launcher. Do not create a chain-only scoring engine outside
`xlog-induce` / `xlog-cuda`.

**Definition of Done.**

- Profile trigger is committed before kernel optimization.
- Bench evidence includes raw timings, fixture sizes, speedup ratio, and
  regression table.
- Kernel can be disabled for A/B validation.

### G086_CSE - GPU-Native Common Subexpression Elimination

**Goal.** Add common subexpression elimination for duplicated subplans for the
purpose of eliminating repeated GPU work in DTS-DLM, Mistaber, and
certification workloads, with respect to relational equivalence and production
runtime dispatch.

**Questions.**

- Q086_CSE.1: Can the optimizer identify equivalent deterministic subplans?
- Q086_CSE.2: Can it avoid unsafe sharing across negation, aggregates,
  probability, or mutable relation generations?
- Q086_CSE.3: Does CSE reduce kernel launches or materialization work on
  consumer fixtures?
- Q086_CSE.4: Are CSE intermediates device-resident and versioned?

**Metrics.**

| Metric | Target |
|---|---|
| M086_CSE.1 equivalence key | structural equivalence key covers relation generation, projection, selection, joins, aggregates, negation, and provenance boundaries |
| M086_CSE.2 correctness | CSE and non-CSE outputs are byte-identical on deterministic/probabilistic fixtures |
| M086_CSE.3 safety rejection | unsafe cross-stratum/cross-generation sharing is rejected with diagnostics |
| M086_CSE.4 performance | duplicated-subplan fixture reduces duplicate kernel launches or materialization work by >=30 percent |
| M086_CSE.5 transfer budget | zero data-plane D2H/H2D added by CSE |
| M086_CSE.6 consumer evidence | DTS-DLM or Mistaber fixture exercises a real duplicated-subplan shape |

**Expected targets.**

- `crates/xlog-logic/src/compile.rs`
- `crates/xlog-ir/*`
- `crates/xlog-runtime/src/executor/*`
- `crates/xlog-prob/*` if probabilistic CSE needs provenance boundaries
- integration tests under `crates/xlog-integration/tests/`

**Required reuse.** Extend existing optimizer/RIR/PIR/runtime plan structures.
Do not build a parallel query planner or external memoizing evaluator.

**Definition of Done.**

- CSE is off/on comparable through an env or config flag.
- Correctness is asserted with output parity and deterministic replay.
- Evidence names every rejected unsafe CSE class.

### G086_ADAPT - Adaptive Runtime Re-Optimization

**Goal.** Add adaptive query re-optimization for the purpose of correcting
stable mis-planning observed through runtime telemetry, with respect to
deterministic replay, rollback safety, and consumer workloads.

**Questions.**

- Q086_ADAPT.1: Which telemetry proves a plan is mis-planned?
- Q086_ADAPT.2: Can xlog switch to a better plan without changing answers?
- Q086_ADAPT.3: Can rollback restore the original plan on failed/adverse
  adaptation?
- Q086_ADAPT.4: Are adaptation decisions deterministic under fixed inputs?

**Metrics.**

| Metric | Target |
|---|---|
| M086_ADAPT.1 telemetry | runtime records actual cardinality/selectivity/heat deltas needed for a deterministic decision |
| M086_ADAPT.2 decision stability | same fixture and seed produce identical adaptation decisions across 100 replays |
| M086_ADAPT.3 correctness | adapted and non-adapted outputs are byte-identical |
| M086_ADAPT.4 rollback | forced bad adaptation rolls back and records a typed diagnostic |
| M086_ADAPT.5 performance | mis-planned fixture improves >=1.2x median runtime or removes a documented correctness blocker |
| M086_ADAPT.6 transfer budget | adaptation uses metadata/control-plane counters only; no relation data-plane downloads |

**Expected targets.**

- `crates/xlog-runtime/src/executor/*`
- `crates/xlog-ir/*`
- `crates/xlog-logic/src/optimizer/*`
- `crates/xlog-core/src/config.rs`
- integration and replay tests

**Required reuse.** Reuse existing runtime telemetry, `StatsSnapshot`,
optimizer decision structures, and executor dispatch controls. Do not add a
separate adaptive execution loop that bypasses `Executor`.

**Definition of Done.**

- Adaptation has explicit enable/disable controls.
- Evidence includes no-adapt/adapt/rollback comparisons.
- Determinism matrix passes under fixed fixture replay.

### G086_INDEX - Persistent Hash Index Manager

**Goal.** Add a persistent hash index manager for the purpose of reusing
GPU-resident access paths across repeated session evaluations, with respect to
relation generations, memory budgets, and invalidation correctness.

**Questions.**

- Q086_INDEX.1: Can repeated session evaluations reuse hash indexes safely?
- Q086_INDEX.2: Are indexes invalidated on relation mutation, schema change,
  or memory pressure?
- Q086_INDEX.3: Can background builds run without blocking unrelated work or
  introducing stream-safety hazards?
- Q086_INDEX.4: Does reuse reduce measured index build cost on consumer
  fixtures?

**Metrics.**

| Metric | Target |
|---|---|
| M086_INDEX.1 manager API | runtime/provider index manager stores indexes by relation id, schema, key, generation, and device |
| M086_INDEX.2 invalidation | mutation/schema/generation tests prove stale indexes are never reused |
| M086_INDEX.3 budget | memory budget and LRU/heat policy bound retained indexes |
| M086_INDEX.4 background build | background build records stream dependencies and passes recorder/strict mode where applicable |
| M086_INDEX.5 performance | repeated evaluation fixture improves >=1.5x for index-build-heavy workload or records a blocker |
| M086_INDEX.6 transfer budget | index build/reuse uses zero data-plane D2H/H2D transfers |

**Expected targets.**

- `crates/xlog-runtime/src/executor/*`
- `crates/xlog-cuda/src/provider/*`
- `crates/xlog-core/src/config.rs`
- `docs/architecture/*index*` or runtime architecture docs
- integration and CUDA provider tests

**Required reuse.** Extend the existing join index cache, relation generation
metadata, memory budget, and CUDA provider allocation/recorder machinery. Do
not add an independent index cache with separate lifetime semantics.

**Definition of Done.**

- Index reuse is observable through stable telemetry.
- Stale-index tests fail on the old path and pass after implementation.
- Memory-budget pressure evicts indexes deterministically.

### G086_CONSUMERS - Consumer Certification

**Goal.** Certify the composed v0.8.6 feature set for the purpose of proving
that DTS-DLM, Mistaber, v0.9.0, and pyxlog users can consume the release
without private hooks or fixture-only paths.

**Questions.**

- Q086_CONSUMERS.1: Do DTS-DLM Stage-4/M37-shaped fixtures exercise delta,
  exact, and optimizer improvements?
- Q086_CONSUMERS.2: Do Mistaber `.xlog` fixtures use scientific/engineering
  naming and production runtime paths?
- Q086_CONSUMERS.3: Do v0.9.0 substrate probes have the runtime primitives they
  need for fully GPU-native epistemic/probabilistic execution?
- Q086_CONSUMERS.4: Do public pyxlog examples remain backward compatible?

**Metrics.**

| Metric | Target |
|---|---|
| M086_CONSUMERS.1 DTS-DLM | DTS-shaped fixture passes and records raw speed/transfer numbers |
| M086_CONSUMERS.2 Mistaber | at least two Mistaber-derived `.xlog` fixtures pass without project-specific terminology |
| M086_CONSUMERS.3 v0.9.0 substrate | exact/index/CSE/adaptive primitives documented for epistemic/solver branch |
| M086_CONSUMERS.4 pyxlog compatibility | v0.8.0/v0.8.5 public examples and source guards remain green |
| M086_CONSUMERS.5 production path | source guards prove no fixture-only bypass of runtime/provider dispatch |
| M086_CONSUMERS.6 reuse audit | source scan and code review show no duplicate engine/helper path for existing subsystems |

**Expected targets.**

- `examples/v086-runtime/`
- `scripts/validate_v086_examples.py`
- `python/tests/test_v086_*`
- `docs/evidence/<date>-v086-consumers/`

**Required reuse.** Reuse existing v0.8.0/v0.8.5 validators, pyxlog examples,
runtime/provider dispatch, and consumer fixture conventions. Mistaber-derived
examples must be rewritten as `.xlog` programs over existing language/runtime
features, not bridged through Mistaber code or a separate adapter.

**Definition of Done.**

- All consumer examples are executable and validator-owned.
- Evidence records command lines, exit codes, raw numbers, and feature coverage.
- Mistaber examples use neutral scientific/engineering vocabulary.

### G086_INT - Integration, Regression, And Certification

**Goal.** Validate the composed v0.8.6 branch for the purpose of preparing
release integration, with respect to correctness, performance, GPU-native
guardrails, docs, examples, and git hygiene.

**Questions.**

- Q086_INT.1: Do all feature-node tests pass together after integration?
- Q086_INT.2: Do v0.8.0 and v0.8.5 compatibility gates remain green?
- Q086_INT.3: Do transfer guards cover every accepted v0.8.6 hot path?
- Q086_INT.4: Are performance gates backed by raw measurements rather than
  prose claims?
- Q086_INT.5: Are roadmap, changelog, architecture docs, evidence links, and
  git hygiene consistent with the implemented scope?

**Metrics.**

| Metric | Target |
|---|---|
| M086_INT.1 formatting | `cargo fmt --check` exit 0 |
| M086_INT.2 workspace | `cargo check --workspace` exit 0 |
| M086_INT.3 targeted Rust | runtime, cuda, induce, prob, logic, integration tests relevant to touched surfaces pass |
| M086_INT.4 Python | pyxlog v0.8.0/v0.8.5/v0.8.6 source/runtime tests pass |
| M086_INT.5 examples | v0.8.0, v0.8.5, and v0.8.6 validators pass |
| M086_INT.6 transfer guards | no-dtoh/no-hidden-transfer guards pass for every accepted hot path |
| M086_INT.7 performance | all stated speedup/overhead gates recorded or explicitly BLOCKED |
| M086_INT.8 docs | roadmap, changelog, architecture docs, and evidence links validate |
| M086_INT.9 git hygiene | no unrelated files; no generated bulk artifacts unless documented |

**Definition of Done.**

- All metrics are PASS or BLOCKED with named blocker.
- A red performance gate cannot be converted to PASS by explanation alone.
- Any pre-existing flake exception must be enumerated by file and evidence, not
  by glob.

**Required reuse.** Reuse existing workspace, crate, Python, example,
DTOH/no-hidden-transfer, JSON, roadmap, and git hygiene validation commands.
Do not replace the existing certification matrix with a new script unless that
script invokes and reports the existing gates explicitly.

### G086_CLOSE - Evidence, Roadmap Sync, And Closure Proposal

**Goal.** Produce a closure proposal for the purpose of letting the coordinator
decide whether v0.8.6 can merge, with respect to the full GQM tree and
consumer evidence.

**Questions.**

- Q086_CLOSE.1: Does the proposal list every G086 node with commit SHA and
  metric status?
- Q086_CLOSE.2: Does the roadmap reflect the actual implemented or blocked
  v0.8.6 state?
- Q086_CLOSE.3: Are all unresolved issues explicitly interpreted as PASS, FAIL,
  BLOCKED, WAIVED, or NOT_APPLICABLE?
- Q086_CLOSE.4: Does the closure package preserve the proposal/approval/merge
  gate split?
- Q086_CLOSE.5: Does the methodology audit prove every sub-goal used GDSP/GQM
  evidence rather than post-hoc narration?

**Metrics.**

| Metric | Target |
|---|---|
| M086_CLOSE.1 sub-goal table | every G086 node listed with commit SHA and metric status |
| M086_CLOSE.2 roadmap sync | v0.8.6 section reflects actual PASS/BLOCKED states |
| M086_CLOSE.3 unresolved issues | all red/yellow metrics have explicit disposition |
| M086_CLOSE.4 release decision | recommendation is `MERGE_READY`, `HOLD_FOR_FIXES`, or `SCOPE_AMENDMENT_REQUIRED` |
| M086_CLOSE.5 no implicit release | no push, tag, board update, or merge without coordinator authorization |
| M086_CLOSE.6 methodology audit | every sub-goal evidence file includes GDSP consumer goal, reused subsystem, GQM questions, raw measurements, and metric interpretation |

**Definition of Done.**

- Closure proposal exists under `docs/plans/`.
- Machine-readable closure summary exists under `docs/evidence/<date>-v086-close/`.
- `git status --short --branch` is clean after final commit.

**Required reuse.** Reuse the established xlog closure proposal, evidence
README, closure summary JSON, roadmap sync, changelog sync, and no-implicit-
release gate patterns from v0.8.0/v0.8.5. Do not invent a separate closure
format or collapse proposal, merge, push, and tag authorization into one step.

## 7. KPIs

| KPI | Gate |
|---|---|
| KPI086.1 Deferred-scope closure | 7/7 v0.8.0 deferred runtime/optimizer items have PASS/BLOCKED metric disposition |
| KPI086.2 GPU-native execution | accepted hot paths report `dtoh_bytes=0`, `dtoh_calls=0`, `htod_bytes=0`, `htod_calls=0` except named control-plane/final-result exceptions |
| KPI086.3 Consumer coverage | DTS-DLM, Mistaber, v0.9.0 substrate, and pyxlog examples each have evidence |
| KPI086.4 Determinism | fixed certified replay matrices produce identical outputs and adaptation decisions |
| KPI086.5 Performance | each feature with a performance claim meets its speedup/overhead target with raw numbers |
| KPI086.6 Public compatibility | v0.8.0 and v0.8.5 public API/example guards remain green |
| KPI086.7 Evidence quality | every sub-goal writes evidence with commands, exit codes, raw values, and commit SHA |

## 8. Worktree Setup

Coordinator command:

```bash
cd /home/dev/projects/xlog
git worktree add .worktrees/v086-runtime-completion -b feat/v086-runtime-completion main
cd .worktrees/v086-runtime-completion
git status --short --branch
```

Agent first actions:

1. Record `git rev-parse HEAD` and `git status --short --branch`.
2. Read this document, `ROADMAP.md`, `docs/architecture/python-bindings.md`,
   `docs/architecture/bounded-exact-induction.md`, v0.8.0/v0.8.5 closure
   proposals, and consumer docs for DTS-DLM and Mistaber.
3. Run G086_PRE baseline commands.
4. Write `docs/evidence/<date>-v086-pre/README.md`.
5. Commit G086_PRE before implementation.

## 9. Reporting Protocol

After each sub-goal, report:

- branch and commit SHA;
- files changed;
- metrics table with PASS/FAIL/BLOCKED;
- exact validation commands and raw numbers;
- DTOH/H2D transfer budget results;
- consumer evidence paths;
- next sub-goal or halt condition.

Do not compress multiple sub-goals into a single "done" report unless all
metric evidence is present.

## 10. Halt Conditions

Halt and ask the coordinator if:

- a feature requires a CPU-only accepted hot path;
- a feature would weaken v0.8.0/v0.8.5 compatibility;
- a feature requires changing v0.9.0 epistemic semantics rather than substrate;
- a GPU-native implementation requires a new public API not covered here;
- a consumer fixture cannot be found or constructed without changing external
  repo scope;
- a metric needs to be weakened, waived, or redefined;
- any action would require push, tag, merge, or release-board mutation.

## 11. Final Deliverable

The final output is a v0.8.6 closure proposal with:

- complete GQM metric table;
- all sub-goal commit SHAs;
- evidence links and raw performance/transfer numbers;
- consumer certification summary for DTS-DLM, Mistaber, v0.9.0 substrate, and
  pyxlog users;
- explicit PASS/BLOCKED disposition for all seven deferred v0.8.0 items;
- remaining risk summary;
- merge recommendation;
- explicit statement that v0.9.0 must rebase or merge after v0.8.6 lands.
