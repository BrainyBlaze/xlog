# v0.8.6 G086_PRE Evidence

Date: 2026-05-19
Goal node: G086_PRE - Baseline Inventory And Worktree Health
Branch: `feat/v086-runtime-completion`
Worktree: `.worktrees/v086-runtime-completion`
Goal document: `docs/plans/2026-05-19-agent-v086-dts-runtime-completion-goal.md`

## GDSP / GQM Trace

GDSP consumer goal: establish the v0.8.6 runtime/optimizer completion branch
before implementation so DTS-DLM, Mistaber, v0.9.0, and pyxlog requirements
are mapped to existing xlog subsystems rather than parallel engines.

Existing xlog subsystem reused: repository status, ROADMAP milestone, existing
architecture docs, v0.8.0/v0.8.5 closure proposal format, evidence directory
layout, cargo and pytest gates, pyxlog session APIs, runtime delta machinery,
bounded exact induction, optimizer/runtime dispatch, CUDA provider, and current
example validators.

GQM questions answered:

- Q086_PRE.1: branch cut from local post-v0.8.5 main at `edff1ba4`.
- Q086_PRE.2: seven v0.8.6 roadmap items are present and open.
- Q086_PRE.3: crate/file/test ownership is mapped below.
- Q086_PRE.4: authoritative profiles and consumer fixtures are inventoried
  below, with missing Mistaber `.xlog` fixtures called out explicitly.

## Baseline Status

| Item | Value |
|---|---|
| `git rev-parse HEAD` | `edff1ba4112f8f303f45b2e0f1f2b0ddd3a5f2a0` |
| `git merge-base HEAD main` | `edff1ba4112f8f303f45b2e0f1f2b0ddd3a5f2a0` |
| Branch status before evidence edit | `## feat/v086-runtime-completion` |
| Local main note | local `main` is ahead of `origin/main` by the v0.8.0 goal-doc commit `9914f9c5` and the v0.8.6 milestone commit `edff1ba4` |
| Worktree directory | `/home/dev/projects/xlog/.worktrees/v086-runtime-completion` |

Interpretation: PASS for M086_PRE.1 and M086_PRE.2. The v0.8.6 branch is cut
from the current local `main` state that contains the governing v0.8.6 goal
document and is clean before implementation. No runtime, CUDA, optimizer, or
pyxlog implementation files were changed before this evidence file.

## Baseline Facts

| Fact | Evidence | Interpretation |
|---|---|---|
| ROADMAP contains planned v0.8.6 milestone with seven open items | `ROADMAP.md` has section `v0.8.6 - DTS-DLM Runtime Completion and GPU-Native Optimizer Pack` with seven unchecked bullets | PASS |
| v0.8.0 goal doc is committed on main lineage | commit `9914f9c5 docs(v080): commit DTS ML Python goal doc`; file `docs/plans/2026-05-18-agent-v080-dts-ml-python-goal.md` present | PASS |
| pyxlog has single-delta APIs but no batch coalescing or callbacks | `crates/pyxlog/src/logic.rs` exposes `insert_relation`, `delete_relation`, `apply_relation_delta`, and `delta_stats`; searches found no public batch coalescing or relation-change callback API | PASS |
| bounded exact induction validates U64 only | `crates/xlog-induce/src/lib.rs` rejects non-`ScalarType::U64`; `crates/xlog-cuda/src/provider/ilp_exact.rs` also expects U64 columns | PASS |
| architecture doc defers U32, Symbol, and chain shared-memory caching | `docs/architecture/bounded-exact-induction.md` lists `U32` and `Symbol` as deferred and names shared-memory caching of chain L rows under non-goals/deferred | PASS |
| optimizer/runtime lacks accepted v0.8.6 CSE/adaptive/persistent-index implementation | searches found only roadmap/architecture backlog references for CSE, adaptive re-optimization, and persistent hash index manager; no v0.8.6 implementation or tests exist | PASS |

## Backlog Map

| ROADMAP item | G086 node | Current state |
|---|---|---|
| Device-resident batch update coalescing for repeated `wmir_committed` updates | G086_DELTA_COALESCE | OPEN |
| Opt-in change notification callbacks for session-managed relations | G086_NOTIFY | OPEN |
| Native exact-induction `U32` and `Symbol` dispatch | G086_EXACT_TYPES | OPEN |
| Chain-topology shared-memory caching of L rows | G086_CHAIN_SMEM | OPEN |
| GPU-native common subexpression elimination | G086_CSE | OPEN |
| Adaptive query re-optimization during execution | G086_ADAPT | OPEN |
| Persistent hash index manager with background GPU-resident build and reuse | G086_INDEX | OPEN |

Interpretation: PASS for M086_PRE.3. All seven deferred items map to exactly
one G086 node.

## Ownership Map

| G086 node | Primary crates/files | Tests/evidence owners |
|---|---|---|
| G086_DELTA_COALESCE | `crates/pyxlog/src/logic.rs`, `crates/xlog-runtime/src/executor/rewrite.rs`, `crates/xlog-gpu/src/logic.rs`, `crates/xlog-cuda/src/provider/*` if device cancellation needs kernels | `python/tests/test_v086_delta_coalescing.py`, `crates/xlog-runtime/tests/*delta*`, DTOH guards |
| G086_NOTIFY | `crates/pyxlog/src/logic.rs`, `crates/pyxlog/python/pyxlog/_native.pyi`, `docs/architecture/python-bindings.md` | `python/tests/test_v086_relation_callbacks.py` |
| G086_EXACT_TYPES | `crates/xlog-induce/src/lib.rs`, `crates/xlog-cuda/src/provider/ilp_exact.rs`, `kernels/ilp_exact.cu`, `crates/pyxlog/src/ilp_exact.rs`, `docs/architecture/bounded-exact-induction.md` | `python/tests/test_ilp_exact_induce.py`, `cargo test -p xlog-induce --lib`, `cargo test -p xlog-cuda --lib ilp_exact` |
| G086_CHAIN_SMEM | `kernels/ilp_exact.cu`, `crates/xlog-cuda/src/provider/ilp_exact.rs`, `crates/xlog-induce/src/lib.rs` | CUDA launcher tests plus benchmark/evidence script to be added after profile trigger |
| G086_CSE | `crates/xlog-logic/src/compile.rs`, `crates/xlog-ir/*`, `crates/xlog-runtime/src/executor/*`, `crates/xlog-prob/*` if provenance boundaries are needed | `crates/xlog-integration/tests/*`, duplicated-subplan fixture and source guards |
| G086_ADAPT | `crates/xlog-runtime/src/executor/*`, `crates/xlog-ir/*`, `crates/xlog-logic/src/optimizer/*`, `crates/xlog-core/src/config.rs` | integration replay tests and rollback diagnostics |
| G086_INDEX | `crates/xlog-runtime/src/executor/*`, `crates/xlog-cuda/src/provider/*`, `crates/xlog-core/src/config.rs`, runtime/index architecture docs | integration and CUDA provider tests for generation invalidation, budget, and reuse |

Interpretation: PASS for M086_PRE.4. Ownership is mapped to existing
subsystems; no duplicate engine path is authorized.

## Consumer Inventory

| Consumer | Current paths found | Fixture status |
|---|---|---|
| DTS-DLM | `/home/dev/projects/dts-dlm/scripts/pyxlog_070_stage4_smoke.py`, `/home/dev/projects/dts-dlm/src/dts_dlm/propagate/xlog_executor.py`, `/home/dev/projects/dts-dlm/src/dts_dlm/integrations/pyxlog/tensorized_ilp.py`, `/home/dev/projects/dts-dlm/src/tests/propagate/test_zero_host_propagate.py`, `/home/dev/projects/dts-dlm/src/tests/integrations/test_tensorized_ilp.py`, `/home/dev/projects/dts-dlm/docs/plans/2026-04-17-m8-phase1-xlog-induce-engine.md`, `/home/dev/projects/dts-dlm/docs/plans/2026-05-19-m37a-plus-b-plan-freeze.md` | AVAILABLE |
| Mistaber | `/home/dev/projects/mistaber/tests/engine/*`, `/home/dev/projects/mistaber/tests/integration/*`, `/home/dev/projects/mistaber/docs/plugin/*`, `/home/dev/projects/mistaber/mistaber/ontology/*` | BLOCKED for v0.8.6 consumer examples: no committed `.xlog` files found under `/home/dev/projects/mistaber` |
| v0.9.0 epistemic/solver | `.worktrees/v090-epistemic/docs/plans/2026-05-18-agent-v090-epistemic-solver-goal.md`, `.worktrees/v090-epistemic/python/tests/test_ilp_exact_induce.py`, `.worktrees/v090-epistemic/python/tests/test_logic_dts_frozen_replay_determinism.py` | AVAILABLE for substrate requirements; v0.9.0 must rebase or merge after v0.8.6 |
| General pyxlog users | `examples/v080-dts/`, `examples/v085-language/`, `scripts/validate_v080_examples.py`, `scripts/validate_v085_examples.py`, `docs/architecture/python-bindings.md`, `python/tests/test_v080_*_source.py`, `python/tests/test_v085_examples_source.py` | AVAILABLE |

Interpretation: BLOCKED for M086_PRE.6 because the Mistaber-derived `.xlog`
fixture is not present in the external repo inventory. G086_CONSUMERS must
construct neutral scientific/engineering `.xlog` examples or obtain an
explicit external fixture path before it can pass. The blocker is scoped to the
consumer-certification node; it does not require changing external repo scope
before G086_DELTA_COALESCE can start.

## Reuse Map

| G086 node | Existing subsystem to extend | Prohibited duplicate path |
|---|---|---|
| G086_DELTA_COALESCE | `RelationDelta`, `apply_deltas_and_recompute`, session relation store, CUDA set/buffer operations | Python-side delta engine or host-row coalescer |
| G086_NOTIFY | pyxlog session mutation commit points and `LogicDeltaStats` | polling loop, relation export hook, or callback payload containing raw rows |
| G086_EXACT_TYPES | `induce_exact`, `CudaKernelProvider::ilp_exact_score`, `kernels/ilp_exact.cu`, pyxlog native bridge | separate exact-induction engine |
| G086_CHAIN_SMEM | existing exact-induction chain topology launcher and kernel | chain-only scorer outside `xlog-induce` / `xlog-cuda` |
| G086_CSE | existing parser/RIR/PIR/optimizer/runtime plan structures | external memoizing evaluator or parallel planner |
| G086_ADAPT | runtime telemetry, stats snapshots, optimizer decisions, executor dispatch controls | adaptive execution loop bypassing `Executor` |
| G086_INDEX | join index cache concepts, relation generations, CUDA provider allocation and recorder machinery, memory budgets | independent index cache with separate lifetime semantics |

Interpretation: PASS for M086_PRE.7. Every sub-goal names an existing
subsystem to reuse.

## Baseline Commands

| Command | Result |
|---|---|
| `cargo fmt --check` | exit 0 |
| `cargo check --workspace` | exit 0; finished in 34.26s |
| `cargo test -p pyxlog --lib` | exit 0; 7 passed |
| `cargo test -p xlog-runtime` | exit 0; 125 lib tests passed, 15 integration tests passed across 4 test binaries, 2 doc tests passed, 2 doc tests ignored |
| `cargo test -p xlog-induce --lib` | exit 0; 23 passed |
| `cargo test -p xlog-cuda kernel_modules` | exit 0; 2 passed; 147 filtered out in lib; integration binaries filtered as expected |
| `cargo test -p xlog-cuda --lib ilp_exact` | exit 0; 3 passed |
| `pytest -q python/tests/test_v080_pyapi_source.py python/tests/test_v080_delta_source.py python/tests/test_v080_exact_source.py python/tests/test_v080_bridge_source.py python/tests/test_v080_examples_source.py python/tests/test_v085_examples_source.py` | exit 0; 20 passed |
| `python scripts/validate_package_metadata.py` | exit 0; README quickstart assumptions and workspace package metadata validated |

Interpretation: PASS for M086_PRE.5. These commands are a baseline only; they
do not prove the seven v0.8.6 feature nodes are implemented.

## Metric Disposition

| Metric | Status | Evidence |
|---|---|---|
| M086_PRE.1 branch base | PASS | `git merge-base HEAD main` equals `edff1ba4112f8f303f45b2e0f1f2b0ddd3a5f2a0` |
| M086_PRE.2 worktree status | PASS | branch was clean before this evidence edit |
| M086_PRE.3 backlog map | PASS | seven ROADMAP items mapped to G086 nodes above |
| M086_PRE.4 ownership map | PASS | ownership table above |
| M086_PRE.5 baseline commands | PASS | command table above |
| M086_PRE.6 consumer inventory | BLOCKED | DTS-DLM, v0.9.0, and pyxlog paths found; Mistaber `.xlog` fixture is missing and must be constructed or supplied before G086_CONSUMERS |
| M086_PRE.7 reuse map | PASS | reuse table above |

## Next-Step Decision

Proceed to G086_DELTA_COALESCE after this evidence-only PRE commit. Do not
start runtime implementation until the commit records this baseline. The
Mistaber `.xlog` fixture gap is not a blocker for G086_DELTA_COALESCE, but it
is a named blocker for G086_CONSUMERS unless resolved by then.
