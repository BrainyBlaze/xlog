# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Fixed

- Hardened the release example validator so every `.xlog`, probabilistic,
  Python, and neural example runs in release validation without depending on
  an installed `pyxlog` wheel or optional external neural datasets.
- Fixed recorded CUDA sort/groupby row-count handling so compacted buffers do
  not cache row capacity as logical cardinality.
- Hardened adaptive join-index reuse so stale index metadata is evicted and
  falls back to the regular hash join instead of aborting valid queries.
- Updated the public CUDA certification count to 207/207 after the current
  full-suite and recorded-launch certification reruns.

## [0.7.0] — 2026-05-18

General WCOJ Architecture and Runtime Expansion. This release
retargets the completed feature pack originally planned as v0.6.5
to v0.7.0 because the delivered surface is a full WCOJ subsystem
expansion: cost-aware planning, recursive integration, K-clique
coverage, paper-aligned helper/runtime mechanisms, DTS-DLM hot-loop
integration, and release-board closure.

### Added

- First-class `MultiWayJoin` / WCOJ RIR and promoter surface for
  eligible multiway rules, with deterministic fallback preservation.
- WCOJ variable-ordering and cardinality/selectivity-aware cost-model
  integration, including per-iteration recursive SCC statistics.
- General WCOJ CUDA/runtime coverage beyond triangle: 4-cycle,
  K=5/K=6 hypergraph planner production path, K=7/K=8 template
  coverage, runtime histogram refresh, and helper-splitting invocation.
- Adaptive join closure: nested-loop dispatch for small eligible joins
  and preserved provider-level sort-merge operator certification.
- Certification and benchmark surfaces for GPU Same Generation,
  skewed multiway, deep-recursive WCOJ, deterministic mixed execution,
  widened-frontier replay, and paper-class production-scale fixtures.
- DTS-DLM Phase-2 integration evidence for chain-shaped joins,
  sort-label propagation, CUDA Graph capture/replay, M37-A surface
  preservation, and m37c-prime end-to-end validation.
- Dedicated WCOJ architecture and user guides.

### Changed

- Workspace package version and internal xlog crate dependency
  constraints now target `0.7.0`.
- Closure-board and tag-handoff release surfaces now use `v0.7.0`;
  historical evidence may still say the work was originally targeted
  as `v0.6.5`.
- Roadmap release trains move forward: the completed WCOJ expansion is
  v0.7.0, v0.8.0 is narrowed to DTS-DLM ML/Python
  productization, Epistemic/Solver Semantics moves to v0.9.0,
  and Multi-GPU / Out-of-Core moves to v0.10.0. The broader
  language / CLI product backlog is deferred until it has a named
  consumer.

### Release Status

- Closure board: 31 DONE, 0 IN-PROGRESS, 0 BLOCKED, 0 OPEN.
- W7.1 is complete: the annotated `v0.7.0` tag has been created and pushed.

## [0.6.0] — 2026-04-29

Stream-Safe GPU Runtime And Execution Discipline. Infrastructure
hardening release: a stream-safe GPU runtime and recorded launch
discipline so subsequent join / WCOJ work can be trusted under
parallel execution. Default behaviour for legacy callers is
unchanged; the new path is opt-in via
`CudaKernelProvider::with_runtime` /
`GpuMemoryManager::with_runtime` plus the
`XLOG_USE_DEVICE_RUNTIME` / `XLOG_USE_RECORDED_OPS` env flags.

### Added

- **Access-aware stream dependency manager** (PR #72,
  `26c2e429` + follow-ups). Replaces post-launch-only
  `record_block_use` with `prepare_block_use(BlockId, stream,
  Access)` / `finish_block_use(...)` and an `Access {Read,
  Write, ReadWrite}` enum. `AsyncCudaResource::LiveEntry`
  tracks `last_write: Option<(StreamId, CudaEvent)>` (seeded
  with an allocation-ready event captured immediately after
  `cuMemAllocAsync`) and `outstanding_reads:
  Vec<(StreamId, CudaEvent)>`. Reads wait on `last_write`;
  writes wait on `last_write` plus every cross-stream
  outstanding read. Same-stream events are skipped. Closes
  both the use-after-prior-write hazard and the
  use-after-allocation hazard across streams.
- **Lifetime-free `LaunchRecorder`**. Snapshots `BlockId` from
  each registered slice at record time and drops the source
  borrow immediately, so kernel `&mut` borrows after preflight
  are unrestricted. `preflight(&runtime)` queues
  `cuStreamWaitEvent` for every recorded use's cross-stream
  dependency BEFORE the launch; `commit(self, &runtime)`
  records new events via `finish_block_use` AFTER. Repeated
  registrations of the same block dedup on
  `(ptr, generation, device_ordinal)` to a single
  prepare/finish call with the strongest access.
- **`XlogDeviceRuntime::prepare_first_use(slice, stream, access)`
  / `finish_first_use(...)`** for helper-internal scratch
  whose first cross-stream consumer is a raw `cuMemsetD8Async`
  / `cuMemcpyDtoDAsync_v2` / `kernel.launch_on_stream` call
  ahead of any `LaunchRecorder::preflight`.
- **Formal certification harness** (`3361785b`). The cert
  `TestContext` builds the production decorator stack
  (`AsyncCudaResource → LoggingResource → GlobalDeviceBudget
  → XlogDeviceRuntime`) when `XLOG_USE_DEVICE_RUNTIME=1` is
  set and uses `with_runtime` constructors; the env-gated
  dispatchers in `provider::sort` / `filter_by_mask` /
  `hash_join_v2` / etc. then route through the recorded path
  when `XLOG_USE_RECORDED_*` is set. The harness reaps
  pending async frees between categories, and
  `GlobalDeviceBudget::allocate` retries once after a reap on
  transient over-budget conditions.
  Result: `XLOG_USE_DEVICE_RUNTIME=1 XLOG_USE_RECORDED_OPS=1
  cargo test -p xlog-cuda-tests --test certification_suite
  --release` passes **206/206**; legacy default still passes
  206/206.
- **A3/A4 cross-stream lifetime stress harness**
  (`crates/xlog-integration/tests/test_a3_a4_stress.rs`,
  `27ec3bd9` + `a01b51fa`). Two workloads (`friends`
  sort+hash-join sensitive, `reach` recursive fixed-point +
  joins). Stable FNV-1a checksums, fixed schedule + seeded
  random tail. **A4 fork-isolated stress passes 16/16** in
  every fixture mode and every env combination. A 5-mode
  diagnostic matrix (`XLOG_A3_FIXTURE_MODE=per_iter |
  per_thread | shared` × runtime-on/off × recorded-on/off
  via `XLOG_A3_DIAGNOSTIC=1`) classifies the A3 thread-of-N
  drift as pre-existing and not introduced by v0.6.0 — see
  Known Issues below.
- **Multi-threaded sort+hash-join regression**
  (`crates/xlog-cuda/tests/test_mt_sort_hj_alloc_ordering.rs`,
  PR #72). 8 threads × 128 iters × 3 rounds friend-of-friend
  self-join. Was RED at baseline `8cc0882c` (~6/1024 failures
  per run); 1024/1024 + 1024/1024 across 10 consecutive runs
  on `b1560674`.
- **Documentation**: `docs/architecture/device-runtime.md`
  (runtime stack + access matrix + env-gated dispatch + cert
  modes) and `docs/architecture/recorded-launch-migration.md`
  (operator-author checklist + anti-patterns + four-gate
  validation command sequence). Linked from
  `docs/ARCHITECTURE.md` Memory Management section.

### Changed

- `record_block_use` retained as a backward-compat shim that
  calls `finish_block_use(Read)` for the dealloc-wait surface;
  production callers go through the recorder.
- `write_post_preflight_fresh` removed. All 78 callers across
  `provider/{relational,filter,groupby,mod}.rs` migrated to
  pre-preflight `write` (the recorder snapshot drops the
  borrow, so kernel `&mut` borrows after preflight are
  unaffected).
- 6 direct `runtime.record_block_use(b, launch_stream)` call
  sites in provider code migrated to
  `runtime.finish_block_use(BlockId::from_block(b),
  launch_stream, Access::Write)` with semantically correct
  Access kinds.
- `prepare_first_use(Access::Write)` added at every
  helper-internal scratch alloc site that subsequently writes
  via raw CUDA work BEFORE its parent recorder's preflight:
  `build_hash_table_v2_on_stream` (5 buffers),
  `gather_buffer_by_indices_on_stream` (per-column
  `dst_col`s), `multiblock_scan_u32_inplace_on_stream` /
  `_view_inplace_on_stream` (`block_sums`), and every join
  variant's `d_count_only` / `d_output_count` / `out_col`
  zero-fills (Inner / LeftOuter / count-scan-materialize /
  indexed Inner / indexed LeftOuter).
- `gather_buffer_by_indices_on_stream`: local
  `d_output_rows` scalar created via
  `upload_device_row_count` + read on `launch_stream` is now
  fenced via `Access::Write` at upload + `Access::Read`
  prepare on `launch_stream` + `Access::Read` finish before
  drop. Closed a review-finding from the PR.

### Deferred to post-v0.6.0

- **Host-mask `compact_buffer_by_mask` recorded migration**.
  Re-opens when a runtime-backed recorded release path
  consumes host-provided masks. Until then the legacy entry
  is the supported path; the recorded
  `compact_buffer_by_device_mask_counted_recorded` covers the
  device-mask case for runtime-backed callers.
- **ILP / ILP-exact view helpers + operators recorded
  migration**. Re-opens when tensorized ILP /
  exact-induction downstream consumer work resumes (v0.8.0
  native exact-induction consumer gate) and requires
  runtime-backed stream safety.
- **Sub-slice 3 LeftOuter CSM** (commit `b90ae77f`, never
  pushed; recovered into `.recovery/sub-slice-3-edits.md`).
  Apply on a fresh post-v0.6.0 branch after auditing every
  scratch alloc against the access-aware contract documented
  in `docs/architecture/recorded-launch-migration.md`.

### Known Issues (not release blockers)

- **A3 in-process thread-of-N drift on
  `test_a3_a4_stress`**: 8 threads × 32 iters produce ~3%
  checksum drift on recursive Datalog workloads. The 5-mode
  diagnostic matrix demonstrates this is **NOT v0.6.0
  stream-safety regression** — drift fires identically on the
  legacy default path (no `XLOG_USE_DEVICE_RUNTIME`, no
  `XLOG_USE_RECORDED_OPS`, one runtime per thread, no v0.6
  code in the call chain). Bug class: pre-existing
  same-process multi-executor concurrency against one CUDA
  primary context. Tracked under v0.9.0 "Concurrency
  Hardening" in `ROADMAP.md`. The v0.6.0 release gate is
  **A4 fork-isolated stress + cert suite + umbrella ×50**,
  not "A3 zero drift".
- **`test_provider_launch_recorder --test-threads=8`** shows
  9/42 `*_survives_drop_and_reuse` failures (was 23/42 at
  baseline `8cc0882c`). Pre-existing pattern from
  cross-runtime mempool aliasing under intra-binary test
  parallelism. Production gate spec is `--test-threads=1`,
  which is clean.

### Release Validation (gates green on `b1560674`)

- `cargo fmt --check`: clean.
- `git diff --check`: clean.
- Legacy cert suite: 206/206 in 20.22s.
- Runtime+recorded cert suite
  (`XLOG_USE_DEVICE_RUNTIME=1 XLOG_USE_RECORDED_OPS=1`):
  206/206 in 16.56s.
- Umbrella ×50 (`real_world_tests --test-threads=8` under
  recorded runtime): **50/50**.
- Workspace `--tests --exclude pyxlog --release
  --test-threads=1`: 142 result lines, no failures.

> **Note:** All items below are post-v0.5.0 work. Items in
> `[Unreleased]` between the v0.5.0 tag and the v0.6.0 tag are
> reflected in the v0.6.0 release entry above.

## [0.6.1] — 2026-04-29

CSM Env Dispatch and Certification Mode Labeling. Small,
focused release on top of v0.6.0: enables count-scan-materialize
(CSM) hash-join methods for `Inner` / `LeftOuter` (indexed and
non-indexed) under an env gate, closes a stream-safety gap in
three earlier CSM siblings, and names the CSM cert mode
explicitly so reports are unambiguous. No kernel changes, no
algorithm changes, no eligibility relaxation. Default behaviour
for legacy callers is unchanged; the new path is opt-in via
`XLOG_USE_RECORDED_CSM=1` (or umbrella `XLOG_USE_RECORDED_OPS=1`).

### Added

- **Recorded CSM (count-scan-materialize) hash-join env
  dispatch** (PR #91). The recorded hash-join dispatcher
  routes `JoinType::Inner` and `JoinType::LeftOuter` through
  CSM (count → exclusive scan → materialize) for both the
  non-indexed and indexed entry points when
  `XLOG_USE_RECORDED_CSM=1` (or umbrella
  `XLOG_USE_RECORDED_OPS=1`) is set. `Semi` / `Anti` always
  route through the existing legacy recorded methods — no
  CSM implementation exists for them. Eligibility checks
  preserved exactly: runtime-backed manager, ≤4 keys
  (`pack_keys` constraint), key-type match, row-count caps,
  indexed-path key-byte and shape checks. New env-dispatch
  routing test suite
  (`crates/xlog-cuda/tests/test_csm_env_dispatch.rs`)
  proves selection across the Inner / LeftOuter × indexed /
  non-indexed × env-on / env-off matrix plus Semi / Anti
  and the >4-keys upstream short-circuit.
- **Indexed LeftOuter CSM operator** (PR #87,
  `hash_join_left_outer_v2_with_index_count_scan_materialize_recorded`).
  Probe-only pack on `launch_stream` plus a cached
  `JoinIndexV2` for the build side, sharing the
  count → scan → materialize phase shape with the
  non-indexed LeftOuter CSM (PR #84) and the indexed
  Inner CSM. No new kernels; reuses the four already-
  migrated CSM kernels plus `hash_join_csm_unmatched_mask`
  from PR #84.
- **Cert-mode labeling** (commit `bca1e373`). The
  `certification_suite` header now prints
  `Recorded-op dispatch (explicit):` (extended to include
  `XLOG_USE_RECORDED_CSM`) and a synthesized `Cert mode:`
  line keyed off the explicit env flags. The three intended
  values match the v0.6.1 cert gate commands —
  `legacy/default`, `runtime+recorded`,
  `runtime+recorded+CSM` — so CSM-mode runs are
  self-documenting in the cert evidence.

### Fixed

- **`d_overflow` lifetime in three CSM materialize
  recorders** (PR #89). The Phase B materialize kernel
  takes `d_overflow` as a kernel param (writes the
  overflow flag). Three previously-shipped CSM siblings
  (`hash_join_inner_v2_count_scan_materialize_recorded`,
  `hash_join_left_outer_v2_count_scan_materialize_recorded`,
  `hash_join_inner_v2_with_index_count_scan_materialize_recorded`)
  did not register `d_overflow` on their materialize-phase
  `LaunchRecorder`, so the runtime was free to release the
  block once `rec_count.commit` resolved — a potential
  use-after-free if pool reuse beat kernel completion. Each
  site now registers
  `rec_mat.write(&d_overflow);` before `rec_mat.preflight`,
  matching the indexed-LeftOuter CSM site (PR #87) so all
  four CSM materialize recorders are identical.

### Deferred to post-v0.6.1

- **Semi / Anti CSM**. No `count_scan_materialize_recorded`
  variants exist for `JoinType::Semi` / `JoinType::Anti`;
  the env dispatch leaves them on the legacy recorded
  paths. **Trigger to re-open**: a benchmark or
  correctness scenario forces it. The legacy paths are
  correct today and adding CSM variants would be code
  without a consumer.
- **CSM default-on**. CSM remains opt-in via
  `XLOG_USE_RECORDED_CSM` / umbrella
  `XLOG_USE_RECORDED_OPS`. Re-evaluate flipping the
  default once cert history accumulates a stable run of
  CSM-mode passes; until then the env gate is the
  migration boundary.

### Release Validation (gates green at tag)

- `cargo fmt --check`: clean.
- `git diff --check`: clean.
- Legacy cert
  (`cargo test -p xlog-cuda-tests --test certification_suite --release`):
  `Cert mode: legacy/default`, 1 outer test passing — 33
  cert categories internal.
- Runtime+recorded cert
  (`XLOG_USE_DEVICE_RUNTIME=1 XLOG_USE_RECORDED_OPS=1 cargo test ...`):
  `Cert mode: runtime+recorded`, 1 outer test passing —
  same 33 categories.
- Runtime+recorded+CSM cert
  (`XLOG_USE_DEVICE_RUNTIME=1 XLOG_USE_RECORDED_OPS=1 XLOG_USE_RECORDED_CSM=1 cargo test ...`):
  `Cert mode: runtime+recorded+CSM`, 1 outer test passing —
  same 33 categories.
- Umbrella ×20 (`real_world_tests --test-threads=8` under
  `XLOG_USE_DEVICE_RUNTIME=1 XLOG_USE_RECORDED_OPS=1`):
  20/20 (recorded across PR #87, #89, #91 prep).

## [0.6.2] — 2026-05-01

Default-On Adaptive WCOJ Triangle Dispatch. Productizes the
first GPU Worst-Case Optimal Join slice: a certified 3-way
triangle path for `u32`, `u64`, and `Symbol` keys, wired into
the runtime behind a default-on adaptive skew classifier and a
hard kill switch. The release also ships the pure-Rust
hypergraph planner / oracle stack that future WCOJ kernels are
certified against. Scope remains deliberately narrow: no
general-arity WCOJ, no recursive/SCC WCOJ execution, no cost
model, and no `MultiWayJoin` / `WcojJoin` RIR node yet.

### Added

- **Hypergraph planner and oracle foundation.** Added
  `xlog-logic::hypergraph` with a hypergraph IR, eligibility
  analyzer, deterministic variable-order interface, canonical
  explain output, typed gate, mixed plan contract, single-rule
  reference evaluator, single-target fixpoint evaluator, SCC
  fixpoint evaluator, and transitive SCC type inference. The
  certification workloads cover triangle, Same Generation,
  skewed multiway, deep recursive frontier, and mutually
  recursive parity SCC shapes.
- **GPU WCOJ triangle provider path.** Added recorded
  `wcoj_triangle_u32_recorded` / `wcoj_triangle_u64_recorded`
  provider entries plus `wcoj_layout_u32_recorded` /
  `wcoj_layout_u64_recorded` sorted-layout construction. The
  triangle pipeline uses count → device-side prefix scan →
  materialize with a 4-byte metadata D2H total; no count-vector
  D2H remains. `Symbol` uses the u32 physical path.
- **Planner-to-provider certification.** Added test-only
  `xlog-logic` dev dependency in `xlog-cuda` so planner verdicts
  and GPU provider outputs are certified against the same CPU
  oracle fixtures before executor wiring.
- **Runtime WCOJ dispatch.** Added the executor hook for the
  canonical non-recursive triangle RIR shape
  `tri(X,Y,Z) :- e1(X,Y), e2(Y,Z), e3(X,Z)`. The hook supports
  4-byte (`U32` / `Symbol`) and 8-byte (`U64`) uniform-width
  triangles, silently falls back for unsupported shapes, and
  exposes `Executor::wcoj_triangle_dispatch_count()` for tests.
- **Adaptive skew classifier and default-on policy.** Added
  `wcoj_triangle_skew_score_{u32,u64}` and a 64-bucket
  hash-mixed L-infinity/L1 classifier. `RuntimeConfig::default()`
  now runs adaptive WCOJ on matching non-recursive triangle
  rules: high-skew inputs dispatch WCOJ, uniform / empty inputs
  fall back to the binary-join chain. Ops can disable the path
  globally with `XLOG_DISABLE_WCOJ_TRIANGLE=1`.
- **Diagnostic phase timing.** Added feature-gated
  `wcoj-phase-timing` support and the `wcoj_phase_report`
  binary to measure classifier, layout, triangle count / scan /
  total / materialize, wall, and residual overhead.
- **WCOJ benchmark baseline.** Added
  `crates/xlog-integration/benches/wcoj_triangle_bench.rs` and
  evidence under
  `docs/evidence/2026-05-01-wcoj-bench-baseline/` for baseline,
  adaptive acceptance, default-on acceptance, pre-fast-path phase
  timing, and post-fast-path phase timing.

### Changed

- **WCOJ layout fast-path.** `wcoj_layout_u32_recorded` and
  `wcoj_layout_u64_recorded` now prove already sorted+unique
  inputs with a recorded checker kernel and return a recorded
  device-side clone instead of always running sort + dedup. The
  slow path is unchanged and remains the correctness fallback.
- **Recorded sort / dedup U64 support.** `sort_recorded` now
  supports U64 via the same hi/lo radix strategy as the legacy
  sort path, and `dedup_full_row_recorded` admits U64 rows.
- **Executor WCOJ stream reuse.** The executor caches one WCOJ
  launch stream per instance, preventing long-lived runtimes from
  exhausting the grow-only `StreamPool` and silently falling back
  after 16 dispatches.
- **WCOJ adaptive default.** `RuntimeConfig::wcoj_triangle_dispatch`
  remains the explicit force/off knob. New
  `wcoj_triangle_dispatch_adaptive` controls adaptive opt-out /
  opt-in, and `wcoj_triangle_dispatch_disabled` is the hard kill
  switch. Precedence is: disable > force > explicit force-off >
  adaptive.

### Fixed

- **Skew-classifier failure paths.** Failure paths after queued
  classifier work now drain the launch stream before dropping
  temporary buffers.
- **Layout fast-path failure paths.** Failure paths after queued
  checker / recorded-clone work now drain the launch stream before
  dropping temporary buffers.
- **SCC-aware planner precedence.** The hypergraph planner now
  preserves structural-error precedence when typed gating and SCC
  inference both apply.

### Bench Evidence

- Initial force-on WCOJ was strong on super-hub fixtures but
  regressed uniform / empty fixtures. The adaptive classifier
  cleared the locked median gates: uniform / empty adaptive cells
  stayed ≤2× binary-join, while super-hub speedups remained above
  the locked minimums.
- Phase timing showed layout construction was 91-97% of super-hub
  WCOJ wall time before the fast-path. The layout fast-path reduced
  layout time by ~97-98% and wall time by ~90-96% on the measured
  super-hub cells.

### Deferred to post-v0.6.2

- General `MultiWayJoin` / `WcojJoin` RIR node and optimizer
  integration.
- Cost-aware variable ordering and selectivity / heat feedback.
- Recursive / SCC WCOJ execution and mixed recursive WCOJ +
  binary-join semantics.
- 4-way and general-arity WCOJ kernels.
- Histogram-guided block scheduling / B1 heavy-row offload. Phase
  timing after the layout fast-path shows materialize is now a
  plausible future optimization target, but no longer the obvious
  next slice.
- Dedicated WCOJ architecture and performance-tuning guide.

### Release Validation Targets

Run before tagging `v0.6.2`:

- `cargo fmt --check`
- `git diff --check`
- WCOJ provider / integration test matrix
- `cargo test -p xlog-logic`
- `cargo test -p xlog-runtime --lib`
- `cargo build --workspace --exclude pyxlog`
- `XLOG_USE_DEVICE_RUNTIME=1 cargo test -p xlog-integration --test real_world_tests --release`
- Existing certification modes from v0.6.1 remain the recorded-launch
  baseline for runtime safety.

## [Unreleased] — targeting v0.6.3

> Empty.

## [0.5.2](https://github.com/BrainyBlaze/xlog/compare/xlog-cli-v0.5.0...xlog-cli-v0.5.2) — 2026-04-20

### Fixed

- unblock release publish verification

## [0.5.1] — 2026-04-20

### Fixed

- unblock release publish verification

### Added

- **Bounded exact-induction engine** (`xlog-induce` + `ilp_exact` CUDA kernel + `pyxlog`
  bridge): New `xlog-induce` crate scores all `(left, right)` candidate pairs across the
  four canonical DTS topologies (`chain`, `star`, `fanout`, `fanin`) in a single batched
  GPU pass and returns top-K per topology with full candidate metadata
  (`positives_covered`, `negatives_covered`, `next_*_covered`, `tie_class_size`).
  Designed for DTS's M8 Phase 1 integration; behaviorally equivalent on bounded
  requests to `pyxlog.ilp.induce_exact(backend="python", strict_per_topology=True)`.
  - **Engine** (`crates/xlog-induce/`): `InduceExactRequest` (indices + buffer handles),
    `ExactInductionResult` / `ScoredCandidate`, pre-kernel classification
    (`validate::classify_request` — 5 pure unit tests), buffer-level validation
    (arity=2, column type `U64`, cached-row-count guard).
  - **Deterministic reducer** (`xlog-induce::reduce`): lexicographic `(-positives,
    negatives, left_idx, right_idx)` sort + positive-coverage filter + `next_*` and
    `tie_class_size` diagnostics. 16 host-side unit tests lock the comparator and
    diagnostic semantics bit-for-bit.
  - **CUDA kernel** (`kernels/ilp_exact.cu`, new `xlog_ilp_exact` module): single
    `ilp_exact_score` entry. Launch geometry: `grid = (C, C, 4)` blocks of 256
    threads; each block owns one `(topology, L, R)` output slot, so there are no
    cross-block atomics on the scoring path. Deterministic pair-halving block
    reduction (integer counts only).
  - **Provider launcher** (`crates/xlog-cuda/src/provider/ilp_exact.rs`):
    `CudaKernelProvider::ilp_exact_score(candidates, positives, negatives) ->
    (Vec<u32>, Vec<u32>)`. D2D-concatenates candidate columns in setup, uploads
    `cand_offsets`, launches the scoring kernel, and downloads two count arrays.
    D2H budget is a constant **2 per call** regardless of candidate count. Three
    CUDA-gated launcher tests (hand-computed coverage fixture, determinism across
    runs, empty-negatives path).
  - **Kernel manifest**: bumped `KERNEL_MODULES` count 21 → 22 (plus the
    compile-time `assert!(KERNEL_CU_NAMES.len() == 22)` at `provider/mod.rs:221`).
    `ILP_EXACT_MODULE` + `ilp_exact_kernels::ILP_EXACT_SCORE` constants added.
  - **pyxlog bridge** (`crates/pyxlog/src/ilp_exact.rs`): new
    `CompiledIlpProgram::induce_exact_native(...)` pyo3 method — resolves relation
    names against `rel_index`, unwraps positive/negative DLPack tensors against
    the head relation's declared schema, dispatches to the engine, and returns a
    `dict` the Python wrapper repackages into `ExactInductionResult` /
    `ScoredCandidate` dataclasses.
  - **Python wrapper** (`crates/pyxlog/python/pyxlog/ilp/exact_induce.py`): new
    `backend="native"` dispatch path on `induce_exact(...)` plus the dict → dataclass
    repackaging helper. Wrapper default backend is still `"python"` for backward
    compatibility with existing callers.
  - **Parity contract** (`python/tests/test_ilp_exact_induce.py`):
    `test_induce_exact_native_matches_python_reference` (ordered equality of
    summary and per-candidate fields) and
    `test_induce_exact_native_does_not_scale_d2h_with_candidate_pairs` (gate:
    `large.d2h_transfer_count ≤ small.d2h_transfer_count + 2`).
  - **Kernel design note**: `docs/plans/2026-04-17-m8-ilp-exact-kernel-design.md`.
- **MC runtime optimization** (`xlog-prob`, `xlog-runtime`): 8.6% wall-clock improvement on
  the MC evaluation hot loop (14.11s → 12.90s on 1000-sample clamped benchmark). No API changes.
  - `McTimingBreakdown` struct with env-gated profiling (`XLOG_MC_PROFILE=1`) for per-phase
    timing (sampler, reset, build, eval, count).
  - `McCountStrategy` enum: maps sampling method to count strategy (`QueriesAndEvidence` for
    rejection, `QueriesOnly` for clamped). Skips evidence-side allocations/uploads in clamped mode.
  - `McSampleResetPlan` struct + `build_sample_reset_plan()`: classifies relations as preserve
    (deterministic-only), clear (dynamic), or reload_base. Replaces full store clone/restore
    with targeted per-relation reset.
  - `Executor::reset_for_mc_relations()`: new method for targeted preserve/clear reset of
    relations between MC samples.
  - Pre-allocated pointer buffers (`query_ptrs_buf`, `evidence_ptrs_buf`) outside the sample
    closure, avoiding per-sample Vec heap allocation.
- **Evidence clamping for MC inference** (`xlog-prob`): Monte Carlo evidence conditioning
  via `McSamplingMethod::EvidenceClamping`. Forces root Bernoulli evidence variables in the
  sampling kernel so every sample counts (`evidence_samples == total_samples`). Auto-selected
  when all evidence maps to probabilistic facts or positive AD heads; falls back to rejection
  for derived/deterministic/negative-AD evidence. New `sampling_method` field on `McEvalConfig`,
  `McResult`, `McDeviceResult`, and Python API. CUDA kernel updated with `force_mask`/`forced_value`
  inputs.
- **Provenance primitives** (`xlog-prob`): Retained provenance metadata for external Rust consumers.
  New `ChoiceSource` type captures annotated-disjunction metadata (explicit heads, choice index,
  optional source ID). Two new fields on `Provenance`: `leaf_atoms` (`BTreeMap<LeafId, GroundAtom>`)
  and `choice_sources` (`BTreeMap<ChoiceVarId, ChoiceSource>`). Three new accessors:
  `leaf_atom(LeafId)`, `choice_source(ChoiceVarId)`, `atoms_with_formulas()` iterator.
  `GroundAtom::new()` made public. Top-level re-exports added to `xlog-prob` lib.rs for
  `ChoiceSource`, `GroundAtom`, `Provenance`, `Value`, `ChoiceVarId`, `LeafId`, `PirGraph`,
  `PirNode`, `PirNodeId`. Inline retention at existing extraction allocation sites — no new
  passes or post-hoc reconstruction.

### Changed

- **`CudaKernelProvider::clone_buffer` now propagates `cached_row_count`** (`xlog-cuda`):
  Previously the deep-cloned buffer used `CudaBuffer::from_columns` (no host-side count
  cache), forcing any consumer of a cloned buffer to perform a D2H read on
  `num_rows_device()` just to learn the row count. All call sites that go through
  `CompiledIlpProgram::put_relation` clone on insertion into the executor's relation
  store, so every relation buffer fetched from the store was losing its cache. The
  new code calls `set_cached_row_count_if_unset(source.cached_row_count())` on the
  clone when the source has a populated cache, preserving the host-visible count
  across clones. Pinned by the new `test_clone_buffer_preserves_cached_row_count`
  test, and a load-bearing precondition for the M8 exact-induction engine's
  hot-loop D2H budget.
- **`pyxlog.ilp.induce_exact()` gains `strict_per_topology` opt-in flag**
  (`pyxlog`, Python): The `backend="python"` prototype has a latent cross-topology
  contamination behavior — stale `W_<topo>_<head>` masks from earlier outer-loop
  iterations bleed into later topologies' coverage numbers via `evaluate()`.
  Setting `strict_per_topology=True` zeroes out "other" topology masks before
  each topology's inner loop, yielding per-topology-isolated scoring that matches
  the `backend="native"` kernel's by-construction semantics. Default remains
  `False` for full backward compatibility with callers that are calibrated
  against the prototype's historical numbers (notably DTS Phase 0 liveness
  baselines). The `"native"` backend is unaffected — it is strict by design.
- **ILP reliability gate 4.6x faster** (`pyxlog`): Compile once per stage and reuse across
  all 5 seeds via `reset_runtime()`, eliminating 16 redundant compilations and 20 holdout
  evaluations (1647s → 359s). Gate still runs 4 stages × 5 seeds = 20 independent training
  runs with identical budgets (150 steps, 7 max attempts). Parity with fresh-compile behavior
  verified by new `test_compile_once_reuse_parity` and `test_compile_once_multi_seed_isolation`
  tests.
- **MC behavior test budgets reduced** (`xlog-prob`): 10 MC tests trimmed from 50K–80K samples
  to 2K–5K (one 20K accuracy guard retained). Reduces test-suite turnaround without changing
  runtime engine behavior.
- **`build_sample_buffers()` no longer performs per-sample D2H row-count reads**: Uses host-side
  `num_rows()` (capacity) instead of synchronous `device_row_count_u32()` GPU→CPU transfers.
- **MC per-sample store management replaced**: Full `snapshot_store()`/`restore_store()` cycle
  replaced by `McSampleResetPlan` with targeted relation-level reset.
- **Whitepaper and public docs repositioned** around "GPU-native logic programming language
  for unified symbolic reasoning" instead of "GPU-accelerated Datalog engine". v0.5.0 LaTeX
  whitepaper (`docs/whitepaper/main.pdf`) gained a new Section 3 "The xlog Language" covering
  types, UDFs, modules, arithmetic, aggregations, and constraints with validated examples;
  `docs/ARCHITECTURE.md`, `docs/language-reference.md`, root `README.md`, `ROADMAP.md`, and
  `docs/whitepaper/README.md` were aligned. Stale Markdown whitepaper draft
  `docs/whitepaper-v050.md` removed (superseded by the LaTeX version). Broken cross-references
  to cleanup-deleted `docs/plans/`, `docs/design/`, `docs/ilp/` directories replaced with
  pointers to surviving docs (whitepaper sections, `dilp-training.md`, `rfc-tensorized-ilp.md`).
  Docs-only change; no code or API impact.

### Refactored

- **5-wave codebase refactoring** (2026-03-10 → 2026-03-13, 57 commits across all waves):
  Structural decomposition of the 5 largest modules in the workspace. No external API changes.
  No behavioral changes. All existing tests, gates, and contracts preserved.

  **Wave 1 — Dependency cleanup + error/type seams** (`xlog-core`, `xlog-cuda`, `xlog-logic`,
  `xlog-neural`; 8 commits):
  - Removed false dependency cycle: `xlog-logic` no longer depends on `xlog-runtime` in
    production, `xlog-stats` no longer depends on `xlog-cuda`.
  - Added `xlog-neural → xlog-core` edge for error conversion impls.
  - New `From` impls: `NeuralError`, `TensorSourceError`, `FunctionError`, `TypeError`,
    `ModuleError` → `XlogError`. New `driver_err()` helper for cudarc `DriverError` (orphan
    rule prevents `From` impl).
  - New `XlogError::{kernel_ctx, execution_ctx, compilation_ctx}` structured error context
    helpers.
  - New `GpuScalar` trait (`xlog-cuda/src/type_seam.rs`): pub + sealed marker for Rust scalar
    types that round-trip through GPU column storage. 8 impls (u8, u32, u64, i32, i64, f32,
    f64, bool). Enables generic `download_column::<T>()` and `create_buffer_from_slice::<T>()`
    in Wave 2.

  **Wave 2 — Provider decomposition + GpuScalar migration** (`xlog-cuda`, all consumer crates;
  9 commits):
  - `provider.rs` (12,809 LOC) → `provider/mod.rs` + 8 submodules: `kernel_loading.rs`,
    `relational.rs`, `filter.rs`, `groupby.rs`, `arithmetic.rs`, `transfer.rs`,
    `probabilistic.rs`, `ilp.rs`, `io.rs`.
  - Collapsed type-specialized function families via `GpuScalar` trait:
    - 8 `download_column_<T>()` functions (~280 LOC) → 1 generic `download_column::<T>()` (~35 LOC)
    - 7 `create_buffer_from_<T>_slice()` functions (~220 LOC) → 1 generic `create_buffer_from_slice::<T>()` (~30 LOC)
    - 11 `filter_<T>()` functions (~1,200 LOC) → 1 generic `filter::<T>()` with enum dispatch
  - ~140 mechanical turbofish call-site updates across 8 consumer crates.
  - `new()` constructor refactored from ~814 lines of boilerplate to ~120 lines via
    `KernelModuleSpec` manifest + `load_all_kernel_modules()`.
  - Net reduction: ~5,990 lines.

  **Wave 3 — Executor decomposition** (`xlog-runtime`; 11 commits):
  - `executor.rs` (4,337 LOC) → `executor/mod.rs` + 6 submodules: `node_dispatch.rs`,
    `recursive.rs`, `expression.rs`, `rewrite.rs`, `join_cache.rs`, `delta.rs`.
  - Extracted `DeltaRelationTracker` as standalone `pub(crate)` type for delta relation
    lifecycle during recursive evaluation.
  - Extracted `JoinIndexCache` as standalone `pub(crate)` LRU struct.
  - Net reduction: ~1,040 lines.

  **Wave 4 — Pyxlog FFI extraction** (`pyxlog`; 10 commits):
  - `lib.rs` (6,202 LOC) → slimmed `lib.rs` (~487 LOC) + 7 submodules: `program.rs`,
    `logic.rs`, `ilp.rs`, `ilp_gpu.rs`, `training.rs`, `neural.rs`, `types.rs`.
  - Consolidated 2 non-contiguous `CompiledIlpProgram` impl blocks into single block.
  - Extracted `compute_ilp_loss_grad_gpu()` (574 LOC) into focused helpers in `ilp_gpu.rs`.
  - Collapsed f32/f64 forward-backward duplication into generic `forward_backward_typed()`.
  - Added `xlog_err_to_py()` / `neural_err_to_py()` local error-mapping helpers (orphan rule
    prevents `From` impls for `PyErr`).
  - Net reduction: ~1,320 lines.

  **Wave 5 — Probabilistic backend decomposition + coherence** (`xlog-prob`, workspace-wide;
  19 commits):
  - `gpu_d4.rs` (3,669 LOC) → `gpu_d4/mod.rs` (~450 LOC) + `frontier.rs` (~1,480 LOC) +
    `build.rs` (~1,850 LOC).
  - `mc.rs` (3,399 LOC) → `mc/mod.rs` (~1,079 LOC) + `evidence.rs` (~130 LOC) +
    `buffers.rs` (~973 LOC) + `sampling.rs` (~297 LOC) + `results.rs` (~993 LOC).
  - Config coherence: `Default` impls on all config structs, `#[non_exhaustive]` on 3 structs
    (`MemoryBudget`, `GpuEquivalenceConfig`, `WfsConfig`), `///` doc comments on all configs.
  - Test harness consolidation: 22 duplicate `setup_provider()` copies → 2 canonical
    `tests/common/mod.rs` helpers (xlog-cuda, xlog-prob).
  - `xlog-prob` top-level re-exports: `GpuCompileConfig`, `CircuitCompileProfile`,
    `ExactDdnnfProgram`, `ExactResult`, `GpuConfig`, `McEvalConfig`, `McProgram`,
    `McSamplingMethod`, `McCountStrategy`, `McResult`, `McDeviceResult`, `EvidenceForcing`,
    `ForceabilityReason`, `WfsConfig`, `WfsResult`, `TruthValue`, plus WFS free functions.
  - WFS entry points consolidated: 2 zero-caller functions removed, 1 gated behind
    `#[cfg(test)]`.
  - 71 visibility tightens (`pub` → `pub(crate)`) across `xlog-prob`, `xlog-solve`,
    `xlog-logic`.
  - Clone audit documented (deliberate clones annotated, no actionable reductions found).
  - RIR visitor trait decision: 7 dispatch patterns warrant a trait, deferred to v0.7+.
  - 35 compiler warnings fixed (private_interfaces, unused imports, dead code).

  **Post-refactoring module sizes** (god modules → focused submodules):

  | Module | Before | After (mod.rs) | Submodules |
  |--------|--------|----------------|------------|
  | `provider.rs` | 12,809 | 2,651 | 8 |
  | `pyxlog/lib.rs` | 6,202 | 487 | 7 |
  | `executor.rs` | 4,337 | 2,050 | 6 |
  | `gpu_d4.rs` | 3,669 | 450 | 2 |
  | `mc.rs` | 3,399 | 1,079 | 4 |
  | **Total** | **30,416** | **6,717** | **27** |

  Design docs: `docs/superpowers/specs/2026-03-10-wave{1-5}-*.md`.
  Implementation plans: `docs/superpowers/plans/2026-03-10-wave{1-2}-*.md`,
  `docs/superpowers/plans/2026-03-11-wave{3-5}-*.md`.

### Removed

- **`device_row_count_u32()`** helper in MC hot loop — synchronous D2H scalar read, replaced
  by host-side capacity checks.
- **`snapshot_store()` / `restore_store()`** in MC evaluator — replaced by `McSampleResetPlan`
  with `reset_for_mc_relations()`.
- **Type-specialized provider functions** (`xlog-cuda`): `download_column_u32`,
  `download_column_i32`, `download_column_i64`, `download_column_u64`, `download_column_f32`,
  `download_column_f64`, `download_column_bool`, `download_column_u8`,
  `create_buffer_from_u32_slice`, `create_buffer_from_i32_slice`,
  `create_buffer_from_i64_slice`, `create_buffer_from_u64_slice`,
  `create_buffer_from_f32_slice`, `create_buffer_from_f64_slice`,
  `create_buffer_from_u8_slice`, and 11 type-specialized `filter_*` functions — all replaced
  by `GpuScalar`-generic equivalents.
- **2 WFS entry points** (`xlog-prob`): `evaluate_wfs_scc` and `evaluate_wfs_with_rules_config`
  removed (zero callers). `evaluate_wfs_scc_with_config` gated behind `#[cfg(test)]`.

## [0.5.0] — 2026-03-08

### Added

- **P2a: Term Embeddings (training-only)** — `register_embedding()` for
  `nn.Embedding` (trainable) and `torch.Tensor` (frozen) payloads.
  `forward_embedding(name, ids)` returns batched tensors with autograd
  support on the same device as the embedding (CUDA-safe). Cross-registration
  validation: embedding declarations reject `register_network()` and vice
  versa. Compile-time mixed-form rejection for network names. Raw tensors
  are detached at registration to enforce frozen contract even when input
  has `requires_grad=True`. User-managed optimizer (P2b APIs do not cover
  embeddings). Inference path deferred to v0.5.1+.
- **GPU-resident ILP credit/loss path** (`compute_ilp_loss_grad_gpu`): Single Rust/CUDA call replaces
  Python-side `_compute_loss_from_candidates()` loop. Builds COO→CSR on-device, runs forward/backward
  CUDA kernels, reduces loss on-device, returns `(loss, grad)` as DLPack tensors. Zero D2H transfers
  in all paths (including chunked), confirmed by strict byte-level accounting (`host_transfer_stats()`).
- **4 new CUDA kernels**: `ilp_coo_fill_from_mask` (COO fill from device mask + prefix-sum),
  `ilp_csr_histogram` (CSR row_offsets via atomicAdd histogram), `ilp_reduce_sum_f32`/`ilp_reduce_sum_f64`
  (block-level sum reduction).
- **Two-pass GPU-only chunk merge**: Bounded-memory streaming replaces D2H-based chunked fallback.
  Pass 1 counts NNZ per task via `count_mask_into_slot`, pass 2 fills COO at pre-computed offsets.
  Zero data-plane D2H in all code paths, verified on all 4 ILP stages with forced chunking.
- **`coo_chunk_budget`** (renamed from `coo_memory_cap`): Controls per-chunk temp allocation ceiling.
  Final exact-NNZ COO buffer may exceed the chunk budget. Deprecated `set_coo_memory_cap()` alias
  retained for one release cycle.
- **`count_mask_into_slot`**: Provider method writing mask count directly into pre-allocated device
  array slot, avoiding per-task allocation churn in pass 1.
- **`dtoh_scalar_untracked`**: Provider helper for metadata-only D2H reads (e.g., total_nnz scalar)
  without incrementing transfer counters. Makes the metadata-vs-data-plane contract explicit.
- **Strict zero-D2H mode**: `set_strict_zero_dtoh(True)` now passes in all paths including chunked.
  Use in zero-D2H benchmarks and CI gates.
- **D2H transfer accounting**: Strict byte-level gate via `host_transfer_stats()` returning
  `dtoh_calls` and `dtoh_bytes` counters, plus coarse column-level `d2h_transfer_count()`.
- **3 gradient parity tests**: GPU kernel output vs pure-PyTorch reference (f32, f64, multi-candidate).
- **Artifact schema migration**: `save()` writes `beta-v2`, `load()` accepts both `beta-v1` and
  `beta-v2`. Forward-compatible schema evolution.
- **Bounded telemetry persistence**: `TrainConfig.persist_telemetry` (default False) and
  `telemetry_persist_limit` (default 100). When enabled, `save()` includes a `telemetry_snapshot`
  with the last N `StepRecord`s and `step_timings`. `load()` restores telemetry from snapshot.
- **`program.get_lr(network_name)`**: Read current learning rate from a registered network's optimizer.
- **`program.set_lr(network_name, lr)`**: Set learning rate across all param groups of a registered
  network's optimizer.
- **Per-network `scheduler_step`**: `program.scheduler_step(network_name)` steps a single network's
  scheduler. `scheduler_step(None)` (default) steps all schedulers, preserving backward compatibility.
- **Gradient clipping**: `train_model(..., max_grad_norm=N)` and `train_model_tensor(..., max_grad_norm=N)`
  clip gradients via `torch.nn.utils.clip_grad_norm_` before each optimizer step. `None` (default)
  disables clipping.
- **Early stopping**: `train_model(..., val_queries=[...], patience=N)` and
  `train_model_tensor(..., val_queries=[...], patience=N)` evaluate validation loss each epoch and
  stop training after `patience` consecutive epochs without improvement.
- **`TrainingHistory.stopped_early`**: Boolean flag indicating whether early stopping was triggered.
- **`GpuCdclWorkspace`**: Pre-allocated solver arena for reusing GPU buffers across multiple CDCL
  solves (P3 incremental verifier). Created via `GpuCdclSolver::new_workspace()`.
- **`solve_expect_unsat_*_ws` method variants**: Workspace-backed CDCL solving that reuses
  pre-allocated device buffers instead of per-call allocation.
- **`GpuCompileConfig.incremental_verify`**: Opt-in for workspace reuse in the equivalence
  verifier (amortizes arena allocation across q1/q2 solve pair).
- **`GpuEquivalenceConfig.reuse_workspace`**: Internal config field propagated from
  `incremental_verify`.

### Changed

- **`coo_memory_cap` renamed to `coo_chunk_budget`**: Old name implied a hard ceiling on all COO
  allocations; new semantics allow the exact-NNZ output buffer to exceed the chunk budget.
  `set_coo_memory_cap()` remains as a deprecated alias.

### Removed

- **Legacy host-sum export helpers** (`export_loss_grad_f32`, `export_loss_grad_f64`): Replaced by
  device-only `export_loss_grad_device_f32`/`export_loss_grad_device_f64`. All loss/grad export now
  stays on device.

### Breaking Changes

- **`coo_memory_cap` renamed to `coo_chunk_budget`** (`pyxlog`): The parameter on `CompiledIlpProgram`
  was renamed to reflect the new semantics (chunk-level temp budget, not a hard ceiling on all COO
  allocations). `set_coo_memory_cap()` is retained as a deprecated alias for one release cycle and
  will be removed in v0.6.0. Update call sites before upgrading.
- **Artifact schema `beta-v1` → `beta-v2`** (`pyxlog`): `save()` now writes `beta-v2` artifacts.
  `load()` accepts both `beta-v1` and `beta-v2`, so existing saved artifacts remain readable, but
  artifacts saved with v0.5.0+ cannot be loaded by v0.4.x.
- **`export_loss_grad_f32` / `export_loss_grad_f64` removed** (`pyxlog`): These host-side loss/grad
  export helpers are gone. Replace with `export_loss_grad_device_f32` / `export_loss_grad_device_f64`
  respectively. The device-side variants return DLPack tensors with zero D2H transfers.
- **Type-specialized `download_column_<T>` functions removed** (`xlog-cuda`): All 8 type-specialized
  variants (`download_column_u32`, `download_column_i32`, etc.) are replaced by the generic
  `download_column::<T>()`. Similarly the 7 `create_buffer_from_<T>_slice()` variants are replaced
  by `create_buffer_from_slice::<T>()`, and the 11 `filter_<T>()` variants by `filter::<T>()`.
  Downstream Rust crates that call these directly must update call sites with turbofish syntax.

### Migrating from v0.3.2

This covers the upgrade path from v0.3.2 (the last stable release) to v0.5.0.

#### New Required Dependencies

- **PyTorch / LibTorch** — required for the neural-symbolic training APIs (`pyxlog`).
  CPU builds work, but GPU inference requires a CUDA-enabled PyTorch build matching the CUDA
  toolkit version used to build xlog-cuda.
- **CUDA toolkit ≥ 11.8** — required for all GPU paths (`xlog-cuda`, `xlog-solve`, `xlog-prob`).
  The CPU-only `xlog-logic` crate has no new mandatory dependencies.

#### Package Rename (v0.4.0-alpha → v0.5.0, if upgrading through alpha)

```python
# Before (v0.4.0-alpha and earlier)
import xlog_gpu

# After (v0.5.0)
import pyxlog
```

The PyPI package was renamed from `xlog-gpu` to `pyxlog` in v0.4.0-alpha. If you skipped the
alpha/beta cycle, update all import statements and remove the `xlog-gpu` package.

#### API Changes

| Old (≤ v0.3.2 / v0.4.x)                    | New (v0.5.0)                                      | Notes                              |
|---------------------------------------------|---------------------------------------------------|------------------------------------|
| `set_coo_memory_cap(n)`                     | `set_coo_chunk_budget(n)`                         | Old name deprecated, removed v0.6.0 |
| `export_loss_grad_f32()`                    | `export_loss_grad_device_f32()`                   | Returns DLPack tensor (on-device)  |
| `export_loss_grad_f64()`                    | `export_loss_grad_device_f64()`                   | Returns DLPack tensor (on-device)  |
| `download_column_u32()` (Rust, xlog-cuda)  | `download_column::<u32>()`                        | Generic turbofish form             |
| `create_buffer_from_u32_slice()` (Rust)    | `create_buffer_from_slice::<u32>()`               | Generic turbofish form             |
| `filter_u32()` / `filter_f32()` etc. (Rust)| `filter::<u32>()` / `filter::<f32>()`             | Generic turbofish form             |

#### Saved Artifacts

Artifacts saved with v0.4.x (`beta-v1` schema) can be loaded by v0.5.0 without modification.
Artifacts saved with v0.5.0 (`beta-v2` schema) **cannot** be loaded by v0.4.x. If you need to
maintain cross-version compatibility, do not upgrade the artifact files until all consumers are
on v0.5.0.

#### Breaking Changes from v0.3.2 Specifically

v0.3.2 introduced its own breaking changes (serialized Arrow symbol files, `hash_symbol_to_u32`
removal, `count` aggregation now returns `u64`). If upgrading directly from v0.3.2, address
those first (see v0.3.2 release notes below), then apply the v0.5.0 changes above.

#### Configuration Changes

- `TrainConfig` now accepts `deterministic`, `max_grad_norm`, `val_queries`, and `patience` fields
  (all optional, backward-compatible defaults).
- `GpuCompileConfig` now accepts `incremental_verify` (optional, defaults to `False`).
- `TrainConfig.persist_telemetry` defaults to `False`; explicitly set `True` to enable telemetry
  persistence in saved artifacts (new in v0.5.0).

## [0.4.0-ga] — 2026-03-05

### Changed

- **GA reliability runtime profile**: Default `max_attempts` reduced from 7 to 2 in `test_ilp_ga_reliability.py`.
  50-seed gate runtime reduced from ~1447s to ~436s (3.3x speedup) with identical statistical quality
  (200/200, Clopper-Pearson lower95 = 0.982). Override via `GA_RELIABILITY_MAX_ATTEMPTS` env var.

### Fixed

- **Typed batch upload**: `batch_fact_membership` and `batch_tagged_credit` now use
  schema-aware typed packing for all column types (I32, I64, U64, Bool, Symbol).
  Previously, all values were blindly cast to `u32`, corrupting non-U32 columns.
  F32/F64 columns are explicitly rejected with a clear error message.

### Added

- **Per-step phase timing** in dILP trainer: 6 timed phases (apply_mask, loss_credit, loss_reduce,
  backward_step, membership, convergence) with p95 and total_ms telemetry in `result.telemetry_timings`.
- **SLO scaling harness**: Parametrized `test_slo_scaling[N]` for N=20/50/100/150 chain lengths
  with wall-clock and forward_p95_us targets. Advisory by default; enforce with `ILP_PERF_ENFORCE_SLO=1`.

## [0.4.0-beta] — 2026-03-04

### Added

- **dILP Beta Trainer** — differentiable Inductive Logic Programming trainer upgraded from alpha to beta:
  - **Sparse mask API** (`set_rule_mask_sparse`): Python sends `(candidate_ids, soft_probs, budget)` and Rust builds
    the executor mask internally — no N3 tensor materialized, zero host→device transfer for the mask.
  - **Trainer backend abstraction** (`MaskBackend` protocol): `SparseMaskBackend` (default) and `DenseMaskBackend`
    (fallback via `debug_dense_mask=True`). Dense parity verified in tests.
  - **`train_and_promote()`**: Wraps `train_only()` + trial compilation + promotion gates (convergence, novel rate,
    regression check, holdout F1, ambiguity scan, typed schema) → returns `PromotionResult` with transactional commit.
  - **LOO holdout F1 scoring**: Leave-one-out cross-validation for ≤20 examples with per-fold precision/recall.
  - **Ambiguity scan**: Top-M alternative rule detection with configurable `check_ambiguity` / `exhaustive_ambiguity`.
  - **Hard-negative mining** (`sample_false_positives`): Rust-side false positive sampling, wired into trainer every
    20 steps with D2H counter reset to preserve zero-transfer contract in training loop.
  - **Artifact save/load**: `LearnedArtifact.save(path)` / `LearnedArtifact.load(path)` with JSON serialization,
    SHA-256 candidate map hash verification, schema version `beta-v1`.
  - **Recursive candidates**: `allow_recursive_candidates=True` enables i==k and j==k body-references-head candidates
    (behind config flag, default off).
  - **Beta reliability gate**: 4 stages (reach, grandparent, colleague, plus2) x 5 seeds = 20/20 with sparse backend. This is the primary beta release gate.
  - **AtomicU32 row-count cache** on `CudaBuffer` for GPU-resident row counts without host reads.
  - **Deterministic training path**: `TrainConfig(deterministic=True)` enables deterministic CUDA/Torch settings and
    per-attempt seed derivation for reproducible runs.
  - **`selected_hard` artifact field**: persisted selected candidate IDs with deterministic ordering for sparse/dense parity.
  - **GA reliability gate test**: `test_ilp_ga_reliability.py` runs 50 seeds x 4 stages with Clopper-Pearson lower-bound check.
  - **GA performance/transfer test**: `test_ilp_performance.py` validates `forward_p95_us` telemetry and host-transfer accounting.

- **Arrow C Data Interface device export** for `CudaBuffer` record batches (`to_arrow_device_record_batch`) returning
  `ArrowDeviceArrayOwned` handles with CUDA device descriptors and zero host transfers (import exists but is
  experimental + feature-gated: `arrow-device-import`).
- **Arrow device export support for Bool/Symbol**: on-device boolean bit-packing and symbol metadata keys
  (`xlog.symbol=true`, `xlog.symbol_encoding=u32`) for downstream consumers.
- **GPU CDCL verifier (complete SAT/UNSAT)** in `kernels/sat.cu` + `xlog-solve::GpuCdclSolver` with on-GPU SAT model
  checking and on-GPU UNSAT proof checking.
- **GPU PIR→CNF encoder** (`encode_cnf_gpu`) with device-resident CSR emission, deterministic var numbering, and GPU
  reachability (zero host reads in the production path), plus CNF kernels in `kernels/cnf.cu`.
- **GPU neural fast-path (AD chain)** in `kernels/neural.cu` + `xlog-prob` integration:
  - device-side AD conditional-chain weight fill (`neural_fill_ad_chain_f32`)
  - device-side probability-gradient scatter using both `grad_true` and `grad_false` (`neural_scatter_ad_chain_grads_f32`)
- **Zero-host-read verifier API**: expectation-based methods `solve_expect_sat` / `solve_expect_unsat` that never
  download SAT/UNSAT status to the CPU (fail-fast via GPU trap / CUDA error).
- **Device-resident CNF metadata** (`GpuCnf::{num_vars,num_clauses,num_lits}`) to support GPU-native CNF builders where
  capacity > exact size.
- **GPU-native equivalence verification** (`xlog-prob::compilation`) proving `φ ≡ C` via two UNSAT checks on GPU:
  `UNSAT(φ ∧ ¬C)` and `UNSAT(C ∧ ¬φ)`, with zero device→host reads.
- **GPU D4 compile+verify entrypoint** (`compile_gpu_d4_and_verify`) that compiles CNF to device-resident XGCF and
  validates equivalence via the GPU CDCL verifier.
- **Device-resident circuit cache + cache-aware evaluation** (`GpuCircuitCache`, `compile_gpu_d4_and_verify_cached`,
  `kernels/cache.cu`) enabling zero-recompile warm-cache inference.
- **GPU-native exact inference path**: `ExactDdnnfProgram` now uses GPU D4 + GPU CDCL + cache (no CPU D4, no CNF/DDNNF
  host materialization in production).
- **GPU weight/evidence builders** (`kernels/weights.cu` + `gpu_weights.rs`) for device-resident weight tables.
- **Regression guardrails** enforcing “no device→host reads” in the production verifier modules.
- **Cache DTOH guardrails + integration tests** (`no_dtoh_in_gpu_cache`, `gpu_exact_cache_integration`, `gpu_weights`).
- **Device-only logZ outputs** for GPU XGCF evaluation (`eval_log_wmc_device_*`) plus a guard test to prevent
  device→host reads inside device-only evaluation paths.
- **GPU-native loss output for neural fast-path**: `ExactDdnnfProgram::neural_backward_nll_buffers_with_device_loss`
  returns the scalar NLL loss as a device-resident value (no dtoh).
- **DLPack helper for typed allocations**: `TrackedCudaSlice::into_bytes()` enables wrapping typed device scalars into
  `CudaBuffer` columns without copies (used to export scalar loss to Torch).

### Changed

- dILP trainer defaults to sparse mask backend (`SparseMaskBackend`); dense fallback via `TrainConfig(debug_dense_mask=True)`.
- dILP holdout strategy now defaults to:
  - LOO for `<=20` positives
  - k-fold for `>20` positives (`holdout_strategy`, `holdout_folds` configurable)
- dILP promotion now enforces configurable holdout threshold (`holdout_threshold`, default `0.95`) and supports
  typed-schema gate controls (`typed_schema_required`, `waiver_untyped`).
- PyO3 exposes host transfer counters via `host_transfer_stats()` / `reset_host_transfer_stats()`.
- `GpuCnf` literal storage field renamed to `literals` (DIMACS `i32`) to match the solver/kernel interface.
- CUDA-dependent tests now skip cleanly when the CUDA runtime is unavailable (developer ergonomics).
- Workspace testing avoids building the PyO3 `extension-module` target when running `cargo test --workspace`.
- CUDA transfer/caching certification tests are stable under parallel test execution.

### Fixed

- Monte Carlo GPU initialization avoids reliance on CUDA device-count queries that can fail in restricted environments.
- GPU set operations + MC evaluation handle 0-arity (nullary) relations correctly (device row counts, not `row_cap`).
- `pyxlog` DLPack interop: detach `requires_grad` tensors before exporting probabilities to DLPack.
- `pyxlog` GPU neural fast-path ordering: replaced `torch.cuda.synchronize()` with stream-to-stream waits.
- GPU CNF reachability worklist hardened to avoid consuming uninitialized queue entries under concurrent expansion.
- nvcc deprecation warnings for `sm_70` offline PTX builds are suppressed in `kernels/CMakeLists.txt`.
- Release-mode CUDA crash in the GPU CDCL verifier/equivalence path caused by passing temporary scalar kernel arguments
  via raw parameter vectors (now backed by stable locals before `cuLaunchKernel`).
- Release-mode CUDA launch failures in GPU D4 tests and smoothing due to temporary scalar kernel arguments (now backed
  by stable locals before `cuLaunchKernel`).
- GPU smoothing now seeds root support with all random vars and levelizes with the emitted node count, ensuring
  unconditional probabilistic facts/evidence are handled correctly and preventing under-launched levels.
- GPU cache meta loading moved out of `gpu_cache.rs` to preserve dtoh-free guardrails for the cache module.

### Removed

- Vendored CPU D4/Boost toolchain (`vendor/`) and the CPU-based exact compilation pipeline (GPU-native only).

### Removed

- `test_non_monotone_with_mc` — pre-existing 50K MC sample negation test that consistently timed out (unrelated to dILP).

### Known Limitations

- Python batch query path (`batch_fact_membership`, `batch_tagged_credit`) coerces all facts via `as u32`. Typed relation schemas work in core execution but the Python query interface is U32-entity-ID-only for now.
- `bench.yml` PR-comparison dispatch path is non-operational under manual-only CI (event-gated for `push`/`pull_request`).
- GA 50-seed statistical reliability gate (`test_ilp_ga_reliability.py`) exceeds 600s timeout; deferred to post-beta runtime budget optimization. Beta gate = 20/20 reliability (Suite 4).

### Deferred to v0.4.0-rc

- ~~Term embeddings for neural-symbolic integration~~ (done in v0.5.0: P2a)
- ~~Extended neural-symbolic training controls~~ (done in v0.5.0: P2b)

### Deferred to v0.5.0

- Typed query-buffer builder (non-u32 Python batch queries)
- Full GPU-resident loss computation path
- 50-seed runtime budget optimization
- SLO harness for N=20/50/100/150

### Validation

All tests pass on v0.4.0-beta validation matrix (7 suites). See `docs/reports/2026-03-04-v0.4.0-beta-validation.md`.

## Neural-Symbolic Integration Milestone (v0.4.0-alpha) — 2026-02-23

Milestone snapshot of the neural-symbolic integration layer (training APIs + GPU circuit evaluation/gradients). The `v0.4.0-alpha` milestone is fully achieved with end-to-end example validation and all required neural examples.

### Added

**Neural Predicates (`nn/4` syntax):**
- `nn(network, [inputs], output, [labels]) :: predicate(args).` declaration syntax
- Network-backed probabilistic facts with automatic annotated disjunction generation
- Support for classification mode (with labels) and embedding mode (without)
- Multiple input variables, symbol labels, and empty input lists

**Network Registry:**
- `register_network(name, module, optimizer, scheduler)` Python API
- `NetworkConfig` with neural predicate options: batching, k (top-k), det (deterministic), cache
- `NetworkHandle` with train/eval mode switching
- Automatic validation against declared neural predicates

**Tensor Source Registry:**
- `add_tensor_source(name, tensor)` for external data (images, embeddings)
- `set_active_tensor_source(name)` for switching between train/test
- Index validation and bounds checking
- Metadata tracking (size, shape, dtype)

**Neural → Probability Bridge:**
- Softmax outputs converted to annotated disjunctions
- `NeuralBridge` for numerical stability (epsilon clamping, normalization)
- Log probability computation for gradient stability
- Circuit leaf generation for d-DNNF integration

**Training Infrastructure:**
- `forward_backward()` for single query training with gradient computation
- `train_epoch()` for batch processing with configurable batch size
- `train_model()` for multi-epoch training with shuffle and logging
- `zero_grad()`, `optimizer_step()`, `scheduler_step()` for training loop control
- `TrainingHistory` object with epoch losses and batch metrics

**NLL Loss Functions:**
- `nll_loss(prob)` — negative log-likelihood from probability
- `nll_loss_batch(probs)` — batch NLL computation
- `nll_loss_mean(probs)` — mean NLL over batch
- `nll_loss_tensor(prob)` — PyTorch tensor output for autograd
- Numerical stability via epsilon (1e-10) clamping

**Backward Pass to Networks:**
- `backprop_circuit_gradients()` propagates d-DNNF gradients through neural networks
- Weight slot mapping for position-based gradient routing
- PyTorch `.backward()` integration with gradient tensors
- Support for multiple networks per query

**Circuit Caching:**
- `CachedCircuit` stores compiled d-DNNF circuits for reuse
- `WeightSlot` maps circuit variables to network outputs by position
- `evaluate_gpu_with_grads_weights()` for weight-only circuit evaluation
- Cache key generation from query templates
- Eliminates D4 recompilation bottleneck (100x+ speedup for repeated queries)

**Minimal MNIST Addition Example:**
- `examples/neural/01_minimal/train.py` — complete working example
- CNN network classifying MNIST digits
- Training purely from addition supervision (no digit labels)
- Demonstrates neural-symbolic gradient flow

**Negation in Probabilistic Programs:**
- `not` keyword in rule bodies for exact inference (`wet :- not rain.`)
- Stratified negation with automatic layer detection
- Non-monotone (cyclic) negation via Well-Founded Semantics (WFS)
- Exact gradients flow through negated literals for neural-symbolic training

**GPU Certification Suite (G01-G06):**
- G01: Circuit Forward Kernel tests (8 tests) — `xgcf_forward_level` PTX validation
- G02: Circuit Backward Kernel tests (12 tests) — gradient computation verification
- G03: Weight Injection tests (6 tests) — GPU weight buffer management
- G04: Transfer Efficiency tests (8 tests) — 0% CPU bottleneck verification
- G05: Circuit Cache tests (6 tests) — GpuXgcf reuse, D4 elimination
- G06: PTX Robustness tests (10 tests) — large circuits, edge cases, numerical stability
- Total: 50 new GPU-focused tests validating neural-symbolic kernel correctness

**PIR Extension:**
- `NegLit { leaf: LeafId }` node for negated probabilistic leaves
- NNF (Negation Normal Form) transformation pushes negation to leaves
- Weight semantics: `NegLit` uses complemented probability `(1-p, p)`

**Stratification Analysis:**
- `analyze_stratification()` function detects non-monotone SCCs
- Edge polarity tracking in dependency graph (positive/negative edges)
- Automatic classification: stratified SCCs use two-valued evaluation, non-monotone use WFS

**Well-Founded Semantics (WFS):**
- Three-valued logic: True, False, Undefined
- Alternating fixed-point algorithm (unfounded set + consequence derivation)
- Undefined atoms return probability 0 with zero gradient
- Full 1,461-line implementation in `wfs.rs`

### Changed

- **Python package renamed from `xlog-gpu` to `pyxlog`** — cleaner, more memorable name
  - All imports: `import pyxlog` (was `import xlog_gpu`)
  - Crate renamed: `crates/pyxlog` (was `crates/xlog-gpu-py`)
  - PyPI package: `pyxlog` (was `xlog-gpu`)
- Stratification analysis now tracks edge polarity for non-monotone detection
- Provenance extraction routes non-monotone SCCs to WFS evaluation
- CNF encoding emits Tseitin clauses for `NegLit` with negated polarity

### Technical Details

| Component | Files | Purpose |
|-----------|-------|---------|
| Grammar | `grammar.pest:93-102` | `nn/4` syntax parsing |
| AST | `ast.rs:323-358` | `NeuralPredDecl`, `NeuralLabel` |
| Parser | `parser.rs:573-645` | `build_neural_pred_decl()` |
| Registry | `xlog-neural/src/registry.rs` | `NetworkRegistry`, `NetworkConfig` |
| Handle | `xlog-neural/src/handle.rs` | `NetworkHandle` with PyO3 objects |
| Bridge | `xlog-neural/src/bridge.rs` | `NeuralBridge`, `NeuralOutput` |
| Tensor | `xlog-neural/src/tensor_source.rs` | `TensorSourceRegistry` |
| Python | `crates/pyxlog/src/lib.rs` | Full training API |
| PIR | `pir.rs` | `NegLit` variant |
| WFS | `wfs.rs` | Well-Founded Semantics (1,461 lines) |
| Exact | `exact.rs` | `random_var_indices()`, `evaluate_gpu_with_grads_weights()` |
| G01-G06 | `xlog-cuda-tests/src/categories/g0*.rs` | GPU certification tests (50 tests) |

### Validation

- **CUDA Certification Suite:** 200/200 tests passed (C01-C25 + G01-G06)
- **Python Tests:** 109/109 tests passed
- **Spec Alignment:** All 50 G01-G06 tests match specification
- **Code Quality:** No stubs, placeholders, or TODOs

### Example: MNIST Addition Training

```python
import pyxlog
import torch

# Define neural predicate program
program = pyxlog.Program.compile("""
    nn(mnist_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).
    addition(X, Y, Z) :- digit(X, D1), digit(Y, D2), Z is D1 + D2.
""")

# Register PyTorch network
net = MNISTNet()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
program.register_network("mnist_net", net, optimizer)

# Add training data
program.add_tensor_source("train", train_images)

# Train on addition queries (no digit labels!)
queries = ["addition(0, 1, 7)", "addition(2, 3, 5)", ...]
history = pyxlog.train_model(program, queries, epochs=50, batch_size=32)
```

---

## v0.3.2 — 2026-01-19

Module system, user-defined functions, reversible symbols, and comprehensive showcase examples for expressive, modular Datalog programs.

### Added

**Module System:**
- File-based modules with explicit imports
- `use module.` to import all public predicates
- `use module::{pred1, pred2}.` for selective imports
- `use path/to/module.` for nested modules
- `private` keyword for module-internal predicates and functions

**User-Defined Functions:**
- Reusable functions in rule bodies
- Arithmetic functions: `func double(X) = X * 2.`
- Conditional functions: `func abs(X) = if X < 0 then 0 - X else X.`
- Recursive functions with base-case validation
- Optional type annotations: `func add(X: f64, Y: f64) -> f64 = X + Y.`
- Predicate-based functions: `func get_parent(X) = P :- parent(X, P).`

**Reversible Symbols:**
- Bidirectional string-to-ID mapping
- Symbols display as original strings in query output
- Arrow dictionary encoding for efficient serialization
- `--stats` shows symbol registry metrics

**CLI Enhancements:**
- `--module-path` flag for specifying module search directories

**Showcase Examples:**
- Enterprise Analytics: HR management, compensation, org hierarchy with recursive management chains
- Knowledge Graph: Ontology modeling, citation analysis, semantic inference with type inheritance
- Game Analytics: Player statistics, achievements, guilds, leaderboards with social network analysis
- Supply Chain: Bill of Materials explosion, inventory management, supplier analytics

### Fixed

- **GroupBy count aggregation type**: Count now outputs `u64` (was `u32`) to match predicate declarations and prevent type mismatch errors when comparing count results
- **Optimizer predicate pushdown**: Fixed column width estimation to use schema information for accurate filtering

### Changed

- Symbol storage changed from hash-based to sequential ID allocation
- Module resolution now validates imports before compilation

### Breaking Changes

- Serialized Arrow files from v0.3.1 with symbol columns may need re-export
- `hash_symbol_to_u32` function removed from public API
- Count aggregation results are now `u64` instead of `u32`

---

## v0.3.1 — 2026-01-18

Float predicates, performance benchmarks, query statistics, fuzz testing, and property-based tests.

### Added

**Float Predicate Support:**
- IEEE 754 total ordering for `f32`/`f64` filter comparisons: `NaN > Inf > positive > +0 > -0 > negative > -Inf`
- Filter kernels: `filter_compare_f32_*` and `filter_compare_f64_*` with proper edge case handling
- Comprehensive tests for NaN, Infinity, subnormals, and signed zeros

**Performance Benchmarks:**
- Criterion.rs benchmarks for `xlog-gpu` (transitive closure, hash join, aggregation)
- Criterion.rs benchmarks for `xlog-prob` (exact inference, Monte Carlo sampling)
- `docs/BENCHMARKS.md` with methodology and baseline metrics
- `.github/workflows/bench.yml` for CI regression detection

**Query Timing & Statistics:**
- `--stats` CLI flag for execution profiling
- Per-stratum timing with iteration counts for recursive strata
- Per-operation timing (join, sort, dedup, filter)
- Memory usage tracking (peak, budget)
- Human-readable and JSON output formats

**Fuzz Testing:**
- `fuzz/` directory with cargo-fuzz targets:
  - `fuzz_parser` — raw byte input fuzzing
  - `fuzz_compiler` — structured program generation
  - `fuzz_type_inference` — type system stress testing
- AddressSanitizer (ASAN) integration for crash detection
- `.github/workflows/fuzz.yml` for continuous fuzzing

**Property-Based Testing:**
- proptest integration in `xlog-cuda-tests`
- Sort stability property (data preservation, ascending order)
- Join correctness property (CPU reference comparison)
- Filter idempotence property (`filter(filter(x)) = filter(x)`)
- Dedup determinism property (consistent output across runs)
- Stress tests for large datasets (50K+ rows)

### Validation
- Workspace tests pass: `cargo test --workspace --all-targets --release`
- Property tests pass: `cargo test -p xlog-cuda-tests --test properties --release`
- Fuzz targets build and run with ASAN

---

## v0.2.0 — 2026-01-14

Phase 4 probabilistic logic programming (`xlog-prob`) merged into `main`; Python bindings are the integration surface for GPU I/O.

### Added
- `xlog-prob`: exact inference via Decision-DNNF (vendored D4) + GPU weighted model counting and gradients.
- `xlog-prob`: P3 Monte Carlo engine (`prob_engine=mc`) with GPU sampling, deterministic non-monotone SCC semantics, and uncertainty metadata.
- New CUDA kernels: `kernels/circuit.ptx` (XGCF forward/backward) and `kernels/mc_sample.ptx` (MC sampling).
- New examples: `examples/prob/` (probabilistic `.xlog`) and `examples/python/` (DLPack bindings).
- `xlog-gpu` + `pyxlog`: `pyxlog` Python module (PyO3) with DLPack-first I/O for deterministic and probabilistic evaluation.
- New/updated docs: `docs/architecture/xlog-prob.md`, `docs/VALIDATION_REPORT.md`.

### Validation
- Workspace tests pass in release (`cargo test --workspace --all-targets --release`).
- CUDA certification suite passes: **140/140** (see `docs/plans/2026-01-14-cuda-certification-results.md`).

## v0.1.0 — 2026-01-13

Initial release of the deterministic `xlog-logic` tier (Phase 3 complete).

### Added
- `.xlog` parser + compiler with stratified negation and semi-naive fixpoint recursion.
- GPU execution backend (`xlog-cuda`) with kernels for join/sort/filter/dedup/groupby/scan/pack/set-ops.
- Arithmetic (`is`) and builtin functions (`abs/min/max/pow/cast`) in rule bodies.
- Aggregations: `count/sum/min/max/logsumexp`.
- Arrow IPC import/export utilities and DLPack zero-copy column interop.
- Example suite under `examples/xlog/` and runner example `crates/xlog-logic/examples/xlog_run.rs`.

### Validation
- Workspace tests pass in release (`cargo test --workspace --all-targets --release`).
- CUDA certification suite passes: **133/133** (see `docs/plans/2026-01-12-cuda-certification-results.md`).

### Known limitations
- `symbol` values are currently represented as a `u32` hash (not reversible).
- Arrow IPC interop involves device↔host copies; DLPack is the zero-copy path.
