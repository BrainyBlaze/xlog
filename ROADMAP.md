# XLOG Development Roadmap

Last updated: May 21, 2026
Current tagged release: v0.8.6. v0.6.0 shipped the stream-safe runtime
and recorded launch discipline. v0.6.1 shipped recorded CSM hash-join
dispatch and explicit CSM cert-mode labeling. v0.6.2 shipped the first
productized WCOJ slice: hypergraph planner / oracle foundations plus
default-on adaptive GPU triangle WCOJ for `U32`, `Symbol`, and `U64`
inputs. v0.7.0 ships the completed WCOJ expansion pack: first-class
multiway RIR, WCOJ cost and variable-ordering models, recursive/SCC
integration, K-clique production planning, K5-K8 CUDA coverage, CUDA
Graph hot-loop support, and DTS-DLM end-to-end validation. v0.8.0 ships
the DTS-DLM ML/Python productization pack: stable `pyxlog` runtime/session
contracts, async and streaming evaluation, relation deltas, diagnostics,
registered neural top-k/deterministic modes, Belnap bridge helpers, native
exact-induction integration, and DTS-focused examples. v0.8.5 ships the
language-completeness pack: finite typed lists and terms, safe meta-predicates,
deterministic NAF, magic-set planning, probabilistic aggregate inference with
GPU-native count-lift exact evaluation, approximate inference controls,
incremental parsing, and explain/REPL/watch developer workflows. v0.8.6 ships
the DTS-DLM runtime completion and GPU-native optimizer pack: device-resident
delta coalescing, relation-change callbacks, typed exact induction,
profile-gated chain shared-memory scoring, runtime CSE, adaptive
re-optimization, persistent hash-index reuse, and behavior-probe-backed
consumer certification. The v0.9.0 release-candidate branch now integrates the
predecessor BFO-derived diagnostic packs: Project 1/v0.8.7 generated-rule and
biomedical graph diagnostics, v0.8.8 living-world provenance refinements, and
the Universal Case Reasoner diagnostic pack with joint `nn/4` plus symbolic
rule-weight training, differentiable proof traces, learned-rule inventories,
CUDA host-transfer audits, module-boundary diagnostics, grouped transfer
metrics, and the BFO UCR validation package.

This roadmap is version-oriented so planned work is not hidden inside subsystem
sections. Historical and current-main work uses checked boxes. Future work uses
unchecked boxes and is assigned to a concrete future version.
After the tagged v0.8.0 feature pack, v0.8.5 completed the Language
Completeness and Developer Experience train. v0.8.6 closed the deferred
DTS-DLM runtime / GPU-native optimizer completion backlog that v0.9.0 needs as
runtime substrate. v0.8.7-v0.8.9 are integrated predecessor diagnostics
surfaces in the v0.9.0 Epistemic/Solver Semantics train. v0.9.1 completes the
bounded epistemic executor into a load-bearing surface (EIR-derived candidate
enumeration, value-level modal membership, per-tuple-key FAEEL foundedness,
epistemic constraints, safe split equivalence, and joint multi-epistemic
solving), and v0.10.0 is the Multi-GPU / Out-of-Core train.

## v0.0.1 - Workspace Foundation

### Repository

- [x] Rust workspace foundation for the core, IR, CUDA, runtime, Python, and CLI crates.
- [x] CUDA kernel source layout and build integration.
- [x] Baseline examples, tests, and developer documentation structure.

### Build and Packaging

- [x] Cargo-based crate build flow.
- [x] Python extension build path through `pyxlog`.
- [x] CLI binary crate foundation.

## v0.1.0 - Deterministic Datalog and GPU Execution

### xlog-logic

- [x] Datalog parsing via PEG grammar with Pest.
- [x] Query syntax with `?- atom.`.
- [x] Constraint syntax with `:- body.`.
- [x] Stratified negation analysis with SCC-based ordering.
- [x] Recursive rule support.
- [x] Comparison operators in rule bodies.
- [x] Arithmetic expressions via `is`.
- [x] Wildcard variables.
- [x] Predicate declarations with type annotations.
- [x] Symbol type support for string values.

### xlog-runtime

- [x] Stratum-ordered execution.
- [x] SCC-aware recursive evaluation.
- [x] Semi-naive delta evaluation.
- [x] Per-rule delta rewriting.
- [x] Configurable iteration limits.
- [x] Versioned relation storage with update tracking.
- [x] Profiling hooks.

### xlog-ir and Optimizer

- [x] Predicate pushdown.
- [x] Cost-based join planning with dynamic programming for up to 10 atoms.
- [x] Greedy bushy join planning fallback.
- [x] Build/probe cost model.
- [x] Statistics-seeded optimization through `StatsSnapshot`.
- [x] Cartesian join support.

### xlog-cuda

- [x] Inner hash join with collision safety.
- [x] Semi join.
- [x] Anti join.
- [x] Left-outer join.
- [x] Bucketed CSR hash layout.
- [x] 64-bit composite FNV-1a hashing.
- [x] Optional unsafe hash-only mode.
- [x] Typed comparison kernels for all scalar types.
- [x] Float predicate total ordering.
- [x] Boolean mask composition.
- [x] Stream compaction without host round-trips.
- [x] Multi-block prefix scan.

### xlog-cli

- [x] `xlog run`.
- [x] Arrow IPC input.
- [x] Multiple output formats.
- [x] Device selection.
- [x] Memory limit configuration.
- [x] Query timing and statistics output.
- [x] Human-readable and JSON statistics formats.

### pyxlog

- [x] PyO3 Python extension foundation.
- [x] `LogicProgram.compile` for deterministic Datalog.
- [x] DLPack input/output bindings.
- [x] DLPack roundtrip coverage.

## v0.2.0 - Adaptive Indexing, Aggregation, and Interop

### xlog-logic

- [x] Built-in functions: `abs`, `min`, `max`, `pow`, and `cast`.
- [x] Aggregation operators: `count`, `sum`, `min`, `max`, and `logsumexp`.

### xlog-runtime

- [x] GPU-resident filter evaluation through mask DAGs.
- [x] GPU-resident arithmetic evaluation.
- [x] GPU-resident groupby finalization.
- [x] Semi-naive delta application API.
- [x] Insert-only incremental updates for monotone SCCs.
- [x] Full recomputation for non-monotone SCCs and dependents.
- [x] Delta propagation through dependent strata.

### Adaptive Indexing

- [x] Per-relation heat tracking.
- [x] Cardinality and byte-size statistics.
- [x] Join selectivity observation.
- [x] Join index cache with LRU eviction.
- [x] Index invalidation on updates.
- [x] Budget-aware index sizing heuristics.
- [x] Build-side hash reuse for hot scan relations.

### xlog-cuda

- [x] Stable 4-bit radix sort for all scalar types.
- [x] Multi-column lexicographic key support.
- [x] GPU-resident permutation generation and application.
- [x] Precomputed per-digit per-block offsets through GPU prefix sums.
- [x] Integer `count`, `sum`, `min`, and `max` aggregation.
- [x] Floating-point `logsumexp` aggregation.
- [x] Multi-key groupby with packed keys.
- [x] GPU boundary detection and group-id assignment.
- [x] Union with deduplication.
- [x] Set difference through sorted binary-search marking.
- [x] Set operations for all scalar types and multi-column schemas.
- [x] Arithmetic kernels: add, subtract, multiply, divide, modulo.
- [x] Arithmetic kernels: absolute value and negation.
- [x] Arithmetic kernels: min, max, pow, and cast.
- [x] Type promotion and casting.

### Memory Management

- [x] Atomic budget reservation.
- [x] RAII memory tracking.
- [x] Configurable memory limits.
- [x] Column-major storage with schema metadata.

### Data Interoperability

- [x] Arrow `RecordBatch` export through device-to-host transfer.
- [x] Arrow `RecordBatch` import.
- [x] Arrow IPC stream read and write.
- [x] DLPack column export.
- [x] DLPack import with schema inference.
- [x] DLPack import with schema validation.

### CUDA Kernel Modules

- [x] `join.cu`: hash join build/probe v2, bucket layout, semi/anti joins, composite hashing.
- [x] `filter.cu`: typed comparisons, mask composition, and stream compaction.
- [x] `sort.cu`: radix histogram, scatter, and permutation application.
- [x] `dedup.cu`: duplicate marking and row compaction.
- [x] `groupby.cu`: boundary detection, key extraction, and aggregation.
- [x] `scan.cu`: exclusive prefix sum and multi-block scan.
- [x] `pack.cu`: key packing, hashing, and packed-row gather.
- [x] `set_ops.cu`: concatenation and sorted difference marking.

## v0.3.0 - Probabilistic Reasoning

### xlog-logic

- [x] Probabilistic facts.
- [x] Annotated disjunctions.
- [x] Evidence declarations.
- [x] Probabilistic queries.

### xlog-prob

- [x] Provenance extraction from positive Datalog.
- [x] PIR graph construction.
- [x] Tseitin CNF with stable variable mapping.
- [x] GPU D4 integration.
- [x] GPU CDCL equivalence verifier.
- [x] Decision-DNNF parsing retained for tests and fixtures.
- [x] XGCF construction.
- [x] GPU forward pass weighted model counting.
- [x] GPU backward pass gradients.
- [x] Conditional probability `P(Query|Evidence)`.
- [x] Per-query gradient output.
- [x] Bernoulli sampling plan compilation.
- [x] GPU Bernoulli matrix sampling.
- [x] Deterministic per-world evaluation.
- [x] Rejection sampling for evidence conditioning.
- [x] Uncertainty reporting.
- [x] Non-monotone SCC handling with skeptical semantics.
- [x] Configurable sample count and seed.
- [x] Evidence clamping for importance-sampling-style execution.

### Negation and Provenance

- [x] Exact negation through NNF/WFS.
- [x] Stratified negation layer detection.
- [x] Non-monotone cyclic negation through WFS.
- [x] Gradient flow through negated literals.
- [x] `NegLit` PIR node.
- [x] Stratification edge polarity tracking.
- [x] `ChoiceSource`.
- [x] `Provenance::leaf_atom`.
- [x] `Provenance::choice_source`.
- [x] `Provenance::atoms_with_formulas`.
- [x] `GroundAtom::new`.
- [x] Inline `leaf_atoms` and `choice_sources`.
- [x] Top-level provenance re-exports.

### Solver and Knowledge-Compilation Kernels

- [x] `circuit.cu`: XGCF forward and backward passes.
- [x] `cache.cu`: GPU circuit cache.
- [x] `cnf.cu`: GPU PIR-to-CNF encoding.
- [x] `d4.cu`: GPU D4 compilation.
- [x] `sat.cu`: GPU CDCL SAT solver and verifier helpers.
- [x] `mc_sample.cu`: Bernoulli sampling.
- [x] `weights.cu`: GPU weight and evidence builders.

### xlog-cli

- [x] `xlog prob`.
- [x] Exact-DDNNF and MC engine selection with `--prob-engine exact_ddnnf|mc`.
- [x] MC options: `--samples`, `--seed`, and `--confidence`.

### pyxlog

- [x] `Program.compile` for probabilistic programs.
- [x] Exact-DDNNF and MC engine selection from Python.
- [x] Gradient output bindings.
- [x] MC uncertainty bindings.

## v0.3.1 - GPU Native Knowledge Compilation and Solver Services

### xlog-prob

- [x] GPU CDCL equivalence verifier with zero host reads.
- [x] Device-resident CNF metadata.
- [x] GPU PIR-to-CNF encoder through `encode_cnf_gpu` with device-resident CSR emission and deterministic variable numbering.
- [x] GPU circuit-to-CNF encoding.
- [x] GPU D4 core.
- [x] GPU D4 compile and verify flow.
- [x] Device-resident circuit cache.
- [x] CPU D4 invocation replaced by GPU-native path.
- [x] GPU smoothing seeds with root support.
- [x] CUDA certification for SAT/CDCL and device counts.

### Solver Services

- [x] Clause and literal representation.
- [x] GPU CDCL verifier.
- [x] Expectation API with zero D2H transfers.
- [x] GPU-native equivalence query construction.
- [x] CLS heuristic.

## v0.3.2 - Modules, Symbols, and User Functions

### xlog-logic

- [x] Reversible symbol values.
- [x] User-defined functions.
- [x] Module system with `use` imports.
- [x] Private module visibility.

### Data Interoperability

- [x] Arrow C Data Interface device export.
- [x] Python DLPack capsule interface.
- [x] DLPack column ownership tracking.

## v0.4.0-alpha - Neural-Symbolic Foundation

### Neural Predicates

- [x] `nn/4` syntax.
- [x] Network registry.
- [x] Tensor source registry.
- [x] Neural output conversion to annotated disjunctions.
- [x] Deterministic and non-deterministic neural modes.

### Training

- [x] PyTorch autograd integration.
- [x] `register_network`.
- [x] `add_tensor_source`.
- [x] `set_active_tensor_source`.
- [x] `train_model`.
- [x] Negative log-likelihood loss.
- [x] `nll_loss`, `nll_loss_batch`, and `nll_loss_tensor`.
- [x] `forward_backward`.
- [x] `forward_backward_tensor` for strict GPU-native training with CUDA tensor loss and no host reads.
- [x] `train_epoch`.
- [x] Circuit caching.
- [x] GPU neural fast-path for cached circuits with device-side AD weight fill, chain-rule gradients, and DLPack interop.
- [x] Python `zero_grad`, `optimizer_step`, and `scheduler_step`.
- [x] Python `TrainingHistory`.

### Examples

- [x] Validate all examples end-to-end.
- [x] Add examples beyond the minimal smoke case.
- [x] Minimal MNIST example.
- [x] Coins example with two coin classifiers.
- [x] Multi-digit SVHN example.
- [x] Handwritten Formula (HWF) example.
- [x] Poker example.
- [x] CLUTRR example.

### CUDA Kernel Modules

- [x] `neural.cu`: neural fast-path support.

## v0.4.0-beta - Differentiable ILP Beta

### Sparse Mask API

- [x] `set_rule_mask_sparse(candidate_ids, soft_probs, budget)` with Rust-side executor mask construction.
- [x] Training without N3 tensor materialization.
- [x] `AtomicU32` row-count cache.

### Trainer Backend

- [x] `MaskBackend` protocol.
- [x] `debug_dense_mask`.
- [x] Dense parity checks.

### Training Pipeline

- [x] `train_only`.
- [x] `train_and_promote`.
- [x] Promotion gates for convergence, novel-rate audit, regression check, holdout F1, ambiguity scan, and typed schema.
- [x] Transactional promotion commit.

### pyxlog.ilp

- [x] `pyxlog.ilp.train_only`.
- [x] `pyxlog.ilp.train_and_promote`.
- [x] `TrainConfig` with temperature, budget, holdout, recursion, determinism, and typed-schema gates.
- [x] `TrainResult`.
- [x] `PromotionResult`.
- [x] `LearnedArtifact`.
- [x] `IlpProgramFactory.compile`.
- [x] `valid_candidates` for recursive and non-recursive candidate enumeration.

### Holdout and Ambiguity

- [x] Leave-one-out holdout F1.
- [x] k-fold holdout.
- [x] Deterministic per-fold assignment.
- [x] Top-M ambiguity scan.

### Hard-Negative Mining

- [x] `sample_false_positives`.
- [x] Hard-negative mining wired every 20 steps.
- [x] D2H counter reset for hard-negative mining.

### Artifact Persistence

- [x] `LearnedArtifact` save/load.
- [x] SHA-256 candidate-map hash.
- [x] Artifact schema `beta-v1`.

### Recursive Candidates

- [x] `allow_recursive_candidates`.
- [x] Recursive candidates default off.

### Reliability

- [x] Beta reliability gate at 20/20 across reach, grandparent, colleague, and plus2 stages.
- [x] Zero D2H column transfers in the training loop.

## v0.4.0-ga - Differentiable ILP General Availability

### Reliability and Determinism

- [x] Deterministic dILP mode.
- [x] Persisted `selected_hard`.
- [x] Holdout-threshold gate.
- [x] Typed-schema gate with waiver-based manual-review fallback.
- [x] Host-transfer telemetry.
- [x] `forward_p95_us` telemetry.
- [x] General-availability reliability statistical gate.
- [x] General-availability performance and transfer smoke test.
- [x] General-availability runtime optimization from 1447 seconds to 436 seconds.

### Completed General-Availability Hardening

- [x] GPU-resident loss computation through `compute_ilp_loss_grad_gpu`, strict zero-D2H gates, and four CUDA kernels.
- [x] Training config restoration on failure.
- [x] Training telemetry persistence with optional size-bounded artifact snapshots.
- [x] Typed query-buffer builder for I32, I64, U64, Bool, and Symbol schemas.
- [x] CI-grade 50-seed general-availability reliability runtime budget optimization.
- [x] Full service-level-objective benchmark harness across 20, 50, 100, and 150 example workloads.

## v0.5.0 - Public-Release Hardening and Bounded GPU ILP

### Bounded Memory dILP

- [x] Two-pass bounded-memory GPU-only chunk merge.
- [x] `coo_chunk_budget` renamed from `coo_memory_cap` with deprecated alias retained.
- [x] `count_mask_into_slot`.
- [x] `dtoh_scalar_untracked`.
- [x] Strict zero-D2H forced chunking.
- [x] Artifact schema `beta-v1` to `beta-v2`.
- [x] Bounded telemetry persistence.

### Term Embeddings and Neural Forms

- [x] `EmbeddingHandle` in `xlog-neural`, parallel to `NetworkHandle`.
- [x] Network-registry embedding storage.
- [x] By-network form index.
- [x] Compile-time mixed-form rejection.
- [x] `register_embedding`.
- [x] `forward_embedding`.
- [x] Raw tensor detach.
- [x] Cross-registration guard.
- [x] Term-embedding tests covering shape, values, frozen tensor behavior, cross-registration, gradient flow, detach, and mixed forms.
- [x] Learnable `nn.Embedding` and pretrained frozen tensor embedding modes.
- [x] Python `register_embedding`.
- [x] Python `forward_embedding`.
- [x] Stop-condition support.
- [x] Learning-rate scheduler support.

### Training Controls

- [x] Gradient clipping.
- [x] Early stopping.
- [x] Scheduler step.
- [x] `get_lr` and `set_lr`.
- [x] `TrainingHistory.stopped_early`.

### GPU CDCL Workspace

- [x] `GpuCdclWorkspace` with 29 buffers.
- [x] New workspace constructor.
- [x] Four workspace methods.
- [x] Incremental verification opt-in.
- [x] `check_equivalence` integration.

### xlog-cli and Packaging

- [x] Published CLI crate is named `xlog`.
- [x] PyPI package distribution readiness.

## v0.5.1 - Bounded Exact Induction

### Exact Induction API

- [x] `InduceExactRequest`.
- [x] `ExactInductionResult`.
- [x] `ScoredCandidate`.
- [x] `induce_exact`.
- [x] `validate::classify_request` for empty-candidate and zero-positive dead ends without CUDA.
- [x] Arity-2 `U64` cached count validation.
- [x] Deterministic `reduce_per_topology`.

### CUDA Exact Induction

- [x] Single `ilp_exact_score` kernel entry with `(C, C, 4)` grid, 256-thread blocks, unique topology slots, and zero cross-block atomics.
- [x] Deterministic block reduction.
- [x] Four topology templates.
- [x] `CudaKernelProvider::ilp_exact_score` launcher with D2D candidate-column concatenation, candidate-offset upload, kernel launch, and two count-array downloads.
- [x] D2H budget of 2 transfers.
- [x] CUDA-gated exact-induction tests for hand-computed coverage, determinism across runs, and empty-negatives handling.

### pyxlog

- [x] `induce_exact_native`.
- [x] Name-to-`RelId` mapping.
- [x] `strict_per_topology` flag preserving historical Python backend numbers by default while enabling parity-isolated scoring.

### Kernel Manifest

- [x] `KERNEL_MODULES` expanded from 21 to 22.
- [x] `ILP_EXACT_MODULE` constants.

### Parity and Evidence

- [x] Native exact induction matches Python parity with ordered equality across summary and candidate fields.
- [x] D2H count does not scale with input size.

## v0.5.2 - Release and Publication Readiness

### CI/CD and Release

- [x] Release-publish verification fixes.
- [x] Release-plz human-gated publish model.
- [x] Release PR creation tuned to avoid recursive release PR spam.
- [x] README badge and release status validation.
- [x] Crates.io and PyPI publication pipeline wiring.
- [x] GitHub organization links updated to `BrainyBlaze/xlog`.

### Documentation

- [x] Architecture guide.
- [x] Language reference at `docs/language-reference.md` for the v0.3.2 language surface.
- [x] Probabilistic tier design.
- [x] Adaptive indexing design.
- [x] Multi-GPU join design.
- [x] Data interoperability guide.
- [x] Examples.
- [x] CUDA certification report.
- [x] rustdoc coverage.

### Testing and Certification

- [x] Workspace test suite.
- [x] CUDA certification suite.
- [x] Hash-collision tests.
- [x] Aggregation overflow and truncation tests.
- [x] Large-input filter and compaction tests.
- [x] Memory-budget tests.
- [x] End-to-end Datalog tests.
- [x] Probabilistic inference tests.
- [x] GPU CDCL verifier tests.
- [x] GPU CDCL workspace tests.
- [x] Workspace equivalence parity tests.
- [x] Performance regression benchmarks.
- [x] Criterion-based benchmark workflow.
- [x] Parser, compiler, and type-inference fuzz tests.
- [x] Cargo-fuzz and ASAN fuzz workflow.
- [x] Property-based kernel correctness tests.
- [x] Proptest coverage for sort stability, join correctness, filter idempotence, and dedup determinism.
- [x] Float edge-case tests.
- [x] dILP beta test suite.
- [x] dILP general-availability reliability gate.
- [x] dILP performance and transfer smoke test.

## main after v0.5.2 - Determinism Closure

### xlog-runtime

- [x] Deterministic recursive SCC predicate processing by replacing randomized `HashSet` iteration with ordered processing.
- [x] Device logical row-count checks using `buffer_row_count()` instead of stale `is_empty()` state.
- [x] Downstream crash-window frozen replay test coverage for deterministic recursive evaluation.
- [x] `XLOG_DETERMINISM_RESTORED` validated against four frozen crash-window relation bundles with 20 fresh subprocess replays each.

### Build and Packaging

- [x] CUDA kernel staging made robust for source builds and wheel builds.
- [x] Build documentation updated for CUDA 13.x and WSL `/dev/dxg` device routing.
- [x] Python install guidance updated to require explicit interpreter selection for `maturin develop`.

### Whitepaper and Public Docs

- [x] Whitepaper corrected against current code architecture.
- [x] README aligned with the audit branch publication-ready narrative and current release/setup data.

## v0.5.5 - Consolidated Deterministic Hardening

Status: closed as deterministic hardening at PRs #49 (strict D2H guard), #50
(GPU full-row dedup / set-difference), #52 (binary-join output counts as
metadata reads), plus crash-window frozen replay evidence. Fully GPU-resident
binary-join materialization through count → prefix → materialize is deferred
to v0.6.x — see "v0.6.x Deferred From v0.5.5" below.

### xlog-runtime

- [x] Replace host-side multi-column full-row dedup/difference fallback with GPU-native deterministic set algebra. (#50)
- [x] Add strict D2H guardrails for deterministic Datalog evaluation. (#49)

### xlog-cuda

- [x] Add GPU-native schema-aware full-row deduplication and set difference. (#50)
- [x] Treat binary-join output counts as metadata reads (control-plane D2H exception, scoped to binary-join shape). (#52)

### Tests and Certification

- [x] Add deterministic Datalog D2H guardrail tests. (#49)
- [x] Add downstream frozen replay certification for crash-window bundles.

### Python and CLI

- [x] Add Python type stubs for IDE support.
      (`crates/pyxlog/python/pyxlog/__init__.pyi`,
      `crates/pyxlog/python/pyxlog/_native.pyi`, plus
      `crates/pyxlog/python/pyxlog/py.typed` marker file.)

### Release Gate

- [x] Public release only after deterministic Datalog guardrails pass locally and in manual GPU certification.
- [x] Public release only after crash-window frozen replay remains deterministic across 20 fresh subprocess replays.
- [x] Public release only after recursive deterministic set operations have zero data-plane D2H transfers. (#50, #52)
- [x] Public release only after docs distinguish release,
      source-build, and development install paths. (See
      `README.md:115` Source install / `README.md:133` GitHub
      release binary install / `README.md:139` PyPI install.)

### Items Originally Scoped to v0.5.5 — Re-Targeted

These were never v0.5.5 closure items in code; they have been
relocated to the release where the work actually belongs:

  * Operator-author docs, runtime-stack docs, deterministic-Datalog
    + general perf tuning guides, getting-started tutorial,
    deployment guide, migration guide, architecture/whitepaper
    binary-vs-WCOJ separation → **v0.6.0 Documentation**.
  * Binary hash-join count → prefix-scan → materialize prototype
    + count-prefix-materialize kernels → **v0.6.0 Recorded Launch
    Paths** (Inner CSM + indexed Inner CSM landed at `510dc33a` /
    `8cc0882c`; sub-slice 3 LeftOuter CSM captured at
    `.recovery/sub-slice-3-edits.md`).
  * Shared-memory optimization for small relations + warp-level
    primitives → **v0.6.2 xlog-cuda** (paired with WCOJ kernels).
  * Mixed binary/recursive/WCOJ determinism + Same Generation /
    triangle / skewed multi-way / recursive-frontier benchmarks +
    multi-way stress baselines + widened-frontier stress replay →
    **v0.6.2 Tests and Certification** (no target operator without
    WCOJ).
  * Native exact-induction tensorized integration + 449/449
    liveness reproduction + documented no-checked-in-PTX packaging
    policy with generated portable PTX staging →
    **v0.8.0 Bounded Exact Induction** (gated on a named
    downstream consumer materializing).
  * Per-call Python memory limit + query progress API →
    **v0.8.0 Python runtime/session API**.
  * CLI explain/plan visualization → **post-v0.10 product backlog**
    unless a DTS-DLM or release-certification consumer materializes.

## v0.6.0 - Stream-Safe GPU Runtime And Execution Discipline

The v0.6.0 release is the prerequisite layer for fully GPU-resident
binary-join retake (deferred from v0.5.5) and for v0.6.2 WCOJ. It is
infrastructure hardening, not a join-algorithm feature: the goal is a
stream-safe GPU runtime so subsequent work can be trusted under parallel
execution.

### xlog-cuda Runtime

- [x] Add v0.6 device-runtime allocator (`AsyncCudaResource`,
      `LoggingResource`, `GlobalDeviceBudget`, `XlogDeviceRuntime`,
      `StreamPool`) as an opt-in path next to legacy cudarc-backed
      allocation. (#54)
- [x] Add ABA guard via `(ptr, generation)` keying on
      `AsyncCudaResource::LiveEntry`.
- [x] Default trait hooks (`record_block_use`, `prepare_block_use`,
      `finish_block_use`) return `ResourceError::StreamMisuse` so
      `DirectCudaResource`-style stacks surface as loud failures
      instead of silent gaps.
- [x] **Access-aware stream dependency manager.** (PR #72,
      `77fd4948`.) Replaces post-launch-only `record_block_use`
      with `prepare_block_use(BlockId, stream, Access)` /
      `finish_block_use(...)` plus an `Access {Read, Write,
      ReadWrite}` enum. `LiveEntry` now tracks `last_write:
      Option<(StreamId, CudaEvent)>` (seeded with an
      allocation-ready event captured immediately after
      `cuMemAllocAsync`) and `outstanding_reads:
      Vec<(StreamId, CudaEvent)>`. Reads wait on `last_write`;
      writes wait on `last_write` plus every cross-stream
      outstanding read. Same-stream events are skipped. Closes
      both the use-after-prior-write hazard (a launch-stream
      reader / writer beginning before the prior cross-stream
      writer's event fires) AND the use-after-allocation hazard
      (a launch-stream consumer beginning before
      `cuMemAllocAsync` completes on the alloc stream). Backward
      compatibility: `record_block_use` is retained as a shim
      that calls `finish_block_use(Access::Read)` for the
      dealloc-wait surface, but production callers go through
      the recorder.
- [x] `XlogDeviceRuntime::prepare_first_use(slice, stream, access)`
      / `finish_first_use(...)` helpers for helper-internal
      scratch allocations whose first cross-stream consumer is
      a raw `cuMemsetD8Async` / `cuMemcpyDtoDAsync_v2` /
      `kernel.launch_on_stream` call ahead of any
      `LaunchRecorder::preflight`. Applied to every helper that
      allocates scratch on the manager's default stream and
      writes it on a caller-supplied `launch_stream`
      (`build_hash_table_v2_on_stream`,
      `gather_buffer_by_indices_on_stream`,
      `multiblock_scan_u32_inplace_on_stream` /
      `_view_inplace_on_stream`, every join variant's
      `d_count_only` / `d_output_count` / `out_col` zero-fills).

### xlog-cuda LaunchRecorder

- [x] Add `LaunchRecorder` (strict / permissive modes) with
      `read` / `write` / `read_write` / `read_column` /
      `write_column` primitives and an explicit preflight +
      commit pattern.
- [x] Make `preflight(&mut self)` stateful; `commit` rejects
      non-empty recorders that were not preflighted (closes the
      "discover-at-commit-time" footgun).
- [x] **Lifetime-free recorder + access-aware preflight.**
      (PR #72.) `LaunchRecorder` snapshots `BlockId` from each
      registered slice at record time and drops the source
      borrow immediately, so `&mut` kernel-param borrows after
      preflight are unrestricted. `preflight(&runtime)` queues
      `cuStreamWaitEvent` for every recorded use's cross-stream
      dependency BEFORE the launch (Read waits on `last_write`;
      Write / ReadWrite waits on `last_write` plus every prior
      reader on a different stream). `commit(self, &runtime)`
      records new events via `finish_block_use` AFTER the
      launch. Repeated registrations of the same block dedup to
      a single prepare/finish call with the strongest access,
      keyed by `(ptr, generation, device_ordinal)` (regression
      test in `launch::tests::dedup_keys_on_full_block_id_not_ptr_alone`
      locks the key shape against ABA collapses).
- [x] **`write_post_preflight_fresh` retired.** All fresh
      runtime-backed outputs now flow through the standard
      `write` API BEFORE preflight; the snapshot release model
      makes the post-launch escape hatch unnecessary. 78 call
      sites across `provider/{relational,filter,groupby,mod}.rs`
      migrated.
- [x] Reject DLPack / Arrow external columns at preflight in
      strict mode (`is_external()` branch in `read_column` /
      `write_column`).
- [x] Propagate `CudaColumn::runtime_block()` identity through
      filter-adjacent view helpers (`column_as_*`, `bytes_as_*`,
      `column_as_typed_view`).

### Recorded Launch Paths (xlog-cuda Provider)

- [x] `memset_recorded` (slice and column variants).
- [x] `compare_const_mask_recorded::<T: GpuScalar>`.
- [x] `compare_columns_mask_recorded::<T: GpuScalar>`.
- [x] `compact_buffer_by_device_mask_counted_recorded` — multi-kernel
      chain (`mask_clamp_rows` → `multiblock_scan_phase1` → recursive
      `multiblock_scan_u32_inplace_on_stream` + `phase3` →
      `capture_compact_count` → explicit `cu_stream.synchronize()` →
      host scalar read → per-column `compact_bytes_by_mask`) all on
      one explicit `launch_stream` via `launch_on_stream`.
- [x] `filter_recorded::<T>` — composed `compare_const_mask_recorded`
      → `compact_buffer_by_device_mask_counted_recorded` end-to-end.
- [x] `filter_columns_recorded::<T>` — composed
      `compare_columns_mask_recorded` →
      `compact_buffer_by_device_mask_counted_recorded` end-to-end.
- [x] Migrate the fused `compare+scan+compact` filter path
      (`u32`, `f64`) to the recorded discipline. (slice #3,
      `filter_fused_scan_recorded`)
- [x] **Decision: defer host-mask compact recorded migration.**
      `compact_buffer_by_mask` stays on its legacy entry; the
      recorded `compact_buffer_by_device_mask_counted_recorded`
      (already in tree) covers the device-mask case for
      runtime-backed callers. **Re-open trigger**: a
      runtime-backed recorded release path begins consuming
      host-provided masks. No current v0.6.x consumer.
- [x] **Decision: defer ILP / ILP-exact recorded migration.**
      Legacy ILP / ILP-exact path stays as-is; runtime block
      identity is not propagated through ILP view helpers.
      **Re-open trigger**: tensorized ILP / exact-induction
      downstream consumer work resumes (v0.8.0 native exact-induction
      consumer gate) and requires runtime-backed stream
      safety. Without that consumer, the current legacy path
      is correct and migration would add complexity for no
      observable gain.
- [x] Migrate sort operator surface to recorded launches against
      `launch_stream`. (slice #5, `sort_recorded` — narrow to
      U32 / Symbol keys; multi-type recorded sort deferred.)
- [x] Migrate dedup (full-row) operator surface to recorded launches
      against `launch_stream`. (slice #5, `dedup_full_row_recorded`
      — narrow to U32 / Symbol columns; key-based dedup,
      `diff_full_row`, and union deferred.)
- [x] Migrate GroupBy operator surface to recorded launches against
      `launch_stream`. (slice #6, `groupby_multi_agg_recorded` /
      `groupby_agg_recorded` — narrow to U32 / Symbol keys + Count /
      Sum / Min / Max aggs; LogSumExp and >4 key-column GroupBy
      deferred.)
- [x] Migrate hash-join operator surfaces to recorded launches
      against `launch_stream`. (slices #7A / #7B / #7C / #7D —
      `hash_join_v2_recorded` covers Inner / Semi / Anti /
      LeftOuter, and `hash_join_v2_with_index_recorded` covers
      the same four types over a cached `JoinIndexV2`.
      Narrow to ≤4 key columns per the `pack_keys` constraint;
      the deferred GPU-resident binary-join materialization
      prototype now has its prerequisite operator coverage.)
- [x] Wire `filter_recorded` / `filter_columns_recorded` into a
      runtime / provider opt-in selector so real callers can route
      filter operations through the recorded path. (slice #2,
      `XLOG_USE_RECORDED_FILTERS` env gate.)
- [x] Make binary hash-join materialization deterministic
      through count, prefix-scan, and materialize phases.
      Inner CSM (`510dc33a`), indexed Inner CSM (`8cc0882c`),
      non-indexed LeftOuter CSM (PR #84), indexed LeftOuter
      CSM (PR #87), `d_overflow`-recorder safety fix across
      the three earlier CSM siblings (PR #89), and env-gated
      dispatch through `XLOG_USE_RECORDED_CSM` / umbrella
      `XLOG_USE_RECORDED_OPS` for Inner / LeftOuter (indexed
      and non-indexed) (PR #91). Inner / LeftOuter CSM are
      now selectable via env in production; Semi / Anti
      remain on existing recorded paths (no CSM
      implementation — see deferral note below).
- [x] **Decision: defer Semi / Anti CSM kernels.** No
      `count_scan_materialize_recorded` variants for
      `JoinType::Semi` / `JoinType::Anti`; env dispatch leaves
      them on the legacy recorded paths. **Re-open trigger**:
      a benchmark or correctness scenario shows the existing
      recorded Semi / Anti paths are insufficient relative to
      the CSM-routed Inner / LeftOuter paths. Until then the
      legacy recorded Semi / Anti are correct and adding CSM
      variants would be code without a consumer.
- [x] Extend env-gated dispatch to recorded sort, dedup_full_row,
      GroupBy, and hash-join (Inner / Semi / Anti / LeftOuter,
      indexed and non-indexed). Per-operator env vars
      (`XLOG_USE_RECORDED_SORT`,
      `XLOG_USE_RECORDED_DEDUP`,
      `XLOG_USE_RECORDED_GROUPBY`,
      `XLOG_USE_RECORDED_HASH_JOIN`) plus the umbrella
      `XLOG_USE_RECORDED_OPS=1` that activates all five. Each
      dispatcher's eligibility check mirrors the recorded
      variant's narrow constraints; mismatches fall through
      to the legacy path. Defaults unchanged. Cert mode
      (`XLOG_USE_RECORDED_OPS=1 XLOG_USE_DEVICE_RUNTIME=1
      cargo test -p xlog-integration --test real_world_tests
      --release -- --test-threads=8`) passes **50/50** stress
      runs on merged main (PR #72, `77fd4948`); the previous
      ~98% pass / ~2% flake under multi-threaded contention is
      closed. xlog-cuda default suite is clean at
      `--test-threads=1` after the prepare/finish migration.

### Tests and Certification

- [x] RED reproducer for cross-stream use-after-free in the runtime
      allocator (`test_runtime_cross_stream_use_after_free.rs`).
- [x] Provider-level drop+reuse tests for every recorded primitive
      and every composed end-to-end migrated path.
- [x] Strict-recorder contract tests in `launch::tests`:
      post-preflight additions are rejected; pre-preflight
      `write` of a freshly-allocated runtime-backed buffer is
      accepted (snapshot drops the borrow so `&mut` kernel-param
      borrows after preflight remain valid).
- [x] **Multi-threaded sort+hash-join regression**
      (`tests/test_mt_sort_hj_alloc_ordering.rs`, PR #72): 8
      threads × 128 iters × 3 rounds friend-of-friend self-join.
      Was RED at baseline `8cc0882c` (~6/1024 failures per
      run); 1024/1024 + 1024/1024 across 10 consecutive runs
      after the fix. Locks both prior hazards (use-after-write
      across streams; use-after-allocation across streams).
- [x] **Recorder dedup key regression**
      (`launch::tests::dedup_keys_on_full_block_id_not_ptr_alone`):
      ABA reuse of a pointer inside a single recorder must
      not collapse distinct generations into one prepare/finish
      pair.
- [x] A3 / A4 stress reproducer suite (cross-stream lifetime
      stress under parallel forks, fixed and random schedules).
      (Commit `27ec3bd9` —
      `crates/xlog-integration/tests/test_a3_a4_stress.rs`.)
      Two workloads (`friends` sort+hash-join sensitive,
      `reach` recursive fixed-point + joins). Stable FNV-1a
      checksum over sorted rows; per-`GraphParams` reference
      computed serially in the parent. **A4 (16 fresh
      subprocess forks × 4 iters per child = 64 measurements):
      16/16 PASS in every fixture mode and every env
      combination tested.** Fork-isolated stream safety is
      verified.

      A3 (8 in-process threads × 32 iters = 256 measurements)
      surfaces drift but a 5-mode diagnostic matrix
      (`per_iter` / `per_thread` / `shared` × runtime+recorded
      / runtime-only / legacy-default) shows the failing
      condition is **NOT introduced by the v0.6.0 stream
      runtime or recorded launches**: drift fires identically
      in `legacy default + per_thread` (no
      `XLOG_USE_DEVICE_RUNTIME`, no `XLOG_USE_RECORDED_OPS`,
      one runtime per thread) — i.e., with no v0.6 code in the
      call chain. The bug class is pre-existing
      same-process multi-executor/provider concurrency against
      one CUDA primary context. Re-scoped as a non-blocking
      residual; see "Known Non-Blocking Residuals" below and
      the future-version backlog item in v0.9.0.
- [x] Public certification of the recorded launch discipline
      against the cert suite. (Commit `3361785b`.)
      `XLOG_USE_DEVICE_RUNTIME=1 XLOG_USE_RECORDED_OPS=1 cargo
      test -p xlog-cuda-tests --test certification_suite
      --release` passes **206/206** in 16s; the legacy default
      (`cargo test -p xlog-cuda-tests --test certification_suite
      --release`) still passes 206/206 in 21s. The cert
      `TestContext` now builds the production decorator stack
      (`AsyncCudaResource → LoggingResource → GlobalDeviceBudget
      → XlogDeviceRuntime`) when `XLOG_USE_DEVICE_RUNTIME=1` is
      set and uses `GpuMemoryManager::with_runtime` +
      `CudaKernelProvider::with_runtime`; the env-gated
      dispatchers in `provider::sort` / `filter_by_mask` /
      `hash_join_v2` / etc. then route through the recorded path
      when `XLOG_USE_RECORDED_*` is set. The harness reaps
      pending async frees between categories, and
      `GlobalDeviceBudget::allocate` now retries once after a
      reap on transient over-budget conditions so tight
      allocate-then-drop loops do not exhaust the reservation
      pool while real GPU memory is free.

### Known Non-Blocking Residuals

These are documented limitations that do NOT gate v0.6.0
release. They are tracked here so they cannot quietly become
blockers later.

- `cargo test -p xlog-cuda --test
  test_provider_launch_recorder -- --test-threads=8` shows
  9/42 `*_survives_drop_and_reuse` failures (was 23/42 at
  baseline `8cc0882c`). Pre-existing pattern from cross-runtime
  mempool aliasing under intra-binary test parallelism (each
  test builds its own `XlogDeviceRuntime`; they share the CUDA
  primary context). Production gate spec is `--test-threads=1`,
  which is clean. Full cross-runtime address coordination is
  out of scope for v0.6.0.

- **A3 in-process thread-of-N drift on the
  `test_a3_a4_stress` harness**: 8 threads × 32 iters each
  produce ~3% checksum drift across two recursive Datalog
  workloads (friends self-join, reach transitive closure).
  The 5-mode diagnostic matrix (commit `27ec3bd9` plus the
  `XLOG_A3_FIXTURE_MODE` / `XLOG_A3_DIAGNOSTIC` selectors)
  demonstrates this is NOT a v0.6.0 stream-safety regression:
  drift fires at comparable rates with `XLOG_USE_DEVICE_RUNTIME`
  unset AND `XLOG_USE_RECORDED_OPS` unset (i.e., legacy
  cudarc allocator + legacy operator dispatch + per-thread
  runtime), which means no v0.6 code path is in the call
  chain. The bug class is pre-existing same-process
  multi-executor/provider concurrency against a shared CUDA
  primary context. Tracked for v0.9.0+ under "Certify
  same-process multi-executor concurrency against one CUDA
  primary context".

  v0.6.0 release gate is **A4 fork-isolated stress + the
  cert suite + the umbrella ×50** — explicitly NOT "A3 must
  be zero drift".

### Documentation

- [x] Document the v0.6 device runtime stack
      (`AsyncCudaResource` / `LoggingResource` /
      `GlobalDeviceBudget`) and the `LaunchRecorder` preflight +
      commit contract, including the access-aware prepare/finish
      semantics introduced in PR #72.
      → [`docs/architecture/device-runtime.md`](architecture/device-runtime.md);
      linked from `docs/ARCHITECTURE.md` Memory Management
      section.
- [x] Add migration guidance for operator authors:
      `read` / `write` / `read_column` BEFORE preflight (no
      `write_post_preflight_fresh` — that API is gone);
      `runtime.prepare_first_use(slice, launch_stream, Access)`
      for helper scratch that runs raw CUDA work ahead of any
      recorder; `cu_stream.synchronize()` before host scalar
      reads; external-column rejection in strict mode.
      → [`docs/architecture/recorded-launch-migration.md`](architecture/recorded-launch-migration.md);
      linked from `docs/ARCHITECTURE.md` Memory Management
      section.
- [x] **Decision: defer non-blocker docs to the v0.6.x docs
      backlog.** The two release-blocker docs landed in v0.6.0
      (device-runtime architecture + recorded-launch migration
      guidance, both linked above). The remaining narrative
      docs — deterministic Datalog tuning guide, general
      performance tuning guide, getting-started tutorial,
      deployment guide, operator-author migration guide
      (separate from the install story already in
      `README.md:115/133/139`), and an architecture/whitepaper
      revision separating current binary-join execution from
      planned WCOJ execution — are not release-evidence
      blockers and were not gates against shipping v0.6.0.
      They live in the post-v0.6.0 docs backlog and re-open
      under their own narrative driver (e.g. user-facing
      performance feedback for the tuning guides; a public
      WCOJ landing for the architecture/whitepaper split).

### Release Gate

- [x] **Gate held: no WCOJ or fully GPU-resident binary-join
      PR merged ahead of v0.6.0.** Verified at v0.6.0 tag
      (`b1560674`). The migration discipline this gate
      protected — `launch_on_stream` on a caller-supplied
      `launch_stream`; all caller-provided buffers recorded
      before `preflight` with the correct `Access` kind; every
      fresh runtime-backed allocation that outlives an
      in-flight kernel registered via the standard `write` API
      BEFORE preflight (the recorder snapshots block identity
      at record time, so the kernel `&mut` borrow after
      preflight is unaffected); helper-internal scratch
      running raw CUDA work ahead of any
      `LaunchRecorder::preflight` calling
      `runtime.prepare_first_use(slice, launch_stream,
      Access::Write)` immediately after alloc; host scalar
      reads inside the chain explicitly ordered against
      `launch_stream` (non-blocking streams do not get
      default-stream implicit synchronization) — is now the
      ongoing operator-author contract documented in
      `docs/architecture/recorded-launch-migration.md`.
- [x] Public release only after the cert suite passes against a
      runtime-backed manager with the recorded launch paths
      exercised. (Closed by `3361785b`: 206/206 under
      `XLOG_USE_DEVICE_RUNTIME=1 XLOG_USE_RECORDED_OPS=1`.)
- [x] Public release only after the A3 / A4 stress reproducer
      suite observes zero use-after-free / stream-misuse
      failures. (Closed by `27ec3bd9`: A4 fork-isolated 16/16,
      symptom tally `stream-misuse=0 uaf=0` in every matrix
      mode. A3 thread-of-N drift is a documented pre-existing
      residual confirmed against legacy default — see "Known
      Non-Blocking Residuals" — not a v0.6.0 release blocker.)

### v0.6.0 Release Blockers Remaining

**All four blockers closed. v0.6.0 has no outstanding gates.**

1. ~~Formal cert harness for the recorded launch discipline.~~
   **DONE — commit `3361785b`.** `XLOG_USE_DEVICE_RUNTIME=1
   XLOG_USE_RECORDED_OPS=1 cargo test -p xlog-cuda-tests --test
   certification_suite --release` passes 206/206.
2. ~~A3 / A4 multi-fork stress harness.~~
   **DONE — commit `27ec3bd9`.** A4 fork-isolated stress passes
   16/16; A3 thread-of-N drift confirmed pre-existing against
   legacy default and re-scoped as non-blocking residual (see
   "Known Non-Blocking Residuals"). The v0.6.0 stream-safety
   gate is **A4 + cert suite + umbrella ×50**, not "A3 zero
   drift".
3. ~~Operator-author migration docs + runtime-stack docs.~~
   **DONE — commit `1b267dbf`.** Both items in the v0.6.0
   Documentation subsection are checked.
   `docs/architecture/device-runtime.md` covers the runtime
   stack; `docs/architecture/recorded-launch-migration.md`
   covers the operator-author checklist. Linked from
   `docs/ARCHITECTURE.md`.
4. ~~Decisions on host-mask compact migration and ILP /
   ILP-exact recorded migration.~~
   **DEFERRED with named consumer triggers (post-v0.6.0).**
   Neither has a current v0.6.0 consumer; pulling either in
   adds risk without improving the release evidence chain.
     * Host-mask compact (`compact_buffer_by_mask` recorded
       migration) re-opens **when a runtime-backed recorded
       release path begins consuming host-provided masks**.
     * ILP / ILP-exact recorded migration re-opens **when
       the tensorized ILP / exact-induction downstream
       consumer work resumes (v0.8.0 native exact-induction
       consumer gate) and requires runtime-backed
       stream safety**.
   Both items are now annotated under Recorded Launch Paths
   above with the same trigger language.

Items NOT on the blocker list (deferred / out of scope):
host-mask compact migration without a consumer, ILP-exact
without downstream tensorized integration, multi-type
recorded sort, key-based dedup recorded migration, LogSumExp
GroupBy recorded migration, WCOJ. These were enumerated in
the section above with deferral reasons.

## v0.6.1 - Recorded CSM Dispatch And Certification

Closed release on top of v0.6.0. Enables count-scan-materialize
(CSM) hash-join methods for `Inner` / `LeftOuter` (indexed and
non-indexed) under an env gate, closes a stream-safety gap in the
three earlier CSM siblings, and names the CSM cert mode explicitly
in the cert harness so reports are unambiguous. No kernel changes,
no algorithm changes, no eligibility relaxation. Default behaviour
for legacy callers is unchanged; the new path is opt-in via
`XLOG_USE_RECORDED_CSM=1` (or umbrella `XLOG_USE_RECORDED_OPS=1`).

### Recorded Launch Paths (xlog-cuda Provider)

- [x] **Indexed LeftOuter CSM operator** (PR #87,
      `hash_join_left_outer_v2_with_index_count_scan_materialize_recorded`).
      Probe-only pack on `launch_stream` plus a cached
      `JoinIndexV2` for the build side, sharing the
      count → scan → materialize phase shape with the
      non-indexed LeftOuter CSM (PR #84) and the indexed Inner
      CSM. No new kernels; reuses the four already-migrated CSM
      kernels plus `hash_join_csm_unmatched_mask` from PR #84.
- [x] **`d_overflow` recorder safety fix in three CSM siblings**
      (PR #89). Phase B's materialize kernel takes `d_overflow`
      as a kernel param. The three earlier CSM methods
      (`hash_join_inner_v2_count_scan_materialize_recorded`,
      `hash_join_left_outer_v2_count_scan_materialize_recorded`,
      `hash_join_inner_v2_with_index_count_scan_materialize_recorded`)
      did not register `d_overflow` on their materialize-phase
      `LaunchRecorder`, so the runtime was free to release the
      block once `rec_count.commit` resolved — a potential
      use-after-free if pool reuse beat kernel completion. Each
      site now registers `rec_mat.write(&d_overflow);` before
      `rec_mat.preflight`, matching the indexed-LeftOuter CSM
      site (PR #87) so all four CSM materialize recorders are
      identical.
- [x] **Recorded CSM hash-join env dispatch** (PR #91). Routes
      `JoinType::Inner` and `JoinType::LeftOuter` through CSM
      for both the non-indexed and indexed entry points when
      `XLOG_USE_RECORDED_CSM=1` (or umbrella
      `XLOG_USE_RECORDED_OPS=1`) is set. `Semi` / `Anti` always
      route through the existing legacy recorded methods — no
      CSM implementation exists for them. Eligibility checks
      preserved exactly: runtime-backed manager, ≤4 keys
      (`pack_keys` constraint), key-type match, row-count caps,
      indexed-path key-byte and shape checks. New env-dispatch
      routing test suite
      (`crates/xlog-cuda/tests/test_csm_env_dispatch.rs`) proves
      selection across the Inner / LeftOuter × indexed /
      non-indexed × env-on / env-off matrix plus Semi / Anti
      and the >4-keys upstream short-circuit.

### Tests and Certification

- [x] **Cert-mode labeling** (commit `bca1e373`). The
      `certification_suite` header now prints
      `Recorded-op dispatch (explicit):` (extended to include
      `XLOG_USE_RECORDED_CSM`) and a synthesized `Cert mode:`
      line keyed off the explicit env flags. The three intended
      values match the v0.6.1 cert gate commands —
      `legacy/default`, `runtime+recorded`,
      `runtime+recorded+CSM` — so CSM-mode runs are
      self-documenting in the cert evidence.

### Release Gate

- [x] Three cert modes pass with explicit headers at tag time:
      `legacy/default`, `runtime+recorded`,
      `runtime+recorded+CSM`. Each runs the full 33-category
      certification suite and reports `1 passed; 0 failed`.

## v0.6.2 - Default-On Adaptive WCOJ Triangle Dispatch

### xlog-ir and Optimizer

- [x] Add hypergraph IR and WCOJ eligibility analysis for rule
      bodies. Shipped in `xlog-logic::hypergraph` with vertices,
      hyperedges, structural boundaries, typed analysis, transitive
      SCC type inference, and canonical explain output.
- [x] Add a mixed plan contract that distinguishes
      `RulePlan::MultiwayCandidate` from `RulePlan::BinaryFallback`
      while preserving every fallback boundary for telemetry and
      future dispatch decisions.
- [x] Keep binary join backend as fallback for unsupported types,
      aggregation boundaries, negation boundaries, low-cardinality /
      non-triangle rules, mixed-width triangles, recursive SCCs, and
      RIR shapes outside the certified matcher.

### xlog-runtime

- [x] Add deterministic default-on adaptive WCOJ execution for the
      certified non-recursive triangle RIR shape
      `tri(X,Y,Z) :- e1(X,Y), e2(Y,Z), e3(X,Z)`.
- [x] Add force / adaptive / hard-disable controls:
      `RuntimeConfig::wcoj_triangle_dispatch`,
      `wcoj_triangle_dispatch_adaptive`,
      `wcoj_triangle_dispatch_disabled`, and env vars
      `XLOG_USE_WCOJ_TRIANGLE_U32`,
      `XLOG_USE_WCOJ_TRIANGLE_ADAPTIVE`,
      `XLOG_DISABLE_WCOJ_TRIANGLE`.
- [x] Add executor dispatch telemetry via
      `Executor::wcoj_triangle_dispatch_count()` and feature-gated
      `Executor::take_wcoj_phase_timing()`.
- [x] Cache the executor WCOJ launch stream to avoid exhausting the
      grow-only runtime `StreamPool` on long-lived executors.

### xlog-cuda

- [x] Add WCOJ physical relation layout construction for 2-column
      triangle inputs: `wcoj_layout_u32_recorded` for `U32` /
      `Symbol`, and `wcoj_layout_u64_recorded` for `U64`. Layouts are
      lex-sorted and full-row deduped.
- [x] Add layout fast-path for already strictly sorted+unique inputs.
      The recorded checker proves the property, then a recorded
      device-side clone skips sort + dedup. Slow path falls back to
      `dedup_full_row_recorded`.
- [x] Add deterministic WCOJ kernels for 3-way conjunctive triangle
      joins (`u32` / `Symbol` physical path and `u64` path). The
      pipeline is count → device-side prefix scan → materialize with
      deterministic offsets and no count-vector D2H.
- [x] Add single-GPU adaptive skew detection for triangle dispatch.
      The classifier uses 64 hash-mixed buckets over the three
      join-key columns and dispatches WCOJ when score ≥ 0.10.
- [x] Add small metadata D2H chokepoint
      `dtoh_small_metadata_untracked` with a hard 4 KB cap for
      classifier histograms and similar metadata; strict-D2H tests
      lock the contract.
- [x] Add feature-gated WCOJ phase timing for count / scan / total /
      materialize GPU phases and dispatch-level classifier / layout /
      residual wall-clock buckets.

### Adaptive Indexing

- [x] Add a WCOJ-specific sort-merge-style fast-path for pre-sorted
      triangle inputs through the layout checker + recorded clone.

### Tests and Certification

- [x] Add WCOJ CPU parity tests through the hypergraph reference
      evaluator and planner-to-provider certification.
- [x] Add WCOJ deterministic output tests for triangle provider
      entries and executor dispatch counters.
- [x] Add public WCOJ certification workloads for Same Generation,
      triangle, skewed multiway, deep recursive frontier, and
      mutually recursive parity SCC in the pure-Rust oracle stack.
- [x] Add runtime/provider certification for triangle WCOJ, including
      u32, Symbol, u64, mixed-width fallback, unsupported-type
      fallback, RIR-shape policy, adaptive/default-on policy, strict
      deterministic-D2H checks, and layout fast-path checks.
- [x] Capture binary-join vs WCOJ triangle benchmark baselines and
      adaptive/default-on acceptance data under
      `docs/evidence/2026-05-01-wcoj-bench-baseline/`.
- [x] Add Same Generation, triangle, skewed multi-way, and deep
      recursive-frontier certification workloads in the oracle stack.

### Documentation

- [x] Document WCOJ benchmark methodology and env/config knobs in
      `docs/BENCHMARKS.md`.
- [x] Record WCOJ baseline, adaptive/default-on acceptance,
      pre-fast-path phase timing, and post-fast-path phase timing in
      `docs/evidence/2026-05-01-wcoj-bench-baseline/`.

## v0.7.0 - General WCOJ Architecture and Runtime Expansion

v0.7.0 owns the WCOJ work intentionally left out of the v0.6.2
triangle release. The goal is to turn the certified triangle accelerator
into a broader WCOJ subsystem without weakening the v0.6.2 fallback and
stream-safety contracts.

### xlog-ir and Optimizer

- [x] Lower eligible plans to a dedicated `MultiWayJoin` /
      `WcojJoin` RIR node. v0.6.2 executor wiring pattern-matches the
      current lowered triangle RIR directly; a first-class RIR node is
      deferred to this release. **Done in the WCOJ expansion pack
      originally tracked as v0.6.5 slice 1 — `RirNode::MultiWayJoin`
      with `inputs`/`slot_vars`/`output_columns`/`fallback`; promoted post-
      optimizer in `xlog-logic::promote::promote_multiway`.**
- [x] Add variable-ordering cost model for WCOJ. v0.6.2 ships only
      deterministic appearance-order planning plus a trait boundary for
      future cost models.
- [x] Add join reordering based on selectivity estimates.

### xlog-runtime

- [x] Integrate WCOJ into semi-naive recursive evaluation.
      **Done in the WCOJ expansion pack originally tracked as v0.6.5
      slice 4 — `Executor::execute_wcoj_or_fallback_node`
      hooks both the seeding pass and per-variant evaluation in
      `execute_recursive_scc`; promoter gates per-rule on in-SCC
      Scan count.**
- [x] Preserve deterministic mixed execution across WCOJ and
      binary-join rules inside recursive SCCs. v0.6.2 preserves
      deterministic fallback for unsupported recursive shapes.
      **Done in the WCOJ expansion pack originally tracked as v0.6.5
      slice 4 — `MultiWayJoin.fallback` identity
      invariant + body-keyed `try_dispatch_wcoj_*_on_body`
      preserve binary-join behavior bit-identically when the
      cost model declines or the shape doesn't match.**
- [x] Add statistics integration into recursive SCC evaluation.

### xlog-cuda

- [x] Add sorted relation accessors beyond the triangle layout helper.
- [x] Add deterministic WCOJ kernels for 4-way conjunctive joins.
      **Done in the WCOJ expansion pack originally tracked as v0.6.5
      slice 2 — `wcoj_4cycle_count` / `wcoj_4cycle_materialize`
      kernels (u32 + u64 + Symbol parity); skew classifier with
      max-reduction over the four join positions; force gate +
      adaptive opt-in.**
- [x] Add general-arity WCOJ after 3-way and 4-way certification.
- [x] Add histogram-guided block scheduling / heavy-row offload.
      v0.6.2 measured this and deferred it: after the layout
      fast-path, materialize is a plausible future target but no
      longer the obvious next slice.
- [x] Add kernel fusion where benchmarks show materialization overhead
      dominates.
- [x] Add shared-memory optimization for small relations.
- [x] Add warp-level primitives for small-relation optimization.

### Adaptive Indexing

- [x] Add nested-loop join for small relations.
- [x] Add general sort-merge join for pre-sorted binary relations.
- [x] Feed selectivity and heat statistics into WCOJ variable ordering.

### Tests and Certification

- [x] Add GPU Same Generation / skewed multiway / deep-recursive WCOJ
      execution gates. v0.6.2 certifies these at the oracle layer,
      not through GPU WCOJ kernels.
- [x] Add skewed multi-way GPU benchmark suite beyond triangle.
- [x] Preserve deterministic mixed execution across WCOJ,
      binary-join, and recursive rules under a single test
      harness. (The binary-join + recursive determinism part
      is already in `xlog-runtime`; v0.6.2 covers non-recursive
      triangle WCOJ only.)
- [x] Add downstream widened-frontier stress replay clean gate.
      (No replay harness is committed today; the harness needs
      to be built alongside the benchmarks above.)

### Documentation

- [x] Add dedicated WCOJ architecture guide. Architecture reference:
      `docs/wcoj-architecture-guide.md`.
- [x] Document WCOJ eligibility, fallback, and performance tuning in a
      user-facing guide rather than only code docs / benchmark docs.
      User guide reference: `docs/wcoj-user-guide.md`.

### v0.7.0 Status (as of 2026-05-18)

**22/22 ROADMAP items DONE** for the General WCOJ Architecture and
Runtime Expansion pack.
**3 internal commitments DONE** — created during slices 4–5,
folded into v0.7.0 closure (NOT deferred):

  * `record_join_result` feedback wiring from WCOJ output back
    into `xlog-stats::StatsManager`.
  * Default-flip `RuntimeConfig::wcoj_cost_model` from
    `SkewClassifier` to `Cardinality`.
  * Multi-recursive WCOJ (≥ 2 in-SCC body Scans).

**Total open for v0.7.0 tag: 0 items.** W7.1 is DONE: the user
authorized "push and tag"; `main` is pushed to `origin/main` through
`94a8e5f8`, and annotated tag `v0.7.0` is pushed with peeled target
`0537348f`.

**Closure board:** [`docs/v065-closure-board.md`](docs/v065-closure-board.md)
— authoritative tracker. Process rules, wave grouping, and
per-item acceptance gates live there. All v0.7.0 board rows are DONE.

**Historical slices originally shipped against this section before
the v0.7.0 retarget**:

  * Slice 1: `MultiWayJoin` RIR + promoter (item #1).
  * Slice 2: 4-cycle WCOJ kernels + adaptive opt-in (item #8).
  * Slice 3: `WcojCostModel` + `SkewScoreSource` seam.
  * Slice 4: Recursive-arm WCOJ dispatch (items #4, #5).
  * Slice 5: `CardinalityAwareCostModel` opt-in.

## v0.8.0 - DTS-DLM ML/Python Productization

v0.8.0 is a DTS-DLM-first release train. Its acceptance target is not
"more language surface"; it is whether DTS-DLM can execute the queued
M37-A+B path with production-grade xlog support: stable pyxlog
contracts, observable GPU memory / host-transfer behavior, incremental
persistent sessions, native exact-induction consumer integration, and
neural-symbolic bridge training hooks.

Broad language, CLI, and general product conveniences remain valid
backlog items, but they are not v0.8.0 gates unless a named DTS-DLM or
release-certification consumer depends on them.

### DTS-DLM Release Gates

- [x] Add a canonical DTS-DLM certification pack in xlog that replays
      the relevant Stage-4 and M37-A+B surfaces without requiring a full
      DTS pilot by default. (`docs/evidence/2026-05-18-v080-cert/`)
- [x] Gate v0.8.0 on pyxlog public-surface preservation for DTS-DLM:
      `LogicProgram.compile`, `program.session`, `session.put_relation`,
      `session.evaluate`, `session.export_relation`,
      `IlpProgramFactory.compile`, `train_on_compiled_relations`,
      `Program.compile`, `register_network`, `register_embedding`,
      `add_tensor_source`, `forward_backward_tensor`, `train_epoch`, and
      `optimizer_step`. (`symbol_coverage=17/17`)
- [x] Add a machine-readable pyxlog API compatibility manifest and diff
      check, modeled on the DTS-DLM pyxlog 0.7.0 surface evidence.
- [x] Add DTS-DLM zero-copy and determinism gates: no tracked hot-path
      D2H/H2D transfers, stable CUDA Graph counters where graph mode is
      enabled, and bit-exact replay on fixed fixtures.

### Python Runtime And Session API

- [x] Add async evaluation API for `CompiledLogicProgram`,
      `LogicRelationSession`, and `CompiledProgram` where the underlying
      operation can run without blocking the Python caller.
- [x] Add streaming results API for large query outputs, preserving
      DLPack zero-copy for chunked tensor columns.
- [x] Add per-call Python memory limit configuration.
- [x] Add query progress reporting for long recursive and neural-symbolic
      evaluations, with stable counters suitable for DTS-DLM pilot logs.
- [x] Expose production diagnostics for DTS-DLM: host-transfer stats,
      CUDA Graph stats, memory-budget usage, and peak-memory snapshots
      through documented pyxlog APIs. Do not fabricate no-op GPU memory
      probes on environments that cannot report them.

### Persistent Relation Maintenance

- [x] Surface relation delta APIs on `LogicRelationSession`: insert,
      delete, and batch update via DLPack column tensors.
- [x] Connect Python session deltas to runtime
      `RelationDelta` / `apply_deltas_and_recompute` so monotone
      insert-only SCCs avoid full recompute where the plan permits it.
- [x] Add DTS-DLM Stage-4 fixture proving delta updates produce
      byte-identical output to full `put_relation` replacement while
      reducing full-table re-upload work.

Deferred completion scope moved to v0.8.6:

- Batch update coalescing for repeated DTS-DLM Stage-4 `wmir_committed`
  updates.
- Change notification callbacks for session-managed relations, scoped to
  explicit Python opt-in.

### Neural-Symbolic Bridge Integration

- [x] Add term embedding inference path coverage to the pyxlog
      compatibility manifest and DTS-DLM certification pack.
- [x] Add foreign tensor predicates suitable for DTS-DLM bridge features
      and other GPU-resident tensor inputs.
- [x] Add neural output caching with cache-hit telemetry and a documented
      invalidation model.
- [x] Add top-k deterministic neural mode with fixed tie-breaking for
      seed-pinned DTS-DLM training and replay.
- [x] Add Belnap-aware dual-channel loss helpers for DTS-DLM M37-A+B:
      pro reward, contra penalty, quarantine penalty, and CFR-oriented
      diagnostics. These are Python/ML helpers; Stage-4 structural
      kernels remain agnostic to Belnap pro/contra semantics.
- [x] Add semantic loss functions required by M37-A+B, then add MSE,
      semantic, and infoloss variants only where a named consumer uses
      them.
- [x] Quantify circuit-cache behavior for repeated
      `forward_backward_tensor` calls: cache-hit rate, hit/miss counters,
      and repeated-query speedup.

### Native Exact Induction Consumer Integration

- [x] Integrate native exact-induction backend into the downstream
      tensorized ILP consumer path. (Native `kernels/ilp_exact.cu` +
      manifest registration + `crates/pyxlog/src/ilp_exact.rs` wrapper
      exist; DTS-DLM's `tensorized_ilp.py` calls the public
      `pyxlog.ilp.exact_induce.induce_exact(..., backend="native")`
      surface directly.)
- [x] Reproduce the downstream 449/449 liveness benchmark with native
      exact induction. (DTS-DLM evidence
      `docs/evidence/2026-04-17-m8-phase1-engine-integration.md`
      records native `both_heads_alive == 449/449`, head 0/1
      `449/449`, rollback `0.0`, and quarantine `0.0`.)
- [x] Decide and document the strict-per-topology compatibility policy
      for DTS-DLM, including how legacy Python-prototype behavior is
      compared against native per-topology-isolated scoring.
- [x] Resolve `ilp_exact.ptx` packaging policy: `kernels/ilp_exact.cu`
      is source; generated `ilp_exact.portable.ptx` and `.cubin`
      artifacts are staged into `pyxlog/kernels/` and embedded by the
      xlog-cuda build, matching the current `ilp.cu` / `ilp_credit.cu`
      convention rather than checking generated PTX into git.

Deferred completion scope moved to v0.8.6:

- Column-type dispatch beyond `U64`, including `U32` and `Symbol`
  callers for downstream tensorized ILP.
- Chain-topology shared-memory caching of L rows, gated by profiling and
  GPU-native speedup evidence.

### Profile-Gated Optimizer Work

Deferred completion scope moved to v0.8.6:

- Common subexpression elimination when DTS-DLM M37-A+B, Mistaber, or
  certification profiles show duplicated subplans on the hot path.
- Adaptive query re-optimization during execution when runtime telemetry
  shows stable mis-planning on consumer fixtures.
- Persistent hash index manager with background building after DTS-DLM,
  Mistaber, or pyxlog-session profiles identify index rebuild cost as a
  release blocker.

### Deferred Product Backlog

These items were intentionally not v0.8.0 gates after the DTS-DLM scope
review. They were promoted into v0.8.5 because they had named consumers:
general XLOG users, DTS-DLM future fixtures, and the v0.9.0
epistemic/solver branch. Their completed tracking lives in the v0.8.5 section
below.

## v0.8.5 - Language Completeness and Developer Experience

Status: closed, merged, pushed, and tagged as `v0.8.5` after the
2026-05-19 closure authorization.

v0.8.5 is a language-surface release. It refreshes the public language
reference, adds finite term/list/meta constructs, makes negation contracts
explicit, adds bound-query magic-set planning, lifts probabilistic aggregate
support into exact and MC paths, promotes approximate inference configuration,
and adds developer-facing explain/REPL/watch surfaces. Accepted execution must
reuse the production parser, AST, RIR, probabilistic IR, optimizer, runtime,
WCOJ, and CLI paths.

### Documentation And Semantic Contract

- [x] Refresh `docs/language-reference.md` to the v0.8.5 language contract,
      including unsupported forms and GPU-native execution guarantees.
- [x] Add `docs/architecture/language-v085.md` with parser, term, probability,
      CLI, and v0.9.0 handoff contracts.
      Evidence: `docs/evidence/2026-05-18-v085-docref/README.md`.

### Type And Term Model

- [x] Add domain alias preservation, named predicate columns, `list<T>`,
      finite `term`, finite `compound`, and static `predref` representation.
- [x] Reject non-finite or non-GPU-lowerable term forms with typed diagnostics.
      Evidence: `docs/evidence/2026-05-18-v085-types/README.md`.

### Lists And Safe Meta-Predicates

- [x] Add finite list syntax and list built-ins.
      Evidence: `docs/evidence/2026-05-19-v085-lists/README.md`.
- [x] Add meta-predicates such as `ground`, `var`, `nonvar`, `=..`, `functor`,
      `findall`, and `maplist` with static finite safety checks.
      Evidence: `docs/evidence/2026-05-19-v085-meta/README.md`.

### Negation And Magic Sets

- [x] Add negation-as-failure syntax and semantics where it is distinct from
      existing probabilistic WFS support.
      Evidence: `docs/evidence/2026-05-19-v085-naf/README.md`.
- [x] Add magic sets transformation for safe bound recursive queries.
      Evidence: `docs/evidence/2026-05-19-v085-magic-sets/README.md`.

### Probabilistic And Approximate Inference

- [x] Add aggregate support in probabilistic programs.
      Evidence: `docs/evidence/2026-05-19-v085-prob-aggregates/README.md`.
- [x] Add aggregate lifting for small domains.
      Evidence: `docs/evidence/2026-05-19-v085-aggregate-lift/README.md`.
- [x] Add approximate inference engine pragmas, CLI configuration, and
      confidence reporting.
      Evidence: `docs/evidence/2026-05-19-v085-approx/README.md`.

### Incremental Parsing And CLI Developer Experience

- [x] Add incremental parsing for interactive use.
      Evidence: `docs/evidence/2026-05-19-v085-incremental-parse/README.md`.
- [x] Add interactive REPL.
- [x] Add watch mode.
- [x] Add CLI explain/plan visualization.
      Evidence: `docs/evidence/2026-05-19-v085-cli/README.md`.

### Examples And Certification

- [x] Add at least 10 advanced v0.8.5 `.xlog` examples covering every feature
      node and at least 5 feature interactions.
      Evidence: `docs/evidence/2026-05-19-v085-examples/README.md`.
- [x] Add `scripts/validate_v085_examples.py` or equivalent validation with
      committed evidence JSON.
      Evidence: `docs/evidence/2026-05-19-v085-examples/validation_summary.json`.

## v0.8.6 - DTS-DLM Runtime Completion and GPU-Native Optimizer Pack

Status: closed, merged, pushed, and tagged as `v0.8.6` after the 2026-05-19
closure authorization. Governing goal document:
`docs/plans/2026-05-19-agent-v086-dts-runtime-completion-goal.md`.

v0.8.6 closes the seven v0.8.0 deferred completion items as a production
runtime/optimizer hardening pack. Acceptance requires fully GPU-native data
paths for accepted workloads, zero data-plane host transfers beyond explicit
control-plane/final-result exceptions, profile-backed optimizer triggers, and
consumer evidence for DTS-DLM, Mistaber `.xlog` workloads, v0.9.0
epistemic/solver prerequisites, and pyxlog session users.

Hard implementation rule: v0.8.6 must extend and reuse the existing xlog
codebase. New code must compose with current parser, RIR/PIR, optimizer,
runtime, CUDA provider, WCOJ, exact-induction, probabilistic, and pyxlog
surfaces. Reimplementation of an existing subsystem, duplicate fixture-only
engines, or parallel helper paths that bypass production dispatch are blockers.

### Persistent Relation Maintenance Completion

- [x] Add device-resident batch update coalescing for repeated DTS-DLM
      Stage-4 `wmir_committed` updates, with row-level insert/delete
      coalescing before recompute and byte-identical output versus sequential
      deltas.
- [x] Add explicit opt-in change notification callbacks for
      session-managed relations, delivered from committed delta summaries
      without forcing data-plane device-to-host transfers.
      Evidence: `docs/evidence/2026-05-19-v086-delta-coalesce/` and
      `docs/evidence/2026-05-19-v086-notify/`.

### Native Exact Induction Completion

- [x] Add native exact-induction column-type dispatch beyond `U64`,
      including `U32` and `Symbol` pair buffers, with explicit typed kernels
      or safe physical-layout dispatch and no silent narrowing.
- [x] Add chain-topology shared-memory caching of L rows only after profile
      evidence identifies the chain scorer as hot, with a required speedup
      gate and parity against the existing strict per-topology semantics.
      Evidence: `docs/evidence/2026-05-19-v086-exact-types/`,
      `docs/evidence/2026-05-19-v086-chain-smem-profile/`, and
      `docs/evidence/2026-05-19-v086-chain-smem/`.

### Profile-Gated Optimizer Completion

- [x] Add GPU-native common subexpression elimination for duplicated
      subplans in DTS-DLM, Mistaber, or certification workloads, including
      semantic equivalence checks and no extra data-plane host transfers.
      Evidence: `docs/evidence/2026-05-19-v086-cse/`.
- [x] Add adaptive query re-optimization during execution when runtime
      telemetry shows stable mis-planning on consumer fixtures, bounded by
      deterministic replay and rollback gates.
      Evidence: `docs/evidence/2026-05-19-v086-adaptive-reoptimization/`.
- [x] Add a persistent hash index manager with generation/schema/device keying,
      stale-index invalidation, deterministic LRU budget eviction, repeated
      session reuse, background-build request/completion/deferred telemetry,
      and a runtime-backed recorded provider build path after profiles identify
      index rebuild cost as a release blocker. The build-heavy repeated-session
      semi-join fixture records cached median 0.079429262s, uncached median
      0.254631847s, `speedup_ratio=3.206`, and zero tracked DTOH/H2D calls.
      Evidence: `docs/evidence/2026-05-19-v086-persistent-hash-index/`.

### Consumer Certification

- [x] Add v0.8.6 runtime consumer certification examples and validator for
      DTS-DLM, neutral Mistaber-derived `.xlog` workloads, v0.9.0 substrate
      primitives, and public pyxlog session compatibility. The validator runs
      the new examples through `xlog-cli run/explain`, invokes the existing
      v0.8.0 and v0.8.5 validators, records raw timings/transfer evidence, and
      audits production-path reuse instead of introducing a fixture-only engine.
      Example execution and consumer certification pass; feature coverage is
      backed by validator-owned behavior probes, and public pyxlog persistent
      index session reuse has a passing behavior probe.
      Evidence: `docs/evidence/2026-05-19-v086-consumers/`.

## v0.8.7 - Living-World Diagnostics and Provenance Pack

Status: integrated into the v0.9.0 release candidate through the local
`integration/v090-v089-union` merge commit `8a7dbd3f`; no standalone v0.8.7
tag or publication is claimed here. Architecture source of truth:
`docs/architecture/living-world-diagnostics-v087.md`.

v0.8.7 closes the initial living-world and Project 1 auditability gaps without
changing the production data path. New surfaces report rule metadata,
generated-rule row decisions, proof frontiers, streamed graph provenance,
relation-delta planner telemetry, validation staging events, relation evidence,
and neural lineage/hot-loop state; they do not force host row materialization
unless the caller already selected a host-readable API.

### Rule And Proof Diagnostics

- [x] Add shared `xlog-logic` diagnostics records for `RuleProvenance`,
      `RuleSourceKind`, and `QueryProofTrace`.
- [x] Extend `xlog explain` text and JSON reports with `rule_provenance` and
      `proof_traces`, including generated magic-set rewrite rules when present.
- [x] Extend `xlog explain --format json` with `generated_rule_diagnostics`
      row decisions, failed predicates, threshold comparisons, and aggregate
      inputs for accepted and rejected generated-rule rows, including external
      candidate rows resolved from colocated execution manifests.
- [x] Expose `rule_provenance()` and `proof_traces()` from deterministic
      pyxlog programs, persistent relation sessions, and probabilistic pyxlog
      programs.

### Induction Provenance

- [x] Add `xlog-induce` generated-rule provenance records with selected rule
      source, search-space size, predicate inventory, positive support rows,
      rejected alternatives, selected-rule falsification count, and stable
      generation trace hash.
- [x] Add an in-memory `InductionProvenanceRegistry` for callers that promote
      generated rules and need to retain their audit records.

### Delta And Temporal Diagnostics

- [x] Add native biomedical graph streaming through `xlog_gpu::biokg`, covering
      JSONL/CSV/N-Triples edge streams, typed edge sinks, row hashes, relation
      histograms, split provenance, and bounded-memory chunk diagnostics.
- [x] Extend persistent-session delta reports with `changed_relation_names` and
      metadata-only `debug_trace` entries.
- [x] Add `LogicRelationSession.apply_relation_delta_debug(...)` with opt-in
      full-recompute comparison through query-store equivalence.
- [x] Add `DeltaPlannerTelemetry` with affected SCCs, cache reuse, fallback
      decisions, estimated/measured speedup, and planner guidance.
- [x] Add pyxlog temporal relation helpers that attach `timestamp_column`,
      `dataset_id`, row hashes, field hashes, uncertainty metadata, stream id,
      ordering column, and source metadata to session-managed relations.
- [x] Add pyxlog general relation evidence APIs:
      `put_relation_with_provenance(...)`, `evidence(...)`, and
      `RelationEvidence.provenance()`.
- [x] Add promote-only-on-PASS validation staging via
      `scripts.validation_staging.ValidationStagingRun`.

### Neural Hot-Loop Diagnostics

- [x] Add `CompiledProgram.neural_hot_loop_diagnostics()` with post-load
      transfer stats, CUDA Graph counters, circuit-cache counters, and explicit
      unavailable statuses for unsupported per-iteration control-plane and
      scalar-sync counters.
- [x] Add nn/4 lineage APIs for `checkpoint_hash`, `split_hashes`,
      `calibration_metrics`, `cuda_device`, `influence_audit`, and
      `changed_acceptance` records.

### Documentation And Validation

- [x] Document the full v0.8.7 diagnostics architecture in
      `docs/architecture/living-world-diagnostics-v087.md`.
- [x] Update the Python bindings, CLI reference, GPU execution, and bounded
      exact-induction architecture docs for the new public surfaces.
- [x] Add source-level coverage for the v0.8.7 Python diagnostics API surface,
      validation staging, Rust tests for CLI generated-rule diagnostics,
      biomedical graph streaming, relation-delta planner telemetry, and
      induction provenance.

## v0.8.8 - Living-World Diagnostics Provenance Refinement

Status: integrated into the v0.9.0 release candidate through the local
`integration/v090-v089-union` merge commit `8a7dbd3f`; no standalone v0.8.8
tag or publication is claimed here. Architecture source of truth:
`docs/architecture/lwm-diagnostics-provenance.md`.

v0.8.8 hardens the living-world diagnostics pack with stable induced-rule
aliases, explicit process-boundary and temporal-order provenance, source-level
coverage for the v0.8.8 pyxlog surface, and BFO reproducer-oriented
documentation.

### Native Induction Provenance

- [x] Add native `InducedRuleProvenance`, `InducedRuleRegistry`, support-row,
      alternative, and source-kind aliases for living-world induced-rule
      consumers while preserving the Project 1 induction provenance registry.
- [x] Preserve search-space size, predicate inventory, support rows, rejected
      alternatives, falsification counts, stable rule ids, and generation trace
      hashes without requiring Python-side law artifacts.

### Delta, Temporal, and Neural Diagnostics

- [x] Keep `LogicRelationSession.apply_relation_delta_debug(...)` visible in
      pyxlog stubs and documentation with changed relation names, equivalence
      evidence, and compact debug trace output.
- [x] Extend temporal provenance docs and wrappers with `process_boundary` and
      `temporal_order` metadata in addition to timestamp, dataset, row hash,
      field hash, uncertainty, stream, order, and source metadata.
- [x] Keep `CompiledProgram.neural_hot_loop_diagnostics()` documented as the
      unified post-load transfer, control-plane, scalar-sync, CUDA graph, and
      circuit-cache diagnostic surface.

### Validation

- [x] Add `python/tests/test_v088_lwm_source.py` for pyxlog stubs, docs, and
      Rust/Python source-surface coverage.
- [x] Add `docs/architecture/lwm-diagnostics-provenance.md` as the issue-by-
      issue v0.8.8 architecture note.

## v0.8.9 - Universal Case Reasoner Diagnostic Pack

Status: integrated into the v0.9.0 release candidate through the local
`integration/v090-v089-union` merge commit `8a7dbd3f`, sourced from
`feat/v089-ucr-xlog-issue-fixes`; no standalone v0.8.9 tag is claimed here.
Architecture summary: `docs/architecture/ucr-xlog-diagnostics.md`.
Issue ledger: `examples/BFO/universal_case_reasoner/xlog_issue_ledger.json`.

v0.8.9 promotes the reusable XLOG gaps exposed by the BFO demos into core XLOG
and pyxlog surfaces. Acceptance requires each Project 1, living-world, and
`UCR-XLOG-*` ledger item to have a reusable implementation, a minimal
reproducer, and a focused regression test outside the project-specific
validator.

### Neural-Symbolic Training

- [x] Add `pyxlog.ilp.neurosymbolic.train_neurosymbolic_program(...)` so one
      source can declare `nn/4` predicates, trainable symbolic rules, and a
      training objective that updates neural parameters and symbolic weights.
      Evidence: `python/tests/test_nn4_dilp_training_surface.py`.
- [x] Allow pure-Python pyxlog helper tests to import `pyxlog` when
      `pyxlog._native` is absent, while keeping native-backed APIs fail-closed.

### Differentiable Proofs And Rule Inventories

- [x] Add `xlog_logic::DifferentiableProofTraceMap` with stable proof IDs,
      support atoms, symbolic clause weights, logistic loss, and nonzero
      gradient hooks.
      Evidence: `crates/xlog-logic/tests/differentiable_proof_trace.rs`.
- [x] Add `pyxlog.ilp.inventory.build_rule_inventory(...)` and
      `PromotionResult.rule_inventory`, including selected/rejected clauses,
      training fold, held-out domains, promotion gates, and base-kernel checksum
      metadata.
      Evidence: `python/tests/test_ilp_rule_inventory.py`.

### Runtime And Transfer Diagnostics

- [x] Add `pyxlog.runtime_audit.CudaExecutionAudit` to fail hot-loop CUDA
      ranking tests on `.cpu()`, `.tolist()`, `.item()`, score-row downloads, or
      recorded H2D/D2H transfers.
      Evidence: `python/tests/test_nn4_cuda_no_host_transfer_contract.py`.
- [x] Add `xlog_logic::diagnose_module_boundaries(...)` for frozen kernel
      predicates, adapter-only fact modules, held-out domain declarations, and
      held-out-label candidate provenance.
      Evidence: `crates/xlog-logic/tests/module_boundary_diagnostics.rs`.
- [x] Add `pyxlog.transfer_diagnostics.compute_transfer_diagnostics(...)` for
      grouped macro F1, minimum-domain F1, bootstrap confidence intervals,
      baseline uplift, paired sign tests, and missing-domain or missing-variant
      failures.
      Evidence: `python/tests/test_transfer_metric_diagnostics.py`.

### BFO UCR Example And Packaging

- [x] Add `examples/BFO/universal_case_reasoner/` with goals, requirements,
      validation plan, BFO programs, evidence files, minimal reproducers,
      project-specific tests, validator tooling, and the resolved UCR issue
      ledger.
- [x] Harden `scripts/stage_pyxlog_kernels.sh` so pyxlog release kernel staging
      builds before resolving the release `OUT_DIR`.
      Evidence: `python/tests/test_kernel_packaging_layout.py`.

## v0.9.0 - Epistemic and Solver Semantics

### xlog-logic

- [x] Add Epistemic Intermediate Representation (EIR). Evidence:
      `docs/evidence/2026-05-18-v090-eir/` and
      `cargo test -p xlog-logic --test test_epistemic_eir`.
- [x] Add G91 semantics as a compatibility mode for classic epistemic logic.
      Evidence: `docs/evidence/2026-05-18-v090-g91/` and
      `cargo test -p xlog-logic --test test_epistemic_g91`.
- [x] Add FAEEL semantics as the default Founded Autoepistemic Equilibrium Logic mode.
      Evidence: `docs/evidence/2026-05-18-v090-faeel/` and
      `cargo test -p xlog-logic --test test_epistemic_faeel`.
- [x] Add Generate-Propagate-Test execution. Evidence:
      `docs/evidence/2026-05-18-v090-gpt/` and
      `cargo test -p xlog-logic --test test_epistemic_gpt`.
- [x] Add epistemic splitting. Evidence:
      `docs/evidence/2026-05-18-v090-split/`,
      `cargo test -p xlog-logic --test test_epistemic_split`, and the
      post-checkpoint split diagnostic amendment `415343c8`.
- [x] Integrate epistemic reasoning with probabilistic inference. Evidence:
      `docs/evidence/2026-05-18-v090-prob/`,
      `cargo test -p xlog-prob --features host-io --test epistemic_prob_gpu_accepted_evidence`,
      and `cargo test -p xlog-prob --test epistemic_prob_production_reuse`.

### Solver Services

- [x] Integrate solver services with `xlog-logic` constraints. Evidence:
      `docs/evidence/2026-05-18-v090-solver/`,
      `cargo test -p xlog-solve --test gpu_solver_accepted_evidence`, and
      `cargo test -p xlog-solve --test gpu_solver_production_reuse`.
- [x] Add incremental SAT semantics. Evidence:
      accepted solver lifecycle and assumption tests in
      `gpu_solver_accepted_evidence` and
      `test_epistemic_gpu_wcoj_execution`.
- [x] Add assumption-based solving. Evidence:
      accepted GPU assumption push/retract lifecycle gates in
      `gpu_solver_accepted_evidence`.
- [x] Add learned-clause transfer for incremental SAT. Evidence:
      same-device learned-clause publication/reuse gates in
      `gpu_solver_accepted_evidence` and production-reuse tests.
- [x] Add MaxSAT with soft constraints. Evidence:
      bounded weighted MaxSAT candidate, search-pruning, encoding, and
      scheduler gates in `gpu_solver_accepted_evidence`.
- [x] Add GPU portfolio solving. Evidence:
      accepted SAT/MaxSAT/UNKNOWN/TIMEOUT portfolio dispatch through the
      GPU solver production adapter.

### Probabilistic Reasoning

- [x] Add incremental circuit updates for dynamic programs. Evidence:
      `changed_assumption_replaces_active_evidence_without_rebuilding_circuit`
      in `cargo test -p xlog-prob --test epistemic_prob`.
- [x] Add alternative knowledge compilers such as c2d and miniC2D. Evidence:
      `c2d_and_minic2d_compiler_adapters_are_explicitly_represented` in
      `cargo test -p xlog-prob --test epistemic_prob`.

### Documentation and Tests

- [x] Add epistemic semantics guide. Evidence:
      `docs/epistemic-solver-semantics-guide.md`.
- [x] Add solver-semantics certification tests. Evidence:
      `cargo test -p xlog-solve --test gpu_solver_accepted_evidence`,
      `cargo test -p xlog-solve --test gpu_solver_production_reuse`, and
      `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution`.

### Concurrency Hardening Retargeted Out Of v0.9.0 Closure

- [ ] **Retargeted: certify same-process multi-executor concurrency
      against one CUDA primary context.** Surfaced by the
      v0.6.0 A3/A4 stress harness
      (`crates/xlog-integration/tests/test_a3_a4_stress.rs`,
      commit `27ec3bd9`): A3 in-process thread-of-N drift
      (~3% on recursive Datalog workloads) is reproducible
      against the legacy default path (no
      `XLOG_USE_DEVICE_RUNTIME`, no `XLOG_USE_RECORDED_OPS`),
      so it is NOT a v0.6.0 stream-safety bug — it is a
      pre-existing same-process multi-executor /
      multi-provider concurrency issue. Re-target candidates:
      `xlog-runtime::Executor` mutability under thread
      contention, `xlog-cuda::CudaKernelProvider` shared
      kernel/index caches, cudarc primary-context module-load
      semantics under concurrent first-launch. Pass criterion:
      A3 thread-of-N drift drops to zero on the harness's
      `per_thread` and `shared` fixture modes (matrix run
      via `XLOG_A3_FIXTURE_MODE=...`). This is not part of the
      `docs/plans/2026-05-18-agent-v090-epistemic-solver-goal.md`
      KPI surface and is not claimed by the v0.9.0 closure proposal;
      it remains a runtime-concurrency backlog item for the next
      runtime hardening train.

## v0.9.1 - Epistemic Executor Completion

Turns the v0.9.0 bounded epistemic executor into a load-bearing execution
surface: candidate worlds are derived from EIR, modal membership is value-level
on the device, FAEEL foundedness is per tuple key, epistemic constraints prune
world views, splits are equivalence-checked, and multi-epistemic-predicate rules
are solved jointly. EIR remains the semantic boundary and direct raw RIR
lowering stays a rejection boundary. All accepted work holds the cross-cutting
locks (no hidden CPU fallback, no fake predicate rewriting, no parallel side
engines, typed fail-closed, real runtime/device pilots). Status summary:
`docs/plans/2026-05-29-v091-epistemic-executor-completion-status.md`.

### xlog-logic / xlog-runtime / xlog-cuda (epistemic executor)

- [x] EGB-02 tuple-key and bound-value modal membership: ground, single/multi
      bound variable, repeated-variable equality, anonymous wildcard, and
      arity-0 keys on the GPU device path; fixed a global-gate soundness bug
      where ground/anonymous/nullary modal literals were left ungated. Evidence:
      `XLOG_USE_DEVICE_RUNTIME=1 cargo test -p xlog-runtime --test test_epistemic_gpu_workspace --release --features epistemic-logic-tests` (`egb02_*`).
- [x] EGB-01 EIR-derived candidate-world enumeration: candidate space derived
      from the program (full device lattice), with generated/propagated/tested/
      accepted/rejected/reason trace counts, deterministic results, empty
      accepted-world-view distinguished from failure, and resource fail-closed
      before partial execution. Evidence: `egb01_*` device pilots.
- [x] EGB-07 FAEEL founded self-support: per-tuple-key foundedness; unfounded
      `p() :- possible p().` rejected; G91 self-support kept separate. Evidence:
      `cargo test -p xlog-logic --test test_epistemic_faeel_foundedness` and
      `--test test_epistemic_g91`.
- [x] EGB-04 epistemic integrity constraints: `:- know/possible/not possible g().`
      prune candidate world views via a GPU constraint kernel (rejection reason
      6), constraints dropped from the reduced ordinary program (no RIR rewrite).
      Evidence: `egb04_*` device pilots.
- [x] EGB-05 safe split dependency and coupling: split/coalesce/reject decisions
      explained via typed `EpistemicComponentMergeReason`; paired split-vs-unsplit
      equivalence; recomposition covers each source rule exactly once. Evidence:
      `cargo test -p xlog-logic --test test_epistemic_split`.
- [x] EGB-06 joint multi-epistemic-predicate solving: rules coupling more than
      one distinct-name epistemic predicate (any operator mix incl. negated
      modal) solved jointly over the candidate world view, matching unsplit.
      Evidence: `egb06_*` device pilots and the integration coupling test.
- [x] EGB-03 nested modal operators (milestone scope): nested forms recognized
      explicitly and rejected with stable typed diagnostics, no parser-precedence
      accident, no fake flattening. Evidence:
      `cargo test -p xlog-logic --test test_epistemic_eir`.
- [x] Fixed nullary EDB fact materialization (pre-existing): `pred().` was
      materialized as 0 rows (read as absent), breaking ordinary nullary queries
      and ground/nullary modal membership. Added
      `CudaKernelProvider::create_zero_arity_buffer`; arity-0 facts now
      materialize one unit tuple. Evidence: `examples/epistemic/*` via
      `cargo test -p xlog-cli --test run_cli_tests test_xlog_run_epistemic_examples`.
- [x] Full-surface verification: device epistemic 116/116, 206-cert suite,
      epistemic logic 74/74, all 5 `xlog run` examples, `xlog-cuda` set ops
      35/35, and `xlog-integration` 206/206.

### Scoped out of v0.9.1 (remain typed fail-closed)

- [ ] Nested modal **semantics** (truth tables, FAEEL-vs-G91 nested behavior):
      nested forms stay rejected; EGB-03 milestone is representation +
      diagnostics only.
- [ ] Mixed per-row (bound-variable) and global (ground/anonymous/nullary)
      modal literals in a single rule.
- [ ] Epistemic constraints with variable tuple keys, constraints mixing
      relational/comparison literals with modal literals, constraint-only
      programs, and per-constraint (vs class-level) rejection attribution.
- [ ] Unsafe same-name multi-arity modal coupling (name-keyed relation store).
- [ ] Aggregate / compound / list / predref modal tuple keys.

## v0.10.0 - Multi-GPU and Out-of-Core Execution

### Runtime and Memory

- [ ] Add out-of-core execution for relations exceeding GPU memory.
- [ ] Add checkpointing and recovery.
- [ ] Add out-of-core spilling.
- [ ] Add memory-pool allocation reuse.
- [ ] Add memory defragmentation.
- [ ] Add memory-budget-aware index eviction policy.

### Multi-GPU

- [ ] Add `DistributedBuffer`.
- [ ] Add hash partitioning across devices.
- [ ] Add local joins on each device.
- [ ] Add gather/concat for distributed results.
- [ ] Add peer-to-peer copy when the GPU topology supports it.
- [ ] Add host-staging fallback.
- [ ] Add skew detection for distributed joins.
- [ ] Add skew rebalancing.
- [ ] Add network partition shuffle.
- [ ] Add distributed coordinator.
- [ ] Add fault tolerance and recovery.

### CUDA Kernels

- [ ] Add `distributed.cu` partitioning and shuffle kernels.
- [ ] Add partitioning kernels for multi-GPU execution.

### Data Interoperability

- [ ] Add direct cuDF DataFrame interchange.
- [ ] Add GPU-accelerated Parquet file reading.

### Tests

- [ ] Add cuDF integration tests.
- [ ] Add PyTorch integration tests.
- [ ] Add multi-GPU partitioning, skew, and recovery certification.

## Cross-Version Risk Register

### v0.5.5 Risks

- [ ] GPU-native deduplication and set-difference complexity can delay deterministic hardening.
- [ ] Deterministic GPU materialization may add measurable overhead unless benchmarks guide kernel boundaries.

### v0.6.0 Risks

- [ ] Stream-safety migration scope can expand beyond the release window if every operator family is migrated in one cut. Mitigated by per-operator slices behind sibling `*_recorded` entry points; legacy paths remain until the recorded path is certified.
- [ ] Composing recorded primitives currently relies on the runtime's record-all + wait-all event semantics (`last_use_events: Vec<CudaEvent>` in `AsyncCudaResource`). If composition depth grows materially, a future event-coalescing optimization may be needed to keep `deallocate` cost bounded.
- [ ] DLPack / Arrow external-memory consumers must coordinate cross-stream synchronization themselves; strict-mode rejection at preflight is loud, but downstream consumers depending on permissive recorders need explicit migration guidance.

### v0.6.2 Risks

- [ ] WCOJ planner and kernel scope can expand beyond the release window without strict 3-way/4-way certification gates.
- [ ] WCOJ kernels must not land before the operators they depend on are migrated to recorded launch discipline; otherwise multi-stream WCOJ execution would re-introduce the cross-stream use-after-free class v0.6.0 just closed.

### v0.9.0 Risks

- [ ] Epistemic semantics can introduce high complexity and must remain isolated from stable Datalog execution.
- [ ] D4 and solver integration must preserve deterministic certification paths.

### v0.10.0 Risks

- [ ] Multi-GPU synchronization and skew handling can dominate performance if partitioning policy is not benchmark-driven.
