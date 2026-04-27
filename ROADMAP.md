# XLOG Development Roadmap

Last updated: April 27, 2026
Current released version: v0.5.2
Current development target: v0.5.5 deterministic hardening, then v0.6.0 WCOJ

This roadmap is version-oriented so planned work is not hidden inside subsystem
sections. Historical and current-main work uses checked boxes. Future work uses
unchecked boxes and is assigned to a concrete future version.

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

### xlog-runtime

- [ ] Replace host-side multi-column full-row dedup/difference fallback with GPU-native deterministic set algebra.
- [ ] Add strict D2H guardrails for deterministic Datalog evaluation.
- [ ] Make binary hash-join materialization deterministic through count, prefix-scan, and materialize phases.
- [ ] Preserve deterministic mixed execution when binary joins, recursive rules, and future WCOJ rules coexist.
- [ ] Add query progress reporting API.

### xlog-cuda

- [ ] Add GPU-native schema-aware full-row deduplication and set difference.
- [ ] Add deterministic count-prefix-materialize binary join kernels.
- [ ] Add shared-memory optimization for small relations.
- [ ] Add warp-level primitives for small-relation optimization.

### Bounded Exact Induction

- [ ] Integrate native exact-induction backend into the downstream tensorized ILP consumer path.
- [ ] Reproduce the downstream 449/449 liveness benchmark with native exact induction.
- [ ] Add committed `kernels/ilp_exact.ptx` artifact once the kernel packaging policy is finalized and aligned with the existing ILP-family kernel convention.

### Python and CLI

- [ ] Add Python type stubs for IDE support.
- [ ] Add per-call Python memory limit configuration.
- [ ] Add CLI explain/plan visualization.

### Tests and Certification

- [ ] Add deterministic Datalog D2H guardrail tests.
- [ ] Add downstream frozen replay certification for crash-window bundles.
- [ ] Add Same Generation, triangle, skewed multi-way, and deep recursive-frontier benchmarks.
- [ ] Add skewed multi-way benchmark suite.

### Documentation

- [ ] Add deterministic Datalog tuning guide.
- [ ] Add general performance tuning guide.
- [ ] Add getting-started tutorial.
- [ ] Add deployment guide.
- [ ] Add migration guide.
- [ ] Update architecture and whitepaper docs to describe current binary-join execution separately from planned WCOJ execution.

### Release Gate

- [ ] Public release only after deterministic Datalog guardrails pass locally and in manual GPU certification.
- [ ] Public release only after crash-window frozen replay remains deterministic across 20 fresh subprocess replays.
- [ ] Public release only after downstream widened-frontier stress replay is clean.
- [ ] Public release only after recursive deterministic set operations have zero data-plane D2H transfers.
- [ ] Public release only after binary-join and multi-way stress benchmark baselines are captured.
- [ ] Public release only after docs distinguish release, source-build, and development install paths.

## v0.6.0 - Worst-Case Optimal Joins

### xlog-ir and Optimizer

- [ ] Add WCOJ eligibility analysis.
- [ ] Lower eligible plans to `MultiWayJoin` / `WcojJoin` RIR nodes.
- [ ] Add variable-ordering cost model for WCOJ.
- [ ] Add join reordering based on selectivity estimates.
- [ ] Keep binary join backend as fallback for unsupported types, aggregation boundaries, negation boundaries, and low-cardinality rules.

### xlog-runtime

- [ ] Add deterministic WCOJ execution path.
- [ ] Integrate WCOJ into semi-naive recursive evaluation.
- [ ] Preserve deterministic mixed execution across WCOJ and binary-join rules.
- [ ] Add statistics integration into recursive SCC evaluation.

### xlog-cuda

- [ ] Add WCOJ physical relation layout: sorted columnar indexes, trie-level key ranges, and projection metadata.
- [ ] Add sorted relation accessors for WCOJ.
- [ ] Add deterministic WCOJ kernels for 3-way conjunctive joins.
- [ ] Add deterministic WCOJ kernels for 4-way conjunctive joins.
- [ ] Add general-arity WCOJ after 3-way and 4-way certification.
- [ ] Add single-GPU skew detection and partitioning for WCOJ.
- [ ] Add kernel fusion where benchmarks show materialization overhead dominates.

### Adaptive Indexing

- [ ] Add nested-loop join for small relations.
- [ ] Add sort-merge join for pre-sorted data.
- [ ] Feed selectivity and heat statistics into WCOJ variable ordering.

### Tests and Certification

- [ ] Add WCOJ CPU parity tests.
- [ ] Add WCOJ K-run determinism tests.
- [ ] Add public WCOJ certification gates for Same Generation, triangle, and skewed multiway workloads.

### Documentation

- [ ] Add WCOJ architecture guide.
- [ ] Document WCOJ eligibility, fallback, and performance tuning.

## v0.7.0 - Epistemic and Solver Semantics

### xlog-logic

- [ ] Add Epistemic Intermediate Representation (EIR).
- [ ] Add G91 semantics as a compatibility mode for classic epistemic logic.
- [ ] Add FAEEL semantics as the default Founded Autoepistemic Equilibrium Logic mode.
- [ ] Add Generate-Propagate-Test execution.
- [ ] Add epistemic splitting.
- [ ] Integrate epistemic reasoning with probabilistic inference.

### Solver Services

- [ ] Integrate solver services with `xlog-logic` constraints.
- [ ] Add incremental SAT semantics.
- [ ] Add assumption-based solving.
- [ ] Add learned-clause transfer for incremental SAT.
- [ ] Add MaxSAT with soft constraints.
- [ ] Add GPU portfolio solving.

### Probabilistic Reasoning

- [ ] Add incremental circuit updates for dynamic programs.
- [ ] Add alternative knowledge compilers such as c2d and miniC2D.

### Documentation and Tests

- [ ] Add epistemic semantics guide.
- [ ] Add solver-semantics certification tests.

## v0.8.0 - Multi-GPU and Out-of-Core Execution

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

## v0.9.0 - Language, ML, and Product Backlog

### xlog-logic

- [ ] Add incremental parsing for interactive use.
- [ ] Add list syntax and list built-ins.
- [ ] Add meta-predicates such as `ground`, `var`, `=..`, `functor`, `findall`, and `maplist`.
- [ ] Add negation-as-failure syntax and semantics where it is distinct from existing WFS support.

### xlog-ir and Optimizer

- [ ] Add common subexpression elimination.
- [ ] Add magic sets transformation.
- [ ] Add adaptive query re-optimization during execution.

### Incremental Maintenance

- [ ] Add delete support with efficient delta propagation.
- [ ] Add batch update coalescing.
- [ ] Add change notification callbacks.

### Adaptive Indexing

- [ ] Add persistent hash index manager with background building.

### Probabilistic Reasoning

- [ ] Add aggregate support in probabilistic programs.
- [ ] Add aggregate lifting for small domains.
- [ ] Add approximate inference engine.

### Neural-Symbolic Integration

- [ ] Add term embedding inference path.
- [ ] Add foreign tensor predicates.
- [ ] Add neural output caching.
- [ ] Add top-k deterministic neural mode.
- [ ] Add semantic loss functions.
- [ ] Add MSE, semantic, and infoloss variants.

### Bounded Exact Induction

- [ ] Add column-type dispatch beyond `U64`, including `U32` and `Symbol` callers when needed.
- [ ] Add chain-topology shared-memory caching of L rows after profiling confirms it is a hot path.

### Python API

- [ ] Add async evaluation API.
- [ ] Add streaming results API.

### CLI

- [ ] Add interactive REPL.
- [ ] Add watch mode.

## Cross-Version Risk Register

### v0.5.5 Risks

- [ ] GPU-native deduplication and set-difference complexity can delay deterministic hardening.
- [ ] Deterministic GPU materialization may add measurable overhead unless benchmarks guide kernel boundaries.

### v0.6.0 Risks

- [ ] WCOJ planner and kernel scope can expand beyond the release window without strict 3-way/4-way certification gates.

### v0.7.0 Risks

- [ ] Epistemic semantics can introduce high complexity and must remain isolated from stable Datalog execution.
- [ ] D4 and solver integration must preserve deterministic certification paths.

### v0.8.0 Risks

- [ ] Multi-GPU synchronization and skew handling can dominate performance if partitioning policy is not benchmark-driven.
