# XLOG Development Roadmap

> **Last Updated:** March 5, 2026
> **Current Version:** v0.4.0-ga (Tagged) + post-GA hardening
> **Current Milestone:** v0.5.0 (in progress)
> **Status:** `main` is post-GA: GPU-resident ILP credit/loss (zero D2H, non-chunked), strict zero-D2H
> CI gate, COO memory cap with chunked fallback, 4 new CUDA kernels. Prior: GPU-native exact path,
> neural-symbolic training, dILP GA trainer, sparse executor.

---

## Overview

XLOG is a GPU-accelerated Datalog query engine. This roadmap tracks implemented features and planned development across all system components.

### Glossary of Terms

| Term | Definition |
|------|------------|
| **SCC** | Strongly Connected Component — a maximal set of predicates where each predicate depends (directly or transitively) on every other |
| **RIR** | Relational Intermediate Representation — the internal tree of relational operators (Scan, Filter, Join, etc.) |
| **PIR** | Provenance Intermediate Representation — graph structure tracking how derived tuples depend on probabilistic facts |
| **CNF** | Conjunctive Normal Form — a standard logical formula format used for knowledge compilation |
| **XGCF** | XLOG GPU Circuit Format — a levelized circuit representation optimized for parallel GPU evaluation |
| **WMC** | Weighted Model Counting — computing the sum of weights over all satisfying assignments of a formula |
| **Decision-DNNF** | Decomposable Negation Normal Form with decision nodes — a compiled circuit format that enables efficient WMC |
| **D4** | A state-of-the-art knowledge compiler that converts CNF formulas to Decision-DNNF circuits |
| **DLPack** | A cross-framework tensor interchange standard enabling zero-copy GPU data sharing |
| **Arrow IPC** | Apache Arrow Inter-Process Communication format for columnar data serialization |
| **Arrow C Data Interface** | A zero-copy Arrow FFI format for sharing arrays (including device memory pointers) |
| **Semi-Naive** | An optimization for recursive Datalog that only processes new (delta) tuples each iteration |
| **HISA** | Heat-based Index Selection Algorithm — adaptive indexing based on query access patterns |
| **LRU** | Least Recently Used — a cache eviction policy |
| **EDB** | Extensional Database — base facts provided as input |
| **IDB** | Intensional Database — derived relations computed by rules |

---

## Core Language & Compiler (`xlog-logic`)

### Implemented ✅

- [x] Datalog parsing via PEG grammar (Pest)
- [x] Stratified negation analysis with SCC-based ordering
- [x] Query syntax (`?- atom.`) for specifying output relations
- [x] Constraint syntax (`:- body.`) for integrity constraints
- [x] Recursive rule support with semi-naive evaluation
- [x] Comparison operators in rule bodies (`X != Y`, `X < Y`, etc.)
- [x] Arithmetic expressions via `is` keyword (`Z is X + Y`)
- [x] Built-in functions: `abs`, `min`, `max`, `pow`, `cast`
- [x] Aggregation operators: `count`, `sum`, `min`, `max`, `logsumexp`
- [x] Wildcard variables (`_`) in rule bodies
- [x] Predicate declarations with type annotations (`pred edge(u32, u32).`)
- [x] Symbol type for string values (reversible as of v0.3.2 — bidirectional string-to-ID mapping)
- [x] Probabilistic facts (`0.7::rain.`) for Bernoulli random variables
- [x] Annotated disjunctions (`0.3::a; 0.7::b.`) for categorical distributions
- [x] Evidence declarations (`evidence(atom, true|false).`)
- [x] Probabilistic queries (`query(atom).`)

### Implemented ✅ (v0.3.2)

- [x] Reversible symbol values (bidirectional string-to-ID mapping with query output display)
- [x] User-defined functions (arithmetic, conditional, recursive)
- [x] Module system with `use` imports and `private` visibility

### Planned 📋

- [ ] Incremental parsing for interactive use

---

## Query Optimizer (`xlog-logic`)

### Implemented ✅

- [x] Predicate pushdown — filters moved as early as possible in the plan tree
- [x] Cost-based join planning with dynamic programming for small rule bodies (≤10 atoms)
- [x] Greedy bushy join planning fallback for large rule bodies
- [x] Build/probe cost model for join tree selection
- [x] Statistics-seeded optimization via `StatsSnapshot` feedback from runtime
- [x] Cartesian join support via constant-key join (handles rules with disconnected atoms)

### Planned 📋

- [ ] Join reordering based on selectivity estimates
- [ ] Common subexpression elimination across rules
- [ ] Magic sets transformation for top-down evaluation
- [ ] Adaptive query re-optimization during execution

---

## Runtime Execution (`xlog-runtime`)

### Implemented ✅

- [x] Stratum-ordered execution respecting negation dependencies
- [x] SCC-aware recursive evaluation with semi-naive delta processing
- [x] Per-rule delta rewriting for efficient recursive joins
- [x] GPU-resident filter evaluation via mask DAG (no host round-trips)
- [x] GPU-resident arithmetic expression evaluation
- [x] GPU-resident groupby finalization (boundary detection, group IDs, key extraction)
- [x] Configurable iteration limits for fixpoint convergence
- [x] Versioned relation storage with update tracking
- [x] Profiling hooks for operation-level timing

### Planned 📋

- [ ] Out-of-core execution for relations exceeding GPU memory
- [ ] Checkpointing and recovery for long-running queries
- [ ] Query progress reporting API

---

## Incremental Maintenance (`xlog-runtime`)

### Implemented ✅

- [x] Semi-naive delta application API
- [x] Insert-only incremental updates for monotone SCCs
- [x] Full recomputation for non-monotone SCCs and their dependents
- [x] Delta propagation through dependent strata

### Planned 📋

- [ ] Delete support with efficient delta propagation
- [ ] Batch update coalescing
- [ ] Change notification callbacks

---

## Adaptive Indexing (`xlog-runtime`, `xlog-stats`)

### Implemented ✅

- [x] Per-relation heat tracking (access frequency)
- [x] Cardinality and byte-size statistics collection
- [x] Join selectivity observation for base-to-base joins
- [x] Join index cache with LRU eviction
- [x] Index invalidation on relation updates
- [x] Budget-aware index sizing heuristics
- [x] Build-side hash reuse for hot scan relations

### Planned 📋

- [ ] NestedLoop join implementation for small relations
- [ ] SortMerge join implementation for pre-sorted data
- [ ] Persistent hash index manager with background building
- [ ] Statistics integration into recursive SCC evaluation
- [ ] Memory-budget-aware index eviction policy

---

## GPU Backend (`xlog-cuda`)

### Implemented ✅

**Hash Joins:**
- [x] Inner join with hash collision safety (key verification)
- [x] Semi join (existence check)
- [x] Anti join (non-existence)
- [x] Left-outer join (unmatched rows zero-filled)
- [x] Bucketed CSR layout for cache-friendly memory access
- [x] 64-bit composite FNV-1a hashing for multi-column keys
- [x] Optional unsafe hash-only mode for performance experiments

**Sorting:**
- [x] Stable 4-bit radix sort for all scalar types
- [x] Multi-column lexicographic key support
- [x] GPU-resident permutation generation and application
- [x] Precomputed per-digit per-block offsets via GPU prefix sums

**Aggregation:**
- [x] Count, sum, min, max for integer types
- [x] LogSumExp for floating-point types (numerically stable)
- [x] Multi-key groupby with packed key encoding
- [x] GPU boundary detection and group ID assignment

**Set Operations:**
- [x] Union with deduplication
- [x] Set difference via sorted binary search marking
- [x] Support for all scalar types and multi-column schemas

**Filtering:**
- [x] Typed comparison kernels for all scalar types
- [x] Float predicate support with IEEE 754 total ordering (`NaN > Inf > nums > +0 > -0 > -Inf`)
- [x] Boolean mask composition (AND, OR, NOT)
- [x] Stream compaction without host round-trips
- [x] Multi-block prefix scan for large inputs

**Arithmetic:**
- [x] Binary operations: add, sub, mul, div, mod
- [x] Unary operations: abs, negation
- [x] Functions: min, max, pow, cast
- [x] Type promotion and casting

**Interop:**
- [x] DLPack export/import for device-resident columns (zero-copy)
- [x] Arrow C Data Interface device export (zero-copy, export-only)

### Planned 📋

- [ ] Partitioning kernels for multi-GPU distribution
- [ ] Peer-to-peer (P2P) GPU copy optimization
- [ ] Skew detection and handling for distributed joins
- [ ] Kernel fusion for common operation sequences
- [ ] Shared memory optimization for small relations

---

## Memory Management (`xlog-cuda`)

### Implemented ✅

- [x] Atomic budget reservation with compare-exchange loop
- [x] RAII-based tracking with automatic deallocation on drop
- [x] Configurable memory limits (fixed or percentage of device memory)
- [x] Column-major storage with typed schema
- [x] DLPack column ownership tracking with custom deleters

### Planned 📋

- [ ] Out-of-core spilling to host memory
- [ ] Memory pool with allocation reuse
- [ ] Defragmentation for long-running sessions

---

## CUDA Kernels (`kernels/`)

### Implemented ✅

| Kernel File | Purpose |
|-------------|---------|
| `join.cu` | Hash join build/probe (v2 with buckets), semi/anti joins, composite hashing |
| `filter.cu` | Typed comparisons, mask composition, stream compaction |
| `sort.cu` | Radix histogram, scatter, permutation apply |
| `dedup.cu` | Duplicate marking, row compaction |
| `groupby.cu` | Boundary detection, key extraction, aggregation kernels |
| `scan.cu` | Exclusive prefix sum, multi-block scan |
| `pack.cu` | Key packing, hashing, packed-row gather |
| `set_ops.cu` | Concatenation, sorted difference marking |
| `circuit.cu` | XGCF forward/backward evaluation for probabilistic inference |
| `cache.cu` | GPU circuit cache kernels (CNF hash, lookup/insert, cache store) |
| `cnf.cu` | GPU PIR→CNF encoding kernels |
| `d4.cu` | GPU D4 compilation kernels (frontier expansion, smoothing, build) |
| `neural.cu` | Neural fast-path kernels: AD-chain weight fill + probability-gradient scatter |
| `sat.cu` | GPU-native CDCL SAT solver + GPU verifier helpers (model/proof checks, CNF construction helpers) |
| `mc_sample.cu` | Bernoulli sampling for Monte Carlo inference |
| `weights.cu` | GPU-native weight/evidence builders for exact inference |

### Planned 📋

- [ ] `distributed.cu` — partitioning and shuffle kernels for multi-GPU
- [ ] Warp-level primitives for small-relation optimization

---

## Data Interoperability (`xlog-cuda`)

### Implemented ✅

**Apache Arrow:**
- [x] Export to Arrow RecordBatch (device → host copy)
- [x] Import from Arrow RecordBatch (host → device copy)
- [x] Arrow IPC stream read/write for file-based interchange
- [x] Arrow C Data Interface device export (zero-copy, export-only)

**DLPack (Zero-Copy):**
- [x] Export columns as DLPack managed tensors
- [x] Import DLPack tensors with schema inference
- [x] Import DLPack tensors with schema validation
- [x] Python capsule interface for framework-agnostic GPU data exchange

### Planned 📋

- [ ] Direct cuDF DataFrame interchange
- [ ] Parquet file reading (GPU-accelerated)

---

## Probabilistic Reasoning (`xlog-prob`)

### Implemented ✅

**Exact Inference (`prob_engine=exact_ddnnf`):**
- [x] Provenance extraction from positive Datalog programs
- [x] PIR (Provenance IR) graph construction
- [x] Tseitin encoding to CNF with stable variable mapping
- [x] GPU D4 integration for knowledge compilation (`kernels/d4.ptx`)
- [x] GPU CDCL equivalence verifier for circuit correctness (`kernels/sat.ptx`)
- [x] Decision-DNNF parsing retained for tests/fixtures (not used by production exact inference)
- [x] XGCF (GPU circuit format) construction with level-by-level layout
- [x] GPU forward pass for log-space weighted model counting
- [x] GPU backward pass for gradient computation
- [x] Conditional probability computation: P(Query | Evidence)
- [x] Per-query gradient output for learning applications

**Monte Carlo Inference (`prob_engine=mc`):**
- [x] Bernoulli sampling plan compilation
- [x] GPU Bernoulli matrix sampling
- [x] Deterministic per-world evaluation
- [x] Rejection sampling for evidence conditioning
- [x] Uncertainty reporting (standard error, confidence intervals)
- [x] Non-monotone SCC handling with skeptical semantics
- [x] Configurable sample count and random seed

### Implemented ✅ (Negation Support)

- [x] Negation in exact inference via NNF transformation and WFS
- [x] Stratified negation with automatic layer detection
- [x] Non-monotone (cyclic) negation via Well-Founded Semantics
- [x] Gradient computation through negated literals (sign flip)
- [x] `NegLit` PIR node for negated probabilistic leaves
- [x] Stratification analysis with edge polarity tracking

### Limitations (Current Version)

- Exact inference does not support aggregation in rule bodies
- Programs with aggregation must use Monte Carlo engine

### Planned 📋

- [ ] Aggregate support in probabilistic programs
- [ ] Alternative knowledge compilers (c2d, miniC2D)
- [ ] Importance sampling for rare-event queries
- [ ] Incremental circuit updates for dynamic programs

---

## Neural-Symbolic Integration (`xlog-neural`) — Phase 5 / v0.4.0

### Implemented ✅ (v0.4.0-alpha)

**Release gate (v0.4.0-alpha):**
- [x] Validate all examples in `examples/` end-to-end (CLI + Python where applicable) via `scripts/validate_examples.py`
- [x] Add additional neural examples beyond `examples/neural/01_minimal` (Coins, Poker, HWF, CLUTRR, etc.)

**Neural Predicates:**
- [x] `nn/4` syntax for neural network integration
- [x] Network registry with optimizer/scheduler management
- [x] Tensor source registry for external data (images, embeddings)
- [x] Neural output to annotated disjunction conversion
- [x] Deterministic vs non-deterministic modes (config options)

**Training Infrastructure:**
- [x] PyTorch autograd integration (backward pass to networks)
- [x] `register_network()` Python API with module/optimizer/scheduler
- [x] `train_model()` API with batch processing
- [x] NLL loss function with numerical stability
- [x] `forward_backward()` for single query training
- [x] `forward_backward_tensor()` for strict GPU-native training (returns CUDA tensor loss; no host reads)
- [x] `train_epoch()` for batch training
- [x] Circuit caching for 100x+ speedup on repeated queries
- [x] GPU neural fast-path for cached circuits (device-side AD weight fill + chain-rule gradients; DLPack interop)

**Inference Enhancements:**
- [x] Negation support in exact inference (NNF transformation + WFS)
- [x] Stratified negation with automatic layer detection
- [x] Non-monotone negation via Well-Founded Semantics
- [x] Gradient flow through negated literals

**Examples:**
- [x] Minimal MNIST Addition example (`examples/neural/01_minimal/`)
- [x] Coins example (`examples/neural/02_coins/`)
- [x] Multi-digit SVHN example (`examples/neural/03_mnist_multidigit/`)
- [x] Handwritten Formula example (`examples/neural/04_hwf/`)
- [x] Poker example (`examples/neural/05_poker/`)
- [x] CLUTRR example (`examples/neural/06_clutrr/`)

### Planned 📋 (v0.4.0-beta and beyond)

**Neural Predicates (Extended):**
- [ ] Term embeddings (learnable + pretrained)
- [ ] Foreign tensor predicates (dot, cosine, rbf, sigmoid, softmax, etc.)
- [ ] Neural output caching with configurable size
- [ ] Top-k deterministic mode

**Training Infrastructure (Extended):**
- [ ] Semantic loss functions (logic-based supervision)
- [ ] Loss functions (MSE, semantic, infoloss)
- [ ] Stop conditions (threshold, plateau detection)
- [ ] Learning rate schedulers

**Language Extensions:**
- [ ] List syntax (`[H|T]`, `[a,b,c]`) and built-ins (member, select, append)
- [ ] Meta-predicates (ground, var, `=..`, functor, findall, maplist)
- [ ] Negation as failure (`\+`)

**Inference Enhancements:**
- [ ] Aggregate lifting for small domains
- [ ] Alternative knowledge compilers (c2d, miniC2D)
- [ ] Importance sampling for Monte Carlo
- [ ] Approximate inference engine (geometric_mean, beam search)

**Neural-Symbolic Examples:**
- [ ] Coins example (two coin classifiers)
- [ ] MNIST multi-digit addition
- [ ] HWF (Handwritten Formula recognition)
- [ ] Poker (card rank classification)
- [ ] CLUTRR (family relationship reasoning)

**Design document:** `docs/plans/2026-01-20-v0.4.0-neural-symbolic-design.md`
**Implementation plan:** `docs/plans/v0.4.0-alpha-implementation.md`

---

## Differentiable ILP (dILP) Trainer (`pyxlog.ilp`) — v0.4.0-beta

### Implemented ✅ (dILP Beta)

**Sparse Mask API (Rust + PyO3):**
- [x] `set_rule_mask_sparse(candidate_ids, soft_probs, budget)` — Rust builds executor mask internally
- [x] No N3 tensor materialization; zero host→device mask transfer
- [x] AtomicU32 row-count cache on `CudaBuffer` for GPU-resident row counts

**Trainer Backend Abstraction:**
- [x] `MaskBackend` protocol with `SparseMaskBackend` (default) and `DenseMaskBackend` (fallback)
- [x] `debug_dense_mask=True` config option for dense parity testing
- [x] Dense-parity verified: sparse and dense backends produce the same discovered rules

**Training Pipeline:**
- [x] `train_only()` — multi-start training with adaptive temperature, entropy regularization, plateau detection
- [x] `train_and_promote()` — wraps `train_only()` + trial compilation + promotion gates → `PromotionResult`
- [x] Promotion gates: convergence, novel rate audit, regression check, holdout F1, ambiguity scan, typed schema
- [x] Transactional commit: trial program compiled with discovered rule before promotion

**Holdout Scoring + Ambiguity:**
- [x] LOO (leave-one-out) holdout F1 for ≤20 examples
- [x] k-fold holdout scoring for larger example sets (`holdout_strategy`, `holdout_folds`)
- [x] Per-fold precision/recall with deterministic fold assignment (`seed`)
- [x] Top-M ambiguity scan for alternative rules (`check_ambiguity`, `exhaustive_ambiguity`)

**Hard-Negative Mining:**
- [x] `sample_false_positives()` Rust API for GPU-side false positive sampling
- [x] Wired into trainer every 20 steps; D2H counter reset preserves zero-transfer contract

**Artifact Persistence:**
- [x] `LearnedArtifact.save(path)` / `.load(path)` with JSON serialization
- [x] SHA-256 candidate map hash verification (`verify_hash=True`)
- [x] Schema version `beta-v1` with forward-compatibility check

**Recursive Candidates:**
- [x] `allow_recursive_candidates=True` enables body-references-head candidates (i==k, j==k)
- [x] Default off; behind config flag

**Reliability:**
- [x] Beta gate: 4 stages (reach, grandparent, colleague, plus2) x 5 seeds = 20/20
- [x] Zero D2H column transfers in training step loop (hard gate verified)

**GA Hardening (current on `main`):**
- [x] Deterministic mode in trainer (`TrainConfig.deterministic`) with reproducible per-attempt seeding
- [x] `selected_hard` persisted in `LearnedArtifact` for deterministic auditability
- [x] Promotion holdout threshold gate (`holdout_threshold`, default 0.95)
- [x] Typed-schema gate with waiver-based manual-review fallback (`typed_schema_required`, `waiver_untyped`)
- [x] Host transfer telemetry via `host_transfer_stats()` / `reset_host_transfer_stats()`
- [x] `forward_p95_us` telemetry in `TrainResult.artifact.telemetry.step_timings`
- [x] GA reliability statistical gate test (`test_ilp_ga_reliability.py`, default 50 seeds, `max_attempts=2`)
- [x] GA performance/transfer accounting test (`test_ilp_performance.py`)
- [x] GA runtime optimization: 1447s → 436s via budget sweep (compile-once + `max_attempts` 7→2)

**Design document:** `docs/plans/2026-02-26-dilp-hardening-design.md`
**Implementation plan:** `docs/plans/2026-02-26-dilp-beta-impl.md`

### Planned 📋 (dILP beyond GA)

- [x] ~~Full GPU-resident loss computation (v0.5.0)~~ (done: `compute_ilp_loss_grad_gpu` with zero D2H in non-chunked paths, strict gate via `set_strict_zero_dtoh`, 4 new CUDA kernels)
- [ ] Config restoration from saved artifact JSON
- [ ] Telemetry persistence in artifact (optional, size-bounded)
- [x] ~~Typed query-buffer builder (non-u32 schemas)~~ (done: schema-aware typed packing for I32/I64/U64/Bool/Symbol, F32/F64 rejected)
- [x] ~~Full CI-grade 50-seed GA reliability runtime budget optimization~~ (done: 1447s → 436s, `max_attempts=2`)
- [x] ~~Full SLO benchmark harness for N=20/50/100/150~~ (done: parametrized `test_slo_scaling[N]` with wall-clock and forward_p95_us targets)

---

## GPU-Native Knowledge Compilation (`xlog-prob` + `xlog-solve`) — Phase 6 / v0.5.0

### Implemented ✅ (Foundations)

- [x] GPU CDCL equivalence verifier with zero host reads (fail-fast, on-GPU SAT/UNSAT validation)
- [x] Device-resident CNF size metadata (`GpuCnf::{num_vars,num_clauses,num_lits}`) to support GPU-native CNF builders
- [x] GPU PIR→CNF encoder (`encode_cnf_gpu`) with device-resident CSR emission and deterministic variable numbering
- [x] GPU-native circuit→CNF encoding for XGCF circuits + query construction helpers for equivalence checking
- [x] GPU D4 compiler core (frontier expansion + per-frontier DFS) with device-resident circuit builder
- [x] GPU D4 compile+verify entrypoint (`compile_gpu_d4_and_verify`)
- [x] Device-resident circuit cache + cache-aware evaluation (`GpuCircuitCache`, `compile_gpu_d4_and_verify_cached`)
- [x] Integration: replace CPU D4 invocation in `ExactDdnnfProgram` with GPU compile+verify (no host CNF/DDNNF materialization)
- [x] GPU smoothing seeds root support with all random vars (unconditional facts/evidence remain correct)
- [x] CUDA certification: SAT/CDCL category (G07) + device-count/row-count category (G08)

### Planned 📋

- [ ] Incremental/assumption interface for verifier reuse (share solver state across related queries)

---

## Epistemic Logic (`xlog-elp`) — Phase 7 / v0.6.0

### Planned 📋

- [ ] EIR (Epistemic Intermediate Representation) implementation
- [ ] G91 semantics (compatibility mode for classic epistemic logic)
- [ ] FAEEL semantics (default: Founded Autoepistemic Equilibrium Logic)
- [ ] Generate-Propagate-Test algorithm for epistemic evaluation
- [ ] Epistemic splitting for modular evaluation
- [ ] Integration with probabilistic tier for epistemic-probabilistic programs

**Prerequisites:** v0.5.0 complete (GPU-native compilation + verifier foundations), solver integration
**Estimated effort:** 3–4 months

---

## Solver Services (`xlog-solve`)

### Implemented ✅

- [x] Clause and literal data structures
- [x] GPU CDCL verifier (complete SAT/UNSAT) with on-GPU SAT model check + UNSAT proof check
- [x] Expectation-based verifier API with zero device→host reads (`solve_expect_sat`, `solve_expect_unsat`)
- [x] GPU-native equivalence-query construction helpers (`φ ∧ ¬C`, `C ∧ ¬φ`) used by `xlog-prob::compilation`
- [x] CLS (Continuous Local Search) algorithm for SAT/MaxSAT (heuristic; non-verifying)

### Planned 📋

- [ ] Integration with `xlog-logic` for constraint solving
- [ ] Incremental/assumption interface for verifier reuse (single solver state across multiple related queries)
- [ ] MaxSAT optimization with soft constraints
- [ ] GPU-accelerated parallel portfolio solving

---

## Python Bindings (`pyxlog`)

### Implemented ✅

- [x] PyO3-based extension module (`pyxlog`)
- [x] DLPack capsule input/output for GPU tensor interchange
- [x] `Program.compile()` for probabilistic programs
- [x] `LogicProgram.compile()` for deterministic programs
- [x] Engine selection: `exact_ddnnf` or `mc`
- [x] Gradient output for learning applications
- [x] Uncertainty metadata for Monte Carlo results
- [x] `dlpack_roundtrip()` helper for interop validation

### Implemented ✅ (v0.4.0-alpha neural-symbolic training)

- [x] `register_network(name, module, optimizer, scheduler)` — PyTorch network registration
- [x] `add_tensor_source(name, tensor)` — external data registration
- [x] `set_active_tensor_source(name)` — switch between train/test data
- [x] `forward_backward(query)` — single query training with gradients
- [x] `train_epoch(queries, batch_size)` — batch training epoch
- [x] `train_model(program, queries, epochs, batch_size)` — full training loop
- [x] `nll_loss(prob)`, `nll_loss_batch(probs)`, `nll_loss_tensor(prob)` — loss functions
- [x] `zero_grad()`, `optimizer_step()`, `scheduler_step()` — training utilities
- [x] `TrainingHistory` — epoch losses and batch metrics

### Implemented ✅ (dILP beta — ILP training)

- [x] `pyxlog.ilp.train_only()` — multi-start dILP training with sparse GPU mask
- [x] `pyxlog.ilp.train_and_promote()` — training + trial compilation + promotion gates
- [x] `TrainConfig` — expanded frozen config (temperature, budget, holdout, recursion, determinism, typed-schema gates)
- [x] `TrainResult` — convergence, metrics, discovered rule, artifact
- [x] `PromotionResult` — gate results, novel count/rate, committed source
- [x] `LearnedArtifact` — save/load with JSON + SHA-256 hash verification
- [x] `IlpProgramFactory.compile()` — compile learnable programs for ILP
- [x] `valid_candidates()` — enumerate candidate rules (recursive/non-recursive)

### Planned 📋

- [ ] PyPI package distribution
- [ ] Type stubs for IDE support
- [ ] Async evaluation API
- [ ] Memory limit configuration per-call
- [ ] Streaming results for large outputs

---

## Command-Line Interface (`xlog-cli`)

### Implemented ✅

- [x] `xlog run` — deterministic program execution
- [x] `xlog prob` — probabilistic program execution
- [x] Arrow IPC input: `--input relation=file.arrow`
- [x] Output formats: `--output pretty|csv|arrow`
- [x] Engine selection: `--prob-engine exact_ddnnf|mc`
- [x] Monte Carlo options: `--samples`, `--seed`, `--confidence`
- [x] Device selection: `--device`
- [x] Memory limit: `--memory-mb`
- [x] Query timing and statistics: `--stats` (human-readable and JSON formats)

### Planned 📋

- [ ] Interactive REPL mode
- [ ] Explain/plan visualization
- [ ] Watch mode for incremental file changes

---

## Multi-GPU & Distributed Execution — Phase 8 / v0.7+

### Planned 📋

**Single-Node Multi-GPU:**
- [ ] `DistributedBuffer` type for partitioned relations
- [ ] Hash-based partitioning across devices
- [ ] Local join execution on each GPU
- [ ] Result gathering and concatenation
- [ ] Peer-to-peer copy when topology supports it
- [ ] Host-staging fallback for non-P2P configurations
- [ ] Skew detection and work rebalancing

**Distributed Execution:**
- [ ] Network-based partition shuffle
- [ ] Distributed coordinator for query planning
- [ ] Fault tolerance and recovery

**Prerequisites:** v0.6.0+ complete, partitioning kernels
**Estimated effort:** 4–6 months

---

## Testing & Validation

### Implemented ✅

- [x] Workspace test suite: `cargo test --workspace --all-targets --exclude pyxlog --release`
- [x] CUDA certification suite: 206/206 tests passing (C01-C25 + G01-G08)
- [x] Hash join collision safety tests
- [x] Aggregation overflow/truncation tests
- [x] Large-input filter/compaction tests
- [x] Memory budget enforcement tests
- [x] End-to-end Datalog query tests
- [x] Probabilistic inference correctness tests
- [x] GPU CDCL verifier tests (SAT/UNSAT) + zero-host-read guardrails for verifier integrations
- [x] Performance regression benchmarks with CI tracking (Criterion.rs, `.github/workflows/bench.yml`)
- [x] Fuzz testing for parser, compiler, and type inference (cargo-fuzz, ASAN, `.github/workflows/fuzz.yml`)
- [x] Property-based testing for kernel correctness (proptest: sort stability, join correctness, filter idempotence, dedup determinism)
- [x] Float predicate edge case tests (NaN, Infinity, subnormals, signed zeros)
- [x] dILP beta test suite: 86 static test functions (124 parametrized), 20/20 reliability gate
- [x] dILP GA reliability gate test (`test_ilp_ga_reliability.py`, 50-seed statistical check)
- [x] dILP performance/transfer telemetry smoke tests (`test_ilp_performance.py`)

### Planned 📋

- [ ] Integration tests with cuDF and PyTorch

---

## Documentation

### Implemented ✅

- [x] Architecture guide (`docs/ARCHITECTURE.md`)
- [x] Language reference manual (`docs/language-reference.md`) (covers the v0.3.2 language surface)
- [x] Probabilistic tier design (`docs/architecture/xlog-prob.md`)
- [x] Adaptive indexing design (`docs/architecture/adaptive-indexing.md`)
- [x] Multi-GPU join design (`docs/architecture/multi-gpu-join.md`)
- [x] Data interoperability guide (`docs/architecture/cudf-interop.md`)
- [x] Example programs with explanations (`examples/`)
- [x] CUDA certification report (`docs/certification/`)
- [x] API documentation via rustdoc

### Planned 📋

- [ ] Getting started tutorial
- [ ] Performance tuning guide
- [ ] Deployment guide for production use
- [ ] Migration guide for ProbLog/Datalog users

---

## Version History

| Version | Status | Key Features |
|---------|--------|--------------|
| v0.4.0-alpha (main) | Achieved | GPU-native exact path (GPU D4 + GPU CDCL verifier + cache), device-only MC counts, Arrow C Device export, neural-symbolic training APIs |
| dILP beta (main) | Achieved | Sparse mask API, trainer backend abstraction, promotion pipeline, holdout F1, hard-negative mining, artifact persistence, recursive candidates, 20/20 reliability |
| dILP GA hardening (main) | Achieved | Deterministic mode, holdout threshold gate, typed-schema gate, host transfer telemetry, GA reliability/performance suites |
| v0.1.0 | Released | Deterministic Datalog, GPU joins/aggregations, basic CLI |
| v0.2.0 | Released | Probabilistic reasoning (exact + MC), Python bindings, GPU-resident execution |
| v0.3.1 | Released | Float predicates (IEEE 754 total ordering), benchmarks, `--stats` flag, fuzz testing, property-based testing |
| v0.3.2 | Released | Module system, UDFs, reversible symbols, showcase examples, count→u64 fix |
| v0.4.0-alpha | Implemented | Neural predicates (`nn/4`) + training milestone (release-gated on full example validation with real datasets) |
| v0.4.0-beta | Achieved | dILP beta trainer, GA-hardened promotion, sparse executor (DLPack-native, no N^3 materialization), deterministic training, artifact persistence. Beta gate = 20/20 reliability. 50-seed GA gate = 200/200 (436s, `max_attempts=2`). |
| v0.4.0-ga | **Achieved** | Typed batch upload fix (schema-aware I32/I64/U64/Bool/Symbol packing), SLO scaling harness, per-step phase timing, GA preflight all-pass. |
| v0.5.0 | In Progress | GPU-resident loss/credit path (done: zero D2H non-chunked, strict gate, 4 CUDA kernels), GPU-only chunk merge (planned), term embeddings, extended training controls |
| v0.6.0 | Planned | Epistemic logic tier (Phase 7) |
| v0.7+ | Planned | Multi-GPU support, distributed execution (Phase 8) |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| GPU-native dedup complexity | Medium | High | Start with single-key, then multi-key; maintain CPU fallback |
| ~~Float predicate semantics (NaN handling)~~ | ~~Medium~~ | ~~Medium~~ | **Resolved in v0.3.1:** IEEE 754 total ordering implemented |
| Epistemic logic complexity explosion | High | High | Strict tier bounds; bounded iteration limits |
| Multi-GPU synchronization overhead | Medium | High | Start with single-node; benchmark before distributed |
| D4 integration challenges | High | Medium | Plan fallback knowledge compilers (c2d, miniC2D) |

---

## Contributing

See the individual architecture documents in `docs/architecture/` for detailed design information on each subsystem. The CUDA certification suite (`crates/xlog-cuda-tests/`) provides examples of expected kernel behavior.

**Build & Test:**
```bash
# Full workspace test (release mode recommended; exclude the PyO3 extension crate)
cargo test --workspace --all-targets --exclude pyxlog --release

# CUDA certification suite
cargo test -p xlog-cuda-tests --test certification_suite --release

# Run example program
cargo run -p xlog-cli --release -- run examples/xlog/00-basics/01_tc_reachability.xlog
```
