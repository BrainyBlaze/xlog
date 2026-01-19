# XLOG Development Roadmap

> **Last Updated:** January 19, 2026
> **Current Version:** v0.3.2
> **Status:** v0.3.2 Complete — Module system, user-defined functions, reversible symbols, and showcase examples

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
- [x] Symbol type for string values (represented as `u32` hash — not reversible in current version)
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
| `mc_sample.cu` | Bernoulli sampling for Monte Carlo inference |

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

**DLPack (Zero-Copy):**
- [x] Export columns as DLPack managed tensors
- [x] Import DLPack tensors with schema inference
- [x] Import DLPack tensors with schema validation
- [x] Python capsule interface for framework-agnostic GPU data exchange

### Planned 📋

- [ ] Zero-copy Arrow device memory extension
- [ ] Direct cuDF DataFrame interchange
- [ ] Parquet file reading (GPU-accelerated)

---

## Probabilistic Reasoning (`xlog-prob`)

### Implemented ✅

**Exact Inference (`prob_engine=exact_ddnnf`):**
- [x] Provenance extraction from positive Datalog programs
- [x] PIR (Provenance IR) graph construction
- [x] Tseitin encoding to CNF with stable variable mapping
- [x] D4 integration for knowledge compilation (vendored build)
- [x] Decision-DNNF parsing and validation
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

### Limitations (Current Version)

- Exact inference requires positive-only rule bodies (no negation or aggregation)
- Programs with negation/aggregation must use Monte Carlo engine

### Planned 📋

- [ ] Negation support in exact inference via program transformation
- [ ] Aggregate support in probabilistic programs
- [ ] Alternative knowledge compilers (c2d, miniC2D)
- [ ] Importance sampling for rare-event queries
- [ ] Incremental circuit updates for dynamic programs

---

## Epistemic Logic (`xlog-elp`) — Phase 5

### Planned 📋

- [ ] EIR (Epistemic Intermediate Representation) implementation
- [ ] G91 semantics (compatibility mode for classic epistemic logic)
- [ ] FAEEL semantics (default: Founded Autoepistemic Equilibrium Logic)
- [ ] Generate-Propagate-Test algorithm for epistemic evaluation
- [ ] Epistemic splitting for modular evaluation
- [ ] Integration with probabilistic tier for epistemic-probabilistic programs

**Prerequisites:** Phase 4 complete, solver integration
**Estimated effort:** 3–4 months

---

## Solver Services (`xlog-solve`)

### Implemented ✅

- [x] Clause and literal data structures
- [x] CLS (Continuous Local Search) algorithm for SAT/MaxSAT
- [x] Proof artifact generation

### Planned 📋

- [ ] Integration with `xlog-logic` for constraint solving
- [ ] CDCL-based exact solver implementation
- [ ] MaxSAT optimization with soft constraints
- [ ] GPU-accelerated parallel portfolio solving

---

## Python Bindings (`xlog-gpu-py`)

### Implemented ✅

- [x] PyO3-based extension module (`xlog_gpu`)
- [x] DLPack capsule input/output for GPU tensor interchange
- [x] `Program.compile()` for probabilistic programs
- [x] `LogicProgram.compile()` for deterministic programs
- [x] Engine selection: `exact_ddnnf` or `mc`
- [x] Gradient output for learning applications
- [x] Uncertainty metadata for Monte Carlo results
- [x] `dlpack_roundtrip()` helper for interop validation

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

## Multi-GPU & Distributed Execution — Phase 6

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

**Prerequisites:** Phases 4–5 complete, partitioning kernels
**Estimated effort:** 4–6 months

---

## Testing & Validation

### Implemented ✅

- [x] Workspace test suite: `cargo test --workspace --all-targets --release`
- [x] CUDA certification suite: 140/140 tests passing (100%)
- [x] Hash join collision safety tests
- [x] Aggregation overflow/truncation tests
- [x] Large-input filter/compaction tests
- [x] Memory budget enforcement tests
- [x] End-to-end Datalog query tests
- [x] Probabilistic inference correctness tests
- [x] Performance regression benchmarks with CI tracking (Criterion.rs, `.github/workflows/bench.yml`)
- [x] Fuzz testing for parser, compiler, and type inference (cargo-fuzz, ASAN, `.github/workflows/fuzz.yml`)
- [x] Property-based testing for kernel correctness (proptest: sort stability, join correctness, filter idempotence, dedup determinism)
- [x] Float predicate edge case tests (NaN, Infinity, subnormals, signed zeros)

### Planned 📋

- [ ] Integration tests with cuDF and PyTorch

---

## Documentation

### Implemented ✅

- [x] Architecture guide (`docs/ARCHITECTURE.md`)
- [x] Probabilistic tier design (`docs/architecture/xlog-prob.md`)
- [x] Adaptive indexing design (`docs/architecture/adaptive-indexing.md`)
- [x] Multi-GPU join design (`docs/architecture/multi-gpu-join.md`)
- [x] Data interoperability guide (`docs/architecture/cudf-interop.md`)
- [x] Example programs with explanations (`examples/`)
- [x] CUDA certification report (`docs/certification/`)
- [x] API documentation via rustdoc

### Planned 📋

- [ ] Getting started tutorial
- [ ] Language reference manual
- [ ] Performance tuning guide
- [ ] Deployment guide for production use
- [ ] Migration guide for ProbLog/Datalog users

---

## Version History

| Version | Status | Key Features |
|---------|--------|--------------|
| v0.1.0 | Released | Deterministic Datalog, GPU joins/aggregations, basic CLI |
| v0.2.0 | Released | Probabilistic reasoning (exact + MC), Python bindings, GPU-resident execution |
| v0.3.1 | Released | Float predicates (IEEE 754 total ordering), benchmarks, `--stats` flag, fuzz testing, property-based testing |
| v0.3.2 | **Current** | Module system, UDFs, reversible symbols, showcase examples, count→u64 fix |
| v0.3.3–v0.3.5 | Planned | Probabilistic power, solver maturity, ecosystem ready (see v0.3.x-scope.md) |
| v0.4–0.5 | Planned | Epistemic logic tier (Phase 5) |
| v0.6+ | Planned | Multi-GPU support, distributed execution (Phase 6) |

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
# Full workspace test (release mode recommended)
cargo test --workspace --all-targets --release

# CUDA certification suite
cargo test -p xlog-cuda-tests --test certification_suite --release

# Run example program
cargo run -p xlog-cli --release -- run examples/xlog/00-basics/01_tc_reachability.xlog
```
