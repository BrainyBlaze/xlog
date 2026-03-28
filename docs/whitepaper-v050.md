# xlog: A GPU-Native Datalog Engine for Unified Symbolic Reasoning

**Version:** 0.5.0 | **Date:** March 2026

## Abstract

<!-- TODO: Write abstract after all sections are complete (Task 10). -->

## 1 Introduction

Symbolic AI and neural AI have followed divergent engineering paths. Datalog engines, probabilistic logic systems such as ProbLog, and inductive logic programming (ILP) frameworks are implemented as CPU-bound interpreters or compilers, processing relations and proofs in main memory. Deep learning frameworks — PyTorch, JAX — execute dense tensor computations on GPUs via highly optimized CUDA kernels. When researchers combine the two paradigms, as in DeepProbLog or NeurASP, the symbolic component remains on the CPU while the neural component runs on the GPU. Every training iteration transfers data across the PCIe bus: the CPU-side logic engine materializes query results, ships them to the GPU for gradient computation, then pulls gradients back to update symbolic parameters. At scale — millions of ground atoms, thousands of training steps — these host–device transfers dominate wall-clock time and memory bandwidth, becoming the primary bottleneck rather than the inference or learning computation itself.

The gap is architectural. Existing systems address individual reasoning tasks on the GPU in isolation: GPU-accelerated Datalog evaluation (GPUlog, VFLog), GPU SAT solvers, or differentiable logic on CPU with GPU-side neural networks. No single system performs deterministic Datalog evaluation, probabilistic inference via knowledge compilation, SAT/MaxSAT verification, and differentiable neural-symbolic training entirely on the GPU with zero host-device data transfers in production paths. The absence of such a platform forces practitioners into multi-system pipelines — a Datalog engine for rule evaluation, a separate probabilistic reasoner, a Python training loop bridging CPU logic to GPU tensors — each with its own data format, memory model, and failure modes.

xlog addresses this gap with a unified, GPU-native Datalog engine spanning four reasoning paradigms: deterministic logic (semi-naive evaluation with stratified negation), probabilistic inference (exact knowledge compilation and Monte Carlo sampling), SAT/MaxSAT solving (GPU CDCL with proof certificates), and neural-symbolic learning (differentiable training with PyTorch interoperability). The system is implemented in Rust with 21 custom CUDA kernel files (14.2K lines of device code) organized into a layered crate architecture. The compilation pipeline transforms Datalog source into a relational intermediate representation (RIR), lowers probabilistic programs through a propositional intermediate representation (PIR) into CNF, compiles decision-DNNF circuits via D4, and encodes the result in a GPU-resident circuit format (XGCF) for forward and backward evaluation. All semantic data structures — fact stores, circuit nodes, solver state, gradient buffers — remain GPU-resident during execution. Host involvement is limited to orchestration, I/O, and compilation.

The principal contributions of this paper are:

- **GPU-resident semi-naive Datalog evaluation.** The `xlog-runtime` and `xlog-cuda` crates implement relational algebra operators (hash join, radix sort, filter, deduplication, set difference, grouped aggregation) as CUDA kernels, executing fixed-point iteration entirely on the GPU with columnar storage and HISA indexing.

- **GPU-native knowledge compilation pipeline.** The `xlog-prob` crate compiles probabilistic Datalog programs through a PIR-to-CNF-to-D4-to-XGCF pipeline, producing GPU-resident arithmetic circuits with compile-once/evaluate-many semantics. Forward and backward passes over the circuit run as level-parallel CUDA kernels, enabling exact weighted model counting and gradient computation without host transfers.

- **End-to-end differentiable neural-symbolic training.** The `xlog-neural` crate and `pyxlog` Python package connect compiled circuits to PyTorch's autograd graph. Circuit structure depends on the logic program, not on neural network weights, so compiled XGCF templates are cached across training iterations. This circuit caching yields a measured 2.74x end-to-end training speedup (95% CI: [2.29, 3.18]) by eliminating redundant D4 recompilation.

- **Zero-copy interoperability with ML frameworks.** The `xlog-cuda` crate exposes GPU-resident query results and gradient tensors via DLPack capsules and Arrow IPC/C Data interfaces, enabling direct consumption by PyTorch, cuDF, and other frameworks without data copies or device synchronization.

- **Differentiable ILP with GPU-resident credit assignment.** The `pyxlog` dILP trainer implements sparse GPU mask computation, a fully device-resident credit/loss path with zero host transfers, and a six-gate promotion pipeline (convergence, novel-rate audit, regression check, holdout F1 threshold, ambiguity scan, typed schema validation) for transactional rule induction. This subsystem is currently in beta.

The remainder of this paper is organized as follows. Section 2 presents the system architecture and crate decomposition. Section 3 describes GPU-native deterministic Datalog evaluation. Section 4 covers the probabilistic inference pipeline. Section 5 details the neural-symbolic bridge and differentiable training. Section 6 discusses interoperability with external frameworks. Section 7 presents evaluation results. Section 8 surveys related work. Section 9 discusses limitations and future directions.

## 2 Architecture

### 2.1 System Overview

xlog comprises 13 Rust crates organized into a five-tier dependency hierarchy. Every directed edge in the diagram below corresponds to a verified `[dependencies]` entry in the respective crate's `Cargo.toml`. The architecture enforces strict layering: no crate depends on a crate in its own tier or a higher tier.

```
Tier 4  User Interfaces
        ┌────────┐  ┌──────────┐  ┌─────────────────┐
        │ pyxlog │  │ xlog-cli │  │ xlog-cuda-tests │
        └───┬────┘  └────┬─────┘  └───────┬─────────┘
            │             │               │
Tier 3  Integrated Reasoning
        ┌─────────┐  ┌──────────┐
        │ xlog-gpu│  │ xlog-prob│
        └───┬─────┘  └────┬─────┘
            │              │
Tier 2  Subsystems
        ┌───────────┐  ┌──────────────┐  ┌────────────┐
        │ xlog-logic│  │ xlog-runtime │  │ xlog-solve │
        └─────┬─────┘  └──────┬───────┘  └──────┬─────┘
              │               │                  │
Tier 1  Domain IRs + Providers
        ┌────────┐  ┌──────────┐  ┌───────────┐  ┌─────────────┐
        │ xlog-ir│  │ xlog-cuda│  │ xlog-stats│  │ xlog-neural │
        └───┬────┘  └────┬─────┘  └─────┬─────┘  └──────┬──────┘
            │             │              │               │
Tier 0  Foundation
        ┌───────────┐
        │ xlog-core │
        └───────────┘
```

**Tier 0** contains `xlog-core`, the leaf crate that defines shared scalar types (`ScalarType`, `Schema`, `AggOp`), the `KernelProvider` trait, error types, memory budgets, runtime configuration, and a bidirectional `SymbolTable` for dictionary-encoded strings. Every other crate in the workspace depends on `xlog-core`.

**Tier 1** provides domain-specific intermediate representations and service providers that depend only on `xlog-core`. `xlog-ir` defines the relational IR node tree (`RirNode`), expression algebra, and `ExecutionPlan` structures. `xlog-cuda` wraps CUDA device management via cudarc, embeds PTX kernels, implements the `CudaKernelProvider` with modular submodules (relational, filter, group-by, arithmetic, I/O, ILP, probabilistic, transfer, kernel loading), and exposes Arrow IPC/C Data and DLPack interoperability. `xlog-stats` provides `StatsManager` for compiler feedback and runtime tracking. `xlog-neural` defines the `NetworkRegistry`, `NetworkHandle`, `TensorSourceRegistry`, `NeuralBridge`, and `BatchCollector` types for neural-symbolic integration, with an optional PyO3 feature gate for Python interop.

**Tier 2** implements the three core subsystems. `xlog-logic` contains the Datalog frontend: PEG parser (pest), SCC-based stratifier, AST-to-RIR lowerer, cost-aware optimizer, macro expansion, module system, name resolution, type inference, and user-defined functions. Its production dependencies are `xlog-core`, `xlog-ir`, and `xlog-stats`. `xlog-runtime` provides the `Executor` (modular across delta computation, expression evaluation, join caching, node dispatch, recursive evaluation, and rewriting), versioned `RelationStore`, profiling, and query statistics. It depends on `xlog-core`, `xlog-ir`, `xlog-cuda`, and `xlog-stats`. `xlog-solve` implements GPU CDCL verification (complete SAT/UNSAT with on-GPU model and proof validation) and continuous local search for SAT/MaxSAT. It depends on `xlog-core` and `xlog-cuda`.

**Tier 3** composes Tier 2 subsystems into integrated reasoning pipelines. `xlog-gpu` provides a high-level deterministic execution API with input/output buffer management; it depends on `xlog-core`, `xlog-cuda`, `xlog-ir`, `xlog-logic`, and `xlog-runtime`. `xlog-prob` implements the full probabilistic tier — provenance tracking, PIR-to-CNF encoding, GPU-native D4 compilation, XGCF circuit construction, exact inference, Monte Carlo sampling, and circuit caching — depending on `xlog-core`, `xlog-cuda`, `xlog-ir`, `xlog-logic`, `xlog-runtime`, and `xlog-solve`.

**Tier 4** presents user-facing interfaces. `pyxlog` is a PyO3 extension module that bridges eight internal crates (`xlog-core`, `xlog-cuda`, `xlog-gpu`, `xlog-prob`, `xlog-logic`, `xlog-neural`, `xlog-runtime`, `xlog-ir`) to Python, exposing DLPack-first evaluation, training APIs, and ILP/dILP trainers. `xlog-cli` provides the `xlog` command-line binary for deterministic and probabilistic execution with Arrow IPC I/O. `xlog-cuda-tests` is an unpublished certification test suite that exercises `xlog-cuda`, `xlog-core`, and `xlog-solve` for release gating.

### 2.2 Compilation Pipeline

The compilation pipeline transforms Datalog source text into a GPU-executable plan through five stages: parsing, stratification, lowering, optimization, and plan assembly.

**Parsing.** The PEG parser in `crates/xlog-logic/src/parser.rs` uses the pest parser generator with a grammar defined in `grammar.pest`. It produces an abstract syntax tree (AST) containing `Program`, `Rule`, `Atom`, `Term`, `BodyLiteral`, and related nodes defined in `crates/xlog-logic/src/ast.rs`. The parser handles standard Datalog syntax extended with probabilistic facts (`p::f.`), annotated disjunctions, neural predicate declarations (`nn/4`), evidence assertions, aggregation expressions, constraints, user-defined functions, and module imports.

**Stratification.** The stratifier in `crates/xlog-logic/src/stratify.rs` builds a predicate dependency graph with three edge types — positive, negative, and aggregate — then computes strongly connected components (SCCs) using Tarjan's algorithm. If any SCC contains a negative or aggregate edge, the program is rejected with a diagnostic error: stratified negation and stratified aggregation require that no recursion passes through negation or aggregation. The output is an ordered sequence of strata, each containing one or more SCCs that can be evaluated before any stratum that depends on them.

**Lowering.** The `Lowerer` in `crates/xlog-logic/src/lower.rs` translates each AST rule into a `RirNode` tree defined in `crates/xlog-ir/src/rir.rs`. Body literals become `Scan` nodes; variable bindings across atoms produce `Join` nodes with key columns derived from shared variable positions; negated literals become `Anti` joins; filters and arithmetic constraints become `Filter` nodes with `Expr` predicates (supporting comparison, Boolean connectives, arithmetic, built-in functions, and conditional expressions). The lowerer also emits `Fixpoint` nodes that wrap recursive SCC bodies for semi-naive iteration, and `GroupBy` nodes for stratified aggregation.

**Optimization.** The optimizer in `crates/xlog-logic/src/optimizer.rs` applies cost-aware transformations using runtime statistics from `xlog-stats::StatsManager`. It performs predicate pushdown — relocating `Filter` nodes below `Project` and `Join` nodes to reduce intermediate result sizes — and cost-based join ordering via dynamic programming. For queries involving more relations than a configurable threshold (`dp_threshold`, default 10), the optimizer falls back to a greedy heuristic to avoid exponential planning time. Cost estimates account for expected row counts, GPU memory consumption, and data transfer volume.

**Plan assembly.** The final output is an `ExecutionPlan` (defined in `crates/xlog-ir/src/plan.rs`) containing SCCs in dependency order, strata for negation ordering, and compiled rule trees grouped by SCC. The `xlog-runtime` executor interprets this plan by dispatching each `RirNode` to the appropriate CUDA kernel via the `CudaKernelProvider`.

### 2.3 GPU Residency Model

xlog enforces a hard guarantee: all runtime semantic state resides in GPU device memory, or the system returns a deterministic error. There is no silent out-of-core spilling, and there is no implicit CPU fallback. If a workload exceeds the GPU memory budget, xlog raises a `RESOURCE_EXHAUSTED` error with diagnostic information rather than degrading to host memory transparently.

The residency contract extends to production query paths, where the system targets zero device-to-host (D2H) transfers. The `CudaKernelProvider` in `crates/xlog-cuda/src/provider/mod.rs` implements byte-level transfer accounting through a `HostTransferStats` structure that tracks `dtoh_bytes` and `htod_bytes` via atomic counters. Callers invoke `reset_host_transfer_stats()` before a performance-critical section and `host_transfer_stats()` afterward to obtain a snapshot. A separate `d2h_transfer_count` atomic counter increments once per `download_column_*` call, enabling assertions — particularly in the ILP trainer — that no column downloads occurred during a device-resident computation. The `reset_d2h_transfer_count()` method zeroes this counter for bracketed verification.

This design matters for performance. PCIe 4.0 x16 delivers roughly 32 GB/s peak bidirectional bandwidth, while GPU HBM provides 1--3 TB/s depending on the device. A single unnecessary D2H round-trip for a moderate relation (e.g., 10M tuples at 16 bytes each, roughly 160 MB) costs approximately 5 ms of bus transfer latency — enough to dominate a semi-naive iteration step that completes in microseconds on-device. By keeping fact stores, delta relations, circuit node values, solver state, and gradient buffers exclusively on the GPU, xlog eliminates this class of bottleneck. Host involvement is restricted to orchestration (launching kernels, managing plan execution order), I/O (reading source files, writing final results), and compilation (parsing, stratification, lowering, and optimization all run on the CPU before execution begins).

An explicit opt-in out-of-core mode (`--allow-ooc`) exists for workloads that exceed device memory. When enabled, xlog may spill immutable intermediates to host-pinned memory, but this mode is never activated silently and carries no performance guarantees.

### 2.4 IR Stack

xlog uses a layered intermediate representation stack designed for extensibility across reasoning paradigms. Each IR level serves a distinct role in the compilation and evaluation pipeline.

**AST** (Abstract Syntax Tree). The parser produces a concrete syntax tree that preserves source-level structure: rules, atoms, terms, probabilistic annotations, neural predicate declarations, and directives. The AST lives in `crates/xlog-logic/src/ast.rs` and carries no execution semantics.

**RIR** (Relational IR). The lowerer transforms the AST into an algebraic tree of `RirNode` variants defined in `crates/xlog-ir/src/rir.rs`. The key variants are: `Scan` (base relation access), `Filter` (predicate evaluation), `Project` (column selection and computation via `ProjectExpr`), `Join` (inner, left-outer, semi, and anti joins with explicit key columns), `GroupBy` (stratified aggregation), `Union`, `Distinct`, `Diff` (set difference for semi-naive delta computation), and `Fixpoint` (recursive SCC wrapper). RIR nodes carry metadata including estimated cardinality ranges, memory peak estimates, skew hints for join partitioning, and incremental update semantics (delta vs. full materialization).

**PIR** (Provenance IR). For probabilistic programs, `xlog-prob` constructs a `PirGraph` (defined in `crates/xlog-prob/src/pir.rs`) whose `PirNode` variants — `Lit`, `NegLit`, `And`, `Or`, `Decision` — represent the weighted Boolean formula derived from provenance tracking over the ground program. PIR captures the probabilistic structure needed for knowledge compilation without encoding execution order.

**XGCF** (xlog GPU Circuit Format). The final compiled form is a levelized DAG stored in `crates/xlog-prob/src/xgcf.rs` as an `Xgcf` structure. Each node has a type (`Const0`, `Const1`, `Lit`, `And`, `Or`, `Decision`), child indices, and variable/literal identifiers. Level offsets enable parallel evaluation: all nodes at the same topological level execute in a single kernel launch. Forward passes compute log-space values; backward passes propagate adjoints for gradient computation. The `GpuXgcf` structure in `crates/xlog-prob/src/gpu.rs` holds the device-resident buffers. Multiple circuits can be batched via `circuit_offsets` for fused evaluation across neural batch dimensions.

**Extensibility.** The IR stack was designed to accommodate additional representations as the system grows. The architecture documentation describes two planned IRs: **EIR** (Epistemic IR) for world-view reasoning with modal operators, split plans, and guess spaces; and **SIR** (Solver IR) for Boolean satisfiability encoding with CNF clauses, cardinality constraints, weight vectors, and proof policies. Neither EIR nor SIR is implemented in v0.5.0. The layered design ensures that adding a new IR requires implementing a lowering pass from RIR (or from another IR at the same level) and a backend that targets either XGCF or a new GPU-resident evaluation format, without modifying existing IR definitions or evaluation paths.

## 3 GPU-Native Datalog Execution

The deterministic evaluation engine executes standard Datalog with stratified negation and aggregation entirely on the GPU. The algorithmic approach — semi-naive fixpoint iteration over stratified programs — is well established. The contribution here is not algorithmic novelty but engineering: every relational operator runs as a custom CUDA kernel, delta relations are maintained on-device, and no host–device transfers occur during evaluation.

### 3.1 Semi-Naive Evaluation on GPU

The executor in `crates/xlog-runtime/src/executor/` processes an `ExecutionPlan` consisting of strata ordered by the dependency analysis from Section 2.2. Each stratum contains one or more strongly connected components (SCCs), and the executor dispatches each SCC according to whether it is recursive or non-recursive.

**Non-recursive SCCs.** The executor evaluates each compiled rule once, passing the RIR tree to the node dispatcher which invokes the appropriate CUDA kernel for each operator (scan, join, filter, project, group-by). Results for the same head predicate are merged via the `union_gpu` kernel and deduplicated. No iteration is needed.

**Recursive SCCs.** The `execute_recursive_scc` method in `crates/xlog-runtime/src/executor/recursive.rs` implements semi-naive evaluation. The algorithm proceeds in three phases:

1. **Seeding.** All rules in the SCC are evaluated once against the current relation store to produce initial results. Per-predicate delta relations are allocated as distinct GPU-resident buffers with dedicated `RelId` identifiers (named `__delta_{pred}_{id}`). The initial delta for each predicate is computed as the set difference between the newly derived tuples and any pre-existing tuples.

2. **Iteration.** On each iteration, the executor re-evaluates rules using delta-rewritten variants. For each rule, it identifies recursive scan occurrences and rewrites each one individually to reference the corresponding delta relation, producing one evaluation variant per recursive scan site. This per-occurrence rewriting handles self-joins correctly: a rule body containing `p(X,Y), p(Y,Z)` generates two variants, each substituting the delta into exactly one scan. Variant results are unioned per head predicate. The raw delta is then differenced against the full relation (`delta_new = delta_raw - full`) and deduplicated, all on-device. A `DeltaRelationTracker` records whether any predicate produced new tuples; if none did, fixpoint has been reached.

3. **Merge and cleanup.** After each iteration, new delta tuples are merged into the full relation via `union_gpu` followed by `dedup_sorted`. When the fixpoint is reached or the configurable iteration limit (default 1000, set via `RuntimeConfig.max_iterations`) is exceeded, delta relations are removed from the store and their `RelId` mappings are unregistered.

All intermediate buffers — full relations, delta relations, union results, difference results — remain GPU-resident throughout. The profiler records per-operator timing, input/output row counts, and peak GPU memory at each step, enabling the feedback loop described in Section 3.3.

### 3.2 Kernel Design

xlog implements eight core relational CUDA kernels, totaling 3,259 lines of device code. Each kernel is a direct CUDA implementation, not a wrapper around cuBLAS, cuSPARSE, or Thrust.

| Kernel | LOC | Purpose |
|--------|-----|---------|
| `join.cu` | 361 | Hash join with linked-list collision chains. Uses FNV-1a composite hashing for multi-column keys. All scalar types supported by hashing raw bytes. |
| `sort.cu` | 452 | 4-bit radix sort (8 passes over 32-bit keys). Stable: elements with equal digits preserve input order. Three-phase per pass: histogram, prefix sum, scatter. |
| `filter.cu` | 942 | Comparison operators (Eq, Ne, Lt, Le, Gt, Ge) for column-vs-constant and column-vs-column, with stream compaction via prefix sum. Supports all scalar types including fused compare-and-scan variants. |
| `groupby.cu` | 248 | Sorted-input grouped aggregation. Detects group boundaries in sorted key columns, then applies per-group reduction (Count, Sum, Min, Max, LogSumExp). |
| `dedup.cu` | 289 | Sort-based deduplication with prefix-sum compaction. Type-aware equality including IEEE 754 float handling: treats -0.0 and +0.0 as equal, NaN equal only when bit-identical. |
| `set_ops.cu` | 116 | Union (concat + sort + dedup) and set difference via sorted-array binary search. Columnar concat kernel copies both inputs into a pre-allocated output buffer. |
| `scan.cu` | 270 | Blelloch parallel prefix sum (inclusive). Block-level scan with shared memory, inter-block propagation via block sums. Used as a building block by filter compaction, dedup, and group-by. |
| `pack.cu` | 581 | Multi-column key packing on device. Packs up to 4 separate column buffers into row-major byte arrays and computes FNV-1a hashes, eliminating host roundtrips for multi-column join key preparation. |

**IEEE 754 total ordering.** The filter kernel implements a correctness-critical design choice for floating-point comparisons. Equality and inequality (Eq, Ne) use standard IEEE 754 semantics where NaN != NaN. Ordered comparisons (Lt, Le, Gt, Ge) use a total ordering transformation: the `float_to_ordered_f64` device function converts an `f64` bit pattern to an `i64` such that the resulting integer order matches the IEEE 754 `totalOrder` predicate. The ordering is: -NaN < -Inf < negative finites < -0.0 < +0.0 < positive finites < +Inf < +NaN. The transformation works by flipping all bits except the sign bit for negative values and flipping only the sign bit for non-negative values, matching Rust's `f64::total_cmp()` algorithm. An analogous `float_to_ordered_f32` function handles 32-bit floats. This ensures that Datalog programs produce deterministic, well-defined results for all floating-point inputs, including edge cases involving NaN and signed zeros.

### 3.3 Adaptive Join Planning

The optimizer in `crates/xlog-logic/src/optimizer.rs` performs cost-based query transformations using runtime statistics collected by the `StatsManager` in `crates/xlog-stats/src/manager.rs`.

**Cost model.** Each plan node receives a `PlanCost` estimate with four dimensions: expected row count, CPU coordination cost, peak GPU memory, and number of host–device transfers. The scalar cost function weights these components, with transfers penalized heavily (default multiplier: 100x) to reflect PCIe latency. GPU memory cost is scaled at 0.001 per byte, making 1 GB equivalent to 1M cost units.

**Predicate pushdown.** When enabled (the default), the optimizer pushes `Filter` nodes below `Project` and `Join` nodes. Filters above joins are decomposed: predicates referencing only left-side columns are pushed into the left input, predicates referencing only right-side columns into the right input, and cross-predicates remain above the join. Filters above projections are remapped through pass-through columns where possible.

**Join ordering.** For queries involving up to `dp_threshold` relations (default: 10), the optimizer uses dynamic programming to enumerate join orderings and select the minimum-cost plan. Join cardinality estimation uses cached selectivity data from `StatsManager.estimate_join_cardinality`, which multiplies left and right cardinalities by a selectivity factor derived from prior executions. When the relation count exceeds the threshold, the optimizer falls back to a greedy heuristic to avoid exponential planning time.

**Feedback loop.** The `Executor` exposes a `stats_snapshot()` method that captures a `StatsSnapshot` containing per-relation cardinality, byte size, access heat, column-level statistics, join selectivities, and a `RelId`-to-predicate-name mapping. This snapshot can be fed back into the `StatsManager` via `merge_snapshot()` before the next compilation, closing the loop between execution and optimization. The predicate-name mapping ensures that statistics are applied correctly across compilations even when `RelId` assignments change.

### 3.4 Reversible Symbols

Datalog programs operate on strings (predicate arguments, constants), but GPU kernels operate on fixed-width numeric columns. xlog bridges this gap with a global symbol table implemented in `crates/xlog-core/src/symbol.rs`.

**Interning.** The `intern(s: &str) -> u32` function assigns sequential 32-bit IDs to unique strings. A read-write lock (`RwLock<SymbolRegistry>`) guards the bidirectional mapping: a `HashMap<String, u32>` for forward lookup and a `Vec<String>` for reverse resolution. The fast path acquires only a read lock; the slow path (new symbol) double-checks after acquiring the write lock to avoid races. Sequential allocation means IDs are dense and start at 0, which benefits GPU memory access patterns.

**Reverse resolution.** The `resolve(id: u32) -> String` function recovers the original string from an ID in O(1) via the vector index. This is used at query output time to present human-readable results.

**Arrow dictionary encoding.** The `ScalarType::Symbol` variant maps to Arrow's `Dictionary(UInt32, Utf8)` type. The `to_arrow` function builds a `DictionaryArray` from a slice of symbol IDs: it collects unique IDs, resolves them to strings to form the dictionary, and remaps keys. The inverse `from_arrow` function interns all dictionary strings and maps Arrow keys back to global symbol IDs. This encoding enables zero-copy export to frameworks that consume Arrow data (cuDF, Polars, DuckDB) while preserving the compact u32 representation used internally.

**GPU representation.** On the GPU, symbol columns are stored as contiguous `u32` arrays — the same representation as any other 4-byte integer column. Join, sort, filter, and dedup kernels operate on these integer IDs without special-casing. The symbol table itself remains on the host, since string data is variable-length and only needed at ingestion and output boundaries.
