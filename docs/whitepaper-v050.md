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

## 4 Probabilistic Inference on GPU

### 4.1 Approach

Probabilistic logic programming systems such as ProbLog compute the probability of queries over programs containing probabilistic facts. The standard technique is knowledge compilation: compile the ground Boolean formula derived from provenance tracking into a tractable circuit (typically a Decision-DNNF), then evaluate weighted model counts over that circuit. ProbLog follows a compile-once/evaluate-many model: the circuit structure is determined by the logical program and is independent of the probability weights, so a single compilation can serve many weight configurations during parameter learning.

xlog adopts the same compile-once/evaluate-many model but moves the entire pipeline onto the GPU. In ProbLog, provenance extraction, CNF encoding, D4 compilation, and circuit evaluation all execute on the CPU. When ProbLog is paired with a neural component (as in DeepProbLog), each training iteration requires transferring circuit evaluation results from CPU to GPU for gradient computation, and pulling gradients back for weight updates. xlog eliminates this architectural split: PIR extraction, Tseitin CNF encoding, D4 compilation, CDCL equivalence verification, and forward/backward circuit evaluation all execute as CUDA kernels on the GPU. The compiled circuit is a GPU-resident data structure (XGCF), weight buffers and gradient buffers are GPU-resident, and no host--device transfers occur in the evaluation hot path. This design turns what would be a PCIe-bottlenecked ping-pong between CPU logic engine and GPU tensor framework into a single-device pipeline.

### 4.2 GPU Knowledge Compilation Pipeline

The probabilistic compilation pipeline transforms a Datalog program with probabilistic annotations into a GPU-resident arithmetic circuit through five stages. Each stage is implemented as one or more CUDA kernel launches, with intermediate data structures remaining in device memory throughout.

```
Probabilistic Program
    |
    v
[PIR Extraction]      pir.cu       (600 LOC)
    |
    v
[CNF Encoding]        cnf.cu       (623 LOC)
    |
    v
[D4 Compilation]      d4.cu      (2,953 LOC)
    |
    v
[CDCL Verification]   sat.cu     (3,268 LOC)
    |
    v
[XGCF Circuit]
    |
    v
[Forward / Backward]  circuit.cu (1,632 LOC)
```

The five kernel files total 9,076 lines of CUDA device code. Together with the eight relational kernels from Section 3.2, they account for the majority of xlog's 14.2K-line CUDA codebase.

**Stage 1: PIR Extraction** (`pir.cu`, 600 LOC; host orchestration in `gpu_pir_intern.rs`). The PIR (Provenance IR) stage constructs a weighted Boolean formula from the ground program. Provenance tracking over the deterministic evaluation produces a `PirGraph` whose node types are `Lit`, `NegLit`, `And`, `Or`, and `Decision`. The GPU PIR interner performs device-side hash-consing: it uploads node batches to the GPU, computes FNV-1a hashes over node signatures (type, leaf ID, child lists), and deduplicates structurally identical subformulas using a GPU hash table with atomic insert operations. This deduplication is critical because grounding can produce exponentially many redundant subformulas; interning on device avoids materializing the full uninterned graph in host memory. The interner outputs a `GpuPirGraph` containing device-resident arrays for node types, child offsets, child indices, leaf IDs, and decision variable metadata.

**Stage 2: CNF Encoding** (`cnf.cu`, 623 LOC; host orchestration in `gpu_cnf.rs`). The CNF encoder transforms the PIR graph into a conjunctive normal form formula using the Tseitin transformation, executed entirely on GPU. The kernel pipeline proceeds in six phases: (1) BFS reachability from query roots to identify live nodes; (2) classification and prefix-sum counting of leaf variables (probabilistic facts) and choice variables (annotated disjunction selectors); (3) variable assignment via exclusive scans that produce compact, gap-free variable IDs; (4) Tseitin clause counting per node (AND nodes produce implication clauses, OR nodes produce reverse-implication clauses, Decision nodes produce if-then-else clauses); (5) clause and literal offset computation via parallel prefix sum; (6) clause emission into a GPU-resident CSR (compressed sparse row) representation. The output is a `GpuCnfEncoding` containing the CSR clause structure (`GpuCnf`) and variable mapping tables that link PIR node IDs to CNF variable IDs. The encoder also computes a `decision_var_limit` that separates semantically meaningful variables (leaf and choice variables eligible for branching) from internal Tseitin variables used only for unit propagation.

**Stage 3: D4 Compilation** (`d4.cu`, 2,953 LOC; host orchestration in `gpu_d4/mod.rs` and `gpu_d4/build.rs`). The D4 compiler transforms the GPU-resident CNF into a Decision-DNNF circuit stored in the XGCF format. This is the most complex stage, implementing a GPU-native variant of the D4 algorithm. The compiler uses a hybrid BFS/DFS strategy controlled by `GpuCompileConfig`: it performs BFS expansion to a configurable frontier depth to expose parallelism, then hands each frontier item to a per-block DFS worker. Each DFS worker performs component detection (identifying independent sub-CNFs via variable connectivity), decision variable selection (using VSIDS-like activity scores), and recursive decomposition. The compiler emits circuit nodes — `Const0`, `Const1`, `Lit`, `And`, `Or`, `Decision` — into a flat device-resident array. After initial compilation, a GPU smoothing pass ensures that all branches of each OR/Decision node mention the same set of random variables, which is required for correct weighted model counting. The smoothed circuit is levelized: nodes are sorted into topological levels via a BFS from leaves, with `level_offsets` and `level_nodes` arrays enabling level-parallel evaluation. The output is a `GpuXgcf` structure ready for forward and backward passes. Configuration parameters include `max_frontier_items` (hard cap on BFS work items), `max_depth` (defensive recursion limit), and `smooth_node_cap`/`smooth_edge_cap` (bounds on smoothing pass output).

**Stage 4: CDCL Verification** (`sat.cu`, 3,268 LOC; host orchestration in `gpu_cdcl.rs` and `compilation/validation.rs`). After D4 produces a circuit C from the CNF formula phi, the pipeline must verify that the circuit is semantically equivalent to the original formula: phi equiv C. This verification is not a heuristic check or a sampling-based confidence test -- it is a complete formal proof. The verifier constructs two equivalence queries on the GPU: q1 = phi AND NOT(C) and q2 = C AND NOT(phi). If both queries are unsatisfiable, then phi and C agree on every possible variable assignment, establishing logical equivalence. Each query is solved by the GPU CDCL (Conflict-Driven Clause Learning) solver, which implements a full SAT solver as a single-block CUDA kernel (256 threads) with watched-literal propagation, 1-UIP conflict analysis, VSIDS decision heuristic, deterministic restarts, and periodic clause-database reduction. The solver operates entirely on device-resident data: variable assignments, decision levels, reason clauses, activity scores, the trail, watched-literal lists, learned clause arena, and resolution proof trace are all GPU-resident buffers. For UNSAT results, the kernel produces a resolution proof trace that is verified on-device by a separate proof-checking kernel (`sat_proof_mark_needed` + `sat_proof_check` + `sat_assert_ok`), ensuring that the UNSAT claim is backed by a machine-checkable certificate. The `solve_expect_unsat` family of methods enforce the UNSAT expectation on GPU without any device-to-host status reads -- if the solver returns SAT when UNSAT was expected, the kernel triggers a device-side trap. Pre-allocated `GpuCdclWorkspace` structures amortize the allocation of the solver's 30+ device buffers across multiple solves.

**Stage 5: Circuit Evaluation** (`circuit.cu`, 1,632 LOC; host orchestration in `gpu.rs`). The compiled and verified XGCF circuit supports two evaluation modes: a forward pass for weighted model counting and a backward pass for gradient computation. The forward pass iterates over topological levels from leaves to root, launching one `xgcf_forward_level` kernel per level. Each kernel thread processes one node: `Const0` nodes receive log-value negative infinity, `Const1` nodes receive log-value 0, `Lit` nodes look up the log-weight of their variable (positive or negative literal), `And` nodes sum their children's log-values, `Or` nodes compute log-sum-exp over their children's log-values, and `Decision` nodes combine their true-branch and false-branch values weighted by the decision variable's log-weights. All arithmetic is in log-space to avoid floating-point underflow. The root node's value after the forward pass is log(Z), the log partition function (or log(Z|evidence) when evidence constraints are applied). The backward pass iterates in reverse topological order, launching three kernels per level: `xgcf_backward_level_propagate` distributes adjoints from parent nodes to children (additive for OR/Decision, pass-through for AND), `xgcf_backward_level_decision_grad` accumulates gradients for decision variables, and `xgcf_backward_level_lit_grad` accumulates gradients into per-variable `grad_true` and `grad_false` buffers. These gradient buffers are GPU-resident and can be consumed directly by PyTorch's autograd graph via DLPack (Section 5).

### 4.3 Circuit Caching

D4 compilation is the most expensive stage in the pipeline. On the MNIST addition benchmark (`01_minimal`, 512 training examples), a cold-start compilation takes approximately 75 seconds. Since this cost is incurred at the beginning of each training epoch in a naive implementation, it can dominate total training time.

The key insight enabling circuit caching is that the XGCF circuit structure depends on the logic program -- specifically, on the propositional structure of the grounded rules and the evidence pattern -- but not on the numerical values of the probability weights. When neural network parameters change between training iterations, the weights assigned to probabilistic facts change, but the Boolean formula and its compiled Decision-DNNF structure remain identical. This means the compiled XGCF can be cached and reused across all training iterations, with only the weight buffers updated before each forward pass.

The `GpuCircuitCache` in `crates/xlog-prob/src/compilation/gpu_cache.rs` implements this caching. On the first evaluation, the full pipeline runs (PIR extraction, CNF encoding, D4 compilation, CDCL verification), and the resulting `GpuXgcf` is stored in the cache keyed by a structural hash of the PIR graph and evidence configuration. On subsequent evaluations, the cache returns the existing circuit, and only the weight-upload and forward/backward passes execute. The cache also stores the CNF variable tables (`GpuCnfVarTables`) needed to map between PIR leaf IDs and CNF variable IDs for weight construction.

Measured impact: on the MNIST addition benchmark (3 epochs, 512 training examples, 3 seeds), cache-enabled training completes in a mean of 88.89 seconds (std: 3.48s), compared to 242.90 seconds (std: 8.19s) with caching disabled. This yields a 2.74x end-to-end speedup (95% CI: [2.29, 3.18]). The speedup is large because D4 compilation cost is amortized across all epochs: after the initial cold start, each subsequent epoch executes only the weight update and circuit evaluation, reducing per-epoch steady-state time from approximately 75 seconds to sub-second forward/backward passes.

### 4.4 Monte Carlo Sampling

For programs where exact inference via knowledge compilation is infeasible -- either because the ground formula is too large for D4 compilation within the available GPU memory, or because the user prefers approximate results with bounded computation -- xlog provides a GPU-parallel Monte Carlo sampling engine implemented in `crates/xlog-prob/src/mc/`.

The MC engine supports two sampling methods, selectable via `McSamplingMethod`:

- **Rejection sampling.** The sampler draws values for all probabilistic facts from their prior distributions on the GPU, then evaluates the deterministic Datalog program in each sampled world. Worlds where the evidence is not satisfied are discarded. Query probabilities are estimated as the fraction of accepted worlds in which each query atom is derived. This method is unbiased but can be inefficient when the evidence has low prior probability, as most samples are rejected.

- **Evidence clamping.** When all evidence atoms correspond to Bernoulli probabilistic facts, the sampler can force those facts to their observed values, guaranteeing that every sample satisfies the evidence. This eliminates rejection waste entirely: the `McCountStrategy` switches to `QueriesOnly` mode, skipping evidence-side buffer allocations and truth-kernel evidence checks. Evidence clamping is auto-selected when the `EvidenceForcing` analysis determines that all evidence is forceable (i.e., each evidence atom maps to a single probabilistic fact with no intermediate rules).

Both methods report confidence intervals for estimated probabilities. The sampling loop is structured to minimize per-sample overhead: probabilistic fact values are sampled in bulk on the GPU via the `sampling` submodule, the relation store is reset using a targeted `McSampleResetPlan` that preserves pure deterministic relations and clears only dynamic relations (avoiding a full store clone per sample), and per-sample pointer buffers are pre-allocated outside the sample loop. Optimizations documented in the MC runtime optimization report (March 2026) reduced the per-1000-sample wall-clock time from 14.11 seconds to 12.90 seconds (8.6% improvement) by eliminating hot-loop device-to-host synchronizations for row counts, replacing full store clone/restore with targeted resets, and reducing clamped-mode count-path overhead. Profiling shows that fixpoint evaluation of the deterministic program dominates at 72--83% of total MC time, with the sampling, reset, build, and count phases accounting for the remainder.

### 4.5 Well-Founded Semantics

Standard stratified negation requires that no recursive cycle passes through a negated literal. Programs that violate this constraint -- such as the classic `p :- not q. q :- not p.` -- have no two-valued stable model and are rejected by most Datalog engines. xlog handles such non-monotone programs via Well-Founded Semantics (WFS), implemented in `crates/xlog-prob/src/wfs.rs`.

WFS assigns three-valued truth to atoms: true, false, or undefined. The algorithm computes an alternating fixed point: (1) find the greatest unfounded set (atoms that cannot be supported by any rule), mark them false; (2) derive all consequences of the current knowledge, mark them true; (3) repeat until stable. Atoms that remain neither true nor false are classified as undefined. For probabilistic programs, true atoms receive normal probability and gradient computation, while false and undefined atoms are assigned probability zero with zero gradients -- matching ProbLog's conservative treatment of non-stratified programs.

WFS is invoked automatically during provenance extraction when a non-monotone SCC is detected. The Monte Carlo engine also handles non-monotone programs: when cycles are detected during per-sample fixpoint evaluation, the MC engine uses the intersection of all states in the cycle (skeptical, invariant tuples only) as the interpretation, providing a deterministic semantics that avoids parity dependence on iteration count. An example program exercising this path is `examples/prob/04-nonmonotone-mc.xlog`, which defines the mutual-negation cycle `p :- flip. q :- not p. p :- not q.` and queries both `p` and `flip` under Monte Carlo inference.

## 5 Neural-Symbolic Bridge

The previous sections describe xlog as a purely symbolic system: Datalog evaluation on the GPU (Section 3), probabilistic inference through knowledge compilation (Section 4). This section introduces the neural-symbolic bridge -- the machinery that connects PyTorch neural networks to the probabilistic logic engine, enabling end-to-end differentiable learning where neural perception and logical reasoning are jointly trained.

### 5.1 Neural Predicates

In xlog, a neural network *is* a predicate. The `nn/4` declaration embeds a network directly into the Datalog program:

```
nn(network_name, [input_vars], output_var, [output_labels]) :: predicate(args).
```

This is not a foreign-function call or an escape hatch. The declaration states that evaluating predicate `predicate(args)` requires a forward pass through the named network. The network's softmax output over the label set becomes a distribution over probabilistic facts, which then participates in the standard knowledge compilation pipeline described in Section 4.

On the Python side, networks are registered via the `pyxlog` API. The `CompiledProgram.register_network()` method accepts a PyTorch `nn.Module`, an optimizer, and optional configuration parameters:

```python
program.register_network("mnist_net", net, optimizer)
```

Under the hood, the Rust-side `NetworkRegistry` (in `crates/xlog-neural/src/registry.rs`) manages a collection of `NetworkHandle` instances, each holding a reference to the PyTorch module, optimizer, and optional learning rate scheduler. The `NetworkConfig` struct exposes builder-pattern configuration for batching (grouping multiple queries into a single forward pass), top-k sampling (restricting the output space when most classes have near-zero probability), deterministic mode (argmax instead of probabilistic sampling), and output caching (avoiding redundant forward passes for repeated inputs, up to a configurable cache size of 10,000 entries by default).

The data flow through a neural predicate proceeds in two directions. **Forward:** when the logic engine encounters an `nn/4`-backed predicate during evaluation, it invokes the registered PyTorch module. The network's softmax output vector is decomposed into weighted probabilistic facts -- one per output label -- which are fed into the PIR/CNF/D4/XGCF compilation pipeline. Because the circuit structure depends on the logic program and not on the weight values, the compiled XGCF template is cached (Section 4.3) and only the leaf weights are updated on subsequent iterations. **Backward:** after the XGCF circuit computes the forward log-probability and backward gradients via level-parallel CUDA kernels, the per-leaf gradient buffers are exported as DLPack tensors and injected into PyTorch's autograd graph. The optimizer then updates the network parameters as usual. The Python GIL is released during GPU-resident work using `py.allow_threads()`, ensuring that CUDA kernel execution and GPU memory operations do not block the Python interpreter.

### 5.2 End-to-End Training Loop

To make the integration concrete, we walk through the MNIST addition example -- the canonical DeepProbLog benchmark. The task: given two MNIST digit images, predict their sum. The network never sees individual digit labels; it learns to classify digits purely from supervision on sums.

**The logic program.** The complete xlog source (`examples/neural/01_minimal/minimal.xlog`) is:

```
// Neural predicate declaration
// nn(network_name, [input_vars], output_var, [output_labels]) :: predicate(args).
//
// This declares that:
// - 'mnist_net' is the name of the neural network
// - [X] means the network takes one input (image at index X)
// - Y is the output variable (the classified digit)
// - [0,1,2,3,4,5,6,7,8,9] are the possible output labels (digit classes)
// - digit(X, Y) is the resulting predicate
//
// The network's softmax output becomes probabilistic facts:
//   P(digit(0, 0)) = network_output[0]
//   P(digit(0, 1)) = network_output[1]
//   ...
//   P(digit(0, 9)) = network_output[9]
nn(mnist_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).

// Addition rule
// The sum of two digits at indices X and Y is Z.
//
// This uses probabilistic inference:
// P(addition(0, 1, 7)) = sum over all (d1, d2) where d1 + d2 = 7 of:
//                        P(digit(0, d1)) * P(digit(1, d2))
//
// For example, with d1=3, d2=4:
// P(addition(0, 1, 7)) includes P(digit(0, 3)) * P(digit(1, 4))
addition(X, Y, Z) :- digit(X, D1), digit(Y, D2), Z is D1 + D2.
```

Two lines of logic define the entire reasoning structure. The `nn/4` declaration connects the `mnist_net` CNN to a `digit/2` predicate. The addition rule computes all consistent decompositions of a sum through probabilistic marginalization.

**The training script.** The Python driver (`examples/neural/01_minimal/train.py`) compiles the program, registers the network, and runs training through the `pyxlog.train_model_tensor` API:

```python
program = pyxlog.Program.compile("""
    nn(mnist_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).
    addition(X, Y, Z) :- digit(X, D1), digit(Y, D2), Z is D1 + D2.
""")

program.register_network("mnist_net", net, optimizer)
program.add_tensor_source("train", train_images)

queries = generate_queries(n_pairs, train_labels)

history = pyxlog.train_model_tensor(
    program,
    queries,
    epochs=args.epochs,
    batch_size=args.batch_size,
    log_iter=args.log_iter,
)
```

Each query has the form `addition(i, j, expected_sum)`. The training objective minimizes the negative log-probability of the query under the compiled circuit: `-log P(addition(i, j, s))`. The `train_model_tensor` function orchestrates the full loop: batching queries that share the same circuit template, running forward and backward passes over the cached XGCF circuit, and propagating gradients back to the PyTorch optimizer.

**Training loop architecture.** The following diagram shows one training iteration:

```
  PyTorch Forward Pass
  (CNN processes digit images)
          |
          v
  Softmax outputs become leaf weights
  in the probabilistic circuit
          |
          v
  ┌─────────────────────────────────┐
  │  XGCF Circuit (cached on GPU)  │
  │  - Compiled once from D4       │
  │  - Reused across all epochs    │
  │  - Only leaf weights updated   │
  └─────────────────────────────────┘
          |
    Forward pass: level-parallel CUDA kernels
    compute log P(query)
          |
          v
  NLL Loss = -log P(addition(i, j, s))
          |
    Backward pass: level-parallel CUDA kernels
    compute d(loss)/d(leaf_weight)
          |
          v
  Gradients exported via DLPack
  (zero-copy GPU tensor sharing)
          |
          v
  PyTorch Backward + Optimizer Step
  (Adam updates CNN parameters)
          |
          v
  Next iteration (circuit reused, new weights)
```

The critical performance property is that the circuit compilation (D4) happens once, during the first forward pass. All subsequent iterations reuse the cached XGCF structure, executing only the weight update and the level-parallel forward/backward kernels. This is the mechanism behind the 2.74x training speedup reported in Section 4.3.

**Results.** On the MNIST addition benchmark (512 training images, 256 addition pairs, batch size 64, learning rate 1e-3, CUDA device), the xlog engine was evaluated over 5 training epochs with two random seeds:

| Seed | Initial Loss | Final Loss | Improvement | Elapsed (s) |
|------|-------------|------------|-------------|-------------|
| 42   | 2.7779      | 2.5904     | 6.7%        | 20,987      |
| 7    | 2.7797      | 2.5693     | 7.6%        | 21,963      |

These runs use the full xlog pipeline -- compilation, circuit caching, GPU-resident forward/backward -- exercised through `pyxlog.train_model_tensor`. The loss reduction across 5 epochs confirms that gradients flow correctly from the circuit evaluation through the logic program back to the CNN parameters. The relatively long wall-clock times reflect the cold-start compilation cost (which is amortized across subsequent epochs) and the small-scale nature of these development-track validation runs.

### 5.3 Term Embeddings (v0.5.0)

Version 0.5.0 introduces a term embedding API that attaches dense vector representations to Datalog symbols. Where `nn/4` neural predicates map perception (images, text) to discrete logical facts, term embeddings go in the opposite direction: they give logical entities a continuous representation that can participate in neural computation.

The Python API provides two methods on `CompiledProgram`:

- **`register_embedding(name, module_or_tensor, trainable=True)`** accepts either a PyTorch `nn.Embedding` module or a raw 2D `torch.Tensor`. For `nn.Embedding`, the weight matrix shape `[vocab_size, dim]` is read from the module. For raw tensors, `trainable` must be `False` (frozen embeddings). The Rust-side `EmbeddingHandle` (in `crates/xlog-neural/src/handle.rs`) stores the module reference, dimensionality, vocabulary size, and trainability flag. Embeddings are registered in the same `NetworkRegistry` as neural predicates, keyed by name.

- **`forward_embedding(name, ids)`** looks up embedding vectors by integer IDs, returning a batched `torch.Tensor` with shape `[len(ids), dim]`. For trainable `nn.Embedding` modules, the returned tensor carries autograd history, so gradients flow through the embedding lookup. For frozen tensors, the result has `requires_grad=False`. Internally, the method determines the device from the embedding weight tensor and creates the ID tensor on the same device to avoid CPU/CUDA mismatches.

This API enables entity representation learning within the logic framework. For example, a knowledge graph completion task can attach learned embeddings to entity symbols, combine them with relational predicates in xlog rules, and jointly train the embeddings alongside neural perception networks. The embeddings are the only gradient-carrying embedding API in v0.5.0; future releases may integrate embedding lookups directly into the circuit evaluation path.

### 5.4 Differentiable ILP (Beta)

> **Note:** The differentiable ILP subsystem is in **beta** as of v0.5.0. The API surface, promotion pipeline, and artifact format may change in future releases. The schema version is `beta-v2`.

xlog includes a differentiable inductive logic programming (dILP) trainer that learns first-order rules from examples. Unlike the neural predicate path (which trains network parameters given fixed rules), dILP searches the space of candidate rules, using gradient-based optimization to assign soft probabilities to candidates and hard selection to commit the best rule.

**Sparse GPU mask API.** A key scalability challenge in dILP is the candidate mask computation. For a program with `R` relations of arity `A`, the naive candidate space is `O(R^A)`, which can easily reach millions of candidates. Materializing a dense `N x N x N` mask on the Python side is prohibitive. The `CompiledIlpProgram` class (in `crates/pyxlog/src/ilp.rs`) provides three sparse mask methods of increasing strictness:

- `set_rule_mask_sparse(name, candidate_ids, soft_probs_dlpack, budget)` -- accepts a DLPack GPU tensor of soft probabilities, performs deterministic top-k selection on the Rust side, and stores a sparse `IlpMask` without dense materialization.
- `set_rule_mask_sparse_selected(name, selected_candidate_ids, selected_soft_probs_dlpack)` -- the caller preselects candidate IDs, avoiding the full-vector probability download entirely.
- `set_rule_mask_sparse_device(name, ...)` -- the strictest path where selected candidate IDs remain as a device tensor, with Rust resolving them against a fixed candidate order uploaded via `set_candidate_map()`.

The GPU loss/gradient computation (`compute_ilp_loss_grad_gpu`) builds a sparse COO structure from the per-entry membership masks, converts it to CSR via device-side sort, and launches forward (credit gather + NLL loss) and backward (gradient scatter) CUDA kernels. The entire credit/loss path is GPU-resident with zero host transfers when using the strict sparse device mask.

**Six-gate promotion pipeline.** When training converges and a candidate rule is selected, xlog does not blindly commit it to the program. Instead, the rule passes through a six-gate promotion pipeline:

1. **Convergence gate** -- verifies that training actually converged (loss plateau, discreteness threshold).
2. **Novel-rate audit** -- checks the fraction of new derivations the rule produces; a configurable `max_novel_rate` threshold catches rules that overgenerate.
3. **Regression check** -- ensures the new rule does not invalidate existing derivations for protected relations.
4. **Holdout F1 gate** -- evaluates the candidate on held-out examples using k-fold cross-validation (default 5 folds, threshold 0.95 F1).
5. **Ambiguity scan** -- detects when multiple candidates have similar soft probabilities, optionally reporting `ambiguous_alternatives` for manual review.
6. **Typed schema validation** -- when `typed_schema_required=True`, verifies that the candidate rule's variable bindings respect declared type annotations on relations.

A rule that fails any gate receives a `PromotionStatus` of `GATE_FAILED` or `MANUAL_REVIEW_REQUIRED`, with per-gate `GateResult` details. Only rules that pass all six gates are committed.

**Hard-negative mining.** The `TrainConfig.max_mined_negatives` parameter enables hard-negative mining: the trainer identifies examples that the current candidate rule incorrectly covers and adds them as negative examples for subsequent training rounds. This improves precision for rules operating in domains with many near-miss patterns.

**Artifact persistence.** Learned rules are serialized as `LearnedArtifact` objects containing the candidate map, final logits, soft probabilities, hard selection, discovered rule text, full `TrainConfig` snapshot, and training telemetry. The `save()` method writes JSON with SHA-256 hashes computed over both the candidate map and the configuration snapshot, enabling integrity verification on `load(path, verify_hash=True)`. The schema version (`beta-v2`) is validated on load to prevent silent format incompatibilities. For strict GPU-native training paths where logits and probabilities remain device-resident, a `StrictLearnedArtifact` variant defers host materialization until `export_compat_artifact()` is explicitly called, avoiding unnecessary device-to-host transfers during automated pipeline runs.
