# XLOG System Architecture

XLOG is a **GPU-native logic programming language for unified symbolic
reasoning**. Its compiler and runtime span deterministic Datalog evaluation,
probabilistic inference via knowledge compilation, SAT/MaxSAT verification, and
differentiable neural-symbolic training on a shared CUDA substrate. A single
language frontend lowers source programs through staged IRs to GPU-resident
execution plans rather than splitting the repository into disconnected engines.

This document is the contributor-facing deep reference for that architecture:
crate decomposition, intermediate representations, residency and memory model,
backend boundaries, and how the runtime coordinates the four reasoning paths.
The companion whitepaper in `docs/whitepaper/main.pdf` explains the language
framing and evaluation results; this document focuses on how the repository is
structured today.

## Table of Contents

1. [Theoretical Foundations](#theoretical-foundations)
2. [System Overview](#system-overview)
3. [Crate Structure](#crate-structure)
4. [Data Types](#data-types)
5. [Compilation Pipeline](#compilation-pipeline)
6. [Execution Pipeline](#execution-pipeline)
7. [CUDA Kernels](#cuda-kernels)
8. [Memory Management](#memory-management)
9. [Key Algorithms](#key-algorithms)
10. [Public APIs](#public-apis)
11. [Dataflow Diagram](#dataflow-diagram)
12. [Error Handling](#error-handling)
13. [Configuration](#configuration)
14. [Testing](#testing)
15. [See Also](#see-also)

---

## Glossary

For release and version context around technical terms used throughout this document (SCC, RIR, PIR, CNF, XGCF, WMC, Decision-DNNF, DLPack, Arrow IPC, Arrow C Data Interface, Semi-Naive, HISA, etc.), see [ROADMAP.md](ROADMAP.md).

---

## Theoretical Foundations

This section describes the formal semantics, design principles, and research foundations underlying XLOG.

### Design Goals

XLOG is designed around these core principles:

| Goal | Description |
|------|-------------|
| **GPU-Resident Execution** | All semantic evaluation data structures (facts, derived relations, solver state, circuit values) remain GPU-resident during execution. Host involvement is limited to orchestration, I/O, and compilation. |
| **Formal Semantics with Explicit Tiers** | XLOG provides explicit semantics choices and "tiers" (exact vs approximate) per subsystem, with machine-checkable boundaries. |
| **Practical Implementability** | A staged implementation plan delivered value early (GPU Datalog + probabilistic facts) and now ships GPU-native epistemic execution, with bounded honest limits at the genuine semantic boundaries. |
| **Robustness and Verifiability** | Where "exactness" is claimed, XLOG includes proof/certificate artifacts or cross-check capability. |

### Declarative Programming Paradigms

XLOG is a unified platform spanning four closely-related reasoning paradigms:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         XLOG REASONING STACK                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐    │
│  │ xlog-logic  │   │ xlog-prob   │   │ EIR/GPU     │   │ xlog-solve  │    │
│  │             │   │             │   │ epistemic   │   │             │    │
│  │ Datalog     │   │ ProbLog     │   │ Epistemic   │   │ SAT/MaxSAT  │    │
│  │ recursion   │   │ inference   │   │ world views │   │ solving     │    │
│  │ stratified  │   │ WMC/MC      │   │ K/M ops     │   │ GPU search  │    │
│  │ negation    │   │ gradients   │   │ splitting   │   │ certificates│    │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘    │
│         │                 │                 │                 │            │
│         └────────────┬────┴────────┬────────┴────────┬────────┘            │
│                      ▼             ▼                 ▼                     │
│              ┌─────────────────────────────────────────────┐               │
│              │         GPU Kernel Provider                 │               │
│              │   joins · sorts · aggregations · circuits   │               │
│              └─────────────────────────────────────────────┘               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Subsystem | Purpose | Primary Inspirations |
|-----------|---------|---------------------|
| **xlog-logic** | Deterministic Datalog-style recursion and stratified negation | GPUlog (HISA indexing), VFLog (columnar GPU Datalog) |
| **xlog-prob** | Probabilistic + differentiable reasoning | ProbLog knowledge compilation (d-DNNF/SDD/BDD), WMC |
| **EIR/GPU epistemic** | Accepted epistemic execution surface with explicit EIR, FAEEL default semantics, Gelfond 1991 compatibility mode, split/joint planning, and GPU-backed runtime paths | eclingo (Gelfond 1991 context), FAEEL founded world views |
| **xlog-solve** | SAT/MaxSAT solver services (GPU CDCL verifier + CLS) | Deterministic CDCL (watched literals + 1-UIP + on-GPU model/proof validation), FastFourierSAT-inspired CLS |

### Intermediate Representations

XLOG uses a layered IR stack to keep the platform coherent across paradigms:

```
Source Code
    │
    ▼
┌─────────┐
│   AST   │  Abstract Syntax Tree from parser
└────┬────┘
     │
     ▼
┌─────────┐
│   RIR   │  Relational IR: joins, projections, unions, dedup, fixpoint loops
└────┬────┘
     │
     ├──────────────────┬──────────────────┐
     ▼                  ▼                  ▼
┌─────────┐       ┌─────────┐       ┌─────────┐
│   PIR   │       │   EIR   │       │   SIR   │
│Provenance│       │Epistemic│       │ Solve  │
│   IR    │       │   IR    │       │   IR   │
└────┬────┘       └────┬────┘       └────┬────┘
     │                  │                 │
     ▼                  │                 │
┌─────────┐             │                 │
│  XGCF   │◄────────────┴─────────────────┘
│GPU Circt│
└─────────┘
```

| IR | Purpose | Key Structures |
|----|---------|----------------|
| **RIR** | Relational operations for deterministic logic | `Unit` (identity), `Scan`, `Filter`, `Project`, `Join`, `GroupBy`, `Union`, `Distinct`, `Diff`, `Fixpoint`, `TensorMaskedJoin` |
| **PIR** | Provenance tracking for probabilistic inference | `PIR_Lit`, `PIR_NegLit`, `PIR_And`, `PIR_Or`, `PIR_Decision` — weighted Boolean formula / circuit terms |
| **EIR** | Epistemic reasoning structures for the accepted GPU-backed surface | epistemic literals/modes, executable plans, split plans, world-view candidates, GPU runtime traces |
| **SIR** | Boolean satisfiability and optimization | `SIR_CNF`, `SIR_Cardinality`, `SIR_Weights`, `SIR_Objective`, `SIR_ProofPolicy` |
| **XGCF** | GPU-evaluable circuit format | Levelized DAG with `node_type[]`, `value[]`, `adj[]` for forward/backward evaluation |

All IR nodes include metadata for:
- Estimated cardinality ranges
- Memory peak estimates
- Skew hints (for join partitioning)
- Incremental update semantics (delta/full)
- Tier requirements (exact vs approximate)

### Formal Semantics

#### Deterministic Logic (`xlog-logic`)

XLOG implements **Datalog with stratified negation**:

- **Evaluation model**: Semi-naive fixpoint iteration
- **Negation**: Must be stratifiable (no cycles through negation)
- **Aggregation**: Stratified (no recursion through aggregates)
- **Exactness**: **Exact** for the supported fragment

The formal semantics is the standard least Herbrand model semantics for Datalog, extended with stratified negation using the iterated fixpoint approach.

#### Probabilistic Logic (`xlog-prob`)

XLOG implements **ProbLog-style distribution semantics**:

- **Base**: Each probabilistic fact `p::f.` independently holds with probability `p`
- **Semantics**: The probability of a query is the sum of probabilities of all worlds where the query holds
- **Evaluation**: Weighted Model Counting (WMC) via knowledge compilation

**Inference tiers:**

| Tier | Name | Description | Requirements |
|------|------|-------------|--------------|
| **Exact circuit tier** | Exact (circuit-evaluable) | Programs compiled into decomposable circuits evaluated exactly on GPU | Compilation succeeds; positive rule bodies only |
| **Exact structural tier** | Exact (restricted structure) | Acyclic probabilistic dependencies or bounded-treewidth fragments | Structural restrictions satisfied |
| **Approximate sampling tier** | Approximate | Monte Carlo sampling with GPU-parallel estimators and calibrated uncertainty | Any program; results include confidence intervals |

**Circuit evaluation** uses log-space arithmetic for numerical stability:
- **LIT node**: `v = log(w_lit)` with evidence masking
- **AND node**: `v = v_a + v_b + …`
- **OR node**: `v = logsumexp(v_a, v_b, …)`
- **DECISION node**: `v = logsumexp(log(p(var=1)) + v_true, log(p(var=0)) + v_false)`

**Reverse-mode autodiff** on XGCF circuits:
- Initialize `adj[root] = dL/dv_root` from the loss
- AND: distribute `adj` unchanged to children
- OR: `adj[child_j] += adj[parent] * exp(v_child_j - v_parent)` (softmax weights)
- Gradients flow back to neural predicates via PyTorch custom autograd

#### Epistemic Logic (EIR/GPU Accepted Surface)

XLOG implements an accepted **epistemic logic programming** surface with world
views through explicit EIR and GPU-backed execution paths:

- **Modal operators**: `know atom(...)`, `possible atom(...)`, `not know atom(...)`, `not possible atom(...)`, and finite nested modal chains normalized by parity/duality.
- **Default semantics**: FAEEL-style founded world views, including foundedness gates for self-support.
- **Compatibility mode**: Gelfond 1991 semantics for explicit interoperability, including accepted self-supported `possible` cases in compatibility mode.
- **Execution surface**: EIR-derived candidate generation, value-level tuple-key membership, safe split/joint solving, same-name multi-arity disambiguation, determined-head stratification, and GPU-backed runtime/WCOJ dispatch for accepted epistemic programs.

**Complexity**: World view existence is **Σ_P^3-complete**, requiring careful engineering.

**Inference tiers:**

| Tier | Name | Description | Feasibility Bounds |
|------|------|-------------|--------------------|
| **Exact structural tier** | Exact (structural) | Epistemically stratified programs or successful splitting | `k <= 24` epistemic atoms per component |
| **Exact bounded-enumeration tier** | Exact (bounded enumeration) | Bounded candidate enumeration with propagation | `k <= 16` or `|candidates| <= 50,000` after propagation |
| **Approximate sampling tier** | Approximate | Sampling-based world-view approximation | Any program; explicit `UNKNOWN/approx` labeling |

**Core algorithm**: Generate–Propagate–Test
1. **Normalize and split**: Apply epistemic splitting to decompose into components
2. **Generate**: Represent guesses as bitvectors over epistemic atoms (GPU-friendly)
3. **Propagate**: Structural propagation (modal consistency) + semantic propagation (batched solver calls)
4. **Test**: Verify candidates against brave/cautious consequences

### XGCF: GPU Circuit Format

The **XLOG GPU Circuit Format (XGCF)** is a levelized DAG representation optimized for parallel GPU evaluation:

```
┌─────────────────────────────────────────────────────────┐
│                    XGCF Structure                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  node_type[i] ∈ {CONST0, CONST1, LIT, AND, OR, DECISION}│
│                                                         │
│  a[i], b[i]      Child indices (node-type dependent)    │
│  var[i]          Variable ID for DECISION nodes         │
│  lit[i]          Literal ID for LIT nodes               │
│                                                         │
│  level_offsets[] Topological levels for parallel eval   │
│                                                         │
│  value[i]        Forward pass values (log-space)        │
│  adj[i]          Backward pass adjoints                 │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Batching**: Multiple circuits can be concatenated with `circuit_offsets` for batch evaluation. Neural batch dimensions are expressed as multiple leaf-weight tables with fused kernels for leaf-gather.

### Knowledge Compilation Pipeline

The probabilistic exact inference path uses knowledge compilation:

```
PIR (Provenance IR)
    │
    ▼ Tseitin encoding
┌─────────┐
│   CNF   │  Conjunctive Normal Form (with weights)
└────┬────┘
     │ Optional GPU preprocessing (simplification)
     ▼
┌─────────┐
│GPU d-DNNF│ Decision-DNNF compiler (GPU-native)
└────┬────┘
     │
     ▼
┌─────────────┐
│Decision-DNNF│  Decomposable circuit form
└──────┬──────┘
       │ Lower to GPU format
       ▼
┌─────────┐
│  XGCF   │  GPU-evaluable levelized circuit
└─────────┘
```

**Compile-once, evaluate-many**: Following ProbLog's operational pattern, XLOG compiles the circuit structure once. Training iterations and evidence updates only modify leaf weights and evidence masks—no recompilation required.

### Memory Residency Contract

XLOG enforces strict GPU residency by default:

**Strict GPU-Resident Mode (default):**
- All runtime semantic state (relations, deltas, circuit values, solver state) must fit in device memory
- If the plan cannot execute within GPU memory budget, XLOG returns a deterministic `RESOURCE_EXHAUSTED` error with diagnostics

**Out-of-Core Mode (explicit opt-in via `--allow-ooc`):**
- XLOG may spill *immutable* intermediates to host-pinned memory
- Never used silently; performance not guaranteed

### Research Foundations

XLOG builds on established research in GPU-accelerated databases and probabilistic logic programming:

| System/Paper | Contribution to XLOG |
|--------------|---------------------|
| **[GPUlog](https://thomas.gilray.org/pdf/datalog-gpu.pdf)** | HISA (Hash-Indexed Sorted Array) for efficient range queries, lock-free deduplication, parallel iteration. Reports up to 45× speedups. |
| **[VFLog](https://arxiv.org/abs/2501.13051)** | Column-oriented GPU Datalog runtime. Demonstrates 200× gains over CPU column engines. |
| **[mnmgDatalog](https://hpcrl.github.io/ICS2025-webpage/program/Proceedings_ICS25/ics25-71.pdf)** | Multi-node multi-GPU Datalog. Radix-hash partitioning, GPU-aware all-to-all. |
| **[ProbLog](https://dtai.cs.kuleuven.be/problog/)** | Knowledge compilation approach to probabilistic logic. Compile-once evaluate-many pattern. |
| **Neural-symbolic AI** | Neural predicates integrated with probabilistic logic. End-to-end differentiable inference. |
| **[D4](https://www.ijcai.org/proceedings/2017/0093.pdf)** | State-of-the-art Decision-DNNF compiler for weighted model counting. |
| **[eclingo](https://arxiv.org/abs/2008.02018)** | Epistemic logic solver implementing Gelfond 1991 semantics via guess-and-check. |
| **[FAEEL](https://arxiv.org/abs/1907.09247)** | Founded Autoepistemic Equilibrium Logic. Avoids self-supported world views. |
| **[FastFourierSAT](https://arxiv.org/abs/2308.15020)** | Massively parallel continuous local search for GPU SAT solving. |
| **[ParaFROST](https://link.springer.com/article/10.1007/s10703-023-00432-z)** | Certified GPU-accelerated SAT inprocessing with proof generation. |

---

## System Overview

```
                                              ┌──> RIR ──> Deterministic executor  (xlog-runtime)
                                              │             (semi-naive fixpoint on GPU)
   xlog source ──> Parser ──> Stratifier ──┤
                                              │──> PIR ──> CNF ──> Decision-DNNF compiler ──> XGCF
                                              │             (probabilistic inference; xlog-prob)
                                              │
                                              │──> Solver IR (planned) ──> GPU CDCL
                                              │             (SAT/MaxSAT verification; xlog-solve)
                                              │
                                              └──> RIR + dILP masks ──> differentiable trainer
                                                            (neural-symbolic; xlog-neural + pyxlog)
```

Common pipeline stages:

- **Parsing**: PEG grammar with Pest in `crates/xlog-logic`
- **Stratification**: SCC analysis over the predicate dependency graph
- **Lowering**: AST to backend-specific IR paths (`RIR`, `PIR`, planned solver IR)
- **Optimization**: predicate pushdown, join planning, and execution-shape rewrites
- **Execution**: backend interpreters dispatch against the shared CUDA provider

Each backend has its own deep-dive document: deterministic evaluation in
[`architecture/gpu-execution.md`](architecture/gpu-execution.md), probabilistic
inference in [`architecture/xlog-prob.md`](architecture/xlog-prob.md), solver
services in [`architecture/solver-services.md`](architecture/solver-services.md),
and neural/dILP execution in
[`architecture/dilp-training.md`](architecture/dilp-training.md).

---

## Crate Structure

```
xlog/
├── crates/
│   ├── xlog-core/       # Foundation types, traits, error handling
│   ├── xlog-ir/         # Intermediate representations (RIR nodes)
│   ├── xlog-logic/      # Datalog frontend (parser, compiler, optimizer)
│   ├── xlog-runtime/    # Query executor, relation storage
│   ├── xlog-cuda/       # CUDA provider, memory management, interop (Arrow IPC/C Data, DLPack)
│   ├── xlog-stats/      # Runtime statistics (optimizer feedback + adaptive indexing)
│   ├── xlog-prob/       # Probabilistic tier (exact inference + Monte Carlo)
│   ├── xlog-neural/     # Neural-symbolic integration
│   ├── xlog-solve/      # Solver services (SAT/MaxSAT)
│   ├── xlog-gpu/        # High-level GPU API (Rust)
│   ├── xlog-induce/     # Bounded exact-induction engine for an external consumer
│   ├── xlog-cli/        # CLI binary (deterministic + probabilistic execution)
│   ├── pyxlog/     # Python module (PyO3 + DLPack + training API)
│   └── xlog-cuda-tests/ # CUDA/PTX certification suite (not published)
├── kernels/             # CUDA source files (.cu) + embedded PTX (.ptx)
├── examples/
│   └── neural/          # Neural-symbolic training examples
```

XLOG no longer vendors a CPU knowledge compiler binary; the exact inference path is GPU-native.

### Dependency Graph

Production `[dependencies]` only (dev-dependencies omitted):

```
Tier 0 (leaf):  xlog-core

Tier 1:         xlog-ir ──────────> xlog-core
                xlog-cuda ─────────> xlog-core
                xlog-stats ────────> xlog-core
                xlog-neural ───────> xlog-core          (Wave 1: new edge)

Tier 2:         xlog-logic ────────> xlog-core + xlog-ir + xlog-stats
                xlog-runtime ──────> xlog-core + xlog-ir + xlog-cuda + xlog-stats
                xlog-solve ────────> xlog-core + xlog-cuda

Tier 3:         xlog-gpu ─────────> xlog-core + xlog-cuda + xlog-ir + xlog-logic + xlog-runtime
                xlog-prob ─────────> xlog-core + xlog-cuda + xlog-logic + xlog-runtime + xlog-solve + xlog-ir
                xlog-induce ──────> xlog-core + xlog-cuda + xlog-runtime

Tier 4:         pyxlog ────────────> 9 crates (integration hub; includes xlog-induce)
                xlog-cli ──────────> xlog-core + xlog-cuda + xlog-logic + xlog-gpu + xlog-prob
                xlog-cuda-tests ───> xlog-cuda + xlog-core + xlog-solve
```

**Wave 1 changes** (2026-03-10): removed `xlog-logic → xlog-runtime` (moved to dev-deps),
removed `xlog-stats → xlog-cuda` (was unused), added `xlog-neural → xlog-core`.

**Epistemic WFS boundary:** `xlog-gpu` intentionally has no `xlog-prob`
dependency. Accepted cyclic negated-modal WFS execution must route through the
GPU-backed WFS plan in `xlog-gpu`; the host `HashMap`/`HashSet` WFS implementation
in `xlog-prob` remains a probabilistic/provenance subsystem and is not an accepted
fallback for production epistemic execution. This boundary is not a
device-resident/no-host-interaction WFS residency guarantee: host orchestration
and metadata row-count reads may still participate in WFS convergence.

### Crate Responsibilities

| Crate | Purpose |
|-------|---------|
| `xlog-core` | Shared types (`ScalarType`, `Schema`, `AggOp`), traits (`KernelProvider`), errors |
| `xlog-ir` | Relational IR nodes (`RirNode`), expressions (`Expr`), execution plans |
| `xlog-logic` | Parser, stratification, lowering (AST → RIR), optimizer (predicate pushdown + join planning), neural predicate syntax (`nn/4`) |
| `xlog-runtime` | `Executor`, versioned `RelationStore`, profiling, incremental maintenance, adaptive join index cache |
| `xlog-cuda` | `CudaKernelProvider`, `GpuMemoryManager`, `CudaBuffer`/`CudaColumn`, PTX embedding, Arrow IPC/C Data + DLPack interop |
| `xlog-stats` | `StatsManager` + `StatsSnapshot` (compiler feedback + runtime tracking) |
| `xlog-prob` | Probabilistic tier: provenance → CNF → GPU Decision-DNNF compiler → XGCF; exact inference + Monte Carlo sampling + circuit caching; includes GPU-native PIR→CNF encoder and GPU Decision-DNNF/CDCL compilation utilities |
| `xlog-neural` | Neural-symbolic integration: `NetworkRegistry`, `NetworkHandle`, `TensorSourceRegistry`, `NeuralBridge` |
| `xlog-solve` | Solver services: GPU CDCL verifier (complete SAT/UNSAT, on-GPU validation) + CLS SAT/MaxSAT (heuristic) |
| `xlog-gpu` | High-level GPU API: deterministic execution + input/output buffers for integration layers |
| `xlog-induce` | Bounded exact-induction engine — scores all `(left, right)` candidate pairs across four fixed 2-body topologies (chain/star/fanout/fanin) in one batched GPU pass via `ilp_exact` kernel; top-K per topology with structured candidate metadata. See [architecture/bounded-exact-induction.md](architecture/bounded-exact-induction.md) |
| `xlog-cli` | `xlog` CLI for deterministic and probabilistic execution with Arrow IPC I/O |
| `pyxlog` | PyO3 extension (`pyxlog` Python module) exposing DLPack-first deterministic + probabilistic evaluation + neural-symbolic training API |
| `xlog-cuda-tests` | CUDA/PTX certification suite (release gating; `publish = false`) |

---

## Data Types

### Scalar Types (`xlog-core`)

```rust
pub enum ScalarType {
    U32, U64,           // Unsigned integers
    I32, I64,           // Signed integers
    F32, F64,           // Floating point
    Bool,               // Boolean
    Symbol,             // Dictionary-encoded strings (hashed to u32)
}
```

**Note**: `Symbol` values are stored as `u32` IDs that map bidirectionally to strings via a global string table. Symbols are **reversible**: the original string can be recovered for display in query output.

### Schema

```rust
pub struct Schema {
    pub columns: Vec<(String, ScalarType)>,  // Column names and types
    pub key_columns: Vec<usize>,              // Key column indices
}
```

### Relational IR Nodes (`xlog-ir`)

```rust
pub enum RirNode {
    Scan { rel: RelId },                                    // Read relation
    Filter { input: Box<RirNode>, predicate: Expr },        // Selection
    Project { input: Box<RirNode>, columns: Vec<ProjectExpr> }, // Projection (pass-through + computed)
    Join {                                                   // Hash join
        left: Box<RirNode>,
        right: Box<RirNode>,
        left_keys: Vec<usize>,
        right_keys: Vec<usize>,
        join_type: JoinType,  // Inner, Semi, Anti, LeftOuter
    },
    GroupBy {                                                // Aggregation
        input: Box<RirNode>,
        key_cols: Vec<usize>,
        aggs: Vec<(usize, AggOp)>,  // (column, operation)
    },
    Union { inputs: Vec<RirNode> },                          // Set union
    Distinct { input: Box<RirNode>, key_cols: Vec<usize> },  // Deduplication
    Diff { left: Box<RirNode>, right: Box<RirNode> },        // Set difference
    Fixpoint {                                               // Recursive iteration
        scc_id: u32,
        base: Box<RirNode>,
        recursive: Box<RirNode>,
        delta_rel: RelId,
        full_rel: RelId,
    },
}
```

### Expressions

```rust
pub enum Expr {
    Column(usize),                              // Column reference
    Const(ConstValue),                          // Literal value
    Compare { left, op, right },                // Comparison (Eq, Ne, Lt, Le, Gt, Ge)
    And(Vec<Expr>),                             // Logical AND
    Or(Vec<Expr>),                              // Logical OR
    Not(Box<Expr>),                             // Logical NOT

    // Arithmetic (used by `is` expressions)
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Mod(Box<Expr>, Box<Expr>),
    Abs(Box<Expr>),
    Min(Box<Expr>, Box<Expr>),
    Max(Box<Expr>, Box<Expr>),
    Pow(Box<Expr>, Box<Expr>),
    Cast(Box<Expr>, ScalarType),
}
```

### Aggregation Operations

```rust
pub enum AggOp {
    Count,      // Count rows per group
    Sum,        // Sum values per group
    Min,        // Minimum value per group
    Max,        // Maximum value per group
    LogSumExp,  // Numerically stable log-sum-exp (for probabilistic inference)
}
```

---

## Compilation Pipeline

### Phase 1: Parsing

**File**: `crates/xlog-logic/src/parser.rs`

Uses Pest PEG grammar to parse Datalog source into AST:

```rust
use xlog_logic::parse_program;

let program = parse_program(source)?;
```

**Input**: Datalog source code
```datalog
edge(1, 2).
edge(2, 3).
reach(X, Y) :- edge(X, Y).
reach(X, Z) :- reach(X, Y), edge(Y, Z).
```

**Output**: `Program` AST with rules, facts, predicate declarations, constraints, and queries

### Queries and Constraints (Desugaring)

XLOG supports:
- **Constraints**: `:- body.` (must have *no* solutions)
- **Queries**: `?- atom.` (what to print)

The compiler desugars these into ordinary rules so they run through the same stratification + lowering pipeline:

- `:- body.` → `__xlog_constraint_N(1) :- body.`
- `?- p(1, X).` → `__xlog_query_N(X) :- p(1, X).`

The runner enforces that all constraint relations are empty, and prints the query relations.

### Phase 2: Stratification

**File**: `crates/xlog-logic/src/stratify.rs`

Ensures safe negation ordering using dependency analysis:

1. Build dependency graph between predicates
2. Classify edges: Positive, Negative, Aggregate
3. Find SCCs using Tarjan's algorithm
4. Detect cycles through negation (stratification failure)
5. Topologically sort strata

```rust
use xlog_logic::stratify;

let strata = stratify(&program)?;
```

**Output**: `Vec<Stratum>` ordered by dependencies

### Phase 3: Lowering

**File**: `crates/xlog-logic/src/lower.rs`

Transforms AST to Relational IR:

1. **Schema inference**: Derive column types from facts
2. **Join planning**: Build join trees for positive atoms (bushy dynamic programming for small bodies, greedy for large bodies) using a cost model; can be seeded from runtime `StatsSnapshot`
3. **Variable tracking**: Map variables to column indices for join keys and projections
4. **Comparisons + arithmetic**: Lower comparisons to `Filter` and `is` expressions to computed projections (`ProjectExpr::Computed`)
5. **Negation handling**: Lower stratified negation via `Diff` + `Semi` join (anti-semi pattern over shared variables)
6. **Recursion**: Mark recursive predicate groups as SCCs in the plan; runtime executes recursive SCCs with semi-naive deltas
7. **Projection**: Project result columns to match rule head (and lower aggregates to `GroupBy`)

```rust
lowerer.set_strata(strata_preds);
let plan = lowerer.lower_program(&program)?;
```

**Output**: `ExecutionPlan` with SCCs, strata, compiled rules

### Compiler Orchestration

**File**: `crates/xlog-logic/src/compile.rs`

```rust
use xlog_logic::Compiler;

pub struct Compiler {
    lowerer: Lowerer,
}

impl Compiler {
    pub fn compile(&mut self, source: &str) -> Result<ExecutionPlan> {
        // Phase 1: parse to AST
        let program = parse_program(source)?;

        // Compiler-internal desugaring of queries/constraints into internal rules

        // Phase 2: stratify + pass predicate groups into the lowerer
        let strata = stratify(&program)?;
        self.lowerer.set_strata(strata.into_iter().map(|s| s.predicates).collect());

        // Phase 3: lower to RIR plan
        let plan = self.lowerer.lower_program(&program)?;

        // Phase 4: optimizer rewrites (predicate pushdown, cost-aware join planning)

        Ok(plan)
    }
}
```

---

## Execution Pipeline

### Executor

**File**: `crates/xlog-runtime/src/executor.rs`

```rust
pub struct Executor {
    provider: Arc<CudaKernelProvider>,   // GPU kernel interface
    store: RelationStore,                 // Versioned relation storage
    rel_names: HashMap<RelId, String>,    // RelId → name mapping
    name_to_rel: HashMap<String, RelId>,  // name → RelId mapping
    stats: StatsManager,                  // Runtime statistics (optimizer feedback)
    join_index_cache: JoinIndexCache,     // Cached build-side indexes (adaptive indexing)
}
```

### Execution Flow

1. **Execute strata in order** (dependency-first)
2. **For each stratum**: Execute all rules in SCCs
3. **For recursive SCCs**: Use semi-naive fixpoint iteration
4. **For each rule**: Recursively evaluate RIR node tree
5. **Store results**: Put computed relations in `RelationStore`

### RIR Node Execution

```rust
fn execute_node(&mut self, node: &RirNode) -> Result<CudaBuffer> {
    match node {
        RirNode::Scan { rel } => {
            self.execute_scan(*rel)
        }
        RirNode::Filter { input, predicate } => {
            let buf = self.execute_node(input)?;
            self.execute_filter(&buf, predicate)
        }
        RirNode::Join { left, right, left_keys, right_keys, join_type } => {
            let l = self.execute_node(left)?;
            let r = self.execute_node(right)?;
            self.provider.hash_join_v2(&l, &r, left_keys, right_keys, *join_type)
        }
        RirNode::GroupBy { input, key_cols, aggs } => {
            let buf = self.execute_node(input)?;
            self.provider.groupby_multi_agg(&buf, key_cols, aggs)
        }
        RirNode::Union { inputs } => {
            let bufs: Vec<_> = inputs.iter()
                .map(|n| self.execute_node(n))
                .collect::<Result<_>>()?;
            self.provider.union_all(&bufs)
        }
        RirNode::Distinct { input, key_cols } => {
            let buf = self.execute_node(input)?;
            self.provider.dedup(&buf, key_cols)
        }
        RirNode::Diff { left, right } => {
            let l = self.execute_node(left)?;
            let r = self.execute_node(right)?;
            self.provider.diff(&l, &r)
        }
        RirNode::Project { input, columns } => {
            let buf = self.execute_node(input)?;
            self.execute_project(&buf, columns)
        }
        RirNode::Fixpoint { .. } => {
            self.execute_fixpoint(node)
        }
    }
}
```

### GPU-Resident Execution

**Filter execution** is fully GPU-resident. Predicate trees are lowered to a **mask DAG**: typed compare kernels generate masks, boolean operators (`mask_and`, `mask_or`, `mask_not`) compose them, and stream compaction selects rows without CPU round-trips. Arithmetic expressions referenced by predicates are materialized on the GPU using the arithmetic kernels.

**GroupBy finalization** is also GPU-resident: group boundaries are detected over packed key bytes, a GPU prefix-sum generates group IDs and group start indices, and packed rows are gathered + unpacked on-device to extract group keys. Aggregation outputs remain on the GPU until final output conversion.

### Recursive SCC Evaluation (Semi-Naive)

Recursive programs are executed at the **SCC level** using semi-naive deltas.

High-level algorithm (mirrors `Executor::execute_recursive_scc`):

1. **Seed** each recursive predicate `P`:
   - `full[P] = dedup(⋃ base-rule outputs for P)`
   - `delta[P] = full[P]`
2. **Iterate** up to `MAX_SCC_ITERATIONS`:
   - For each recursive rule, evaluate it semi-naively by **rewriting exactly one recursive scan occurrence** in the rule body to use `delta[...]`, then union all such variants for the head predicate.
   - `delta_new[P] = dedup(delta_raw[P] - full[P])`
   - Stop if all `delta_new[P]` are empty.
   - `full[P] = dedup(full[P] ∪ delta_new[P])`; set `delta[P] = delta_new[P]`.

Note: `RirNode::Fixpoint` exists and is interpreted by `Executor::execute_fixpoint`, but the current compiler emits recursion via SCC metadata + delta rewriting rather than explicit `Fixpoint` nodes.

---

## CUDA Kernels

### Kernel Files

| File | Kernels | Purpose |
|------|---------|---------|
| `arith.cu` | `arith_add_*`, `arith_sub_*`, `arith_mul_*`, `arith_div_*`, `arith_mod_*`, `arith_abs_*`, `arith_neg_*` | Arithmetic operations for `is` expressions |
| `join.cu` | `hash_join_bucket_count_v2`, `hash_join_scatter_v2`, `hash_join_probe_v2`, `hash_join_semi`, `hash_join_anti`, `compute_composite_hash` | Hash joins (v2 with bucketed layout) + composite hashing |
| `pack.cu` | `pack_keys`, `pack_and_hash_keys`, `hash_packed_keys`, `gather_packed_rows`, `compare_packed_keys` | Key packing/hashing + packed-row utilities |
| `dedup.cu` | `mark_unique_*`, `compact_rows` | Sort-based deduplication |
| `filter.cu` | `filter_compare_*`, `compact_*_by_mask`, `mask_{and,or,not}` | Filtering and stream compaction |
| `sort.cu` | `radix_histogram`, `radix_scatter_*`, `init_indices`, `apply_permutation_*`, `gather_keys_*` | Stable radix sort + permutation apply |
| `groupby.cu` | `detect_group_boundaries`, `extract_group_keys`, `groupby_*`, `groupby_logsumexp_*` | Sorted aggregation |
| `scan.cu` | `exclusive_scan_mask`, `count_mask`, `multiblock_scan_*` | Prefix sum operations |
| `set_ops.cu` | `concat_{u32,bytes}`, `sorted_diff_mark` | Union/difference operations |
| `circuit.cu` | `xgcf_forward_level`, `xgcf_backward_level_*` | XGCF circuit eval + reverse-mode gradients (probabilistic) |
| `sat.cu` | `sat_cdcl_solve`, `sat_check_model`, `sat_proof_check`, `sat_assert_*`, `sat_xgcf_cnf_*`, `sat_emit_not_phi` | GPU CDCL verifier + equivalence query construction helpers |
| `mc_sample.cu` | `mc_sample_bernoulli` | Bernoulli sampling (Monte Carlo inference) |

### Hash Join Implementation (v2)

The runtime uses a **bucketed "CSR buckets"** layout for joins:

- Compute a 64-bit composite hash for the join key columns (via packed key bytes).
- **Build** (right side):
  - `hash_join_bucket_count_v2`: count build rows per bucket (bucket = low bits of hash)
  - GPU exclusive scan → `bucket_offsets`
  - `hash_join_scatter_v2`: scatter build row indices contiguously per bucket and store aligned hashes
- **Probe** (left side):
  - `hash_join_probe_v2`: probe the bucket range and compare hashes
  - optional **key verification** compares packed key bytes to eliminate hash-collision false positives

### Multi-Column Composite Hash

```cuda
__global__ void compute_composite_hash(
    const uint8_t* data,          // Packed row data
    const uint32_t* col_offsets,  // Byte offset of each key column
    const uint32_t* col_sizes,    // Byte size of each key column
    uint32_t num_key_cols,
    uint32_t num_rows,
    uint32_t row_stride,
    uint64_t* hashes              // Output: 64-bit hash per row
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;

    const uint8_t* row = data + gid * row_stride;

    // FNV-1a hash combining all key columns
    uint64_t hash = 14695981039346656037ULL;
    for (uint32_t k = 0; k < num_key_cols; k++) {
        const uint8_t* col_start = row + col_offsets[k];
        for (uint32_t b = 0; b < col_sizes[k]; b++) {
            hash ^= col_start[b];
            hash *= 1099511628211ULL;
        }
    }
    hashes[gid] = hash;
}
```

### Radix Sort (4-bit, stable)

```cuda
// Radix sort operates on u32 digit segments; passes depend on key width (8 passes per 32 bits).
__global__ void radix_histogram(
    const uint32_t* keys,
    uint32_t n,
    uint32_t* histograms,  // [grid_size * 16]
    uint32_t shift
) {
    __shared__ uint32_t local_hist[16];

    // Initialize shared memory
    if (threadIdx.x < 16) local_hist[threadIdx.x] = 0;
    __syncthreads();

    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        uint32_t digit = (keys[gid] >> shift) & 0xF;
        atomicAdd(&local_hist[digit], 1);
    }
    __syncthreads();

    // Write to global histogram
    if (threadIdx.x < 16) {
        histograms[blockIdx.x * 16 + threadIdx.x] = local_hist[threadIdx.x];
    }
}
```

### GroupBy Aggregation

```cuda
__global__ void groupby_sum(
    const uint32_t* values,
    const uint32_t* group_ids,  // Pre-computed from sorted keys
    uint32_t num_rows,
    uint64_t* sums              // Output: sum per group
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;

    uint32_t group = group_ids[gid];
    atomicAdd((unsigned long long*)&sums[group], values[gid]);
}
```

**GroupBy Pipeline:**
1. Sort by `key_cols` on GPU
2. Detect group boundaries over packed key bytes
3. Compute group IDs and start indices via prefix-sum
4. Gather group key columns on GPU (packed-row gather + unpack)
5. Run aggregation kernels

**Current value-type support:**
- `count`: any value type (counts rows)
- `sum`/`min`/`max`: `u32` values (output `u64` for `sum`, `u32` for `min`/`max`)
- `logsumexp`: `f64` values (output `f64`)

---

## Memory Management

### CudaBuffer

**File**: `crates/xlog-cuda/src/memory.rs`

```rust
pub enum CudaColumn {
    Owned(TrackedCudaSlice<u8>),      // XLOG-allocated GPU memory
    Dlpack(DlpackColumn),              // External memory via DLPack (calls deleter on drop)
}

pub struct CudaBuffer {
    pub columns: Vec<CudaColumn>,      // Column-major storage, bytes; typed by schema
    pub num_rows: u64,
    pub schema: Schema,
}

impl CudaBuffer {
    pub fn column(&self, idx: usize) -> Option<&CudaColumn>;
    pub fn num_rows(&self) -> u64;
    pub fn arity(&self) -> usize;      // Number of columns
    pub fn schema(&self) -> &Schema;
    pub fn is_empty(&self) -> bool;
}
```

### GPU Memory Manager

**File**: `crates/xlog-cuda/src/memory.rs`

```rust
pub struct GpuMemoryManager {
    device: Arc<CudaDevice>,
    budget: MemoryBudget,
    allocated: AtomicU64,
}

impl GpuMemoryManager {
    pub fn alloc<T: DeviceRepr>(self: &Arc<Self>, len: usize) -> Result<TrackedCudaSlice<T>> {
        let bytes = (len as u64) * (std::mem::size_of::<T>() as u64);

        // Atomic budget check with compare-exchange loop
        loop {
            let current = self.allocated.load(Ordering::SeqCst);
            let new = current.checked_add(bytes)
                .ok_or_else(|| XlogError::ResourceExhausted {
                    context: "allocation overflow".into(),
                    estimated_bytes: bytes,
                    budget_bytes: self.budget.device_bytes,
                })?;

            if new > self.budget.device_bytes {
                return Err(XlogError::ResourceExhausted {
                    context: "GPU memory budget exceeded".into(),
                    estimated_bytes: bytes,
                    budget_bytes: self.budget.device_bytes,
                });
            }

            if self.allocated.compare_exchange(
                current, new, Ordering::SeqCst, Ordering::SeqCst
            ).is_ok() {
                break;
            }
        }

        // Allocate via cudarc; tracked slice decrements budget on drop
        let slice = self.device.inner().alloc::<T>(len)?;
        Ok(TrackedCudaSlice::new(slice, bytes, self.clone()))
    }
}
```

### Device Runtime + Recorded Launch Discipline

The device runtime adds an opt-in stack atop the
`GpuMemoryManager` shown above:
`AsyncCudaResource → LoggingResource → GlobalDeviceBudget →
XlogDeviceRuntime + StreamPool`. It tracks per-allocation
events for cross-stream lifetime safety and exposes an
access-aware `prepare_block_use` / `finish_block_use` /
`prepare_first_use` / `finish_first_use` API that the
`LaunchRecorder` uses to fence kernels safely on
non-default streams. Activated via
`CudaKernelProvider::with_runtime` /
`GpuMemoryManager::with_runtime`; `XLOG_USE_DEVICE_RUNTIME=1`
flips integration-test fixtures onto the runtime stack and
`XLOG_USE_RECORDED_OPS=1` (or per-operator
`XLOG_USE_RECORDED_*`) routes operator dispatch through the
recorded variants. Default `new` constructors are unchanged.

  * Architecture: [`architecture/device-runtime.md`](architecture/device-runtime.md).
  * Operator-author migration checklist:
    [`architecture/recorded-launch-migration.md`](architecture/recorded-launch-migration.md).

### Memory Budget Configuration

```rust
pub struct MemoryBudget {
    pub device_bytes: u64,      // Hard limit
    pub allow_ooc: bool,        // Out-of-core spill (future feature)
    pub abort_on_exceed: bool,  // Fail vs warn
}

impl MemoryBudget {
    /// Use 80% of total device memory
    pub fn from_device_memory(total: u64) -> Self {
        Self {
            device_bytes: total * 80 / 100,
            allow_ooc: false,
            abort_on_exceed: true,
        }
    }

    /// Fixed byte limit
    pub fn with_limit(bytes: u64) -> Self {
        Self {
            device_bytes: bytes,
            allow_ooc: false,
            abort_on_exceed: true,
        }
    }
}
```

---

## Key Algorithms

### Hash Join

1. **Key prep**: Pack join key columns into row-major bytes and compute a 64-bit composite hash (FNV-1a) on GPU.
2. **Build phase (v2 buckets)**: Bucket build rows by low hash bits (count → scan offsets → scatter contiguous bucket entries).
3. **Probe phase**: For each probe row, scan the bucket range, compare hashes, and (optionally) verify key bytes to eliminate hash collisions.
4. **Materialization**: Gather left/right columns into the output schema; for left-outer joins, unmatched right columns are zero-filled.

**Join Types**:
- **Inner**: Output matching pairs
- **Semi**: Output left rows that have any match (existence check)
- **Anti**: Output left rows with no matches
- **LeftOuter**: Output all left rows; unmatched right-side columns are zero-filled

### Sort-Based Deduplication

1. Sort rows by all columns (or key columns)
2. Mark duplicates (compare adjacent rows)
3. Stream compaction to remove marked rows

### Fixpoint Iteration (Semi-Naive)

1. Seed `full` and `delta` relations for each recursive predicate in an SCC.
2. Iterate:
   - Evaluate rule variants where exactly one recursive scan uses `delta` (semi-naive).
   - `delta_new = dedup(delta_raw - full)`
   - `full = dedup(full ∪ delta_new)`; set `delta = delta_new`
3. Stop when all `delta_new` are empty (or iteration limit reached).

### Stream Compaction

1. Create mask array (1 = keep, 0 = discard)
2. Compute exclusive prefix sum of mask
3. Scatter elements to positions indicated by prefix sum

### Radix Sort

Radix sort uses 4-bit digits over `u32` segments; the number of passes depends on key width (e.g., 8 passes per 32 bits):
1. Compute per-block histograms for current digit
2. Global prefix sum of histograms
3. Scatter elements to sorted positions
4. Swap input/output buffers, move to next digit

---

## Public APIs

### Compilation

```rust
use xlog_logic::Compiler;

let mut compiler = Compiler::new();
let plan = compiler.compile(r#"
    pred edge(u32, u32).
    pred reach(u32, u32).

    edge(1, 2).
    edge(2, 3).

    reach(X, Y) :- edge(X, Y).
    reach(X, Z) :- reach(X, Y), edge(Y, Z).

    ?- reach(1, N).
"#)?;
```

### Execution

```rust
use std::sync::Arc;
use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_runtime::Executor;

// Initialize GPU
let device = Arc::new(CudaDevice::new(0)?);
let budget = MemoryBudget::with_limit(1 << 30);  // 1 GB
let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
let provider = Arc::new(CudaKernelProvider::new(device, memory)?);

// Create executor
let mut executor = Executor::new(provider);

// Register input relations
executor.register_relation(RelId(0), "edge");
executor.store_mut().put("edge", edge_buffer);

// Execute plan
let result = executor.execute_plan(&plan)?;

// Read results
let reach = executor.store().get("reach").unwrap();
println!("Reachable pairs: {} rows", reach.num_rows());
```

### High-Level API (xlog-gpu)

```rust
use pyxlog::LogicProgram;

let program = LogicProgram::compile(source)?;
let results = program.run()?;

for (name, buffer) in results {
    println!("{}: {} rows", name, buffer.num_rows());
}
```

### CLI

The `xlog` CLI is a production entry point for deterministic and probabilistic programs. It reads `.xlog` sources, accepts Arrow IPC inputs for EDB relations, and emits query results as pretty tables, CSV, or Arrow IPC streams.

```bash
# Deterministic execution
xlog run examples/xlog/00-basics/01_tc_reachability.xlog
xlog run --input edge=data.arrow --output csv program.xlog

# Probabilistic execution
xlog prob examples/prob/01-wet-conditioning.xlog --prob-engine exact_ddnnf
xlog prob program.xlog --prob-engine mc --samples 10000 --seed 42

# Output formats
xlog run program.xlog --output pretty   # Default: formatted tables
xlog run program.xlog --output csv      # CSV format
xlog run program.xlog --output arrow --output-dir ./results  # Arrow IPC files
```

### Python API

```python
import pyxlog

# Deterministic execution
program = pyxlog.LogicProgram.compile(source)
results = program.evaluate()

# Results are DLPack capsules (zero-copy GPU tensors)
for name, capsule in results.items():
    import torch
    tensor = torch.from_dlpack(capsule)
    print(f"{name}: {tensor.shape}")

# Probabilistic execution
prob_program = pyxlog.Program.compile(source, prob_engine="exact_ddnnf")
prob_results = prob_program.evaluate()
import torch
prob = torch.from_dlpack(prob_results.prob)  # f64 CUDA tensor, shape [num_queries]
print(prob_results.atoms)
print(prob)
# Optional host read for a single scalar:
print(float(prob[0].item()))

# Neural-symbolic training
program = pyxlog.Program.compile("""
    nn(mnist_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).
    addition(X, Y, Z) :- digit(X, DigitX), digit(Y, DigitY), Z is DigitX + DigitY.
""")

# Register PyTorch network
program.register_network("mnist_net", model, optimizer)

# Add tensor data source
program.add_tensor_source("train", images_tensor)

# Train on addition queries
history = pyxlog.train_model(program, queries, epochs=50, batch_size=32)
print(f"Final loss: {history.epoch_losses[-1]}")
```

### Profiling

```rust
use xlog_runtime::{Profiler, OpStats};

let mut profiler = Profiler::new(true);  // enabled
profiler.record(OpStats::new("hash_join", 1000, 500, 1234, 8000));
profiler.record(OpStats::new("filter", 500, 100, 567, 4000));
println!("{}", profiler.summary());
```

---

## Dataflow Diagram

The diagram below depicts the **deterministic Datalog evaluation path**
end-to-end. The probabilistic, solver, and neural-symbolic paths share the
parser, stratifier, and kernel provider but branch into their own IRs and
executors after lowering.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         XLOG SYSTEM DATAFLOW                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐                                                    │
│  │ Datalog Source  │                                                    │
│  │                 │                                                    │
│  │ edge(1,2).      │                                                    │
│  │ reach(X,Y) :-   │                                                    │
│  │   edge(X,Y).    │                                                    │
│  └────────┬────────┘                                                    │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │ PARSER (Pest)   │  Tokenize → Parse → AST                           │
│  └────────┬────────┘                                                    │
│           │ Program { rules, facts, predicates }                        │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │ STRATIFIER      │  Dependency graph → SCCs → Strata ordering        │
│  └────────┬────────┘                                                    │
│           │ Vec<Stratum>                                                │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │ LOWERER         │  AST → Join trees → RIR nodes                     │
│  │                 │  Schema inference, variable tracking               │
│  └────────┬────────┘                                                    │
│           │ ExecutionPlan { sccs, strata, rules_by_scc }               │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │ EXECUTOR        │  Execute strata → Execute rules → RIR nodes       │
│  │                 │                                                    │
│  │  for stratum:   │                                                    │
│  │    for scc:     │                                                    │
│  │      for rule:  │                                                    │
│  │        eval()   │───────────────────────┐                           │
│  └─────────────────┘                       │                           │
│                                            ▼                           │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │ RIR NODE EVALUATION                                      │          │
│  │                                                          │          │
│  │  Scan ──────► Read from RelationStore                    │          │
│  │  Filter ────► GPU: mask DAG + compact                    │          │
│  │  Project ───► GPU: column selection + arithmetic         │          │
│  │  Join ──────► GPU: hash_join_v2 (build/probe)           │          │
│  │  GroupBy ───► GPU: sort → boundaries → ids/keys → agg    │          │
│  │  Union ─────► GPU: concat → sort → dedup                 │          │
│  │  Diff ──────► GPU: sort both → binary search mark        │          │
│  │  Distinct ──► GPU: sort → mark dups → compact            │          │
│  │  Fixpoint ──► Loop until convergence (semi-naive)        │          │
│  └──────────────────────────────────────────────────────────┘          │
│                          │                                              │
│                          ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ GPU KERNEL PROVIDER                                              │   │
│  │                                                                  │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │   │
│  │  │ JOIN    │ │ FILTER  │ │ SORT    │ │ GROUPBY │ │ SET_OPS │   │   │
│  │  │ .cu     │ │ .cu     │ │ .cu     │ │ .cu     │ │ .cu     │   │   │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘   │   │
│  │       │           │           │           │           │         │   │
│  │       └───────────┴───────────┼───────────┴───────────┘         │   │
│  │                               ▼                                  │   │
│  │                    ┌─────────────────────┐                       │   │
│  │                    │ GPU MEMORY          │                       │   │
│  │                    │ (Column-Oriented)   │                       │   │
│  │                    │                     │                       │   │
│  │                    │ CudaBuffer {        │                       │   │
│  │                    │   columns: Vec<>    │                       │   │
│  │                    │   num_rows          │                       │   │
│  │                    │   schema            │                       │   │
│  │                    │ }                   │                       │   │
│  │                    └─────────────────────┘                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                          │                                              │
│                          ▼                                              │
│  ┌─────────────────┐                                                    │
│  │ RELATION STORE  │  HashMap<String, CudaBuffer>                      │
│  │                 │  Intermediate and final relations                  │
│  └────────┬────────┘                                                    │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │ RESULT          │  CudaBuffer (GPU-resident query result)           │
│  └─────────────────┘                                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Error Handling

```rust
pub enum XlogError {
    /// Syntax errors during parsing
    Parse(String),

    /// Unstratifiable negation (cycle through negation)
    StratificationCycle(Vec<String>),

    /// Domain safety violation (variable not bound in positive atoms)
    UnsafeVariable(String),

    /// Memory budget exceeded
    ResourceExhausted {
        context: String,
        estimated_bytes: u64,
        budget_bytes: u64,
    },

    /// CUDA kernel errors
    Kernel(String),

    /// Type mismatches
    Type(String),

    /// Semantic errors during compilation
    Compilation(String),

    /// Runtime failures
    Execution(String),
}
```

---

## Configuration

### Memory Budget

```rust
// Use 80% of total device memory
let total_device_bytes = device.total_memory()?;
let budget = MemoryBudget::from_device_memory(total_device_bytes);

// Fixed limit (e.g., 4 GB)
let budget = MemoryBudget::with_limit(4 * 1024 * 1024 * 1024);
```

### Execution Limits

```rust
// Maximum fixpoint iterations (prevent infinite loops)
const MAX_FIXPOINT_ITERATIONS: usize = 1000;
const MAX_SCC_ITERATIONS: usize = 1000;
```

---

## Testing

### Workspace Tests

```bash
# Debug mode
cargo test --workspace --all-targets --exclude pyxlog

# Release mode (recommended for GPU tests)
cargo test --workspace --all-targets --exclude pyxlog --release
```

### CUDA Certification Suite

The certification suite validates all CUDA kernel operations:

```bash
cargo test -p xlog-cuda-tests --test certification_suite --release -- --nocapture
```

**Coverage**: 140 tests covering hash joins, filter, sort, dedup, groupby, scan, pack, set_ops, circuit evaluation, and Monte Carlo sampling.

---

## See Also

### Core Documentation

- [Roadmap](ROADMAP.md) — Feature status, planned development, glossary

### Architecture Documents

| Document | Description |
|----------|-------------|
| [GPU Execution](architecture/gpu-execution.md) | GPU-resident filter, groupby, and arithmetic evaluation |
| [Query Optimizer](architecture/query-optimizer.md) | Cost-based join ordering, predicate pushdown, statistics |
| [Arithmetic Expressions](architecture/arithmetic-expressions.md) | `is` syntax, type inference, GPU evaluation |
| [Probabilistic Tier](architecture/xlog-prob.md) | Exact inference (Decision-DNNF/XGCF) and Monte Carlo sampling |
| [dILP Training](architecture/dilp-training.md) | Differentiable ILP trainer architecture and the RFC-backed execution model |
| [Solver Services](architecture/solver-services.md) | GPU CDCL verifier (zero host reads) + CLS SAT/MaxSAT services |
| [Adaptive Indexing](architecture/adaptive-indexing.md) | HISA-based heat tracking and index selection |
| [Multi-GPU Joins](architecture/multi-gpu-join.md) | Distributed join design (planned) |
| [Data Interoperability](architecture/cudf-interop.md) | Arrow IPC/C Data and DLPack integration |
| [CUDA Certification](architecture/cuda-certification.md) | PTX kernel test suite (140 tests, 24 categories) |
| [CLI Reference](architecture/cli-reference.md) | `xlog run` and `xlog prob` commands |
| [Python Bindings](architecture/python-bindings.md) | PyO3 + DLPack API (`pyxlog` module) |
