# XLOG System Architecture

XLOG is a **GPU-accelerated Datalog query engine** built in Rust with CUDA kernels. It compiles Datalog programs into relational algebra plans and executes them efficiently on NVIDIA GPUs.

## Table of Contents

1. [System Overview](#system-overview)
2. [Crate Structure](#crate-structure)
3. [Data Types](#data-types)
4. [Compilation Pipeline](#compilation-pipeline)
5. [Execution Pipeline](#execution-pipeline)
6. [CUDA Kernels](#cuda-kernels)
7. [Memory Management](#memory-management)
8. [Key Algorithms](#key-algorithms)
9. [Public APIs](#public-apis)
10. [Dataflow Diagram](#dataflow-diagram)

---

## System Overview

```
Datalog Source → Parser → Stratifier → Lowerer/Optimizer → Executor → GPU Kernels → Results
```

XLOG transforms declarative Datalog rules into efficient GPU-parallel operations:

- **Parsing**: PEG-based grammar with Pest
- **Stratification**: Ensures safe negation ordering via SCC analysis
- **Lowering/Optimization**: Converts AST to Relational IR and applies rewrites (predicate pushdown, join planning)
- **Execution**: Interprets RIR nodes using GPU kernels
- **GPU Kernels**: CUDA implementations of joins, sorts, aggregations, set operations

---

## Crate Structure

```
xlog/
├── crates/
│   ├── xlog-core/       # Foundation types, traits, error handling
│   ├── xlog-ir/         # Intermediate representations (RIR nodes)
│   ├── xlog-logic/      # Datalog frontend (parser, compiler)
│   ├── xlog-runtime/    # Query executor, relation storage
│   ├── xlog-cuda/       # CUDA provider, memory management, interop (Arrow/DLPack)
│   ├── xlog-stats/      # Runtime stats snapshots (optimizer + adaptive indexing)
│   ├── xlog-solve/      # Solver services (CLS SAT/MaxSAT MVP)
│   └── xlog-cuda-tests/ # CUDA/PTX certification suite (not published)
└── kernels/             # CUDA source files (.cu) + embedded PTX (.ptx)
```

### Dependency Graph

```
xlog-logic ───────┐
xlog-ir ──────────┼──> xlog-runtime ──┐
xlog-stats ───────┘                   ├──> xlog-cuda
xlog-core  <──────────────────────────┘

xlog-solve ───────────────┬──────────────> xlog-cuda
                           └──────────────> xlog-core

xlog-cuda-tests ──────────────────────────> xlog-cuda (+ xlog-core)
```

### Crate Responsibilities

| Crate | Purpose |
|-------|---------|
| `xlog-core` | Shared types (`ScalarType`, `Schema`, `AggOp`), traits (`KernelProvider`), errors |
| `xlog-ir` | Relational IR nodes (`RirNode`), expressions (`Expr`), execution plans |
| `xlog-logic` | Parser, stratification, lowering (AST → RIR), optimizer (predicate pushdown + join planning) |
| `xlog-runtime` | `Executor`, versioned `RelationStore`, profiling, incremental maintenance hooks, adaptive join index cache |
| `xlog-cuda` | `CudaKernelProvider`, `GpuMemoryManager`, `CudaBuffer`/`CudaColumn`, PTX embedding, Arrow IPC + DLPack interop |
| `xlog-stats` | `StatsManager` + `StatsSnapshot` (compiler feedback + runtime tracking) |
| `xlog-solve` | Solver services (CLS SAT/MaxSAT MVP; not used by `xlog-logic` in v0.1.0) |
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
    Symbol,             // Dictionary-encoded strings
}
```

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
    LogSumExp,  // Numerically stable log-sum-exp
}
```

---

## Compilation Pipeline

### Phase 1: Parsing

**File**: `crates/xlog-logic/src/parser.rs`

Uses Pest PEG grammar to parse Datalog source into AST:

```rust
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

The compiler desugars these into ordinary rules so they run through the same stratification + lowering
pipeline:

- `:- body.` → `__xlog_constraint_N(1) :- body.`
- `?- p(1, X).` → `__xlog_query_N(X) :- p(1, X).`

The example runner (`crates/xlog-logic/examples/xlog_run.rs`) enforces that all constraint relations
are empty, and prints the query relations.

### Phase 2: Stratification

**File**: `crates/xlog-logic/src/stratify.rs`

Ensures safe negation ordering using dependency analysis:

1. Build dependency graph between predicates
2. Classify edges: Positive, Negative, Aggregate
3. Find SCCs using Tarjan's algorithm
4. Detect cycles through negation (stratification failure)
5. Topologically sort strata

```rust
let strata = stratify(&program)?;
```

**Output**: `Vec<Stratum>` ordered by dependencies

### Phase 3: Lowering

**File**: `crates/xlog-logic/src/lower.rs`

Transforms AST to Relational IR:

1. **Schema inference**: Derive column types from facts
2. **Join planning**: Build join trees for positive atoms (bushy DP for small bodies, greedy for large bodies) using a simple cost model; can be seeded from runtime `StatsSnapshot`
3. **Variable tracking**: Map variables to column indices for join keys and projections
4. **Comparisons + arithmetic**: Lower comparisons to `Filter` and `is` expressions to computed projections (`ProjectExpr::Computed`)
5. **Negation handling**: Lower stratified negation via `Diff` + `Semi` join (anti-semi pattern over shared variables)
6. **Recursion**: Mark recursive predicate groups as SCCs in the plan; runtime executes recursive SCCs with semi-naive deltas (the compiler does not currently emit `RirNode::Fixpoint`)
7. **Projection**: Project result columns to match rule head (and lower aggregates to `GroupBy`)

```rust
lowerer.set_strata(strata_preds);
let plan = lowerer.lower_program(&program)?;
```

**Output**: `ExecutionPlan` with SCCs, strata, compiled rules

### Compiler Orchestration

**File**: `crates/xlog-logic/src/compile.rs`

```rust
pub struct Compiler {
    lowerer: Lowerer,
}

impl Compiler {
    pub fn compile(&mut self, source: &str) -> Result<ExecutionPlan> {
        // Phase 1: parse to AST
        let program = parse_program(source)?;

        // Compiler-internal desugaring of queries/constraints into internal rules happens here.

        // Phase 2: stratify + pass predicate groups into the lowerer
        let strata = stratify(&program)?;
        self.lowerer.set_strata(strata.into_iter().map(|s| s.predicates).collect());

        // Phase 3: lower to RIR plan
        let mut plan = self.lowerer.lower_program(&program)?;

        // Phase 4: optimizer rewrites (predicate pushdown, cost-aware join planning) happen here.

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
    stats: StatsManager,                  // Runtime stats (optimizer feedback)
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
        RirNode::Scan { rel } => self.execute_scan(*rel),
        RirNode::Filter { input, predicate } => {
            let buf = self.execute_node(input)?;
            self.execute_filter(&buf, predicate)
        }
        RirNode::Join { left, right, left_keys, right_keys, join_type } => {
            let l = self.execute_node(left)?;
            let r = self.execute_node(right)?;
            self.provider.hash_join_v2(&l, &r, left_keys, right_keys, *join_type)
        }
        // ... other operations
    }
}
```

### Recursive SCC Evaluation (Semi-Naive)

Recursive programs are executed at the **SCC level** (see `ExecutionPlan.sccs` and `ExecutionPlan.rules_by_scc`), using semi-naive deltas.

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
| `join.cu` | `hash_join_build`, `hash_join_probe` (legacy), `hash_join_bucket_count_v2`, `hash_join_scatter_v2`, `hash_join_probe_v2`, `hash_join_semi`, `hash_join_anti`, `init_hash_table`, `compute_composite_hash` | Hash joins (v2 default) + composite hashing |
| `pack.cu` | `pack_keys`, `pack_and_hash_keys`, `hash_packed_keys`, `gather_packed_rows`, `compare_packed_keys` | Key packing/hashing + packed-row utilities |
| `dedup.cu` | `mark_unique_*`, `compact_rows` | Sort-based deduplication |
| `filter.cu` | `filter_compare_*`, `compact_*_by_mask`, `mask_{and,or,not}` | Filtering and stream compaction |
| `sort.cu` | `radix_histogram`, `radix_scatter_*`, `init_indices`, `apply_permutation_*`, `gather_keys_*` | Stable radix sort + permutation apply |
| `groupby.cu` | `detect_group_boundaries`, `extract_group_keys`, `groupby_*`, `groupby_logsumexp_*` | Sorted aggregation |
| `scan.cu` | `exclusive_scan_mask`, `count_mask`, `multiblock_scan_*` | Prefix sum operations |
| `set_ops.cu` | `concat_{u32,bytes}`, `sorted_diff_mark` | Union/difference operations |

### Hash Join Implementation (v2 default)

The runtime uses a **bucketed “CSR buckets”** layout for join v2 (see `kernels/join.cu` + `CudaKernelProvider::hash_join_v2`):

- Compute a 64-bit composite hash for the join key columns (typically via packed key bytes in `kernels/pack.cu`).
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
    const uint8_t* data,
    const uint32_t* col_offsets,
    const uint32_t* col_sizes,
    uint32_t num_key_cols,
    uint32_t num_rows,
    uint32_t row_stride,
    uint64_t* hashes
) {
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
// Radix sort operates on u32 digit segments; passes depend on key width.
__global__ void radix_histogram(
    const uint32_t* keys,
    uint32_t* histograms,  // [grid_size * 16]
    uint32_t shift
) {
    __shared__ uint32_t local_hist[16];
    // Initialize shared memory
    if (threadIdx.x < 16) local_hist[threadIdx.x] = 0;
    __syncthreads();

    uint32_t digit = (keys[gid] >> shift) & 0xF;
    atomicAdd(&local_hist[digit], 1);
    __syncthreads();

    // Write to global
    if (threadIdx.x < 16)
        histograms[blockIdx.x * 16 + threadIdx.x] = local_hist[threadIdx.x];
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

Notes:
- `groupby_multi_agg` sorts by `key_cols` on GPU, then detects group boundaries over packed key bytes.
- Group IDs are currently computed on the host from the boundary mask (MVP).
- Multi-key groupby is supported by packing key columns into a byte key and detecting boundaries on-device; key packing currently requires a 4-byte segment width, so `Bool` keys are not supported.
- Current value-type support (MVP):
  - `count`: any value type (counts rows)
  - `sum`/`min`/`max`: `u32` values (output `u64` for `sum`, `u32` for `min`/`max`)
  - `logsumexp`: `f64` values (output `f64`)

---

## Memory Management

### CudaBuffer

**File**: `crates/xlog-cuda/src/memory.rs`

```rust
pub enum CudaColumn {
    Owned(TrackedCudaSlice<u8>),
    Dlpack(DlpackColumn), // calls the DLPack deleter on drop
}

pub struct CudaBuffer {
    pub columns: Vec<CudaColumn>, // Column-major, bytes; typed by `schema`
    pub num_rows: u64,
    pub schema: Schema,
}

impl CudaBuffer {
    pub fn column(&self, idx: usize) -> Option<&CudaColumn>;
    pub fn num_rows(&self) -> u64;
    pub fn arity(&self) -> usize;  // Number of columns
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
                .ok_or(XlogError::ResourceExhausted { ... })?;

            if new > self.budget.device_bytes {
                return Err(XlogError::ResourceExhausted { ... });
            }

            if self.allocated.compare_exchange(current, new, ...).is_ok() {
                break;
            }
        }

        // Allocate via cudarc; tracked slice decrements budget on drop
        self.device.inner().alloc::<T>(len)
    }
}
```

### Memory Budget Configuration

```rust
pub struct MemoryBudget {
    pub device_bytes: u64,      // Hard limit
    pub allow_ooc: bool,        // Out-of-core spill (future)
    pub abort_on_exceed: bool,  // Fail vs warn
}

impl MemoryBudget {
    pub fn from_device_memory(total: u64) -> Self {
        Self { device_bytes: total * 80 / 100, .. }  // 80% of device memory
    }

    pub fn with_limit(bytes: u64) -> Self { ... }
}
```

---

## Key Algorithms

### Hash Join

1. **Key prep**: Pack join key columns into row-major bytes and compute a 64-bit composite hash (FNV-1a) on GPU.
2. **Build phase (v2 buckets)**: Bucket build rows by low hash bits (count → scan offsets → scatter contiguous bucket entries).
3. **Probe phase**: For each probe row, scan the bucket range, compare hashes, and (optionally) verify key bytes to eliminate hash collisions.
4. **Materialization**: Gather left/right columns into the output schema; for left-outer joins, unmatched right columns are zero-filled (MVP null representation).

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
    edge(1, 2).
    edge(2, 3).
    reach(X, Y) :- edge(X, Y).
    reach(X, Z) :- reach(X, Y), edge(Y, Z).
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

// Execute
let result = executor.execute_plan(&plan)?;

// Read results
let reach = executor.store().get("reach").unwrap();
```

### Profiling

```rust
use xlog_runtime::{Profiler, OpStats};

let mut profiler = Profiler::new(true);  // enabled
profiler.record(OpStats::new("hash_join", 1000, 500, 1234, 8000));
println!("{}", profiler.summary());
```

---

## Dataflow Diagram

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
│  │  Filter ────► GPU: compare + compact                     │          │
│  │  Project ───► GPU: column selection                      │          │
│  │  Join ──────► GPU: hash_join_v2 (build/probe)           │          │
│  │  GroupBy ───► GPU: sort → boundaries → aggregation       │          │
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
    Parse(String),                    // Syntax errors
    StratificationCycle(Vec<String>), // Unstratifiable negation
    UnsafeVariable(String),           // Domain safety violation
    ResourceExhausted {               // Memory budget exceeded
        context: String,
        estimated_bytes: u64,
        budget_bytes: u64,
    },
    Kernel(String),                   // CUDA errors
    Type(String),                     // Type mismatches
    Compilation(String),              // Semantic errors
    Execution(String),                // Runtime failures
}
```

---

## Configuration

### Memory Budget

```rust
// Use 80% of total device memory (caller provides total bytes)
let total_device_bytes = /* query via CUDA driver */;
let budget = MemoryBudget::from_device_memory(total_device_bytes);

// Fixed limit
let budget = MemoryBudget::with_limit(4 * 1024 * 1024 * 1024); // 4 GB

// Allow out-of-core spill (future)
let budget = budget.with_ooc();
```

### Execution Limits

```rust
// Maximum fixpoint iterations (prevent infinite loops)
const MAX_FIXPOINT_ITERATIONS: usize = 1000;
const MAX_SCC_ITERATIONS: usize = 1000;
```

---

## Test Coverage

- Workspace unit/integration tests: `cargo test --workspace --all-targets` (debug) or `cargo test --workspace --all-targets --release`.
- CUDA/PTX certification suite (140 tests): `cargo test -p xlog-cuda-tests --test certification_suite --release -- --nocapture`.
