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
Datalog Source → Parser → Stratifier → Lowerer → Executor → GPU Kernels → Results
```

XLOG transforms declarative Datalog rules into efficient GPU-parallel operations:

- **Parsing**: PEG-based grammar with Pest
- **Stratification**: Ensures safe negation ordering via SCC analysis
- **Lowering**: Converts AST to Relational IR (join trees, projections, filters)
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
│   └── xlog-cuda/       # GPU kernels, memory management
└── kernels/             # CUDA source files (.cu)
```

### Dependency Graph

```
xlog-logic ──┐
             ├──> xlog-runtime ──┐
xlog-ir ─────┤                   ├──> xlog-cuda
xlog-core <──┴───────────────────┘
```

### Crate Responsibilities

| Crate | Purpose |
|-------|---------|
| `xlog-core` | Shared types (`ScalarType`, `Schema`, `AggOp`), traits (`KernelProvider`), errors |
| `xlog-ir` | Relational IR nodes (`RirNode`), expressions (`Expr`), execution plans |
| `xlog-logic` | Parser, stratification, lowering (AST → RIR) |
| `xlog-runtime` | `Executor`, `RelationStore`, `Profiler` |
| `xlog-cuda` | `CudaKernelProvider`, `GpuMemoryManager`, `CudaBuffer` |

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
2. **Join tree construction**: Build left-deep join trees from rule bodies
3. **Variable tracking**: Map variables to column indices for join keys
4. **Negation handling**: Convert negated atoms to `Diff` nodes
5. **Recursion wrapping**: Wrap recursive rules in `Fixpoint` nodes
6. **Projection**: Project result columns to match rule head

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
    store: RelationStore,                 // Named relation storage
    rel_names: HashMap<RelId, String>,    // RelId → name mapping
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

### Fixpoint Iteration (Recursion)

Semi-naive evaluation algorithm:

```rust
fn execute_recursive_scc(&mut self, rules: &[CompiledRule]) -> Result<()> {
    // 1. Execute all rules once (initial results)
    for rule in rules {
        let result = self.execute_node(&rule.body)?;
        self.store.put(&rule.head, self.provider.dedup(&result)?);
    }

    // 2. Iterate until fixpoint
    loop {
        let mut any_changed = false;
        for rule in rules {
            let old_count = self.store.get(&rule.head).map(|b| b.num_rows());
            let new_result = self.execute_node(&rule.body)?;
            let merged = self.provider.union(existing, &new_result)?;
            let deduped = self.provider.dedup(&merged)?;

            if deduped.num_rows() > old_count {
                any_changed = true;
            }
            self.store.put(&rule.head, deduped);
        }
        if !any_changed { break; }  // Fixpoint reached
    }
}
```

---

## CUDA Kernels

### Kernel Files

| File | Kernels | Purpose |
|------|---------|---------|
| `join.cu` | `hash_join_build`, `hash_join_probe`, `hash_join_v2`, `hash_join_semi`, `hash_join_anti`, `compute_composite_hash` | Hash joins with multi-column support |
| `dedup.cu` | `mark_duplicates`, `compact_unique` | Sort-based deduplication |
| `filter.cu` | `filter_compare_u32/i64/f64`, `compact_*_by_mask`, `mask_and/or/not` | Filtering and stream compaction |
| `sort.cu` | `radix_histogram`, `radix_scatter`, `apply_permutation_*` | 4-bit radix sort |
| `groupby.cu` | `detect_boundaries`, `groupby_count/sum/min/max`, `extract_group_keys` | Sorted aggregation |
| `scan.cu` | `block_inclusive_scan`, `exclusive_scan_mask`, `count_mask` | Prefix sum operations |
| `set_ops.cu` | `concat_u32`, `sorted_diff_mark` | Union/difference operations |

### Hash Join Implementation

**Build Phase**:
```cuda
__global__ void hash_join_build(
    const uint32_t* keys,
    uint32_t* hash_table,
    uint32_t* next_ptrs,      // Linked list for collisions
    uint32_t hash_table_size
) {
    uint32_t hash = keys[gid] % hash_table_size;
    // Atomic linked-list insertion
    next_ptrs[gid] = atomicExch(&hash_table[hash], gid + 1);
}
```

**Probe Phase**:
```cuda
__global__ void hash_join_probe(
    const uint32_t* probe_keys,
    const uint32_t* build_keys,
    uint32_t* hash_table,
    uint32_t* next_ptrs,
    uint32_t* output_left,
    uint32_t* output_right,
    uint32_t* match_count
) {
    uint32_t hash = probe_keys[gid] % hash_table_size;
    uint32_t idx = hash_table[hash];
    while (idx != 0) {
        if (build_keys[idx - 1] == probe_keys[gid]) {
            uint32_t out_idx = atomicAdd(match_count, 1);
            output_left[out_idx] = gid;
            output_right[out_idx] = idx - 1;
        }
        idx = next_ptrs[idx - 1];
    }
}
```

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

### Radix Sort (4-bit)

```cuda
// 8 passes for 32-bit keys (4 bits per pass)
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
- Multi-key groupby is supported by packing key columns into a byte key and detecting boundaries on-device.
- Current value-type support (MVP):
  - `count`: any value type (counts rows)
  - `sum`/`min`/`max`: `u32` values (output `u64` for `sum`, `u32` for `min`/`max`)
  - `logsumexp`: `f64` values (output `f64`)

---

## Memory Management

### CudaBuffer

**File**: `xlog-cuda/src/buffer.rs`

```rust
pub struct CudaBuffer {
    columns: Vec<CudaSlice<u8>>,  // Column-major storage
    num_rows: u64,
    schema: Schema,
}

impl CudaBuffer {
    pub fn column(&self, idx: usize) -> Option<&CudaSlice<u8>>;
    pub fn num_rows(&self) -> u64;
    pub fn arity(&self) -> usize;  // Number of columns
    pub fn schema(&self) -> &Schema;
    pub fn is_empty(&self) -> bool;
}
```

### GPU Memory Manager

**File**: `xlog-cuda/src/memory.rs`

```rust
pub struct GpuMemoryManager {
    device: Arc<CudaDevice>,
    budget: MemoryBudget,
    allocated: AtomicU64,
}

impl GpuMemoryManager {
    pub fn alloc<T>(&self, len: usize) -> Result<CudaSlice<T>> {
        let bytes = len * std::mem::size_of::<T>();

        // Atomic budget check with compare-exchange loop
        loop {
            let current = self.allocated.load(Ordering::SeqCst);
            let new = current.checked_add(bytes as u64)
                .ok_or(XlogError::ResourceExhausted { ... })?;

            if new > self.budget.device_bytes {
                return Err(XlogError::ResourceExhausted { ... });
            }

            if self.allocated.compare_exchange(current, new, ...).is_ok() {
                break;
            }
        }

        // Allocate via cudarc
        self.device.inner().alloc(len)
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

1. **Build phase**: Insert right relation into hash table with linked-list collision handling
2. **Probe phase**: For each left row, walk collision chain looking for matches
3. **Output**: Concatenated rows from both relations

**Join Types**:
- **Inner**: Output matching pairs
- **Semi**: Output left rows that have any match (existence check)
- **Anti**: Output left rows with no matches
- **LeftOuter**: Output all left rows, with NULLs for non-matches

### Sort-Based Deduplication

1. Sort rows by all columns (or key columns)
2. Mark duplicates (compare adjacent rows)
3. Stream compaction to remove marked rows

### Fixpoint Iteration (Semi-Naive)

1. Compute initial result from base case
2. Repeat:
   - Evaluate recursive rules using current delta
   - Compute new tuples: `delta_new = recursive_result - current_result`
   - If `delta_new` is empty, fixpoint reached
   - Otherwise: `result = result ∪ delta_new`, `delta = delta_new`

### Stream Compaction

1. Create mask array (1 = keep, 0 = discard)
2. Compute exclusive prefix sum of mask
3. Scatter elements to positions indicated by prefix sum

### Radix Sort

8 passes for 32-bit keys (4 bits per pass):
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
// Use 80% of device memory
let budget = MemoryBudget::from_device_memory(device.total_memory());

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

| Test Suite | Tests | Coverage |
|------------|-------|----------|
| `xlog-core` | 11 | Types, schemas, errors |
| `xlog-cuda` (unit) | 35 | Provider, buffer, memory |
| `xlog-cuda` (filter) | 6 | Filter operations |
| `xlog-cuda` (groupby) | 8 | Aggregations |
| `xlog-cuda` (join_v2) | 10 | All join types |
| `xlog-cuda` (scan) | 5 | Prefix sum |
| `xlog-cuda` (set_ops) | 15 | Union/diff |
| `xlog-cuda` (sort) | 6 | Radix sort |
| `xlog-cuda` (type_coverage) | 26 | Multi-type support |
| `xlog-logic` (e2e) | 11 | End-to-end Datalog |
| `xlog-runtime` | 71 | Executor operations |

**Total: ~275 tests**
