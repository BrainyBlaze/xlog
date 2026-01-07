# XLOG-Logic Implementation Design

**Date:** 2026-01-07
**Status:** Approved
**Scope:** Phase 0 (Foundations) + Phase 1 (xlog-logic MVP)

---

## Summary of Decisions

| Decision | Choice |
|----------|--------|
| Language | Rust + CUDA |
| Phase | 0 + 1 combined (foundations + xlog-logic) |
| Structure | Monorepo workspace |
| CUDA integration | Hybrid: cudarc + C++ kernels compiled to PTX |
| Scope | Full xlog-logic v0.1 spec (n-ary, recursion, stratified negation, aggregates) |
| Approach | Skeleton + iterate |

---

## 1. Project Structure

```
xlog/
├── Cargo.toml                    # Workspace definition
├── crates/
│   ├── xlog-core/                # Foundational types & traits
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── error.rs          # XlogError, Result alias
│   │       ├── config.rs         # RuntimeConfig, MemoryBudget
│   │       ├── types.rs          # ScalarType, Domain, Schema
│   │       └── traits.rs         # KernelProvider, RelationStore traits
│   │
│   ├── xlog-ir/                  # Intermediate representations
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── rir.rs            # Relational IR nodes
│   │       ├── metadata.rs       # Cardinality, skew, memory estimates
│   │       └── plan.rs           # Execution plan, SCC ordering
│   │
│   ├── xlog-cuda/                # GPU kernel provider
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── device.rs         # Device management, stream pools
│   │       ├── memory.rs         # GPU allocator, budget enforcement
│   │       ├── provider.rs       # CudaKernelProvider impl
│   │       └── kernels/          # Kernel wrappers (join, dedup, etc.)
│   │
│   ├── xlog-runtime/             # Execution engine
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── relation.rs       # GpuRelation, column storage
│   │       ├── executor.rs       # Plan executor, fixpoint loop
│   │       └── profiler.rs       # Row counts, memory tracking
│   │
│   └── xlog-logic/               # Datalog frontend
│       └── src/
│           ├── lib.rs
│           ├── parser.rs         # Pest grammar
│           ├── ast.rs            # Surface AST
│           ├── stratify.rs       # Negation stratification
│           ├── lower.rs          # AST → RIR
│           └── compile.rs        # Full compilation pipeline
│
├── kernels/                      # CUDA source files
│   ├── join.cu
│   ├── dedup.cu
│   ├── groupby.cu
│   └── CMakeLists.txt            # nvcc build
│
└── tests/                        # Integration tests
    └── logic/
        ├── tc.xlog               # Transitive closure
        └── stratified.xlog       # Negation tests
```

---

## 2. Core Types (`xlog-core`)

### Error Types

```rust
#[derive(Debug, thiserror::Error)]
pub enum XlogError {
    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Stratification failed: cycle through negation involving {0:?}")]
    StratificationCycle(Vec<String>),

    #[error("Domain safety: variable {0} not bound in positive literal")]
    UnsafeVariable(String),

    #[error("Resource exhausted: {context}, estimated {estimated_bytes} bytes, budget {budget_bytes} bytes")]
    ResourceExhausted { context: String, estimated_bytes: u64, budget_bytes: u64 },

    #[error("Kernel error: {0}")]
    Kernel(String),
}
```

### Scalar Types & Schema

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    U32, U64, I32, I64, F32, F64, Bool, Symbol,
}

#[derive(Debug, Clone)]
pub struct Schema {
    pub columns: Vec<(String, ScalarType)>,
    pub key_columns: Vec<usize>,
}
```

### Configuration

```rust
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    pub device_bytes: u64,
    pub allow_ooc: bool,
    pub abort_on_exceed: bool,
}

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub memory: MemoryBudget,
    pub deterministic: bool,
    pub profile: bool,
}
```

### Kernel Provider Trait

```rust
pub trait KernelProvider: Send + Sync {
    fn hash_join(&self, left: &GpuBuffer, right: &GpuBuffer,
                 left_keys: &[usize], right_keys: &[usize]) -> Result<GpuBuffer>;
    fn dedup(&self, input: &GpuBuffer, key_cols: &[usize]) -> Result<GpuBuffer>;
    fn union(&self, a: &GpuBuffer, b: &GpuBuffer) -> Result<GpuBuffer>;
    fn diff(&self, a: &GpuBuffer, b: &GpuBuffer) -> Result<GpuBuffer>;
    fn groupby_agg(&self, input: &GpuBuffer, key_cols: &[usize],
                   agg: AggOp, value_col: usize) -> Result<GpuBuffer>;
}
```

---

## 3. Relational IR (`xlog-ir`)

### Node Metadata

```rust
#[derive(Debug, Clone)]
pub struct RirMeta {
    pub schema: Schema,
    pub est_rows: (u64, u64),
    pub est_bytes: (u64, u64),
    pub skew: Option<SkewSignature>,
    pub deterministic: bool,
    pub layout_hint: LayoutHint,
}

#[derive(Debug, Clone, Copy, Default)]
pub enum LayoutHint {
    #[default]
    CudfTable,
    HisaIndexed,
    VflogColumnar,
}
```

### RIR Nodes

```rust
#[derive(Debug, Clone)]
pub enum RirNode {
    Scan { rel: RelId },
    Filter { input: Box<RirNode>, predicate: Expr },
    Project { input: Box<RirNode>, columns: Vec<usize> },
    Join {
        left: Box<RirNode>,
        right: Box<RirNode>,
        left_keys: Vec<usize>,
        right_keys: Vec<usize>,
        join_type: JoinType,
    },
    GroupBy {
        input: Box<RirNode>,
        key_cols: Vec<usize>,
        aggs: Vec<(usize, AggOp)>,
    },
    Union { inputs: Vec<RirNode> },
    Distinct { input: Box<RirNode>, key_cols: Vec<usize> },
    Diff { left: Box<RirNode>, right: Box<RirNode> },
    Fixpoint {
        scc_id: u32,
        base: Box<RirNode>,
        recursive: Box<RirNode>,
        delta_rel: RelId,
        full_rel: RelId,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum JoinType {
    Inner,
    LeftOuter,
    Semi,
    Anti,
}
```

---

## 4. GPU Memory Management (`xlog-cuda`)

### GPU Buffer

```rust
pub struct GpuBuffer {
    pub columns: Vec<GpuColumn>,
    pub num_rows: u64,
    pub schema: Schema,
}

pub struct GpuColumn {
    pub data: CudaSlice<u8>,
    pub dtype: ScalarType,
    pub null_mask: Option<CudaSlice<u8>>,
}
```

### Memory Manager

```rust
pub struct GpuMemoryManager {
    device: Arc<CudaDevice>,
    budget: MemoryBudget,
    allocated: Mutex<u64>,
}

impl GpuMemoryManager {
    pub fn allocate(&self, bytes: u64) -> Result<CudaSlice<u8>>;
    pub fn check_budget(&self, estimated_bytes: u64) -> Result<()>;
}
```

### Kernel Provider Implementation

- Load PTX at startup via `include_str!`
- Pre-flight budget check before each kernel
- Modules: join, dedup, groupby

---

## 5. Runtime Executor (`xlog-runtime`)

### Semi-Naive Fixpoint

```rust
fn execute_fixpoint(&mut self, base: &RirNode, recursive: &RirNode,
                    delta_rel: RelId, full_rel: RelId) -> Result<GpuBuffer> {
    // 1. Initialize: R = base, Δ = base
    // 2. Loop:
    //    a. Δnew = recursive(Δ) - R
    //    b. If Δnew = ∅, break
    //    c. R = R ∪ Δnew, Δ = Δnew
    // 3. Return R
}
```

---

## 6. Parser & Compiler (`xlog-logic`)

### Surface AST

- `Program`: domains, predicates, facts, rules, queries
- `Rule`: head atom + body literals
- `BodyLiteral`: Positive, Negated, Aggregate, Comparison

### Stratification

- Build dependency graph
- Tarjan's SCC
- Check for cycles through negation
- Topologically sort strata

### Lowering

- Body literals → Scan nodes
- Join tree construction (left-deep)
- Negation → Anti join
- Recursive strata → Fixpoint nodes

---

## 7. CUDA Kernels

### join.cu

- `hash_join_build`: Atomic linked-list insertion
- `hash_join_probe`: Walk linked lists, output matching pairs

### dedup.cu

- `mark_duplicates`: Compare adjacent sorted keys
- `compact`: Stream compaction via prefix sum

### groupby.cu

- `groupby_count`: Sorted input, detect group boundaries

### Build System

- CMake compiles .cu → .ptx
- PTX embedded in Rust binary
- Runtime loading via cudarc

---

## 8. Implementation Order

1. **Skeleton setup**: Cargo workspace, crate stubs, CI
2. **xlog-core**: Error types, config, traits
3. **xlog-ir**: RIR node definitions, metadata
4. **CUDA build**: CMakeLists, empty kernel stubs
5. **xlog-cuda**: Memory manager, kernel provider skeleton
6. **xlog-runtime**: Executor skeleton, fixpoint loop
7. **xlog-logic**: Parser (pest), AST, stratifier, lowerer
8. **Kernel implementations**: join, dedup, groupby
9. **Integration tests**: TC, stratified negation, aggregates
10. **Benchmarks**: Compare against reference CPU Datalog

---

## 9. Success Criteria

The MVP is complete when:

```prolog
% Transitive closure works
edge(1, 2). edge(2, 3). edge(3, 4).
reach(X, Y) :- edge(X, Y).
reach(X, Z) :- reach(X, Y), edge(Y, Z).
?- reach(1, N).  % Returns: 2, 3, 4

% Stratified negation works
node(1). node(2). node(3).
edge(1, 2).
isolated(X) :- node(X), not edge(X, _), not edge(_, X).
?- isolated(N).  % Returns: 3

% Aggregates work
edge(1, 2). edge(1, 3). edge(2, 4).
out_degree(X, count(Y)) :- edge(X, Y).
?- out_degree(1, N).  % Returns: 2
```

All queries execute on GPU with data remaining GPU-resident throughout.
