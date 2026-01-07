# Phase 2: GPU Execution Engine Design

**Date:** 2026-01-07
**Status:** Approved
**Scope:** xlog-cuda, xlog-runtime, xlog-logic lowering, end-to-end integration

---

## Summary

Phase 2 connects the foundations from Phase 1 into a working GPU Datalog engine. This completes the MVP by implementing:

- GPU kernel provider (xlog-cuda)
- Plan executor with semi-naive fixpoint (xlog-runtime)
- AST → RIR lowering (xlog-logic)
- End-to-end query execution

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        xlog-logic                           │
│  ┌─────────┐    ┌──────────┐    ┌─────────┐    ┌─────────┐ │
│  │ Parser  │───▶│   AST    │───▶│ Lower   │───▶│ Compile │ │
│  │ (done)  │    │  (done)  │    │ (new)   │    │  (new)  │ │
│  └─────────┘    └──────────┘    └─────────┘    └─────────┘ │
└─────────────────────────────────────┬───────────────────────┘
                                      │ RIR + ExecutionPlan
                                      ▼
┌─────────────────────────────────────────────────────────────┐
│                       xlog-runtime                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Executor │───▶│ Fixpoint │───▶│ Profiler │              │
│  │  (new)   │    │   (new)  │    │   (new)  │              │
│  └─────┬────┘    └──────────┘    └──────────┘              │
└────────┼────────────────────────────────────────────────────┘
         │ KernelProvider trait
         ▼
┌─────────────────────────────────────────────────────────────┐
│                        xlog-cuda                            │
│  ┌─────────┐    ┌────────┐    ┌──────────────────────────┐ │
│  │ Device  │───▶│ Memory │───▶│ CudaKernelProvider      │ │
│  │  (new)  │    │ (new)  │    │ (join, dedup, groupby)  │ │
│  └─────────┘    └────────┘    └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
         │
         ▼ PTX (compiled .cu files from Phase 1)
```

**Data flow:**
1. Parse `.xlog` → AST
2. Stratify (done) → Strata ordering
3. Lower → RIR trees per stratum
4. Execute → GPU operations via CudaKernelProvider
5. Return → Query results

---

## 2. xlog-cuda Implementation

### Device Management

```rust
// device.rs
pub struct CudaDevice {
    device: Arc<cudarc::driver::CudaDevice>,
    stream: CudaStream,
}

impl CudaDevice {
    pub fn new(ordinal: usize) -> Result<Self>;
    pub fn synchronize(&self) -> Result<()>;
}
```

### Memory Management

```rust
// memory.rs
pub struct GpuMemoryManager {
    device: Arc<CudaDevice>,
    budget: MemoryBudget,
    allocated: AtomicU64,
}

impl GpuMemoryManager {
    pub fn alloc<T>(&self, len: usize) -> Result<CudaSlice<T>>;
    pub fn free<T>(&self, slice: CudaSlice<T>);
    pub fn check_budget(&self, bytes: u64) -> Result<()>;
}

pub struct CudaBuffer {
    pub columns: Vec<CudaSlice<u8>>,
    pub num_rows: u64,
    pub schema: Schema,
}
```

### Kernel Provider

```rust
// provider.rs
pub struct CudaKernelProvider {
    device: Arc<CudaDevice>,
    memory: Arc<GpuMemoryManager>,
    join_module: CudaModule,
    dedup_module: CudaModule,
    groupby_module: CudaModule,
}

impl KernelProvider for CudaKernelProvider {
    fn hash_join(&self, left: &GpuBuffer, right: &GpuBuffer,
                 left_keys: &[usize], right_keys: &[usize]) -> Result<GpuBuffer>;
    fn dedup(&self, input: &GpuBuffer, key_cols: &[usize]) -> Result<GpuBuffer>;
    fn union(&self, a: &GpuBuffer, b: &GpuBuffer) -> Result<GpuBuffer>;
    fn diff(&self, a: &GpuBuffer, b: &GpuBuffer) -> Result<GpuBuffer>;
    fn groupby_agg(&self, input: &GpuBuffer, key_cols: &[usize],
                   agg: AggOp, value_col: usize) -> Result<GpuBuffer>;
}
```

**PTX loading:** Embed compiled PTX via `include_str!` in build.rs, load at provider construction.

---

## 3. xlog-runtime Implementation

### Relation Storage

```rust
// relation.rs
pub struct RelationStore {
    relations: HashMap<String, CudaBuffer>,
}

impl RelationStore {
    pub fn get(&self, name: &str) -> Option<&CudaBuffer>;
    pub fn put(&mut self, name: &str, buffer: CudaBuffer);
    pub fn get_or_empty(&self, name: &str) -> CudaBuffer;
}
```

### Plan Executor

```rust
// executor.rs
pub struct Executor<K: KernelProvider> {
    provider: Arc<K>,
    store: RelationStore,
    profiler: Profiler,
}

impl<K: KernelProvider> Executor<K> {
    pub fn execute_plan(&mut self, plan: &ExecutionPlan) -> Result<CudaBuffer>;
    fn execute_node(&mut self, node: &RirNode) -> Result<CudaBuffer>;
    fn execute_stratum(&mut self, stratum: &Stratum) -> Result<()>;
    fn execute_fixpoint(&mut self, base: &RirNode, recursive: &RirNode,
                        delta_rel: &str, full_rel: &str) -> Result<CudaBuffer>;
}
```

### Semi-Naive Fixpoint

```rust
fn execute_fixpoint(&mut self, base: &RirNode, recursive: &RirNode,
                    delta_rel: &str, full_rel: &str) -> Result<CudaBuffer> {
    // 1. R = evaluate(base), Δ = R
    // 2. Loop:
    //    a. Δ_new = evaluate(recursive, using Δ) - R
    //    b. if Δ_new.is_empty() { break }
    //    c. R = union(R, Δ_new), Δ = Δ_new
    // 3. Return R
}
```

### Node Execution Mapping

| RirNode | KernelProvider method |
|---------|----------------------|
| Scan | store.get() |
| Join (Inner) | hash_join() |
| Join (Anti) | diff() |
| Union | union() |
| Distinct | dedup() |
| Diff | diff() |
| GroupBy | groupby_agg() |
| Fixpoint | loop with diff/union |

### Profiler

```rust
// profiler.rs
pub struct Profiler {
    enabled: bool,
    stats: Vec<OpStats>,
}

pub struct OpStats {
    pub op_name: String,
    pub input_rows: u64,
    pub output_rows: u64,
    pub duration_us: u64,
    pub memory_bytes: u64,
}
```

---

## 4. xlog-logic Lowering

### Lowerer

```rust
// lower.rs
pub struct Lowerer {
    schemas: HashMap<String, Schema>,
    strata: Vec<Vec<String>>,
}

impl Lowerer {
    pub fn lower_program(&mut self, program: &Program) -> Result<ExecutionPlan>;
    fn lower_rule(&mut self, rule: &Rule) -> Result<RirNode>;
    fn lower_body(&mut self, body: &[BodyLiteral]) -> Result<RirNode>;
    fn build_join_tree(&mut self, atoms: &[&Atom]) -> Result<RirNode>;
}
```

### Lowering Strategy

1. Start with first positive literal as Scan
2. Left-deep join tree for remaining positive literals
3. Apply filters for comparisons
4. Anti-join for negated literals
5. GroupBy for aggregates
6. Project to match head variables

### Examples

```prolog
% reach(X,Z) :- reach(X,Y), edge(Y,Z).
```
→
```
Project [0, 3]
  └─ Join (keys: [1] = [0])
       ├─ Scan "reach"
       └─ Scan "edge"
```

```prolog
% isolated(X) :- node(X), not edge(X, _).
```
→
```
Project [0]
  └─ Diff
       ├─ Scan "node"
       └─ Project [0]
            └─ Scan "edge"
```

### Compiler

```rust
// compile.rs
pub struct Compiler {
    lowerer: Lowerer,
}

impl Compiler {
    pub fn compile(&mut self, source: &str) -> Result<ExecutionPlan> {
        let program = parse_program(source)?;
        let strata = stratify(&program)?;
        self.lowerer.set_strata(strata);
        self.lowerer.lower_program(&program)
    }
}
```

---

## 5. Integration

### Top-Level API

```rust
pub struct XlogEngine {
    compiler: Compiler,
    executor: Executor<CudaKernelProvider>,
}

impl XlogEngine {
    pub fn new() -> Result<Self>;
    pub fn run(&mut self, source: &str) -> Result<QueryResults>;
}
```

---

## 6. Success Criteria

The MVP is complete when these tests pass:

```rust
#[test]
fn test_transitive_closure() {
    let mut engine = XlogEngine::new().unwrap();
    let results = engine.run(r#"
        edge(1, 2). edge(2, 3). edge(3, 4).
        reach(X, Y) :- edge(X, Y).
        reach(X, Z) :- reach(X, Y), edge(Y, Z).
        ?- reach(1, N).
    "#).unwrap();
    assert_eq!(results.values(), vec![2, 3, 4]);
}

#[test]
fn test_stratified_negation() {
    let mut engine = XlogEngine::new().unwrap();
    let results = engine.run(r#"
        node(1). node(2). node(3).
        edge(1, 2).
        isolated(X) :- node(X), not edge(X, _), not edge(_, X).
        ?- isolated(N).
    "#).unwrap();
    assert_eq!(results.values(), vec![3]);
}

#[test]
fn test_aggregates() {
    let mut engine = XlogEngine::new().unwrap();
    let results = engine.run(r#"
        edge(1, 2). edge(1, 3). edge(2, 4).
        out_degree(X, count(Y)) :- edge(X, Y).
        ?- out_degree(1, N).
    "#).unwrap();
    assert_eq!(results.values(), vec![2]);
}
```

All queries execute on GPU with data remaining GPU-resident throughout.

---

## 7. Implementation Order

1. xlog-cuda device management
2. xlog-cuda memory manager
3. xlog-cuda kernel provider (PTX loading)
4. xlog-cuda kernel wrappers (join, dedup, union, diff, groupby)
5. xlog-runtime relation store
6. xlog-runtime executor skeleton
7. xlog-runtime fixpoint evaluation
8. xlog-runtime profiler
9. xlog-logic lowerer
10. xlog-logic compiler
11. Integration tests (TC, negation, aggregates)
