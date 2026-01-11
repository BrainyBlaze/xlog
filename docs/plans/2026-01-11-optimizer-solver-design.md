# Unified Optimizer & Solver Design

**Date:** 2026-01-11
**Status:** Approved
**Author:** Claude + User collaboration

## Overview

This document describes the design for two major xlog enhancements:
1. **Query Optimizer** - Cost-based join ordering, predicate pushdown, adaptive indexing
2. **xlog-solve** - GPU-native FastFourierSAT-style Continuous Local Search solver

Both systems share a **Unified Statistics Layer** for GPU-resident relation metadata.

## Design Constraints

All four constraints must be satisfied:
- **Speed to MVP** - Phased delivery, working code fast
- **GPU-residency** - Minimize host transfers from the start
- **Correctness with proofs** - Verifiable results, certificates
- **Integration-ready** - Clean APIs for xlog-prob/xlog-elp

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Integration Layer                        │
│  (APIs for xlog-prob, xlog-elp, external consumers)        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────┐    ┌─────────────────────┐        │
│  │   Query Optimizer   │    │     xlog-solve      │        │
│  │  - Join ordering    │    │  - CLS solver       │        │
│  │  - Predicate push   │    │  - Proof generation │        │
│  │  - Plan selection   │    │  - Constraint API   │        │
│  └──────────┬──────────┘    └──────────┬──────────┘        │
│             │                          │                    │
│             └──────────┬───────────────┘                    │
│                        ▼                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Unified Statistics Layer                │   │
│  │  - Cardinality estimates    - Heat tracking         │   │
│  │  - Selectivity histograms   - Memory cost model     │   │
│  │  - Adaptive index manager   - GPU-resident stats    │   │
│  └─────────────────────────────────────────────────────┘   │
│                        │                                    │
├────────────────────────┼────────────────────────────────────┤
│                        ▼                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           GPU Kernel Layer (existing + new)          │   │
│  │  join | filter | sort | dedup | groupby | pack | CLS │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Key principle:** Statistics layer is GPU-resident. Both optimizer and solver read from it without host roundtrips.

---

## Component 1: Unified Statistics Layer

**New crate: `xlog-stats`**

### Data Structures

```rust
/// GPU-resident relation statistics
pub struct RelationStats {
    pub rel_id: RelId,
    pub cardinality: u64,              // Row count
    pub byte_size: u64,                // Total memory footprint
    pub column_stats: Vec<ColumnStats>, // Per-column statistics
    pub heat: f32,                      // Access frequency (0.0-1.0)
    pub last_access: u64,              // Timestamp for LRU
    pub has_index: bool,               // HISA index exists?
}

/// Per-column statistics (GPU-resident histograms)
pub struct ColumnStats {
    pub col_idx: usize,
    pub dtype: ScalarType,
    pub null_count: u64,
    pub distinct_estimate: u64,        // HyperLogLog estimate
    pub min_value: ConstValue,
    pub max_value: ConstValue,
    pub histogram: Option<CudaBuffer>, // Equi-depth histogram on GPU
}

/// Join selectivity model
pub struct JoinSelectivity {
    pub left_key: Vec<usize>,
    pub right_key: Vec<usize>,
    pub selectivity: f64,              // Estimated output/input ratio
    pub is_pk_fk: bool,                // Primary-foreign key join?
}
```

### Collection Points

- After each `Scan`: Update cardinality, heat
- After each `Join`: Update selectivity model
- After each `Dedup`: Update distinct estimates

---

## Component 2: Query Optimizer

**New module in `xlog-logic`: `optimizer.rs`**

### Optimization Passes

```rust
pub struct Optimizer {
    stats: Arc<StatsManager>,
    config: OptimizerConfig,
}

impl Optimizer {
    pub fn optimize(&self, plan: RirNode) -> RirNode {
        let plan = self.predicate_pushdown(plan);    // 1. Push filters early
        let plan = self.join_reorder(plan);          // 2. Optimal join order
        let plan = self.projection_pruning(plan);    // 3. Remove unused columns
        let plan = self.index_selection(plan);       // 4. Use HISA if hot
        plan
    }
}
```

### Join Ordering (Dynamic Programming)

```rust
/// Cost model for join ordering
pub struct JoinCost {
    pub rows: u64,           // Estimated output cardinality
    pub cpu_cost: f64,       // Hash table build + probe
    pub gpu_mem: u64,        // Peak GPU memory needed
    pub transfers: u32,      // Host-GPU transfers (minimize!)
}

/// DP-based join enumerator (Selinger-style)
fn find_best_join_order(
    relations: &[RelId],
    stats: &StatsManager,
) -> JoinTree {
    // For small plans (<10 relations): exhaustive DP
    // For large plans: greedy or genetic algorithm
}
```

### Predicate Pushdown Rules

- Push `Filter` below `Join` when predicate references single relation
- Push `Filter` below `Project` always
- Merge adjacent `Filter` nodes with `And`

### Index Selection

- If `heat > 0.7` and no index: build HISA
- If `heat < 0.1` and has index: drop index (reclaim memory)

---

## Component 3: GPU-Side Key Packing

**Problem:** Current `hash_join_v2` downloads keys to host, packs them, re-uploads.

**Solution:** New CUDA kernel for GPU-native key packing.

### New Kernel: `kernels/pack.cu`

```cuda
/// Pack multiple columns into row-major byte array (GPU-side)
__global__ void pack_keys_kernel(
    const uint8_t** columns,      // Array of column pointers
    const uint32_t* col_sizes,    // Byte size per column
    const uint32_t* col_offsets,  // Offset into packed row
    uint32_t num_cols,
    uint32_t num_rows,
    uint32_t row_size,            // Total packed row size
    uint8_t* packed_output        // Output: row-major packed keys
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    uint8_t* out_row = packed_output + row * row_size;
    for (uint32_t c = 0; c < num_cols; c++) {
        memcpy(out_row + col_offsets[c],
               columns[c] + row * col_sizes[c],
               col_sizes[c]);
    }
}

/// Compute FNV-1a hash of packed keys (fused with packing)
__global__ void pack_and_hash_kernel(
    /* same inputs */
    uint64_t* hashes_output       // Output: hash per row
) {
    // Pack + hash in single pass - better cache utilization
}
```

### Provider Integration

```rust
/// GPU-resident key packing (no host roundtrip)
fn pack_keys_gpu(
    &self,
    buffer: &CudaBuffer,
    key_cols: &[usize],
) -> Result<(CudaBuffer, CudaSlice<u64>)> {
    // Returns: (packed_keys, hashes) - all GPU-resident
}
```

**Performance impact:**
- Eliminates 2 host transfers per join
- Fused pack+hash improves cache locality
- Estimated 2-5x speedup for multi-column joins

---

## Component 4: xlog-solve (CLS Solver)

**New crate: `xlog-solve`**

FastFourierSAT-style Continuous Local Search - treats SAT as continuous optimization.

### Core Concept

- Variables are continuous values in [0,1] instead of boolean
- Clauses become differentiable loss functions
- GPU computes gradient-like updates in parallel
- Discretize to boolean when solution found

### Data Structures

```rust
/// Solve IR - constraint representation
pub struct SolveInstance {
    pub num_vars: u32,
    pub clauses: Vec<Clause>,           // CNF clauses
    pub weights: Option<Vec<f64>>,      // For MaxSAT/weighted
    pub objective: Objective,           // SAT, MaxSAT, or MinUnsat
}

pub struct Clause {
    pub literals: Vec<Literal>,         // Variable + polarity
}

pub struct Literal {
    pub var: u32,
    pub negated: bool,
}

/// GPU-resident solver state
pub struct SolverState {
    pub assignments: CudaSlice<f32>,    // Continuous [0,1] per variable
    pub velocities: CudaSlice<f32>,     // Momentum for updates
    pub clause_sat: CudaSlice<f32>,     // Satisfaction degree per clause
    pub gradients: CudaSlice<f32>,      // ∂loss/∂var
}

/// Solver configuration
pub struct SolverConfig {
    pub max_iterations: u32,            // Default: 10_000
    pub learning_rate: f32,             // Step size
    pub momentum: f32,                  // Velocity decay
    pub discretize_threshold: f32,      // When to snap to 0/1
}
```

### Loss Function

```
clause_loss(c) = ∏(1 - lit_value(l)) for l in c
               = 0 when clause satisfied
               > 0 when clause violated
```

### GPU Kernels: `kernels/solve.cu`

```cuda
/// Phase 1: Evaluate clause satisfaction (parallel per clause)
__global__ void cls_evaluate_clauses(
    const float* assignments,      // [num_vars]
    const int32_t* clause_lits,    // Packed literals
    const uint32_t* clause_offsets,// Start of each clause
    uint32_t num_clauses,
    float* clause_sat              // Output: satisfaction [0,1]
) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= num_clauses) return;

    float product = 1.0f;
    for (uint32_t i = clause_offsets[c]; i < clause_offsets[c+1]; i++) {
        int32_t lit = clause_lits[i];
        uint32_t var = abs(lit) - 1;
        float val = (lit > 0) ? assignments[var] : 1.0f - assignments[var];
        product *= (1.0f - val);
    }
    clause_sat[c] = product;
}

/// Phase 2: Compute gradients (parallel per variable)
__global__ void cls_compute_gradients(
    const float* assignments,
    const float* clause_sat,
    const int32_t* var_clauses,    // Which clauses mention var
    const uint32_t* var_offsets,
    uint32_t num_vars,
    float* gradients               // Output: ∂loss/∂var
) {
    uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vars) return;

    float grad = 0.0f;
    for (uint32_t i = var_offsets[v]; i < var_offsets[v+1]; i++) {
        grad += partial_derivative(v, var_clauses[i], ...);
    }
    gradients[v] = grad;
}

/// Phase 3: Update with momentum (parallel per variable)
__global__ void cls_update_assignments(
    float* assignments,
    float* velocities,
    const float* gradients,
    float learning_rate,
    float momentum,
    uint32_t num_vars
) {
    uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vars) return;

    velocities[v] = momentum * velocities[v] - learning_rate * gradients[v];
    assignments[v] = clamp(assignments[v] + velocities[v], 0.0f, 1.0f);
}
```

---

## Component 5: Proof Generation

### Proof Types

```rust
pub enum SolveProof {
    /// SAT: satisfying assignment is the proof
    Satisfying {
        assignment: Vec<bool>,
        checksum: u64,
    },

    /// UNSAT: resolution proof
    Unsatisfiable {
        learned_clauses: Vec<Clause>,
        resolution_chain: Vec<ResolutionStep>,
        checksum: u64,
    },

    /// Optimum: assignment + bound proof
    Optimal {
        assignment: Vec<bool>,
        objective_value: f64,
        bound_certificate: BoundCert,
    },

    /// Approximate: best-effort with quality
    Approximate {
        assignment: Vec<bool>,
        satisfied_clauses: u32,
        total_clauses: u32,
        iterations: u32,
    },
}
```

### GPU-Accelerated Verification

```rust
pub fn verify_proof_gpu(
    instance: &SolveInstance,
    proof: &SolveProof,
    provider: &CudaKernelProvider,
) -> Result<VerifyResult> {
    match proof {
        SolveProof::Satisfying { assignment, checksum } => {
            let sat_count = provider.count_satisfied_clauses(
                &instance.clauses,
                assignment,
            )?;
            Ok(sat_count == instance.clauses.len())
        }
        // ... other proof types
    }
}
```

---

## Component 6: Integration APIs

### Solver API

```rust
pub struct Solver {
    provider: Arc<CudaKernelProvider>,
    stats: Arc<StatsManager>,
    config: SolverConfig,
}

impl Solver {
    pub fn solve(&self, instance: SolveInstance) -> Result<SolveResult>;

    pub fn solve_with_timeout(
        &self,
        instance: SolveInstance,
        timeout: Duration,
    ) -> Result<SolveResult>;

    pub fn solve_batch(
        &self,
        instances: Vec<SolveInstance>,
    ) -> Result<Vec<SolveResult>>;

    pub fn solve_incremental(
        &self,
        state: &mut IncrementalState,
        new_clauses: Vec<Clause>,
    ) -> Result<SolveResult>;
}

pub struct SolveResult {
    pub status: SolveStatus,
    pub proof: SolveProof,
    pub stats: SolveStats,
}

pub enum SolveStatus {
    Sat,
    Unsat,
    Unknown,
    Optimal(f64),
}
```

### Optimizer API

```rust
pub struct QueryOptimizer {
    stats: Arc<StatsManager>,
    config: OptimizerConfig,
}

impl QueryOptimizer {
    pub fn optimize(&self, plan: ExecutionPlan) -> ExecutionPlan;
    pub fn explain(&self, plan: &ExecutionPlan) -> OptimizeExplanation;
    pub fn update_stats(&self, rel: RelId, actual_rows: u64);
}
```

---

## Implementation Plan

### Phase A: Statistics Foundation (Week 1)
- Create xlog-stats crate
- Implement RelationStats, ColumnStats
- Add collection points in executor
- Tests: stats accuracy, GPU-residency

### Phase B: GPU Key Packing (Week 1-2)
- Implement pack.cu kernels
- Integrate into hash_join_v2
- Remove host roundtrip path
- Tests: correctness, performance benchmarks

### Phase C: Query Optimizer (Week 2-3)
- Predicate pushdown pass
- Join reordering (DP for small, greedy for large)
- Index selection logic
- Tests: plan quality, regression suite

### Phase D: CLS Solver MVP (Week 3-5)
- Create xlog-solve crate
- Implement solve.cu kernels
- Basic SAT solving with proofs
- Tests: SAT competition benchmarks

### Phase E: Integration (Week 5-6)
- Wire optimizer into compiler pipeline
- Add solver APIs for xlog-prob
- End-to-end integration tests
- Documentation and examples

---

## Testing Strategy

| Component | Test Type | Coverage Target |
|-----------|-----------|-----------------|
| Statistics | Unit + property | Accuracy within 10% |
| Key packing | Correctness + perf | All type combinations |
| Optimizer | Plan regression | 50+ query patterns |
| CLS solver | SAT benchmarks | SATLIB instances |
| Proofs | Verification | 100% proof checking |
| Integration | E2E | Real xlog programs |

## Benchmarks

- Join throughput (rows/sec) before/after GPU packing
- Plan quality (cost reduction %)
- Solver iterations to convergence
- Memory efficiency (bytes/variable)

---

## Files to Create/Modify

### New Crates
- `crates/xlog-stats/` - Statistics infrastructure
- `crates/xlog-solve/` - CLS solver

### New Kernels
- `kernels/pack.cu` - GPU key packing
- `kernels/solve.cu` - CLS solver kernels

### Modified Files
- `crates/xlog-logic/src/optimizer.rs` - Query optimizer (new)
- `crates/xlog-cuda/src/provider.rs` - GPU packing integration
- `crates/xlog-runtime/src/executor.rs` - Stats collection
- `Cargo.toml` - Workspace members

---

## Success Criteria

1. **GPU-residency:** Zero host roundtrips in optimized join path
2. **Optimizer quality:** 2x+ improvement on multi-join queries
3. **Solver MVP:** Solve 3-SAT instances with 1000+ variables
4. **Proofs:** 100% verification pass rate
5. **Integration:** xlog-prob can call solver API
