# Query Optimizer

This document describes XLOG's cost-based query optimizer, including join ordering, predicate pushdown, and integration with the unified statistics layer.

## Overview

The query optimizer transforms RIR (Relational IR) plans to minimize execution cost while preserving semantics. It operates during the lowering phase in `xlog-logic`.

```
AST Rules
    │
    ▼
Join Planning (DP or greedy)
    │
    ▼
Predicate Pushdown
    │
    ▼
Projection Pruning
    │
    ▼
Index Selection (via stats)
    │
    ▼
Optimized RIR Plan
```

## Optimization Passes

### 1. Predicate Pushdown

Moves filter conditions as early as possible in the plan tree:

- Push `Filter` below `Join` when predicate references single relation
- Push `Filter` below `Project` always
- Merge adjacent `Filter` nodes with `And`

**Example:**
```
Before:                     After:
Filter(p1 AND p2)          Join
  └─ Join                    ├─ Filter(p1)
       ├─ Scan(R)           │    └─ Scan(R)
       └─ Scan(S)            └─ Filter(p2)
                                  └─ Scan(S)
```

### 2. Join Ordering

The optimizer selects join order using dynamic programming for small rule bodies and greedy heuristics for large ones.

#### Cost Model

```rust
pub struct JoinCost {
    pub rows: u64,           // Estimated output cardinality
    pub cpu_cost: f64,       // Hash table build + probe
    pub gpu_mem: u64,        // Peak GPU memory needed
    pub transfers: u32,      // Host-GPU transfers (minimize!)
}
```

#### Algorithm Selection

| Rule Body Size | Algorithm | Complexity |
|----------------|-----------|------------|
| ≤10 atoms | Dynamic Programming (Selinger-style) | O(3^n) |
| >10 atoms | Greedy bushy join | O(n²) |

#### Join Tree Construction

For each pair of relations, the optimizer computes:
- Build cost (hash table construction)
- Probe cost (lookups)
- Output cardinality estimate

The DP algorithm finds the optimal bushy join tree minimizing total cost.

### 3. Projection Pruning

Removes columns not needed by downstream operators:
- Track column usage through the plan tree
- Insert projections to drop unused columns early
- Reduces memory footprint and improves cache utilization

### 4. Index Selection

Uses heat tracking from the statistics layer:
- If `heat > 0.7` and no index: build HISA index
- If `heat < 0.1` and has index: drop index (reclaim memory)

## Unified Statistics Layer

The optimizer integrates with `xlog-stats` for runtime feedback.

### Statistics Collection

```rust
pub struct RelationStats {
    pub rel_id: RelId,
    pub cardinality: u64,              // Row count
    pub byte_size: u64,                // Total memory footprint
    pub column_stats: Vec<ColumnStats>, // Per-column statistics
    pub heat: f32,                      // Access frequency (0.0-1.0)
    pub last_access: u64,              // Timestamp for LRU
    pub has_index: bool,               // HISA index exists?
}

pub struct ColumnStats {
    pub col_idx: usize,
    pub dtype: ScalarType,
    pub null_count: u64,
    pub distinct_estimate: u64,        // HyperLogLog estimate
    pub min_value: ConstValue,
    pub max_value: ConstValue,
    pub histogram: Option<CudaBuffer>, // Equi-depth histogram on GPU
}
```

### Collection Points

| Event | Statistics Updated |
|-------|-------------------|
| After `Scan` | Cardinality, heat |
| After `Join` | Selectivity model |
| After `Dedup` | Distinct estimates |

### Join Selectivity

```rust
pub struct JoinSelectivity {
    pub left_key: Vec<usize>,
    pub right_key: Vec<usize>,
    pub selectivity: f64,              // Estimated output/input ratio
    pub is_pk_fk: bool,                // Primary-foreign key join?
}
```

The optimizer uses observed selectivities to refine cardinality estimates for future queries.

## StatsSnapshot Feedback

Runtime statistics feed back into compilation for adaptive optimization:

```rust
// Compiler side
let mut compiler = Compiler::new();
compiler.set_stats_snapshot(runtime_stats);  // Seed from previous execution
let plan = compiler.compile(source)?;

// Runtime side (after execution)
let stats = executor.stats_snapshot();
// Store for next compilation
```

## GPU Key Packing

The optimizer assumes GPU-resident key packing for join cost estimation. Multi-column join keys are packed on-device using fused pack+hash kernels:

```
Column data (GPU)
    │
    ▼
pack_and_hash_kernel ───► Packed keys + 64-bit hashes (GPU)
    │
    ▼
Hash join build/probe (GPU)
```

This eliminates host roundtrips that would otherwise dominate join cost for multi-column keys.

## Cartesian Join Support

For rules with disconnected atoms (no shared variables), the optimizer generates a constant-key join:

```datalog
result(X, Y) :- a(X), b(Y).  % No shared variables
```

Lowered as a join on a constant key column, producing the Cartesian product.

## Configuration

```rust
pub struct OptimizerConfig {
    pub enable_pushdown: bool,         // Default: true
    pub enable_join_reorder: bool,     // Default: true
    pub dp_threshold: usize,           // Max atoms for DP (default: 10)
    pub stats_feedback: bool,          // Use runtime stats (default: true)
}
```

## Future Enhancements

Planned optimizer improvements (not yet implemented):

- Join reordering based on selectivity estimates
- Common subexpression elimination across rules
- Magic sets transformation for top-down evaluation
- Adaptive query re-optimization during execution

## See Also

- [Adaptive Indexing](adaptive-indexing.md) — HISA index management
- [GPU Execution](gpu-execution.md) — Runtime execution model
