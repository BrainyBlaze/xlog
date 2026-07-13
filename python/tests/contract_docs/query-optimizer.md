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

Heat-based index selection is wired into the runtime through the persistent
hash index manager. The optimizer/runtime records heat and the executor can
reuse build-side hash indexes across repeated session evaluations when the
memory budget allows it. The current decision logic is:

- If relation heat crosses the size-adjusted threshold and no valid index
  exists: build a persistent hash index.
- If a relation generation, schema, key, or device changes: invalidate or miss
  the stale key.
- If retained bytes exceed budget: evict least-recently used indexes.

See [`adaptive-indexing.md`](adaptive-indexing.md) for the runtime manager.

### 5. Common Subexpression Elimination

Runtime common subexpression elimination is available for safe deterministic RIR
subplans. The control is explicit:

- `RuntimeConfig::with_common_subexpression_elimination(Some(true))`
  enables CSE for one runtime.
- `RuntimeConfig::with_common_subexpression_elimination(Some(false))`
  disables it for A/B comparison.
- `XLOG_CSE=1` enables it when the config field is `None`.

The cache lives in `xlog-runtime::Executor`, not in a separate evaluator.
`execute_node` builds a structural key for cacheable subplans, executes the
existing runtime/provider path on a miss, and returns a device-to-device clone
of the cached `CudaBuffer` on a hit. Relation scans in the key include the
current `RelationStore` generation, and the cache is cleared on relation
mutation and plan execution boundaries.

Cacheable deterministic nodes:

- `Filter`, keyed by input key plus predicate structure.
- `Project`, keyed by input key plus projection expressions.
- inner `Join`, keyed by input keys and join columns.
- `Union` and `Distinct`, keyed by child keys and identity columns.

Unsafe boundaries are rejected with diagnostics rather than shared:

- non-inner joins and `Diff` use the `negation_or_difference_boundary` /
  `negation_or_outer_join_boundary` classes;
- `GroupBy` uses `aggregate_boundary`;
- `TensorMaskedJoin` uses `provenance_or_tensor_boundary`;
- `Fixpoint` uses `recursive_or_mutable_boundary`;
- specialized `MultiWayJoin` and `ChainJoin` use
  `specialized_dispatch_boundary`.

The common-subexpression-elimination evidence records output parity, generation
invalidation, unsafe-boundary rejection, CSE hit/miss telemetry, and zero added
data-plane device-to-host calls for the duplicated-subplan fixture.

### 6. Adaptive Runtime Re-Optimization

Adaptive runtime re-optimization adds an adoption gate for compiler-supplied
candidate plans. The runtime does not reparse or recompile source text inside
`Executor`; instead, callers compile a baseline plan and a candidate plan using
the existing compiler and `StatsSnapshot` feedback path. `Executor` then owns
the runtime safety checks:

- `RuntimeConfig::with_adaptive_reoptimization(Some(true))` enables candidate
  adoption for one runtime.
- `RuntimeConfig::with_adaptive_reoptimization(Some(false))` disables it for
  A/B comparison.
- `XLOG_ADAPTIVE_REOPT=1` enables it when the config field is `None`.
- `XLOG_ADAPTIVE_REOPT_MIN_RATIO` overrides the deterministic mis-plan ratio
  threshold; unset defaults to `1.2`.

The baseline always runs first through `Executor::execute_plan`, which records
join observations before updating `StatsManager`: estimated rows, actual rows,
cardinality delta, estimated/actual selectivity, relation heat, heat delta, and
the deterministic mis-plan ratio. If the maximum ratio crosses the threshold,
the candidate runs through the same `execute_plan` path. Candidate outputs are
compared with the baseline snapshot by GPU full-row set difference in both
directions; only metadata/control-plane row counts are read. A candidate that
fails execution or diverges rolls back the baseline relation/statistics
snapshot and records a typed diagnostic.

This keeps adaptive execution inside the existing runtime/provider dispatch
surface while allowing the compiler to keep owning plan construction.

### Epistemic/Solver Substrate Handoff

The epistemic/solver line consumes the completed production runtime primitives
rather than introducing a private execution path:

- exact induction should use the typed native `U64`, `U32`, and `Symbol`
  dispatch recorded in the native exact-induction type-dispatch evidence;
- chain-shaped exact scorers should use the profile-gated shared-memory scorer
  only when the topology gate fires;
- duplicated deterministic solver subplans should use runtime CSE and its
  unsafe-boundary diagnostics;
- adaptive solver candidates should be compiled outside the executor and
  passed through `Executor::execute_plan_with_adaptive_candidate`;
- repeated solver joins should use the persistent hash-index manager and its
  relation-generation/schema/device keys.

The consumer certification fixture documents the public `.xlog` shape used for
this handoff. Full asynchronous recorded persistent-index builds are not claimed
by the current runtime substrate; the current manager records background-build
request/completion telemetry on the existing provider build/reuse path.

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

For runtime adaptive re-optimization, callers can compile a second candidate
with a previous `StatsSnapshot` and pass both plans to
`Executor::execute_plan_with_adaptive_candidate`. The executor adopts the
candidate only when deterministic telemetry and GPU equivalence checks pass.

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
- Magic sets transformation for top-down evaluation
- Adaptive query re-optimization during execution

## See Also

- [Adaptive Indexing](adaptive-indexing.md) — HISA index management
- [GPU Execution](gpu-execution.md) — Runtime execution model
