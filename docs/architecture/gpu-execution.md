# GPU-Resident Execution

This document describes XLOG's GPU-resident execution model, where filters, groupby operations, and arithmetic expressions are evaluated entirely on the GPU without CPU round-trips.

## Design Goals

1. **Eliminate CPU round-trips** in the deterministic execution path
2. **Preserve deterministic behavior** across all execution modes
3. **Explicit error reporting** with actionable diagnostics

## GPU Predicate Engine

The `Executor::execute_filter` implementation uses a GPU mask pipeline that supports the full `Expr` tree.

### Architecture

```
Expr Tree
    │
    ├─ Arithmetic nodes ───► GPU arithmetic kernels ───► Intermediate columns
    │
    ├─ Comparison nodes ───► Typed compare kernels ───► Boolean masks
    │
    └─ Boolean nodes ──────► mask_and/or/not ─────────► Composed mask
                                                              │
                                                              ▼
                                                    Stream compaction
                                                              │
                                                              ▼
                                                     Filtered CudaBuffer
```

### Mask DAG Evaluation

Predicates are lowered to a **mask DAG**:

1. **Typed compare kernels** generate boolean masks for comparisons (`Eq`, `Lt`, `Gt`, etc.)
2. **Boolean operators** (`mask_and`, `mask_or`, `mask_not`) compose masks
3. **Stream compaction** selects rows using multi-block prefix scan

This approach is deterministic, uses existing PTX modules, and scales across complex predicates without runtime compilation.

### Supported Types

Filter comparisons support all scalar types:
- `u32`, `u64`, `i32`, `i64`
- `f32`, `f64`
- `bool`, `symbol`

### Float Comparison Semantics

Float comparisons (`f32`, `f64`) use hybrid semantics:

| Operator | Semantics | `NaN == NaN` | `NaN > Inf` | `-0.0 < +0.0` |
|----------|-----------|--------------|-------------|---------------|
| `Eq`, `Ne` | IEEE 754 | `false` | N/A | N/A |
| `Lt`, `Le`, `Gt`, `Ge` | Total Order | N/A | `true` | `true` |

**IEEE 754 (equality)**: Standard IEEE behavior where `NaN` is not equal to anything, including itself.

**Total ordering (relational)**: Uses IEEE 754 `totalOrder` predicate semantics:

```
-NaN < -Inf < -MAX < ... < -MIN > -0.0 < +0.0 < ... < +MIN < +MAX < +Inf < +NaN
```

This ordering:
- Places negative `NaN` below all other values
- Places positive `NaN` above all other values
- Distinguishes `-0.0` from `+0.0` (unlike IEEE equality)
- Matches Rust's `f64::total_cmp` behavior

**Implementation**: Bit manipulation transforms floats to integers that sort correctly with signed comparison, avoiding branches for performance.

## GPU GroupBy Finalization

The groupby pipeline is fully GPU-resident, eliminating host round-trips for boundary detection and key extraction.

### Pipeline

```
Sorted input buffer
        │
        ▼
detect_group_boundaries ───► Boundary mask (1 at group starts)
        │
        ▼
GPU prefix sum ────────────► Group IDs per row
        │
        ▼
GPU prefix sum (boundaries) ► Group start indices
        │
        ▼
gather_packed_rows ─────────► Group key columns
        │
        ▼
Aggregation kernels ────────► Aggregated values
        │
        ▼
Final CudaBuffer (keys + aggregates)
```

### Key Operations

| Operation | Kernel | Description |
|-----------|--------|-------------|
| Boundary detection | `detect_group_boundaries` | Compare adjacent packed key bytes |
| Group ID assignment | `multiblock_scan_*` | Prefix sum over boundary mask |
| Key extraction | `gather_packed_rows` | Gather first row per group |
| Aggregation | `groupby_sum`, `groupby_min`, etc. | Per-group reduction |

### Supported Aggregations

| Operation | Input Types | Output Type |
|-----------|-------------|-------------|
| `count` | Any | `u64` |
| `sum` | `u32`, `u64`, `i32`, `i64` | `u64` / `i64` |
| `min` | All numeric | Same as input |
| `max` | All numeric | Same as input |
| `logsumexp` | `f64` | `f64` |

## GPU Arithmetic Evaluation

Arithmetic expressions from `is` clauses are evaluated on the GPU as computed projections.

### Supported Operations

| Category | Operations |
|----------|------------|
| Binary | `+`, `-`, `*`, `/`, `%` |
| Unary | `abs`, negation |
| Functions | `min`, `max`, `pow`, `cast` |

### Type Handling

- **Same-type requirement**: Binary operators require matching operand types
- **Explicit casting**: Use `cast(expr, type)` for type conversion
- **pow() returns f64**: Always produces floating-point result

### Error Semantics

| Condition | Integer Behavior | Float Behavior |
|-----------|------------------|----------------|
| Division by zero | `INT64_MAX` | `NaN` / `Inf` |
| Overflow | Wraps (standard) | IEEE 754 |

## Multi-Block Prefix Scan

Large inputs (>256 elements per block) use a 3-phase multi-block scan:

```
Phase 1: Per-block local scan
    │
    ▼
Phase 2: Scan of block totals
    │
    ▼
Phase 3: Add block offsets to local results
```

This removes the 256-element limit from the original single-block implementation.

## Memory Flow

All intermediate buffers remain GPU-resident:

```
Input CudaBuffer
    │
    ├─► Filter evaluation (masks on GPU)
    │
    ├─► Arithmetic evaluation (columns on GPU)
    │
    ├─► Compaction (output on GPU)
    │
    └─► Output CudaBuffer
```

Host involvement is limited to:
- Kernel launch orchestration
- Error checking
- Final output conversion (if requested)

## Integration with Runtime

The `Executor` uses these GPU-resident paths automatically:

```rust
fn execute_filter(&mut self, buf: &CudaBuffer, predicate: &Expr) -> Result<CudaBuffer> {
    // 1. Build mask DAG from predicate tree
    // 2. Evaluate arithmetic subexpressions on GPU
    // 3. Apply typed comparisons on GPU
    // 4. Compose boolean masks on GPU
    // 5. Stream compact on GPU
    // Returns: GPU-resident filtered buffer
}

fn execute_groupby(&mut self, buf: &CudaBuffer, keys: &[usize], aggs: &[(usize, AggOp)])
    -> Result<CudaBuffer>
{
    // 1. Sort by key columns on GPU
    // 2. Detect boundaries on GPU
    // 3. Compute group IDs on GPU
    // 4. Extract keys on GPU
    // 5. Run aggregations on GPU
    // Returns: GPU-resident result buffer
}
```

## See Also

- [CUDA Kernels](../ARCHITECTURE.md#cuda-kernels) — Kernel file reference
- [Memory Management](../ARCHITECTURE.md#memory-management) — GPU memory allocation
