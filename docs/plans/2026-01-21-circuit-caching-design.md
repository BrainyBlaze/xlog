# Circuit Caching for 100% GPU Training

## Overview

Eliminate D4 recompilation bottleneck by caching compiled circuits and reusing them across training iterations. Currently, every `forward_backward()` call invokes D4 (CPU), causing 50+ D4 compilations per epoch instead of 1.

**Status**: Design ready for implementation

## Problem

```
forward_backward_complex() @ xlog-gpu-py/src/lib.rs:943
    → ExactDdnnfProgram::compile_source_with_gpu()  // D4 called EVERY time
    → 50 queries/epoch × 120ms = 6 seconds of CPU blocking
```

## Solution

Cache compiled circuits by **template signature**. Circuits with identical structure (same inputs, same label counts) share the same compiled DNNF - only weights differ.

## Cache Key Design

Based on DeepProbLog examples analysis, cache key must handle:
- Multiple networks with different label counts (HWF, CLUTRR, Forth)
- Variable input indices per query
- Different query predicates

**Cache Key Format**:
```
{predicate}:{input_count}:{network_signatures}
```

Example for `addition(0, 1, 7)`:
```
addition:2:mnist_net[10]
```

Example for HWF `formula(0, 1, 2, 42)` (number, operator, number, result):
```
formula:3:net1[10],net2[4],net1[10]
```

## Data Structures

```rust
// Add to CompiledProgram struct
pub struct CompiledProgram {
    // ... existing fields ...

    /// Cache of compiled circuits by template signature
    circuit_cache: HashMap<String, CachedCircuit>,
}

pub struct CachedCircuit {
    /// The compiled program (contains XGCF on GPU)
    program: ExactDdnnfProgram,

    /// Mapping from network output index to leaf ID in circuit
    /// Enables weight updates without recompilation
    weight_slots: Vec<WeightSlot>,
}

pub struct WeightSlot {
    /// Which network produced this weight
    network_name: String,
    /// Index into network output (0-9 for digits)
    output_index: usize,
    /// Leaf ID in the XGCF circuit
    leaf_id: LeafId,
}
```

## Algorithm

### Cache Lookup Flow

```
forward_backward_complex(query):
    1. Parse query → extract input indices
    2. Run neural networks → get output probabilities
    3. Generate template key from (predicate, inputs, network signatures)

    4. IF key in cache:
         a. Get cached circuit
         b. Update weight buffer with new network outputs
         c. Evaluate on GPU (no D4!)
         d. Return result

    5. ELSE (first time):
         a. Generate expanded source with annotated disjunctions
         b. Compile with D4 (one-time cost)
         c. Extract weight slot mapping
         d. Store in cache
         e. Evaluate and return
```

### Weight Update (No Recompilation)

```rust
fn update_weights(
    cached: &CachedCircuit,
    network_outputs: &[(String, Vec<f64>)],  // (network_name, probs)
) -> Vec<f64> {
    let mut weights = vec![0.0; cached.weight_slots.len()];

    for slot in &cached.weight_slots {
        let probs = network_outputs.get(&slot.network_name);
        weights[slot.leaf_id] = probs[slot.output_index].ln();
    }

    weights
}
```

## Files to Modify

| File | Change |
|------|--------|
| `crates/xlog-gpu-py/src/lib.rs` | Add cache to CompiledProgram, modify forward_backward_complex |
| `crates/xlog-prob/src/exact.rs` | Add method to update weights without recompile |

## Performance Impact

| Metric | Before | After |
|--------|--------|-------|
| D4 calls/epoch (10 queries) | 50 | **1** |
| Time per epoch | ~6.75s | **~0.87s** |
| Speedup | - | **7.7x** |

For 100 unique queries, 100 epochs:
- Before: 10,000 D4 calls
- After: ~10-20 D4 calls (one per unique structure)
- **Speedup: 500-1000x** for D4 time

## Edge Cases

1. **Cache invalidation**: Not needed - circuit structure is immutable once compiled
2. **Memory pressure**: Add optional `max_cache_size` with LRU eviction
3. **Different query values**: Same key (structure matters, not specific values)

## Testing Strategy

1. Verify cache hit on repeated queries
2. Verify different input indices → same cache entry
3. Verify different label counts → different cache entries
4. Benchmark: measure D4 calls before/after
5. Correctness: probabilities match non-cached version

## Success Criteria

1. D4 called only once per unique circuit structure
2. Training speed improved 5x+ on MNIST addition
3. All existing tests pass
4. Memory overhead < 10MB for typical workloads
