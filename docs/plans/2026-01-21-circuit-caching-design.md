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

**v0.4.0-alpha scope:** Single network per program (matches existing `forward_backward_complex` limitation).

Cache key must handle:
- Variable input indices per query
- Different query predicates
- Different label counts (e.g., 10 for MNIST digits, 4 for operators)

**Cache Key Format (v0.4.0-alpha)**:
```
{predicate}:{input_count}:{num_labels}
```

Example for `addition(0, 1, 7)` with 10-class MNIST:
```
addition:2:10
```

**Future multi-network format** (post v0.4.0):
```
{predicate}:{input_count}:{network_signatures}
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

    /// Mapping from network output index to circuit variable
    /// Enables weight updates without recompilation
    weight_slots: Vec<WeightSlot>,
}

/// Maps a network output to a circuit variable (v0.4.0-alpha: single network)
pub struct WeightSlot {
    /// Input index in the query (e.g., 0 or 1 for addition(0, 1, 7))
    input_idx: usize,
    /// Label index (0-9 for 10-class classification)
    label_idx: usize,
    /// Circuit variable index (1-indexed, DIMACS convention)
    var_idx: u32,
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
fn update_circuit_weights(
    cached: &CachedCircuit,
    network_outputs: &[(usize, Vec<f64>, PyObject)],  // (input_idx, probs, tensor)
) -> Vec<(f64, f64)> {
    let num_vars = cached.program.num_vars();
    let mut weights: Vec<(f64, f64)> = vec![(0.0, 0.0); num_vars + 1];

    for slot in &cached.weight_slots {
        if let Some((_, probs, _)) = network_outputs.iter()
            .find(|(idx, _, _)| *idx == slot.input_idx)
        {
            let p = probs[slot.label_idx];
            // Normalize and convert to log space
            let sum: f64 = probs.iter().sum();
            let normalized_p = (p * 0.9999999 / sum).max(1e-10);
            weights[slot.var_idx as usize] = (normalized_p.ln(), (1.0 - normalized_p).ln());
        }
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
