# Adaptive Indexing Architecture

## Overview

Heat-based index selection (HISA) tracks query patterns and builds indexes
for frequently accessed relations.

## Components

### 1. StatsManager (Current)
- Tracks per-relation heat + cardinality/bytes
- Caches join selectivities + observed join keys
- Location: `crates/xlog-stats`

### 2. JoinStrategy
- Selects optimal join algorithm
- Options: Hash, NestedLoop, SortMerge, IndexNestedLoop
- Location: `xlog-runtime/src/statistics.rs`

### 3. Executor Integration (Implemented: statistics wiring)

`Executor` records:
- Scan heat + cardinality/bytes during `Scan`
- Join selectivity observations for base/base joins (both sides are `Scan`)

See: `crates/xlog-runtime/src/executor.rs`

## Index Building Decisions

When to build an index:
1. Relation heat exceeds threshold
2. Memory budget allows
3. Relation is stable (not being modified)

```rust
fn maybe_build_index(&mut self, relation: &str) {
    let heat = self.stats.heat(relation);
    if heat > INDEX_HEAT_THRESHOLD && self.memory.remaining_bytes() > INDEX_MIN_MEMORY {
        self.build_hash_index(relation);
    }
}
```

## Future Work

1. Implement NestedLoop and SortMerge joins
2. Add index manager with hash index support + invalidation
3. Integrate statistics into fixpoint loop (recursive SCCs)
4. Add memory-budget-aware index eviction
