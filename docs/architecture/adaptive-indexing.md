# Adaptive Indexing Architecture

## Overview

Heat-based index selection (HISA) tracks query patterns and builds indexes
for frequently accessed relations.

## Components

### 1. QueryStatistics
- Tracks scan counts, join selectivities
- Calculates "heat" score per relation
- Location: `xlog-runtime/src/statistics.rs`

### 2. JoinStrategy
- Selects optimal join algorithm
- Options: Hash, NestedLoop, SortMerge, IndexNestedLoop
- Location: `xlog-runtime/src/statistics.rs`

### 3. Executor Integration (TODO)

Update `executor.rs` execute_join():

```rust
fn execute_join(&mut self, ...) -> Result<CudaBuffer> {
    // 1. Record statistics
    self.stats.record_scan(&left_rel);
    self.stats.record_scan(&right_rel);

    // 2. Select strategy
    let strategy = JoinStrategy::select(
        left.num_rows(),
        right.num_rows(),
        self.is_sorted(&left, &left_keys),
        &self.stats,
    );

    // 3. Execute with selected strategy
    match strategy {
        JoinStrategy::Hash => self.provider.hash_join_v2(...),
        JoinStrategy::NestedLoop => self.provider.nested_loop_join(...),
        JoinStrategy::SortMerge => self.provider.sort_merge_join(...),
        JoinStrategy::IndexNestedLoop => self.provider.index_join(...),
    }

    // 4. Record selectivity
    let selectivity = result.num_rows() as f64 /
        (left.num_rows() * right.num_rows()) as f64;
    self.stats.record_join(&left_rel, &right_rel, selectivity);

    Ok(result)
}
```

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
2. Add index manager with hash index support
3. Integrate statistics into fixpoint loop
4. Add index invalidation on relation updates
