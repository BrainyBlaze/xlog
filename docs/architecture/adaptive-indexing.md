# Adaptive Indexing Architecture

> **Implementation status (v0.8.6):** statistics gathering and the persistent
> build-side hash index manager are implemented. `Executor` reuses the existing
> join-index cache across repeated session evaluations, keyed by relation ID,
> relation generation, schema signature, key columns, and CUDA device ordinal.

## Overview

Heat-based index selection (HISA) tracks query patterns and builds reusable
hash indexes for frequently accessed build-side relations when the memory
budget allows it.

## Components

### 1. StatsManager (Current)
- Tracks per-relation heat + cardinality/bytes
- Caches join selectivities + observed join keys
- Location: `crates/xlog-stats`

### 2. JoinStrategy
- Selects optimal join algorithm
- Options: Hash, NestedLoop, SortMerge, IndexNestedLoop
- Current runtime dispatch remains Hash-join oriented; the broader strategy
  space is future work.

### 3. Executor Integration

`Executor` records:
- Scan heat + cardinality/bytes during `Scan`
- Join selectivity observations for base/base joins (both sides are `Scan`)
- Persistent hash-index cache hits, misses, builds, evictions, invalidations,
  stale rejections, and background-build mode requests/completions

See: `crates/xlog-runtime/src/executor.rs`

## Persistent Hash Index Manager

v0.8.6 extends the existing `JoinIndexCache`; it does not introduce a separate
index lifetime or fixture-only cache. Runtime controls:

- `RuntimeConfig::with_persistent_hash_indexes(Some(true))` enables reuse.
- `RuntimeConfig::with_persistent_hash_indexes(Some(false))` disables reuse.
- `XLOG_PERSISTENT_HASH_INDEXES=0|false|off|no` disables reuse when the config
  field is `None`; unset defaults to enabled to preserve adaptive-indexing
  behavior.
- `RuntimeConfig::with_persistent_hash_index_background_build(Some(true))`
  records background-build request/completion telemetry while staying on the
  existing provider build/reuse path.

Cache keys include:

- relation ID and `RelationStore` generation;
- key column list;
- schema signature derived from column types and row width;
- CUDA device ordinal.

The manager invalidates relation entries on `put_relation`, relation removal,
Monte Carlo/ILP resets, and stale provider mismatch diagnostics. Retained
indexes are bounded by `JoinIndexCache::max_bytes`; insertion evicts
least-recently used entries until the new index fits.

## Index Building Decisions

When to build an index:
1. Relation heat exceeds threshold
2. Memory budget allows
3. Relation is stable (not being modified)

```rust
fn maybe_build_index(&mut self, relation: RelId, right: &CudaBuffer, keys: &[usize]) {
    let heat = self.stats.get_relation_stats(relation).map(|s| s.heat);
    let bytes = estimate_join_index_bytes(right, keys);
    if self.join_index_cache.should_build(bytes, heat.unwrap_or(0.0), remaining, budget) {
        let index = self.provider.build_join_index_v2(right, keys)?;
        self.join_index_cache.insert(key, index);
    }
}
```

## Future Work

1. Implement NestedLoop and SortMerge joins
2. Move background-build mode from telemetry to fully asynchronous recorded
   builds once provider stream dependencies are promoted for join-index build
   kernels.
3. Integrate statistics into fixpoint loop (recursive SCCs)
