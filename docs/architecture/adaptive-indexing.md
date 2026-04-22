# Adaptive Indexing Architecture

> **Implementation status (v0.5.0):** Statistics-gathering infrastructure is implemented; **index-building decisions are design-only** (the snippet below is illustrative, not running code). Concretely: `StatsManager` records per-relation heat and join selectivities at execution time, and `OptimizerConfig.index_heat_threshold` exists as a knob, but the runtime does not yet construct on-the-fly hash indexes based on those observations. Use this document to understand the planned architecture; the heat-based index builder is future work.

## Overview

Heat-based index selection (HISA) tracks query patterns and is intended to build indexes for frequently accessed relations.

## Components

### 1. StatsManager — implemented
- Tracks per-relation heat and cardinality/bytes
- Caches join selectivities and observed join keys
- Location: `crates/xlog-stats`

### 2. JoinStrategy — design
- Selects optimal join algorithm
- Options: Hash, NestedLoop, SortMerge, IndexNestedLoop
- Current runtime dispatches Hash joins only; other strategies are future work

### 3. Executor integration — implemented (statistics wiring only)

`Executor` records:
- Scan heat and cardinality/bytes during `Scan`
- Join selectivity observations for base/base joins (both sides are `Scan`)

See: `crates/xlog-runtime/src/executor.rs`

## Index-Building Decisions (design)

The intended trigger for on-the-fly index construction:

1. Relation heat exceeds `OptimizerConfig.index_heat_threshold`
2. Memory budget allows it
3. Relation is stable (not being modified)

The illustrative shape of such a decision (not currently called by the runtime):

```rust
// DESIGN ONLY — not wired into the executor as of v0.5.0
fn maybe_build_index(&mut self, relation: &str) {
    let heat = self.stats.heat(relation);
    if heat > INDEX_HEAT_THRESHOLD && self.memory.remaining_bytes() > INDEX_MIN_MEMORY {
        self.build_hash_index(relation);
    }
}
```

## Future Work

1. Wire an index manager that consumes `StatsManager` observations and constructs hash indexes at the thresholds above
2. Implement NestedLoop and SortMerge join strategies beyond the current Hash-only dispatch
3. Integrate statistics into the fixpoint loop (recursive SCCs)
4. Add memory-budget-aware index eviction
