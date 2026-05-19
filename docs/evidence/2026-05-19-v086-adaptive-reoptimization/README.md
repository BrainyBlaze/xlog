# v0.8.6 G086_ADAPT Evidence

Node: `G086_ADAPT` - Adaptive Runtime Re-Optimization.

This evidence records the first production adaptive re-optimization slice:
the compiler supplies the candidate plan, and `Executor` owns deterministic
telemetry review, candidate adoption, GPU output equivalence, and rollback.
The path reuses the normal `Executor::execute_plan` loop for both baseline and
candidate plans.

## Metrics

- `M086_ADAPT.1 telemetry`: baseline joins record estimated rows, actual rows,
  cardinality deltas, selectivity deltas, relation heat, and a deterministic
  mis-plan ratio before `StatsManager::record_join_result` updates the model.
- `M086_ADAPT.2 decision stability`: the same captured fixture telemetry
  replays to an identical decision across 100 deterministic replays.
- `M086_ADAPT.3 correctness`: the accepted candidate is compared against the
  baseline output with GPU full-row set difference in both directions.
- `M086_ADAPT.4 rollback`: an adverse candidate restores the baseline
  relation/statistics snapshot and records a typed
  `CandidateOutputMismatch` diagnostic.
- `M086_ADAPT.5 performance/blocker`: rollback removes the correctness blocker
  for adverse adaptation; a candidate cannot replace the baseline unless GPU
  equivalence succeeds.
- `M086_ADAPT.6 transfer budget`: adoption and rollback checks use
  metadata/control-plane counters and GPU set operations only; no tracked
  data-plane DTOH calls are added.

## Fresh Checks

- `cargo test -p xlog-runtime adaptive_reoptimization -- --nocapture`
  - 4 adaptive tests passed.

Machine-readable evidence: `measurements.json`.
