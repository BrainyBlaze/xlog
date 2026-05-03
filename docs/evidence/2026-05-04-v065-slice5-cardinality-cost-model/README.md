# v0.6.5 Slice 5 — Cardinality-Aware Cost Model Evidence

**Date:** 2026-05-04
**Branch:** `feat/v065-cardinality-cost-model`
**Base:** `main` at `c769df38` (slice 4 amendment)
**Plan:** `docs/plans/2026-05-04-v065-slice5-cardinality-cost-model-plan.md`

## Slice Summary

Adds an opt-in `CardinalityAwareCostModel` impl of slice 3's
`WcojCostModel` seam. The model fuses the skew classifier with
cardinality estimates from `xlog_stats::StatsManager`. **Default
behavior is unchanged**: the slice 1–4 default
(`SkewClassifierCostModel`) stays the production wiring. The new
model is opt-in via `RuntimeConfig::with_wcoj_cost_model(...)`
or the `XLOG_WCOJ_COST_MODEL=cardinality` env var, with
config-field-wins precedence.

## Decision Rule (locked)

For both triangle and 4-cycle dispatch:

1. **Missing-stats safety floor** — if any slot relation has
   missing stats or `cardinality == 0`, delegate to
   `SkewClassifierCostModel`. Slice 1–4 behavior preserved
   for recursive deltas, freshly-uploaded relations, or any
   predicate the user hasn't seeded.
2. **Classifier failure is never overridden** — `Ok(None)` /
   `Err(_)` always fall back regardless of cardinality.
3. With populated stats AND a real score:
   * `binary_est >= LARGE_CARDINALITY_BINARY_INTERMEDIATE`
     (1M) → dispatch (AGM-bound clause).
   * `binary_est >= MIN_CARDINALITY_BINARY_INTERMEDIATE`
     (4096) AND `score >= MIN_SKEW_FOR_CARDINALITY` (0.05)
     → dispatch.
   * Else → fall back.

`binary_est` uses
`stats.estimate_join_cardinality(slot_rels[0], slot_rels[1],
&[1], &[0])` — the canonical inner-join intermediate the slice
1 (triangle) and slice 2 (4-cycle) lowered shapes materialize
first under the binary-join fallback.

## Acceptance Gates

| # | Gate | Status |
|---|------|--------|
| 1 | `CardinalityAwareCostModel` compiles + 8+ stub unit tests pass | PASS — 10 stub tests (delegate-on-missing, classifier-failure-falls-back × 2, large-binary clause, skew+size clause, below-MIN clause, 4-cycle counterparts × 2, threshold pinning) |
| 2 | Default `RuntimeConfig` selects `SkewClassifier`; slice 4 cert dispatch counts byte-identical | PASS — `cardinality_default_off_keeps_slice4_dispatch_counts` (counter == 1 matches slice 4) |
| 3 | Opt-in via env var OR config field switches the impl deterministically | PASS — `cardinality_opt_in_via_env_var_matches_config_field` (env-locked) |
| 4 | Large-binary fixture under cardinality model dispatches; small-binary fixture falls back | PASS — `cardinality_opt_in_with_seeded_large_cards_dispatches_via_adaptive` (counter ≥ 1) and `cardinality_opt_in_with_small_cards_falls_back_to_binary` (counter == 0) |
| 5 | Workspace + CUDA cert + WCOJ regression no regression | PASS — see workspace tally |

## Cert Test Results

```
cargo test -p xlog-integration --release --test test_wcoj_cardinality_cost_model
running 5 tests
test cardinality_opt_in_via_env_var_matches_config_field ... ok
test cardinality_default_off_keeps_slice4_dispatch_counts ... ok
test cardinality_opt_in_without_seeded_stats_delegates_to_skew_model ... ok
test cardinality_opt_in_with_seeded_large_cards_dispatches_via_adaptive ... ok
test cardinality_opt_in_with_small_cards_falls_back_to_binary ... ok
test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured
```

```
cargo test -p xlog-runtime --lib --release wcoj_cost_model
running 23 tests
... (10 cardinality + 13 prior, all pass)
test result: ok. 23 passed; 0 failed; 0 ignored; 0 measured
```

```
cargo test -p xlog-core --lib --release cost_model
running 6 tests
... (env precedence: default, env, garbage, whitespace, field>env × 2)
test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured
```

## Workspace Tally

| Crate                | PASS | FAIL | IGN |
|----------------------|------|------|-----|
| `xlog-cuda`          | 507  | 0    | 6   |
| `xlog-runtime`       | 135  | 0    | 2   |
| `xlog-logic`         | 503  | 0    | 5   |
| `xlog-integration`   | 123  | 0    | 0   |
| `xlog-core`          | 33   | 0    | 0   |
| `xlog-cuda-tests` (cert) | 1 (cert pass) | 0 | 0 |
| Other crates (sum)   | 476  | 0    | 4   |
| **Workspace**        | **1777+1 cert** | **0** | **17** |

Slice 1–4 regression bit-identical:

* 6/6 slice 4 recursive-WCOJ cert tests
* 39/39 xlog-runtime lib WCOJ tests
* 22/22 promoter unit tests

## Code-Level Changes

| File | Change |
|------|--------|
| `crates/xlog-core/src/config.rs` | New `CostModelKind` enum; `RuntimeConfig::wcoj_cost_model` field + `with_wcoj_cost_model` builder; `resolved_wcoj_cost_model` precedence helper; 6 env-precedence unit tests |
| `crates/xlog-core/src/lib.rs` | Re-export `CostModelKind` from lib root |
| `crates/xlog-runtime/src/executor/wcoj_cost_model.rs` | New `CardinalityAwareCostModel` impl with delegate-on-missing-stats safety floor; pinned threshold constants (`MIN_/LARGE_CARDINALITY_BINARY_INTERMEDIATE`, `MIN_SKEW_FOR_CARDINALITY`); `build_wcoj_cost_model` factory; 10 stub unit tests |
| `crates/xlog-runtime/src/executor/wcoj_dispatch.rs` | Both adaptive dispatch sites use the factory (one virtual call per dispatch decision); UFCS-style call replaced with method call |
| `crates/xlog-integration/tests/test_wcoj_cardinality_cost_model.rs` | New cert file — 5 tests (default-off, large/small/missing stats, env↔config parity) using runtime-stats seeding pattern |

## Risks / Out-of-Slice

* **Q1 — Thresholds are opt-in experimental.** Pinned by tests
  so slice 5.2 / v0.6.6 must update them explicitly when
  bench evidence is in hand. Default impl unchanged → no
  benchmark blocker.
* **Q2 — Selectivity feedback loop.** The cardinality model
  reads from `estimate_join_cardinality`'s default-selectivity
  path (column-distinct fallback). Wiring WCOJ output back
  into `record_join_result` would tighten the estimate over
  time; deferred to v0.6.6.
* **Q3 — Default flip.** Slice 5.2 / v0.6.6 may flip the
  default to `Cardinality` once benchmarks demonstrate parity
  / improvement on representative workloads. Slice 5 only
  ships the impl + opt-in mechanism.

## Test-Fixture Pattern (informational)

For future cardinality-driven cert authors:

```rust
// 1. Compile + register relations.
let plan = compiler.compile(source).expect("compile");
for (name, rid) in compiler.rel_ids() {
    executor.register_relation(*rid, name);
}

// 2. Upload EDB buffers.
for (name, rows) in inputs {
    executor.put_relation(name, upload_binary_u32(memory, rows));
}

// 3. Seed runtime stats — the cost model reads
//    Executor::stats at dispatch time, NOT compile-time-
//    inferred stats. Skipping this step makes the
//    cardinality model delegate to the skew classifier.
for (name, card) in seeded_cards {
    let rid = compiler.rel_ids().get(name).copied().unwrap();
    executor.stats_mut().register_relation(rid);
    executor.stats_mut().update_cardinality(rid, card);
}

// 4. Run.
executor.execute_plan(&plan).expect("execute_plan");
```

This pattern is documented in the test file's header.
