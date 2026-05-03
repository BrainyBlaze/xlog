# v0.6.5 Slice 5 — Cardinality-Aware Cost Model (Plan)

**Date:** 2026-05-04
**Branch (proposed):** `feat/v065-cardinality-cost-model`
**Worktree (proposed):** `.worktrees/v065-cardinality-cost-model`
**Base:** `main` at `c769df38` (slice 4 amendment)

## Slice Goal

Redeem slice 3's `WcojCostModel` seam by shipping a second impl
that consults `StatsManager` cardinality + selectivity estimates
when the skew classifier alone is insufficient. **Opt-in** via
config / env var; the slice 1–4 default
(`SkewClassifierCostModel`) stays the production wiring. Same
shape as slice 3's "S1 infrastructure-only" pattern: prove the
seam carries weight without changing defaults.

## Scope (S1 — infrastructure-only)

* Add `CardinalityAwareCostModel: WcojCostModel` impl in
  `crates/xlog-runtime/src/executor/wcoj_cost_model.rs`.
* Read `WcojDispatchCtx.{stats, slot_rels, width, launch_stream}`
  + `&dyn SkewScoreSource` to make a cardinality-aware decision.
* Wire selection at the call site:
  - Default: `SkewClassifierCostModel` (slice 1–4 behavior).
  - Opt-in: `XLOG_WCOJ_COST_MODEL=cardinality` OR
    `RuntimeConfig::wcoj_cost_model = Some(CostModelKind::Cardinality)`
    selects the new impl.
* No threshold changes for the legacy default.
* No call-site refactor (slice 3's seam already takes the model
  through trait-object dispatch).

## What Already Works (Pre-Slice-5)

* `WcojCostModel` trait + `WcojDispatchCtx` with
  `stats: &StatsManager` + `slot_rels: &[RelId]` plumbed
  (slice 3).
* `SkewScoreSource` sub-seam with stub-friendly trait surface
  (slice 3 amendment).
* `StatsManager` API:
  - `get_relation_stats(rel_id) -> Option<&RelationStats>` →
    `cardinality: u64`.
  - `estimate_join_cardinality(left, right, l_keys, r_keys) -> u64`
    — used by the binary-join planner; cached + selectivity-
    aware (`JoinSelectivity::estimate_output_rows`).
  - `get_join_selectivity(left, right) -> Option<f64>` —
    observed selectivity from prior joins.

## What's Missing

1. A `WcojCostModel` impl that uses any of the above.
2. A selection mechanism (env var + RuntimeConfig field) so
   integration tests can opt in deterministically.
3. Cert tests pinning the new model's decisions on synthetic
   stats.

## Decision Rule (locked)

For triangle dispatch (`should_dispatch_triangle`):

1. Lookup per-slot cardinalities. If **any** slot relation
   has missing or zero cardinality (recursive deltas / stale
   stats are common cases), the cardinality model **delegates
   to `SkewClassifierCostModel`** — recursive/delta dispatch
   stays slice-1–4-equivalent until stats are populated. This
   is the safety floor.
2. Compute the "binary-join intermediate" estimate as
   `binary_est = stats.estimate_join_cardinality(slot_rels[0],
   slot_rels[1], &[1], &[0])` — the inner `e1 ⋈ e2` join. This
   is the exact intermediate the slice-1 binary-join fallback
   would materialize first.
3. Fetch skew score via `scorer.triangle_skew_score(launch, width)`.
4. **Classifier failure is never overridden.** If the score is
   `Ok(None)` or `Err(_)`, fall back regardless of cardinality.
   This preserves the slice-1 WCOJ safety invariant ("classifier
   declined / errored → no dispatch").

Decision (only when score is `Ok(Some(s))` AND all slot cards
are populated):

* `binary_est >= LARGE_BINARY_INTERMEDIATE` (opt-in default
  `1_000_000`) → dispatch (even on low skew; AGM bound dominates).
* `binary_est >= MIN_BINARY_INTERMEDIATE` AND `s >= MIN_SKEW_FOR_CARDINALITY`
  (opt-in defaults `4096` / `0.05`) → dispatch.
* Otherwise → fall back.

Pseudocode:

```rust
// Step 1: card floor — delegate to skew model on missing stats.
let cards: Option<Vec<u64>> = ctx.slot_rels.iter()
    .map(|r| ctx.stats.get_relation_stats(*r)
        .map(|s| s.cardinality).filter(|c| *c > 0))
    .collect();
let Some(_cards) = cards else {
    return SkewClassifierCostModel::default()
        .should_dispatch_triangle(ctx, scorer);
};

// Step 2: estimate binary intermediate.
let binary_est = ctx.stats.estimate_join_cardinality(
    ctx.slot_rels[0], ctx.slot_rels[1], &[1], &[0]
);

// Step 3 + 4: classifier failure is never overridden.
let score = match scorer.triangle_skew_score(ctx.launch_stream, ctx.width) {
    Ok(Some(s)) => s,
    Ok(None) | Err(_) => return false,
};

// Cardinality + skew decision.
binary_est >= LARGE_BINARY_INTERMEDIATE
    || (binary_est >= MIN_BINARY_INTERMEDIATE && score >= MIN_SKEW_FOR_CARDINALITY)
```

For 4-cycle: identical structure with the inner join's
`(slot_rels[0], slot_rels[1])` pair (per slice 2's lowered
shape) and `cycle4_skew_score`.

**Thresholds are opt-in experimental constants** for the new
model — not production-tuned. The default impl
(`SkewClassifierCostModel`) is unchanged from slice 1–4, so
benchmark evidence isn't a blocker for slice 5 to land.
Threshold tuning + a default flip belong to slice 5.2 /
v0.6.6 once we have workload-driven evidence.

## What Slice 5 S1 Does NOT Do

* **No default change.** `SkewClassifierCostModel` stays the
  default; opt-in is required. Slice 5.2 may flip after
  benchmarks demonstrate parity / improvement.
* **No threshold tuning.** Defaults are best-guess + pinned;
  v0.6.6 pass tunes with real workloads.
* **No new shapes / kernels.** Triangle + 4-cycle only.
* **No selectivity_pass population.** Slice 3's no-op
  `selectivity_pass` stays no-op — that's a separate slice
  (v0.6.6+ candidate).
* **No multi-recursive WCOJ.** Slice 4.2's deferred work.
* **No recursive-arm cardinality plumbing changes.** Slice 4
  already passes `&self.stats` into the dispatch ctx; the
  new model reads it the same way the legacy model ignores it.

## Step Plan

### Step 1 — `CostModelKind` enum + RuntimeConfig field

* Add `pub enum CostModelKind { SkewClassifier, Cardinality }`
  to `xlog-core::config` (or wherever `RuntimeConfig` lives).
* Add field `RuntimeConfig::wcoj_cost_model: Option<CostModelKind>`.
* Default `None`.
* Builder: `with_wcoj_cost_model(kind)`.
* **Pinned precedence**:
  1. `RuntimeConfig::wcoj_cost_model = Some(...)` wins (highest).
  2. Else `XLOG_WCOJ_COST_MODEL` env var:
     `cardinality` → `Cardinality`;
     `skew` or unrecognized → `SkewClassifier`.
  3. Else default `SkewClassifier`.
* Tests use the existing env-lock pattern (the harness already
  serializes env-mutating tests). Cover:
  - env-var unset → SkewClassifier.
  - env-var `cardinality` → Cardinality.
  - env-var `garbage_value` → SkewClassifier (graceful).
  - config field `Some(Cardinality)` overrides env-var `skew`.

### Step 2 — `CardinalityAwareCostModel` impl

* Add `pub(super) struct CardinalityAwareCostModel { … }` to
  `wcoj_cost_model.rs`.
* Fields: `min_binary_intermediate`, `large_binary_intermediate`,
  `min_skew_for_cardinality` — all pinned constants in
  `Default`. (No `triangle_threshold`/`cycle4_threshold` field
  — the cardinality model uses `min_skew_for_cardinality`
  uniformly across both shapes; the slice 1–4 0.10 thresholds
  remain inside `SkewClassifierCostModel` and the
  delegate-on-missing-stats path.)
* Implement `WcojCostModel`:
  - `should_dispatch_triangle` — see decision rule above:
    missing stats → delegate to `SkewClassifierCostModel`;
    classifier failure → fall back; otherwise apply the
    binary_est / skew rule.
  - `should_dispatch_4cycle` — analogous; reuses the same
    delegate-on-missing-stats safety floor.
* Stub-based unit tests (mirror slice 3 amendment pattern):
  - Synthetic `StubScorer` returns a configured score.
  - Synthetic `StatsManager` is built via `register_relation` +
    `update_cardinality` to pin specific estimates.
  - Cover all six branches in the decision rule:
    1. Missing stats → delegate (asserts skew model's decision
       reaches the call).
    2. Zero cardinality on one slot → delegate.
    3. Score `Err(_)` → fall back even when binary_est huge.
    4. Score `Ok(None)` → fall back even when binary_est huge.
    5. Stats populated, binary_est ≥ LARGE, score below skew
       threshold → dispatch (asymptotic clause).
    6. Stats populated, binary_est ≥ MIN, score ≥ skew thr →
       dispatch (skew + size clause).
    7. Stats populated, binary_est below MIN → fall back.
    8. 4-cycle counterpart of the dispatch case.
  - Pinned-threshold test (`thresholds_pinned_at_…`) catches
    accidental drift during slice 5.2 tuning.

### Step 3 — Dispatch-site selection

* In `wcoj_dispatch.rs`, where the dispatch site currently
  builds `let model = SkewClassifierCostModel::default();`,
  switch to a small factory:
  ```rust
  let model: Box<dyn WcojCostModel> =
      match resolved_cost_model_kind(&self.config) {
          CostModelKind::SkewClassifier =>
              Box::new(SkewClassifierCostModel::default()),
          CostModelKind::Cardinality =>
              Box::new(CardinalityAwareCostModel::default()),
      };
  ```
* Add `resolved_cost_model_kind(config) -> CostModelKind` that
  consults `config.wcoj_cost_model`, then env var, then
  defaults to `SkewClassifier`.
* Both dispatch sites (triangle, 4-cycle) consume the same
  factory — keep them symmetric.

### Step 4 — Cert tests

In `crates/xlog-integration/tests/test_wcoj_cardinality_cost_model.rs`:

* `cardinality_default_off_keeps_slice4_dispatch_counts` —
  Same fixture as slice 4 stable triangle. Default
  `RuntimeConfig` (no env var, no field set) — counter == 1
  bit-identical with slice 4.
* `cardinality_opt_in_with_seeded_stats_dispatches_on_large_binary` —
  Triangle program; **after `executor.put_relation(...)`, also
  call `executor.stats_mut().update_cardinality(rel, n)`** to
  seed populated cardinalities (the cost model reads runtime
  stats, not compile-time-inferred). Stats populated to make
  binary_est >> LARGE_BINARY_INTERMEDIATE; opt-in via config
  field. Assert counter ≥ 1.
* `cardinality_opt_in_falls_back_when_binary_est_is_small` —
  Same flow but cardinalities seeded small enough that
  binary_est < MIN_BINARY_INTERMEDIATE. Opt-in. Assert counter
  == 0.
* `cardinality_opt_in_delegates_when_stats_missing` — Opt-in,
  but DON'T seed runtime stats. Assert behavior matches
  `SkewClassifierCostModel` on the same fixture (delegate
  path).
* `cardinality_opt_in_via_env_var_matches_config_field` —
  Use the env-lock pattern. Confirm env var and config field
  produce the same selection.

The "seed runtime stats" pattern is documented in the test-
file header so future cardinality-driven tests use it
consistently.

### Step 5 — Workspace gate

* `cargo fmt --all -- --check`
* `cargo test --workspace --release --exclude pyxlog`
* CUDA cert: `cargo test -p xlog-cuda-tests --test certification_suite --release`
* WCOJ regression: 6 slice 4 cert + 39 lib WCOJ + 22 promoter
  unit + new step 2/4 tests.
* Evidence file at
  `docs/evidence/2026-05-04-v065-slice5-cardinality-cost-model/README.md`
  with: per-step test names, threshold rationale, decision
  rule pseudocode, and a fresh dispatch-counter ladder.

### Step 6 — FF-merge to local main

* No push, no tag.
* Same FF-merge pattern as slice 4.

## Acceptance Gates

| # | Gate | Owner |
|---|------|-------|
| 1 | `CardinalityAwareCostModel` compiles + 8+ stub unit tests pass | step 2 |
| 2 | Default `RuntimeConfig` selects `SkewClassifier` — slice 4 cert dispatch counts byte-identical | step 4 |
| 3 | Opt-in via `XLOG_WCOJ_COST_MODEL=cardinality` OR config field switches the impl deterministically | step 4 |
| 4 | Large-binary fixture under cardinality model dispatches; small-binary fixture falls back | step 4 |
| 5 | Workspace + CUDA cert + WCOJ regression no regression | step 5 |

## Risk & Open Questions

* **Q1 — Statistics may be missing at first dispatch.**
  Recursive deltas, freshly-uploaded relations, and any
  predicate the user hasn't called `update_cardinality` on
  expose `Option::None` from `get_relation_stats` (or
  `cardinality == 0`). The model **delegates to
  `SkewClassifierCostModel`** in that case — slice 1–4
  behavior preserved. Per-slot card lookup happens before
  any cardinality math, so we never use bogus
  `unwrap_or(1000)` defaults inside the cardinality
  decision.
* **Q2 — Selectivity cache is populated by `record_join_result`,
  which the WCOJ path doesn't currently call.** That's fine
  for slice 5: the cardinality model uses the column-distinct
  fallback path inside `estimate_join_cardinality`. Wiring
  WCOJ output back into `record_join_result` is a separate
  slice (probably v0.6.6) — out of scope here.
* **Q3 — Threshold values are opt-in experimental, not
  production-tuned.** `MIN_BINARY_INTERMEDIATE = 4096` (kernel
  launch overhead is ~tens of microseconds; a join smaller
  than this isn't worth dispatching even if skew is high —
  the floor is intentionally generous and not row-byte-tied
  because per-row width varies across triangle/4-cycle and
  u32/u64). `LARGE_BINARY_INTERMEDIATE = 1_000_000` (clearly
  past the AGM-bound crossover for a 1M-row triangle
  intermediate vs. ~30K WCOJ output, again
  size-not-byte-tied). `MIN_SKEW_FOR_CARDINALITY = 0.05`
  (half the slice 2 0.10 threshold; the cardinality clause
  already requires non-trivial sizes). Pinned by unit tests
  so any future tuning is an explicit change. **Default
  impl is unchanged**, so no benchmark evidence is required
  to land slice 5; the threshold debate happens in slice
  5.2 / v0.6.6 when a default flip is on the table.

## Out-of-Slice (Deferred)

* Default flip to cardinality model — slice 5.2 / v0.6.6 after
  benchmarks.
* Threshold tuning with bench evidence — slice 5.2 / v0.6.6.
* `selectivity_pass` population — separate slice.
* Recursive-arm per-iteration stats updates — separate slice.
* Cardinality feedback loop (`record_join_result` from WCOJ
  output) — v0.6.6.
* General-arity / 4-clique kernels — v0.6.6+.
