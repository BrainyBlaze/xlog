# W2.3 Plan — Recursive-SCC Stats Integration

**Closes W2.3.** No W2.5 default flip, no W2.6 heat/selectivity
feedback, no W4.1 multi-recursive expansion. **Direction (b)**:
per-iteration cardinality update fires from the recursive
fixpoint loop **regardless of whether the iteration used WCOJ
or binary fallback**. W2.4 owns WCOJ-specific selectivity
recording and stays untouched.

**Date:** 2026-05-04
**Branch (proposed):** `feat/w23-recursive-scc-stats-integration`
**Worktree (proposed):** `.worktrees/w23-recursive-scc-stats-integration`
**Base:** `main` at `da644e3d` (W2.1 closure-board commit).
**Board entry:** `docs/v065-closure-board.md` Wave 2, W2.3.

## Goal

Make every iteration of `Executor::execute_recursive_scc` push
the iteration's actual delta cardinality + the iteration's
full-rel cardinality into `StatsManager`. The cost-model lookups
on **subsequent iterations** then see current-iteration cards
instead of seed-only stats from compile time. The closure board's
acceptance line — "each iteration's `binary_est` reflects the
iteration's actual delta, not the seed" — drops out of this
because the cost model is the only consumer that reads the
manager.

## In Scope

* **Per-iteration cardinality update** in
  `xlog-runtime/src/executor/recursive.rs::execute_recursive_scc`.
  Call `StatsManager::update_cardinality(rel_id, rows)` for
  both the **delta RelId** (with the actual delta_initial /
  delta_new row count) and the **full RelId** (with the new
  full row count) at the correct points in the seed pass and
  fixpoint loop, matching the actual code's update order.
* **Both shapes**: recursive triangle + recursive 4-cycle.
  Slice 4 already routes both through
  `execute_wcoj_or_fallback_node` when `recursive_scan_count <= 1`.
* **Default-config bit-identical**: row sets and dispatch
  counters identical to pre-W2.3 on the same fixtures. Stats
  updates are observable only via the cost model (W2.1's
  `CompilerConfig::default()` is `Disabled`, so the W2.1 path
  isn't perturbed; the slice 5 cardinality cost model DOES
  consult `StatsManager` and ITS decisions on recursive bodies
  WILL evolve across iterations — this is the **intended W2.3
  semantic**, not a regression).

## Out of Scope (owned elsewhere)

* **W2.5** default flip of `RuntimeConfig::wcoj_cost_model` —
  out of scope. W2.5's blocker set narrows to `{W3.2, W4.1, W5.1, W5.2}`
  after W2.3 lands.
* **W2.6** heat/selectivity feedback into variable ordering —
  out of scope. W2.3 only updates **cardinality**.
* **W4.1** multi-recursive bodies (`recursive_scan_count > 1`)
  — remain gated out of WCOJ promotion at `promote.rs:107-108`.
  Per-iteration update fires on those bodies too, but their
  cost-model decisions stay binary-join-only. Part D pins the
  promoter gate.
* **W2.4 WCOJ selectivity recording** — already shipped at
  `f586ce34`; left untouched.

## Step 1 — Audit (read-only, before any code change)

The plan is gated on these audit items being true. If any
audit item invalidates a premise, the plan is amended **before**
implementation begins.

* **A1. Delta RelIds are stats-registered.**
  `recursive.rs:279` → `self.register_relation(rel_id, &name)`
  → `mod.rs:332-336` does `self.stats.register_relation(rel_id)`.
  ✓ confirmed.

* **A2. Full-rel RelIds are stats-registered at the executor
  boundary.** When the user / pipeline calls
  `Executor::register_relation(rel_id, name)` for an IDB head
  (e.g., `reach`), `mod.rs:332-336` registers it in
  `self.stats`. The compile-time `mgr` in `compile.rs:175` is a
  **separate** `StatsManager` instance attached to the
  `ExecutionPlan`'s `optimizer` — NOT the runtime manager. W2.3
  reads/writes `executor.stats`, not the compile-time manager.

* **A3. `rel_id_of(pred)` resolution.** Inside
  `execute_recursive_scc`, the IDB head's RelId is reachable
  via `executor.name_to_rel.get(pred)` (the inverse of
  `rel_names`). Plan uses an explicit
  `Executor::name_to_rel_id(name) -> Option<RelId>` helper
  (or threads the lookup directly). If a recursive predicate
  has no executor-registered RelId at this point, the plan
  **skips the per-iteration update for that predicate** (no
  panic). Production callers always register IDB rel_ids
  before `execute_plan`, but defensive skip preserves the
  bit-identical regression invariant when stats can't land.

* **A4. `cached_row_count` preservation through `union_gpu` /
  `diff_gpu` / `dedup`.** The plan must verify these primitives
  set `cached_row_count` on their outputs. If any path leaves
  `cached_row_count = None`, W2.3's helper falls back to
  `Executor::buffer_row_count(buf)` (already in use at
  `recursive.rs:355-356, 409, 503, 527, 551` — the existing
  primitive that this code uses everywhere row-counts are
  needed). **No new D2H-on-data-plane**: `buffer_row_count` is
  the existing path the recursive loop already calls; W2.3
  reuses it.

The audit results are recorded in the implementation evidence
README (step 7) so reviewers can re-verify.

## Step 2 — `name_to_rel_id` accessor (no other helper)

Add a single private `Executor` method:

```rust
impl Executor {
    /// Reverse-lookup a RelId by predicate name. Used by W2.3
    /// to resolve recursive IDB head RelIds for stats updates.
    fn name_to_rel_id(&self, name: &str) -> Option<RelId> {
        self.name_to_rel.get(name).copied()
    }
}
```

**No `record_recursive_iteration_stats` helper is added.** The
two W2.3 update sites (Phase 2 — delta only, no full row count
in scope; Phase 4 — full only, no delta row count in scope)
have **distinct row-count sources**, so a helper that takes
pre-computed `(full_rows, delta_rows)` would force callers to
fabricate one of them. The plan inlines two single-line
`stats.update_cardinality(...)` writes at the distinct phase
sites.

`update_cardinality` is a single `BTreeMap::get_mut` + field
write; cost is negligible vs. the iteration's GPU work.

## Step 3 — Wire seed pass

The seed pass produces, per recursive predicate `pred`:
* `full_new` (the union of all rule outputs whose head is
  `pred`, deduped) — `recursive.rs:340-351`.
* `full_old_rows`, `full_new_rows` — `recursive.rs:355-356`.
* `delta_initial` — incremental delta computed as
  `full_new − full_old` when both are non-empty; `clone(full_new)`
  when `full_old_rows == 0`; empty when `full_new_rows == 0`
  — `recursive.rs:357-372`.
* `store_put(pred, full_new)` then `store_put(delta_name, delta_initial)`
  — `recursive.rs:374-375`.

W2.3 update insertion point: **after** line 375
(`store_put(delta_name, delta_initial)`), per predicate:

```rust
// W2.3 step 3 — seed-iteration stats refresh.
if let Some(full_rel) = self.name_to_rel_id(pred) {
    let delta_rel = delta_tracker.delta_rel_id(pred)?;
    let full_rows = full_new_rows;
    let delta_rows = self.buffer_row_count(
        self.store.get(&delta_name).expect("delta just stored"),
    )?;
    self.stats.update_cardinality(full_rel, full_rows);
    self.stats.update_cardinality(delta_rel, delta_rows);
}
```

`delta_rows` reads back the row count of the **actual
`delta_initial` buffer** (NOT `full_rows`). On the seed
iteration, `delta_initial == full_new` only when `full_old_rows
== 0`; in general it is `full_new - full_old` and the row counts
differ.

## Step 4 — Wire fixpoint loop

The actual code's update order (`recursive.rs:382-580`):

```text
for iteration in 0..max_iterations:
  // Phase 1 — compute per-rule delta_raw, group by head.
  delta_new_raw_by_head: HashMap<String, CudaBuffer> = ...

  // Phase 2 — finalize delta_new per pred:
  //   delta_new = dedup(delta_raw - full).
  // Code at recursive.rs:495-531.
  delta_tracker.begin_iteration()
  for pred in &recursive_preds:
    full = store.get(pred)
    delta_new = ... (raw - full, deduped)
    store_put(delta_name, delta_new)        // recursive.rs:530
    if delta_new.row_count() != 0:
      delta_tracker.mark_changed()

  // Phase 3 — convergence check.
  if delta_tracker.is_converged():           // recursive.rs:534
    reached_fixpoint = true
    break  // FULL is unchanged this iteration; deltas are zero.

  // Phase 4 — merge deltas into full relations.
  // Code at recursive.rs:540-580.
  for pred in &recursive_preds:
    full_old = store.remove(pred)
    delta = store_remove(delta_name)
    if delta.row_count() == 0:
      store_put(pred, full_old)
      store_put(delta_name, delta)
      continue
    merged = union_gpu(full_old, delta)
    ... dedup ...
    store_put(pred, deduped)
    store_put(delta_name, delta)
```

W2.3 inserts updates at **two phase boundaries** per iteration:

* **After Phase 2 (post-`store_put(delta_name, delta_new)`,
  inside the per-pred loop at recursive.rs:530)** — record
  `delta_rel`'s new card. Full-rel card is **not yet updated**
  this iteration (full hasn't changed at this point); leave
  the existing full-rel stat as-is (still the previous
  iteration's full card, which is correct).

* **After Phase 4 (post-merge `store_put(pred, deduped)` at
  recursive.rs:578-ish)** — record the new `full_rel` card.
  No need to touch delta here; Phase 2's delta record stands.

* **On convergence (`delta_tracker.is_converged()` true at
  recursive.rs:534)**: BEFORE breaking, record the converged
  iteration's stats per pred:
  - `delta_rel` card = 0 (delta_new is empty by definition of
    convergence — already store_put'd in Phase 2 with
    row_count 0; Phase 2's record above already wrote 0).
  - `full_rel` card = current row count of `store.get(pred)`
    (unchanged this iteration — still the previous iteration's
    full, which is the converged result). The previous
    iteration's Phase 4 record already wrote this value, so
    the converged-break path needs **no additional write**.

  Net: convergence break adds **no new W2.3 calls**; Phase 2's
  per-pred zero-delta record completes the iteration's stats
  state.

* **Phase 4 zero-delta short-circuit** (`recursive.rs:551-555`,
  the `if delta.row_count() == 0 { ... continue }` path):
  full is restored unchanged; delta is restored unchanged. No
  W2.3 update needed (nothing changed).

### Pseudo-pseudocode of the W2.3-decorated fixpoint:

```text
for iteration in 0..max_iterations:
  // Phase 1 unchanged.
  // Phase 2:
  for pred:
    delta_new = ...
    store_put(delta_name, delta_new)
    // W2.3: record delta_rel's new card.
    let delta_rows = self.buffer_row_count(&store.get(delta_name))?
    self.stats.update_cardinality(delta_rel_id(pred), delta_rows)

  // Phase 3 convergence check unchanged.
  if delta_tracker.is_converged(): break

  // Phase 4:
  for pred:
    if delta.row_count() == 0:
      store_put(pred, full_old)            // restore as-is
      store_put(delta_name, delta)
      continue
    merged = union_gpu(full_old, delta)
    ... dedup ...
    store_put(pred, deduped)
    // W2.3: record full_rel's new card.
    let full_rows = self.buffer_row_count(&store.get(pred))?
    if let Some(full_rel) = self.name_to_rel_id(pred):
      self.stats.update_cardinality(full_rel, full_rows)
    store_put(delta_name, delta)
```

Both update points are direct
`self.stats.update_cardinality(...)` writes at the distinct
phase sites. Phase 2 supplies `delta_rel` + buffer-fresh
`delta_rows`; Phase 4 supplies `full_rel` + buffer-fresh
`full_rows`. No helper is added (see Step 2).

## Caller Audit

`name_to_rel_id` (Step 2) is the only new private accessor
W2.3 introduces on `Executor`. No external callers; W2.3 steps
3, 5, 6 are the only call sites.

## Acceptance Gate

### Test-only observability seam (1 trace mechanism, used by Parts A + B)

Mid-iteration observability is **not possible** from outside
`execute_recursive_scc` today — it runs to fixpoint and then
unregisters delta relations + their stats at
`recursive.rs:592-597`. W2.3 adds a recursive-stats trace
recorder gated on the `recursive-stats-trace` Cargo feature
(default OFF — production builds carry zero overhead) so
Part A + Part B can snapshot per-iteration state without
changing production behavior. The Cargo feature replaces the
`#[cfg(test)]` gating originally drafted because integration
tests in `tests/` cannot reference lib `cfg(test)` items
across crates; the feature plus a `[[test]]`
`required-features` declaration achieves the same
production-zero-overhead contract while making the seam
visible to the W2.3 acceptance gate.

All trace types, the `Executor` field that holds them, and the
accessor are gated on the **`recursive-stats-trace` Cargo
feature** (default OFF). Production builds do not see these
symbols; the recording calls in `execute_recursive_scc` are
also feature-gated. Production builds carry zero trace
overhead — no struct field, no populating call site, no
compile-time cost.

```rust
#[cfg(feature = "recursive-stats-trace")]
pub struct RecursiveStatsTrace {
    pub entries: Vec<RecursiveStatsTraceEntry>,
}
#[cfg(feature = "recursive-stats-trace")]
pub struct RecursiveStatsTraceEntry {
    pub iteration: usize,             // 0 = seed pass, 1+ = fixpoint iter.
    pub pred: String,
    pub full_rel: RelId,
    pub delta_rel: RelId,
    pub full_rows: u64,
    pub delta_rows: u64,
    pub binary_est_for_variant: Option<u64>,
}

#[cfg(feature = "recursive-stats-trace")]
impl Executor {
    pub fn last_recursive_stats_trace(&self) -> &RecursiveStatsTrace {
        &self.last_recursive_stats_trace
    }
}

// Inside `pub struct Executor { ... }`:
#[cfg(feature = "recursive-stats-trace")]
last_recursive_stats_trace: RecursiveStatsTrace,
```

The W2.3 test target in `crates/xlog-runtime/Cargo.toml`
declares `required-features = ["recursive-stats-trace"]`, so
the test only compiles when the feature is enabled. The
workspace gate command is:

```
cargo test --workspace --release --tests --exclude pyxlog \
  --features xlog-runtime/recursive-stats-trace
```

`execute_recursive_scc` populates this trace **only when the
feature is enabled**, inline after each Phase 2 (delta record)
and Phase 4 (full record) update site (and after the seed
pass). `binary_est_for_variant` is populated by re-running
`stats.estimate_join_cardinality(...)` with the iteration's
delta_rel as the rewritten Scan target — pinning the
closure-board acceptance line ("`binary_est` reflects the
iteration's actual delta") to a directly testable quantity.

### Part A — Iteration-level cardinality evolution (3 tests)

`crates/xlog-runtime/tests/test_w23_recursive_stats.rs`. Each
test snapshots `executor.last_recursive_stats_trace()` and
asserts on the per-iteration entries.

Fixtures use the **slice 4 linear-recursive triangle** and
**slice 4 linear-recursive 4-cycle** templates (their
acceptance certs already live at
`crates/xlog-integration/tests/test_recursive_wcoj_*.rs` and
their slice-4 evidence README; W2.3 reuses these exact rule
shapes + edge inputs to keep the regression aligned). The W2.3
tests do NOT replicate the slice 4 cert; they fork the same
fixtures and add the trace assertions.

**Fixture anchor (Parts A, B, C, D)**: the actual slice-4
linear-recursive triangle and 4-cycle programs in
`crates/xlog-integration/tests/test_wcoj_recursive_dispatch.rs`
(`LINEAR_REC_TRIANGLE` at :586 and `LINEAR_REC_4CYCLE` at :669).
**Both fixtures' recursive predicate is `e1`** (the recursive
input relation that the WCOJ rule scans), NOT the head
`tri` / `cyc`. Slice 4 promotes the head rule
`tri(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z).` (recursive_scan_count
== 1, the single recursive Scan being `e1`); semi-naive
rewrites that single `Scan(e1)` to `Scan(delta_e1)` for the
iteration's variant. Same pattern on 4-cycle:
`cyc(W, X, Y, Z) :- e1(W, X), e2(X, Y), e3(Y, Z), e4(Z, W).`
rewrites `Scan(e1)` to `Scan(delta_e1)`.

`tri` and `cyc` are the heads — their full-rel cards still
evolve and are observable via the trace, but the **W2.3
acceptance line is anchored on the rewritten recursive scan's
predicate**, which is `e1` for both fixtures.

* **`recursive_triangle_e1_full_card_grows_across_iterations`**:
  scope: trace entries with `pred == "e1"` on the slice-4
  `LINEAR_REC_TRIANGLE` fixture. For each iteration N up to
  convergence, the trace's `full_rows` for `e1` is ≥ iteration
  `N-1`'s `full_rows` for `e1`. Strict `>` on at least one
  iteration (the chain `(1,2) → tri(1,2,3) → e1(1,3) → tri(1,3,4)`
  produces ≥ 2 distinct `e1` cardinalities across iterations).
* **`recursive_triangle_e1_delta_evolves_across_iterations`**:
  scope: trace entries with `pred == "e1"`. Assert
  **(a)** at least one pre-convergence iteration's
  `delta_rows` for `e1` is non-zero, AND
  **(b)** the converged iteration's Phase 2 record for `e1`
  has `delta_rows == 0`. (Stronger "all pre-convergence deltas
  non-zero" is NOT asserted; that holds for the slice-4
  single-recursive-predicate fixture but would fail under
  mutual recursion where one predicate's delta can be zero
  while another's still changes — Part A is fixture-scoped to
  `e1` to keep the predicate boundary explicit.)
* **`recursive_4cycle_e1_full_card_grows_across_iterations`**:
  scope: trace entries with `pred == "e1"` on the slice-4
  `LINEAR_REC_4CYCLE` fixture. Same shape as the triangle's
  `full_rel` test, applied to `e1`'s `full_rows`.

### Part B — `binary_est` reflects the iteration's actual delta (2 tests)

`crates/xlog-integration/tests/test_w23_recursive_stats.rs`.

Recursive WCOJ rewrites the recursive Scan to the **delta
RelId** (`recursive.rs:430` —
`Self::rewrite_scan_nth(&rule.body, rel_id, occ, delta_rel_id)`).
The cost model on iteration N+1 evaluates the variant body that
joins `delta_rel(pred)` against the EDB; the relevant
`binary_est` is the join estimate **with `delta_rel_id` on one
side**, not the full IDB rel.

#### Triangle variant (slice-4 `LINEAR_REC_TRIANGLE` fixture)

The slice-4 fixture (anchor at
`crates/xlog-integration/tests/test_wcoj_recursive_dispatch.rs:586`):

```text
e1(X, Y) :- e1_seed(X, Y).
e1(X, Y) :- tri(X, Z, Y).
tri(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z).
```

The promoted rule is the head rule
`tri(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z).` — one
recursive Scan (`Scan(e1)`, count == 1, slice-4 gate
satisfied). Semi-naive rewrites that single `Scan(e1)` to
`Scan(delta_e1)` for the iteration's variant
(`recursive.rs:430` →
`Self::rewrite_scan_nth(&rule.body, e1_rel, 0, delta_e1_rel)`).
The variant body becomes
`tri(X, Y, Z) :- delta_e1(X, Y), e2(Y, Z), e3(X, Z).`. The
canonical promoter inputs are `[e_xy, e_yz, e_xz]` →
`[delta_e1, e2, e3]` after rewrite. The cost model's first
binary-join hop in the variant body joins `delta_e1` with
`e2` on the shared variable `Y` (`delta_e1.col1 = e2.col0`):

```
estimate_join_cardinality(
    delta_e1_rel_id,       // left = slot 0 (X, Y), the rewritten recursive Scan
    e2_rel_id,             // right = slot 1 (Y, Z), the next slot
    &[1],                  // delta_e1.col1 (the Y variable)
    &[0],                  // e2.col0       (the Y variable)
)
```

`binary_est_for_variant` records this value per iteration.
Test
`triangle_binary_est_reflects_delta_e1_card_per_iteration`
asserts that across iterations, this estimate is NOT constant
(it tracks `e1`'s `delta_rows`, which evolves as `tri`'s
recursive feedback grows `e1`) — specifically: there exist two
iterations N, M with
`binary_est_for_variant[N] != binary_est_for_variant[M]`.

#### 4-cycle variant (slice-4 `LINEAR_REC_4CYCLE` fixture)

The slice-4 fixture (anchor at
`crates/xlog-integration/tests/test_wcoj_recursive_dispatch.rs:669`):

```text
e1(W, X) :- e1_seed(W, X).
e1(W, X) :- cyc(Y, W, X, Z).
cyc(W, X, Y, Z) :- e1(W, X), e2(X, Y), e3(Y, Z), e4(Z, W).
```

The promoted rule is the head rule
`cyc(W, X, Y, Z) :- e1(W, X), e2(X, Y), e3(Y, Z), e4(Z, W).` —
one recursive Scan (`Scan(e1)`, count == 1). Semi-naive
rewrites `Scan(e1)` to `Scan(delta_e1)`. The variant body
becomes
`cyc(W, X, Y, Z) :- delta_e1(W, X), e2(X, Y), e3(Y, Z), e4(Z, W).`.
The canonical promoter inputs are
`[e_wx, e_xy, e_yz, e_zw]` → `[delta_e1, e2, e3, e4]` after
rewrite. The cost model's first binary-join hop in the variant
body joins `delta_e1` with `e2` on the shared variable `X`
(`delta_e1.col1 = e2.col0`):

```
estimate_join_cardinality(
    delta_e1_rel_id,       // left = slot 0 (W, X), the rewritten recursive Scan
    e2_rel_id,             // right = slot 1 (X, Y), the next slot
    &[1],                  // delta_e1.col1 (the X variable)
    &[0],                  // e2.col0       (the X variable)
)
```

`binary_est_for_variant` records this value per iteration.
Test `cycle4_binary_est_reflects_delta_e1_card_per_iteration`
asserts non-constancy across iterations under the same
"exist N, M with values differing" predicate as the triangle
case.

In both fixtures the rewritten Scan is at canonical slot 0.
The trace records `binary_est_for_variant` for the (slot 0,
slot 1) adjacency pair; if a future fixture rewrites a non-
slot-0 occurrence, the trace contract still applies — record
the `(left_rel, right_rel, left_keys, right_keys)` 4-tuple
the cost model would use for the variant body's first binary
hop, and assert non-constancy of the resulting estimate.

### Part C — Row-set parity + dispatch counter regression (4 tests)

`crates/xlog-integration/tests/test_w23_recursive_stats.rs`.
On the same fixtures as Parts A + B, run two compiles:
* `CompilerConfig::default()` + force-WCOJ-on (W2.1 stays
  Disabled; W2.4 cardinality cost model evolves stats but
  does not flip the dispatch path).
* `CompilerConfig::default()` + force-WCOJ-off (binary
  reference).

* **`recursive_triangle_row_set_unchanged_under_default_config`**:
  W2.3-on row set equals the binary-join reference row set
  bit-for-bit (sorted, deduped tuple comparison).
* **`recursive_triangle_dispatch_counter_unchanged_under_default_config`**:
  `executor.wcoj_triangle_dispatch_count()` post-W2.3 equals
  the slice 4 baseline (captured from `da644e3d` HEAD).
* **`recursive_4cycle_row_set_unchanged_under_default_config`**.
* **`recursive_4cycle_dispatch_counter_unchanged_under_default_config`**.

### Part D — Multi-recursive bodies untouched (1 test)

`crates/xlog-integration/tests/test_w23_recursive_stats.rs`.

* **`multi_recursive_triangle_per_iteration_update_does_not_promote`**:
  rule with `recursive_scan_count > 1` (e.g.,
  `p(X, Y, Z) :- p(X, Y), p(Y, Z), e(X, Z).`). Per-iteration
  trace fires (Part A pattern: trace contains entries). Promoter
  still skips promotion (W4.1 owns lifting this gate). Body
  executes via binary join. Row set matches binary-join
  reference. WCOJ dispatch counter for this rule = 0.

**Total acceptance cert count**: Part A 3 + Part B 2 + Part C 4
+ Part D 1 = **10 tests**, all independently named.

## Step Plan

1. **Audit** (read-only) — record A1, A2, A3, A4 results in
   the implementation evidence README. If any audit fails,
   amend plan before code changes.
2. **`name_to_rel_id` accessor**: add the single private
   `Executor::name_to_rel_id(name: &str) -> Option<RelId>`
   method (1-line body wrapping `self.name_to_rel.get`). No
   other helper introduced.
3. **Trace seam** (`recursive-stats-trace` feature-gated): add
   `RecursiveStatsTrace` + `RecursiveStatsTraceEntry` types in
   `xlog-runtime` and a `last_recursive_stats_trace()` accessor
   on `Executor`. `execute_recursive_scc` populates it only
   when the feature is enabled.
4. **Wire seed pass**: insert `update_cardinality(delta_rel,
   buffer_row_count(delta_initial))` and `update_cardinality(full_rel,
   full_new_rows)` per predicate after `recursive.rs:375`'s
   `store_put(delta_name, delta_initial)`.
5. **Wire fixpoint Phase 2**: insert
   `update_cardinality(delta_rel, buffer_row_count(delta_new))`
   per predicate after `recursive.rs:530`'s
   `store_put(&delta_name, delta_new)`.
6. **Wire fixpoint Phase 4**: insert
   `update_cardinality(full_rel, buffer_row_count(deduped))`
   per predicate after `recursive.rs:578`-area
   `store_put(pred, deduped)` (skip on the zero-delta
   short-circuit `recursive.rs:551-555`).
7. **Tests** — Part A + B + C + D.
8. **Workspace gate**: full slice 1–5 + W2.4 + W2.2 + W2.1
   regression bit-identical when `CompilerConfig::default()`
   is in effect. Recursive triangle / 4-cycle dispatch counters
   identical to `da644e3d` HEAD baseline.
9. **Evidence README + closure proposal + FF-merge**.

## Risk & Open Questions

* **Q1 — `cached_row_count` preservation through `union_gpu` /
  `diff_gpu` / `dedup`**: addressed by step 1 audit (A4) and the
  `Executor::buffer_row_count` fallback (already in use in this
  loop). No new D2H-on-data-plane risk; W2.3 reuses the
  existing path.
* **Q2 — `name_to_rel_id` returns None**: defensive skip
  (no panic). Production callers always register IDB heads
  before `execute_plan`. Test fixtures that omit
  registration would skip the W2.3 update for that predicate
  — bit-identical to pre-W2.3 behavior.
* **Q3 — Default-config regression**: W2.1's cost model is
  Disabled by default; W2.3's stats updates land but don't
  affect leader selection. The slice-5 cardinality cost model
  DOES consult `executor.stats` and ITS decisions on
  recursive bodies will start evolving across iterations.
  This is the **intended W2.3 semantic** ("cost model sees
  current iteration stats"). Part C row-set + dispatch-counter
  parity is the safety net.
* **Q4 — Multi-recursive bodies**: per-iteration update fires
  on them too (Phase 2 + Phase 4 are predicate-level, not
  promoter-level). Promoter gate stays at
  `recursive_scan_count <= 1` per W4.1's scope. Part D pins
  this.
* **Q5 — Mutual recursion**: `execute_recursive_scc` already
  loops `for pred in &recursive_preds` per phase; W2.3's per-
  pred update naturally fans out to all mutual-recursion
  participants. No special case needed.
* **Q6 — `rel_id_of(pred)` for the recursive head when the
  rule's head is a recursive IDB**: resolved via
  `name_to_rel_id` at use site. `recursive_preds` is the list
  of recursive head names; their RelIds were registered at
  `Executor::register_relation` time (production) or at
  fixture setup (tests). Step 1 A2/A3 audit confirms.
* **Q7 — Stats-unregister at fixpoint exit**
  (`recursive.rs:592-597`): the existing code calls
  `self.stats.unregister_relation(rel_id)` for delta RelIds
  after fixpoint. W2.3's trace seam must capture trace entries
  BEFORE this unregister (otherwise the cost model's lookup of
  delta_rel post-fixpoint would already have been wiped).
  Trace population happens inline at Phase 2 / Phase 4, well
  before the Step-3-cleanup unregister. ✓

## Provenance

* Closure board `docs/v065-closure-board.md` Wave 2, W2.3 +
  this commit's intended OPEN → DONE update.
* ROADMAP.md item #6.
* W2.4 (`f586ce34`) selectivity recording from successful WCOJ
  dispatch — **complementary**: W2.4 records selectivity post-
  WCOJ; W2.3 updates cardinality per recursive iteration
  regardless of operator. Both feed `executor.stats` from
  different observation points.
* W2.1 (`d1b13951..f82f9995..da644e3d`) introduced
  `WcojVariableOrderingModel` reading from `StatsManager`;
  with W2.3 wired, those reads on iteration N+1 see
  iteration N's card.
* Slice 4 (`13722751`, `c769df38`, `dfed0e24`) recursive
  triangle + 4-cycle fixtures — Parts A/B/C/D reuse them as
  named cert templates.
* Code anchors: `crates/xlog-runtime/src/executor/recursive.rs`
  lines 279 (delta rel_id register), 353-372 (seed delta
  computation), 374-375 (seed store_put), 430 (variant scan
  rewrite), 526-531 (Phase 2 store_put delta_new), 533-538
  (convergence break), 540-580 (Phase 4 merge + dedup +
  store_put), 592-597 (unregister stats); `crates/xlog-runtime/src/executor/mod.rs:332-336`
  (register_relation fans out to stats).

## Process Rule Compliance

* **Process rule #1**: this slice does NOT self-mark W2.3
  DONE. End-of-slice commit proposes the OPEN → DONE
  transition; user reviews + explicitly approves; a separate
  follow-up commit applies the board update.
* **Process rule #2**: every commit references W2.3.
* **Process rule #3**: this plan header opens with "Closes
  W2.3."
* **Process rule #5**: no `v0.6.6` references.
* **Process rule #6**: no push, no tag.
