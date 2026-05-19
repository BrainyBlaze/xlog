# W2.6 Plan — Heat + Selectivity Feedback into Variable Ordering

**Closes W2.6.** No W2.5 default flip, no W3.2 kernel expansion,
no W4.1 multi-recursive expansion. New cost model is **opt-in**;
W2.1's existing `LeaderCardinalityModel` (default) is preserved
bit-identically.

**Date:** 2026-05-04
**Branch (proposed):** `feat/w26-heat-selectivity-variable-ordering`
**Worktree (proposed):** `.worktrees/w26-heat-selectivity-variable-ordering`
**Base:** `main` at `cf57f3a1` (W2.3 closure-board commit).
**Board entry:** `docs/v065-closure-board.md` Wave 2, W2.6.

## Goal

Close the loop between **observed runtime stats** (heat from
`record_access`, selectivity from `record_join_result`) and the
**compile-time WCOJ variable-ordering decision** from W2.1.

Concretely: introduce `HeatAwareLeaderModel` as a second
implementation of W2.1's `WcojVariableOrderingModel` trait. The
model picks the leader (slot 0 = iteration key) by combining
cardinality + selectivity + heat into a composite score:

* **Hot relation** (high `RelationStats.heat`) → demoted from
  leader; pushed to a lookup/inner slot.
* **Cold extensional** (low cardinality + low heat) → preferred
  as leader = iteration key.
* **Highly selective / tight edge** (low `JoinSelectivity.selectivity`
  value at the joined slot) → the relation on that edge is treated
  as a tight filter and pulled toward the inner slot. Throughout
  this plan, "tight" and "low selectivity" mean the same thing:
  small `selectivity ∈ [0.0, 1.0]` ⇒ few output rows per probe ⇒
  strong filter.

Default `CompilerConfig::default()` keeps `WcojVarOrderingKind::Disabled`
(W2.1 contract). Activation requires explicit
`WcojVarOrderingKind::HeatAware`.

## In Scope

* **New trait impl** in `xlog-logic::wcoj_var_ordering`:
  ```rust
  pub struct HeatAwareLeaderModel;
  impl WcojVariableOrderingModel for HeatAwareLeaderModel { ... }
  ```
  Reuses W2.1's locked permutation tables (`triangle_lookup_perms`,
  `cycle4_lookup_perms`, `triangle_kernel_output_cols`,
  `cycle4_kernel_output_cols`) and W2.1's `effective_wcoj_var_ordering_threshold()`
  resolver.
* **New enum variant** `WcojVarOrderingKind::HeatAware` on
  `CompilerConfig`. `LeaderCardinalityModel` and `Disabled` stay.
* **Promoter wiring**: `try_promote_triangle` /
  `try_promote_4cycle` already dispatch to `LeaderCardinalityModel`
  unconditionally (W2.1 step 5). W2.6 adds branching on
  `config.wcoj_variable_ordering`:
  * `Disabled` → no leader pick (W2.1 contract).
  * `LeaderCardinality` → `LeaderCardinalityModel` (W2.1).
  * `HeatAware` → `HeatAwareLeaderModel` (W2.6).
* **Composite score** definition (locked, including constants):
  ```
  score(rel) = cardinality(rel)
             * (1.0 + 4.0 * heat(rel))
             * sel_penalty(rel)

  sel_penalty(rel) = Σ_{e ∈ edges(rel)} 1.0 / max(0.01, observed_sel_or_one(e))
  ```
  Higher score = more expensive to iterate over → demote from
  leader. The model picks `argmin(score)` as leader. The
  threshold gate from W2.1 stays:
  `min_score / default_leader_score ≤ effective_threshold` else
  `var_order = None`.

  **Heat weight = `4.0`** (constant `HEAT_WEIGHT = 4.0`,
  `pub const` on `HeatAwareLeaderModel`). Locked rationale:
  with the W2.1 default threshold of `0.5`, the gate fires
  when `min/default ≤ 0.5`, i.e., the default leader's score
  is at least 2× the candidate leader's. With `heat = 0` on
  the cold side and `heat = h` on the hot side and equal
  cardinalities + uniform selectivities, the heat ratio is
  `(1 + 4h) / 1 = 1 + 4h`. For `1 + 4h ≥ 2.0` → `h ≥ 0.25`,
  achieved after 3 `record_access` calls
  (`heat = 1 - 0.9^3 = 0.271`). Heat weight `1.0` (the
  iteration-1 draft) needed `h ≥ 1.0` which is unreachable
  (asymptote = 1.0); weight `2.0` needed `h ≥ 0.5` (~7
  calls); weight `4.0` is the minimum that keeps test
  fixtures small.

  **Selectivity penalty: sum over the rel's incident edges**,
  NOT just the candidate-leader's adjacent slot. For triangle,
  each rel is in 2 edges (canonical edges
  `{(0,1), (1,2), (0,2)}` so rel 0 ∈ `{(0,1),(0,2)}`, rel 1 ∈
  `{(0,1),(1,2)}`, rel 2 ∈ `{(1,2),(0,2)}`). For 4-cycle,
  each rel is in 2 edges of the cycle
  `{(0,1),(1,2),(2,3),(3,0)}`. The sum aggregation means
  **any rel participating in a tight edge is penalized**, so
  Part B's "leader is the rel NOT in the tight pair" assertion
  follows deterministically from the formula:
  * 1 tight edge with `sel = 0.01`, all others default `sel
    = 1.0`: rels in the tight edge get penalty `100 + 1 =
    101`; the rel NOT in the tight edge gets penalty `1 + 1 =
    2`. argmin = the not-in-tight rel.

  **Inputs to the score**:
  * `cardinality` from `StatsManager::get_relation_stats(rel).cardinality`.
  * `heat` from `RelationStats.heat: f32` (EMA, populated by
    `record_access` at `node_dispatch.rs:26`).
  * `observed_sel_or_one(e)` from
    `StatsManager::get_join_selectivity(rel_a, rel_b)` (W2.4
    output) **with key-column validation**: `StatsManager`
    indexes selectivity by relation pair only (no keys in the
    map key), but stored `JoinSelectivity` records carry
    `left_keys` / `right_keys`. The W2.6 model derives the
    candidate `(left_keys, right_keys)` per edge from the
    canonical kernel topology (triangle: shared-variable
    column index per the locked permutation table; 4-cycle:
    same shared-variable derivation). The model **only
    consumes** a cached `JoinSelectivity` when its
    `left_keys` / `right_keys` match the candidate after
    canonicalization (`StatsManager::canonical_join_key`
    swaps the relation order to a canonical
    (smaller_rel, larger_rel) pair, so the W2.6 model swaps
    keys correspondingly before comparison). On mismatch, the
    model treats the edge as having no observed selectivity
    and uses `1.0` (no filter assumption — penalty
    contribution = 1).
* **Threshold reuse**: `CompilerConfig::wcoj_var_ordering_threshold`
  applies unchanged; the formula's denominator is the
  default-leader score (canonical idx 0).
* **Stats-snapshot feedback loop seam** (the closure-board
  acceptance line "selectivity + heat statistics feed into the
  variable ordering"): documented and tested explicitly in this
  slice. See §"Feedback Loop Seam" below.

## Feedback Loop Seam

The closure-board line says: "selectivity + heat statistics feed
into the variable ordering from W2.1." This requires a path from
**runtime-observed stats** (W2.3 per-iteration card,
W2.4 record_join_result selectivity, `record_access` heat) into
**compile-time cost-model decisions** (`HeatAwareLeaderModel`'s
`pick_*_leader` calls during `promote_multiway`).

**Existing path (no new seam needed):**

1. **Runtime observation phase**:
   * `node_dispatch.rs:26` writes `executor.stats.update_cardinality(...)`
     and `executor.stats.record_access(...)` on every Scan.
   * `Executor::record_wcoj_feedback` (W2.4) writes
     `executor.stats.record_join_result(left_rel, right_rel,
     vec![1], vec![0], input_rows, out_rows)` on successful
     WCOJ dispatch (`wcoj_dispatch.rs:680`).
     `record_join_result` applies an EMA smoothing
     `new = 0.7*old + 0.3*observed` (`manager.rs:345-348`)
     starting from default selectivity = 1.0; tests must
     account for this (≥ 4 dispatches converge enough to
     drive a leader change with `observed ≈ 0.01`, see Part
     C below).
   * `execute_recursive_scc` (W2.3) writes per-iteration deltas
     into `executor.stats`.

2. **Snapshot capture**:
   * The seam is **`Executor::stats_snapshot()`** at
     `crates/xlog-runtime/src/executor/mod.rs:382` — NOT the
     raw `StatsManager::snapshot()` at `manager.rs:94`. The
     executor wrapper adds `rel_names` (the predicate-name ↔
     RelId mapping the executor maintains via
     `register_relation`); the raw `StatsManager::snapshot()`
     leaves `rel_names` empty, which would cause
     `compile.rs:186-260`'s name-keyed remap path to fall
     through to `merge_snapshot` and apply runtime RelIds
     directly — incorrect across recompiles where new
     `Lowerer::rel_ids()` may differ. **The plan's seam
     contract is `executor.stats_snapshot()`**; tests and
     production callers MUST use this path.

3. **Compile-time injection**:
   * `Compiler::compile_with_config_and_stats_snapshot(source,
     config, Some(&snapshot))` (W2.1 step 4 entry point) merges
     the snapshot into the optimizer's `mgr` (compile-time
     `StatsManager` — `compile.rs:186-260`).
   * The `merge_snapshot` path covers `relations` (cardinality
     + heat) AND `join_selectivities`, so all three signals
     reach the compile-time manager.
   * `promote_multiway` (W2.1 step 5) is called with this same
     `&stats_arc`, so `HeatAwareLeaderModel::pick_*_leader`
     reads the merged cardinality + heat + selectivity at
     decision time.

4. **Step 1 audit verifies**: `RelationStats.heat` is included
   in `Executor::stats_snapshot()`'s output AND in
   `compile.rs:186-260`'s `merge_snapshot` /
   name-keyed-remap relation-copy path. If either is missing,
   W2.6 step 6 adds the missing field copy. **Plan amends if
   the audit invalidates this premise.**

The seam is therefore the **W2.4 `record_wcoj_feedback` →
`record_join_result` (EMA-smoothed) + W2.3 per-iteration
cardinality update + `record_access`** writes feeding
`StatsManager`, captured via **`Executor::stats_snapshot()`**
(NOT the raw `StatsManager::snapshot()`), replayed at compile
time via `compile_with_config_and_stats_snapshot`. **No new IPC**,
**no new env var**, **no new Compiler entry point**.

If the audit reveals `heat` is omitted from the snapshot path
or from the merge-snapshot's `RelationStats` field copy, the
minimal seam is a 1-line copy fix in `xlog-stats`
(`StatsSnapshot::merge_snapshot` or
`StatsManager::snapshot`). Already handled by `RelationStats::clone`
if the snapshot path round-trips a full clone — verified at
step 1.

### W2.4 feedback pair must reflect post-W2.1 leader rotation

W2.4's `record_wcoj_feedback` (`wcoj_dispatch.rs:657`) currently
records selectivity on the **canonical-order** slot pair
(`slot_rels[0]`, `slot_rels[1]` = `e_xy`, `e_yz` for triangle;
`e_wx`, `e_xy` for 4-cycle) with hardcoded keys `[1] / [0]`.
After W2.1's variable-ordering rotation via
`prepare_leader_inputs`, the actual kernel slot 0 / slot 1
pair may be a different `(rel, rel)` and a different
`(left_keys, right_keys)`. **W2.6 consuming the W2.4-recorded
selectivity would close the loop over the wrong pair when the
plan's `var_order` is `Some(_)`.**

**W2.6 step 5** rewrites
`record_wcoj_feedback`'s call sites at
`wcoj_dispatch.rs:882` (triangle dispatch success) and
`wcoj_dispatch.rs:1323` (4-cycle dispatch success) to derive
the **post-rotation** slot 0 / slot 1 pair from the matched
body's `var_order`:

* `var_order = None` → record on canonical
  `(slot_rels[0], slot_rels[1])` with keys `[1]/[0]` (the
  current W2.4 behavior; **bit-identical preservation**).
* `var_order = Some(vo)` → record on the rotated slots
  AND with the rotated key indices. Both the relation pair
  AND the underlying-relation key columns come from the
  **locked rotated-feedback table in §"Step 5"** below:
  triangle non-default leaders use `[1]/[1]` because slot 1
  is a swapped 2-col view but the underlying rel's native
  column indexing places the join variable at col1; 4-cycle
  is rotation-only so all slots stay at `[1]/[0]`.

Helper exposed: a small private function
`feedback_pair_from_var_order(canonical_slot_rels: &[RelId],
var_order: Option<&VariableOrder>) -> (RelId, RelId, Vec<usize>,
Vec<usize>)` in `wcoj_dispatch.rs` (full signature locked at
Step 5 below), used by both triangle and 4-cycle success
paths.

**Default config (`Disabled`) preserves the canonical W2.4
behavior bit-identically** — the `var_order = None` branch
takes the existing path. Part D's regression test pins this.

## Step 1 — Audit (read-only)

Plan is gated on these audit items being true:

* **A1. `RelationStats.heat: f32`** exists at `xlog-stats/src/stats.rs:25`.
  ✓ confirmed.
* **A2. `record_access` writes heat** via EMA at
  `xlog-stats/src/stats.rs:78`. `node_dispatch.rs:26` invokes it
  per Scan. ✓ confirmed.
* **A3. `StatsManager::get_join_selectivity(a, b)`** returns
  the W2.4-recorded selectivity. Verified path:
  `Executor::record_wcoj_feedback` (`wcoj_dispatch.rs:680`)
  → `StatsManager::record_join_result` (EMA-smoothed via
  `manager.rs:345-348`) → retrievable. EMA weights
  `0.7*old + 0.3*observed`.
* **A4. Snapshot capture path**: verify
  `Executor::stats_snapshot()` exists at
  `crates/xlog-runtime/src/executor/mod.rs:382` and returns a
  `StatsSnapshot` populated with `rel_names` from the
  executor's `name_to_rel` map. ✓ confirmed at audit (file
  + line ref above). Verify the snapshot carries heat +
  selectivity + cardinality through `RelationStats::clone` /
  `JoinSelectivity::clone`. If `heat` is dropped during the
  W2.2 rel_id-remap path in `compile.rs:186-260`, step 6
  adds the field-copy fix.
* **A5. Snapshot merge path**: verify `compile.rs:186-260`'s
  snapshot merge copies heat + selectivity into the compile-time
  manager's `RelationStats`. If `heat` is dropped during
  remap (the W2.2 path that filters by predicate name), the
  audit identifies the minimal preservation fix.

Audit results recorded in the W2.6 evidence README (step 9). If
any item fails, the plan is amended before code changes.

## Step Plan

1. **Audit** (read-only) — record A1-A5 results. Amend plan on
   any audit failure.
2. **`HeatAwareLeaderModel` impl** (`xlog-logic::wcoj_var_ordering`):
   the new struct + trait impl with the locked composite score
   formula. Reuses W2.1's `triangle_lookup_perms`,
   `cycle4_lookup_perms`, `triangle_kernel_output_cols`,
   `cycle4_kernel_output_cols`, and threshold resolver.
3. **`WcojVarOrderingKind::HeatAware`** variant added to the
   `CompilerConfig` enum (xlog-logic::compiler_config).
4. **Promoter dispatch**: `try_promote_triangle` /
   `try_promote_4cycle` (xlog-logic::promote) branch on
   `config.wcoj_variable_ordering`:
   * `Disabled` → existing W2.1 path (None).
   * `LeaderCardinality` → existing W2.1 path
     (`LeaderCardinalityModel`).
   * `HeatAware` → new W2.6 path (`HeatAwareLeaderModel`).
   The branch is a single `match` at the existing W2.1 site;
   no signature change to `promote_multiway`.
5. **W2.4 feedback pair becomes var_order-aware, including
   col-swapped key indices**
   (`xlog-runtime::executor::wcoj_dispatch`). Add private
   helper:
   ```rust
   fn feedback_pair_from_var_order(
       canonical_slot_rels: &[RelId],
       var_order: Option<&VariableOrder>,
   ) -> (RelId, RelId, Vec<usize>, Vec<usize>);
   ```
   The helper returns BOTH the relation pair AND the
   underlying-relation key indices for the slot-0 ⋈ slot-1
   join. This is required because triangle non-default
   leaders use **swapped** lookup atoms (slot 1 is an owned
   2-col view with cols swapped), and `record_join_result`
   stores keys against each relation's NATIVE column
   indexing — not the swapped view. The kernel invariant
   `slot0.col1 ≡ slot1.col0` holds for the views, but maps
   to different native key indices when slot 1 is swapped.

   **Locked rotated-feedback table** (derived from the W2.1
   locked permutation tables):

   | Shape | Leader | (rel_a, rel_b) | (left_keys, right_keys) | Source slot 1 swap |
   |-------|--------|----------------|-------------------------|--------------------|
   | Triangle | 0 (e_xy default) | (rel_xy, rel_yz) | [1] / [0] | none |
   | Triangle | 1 (e_yz)         | (rel_yz, rel_xz) | **[1] / [1]** | yes — slot 1 = e_xz↔ |
   | Triangle | 2 (e_xz)         | (rel_xz, rel_yz) | **[1] / [1]** | yes — slot 1 = e_yz↔ |
   | 4-cycle  | 0 (e_wx default) | (rel_wx, rel_xy) | [1] / [0] | none |
   | 4-cycle  | 1 (e_xy)         | (rel_xy, rel_yz) | [1] / [0] | none (rotation-only) |
   | 4-cycle  | 2 (e_yz)         | (rel_yz, rel_zw) | [1] / [0] | none (rotation-only) |
   | 4-cycle  | 3 (e_zw)         | (rel_zw, rel_wx) | [1] / [0] | none (rotation-only) |

   **Derivation** (verified against W2.1 plan §"Permutation
   Tables"): for triangle leader e_xz (idx 2), slot 0 =
   `rel_xz` (native (X, Z); join key = col1 = Z → native
   index 1). Slot 1 = `rel_yz` accessed via a swapped 2-col
   view exposing (Z, Y) to the kernel — but the underlying
   buffer's native layout is (Y, Z), and the join key Z lives
   at native col1, so the recorded key index for `rel_yz` =
   1, not 0. Same shape derivation for triangle leader e_yz.
   For 4-cycle (rotation-only, no swaps), every slot
   accesses its rel in native layout, so [1]/[0] is preserved.

   At the two `record_wcoj_feedback` call sites
   (`wcoj_dispatch.rs:882` triangle, `:1323` 4-cycle), replace
   the hardcoded `&slot_rels[..2]` + `vec![1] / vec![0]` with
   the helper's output:
   * `var_order = None` → `(slot_rels[0], slot_rels[1],
     vec![1], vec![0])` — bit-identical to current W2.4
     behavior.
   * `var_order = Some(vo)` → 4-tuple from the locked table
     above, looked up by `(shape, leader_idx)`.

   **Default `Disabled` config preserves W2.4 bit-identical**:
   the `var_order = None` branch matches the pre-W2.6 path
   (verified by the new Part D regression cert).

   **W2.6 cost-model key validation**: `HeatAwareLeaderModel`'s
   selectivity lookup must match against these same rotated
   keys. The model derives candidate keys per edge using the
   identical lookup-table logic so cached selectivity records
   produced by post-rotation feedback are correctly consumed
   on the next compile (no key mismatch → no cache poisoning).
6. **Snapshot heat preservation** (only if step 1 A4/A5 audit
   reveals a gap): minimal field-copy fix in `StatsSnapshot`
   round-trip / `merge_snapshot`. **No new struct fields**;
   reuses existing `RelationStats.heat`.
7. **Acceptance gates**: Part A unit (5) + Part B compile-time
   (4) + Part C real-runtime end-to-end (3) + Part D regression
   (2) + Part E rotated-feedback (1) = 15 tests. See §"Acceptance
   Gate" below for the specific tests.
8. **Workspace gate**: full slice 1–5 + W2.4 + W2.2 + W2.1 +
   W2.3 regression bit-identical when `CompilerConfig::default()`
   is in effect. The W2.1 acceptance gate (32 tests) must stay
   green; W2.3's 10 tests must stay green; W2.4 feedback path
   under `var_order = None` must produce bit-identical
   `StatsManager` state to pre-W2.6 (verified by
   `test_wcoj_record_join_result_feedback`'s 3 existing certs
   passing unchanged); this is the load-bearing default-config
   preservation.
9. **Evidence README + closure proposal + FF-merge**.

## Acceptance Gate

### Part A — Composite-score unit tests (5 tests)

`xlog-logic::wcoj_var_ordering::tests` lib module. All tests
construct `StatsManager` by hand (`StatsManager::new()` +
`register_relation` + `update_cardinality` + direct
`get_relation_stats_mut(rel).heat = h` for heat injection;
`set_join_selectivity(rel_a, rel_b, left_keys, right_keys, sel)`
for direct selectivity injection — sidesteps EMA so unit tests
pin exact values; this is acceptable here because Part A
exercises the formula in isolation, NOT the runtime feedback
loop). Locked formula constants: `HEAT_WEIGHT = 4.0`,
threshold = `0.5`.

* **`heat_aware_leader_picks_cold_when_hot_relation_at_default_idx`**:
  Triangle. Cards: all 3 rels = 100. Rel at idx 0 (default
  leader): heat = 0.5 → factor `(1 + 4*0.5) = 3`. Rels at idx
  1, 2: heat = 0 → factor 1. Sels: all default 1.0 → penalty
  = `1+1 = 2` per rel. Scores: idx0 = `100*3*2 = 600`, idx1 =
  idx2 = `100*1*2 = 200`. argmin = idx 1 (first hit). default
  = idx 0 = 600. Ratio `200/600 = 0.333 ≤ 0.5` → returns
  `Some(1)`.
* **`heat_aware_leader_demotes_relation_in_tight_edge`**:
  Triangle. Cards: 100 each. Heat: 0 each. One edge `(rel(0),
  rel(1))` has selectivity = 0.01 (tight); other edges
  default 1.0. Penalties: rel0 ∈ {(0,1),(0,2)} → `100 + 1 =
  101`. rel1 ∈ {(0,1),(1,2)} → `101`. rel2 ∈ {(1,2),(0,2)}
  → `1 + 1 = 2`. Scores: rel0 = rel1 = `100*1*101 = 10100`.
  rel2 = `100*1*2 = 200`. argmin = idx 2. default = idx 0 =
  10100. Ratio `200/10100 ≈ 0.02 ≤ 0.5` → returns `Some(2)`.
* **`heat_aware_leader_returns_none_when_heat_too_low`**:
  Triangle. Cards: 100 each. Heat at idx 0 = 0.1 (one
  `record_access` from cold), others = 0. Factor at idx 0 =
  `1 + 4*0.1 = 1.4`. Score idx 0 = `100*1.4*2 = 280`. Score
  idx 1, 2 = `200`. Ratio `200/280 ≈ 0.71 > 0.5` → returns
  `None`. Pins the heat-too-weak threshold boundary.
* **`heat_aware_leader_disabled_short_circuit`**:
  `WcojVarOrderingKind::Disabled` → returns `None` even with
  strong signals.
* **`heat_aware_leader_missing_card_safety_floor`**:
  One rel has zero/unregistered card → matches W2.1's
  missing-stats policy via shared `card_of` helper → returns
  `None`.

### Part B — Compile-time leader divergence via hand-built snapshot (4 tests)

`xlog-logic/tests/test_w26_part_b.rs`. Hand-built
`StatsSnapshot` (constructed via the existing public
`StatsSnapshot { relations, join_selectivities, rel_names }`
literal) lets each test pin the EXACT heat / selectivity /
cardinality values reaching the compile-time cost model. This
sidesteps EMA smoothing and proves the
`HeatAwareLeaderModel.pick_*_leader` decision is
deterministic against the locked formula. Part C exercises
the **runtime → snapshot** capture path end-to-end; Part B
exercises the **snapshot → cost-model** half independently.

For each of triangle + 4-cycle, two tests (hand-built
snapshots):

* **Same-cardinality + heat-bias produces non-default leader
  under HeatAware**: snapshot has all rels at card = 100;
  rel at canonical idx 0 has `heat = 0.5`; other rels have
  `heat = 0.0`. No `join_selectivities` entries (all default
  to 1.0 in the formula). Compile with
  `WcojVarOrderingKind::HeatAware`; assert
  `var_order.leader_idx ≠ 0`. Compile with
  `WcojVarOrderingKind::LeaderCardinality` on the SAME
  snapshot; assert `var_order is None` (cards are equal so
  W2.1 picks default leader and short-circuits).
* **Same-cardinality + tight-edge selectivity produces
  not-in-tight-edge leader under HeatAware**: snapshot has
  all rels at card = 100, all heat = 0.0. ONE
  `JoinSelectivity` record with `selectivity = 0.01` on the
  canonical edge `(rel_idx_0, rel_idx_1)` with the locked
  shared-variable keys. Compile with HeatAware; assert
  `var_order.leader_idx == 2` for triangle (the rel NOT in
  the tight edge); for 4-cycle's edge `(rel_idx_0,
  rel_idx_1)`, assert leader ∈ `{2, 3}` (NEITHER rel in
  the tight edge — both rel 2 and rel 3 have penalty `2`,
  so argmin picks the first non-zero non-tight-edge
  candidate). Compile with LeaderCardinality; assert
  `None` (cards equal).

Two tests × two shapes = 4 tests.

### Part C — Real-runtime observed-stat → leader change (3 tests)

`xlog-integration/tests/test_w26_heat_selectivity.rs`. These
certs prove the closure-board acceptance line — "selectivity +
heat statistics feed into the variable ordering" —
**end-to-end with REAL runtime-observed signals**, not
hand-built snapshots. Each cert: warm-up → capture
`executor.stats_snapshot()` → re-compile under HeatAware →
**assert the leader actually changed** vs LeaderCardinality
on the same snapshot AND row-set parity vs binary-join
reference.

#### Part C.1 — Selectivity drives leader (real W2.4 record_join_result)

**Fixture**: a **non-recursive** triangle with equal-card
EDBs. Cardinality stays equal because the rule is
non-recursive (no feedback loop to grow any rel). To get
the 4 dispatches needed for EMA convergence, the test
**re-uses the same compiled plan + executor** for 4 sequential
`execute_plan` invocations — each invocation produces 1
WCOJ dispatch and 1 `record_join_result` write.

```text
pred e1(u32, u32). pred e2(u32, u32). pred e3(u32, u32).
pred tri(u32, u32, u32).
tri(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z).
```

**Locked EDB inputs (5 rows each, 1 valid triangle)**:
```text
e1 = [(1,2), (10,99), (20,98), (30,97), (40,96)]
e2 = [(2,3), (50,51), (60,61), (70,71), (80,81)]
e3 = [(1,3), (50,52), (60,62), (70,72), (80,82)]
```

Trace `tri(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z)`:
* `e1(1,2), e2(2,3), e3(1,3)`: ✓ → `tri(1,2,3)` (1 row).
* `e1(10,99), e2(99,_)` — no e2 entries with Y=99. ∅.
* `e1(20,98), e2(98,_)` — none. ∅.
* `e1(30,97), e2(97,_)` — none. ∅.
* `e1(40,96), e2(96,_)` — none. ∅.

Output per dispatch = 1 triangle row. Cards: `e1.card == 5,
e2.card == 5, e3.card == 5` (no recursion → cards never
change).

**EMA selectivity progression** (input_rows = card_a *
card_b = 5*5 = 25 per `record_wcoj_feedback`'s
`a.saturating_mul(b)` at `wcoj_dispatch.rs:677`;
output_rows = 1; observed_sel = 1/25 = 0.04):

| Dispatch # | `record_join_result` write | `(rel_xy, rel_yz).selectivity` after EMA |
|------------|-----------------------------|------------------------------------------|
| 1 | `0.7*1.0 + 0.3*0.04` | **0.712** |
| 2 | `0.7*0.712 + 0.3*0.04` | **0.510** |
| 3 | `0.7*0.510 + 0.3*0.04` | **0.369** |
| 4 | `0.7*0.369 + 0.3*0.04` | **0.270** |

After 4 dispatches: `(rel_xy, rel_yz).selectivity ≈ 0.270`.
Penalty contribution `1/0.270 ≈ 3.70`.

**Heat under force-WCOJ-on**: when WCOJ dispatches succeed,
the dispatcher reads input buffers directly from the relation
store and **bypasses** `node_dispatch::execute_scan` for the
matched MultiWayJoin's input rels — so `record_access` does
NOT fire for `e1`, `e2`, `e3` during the 4 force-WCOJ-on
invocations. Heat for all three rels remains at the W2.4
+ W2.3 baseline (`0.0` if no other rule scanned them; in
this fixture the only rule is `tri`, so heat = 0). Heat is
therefore uniform `0.0` across all three rels in Part C.1.
This keeps Part C.1 a **purely selectivity-driven** cert.

**HeatAware score** (cards = 5, heat = 0 uniform):
* rel_xy ∈ edges {(xy,yz), (xy,xz)}: penalty `3.70 + 1 = 4.70`.
* rel_yz ∈ edges {(xy,yz), (yz,xz)}: penalty `4.70`.
* rel_xz ∈ edges {(yz,xz), (xy,xz)}: penalty `1 + 1 = 2`.
* Heat factor: `1 + 4*0 = 1.0` (uniform).
* score(rel_xy) = `5 * 1 * 4.70 = 23.50`.
* score(rel_yz) = `5 * 1 * 4.70 = 23.50`.
* score(rel_xz) = `5 * 1 * 2.00 = 10.00`.

argmin = rel_xz (canonical idx 2). default leader = canonical
idx 0 = rel_xy = 23.50. Ratio `10.00 / 23.50 ≈ 0.425 ≤ 0.5`
→ returns `Some(2)`.

Under `LeaderCardinality` on the same snapshot: cards equal
→ argmin == default idx 0 → returns `None`.

* **`triangle_real_observed_selectivity_drives_heat_aware_leader_to_idx_2`**:
  Phase 1: register relations, put EDBs, compile under
  `CompilerConfig::default()`, execute_plan **4 times** with
  force-WCOJ-on. After all 4 calls, assert
  `executor.wcoj_triangle_dispatch_count() == 4`.
  Phase 2: assert
  `executor.stats_snapshot().relations[rel_xy].cardinality
  == 5` and same for `rel_yz`, `rel_xz` — pins equal-card
  invariant. Assert
  `executor.stats_snapshot().join_selectivities` contains
  exactly one entry on the canonical `(rel_xy, rel_yz)`
  pair with `selectivity` in the band `[0.25, 0.30]`
  (allows ±2% f64 rounding; expected exact value 0.270 per
  the EMA table above).
  Phase 3: `let snap = executor.stats_snapshot();`. Re-compile
  the SAME source via
  `compile_with_config_and_stats_snapshot(source,
  &cfg_heat_aware, Some(&snap))`. Inspect the resulting plan
  for `MultiWayJoin.var_order`. Assert `var_order ==
  Some(VariableOrder { leader_idx: 2, .. })`.
  Phase 4: same snapshot, re-compile under
  `WcojVarOrderingKind::LeaderCardinality`. Assert
  `var_order == None` (cards equal → W2.1 short-circuits).
  This proves the leader change is **selectivity-driven**, NOT
  cardinality-driven.
  Phase 5: execute the HeatAware-compiled plan + force-WCOJ-on
  on a fresh executor with the same EDBs. Assert row set on
  `tri` equals binary-join reference (a separate fresh-executor
  force-WCOJ-off run on the same source + EDBs); both yield
  exactly `{(1, 2, 3)}`.

#### Part C.2 — Heat drives leader (real `record_access`)

**Fixture**: a non-recursive triangle with equal-card EDBs +
two warm-up phases that drive heat differential through real
`record_access` invocations:

```text
pred e1(u32, u32). pred e2(u32, u32). pred e3(u32, u32).
pred tri(u32, u32, u32).
pred dummy_e1(u32).
dummy_e1(X) :- e1(X, _).
tri(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z).
```

EDBs: each 5 rows, equal card.

**Warm-up phase A** (card seeding + uniform heat baseline):
compile with `dummy_e1` rule + `tri` rule together; execute_plan
once under force-WCOJ-off. Per-rel `record_access` calls per
this single invocation:
* e1: 1 (from `dummy_e1` body) + 1 (from `tri` body) = 2.
* e2: 1 (from `tri` body) = 1.
* e3: 1 (from `tri` body) = 1.

After Phase A: e1.heat = `1 - 0.9^2 = 0.19`; e2.heat = e3.heat
= `0.1`. Cards: each = 5.

**Warm-up phase B** (drive e1 hot): compile with ONLY the
`dummy_e1` rule (a separate source); execute_plan N=10 times.
Each call: e1 gets 1 access; e2, e3 get 0 accesses (not
referenced in the program). After Phase B: e1.heat continues
EMA from 0.19 through 10 more accesses: `1 - 0.9^12 = 0.718`.
e2.heat / e3.heat unchanged at `0.1`.

Score (cards equal = 5, no observed selectivity):
* e1 factor = `1 + 4*0.718 = 3.872`.
* e2 / e3 factor = `1 + 4*0.1 = 1.4`.
* Penalty per rel (no selectivity records): 2.
* e1 score = `5 * 3.872 * 2 = 38.72`.
* e2 / e3 score = `5 * 1.4 * 2 = 14`.

argmin = e2 (canonical idx 1, first hit). default = e1 (idx 0)
= 38.72. Ratio `14/38.72 = 0.362 ≤ 0.5` → returns `Some(1)`. ✓

* **`triangle_real_observed_heat_drives_heat_aware_leader_to_idx_1`**:
  Phase A: `let mut compiler = Compiler::new();
  let plan_seed = compiler.compile_with_config_and_stats_snapshot(SOURCE_AB, &cfg_default, None)?;`
  where `SOURCE_AB` is the combined `dummy_e1` + `tri` source.
  `executor.execute_plan(&plan_seed)?;` once under
  force-WCOJ-off. Assert `executor.stats().get_relation_stats(rel_e1).cardinality
  == 5` (card seeding worked); same for e2, e3.
  Phase B: compile a heater-only source `dummy_e1(X) :- e1(X, _).`,
  execute_plan N=10 times against the same executor.
  Phase C: `let snap = executor.stats_snapshot();`. Assert
  `snap.relations[e1].heat ≥ 0.6` AND
  `snap.relations[e2].heat ≤ 0.15` AND
  `snap.relations[e3].heat ≤ 0.15` (pins the warm-up
  produced the expected differential).
  Phase D: re-compile `tri`-only source via
  `compile_with_config_and_stats_snapshot(SOURCE_TRI, &cfg_heat_aware,
  Some(&snap))`. Assert `var_order == Some(VariableOrder {
  leader_idx: 1, .. })`. Re-compile same snapshot under
  LeaderCardinality. Assert `var_order == None` (cards equal).
  Phase E: execute the HeatAware plan + force-WCOJ-on. Assert
  row set on `tri` equals binary-join reference.

#### Part C.3 — 4-cycle real-observed selectivity drives leader

Same **selectivity-driven** shape as Part C.1, on the
4-cycle. Non-recursive 4-cycle source with equal-card
EDBs (5 each, 1 valid 4-cycle output). 4 sequential
`execute_plan` invocations against the same compiled plan +
executor produce 4 `record_join_result` writes on the
canonical `(rel_e1, rel_e2)` pair (4-cycle slot 0 = `e_wx`,
slot 1 = `e_xy` per the locked permutation table —
rotation-only, no swaps; keys `[1]/[0]`).

EMA progression: same as Part C.1 (input_rows = 5*5 = 25,
output_rows = 1, observed_sel = 0.04, after 4 dispatches
`(rel_e1, rel_e2).selectivity ≈ 0.270`).

For 4-cycle, each rel is in 2 of the 4 cycle edges
`{(0,1), (1,2), (2,3), (3,0)}`:
* rel_e1 ∈ `{(0,1), (3,0)}`: penalty `3.70 + 1 = 4.70`.
* rel_e2 ∈ `{(0,1), (1,2)}`: penalty `4.70`.
* rel_e3 ∈ `{(1,2), (2,3)}`: penalty `1 + 1 = 2`.
* rel_e4 ∈ `{(2,3), (3,0)}`: penalty `2`.

**Heat under force-WCOJ-on**: same rationale as Part C.1 —
WCOJ-success bypasses `execute_scan`, so `record_access`
does NOT fire for the matched 4-cycle's input rels during
the 4 invocations. Heat is uniform `0.0` across all four
rels.

Score (cards 5, heat 0 uniform):
* score(rel_e1) = `5 * 1 * 4.70 = 23.50`.
* score(rel_e2) = `23.50`.
* score(rel_e3) = `5 * 1 * 2 = 10.00`.
* score(rel_e4) = `10.00`.

argmin = rel_e3 (canonical idx 2, first hit; rel_e4 ties
but argmin breaks ties to first index). Ratio `10.00/23.50
≈ 0.425 ≤ 0.5` → returns `Some(2)`.

* **`cycle4_real_observed_selectivity_drives_heat_aware_leader_to_idx_2`**:
  same Phase 1-5 structure as Part C.1. Phase 3 asserts
  `var_order == Some(VariableOrder { leader_idx: 2, .. })`.
  Phase 4 (LeaderCardinality on same snapshot): asserts
  `var_order == None`. Phase 5 row-set parity. Leader
  change is **selectivity-driven** — cards equal, heat
  uniform 0 (WCOJ-success bypasses `execute_scan`), so the
  only differential signal is the recorded selectivity on
  `(rel_e1, rel_e2)`.

### Part D — Default-config bit-identical regression (2 tests)

`xlog-integration/tests/test_w26_heat_selectivity.rs`.

* **`default_config_bit_identical_to_w23_baseline`**: compile +
  execute the slice-4 linear-recursive triangle fixture under
  `CompilerConfig::default()` (Disabled). Assert dispatch counter
  matches the W2.3 baseline (`== 4` for triangle) AND row sets
  match the binary-join reference. Same shape proven by W2.1
  Part D and W2.3 Part C; W2.6 must not perturb.
* **`record_wcoj_feedback_var_order_none_pair_unchanged`**:
  pin the W2.6 step-5 contract — when `var_order = None`,
  `record_wcoj_feedback` records on the canonical
  `(slot_rels[0], slot_rels[1])` pair with keys `[1]/[0]`.
  Run a triangle fixture under `CompilerConfig::default()`
  + force-WCOJ-on (one dispatch). Capture
  `executor.stats_snapshot()`. Iterate
  `snapshot.join_selectivities` looking for an entry whose
  rel-pair (canonicalized via the same `canonical_join_key`
  rule) matches `(rel_xy, rel_yz)` with `left_keys == [1]`
  and `right_keys == [0]`. Assert that exactly one such
  entry exists. Pre-W2.6 baseline preserved.

### Part E — `var_order = Some` rotated-feedback cert (1 test)

`xlog-integration/tests/test_w26_heat_selectivity.rs`.

Pins the **W2.6 step-5 contract**: when HeatAware produces a
non-default leader, the dispatcher's `record_wcoj_feedback`
(rerouted through `feedback_pair_from_var_order`) records
selectivity on the **rotated** slot-0 / slot-1 pair, NOT the
canonical one.

**Fixture design (revised to avoid the iteration-3
impossibility)**: the leader change must be driven by **HEAT
only** in the input snapshot, leaving `join_selectivities`
**empty** before execution. Selectivity records appear ONLY
post-execution from `record_wcoj_feedback`'s W2.6-rotated
write. This makes the "no canonical entry" assertion
verifiable: the canonical pair was never recorded, full stop.

Specifically:

* Build a hand-built `StatsSnapshot` with:
  * 3 rels (`rel_xy`, `rel_yz`, `rel_xz`) at equal card = 100.
  * `rel_xy.heat = 0` (canonical idx 0; default leader).
  * `rel_yz.heat = 0` (canonical idx 1).
  * `rel_xz.heat = 0` (canonical idx 2).
  * **`rel_xy.heat = 0.5`** ← demote canonical idx 0.
    Heat factor: 3.0; others: 1.0.
  * `rel_yz.heat = 0.5` ← also demote idx 1, so argmin = idx 2.
    Now: rel_xy factor 3, rel_yz factor 3, rel_xz factor 1.
  * Score: rel_xy = `100*3*2 = 600`; rel_yz = same;
    rel_xz = `100*1*2 = 200`. argmin = idx 2 (rel_xz).
    default = idx 0 = 600. Ratio `200/600 = 0.333 ≤ 0.5`. ✓
  * `join_selectivities`: **empty** (no pre-recorded entries).
  * `rel_names` populated for the 3 rels.

* **`heat_aware_rotated_leader_records_feedback_on_rotated_pair`**:
  Phase 1: Build the snapshot above. Compile a triangle source
  via `compile_with_config_and_stats_snapshot(source,
  &cfg_heat_aware, Some(&snapshot))`. Inspect the resulting
  plan: assert `var_order == Some(VariableOrder { leader_idx:
  2, .. })`.
  Phase 2: Build a fresh executor; register the 3 rels;
  put EDB buffers (5 rows each, structured to produce ≥ 1
  triangle output row). **Pre-condition assertion**:
  `executor.stats_snapshot().join_selectivities.is_empty()`
  (no canonical pair, no rotated pair — fresh executor).
  Phase 3: execute the HeatAware plan + force-WCOJ-on.
  Assert `executor.wcoj_triangle_dispatch_count() ≥ 1`.
  Phase 4: capture `executor.stats_snapshot()`. Assert
  exactly **one** `JoinSelectivity` entry exists in
  `snapshot.join_selectivities`. Inspect that entry per the
  step-5 locked rotated-feedback table:
  * The locked permutation table for triangle e_xz-leader
    (canonical idx 2): slots are `[e_xz, e_yz↔, e_xy]` (W2.1
    plan §"Permutation Tables / Triangle"). Slot 0 = rel_xz
    native (X, Z), slot 1 = rel_yz **swapped** view exposing
    (Z, Y) to the kernel; underlying rel_yz buffer remains
    in native (Y, Z) layout.
  * **Step-5 locked entry for triangle leader = 2**:
    `(rel_a, rel_b) = (rel_xz, rel_yz)`,
    `(left_keys, right_keys) = ([1], [1])`. Both keys are
    `[1]` because the join variable Z lives at native col1
    in BOTH `rel_xz` (native (X, Z)) and `rel_yz` (native
    (Y, Z)) — the swap reshapes the kernel's view but does
    not change the underlying relation's column indexing.
  * Assert the entry's `(left_rel, right_rel)` after
    `StatsManager::canonical_join_key` canonicalization
    matches the canonicalized form of `(rel_xz, rel_yz)`.
  * Assert `left_keys == [1]` AND `right_keys == [1]` after
    the same canonicalization-induced key swap (if the
    canonical_join_key swapped the relations, the keys
    swap correspondingly — but here both keys are `[1]`,
    so the assertion is symmetric).
  * Assert NO entry on the canonicalized form of
    `(rel_xy, rel_yz)` with keys `[1]/[0]` (the pre-W2.6
    feedback target — proven absent because Phase 2
    verified empty pre-condition AND Phase 4 asserts
    exactly one entry on the rotated pair).

**Total: Part A 5 + Part B 4 + Part C 3 + Part D 2 + Part E 1
= 15 tests.**

Part C's 3 tests are **REAL-runtime** end-to-end certs.
Across the set, both observed signals are exercised:
* Part C.1 (triangle) and C.2 (heat) each independently
  prove ONE side of the closure-board acceptance line.
  C.1 fires real `record_join_result` (selectivity);
  C.2 fires real `record_access` (heat). Neither fires
  both signals as differential drivers — under
  force-WCOJ-on, `record_access` is bypassed for the
  matched MultiWayJoin's input rels, so C.1's leader
  change is purely selectivity-driven and C.2's is purely
  heat-driven.
* Part C.3 mirrors C.1 on 4-cycle (selectivity-driven).

This satisfies the closure-board acceptance line
"Selectivity + heat statistics feed into the variable
ordering" with observed signals — selectivity proven by
C.1 + C.3, heat proven by C.2 — not synthetic snapshots.

## Decisions / Limitations

* **Composite score is locked** (per the formula above) so the
  acceptance gate is deterministic. Tuning the score (e.g., heat
  weight, selectivity weight, exponents) is **owned by W5.2**
  benchmark work; not in W2.6 scope.
* **No new env var**, no new compile entry point, no new IPC.
  Existing `compile_with_config_and_stats_snapshot` carries the
  loop end-to-end.
* **Default config (`Disabled`) preserves W2.1 + W2.3 + W2.4
  semantics bit-identically.** Part D pins this.
* **Hot/cold inversion under HeatAware**: a hot rel with very
  small cardinality may still get picked as leader (cardinality
  dominates the score). The score's heat factor is
  `(1 + 4 * heat)`, so heat ∈ [0, 1) yields factor ∈ [1, 5);
  heat alone can produce up to ~5× score multiplier vs a cold
  rel of the same card. Cards differing by more than ~5× still
  override heat.
* **No multi-recursive support**: W4.1 owns the gate; same
  carve-out as W2.3.
* **No kernel changes**, no IR changes, no new CUDA helpers.
  W2.6 is purely a new cost-model implementation slotted in
  behind the W2.1 trait.

## Risk & Open Questions

* **Q1 — Heat propagation through snapshot**: if step 1's audit
  reveals that `RelationStats.heat` is dropped during the
  W2.2 rel_id-remap path in `compile.rs`, step 5 adds a
  minimal preservation fix. Cost: 1 line of field copy.
* **Q2 — Selectivity propagation**: W2.4's
  `record_join_result` writes (EMA-smoothed) are remapped via
  the snapshot's `join_selectivities` field at
  `compile.rs:222-258`. Already end-to-end. No fix needed.
* **Q3 — Default `selectivity = 1.0` for unobserved pairs**:
  the formula uses `1.0 / max(0.01, selectivity)`, so
  unobserved pairs default to `1.0` (no filter assumption).
  This means the model degrades to W2.1-like behavior when
  no selectivity has been observed. Acceptable; W2.6 by design
  needs observed stats.
* **Q4 — Same-card + same-heat fixtures**: Part B's test
  fixtures need explicit signal injection because the slice-4
  EDB cardinalities are too uniform to drive the model
  organically. Tests construct `StatsSnapshot` by hand; no
  fixture re-design needed in Part C since the warm-up phase
  populates real stats.
* **Q5 — Promoter dispatch site**: the existing W2.1 wiring
  calls `LeaderCardinalityModel.pick_triangle_leader(...)` at
  one site in `promote.rs::try_promote_triangle` (and one in
  `try_promote_4cycle`). W2.6's `match` adds 2 lines per site.
  Static dispatch via `match` chosen over a trait-object
  `Box<dyn WcojVariableOrderingModel>` to avoid heap allocation
  at the promote-time hot path.

## Provenance

* Closure board `docs/v065-closure-board.md` Wave 2, W2.6.
* ROADMAP item #16.
* W2.1 (`d1b13951..f82f9995..da644e3d`) introduced
  `WcojVariableOrderingModel` trait + `LeaderCardinalityModel`
  default. W2.6 is the second impl.
* W2.4 (`f586ce34`) wired
  `Executor::record_wcoj_feedback` →
  `StatsManager::record_join_result` (EMA-smoothed,
  `manager.rs:345-348`). W2.6 reads via
  `StatsManager::get_join_selectivity`.
* W2.3 (`d10bb72a..72988c6c..cf57f3a1`) wired per-iteration
  cardinality update for recursive SCCs. W2.6 reads.
* `node_dispatch.rs:26` `record_access` heat path is the
  primary heat signal source.
* Code anchors: `crates/xlog-stats/src/stats.rs:25` (heat field),
  `crates/xlog-stats/src/manager.rs:204` (record_access),
  `crates/xlog-stats/src/manager.rs:239` (estimate_join_cardinality
  pattern), `crates/xlog-logic/src/wcoj_var_ordering.rs`
  (W2.1's W2.6-extension target).

## Process Rule Compliance

* **Process rule #1**: this slice does NOT self-mark W2.6 DONE.
  End-of-slice commit proposes the OPEN → DONE transition;
  user reviews + explicitly approves; a separate follow-up
  commit applies the board update.
* **Process rule #2**: every commit references W2.6.
* **Process rule #3**: this plan header opens with "Closes
  W2.6."
* **Process rule #5**: no `v0.6.6` references introduced.
* **Process rule #6**: no push, no tag.
