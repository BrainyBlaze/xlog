# W2.6 Evidence — Heat + Selectivity Feedback into Variable Ordering

**Closes board item: W2.6 only.**
**Date:** 2026-05-04
**Branch:** `feat/w26-heat-selectivity-variable-ordering`
**Base:** `main` at `cf57f3a1` (W2.3 closure commit).
**Head:** `7e76b3dd` (3 commits on branch).
**Plan:** `docs/plans/2026-05-04-w26-heat-selectivity-variable-ordering-plan.md`
(approved iteration 7).

## Summary

Successful WCOJ dispatches now consume **two** runtime-observed
signals — access **heat** (EMA-smoothed via `record_access`) and
join **selectivity** (EMA-smoothed via `record_join_result`) —
to pick a non-default leader slot for triangle and 4-cycle
WCOJ shapes. The new `HeatAwareLeaderModel` and
`WcojVarOrderingKind::HeatAware` opt-in selector composite-score
each input; hot relations and rels in tight (low-selectivity)
edges get demoted from the leader slot, cold relations are
preferred. The closure-board acceptance line — *"hot relation
gets preferred lookup-key slot; cold extensional relation gets
iteration-key slot; row-set agreement preserved"* — is locked
end-to-end across 16 tests on real-runtime signals.

`CompilerConfig::default()` continues to select
`WcojVarOrderingKind::Disabled`, preserving slice 1/2/4 + W2.2
+ W2.1 + W2.3 + W2.4 dispatch and row-set semantics
**bit-identically**.

## Acceptance Properties (16 tests across Part A / B / C / D / E)

| Part | # tests | Location | What it locks |
|------|---------|----------|---------------|
| A | 5 | `crates/xlog-logic/src/wcoj_var_ordering.rs::tests` | `HeatAwareLeaderModel` unit-level: locked composite formula, threshold gate, key-validation safety floor, disabled short-circuit. |
| B | 4 | `crates/xlog-logic/tests/test_w26_part_b.rs` | Compile-time leader divergence via hand-built `StatsSnapshot` (2 shapes × 2 signal types). Sidesteps EMA smoothing — pins exact heat/selectivity values reaching the cost model. |
| C | 4 | `crates/xlog-integration/tests/test_w26_heat_selectivity.rs` | **Real-runtime** end-to-end certs: warm-up → `executor.stats_snapshot()` → re-compile under HeatAware → leader changes vs LeaderCardinality on the same snapshot AND row-set parity vs binary-join reference. C.1: triangle selectivity-driven; C.2: triangle heat-driven, zero cold-baseline; C.3: 4-cycle selectivity-driven; C.4: triangle heat-driven, non-zero cold-baseline (≈ 0.1 on cold rels). |
| D | 2 | same | Default `CompilerConfig::default()` is bit-identical to W2.3 baseline; W2.4 feedback's canonical `(slot_rels[0], slot_rels[1])` pair with `[1]/[0]` keys preserved when `var_order = None`. |
| E | 1 | same | When HeatAware emits a non-default leader on triangle (idx 2), W2.6's `feedback_pair_from_var_order` reroute records selectivity on the **rotated** pair (canonicalized) with `[1]/[1]` keys. |

### Part A — `HeatAwareLeaderModel` unit certs (5 tests)

| Test | Property |
|------|----------|
| `heat_aware_leader_picks_cold_when_hot_relation_at_default_idx` | Hot relation at canonical idx 0 demoted; cold rel at non-default idx wins. |
| `heat_aware_leader_demotes_relation_in_tight_edge` | Tight selectivity record on (idx 0, idx 1) inflates penalty for both — argmin = idx 2. |
| `heat_aware_leader_returns_none_when_heat_too_low` | Heat below threshold → ratio above 0.5 → `None` returned. |
| `heat_aware_leader_disabled_short_circuit` | `Disabled` config → `None` regardless of stats. |
| `heat_aware_leader_missing_card_safety_floor` | Any rel with missing/zero card → `None` (no mis-pick on partial stats). |

### Part B — Hand-built snapshot leader divergence (4 tests)

| Test | Locked math |
|------|-------------|
| `triangle_heat_bias_heat_aware_picks_non_default_leader_card_eq_returns_none` | card=100 each, heat=(0.5, 0, 0). Score idx0 = 600, idx1=idx2 = 200. argmin=idx1 (first hit). Ratio 200/600 = 0.333 ≤ 0.5 → `Some(1)`. LeaderCardinality on same snapshot: cards equal → `None`. |
| `triangle_selectivity_bias_heat_aware_picks_not_in_tight_edge` | card=100 each, heat=0, sel(e1,e2)=0.01. Penalties: rel_e1=101, rel_e2=101, rel_e3=2. Scores: idx0=idx1=10100, idx2=200. argmin=idx2. Ratio 200/10100 ≈ 0.020 → `Some(2)`. LeaderCardinality: `None`. |
| `cycle4_heat_bias_heat_aware_picks_non_default_leader_card_eq_returns_none` | Same shape on 4-cycle — heat 0.5 on idx 0 → `Some(1)`; cards equal → LeaderCardinality `None`. |
| `cycle4_selectivity_bias_heat_aware_picks_not_in_tight_edge` | Tight sel(e1,e2)=0.01 → score idx0=idx1=10100, idx2=idx3=200; argmin=idx2 (first hit) → `Some(2)`. LeaderCardinality: `None`. |

### Part C — Real-runtime end-to-end certs (4 tests)

#### C.1 `triangle_real_observed_selectivity_drives_heat_aware_leader_to_idx_2`

* **Warm-up phase**: 4 sequential `execute_plan` calls under
  default compiler config + force-WCOJ-on. Each call dispatches
  WCOJ and fires `record_join_result(rel_xy, rel_yz, [1], [0],
  25, 1)` — input_rows = card_e1 × card_e2 = 5*5 = 25;
  observed_sel = 1/25 = 0.04.
* **EMA progression** (cards seeded at 5, EDBs are 5 rows; under
  force-WCOJ-on the WCOJ-success path bypasses
  `node_dispatch::execute_scan`'s auto-update so seeded cards
  persist through all 4 calls):

  | Dispatch # | EMA `selectivity` |
  |-----------:|-----------------:|
  | 1 | 0.712 |
  | 2 | 0.510 |
  | 3 | 0.369 |
  | 4 | **0.270** |

* **Snapshot pre-conditions** (asserted in test):
  cards remain at the seeded `5` (force-WCOJ-on bypasses
  `node_dispatch::execute_scan`'s auto-update for matched WCOJ
  inputs). Exactly one `JoinSelectivity` entry on
  canonical(rel_xy, rel_yz) with
  `(left_keys, right_keys) = ([1], [0])` and
  `selectivity ∈ [0.25, 0.30]`.
* **HeatAware re-compile** with this snapshot:
  * Penalty(rel_xy) = 1/0.270 + 1 ≈ 4.70
  * Penalty(rel_yz) = 4.70
  * Penalty(rel_xz) = 1 + 1 = 2.0
  * Heat factor uniform 1.0 (cards seeded; force-WCOJ-on
    bypasses scan-driven `record_access` for matched WCOJ
    inputs).
  * score(rel_xy) = 5 × 1 × 4.70 = 23.50
  * score(rel_yz) = 5 × 1 × 4.70 = 23.50
  * score(rel_xz) = 5 × 1 × 2.0 = 10.00
  * argmin = idx 2; default(idx 0) = 23.50; ratio 10/23.50 ≈
    0.425 ≤ 0.5 → `Some(2)`. ✓ Asserted.
* **Promoter shape note**: at card=5, the lowerer's bushy DP
  emits a right-deep `Project(Join(Scan, Join(Scan, Scan)))`
  triangle. W2.6's `normalize_triangle_to_left_deep`
  (`crates/xlog-logic/src/promote.rs`) commutativity-rewrites
  to canonical left-deep before the matcher runs.
* **LeaderCardinality** on the same snapshot returns `None`
  (cards equal — W2.1 short-circuits at idx 0). ✓ Asserted.
* **Row-set parity**: HeatAware-compiled plan + force-WCOJ-on
  vs. binary-join reference (force-WCOJ-OFF) on a fresh
  executor with the same EDBs — both yield exactly `{(1,2,3)}`.
  ✓ Asserted.

#### C.2 `triangle_real_observed_heat_drives_heat_aware_leader_to_idx_1` (zero cold-baseline)

* **Warm-up phase**: heater-only source `dummy_e1(X) :- e1(X, _).`
  × **11 sequential** `execute_plan` calls under triangle WCOJ
  kill-switch (`with_wcoj_triangle_dispatch_disabled(Some(true))`).
  Each call scans `e1` once via `node_dispatch::execute_scan`,
  which advances `e1.heat` by one EMA step
  (`heat = heat * 0.9 + 0.1`). `e2` / `e3` are NEVER scanned in
  this rule, so their heat stays at the initial `0.0`.
* **Why heater-only**: the binary-join path
  (`crates/xlog-runtime/src/executor/node_dispatch.rs:343`) calls
  `record_join_result` after EVERY hash join, which would create
  a `(rel_xy, rel_yz)` selectivity record (`sel ≈ 0.712` after
  one EMA step) — that would perturb the intended heat-only
  signal. Heater-only keeps the snapshot's `join_selectivities`
  empty (also asserted) and the cert purely heat-driven at the
  zero-cold-baseline edge case. The non-zero-baseline case the
  plan iteration 7 originally specified is covered by C.4 below.
* **Heat math** (asserted):
  * `e1.heat = 1 - 0.9^11 ≈ 0.686` ≥ 0.6 ✓
  * `e2.heat = 0` ≤ 0.05 ✓
  * `e3.heat = 0` ≤ 0.05 ✓
  * `snap.join_selectivities.is_empty()` ✓
* **HeatAware re-compile** with this snapshot:
  * Heat factor: e1 = 1+4·0.686 = 3.744; e2/e3 = 1.0.
  * No selectivity records → penalty = 1+1 = 2 per rel.
  * score(e1) = 5 × 3.744 × 2 = 37.44
  * score(e2) = score(e3) = 5 × 1.0 × 2 = 10
  * argmin = idx 1 (e2, first-hit ties); default(idx 0) = 37.44;
    ratio 10/37.44 ≈ 0.267 ≤ 0.5 → `Some(1)`. ✓ Asserted.
* **LeaderCardinality**: `None` (cards equal). ✓ Asserted.
* **Row-set parity**: HeatAware + force-WCOJ-on equals
  binary-join reference; both yield `{(1,2,3)}`. ✓ Asserted.

#### C.3 `cycle4_real_observed_selectivity_drives_heat_aware_leader_to_idx_2`

Same shape as C.1 on 4-cycle: 4 sequential `execute_plan`
invocations on `cyc(W, X, Y, Z) :- e1(W, X), e2(X, Y), e3(Y, Z),
e4(Z, W)` under default config + force-4cycle-on. EMA progression
identical to C.1 (5×5 EDBs, seeded cards 5, sel=0.04 per dispatch
→ EMA after 4 ≈ 0.270).

* Pre-conditions: cards remain 5 each; one
  `JoinSelectivity` entry on canonical(e1, e2) keys [1]/[0],
  `selectivity ∈ [0.25, 0.30]`.
* **HeatAware re-compile**:
  * 4-cycle is rotation-only (no slot-swaps); every edge's
    keys are `[1]/[0]` in canonical layout. Edge (e1, e2) with
    `sel ≈ 0.270` → penalty `1/0.270 ≈ 3.70`. Each rel sits
    in 2 of the 4 cycle edges:
    * rel_e1 ∈ {(0,1) tight, (3,0) default}: 3.70 + 1 = 4.70.
    * rel_e2 ∈ {(0,1) tight, (1,2) default}: 4.70.
    * rel_e3 ∈ {(1,2), (2,3)} both default: 1 + 1 = 2.
    * rel_e4 ∈ {(2,3), (3,0)} both default: 2.
  * score(e1)=score(e2) = 5 × 1 × 4.70 = 23.50;
    score(e3)=score(e4) = 5 × 1 × 2 = 10.
  * argmin = idx 2 (e3, first-hit ties). Ratio 10/23.50 ≈
    0.425 ≤ 0.5 → `Some(2)`. ✓ Asserted.
* **Promoter shape note**: at card=5, the lowerer's bushy DP
  emits a fully-right-deep
  `Project(Join(Scan, Join(Scan, Join(Scan, Scan))))` 4-cycle.
  W2.6's `normalize_4cycle_to_bushy` (`crates/xlog-logic/src/promote.rs`)
  detects the rotation-only canonical-cycle pattern and rebuilds
  the canonical bushy `Project(Join(Join, Join))` form before the
  matcher runs.
* **LeaderCardinality**: `None` (cards equal). ✓ Asserted.
* **Row-set parity**: HeatAware + force-4cycle-on vs.
  binary-join reference; both yield `{(1,2,3,4)}`. ✓ Asserted.

#### C.4 `triangle_real_observed_heat_with_baseline_drives_heat_aware_leader_to_idx_1` (non-zero cold-baseline)

The plan iteration 7 case for "heat differential where cold
rels have a realistic non-zero baseline (~ 0.1)". Replaces the
plan's combined `dummy_e1 + tri` source — which would have
introduced a binary-join `record_join_result` selectivity entry
(see C.2 rationale above) — with a join-free triple-dummy source
`dummy_e1 + dummy_e2 + dummy_e3`, three single-Scan rules. Each
rule scans one EDB, so Phase A gives every rel exactly one
`record_access` call without any join.

* **Phase A (baseline)**: triple-dummy source × 1 →
  `e1.heat = e2.heat = e3.heat = 0.1`.
* **Phase B (heater)**: dummy_e1-only × 11 →
  `e1.heat = 1 - 0.9^12 ≈ 0.7176`; e2 / e3 unchanged at 0.1.
* **Asserted heat values** (band-tested for robustness):
  * `e1.heat ≥ 0.6` ✓
  * `e2.heat ∈ [0.05, 0.15]` (≈ 0.1) ✓
  * `e3.heat ∈ [0.05, 0.15]` (≈ 0.1) ✓
  * `snap.join_selectivities.is_empty()` — triple-dummy + heater
    introduces zero `record_join_result` calls. ✓ Asserted.
* **HeatAware re-compile** (cards 5, no selectivity records):
  * Heat factor: e1 = 1 + 4·0.7176 ≈ 3.870; e2/e3 = 1 + 4·0.1 = 1.4.
  * Penalty per rel = 1 + 1 = 2.
  * score(e1) = 5 × 3.870 × 2 ≈ 38.70
  * score(e2) = score(e3) = 5 × 1.4 × 2 = 14
  * argmin = idx 1 (e2, first-hit ties). Ratio 14/38.70 ≈ 0.362
    ≤ 0.5 → `Some(1)`. ✓ Asserted.
* **LeaderCardinality**: `None` (cards equal). ✓ Asserted.
* **Row-set parity**: HeatAware + force-WCOJ-on equals binary-join
  reference; both yield `{(1,2,3)}`. ✓ Asserted.

### Part D — Default-config bit-identical regression (2 tests)

#### D.1 `default_config_bit_identical_to_w23_baseline`

* Source: `LINEAR_REC_TRIANGLE` (slice-4 anchor, recursive).
* Reference: gate-OFF (binary join only) →
  `wcoj_triangle_dispatch_count() == 0`.
* Default `CompilerConfig::default()` + adaptive runtime gate:
  * Counter == **3** (1 seeding + 1 e1_delta(1,3) variant +
    1 e1_delta(1,4) variant; the last iteration has empty
    delta and skips). Pinned exactly here for the first time —
    the existing slice-4 cert at
    `test_wcoj_recursive_dispatch.rs:649` only asserts `>= 2`.
  * Row set matches binary-join reference exactly.

> **Plan deviation note:** plan iteration 7 stated `counter == 4`
> for this anchor; that was a planning-phase conjecture that did
> not match observed behavior. The probe at fixture-build time
> measured `== 3`; the cert pins that exact value going forward.
> No code path changed — only the test's assertion was corrected
> to match what the slice-4 baseline has always actually emitted.

#### D.2 `record_wcoj_feedback_var_order_none_pair_unchanged`

* Triangle non-recursive + force-WCOJ-on, default compiler
  config (`var_order = None`, no leader rotation).
* One dispatch fires `record_wcoj_feedback`, which calls
  `feedback_pair_from_var_order(slot_rels, None)` → returns the
  canonical `(slot_rels[0], slot_rels[1])` pair with
  `(left_keys, right_keys) = ([1], [0])`.
* Asserted: exactly one `JoinSelectivity` entry on
  canonical(rel_xy, rel_yz) with the `[1]/[0]` keys (or the
  swap-counterpart if `canonical_join_key` flipped them — the
  test handles both orientations).

### Part E — `var_order = Some` rotated-feedback cert (1 test)

`heat_aware_rotated_leader_records_feedback_on_rotated_pair`:

* **Snapshot**: hand-built; 3 rels at card=100; heat = (0.5, 0.5,
  0.0); empty `join_selectivities`. Score:
  rel_xy = 100×3×2 = 600; rel_yz = 600; rel_xz = 100×1×2 = 200.
  argmin = idx 2. Ratio 200/600 = 0.333 ≤ 0.5 → `Some(2)`.
* **Phase 1**: compile under HeatAware. Asserted:
  `var_order == Some(VariableOrder { leader_idx: 2, .. })`.
* **Phase 2**: fresh executor; register 3 rels; put 5-row EDBs;
  pre-condition `executor.stats_snapshot().join_selectivities.is_empty()`
  ✓ asserted.
* **Phase 3**: execute HeatAware plan + force-WCOJ-on. Counter
  advances to 1.
* **Phase 4 — the W2.6 step-5 contract proof**: post-execution
  snapshot has exactly **one** `JoinSelectivity` entry. The
  entry's:
  * `(left_rel, right_rel)` = `canonical(rel_xz, rel_yz)`
    (slot 0 / slot 1 of the rotated leader-2 layout).
  * `left_keys = [1]` AND `right_keys = [1]` — both `[1]`
    because the join variable Z lives at native col 1 in BOTH
    `rel_xz` (native (X, Z)) and `rel_yz` (native (Y, Z)). The
    canonical-rel swap is symmetric in keys here.
  * canonical(rel_xy, rel_yz) (the **pre-W2.6 default-leader
    feedback target**) is **absent** — also asserted, completing
    the negative half of the rotated-feedback contract.

## Cert Test Results

```
cargo test -p xlog-logic --release --lib wcoj_var_ordering
running 17 tests   # 12 pre-existing W2.1 + 5 W2.6 Part A
test result: ok. 17 passed; 0 failed; 0 ignored
  (210 filtered out)

cargo test -p xlog-logic --release --test test_w26_part_b
running 4 tests
test triangle_heat_bias_heat_aware_picks_non_default_leader_card_eq_returns_none ... ok
test triangle_selectivity_bias_heat_aware_picks_not_in_tight_edge ... ok
test cycle4_heat_bias_heat_aware_picks_non_default_leader_card_eq_returns_none ... ok
test cycle4_selectivity_bias_heat_aware_picks_not_in_tight_edge ... ok
test result: ok. 4 passed; 0 failed; 0 ignored

cargo test -p xlog-integration --release --test test_w26_heat_selectivity
running 7 tests
test triangle_real_observed_selectivity_drives_heat_aware_leader_to_idx_2 ... ok
test triangle_real_observed_heat_drives_heat_aware_leader_to_idx_1 ... ok
test triangle_real_observed_heat_with_baseline_drives_heat_aware_leader_to_idx_1 ... ok
test cycle4_real_observed_selectivity_drives_heat_aware_leader_to_idx_2 ... ok
test default_config_bit_identical_to_w23_baseline ... ok
test record_wcoj_feedback_var_order_none_pair_unchanged ... ok
test heat_aware_rotated_leader_records_feedback_on_rotated_pair ... ok
test result: ok. 7 passed; 0 failed; 0 ignored
```

**W2.6 acceptance total: 5 + 4 + 4 + 2 + 1 = 16 tests, 16/16 PASS.**

## Workspace Tally

| Suite | PASS | FAIL | IGN |
|-------|------|------|-----|
| Workspace tests (default features, lib + integration only) — `cargo test --workspace --release --exclude pyxlog --exclude xlog-cuda-tests` | 1875 | 0 | 17 |
| W2.3 trace gate — `cargo test -p xlog-runtime --release --features recursive-stats-trace --test test_w23_recursive_stats` | 10 | 0 | 0 |
| W2.4 cert — `cargo test -p xlog-integration --release --test test_wcoj_record_join_result_feedback` | 3 | 0 | 0 |
| W2.1 cert — `cargo test -p xlog-integration --release --test test_w21_variable_ordering` | 11 | 0 | 0 |
| Slice-4 cert — `cargo test -p xlog-integration --release --test test_wcoj_recursive_dispatch` | 6 | 0 | 0 |
| CUDA certification suite — `cargo test -p xlog-cuda-tests --test certification_suite --release` | 1 (run_full_certification — meta-test running 206 cert sub-tests) | 0 | 0 |
| `cargo fmt --check --all` | clean | — | — |

Slice 1–5 + W2.1 + W2.2 + W2.3 + W2.4 row-set parity preserved
bit-identically under `CompilerConfig::default()`. Confirmed by
running each prior slice's cert suite unchanged.

## Code-Level Changes

| File | Change |
|------|--------|
| `crates/xlog-logic/src/compiler_config.rs` | Add `WcojVarOrderingKind::HeatAware` variant. Default remains `Disabled`. |
| `crates/xlog-logic/src/wcoj_var_ordering.rs` | New `HeatAwareLeaderModel` with locked composite-score formula `card · (1 + 4·heat) · Σ_e 1/max(0.01, sel(e))`. Same threshold gate as W2.1. 5 unit tests for Part A. |
| `crates/xlog-logic/src/promote.rs` | Promoter dispatches on `config.wcoj_variable_ordering` (Disabled / LeaderCardinality / HeatAware) for both `try_promote_triangle` and `try_promote_4cycle`. **Plus** two new normalization helpers (W2.6): `normalize_triangle_to_left_deep` and `normalize_4cycle_to_bushy` invoked before each `try_promote_*` to commutativity-rewrite right-deep / fully-right-deep shapes the lowerer's bushy DP can emit at small cardinalities into the canonical forms the matcher accepts. Conservative: any unrecognized shape returns `None` and falls through unpromoted. Idempotent on already-canonical bodies. |
| `crates/xlog-runtime/src/executor/wcoj_dispatch.rs` | New module-scope helper `feedback_pair_from_var_order(slot_rels, var_order) -> Option<(RelId, RelId, Vec<usize>, Vec<usize>)>`. `record_wcoj_feedback` now takes `var_order: Option<&VariableOrder>` and routes feedback through this helper — `var_order = None` returns the canonical pre-W2.6 W2.4 pair (bit-identical); `Some(_)` returns the rotated pair + correct `[1]/[1]` keys for triangle non-default leaders or `[1]/[0]` for 4-cycle (rotation-only). |
| `crates/xlog-logic/tests/test_w26_part_b.rs` | NEW. 4 hand-built-snapshot tests (Part B). |
| `crates/xlog-integration/tests/test_w26_heat_selectivity.rs` | NEW. 7 real-runtime tests (Part C × 4 + D × 2 + E × 1). |

## Decision Mapping

| Decision | Rationale |
|----------|-----------|
| Heat weight = `4.0` (locked) | With W2.1 default threshold 0.5, gate fires when `min/default ≤ 0.5`. With cards equal + heat `h` on hot rel, ratio = `1 / (1 + 4h)`. For ratio ≤ 0.5 → `h ≥ 0.25` (~3 `record_access` calls). Lower weights would require unrealistically many accesses to flip the leader. |
| `NO_OBSERVED_SEL = 1.0` | An edge with no observed `JoinSelectivity` record contributes penalty 1 to the sum, treating "unknown" as "no useful filter info" rather than "definitely no filter". |
| `SEL_FLOOR = 0.01` | Used in `1/max(0.01, sel)` to bound the per-edge penalty at 100×. Tightly observed edges (`sel < 0.01`) cap at this value rather than spiking to the divide-by-zero limit. |
| Key-validation in `observed_sel_or_one` | When `StatsManager::canonical_join_key` swaps the rel order, the candidate keys must be swapped correspondingly. On mismatch (stored keys ≠ candidate keys after swap), return `NO_OBSERVED_SEL` — the model treats key-mismatched records as "wrong topology, ignore". |
| `card_of` returns `None` for `cardinality == 0` | Same safety floor as `LeaderCardinalityModel` — partial stats degrade to default-leader rather than mis-picking. |
| Triangle non-default-leader feedback uses `[1]/[1]` keys | For triangle with leader idx 1 or 2: slot 0 is the leader rel native, slot 1 is a swapped 2-col view of another rel. The kernel's swap reshapes the slot-1 *view* but does NOT change the underlying relation's column indexing. Z-shared edges in canonical layout join on col 1 of both rels → `[1]/[1]`. |
| 4-cycle non-default-leader feedback uses `[1]/[0]` keys | 4-cycle is rotation-only (no slot-swaps in the locked permutation table). The (slot 0, slot 1) edge in the rotated layout is always `[1]/[0]` regardless of leader. |
| `feedback_pair_from_var_order` returns `Option` | None indicates "shape we don't have a feedback table for" — the dispatcher then skips the EMA write. Conservative: never write a record under uncertainty. |

## Process Rule Compliance

* Process rule #1: this slice does **not** self-mark W2.6 DONE.
  The closure proposal below describes the OPEN → DONE
  transition; the user reviews and explicitly approves; a
  separate follow-up commit applies the board update.
* Process rule #2: every commit references W2.6.
* Process rule #3: plan header opens with "Closes W2.6 only."
* Process rule #5: no `v0.6.6` references in this slice.
* Process rule #6: no push, no tag.

## Plan-Iteration-7 Adjustments (Implemented)

Three plan-locked values shifted during execution. Each is
documented here with the implemented resolution. None changes
a contract the plan locked — the leader-pick contract, ratio
threshold, and row-set parity all hold under the executed
values.

### Adjustment 1 — Slice-1 promoter normalizers (option B from prior iteration)

**Plan iteration 7 specified**: Part C.1 / C.3 / E.1 seed cards
at 5 (matching 5-row EDBs); EMA selectivity converges to 0.270;
assertion band `[0.25, 0.30]`.

**Issue at execution**: at cards=5/5/5 the lowerer's bushy DP
planner picks non-canonical lowered shapes that slice-1's
matchers reject:
* Triangle: right-deep `Project(Join(Scan, Join(Scan, Scan)))`
  vs. matcher's left-deep `Project(Join(Join, Scan))`.
* 4-cycle: fully-right-deep
  `Project(Join(Scan, Join(Scan, Join(Scan, Scan))))` vs.
  matcher's bushy `Project(Join(Join, Join))`.
Without a fix, `HeatAware` silently produces `var_order = None`
and the closure-board acceptance line would not hold for any
small-table snapshot recompile flow.

**Implemented fix** (commit `07eaf5c3`):
`crates/xlog-logic/src/promote.rs` gains two normalization
helpers invoked before each `try_promote_*`:

* `normalize_triangle_to_left_deep` — detects right-deep and
  commutativity-rewrites: swap outer Join's left/right + their
  keys; remap Project columns via `(k + 4) % 6` for the swapped
  output layout. Inner-Join keys unchanged. Triangle is
  symmetric under inner-join commutativity, so the rewrite is
  semantics-preserving.
* `normalize_4cycle_to_bushy` — detects the rotation-only
  canonical-cycle fully-right-deep pattern (every inner Join
  has `[1]/[0]` keys; outer has `[0,1]/[5,0]`) and rebuilds as
  bushy `Join(Join(R0, R1, [1], [0]), Join(R2, R3, [1], [0]),
  [3, 0], [0, 3])`. Output column layout is preserved across
  the rewrite — Project columns pass through unchanged.

Both helpers are conservative: any shape they don't recognize
returns `None` and the body falls through unpromoted (pre-W2.6
behavior). Already-canonical / bushy bodies bypass the rewrite
path entirely.

**Result**: plan-locked card=5 fixture restored across Part C.
Math: input_rows = 5×5 = 25, observed_sel = 1/25 = 0.04, EMA
after 4 dispatches converges to 0.270, band `[0.25, 0.30]`.
Promoter matches; HeatAware picks `Some(2)`; production small-
table flow now works.

### Adjustment 2 — New Part C.4 cert (option D from prior iteration)

**Plan iteration 7 specified**: Part C.2 with combined
`dummy_e1 + tri` Phase A → expected `e2.heat ≈ 0.1` baseline.

**Issue at execution**: the binary-join `tri` rule traverses
`node_dispatch.rs:343` which calls `record_join_result` after
EVERY hash join. The combined-source Phase A would write a
`(rel_xy, rel_yz)` selectivity entry (`sel ≈ 0.712` after one
EMA step) that perturbs the heat-only signal — HeatAware would
score `argmin = idx 2` rather than the plan's `idx 1` because
the selectivity penalty on (e1, e2) demotes them more than the
heat factor demotes e1.

**Implemented fix** (commit `07eaf5c3`):
* Existing C.2 (`triangle_real_observed_heat_drives_heat_aware_leader_to_idx_1`)
  retained as the **zero cold-baseline** cert: heater-only
  `dummy_e1` source × 11 calls. Empty `join_selectivities`
  asserted as a pre-condition.
* **New C.4** (`triangle_real_observed_heat_with_baseline_drives_heat_aware_leader_to_idx_1`)
  covers the **non-zero cold-baseline** case the plan
  intended. Uses a join-free triple-dummy source
  `dummy_e1 + dummy_e2 + dummy_e3` for Phase A so each rel
  gets a baseline scan giving `e1=e2=e3.heat = 0.1` without
  any `record_join_result` side effect (no joins → no hash-join
  path, no auto-recorded selectivity). Phase B heater × 11
  advances `e1.heat` to ≈ 0.7176. HeatAware picks `Some(1)`
  with score(e1) ≈ 38.70 vs score(e2)=score(e3) = 14;
  ratio 14/38.70 ≈ 0.362 ≤ 0.5.

The user's direction explicitly required the triple-dummy
approach (no selectivity mutation, no snapshot subtraction) —
the implementation matches that direction.

**Result**: total Part C tests = 4 (was 3). W2.6 acceptance
total = 16 (was 15).

### Adjustment 3 — Part D.1 counter `4 → 3`

| Plan iteration 7 | Executed value | Reason |
|------------------|----------------|--------|
| `wcoj_triangle_dispatch_count() == 4` for the `LINEAR_REC_TRIANGLE` slice-4 anchor under default config | counter `== 3` | Empirically measured via probe; the existing slice-4 cert at `test_wcoj_recursive_dispatch.rs:649` only asserts `>= 2`. The actual baseline is 1 seeding + 1 e1_delta(1,3) variant + 1 e1_delta(1,4) variant = 3 (the last iteration has empty delta and skips dispatch). No code path changed — only the cert's assertion is corrected to the actual measured baseline. |

## Closure Board Update Proposal

After explicit user "mark W2.6 DONE" approval, a follow-up
commit applies:

* `docs/v065-closure-board.md` — W2.6 status `OPEN → DONE`,
  status tally updated (DONE: 8 → 9; OPEN: 9 → 8 — verify
  current counts at apply-time).
* `docs/v065-closure-board.md` "Completed" section gets a W2.6
  entry referencing commits:
  * `d3ef4cda` — plan iteration 7 (approved).
  * `c51e07bb` — HeatAwareLeaderModel + var_order-aware W2.4
    feedback (steps 1-6).
  * `7e76b3dd` — 15 acceptance tests (step 7).
  * `07eaf5c3` — slice-1 promoter normalizers + Part C.4 cert
    (step 9 follow-up; resolves both plan deviations the user
    flagged).
  * (this commit, evidence README — step 9).
* FF-merge `feat/w26-heat-selectivity-variable-ordering` into
  `main`. No tag, no push (per process rule #6).
