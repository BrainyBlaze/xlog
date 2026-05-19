# W2.4 Plan — `record_join_result` Feedback from WCOJ Dispatch

**Closes W2.4 only.** No other v0.6.5 closure-board items move on
this slice. No default flip (W2.5 stays BLOCKED). No
variable-ordering work (W2.1 untouched). No selectivity_pass
work (W2.2 untouched). No recursive-arm stats integration (W2.3
untouched). No new ROADMAP items added.

**Date:** 2026-05-04
**Branch (proposed):** `feat/w24-record-join-result-feedback`
**Worktree (proposed):** `.worktrees/w24-record-join-result-feedback`
**Base:** `main` at `55702bb8` (W1.1 commit).
**Board entry:** `docs/v065-closure-board.md` Wave 2, W2.4.

## Goal

Wire successful WCOJ dispatches (triangle + 4-cycle) to call
`xlog_stats::StatsManager::record_join_result(...)` so observed
selectivity from the dispatch's output flows back into the
stats cache. The slice 5 cardinality cost model already consumes
stats but does not write any back; W2.4 closes that loop for
the load-bearing dispatch path.

## In Scope

* Add a single call site in `wcoj_dispatch.rs` for the
  triangle-success path, post-`run_wcoj_triangle_pipeline`,
  reading the output buffer's row count and the input
  cardinalities, then calling `stats.record_join_result(...)`.
* Add the analogous call site for the 4-cycle-success path,
  post-`run_wcoj_4cycle_pipeline`.
* The recording uses the **inner-join slot pair**
  (`slot_rels[0]`, `slot_rels[1]`) with keys `[1] / [0]` — the
  same pair the cardinality model's `binary_est` reads via
  `estimate_join_cardinality`. This keeps the writer ↔ reader
  pair coherent (the EMA tightens the very estimate the model
  consults).
* Skip the recording when any input slot has missing or zero
  cardinality (the `populated_cards` analog from slice 5 — no
  stats means the EMA pollutes the cache with default-derived
  noise).
* Cert that proves the dispatch path **actually calls**
  `record_join_result`, not manual `StatsManager` mutation.

## Not In Scope

* No new public API on `StatsManager`. The `record_join_result`
  signature is reused as-is.
* No new public API on `Executor`. The new call site uses
  `&mut self` already in scope at the dispatch path.
* No threshold-tuning, no default-flip, no benchmark
  evidence. (W2.5 BLOCKED on W5.2 evidence; this slice does
  not pre-empt either.)
* No recursive-arm per-iteration update beyond what the
  existing variant loop already does — that's W2.3.
* No `selectivity_pass` change — that's W2.2.

## Mapping the WCOJ output to a Binary Selectivity

Both the WCOJ triangle and 4-cycle kernels skip the binary
intermediate that `estimate_join_cardinality` models. We
therefore record an **upper-bounded selectivity** for the
inner-pair:

```
input_rows  = card(slot_rels[0]) * card(slot_rels[1])
output_rows = wcoj_output_row_count   // triangle / 4-cycle output
```

The triangle / 4-cycle output is a strict subset of the
inner-join intermediate (the third / additional atoms further
filter it). So `output_rows ≤ true_intermediate_rows`, which
means the recorded selectivity is **≤** the true binary
selectivity. The cardinality model uses the recorded
selectivity to compute `binary_est`, and a too-low selectivity
gives a too-low `binary_est`, which makes WCOJ less likely to
dispatch on the next call — which is the correct conservative
direction (don't over-claim the kernel's win).

This mapping is documented in code comments at the call sites
and in the cert's expected-behavior block.

## Stats Preservation Across the Cert's Two Runs

The cert needs to observe stats state both **before** and
**after** a WCOJ dispatch. `StatsManager` lives on `Executor`
and persists across `execute_plan` calls (it is not cleared
between runs), so the cert uses the **same executor** across
both runs:

```rust
let mut executor = build_executor(...);
seed_runtime_stats(&mut executor, ...);

// Run 1: cold (no prior WCOJ feedback).
let pre_sel = executor.stats().get_join_selectivity(rel_a, rel_b);
assert!(pre_sel.is_none(), "no feedback yet");
executor.execute_plan(&plan)?;
assert!(executor.wcoj_triangle_dispatch_count() >= 1);
let post_run1_sel = executor.stats().get_join_selectivity(rel_a, rel_b);
assert!(post_run1_sel.is_some(), "WCOJ dispatch must record");

// Run 2: warm.
executor.execute_plan(&plan)?;
let post_run2_sel = executor.stats().get_join_selectivity(rel_a, rel_b);
assert!(post_run2_sel.is_some());
```

**No snapshot/restore helper is required.** `StatsManager`
already exposes `snapshot()` / `merge_snapshot()` if a future
cert needs cross-executor preservation; this cert doesn't.

## Acceptance Gate (corrected per review)

W2.4 records feedback. It does **not** require any dispatch
decision change between runs — a decision change is fine if
it happens but is NOT the gate.

The cert is in
`crates/xlog-integration/tests/test_wcoj_record_join_result_feedback.rs`
and asserts **all three** properties:

1. **The dispatch path actually calls `record_join_result`.**
   Pre-dispatch: `executor.stats().get_join_selectivity(slot_rels[0], slot_rels[1])`
   returns `None`. Post-dispatch (after `execute_plan` produces
   `wcoj_triangle_dispatch_count() >= 1`):
   `get_join_selectivity` returns `Some(_)`. The transition
   from `None → Some` is what proves the dispatch path made
   the call (not test-side `update_cardinality`, which doesn't
   touch the selectivity cache).
2. **`binary_est` differs between runs.** The cert reads
   `binary_est` directly via
   `executor.stats().estimate_join_cardinality(slot_rels[0], slot_rels[1], &[1], &[0])`
   — same call the cardinality model uses, no new public API,
   no test-only peek field. Assert
   `binary_est_run2 != binary_est_run1` after two consecutive
   `execute_plan` calls. (No requirement that the dispatch
   decision changes — only that the underlying estimate moved.)
3. **Row-set parity remains unchanged across runs.**
   `download_triples(executor.store().get("tri"))` after run 1
   equals `download_triples(...)` after run 2 (recursive store
   converged to the same fixpoint).

The 4-cycle counterpart cert is analogous, gated on
`wcoj_4cycle_dispatch_count` and `download_quads`.

## Step Plan

### Step 1 — Helpers at the dispatch sites

Two `pub(super)` helpers on `Executor`:

```rust
/// Read the WCOJ output buffer's logical row count, returning
/// `None` when the cache isn't populated. **Never returns
/// `Some(0)` for an unknown row count** — only for an
/// observed-empty output.
fn wcoj_output_rows(buf: &CudaBuffer) -> Option<u64> {
    buf.cached_row_count()
}

/// Wire successful WCOJ dispatches into `StatsManager`.
fn record_wcoj_feedback(
    &mut self,
    slot_rels: &[RelId],
    output_rows: Option<u64>,
)
```

`record_wcoj_feedback` internally:

- Returns early if `slot_rels.len() < 2`.
- Returns early if `output_rows.is_none()` — unknown logical
  count must not become a `0` selectivity record.
- Looks up `card_a = stats.get_relation_stats(slot_rels[0]).map(|s| s.cardinality).filter(|c| *c > 0)`
  and `card_b` analogously. If either is `None`, returns
  early — the `populated_cards` analog from slice 5 keeps
  the EMA cache clean.
- Computes `input_rows = card_a.saturating_mul(card_b)`.
- Calls `record_join_result` with **owned `Vec<usize>` keys**
  (the `StatsManager` API takes `Vec<usize>`, not slices):
  ```rust
  self.stats.record_join_result(
      slot_rels[0],
      slot_rels[1],
      vec![1],       // inner-pair left_keys
      vec![0],       // inner-pair right_keys
      input_rows,
      output_rows.expect("checked above"),
  );
  ```

Recording an observed-empty output (`Some(0)`) is correct — the
EMA tightens future estimates toward zero, which makes WCOJ
less likely on the same inputs next call (the kernel produced
nothing). Recording an *unknown* output (`None`) is incorrect
and forbidden by the early-return.

The helpers are `pub(super)` so the triangle / 4-cycle paths
can both call them without exposing the surface outside the
executor module.

### Step 2 — Triangle call site

In `wcoj_dispatch.rs` at the `try_dispatch_wcoj_triangle_on_body`
success-arm (post-`run_wcoj_triangle_pipeline`'s `Ok(buf)`):

* Read `output_rows = Self::wcoj_output_rows(&buf)` (returns
  `Option<u64>`).
* Build slot rels: `[matched.rel_xy, matched.rel_yz, matched.rel_xz]`.
* Call `self.record_wcoj_feedback(&slot_rels, output_rows)`.
  Inputs cardinalities + key vectors are derived inside the
  helper so each call site stays a single line.
* The call happens BEFORE the counter increment so that any
  helper panic does not advance the counter. The helper is
  panic-free in practice (early-returns on missing data).

### Step 3 — 4-cycle call site

Analogous to Step 2 in `try_dispatch_wcoj_4cycle_on_body`:
slot rels `[matched.rel_e1, matched.rel_e2, matched.rel_e3, matched.rel_e4]`,
output rows from the 4-cycle output buffer. Same helper. Inner
pair recording (`slot_rels[0]`, `slot_rels[1]`, keys `vec![1] / vec![0]`)
is handled inside `record_wcoj_feedback`.

### Step 4 — Cert

`crates/xlog-integration/tests/test_wcoj_record_join_result_feedback.rs`:

* `triangle_dispatch_records_join_result_into_stats_manager` —
  triangle fixture from the slice 5 large-cards fixture (so
  the cardinality model is the dispatcher and the cardinality
  path is exercised). Cert all three acceptance properties.
* `cycle4_dispatch_records_join_result_into_stats_manager` —
  4-cycle fixture, analogous.
* `wcoj_dispatch_does_not_record_when_input_cards_missing` —
  unseeded stats fixture. Pre + post: `get_join_selectivity`
  stays `None`; counter advances (delegate-on-missing keeps
  dispatch via skew model); row set parity preserved.

### Step 5 — Workspace gate

* `cargo fmt --all -- --check`
* `cargo test -p xlog-integration --release --test test_wcoj_record_join_result_feedback`
* Slice 1–5 regression preserved bit-identical:
  - `cargo test -p xlog-integration --release --test test_wcoj_executor_wiring --test test_wcoj_4cycle_executor_wiring --test test_wcoj_recursive_dispatch --test test_wcoj_cardinality_cost_model`
  - `cargo test -p xlog-runtime --lib --release wcoj_cost_model`
  - `cargo test -p xlog-cuda --release` WCOJ-filtered.
  - CUDA cert: `cargo test -p xlog-cuda-tests --test certification_suite --release`
* Evidence file at
  `docs/evidence/2026-05-04-w24-record-join-result-feedback/README.md`.

### Step 6 — End-of-slice closure update

Per process rule #4: this slice's commit-of-evidence proposes
the board update for W2.4 (OPEN → DONE), but does NOT mark
DONE itself. Per process rule #1, the user reviews the slice
and explicitly approves "mark W2.4 DONE"; a separate follow-up
commit then applies the board update.

### Step 7 — FF-merge to local main

* No push, no tag. Working tree clean. Same FF-merge pattern
  as prior slices.

## Risk & Open Questions

* **Q1 — Reading `binary_est` for property #2.** Cert reads it
  directly via `executor.stats().estimate_join_cardinality(...)`
  — the same path the cardinality model uses. No new public
  API, no test-only field. Confirmed approved.
* **Q2 — Skew classifier path.** The slice 1–2 default
  (`SkewClassifierCostModel`) does NOT use cardinality; the
  feedback wired in W2.4 doesn't change the skew model's
  decision either way. Cert pinning behavior under the skew
  default is therefore "stats updated, decision unchanged" —
  asserted by `wcoj_dispatch_does_not_record_when_input_cards_missing`
  (third cert).
* **Q3 — Recording on `Ok(buf)` with `output_rows == 0`.**
  An empty triangle output (e.g., classifier dispatched but
  the kernel found no triangles) would record selectivity
  `0`. This is correct: zero triangles in the output is the
  observed selectivity. The EMA tightens future estimates
  toward zero, which makes WCOJ less likely on the same
  inputs next call — also correct (kernel didn't pay off).
  Documented in step 1's helper comment.

## Provenance

- Closure board: `docs/v065-closure-board.md` Wave 2, W2.4.
- Internal commitment origin: slice 5 plan §"Out-of-Slice
  (Deferred)" (now `OPEN` on the board, not deferred).
- Reader/writer coherence: pairs with the `binary_est` reader
  in `crates/xlog-runtime/src/executor/wcoj_cost_model.rs`'s
  `CardinalityAwareCostModel::should_dispatch_triangle` /
  `should_dispatch_4cycle` (slice 5 step 2).
