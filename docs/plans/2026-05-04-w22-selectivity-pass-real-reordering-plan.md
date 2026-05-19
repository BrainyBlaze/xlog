# W2.2 Plan — Real `selectivity_pass` Join Reordering

**Closes W2.2.** v0.6.5 ships triangle + 4-cycle WCOJ; this
slice covers BOTH so the closure board can honestly mark W2.2
DONE. No W2.1 variable-ordering work. No W2.3 recursive-arm
stats. No W2.5 default flip. No new ROADMAP items added.

**Date:** 2026-05-04
**Branch (proposed):** `feat/w22-selectivity-pass-real-reordering`
**Worktree (proposed):** `.worktrees/w22-selectivity-pass-real-reordering`
**Base:** `main` at `96ebf8bc` (W2.4 closure-board commit).
**Board entry:** `docs/v065-closure-board.md` Wave 2, W2.2.

## Goal

Replace slice 3's no-op `xlog_logic::optimizer::selectivity_pass`
with a real selectivity-driven join reordering pass for the
canonical lowered **triangle and 4-cycle** bodies. The pass
consults `xlog_stats::StatsManager::estimate_join_cardinality`
with **pair-derived join keys** from each candidate's
shared-variable mapping, then rewrites the body so the
smallest-cost pairing is materialized first — feeding the
slice 1 / slice 2 promoter (extended in this slice — see
step 2a) or the binary-join executor a body whose
intermediate is the cheapest valid choice.

## In Scope

* Update `selectivity_pass::run` signature to accept the
  predicate-name → RelId map (mirrors slice 4's
  `promote_multiway` signature change).
* **Triangle reordering**: recognize the canonical lowered
  shape `Project { Join { Join { Scan(A), Scan(B) }, Scan(C) } }`
  (slice 1 keys, output columns `[0, 1, 3]`). For a recognized
  body, compute three candidate inner-pair selectivities and
  pick the smallest.
* **4-cycle reordering**: recognize the canonical lowered
  bushy shape `Project { Join { Join(A,B), Join(C,D) } }`
  (slice 2 keys, output columns `[0, 1, 3, 5]`). For a
  recognized body, compute the cost of the **two valid
  bushy pairings** (those whose left and right pairs each
  share a variable) and pick the smaller. The two pairings
  for edges `WX, XY, YZ, ZW` are:
  - **(WX⋈XY on X) + (YZ⋈ZW on Z)** — left pair shares X,
    right pair shares Z. Outer join on `(W, Y)`.
  - **(XY⋈YZ on Y) + (ZW⋈WX on W)** — left pair shares Y,
    right pair shares W. Outer join on `(X, Z)`.
  Opposite-edge pairings (WX⋈YZ, XY⋈ZW) share NO variable
  and are NOT valid candidates — they would cross-product
  rather than join. The pass enumerates exactly these two
  valid groupings.
  Cost formula: `est_left + est_right` for each grouping
  (sum of the two binary intermediates). Tie → keep the
  optimizer's existing order (deterministic no-op).
* **Pair-derived join keys**, NOT fixed `[1] / [0]`. The keys
  for each candidate pair are computed from the rule's
  variable mapping. For triangle:
  - `e1(X,Y) ⋈ e2(Y,Z)` shares variable Y → keys `[1] / [0]`.
  - `e2(Y,Z) ⋈ e3(X,Z)` shares variable Z → keys `[1] / [1]`.
  - `e1(X,Y) ⋈ e3(X,Z)` shares variable X → keys `[0] / [0]`.
  For 4-cycle:
  - `WX(0,1) ⋈ XY(0,1)` shares X → keys `[1] / [0]`.
  - `YZ(0,1) ⋈ ZW(0,1)` shares Z → keys `[1] / [0]`.
  - `XY(0,1) ⋈ YZ(0,1)` shares Y → keys `[1] / [0]`.
  - `ZW(0,1) ⋈ WX(0,1)` shares W → keys `[1] / [0]`.
* **Semantic slot preservation**. WCOJ kernels expect
  semantic slots (triangle: XY, YZ, XZ; 4-cycle: WX, XY, YZ,
  ZW), not arbitrary positional rotation. The pass MUST
  preserve the slot identity — so after reordering, the
  promoter captures the same semantic atoms at their
  semantic positions. Concretely: the rewriter only changes
  WHICH binary join is computed first; the final
  MultiWayJoin's `slot_vars` and the kernel's per-slot
  expectations stay correct because the semantic
  variable-to-position mapping is preserved by the rewriter
  (see "Mechanism" below).
* Missing-stats safety floor: if **any** body Scan's RelId
  has no entry in `StatsManager` (or `cardinality == 0`),
  leave the body unchanged. The pass is opt-in on populated
  stats; recursive deltas / freshly-uploaded relations stay
  on the optimizer's default order.

## Not In Scope

* General-arity (k > 4) reordering. Folds into W3.2's
  general-arity template work, not W2.2.
* Right-deep input handling. selectivity_pass receives the
  optimizer's output, which is left-deep for triangle (and
  bushy for 4-cycle) in current code paths. If a future
  optimizer change emits other shapes, that's a separate
  slice's input.
* Variable-ordering cost model (W2.1). selectivity_pass
  reorders the binary-join tree; W2.1 separately decides slot
  ordering inside the WCOJ kernel.
* Selectivity feedback into variable ordering (W2.6 — BLOCKED
  on W2.1 + W2.4).
* Default-flip (W2.5 — BLOCKED).
* Recursive-arm per-iteration update (W2.3).
* Shape changes to the lowered IR (no new RIR variants).

## Mechanism

### Triangle reordering invariant

Slice 1's canonical lowered triangle:

```text
Project { input: outer, columns: [Column(0), Column(1), Column(3)] }
where outer = Join {
    left:  inner,
    right: Scan(C),         // 3rd atom
    left_keys:  [0, 3],
    right_keys: [0, 1],
    join_type: Inner,
}
where inner = Join {
    left:  Scan(A),         // 1st atom
    right: Scan(B),         // 2nd atom
    left_keys:  [1],
    right_keys: [0],
    join_type: Inner,
}
```

For triangle `t(X,Y,Z) :- e1(X,Y), e2(Y,Z), e3(X,Z)` the
"natural" ordering puts e1+e2 inner (joining on Y), with e3
filtering on X+Z outer.

The three valid orderings of which atom-pair is inner each
correspond to a different intermediate cardinality:
  * `(A,B)` inner — current canonical; intermediate joins
    on the variable shared between A and B.
  * `(B,C)` inner — joins on the variable shared between B and
    C; A becomes the outer's right scan.
  * `(A,C)` inner — joins on the variable shared between A and
    C; B becomes the outer's right scan.

Each rewrite preserves the rule's semantics (the head is
always `(X, Y, Z)`); only the join order changes.

### Pair-selectivity lookup with pair-derived keys

For each candidate pair, the join keys are derived from the
rule's variable mapping (NOT a fixed `[1] / [0]`). For
triangle:

```rust
// Three candidate inner pairs. Each is keyed on the variable
// shared between the two atoms.
let est_ab = stats.estimate_join_cardinality(rel_a, rel_b, &[1], &[0]);  // Y shared
let est_bc = stats.estimate_join_cardinality(rel_b, rel_c, &[1], &[1]);  // Z shared
let est_ac = stats.estimate_join_cardinality(rel_a, rel_c, &[0], &[0]);  // X shared
```

For 4-cycle, the same principle: keys derived from each
pair's shared variable per the lowered shape's
variable-to-column mapping.

`StatsManager::estimate_join_cardinality` returns `u64`. Pick
the pair with the smallest estimate. Ties keep the optimizer's
current order (deterministic no-op).

### Semantic-slot preservation

After reordering, the promoter still emits a MultiWayJoin
whose `slot_vars` are the canonical semantic slot
positions for the shape:
* Triangle: slot 0 = XY-edge, slot 1 = YZ-edge, slot 2 = XZ-edge.
* 4-cycle: slot 0 = WX, slot 1 = XY, slot 2 = YZ, slot 3 = ZW.

The reordering changes WHICH binary intermediate is computed
first under the binary-join fallback, but it does NOT change
the final semantic role of each input. The rewriter
re-emits the canonical lowered shape with the smallest pair
at the inner Join, and re-derives the variable-to-column
projections so the head's `(X, Y, Z)` (or
`(W, X, Y, Z)`) output is preserved at the same Project
columns. The promoter captures positional Scan RelIds and
emits semantic slot_vars per the shape — those slot_vars are
shape-fixed, not order-dependent — so the WCOJ kernel
continues to receive XY/YZ/XZ (or WX/XY/YZ/ZW) buffers in
the right semantic positions regardless of which pair the
selectivity_pass chose to materialize first.

This is the load-bearing invariant; the cert (Part C below)
gates on it directly via force-WCOJ dispatch + counter.

### Rewrite rule

If the smallest-est pair is already the inner pair, the body is
unchanged (most common case once W2.4's feedback loop has
converged).

If a different pair `(p, q)` is smallest:

* **All Join keys are recomputed from the variable mapping**,
  not preserved verbatim. The chosen inner pair shares some
  variable; that variable's column index in each scan
  determines the inner-Join keys (e.g. XY⋈YZ → `[1]/[0]`,
  YZ⋈XZ → `[1]/[1]`, XY⋈XZ → `[0]/[0]`). The outer Join's
  keys are likewise recomputed from the third atom's variable
  mapping against the inner output's column layout. NO key
  shape is reused from the optimizer's input.
* Project columns at the head are recomputed so the rule's
  semantic head `(X, Y, Z)` (or `(W, X, Y, Z)` for 4-cycle)
  is still produced at the canonical output positions
  (`[0, 1, 3]` for triangle, `[0, 1, 3, 5]` for 4-cycle). The
  rewriter computes the column indices algorithmically from
  the variable-to-column mapping of the chosen ordering.

A helper `rewrite_triangle_with_inner(body, inner_pair, bindings) -> Option<RirNode>`
handles the column / key recomputation. Same shape applies to
`rewrite_4cycle_with_inner_grouping(body, grouping, bindings)`.
If the recomputation detects an unsupported edge case, returns
`None` and the body stays unchanged (safety floor).

### Promoter must be extended (load-bearing for W2.2)

**The current promoter cannot survive non-default reordered
shapes.** `try_promote_triangle` (`crates/xlog-logic/src/promote.rs:176`)
and `try_promote_4cycle` (`promote.rs:277`) are positional
matchers:

* Triangle: assumes `inner.left = XY`, `inner.right = YZ`,
  `outer.right = XZ`, with hardcoded inner keys `[1]/[0]` and
  outer keys `[0,3]/[0,1]`.
* 4-cycle: assumes `outer.left = (WX⋈XY)`, `outer.right =
  (YZ⋈ZW)` with hardcoded keys throughout.

If selectivity_pass moves scans to a non-default position
(e.g., picks `(YZ, XZ)` as the triangle's inner pair → inner
keys `[1]/[1]`), the existing matcher's hardcoded key check
fails and promotion silently bails. WCOJ dispatch counter
would drop to 0 on reordered bodies, violating Part C of the
acceptance gate.

**W2.2 therefore extends the promoter** to recognize the
canonical *semantic* shape regardless of which atom is at
which position. Concretely:

* The matcher accepts the three valid inner-key combinations
  for triangle (`[1]/[0]`, `[1]/[1]`, `[0]/[0]`) and the
  corresponding outer-key combinations.
* For each accepted shape, the matcher computes which atom
  is the XY-edge / YZ-edge / XZ-edge based on the variable
  mapping inferred from the keys.
* The emitted `MultiWayJoin`'s `slot_vars` and `inputs` are
  arranged in semantic order (slot 0 = XY, slot 1 = YZ,
  slot 2 = XZ for triangle; slot 0 = WX, slot 1 = XY, slot
  2 = YZ, slot 3 = ZW for 4-cycle) regardless of the
  positional layout in the rewritten body.

This is a real scope expansion vs. the original W2.2 plan
header. It is REQUIRED for W2.2 to honestly close because
selectivity_pass without promoter-extension breaks WCOJ
integration. Both pieces ship in the same slice.

The cert pins the invariant via Part C (force-WCOJ): a
reordered body must still dispatch the WCOJ kernel
(`counter ≥ 1`) and the kernel's output must equal the
gate-off binary-join reference row set.

## Acceptance Gate

Three parts:

### Part A — Compile-time RIR shape (xlog-logic unit test)

In `crates/xlog-logic/src/optimizer.rs::tests` (or a new
`selectivity_pass_tests` module):

* Build a canonical triangle body fixture with three Scans
  for RelIds A=1, B=2, C=3 (variables X,Y,Z mapped to
  e1=XY, e2=YZ, e3=XZ).
* Build `rel_ids = {"e1": 1, "e2": 2, "e3": 3}`.
* Build `StatsSnapshot` **#1**: cards make pair (A,B) the
  smallest est.
* Build `StatsSnapshot` **#2**: cards make pair (B,C) the
  smallest est.
* Run `selectivity_pass::run(plan, &stats, &rel_ids)`
  (signature: `&StatsManager`, no `Arc`, no lock) on the same
  body under each snapshot. Capture the
  resulting body's inner-Join's left + right Scan RelIds.
* Assert: snapshot 1 inner ∈ {(A,B), (B,A)}.
* Assert: snapshot 2 inner ∈ {(B,C), (C,B)}.
* Assert the two outputs differ — proves stats drive the
  choice. Deterministic canonicalization CANNOT pass this
  gate.

4-cycle counterpart: analogous, with two snapshots driving
two different inner-pair groupings on the bushy shape.

Negative-direction certs:
* `selectivity_pass_skips_when_any_card_missing` — one slot
  has no `StatsManager` entry (or `cardinality == 0`); body
  unchanged.
* `selectivity_pass_skips_when_already_optimal` — snapshot
  makes the existing inner pair the smallest; body unchanged
  (no-op when the optimizer already chose the best order).

### Part B — End-to-end row-set parity (xlog-integration)

In
`crates/xlog-integration/tests/test_selectivity_pass_reordering.rs`:

* Use `Compiler::compile_with_stats_snapshot(source, Some(&snapshot))`
  — the API already exists at
  `crates/xlog-logic/src/compile.rs:109`. NO new public API
  is added by this slice unless the implementation step
  proves that exact path is insufficient.
* Compile + execute the same triangle program twice on
  separate executors:
  * Run 1: snapshot 1 → small inner pair = (e1, e2).
  * Run 2: snapshot 2 → small inner pair = (e2, e3).
* Assert run 1's `tri` row set equals run 2's `tri` row set
  via `download_triples`. (Reordering preserves semantics;
  only execution order changes.)
* 4-cycle counterpart with `download_quads`.

### Part C — WCOJ dispatch preservation (xlog-integration)

This is the load-bearing slot-semantics gate. After
selectivity_pass reorders, the WCOJ kernel must still
dispatch successfully and produce the correct row set.

In the same `test_selectivity_pass_reordering.rs`:

* `selectivity_pass_reordered_triangle_still_dispatches_wcoj` —
  Force-WCOJ on the triangle dispatch
  (`with_wcoj_triangle_dispatch(Some(true))`). Use a
  `StatsSnapshot` that makes the reorderer pick a non-default
  inner pair. Assert:
  * `wcoj_triangle_dispatch_count() >= 1` (kernel actually
    fired despite reordering).
  * `download_triples` row set equals the gate-off
    binary-join reference row set on the same fixture.
* `selectivity_pass_reordered_4cycle_still_dispatches_wcoj` —
  4-cycle counterpart with `with_wcoj_4cycle_dispatch(Some(true))`
  and `download_quads`.

If the kernel dispatches with the wrong slot identities, the
row set will diverge from binary-join (the kernel reads
buffer-by-RelId and the WCOJ count/materialize logic depends
on the variable signature being consistent with slot_vars).
This cert catches that failure mode directly.

## Step Plan

### Step 1 — Signature change

* Update `selectivity_pass::run` from
  `run(plan: &mut ExecutionPlan, _stats: &StatsManager)`
  (verified at `crates/xlog-logic/src/optimizer.rs:1241`)
  to
  `run(plan: &mut ExecutionPlan, stats: &StatsManager, rel_ids: &HashMap<String, RelId>)`.
* `&StatsManager` (no `Arc`, no lock). The pass calls
  `stats.estimate_join_cardinality(...)` directly on the
  immutable handle.
* Update `compile.rs` caller to pass `self.lowerer.rel_ids()`.
* The signature change is mechanical; all callers are
  in-crate.

### Step 2 — Reorder helpers (triangle)

* Add `match_canonical_triangle_body(body) -> Option<TriangleBindings>`
  that recognizes the slice 1 lowered shape and returns the
  three Scan RelIds plus their semantic variable mapping
  (which atom is the XY-edge, which is YZ, which is XZ).
* Add `rewrite_triangle_with_inner(body, inner_pair, bindings) -> Option<RirNode>`
  that emits a canonical lowered triangle body with the
  specified atoms at the inner Join. The variable-to-column
  projection is recomputed from the semantic mapping so the
  head's `(X, Y, Z)` output is preserved at columns
  `[0, 1, 3]`. Returns `None` if the variable mapping doesn't
  fit the canonical shape (defensive).
* Add the selectivity-driven choice loop with **pair-derived
  keys**:
  - For triangle's 3 candidate pairs, compute
    `estimate_join_cardinality` with the variable-derived
    keys (`[1]/[0]` for XY⋈YZ, `[1]/[1]` for YZ⋈XZ,
    `[0]/[0]` for XY⋈XZ).
  - **Safety floor (per Q4)**: skip if any of the three
    input atoms has no `StatsManager` entry OR
    `cardinality == 0`. The pass cannot detect the
    10%-fallback path inside `estimate_join_cardinality`;
    accepting a spurious winner from fully-populated cards
    is the documented edge case.
  - Pick the smallest; ties keep the existing order.
  - Call `rewrite_triangle_with_inner` if the chosen pair
    differs from the existing inner.

### Step 2a — Extend the promoter (semantic-slot inference)

This step is REQUIRED for W2.2 closure (see "Promoter must be
extended" above).

* Refactor `try_promote_triangle` (`promote.rs:176`) to:
  - Accept the three valid inner-key combinations
    (`[1]/[0]`, `[1]/[1]`, `[0]/[0]`) and the three
    corresponding outer-key combinations.
  - Infer which Scan is the XY-edge / YZ-edge / XZ-edge from
    the key signature of the matched body.
  - Emit `MultiWayJoin { inputs, slot_vars, … }` with
    `inputs` arranged in semantic order
    `[XY-Scan, YZ-Scan, XZ-Scan]` regardless of the body's
    positional layout, and `slot_vars` always
    `[[V_X,V_Y], [V_Y,V_Z], [V_X,V_Z]]`.
* Refactor `try_promote_4cycle` (`promote.rs:277`) analogously
  for the 4-cycle's bushy shape — accept the valid key
  combinations for each candidate inner-pair grouping; emit
  `inputs` in semantic order `[WX, XY, YZ, ZW]`; emit
  `slot_vars` `[[V_W,V_X], [V_X,V_Y], [V_Y,V_Z], [V_Z,V_W]]`.
* Existing in-crate promoter tests in
  `crates/xlog-logic/src/promote.rs::tests` get extended:
  - Existing `promotes_canonical_triangle` /
    `promotes_canonical_4cycle` keep passing (default
    positional layout).
  - New `promotes_triangle_with_inner_yz_xz_pair` (and
    `xy_xz_pair`) — bodies whose inner pair is the
    non-default but key-consistent variant. Expected:
    promotion succeeds AND emitted `MultiWayJoin.inputs` is
    in canonical semantic order.
  - 4-cycle counterparts for the alternative inner-pair
    groupings.

### Step 2b — Reorder helpers (4-cycle)

* Add `match_canonical_4cycle_body(body) -> Option<Cycle4Bindings>`
  recognizing the slice 2 lowered bushy shape.
* Add `rewrite_4cycle_with_inner_grouping(body, grouping, bindings) -> Option<RirNode>`
  that emits the canonical 4-cycle body with the chosen
  inner-pair grouping.
* The 4-cycle has 4 atoms (semantic slots WX, XY, YZ, ZW).
  The bushy shape's left + right subtrees each compute one
  binary join, then the outer join combines them. The
  implementation step enumerates **exactly two valid
  groupings** (those whose left and right pairs each share
  a variable):
  - `(WX ⋈ XY) + (YZ ⋈ ZW)` — left shares X, right shares Z.
  - `(XY ⋈ YZ) + (ZW ⋈ WX)` — left shares Y, right shares W.
  Opposite-edge pairings (WX⋈YZ, XY⋈ZW) share NO variable
  and are NOT enumerated — they would cross-product, not
  join. Exactly two candidates, no more.
* Cost formula: `est_left + est_right` per grouping (sum of
  the two binary intermediates). Pick the smaller. Tie →
  keep the optimizer's existing order (deterministic no-op).
* Same safety floor as triangle — any unseeded card → no
  rewrite.

### Step 3 — Compile-time cert (Part A)

In `crates/xlog-logic/src/optimizer.rs::tests`:

* Triangle tests:
  - `selectivity_pass_triangle_picks_smallest_inner_pair_snapshot_1`
  - `selectivity_pass_triangle_picks_smallest_inner_pair_snapshot_2`
  - `selectivity_pass_triangle_skips_when_card_missing_or_zero`
  - `selectivity_pass_triangle_skips_when_already_optimal`
* 4-cycle tests:
  - `selectivity_pass_4cycle_picks_smallest_grouping_snapshot_1`
  - `selectivity_pass_4cycle_picks_smallest_grouping_snapshot_2`
  - `selectivity_pass_4cycle_skips_when_card_missing_or_zero`
* Helpers: `build_canonical_triangle_body(a, b, c)`,
  `build_canonical_4cycle_body(a, b, c, d)`,
  `inspect_inner_pair(body) -> Option<(RelId, RelId)>`,
  `inspect_4cycle_inner_grouping(body) -> Option<Grouping>`.

### Step 4 — End-to-end + WCOJ-dispatch certs (Part B + C)

In
`crates/xlog-integration/tests/test_selectivity_pass_reordering.rs`:

* Use `Compiler::compile_with_stats_snapshot(source, Some(&snapshot))`
  (existing API at compile.rs:109).
* Tests:
  - `selectivity_pass_triangle_two_snapshots_produce_same_row_set`
  - `selectivity_pass_4cycle_two_snapshots_produce_same_row_set`
  - `selectivity_pass_reordered_triangle_still_dispatches_wcoj`
    (Part C — force-WCOJ + counter ≥ 1 + row set equals
    binary reference).
  - `selectivity_pass_reordered_4cycle_still_dispatches_wcoj`
    (Part C — force-WCOJ on 4-cycle).

### Step 5 — Workspace gate + evidence

* `cargo fmt --all -- --check`
* `cargo test -p xlog-logic --release` (with new compile-time tests)
* `cargo test -p xlog-integration --release --test test_selectivity_pass_reordering`
* Slice 1–5 + W2.4 regression preserved bit-identical:
  - `cargo test -p xlog-integration --release` (slice 1–5 + W2.4 certs)
  - `cargo test -p xlog-runtime --lib --release wcoj_cost_model` (slice 5)
  - `cargo test -p xlog-cuda-tests --test certification_suite --release`
* Evidence file at
  `docs/evidence/2026-05-04-w22-selectivity-pass-real-reordering/README.md`.

### Step 6 — End-of-slice closure update

Per process rule #4: this slice's commit-of-evidence proposes
the board update for W2.2 (OPEN → DONE), but does NOT
self-mark DONE. Per process rule #1, the user reviews the
slice and explicitly approves "mark W2.2 DONE"; a separate
follow-up commit then applies the board update.

### Step 7 — FF-merge to local main

* No push, no tag. Working tree clean. Same FF-merge pattern
  as W2.4.

## Risk & Open Questions

* **Q1 — Compile-time stats access.** Current
  `selectivity_pass::run` signature is
  `run(plan: &mut ExecutionPlan, _stats: &StatsManager)`
  (verified at `crates/xlog-logic/src/optimizer.rs:1241`).
  No `Arc` in the pass signature, no lock. The W2.2
  signature stays `&StatsManager` and just adds `rel_ids`.
* **Q2 — Pre-compile stats seeding for Part B / Part C
  certs.** Use the existing
  `Compiler::compile_with_stats_snapshot(source, Some(&snapshot))`
  at `crates/xlog-logic/src/compile.rs:109`. NO new public
  API is added by this slice. If the implementation step
  proves that exact path is insufficient (e.g., `selectivity_pass`
  doesn't yet receive the snapshot at the right phase), the
  step plumbs the existing snapshot through — never adds a
  new public API.
* **Q3 — Promoter compatibility (resolved by step 2a).** The
  existing matcher is positional (verified at
  `promote.rs:176` / `:277`); it cannot survive non-default
  reordered shapes. W2.2 includes step 2a to extend both
  matchers to recognize the canonical *semantic* shape
  regardless of positional layout, and to emit
  `MultiWayJoin.inputs` + `slot_vars` in semantic order. Part
  C cert verifies the invariant end-to-end.
* **Q4 — Default-selectivity fallback (acceptable per
  user, with explicit docs/tests).**
  `StatsManager::estimate_join_cardinality` returns `u64`
  with no provenance — the caller cannot tell whether the
  estimate came from the cached `JoinSelectivity` table, the
  column-distinct heuristic, or the 10% default fallback.
  The plan does NOT claim fallback detection. Safety floor:
  any input atom with no `StatsManager` entry OR
  `cardinality == 0` → body unchanged.
  **Documentation requirements (per user)**:
  - The selectivity_pass module-level doc comment in
    `optimizer.rs` explicitly states: "The 10% default
    fallback inside `estimate_join_cardinality` may produce
    uninformative pair estimates when relation cardinalities
    are populated but column statistics are not. The pass
    accepts this trade-off; the integration certs (Part B +
    Part C) gate on row-set parity, which holds regardless
    of selectivity quality."
  - One unit test pins the fallback-edge-case behavior:
    `selectivity_pass_with_only_relation_cards_may_pick_arbitrary_pair`
    — three atoms with seeded cardinality but no column
    stats. Assert: the pass either makes some choice (any
    of the three pairs) OR leaves the body unchanged; the
    test is tolerant by design and documents the
    uninformative-fallback case explicitly. Row-set parity
    is NOT asserted by this unit test (it's not an
    end-to-end test); Parts B + C carry that load.

## Provenance

- Closure board: `docs/v065-closure-board.md` Wave 2, W2.2.
- ROADMAP item #3: "Add join reordering based on selectivity
  estimates."
- Pairs with W2.4 (`record_join_result` feedback, DONE) — the
  feedback loop populates the selectivity cache that this
  pass reads. After warm runs, this pass picks the empirically
  best inner pair, not just the default-fallback one.
- Reader of cardinality estimates: this pass calls
  `StatsManager::estimate_join_cardinality(p, q, l_keys, r_keys)`
  with **pair-derived join keys from each candidate's
  shared-variable mapping** (see "Pair-selectivity lookup
  with pair-derived keys" above). Slice 5
  `CardinalityAwareCostModel::should_dispatch_*` and W2.4
  `record_wcoj_feedback` use the same API but with their
  own pair-specific keys (slice 5 / W2.4 currently key on
  the inner-pair `[1] / [0]` because they only consume the
  default-canonical inner pair; W2.2's per-candidate
  derivation is the more general pattern future stats
  consumers should follow).
