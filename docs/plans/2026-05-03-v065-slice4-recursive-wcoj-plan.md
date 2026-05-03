# v0.6.5 Slice 4 — Semi-naive Recursive WCOJ (Plan)

**Date:** 2026-05-03
**Branch (proposed):** `feat/v065-recursive-wcoj`
**Worktree (proposed):** `.worktrees/v065-recursive-wcoj`
**Base:** `main` at `616ec628` (slice 3 amendment)

## Slice Goal

Allow the WCOJ triangle and 4-cycle dispatch to fire **inside
recursive SCCs**, on the semi-naive iteration path. Today, the
promoter blanket-skips recursive SCCs and `execute_node` unwraps
`MultiWayJoin` bodies to their fallback, so recursive evaluations
never see a WCOJ kernel — only the binary-join tree.

This is the slice 4 cliff. After it lands, the same rule body
that gets WCOJ-dispatched in the non-recursive arm will be WCOJ-
dispatched in the recursive arm, with the recursive scan swapped
to its delta on each variant pass.

## What Already Works (Pre-Slice-4)

* **Walker hardening landed in main** (`rewrite_scan_nth_impl` at
  `crates/xlog-runtime/src/executor/rewrite.rs:477`). The walker
  covers `MultiWayJoin` arms — both `inputs` and `fallback` get
  the recursive Scan→delta rewrite. Test:
  `rewrite_scan_nth_rewrites_inputs_and_fallback`
  (`rewrite.rs:574`).
* **Non-recursive WCOJ dispatch** is the slice 1–3 stack: promoter
  emits `MultiWayJoin`, the non-recursive executor arm calls
  `try_dispatch_wcoj_{triangle,4cycle}`, falls back to
  `MultiWayJoin.fallback` on decline. Cost-model seam (slice 3)
  consults `WcojCostModel`.
* **Per-variant rewrite** in semi-naive
  (`recursive.rs:380`): each recursive Scan occurrence in the
  body becomes one variant with that occurrence's RelId swapped
  to delta. Multi-recursive bodies generate N variants per
  iteration, unioned into a per-rule delta.

## What's Missing

1. **Promoter guard** (`crates/xlog-logic/src/promote.rs:80`):
   `if recursive { continue; }` blanket-skips recursive SCCs.
   Remove it (with a refined condition; see step 1 below).
2. **Promoter signature** (`promote_multiway(&mut plan)`):
   `ExecutionPlan` only carries SCC *predicate names*; `RirNode::Scan`
   only carries `RelId`. The promoter cannot determine whether a
   scanned `RelId` belongs to the head SCC without an external
   `RelId → predicate name` map. Amend the public surface to
   accept it (see step 1 below).
3. **Recursive seeding pass**
   (`crates/xlog-runtime/src/executor/recursive.rs:244`):
   stable rules (zero recursive Scans) only run on the seeding
   pass — they don't reach the variant loop. Without a WCOJ hook
   here, a stable triangle in a recursive SCC executes the
   fallback only.
4. **Recursive variant evaluation**
   (`crates/xlog-runtime/src/executor/recursive.rs:390`):
   `execute_node(&variant_node)` walks the rewritten body. When
   `variant_node` is a `MultiWayJoin`, `execute_node`'s arm at
   `node_dispatch.rs:239` unwraps to fallback — no WCOJ
   dispatch occurs. Same fix as the seeding pass; share a helper.
5. **Stale doc comments** in `promote.rs` (header, line 65–72)
   and `wcoj_dispatch.rs` declaring "recursive WCOJ is excluded"
   need updating after the recursive arm gets the hook.

## Locked Scope (S1: linear-recursion only)

A rule body can have **0, 1, or N** recursive Scan occurrences.
Slice 4 S1 promotes only the first two:

| Recursive Scan count | Behavior                                        | Rationale                        |
|----------------------|-------------------------------------------------|----------------------------------|
| **0** (stable rule)  | Promote → WCOJ on seeding pass; subsequent iterations skip via `variants.is_empty()` | Trivial extension of slice 1–3   |
| **1** (linear-rec)   | Promote → 1 variant per iteration → 1 WCOJ dispatch on (full, …, delta, …, full) | The minimum semi-naive case      |
| **≥ 2** (multi-rec)  | **Skip promotion**; preserve binary-join semi-naive | Multi-version explosion + dedup risk; defer to slice 4.2 / v0.6.6 |

Counting "recursive Scans" = Scans whose RelId is in the same SCC
as the rule head. The promoter has access to `plan.sccs` and the
rule's head SCC, so this is local.

## What Slice 4 S1 Does NOT Do

* **No multi-recursive WCOJ.** A body with two `tc(_,_)` scans (or
  any pattern with ≥2 recursive Scans) stays binary-join in the
  recursive engine. Defer to slice 4.2.
* **No new shapes.** Triangle + 4-cycle only — same as slice 1–3.
* **No new kernels.** The WCOJ dispatcher's existing entry points
  are reused; only the call site is new.
* **No threshold change.** `WCOJ_ADAPTIVE_*_SKEW_THRESHOLD = 0.10`.
* **No cost-model change.** `SkewClassifierCostModel` is the only
  impl; slice 4 doesn't introduce stats-driven dispatch.
* **No env-var change.** Existing `XLOG_USE_WCOJ_*` flags govern
  recursive dispatch the same way they govern non-recursive.
* **No selectivity-pass change.** The slice 3 `selectivity_pass`
  no-op stays no-op.

## Step Plan (S1)

### Step 1 — Promoter signature + guard refinement (`xlog-logic`)

**Signature change.** The current
`promote_multiway(plan: &mut ExecutionPlan)` cannot resolve
`RelId → predicate name` to test SCC membership. Amend to:

```rust
pub fn promote_multiway(
    plan: &mut ExecutionPlan,
    rel_ids: &HashMap<String, RelId>,  // or &[(String, RelId)]
)
```

Caller (`compile.rs`) passes `self.lowerer.rel_ids()` — the
canonical relation table. Existing in-crate tests that synthesize
plans must construct and pass this mapping; the slice 1–3
"canonical triangle" tests get a 3-entry map, the 4-cycle tests
get a 4-entry one.

**Guard refinement.** Replace the blanket
`if recursive { continue; }` (`promote.rs:80`) with a per-rule
check:

* Add a helper
  `recursive_scan_count(body: &RirNode, head_scc: &Scc, rel_ids: &…) -> usize`
  that walks the body, resolves each Scan's RelId to its predicate
  name, and counts hits in `head_scc.predicates`.
* Predicate: promote iff `recursive_scan_count <= 1` AND the body
  matches one of the existing triangle/4-cycle patterns.
* Keep the existing `is_recursive` SCC flag — the helper consults
  the SCC's predicate set, not the flag, but the flag still
  short-circuits cheap cases.

**Tests** (in-crate, byte-preservation style):

* `promotes_stable_rule_in_recursive_scc` — rule body uses only
  extensional relations; SCC marked recursive; verify promotion.
* `promotes_linear_recursive_triangle` — rule has exactly one
  recursive Scan; verify promotion.
* `skips_multirec_triangle_in_recursive_scc` — ≥ 2 recursive
  Scans → no promotion.
* Update / replace `skips_recursive_scc_bodies` and
  `skips_recursive_scc_4cycle` to reflect refined contract.
* Update the existing slice 1–3 unit tests that call
  `promote_multiway(&mut plan)` to pass an empty or appropriate
  `rel_ids` map (their plans are non-recursive, so the count
  helper short-circuits).

### Step 2 — Per-body WCOJ dispatch entry points (`xlog-runtime`)

The current `try_dispatch_wcoj_{triangle,4cycle}` in
`wcoj_dispatch.rs` takes `&CompiledRule` and reads `rule.body`.
For the recursive arm, the body has been rewritten — we need a
variant that takes a `&RirNode`.

Two possible refactors:

* **A. Extract a `_on_body` entry point.** Add public-to-executor
  `try_dispatch_wcoj_triangle_on_body(&self, body: &RirNode) ->
  Result<Option<CudaBuffer>>`. Existing `try_dispatch_wcoj_triangle`
  becomes a thin wrapper: `self.try_dispatch_wcoj_triangle_on_body(&rule.body)`.
* **B. Refactor the rule param to body param.** All current call
  sites get `&rule.body` instead of `rule`.

Pick **A** — minimizes diff, leaves slice 1–3 call sites byte-
identical. The body extracted here is the rewritten variant_node
in the recursive arm, OR `rule.body` in the non-recursive arm.

* **Tests:** the existing slice 1–3 cert tests continue to pass
  (the wrapper is byte-identical). One new in-crate unit test
  pinning that `try_dispatch_wcoj_triangle_on_body(&rule.body) ==
  try_dispatch_wcoj_triangle(rule)` for the canonical triangle
  body.

### Step 3 — Recursive seeding + variant hook (`xlog-runtime`)

Two call sites need the WCOJ dispatch hook:

* **Seeding pass** (`recursive.rs:244` — the first non-recursive
  evaluation of each rule when delta is empty/being seeded).
  Stable rules (zero recursive Scans) **only** reach this site;
  without a hook here, stable triangles in recursive SCCs would
  execute fallback exclusively.
* **Variant evaluation** (`recursive.rs:390` — the per-variant
  loop where the recursive Scan is rewritten to delta).

Introduce a single helper to share the logic:

```rust
fn execute_wcoj_or_fallback_node(&mut self, node: &RirNode) -> Result<CudaBuffer> {
    if let RirNode::MultiWayJoin { .. } = node {
        if let Some(buf) = self.try_dispatch_wcoj_on_body(node)? {
            return Ok(buf);
        }
    }
    self.execute_node(node)
}
```

Where `try_dispatch_wcoj_on_body` tries triangle first then
4-cycle (slice 1 ordering — body cannot match both), returning
`Ok(None)` if neither shape matches. Both `recursive.rs:244` (or
the equivalent seeding `execute_node` call — confirm exact line
during implementation) and `recursive.rs:390` call this helper
instead of `execute_node` directly.

Behavior preservation: in slice 4, `MultiWayJoin` only enters
either path because step 1 promoted the rule. Pre-slice-4 plans
never produce a `MultiWayJoin` for recursive bodies, so the new
helper is a no-op on slice 1–3 plans (it just calls `execute_node`).
The non-recursive arm at `recursive.rs:104–128` keeps its existing
inline dispatch — unchanged in slice 4 to keep the diff small.

**Counter semantics.** Existing
`wcoj_triangle_dispatch_count` / `wcoj_4cycle_dispatch_count`
fields are reused. They count "successful WCOJ kernel result per
seeding/iteration/variant" — not rule-level dispatches. Document
this in the field doc comments and the slice 4 evidence file.
**No new recursive-arm counter fields.**

* **Tests** (integration, in `xlog-integration/tests/`):
  * `test_wcoj_recursive_stable_triangle.rs` — recursive SCC
    where one rule's body is a stable triangle (zero recursive
    Scans, all extensional). Force WCOJ on; assert the triangle
    dispatch counter == 1 (seeding pass) and the recursive
    fixpoint produces the correct row set vs binary-join
    reference.
  * `test_wcoj_recursive_linear_triangle.rs` — recursive SCC
    with one rule whose body is a triangle with exactly one
    recursive Scan. Force WCOJ on; assert the dispatch counter
    increments per iteration until fixpoint, and final row set
    matches binary-join reference. Bound iterations to small N
    (3–5) so the test is fast.
  * `test_wcoj_recursive_multirec_falls_back.rs` — recursive
    triangle with 2+ recursive Scans. Force WCOJ on; assert
    counter == 0 (no dispatch) and the row set matches binary-
    join. Confirms multi-rec exclusion holds.
  * One 4-cycle counterpart for stable and linear-rec.

### Step 4 — Adaptive parity test (`xlog-integration`)

Confirm the adaptive classifier (slice 2 default-off, slice 3
cost-model-routed) makes the same dispatch decision in the
recursive arm as in the non-recursive arm for the same fixture.

* **Test:** `test_wcoj_recursive_adaptive_parity.rs` — super-hub
  fixture used in `test_wcoj_4cycle_adaptive_dispatch.rs`,
  wrapped in a recursive SCC via a no-op recursive rule. Adaptive
  on; assert dispatch counter > 0 on the recursive triangle/
  4-cycle slot.

### Step 5 — Stale-doc cleanup

After step 3 lands, the following doc claims are no longer true
and must be updated:

* `crates/xlog-logic/src/promote.rs` header (lines 30–55) and
  inline (lines 65–72) state "recursive SCC bodies are skipped"
  and "the executor's recursive engine never invokes the WCOJ
  dispatch hook." Both become false. Replace with the slice 4
  contract: "promote per-rule when the body has ≤ 1 recursive
  Scan; the recursive arm dispatches via
  `execute_wcoj_or_fallback_node` on both seeding and variant
  passes."
* `crates/xlog-runtime/src/executor/wcoj_dispatch.rs` — any
  comments that say WCOJ "fires only on the non-recursive
  branch" or similar. Update to describe the recursive entry
  via `execute_wcoj_or_fallback_node`.
* `crates/xlog-runtime/src/executor/recursive.rs` — the
  comment block at lines 65–72 about why recursive WCOJ is
  excluded. Replace with a note pointing to the new helper.

These edits are surgical (comments only) but load-bearing for
future readers — they prevent the next slice from re-introducing
the same blanket exclusion.

### Step 6 — Workspace gate + docs

* `cargo fmt --all -- --check`
* `cargo test --workspace --release --exclude pyxlog`
* CUDA cert suite: `cargo test -p xlog-cuda-tests --test certification_suite --release` — must remain 206/206
* WCOJ regression: 69 cuda + 59 integration + 13 cost-model unit + new step 3/4 tests; all pass
* `real_world_tests` under `XLOG_USE_DEVICE_RUNTIME=1` — confirm
  no regression on the existing recursive fixtures (transitive
  closure, reachability)
* Evidence file at `docs/evidence/2026-05-03-v065-slice4-recursive-wcoj/README.md`
  with: per-step test names, dispatch counter assertions, and a
  pinned dispatch-counter ladder for the new tests.

### Step 7 — FF-merge to local main

* No push, no tag.
* Working-tree clean check.
* Same FF-merge pattern as slice 3 amendment.

## Acceptance Gates

| # | Gate | Owner |
|---|------|-------|
| 1 | Stable triangle rule in recursive SCC dispatches WCOJ on the seeding pass (counter == 1) | step 3 |
| 2 | Linear-recursive triangle dispatches WCOJ per iteration; row set matches binary-join | step 3 |
| 3 | Multi-recursive triangle skips WCOJ (counter == 0) and matches binary-join | step 3 |
| 4 | All three gates above for 4-cycle | step 3 |
| 5 | Adaptive classifier makes same decision in recursive vs non-recursive arm | step 4 |
| 6 | Stale doc claims about "recursive WCOJ excluded" are removed | step 5 |
| 7 | Workspace + CUDA cert + real_world tests no regression | step 6 |

## Risk & Open Questions

* **Q1**: Per-iteration classifier cost. The recursive arm runs
  the classifier on each iteration with unchanged buffers. **Out
  of slice 4 scope** — document as a perf opportunity for slice
  5 / v0.6.6. No phase-timing evidence in slice 4 (correctness
  + certification only, per scope lock).
* **Q2**: Counter semantics — locked. "Successful WCOJ kernel
  result per seeding/iteration/variant." No new counter fields.
  Documented in field doc comments + evidence file.
* **Q3**: The variant_node passed into `try_dispatch_wcoj_on_body`
  has the recursive Scan's RelId swapped to the delta RelId.
  The slot's CudaBuffer therefore comes from the delta store
  entry, not the full store. The dispatcher reads the buffer
  via slot RelId → store lookup, so the delta-vs-full
  distinction is invisible to the kernel. **Confirmation
  needed**: trace the buffer-resolution path in slice 1's
  `try_dispatch_wcoj_triangle` (likely already RelId-keyed) and
  pin in the step 3 unit test.

## Out-of-Slice (Deferred)

* Multi-recursive WCOJ (≥ 2 recursive Scans per body) — slice 4.2 or later.
* Stats-driven cost model for recursive arm — slice 5.
* General-arity kernels — slice 5.
* Histogram caching across iterations — v0.6.6 perf pass.
* Recursive 4-clique / k-clique — needs new kernels (v0.6.6+).
* DTS provenance integration with recursive WCOJ — independent track.

## Provenance Note

The summary that opens this conversation references slice 1's
`MultiWayJoin.fallback` as a "post-optimizer binary-join tree";
preserving fallback behavior on decline is load-bearing for
non-recursive byte-identity. Slice 4 inherits that contract: in
the recursive arm, when WCOJ dispatch declines, the fallback
binary-join tree (with its already-rewritten Scans from
`rewrite_scan_nth`) is the correct execution path.
