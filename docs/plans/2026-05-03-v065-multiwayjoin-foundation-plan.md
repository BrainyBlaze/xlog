# v0.6.5 Slice 1 — `MultiWayJoin` RIR Foundation

**Date:** 2026-05-03
**Branch (proposed):** `feat/v065-multiwayjoin-rir`
**Worktree (proposed):** `.worktrees/v065-multiwayjoin-rir`
**Baseline commit:** `b48c2efd` (verified pushed to `origin/main`; `git push` returns "Everything up-to-date")
**Status:** Plan, post-review amendments. Implementation may proceed only after this revision is acknowledged.

## Goal

Replace the executor's strict tree-shape pattern matcher
(`match_triangle_rir` in `crates/xlog-runtime/src/executor/wcoj_dispatch.rs:194`)
with an explicit `RirNode::MultiWayJoin` IR surface. The matcher
becomes a node-shape check plus output-column verification. **No
observable behavior change**: gate semantics, kernel selection,
adaptive classifier, and store mutations are byte-identical to
v0.6.2.

This slice is the foundation for the rest of v0.6.5 (4-way kernels,
recursion, cost model). Each later slice extends the promoter and
the executor's `MultiWayJoin` arm rather than the fragile pattern
matcher.

## Naming

`MultiWayJoin`. Operator-neutral; future cost models may pick
sort-merge or hash-multiway physical strategies without an enum
migration.

## Architecture

### Two-phase, fallback-embedding, post-optimizer placement

1. **`Lowerer::lower_program` is unchanged.** It produces the same
   nested-`Join` + `Project` tree.
2. **`Optimizer::optimize` runs as today** inside
   `Compiler::compile_program_with_stats_snapshot`
   (`crates/xlog-logic/src/compile.rs:278-282`), rewriting each
   `rule.body` for predicate pushdown.
3. **`promote_multiway(&mut plan)` runs after the optimizer loop**,
   inside `Compiler::compile_program_with_stats_snapshot`. The
   optimizer never sees `MultiWayJoin`; the promoter walks the
   already-optimized RIR and rewrites recognized triangle subtrees
   to:

   ```rust
   RirNode::MultiWayJoin {
       inputs: vec![Scan(rel_xy), Scan(rel_yz), Scan(rel_xz)],
       slot_vars: vec![
           vec![Some(0), Some(1)],   // [V_X, V_Y]
           vec![Some(1), Some(2)],   // [V_Y, V_Z]
           vec![Some(0), Some(2)],   // [V_X, V_Z]
       ],
       output_columns: vec![ProjectExpr::Column(0), ProjectExpr::Column(1), ProjectExpr::Column(3)],
       fallback: Box::new(<original Project { Join { Join, Scan } } tree, post-optimizer>),
   }
   ```

4. **Executor matches `MultiWayJoin`.** WCOJ gate decision uses
   `slot_vars` + `inputs` + `output_columns` (see
   "Eligibility Validation" below). On any non-dispatch outcome,
   the executor descends into `fallback`.

The `fallback` field is the safety net. It is the *exact tree the
optimizer produced*, captured before the promoter wraps it.

### Why post-optimizer

Per review:
`Compiler::compile_program_with_stats_snapshot` runs
`Optimizer::optimize` after `Lowerer::lower_program`. The optimizer
is exhaustive over `RirNode` variants (`predicate_pushdown`,
`estimate_width`, `estimate_cost`, `find_column_relation`); putting
`MultiWayJoin` upstream of it would force the optimizer to learn the
new variant immediately. Running the promoter *after* optimization
means the optimizer code base sees only the legacy variants. The
optimizer still picks up an exhaustive-match arm for compile safety
(see "Cross-Crate Walker Audit"), but that arm is unreachable in
practice — the promoter runs after optimization completes.

### Why not just walk the IR-without-fallback

Three alternatives considered and rejected:

| Option | Reason rejected |
|---|---|
| Strip the binary-join tree and reconstruct on fallback | Reconstruction is non-trivial (key derivation, column re-numbering); guaranteeing identity to the optimizer's output adds risk. |
| Tag the existing tree with a sidecar `multiway_eligible: true` flag in `RirMeta` | Doesn't solve the executor matcher's fragility; matcher still walks the tree. Also: `RirMeta` lives on `CompiledRule`, not `RirNode` — no node-level flag site exists. |
| Make `MultiWayJoin` a wrapper that lazily computes inputs from a child tree | Breaks pattern-match ergonomics; future cost models can't see the inputs without executing. |

Embedding the `fallback` tree pays one box-allocation per eligible
rule at compile time and zero at execution time when dispatch fires.
Acceptable cost.

## Surface Changes

### `crates/xlog-ir/src/rir.rs`

Add to `pub enum RirNode`:

```rust
/// A multi-way conjunctive join that the executor MAY dispatch
/// to a specialized physical operator (e.g. GPU WCOJ). The
/// fallback subtree is the IR-equivalent binary-join plan and
/// is executed verbatim when the dispatch declines.
///
/// **Invariant**: Executing `fallback` produces the same row set
/// as a successful specialized dispatch. The promoter is
/// responsible for upholding this — see
/// `xlog-logic::promote::promote_multiway`.
MultiWayJoin {
    /// Input scans, in physical-plan slot order. For the v0.6.5
    /// initial promoter, this is exactly `[Scan(rel_xy),
    /// Scan(rel_yz), Scan(rel_xz)]` for a recognized triangle.
    /// Each input MUST be `RirNode::Scan { rel }` in v1.
    inputs: Vec<RirNode>,
    /// Per-slot, per-column variable-class id. Same id across
    /// slots → join on that variable. For the triangle, this is
    /// `[[Some(0), Some(1)], [Some(1), Some(2)], [Some(0), Some(2)]]`.
    /// `None` is reserved for constant-bound or don't-care
    /// columns; the v1 promoter never emits `None`.
    slot_vars: Vec<Vec<Option<u32>>>,
    /// Output projection in head-tuple order, identical to what
    /// the equivalent `Project { input: Join { ... } }` carries.
    /// For the triangle, MUST be exactly
    /// `[Column(0), Column(1), Column(3)]`. The executor
    /// re-validates this; a malformed or rotated projection is
    /// treated as ineligible (no dispatch).
    output_columns: Vec<ProjectExpr>,
    /// IR-equivalent binary-join plan. Executed verbatim on
    /// dispatch decline. Promoter MUST capture this from the
    /// post-optimizer tree, not synthesize it.
    fallback: Box<RirNode>,
},
```

`impl RirNode::collect_relations` (line 261, recursion helper for
`referenced_relations`): add an arm that recurses into `inputs`
only. The `fallback` references the same set by construction
(promoter invariant); we keep the canonical answer minimal.

`impl RirNode::is_leaf`: returns `false` for `MultiWayJoin`.

### `crates/xlog-logic/src/promote.rs` (new file)

Single public entry:

```rust
/// Walk an `ExecutionPlan` (post-lowering, post-optimizer) and
/// rewrite eligible triangle subtrees to `MultiWayJoin`. Returns
/// nothing; mutates `plan.rules_by_scc[*].body` in place.
/// Idempotent.
pub fn promote_multiway(plan: &mut ExecutionPlan);
```

Eligibility: matches the *exact* tree `match_triangle_rir`
matches today (a strict shape check covering outer Project,
both Joins, all three Scans, and key shapes `[1]/[0]` and
`[0,3]/[0,1]`). The promoter does not introduce new eligibility;
that is later-slice work.

`CompiledRule.meta` is preserved unchanged. The promoter only
rewrites `rule.body`.

### `crates/xlog-logic/src/compile.rs`

In `compile_program_with_stats_snapshot`, after the
`for rule in rules { rule.body = optimizer.optimize(...); }` loop
(currently line 278-282), invoke:

```rust
crate::promote::promote_multiway(&mut plan);
```

before `Ok(plan)`. Single-line change.

### `crates/xlog-runtime/src/executor/node_dispatch.rs`

Add to `execute_node`'s match (line 46-end):

```rust
RirNode::MultiWayJoin { fallback, .. } => self.execute_node(fallback),
```

This arm makes any direct caller of `execute_node` safe — even ones
that bypass the WCOJ dispatch (probabilistic eval, neural store
walks, future visitors). Since v0.6.5 slice 1 dispatches WCOJ only
through `recursive.rs:73`'s non-recursive arm, recursive bodies that
nest a `MultiWayJoin` (none today, but possible after later slices)
take the fallback path automatically. This is the load-bearing
defensive arm.

### `crates/xlog-runtime/src/executor/wcoj_dispatch.rs`

`match_triangle_rir` is **deleted**. Replacement:

```rust
fn match_multiway_triangle(body: &RirNode) -> Option<TriangleRirMatch> {
    let RirNode::MultiWayJoin { inputs, slot_vars, output_columns, .. } = body else {
        return None;
    };
    if inputs.len() != 3 { return None; }
    if !slot_vars_match_canonical_triangle(slot_vars) { return None; }
    if !output_columns_match_canonical_triangle(output_columns) { return None; }
    let rel_xy = scan_rel(&inputs[0])?;
    let rel_yz = scan_rel(&inputs[1])?;
    let rel_xz = scan_rel(&inputs[2])?;
    Some(TriangleRirMatch { rel_xy, rel_yz, rel_xz })
}

fn slot_vars_match_canonical_triangle(slot_vars: &[Vec<Option<u32>>]) -> bool {
    // Exact canonical shape: three slots of arity 2 with var ids
    // in the [[A,B],[B,C],[A,C]] pattern. Distinct ids A, B, C;
    // A != B != C != A. Anything else returns false.
    if slot_vars.len() != 3 { return false; }
    let s0 = &slot_vars[0];
    let s1 = &slot_vars[1];
    let s2 = &slot_vars[2];
    if s0.len() != 2 || s1.len() != 2 || s2.len() != 2 { return false; }
    let (a, b) = match (s0[0], s0[1]) {
        (Some(a), Some(b)) if a != b => (a, b),
        _ => return false,
    };
    let c = match (s1[0], s1[1]) {
        (Some(b1), Some(c)) if b1 == b && c != a && c != b => c,
        _ => return false,
    };
    matches!((s2[0], s2[1]), (Some(a2), Some(c2)) if a2 == a && c2 == c)
}

fn output_columns_match_canonical_triangle(cols: &[ProjectExpr]) -> bool {
    // The certified GPU kernel emits in (X, Y, Z) order. Any
    // rotation, drop, or computed expression returns false.
    cols.len() == 3
        && matches!(cols[0], ProjectExpr::Column(0))
        && matches!(cols[1], ProjectExpr::Column(1))
        && matches!(cols[2], ProjectExpr::Column(3))
}

fn scan_rel(node: &RirNode) -> Option<RelId> {
    match node {
        RirNode::Scan { rel } => Some(*rel),
        _ => None,
    }
}
```

Per review: a malformed or rotated head projection MUST decline
dispatch even if `slot_vars` and `inputs` look correct. The
executor's `MultiWayJoin` arm in `execute_node` is what makes
"decline" safe — it falls through to `fallback`.

### `crates/xlog-runtime/src/executor/recursive.rs`

The non-recursive arm (line 73-83) calls
`try_dispatch_wcoj_triangle(rule)`. Today, `rule.body` is a
`Project { Join { ... } }`; after the promoter, eligible bodies are
`MultiWayJoin`. When dispatch returns `Ok(None)`:

```rust
let body_to_execute = match &rule.body {
    RirNode::MultiWayJoin { fallback, .. } => fallback.as_ref(),
    other => other,
};
let result = self.execute_node(body_to_execute)?;
```

This is the only behavioral hot-path change. Identity of `fallback`
to the post-optimizer tree → identity of result row set to v0.6.2.

(Equivalently, since `execute_node` itself now has a
`MultiWayJoin -> execute_node(fallback)` arm, the explicit
destructuring above is redundant. Pick one site for fallback descent
to avoid double work; recommendation is to keep the explicit match
in `recursive.rs` — clearer at the dispatch site — and have
`execute_node`'s arm act as a defensive default for any other caller.
Resolved during implementation; no API impact.)

## Cross-Crate Walker Audit (mandatory per review)

Every exhaustive `RirNode` match in the workspace gets a new arm.
Audit produced these sites; each is a one-line addition:

| File | Function | Required arm |
|---|---|---|
| `xlog-ir/src/rir.rs:261` | `collect_relations` | recurse into `inputs` |
| `xlog-logic/src/optimizer.rs:260` | `predicate_pushdown` | `MultiWayJoin { .. } => node` (pass-through; promoter runs after optimizer, so this arm is unreachable in production but required for compile safety) |
| `xlog-logic/src/optimizer.rs:573` | `estimate_width` | use `output_columns.len()` |
| `xlog-logic/src/optimizer.rs:779` | `estimate_cost` | sum cost of `inputs` (heuristic — exact cost model is later-slice work) |
| `xlog-logic/src/optimizer.rs:1147` | `find_column_relation` | walk `inputs[0]` for col 0–arity, `inputs[1]` for next, etc.; v1 returns `None` (sufficient for current callers, all of which are pushdown-internal and only see pre-optimizer trees) |
| `xlog-logic/src/compile.rs:660,735,740` | `contains_fixpoint` / projection-stripping helpers | `MultiWayJoin { fallback, .. } => recurse into fallback` |
| `xlog-runtime/src/executor/node_dispatch.rs:46` | `execute_node` | `MultiWayJoin { fallback, .. } => self.execute_node(fallback)` (safety net) |
| `xlog-runtime/src/executor/rewrite.rs:116` | `contains_non_monotonic_ops` | recurse into `inputs` (and `fallback` if pessimistic — recommend `inputs` only since `fallback` is structurally derivable) |
| `xlog-runtime/src/executor/rewrite.rs:241` | relation-collect | recurse into `inputs` |
| `xlog-runtime/src/executor/rewrite.rs:291` | rel-rewrite | rebuild `MultiWayJoin` with rewritten `inputs` and `fallback` |
| `pyxlog/src/ilp.rs:445` | `walk_tmj` | descend into `fallback` (catch-all `_ => None` is *not* sufficient — a `MultiWayJoin` wrapping a TMJ-bearing fallback would silently miss it). Add explicit `MultiWayJoin { fallback, .. } => walk_tmj(fallback, target_mask)`. |

Every audit entry gets a unit test in its home crate that
constructs a `MultiWayJoin` and exercises the visitor.

## Lowerer / Planner Boundary

* `Lowerer::lower_program` is unchanged.
* `Optimizer::optimize` is unchanged in behavior; gains a no-op
  pass-through arm for `MultiWayJoin` (compile safety).
* `Compiler::compile_program_with_stats_snapshot` gains a single
  call to `promote_multiway(&mut plan)` after the optimizer loop.
* `ExecutionPlan` and `CompiledRule` are unchanged.
* `RirMeta` (which lives on `CompiledRule`, not `RirNode`) is
  preserved unchanged by the promoter.

## Executor Fallback Behavior

A `MultiWayJoin` body produces a row set via one of two paths:

1. **Specialized dispatch** (`try_dispatch_wcoj_triangle` returns
   `Some(buf)`). Buffer installed exactly as today; counter
   `wcoj_triangle_dispatch_count` increments; phase timing under
   feature gate populates as today.
2. **Fallback** (`Ok(None)`, for any of the existing reasons:
   gate off, kill switch, classifier below threshold, missing
   buffer, mixed width, kernel error, malformed `output_columns`,
   non-canonical `slot_vars`). The executor descends into
   `fallback`; result is unioned/installed via the same path the
   non-eligible branch uses.

Failure isolation contract is unchanged.

## Tests

### RED-first surface

1. **`xlog-ir`** (new file `tests/test_multiway_rir.rs`):
   - Construct a `MultiWayJoin` directly. Assert
     `referenced_relations()` agrees with the `inputs` slice.
   - `is_leaf()` returns `false`.

2. **`xlog-logic`** (new file `tests/test_promote_multiway.rs`):
   - Lower a triangle program through the full `Compiler`
     pipeline. Assert the post-optimizer, post-promoter
     `rule.body` is `MultiWayJoin` with the right `inputs`,
     `slot_vars`, and `output_columns`.
   - Assert the embedded `fallback` is **structurally equal** to
     a separately-compiled plan that runs lower → optimize → no
     promotion. This pins the no-behavior-change guarantee into
     a unit test.
   - Lower a non-triangle program (4-arity head, recursive SCC,
     mixed-arity body, predicate-pushdown-altered triangle that
     no longer matches the strict shape). Assert the body is
     **not** rewritten.
   - Assert `CompiledRule.meta` is preserved byte-for-byte.

3. **`xlog-logic`** (extend optimizer mod tests):
   - `Optimizer::optimize` on a synthesized `MultiWayJoin`
     returns the node unchanged. Smoke for the pass-through arm.

4. **`xlog-runtime`** (extend
   `crates/xlog-runtime/src/executor/wcoj_dispatch.rs` mod tests):
   - `match_multiway_triangle` succeeds on a synthesized
     `MultiWayJoin` with canonical slot_vars and output_columns.
   - Returns `None` for: rotated `output_columns`
     (`[Column(1),Column(0),Column(3)]`), arity-mismatched
     `output_columns`, non-canonical `slot_vars`
     (`[[A,B],[B,C],[A,B]]` and other malformed shapes),
     non-Scan inputs, and `inputs.len() != 3`.
   - `execute_node`'s `MultiWayJoin` arm with a stub fallback
     produces the fallback's result.

5. **`xlog-runtime/executor/rewrite.rs`** mod tests:
   - `contains_non_monotonic_ops` on a `MultiWayJoin` whose
     `fallback` contains a `Diff` returns the same answer as
     calling it on the `fallback` directly.
   - relation-rewrite over a `MultiWayJoin` rewrites both
     `inputs` and `fallback` consistently.

6. **`pyxlog`** ILP unit tests:
   - `walk_tmj` finds a TMJ wrapped in a `MultiWayJoin`'s
     `fallback`. Single test, gated by the existing pyxlog test
     harness.

7. **`xlog-integration`** existing certification tests continue
   to pass without modification:
   - `tests/test_wcoj_rir_shape_cert.rs` — RIR-shape certification
     (planner → provider). Now exercises the promoted shape.
   - `tests/test_wcoj_executor_wiring.rs`
   - `tests/test_wcoj_adaptive_dispatch.rs`
   - `tests/test_wcoj_adaptive_default_on.rs`
   - `tests/test_wcoj_dispatch_stream_reuse.rs`
   - `tests/test_wcoj_dispatch.rs`

8. **Bench parity**:
   - `crates/xlog-integration/benches/wcoj_triangle_bench.rs` — no
     change required. Triangle wall numbers should fall in the same
     band as v0.6.2 locked acceptance gates.

### GREEN gate

* `cargo test -p xlog-ir -p xlog-logic -p xlog-runtime --release`
* `cargo test -p xlog-integration --release` (hits all
  `test_wcoj_*` and `wcoj_phase_report` smoke).
* `cargo test --workspace --all-targets --exclude pyxlog --release`
  before merge.
* CUDA cert suite: `cargo test -p xlog-cuda-tests --test certification_suite --release`.
* Real-world device-runtime tests under `XLOG_USE_DEVICE_RUNTIME=1`.
* `cargo fmt --all -- --check`, `cargo build --release`.

### Equivalence sanity (manual probe before merge)

Run an integration test once with the promoter disabled (gated
via a `#[cfg(test)]` knob if needed) and once with it on; assert
byte-identical row sets. One-time validation, not a permanent
gate.

## Out of Scope (explicit)

* **No GPU kernel changes.** `wcoj.cu` untouched. No 4-way,
  general-arity, shared-memory, or warp-level work.
* **No recursive WCOJ.** SCC mixed-execution path is unchanged.
* **No cost model.** The promoter recognizes only the existing
  triangle shape; variable ordering, selectivity, join reordering
  deferred.
* **No new CUDA dispatch surface.** Force gate, adaptive
  classifier, kill switch, phase timing all live in
  `wcoj_dispatch.rs` exactly as today.
* **No AST-level changes.** `xlog-integration::wcoj_dispatch`
  legacy surface untouched.
* **No new env vars.** Config knobs unchanged.
* **No documentation overhaul.** `docs/BENCHMARKS.md`, the
  evidence README, and the WCOJ architecture guide are not
  rewritten; only `RirNode` doc-comments and one short paragraph
  in `wcoj_dispatch.rs`'s module doc are updated.
* **No `MultiWayJoin` consumer outside the executor's WCOJ
  path.** Other IR walkers descend into `fallback` (or `inputs`
  when collecting relations).
* **No release tag.** Slice on top of v0.6.2; tag decision is for
  the eventual v0.6.5 cumulative release.
* **No fast-path optimization.** When the executor descends into
  `fallback`, it pays the same cost as v0.6.2's binary-join path,
  not less. Fast-path layout reuse for `fallback` is a later
  slice if it ever shows up in a profile.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Optimizer (or other walker) sees `MultiWayJoin` and is missing an arm. | Audit table above enumerates every site; each gets a one-line arm and a unit test. |
| Predicate pushdown after slice 1 mutates the triangle's inner Join shape so the strict matcher rejects it; dispatch silently turns off in production. | Add an integration test that lowers, optimizes, and asserts the promoter still recognizes a triangle program. If pushdown breaks the match, that is real coverage loss; the response is to either teach the promoter the post-pushdown shape (later slice) or restrict pushdown for triangle bodies (out of scope for this slice — file a follow-up). |
| `walk_tmj` in pyxlog silently misses a TMJ inside `MultiWayJoin.fallback`. | Audit covered. Explicit arm added with test. |
| `RirMeta` semantics drift from `CompiledRule.meta` after promotion. | Unit test asserts `meta` byte-equal pre/post-promotion. |
| `output_columns` validation rejects future-promoter shapes (e.g. column-projected triangle). | Intentional. v1 only certifies the canonical (X,Y,Z) order; future slices either generalize the kernel or generalize the matcher in tandem. |
| Fallback path doubles compile-time memory in pathological programs. | Bounded by `count(eligible rules) * Box<RirNode>` overhead. Spot-check `wcoj_phase_report` and confirm wall-time bands. |

## Build Sequence (when implementation starts)

1. Spawn worktree `feat/v065-multiwayjoin-rir` off `b48c2efd`.
2. Land `RirNode::MultiWayJoin` enum variant + IR-side recursion
   helpers + `xlog-ir` unit tests. Cargo green.
3. Add the cross-crate compile-safety arms (audit table). Each
   site gets a unit test. Cargo green; existing behavior
   unchanged.
4. Land `xlog-logic::promote::promote_multiway` + unit tests
   (`test_promote_multiway.rs`). Promoter is **not yet wired
   into `Compiler::compile_program_with_stats_snapshot`**.
   Cargo green.
5. Wire promoter into `Compiler::compile_program_with_stats_snapshot`
   *after* the optimizer loop. The executor still runs
   `match_triangle_rir` against the raw tree → it now sees
   `MultiWayJoin` and falls through silently. Cert suite:
   dispatch counter goes to 0 (intentional, transient).
6. Replace `match_triangle_rir` with `match_multiway_triangle`
   in `wcoj_dispatch.rs` + executor fallback descent in
   `recursive.rs`. Cert suite: dispatch counter back on the
   v0.6.2 numbers. Equivalence sanity probe.
7. Delete dead code (`match_triangle_rir`, any tests that
   constructed the raw triangle tree to feed the matcher).
8. Workspace gate run. FF-merge to local main. STOP. Push
   decision is owner's.

Each step is its own commit; steps 5–7 are the load-bearing
behavior-preservation block and must run all WCOJ cert tests
between commits.

## Acceptance

Slice is done when all of:

* `cargo test --workspace --release` is green with the cert
  suite.
* `wcoj_triangle_dispatch_count` matches v0.6.2 baseline on the
  executor wiring tests (within run-to-run variance).
* `wcoj_phase_report` runs to completion and produces a
  bench-readable output qualitatively identical to the v0.6.2
  phase-timing report.
* No new env var, no new public API beyond the `RirNode` variant
  + `promote_multiway` entry point.
* `match_triangle_rir` is gone.
