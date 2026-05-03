# v0.6.5 Slice 3 — Cost-Model Foundation (S1: infrastructure-only)

**Date:** 2026-05-03
**Branch (proposed):** `feat/v065-cost-model-foundation`
**Worktree (proposed):** `.worktrees/v065-cost-model-foundation`
**Baseline commit:** `c4e6b3d6` (origin/main HEAD; v0.6.5 slice 2 landed)
**Status:** Plan, post-review amendments. Approved scope is **S1: infrastructure-only**, no behavior change.

## Goal

Lay the cost-model + selectivity-stats foundation that slices 4
(recursive WCOJ) and 5 (general-arity kernels) will consume.
**Behavior is unchanged this slice**: every dispatch decision and
every join tree the workspace produces today is preserved
byte-for-byte.

What ships:

1. A `WcojCostModel` trait + a default implementation that wraps
   the existing skew-classifier dispatch decision verbatim. The
   trait is the seam; the default impl preserves current behavior.
2. Stats-aware **inputs** wired into the seam — the cost model
   receives a `&StatsManager` reference, but the default impl
   ignores it (matches today's classifier-only logic).
3. An `Optimizer::selectivity_pass` module / entry point that
   exists, is invoked, and is a **no-op by default** — it walks
   the IR and returns it unchanged. Future slices add real
   reordering logic behind this seam.
4. Tests proving the seam exists, preserves current plans, and
   can be swapped in controlled unit tests.

What does **not** ship (locked out of scope per S1):

* Threshold replacement (`WCOJ_ADAPTIVE_*_SKEW_THRESHOLD` stays at `0.10`).
* Binary-join selectivity-driven reordering (workspace-wide
  blast-radius — separate slice if/when needed).
* Default-on adaptive flips for either triangle or 4-cycle.
* Recursive WCOJ.
* General-arity / 4-way kernels.
* Heat statistics integration / adaptive indexing.
* Any change to `wcoj_4cycle_dispatch_count` / `wcoj_triangle_dispatch_count`
  numbers on existing cert tests.

## Non-Goals (locked, restating)

* No new public API beyond the `WcojCostModel` trait + the
  optimizer's `selectivity_pass` entry.
* No changes to `RuntimeConfig` knobs, env vars, or kernel manifest.
* No CUDA kernel changes.
* No release tag.

## Existing Infrastructure (verified)

`xlog-stats` already exposes the inputs the cost model needs:

* `RelationStats { cardinality, byte_size, heat, ... }`
* `ColumnStats { distinct, range, null_count, ... }` with
  `equality_selectivity(total_rows) -> f64` and
  `range_selectivity(low, high) -> f64`.
* `JoinSelectivity::estimate_selectivity_from_stats(left_distinct, right_distinct) -> f64`
* `StatsManager::estimate_join_cardinality(left, right, left_keys, right_keys) -> u64`

`xlog-logic::Optimizer` already consumes `Arc<StatsManager>` via
`Optimizer::new(stats)`. `Optimizer::estimate_cost(node) -> PlanCost`
exists.

`xlog-runtime::executor::wcoj_dispatch.rs` already has the
classifier-based dispatch path; the cost-model seam wraps the
existing call sites without changing their inputs or outputs.

## Architecture

### Trait location and shape

New module `crates/xlog-runtime/src/executor/wcoj_cost_model.rs`:

```rust
use xlog_cuda::CudaBuffer;
use xlog_cuda::device_runtime::StreamId;
use xlog_cuda::CudaKernelProvider;
use xlog_stats::StatsManager;

/// v0.6.5 slice 3 — cost-model seam for WCOJ dispatch.
///
/// Implementations decide whether a recognized WCOJ-eligible rule
/// should dispatch the GPU kernel or fall back to the binary-join
/// chain. Slice 3 ships the default impl `SkewClassifierCostModel`
/// that preserves the v0.6.5 slice 2 behavior verbatim. Future
/// slices replace the trait's default impl OR provide additional
/// impls (e.g. `StatsDrivenCostModel`) without rewriting dispatch
/// call sites.
pub(super) trait WcojCostModel: Send + Sync {
    /// Decide dispatch for a triangle WCOJ. Returns `true` to
    /// dispatch the kernel, `false` to fall back. `stats` on the
    /// ctx is informational; the default impl ignores it. The
    /// classifier score is fetched via `scorer` (a SkewScoreSource
    /// — production = `CudaKernelProvider`, tests = stub).
    fn should_dispatch_triangle(
        &self,
        ctx: &WcojDispatchCtx,
        scorer: &dyn SkewScoreSource,
    ) -> bool;

    /// Decide dispatch for a 4-cycle WCOJ. Same shape as triangle.
    fn should_dispatch_4cycle(
        &self,
        ctx: &WcojDispatchCtx,
        scorer: &dyn SkewScoreSource,
    ) -> bool;
}

/// Inputs to a cost-model decision. Shape-agnostic; carries the
/// minimum context every implementation needs. Future slices
/// extend this with shape-specific fields (e.g. `slot_vars`).
///
/// Note: `provider` is NOT a field — the classifier score is
/// fetched via a separate `SkewScoreSource` parameter so trait-
/// swap tests don't need a real `CudaKernelProvider`.
pub(super) struct WcojDispatchCtx<'a> {
    pub stats: &'a StatsManager,
    pub launch_stream: StreamId,
    pub width: WcojKeyWidth,
    pub buffers: &'a [&'a CudaBuffer],
    /// Pre-resolved relation IDs in WCOJ slot order. Cost models
    /// that consult `stats` use these to look up cardinality and
    /// per-column selectivity for the rule's actual inputs.
    pub slot_rels: &'a [xlog_core::RelId],
}
```

The trait + ctx struct are `pub(super)` — visible inside the
`executor` module tree but not publicly exported. v0.6.5 slice 3
keeps the surface internal; slice 4 / 5 may promote it later.

### `SkewScoreSource` sub-seam (locked, per review)

A small mockable trait between the cost model and
`CudaKernelProvider` so trait-swap unit tests don't need a CUDA
fixture:

```rust
/// v0.6.5 slice 3 — abstraction over the GPU skew classifier
/// for cost-model unit testing. Production wiring uses
/// `CudaKernelProvider` as the real impl; trait-swap tests use
/// a stub that returns a configured score.
pub(super) trait SkewScoreSource {
    fn triangle_skew_score(
        &self,
        e_xy: &CudaBuffer,
        e_yz: &CudaBuffer,
        e_xz: &CudaBuffer,
        launch_stream: StreamId,
        width: WcojKeyWidth,
    ) -> Result<Option<f64>>;

    fn cycle4_skew_score(
        &self,
        e1: &CudaBuffer,
        e2: &CudaBuffer,
        e3: &CudaBuffer,
        e4: &CudaBuffer,
        launch_stream: StreamId,
        width: WcojKeyWidth,
    ) -> Result<Option<f64>>;
}

/// Production impl: dispatches to the existing
/// `wcoj_*_skew_score_*` provider entries. Verbatim wrap of the
/// current inline logic.
impl SkewScoreSource for CudaKernelProvider {
    fn triangle_skew_score(...) -> Result<Option<f64>> {
        match width {
            WcojKeyWidth::FourByte => self.wcoj_triangle_skew_score_u32(e_xy, e_yz, e_xz, launch_stream),
            WcojKeyWidth::EightByte => self.wcoj_triangle_skew_score_u64(e_xy, e_yz, e_xz, launch_stream),
        }
    }
    fn cycle4_skew_score(...) -> Result<Option<f64>> { /* same shape */ }
}
```

The trait stays internal (`pub(super)`); slice 4/5 promote if
they need the seam wider.

### Default impl: `SkewClassifierCostModel`

```rust
pub(super) struct SkewClassifierCostModel {
    triangle_threshold: f64,
    cycle4_threshold: f64,
}

impl Default for SkewClassifierCostModel {
    fn default() -> Self {
        Self {
            triangle_threshold: WCOJ_ADAPTIVE_SKEW_THRESHOLD,
            cycle4_threshold: WCOJ_ADAPTIVE_4CYCLE_SKEW_THRESHOLD,
        }
    }
}

impl WcojCostModel for SkewClassifierCostModel {
    fn should_dispatch_triangle(
        &self,
        ctx: &WcojDispatchCtx,
        scorer: &dyn SkewScoreSource,
    ) -> bool {
        // Verbatim copy of the existing logic in
        // try_dispatch_wcoj_triangle's adaptive branch, with the
        // provider call routed through SkewScoreSource.
        let score = scorer.triangle_skew_score(
            ctx.buffers[0], ctx.buffers[1], ctx.buffers[2],
            ctx.launch_stream, ctx.width,
        );
        match score {
            Ok(Some(s)) => s >= self.triangle_threshold,
            _ => false,
        }
    }

    fn should_dispatch_4cycle(
        &self,
        ctx: &WcojDispatchCtx,
        scorer: &dyn SkewScoreSource,
    ) -> bool {
        let score = scorer.cycle4_skew_score(
            ctx.buffers[0], ctx.buffers[1], ctx.buffers[2], ctx.buffers[3],
            ctx.launch_stream, ctx.width,
        );
        match score {
            Ok(Some(s)) => s >= self.cycle4_threshold,
            _ => false,
        }
    }
}
```

The `_ => false` arm handles `Ok(None)` (classifier failure) and
`Err(_)` (kernel error) the same way the existing inline code does
— silent fall-back to binary-join.

The trait shape changes from the initial sketch: `should_dispatch_*`
now takes `(&WcojDispatchCtx, &dyn SkewScoreSource)`. The ctx no
longer needs a `&CudaKernelProvider` field — it's accessed via the
scorer trait. `WcojDispatchCtx` keeps `stats`, `launch_stream`,
`width`, `buffers`, `slot_rels`.

### Integration into existing dispatch paths

In `try_dispatch_wcoj_triangle` and `try_dispatch_wcoj_4cycle`,
the adaptive branch's classifier call is replaced by:

```rust
if mode == DispatchMode::Adaptive {
    let model = SkewClassifierCostModel::default();
    let ctx = WcojDispatchCtx {
        stats: &self.stats,
        launch_stream,
        width,
        buffers: &[buf_xy, buf_yz, buf_xz],   // or 4-cycle's 4 bufs
        slot_rels: &[matched.rel_xy, matched.rel_yz, matched.rel_xz],
    };
    // CudaKernelProvider impls SkewScoreSource (production wiring).
    if !model.should_dispatch_triangle(&ctx, self.provider.as_ref()) {
        return Ok(None);
    }
}
```

This is a **mechanical refactor**: the body of the adaptive branch
moves into the trait impl. Inputs and outputs at every call site
are preserved bit-for-bit.

### Optimizer's `selectivity_pass`

Inline `pub mod selectivity_pass` inside `crates/xlog-logic/src/optimizer.rs`:

```rust
pub mod selectivity_pass {
    use xlog_ir::ExecutionPlan;
    use xlog_stats::StatsManager;

    /// v0.6.5 slice 3 — selectivity-aware optimizer pass.
    ///
    /// **No-op by default.** Slice 3 lays the seam; slices 4 / 5
    /// may add real reordering logic that consults `stats` to
    /// pick join orderings on selectivity.
    ///
    /// Walks `plan.rules_by_scc[*].body` and rewrites nodes in
    /// place. The default no-op preserves every existing plan
    /// tree byte-for-byte. Slice 3 tests assert structural
    /// equality (via `format!("{:?}", body)` or equivalent
    /// shape walker) before and after the pass.
    pub fn run(plan: &mut ExecutionPlan, _stats: &StatsManager) {
        // No-op default. Real reordering lands in a future slice
        // alongside its bench-evidence + cert updates.
        let _ = plan;
    }
}
```

### Compile pipeline ordering (LOAD-BEARING)

Per slice-review constraint, the exact post-amendment ordering
inside `Compiler::compile_program_with_stats_snapshot` is:

```
lower (already done upstream of this fn)
  → optimizer.optimize loop          (existing, line 278–282)
  → selectivity_pass::run             (NEW, slice 3)
  → promote::promote_multiway         (existing, line 289)
```

This preserves the slice 1 invariant that the promoter runs
**after** the optimizer; selectivity_pass slots between them.
Order is locked because:

1. `optimize` runs predicate-pushdown rewrites that produce the
   canonical lowered+optimized tree the promoter recognizes.
2. `selectivity_pass` is no-op in slice 3, so order vs. promote
   doesn't matter for behavior; locking the order now stops any
   future slice from accidentally moving `promote_multiway`
   before `selectivity_pass` (which would force the promoter to
   handle reordered RIR shapes that aren't in its canonical-shape
   matcher).
3. `promote_multiway` rewrites bodies to `MultiWayJoin`; future
   selectivity work that needs to walk pre-promotion shapes
   already runs before promotion.

Stats-access pattern at the call site:

```rust
// existing
let stats_arc = Arc::new(mgr);
let mut optimizer = Optimizer::new(Arc::clone(&stats_arc));
optimizer.set_schemas(schemas_by_rel_id);
for rules in &mut plan.rules_by_scc {
    for rule in rules {
        rule.body = optimizer.optimize(rule.body.clone());
    }
}

// NEW slice 3 — between optimizer and promote
crate::optimizer::selectivity_pass::run(&mut plan, &stats_arc);

// existing
crate::promote::promote_multiway(&mut plan);
```

The `stats_arc` rebind (vs. the current `Arc::new(mgr)` inline) is
the minimal refactor needed to keep `&StatsManager` reachable
after `optimizer` consumes it. The cost-model seam in
`Executor` reads `&self.stats` directly (no Arc gymnastics on
the runtime side).

The slice 1 promoter invariant (matches the *post-optimizer*
shape, runs idempotently, skips recursive SCCs) is unchanged.

### Stats reference flow

`Executor` already holds `stats: StatsManager` (verified in
`executor/mod.rs:141`). The dispatch path constructs
`WcojDispatchCtx { stats: &self.stats, ... }` — no new field on
`Executor`, no new lifetime work.

`Compiler` already constructs `Optimizer::new(Arc::new(mgr))`
where `mgr` is the per-compile stats manager. Selectivity pass
receives the same `&StatsManager` via the call site.

## Surface Changes (precise)

| File | Change |
|---|---|
| `crates/xlog-runtime/src/executor/mod.rs` | Add `pub(super) mod wcoj_cost_model;` |
| `crates/xlog-runtime/src/executor/wcoj_cost_model.rs` (new) | Trait + ctx struct + `SkewClassifierCostModel` default impl |
| `crates/xlog-runtime/src/executor/wcoj_dispatch.rs` | Replace inline adaptive branch with `SkewClassifierCostModel::default().should_dispatch_*(&ctx)`. Preserves identical behavior. |
| `crates/xlog-logic/src/optimizer/mod.rs` (new module split) OR `optimizer.rs` | Add `selectivity_pass` module + `pub fn selectivity_pass(plan, stats)` no-op |
| `crates/xlog-logic/src/compile.rs` | After `optimizer.optimize` loop and after `promote_multiway`, invoke `selectivity_pass(&mut plan, &stats_mgr)`. One-line addition. |
| `crates/xlog-runtime/src/executor/wcoj_cost_model.rs` mod tests | Trait swap test (custom impl returns alternate decision) |
| `crates/xlog-logic/src/optimizer/selectivity_pass.rs` mod tests OR existing optimizer tests | Pass-is-no-op test |

Estimated LOC: ~250 new (trait + impl + tests + selectivity stub),
~50 modified (existing dispatch call sites).

## Tests

### RED-first surface

1. **`xlog-runtime/executor/wcoj_cost_model.rs` mod tests:**
   * `default_skew_classifier_dispatches_triangle_above_threshold` — set up a fake `WcojDispatchCtx` with a mock provider returning a score above 0.10; default impl returns `true`.
   * `default_skew_classifier_falls_back_below_threshold` — score below 0.10 returns `false`.
   * `default_skew_classifier_handles_classifier_error` — provider returns `Err(_)`; default impl returns `false` (silent fall-back contract).
   * `custom_cost_model_can_override_default` — define a `AlwaysTrueCostModel`; demonstrate the trait swap works in isolation.
   (Mock provider impl: minimal stub that returns a configured `Result<Option<f64>>`; lives in the same test module.)

2. **`xlog-logic/src/optimizer/selectivity_pass.rs` mod tests:**
   * `selectivity_pass_is_noop_for_triangle_plan` — compile a triangle program, snapshot the body, run `selectivity_pass`, assert byte-equal body.
   * `selectivity_pass_is_noop_for_4cycle_plan` — same for 4-cycle.
   * `selectivity_pass_is_noop_for_recursive_scc` — ensures recursive bodies aren't perturbed.

3. **`xlog-integration` regression — existing WCOJ cert tests preserve dispatch counters bit-for-bit:**
   No new test file. The slice 2 cert suite (`test_wcoj_*`) is the regression gate. **Acceptance: every existing dispatch counter assertion passes unchanged.** If any counter shifts, the slice has a behavior change and is rejected.

4. **`xlog-logic/src/promote.rs` mod tests** (existing) continue to pass: the selectivity pass after `promote_multiway` does not perturb promoted nodes.

### GREEN gate

* `cargo test -p xlog-runtime --lib --release wcoj_cost_model` — new tests green.
* `cargo test -p xlog-logic --lib --release selectivity_pass` — new tests green.
* `cargo test --workspace --release --exclude pyxlog` — workspace green; **dispatch counters in existing cert tests must match v0.6.5 slice 2 numbers exactly**.
* `XLOG_USE_DEVICE_RUNTIME=1 cargo test -p xlog-integration --test real_world_tests --release` — 13/13.
* CUDA cert suite: 1/1.
* `cargo fmt --all -- --check` clean.

## Out of Scope (locked, per S1)

* **No threshold replacement.** `WCOJ_ADAPTIVE_*_SKEW_THRESHOLD` constants remain at `0.10`.
* **No selectivity-driven binary-join reordering.** Lowerer's bushy planner keeps its current heuristic.
* **No default-on adaptive flip** for triangle or 4-cycle.
* **No recursive WCOJ** integration.
* **No general-arity / 4-way kernel changes.**
* **No heat statistics consumption** in the seam.
* **No new `RuntimeConfig` knobs**, no env vars, no public API beyond the trait + selectivity_pass entry.
* **No release tag.**

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Refactoring the inline adaptive branch into the trait introduces subtle bit-level differences (e.g. how `Ok(None)` vs `Err(_)` are handled). | The default impl is a verbatim copy of the inline match; both paths return `false` for both cases. Slice 2 cert tests are the regression gate — any divergence shows up as a counter mismatch. |
| `selectivity_pass` invocation order vs `promote_multiway` matters in some edge case. | No-op pass is order-insensitive. Slice 4/5, when adding real reordering, will lock the order in the call site (likely `optimize → selectivity_pass → promote`). The no-op test pins the slice 3 contract. |
| `WcojDispatchCtx` struct shape locks slice 4/5 into a specific input set. | The struct is `pub(super)` and intentionally minimal. Slice 4/5 can extend it without breaking external consumers (there are none). |
| Trait swap unit tests need a mock `CudaKernelProvider`, which is a heavy type. | Mock the higher-level interface: introduce a mockable `SkewScoreSource` trait the cost model consults, with `CudaKernelProvider` as one impl. Trait tests use a stub impl of `SkewScoreSource`. The `WcojCostModel` trait's signature stays in terms of `WcojDispatchCtx`, but its default impl delegates to whatever `SkewScoreSource` is wired in. **Note: this is a small additional seam — confirm during implementation review whether to land it now or defer to slice 4.** |

## Build Sequence (when implementation starts)

1. Spawn worktree `feat/v065-cost-model-foundation` off `c4e6b3d6`.
2. **Trait + default impl + mod tests:** add `wcoj_cost_model.rs` + tests. Cargo green; no integration yet.
3. **Optimizer selectivity_pass module + tests:** add the no-op + tests. Cargo green; not yet wired into Compiler.
4. **Wire selectivity_pass into Compiler:** one-line addition after `optimizer.optimize` loop. Cargo green; no behavior change because pass is no-op.
5. **Migrate inline adaptive branches to use the cost model:** replace the inline match in `try_dispatch_wcoj_triangle` and `try_dispatch_wcoj_4cycle`'s adaptive arms with `SkewClassifierCostModel::default().should_dispatch_*(&ctx)`. Run the WCOJ cert suite — every counter must match slice 2 baseline.
6. Workspace gate: `cargo test --workspace --release`, real_world tests under `XLOG_USE_DEVICE_RUNTIME=1`, CUDA cert suite, fmt, build.
7. FF-merge to local main. STOP. No push, no tag.

Each step is its own commit. Step 5 is the load-bearing
behavior-preservation block; cert suite must run between commits.

## Acceptance

* All existing tests remain green (slice 2 contract preserved).
* All new tests pass.
* Workspace build clean, no warnings.
* `cargo fmt --all -- --check` clean.
* **Every WCOJ dispatch counter in the slice 2 cert suite reads identical to slice 2 baseline.** This is the load-bearing acceptance.
* No new public API beyond `WcojCostModel` trait (pub(super)) + `pub fn selectivity_pass(...)`.
* No production-code behavior changes. Lowerer / promoter / kernel paths unchanged.

## Locked Decisions (post-review)

All five plan-review questions resolved:

1. **`SkewScoreSource` sub-seam:** **landed now**, internal `pub(super)`. Mockable trait between cost model and `CudaKernelProvider` keeps trait-swap unit tests free of CUDA fixtures.
2. **`selectivity_pass` location:** **inline `pub mod` inside `optimizer.rs`**. Splits cleanly later if needed; minimizes slice 3 diff.
3. **Trait visibility:** **`pub(super)`** for both `WcojCostModel` and `SkewScoreSource`. Slice 4/5 promote if/when they have a real consumer.
4. **Stats injection:** **read `&self.stats` directly** from `Executor`. No `Option<&StatsManager>` plumbing.
5. **Compiler test fixtures:** **reuse** the slice 1 / slice 2 triangle and 4-cycle source programs. Tests assert byte/shape preservation (Debug-format equality before vs. after the no-op pass).

## Test Strengthening (per review)

Slice review locked: selectivity_pass tests must **assert
byte/shape preservation**, not just "doesn't crash." Concretely:

```rust
#[test]
fn selectivity_pass_is_noop_for_triangle_plan() {
    let mut compiler = Compiler::new();
    let plan = compiler.compile("tri(X,Y,Z) :- e1(X,Y), e2(Y,Z), e3(X,Z).")
        .expect("compile");
    let snapshot_before: Vec<String> = plan.rules_by_scc.iter()
        .flatten()
        .map(|r| format!("{:?}", r.body))
        .collect();

    // Re-run the pass on the same plan in isolation. The current
    // Compiler invocation already includes selectivity_pass; this
    // re-runs it and asserts idempotence.
    let stats = StatsManager::new();
    let mut plan2 = plan.clone();
    selectivity_pass::run(&mut plan2, &stats);
    let snapshot_after: Vec<String> = plan2.rules_by_scc.iter()
        .flatten()
        .map(|r| format!("{:?}", r.body))
        .collect();

    assert_eq!(snapshot_before, snapshot_after,
        "selectivity_pass must preserve every rule body byte-for-byte");
}
```

Same shape for the 4-cycle test and the recursive-SCC test.

Awaiting plan-review go-ahead before spawning the worktree.
