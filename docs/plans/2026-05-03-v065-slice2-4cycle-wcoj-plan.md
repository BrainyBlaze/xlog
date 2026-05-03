# v0.6.5 Slice 2 — 4-cycle WCOJ Kernels (amended)

**Date:** 2026-05-03
**Branch (proposed):** `feat/v065-4cycle-wcoj`
**Worktree (proposed):** `.worktrees/v065-4cycle-wcoj`
**Baseline commit:** `e22c0438` (origin/main HEAD)
**Status:** Plan, post-review amendments. Approved as **force-gate + explicit adaptive only** (no default-on in this slice). No code until plan amendments are acknowledged.

## Goal

Add a GPU-accelerated 4-cycle WCOJ operator to xlog, mirroring the
v0.6.2 triangle pipeline. Direct generalization of the 3-cycle
(triangle) work: head arity 4, four binary input edges forming a
cycle, four shared variables.

Target rule:

```text
cycle4(W, X, Y, Z) :- e1(W, X), e2(X, Y), e3(Y, Z), e4(Z, W).
```

Scope ships in one slice: u32 + u64 + Symbol kernels, force-gate
dispatch, **explicit adaptive opt-in** dispatch (not default-on),
layout fast-path reuse, phase-timing scaffolding, cert tests, bench
+ baseline evidence, and a slice-1-style promoter that emits
`RirNode::MultiWayJoin` for the canonical lowered shape.

**Default-on behavior is out of slice.** Default-on for triangle
required a separate sub-commit after baseline evidence proved
adaptive parity on every cell. 4-cycle follows the same cadence —
this slice ends with adaptive opt-in; the default-on flip is a
follow-up slice driven by the bench evidence landed here.

## Naming (per slice 2 walker contract)

Shape-locked helpers carry an explicit `_4cycle` qualifier:

* `match_multiway_4cycle` (executor matcher)
* `try_promote_4cycle` (logic promoter)
* `wcoj_4cycle_count` / `wcoj_4cycle_materialize` (CUDA, u32)
* `wcoj_4cycle_count_u64` / `wcoj_4cycle_materialize_u64`
* `wcoj_4cycle_skew_histogram_u32` / `wcoj_4cycle_skew_histogram_u64` (combined classifier across all 4 join positions, mirrors the single `wcoj_triangle_skew_histogram_*` per width)
* `wcoj_4cycle_u32_recorded` / `wcoj_4cycle_u64_recorded` (provider entries)
* `wcoj_4cycle_skew_score_u32` / `wcoj_4cycle_skew_score_u64` (provider entries that wrap the skew histogram + reduction)

The triangle-specific helpers remain unchanged. 4-cycle is **additive**, not a replacement.

## Canonical Lowered Shape (verified against the lowerer)

A test probe against `Compiler::compile()` produced the post-optimizer
tree for the target rule:

```text
Project {
    columns: [Column(0), Column(1), Column(3), Column(5)],   // (W, X, Y, Z)
    input: Join {                                            // outer
        join_type: Inner,
        left_keys: [0, 3],   // outer-left col 0 = W, col 3 = Y
        right_keys: [3, 0],  // outer-right col 3 = W, col 0 = Y
        left:  Join {                                        // (W, X) ⋈ (X, Y) on X
            join_type: Inner,
            left_keys: [1], right_keys: [0],
            left:  Scan(e1), right: Scan(e2),
        },
        right: Join {                                        // (Y, Z) ⋈ (Z, W) on Z
            join_type: Inner,
            left_keys: [1], right_keys: [0],
            left:  Scan(e3), right: Scan(e4),
        },
    },
}
```

This is **bushy** (both children of the outer Join are Joins),
unlike triangle which is **left-deep**. The promoter and matcher
must validate this exact tree.

## Architecture

### Slot order and variable classes

The MultiWayJoin emitted by `try_promote_4cycle`:

```rust
RirNode::MultiWayJoin {
    inputs: vec![
        RirNode::Scan { rel: rel_e1 },
        RirNode::Scan { rel: rel_e2 },
        RirNode::Scan { rel: rel_e3 },
        RirNode::Scan { rel: rel_e4 },
    ],
    slot_vars: vec![
        vec![Some(0), Some(1)],  // e1: (W, X) = (V_W, V_X)
        vec![Some(1), Some(2)],  // e2: (X, Y) = (V_X, V_Y)
        vec![Some(2), Some(3)],  // e3: (Y, Z) = (V_Y, V_Z)
        vec![Some(3), Some(0)],  // e4: (Z, W) = (V_Z, V_W)
    ],
    output_columns: vec![
        ProjectExpr::Column(0),  // W
        ProjectExpr::Column(1),  // X
        ProjectExpr::Column(3),  // Y
        ProjectExpr::Column(5),  // Z
    ],
    fallback: Box::new(<original Project { Join { Join, Join } } tree>),
}
```

Strict matcher invariants:

* `slot_vars[0] = [a, b]` with `a != b`
* `slot_vars[1] = [b, c]` (shared `b`); `c != a`, `c != b`
* `slot_vars[2] = [c, d]` (shared `c`); `d ∉ {a, b, c}`
* `slot_vars[3] = [d, a]` (shared `d` and shared `a` — closes the cycle)
* `output_columns == [Column(0), Column(1), Column(3), Column(5)]` exactly

### CUDA kernel design

Count-then-materialize, mirrors triangle:

For each edge `(w, x)` in slot 0:
1. Probe slot 1 for matches on `x` → candidate triples `(w, x, y)`.
2. For each `y`, probe slot 2 for matches on `y` → candidate quads `(w, x, y, z)`.
3. For each `z`, probe slot 3 for `(z, w)` to close the cycle.

Two phases:

* **Count phase** (`wcoj_4cycle_count`): parallel over slot-0 rows; per-thread per-block accumulators.
* **Device exclusive scan**: reuses the existing
  `multiblock_scan_u32_inplace_on_stream` helper that triangle's
  `wcoj_triangle_u32_recorded` already uses on its `count_buf →
  offsets_buf` transition (`provider/wcoj.rs:527`). 4-cycle reuses
  the same primitive — no new scan kernel.
* **Materialize phase** (`wcoj_4cycle_materialize`): re-iterate slot 0 with offsets, emit `(w, x, y, z)` quads.

Layout invariant (sorted lex over 2 cols + deduped per slot) is
established by the existing `wcoj_layout_*_recorded` helpers,
unchanged.

### Width branching

Mirrors triangle: 4-byte (U32 / Symbol) → `wcoj_4cycle_*_u32_recorded`; 8-byte (U64) → `wcoj_4cycle_*_u64_recorded`. Mixed-width across the four slots falls back. The existing `WcojKeyWidth` enum and `classify_two_col_wcoj_width` helper extend trivially.

### Adaptive skew classifier

4-cycle has 4 join positions:

* `J_X`: slot 0 col 1 ⋈ slot 1 col 0
* `J_Y`: slot 1 col 1 ⋈ slot 2 col 0
* `J_Z`: slot 2 col 1 ⋈ slot 3 col 0
* `J_W`: slot 3 col 1 ⋈ slot 0 col 0

`wcoj_4cycle_skew_histogram_*` computes per-position skew scores
(same hash-mixed bucket math as `wcoj_triangle_skew_histogram_*`).
The provider's `wcoj_4cycle_skew_score_*` reduces them to a single
scalar.

**Reduction: `max(score_J_X, score_J_Y, score_J_Z, score_J_W)`.** Per
review: simple sum can exceed 1 and is not comparable to triangle's
0.10 threshold. Max keeps the score in the same `[0, 1]` range as
triangle. Constant `WCOJ_ADAPTIVE_4CYCLE_SKEW_THRESHOLD` is locked
at `0.10` to mirror the triangle threshold (matching range, matching
acceptance criterion). The bench evidence under
`docs/evidence/2026-05-?-wcoj-4cycle-bench-baseline/` verifies this
threshold has the same ≥1.7× headroom on each side; if the evidence
shows that max-reduction needs a different threshold, lock it before
landing the slice's adaptive cert tests.

If a future slice gathers evidence that sum-reduction wins on
heterogeneous skew, that threshold derivation must explicitly
account for a `[0, 4]` range (sum of four `[0, 1]` scores).

### Layout fast-path

Reused unchanged from triangle. The existing `wcoj_layout_*_recorded` accepts any 2-column input; 4-cycle dispatch invokes it 4× per call (one per slot). No new layout kernels.

### Stream and dispatch reuse

Approved per review: rename `Executor::wcoj_triangle_stream` →
`Executor::wcoj_dispatch_stream`. Single cached `OnceLock<StreamId>`
shared across triangle and 4-cycle dispatch. Mechanical rename;
landed as a separate commit before kernel work begins.

## Surface Changes

### `crates/xlog-cuda/kernels/wcoj.cu`

Add 6 new kernels:

```c
extern "C" __global__ void wcoj_4cycle_count(...);
extern "C" __global__ void wcoj_4cycle_materialize(...);
extern "C" __global__ void wcoj_4cycle_count_u64(...);
extern "C" __global__ void wcoj_4cycle_materialize_u64(...);
extern "C" __global__ void wcoj_4cycle_skew_histogram_u32(...);
extern "C" __global__ void wcoj_4cycle_skew_histogram_u64(...);
```

Estimated CUDA LOC: ~200 per main kernel × 4 + ~80 per skew × 2 ≈ 960 LOC.

### `crates/xlog-cuda/src/kernel_manifest_data.rs`

**Exactly 6 new manifest entries**, paired with the kernels above:

```text
wcoj_4cycle_count
wcoj_4cycle_materialize
wcoj_4cycle_count_u64
wcoj_4cycle_materialize_u64
wcoj_4cycle_skew_histogram_u32
wcoj_4cycle_skew_histogram_u64
```

`wcoj_compute_total` and the `wcoj_layout_check_sorted_unique_*`
kernels are reused unchanged (verified: `wcoj_compute_total` is
shape-agnostic — it sums the last entry of `counts` + `offsets`).

### `crates/xlog-cuda/src/provider/wcoj.rs`

Add provider entries:

```rust
pub fn wcoj_4cycle_u32_recorded(
    &self,
    layout_e1: &CudaBuffer,
    layout_e2: &CudaBuffer,
    layout_e3: &CudaBuffer,
    layout_e4: &CudaBuffer,
    launch_stream: StreamId,
) -> Result<CudaBuffer>;

pub fn wcoj_4cycle_u64_recorded(...) -> Result<CudaBuffer>;

pub fn wcoj_4cycle_skew_score_u32(
    &self,
    buf_e1: &CudaBuffer, buf_e2: &CudaBuffer,
    buf_e3: &CudaBuffer, buf_e4: &CudaBuffer,
    launch_stream: StreamId,
) -> Result<Option<f64>>;

pub fn wcoj_4cycle_skew_score_u64(...) -> Result<Option<f64>>;
```

Each `*_recorded` entry: launch count → `multiblock_scan_u32_inplace_on_stream` on counts → launch materialize. Estimated provider LOC: ~600.

### `crates/xlog-logic/src/promote.rs`

Add `try_promote_4cycle(node: &RirNode) -> Option<RirNode>`. Strict bushy-shape match; returns `None` for any deviation. `promote_multiway` walks all rule bodies, calls `try_promote_triangle` first, then `try_promote_4cycle`. (A body cannot match both — different atom counts — but the explicit ordering documents intent.)

Recursive SCC skip preserved (slice 1 fix).

### `crates/xlog-runtime/src/executor/wcoj_dispatch.rs`

Add:

* `match_multiway_4cycle(body: &RirNode) -> Option<FourCycleRirMatch>` — strict matcher mirroring `match_multiway_triangle`. Validates `inputs.len() == 4`, the canonical 4-cycle slot_vars layout, and the exact `output_columns = [Column(0), Column(1), Column(3), Column(5)]`.
* `FourCycleRirMatch { rel_e1, rel_e2, rel_e3, rel_e4 }` struct.
* `try_dispatch_wcoj_4cycle(rule)` mirroring `try_dispatch_wcoj_triangle`. Same gate decision tree (kill > force > adaptive opt-in > none). Same width branching. Separate dispatch counter `wcoj_4cycle_dispatch_count`.
* New env vars (**width-neutral**, not following triangle's `_U32` debt):
  * `XLOG_USE_WCOJ_4CYCLE` (force)
  * `XLOG_USE_WCOJ_4CYCLE_ADAPTIVE` (explicit adaptive opt-in)
  * `XLOG_DISABLE_WCOJ_4CYCLE` (kill switch)
* New constant `WCOJ_ADAPTIVE_4CYCLE_SKEW_THRESHOLD` = `0.10` (matches triangle).
* Rename `wcoj_triangle_stream` → `wcoj_dispatch_stream` on `Executor`.

The triangle dispatch surface is unchanged. Triangle's existing `_U32` env name stays as historical debt; this slice does NOT propagate that pattern to 4-cycle.

### `crates/xlog-runtime/src/executor/recursive.rs`

Non-recursive arm tries triangle dispatch first, then 4-cycle:

```rust
if let Some(buf) = self.try_dispatch_wcoj_triangle(rule)? { /* … */ continue; }
if let Some(buf) = self.try_dispatch_wcoj_4cycle(rule)? { /* mirror */ continue; }
// fallback descent into MultiWayJoin's `fallback` (slice 1 contract).
```

### `crates/xlog-core/src/config.rs`

Add three new `RuntimeConfig` fields mirroring the triangle ones:

```rust
pub wcoj_4cycle_dispatch: Option<bool>,           // force
pub wcoj_4cycle_dispatch_adaptive: Option<bool>,  // adaptive opt-in
pub wcoj_4cycle_dispatch_disabled: Option<bool>,  // kill switch
```

Builder methods `with_wcoj_4cycle_dispatch(...)` / `_adaptive(...)` / `_disabled(...)`.

**Adaptive default policy:** `wcoj_4cycle_dispatch_adaptive = None` resolves to `false` (opt-in). This is the explicit difference from triangle's default-on behavior — locked by the env resolver and a unit test in `wcoj_dispatch.rs::tests`.

## Cross-Crate Walker Audit

Slice 2's walker contract (committed to `RirNode::MultiWayJoin` doc) states: generic walkers shape-agnostic; only named matchers/promoters lock shape. Slice 2 already pinned this with 4-input synthesized tests in D4. **No additional walker arms needed for this slice** — adding a real 4-cycle MultiWayJoin produced by the promoter is the production exercise of those guards.

## Lowerer / Promoter Boundary

* `Lowerer::lower_program` unchanged — produces the bushy 4-cycle tree.
* `Optimizer::optimize` unchanged.
* `Compiler::compile_program_with_stats_snapshot` already invokes `promote_multiway(&mut plan)` (slice 1). Internally, the order is `try_promote_triangle` then `try_promote_4cycle`.
* `RirMeta` on `CompiledRule` preserved.

## Executor Fallback Behavior

Mirrors slice 1 exactly: dispatch → buffer + counter; decline → descend into `fallback`. Failure isolation contract unchanged.

## Tests

### RED-first surface

#### `crates/xlog-cuda/tests/` (provider/kernel correctness — alongside existing `test_wcoj_triangle_*` files)

* `tests/test_wcoj_4cycle_u32.rs` — small fixture, expected quad set; assert kernel output matches.
* `tests/test_wcoj_4cycle_u64.rs` — same fixture under U64.
* `tests/test_wcoj_4cycle_layout.rs` — confirm layout helper handles 4-cycle inputs (no kernel changes; smoke).
* `tests/test_wcoj_4cycle_skew.rs` — classifier returns sensible scores for uniform vs super-hub fixtures.

(`xlog-cuda-tests` is the **certification suite crate** — separate from these. Tests under `crates/xlog-cuda/tests/` are integration tests for `xlog-cuda` and run as part of `cargo test -p xlog-cuda`; the certification suite does NOT execute them transitively. Both gates run explicitly in the workspace test plan: `cargo test --workspace --release` covers `crates/xlog-cuda/tests/`, and `cargo test -p xlog-cuda-tests --test certification_suite --release` is the cert-suite gate. We do not place ordinary WCOJ tests in the cert suite.)

#### `xlog-logic` (promoter)

* `xlog-logic/src/promote.rs` mod tests — extend with:
  * `promotes_canonical_4cycle` (positive)
  * `rejects_4cycle_with_rotated_columns`
  * `rejects_4cycle_with_left_deep_shape`
  * `4cycle_fallback_is_structurally_equal_to_input`
  * `4cycle_idempotent`
  * `skips_recursive_scc_4cycle`
* `crates/xlog-logic/tests/test_promote_multiway.rs` — extend with a 4-cycle pipeline test asserting `MultiWayJoin` emission via the full Compiler pipeline.

#### `xlog-runtime` (matcher + dispatch)

* `executor/wcoj_dispatch.rs` mod tests — extend with:
  * `match_multiway_4cycle` positive (canonical input)
  * negatives: rotated `output_columns`, malformed `slot_vars` (each violation tested), non-Scan inputs, wrong arity (3 or 5 inputs)
  * env-resolver tests for the three new env vars
  * `adaptive_resolver_defaults_off_when_env_unset` (locks the **opt-in** policy, contrasts with triangle's default-on)

#### `xlog-integration` (cert + wiring)

* `tests/test_wcoj_4cycle_executor_wiring.rs` — gate-off vs gate-on, dispatch counter == 1, row-set matches binary-join reference.
* `tests/test_wcoj_4cycle_rir_shape_cert.rs` — RIR-shape policy table for syntactic 4-cycle variants (rotated atoms, head-var rotation, comparison filter, 3-arity head, etc. — mirrors `test_wcoj_rir_shape_cert.rs`).
* `tests/test_wcoj_4cycle_adaptive_dispatch.rs` — adaptive opt-in engages on super-hub fixtures (with `XLOG_USE_WCOJ_4CYCLE_ADAPTIVE=1`), falls back on uniform.
* `tests/test_wcoj_4cycle_dispatch_stream_reuse.rs` — confirms the cached `wcoj_dispatch_stream` is reused across triangle and 4-cycle dispatches in the same executor.

**No `test_wcoj_4cycle_adaptive_default_on.rs`** — default-on is out of slice.

#### Bench

* `benches/wcoj_4cycle_bench.rs` — mirror `wcoj_triangle_bench.rs`. {u32,u64} × {uniform, superhub, empty} × {sizes} × {Off, Force, Adaptive}. **Acceptance:** force-mode dispatches and produces correct row sets vs Off; adaptive-mode engages on super-hub and falls back on uniform; threshold selection at `0.10` clears the gap with ≥1.7× headroom on each side. **The bench does NOT gate default-on landing in this slice** — default-on is a separate follow-up slice.

### GREEN gate

* `cargo test --workspace --release` green.
* `cargo test -p xlog-cuda-tests --test certification_suite --release` green.
* `XLOG_USE_DEVICE_RUNTIME=1 cargo test -p xlog-integration --test real_world_tests --release`: 13/13.
* All new `test_wcoj_4cycle_*` tests pass; counter numbers match locked acceptance.
* `cargo fmt --all -- --check` clean.

## Out of Scope

* **Default-on adaptive dispatch.** Force + explicit adaptive only this slice.
* No K_4 (4-clique) kernel — different shape.
* No 4-path kernel — binary-join handles it.
* No general-arity kernel template — slice 5.
* No recursive 4-cycle dispatch — slice 4.
* No cost-model / variable-ordering work — slice 3.
* No `RirNode` variant changes — `MultiWayJoin` already shape-agnostic.
* No new public API beyond the dispatch knobs and provider entries.
* No release tag.
* **No retroactive renaming** of `XLOG_USE_WCOJ_TRIANGLE_U32` to drop its `_U32` suffix. That's pre-existing debt; a separate cleanup slice can address it.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| 4-cycle kernel correctness on edge cases (empty, single-row, fully-connected vs sparse). | Dedicated 4-cycle fixture (approved) — small graph with known-quantity 4-cycles. Tests for empty, single-edge, no-cycle, super-hub. Property-style fuzz against binary-join reference if scope permits. |
| Bushy lowered shape might shift under future optimizer changes. | Pin via `test_promote_multiway.rs` Compiler-pipeline test. If the optimizer ever reorders, the test fails with a clear diff. Same risk model as triangle. |
| Threshold `0.10` may not transfer cleanly under max-reduction. | Bench evidence verifies. If the gap doesn't have ≥1.7× headroom under max-reduction, lock a different value before merging. The threshold is a single constant; tuning is mechanical. |
| Stream rename touches existing slice 1 code. | Mechanical rename, single dedicated commit (step 2 in build sequence). Cert suite green between commits. |
| Promoter ordering: triangle-first vs 4-cycle-first. | Body cannot match both (different atom counts). Order doesn't affect correctness. Documented in the function and pinned by a unit test that runs both promoters on each canonical body and asserts only one fires. |

## Build Sequence

Each step is its own commit; the slice merges as a clean linear history.

1. **Spawn worktree** `feat/v065-4cycle-wcoj` off `e22c0438`.
2. **Stream rename**: `wcoj_triangle_stream` → `wcoj_dispatch_stream` on `Executor`. Cargo green; cert suite green; trivial mechanical change.
3. **CUDA u32 kernels**: `wcoj_4cycle_count`, `wcoj_4cycle_materialize`, manifest entries. Provider entry `wcoj_4cycle_u32_recorded` (count → `multiblock_scan_u32_inplace_on_stream` → materialize). `crates/xlog-cuda/tests/test_wcoj_4cycle_u32.rs` correctness.
4. **CUDA u64 kernels**: same pattern as step 3 for u64 + `crates/xlog-cuda/tests/test_wcoj_4cycle_u64.rs`.
5. **Layout reuse smoke**: `crates/xlog-cuda/tests/test_wcoj_4cycle_layout.rs` — confirms the existing `wcoj_layout_*_recorded` works on 4-cycle inputs.
6. **Promoter**: `try_promote_4cycle` + 4-cycle promote tests in `xlog-logic`.
7. **Matcher + dispatch**: `match_multiway_4cycle`, `try_dispatch_wcoj_4cycle`, force-gate-only first. Width-neutral env vars (`XLOG_USE_WCOJ_4CYCLE`, `XLOG_USE_WCOJ_4CYCLE_ADAPTIVE`, `XLOG_DISABLE_WCOJ_4CYCLE`). Config knobs. `executor/wcoj_dispatch.rs` mod tests including the **adaptive-defaults-off** lock.
8. **Wire dispatch**: `recursive.rs` non-recursive arm tries 4-cycle after triangle. `xlog-integration` wiring + RIR-shape cert tests.
9. **Skew classifier**: `wcoj_4cycle_skew_histogram_*` kernels + manifest entries (2) + provider `wcoj_4cycle_skew_score_*` (max-reduction). `crates/xlog-cuda/tests/test_wcoj_4cycle_skew.rs`. Adaptive cert tests.
10. **Bench + baseline evidence**: `benches/wcoj_4cycle_bench.rs` + `docs/evidence/2026-05-?-wcoj-4cycle-bench-baseline/`. Lock the threshold value (default `0.10`; adjust if evidence shows max-reduction needs different).
11. Workspace gate, FF-merge to local main. STOP. No push, no tag.

**No default-on commit.** That is a separate follow-up slice driven by the bench evidence landed at step 10.

## Acceptance

* `cargo test --workspace --release` green.
* `cargo test -p xlog-cuda-tests --test certification_suite --release` green.
* `XLOG_USE_DEVICE_RUNTIME=1 cargo test -p xlog-integration --test real_world_tests --release`: 13/13.
* All `test_wcoj_4cycle_*` tests pass; force-gate counter == 1 on canonical input, adaptive-opt-in counter == 1 on super-hub fixture, both counters == 0 on uniform/empty fixtures and when kill switch is on.
* Triangle behavior **unchanged** — every existing v0.6.2 + slice 1 + slice 2 (interlock) WCOJ test stays green.
* `cargo fmt --all -- --check` clean. Workspace warnings: zero new.

## Open Plan-Review Questions (final)

None. All previous open questions resolved by the amendment:

* Default-on policy: out of slice ✓
* Stream rename: shared `wcoj_dispatch_stream` ✓
* Bench framing: full matrix; acceptance is force/adaptive + threshold (not default-on) ✓
* Test fixture: dedicated 4-cycle ✓
* Adaptive fold: max(score_per_join_position) at threshold 0.10 ✓
* Env naming: width-neutral ✓
* Test crate location: `crates/xlog-cuda/tests/` ✓
* Manifest count: exactly 6 entries enumerated ✓
* Scan helper: `multiblock_scan_u32_inplace_on_stream` ✓

Awaiting acknowledgement before spawning the worktree.
