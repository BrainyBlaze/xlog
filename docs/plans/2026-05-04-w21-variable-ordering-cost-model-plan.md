# W2.1 Plan — Variable-Ordering Cost Model (Real WCOJ Slot Order)

**Closes W2.1.** No W2.3 recursive-arm work, no W2.5 default
flip, no W2.6 heat/selectivity feedback. **No kernel
signature changes**; the variable order is expressed at
dispatch time as input rotation + (triangle) column-swap +
recorded sort/dedupe via the existing
`wcoj_layout_*_recorded` helper.

**Date:** 2026-05-04
**Branch (proposed):** `feat/w21-variable-ordering-cost-model`
**Worktree (proposed):** `.worktrees/w21-variable-ordering-cost-model`
**Base:** `main` at `0c176e6a` (W2.2 closure-board commit).
**Board entry:** `docs/v065-closure-board.md` Wave 2, W2.1.

## Goal

Make the WCOJ kernel's leader (slot 0) a stats-driven
choice. Currently `inputs[0]` is hardcoded to e_xy
(triangle) or e_wx (4-cycle); W2.1 lets the smallest
populated input become the leader, reducing iteration cost.

## Permutation Tables — Locked, Not Handwaved

### Triangle (3 leaders)

Cycle topology: vars `{X, Y, Z}`, edges `e_xy=(X,Y)`,
`e_yz=(Y,Z)`, `e_xz=(X,Z)`. Kernel slot_vars pattern
`[[a,b], [b,c], [a,c]]` (slice 1 contract). Per-leader
table:

| Leader | Slot 0 | Slot 1 | Slot 2 | Lookup col-swaps | Kernel-direct output | head_proj |
|--------|--------|--------|--------|------------------|----------------------|-----------|
| **e_xy** (default) | e_xy as-is | e_yz as-is | e_xz as-is | none | (X, Y, Z) | `[0, 1, 2]` |
| **e_yz** | e_yz as-is | e_xz **swap (Z,X)** | e_xy **swap (Y,X)** | e_xz, e_xy | (Y, Z, X) | `[2, 0, 1]` |
| **e_xz** | e_xz as (X,Z) → (X is "a", Z is "b") | e_yz **swap (Z,Y)** | e_xy as (X,Y) | e_yz | (X, Z, Y) | `[0, 2, 1]` |

Verification per leader: slot 0's col1 ≡ slot 1's col0,
slot 0's col0 ≡ slot 2's col0, slot 1's col1 ≡ slot 2's
col1. (Manual trace verified for each row above.)

### 4-cycle (4 leaders, all rotation-only)

Cycle topology: vars `{W, X, Y, Z}`, edges `e_wx=(W,X)`,
`e_xy=(X,Y)`, `e_yz=(Y,Z)`, `e_zw=(Z,W)`. Kernel slot_vars
pattern `[[a,b], [b,c], [c,d], [d,a]]`.

The 4-cycle has the **rotational symmetry property**: each
edge's columns naturally fit the next slot's `(a, b)` /
`(b, c)` / etc. positions because every edge is already
sorted on its first variable in the cycle direction.
Therefore **no column swaps are needed for any 4-cycle
leader** — only input rotation.

| Leader | Slot 0 | Slot 1 | Slot 2 | Slot 3 | Col swaps | Kernel-direct output | head_proj |
|--------|--------|--------|--------|--------|-----------|----------------------|-----------|
| **e_wx** (default) | e_wx | e_xy | e_yz | e_zw | none | (W, X, Y, Z) | `[0, 1, 2, 3]` |
| **e_xy** | e_xy | e_yz | e_zw | e_wx | none | (X, Y, Z, W) | `[3, 0, 1, 2]` |
| **e_yz** | e_yz | e_zw | e_wx | e_xy | none | (Y, Z, W, X) | `[2, 3, 0, 1]` |
| **e_zw** | e_zw | e_wx | e_xy | e_yz | none | (Z, W, X, Y) | `[1, 2, 3, 0]` |

Verification per leader: each row's slot-pair shares the
correct cycle variable (e.g., e_xy.col1 = Y = e_yz.col0 in
the e_xy-leader row). All four traces pass without column
swap.

### Triangle layout-overhead caveat

Triangle leaders e_yz and e_xz require column swaps on
lookup atoms. Each swapped atom needs a 2-col projected
view, and the layout (`wcoj_layout_*_recorded`) re-sorts
that view by its NEW col0. This is one re-sort per
swapped lookup atom (max 2 for triangle, 0 for 4-cycle).

## Sorted-Layout Compliance — Owned 2-Col Projection

Per review, the dispatcher cannot raw-pointer-swap columns
(layout requires sorted col0). The W2.1 dispatcher uses
**owned 2-col projected views** materialized via a new
recorded helper:

```rust
// In xlog_cuda::CudaKernelProvider:
pub fn wcoj_project_2col_swap_recorded(
    &self,
    src: &CudaBuffer,
    stream: StreamId,
) -> Result<CudaBuffer>
```

Semantics:
* Input: 2-col `CudaBuffer` (any width).
* Output: 2-col owned `CudaBuffer` with cols swapped:
  output.col(0) = DtoD-copy(src.col(1)),
  output.col(1) = DtoD-copy(src.col(0)). Schema reflects
  the swap (column names re-derived).
  `cached_row_count` carried over from `src` (logical row
  count is invariant under column permutation).
  `num_rows_device` allocated as a fresh device scalar
  initialized via DtoD-copy from `src.num_rows_device()`
  so the produced buffer is a self-contained owned
  `CudaBuffer` and the executor's downstream readers see a
  valid device row count.
* Recorded: every DtoD-copy is declared on a
  `LaunchRecorder`, then `preflight(runtime)` is called.
  Both column buffers and the new `num_rows_device` are
  declared as writes; `src` columns and `src.num_rows_device()`
  are declared as reads.
* **Failure drain**: matches the slice 2 / W2.4
  launch-stream safety pattern (`wcoj.rs` ≈ line 2140 +
  `commit/preflight` discipline). Any error after the first
  queued copy must `cu_stream.synchronize()` before the
  function returns, because the partially-allocated output
  buffers are about to drop and an in-flight DtoD copy
  would race the runtime dealloc.
* No sort happens here — sorting is the layout helper's
  job (called next).

The dispatcher then calls
`wcoj_layout_*_recorded(projected_view, stream)` to obtain
the sorted+deduped layout the kernel expects.

**Lifetime/aliasing**: the projected view OWNS new column
buffers (DtoD-copy from source), so source/projected/layout
buffers don't alias. Standard `LaunchRecorder` write
declarations apply per buffer.

**Tests**: 6 unit tests in `xlog-cuda` cert covering
`wcoj_project_2col_swap_recorded`:
* u32 round-trip: project then un-project equals original.
* u64 round-trip.
* Symbol round-trip (4-byte path; WCOJ Symbol parity).
* Empty buffer (n=0) → empty output, num_rows_device == 0.
* Schema swap: column names reflect the swap.
* `cached_row_count` and `num_rows_device` device scalar
  match `src` after swap.

## Output Buffer — Owned Projected Result, Not Accessor Remap

Per review, the W2.1 output projection materializes a NEW
head-ordered output buffer. It does NOT redefine
`CudaBuffer::column(i)` semantics. The dispatcher applies a
column-reorder helper post-kernel:

```rust
pub fn wcoj_project_output_columns_recorded(
    &self,
    src: &CudaBuffer,
    perm: &[usize],
    head_schema: Schema,
    stream: StreamId,
) -> Result<CudaBuffer>
```

Semantics:
* Input: kernel-direct output buffer (cols in
  kernel-leader order).
* Output: owned buffer with cols re-permuted per `perm`,
  schema replaced with `head_schema`. Columns are
  DtoD-copied in the new order.
* `cached_row_count` carried over from `src` (logical row
  count is invariant under column permutation; the kernel
  has already computed it).
* `num_rows_device` allocated as a fresh device scalar
  initialized via DtoD-copy from `src.num_rows_device()`
  so the produced buffer is self-contained and the
  executor's downstream readers see a valid device row
  count (mirrors the swap helper's discipline).
* Recorded: every per-column DtoD-copy and the
  `num_rows_device` copy is declared on a
  `LaunchRecorder`; `preflight(runtime)` runs before any
  copy is issued.
* **Failure drain**: matches the slice 2 / W2.4
  launch-stream safety pattern. Any error after the first
  queued copy must `cu_stream.synchronize()` before the
  function returns, because the partially-allocated output
  buffer is about to drop and an in-flight DtoD copy
  would race the runtime dealloc.
* Lifetimes: source buffer drops after the projection;
  result buffer is what the executor stores in the
  relation table. No aliasing.

Default leader (`var_order = None`) uses
`output_columns` (slice 1/2 binary-fallback projection)
unchanged — no W2.1 output-projection helper called. **Slice
1/2/W2.2 regression bit-identical.**

**Tests**: 5 unit tests in `xlog-cuda` cert for
`wcoj_project_output_columns_recorded`:
* u32 round-trip with permutation `[2, 0, 1]` (triangle
  e_yz-leader head_proj).
* u64 round-trip with permutation `[3, 0, 1, 2]` (4-cycle
  e_xy-leader head_proj).
* Symbol round-trip with permutation (4-byte path).
* `cached_row_count` and `num_rows_device` device scalar
  match `src` after permutation; identity permutation
  produces a buffer equal-by-content to `src`.
* **Empty output (n=0)** with a non-identity permutation:
  schema is the requested `head_schema`,
  `cached_row_count == Some(0)`, `num_rows_device` device
  scalar reads back as `0`, no per-column allocations
  fail. WCOJ legitimately produces empty output and the
  helper must not divide-by-zero or fail-fast.

## In Scope

* **IR change** (`xlog-ir`):
  - `RirNode::MultiWayJoin` gains
    `var_order: Option<VariableOrder>`.
  - New types `VariableOrder { leader_idx: u8,
    lookup_perms: Vec<LookupPerm>, kernel_output_cols:
    Vec<ProjectExpr> }` and `LookupPerm { input_idx: u8,
    swap_cols: bool }`.
  - All slice 1–4 + W2.2 walkers/matchers ignore the new
    field. Default `None` preserves all prior behavior.
* **Cost model** (`xlog-logic`):
  - `WcojVariableOrderingModel` trait + default impl
    `LeaderCardinalityModel`.
  - **Threshold gate (per review)**: leader chosen ONLY
    when `min_card / default_leader_card ≤
    config.effective_wcoj_var_ordering_threshold()`. The
    promoter MUST call the resolver method, NOT read the
    raw `wcoj_var_ordering_threshold` field, so
    out-of-range struct-literal values fall back to
    `DEFAULT_THRESHOLD` rather than silently widening the
    gate. Marginal cases (ratio above the resolved
    threshold) leave `var_order = None`.
* **Promoter wiring** (`xlog-logic`): `promote_multiway`
  signature change to `(plan, rel_ids, stats, config)`.
  Caller audit list below.
* **CompilerConfig + composable API**:
  - ```rust
    pub struct CompilerConfig {
        pub wcoj_variable_ordering: WcojVarOrderingKind,
        /// Raw threshold field. Public to keep struct
        /// literal construction available. Promoter MUST
        /// NOT read this field directly; it MUST go
        /// through `effective_wcoj_var_ordering_threshold()`
        /// so out-of-range values are clamped at use,
        /// not silently honored.
        pub wcoj_var_ordering_threshold: f64,
    }

    pub enum WcojVarOrderingKind {
        /// W2.1 disabled: promoter never sets
        /// `var_order`; behavior bit-identical to slice
        /// 1/2/4 + W2.2.
        Disabled,
        /// Use the default `LeaderCardinalityModel`.
        LeaderCardinality,
    }

    impl Default for CompilerConfig {
        fn default() -> Self {
            Self {
                wcoj_variable_ordering:
                    WcojVarOrderingKind::Disabled,
                wcoj_var_ordering_threshold:
                    Self::DEFAULT_THRESHOLD,
            }
        }
    }

    impl CompilerConfig {
        pub const DEFAULT_THRESHOLD: f64 = 0.5;

        /// Resolves the threshold the promoter actually
        /// uses. NaN, ≤ 0.0, or > 1.0 fall back to
        /// `DEFAULT_THRESHOLD`. Struct-literal callers
        /// CAN'T silently disable the gate by setting
        /// `f64::INFINITY`; the resolver caps it.
        pub fn effective_wcoj_var_ordering_threshold(
            &self,
        ) -> f64 {
            let t = self.wcoj_var_ordering_threshold;
            if !t.is_finite() || t <= 0.0 || t > 1.0 {
                Self::DEFAULT_THRESHOLD
            } else {
                t
            }
        }
    }
    ```
  - **Public fields, use-site clamping.** Fields stay
    public so callers can construct via struct literal,
    but the promoter MUST call
    `effective_wcoj_var_ordering_threshold()` rather than
    reading the field directly. This is the only contract
    that survives an arbitrary literal.
  - **Default disables W2.1.** `CompilerConfig::default()`
    sets `wcoj_variable_ordering = Disabled`, so
    `compile()` and `compile_with_stats_snapshot(...)`
    behave bit-identically to pre-W2.1. **No env opt-in
    in this slice; no env work is committed to a future
    slice either** — adding env precedence is out of
    W2.1 scope and would require a new closure-board
    item before being referenced. Activation requires
    explicitly constructing a `CompilerConfig` with
    `LeaderCardinality` and calling
    `compile_with_config_and_stats_snapshot(...)`.
  - **Precedence**: caller-supplied `&CompilerConfig`
    only. No env override, no global default
    side-channel.
  - `Compiler::compile_with_config_and_stats_snapshot(...)`.
  - `compile()` and `compile_with_stats_snapshot(...)`
    delegate via `CompilerConfig::default()` →
    `Disabled` → no W2.1 activation.
  - **Resolver tests** (xlog-logic): 4 unit tests on
    `effective_wcoj_var_ordering_threshold`:
    * valid in-range value `0.3` → `0.3`.
    * `0.0` → `DEFAULT_THRESHOLD` (boundary, not allowed).
    * `1.5` → `DEFAULT_THRESHOLD` (above 1.0).
    * `f64::NAN` → `DEFAULT_THRESHOLD`.
* **New CUDA helpers** (`xlog-cuda`):
  - `wcoj_project_2col_swap_recorded` (triangle col-swap).
  - `wcoj_project_output_columns_recorded` (head
    projection).
* **Dispatcher reroute** (`xlog-runtime`):
  - Triangle dispatch site (`wcoj_dispatch.rs:293`) +
    4-cycle (`:392`) become per-leader.
  - New `prepare_leader_inputs(matched, var_order, stream)`
    helper builds the per-leader layouts (rotation +
    optional col-swap + sort).
  - Post-kernel: if `var_order.is_some()`, apply
    `wcoj_project_output_columns_recorded` to remap to
    head order.

## Caller Audit — `promote_multiway` Signature Change

Pre-W2.1 signature:
`promote_multiway(plan: &mut ExecutionPlan, rel_ids:
&HashMap<String, RelId>)`.

Post-W2.1 signature:
`promote_multiway(plan: &mut ExecutionPlan, rel_ids:
&HashMap<String, RelId>, stats: &StatsManager, config:
&CompilerConfig)`.

**Callers** (verified by `grep -rn 'promote_multiway('
crates/` 2026-05-04 — corrected after iteration 4 review):

1. **`crates/xlog-logic/src/compile.rs:304`** —
   production caller. Passes `self.lowerer.rel_ids()`
   today; W2.1 also passes the compile-time
   `&stats_arc` (already in scope) and `&config`. The
   config is threaded through
   `Compiler::compile_with_config_and_stats_snapshot(...)`
   (the new entry point). The legacy `compile()` and
   `compile_with_stats_snapshot(...)` entry points
   delegate to the new entry point with
   `CompilerConfig::default()`. **No `Compiler` struct
   field** is added — the config is a per-call argument.
2. **`crates/xlog-logic/src/promote.rs::tests`** — **23
   in-crate test sites** (verified by `grep -c`). Existing
   tests pass `&HashMap::new()` or `&rel_ids` for
   rel_ids; W2.1 adds `&StatsManager::new()` and
   `&CompilerConfig::default()`. **Default behavior**:
   empty stats → safety floor → `var_order = None` →
   identical to pre-W2.1. All slice 4 / W2.2 promoter
   tests bit-identical.
3. **`crates/xlog-integration/tests/test_selectivity_pass_reordering.rs:648`**
   — **cross-crate caller** (W2.2 Part C synthesized
   post-selectivity body harness). The W2.2 cert calls
   `xlog_logic::promote::promote_multiway(&mut plan,
   compiler.rel_ids())`. W2.1 step 5 updates this single
   call site to pass `&StatsManager::new()` and
   `&CompilerConfig::default()` so the W2.2 acceptance
   gate stays green; row sets remain bit-identical. The
   step-5 commit references both W2.1 and W2.2.
4. **`crates/xlog-logic/tests/test_promote_multiway.rs`**
   — file exists but has zero `promote_multiway(` call
   sites today (verified by `grep -c`). No update needed.
5. **No `xlog-runtime` / `xlog-cuda` callers** verified
   by the same grep — only doc-comment references in
   `wcoj_dispatch.rs:29`.

Total signature-update sites: **1 production + 23 in-crate
tests + 1 cross-crate test = 25 call sites across 3
files**.

If the audit's grep pass during step 1 reveals additional
callers (e.g., introduced between this plan's date and
implementation start), they're added to this list and
updated in the same step.

## Acceptance Gate

### Part A — Compile-time leader decision (10 tests)

`xlog-logic::optimizer::variable_ordering::tests`. Pure
leader-choice tests; threshold-gate tests live in Part E.

* **Triangle leader picks**: 3 tests, one per leader,
  each with a stats snapshot that puts the target
  leader's `min_card / default_leader_card` ratio
  comfortably **at or below** the 0.5 threshold (i.e.,
  the gate fires and `var_order = Some(...)`).
* **4-cycle leader picks**: 4 tests, one per leader,
  same construction (ratio ≤ 0.5 so the gate fires).
* **Missing-stats safety floor**: 1 test — empty
  `StatsSnapshot` produces `var_order = None` for both
  triangle + 4-cycle promoter inputs (single test
  covering both shapes).
* **Activation contract** (2 tests, paired): a single
  fixture (stats snapshot + canonical triangle body)
  that would trigger leader-change if W2.1 is active.
  - `default_config_leaves_var_order_none_even_with_triggering_stats`
    — `CompilerConfig::default()` (`Disabled`) →
    `var_order = None`. Bit-identical to pre-W2.1
    promoter output on the same body.
  - `leader_cardinality_config_sets_var_order_some_with_same_stats`
    — same fixture, `CompilerConfig {
    wcoj_variable_ordering: LeaderCardinality, .. }` →
    `var_order = Some(...)` with the expected
    `leader_idx`.

  These two MUST share the same stats snapshot and same
  body; the only diff is the `WcojVarOrderingKind`. That
  is the activation cert.

### Part B — Dispatch routing per leader (7 tests, runtime)

`xlog-runtime` lib tests using a `prepare_leader_inputs`
helper invocation. **Pointer-identity assertions are
invalid here** because the helper produces owned
projected/layout buffers (per the swap and layout
sections); test the observable consequences instead.

* Triangle: 3 tests (one per leader).
* 4-cycle: 4 tests (one per leader).

Each test builds a synthetic MultiWayJoin with
`var_order` set to a specific leader, calls
`prepare_leader_inputs`, and asserts:
* **Per-slot schema** matches the locked permutation
  table (e.g., for triangle e_yz-leader: slot 0 schema
  cols are `(Y, Z)`, slot 1 `(Z, X)`, slot 2 `(Y, X)`).
* **Per-slot column content** matches a CPU-computed
  reference (download via the existing test-only
  download path → host vec equality with the expected
  swapped/rotated columns).
* **`var_order.kernel_output_cols`** on the generated
  plan node matches the per-leader `head_proj` from the
  table (the W2.1 kernel projection lives **on
  `var_order`**, NOT on `MultiWayJoin.output_columns`).
* **`MultiWayJoin.output_columns` is unchanged** from the
  pre-W2.1 binary-fallback projection — slice 1/2/W2.2
  matchers continue to read it directly. The W2.1 plan
  pinned this separation in the "Output Buffer" section;
  Part B re-asserts it.
* **`var_order.leader_idx`** equals the requested leader
  on the resulting RIR node.

### Part C — End-to-end row-set equality (7 tests)

`crates/xlog-integration/tests/test_w21_variable_ordering.rs`.
**No test hook is introduced.** Each test compiles
through the public entry point
`Compiler::compile_with_config_and_stats_snapshot(source,
&CompilerConfig { wcoj_variable_ordering:
WcojVarOrderingKind::LeaderCardinality, .. },
Some(&stats_snapshot))` with a stats snapshot crafted to
make `LeaderCardinalityModel` pick the target leader
(min-card relation comfortably ≤ 0.5 ratio of the
default-leader card). The promoter then sets
`var_order` accordingly via the normal compile path —
NO bypass, NO unnamed hook.

* Triangle: 3 tests, one per leader. Force-WCOJ on,
  per-test stats snapshot biases the cost model toward
  the target leader. Counter ≥ 1 + row-set equality vs
  binary-join reference.
* 4-cycle: 4 tests, one per leader. Same.

### Part D — Stats-driven divergence (2 tests)

* Triangle + 4-cycle. Two stats snapshots favoring
  different leaders → `var_order.leader_idx` differs
  between the two compiled plans.

### Part E — Threshold gate cert (2 tests)

Pure threshold-policy tests, separate from Part A so
the policy boundary can be amended (e.g., rebalanced from
0.5 to 0.4) without disturbing leader-choice tests.

* `marginal_leader_cardinality_does_not_trigger_var_order`
  — leader card at 60% of default leader's card; W2.1
  cost model returns `var_order = None`. Triangle.
* `clear_win_leader_cardinality_triggers_var_order` —
  leader card at 30%; W2.1 returns `var_order =
  Some(...)`. Triangle.

(4-cycle threshold case folds into Part A's 4-cycle
leader tests; that test family always uses ratios **≤
0.5** so every test fires the gate. Part E covers the
above-threshold "no-fire" case for triangle.)

**Total acceptance cert count**: Part A 10 + Part B 7 +
Part C 7 + Part D 2 + Part E 2 + threshold-resolver 4 =
**32 tests**, all independently named. (The
threshold-resolver tests are unit-tested at the
`CompilerConfig` site in xlog-logic; they're listed under
the CompilerConfig section rather than a numbered Part.)

## Step Plan

1. **IR change** (xlog-ir): add `var_order` field +
   `VariableOrder` / `LookupPerm` types. Slice 1–4 + W2.2
   regression bit-identical (default `None`).
2. **CUDA helpers** (xlog-cuda): implement
   `wcoj_project_2col_swap_recorded` (failure-drain on
   sync) + `wcoj_project_output_columns_recorded`
   (failure-drain on sync; carries `cached_row_count` +
   fresh `num_rows_device` DtoD-copy). **11 unit tests**
   in xlog-cuda cert: 6 for swap (u32, u64, Symbol,
   empty, schema-swap, row-count parity) + 5 for output
   projection (u32 with triangle perm, u64 with 4-cycle
   perm, Symbol, row-count + identity-perm equality,
   empty-output n=0 with non-identity permutation).
3. **Cost-model trait + default impl** (xlog-logic):
   `WcojVariableOrderingModel` + `LeaderCardinalityModel`
   with the locked permutation tables for triangle (3
   leaders) and 4-cycle (4 leaders), plus the ≤50%
   threshold gate.
4. **CompilerConfig + composable API**: new struct +
   `compile_with_config_and_stats_snapshot`. Existing
   `compile()` / `compile_with_stats_snapshot()` delegate.
5. **Promoter wiring**: `promote_multiway(plan, rel_ids,
   stats, config)` — sweep all 25 caller sites per the
   audit list (1 production in `compile.rs:304`, 23 in
   `xlog-logic::promote::tests`, 1 cross-crate in
   `xlog-integration/tests/test_selectivity_pass_reordering.rs:648`).
   The cross-crate update is part of this commit; commit
   message references both W2.1 and W2.2 so the W2.2
   evidence trail stays consistent.
6. **Dispatcher reroute** (xlog-runtime): triangle +
   4-cycle dispatch sites consume `var_order`; per-leader
   `prepare_leader_inputs` helper; post-kernel output
   projection.
7. **Tests**: Part A + B + C + D + E.
8. **Workspace gate**: full slice 1–5 + W2.4 + W2.2
   regression bit-identical when `var_order = None`.
9. **Closure proposal + FF-merge**.

## Risk & Open Questions

* **Q1 — Per-dispatch layout overhead is unproven by W2.1
  certs.** Triangle non-default leaders allocate +
  re-sort up to 2 lookup atoms; 4-cycle non-default
  leaders allocate the rotation but DO NOT need col-swap
  (no extra sort beyond the existing one). The 0.5
  threshold is a **policy heuristic**, not a proven
  performance guarantee. Part E pins the policy
  boundary, **not** the perf claim. Performance
  validation of the threshold (i.e., does iteration
  saving really dominate layout cost at 0.5?) is folded
  into **closure board item W5.2** (skewed multi-way GPU
  benchmark suite); no untracked ticket is created.
  W5.2's existing acceptance gate already covers
  triangle / 4-cycle / 5-clique / pivot-heavy patterns,
  which is the right scope for measuring W2.1's
  threshold under realistic workloads. If a workload
  finds the default 0.5 threshold harmful before W5.2
  lands, the threshold is configurable per-compile via
  `CompilerConfig::wcoj_var_ordering_threshold` (e.g.,
  **lower** to `0.3` to **tighten** the gate — the gate
  fires on `ratio ≤ threshold`, so a smaller threshold
  demands a clearer win before reordering) without
  touching kernels.
* **Q2 — Output buffer re-permutation**: DtoD-copied per
  column post-kernel. For a triangle output of N rows,
  this is O(N) extra copy. For typical WCOJ output sizes
  this is much smaller than the kernel itself; not
  expected to be a performance concern, but unproven
  until W5 benchmarks land.
* **Q3 — `promote_multiway` callers** (corrected from
  iteration 4): **1 production caller + 23 in-crate
  tests + 1 cross-crate test = 25 sites across 3 files**.
  See "Caller Audit" section.
* **Q4 — Default-`None` regression**: bit-identical
  preservation gated on full WCOJ regression suite +
  CUDA cert + slice 1–5 / W2.4 / W2.2 cert files.
  Workspace gate (step 8) is the load-bearing check.
* **Q5 — Recorded helper failure drain**: both new
  helpers must drain on failure (sync-then-return)
  before partially-allocated owned buffers drop, per the
  slice 2 / W2.4 launch-stream safety pattern. Spec
  is included in each helper's section.

## Provenance

- Closure board: `docs/v065-closure-board.md` Wave 2, W2.1.
- ROADMAP item #2: "Variable-ordering cost model for WCOJ."
- Permutation tables grounded in `wcoj.cu:240` (triangle
  count) + `wcoj.cu:404` (4-cycle count) + the existing
  `wcoj_layout_*_recorded` sort/dedupe primitive.
- IR additive-Option pattern: slice 1–4 + W2.2 consumers
  ignore.
- Cost-model trait pattern: slice 3 / slice 5 / W2.4.
