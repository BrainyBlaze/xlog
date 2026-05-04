# W2.1 Evidence — Variable-Ordering Cost Model

**Closes board item: W2.1.**
**Date:** 2026-05-04
**Branch:** `feat/w21-variable-ordering-cost-model`
**Base:** `main` at `0c176e6a` (W2.2 closure-board commit).
**Plan:** `docs/plans/2026-05-04-w21-variable-ordering-cost-model-plan.md`
(approved iteration 8, committed as `d1b13951`).

## Summary

Adds a real variable-ordering cost model for triangle and 4-cycle
WCOJ. The cost model decides which input becomes the kernel's
**leader slot** (slot 0) at compile time; the dispatcher rotates
the kernel inputs and (for triangle) col-swaps non-leader lookups
through new `wcoj_project_*_recorded` helpers, then post-projects
the kernel-direct output back to canonical head order.

**No kernel signature changes.** **Default disables the path**
via `CompilerConfig` so slice 1/2/4 + W2.2 dispatch and row sets
are bit-identical when the default config is in effect. Activation
requires explicit construction of a `CompilerConfig` with
`WcojVarOrderingKind::LeaderCardinality` and the new
`Compiler::compile_with_config_and_stats_snapshot(...)` entry
point. **No env opt-in** (env-driven activation is out of W2.1
scope and would require a new closure-board item before being
referenced).

## Acceptance Gate

Cert tally (per the W2.1 plan §"Acceptance Gate"):

| Part | Tests | Path | File |
|------|-------|------|------|
| Resolver | 4 | `xlog-logic` lib (compiler_config) | `crates/xlog-logic/src/compiler_config.rs::tests` |
| A — Compile-time leader decision | 10 | `xlog-logic` test | `crates/xlog-logic/tests/test_w21_part_a.rs` |
| B — Dispatch routing per leader | 7 | `xlog-logic` test | `crates/xlog-logic/tests/test_w21_part_b.rs` |
| C — End-to-end row-set parity | 7 | `xlog-integration` test | `crates/xlog-integration/tests/test_w21_variable_ordering.rs` |
| D — Stats-driven divergence | 2 | `xlog-integration` test | same file |
| E — Threshold gate cert | 2 | `xlog-integration` test | same file |
| **Total** | **32** | | |

In addition to the 32 plan-specified acceptance tests:

* **Step 2 CUDA helpers** add 11 unit tests in
  `crates/xlog-cuda/tests/test_wcoj_project.rs` (6 swap + 5
  output-projection). All pass on real CUDA.
* **Step 3 cost-model trait** adds 12 internal unit tests in
  `crates/xlog-logic/src/wcoj_var_ordering.rs::tests` (cost-model
  trait short-circuits + locked permutation table sanity).

### Part A (10 tests, all pass)

```
test cycle4_picks_e_xy_when_e_xy_smallest ... ok
test cycle4_picks_e_wx_default_when_e_wx_smallest ... ok
test cycle4_picks_e_zw_when_e_zw_smallest ... ok
test leader_cardinality_config_sets_var_order_some_with_same_stats ... ok
test cycle4_picks_e_yz_when_e_yz_smallest ... ok
test triangle_picks_e_xz_when_e_xz_smallest ... ok
test triangle_picks_e_xy_default_when_e_xy_smallest ... ok
test triangle_picks_e_yz_when_e_yz_smallest ... ok
test default_leader_already_min_returns_none_for_both_shapes ... ok
test default_config_leaves_var_order_none_even_with_triggering_stats ... ok
```

**Note** on the 9th Part A test: the W2.1 plan §"Part A" framed
it as "Missing-stats safety floor". At the compile-time /
promoter level, the actual reachable short-circuit on uniform
stats is the "default leader is already min" rule (cost model
returns None when argmin == 0). True missing-stats semantics
(`card_of` returning None on zero card) is unit-tested at
`xlog_logic::wcoj_var_ordering::tests::
missing_stats_returns_none_safety_floor` (step 3, in the unit
test bucket).

### Part B (7 tests, all pass on real CUDA)

Per the W2.1 plan §"Part B", these are **xlog-runtime tests
that invoke `prepare_leader_inputs` directly** with synthesized
`VariableOrder` values from the locked permutation tables. Each
test asserts:
* per-slot **schema** matches the locked table (e.g., triangle
  e_yz-leader: slot 0 = `(Y, Z)`, slot 1 = `(Z, X)` after
  col-swap, slot 2 = `(Y, X)` after col-swap);
* per-slot **content** matches a CPU-computed reference
  downloaded via `cuMemcpyDtoH_v2`;
* `var_order.kernel_output_cols` matches the locked
  `head_proj`;
* `var_order.leader_idx` equals the requested leader.

```
test part_b_triangle_e_yz_leader ... ok
test part_b_triangle_e_xz_leader ... ok
test part_b_triangle_e_xy_default_leader ... ok
test part_b_cycle4_e_xy_leader ... ok
test part_b_cycle4_e_wx_default_leader ... ok
test part_b_cycle4_e_yz_leader ... ok
test part_b_cycle4_e_zw_leader ... ok
```

Lives in `crates/xlog-runtime/tests/test_w21_part_b.rs`.
`Executor::prepare_leader_inputs` is the new public runtime
helper that the production W2.1 path
(`run_wcoj_*_pipeline_w21`) and these tests both consume.

### Part C / D / E (11 tests, all pass on real CUDA)

```
test part_e_marginal_leader_cardinality_does_not_trigger_var_order ... ok
test part_e_clear_win_leader_cardinality_triggers_var_order ... ok
test part_d_triangle_two_snapshots_produce_different_leader_idx ... ok
test part_d_cycle4_two_snapshots_produce_different_leader_idx ... ok
test part_c_triangle_leader_e_xz ... ok
test part_c_triangle_default_leader_e_xy ... ok
test part_c_triangle_leader_e_yz ... ok
test part_c_cycle4_leader_e_yz ... ok
test part_c_cycle4_leader_e_zw ... ok
test part_c_cycle4_leader_e_xy ... ok
test part_c_cycle4_default_leader_e_wx ... ok
```

## Workspace Tally (step 8)

```
cargo test --workspace --release --tests --exclude pyxlog
  Pre-W2.1 (main @ 0c176e6a): PASS=1859 FAIL=0
  Post-W2.1 (HEAD):           PASS=1914 FAIL=0

cargo test -p xlog-cuda-tests --release --test certification_suite
test run_full_certification ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured
```

Slice 1–5 + W2.4 + W2.2 regression preserved bit-identically
under the `CompilerConfig::default()` (Disabled) path.

**Test delta**: +55 W2.1 tests (1859 → 1914), broken down as:

| Tests | Step | Location |
|-------|------|----------|
| 4 | Step 4 | `xlog-logic` resolver unit tests |
| 12 | Step 3 | `xlog-logic::wcoj_var_ordering::tests` cost-model trait + locked-permutation-table sanity |
| 11 | Step 2 | `xlog-cuda` helper certs (real CUDA) |
| 10 | Step 7 | Part A — compile-time leader decision |
| 7 | Step 7 | Part B — dispatch routing per leader |
| 7 | Step 7 | Part C — end-to-end row-set parity (real CUDA) |
| 2 | Step 7 | Part D — stats-driven divergence |
| 2 | Step 7 | Part E — threshold gate |
| **55** | | |

The 4 + 12 + 11 = 27 unit tests are **above** the 28 plan-spec
acceptance gate tests; the plan §"Acceptance Gate" total of 32
counts the resolver 4 alongside the 28 Part A–E tests.

## Code-Level Changes

| File | Change |
|------|--------|
| `crates/xlog-ir/src/rir.rs` | New `LookupPerm` + `VariableOrder` types; new `var_order: Option<VariableOrder>` field on `RirNode::MultiWayJoin`. 19 construction sites + 6 pattern destructures updated. |
| `crates/xlog-logic/src/compiler_config.rs` | New module: `WcojVarOrderingKind { Disabled, LeaderCardinality }`, `CompilerConfig` struct, `DEFAULT_THRESHOLD = 0.5`, `effective_wcoj_var_ordering_threshold()` resolver, 4 unit tests. |
| `crates/xlog-logic/src/compile.rs` | New entry points `Compiler::compile_with_config_and_stats_snapshot(...)` and program-level variant. Existing `compile()` / `compile_with_stats_snapshot(...)` delegate via `CompilerConfig::default()`. |
| `crates/xlog-logic/src/wcoj_var_ordering.rs` | New module: `WcojVariableOrderingModel` trait, default `LeaderCardinalityModel`, locked permutation tables for triangle (3 leaders) and 4-cycle (4 leaders), 12 unit tests. |
| `crates/xlog-logic/src/promote.rs` | `promote_multiway` signature: `(plan, rel_ids, stats, config)`. `try_promote_triangle` / `try_promote_4cycle` build `var_order` via the cost model + locked tables. 23 in-crate test sites updated. |
| `crates/xlog-integration/tests/test_selectivity_pass_reordering.rs` | Cross-crate caller updated to new `promote_multiway` signature. W2.2 cert continues to exercise the legacy slice 1/2/W2.2 dispatch (no W2.1 var ordering activated; row sets remain bit-identical). |
| `crates/xlog-cuda/src/provider/wcoj_project.rs` | New module: `wcoj_project_2col_swap_recorded` + `wcoj_project_output_columns_recorded`. Failure-drain on Err per slice 2 / W2.4 launch-stream safety pattern; carries `cached_row_count` + DtoD-copies `num_rows_device`. |
| `crates/xlog-cuda/tests/test_wcoj_project.rs` | New cert file: 11 unit tests on real CUDA. |
| `crates/xlog-runtime/src/executor/wcoj_dispatch.rs` | `try_dispatch_wcoj_*_on_body` extract `var_order` from body and pass `Option<&VariableOrder>` to the inner pipeline. New `run_wcoj_*_pipeline_w21` variants delegate slot prep to `Executor::prepare_leader_inputs` (new pub method) and apply post-kernel projection. Module-scope helpers: `perm_indices_from_kernel_output_cols`, `build_triangle_head_schema`, `build_4cycle_head_schema`. `prepare_leader_inputs` materializes owned slot inputs (DtoD-copy via the swap helper, double-swap for no-swap pass-through) so it has a uniform owned-buffer return type — Part B tests assert per-slot schema + content against this output. `wcoj_dispatch_stream_or_init` promoted from `pub(super)` to `pub` to support the same tests. |
| `crates/xlog-logic/tests/test_w21_part_a.rs` | New cert file: 10 Part A tests. |
| `crates/xlog-runtime/tests/test_w21_part_b.rs` | New cert file: 7 Part B tests (per-slot schema + content + var_order metadata). |
| `crates/xlog-integration/tests/test_w21_variable_ordering.rs` | New cert file: 7 Part C + 2 Part D + 2 Part E tests. |

## Decisions / Limitations

* **Default disables W2.1.** `CompilerConfig::default()` is
  `Disabled`. Activation requires explicit construction +
  `compile_with_config_and_stats_snapshot(...)`. No env override.
* **Threshold clamping at use-site.** `wcoj_var_ordering_threshold`
  field is `pub` (struct-literal access preserved), but the
  promoter MUST call `effective_wcoj_var_ordering_threshold()` —
  the doc comment explicitly redirects readers. Out-of-range
  values (`NaN`, ≤ 0.0, > 1.0) clamp to `DEFAULT_THRESHOLD = 0.5`.
* **Performance is unproven by W2.1 certs.** The 0.5 threshold
  is a **policy heuristic**. Performance validation of the
  threshold (does iteration saving dominate layout cost at 0.5?)
  is folded into closure-board item **W5.2** (skewed multi-way
  GPU benchmark suite). Per-compile threshold is configurable for
  early workload remediation.
* **`prepare_leader_inputs` extracted as a `pub` runtime helper.**
  The W2.1 plan §"Part B" calls for an extracted helper that
  Part B tests can invoke directly. `Executor::prepare_leader_inputs(canonical, var_order, stream)`
  materializes owned slot inputs (using the existing
  `wcoj_project_2col_swap_recorded` for swap; double-swap for
  no-swap pass-through) and returns `Vec<CudaBuffer>` with
  uniform owned ownership. Both production callers
  (`run_wcoj_*_pipeline_w21`) and Part B tests consume it.
  `wcoj_dispatch_stream_or_init` was promoted from `pub(super)`
  to `pub` to support these tests as well.
* **Phase timing** instrumentation NOT added on the W2.1 path
  (per the plan §"Risk & Open Questions / Q1" — perf validation
  deferred to W5.2).
* **Rotated-head triangle rules.** When the rule head is in a
  non-canonical order (e.g., `triangle(Z, X, Y) :- ...`), the
  W2.2 matcher already gates dispatch eligibility on canonical
  output_columns layouts. Such rules decline WCOJ dispatch and
  fall through to binary-join — the W2.1 path is never invoked
  on them. Var_order may be set on the IR (cost model fires),
  but it has no effect.

## Process Rule Compliance

* **Process rule #1**: this slice does NOT self-mark W2.1 DONE.
  End-of-slice commit proposes the OPEN → DONE transition in the
  commit message; user reviews and explicitly approves; a
  separate follow-up commit applies the board update.
* **Process rule #2**: every commit references W2.1. Commits to
  date (chronological): plan + 7 implementation steps (1, 4, 3,
  5, 2, 6, 7) + step 9 evidence + step 9' rename/count
  amendment + step 9'' fmt + Part B helper extraction = **11
  commits total**.
* **Process rule #3**: plan header opens with "Closes W2.1."
* **Process rule #5**: no `v0.6.6` references introduced in any
  W2.1 file/comment/plan/evidence/commit message.
* **Process rule #6**: no push, no tag.

## Closure Board Update Proposal

After user review and explicit "mark W2.1 DONE" approval, a
follow-up commit applies:

* `docs/v065-closure-board.md` — W2.1 status `OPEN → DONE`,
  status tally updated (DONE: 2 → 3; OPEN: 16 → 15).
* `docs/v065-closure-board.md` "Completed" section gets a W2.1
  entry referencing the full commit list (plan, IR,
  CompilerConfig, cost-model trait, promoter wiring, CUDA
  helpers, dispatcher reroute, acceptance gates, evidence
  README, rename/count amendment, fmt + Part B helper
  extraction).
* W2.6 unblocking note: with W2.1 now DONE, W2.6's
  `Blocked by` set drops to `{W2.4}` only — and W2.4 is also
  DONE. W2.6 transitions to OPEN.
