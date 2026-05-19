# W2.5 Closure Proposal - Cost-Model Default Flip

Date: 2026-05-12

Branch: `feat/w25-cost-model-default-flip`

Plan: `docs/plans/2026-05-11-w25-default-flip-plan.md`

Evidence inputs:
- W2.5 implementation and integration cert commits on this branch.
- W5.2 benchmark corpus: `docs/evidence/2026-05-12-w52-skewed-multiway-bench/README.md`.

## Status

W2.5 is implemented as a narrow runtime-configuration default flip. Bare
`RuntimeConfig::default()` now resolves `wcoj_cost_model` to
`CostModelKind::Cardinality`; explicit `XLOG_WCOJ_COST_MODEL=skew` remains the
legacy opt-out; missing-stats safety-floor behavior still delegates to the skew
classifier; and the slice-4 stable-triangle counter remains exactly `== 1` under
the new bare default.

This proposal does not edit `docs/v065-closure-board.md`, does not push, does
not tag, and does not merge. Those follow-up actions require the approved
closure-board response.

## Commit Anchor

Commit count is anchored to the closure-proposal commit that contains this file,
using the command form:

```text
git rev-list --count main..<closure-proposal-commit>
```

At the closure-proposal commit, W2.5 has 6 commits on top of local `main`:
5 pre-proposal commits plus this closure-proposal commit. The final review
request records the concrete closure-proposal commit hash after Git resolves the
exact object.

Pre-proposal branch commits:

| Commit | Subject |
| --- | --- |
| `56685fa3` | `docs(plan): W2.5 iteration 1 — cost-model default-flip (Skew → Cardinality)` |
| `0f5b30d2` | `feat(w25): default-flip wcoj_cost_model resolver to Cardinality` |
| `37133ca0` | `test(w25): cert safety-floor under new default` |
| `d7e69101` | `test(w25): verify slice-4 stable-triangle counter == 1 under new default` |
| `36e8e46d` | `test(w25): refresh W2.6 default-flip regression cert` |

## Verbatim Plan Excerpts

D2 from plan commit `56685fa3`:

<!-- BEGIN VERBATIM D2 -->
| **D2** | **LOCKED: default ships by resolver change.** | W2.5 implementation changes `RuntimeConfig::resolved_wcoj_cost_model()` so unset field + unset env resolves to `CostModelKind::Cardinality`; do not add a new `RuntimeConfig` field or env var. |
<!-- END VERBATIM D2 -->

D5 from plan commit `56685fa3`:

<!-- BEGIN VERBATIM D5 -->
| **D5** | **LOCKED: missing-stats safety floor delegates.** | `CardinalityAwareCostModel` must continue to call the skew fallback whenever any slot relation has missing or zero stats, and classifier `Err` / `Ok(None)` must never be overridden by cardinality. |
<!-- END VERBATIM D5 -->

D6 from plan commit `56685fa3`:

<!-- BEGIN VERBATIM D6 -->
| **D6** | **LOCKED: slice-4 stable triangle stays exact.** | Under the new default, the stable-triangle recursive SCC cert must still assert `wcoj_triangle_dispatch_count() == 1` and row-set parity against the binary reference. |
<!-- END VERBATIM D6 -->

D7 from plan commit `56685fa3`:

<!-- BEGIN VERBATIM D7 -->
| **D7** | **LOCKED: W5.2 is the bench-spike-first input.** | Do not create a W2.5 spike branch. Cite W5.2's 36-measurement corpus as the benchmark evidence for default-flip parity / improvement. |
<!-- END VERBATIM D7 -->

Acceptance Grid from plan commit `56685fa3`:

<!-- BEGIN VERBATIM ACCEPTANCE GRID -->
## Acceptance Grid

| Sub-clause | Planned evidence | Gate |
|---|---|---|
| New default ships | `RuntimeConfig::default().resolved_wcoj_cost_model() == CostModelKind::Cardinality`; factory selects `CardinalityAwareCostModel` for bare default. | `cargo test -p xlog-core --lib --release wcoj_cost_model`; runtime integration certs using bare `RuntimeConfig::default()`. |
| Slice-4 stable-triangle counter still `== 1` | W2.5 branch of `stable_triangle_in_recursive_scc_dispatches_wcoj_on_seeding` runs bare default and asserts `wcoj_triangle_dispatch_count() == 1` plus row-set parity. | `cargo test -p xlog-integration --release --test test_wcoj_recursive_dispatch stable_triangle_in_recursive_scc_dispatches_wcoj_on_seeding -- --nocapture`. |
| Env opt-out restores legacy | `XLOG_WCOJ_COST_MODEL=skew` resolves to `SkewClassifier` and an integration cert matches explicit skew baseline counter + row set. | `cargo test -p xlog-core --lib --release wcoj_cost_model`; `cargo test -p xlog-integration --release --test test_wcoj_cardinality_cost_model -- --nocapture`. |
| W5.2 bench evidence documents parity / improvement | Closure proposal cites W5.2 36-measurement corpus: 4-cycle GPU 12/12, 5-clique HASH 12/12, pivot-heavy K5 HASH 12/12, zero direction flips. | Source audit of `docs/evidence/2026-05-12-w52-skewed-multiway-bench/README.md:339-358` and closure proposal review. |
<!-- END VERBATIM ACCEPTANCE GRID -->

## Evidence Summary

| Acceptance sub-clause | Delivered evidence |
| --- | --- |
| New default ships | `0f5b30d2` changes only the resolver branch/docs/tests in `crates/xlog-core/src/config.rs`; `RuntimeConfig::default().resolved_wcoj_cost_model() == CostModelKind::Cardinality`; core resolver certs pass 2/0. |
| Missing-stats safety floor delegates | `37133ca0` updates `test_wcoj_cardinality_cost_model.rs` so explicit `SkewClassifier` and bare `RuntimeConfig::default()` run with no seeded stats and assert identical counter + row set; full file passes 7/0. RED-via-temporary-production-mutation failed with counter 0 vs skew baseline 1, proving regression detection. |
| Slice-4 stable-triangle counter still `== 1` | `d7e69101` extends `stable_triangle_in_recursive_scc_dispatches_wcoj_on_seeding` with a bare-default branch; the branch asserts `wcoj_triangle_dispatch_count() == 1` and row-set parity against binary reference; targeted cert passes 1/0. |
| Env opt-out restores legacy | `0f5b30d2` adds/updates resolver certs for `XLOG_WCOJ_COST_MODEL=skew`; explicit skew remains highest-precedence and conservative env fallback still resolves to `SkewClassifier`. |
| W5.2 bench evidence documents parity / improvement | W5.2 README lines 339-358 record 36 measurements: 4-cycle hub-filtered GPU 12/12, 5-clique diagonal HASH 12/12, pivot-heavy K5 HASH 12/12, zero direction flips. Lines 117-120 and 360-362 keep paper scope to P2/P5 only and leave P3 to W3.3. |

Step 5 regression sweep also refreshed the W2.6 default-regression cert in
`36e8e46d`: explicit legacy `SkewClassifier` keeps the old W2.3 counter `== 3`,
while the bare W2.5 cardinality default keeps the binary path at counter `== 0`
for that small-cardinality fixture; both preserve row-set parity against the
binary reference.

## Final Gates

Final gates were run before this closure-proposal commit.

| Gate | Result |
| --- | --- |
| `cargo fmt --check --all` | Exit 0. |
| `RUSTFLAGS="-D warnings" cargo build --release --workspace --exclude pyxlog` | Exit 0. |
| `cargo test -p xlog-core --lib --release wcoj_cost_model` | Exit 0; 2 passed, 0 failed. |
| `cargo test -p xlog-runtime --lib --release wcoj_cost_model` | Exit 0; 23 passed, 0 failed. |
| `cargo test -p xlog-integration --release --test test_wcoj_cardinality_cost_model -- --nocapture` | Exit 0; 7 passed, 0 failed. |
| `cargo test -p xlog-integration --release --test test_wcoj_recursive_dispatch stable_triangle_in_recursive_scc_dispatches_wcoj_on_seeding -- --nocapture` | Exit 0; 1 passed, 0 failed. |
| `cargo test -p xlog-integration --release --test test_w21_variable_ordering -- --nocapture` | Exit 0; 11 passed, 0 failed. |
| `cargo test -p xlog-integration --release --test test_selectivity_pass_reordering -- --nocapture` | Exit 0; 6 passed, 0 failed. |
| `cargo test -p xlog-runtime --release --features recursive-stats-trace --test test_w23_recursive_stats -- --nocapture` | Exit 0; 10 passed, 0 failed. |
| `cargo test -p xlog-integration --release --test test_wcoj_record_join_result_feedback -- --nocapture` | Exit 0; 3 passed, 0 failed. |
| `cargo test -p xlog-integration --release --test test_w26_heat_selectivity -- --nocapture` | Exit 0; 7 passed, 0 failed. |
| `cargo test -p xlog-cuda-tests --test certification_suite --release` | Exit 0; 1 passed, 0 failed. |
| `cargo test --workspace --release --exclude pyxlog --exclude xlog-cuda-tests` | Accepted under F-W43-12/F-W43-15 enumerated layout-file exception. First attempt exited 101 in `crates/xlog-cuda/tests/test_wcoj_layout_u32.rs::wcoj_layout_u32_already_sorted_deduped_round_trips`; targeted rerun of that test exited 0. Non-exempt `xlog-cuda` integration sweep excluding only `test_wcoj_layout_fast_path`, `test_wcoj_layout_u32`, and `test_wcoj_layout_u64` exited 0. Workspace sweep excluding `xlog-cuda`, `xlog-cuda-tests`, and `pyxlog` exited 0. |

The `g04_transfer_efficiency` exception was not consumed.

## Scope And Holds

No production kernel/provider/executor/runtime files changed in W2.5 beyond the
single `RuntimeConfig` resolver branch in `xlog-core`. No new `RuntimeConfig`
field, env var, cost-model variant, benchmark harness, provider route, or kernel
path was added.

W2.5 makes no new paper claim. It inherits W2.1's cost-model basis and uses
W5.2's P2/P5 benchmark evidence as the bench-spike-first input.

## Closure Board Response Options

| Response | Option | Outcome |
| --- | --- | --- |
| 1 | Accept as DONE (Recommended) | Accept W2.5 as closure-ready based on the resolver flip, safety-floor cert, slice-4 counter cert, env opt-out cert, W5.2 evidence input, and final gate results. A follow-up commit may update the closure board and integrate the branch. |
| 2 | Reject | Keep W2.5 OPEN and specify the resolver, safety-floor, counter, opt-out, evidence, or gate issue that must be corrected. |
| 3 | Defer | Keep W2.5 OPEN and carry the closure decision forward without changing the board. |
