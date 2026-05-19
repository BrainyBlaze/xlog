# W5.2 Closure Proposal - Skewed Multiway Bench

Date: 2026-05-12

Branch: `feat/w52-skewed-multiway-bench`

Plan: `docs/plans/2026-05-11-w52-bench-plan.md`

Evidence: `docs/evidence/2026-05-12-w52-skewed-multiway-bench/README.md`

## Commit Anchor

Commit count is anchored to the closure-proposal commit that contains this
file, using the command form:

```text
git rev-list --count main..<closure-proposal-commit>
```

The concrete closure-proposal commit hash and count are recorded in the final
review request after Git resolves the exact commit that contains this file.

Pre-proposal branch commits:

| Commit | Subject |
| --- | --- |
| `c2e7aaf4` | `docs(plan): W5.2 iteration 1 — skewed multiway bench (4cycle + 5clique + pivot-heavy)` |
| `4e4f6d15` | `bench(w52): add skewed multiway bench skeleton` |
| `1090a7af` | `bench(w52): measure 4-cycle crossover` |
| `12ed487c` | `bench(w52): measure 5-clique crossover` |
| `05fe9a0c` | `bench(w52): measure pivot-heavy multiway crossover` |
| `a2a259a1` | `docs(w52): aggregate cross-workload evidence` |

## Verbatim Plan Excerpts

D7 from plan commit `c2e7aaf4`:

<!-- BEGIN VERBATIM D7 -->
| D7 | **LP-MULTI-RUN is locked.** | "all crossover/ratio claims require ≥ 3 runs + min/median/max + win-direction-stability per cell". Any unstable cell must be reported as unstable and cannot support a closure-grade threshold claim without a stabilization step. |
<!-- END VERBATIM D7 -->

Acceptance Grid from plan commit `c2e7aaf4`:

<!-- BEGIN VERBATIM ACCEPTANCE GRID -->
## Acceptance Grid

| Workload | GPU WCOJ path | Binary hash baseline | Cell minimum | LP-MULTI-RUN evidence | Acceptance |
|---|---|---|---:|---|---|
| 4-cycle | `wcoj_layout_u32_recorded` x4 + `wcoj_4cycle_u32_recorded` | Three `hash_join_v2` joins + projection to `WXYZ` | 4 cells | Three independent runs; ratio min/median/max; direction stability per cell | Row-set parity, non-empty output, threshold or stable no-crossover finding |
| 5-clique | `wcoj_layout_sort_u32_recorded` x10 + `wcoj_clique5_u32_recorded` | Deterministic binary hash-chain over canonical K5 edge order | 4 cells | Three independent runs; ratio min/median/max; direction stability per cell | Row-set parity, non-empty output, threshold or stable no-crossover finding |
| Pivot-heavy K5 | Same clique5 path, with pivot-heavy fixture | Pivot-incident hash-chain before leaf filters | 4 cells | Three independent runs; ratio min/median/max; direction stability per cell | Row-set parity, exact expected pivot rows, threshold or stable no-crossover finding |
<!-- END VERBATIM ACCEPTANCE GRID -->

## Evidence Summary

W5.2 has 36 cell-run measurements: 3 workloads x 4 cells x 3 independent
Criterion invocations. Every cell asserted provider-direct GPU row-set parity
before timing. The 5-clique and pivot-heavy K5 cells also asserted exact
expected output rows before timing.

| Workload | Cells | Ratio Min | Ratio Median | Ratio Max | Direction Stability | Finding |
| --- | --- | ---: | ---: | ---: | --- | --- |
| 4-cycle hub-filtered | `N={50,250,1000,2000}` | 2.1156x | 3.8628x | 7.0174x | GPU 12/12 | Stable GPU win; threshold is at or below `N=50` for the tested shape. |
| 5-clique diagonal | `N={10,25,50,100}` | 0.4945x | 0.5446x | 0.5945x | HASH 12/12 | Stable no-GPU-crossover finding in the tested range. |
| Pivot-heavy K5 | `N={10,20,30,40}` | 0.5365x | 0.6294x | 0.9098x | HASH 12/12 | Stable hash win with a trend toward parity at larger pivot fanout. |

Direction flips: none.

The G3 single-run non-monotonic anomaly is retracted by the W5.2 multi-run
corpus. The stable result is workload-specific: 4-cycle is GPU-favored in the
tested range, while the diagonal 5-clique and pivot-heavy K5 fixtures are
hash-favored in the tested range.

Paper-alignment scope is P2/P5 only. W5.2 does not claim P1, P3, or P4; the
pivot-heavy fixture is evidence about skew sensitivity, not an implementation
of histogram-guided launch balancing.

## Acceptance Status

| Acceptance Item | Status |
| --- | --- |
| 4-cycle row-set parity and non-empty output | Satisfied before timing for every cell. |
| 5-clique row-set parity and exact diagonal rows | Satisfied before timing for every cell. |
| Pivot-heavy K5 row-set parity and exact pivot rows | Satisfied before timing for every cell. |
| LP-MULTI-RUN evidence | Satisfied: 4 cells x 3 runs per workload. |
| Threshold or stable no-crossover finding | Satisfied per workload, with stable direction in all 36 cell-runs. |
| W5.2 closure-board deliverable | Satisfied: bench harness committed and evidence README documents crossover thresholds versus binary hash. |

W5.2 closure unblocks W2.5's remaining blocker. The closure board currently
states that W2.5's default flip is blocked until W5.2 proves the cardinality
model is at least at parity on representative workloads.

## Final Gates

Final gates were run before this closure-proposal commit.

| Gate | Result |
| --- | --- |
| `cargo fmt --check --all` | Exit 0. |
| `RUSTFLAGS="-D warnings" cargo build --release --workspace --exclude pyxlog` | Exit 0. |
| `cargo bench -p xlog-integration --bench w52_skewed_multiway_bench --no-run` | Exit 0. |
| `cargo test -p xlog-cuda-tests --test certification_suite --release` | Exit 0; 1 passed, 0 failed. |
| `cargo test --workspace --release --exclude pyxlog --exclude xlog-cuda-tests` | Accepted under F-W43-12/F-W43-15 enumerated layout-file exception. First attempt exited 101 in `crates/xlog-cuda/tests/test_wcoj_layout_u32.rs::wcoj_layout_u32_already_sorted_deduped_round_trips`; targeted rerun of that test exited 0. Retry after settle exited 101 in `crates/xlog-cuda/tests/test_wcoj_layout_fast_path.rs::fast_path_symbol_sorted_unique_increments_counter`. Non-exempt `xlog-cuda` integration sweep excluding `test_wcoj_layout_fast_path`, `test_wcoj_layout_u32`, and `test_wcoj_layout_u64` exited 0. Workspace sweep excluding `xlog-cuda`, `xlog-cuda-tests`, and `pyxlog` exited 0. |

The `g04_transfer_efficiency` exception was not consumed.

## Scope And Holds

No `crates/xlog-cuda/` files changed in W5.2.

This proposal does not edit `docs/v065-closure-board.md`, does not mark W5.2
DONE, does not FF-merge, does not push, and does not tag. Those actions remain
outside this proposal until supervisor approval.

## Closure Board Response Options

| Response | Option | Outcome |
| --- | --- | --- |
| 1 | Accept as DONE (Recommended) | Accept W5.2 as closure-ready based on the committed bench harness, 36-run evidence corpus, stable per-workload findings, and final gate results. A later authorized follow-up may update the closure board and integrate the branch. |
| 2 | Reject | Keep W5.2 OPEN and specify the evidence, gate, or scope issue that must be corrected. |
| 3 | Defer | Keep W5.2 OPEN and carry the closure decision forward without changing the board. |
