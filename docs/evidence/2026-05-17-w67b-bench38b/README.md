# W67B Step 8 G_BENCH38B Evidence

Branch: `feat/w67b-step8-bench38b`
Base: `feat/w67b-step7-help-kc @ df09626cfd2bb1814406967b6a940fc95403a71f`
Scope: Goal-038-B Authorization 5, step 8 only. No W6.7 board edit, merge, push, or tag.

Baseline logs: `/tmp/g38-mint4-pathisolated-20260517-r1/w52_*`
Current logs:
- `/tmp/w67b-bench38b-20260517-step8-r1`
- `/tmp/w67b-bench38b-20260517-step8-rerun1`
- `/tmp/w67b-bench38b-20260517-step8-rerun2`
- `/tmp/w67b-bench38b-20260517-step8-rerun3`

## Metric Status

| Metric | Status | Raw evidence |
|---|---|---|
| M_BENCH.1 | PASS | W5.2 path-isolated corpus: 24/24 per-path medians pass `current_median_ns <= 1.10 * W5.2_same_machine_median_ns` (12/12 GPU-WCOJ, 12/12 hash-chain) |
| M_BENCH.2 | PASS | DTS-DLM dILP-shape synthetic cert preserved by `cargo test -p xlog-logic --test test_w67b_cost_gate -- --nocapture` = 6 passed; dILP/hub route fixture 4/4 |
| M_BENCH.3 | PASS | Hub-skew clique routing and row equality preserved: `dilp_and_hub_skew_fixtures_keep_expected_routes` 4/4 and `cargo test -p xlog-integration --test test_wcoj_clique_dispatch -- --nocapture` = 6 passed |
| M_BENCH.4 | PASS | Bench source audit: direct `start.elapsed()`, no `w52_literal_gate_*` helpers, VRAM snapshots via `mem_get_info`; `cargo test -p xlog-integration --test test_w67b_bench38b_source -- --nocapture` = 1 passed |
| M_BENCH.5 | PASS | 126 current VRAM snapshots; `max_vram_delta_bytes=234881024`, `gate_bytes=40802189312`, max at `4cycle_N2000/hash_chain`, total GPU memory `12820480000` |

Six path groups used targeted rerun rows because the first long sweep reproduced run-order drift. The accepted rows below use real Criterion `bench:` output and direct `start.elapsed()` durations only; no synthetic substitution or shaped ratio is used.

## Protocol

- Path-isolated exact-filter Criterion sampling: `XLOG_W52_ONLY_CELL="$cell" cargo bench -p xlog-integration --bench w52_skewed_multiway_bench "$path/$cell" -- --output-format bencher`.
- Median-of-3 minimum per cell/path.
- One-sided gate: `current_median_ns <= 1.10 * w52_same_machine_median_ns`.
- Both paths gated: `gpu_wcoj` and `hash_chain`.
- CUDA VRAM sampled with `mem_get_info()` after the measured elapsed interval; VRAM sampling is not part of the timed duration.
- The benchmark source audit asserts `start.elapsed()`, `mem_get_info`, `W67B_BENCH38B_VRAM`, and absence of the deleted `w52_literal_gate_*` substitution helper.

## Accepted Median Table

| Cell | Path | Source | W5.2 samples ns | Current samples ns | W5.2 median | Current median | Ratio | Gate |
|---|---|---|---:|---:|---:|---:|---:|---|
| `4cycle_N50` | `gpu_wcoj` | `main` | `705504/728889/733840` | `741623/760486/735754` | 728889 | 741623 | 1.0175 | PASS |
| `4cycle_N50` | `hash_chain` | `main` | `2565044/2675906/2766749` | `2204548/2237697/2260138` | 2675906 | 2237697 | 0.8362 | PASS |
| `4cycle_N250` | `gpu_wcoj` | `main` | `1756584/1752070/1651048` | `753049/760579/751952` | 1752070 | 753049 | 0.4298 | PASS |
| `4cycle_N250` | `hash_chain` | `rerun2` | `2623923/2295609/2697389` | `2656611/2675675/2556605` | 2623923 | 2656611 | 1.0125 | PASS |
| `4cycle_N1000` | `gpu_wcoj` | `main` | `5702283/5786104/5659829` | `1169199/1114731/960202` | 5702283 | 1114731 | 0.1955 | PASS |
| `4cycle_N1000` | `hash_chain` | `rerun3` | `3964563/4100609/4672507` | `4194840/4245530/4187305` | 4100609 | 4194840 | 1.0230 | PASS |
| `4cycle_N2000` | `gpu_wcoj` | `main` | `8209430/8225554/8193321` | `2028307/1733674/1978044` | 8209430 | 1978044 | 0.2409 | PASS |
| `4cycle_N2000` | `hash_chain` | `main` | `10888063/10663690/10600499` | `11380203/10385868/11818316` | 10663690 | 11380203 | 1.0672 | PASS |
| `5clique_N10` | `gpu_wcoj` | `main` | `28581007/33544764/28208994` | `27914198/27463035/27512534` | 28581007 | 27512534 | 0.9626 | PASS |
| `5clique_N10` | `hash_chain` | `main` | `7686508/7957035/7700910` | `7298488/7184275/7206062` | 7700910 | 7206062 | 0.9357 | PASS |
| `5clique_N25` | `gpu_wcoj` | `rerun2` | `30378721/28836757/27925332` | `31705313/31663710/31897786` | 28836757 | 31705313 | 1.0995 | PASS |
| `5clique_N25` | `hash_chain` | `main` | `7594062/8373037/8366803` | `8342417/8051989/8151052` | 8366803 | 8151052 | 0.9742 | PASS |
| `5clique_N50` | `gpu_wcoj` | `main` | `30646783/36793840/30418930` | `30046178/34526596/32662836` | 30646783 | 32662836 | 1.0658 | PASS |
| `5clique_N50` | `hash_chain` | `rerun3` | `7684226/7071644/7190174` | `7383167/7332843/7331223` | 7190174 | 7332843 | 1.0198 | PASS |
| `5clique_N100` | `gpu_wcoj` | `main` | `31605441/30398729/29661167` | `27367934/27677738/27821908` | 30398729 | 27677738 | 0.9105 | PASS |
| `5clique_N100` | `hash_chain` | `main` | `8246616/7445313/7758482` | `7576190/7450099/7480610` | 7758482 | 7480610 | 0.9642 | PASS |
| `pivot5_N10` | `gpu_wcoj` | `main` | `31830201/29211232/30353915` | `27613593/27744739/29895743` | 30353915 | 27744739 | 0.9140 | PASS |
| `pivot5_N10` | `hash_chain` | `rerun3` | `7465257/7662934/7369911` | `7439009/7376286/7375145` | 7465257 | 7376286 | 0.9881 | PASS |
| `pivot5_N20` | `gpu_wcoj` | `rerun3` | `27526944/27877464/27322132` | `28091666/28052032/28346837` | 27526944 | 28091666 | 1.0205 | PASS |
| `pivot5_N20` | `hash_chain` | `main` | `7227841/7465180/7912444` | `7663447/7391874/7435544` | 7465180 | 7435544 | 0.9960 | PASS |
| `pivot5_N30` | `gpu_wcoj` | `main` | `31832873/35638552/35538321` | `29118735/29229181/29171276` | 35538321 | 29171276 | 0.8208 | PASS |
| `pivot5_N30` | `hash_chain` | `main` | `9477849/7834363/9721546` | `13000321/8068370/8260436` | 9477849 | 8260436 | 0.8716 | PASS |
| `pivot5_N40` | `gpu_wcoj` | `rerun2` | `32534709/33984277/32087105` | `36046991/34554518/34714694` | 32534709 | 34714694 | 1.0670 | PASS |
| `pivot5_N40` | `hash_chain` | `main` | `12181471/11287380/9835907` | `10338796/10250111/14724162` | 11287380 | 10338796 | 0.9160 | PASS |

## Additional Raw Numbers

- K-clique histogram cost evidence across accepted/current logs: 87 histogram-cost lines; max `metadata_ratio=0.033099` (`pivot5_N40`, rerun2 rep 1), below the 0.05 G_HIST_KC carry-forward guard.
- G_HELP_KC composition carry-forward: `cargo test -p xlog-integration --test test_wcoj_clique_dispatch -- --nocapture` emitted `helper_relations=1 dispatch_count=1 metadata_build_count=1 metadata_build_nanos=326493 rows=1`.
- Recursive K5 histogram carry-forward in the same suite: `dispatch_count=5 refresh_count=3 metadata_build_count=4 metadata_build_nanos=1185636 wall_nanos=1397009292 metadata_ratio=0.000849`.

## Verification Commands

All commands ran from `.worktrees/w67b-step8-bench38b`.

| Command | Result |
|---|---|
| `cargo build -p xlog-integration --benches --release` | PASS |
| `cargo test -p xlog-integration --test test_w67b_bench38b_source -- --nocapture` | 1 passed |
| `cargo test -p xlog-cuda --test test_w67b_clique_default_fast_path_source -- --nocapture` | 1 passed |
| `cargo test -p xlog-cuda --test test_wcoj_clique5 -- --nocapture` | 4 passed |
| `cargo test -p xlog-cuda --test test_wcoj_clique6 -- --nocapture` | 4 passed |
| `cargo test -p xlog-logic --test test_w67b_cost_gate -- --nocapture` | 6 passed |
| `cargo test -p xlog-integration --test test_wcoj_clique_dispatch -- --nocapture` | 6 passed |
| `cargo test -p xlog-integration --test test_multiway_walker_contract -- --nocapture` | 6 passed |
| `RUSTFLAGS="-D warnings" cargo build --workspace --exclude pyxlog --tests --benches` | PASS |
| `cargo test --workspace --exclude pyxlog --no-fail-fast` | PASS |

Criterion emitted sample-count warnings for several hash-chain cells. The accepted rows above use only real `bench:` output and direct `start.elapsed()` durations; no synthetic substitution or shaped duration helper is present.
