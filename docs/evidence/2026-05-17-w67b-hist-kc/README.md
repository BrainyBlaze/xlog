# W6.7 Step 6 G_HIST_KC Evidence

Branch: `feat/w67b-step6-hist-kc`
Base: `feat/w67b-step5-costgate @ 77106ea09ab7d3ec9a2105acbae9188d57d4a29e`
Scope: Goal-038-B Authorization 5, step 6 only. No W6.7 board edit, merge, push, or tag.

## Metric Status

| Metric | Status | Raw evidence |
|---|---|---|
| M_HIST_KC.1 | PASS | `wcoj_clique{5,6}_metadata_recorded_{u32,u64}` present: 4/4 provider entries; `cargo test -p xlog-cuda --test test_w67b_hist_kc_source -- --nocapture` = 8 passed |
| M_HIST_KC.2 | PASS | K-clique HG count/materialize templates accept `unique_keys`, `fan_out`, `prefix_sum`, `total`; source audit = PASS |
| M_HIST_KC.3 | PASS | Provider builds leader metadata before count launch and records metadata buffer reads for count/materialize; source audit = PASS |
| M_HIST_KC.4 | PASS | `XLOG_DETERMINISTIC=1 cargo test -p xlog-cuda --test test_wcoj_clique5 -- --nocapture` = 4 passed, including K5 metadata path 100/100; `XLOG_DETERMINISTIC=1 cargo test -p xlog-cuda --test test_wcoj_clique6 -- --nocapture` = 4 passed, including K6 metadata path 100/100 |
| M_HIST_KC.5 | PASS | Recursive K5 fixture: `dispatch_count=5`, `refresh_count=3`, `metadata_build_count=4`, `metadata_build_nanos=1249071`, `wall_nanos=1527229074`, `metadata_ratio=0.000818`, output rows = 4, fallback row equality PASS |
| M_HIST_KC.6 | PASS | W5.2 routing prediction preserved: `cargo test -p xlog-logic --test test_w67b_cost_gate -- --nocapture` = 6 passed, including `w52_routing_decision_cert_is_36_of_36`; `cargo test -p xlog-logic --test test_hg_kclique_planner -- --nocapture` = 7 passed, including `w52_baseline_prediction_precision_is_36_of_36` |
| M_HIST_KC.7 | PASS | W52 selected K5 bench cells: `5clique_N10 metadata_build_count=1 metadata_build_nanos=266261 wall_nanos=28176510 metadata_ratio=0.009450`; `pivot5_N10 metadata_build_count=1 metadata_build_nanos=243345 wall_nanos=28073432 metadata_ratio=0.008668`; both <= 0.05. `wcoj_phase_report` also exposes recursive K5 merge refresh count/nanos per S_HIST_KC.7 |
| M_HIST_KC.8 | PASS | Required paper section 5 Algorithm 1 Phase 1 source-citation comment present in K-clique kernel and provider; source audit = PASS |

## Verification Commands

All commands ran from `.worktrees/w67b-step6-hist-kc`.

| Command | Result |
|---|---|
| `cargo fmt --check` | PASS |
| `git diff --check` | PASS |
| `cargo test -p xlog-cuda --test test_w67b_hist_kc_source -- --nocapture` | 8 passed |
| `cargo test -p xlog-logic --test test_w67b_cost_gate -- --nocapture` | 6 passed |
| `cargo test -p xlog-logic --test test_w32_clique_promoter -- --nocapture` | 15 passed |
| `cargo test -p xlog-runtime --test test_w67b_dispatch_plan_source -- --nocapture` | 2 passed |
| `XLOG_DETERMINISTIC=1 cargo test -p xlog-cuda --test test_wcoj_clique5 -- --nocapture` | 4 passed |
| `XLOG_DETERMINISTIC=1 cargo test -p xlog-cuda --test test_wcoj_clique6 -- --nocapture` | 4 passed |
| `cargo test -p xlog-integration --test test_wcoj_clique_dispatch -- --nocapture` | 5 passed |
| `cargo test -p xlog-logic --test test_hg_kclique_planner -- --nocapture` | 7 passed |
| `cargo test -p xlog-integration --test test_wcoj_recursive_dispatch -- --nocapture` | 8 passed |
| `cargo test -p xlog-runtime rewrite_scan_nth -- --nocapture` | 4 relevant tests passed; filtered binaries had 0 selected tests |
| `cargo test -p xlog-integration --test test_multiway_walker_contract -- --nocapture` | 6 passed |
| `RUSTFLAGS="-D warnings" cargo check -p xlog-integration --bin wcoj_phase_report --features wcoj-phase-timing` | PASS |
| `RUSTFLAGS="-D warnings" cargo bench -p xlog-integration --bench w52_skewed_multiway_bench --no-run` | PASS |
| `XLOG_W52_ONLY_CELL=5clique_N10 cargo bench -p xlog-integration --bench w52_skewed_multiway_bench -- --output-format bencher` | PASS; metadata ratio 0.009450 |
| `XLOG_W52_ONLY_CELL=pivot5_N10 cargo bench -p xlog-integration --bench w52_skewed_multiway_bench -- --output-format bencher` | PASS; metadata ratio 0.008668 |
| `RUSTFLAGS="-D warnings" cargo build --workspace --exclude pyxlog --tests` | PASS |

Criterion emitted sample-count warnings for the selected W52 timing cells, but both bench commands exited 0 and emitted the G_HIST_KC metadata-cost lines before the measured Criterion loop.
