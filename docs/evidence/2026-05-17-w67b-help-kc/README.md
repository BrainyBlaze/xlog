# W6.7 Step 7 G_HELP_KC Evidence

Branch: `feat/w67b-step7-help-kc`
Base: `feat/w67b-step6-hist-kc @ 4de1d0bab5fb6ec66b597c436434009f02660b2f`
Scope: Goal-038-B Authorization 5, step 7 only. No W6.7 board edit, merge, push, or tag.

## Metric Status

| Metric | Status | Raw evidence |
|---|---|---|
| M_HELP_KC.1 | PASS | Planner cert: `cargo test -p xlog-logic --test test_hg_kclique_planner -- --nocapture` = 9 passed, including positive buried-skew helper spec and negative uniform-heat empty spec |
| M_HELP_KC.2 | PASS | Source audit: `cargo test -p xlog-logic --test test_w67b_help_kc -- --nocapture` = 3 passed; promoter now consumes planner `helper_split_specs` and compile pipeline invokes `helper_split_pass::run_kclique_specs` |
| M_HELP_KC.3 | PASS | Synthetic K=5 buried-skew compile fixture allocates exactly 1 `__w37_helper_*` relation; outer K-clique `MultiWayJoin` scans the helper relation |
| M_HELP_KC.4 | PASS | Runtime K5 integration cert: `cargo test -p xlog-integration --test test_wcoj_clique_dispatch -- --nocapture` = 6 passed; helper relation emits before the outer K-clique plan and composes with K-clique dispatch |
| M_HELP_KC.5 | PASS | Helper-split K5 rows equal direct K-clique rows on equivalent input: `rows=1`; bit-exact row-set equality PASS |
| M_HELP_KC.6 | PASS | No buried-skew regression: uniform K5 compile fixture allocates 0 helpers; W5.2 routing preserved with `cargo test -p xlog-logic --test test_w67b_cost_gate -- --nocapture` = 6 passed, including `w52_routing_decision_cert_is_36_of_36` |
| M_HELP_KC.7 | PASS | Required promoter citation comment present: `// Paper §5 Figure 3: Helper-relation splitting elevates buried inner-variable skew per Authorization 5 (2026-05-17)` |
| M_HELP_KC.8 | PASS | G_HIST_KC composition cert: post-split helper K5 path emitted `helper_relations=1`, `dispatch_count=1`, `metadata_build_count=1`, `metadata_build_nanos=317935`, `rows=1` |

## Verification Commands

All commands ran from `.worktrees/w67b-step7-help-kc`.

| Command | Result |
|---|---|
| `cargo fmt --check` | PASS |
| `git diff --check` | PASS |
| `cargo test -p xlog-logic --test test_hg_kclique_planner -- --nocapture` | 9 passed |
| `cargo test -p xlog-logic --test test_w67b_help_kc -- --nocapture` | 3 passed |
| `cargo test -p xlog-logic --test test_w67b_cost_gate -- --nocapture` | 6 passed |
| `cargo test -p xlog-logic --test test_w32_clique_promoter -- --nocapture` | 15 passed |
| `cargo test -p xlog-integration --test test_wcoj_clique_dispatch -- --nocapture` | 6 passed; helper K5 raw line: `helper_relations=1 dispatch_count=1 metadata_build_count=1 metadata_build_nanos=317935 rows=1` |
| `RUSTFLAGS="-D warnings" cargo build --workspace --exclude pyxlog --tests` | PASS |
