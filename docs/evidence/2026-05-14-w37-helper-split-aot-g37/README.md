# W3.7 Helper-Split AOT Production Evidence

**Goal:** G37 / G4 W3.7 helper-relation splitting.
**Production branch:** `feat/w37-helper-split-aot-g37`.
**Production implementation commit:** `b2f43f50`.
**Base branch:** `bench-spike/w37-helper-split-hand-g37` at `7203a7e6`.
**Architectural anchor:** paper section 5 helper-relation splitting; G37 S4.2.

## Implementation Summary

- `crates/xlog-logic/src/optimizer.rs` adds `helper_split_pass`, an AOT pass that normalizes inner-join trees into relation leaves plus variable-class equalities, detects depth-buried skew using relation cardinality divided by column distinct count, emits a helper rule before the outer rule, and rewrites the outer body to scan that helper relation.
- `crates/xlog-logic/src/lower.rs` adds compiler-owned helper predicate allocation so helper relations receive stable `RelId` and schema entries.
- `crates/xlog-logic/src/compile.rs` runs the helper split immediately after lowering, then lets the normal optimizer, selectivity pass, and promotion pipeline process the rewritten plan.

## Metric Status

| Metric | Evidence | Verdict |
|---|---:|---|
| M4.1 callgraph analog speedup | `3.324x` from preserved spike `7203a7e6` | PASS |
| M4.2 heapalloc analog speedup | `47.849x` from preserved spike `7203a7e6` | PASS |
| M4.3 row equality | PASS on both spike fixtures; helper rewrite structural cert PASS | PASS |
| M4.4 no buried-skew rewrite | `helper_split_ignores_flat_distribution` PASS; no helper scan inserted | PASS |
| M4.5 W4.1 recursive composition | 3 targeted recursive dispatch certs PASS | PASS |

## Verification

| Command | Result |
|---|---:|
| `cargo test -p xlog-logic --lib helper -- --nocapture` | 3/0 PASS |
| `cargo test -p xlog-logic --lib` | 230/0 PASS |
| `RUSTFLAGS="-D warnings" cargo build -p xlog-logic --release` | EXIT 0 |
| `cargo bench -p xlog-integration --bench wcoj_w37_helper_split --no-run` | EXIT 0 |
| `cargo test -p xlog-integration --test test_wcoj_recursive_dispatch multirec_triangle_dispatches_wcoj_and_matches_binary_join -- --nocapture` | 1/0 PASS |
| `cargo test -p xlog-integration --test test_wcoj_recursive_dispatch multirec_4cycle_dispatches_wcoj_and_matches_binary_join -- --nocapture` | 1/0 PASS |
| `cargo test -p xlog-integration --test test_wcoj_recursive_dispatch selfrec_triangle_dispatches_wcoj_and_matches_binary_join -- --nocapture` | 1/0 PASS |
| `cargo fmt --check --all` | EXIT 0 |
| `git diff --check` | EXIT 0 |
| added-line process scan | clean |
| removed W3.3 skew-symbol source scan | clean |

## Production Cert Coverage

- `compile::tests::test_compile_with_named_stats_snapshot_creates_helper_relation` proves the compiler hook allocates and scans a helper relation from named stats.
- `optimizer::helper_split_pass_tests::helper_split_extracts_buried_pair` proves the pass extracts the `(de, ef)` inner pair and inserts the helper rule before the outer rule.
- `optimizer::helper_split_pass_tests::helper_split_ignores_flat_distribution` proves a flat distribution remains on the original rule shape, which is the M4.4 no-regression guard for rules with no buried skew.

## Review Notes

- The pass intentionally runs before `Optimizer::optimize`; lowerer-side bushy planning otherwise hides the source-order deep pair. The pass now normalizes both left-deep and bushy inner-join trees, so it is not dependent on source syntax shape.
- No CUDA provider or kernel files changed in G4 production.
- No new environment controls were added.
