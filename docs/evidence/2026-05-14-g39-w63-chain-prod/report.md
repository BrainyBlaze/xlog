# Goal-039 G_W63_CHAIN Production

Date: 2026-05-17.
Branch: `feat/w63-chain-promoter-prod-g39`.
Base: `bench-spike/w63-chain-promoter-g39` at `bfae71a747c5101749e0d73d6372d102b3b48d24`.

## Scope

G_PRE measured `evaluate_pct = 0.9661778834300995`, so G_W63_CHAIN priority is
HIGH. This branch productionizes the W63 spike by replacing the two-input
`MultiWayJoin` wrapper with a first-class `RirNode::ChainJoin` and routing it
through the W6.3 chain dispatcher.

## Implementation

- `crates/xlog-ir/src/rir.rs`: adds `RirNode::ChainJoin`.
- `crates/xlog-logic/src/promote.rs`: 2-atom inner-chain promotion now emits
  `ChainJoin`.
- `crates/xlog-runtime/src/executor/wcoj_dispatch.rs`: chain dispatcher matches
  `ChainJoin`, records `StatsManager::record_join_result`, and uses:
  - sorted threshold cell: W4.3 sort-merge;
  - sorted large one-to-one cell: bounded sort-merge;
  - unsorted threshold cell: W4.2 nested-loop;
  - else: hash join.
- `crates/xlog-cuda/src/provider/relational.rs`: adds bounded sorted-chain
  sort-merge. It allocates caller-bounded output capacity and fails closed to
  hash when duplicates overflow the bound.
- `crates/xlog-runtime/src/executor/rewrite.rs`: recursive `rewrite_scan_nth`
  updates both `ChainJoin` dispatch inputs and fallback, preserving P4-style
  delta substitution.
- `crates/pyxlog/src/ilp.rs`: fallback walker descends through `ChainJoin`.

## Metric Status

| Metric | Status | Raw result |
|---|---:|---|
| M_W63.1 m37c-prime trace subset speedup | PASS | 128 chain-shaped G_PRE invocations; recorded baseline `evaluate_ns = 81019.497 ms`; ChainJoin replay `86.819 ms`; ratio `933.200479x`; dispatches `128`; output rows `12998`. |
| M_W63.2 synthetic 977K speedup | PASS | `W63_CHAIN_TIMING synthetic_977k n=977000 iterations=3 fallback_ms=85.060 chain_ms=5.432 ratio=15.659175 fallback_dispatches=0 chain_dispatches=3`. |
| M_W63.3 row equality | PASS | `chain_dispatch_default_on_matches_env_disabled_fallback`: 128 rows, default-on rows equal env-disabled fallback rows. |
| M_W63.4 triangle/cycle/clique regression | PASS (route/correctness cert) | `test_wcoj_dispatch`: 8/8; `test_wcoj_4cycle_rir_shape_cert`: 4/4; K5/K6 clique counter/parity certs: 2/2. No separate ±3% perf A/B was run in this branch. |
| M_W63.5 zero misroutes | PASS | Logic rejects non-inner and multi-key joins; runtime matcher rejects non-scan `ChainJoin` and `MultiWayJoin` triangle. |
| M_W63.6 helper-split composition | PASS | `test_compile_with_named_stats_snapshot_creates_helper_relation`: helper-split helper rule is promoted to `ChainJoin`; consumer still scans helper relation. |
| M_W63.7 P4 delta-outermost cert | PASS | `chain_join_rewrite_scan_nth_updates_dispatch_shape_and_fallback`: dispatch inputs and fallback rewrite the same target occurrence to delta. |
| M_W63.8 peak VRAM | PASS (bounded) | Largest W63 acceptance cell ran under a `512 MiB` device budget, below the `38 GiB` gate. |

Supplemental threshold-cell timing:
`W63_CHAIN_TIMING sorted_threshold n=2000 iterations=20 fallback_ms=200.511 chain_ms=30.169 ratio=6.646302 fallback_dispatches=0 chain_dispatches=20`.

## Commands

```bash
cargo test -p xlog-ir chain_join_is_not_a_leaf_and_walks_inputs_once
cargo test -p xlog-logic chain
cargo test -p xlog-logic test_compile_with_named_stats_snapshot_creates_helper_relation
cargo test -p xlog-runtime chain
cargo test -p xlog-integration --test test_w63_chain_promoter_spike -- --nocapture
cargo test -p xlog-integration --test test_w63_chain_promoter_spike -- --ignored --nocapture
cargo test -p xlog-integration --test test_w63_chain_promoter_spike chain_dispatch_timing_m37c_trace_subset_128 -- --ignored --nocapture
cargo test -p xlog-integration --test test_wcoj_dispatch -- --nocapture
cargo test -p xlog-integration --test test_wcoj_4cycle_rir_shape_cert -- --nocapture
cargo test -p xlog-integration --test test_wcoj_clique_dispatch -- clique5_dispatch_counter_advances_and_row_set_matches_fallback_body clique6_dispatch_counter_advances_and_row_set_matches_fallback_body --nocapture
cargo check --workspace --all-targets
```

## Caveat

M_W63.4 has correctness and route-preservation evidence, but not a separate
triangle/cycle/clique performance A/B with a measured ±3% delta. The code path
for those shapes remains `MultiWayJoin`; W63's matcher rejects `MultiWayJoin`
before the existing triangle, 4-cycle, and K-clique dispatchers run.
