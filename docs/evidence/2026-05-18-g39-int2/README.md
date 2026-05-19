# G_INT2 Phase-2 Integration Evidence

**Goal:** Goal-039 G_INT2, Phase-2 integration gate.
**Branch:** `feat/w6-bundle-integration-g39`
**Worktree:** `.worktrees/g39-w6-bundle-integration`
**Date:** 2026-05-18
**Scope:** Integration fixes plus M_INT2.1-13 verification. No main merge, push,
tag, or closure-board edit.

## Result

G_INT2 is green on the Phase-2 integration branch.

## Metric Matrix

| Metric | Result |
|---|---|
| M_INT2.1 W3.4 successor re-validation | PASS. `cargo test -p xlog-integration --bench wcoj_w33_superhub --no-run` exit 0. `cargo bench -p xlog-integration --bench wcoj_w33_superhub -- --output-format bencher`: `uniform-50K` row equality PASS rows=0, public `789,229 ns`, HG `230,488 ns`, ratio `3.424x`; `superhub-50K` row equality PASS rows=29,539, public `666,918 ns`, HG `158,424 ns`, ratio `4.209x`, gate `>= 1.51x`. |
| M_INT2.2 W4.1 cert regression | PASS. `cargo test -p xlog-integration --test test_wcoj_recursive_dispatch -- --nocapture`: `8 passed; 0 failed`. |
| M_INT2.3 W5.1 cert trio EXACT | PASS. `cargo test -p xlog-integration --release --test test_w51_same_generation_gpu_cert --test test_w51_skewed_multiway_gpu_cert --test test_w51_deep_recursive_wcoj_cert -- --nocapture`: `3 passed; 0 failed`. Same-generation row set `14`, `wcoj_4cycle_dispatch_count=1`; skewed multiway row set `4`, `wcoj_triangle_dispatch_count=1`, 4-cycle/clique counters `0`; deep recursive row set `4`, `wcoj_triangle_dispatch_count=6`, 4-cycle/clique counters `0`. |
| M_INT2.4 W5.2 bench corpus | PASS. `cargo test -p xlog-integration --bench w52_skewed_multiway_bench --no-run` exit 0. `cargo bench -p xlog-integration --bench w52_skewed_multiway_bench -- --output-format bencher` measured all 12 workload cells with expected route direction preserved; source/routing audits passed: `test_w67b_bench38b_source` `1/1`, `test_w52_measured_duration_source_audit` `1/1`, `test_w67b_cost_gate` `6/6`, including `w52_routing_decision_cert_is_36_of_36`. |
| M_INT2.5 W2.5 default-flip cert | PASS. `cargo test -p xlog-runtime test_w25_default_flip -- --nocapture`: runtime unit tests `2/2`; integration default-flip tests `3/3`. |
| M_INT2.6 Workspace fmt | PASS. `cargo fmt --check`: exit `0`. |
| M_INT2.7 Workspace build `-D warnings` | PASS. `RUSTFLAGS="-D warnings" cargo build --workspace --exclude pyxlog --tests --benches`: exit `0`. |
| M_INT2.8 Workspace test | PASS. `cargo test --workspace --exclude pyxlog --no-fail-fast`: exit `0`. |
| M_INT2.9 CUDA cert suite | PASS. `cargo test -p xlog-cuda-tests --test certification_suite --release -- --nocapture`: `207/207` passed, `0` failed, `0` skipped; pass rate `100.0%`; total duration `23.38s`; test finished in `26.13s`. |
| M_INT2.10 Peak VRAM | PASS. `cargo test -p xlog-cuda-tests --test g38_mint11_vram --release -- --nocapture`: peak label `C18_host_device`, peak delta `203,423,744` bytes, gate `40,802,189,312` bytes, total memory `12,820,480,000` bytes. W5.2 bench max observed VRAM delta `234,881,024` bytes. |
| M_INT2.11 Per-stream pool sizing | PASS. Added runtime contract in `StreamPool`: `XLOG_WCOJ_POOL_MB_PER_STREAM`, default `256` MiB per stream, and 4-arm x 4-stream planned budget helper. `cargo test -p xlog-cuda device_runtime::stream_pool::tests -- --nocapture`: stream-pool tests `7 passed; 0 failed`, including env override and default 4 x 4 budget `4,294,967,296` bytes. Observed release cert peak leaves more than the planned `3.2 GB` headroom record. |
| M_INT2.12 DLPack zero-copy preserved | PASS. `cargo test -p xlog-cuda --test dlpack_tests -- --nocapture`: `4/4`; `cargo test -p xlog-prob --test no_dtoh_in_gpu_neural_fast_path --test no_dtoh_in_neural_backward_nll --test no_dtoh_in_gpu_eval_device -- --nocapture`: `3/3`. |
| M_INT2.13 Witness-chain integrity | PASS. `cargo test -p xlog-prob --test test_provenance_primitives -- --nocapture`: `6 passed; 0 failed`, covering `choice_sources` and `leaf_atoms` accessors. |

## W5.2 Raw Cells

| Cell | GPU ns | Hash ns | VRAM delta bytes | Route direction |
|---|---:|---:|---:|---|
| 4cycle_N50 | 1,063,467 | 3,120,172 | 0 | GPU-favored |
| 4cycle_N250 | 776,308 | 2,878,302 | 0 | GPU-favored |
| 4cycle_N1000 | 1,012,701 | 4,994,600 | 33,554,432 | GPU-favored |
| 4cycle_N2000 | 1,672,212 | 11,625,493 | 234,881,024 | GPU-favored |
| 5clique_N10 | 30,629,101 | 12,255,585 | 0 | hash-favored |
| 5clique_N25 | 34,733,309 | 12,669,516 | 0 | hash-favored |
| 5clique_N50 | 36,330,565 | 13,629,917 | 0 | hash-favored |
| 5clique_N100 | 35,431,582 | 14,593,382 | 0 | hash-favored |
| pivot5_N10 | 37,054,887 | 15,409,056 | 0 | hash-favored |
| pivot5_N20 | 38,680,807 | 16,163,874 | 0 | hash-favored |
| pivot5_N30 | 40,867,822 | 18,330,190 | 0 | hash-favored |
| pivot5_N40 | 43,851,247 | 20,517,073 | 67,108,864 | hash-favored |

## Integration Fixes

The first full workspace run found four composition failures, all fixed in this
G_INT2 package before the final workspace rerun:

| Failure | Root cause | Fix | Targeted rerun |
|---|---|---|---|
| `strict_deterministic_d2h_inner_join_materialize_clean` duplicate `(1,4)` | Non-recursive `ChainJoin` stored a first-generation result without the fallback path's all-column dedup. | Dedup the no-existing-head `ChainJoin` result before `store_put`. | `cargo test -p xlog-integration --test executor_config_tests strict_deterministic_d2h_inner_join_materialize_clean -- --nocapture`: `1/1` |
| `small_small_dispatches_nested_loop_and_matches_hash` counter `0` | W63 chain dispatch used the W4.2 nested-loop provider but did not increment `nested_loop_dispatch_count`. | Track the nested-loop branch in `try_dispatch_w63_chain_on_body` and increment the counter after projection. | `cargo test -p xlog-integration --test test_w42_nested_loop_dispatch small_small_dispatches_nested_loop_and_matches_hash -- --nocapture`: `1/1` |
| `c6_pyxlog_walk_tmj_has_explicit_multiway_arm` source contract | `pyxlog` had a combined `MultiWayJoin | ChainJoin` arm, hiding the explicit MultiWayJoin descent from the contract test. | Split the arms while preserving behavior. | `cargo test -p xlog-integration --test test_multiway_walker_contract c6_pyxlog_walk_tmj_has_explicit_multiway_arm -- --nocapture`: `1/1` |
| `test_compile_with_named_stats_snapshot_reorders_joins` stale expectation | W63 now intentionally wraps the stats-ordered two-atom join in `ChainJoin`. | Update the test to assert both the chain node and captured fallback preserve named-stats ordering. | `cargo test -p xlog-logic test_compile_with_named_stats_snapshot_reorders_joins -- --nocapture`: `1/1` |

Follow-up full-target reruns:

```text
cargo test -p xlog-integration --test executor_config_tests --test test_multiway_walker_contract --test test_w42_nested_loop_dispatch -- --nocapture
executor_config_tests: 9 passed; 0 failed
test_multiway_walker_contract: 6 passed; 0 failed
test_w42_nested_loop_dispatch: 5 passed; 0 failed

cargo test -p xlog-logic --lib -- --nocapture
236 passed; 0 failed
```

## Hygiene

```text
git diff --check
cargo fmt --check
```

Both commands exited `0`.
