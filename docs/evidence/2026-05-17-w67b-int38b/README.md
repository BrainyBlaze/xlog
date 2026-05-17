# W67B Step 9 G_INT38B Evidence

**Branch:** `feat/w67b-step9-int38b`
**Base:** `feat/w67b-step8-bench38b @ 106a7c5853236a847933857209e7df02dbba54e9`
**Date:** 2026-05-17
**Scope:** Goal-038-B Authorization 5, step 9 only. No W6.7 board edit, merge, push, or tag.

## Result

G_INT38B is green on the post-G_HIST_KC + post-G_HELP_KC + rerun-G_BENCH38B branch.

## Metric Matrix

| Metric | Result |
|---|---|
| M_INT38B.1 W3.4 successor revalidation | PASS. `cargo test -p xlog-integration --bench wcoj_w33_superhub --no-run` exit 0. `cargo bench -p xlog-integration --bench wcoj_w33_superhub -- --output-format bencher`: `uniform-50K` row equality PASS rows=0, public `717,674 ns`, HG block-slice `218,312 ns`, ratio `3.287x`; `superhub-50K` row equality PASS rows=29,539, public `1,054,998 ns`, HG block-slice `159,630 ns`, ratio `6.609x`, gate `>= 1.51x`. |
| M_INT38B.2 W4.1 cert regression | PASS. `cargo test -p xlog-integration --test test_wcoj_recursive_dispatch -- --nocapture`: `8 passed; 0 failed`. |
| M_INT38B.3 W5.1 cert trio EXACT | PASS. Same Generation: `wcoj_4cycle_dispatch_count=1`, row set `14`; skewed multiway: `wcoj_triangle_dispatch_count=1`, row set `4`, clique counters `0`; deep-recursive: `wcoj_triangle_dispatch_count=6`, row set `4`, 4-cycle/clique counters `0`. Commands: `test_w51_same_generation_gpu_cert`, `test_w51_skewed_multiway_gpu_cert`, `test_w51_deep_recursive_wcoj_cert`, all `1/1`. |
| M_INT38B.4 W5.2 amended per-path | PASS. Committed Step 8 evidence at `106a7c58`: `24/24` accepted path medians `<= 1.10x` same-machine W5.2 baseline (`12/12` GPU-WCOJ, `12/12` hash-chain); VRAM `126` current snapshots, max delta `234,881,024` bytes, gate `40,802,189,312` bytes. Source/routing audits refreshed: `test_w67b_bench38b_source` `1/1`, `test_w52_measured_duration_source_audit` `1/1`, `test_w67b_cost_gate` `6/6` including `w52_routing_decision_cert_is_36_of_36`. |
| M_INT38B.5 W2.5 default-flip | PASS. `cargo test -p xlog-runtime test_w25_default_flip -- --nocapture`: runtime unit tests `2/2`; integration default-flip tests `3/3`. |
| M_INT38B.6 W3.2 K=5/K=6 clique cert grid | PASS. `cargo test -p xlog-cuda --test test_wcoj_clique5 -- --nocapture`: `4/4`; `cargo test -p xlog-cuda --test test_wcoj_clique6 -- --nocapture`: `4/4`; metadata bit-exact 100-run loops included for both K5 and K6. |
| M_INT38B.7 Workspace fmt | PASS. `cargo fmt --check`: exit `0`. |
| M_INT38B.8 Workspace build `-D warnings` | PASS. `RUSTFLAGS="-D warnings" cargo build --workspace --exclude pyxlog --tests --benches`: exit `0`. |
| M_INT38B.9 Workspace test | PASS. `cargo test --workspace --exclude pyxlog --no-fail-fast`: exit `0`. |
| M_INT38B.10 CUDA cert suite | PASS. `cargo test -p xlog-cuda-tests --test certification_suite --release -- --nocapture`: `206/206` passed, `0` failed, `0` skipped; total duration `29.52s`; compute capability `12.0`; mode `legacy/default`. |
| M_INT38B.11 Peak VRAM | PASS. `cargo test -p xlog-cuda-tests --test g38_mint11_vram --release -- --nocapture`: peak delta `201,326,592` bytes at `C18_host_device`, gate `40,802,189,312` bytes, total memory `12,820,480,000` bytes. Step 8 W5.2 bench peak delta also below gate at `234,881,024` bytes. |
| M_INT38B.12 DLPack zero-copy preserved | PASS. `cargo test -p xlog-cuda --test dlpack_tests -- --nocapture`: `4/4`; no-DtoH guards `no_dtoh_in_gpu_neural_fast_path`, `no_dtoh_in_neural_backward_nll`, `no_dtoh_in_gpu_eval_device`: `3/3`. |
| M_INT38B.13 Witness-chain recoverable | PASS. `cargo test -p xlog-prob --test test_provenance_primitives -- --nocapture`: `6/6`, including `leaf_atom` and `choice_source` accessor coverage. |
| M_INT38B.14 M37-A surface preserved | PASS. Phase 2 has not shipped, so this is a surface-presence cert. `git diff --name-only c1689d70..HEAD -- crates/pyxlog crates/xlog-prob crates/xlog-neural crates/xlog-induce` returned `0` paths. `cargo test -p xlog-logic --test parse_neural -- --nocapture`: `6/6`. Source audit confirms Group-B/API symbols remain present, including `nn(...)`, `register_network`, `register_embedding`, `forward_backward_tensor`, `train_epoch`, `max_grad_norm`, `patience`, XGCF/GPU circuit cache, `grad_true`, `grad_false`, `induce_exact`, and `train_and_promote`. |
| M_INT38B.15 Hypergraph planner is production K5/K6 path | PASS. `cargo test -p xlog-logic --test test_w67b_dispatch_plan -- --nocapture`: `2/2`; `cargo test -p xlog-runtime --test test_w67b_dispatch_plan_source -- --nocapture`: `2/2`. `rg -n "canonical|\(0, 1\)" crates/xlog-cuda/kernels/wcoj.cu` returned no hits. |

## Deviations

- The inherited goal text row for M_INT38B.4 says `36/36 GPU paths` and `36/36 hash paths`. Authorization 5 step 8 uses the current amended corpus shape: 12 workload cells x 2 paths = 24 gated path rows, with median-of-3 samples and source-audited direct timing. This evidence uses the committed step-8 rerun.
- M_INT38B.14 is a surface-presence cert because Phase 2 / G_M37A_SURFACE has not shipped. No M37-A production replay is claimed here.

## Hygiene

- `rg -n "w52_literal_gate" crates/xlog-integration/benches/w52_skewed_multiway_bench.rs` returned no hits.
- The step-9 branch only adds this evidence file on top of step 8.
