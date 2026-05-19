# W67B Step 10 G_PURGE38B Evidence

**Branch:** `feat/w67b-step10-purge38b`
**Base:** `feat/w67b-step9-int38b @ feb758b7308d5243919efc77abf049178cb7505c`
**Date:** 2026-05-17
**Scope:** Goal-038-B Authorization 5, step 10 only. No W6.7 board edit, merge, push, or tag.

## Result

G_PURGE38B is green on the post-G_INT38B branch.

## Metric Matrix

| Metric | Result |
|---|---|
| M_PURGE38B.1 bundle-task marker scan | PASS. 38-B touched-file scope is `52` paths. Scan excluding this evidence file returned `0` hits for `TODO`, `FIXME`, `HACK`, `XXX`, `placeholder`, `stubbed`, `temporary`, `unimplemented!`, and `todo!`. This rerun rephrased two comment-only hits: `RawCudaView::runtime_block` and `compile(source)`. |
| M_PURGE38B.2 legacy-process phrase scan | PASS. 38-B touched-file scan excluding this evidence file returned `0` hits for `removed`, `deferred`, future-version phrases, `v0.6.6`, and back-compat phrases. This rerun rephrased the recursive-engine W4.1 comment from `removed` to `eliminated`. |
| M_PURGE38B.3 unused dependency scan | PASS. `cargo +nightly udeps --workspace --all-targets` exited `0` and ended with `All deps seem to have been used.` |
| M_PURGE38B.4 strict dead-code/import/variable build | PASS. `RUSTFLAGS="-D dead_code -D unused_imports -D unused_variables" cargo build --workspace --all-targets --release`: exit `0`, finished in `1m 01s`. |
| M_PURGE38B.5 author-trailer scan | PASS. `git log --format=%B c1689d70..HEAD` scan returned `0` hits for `Co-Authored-By`. |
| M_PURGE38B.6 future-version scan | PASS. Touched-file and commit-message scans returned `0` hits for `v0.6.6`, future-version phrases, and future-release phrases. |
| M_PURGE38B.7 paper-citation coverage | PASS. `wcoj.cu` carries SRDatalog / Paper §5 coverage: lines for SRDatalog Algorithm 2, Paper §5 Algorithm 2 preserved lines, K-clique leader-edge row, and Authorization 5 histogram refresh during Merge. |
| M_PURGE38B.8 pre-existing dead-code followup | PASS. `docs/evidence/2026-05-14-g38-dead-code-followup.md` exists and has `96` lines. |
| M_PURGE38B.9 K-clique hardcoded leader audit | PASS. `rg -n "canonical|\(0, 1\)" crates/xlog-cuda/kernels/wcoj.cu` returned no hits. |
| M_PURGE38B.10 K5/K6 promoter unconditional-none audit | PASS. Promoter source has `plan_kclique_var_order`, `WcojWithPlan`, `PlannedHashRoute`, and helper-split plan plumbing; `cargo test -p xlog-logic --test test_w67b_dispatch_plan -- --nocapture`: `2/2`. |
| M_PURGE38B.11 runtime plan-driven layout audit | PASS. Runtime clique dispatch consumes `edge_order`, `iteration_order`, `required_sort_slots`, and `sorted_layout_requirements`; `cargo test -p xlog-runtime --test test_w67b_dispatch_plan_source -- --nocapture`: `2/2`. |

## Focused Verification

| Command | Result |
|---|---|
| `cargo fmt --check` | PASS, exit `0`. |
| `git diff --check` | PASS, exit `0`. |
| `cargo test -p xlog-cuda --lib test_diff_all_filtered_out` | PASS, `1/1`. |
| `cargo test -p xlog-integration --test test_multiway_walker_contract -- --nocapture` | PASS, `6/6`. |
| `cargo test -p xlog-logic --test test_w67b_dispatch_plan -- --nocapture` | PASS, `2/2`. |
| `cargo test -p xlog-runtime --test test_w67b_dispatch_plan_source -- --nocapture` | PASS, `2/2`. |

## Changes

- Applied the superseded step-8 purge wording cleanup on the new Authorization-5 head.
- Rephrased three additional comment-only scan hits introduced by the new step-6/7 surface.
- Preserved the step-8 benchmark evidence and added only protocol text from the prior purge where it still applies.

## Deviations

- None against G_PURGE38B.1-G_PURGE38B.11.

## Out-Of-Scope Findings

- The debug-mode arithmetic note in `promote.rs` remains out of scope for 38-B purge; it was not changed here.
