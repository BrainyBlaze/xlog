# Runtime-optimization artifacts

Measurement records backing the two quantitative claims in the Runtime
Optimization subsection of `sections/10_evaluation.tex` (`sec:runtime-eval`).
Both are **single-system ablations** — xlog against its own baseline — not
head-to-head comparisons; those live in `../head-to-head/`.

| File | Claim | Hardware | Fixture | n | Aggregation |
|------|-------|----------|---------|---|-------------|
| `persistent_hash_index.json` | 8.334x with the persistent hash-index manager | A100 80GB PCIe, release build | build-heavy repeated-session semi-join, 8 x 8,000,000 rows | 9 timed, 12 warm-up | median per arm |
| `chain_shared_memory_scorer.json` | 7.198x with the profile-gated shared-memory chain scorer | A100-SXM4 80GB | chain-hot, 768 rows per candidate (gate threshold 256) | 12 timed, 3 warm-up | median per arm |

Two things about those numbers that an earlier version of this file got wrong,
and that the reader needs before using them:

**The index fixture is now measured in release, and the earlier caveat here
was backwards.** A previous version of this file recorded the cargo command
with a `--release` flag the run had not used; the measurement behind `7.078x`
was a debug build, and this file then guessed that release would come out
*lower*. It does not. A re-run put both profiles on one pod, one commit, one
fixture: release `8.334x`, debug `7.716x`. The arms sit on different sides of
the bus — the index-rebuilding arm is GPU-bound and barely moves (15.83 ms
debug against 15.78 ms release), while the cached arm is host-side and does
get faster (2.05 ms against 1.89 ms) — so optimizing the build widens the gap
instead of closing it. The fixture's own >=1.5x gate passed in both profiles.
Worth keeping in view: the debug arm of that re-run reads 7.716x against the
7.078x of the first debug run on a different pod, so about 8% of the spread
on this fixture is host rather than profile.

**The chain scorer's ratio rose partly because its baseline got slower.** Both
arms are slower here in absolute terms than in the earlier record — baseline
27.51 ms to 52.42 ms, optimized 4.93 ms to 7.28 ms — and the ratio moved from
5.58x to 7.198x only because the baseline slowed by more. That is a change of
machine and of engine version, not a like-for-like gain.

## Provenance

Both records were **re-measured on ephemeral RunPod GPUs** — the chain scorer
on 2026-09-02, the index manager on 2026-09-05 — and each file now records
its own device, driver, CPU quota and commit.

That replaces the previous situation, which is worth remembering because it is
the failure this directory was written to avoid. The earlier values (3.21x and
5.58x) came from the v0.8.6 campaign of 2026-05-19 and lived under
`docs/evidence/`, an untracked agent workspace; commit `5fd0f224` removed the
directory and the records went with it. Neither bundle recorded the device, so
it had to be *attributed* — from a runtime probe taken a day earlier that named
an RTX PRO 3000 laptop GPU. Attribution is not measurement, and the new files do
not need it.

The new numbers are larger than the old ones on both fixtures, and it would be
wrong to put that down to hardware alone. These runs are at `a2bafef0`, the same
engine build as the head-to-head set; the records they replace are from the
v0.8.6 campaign at `df2dbc03` and `ce78e32f`. Machine, build profile and engine
version all differ, so neither ratio is a controlled before-and-after of the
optimization — each is a fresh measurement of the current engine on a named
machine.

## Reproduction

- Index manager:
  `cargo test -p xlog-runtime --release test_persistent_hash_index_performance_fixture_meets_speedup_target -- --nocapture`.
  The test prints the raw medians; the artifact carries that line verbatim under
  `harness.raw_line`. The gate is `persistent_hash_indexes` in
  `crates/xlog-core/src/config.rs` (env `XLOG_PERSISTENT_HASH_INDEXES`).
- Chain scorer: `scripts/measure_chain_shared_memory.py`. The A/B is driven by
  `XLOG_ILP_EXACT_CHAIN_SMEM`; the row gate is
  `XLOG_ILP_EXACT_CHAIN_SMEM_MIN_ROWS`, default 256.

## What the records do and do not carry

The index-manager test emits only the median per arm, so that fixture still has
no dispersion, and the artifact says so.

The chain scorer no longer has that limitation: the runner records every timed
iteration alongside the median, min and max, so the new artifact carries a real
distribution. It also records the observed host-transfer count per arm instead
of asserting a fixed one. That matters — the previous script asserted
`dtoh_calls == 2` and aborted three separate runs on this hardware, where both
arms make one. The invariant the claim rests on is that the shared-memory arm
adds **no** transfers relative to baseline, which is a difference between arms
and not a constant of the card; the artifact reports `added_dtoh_calls: 0`.
