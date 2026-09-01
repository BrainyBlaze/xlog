# Runtime-optimization artifacts

Measurement records backing the two quantitative claims in the Runtime
Optimization subsection of `sections/10_evaluation.tex` (`sec:runtime-eval`).
Both are **single-system ablations** — xlog against its own baseline on the
development RTX PRO 3000 — not head-to-head comparisons; those live in
`../head-to-head/`.

| File | Claim | Fixture | n | Aggregation |
|------|-------|---------|---|-------------|
| `persistent_hash_index.json` | 3.21x with the persistent hash-index manager | build-heavy repeated-session semi-join, 8 x 8,000,000 rows | 9 timed, 12 warm-up | median per arm |
| `chain_shared_memory_scorer.json` | 5.58x with the profile-gated shared-memory chain scorer | chain-hot, 768 rows per candidate (gate threshold 256) | 12 timed, 3 warm-up | median per arm |

## Provenance

Both records were produced during the v0.8.6 runtime-completion campaign on
2026-05-19 and lived under `docs/evidence/`, which is an untracked agent
workspace; commit `5fd0f224` removed the whole directory from the tree and the
records went with it. The values here are transcribed verbatim from the
bundles at their measurement commits — `df2dbc03` for the index manager and
`ce78e32f` for the chain scorer — and each file names its own source bundle.

Neither bundle recorded the device. It is attributed from the same campaign's
runtime probe one day earlier
(`docs/evidence/2026-05-18-v080-exact/runtime_probe.json`), which names the
NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU and shares the host paths
and kernel packaging of these runs.

## Reproduction

- Index manager: `cargo test -p xlog-runtime test_persistent_hash_index_performance_fixture_meets_speedup_target -- --nocapture`. The test still exists and prints the raw medians; the gate is `persistent_hash_indexes` in `crates/xlog-core/src/config.rs` (env `XLOG_PERSISTENT_HASH_INDEXES`).
- Chain scorer: `scripts/measure_chain_shared_memory.py` (named `scripts/measure_v086_chain_smem.py` when the run was made). The A/B is driven by `XLOG_ILP_EXACT_CHAIN_SMEM`; the row gate is `XLOG_ILP_EXACT_CHAIN_SMEM_MIN_ROWS`, default 256.

## What the runners do not record

The index-manager runner emits only the median per arm, so that fixture has no
dispersion at all, and says so in its `protocol` block. The published
chain-scorer measurement has the same limitation one step smaller — median, min
and max, so a min-max range rather than an IQR.

`scripts/measure_chain_shared_memory.py` now emits every timed iteration
alongside the median, min and max, so a re-measured chain-scorer artifact
carries a real distribution. It also records the observed host-transfer count
per arm instead of asserting a fixed one: the invariant the claim rests on is
that the shared-memory arm adds no transfers, and that is a difference between
arms, not a constant of the card.
