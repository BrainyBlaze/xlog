# v0.8.6 G086_CHAIN_SMEM Profile Trigger Evidence

Date: 2026-05-19
Goal node: G086_CHAIN_SMEM - Chain-Topology Shared-Memory Exact Scorer
Evidence phase: PROFILE_TRIGGER_ONLY
Branch: `feat/v086-runtime-completion`
Worktree: `.worktrees/v086-runtime-completion`
Goal document: `docs/plans/2026-05-19-agent-v086-dts-runtime-completion-goal.md`

## GDSP / GQM Trace

GDSP consumer goal: determine whether chain-topology exact induction has a
measurable hot profile before adding shared-memory caching to the production
kernel.

Existing xlog subsystem reused: pyxlog `induce_exact(backend="native")`,
`xlog_induce::induce_exact`, `CudaKernelProvider::ilp_exact_score`, and
`kernels/ilp_exact.cu`. This profile trigger does not add a new scorer or
modify production dispatch.

GQM question answered in this pre-optimization commit:

- Q086_CHAIN.1: a certified synthetic fixture justifies optimizing the chain
  topology. The fixture routes through the native exact-induction path and
  shapes rows so the chain predicate performs the worst-case `|L| * |R|`
  scan while non-chain topology checks remain linear.

## Fixture

The profile script constructs:

- `p_B(X, Z)` with `768` rows where every `X` equals the positive query key.
- `p_C(Z, Y)` with matching `Z` keys but no row containing the queried `Y`.
- `32` positive and `32` negative query pairs.

This forces the chain predicate to scan all right rows for every matching left
row and query pair. It still uses normal pyxlog relation upload and native
exact-induction dispatch, so it is a production-path profile trigger rather
than a direct kernel microbenchmark.

## Raw Measurements

| Measurement | Small fixture | Chain-hot fixture |
|---|---:|---:|
| Rows per candidate | `32` | `768` |
| Positive query pairs | `8` | `32` |
| Negative query pairs | `8` | `32` |
| Iterations | `12` | `12` |
| Warmup | `3` | `3` |
| Median seconds | `0.0008779879717621952` | `0.027476257004309446` |
| Min seconds | `0.0007938360213302076` | `0.027041922963690013` |
| Max seconds | `0.0010512439766898751` | `0.027681432955432683` |
| D2H calls | `2` | `2` |

Hot/small median ratio: `31.294571096643047x`.

Interpretation: PASS for M086_CHAIN.1. The synthetic profile documents a
chain-heavy path whose runtime is dominated by the existing chain scan while
the D2H budget remains fixed. This authorizes the next commit to add an A/B
shared-memory chain implementation and compare it against this baseline.

## Metric Disposition

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M086_CHAIN.1 profile trigger | profile evidence names chain exact scorer as hot, or certified synthetic fixture documents why it is needed | PASS | `scripts/measure_v086_chain_smem.py` and `measurements.json` record a `31.294571096643047x` hot/small median ratio |
| M086_CHAIN.2 parity | shared-memory and baseline kernels produce identical coverage arrays | PROFILE_TRIGGER_ONLY | deferred to the optimization commit |
| M086_CHAIN.3 speedup | median kernel/runtime speedup >= 1.2x on hot fixture | PROFILE_TRIGGER_ONLY | deferred to the optimization commit |
| M086_CHAIN.4 small-case guard | no >5 percent regression on small certified fixtures | PROFILE_TRIGGER_ONLY | deferred to the optimization commit |
| M086_CHAIN.5 transfer budget | zero added data-plane D2H/H2D transfers | PROFILE_TRIGGER_ONLY | baseline profile reports fixed `dtoh_calls=2`; added-transfer comparison deferred |
| M086_CHAIN.6 fallback | non-chain topologies remain on existing path unless separately justified | PROFILE_TRIGGER_ONLY | source comparison deferred to the optimization commit |

## Validation Commands

| Command | Result |
|---|---|
| `pytest -q python/tests/test_v086_chain_smem_profile_source.py` before evidence | exit 1; evidence README missing, script source guard passed |
| `/tmp/xlog-v086-exact-types-venv/bin/python scripts/measure_v086_chain_smem.py` | exit 0; JSON recorded in `measurements.json` |
| `pytest -q python/tests/test_v086_chain_smem_profile_source.py` after evidence | exit 0; 2 passed |
| `python -m json.tool docs/evidence/2026-05-19-v086-chain-smem-profile/measurements.json >/dev/null` | exit 0 |
| `python -m py_compile scripts/measure_v086_chain_smem.py python/tests/test_v086_chain_smem_profile_source.py` | exit 0 |
| `git diff --check` | exit 0 |

## Next-Step Decision

The profile trigger is ready to commit as the required pre-optimization record.
The next G086_CHAIN_SMEM commit may implement shared-memory caching for chain
topology only, with A/B controls, parity, speedup, small-case, fallback, and
transfer-budget evidence. This evidence does not authorize push, merge, tag,
or release-board updates.
