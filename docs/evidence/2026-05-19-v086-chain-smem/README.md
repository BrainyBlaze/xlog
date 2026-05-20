# v0.8.6 G086_CHAIN_SMEM Evidence

Date: 2026-05-19
Goal node: G086_CHAIN_SMEM - Chain-Topology Shared-Memory Exact Scorer
Branch: `feat/v086-runtime-completion`
Worktree: `.worktrees/v086-runtime-completion`
Goal document: `docs/plans/2026-05-19-agent-v086-dts-runtime-completion-goal.md`

## GDSP / GQM Trace

GDSP consumer goal: reduce global-memory pressure for exact-induction chain
topology when profiling proves the chain scorer is hot, while keeping native
exact induction usable for DTS-DLM, v0.9.0, and pyxlog ILP users.

Existing xlog subsystem reused: pyxlog `induce_exact(backend="native")`,
`xlog_induce::induce_exact`, `CudaKernelProvider::ilp_exact_score`, and
`kernels/ilp_exact.cu`. The implementation extends the production CUDA
provider and kernel module; it does not add a chain-only scoring engine or a
Python-side scorer.

GQM questions answered:

- Q086_CHAIN.1: the profile-trigger commit recorded a certified synthetic
  chain-hot fixture before optimization.
- Q086_CHAIN.2: the optimized and baseline kernels return identical public
  coverage signatures for the small and chain-hot fixtures.
- Q086_CHAIN.3: the optimized kernel is faster on the chain-hot fixture and
  does not regress the small certified fixture.

## Artifacts

| Artifact | Purpose |
|---|---|
| `crates/xlog-cuda/kernels/ilp_exact.cu` | Adds `ilp_exact_score_chain_smem` and `ilp_exact_score_chain_smem_u32` entry points plus the tiled chain scorer over dynamic shared memory |
| `crates/xlog-cuda/src/provider/ilp_exact.rs` | Adds `XLOG_ILP_EXACT_CHAIN_SMEM`, the minimum-row guard, shared-memory byte sizing, and A/B dispatch |
| `crates/xlog-cuda/src/provider/mod.rs` | Names the chain-smem kernel entry points |
| `crates/xlog-cuda/src/kernel_manifest_data.rs` | Registers chain-smem entry points in the `ilp_exact` module manifest |
| `scripts/measure_v086_chain_smem.py` | Runs baseline versus chain-smem timing, parity, and transfer-budget measurement through pyxlog native exact induction |
| `python/tests/test_v086_chain_smem_source.py` | Source and evidence guard for the chain-smem implementation |
| `docs/architecture/bounded-exact-induction.md` | Public architecture note for typed dispatch plus profile-gated chain shared memory |

## Raw Measurements

| Measurement | Small baseline | Small smem | Chain-hot baseline | Chain-hot smem |
|---|---:|---:|---:|---:|
| Rows per candidate | `32` | `32` | `768` | `768` |
| Positive query pairs | `8` | `8` | `32` | `32` |
| Negative query pairs | `8` | `8` | `32` | `32` |
| Iterations | `12` | `12` | `12` | `12` |
| Warmup | `3` | `3` | `3` | `3` |
| Median seconds | `0.0011941255070269108` | `0.0011783409863710403` | `0.027507680497365072` | `0.004927040485199541` |
| Min seconds | `0.0008972260402515531` | `0.0011135960230603814` | `0.02734149800380692` | `0.004898363025858998` |
| Max seconds | `0.0012547150254249573` | `0.0016609960002824664` | `0.028317899035755545` | `0.00541784503730014` |
| D2H calls | `2` | `2` | `2` | `2` |

Chain-hot speedup ratio: `5.58300273358745x`.
Small fixture regression percent: `-1.3218477088869953`.
Added D2H calls: `0`.

Public coverage signatures:

- Small baseline and smem: `[["chain", "p_B", "p_C", 8, 0, 0]]`.
- Chain-hot baseline and smem: `[["chain", "p_B", "p_C", 32, 0, 0]]`.

Interpretation: PASS. The chain-hot fixture stays on the production pyxlog
native exact-induction path, the shared-memory kernel preserves strict
per-topology coverage, median runtime improves by more than `1.2x`, the
small fixture does not regress, and the provider records no added D2H calls.

## Metric Disposition

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M086_CHAIN.1 profile trigger | profile evidence names chain exact scorer as hot, or certified synthetic fixture documents why it is needed | PASS | `docs/evidence/2026-05-19-v086-chain-smem-profile/` was committed before optimization |
| M086_CHAIN.2 parity | shared-memory and baseline kernels produce identical coverage arrays | PASS | `measurements.json` reports identical non-empty result signatures for baseline and smem fixtures |
| M086_CHAIN.3 speedup | median kernel/runtime speedup >= 1.2x on hot fixture | PASS | chain-hot median speedup is `5.58300273358745x` |
| M086_CHAIN.4 small-case guard | no >5 percent regression on small certified fixtures | PASS | small fixture median delta is `-1.3218477088869953` percent |
| M086_CHAIN.5 transfer budget | zero added data-plane D2H/H2D transfers | PASS | baseline and smem runs both report `dtoh_calls=2`, with `added_dtoh_calls=0` |
| M086_CHAIN.6 fallback | non-chain topologies remain on existing path unless separately justified | PASS | source guard confirms non-chain cases stay in the baseline matcher inside the chain-smem kernels; provider dispatch is gated to chain-smem entry points only when enabled and row threshold is met |

## Validation Commands

| Command | Result |
|---|---|
| `cargo test -p xlog-cuda --lib ilp_exact_score_ -- --nocapture` | exit 0; 7 passed |
| `cargo test -p xlog-cuda kernel_modules` | exit 0; manifest tests 2 passed and filtered integration binaries passed |
| `VIRTUAL_ENV=/tmp/xlog-v086-exact-types-venv PATH=/tmp/xlog-v086-exact-types-venv/bin:$PATH /home/dev/.local/bin/maturin develop --manifest-path crates/pyxlog/Cargo.toml --features host-io,extension-module` | exit 0; installed `pyxlog-0.8.5` editable wheel |
| `/tmp/xlog-v086-exact-types-venv/bin/python scripts/measure_v086_chain_smem.py` | exit 0; JSON recorded in `measurements.json` |
| `pytest -q python/tests/test_v086_chain_smem_source.py python/tests/test_v086_chain_smem_profile_source.py` | exit 0; 4 passed |
| `python -m json.tool docs/evidence/2026-05-19-v086-chain-smem/measurements.json >/dev/null` | exit 0 |
| `python -m py_compile scripts/measure_v086_chain_smem.py python/tests/test_v086_chain_smem_source.py` | exit 0 |
| `/tmp/xlog-v086-exact-types-venv/bin/python -m pytest -q python/tests/test_ilp_exact_induce.py python/tests/test_v086_exact_types_runtime.py` | exit 0; 5 passed |
| `cargo fmt --check` | exit 0 |
| `git diff --check` | exit 0 |
| `python scripts/validate_package_metadata.py` | exit 0 |
| `cargo check --workspace` | exit 0 |

## Next-Step Decision

G086_CHAIN_SMEM has implementation evidence for parity, speedup, small-case
guarding, transfer budget, and fallback behavior. This evidence does not
authorize push, merge, tag, or release-board updates.
