# v0.8.6 G086_EXACT_TYPES Evidence

Date: 2026-05-19
Goal node: G086_EXACT_TYPES - Native Exact-Induction U32/Symbol Dispatch
Branch: `feat/v086-runtime-completion`
Worktree: `.worktrees/v086-runtime-completion`
Goal document: `docs/plans/2026-05-19-agent-v086-dts-runtime-completion-goal.md`

## GDSP / GQM Trace

GDSP consumer goal: let DTS-DLM, v0.9.0 epistemic/solver, and pyxlog ILP
users run native exact induction on typed tensorized schemas beyond `U64`
without host-side widening or schema erasure.

Existing xlog subsystem reused: `xlog_induce::induce_exact`,
`CudaKernelProvider::ilp_exact_score`, `kernels/ilp_exact.cu`, pyxlog
`CompiledIlpProgram::induce_exact_native`, and schema-checked DLPack import.
No separate exact-induction engine or Python-side scoring path was added.

GQM questions answered:

- Q086_EXACT_TYPES.1: `U32` pair buffers score through the native provider
  path and match the strict Python reference.
- Q086_EXACT_TYPES.2: `Symbol` pair buffers use the `u32` physical kernel
  while preserving `symbol` relation type annotations.
- Q086_EXACT_TYPES.3: runtime parity holds for `U32`, `Symbol`, and the
  existing `U64` release fixture.
- Q086_EXACT_TYPES.4: D2H count remains exactly two count-array transfers for
  small and larger typed requests, independent of candidate-pair count.

## Artifacts

| Artifact | Purpose |
|---|---|
| `crates/xlog-induce/src/lib.rs` | Engine-side typed pair validation and exact logical type matching |
| `crates/xlog-cuda/src/provider/ilp_exact.rs` | Provider-side typed layout validation, `U64`/`U32` dispatch, D2H counter budget |
| `crates/xlog-cuda/kernels/ilp_exact.cu` | Templated matcher plus `ilp_exact_score_u32` CUDA entry for `U32`/`Symbol` physical rows |
| `crates/xlog-cuda/src/kernel_manifest_data.rs` | Registers `ilp_exact_score_u32` in the `ilp_exact` module |
| `crates/xlog-core/src/types.rs` | Schema-checked DLPack compatibility for physical `u32` symbol IDs |
| `crates/pyxlog/src/ilp_exact.rs` | Existing pyxlog native bridge reused for typed buffers |
| `docs/architecture/bounded-exact-induction.md` | Public typed-dispatch architecture and D2H contract |
| `python/tests/test_v086_exact_types_runtime.py` | Runtime `U32`/`Symbol` parity, type annotation, D2H, and mixed-type diagnostics |
| `python/tests/test_v086_exact_types_source.py` | Source guards for typed native dispatch and evidence |
| `docs/evidence/2026-05-19-v086-exact-types/measurements.json` | Raw typed parity, transfer, and diagnostic measurements |

## Raw Measurements

| Measurement | Value |
|---|---|
| Runtime venv | `/tmp/xlog-v086-exact-types-venv` |
| `U32` 3-candidate parity | `true` |
| `U32` 5-candidate parity | `true` |
| `U32` D2H calls | `2` for both 3 and 5 candidates |
| `U32` schema | `["u32", "u32"]` |
| `Symbol` 3-candidate parity | `true` |
| `Symbol` 5-candidate parity | `true` |
| `Symbol` D2H calls | `2` for both 3 and 5 candidates |
| `Symbol` schema | `["symbol", "symbol"]` |
| Mixed logical type error | `induce_exact: candidate[0] buffer type mismatch: expected U32, got Symbol` |
| Provider typed tests | `7 passed` |
| Engine unit tests | `23 passed` |
| Core DLPack compatibility test | `1 passed` |

Interpretation: PASS. `U32` and `Symbol` requests route through the native
exact-induction engine, keep constant count-array D2H, and reject mixed
logical pair types. `Symbol` uses the 32-bit physical kernel only after
schema-checked import, so public schema identity is preserved.

## Metric Disposition

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M086_EXACT_TYPES.1 U32 dispatch | native `U32` exact-induction fixture passes parity | PASS | `test_induce_exact_native_matches_python_reference_for_32_bit_pair_types[u32]` and measurements show parity `true` |
| M086_EXACT_TYPES.2 Symbol dispatch | native `Symbol` exact-induction fixture passes parity and preserves schema names/types | PASS | `symbol` runtime fixture parity `true`; `relation_type_annotations()["p_A"] == ["symbol", "symbol"]` |
| M086_EXACT_TYPES.3 U64 non-regression | existing `U64` tests remain green | PASS | `python/tests/test_ilp_exact_induce.py` remains in validation set; provider U64 hand/determinism/empty-negative tests passed |
| M086_EXACT_TYPES.4 transfer budget | count-array D2H bounded exactly as documented; no type-conversion D2H | PASS | `U32` and `Symbol` 3/5-candidate runs all report `dtoh_calls=2`; source guard rejects provider `download_column` use |
| M086_EXACT_TYPES.5 typed diagnostics | mixed/unsupported types fail with explicit typed errors | PASS | Python mixed `U32`/`Symbol` fixture reports exact type mismatch; provider rejects `I32` with `expected U64, U32, or Symbol` |
| M086_EXACT_TYPES.6 consumer fixture | DTS-DLM or Mistaber typed ILP fixture exercises at least one non-`U64` path | PASS | DTS-shaped bounded ILP fixture exercises both `U32` and `Symbol` tensorized exact-induction paths |

## Validation Commands

| Command | Result |
|---|---|
| `cargo test -p xlog-cuda --lib ilp_exact_score_ -- --nocapture` | exit 0; 7 passed |
| `cargo test -p xlog-induce --lib` | exit 0; 23 passed |
| `cargo test -p xlog-core test_dlpack_compatible` | exit 0; 1 passed |
| `/tmp/xlog-v086-exact-types-venv/bin/python -m pytest -q python/tests/test_v086_exact_types_runtime.py` | exit 0; 3 passed in 7.80s |
| `pytest -q python/tests/test_v086_exact_types_source.py python/tests/test_v080_exact_source.py` before evidence | exit 1; evidence README missing, 5 source tests passed |
| `pytest -q python/tests/test_v086_exact_types_source.py python/tests/test_v080_exact_source.py` after evidence | exit 0; 6 passed |
| `/tmp/xlog-v086-exact-types-venv/bin/python -m pytest -q python/tests/test_ilp_exact_induce.py python/tests/test_v086_exact_types_runtime.py` | exit 0; 5 passed in 15.95s |
| `cargo test -p xlog-cuda kernel_modules` | exit 0; manifest tests 2 passed and filtered integration binaries passed |
| `cargo test -p pyxlog --lib` | exit 0; 7 passed |
| `cargo fmt --check` | exit 0 |
| `git diff --check` | exit 0 |
| `python scripts/validate_package_metadata.py` | exit 0 |
| `python -m json.tool docs/evidence/2026-05-19-v086-exact-types/measurements.json >/dev/null` | exit 0 |
| `cargo check --workspace` | exit 0 |
| typed measurement probe in `/tmp/xlog-v086-exact-types-venv` | exit 0; JSON recorded in `measurements.json` |

## Next-Step Decision

G086_EXACT_TYPES is ready for final hygiene checks and a slice commit. Proceed
next to G086_CHAIN_SMEM only after this implementation and evidence are
committed. This evidence does not authorize push, merge, tag, or release-board
updates.
