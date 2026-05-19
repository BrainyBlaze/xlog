# v0.8.6 Runtime Completion Closure Proposal

Date: 2026-05-19
Branch: `feat/v086-runtime-completion`
Validation head after persistent-index amendment: `0e2a5420`
Governing goal: `docs/plans/2026-05-19-agent-v086-dts-runtime-completion-goal.md`

## Recommendation

`MERGE_READY` after final validation.

The branch is integration-clean for the implemented v0.8.6 runtime and
optimizer surfaces: formatting, workspace check, runtime/cuda/induce/prob/logic
/integration Rust suites, Python source/runtime guards, v0.8.0/v0.8.5/v0.8.6
example validators, JSON validation, package metadata validation, and diff
whitespace checks passed.

The previous scope blocker was removed: G086_INDEX now validates deterministic
persistent index reuse, complete stale-key rejection, LRU budget eviction,
device/schema/generation keying, background request/completion/deferred
telemetry, and a runtime-backed recorded provider build path that composes the
existing recorded pack/table helpers. The performance evidence still does not
claim a >=1.5x timing speedup for the persistent-index fixture; it claims
observed retained-index reuse and records the unclaimed timing speedup
explicitly.

No merge, push, tag, or release-board update is authorized or performed by this
proposal.

## Sub-Goal Table

| Goal | Commit | Status | Evidence |
|---|---|---|---|
| G086_PRE | `e6edb49c` | PASS | `docs/evidence/2026-05-19-v086-pre/README.md` |
| G086_DELTA_COALESCE | `6b18a7b9` | PASS | `docs/evidence/2026-05-19-v086-delta-coalesce/README.md` |
| G086_NOTIFY | `9a79faea` | PASS | `docs/evidence/2026-05-19-v086-notify/README.md` |
| G086_EXACT_TYPES | `d1967d94` | PASS | `docs/evidence/2026-05-19-v086-exact-types/README.md` |
| G086_CHAIN_SMEM | `e1cddbb7` + `ce78e32f` | PASS | `docs/evidence/2026-05-19-v086-chain-smem-profile/README.md`, `docs/evidence/2026-05-19-v086-chain-smem/README.md` |
| G086_CSE | `1363b05e` | PASS | `docs/evidence/2026-05-19-v086-cse/README.md` |
| G086_ADAPT | `2d9bdc0f` | PASS | `docs/evidence/2026-05-19-v086-adaptive-reoptimization/README.md` |
| G086_INDEX | `702e1f8f` + `0e2a5420` | PASS | `docs/evidence/2026-05-19-v086-persistent-hash-index/README.md` |
| G086_CONSUMERS | `37f16651` | PASS | `docs/evidence/2026-05-19-v086-consumers/README.md` |
| G086_INT | `b72f61ea` + final validation refresh | PASS | `docs/evidence/2026-05-19-v086-int/README.md` |
| G086_CLOSE | `09f5ad4b` + final evidence refresh | PASS | `docs/evidence/2026-05-19-v086-close/README.md` |

## GQM Metric Table

| Metric | Status | Raw result |
|---|---|---|
| M086_PRE.* | PASS | worktree, roadmap mapping, baseline inventory, and reuse plan recorded in `2026-05-19-v086-pre` |
| M086_DELTA_COALESCE.* | PASS | `wmir_committed(u32)` fixture records `recompute_call_reduction_ratio=3.0`, `hot_path_dtoh_calls=0`, and matching sequential/coalesced/replacement rows |
| M086_NOTIFY.* | PASS | relation callback API exposes metadata-only post-commit callbacks; disabled path has zero callback transfer stats |
| M086_EXACT_TYPES.* | PASS | U32 and Symbol typed native exact-induction dispatch parity recorded; provider typed tests passed |
| M086_CHAIN_SMEM profile | PASS | profile trigger evidence records `hot_to_small_median_ratio=31.294571096643047` and `profile_trigger_pass=true` |
| M086_CHAIN_SMEM implementation | PASS | chain-hot fixture records `speedup_ratio=5.58300273358745`, parity true, and `added_dtoh_calls=0` |
| M086_CSE.* | PASS | duplicate-subplan fixture records 50% duplicate work reduction, output parity, generation invalidation, unsafe-boundary rejection, and `added_dtoh_calls=0` |
| M086_ADAPT.* | PASS | adaptive adoption, rollback, replay determinism, and data-plane DTOH budget passed; speedup not claimed |
| M086_INDEX correctness | PASS | key hardening, invalidation, LRU budget eviction, repeated-session reuse, and transfer budget passed |
| M086_INDEX recorded background build | PASS | runtime-backed provider test builds on a recorded stream and consumes through recorded indexed join; executor defers current-evaluation indexed reuse |
| M086_INDEX performance speedup | BLOCKED | no >=1.5x timing speedup is claimed; observed retained-index reuse is recorded |
| M086_CONSUMERS.* | PASS | DTS-DLM, two neutral scientific/engineering fixtures, v0.9.0 substrate, and pyxlog compatibility covered; v0.8.0/v0.8.5 validators passed |
| M086_INT.1 formatting | PASS | `cargo fmt --check` exit 0 |
| M086_INT.2 workspace | PASS | `cargo check --workspace` exit 0 |
| M086_INT.3 targeted Rust | PASS | runtime, cuda, induce, prob, logic, and integration crates exited 0 |
| M086_INT.4 Python | PASS | `39 passed in 52.33s` for v0.8.0/v0.8.5/v0.8.6 source/runtime bundle |
| M086_INT.5 examples | PASS | v0.8.0 examples 5, v0.8.5 examples 10, v0.8.6 examples 5 |
| M086_INT.6 transfer guards | PASS | xlog-prob no-D2H guards, integration strict D2H tests, and v0.8.6 source/runtime transfer guards passed |
| M086_INT.7 performance | PASS_WITH_BLOCKED_METRIC | raw speed/transfer evidence recorded; persistent-index timing speedup not claimed |
| M086_INT.8 docs | PASS | JSON, py_compile, package metadata validation, and evidence links passed |
| M086_INT.9 git hygiene | PASS | generated artifacts limited to evidence; `git diff --check` passed |
| M086_CLOSE.1 sub-goal table | PASS | this proposal and `closure_summary.json` |
| M086_CLOSE.2 roadmap sync | PASS | `ROADMAP.md` reflects recorded background build and unclaimed timing speedup |
| M086_CLOSE.3 unresolved issues | PASS | no unresolved scope blocker; persistent-index timing speedup remains explicitly unclaimed |
| M086_CLOSE.4 release decision | PASS | `MERGE_READY` after final validation |
| M086_CLOSE.5 no implicit release | PASS | no board update, merge, push, or tag |
| M086_CLOSE.6 methodology audit | PASS | GDSP/GQM evidence sections present across v0.8.6 evidence |

## Verification Matrix

| Command | Result |
|---|---|
| `cargo fmt --check` | exit 0 |
| `cargo check --workspace` | exit 0 |
| `cargo test -p xlog-runtime` | exit 0; 141 lib tests, 15 integration tests, 2 doc tests passed, 2 doc tests ignored |
| `cargo test -p xlog-cuda kernel_modules` | exit 0; 2 passed |
| `cargo test -p xlog-induce` | exit 0; 23 passed |
| `cargo test -p xlog-prob` | exit 0; includes no-D2H/native GPU guards |
| `cargo test -p xlog-logic` | exit 0 |
| `cargo test -p xlog-integration` | exit 0; includes strict deterministic D2H, cross-mode determinism, WCOJ, and widened-frontier suites |
| `PYTHONPATH=target/debug pytest -q python/tests/test_v080_examples_source.py python/tests/test_v085_examples_source.py python/tests/test_v086_delta_coalescing.py python/tests/test_v086_relation_callbacks.py python/tests/test_v086_relation_callbacks_runtime.py python/tests/test_v086_exact_types_source.py python/tests/test_v086_exact_types_runtime.py python/tests/test_v086_chain_smem_profile_source.py python/tests/test_v086_chain_smem_source.py python/tests/test_v086_cse_source.py python/tests/test_v086_adaptive_reoptimization_source.py python/tests/test_v086_persistent_hash_index_source.py python/tests/test_v086_consumers_source.py` | exit 0; 39 passed in 52.33s |
| `python scripts/validate_v086_examples.py` | exit 0; v0.8.0 examples 5, v0.8.5 examples 10, v0.8.6 examples 5 |
| `python -m json.tool` over v0.8.6 evidence and expected JSON files | exit 0 |
| `python -m py_compile scripts/validate_v086_examples.py python/tests/test_v086_persistent_hash_index_source.py python/tests/test_v086_consumers_source.py` | exit 0 |
| `python scripts/validate_package_metadata.py` | exit 0 |
| `git diff --check` | exit 0 |

## Methodology Audit

Every v0.8.6 sub-goal evidence directory names:

- the consumer goal or release reason;
- the existing xlog subsystem reused;
- the GQM questions or metrics answered;
- raw measurement files or command output;
- metric interpretation, including PASS/BLOCKED disposition.

G086_ADAPT and G086_INDEX evidence were amended during closure to make the
GDSP/GQM sections explicit rather than implicit. G086_INDEX was further amended
to add deferred background-build reuse telemetry and the runtime-backed recorded
provider build test.

## Known Unsupported Or Blocked Scope

- No >=1.5x persistent-index timing speedup is claimed. The completed manager
  validates recorded background building, deferred current-evaluation indexed
  reuse, retained-index reuse, invalidation, and budget behavior.
- v0.9.0 EIR, world-view semantics, solver services, MaxSAT, epistemic
  splitting, and multi-GPU/out-of-core work remain out of scope.
- No release action has been taken.

## v0.9.0 Rebase Note

v0.9.0 should rebase on or merge after any accepted v0.8.6 landing because
typed exact induction, chain shared-memory scoring, runtime CSE, adaptive
candidate adoption, persistent-index telemetry, and consumer examples now form
the runtime substrate that the epistemic/solver branch should reuse.

## Coordinator Actions

1. Review the amendment evidence and final validation output.
2. Separately authorize any release-board update, merge, push, and tag.
