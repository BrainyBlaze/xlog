# v0.9.0 G090_FAEEL Evidence

Date: 2026-05-18

Goal node: `G090_FAEEL - FAEEL Default Semantics`

Branch: `feat/v090-epistemic-solver-semantics`

Predecessor evidence:

- `docs/evidence/2026-05-18-v090-pre/README.md`
- `docs/evidence/2026-05-18-v090-eir/README.md`
- `docs/evidence/2026-05-18-v090-g91/README.md`

## Implementation Summary

| Requirement | Evidence |
|---|---|
| Default mode | `Directives::epistemic_mode_or_default()` returns `EpistemicMode::Faeel`. |
| Minimal executable core | `crates/xlog-logic/src/epistemic.rs` adds `evaluate_faeel_candidate` and typed `FaeelCandidateResult`. |
| Foundedness fixture | `test_epistemic_faeel.rs` accepts a candidate where `know fact()` is backed by `known fact/0`. |
| No-model behavior | `test_epistemic_faeel.rs` returns typed `NoModel` values for unfounded possible-only support and contradictions. |
| G91 distinction | `test_epistemic_g91.rs` preserves the intentional G91/FAEEL difference for `possible fact()`. |
| Production executable guard | `compile_epistemic_gpu_execution` calls the FAEEL foundedness guard before reduced runtime compilation, rejecting unsupported `p() :- possible p().` under the default mode, permitting zero-arity self-`possible` only when the same predicate has independent ordinary founded support, rejecting nonzero-arity self-`possible` without tuple-level foundedness proof, and preserving the explicit G91 compatibility fixture through accepted GPU runtime execution. |
| Accepted runtime founded support | `test_epistemic_gpu_wcoj_execution::faeel_independently_founded_self_possible_reaches_gpu_runtime_path` proves an independently founded default-FAEEL self-`possible` fixture reaches accepted GPU runtime execution and materializes `p()`. |
| Documentation | `docs/architecture/epistemic-semantics.md` documents bounded FAEEL semantics and no-model reasons. |

## Validation

| Command | Result |
|---|---|
| `cargo fmt --check` | PASS |
| `cargo test -p xlog-logic --test test_epistemic_faeel` | PASS, 3 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_g91` | PASS, 3 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_executable_plan` | PASS, 8 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution faeel_independently_founded_self_possible_reaches_gpu_runtime_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_eir` | PASS, 4 passed, 0 failed |
| `cargo test -p xlog-logic --lib` | PASS, 238 passed, 0 failed |
| `cargo check -p xlog-logic -p xlog-ir -p xlog-solve -p xlog-prob` | PASS |
| `cargo check -p pyxlog` | PASS |

## Metric Status

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M090_FAEEL.1 core semantics | minimal core implemented and documented | PASS | `evaluate_faeel_candidate` and architecture doc. |
| M090_FAEEL.2 golden fixtures | 100 percent pass | PASS | `cargo test -p xlog-logic --test test_epistemic_faeel`: 3/3 passed. |
| M090_FAEEL.3 no-model behavior | typed result or diagnostic, not panic | PASS | `FaeelCandidateResult::NoModel(FaeelNoModelReason::...)` fixtures. |
| M090_FAEEL.4 G91 distinction | fixtures show at least one intentional G91/FAEEL difference | PASS | `test_epistemic_g91::g91_possible_fixture_differs_from_faeel_default`. |
| M090_FAEEL.5 foundedness guard | self-supported epistemic fixture rejected with documented reason | PASS | `test_epistemic_executable_plan::faeel_gpu_execution_rejects_self_supported_possible_before_runtime_dispatch` rejects default `p() :- possible p().` as `FAEEL foundedness guard`; `faeel_gpu_execution_allows_self_possible_with_independent_founded_support` permits the zero-arity default-FAEEL case with ordinary support; `faeel_gpu_execution_rejects_nonzero_self_possible_without_tuple_level_foundedness_proof` rejects `p(X) :- node(X), possible p(X)` even when `p/1` has partial ordinary support, because tuple-level foundedness is not proven; `g91_gpu_execution_allows_self_supported_possible_compatibility_fixture` preserves explicit G91 compatibility and `test_epistemic_gpu_wcoj_execution::g91_self_supported_possible_reaches_gpu_runtime_path` proves the compatibility fixture reaches accepted GPU runtime execution. |

## Coordination Notes

- This is bounded fixture semantics plus a production executable-plan guard; it
  is not full production epistemic execution.
- Generate-Propagate-Test execution remains `G090_GPT` scope.
- No pyxlog public API signatures were changed.
- No push, tag, release-board update, or merge was performed.
