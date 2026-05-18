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
| Documentation | `docs/architecture/epistemic-semantics.md` documents bounded FAEEL semantics and no-model reasons. |

## Validation

| Command | Result |
|---|---|
| `cargo fmt --check` | PASS |
| `cargo test -p xlog-logic --test test_epistemic_faeel` | PASS, 3 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_g91` | PASS, 3 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_eir` | PASS, 3 passed, 0 failed |
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

## Coordination Notes

- This is bounded fixture semantics, not full production epistemic execution.
- Generate-Propagate-Test execution remains `G090_GPT` scope.
- No pyxlog public API signatures were changed.
- No push, tag, release-board update, or merge was performed.
