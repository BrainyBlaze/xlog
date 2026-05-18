# v0.9.0 G090_SPLIT Evidence

Date: 2026-05-18

Goal node: `G090_SPLIT - Epistemic Splitting`

Branch: `feat/v090-epistemic-solver-semantics`

Predecessor evidence:

- `docs/evidence/2026-05-18-v090-pre/README.md`
- `docs/evidence/2026-05-18-v090-eir/README.md`
- `docs/evidence/2026-05-18-v090-g91/README.md`
- `docs/evidence/2026-05-18-v090-faeel/README.md`
- `docs/evidence/2026-05-18-v090-gpt/README.md`

## Implementation Summary

| Requirement | Evidence |
|---|---|
| Deterministic graph | `build_epistemic_dependency_graph` builds sorted predicate components with source rule indices. |
| Valid split fixture | `test_epistemic_split.rs` verifies two independent epistemic rules split into two deterministic components. |
| Invalid split rejection | A rule coupling `know p()` and `possible q()` returns `UnsupportedEpistemicConstruct { construct: "epistemic splitting" }`. |
| Recomposition | `EpistemicSplitPlan::recomposed_rule_indices()` sorts component rule indices and equals the unsplit source order in fixtures. |
| GPU split execution | Not implemented in this CPU semantic-oracle fixture. |
| Documentation | `docs/architecture/epistemic-semantics.md` documents the bounded split graph, rejection, and recomposition contract. |

## Validation

| Command | Result |
|---|---|
| `cargo fmt --check` | PASS |
| `cargo test -p xlog-logic --test test_epistemic_split` | PASS, 3 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_gpt` | PASS, 2 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_faeel` | PASS, 3 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_g91` | PASS, 3 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_eir` | PASS, 4 passed, 0 failed |
| `cargo test -p xlog-logic --lib` | PASS, 238 passed, 0 failed |
| `cargo check -p xlog-logic -p xlog-ir -p xlog-solve -p xlog-prob` | PASS |
| `cargo check -p pyxlog` | PASS |

## Metric Status

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M090_SPLIT.1 graph construction | deterministic dependency graph | PASS for oracle | Sorted component fixture. |
| M090_SPLIT.2 valid split fixtures | 100 percent pass | PASS for oracle | `test_epistemic_split`: 3/3 passed. |
| M090_SPLIT.3 invalid split fixtures | typed rejection | PASS for oracle | `UnsupportedEpistemicConstruct` invalid coupling fixture. |
| M090_SPLIT.4 recomposition | recomposed output equals unsplit output on fixtures | PASS for oracle | `recomposed_rule_indices() == [0, 1]` fixture. |
| M090_SPLIT.5 modal coupling guard | fixture with cross-component epistemic dependency is not split | PASS for oracle | Invalid coupling fixture rejects split. |
| M090_SPLIT.6 GPU split execution | valid split components execute through GPU-native subplans, not CPU-only recomposition | BLOCKED | Current split fixture does not execute GPU subplans. |

## Coordination Notes

- This is bounded split planning, not GPU-native component execution.
- No pyxlog public API signatures were changed.
- No push, tag, release-board update, or merge was performed.
