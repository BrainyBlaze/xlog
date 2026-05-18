# v0.9.0 G090_DOC Evidence

Date: 2026-05-18

Goal node: `G090_DOC - Documentation`

Branch: `feat/v090-epistemic-solver-semantics`

Predecessor evidence:

- `docs/evidence/2026-05-18-v090-pre/README.md`
- `docs/evidence/2026-05-18-v090-eir/README.md`
- `docs/evidence/2026-05-18-v090-g91/README.md`
- `docs/evidence/2026-05-18-v090-faeel/README.md`
- `docs/evidence/2026-05-18-v090-gpt/README.md`
- `docs/evidence/2026-05-18-v090-split/README.md`
- `docs/evidence/2026-05-18-v090-solver/README.md`
- `docs/evidence/2026-05-18-v090-prob/README.md`
- `docs/evidence/2026-05-18-v090-cert/README.md`

## Documentation Scope

| Requirement | Evidence |
|---|---|
| Epistemic guide | `docs/epistemic-solver-semantics-guide.md` explains EIR, G91, FAEEL, GPT, splitting, world-view fixtures, and the corrected GPU-native blocker. |
| Solver guide | The same guide explains assumptions, incremental SAT, learned transfer, MaxSAT, failure states, and why the current CPU facade does not close `G090_SOLVER`. |
| Runnable examples | `examples/epistemic/` has five `.xlog` fixtures run by `test_epistemic_examples`. |
| Roadmap sync | No ROADMAP or release-board rows were edited in this slice. |

## Validation

| Command | Result |
|---|---|
| `cargo fmt --check` | PASS |
| `cargo test -p xlog-logic --test test_epistemic_examples` | PASS, 5 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_eir --test test_epistemic_g91 --test test_epistemic_faeel --test test_epistemic_gpt --test test_epistemic_split --test test_epistemic_world_view --test test_epistemic_examples` | PASS, 22 passed, 0 failed |
| `cargo test -p xlog-solve --test solver_service_semantics` | PASS, 5 passed, 0 failed |
| `cargo test -p xlog-prob --test epistemic_prob` | PASS, 5 passed, 0 failed |
| `cargo check -p xlog-logic -p xlog-ir -p xlog-solve -p xlog-prob` | PASS |
| `cargo check -p pyxlog` | PASS |

## Metric Status

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M090_DOC.1 epistemic guide | guide explains EIR, G91, FAEEL, GPT, splitting | PASS for oracle | `docs/epistemic-solver-semantics-guide.md`. |
| M090_DOC.2 solver guide | guide explains GPU-native assumptions, incremental SAT, MaxSAT, portfolio dispatch, and failure states | PARTIAL | Guide documents current CPU facade and the missing GPU-native path. |
| M090_DOC.3 examples | at least one runnable example per implemented major semantic mode | PASS for oracle | `examples/epistemic/` and `test_epistemic_examples`: 5/5 passed. |
| M090_DOC.4 roadmap sync | ROADMAP v0.9.0 rows updated only at closure, not prematurely marked done | PASS | No `ROADMAP.md` or board edits in this slice. |
| M090_DOC.5 GPU/WCOJ execution | guide documents the production GPU-native and WCOJ-backed epistemic execution path | BLOCKED | Guide documents requirements and missing path; production path is not implemented. |

## Coordination Notes

- Epistemic examples are run by the fixture harness, not production `xlog run`.
- This documentation evidence is not a release-close claim.
- The guide repeats that the v0.8 compatibility rerun remains pending until rebase.
- No push, tag, release-board update, or merge was performed.
