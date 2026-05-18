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
| Epistemic guide | `docs/epistemic-solver-semantics-guide.md` explains EIR, G91, FAEEL, GPT, and splitting. |
| Solver guide | The same guide explains assumptions, incremental SAT, learned transfer, MaxSAT, GPU deferral, and failure states. |
| Runnable examples | `examples/epistemic/` has five `.xlog` fixtures run by `test_epistemic_examples`. |
| Roadmap sync | No ROADMAP or release-board rows were edited in this slice. |

## Validation

| Command | Result |
|---|---|
| `cargo fmt --check` | PASS |
| `cargo test -p xlog-logic --test test_epistemic_examples` | PASS, 5 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_eir --test test_epistemic_g91 --test test_epistemic_faeel --test test_epistemic_gpt --test test_epistemic_split` | PASS, 14 passed, 0 failed |
| `cargo test -p xlog-solve --test solver_service_semantics` | PASS, 5 passed, 0 failed |
| `cargo test -p xlog-prob --test epistemic_prob` | PASS, 4 passed, 0 failed |
| `cargo check -p xlog-logic -p xlog-ir -p xlog-solve -p xlog-prob` | PASS |
| `cargo check -p pyxlog` | PASS |

## Metric Status

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M090_DOC.1 epistemic guide | guide explains EIR, G91, FAEEL, GPT, splitting | PASS | `docs/epistemic-solver-semantics-guide.md`. |
| M090_DOC.2 solver guide | guide explains assumptions, incremental SAT, MaxSAT, and failure states | PASS | Solver Services section in guide. |
| M090_DOC.3 examples | at least one runnable example per implemented major semantic mode | PASS | `examples/epistemic/` and `test_epistemic_examples`: 5/5 passed. |
| M090_DOC.4 roadmap sync | ROADMAP v0.9.0 rows updated only at closure, not prematurely marked done | PASS | No `ROADMAP.md` or board edits in this slice. |

## Coordination Notes

- Epistemic examples are run by the fixture harness, not production `xlog run`.
- The guide repeats that the v0.8 compatibility rerun remains pending until rebase.
- No push, tag, release-board update, or merge was performed.
