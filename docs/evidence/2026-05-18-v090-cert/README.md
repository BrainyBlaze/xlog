# v0.9.0 G090_CERT Evidence

Date: 2026-05-18

Goal node: `G090_CERT - Certification And Regression Gates`

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

## Certification Scope

| Gate | Evidence |
|---|---|
| Semantic golden fixtures | EIR, G91, FAEEL, GPT, and split fixtures passed 14/14. |
| Solver service fixtures | SAT assumptions, learned transfer, MaxSAT, GPU deferral, and failure modes passed 5/5. |
| Probabilistic coherence fixtures | Epistemic evidence, incremental circuit update, adapter design, and tolerance fixtures passed 4/4. |
| Parser diagnostics | Positive syntax and negative nested-epistemic typed diagnostics passed in `test_epistemic_eir`. |
| Workspace health subset | Logic, solver, and probabilistic lib suites passed; cross-crate cargo checks passed. |

## Validation

| Command | Result |
|---|---|
| `cargo fmt --check` | PASS |
| `cargo test -p xlog-logic --test test_epistemic_eir --test test_epistemic_g91 --test test_epistemic_faeel --test test_epistemic_gpt --test test_epistemic_split` | PASS, 14 passed, 0 failed |
| `cargo test -p xlog-solve --test solver_service_semantics` | PASS, 5 passed, 0 failed |
| `cargo test -p xlog-prob --test epistemic_prob` | PASS, 4 passed, 0 failed |
| `cargo test -p xlog-logic --lib` | PASS, 238 passed, 0 failed |
| `cargo test -p xlog-solve --lib` | PASS, 111 passed, 0 failed |
| `cargo test -p xlog-prob --lib` | PASS, 56 passed, 0 failed |
| `cargo check -p xlog-logic -p xlog-ir -p xlog-solve -p xlog-prob` | PASS |
| `cargo check -p pyxlog` | PASS |

## Metric Status

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M090_CERT.1 semantic golden tests | 100 percent pass | PASS | 14/14 semantic fixtures passed. |
| M090_CERT.2 solver tests | 100 percent pass for implemented solver scope | PASS | `solver_service_semantics`: 5/5 passed. |
| M090_CERT.3 parser diagnostics | positive and negative syntax fixtures pass | PASS | `test_epistemic_eir`: 3/3 passed. |
| M090_CERT.4 v0.8 compatibility | v0.8 pyxlog/DTS cert subset rerun after rebase | BLOCKED | v0.8 integration/rebase has not happened; current `cargo check -p pyxlog` passed as a pre-rebase proxy only. |
| M090_CERT.5 formatting | `cargo fmt --check` pass | PASS | Formatting gate passed. |
| M090_CERT.6 workspace health | agreed cargo test subset pass | PASS | Logic/solve/prob lib suites and cross-crate checks passed. |

## Coordination Notes

- This cert snapshot is not a closure claim because M090_CERT.4 waits for the v0.8 integration/rebase gate.
- No v0.8-owned pyxlog public API signatures were changed in this branch.
- No push, tag, release-board update, or merge was performed.
