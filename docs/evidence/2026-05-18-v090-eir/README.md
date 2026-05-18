# v0.9.0 G090_EIR Evidence

Date: 2026-05-18

Goal node: `G090_EIR - Epistemic Intermediate Representation`

Branch: `feat/v090-epistemic-solver-semantics`

Predecessor evidence: `docs/evidence/2026-05-18-v090-pre/README.md`

## Implementation Summary

| Requirement | Evidence |
|---|---|
| Explicit AST representation | `crates/xlog-logic/src/ast.rs` adds `EpistemicMode`, `EpistemicOp`, `EpistemicLiteral`, `Directives::epistemic_mode`, and `BodyLiteral::Epistemic`. |
| Parser syntax | `crates/xlog-logic/src/grammar.pest` and `crates/xlog-logic/src/parser.rs` add `#pragma epistemic_mode = faeel|g91`, `know atom(...)`, `possible atom(...)`, and negated forms. |
| EIR representation | `crates/xlog-ir/src/eir.rs` adds `EirProgram`, `EirRule`, `EirBodyLiteral`, `EirEpistemicLiteral`, `EirEpistemicMode`, and `EirEpistemicOp`; `xlog_logic::build_eir` builds EIR without RIR lowering. |
| Typed diagnostics | `xlog_core::XlogError::UnsupportedEpistemicConstruct` is returned for nested epistemic literals and direct RIR/probabilistic lowering attempts. |
| Lowering boundary | `docs/architecture/epistemic-semantics.md` documents that EIR is the required boundary before G91/FAEEL execution and that RIR lowering rejects epistemic literals for now. |
| Compatibility handling | Existing non-epistemic paths remain exhaustive. Downstream `xlog-prob`, `xlog-gpu`, and `pyxlog` compile after adding explicit handling/rejection for the new AST variant. |

## Validation

| Command | Result |
|---|---|
| `cargo fmt --check` | PASS |
| `cargo test -p xlog-logic --test test_epistemic_eir` | PASS, 3 passed, 0 failed |
| `cargo test -p xlog-logic test_parse_epistemic --lib` | PASS, 2 passed, 0 failed |
| `cargo check -p xlog-logic -p xlog-ir -p xlog-solve -p xlog-prob` | PASS |
| `cargo test -p xlog-logic --lib` | PASS, 238 passed, 0 failed |
| `cargo test -p xlog-ir --lib` | PASS, 14 passed, 0 failed |
| `cargo test -p xlog-solve --lib` | PASS, 111 passed, 0 failed |
| `cargo test -p xlog-prob --lib` | PASS, 56 passed, 0 failed |
| `cargo check -p pyxlog` | PASS |

## Metric Status

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M090_EIR.1 AST/EIR nodes | explicit representation committed | PASS | AST and EIR files listed above. |
| M090_EIR.2 parser tests | positive and negative syntax fixtures pass | PASS | `test_parse_epistemic_*` and `test_epistemic_eir` positive/negative diagnostics. |
| M090_EIR.3 lowering boundary | EIR-to-existing-IR boundary documented | PASS | `docs/architecture/epistemic-semantics.md`. |
| M090_EIR.4 diagnostics | unsupported constructs return typed errors | PASS | `UnsupportedEpistemicConstruct` tests for nested literal and RIR boundary. |
| M090_EIR.5 no DTS regression | v0.8 pyxlog compatibility tests still pass or are not touched | PASS | `cargo check -p pyxlog` PASS; no `crates/pyxlog/` files edited. |

## Coordination Notes

- No pyxlog public API signatures were changed.
- No push, tag, release-board update, or merge was performed.
- G91 and FAEEL execution are not implemented in this commit; direct RIR/probabilistic lowering returns typed `UnsupportedEpistemicConstruct` diagnostics until those sub-goals add semantics.
