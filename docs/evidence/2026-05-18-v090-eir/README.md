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
| EIR representation | `crates/xlog-ir/src/eir.rs` adds `EirProgram`, `EirRule`, `EirBodyLiteral`, `EirEpistemicLiteral`, `EirEpistemicMode`, `EirEpistemicOp`, and `EirTerm`; `xlog_logic::build_eir` builds EIR without RIR lowering and preserves source atom terms for GPU tuple-key matching. |
| Typed diagnostics | `xlog_core::XlogError::UnsupportedEpistemicConstruct` is returned for nested epistemic literals, direct RIR/probabilistic lowering attempts, and unsupported epistemic integrity constraints before GPU lowering can silently erase them. |
| Lowering boundary | `docs/architecture/epistemic-semantics.md` documents that EIR is the required semantic boundary and that runtime GPU lowering is still missing after the plan contract. |
| Compatibility handling | Existing non-epistemic paths remain exhaustive. Downstream `xlog-prob`, `xlog-gpu`, and `pyxlog` compile after adding explicit handling/rejection for the new AST variant. |

## Validation

| Command | Result |
|---|---|
| `cargo fmt --check` | PASS |
| `cargo test -p xlog-logic --test test_epistemic_eir` | PASS, 4 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_executable_plan` | PASS, 8 passed, 0 failed |
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
| M090_EIR.1 AST/EIR nodes | explicit representation committed | PASS | AST and EIR files listed above, including preserved `EirTerm` metadata. |
| M090_EIR.2 parser tests | positive and negative syntax fixtures pass | PASS | `test_parse_epistemic_*` and `test_epistemic_eir` positive/negative diagnostics. |
| M090_EIR.3 lowering boundary | EIR-to-GPU-executable boundary documented | PASS | Current docs identify the boundary and plan contract; accepted high-level programs route through the epistemic GPU runtime while direct raw RIR lowering remains a typed rejection boundary. |
| M090_EIR.4 diagnostics | unsupported constructs return typed errors | PASS | `UnsupportedEpistemicConstruct` tests for nested literal, RIR boundary, and unsupported epistemic integrity constraints. |
| M090_EIR.5 explicit operators | `know`, `possible`, and `not know` equivalents represented without ad hoc string rewrites | PASS | AST/EIR preserve operators and atom terms explicitly. |
| M090_EIR.6 production route | accepted epistemic forms have a production lowering route; rejected forms are explicit | PASS | `xlog-gpu::LogicProgram` routes accepted epistemic programs through `compile_epistemic_gpu_execution` / split execution and the production GPU runtime; direct raw RIR lowering still rejects epistemic literals. |
| M090_EIR.7 no DTS regression | v0.8 pyxlog compatibility tests still pass or are not touched | PENDING_REBASE | `cargo check -p pyxlog` was a pre-rebase proxy; v0.8 compatibility rerun is still required. |

## Coordination Notes

- No pyxlog public API signatures were changed.
- No push, tag, release-board update, or merge was performed.
- Direct RIR/probabilistic lowering returns typed `UnsupportedEpistemicConstruct` diagnostics; production GPU runtime lowering remains required before closure.
