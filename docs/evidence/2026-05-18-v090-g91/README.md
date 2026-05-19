# v0.9.0 G090_G91 Evidence

Date: 2026-05-18

Goal node: `G090_G91 - G91 Compatibility Mode`

Branch: `feat/v090-epistemic-solver-semantics`

Predecessor evidence:

- `docs/evidence/2026-05-18-v090-pre/README.md`
- `docs/evidence/2026-05-18-v090-eir/README.md`

## Implementation Summary

| Requirement | Evidence |
|---|---|
| Explicit mode selection | `#pragma epistemic_mode = g91` is parsed into `Directives::epistemic_mode = Some(EpistemicMode::G91)`. |
| Compatibility fixture | `crates/xlog-logic/src/epistemic.rs` adds a bounded fixture evaluator where G91 accepts compatibility-only `possible` atoms. |
| G91/FAEEL distinction | `crates/xlog-logic/tests/test_epistemic_g91.rs` proves `possible fact()` is true under G91 when listed as possible, but false under the default FAEEL fixture semantics unless known. |
| Mode isolation | `test_epistemic_g91.rs` proves a non-epistemic program compiles to the same RIR plan with and without `#pragma epistemic_mode = g91`. |
| Accepted GPU runtime fixture | `test_epistemic_gpu_wcoj_execution::g91_self_supported_possible_reaches_gpu_runtime_path` proves explicit G91 self-supported `possible` lowers through the existing reduced runtime/fact-buffer path, records stable tuple-source membership, accepts one world view, and materializes `p()` with zero CPU candidate/world-view fallback counters. |
| Documentation | `docs/architecture/epistemic-semantics.md` documents the bounded G91 fixture semantics and states that full FAEEL/GPT execution remains later scope. |

## Validation

| Command | Result |
|---|---|
| `cargo fmt --check` | PASS |
| `cargo test -p xlog-logic --test test_epistemic_g91` | PASS, 3 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_eir` | PASS, 4 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution g91_self_supported_possible_reaches_gpu_runtime_path -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-logic --lib` | PASS, 238 passed, 0 failed |
| `cargo check -p xlog-logic -p xlog-ir -p xlog-solve -p xlog-prob` | PASS |
| `cargo check -p pyxlog` | PASS |

## Metric Status

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M090_G91.1 mode selection | explicit config, flag, or source annotation | PASS | `#pragma epistemic_mode = g91` parser and directive test. |
| M090_G91.2 golden fixtures | 100 percent pass | PASS | `cargo test -p xlog-logic --test test_epistemic_g91`: 3/3 passed. |
| M090_G91.3 mode isolation | default mode output unchanged on non-epistemic fixtures | PASS | Non-epistemic compile-output equality fixture. |
| M090_G91.4 docs | compatibility behavior documented | PASS | `docs/architecture/epistemic-semantics.md`. |
| M090_GPU.7 G91 runtime parity slice | explicit G91 self-supported `possible` reaches accepted GPU runtime path | PASS for fixture | `test_epistemic_gpu_wcoj_execution::g91_self_supported_possible_reaches_gpu_runtime_path`. Broader G91/FAEEL/GPT/splitting parity remains open. |

## Coordination Notes

- G91 is present as a bounded compatibility fixture layer plus one accepted
  self-supported `possible` GPU runtime fixture; this is not full epistemic
  execution parity.
- No pyxlog public API signatures were changed.
- No push, tag, release-board update, or merge was performed.
