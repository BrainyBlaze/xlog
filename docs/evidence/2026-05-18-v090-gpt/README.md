# v0.9.0 G090_GPT Evidence

Date: 2026-05-18

Goal node: `G090_GPT - Generate-Propagate-Test Execution`

Branch: `feat/v090-epistemic-solver-semantics`

Predecessor evidence:

- `docs/evidence/2026-05-18-v090-pre/README.md`
- `docs/evidence/2026-05-18-v090-eir/README.md`
- `docs/evidence/2026-05-18-v090-g91/README.md`
- `docs/evidence/2026-05-18-v090-faeel/README.md`

## Implementation Summary

| Requirement | Evidence |
|---|---|
| Phase separation | `run_generate_propagate_test` has explicit generate, propagate, and test phases. |
| Trace output | `GeneratePropagateTestTrace` reports generated, guesses, propagated, pruned, reduced-program models, tested, accepted, accepted world views, rejected, and rejection reasons. |
| Correctness fixtures | `test_epistemic_gpt.rs` covers one accepted candidate, one FAEEL-rejected candidate, and one propagation-pruned contradiction. |
| Bounded behavior | `GeneratePropagateTestConfig::max_candidates` rejects oversized candidate sets with `XlogError::ResourceExhausted`. |
| GPU residency | Not implemented in this CPU semantic-oracle fixture. |
| Documentation | `docs/architecture/epistemic-semantics.md` documents the GPT phase contract and guard. |

## Validation

| Command | Result |
|---|---|
| `cargo fmt --check` | PASS |
| `cargo test -p xlog-logic --test test_epistemic_gpt` | PASS, 2 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_faeel` | PASS, 3 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_g91` | PASS, 3 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_eir` | PASS, 3 passed, 0 failed |
| `cargo test -p xlog-logic --lib` | PASS, 238 passed, 0 failed |
| `cargo check -p xlog-logic -p xlog-ir -p xlog-solve -p xlog-prob` | PASS |
| `cargo check -p pyxlog` | PASS |

## Metric Status

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M090_GPT.1 phase separation | generate, propagate, test boundaries visible in code | PASS for oracle | `run_generate_propagate_test` implementation and test fixture. |
| M090_GPT.2 trace output | debug/trace mode reports phase counts and GPU launch counters | PARTIAL | CPU trace counts are asserted; GPU launch counters are missing. |
| M090_GPT.3 correctness fixtures | accepted/rejected candidate fixtures pass | PASS for oracle | `test_epistemic_gpt`: 2/2 passed. |
| M090_GPT.4 bounded behavior | candidate explosion guard implemented or explicitly scoped | PASS for oracle | `ResourceExhausted` guard fixture. |
| M090_GPT.5 world-view validation | trace records guess count, reduced-program model count, accepted world-view count, and rejection reasons | PASS for oracle | CPU trace fields are asserted in `test_epistemic_gpt`. |
| M090_GPT.6 GPU residency | candidate generation, propagation, and world-view validation hot path uses GPU-resident buffers | BLOCKED | Current GPT fixture uses explicit CPU candidate inputs. |

## Coordination Notes

- Candidate generation is explicit-input bounded fixture generation; arbitrary EIR world enumeration on GPU remains required production scope.
- No pyxlog public API signatures were changed.
- No push, tag, release-board update, or merge was performed.
