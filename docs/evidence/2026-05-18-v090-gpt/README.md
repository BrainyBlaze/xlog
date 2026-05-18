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
| GPU candidate generation | `epistemic_generate_candidate_assumptions_u8` populates bounded candidate-assumption bitsets in the runtime workspace. |
| GPU propagation staging | `epistemic_propagate_candidates_u8` stages generated candidates into world-view/rejection buffers in the runtime workspace. |
| GPU candidate validation | `epistemic_validate_candidate_bits_u8` validates staged candidate bitsets and world-view activity in the runtime workspace. |
| GPU model-membership staging | `epistemic_populate_model_membership_u8` writes row-count-gated candidate-scoped model-membership bytes in the runtime workspace. |
| GPU world-view validation staging | `epistemic_validate_world_views_u8` checks staged model-membership bytes against active world-view slots and updates rejection codes. |
| GPU materialization staging | `epistemic_materialize_accepted_candidates_u8` writes accepted-candidate flags from rejection codes into world-view slots. |
| GPU final-result flag staging | `epistemic_materialize_final_result_flags_u8` writes final-result flags from reduced output device row-count metadata and rejection codes into world-view slots. |
| GPU final tuple materialization | `epistemic_materialize_final_tuple_column_u8` writes a device-resident final-output tuple buffer and final row-count metadata from reduced output columns. |
| GPU residency | Candidate-assumption generation, propagation staging, candidate-buffer validation, row-count-gated model-membership staging, bounded world-view validation staging, accepted-candidate materialization staging, final-result flag staging, and final tuple materialization are implemented as bounded CUDA kernels; actual reduced stable-model tuple membership population remains a runtime gap. |
| Documentation | `docs/architecture/epistemic-semantics.md` documents the GPT phase contract and guard. |

## Validation

| Command | Result |
|---|---|
| `cargo fmt --check` | PASS |
| `cargo test -p xlog-runtime --test test_epistemic_gpu_workspace candidate_generation` | PASS, 2 passed, 0 failed |
| `cargo test -p xlog-runtime --test test_epistemic_gpu_workspace propagation` | PASS, 3 passed, 0 failed |
| `cargo test -p xlog-runtime --test test_epistemic_gpu_workspace candidate_validation` | PASS, 2 passed, 0 failed |
| `cargo test -p xlog-runtime --test test_epistemic_gpu_workspace validation` | PASS, 7 passed, 0 failed |
| `cargo test -p xlog-runtime --test test_epistemic_gpu_workspace materialization` | PASS, 5 passed, 0 failed |
| `cargo test -p xlog-runtime --test test_epistemic_gpu_workspace` | PASS, 38 passed, 0 failed |
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
| M090_GPT.2 trace output | debug/trace mode reports phase counts and GPU launch counters | PARTIAL | CPU trace counts are asserted; candidate-generation, propagation, candidate-validation, row-count-gated model-membership, world-view-validation, accepted-candidate materialization, final-result flag, and final tuple traces each record GPU launches with CUDA-event elapsed timing, but full reduced-runtime stable-model tuple membership counters are missing. |
| M090_GPT.3 correctness fixtures | accepted/rejected candidate fixtures pass | PASS for oracle | `test_epistemic_gpt`: 2/2 passed. |
| M090_GPT.4 bounded behavior | candidate explosion guard implemented or explicitly scoped | PASS for oracle | `ResourceExhausted` guard fixture. |
| M090_GPT.5 world-view validation | trace records guess count, reduced-program model count, accepted world-view count, and rejection reasons | PASS for oracle | CPU trace fields are asserted in `test_epistemic_gpt`. |
| M090_GPT.6 GPU residency | candidate generation, propagation, and world-view validation hot path uses GPU-resident buffers | PARTIAL | Candidate-assumption generation, propagation staging, candidate-buffer validation, row-count-gated model-membership staging, bounded world-view validation staging, accepted-candidate materialization staging, final-result flag staging, and final tuple materialization use GPU-resident workspace/output buffers through CUDA kernels; actual reduced stable-model tuple membership population is still not wired. |

## Coordination Notes

- Candidate generation, propagation staging, candidate-buffer validation, row-count-gated model-membership staging, bounded world-view validation staging, accepted-candidate materialization staging, final-result flag staging, and final tuple materialization are now available as bounded GPU workspace/output kernels. Row-count-only membership now fails closed before accepted runtime return; arbitrary EIR world enumeration plus actual reduced-runtime stable-model membership/test phases remain required production scope.
- No pyxlog public API signatures were changed.
- No push, tag, release-board update, or merge was performed.
