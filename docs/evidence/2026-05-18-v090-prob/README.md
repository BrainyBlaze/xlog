# v0.9.0 G090_PROB Evidence

Date: 2026-05-18

Goal node: `G090_PROB - Probabilistic And Circuit Integration`

Branch: `feat/v090-epistemic-solver-semantics`

Predecessor evidence:

- `docs/evidence/2026-05-18-v090-pre/README.md`
- `docs/evidence/2026-05-18-v090-eir/README.md`
- `docs/evidence/2026-05-18-v090-g91/README.md`
- `docs/evidence/2026-05-18-v090-faeel/README.md`
- `docs/evidence/2026-05-18-v090-gpt/README.md`
- `docs/evidence/2026-05-18-v090-split/README.md`
- `docs/evidence/2026-05-18-v090-solver/README.md`

## Implementation Summary

| Requirement | Evidence |
|---|---|
| Semantic contract | `xlog_prob::epistemic` represents epistemic assumptions as probabilistic evidence conditions. |
| Incremental circuit fixture | `EpistemicCircuit::apply_assumption` updates active evidence without changing the circuit fingerprint when the adapter supports incremental evidence. |
| Compiler adapter | `KnowledgeCompilerAdapter::external_ddnnf_text` records an alternative Decision-DNNF text adapter design. |
| Numerical stability | `conditional_probability_from_logs` normalizes conditional probabilities with `EPISTEMIC_PROBABILITY_TOLERANCE = 1e-12`. |
| Documentation | `docs/architecture/xlog-prob.md` and `docs/architecture/epistemic-semantics.md` document the bounded probabilistic contract. |

## Validation

| Command | Result |
|---|---|
| `cargo fmt --check` | PASS |
| `cargo test -p xlog-prob --test epistemic_prob` | PASS, 4 passed, 0 failed |
| `cargo test -p xlog-prob --lib` | PASS, 56 passed, 0 failed |
| `cargo check -p xlog-logic -p xlog-ir -p xlog-solve -p xlog-prob` | PASS |
| `cargo check -p pyxlog` | PASS |

## Metric Status

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M090_PROB.1 semantic contract | documented interaction between epistemic and probabilistic layers | PASS | `EpistemicProbabilisticRole::EvidenceConditioning` and architecture docs. |
| M090_PROB.2 incremental circuit fixture | changed assumption updates circuit without full rebuild where supported | PASS | `incremental_assumption_update_reuses_circuit_when_adapter_supports_it`. |
| M090_PROB.3 compiler adapter | at least one alternative compiler adapter design or implementation | PASS | `external_ddnnf_text_compiler_adapter_is_explicitly_represented`. |
| M090_PROB.4 numerical stability | deterministic fixture within documented tolerance | PASS | `log_space_conditional_probability_is_tolerance_bounded`. |

## Coordination Notes

- This is bounded fixture integration; production WFS/provenance still rejects direct epistemic literals.
- The external Decision-DNNF adapter is a design contract, not a dispatch path.
- No pyxlog public API signatures were changed.
- No push, tag, release-board update, or merge was performed.
