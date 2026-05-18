# v0.9.0 G090_PROB Semantic-Oracle Evidence

Date: 2026-05-18

Goal node: `G090_PROB - Probabilistic And Circuit Integration`

Branch: `feat/v090-epistemic-solver-semantics`

## Implementation Summary

The current branch contains fixture-level probabilistic integration for accepted
world-view evidence. It proves that probabilistic evidence is gated by an
accepted `EpistemicWorldView`, but it does not yet prove accepted epistemic
probabilistic execution on the GPU-native exact/provenance path.

| Requirement | Evidence |
|---|---|
| Accepted world-view evidence | `AcceptedWorldViewEvidence` is constructed from a non-empty `EpistemicWorldView`. |
| Semantic contract | `xlog_prob::epistemic` represents accepted epistemic assumptions as probabilistic evidence conditions. |
| Incremental circuit fixture | `EpistemicCircuit::apply_accepted_world_view` updates active evidence without changing the circuit fingerprint when the adapter supports incremental evidence. |
| Compiler adapter | `KnowledgeCompilerAdapter::external_ddnnf_text` records an alternative Decision-DNNF text adapter design. |
| Numerical stability | `conditional_probability_from_logs` normalizes conditional probabilities with `EPISTEMIC_PROBABILITY_TOLERANCE = 1e-12`. |

## Validation

| Command | Result |
|---|---|
| `cargo fmt --check` | PASS |
| `cargo test -p xlog-prob --test epistemic_prob` | PASS, 5 passed, 0 failed |
| `cargo test -p xlog-prob --lib` | PASS, 56 passed, 0 failed |
| `cargo check -p xlog-logic -p xlog-ir -p xlog-solve -p xlog-prob` | PASS |
| `cargo check -p pyxlog` | PASS |

## Metric Status

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M090_PROB.1 semantic contract | documented interaction between epistemic and probabilistic layers | PASS for oracle | `AcceptedWorldViewEvidence` and architecture docs. |
| M090_PROB.2 incremental circuit fixture | changed assumption updates circuit without full rebuild where supported | PASS for oracle | `evidence_conditioning_consumes_accepted_world_view`. |
| M090_PROB.3 compiler adapter | at least one alternative compiler adapter design or implementation | PASS for oracle | `external_ddnnf_text_compiler_adapter_is_explicitly_represented`. |
| M090_PROB.4 numerical stability | deterministic fixture within documented tolerance | PASS for oracle | `log_space_conditional_probability_is_tolerance_bounded`. |
| M090_PROB.5 evidence conditioning | probabilistic integration consumes accepted world views, not raw unvalidated guesses | PASS for oracle | `AcceptedWorldViewEvidence` requires an `EpistemicWorldView`. |
| M090_PROB.6 GPU exact integration | accepted world-view evidence updates the GPU-native exact/provenance path | BLOCKED | Fixture API does not yet run accepted epistemic evidence through production exact inference. |
| M090_PROB.7 CPU recompute ban | accepted probabilistic epistemic path records zero CPU-only probability recomputation | BLOCKED | No accepted GPU execution path or zero-fallback counter evidence exists yet. |

## Coordination Notes

- This file is not release-close evidence for `G090_PROB`.
- Production WFS/provenance still rejects direct epistemic literals.
- The external Decision-DNNF adapter is a design contract, not a dispatch path.
- No pyxlog public API signatures were changed.
- No push, tag, release-board update, or merge was performed.
