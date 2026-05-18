# v0.9.0 G090_PROB Semantic And Production-Reuse Evidence

Date: 2026-05-18

Goal node: `G090_PROB - Probabilistic And Circuit Integration`

Branch: `feat/v090-epistemic-solver-semantics`

## Implementation Summary

The current branch contains fixture-level probabilistic integration for accepted
world-view evidence plus a thin production adapter. The fixture layer proves
that probabilistic evidence is gated by an accepted `EpistemicWorldView`.
`EpistemicProbProductionAdapter` then proves that accepted evidence can gate
calls into the existing GPU-native `ExactDdnnfProgram` exact/provenance path
without using the bounded fixture circuit.

This remains partial evidence. The accepted epistemic runtime is not yet wired
to feed validated world views into production probabilistic execution end to end.

| Requirement | Evidence |
|---|---|
| Accepted world-view evidence | `AcceptedWorldViewEvidence` is constructed from a non-empty `EpistemicWorldView`. |
| Semantic contract | `xlog_prob::epistemic` represents accepted epistemic assumptions as probabilistic evidence conditions. |
| Production exact adapter | `EpistemicProbProductionAdapter` gates on `AcceptedWorldViewEvidence` and compiles through `ExactDdnnfProgram::compile_source_with_gpu` or `ExactDdnnfProgram::compile_from_program`. |
| CPU probability isolation | `EpistemicProbProductionTrace` records zero CPU-only probability recomputation and zero fixture-circuit evaluations; the source guard rejects `EpistemicCircuit::compile` in the production adapter. |
| Incremental circuit fixture | `EpistemicCircuit::apply_accepted_world_view` updates active evidence without changing the circuit fingerprint when the adapter supports incremental evidence. |
| Compiler adapter | `KnowledgeCompilerAdapter::external_ddnnf_text` records an alternative Decision-DNNF text adapter design. |
| Numerical stability | `conditional_probability_from_logs` normalizes conditional probabilities with `EPISTEMIC_PROBABILITY_TOLERANCE = 1e-12`. |

## Validation

| Command | Result |
|---|---|
| `cargo fmt --check` | PASS |
| `cargo test -p xlog-prob --test epistemic_prob_production_reuse` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-prob --test epistemic_prob` | PASS, 5 passed, 0 failed |
| `cargo test -p xlog-prob --test no_cpu_d4_in_exact` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-prob --test no_dtoh_in_gpu_exact_path` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-prob --lib` | PASS, 56 passed, 0 failed |
| `cargo check -p xlog-prob --features host-io` | PASS |
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
| M090_PROB.6 GPU exact integration | accepted world-view evidence updates the GPU-native exact/provenance path | PARTIAL | `EpistemicProbProductionAdapter` requires accepted evidence before compiling through `ExactDdnnfProgram`; end-to-end accepted runtime wiring is missing. |
| M090_PROB.7 CPU recompute ban | accepted probabilistic epistemic path records zero CPU-only probability recomputation | PARTIAL | Production trace exposes zero CPU-only recomputation and zero fixture-circuit counters; accepted runtime trace is missing. |
| M090_PROB.8 production prob reuse | accepted probabilistic fixtures execute through existing GPU exact/provenance/PIR/knowledge-compilation APIs | PARTIAL | Source guard proves the production adapter calls `ExactDdnnfProgram` GPU compile/evaluate APIs and does not use `EpistemicCircuit`. |
| M090_PROB.9 fixture isolation | bounded epistemic probability fixtures are marked oracle-only and cannot satisfy closure metrics | PARTIAL | Evidence docs separate `EpistemicCircuit` fixtures from `EpistemicProbProductionAdapter`; an automated closure gate is still missing. |

## Coordination Notes

- This file is not release-close evidence for `G090_PROB`.
- Production WFS/provenance still rejects direct epistemic literals.
- The production adapter is partial exact-path reuse evidence only.
- The external Decision-DNNF adapter is a design contract, not a dispatch path.
- No pyxlog public API signatures were changed.
- No push, tag, release-board update, or merge was performed.
