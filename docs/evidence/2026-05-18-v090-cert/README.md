# v0.9.0 G090_CERT Evidence

Date: 2026-05-18

Goal node: `G090_CERT - Certification And Regression Gates`

Branch: `feat/v090-epistemic-solver-semantics`

## Certification Scope

This file now records semantic-oracle validation only. The corrected v0.9.0
goal requires GPU-native accepted epistemic execution before `G090_CERT` can
close. The current CPU fixture layer is useful regression evidence, but it is
not certification evidence for `M090_CERT.2`, `M090_CERT.8`, `M090_CERT.9`, or
the final release decision.

## Semantic-Oracle Validation

| Gate | Evidence |
|---|---|
| Semantic golden fixtures | EIR, G91, FAEEL, GPT, split, examples, world-view, GPU-plan contract, executable-plan contract, GPU-workspace layout/reset, candidate-generation, propagation-staging, candidate-validation, model-membership staging, world-view validation staging, materialization-staging, final-result flag staging, and transfer-budget contract fixtures pass locally. |
| Solver service fixtures | SAT assumptions, learned transfer, MaxSAT, GPU-unimplemented status, and failure modes pass as CPU fixtures. |
| Probabilistic coherence fixtures | Epistemic evidence, accepted-world-view evidence, incremental circuit update, adapter design, and tolerance fixtures pass locally. |
| Parser diagnostics | Positive syntax and negative nested-epistemic typed diagnostics pass in `test_epistemic_eir`. |
| Workspace health subset | Logic, solver, and probabilistic lib suites plus cross-crate checks are the local non-GPU health proxy. |

## Post-Correction Validation

| Command | Result |
|---|---|
| `git diff --check` | PASS |
| `cargo fmt --check` | PASS |
| `cargo test -p xlog-logic --test test_epistemic_gpu_plan` | PASS, 3 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_executable_plan` | PASS, 3 passed, 0 failed |
| `cargo test -p xlog-runtime --test test_epistemic_gpu_workspace` | PASS, 32 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_eir --test test_epistemic_g91 --test test_epistemic_faeel --test test_epistemic_gpt --test test_epistemic_split --test test_epistemic_world_view --test test_epistemic_examples` | PASS, 22 passed, 0 failed |
| `cargo test -p xlog-solve --test solver_service_semantics` | PASS, 5 passed, 0 failed |
| `cargo test -p xlog-prob --test epistemic_prob` | PASS, 5 passed, 0 failed |
| `cargo test -p xlog-ir --lib` | PASS, 14 passed, 0 failed |
| `cargo test -p xlog-runtime --lib` | PASS, 125 passed, 0 failed |
| `cargo test -p xlog-logic --lib` | PASS, 238 passed, 0 failed |
| `cargo test -p xlog-solve --lib` | PASS, 111 passed, 0 failed |
| `cargo test -p xlog-prob --lib` | PASS, 56 passed, 0 failed |
| `cargo check -p xlog-logic -p xlog-ir -p xlog-solve -p xlog-prob` | PASS |
| `cargo check -p xlog-cuda -p xlog-runtime -p xlog-logic -p xlog-ir` | PASS |
| `cargo check -p pyxlog` | PASS |

## Metric Status

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M090_CERT.1 semantic golden tests | 100 percent pass | PARTIAL | Semantic-oracle tests pass, but GPU parity is not proven. |
| M090_CERT.2 solver tests | 100 percent pass for GPU-native solver scope | BLOCKED | Existing `solver_service_semantics` is CPU fixture enumeration, not GPU-native epistemic SAT/MaxSAT/portfolio solving. |
| M090_CERT.3 parser diagnostics | positive and negative syntax fixtures pass | PASS | `test_epistemic_eir` covers explicit syntax and typed nested-epistemic rejection. |
| M090_CERT.4 v0.8 compatibility | v0.8 pyxlog/DTS cert subset rerun after rebase | BLOCKED | v0.8 integration/rebase has not happened. |
| M090_CERT.5 formatting | `cargo fmt --check` pass | PASS | Post-correction formatting gate passed. |
| M090_CERT.6 workspace health | agreed cargo test subset pass | PASS for oracle | Runtime, logic, solve, and prob fixture/lib suites plus cross-crate checks passed. |
| M090_CERT.7 semantic trace fixtures | GPT traces include generated, accepted, and rejected candidate counts | PARTIAL | CPU traces include generated/guess/reduced-model/accepted-world-view/rejection reason fields; candidate-generation, propagation, candidate-validation, row-count-gated model-membership, world-view-validation, accepted-candidate materialization, and final-result flag traces include GPU launch counts with CUDA-event elapsed timing, but full reduced stable-model tuple membership/full final tuple materialization trace counters are missing. |
| M090_CERT.8 GPU-native evidence | GPU launch counts, kernel timings, and zero CPU fallback counters | BLOCKED | GPU-plan, workspace allocation/reset, bounded candidate-generation, propagation, candidate-validation, row-count-gated model-membership, world-view-validation, accepted-candidate materialization, final-result flag kernels, hot-path transfer-budget trace, preflight, counter-guard, and reduced-plan trace contracts exist, but actual stable-model tuple membership population/full final tuple materialization evidence and complete accepted-execution timing are missing. |
| M090_CERT.9 WCOJ evidence | at least one WCOJ-eligible epistemic reduction proves WCOJ planner/runtime dispatch | PARTIAL | Executable-plan and runtime-preflight fixtures prove WCOJ promotion plus 38-B K-clique planner/layout/helper-split metadata, and the counter guard rejects preflight-only WCOJ evidence; runtime dispatch and launch evidence are still missing. |

## Required GPU-Native Evidence Before Closure

Certification must add evidence for:

- production lowering from accepted EIR to executable runtime plans;
- GPU-resident candidate, world-view, model-membership, and rejection buffers;
- GPU kernels for Generate-Propagate-Test phases and full result
  materialization;
- zero CPU fallback counters for candidate enumeration and world-view
  validation;
- WCOJ planner/layout/scheduling/helper-splitting evidence for eligible
  reductions;
- GPU-native SAT/MaxSAT/portfolio assumption lifecycle evidence;
- accepted-world-view probabilistic evidence on the GPU-native exact/provenance
  path with zero CPU-only recomputation.

## Coordination Notes

- This cert snapshot is not a closure claim.
- No v0.8-owned pyxlog public API signatures were changed in this branch.
- No push, tag, release-board update, or merge was performed.
