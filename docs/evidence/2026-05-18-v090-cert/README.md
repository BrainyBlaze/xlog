# v0.9.0 G090_CERT Evidence

Date: 2026-05-18

Goal node: `G090_CERT - Certification And Regression Gates`

Branch: `feat/v090-epistemic-solver-semantics`

## Certification Scope

This file records semantic-oracle validation plus partial production-reuse
adapter evidence. The corrected v0.9.0 goal requires GPU-native accepted
epistemic execution before `G090_CERT` can close. The current fixture layer and
thin adapters are useful regression evidence, but they are not full
certification evidence for `M090_CERT.8` or the final release decision.

## Semantic-Oracle Validation

| Gate | Evidence |
|---|---|
| Semantic golden fixtures | EIR, G91, FAEEL, GPT, split, examples, world-view, GPU-plan tuple-membership contract, executable-plan contract, GPU-workspace layout/reset, candidate-generation, propagation-staging, candidate-validation, model-membership staging, pre-validation stable-tuple-source certification, world-view validation staging, materialization-staging, final-result flag staging, membership-gated final tuple materialization, WCOJ fail-closed gate, accepted K5 WCOJ execution, and transfer-budget contract fixtures pass locally. |
| Solver service fixtures | SAT assumptions, learned transfer, MaxSAT, GPU-unimplemented status, and failure modes pass as CPU fixtures; the production adapter source test proves SAT/UNSAT reuse of the existing GPU CDCL API without `SolverService` and reports MaxSAT/portfolio as blocked production capabilities. |
| Probabilistic coherence fixtures | Epistemic evidence, accepted-world-view evidence, incremental circuit update, adapter design, and tolerance fixtures pass locally; the production adapter source test proves accepted evidence gates the existing GPU exact path without `EpistemicCircuit`. |
| Parser diagnostics | Positive syntax and negative nested-epistemic typed diagnostics pass in `test_epistemic_eir`. |
| Workspace health subset | Logic, solver, and probabilistic lib suites plus cross-crate checks are the local non-GPU health proxy. |

## Post-Correction Validation

| Command | Result |
|---|---|
| `git diff --check` | PASS |
| `cargo fmt --check` | PASS |
| `cargo test -p xlog-logic --test test_epistemic_gpu_plan` | PASS, 8 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_executable_plan` | PASS, 3 passed, 0 failed |
| `cargo test -p xlog-runtime --test test_epistemic_gpu_workspace` | PASS, 47 passed, 0 failed |
| `cargo test -p xlog-runtime --test test_epistemic_production_reuse_audit -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_eir --test test_epistemic_g91 --test test_epistemic_faeel --test test_epistemic_gpt --test test_epistemic_split --test test_epistemic_world_view --test test_epistemic_examples` | PASS, 23 passed, 0 failed |
| `cargo test -p xlog-solve --test gpu_solver_production_reuse` | PASS, 2 passed, 0 failed |
| `cargo test -p xlog-solve --test solver_service_semantics` | PASS, 5 passed, 0 failed |
| `cargo test -p xlog-solve --test no_dtoh_in_gpu_cdcl` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-prob --test epistemic_prob_production_reuse` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-prob --test epistemic_prob` | PASS, 5 passed, 0 failed |
| `cargo test -p xlog-prob --test no_cpu_d4_in_exact` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-prob --test no_dtoh_in_gpu_exact_path` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-ir --lib` | PASS, 14 passed, 0 failed |
| `cargo test -p xlog-runtime --lib` | PASS, 128 passed, 0 failed |
| `cargo test -p xlog-logic --lib` | PASS, 238 passed, 0 failed |
| `cargo test -p xlog-solve --lib` | PASS, 111 passed, 0 failed |
| `cargo test -p xlog-prob --lib` | PASS, 56 passed, 0 failed |
| `cargo check -p xlog-logic -p xlog-ir -p xlog-solve -p xlog-prob` | PASS |
| `cargo check -p xlog-prob --features host-io` | PASS |
| `cargo check -p xlog-cuda -p xlog-runtime -p xlog-logic -p xlog-ir` | PASS |
| `cargo check -p pyxlog` | PASS |

## Metric Status

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M090_CERT.1 semantic golden tests | 100 percent pass | PARTIAL | Semantic-oracle tests pass, but GPU parity is not proven. |
| M090_CERT.2 solver tests | 100 percent pass for GPU-native solver scope | BLOCKED | `gpu_solver_production_reuse` proves SAT/UNSAT production-adapter reuse of `GpuCdclSolver`, but `production_capabilities` explicitly blocks GPU-native MaxSAT and SAT/MaxSAT portfolio execution. |
| M090_CERT.3 parser diagnostics | positive and negative syntax fixtures pass | PASS | `test_epistemic_eir` covers explicit syntax, source-term preservation, and typed nested-epistemic rejection. |
| M090_CERT.4 v0.8 compatibility | v0.8 pyxlog/DTS cert subset rerun after rebase | BLOCKED | v0.8 integration/rebase has not happened. |
| M090_CERT.5 formatting | `cargo fmt --check` pass | PASS | Post-correction formatting gate passed. |
| M090_CERT.6 workspace health | agreed cargo test subset pass | PASS for oracle | Runtime, logic, solve, and prob fixture/lib suites plus cross-crate checks passed. |
| M090_CERT.7 semantic trace fixtures | GPT traces include generated, accepted, and rejected candidate counts | PARTIAL | CPU traces include generated/guess/reduced-model/accepted-world-view/rejection reason fields; candidate-generation, propagation, candidate-validation, tuple-source model-membership staging with specialized arity-one/two/three and generic arity-N row-scoped ground-key comparison plus generic arity-N variable-bound comparison, world-view-validation, accepted-candidate materialization, final-result flag, and membership-gated final tuple traces include GPU launch counts with CUDA-event elapsed timing, but semantic parity trace counters are missing. |
| M090_CERT.8 GPU-native evidence | GPU launch counts, kernel timings, and zero CPU fallback counters | BLOCKED | GPU-plan, workspace allocation/reset, bounded candidate-generation, propagation, candidate-validation, tuple-source model-membership staging with specialized arity-one/two/three and generic arity-N row-scoped ground-key comparison plus generic arity-N variable-bound comparison, world-view-validation, accepted-candidate materialization, final-result flag, membership-gated final tuple materialization kernels, hot-path transfer-budget trace, preflight, counter-guard, accepted K5 WCOJ dispatch trace, solver production-adapter zero CPU search counters, and probabilistic production-adapter zero CPU recompute counters exist, but full semantic parity and solver/probability accepted-runtime traces are missing. |
| M090_CERT.9 WCOJ evidence | at least one WCOJ-eligible epistemic reduction proves WCOJ planner/runtime dispatch | PASS | `test_epistemic_gpu_wcoj_execution` compiles an epistemic K5 rule with `know gate()`, registers `EpistemicExecutablePlan::relation_ids`, executes the reduced production runtime plan, observes `EpistemicGpuRuntimeWcojCertification::Certified`, requires `wcoj_clique5_dispatch_count >= 1`, and materializes one final device-row from the accepted world-view path. |
| M090_CERT.10 nonzero-arity membership | certification includes GPU tuple-key membership evidence for arity >= 1 predicates | PARTIAL | EIR/GPU-plan tests preserve source tuple terms, and runtime source/trace tests require arity-one, arity-two, arity-three, and generic arity-N tuple-key kernels over existing relation columns with encoded expected key bits/type codes, device metadata arrays for wider keys, reduced-output bound-value column metadata for variable keys, and device byte comparison for ground and variable-bound keys; semantic parity fixtures are still missing. |
| M090_CERT.11 solver production reuse | certification includes traces proving accepted SAT/MaxSAT work used existing GPU solver production paths | PARTIAL | `GpuSolverProductionAdapter` reuses existing `GpuCdclSolver` SAT/UNSAT APIs and exposes zero CPU search counters; `production_capabilities` blocks MaxSAT/portfolio until existing production paths exist. |
| M090_CERT.12 prob production reuse | certification includes traces proving accepted probabilistic evidence used existing GPU exact/provenance paths | PARTIAL | `EpistemicProbProductionAdapter` requires accepted evidence before calling `ExactDdnnfProgram` GPU exact/provenance APIs and exposes zero CPU recompute counters; accepted runtime trace is missing. |
| M090_CERT.13 no parallel engines | source audit reports zero new epistemic-only WCOJ, solver-search, probability-inference, or tuple-store engines in accepted paths | PARTIAL | `docs/evidence/2026-05-18-v090-production-reuse-audit/README.md` plus `test_epistemic_production_reuse_audit` source-check the accepted runtime, tuple-membership, solver, and probability paths for reuse of existing RIR/WCOJ metadata, `CudaBuffer` relation columns, `GpuCdclSolver`, and `ExactDdnnfProgram`; this is still partial because MaxSAT/portfolio and accepted probabilistic runtime traces remain blocked. |

## Required GPU-Native Evidence Before Closure

Certification must add evidence for:

- production lowering from accepted EIR to executable runtime plans;
- GPU-resident candidate, world-view, model-membership, and rejection buffers;
- GPU kernels for Generate-Propagate-Test phases and full result
  materialization;
- zero CPU fallback counters for candidate enumeration and world-view
  validation;
- broader WCOJ planner/layout/scheduling/helper-splitting coverage beyond the
  accepted K5 fixture;
- GPU-native SAT/MaxSAT/portfolio assumption lifecycle evidence;
- accepted-world-view probabilistic evidence on the GPU-native exact/provenance
  path with zero CPU-only recomputation.

## Coordination Notes

- This cert snapshot is not a closure claim.
- No v0.8-owned pyxlog public API signatures were changed in this branch.
- No push, tag, release-board update, or merge was performed.
