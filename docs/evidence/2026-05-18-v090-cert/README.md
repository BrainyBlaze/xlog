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
| Semantic golden fixtures | EIR, G91, FAEEL, GPT, split executable-subplan lowering, examples, world-view, GPU-plan tuple-membership contract, executable-plan contract, GPU-workspace layout/reset, candidate-generation, propagation-staging, candidate-validation, model-membership staging, pre-validation stable-tuple-source certification, world-view validation staging, materialization-staging, final-result flag staging, membership-gated final tuple materialization, WCOJ fail-closed gate, accepted K5/K7/K8 WCOJ execution, K7/K8 K-clique planner preflight reuse, accepted unary/binary/multi-membership and negated `not know` nonzero-arity final-row filtering, transfer-budget contract, and final-result transfer accounting fixtures pass locally. |
| Solver service fixtures | SAT assumptions, learned transfer, MaxSAT, GPU-unimplemented status, and failure modes pass as CPU fixtures; the production adapter source test proves SAT/UNSAT, reusable workspace-backed UNSAT, bounded lifecycle, learned-clause arena publication, same-device-CNF learned-clause import/reuse, distinct-CNF learned-clause import rejection, bounded MaxSAT candidate, and bounded SAT/MaxSAT/status-aware portfolio reuse of the existing GPU CDCL API without `SolverService`; accepted runtime fixtures gate SAT, UNSAT, workspace-backed UNSAT, balanced push/retract lifecycle steps, learned-clause arena publication, same-device-CNF learned-clause import/reuse, distinct-CNF learned-clause import rejection, bounded MaxSAT, bounded portfolio dispatch through GPU CDCL, and UNKNOWN/TIMEOUT portfolio status propagation. |
| Probabilistic coherence fixtures | Epistemic evidence, accepted-world-view evidence, incremental circuit update, adapter design, and tolerance fixtures pass locally; the production adapter source test proves accepted evidence gates the existing GPU exact source/program compile path, source/program bounded compile/evaluate path, PIR/CNF encoding path, and query/gradient evaluation path without `EpistemicCircuit`. |
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
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution -- --nocapture` | PASS, 24 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_gates_solver_same_cnf_learned_clause_reuse -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_gpu_execution_result_rejects_distinct_cnf_learned_clause_reuse -- --nocapture` | PASS, 1 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_eir --test test_epistemic_g91 --test test_epistemic_faeel --test test_epistemic_gpt --test test_epistemic_split --test test_epistemic_world_view --test test_epistemic_examples` | PASS, 24 passed, 0 failed |
| `cargo test -p xlog-solve --test gpu_solver_production_reuse` | PASS, 3 passed, 0 failed |
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
| M090_CERT.2 solver tests | 100 percent pass for GPU-native solver scope | PARTIAL | `gpu_solver_production_reuse` plus accepted runtime integration prove SAT/UNSAT, workspace-backed UNSAT, bounded lifecycle, learned-clause arena publication, same-device-CNF learned-clause import/reuse, distinct-CNF learned-clause import rejection, bounded MaxSAT candidate, and bounded SAT/MaxSAT/UNKNOWN/TIMEOUT portfolio production-adapter reuse of `GpuCdclSolver` plus status propagation; broader multi-candidate lifecycle coverage remains missing. |
| M090_CERT.3 parser diagnostics | positive and negative syntax fixtures pass | PASS | `test_epistemic_eir` covers explicit syntax, source-term preservation, and typed nested-epistemic rejection. |
| M090_CERT.4 v0.8 compatibility | v0.8 pyxlog/DTS cert subset rerun after rebase | BLOCKED | v0.8 integration/rebase has not happened. |
| M090_CERT.5 formatting | `cargo fmt --check` pass | PASS | Post-correction formatting gate passed. |
| M090_CERT.6 workspace health | agreed cargo test subset pass | PASS for oracle | Runtime, logic, solve, and prob fixture/lib suites plus cross-crate checks passed. |
| M090_CERT.7 semantic trace fixtures | GPT traces include generated, accepted, and rejected candidate counts | PARTIAL | CPU traces include generated/guess/reduced-model/accepted-world-view/rejection reason fields; candidate-generation, propagation, candidate-validation, tuple-source model-membership staging with specialized arity-one/two/three and generic arity-N row-scoped ground-key comparison plus generic arity-N variable-bound comparison, world-view-validation, accepted-candidate materialization, final-result flag, final tuple row-map, and membership-gated final tuple traces include GPU launch counts with CUDA-event elapsed timing, but semantic parity trace counters are missing. |
| M090_CERT.8 GPU-native evidence | GPU launch counts, kernel timings, and zero CPU fallback counters | BLOCKED | GPU-plan, workspace allocation/reset, bounded candidate-generation, propagation, candidate-validation, tuple-source model-membership staging with specialized arity-one/two/three and generic arity-N row-scoped ground-key comparison plus generic arity-N variable-bound comparison and binding polarity, world-view-validation, accepted-candidate materialization, final-result flag, membership-gated final tuple materialization with a device row-map, hot-path transfer-budget trace, final-result transfer accounting, preflight, counter-guard, accepted K5/K7/K8 WCOJ dispatch traces, K7/K8 preflight reuse of production K-clique edge-permutation metadata, accepted unary/binary/multi-membership/not-know nonzero-arity row-filter traces, accepted-runtime solver CDCL SAT/UNSAT plus workspace-backed UNSAT, bounded lifecycle, learned-clause arena publication, same-device-CNF learned-clause import/reuse, distinct-CNF learned-clause import rejection, bounded MaxSAT, and bounded status-aware portfolio traces with zero CPU search counters, and accepted-runtime probabilistic source/program exact-compile, source/program bounded compile/evaluate, zero-arity evidence conditioning, PIR/CNF encode, query-evaluation, and gradient-evaluation traces with zero CPU recompute counters exist, but full semantic parity, broader multi-candidate lifecycle coverage, and broader probabilistic knowledge-compilation coverage are missing. |
| M090_CERT.9 WCOJ evidence | at least one WCOJ-eligible epistemic reduction proves WCOJ planner/runtime dispatch | PASS | `test_epistemic_gpu_wcoj_execution` compiles epistemic K5, K7, and K8 rules with `know gate()`, registers `EpistemicExecutablePlan::relation_ids`, executes the reduced production runtime plans, observes `EpistemicGpuRuntimeWcojCertification::Certified`, requires `wcoj_clique5_dispatch_count >= 1`, `wcoj_clique7_dispatch_count >= 1`, and `wcoj_clique8_dispatch_count >= 1`, and materializes one final device-row from each accepted world-view path. The K5 fixture also requires sorted-layout and helper-split preflight metadata; K7/K8 fixtures prove epistemic reductions reuse the G39 W6.4 K-clique planner surface with `kclique_wcoj_max_arity` and full edge-permutation counts before production dispatch. |
| M090_CERT.10 nonzero-arity membership | certification includes GPU tuple-key membership evidence for arity >= 1 predicates | PARTIAL | EIR/GPU-plan tests preserve source tuple terms; runtime source/trace tests require arity-one, arity-two, arity-three, and generic arity-N tuple-key kernels over existing relation columns with encoded expected key bits/type codes, device metadata arrays for wider keys, reduced-output bound-value column metadata for variable keys, device byte comparison for ground and variable-bound keys, and binding polarity; accepted runtime fixtures prove unary, binary, multi-membership, and `not know` variable-bound final-row filtering. Broader semantic parity fixtures are still missing. |
| M090_CERT.11 solver production reuse | certification includes traces proving accepted SAT/MaxSAT work used existing GPU solver production paths | PARTIAL | `GpuSolverProductionAdapter` requires accepted GPU runtime evidence before calling existing `GpuCdclSolver` SAT/UNSAT, reusable workspace-backed UNSAT, learned-clause arena publication, same-device-CNF learned-clause import/reuse, bounded lifecycle, bounded MaxSAT candidate, and bounded SAT/MaxSAT portfolio APIs; it rejects distinct-CNF learned-clause import before GPU arena reuse, propagates UNKNOWN/TIMEOUT portfolio statuses with zero CPU search counters, and `GpuSolverProductionTrace::require_production_metric_eligibility` rejects CPU-oracle-only metric traces. Broader multi-candidate lifecycle coverage remains missing. |
| M090_CERT.12 prob production reuse | certification includes traces proving accepted probabilistic evidence used existing GPU exact/provenance paths | PARTIAL | `EpistemicProbProductionAdapter` requires accepted GPU runtime evidence before calling `ExactDdnnfProgram` GPU exact/provenance source/program compile, source/program bounded compile/evaluate with distinct trace counters, zero-arity conditioned source evaluation through parsed `Evidence` AST entries, `GpuPirGraph`/`GpuPirRoots` upload plus `encode_cnf_gpu`, and query/gradient-evaluation APIs, exposes zero CPU recompute counters, and `EpistemicProbProductionTrace::require_production_metric_eligibility` rejects fixture-only metric traces. Broader knowledge-compilation coverage is missing. |
| M090_CERT.13 no parallel engines | source audit reports zero new epistemic-only WCOJ, solver-search, probability-inference, or tuple-store engines in accepted paths | PARTIAL | `docs/evidence/2026-05-18-v090-production-reuse-audit/README.md` plus `test_epistemic_production_reuse_audit` source-check the accepted runtime, tuple-membership, solver, and probability paths for reuse of existing RIR/WCOJ metadata, `CudaBuffer` relation columns, `GpuCdclSolver`, `ExactDdnnfProgram`, `GpuPirGraph`, and `encode_cnf_gpu`; this is still partial because broader multi-candidate solver lifecycle coverage and broader probabilistic knowledge-compilation coverage remain blocked. |

## Required GPU-Native Evidence Before Closure

Certification must add evidence for:

- production lowering from accepted EIR to executable runtime plans;
- GPU-resident candidate, world-view, model-membership, and rejection buffers;
- GPU kernels for Generate-Propagate-Test phases and full result
  materialization;
- zero CPU fallback counters for candidate enumeration and world-view
  validation;
- broader WCOJ runtime-dispatch and helper/skew coverage beyond the accepted
  K5/K7/K8 fixtures;
- broader accepted SAT multi-candidate lifecycle and MaxSAT coverage beyond
  same-device-CNF reuse plus distinct-CNF rejection;
- accepted-world-view probabilistic evidence on broader GPU-native
  knowledge-compilation paths with zero CPU-only recomputation.

## Coordination Notes

- This cert snapshot is not a closure claim.
- No v0.8-owned pyxlog public API signatures were changed in this branch.
- No push, tag, release-board update, or merge was performed.
