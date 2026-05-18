# v0.9.0 GPU-Native Gate Correction

Date: 2026-05-18

Goal document: `docs/plans/2026-05-18-agent-v090-epistemic-solver-goal.md`

Branch: `feat/v090-epistemic-solver-semantics`

## Correction Summary

The corrected goal document makes fully GPU-native accepted epistemic execution
mandatory for v0.9.0. The current branch has valuable CPU-side semantic oracle
fixtures, but those fixtures are incomplete scaffolding and cannot close the GPU
release gate.

## Current Branch Classification

| Area | Current branch state | Release status |
|---|---|---|
| EIR/GPU plan | Epistemic syntax is represented explicitly; `EpistemicGpuPlan` records the GPU contract; `EpistemicExecutablePlan` carries the reduced production runtime plan. | PARTIAL until accepted forms execute through production runtime/GPU dispatch. |
| GPU workspace | `EpistemicGpuWorkspace` maps required buffer categories to runtime `TrackedCudaSlice` handles; `EpistemicGpuWorkspaceResetTrace` records device-side reset with zero host writes; `EpistemicGpuCandidateGenerationTrace` records bounded candidate-assumption kernel launches with CUDA-event elapsed timing; `EpistemicGpuPropagationTrace` records bounded propagation staging launches with CUDA-event elapsed timing; `EpistemicGpuCandidateValidationTrace` records bounded candidate-buffer validation launches with CUDA-event elapsed timing; `EpistemicGpuModelMembershipTrace` records bounded candidate-scoped model-membership staging launches with reduced-output row-count device reads, `ReducedOutputRowCountOnly` source, CUDA-event elapsed timing, and a fail-closed tuple-source certification method; `EpistemicGpuWorldViewValidationTrace` records bounded model-membership/world-view validation launches with CUDA-event elapsed timing; `EpistemicGpuMaterializationTrace` records bounded accepted-candidate materialization launches with CUDA-event elapsed timing; `EpistemicGpuFinalResultMaterializationTrace` records final-result flag launches from reduced-output device row-count metadata with CUDA-event elapsed timing; `EpistemicGpuFinalTupleMaterializationTrace` records final-output tuple buffer launches with device row-count read/write metadata and CUDA-event elapsed timing; `EpistemicGpuTransferBudgetTrace` records zero tracked hot-path host transfers; `EpistemicGpuRuntimePreflight` consumes executable plans and records WCOJ/helper route metadata; `EpistemicGpuRuntimeWcojCertification` rejects WCOJ evidence unless runtime counters advance; `EpistemicGpuRuntimeTrace` records reduced-plan counter deltas and the runtime entry point now fails closed on missing required WCOJ dispatch evidence or row-count-only model membership. | PARTIAL until model membership is populated from actual stable-model tuples and final query results are semantically gated by that membership. |
| World views | `EpistemicWorldView` fixtures test `know`, `possible`, and `not know`. | ORACLE ONLY until world views are generated/validated on GPU. |
| GPT | CPU fixture records guesses, reduced models, accepted world views, and rejection reasons. | PARTIAL: candidate generation, propagation staging, candidate-buffer validation, row-count-gated model-membership staging, bounded world-view validation staging, accepted-candidate materialization staging, final-result flag staging, and final tuple materialization use GPU-resident buffers; actual stable-model tuple membership population still does not. |
| Splitting | CPU split/recompose fixtures pass. | PARTIAL until valid split components execute through GPU-native subplans. |
| Solver | `SolverService` is a CPU fixture facade with SAT/UNSAT/UNKNOWN/TIMEOUT/Optimal statuses. | BLOCKED until GPU-native SAT/MaxSAT/portfolio execution is wired to epistemic candidates. |
| Probabilistic | `AcceptedWorldViewEvidence` guards evidence conditioning in fixtures. | BLOCKED until accepted world-view evidence flows through the GPU-native exact/provenance path without CPU-only recomputation. |
| Certification | Semantic-oracle and GPU-plan contract tests can pass locally. | BLOCKED until full accepted-execution GPU timing, WCOJ evidence, and zero CPU fallback counters exist. |

## Explicit Non-Closure Items

The following corrected goal nodes remain unclosed:

- `G090_GPU`
- `G090_SOLVER`
- `G090_PROB`
- `G090_CERT`
- `G090_CLOSE`

`G090_GPT` and `G090_SPLIT` are also only partial because their GPU-residency
metrics are not implemented.

## Required Next Implementation Slice

The next production slice should start at the lowering/runtime boundary:

1. Define an epistemic executable-plan representation that preserves the
   `EpistemicWorldView` contract and attaches zero-fallback counters. DONE for
   the plan contract in `EpistemicGpuPlan`; runtime execution remains open.
2. Map plan buffer categories to runtime GPU workspace allocations. DONE for
   layout, `TrackedCudaSlice` handles, and device-side reset; kernel use remains
   open.
3. Lower accepted EIR into production runtime plans instead of the current
   `UnsupportedEpistemicConstruct` boundary. DONE for
   `compile_epistemic_gpu_execution` and its stats-aware variant; runtime
   dispatch remains open.
4. Add GPU-resident candidate/world-view/rejection buffer population and launch
   telemetry. PARTIAL for bounded candidate-assumption generation, propagation
   staging, candidate-buffer validation, row-count-gated model-membership staging, bounded
   world-view validation staging, accepted-candidate materialization staging,
   final-result flag staging, final tuple materialization, CUDA-event elapsed
   timing for those staging launches, and hot-path transfer-budget tracing;
   actual stable-model tuple membership population and accepted
   rejection-reason semantic population remain open.
5. Route WCOJ-eligible reductions through existing planner/layout/dispatch
   machinery, including helper-splitting evidence where applicable.
6. Replace CPU solver fixture search in accepted execution with GPU-native
   SAT/MaxSAT/portfolio services or a documented GPU-backed adapter.
7. Feed accepted world-view evidence into the existing GPU-native
   exact/provenance path and report zero CPU-only probability recomputation.

## Validation Status

| Command | Result |
|---|---|
| `git diff --check` | PASS |
| `cargo fmt --check` | PASS |
| `cargo test -p xlog-logic --test test_epistemic_gpu_plan` | PASS, 3 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_executable_plan` | PASS, 3 passed, 0 failed |
| `cargo test -p xlog-runtime --test test_epistemic_gpu_workspace` | PASS, 38 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_eir --test test_epistemic_g91 --test test_epistemic_faeel --test test_epistemic_gpt --test test_epistemic_split --test test_epistemic_world_view --test test_epistemic_examples` | PASS, 22 passed, 0 failed |
| `cargo test -p xlog-solve --test solver_service_semantics` | PASS, 5 passed, 0 failed |
| `cargo test -p xlog-prob --test epistemic_prob` | PASS, 5 passed, 0 failed |
| `cargo test -p xlog-runtime --lib` | PASS, 125 passed, 0 failed |
| `cargo test -p xlog-logic --lib` | PASS, 238 passed, 0 failed |
| `cargo test -p xlog-solve --lib` | PASS, 111 passed, 0 failed |
| `cargo test -p xlog-prob --lib` | PASS, 56 passed, 0 failed |
| `cargo check -p xlog-logic -p xlog-ir -p xlog-solve -p xlog-prob` | PASS |
| `cargo check -p xlog-cuda -p xlog-runtime -p xlog-logic -p xlog-ir` | PASS |
| `cargo check -p pyxlog` | PASS |

These are semantic-oracle and workspace-health checks only. They do not satisfy
the corrected GPU-native release gate.
