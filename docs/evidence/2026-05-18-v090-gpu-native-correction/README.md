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
| EIR/GPU plan | Epistemic syntax is represented explicitly and `EpistemicGpuPlan` records the first production-facing GPU contract. | PARTIAL until accepted forms execute through production runtime/GPU dispatch. |
| World views | `EpistemicWorldView` fixtures test `know`, `possible`, and `not know`. | ORACLE ONLY until world views are generated/validated on GPU. |
| GPT | CPU fixture records guesses, reduced models, accepted world views, and rejection reasons. | PARTIAL until candidate generation/propagation/validation use GPU-resident buffers. |
| Splitting | CPU split/recompose fixtures pass. | PARTIAL until valid split components execute through GPU-native subplans. |
| Solver | `SolverService` is a CPU fixture facade with SAT/UNSAT/UNKNOWN/TIMEOUT/Optimal statuses. | BLOCKED until GPU-native SAT/MaxSAT/portfolio execution is wired to epistemic candidates. |
| Probabilistic | `AcceptedWorldViewEvidence` guards evidence conditioning in fixtures. | BLOCKED until accepted world-view evidence flows through the GPU-native exact/provenance path without CPU-only recomputation. |
| Certification | Semantic-oracle and GPU-plan contract tests can pass locally. | BLOCKED until GPU launch counts, kernel timings, WCOJ evidence, and zero CPU fallback counters exist. |

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
2. Lower accepted EIR into production runtime plans instead of the current
   `UnsupportedEpistemicConstruct` boundary.
3. Add GPU-resident candidate/world-view/rejection buffers and launch telemetry.
4. Route WCOJ-eligible reductions through existing planner/layout/dispatch
   machinery, including helper-splitting evidence where applicable.
5. Replace CPU solver fixture search in accepted execution with GPU-native
   SAT/MaxSAT/portfolio services or a documented GPU-backed adapter.
6. Feed accepted world-view evidence into the existing GPU-native
   exact/provenance path and report zero CPU-only probability recomputation.

## Validation Status

| Command | Result |
|---|---|
| `git diff --check` | PASS |
| `cargo fmt --check` | PASS |
| `cargo test -p xlog-logic --test test_epistemic_gpu_plan` | PASS, 3 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_eir --test test_epistemic_g91 --test test_epistemic_faeel --test test_epistemic_gpt --test test_epistemic_split --test test_epistemic_world_view --test test_epistemic_examples` | PASS, 22 passed, 0 failed |
| `cargo test -p xlog-solve --test solver_service_semantics` | PASS, 5 passed, 0 failed |
| `cargo test -p xlog-prob --test epistemic_prob` | PASS, 5 passed, 0 failed |
| `cargo test -p xlog-logic --lib` | PASS, 238 passed, 0 failed |
| `cargo test -p xlog-solve --lib` | PASS, 111 passed, 0 failed |
| `cargo test -p xlog-prob --lib` | PASS, 56 passed, 0 failed |
| `cargo check -p xlog-logic -p xlog-ir -p xlog-solve -p xlog-prob` | PASS |
| `cargo check -p pyxlog` | PASS |

These are semantic-oracle and workspace-health checks only. They do not satisfy
the corrected GPU-native release gate.
