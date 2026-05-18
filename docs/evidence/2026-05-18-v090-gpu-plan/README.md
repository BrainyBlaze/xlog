# v0.9.0 G090_GPU Plan-Contract Evidence

Date: 2026-05-18

Goal node: `G090_GPU - GPU-Native Runtime And WCOJ Execution`

Branch: `feat/v090-epistemic-solver-semantics`

## Scope

This slice adds a production-facing GPU execution plan contract. Later runtime
evidence adds bounded candidate-generation, propagation, candidate-buffer
validation, model-membership, world-view-validation, and materialization-staging
kernels, including final-result flag staging and final tuple materialization,
but this plan-contract evidence alone does not close `G090_GPU`.

## Implementation Summary

| Requirement | Evidence |
|---|---|
| Accepted EIR reaches a production-facing plan contract | `plan_epistemic_gpu_execution` builds `EpistemicGpuPlan` from parsed AST/EIR. |
| GPU hot-path phases are explicit | `EpistemicGpuHotPathPhase` requires candidate generation, propagation, world-view validation, and result materialization. |
| GPU-resident buffers are explicit | `EpistemicGpuBufferKind` requires candidate assumptions, world views, model membership, and rejection reasons. |
| Stable-model tuple membership bindings are explicit | `EpistemicTupleMembershipBinding` records the literal index, reduction index, predicate, arity, key columns, key terms, operator, and negation flag for each epistemic literal. |
| CPU fallback counters are tracked | `EpistemicCpuFallbackCounters::is_zero()` covers candidate enumeration, world-view validation, solver search, and probabilistic recomputation. |
| WCOJ planner obligation is visible | Multi-relation reduced bodies are marked `RequiresPlannerEligibility` rather than bypassing the WCOJ planner. |
| Non-epistemic programs stay out of the epistemic plan path | Non-epistemic source returns a typed `UnsupportedEpistemicConstruct` error for this API. |

## Validation

| Command | Result |
|---|---|
| `cargo fmt` | PASS |
| `cargo test -p xlog-logic --test test_epistemic_gpu_plan` | PASS, 8 passed, 0 failed |
| `cargo test -p xlog-ir --lib` | PASS, 14 passed, 0 failed |
| `cargo test -p xlog-logic --lib` | PASS, 238 passed, 0 failed |
| `cargo check -p xlog-logic -p xlog-ir -p xlog-solve -p xlog-prob` | PASS |
| `cargo check -p pyxlog` | PASS |

## Metric Status

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M090_GPU.1 production lowering | accepted epistemic fixture runs through production runtime dispatch | PARTIAL | Plan contract exists; later runtime evidence launches candidate generation/propagation/candidate validation before reduced-plan dispatch, then arity 0-2 tuple-source model-membership staging/world-view-validation/materialization, final-result flag staging, and final tuple materialization; complete tuple-key matching and full accepted semantics remain missing. |
| M090_GPU.2 WCOJ eligibility | at least one epistemic reduction uses the WCOJ planner/path where eligible | PARTIAL | Reduced body is marked `RequiresPlannerEligibility`; later executable-plan evidence reaches the 38-B WCOJ planner surface and runtime evidence fails closed on missing required dispatch counters, but successful planner/runtime dispatch is not certified yet. |
| M090_GPU.3 GPU buffers | candidate, world-view, and rejection state have GPU-resident representations | PARTIAL | Buffer categories are explicit; later runtime evidence allocates/resets them and uses bounded candidate/propagation/validation/arity 0-2 tuple-source model-membership/world-view-validation/materialization/final-result flag/final tuple kernels. |
| M090_GPU.4 kernel coverage | GPU kernels cover candidate generation, propagation, validation, and materialization hot paths | PARTIAL | Later runtime evidence launches candidate-generation, propagation-staging, candidate-buffer validation, arity 0-2 tuple-source model-membership staging, world-view-validation staging, materialization-staging, final-result flag, and final tuple materialization kernels; complete value-level tuple matching remains missing. |
| M090_GPU.5 CPU fallback ban | accepted execution trace records zero CPU candidate enumeration/world-view validation fallbacks | PARTIAL | Plan counters initialize to zero and runtime preflight rejects nonzero counters; arity 0-2 tuple-source staging now has device relation-column and source-term metadata evidence, but complete tuple-matching evidence is missing. |
| M090_GPU.6 launch evidence | certification logs include nonzero GPU launch counts and kernel timing for epistemic execution | PARTIAL | Later runtime traces record candidate-generation, propagation, candidate-validation, arity 0-2 tuple-source model-membership staging, world-view-validation, accepted-candidate materialization, final-result flag, and final tuple materialization launches with CUDA-event elapsed timing; complete stable-model membership timing is missing. |
| M090_GPU.7 parity | GPU output matches semantic oracle on all G91, FAEEL, GPT, and splitting fixtures | BLOCKED | Bounded final-output device buffers exist, but semantic oracle-matched GPU output from accepted stable-model membership is not proven yet. |
| M090_GPU.8 transfer budget | host-device transfers are bounded and reported; no per-candidate host round trip in hot path | PARTIAL | Later runtime evidence records hot-path provider transfer snapshots and rejects tracked H2D/D2H deltas; final-result transfer accounting is still missing. |

## Remaining Blocker

The next slice must replace fixed arity 0-2 column-read staging with complete
tuple-key matching, semantically gate final tuple output with that membership,
prove successful WCOJ planner/runtime dispatch evidence for eligible
reductions, and produce a real accepted execution trace with full timing and
zero CPU fallback counters.
