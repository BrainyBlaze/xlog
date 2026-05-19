# v0.9.0 G090_GPU Executable-Plan Evidence

Date: 2026-05-18

Goal node: `G090_GPU - GPU-Native Runtime And WCOJ Execution`

Branch: `feat/v090-epistemic-solver-semantics`

## Scope

This slice added the first production-facing lowering route after the explicit
epistemic semantic contract was built. Later runtime evidence adds bounded
candidate-generation, propagation, candidate-buffer validation,
model-membership, world-view-validation, materialization-staging, and
final-result flag plus membership-gated final tuple materialization kernels;
this file alone is not a GPU execution claim and does not close `G090_GPU`.

## Reuse Audit

The prior closure/evidence docs were inspected before finalizing this slice:

| Prior goal | Reuse decision |
|---|---|
| G38 | Reuse the prompt-to-artifact audit style and avoid treating stale proxy gates or superseded evidence as closure evidence. |
| G38-B | Reuse the production WCOJ planner surface: `StatsSnapshot`, `MultiwayPlan`, `KCliqueVariableOrder`, sorted-layout requirements, runtime histogram metadata, cost-gated hash routing, helper-splitting specs, and compiler-created helper relation rewrites. Do not create a parallel epistemic WCOJ route. |
| G39 | Reuse the existing production substrate for chain dispatch, K7/K8 templates, sort labels, DLPack/zero-transfer discipline, CUDA Graphs, and DTS replay certification only when the v0.9 epistemic path actually touches those surfaces. Preserve user-gated board/tag/merge discipline and explicit PASS/PENDING/MISSING audit language. |

## Implementation Summary

| Requirement | Evidence |
|---|---|
| Explicit semantic boundary first | `compile_epistemic_gpu_execution` calls `plan_epistemic_gpu_execution` before any reduced ordinary program is compiled. |
| Production runtime plan route | The reduced ordinary program is compiled through `Compiler::compile_program_with_stats_snapshot`. |
| Runtime registration metadata | `EpistemicExecutablePlan` carries the reduced compiler's predicate-to-`RelId` map so accepted runtime callers can register relation buffers against the same IDs used by the production plan. |
| Reduced output boundary | `EpistemicReductionPlan` carries the reduced head predicate name so accepted runtime materialization uses the relation stored by `execute_plan`. |
| FAEEL/G91 foundedness guard | Default FAEEL executable-plan lowering rejects unsupported self-supported `possible` rules before reduced runtime compilation, allows self-`possible` rules with independent ordinary support, and explicit G91 compatibility mode lowers the same bounded self-support fixture through accepted GPU runtime execution. |
| WCOJ planner reuse | `test_epistemic_executable_plan` proves a WCOJ-eligible reduced body reaches `RirNode::MultiWayJoin`. |
| 38-B K-clique reuse | The stats-aware K5 fixture proves `MultiwayPlan::WcojWithPlan`, `KCliqueVariableOrder`, sorted-layout requirements, and helper-splitting specs are preserved for epistemic reductions; runtime preflight now requires the production helper relation rule and WCOJ helper input scan before helper-split metadata can certify reuse. |
| G38-B/G39 K-clique reuse | `test_epistemic_gpu_wcoj_execution::accepted_epistemic_k5_execution_certifies_production_wcoj_dispatch` and `accepted_epistemic_k6_execution_certifies_g38b_helper_histogram_path` prove accepted K5/K6 execution reaches G38-B skew-scheduled helper-split counts, production helper relation rewrites, and the K6 runtime-histogram metadata path; `accepted_epistemic_k7_execution_certifies_production_wcoj_dispatch` and `accepted_epistemic_k8_execution_certifies_production_wcoj_dispatch` prove generated epistemic K7/K8 reductions reach production K7/K8 WCOJ runtime counters and final tuple materialization; `epistemic_k7_k8_reductions_reuse_g39_kclique_planner_preflight_surface` proves K7/K8 reductions carry production K-clique max-arity, full edge-permutation, and stream-group preflight metadata. |

## Validation

| Command | Result |
|---|---|
| `git diff --check` | PASS |
| `cargo fmt --check` | PASS |
| `cargo test -p xlog-logic --test test_epistemic_executable_plan` | PASS, 6 passed, 0 failed |
| `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution -- --nocapture` | PASS, 72 passed, 0 failed |
| `cargo test -p xlog-logic --test test_epistemic_gpu_plan` | PASS, 8 passed, 0 failed |
| `cargo test -p xlog-runtime --test test_epistemic_gpu_workspace` | PASS, 53 passed, 0 failed |
| `cargo test -p xlog-ir --lib` | PASS, 14 passed, 0 failed |
| `cargo test -p xlog-logic --lib` | PASS, 238 passed, 0 failed |
| `cargo test -p xlog-runtime --lib` | PASS, 128 passed, 0 failed |
| `cargo check -p xlog-logic -p xlog-ir -p xlog-runtime` | PASS |
| `cargo check -p pyxlog` | PASS |

## Metric Status

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M090_EIR.6 production route | accepted epistemic forms have a production lowering route | PARTIAL | Executable route exists after the semantic contract; direct `xlog run` lowering still rejects epistemic literals. |
| M090_FAEEL.5 foundedness guard | self-supported epistemic fixture rejected with documented reason | PASS | Default FAEEL `p() :- possible p().` fails closed as `FAEEL foundedness guard` before runtime dispatch, independently founded default FAEEL `p() :- seed().` plus `p() :- possible p().` lowers and executes, and `#pragma epistemic_mode = g91` permits the compatibility fixture through accepted GPU runtime execution. |
| M090_GPU.1 production lowering | accepted epistemic fixture runs through production runtime dispatch | PARTIAL | Reduced runtime plan is produced through the production compiler; later runtime evidence launches candidate generation/propagation/candidate validation before reduced-plan dispatch, then tuple-source model-membership staging with fixed arity-one/two/three and generic arity-N row-scoped ground-key comparison plus generic arity-N variable-bound comparison, all-required-membership world-view-validation/materialization, final-result flag staging, final-row map construction, and membership-gated final tuple materialization; full accepted semantics remain missing. |
| M090_GPU.2 WCOJ eligibility | at least one epistemic reduction uses the WCOJ planner/path where eligible | PASS | Reduced fixtures reach `RirNode::MultiWayJoin` and K-clique `MultiwayPlan`; accepted runtime evidence executes epistemic K5, K6, K7, and K8 fixtures and observes certified production WCOJ dispatch; the runtime certificate rejects sorted-layout obligations without layout sort or layout fast-path evidence, and preflight rejects helper-split specs without production helper relation rules and WCOJ input scans; K5/K6/K7/K8 carry production stream-group scheduling counts; K5/K6 carry explicit skew-scheduled helper counts; K6 proves G38-B helper/histogram reuse, and K7/K8 preflight evidence proves broader G39 K-clique planner metadata is reused. |
| M090_GPU.4 kernel coverage | kernels cover GPT hot paths | PARTIAL | Later runtime evidence launches candidate-generation, propagation-staging, candidate-buffer validation, tuple-source model-membership staging with fixed arity-one/two/three and generic arity-N row-scoped ground-key comparison plus generic arity-N variable-bound comparison, candidate-assumption-aware world-view-validation staging, materialization-staging, final-result flag, final-row map, and membership-gated final tuple materialization kernels; accepted semantic parity remains missing. |
| M090_GPU.6 launch evidence | nonzero GPU launches and timing | PARTIAL | Later runtime traces record candidate-generation, propagation, candidate-validation, tuple-source model-membership staging with fixed arity-one/two/three and generic arity-N row-scoped ground-key comparison plus generic arity-N variable-bound comparison, candidate-assumption-aware world-view-validation, accepted-candidate materialization, final-result flag, final-row map, and membership-gated final tuple materialization launches with CUDA-event elapsed timing; accepted semantic parity timing is incomplete. |

## Remaining Blocker

The next runtime slice must broaden accepted semantic parity beyond the K5/K6/K7/K8
WCOJ fixtures and complete broader helper/skew, solver, and probability
accepted-runtime traces.
