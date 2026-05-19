# v0.9.0 Epistemic And Solver Semantics Guide

This guide describes the bounded semantic-oracle layer currently implemented on
`feat/v090-epistemic-solver-semantics`. The corrected v0.9.0 goal requires
fully GPU-native accepted epistemic execution; this fixture layer is useful
evidence, but it is not a release path and does not close `G090_GPU`,
`G090_SOLVER`, `G090_PROB`, `G090_CERT`, or `G090_CLOSE`.

## Current Boundary

Epistemic literals are parsed and represented explicitly. Direct lowering to RIR
still returns `UnsupportedEpistemicConstruct`, so `xlog run` is not the accepted
execution path for epistemic programs yet. Use the fixture tests listed below as
semantic oracle tests only.

`plan_epistemic_gpu_execution` now builds a production-facing GPU execution
contract from parsed AST/EIR. That contract records required GPU phases,
GPU-resident buffer categories, WCOJ planner obligations for eligible reduced
ordinary bodies, and zero CPU fallback counters. It does not launch kernels or
close the GPU-native gate.

`compile_epistemic_gpu_execution` now adds a production-lowering step after the
GPU contract is proven. The stats-aware
`compile_epistemic_gpu_execution_with_stats_snapshot` variant forwards
`StatsSnapshot` into the normal compiler pipeline, including optimizer passes,
helper splitting, and WCOJ promotion. This proves the lowering route and WCOJ
planner surface for eligible reductions; it still does not run
Generate-Propagate-Test kernels or validate world views.

`xlog-runtime` also exposes `EpistemicGpuWorkspaceLayout` and
`Executor::allocate_epistemic_gpu_workspace`, which map the plan contract to
device-buffer allocations for candidate assumptions, world views, model
membership, and rejection reasons. `EpistemicGpuRuntimePreflight` consumes an
`EpistemicExecutablePlan`, computes the workspace layout, rejects nonzero CPU
fallback counters, and records WCOJ route/helper metadata before launch.
`Executor::prepare_epistemic_gpu_execution` resets all four workspace buffers
with device `memset_zeros` calls and records `EpistemicGpuWorkspaceResetTrace`
with `host_write_ops = 0`.
`Executor::generate_epistemic_gpu_candidates` launches the
`epistemic_generate_candidate_assumptions_u8` CUDA kernel and records
`EpistemicGpuCandidateGenerationTrace` with one kernel launch, zero host
writes, and CUDA-event elapsed timing for bounded candidate bitsets.
`Executor::propagate_epistemic_gpu_candidates` launches the
`epistemic_propagate_candidates_u8` CUDA kernel and records
`EpistemicGpuPropagationTrace` with one kernel launch, zero host writes, and
CUDA-event elapsed timing for world-view/rejection staging.
`Executor::validate_epistemic_gpu_candidates` launches the
`epistemic_validate_candidate_bits_u8` CUDA kernel and records
`EpistemicGpuCandidateValidationTrace` with one kernel launch, zero host
writes, and CUDA-event elapsed timing for candidate-buffer validation.
`Executor::populate_epistemic_gpu_model_membership_from_tuple_sources` launches
`epistemic_populate_model_membership_from_tuple_source_u8` for zero-arity
bindings and arity-specific tuple-key kernels for arity-one, arity-two, and
arity-three bindings, plus a generic arity-N tuple-key kernel for wider keys
and variable-bound keys. It records `EpistemicGpuModelMembershipTrace` with
stable tuple source row-count reads, tuple-key column device reads,
reduced-output row-count reads for bound-variable tuple keys, zero host writes,
and CUDA-event elapsed timing. This slice preserves source tuple terms in
EIR/GPU-plan metadata and certifies row-scoped reduced stable-model
tuple-source staging from named GPU relation buffers and existing `CudaBuffer`
columns. Ground tuple keys are encoded as expected raw bits plus scalar type
codes and compared against existing relation-cell bytes on device for the
current model-slot row in the specialized arity-one/two/three kernels or the
generic arity-N kernel. Variable-bound tuple keys are matched in the generic
arity-N kernel against reduced-output `CudaBuffer` columns selected from the
reduced rule head column binding. Anonymous keys, aggregate keys, and full
semantic parity still fail closed.
`Executor::validate_epistemic_gpu_world_views` launches the
`epistemic_validate_world_views_u8` CUDA kernel and records
`EpistemicGpuWorldViewValidationTrace` with one kernel launch, zero host
writes, and CUDA-event elapsed timing for bounded world-view validation
staging.
`Executor::materialize_epistemic_gpu_candidates` launches the
`epistemic_materialize_accepted_candidates_u8` CUDA kernel and records
`EpistemicGpuMaterializationTrace` with one kernel launch, zero host writes, and
CUDA-event elapsed timing for accepted-candidate materialization staging.
`Executor::materialize_epistemic_gpu_final_results` launches the
`epistemic_materialize_final_result_flags_u8` CUDA kernel and records
`EpistemicGpuFinalResultMaterializationTrace` with one device row-count read
from `output.num_rows_device()`, one kernel launch, zero host writes, and
CUDA-event elapsed timing for final-result flag staging.
`Executor::materialize_epistemic_gpu_final_tuples` launches
`epistemic_build_final_tuple_row_map_u8` followed by
`epistemic_materialize_final_tuple_column_u8` and records
`EpistemicGpuFinalTupleMaterializationTrace` for a device-resident final-output
`CudaBuffer`, including covered tuple bytes, device row-count read/write
metadata, model-membership bytes checked, world-view slots checked, kernel
launches, zero host writes, and CUDA-event elapsed timing. The row-map kernel
filters output rows by accepted membership, world-view state, and all
variable-bound tuple-key relation matches before the final tuple kernel
compacts reduced output columns. Accepted unary, binary, and multi-membership
fixtures now prove final rows are filtered by bound tuple keys on device.
`EpistemicGpuRuntimeWcojCertification` then requires actual production WCOJ
counter deltas before WCOJ evidence can be certified.
`Executor::execute_epistemic_gpu_execution` wraps the reduced production
runtime plan with preflight, workspace allocation, candidate-generation,
propagation, candidate-validation, `execute_plan` plus before/after counter
tracing, then model-membership staging, world-view validation staging, and
accepted-candidate, final-result flag, and membership-gated final tuple
materialization-staging kernel launches. It also snapshots provider host-transfer counters around the hot path
and records `EpistemicGpuTransferBudgetTrace`, which rejects tracked data-plane
H2D/D2H deltas instead of resetting shared telemetry. That is still incomplete
for the epistemic hot path; tuple-source staging is GPU-backed over existing
relation buffers with row-scoped ground-key comparison through specialized
arity-one/two/three kernels and a generic arity-N kernel, plus row-scoped
variable-bound comparison against reduced-output columns through the generic
arity-N kernel. Final tuple output is gated by the staged membership,
world-view buffers, and accepted unary/binary/multi-membership bound-key
row-filter fixtures. Full world-view semantics, solver coupling, probabilistic
production-path reuse, and broader accepted semantic parity do not dispatch yet.

## GPU And WCOJ Scope

The current epistemic algorithms are not fully GPU-native and do not use the
full WCOJ execution stack. In particular, the bounded v0.9 layer does not route
epistemic Generate-Propagate-Test, splitting, or solver-service execution
through:

- WCOJ planner eligibility;
- WCOJ layout construction;
- skew-aware scheduling;
- helper splitting;
- semantic population of GPU-resident world-view/candidate buffers from actual
  reduced stable-model tuples;
- semantic final query tuple materialization from accepted world views;
- GPU portfolio SAT/MaxSAT dispatch.

Those paths are required G090_GPU/G090_SOLVER/G090_PROB work before v0.9.0 can
close. Existing non-epistemic programs continue to use the normal parser,
stratifier, RIR lowering, runtime, and WCOJ infrastructure where eligible. The
current epistemic branch proves semantic boundary, fixture contracts, GPU-plan,
reduced-runtime-plan, workspace, runtime-preflight, dispatch-counter guard, and
reduced-plan execution-trace contracts plus bounded staging kernels only.

## Prior Goal Reuse

The current v0.9 branch reuses prior closure evidence only at the boundaries the
runtime actually touches:

- Goal 038: reuse the audit discipline. Historical proxy gates, superseded
  evidence, and board/tag actions are not treated as current v0.9 closure
  evidence.
- Goal 038-B: reuse the production K-clique WCOJ path: `MultiwayPlan`,
  `KCliqueVariableOrder`, sorted-layout requirements, runtime histogram
  metadata, cost-gated hash routing, and helper-splitting specs.
- Goal 039: reuse the existing production substrate for chain dispatch, K7/K8
  templates, sort labels, DLPack/zero-transfer discipline, CUDA Graphs, and DTS
  replay certification only when the epistemic runtime path actually invokes
  those surfaces.

Today the epistemic runtime consumes 38-B route metadata and fails closed when a
WCOJ-required K-clique reduction lacks production counter deltas. It now
certifies one accepted K5 WCOJ dispatch through production runtime counters,
and records K7/K8 K-clique max-arity plus full edge-permutation metadata through
runtime preflight, but broader K7/K8 dispatch, skew-scheduling,
helper-splitting, and semantic parity coverage remain incomplete.

The reduced-runtime-plan contract reuses the Goal-038-B WCOJ surfaces. K-clique
epistemic reductions must pass through `MultiwayPlan`, `KCliqueVariableOrder`,
sorted-layout requirements, and helper-splitting specs rather than a parallel
epistemic WCOJ planner. The same plan contract now also records one
`EpistemicTupleMembershipBinding` per epistemic literal so runtime preflight can
reject plans that cannot identify the reduced stable-model tuple predicate to
check.

Release certification must replace the current CPU fixture hot paths with:

- runtime dispatch from accepted EIR into executable GPU plans;
- runtime allocation/use of GPU-resident candidate, world-view,
  model-membership, and rejection buffers;
- GPU kernels for candidate generation, propagation, validation, and
  materialization, including semantic final tuple materialization from accepted
  world views;
- post-hot-path final-result transfer accounting for accepted device outputs;
- WCOJ planner eligibility, layout construction, skew scheduling, and helper
  splitting for eligible epistemic reductions;
- GPU-native SAT/MaxSAT/portfolio solving or documented GPU-backed adapters;
- zero CPU fallback counters for candidate enumeration, world-view validation,
  solver search, and probabilistic recomputation.

## World-View Boundary

`EpistemicWorldView` is the explicit semantic boundary object used by the
fixtures. It is a non-empty set of accepted stable models. Over a world view:

- `know p/arity` is true when `p/arity` appears in every world;
- `possible p/arity` is true when `p/arity` appears in at least one world;
- `not know p/arity` is true when `know p/arity` is false.

The current implementation constructs these world views directly in tests. It
does not yet derive them from arbitrary EIR through GPU-native
Generate-Propagate-Test execution.

## Epistemic Source Surface

The accepted source forms are:

```xlog
#pragma epistemic_mode = faeel
#pragma epistemic_mode = g91

accepted() :- know fact().
accepted() :- possible fact().
accepted() :- not know fact().
accepted() :- not possible fact().
```

Nested epistemic operators are rejected with a typed diagnostic. The EIR
boundary is exposed through `xlog_logic::build_eir`, which preserves epistemic
literals as `EirBodyLiteral::Epistemic`.

## G91 Compatibility

G91 is selected explicitly with:

```xlog
#pragma epistemic_mode = g91
accepted() :- possible fact().
```

In the bounded fixture evaluator, `possible p/arity` succeeds when `p/arity` is
either known or compatibility-possible. Non-epistemic programs remain isolated:
the same non-epistemic source lowers to the same RIR under default mode and G91.

## FAEEL Default

FAEEL is the default mode. In the bounded fixture evaluator:

- `know p/arity` requires founded knowledge;
- `possible p/arity` also requires founded knowledge;
- possible-only support is rejected as `UnfoundedPossible`;
- known plus rejected support is rejected as `Contradiction`;
- otherwise unsatisfied epistemic literals are reported as `UnsatisfiedLiteral`.

## Generate-Propagate-Test

`run_generate_propagate_test` executes a bounded three-phase fixture:

- generate: accept an explicit candidate list and enforce `max_candidates`;
- propagate: prune immediate known/rejected contradictions;
- test: evaluate remaining candidates under bounded FAEEL.

The returned trace records generated, guess, propagated, pruned,
reduced-program-model, tested, accepted, accepted-world-view, rejected, and
rejection-reason counts. These are CPU fixture counts; release certification
still requires GPU launch counters, kernel timings, and zero CPU fallback
counters for the same semantic phases.

## Epistemic Splitting

`split_epistemic_program` builds deterministic components from source rules and
rejects a rule that couples more than one distinct epistemic body predicate. For
accepted split fixtures, `recomposed_rule_indices()` must recover the original
source rule order.

`compile_epistemic_gpu_split_execution` lowers valid epistemic split components
through `compile_epistemic_gpu_execution_with_stats_snapshot`, producing one
GPU executable subplan per epistemic component. This reuses the same reduced
runtime compiler, WCOJ promotion, and helper-splitting surfaces as unsplit
epistemic execution; it is bounded executable-plan evidence, not complete
accepted-runtime parity.

## Solver Services

`xlog_solve::GpuSolverProductionAdapter` is the production-facing solver reuse
adapter for epistemic callers. It is a thin wrapper over the existing
`GpuCdclSolver`; it dispatches `solve_expect_sat`, `solve_expect_unsat`,
workspace-backed UNSAT, bounded weighted MaxSAT candidate checks, and bounded
SAT/MaxSAT portfolio jobs through the GPU CDCL path and exposes zero CPU
assignment/MaxSAT enumeration counters in `GpuSolverProductionTrace`.
`solve_expect_sat_with_gpu_execution_result` and
`solve_expect_unsat_with_gpu_execution_result` additionally accept an
`EpistemicGpuExecutionResult`; the reusable-workspace variant is
`solve_expect_unsat_with_branch_limit_ws_with_gpu_execution_result`. These gates
require stable tuple-source membership, GPU model-membership, world-view, and
materialization traces, zero hot-path transfers, and non-empty final device
output, then dispatch SAT/UNSAT through GPU CDCL.
`solve_assumption_lifecycle_with_gpu_execution_result` applies the same
accepted runtime boundary before recording balanced push/retract counters and
dispatching a bounded SAT/UNSAT lifecycle through existing GPU CDCL calls and a
reusable workspace.
`solve_unsat_and_publish_learned_clause_arena_with_gpu_execution_result` applies
the same boundary before running workspace-backed GPU CDCL UNSAT and publishing
the existing device learned-clause/proof arena plus learned-count buffer with
zero CPU learned-clause transfers.
`solve_unsat_then_reuse_learned_clauses_with_gpu_execution_result` applies the
same boundary before importing that existing device arena into a second
workspace-backed UNSAT solve over the same GPU CNF.
`solve_weighted_maxsat_candidates_with_gpu_execution_result` applies the same
boundary before certifying bounded MaxSAT candidate CNFs through GPU CDCL and
returning the best declared score. `solve_portfolio_with_gpu_execution_result`
applies the boundary before dispatching bounded SAT and MaxSAT jobs through the
same adapter, propagating UNKNOWN/TIMEOUT portfolio statuses without CPU search,
and recording portfolio counters.
`xlog_solve::production_capabilities` reports that GPU CDCL SAT/UNSAT is
available along with the bounded GPU-backed MaxSAT and SAT/MaxSAT portfolio
adapters.

The adapter is partial v0.9 evidence only. It does not yet prove learned-clause
validity across distinct candidate CNFs or full MaxSAT coverage.

`xlog_solve::SolverService` provides the bounded solver API used by semantic
fixtures:

- `assume` and `retract_assumption` model incremental SAT assumptions;
- learned clauses are transferred through `transfer_learned_clauses_to` and
  counted in `SolverServiceTrace`;
- learned clauses derived under active assumptions remain scoped to those
  assumptions;
- `SolveInstance::with_weights` gives fixture-scale MaxSAT soft constraints;
- `SolverServiceStatus` distinguishes `Sat`, `Unsat`, `Unknown`, `Timeout`, and
  `Optimal`;
- GPU portfolio solving is explicitly reported as not implemented for this
  fixture facade.

This facade enumerates assignments on CPU for bounded tests. It is not the
GPU-native solver service required for v0.9.0 release certification.

Run the solver service fixtures and production-adapter source guard:

```bash
cargo test -p xlog-solve --test gpu_solver_production_reuse
cargo test -p xlog-solve --test solver_service_semantics
```

## Probabilistic Integration

`xlog_prob::epistemic` records the bounded probabilistic contract:

- accepted world views become probabilistic evidence through
  `AcceptedWorldViewEvidence`; raw unvalidated guesses must not be consumed as
  evidence;
- epistemic assumptions become fixture evidence literals such as
  `know:rain/0=true`;
- GPU-D4/XGCF supports fixture-level incremental evidence updates without
  changing the circuit fingerprint;
- an external Decision-DNNF text adapter is represented as `DesignOnly`;
- `conditional_probability_from_logs` normalizes probabilities with
  `EPISTEMIC_PROBABILITY_TOLERANCE = 1e-12`.

`xlog_prob::epistemic_production::EpistemicProbProductionAdapter` is the
production-facing exact-path reuse adapter. It requires
`AcceptedWorldViewEvidence`, then routes source or parsed programs through
`ExactDdnnfProgram` GPU exact/provenance compilation and exposes zero CPU-only
probability recomputation counters in `EpistemicProbProductionTrace`.
`compile_source_with_gpu_execution_result` and
`compile_program_with_gpu_execution_result` build that evidence from an accepted
`EpistemicGpuExecutionResult` only after stable-model tuple-source membership,
GPU model-membership/world-view/final-result/final-tuple kernel traces, zero
hot-path transfers, and non-empty final device output are proven.
`compile_and_evaluate_source_with_gpu_execution_result` and
`compile_and_evaluate_program_with_gpu_execution_result` consume the same
accepted runtime evidence once before compiling through `ExactDdnnfProgram` and
evaluating queries from that compiled GPU exact state. The production trace
keeps separate source and parsed-program end-to-end counters as well as the
aggregate knowledge-compilation counter.
`encode_source_pir_cnf_with_gpu_execution_result` and
`encode_program_pir_cnf_with_gpu_execution_result` apply the same accepted
runtime boundary before uploading `GpuPirGraph`/`GpuPirRoots` and calling
`encode_cnf_gpu`.
`evaluate_with_gpu_execution_result` applies the same accepted runtime boundary
before calling `ExactDdnnfProgram::evaluate`.
`evaluate_gpu_with_grads_with_gpu_execution_result` applies the same accepted
runtime boundary before calling `ExactDdnnfProgram::evaluate_gpu_with_grads`.

This adapter is partial v0.9 evidence only. It does not yet cover the broader
probabilistic knowledge-compilation matrix over accepted runtime world views.

Run the probabilistic fixture and production-adapter source guard:

```bash
cargo test -p xlog-prob --test epistemic_prob_production_reuse
cargo test -p xlog-prob --test epistemic_prob
```

## Runnable Examples

The example sources live in `examples/epistemic/`:

| Example | Path | Covered behavior |
|---|---|---|
| EIR boundary | `examples/epistemic/01-eir-boundary.xlog` | explicit EIR epistemic literal |
| G91 compatibility | `examples/epistemic/02-g91-compatibility.xlog` | possible-only support under G91 |
| FAEEL default | `examples/epistemic/03-faeel-default.xlog` | founded known candidate |
| Generate-Propagate-Test | `examples/epistemic/04-gpt-candidate-filter.xlog` | candidate trace counts |
| Splitting | `examples/epistemic/05-splitting.xlog` | deterministic component recomposition |

Run all epistemic examples:

```bash
cargo test -p xlog-logic --test test_epistemic_examples
```

## Certification Commands

Current semantic-oracle validation snapshot:

```bash
cargo fmt --check
cargo test -p xlog-runtime --test test_epistemic_gpu_workspace
cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution -- --nocapture
cargo test -p xlog-logic --test test_epistemic_eir --test test_epistemic_g91 --test test_epistemic_faeel --test test_epistemic_gpt --test test_epistemic_split
cargo test -p xlog-logic --test test_epistemic_examples
cargo test -p xlog-solve --test solver_service_semantics
cargo test -p xlog-prob --test epistemic_prob
cargo test -p xlog-logic --lib
cargo test -p xlog-solve --lib
cargo test -p xlog-prob --lib
cargo check -p xlog-cuda -p xlog-runtime -p xlog-logic -p xlog-ir
cargo check -p xlog-logic -p xlog-ir -p xlog-solve -p xlog-prob
cargo check -p pyxlog
```

These commands validate the CPU-side semantic oracle only. They are not a
substitute for the remaining GPU-native certification evidence, which must
broaden launch counts, kernel timings, accepted semantic parity, solver and
probability traces, and zero CPU fallback counters. The v0.8 pyxlog/DTS
compatibility subset still must be rerun after the v0.8 branch lands and this
branch is rebased or merged onto it.

## Roadmap Status

This guide does not mark v0.9.0 roadmap rows DONE. ROADMAP and release-board
state are closure artifacts and should be updated only after the v0.8 rebase,
full certification, and coordinator approval.
