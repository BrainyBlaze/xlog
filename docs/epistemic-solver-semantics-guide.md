# v0.9.0 Epistemic And Solver Semantics Guide

This guide describes the bounded semantic-oracle layer and the partial accepted
GPU runtime/production-reuse evidence currently implemented on
`feat/v090-epistemic-solver-semantics`. The corrected v0.9.0 goal still
requires fully GPU-native accepted epistemic execution across the semantic
matrix; the current fixture and bounded accepted-runtime layers are useful
evidence, but they do not close `G090_GPU`, `G090_SOLVER`, `G090_PROB`,
`G090_CERT`, or `G090_CLOSE`.

## Current Boundary

Epistemic literals are parsed and represented explicitly. Direct lowering to RIR
still returns `UnsupportedEpistemicConstruct`, so `xlog run` is not the accepted
execution path for arbitrary epistemic programs yet. Use the fixture and
integration tests listed below as bounded semantic-oracle and accepted-runtime
evidence, not as release closure.

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
staging. The kernel reads the candidate-assumption matrix as the semantic
boundary and rejects a candidate unless every required epistemic literal has
tuple-source support in the model-membership buffer.
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
filters output rows by accepted membership, world-view state, all variable-bound
tuple-key relation matches, and binding polarity before the final tuple kernel
compacts reduced output columns. The materialization trace records
`row_filter_count` and `negated_row_filter_count`. Accepted unary, possible,
not-possible, binary, quaternary generic-arity, multi-membership,
missing-required multi-membership, and `not know` fixtures now prove final rows
are filtered or rejected by bound tuple keys on device, and preflight records
explicit `know`/`possible`/`not know`/`not possible` operator counts.
`EpistemicGpuRuntimeWcojCertification` then requires actual production WCOJ
counter deltas before WCOJ evidence can be certified, and its certified result
carries the dispatched plan's edge-permutation, stream-group scheduling,
skew-scheduled helper count, sorted-layout, and helper-split counts.
`Executor::execute_epistemic_gpu_execution` wraps the reduced production
runtime plan with preflight, workspace allocation, candidate-generation,
propagation, candidate-validation, `execute_plan` plus before/after counter
tracing, then model-membership staging, candidate-assumption-aware world-view
validation staging, and accepted-candidate, final-result flag, and
membership-gated final tuple materialization-staging kernel launches. It also
snapshots provider
host-transfer counters around the hot path and records
`EpistemicGpuTransferBudgetTrace`, which rejects tracked data-plane H2D/D2H
deltas instead of resetting shared telemetry. That is still incomplete for the
epistemic hot path; tuple-source staging is GPU-backed over existing relation
buffers with row-scoped ground-key comparison through specialized
arity-one/two/three kernels and a generic arity-N kernel, plus row-scoped
variable-bound comparison against reduced-output columns and negated polarity
through the generic arity-N kernel. Final tuple output is gated by the staged
membership and world-view buffers, with accepted unary, possible, not-possible,
binary `know`, binary `possible`, binary `not possible`, binary `not know`,
quaternary generic arity-N `know`, multi-membership, missing-required
multi-membership, and unary `not know` bound-key row-filter
fixtures. Full arbitrary-world enumeration, complete semantic parity, and
release-wide solver/probabilistic coverage do not dispatch yet, but bounded
accepted runtime fixtures now feed the solver and probabilistic production
adapters described below.

## GPU And WCOJ Scope

The current epistemic algorithms are still not fully GPU-native across the full
semantic matrix, but the accepted bounded runtime path now exercises the
production GPU/WCOJ stack for specific certification fixtures:

- WCOJ planner eligibility and runtime dispatch are observed for an accepted
  v0.7.0 4-cycle `MultiWayJoin` reduction, and layout, skew-aware scheduling,
  helper-splitting specs, helper relation rewrites, and runtime histogram
  metadata are observed for accepted K5/K6/K7/K8 epistemic reductions.
- Candidate generation, propagation staging, candidate-buffer validation,
  tuple-source model-membership staging, world-view validation,
  accepted-candidate materialization, final-result flag staging, final-row map
  construction, and membership-gated final tuple materialization use GPU
  workspace/output buffers with zero CPU candidate/world-view fallback counters.
- Unary and binary nonzero-arity `know`/`possible`/`not possible` slices,
  binary `not know`, quaternary generic arity-N `know`, unary `not know`,
  multi-membership, missing-required rejection, split
  possible-vs-not-known, split binary `possible`/`not possible`, split
  all-binary-operator, G91 self-support, and independently founded FAEEL
  fixtures compare bounded GPU traces against semantic or GPT oracles.
- Solver SAT/UNSAT, lifecycle, split-batch learned-clause, MaxSAT,
  weighted MaxSAT encoding, scheduler, and portfolio slices route accepted GPU evidence into existing
  GPU CDCL/CNF adapter paths.
- Probabilistic source/program compile, condition, PIR/CNF encode, query, and
  gradient slices route accepted GPU evidence into existing GPU exact/provenance
  paths with accepted split-batch conditioned source/program query and gradient
  counters,
  source/program-specific exact-query, PIR/CNF, and conditioned-gradient
  counters.

Those paths are still partial G090_GPU/G090_SOLVER/G090_PROB evidence. Before
v0.9.0 can close, release certification must broaden arbitrary EIR semantic
parity, accepted split coverage, solver semantic integration, probabilistic
world-view coverage, and the post-v0.7.0/v0.8.0/v0.8.5/v0.8.6 compatibility rerun. Existing
non-epistemic programs continue to use the normal parser, stratifier, RIR
lowering, runtime, and WCOJ infrastructure where eligible.

## Prior Goal Reuse

The current v0.9 branch reuses prior closure evidence only at the boundaries the
runtime actually touches:

- v0.7.0: reuse the general WCOJ architecture and runtime expansion, including
  first-class `RirNode::MultiWayJoin`, deterministic 4-cycle dispatch,
  recursive/SCC support, and variable-ordering/cost surfaces.
- Goal 038: reuse the audit discipline. Historical proxy gates, superseded
  evidence, and board/tag actions are not treated as current v0.9 closure
  evidence.
- Goal 038-B: reuse the production K-clique WCOJ path: `MultiwayPlan`,
  `KCliqueVariableOrder`, sorted-layout requirements, runtime histogram
  metadata count/timing, cost-gated hash routing, skew scheduling, and
  helper-splitting specs.
- Goal 039: reuse the existing production substrate for chain dispatch, K7/K8
  templates, sort labels, DLPack/zero-transfer discipline, CUDA Graphs, and DTS
  replay certification only when the epistemic runtime path actually invokes
  those surfaces.

Today the epistemic runtime consumes v0.7.0 `MultiWayJoin` metadata plus 38-B
route metadata and fails closed when a WCOJ-required non-hash `MultiWayJoin` or
K-clique reduction lacks production counter deltas. It now certifies accepted
v0.7.0 4-cycle plus K5, K6, K7, and K8 WCOJ dispatch through production runtime
counters. The v0.7.0 trace records `certified_multiway_reductions`; the K5/K6
certified dispatch traces include edge-permutation, stream-group scheduling,
skew-scheduled helper, sorted-layout, and helper-split counts; K6 also carries
runtime histogram metadata-build timing, while K7/K8 record K-clique max-arity
plus full edge-permutation and stream-group metadata through runtime preflight.
Broader semantic parity coverage remains incomplete.

The reduced-runtime-plan contract reuses the v0.7.0 general WCOJ surfaces and
the Goal-038-B WCOJ surfaces. Non-K-clique epistemic reductions must pass
through `RirNode::MultiWayJoin` and the existing production dispatch counters;
K-clique epistemic reductions must pass through `MultiwayPlan`,
`KCliqueVariableOrder`, sorted-layout requirements, and helper-splitting specs
rather than a parallel epistemic WCOJ planner. The same plan contract now also records one
`EpistemicTupleMembershipBinding` per epistemic literal so runtime preflight can
reject plans that cannot identify the reduced stable-model tuple predicate to
check.

Release certification must finish and broaden:

- arbitrary EIR runtime dispatch into executable GPU plans;
- GPU-resident candidate, world-view, model-membership, and rejection buffers
  across the full fixture matrix;
- GPU kernels for all Generate-Propagate-Test phases and semantic final tuple
  materialization from accepted world views;
- post-hot-path final-result transfer accounting for accepted device outputs;
- WCOJ planner eligibility, layout construction, skew scheduling, and helper
  splitting beyond the current bounded v0.7.0 4-cycle and K-clique fixtures;
- GPU-native SAT/MaxSAT/portfolio solving or documented GPU-backed adapters
  for the remaining semantic cases;
- zero CPU fallback counters for candidate enumeration, world-view validation,
  solver search, and probabilistic recomputation.

## World-View Boundary

`EpistemicWorldView` is the explicit semantic boundary object used by the
fixtures. It is a non-empty set of accepted stable models. Over a world view:

- `know p/arity` is true when `p/arity` appears in every world;
- `possible p/arity` is true when `p/arity` appears in at least one world;
- `not know p/arity` is true when `know p/arity` is false.
- `not possible p/arity` is true when `possible p/arity` is false.

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

Accepted GPU runtime coverage now includes
`test_epistemic_gpu_wcoj_execution::g91_self_supported_possible_reaches_gpu_runtime_path`.
That fixture runs explicit `#pragma epistemic_mode = g91` with
`p() :- possible p().`, loads the reduced nullary fact through the existing
relation-buffer path, validates stable tuple-source membership, accepts one
world view, materializes `p()`, and records zero CPU candidate/world-view
fallback counters. This is a G91 runtime parity slice, not full parity closure.

## FAEEL Default

FAEEL is the default mode. In the bounded fixture evaluator:

- `know p/arity` requires founded knowledge;
- `possible p/arity` also requires founded knowledge;
- possible-only support is rejected as `UnfoundedPossible`;
- known plus rejected support is rejected as `Contradiction`;
- otherwise unsatisfied epistemic literals are reported as `UnsatisfiedLiteral`.

At the production executable-plan boundary, default FAEEL also rejects direct
self-support such as `p() :- possible p().` before the reduced ordinary runtime
plan is compiled. If `p/arity` has separate ordinary support without epistemic
body literals, the self-`possible` rule is treated as independently founded and
may lower into accepted GPU runtime execution. The unsupported self-support
fixture is allowed only with explicit `#pragma epistemic_mode = g91`.

## Generate-Propagate-Test

`run_generate_propagate_test` executes a bounded three-phase fixture under the
default FAEEL semantics. `run_generate_propagate_test_with_mode` uses the same
pipeline with an explicit semantics mode, including G91 compatibility checks:

- generate: accept an explicit candidate list and enforce `max_candidates`;
- propagate: prune immediate known/rejected contradictions;
- test: evaluate remaining candidates under the selected bounded semantics.

The returned trace records generated, guess, propagated, pruned,
reduced-program-model, tested, accepted, accepted-world-view, rejected, and
rejection-reason counts. The outcome also records accepted and rejected
candidate indices in oracle order. These are CPU fixture counts; release
certification still requires GPU launch counters, kernel timings, and zero CPU
fallback counters for the same semantic phases.

Accepted GPU execution also records `EpistemicGpuSemanticTrace` after the
hot-path transfer-budget window. That trace reads bounded rejection-reason
metadata from the device buffer and reports generated, guess, propagated,
pruned, tested, reduced-model-slot, accepted-world-view, rejected-candidate,
accepted/rejected candidate indices, and rejection-reason counts with zero CPU
candidate enumeration and zero CPU world-view validation counters.
`EpistemicGpuRejectionReason` decodes nonzero device rejection codes so bounded
GPU traces can be compared with the GPT oracle's phase counts,
accepted/rejected candidate indices, and typed rejection expectations. The G91
self-supported `possible p()` runtime fixture uses the mode-aware oracle for
generated, propagated, tested, accepted, rejected, and candidate-index parity.
The independently founded FAEEL self-`possible p()` runtime fixture uses the
default oracle for the same trace/candidate-index parity.
The accepted multi-membership fixture also compares the two-literal
`know edge(X), know color(X)` candidate matrix against the bounded GPT oracle,
including generated, propagated, tested, accepted, rejected, and
accepted/rejected candidate-index fields.
The unary nonzero-arity `possible edge(X)`, `not possible edge(X)`, and
`not know edge(X)` fixtures, plus the binary `not know edge(X, Y)` fixture,
compare the same trace and candidate-index fields against bounded GPT oracles;
for the negated operators, candidate index 1 is the oracle slot where the
negated literal is true.
The quaternary `know fact4(A, B, C, D)` fixture exercises the generic arity-N
bound-output tuple-key path and compares the same trace and candidate-index
fields against a bounded GPT oracle.
This is certification evidence for
bounded runtime fixtures, not full semantic parity across every
G91/FAEEL/GPT/splitting case.

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
`Executor::execute_epistemic_gpu_execution_batch` executes component executable
plans in order by delegating each item to the existing single-plan GPU runtime
path. The traced wrapper
`Executor::execute_epistemic_gpu_execution_batch_with_trace` returns
`EpistemicGpuBatchExecutionTrace`, which aggregates component executions,
accepted/rejected counts, zero CPU recomposition steps, zero CPU
candidate/world-view fallback counters, and zero per-candidate host round trips.
The accepted split integration fixture uses this adapter for two independent
components and checks final component rows against tuple-intersection oracles.
It also compares each component's generated, propagated, tested, accepted,
rejected, and candidate-index trace fields against bounded GPT oracles while
preserving zero CPU candidate/world-view fallback counters.

Split runtime coverage also includes binary operator components: independent
`possible edge(X, Y)` and `not possible blocked(X, Y)` components execute over
shared `pair/2` input, map results by source rule index, compare each component
against bounded GPT oracle traces, and record aggregate `possible` and
`not possible` operator counts with zero CPU fallback counters.
The all-binary-operator split fixture extends this to four independent
components over the same `pair/2` input, covering `know`, `possible`,
`not possible`, and `not know` together with per-component GPT oracle
trace/candidate-index parity, row-filter polarity counts, one aggregate counter
for each operator, zero CPU recomposition, zero fallback, zero tracked D2H, and
zero per-candidate host round trips.
It also includes the existing world-view distinction between
absent `possible` and true `not know`: split `possible edge(X)` and
`not know edge(X)` components execute over the same absent `edge` tuple source,
returning `[]` for `possible_edge` and `[1, 2, 3]` for `not_known_edge` with
zero CPU candidate/world-view fallback counters.

## Solver Services

`xlog_solve::GpuSolverProductionAdapter` is the production-facing solver reuse
adapter for epistemic callers. It is a thin wrapper over the existing
`GpuCdclSolver`; it dispatches `solve_expect_sat`, `solve_expect_unsat`,
workspace-backed UNSAT, bounded weighted MaxSAT candidate checks, single-result,
multi-result, and accepted split-batch MaxSAT search pruning, single-result,
multi-result, and accepted split-batch weighted MaxSAT selection encoding,
heterogeneous and accepted split-batch MaxSAT scheduler jobs, and
single-result plus multi-result bounded SAT/MaxSAT portfolio jobs
through the GPU CDCL path and exposes zero CPU assignment/MaxSAT enumeration
counters in `GpuSolverProductionTrace`.
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
reusable workspace. Lifecycle expectations also support bounded UNKNOWN and
TIMEOUT status propagation with diagnostic/budget validation and zero CPU search
counters.
`solve_multi_candidate_assumption_lifecycle_with_gpu_execution_results` applies
that boundary to multiple accepted GPU runtime results, dispatches the same
SAT/UNSAT lifecycle per result through GPU CDCL, reports
`candidate_evidence_records`, and records aggregate balanced push/retract and
workspace-reuse counters without CPU search.
`solve_assumption_lifecycle_with_gpu_batch_execution_result` consumes
`GpuSolverProductionBatchExecutionEvidence` from
`execute_epistemic_gpu_execution_batch_with_trace`, validates aggregate
`EpistemicGpuBatchExecutionResult` evidence for zero CPU recomposition, zero
CPU candidate/world-view fallback, zero tracked D2H, and zero per-candidate host
round trips, then dispatches each accepted component through the existing
multi-candidate GPU CDCL lifecycle path. The trace records
`accepted_gpu_batch_candidate_evidence_consumed` and
`accepted_gpu_batch_candidate_component_evidence_consumed` alongside the usual
candidate evidence, balanced push/retract, workspace reuse, and GPU CDCL solve
counters.
`solve_unsat_and_publish_learned_clause_arena_with_gpu_execution_result` applies
the same boundary before running workspace-backed GPU CDCL UNSAT and publishing
the existing device learned-clause/proof arena plus learned-count buffer with
zero CPU learned-clause transfers.
`solve_unsat_then_reuse_learned_clauses_with_gpu_execution_result` applies the
same boundary before importing that existing device arena into a second
workspace-backed UNSAT solve over the same GPU CNF. Distinct candidate CNFs
are rejected before import, incrementing
`gpu_learned_clause_reuse_rejections` while keeping CPU learned-clause transfer
counters at zero.
`solve_multi_candidate_learned_clause_reuse_with_gpu_execution_results`
validates multiple accepted GPU runtime results up front, then repeats the
same-device-CNF learned-clause publication/import path through the existing GPU
CDCL workspace once per accepted evidence record while keeping learned-clause
transfer counters at zero.
`solve_learned_clause_reuse_with_gpu_batch_execution_result` consumes the same
accepted split-batch evidence as the lifecycle adapter, then delegates each
component to the existing multi-candidate same-device-CNF learned-clause reuse
path. It records the split-batch candidate/component counters together with
accepted candidate records, workspace-backed UNSAT solves, arena publications,
imports, reused solves, and zero CPU learned-clause transfers.
`solve_weighted_maxsat_candidates_with_gpu_execution_result` applies the same
boundary before certifying bounded MaxSAT candidate CNFs through GPU CDCL and
returning the best declared score.
`solve_multi_candidate_weighted_maxsat_with_gpu_execution_results` validates
multiple accepted GPU runtime results up front, then repeats the same bounded
MaxSAT candidate-set certification through existing GPU CDCL calls once per
accepted evidence record.
`solve_weighted_maxsat_candidates_with_gpu_batch_execution_result` consumes the
same accepted split-batch evidence as the lifecycle adapter, then delegates
each component to the existing multi-candidate weighted MaxSAT path. It records
the split-batch candidate/component counters together with accepted candidate
records, GPU CDCL candidate solves, optima, and zero CPU MaxSAT enumeration.
`solve_weighted_maxsat_search_with_gpu_execution_result` applies the accepted
runtime boundary before scoring satisfiable MaxSAT candidates through GPU CDCL
SAT and pruning UNSAT candidates through the workspace-backed GPU CDCL UNSAT
path, recording `gpu_maxsat_unsat_candidate_prunes` with zero CPU MaxSAT
enumeration.
`solve_multi_candidate_weighted_maxsat_search_with_gpu_execution_results`
validates multiple accepted GPU runtime results up front, then repeats that
bounded SAT/UNSAT MaxSAT search-pruning path once per accepted evidence record
through the same GPU CDCL workspace.
`solve_weighted_maxsat_search_with_gpu_batch_execution_result` consumes the
same accepted split-batch evidence as the lifecycle adapter, then delegates
each component to the existing multi-candidate MaxSAT search-pruning path. It
records split-batch candidate/component counters, satisfiable candidates, UNSAT
prunes, GPU CDCL candidate solves, optima, and zero CPU MaxSAT enumeration.
`solve_weighted_maxsat_encoded_search_with_gpu_execution_result` applies the
same boundary before converting caller-declared weighted soft-clause selections
into satisfaction CNFs, uploading those candidates through the existing GPU CNF
layout, and reusing the bounded MaxSAT search path. It records
`gpu_maxsat_candidate_encodes` and `gpu_cdcl_candidate_encodes` alongside GPU
candidate solves and UNSAT prunes, without CPU assignment or subset
enumeration.
`solve_multi_candidate_weighted_maxsat_encoded_search_with_gpu_execution_results`
validates multiple accepted GPU runtime results up front, then repeats that
weighted-selection encoding and GPU CDCL search path once per accepted evidence
record while aggregating candidate-evidence, encode, solve, prune, and optimum
counters.
`solve_weighted_maxsat_encoded_search_with_gpu_batch_execution_result` consumes
the same accepted split-batch evidence as the lifecycle adapter, then delegates
each component to the existing multi-candidate weighted-selection encoding and
GPU CDCL search path while recording split-batch candidate/component counters.
`solve_maxsat_schedule_with_gpu_batch_execution_result` consumes accepted
split-batch evidence before delegating each component to the heterogeneous
MaxSAT scheduler, including candidate-set, search-pruning, weighted encoded
search, UNKNOWN, and TIMEOUT jobs without CPU MaxSAT enumeration.
`solve_portfolio_with_gpu_execution_result`
applies the boundary before dispatching bounded SAT and MaxSAT jobs through the
same adapter, propagating UNKNOWN/TIMEOUT portfolio statuses without CPU search,
and recording portfolio counters.
`solve_multi_candidate_portfolio_with_gpu_execution_results` validates multiple
accepted GPU runtime results up front, then repeats the same SAT/MaxSAT/status
portfolio jobs once per accepted evidence record while aggregating
candidate-evidence and `gpu_portfolio_*` counters.
`solve_portfolio_with_gpu_batch_execution_result` consumes the same accepted
split-batch evidence as the lifecycle adapter, then delegates each component to
the existing multi-candidate portfolio path. It records the split-batch
candidate/component counters together with doubled SAT/MaxSAT/UNKNOWN/TIMEOUT
portfolio counters and zero CPU search.
Accepted GPU candidate evidence also preserves the runtime epistemic mode;
`GpuSolverProductionTrace` counts G91 and default FAEEL candidate evidence
separately when either mode gates solver production work.
The same trace records accepted operator-family evidence counters for `know`,
`possible`, `not possible`, and `not know` when accepted runtime evidence gates
solver lifecycle work.
Accepted ternary and quaternary nonzero-arity evidence fixtures also gate the
SAT path, requiring tuple-source membership and recording nonzero-arity
candidate evidence plus tuple-key column reads before GPU CDCL dispatch.
`xlog_solve::production_capabilities` reports that GPU CDCL SAT/UNSAT is
available along with the bounded GPU-backed MaxSAT and SAT/MaxSAT portfolio
adapters. `GpuSolverProductionTrace::require_production_metric_eligibility`
is the automated metric gate: it rejects traces that did not consume accepted
GPU candidate evidence, that have no existing GPU solver production counter,
or that record CPU assignment, MaxSAT, or learned-clause transfer counters.

The adapter is partial v0.9 evidence only. It now proves same-CNF reuse,
distinct-CNF fail-closed rejection, a two-record accepted lifecycle, and bounded
UNKNOWN/TIMEOUT lifecycle propagation, plus two-record same-CNF learned-clause
reuse, a mixed unary and binary `possible`/`not possible` plus binary `not know`
operator-result lifecycle,
accepted split-batch lifecycle, learned-clause reuse, MaxSAT, MaxSAT search pruning,
weighted MaxSAT encoding/search, generalized MaxSAT scheduling, and portfolio evidence
with batch/component counters,
two-record/two-CNF bounded MaxSAT candidate-set execution, bounded GPU-CDCL
pruning of UNSAT MaxSAT search candidates for one, two, and split-batch accepted evidence
records, and bounded weighted soft-clause selection encoding for one and two
accepted evidence records, plus two-record and split-batch heterogeneous MaxSAT schedulers
over candidate-set, search-prune, encoded-search, UNKNOWN, and TIMEOUT jobs and
two-record status-aware portfolio dispatch, with G91/default FAEEL mode-specific
and operator-family accepted-evidence trace counters plus ternary and quaternary
nonzero-arity SAT evidence counters. Broader solver semantic integration remains
open.

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
cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_solver_lifecycle_path -- --nocapture
cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_solver_learned_clause_reuse_path -- --nocapture
cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_solver_maxsat_path -- --nocapture
cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_solver_maxsat_search_pruning -- --nocapture
cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_solver_encoded_maxsat_and_scheduler_paths -- --nocapture
cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_solver_portfolio_path -- --nocapture
cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_quaternary_gpu_execution_result_records_solver_nonzero_arity_evidence_trace -- --exact --nocapture
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
hot-path transfers, and non-empty final device output are proven. That accepted
evidence preserves the runtime epistemic mode so the production trace can count
G91 and default FAEEL evidence consumptions separately.
`compile_and_evaluate_source_with_gpu_execution_result` and
`compile_and_evaluate_program_with_gpu_execution_result` consume the same
accepted runtime evidence once before compiling through `ExactDdnnfProgram` and
evaluating queries from that compiled GPU exact state. The production trace
keeps separate source and parsed-program end-to-end counters as well as the
aggregate knowledge-compilation counter.
`compile_and_evaluate_source_for_gpu_execution_results` validates multiple
accepted GPU runtime evidence records up front, then runs one source
compile/evaluate through `ExactDdnnfProgram` per accepted record while
incrementing the same accepted-evidence, source-compile, query-evaluation, and
knowledge-compilation counters.
`compile_and_evaluate_program_for_gpu_execution_results` provides the same
accepted-evidence batch gate for parsed probabilistic programs while keeping the
parsed-program compile and program knowledge-compilation counters separate from
source counters.
`compile_and_evaluate_source_for_gpu_batch_execution_result` and
`compile_and_evaluate_program_for_gpu_batch_execution_result` consume
`EpistemicProbGpuBatchExecutionEvidence` from accepted split execution,
validate aggregate zero CPU recomposition, zero CPU candidate/world-view
fallback, zero tracked D2H, and zero per-candidate host round trips, then route
each accepted component through the same source or parsed-program
compile/evaluate exact path. The trace records batch and component evidence
counters before source/program compile, exact-query, and end-to-end counters
advance.
`compile_and_evaluate_conditioned_source_with_gpu_execution_result` and
`compile_and_evaluate_conditioned_program_with_gpu_execution_result`
additionally turn accepted zero-arity and concrete nonzero-arity tuple
assumptions into parsed exact `Evidence` AST entries before evaluating through
the same GPU exact path. Unary and binary false assumptions from `not know` are
preserved as false parsed evidence entries and counted separately. The trace
records `accepted_evidence_assumptions_consumed`,
`gpu_conditioned_evidence_facts`, `gpu_conditioned_negative_evidence_facts`,
`gpu_conditioned_nonzero_arity_evidence_facts`,
`gpu_conditioned_max_evidence_arity`,
source/program-specific conditioned fact counters, source/program-specific
negative conditioned fact counters, and operator-specific counts for true
`know`, true `possible`, false `know` (`not know`), and false `possible`
(`not possible`) evidence facts. The trace also splits those operator-specific
conditioned evidence counters by source and parsed-program paths, so a mixed
source/program run can prove which exact input path consumed each operator
family.
`compile_and_evaluate_conditioned_source_for_gpu_execution_results` and
`compile_and_evaluate_conditioned_program_for_gpu_execution_results` validate a
batch of accepted GPU runtime records before running per-record conditioning,
so different accepted world views can condition a shared source or parsed
program query set without bypassing `ExactDdnnfProgram`. The source batch path
also preserves false tuple assumptions per accepted record, so a two-record
negative batch records two negative evidence facts and keeps probability
recomputation on the existing exact path.
`compile_and_evaluate_conditioned_source_for_gpu_batch_execution_result`,
`compile_and_evaluate_conditioned_program_for_gpu_batch_execution_result`,
`compile_and_evaluate_conditioned_source_with_grads_for_gpu_batch_execution_result`,
and
`compile_and_evaluate_conditioned_program_with_grads_for_gpu_batch_execution_result`
consume `EpistemicProbGpuBatchExecutionEvidence` from accepted split execution,
validate the aggregate `EpistemicGpuBatchExecutionTrace` for one GPU runtime
execution per component plus zero CPU recomposition, zero CPU candidate/world-view
fallback, and zero per-candidate host round trips, then route each component's
accepted assumptions through the same conditioned source or parsed-program
query/gradient exact path. The trace records
`accepted_gpu_batch_evidence_consumed` and
`accepted_gpu_batch_component_evidence_consumed` before the existing accepted
world-view assumption, exact query, or conditioned gradient counters advance.
`encode_source_pir_cnf_with_gpu_execution_result` and
`encode_program_pir_cnf_with_gpu_execution_result` apply the same accepted
runtime boundary before uploading `GpuPirGraph`/`GpuPirRoots` and calling
`encode_cnf_gpu`, with source/program-specific PIR graph upload and CNF encode
counters.
`encode_source_pir_cnf_for_gpu_execution_results` and
`encode_program_pir_cnf_for_gpu_execution_results` validate accepted evidence
records up front, then reuse the existing GPU PIR/CNF encoder once per record.
`evaluate_with_gpu_execution_result` applies the same accepted runtime boundary
before calling `ExactDdnnfProgram::evaluate`.
`evaluate_gpu_with_grads_with_gpu_execution_result` applies the same accepted
runtime boundary before calling `ExactDdnnfProgram::evaluate_gpu_with_grads`.
Conditioned source and parsed-program gradient paths also record
source/program-specific conditioned gradient counters, so the production trace
distinguishes source-conditioned gradient evaluation from parsed-program
conditioned gradient evaluation.
`evaluate_for_gpu_execution_results` and
`evaluate_gpu_with_grads_for_gpu_execution_results` validate all accepted GPU
runtime evidence records before reusing the already-compiled exact program for
per-record query or gradient evaluation.
`EpistemicProbProductionCapabilities` reports fixture circuits as disallowed
for production metrics, and
`EpistemicProbProductionTrace::require_production_metric_eligibility` rejects
traces that lack accepted world-view evidence, lack an existing GPU
exact/provenance/PIR/CNF counter, or record CPU/fixture recomputation.

This adapter is partial v0.9 evidence only. It covers bounded zero-arity,
nonzero-arity, negative nonzero-arity, parsed-program, ternary and quaternary
source nonzero-arity evidence, quaternary parsed-program nonzero-arity evidence,
two-record source-conditioned query, split-batch source/program compile/evaluate,
split-batch conditioned source/program query and gradient, all-binary-operator
split-batch conditioned source query, and two-record parsed-program-conditioned query
cases, including true `know`, true `possible`, false `possible`/`not possible`,
and false `know`/`not know` operator-result
conditioning, accepted G91/default FAEEL mode-specific trace counters,
source/program-specific exact-query counters,
source/program-specific conditioned gradient counters,
source/program-specific conditioned evidence counters,
source/program-specific operator-conditioned evidence counters,
source/program-specific PIR/CNF counters, plus query/gradient/PIR-CNF reuse,
but not the full query-conditioned
probabilistic matrix over accepted runtime world views.

Run the probabilistic fixture and production-adapter source guard:

```bash
cargo test -p xlog-prob --test epistemic_prob_production_reuse
cargo test -p xlog-prob --test epistemic_prob
cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_probabilistic_source_and_program_end_to_end_paths -- --nocapture
cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_probabilistic_conditioned_source_path -- --nocapture
cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_all_binary_operator_batch_conditions_probabilistic_evidence -- --exact --nocapture
cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_probabilistic_conditioned_program_path -- --nocapture
cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_quaternary_source_probabilistic_evidence_records_nonzero_arity_trace -- --exact --nocapture
cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_quaternary_parsed_program_probabilistic_evidence_records_nonzero_arity_trace -- --exact --nocapture
cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_probabilistic_conditioned_source_gradients -- --nocapture
cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution accepted_split_batch_gates_probabilistic_conditioned_program_gradients -- --nocapture
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

Current partial semantic-oracle, accepted GPU runtime, and production-reuse
validation snapshot:

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

These commands validate the current bounded semantic oracle, accepted GPU
runtime fixtures, and solver/probability production-adapter slices. They are
not a substitute for the remaining release certification evidence, which must
broaden launch counts, kernel timings, accepted semantic parity, solver and
probability traces, and zero CPU fallback counters. The
v0.7.0/v0.8.0/v0.8.5/v0.8.6 compatibility subset must remain green after this
branch is rebased or merged onto it.

## Roadmap Status

This guide does not mark v0.9.0 roadmap rows DONE. ROADMAP and release-board
state are closure artifacts and should be updated only after the
v0.7.0/v0.8.0/v0.8.5/v0.8.6 reuse baseline, full certification, and coordinator
approval.
