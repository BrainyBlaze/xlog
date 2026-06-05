# Epistemic Semantics And EIR

This document records the v0.9.x Epistemic Intermediate Representation (EIR)
boundary and accepted GPU execution surface. EIR exists so epistemic constructs
stay explicit until a semantic mode can evaluate them; they must not be hidden
as ordinary predicate rewrites.

## Source Surface

The accepted frontend surface is finite and explicit:

- `#pragma epistemic_mode = faeel`
- `#pragma epistemic_mode = g91`
- `know atom(...)`
- `possible atom(...)`
- `not know atom(...)`
- `not possible atom(...)`
- finite nested modal chains such as `know possible atom(...)`,
  `not know possible atom(...)`, and `know not possible atom(...)`

`faeel` is the default mode when no pragma is present. `g91` is an explicit
compatibility mode. Finite nested chains are accepted and normalized by the
parser's modal parity/duality rules before EIR/GPU planning; genuinely
unbounded, unsafe, or unfounded modal cycles still fail closed with typed
diagnostics.

## Frontend Representation

`crates/xlog-logic/src/ast.rs` represents epistemic constructs explicitly:

- `EpistemicMode` stores the selected semantics mode in `Directives`.
- `EpistemicOp` stores `know` versus `possible`.
- `EpistemicLiteral` stores operator, explicit negation, and the atom under the
  operator.
- `BodyLiteral::Epistemic` keeps epistemic literals separate from ordinary
  positive and negated atoms.

## EIR Boundary

`crates/xlog-ir/src/eir.rs` defines the crate-level EIR boundary:

- `EirProgram`
- `EirRule`
- `EirBodyLiteral`
- `EirEpistemicLiteral`
- `EirEpistemicMode`
- `EirEpistemicOp`

`xlog_logic::build_eir` converts parsed AST to EIR without lowering to RIR. This
is the required entry point for G91, FAEEL, Generate-Propagate-Test, epistemic
splitting, and production GPU executable-plan lowering.

## Lowering Boundary

Direct ordinary RIR lowering rejects `BodyLiteral::Epistemic` with
`XlogError::UnsupportedEpistemicConstruct { construct: "RIR lowering boundary",
... }`.

That rejection is the direct-RIR boundary, not the accepted release path.
Accepted epistemic programs lower from EIR into production executable plans and
dispatch through GPU-native runtime and WCOJ paths where eligible. Non-epistemic
programs continue using the existing parser, stratifier, RIR lowering, runtime,
and probabilistic paths.

The probabilistic WFS/provenance code still rejects direct epistemic literals
with typed `UnsupportedEpistemicConstruct` errors. The accepted probabilistic
path consumes validated world-view evidence through the GPU exact/provenance
production path; fixture-scale circuit update tests are the test/reference
surface and do not replace production provenance execution.

## World-View Boundary

`EpistemicWorldView` is the explicit semantic boundary object for current
fixtures. It is a non-empty set of accepted stable models:

- `know p/arity` is true when `p/arity` appears in every world;
- `possible p/arity` is true when `p/arity` appears in at least one world;
- `not know p/arity` is true when `know p/arity` is false.
- `not possible p/arity` is true when `possible p/arity` is false.

The semantic fixtures still construct world views directly so operator behavior
remains independently testable. Accepted runtime pilots separately validate
stable-model tuple membership, world-view validation, and accepted tuple
materialization through the GPU-native execution path.

## GPU Execution Plan Contract

`plan_epistemic_gpu_execution` builds a production-facing contract from parsed
AST through EIR. It preserves epistemic literals, records one reduction summary
per epistemic rule, binds each epistemic literal to the reduced stable-model
tuple predicate that must be checked, requires the four GPU hot-path phases,
and initializes the forbidden CPU fallback counters at zero:

- candidate generation;
- propagation;
- world-view validation;
- result materialization.

The contract also requires GPU-resident candidate-assumption, world-view,
model-membership, and rejection-reason buffers. A reduced ordinary body with
three or more positive relational atoms is marked
`RequiresPlannerEligibility`, which means it must still pass through the
production WCOJ planner rather than bypassing eligibility, layout, scheduling, or
helper-splitting decisions.
`EpistemicGpuPlan::validate_tuple_membership_bindings` requires exactly one
matching `EpistemicTupleMembershipBinding` per epistemic literal before runtime
preflight can proceed.

This contract is the planning front half of the shipped execution path:
`plan_epistemic_gpu_execution` itself builds and validates the contract; the
runtime executor consumes it to launch kernels and dispatch the reduced plan
(see the GPU Workspace Contract section).

## Executable Lowering Contract

`compile_epistemic_gpu_execution` and
`compile_epistemic_gpu_execution_with_stats_snapshot` are the first
production-facing lowering routes for accepted epistemic programs. They first
build the explicit `EpistemicGpuPlan`; only after that semantic contract exists
do they strip epistemic literals from the reduced ordinary program and send
that reduced program through the normal `Compiler` pipeline.

That means reduced ordinary bodies reuse production lowering, optimizer passes,
statistics snapshots, helper splitting, and WCOJ promotion. A WCOJ-eligible
epistemic reduction can therefore produce a v0.7.0 `RirNode::MultiWayJoin`,
including the deterministic 4-cycle WCOJ dispatch route, in the reduced runtime
plan instead of bypassing the WCOJ planner surface. K-clique reductions reuse
the Goal-038-B `MultiwayPlan::WcojWithPlan` /
`PlannedHashRoute` route, `KCliqueVariableOrder`, sorted-layout requirements,
helper-splitting specs, and the compiler-created `__w37_helper_*` relation
rewrite when buried-skew helper splitting is selected.

The reuse boundary follows the existing production WCOJ substrate. v0.7.0 is
reused as the general WCOJ architecture and runtime expansion: epistemic lowering
consumes the existing `MultiWayJoin`, 4-cycle dispatch, recursive/SCC,
variable-ordering, and cost surfaces before any K-clique-specific path. Goal 038-B
is reused as the production K-clique WCOJ substrate: epistemic lowering consumes
its planner, sorted-layout, histogram, cost-gate, and helper-splitting surfaces
rather than defining a parallel WCOJ path. Goal 039 is reused as existing
production substrate for chain dispatch, K7/K8 templates, sort-label/DLPack
discipline, CUDA Graphs, and DTS replay certification; the epistemic runtime path
dispatches through those surfaces when a reduction is eligible.

This is the lowering front half of the shipped execution path. The runtime
kernels described in the GPU Workspace Contract section populate and validate the
candidate/world-view buffers, so the reduced plan executes on the accepted v0.9.x
release path.

## GPU Workspace Contract

`xlog-runtime` maps an `EpistemicGpuPlan` to an
`EpistemicGpuWorkspaceLayout` and `EpistemicGpuWorkspace`. The workspace API
allocates the required device buffers as `TrackedCudaSlice` values:

- `candidate_assumptions: TrackedCudaSlice<u8>`;
- `world_views: TrackedCudaSlice<u8>`;
- `model_membership: TrackedCudaSlice<u8>`;
- `rejection_reasons: TrackedCudaSlice<u32>`.

`EpistemicGpuWorkspaceLayout::for_plan` computes concrete buffer sizes from the
number of epistemic literals, reductions, candidate capacity, world capacity,
and reduced-model capacity. Model-membership storage is candidate-scoped:
`max_candidates * reductions * max_models_per_reduction * literal_count`.
Zero capacities are rejected with typed resource errors so the accepted path
cannot silently use empty host-side structures.

`Executor::prepare_epistemic_gpu_execution` now initializes those workspace
buffers on device. The reset path submits `memset_zeros` for candidate
assumptions, world views, model membership, and rejection reasons, then records
`EpistemicGpuWorkspaceResetTrace` with zero host writes. This is the required
initial state for later Generate-Propagate-Test kernels; it is not a substitute
for those kernels.

`Executor::generate_epistemic_gpu_candidates` launches
`epistemic_generate_candidate_assumptions_u8` from the `xlog_epistemic` CUDA
module to populate candidate-assumption bitsets directly in the workspace. The
current kernel covers only bounded bit-mask candidate enumeration and records
`EpistemicGpuCandidateGenerationTrace` with one kernel launch, zero host
writes, and CUDA-event elapsed timing. `Executor::propagate_epistemic_gpu_candidates` then launches
`epistemic_propagate_candidates_u8` to stage generated candidates into the
world-view and rejection-reason buffers, recording
`EpistemicGpuPropagationTrace` with one kernel launch, zero host writes, and
CUDA-event elapsed timing.
`Executor::validate_epistemic_gpu_candidates` launches
`epistemic_validate_candidate_bits_u8` to validate staged candidate bitsets and
world-view activity in device buffers, recording
`EpistemicGpuCandidateValidationTrace` with one kernel launch, zero host
writes, and CUDA-event elapsed timing.
`Executor::populate_epistemic_gpu_model_membership_from_tuple_sources` resolves
`EpistemicTupleMembershipBinding` entries to named reduced stable-model
relations in the executor store and launches
`epistemic_populate_model_membership_from_tuple_source_u8` for zero-arity
bindings and fixed arity-specific tuple-key kernels for arity-one, arity-two,
and arity-three bindings. `EpistemicTupleMembershipBinding::key_columns` records identity
tuple-key column metadata derived from the EIR atom arity, while `key_terms`
preserves the source atom terms required for value-level tuple-key comparison.
The runtime resolves the column references through the existing `CudaBuffer`
schema, relation columns, and device row-count buffers. For ground tuple keys,
the runtime encodes expected raw bits and scalar type codes, then the arity-one,
arity-two, and arity-three CUDA kernels compare the current model-slot
relation-cell bytes against those encoded keys on device. Variable, anonymous,
and aggregate tuple keys still fail closed until bound value buffers exist.
`EpistemicGpuModelMembershipTrace` records zero
reduced-output row-count reads, tuple-source row-count reads, tuple-key column
device reads, zero host writes, `StableModelTupleBuffer`, and CUDA-event
elapsed timing. The old `ReducedOutputRowCountOnly` trace remains as a
fail-closed staging marker for negative tests.
`Executor::validate_epistemic_gpu_world_views` launches
`epistemic_validate_world_views_u8` to check staged model-membership bytes
against active candidate world views and update rejection codes on device,
recording `EpistemicGpuWorldViewValidationTrace` with one kernel launch, zero
host writes, and CUDA-event elapsed timing.
`Executor::materialize_epistemic_gpu_candidates` launches
`epistemic_materialize_accepted_candidates_u8` to stage accepted-candidate flags
back into the world-view buffer from rejection codes, recording
`EpistemicGpuMaterializationTrace` with one kernel launch, zero host writes, and
CUDA-event elapsed timing.
`Executor::materialize_epistemic_gpu_final_results` launches
`epistemic_materialize_final_result_flags_u8` to combine the reduced runtime
output's device row-count scalar with rejection codes and write final-result
flags into world-view slots, recording
`EpistemicGpuFinalResultMaterializationTrace` with one device row-count read,
one kernel launch, zero host writes, and CUDA-event elapsed timing.
`Executor::materialize_epistemic_gpu_final_tuples` launches
`epistemic_materialize_final_tuple_column_u8` to copy reduced-output tuple
columns into a device-resident final-output `CudaBuffer` and write the final
row-count scalar on device, recording
`EpistemicGpuFinalTupleMaterializationTrace` with output column count, row
capacity, covered tuple bytes, one output row-count device read, one final
row-count device write, kernel launches, zero host writes, and CUDA-event
elapsed timing.
These traces now participate in accepted runtime pilots that couple
model-membership bytes to reduced-runtime tuple output, final tuple
materialization, and solver/probabilistic production gates.

`EpistemicGpuRuntimePreflight::for_executable_plan` consumes an
`EpistemicExecutablePlan` before launch. It computes the workspace layout,
rejects nonzero forbidden CPU fallback counters, validates tuple-membership
bindings, and records the reduced runtime rule count plus WCOJ route surfaces,
including total `MultiWayJoin` reductions, planned-hash routes, K-clique WCOJ
plans, K-clique max arity, live edge-permutation slot counts, distinct
stream-group scheduling ids, skew-scheduled helper-plan counts,
sorted-layout requirements, helper-splitting specs, and the certified helper
relation rule/scan counts plus tuple-membership binding count. If a
K-clique route carries helper-split specs but the reduced plan lacks matching
compiler-created helper relation rules and WCOJ input scans of those helpers,
preflight fails closed with `epistemic GPU helper-split certification`.
`EpistemicGpuRuntimeCounters` snapshots the existing production WCOJ counters
around a future epistemic dispatch, and
`EpistemicGpuRuntimeWcojCertification` rejects preflight-only WCOJ metadata
when required non-hash `MultiWayJoin` dispatch counters do not advance, rejects
K-clique plans when K-clique dispatch counters do not advance, and rejects
dispatched K-clique evidence when the plan has sorted-layout obligations but no
layout sort or layout fast-path counter advanced. Certified traces carry
`certified_multiway_reductions`, skew-scheduled helper-plan counts,
helper-split specs, helper relation rules, and WCOJ helper input scans only
after the reduced production plan has passed the helper rewrite gate. The
accepted runtime entry point calls this certification gate immediately after
reduced-plan dispatch and before model-membership/world-view staging, so a
WCOJ-required epistemic reduction now fails closed if the production counters do
not prove dispatch, layout reuse, and the helper rewrite surface.
`Executor::execute_epistemic_gpu_execution` now wraps the reduced production
runtime plan with preflight, workspace allocation, candidate-generation,
propagation, candidate-validation, `execute_plan` plus before/after counter
trace, then model-membership staging, candidate-assumption-aware world-view
validation staging, and accepted-candidate, final-result flag, and final tuple
materialization-staging kernel launches. The same execution path snapshots
provider host-transfer counters
around that hot path and records `EpistemicGpuTransferBudgetTrace`, rejecting
tracked data-plane H2D/D2H deltas instead of resetting shared provider
telemetry.

This workspace is the production path for the current accepted v0.9.x epistemic
execution surface. It proves the buffer categories are allocatable, initialized
on device, inspectable on the runtime side, and tied to actual counter deltas
around production reduced-plan dispatch. Candidate-assumption generation,
propagation staging, candidate-buffer validation, arity-zero through bounded
nonzero-arity stable-model tuple-source model-membership staging, row-scoped
ground and bound-variable tuple-key comparison over existing relation columns,
bounded all-required-membership world-view validation staging,
accepted-candidate materialization staging, final-result flag staging, and final
tuple materialization staging have CUDA kernels. The bounded hot path records
zero tracked host transfers for the accepted runtime evidence. Remaining
non-claims are genuinely unbounded or dynamically shaped tuple keys, unsupported
arities outside the finite metadata layout, and stronger device-resident/no-host
WFS residency; they are not open gaps in the accepted v0.9.x release surface.

## G91 Compatibility Fixture Semantics

`crates/xlog-logic/src/epistemic.rs` contains the bounded fixture evaluator used
by mode-selection tests. It remains an oracle layer for unit-level G91/FAEEL
distinctions; production G91, FAEEL foundedness, GPT, split, and integration
evidence now lives in the accepted runtime gates.

The fixture evaluator uses an `EpistemicInterpretation` with two predicate/arity
sets:

- `known`: facts known in both modes;
- `possible`: compatibility-only possible facts.

For `know p(...)`, both G91 and FAEEL require `p/arity` to be in `known`.
For `possible p(...)`, G91 accepts either `known` or `possible`; FAEEL accepts
only `known` in this bounded fixture layer. That gives a deterministic golden
distinction without routing epistemic programs through RIR.

Non-epistemic programs remain isolated from mode selection. A program with no
`BodyLiteral::Epistemic` compiles to the same RIR plan under the default mode
and under `#pragma epistemic_mode = g91`.

The accepted runtime fixture
`test_epistemic_gpu_wcoj_execution::g91_self_supported_possible_reaches_gpu_runtime_path`
exercises the G91-only self-support case `p() :- possible p().` through the
production reduced runtime path. The reduced empty-body nullary fact is loaded
through the existing relation-buffer/fact path, then model membership is read
from the stable tuple-source buffer, one world view is accepted, and CPU
candidate/world-view fallback counters remain zero. Together with the FAEEL
foundedness, G91, split, GPT, and integration matrices, this closes the
v0.9.x accepted-surface parity gap between the explicit FAEEL and G91 modes.
Broader non-finite or unsupported semantic forms remain typed boundaries, not
release-surface parity gaps.

## FAEEL Default Fixture Semantics

`faeel` is the default epistemic mode. The bounded FAEEL layer in
`crates/xlog-logic/src/epistemic.rs` evaluates parsed epistemic literals against
an `EpistemicInterpretation` and returns `FaeelCandidateResult`.

The current executable fixture core is intentionally small:

- `know p(...)` succeeds only when `p/arity` is in `known`;
- `possible p(...)` succeeds only when `p/arity` is in `known`;
- a `possible` atom that is only in the compatibility `possible` set is rejected
  as `FaeelNoModelReason::UnfoundedPossible`;
- an atom present in both `known` and `rejected` is rejected as
  `FaeelNoModelReason::Contradiction`;
- any other unsatisfied epistemic literal returns
  `FaeelNoModelReason::UnsatisfiedLiteral`.

The bounded FAEEL evaluator is a test fixture for foundedness and
no-model behavior. The GPU accepted path validates the Generate-Propagate-Test
trace contract separately through the runtime epistemic pilots.

At the production executable-plan boundary, default FAEEL also rejects direct
self-support such as `p() :- possible p().` before reduced ordinary runtime
lowering. For zero-arity predicates, if `p/0` has a separate ordinary rule with
no epistemic body literals, the self-`possible` rule is treated as
independently founded and may lower into accepted GPU runtime execution.
Nonzero-arity self-`possible` rules are accepted when lowering can prove
tuple-level foundedness for each finite bound key; unsupported unbounded
self-support still fails closed in default FAEEL. Self-supported world views are
allowed only with explicit `#pragma epistemic_mode = g91`.

## Generate-Propagate-Test Fixture Execution

`run_generate_propagate_test` in `crates/xlog-logic/src/epistemic.rs` provides a
bounded fixture implementation of the Generate-Propagate-Test pipeline:

1. **Generate:** accept an explicit candidate list and enforce
   `GeneratePropagateTestConfig::max_candidates`;
2. **Propagate:** prune candidates with immediate known/rejected
   contradictions;
3. **Test:** evaluate surviving candidates with the bounded FAEEL evaluator.

The outcome carries `GeneratePropagateTestTrace` with generated, guess,
propagated, pruned, reduced-program-model, tested, accepted,
accepted-world-view, rejected, and rejection-reason counts. If the generate
phase exceeds the configured candidate budget, it returns
`XlogError::ResourceExhausted` with context `epistemic GPT candidate guard`.

This CPU fixture makes the phase boundary auditable and serves as a bounded
oracle that takes explicit candidate input. As of v0.9.1 (EGB-01), the accepted
production device path (`compile_epistemic_gpu_execution` →
`execute_epistemic_gpu_execution`) derives the candidate epistemic-assumption
space from the EIR program itself — enumerating the full candidate lattice on the
device — rather than from explicit candidate input, while preserving the same
generated/propagated/tested/accepted/rejected/reason trace contract.

The accepted runtime pilots run these Generate-Propagate-Test phases on
GPU-resident buffers with launch counters, kernel timings, and zero CPU fallback
counters for the current certification matrix.

## Epistemic Splitting Fixture Contract

`build_epistemic_dependency_graph` builds deterministic connected components
from source-order rules. Each component records its rule head plus ordinary,
epistemic, and negated body predicates, but connectivity is restricted to
predicates produced by rule heads and integrity constraints that mention those
heads. Shared extensional inputs can therefore be reused by independent
epistemic components without forcing them into one component, while ordinary
derived dependencies or constraints between epistemic rules still coalesce them
instead of treating them as independently solvable:

- each component records sorted predicate names;
- each component records source rule indices;
- components are sorted lexicographically by predicate list.

This prevents unsafe split certificates such as `a() :- know p().` and
`b() :- a(), know q().` from becoming two independent components. The shared
ordinary predicate `a` keeps the rules in one component. Likewise,
`:- a(), b().` coalesces the `a` and `b` components, while an `a`-only
constraint stays only with the `a` component during split executable lowering.
By contrast, fixtures such as `a(X) :- node(X), know edge(X).` and
`b(X) :- node(X), know color(X).` keep two components because `node/1` is a
shared extensional input rather than a derived component head.

Epistemic integrity constraints are not silently lowered through the reduced
ordinary program (no ordinary-RIR constraint rewrite). As of v0.9.x, ground
modal integrity constraints (`:- know g().`, `:- possible g().`,
`:- not possible g().`) have an explicit GPU semantic representation: a
constraint kernel prunes candidate world views whose body holds, recording the
firing constraint index per candidate (surfaced as
`result.semantic_trace.constraint_violation_indices`). The accepted surface also
covers safe variable-keyed modal constraints and shared-variable joins after
program-level normalization. Constraint forms outside the finite, safe surface
— genuinely unbounded tuple keys, unsupported dynamic predicate shapes, or
unfounded modal cycles — still fail closed with typed diagnostics.

As of v0.9.1 (EGB-06), `split_epistemic_program` no longer blanket-rejects a rule
that couples more than one distinct epistemic body predicate. Such a rule's modal
predicates are unioned into a single component that the joint path solves as a
full modal conjunction over the candidate world view (matching unsplit
execution); independent heads remain independent components. As of v0.9.2,
same-name multi-arity coupling is resolved by arity-qualified tuple sources, and
finite nested-modal-dependent joint conditions are normalized before the joint
path. Only genuinely unsafe, unbounded, or unfounded coupling remains a typed
fail-closed boundary.

For valid split fixtures, `EpistemicSplitPlan::recomposed_rule_indices` sorts the
component rule indices and must equal the unsplit source rule order. This gives a
stable recomposition certificate before later execution layers attach actual
candidate solving to each component.

`compile_epistemic_gpu_split_execution` attaches bounded executable subplans to
valid epistemic split components. Each subprogram is lowered through
`compile_epistemic_gpu_execution_with_stats_snapshot`, so split execution reuses
the same GPU contract, reduced production compiler pipeline, WCOJ promotion, and
helper-splitting surfaces as the unsplit epistemic executable path. This is not a
separate split-only WCOJ or tuple-store engine; it is split-plan evidence paired
with the accepted runtime parity gates.

Accepted split runtime evidence now also includes a world-view parity fixture:
split `possible edge(X)` and `not know edge(X)` components execute through
`Executor::execute_epistemic_gpu_execution_batch` over the same absent stable
tuple source. The GPU path returns no `possible_edge` rows and all
`not_known_edge` node rows, preserving the oracle distinction without CPU
candidate or world-view fallback.
