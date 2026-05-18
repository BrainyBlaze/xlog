# Epistemic Semantics And EIR

This document records the v0.9.0 Epistemic Intermediate Representation (EIR)
boundary. EIR exists so epistemic constructs stay explicit until a semantic
mode can evaluate them; they must not be hidden as ordinary predicate rewrites.

## Source Surface

The initial frontend surface is intentionally small:

- `#pragma epistemic_mode = faeel`
- `#pragma epistemic_mode = g91`
- `know atom(...)`
- `possible atom(...)`
- `not know atom(...)`
- `not possible atom(...)`

`faeel` is the default mode when no pragma is present. `g91` is an explicit
compatibility mode. Nested epistemic operators such as `know possible p(X)` are
recognized as unsupported epistemic constructs and return a typed diagnostic.

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
splitting, and the still-missing production GPU lowering work.

## Lowering Boundary

Current RIR lowering rejects `BodyLiteral::Epistemic` with
`XlogError::UnsupportedEpistemicConstruct { construct: "RIR lowering boundary",
... }`.

That rejection is a current implementation boundary, not a release solution.
Under the corrected v0.9.0 goal, accepted epistemic programs must lower from EIR
into production executable plans and dispatch through GPU-native runtime and
WCOJ paths where eligible. Non-epistemic programs continue using the existing
parser, stratifier, RIR lowering, runtime, and probabilistic paths.

The probabilistic WFS/provenance code still rejects direct epistemic literals
with typed `UnsupportedEpistemicConstruct` errors. The bounded `G090_PROB`
contract lives in `xlog_prob::epistemic`: accepted world views are compiled as
probabilistic evidence conditions for fixture-scale circuit update tests, not as
hidden rewrites in the production provenance path.

## World-View Boundary

`EpistemicWorldView` is the explicit semantic boundary object for current
fixtures. It is a non-empty set of accepted stable models:

- `know p/arity` is true when `p/arity` appears in every world;
- `possible p/arity` is true when `p/arity` appears in at least one world;
- `not know p/arity` is true when `know p/arity` is false.

The current fixtures construct world views directly so operator behavior is
testable before production execution exists. They do not yet enumerate stable
models, validate world views, or materialize accepted results through
GPU-resident buffers.

## GPU Execution Plan Contract

`plan_epistemic_gpu_execution` builds a production-facing contract from parsed
AST through EIR. It preserves epistemic literals, records one reduction summary
per epistemic rule, requires the four GPU hot-path phases, and initializes the
forbidden CPU fallback counters at zero:

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

This is a planning boundary only. It does not launch kernels, dispatch runtime
plans, or certify GPU execution.

## Executable Lowering Contract

`compile_epistemic_gpu_execution` and
`compile_epistemic_gpu_execution_with_stats_snapshot` are the first
production-facing lowering routes for accepted epistemic programs. They first
build the explicit `EpistemicGpuPlan`; only after that semantic contract exists
do they strip epistemic literals from the reduced ordinary program and send
that reduced program through the normal `Compiler` pipeline.

That means reduced ordinary bodies reuse production lowering, optimizer passes,
statistics snapshots, helper splitting, and WCOJ promotion. A WCOJ-eligible
epistemic reduction can therefore produce a `RirNode::MultiWayJoin` in the
reduced runtime plan instead of bypassing the WCOJ planner surface. K-clique
reductions reuse the Goal-038-B `MultiwayPlan::WcojWithPlan` /
`PlannedHashRoute` route, `KCliqueVariableOrder`, sorted-layout requirements,
and helper-splitting specs.

This still does not execute epistemic semantics. Runtime kernels must populate
and validate the candidate/world-view buffers before the reduced plan is a
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
and reduced-model capacity. Zero capacities are rejected with typed resource
errors so the accepted path cannot silently use empty host-side structures.

`EpistemicGpuRuntimePreflight::for_executable_plan` consumes an
`EpistemicExecutablePlan` before launch. It computes the workspace layout,
rejects nonzero forbidden CPU fallback counters, and records the reduced
runtime rule count plus WCOJ route surfaces, including K-clique WCOJ plans,
planned-hash routes, sorted-layout requirements, and helper-splitting specs.
`EpistemicGpuRuntimeCounters` snapshots the existing production WCOJ counters
around a future epistemic dispatch, and
`EpistemicGpuRuntimeWcojCertification` rejects preflight-only WCOJ metadata
when required K-clique dispatch counters do not advance.

This workspace is still pre-kernel plumbing. It proves the buffer categories are
allocatable and inspectable on the runtime side and that WCOJ certification is
tied to actual counter deltas; it does not yet populate buffers, launch
Generate-Propagate-Test kernels, or produce epistemic kernel timing evidence.

## G91 Compatibility Fixture Semantics

`crates/xlog-logic/src/epistemic.rs` contains the current bounded fixture
evaluator for mode-selection tests. It is not the full production epistemic
executor. It exists to make the G91 compatibility mode testable before the
later FAEEL and Generate-Propagate-Test sub-goals land.

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

The bounded FAEEL evaluator is a certification fixture for foundedness and
no-model behavior. It is not yet the full Generate-Propagate-Test executor; that
pipeline is owned by `G090_GPT`.

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

This fixture makes the phase boundary auditable. It does not yet enumerate
candidate worlds from arbitrary EIR programs; later solver and splitting work can
replace the explicit candidate input while preserving the trace contract.

The release gate additionally requires these same Generate-Propagate-Test phases
to run on GPU-resident buffers with launch counters, kernel timings, and zero CPU
fallback counters.

## Epistemic Splitting Fixture Contract

`build_epistemic_dependency_graph` builds deterministic split components from
source-order rules:

- each component records sorted predicate names;
- each component records source rule indices;
- components are sorted lexicographically by predicate list.

`split_epistemic_program` rejects a rule that couples more than one distinct
epistemic body predicate. In this bounded split layer, such a rule would require
cross-component solving and is rejected with
`XlogError::UnsupportedEpistemicConstruct { construct: "epistemic splitting",
... }`.

For valid split fixtures, `EpistemicSplitPlan::recomposed_rule_indices` sorts the
component rule indices and must equal the unsplit source rule order. This gives a
stable recomposition certificate before later execution layers attach actual
candidate solving to each component.
