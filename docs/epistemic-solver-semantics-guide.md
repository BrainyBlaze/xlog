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
fallback counters, and records WCOJ route/helper metadata before launch. That is
pre-kernel plumbing only; no epistemic runtime dispatch exists yet.

## GPU And WCOJ Scope

The current epistemic algorithms are not fully GPU-native and do not use the
full WCOJ execution stack. In particular, the bounded v0.9 layer does not route
epistemic Generate-Propagate-Test, splitting, or solver-service execution
through:

- WCOJ planner eligibility;
- WCOJ layout construction;
- skew-aware scheduling;
- helper splitting;
- GPU-resident world-view/candidate buffers;
- GPU portfolio SAT/MaxSAT dispatch.

Those paths are required G090_GPU/G090_SOLVER/G090_PROB work before v0.9.0 can
close. Existing non-epistemic programs continue to use the normal parser,
stratifier, RIR lowering, runtime, and WCOJ infrastructure where eligible. The
current epistemic branch proves semantic boundary, fixture contracts, GPU-plan,
reduced-runtime-plan, workspace, and runtime-preflight contracts only.

The reduced-runtime-plan contract reuses the Goal-038-B WCOJ surfaces. K-clique
epistemic reductions must pass through `MultiwayPlan`, `KCliqueVariableOrder`,
sorted-layout requirements, and helper-splitting specs rather than a parallel
epistemic WCOJ planner.

Release certification must replace the current CPU fixture hot paths with:

- runtime dispatch from accepted EIR into executable GPU plans;
- runtime allocation/use of GPU-resident candidate, world-view,
  model-membership, and rejection buffers;
- GPU kernels for candidate generation, propagation, validation, and
  materialization;
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

## Solver Services

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

Run the solver service fixture:

```bash
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

Run the probabilistic fixture:

```bash
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
cargo test -p xlog-logic --test test_epistemic_eir --test test_epistemic_g91 --test test_epistemic_faeel --test test_epistemic_gpt --test test_epistemic_split
cargo test -p xlog-logic --test test_epistemic_examples
cargo test -p xlog-solve --test solver_service_semantics
cargo test -p xlog-prob --test epistemic_prob
cargo test -p xlog-logic --lib
cargo test -p xlog-solve --lib
cargo test -p xlog-prob --lib
cargo check -p xlog-logic -p xlog-ir -p xlog-solve -p xlog-prob
cargo check -p pyxlog
```

These commands validate the CPU-side semantic oracle only. They are not a
substitute for the required GPU-native certification evidence, which must include
GPU launch counts, kernel timings, WCOJ dispatch evidence, and zero CPU fallback
counters. The v0.8 pyxlog/DTS compatibility subset still must be rerun after the
v0.8 branch lands and this branch is rebased or merged onto it.

## Roadmap Status

This guide does not mark v0.9.0 roadmap rows DONE. ROADMAP and release-board
state are closure artifacts and should be updated only after the v0.8 rebase,
full certification, and coordinator approval.
