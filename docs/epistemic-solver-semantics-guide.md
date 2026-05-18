# v0.9.0 Epistemic And Solver Semantics Guide

This guide describes the bounded v0.9.0 semantics implemented on
`feat/v090-epistemic-solver-semantics`. The current implementation is a
semantic fixture layer: it makes EIR, epistemic modes, solver services, and
probabilistic integration testable without routing epistemic programs through
the production RIR/runtime path.

## Current Boundary

Epistemic literals are parsed and represented explicitly. Direct lowering to RIR
still returns `UnsupportedEpistemicConstruct`, so `xlog run` is not the execution
path for these examples yet. Use the fixture tests listed below to run them.

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

The returned trace records generated, propagated, pruned, tested, accepted, and
rejected counts.

## Epistemic Splitting

`split_epistemic_program` builds deterministic components from source rules and
rejects a rule that couples more than one distinct epistemic body predicate. For
accepted split fixtures, `recomposed_rule_indices()` must recover the original
source rule order.

## Solver Services

`xlog_solve::SolverService` provides the bounded solver API used by v0.9.0
fixtures:

- `assume` and `retract_assumption` model incremental SAT assumptions;
- learned clauses are transferred through `transfer_learned_clauses_to` and
  counted in `SolverServiceTrace`;
- learned clauses derived under active assumptions remain scoped to those
  assumptions;
- `SolveInstance::with_weights` gives fixture-scale MaxSAT soft constraints;
- `SolverServiceStatus` distinguishes `Sat`, `Unsat`, `Unknown`, `Timeout`, and
  `Optimal`;
- GPU portfolio solving is explicitly deferred with rationale.

Run the solver service fixture:

```bash
cargo test -p xlog-solve --test solver_service_semantics
```

## Probabilistic Integration

`xlog_prob::epistemic` records the bounded probabilistic contract:

- epistemic assumptions become probabilistic evidence literals such as
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

Current pre-rebase certification snapshot:

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

The v0.8 pyxlog/DTS compatibility subset still must be rerun after the v0.8
branch lands and this branch is rebased or merged onto it.

## Roadmap Status

This guide does not mark v0.9.0 roadmap rows DONE. ROADMAP and release-board
state are closure artifacts and should be updated only after the v0.8 rebase,
full certification, and coordinator approval.
