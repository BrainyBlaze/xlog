# v0.9.0 G090_PRE Evidence

Date: 2026-05-18

Goal node: `G090_PRE - Baseline Inventory And Semantic Fixture Selection`

Branch: `feat/v090-epistemic-solver-semantics`

Worktree: `/home/dev/projects/xlog/.worktrees/v090-epistemic`

## Baseline State

| Check | Evidence |
|---|---|
| Base commit | `656a8c6232f4611caf6c571eb0bcf1282e9a7339` |
| Base subject | `docs(roadmap): focus v080 on dts ml python productization` |
| Starting status | `git status --short --branch` returned only `## feat/v090-epistemic-solver-semantics` |
| Worktree isolation | Project-local `.worktrees/` exists and is ignored by `.gitignore:27` |
| Dispatch docs | v0.9.0 and v0.8.0 goal docs were read from the coordinator checkout at `/home/dev/projects/xlog/docs/plans/2026-05-18-agent-*.md`; those files were untracked in the coordinator checkout at dispatch time and therefore were not present in this new worktree. |

## Sources Read

| Source | Use |
|---|---|
| `/home/dev/projects/xlog/docs/plans/2026-05-18-agent-v090-epistemic-solver-goal.md` | Governing v0.9.0 GQM goal document. |
| `/home/dev/projects/xlog/docs/plans/2026-05-18-agent-v080-dts-ml-python-goal.md` | v0.8.0 coordination lock and compatibility rerun source. |
| `ROADMAP.md` | v0.9.0 roadmap rows and cross-version risks. |
| `docs/architecture/solver-services.md` | Existing SAT/CDCL, GPU verifier, and solver-service contract. |
| `docs/architecture/xlog-prob.md` | Existing probabilistic, circuit, GPU D4, and solver-verifier integration. |
| `docs/architecture/python-bindings.md` | v0.8-owned pyxlog compatibility surface. |
| `docs/architecture/bounded-exact-induction.md` | v0.8-owned native exact-induction consumer surface. |

## Ownership Map

| Area | Current owners | Current state | v0.9.0 responsibility |
|---|---|---|---|
| Parser and syntax | `crates/xlog-logic/src/grammar.pest`, `crates/xlog-logic/src/parser.rs` | Pest grammar covers Datalog, negation, pragmas, probabilistic facts, evidence, queries, neural declarations, and learnable rules. | Add epistemic syntax and parser diagnostics without changing non-epistemic defaults. |
| AST and semantic frontend | `crates/xlog-logic/src/ast.rs`, `crates/xlog-logic/src/stratify.rs`, `crates/xlog-logic/src/compile.rs` | AST represents Datalog, constraints, probabilistic directives, and WFS/probabilistic stratification hooks. | Add explicit epistemic AST/EIR entry points, mode selection, G91/FAEEL separation, and typed unsupported-construct diagnostics. |
| Relational IR boundary | `crates/xlog-ir/src/rir.rs`, `crates/xlog-ir/src/plan.rs`, `crates/xlog-logic/src/lower.rs` | RIR covers scans, joins, anti-joins, filters, projections, fixpoints, WCOJ metadata, and lowering from AST. | Define EIR-to-RIR/probability/solver lowering boundaries instead of hiding epistemic rewrites in RIR. |
| Generate-Propagate-Test execution | `crates/xlog-logic/src/compile.rs`, `crates/xlog-runtime/src/executor/recursive.rs`, `crates/xlog-prob/src/mc.rs` | Existing pipelines have compile/lower, recursive fixpoint, and MC sampling phases, but no epistemic candidate pipeline. | Add visible generate, propagate, and test phases with traceable phase counts and explosion guards. |
| Epistemic splitting | `crates/xlog-logic/src/stratify.rs`, `crates/xlog-logic/src/hypergraph/`, `crates/xlog-ir/src/plan.rs` | Dependency graphs and SCCs exist for stratification, recursive planning, and hypergraph optimization. | Add deterministic epistemic dependency graph, valid split/recomposition path, and typed invalid-split rejection. |
| Solver services | `crates/xlog-solve/src/lib.rs`, `crates/xlog-solve/src/instance.rs`, `crates/xlog-solve/src/solver.rs`, `crates/xlog-solve/src/proof.rs`, `crates/xlog-solve/src/gpu_cdcl.rs` | CPU CLS solver, complete GPU CDCL verifier, CNF instance/proof/result types, and GPU CNF interface exist. | Add explicit xlog-logic solver interface, SAT assumptions, learned-clause transfer trace/test doubles, MaxSAT soft constraints, and distinct UNSAT/UNKNOWN/TIMEOUT behavior. |
| Probabilistic integration | `crates/xlog-prob/src/provenance.rs`, `crates/xlog-prob/src/pir.rs`, `crates/xlog-prob/src/cnf.rs`, `crates/xlog-prob/src/compilation/`, `crates/xlog-prob/src/exact.rs`, `crates/xlog-prob/src/mc.rs` | Exact GPU D4/CDCL and MC/WFS paths exist with circuit cache and deterministic fixtures. | Define epistemic/probabilistic interaction, incremental circuit update fixtures, and at least one compiler-adapter path or design. |
| v0.8-owned compatibility | `crates/pyxlog/`, `crates/xlog-runtime/`, `crates/xlog-integration/tests/test_m37a_surface_preservation.rs`, `docs/architecture/python-bindings.md`, `docs/architecture/bounded-exact-induction.md` | v0.8 branch owns pyxlog runtime/session APIs, DTS certs, relation deltas, neural bridge, and native exact-induction consumer work. | Keep v0.9 changes default-off and rerun the compatibility subset after v0.8 lands and this branch rebases. |

## Semantic Fixture Inventory

These are selected golden fixture intents for G090 implementation. They are inventory entries, not executable tests yet; G090_EIR must turn them into parser/semantic fixtures with concrete syntax and typed diagnostics.

| Mode | Positive fixture intent | Negative fixture intent |
|---|---|---|
| EIR syntax and diagnostics | Program with one epistemic literal in a rule body lowers to an explicit EIR node rather than a plain predicate rewrite. | Nested or unsupported epistemic operator is rejected with a typed diagnostic identifying the construct and source span. |
| G91 compatibility | Same epistemic program evaluated with explicit `g91` mode preserves classic compatibility behavior for known literals. | Default mode is not silently changed by selecting G91; a fixture must prove at least one G91-only result differs from FAEEL and is isolated to compatibility mode. |
| FAEEL default | Founded equilibrium example has a stable founded model under default epistemic mode. | Self-supporting belief or contradictory candidate returns a typed no-model/diagnostic result, not a panic or silent fallback. |
| Generate-Propagate-Test | Candidate generation emits multiple candidates, propagation prunes at least one, and test accepts the expected survivor with trace counts. | Candidate explosion or unbounded recursive candidate generation trips a deterministic guard with a typed bounded-behavior diagnostic. |
| Epistemic splitting | Two independent epistemic components solve separately and recombine to the same result as the unsplit program. | Cross-component dependency that violates splitting preconditions is rejected with a typed invalid-split diagnostic. |
| Solver integration | xlog-logic constraint lowers to a solver request with assumptions, returns SAT/UNSAT distinctly, and records assumption scope. | Retraction of a missing assumption or solver timeout returns a distinct UNKNOWN/TIMEOUT/failure state rather than conflating it with UNSAT. |
| Probabilistic integration | Changing an epistemic assumption updates a probabilistic/circuit query within a documented tolerance and reuses supported cache state. | Probabilistic aggregate or compiler-adapter feature outside v0.9 scope is rejected or explicitly deferred with a documented typed reason. |

## v0.8 Compatibility Rerun List After Rebase

After v0.8.0 lands and this branch rebases or merges `main`, rerun at least:

| Gate | Command or source |
|---|---|
| Formatting | `cargo fmt --check` |
| v0.9 crate health | `cargo check -p xlog-logic -p xlog-ir -p xlog-solve -p xlog-prob` |
| pyxlog compatibility compile | `cargo check -p pyxlog` |
| xlog-logic parser/lowering baseline | `cargo test -p xlog-logic --lib` |
| xlog-ir baseline | `cargo test -p xlog-ir --lib` |
| xlog-solve baseline | `cargo test -p xlog-solve --lib` |
| xlog-prob baseline | `cargo test -p xlog-prob --lib` |
| v0.8 DTS surface preservation | `cargo test -p xlog-integration --test test_m37a_surface_preservation` |
| v0.8 runtime delta/recursive surface | `cargo test -p xlog-runtime --test test_w23_recursive_stats` plus the v0.8 branch's newly added delta certs. |
| v0.8 pyxlog API/cert pack | The v0.8 branch's committed DTS-DLM certification pack and pyxlog public-surface manifest. |

## Baseline Validation

| Command | Result |
|---|---|
| `git rev-parse HEAD` | `656a8c6232f4611caf6c571eb0bcf1282e9a7339` |
| `git status --short --branch` | clean: `## feat/v090-epistemic-solver-semantics` |
| `cargo fmt --check` | PASS |
| `cargo check -p xlog-logic -p xlog-ir -p xlog-solve -p xlog-prob` | PASS, finished dev profile in 24.85s |
| `cargo test -p xlog-logic --lib` | PASS, 236 passed, 0 failed |
| `cargo test -p xlog-ir --lib` | PASS, 14 passed, 0 failed |
| `cargo test -p xlog-solve --lib` | PASS, 111 passed, 0 failed |
| `cargo test -p xlog-prob --lib` | PASS, 56 passed, 0 failed |
| `cargo check -p pyxlog` | PASS, finished dev profile in 10.64s |

## Metric Status

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M090_PRE.1 branch base | base recorded and clean before implementation | PASS | Baseline state and validation tables above. |
| M090_PRE.2 ownership map | crate/file ownership table committed in evidence | PASS | Ownership map above. |
| M090_PRE.3 fixture inventory | at least one positive and one negative fixture for each semantic mode | PASS | Semantic fixture inventory above. |
| M090_PRE.4 compatibility list | v0.8-owned tests to rerun after rebase listed | PASS | Compatibility rerun list above. |

## Coordination Notes

- No semantic implementation has started.
- No v0.8-owned pyxlog/runtime files were edited.
- No push, tag, release-board update, or merge was performed.
- G090_EIR may start only after this evidence is committed and reported.
