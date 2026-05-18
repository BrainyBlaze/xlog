# XLOG Example Programs

This repo includes (and is growing) a curated suite of `.xlog` programs intended to:

- Demonstrate XLOG’s supported language features end-to-end
- Provide realistic, domain-based “problem solving” demos (not toy-only)
- Act as regression fixtures for compilation + execution behavior

## How To Run

XLOG ships a production CLI (`xlog`) for deterministic and probabilistic execution. For deterministic
examples:

```bash
xlog run examples/xlog/00-basics/01_tc_reachability.xlog
xlog run --input edge=data.arrow examples/xlog/00-basics/01_tc_reachability.xlog
```

The workspace also provides a small runner for examples:

```bash
cargo run -p xlog-integration --release --bin xlog_run -- examples/xlog/00-basics/01_tc_reachability.xlog
```

By default the runner:

- Compiles and executes the program on CUDA device `0`
- Loads all facts declared in the `.xlog` source into the `Executor` store
- Enforces `:- ... .` constraints (fails if any constraint body has a solution)
- Prints results for all `?- ... .` queries in the file (if present)

Notes:
- The runner requires a CUDA-capable GPU and will execute on CUDA device `0` by default.
- Use `--device N`, `--memory-mb MB`, and `--limit N` to control execution and output volume.

## Example Suite Layout

Examples live under `examples/xlog/`, grouped by intent:

- `00-basics/`: minimal, "learn the syntax" programs
- `10-arithmetic/`: `is` expressions and numeric reasoning patterns
- `15-float-predicates/`: IEEE 754 float support with total ordering (v0.3.1)
- `20-graphs/`: graph reachability and social-network style queries
- `30-security/`: RBAC and policy derivations
- `40-supply-chain/`: BOM explosion and dependency reasoning
- `50-program-analysis/`: points-to and call graph construction
- `60-database-style/`: realistic multi-way join patterns
- `70-aggregates/`: aggregation queries (count/sum/min/max/logsumexp)
- `80-v032-showcase/`: comprehensive v0.3.2 feature demonstrations (symbol type, recursion, aggregations)
- `90-negative-tests/`: programs expected to fail (stratification/type errors)

Phase 4 examples live under:
- `examples/prob/`: probabilistic `.xlog` programs (prob facts, AD, evidence/query, and `prob_engine=mc`)
- `examples/python/`: Python scripts exercising `pyxlog` via DLPack (Torch optional)

Neural-symbolic examples, introduced during the `v0.4.0-alpha` milestone and carried
forward in the current `v0.5.x` release line, live under:
- `examples/neural/`: Neural-symbolic training examples

v0.8.0 DTS-DLM productization examples live under:
- `examples/v080-dts/`: Small certification-friendly showcase examples for
  async/streaming runtime controls, relation deltas, neural bridge helpers,
  native exact induction, and probabilistic async diagnostics.

## Example Index

**00-basics/**
- `examples/xlog/00-basics/01_tc_reachability.xlog`: transitive closure reachability
- `examples/xlog/00-basics/02_stratified_isolated.xlog`: stratified negation + wildcards
- `examples/xlog/00-basics/03_constraints_acyclic.xlog`: constraint enforcement (`:-`)
- `examples/xlog/00-basics/04_comparisons_and_equality.xlog`: comparisons (`=`, `<=`, `>=`)

**10-arithmetic/**
- `examples/xlog/10-arithmetic/01_arithmetic_demo.xlog`: `is` + `abs()` + `cast()`
- `examples/xlog/10-arithmetic/02_builtins_and_precedence.xlog`: full operator/builtin coverage (`+ - * / %`, `abs/min/max/pow/cast`)

**15-float-predicates/** (IEEE 754 float support - v0.3.1)
- `examples/xlog/15-float-predicates/01_nan_handling.xlog`: NaN detection and filtering
- `examples/xlog/15-float-predicates/02_infinity_detection.xlog`: infinity handling
- `examples/xlog/15-float-predicates/03_signed_zero.xlog`: signed zero (+0/-0) distinction
- `examples/xlog/15-float-predicates/04_data_quality_pipeline.xlog`: data quality checks with floats
- `examples/xlog/15-float-predicates/05_statistical_analysis.xlog`: statistical aggregations with floats

**20-graphs/**
- `examples/xlog/20-graphs/01_triangle_detection.xlog`: triangle detection + inequality constraints
- `examples/xlog/20-graphs/02_social_network_recommendations.xlog`: friends-of-friends + negation-based recommendation
- `examples/xlog/20-graphs/03_network_connectivity.xlog`: reachability + “no internet” isolation via negation

**30-security/**
- `examples/xlog/30-security/01_rbac_permissions.xlog`: RBAC role hierarchy + derived permissions
- `examples/xlog/30-security/02_rbac_separation_of_duties.xlog`: RBAC + separation-of-duties constraint

**40-supply-chain/**
- `examples/xlog/40-supply-chain/01_bom_explosion_leaf_parts.xlog`: BOM explosion + leaf part detection
- `examples/xlog/40-supply-chain/02_bom_quantities_and_totals.xlog`: recursion + arithmetic + `sum()`

**50-program-analysis/**
- `examples/xlog/50-program-analysis/01_points_to_copy.xlog`: points-to via copy propagation
- `examples/xlog/50-program-analysis/02_call_graph.xlog`: call graph + transitive reachability

**60-database-style/**
- `examples/xlog/60-database-style/01_local_orders_join.xlog`: multi-way join on shared key
- `examples/xlog/60-database-style/02_star_schema_sales_agg.xlog`: star-schema join + filter + aggregation

**70-aggregates/**
- `examples/xlog/70-aggregates/01_out_degree_count.xlog`: `count()`
- `examples/xlog/70-aggregates/02_multi_key_sum.xlog`: multi-key `sum()` (returns `u64`)
- `examples/xlog/70-aggregates/03_logsumexp.xlog`: `logsumexp()` for `f64`
- `examples/xlog/70-aggregates/04_min_max_latency_stats.xlog`: `min()` + `max()`
- `examples/xlog/70-aggregates/05_average_from_sum_count.xlog`: derived average via `sum`/`count` + `cast`

**80-v032-showcase/** (v0.3.2 feature demonstrations)
- `examples/xlog/80-v032-showcase/01-enterprise/`: HR analytics with org hierarchy, salary calculations
- `examples/xlog/80-v032-showcase/02-knowledge-graph/`: Scientific ontology with type inheritance, citations
- `examples/xlog/80-v032-showcase/03-game-analytics/`: Gaming platform with player stats, achievements, guilds
- `examples/xlog/80-v032-showcase/04-supply-chain/`: Manufacturing with BOM explosion, inventory management

**90-negative-tests/** (expected failures)
- `examples/xlog/90-negative-tests/01_constraint_cycle_violation.xlog`: constraint violation at runtime
- `examples/xlog/90-negative-tests/02_stratification_negation_cycle.xlog`: unstratifiable negation cycle (compile-time)
- `examples/xlog/90-negative-tests/03_arithmetic_type_mismatch.xlog`: missing `cast()` type mismatch (compile-time)
- `examples/xlog/90-negative-tests/04_is_target_already_bound.xlog`: invalid `is` target (compile-time)

## Neural-Symbolic Examples (Introduced In `v0.4.0-alpha`, Available In Current `v0.5.x` Releases)

Neural-symbolic training examples demonstrating integration where neural network
outputs become probabilistic facts in logic programs.

**neural/01_minimal/** — MNIST Addition
- `train.py`: Train a CNN to classify MNIST digits using only addition supervision
- Demonstrates: neural predicates (`nn/4`), network registration, tensor sources, gradient flow

To run:
```bash
cd examples/neural/01_minimal
python train.py --epochs 10 --batch-size 32
```

The network learns digit classification purely from sum labels — no individual digit labels provided!

`scripts/validate_examples.py` smoke-runs the neural examples with deterministic
fixture data when optional external datasets are absent. Set
`XLOG_VALIDATE_NEURAL_FULL=1` to require the real datasets during validation.

---

## Feature Coverage Goals

Each directory is designed to collectively cover:

- Facts and rules
- Multi-way joins, self-joins, and inequality constraints (`!=`)
- Recursion (fixpoint) patterns (TC, influence, reachability)
- Stratified negation (`not`) including common “isolation/absence” patterns
- Comparisons (`=`, `!=`, `<`, `<=`, `>`, `>=`)
- Wildcards (`_`) in rule bodies
- Arithmetic via `is` (`+ - * / %`), parentheses, and built-ins (`abs/min/max/pow/cast`)
- Aggregation in rule heads (`count/sum/min/max` and `logsumexp` for `f64`)
- Queries (`?- ... .`) as “what to print”
- Constraints (`:- ... .`) as “must hold” invariants

## Current Notes / Limitations

XLOG is still evolving; the example suite prefers correctness and clarity over exotic features.
When a language feature is syntactically accepted but not yet executed end-to-end, examples are
kept in `70-aggregates/` or `90-negative-tests/` and clearly marked.

Additional runner/runtime notes:
- `symbol` values are stored as `u32` IDs with a bidirectional string table; as of v0.3.2, symbols are **reversible** and display as readable strings in query output.
- Some aggregate implementations have value-type restrictions (documented in `docs/ARCHITECTURE.md`).
