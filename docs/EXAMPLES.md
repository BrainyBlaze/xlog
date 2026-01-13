# XLOG Example Programs

This repo includes (and is growing) a curated suite of `.xlog` programs intended to:

- Demonstrate XLOG’s supported language features end-to-end
- Provide realistic, domain-based “problem solving” demos (not toy-only)
- Act as regression fixtures for compilation + execution behavior

## How To Run

XLOG does not yet ship a full interactive CLI/REPL, but the workspace provides a small runner for
examples:

```bash
cargo run -p xlog-logic --release --example xlog_run -- examples/xlog/00-basics/01_tc_reachability.xlog
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

- `00-basics/`: minimal, “learn the syntax” programs
- `10-arithmetic/`: `is` expressions and numeric reasoning patterns
- `20-graphs/`: graph reachability and social-network style queries
- `30-security/`: RBAC and policy derivations
- `40-supply-chain/`: BOM explosion and dependency reasoning
- `50-program-analysis/`: points-to and call graph construction
- `60-database-style/`: realistic multi-way join patterns
- `70-aggregates/`: aggregation queries (count/sum/min/max/logsumexp)
- `90-negative-tests/`: programs expected to fail (stratification/type errors)

## Example Index

**00-basics/**
- `examples/xlog/00-basics/01_tc_reachability.xlog`: transitive closure reachability
- `examples/xlog/00-basics/02_stratified_isolated.xlog`: stratified negation + wildcards
- `examples/xlog/00-basics/03_constraints_acyclic.xlog`: constraint enforcement (`:-`)
- `examples/xlog/00-basics/04_comparisons_and_equality.xlog`: comparisons (`=`, `<=`, `>=`)

**10-arithmetic/**
- `examples/xlog/10-arithmetic/01_arithmetic_demo.xlog`: `is` + `abs()` + `cast()`
- `examples/xlog/10-arithmetic/02_builtins_and_precedence.xlog`: full operator/builtin coverage (`+ - * / %`, `abs/min/max/pow/cast`)

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

**90-negative-tests/** (expected failures)
- `examples/xlog/90-negative-tests/01_constraint_cycle_violation.xlog`: constraint violation at runtime
- `examples/xlog/90-negative-tests/02_stratification_negation_cycle.xlog`: unstratifiable negation cycle (compile-time)
- `examples/xlog/90-negative-tests/03_arithmetic_type_mismatch.xlog`: missing `cast()` type mismatch (compile-time)
- `examples/xlog/90-negative-tests/04_is_target_already_bound.xlog`: invalid `is` target (compile-time)

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
- `symbol` values are currently stored as a `u32` hash (MVP); outputs print the hashed value.
- Some aggregate implementations currently have value-type restrictions (documented in `docs/ARCHITECTURE.md`).
