# v0.3.2 Feature Showcase Examples

This directory contains comprehensive real-world examples demonstrating all v0.3.2 language features:

- **Symbols**: String values with efficient interning (`symbol` type)
- **User-Defined Functions**: Reusable calculations (`func name(args) = expr.`)
- **Module System**: Code organization with imports and visibility

## Examples

| Directory | Domain | Features Highlighted |
|-----------|--------|---------------------|
| `01-enterprise/` | HR & Compensation | Symbols for names, UDFs for salary/tax, org hierarchy |
| `02-knowledge-graph/` | Movie Database | Symbols for entities, UDFs for scoring, type inference |
| `03-game-analytics/` | Gaming Platform | Symbols for players, UDFs for ELO, leaderboards |
| `04-supply-chain/` | Logistics Network | Symbols for SKUs, UDFs for costs, route optimization |

## Running Examples

```bash
# Run any example
cargo run -p xlog-cli -- run examples/xlog/80-v032-showcase/01-enterprise/main.xlog

# Run with statistics
cargo run -p xlog-cli -- run --stats examples/xlog/80-v032-showcase/01-enterprise/main.xlog
```

## Feature Coverage

Each example demonstrates:

| Feature | Description | Example |
|---------|-------------|---------|
| `symbol` type | Human-readable string values | `employee(e001, "Alice Chen", eng)` |
| `func` (arithmetic) | Mathematical calculations | `func bonus(Salary) = Salary * 20 / 100.` |
| `func` (conditional) | Branching logic | `func tier(X) = if X > 100 then 1 else 2.` |
| `use` imports | Module dependencies | `use finance/compensation.` |
| `private` predicate | Encapsulation | `private pred helper(u32).` |
| Recursion | Self-referential rules | `reach(X, Z) :- reach(X, Y), edge(Y, Z).` |
| Aggregation | count, sum, min, max | `total(Dept, sum(Salary)) :- ...` |
| Negation | Stratified negation | `best(X) :- option(X), not better(X).` |

## Data Volumes

| Domain | Facts | Predicates | UDFs |
|--------|-------|------------|------|
| Enterprise | ~200 | 15 | 7 |
| Knowledge Graph | ~300 | 20 | 7 |
| Game Analytics | ~500 | 18 | 6 |
| Supply Chain | ~400 | 16 | 8 |

## Architecture

Each domain follows a consistent module structure:

```
<domain>/
├── main.xlog           # Entry point with queries
├── README.md           # Domain documentation
└── <module>/
    └── <component>.xlog  # Feature modules
```

Modules are imported via `use <module>/<component>.` syntax and can reference
each other's public predicates. Private predicates (marked with `private`)
are only visible within their defining module.
