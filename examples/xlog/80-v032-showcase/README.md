# Feature Showcase Examples

This directory contains comprehensive real-world examples demonstrating XLOG
language features:

- **Reversible Symbols**: String values with efficient interning (`symbol` type) that display as readable strings
- **User-Defined Functions**: Reusable calculations (`func name(args) = expr.`)
- **Module System**: Code organization with imports and visibility
- **Aggregations**: count (u64), sum, min, max with comparison support
- **Recursive Rules**: Transitive closure, management chains, type hierarchies

## Examples

| Directory | Domain | Key Features |
|-----------|--------|--------------|
| `01-enterprise/` | Human resources and compensation | Employee tracking, salary calculations, recursive organization hierarchy, management chains |
| `02-knowledge-graph/` | Movie ontology | Type inheritance, movie analytics, collaboration networks, semantic inference |
| `03-game-analytics/` | Gaming Platform | Player stats, achievement chains, guild analytics, leaderboards, social graphs |
| `04-supply-chain/` | Logistics Network | Bill-of-materials expansion, inventory management, supplier analytics, order tracking |

## Running Examples

```bash
# From this showcase directory, run any example.
cargo run -p xlog-cli -- run 01-enterprise/main.xlog

# Run with statistics
cargo run -p xlog-cli -- run --stats 01-enterprise/main.xlog

# Run all showcase examples
for dir in 01-enterprise 02-knowledge-graph 03-game-analytics 04-supply-chain; do
    cargo run -p xlog-cli -- run "$dir/main.xlog"
done
```

## Feature Coverage Matrix

| Feature | Enterprise | Knowledge Graph | Game Analytics | Supply Chain |
|---------|:----------:|:---------------:|:--------------:|:------------:|
| `symbol` type | Names, identifiers | Entities, labels | Players, items | Products, suppliers |
| Recursive rules | Organization hierarchy | Type inheritance | Achievement chains | Bill-of-materials expansion |
| count aggregation | Direct reports | Filmography counts | Match stats | Order lines |
| sum aggregation | Department budgets | Box-office totals | Experience-point totals | Inventory value |
| Comparisons (>=, <) | Tenure checks | Depth filters | Score thresholds | Stock alerts |
| Arithmetic (`is`) | Salary calculations | Return on investment | Kill/death/assist ratios | Cost calculations |

## Example Highlights

### Enterprise Analytics
```xlog
// Recursive management chain with depth tracking
management_chain(Employee, Manager, 1) :- reports_to(Employee, Manager).
management_chain(Employee, TopManager, Level) :-
    reports_to(Employee, Manager),
    management_chain(Manager, TopManager, PreviousLevel),
    Level is PreviousLevel + cast(1, u32).

// Aggregation + comparison for filtering
large_team_manager(Name, Department, Count) :-
    employee(Manager, Name, Department),
    direct_report_count(Manager, Count),
    Count >= 3.
```

### Knowledge Graph
```xlog
// Type hierarchy inference (transitive closure)
is_subclass(Child, Parent) :- subclass_of(Child, Parent).
is_subclass(Child, Ancestor) :-
    subclass_of(Child, Parent),
    is_subclass(Parent, Ancestor).

// Movie return on investment
movie_roi(Movie, Title, ReturnOnInvestment) :-
    movie(Movie, Title, _, _, Budget, BoxOffice),
    ReturnOnInvestment is roi_pct(BoxOffice, Budget).
```

### Game Analytics
```xlog
// Achievement prerequisite chains
all_prerequisites(Achievement, Prerequisite) :-
    achievement_requires(Achievement, Prerequisite).
all_prerequisites(Achievement, TransitivePrerequisite) :-
    achievement_requires(Achievement, DirectPrerequisite),
    all_prerequisites(DirectPrerequisite, TransitivePrerequisite).

// Guild power calculation
guild_power_ranking(GuildName, LeaderName, TotalEloRating, CompletedAchievementMembers) :-
    guild(Guild, GuildName, Leader),
    player(Leader, LeaderName, _, _),
    guild_total_elo_rating(Guild, TotalEloRating),
    guild_total_achievements(Guild, CompletedAchievementMembers).
```

### Supply Chain
```xlog
// Bill-of-materials expansion (recursive)
bom_exploded(Product, Component, Quantity) :- bom(Product, Component, Quantity).
bom_exploded(Product, SubComponent, TotalQuantity) :-
    bom(Product, Component, ParentQuantity),
    bom_exploded(Component, SubComponent, ChildQuantity),
    TotalQuantity is ParentQuantity * ChildQuantity.

// Inventory alerts
low_stock_alert(Warehouse, Product, CurrentQuantity, ReorderPoint) :-
    stock(Warehouse, Product, CurrentQuantity),
    reorder_point(Product, ReorderPoint),
    CurrentQuantity < ReorderPoint.
```

## Data Scale

| Domain | Facts | Predicates | Derived Relations | Queries |
|--------|-------|------------|-------------------|---------|
| Enterprise | ~150 | 12 | 10 | 7 |
| Knowledge Graph | ~200 | 18 | 15 | 7 |
| Game Analytics | ~350 | 25 | 20 | 7 |
| Supply Chain | ~250 | 20 | 18 | 7 |

## Validation

All examples are validated as part of the test suite:

```bash
# Run all examples and verify they execute without errors
cargo test --workspace
```

All 4 showcase examples execute successfully with meaningful query results.
