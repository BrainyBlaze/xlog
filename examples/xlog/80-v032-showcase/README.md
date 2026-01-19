# v0.3.2 Feature Showcase Examples

This directory contains comprehensive real-world examples demonstrating all v0.3.2 language features:

- **Reversible Symbols**: String values with efficient interning (`symbol` type) that display as readable strings
- **User-Defined Functions**: Reusable calculations (`func name(args) = expr.`)
- **Module System**: Code organization with imports and visibility
- **Aggregations**: count (u64), sum, min, max with comparison support
- **Recursive Rules**: Transitive closure, management chains, type hierarchies

## Examples

| Directory | Domain | Key Features |
|-----------|--------|--------------|
| `01-enterprise/` | HR & Compensation | Employee tracking, salary calculations, recursive org hierarchy, management chains |
| `02-knowledge-graph/` | Scientific Ontology | Type inheritance, citation analysis, co-authorship networks, semantic inference |
| `03-game-analytics/` | Gaming Platform | Player stats, achievement chains, guild analytics, leaderboards, social graphs |
| `04-supply-chain/` | Logistics Network | BOM explosion, inventory management, supplier analytics, order tracking |

## Running Examples

```bash
# Run any example
cargo run --release -- run examples/xlog/80-v032-showcase/01-enterprise/main.xlog

# Run with statistics
cargo run --release -- run --stats examples/xlog/80-v032-showcase/01-enterprise/main.xlog

# Run all showcase examples
for dir in examples/xlog/80-v032-showcase/*/; do
    cargo run --release -- run "${dir}main.xlog"
done
```

## Feature Coverage Matrix

| Feature | Enterprise | Knowledge Graph | Game Analytics | Supply Chain |
|---------|:----------:|:---------------:|:--------------:|:------------:|
| `symbol` type | Names, IDs | Entities, Labels | Players, Items | Products, Suppliers |
| Recursive rules | Org hierarchy | Type inheritance | Achievement chains | BOM explosion |
| count aggregation | Direct reports | Citations | Match stats | Order lines |
| sum aggregation | Dept budgets | - | XP totals | Inventory value |
| Comparisons (>=, <) | Tenure checks | Depth filters | Score thresholds | Stock alerts |
| Arithmetic (`is`) | Salary calcs | - | KDA ratios | Cost calculations |

## Example Highlights

### Enterprise Analytics
```xlog
// Recursive management chain with depth tracking
management_chain(Emp, Mgr, 1) :- reports_to(Emp, Mgr).
management_chain(Emp, TopMgr, Level) :-
    reports_to(Emp, Mgr),
    management_chain(Mgr, TopMgr, PrevLevel),
    Level is PrevLevel + cast(1, u32).

// Aggregation + comparison for filtering
large_team_manager(Name, DeptId, Count) :-
    employee(MgrId, Name, DeptId),
    direct_report_count(MgrId, Count),
    Count >= 3.
```

### Knowledge Graph
```xlog
// Type hierarchy inference (transitive closure)
is_a(Child, Parent) :- subclass_of(Child, Parent).
is_a(Child, Ancestor) :-
    subclass_of(Child, Parent),
    is_a(Parent, Ancestor).

// Citation analysis with depth
cites_transitively(A, B, 1) :- cites(A, B).
cites_transitively(A, C, Depth) :-
    cites(A, B),
    cites_transitively(B, C, PrevDepth),
    Depth is PrevDepth + cast(1, u32).
```

### Game Analytics
```xlog
// Achievement prerequisite chains
all_prerequisites(AchId, PrereqId) :- achievement_requires(AchId, PrereqId).
all_prerequisites(AchId, TransPrereq) :-
    achievement_requires(AchId, DirectPrereq),
    all_prerequisites(DirectPrereq, TransPrereq).

// Guild power calculation
guild_power(GuildId, sum(Level)) :-
    guild_member(GuildId, PlayerId),
    player(PlayerId, _, Level).
```

### Supply Chain
```xlog
// Bill of Materials explosion (recursive)
bom_exploded(Product, Component, Qty) :- bom(Product, Component, Qty).
bom_explosion_recursive(Product, SubComponent, TotalQty) :-
    bom(Product, Component, ParentQty),
    bom_exploded(Component, SubComponent, ChildQty),
    TotalQty is ParentQty * ChildQty.

// Inventory alerts
low_stock_alert(WarehouseName, ProductName, Category, CurrentQty, ReorderPt) :-
    warehouse(WarehouseId, WarehouseName, _),
    inventory(WarehouseId, ProductId, CurrentQty),
    product(ProductId, ProductName, Category),
    reorder_point(ProductId, ReorderPt),
    CurrentQty < ReorderPt.
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
