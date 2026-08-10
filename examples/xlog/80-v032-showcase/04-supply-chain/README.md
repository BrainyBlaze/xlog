# Supply Chain Example

Manufacturing supply chain analytics demonstrating XLOG's capabilities for
bill-of-materials processing, inventory management, and multi-carrier shipping logistics.

## Domain Model

This example models a supply chain ecosystem with:

- **Products**: Electronics, components, office supplies, furniture, and packaging with hierarchical bill-of-materials structure
- **Warehouses**: Regional distribution centers with per-product stock levels
- **Inventory**: Stock levels, reorder points, daily demand, and lead times
- **Shipping**: Carriers, shipping lanes, direct routes, and transit times between warehouses

## Features Demonstrated

| Feature | Usage |
|---------|-------|
| **symbol type** | Product identifiers, warehouse names, carrier names |
| **Recursive rules** | Bill-of-materials expansion for nested components; multi-hop shipping reachability |
| **count aggregation** | Routes per carrier, reachable destinations per warehouse |
| **sum aggregation** | Warehouse inventory value, total assembly cost |
| **Comparisons** | Low stock alerts, critical inventory urgency |
| **Arithmetic** | Cost calculations, shipping surcharges, days-of-stock |

## Key Predicates

### Base Data
```xlog
pred product(symbol, symbol, symbol).        // product_id, name, category
pred warehouse(symbol, symbol, symbol).      // warehouse_id, name, region
pred stock(symbol, symbol, u32).             // warehouse_id, product_id, quantity
pred unit_cost(symbol, u32).                 // product_id, cost_cents
pred reorder_point(symbol, u32).             // product_id, min_quantity
pred bom(symbol, symbol, u32).               // parent_product, component_product, quantity_needed
pred carrier(symbol, symbol, symbol).        // carrier_id, name, service_level
pred direct_route(symbol, symbol, symbol, u32, u32). // origin, destination, carrier_id, distance_km, base_cost_cents
```

### Derived Relations
```xlog
// Bill-of-materials expansion (recursive)
pred bom_exploded(symbol, symbol, u32).
bom_exploded(Product, Component, Quantity) :- bom(Product, Component, Quantity).
bom_exploded(Product, SubComponent, TotalQuantity) :-
    bom(Product, Component, ParentQuantity),
    bom_exploded(Component, SubComponent, ChildQuantity),
    TotalQuantity is ParentQuantity * ChildQuantity.

// Inventory analytics
pred warehouse_value(symbol, u64).
warehouse_value(Warehouse, sum(Value)) :-
    stock(Warehouse, Product, Quantity),
    unit_cost(Product, Cost),
    Value is Quantity * Cost.

// Low stock alerts
pred low_stock_alert(symbol, symbol, u32, u32).
low_stock_alert(Warehouse, Product, CurrentQuantity, ReorderPoint) :-
    stock(Warehouse, Product, CurrentQuantity),
    reorder_point(Product, ReorderPoint),
    CurrentQuantity < ReorderPoint.
```

## Queries

`main.xlog` cross-references the `inventory/stock`, `shipping/routes`, and `cost/calculator`
modules with 15 queries:

1. **Bill-of-materials expansion**: All components needed for the PowerTower Desktop (recursive)
2. **Critical inventory**: Items with urgency level 1 or 2 (under 7 days of stock)
3. **Best shipping option**: Lowest-cost carrier for each route
4. **Fastest shipping option**: Quickest carrier for each route
5. **Warehouse summary**: Total inventory value per warehouse
6. **Assembly cost breakdown**: Component costs for the ProBook Laptop 15
7. **Assembly total cost**: Total component cost per manufactured product
8. **Shipping reachability**: Destinations reachable from Seattle Distribution Center (recursive)
9. **Warehouse connectivity**: Count of reachable destinations per warehouse
10. **Volume discount tiers**: Bulk order discount percentages by quantity
11. **Low stock by category**: Products below reorder point, grouped by category
12. **Carrier coverage**: Regions served per carrier and service level
13. **Multi-carrier routes**: Number of carriers available per route
14. **Category stock totals**: Total units in stock per product category
15. **Inventory days remaining**: Days of stock remaining per product per warehouse

## Running

From this example directory:

```bash
cargo run -p xlog-cli -- run main.xlog
```

## Data Statistics

- 55 products across 5 categories (electronics, components, office supplies, furniture, packaging)
- 8 regional warehouses
- 12 shipping carriers (national, regional, and freight)
- Nested bill-of-materials structure (PowerTower Desktop -> Motherboard -> CPU)

## Use Cases

This example demonstrates patterns applicable to:

- **Manufacturing**: Bill-of-materials expansion for production planning
- **Retail**: Inventory management and reorder automation
- **Logistics**: Multi-carrier route selection and warehouse connectivity
- **Procurement**: Reorder urgency based on lead time and daily demand
