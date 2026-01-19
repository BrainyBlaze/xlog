# Supply Chain Example

Manufacturing supply chain analytics demonstrating XLOG's capabilities for Bill of Materials (BOM) processing, inventory management, and supplier analysis.

## Domain Model

This example models a supply chain ecosystem with:

- **Products**: Electronics and components with hierarchical structure
- **Suppliers**: Global supplier network with ratings
- **Warehouses**: Regional distribution centers
- **Inventory**: Stock levels with reorder points
- **Orders**: Customer orders and shipments

## Features Demonstrated

| Feature | Usage |
|---------|-------|
| **symbol type** | Product IDs, supplier names, warehouse locations |
| **Recursive rules** | BOM explosion (nested components) |
| **count aggregation** | Products per warehouse, suppliers per product |
| **sum aggregation** | Inventory value, order totals |
| **Comparisons** | Low stock alerts, premium suppliers (rating >= 85) |
| **Arithmetic** | Cost calculations, quantity multiplication |

## Key Predicates

### Base Data
```xlog
pred product(symbol, symbol, symbol).        // id, name, category
pred bom(symbol, symbol, u32).               // parent, component, quantity
pred supplier(symbol, symbol, symbol).       // id, name, country
pred supplies(symbol, symbol, u32).          // supplier, product, price
pred inventory(symbol, symbol, u32).         // warehouse, product, quantity
pred order(symbol, symbol, symbol, u32).     // id, customer, status, timestamp
pred order_line(symbol, symbol, u32, u32).   // order, product, qty, price
```

### Derived Relations
```xlog
// BOM explosion (recursive)
pred bom_exploded(symbol, symbol, u32).
bom_exploded(Product, Component, Qty) :- bom(Product, Component, Qty).

pred bom_explosion_recursive(symbol, symbol, u32).
bom_explosion_recursive(Product, SubComponent, TotalQty) :-
    bom(Product, Component, ParentQty),
    bom_exploded(Component, SubComponent, ChildQty),
    TotalQty is ParentQty * ChildQty.

// Inventory analytics
pred warehouse_inventory_value(symbol, u64).
warehouse_inventory_value(WarehouseId, sum(Value)) :-
    inventory(WarehouseId, ProductId, Qty),
    unit_cost(ProductId, Cost),
    Value is Qty * Cost.

// Low stock alerts
pred low_stock_alert(symbol, symbol, symbol, u32, u32).
low_stock_alert(WarehouseName, ProductName, Category, CurrentQty, ReorderPt) :-
    warehouse(WarehouseId, WarehouseName, _),
    inventory(WarehouseId, ProductId, CurrentQty),
    product(ProductId, ProductName, Category),
    reorder_point(ProductId, ReorderPt),
    CurrentQty < ReorderPt.
```

## Queries

1. **BOM explosion**: Full component tree for Laptop X1
2. **Low stock alerts**: Products below reorder point
3. **Premium suppliers**: High-rated suppliers (rating >= 85)
4. **Single-source products**: Supply chain risk (only 1 supplier)
5. **Large orders**: Orders with value > $5000
6. **Major warehouses**: Well-stocked warehouses with inventory value
7. **Top customers**: Customers with 2+ orders

## Running

```bash
cargo run --release -- run examples/xlog/80-v032-showcase/04-supply-chain/main.xlog
```

## Sample Output

```
__xlog_query_0 (BOM explosion for Laptop X1)
+-------+-------+
| col_0 | col_1 |
+-------+-------+
| p002  | 1     |  (Display Panel)
| p003  | 1     |  (CPU Chip)
| p004  | 2     |  (RAM Module - direct)
| p004  | 4     |  (RAM Module - via Motherboard)
| p005  | 1     |  (SSD Drive)
| p006  | 1     |  (Battery Pack)
| p007  | 1     |  (Keyboard Assembly)
| p008  | 1     |  (Motherboard)
+-------+-------+

__xlog_query_2 (Premium suppliers)
+------------------+---------+-------+-------+
| col_0            | col_1   | col_2 | col_3 |
+------------------+---------+-------+-------+
| TechParts Inc    | usa     | s001  | 85    |
| EuroTech Supply  | germany | s003  | 92    |
| AsiaChip Co      | taiwan  | s004  | 88    |
| QualityParts Ltd | japan   | s005  | 95    |
+------------------+---------+-------+-------+

__xlog_query_5 (Major warehouses)
+----------------------+---------+-------+-------+----------+
| col_0                | col_1   | col_2 | col_3 | col_4    |
+----------------------+---------+-------+-------+----------+
| West Coast Hub       | west    | w001  | 7     | 21050000 |
| East Coast Hub       | east    | w002  | 7     | 20500000 |
| Central Distribution | central | w003  | 7     | 23250000 |
| Europe Center        | europe  | w004  | 6     | 9250000  |
+----------------------+---------+-------+-------+----------+
```

## Data Statistics

- 15 products (electronics, components, accessories)
- 6 suppliers across 5 countries
- 4 regional warehouses
- 5 customers in different segments
- 6 orders with multiple line items
- Nested BOM structure (Laptop → Motherboard → components)

## Use Cases

This example demonstrates patterns applicable to:

- **Manufacturing**: BOM explosion for production planning
- **Retail**: Inventory management and reorder automation
- **Logistics**: Warehouse optimization and routing
- **Procurement**: Supplier risk analysis and sourcing decisions
