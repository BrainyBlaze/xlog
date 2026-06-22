# Supply Chain Example

Manufacturing supply chain analytics demonstrating XLOG's capabilities for
bill-of-materials processing, inventory management, and supplier analysis.

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
| **symbol type** | Product identifiers, supplier names, warehouse locations |
| **Recursive rules** | Bill-of-materials expansion for nested components |
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
// Bill-of-materials expansion (recursive)
pred bom_exploded(symbol, symbol, u32).
bom_exploded(Product, Component, Quantity) :- bom(Product, Component, Quantity).

pred bom_explosion_recursive(symbol, symbol, u32).
bom_explosion_recursive(Product, SubComponent, TotalQuantity) :-
    bom(Product, Component, ParentQuantity),
    bom_exploded(Component, SubComponent, ChildQuantity),
    TotalQuantity is ParentQuantity * ChildQuantity.

// Inventory analytics
pred warehouse_inventory_value(symbol, u64).
warehouse_inventory_value(Warehouse, sum(Value)) :-
    inventory(Warehouse, Product, Quantity),
    unit_cost(Product, Cost),
    Value is Quantity * Cost.

// Low stock alerts
pred low_stock_alert(symbol, symbol, symbol, u32, u32).
low_stock_alert(WarehouseName, ProductName, Category, CurrentQuantity, ReorderPoint) :-
    warehouse(Warehouse, WarehouseName, _),
    inventory(Warehouse, Product, CurrentQuantity),
    product(Product, ProductName, Category),
    reorder_point(Product, ReorderPoint),
    CurrentQuantity < ReorderPoint.
```

## Queries

1. **Bill-of-materials expansion**: Full component tree for Laptop X1
2. **Low stock alerts**: Products below reorder point
3. **Premium suppliers**: High-rated suppliers (rating >= 85)
4. **Single-source products**: Supply chain risk (only 1 supplier)
5. **Large orders**: Orders with value > $5000
6. **Major warehouses**: Well-stocked warehouses with inventory value
7. **Top customers**: Customers with 2+ orders

## Running

From this example directory:

```bash
cargo run -p xlog-cli -- run main.xlog
```

## Sample Output

```
__xlog_query_0 (Bill-of-materials expansion for Laptop X1)
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
- Nested bill-of-materials structure (Laptop -> Motherboard -> components)

## Use Cases

This example demonstrates patterns applicable to:

- **Manufacturing**: Bill-of-materials expansion for production planning
- **Retail**: Inventory management and reorder automation
- **Logistics**: Warehouse optimization and routing
- **Procurement**: Supplier risk analysis and sourcing decisions
