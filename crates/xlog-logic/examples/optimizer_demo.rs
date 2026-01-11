//! Real-world optimizer demonstration for XLOG
//!
//! This example demonstrates the query optimizer's capabilities with realistic
//! scenarios from various domains:
//!
//! 1. **Social Network Analysis**: Friend-of-friend queries, influence propagation
//! 2. **Supply Chain Optimization**: Multi-hop supplier relationships, cost accumulation
//! 3. **Graph Analytics**: Path finding, connected components, cycle detection
//! 4. **Business Intelligence**: Multi-way joins, aggregations, complex filtering
//! 5. **Recursive Query Patterns**: Bill of materials, organizational hierarchies
//!
//! Run with: `cargo run --example optimizer_demo`

use std::sync::Arc;
use xlog_core::{RelId, ScalarType};
use xlog_ir::{CompareOp, ConstValue, Expr, JoinType, ProjectExpr, RirNode};
use xlog_logic::{Compiler, Optimizer, OptimizerConfig, PlanCost};
use xlog_stats::{ColumnStats, StatsManager};

/// Helper to create a horizontal line for output formatting
fn separator() {
    println!("{}", "=".repeat(80));
}

/// Helper to format large numbers with commas
fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

/// Display cost comparison between original and optimized plans
fn display_cost_comparison(name: &str, before: &PlanCost, after: &PlanCost) {
    println!("\n  {} Cost Analysis:", name);
    println!("  {:-<60}", "");
    println!(
        "  {:20} {:>15} {:>15} {:>10}",
        "Metric", "Before", "After", "Reduction"
    );
    println!("  {:-<60}", "");

    let row_reduction = if before.rows > 0 {
        100.0 - (after.rows as f64 / before.rows as f64 * 100.0)
    } else {
        0.0
    };
    println!(
        "  {:20} {:>15} {:>15} {:>9.1}%",
        "Estimated Rows",
        format_number(before.rows),
        format_number(after.rows),
        row_reduction
    );

    let cpu_reduction = if before.cpu_cost > 0.0 {
        100.0 - (after.cpu_cost / before.cpu_cost * 100.0)
    } else {
        0.0
    };
    println!(
        "  {:20} {:>15.1} {:>15.1} {:>9.1}%",
        "CPU Cost", before.cpu_cost, after.cpu_cost, cpu_reduction
    );

    let mem_reduction = if before.gpu_mem > 0 {
        100.0 - (after.gpu_mem as f64 / before.gpu_mem as f64 * 100.0)
    } else {
        0.0
    };
    println!(
        "  {:20} {:>15} {:>15} {:>9.1}%",
        "GPU Memory (bytes)",
        format_number(before.gpu_mem),
        format_number(after.gpu_mem),
        mem_reduction
    );

    println!(
        "  {:20} {:>15} {:>15}",
        "Data Transfers", before.transfers, after.transfers
    );

    let before_total = before.total_cost(100.0);
    let after_total = after.total_cost(100.0);
    let total_reduction = if before_total > 0.0 {
        100.0 - (after_total / before_total * 100.0)
    } else {
        0.0
    };
    println!("  {:-<60}", "");
    println!(
        "  {:20} {:>15.1} {:>15.1} {:>9.1}%",
        "Total Cost", before_total, after_total, total_reduction
    );
}

// =============================================================================
// Scenario 1: Social Network Analysis
// =============================================================================

/// Demonstrates optimizer handling of social network queries including:
/// - Friend-of-friend (transitive closure)
/// - Influence propagation
/// - Community detection patterns
fn demo_social_network_analysis() {
    separator();
    println!("SCENARIO 1: SOCIAL NETWORK ANALYSIS");
    separator();

    let mut compiler = Compiler::new();

    // Social network program with multiple derived relations
    let source = r#"
        // Base relations representing the social graph
        // user(id, name, influence_score)
        // follows(follower, followee)
        // posts(user_id, post_id, timestamp)
        // likes(user_id, post_id)

        // Sample data to establish schema
        user(1, "alice", 95).
        user(2, "bob", 75).
        user(3, "carol", 60).
        user(4, "dave", 45).
        user(5, "eve", 30).

        follows(1, 2).
        follows(2, 3).
        follows(3, 4).
        follows(4, 5).
        follows(2, 1).

        // ============================================
        // Friend-of-friend (two-hop connections)
        // ============================================
        friend_of_friend(X, Z) :- follows(X, Y), follows(Y, Z), X != Z.

        // ============================================
        // Transitive reachability (influence chain)
        // ============================================
        can_influence(X, Y) :- follows(X, Y).
        can_influence(X, Z) :- can_influence(X, Y), follows(Y, Z).

        // ============================================
        // Mutual follows (bidirectional friendship)
        // ============================================
        mutual_friends(X, Y) :- follows(X, Y), follows(Y, X), X < Y.

        // ============================================
        // Influence aggregation per user
        // ============================================
        follower_count(X, count(Y)) :- follows(Y, X).

        // ============================================
        // Popular users (high follower count threshold)
        // This uses stratified negation
        // ============================================
        has_followers(X) :- follows(Y, X).
        unpopular(X) :- user(X, N, S), not has_followers(X).
    "#;

    println!("\nCompiling social network analysis program...");
    let plan = match compiler.compile(source) {
        Ok(p) => p,
        Err(e) => {
            println!("  Compilation error: {:?}", e);
            return;
        }
    };

    println!("  Compiled {} SCCs, {} strata", plan.sccs.len(), plan.strata.len());
    println!("  Recursive: {}", plan.has_recursion());

    // Create realistic statistics for a medium-sized social network
    let mut stats_mgr = StatsManager::new();

    // Register relations with realistic cardinalities
    let rel_configs: Vec<(&str, u64, Vec<(usize, u64, i64, i64)>)> = vec![
        ("user", 1_000_000, vec![(0, 1_000_000, 1, 1_000_000), (2, 100, 0, 100)]),
        ("follows", 50_000_000, vec![(0, 1_000_000, 1, 1_000_000), (1, 1_000_000, 1, 1_000_000)]),
        ("posts", 100_000_000, vec![(0, 1_000_000, 1, 1_000_000)]),
        ("likes", 500_000_000, vec![(0, 1_000_000, 1, 1_000_000)]),
    ];

    for (name, cardinality, columns) in &rel_configs {
        if let Some(rel_id) = compiler.rel_ids().get(*name) {
            stats_mgr.register_relation(*rel_id);
            stats_mgr.update_cardinality(*rel_id, *cardinality);
            stats_mgr.update_byte_size(*rel_id, cardinality * 24);

            for (col_idx, distinct, min, max) in columns {
                let mut col_stats = ColumnStats::new(*col_idx, ScalarType::I64);
                col_stats.update_distinct(*distinct);
                col_stats.update_range(*min, *max);
                stats_mgr.add_column_stats(*rel_id, col_stats);
            }
        }
    }

    // Register join selectivity for follows self-join (friend-of-friend)
    if let Some(&follows_id) = compiler.rel_ids().get("follows") {
        // Record selectivity for follows self-join
        // Observed selectivity from historical queries
        stats_mgr.record_join_result(
            follows_id,
            follows_id,
            vec![1],
            vec![0],
            50_000_000 * 50_000_000 / 1000, // sampled product
            250_000_000,                     // typical result size
        );
    }

    let stats = Arc::new(stats_mgr);
    let optimizer = Optimizer::new(stats);

    println!("\n  Analyzing query costs:");

    // Analyze each compiled rule
    for scc_rules in &plan.rules_by_scc {
        for rule in scc_rules {
            let before_cost = optimizer.estimate_cost(&rule.body);
            let optimized_body = optimizer.optimize(rule.body.clone());
            let after_cost = optimizer.estimate_cost(&optimized_body);

            display_cost_comparison(&format!("{} rule", rule.head), &before_cost, &after_cost);
        }
    }

    // Test predicate pushdown specifically for friend-of-friend pattern
    println!("\n  Demonstrating predicate pushdown for social queries:");
    if let Some(&follows_id) = compiler.rel_ids().get("follows") {
        // Simulate: Filter(user=1000, Join(follows, follows))
        // Predicate should be pushed to the left side of the join
        let filter_on_join = RirNode::Filter {
            input: Box::new(RirNode::Join {
                left: Box::new(RirNode::Scan { rel: follows_id }),
                right: Box::new(RirNode::Scan { rel: follows_id }),
                left_keys: vec![1],
                right_keys: vec![0],
                join_type: JoinType::Inner,
            }),
            predicate: Expr::Compare {
                left: Box::new(Expr::Column(0)),
                op: CompareOp::Eq,
                right: Box::new(Expr::Const(ConstValue::I64(1000))),
            },
        };

        let before = optimizer.estimate_cost(&filter_on_join);
        let optimized = optimizer.optimize(filter_on_join);
        let after = optimizer.estimate_cost(&optimized);

        display_cost_comparison("Predicate Pushdown", &before, &after);

        // Verify predicate was pushed down
        match &optimized {
            RirNode::Join { left, .. } => {
                if matches!(left.as_ref(), RirNode::Filter { .. }) {
                    println!("\n  SUCCESS: Filter predicate pushed into join's left child");
                }
            }
            _ => {}
        }
    }
}

// =============================================================================
// Scenario 2: Supply Chain Optimization
// =============================================================================

/// Demonstrates optimizer handling of supply chain queries including:
/// - Multi-hop supplier relationships
/// - Part dependency tracking
/// - Cost accumulation queries
/// - Complex joins between inventory, orders, suppliers
fn demo_supply_chain_optimization() {
    separator();
    println!("\nSCENARIO 2: SUPPLY CHAIN OPTIMIZATION");
    separator();

    let mut compiler = Compiler::new();

    let source = r#"
        // Supply chain schema
        // supplier(id, name, country, reliability_score)
        // part(id, name, category, unit_cost)
        // part_supplier(part_id, supplier_id, lead_time, price)
        // inventory(warehouse_id, part_id, quantity, reorder_point)
        // orders(order_id, part_id, quantity, due_date)
        // assembly(parent_part, child_part, quantity_needed)

        // Sample data
        supplier(1, "acme", "usa", 95).
        supplier(2, "globex", "china", 85).
        supplier(3, "initech", "germany", 90).

        part(100, "widget", "electronic", 10).
        part(101, "gadget", "mechanical", 25).
        part(102, "gizmo", "assembly", 50).

        part_supplier(100, 1, 7, 10).
        part_supplier(100, 2, 14, 8).
        part_supplier(101, 1, 5, 25).

        inventory(1, 100, 500, 100).
        inventory(1, 101, 200, 50).

        orders(1, 100, 50, 20250101).

        assembly(102, 100, 3).
        assembly(102, 101, 2).

        // ============================================
        // Multi-hop part dependencies (Bill of Materials)
        // Recursive: finds all parts needed for assembly
        // ============================================
        needs_part(Parent, Child, Qty) :- assembly(Parent, Child, Qty).
        needs_part(Parent, Grandchild, Qty2) :-
            needs_part(Parent, Child, Qty1),
            assembly(Child, Grandchild, Qty2).

        // ============================================
        // Supplier chain analysis
        // Find all suppliers that can provide parts for an assembly
        // ============================================
        can_supply_part(Supplier, Part) :- part_supplier(Part, Supplier, L, P).
        can_supply_assembly(Supplier, Assembly) :-
            needs_part(Assembly, Part, Q),
            can_supply_part(Supplier, Part).

        // ============================================
        // Inventory shortage detection
        // Parts where inventory < reorder_point
        // ============================================
        below_reorder(Warehouse, Part, Qty, ReorderPt) :-
            inventory(Warehouse, Part, Qty, ReorderPt),
            Qty < ReorderPt.

        // ============================================
        // Order fulfillment check
        // Match orders against inventory
        // ============================================
        can_fulfill(OrderId, PartId, Available) :-
            orders(OrderId, PartId, Needed, Due),
            inventory(Wh, PartId, Available, Rp),
            Available >= Needed.

        // ============================================
        // Supplier reliability aggregation
        // ============================================
        avg_lead_time(Supplier, count(Part)) :- part_supplier(Part, Supplier, Lead, Price).
    "#;

    println!("\nCompiling supply chain optimization program...");
    let plan = match compiler.compile(source) {
        Ok(p) => p,
        Err(e) => {
            println!("  Compilation error: {:?}", e);
            return;
        }
    };

    println!("  Compiled {} SCCs", plan.sccs.len());
    println!("  Recursive SCCs: {}", plan.recursive_scc_count());

    // Create realistic statistics for enterprise supply chain
    let mut stats_mgr = StatsManager::new();

    let rel_configs = vec![
        ("supplier", 10_000u64),
        ("part", 500_000),
        ("part_supplier", 2_000_000),
        ("inventory", 5_000_000),
        ("orders", 10_000_000),
        ("assembly", 1_000_000),
    ];

    for (name, cardinality) in &rel_configs {
        if let Some(rel_id) = compiler.rel_ids().get(*name) {
            stats_mgr.register_relation(*rel_id);
            stats_mgr.update_cardinality(*rel_id, *cardinality);
            stats_mgr.update_byte_size(*rel_id, cardinality * 32);

            // Add column stats for join selectivity
            let mut col0 = ColumnStats::new(0, ScalarType::I64);
            col0.update_distinct((*cardinality as f64 * 0.8) as u64);
            stats_mgr.add_column_stats(*rel_id, col0);
        }
    }

    // Record join selectivities based on typical supply chain queries
    if let (Some(&ps_id), Some(&inv_id)) = (
        compiler.rel_ids().get("part_supplier"),
        compiler.rel_ids().get("inventory"),
    ) {
        // part_supplier JOIN inventory on part_id
        stats_mgr.record_join_result(
            ps_id,
            inv_id,
            vec![0],
            vec![1],
            2_000_000 * 5_000_000 / 10000,
            8_000_000,
        );
    }

    let stats = Arc::new(stats_mgr);

    // Test with different optimizer configurations
    let configs = vec![
        ("Default", OptimizerConfig::default()),
        (
            "Aggressive Pushdown",
            OptimizerConfig {
                enable_pushdown: true,
                default_filter_selectivity: 0.05,
                ..Default::default()
            },
        ),
        (
            "Memory Optimized",
            OptimizerConfig {
                default_bytes_per_row: 64,
                transfer_cost_multiplier: 200.0,
                ..Default::default()
            },
        ),
    ];

    for (config_name, config) in configs {
        println!("\n  Configuration: {}", config_name);
        let optimizer = Optimizer::with_config(Arc::clone(&stats), config);

        let mut total_cost = PlanCost::default();
        for scc_rules in &plan.rules_by_scc {
            for rule in scc_rules {
                let cost = optimizer.estimate_cost(&rule.body);
                total_cost.rows += cost.rows;
                total_cost.cpu_cost += cost.cpu_cost;
                total_cost.gpu_mem = total_cost.gpu_mem.max(cost.gpu_mem);
                total_cost.transfers += cost.transfers;
            }
        }

        println!(
            "    Total estimated rows: {}",
            format_number(total_cost.rows)
        );
        println!("    Total CPU cost: {:.1}", total_cost.cpu_cost);
        println!(
            "    Peak GPU memory: {} bytes",
            format_number(total_cost.gpu_mem)
        );
    }

    // Demonstrate join ordering impact on multi-way join
    println!("\n  Multi-way Join Analysis (Orders-Inventory-PartSupplier):");

    if let (Some(&orders_id), Some(&inv_id), Some(&ps_id)) = (
        compiler.rel_ids().get("orders"),
        compiler.rel_ids().get("inventory"),
        compiler.rel_ids().get("part_supplier"),
    ) {
        let stats_ref = Arc::clone(&stats);
        let optimizer = Optimizer::new(stats_ref);

        // Plan A: (Orders JOIN Inventory) JOIN PartSupplier
        let plan_a = RirNode::Join {
            left: Box::new(RirNode::Join {
                left: Box::new(RirNode::Scan { rel: orders_id }),
                right: Box::new(RirNode::Scan { rel: inv_id }),
                left_keys: vec![1],
                right_keys: vec![1],
                join_type: JoinType::Inner,
            }),
            right: Box::new(RirNode::Scan { rel: ps_id }),
            left_keys: vec![1],
            right_keys: vec![0],
            join_type: JoinType::Inner,
        };

        // Plan B: Orders JOIN (Inventory JOIN PartSupplier)
        let plan_b = RirNode::Join {
            left: Box::new(RirNode::Scan { rel: orders_id }),
            right: Box::new(RirNode::Join {
                left: Box::new(RirNode::Scan { rel: inv_id }),
                right: Box::new(RirNode::Scan { rel: ps_id }),
                left_keys: vec![1],
                right_keys: vec![0],
                join_type: JoinType::Inner,
            }),
            left_keys: vec![1],
            right_keys: vec![0],
            join_type: JoinType::Inner,
        };

        let cost_a = optimizer.estimate_cost(&plan_a);
        let cost_b = optimizer.estimate_cost(&plan_b);

        println!(
            "    Plan A [(Orders JOIN Inventory) JOIN PartSupplier]:"
        );
        println!(
            "      Rows: {}, CPU: {:.1}, Memory: {}",
            format_number(cost_a.rows),
            cost_a.cpu_cost,
            format_number(cost_a.gpu_mem)
        );

        println!(
            "    Plan B [Orders JOIN (Inventory JOIN PartSupplier)]:"
        );
        println!(
            "      Rows: {}, CPU: {:.1}, Memory: {}",
            format_number(cost_b.rows),
            cost_b.cpu_cost,
            format_number(cost_b.gpu_mem)
        );

        let better = if cost_a.total_cost(100.0) < cost_b.total_cost(100.0) {
            "Plan A"
        } else {
            "Plan B"
        };
        println!("    Recommended: {}", better);
    }
}

// =============================================================================
// Scenario 3: Graph Analytics
// =============================================================================

/// Demonstrates optimizer handling of graph algorithms including:
/// - Shortest path patterns
/// - Connected components
/// - Cycle detection
/// - Join ordering optimization
fn demo_graph_analytics() {
    separator();
    println!("\nSCENARIO 3: GRAPH ANALYTICS");
    separator();

    let mut compiler = Compiler::new();

    let source = r#"
        // Graph schema
        // vertex(id, label, weight)
        // edge(src, dst, weight)

        vertex(1, "a", 10).
        vertex(2, "b", 20).
        vertex(3, "c", 30).
        vertex(4, "d", 40).
        vertex(5, "e", 50).

        edge(1, 2, 5).
        edge(2, 3, 3).
        edge(3, 4, 7).
        edge(4, 5, 2).
        edge(1, 3, 10).
        edge(3, 1, 10).

        // ============================================
        // Reachability (transitive closure)
        // ============================================
        reachable(X, Y) :- edge(X, Y, W).
        reachable(X, Z) :- reachable(X, Y), edge(Y, Z, W).

        // ============================================
        // Connected component seeds (bidirectional edges)
        // ============================================
        undirected(X, Y) :- edge(X, Y, W).
        undirected(X, Y) :- edge(Y, X, W).

        same_component(X, Y) :- undirected(X, Y).
        same_component(X, Z) :- same_component(X, Y), undirected(Y, Z).

        // ============================================
        // Cycle detection
        // A node is in a cycle if it can reach itself
        // ============================================
        in_cycle(X) :- reachable(X, X).

        // ============================================
        // Source/Sink detection using negation
        // ============================================
        has_incoming(X) :- edge(Y, X, W).
        has_outgoing(X) :- edge(X, Y, W).
        source_node(X) :- vertex(X, L, W), not has_incoming(X).
        sink_node(X) :- vertex(X, L, W), not has_outgoing(X).

        // ============================================
        // Degree computation
        // ============================================
        out_degree(X, count(Y)) :- edge(X, Y, W).
        in_degree(X, count(Y)) :- edge(Y, X, W).

        // ============================================
        // Triangle detection pattern
        // ============================================
        triangle(A, B, C) :-
            edge(A, B, W1),
            edge(B, C, W2),
            edge(C, A, W3),
            A < B, B < C.
    "#;

    println!("\nCompiling graph analytics program...");
    let plan = match compiler.compile(source) {
        Ok(p) => p,
        Err(e) => {
            println!("  Compilation error: {:?}", e);
            return;
        }
    };

    println!("  Compiled {} SCCs", plan.sccs.len());
    println!("  Contains recursive predicates: {}", plan.has_recursion());

    // Create statistics for a large graph (e.g., web graph)
    let mut stats_mgr = StatsManager::new();

    // Large web-scale graph statistics
    let vertex_count: u64 = 10_000_000;
    let edge_count: u64 = 100_000_000;

    if let Some(&vertex_id) = compiler.rel_ids().get("vertex") {
        stats_mgr.register_relation(vertex_id);
        stats_mgr.update_cardinality(vertex_id, vertex_count);
        stats_mgr.update_byte_size(vertex_id, vertex_count * 24);

        let mut col0 = ColumnStats::new(0, ScalarType::I64);
        col0.update_distinct(vertex_count);
        col0.update_range(1, vertex_count as i64);
        stats_mgr.add_column_stats(vertex_id, col0);
    }

    if let Some(&edge_id) = compiler.rel_ids().get("edge") {
        stats_mgr.register_relation(edge_id);
        stats_mgr.update_cardinality(edge_id, edge_count);
        stats_mgr.update_byte_size(edge_id, edge_count * 24);

        // Source column - many edges per source
        let mut col0 = ColumnStats::new(0, ScalarType::I64);
        col0.update_distinct(vertex_count);
        stats_mgr.add_column_stats(edge_id, col0.clone());

        // Destination column
        let mut col1 = ColumnStats::new(1, ScalarType::I64);
        col1.update_distinct(vertex_count);
        stats_mgr.add_column_stats(edge_id, col1);
    }

    // Record join selectivity for edge self-join (reachability)
    if let Some(&edge_id) = compiler.rel_ids().get("edge") {
        stats_mgr.record_join_result(
            edge_id,
            edge_id,
            vec![1], // dst
            vec![0], // src
            edge_count * edge_count / 100000,
            edge_count * 10, // average path length ~10
        );
    }

    let stats = Arc::new(stats_mgr);
    let optimizer = Optimizer::new(stats);

    println!("\n  Fixpoint (Recursive) Cost Analysis:");

    // Find recursive rules and analyze fixpoint cost
    for scc_rules in &plan.rules_by_scc {
        for rule in scc_rules {
            // Check if this rule body contains a Fixpoint node
            fn contains_fixpoint(node: &RirNode) -> bool {
                match node {
                    RirNode::Fixpoint { .. } => true,
                    RirNode::Filter { input, .. } => contains_fixpoint(input),
                    RirNode::Project { input, .. } => contains_fixpoint(input),
                    RirNode::Join { left, right, .. } => {
                        contains_fixpoint(left) || contains_fixpoint(right)
                    }
                    RirNode::Union { inputs } => inputs.iter().any(contains_fixpoint),
                    RirNode::GroupBy { input, .. } => contains_fixpoint(input),
                    RirNode::Distinct { input, .. } => contains_fixpoint(input),
                    RirNode::Diff { left, right } => {
                        contains_fixpoint(left) || contains_fixpoint(right)
                    }
                    RirNode::Scan { .. } => false,
                }
            }

            let cost = optimizer.estimate_cost(&rule.body);
            let is_recursive = contains_fixpoint(&rule.body);

            println!(
                "    {} [{}]: rows={}, cpu={:.1}, gpu_mem={}",
                rule.head,
                if is_recursive { "recursive" } else { "non-recursive" },
                format_number(cost.rows),
                cost.cpu_cost,
                format_number(cost.gpu_mem)
            );
        }
    }

    // Demonstrate triangle detection query optimization
    println!("\n  Triangle Detection Join Ordering:");

    if let Some(&edge_id) = compiler.rel_ids().get("edge") {
        // Three edge scans for triangle pattern
        let e1 = RirNode::Scan { rel: edge_id };
        let e2 = RirNode::Scan { rel: edge_id };
        let e3 = RirNode::Scan { rel: edge_id };

        // Plan 1: ((E1 JOIN E2) JOIN E3)
        let plan1 = RirNode::Join {
            left: Box::new(RirNode::Join {
                left: Box::new(e1.clone()),
                right: Box::new(e2.clone()),
                left_keys: vec![1], // E1.dst = E2.src
                right_keys: vec![0],
                join_type: JoinType::Inner,
            }),
            right: Box::new(e3.clone()),
            left_keys: vec![2], // E2.dst = E3.src (mapped column)
            right_keys: vec![0],
            join_type: JoinType::Inner,
        };

        // Plan 2: (E1 JOIN (E2 JOIN E3))
        let plan2 = RirNode::Join {
            left: Box::new(e1.clone()),
            right: Box::new(RirNode::Join {
                left: Box::new(e2.clone()),
                right: Box::new(e3.clone()),
                left_keys: vec![1],
                right_keys: vec![0],
                join_type: JoinType::Inner,
            }),
            left_keys: vec![1],
            right_keys: vec![0],
            join_type: JoinType::Inner,
        };

        let cost1 = optimizer.estimate_cost(&plan1);
        let cost2 = optimizer.estimate_cost(&plan2);

        println!(
            "    Left-deep ((E1 JOIN E2) JOIN E3):"
        );
        println!(
            "      Total cost: {:.1}",
            cost1.total_cost(100.0)
        );

        println!(
            "    Bushy (E1 JOIN (E2 JOIN E3)):"
        );
        println!(
            "      Total cost: {:.1}",
            cost2.total_cost(100.0)
        );

        // Check if greedy should be used for this query
        let test_plan = plan1.clone();
        println!(
            "\n    Algorithm recommendation: {}",
            if optimizer.should_use_greedy(&test_plan) {
                "Greedy (exceeds DP threshold)"
            } else {
                "Dynamic Programming (within DP threshold)"
            }
        );
    }
}

// =============================================================================
// Scenario 4: Business Intelligence Queries
// =============================================================================

/// Demonstrates optimizer handling of BI queries including:
/// - Multi-way joins (5+ relations)
/// - Aggregation with grouping
/// - Filtering with complex predicates
/// - Predicate pushdown through projections
fn demo_business_intelligence() {
    separator();
    println!("\nSCENARIO 4: BUSINESS INTELLIGENCE QUERIES");
    separator();

    let mut compiler = Compiler::new();

    let source = r#"
        // Data warehouse schema
        // fact_sales(sale_id, product_id, customer_id, store_id, date_id, quantity, amount)
        // dim_product(product_id, name, category, brand, price)
        // dim_customer(customer_id, name, segment, region, join_date)
        // dim_store(store_id, name, city, state, country)
        // dim_date(date_id, year, month, day, quarter, is_holiday)

        // Sample data for schema establishment
        fact_sales(1, 100, 1000, 10, 20240101, 5, 500).
        dim_product(100, "laptop", "electronics", "acme", 999).
        dim_customer(1000, "john", "premium", "west", 20200101).
        dim_store(10, "downtown", "seattle", "wa", "usa").
        dim_date(20240101, 2024, 1, 1, 1, 0).

        // ============================================
        // Sales by product category
        // ============================================
        sales_by_category(Category, sum(Amount)) :-
            fact_sales(S, P, C, St, D, Q, Amount),
            dim_product(P, N, Category, B, Pr).

        // ============================================
        // Customer segment analysis
        // ============================================
        segment_revenue(Segment, sum(Amount)) :-
            fact_sales(S, P, C, St, D, Q, Amount),
            dim_customer(C, N, Segment, R, J).

        // ============================================
        // Regional sales performance
        // ============================================
        regional_sales(Region, sum(Amount)) :-
            fact_sales(S, P, C, St, D, Q, Amount),
            dim_customer(C, N, Sg, Region, J).

        // ============================================
        // Holiday vs non-holiday sales
        // ============================================
        holiday_sales(IsHoliday, sum(Amount)) :-
            fact_sales(S, P, C, St, D, Q, Amount),
            dim_date(D, Y, M, Day, Qtr, IsHoliday).

        // ============================================
        // Multi-dimensional analysis
        // Category x Region x Quarter
        // ============================================
        cube_analysis(Category, Region, Quarter, sum(Amount)) :-
            fact_sales(S, P, C, St, D, Q, Amount),
            dim_product(P, N, Category, B, Pr),
            dim_customer(C, Cn, Sg, Region, J),
            dim_date(D, Y, M, Day, Quarter, H).

        // ============================================
        // Top customer detection (using aggregation)
        // ============================================
        customer_total(CustomerId, sum(Amount)) :-
            fact_sales(S, P, CustomerId, St, D, Q, Amount).

        // ============================================
        // Product affinity (frequently bought together)
        // ============================================
        same_customer_products(P1, P2) :-
            fact_sales(S1, P1, C, St1, D1, Q1, A1),
            fact_sales(S2, P2, C, St2, D2, Q2, A2),
            P1 < P2.
    "#;

    println!("\nCompiling business intelligence program...");
    let plan = match compiler.compile(source) {
        Ok(p) => p,
        Err(e) => {
            println!("  Compilation error: {:?}", e);
            return;
        }
    };

    println!("  Compiled {} SCCs", plan.sccs.len());
    println!("  Total rules: {}", plan.rules_by_scc.iter().map(|r| r.len()).sum::<usize>());

    // Create realistic data warehouse statistics
    let mut stats_mgr = StatsManager::new();

    let fact_rows: u64 = 1_000_000_000; // 1B fact rows
    let rel_configs = vec![
        ("fact_sales", fact_rows, 7),
        ("dim_product", 100_000, 5),
        ("dim_customer", 10_000_000, 5),
        ("dim_store", 5_000, 5),
        ("dim_date", 10_000, 6), // ~27 years of dates
    ];

    for (name, cardinality, num_cols) in &rel_configs {
        if let Some(rel_id) = compiler.rel_ids().get(*name) {
            stats_mgr.register_relation(*rel_id);
            stats_mgr.update_cardinality(*rel_id, *cardinality);
            stats_mgr.update_byte_size(*rel_id, cardinality * (*num_cols as u64) * 8);

            // Add distinct counts for join columns
            let distinct = if *name == "fact_sales" {
                // Fact table FK columns have lower distinct than dimension PKs
                match num_cols {
                    _ => *cardinality / 100,
                }
            } else {
                *cardinality // Dimension table PKs are fully distinct
            };

            let mut col0 = ColumnStats::new(0, ScalarType::I64);
            col0.update_distinct(distinct);
            stats_mgr.add_column_stats(*rel_id, col0);
        }
    }

    // Record FK-PK join selectivities
    if let (Some(&fact_id), Some(&product_id)) = (
        compiler.rel_ids().get("fact_sales"),
        compiler.rel_ids().get("dim_product"),
    ) {
        // Fact JOIN Product on product_id
        stats_mgr.record_join_result(
            fact_id,
            product_id,
            vec![1],
            vec![0],
            fact_rows * 100_000 / 100000,
            fact_rows, // FK-PK join preserves fact rows
        );
    }

    let stats = Arc::new(stats_mgr);
    let optimizer = Optimizer::new(stats);

    println!("\n  Query Cost Analysis:");
    println!("  (Data warehouse: {} fact rows, 4 dimension tables)", format_number(fact_rows));

    for scc_rules in &plan.rules_by_scc {
        for rule in scc_rules {
            let cost = optimizer.estimate_cost(&rule.body);
            let optimized = optimizer.optimize(rule.body.clone());
            let opt_cost = optimizer.estimate_cost(&optimized);

            let improvement = if cost.total_cost(100.0) > 0.0 {
                (1.0 - opt_cost.total_cost(100.0) / cost.total_cost(100.0)) * 100.0
            } else {
                0.0
            };

            println!(
                "\n    {}:",
                rule.head
            );
            println!(
                "      Original: rows={}, cpu={:.1}",
                format_number(cost.rows),
                cost.cpu_cost
            );
            println!(
                "      Optimized: rows={}, cpu={:.1} ({:.1}% improvement)",
                format_number(opt_cost.rows),
                opt_cost.cpu_cost,
                improvement
            );
        }
    }

    // Demonstrate predicate pushdown through projection
    println!("\n  Predicate Pushdown Through Projection:");

    if let Some(&fact_id) = compiler.rel_ids().get("fact_sales") {
        // Query: SELECT product_id, amount FROM fact_sales WHERE quantity > 10
        // Without pushdown: Project(Filter(Scan))
        // With pushdown: Project(Filter(Scan)) - filter stays close to scan

        let scan = RirNode::Scan { rel: fact_id };

        // Pattern: Filter on top of Project on top of Scan
        // Filter references column 0 which maps to column 1 in original (product_id)
        let filter_over_project = RirNode::Filter {
            input: Box::new(RirNode::Project {
                input: Box::new(scan.clone()),
                columns: vec![
                    ProjectExpr::Column(1), // product_id
                    ProjectExpr::Column(6), // amount
                    ProjectExpr::Column(5), // quantity (for filter)
                ],
            }),
            predicate: Expr::Compare {
                left: Box::new(Expr::Column(2)), // quantity in projected output
                op: CompareOp::Gt,
                right: Box::new(Expr::Const(ConstValue::I64(10))),
            },
        };

        let before_cost = optimizer.estimate_cost(&filter_over_project);
        let optimized = optimizer.optimize(filter_over_project);
        let after_cost = optimizer.estimate_cost(&optimized);

        display_cost_comparison("Filter Through Projection", &before_cost, &after_cost);
    }

    // 5-way join analysis
    println!("\n  5-Way Star Join Analysis (Fact + 4 Dimensions):");

    if let (
        Some(&fact_id),
        Some(&product_id),
        Some(&customer_id),
        Some(&store_id),
        Some(&date_id),
    ) = (
        compiler.rel_ids().get("fact_sales"),
        compiler.rel_ids().get("dim_product"),
        compiler.rel_ids().get("dim_customer"),
        compiler.rel_ids().get("dim_store"),
        compiler.rel_ids().get("dim_date"),
    ) {
        // Star schema join pattern
        let star_join = RirNode::Join {
            left: Box::new(RirNode::Join {
                left: Box::new(RirNode::Join {
                    left: Box::new(RirNode::Join {
                        left: Box::new(RirNode::Scan { rel: fact_id }),
                        right: Box::new(RirNode::Scan { rel: product_id }),
                        left_keys: vec![1],
                        right_keys: vec![0],
                        join_type: JoinType::Inner,
                    }),
                    right: Box::new(RirNode::Scan { rel: customer_id }),
                    left_keys: vec![2],
                    right_keys: vec![0],
                    join_type: JoinType::Inner,
                }),
                right: Box::new(RirNode::Scan { rel: store_id }),
                left_keys: vec![3],
                right_keys: vec![0],
                join_type: JoinType::Inner,
            }),
            right: Box::new(RirNode::Scan { rel: date_id }),
            left_keys: vec![4],
            right_keys: vec![0],
            join_type: JoinType::Inner,
        };

        let cost = optimizer.estimate_cost(&star_join);
        println!(
            "    Estimated rows: {}",
            format_number(cost.rows)
        );
        println!("    CPU cost: {:.1}", cost.cpu_cost);
        println!(
            "    GPU memory: {} bytes",
            format_number(cost.gpu_mem)
        );
        println!(
            "    Should use greedy algorithm: {}",
            optimizer.should_use_greedy(&star_join)
        );
    }
}

// =============================================================================
// Scenario 5: Recursive Query Patterns
// =============================================================================

/// Demonstrates optimizer handling of recursive patterns including:
/// - Bill of materials explosion
/// - Organizational hierarchy traversal
/// - Fixpoint cost estimation
fn demo_recursive_patterns() {
    separator();
    println!("\nSCENARIO 5: RECURSIVE QUERY PATTERNS");
    separator();

    let mut compiler = Compiler::new();

    let source = r#"
        // Bill of Materials schema
        // component(id, name, type, unit_cost)
        // composition(parent_id, child_id, quantity)
        //
        // Organization schema
        // employee(id, name, title, salary)
        // reports_to(employee_id, manager_id)

        // Sample BOM data
        component(1, "car", "final", 25000).
        component(2, "engine", "assembly", 5000).
        component(3, "chassis", "assembly", 3000).
        component(4, "piston", "part", 100).
        component(5, "cylinder", "part", 200).
        component(6, "steel_frame", "part", 500).

        composition(1, 2, 1).
        composition(1, 3, 1).
        composition(2, 4, 8).
        composition(2, 5, 4).
        composition(3, 6, 1).

        // Sample org data
        employee(1, "ceo", "chief", 500000).
        employee(2, "vp_eng", "vp", 300000).
        employee(3, "vp_sales", "vp", 280000).
        employee(4, "eng_mgr", "manager", 180000).
        employee(5, "dev1", "engineer", 120000).
        employee(6, "dev2", "engineer", 115000).

        reports_to(2, 1).
        reports_to(3, 1).
        reports_to(4, 2).
        reports_to(5, 4).
        reports_to(6, 4).

        // ============================================
        // Bill of Materials Explosion
        // Find all parts needed for a product (recursive)
        // ============================================
        bom_explode(Parent, Child, Qty) :- composition(Parent, Child, Qty).
        bom_explode(Parent, Grandchild, Qty2) :-
            bom_explode(Parent, Child, Qty1),
            composition(Child, Grandchild, Qty2).

        // ============================================
        // Part identification
        // ============================================
        part_cost(PartId, Cost) :-
            component(PartId, N, "part", Cost).

        // ============================================
        // Assembly parts list
        // ============================================
        assembly_parts(ParentId, ChildId, Qty) :-
            bom_explode(ParentId, ChildId, Qty),
            part_cost(ChildId, UnitCost).

        // ============================================
        // Organizational Hierarchy
        // Find all reports (direct and indirect)
        // ============================================
        manages(Manager, Employee) :- reports_to(Employee, Manager).
        manages(Manager, Indirect) :-
            manages(Manager, Direct),
            reports_to(Indirect, Direct).

        // ============================================
        // Team size calculation
        // ============================================
        team_size(Manager, count(Employee)) :- manages(Manager, Employee).

        // ============================================
        // Management chain (path to CEO)
        // ============================================
        management_chain(Employee, Manager) :- reports_to(Employee, Manager).
        management_chain(Employee, TopMgr) :-
            management_chain(Employee, MidMgr),
            reports_to(MidMgr, TopMgr).

        // ============================================
        // Leaf employees (no direct reports)
        // ============================================
        has_reports(M) :- reports_to(E, M).
        individual_contributor(E) :- employee(E, N, T, S), not has_reports(E).
    "#;

    println!("\nCompiling recursive query program...");
    let plan = match compiler.compile(source) {
        Ok(p) => p,
        Err(e) => {
            println!("  Compilation error: {:?}", e);
            return;
        }
    };

    println!("  Compiled {} SCCs", plan.sccs.len());
    println!("  Recursive SCCs: {}", plan.recursive_scc_count());
    println!("  Strata (for negation): {}", plan.strata.len());

    // Create statistics for manufacturing company
    let mut stats_mgr = StatsManager::new();

    // BOM statistics
    let component_count: u64 = 100_000;
    let composition_count: u64 = 500_000; // avg 5 children per component
    let employee_count: u64 = 50_000;
    let reports_count: u64 = 49_999; // all but CEO report to someone

    let rel_configs = vec![
        ("component", component_count),
        ("composition", composition_count),
        ("employee", employee_count),
        ("reports_to", reports_count),
    ];

    for (name, cardinality) in &rel_configs {
        if let Some(rel_id) = compiler.rel_ids().get(*name) {
            stats_mgr.register_relation(*rel_id);
            stats_mgr.update_cardinality(*rel_id, *cardinality);
            stats_mgr.update_byte_size(*rel_id, cardinality * 32);

            let mut col0 = ColumnStats::new(0, ScalarType::I64);
            col0.update_distinct(*cardinality);
            stats_mgr.add_column_stats(*rel_id, col0);
        }
    }

    // Record recursive join selectivity (composition self-join for BOM)
    if let Some(&comp_id) = compiler.rel_ids().get("composition") {
        // BOM explosion typically expands by factor of depth
        // Average BOM depth ~5, so output is ~5x input
        stats_mgr.record_join_result(
            comp_id,
            comp_id,
            vec![1], // child
            vec![0], // parent
            composition_count * composition_count / 10000,
            composition_count * 5,
        );
    }

    let stats = Arc::new(stats_mgr);

    // Analyze with different iteration estimates
    let configs = vec![
        (
            "Default (log2 iterations)",
            OptimizerConfig::default(),
        ),
        (
            "Shallow hierarchy",
            OptimizerConfig {
                default_filter_selectivity: 0.2,
                ..Default::default()
            },
        ),
    ];

    for (config_name, config) in configs {
        println!("\n  Configuration: {}", config_name);
        let optimizer = Optimizer::with_config(Arc::clone(&stats), config);

        // Find and analyze recursive predicates
        let recursive_preds = vec!["bom_explode", "manages", "management_chain"];

        for pred in &recursive_preds {
            // Find rules for this predicate
            let rules: Vec<_> = plan
                .rules_by_scc
                .iter()
                .flatten()
                .filter(|r| &r.head == *pred)
                .collect();

            if !rules.is_empty() {
                let total_cost: f64 = rules
                    .iter()
                    .map(|r| optimizer.estimate_cost(&r.body).total_cost(100.0))
                    .sum();

                println!(
                    "    {}: {} rules, total cost = {:.1}",
                    pred,
                    rules.len(),
                    total_cost
                );
            }
        }
    }

    // Demonstrate fixpoint cost breakdown
    println!("\n  Fixpoint Cost Breakdown:");

    // Create a synthetic fixpoint node for analysis
    if let Some(&comp_id) = compiler.rel_ids().get("composition") {
        let base = RirNode::Scan { rel: comp_id };
        let recursive = RirNode::Join {
            left: Box::new(RirNode::Scan { rel: RelId(100) }), // delta placeholder
            right: Box::new(RirNode::Scan { rel: comp_id }),
            left_keys: vec![1],
            right_keys: vec![0],
            join_type: JoinType::Inner,
        };

        let fixpoint = RirNode::Fixpoint {
            scc_id: 0,
            base: Box::new(base.clone()),
            recursive: Box::new(recursive.clone()),
            delta_rel: RelId(100),
            full_rel: RelId(101),
        };

        let optimizer = Optimizer::new(Arc::clone(&stats));

        let base_cost = optimizer.estimate_cost(&base);
        let recursive_cost = optimizer.estimate_cost(&recursive);
        let fixpoint_cost = optimizer.estimate_cost(&fixpoint);

        println!("    Base case cost:");
        println!("      Rows: {}", format_number(base_cost.rows));
        println!("      CPU: {:.1}", base_cost.cpu_cost);

        println!("    Per-iteration cost:");
        println!("      Rows: {}", format_number(recursive_cost.rows));
        println!("      CPU: {:.1}", recursive_cost.cpu_cost);

        let estimated_iterations = ((base_cost.rows as f64).log2().ceil() as u64).max(1);
        println!("    Estimated iterations: {}", estimated_iterations);

        println!("    Total fixpoint cost:");
        println!("      Rows: {}", format_number(fixpoint_cost.rows));
        println!("      CPU: {:.1}", fixpoint_cost.cpu_cost);
        println!(
            "      GPU memory: {} bytes",
            format_number(fixpoint_cost.gpu_mem)
        );
    }

    // Show hot relation recommendations
    println!("\n  Index Recommendations:");
    let mut stats_mgr = StatsManager::new();

    // Simulate access patterns
    if let Some(&comp_id) = compiler.rel_ids().get("composition") {
        stats_mgr.register_relation(comp_id);
        for _ in 0..100 {
            stats_mgr.record_access(comp_id);
        }
    }

    if let Some(&reports_id) = compiler.rel_ids().get("reports_to") {
        stats_mgr.register_relation(reports_id);
        for _ in 0..50 {
            stats_mgr.record_access(reports_id);
        }
    }

    let config = OptimizerConfig {
        index_heat_threshold: 0.5,
        ..Default::default()
    };
    let optimizer = Optimizer::with_config(Arc::new(stats_mgr), config);
    let hot = optimizer.recommend_indexes();

    println!("    Relations recommended for indexing:");
    for rel_id in hot {
        // Map RelId back to name
        for (name, &id) in compiler.rel_ids() {
            if id == rel_id {
                println!("      - {} (RelId: {:?})", name, rel_id);
                break;
            }
        }
    }
}

// =============================================================================
// Summary and Recommendations
// =============================================================================

fn print_summary() {
    separator();
    println!("\nOPTIMIZER DEMONSTRATION SUMMARY");
    separator();

    println!(
        r#"
The XLOG query optimizer provides:

1. COST-BASED OPTIMIZATION
   - Row count estimation using relation cardinalities
   - CPU cost modeling for operators (scan, filter, join, aggregate)
   - GPU memory estimation for resource planning
   - Data transfer counting for GPU/CPU boundary optimization

2. PREDICATE PUSHDOWN
   - Pushes filters below projections when columns are pass-through
   - Pushes filters into join sides when referencing only that side
   - Merges consecutive filters into conjunctions
   - Preserves semantics while reducing intermediate result sizes

3. JOIN OPTIMIZATION
   - Uses cached selectivity from historical query execution
   - Estimates join cardinality from column statistics
   - Supports different join types (Inner, LeftOuter, Semi, Anti)
   - Adaptive algorithm selection (DP vs Greedy based on relation count)

4. RECURSIVE QUERY SUPPORT
   - Fixpoint cost estimation based on iteration estimates
   - Handles transitive closure patterns efficiently
   - Supports stratified negation in recursive contexts

5. STATISTICS INTEGRATION
   - Per-relation cardinality and byte size tracking
   - Column-level statistics (distinct counts, value ranges)
   - Join selectivity caching with exponential moving average updates
   - Access heat tracking for index recommendations

BEST PRACTICES:
- Register all relations with realistic cardinality estimates
- Add column statistics for frequently joined columns
- Record historical join results to improve selectivity estimates
- Use appropriate configuration for your workload characteristics
"#
    );
}

fn main() {
    println!("\n");
    println!("  XLOG QUERY OPTIMIZER DEMONSTRATION");
    println!("  Real-World Scenarios and Cost Analysis");
    println!("\n");

    // Run all demonstrations
    demo_social_network_analysis();
    demo_supply_chain_optimization();
    demo_graph_analytics();
    demo_business_intelligence();
    demo_recursive_patterns();

    print_summary();

    separator();
    println!("\nDemo complete. Run with: cargo run --example optimizer_demo");
    separator();
}
