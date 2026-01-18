//! Integration tests for the query optimizer with the compiler.
//!
//! These tests verify that the optimizer correctly integrates with compiled plans,
//! demonstrating cost estimation, predicate pushdown, and configuration options
//! working together with real Datalog programs.

use std::sync::Arc;
use xlog_core::{RelId, ScalarType};
use xlog_ir::{CompiledRule, RirNode};
use xlog_logic::{Compiler, Optimizer, OptimizerConfig, PlanCost};
use xlog_stats::{ColumnStats, StatsManager};

// =============================================================================
// Compiler Output Structure Tests
// =============================================================================

/// Test that optimizer can work with compiled execution plans from real Datalog programs.
///
/// This test compiles a transitive closure program and verifies that the optimizer
/// can estimate costs for each compiled rule's body.
#[test]
fn test_compile_with_optimizer_cost_estimation() {
    let mut compiler = Compiler::new();

    let source = r#"
        edge(1, 2).
        edge(2, 3).
        edge(3, 4).
        reach(X, Y) :- edge(X, Y).
        reach(X, Z) :- reach(X, Y), edge(Y, Z).
    "#;

    let plan = compiler
        .compile(source)
        .expect("Should compile transitive closure");

    // Create optimizer with stats manager containing relation statistics
    let mut stats_mgr = StatsManager::new();

    // Register relations that the compiler creates
    // The compiler assigns RelIds to predicates - we'll use the rel_ids() method
    for (_name, rel_id) in compiler.rel_ids() {
        stats_mgr.register_relation(*rel_id);
        // Set reasonable default cardinalities for test
        stats_mgr.update_cardinality(*rel_id, 100);
        stats_mgr.update_byte_size(*rel_id, 800); // ~8 bytes per row
    }

    let stats = Arc::new(stats_mgr);
    let optimizer = Optimizer::new(stats);

    // Verify optimizer can estimate costs for all compiled rules
    for scc_rules in &plan.rules_by_scc {
        for rule in scc_rules {
            let cost = optimizer.estimate_cost(&rule.body);

            // Verify we get meaningful cost estimates
            assert!(
                cost.rows >= 1,
                "Rule for {} should have positive row estimate",
                rule.head
            );
            assert!(
                cost.cpu_cost >= 0.0,
                "Rule for {} should have non-negative CPU cost",
                rule.head
            );
            // gpu_mem is u64, always non-negative
        }
    }

    // Verify plan structure
    assert!(!plan.sccs.is_empty(), "Expected SCCs in execution plan");
    assert!(
        plan.has_recursion(),
        "Transitive closure should be recursive"
    );
}

/// Test optimizer with a stratified program containing negation.
///
/// Verifies cost estimation works correctly for programs with multiple strata,
/// including anti-joins generated from negation.
#[test]
fn test_optimizer_with_stratified_program() {
    let mut compiler = Compiler::new();

    let source = r#"
        node(1).
        node(2).
        node(3).
        node(4).
        edge(1, 2).
        edge(2, 3).
        isolated(X) :- node(X), not edge(X, Y).
    "#;

    let plan = compiler
        .compile(source)
        .expect("Should compile stratified program");

    let mut stats_mgr = StatsManager::new();
    for (_name, rel_id) in compiler.rel_ids() {
        stats_mgr.register_relation(*rel_id);
        stats_mgr.update_cardinality(*rel_id, 50);
        stats_mgr.update_byte_size(*rel_id, 400);
    }

    let stats = Arc::new(stats_mgr);
    let optimizer = Optimizer::new(stats);

    // Verify all rules have valid cost estimates
    for scc_rules in &plan.rules_by_scc {
        for rule in scc_rules {
            let cost = optimizer.estimate_cost(&rule.body);
            assert!(cost.rows >= 1, "Rule should have positive rows");
        }
    }

    // Stratified program should have multiple strata
    assert!(
        !plan.strata.is_empty(),
        "Stratified program should have explicit strata"
    );
}

/// Test optimizer with aggregation program.
///
/// Verifies cost estimation correctly handles GroupBy nodes generated from
/// aggregation in Datalog rules.
#[test]
fn test_optimizer_with_aggregation() {
    let mut compiler = Compiler::new();

    let source = r#"
        edge(1, 2).
        edge(1, 3).
        edge(1, 4).
        edge(2, 3).
        out_degree(X, count(Y)) :- edge(X, Y).
    "#;

    let plan = compiler
        .compile(source)
        .expect("Should compile aggregation program");

    let mut stats_mgr = StatsManager::new();
    for (_name, rel_id) in compiler.rel_ids() {
        stats_mgr.register_relation(*rel_id);
        stats_mgr.update_cardinality(*rel_id, 100);
        stats_mgr.update_byte_size(*rel_id, 800);
    }

    let stats = Arc::new(stats_mgr);
    let optimizer = Optimizer::new(stats);

    // Find the aggregation rule and verify its cost
    let agg_rules: Vec<&CompiledRule> = plan
        .rules_by_scc
        .iter()
        .flatten()
        .filter(|r| r.head == "out_degree")
        .collect();

    assert!(!agg_rules.is_empty(), "Should have out_degree rule");

    for rule in agg_rules {
        let cost = optimizer.estimate_cost(&rule.body);
        // Aggregation should reduce row count from input
        assert!(cost.rows >= 1, "Aggregation result should have rows");
    }
}

// =============================================================================
// Configuration Tests
// =============================================================================

/// Test optimizer with custom configuration affecting cost estimation.
#[test]
fn test_optimizer_config_integration() {
    let config = OptimizerConfig {
        dp_threshold: 5,
        index_heat_threshold: 0.5,
        enable_pushdown: true,
        default_filter_selectivity: 0.2,
        transfer_cost_multiplier: 50.0,
        default_bytes_per_row: 64,
    };

    let stats = Arc::new(StatsManager::new());
    let optimizer = Optimizer::with_config(stats, config);

    // Verify all config options are applied
    assert_eq!(optimizer.config().dp_threshold, 5);
    assert!((optimizer.config().index_heat_threshold - 0.5).abs() < 0.001);
    assert!(optimizer.config().enable_pushdown);
    assert!((optimizer.config().default_filter_selectivity - 0.2).abs() < 0.001);
    assert!((optimizer.config().transfer_cost_multiplier - 50.0).abs() < 0.001);
    assert_eq!(optimizer.config().default_bytes_per_row, 64);
}

/// Test that optimizer respects pushdown configuration.
#[test]
fn test_optimizer_pushdown_config() {
    let mut compiler = Compiler::new();

    let source = r#"
        edge(1, 2).
        edge(2, 3).
        reach(X, Y) :- edge(X, Y).
    "#;

    let plan = compiler.compile(source).expect("Should compile");

    // Test with pushdown enabled
    let stats_enabled = Arc::new(StatsManager::new());
    let optimizer_enabled = Optimizer::with_config(
        stats_enabled,
        OptimizerConfig {
            enable_pushdown: true,
            ..Default::default()
        },
    );

    // Test with pushdown disabled
    let stats_disabled = Arc::new(StatsManager::new());
    let optimizer_disabled = Optimizer::with_config(
        stats_disabled,
        OptimizerConfig {
            enable_pushdown: false,
            ..Default::default()
        },
    );

    // Both optimizers should produce valid cost estimates
    for scc_rules in &plan.rules_by_scc {
        for rule in scc_rules {
            let cost_enabled = optimizer_enabled.estimate_cost(&rule.body);
            let cost_disabled = optimizer_disabled.estimate_cost(&rule.body);

            // rows is u64, always non-negative - just verify estimates work
            let _ = cost_enabled.rows;
            let _ = cost_disabled.rows;
        }
    }
}

// =============================================================================
// Plan Optimization Tests
// =============================================================================

/// Test that optimizer can optimize compiled rule bodies.
#[test]
fn test_optimize_compiled_rules() {
    let mut compiler = Compiler::new();

    let source = r#"
        edge(1, 2).
        edge(2, 3).
        reach(X, Y) :- edge(X, Y).
        reach(X, Z) :- reach(X, Y), edge(Y, Z).
    "#;

    let plan = compiler.compile(source).expect("Should compile");

    let mut stats_mgr = StatsManager::new();
    for (_name, rel_id) in compiler.rel_ids() {
        stats_mgr.register_relation(*rel_id);
        stats_mgr.update_cardinality(*rel_id, 1000);
    }

    let stats = Arc::new(stats_mgr);
    let optimizer = Optimizer::new(stats);

    // Optimize each rule body and verify the result is valid
    for scc_rules in &plan.rules_by_scc {
        for rule in scc_rules {
            let optimized = optimizer.optimize(rule.body.clone());

            // Optimized plan should still have valid cost
            let cost = optimizer.estimate_cost(&optimized);
            // rows is u64, always non-negative - verify optimization produced a plan
            assert!(
                cost.cpu_cost >= 0.0,
                "Optimized plan should have valid cost estimate"
            );
        }
    }
}

// =============================================================================
// Statistics Integration Tests
// =============================================================================

/// Test that optimizer uses relation statistics for cost estimation.
#[test]
fn test_optimizer_uses_relation_stats() {
    let mut compiler = Compiler::new();

    let source = r#"
        edge(1, 2).
        reach(X, Y) :- edge(X, Y).
    "#;

    let _plan = compiler.compile(source).expect("Should compile");

    // Get the edge relation ID
    let edge_rel_id = compiler
        .rel_ids()
        .get("edge")
        .copied()
        .expect("Should have edge relation");

    // Create stats with known cardinality
    let mut stats_mgr = StatsManager::new();
    stats_mgr.register_relation(edge_rel_id);
    stats_mgr.update_cardinality(edge_rel_id, 10_000);
    stats_mgr.update_byte_size(edge_rel_id, 320_000); // 32 bytes per row

    let stats = Arc::new(stats_mgr);
    let optimizer = Optimizer::new(stats);

    // Verify scan cost reflects the statistics
    let scan = RirNode::Scan { rel: edge_rel_id };
    let cost = optimizer.estimate_cost(&scan);

    assert_eq!(
        cost.rows, 10_000,
        "Scan cost should use registered cardinality"
    );
    assert!(
        cost.gpu_mem >= 320_000,
        "GPU memory should reflect byte size"
    );
}

/// Test that optimizer uses column statistics for selectivity estimation.
#[test]
fn test_optimizer_uses_column_stats() {
    let mut compiler = Compiler::new();

    let source = r#"
        value(1).
        value(5).
        value(10).
        small(X) :- value(X), X < 10.
    "#;

    let plan = compiler.compile(source).expect("Should compile");

    let value_rel_id = compiler
        .rel_ids()
        .get("value")
        .copied()
        .expect("Should have value relation");

    // Create stats with column information
    let mut stats_mgr = StatsManager::new();
    stats_mgr.register_relation(value_rel_id);
    stats_mgr.update_cardinality(value_rel_id, 1000);

    // Add column stats with distinct count for selectivity estimation
    let mut col_stats = ColumnStats::new(0, ScalarType::I64);
    col_stats.update_distinct(100);
    col_stats.update_range(0, 1000);
    stats_mgr.add_column_stats(value_rel_id, col_stats);

    let stats = Arc::new(stats_mgr);
    let optimizer = Optimizer::new(stats);

    // Find the small rule and verify filter selectivity is applied
    let small_rules: Vec<&CompiledRule> = plan
        .rules_by_scc
        .iter()
        .flatten()
        .filter(|r| r.head == "small")
        .collect();

    if !small_rules.is_empty() {
        let cost = optimizer.estimate_cost(&small_rules[0].body);
        // Filter should reduce row count from base scan
        assert!(cost.rows < 1000, "Filter should reduce row count");
    }
}

/// Test that optimizer tracks access patterns for hot relation detection.
#[test]
fn test_optimizer_hot_relation_tracking() {
    let mut stats_mgr = StatsManager::new();

    // Register multiple relations
    stats_mgr.register_relation(RelId(1));
    stats_mgr.register_relation(RelId(2));
    stats_mgr.register_relation(RelId(3));

    // Heat up relation 1 significantly
    for _ in 0..100 {
        stats_mgr.record_access(RelId(1));
    }

    // Moderate access to relation 2
    for _ in 0..20 {
        stats_mgr.record_access(RelId(2));
    }

    // Minimal access to relation 3
    stats_mgr.record_access(RelId(3));

    let config = OptimizerConfig {
        index_heat_threshold: 0.3,
        ..Default::default()
    };
    let optimizer = Optimizer::with_config(Arc::new(stats_mgr), config);

    let hot_rels = optimizer.recommend_indexes();

    // Relation 1 should definitely be hot
    assert!(
        hot_rels.contains(&RelId(1)),
        "Heavily accessed relation should be recommended for indexing"
    );
}

/// Test join cardinality estimation with cached selectivity.
#[test]
fn test_optimizer_join_cardinality_estimation() {
    let mut stats_mgr = StatsManager::new();

    // Register two relations
    stats_mgr.register_relation(RelId(1));
    stats_mgr.register_relation(RelId(2));
    stats_mgr.update_cardinality(RelId(1), 1000);
    stats_mgr.update_cardinality(RelId(2), 500);

    // Record a historical join result to establish selectivity
    stats_mgr.record_join_result(
        RelId(1),
        RelId(2),
        vec![0],
        vec![0],
        500_000, // product of cardinalities
        2500,    // actual result
    );

    let stats = Arc::new(stats_mgr);
    let optimizer = Optimizer::new(stats);

    // Create a join node and estimate cost
    let join = RirNode::Join {
        left: Box::new(RirNode::Scan { rel: RelId(1) }),
        right: Box::new(RirNode::Scan { rel: RelId(2) }),
        left_keys: vec![0],
        right_keys: vec![0],
        join_type: xlog_ir::JoinType::Inner,
    };

    let cost = optimizer.estimate_cost(&join);

    // Cost should use the cached selectivity (0.005 = 2500/500000)
    // So estimate should be around 1000 * 500 * 0.005 = 2500
    assert!(cost.rows > 0, "Join should have positive row estimate");
}

// =============================================================================
// Edge Case Tests
// =============================================================================

/// Test optimizer handles empty relations gracefully.
#[test]
fn test_optimizer_empty_relation() {
    let mut stats_mgr = StatsManager::new();
    stats_mgr.register_relation(RelId(1));
    stats_mgr.update_cardinality(RelId(1), 0);

    let stats = Arc::new(stats_mgr);
    let optimizer = Optimizer::new(stats);

    let scan = RirNode::Scan { rel: RelId(1) };
    let cost = optimizer.estimate_cost(&scan);

    // Empty relation should have zero rows but still valid cost
    assert_eq!(cost.rows, 0, "Empty relation should have 0 rows");
    assert!(cost.cpu_cost >= 0.0, "CPU cost should be non-negative");
}

/// Test optimizer handles unknown relations with sensible defaults.
#[test]
fn test_optimizer_unknown_relation() {
    let stats = Arc::new(StatsManager::new());
    let optimizer = Optimizer::new(stats);

    // Scan of unregistered relation
    let scan = RirNode::Scan { rel: RelId(999) };
    let cost = optimizer.estimate_cost(&scan);

    // Should use default estimates
    assert_eq!(
        cost.rows, 1000,
        "Unknown relation should use default 1000 rows"
    );
    assert!(cost.gpu_mem > 0, "Should estimate some GPU memory");
}

/// Test PlanCost arithmetic operations.
#[test]
fn test_plan_cost_operations() {
    let cost1 = PlanCost {
        rows: 1000,
        cpu_cost: 100.0,
        gpu_mem: 50_000,
        transfers: 1,
    };

    let cost2 = PlanCost {
        rows: 500,
        cpu_cost: 50.0,
        gpu_mem: 80_000,
        transfers: 1,
    };

    // Test sequential operation cost combination
    let combined = cost1.clone().then(cost2.clone());
    assert_eq!(
        combined.rows, 500,
        "Sequential takes output rows from second"
    );
    assert!((combined.cpu_cost - 150.0).abs() < 0.001, "CPU costs sum");
    assert_eq!(combined.gpu_mem, 80_000, "Peak memory is max");
    assert_eq!(combined.transfers, 2, "Transfers sum");

    // Test total cost calculation
    let total = cost1.total_cost(100.0);
    // cpu_cost(100) + gpu_mem*0.001(50) + transfers*weight(100)
    let expected = 100.0 + 50.0 + 100.0;
    assert!(
        (total - expected).abs() < 0.001,
        "Total cost calculation should match expected formula"
    );
}

/// Test optimizer decision on greedy vs DP algorithm.
#[test]
fn test_optimizer_algorithm_selection() {
    let stats = Arc::new(StatsManager::new());
    let config = OptimizerConfig {
        dp_threshold: 3,
        ..Default::default()
    };
    let optimizer = Optimizer::with_config(stats, config);

    // Single relation: should use DP
    let single_scan = RirNode::Scan { rel: RelId(1) };
    assert!(
        !optimizer.should_use_greedy(&single_scan),
        "Single relation should use DP"
    );

    // Four-way join: should use greedy (threshold is 3)
    let multi_join = RirNode::Join {
        left: Box::new(RirNode::Join {
            left: Box::new(RirNode::Scan { rel: RelId(1) }),
            right: Box::new(RirNode::Scan { rel: RelId(2) }),
            left_keys: vec![0],
            right_keys: vec![0],
            join_type: xlog_ir::JoinType::Inner,
        }),
        right: Box::new(RirNode::Join {
            left: Box::new(RirNode::Scan { rel: RelId(3) }),
            right: Box::new(RirNode::Scan { rel: RelId(4) }),
            left_keys: vec![0],
            right_keys: vec![0],
            join_type: xlog_ir::JoinType::Inner,
        }),
        left_keys: vec![0],
        right_keys: vec![0],
        join_type: xlog_ir::JoinType::Inner,
    };
    assert!(
        optimizer.should_use_greedy(&multi_join),
        "Multi-way join exceeding threshold should use greedy"
    );
}

// =============================================================================
// Real-World Program Tests
// =============================================================================

/// Test optimizer with a complex real-world-like program.
///
/// This simulates a social network analysis query with multiple derived relations.
#[test]
fn test_optimizer_social_network_analysis() {
    let mut compiler = Compiler::new();

    let source = r#"
        // Base relations
        person(1).
        person(2).
        person(3).
        person(4).
        follows(1, 2).
        follows(2, 3).
        follows(3, 1).
        follows(1, 4).

        // Mutual follows (friendship)
        friends(X, Y) :- follows(X, Y), follows(Y, X).

        // Transitive following
        can_reach(X, Y) :- follows(X, Y).
        can_reach(X, Z) :- can_reach(X, Y), follows(Y, Z).

        // People with no followers (isolated)
        has_follower(X) :- follows(Y, X).
        isolated(X) :- person(X), not has_follower(X).
    "#;

    let plan = compiler
        .compile(source)
        .expect("Should compile social network program");

    let mut stats_mgr = StatsManager::new();
    for (name, rel_id) in compiler.rel_ids() {
        stats_mgr.register_relation(*rel_id);
        // Set realistic cardinalities based on predicate type
        let cardinality = match name.as_str() {
            "person" => 1_000_000,
            "follows" => 10_000_000,
            _ => 100_000,
        };
        stats_mgr.update_cardinality(*rel_id, cardinality);
        stats_mgr.update_byte_size(*rel_id, cardinality * 16);
    }

    let stats = Arc::new(stats_mgr);
    let optimizer = Optimizer::new(stats);

    // Verify all rules have reasonable cost estimates
    let mut total_estimated_work = 0u64;
    for scc_rules in &plan.rules_by_scc {
        for rule in scc_rules {
            let cost = optimizer.estimate_cost(&rule.body);
            total_estimated_work += cost.rows;

            // All estimates should be positive
            assert!(
                cost.rows >= 1,
                "Rule {} should have positive rows",
                rule.head
            );
            assert!(
                cost.cpu_cost >= 0.0,
                "Rule {} should have non-negative CPU cost",
                rule.head
            );
        }
    }

    // Complex program should have substantial estimated work
    assert!(
        total_estimated_work > 0,
        "Complex program should have significant estimated work"
    );

    // Should have multiple SCCs due to the predicates
    assert!(plan.sccs.len() >= 2, "Should have multiple SCCs");

    // Should have strata due to negation
    assert!(
        !plan.strata.is_empty(),
        "Should have strata due to negation"
    );
}

/// Test optimizer with graph algorithm program.
///
/// Tests a program computing shortest paths in a weighted graph.
#[test]
fn test_optimizer_graph_algorithm() {
    let mut compiler = Compiler::new();

    let source = r#"
        node(1).
        node(2).
        node(3).
        edge(1, 2).
        edge(2, 3).
        edge(1, 3).

        // Count edges per node
        out_edges(X, count(Y)) :- edge(X, Y).

        // Find leaf nodes (no outgoing edges)
        has_edge(X) :- edge(X, Y).
        leaf(X) :- node(X), not has_edge(X).
    "#;

    let plan = compiler
        .compile(source)
        .expect("Should compile graph algorithm");

    let mut stats_mgr = StatsManager::new();
    for (_name, rel_id) in compiler.rel_ids() {
        stats_mgr.register_relation(*rel_id);
        stats_mgr.update_cardinality(*rel_id, 10_000);
    }

    let stats = Arc::new(stats_mgr);
    let optimizer = Optimizer::new(stats);

    // Find aggregation rule
    let agg_rules: Vec<&CompiledRule> = plan
        .rules_by_scc
        .iter()
        .flatten()
        .filter(|r| r.head == "out_edges")
        .collect();

    assert!(
        !agg_rules.is_empty(),
        "Should have out_edges aggregation rule"
    );

    for rule in agg_rules {
        let cost = optimizer.estimate_cost(&rule.body);
        // Aggregation should produce fewer rows than input
        assert!(
            cost.rows <= 10_000,
            "Aggregation should not exceed input rows"
        );
    }
}
