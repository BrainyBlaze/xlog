//! Tests for query statistics tracking

use xlog_runtime::QueryStatistics;

#[test]
fn test_statistics_tracking() {
    let mut stats = QueryStatistics::new();

    // Record some accesses
    stats.record_scan("users");
    stats.record_scan("users");
    stats.record_scan("orders");
    stats.record_join("users", "orders", 0.1); // 10% selectivity

    assert_eq!(stats.scan_count("users"), 2);
    assert_eq!(stats.scan_count("orders"), 1);
    assert_eq!(stats.scan_count("nonexistent"), 0);

    let join_stats = stats.join_stats("users", "orders").unwrap();
    assert!((join_stats.avg_selectivity - 0.1).abs() < 0.01);
}

#[test]
fn test_heat_calculation() {
    let mut stats = QueryStatistics::new();

    // Hot relation: accessed many times
    for _ in 0..100 {
        stats.record_scan("hot_table");
    }

    // Cold relation: accessed once
    stats.record_scan("cold_table");

    assert!(stats.heat("hot_table") > stats.heat("cold_table"));
}
