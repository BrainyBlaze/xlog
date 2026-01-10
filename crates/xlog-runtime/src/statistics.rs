//! Query statistics tracking for adaptive optimization
//!
//! Tracks access patterns and selectivity to guide index building decisions.

use std::collections::HashMap;

/// Statistics for a specific join pair
#[derive(Debug, Clone, Default)]
pub struct JoinStats {
    pub count: u64,
    pub total_selectivity: f64,
    pub avg_selectivity: f64,
}

/// Query statistics tracker
#[derive(Debug, Default)]
pub struct QueryStatistics {
    scan_counts: HashMap<String, u64>,
    join_stats: HashMap<(String, String), JoinStats>,
    total_ops: u64,
}

impl QueryStatistics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_scan(&mut self, relation: &str) {
        *self.scan_counts.entry(relation.to_string()).or_insert(0) += 1;
        self.total_ops += 1;
    }

    pub fn record_join(&mut self, left: &str, right: &str, selectivity: f64) {
        let key = (left.to_string(), right.to_string());
        let stats = self.join_stats.entry(key).or_default();
        stats.count += 1;
        stats.total_selectivity += selectivity;
        stats.avg_selectivity = stats.total_selectivity / stats.count as f64;
        self.total_ops += 1;
    }

    pub fn scan_count(&self, relation: &str) -> u64 {
        self.scan_counts.get(relation).copied().unwrap_or(0)
    }

    pub fn join_stats(&self, left: &str, right: &str) -> Option<&JoinStats> {
        self.join_stats.get(&(left.to_string(), right.to_string()))
    }

    pub fn heat(&self, relation: &str) -> u64 {
        let scan_heat = self.scan_count(relation);
        let join_heat: u64 = self.join_stats.iter()
            .filter(|((l, r), _)| l == relation || r == relation)
            .map(|(_, stats)| stats.count * 2)
            .sum();
        scan_heat + join_heat
    }

    pub fn relations_by_heat(&self) -> Vec<(String, u64)> {
        let mut relations: Vec<_> = self.scan_counts.keys()
            .map(|r| (r.clone(), self.heat(r)))
            .collect();

        for (left, right) in self.join_stats.keys() {
            if !self.scan_counts.contains_key(left) {
                relations.push((left.clone(), self.heat(left)));
            }
            if !self.scan_counts.contains_key(right) {
                relations.push((right.clone(), self.heat(right)));
            }
        }

        relations.sort_by(|a, b| b.1.cmp(&a.1));
        relations.dedup_by(|a, b| a.0 == b.0);
        relations
    }

    pub fn clear(&mut self) {
        self.scan_counts.clear();
        self.join_stats.clear();
        self.total_ops = 0;
    }

    pub fn total_ops(&self) -> u64 {
        self.total_ops
    }
}
