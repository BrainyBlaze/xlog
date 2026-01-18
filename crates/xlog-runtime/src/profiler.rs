//! Performance profiler for execution statistics
//!
//! This module provides [`Profiler`] for tracking per-operation and per-stratum
//! statistics during query execution. It can be used to identify performance
//! bottlenecks and understand resource usage patterns.
//!
//! # Example
//!
//! ```
//! use xlog_runtime::profiler::{Profiler, OpStats};
//!
//! let mut profiler = Profiler::new(true);
//!
//! // Record operation statistics
//! profiler.record(OpStats {
//!     op_name: "hash_join".to_string(),
//!     input_rows: 1000,
//!     output_rows: 500,
//!     duration_us: 1500,
//!     memory_bytes: 4096,
//! });
//!
//! // Get summary
//! println!("{}", profiler.summary());
//! ```

use std::collections::HashMap;
use std::time::Instant;

/// Statistics for a single operation
///
/// Tracks the name, row counts, duration, and memory usage for an operation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OpStats {
    /// Name of the operation (e.g., "hash_join", "filter", "scan")
    pub op_name: String,
    /// Number of input rows processed
    pub input_rows: u64,
    /// Number of output rows produced
    pub output_rows: u64,
    /// Duration in microseconds
    pub duration_us: u64,
    /// Memory used in bytes
    pub memory_bytes: u64,
}

impl OpStats {
    /// Create a new OpStats with all fields
    pub fn new(
        op_name: impl Into<String>,
        input_rows: u64,
        output_rows: u64,
        duration_us: u64,
        memory_bytes: u64,
    ) -> Self {
        Self {
            op_name: op_name.into(),
            input_rows,
            output_rows,
            duration_us,
            memory_bytes,
        }
    }

    /// Create OpStats for an operation with no memory tracking
    pub fn timed(op_name: impl Into<String>, input_rows: u64, output_rows: u64, duration_us: u64) -> Self {
        Self {
            op_name: op_name.into(),
            input_rows,
            output_rows,
            duration_us,
            memory_bytes: 0,
        }
    }
}

/// Statistics for a single stratum
#[derive(Debug, Clone, Default)]
pub struct StratumStats {
    /// Stratum index (0-based)
    pub stratum_id: usize,
    /// Number of rules in this stratum
    pub num_rules: usize,
    /// Whether this stratum contains recursive rules
    pub is_recursive: bool,
    /// Number of iterations (1 for non-recursive, N for fixpoint)
    pub iterations: usize,
    /// Total duration in microseconds
    pub duration_us: u64,
    /// Operations within this stratum
    pub ops: Vec<OpStats>,
}

impl StratumStats {
    /// Create a new StratumStats
    pub fn new(stratum_id: usize, num_rules: usize, is_recursive: bool) -> Self {
        Self {
            stratum_id,
            num_rules,
            is_recursive,
            iterations: if is_recursive { 0 } else { 1 },
            duration_us: 0,
            ops: Vec::new(),
        }
    }

    /// Get aggregated operation counts by operation name
    pub fn op_summary(&self) -> HashMap<String, (usize, u64)> {
        let mut summary: HashMap<String, (usize, u64)> = HashMap::new();
        for op in &self.ops {
            let entry = summary.entry(op.op_name.clone()).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += op.duration_us;
        }
        summary
    }
}

/// Final execution statistics returned to CLI
#[derive(Debug, Clone, Default)]
pub struct ExecutionStats {
    /// Total execution duration in microseconds
    pub total_duration_us: u64,
    /// Per-stratum statistics
    pub strata: Vec<StratumStats>,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: u64,
    /// Memory budget in bytes
    pub memory_budget_bytes: u64,
    /// Total output rows across all queries
    pub total_output_rows: u64,
}

impl ExecutionStats {
    /// Format stats as human-readable string
    pub fn format_human(&self) -> String {
        let total_secs = self.total_duration_us as f64 / 1_000_000.0;
        let mut output = String::new();

        output.push_str(&format!("Execution completed in {:.2}s\n\n", total_secs));

        for stratum in &self.strata {
            let stratum_secs = stratum.duration_us as f64 / 1_000_000.0;
            let recursive_info = if stratum.is_recursive {
                format!(", recursive, {} iterations", stratum.iterations)
            } else {
                String::new()
            };

            output.push_str(&format!(
                "Stratum {}: {:.2}s ({} rules{})\n",
                stratum.stratum_id, stratum_secs, stratum.num_rules, recursive_info
            ));

            // Aggregate operations by name
            let op_summary = stratum.op_summary();
            let mut ops: Vec<_> = op_summary.into_iter().collect();
            ops.sort_by(|a, b| b.1 .1.cmp(&a.1 .1)); // Sort by duration descending

            for (op_name, (count, duration_us)) in ops {
                let op_secs = duration_us as f64 / 1_000_000.0;
                output.push_str(&format!(
                    "  - {}: {:.2}s ({} calls)\n",
                    op_name, op_secs, count
                ));
            }
        }

        let peak_mb = self.peak_memory_bytes as f64 / (1024.0 * 1024.0);
        let budget_mb = self.memory_budget_bytes as f64 / (1024.0 * 1024.0);
        output.push_str(&format!(
            "\nMemory: {:.0} MB peak / {:.0} MB budget\n",
            peak_mb, budget_mb
        ));
        output.push_str(&format!("Output: {} rows\n", format_rows(self.total_output_rows)));

        output
    }

    /// Format stats as JSON string
    pub fn format_json(&self) -> String {
        let total_ms = self.total_duration_us / 1000;
        let strata_json: Vec<String> = self.strata.iter().map(|s| {
            let ops_json: Vec<String> = s.op_summary().iter().map(|(name, (count, duration))| {
                format!(
                    r#"{{"op":"{}","calls":{},"duration_ms":{}}}"#,
                    name, count, duration / 1000
                )
            }).collect();
            format!(
                r#"{{"stratum":{},"rules":{},"recursive":{},"iterations":{},"duration_ms":{},"ops":[{}]}}"#,
                s.stratum_id, s.num_rules, s.is_recursive, s.iterations, s.duration_us / 1000,
                ops_json.join(",")
            )
        }).collect();

        format!(
            r#"{{"total_ms":{},"strata":[{}],"peak_memory_mb":{},"budget_memory_mb":{},"output_rows":{}}}"#,
            total_ms,
            strata_json.join(","),
            self.peak_memory_bytes / (1024 * 1024),
            self.memory_budget_bytes / (1024 * 1024),
            self.total_output_rows
        )
    }
}

/// Format row count with commas for readability
fn format_rows(rows: u64) -> String {
    let s = rows.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.insert(0, ',');
        }
        result.insert(0, c);
    }
    result
}

/// Execution profiler for tracking operation statistics
///
/// The profiler collects statistics for each operation during query execution.
/// It can be enabled or disabled; when disabled, `record` is a no-op for
/// minimal overhead.
///
/// # Thread Safety
///
/// This implementation is NOT thread-safe. It is designed for single-threaded
/// execution in the MVP.
///
/// # Example
///
/// ```
/// use xlog_runtime::profiler::{Profiler, OpStats};
///
/// // Create an enabled profiler
/// let mut profiler = Profiler::new(true);
///
/// // Record some stats
/// profiler.record(OpStats::timed("scan", 0, 1000, 100));
/// profiler.record(OpStats::timed("filter", 1000, 500, 200));
///
/// // Check totals
/// assert_eq!(profiler.total_duration_us(), 300);
///
/// // Get summary
/// println!("{}", profiler.summary());
/// ```
pub struct Profiler {
    /// Whether profiling is enabled
    enabled: bool,
    /// Collected operation statistics (flat list for backward compatibility)
    stats: Vec<OpStats>,
    /// Per-stratum statistics
    strata: Vec<StratumStats>,
    /// Currently active stratum index
    current_stratum: Option<usize>,
    /// Stratum start time
    stratum_start: Option<Instant>,
    /// Peak memory observed during execution
    peak_memory_bytes: u64,
    /// Memory budget
    memory_budget_bytes: u64,
}

impl Profiler {
    /// Create a new profiler
    ///
    /// # Arguments
    /// * `enabled` - Whether to collect statistics. When disabled, `record` is a no-op.
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            stats: Vec::new(),
            strata: Vec::new(),
            current_stratum: None,
            stratum_start: None,
            peak_memory_bytes: 0,
            memory_budget_bytes: 0,
        }
    }

    /// Set memory budget for reporting
    pub fn set_memory_budget(&mut self, budget_bytes: u64) {
        self.memory_budget_bytes = budget_bytes;
    }

    /// Begin timing a stratum
    ///
    /// # Arguments
    /// * `stratum_id` - The stratum index
    /// * `num_rules` - Number of rules in the stratum
    /// * `is_recursive` - Whether the stratum is recursive
    pub fn begin_stratum(&mut self, stratum_id: usize, num_rules: usize, is_recursive: bool) {
        if !self.enabled {
            return;
        }
        self.current_stratum = Some(stratum_id);
        self.stratum_start = Some(Instant::now());
        self.strata.push(StratumStats::new(stratum_id, num_rules, is_recursive));
    }

    /// End timing the current stratum
    pub fn end_stratum(&mut self) {
        if !self.enabled {
            return;
        }
        if let (Some(start), Some(_idx)) = (self.stratum_start.take(), self.current_stratum.take()) {
            let duration = start.elapsed();
            if let Some(stratum) = self.strata.last_mut() {
                stratum.duration_us = duration.as_micros() as u64;
            }
        }
    }

    /// Record fixpoint iteration count for the current stratum
    pub fn record_iterations(&mut self, iterations: usize) {
        if !self.enabled {
            return;
        }
        if let Some(stratum) = self.strata.last_mut() {
            stratum.iterations = iterations;
        }
    }

    /// Record an operation with timing
    ///
    /// This is a convenience method that calculates duration from a start time.
    ///
    /// # Arguments
    /// * `op_name` - Name of the operation (e.g., "join", "filter", "scan")
    /// * `input_rows` - Number of input rows
    /// * `output_rows` - Number of output rows
    /// * `start` - The instant when the operation started
    /// * `memory_bytes` - Memory used by the operation
    pub fn record_op(
        &mut self,
        op_name: impl Into<String>,
        input_rows: u64,
        output_rows: u64,
        start: Instant,
        memory_bytes: u64,
    ) {
        if !self.enabled {
            return;
        }
        let duration = start.elapsed();
        self.record(OpStats {
            op_name: op_name.into(),
            input_rows,
            output_rows,
            duration_us: duration.as_micros() as u64,
            memory_bytes,
        });
    }

    /// Start timing an operation
    ///
    /// Returns the current instant if profiling is enabled, None otherwise.
    /// This allows zero-overhead timing when profiling is disabled.
    #[inline]
    pub fn start_op(&self) -> Option<Instant> {
        if self.enabled {
            Some(Instant::now())
        } else {
            None
        }
    }

    /// Record peak memory observation
    pub fn record_peak_memory(&mut self, memory_bytes: u64) {
        if !self.enabled {
            return;
        }
        if memory_bytes > self.peak_memory_bytes {
            self.peak_memory_bytes = memory_bytes;
        }
    }

    /// Get execution stats for CLI output
    pub fn execution_stats(&self, total_output_rows: u64) -> ExecutionStats {
        ExecutionStats {
            total_duration_us: self.strata.iter().map(|s| s.duration_us).sum(),
            strata: self.strata.clone(),
            peak_memory_bytes: self.peak_memory_bytes,
            memory_budget_bytes: self.memory_budget_bytes,
            total_output_rows,
        }
    }

    /// Check if profiling is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Record operation statistics
    ///
    /// If the profiler is disabled, this is a no-op.
    /// If a stratum is active, the operation is also recorded in the stratum.
    ///
    /// # Arguments
    /// * `stats` - The operation statistics to record
    pub fn record(&mut self, stats: OpStats) {
        if self.enabled {
            // Also add to current stratum if one is active
            if self.current_stratum.is_some() {
                if let Some(stratum) = self.strata.last_mut() {
                    stratum.ops.push(stats.clone());
                }
            }
            self.stats.push(stats);
        }
    }

    /// Get all recorded statistics
    ///
    /// Returns a slice of all operation statistics collected so far.
    pub fn stats(&self) -> &[OpStats] {
        &self.stats
    }

    /// Clear all recorded statistics
    ///
    /// Removes all collected statistics but keeps the profiler enabled/disabled state.
    pub fn clear(&mut self) {
        self.stats.clear();
        self.strata.clear();
        self.current_stratum = None;
        self.stratum_start = None;
        self.peak_memory_bytes = 0;
    }

    /// Get total duration across all operations in microseconds
    pub fn total_duration_us(&self) -> u64 {
        self.stats.iter().map(|s| s.duration_us).sum()
    }

    /// Get total memory usage across all operations in bytes
    ///
    /// Note: This is the sum of memory reported by each operation, which may
    /// include overlapping allocations. It represents total memory activity
    /// rather than peak memory usage.
    pub fn total_memory_bytes(&self) -> u64 {
        self.stats.iter().map(|s| s.memory_bytes).sum()
    }

    /// Get peak memory usage across all operations in bytes
    ///
    /// Returns the maximum memory_bytes value across all recorded operations.
    /// Returns 0 if no operations have been recorded.
    pub fn peak_memory_bytes(&self) -> u64 {
        self.stats.iter().map(|s| s.memory_bytes).max().unwrap_or(0)
    }

    /// Get the number of recorded operations
    pub fn operation_count(&self) -> usize {
        self.stats.len()
    }

    /// Generate a human-readable summary of the profiling data
    ///
    /// The summary includes:
    /// - Total operation count
    /// - Total duration in milliseconds
    /// - Total memory usage
    /// - Per-operation breakdown with timing and row counts
    pub fn summary(&self) -> String {
        if self.stats.is_empty() {
            return "Profiler: No operations recorded".to_string();
        }

        let total_duration_us = self.total_duration_us();
        let total_duration_ms = total_duration_us as f64 / 1000.0;
        let total_memory = self.total_memory_bytes();
        let peak_memory = self.peak_memory_bytes();

        let mut output = String::new();
        output.push_str("=== Execution Profile ===\n");
        output.push_str(&format!("Operations: {}\n", self.stats.len()));
        output.push_str(&format!("Total duration: {:.3} ms ({} us)\n", total_duration_ms, total_duration_us));
        output.push_str(&format!("Total memory: {} bytes\n", total_memory));
        output.push_str(&format!("Peak memory: {} bytes\n", peak_memory));
        output.push_str("\n--- Operations ---\n");

        for (i, stat) in self.stats.iter().enumerate() {
            let duration_ms = stat.duration_us as f64 / 1000.0;
            let percentage = if total_duration_us > 0 {
                (stat.duration_us as f64 / total_duration_us as f64) * 100.0
            } else {
                0.0
            };

            output.push_str(&format!(
                "{:3}. {:<20} | {:>10} -> {:>10} rows | {:>8.3} ms ({:>5.1}%) | {:>10} bytes\n",
                i + 1,
                truncate_name(&stat.op_name, 20),
                stat.input_rows,
                stat.output_rows,
                duration_ms,
                percentage,
                stat.memory_bytes
            ));
        }

        output
    }

    /// Enable or disable the profiler
    ///
    /// When disabled, `record` becomes a no-op. Existing stats are preserved.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

impl Default for Profiler {
    /// Creates a disabled profiler by default
    fn default() -> Self {
        Self {
            enabled: false,
            stats: Vec::new(),
            strata: Vec::new(),
            current_stratum: None,
            stratum_start: None,
            peak_memory_bytes: 0,
            memory_budget_bytes: 0,
        }
    }
}

/// RAII guard for measuring operation timing
///
/// Records the operation duration when dropped.
pub struct MeasureGuard<'a> {
    profiler: &'a mut Profiler,
    op_name: String,
    input_rows: u64,
    start: Instant,
    output_rows: Option<u64>,
}

impl<'a> MeasureGuard<'a> {
    /// Create a new measure guard
    pub fn new(profiler: &'a mut Profiler, op_name: impl Into<String>, input_rows: u64) -> Self {
        Self {
            profiler,
            op_name: op_name.into(),
            input_rows,
            start: Instant::now(),
            output_rows: None,
        }
    }

    /// Set the output row count and finish timing
    pub fn finish(mut self, output_rows: u64) {
        self.output_rows = Some(output_rows);
        // Drop will record the stats
    }
}

impl<'a> Drop for MeasureGuard<'a> {
    fn drop(&mut self) {
        if self.profiler.is_enabled() {
            let duration = self.start.elapsed();
            self.profiler.record(OpStats {
                op_name: std::mem::take(&mut self.op_name),
                input_rows: self.input_rows,
                output_rows: self.output_rows.unwrap_or(0),
                duration_us: duration.as_micros() as u64,
                memory_bytes: 0,
            });
        }
    }
}

/// Truncate a name to fit within max_len characters
fn truncate_name(name: &str, max_len: usize) -> String {
    if name.len() <= max_len {
        name.to_string()
    } else {
        format!("{}...", &name[..max_len.saturating_sub(3)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============== OpStats Tests ==============

    #[test]
    fn test_opstats_new() {
        let stats = OpStats::new("hash_join", 1000, 500, 1500, 4096);

        assert_eq!(stats.op_name, "hash_join");
        assert_eq!(stats.input_rows, 1000);
        assert_eq!(stats.output_rows, 500);
        assert_eq!(stats.duration_us, 1500);
        assert_eq!(stats.memory_bytes, 4096);
    }

    #[test]
    fn test_opstats_timed() {
        let stats = OpStats::timed("filter", 1000, 800, 200);

        assert_eq!(stats.op_name, "filter");
        assert_eq!(stats.input_rows, 1000);
        assert_eq!(stats.output_rows, 800);
        assert_eq!(stats.duration_us, 200);
        assert_eq!(stats.memory_bytes, 0);
    }

    #[test]
    fn test_opstats_default() {
        let stats = OpStats::default();

        assert_eq!(stats.op_name, "");
        assert_eq!(stats.input_rows, 0);
        assert_eq!(stats.output_rows, 0);
        assert_eq!(stats.duration_us, 0);
        assert_eq!(stats.memory_bytes, 0);
    }

    #[test]
    fn test_opstats_clone() {
        let stats = OpStats::new("scan", 0, 1000, 100, 2048);
        let cloned = stats.clone();

        assert_eq!(stats, cloned);
    }

    #[test]
    fn test_opstats_debug() {
        let stats = OpStats::new("test_op", 100, 50, 10, 1024);
        let debug_str = format!("{:?}", stats);

        assert!(debug_str.contains("test_op"));
        assert!(debug_str.contains("100"));
        assert!(debug_str.contains("50"));
    }

    // ============== Profiler Creation Tests ==============

    #[test]
    fn test_profiler_new_enabled() {
        let profiler = Profiler::new(true);

        assert!(profiler.is_enabled());
        assert!(profiler.stats().is_empty());
    }

    #[test]
    fn test_profiler_new_disabled() {
        let profiler = Profiler::new(false);

        assert!(!profiler.is_enabled());
        assert!(profiler.stats().is_empty());
    }

    #[test]
    fn test_profiler_default() {
        let profiler = Profiler::default();

        assert!(!profiler.is_enabled());
        assert!(profiler.stats().is_empty());
    }

    // ============== Profiler Recording Tests ==============

    #[test]
    fn test_profiler_record_when_enabled() {
        let mut profiler = Profiler::new(true);

        profiler.record(OpStats::new("op1", 100, 50, 10, 1024));
        profiler.record(OpStats::new("op2", 50, 25, 5, 512));

        assert_eq!(profiler.stats().len(), 2);
        assert_eq!(profiler.stats()[0].op_name, "op1");
        assert_eq!(profiler.stats()[1].op_name, "op2");
    }

    #[test]
    fn test_profiler_record_when_disabled() {
        let mut profiler = Profiler::new(false);

        profiler.record(OpStats::new("op1", 100, 50, 10, 1024));
        profiler.record(OpStats::new("op2", 50, 25, 5, 512));

        assert!(profiler.stats().is_empty());
    }

    #[test]
    fn test_profiler_set_enabled() {
        let mut profiler = Profiler::new(false);

        // Initially disabled, record should be no-op
        profiler.record(OpStats::new("op1", 100, 50, 10, 1024));
        assert!(profiler.stats().is_empty());

        // Enable and record
        profiler.set_enabled(true);
        assert!(profiler.is_enabled());
        profiler.record(OpStats::new("op2", 50, 25, 5, 512));
        assert_eq!(profiler.stats().len(), 1);
        assert_eq!(profiler.stats()[0].op_name, "op2");

        // Disable again
        profiler.set_enabled(false);
        assert!(!profiler.is_enabled());
        profiler.record(OpStats::new("op3", 25, 10, 2, 256));
        assert_eq!(profiler.stats().len(), 1); // Still only op2
    }

    // ============== Profiler Clear Tests ==============

    #[test]
    fn test_profiler_clear() {
        let mut profiler = Profiler::new(true);

        profiler.record(OpStats::new("op1", 100, 50, 10, 1024));
        profiler.record(OpStats::new("op2", 50, 25, 5, 512));
        assert_eq!(profiler.stats().len(), 2);

        profiler.clear();

        assert!(profiler.stats().is_empty());
        assert!(profiler.is_enabled()); // Enabled state preserved
    }

    #[test]
    fn test_profiler_clear_preserves_enabled_state() {
        let mut profiler = Profiler::new(true);
        profiler.record(OpStats::new("op1", 100, 50, 10, 1024));
        profiler.clear();

        assert!(profiler.is_enabled());

        profiler.set_enabled(false);
        profiler.clear();

        assert!(!profiler.is_enabled());
    }

    // ============== Profiler Aggregation Tests ==============

    #[test]
    fn test_total_duration_us() {
        let mut profiler = Profiler::new(true);

        profiler.record(OpStats::new("op1", 100, 50, 100, 0));
        profiler.record(OpStats::new("op2", 50, 25, 200, 0));
        profiler.record(OpStats::new("op3", 25, 10, 150, 0));

        assert_eq!(profiler.total_duration_us(), 450);
    }

    #[test]
    fn test_total_duration_us_empty() {
        let profiler = Profiler::new(true);

        assert_eq!(profiler.total_duration_us(), 0);
    }

    #[test]
    fn test_total_memory_bytes() {
        let mut profiler = Profiler::new(true);

        profiler.record(OpStats::new("op1", 100, 50, 10, 1024));
        profiler.record(OpStats::new("op2", 50, 25, 5, 2048));
        profiler.record(OpStats::new("op3", 25, 10, 2, 512));

        assert_eq!(profiler.total_memory_bytes(), 3584);
    }

    #[test]
    fn test_total_memory_bytes_empty() {
        let profiler = Profiler::new(true);

        assert_eq!(profiler.total_memory_bytes(), 0);
    }

    #[test]
    fn test_peak_memory_bytes() {
        let mut profiler = Profiler::new(true);

        profiler.record(OpStats::new("op1", 100, 50, 10, 1024));
        profiler.record(OpStats::new("op2", 50, 25, 5, 4096));
        profiler.record(OpStats::new("op3", 25, 10, 2, 2048));

        assert_eq!(profiler.peak_memory_bytes(), 4096);
    }

    #[test]
    fn test_peak_memory_bytes_empty() {
        let profiler = Profiler::new(true);

        assert_eq!(profiler.peak_memory_bytes(), 0);
    }

    #[test]
    fn test_operation_count() {
        let mut profiler = Profiler::new(true);

        assert_eq!(profiler.operation_count(), 0);

        profiler.record(OpStats::new("op1", 100, 50, 10, 1024));
        assert_eq!(profiler.operation_count(), 1);

        profiler.record(OpStats::new("op2", 50, 25, 5, 512));
        assert_eq!(profiler.operation_count(), 2);

        profiler.clear();
        assert_eq!(profiler.operation_count(), 0);
    }

    // ============== Profiler Summary Tests ==============

    #[test]
    fn test_summary_empty() {
        let profiler = Profiler::new(true);
        let summary = profiler.summary();

        assert!(summary.contains("No operations recorded"));
    }

    #[test]
    fn test_summary_with_operations() {
        let mut profiler = Profiler::new(true);

        profiler.record(OpStats::new("scan", 0, 1000, 100, 4096));
        profiler.record(OpStats::new("filter", 1000, 500, 200, 2048));
        profiler.record(OpStats::new("hash_join", 500, 250, 500, 8192));

        let summary = profiler.summary();

        // Check header
        assert!(summary.contains("=== Execution Profile ==="));
        assert!(summary.contains("Operations: 3"));

        // Check timing
        assert!(summary.contains("Total duration:"));
        assert!(summary.contains("800 us"));

        // Check memory
        assert!(summary.contains("Total memory: 14336 bytes"));
        assert!(summary.contains("Peak memory: 8192 bytes"));

        // Check operations listed
        assert!(summary.contains("scan"));
        assert!(summary.contains("filter"));
        assert!(summary.contains("hash_join"));

        // Check row counts are present
        assert!(summary.contains("1000"));
        assert!(summary.contains("500"));
        assert!(summary.contains("250"));
    }

    #[test]
    fn test_summary_percentages() {
        let mut profiler = Profiler::new(true);

        // Two operations with known durations for percentage calculation
        profiler.record(OpStats::new("fast_op", 100, 50, 250, 0));
        profiler.record(OpStats::new("slow_op", 100, 50, 750, 0));

        let summary = profiler.summary();

        // fast_op should be 25%, slow_op should be 75%
        assert!(summary.contains("25.0%") || summary.contains("25."));
        assert!(summary.contains("75.0%") || summary.contains("75."));
    }

    // ============== Truncate Name Tests ==============

    #[test]
    fn test_truncate_name_short() {
        let result = truncate_name("short", 20);
        assert_eq!(result, "short");
    }

    #[test]
    fn test_truncate_name_exact() {
        let name = "exactly_twenty_chars"; // 20 chars
        let result = truncate_name(name, 20);
        assert_eq!(result, name);
    }

    #[test]
    fn test_truncate_name_long() {
        let name = "this_is_a_very_long_operation_name";
        let result = truncate_name(name, 20);
        assert_eq!(result.len(), 20);
        assert!(result.ends_with("..."));
    }

    // ============== Integration Tests ==============

    #[test]
    fn test_profiler_full_workflow() {
        // Simulate a typical profiling workflow
        let mut profiler = Profiler::new(true);

        // Simulate query execution
        profiler.record(OpStats::new("scan_edge", 0, 10000, 500, 40000));
        profiler.record(OpStats::new("scan_node", 0, 1000, 100, 4000));
        profiler.record(OpStats::new("hash_join", 11000, 5000, 2000, 100000));
        profiler.record(OpStats::new("filter", 5000, 2000, 300, 20000));
        profiler.record(OpStats::new("project", 2000, 2000, 50, 8000));
        profiler.record(OpStats::new("dedup", 2000, 1500, 400, 12000));

        // Verify stats
        assert_eq!(profiler.operation_count(), 6);
        assert_eq!(profiler.total_duration_us(), 3350);
        assert_eq!(profiler.total_memory_bytes(), 184000);
        assert_eq!(profiler.peak_memory_bytes(), 100000);

        // Generate summary
        let summary = profiler.summary();
        assert!(summary.contains("6"));
        assert!(summary.contains("scan_edge"));
        assert!(summary.contains("hash_join"));
        assert!(summary.contains("dedup"));

        // Clear and verify
        profiler.clear();
        assert_eq!(profiler.operation_count(), 0);
        assert!(profiler.is_enabled());
    }

    #[test]
    fn test_profiler_disabled_has_zero_overhead() {
        // When disabled, nothing should be stored
        let mut profiler = Profiler::new(false);

        for i in 0..1000 {
            profiler.record(OpStats::new(format!("op_{}", i), i as u64, i as u64, i as u64, i as u64));
        }

        // Should have zero stats
        assert_eq!(profiler.operation_count(), 0);
        assert_eq!(profiler.total_duration_us(), 0);
        assert_eq!(profiler.total_memory_bytes(), 0);
    }

    #[test]
    fn test_profiler_stats_immutable_reference() {
        let mut profiler = Profiler::new(true);

        profiler.record(OpStats::new("op1", 100, 50, 10, 1024));

        // Get immutable reference
        let stats = profiler.stats();
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].op_name, "op1");

        // Can still record after getting immutable reference (in separate scope)
        profiler.record(OpStats::new("op2", 50, 25, 5, 512));
        assert_eq!(profiler.stats().len(), 2);
    }

    #[test]
    fn test_opstats_equality() {
        let stats1 = OpStats::new("op", 100, 50, 10, 1024);
        let stats2 = OpStats::new("op", 100, 50, 10, 1024);
        let stats3 = OpStats::new("op", 100, 50, 10, 2048); // Different memory

        assert_eq!(stats1, stats2);
        assert_ne!(stats1, stats3);
    }
}
