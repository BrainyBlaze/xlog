//! Performance profiler for execution statistics
//!
//! This module provides [`Profiler`] for tracking per-operation statistics during
//! query execution. It can be used to identify performance bottlenecks and
//! understand resource usage patterns.
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
    /// Collected operation statistics
    stats: Vec<OpStats>,
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
        }
    }

    /// Check if profiling is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Record operation statistics
    ///
    /// If the profiler is disabled, this is a no-op.
    ///
    /// # Arguments
    /// * `stats` - The operation statistics to record
    pub fn record(&mut self, stats: OpStats) {
        if self.enabled {
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
        Self::new(false)
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
