//! Test diagnostics, failure reporting, and result tracking.

use std::time::{Duration, Instant};

/// Rich failure diagnostic with context.
#[derive(Debug, Clone)]
pub(crate) struct FailureDiagnostic {
    pub(crate) category: &'static str,
    pub(crate) test_name: &'static str,
    pub(crate) input_size: usize,
    pub(crate) input_sample: String,
    pub(crate) expected_sample: String,
    pub(crate) actual_sample: String,
    pub(crate) first_diff_index: Option<usize>,
    pub(crate) diff_count: usize,
    pub(crate) error_message: String,
}

impl FailureDiagnostic {
    /// Create a new failure diagnostic.
    pub(crate) fn new(
        category: &'static str,
        test_name: &'static str,
        input_size: usize,
        error_message: String,
    ) -> Self {
        Self {
            category,
            test_name,
            input_size,
            input_sample: String::new(),
            expected_sample: String::new(),
            actual_sample: String::new(),
            first_diff_index: None,
            diff_count: 0,
            error_message,
        }
    }

    /// Add input sample (first N elements).
    pub(crate) fn with_input_sample(mut self, sample: String) -> Self {
        self.input_sample = sample;
        self
    }

    /// Add expected/actual samples.
    pub(crate) fn with_comparison(
        mut self,
        expected: String,
        actual: String,
        first_diff: Option<usize>,
        diff_count: usize,
    ) -> Self {
        self.expected_sample = expected;
        self.actual_sample = actual;
        self.first_diff_index = first_diff;
        self.diff_count = diff_count;
        self
    }

    /// Generate human-readable failure report.
    pub(crate) fn report(&self) -> String {
        let mut report = String::new();
        report.push_str(&format!(
            "=== FAILURE: {}/{} ===\n",
            self.category, self.test_name
        ));
        report.push_str(&format!("Input size: {}\n", self.input_size));
        report.push_str(&format!("Error: {}\n", self.error_message));

        if !self.input_sample.is_empty() {
            report.push_str(&format!("Input sample: {}\n", self.input_sample));
        }

        if self.diff_count > 0 {
            report.push_str(&format!("Differences: {} total\n", self.diff_count));
            if let Some(idx) = self.first_diff_index {
                report.push_str(&format!("First diff at index: {}\n", idx));
            }
            report.push_str(&format!("Expected: {}\n", self.expected_sample));
            report.push_str(&format!("Actual:   {}\n", self.actual_sample));
        }

        report.push_str("===\n");
        report
    }
}

/// Test execution status.
#[derive(Debug, Clone)]
pub(crate) enum TestStatus {
    Passed,
    Failed,
    Skipped { reason: &'static str },
    Error { message: String },
}

impl TestStatus {
    pub(crate) fn is_passed(&self) -> bool {
        matches!(self, TestStatus::Passed)
    }

    pub(crate) fn is_failed(&self) -> bool {
        matches!(self, TestStatus::Failed | TestStatus::Error { .. })
    }
}

/// Individual test result.
#[derive(Debug, Clone)]
pub(crate) struct TestResult {
    pub(crate) name: &'static str,
    pub(crate) status: TestStatus,
    pub(crate) duration: Duration,
    pub(crate) diagnostic: Option<FailureDiagnostic>,
}

impl TestResult {
    pub(crate) fn passed(name: &'static str, duration: Duration) -> Self {
        Self {
            name,
            status: TestStatus::Passed,
            duration,
            diagnostic: None,
        }
    }

    pub(crate) fn failed(name: &'static str, duration: Duration, diagnostic: FailureDiagnostic) -> Self {
        Self {
            name,
            status: TestStatus::Failed,
            duration,
            diagnostic: Some(diagnostic),
        }
    }

    pub(crate) fn skipped(name: &'static str, reason: &'static str) -> Self {
        Self {
            name,
            status: TestStatus::Skipped { reason },
            duration: Duration::ZERO,
            diagnostic: None,
        }
    }

    pub(crate) fn error(name: &'static str, duration: Duration, message: String) -> Self {
        Self {
            name,
            status: TestStatus::Error { message },
            duration,
            diagnostic: None,
        }
    }
}

/// Category-level result aggregation.
#[derive(Debug, Clone)]
pub(crate) struct CategoryResult {
    pub(crate) name: &'static str,
    pub(crate) tests: Vec<TestResult>,
    pub(crate) duration: Duration,
}

impl CategoryResult {
    pub(crate) fn new(name: &'static str) -> Self {
        Self {
            name,
            tests: Vec::new(),
            duration: Duration::ZERO,
        }
    }

    pub(crate) fn add_result(&mut self, result: TestResult) {
        self.tests.push(result);
    }

    pub(crate) fn set_duration(&mut self, duration: Duration) {
        self.duration = duration;
    }

    pub(crate) fn passed_count(&self) -> usize {
        self.tests.iter().filter(|t| t.status.is_passed()).count()
    }

    pub(crate) fn failed_count(&self) -> usize {
        self.tests.iter().filter(|t| t.status.is_failed()).count()
    }

    pub(crate) fn skipped_count(&self) -> usize {
        self.tests
            .iter()
            .filter(|t| matches!(t.status, TestStatus::Skipped { .. }))
            .count()
    }

    pub(crate) fn total_count(&self) -> usize {
        self.tests.len()
    }

    pub(crate) fn all_passed(&self) -> bool {
        self.tests
            .iter()
            .all(|t| t.status.is_passed() || matches!(t.status, TestStatus::Skipped { .. }))
    }
}

/// Full certification results.
#[derive(Debug)]
pub(crate) struct CertificationResults {
    pub(crate) categories: Vec<CategoryResult>,
    pub(crate) start_time: Instant,
    pub(crate) total_duration: Duration,
}

impl CertificationResults {
    pub(crate) fn new() -> Self {
        Self {
            categories: Vec::new(),
            start_time: Instant::now(),
            total_duration: Duration::ZERO,
        }
    }

    /// Run a category and record results.
    pub(crate) fn run_category<F>(&mut self, _name: &'static str, f: F)
    where
        F: FnOnce() -> CategoryResult,
    {
        let start = Instant::now();
        let mut result = f();
        result.set_duration(start.elapsed());
        self.categories.push(result);
    }

    /// Add a pre-computed category result.
    pub(crate) fn add_category(&mut self, result: CategoryResult) {
        self.categories.push(result);
    }

    /// Finalize results and compute total duration.
    pub(crate) fn finalize(&mut self) {
        self.total_duration = self.start_time.elapsed();
    }

    pub(crate) fn total_tests(&self) -> usize {
        self.categories.iter().map(|c| c.total_count()).sum()
    }

    pub(crate) fn total_passed(&self) -> usize {
        self.categories.iter().map(|c| c.passed_count()).sum()
    }

    pub(crate) fn total_failed(&self) -> usize {
        self.categories.iter().map(|c| c.failed_count()).sum()
    }

    pub(crate) fn total_skipped(&self) -> usize {
        self.categories.iter().map(|c| c.skipped_count()).sum()
    }

    pub(crate) fn all_passed(&self) -> bool {
        self.categories.iter().all(|c| c.all_passed())
    }

    /// Print summary to stdout.
    pub(crate) fn print_summary(&self) {
        println!("\n========== CERTIFICATION RESULTS ==========");
        println!("Total Duration: {:.2}s", self.total_duration.as_secs_f64());
        println!();

        for cat in &self.categories {
            let status = if cat.all_passed() { "PASS" } else { "FAIL" };
            println!(
                "[{}] {}: {}/{} passed, {} skipped ({:.2}s)",
                status,
                cat.name,
                cat.passed_count(),
                cat.total_count(),
                cat.skipped_count(),
                cat.duration.as_secs_f64()
            );
        }

        println!();
        println!("========== SUMMARY ==========");
        println!("Total Tests:   {}", self.total_tests());
        println!("Passed:        {}", self.total_passed());
        println!("Failed:        {}", self.total_failed());
        println!("Skipped:       {}", self.total_skipped());
        println!(
            "Pass Rate:     {:.1}%",
            (self.total_passed() as f64 / self.total_tests().max(1) as f64) * 100.0
        );
        println!("=====================================\n");
    }

    /// Generate detailed failure report.
    pub(crate) fn failure_report(&self) -> String {
        let mut report = String::new();

        for cat in &self.categories {
            for test in &cat.tests {
                if let Some(diag) = &test.diagnostic {
                    report.push_str(&diag.report());
                    report.push('\n');
                } else if let TestStatus::Error { message } = &test.status {
                    report.push_str(&format!("=== ERROR: {}/{} ===\n", cat.name, test.name));
                    report.push_str(&format!("Error: {}\n", message));
                    report.push_str("===\n\n");
                }
            }
        }

        report
    }
}

impl Default for CertificationResults {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper macro for running a test and capturing results.
#[macro_export]
macro_rules! run_test {
    ($results:expr, $name:expr, $body:expr) => {{
        let start = std::time::Instant::now();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $body));
        let duration = start.elapsed();

        match result {
            Ok(()) => $crate::harness::TestResult::passed($name, duration),
            Err(e) => {
                let message = if let Some(s) = e.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = e.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "Unknown panic".to_string()
                };
                $crate::harness::TestResult::error($name, duration, message)
            }
        }
    }};
}

/// Helper to format a slice sample for diagnostics.
pub(crate) fn format_sample<T: std::fmt::Debug>(data: &[T], max_items: usize) -> String {
    if data.len() <= max_items {
        format!("{:?}", data)
    } else {
        let sample: Vec<_> = data.iter().take(max_items).collect();
        format!("{:?}... ({} more)", sample, data.len() - max_items)
    }
}
