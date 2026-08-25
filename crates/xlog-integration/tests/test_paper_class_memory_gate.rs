#[path = "../benches/fixtures/benchmark_memory.rs"]
mod benchmark_memory;

use benchmark_memory::{
    enforce_benchmark_memory_limits, BenchmarkMemoryLimit, BenchmarkMemoryObservation,
    PROCESS_VISIBLE_LIMIT_BYTES, PROVIDER_BUDGET_BYTES,
};

#[test]
fn paper_class_memory_limits_accept_the_exact_boundaries() {
    enforce_benchmark_memory_limits(BenchmarkMemoryObservation {
        provider_tracked_peak_bytes: PROVIDER_BUDGET_BYTES,
        process_visible_peak_bytes: PROCESS_VISIBLE_LIMIT_BYTES,
    })
    .expect("both documented memory boundaries are inclusive");
}

#[test]
fn paper_class_memory_observations_merge_independent_provider_peaks() {
    let merged = BenchmarkMemoryObservation {
        provider_tracked_peak_bytes: 7,
        process_visible_peak_bytes: 11,
    }
    .merge(BenchmarkMemoryObservation {
        provider_tracked_peak_bytes: 13,
        process_visible_peak_bytes: 5,
    });

    assert_eq!(merged.provider_tracked_peak_bytes, 13);
    assert_eq!(merged.process_visible_peak_bytes, 11);
}

#[test]
fn paper_class_memory_limits_distinguish_provider_budget_exhaustion() {
    let violation = enforce_benchmark_memory_limits(BenchmarkMemoryObservation {
        provider_tracked_peak_bytes: PROVIDER_BUDGET_BYTES + 1,
        process_visible_peak_bytes: PROCESS_VISIBLE_LIMIT_BYTES,
    })
    .expect_err("tracked provider memory above 8 GiB must fail");

    assert_eq!(violation.limit, BenchmarkMemoryLimit::ProviderTracked);
    assert_eq!(violation.observed_bytes, PROVIDER_BUDGET_BYTES + 1);
    assert_eq!(violation.limit_bytes, PROVIDER_BUDGET_BYTES);
    assert!(violation.to_string().contains("provider-tracked"));
}

#[test]
fn paper_class_memory_limits_distinguish_process_visible_exhaustion() {
    let violation = enforce_benchmark_memory_limits(BenchmarkMemoryObservation {
        provider_tracked_peak_bytes: PROVIDER_BUDGET_BYTES,
        process_visible_peak_bytes: PROCESS_VISIBLE_LIMIT_BYTES + 1,
    })
    .expect_err("process-visible device memory above 38 GiB must fail");

    assert_eq!(violation.limit, BenchmarkMemoryLimit::ProcessVisibleDevice);
    assert_eq!(violation.observed_bytes, PROCESS_VISIBLE_LIMIT_BYTES + 1);
    assert_eq!(violation.limit_bytes, PROCESS_VISIBLE_LIMIT_BYTES);
    assert!(violation.to_string().contains("process-visible"));
}
