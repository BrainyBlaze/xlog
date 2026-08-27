use std::fmt;

pub(crate) const PROVIDER_BUDGET_BYTES: u64 = 8 * 1024 * 1024 * 1024;
pub(crate) const PROCESS_VISIBLE_LIMIT_BYTES: u64 = 38 * 1024 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BenchmarkMemoryLimit {
    ProviderTracked,
    ProcessVisibleDevice,
}

impl fmt::Display for BenchmarkMemoryLimit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProviderTracked => formatter.write_str("provider-tracked"),
            Self::ProcessVisibleDevice => formatter.write_str("process-visible"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BenchmarkMemoryObservation {
    pub(crate) provider_tracked_peak_bytes: u64,
    pub(crate) process_visible_peak_bytes: u64,
}

impl BenchmarkMemoryObservation {
    pub(crate) fn merge(self, other: Self) -> Self {
        Self {
            provider_tracked_peak_bytes: self
                .provider_tracked_peak_bytes
                .max(other.provider_tracked_peak_bytes),
            process_visible_peak_bytes: self
                .process_visible_peak_bytes
                .max(other.process_visible_peak_bytes),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BenchmarkMemoryViolation {
    pub(crate) limit: BenchmarkMemoryLimit,
    pub(crate) observed_bytes: u64,
    pub(crate) limit_bytes: u64,
}

impl fmt::Display for BenchmarkMemoryViolation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} benchmark memory exceeded: observed_bytes={} limit_bytes={}",
            self.limit, self.observed_bytes, self.limit_bytes
        )
    }
}

impl std::error::Error for BenchmarkMemoryViolation {}

pub(crate) fn enforce_benchmark_memory_limits(
    observation: BenchmarkMemoryObservation,
) -> Result<(), BenchmarkMemoryViolation> {
    if observation.provider_tracked_peak_bytes > PROVIDER_BUDGET_BYTES {
        return Err(BenchmarkMemoryViolation {
            limit: BenchmarkMemoryLimit::ProviderTracked,
            observed_bytes: observation.provider_tracked_peak_bytes,
            limit_bytes: PROVIDER_BUDGET_BYTES,
        });
    }
    if observation.process_visible_peak_bytes > PROCESS_VISIBLE_LIMIT_BYTES {
        return Err(BenchmarkMemoryViolation {
            limit: BenchmarkMemoryLimit::ProcessVisibleDevice,
            observed_bytes: observation.process_visible_peak_bytes,
            limit_bytes: PROCESS_VISIBLE_LIMIT_BYTES,
        });
    }
    Ok(())
}
