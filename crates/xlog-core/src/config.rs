//! Configuration types for XLOG runtime

/// GPU memory budget configuration
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    /// Maximum device memory to use in bytes
    pub device_bytes: u64,
    /// Allow out-of-core execution (spill to host)
    pub allow_ooc: bool,
    /// Abort on memory budget exceeded (vs try to continue)
    pub abort_on_exceed: bool,
}

impl Default for MemoryBudget {
    fn default() -> Self {
        Self {
            device_bytes: 0, // Will be set from device query
            allow_ooc: false,
            abort_on_exceed: true,
        }
    }
}

impl MemoryBudget {
    /// Create a budget using 80% of available device memory
    pub fn from_device_memory(total_bytes: u64) -> Self {
        Self {
            device_bytes: (total_bytes as f64 * 0.8) as u64,
            allow_ooc: false,
            abort_on_exceed: true,
        }
    }

    /// Create a budget with explicit byte limit
    pub fn with_limit(device_bytes: u64) -> Self {
        Self {
            device_bytes,
            allow_ooc: false,
            abort_on_exceed: true,
        }
    }

    /// Enable out-of-core mode
    pub fn with_ooc(mut self) -> Self {
        self.allow_ooc = true;
        self
    }
}

/// Runtime configuration for XLOG execution
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Memory budget settings
    pub memory: MemoryBudget,
    /// Use deterministic execution (may be slower)
    pub deterministic: bool,
    /// Enable profiling (row counts, memory tracking)
    pub profile: bool,
    /// Maximum fixpoint iterations before abort
    pub max_iterations: u32,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            memory: MemoryBudget::default(),
            deterministic: true,
            profile: false,
            max_iterations: 1_000_000,
        }
    }
}

impl RuntimeConfig {
    /// Enable profiling
    pub fn with_profiling(mut self) -> Self {
        self.profile = true;
        self
    }

    /// Set memory budget
    pub fn with_memory(mut self, memory: MemoryBudget) -> Self {
        self.memory = memory;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_budget_default() {
        let budget = MemoryBudget::default();
        assert!(!budget.allow_ooc);
        assert!(budget.abort_on_exceed);
    }

    #[test]
    fn test_runtime_config_default() {
        let config = RuntimeConfig::default();
        assert!(config.deterministic);
        assert!(!config.profile);
    }

    #[test]
    fn test_memory_budget_from_device() {
        let budget = MemoryBudget::from_device_memory(10_000_000_000);
        assert_eq!(budget.device_bytes, 8_000_000_000);
    }
}
