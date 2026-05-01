//! Configuration types for XLOG runtime

/// GPU memory budget configuration.
///
/// Use [`MemoryBudget::default()`] or the builder methods ([`MemoryBudget::from_device_memory`],
/// [`MemoryBudget::with_limit`], [`MemoryBudget::with_ooc`]) to construct.
#[derive(Debug, Clone)]
#[non_exhaustive]
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

/// Runtime configuration for XLOG execution.
///
/// Use [`RuntimeConfig::default()`] and the builder methods to construct.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RuntimeConfig {
    /// Memory budget settings
    pub memory: MemoryBudget,
    /// Use deterministic execution (may be slower)
    pub deterministic: bool,
    /// Enable profiling (row counts, memory tracking)
    pub profile: bool,
    /// Maximum fixpoint iterations before abort
    pub max_iterations: u32,
    /// Opt-in: enforce the strict deterministic-Datalog D2H gate during
    /// `Executor::execute_plan`. When `true`, any data-plane device-to-host
    /// transfer (column downloads, internal `dtoh_sync_copy_into_tracked`
    /// calls) returns `XlogError::Execution` and increments the provider's
    /// `deterministic_d2h_violation_count`. Metadata reads via
    /// `dtoh_scalar_untracked` remain allowed.
    ///
    /// Default `false`: v0.5.5 still has known data-plane D2H paths in
    /// relational set difference and binary-join count/materialize that are
    /// scheduled for replacement before the default flips.
    pub strict_deterministic_d2h: bool,
    /// Override the env-driven WCOJ triangle dispatch gate
    /// (`XLOG_USE_WCOJ_TRIANGLE_U32`). `None` (default) consults
    /// the env var; `Some(true)` / `Some(false)` force the
    /// runtime to ignore the env and use the explicit value.
    /// Test-only knob — production callers should leave this
    /// `None` and configure via the env var.
    pub wcoj_triangle_dispatch: Option<bool>,
    /// Override the env-driven WCOJ adaptive-dispatch gate
    /// (`XLOG_USE_WCOJ_TRIANGLE_ADAPTIVE`). Post-default-on:
    /// `None` (default) means "consult env, fall back to
    /// adaptive-on if env is unset". `Some(true)` is an
    /// explicit opt-in (no-op vs default). `Some(false)` is
    /// an explicit opt-out that disables the default-on for
    /// this runtime.
    ///
    /// Decision tree (highest → lowest):
    ///   1. Kill switch (`wcoj_triangle_dispatch_disabled` /
    ///      `XLOG_DISABLE_WCOJ_TRIANGLE`) → no dispatch.
    ///   2. Force (`wcoj_triangle_dispatch=Some(true)` /
    ///      `XLOG_USE_WCOJ_TRIANGLE_U32=1`) → WCOJ pipeline,
    ///      classifier bypassed.
    ///   3. Explicit force-off
    ///      (`wcoj_triangle_dispatch=Some(false)`) → no dispatch.
    ///   4. Adaptive resolution (config → env → default-on).
    ///      Adaptive on → classifier runs; score ≥ 0.10 → WCOJ.
    ///      Else → no dispatch.
    pub wcoj_triangle_dispatch_adaptive: Option<bool>,
    /// Hard kill switch for ALL WCOJ triangle dispatch.
    /// `Some(true)` (or env `XLOG_DISABLE_WCOJ_TRIANGLE=1`)
    /// pins dispatch off — beats force, beats adaptive, beats
    /// the default-on. Use case: ops emergency to disable
    /// WCOJ without touching application code or other env
    /// vars. `None` (default) consults the env. `Some(false)`
    /// is an explicit "do not engage the kill switch"
    /// (programmatic override over an env-set kill).
    pub wcoj_triangle_dispatch_disabled: Option<bool>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            memory: MemoryBudget::default(),
            deterministic: true,
            profile: false,
            max_iterations: 1_000_000,
            strict_deterministic_d2h: false,
            wcoj_triangle_dispatch: None,
            wcoj_triangle_dispatch_adaptive: None,
            wcoj_triangle_dispatch_disabled: None,
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

    /// Enable the strict deterministic-Datalog D2H gate for this runtime.
    pub fn with_strict_deterministic_d2h(mut self) -> Self {
        self.strict_deterministic_d2h = true;
        self
    }

    /// Override the env-driven WCOJ triangle dispatch gate. Pass
    /// `Some(true)` / `Some(false)` to force the runtime to ignore
    /// `XLOG_USE_WCOJ_TRIANGLE_U32`; `None` to consult the env var
    /// (the production default). Test-only knob.
    pub fn with_wcoj_triangle_dispatch(mut self, override_value: Option<bool>) -> Self {
        self.wcoj_triangle_dispatch = override_value;
        self
    }

    /// Override the env-driven WCOJ adaptive-dispatch gate. Pass
    /// `Some(true)` / `Some(false)` to force the runtime to ignore
    /// `XLOG_USE_WCOJ_TRIANGLE_ADAPTIVE`; `None` to consult the
    /// env var (the production default). Force-WCOJ
    /// (`with_wcoj_triangle_dispatch(Some(true))`) takes
    /// precedence and bypasses the classifier entirely.
    pub fn with_wcoj_triangle_dispatch_adaptive(mut self, override_value: Option<bool>) -> Self {
        self.wcoj_triangle_dispatch_adaptive = override_value;
        self
    }

    /// Engage / disengage the WCOJ triangle dispatch kill
    /// switch. `Some(true)` pins dispatch off across every
    /// other flag (force, adaptive, default-on). `Some(false)`
    /// explicitly does NOT engage the kill switch (useful for
    /// programmatically overriding `XLOG_DISABLE_WCOJ_TRIANGLE=1`
    /// in a test). `None` consults the env var.
    pub fn with_wcoj_triangle_dispatch_disabled(mut self, override_value: Option<bool>) -> Self {
        self.wcoj_triangle_dispatch_disabled = override_value;
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
