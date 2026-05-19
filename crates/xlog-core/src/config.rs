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
    /// Override the stats-backed WCOJ triangle dispatch gate.
    /// `None` uses the production default. `Some(true)` enables
    /// the cardinality model; `Some(false)` disables this runtime's
    /// default stats-backed decision.
    pub wcoj_triangle_dispatch_adaptive: Option<bool>,
    /// Runtime-local hard stop for WCOJ triangle dispatch.
    /// `Some(true)` pins dispatch off across force and stats mode.
    /// `Some(false)` leaves dispatch available for this runtime.
    /// `None` uses the production default.
    pub wcoj_triangle_dispatch_disabled: Option<bool>,

    /// v0.6.5 slice 2 — force gate for the 4-cycle WCOJ dispatch.
    /// `Some(true)` / env `XLOG_USE_WCOJ_4CYCLE=1` forces every
    /// recognized 4-cycle to dispatch the GPU kernel. `Some(false)` is explicit force-off. `None`
    /// (default) consults the env.
    pub wcoj_4cycle_dispatch: Option<bool>,
    /// v0.6.5 slice 2 — adaptive opt-in for 4-cycle WCOJ. **Unlike
    /// triangle, 4-cycle adaptive is opt-in by default**, not
    /// default-on: `None` resolves to `false`. Default-on for
    /// 4-cycle is a separate follow-up slice gated by bench evidence.
    pub wcoj_4cycle_dispatch_adaptive: Option<bool>,
    /// v0.6.5 slice 2 — kill switch for 4-cycle WCOJ. Same shape
    /// as triangle's kill switch: beats force + adaptive.
    pub wcoj_4cycle_dispatch_disabled: Option<bool>,
    /// v0.6.5 W2.5 — selects the runtime WCOJ cost model.
    /// `None` resolves by env/default precedence; see
    /// [`RuntimeConfig::with_wcoj_cost_model`].
    pub wcoj_cost_model: Option<CostModelKind>,
    /// v0.8.6 G086_CSE — runtime common subexpression elimination.
    ///
    /// `Some(true)` enables structural CSE for safe deterministic subplans.
    /// `Some(false)` disables it. `None` consults `XLOG_CSE`; unset defaults
    /// to disabled so existing runtime behavior is preserved unless the caller
    /// opts in.
    pub common_subexpression_elimination: Option<bool>,
    /// v0.8.6 G086_ADAPT — runtime adaptive re-optimization adoption gate.
    ///
    /// `Some(true)` allows an executor to compare a baseline plan against a
    /// compiler-supplied candidate plan using runtime telemetry, adopt the
    /// candidate only when deterministic mis-plan thresholds trigger, and roll
    /// back on adverse candidates. `Some(false)` disables the adoption path.
    /// `None` consults `XLOG_ADAPTIVE_REOPT`; unset defaults to disabled.
    pub adaptive_reoptimization: Option<bool>,
    /// Minimum mis-plan ratio required before the executor attempts to adopt a
    /// candidate re-optimized plan. `None` consults
    /// `XLOG_ADAPTIVE_REOPT_MIN_RATIO`; unset or invalid values default to 1.2.
    pub adaptive_reoptimization_min_misplan_ratio: Option<f64>,
    /// v0.8.6 G086_INDEX — persistent hash index manager gate.
    ///
    /// `Some(true)` enables persistent build-side hash index reuse in the
    /// existing executor join-index cache. `Some(false)` disables the manager.
    /// `None` consults `XLOG_PERSISTENT_HASH_INDEXES`; unset defaults to
    /// enabled to preserve the existing adaptive-indexing behavior.
    pub persistent_hash_indexes: Option<bool>,
    /// v0.8.6 G086_INDEX — record background-build mode for the persistent
    /// hash index manager. The current runtime keeps builds on the existing
    /// provider path but records background build requests/completions so the
    /// transition to recorded asynchronous builds has stable telemetry.
    pub persistent_hash_index_background_build: Option<bool>,
}

/// v0.6.5 W2.5 cost-model selector for WCOJ dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostModelKind {
    /// Legacy skew-classifier opt-out selector.
    ///
    /// On current G38 integration code the GPU classifier surface is absent,
    /// so this selector is implemented as a conservative opt-out from
    /// stats/cardinality dispatch.
    SkewClassifier,
    /// Stats/cardinality-backed dispatch selector.
    Cardinality,
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
            wcoj_4cycle_dispatch: None,
            wcoj_4cycle_dispatch_adaptive: None,
            wcoj_4cycle_dispatch_disabled: None,
            wcoj_cost_model: None,
            common_subexpression_elimination: None,
            adaptive_reoptimization: None,
            adaptive_reoptimization_min_misplan_ratio: None,
            persistent_hash_indexes: None,
            persistent_hash_index_background_build: None,
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

    /// Override the stats-backed WCOJ triangle dispatch gate.
    /// Force-WCOJ (`with_wcoj_triangle_dispatch(Some(true))`)
    /// takes precedence.
    pub fn with_wcoj_triangle_dispatch_adaptive(mut self, override_value: Option<bool>) -> Self {
        self.wcoj_triangle_dispatch_adaptive = override_value;
        self
    }

    /// Engage / disengage the runtime-local WCOJ triangle
    /// dispatch hard stop.
    pub fn with_wcoj_triangle_dispatch_disabled(mut self, override_value: Option<bool>) -> Self {
        self.wcoj_triangle_dispatch_disabled = override_value;
        self
    }

    /// v0.6.5 slice 2 — override the 4-cycle force-gate.
    /// `Some(true)` forces the GPU kernel; `Some(false)` is
    /// explicit force-off; `None` consults `XLOG_USE_WCOJ_4CYCLE`.
    pub fn with_wcoj_4cycle_dispatch(mut self, override_value: Option<bool>) -> Self {
        self.wcoj_4cycle_dispatch = override_value;
        self
    }

    /// v0.6.5 slice 2 — override the 4-cycle stats opt-in.
    /// `Some(true)` engages the cardinality model; `Some(false)` skips it.
    /// `None` resolves to `false` (opt-in by default — 4-cycle
    /// does NOT inherit triangle's default-on behavior).
    pub fn with_wcoj_4cycle_dispatch_adaptive(mut self, override_value: Option<bool>) -> Self {
        self.wcoj_4cycle_dispatch_adaptive = override_value;
        self
    }

    /// v0.6.5 slice 2 — engage / disengage the 4-cycle kill switch.
    /// Same shape as the triangle kill switch.
    pub fn with_wcoj_4cycle_dispatch_disabled(mut self, override_value: Option<bool>) -> Self {
        self.wcoj_4cycle_dispatch_disabled = override_value;
        self
    }

    /// Select which WCOJ cost-model implementation the runtime consults.
    ///
    /// Precedence:
    /// 1. Explicit config field set here.
    /// 2. `XLOG_WCOJ_COST_MODEL=cardinality` or `skew`.
    /// 3. Default `Cardinality`.
    pub fn with_wcoj_cost_model(mut self, kind: Option<CostModelKind>) -> Self {
        self.wcoj_cost_model = kind;
        self
    }

    /// Enable or disable runtime common subexpression elimination.
    pub fn with_common_subexpression_elimination(mut self, override_value: Option<bool>) -> Self {
        self.common_subexpression_elimination = override_value;
        self
    }

    /// Enable or disable adaptive runtime re-optimization adoption.
    pub fn with_adaptive_reoptimization(mut self, override_value: Option<bool>) -> Self {
        self.adaptive_reoptimization = override_value;
        self
    }

    /// Set the minimum mis-plan ratio for adaptive runtime re-optimization.
    pub fn with_adaptive_reoptimization_min_misplan_ratio(
        mut self,
        override_value: Option<f64>,
    ) -> Self {
        self.adaptive_reoptimization_min_misplan_ratio = override_value;
        self
    }

    /// Enable or disable persistent build-side hash index reuse.
    pub fn with_persistent_hash_indexes(mut self, override_value: Option<bool>) -> Self {
        self.persistent_hash_indexes = override_value;
        self
    }

    /// Enable or disable persistent hash-index background-build telemetry.
    pub fn with_persistent_hash_index_background_build(
        mut self,
        override_value: Option<bool>,
    ) -> Self {
        self.persistent_hash_index_background_build = override_value;
        self
    }

    /// Resolve runtime common subexpression elimination by config/env/default.
    pub fn resolved_common_subexpression_elimination(&self) -> bool {
        if let Some(enabled) = self.common_subexpression_elimination {
            return enabled;
        }

        std::env::var("XLOG_CSE")
            .map(|raw| {
                matches!(
                    raw.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "on" | "yes"
                )
            })
            .unwrap_or(false)
    }

    /// Resolve adaptive runtime re-optimization by config/env/default.
    pub fn resolved_adaptive_reoptimization(&self) -> bool {
        if let Some(enabled) = self.adaptive_reoptimization {
            return enabled;
        }

        std::env::var("XLOG_ADAPTIVE_REOPT")
            .map(|raw| {
                matches!(
                    raw.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "on" | "yes"
                )
            })
            .unwrap_or(false)
    }

    /// Resolve the deterministic mis-plan threshold for adaptive re-optimization.
    pub fn resolved_adaptive_reoptimization_min_misplan_ratio(&self) -> f64 {
        const DEFAULT_MIN_RATIO: f64 = 1.2;
        if let Some(value) = self.adaptive_reoptimization_min_misplan_ratio {
            return sanitize_adaptive_reoptimization_ratio(value, DEFAULT_MIN_RATIO);
        }

        std::env::var("XLOG_ADAPTIVE_REOPT_MIN_RATIO")
            .ok()
            .and_then(|raw| raw.trim().parse::<f64>().ok())
            .map(|value| sanitize_adaptive_reoptimization_ratio(value, DEFAULT_MIN_RATIO))
            .unwrap_or(DEFAULT_MIN_RATIO)
    }

    /// Resolve persistent hash-index reuse by config/env/default.
    pub fn resolved_persistent_hash_indexes(&self) -> bool {
        if let Some(enabled) = self.persistent_hash_indexes {
            return enabled;
        }

        std::env::var("XLOG_PERSISTENT_HASH_INDEXES")
            .map(|raw| {
                !matches!(
                    raw.trim().to_ascii_lowercase().as_str(),
                    "0" | "false" | "off" | "no"
                )
            })
            .unwrap_or(true)
    }

    /// Resolve background-build telemetry for persistent hash indexes.
    pub fn resolved_persistent_hash_index_background_build(&self) -> bool {
        if let Some(enabled) = self.persistent_hash_index_background_build {
            return enabled;
        }

        std::env::var("XLOG_PERSISTENT_HASH_INDEX_BACKGROUND_BUILD")
            .map(|raw| {
                matches!(
                    raw.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "on" | "yes"
                )
            })
            .unwrap_or(false)
    }

    /// Resolve the effective WCOJ cost-model selector.
    pub fn resolved_wcoj_cost_model(&self) -> CostModelKind {
        if let Some(kind) = self.wcoj_cost_model {
            return kind;
        }
        let raw = std::env::var("XLOG_WCOJ_COST_MODEL").ok();
        let normalized = raw.as_deref().map(|s| s.trim().to_ascii_lowercase());
        match normalized.as_deref() {
            Some("cardinality") => CostModelKind::Cardinality,
            Some("skew") | Some("skewclassifier") | Some(_) => CostModelKind::SkewClassifier,
            None => CostModelKind::Cardinality,
        }
    }
}

fn sanitize_adaptive_reoptimization_ratio(value: f64, fallback: f64) -> f64 {
    if value.is_finite() && value >= 1.0 {
        value
    } else {
        fallback
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
