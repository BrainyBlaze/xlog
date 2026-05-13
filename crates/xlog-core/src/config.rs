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

    /// v0.6.5 slice 2 — force gate for the 4-cycle WCOJ dispatch.
    /// `Some(true)` / env `XLOG_USE_WCOJ_4CYCLE=1` forces every
    /// recognized 4-cycle to dispatch the GPU kernel (classifier
    /// bypassed). `Some(false)` is explicit force-off. `None`
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

    /// v0.6.5 slice 5 — selects which `WcojCostModel` impl
    /// the executor consults when deciding whether to dispatch
    /// WCOJ in adaptive mode. `None` (default) falls through
    /// the precedence ladder — see `RuntimeConfig::with_wcoj_cost_model`.
    pub wcoj_cost_model: Option<CostModelKind>,
}

/// v0.6.5 slice 5 — cost model selector for WCOJ dispatch.
///
/// The legacy `SkewClassifier` model makes dispatch decisions on
/// classifier-only signal. The default
/// `Cardinality` model fuses classifier with cardinality
/// estimates from `xlog_stats::StatsManager` and gracefully
/// delegates to `SkewClassifier` when stats are missing.
///
/// See `with_wcoj_cost_model` for the precedence rules and the
/// explicit `skew` opt-out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostModelKind {
    /// Slice 1–4 default: skew-classifier-only dispatch decision.
    SkewClassifier,
    /// Slice 5 opt-in: classifier blended with cardinality estimates.
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

    /// v0.6.5 slice 2 — override the 4-cycle force-gate.
    /// `Some(true)` forces the GPU kernel; `Some(false)` is
    /// explicit force-off; `None` consults `XLOG_USE_WCOJ_4CYCLE`.
    pub fn with_wcoj_4cycle_dispatch(mut self, override_value: Option<bool>) -> Self {
        self.wcoj_4cycle_dispatch = override_value;
        self
    }

    /// v0.6.5 slice 2 — override the 4-cycle adaptive opt-in.
    /// `Some(true)` engages the classifier; `Some(false)` skips it.
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

    /// v0.6.5 slice 5 — select which `WcojCostModel` impl the
    /// executor consults for adaptive dispatch decisions.
    ///
    /// **Precedence (pinned)**:
    ///   1. `Some(CostModelKind)` set here wins.
    ///   2. Else `XLOG_WCOJ_COST_MODEL` env var:
    ///      `cardinality` → `Cardinality`; `skew` /
    ///      `skewclassifier` / unrecognized → `SkewClassifier`.
    ///   3. Else default: `Cardinality`.
    ///
    /// W2.5 ships the cardinality model as the production default;
    /// the legacy skew classifier remains available as an explicit
    /// conservative opt-out.
    pub fn with_wcoj_cost_model(mut self, kind: Option<CostModelKind>) -> Self {
        self.wcoj_cost_model = kind;
        self
    }

    /// v0.6.5 slice 5 — resolve the effective cost-model kind
    /// from the precedence ladder. See `with_wcoj_cost_model`
    /// for the rules.
    ///
    /// Test note: env-var resolution is non-deterministic
    /// across parallel tests; consumers that need to set
    /// `XLOG_WCOJ_COST_MODEL` must use the env-lock pattern.
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

    // -----------------------------------------------------------
    // v0.6.5 slice 5 — wcoj_cost_model precedence + env resolution
    // -----------------------------------------------------------

    /// Serialize `XLOG_WCOJ_COST_MODEL` mutation across tests —
    /// process-global env is shared. Mirrors the pattern in
    /// `xlog-runtime::executor::wcoj_dispatch::tests`.
    fn cost_model_env_lock() -> &'static std::sync::Mutex<()> {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(()))
    }

    /// RAII snapshot of `XLOG_WCOJ_COST_MODEL` — captures the
    /// pre-test value, clears it, restores on drop. Held while
    /// `cost_model_env_lock` is locked so concurrent tests don't
    /// race.
    struct CostModelEnvSnapshot(Option<String>);

    impl CostModelEnvSnapshot {
        fn capture_and_clear() -> Self {
            let prior = std::env::var("XLOG_WCOJ_COST_MODEL").ok();
            // SAFETY: caller holds `cost_model_env_lock`.
            unsafe {
                std::env::remove_var("XLOG_WCOJ_COST_MODEL");
            }
            Self(prior)
        }
    }

    impl Drop for CostModelEnvSnapshot {
        fn drop(&mut self) {
            // SAFETY: snapshot drops before the lock is released.
            unsafe {
                match self.0.take() {
                    Some(v) => std::env::set_var("XLOG_WCOJ_COST_MODEL", v),
                    None => std::env::remove_var("XLOG_WCOJ_COST_MODEL"),
                }
            }
        }
    }

    fn with_cost_model_env<R>(f: impl FnOnce() -> R) -> R {
        let _guard = cost_model_env_lock()
            .lock()
            .expect("cost-model env lock poisoned");
        let _snap = CostModelEnvSnapshot::capture_and_clear();
        f()
    }

    #[test]
    fn wcoj_cost_model_default_is_cardinality_when_unset() {
        with_cost_model_env(|| {
            let cfg = RuntimeConfig::default();
            assert_eq!(cfg.resolved_wcoj_cost_model(), CostModelKind::Cardinality);
        });
    }

    #[test]
    fn cost_model_env_var_cardinality_resolves_to_cardinality() {
        with_cost_model_env(|| {
            // SAFETY: caller holds env lock.
            unsafe {
                std::env::set_var("XLOG_WCOJ_COST_MODEL", "cardinality");
            }
            let cfg = RuntimeConfig::default();
            assert_eq!(cfg.resolved_wcoj_cost_model(), CostModelKind::Cardinality);
        });
    }

    #[test]
    fn wcoj_cost_model_env_var_skew_resolves_to_skew_classifier() {
        with_cost_model_env(|| {
            unsafe {
                std::env::set_var("XLOG_WCOJ_COST_MODEL", "skew");
            }
            let cfg = RuntimeConfig::default();
            assert_eq!(
                cfg.resolved_wcoj_cost_model(),
                CostModelKind::SkewClassifier
            );
        });
    }

    #[test]
    fn cost_model_env_var_garbage_resolves_to_skew_classifier() {
        with_cost_model_env(|| {
            unsafe {
                std::env::set_var("XLOG_WCOJ_COST_MODEL", "not-a-real-model");
            }
            let cfg = RuntimeConfig::default();
            assert_eq!(
                cfg.resolved_wcoj_cost_model(),
                CostModelKind::SkewClassifier
            );
        });
    }

    #[test]
    fn cost_model_env_var_cardinality_with_whitespace_and_case_resolves() {
        with_cost_model_env(|| {
            unsafe {
                std::env::set_var("XLOG_WCOJ_COST_MODEL", "  Cardinality  ");
            }
            let cfg = RuntimeConfig::default();
            assert_eq!(cfg.resolved_wcoj_cost_model(), CostModelKind::Cardinality);
        });
    }

    #[test]
    fn cost_model_config_field_overrides_env_var() {
        with_cost_model_env(|| {
            unsafe {
                std::env::set_var("XLOG_WCOJ_COST_MODEL", "skew");
            }
            let cfg =
                RuntimeConfig::default().with_wcoj_cost_model(Some(CostModelKind::Cardinality));
            assert_eq!(cfg.resolved_wcoj_cost_model(), CostModelKind::Cardinality);
        });
    }

    #[test]
    fn cost_model_config_field_skew_overrides_env_var_cardinality() {
        with_cost_model_env(|| {
            unsafe {
                std::env::set_var("XLOG_WCOJ_COST_MODEL", "cardinality");
            }
            let cfg =
                RuntimeConfig::default().with_wcoj_cost_model(Some(CostModelKind::SkewClassifier));
            assert_eq!(
                cfg.resolved_wcoj_cost_model(),
                CostModelKind::SkewClassifier
            );
        });
    }
}
