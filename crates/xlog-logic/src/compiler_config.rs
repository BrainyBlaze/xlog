//! Compile-time configuration for the W2.1 variable-ordering cost
//! model.
//!
//! `CompilerConfig` is a per-call argument to
//! [`crate::compile::Compiler::compile_with_config_and_stats_snapshot`].
//! `CompilerConfig::default()` disables W2.1 — slice 1/2/4 + W2.2
//! dispatch behavior is bit-identical when the default config is in
//! effect.
//!
//! Activation requires explicitly constructing a `CompilerConfig`
//! with [`WcojVarOrderingKind::LeaderCardinality`]. There is no
//! environment override on this path; env-driven activation is out
//! of W2.1 scope.
//!
//! # Threshold contract
//!
//! `wcoj_var_ordering_threshold` is `pub` to allow struct-literal
//! construction, but the promoter MUST go through
//! [`CompilerConfig::effective_wcoj_var_ordering_threshold`] so
//! out-of-range struct-literal values fall back to
//! [`CompilerConfig::DEFAULT_THRESHOLD`] rather than silently
//! widening the gate.

/// Selector for the W2.1 variable-ordering cost model.
///
/// `Disabled` is the load-bearing default: when set, the promoter
/// never emits `RirNode::MultiWayJoin::var_order`, and slice
/// 1/2/4/W2.2 dispatch + row-set semantics are bit-identical.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WcojVarOrderingKind {
    /// W2.1 disabled: promoter never sets `var_order`. Bit-identical
    /// to slice 1/2/4 + W2.2.
    Disabled,
    /// Use the default `LeaderCardinalityModel` to pick a
    /// stats-driven leader for triangle / 4-cycle WCOJ inputs.
    LeaderCardinality,
}

/// Compile-time configuration for the W2.1 variable-ordering cost
/// model.
///
/// See module docs for activation semantics + threshold contract.
#[derive(Debug, Clone, PartialEq)]
pub struct CompilerConfig {
    /// Variable-ordering cost-model selector. Default `Disabled`.
    pub wcoj_variable_ordering: WcojVarOrderingKind,

    /// Raw threshold field. Public to keep struct-literal
    /// construction available, but the promoter MUST NOT read this
    /// field directly. Use
    /// [`CompilerConfig::effective_wcoj_var_ordering_threshold`] so
    /// out-of-range values are clamped at use, not silently honored.
    pub wcoj_var_ordering_threshold: f64,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            wcoj_variable_ordering: WcojVarOrderingKind::Disabled,
            wcoj_var_ordering_threshold: Self::DEFAULT_THRESHOLD,
        }
    }
}

impl CompilerConfig {
    /// Default ratio at or below which a leader candidate triggers
    /// `var_order = Some(...)`. The gate fires on
    /// `min_card / default_leader_card ≤ threshold`. A smaller
    /// threshold demands a clearer win.
    pub const DEFAULT_THRESHOLD: f64 = 0.5;

    /// Resolve the threshold the promoter actually uses.
    ///
    /// Out-of-range values fall back to [`Self::DEFAULT_THRESHOLD`]:
    /// * `NaN`
    /// * non-finite (`±INFINITY`)
    /// * `≤ 0.0` (would never fire — clamps to default to keep the
    ///   gate honest)
    /// * `> 1.0` (would always fire — clamps to default to prevent
    ///   silent gate-disable via struct-literal)
    pub fn effective_wcoj_var_ordering_threshold(&self) -> f64 {
        let t = self.wcoj_var_ordering_threshold;
        if !t.is_finite() || t <= 0.0 || t > 1.0 {
            Self::DEFAULT_THRESHOLD
        } else {
            t
        }
    }
}

#[cfg(test)]
mod tests {
    //! W2.1 step 4: 4 resolver unit tests pinning the
    //! out-of-range fallback contract.
    use super::*;

    #[test]
    fn default_threshold_is_half() {
        let c = CompilerConfig::default();
        assert_eq!(c.wcoj_var_ordering_threshold, 0.5);
        assert_eq!(c.effective_wcoj_var_ordering_threshold(), 0.5);
        assert_eq!(c.wcoj_variable_ordering, WcojVarOrderingKind::Disabled);
    }

    #[test]
    fn resolver_passes_through_valid_in_range() {
        let c = CompilerConfig {
            wcoj_var_ordering_threshold: 0.3,
            ..CompilerConfig::default()
        };
        assert_eq!(c.effective_wcoj_var_ordering_threshold(), 0.3);
    }

    #[test]
    fn resolver_clamps_zero_and_negative_to_default() {
        // `0.0` boundary: the gate would never fire — clamp.
        let zero = CompilerConfig {
            wcoj_var_ordering_threshold: 0.0,
            ..CompilerConfig::default()
        };
        assert_eq!(
            zero.effective_wcoj_var_ordering_threshold(),
            CompilerConfig::DEFAULT_THRESHOLD
        );
        let neg = CompilerConfig {
            wcoj_var_ordering_threshold: -0.5,
            ..CompilerConfig::default()
        };
        assert_eq!(
            neg.effective_wcoj_var_ordering_threshold(),
            CompilerConfig::DEFAULT_THRESHOLD
        );
    }

    #[test]
    fn resolver_clamps_above_one_and_nonfinite_to_default() {
        let above = CompilerConfig {
            wcoj_var_ordering_threshold: 1.5,
            ..CompilerConfig::default()
        };
        assert_eq!(
            above.effective_wcoj_var_ordering_threshold(),
            CompilerConfig::DEFAULT_THRESHOLD
        );
        let nan = CompilerConfig {
            wcoj_var_ordering_threshold: f64::NAN,
            ..CompilerConfig::default()
        };
        assert_eq!(
            nan.effective_wcoj_var_ordering_threshold(),
            CompilerConfig::DEFAULT_THRESHOLD
        );
        let inf = CompilerConfig {
            wcoj_var_ordering_threshold: f64::INFINITY,
            ..CompilerConfig::default()
        };
        assert_eq!(
            inf.effective_wcoj_var_ordering_threshold(),
            CompilerConfig::DEFAULT_THRESHOLD
        );
    }
}
