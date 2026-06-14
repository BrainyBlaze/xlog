use std::sync::{Mutex, OnceLock};

use xlog_core::{CostModelKind, RuntimeConfig};

fn cost_model_env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

struct CostModelEnvSnapshot(Option<String>);

impl CostModelEnvSnapshot {
    fn capture_and_clear() -> Self {
        let prior = std::env::var("XLOG_WCOJ_COST_MODEL").ok();
        unsafe {
            std::env::remove_var("XLOG_WCOJ_COST_MODEL");
        }
        Self(prior)
    }
}

impl Drop for CostModelEnvSnapshot {
    fn drop(&mut self) {
        unsafe {
            match self.0.take() {
                Some(value) => std::env::set_var("XLOG_WCOJ_COST_MODEL", value),
                None => std::env::remove_var("XLOG_WCOJ_COST_MODEL"),
            }
        }
    }
}

fn with_cost_model_env<R>(f: impl FnOnce() -> R) -> R {
    let _guard = cost_model_env_lock()
        .lock()
        .expect("cost-model env lock poisoned");
    let _snapshot = CostModelEnvSnapshot::capture_and_clear();
    f()
}

#[test]
fn runtime_config_cardinality_is_default_cost_model() {
    with_cost_model_env(|| {
        assert_eq!(
            RuntimeConfig::default().resolved_wcoj_cost_model(),
            CostModelKind::Cardinality,
            "bare RuntimeConfig must select Cardinality by default"
        );
    });
}

#[test]
fn runtime_config_env_skew_opt_out_preserved() {
    with_cost_model_env(|| {
        unsafe {
            std::env::set_var("XLOG_WCOJ_COST_MODEL", "skew");
        }
        assert_eq!(
            RuntimeConfig::default().resolved_wcoj_cost_model(),
            CostModelKind::SkewClassifier,
            "XLOG_WCOJ_COST_MODEL=skew must preserve the explicit legacy opt-out"
        );
    });
}

#[test]
fn runtime_config_override_beats_env_cost_model() {
    with_cost_model_env(|| {
        unsafe {
            std::env::set_var("XLOG_WCOJ_COST_MODEL", "skew");
        }
        assert_eq!(
            RuntimeConfig::default()
                .with_wcoj_cost_model(Some(CostModelKind::Cardinality))
                .resolved_wcoj_cost_model(),
            CostModelKind::Cardinality,
            "explicit config override must stay higher precedence than env"
        );
    });
}
