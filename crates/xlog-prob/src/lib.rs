//! Probabilistic reasoning tier for XLOG.
#![warn(missing_docs)]

mod aggregates;
pub mod cnf;
pub mod compilation;
mod decision_order;
pub mod epistemic;
pub mod epistemic_production;
pub mod exact;
pub mod gpu;
/// Knowledge-compilation circuit representations.
pub mod kc;
mod logsumexp;
pub mod mc;
pub mod neural_fast_path;
pub mod pir;
pub mod provenance;
pub mod wfs;
pub mod xgcf;

pub use pir::{ChoiceVarId, LeafId, PirGraph, PirNode, PirNodeId};
pub use provenance::{
    AggregateLiftReport, AggregateLiftStatus, ChoiceSource, GroundAtom, Provenance, Value,
};

// Primary entry points (convenience re-exports)
pub use compilation::{
    compile_gpu_d4_and_verify, compile_gpu_d4_and_verify_cached, CircuitCompileProfile,
    GpuCompileConfig,
};
pub use exact::{ExactDdnnfProgram, ExactResult, GpuConfig, ProbVarInfo};
pub use mc::{
    EvidenceForcing, ForceabilityReason, McCountStrategy, McDeviceResult, McEvalConfig,
    McHotLoopTransfers, McProgram, McResult, McSamplingMethod,
};
pub use wfs::{
    evaluate_wfs_rules, evaluate_wfs_with_rules, TruthValue, WfsAtom, WfsConfig, WfsLiteral,
    WfsResult, WfsRule,
};

#[cfg(test)]
pub(crate) mod test_gpu_lock {
    use std::sync::{Mutex, MutexGuard, OnceLock};

    pub(crate) fn lock() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}
