//! Probabilistic reasoning tier for XLOG (Phase 4).
#![warn(missing_docs)]

mod aggregates;
#[allow(missing_docs)] // TODO(v0.6): document or make pub(crate)
pub mod cnf;
#[allow(missing_docs)] // TODO(v0.6): document or make pub(crate)
pub mod compilation;
pub mod epistemic;
pub mod epistemic_production;
#[allow(missing_docs)] // TODO(v0.6): document or make pub(crate)
pub mod exact;
#[allow(dead_code)] // reserved: GPU-only exact compilation path (not yet wired to main pipeline)
#[allow(missing_docs)]
pub mod exact_gpu;
#[allow(missing_docs)] // TODO(v0.6): document or make pub(crate)
pub mod gpu;
#[allow(missing_docs)] // TODO(v0.6): document or make pub(crate)
pub mod kc;
#[allow(missing_docs)] // TODO(v0.6): document or make pub(crate)
pub mod mc;
#[allow(missing_docs)] // TODO(v0.6): document or make pub(crate)
pub mod neural_fast_path;
#[allow(missing_docs)] // TODO(v0.6): document or make pub(crate)
pub mod pir;
#[allow(missing_docs)] // TODO(v0.6): document or make pub(crate)
pub mod provenance;
#[allow(missing_docs)] // TODO(v0.6): document or make pub(crate)
pub mod wfs;
#[allow(missing_docs)] // TODO(v0.6): document or make pub(crate)
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
pub use exact::{ExactDdnnfProgram, ExactResult, GpuConfig};
pub use mc::{
    EvidenceForcing, ForceabilityReason, McCountStrategy, McDeviceResult, McEvalConfig,
    McHotLoopTransfers, McProgram, McResult, McSamplingMethod,
};
pub use wfs::{
    evaluate_wfs_rules, evaluate_wfs_with_rules, TruthValue, WfsAtom, WfsConfig, WfsLiteral,
    WfsResult, WfsRule,
};
