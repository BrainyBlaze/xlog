//! Probabilistic reasoning tier for XLOG (Phase 4).

pub mod cnf;
pub mod compilation;
pub mod exact;
#[allow(dead_code)] // reserved: GPU-only exact compilation path (not yet wired to main pipeline)
pub mod exact_gpu;
pub mod gpu;
pub mod kc;
pub mod mc;
pub mod neural_fast_path;
pub mod pir;
pub mod provenance;
pub mod wfs;
pub mod xgcf;

pub use pir::{ChoiceVarId, LeafId, PirGraph, PirNode, PirNodeId};
pub use provenance::{ChoiceSource, GroundAtom, Provenance, Value};

// Primary entry points (convenience re-exports)
pub use compilation::{
    compile_gpu_d4_and_verify, compile_gpu_d4_and_verify_cached,
    GpuCompileConfig, CircuitCompileProfile,
};
pub use exact::{ExactDdnnfProgram, ExactResult, GpuConfig};
pub use mc::{
    McEvalConfig, McProgram, McSamplingMethod, McCountStrategy,
    McResult, McDeviceResult, EvidenceForcing, ForceabilityReason,
};
pub use wfs::{
    WfsConfig, WfsResult, TruthValue, WfsAtom, WfsRule, WfsLiteral,
    evaluate_wfs_rules, evaluate_wfs_with_rules,
};
