//! Probabilistic reasoning tier for XLOG (Phase 4).

pub mod cnf;
pub mod compilation;
pub mod exact;
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
