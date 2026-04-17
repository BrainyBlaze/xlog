//! xlog-induce — bounded exact-induction engine for DTS.
//!
//! Scores all `(left, right)` candidate pairs for the four DTS topologies
//! (chain, star, fanout, fanin) in a single batched GPU pass and returns the
//! top-K per topology with full candidate metadata.
//!
//! Behaviorally equivalent to the `backend="python"` reference implementation
//! in `crates/pyxlog/python/pyxlog/ilp/exact_induce.py` on bounded requests;
//! the parity contract is locked by `python/tests/test_ilp_exact_induce.py`.
//!
//! M8 Phase 1 Task 2 provides the crate scaffolding, types, and public
//! entrypoint. Task 3 implements the batched engine and CUDA kernels.

pub mod index;
pub mod reduce;
pub mod score;
pub mod types;

pub use reduce::{reduce_per_topology, ScoredPair};
pub use types::{
    ExactInductionConfig, ExactInductionResult, ScoredCandidate, Topology,
};

use xlog_core::{RelId, Result, XlogError};
use xlog_cuda::CudaBuffer;

/// Inputs to one `induce_exact()` call.
///
/// `positives` and `negatives` are 2-column binary-pair relations already
/// uploaded into the compiled program's relation store. The engine does not
/// re-upload them — it operates directly on the `CudaBuffer` handles.
pub struct InduceExactRequest<'a> {
    pub head_rel_idx: RelId,
    pub candidate_rel_indices: &'a [RelId],
    pub positives: &'a CudaBuffer,
    pub negatives: Option<&'a CudaBuffer>,
    pub config: ExactInductionConfig,
}

/// Run exact induction against one request.
///
/// Task 3 replaces this stub with the native batched engine. Task 2 keeps
/// the crate boundary real so `crates/pyxlog/src/ilp_exact.rs` can wire up
/// its pyo3 method against the final signature.
pub fn induce_exact(_request: &InduceExactRequest<'_>) -> Result<ExactInductionResult> {
    Err(XlogError::Execution(
        "xlog-induce engine not implemented yet (M8 Phase 1 Task 3)".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topology_as_str_matches_python_contract() {
        assert_eq!(Topology::Chain.as_str(), "chain");
        assert_eq!(Topology::Star.as_str(), "star");
        assert_eq!(Topology::Fanout.as_str(), "fanout");
        assert_eq!(Topology::Fanin.as_str(), "fanin");
    }

    #[test]
    fn topology_all_is_engine_order() {
        assert_eq!(
            Topology::ALL,
            [Topology::Chain, Topology::Star, Topology::Fanout, Topology::Fanin],
        );
    }
}
