//! xlog-induce — bounded exact-induction engine for external consumer.
//!
//! Scores all `(left, right)` candidate pairs for the four external consumer topologies
//! (chain, star, fanout, fanin) in a single batched GPU pass and returns the
//! top-K per topology with full candidate metadata.
//!
//! Behaviorally equivalent to the `backend="python"` reference implementation
//! in `crates/pyxlog/python/pyxlog/ilp/exact_induce.py` on bounded requests;
//! the parity contract is locked by `python/tests/test_ilp_exact_induce.py`.
//!
//! M8 Phase 1 Task 3 Stage B is landing incrementally:
//!   * Task 2 (done):  crate scaffolding, types, public entrypoint stub.
//!   * Stage A (done): deterministic reduction + 16 unit tests locking the
//!     comparator and diagnostics bit-for-bit against Python.
//!   * Stage B 3B.1 (done): native request validation + trivial-dead-end
//!     early returns.
//!   * Stage B 3B.4 (done): engine calls the batched scoring kernel.
//!   * Stage B 3B.5 (this change): native production dispatch uses
//!     device-side top-K selection and only transfers compact selected rows.

pub mod index;
pub mod provenance;
pub mod reduce;
pub mod score;
pub mod types;
mod validate;

pub use provenance::InductionProvenanceRegistry;
pub use reduce::{reduce_per_topology, ScoredPair};
pub use types::{
    ExactInductionConfig, ExactInductionResult, InducedRuleProvenance, InducedRuleRegistry,
    InductionAlternative, InductionSupportRow, RuleSourceKind, ScoredCandidate, Topology,
};

use xlog_core::{RelId, Result, ScalarType, XlogError};
use xlog_cuda::{CudaBuffer, CudaKernelProvider};

use validate::{classify_request, PreKernelOutcome, RequestMetadata};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExactPairType {
    U64,
    U32,
    Symbol,
}

/// Inputs to one [`induce_exact`] call.
///
/// Each candidate is a `(RelId, &CudaBuffer)` pair: the `RelId` is a label that
/// flows through to every [`ScoredCandidate`] produced from that buffer, and
/// the `CudaBuffer` carries the relation's current binary-pair facts.
///
/// `positives` and `negatives` are themselves binary-pair buffers (arity 2,
/// column type `U64`). Name-to-`RelId` resolution and relation-store lookup
/// happen at the pyxlog boundary — the engine only sees indices + handles.
pub struct InduceExactRequest<'a> {
    pub head_rel_idx: RelId,
    pub candidates: &'a [(RelId, &'a CudaBuffer)],
    pub positives: &'a CudaBuffer,
    pub negatives: Option<&'a CudaBuffer>,
    pub config: ExactInductionConfig,
}

/// Run exact induction against one request.
///
/// Returns an [`ExactInductionResult`] matching the Python reference on
/// bounded inputs.
///
/// The `provider` argument owns the kernel launcher — it's passed separately
/// from the request so the engine can also materialize short-lived GPU
/// buffers (an empty negatives buffer when `request.negatives` is `None`).
pub fn induce_exact(
    provider: &CudaKernelProvider,
    request: &InduceExactRequest<'_>,
) -> Result<ExactInductionResult> {
    // Empty candidates is a trivial dead-end and needs no CUDA inspection —
    // matches the Python reference's `if not body_indices: return ...` path.
    if request.candidates.is_empty() {
        return Ok(ExactInductionResult::default());
    }

    // Buffer-level validation (arity 2, accepted typed pair columns). Runs before
    // metadata extraction so we fail loud on pyxlog-side assembly bugs.
    let pair_type = validate_pair_buffer(request.positives, "positives")?;
    if let Some(neg) = request.negatives {
        require_pair_type(neg, "negatives", pair_type)?;
    }
    for (i, (_, buf)) in request.candidates.iter().enumerate() {
        require_pair_type(buf, &format!("candidate[{}]", i), pair_type)?;
    }

    // Extract row counts from the cached host-side metadata. The DLPack ingest
    // path (`CudaKernelProvider::from_dlpack_tensors_with_schema`) populates
    // `cached_row_count`, so this is a pure struct read — no D2H. That's how
    // we keep the hot-loop D2H budget flat across candidate counts.
    let pos_count = cached_rows(request.positives, "positives")?;
    let neg_count = request
        .negatives
        .map(|b| cached_rows(b, "negatives"))
        .transpose()?
        .unwrap_or(0);

    let meta = RequestMetadata {
        candidate_count: request.candidates.len() as u32,
        positive_count: pos_count,
        negative_count: neg_count,
        k_per_topology: request.config.k_per_topology,
    };

    match classify_request(meta) {
        PreKernelOutcome::TrivialEmpty(result) => Ok(result),
        PreKernelOutcome::Proceed(m) => score_and_reduce(provider, request, m),
    }
}

fn score_and_reduce(
    provider: &CudaKernelProvider,
    request: &InduceExactRequest<'_>,
    meta: RequestMetadata,
) -> Result<ExactInductionResult> {
    // ── Normalize negatives: engine + launcher expect an always-present
    //    buffer. When the caller passes `None`, construct an empty U64 pair
    //    buffer (zero rows) using the positives' schema. This keeps the
    //    launcher signature and kernel signature uniform.
    let empty_neg_holder: Option<CudaBuffer> = if request.negatives.is_none() {
        Some(provider.create_empty_buffer(request.positives.schema().clone())?)
    } else {
        None
    };
    let negatives: &CudaBuffer = match request.negatives {
        Some(b) => b,
        None => empty_neg_holder
            .as_ref()
            .expect("holder populated in the None branch above"),
    };

    // ── Drive the batched scoring kernel and device-side top-K selector.
    let candidate_buffers: Vec<&CudaBuffer> = request.candidates.iter().map(|(_, b)| *b).collect();
    let selected = provider.ilp_exact_score_topk(
        &candidate_buffers,
        request.positives,
        negatives,
        request.config.k_per_topology,
    )?;
    let mut candidates = Vec::with_capacity(selected.len());
    for row in selected {
        let topology = topology_from_kernel_idx(row.topology_idx)?;
        let left_idx = row.left_idx as usize;
        let right_idx = row.right_idx as usize;
        let (left_rel_idx, _) = request.candidates.get(left_idx).ok_or_else(|| {
            XlogError::Execution(format!(
                "induce_exact: device selector returned left index {} for {} candidates",
                left_idx,
                request.candidates.len()
            ))
        })?;
        let (right_rel_idx, _) = request.candidates.get(right_idx).ok_or_else(|| {
            XlogError::Execution(format!(
                "induce_exact: device selector returned right index {} for {} candidates",
                right_idx,
                request.candidates.len()
            ))
        })?;
        candidates.push(ScoredCandidate {
            topology,
            head_rel_idx: request.head_rel_idx,
            left_rel_idx: *left_rel_idx,
            right_rel_idx: *right_rel_idx,
            positives_covered: row.positives_covered,
            negatives_covered: row.negatives_covered,
            local_rank: row.local_rank,
            next_positives_covered: row.next_positives_covered,
            next_negatives_covered: row.next_negatives_covered,
            tie_class_size: row.tie_class_size,
        });
    }
    let total_scored = 4u32
        .checked_mul(meta.candidate_count)
        .and_then(|v| v.checked_mul(meta.candidate_count))
        .ok_or_else(|| XlogError::Execution("induce_exact: total_scored overflow".into()))?;

    Ok(ExactInductionResult {
        candidates,
        total_scored,
        candidate_count: meta.candidate_count,
        positive_count: meta.positive_count,
        negative_count: meta.negative_count,
    })
}

fn topology_from_kernel_idx(idx: u32) -> Result<Topology> {
    match idx {
        0 => Ok(Topology::Chain),
        1 => Ok(Topology::Star),
        2 => Ok(Topology::Fanout),
        3 => Ok(Topology::Fanin),
        _ => Err(XlogError::Execution(format!(
            "induce_exact: device selector returned topology index {}",
            idx
        ))),
    }
}

fn validate_pair_buffer(buf: &CudaBuffer, label: &str) -> Result<ExactPairType> {
    if buf.arity() != 2 {
        return Err(XlogError::Execution(format!(
            "induce_exact: {} buffer has arity {}, expected 2",
            label,
            buf.arity(),
        )));
    }
    let mut pair_type = None;
    for col_idx in 0..2 {
        let t = buf.schema().column_type(col_idx).ok_or_else(|| {
            XlogError::Type(format!(
                "induce_exact: {} buffer column {} has no schema type",
                label, col_idx,
            ))
        })?;
        let col_type = match t {
            ScalarType::U64 => ExactPairType::U64,
            ScalarType::U32 => ExactPairType::U32,
            ScalarType::Symbol => ExactPairType::Symbol,
            _ => {
                return Err(XlogError::Type(format!(
                    "induce_exact: {} buffer column {} has type {:?}, expected U64, U32, or Symbol",
                    label, col_idx, t,
                )));
            }
        };
        if let Some(expected) = pair_type {
            if expected != col_type {
                return Err(XlogError::Type(format!(
                    "induce_exact: {} buffer column {} type mismatch: {:?} vs {:?}",
                    label, col_idx, expected, col_type,
                )));
            }
        } else {
            pair_type = Some(col_type);
        }
    }
    Ok(pair_type.expect("arity 2 loop sets pair type"))
}

fn require_pair_type(buf: &CudaBuffer, label: &str, expected: ExactPairType) -> Result<()> {
    let actual = validate_pair_buffer(buf, label)?;
    if actual != expected {
        return Err(XlogError::Type(format!(
            "induce_exact: {} buffer type mismatch: expected {:?}, got {:?}",
            label, expected, actual,
        )));
    }
    Ok(())
}

fn cached_rows(buf: &CudaBuffer, label: &str) -> Result<u32> {
    buf.cached_row_count().ok_or_else(|| {
        XlogError::Execution(format!(
            "induce_exact: {} buffer has no cached row count \
             (DLPack ingest path should populate it; required to avoid hot-loop D2H)",
            label,
        ))
    })
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
            [
                Topology::Chain,
                Topology::Star,
                Topology::Fanout,
                Topology::Fanin
            ],
        );
    }
}
