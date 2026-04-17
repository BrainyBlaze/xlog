//! Pre-kernel request classification — pure host-side helpers.
//!
//! Separates the numeric early-return logic (empty candidates, zero positives)
//! from CUDA-dependent buffer validation so the dead-end cases can be unit
//! tested without a GPU. Behaviorally mirrors the early returns in the
//! Python reference (`crates/pyxlog/python/pyxlog/ilp/exact_induce.py`).

use crate::types::ExactInductionResult;

/// Request metadata extracted after buffer-level validation passes.
/// Pure host-side; contains no CUDA types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RequestMetadata {
    pub candidate_count: u32,
    pub positive_count: u32,
    pub negative_count: u32,
    pub k_per_topology: u32,
}

/// Outcome of classifying a request against trivial dead-ends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PreKernelOutcome {
    /// Return this result directly — skip the scoring kernel.
    TrivialEmpty(ExactInductionResult),
    /// Proceed to the scoring kernel with this metadata.
    Proceed(RequestMetadata),
}

/// Classify against trivial dead-ends, matching the Python reference's early
/// returns:
///   * no candidates   → all-zero `ExactInductionResult`
///   * no positives    → result with zero positives + retained neg/candidate counts
///   * otherwise       → proceed with the extracted metadata
pub(crate) fn classify_request(meta: RequestMetadata) -> PreKernelOutcome {
    if meta.candidate_count == 0 {
        return PreKernelOutcome::TrivialEmpty(ExactInductionResult::default());
    }
    if meta.positive_count == 0 {
        return PreKernelOutcome::TrivialEmpty(ExactInductionResult {
            candidates: Vec::new(),
            total_scored: 0,
            candidate_count: meta.candidate_count,
            positive_count: 0,
            negative_count: meta.negative_count,
        });
    }
    PreKernelOutcome::Proceed(meta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_empty_candidates_returns_default_result() {
        let meta = RequestMetadata {
            candidate_count: 0,
            positive_count: 10,
            negative_count: 5,
            k_per_topology: 2,
        };
        match classify_request(meta) {
            PreKernelOutcome::TrivialEmpty(r) => {
                // All-zero Default — pos/neg counts are discarded when
                // there are no candidates, matching the Python reference.
                assert!(r.candidates.is_empty());
                assert_eq!(r.total_scored, 0);
                assert_eq!(r.candidate_count, 0);
                assert_eq!(r.positive_count, 0);
                assert_eq!(r.negative_count, 0);
            }
            PreKernelOutcome::Proceed(_) => panic!("expected TrivialEmpty"),
        }
    }

    #[test]
    fn classify_zero_positives_preserves_neg_and_candidate_counts() {
        let meta = RequestMetadata {
            candidate_count: 3,
            positive_count: 0,
            negative_count: 7,
            k_per_topology: 2,
        };
        match classify_request(meta) {
            PreKernelOutcome::TrivialEmpty(r) => {
                // When positives are zero the Python ref still reports the
                // candidate and negative counts so callers can log "we saw
                // facts but had nothing to score."
                assert!(r.candidates.is_empty());
                assert_eq!(r.total_scored, 0);
                assert_eq!(r.candidate_count, 3);
                assert_eq!(r.positive_count, 0);
                assert_eq!(r.negative_count, 7);
            }
            PreKernelOutcome::Proceed(_) => panic!("expected TrivialEmpty"),
        }
    }

    #[test]
    fn classify_normal_case_returns_proceed_with_metadata() {
        let meta = RequestMetadata {
            candidate_count: 4,
            positive_count: 12,
            negative_count: 2,
            k_per_topology: 2,
        };
        match classify_request(meta) {
            PreKernelOutcome::Proceed(m) => assert_eq!(m, meta),
            PreKernelOutcome::TrivialEmpty(_) => panic!("expected Proceed"),
        }
    }

    #[test]
    fn classify_positives_only_without_negatives_still_proceeds() {
        let meta = RequestMetadata {
            candidate_count: 2,
            positive_count: 5,
            negative_count: 0,
            k_per_topology: 1,
        };
        match classify_request(meta) {
            PreKernelOutcome::Proceed(m) => assert_eq!(m.negative_count, 0),
            PreKernelOutcome::TrivialEmpty(_) => panic!("expected Proceed"),
        }
    }

    #[test]
    fn classify_empty_candidates_takes_precedence_over_zero_positives() {
        // Both zero — candidates check fires first, giving all-zero result
        // (the Python ref also returns the all-zero default here).
        let meta = RequestMetadata {
            candidate_count: 0,
            positive_count: 0,
            negative_count: 3,
            k_per_topology: 2,
        };
        match classify_request(meta) {
            PreKernelOutcome::TrivialEmpty(r) => {
                assert_eq!(r.candidate_count, 0);
                assert_eq!(r.negative_count, 0); // discarded, all-zero default
            }
            PreKernelOutcome::Proceed(_) => panic!("expected TrivialEmpty"),
        }
    }
}
