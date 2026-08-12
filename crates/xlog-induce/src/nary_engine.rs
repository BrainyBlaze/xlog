//! Engine orchestration for bounded exact n-ary induction.
//!
//! The n-ary counterpart of the binary [`crate::induce_exact`] pipeline:
//! enumerate every canonical pattern ([`crate::nary::enumerate_patterns`]),
//! flatten the batch into the device layout
//! ([`crate::nary_layout::flatten_patterns`]), score it through the
//! device-resident launcher
//! ([`CudaKernelProvider::ilp_exact_nary_score_device`]), and reduce the
//! per-pattern coverage counts deterministically
//! ([`crate::reduce::reduce_nary`]).
//!
//! Transfer budget: relation and example values never visit the host —
//! ingestion is device-to-device column copies inside the launcher; row
//! counts come from cached host-side metadata (a pure struct read); the
//! only device-to-host transfers are the two per-pattern count arrays.
//! The reduction then runs on those counts entirely host-side.

use xlog_core::{RelId, Result, ScalarType, XlogError};
use xlog_cuda::{CudaBuffer, CudaKernelProvider, IlpExactNaryPatterns};

use crate::nary::{enumerate_patterns, NaryEnumerationConfig, NaryRulePattern};
use crate::nary_layout::{flatten_patterns, NARY_MAX_ATOM_ARITY};
use crate::reduce::reduce_nary;

/// Head arity bound of the device contract: the kernel gathers one example
/// tuple into a fixed per-thread array of this size, and the launcher
/// refuses wider heads fail-closed. Mirrored here so the engine refuses
/// with a typed error before enumerating a batch no kernel could score.
const NARY_MAX_HEAD_ARITY: usize = 8;

/// Bounds and selection size for one [`induce_exact_nary`] call.
#[derive(Debug, Clone, Copy)]
pub struct NaryInductionConfig {
    /// Pattern-space bounds (body atoms, join variables, hard pattern cap).
    pub enumeration: NaryEnumerationConfig,
    /// Number of top-ranked patterns to keep.
    pub k: u32,
}

/// Inputs to one [`induce_exact_nary`] call.
///
/// Each candidate is a `(RelId, &CudaBuffer)` pair exactly as in the binary
/// engine: the `RelId` is a label that flows through to the scored output,
/// and the buffer carries the relation's facts as all-`U64` columns. The
/// head arity is the arity of `positives`; `negatives`, when present, must
/// match it. Name resolution and relation-store lookup happen at the
/// pyxlog boundary — the engine only sees indices + handles.
pub struct InduceExactNaryRequest<'a> {
    pub head_rel_idx: RelId,
    pub candidates: &'a [(RelId, &'a CudaBuffer)],
    pub positives: &'a CudaBuffer,
    pub negatives: Option<&'a CudaBuffer>,
    pub config: NaryInductionConfig,
}

/// One kept pattern with full metadata and diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScoredNaryCandidate {
    pub head_rel_idx: RelId,
    /// Relation identity of each body atom, in body order — the
    /// `candidate_slot` of the corresponding [`NaryRulePattern`] atom
    /// resolved against the request's candidate list.
    pub body_rel_idxs: Vec<RelId>,
    /// The full canonical pattern (bindings included), for rule assembly.
    pub pattern: NaryRulePattern,
    pub positives_covered: u32,
    pub negatives_covered: u32,
    pub local_rank: u32,
    pub next_positives_covered: u32,
    pub next_negatives_covered: u32,
    pub tie_class_size: u32,
}

/// Result of one [`induce_exact_nary`] call.
///
/// `total_scored` counts patterns actually scored by the kernel; the
/// trivial early-outs (no candidates, no patterns, `k == 0`, no positive
/// examples) report `0` and an empty candidate list.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NaryInductionResult {
    pub candidates: Vec<ScoredNaryCandidate>,
    pub total_scored: u32,
    pub candidate_count: u32,
    pub positive_count: u32,
    pub negative_count: u32,
}

/// Run exact n-ary induction against one request.
///
/// The `provider` owns the kernel launcher and is also used to
/// materialize the short-lived empty negatives buffer when
/// `request.negatives` is `None` — the same normalization the binary
/// engine performs.
///
/// Diagnostic asymmetry, stated rather than left to be discovered: the
/// EMPTY-CANDIDATES early-out returns a fully default result (zero
/// example counts) because it precedes buffer validation — with no
/// candidate there is nothing whose arity the examples must agree with.
/// Every other trivial early-out (`k == 0`, zero positives) validates
/// first and therefore reports the real `positive_count` /
/// `negative_count`.
pub fn induce_exact_nary(
    provider: &CudaKernelProvider,
    request: &InduceExactNaryRequest<'_>,
) -> Result<NaryInductionResult> {
    // Empty candidates is a trivial dead-end and needs no CUDA inspection.
    if request.candidates.is_empty() {
        return Ok(NaryInductionResult::default());
    }

    // ── Buffer validation (typed, before any enumeration cost). The
    //    launcher re-validates fail-closed — this pass exists to fail loud
    //    on pyxlog-side assembly bugs with engine-shaped errors.
    let (head_arity_u32, pos_count) = require_u64_columnar(request.positives, "positives")?;
    let head_arity = head_arity_u32 as usize;
    if head_arity == 0 || head_arity > NARY_MAX_HEAD_ARITY {
        return Err(XlogError::Type(format!(
            "induce_exact_nary: positives arity {head_arity} outside supported \
             head arity 1..={NARY_MAX_HEAD_ARITY}",
        )));
    }
    let neg_count = match request.negatives {
        Some(neg) => {
            let (neg_arity, neg_count) = require_u64_columnar(neg, "negatives")?;
            if neg_arity != head_arity_u32 {
                return Err(XlogError::Type(format!(
                    "induce_exact_nary: negatives arity {neg_arity} != head arity {head_arity}",
                )));
            }
            neg_count
        }
        None => 0,
    };
    let mut candidate_arities: Vec<u8> = Vec::with_capacity(request.candidates.len());
    for (i, (_, buf)) in request.candidates.iter().enumerate() {
        let (arity, _) = require_u64_columnar(buf, &format!("candidate[{i}]"))?;
        if arity == 0 || arity as usize > NARY_MAX_ATOM_ARITY {
            return Err(XlogError::Type(format!(
                "induce_exact_nary: candidate[{i}] arity {arity} outside supported \
                 atom arity 1..={NARY_MAX_ATOM_ARITY}",
            )));
        }
        candidate_arities.push(arity as u8);
    }
    let candidate_count = request.candidates.len() as u32;

    // ── Enumerate the canonical pattern space (typed refusal on blow-up,
    //    never a silent cap).
    let patterns = enumerate_patterns(
        head_arity as u8,
        &candidate_arities,
        &request.config.enumeration,
    )?;

    let counts_only = NaryInductionResult {
        candidates: Vec::new(),
        total_scored: 0,
        candidate_count,
        positive_count: pos_count,
        negative_count: neg_count,
    };
    if patterns.is_empty() || request.config.k == 0 || pos_count == 0 {
        // No pattern can be kept: nothing to enumerate, nothing requested,
        // or no positive example can ever push coverage above zero. All
        // three are provable host-side without a launch.
        return Ok(counts_only);
    }

    // ── Flatten into the device layout. The enumeration already
    //    guarantees canonical form; this refuses only device-bound
    //    violations (a config wider than the kernel's fixed state).
    let layout = flatten_patterns(&patterns)
        .map_err(|e| XlogError::Type(format!("induce_exact_nary: flatten: {e}")))?;
    let patterns_view = IlpExactNaryPatterns {
        body_offset: &layout.body_offset,
        body_len: &layout.body_len,
        atom_candidate_slot: &layout.atom_candidate_slot,
        atom_arity: &layout.atom_arity,
        atom_binding_offset: &layout.atom_binding_offset,
        binding_codes: &layout.binding_codes,
        head_arity: head_arity_u32,
    };

    // ── Normalize negatives: the launcher expects an always-present
    //    buffer. Same construction as the binary engine — an empty buffer
    //    with the positives' schema.
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

    // ── Score on device (D2D ingest inside; the two count arrays are the
    //    only D2H) and reduce deterministically on the counts.
    let candidate_buffers: Vec<&CudaBuffer> = request.candidates.iter().map(|(_, b)| *b).collect();
    let (pos_covered, neg_covered) = provider.ilp_exact_nary_score_device(
        &patterns_view,
        &candidate_buffers,
        request.positives,
        negatives,
    )?;
    if pos_covered.len() != patterns.len() || neg_covered.len() != patterns.len() {
        return Err(XlogError::Execution(format!(
            "induce_exact_nary: launcher returned {}/{} counts for {} patterns",
            pos_covered.len(),
            neg_covered.len(),
            patterns.len(),
        )));
    }
    let coverage: Vec<(u32, u32)> = pos_covered.into_iter().zip(neg_covered).collect();
    let kept = reduce_nary(&coverage, request.config.k);

    let mut candidates = Vec::with_capacity(kept.len());
    for keep in kept {
        let pattern = patterns
            .get(keep.pattern_idx)
            .ok_or_else(|| {
                XlogError::Execution(format!(
                    "induce_exact_nary: reduction returned pattern index {} for {} patterns",
                    keep.pattern_idx,
                    patterns.len(),
                ))
            })?
            .clone();
        let body_rel_idxs = pattern
            .body
            .iter()
            .map(|atom| request.candidates[atom.candidate_slot as usize].0)
            .collect();
        candidates.push(ScoredNaryCandidate {
            head_rel_idx: request.head_rel_idx,
            body_rel_idxs,
            pattern,
            positives_covered: keep.positives_covered,
            negatives_covered: keep.negatives_covered,
            local_rank: keep.local_rank,
            next_positives_covered: keep.next_positives_covered,
            next_negatives_covered: keep.next_negatives_covered,
            tie_class_size: keep.tie_class_size,
        });
    }

    Ok(NaryInductionResult {
        candidates,
        total_scored: patterns.len() as u32,
        candidate_count,
        positive_count: pos_count,
        negative_count: neg_count,
    })
}

fn require_u64_columnar(buf: &CudaBuffer, label: &str) -> Result<(u32, u32)> {
    let arity = buf.arity();
    for col_idx in 0..arity {
        let t = buf.schema().column_type(col_idx).ok_or_else(|| {
            XlogError::Type(format!(
                "induce_exact_nary: {label} buffer column {col_idx} has no schema type",
            ))
        })?;
        if t != ScalarType::U64 {
            return Err(XlogError::Type(format!(
                "induce_exact_nary: {label} buffer column {col_idx} has type {t:?}, expected U64",
            )));
        }
    }
    let rows = buf.cached_row_count().ok_or_else(|| {
        XlogError::Execution(format!(
            "induce_exact_nary: {label} buffer has no cached row count \
             (device-resident ingest should populate it; required to avoid \
             a hot-loop device-to-host transfer)",
        ))
    })?;
    let arity = u32::try_from(arity)
        .map_err(|_| XlogError::Type(format!("induce_exact_nary: {label} arity exceeds u32")))?;
    Ok((arity, rows))
}
