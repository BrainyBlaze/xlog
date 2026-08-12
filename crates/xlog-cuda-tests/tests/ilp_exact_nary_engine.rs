//! Engine-level certification for `xlog_induce::induce_exact_nary`.
//!
//! The orchestration-level leg of the n-ary parity chain: the engine
//! enumerates the canonical pattern space, flattens it, scores it through
//! the device-resident launcher, and reduces deterministically — and the
//! result must equal the same pipeline computed with the host reference
//! scorer (`xlog_induce::nary_reference`) pattern-for-pattern, including
//! every diagnostic field. Also pins the engine-level transfer budget:
//! exactly two counter-tracked device-to-host transfers per scored call
//! (the two per-pattern count arrays), zero for trivial early-outs.
//!
//! Skips silently without a GPU; the pod leg runs it for real.

use xlog_core::{RelId, ScalarType, Schema, XlogError};
use xlog_cuda::CudaBuffer;
use xlog_cuda_tests::harness::TestContext;
use xlog_induce::nary_reference::{score_pattern_reference, HostRelation};
use xlog_induce::{
    enumerate_patterns, induce_exact_nary, reduce_nary, InduceExactNaryRequest,
    NaryEnumerationConfig, NaryInductionConfig, NaryInductionResult, ScoredNaryCandidate,
};

fn make_ctx() -> Option<TestContext> {
    match TestContext::new() {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("Skipping ilp_exact_nary_engine test: no CUDA device ({e})");
            None
        }
    }
}

/// Upload host rows as an all-U64 columnar buffer (device-resident).
fn columnar_u64(ctx: &TestContext, rows: &[Vec<u64>], arity: usize) -> CudaBuffer {
    let schema = Schema::new(
        (0..arity)
            .map(|i| (format!("arg{i}"), ScalarType::U64))
            .collect(),
    );
    let n = rows.len() as u32;
    if rows.is_empty() {
        return ctx
            .provider
            .create_empty_buffer(schema)
            .expect("empty buffer");
    }
    let device = ctx.memory.device().inner();
    let mut columns = Vec::with_capacity(arity);
    for col_idx in 0..arity {
        let bytes: Vec<u8> = rows
            .iter()
            .flat_map(|row| {
                assert_eq!(row.len(), arity, "ragged fixture row");
                row[col_idx].to_le_bytes()
            })
            .collect();
        let mut col = ctx.memory.alloc::<u8>(bytes.len()).expect("alloc column");
        device
            .htod_sync_copy_into(&bytes, &mut col)
            .expect("htod column");
        columns.push(col.into());
    }
    let mut d_num_rows = ctx.memory.alloc::<u32>(1).expect("alloc d_num_rows");
    device
        .htod_sync_copy_into(&[n], &mut d_num_rows)
        .expect("htod d_num_rows");
    CudaBuffer::from_columns_with_host_count(columns, n as u64, d_num_rows, schema, n)
}

/// Compute the expected engine result entirely on host: same enumeration,
/// reference scorer per pattern, same deterministic reduction.
fn expected_result(
    head_rel_idx: RelId,
    candidates: &[(RelId, HostRelation, u8)],
    positives: &[Vec<u64>],
    negatives: &[Vec<u64>],
    config: &NaryInductionConfig,
) -> NaryInductionResult {
    let head_arity = positives[0].len() as u8;
    let arities: Vec<u8> = candidates.iter().map(|(_, _, a)| *a).collect();
    let host_rels: Vec<HostRelation> = candidates.iter().map(|(_, r, _)| r.clone()).collect();
    let patterns = enumerate_patterns(head_arity, &arities, &config.enumeration)
        .expect("reference enumeration");
    let coverage: Vec<(u32, u32)> = patterns
        .iter()
        .map(|p| {
            let c = score_pattern_reference(p, &host_rels, positives, negatives);
            (c.positives_covered, c.negatives_covered)
        })
        .collect();
    let kept = reduce_nary(&coverage, config.k);
    let scored = kept
        .into_iter()
        .map(|k| {
            let pattern = patterns[k.pattern_idx].clone();
            let body_rel_idxs = pattern
                .body
                .iter()
                .map(|atom| candidates[atom.candidate_slot as usize].0)
                .collect();
            ScoredNaryCandidate {
                head_rel_idx,
                body_rel_idxs,
                pattern,
                positives_covered: k.positives_covered,
                negatives_covered: k.negatives_covered,
                local_rank: k.local_rank,
                next_positives_covered: k.next_positives_covered,
                next_negatives_covered: k.next_negatives_covered,
                tie_class_size: k.tie_class_size,
            }
        })
        .collect();
    NaryInductionResult {
        candidates: scored,
        total_scored: patterns.len() as u32,
        candidate_count: candidates.len() as u32,
        positive_count: positives.len() as u32,
        negative_count: negatives.len() as u32,
    }
}

fn rel(rows: &[&[u64]]) -> HostRelation {
    HostRelation {
        rows: rows.iter().map(|r| r.to_vec()).collect(),
    }
}

/// Binary chain fixture through the FULL n-ary engine: the reference
/// pipeline and the device pipeline must agree on the complete result,
/// and the top pattern must be the two-positive/zero-negative chain.
#[test]
fn engine_matches_reference_over_binary_chain_space() {
    let Some(ctx) = make_ctx() else {
        return;
    };
    let p_b = rel(&[&[1, 2], &[2, 3]]);
    let p_c = rel(&[&[2, 4], &[3, 5], &[4, 6]]);
    let positives: Vec<Vec<u64>> = vec![vec![1, 4], vec![2, 5]];
    let negatives: Vec<Vec<u64>> = vec![vec![7, 8]];
    let config = NaryInductionConfig {
        enumeration: NaryEnumerationConfig {
            max_body_atoms: 2,
            max_join_vars: 1,
            max_patterns: 10_000,
        },
        k: 5,
    };
    let host_candidates = [(RelId(10), p_b.clone(), 2u8), (RelId(11), p_c.clone(), 2u8)];
    let expected = expected_result(RelId(7), &host_candidates, &positives, &negatives, &config);
    assert!(
        expected
            .candidates
            .first()
            .is_some_and(|c| c.positives_covered == 2 && c.negatives_covered == 0),
        "fixture sanity: the chain pattern must top the reference ranking",
    );

    let b_buf = columnar_u64(&ctx, &p_b.rows, 2);
    let c_buf = columnar_u64(&ctx, &p_c.rows, 2);
    let pos_buf = columnar_u64(&ctx, &positives, 2);
    let neg_buf = columnar_u64(&ctx, &negatives, 2);
    let request = InduceExactNaryRequest {
        head_rel_idx: RelId(7),
        candidates: &[(RelId(10), &b_buf), (RelId(11), &c_buf)],
        positives: &pos_buf,
        negatives: Some(&neg_buf),
        config,
    };

    ctx.provider.reset_d2h_transfer_count();
    let result = induce_exact_nary(&ctx.provider, &request).expect("engine run");
    assert_eq!(
        ctx.provider.d2h_transfer_count(),
        2,
        "engine transfer budget: exactly the two count arrays",
    );
    assert_eq!(result, expected);
}

/// Ternary head through the engine with `negatives: None` (the engine
/// materializes the empty negatives buffer itself): full reference parity
/// and the same two-transfer budget.
#[test]
fn engine_matches_reference_over_ternary_space() {
    let Some(ctx) = make_ctx() else {
        return;
    };
    let t = rel(&[&[1, 2, 9], &[4, 5, 8]]);
    let p = rel(&[&[9, 3], &[8, 7]]);
    let positives: Vec<Vec<u64>> = vec![vec![1, 2, 3], vec![4, 5, 6], vec![1, 2, 7]];
    let config = NaryInductionConfig {
        enumeration: NaryEnumerationConfig {
            max_body_atoms: 2,
            max_join_vars: 2,
            max_patterns: 1_000_000,
        },
        k: 3,
    };
    let host_candidates = [(RelId(20), t.clone(), 3u8), (RelId(21), p.clone(), 2u8)];
    let expected = expected_result(RelId(5), &host_candidates, &positives, &[], &config);

    let t_buf = columnar_u64(&ctx, &t.rows, 3);
    let p_buf = columnar_u64(&ctx, &p.rows, 2);
    let pos_buf = columnar_u64(&ctx, &positives, 3);
    let request = InduceExactNaryRequest {
        head_rel_idx: RelId(5),
        candidates: &[(RelId(20), &t_buf), (RelId(21), &p_buf)],
        positives: &pos_buf,
        negatives: None,
        config,
    };

    ctx.provider.reset_d2h_transfer_count();
    let result = induce_exact_nary(&ctx.provider, &request).expect("engine run");
    assert_eq!(ctx.provider.d2h_transfer_count(), 2);
    assert_eq!(result, expected);
}

/// Trivial early-outs never launch and never transfer; refusals are typed.
#[test]
fn engine_trivial_paths_and_typed_refusals() {
    let Some(ctx) = make_ctx() else {
        return;
    };
    let facts = rel(&[&[1, 2], &[2, 3]]);
    let fact_buf = columnar_u64(&ctx, &facts.rows, 2);
    let pos_buf = columnar_u64(&ctx, &[vec![1, 3]], 2);
    let config = NaryInductionConfig {
        enumeration: NaryEnumerationConfig {
            max_body_atoms: 2,
            max_join_vars: 1,
            max_patterns: 10_000,
        },
        k: 2,
    };

    // Empty candidate list: the fully-default result, no CUDA work.
    ctx.provider.reset_d2h_transfer_count();
    let result = induce_exact_nary(
        &ctx.provider,
        &InduceExactNaryRequest {
            head_rel_idx: RelId(7),
            candidates: &[],
            positives: &pos_buf,
            negatives: None,
            config,
        },
    )
    .expect("empty candidates");
    assert_eq!(result, NaryInductionResult::default());
    assert_eq!(ctx.provider.d2h_transfer_count(), 0);

    // k == 0: counts populated, nothing scored, no transfers.
    let result = induce_exact_nary(
        &ctx.provider,
        &InduceExactNaryRequest {
            head_rel_idx: RelId(7),
            candidates: &[(RelId(10), &fact_buf)],
            positives: &pos_buf,
            negatives: None,
            config: NaryInductionConfig { k: 0, ..config },
        },
    )
    .expect("k = 0");
    assert!(result.candidates.is_empty());
    assert_eq!(result.total_scored, 0);
    assert_eq!(result.candidate_count, 1);
    assert_eq!(result.positive_count, 1);
    assert_eq!(ctx.provider.d2h_transfer_count(), 0);

    // Zero positive rows: coverage can never exceed zero, no launch.
    let empty_pos = columnar_u64(&ctx, &[], 2);
    let result = induce_exact_nary(
        &ctx.provider,
        &InduceExactNaryRequest {
            head_rel_idx: RelId(7),
            candidates: &[(RelId(10), &fact_buf)],
            positives: &empty_pos,
            negatives: None,
            config,
        },
    )
    .expect("no positives");
    assert!(result.candidates.is_empty());
    assert_eq!(result.total_scored, 0);
    assert_eq!(ctx.provider.d2h_transfer_count(), 0);

    // Negatives arity mismatch: typed refusal.
    let neg_ternary = columnar_u64(&ctx, &[vec![1, 2, 3]], 3);
    let err = induce_exact_nary(
        &ctx.provider,
        &InduceExactNaryRequest {
            head_rel_idx: RelId(7),
            candidates: &[(RelId(10), &fact_buf)],
            positives: &pos_buf,
            negatives: Some(&neg_ternary),
            config,
        },
    )
    .expect_err("arity mismatch must refuse");
    assert!(matches!(err, XlogError::Type(_)), "typed refusal: {err:?}");

    // Non-U64 candidate column: typed refusal before any enumeration.
    let u32_schema = Schema::new(vec![
        ("arg0".to_string(), ScalarType::U32),
        ("arg1".to_string(), ScalarType::U32),
    ]);
    let u32_buf = ctx
        .provider
        .create_empty_buffer(u32_schema)
        .expect("u32 buffer");
    let err = induce_exact_nary(
        &ctx.provider,
        &InduceExactNaryRequest {
            head_rel_idx: RelId(7),
            candidates: &[(RelId(10), &u32_buf)],
            positives: &pos_buf,
            negatives: None,
            config,
        },
    )
    .expect_err("non-U64 candidate must refuse");
    assert!(matches!(err, XlogError::Type(_)), "typed refusal: {err:?}");

    // Pattern-space blow-up: typed refusal, never a silent cap.
    let err = induce_exact_nary(
        &ctx.provider,
        &InduceExactNaryRequest {
            head_rel_idx: RelId(7),
            candidates: &[(RelId(10), &fact_buf)],
            positives: &pos_buf,
            negatives: None,
            config: NaryInductionConfig {
                enumeration: NaryEnumerationConfig {
                    max_patterns: 1,
                    ..config.enumeration
                },
                ..config
            },
        },
    )
    .expect_err("pattern cap must refuse");
    // The enumerator's cap law refuses as Execution (never a silent cap).
    assert!(
        matches!(err, XlogError::Execution(_)),
        "typed refusal: {err:?}"
    );
}
