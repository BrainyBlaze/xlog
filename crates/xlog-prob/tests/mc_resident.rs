//! GPU-resident Datalog/MC engine: no-host instrumentation + exact-value pilots.
//!
//! These are GPU-native acceptance tests. The engine evaluates ALL worlds in a
//! single megakernel launch; the measured region must show zero host
//! interaction (no tracked transfers, no untracked metadata reads, no host
//! sample loop, no per-sample host launches) — and those counters must be
//! CONSTANT across sample counts, which is the strongest evidence that nothing
//! scales per-sample on the host.

mod common;
use common::setup_provider;

use cudarc::driver::DeviceSlice;
use xlog_prob::mc::{
    compile_resident_plan, McEvalConfig, McNoHostStats, McProgram, McResidentResult,
    McSamplingMethod, ResidentRejectKind,
};

fn cfg(samples: usize, seed: u64) -> McEvalConfig {
    let mut c = McEvalConfig::default();
    c.samples = samples;
    c.seed = seed;
    c.confidence = 0.95;
    c.max_nonmonotone_iterations = 64;
    c
}

/// Download device-resident counts AFTER the measured region (test-side).
fn download(
    provider: &std::sync::Arc<xlog_cuda::CudaKernelProvider>,
    r: &McResidentResult,
) -> (Vec<u32>, u32) {
    let mut counts = vec![0u32; r.query_counts.len()];
    if !counts.is_empty() {
        provider
            .device()
            .inner()
            .dtoh_sync_copy_into(&r.query_counts, &mut counts)
            .expect("dtoh query counts");
    }
    let mut ev = [0u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&r.evidence_count, &mut ev)
        .expect("dtoh evidence count");
    (counts, ev[0])
}

/// Per-world fixpoint iteration trace (test-side, after region).
fn download_iters(
    provider: &std::sync::Arc<xlog_cuda::CudaKernelProvider>,
    r: &McResidentResult,
) -> Vec<u32> {
    let mut v = vec![0u32; r.iter_trace.len()];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&r.iter_trace, &mut v)
        .expect("dtoh iter trace");
    v
}

/// Device-populated sparse row counts and offsets, downloaded after the measured
/// region. These are acceptance evidence that the resident path maintains a
/// world-segmented sparse relation arena in addition to dense membership bits.
fn download_sparse_trace(
    provider: &std::sync::Arc<xlog_cuda::CudaKernelProvider>,
    r: &McResidentResult,
) -> (Vec<u32>, Vec<u32>) {
    let mut counts = vec![0u32; r.sparse_final_row_counts.len()];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&r.sparse_final_row_counts, &mut counts)
        .expect("dtoh sparse final row counts");
    let mut offsets = vec![0u32; r.sparse_offsets.len()];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&r.sparse_offsets, &mut offsets)
        .expect("dtoh sparse offsets");
    (counts, offsets)
}

/// Device-populated convergence and sparse-arena overflow diagnostics,
/// downloaded after the measured region. Convergence and overflow state must be
/// kernel-written, not inferred from host loop control.
fn download_resident_diagnostics(
    provider: &std::sync::Arc<xlog_cuda::CudaKernelProvider>,
    r: &McResidentResult,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut flags = vec![0u32; r.resident_status_flags.len()];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&r.resident_status_flags, &mut flags)
        .expect("dtoh resident status flags");
    let worlds = r.total_samples;
    let converged = flags[0..worlds].to_vec();
    let overflow = flags[worlds..worlds * 2].to_vec();
    let participation = flags[worlds * 2..worlds * 3].to_vec();
    (converged, overflow, participation)
}

/// K1 walking-skeleton proof: fact-marginal, no rules. The measured region has
/// zero host interaction, and every in-region counter is identical at N=128 and
/// N=1024 (constant in N => nothing is per-sample on the host).
#[test]
fn resident_fact_marginal_no_host_constant_in_n() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let program = McProgram::compile_source(
        r#"
1.0::coin().
query(coin()).
"#,
    )
    .expect("compile fact-marginal");

    let r128 = program
        .evaluate_resident_with_provider(cfg(128, 7), provider.clone())
        .expect("resident 128");
    let r1024 = program
        .evaluate_resident_with_provider(cfg(1024, 7), provider.clone())
        .expect("resident 1024");

    // K1: no host interaction in the measured region.
    assert!(r128.no_host.is_no_host(), "N=128: {:?}", r128.no_host);
    assert!(r1024.no_host.is_no_host(), "N=1024: {:?}", r1024.no_host);

    // Exactly one global engine launch (not per-sample).
    assert_eq!(r128.no_host.engine_launches, 1);
    assert_eq!(r1024.no_host.engine_launches, 1);

    // Strongest evidence: the in-region instrumentation is CONSTANT in N.
    let expected = McNoHostStats {
        tracked_htod_calls: 0,
        tracked_dtoh_calls: 0,
        untracked_metadata_reads: 0,
        engine_launches: 1,
        host_loop_iterations: 0,
        per_sample_host_launches: 0,
        host_fixpoint_iterations: 0,
        per_operator_host_allocations: 0,
    };
    assert_eq!(r128.no_host, expected, "N=128 in-region stats");
    assert_eq!(r1024.no_host, expected, "N=1024 in-region stats");
    assert_eq!(
        r128.no_host, r1024.no_host,
        "in-region host interaction must not scale with N"
    );

    // Exact counts (1.0::coin() holds in every world; no evidence => all count).
    let (counts, evidence) = download(&provider, &r1024);
    assert_eq!(evidence as usize, 1024, "no-evidence => all worlds count");
    assert_eq!(counts.len(), 1);
    assert_eq!(counts[0] as usize, 1024, "1.0::coin() true in all worlds");
}

/// Seed-matched CPU oracle count for query `qi` (answer-key only; NOT no-host
/// evidence — the oracle runs entirely on the host).
fn oracle_count(program: &McProgram, c: McEvalConfig, qi: usize) -> usize {
    let r = program.evaluate_cpu(c).expect("cpu oracle");
    (r.query_estimates[qi].prob * r.evidence_samples as f64).round() as usize
}

/// Pilot 2 — probabilistic fact marginal, exact vs seeded oracle.
#[test]
fn resident_pilot_prob_fact_marginal_exact() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };
    let program = McProgram::compile_source("0.5::a().\nquery(a()).\n").expect("compile");
    let c = cfg(1000, 17);
    let want = oracle_count(&program, c.clone(), 0);
    let r = program
        .evaluate_resident_with_provider(c.clone(), provider.clone())
        .expect("resident");
    assert!(r.no_host.is_no_host(), "{:?}", r.no_host);
    let (counts, ev) = download(&provider, &r);
    assert_eq!(ev as usize, c.samples);
    assert_eq!(counts[0] as usize, want, "device count == seeded oracle");
}

/// Pilot 3 — evidence conditioning (clamped), exact vs seeded oracle.
#[test]
fn resident_pilot_evidence_conditioning_exact() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };
    let program =
        McProgram::compile_source("0.5::a().\n0.5::b().\nevidence(a(), true).\nquery(b()).\n")
            .expect("compile");
    let c = cfg(1000, 17);
    let want = oracle_count(&program, c.clone(), 0);
    let r = program
        .evaluate_resident_with_provider(c.clone(), provider.clone())
        .expect("resident");
    assert!(r.no_host.is_no_host(), "{:?}", r.no_host);
    assert_eq!(r.sampling_method, McSamplingMethod::EvidenceClamping);
    let (counts, ev) = download(&provider, &r);
    assert_eq!(ev as usize, c.samples, "clamped => evidence == samples");
    assert_eq!(counts[0] as usize, want, "device b-count == seeded oracle");
}

/// Pilot 4 — multi-evidence (two evidence atoms), exact vs seeded oracle.
#[test]
fn resident_pilot_multi_evidence_exact() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };
    let program = McProgram::compile_source(
        "0.5::a().\n0.5::b().\n0.5::c().\nevidence(a(), true).\nevidence(b(), true).\nquery(c()).\n",
    )
    .expect("compile");
    let c = cfg(2000, 29);
    let want = oracle_count(&program, c.clone(), 0);
    let r = program
        .evaluate_resident_with_provider(c.clone(), provider.clone())
        .expect("resident");
    assert!(r.no_host.is_no_host(), "{:?}", r.no_host);
    let (counts, ev) = download(&provider, &r);
    assert_eq!(
        ev as usize, c.samples,
        "clamped multi-evidence => evidence == samples"
    );
    assert_eq!(counts[0] as usize, want, "device c-count == seeded oracle");
}

/// Pilot 4b — annotated disjunction / exclusive choice. Exactly one of x()/y()
/// holds per world (exact RNG-independent invariant: count(x)+count(y) == N),
/// and each individual count matches the seeded oracle.
#[test]
fn resident_pilot_annotated_disjunction_exclusive() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };
    let program = McProgram::compile_source("0.6::x(); 0.4::y().\nquery(x()).\nquery(y()).\n")
        .expect("compile");
    let c = cfg(1000, 5);
    let ox = oracle_count(&program, c.clone(), 0);
    let oy = oracle_count(&program, c.clone(), 1);
    let r = program
        .evaluate_resident_with_provider(c.clone(), provider.clone())
        .expect("resident");
    assert!(r.no_host.is_no_host(), "{:?}", r.no_host);
    let (counts, _ev) = download(&provider, &r);
    assert_eq!(counts.len(), 2);
    assert_eq!(
        counts[0] as usize + counts[1] as usize,
        c.samples,
        "exclusive choice: exactly one of x()/y() per world"
    );
    assert_eq!(counts[0] as usize, ox, "x() device count == seeded oracle");
    assert_eq!(counts[1] as usize, oy, "y() device count == seeded oracle");
}

/// Pilot 5 — recursive transitive closure (K3: non-base derived tuple,
/// K4: device-side fixpoint trace shows >1 iteration). Deterministic (1.0 edges).
#[test]
fn resident_pilot_transitive_closure_recursion() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };
    let program = McProgram::compile_source(
        r#"
1.0::edge(1, 2).
1.0::edge(2, 3).
reach(X, Y) :- edge(X, Y).
reach(X, Z) :- reach(X, Y), edge(Y, Z).
query(reach(1, 3)).
query(reach(1, 2)).
"#,
    )
    .expect("compile");
    let c = cfg(128, 3);
    let r = program
        .evaluate_resident_with_provider(c.clone(), provider.clone())
        .expect("resident");
    assert!(r.no_host.is_no_host(), "{:?}", r.no_host);

    let (counts, ev) = download(&provider, &r);
    assert_eq!(ev as usize, c.samples);
    // K3: reach(1,3) is a NON-base derived tuple (needs reach(1,2) ⋈ edge(2,3)).
    assert_eq!(
        counts[0] as usize, c.samples,
        "reach(1,3) derived in all worlds"
    );
    assert_eq!(
        counts[1] as usize, c.samples,
        "reach(1,2) holds in all worlds"
    );

    // K4: device-side fixpoint trace recorded a multi-pass derivation.
    let iters = download_iters(&provider, &r);
    assert!(
        iters.iter().all(|&i| i >= 2),
        "transitive closure must take >1 fixpoint pass, got {:?}",
        &iters[..iters.len().min(4)]
    );
}

/// Pilot 6 — recursion genuinely requiring >1 iteration: a 4-node chain whose
/// longest derived path (reach(1,4)) cannot appear before the 2nd pass.
#[test]
fn resident_pilot_recursion_requires_multiple_iterations() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };
    let program = McProgram::compile_source(
        r#"
1.0::edge(1, 2).
1.0::edge(2, 3).
1.0::edge(3, 4).
reach(X, Y) :- edge(X, Y).
reach(X, Z) :- reach(X, Y), edge(Y, Z).
query(reach(1, 4)).
query(reach(1, 3)).
"#,
    )
    .expect("compile");
    let c = cfg(128, 5);
    let r = program
        .evaluate_resident_with_provider(c.clone(), provider.clone())
        .expect("resident");
    assert!(r.no_host.is_no_host(), "{:?}", r.no_host);

    let (counts, _ev) = download(&provider, &r);
    // reach(1,4) needs three hops (1→2→3→4): a strictly multi-iteration derivation.
    assert_eq!(
        counts[0] as usize, c.samples,
        "reach(1,4) derived in all worlds"
    );
    assert_eq!(
        counts[1] as usize, c.samples,
        "reach(1,3) derived in all worlds"
    );

    let iters = download_iters(&provider, &r);
    // Naive fixpoint: hop k appears at pass k, plus one confirming pass.
    assert!(
        iters.iter().all(|&i| i == 4),
        "3-hop closure must take exactly 4 device fixpoint passes, got {:?}",
        &iters[..iters.len().min(4)]
    );
}

#[test]
fn resident_sparse_wcoj_single_join_populates_world_segmented_sparse_counts() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };
    let program = McProgram::compile_source(
        r#"
left_rel(1, 2).
left_rel(9, 9).
right_rel(2, 3).
join_rel(X, Z) :- left_rel(X, Y), right_rel(Y, Z).
query(join_rel(1, 3)).
query(join_rel(9, 3)).
"#,
    )
    .expect("compile sparse single join");
    let c = cfg(128, 23);
    let r = program
        .evaluate_resident_with_provider(c.clone(), provider.clone())
        .expect("resident sparse single join");
    assert!(r.no_host.is_no_host(), "{:?}", r.no_host);

    let (counts, ev) = download(&provider, &r);
    assert_eq!(ev as usize, c.samples);
    assert_eq!(counts[0] as usize, c.samples);
    assert_eq!(counts[1], 0);

    let (sparse_counts, sparse_offsets) = download_sparse_trace(&provider, &r);
    assert!(
        sparse_counts.iter().all(|&count| count == 4),
        "expected 3 EDB rows + 1 derived row per world, got {:?}",
        &sparse_counts[..sparse_counts.len().min(8)]
    );
    assert_eq!(sparse_offsets[0], 0);
    assert!(
        sparse_offsets.windows(2).all(|w| w[1] > w[0]),
        "world sparse offsets must be monotone: {:?}",
        &sparse_offsets[..sparse_offsets.len().min(8)]
    );
}

#[test]
fn resident_sparse_wcoj_branching_join_outputs_exact_rows() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };
    let program = McProgram::compile_source(
        r#"
lhs_rel(1, 7).
lhs_rel(2, 7).
rhs_rel(7, 9).
rhs_rel(8, 9).
out_rel(X, Z) :- lhs_rel(X, Y), rhs_rel(Y, Z).
query(out_rel(1, 9)).
query(out_rel(2, 9)).
query(out_rel(3, 9)).
"#,
    )
    .expect("compile sparse branching join");
    let c = cfg(96, 31);
    let r = program
        .evaluate_resident_with_provider(c.clone(), provider.clone())
        .expect("resident sparse branching join");
    assert!(r.no_host.is_no_host(), "{:?}", r.no_host);

    let (counts, ev) = download(&provider, &r);
    assert_eq!(ev as usize, c.samples);
    assert_eq!(counts[0] as usize, c.samples);
    assert_eq!(counts[1] as usize, c.samples);
    assert_eq!(counts[2], 0);
    let (sparse_counts, _) = download_sparse_trace(&provider, &r);
    assert!(
        sparse_counts.iter().all(|&count| count == 6),
        "expected 4 EDB rows + 2 derived rows per world, got {:?}",
        &sparse_counts[..sparse_counts.len().min(8)]
    );
}

#[test]
fn resident_sparse_wcoj_chain_join_supports_four_distinct_variables() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };
    let program = McProgram::compile_source(
        r#"
ab_rel(1, 2).
bc_rel(2, 3).
cd_rel(3, 4).
out_rel(A, D) :- ab_rel(A, B), bc_rel(B, C), cd_rel(C, D).
query(out_rel(1, 4)).
query(out_rel(1, 3)).
"#,
    )
    .expect("compile sparse 4-variable chain join");
    let c = cfg(88, 47);
    let r = program
        .evaluate_resident_with_provider(c.clone(), provider.clone())
        .expect("resident sparse 4-variable chain join");
    assert!(r.no_host.is_no_host(), "{:?}", r.no_host);

    let (counts, ev) = download(&provider, &r);
    assert_eq!(ev as usize, c.samples);
    assert_eq!(counts[0] as usize, c.samples);
    assert_eq!(counts[1], 0);
    let (sparse_counts, _) = download_sparse_trace(&provider, &r);
    assert!(
        sparse_counts.iter().all(|&count| count == 4),
        "expected 3 EDB rows + one 4-variable join output per world, got {:?}",
        &sparse_counts[..sparse_counts.len().min(8)]
    );
}

#[test]
fn resident_sparse_wcoj_constant_constrained_join_exact_rows() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };
    let program = McProgram::compile_source(
        r#"
pair_rel(1, 7).
pair_rel(2, 8).
flag_rel(7).
out_rel(X) :- pair_rel(X, 7), flag_rel(7).
query(out_rel(1)).
query(out_rel(2)).
"#,
    )
    .expect("compile sparse constant join");
    let c = cfg(64, 37);
    let r = program
        .evaluate_resident_with_provider(c.clone(), provider.clone())
        .expect("resident sparse constant join");
    assert!(r.no_host.is_no_host(), "{:?}", r.no_host);

    let (counts, ev) = download(&provider, &r);
    assert_eq!(ev as usize, c.samples);
    assert_eq!(counts[0] as usize, c.samples);
    assert_eq!(counts[1], 0);
    let (sparse_counts, _) = download_sparse_trace(&provider, &r);
    assert!(
        sparse_counts.iter().all(|&count| count == 4),
        "expected 3 EDB rows + 1 derived row per world, got {:?}",
        &sparse_counts[..sparse_counts.len().min(8)]
    );
}

#[test]
fn resident_sparse_wcoj_rule_chaining_uses_sparse_derived_relation() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };
    let program = McProgram::compile_source(
        r#"
first_rel(1, 2).
second_rel(2, 3).
gate_rel(3).
mid_rel(X, Z) :- first_rel(X, Y), second_rel(Y, Z).
out_rel(X) :- mid_rel(X, Z), gate_rel(Z).
query(mid_rel(1, 3)).
query(out_rel(1)).
query(out_rel(2)).
"#,
    )
    .expect("compile sparse chained rules");
    let c = cfg(80, 41);
    let r = program
        .evaluate_resident_with_provider(c.clone(), provider.clone())
        .expect("resident sparse chained rules");
    assert!(r.no_host.is_no_host(), "{:?}", r.no_host);

    let (counts, ev) = download(&provider, &r);
    assert_eq!(ev as usize, c.samples);
    assert_eq!(counts[0] as usize, c.samples);
    assert_eq!(counts[1] as usize, c.samples);
    assert_eq!(counts[2], 0);
    let iters = download_iters(&provider, &r);
    assert!(
        iters.iter().all(|&i| i == 3),
        "trace: {:?}",
        &iters[..iters.len().min(8)]
    );
    let (sparse_counts, _) = download_sparse_trace(&provider, &r);
    assert!(
        sparse_counts.iter().all(|&count| count == 5),
        "expected 3 EDB rows + mid + out per world, got {:?}",
        &sparse_counts[..sparse_counts.len().min(8)]
    );
}

#[test]
fn resident_sparse_recursive_generic_closure_exact_trace_and_rows() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };
    let program = McProgram::compile_source(
        r#"
link_rel(1, 2).
link_rel(2, 3).
link_rel(3, 4).
path_rel(X, Y) :- link_rel(X, Y).
path_rel(X, Z) :- path_rel(X, Y), link_rel(Y, Z).
query(path_rel(1, 4)).
query(path_rel(2, 4)).
query(path_rel(4, 1)).
"#,
    )
    .expect("compile sparse recursive generic closure");
    let c = cfg(72, 43);
    let r = program
        .evaluate_resident_with_provider(c.clone(), provider.clone())
        .expect("resident sparse recursive generic closure");
    assert!(r.no_host.is_no_host(), "{:?}", r.no_host);

    let (counts, ev) = download(&provider, &r);
    assert_eq!(ev as usize, c.samples);
    assert_eq!(counts[0] as usize, c.samples);
    assert_eq!(counts[1] as usize, c.samples);
    assert_eq!(counts[2], 0);
    let iters = download_iters(&provider, &r);
    assert!(
        iters.iter().all(|&i| i == 4),
        "4-node closure must take exactly 4 device passes, got {:?}",
        &iters[..iters.len().min(8)]
    );
    let (sparse_counts, _) = download_sparse_trace(&provider, &r);
    assert!(
        sparse_counts.iter().all(|&count| count == 9),
        "expected 3 link rows + 6 path rows per world, got {:?}",
        &sparse_counts[..sparse_counts.len().min(8)]
    );
}

/// Pilot 7 — sparse/WCOJ multiway join: three positive body atoms share
/// variables, the output predicate is not name-special-cased, and the measured
/// region remains host-free.
#[test]
fn resident_sparse_wcoj_three_way_join_exact_no_host() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };
    let program = McProgram::compile_source(
        r#"
left_rel(1, 2).
right_rel(2, 4).
right_rel(3, 4).
guard_rel(1, 4).
out_rel(X, Z) :- left_rel(X, Y), right_rel(Y, Z), guard_rel(X, Z).
query(out_rel(1, 4)).
query(out_rel(2, 4)).
"#,
    )
    .expect("compile sparse 3-way join");
    let c = cfg(256, 11);
    let r = program
        .evaluate_resident_with_provider(c.clone(), provider.clone())
        .expect("resident sparse 3-way join");
    let expected_no_host = McNoHostStats {
        tracked_htod_calls: 0,
        tracked_dtoh_calls: 0,
        untracked_metadata_reads: 0,
        engine_launches: 1,
        host_loop_iterations: 0,
        per_sample_host_launches: 0,
        host_fixpoint_iterations: 0,
        per_operator_host_allocations: 0,
    };
    assert_eq!(r.no_host, expected_no_host, "{:?}", r.no_host);
    assert!(r.no_host.is_no_host(), "{:?}", r.no_host);

    let (counts, ev) = download(&provider, &r);
    assert_eq!(ev as usize, c.samples);
    assert_eq!(
        counts[0] as usize, c.samples,
        "left/right/guard 3-way join derives out_rel(1,4) in every world"
    );
    assert_eq!(
        counts[1], 0,
        "out_rel(2,4) has no supporting left_rel/guard_rel witness"
    );

    let (sparse_counts, sparse_offsets) = download_sparse_trace(&provider, &r);
    assert!(
        sparse_counts.iter().all(|&count| count == 5),
        "expected 4 EDB rows + 1 derived row per world, got {:?}",
        &sparse_counts[..sparse_counts.len().min(8)]
    );
    assert_eq!(sparse_offsets[0], 0);
    assert!(sparse_offsets.windows(2).all(|w| w[1] > w[0]));
}

#[test]
fn resident_sparse_wcoj_ternary_relation_join_exact_no_host() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };
    let program = McProgram::compile_source(
        r#"
tri_rel(1, 2, 3).
tri_rel(1, 5, 6).
key_rel(3, 4).
out_rel(A, D) :- tri_rel(A, B, C), key_rel(C, D).
query(out_rel(1, 4)).
query(out_rel(2, 4)).
"#,
    )
    .expect("compile sparse ternary join");
    let c = cfg(112, 53);
    let r = program
        .evaluate_resident_with_provider(c.clone(), provider.clone())
        .expect("resident sparse ternary join");
    assert!(r.no_host.is_no_host(), "{:?}", r.no_host);

    let (counts, ev) = download(&provider, &r);
    assert_eq!(ev as usize, c.samples);
    assert_eq!(
        counts[0] as usize, c.samples,
        "tri_rel(_,_,C) joined key_rel(C,_) should derive out_rel(1,4)"
    );
    assert_eq!(counts[1], 0, "no ternary witness exists for out_rel(2,4)");
    let (sparse_counts, _) = download_sparse_trace(&provider, &r);
    assert!(
        sparse_counts.iter().all(|&count| count == 4),
        "expected 3 EDB rows + one ternary-derived output per world, got {:?}",
        &sparse_counts[..sparse_counts.len().min(8)]
    );
}

#[test]
fn resident_device_diagnostics_are_resident_exact_and_clean() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };
    let program = McProgram::compile_source(
        r#"
link_rel(1, 2).
link_rel(2, 3).
link_rel(3, 4).
path_rel(X, Y) :- link_rel(X, Y).
path_rel(X, Z) :- path_rel(X, Y), link_rel(Y, Z).
query(path_rel(1, 4)).
"#,
    )
    .expect("compile diagnostic recursive program");
    let c = cfg(48, 59);
    let r = program
        .evaluate_resident_with_provider(c.clone(), provider.clone())
        .expect("resident diagnostic program");
    assert!(r.no_host.is_no_host(), "{:?}", r.no_host);

    let (counts, ev) = download(&provider, &r);
    assert_eq!(ev as usize, c.samples);
    assert_eq!(counts[0] as usize, c.samples);

    let iters = download_iters(&provider, &r);
    assert!(iters.iter().all(|&i| i == 4), "trace: {:?}", &iters);

    let (sparse_counts, sparse_offsets) = download_sparse_trace(&provider, &r);
    assert!(sparse_counts.iter().all(|&count| count == 9));
    assert_eq!(sparse_offsets.len(), c.samples + 1);
    let stride = sparse_offsets[1];
    let expected_offsets: Vec<u32> = (0..=c.samples as u32).map(|w| w * stride).collect();
    assert_eq!(sparse_offsets, expected_offsets);

    let (converged, overflow, participation) = download_resident_diagnostics(&provider, &r);
    assert!(
        converged.iter().all(|&flag| flag == 1),
        "device convergence flags: {:?}",
        &converged[..converged.len().min(8)]
    );
    assert!(
        overflow.iter().all(|&flag| flag == 0),
        "sparse arena overflow flags: {:?}",
        &overflow[..overflow.len().min(8)]
    );
    assert!(
        participation.iter().all(|&blocks| blocks == 1),
        "single-block participation flags: {:?}",
        &participation[..participation.len().min(8)]
    );
}

#[test]
fn resident_multiblock_world_executes_with_device_coordination() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };
    let program = McProgram::compile_source(
        r#"
link_rel(1, 2).
link_rel(2, 3).
link_rel(3, 4).
path_rel(X, Y) :- link_rel(X, Y).
path_rel(X, Z) :- path_rel(X, Y), link_rel(Y, Z).
query(path_rel(1, 4)).
query(path_rel(4, 1)).
"#,
    )
    .expect("compile multiblock recursive program");
    std::env::set_var("XLOG_MC_RESIDENT_BLOCKS_PER_WORLD", "2");
    let result = program.evaluate_resident_with_provider(cfg(8, 61), provider.clone());
    std::env::remove_var("XLOG_MC_RESIDENT_BLOCKS_PER_WORLD");
    let r = result.expect("resident multiblock recursive program");
    assert!(r.no_host.is_no_host(), "{:?}", r.no_host);

    let (counts, ev) = download(&provider, &r);
    let iters = download_iters(&provider, &r);
    let (sparse_counts, _) = download_sparse_trace(&provider, &r);
    let (converged, overflow, participation) = download_resident_diagnostics(&provider, &r);
    assert_eq!(ev as usize, r.total_samples);
    assert_eq!(
        counts[0] as usize, r.total_samples,
        "counts={counts:?} iters={iters:?} sparse_counts={sparse_counts:?} converged={converged:?} overflow={overflow:?} participation={participation:?}"
    );
    assert_eq!(counts[1], 0);
    assert!(iters.iter().all(|&i| i == 4), "trace: {:?}", &iters);
    assert!(sparse_counts.iter().all(|&count| count == 9));

    assert!(converged.iter().all(|&flag| flag == 1));
    assert!(overflow.iter().all(|&flag| flag == 0));
    assert_eq!(
        participation,
        vec![2u32; r.total_samples],
        "device-written block participation must prove two blocks per world"
    );
}

#[test]
fn resident_sparse_wcoj_over_budget_fails_closed_before_execution() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };
    let program = McProgram::compile_source(
        r#"
a_rel(1, 2).
b_rel(2, 3).
c_rel(1, 3).
out_rel(X, Z) :- a_rel(X, Y), b_rel(Y, Z), c_rel(X, Z).
query(out_rel(1, 3)).
"#,
    )
    .expect("compile sparse budget program");

    std::env::set_var("XLOG_MC_RESIDENT_MEMORY_BUDGET_BYTES", "64");
    let err = match program.evaluate_resident_with_provider(cfg(512, 19), provider) {
        Ok(_) => panic!("resident engine must fail closed before over-budget execution"),
        Err(err) => err,
    };
    std::env::remove_var("XLOG_MC_RESIDENT_MEMORY_BUDGET_BYTES");

    let msg = format!("{err:?}");
    assert!(
        msg.contains("resident_resource_budget"),
        "typed budget diagnostic missing: {msg}"
    );
    assert!(msg.contains("bound_bytes"), "bound missing: {msg}");
    assert!(msg.contains("budget_bytes"), "budget missing: {msg}");
}

// ---------------------------------------------------------------------------
// K5 — fail-closed negative tests. Compilation is pure (host-only): unsupported
// fragments must be rejected BEFORE execution with a typed diagnostic carrying
// kind + non-empty construct + context. No GPU needed.
// ---------------------------------------------------------------------------

fn reject(src: &str) -> xlog_prob::mc::ResidentRejection {
    let mc = McProgram::compile_source(src).expect("program should parse/compile");
    compile_resident_plan(&mc).expect_err("program must be rejected by resident engine")
}

#[test]
fn resident_rejects_negation_fail_closed() {
    let r = reject("1.0::a().\nb() :- not a().\nquery(b()).\n");
    assert_eq!(r.kind, ResidentRejectKind::Negation);
    assert!(!r.construct.is_empty(), "construct must name the predicate");
    assert!(r.context.contains("negat"), "context: {}", r.context);
}

#[test]
fn resident_rejects_arity_too_high_fail_closed() {
    let r = reject("1.0::p(1, 2, 3, 4).\nquery(p(1, 2, 3, 4)).\n");
    assert_eq!(r.kind, ResidentRejectKind::ArityTooHigh);
    assert_eq!(r.construct, "p");
    assert!(r.context.contains("arity"), "context: {}", r.context);
}

#[test]
fn resident_rejects_comparison_literal_fail_closed() {
    let r = reject("1.0::a(1).\nb(X) :- a(X), X < 5.\nquery(b(1)).\n");
    assert_eq!(r.kind, ResidentRejectKind::NonRelationalLiteral);
    assert!(!r.construct.is_empty());
    assert!(
        r.context.contains("non-relational"),
        "context: {}",
        r.context
    );
}

#[test]
fn resident_rejects_body_too_long_fail_closed() {
    let r = reject("1.0::a(1).\n1.0::b(1).\n1.0::c(1).\n1.0::d(1).\nout(X) :- a(X), b(X), c(X), d(X).\nquery(out(1)).\n");
    assert_eq!(r.kind, ResidentRejectKind::BodyTooLong);
    assert!(r.context.contains("body length"), "context: {}", r.context);
}
