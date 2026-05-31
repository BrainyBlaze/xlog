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
        iters.iter().all(|&i| i >= 3),
        "3-hop closure must take >=3 fixpoint passes, got {:?}",
        &iters[..iters.len().min(4)]
    );
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
    let r = reject("1.0::p(1, 2, 3).\nquery(p(1, 2, 3)).\n");
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
    let r = reject("1.0::a(1).\n1.0::b(1).\n1.0::c(1).\nd(X) :- a(X), b(X), c(X).\nquery(d(1)).\n");
    assert_eq!(r.kind, ResidentRejectKind::BodyTooLong);
    assert!(r.context.contains("body length"), "context: {}", r.context);
}
