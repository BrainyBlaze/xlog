mod common;
use common::setup_provider;

use cudarc::driver::DeviceSlice;
use xlog_cuda::LaunchAsync;
use xlog_prob::mc::{McDeviceResult, McEvalConfig, McProgram, McSamplingMethod};

fn mc_config(samples: usize, seed: u64, max_nonmonotone_iterations: usize) -> McEvalConfig {
    let mut config = McEvalConfig::default();
    config.samples = samples;
    config.seed = seed;
    config.confidence = 0.95;
    config.max_nonmonotone_iterations = max_nonmonotone_iterations;
    config
}

/// Download device-resident query/evidence counts to host for exact-value
/// assertions. This is a *test-side* materialization performed after the hot
/// loop has completed — it is deliberately not part of the measured GPU-native
/// MC loop (the engine snapshots transfers around the loop internally).
fn download_counts(
    provider: &std::sync::Arc<xlog_cuda::CudaKernelProvider>,
    device_result: &McDeviceResult,
) -> (Vec<u32>, u32) {
    let mut host_counts = vec![0u32; device_result.query_counts.len()];
    if !host_counts.is_empty() {
        provider
            .device()
            .inner()
            .dtoh_sync_copy_into(&device_result.query_counts, &mut host_counts)
            .expect("dtoh query counts");
    }
    let mut host_evidence = [0u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&device_result.evidence_count, &mut host_evidence)
        .expect("dtoh evidence count");
    (host_counts, host_evidence[0])
}

/// K1 transfer-budget gate.
///
/// Asserts that the MC sample/evaluate/count hot loop performs **zero** tracked
/// HtoD and zero tracked DtoH transfers. The measurement is taken inside the
/// engine (`McHotLoopTransfers` on `McDeviceResult`): the transfer counters are
/// snapshotted immediately before the `for sample_idx` loop and again after a
/// post-loop `synchronize()`, so static setup uploads (before) and final-count
/// downloads (after) are excluded by construction.
///
/// Both count strategies are exercised so the gate cannot pass by only covering
/// the clamped (`QueriesOnly`) path:
///   * clamped path — evidence forced, only query pointers refreshed per sample;
///   * rejection path (`QueriesAndEvidence`) — query *and* evidence pointers
///     refreshed per sample (the second per-sample transfer source).
#[test]
fn mc_hot_loop_is_zero_transfer_both_strategies() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // --- Clamped path (QueriesOnly): no evidence, deterministic count. ---
    let clamped = McProgram::compile_source(
        r#"
1.0::coin().
query(coin()).
"#,
    )
    .expect("compile clamped program");
    let cfg = mc_config(256, 11, 16);
    let res = clamped
        .evaluate_gpu_device_with_provider(cfg.clone(), provider.clone())
        .expect("evaluate clamped");
    assert!(
        res.hot_loop_transfers.is_zero(),
        "clamped hot loop must be zero-transfer, got {:?}",
        res.hot_loop_transfers
    );
    let (counts, evidence) = download_counts(&provider, &res);
    assert_eq!(evidence as usize, cfg.samples, "clamped evidence count");
    assert_eq!(counts[0] as usize, cfg.samples, "clamped query count");

    // --- Rejection path (QueriesAndEvidence): evidence present + explicit
    // Rejection forces the evidence-pointer per-sample refresh. 1.0::a() makes
    // evidence always satisfied so every sample counts and the count is exact. ---
    let mut rej_cfg = mc_config(256, 11, 16);
    rej_cfg.sampling_method = Some(McSamplingMethod::Rejection);
    let rejection = McProgram::compile_source(
        r#"
1.0::a().
evidence(a(), true).
query(a()).
"#,
    )
    .expect("compile rejection program");
    let res = rejection
        .evaluate_gpu_device_with_provider(rej_cfg.clone(), provider.clone())
        .expect("evaluate rejection");
    assert_eq!(
        res.sampling_method,
        McSamplingMethod::Rejection,
        "explicit Rejection must be honored to exercise QueriesAndEvidence"
    );
    assert!(
        res.hot_loop_transfers.is_zero(),
        "rejection hot loop must be zero-transfer, got {:?}",
        res.hot_loop_transfers
    );
    let (counts, evidence) = download_counts(&provider, &res);
    assert_eq!(
        evidence as usize, rej_cfg.samples,
        "rejection evidence count"
    );
    assert_eq!(counts[0] as usize, rej_cfg.samples, "rejection query count");
}

/// K1 multi-evidence coverage.
///
/// The budget test above and the evidence pilot use a single evidence atom, so
/// the per-sample loop over `plan.evidence_rel_specs` for `>1` evidence atoms
/// (the path modified to use stable evidence row-count buffers + per-element
/// device-to-device copies) is otherwise unexercised. Here two evidence atoms
/// (`a`, `b`) are both deterministically satisfied (`1.0::`), so every sample
/// counts and the query is exact — and the hot loop must still be zero-transfer
/// while populating a 2-element evidence row-count buffer each sample.
#[test]
fn mc_hot_loop_zero_transfer_multi_evidence() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let mut cfg = mc_config(300, 13, 16);
    cfg.sampling_method = Some(McSamplingMethod::Rejection);
    let program = McProgram::compile_source(
        r#"
1.0::a().
1.0::b().
1.0::d().
evidence(a(), true).
evidence(b(), true).
query(d()).
"#,
    )
    .expect("compile multi-evidence program");

    let res = program
        .evaluate_gpu_device_with_provider(cfg.clone(), provider.clone())
        .expect("evaluate multi-evidence");
    assert_eq!(res.sampling_method, McSamplingMethod::Rejection);
    assert!(
        res.hot_loop_transfers.is_zero(),
        "multi-evidence hot loop must be zero-transfer, got {:?}",
        res.hot_loop_transfers
    );

    let (counts, evidence) = download_counts(&provider, &res);
    assert_eq!(
        evidence as usize, cfg.samples,
        "both evidence atoms always satisfied => every sample counts"
    );
    assert_eq!(
        counts[0] as usize, cfg.samples,
        "1.0::d() true in all worlds"
    );
}

#[test]
fn mc_gpu_device_counts_match_expected_small() {
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
    .expect("compile program");

    let cfg = mc_config(128, 7, 16);

    let device_result = program
        .evaluate_gpu_device_with_provider(cfg.clone(), provider.clone())
        .expect("evaluate_gpu_device_with_provider");

    assert_eq!(device_result.query_counts.len(), 1);

    let mut host_counts = vec![0u32; device_result.query_counts.len()];
    if !host_counts.is_empty() {
        provider
            .device()
            .inner()
            .dtoh_sync_copy_into(&device_result.query_counts, &mut host_counts)
            .expect("dtoh query counts");
    }
    let mut host_evidence = [0u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&device_result.evidence_count, &mut host_evidence)
        .expect("dtoh evidence count");

    assert_eq!(
        host_evidence[0] as usize,
        cfg.samples,
        "evidence_count={} query_count={}",
        host_evidence[0],
        host_counts.first().copied().unwrap_or(0)
    );
    assert_eq!(
        host_counts.first().copied().unwrap_or(0) as usize,
        cfg.samples
    );
}

#[test]
fn mc_eval_kernels_set_evidence_ok_without_evidence() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let mut d_query_count = provider
        .memory()
        .alloc::<u32>(1)
        .expect("alloc query count");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&[1u32], &mut d_query_count)
        .expect("copy query count");
    let query_ptr = *d_query_count.device_ptr();

    let mut d_query_ptrs = provider.memory().alloc::<u64>(1).expect("alloc query ptrs");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&[query_ptr], &mut d_query_ptrs)
        .expect("copy query ptrs");

    let mut d_evidence_ptrs = provider
        .memory()
        .alloc::<u64>(1)
        .expect("alloc evidence ptrs");
    provider
        .device()
        .inner()
        .memset_zeros(&mut d_evidence_ptrs)
        .expect("zero evidence ptrs");
    let mut d_evidence_expected = provider
        .memory()
        .alloc::<u8>(1)
        .expect("alloc evidence expected");
    provider
        .device()
        .inner()
        .memset_zeros(&mut d_evidence_expected)
        .expect("zero evidence expected");

    let mut d_query_flags = provider.memory().alloc::<u8>(1).expect("alloc query flags");
    let mut d_evidence_ok = provider.memory().alloc::<u8>(1).expect("alloc evidence ok");

    let truth_fn = provider
        .device()
        .inner()
        .get_func(
            xlog_cuda::provider::MC_EVAL_MODULE,
            xlog_cuda::provider::mc_eval_kernels::MC_EVAL_QUERY_EVIDENCE_TRUTH,
        )
        .expect("mc_eval_query_evidence_truth kernel");

    // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
    unsafe {
        truth_fn
            .clone()
            .launch(
                cudarc::driver::LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &d_query_ptrs,
                    1u32,
                    &d_evidence_ptrs,
                    &d_evidence_expected,
                    0u32,
                    &mut d_query_flags,
                    &mut d_evidence_ok,
                ),
            )
            .expect("launch truth kernel");
    }

    provider
        .device()
        .synchronize()
        .expect("sync after truth kernel");

    let mut host_flags = [0u8];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&d_query_flags, &mut host_flags)
        .expect("copy query flags");
    let mut host_ok = [0u8];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&d_evidence_ok, &mut host_ok)
        .expect("copy evidence ok");

    assert_eq!(host_flags[0], 1u8);
    assert_eq!(host_ok[0], 1u8);
}

#[test]
fn mc_accumulate_counts_increments_on_ok() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let mut d_query_flags = provider.memory().alloc::<u8>(1).expect("alloc query flags");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&[1u8], &mut d_query_flags)
        .expect("copy query flags");
    let mut d_evidence_ok = provider.memory().alloc::<u8>(1).expect("alloc evidence ok");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&[1u8], &mut d_evidence_ok)
        .expect("copy evidence ok");

    let mut d_query_counts = provider
        .memory()
        .alloc::<u32>(1)
        .expect("alloc query counts");
    provider
        .device()
        .inner()
        .memset_zeros(&mut d_query_counts)
        .expect("zero query counts");
    let mut d_evidence_count = provider
        .memory()
        .alloc::<u32>(1)
        .expect("alloc evidence count");
    provider
        .device()
        .inner()
        .memset_zeros(&mut d_evidence_count)
        .expect("zero evidence count");

    let accum_fn = provider
        .device()
        .inner()
        .get_func(
            xlog_cuda::provider::MC_EVAL_MODULE,
            xlog_cuda::provider::mc_eval_kernels::MC_EVAL_ACCUMULATE_COUNTS,
        )
        .expect("mc_accumulate_counts kernel");

    // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
    unsafe {
        accum_fn
            .clone()
            .launch(
                cudarc::driver::LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &d_query_flags,
                    1u32,
                    &d_evidence_ok,
                    &mut d_query_counts,
                    &mut d_evidence_count,
                ),
            )
            .expect("launch accumulate kernel");
    }

    provider
        .device()
        .synchronize()
        .expect("sync after accumulate kernel");

    let mut host_query_counts = [0u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&d_query_counts, &mut host_query_counts)
        .expect("copy query counts");
    let mut host_evidence_count = [0u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&d_evidence_count, &mut host_evidence_count)
        .expect("copy evidence count");

    assert_eq!(host_query_counts[0], 1u32);
    assert_eq!(host_evidence_count[0], 1u32);
}

// ---------------------------------------------------------------------------
// GPU-native MC pilots (K2/K3)
//
// Each pilot runs the device-resident API (`evaluate_gpu_device_with_provider`)
// and validates **exact** device-resident counts -- either a deterministic
// closed-form value (1.0/0.0 facts, exclusive-choice invariant) or an exact
// match against the seeded CPU oracle (`evaluate_cpu`) computed *outside* the
// hot loop. Every pilot also asserts the hot loop was zero-transfer (K1), so
// the pilots double as transfer-budget coverage across recursion and AD shapes
// that the dedicated budget test does not exercise.
// ---------------------------------------------------------------------------

/// K2 pilot 1 -- simple fact marginal, deterministic exact counts.
/// `1.0::a()` is true in every world, `0.0::b()` in none.
#[test]
fn pilot_simple_fact_marginal_exact() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let program = McProgram::compile_source(
        r#"
1.0::a().
0.0::b().
query(a()).
query(b()).
"#,
    )
    .expect("compile fact-marginal pilot");
    let cfg = mc_config(512, 3, 16);

    let res = program
        .evaluate_gpu_device_with_provider(cfg.clone(), provider.clone())
        .expect("evaluate fact-marginal pilot");
    assert!(
        res.hot_loop_transfers.is_zero(),
        "fact-marginal hot loop must be zero-transfer, got {:?}",
        res.hot_loop_transfers
    );

    let (counts, evidence) = download_counts(&provider, &res);
    assert_eq!(
        evidence as usize, cfg.samples,
        "no-evidence => all samples count"
    );
    assert_eq!(counts.len(), 2);
    assert_eq!(
        counts[0] as usize, cfg.samples,
        "1.0::a() true in all worlds"
    );
    assert_eq!(counts[1], 0, "0.0::b() true in no world");
}

/// K2 pilot 2 -- evidence conditioning, exact GPU == seeded CPU-oracle parity.
/// `evidence(a(), true)` is forceable so clamping is auto-selected; `b()` is a
/// genuine 0<p<1 query whose device count must match the CPU oracle bit-for-bit
/// (both paths sample the same seeded Bernoulli matrix).
#[cfg(feature = "host-io")]
#[test]
fn pilot_evidence_conditioning_matches_oracle() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let program = McProgram::compile_source(
        r#"
0.5::a().
0.5::b().
evidence(a(), true).
query(b()).
"#,
    )
    .expect("compile evidence pilot");
    let cfg = mc_config(1000, 17, 16);

    // CPU oracle (host evaluation) -- computed outside the GPU hot loop.
    let oracle = program.evaluate_cpu(cfg.clone()).expect("cpu oracle");
    assert_eq!(
        oracle.sampling_method,
        McSamplingMethod::EvidenceClamping,
        "forceable evidence => clamping"
    );
    let oracle_count =
        (oracle.query_estimates[0].prob * oracle.evidence_samples as f64).round() as usize;

    let res = program
        .evaluate_gpu_device_with_provider(cfg.clone(), provider.clone())
        .expect("evaluate evidence pilot");
    assert!(
        res.hot_loop_transfers.is_zero(),
        "evidence-conditioning hot loop must be zero-transfer, got {:?}",
        res.hot_loop_transfers
    );
    assert_eq!(res.sampling_method, McSamplingMethod::EvidenceClamping);

    let (counts, evidence) = download_counts(&provider, &res);
    assert_eq!(
        evidence as usize, cfg.samples,
        "clamped => evidence_count == samples"
    );
    assert_eq!(
        counts[0] as usize, oracle_count,
        "device b()-count must match seeded CPU oracle exactly"
    );
}

/// K2 pilot 3 -- annotated disjunction / exclusive choice.
/// `0.6::x(); 0.4::y()` is an exclusive choice: exactly one of x/y holds in each
/// world, so `count(x) + count(y) == samples` is an exact RNG-independent
/// invariant. Each individual count is additionally checked against the CPU
/// oracle for exact parity (host-io only).
#[test]
fn pilot_annotated_disjunction_exclusive() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let program = McProgram::compile_source(
        r#"
0.6::x(); 0.4::y().
query(x()).
query(y()).
"#,
    )
    .expect("compile AD pilot");
    let cfg = mc_config(1000, 5, 16);

    let res = program
        .evaluate_gpu_device_with_provider(cfg.clone(), provider.clone())
        .expect("evaluate AD pilot");
    assert!(
        res.hot_loop_transfers.is_zero(),
        "AD hot loop must be zero-transfer, got {:?}",
        res.hot_loop_transfers
    );

    let (counts, _evidence) = download_counts(&provider, &res);
    assert_eq!(counts.len(), 2);
    assert_eq!(
        counts[0] as usize + counts[1] as usize,
        cfg.samples,
        "exclusive choice: exactly one of x()/y() per world (exact invariant)"
    );

    #[cfg(feature = "host-io")]
    {
        let oracle = program.evaluate_cpu(cfg.clone()).expect("cpu oracle");
        let denom = oracle.evidence_samples as f64;
        let ox = (oracle.query_estimates[0].prob * denom).round() as usize;
        let oy = (oracle.query_estimates[1].prob * denom).round() as usize;
        assert_eq!(counts[0] as usize, ox, "x() device count == oracle");
        assert_eq!(counts[1] as usize, oy, "y() device count == oracle");
    }
}

/// K2 pilot 4 -- recursive transitive closure, zero-transfer on the recursive
/// eval path. With `1.0::edge(..)` facts the closure is deterministic, so
/// `reach(1,3)` and `reach(1,2)` hold in every world. This pilot exercises
/// `execute_recursive_scc` inside the hot loop (the dedicated budget test only
/// covers non-recursive programs) and asserts that path is zero-transfer too.
#[test]
fn pilot_recursive_transitive_closure_zero_transfer() {
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
    .expect("compile recursive pilot");
    let cfg = mc_config(128, 21, 64);

    let res = program
        .evaluate_gpu_device_with_provider(cfg.clone(), provider.clone())
        .expect("evaluate recursive pilot");
    assert!(
        res.hot_loop_transfers.is_zero(),
        "recursive hot loop must be zero-transfer, got {:?}",
        res.hot_loop_transfers
    );

    let (counts, evidence) = download_counts(&provider, &res);
    assert_eq!(evidence as usize, cfg.samples);
    assert_eq!(counts.len(), 2);
    assert_eq!(
        counts[0] as usize, cfg.samples,
        "reach(1,3) holds in all worlds"
    );
    assert_eq!(
        counts[1] as usize, cfg.samples,
        "reach(1,2) holds in all worlds"
    );
}
