//! D3 S3 spike — factorized recursive delta: parity + gate measurement.
//!
//! Design: `docs/plans/2026-06-12-d3-factorized-delta-design.md`.
//!
//! Parity tests (local): the fused bitmap novel-set step
//! (`fj_delta_novel_u32_recorded`) and the spike TC loop built on it
//! must produce the same row sets as a CPU oracle AND the production
//! executor running the equivalent recursive program. The bitmap path
//! is shape-agnostic, so parity fixtures are deliberately irregular
//! (cycles, diamonds, hubs, dead ends, self-loops) — not the block
//! fixture the gate uses.
//!
//! Gate tests (`#[ignore]`, RunPod only per the no-local-perf rule):
//! transitive closure on a dense block-cycle digraph. Baseline = the
//! unmodified production executor (semi-naive `execute_recursive_scc`:
//! hash_join_v2 → diff_gpu → union_gpu); spike = layout-normalize +
//! fj_delta loop sharing the same `union_gpu`. Gate: ≥5× peak-memory
//! reduction at wall-clock ≤1.2×, row-set parity.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::sys;
use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamId, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::{CudaBuffer, CudaColumn};
use xlog_cuda::{CudaDevice, CudaKernelProvider, FjDeltaCols, GpuMemoryManager};
use xlog_logic::Compiler;
use xlog_runtime::Executor;

struct DiscardSink;
impl LoggingSink for DiscardSink {
    fn emit(&self, _record: LogRecord) -> Result<(), SinkError> {
        Ok(())
    }
}

struct Fixture {
    memory: Arc<GpuMemoryManager>,
    provider: Arc<CudaKernelProvider>,
}

fn make_fixture_with_budget(budget_bytes: u64) -> Option<Fixture> {
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
    let async_resource: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(
        AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool)),
    );
    let logging: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(LoggingResource::new(
        async_resource,
        Arc::new(DiscardSink) as Arc<dyn LoggingSink>,
    ));
    let budget: Box<dyn DeviceMemoryResource + Send + Sync> =
        Box::new(GlobalDeviceBudget::new(logging, budget_bytes as usize));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(budget_bytes),
        runtime,
    ));
    let provider =
        Arc::new(CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory)).ok()?);
    Some(Fixture { memory, provider })
}

fn make_fixture() -> Option<Fixture> {
    make_fixture_with_budget(512 * 1024 * 1024)
}

fn upload_binary_u32(memory: &Arc<GpuMemoryManager>, rows: &[(u32, u32)]) -> CudaBuffer {
    let n = rows.len() as u32;
    let mut col0 = memory
        .alloc::<u8>(rows.len() * std::mem::size_of::<u32>())
        .expect("alloc col0");
    let mut col1 = memory
        .alloc::<u8>(rows.len() * std::mem::size_of::<u32>())
        .expect("alloc col1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let col0_bytes: Vec<u8> = rows.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
    let col1_bytes: Vec<u8> = rows.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
    let device = memory.device().inner();
    device
        .htod_sync_copy_into(&col0_bytes, &mut col0)
        .expect("htod col0");
    device
        .htod_sync_copy_into(&col1_bytes, &mut col1)
        .expect("htod col1");
    device
        .htod_sync_copy_into(&[n], &mut d_num_rows)
        .expect("htod d_num_rows");
    let schema = Schema::new(vec![
        ("col0".to_string(), ScalarType::U32),
        ("col1".to_string(), ScalarType::U32),
    ]);
    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into()],
        n as u64,
        d_num_rows,
        schema,
        n,
    )
}

fn buffer_rows(memory: &Arc<GpuMemoryManager>, buffer: &CudaBuffer) -> usize {
    if let Some(n) = buffer.cached_row_count() {
        return n as usize;
    }
    let mut host = [0u32; 1];
    memory
        .device()
        .inner()
        .dtoh_sync_copy_into(buffer.num_rows_device(), &mut host)
        .expect("dtoh num_rows");
    host[0] as usize
}

fn download_column_u32(
    memory: &Arc<GpuMemoryManager>,
    buffer: &CudaBuffer,
    col: usize,
) -> Vec<u32> {
    let n = buffer_rows(memory, buffer);
    let mut bytes = vec![0u8; n * 4];
    if n == 0 {
        return Vec::new();
    }
    let CudaColumn::Owned(c) = buffer.column(col).expect("column") else {
        panic!("column must be owned");
    };
    unsafe {
        let res = sys::cuMemcpyDtoH_v2(bytes.as_mut_ptr() as *mut _, *c.device_ptr(), bytes.len());
        assert_eq!(res, sys::cudaError_enum::CUDA_SUCCESS, "dtoh column copy");
    }
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Deduped sorted row set (Datalog set semantics).
fn download_row_set(memory: &Arc<GpuMemoryManager>, buffer: &CudaBuffer) -> Vec<(u32, u32)> {
    let c0 = download_column_u32(memory, buffer, 0);
    let c1 = download_column_u32(memory, buffer, 1);
    let mut rows: Vec<(u32, u32)> = c0.into_iter().zip(c1).collect();
    rows.sort_unstable();
    rows.dedup();
    rows
}

/// CPU oracle: one semi-naive novel step.
fn oracle_novel_step(
    delta: &[(u32, u32)],
    edge: &[(u32, u32)],
    full_r: &[(u32, u32)],
) -> Vec<(u32, u32)> {
    let r: BTreeSet<(u32, u32)> = full_r.iter().copied().collect();
    let mut succ: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();
    for &(y, z) in edge {
        succ.entry(y).or_default().insert(z);
    }
    let mut novel: BTreeSet<(u32, u32)> = BTreeSet::new();
    for &(x, y) in delta {
        if let Some(zs) = succ.get(&y) {
            for &z in zs {
                if !r.contains(&(x, z)) {
                    novel.insert((x, z));
                }
            }
        }
    }
    novel.into_iter().collect()
}

/// CPU oracle: full transitive closure of `edge`.
fn oracle_tc(edge: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut r: BTreeSet<(u32, u32)> = edge.iter().copied().collect();
    let mut delta: Vec<(u32, u32)> = r.iter().copied().collect();
    loop {
        let novel = oracle_novel_step(&delta, edge, &r.iter().copied().collect::<Vec<_>>());
        if novel.is_empty() {
            break;
        }
        r.extend(novel.iter().copied());
        delta = novel;
    }
    r.into_iter().collect()
}

const TC_SOURCE: &str = "pred edge(u32, u32).\n\
                         pred q(u32, u32).\n\
                         q(X, Y) :- edge(X, Y).\n\
                         q(X, Z) :- q(X, Y), edge(Y, Z).";

/// Production-engine TC: compile + execute the recursive program and
/// return the executor (so the caller can read `q` and counters).
fn run_engine_tc(fix: &Fixture, edge_buf: CudaBuffer) -> Executor {
    let mut compiler = Compiler::new();
    let plan = compiler.compile(TC_SOURCE).expect("compile TC program");
    let mut executor = Executor::new(Arc::clone(&fix.provider));
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    executor.put_relation("edge", edge_buf);
    executor.execute_plan(&plan).expect("execute TC plan");
    executor
}

/// Spike TC: layout-normalize `edge` once, then iterate the fused
/// factorized novel step + `union_gpu` until fixpoint. Returns the
/// final stable relation.
fn run_spike_tc(fix: &Fixture, edge_buf: &CudaBuffer, domain: u32) -> (CudaBuffer, u32) {
    let provider = &fix.provider;
    let edge_norm = provider
        .wcoj_layout_u32_recorded(edge_buf, StreamId::DEFAULT)
        .expect("layout-normalize edge");
    // Base case: R0 = delta0 = edge (sorted, deduped).
    let mut full_r = provider.clone_buffer(&edge_norm).expect("clone R0");
    let mut delta = provider.clone_buffer(&edge_norm).expect("clone delta0");
    let mut iterations = 0u32;
    loop {
        iterations += 1;
        assert!(
            iterations <= 64,
            "spike TC failed to converge in 64 iterations"
        );
        let novel = provider
            .fj_delta_novel_u32_recorded(
                &delta,
                &edge_norm,
                &full_r,
                FjDeltaCols::CANONICAL,
                domain,
                StreamId::DEFAULT,
            )
            .expect("fj_delta_novel step");
        if buffer_rows(&fix.memory, &novel) == 0 {
            break;
        }
        full_r = provider.union_gpu(&full_r, &novel).expect("union R");
        delta = novel;
    }
    (full_r, iterations)
}

/// Irregular digraph: cycle 0→1→2→0, diamond 3→{4,5}→6 (two witnesses
/// for (3,6)), hub 8 with skewed out-degree, dead end 7, self-loop 9,
/// duplicate edges, and an isolated id gap (10 unused).
fn irregular_edges() -> Vec<(u32, u32)> {
    vec![
        (0, 1),
        (1, 2),
        (2, 0),
        (3, 4),
        (3, 5),
        (4, 6),
        (5, 6),
        (6, 7),
        (8, 0),
        (8, 3),
        (8, 7),
        (8, 9),
        (9, 9),
        (3, 4), // duplicate edge
        (11, 1),
    ]
}

#[test]
fn fj_delta_step_matches_cpu_oracle() {
    let Some(fix) = make_fixture() else {
        eprintln!("skipping fj_delta_step: no CUDA device");
        return;
    };
    let edge = irregular_edges();
    // Arbitrary mid-fixpoint state: delta has duplicate rows, rows
    // whose y has no successors, and rows whose candidates are
    // partially known in R already.
    let delta = vec![(0, 1), (0, 1), (2, 0), (8, 3), (7, 7), (3, 6)];
    let full_r = vec![(0, 2), (2, 1), (8, 4)];
    let expected = oracle_novel_step(&delta, &edge, &full_r);
    assert!(!expected.is_empty(), "fixture must produce novel rows");

    let edge_buf = upload_binary_u32(&fix.memory, &edge);
    let edge_norm = fix
        .provider
        .wcoj_layout_u32_recorded(&edge_buf, StreamId::DEFAULT)
        .expect("normalize edge");
    let delta_buf = upload_binary_u32(&fix.memory, &delta);
    let r_buf = upload_binary_u32(&fix.memory, &full_r);
    let novel = fix
        .provider
        .fj_delta_novel_u32_recorded(
            &delta_buf,
            &edge_norm,
            &r_buf,
            FjDeltaCols::CANONICAL,
            64,
            StreamId::DEFAULT,
        )
        .expect("novel step");
    let got = download_row_set(&fix.memory, &novel);
    assert_eq!(got, expected, "novel step must match the CPU oracle");

    // The emitted buffer must be lex-sorted and deduped as-is (it is
    // the next iteration's delta without any post-processing).
    let raw_c0 = download_column_u32(&fix.memory, &novel, 0);
    let raw_c1 = download_column_u32(&fix.memory, &novel, 1);
    let raw: Vec<(u32, u32)> = raw_c0.into_iter().zip(raw_c1).collect();
    assert_eq!(raw, expected, "emit order must be lex-sorted and deduped");
}

#[test]
fn fj_delta_tc_loop_matches_engine_and_oracle() {
    let Some(fix) = make_fixture() else {
        eprintln!("skipping fj_delta_tc_loop: no CUDA device");
        return;
    };
    let edge = irregular_edges();
    let expected = oracle_tc(&edge);

    let edge_buf = upload_binary_u32(&fix.memory, &edge);
    let (spike_r, iterations) = run_spike_tc(&fix, &edge_buf, 64);
    let spike_rows = download_row_set(&fix.memory, &spike_r);
    assert_eq!(spike_rows, expected, "spike TC must match the CPU oracle");
    assert!(iterations >= 3, "fixture must need several iterations");

    let engine_edge = upload_binary_u32(&fix.memory, &edge);
    let executor = run_engine_tc(&fix, engine_edge);
    let q = executor.store().get("q").expect("q relation");
    let engine_rows = download_row_set(&fix.memory, q);
    assert_eq!(
        engine_rows, expected,
        "production engine must match the CPU oracle"
    );
}

#[test]
fn fj_delta_out_of_domain_fails_closed() {
    let Some(fix) = make_fixture() else {
        eprintln!("skipping fj_delta_domain: no CUDA device");
        return;
    };
    let edge = vec![(1, 99), (2, 3)];
    let edge_buf = upload_binary_u32(&fix.memory, &edge);
    let edge_norm = fix
        .provider
        .wcoj_layout_u32_recorded(&edge_buf, StreamId::DEFAULT)
        .expect("normalize edge");
    let delta_buf = upload_binary_u32(&fix.memory, &[(0, 1)]);
    let r_buf = upload_binary_u32(&fix.memory, &[(0, 2)]);
    // z = 99 >= domain 64 → the mark kernel must set the error flag
    // and the entry must fail closed instead of writing out of bounds.
    let result = fix.provider.fj_delta_novel_u32_recorded(
        &delta_buf,
        &edge_norm,
        &r_buf,
        FjDeltaCols::CANONICAL,
        64,
        StreamId::DEFAULT,
    );
    let Err(err) = result else {
        panic!("out-of-domain id must fail closed");
    };
    let msg = format!("{err}");
    assert!(msg.contains("outside domain"), "unexpected error: {msg}");
}

#[test]
fn fj_delta_empty_inputs_yield_empty_novel() {
    let Some(fix) = make_fixture() else {
        eprintln!("skipping fj_delta_empty: no CUDA device");
        return;
    };
    let edge_buf = upload_binary_u32(&fix.memory, &[(1, 2)]);
    let edge_norm = fix
        .provider
        .wcoj_layout_u32_recorded(&edge_buf, StreamId::DEFAULT)
        .expect("normalize edge");
    let empty_delta = upload_binary_u32(&fix.memory, &[]);
    let r_buf = upload_binary_u32(&fix.memory, &[(1, 2)]);
    let novel = fix
        .provider
        .fj_delta_novel_u32_recorded(
            &empty_delta,
            &edge_norm,
            &r_buf,
            FjDeltaCols::CANONICAL,
            8,
            StreamId::DEFAULT,
        )
        .expect("empty delta step");
    assert_eq!(buffer_rows(&fix.memory, &novel), 0);

    // Saturated state: delta full, but every candidate already in R.
    let delta = upload_binary_u32(&fix.memory, &[(0, 1)]);
    let r_full = upload_binary_u32(&fix.memory, &[(0, 2)]);
    let novel2 = fix
        .provider
        .fj_delta_novel_u32_recorded(
            &delta,
            &edge_norm,
            &r_full,
            FjDeltaCols::CANONICAL,
            8,
            StreamId::DEFAULT,
        )
        .expect("saturated step");
    assert_eq!(buffer_rows(&fix.memory, &novel2), 0);
}

/// Dense block-cycle digraph: k blocks of b nodes (block-major ids),
/// complete bipartite edges B_i → B_{(i+1) mod k}. TC = n² pairs;
/// every novel pair has exactly b duplicate witnesses and the final
/// iteration is 100% rediscovery — the S3 "delta blowup" workload.
fn block_cycle_edges(k: u32, b: u32) -> Vec<(u32, u32)> {
    let mut edges = Vec::with_capacity((k * b * b) as usize);
    for i in 0..k {
        let src = i * b;
        let dst = ((i + 1) % k) * b;
        for u in 0..b {
            for v in 0..b {
                edges.push((src + u, dst + v));
            }
        }
    }
    edges
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn gate_run(k: u32, b: u32, budget_bytes: u64) {
    let n = k * b;
    let edges = block_cycle_edges(k, b);
    let expected_rows = (n as usize) * (n as usize);
    eprintln!(
        "S3 gate fixture: k={k} b={b} n={n} |E|={} expected |TC|={expected_rows}",
        edges.len()
    );

    const REPS: usize = 3;
    let mut base_ms = Vec::new();
    let mut base_peak = Vec::new();
    let mut spike_ms = Vec::new();
    let mut spike_peak = Vec::new();

    // Baseline: the LEGACY engine path. Phase B made the factorized
    // dispatch the engine default, so the baseline pins the kill
    // switch for its reps (gate tests run isolated/serial — no
    // concurrent test observes the process-global env var).
    std::env::set_var("XLOG_DISABLE_FACTORIZED_DELTA", "1");
    for rep in 0..REPS {
        let fix = make_fixture_with_budget(budget_bytes).expect("CUDA fixture");
        let edge_buf = upload_binary_u32(&fix.memory, &edges);
        fix.memory.reset_peak();
        let live_before = fix.memory.allocated_bytes();
        let t0 = Instant::now();
        let executor = run_engine_tc(&fix, edge_buf);
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        let peak = fix.memory.peak_bytes();
        let q = executor.store().get("q").expect("q relation");
        let rows = buffer_rows(&fix.memory, q);
        assert_eq!(rows, expected_rows, "baseline TC row count");
        if rep == 0 {
            let row_set = download_row_set(&fix.memory, q);
            assert_eq!(row_set.len(), expected_rows, "baseline TC row set size");
        }
        eprintln!(
            "  baseline rep {rep}: {dt:.1} ms, peak {} B (live before: {} B)",
            peak, live_before
        );
        base_ms.push(dt);
        base_peak.push(peak as f64);
    }

    std::env::remove_var("XLOG_DISABLE_FACTORIZED_DELTA");

    // Spike: factorized novel-set loop sharing union_gpu.
    let mut spike_row_set: Option<Vec<(u32, u32)>> = None;
    for rep in 0..REPS {
        let fix = make_fixture_with_budget(budget_bytes).expect("CUDA fixture");
        let edge_buf = upload_binary_u32(&fix.memory, &edges);
        fix.memory.reset_peak();
        let live_before = fix.memory.allocated_bytes();
        let t0 = Instant::now();
        let (full_r, iterations) = run_spike_tc(&fix, &edge_buf, n);
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        let peak = fix.memory.peak_bytes();
        let rows = buffer_rows(&fix.memory, &full_r);
        assert_eq!(rows, expected_rows, "spike TC row count");
        if rep == 0 {
            spike_row_set = Some(download_row_set(&fix.memory, &full_r));
        }
        eprintln!(
            "  spike rep {rep}: {dt:.1} ms, peak {} B, {iterations} iterations \
             (live before: {} B)",
            peak, live_before
        );
        spike_ms.push(dt);
        spike_peak.push(peak as f64);
    }

    // Row-set parity (count parity asserted every rep above; the full
    // set comparison runs once — n² rows).
    let spike_set = spike_row_set.expect("spike row set");
    assert_eq!(spike_set.len(), expected_rows, "spike TC row set size");

    let bm = median(&mut base_ms);
    let bp = median(&mut base_peak);
    let sm = median(&mut spike_ms);
    let sp = median(&mut spike_peak);
    eprintln!(
        "S3 gate (k={k} b={b}): baseline median {bm:.1} ms / peak {:.1} MiB; \
         spike median {sm:.1} ms / peak {:.1} MiB; \
         peak ratio {:.2}x (gate >=5x), wall-clock ratio {:.3}x (gate <=1.2x)",
        bp / (1024.0 * 1024.0),
        sp / (1024.0 * 1024.0),
        bp / sp,
        sm / bm,
    );
}

/// Primary gate scale. RunPod-only (no local perf measurement).
#[test]
#[ignore = "S3 gate measurement — run on RunPod, never locally"]
fn s3_measurement_factorized_delta_gate() {
    if make_fixture().is_none() {
        eprintln!("skipping s3 gate: no CUDA device");
        return;
    }
    gate_run(4, 256, 10 * 1024 * 1024 * 1024);
}

/// Secondary scale (ratio trend evidence).
#[test]
#[ignore = "S3 gate measurement — run on RunPod, never locally"]
fn s3_measurement_factorized_delta_gate_secondary() {
    if make_fixture().is_none() {
        eprintln!("skipping s3 secondary: no CUDA device");
        return;
    }
    gate_run(4, 384, 13 * 1024 * 1024 * 1024);
}
