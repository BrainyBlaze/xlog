//! v0.6.2 WCOJ triangle benchmark baseline.
//!
//! Bench-only — no provider, kernel, or runtime changes. Compares
//! the env-gated GPU 3-way WCOJ dispatch against the existing
//! binary-join chain on identical fixtures, across u32, u64, and
//! a single Symbol sanity case.
//!
//! # Default matrix (no env)
//!
//! widths × fixtures × sizes × gates =
//!   {u32, u64} × {uniform, superhub, empty} × {10K, 50K} × {off, on}
//!   + 1 Symbol uniform 10K gate=on sanity case
//! = 25 cells.
//!
//! `WCOJ_BENCH_FULL=1` adds {100K, 250K} sizes for the same
//! width/fixture cross-product. The full matrix is intentionally
//! NOT the default: criterion's bench loop replays each cell
//! ~10× to converge, so the default has to fit in a tractable
//! wall-clock budget for the validation pass.
//!
//! # Methodology
//!
//! * Gate is forced via
//!   [`xlog_core::RuntimeConfig::with_wcoj_triangle_dispatch`] —
//!   bench never mutates the process-global env (env equivalent
//!   `XLOG_USE_WCOJ_TRIANGLE_U32=1` documented in BENCHMARKS.md
//!   for production callers).
//! * Timed region = `Executor::execute_plan` only — driven via
//!   `b.iter_custom(...)` so we own the per-iteration loop and
//!   can keep `put_relation` uploads + `store.remove("tri")`
//!   cleanup OUT of the measured time. Each cell builds ONE
//!   long-lived `Executor` so the executor's cached
//!   `wcoj_triangle_stream` (`OnceLock<StreamId>`) is acquired
//!   once and reused for every iteration. Building a fresh
//!   Executor per iteration would acquire a new stream each
//!   time and drain the runtime's `StreamPool` (cap 16,
//!   grow-only); past iteration 16 every dispatch would
//!   silently fall back to binary-join, biasing the timing.
//! * Bench-only: the `StreamPool` cap is bumped to 1024 in
//!   `make_provider`. Production runs at 16; the bench needs
//!   headroom across many short-lived correctness-check
//!   executors that each acquire one stream.
//! * Each (width, fixture, size) cell pre-runs a one-shot
//!   correctness check OUTSIDE the timed region: gate-off vs
//!   gate-on must produce the same row set (sorted+deduped),
//!   and the dispatch counter must be 0 vs 1 respectively. This
//!   keeps the bench from quietly drifting if a future kernel
//!   change breaks correctness. Fixtures are deduped host-side
//!   before upload so the binary-join chain (bag-semantic on
//!   inputs) and the WCOJ layout pass (set-semantic via
//!   sort+dedup) agree on the input set.
//! * `Throughput::Elements` is the sum of input rows across the
//!   three relations so criterion reports rows/sec consistently.
//! * Memory budget: 8 GB per provider, mirroring the existing
//!   `bench_multiway_join` precedent in xlog-gpu.
//!
//! # Out of scope
//!
//! * No actual histogram / skew kernel work — this slice
//!   identifies the regime where it's needed.
//! * No CI integration.
//! * No real-graph imports.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use cudarc::driver::sys;

use xlog_core::{MemoryBudget, RuntimeConfig, ScalarType, Schema};
use xlog_cuda::memory::CudaBuffer;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_ir::ExecutionPlan;
use xlog_logic::Compiler;
use xlog_runtime::executor::Executor;

// ---------------------------------------------------------------
// Fixture generation
// ---------------------------------------------------------------

/// Reproducible PCG-style LCG. Same constant used by xlog-gpu's
/// existing benches, so seeded determinism stays consistent across
/// the workspace.
#[inline]
fn lcg_next(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    *state
}

/// Uniform Erdős-Rényi: each row's (a, b) drawn iid from
/// `[0, key_range)`. `key_range = (rows / 10).max(1000)` keeps
/// per-key degree roughly Poisson around 10.
fn uniform_pairs(seed: u64, rows: u32, key_range: u32) -> Vec<(u32, u32)> {
    let mut state = seed;
    (0..rows)
        .map(|_| {
            let a = (lcg_next(&mut state) % key_range as u64) as u32;
            let b = (lcg_next(&mut state) % key_range as u64) as u32;
            (a, b)
        })
        .collect()
}

/// Super-hub fixture targeting the WCOJ kernel's per-thread-per-row
/// degeneracy: ~50% of edges concentrated on a single hub key.
///
/// For e_xy: 50% of rows have Y == HUB_Y; the other 50% are uniform.
/// For e_yz: 50% of rows have first column (Y) == HUB_Y; remainder uniform.
/// For e_xz: 50% of rows have first column (X) == HUB_X; remainder uniform.
///
/// Those concentrations interact: count-kernel threads with
/// (X = anything, Y = HUB_Y) walk an enormous Y-range in e_yz, and
/// threads with X = HUB_X walk an enormous X-range in e_xz. The
/// (HUB_X, HUB_Y) thread does both — quadratic intersect work
/// concentrated on a tiny number of threads while the rest are idle.
/// That's exactly the histogram-targetable shape.
fn superhub_pairs_xy(seed: u64, rows: u32, key_range: u32, hub_y: u32) -> Vec<(u32, u32)> {
    let mut state = seed;
    (0..rows)
        .map(|i| {
            let a = (lcg_next(&mut state) % key_range as u64) as u32;
            let b = if i.is_multiple_of(2) {
                hub_y
            } else {
                (lcg_next(&mut state) % key_range as u64) as u32
            };
            (a, b)
        })
        .collect()
}

fn superhub_pairs_first(seed: u64, rows: u32, key_range: u32, hub_first: u32) -> Vec<(u32, u32)> {
    let mut state = seed;
    (0..rows)
        .map(|i| {
            let a = if i.is_multiple_of(2) {
                hub_first
            } else {
                (lcg_next(&mut state) % key_range as u64) as u32
            };
            let b = (lcg_next(&mut state) % key_range as u64) as u32;
            (a, b)
        })
        .collect()
}

/// Empty-result triangle: three uniform relations over disjoint
/// key ranges so no triangles exist. Locks the count→scan→total
/// fast path on no-output (kernel still runs count + scan + a
/// 4-byte D2H of the inclusive total, then early-returns).
fn disjoint_pairs(seed: u64, rows: u32, base: u32, range: u32) -> Vec<(u32, u32)> {
    let mut state = seed;
    (0..rows)
        .map(|_| {
            let a = base + (lcg_next(&mut state) % range as u64) as u32;
            let b = base + (lcg_next(&mut state) % range as u64) as u32;
            (a, b)
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Width {
    U32,
    U64,
    Symbol,
}

impl Width {
    fn label(self) -> &'static str {
        match self {
            Width::U32 => "u32",
            Width::U64 => "u64",
            Width::Symbol => "sym",
        }
    }

    fn scalar(self) -> ScalarType {
        match self {
            Width::U32 => ScalarType::U32,
            Width::U64 => ScalarType::U64,
            Width::Symbol => ScalarType::Symbol,
        }
    }

    fn bytes_per_key(self) -> usize {
        match self {
            Width::U32 | Width::Symbol => 4,
            Width::U64 => 8,
        }
    }
}

#[derive(Clone)]
struct Fixture {
    e1: Vec<(u64, u64)>,
    e2: Vec<(u64, u64)>,
    e3: Vec<(u64, u64)>,
}

impl Fixture {
    fn total_rows(&self) -> u64 {
        (self.e1.len() + self.e2.len() + self.e3.len()) as u64
    }
}

/// Lift `Vec<(u32, u32)>` to `Vec<(u64, u64)>` AND dedup. The
/// dedup is critical for fair gate-off vs gate-on comparison:
/// the WCOJ layout pass deduplicates inputs, while the binary-
/// join chain does not. Without host-side dedup, the two paths
/// see different effective inputs on fixtures with sampled-
/// with-replacement duplicates (every fixture generator here
/// produces duplicates with non-trivial probability), and the
/// row sets diverge — not because the paths disagree but
/// because their semantics treat duplicates differently.
/// Datalog is set-semantic; deduping host-side aligns both
/// paths to set semantics for the bench.
fn dedup_pairs_to_u64(v: Vec<(u32, u32)>) -> Vec<(u64, u64)> {
    let mut out: Vec<(u64, u64)> = v.into_iter().map(|(a, b)| (a as u64, b as u64)).collect();
    out.sort();
    out.dedup();
    out
}

fn make_uniform(rows: u32) -> Fixture {
    let key_range = (rows / 10).max(1000);
    Fixture {
        e1: dedup_pairs_to_u64(uniform_pairs(1, rows, key_range)),
        e2: dedup_pairs_to_u64(uniform_pairs(2, rows, key_range)),
        e3: dedup_pairs_to_u64(uniform_pairs(3, rows, key_range)),
    }
}

fn make_superhub(rows: u32) -> Fixture {
    // Pick the hubs inside the key range so they collide with
    // ordinary uniform rows; that interaction is what creates
    // the per-thread workload imbalance the histogram targets.
    let key_range = (rows / 10).max(1000);
    let hub_y: u32 = 7;
    let hub_x: u32 = 13;
    Fixture {
        e1: dedup_pairs_to_u64(superhub_pairs_xy(101, rows, key_range, hub_y)),
        e2: dedup_pairs_to_u64(superhub_pairs_first(202, rows, key_range, hub_y)),
        e3: dedup_pairs_to_u64(superhub_pairs_first(303, rows, key_range, hub_x)),
    }
}

fn make_empty(rows: u32) -> Fixture {
    let range = (rows / 10).max(1000);
    Fixture {
        e1: dedup_pairs_to_u64(disjoint_pairs(11, rows, 0, range)),
        e2: dedup_pairs_to_u64(disjoint_pairs(22, rows, 1_000_000, range)),
        e3: dedup_pairs_to_u64(disjoint_pairs(33, rows, 2_000_000, range)),
    }
}

// ---------------------------------------------------------------
// GPU upload
// ---------------------------------------------------------------

fn upload_binary(memory: &Arc<GpuMemoryManager>, rows: &[(u64, u64)], width: Width) -> CudaBuffer {
    let n = rows.len() as u32;
    let bytes_per_col = (n as usize) * width.bytes_per_key();
    let mut col0 = memory.alloc::<u8>(bytes_per_col).expect("alloc col0");
    let mut col1 = memory.alloc::<u8>(bytes_per_col).expect("alloc col1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let device = memory.device().inner();
    if n > 0 {
        match width {
            Width::U64 => {
                let c0: Vec<u8> = rows.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
                let c1: Vec<u8> = rows.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
                device.htod_sync_copy_into(&c0, &mut col0).expect("htod c0");
                device.htod_sync_copy_into(&c1, &mut col1).expect("htod c1");
            }
            Width::U32 | Width::Symbol => {
                let c0: Vec<u8> = rows
                    .iter()
                    .flat_map(|(a, _)| (*a as u32).to_le_bytes())
                    .collect();
                let c1: Vec<u8> = rows
                    .iter()
                    .flat_map(|(_, b)| (*b as u32).to_le_bytes())
                    .collect();
                device.htod_sync_copy_into(&c0, &mut col0).expect("htod c0");
                device.htod_sync_copy_into(&c1, &mut col1).expect("htod c1");
            }
        }
    }
    device
        .htod_sync_copy_into(&[n], &mut d_num_rows)
        .expect("htod d_num_rows");
    let schema = Schema::new(vec![
        ("col0".to_string(), width.scalar()),
        ("col1".to_string(), width.scalar()),
    ]);
    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into()],
        n as u64,
        d_num_rows,
        schema,
        n,
    )
}

const TRIANGLE_SOURCE: &str = "tri(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z).";

// ---------------------------------------------------------------
// Provider fixture (one provider per bench function — built once,
// reused across cells; matches the xlog-gpu bench convention).
// ---------------------------------------------------------------

#[allow(dead_code)]
struct ProviderFixture {
    device: Arc<CudaDevice>,
    memory: Arc<GpuMemoryManager>,
    provider: Arc<CudaKernelProvider>,
}

fn make_provider(memory_mb: u64) -> Option<ProviderFixture> {
    use xlog_cuda::device_runtime::{
        AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
        LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
    };

    struct DiscardSink;
    impl LoggingSink for DiscardSink {
        fn emit(&self, _record: LogRecord) -> Result<(), SinkError> {
            Ok(())
        }
    }

    let device = Arc::new(CudaDevice::new(0).ok()?);
    // Bench-only: bump the stream pool cap well above the
    // production default (16). Every (width × fixture × size ×
    // gate) cell builds at least one Executor + correctness-
    // check executor, each of which acquires one cached
    // launch stream from the pool. The runtime's grow-only
    // pool would saturate at cap=16 across our 25-cell matrix
    // and silently route subsequent dispatches through the
    // binary-join fallback — corrupting the bench numbers in
    // the same way the correctness_check now panics on. The
    // production runtime cap is correct for production
    // (matches CudaKernelProvider::recorded_op_stream's
    // one-stream-per-provider model); the bench just needs
    // headroom for many short-lived Executors.
    let pool = Arc::new(StreamPool::new(Arc::clone(&device), 1024));
    let budget_bytes: usize = (memory_mb * 1024 * 1024) as usize;
    let async_resource: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(
        AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool)),
    );
    let logging: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(LoggingResource::new(
        async_resource,
        Arc::new(DiscardSink) as Arc<dyn LoggingSink>,
    ));
    let budget: Box<dyn DeviceMemoryResource + Send + Sync> =
        Box::new(GlobalDeviceBudget::new(logging, budget_bytes));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(budget_bytes as u64),
        Arc::clone(&runtime),
    ));
    let provider =
        Arc::new(CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory)).ok()?);
    Some(ProviderFixture {
        device,
        memory,
        provider,
    })
}

// ---------------------------------------------------------------
// Executor build (timed and untimed paths share this).
// ---------------------------------------------------------------

fn build_executor(
    fix: &ProviderFixture,
    fixture: &Fixture,
    width: Width,
    gate: bool,
) -> (Executor, ExecutionPlan) {
    let mut compiler = Compiler::new();
    let plan = compiler.compile(TRIANGLE_SOURCE).expect("compile");
    let config = RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(gate));
    let mut executor = Executor::new_with_config(Arc::clone(&fix.provider), config);
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    let buf_e1 = upload_binary(&fix.memory, &fixture.e1, width);
    let buf_e2 = upload_binary(&fix.memory, &fixture.e2, width);
    let buf_e3 = upload_binary(&fix.memory, &fixture.e3, width);
    executor.put_relation("e1", buf_e1);
    executor.put_relation("e2", buf_e2);
    executor.put_relation("e3", buf_e3);
    (executor, plan)
}

// ---------------------------------------------------------------
// Correctness check (run once per fixture cell, outside the timed
// region — panics on divergence so the bench cannot quietly drift).
// ---------------------------------------------------------------

fn download_triples_u32(buf: &CudaBuffer) -> BTreeSet<(u64, u64, u64)> {
    let n = buf.num_rows() as usize;
    if n == 0 {
        return BTreeSet::new();
    }
    let mut bytes = vec![vec![0u8; n * 4]; 3];
    for col_idx in 0..3 {
        unsafe {
            let res = sys::cuMemcpyDtoH_v2(
                bytes[col_idx].as_mut_ptr() as *mut _,
                *buf.column(col_idx).unwrap().device_ptr(),
                bytes[col_idx].len(),
            );
            assert_eq!(res, sys::cudaError_enum::CUDA_SUCCESS);
        }
    }
    (0..n)
        .map(|i| {
            (
                u32::from_le_bytes(bytes[0][i * 4..i * 4 + 4].try_into().unwrap()) as u64,
                u32::from_le_bytes(bytes[1][i * 4..i * 4 + 4].try_into().unwrap()) as u64,
                u32::from_le_bytes(bytes[2][i * 4..i * 4 + 4].try_into().unwrap()) as u64,
            )
        })
        .collect()
}

fn download_triples_u64(buf: &CudaBuffer) -> BTreeSet<(u64, u64, u64)> {
    let n = buf.num_rows() as usize;
    if n == 0 {
        return BTreeSet::new();
    }
    let mut bytes = vec![vec![0u8; n * 8]; 3];
    for col_idx in 0..3 {
        unsafe {
            let res = sys::cuMemcpyDtoH_v2(
                bytes[col_idx].as_mut_ptr() as *mut _,
                *buf.column(col_idx).unwrap().device_ptr(),
                bytes[col_idx].len(),
            );
            assert_eq!(res, sys::cudaError_enum::CUDA_SUCCESS);
        }
    }
    (0..n)
        .map(|i| {
            (
                u64::from_le_bytes(bytes[0][i * 8..i * 8 + 8].try_into().unwrap()),
                u64::from_le_bytes(bytes[1][i * 8..i * 8 + 8].try_into().unwrap()),
                u64::from_le_bytes(bytes[2][i * 8..i * 8 + 8].try_into().unwrap()),
            )
        })
        .collect()
}

fn correctness_check(fix: &ProviderFixture, fixture: &Fixture, width: Width, label: &str) {
    let (mut exec_off, plan_off) = build_executor(fix, fixture, width, false);
    exec_off.execute_plan(&plan_off).expect("execute gate-off");
    let off_counter = exec_off.wcoj_triangle_dispatch_count();
    assert_eq!(
        off_counter, 0,
        "[{label}] gate=false must NOT dispatch; got counter {off_counter}"
    );
    let rows_off = {
        let tri_off = exec_off.store().get("tri").expect("tri gate-off");
        match width {
            Width::U64 => download_triples_u64(tri_off),
            Width::U32 | Width::Symbol => download_triples_u32(tri_off),
        }
    };

    let (mut exec_on, plan_on) = build_executor(fix, fixture, width, true);
    exec_on.execute_plan(&plan_on).expect("execute gate-on");
    let on_counter = exec_on.wcoj_triangle_dispatch_count();
    assert_eq!(
        on_counter, 1,
        "[{label}] gate=true must dispatch exactly once; got counter {on_counter}"
    );
    let rows_on = {
        let tri_on = exec_on.store().get("tri").expect("tri gate-on");
        match width {
            Width::U64 => download_triples_u64(tri_on),
            Width::U32 | Width::Symbol => download_triples_u32(tri_on),
        }
    };

    assert_eq!(
        rows_off.len(),
        rows_on.len(),
        "[{label}] row count mismatch — binary {} vs WCOJ {}",
        rows_off.len(),
        rows_on.len()
    );
    assert_eq!(
        rows_off, rows_on,
        "[{label}] row sets diverge between binary-join and WCOJ paths"
    );
}

// ---------------------------------------------------------------
// Bench cells (one criterion::bench_with_input call per cell).
// ---------------------------------------------------------------

fn bench_cell(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    fix: &ProviderFixture,
    fixture: &Fixture,
    width: Width,
    rows: u32,
    gate: bool,
) {
    let label = format!(
        "{}-{}-gate{}",
        width.label(),
        format_args!("{}K", rows / 1000),
        if gate { "on" } else { "off" }
    );
    group.throughput(Throughput::Elements(fixture.total_rows()));
    group.bench_with_input(BenchmarkId::from_parameter(&label), &(), |b, _| {
        // Long-lived Executor per cell. The executor's cached
        // WCOJ launch stream (`Executor::wcoj_triangle_stream`)
        // is acquired exactly once on first dispatch and reused
        // for every iteration of this cell. Building a fresh
        // Executor per iteration would acquire a new stream
        // each time, draining the runtime's grow-only
        // `StreamPool` after 16 iterations and silently
        // routing the remaining iterations through the
        // binary-join fallback — invalidating the timing.
        //
        // Inputs are re-uploaded each iteration (`put_relation`
        // overwrites) to mirror real-world allocation pressure.
        // The output `tri` relation is removed after each
        // iteration so subsequent dispatches don't pay
        // `union_gpu(growing_tri, new_result)` cost — that
        // would bias the timing as iterations accumulate.
        let (mut executor, plan) = build_executor(fix, fixture, width, gate);
        b.iter_custom(|iters| {
            // Counter delta lock: gate=true must dispatch every
            // iteration (delta == iters); gate=false must never
            // dispatch (delta == 0). Without this in-loop check,
            // a future regression that silently fell back to
            // binary-join (e.g. a stream-pool exhaustion bug
            // re-emerging in a different shape, or a kernel
            // error swallowed by the dispatch hook) would
            // produce valid binary-join *output* with timing
            // labelled as WCOJ — i.e. silently corrupt the
            // baseline. The counter is the source of truth
            // for "WCOJ actually fired"; assert it.
            let counter_before = executor.wcoj_triangle_dispatch_count();
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                // Setup (untimed): fresh GPU uploads each iter.
                executor.put_relation("e1", upload_binary(&fix.memory, &fixture.e1, width));
                executor.put_relation("e2", upload_binary(&fix.memory, &fixture.e2, width));
                executor.put_relation("e3", upload_binary(&fix.memory, &fixture.e3, width));
                // Timed region: execute_plan only.
                let start = Instant::now();
                let _ = executor.execute_plan(&plan).expect("execute_plan");
                total += start.elapsed();
                // Cleanup (untimed): remove `tri` so next iter
                // starts from the same store state.
                let _ = executor.store_mut().remove("tri");
            }
            let counter_after = executor.wcoj_triangle_dispatch_count();
            let delta = counter_after - counter_before;
            let expected = if gate { iters } else { 0 };
            assert_eq!(
                delta,
                expected,
                "[bench cell {label_for_assert}] counter delta {delta} != expected {expected} \
                 across {iters} iterations (gate={gate}). The dispatch path silently fell \
                 back somewhere in the hot loop; recorded timing for this cell is \
                 contaminated.",
                label_for_assert = label,
            );
            total
        });
    });
}

fn default_sizes() -> &'static [u32] {
    &[10_000, 50_000]
}

fn full_extra_sizes() -> &'static [u32] {
    &[100_000, 250_000]
}

fn full_matrix() -> bool {
    std::env::var("WCOJ_BENCH_FULL")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn run_family(
    c: &mut Criterion,
    fix: &ProviderFixture,
    family_label: &str,
    make_fixture: fn(u32) -> Fixture,
) {
    let mut group = c.benchmark_group(format!("wcoj_triangle/{family_label}"));
    group.sample_size(10);

    let mut sizes: Vec<u32> = default_sizes().to_vec();
    if full_matrix() {
        sizes.extend_from_slice(full_extra_sizes());
    }

    for &rows in &sizes {
        let fixture = make_fixture(rows);
        for width in [Width::U32, Width::U64] {
            // Correctness pre-check (untimed). Runs once per
            // (family, size, width) combo; panics on any
            // divergence so the bench can never silently report
            // numbers from a broken kernel.
            let cell_label = format!("{}-{}-{}K", family_label, width.label(), rows / 1000);
            correctness_check(fix, &fixture, width, &cell_label);
            for gate in [false, true] {
                bench_cell(&mut group, fix, &fixture, width, rows, gate);
            }
        }
    }

    group.finish();
}

fn bench_uniform(c: &mut Criterion) {
    let Some(fix) = make_provider(8 * 1024) else {
        eprintln!("Skipping bench_uniform: No CUDA device");
        return;
    };
    run_family(c, &fix, "uniform", make_uniform);
}

fn bench_superhub(c: &mut Criterion) {
    let Some(fix) = make_provider(8 * 1024) else {
        eprintln!("Skipping bench_superhub: No CUDA device");
        return;
    };
    run_family(c, &fix, "superhub", make_superhub);
}

fn bench_empty(c: &mut Criterion) {
    let Some(fix) = make_provider(8 * 1024) else {
        eprintln!("Skipping bench_empty: No CUDA device");
        return;
    };
    run_family(c, &fix, "empty", make_empty);
}

/// One Symbol sanity case at the smallest default size — just
/// confirms the dispatch path is exercised on a Symbol triangle
/// and produces correct output. Not a perf datapoint per se;
/// Symbol shares u32's physical layout so it's expected to track.
fn bench_symbol_sanity(c: &mut Criterion) {
    let Some(fix) = make_provider(8 * 1024) else {
        eprintln!("Skipping bench_symbol_sanity: No CUDA device");
        return;
    };
    let mut group = c.benchmark_group("wcoj_triangle/symbol_sanity");
    group.sample_size(10);
    let rows = 10_000u32;
    let fixture = make_uniform(rows);
    correctness_check(&fix, &fixture, Width::Symbol, "symbol-uniform-10K");
    bench_cell(&mut group, &fix, &fixture, Width::Symbol, rows, true);
    group.finish();
}

// Re-export `BTreeMap` used in build_executor; keeping it in the
// import block above would dwarf the one usage here.
#[allow(dead_code)]
fn _unused() -> BTreeMap<&'static str, ()> {
    BTreeMap::new()
}

criterion_group!(
    name = wcoj_triangle_bench;
    config = Criterion::default();
    targets = bench_uniform, bench_superhub, bench_empty, bench_symbol_sanity
);
criterion_main!(wcoj_triangle_bench);
