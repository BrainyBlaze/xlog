//! Cross-stream lifetime stress under in-process parallelism and fork isolation.
//!
//! Closes v0.6.0 release blocker #2: the recorded launch
//! discipline now has a public stress harness that exercises
//! both in-process parallel scheduling against a shared CUDA primary
//! context (fixed and seeded-random schedules) and fresh subprocess
//! isolation with a cold CUDA context and bounded iterations. The
//! provider-owned runtime is invariant; `XLOG_USE_RECORDED_OPS=1`
//! selects the recorded operator routes under stress.
//!
//! # Final gate command
//!
//! ```sh
//! XLOG_USE_RECORDED_OPS=1 \
//!     cargo test -p xlog-integration --test test_cross_stream_lifetime_stress \
//!     --release -- --test-threads=1 --nocapture
//! ```
//!
//! The harness asserts the recorded-operator switch is enabled on
//! parent-mode entry; running without it surfaces a clear "skipped"
//! message rather than exercising a different dispatch route.
//! `--test-threads=1` is required because the parent
//! orchestrates its own intra-test parallelism.
//!
//! # Pass criteria
//!
//!   * Zero failures across all in-process thread × iteration combinations.
//!   * Zero failures across all fork-isolated child × iteration combinations.
//!   * Final in-process thread runtime `bytes_outstanding == 0` after
//!     reap (no allocator leaks).
//!   * No CUDA error / panic / non-zero subprocess exit.
//!   * Result-set checksums match the per-config reference
//!     computed by one serial run on the parent.
//!
//! # Workloads (two — distinct sensitivity profiles)
//!
//!   1. `friends`: sort + hash-join sensitive path
//!      (`social_network_friend_recommendations` shape):
//!      `fof(X,Z) :- friend(X,Y), friend(Y,Z), X != Z` over a
//!      randomized clustered friendship graph.
//!   2. `reach`: recursive fixed-point + joins
//!      (`network_connectivity` / transitive-closure shape):
//!      `reach(X,Y) :- edge(X,Y). reach(X,Z) :- reach(X,Y),
//!      edge(Y,Z).` over a randomized DAG.
//!
//! Both workloads are deterministic given (size_seed, density_seed),
//! so a serial reference checksum applies to every concurrent
//! / forked run with identical params.
//!
//! # Reproducibility
//!
//! Failure messages report `base_seed`, `worker_id`, `iter`,
//! `workload`, `graph params`, and the diverging checksum so a
//! single run can be re-played by re-invoking the harness with
//! the same `XLOG_CROSS_STREAM_STRESS_BASE_SEED` environment variable.

use std::collections::HashSet;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use xlog_core::{read_bool_env, MemoryBudget, ScalarType, Schema};
use xlog_cuda::device_runtime::{LogRecord, LoggingSink, SinkError, XlogDeviceRuntime};
use xlog_cuda::{CudaBuffer, CudaKernelProvider};
use xlog_logic::Compiler;
use xlog_runtime::Executor;

const TEST_BUDGET_BYTES: usize = 512 * 1024 * 1024;

/// Subprocess-mode env marker. Parent sets this when re-invoking
/// `current_exe()`; the test entry point branches on it.
const FORK_CHILD_ENV: &str = "XLOG_CROSS_STREAM_STRESS_CHILD";
/// Optional override for the deterministic base seed. Defaults
/// to `42` when unset; surfaced in every failure message so a
/// repro run is `XLOG_CROSS_STREAM_STRESS_BASE_SEED=<x> cargo test ...`.
const BASE_SEED_ENV: &str = "XLOG_CROSS_STREAM_STRESS_BASE_SEED";

const PARALLEL_THREADS: usize = 8;
const PARALLEL_ITERS_PER_THREAD: usize = 32;

const FORK_CHILDREN: usize = 16;
const FORK_ITERS_PER_CHILD: usize = 4;

// ---------------------------------------------------------------
// Workload identifiers + parameters.
// ---------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Workload {
    Friends,
    Reach,
}

impl Workload {
    fn name(self) -> &'static str {
        match self {
            Workload::Friends => "friends",
            Workload::Reach => "reach",
        }
    }
}

/// Per-iter graph parameters. Deterministic given (workload,
/// size_class, density_class). Kept small so a `--release` run
/// of the full harness completes in tens of seconds rather than
/// minutes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct GraphParams {
    workload: Workload,
    nodes: u32,
    /// Edges per node, integer. Total edges ~ nodes * edges_per_node.
    edges_per_node: u32,
}

impl GraphParams {
    fn label(&self) -> String {
        format!(
            "{}/n={}/e_per_n={}",
            self.workload.name(),
            self.nodes,
            self.edges_per_node
        )
    }
}

/// Fixed schedule used by both stress modes to exercise both workloads at
/// a few sizes. Plus a seeded-random tail picks parameters from
/// the same domain so the workers visit the same param space the
/// reference table covers.
const FIXED_SCHEDULE: &[GraphParams] = &[
    GraphParams {
        workload: Workload::Friends,
        nodes: 64,
        edges_per_node: 4,
    },
    GraphParams {
        workload: Workload::Friends,
        nodes: 128,
        edges_per_node: 6,
    },
    GraphParams {
        workload: Workload::Reach,
        nodes: 32,
        edges_per_node: 3,
    },
    GraphParams {
        workload: Workload::Reach,
        nodes: 64,
        edges_per_node: 5,
    },
];

/// Seeded-random schedule param domain: chosen so all
/// combinations stay inside the fixed-schedule envelope and the
/// reference table can pre-compute every checksum.
const RANDOM_NODE_CHOICES: &[u32] = &[32, 64, 96, 128];
const RANDOM_EDGES_PER_NODE_CHOICES: &[u32] = &[3, 4, 5, 6];

fn enumerate_all_params() -> Vec<GraphParams> {
    let mut out = Vec::new();
    for &workload in &[Workload::Friends, Workload::Reach] {
        for &nodes in RANDOM_NODE_CHOICES {
            for &edges_per_node in RANDOM_EDGES_PER_NODE_CHOICES {
                out.push(GraphParams {
                    workload,
                    nodes,
                    edges_per_node,
                });
            }
        }
    }
    for &p in FIXED_SCHEDULE {
        if !out.contains(&p) {
            out.push(p);
        }
    }
    out
}

// ---------------------------------------------------------------
// Deterministic graph generator (seeded by graph params alone, so
// every worker building the same params gets the same graph).
// ---------------------------------------------------------------

fn graph_seed(p: GraphParams) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    let bytes = [
        p.workload as u8,
        (p.nodes & 0xff) as u8,
        ((p.nodes >> 8) & 0xff) as u8,
        ((p.nodes >> 16) & 0xff) as u8,
        ((p.nodes >> 24) & 0xff) as u8,
        (p.edges_per_node & 0xff) as u8,
        ((p.edges_per_node >> 8) & 0xff) as u8,
    ];
    for b in &bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn build_friend_edges(p: GraphParams) -> Vec<(u32, u32)> {
    debug_assert_eq!(p.workload, Workload::Friends);
    let mut rng = ChaCha8Rng::seed_from_u64(graph_seed(p));
    let mut set: HashSet<(u32, u32)> = HashSet::new();
    let target = (p.nodes as usize) * (p.edges_per_node as usize);
    while set.len() < target {
        let a: u32 = rng.gen_range(1..=p.nodes);
        let b: u32 = rng.gen_range(1..=p.nodes);
        if a != b {
            set.insert((a, b));
            // Bidirectional friendship is the canonical shape.
            set.insert((b, a));
        }
    }
    let mut out: Vec<(u32, u32)> = set.into_iter().collect();
    out.sort();
    out
}

fn build_reach_edges(p: GraphParams) -> Vec<(u32, u32)> {
    debug_assert_eq!(p.workload, Workload::Reach);
    // DAG: only edges (a, b) where a < b. Avoids unbounded
    // recursion blow-up while still exercising fixed-point.
    let mut rng = ChaCha8Rng::seed_from_u64(graph_seed(p));
    let mut set: HashSet<(u32, u32)> = HashSet::new();
    // Include the type-anchor sentinel `(0, 0)` so the buffer
    // matches the program's `edge(0, 0).` literal — without
    // this match, the executor's pre-computed join index for
    // `edge` (built at compile from the literal) goes out of
    // sync with the buffer at execution time and the runtime
    // emits "Join index row count does not match right
    // relation".
    set.insert((0, 0));
    let target = (p.nodes as usize) * (p.edges_per_node as usize);
    while set.len() < target + 1 {
        let a: u32 = rng.gen_range(1..p.nodes);
        let b: u32 = rng.gen_range(a + 1..=p.nodes);
        set.insert((a, b));
    }
    let mut out: Vec<(u32, u32)> = set.into_iter().collect();
    out.sort();
    out
}

// ---------------------------------------------------------------
// Stable result-set checksum: FNV-1a over sorted, encoded rows.
// `DefaultHasher` is intentionally avoided — its semantics are
// not stable across stdlib versions, and fork-isolated children must agree
// even across mildly different toolchain builds.
// ---------------------------------------------------------------

fn checksum_pairs(pairs: &mut [(u32, u32)]) -> u64 {
    pairs.sort();
    let mut h: u64 = 0xcbf29ce484222325;
    for (a, b) in pairs.iter() {
        for &v in &[*a, *b] {
            for shift in 0..4 {
                h ^= ((v >> (shift * 8)) & 0xff) as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        // Row separator so (1,23) and (12,3) hash differently.
        h ^= 0xff;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// ---------------------------------------------------------------
// Runtime-backed executor + provider builder. Locally copied from
// `real_world_tests.rs` per the design rule: no shared-helper
// refactor. The fixture is asserted to be active (the parent
// will not start a stress run without the required provider runtime).
// ---------------------------------------------------------------

struct DiscardSink;

impl LoggingSink for DiscardSink {
    fn emit(&self, _record: LogRecord) -> Result<(), SinkError> {
        Ok(())
    }
}

struct RuntimeFixture {
    provider: Arc<CudaKernelProvider>,
    /// Held to keep the device-runtime + stream pool alive for
    /// the lifetime of the fixture. The harness does not call
    /// runtime methods directly; on drop the runtime tears down
    /// in stack-FILO order via the `Arc` chain.
    _runtime: Arc<XlogDeviceRuntime>,
}

fn build_runtime_fixture() -> Option<RuntimeFixture> {
    let provider = Arc::new(
        xlog_cuda::CudaProviderBuilder::new(0, MemoryBudget::with_limit(TEST_BUDGET_BYTES as u64))
            .with_logging_sink(Arc::new(DiscardSink) as Arc<dyn LoggingSink>)
            .build()
            .ok()?,
    );
    let _device = Arc::clone(provider.device());
    let memory = Arc::clone(provider.memory());
    let runtime = Arc::clone(memory.runtime()?);
    let _pool = Arc::clone(runtime.stream_pool());
    Some(RuntimeFixture {
        provider,
        _runtime: runtime,
    })
}

fn create_edge_buffer(provider: &CudaKernelProvider, edges: &[(u32, u32)]) -> CudaBuffer {
    let schema = Schema::new(vec![
        ("c0".to_string(), ScalarType::U32),
        ("c1".to_string(), ScalarType::U32),
    ]);
    if edges.is_empty() {
        return provider
            .create_empty_buffer(schema)
            .expect("create empty buffer");
    }
    let col0: Vec<u8> = edges.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
    let col1: Vec<u8> = edges.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
    provider
        .create_buffer_from_slices(&[&col0, &col1], schema)
        .expect("create buffer")
}

fn read_pairs(provider: &CudaKernelProvider, buffer: &CudaBuffer) -> Vec<(u32, u32)> {
    if buffer.is_empty() || buffer.column(0).is_none() || buffer.column(1).is_none() {
        return Vec::new();
    }
    let c0 = provider
        .download_column::<u32>(buffer, 0)
        .unwrap_or_default();
    let c1 = provider
        .download_column::<u32>(buffer, 1)
        .unwrap_or_default();
    // Mirror `real_world_tests::read_pairs`: zip drops any
    // trailing slack (e.g., row_cap padding) automatically.
    c0.into_iter().zip(c1).collect()
}

// ---------------------------------------------------------------
// Workload programs.
// ---------------------------------------------------------------

const FRIENDS_PROGRAM: &str = r#"
    fof(X, Z) :- friend(X, Y), friend(Y, Z), X != Z.
"#;

// Type-anchor seed: a sentinel `(0, 0)` edge fact gives the
// compiler a concrete U32×U32 schema for `edge` before the
// recursive rule for `reach` is type-checked. Without this seed
// the compiler infers different widths for `Y` in `reach(X, Y)`
// vs `edge(Y, Z)` and rejects the Union of the two recursive
// branches with "Schema mismatch". The `0` rows do not appear
// in any randomized graph (graphs use ids ≥ 1) so the seed is
// idempotent under set semantics — if it does end up in the
// result it appears identically across every worker and the
// checksum stays stable.
const REACH_PROGRAM: &str = r#"
    edge(0, 0).
    reach(X, Y) :- edge(X, Y).
    reach(X, Z) :- reach(X, Y), edge(Y, Z).
"#;

/// Run one iteration of a workload against a FRESH runtime +
/// provider + executor. Each iter gets its own runtime stack so
/// the workload-level state (executor relation store, provider
/// kernel join-index cache, etc.) cannot bleed across iters
/// when params change.
///
/// The stream-safety property the harness targets — "many
/// concurrent / sequential runtime stacks coexisting on a
/// shared CUDA primary context produce identical results" —
/// is still exercised: every fresh fixture allocates against
/// the same per-process CUDA primary context, the access-aware
/// prepare/finish code paths run on every kernel launch, and
/// concurrent in-process workers hit the context concurrently.
/// Sharing one fixture across iters is NOT required for that
/// property; sharing the CUDA context (which we do) is.
/// Process-wide mutex serializing the `xlog_logic::Compiler`
/// step. The compiler's symbol-interning + relation-id table
/// is process-global; under concurrent in-process compile calls
/// it produces inconsistent rel-id orderings that read out as
/// per-thread drift in the result-set. Serializing compile is
/// not a stream-safety concern (it sits above the GPU layer)
/// and removes the noise so the parallel stress measures what it is meant to
/// measure.
fn compile_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// Run one workload iter against a CALLER-PROVIDED fixture.
/// The caller controls the fixture lifetime — fresh-per-iter,
/// per-thread, or shared across threads — so the diagnostic
/// matrix can isolate cross-runtime churn from shared-runtime
/// concurrency.
fn run_workload_in(fx: &RuntimeFixture, p: GraphParams) -> Result<u64, String> {
    let (program, fact_relation) = match p.workload {
        Workload::Friends => (FRIENDS_PROGRAM, "friend"),
        Workload::Reach => (REACH_PROGRAM, "edge"),
    };
    let edges = match p.workload {
        Workload::Friends => build_friend_edges(p),
        Workload::Reach => build_reach_edges(p),
    };

    let (compiler, plan) = {
        let _guard = compile_lock();
        let mut compiler = Compiler::new();
        let plan = compiler
            .compile(program)
            .map_err(|e| format!("compile {}: {}", p.label(), e))?;
        (compiler, plan)
    };

    let buffer = create_edge_buffer(&fx.provider, &edges);
    // Per-iter executor so state from prior iters does not
    // poison this one's derived relations.
    let mut executor = Executor::new(Arc::clone(&fx.provider));
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    executor.store_mut().put(fact_relation, buffer);

    executor
        .execute_plan(&plan)
        .map_err(|e| format!("execute {}: {}", p.label(), e))?;

    // Pull the derived relation. `fof` for friends, `reach` for reach.
    let derived = match p.workload {
        Workload::Friends => "fof",
        Workload::Reach => "reach",
    };
    let dst = executor
        .store()
        .get(derived)
        .ok_or_else(|| format!("derived {} missing for {}", derived, p.label()))?;
    let mut pairs = read_pairs(&fx.provider, dst);
    Ok(checksum_pairs(&mut pairs))
}

/// Original convenience: build a fresh runtime fixture for one
/// iter. Equivalent to mode `per_iter` in the diagnostic
/// matrix. Used by the reference-table builder and fork-isolated children
/// (where per-iter fresh is the desired semantics).
fn run_workload_once(p: GraphParams) -> Result<u64, String> {
    let fx = build_runtime_fixture()
        .ok_or_else(|| format!("build_runtime_fixture for {}", p.label()))?;
    run_workload_in(&fx, p)
}

/// Diagnostic-matrix fixture-mode selector for in-process stress. Set via
/// `XLOG_PARALLEL_STRESS_FIXTURE_MODE` to one of:
///   * `per_iter` (default) — every iter builds a fresh runtime
///     fixture. Tests cross-runtime-churn under one CUDA primary
///     context.
///   * `per_thread` — each thread builds ONE fixture, reuses it
///     across all iters. Isolates whether single-runtime
///     thread-of-N usage is safe.
///   * `shared` — one process-wide fixture shared across all
///     threads. Isolates whether one runtime under N concurrent
///     callers is safe.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ParallelFixtureMode {
    PerIter,
    PerThread,
    Shared,
}

impl ParallelFixtureMode {
    fn from_env() -> Self {
        match std::env::var("XLOG_PARALLEL_STRESS_FIXTURE_MODE")
            .as_deref()
            .map(str::trim)
        {
            Ok("per_thread") => ParallelFixtureMode::PerThread,
            Ok("shared") => ParallelFixtureMode::Shared,
            _ => ParallelFixtureMode::PerIter,
        }
    }

    fn name(self) -> &'static str {
        match self {
            ParallelFixtureMode::PerIter => "per_iter",
            ParallelFixtureMode::PerThread => "per_thread",
            ParallelFixtureMode::Shared => "shared",
        }
    }
}

// ---------------------------------------------------------------
// Reference table: per-`GraphParams`, what's the expected
// checksum? Computed once on the parent against the runtime
// fixture, then propagated to fork-isolated children through an environment value.
// ---------------------------------------------------------------

/// Compute the reference checksum table. With fresh
/// fixture-per-iter inside `run_workload_once`, every call
/// runs against a clean runtime/provider/executor — there is
/// no cold/warm or order-dependence to compensate for.
fn compute_reference_table(params: &[GraphParams]) -> Result<Vec<(GraphParams, u64)>, String> {
    let mut out = Vec::with_capacity(params.len());
    for &p in params {
        let cs = run_workload_once(p)?;
        out.push((p, cs));
    }
    Ok(out)
}

// ---------------------------------------------------------------
// In-process parallel stress.
// ---------------------------------------------------------------

#[derive(Debug)]
struct StressFailure {
    worker_kind: &'static str,
    worker_id: usize,
    iter: usize,
    params: GraphParams,
    base_seed: u64,
    detail: String,
}

impl StressFailure {
    fn render(&self) -> String {
        format!(
            "[{kind} worker={id} iter={it} params={lbl} base_seed={seed}] {d}",
            kind = self.worker_kind,
            id = self.worker_id,
            it = self.iter,
            lbl = self.params.label(),
            seed = self.base_seed,
            d = self.detail,
        )
    }
}

/// Per-thread schedule: fixed prefix then seeded random tail.
fn schedule_for_worker(base_seed: u64, worker_id: usize, iters: usize) -> Vec<GraphParams> {
    let mut sched: Vec<GraphParams> = FIXED_SCHEDULE.to_vec();
    let want_random = iters.saturating_sub(sched.len());
    if want_random > 0 {
        let worker_seed = base_seed
            .wrapping_add(worker_id as u64)
            .wrapping_mul(0x9E3779B97F4A7C15);
        let mut rng = ChaCha8Rng::seed_from_u64(worker_seed);
        for _ in 0..want_random {
            let workload = if rng.gen::<bool>() {
                Workload::Friends
            } else {
                Workload::Reach
            };
            let nodes = RANDOM_NODE_CHOICES[rng.gen_range(0..RANDOM_NODE_CHOICES.len())];
            let edges_per_node = RANDOM_EDGES_PER_NODE_CHOICES
                [rng.gen_range(0..RANDOM_EDGES_PER_NODE_CHOICES.len())];
            sched.push(GraphParams {
                workload,
                nodes,
                edges_per_node,
            });
        }
    }
    sched.truncate(iters);
    sched
}

fn run_parallel(base_seed: u64, reference: &[(GraphParams, u64)]) -> Vec<StressFailure> {
    let reference_map: std::collections::HashMap<GraphParams, u64> =
        reference.iter().copied().collect();
    let reference_arc = Arc::new(reference_map);

    // Pre-warm CUDA kernel modules in the parent thread before
    // spawning parallel workers. First-launch module load on the CUDA
    // primary context is fragile under concurrent attempts.
    for &p in reference.iter().map(|(p, _)| p) {
        let _ = run_workload_once(p);
    }

    let mode = ParallelFixtureMode::from_env();
    eprintln!("[parallel stress] fixture mode = {}", mode.name());

    // For `Shared` mode, build the single fixture in the parent
    // and clone its Arcs into each thread.
    let shared_fixture: Option<Arc<RuntimeFixture>> = if mode == ParallelFixtureMode::Shared {
        Some(Arc::new(
            build_runtime_fixture().expect("build_runtime_fixture for Shared mode"),
        ))
    } else {
        None
    };

    let failures: Arc<Mutex<Vec<StressFailure>>> = Arc::new(Mutex::new(Vec::new()));
    let pass_counter = Arc::new(AtomicUsize::new(0));

    let reference = reference_arc;
    let mut handles = Vec::with_capacity(PARALLEL_THREADS);
    for tid in 0..PARALLEL_THREADS {
        let reference = Arc::clone(&reference);
        let failures = Arc::clone(&failures);
        let pass_counter = Arc::clone(&pass_counter);
        let shared = shared_fixture.as_ref().map(Arc::clone);
        handles.push(thread::spawn(move || {
            let schedule = schedule_for_worker(base_seed, tid, PARALLEL_ITERS_PER_THREAD);

            // PerThread: each thread builds ONE fixture, reuses
            // for every iter. PerIter / Shared: handled inside
            // the loop.
            let per_thread_fx: Option<RuntimeFixture> = match mode {
                ParallelFixtureMode::PerThread => Some(match build_runtime_fixture() {
                    Some(fx) => fx,
                    None => {
                        failures.lock().unwrap().push(StressFailure {
                            worker_kind: "parallel",
                            worker_id: tid,
                            iter: 0,
                            params: schedule[0],
                            base_seed,
                            detail: "build_runtime_fixture for PerThread mode failed".to_string(),
                        });
                        return;
                    }
                }),
                _ => None,
            };

            for (it, p) in schedule.iter().enumerate() {
                let result = match mode {
                    ParallelFixtureMode::PerIter => run_workload_once(*p),
                    ParallelFixtureMode::PerThread => {
                        run_workload_in(per_thread_fx.as_ref().unwrap(), *p)
                    }
                    ParallelFixtureMode::Shared => run_workload_in(shared.as_ref().unwrap(), *p),
                };
                match result {
                    Ok(cs) => {
                        let expected = reference
                            .get(p)
                            .copied()
                            .expect("reference must cover all scheduled params");
                        if cs != expected {
                            failures.lock().unwrap().push(StressFailure {
                                worker_kind: "parallel",
                                worker_id: tid,
                                iter: it,
                                params: *p,
                                base_seed,
                                detail: format!(
                                    "checksum drift: got {:#x} expected {:#x}",
                                    cs, expected
                                ),
                            });
                            return;
                        }
                        pass_counter.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(msg) => {
                        failures.lock().unwrap().push(StressFailure {
                            worker_kind: "parallel",
                            worker_id: tid,
                            iter: it,
                            params: *p,
                            base_seed,
                            detail: msg,
                        });
                        return;
                    }
                }
            }
        }));
    }
    for h in handles {
        h.join().expect("parallel stress thread join");
    }
    let total = PARALLEL_THREADS * PARALLEL_ITERS_PER_THREAD;
    let passed = pass_counter.load(Ordering::Relaxed);
    eprintln!(
        "[parallel stress] pass={}/{} failures={}",
        passed,
        total,
        failures.lock().unwrap().len()
    );
    Arc::try_unwrap(failures).unwrap().into_inner().unwrap()
}

// ---------------------------------------------------------------
// Fork-isolated subprocess stress.
//
// Parent serializes the reference table to a single env var
// (`XLOG_CROSS_STREAM_STRESS_REFERENCE`), then spawns N children with
// `XLOG_CROSS_STREAM_STRESS_CHILD=<id>` and
// `XLOG_CROSS_STREAM_STRESS_BASE_SEED=<seed>`.
// Each child boots cold, runs `FORK_ITERS_PER_CHILD` iterations of a
// per-child seeded schedule, exits 0 on full pass / non-zero on
// any failure (with a structured failure line on stderr).
// ---------------------------------------------------------------

const FORK_REFERENCE_ENV: &str = "XLOG_CROSS_STREAM_STRESS_REFERENCE";

fn serialize_reference(reference: &[(GraphParams, u64)]) -> String {
    // Tiny, brittle-on-purpose: each tuple as
    // "workload,nodes,edges_per_node,checksum" pipe-separated.
    let mut out = String::new();
    for (p, cs) in reference {
        if !out.is_empty() {
            out.push('|');
        }
        out.push_str(p.workload.name());
        out.push(',');
        out.push_str(&p.nodes.to_string());
        out.push(',');
        out.push_str(&p.edges_per_node.to_string());
        out.push(',');
        out.push_str(&format!("{:x}", cs));
    }
    out
}

fn parse_reference(s: &str) -> Vec<(GraphParams, u64)> {
    s.split('|')
        .filter(|t| !t.is_empty())
        .map(|t| {
            let parts: Vec<&str> = t.split(',').collect();
            assert_eq!(parts.len(), 4, "malformed reference tuple {}", t);
            let workload = match parts[0] {
                "friends" => Workload::Friends,
                "reach" => Workload::Reach,
                other => panic!("unknown workload {}", other),
            };
            let nodes: u32 = parts[1].parse().expect("nodes parse");
            let edges_per_node: u32 = parts[2].parse().expect("epn parse");
            let cs = u64::from_str_radix(parts[3], 16).expect("cs parse");
            (
                GraphParams {
                    workload,
                    nodes,
                    edges_per_node,
                },
                cs,
            )
        })
        .collect()
}

fn run_fork_isolated(base_seed: u64, reference: &[(GraphParams, u64)]) -> Vec<StressFailure> {
    let exe = std::env::current_exe().expect("current_exe");
    let reference_env = serialize_reference(reference);
    let mut failures: Vec<StressFailure> = Vec::new();
    let mut pass_count = 0usize;

    for child_id in 0..FORK_CHILDREN {
        let mut cmd = Command::new(&exe);
        // Re-invoke the SAME test binary, filtering to a single
        // tiny test fn (`fork_child_marker`) that branches on the
        // child env var. `--exact` + `--nocapture` keep the
        // child output ungated by libtest filtering.
        cmd.arg("--test-threads=1")
            .arg("--exact")
            .arg("--nocapture")
            .arg("fork_child_marker");
        cmd.env(FORK_CHILD_ENV, child_id.to_string());
        cmd.env(BASE_SEED_ENV, base_seed.to_string());
        cmd.env(FORK_REFERENCE_ENV, &reference_env);
        // Forward recorded-operator selection to the child.
        cmd.env("XLOG_USE_RECORDED_OPS", "1");

        let output = cmd.output();
        match output {
            Ok(out) if out.status.success() => {
                pass_count += 1;
            }
            Ok(out) => {
                let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
                let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
                failures.push(StressFailure {
                    worker_kind: "fork-isolated",
                    worker_id: child_id,
                    iter: 0,
                    params: FIXED_SCHEDULE[0],
                    base_seed,
                    detail: format!(
                        "child exit {:?}; stderr=<<<{}>>> stdout=<<<{}>>>",
                        out.status.code(),
                        stderr.trim(),
                        stdout.trim(),
                    ),
                });
            }
            Err(e) => {
                failures.push(StressFailure {
                    worker_kind: "fork-isolated",
                    worker_id: child_id,
                    iter: 0,
                    params: FIXED_SCHEDULE[0],
                    base_seed,
                    detail: format!("spawn child: {}", e),
                });
            }
        }
    }
    eprintln!(
        "[fork-isolated stress] pass={}/{} failures={}",
        pass_count,
        FORK_CHILDREN,
        failures.len()
    );
    failures
}

/// Marker test function the fork-isolated parent re-invokes via subprocess. Acts
/// as the single entry point for child-mode work. When the
/// child env var is unset, this is a no-op (so a normal
/// `cargo test` run does not double-execute fork-isolated work).
#[test]
fn fork_child_marker() {
    let Ok(child_id_s) = std::env::var(FORK_CHILD_ENV) else {
        return;
    };
    let child_id: usize = child_id_s.parse().expect("child id parse");
    let base_seed: u64 = std::env::var(BASE_SEED_ENV)
        .expect("BASE_SEED_ENV")
        .parse()
        .expect("base_seed parse");
    let reference_env = std::env::var(FORK_REFERENCE_ENV).expect("reference env");
    let reference: std::collections::HashMap<GraphParams, u64> =
        parse_reference(&reference_env).into_iter().collect();

    let schedule = schedule_for_worker(base_seed, child_id, FORK_ITERS_PER_CHILD);
    for (it, p) in schedule.iter().enumerate() {
        match run_workload_once(*p) {
            Ok(cs) => {
                let expected = reference
                    .get(p)
                    .copied()
                    .expect("reference must cover scheduled params");
                if cs != expected {
                    eprintln!(
                        "[fork-isolated child {} iter {} {}] checksum drift: got {:#x} expected {:#x}",
                        child_id,
                        it,
                        p.label(),
                        cs,
                        expected,
                    );
                    std::process::exit(3);
                }
            }
            Err(msg) => {
                eprintln!(
                    "[fork-isolated child {} iter {} {}] error: {}",
                    child_id,
                    it,
                    p.label(),
                    msg,
                );
                std::process::exit(4);
            }
        }
    }
    // Successful child exits via std::process::exit(0) below.
    // libtest would otherwise complete the test fn cleanly,
    // which already implies exit(0); we make it explicit so
    // future libtest behavior changes do not silently break fork isolation.
    std::process::exit(0);
}

// ---------------------------------------------------------------
// Parent test fn — the gate.
// ---------------------------------------------------------------

fn parent_env_check() -> Result<u64, String> {
    match read_bool_env("XLOG_USE_RECORDED_OPS").map_err(|error| error.to_string())? {
        Some(true) => {}
        Some(false) | None => return Err("XLOG_USE_RECORDED_OPS is not enabled".to_string()),
    }
    let base_seed: u64 = std::env::var(BASE_SEED_ENV)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    Ok(base_seed)
}

#[test]
fn cross_stream_lifetime_stress() {
    // If we are running as a fork-isolated child, the marker test handles
    // it; the parent test exits early.
    if std::env::var(FORK_CHILD_ENV).is_ok() {
        return;
    }

    let base_seed = match parent_env_check() {
        Ok(s) => s,
        Err(why) => {
            eprintln!(
                "[cross-stream lifetime stress] SKIPPED: {} — gate command is \
                 `XLOG_USE_RECORDED_OPS=1 \
                 cargo test -p xlog-integration --test test_cross_stream_lifetime_stress \
                 --release -- --test-threads=1 --nocapture`",
                why
            );
            return;
        }
    };
    eprintln!("[cross-stream lifetime stress] base_seed={}", base_seed);

    let all_params = enumerate_all_params();
    eprintln!(
        "[cross-stream lifetime stress] computing reference for {} parameter tuples…",
        all_params.len()
    );
    let reference = match compute_reference_table(&all_params) {
        Ok(t) => t,
        Err(e) => {
            eprintln!(
                "[cross-stream lifetime stress] reference table failed: {}",
                e
            );
            // No CUDA → skip rather than fail, matching
            // real_world_tests behaviour.
            return;
        }
    };
    eprintln!(
        "[cross-stream lifetime stress] reference checksums computed (len={})",
        reference.len()
    );

    let parallel_failures = run_parallel(base_seed, &reference);
    let fork_failures = run_fork_isolated(base_seed, &reference);

    let mut all = Vec::new();
    all.extend(parallel_failures);
    all.extend(fork_failures);

    if !all.is_empty() {
        eprintln!(
            "\n[cross-stream lifetime stress] FAILURE SUMMARY (first {})",
            all.len().min(10)
        );
        for f in all.iter().take(10) {
            eprintln!("  {}", f.render());
        }
        // Classify by symptom for the summary line — a tally,
        // not a gate. The gate is "any failure at all".
        let stream_misuse = all
            .iter()
            .filter(|f| f.detail.contains("stream-ordered contract"))
            .count();
        let uaf = all
            .iter()
            .filter(|f| f.detail.contains("use-after-free"))
            .count();
        let drift = all
            .iter()
            .filter(|f| f.detail.contains("checksum drift"))
            .count();
        let leak = all
            .iter()
            .filter(|f| f.detail.contains("runtime leak"))
            .count();
        eprintln!(
            "[cross-stream lifetime stress] symptom tally: stream-misuse={} uaf={} drift={} leak={} other={}",
            stream_misuse,
            uaf,
            drift,
            leak,
            all.len() - stream_misuse - uaf - drift - leak,
        );
        panic!(
            "cross-stream lifetime stress harness reported {} failures",
            all.len()
        );
    }

    eprintln!(
        "[cross-stream lifetime stress] PASS — parallel: {}/{} fork-isolated: {}/{}",
        PARALLEL_THREADS * PARALLEL_ITERS_PER_THREAD,
        PARALLEL_THREADS * PARALLEL_ITERS_PER_THREAD,
        FORK_CHILDREN,
        FORK_CHILDREN,
    );
}
