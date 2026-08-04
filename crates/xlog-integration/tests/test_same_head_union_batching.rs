//! Many-rule predicate heads must merge with one multiway union per head,
//! not one union per rule (issue #182: stratum-0 union explosion).
//!
//! A corpus-scale program can carry thousands of rules for one head; folding
//! them into the head relation one union at a time re-sorts the growing
//! accumulator per rule and goes quadratic. These tests pin the batched
//! behavior at the `--stats` op level while asserting the derived rows stay
//! byte-identical.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use xlog_core::MemoryBudget;
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::GpuMemoryManager;
use xlog_cuda::{CudaDevice, CudaKernelProvider};
use xlog_gpu::logic::LogicProgram;

fn provider() -> Option<Arc<CudaKernelProvider>> {
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let stream_pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
    let budget = MemoryBudget::with_limit(2 * 1024 * 1024 * 1024);
    let async_resource: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(
        AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&stream_pool)),
    );
    let budgeted: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(GlobalDeviceBudget::new(
        async_resource,
        budget.device_bytes as usize,
    ));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        stream_pool,
        budgeted,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        budget,
        runtime,
    ));
    Some(Arc::new(
        CudaKernelProvider::with_runtime(device, memory).ok()?,
    ))
}

/// Both tests build a full device runtime; running them concurrently in one
/// process makes the fixtures race on the device. Serialize them.
fn test_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// Total (calls, duration_us) per op name across all strata.
fn op_totals(stats: &xlog_runtime::ExecutionStats) -> HashMap<String, (usize, u64)> {
    let mut totals = HashMap::<String, (usize, u64)>::new();
    for stratum in &stats.strata {
        for op in &stratum.ops {
            let entry = totals.entry(op.op_name.clone()).or_default();
            entry.0 += 1;
            entry.1 += op.duration_us;
        }
    }
    totals
}

#[test]
fn non_recursive_many_rule_head_unions_once() {
    let _lock = test_lock();
    let Some(provider) = provider() else {
        eprintln!("CUDA unavailable; skipping");
        return;
    };

    // Eight bridging rules feed one head. Before batching this cost one
    // union per rule (7+ unions); batched it is a single multiway union.
    let source = r#"
in0(1). in0(2).
in1(2). in1(3).
in2(3). in2(4).
in3(4). in3(5).
in4(5). in4(6).
in5(6). in5(1).
in6(7). in6(2).
in7(8). in7(3).

out(X) :- in0(X).
out(X) :- in1(X).
out(X) :- in2(X).
out(X) :- in3(X).
out(X) :- in4(X).
out(X) :- in5(X).
out(X) :- in6(X).
out(X) :- in7(X).

?- out(X).
"#;
    let program = LogicProgram::compile(source).expect("compile");
    let result = program
        .evaluate_with_options(Arc::clone(&provider), HashMap::new(), true)
        .expect("evaluate");

    let mut rows = provider
        .download_column::<u32>(&result.queries[0].buffer, 0)
        .expect("query rows");
    rows.sort_unstable();
    assert_eq!(rows, vec![1, 2, 3, 4, 5, 6, 7, 8]);

    let stats = result.stats.expect("stats");
    let totals = op_totals(&stats);
    let union_ops = totals.get("union").copied().unwrap_or_default().0;
    // Every declared relation is pre-seeded with an empty buffer, so each
    // single-rule SCC (8 fact predicates + 1 query rule) records one union
    // when it merges into its pre-seeded relation, and the eight-rule `out`
    // head records exactly one multiway union. Before batching, `out` alone
    // recorded eight unions (17 total).
    assert_eq!(
        union_ops,
        8 + 1 + 1,
        "eight same-head rules must merge in one multiway union; totals={totals:?}"
    );
}

#[test]
fn recursive_many_rule_head_unions_once_per_pass() {
    let _lock = test_lock();
    let Some(provider) = provider() else {
        eprintln!("CUDA unavailable; skipping");
        return;
    };

    // Twelve base rules plus one recursive rule share the `path` head. The
    // seeding pass must merge all thirteen contributions (plus the prior
    // full relation) in one multiway union; afterwards each iteration adds
    // at most one delta-merge union. Before batching the seeding pass alone
    // cost one union per rule.
    let source = r#"
e0(1, 2).
e1(2, 3).
e2(3, 4).
e3(4, 5).
e4(5, 6).
e5(6, 7).
e6(7, 8).
e7(8, 9).
e8(9, 10).
e9(10, 11).
e10(11, 12).
e11(1, 3).

path(X, Y) :- e0(X, Y).
path(X, Y) :- e1(X, Y).
path(X, Y) :- e2(X, Y).
path(X, Y) :- e3(X, Y).
path(X, Y) :- e4(X, Y).
path(X, Y) :- e5(X, Y).
path(X, Y) :- e6(X, Y).
path(X, Y) :- e7(X, Y).
path(X, Y) :- e8(X, Y).
path(X, Y) :- e9(X, Y).
path(X, Y) :- e10(X, Y).
path(X, Y) :- e11(X, Y).
path(X, Z) :- path(X, Y), e2(Y, Z).

?- path(X, Y).
"#;
    let program = LogicProgram::compile(source).expect("compile");
    let result = program
        .evaluate_with_options(Arc::clone(&provider), HashMap::new(), true)
        .expect("evaluate");

    let xs = provider
        .download_column::<u32>(&result.queries[0].buffer, 0)
        .expect("query xs");
    let ys = provider
        .download_column::<u32>(&result.queries[0].buffer, 1)
        .expect("query ys");
    let mut rows: Vec<(u32, u32)> = xs.into_iter().zip(ys).collect();
    rows.sort_unstable();
    // Base edges plus the recursive extensions through e2 = (3, 4): paths
    // ending in 3 — (2, 3) and (1, 3) — extend to (2, 4) and (1, 4). No
    // path ends in the new targets' source 3 afterwards, so the fixpoint
    // converges on the next iteration.
    let mut expected = vec![
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (1, 3),
        (2, 4),
        (1, 4),
    ];
    expected.sort_unstable();
    assert_eq!(rows, expected);

    let stats = result.stats.expect("stats");
    let recursive_stratum = stats
        .strata
        .iter()
        .find(|s| s.is_recursive)
        .expect("recursive stratum");
    let summary = recursive_stratum.op_summary();
    let union_ops = summary.get("union").copied().unwrap_or_default().0;
    let num_rules = recursive_stratum.num_rules;
    let iterations = recursive_stratum.iterations;
    // The stratum holds 26 rules: 12 facts, 13 `path` rules, 1 query rule.
    // Each single-rule SCC (12 facts + 1 query) merges once into its
    // pre-seeded relation; the recursive `path` SCC seeds all 13 rule
    // contributions in one multiway union and then merges one non-empty
    // delta per productive iteration (the final iteration converges without
    // a merge). Before batching, the seeding pass alone recorded one union
    // per `path` rule (~27 total).
    assert_eq!(
        union_ops,
        12 + 1 + 1 + (iterations - 1),
        "recursive stratum must union once per pass, not once per rule: \
         union_ops={union_ops} rules={num_rules} iterations={iterations} summary={summary:?}"
    );
    assert!(
        union_ops < num_rules,
        "union count must not scale with same-head rule count: \
         union_ops={union_ops} rules={num_rules}"
    );
}
