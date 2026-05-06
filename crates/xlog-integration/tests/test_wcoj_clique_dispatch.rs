// crates/xlog-integration/tests/test_wcoj_clique_dispatch.rs
//! W3.2 — Runtime dispatch certs for k=5/k=6 clique WCOJ.
//!
//! 4 cells:
//!   1. clique5 counter advances + row set matches MultiWayJoin.fallback.
//!   2. clique6 same at k=6.
//!   3. clique5 dispatcher decline does NOT advance counter +
//!      row set matches fallback (malformed dispatch path).
//!   4. clique6 same.
//!
//! Tests 1 + 2 build a small K-clique rule via the compiler, run
//! under default config, assert
//! `executor.wcoj_clique{5,6}_dispatch_count() >= 1` AND row set
//! equals the body that would result from `MultiWayJoin.fallback`
//! (built via a test-only RIR rewrite helper that substitutes
//! MultiWayJoin nodes with their fallback field). NO new
//! force/kill/adaptive runtime knobs (per W3.2 D8 lock).
//!
//! Tests 3 + 4 deliberately route through a code path where the
//! promoter emits MultiWayJoin but the dispatcher declines
//! internally. The decline is engineered by NOT registering one
//! of the clique's input relations on the executor — the
//! dispatcher's relation-resolve step returns None →
//! Ok(None) → counter stays at 0 → executor falls through to
//! `MultiWayJoin.fallback` for the row set. The fallback path
//! has the same un-registered rel, so it also fails — but we
//! verify the COUNTER state independently.

use std::collections::BTreeMap;
use std::sync::Arc;

use cudarc::driver::sys;
use xlog_core::{MemoryBudget, RuntimeConfig, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::CudaBuffer;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_ir::{ExecutionPlan, RirNode};
use xlog_logic::Compiler;
use xlog_runtime::Executor;

struct DiscardSink;
impl LoggingSink for DiscardSink {
    fn emit(&self, _record: LogRecord) -> Result<(), SinkError> {
        Ok(())
    }
}

#[allow(dead_code)]
struct RuntimeBackedFixture {
    device: Arc<CudaDevice>,
    runtime: Arc<XlogDeviceRuntime>,
    memory: Arc<GpuMemoryManager>,
    provider: Arc<CudaKernelProvider>,
    pool: Arc<StreamPool>,
}

fn make_runtime_backed_fixture() -> Option<RuntimeBackedFixture> {
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
    let async_r: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(AsyncCudaResource::new(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
    ));
    let logging: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(LoggingResource::new(
        async_r,
        Arc::new(DiscardSink) as Arc<dyn LoggingSink>,
    ));
    let budget: Box<dyn DeviceMemoryResource + Send + Sync> =
        Box::new(GlobalDeviceBudget::new(logging, 64 * 1024 * 1024));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(64 * 1024 * 1024),
        Arc::clone(&runtime),
    ));
    let provider =
        Arc::new(CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory)).ok()?);
    Some(RuntimeBackedFixture {
        device,
        runtime,
        memory,
        provider,
        pool,
    })
}

fn upload_binary_u32(memory: &Arc<GpuMemoryManager>, rows: &[(u32, u32)]) -> CudaBuffer {
    let n = rows.len() as u32;
    let bpc = (n as usize).max(1) * 4;
    let mut col0 = memory.alloc::<u8>(bpc).expect("alloc col0");
    let mut col1 = memory.alloc::<u8>(bpc).expect("alloc col1");
    let mut d_n = memory.alloc::<u32>(1).expect("alloc d_n");
    let dev = memory.device().inner();
    if n > 0 {
        let c0: Vec<u8> = rows.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
        let c1: Vec<u8> = rows.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
        dev.htod_sync_copy_into(&c0, &mut col0).unwrap();
        dev.htod_sync_copy_into(&c1, &mut col1).unwrap();
    }
    dev.htod_sync_copy_into(&[n], &mut d_n).unwrap();
    let schema = Schema::new(vec![
        ("c0".to_string(), ScalarType::U32),
        ("c1".to_string(), ScalarType::U32),
    ]);
    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into()],
        n as u64,
        d_n,
        schema,
        n,
    )
}

/// XLOG source for K_5 clique evaluation. 10 edges in canonical
/// (i, j) order: e01, e02, e03, e04, e12, e13, e14, e23, e24, e34.
const CLIQUE5_SRC: &str = r#"
    pred e01(u32, u32). pred e02(u32, u32). pred e03(u32, u32). pred e04(u32, u32).
    pred e12(u32, u32). pred e13(u32, u32). pred e14(u32, u32).
    pred e23(u32, u32). pred e24(u32, u32).
    pred e34(u32, u32).
    pred clique5(u32, u32, u32, u32, u32).
    clique5(V0, V1, V2, V3, V4) :-
        e01(V0, V1), e02(V0, V2), e03(V0, V3), e04(V0, V4),
        e12(V1, V2), e13(V1, V3), e14(V1, V4),
        e23(V2, V3), e24(V2, V4),
        e34(V3, V4).
"#;

/// XLOG source for K_6 clique. 15 edges.
const CLIQUE6_SRC: &str = r#"
    pred e01(u32, u32). pred e02(u32, u32). pred e03(u32, u32).
    pred e04(u32, u32). pred e05(u32, u32).
    pred e12(u32, u32). pred e13(u32, u32). pred e14(u32, u32). pred e15(u32, u32).
    pred e23(u32, u32). pred e24(u32, u32). pred e25(u32, u32).
    pred e34(u32, u32). pred e35(u32, u32).
    pred e45(u32, u32).
    pred clique6(u32, u32, u32, u32, u32, u32).
    clique6(V0, V1, V2, V3, V4, V5) :-
        e01(V0, V1), e02(V0, V2), e03(V0, V3), e04(V0, V4), e05(V0, V5),
        e12(V1, V2), e13(V1, V3), e14(V1, V4), e15(V1, V5),
        e23(V2, V3), e24(V2, V4), e25(V2, V5),
        e34(V3, V4), e35(V3, V5),
        e45(V4, V5).
"#;

/// Build a complete-K_K fixture on K vertices. Returns
/// `[(edge_name, rows)]`.
fn k_clique_inputs(k: usize) -> BTreeMap<String, Vec<(u32, u32)>> {
    let mut m: BTreeMap<String, Vec<(u32, u32)>> = BTreeMap::new();
    for i in 0u32..(k as u32) {
        for j in (i + 1)..(k as u32) {
            // Edge name e{i}{j} carries the single tuple (i+1, j+1).
            let name = format!("e{}{}", i, j);
            m.insert(name, vec![(i + 1, j + 1)]);
        }
    }
    m
}

/// Test-only RIR rewrite helper: walk the plan tree, detect
/// `RirNode::MultiWayJoin` nodes, and substitute each with its
/// `fallback` field. Used to build the binary-join reference
/// row set without introducing new force/kill/adaptive knobs
/// (per W3.2 D8 lock).
fn replace_multiway_with_fallback(mut plan: ExecutionPlan) -> ExecutionPlan {
    fn rewrite(node: &RirNode) -> RirNode {
        match node {
            RirNode::MultiWayJoin { fallback, .. } => rewrite(fallback),
            RirNode::Project { input, columns } => RirNode::Project {
                input: Box::new(rewrite(input)),
                columns: columns.clone(),
            },
            RirNode::Filter { input, predicate } => RirNode::Filter {
                input: Box::new(rewrite(input)),
                predicate: predicate.clone(),
            },
            RirNode::Join {
                left,
                right,
                left_keys,
                right_keys,
                join_type,
            } => RirNode::Join {
                left: Box::new(rewrite(left)),
                right: Box::new(rewrite(right)),
                left_keys: left_keys.clone(),
                right_keys: right_keys.clone(),
                join_type: *join_type,
            },
            RirNode::Union { inputs } => RirNode::Union {
                inputs: inputs.iter().map(rewrite).collect(),
            },
            RirNode::Diff { left, right } => RirNode::Diff {
                left: Box::new(rewrite(left)),
                right: Box::new(rewrite(right)),
            },
            RirNode::Distinct { input, key_cols } => RirNode::Distinct {
                input: Box::new(rewrite(input)),
                key_cols: key_cols.clone(),
            },
            other => other.clone(),
        }
    }
    for rules in plan.rules_by_scc.iter_mut() {
        for rule in rules.iter_mut() {
            rule.body = rewrite(&rule.body);
        }
    }
    plan
}

fn download_k_row_set(buf: &CudaBuffer, k: usize) -> std::collections::BTreeSet<Vec<u32>> {
    let n = match buf.cached_row_count() {
        Some(c) => c as usize,
        None => {
            let mut count = [0u32; 1];
            unsafe {
                sys::cuMemcpyDtoH_v2(
                    count.as_mut_ptr() as *mut _,
                    *buf.num_rows_device().device_ptr(),
                    4,
                );
            }
            count[0] as usize
        }
    };
    if n == 0 {
        return std::collections::BTreeSet::new();
    }
    let mut cols: Vec<Vec<u8>> = (0..k).map(|_| vec![0u8; n * 4]).collect();
    for c in 0..k {
        unsafe {
            sys::cuMemcpyDtoH_v2(
                cols[c].as_mut_ptr() as *mut _,
                *buf.column(c).unwrap().device_ptr(),
                cols[c].len(),
            );
        }
    }
    (0..n)
        .map(|r| {
            (0..k)
                .map(|c| {
                    let off = r * 4;
                    u32::from_le_bytes([
                        cols[c][off],
                        cols[c][off + 1],
                        cols[c][off + 2],
                        cols[c][off + 3],
                    ])
                })
                .collect()
        })
        .collect()
}

/// Run a clique-K test: compile, build two executors (dispatch
/// path + fallback-only path), assert counter ≥ 1 on dispatch
/// AND row-set equality between dispatch output and
/// `replace_multiway_with_fallback` reference.
fn run_counter_advance_test(
    fix: &RuntimeBackedFixture,
    src: &str,
    head_name: &str,
    k: usize,
    check_counter: fn(&Executor) -> u64,
) {
    let inputs = k_clique_inputs(k);

    // 1. Dispatch path: compile + run under default dispatch.
    let mut compiler = Compiler::new();
    let plan = compiler.compile(src).expect("compile");
    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rid) in compiler.rel_ids().clone() {
        executor.register_relation(rid, &name);
    }
    for (name, rows) in &inputs {
        executor.put_relation(name, upload_binary_u32(&fix.memory, rows));
    }
    let _ = executor.execute_plan(&plan).expect("execute dispatch");
    let dispatch_rows = download_k_row_set(executor.store().get(head_name).expect("head"), k);
    assert!(
        check_counter(&executor) >= 1,
        "expected ≥ 1 clique dispatch; got {}",
        check_counter(&executor)
    );

    // 2. Fallback reference: rewrite the plan to replace
    //    MultiWayJoin nodes with their fallback bodies, then
    //    run on a fresh executor with the same inputs. This
    //    exercises the binary-join path without any new
    //    force/kill/adaptive knobs (per W3.2 D8 lock).
    let mut compiler_ref = Compiler::new();
    let plan_ref = compiler_ref.compile(src).expect("compile ref");
    let fallback_plan = replace_multiway_with_fallback(plan_ref);
    let mut executor_ref =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rid) in compiler_ref.rel_ids().clone() {
        executor_ref.register_relation(rid, &name);
    }
    for (name, rows) in &inputs {
        executor_ref.put_relation(name, upload_binary_u32(&fix.memory, rows));
    }
    let _ = executor_ref
        .execute_plan(&fallback_plan)
        .expect("execute fallback");
    let fallback_rows =
        download_k_row_set(executor_ref.store().get(head_name).expect("head ref"), k);

    // 3. Row-set parity: dispatch output == fallback output.
    assert_eq!(
        dispatch_rows, fallback_rows,
        "K={} dispatch row set must equal MultiWayJoin.fallback reference",
        k
    );
}

#[test]
fn clique5_dispatch_counter_advances_and_row_set_matches_fallback_body() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    run_counter_advance_test(&fix, CLIQUE5_SRC, "clique5", 5, |e| {
        e.wcoj_clique5_dispatch_count()
    });
}

#[test]
fn clique6_dispatch_counter_advances_and_row_set_matches_fallback_body() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    run_counter_advance_test(&fix, CLIQUE6_SRC, "clique6", 6, |e| {
        e.wcoj_clique6_dispatch_count()
    });
}

/// Run a clique-K dispatcher-decline test: one edge is NOT put
/// into the executor's store so the dispatcher's
/// `self.rel_names.get(rid)` → `store.get(name)` returns None →
/// dispatcher returns Ok(None) → counter stays at 0 → executor
/// falls through to `MultiWayJoin.fallback`. The fallback body
/// (the original binary-join tree) ALSO cannot find the missing
/// edge's buffer and executes with an empty Scan for that rel →
/// produces an empty result (no cliques when an edge is missing).
///
/// Reference: `replace_multiway_with_fallback` on the SAME
/// missing-input fixture produces the same empty result, proving
/// row-set parity between the decline path and the fallback
/// reference under identical input conditions.
///
/// Asserts: (a) counter == 0, (b) row set from decline path ==
/// row set from fallback-only plan on the same missing-input
/// fixture.
fn run_dispatcher_decline_test(
    fix: &RuntimeBackedFixture,
    src: &str,
    head_name: &str,
    k: usize,
    missing_edge_name: &str,
    check_counter: fn(&Executor) -> u64,
) {
    let inputs = k_clique_inputs(k);

    // 1. Decline executor: all edges except one are present.
    let mut compiler = Compiler::new();
    let plan = compiler.compile(src).expect("compile");
    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rid) in compiler.rel_ids().clone() {
        executor.register_relation(rid, &name);
    }
    for (name, rows) in &inputs {
        if name == missing_edge_name {
            continue;
        }
        executor.put_relation(name, upload_binary_u32(&fix.memory, rows));
    }
    let _ = executor.execute_plan(&plan);
    assert_eq!(
        check_counter(&executor),
        0,
        "dispatcher decline must NOT advance the K={} counter; got {}",
        k,
        check_counter(&executor)
    );

    // 2. Fallback reference: same missing-input fixture, plan
    //    rewritten to bypass MultiWayJoin → pure binary-join.
    let mut compiler_ref = Compiler::new();
    let plan_ref = compiler_ref.compile(src).expect("compile ref");
    let fallback_plan = replace_multiway_with_fallback(plan_ref);
    let mut executor_ref =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rid) in compiler_ref.rel_ids().clone() {
        executor_ref.register_relation(rid, &name);
    }
    for (name, rows) in &inputs {
        if name == missing_edge_name {
            continue;
        }
        executor_ref.put_relation(name, upload_binary_u32(&fix.memory, rows));
    }
    let _ = executor_ref.execute_plan(&fallback_plan);

    // 3. Row-set parity: both paths see the same missing input
    //    and produce the same (empty) result. This proves the
    //    decline path's fallback behavior matches the binary-join
    //    reference under identical conditions.
    let decline_rows = executor
        .store()
        .get(head_name)
        .map(|b| download_k_row_set(b, k))
        .unwrap_or_default();
    let fallback_rows = executor_ref
        .store()
        .get(head_name)
        .map(|b| download_k_row_set(b, k))
        .unwrap_or_default();
    assert_eq!(
        decline_rows, fallback_rows,
        "K={} dispatcher-decline row set must equal fallback reference \
         (both should be empty since edge {} is missing)",
        k, missing_edge_name
    );
}

#[test]
fn clique5_dispatcher_decline_does_not_advance_counter_and_row_set_matches_fallback() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    run_dispatcher_decline_test(&fix, CLIQUE5_SRC, "clique5", 5, "e34", |e| {
        e.wcoj_clique5_dispatch_count()
    });
}

#[test]
fn clique6_dispatcher_decline_does_not_advance_counter_and_row_set_matches_fallback() {
    let Some(fix) = make_runtime_backed_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    run_dispatcher_decline_test(&fix, CLIQUE6_SRC, "clique6", 6, "e45", |e| {
        e.wcoj_clique6_dispatch_count()
    });
}
