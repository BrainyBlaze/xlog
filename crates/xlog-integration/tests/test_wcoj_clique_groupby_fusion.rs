//! Aggregate-fused K-clique count: end-to-end executor wiring.
//!
//! A count-aggregate head over a complete K-clique body (K = 5, 6) must
//! compile (promoter descends the aggregate wrapper), dispatch the fused
//! group-by-root kernel (counter == 1) when the group key is the plan's
//! root variable, and produce the same rows as the fallback+groupby path
//! (kill switch forces the unfused path; rows must be identical). The
//! root under `KCliqueVariableOrder` is plan-dependent: a hot-variable
//! stats snapshot moves the root away from V0, and the fusion must follow
//! the plan (fires on the planned root, declines on any other key).
//! Fused/kill-switch phases run inside ONE test because the kill switch
//! is a process-global env var.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, MutexGuard};

use cudarc::driver::sys;
use xlog_core::{MemoryBudget, RelId, RuntimeConfig, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::{CudaBuffer, CudaColumn};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_ir::{ExecutionPlan, RirNode};
use xlog_logic::Compiler;
use xlog_stats::{
    ColumnStats, JoinSelectivity, KeyHeatStats, PrefixDegreeStats, RelationStats, StatsSnapshot,
};

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

fn make_fixture() -> Option<Fixture> {
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
        Box::new(GlobalDeviceBudget::new(logging, 256 * 1024 * 1024));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(256 * 1024 * 1024),
        runtime,
    ));
    let provider =
        Arc::new(CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory)).ok()?);
    Some(Fixture { memory, provider })
}

fn upload_binary_u32(memory: &Arc<GpuMemoryManager>, rows: &[(u32, u32)]) -> CudaBuffer {
    let n = rows.len() as u32;
    let bpc = rows.len().max(1) * std::mem::size_of::<u32>();
    let mut col0 = memory.alloc::<u8>(bpc).expect("alloc col0");
    let mut col1 = memory.alloc::<u8>(bpc).expect("alloc col1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let device = memory.device().inner();
    if n > 0 {
        let col0_bytes: Vec<u8> = rows.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
        let col1_bytes: Vec<u8> = rows.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
        device
            .htod_sync_copy_into(&col0_bytes, &mut col0)
            .expect("htod col0");
        device
            .htod_sync_copy_into(&col1_bytes, &mut col1)
            .expect("htod col1");
    }
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

fn download_column_bytes(
    memory: &Arc<GpuMemoryManager>,
    buffer: &CudaBuffer,
    col: usize,
    elem_size: usize,
) -> Vec<u8> {
    let n = buffer_rows(memory, buffer);
    let mut bytes = vec![0u8; n * elem_size];
    if n == 0 {
        return bytes;
    }
    let CudaColumn::Owned(c) = buffer.column(col).expect("column") else {
        panic!("column must be owned");
    };
    unsafe {
        let res = sys::cuMemcpyDtoH_v2(bytes.as_mut_ptr() as *mut _, *c.device_ptr(), bytes.len());
        assert_eq!(res, sys::cudaError_enum::CUDA_SUCCESS, "dtoh column copy");
    }
    bytes
}

fn download_group_counts(memory: &Arc<GpuMemoryManager>, buffer: &CudaBuffer) -> Vec<(u32, u64)> {
    assert_eq!(buffer.arity(), 2, "expected (root, count) output");
    let keys: Vec<u32> = download_column_bytes(memory, buffer, 0, 4)
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let counts: Vec<u64> = download_column_bytes(memory, buffer, 1, 8)
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let mut out: Vec<(u32, u64)> = keys.into_iter().zip(counts).collect();
    out.sort();
    out
}

/// The kill switch is a process-global env var: every test that toggles it
/// (or asserts the fused counter fired) takes this lock so a concurrent
/// kill-switch phase cannot leak into another test's fused phase.
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn env_lock() -> MutexGuard<'static, ()> {
    ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn canonical_edge_list(k: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for i in 0..k {
        for j in (i + 1)..k {
            pairs.push((i, j));
        }
    }
    pairs
}

/// K-clique edge fixture keyed by the canonical (i, j) pair.
type EdgeMap = BTreeMap<(usize, usize), Vec<(u32, u32)>>;

fn sorted_unique(rows: impl IntoIterator<Item = (u32, u32)>) -> Vec<(u32, u32)> {
    let set: BTreeSet<(u32, u32)> = rows.into_iter().collect();
    set.into_iter().collect()
}

/// Banded K-clique fixture: variable position v draws from band v.
/// `closing_roots` (variable 0 values) participate in the full cross
/// product over the bands; `dangling_roots` get only an e01 edge and must
/// never appear in any output.
fn banded_fixture(
    k: usize,
    closing_roots: &[u32],
    dangling_roots: &[u32],
    band_width: u32,
) -> EdgeMap {
    let band = |v: usize| -> Vec<u32> {
        (0..band_width)
            .map(|i| (v as u32) * 1_000_000 + i)
            .collect()
    };
    let mut edges: EdgeMap = BTreeMap::new();
    for (i, j) in canonical_edge_list(k) {
        let mut rows = Vec::new();
        if i == 0 {
            for &r in closing_roots {
                for &b in &band(j) {
                    rows.push((r, b));
                }
            }
            if j == 1 {
                for &r in dangling_roots {
                    rows.push((r, band(1)[0]));
                }
            }
        } else {
            for &a in &band(i) {
                for &b in &band(j) {
                    rows.push((a, b));
                }
            }
        }
        edges.insert((i, j), sorted_unique(rows));
    }
    edges
}

/// Host brute-force oracle: per-`group_var` count of complete K-clique
/// completions over the canonical edge adjacency.
fn oracle_group_counts(k: usize, edges: &EdgeMap, group_var: usize) -> Vec<(u32, u64)> {
    type Adj = BTreeMap<(usize, usize), BTreeMap<u32, BTreeSet<u32>>>;
    let mut adj: Adj = BTreeMap::new();
    for (&(i, j), rows) in edges {
        let by_left = adj.entry((i, j)).or_default();
        for &(a, b) in rows {
            by_left.entry(a).or_default().insert(b);
        }
    }

    fn extend(
        k: usize,
        level: usize,
        binding: &mut Vec<u32>,
        adj: &Adj,
        group_var: usize,
        counts: &mut BTreeMap<u32, u64>,
    ) {
        if level == k {
            *counts.entry(binding[group_var]).or_default() += 1;
            return;
        }
        let mut candidates: Option<BTreeSet<u32>> = None;
        for prior in 0..level {
            let allowed = adj
                .get(&(prior, level))
                .and_then(|m| m.get(&binding[prior]))
                .cloned()
                .unwrap_or_default();
            let next = match candidates {
                None => allowed,
                Some(current) => current.intersection(&allowed).copied().collect(),
            };
            if next.is_empty() {
                return;
            }
            candidates = Some(next);
        }
        for cand in candidates.unwrap_or_default() {
            binding.push(cand);
            extend(k, level + 1, binding, adj, group_var, counts);
            binding.pop();
        }
    }

    let mut counts: BTreeMap<u32, u64> = BTreeMap::new();
    if let Some(e01) = adj.get(&(0, 1)) {
        for (&v0, v1s) in e01 {
            for &v1 in v1s {
                let mut binding = vec![v0, v1];
                extend(k, 2, &mut binding, &adj, group_var, &mut counts);
            }
        }
    }
    counts.into_iter().collect()
}

/// Complete planner stats for the canonical edge relations `e{i}{j}`.
/// `cool` optionally lowers one variable's key heat so the greedy planner
/// chooses it as the variable-order root (lowest score wins the root).
fn clique_stats(k: u8, cool: Option<(u8, f64)>) -> StatsSnapshot {
    let mut snapshot = StatsSnapshot::default();
    let mut edges = Vec::new();
    let mut rel_id = 1u32;

    for i in 0..k {
        for j in (i + 1)..k {
            let rel = RelId(rel_id);
            rel_id += 1;
            snapshot.rel_names.push((rel, format!("e{i}{j}")));
            edges.push((rel, i, j));

            let mut stats = RelationStats::new(rel);
            stats.update_cardinality(2_000 + u64::from(k));
            for (col_idx, variable) in [(0usize, i), (1usize, j)] {
                let mut col = ColumnStats::new(col_idx, ScalarType::U32);
                col.update_distinct(1_000 + u64::from(k));
                stats.add_column(col);
                stats.add_prefix_degree(PrefixDegreeStats::new(col_idx, 2.0, 2.5));
                let heat = match cool {
                    Some((cool_variable, cool_heat)) if cool_variable == variable => cool_heat,
                    _ => 0.75,
                };
                stats.add_key_heat(KeyHeatStats::new(col_idx, heat, heat));
            }
            snapshot.relations.push(stats);
        }
    }

    for (left_idx, (left_rel, left_i, left_j)) in edges.iter().enumerate() {
        for (right_rel, right_i, right_j) in edges.iter().skip(left_idx + 1) {
            if left_i == right_i || left_i == right_j || left_j == right_i || left_j == right_j {
                let mut sel = JoinSelectivity::new(*left_rel, *right_rel);
                sel.set_keys(vec![0], vec![0]);
                sel.set_selectivity(0.001);
                snapshot.join_selectivities.push(sel);
            }
        }
    }

    snapshot
}

/// True iff some GroupBy input subtree contains a MultiWayJoin with
/// `inputs.len() == arity` (the promoter's inside-aggregate descent
/// fired).
fn plan_promotes_clique_under_groupby(plan: &ExecutionPlan, arity: usize) -> bool {
    fn has_multiway(node: &RirNode, arity: usize) -> bool {
        match node {
            RirNode::MultiWayJoin { inputs, .. } => inputs.len() == arity,
            RirNode::Project { input, .. } => has_multiway(input, arity),
            _ => false,
        }
    }
    fn walk(node: &RirNode, arity: usize) -> bool {
        match node {
            RirNode::GroupBy { input, .. } => has_multiway(input, arity) || walk(input, arity),
            RirNode::Project { input, .. } | RirNode::Distinct { input, .. } => walk(input, arity),
            RirNode::Filter { input, .. } => walk(input, arity),
            RirNode::Join { left, right, .. } | RirNode::Diff { left, right } => {
                walk(left, arity) || walk(right, arity)
            }
            RirNode::Union { inputs } => inputs.iter().any(|n| walk(n, arity)),
            _ => false,
        }
    }
    plan.rules_by_scc
        .iter()
        .any(|rules| rules.iter().any(|rule| walk(&rule.body, arity)))
}

/// XLOG aggregate source over the complete K_5 body, grouped by `group`.
/// Edge predicates are declared so `compile_with_stats_snapshot` can remap
/// the snapshot's column stats onto the lowered relations (undeclared
/// predicates get their column stats cleared, which downgrades the planner
/// to `IncompleteStatsSafeDefault` → no `KCliqueVariableOrder`).
fn clique5_agg_src(group: &str) -> String {
    format!(
        "pred e01(u32, u32). pred e02(u32, u32). pred e03(u32, u32). pred e04(u32, u32). \
         pred e12(u32, u32). pred e13(u32, u32). pred e14(u32, u32). \
         pred e23(u32, u32). pred e24(u32, u32). \
         pred e34(u32, u32). \
         agg({group}, count(V4)) :- \
         e01(V0, V1), e02(V0, V2), e03(V0, V3), e04(V0, V4), \
         e12(V1, V2), e13(V1, V3), e14(V1, V4), \
         e23(V2, V3), e24(V2, V4), \
         e34(V3, V4)."
    )
}

fn clique6_agg_src(group: &str) -> String {
    format!(
        "pred e01(u32, u32). pred e02(u32, u32). pred e03(u32, u32). \
         pred e04(u32, u32). pred e05(u32, u32). \
         pred e12(u32, u32). pred e13(u32, u32). pred e14(u32, u32). pred e15(u32, u32). \
         pred e23(u32, u32). pred e24(u32, u32). pred e25(u32, u32). \
         pred e34(u32, u32). pred e35(u32, u32). \
         pred e45(u32, u32). \
         agg({group}, count(V5)) :- \
         e01(V0, V1), e02(V0, V2), e03(V0, V3), e04(V0, V4), e05(V0, V5), \
         e12(V1, V2), e13(V1, V3), e14(V1, V4), e15(V1, V5), \
         e23(V2, V3), e24(V2, V4), e25(V2, V5), \
         e34(V3, V4), e35(V3, V5), \
         e45(V4, V5)."
    )
}

/// Compile + run one aggregate program; returns (rows, fusion counter,
/// promoted-under-groupby flag).
fn run_agg_program(
    fix: &Fixture,
    src: &str,
    k: usize,
    edges: &EdgeMap,
    snapshot: &StatsSnapshot,
) -> (Vec<(u32, u64)>, u64, bool) {
    let mut compiler = Compiler::new();
    let plan = compiler
        .compile_with_stats_snapshot(src, Some(snapshot))
        .expect("compile aggregate clique rule");
    let promoted = plan_promotes_clique_under_groupby(&plan, k * (k - 1) / 2);
    let mut executor = xlog_runtime::Executor::new_with_config(
        Arc::clone(&fix.provider),
        RuntimeConfig::default(),
    );
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    for (i, j) in canonical_edge_list(k) {
        let rows = edges.get(&(i, j)).cloned().unwrap_or_default();
        executor.put_relation(&format!("e{i}{j}"), upload_binary_u32(&fix.memory, &rows));
    }
    executor.execute_plan(&plan).expect("execute plan");
    let agg = executor.store().get("agg").expect("agg relation");
    let rows = download_group_counts(&fix.memory, agg);
    (
        rows,
        executor.wcoj_groupby_fusion_dispatch_count(),
        promoted,
    )
}

/// Fused phase (counter == 1, oracle parity) then kill-switch phase
/// (counter == 0, identical rows).
fn assert_clique_fusion_parity(
    fix: &Fixture,
    src: &str,
    k: usize,
    edges: &EdgeMap,
    snapshot: &StatsSnapshot,
    expected: &[(u32, u64)],
) {
    let (fused, fused_count, promoted) = run_agg_program(fix, src, k, edges, snapshot);
    assert!(
        promoted,
        "promoter must descend the aggregate wrapper: {src}"
    );
    assert_eq!(fused, expected, "fused path row set: {src}");
    assert_eq!(fused_count, 1, "fused dispatch must fire once: {src}");

    // SAFETY: single-threaded phase of this test; restored below.
    unsafe {
        std::env::set_var("XLOG_DISABLE_WCOJ_GROUPBY_FUSION", "1");
    }
    let (unfused, unfused_count, _) = run_agg_program(fix, src, k, edges, snapshot);
    unsafe {
        std::env::remove_var("XLOG_DISABLE_WCOJ_GROUPBY_FUSION");
    }
    assert_eq!(unfused, expected, "kill-switch path row set: {src}");
    assert_eq!(unfused_count, 0, "kill switch must keep the counter at 0");
}

#[test]
fn clique5_count_fusion_fires_end_to_end_with_parity() {
    let Some(fix) = make_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let _guard = env_lock();
    let edges = banded_fixture(5, &[1, 2], &[9], 2);
    // Uniform stats: the greedy planner roots at V0 (score tie broken by
    // vertex id), so grouping by V0 is the fusable key.
    let snapshot = clique_stats(5, None);
    let expected = oracle_group_counts(5, &edges, 0);
    assert!(!expected.is_empty());
    assert_clique_fusion_parity(
        &fix,
        &clique5_agg_src("V0"),
        5,
        &edges,
        &snapshot,
        &expected,
    );
}

#[test]
fn clique6_count_fusion_fires_end_to_end_with_parity() {
    let Some(fix) = make_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let _guard = env_lock();
    let edges = banded_fixture(6, &[1, 2], &[9], 2);
    let snapshot = clique_stats(6, None);
    let expected = oracle_group_counts(6, &edges, 0);
    assert!(!expected.is_empty());
    assert_clique_fusion_parity(
        &fix,
        &clique6_agg_src("V0"),
        6,
        &edges,
        &snapshot,
        &expected,
    );
}

#[test]
fn clique5_count_fusion_follows_plan_dependent_root() {
    let Some(fix) = make_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let _guard = env_lock();
    let edges = banded_fixture(5, &[1, 2], &[9], 2);
    // Cooling V3's key heat gives it the lowest root score: the planner
    // roots the variable order at V3 (leader edge e03 column-swapped so
    // its col0 carries V3 values).
    let snapshot = clique_stats(5, Some((3, 0.05)));

    // Cell A: grouping by the planned root V3 fuses with parity.
    let expected_v3 = oracle_group_counts(5, &edges, 3);
    assert!(!expected_v3.is_empty());
    assert_clique_fusion_parity(
        &fix,
        &clique5_agg_src("V3"),
        5,
        &edges,
        &snapshot,
        &expected_v3,
    );

    // Cell B: grouping by V0 (NOT the planned root) must decline
    // silently — counter stays 0, rows still correct via the embedded
    // fallback + groupby path.
    let expected_v0 = oracle_group_counts(5, &edges, 0);
    let (rows, count, promoted) =
        run_agg_program(&fix, &clique5_agg_src("V0"), 5, &edges, &snapshot);
    assert!(promoted, "promoter must still promote the clique body");
    assert_eq!(rows, expected_v0, "non-root key declines with parity");
    assert_eq!(count, 0, "non-root key must not advance the fusion counter");
}

#[test]
fn non_clique_aggregate_body_declines_silently() {
    let Some(fix) = make_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let _guard = env_lock();
    // 9-atom body (e34 dropped): not a complete K_5, so the CLIQUE
    // promoter must leave it alone. The general multiway promoter
    // picks such bodies up instead, and the
    // executor fuses them through the factorized Free Join
    // count-by-root route — the fusion counter advances via the
    // FREE JOIN dispatch counter, never the clique path.
    let src = "agg(V0, count(V4)) :- \
               e01(V0, V1), e02(V0, V2), e03(V0, V3), e04(V0, V4), \
               e12(V1, V2), e13(V1, V3), e14(V1, V4), \
               e23(V2, V3), e24(V2, V4).";
    let edges = banded_fixture(5, &[1, 2], &[9], 2);
    let snapshot = clique_stats(5, None);

    let mut compiler = Compiler::new();
    let plan = compiler
        .compile_with_stats_snapshot(src, Some(&snapshot))
        .expect("compile 9-atom aggregate rule");
    assert!(
        !plan_promotes_clique_under_groupby(&plan, 10),
        "incomplete clique must not promote"
    );
    let mut executor = xlog_runtime::Executor::new_with_config(
        Arc::clone(&fix.provider),
        RuntimeConfig::default(),
    );
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    for (i, j) in canonical_edge_list(5) {
        if (i, j) == (3, 4) {
            continue;
        }
        let rows = edges.get(&(i, j)).cloned().unwrap_or_default();
        executor.put_relation(&format!("e{i}{j}"), upload_binary_u32(&fix.memory, &rows));
    }
    executor.execute_plan(&plan).expect("execute plan");
    // The body fuses through the generic Free Join count route
    // (one fused dispatch, attributed to the Free Join counter) —
    // NOT through any clique kernel (the clique promotion assert
    // above pins that).
    assert_eq!(
        executor.wcoj_groupby_fusion_dispatch_count(),
        1,
        "non-clique body fuses via Free Join count"
    );
    assert_eq!(
        executor.free_join_dispatch_count(),
        1,
        "the fused dispatch must be the Free Join route"
    );
    let agg = executor.store().get("agg").expect("agg relation");
    let rows = download_group_counts(&fix.memory, agg);
    assert!(!rows.is_empty(), "9-atom body still produces counts");

    // Behavioral parity: the unfused (kill-switch) path must produce
    // the identical group counts.
    // SAFETY: single-threaded phase of this test; restored below.
    unsafe {
        std::env::set_var("XLOG_DISABLE_FREE_JOIN", "1");
    }
    let mut unfused_executor = xlog_runtime::Executor::new_with_config(
        Arc::clone(&fix.provider),
        RuntimeConfig::default(),
    );
    for (name, rel_id) in compiler.rel_ids() {
        unfused_executor.register_relation(*rel_id, name);
    }
    for (i, j) in canonical_edge_list(5) {
        if (i, j) == (3, 4) {
            continue;
        }
        let edge_rows = edges.get(&(i, j)).cloned().unwrap_or_default();
        unfused_executor.put_relation(
            &format!("e{i}{j}"),
            upload_binary_u32(&fix.memory, &edge_rows),
        );
    }
    let unfused_result = unfused_executor.execute_plan(&plan);
    unsafe {
        std::env::remove_var("XLOG_DISABLE_FREE_JOIN");
    }
    unfused_result.expect("execute plan (kill switch)");
    assert_eq!(
        unfused_executor.free_join_dispatch_count(),
        0,
        "kill switch must keep the free join counter at 0"
    );
    let unfused_agg = unfused_executor.store().get("agg").expect("agg relation");
    let unfused_rows = download_group_counts(&fix.memory, unfused_agg);
    assert_eq!(rows, unfused_rows, "fused vs unfused group-count parity");
}
