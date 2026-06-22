//! Aggregate-fused WCOJ: group-by-root count over the K-clique shape
//! (K = 5, 6; u32 width-class).
//!
//! Contract under test:
//! `wcoj_clique{5,6}_groupby_root_count_u32_recorded_planned` computes,
//! for `q(R, count(*)) :- <complete K-clique body>` grouped by the plan's
//! position-0 root variable, the same (root, count) row set as the unfused
//! production path (materialize cliques via the planned clique entry, then
//! groupby count) — WITHOUT materializing the clique rows. Both paths are
//! checked against a host brute-force oracle.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use cudarc::driver::sys;

use xlog_core::{AggOp, MemoryBudget, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::{CudaBuffer, CudaColumn};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

struct DiscardSink;
impl LoggingSink for DiscardSink {
    fn emit(&self, _record: LogRecord) -> Result<(), SinkError> {
        Ok(())
    }
}

struct Fixture {
    memory: Arc<GpuMemoryManager>,
    provider: Arc<CudaKernelProvider>,
    pool: Arc<StreamPool>,
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
        Box::new(GlobalDeviceBudget::new(logging, 512 * 1024 * 1024));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(512 * 1024 * 1024),
        runtime,
    ));
    let provider =
        Arc::new(CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory)).ok()?);
    Some(Fixture {
        memory,
        provider,
        pool,
    })
}

fn upload_binary_u32(memory: &Arc<GpuMemoryManager>, rows: &[(u32, u32)]) -> CudaBuffer {
    let n = rows.len() as u32;
    let bpc = (rows.len()).max(1) * std::mem::size_of::<u32>();
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

/// Raw partial-column download: groupby outputs allocate columns at
/// capacity, so copies must use the logical byte length, not the
/// allocation length.
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

fn download_u32_column(
    memory: &Arc<GpuMemoryManager>,
    buffer: &CudaBuffer,
    col: usize,
) -> Vec<u32> {
    download_column_bytes(memory, buffer, col, 4)
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn download_u64_column(
    memory: &Arc<GpuMemoryManager>,
    buffer: &CudaBuffer,
    col: usize,
) -> Vec<u64> {
    download_column_bytes(memory, buffer, col, 8)
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Sorted (root, count) pairs from a 2-column (U32 key, U64 count) buffer.
fn download_group_counts(memory: &Arc<GpuMemoryManager>, buffer: &CudaBuffer) -> Vec<(u32, u64)> {
    let keys = download_u32_column(memory, buffer, 0);
    let counts = download_u64_column(memory, buffer, 1);
    assert_eq!(keys.len(), counts.len());
    let mut out: Vec<(u32, u64)> = keys.into_iter().zip(counts).collect();
    out.sort();
    out
}

/// K-clique edge fixture keyed by the canonical (i, j) pair, i < j < k.
type EdgeMap = BTreeMap<(usize, usize), Vec<(u32, u32)>>;

fn canonical_edge_list(k: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for i in 0..k {
        for j in (i + 1)..k {
            pairs.push((i, j));
        }
    }
    pairs
}

/// Host brute-force oracle: per-root (variable 0) count of complete
/// K-clique completions. Generic recursive extension over the canonical
/// edge adjacency, identical in spirit to the device traversal but built
/// from independent host-side set logic.
fn oracle_group_counts(k: usize, edges: &EdgeMap) -> Vec<(u32, u64)> {
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
        counts: &mut BTreeMap<u32, u64>,
    ) {
        if level == k {
            *counts.entry(binding[0]).or_default() += 1;
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
            extend(k, level + 1, binding, adj, counts);
            binding.pop();
        }
    }

    let mut counts: BTreeMap<u32, u64> = BTreeMap::new();
    if let Some(e01) = adj.get(&(0, 1)) {
        for (&v0, v1s) in e01 {
            for &v1 in v1s {
                let mut binding = vec![v0, v1];
                extend(k, 2, &mut binding, &adj, &mut counts);
            }
        }
    }
    counts.into_iter().collect()
}

fn sorted_unique(rows: impl IntoIterator<Item = (u32, u32)>) -> Vec<(u32, u32)> {
    let set: BTreeSet<(u32, u32)> = rows.into_iter().collect();
    set.into_iter().collect()
}

fn upload_edges(fix: &Fixture, k: usize, edges: &EdgeMap) -> Vec<CudaBuffer> {
    canonical_edge_list(k)
        .iter()
        .map(|pair| {
            upload_binary_u32(
                &fix.memory,
                edges.get(pair).map(Vec::as_slice).unwrap_or(&[]),
            )
        })
        .collect()
}

fn identity_orders(k: usize) -> (Vec<u8>, Vec<u8>) {
    let expected_edges = k * (k - 1) / 2;
    ((0..expected_edges as u8).collect(), (0..k as u8).collect())
}

/// Unfused production baseline: materialize cliques via the planned
/// clique entry (identity plan), then groupby count on the root column.
fn baseline_group_counts(
    fix: &Fixture,
    k: usize,
    bufs: &[CudaBuffer],
    stream: xlog_cuda::device_runtime::StreamId,
) -> Vec<(u32, u64)> {
    let refs: Vec<&CudaBuffer> = bufs.iter().collect();
    let (edge_order, iteration_order) = identity_orders(k);
    let cliques = match k {
        5 => {
            let arr: &[&CudaBuffer; 10] = refs.as_slice().try_into().expect("10 edges");
            fix.provider
                .wcoj_clique5_u32_recorded_planned(arr, 0, &edge_order, &iteration_order, stream)
                .expect("baseline clique5 materialize")
        }
        6 => {
            let arr: &[&CudaBuffer; 15] = refs.as_slice().try_into().expect("15 edges");
            fix.provider
                .wcoj_clique6_u32_recorded_planned(arr, 0, &edge_order, &iteration_order, stream)
                .expect("baseline clique6 materialize")
        }
        _ => unreachable!("baseline supports k = 5, 6"),
    };
    let grouped = fix
        .provider
        .groupby_multi_agg(&cliques, &[0], &[(1, AggOp::Count)])
        .expect("baseline groupby count");
    download_group_counts(&fix.memory, &grouped)
}

fn fused_group_counts(
    fix: &Fixture,
    k: usize,
    bufs: &[CudaBuffer],
    stream: xlog_cuda::device_runtime::StreamId,
) -> Vec<(u32, u64)> {
    let refs: Vec<&CudaBuffer> = bufs.iter().collect();
    let (edge_order, iteration_order) = identity_orders(k);
    let fused = match k {
        5 => {
            let arr: &[&CudaBuffer; 10] = refs.as_slice().try_into().expect("10 edges");
            fix.provider
                .wcoj_clique5_groupby_root_count_u32_recorded_planned(
                    arr,
                    0,
                    &edge_order,
                    &iteration_order,
                    stream,
                )
                .expect("fused clique5 groupby-root count")
        }
        6 => {
            let arr: &[&CudaBuffer; 15] = refs.as_slice().try_into().expect("15 edges");
            fix.provider
                .wcoj_clique6_groupby_root_count_u32_recorded_planned(
                    arr,
                    0,
                    &edge_order,
                    &iteration_order,
                    stream,
                )
                .expect("fused clique6 groupby-root count")
        }
        _ => unreachable!("fused supports k = 5, 6"),
    };
    download_group_counts(&fix.memory, &fused)
}

fn run_case(name: &str, k: usize, edges: &EdgeMap) {
    let Some(fix) = make_fixture() else {
        eprintln!("skipping {name}: no CUDA device");
        return;
    };
    let expected = oracle_group_counts(k, edges);
    assert!(
        !expected.is_empty(),
        "{name}: fixture must contain at least one {k}-clique"
    );

    let bufs = upload_edges(&fix, k, edges);
    let stream = fix.pool.acquire().expect("stream");
    let baseline = baseline_group_counts(&fix, k, &bufs, stream);
    assert_eq!(baseline, expected, "{name}: unfused baseline vs oracle");

    let fused = fused_group_counts(&fix, k, &bufs, stream);
    assert_eq!(fused, expected, "{name}: fused vs oracle");
}

/// Banded K-clique fixture: variable position v draws from band v.
/// `roots` are the position-0 values; each root in `closing_roots`
/// participates in the full cross product over the bands; other roots get
/// only e01 edges (must NOT appear in the output).
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

#[test]
fn clique5_groupby_root_count_matches_oracle_small() {
    // Roots 1 and 2 close band_width^4 cliques each; root 9 has only an
    // e01 edge and must be absent from the output.
    let edges = banded_fixture(5, &[1, 2], &[9], 2);
    run_case("clique5_small", 5, &edges);
}

#[test]
fn clique6_groupby_root_count_matches_oracle_small() {
    let edges = banded_fixture(6, &[1, 2], &[9], 2);
    run_case("clique6_small", 6, &edges);
}

#[test]
fn clique5_groupby_root_count_matches_oracle_skewed_hub() {
    // Hub root 0 with a wide V1 fanout; bands of 4 for V2..V4. Heavy
    // per-root fanout: many leader rows feed the same root counter.
    let band =
        |v: usize, w: u32| -> Vec<u32> { (0..w).map(|i| (v as u32) * 1_000_000 + i).collect() };
    let mut edges: EdgeMap = BTreeMap::new();
    for (i, j) in canonical_edge_list(5) {
        let left: Vec<u32> = if i == 0 {
            vec![0]
        } else if i == 1 {
            band(1, 64)
        } else {
            band(i, 4)
        };
        let right: Vec<u32> = if j == 1 { band(1, 64) } else { band(j, 4) };
        let mut rows = Vec::new();
        for &a in &left {
            for &b in &right {
                rows.push((a, b));
            }
        }
        edges.insert((i, j), sorted_unique(rows));
    }
    // Uniform background away from the hub.
    for (idx, (i, j)) in canonical_edge_list(5).into_iter().enumerate() {
        let rows = edges.get_mut(&(i, j)).expect("edge");
        for t in 0..50u32 {
            rows.push((10_000_000 + (idx as u32) * 100_000 + t, 90_000_000 + t));
        }
        *rows = sorted_unique(rows.iter().copied());
    }
    run_case("clique5_skewed_hub", 5, &edges);
}

#[test]
fn clique5_groupby_root_count_layout_normalizes_unsorted_input() {
    // Same fixture as the small case, but every edge is uploaded
    // REVERSE-sorted with duplicated rows. The fused entry must
    // layout-normalize per dispatch (31b0ccf0 contract) and produce the
    // oracle row set anyway.
    let edges = banded_fixture(5, &[1, 2], &[9], 2);
    let Some(fix) = make_fixture() else {
        eprintln!("skipping clique5_layout_normalize: no CUDA device");
        return;
    };
    let expected = oracle_group_counts(5, &edges);
    assert!(!expected.is_empty());

    let mangled: Vec<CudaBuffer> = canonical_edge_list(5)
        .iter()
        .map(|pair| {
            let mut rows = edges.get(pair).cloned().unwrap_or_default();
            rows.sort_by(|a, b| b.cmp(a));
            let dup: Vec<(u32, u32)> = rows.iter().copied().chain(rows.iter().copied()).collect();
            upload_binary_u32(&fix.memory, &dup)
        })
        .collect();
    let stream = fix.pool.acquire().expect("stream");
    let fused = fused_group_counts(&fix, 5, &mangled, stream);
    assert_eq!(fused, expected, "fused must normalize unsorted+dup input");
}

/// Aggregate-fused WCOJ K-clique count measurement (gate: fused >= 3x vs unfused on the skewed K=5 hub
/// fixture). Run explicitly:
/// `cargo test -p xlog-cuda-tests --test test_wcoj_clique_groupby_root_count \
///    --release -- --ignored --nocapture`
/// Asserts parity; timing ratios are PRINTED and recorded as evidence, not
/// asserted (wall-clock assertions are machine-dependent).
#[test]
#[ignore = "aggregate-fused WCOJ K-clique count measurement: run explicitly with --ignored --nocapture"]
fn wcoj_clique_groupby_root_count_measurement_fused_vs_unfused() {
    let Some(fix) = make_fixture() else {
        eprintln!("skipping aggregate-fused WCOJ K-clique count measurement: no CUDA device");
        return;
    };

    // Hub root 0: e01 fans to n_x V1 values; V2..V4 draw from 16-wide
    // bands with all band pairs present. Completions per leader row:
    // 16^3 = 4096 — the materialized clique row count is n_x * 4096.
    let hub = |n_x: u32| -> EdgeMap {
        let band =
            |v: usize, w: u32| -> Vec<u32> { (0..w).map(|i| (v as u32) * 1_000_000 + i).collect() };
        let mut edges: EdgeMap = BTreeMap::new();
        for (i, j) in canonical_edge_list(5) {
            let left: Vec<u32> = if i == 0 {
                vec![0]
            } else if i == 1 {
                band(1, n_x)
            } else {
                band(i, 16)
            };
            let right: Vec<u32> = if j == 1 { band(1, n_x) } else { band(j, 16) };
            let mut rows = Vec::new();
            for &a in &left {
                for &b in &right {
                    rows.push((a, b));
                }
            }
            edges.insert((i, j), sorted_unique(rows));
        }
        // Uniform background so the group column is not a single value.
        for (idx, (i, j)) in canonical_edge_list(5).into_iter().enumerate() {
            let rows = edges.get_mut(&(i, j)).expect("edge");
            for t in 0..1000u32 {
                rows.push((10_000_000 + (idx as u32) * 100_000 + t, 90_000_000 + t));
            }
            *rows = sorted_unique(rows.iter().copied());
        }
        edges
    };

    let cases: Vec<(&str, EdgeMap)> = vec![
        ("clique5_hub_500", hub(500)),
        ("clique5_hub_1000", hub(1000)),
    ];

    const REPS: usize = 5;
    for (name, edges) in &cases {
        let expected = oracle_group_counts(5, edges);
        let bufs = upload_edges(&fix, 5, edges);

        // One stream per case, reused across reps (grow-only StreamPool).
        let stream = fix.pool.acquire().expect("stream");
        // Warmup both paths once (kernel/module JIT, stream init).
        let warm_baseline = baseline_group_counts(&fix, 5, &bufs, stream);
        assert_eq!(warm_baseline, expected, "{name}: baseline parity");
        let warm_fused = fused_group_counts(&fix, 5, &bufs, stream);
        assert_eq!(warm_fused, expected, "{name}: fused parity");

        let mut unfused_ms = Vec::with_capacity(REPS);
        let mut fused_ms = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let t = std::time::Instant::now();
            let b = baseline_group_counts(&fix, 5, &bufs, stream);
            fix.provider.device().inner().synchronize().expect("sync");
            unfused_ms.push(t.elapsed().as_secs_f64() * 1e3);
            drop(b);

            let t = std::time::Instant::now();
            let f = fused_group_counts(&fix, 5, &bufs, stream);
            fix.provider.device().inner().synchronize().expect("sync");
            fused_ms.push(t.elapsed().as_secs_f64() * 1e3);
            drop(f);
        }
        unfused_ms.sort_by(|a, b| a.total_cmp(b));
        fused_ms.sort_by(|a, b| a.total_cmp(b));
        let med_unfused = unfused_ms[REPS / 2];
        let med_fused = fused_ms[REPS / 2];
        let n_rows: Vec<usize> = canonical_edge_list(5)
            .iter()
            .map(|p| edges.get(p).map(Vec::len).unwrap_or(0))
            .collect();
        println!(
            "aggregate-fused WCOJ K-clique count {name}: unfused median {med_unfused:.3} ms, fused median {med_fused:.3} ms, \
             speedup {:.2}x (edge rows {:?})",
            med_unfused / med_fused,
            n_rows
        );
    }
}
