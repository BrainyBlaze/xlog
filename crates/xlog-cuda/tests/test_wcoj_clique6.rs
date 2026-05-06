// crates/xlog-cuda/tests/test_wcoj_clique6.rs
//! W3.2 — k=6 clique provider × width-class round-trip certs.
//!
//! 3 cells: u32 / u64 / Symbol. Each:
//!   1. Build a small clique fixture (≤ 6 vertices).
//!   2. Upload each of the 10 edges to a CudaBuffer.
//!   3. Layout-sort+dedup each via W3.1's wcoj_layout_sort_*_recorded
//!      (the runtime dispatcher's pre-condition contract; certs
//!      satisfy it explicitly).
//!   4. Call provider entry on the laid-out edges.
//!   5. Download output, compare against `cpu_clique6_reference`
//!      brute-force oracle (set equality modulo row order).

use std::collections::BTreeSet;
use std::sync::Arc;

use cudarc::driver::sys;
use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamId, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::{CudaBuffer, CudaColumn};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

struct DiscardSink;
impl LoggingSink for DiscardSink {
    fn emit(&self, _record: LogRecord) -> Result<(), SinkError> {
        Ok(())
    }
}

#[allow(dead_code)]
struct RuntimeFixture {
    device: Arc<CudaDevice>,
    runtime: Arc<XlogDeviceRuntime>,
    memory: Arc<GpuMemoryManager>,
    provider: CudaKernelProvider,
    pool: Arc<StreamPool>,
}

fn make_runtime_fixture() -> Option<RuntimeFixture> {
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
        CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory)).ok()?;
    Some(RuntimeFixture {
        device,
        runtime,
        memory,
        provider,
        pool,
    })
}

/// Brute-force CPU oracle for K-clique enumeration. Generic over
/// the cell type T (u32 / u64). Edges in canonical lex (i, j)
/// order: (0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4),
/// (2,3), (2,4), (3,4) — 10 edges for K=6; 15 for K=6.
fn cpu_clique_reference<T, const K: usize>(edges: &[Vec<(T, T)>]) -> Vec<[T; K]>
where
    T: Copy + Ord + std::hash::Hash,
{
    assert_eq!(
        edges.len(),
        K * (K - 1) / 2,
        "clique-K oracle requires C(K,2) edge lists"
    );
    // For each edge build a HashSet for O(1) membership.
    let edge_sets: Vec<BTreeSet<(T, T)>> = edges
        .iter()
        .map(|rows| rows.iter().copied().collect())
        .collect();
    fn edge_idx(i: usize, j: usize, k: usize) -> usize {
        i * (k - 1) - i * (i - 1) / 2 + (j - i - 1)
    }
    // Vertex set = union of all values in all edges.
    let mut vertices: BTreeSet<T> = BTreeSet::new();
    for rows in edges {
        for (a, b) in rows {
            vertices.insert(*a);
            vertices.insert(*b);
        }
    }
    let vertex_vec: Vec<T> = vertices.into_iter().collect();
    // Generate all K-tuples (v_0, ..., v_{K-1}) of distinct vertices.
    let n = vertex_vec.len();
    let mut results: Vec<[T; K]> = Vec::new();
    let mut binding: [Option<T>; K] = [None; K];
    fn recurse<T: Copy + Ord, const K: usize>(
        depth: usize,
        binding: &mut [Option<T>; K],
        vertex_vec: &[T],
        edge_sets: &[BTreeSet<(T, T)>],
        results: &mut Vec<[T; K]>,
    ) {
        if depth == K {
            let row: [T; K] = std::array::from_fn(|i| binding[i].unwrap());
            results.push(row);
            return;
        }
        for &v in vertex_vec {
            // Reject duplicates within the binding.
            if binding.iter().take(depth).any(|x| x == &Some(v)) {
                continue;
            }
            // Check edge (j, depth) exists for all j < depth.
            let mut ok = true;
            for j in 0..depth {
                let vj = binding[j].unwrap();
                let e_idx = edge_idx(j, depth, K);
                if !edge_sets[e_idx].contains(&(vj, v)) {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            binding[depth] = Some(v);
            recurse::<T, K>(depth + 1, binding, vertex_vec, edge_sets, results);
            binding[depth] = None;
        }
    }
    recurse::<T, K>(0, &mut binding, &vertex_vec, &edge_sets, &mut results);
    results
}

/// Upload a 2-column buffer of u32-typed cells (used for both U32
/// and Symbol fixtures — Symbol is u32-physical with distinct
/// schema type).
fn upload_2col_u32(
    memory: &Arc<GpuMemoryManager>,
    rows: &[(u32, u32)],
    col_type: ScalarType,
) -> CudaBuffer {
    let n = rows.len() as u32;
    let bytes_per_col = (n as usize).max(1) * std::mem::size_of::<u32>();
    let mut col0 = memory.alloc::<u8>(bytes_per_col).expect("alloc col0");
    let mut col1 = memory.alloc::<u8>(bytes_per_col).expect("alloc col1");
    let mut d_n = memory.alloc::<u32>(1).expect("alloc d_n");
    let dev = memory.device().inner();
    if n > 0 {
        let c0: Vec<u8> = rows.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
        let c1: Vec<u8> = rows.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
        dev.htod_sync_copy_into(&c0, &mut col0).expect("htod col0");
        dev.htod_sync_copy_into(&c1, &mut col1).expect("htod col1");
    }
    dev.htod_sync_copy_into(&[n], &mut d_n).expect("htod d_n");
    let schema = Schema::new(vec![
        ("c0".to_string(), col_type),
        ("c1".to_string(), col_type),
    ]);
    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into()],
        n as u64,
        d_n,
        schema,
        n,
    )
}

fn upload_2col_u64(memory: &Arc<GpuMemoryManager>, rows: &[(u64, u64)]) -> CudaBuffer {
    let n = rows.len() as u32;
    let bytes_per_col = (n as usize).max(1) * std::mem::size_of::<u64>();
    let mut col0 = memory.alloc::<u8>(bytes_per_col).expect("alloc col0");
    let mut col1 = memory.alloc::<u8>(bytes_per_col).expect("alloc col1");
    let mut d_n = memory.alloc::<u32>(1).expect("alloc d_n");
    let dev = memory.device().inner();
    if n > 0 {
        let c0: Vec<u8> = rows.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
        let c1: Vec<u8> = rows.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
        dev.htod_sync_copy_into(&c0, &mut col0).expect("htod col0");
        dev.htod_sync_copy_into(&c1, &mut col1).expect("htod col1");
    }
    dev.htod_sync_copy_into(&[n], &mut d_n).expect("htod d_n");
    let schema = Schema::new(vec![
        ("c0".to_string(), ScalarType::U64),
        ("c1".to_string(), ScalarType::U64),
    ]);
    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into()],
        n as u64,
        d_n,
        schema,
        n,
    )
}

/// Build a complete K_5 fixture on a 5-vertex set
/// `{1, 2, 3, 4, 5}`. Returns the 10 host-side edge lists in
/// canonical lex (i, j) order. Vertex i in the canonical
/// numbering corresponds to host value `i + 1`.
fn k6_fixture_u32() -> Vec<Vec<(u32, u32)>> {
    let mut edges: Vec<Vec<(u32, u32)>> = Vec::with_capacity(15);
    for i in 0u32..6 {
        for j in (i + 1)..6 {
            // Edge (i, j) carries the single tuple (i+1, j+1).
            edges.push(vec![(i + 1, j + 1)]);
        }
    }
    edges
}

fn k6_fixture_u64() -> Vec<Vec<(u64, u64)>> {
    let mut edges: Vec<Vec<(u64, u64)>> = Vec::with_capacity(15);
    for i in 0u64..6 {
        for j in (i + 1)..6 {
            edges.push(vec![(i + 1, j + 1)]);
        }
    }
    edges
}

fn download_k6_u32(buf: &CudaBuffer) -> Vec<[u32; 6]> {
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
        return Vec::new();
    }
    let mut cols: Vec<Vec<u8>> = (0..6).map(|_| vec![0u8; n * 4]).collect();
    for c in 0..6 {
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
            std::array::from_fn(|c| {
                let off = r * 4;
                u32::from_le_bytes([
                    cols[c][off],
                    cols[c][off + 1],
                    cols[c][off + 2],
                    cols[c][off + 3],
                ])
            })
        })
        .collect()
}

fn download_k6_u64(buf: &CudaBuffer) -> Vec<[u64; 6]> {
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
        return Vec::new();
    }
    let mut cols: Vec<Vec<u8>> = (0..6).map(|_| vec![0u8; n * 8]).collect();
    for c in 0..6 {
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
            std::array::from_fn(|c| {
                let off = r * 8;
                u64::from_le_bytes([
                    cols[c][off],
                    cols[c][off + 1],
                    cols[c][off + 2],
                    cols[c][off + 3],
                    cols[c][off + 4],
                    cols[c][off + 5],
                    cols[c][off + 6],
                    cols[c][off + 7],
                ])
            })
        })
        .collect()
}

fn run_k6_test_u32_or_symbol(col_type: ScalarType) {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let host_edges = k6_fixture_u32();
    let raw_bufs: Vec<CudaBuffer> = host_edges
        .iter()
        .map(|rows| upload_2col_u32(&fix.memory, rows, col_type))
        .collect();
    let stream = fix.pool.acquire().expect("stream");
    let laid_out: Vec<CudaBuffer> = raw_bufs
        .iter()
        .map(|b| {
            fix.provider
                .wcoj_layout_sort_u32_recorded(b, stream)
                .expect("layout sort")
        })
        .collect();
    let edge_refs: Vec<&CudaBuffer> = laid_out.iter().collect();
    let arr: &[&CudaBuffer; 15] = edge_refs.as_slice().try_into().expect("15 edges");
    let out = fix
        .provider
        .wcoj_clique6_u32_recorded(arr, stream)
        .expect("clique6 u32");
    let actual: BTreeSet<[u32; 6]> = download_k6_u32(&out).into_iter().collect();
    let expected: BTreeSet<[u32; 6]> = cpu_clique_reference::<u32, 6>(&host_edges)
        .into_iter()
        .collect();
    assert_eq!(
        actual, expected,
        "K=6 ({:?}) row set must match CPU oracle",
        col_type
    );
    // Schema preservation.
    for col_idx in 0..6 {
        assert_eq!(
            out.schema.column_type(col_idx),
            Some(col_type),
            "K=6 ({:?}) col {} schema must be preserved",
            col_type,
            col_idx
        );
    }
}

#[test]
fn clique6_u32_round_trips_against_cpu_oracle() {
    run_k6_test_u32_or_symbol(ScalarType::U32);
}

#[test]
fn clique6_symbol_round_trips_against_cpu_oracle() {
    run_k6_test_u32_or_symbol(ScalarType::Symbol);
}

#[test]
fn clique6_u64_round_trips_against_cpu_oracle() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let host_edges = k6_fixture_u64();
    let raw_bufs: Vec<CudaBuffer> = host_edges
        .iter()
        .map(|rows| upload_2col_u64(&fix.memory, rows))
        .collect();
    let stream = fix.pool.acquire().expect("stream");
    let laid_out: Vec<CudaBuffer> = raw_bufs
        .iter()
        .map(|b| {
            fix.provider
                .wcoj_layout_sort_u64_recorded(b, stream)
                .expect("layout sort u64")
        })
        .collect();
    let edge_refs: Vec<&CudaBuffer> = laid_out.iter().collect();
    let arr: &[&CudaBuffer; 15] = edge_refs.as_slice().try_into().expect("15 edges");
    let out = fix
        .provider
        .wcoj_clique6_u64_recorded(arr, stream)
        .expect("clique6 u64");
    let actual: BTreeSet<[u64; 6]> = download_k6_u64(&out).into_iter().collect();
    let expected: BTreeSet<[u64; 6]> = cpu_clique_reference::<u64, 6>(&host_edges)
        .into_iter()
        .collect();
    assert_eq!(actual, expected, "K=6 (U64) row set must match CPU oracle");
    for col_idx in 0..6 {
        assert_eq!(
            out.schema.column_type(col_idx),
            Some(ScalarType::U64),
            "K=6 (U64) col {} schema must be preserved",
            col_idx
        );
    }
}
