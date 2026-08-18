// crates/xlog-cuda/tests/set_ops_tests.rs
//! Tests for GPU-native set operations (union, diff)

mod common;
use std::sync::{Mutex, MutexGuard, OnceLock};

use common::setup_provider;
use xlog_core::{ScalarType, Schema};
use xlog_cuda::{CudaBuffer, CudaKernelProvider};

fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

struct EnvGuard {
    _lock: MutexGuard<'static, ()>,
    old_graph: Option<String>,
}

impl EnvGuard {
    fn graph_mode() -> Self {
        let lock = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        let old_graph = std::env::var("XLOG_USE_CSM_CUDA_GRAPH").ok();
        std::env::set_var("XLOG_USE_CSM_CUDA_GRAPH", "1");
        Self {
            _lock: lock,
            old_graph,
        }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        match &self.old_graph {
            Some(value) => std::env::set_var("XLOG_USE_CSM_CUDA_GRAPH", value),
            None => std::env::remove_var("XLOG_USE_CSM_CUDA_GRAPH"),
        }
    }
}

struct ChunkBudgetGuard {
    _lock: MutexGuard<'static, ()>,
    old_budget: Option<String>,
}

impl ChunkBudgetGuard {
    fn with_budget(bytes: &str) -> Self {
        let lock = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        let old_budget = std::env::var("XLOG_UNION_CHUNK_BYTES").ok();
        std::env::set_var("XLOG_UNION_CHUNK_BYTES", bytes);
        Self {
            _lock: lock,
            old_budget,
        }
    }
}

impl Drop for ChunkBudgetGuard {
    fn drop(&mut self) {
        match &self.old_budget {
            Some(value) => std::env::set_var("XLOG_UNION_CHUNK_BYTES", value),
            None => std::env::remove_var("XLOG_UNION_CHUNK_BYTES"),
        }
    }
}

fn device_row_count(
    provider: &CudaKernelProvider,
    rows: u32,
) -> xlog_cuda::memory::TrackedCudaSlice<u32> {
    let mut d_num_rows = provider.memory().alloc::<u32>(1).expect("alloc");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&[rows], &mut d_num_rows)
        .expect("htod row count");
    d_num_rows
}

fn host_row_count(provider: &CudaKernelProvider, buffer: &CudaBuffer) -> u32 {
    let mut host = [0u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(buffer.num_rows_device(), &mut host)
        .expect("dtoh row count");
    host[0]
}

fn zero_arity_buffer(provider: &CudaKernelProvider, rows: u32) -> CudaBuffer {
    let schema = Schema::new(vec![]);
    let d_num_rows = device_row_count(provider, rows);
    CudaBuffer::from_columns(Vec::new(), rows as u64, d_num_rows, schema)
}

fn buffer_with_row_cap(
    provider: &CudaKernelProvider,
    data: &[u32],
    row_cap: u64,
    actual_rows: u32,
    schema: Schema,
) -> CudaBuffer {
    assert!(row_cap as usize >= data.len(), "row_cap must fit data");
    assert!(
        actual_rows as u64 <= row_cap,
        "actual_rows must be <= row_cap"
    );

    let mut bytes = Vec::with_capacity((row_cap as usize) * 4);
    for &v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    while bytes.len() < (row_cap as usize) * 4 {
        bytes.extend_from_slice(&0u32.to_le_bytes());
    }

    let mut col = provider.memory().alloc::<u8>(bytes.len()).expect("alloc");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&bytes, &mut col)
        .expect("htod data");

    let d_num_rows = device_row_count(provider, actual_rows);
    CudaBuffer::from_columns(vec![col.into()], row_cap, d_num_rows, schema)
}

fn buffer_i64_triples(provider: &CudaKernelProvider, rows: &[(i64, i64, i64)]) -> CudaBuffer {
    let c0: Vec<i64> = rows.iter().map(|r| r.0).collect();
    let c1: Vec<i64> = rows.iter().map(|r| r.1).collect();
    let c2: Vec<i64> = rows.iter().map(|r| r.2).collect();
    let schema = Schema::new(vec![
        ("c0".to_string(), ScalarType::I64),
        ("c1".to_string(), ScalarType::I64),
        ("c2".to_string(), ScalarType::I64),
    ]);
    let bytes0: Vec<u8> = c0.iter().flat_map(|v| v.to_le_bytes()).collect();
    let bytes1: Vec<u8> = c1.iter().flat_map(|v| v.to_le_bytes()).collect();
    let bytes2: Vec<u8> = c2.iter().flat_map(|v| v.to_le_bytes()).collect();
    provider
        .create_buffer_from_slices(&[&bytes0, &bytes1, &bytes2], schema)
        .expect("create i64 triple buffer")
}

fn read_i64_triples(provider: &CudaKernelProvider, buffer: &CudaBuffer) -> Vec<(i64, i64, i64)> {
    let c0 = provider.download_column::<i64>(buffer, 0).expect("c0");
    let c1 = provider.download_column::<i64>(buffer, 1).expect("c1");
    let c2 = provider.download_column::<i64>(buffer, 2).expect("c2");
    c0.into_iter()
        .zip(c1)
        .zip(c2)
        .map(|((a, b), c)| (a, b, c))
        .collect()
}

// ============== Union Tests ==============

#[test]
fn test_union_gpu_zero_arity() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let empty = zero_arity_buffer(&provider, 0);
    let unit = zero_arity_buffer(&provider, 1);

    let u1 = provider.union_gpu(&empty, &unit).unwrap();
    assert_eq!(host_row_count(&provider, &u1), 1);

    let u2 = provider.union_gpu(&unit, &empty).unwrap();
    assert_eq!(host_row_count(&provider, &u2), 1);

    let u3 = provider.union_gpu(&unit, &unit).unwrap();
    assert_eq!(host_row_count(&provider, &u3), 1);

    let u4 = provider.union_gpu(&empty, &empty).unwrap();
    assert_eq!(host_row_count(&provider, &u4), 0);
    assert!(u4.is_empty());
}

#[test]
fn test_union_gpu_basic() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // a = [1, 2, 3], b = [3, 4, 5]
    // Union should deduplicate: [1, 2, 3, 4, 5]
    let a: Vec<u32> = vec![1, 2, 3];
    let b: Vec<u32> = vec![3, 4, 5];
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    let buf_a = provider
        .create_buffer_from_slice::<u32>(&a, schema.clone())
        .unwrap();
    let buf_b = provider
        .create_buffer_from_slice::<u32>(&b, schema.clone())
        .unwrap();

    let result = provider.union_gpu(&buf_a, &buf_b).unwrap();
    let result_data = provider.download_column::<u32>(&result, 0).unwrap();

    assert_eq!(result_data, vec![1, 2, 3, 4, 5]);
}

#[test]
fn bounded_cuda_graph_small_i64_full_row_set_ops_match_baseline_and_use_small_sort() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let a = buffer_i64_triples(
        &provider,
        &[(3, 30, -1), (-5, 7, 8), (3, 30, -1), (1, 2, 3)],
    );
    let b = buffer_i64_triples(&provider, &[(1, 2, 3), (9, 0, -4), (-5, 7, 8), (4, 4, 4)]);

    let baseline_union = provider.union_gpu(&a, &b).expect("baseline union");
    let baseline_diff = provider
        .diff_gpu(&baseline_union, &b)
        .expect("baseline diff");
    let baseline_union_rows = read_i64_triples(&provider, &baseline_union);
    let baseline_diff_rows = read_i64_triples(&provider, &baseline_diff);

    let _guard = EnvGuard::graph_mode();
    let before = provider.small_full_row_sort_invocations();
    let graph_union = provider.union_gpu(&a, &b).expect("graph union");
    let graph_diff = provider.diff_gpu(&graph_union, &b).expect("graph diff");
    let after = provider.small_full_row_sort_invocations();

    assert_eq!(
        read_i64_triples(&provider, &graph_union),
        baseline_union_rows
    );
    assert_eq!(read_i64_triples(&provider, &graph_diff), baseline_diff_rows);
    assert!(
        after >= before + 3,
        "graph-mode union+diff should route small full-row set maintenance \
         through the bounded CUDA Graph small-sort path; before={before} after={after}"
    );
}

#[test]
fn test_union_gpu_no_overlap() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // a = [1, 2], b = [3, 4]
    // Union: [1, 2, 3, 4]
    let a: Vec<u32> = vec![1, 2];
    let b: Vec<u32> = vec![3, 4];
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    let buf_a = provider
        .create_buffer_from_slice::<u32>(&a, schema.clone())
        .unwrap();
    let buf_b = provider
        .create_buffer_from_slice::<u32>(&b, schema.clone())
        .unwrap();

    let result = provider.union_gpu(&buf_a, &buf_b).unwrap();
    let result_data = provider.download_column::<u32>(&result, 0).unwrap();

    assert_eq!(result_data, vec![1, 2, 3, 4]);
}

#[test]
fn test_union_gpu_complete_overlap() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // a = [1, 2, 3], b = [1, 2, 3]
    // Union: [1, 2, 3]
    let a: Vec<u32> = vec![1, 2, 3];
    let b: Vec<u32> = vec![1, 2, 3];
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    let buf_a = provider
        .create_buffer_from_slice::<u32>(&a, schema.clone())
        .unwrap();
    let buf_b = provider
        .create_buffer_from_slice::<u32>(&b, schema.clone())
        .unwrap();

    let result = provider.union_gpu(&buf_a, &buf_b).unwrap();
    let result_data = provider.download_column::<u32>(&result, 0).unwrap();

    assert_eq!(result_data, vec![1, 2, 3]);
}

#[test]
fn test_union_gpu_empty_a() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // a = [], b = [1, 2]
    // Union: [1, 2]
    let b: Vec<u32> = vec![1, 2];
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    let buf_a = provider.create_empty_buffer(schema.clone()).unwrap();
    let buf_b = provider
        .create_buffer_from_slice::<u32>(&b, schema.clone())
        .unwrap();

    let result = provider.union_gpu(&buf_a, &buf_b).unwrap();
    let result_data = provider.download_column::<u32>(&result, 0).unwrap();

    assert_eq!(result_data, vec![1, 2]);
}

#[test]
fn test_union_gpu_empty_b() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // a = [1, 2], b = []
    // Union: [1, 2]
    let a: Vec<u32> = vec![1, 2];
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    let buf_a = provider
        .create_buffer_from_slice::<u32>(&a, schema.clone())
        .unwrap();
    let buf_b = provider.create_empty_buffer(schema.clone()).unwrap();

    let result = provider.union_gpu(&buf_a, &buf_b).unwrap();
    let result_data = provider.download_column::<u32>(&result, 0).unwrap();

    assert_eq!(result_data, vec![1, 2]);
}

#[test]
fn test_union_gpu_both_empty() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // a = [], b = []
    // Union: []
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    let buf_a = provider.create_empty_buffer(schema.clone()).unwrap();
    let buf_b = provider.create_empty_buffer(schema.clone()).unwrap();

    let result = provider.union_gpu(&buf_a, &buf_b).unwrap();
    let result_data = provider.download_column::<u32>(&result, 0).unwrap();
    assert!(result_data.is_empty());
}

#[test]
fn test_union_gpu_with_duplicates_in_input() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // a = [1, 1, 2], b = [2, 3, 3]
    // Union (deduplicated): [1, 2, 3]
    let a: Vec<u32> = vec![1, 1, 2];
    let b: Vec<u32> = vec![2, 3, 3];
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    let buf_a = provider
        .create_buffer_from_slice::<u32>(&a, schema.clone())
        .unwrap();
    let buf_b = provider
        .create_buffer_from_slice::<u32>(&b, schema.clone())
        .unwrap();

    let result = provider.union_gpu(&buf_a, &buf_b).unwrap();
    let result_data = provider.download_column::<u32>(&result, 0).unwrap();

    assert_eq!(result_data, vec![1, 2, 3]);
}

#[test]
fn test_union_gpu_uses_device_row_count() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);
    let data = vec![1u32, 2, 99, 100];

    let buf_a = buffer_with_row_cap(&provider, &data, 4, 2, schema.clone());
    let buf_b = buffer_with_row_cap(&provider, &data, 4, 2, schema.clone());

    let result = provider.union_gpu(&buf_a, &buf_b).unwrap();
    let result_data = provider.download_column::<u32>(&result, 0).unwrap();

    assert_eq!(result_data, vec![1, 2]);
}

// ============== Multiway Union Tests ==============

#[test]
fn test_union_many_gpu_matches_pairwise_fold() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Overlapping inputs with internal duplicates: the one-pass multiway
    // union must produce the same sorted deduplicated relation as folding
    // union_gpu pairwise.
    let inputs: Vec<Vec<u32>> = vec![
        vec![5, 1, 5, 9],
        vec![2, 9, 3],
        vec![7, 1, 8, 8],
        vec![4, 3, 6],
    ];
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);
    let buffers: Vec<CudaBuffer> = inputs
        .iter()
        .map(|data| {
            provider
                .create_buffer_from_slice::<u32>(data, schema.clone())
                .unwrap()
        })
        .collect();

    let mut folded = provider.union_gpu(&buffers[0], &buffers[1]).unwrap();
    for buf in &buffers[2..] {
        folded = provider.union_gpu(&folded, buf).unwrap();
    }
    let folded_data = provider.download_column::<u32>(&folded, 0).unwrap();

    let refs: Vec<&CudaBuffer> = buffers.iter().collect();
    let many = provider.union_many_gpu(&refs).unwrap();
    let many_data = provider.download_column::<u32>(&many, 0).unwrap();

    assert_eq!(many_data, folded_data);
    assert_eq!(many_data, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
}

#[test]
fn test_union_many_gpu_multi_column() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Three multi-column inputs with cross-input duplicates exercise the
    // N-way concat over 8-byte columns.
    let a = buffer_i64_triples(&provider, &[(3, 30, -1), (-5, 7, 8)]);
    let b = buffer_i64_triples(&provider, &[(1, 2, 3), (3, 30, -1)]);
    let c = buffer_i64_triples(&provider, &[(-5, 7, 8), (9, 0, -4), (1, 2, 3)]);

    let folded = provider
        .union_gpu(&provider.union_gpu(&a, &b).unwrap(), &c)
        .unwrap();
    let many = provider.union_many_gpu(&[&a, &b, &c]).unwrap();

    let folded_rows = read_i64_triples(&provider, &folded);
    let many_rows = read_i64_triples(&provider, &many);
    assert_eq!(many_rows, folded_rows);
    // Hand-pinned expectation: the full-row deterministic order sorts the
    // four distinct triples numerically, independent of input order.
    assert_eq!(
        many_rows,
        vec![(-5, 7, 8), (1, 2, 3), (3, 30, -1), (9, 0, -4)]
    );
}

#[test]
fn test_union_many_gpu_many_inputs_high_multiplicity() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Eight inputs where every value appears in at least three of them:
    // exercises the N-way concat with more inputs than any pairwise-shaped
    // path and pins the exact deduplicated result by hand.
    let inputs: Vec<Vec<u32>> = vec![
        vec![1, 2, 3],
        vec![2, 3, 4],
        vec![3, 4, 5],
        vec![4, 5, 1],
        vec![5, 1, 2],
        vec![1, 3, 5],
        vec![2, 4, 1],
        vec![5, 2, 3],
    ];
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);
    let buffers: Vec<CudaBuffer> = inputs
        .iter()
        .map(|data| {
            provider
                .create_buffer_from_slice::<u32>(data, schema.clone())
                .unwrap()
        })
        .collect();

    let refs: Vec<&CudaBuffer> = buffers.iter().collect();
    let many = provider.union_many_gpu(&refs).unwrap();
    let many_data = provider.download_column::<u32>(&many, 0).unwrap();
    assert_eq!(many_data, vec![1, 2, 3, 4, 5]);

    let mut folded = provider.union_gpu(&buffers[0], &buffers[1]).unwrap();
    for buf in &buffers[2..] {
        folded = provider.union_gpu(&folded, buf).unwrap();
    }
    let folded_data = provider.download_column::<u32>(&folded, 0).unwrap();
    assert_eq!(many_data, folded_data);
}

#[test]
fn test_union_many_gpu_multi_chunk_fold_matches_single_chunk() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Same inputs as the high-multiplicity test, but with the chunk budget
    // forced to one byte so every input becomes its own fold pass. The
    // multi-chunk fold must produce exactly the single-chunk result.
    let inputs: Vec<Vec<u32>> = vec![
        vec![1, 2, 3],
        vec![2, 3, 4],
        vec![3, 4, 5],
        vec![4, 5, 1],
        vec![5, 1, 2],
        vec![1, 3, 5],
        vec![2, 4, 1],
        vec![5, 2, 3],
    ];
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);
    let buffers: Vec<CudaBuffer> = inputs
        .iter()
        .map(|data| {
            provider
                .create_buffer_from_slice::<u32>(data, schema.clone())
                .unwrap()
        })
        .collect();
    let refs: Vec<&CudaBuffer> = buffers.iter().collect();

    let single_chunk = provider.union_many_gpu(&refs).unwrap();
    let single_chunk_data = provider.download_column::<u32>(&single_chunk, 0).unwrap();

    let _guard = ChunkBudgetGuard::with_budget("1");
    let multi_chunk = provider.union_many_gpu(&refs).unwrap();
    let multi_chunk_data = provider.download_column::<u32>(&multi_chunk, 0).unwrap();

    assert_eq!(multi_chunk_data, single_chunk_data);
    assert_eq!(multi_chunk_data, vec![1, 2, 3, 4, 5]);
}

#[test]
fn test_union_many_gpu_multi_chunk_fold_multi_column() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Multi-column inputs through the forced multi-pass fold: cross-input
    // duplicates must still collapse to the hand-pinned deterministic set.
    let a = buffer_i64_triples(&provider, &[(3, 30, -1), (-5, 7, 8)]);
    let b = buffer_i64_triples(&provider, &[(1, 2, 3), (3, 30, -1)]);
    let c = buffer_i64_triples(&provider, &[(-5, 7, 8), (9, 0, -4), (1, 2, 3)]);

    let _guard = ChunkBudgetGuard::with_budget("1");
    let many = provider.union_many_gpu(&[&a, &b, &c]).unwrap();
    assert_eq!(
        read_i64_triples(&provider, &many),
        vec![(-5, 7, 8), (1, 2, 3), (3, 30, -1), (9, 0, -4)]
    );
}

#[test]
fn bounded_cuda_graph_union_many_matches_baseline() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Three small multi-column inputs: in graph mode the batched union must
    // route through the bounded CUDA-graph small-sort path and still match
    // the baseline result byte for byte.
    let a = buffer_i64_triples(&provider, &[(3, 30, -1), (-5, 7, 8), (3, 30, -1)]);
    let b = buffer_i64_triples(&provider, &[(1, 2, 3), (9, 0, -4)]);
    let c = buffer_i64_triples(&provider, &[(-5, 7, 8), (4, 4, 4), (1, 2, 3)]);

    let baseline = provider.union_many_gpu(&[&a, &b, &c]).expect("baseline");
    let baseline_rows = read_i64_triples(&provider, &baseline);

    let _guard = EnvGuard::graph_mode();
    let before = provider.small_full_row_sort_invocations();
    let graph = provider.union_many_gpu(&[&a, &b, &c]).expect("graph union");
    let after = provider.small_full_row_sort_invocations();

    assert_eq!(read_i64_triples(&provider, &graph), baseline_rows);
    assert_eq!(
        baseline_rows,
        vec![(-5, 7, 8), (1, 2, 3), (3, 30, -1), (4, 4, 4), (9, 0, -4)]
    );
    assert!(
        after > before,
        "graph-mode multiway union should route through the bounded CUDA \
         Graph small-sort path; before={before} after={after}"
    );
}

#[test]
fn test_union_many_gpu_single_input_dedups() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);
    let buf = provider
        .create_buffer_from_slice::<u32>(&[3, 1, 3, 2, 1], schema)
        .unwrap();

    // Set semantics: a single input is still deduplicated.
    let result = provider.union_many_gpu(&[&buf]).unwrap();
    let result_data = provider.download_column::<u32>(&result, 0).unwrap();
    assert_eq!(result_data, vec![1, 2, 3]);
}

#[test]
fn test_union_many_gpu_single_certified_set_avoids_metadata_read_and_rededup() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);
    let bag = provider
        .create_buffer_from_slice::<u32>(&[3, 1, 3, 2, 1], schema)
        .unwrap();
    provider.reset_host_transfer_stats();
    let set = provider.union_many_gpu(&[&bag]).unwrap();
    assert_eq!(
        provider.host_transfer_stats().dtoh_calls,
        0,
        "a cached exact input count must not require a metadata D2H"
    );
    assert!(set.canonical_full_row_set_certified());
    let empty = provider.create_empty_buffer(set.schema().clone()).unwrap();

    provider.reset_host_transfer_stats();
    provider.reset_untracked_metadata_dtoh_count();
    provider.memory().reset_alloc_count();
    let result = provider.union_many_gpu(&[&empty, &set, &empty]).unwrap();

    let transfers = provider.host_transfer_stats();
    assert_eq!(
        transfers.dtoh_calls, 0,
        "cached set reuse must stay on device"
    );
    assert_eq!(provider.untracked_metadata_dtoh_count(), 0);
    assert_eq!(
        provider.memory().alloc_count(),
        (set.arity() + 1) as u64,
        "the fast path should only clone columns and the device row count"
    );
    assert!(result.canonical_full_row_set_certified());
    assert_eq!(
        provider.download_column::<u32>(&result, 0).unwrap(),
        vec![1, 2, 3]
    );
}

#[test]
fn test_union_many_gpu_partial_key_dedup_is_not_a_canonical_set_certificate() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![
        ("left".to_string(), ScalarType::U32),
        ("right".to_string(), ScalarType::U32),
    ]);
    let input = provider
        .create_buffer_from_u32_columns(&[&[2, 1], &[1, 2]], schema)
        .unwrap();
    let by_second_column = provider.dedup(&input, &[1]).unwrap();
    assert!(!by_second_column.canonical_full_row_set_certified());

    let result = provider.union_many_gpu(&[&by_second_column]).unwrap();
    assert_eq!(
        provider.download_column::<u32>(&result, 0).unwrap(),
        vec![1, 2]
    );
    assert_eq!(
        provider.download_column::<u32>(&result, 1).unwrap(),
        vec![2, 1]
    );
    assert!(result.canonical_full_row_set_certified());
}

#[test]
fn dedup_sorted_does_not_certify_an_unchecked_sorted_precondition() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);
    let bag = provider
        .create_buffer_from_slice::<u32>(&[2, 1, 2], schema)
        .unwrap();
    let adjacency_only = provider.dedup_sorted(&bag, &[0]).unwrap();
    assert!(!adjacency_only.canonical_full_row_set_certified());

    let canonical = provider.union_many_gpu(&[&adjacency_only]).unwrap();
    assert_eq!(
        provider.download_column::<u32>(&canonical, 0).unwrap(),
        vec![1, 2]
    );
    assert!(canonical.canonical_full_row_set_certified());
}

#[test]
fn full_row_canonicalization_rejects_device_count_above_capacity() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);
    let mut column = provider.memory().alloc::<u8>(4).expect("column");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&7u32.to_ne_bytes(), &mut column)
        .expect("column upload");
    let count = device_row_count(&provider, 2);
    let malformed = CudaBuffer::from_columns(vec![column.into()], 1, count, schema);
    assert!(!malformed.canonical_full_row_set_certified());

    let error = match provider.dedup(&malformed, &[0]) {
        Err(error) => error,
        Ok(_) => panic!("count above capacity was accepted by dedup"),
    };
    assert!(
        error
            .to_string()
            .contains("Logical row count 2 exceeds row capacity 1"),
        "unexpected error: {error}"
    );
    assert!(!malformed.canonical_full_row_set_certified());

    let error = match provider.union_many_gpu(&[&malformed]) {
        Err(error) => error,
        Ok(_) => panic!("count above capacity was accepted by union"),
    };
    assert!(
        error
            .to_string()
            .contains("Logical row count 2 exceeds row capacity 1"),
        "unexpected error: {error}"
    );
    assert!(!malformed.canonical_full_row_set_certified());

    let multi_schema = Schema::new(vec![
        ("left".to_string(), ScalarType::U32),
        ("right".to_string(), ScalarType::U32),
    ]);
    let mut left = provider.memory().alloc::<u8>(4).expect("left column");
    let mut right = provider.memory().alloc::<u8>(4).expect("right column");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&1u32.to_ne_bytes(), &mut left)
        .expect("left upload");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&2u32.to_ne_bytes(), &mut right)
        .expect("right upload");
    let count = device_row_count(&provider, 2);
    let malformed_multi = CudaBuffer::from_columns(
        vec![left.into(), right.into()],
        1,
        count,
        multi_schema.clone(),
    );
    let empty = provider
        .create_empty_buffer(multi_schema)
        .expect("empty subtractor");
    let error = match provider.diff_gpu(&malformed_multi, &empty) {
        Err(error) => error,
        Ok(_) => panic!("count above capacity was accepted by full-row diff"),
    };
    assert!(
        error
            .to_string()
            .contains("Logical row count 2 exceeds row capacity 1"),
        "unexpected error: {error}"
    );
    assert!(!malformed_multi.canonical_full_row_set_certified());
}

#[test]
fn test_union_many_gpu_public_mutation_invalidates_count_and_set_metadata() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);
    let mut empty = provider.create_empty_buffer(schema.clone()).unwrap();
    assert_eq!(empty.cached_row_count(), Some(0));
    let mut singleton_column = provider.memory().alloc::<u8>(4).unwrap();
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&7u32.to_le_bytes(), &mut singleton_column)
        .unwrap();
    empty.columns_mut()[0] = singleton_column.into();
    empty.set_row_capacity(1);
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&[1], empty.num_rows_device_mut())
        .unwrap();
    assert_eq!(empty.cached_row_count(), None);
    assert!(!empty.canonical_full_row_set_certified());
    let singleton = provider.union_many_gpu(&[&empty]).unwrap();
    assert_eq!(
        provider.download_column::<u32>(&singleton, 0).unwrap(),
        vec![7]
    );

    let bag = provider
        .create_buffer_from_slice::<u32>(&[1, 2, 3], schema)
        .unwrap();
    let mut canonical = provider.union_many_gpu(&[&bag]).unwrap();
    assert!(canonical.canonical_full_row_set_certified());
    let mut duplicate_column = provider.memory().alloc::<u8>(12).unwrap();
    let duplicate_bytes: Vec<u8> = [1u32, 1, 2]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&duplicate_bytes, &mut duplicate_column)
        .unwrap();
    canonical.columns_mut()[0] = duplicate_column.into();
    assert!(!canonical.canonical_full_row_set_certified());
    let deduped = provider.union_many_gpu(&[&canonical]).unwrap();
    assert_eq!(
        provider.download_column::<u32>(&deduped, 0).unwrap(),
        vec![1, 2]
    );
}

#[test]
fn test_union_many_gpu_skips_empty_inputs() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);
    let empty = provider.create_empty_buffer(schema.clone()).unwrap();
    let a = provider
        .create_buffer_from_slice::<u32>(&[2, 1], schema.clone())
        .unwrap();
    let b = provider
        .create_buffer_from_slice::<u32>(&[3, 2], schema.clone())
        .unwrap();

    let result = provider.union_many_gpu(&[&empty, &a, &empty, &b]).unwrap();
    let result_data = provider.download_column::<u32>(&result, 0).unwrap();
    assert_eq!(result_data, vec![1, 2, 3]);

    let all_empty = provider.union_many_gpu(&[&empty, &empty]).unwrap();
    let all_empty_data = provider.download_column::<u32>(&all_empty, 0).unwrap();
    assert!(all_empty_data.is_empty());
}

#[test]
fn test_union_many_gpu_zero_arity() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![]);
    let empty = provider
        .create_zero_arity_buffer(schema.clone(), 0)
        .unwrap();
    let unit = provider
        .create_zero_arity_buffer(schema.clone(), 1)
        .unwrap();
    let multiplicity = provider.create_zero_arity_buffer(schema, 2).unwrap();
    assert!(empty.canonical_full_row_set_certified());
    assert!(unit.canonical_full_row_set_certified());
    assert!(!multiplicity.canonical_full_row_set_certified());

    let u1 = provider.union_many_gpu(&[&empty, &unit, &empty]).unwrap();
    assert_eq!(host_row_count(&provider, &u1), 1);

    let u2 = provider.union_many_gpu(&[&empty, &empty, &empty]).unwrap();
    assert_eq!(host_row_count(&provider, &u2), 0);
    assert!(u2.is_empty());
}

#[test]
fn test_union_many_gpu_no_inputs_is_error() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    assert!(provider.union_many_gpu(&[]).is_err());
}

#[test]
fn test_union_many_gpu_respects_row_caps() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Buffers whose row capacity exceeds their logical row count: the N-way
    // concat must copy logical rows only, never capacity padding.
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);
    let data = vec![1u32, 2, 99, 100];
    let a = buffer_with_row_cap(&provider, &data, 4, 2, schema.clone());
    let b = buffer_with_row_cap(&provider, &data, 4, 2, schema.clone());
    let c = buffer_with_row_cap(&provider, &[3u32, 98, 97], 3, 1, schema.clone());

    let result = provider.union_many_gpu(&[&a, &b, &c]).unwrap();
    let result_data = provider.download_column::<u32>(&result, 0).unwrap();
    assert_eq!(result_data, vec![1, 2, 3]);
}

#[test]
fn test_union_many_gpu_incompatible_schemas_is_error() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let a = provider
        .create_buffer_from_slice::<u32>(
            &[1, 2],
            Schema::new(vec![("val".to_string(), ScalarType::U32)]),
        )
        .unwrap();
    let b = buffer_i64_triples(&provider, &[(1, 2, 3)]);

    assert!(provider.union_many_gpu(&[&a, &b]).is_err());
}

// ============== Diff Tests ==============

#[test]
fn test_diff_gpu_zero_arity() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let empty = zero_arity_buffer(&provider, 0);
    let unit = zero_arity_buffer(&provider, 1);

    // unit - empty = unit
    let d1 = provider.diff_gpu(&unit, &empty).unwrap();
    assert_eq!(host_row_count(&provider, &d1), 1);

    // unit - unit = empty
    let d2 = provider.diff_gpu(&unit, &unit).unwrap();
    assert_eq!(host_row_count(&provider, &d2), 0);
    assert!(d2.is_empty());

    // empty - unit = empty
    let d3 = provider.diff_gpu(&empty, &unit).unwrap();
    assert_eq!(host_row_count(&provider, &d3), 0);
    assert!(d3.is_empty());
}

#[test]
fn test_compact_buffer_by_device_mask_counted_empty_result_has_zero_device_rows() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);
    let input = provider
        .create_buffer_from_slice::<u32>(&[1u32, 2u32, 3u32], schema.clone())
        .unwrap();

    let mut d_mask = provider.memory().alloc::<u8>(3).unwrap();
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&[0u8, 0u8, 0u8], &mut d_mask)
        .unwrap();

    let out = provider
        .compact_buffer_by_device_mask_counted(&input, &d_mask)
        .unwrap();
    assert_eq!(host_row_count(&provider, &out), 0);
    assert_eq!(out.num_rows(), input.num_rows());
    assert_eq!(out.schema().arity(), schema.arity());
}

#[test]
fn test_diff_gpu_basic() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // a = [1, 2, 3, 4], b = [2, 4]
    // Diff: a - b = [1, 3]
    let a: Vec<u32> = vec![1, 2, 3, 4];
    let b: Vec<u32> = vec![2, 4];
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    let buf_a = provider
        .create_buffer_from_slice::<u32>(&a, schema.clone())
        .unwrap();
    let buf_b = provider
        .create_buffer_from_slice::<u32>(&b, schema.clone())
        .unwrap();

    let result = provider.diff_gpu(&buf_a, &buf_b).unwrap();
    let result_data = provider.download_column::<u32>(&result, 0).unwrap();

    assert_eq!(result_data, vec![1, 3]);
}

#[test]
fn test_diff_gpu_no_overlap() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // a = [1, 2, 3], b = [4, 5, 6]
    // Diff: a - b = [1, 2, 3] (no overlap)
    let a: Vec<u32> = vec![1, 2, 3];
    let b: Vec<u32> = vec![4, 5, 6];
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    let buf_a = provider
        .create_buffer_from_slice::<u32>(&a, schema.clone())
        .unwrap();
    let buf_b = provider
        .create_buffer_from_slice::<u32>(&b, schema.clone())
        .unwrap();

    let result = provider.diff_gpu(&buf_a, &buf_b).unwrap();
    let result_data = provider.download_column::<u32>(&result, 0).unwrap();

    assert_eq!(result_data, vec![1, 2, 3]);
}

#[test]
fn test_diff_gpu_complete_overlap() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // a = [1, 2, 3], b = [1, 2, 3]
    // Diff: a - b = [] (complete overlap)
    let a: Vec<u32> = vec![1, 2, 3];
    let b: Vec<u32> = vec![1, 2, 3];
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    let buf_a = provider
        .create_buffer_from_slice::<u32>(&a, schema.clone())
        .unwrap();
    let buf_b = provider
        .create_buffer_from_slice::<u32>(&b, schema.clone())
        .unwrap();

    let result = provider.diff_gpu(&buf_a, &buf_b).unwrap();
    let result_data = provider.download_column::<u32>(&result, 0).unwrap();
    assert!(result_data.is_empty());
}

#[test]
fn test_diff_gpu_empty_a() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // a = [], b = [1, 2]
    // Diff: [] - [1,2] = []
    let b: Vec<u32> = vec![1, 2];
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    let buf_a = provider.create_empty_buffer(schema.clone()).unwrap();
    let buf_b = provider
        .create_buffer_from_slice::<u32>(&b, schema.clone())
        .unwrap();

    let result = provider.diff_gpu(&buf_a, &buf_b).unwrap();
    let result_data = provider.download_column::<u32>(&result, 0).unwrap();
    assert!(result_data.is_empty());
}

#[test]
fn test_diff_gpu_empty_b() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // a = [1, 2, 3], b = []
    // Diff: [1,2,3] - [] = [1, 2, 3]
    let a: Vec<u32> = vec![1, 2, 3];
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    let buf_a = provider
        .create_buffer_from_slice::<u32>(&a, schema.clone())
        .unwrap();
    let buf_b = provider.create_empty_buffer(schema.clone()).unwrap();

    let result = provider.diff_gpu(&buf_a, &buf_b).unwrap();
    let result_data = provider.download_column::<u32>(&result, 0).unwrap();

    assert_eq!(result_data, vec![1, 2, 3]);
}

#[test]
fn test_diff_gpu_b_superset() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // a = [2, 3], b = [1, 2, 3, 4, 5]
    // Diff: [2,3] - [1,2,3,4,5] = []
    let a: Vec<u32> = vec![2, 3];
    let b: Vec<u32> = vec![1, 2, 3, 4, 5];
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    let buf_a = provider
        .create_buffer_from_slice::<u32>(&a, schema.clone())
        .unwrap();
    let buf_b = provider
        .create_buffer_from_slice::<u32>(&b, schema.clone())
        .unwrap();

    let result = provider.diff_gpu(&buf_a, &buf_b).unwrap();
    let result_data = provider.download_column::<u32>(&result, 0).unwrap();
    assert!(result_data.is_empty());
}

#[test]
fn test_diff_gpu_unsorted_inputs() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Unsorted inputs should still work (sorted internally)
    // a = [4, 1, 3, 2], b = [2, 4]
    // Diff: {1,2,3,4} - {2,4} = [1, 3]
    let a: Vec<u32> = vec![4, 1, 3, 2];
    let b: Vec<u32> = vec![2, 4];
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    let buf_a = provider
        .create_buffer_from_slice::<u32>(&a, schema.clone())
        .unwrap();
    let buf_b = provider
        .create_buffer_from_slice::<u32>(&b, schema.clone())
        .unwrap();

    let result = provider.diff_gpu(&buf_a, &buf_b).unwrap();
    let result_data = provider.download_column::<u32>(&result, 0).unwrap();

    // Result should be sorted: [1, 3]
    assert_eq!(result_data, vec![1, 3]);
}

#[test]
fn test_diff_gpu_with_duplicates() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // a has duplicates: [1, 1, 2, 2, 3], b = [2]
    // After dedup and diff: {1, 2, 3} - {2} = [1, 3]
    let a: Vec<u32> = vec![1, 1, 2, 2, 3];
    let b: Vec<u32> = vec![2];
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    let buf_a = provider
        .create_buffer_from_slice::<u32>(&a, schema.clone())
        .unwrap();
    let buf_b = provider
        .create_buffer_from_slice::<u32>(&b, schema.clone())
        .unwrap();

    let result = provider.diff_gpu(&buf_a, &buf_b).unwrap();
    let result_data = provider.download_column::<u32>(&result, 0).unwrap();

    assert_eq!(result_data, vec![1, 3]);
}

// ============== U64 Union Tests ==============

#[test]
fn test_union_u64() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::U64)]);

    let a_vals: Vec<u64> = vec![1, 2, 3];
    let b_vals: Vec<u64> = vec![2, 3, 4];

    let a_buf = provider
        .create_buffer_from_slice::<u64>(&a_vals, schema.clone())
        .unwrap();
    let b_buf = provider
        .create_buffer_from_slice::<u64>(&b_vals, schema)
        .unwrap();

    let result = provider.union_gpu(&a_buf, &b_buf).unwrap();
    let result_data = provider.download_column::<u64>(&result, 0).unwrap();
    assert_eq!(result_data.len(), 4); // 1, 2, 3, 4
    assert_eq!(result_data, vec![1, 2, 3, 4]);
}

#[test]
fn test_union_u64_with_duplicates() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::U64)]);

    // Input with duplicates
    let a_vals: Vec<u64> = vec![1, 1, 2];
    let b_vals: Vec<u64> = vec![2, 3, 3];

    let a_buf = provider
        .create_buffer_from_slice::<u64>(&a_vals, schema.clone())
        .unwrap();
    let b_buf = provider
        .create_buffer_from_slice::<u64>(&b_vals, schema)
        .unwrap();

    let result = provider.union_gpu(&a_buf, &b_buf).unwrap();
    let result_data = provider.download_column::<u64>(&result, 0).unwrap();
    assert_eq!(result_data.len(), 3); // 1, 2, 3
    assert_eq!(result_data, vec![1, 2, 3]);
}

#[test]
fn test_union_u64_empty_a() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::U64)]);

    let b_vals: Vec<u64> = vec![1, 2, 3];

    let a_buf = provider.create_empty_buffer(schema.clone()).unwrap();
    let b_buf = provider
        .create_buffer_from_slice::<u64>(&b_vals, schema)
        .unwrap();

    let result = provider.union_gpu(&a_buf, &b_buf).unwrap();
    let result_data = provider.download_column::<u64>(&result, 0).unwrap();
    assert_eq!(result_data.len(), 3);
    assert_eq!(result_data, vec![1, 2, 3]);
}

// ============== U64 Diff Tests ==============

#[test]
fn test_diff_u64() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::U64)]);

    let a_vals: Vec<u64> = vec![1, 2, 3, 4];
    let b_vals: Vec<u64> = vec![2, 4];

    let a_buf = provider
        .create_buffer_from_slice::<u64>(&a_vals, schema.clone())
        .unwrap();
    let b_buf = provider
        .create_buffer_from_slice::<u64>(&b_vals, schema)
        .unwrap();

    let result = provider.diff_gpu(&a_buf, &b_buf).unwrap();
    let result_data = provider.download_column::<u64>(&result, 0).unwrap();
    assert_eq!(result_data.len(), 2); // 1, 3
    assert_eq!(result_data, vec![1, 3]);
}

#[test]
fn test_diff_u64_no_overlap() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::U64)]);

    let a_vals: Vec<u64> = vec![1, 2, 3];
    let b_vals: Vec<u64> = vec![4, 5, 6];

    let a_buf = provider
        .create_buffer_from_slice::<u64>(&a_vals, schema.clone())
        .unwrap();
    let b_buf = provider
        .create_buffer_from_slice::<u64>(&b_vals, schema)
        .unwrap();

    let result = provider.diff_gpu(&a_buf, &b_buf).unwrap();
    let result_data = provider.download_column::<u64>(&result, 0).unwrap();
    assert_eq!(result_data.len(), 3); // All remain
    assert_eq!(result_data, vec![1, 2, 3]);
}

#[test]
fn test_diff_u64_complete_overlap() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::U64)]);

    let a_vals: Vec<u64> = vec![1, 2, 3];
    let b_vals: Vec<u64> = vec![1, 2, 3];

    let a_buf = provider
        .create_buffer_from_slice::<u64>(&a_vals, schema.clone())
        .unwrap();
    let b_buf = provider
        .create_buffer_from_slice::<u64>(&b_vals, schema)
        .unwrap();

    let result = provider.diff_gpu(&a_buf, &b_buf).unwrap();
    let result_data = provider.download_column::<u64>(&result, 0).unwrap();
    assert!(result_data.is_empty()); // Complete overlap
}

#[test]
fn test_diff_u64_empty_b() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::U64)]);

    let a_vals: Vec<u64> = vec![1, 2, 3];

    let a_buf = provider
        .create_buffer_from_slice::<u64>(&a_vals, schema.clone())
        .unwrap();
    let b_buf = provider.create_empty_buffer(schema).unwrap();

    let result = provider.diff_gpu(&a_buf, &b_buf).unwrap();
    let result_data = provider.download_column::<u64>(&result, 0).unwrap();
    assert_eq!(result_data.len(), 3); // All remain
    assert_eq!(result_data, vec![1, 2, 3]);
}

// ============== I64 Union Tests ==============

#[test]
fn test_union_i64() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::I64)]);

    // Include negative values to test proper ordering
    let a_vals: Vec<i64> = vec![-10, -5, 0, 5];
    let b_vals: Vec<i64> = vec![-5, 0, 10, 20];

    let a = provider
        .create_buffer_from_slice::<i64>(&a_vals, schema.clone())
        .unwrap();
    let b = provider
        .create_buffer_from_slice::<i64>(&b_vals, schema)
        .unwrap();

    let result = provider.union_gpu(&a, &b).unwrap();
    let result_data = provider.download_column::<i64>(&result, 0).unwrap();
    assert_eq!(result_data.len(), 6); // -10, -5, 0, 5, 10, 20
    assert_eq!(result_data, vec![-10, -5, 0, 5, 10, 20]);
}

#[test]
fn test_union_i64_with_duplicates() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::I64)]);

    let a_vals: Vec<i64> = vec![-5, -5, 0];
    let b_vals: Vec<i64> = vec![0, 5, 5];

    let a = provider
        .create_buffer_from_slice::<i64>(&a_vals, schema.clone())
        .unwrap();
    let b = provider
        .create_buffer_from_slice::<i64>(&b_vals, schema)
        .unwrap();

    let result = provider.union_gpu(&a, &b).unwrap();
    let result_data = provider.download_column::<i64>(&result, 0).unwrap();
    assert_eq!(result_data.len(), 3); // -5, 0, 5
    assert_eq!(result_data, vec![-5, 0, 5]);
}

// ============== F64 Union Tests ==============

#[test]
fn test_union_f64() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::F64)]);

    let a_vals: Vec<f64> = vec![1.5, 2.5, 3.5];
    let b_vals: Vec<f64> = vec![2.5, 3.5, 4.5];

    let a = provider
        .create_buffer_from_slice::<f64>(&a_vals, schema.clone())
        .unwrap();
    let b = provider
        .create_buffer_from_slice::<f64>(&b_vals, schema)
        .unwrap();

    let result = provider.union_gpu(&a, &b).unwrap();
    let result_data = provider.download_column::<f64>(&result, 0).unwrap();
    assert_eq!(result_data.len(), 4); // 1.5, 2.5, 3.5, 4.5
    assert_eq!(result_data, vec![1.5, 2.5, 3.5, 4.5]);
}

#[test]
fn test_union_f64_with_duplicates() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::F64)]);

    let a_vals: Vec<f64> = vec![1.5, 1.5, 2.5];
    let b_vals: Vec<f64> = vec![2.5, 3.5, 3.5];

    let a = provider
        .create_buffer_from_slice::<f64>(&a_vals, schema.clone())
        .unwrap();
    let b = provider
        .create_buffer_from_slice::<f64>(&b_vals, schema)
        .unwrap();

    let result = provider.union_gpu(&a, &b).unwrap();
    let result_data = provider.download_column::<f64>(&result, 0).unwrap();
    assert_eq!(result_data.len(), 3); // 1.5, 2.5, 3.5
    assert_eq!(result_data, vec![1.5, 2.5, 3.5]);
}

// ============== I64 Diff Tests ==============

#[test]
fn test_diff_i64() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::I64)]);

    let a_vals: Vec<i64> = vec![-10, -5, 0, 5, 10];
    let b_vals: Vec<i64> = vec![-5, 5];

    let a = provider
        .create_buffer_from_slice::<i64>(&a_vals, schema.clone())
        .unwrap();
    let b = provider
        .create_buffer_from_slice::<i64>(&b_vals, schema)
        .unwrap();

    let result = provider.diff_gpu(&a, &b).unwrap();
    let result_data = provider.download_column::<i64>(&result, 0).unwrap();
    assert_eq!(result_data.len(), 3); // -10, 0, 10
    assert_eq!(result_data, vec![-10, 0, 10]);
}

#[test]
fn test_diff_i64_no_overlap() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::I64)]);

    let a_vals: Vec<i64> = vec![-10, -5, 0];
    let b_vals: Vec<i64> = vec![5, 10, 15];

    let a = provider
        .create_buffer_from_slice::<i64>(&a_vals, schema.clone())
        .unwrap();
    let b = provider
        .create_buffer_from_slice::<i64>(&b_vals, schema)
        .unwrap();

    let result = provider.diff_gpu(&a, &b).unwrap();
    let result_data = provider.download_column::<i64>(&result, 0).unwrap();
    assert_eq!(result_data.len(), 3); // All remain: -10, -5, 0
    assert_eq!(result_data, vec![-10, -5, 0]);
}

// ============== F64 Diff Tests ==============

#[test]
fn test_diff_f64() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::F64)]);

    let a_vals: Vec<f64> = vec![1.5, 2.5, 3.5, 4.5];
    let b_vals: Vec<f64> = vec![2.5, 4.5];

    let a = provider
        .create_buffer_from_slice::<f64>(&a_vals, schema.clone())
        .unwrap();
    let b = provider
        .create_buffer_from_slice::<f64>(&b_vals, schema)
        .unwrap();

    let result = provider.diff_gpu(&a, &b).unwrap();
    let result_data = provider.download_column::<f64>(&result, 0).unwrap();
    assert_eq!(result_data.len(), 2); // 1.5, 3.5
    assert_eq!(result_data, vec![1.5, 3.5]);
}

#[test]
fn test_diff_f64_no_overlap() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::F64)]);

    let a_vals: Vec<f64> = vec![1.5, 2.5, 3.5];
    let b_vals: Vec<f64> = vec![4.5, 5.5, 6.5];

    let a = provider
        .create_buffer_from_slice::<f64>(&a_vals, schema.clone())
        .unwrap();
    let b = provider
        .create_buffer_from_slice::<f64>(&b_vals, schema)
        .unwrap();

    let result = provider.diff_gpu(&a, &b).unwrap();
    let result_data = provider.download_column::<f64>(&result, 0).unwrap();
    assert_eq!(result_data.len(), 3); // All remain: 1.5, 2.5, 3.5
    assert_eq!(result_data, vec![1.5, 2.5, 3.5]);
}
