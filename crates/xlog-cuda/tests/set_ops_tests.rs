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
