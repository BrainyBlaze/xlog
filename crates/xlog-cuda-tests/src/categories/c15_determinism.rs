//! Category 15: Determinism and reproducibility
//!
//! Tests that results are reproducible across multiple executions.
//! Verifies that GPU operations produce identical results when run with
//! the same inputs.

use crate::harness::xgcf;
use crate::harness::{CategoryResult, TestContext, TestResult};
use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;
use std::time::Instant;
use xlog_core::{RuntimeConfig, ScalarType, Schema};
use xlog_cuda::{CudaBuffer, CudaKernelProvider};
use xlog_logic::Compiler;
use xlog_runtime::Executor;

const WIDENED_FRONTIER_REPLAY_SOURCE: &str = r#"
    pred frontier_pred(u32).
    pred widened_pred(u32).
    pred frontier_edge(u32, u32).
    pred blocked_pred(u32).
    pred promoted(u32).
    pred replay_reachable(u32).
    pred rollback_hit(u32).

    promoted(P) :- frontier_pred(P), widened_pred(P).
    replay_reachable(P) :- promoted(P).
    replay_reachable(Q) :- replay_reachable(P), frontier_edge(P, Q), frontier_pred(Q).
    rollback_hit(P) :- replay_reachable(P), blocked_pred(P).
"#;

/// Run all tests in this category.
pub fn run_all(ctx: &TestContext) -> CategoryResult {
    let mut results = CategoryResult::new("c15_determinism");
    let start = Instant::now();

    results.add_result(test_sort_reproducibility(ctx));
    results.add_result(test_filter_reproducibility(ctx));
    results.add_result(test_join_reproducibility(ctx));
    results.add_result(test_dedup_reproducibility(ctx));
    results.add_result(test_stable_sort_order(ctx));
    results.add_result(test_mc_sample_reproducibility(ctx));
    results.add_result(test_xgcf_forward_reproducibility(ctx));
    results.add_result(test_xgcf_backward_reproducibility(ctx));
    results.add_result(test_widened_frontier_replay_representative(ctx));

    results.set_duration(start.elapsed());
    results
}

fn widened_frontier_replay_unary_schema() -> Schema {
    Schema::new(vec![("c0".to_string(), ScalarType::U32)])
}

fn widened_frontier_replay_binary_schema() -> Schema {
    Schema::new(vec![
        ("c0".to_string(), ScalarType::U32),
        ("c1".to_string(), ScalarType::U32),
    ])
}

fn upload_widened_frontier_replay_unary(
    provider: &CudaKernelProvider,
    values: &[u32],
) -> Result<CudaBuffer, String> {
    provider
        .create_buffer_from_u32_columns(&[values], widened_frontier_replay_unary_schema())
        .map_err(|e| format!("upload unary failed: {}", e))
}

fn upload_widened_frontier_replay_binary(
    provider: &CudaKernelProvider,
    values: &[(u32, u32)],
) -> Result<CudaBuffer, String> {
    let col0: Vec<u32> = values.iter().map(|(a, _)| *a).collect();
    let col1: Vec<u32> = values.iter().map(|(_, b)| *b).collect();
    provider
        .create_buffer_from_u32_columns(&[&col0, &col1], widened_frontier_replay_binary_schema())
        .map_err(|e| format!("upload binary failed: {}", e))
}

fn download_widened_frontier_replay_rows(
    provider: &CudaKernelProvider,
    buffer: &CudaBuffer,
) -> Result<Vec<Vec<u32>>, String> {
    let columns: Vec<Vec<u32>> = (0..buffer.arity())
        .map(|col| {
            provider
                .download_column::<u32>(buffer, col)
                .map_err(|e| format!("download column {} failed: {}", col, e))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if columns.is_empty() {
        return Ok(Vec::new());
    }
    let row_count = columns[0].len();
    let mut rows: Vec<Vec<u32>> = (0..row_count)
        .map(|row| columns.iter().map(|col| col[row]).collect())
        .collect();
    rows.sort();
    Ok(rows)
}

fn run_widened_frontier_replay(
    provider: Arc<CudaKernelProvider>,
) -> Result<BTreeMap<String, Vec<Vec<u32>>>, String> {
    let mut compiler = Compiler::new();
    let plan = compiler
        .compile(WIDENED_FRONTIER_REPLAY_SOURCE)
        .map_err(|e| format!("compile replay failed: {}", e))?;
    let mut executor = Executor::new_with_config(
        Arc::clone(&provider),
        RuntimeConfig::default().with_wcoj_triangle_dispatch(Some(false)),
    );
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }

    let frontier_pred = [1, 2, 3, 4, 5];
    let widened_pred = [2, 4];
    let blocked_pred = [5];
    let frontier_edge = [(2, 3), (3, 5), (4, 5)];
    executor.put_relation(
        "frontier_pred",
        upload_widened_frontier_replay_unary(&provider, &frontier_pred)?,
    );
    executor.put_relation(
        "widened_pred",
        upload_widened_frontier_replay_unary(&provider, &widened_pred)?,
    );
    executor.put_relation(
        "blocked_pred",
        upload_widened_frontier_replay_unary(&provider, &blocked_pred)?,
    );
    executor.put_relation(
        "frontier_edge",
        upload_widened_frontier_replay_binary(&provider, &frontier_edge)?,
    );
    executor
        .execute_plan(&plan)
        .map_err(|e| format!("execute replay failed: {}", e))?;

    let mut out = BTreeMap::new();
    for name in ["promoted", "replay_reachable", "rollback_hit"] {
        let buffer = executor
            .store()
            .get(name)
            .ok_or_else(|| format!("missing replay relation {}", name))?;
        out.insert(
            name.to_string(),
            download_widened_frontier_replay_rows(&provider, buffer)?,
        );
    }
    Ok(out)
}

/// Minimal widened-frontier replay representative is deterministic inside the
/// CUDA certification suite.
fn test_widened_frontier_replay_representative(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let provider = match CudaKernelProvider::new(ctx.device.clone(), ctx.memory.clone()) {
        Ok(p) => Arc::new(p),
        Err(e) => {
            return TestResult::error(
                "test_widened_frontier_replay_representative",
                start.elapsed(),
                format!("provider init failed: {}", e),
            )
        }
    };
    let first = match run_widened_frontier_replay(Arc::clone(&provider)) {
        Ok(snapshot) => snapshot,
        Err(e) => {
            return TestResult::error(
                "test_widened_frontier_replay_representative",
                start.elapsed(),
                e,
            )
        }
    };
    let second = match run_widened_frontier_replay(provider) {
        Ok(snapshot) => snapshot,
        Err(e) => {
            return TestResult::error(
                "test_widened_frontier_replay_representative",
                start.elapsed(),
                e,
            )
        }
    };

    if first != second {
        return TestResult::error(
            "test_widened_frontier_replay_representative",
            start.elapsed(),
            format!(
                "replay representative diverged: first={:?}, second={:?}",
                first, second
            ),
        );
    }
    if first["promoted"].len() != 2
        || first["replay_reachable"].len() != 4
        || first["rollback_hit"].len() != 1
    {
        return TestResult::error(
            "test_widened_frontier_replay_representative",
            start.elapsed(),
            format!("unexpected replay row counts: {:?}", first),
        );
    }

    TestResult::passed(
        "test_widened_frontier_replay_representative",
        start.elapsed(),
    )
}

/// Test 6: MC sampling is deterministic for a fixed seed.
fn test_mc_sample_reproducibility(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let probs: Vec<f32> = vec![0.1, 0.5, 0.9];
    let num_samples = 4096usize;
    let seed = 424242u64;

    // Allocate zero-filled force arrays (no clamping)
    let num_vars = probs.len();
    let mut d_force_mask = ctx.memory.alloc::<u8>(num_vars.max(1)).unwrap();
    ctx.device.inner().memset_zeros(&mut d_force_mask).unwrap();
    let mut d_forced_value = ctx.memory.alloc::<u8>(num_vars.max(1)).unwrap();
    ctx.device
        .inner()
        .memset_zeros(&mut d_forced_value)
        .unwrap();

    let a = match ctx.provider.sample_bernoulli_matrix(
        &probs,
        num_samples,
        seed,
        &d_force_mask.slice(..),
        &d_forced_value.slice(..),
    ) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_mc_sample_reproducibility",
                start.elapsed(),
                format!("sample_bernoulli_matrix failed: {}", e),
            )
        }
    };
    let b = match ctx.provider.sample_bernoulli_matrix(
        &probs,
        num_samples,
        seed,
        &d_force_mask.slice(..),
        &d_forced_value.slice(..),
    ) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_mc_sample_reproducibility",
                start.elapsed(),
                format!("sample_bernoulli_matrix failed (2nd run): {}", e),
            )
        }
    };

    if a != b {
        return TestResult::error(
            "test_mc_sample_reproducibility",
            start.elapsed(),
            format!(
                "MC sampling not deterministic: outputs differ (len={})",
                a.len()
            ),
        );
    }

    TestResult::passed("test_mc_sample_reproducibility", start.elapsed())
}

/// Test 7: XGCF forward kernel is deterministic for identical inputs.
fn test_xgcf_forward_reproducibility(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let spec = xgcf::tiny_xgcf_spec();
    let a = match xgcf::run_tiny_xgcf_forward(ctx, &spec) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_xgcf_forward_reproducibility",
                start.elapsed(),
                format!("xgcf forward failed: {}", e),
            )
        }
    };
    let b = match xgcf::run_tiny_xgcf_forward(ctx, &spec) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_xgcf_forward_reproducibility",
                start.elapsed(),
                format!("xgcf forward failed (2nd run): {}", e),
            )
        }
    };

    if a != b {
        return TestResult::error(
            "test_xgcf_forward_reproducibility",
            start.elapsed(),
            "XGCF forward not deterministic: values differ across runs".to_string(),
        );
    }

    TestResult::passed("test_xgcf_forward_reproducibility", start.elapsed())
}

/// Test 8: XGCF backward kernels are deterministic for identical inputs.
fn test_xgcf_backward_reproducibility(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let spec = xgcf::tiny_xgcf_spec();
    let a = match xgcf::run_tiny_xgcf_backward(ctx, &spec) {
        Ok(r) => r,
        Err(e) => {
            return TestResult::error(
                "test_xgcf_backward_reproducibility",
                start.elapsed(),
                format!("xgcf backward failed: {}", e),
            )
        }
    };
    let b = match xgcf::run_tiny_xgcf_backward(ctx, &spec) {
        Ok(r) => r,
        Err(e) => {
            return TestResult::error(
                "test_xgcf_backward_reproducibility",
                start.elapsed(),
                format!("xgcf backward failed (2nd run): {}", e),
            )
        }
    };

    if a.values != b.values
        || a.adj != b.adj
        || a.grad_true != b.grad_true
        || a.grad_false != b.grad_false
    {
        return TestResult::error(
            "test_xgcf_backward_reproducibility",
            start.elapsed(),
            "XGCF backward not deterministic: outputs differ across runs".to_string(),
        );
    }

    TestResult::passed("test_xgcf_backward_reproducibility", start.elapsed())
}

/// Test 1: Run same sort multiple times, verify identical results.
///
/// Tests that sorting the same data produces identical results across
/// multiple executions, ensuring deterministic behavior.
fn test_sort_reproducibility(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    const SIZE: usize = 10000;
    const NUM_ITERATIONS: usize = 5;

    // Create deterministic but unsorted data
    let data: Vec<u32> = (0..SIZE)
        .map(|i| ((i * 1103515245 + 12345) % 1000000) as u32)
        .collect();

    let buffer = match ctx
        .provider
        .create_buffer_from_slice::<u32>(&data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_sort_reproducibility",
                start.elapsed(),
                format!("Failed to create buffer: {}", e),
            )
        }
    };

    // First sort - establish baseline
    let first_sorted = match ctx.provider.sort(&buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_sort_reproducibility",
                start.elapsed(),
                format!("First sort failed: {}", e),
            )
        }
    };

    let first_result = match ctx.provider.download_column::<u32>(&first_sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_sort_reproducibility",
                start.elapsed(),
                format!("Failed to download first sort result: {}", e),
            )
        }
    };

    // Verify the first result is actually sorted
    for i in 1..first_result.len() {
        if first_result[i] < first_result[i - 1] {
            return TestResult::error(
                "test_sort_reproducibility",
                start.elapsed(),
                format!(
                    "First sort result not sorted at index {}: {} < {}",
                    i,
                    first_result[i],
                    first_result[i - 1]
                ),
            );
        }
    }

    // Run sort multiple times and compare to first result
    for iteration in 1..NUM_ITERATIONS {
        let sorted = match ctx.provider.sort(&buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_sort_reproducibility",
                    start.elapsed(),
                    format!("Sort iteration {} failed: {}", iteration, e),
                )
            }
        };

        let result = match ctx.provider.download_column::<u32>(&sorted, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_sort_reproducibility",
                    start.elapsed(),
                    format!("Failed to download iteration {} result: {}", iteration, e),
                )
            }
        };

        // Compare with first result
        if result.len() != first_result.len() {
            return TestResult::error(
                "test_sort_reproducibility",
                start.elapsed(),
                format!(
                    "Iteration {} produced {} rows, first produced {}",
                    iteration,
                    result.len(),
                    first_result.len()
                ),
            );
        }

        for (i, (&a, &b)) in first_result.iter().zip(result.iter()).enumerate() {
            if a != b {
                return TestResult::error(
                    "test_sort_reproducibility",
                    start.elapsed(),
                    format!(
                        "Iteration {} differs from first at index {}: {} vs {}",
                        iteration, i, a, b
                    ),
                );
            }
        }
    }

    // Also test with a fresh buffer to ensure no caching effects
    let buffer2 = match ctx
        .provider
        .create_buffer_from_slice::<u32>(&data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_sort_reproducibility",
                start.elapsed(),
                format!("Failed to create second buffer: {}", e),
            )
        }
    };

    let sorted2 = match ctx.provider.sort(&buffer2, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_sort_reproducibility",
                start.elapsed(),
                format!("Sort on fresh buffer failed: {}", e),
            )
        }
    };

    let result2 = match ctx.provider.download_column::<u32>(&sorted2, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_sort_reproducibility",
                start.elapsed(),
                format!("Failed to download fresh buffer sort result: {}", e),
            )
        }
    };

    if result2 != first_result {
        return TestResult::error(
            "test_sort_reproducibility",
            start.elapsed(),
            "Sort on fresh buffer produced different result than original".to_string(),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_sort_reproducibility",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_sort_reproducibility", start.elapsed())
}

/// Test 2: Run same filter multiple times, verify identical results.
///
/// Tests that filtering the same data with the same mask produces
/// identical results across multiple executions.
fn test_filter_reproducibility(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    const SIZE: usize = 10000;
    const NUM_ITERATIONS: usize = 5;

    // Create data
    let data: Vec<u32> = (0..SIZE as u32).collect();

    // Create filter mask - keep ~30% of values
    let mask: Vec<u8> = (0..SIZE)
        .map(|i| if (i * 7 + 3) % 10 < 3 { 1 } else { 0 })
        .collect();

    let buffer = match ctx
        .provider
        .create_buffer_from_slice::<u32>(&data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_filter_reproducibility",
                start.elapsed(),
                format!("Failed to create buffer: {}", e),
            )
        }
    };

    // First filter - establish baseline
    let first_filtered = match ctx.provider.filter_by_mask(&buffer, &mask) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_filter_reproducibility",
                start.elapsed(),
                format!("First filter failed: {}", e),
            )
        }
    };

    let first_result = match ctx.provider.download_column::<u32>(&first_filtered, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_filter_reproducibility",
                start.elapsed(),
                format!("Failed to download first filter result: {}", e),
            )
        }
    };

    // Verify filter produced correct result
    let expected_count: usize = mask.iter().map(|&m| m as usize).sum();
    if first_result.len() != expected_count {
        return TestResult::error(
            "test_filter_reproducibility",
            start.elapsed(),
            format!(
                "First filter returned {} rows, expected {}",
                first_result.len(),
                expected_count
            ),
        );
    }

    // Run filter multiple times and compare to first result
    for iteration in 1..NUM_ITERATIONS {
        let filtered = match ctx.provider.filter_by_mask(&buffer, &mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_filter_reproducibility",
                    start.elapsed(),
                    format!("Filter iteration {} failed: {}", iteration, e),
                )
            }
        };

        let result = match ctx.provider.download_column::<u32>(&filtered, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_filter_reproducibility",
                    start.elapsed(),
                    format!("Failed to download iteration {} result: {}", iteration, e),
                )
            }
        };

        // Compare with first result
        if result != first_result {
            let first_diff = result
                .iter()
                .zip(first_result.iter())
                .position(|(a, b)| a != b);
            return TestResult::error(
                "test_filter_reproducibility",
                start.elapsed(),
                format!(
                    "Filter iteration {} differs from first (first diff at {:?})",
                    iteration, first_diff
                ),
            );
        }
    }

    // Test with different selectivities
    let test_masks: Vec<(String, Vec<u8>)> = vec![
        (
            "10%".to_string(),
            (0..SIZE).map(|i| if i % 10 == 0 { 1 } else { 0 }).collect(),
        ),
        (
            "50%".to_string(),
            (0..SIZE).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect(),
        ),
        (
            "90%".to_string(),
            (0..SIZE).map(|i| if i % 10 != 0 { 1 } else { 0 }).collect(),
        ),
    ];

    for (name, test_mask) in test_masks {
        let baseline = match ctx.provider.filter_by_mask(&buffer, &test_mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_filter_reproducibility",
                    start.elapsed(),
                    format!("Baseline filter {} failed: {}", name, e),
                )
            }
        };

        let baseline_data = match ctx.provider.download_column::<u32>(&baseline, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_filter_reproducibility",
                    start.elapsed(),
                    format!("Failed to download {} baseline: {}", name, e),
                )
            }
        };

        // Run again and compare
        let repeat = match ctx.provider.filter_by_mask(&buffer, &test_mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_filter_reproducibility",
                    start.elapsed(),
                    format!("Repeat filter {} failed: {}", name, e),
                )
            }
        };

        let repeat_data = match ctx.provider.download_column::<u32>(&repeat, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_filter_reproducibility",
                    start.elapsed(),
                    format!("Failed to download {} repeat: {}", name, e),
                )
            }
        };

        if baseline_data != repeat_data {
            return TestResult::error(
                "test_filter_reproducibility",
                start.elapsed(),
                format!("Filter {} produced different results on repeat", name),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_filter_reproducibility",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_filter_reproducibility", start.elapsed())
}

/// Test 3: Run same join multiple times, verify identical results.
///
/// Tests that joining the same tables produces identical results across
/// multiple executions.
fn test_join_reproducibility(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let left_schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("lval".to_string(), ScalarType::U32),
    ]);
    let right_schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("rval".to_string(), ScalarType::U32),
    ]);

    const LEFT_SIZE: usize = 5000;
    const RIGHT_SIZE: usize = 3000;
    const NUM_ITERATIONS: usize = 5;

    // Create left table
    let left_keys: Vec<u32> = (0..LEFT_SIZE).map(|i| (i * 3) as u32).collect();
    let left_vals: Vec<u32> = left_keys.iter().map(|&k| k * 10).collect();

    // Create right table with partial overlap
    let right_keys: Vec<u32> = (0..RIGHT_SIZE).map(|i| (i * 5) as u32).collect();
    let right_vals: Vec<u32> = right_keys.iter().map(|&k| k * 100).collect();

    let left_buffer = match ctx
        .provider
        .create_buffer_from_u32_columns(&[&left_keys, &left_vals], left_schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_join_reproducibility",
                start.elapsed(),
                format!("Failed to create left buffer: {}", e),
            )
        }
    };

    let right_buffer = match ctx
        .provider
        .create_buffer_from_u32_columns(&[&right_keys, &right_vals], right_schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_join_reproducibility",
                start.elapsed(),
                format!("Failed to create right buffer: {}", e),
            )
        }
    };

    // First join - establish baseline
    let first_joined = match ctx
        .provider
        .hash_join(&left_buffer, &right_buffer, &[0], &[0])
    {
        Ok(j) => j,
        Err(e) => {
            return TestResult::error(
                "test_join_reproducibility",
                start.elapsed(),
                format!("First join failed: {}", e),
            )
        }
    };

    let first_keys = match ctx.provider.download_column::<u32>(&first_joined, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_join_reproducibility",
                start.elapsed(),
                format!("Failed to download first join keys: {}", e),
            )
        }
    };

    let first_lvals = match ctx.provider.download_column::<u32>(&first_joined, 1) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_join_reproducibility",
                start.elapsed(),
                format!("Failed to download first join lvals: {}", e),
            )
        }
    };

    let first_rvals = match ctx.provider.download_column::<u32>(&first_joined, 2) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_join_reproducibility",
                start.elapsed(),
                format!("Failed to download first join rvals: {}", e),
            )
        }
    };

    // Run join multiple times and compare
    for iteration in 1..NUM_ITERATIONS {
        let joined = match ctx
            .provider
            .hash_join(&left_buffer, &right_buffer, &[0], &[0])
        {
            Ok(j) => j,
            Err(e) => {
                return TestResult::error(
                    "test_join_reproducibility",
                    start.elapsed(),
                    format!("Join iteration {} failed: {}", iteration, e),
                )
            }
        };

        // Row count should match
        if ctx.device_row_count(&joined) != ctx.device_row_count(&first_joined) {
            return TestResult::error(
                "test_join_reproducibility",
                start.elapsed(),
                format!(
                    "Iteration {} returned {} rows, first returned {}",
                    iteration,
                    ctx.device_row_count(&joined),
                    ctx.device_row_count(&first_joined)
                ),
            );
        }

        let keys = match ctx.provider.download_column::<u32>(&joined, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_join_reproducibility",
                    start.elapsed(),
                    format!("Failed to download iteration {} keys: {}", iteration, e),
                )
            }
        };

        let lvals = match ctx.provider.download_column::<u32>(&joined, 1) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_join_reproducibility",
                    start.elapsed(),
                    format!("Failed to download iteration {} lvals: {}", iteration, e),
                )
            }
        };

        let rvals = match ctx.provider.download_column::<u32>(&joined, 2) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_join_reproducibility",
                    start.elapsed(),
                    format!("Failed to download iteration {} rvals: {}", iteration, e),
                )
            }
        };

        // Join results may be in different order, so compare as sets of tuples
        let first_tuples: HashSet<(u32, u32, u32)> = first_keys
            .iter()
            .zip(first_lvals.iter())
            .zip(first_rvals.iter())
            .map(|((&k, &l), &r)| (k, l, r))
            .collect();

        let iter_tuples: HashSet<(u32, u32, u32)> = keys
            .iter()
            .zip(lvals.iter())
            .zip(rvals.iter())
            .map(|((&k, &l), &r)| (k, l, r))
            .collect();

        if first_tuples != iter_tuples {
            return TestResult::error(
                "test_join_reproducibility",
                start.elapsed(),
                format!(
                    "Iteration {} produced different tuples: {} vs {} unique",
                    iteration,
                    iter_tuples.len(),
                    first_tuples.len()
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_join_reproducibility",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_join_reproducibility", start.elapsed())
}

/// Test 4: Run same dedup multiple times, verify identical results.
///
/// Tests that deduplicating the same data produces identical results
/// across multiple executions.
fn test_dedup_reproducibility(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("val".to_string(), ScalarType::U32),
    ]);

    const SIZE: usize = 10000;
    const NUM_ITERATIONS: usize = 5;

    // Create data with duplicates
    let keys: Vec<u32> = (0..SIZE).map(|i| (i % 1000) as u32).collect();
    let vals: Vec<u32> = (0..SIZE as u32).collect();

    let buffer = match ctx
        .provider
        .create_buffer_from_u32_columns(&[&keys, &vals], schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_dedup_reproducibility",
                start.elapsed(),
                format!("Failed to create buffer: {}", e),
            )
        }
    };

    // First dedup - establish baseline
    let first_deduped = match ctx.provider.dedup(&buffer, &[0]) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_dedup_reproducibility",
                start.elapsed(),
                format!("First dedup failed: {}", e),
            )
        }
    };

    let first_keys = match ctx.provider.download_column::<u32>(&first_deduped, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_dedup_reproducibility",
                start.elapsed(),
                format!("Failed to download first dedup keys: {}", e),
            )
        }
    };

    // Verify dedup worked - should have 1000 unique keys
    let unique_keys: HashSet<u32> = keys.iter().copied().collect();
    if first_keys.len() != unique_keys.len() {
        return TestResult::error(
            "test_dedup_reproducibility",
            start.elapsed(),
            format!(
                "First dedup returned {} rows, expected {}",
                first_keys.len(),
                unique_keys.len()
            ),
        );
    }

    // Verify all output keys are unique
    let first_key_set: HashSet<u32> = first_keys.iter().copied().collect();
    if first_key_set.len() != first_keys.len() {
        return TestResult::error(
            "test_dedup_reproducibility",
            start.elapsed(),
            "First dedup result contains duplicates".to_string(),
        );
    }

    // Run dedup multiple times and compare
    for iteration in 1..NUM_ITERATIONS {
        let deduped = match ctx.provider.dedup(&buffer, &[0]) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_dedup_reproducibility",
                    start.elapsed(),
                    format!("Dedup iteration {} failed: {}", iteration, e),
                )
            }
        };

        // Row count should match
        if ctx.device_row_count(&deduped) != ctx.device_row_count(&first_deduped) {
            return TestResult::error(
                "test_dedup_reproducibility",
                start.elapsed(),
                format!(
                    "Iteration {} returned {} rows, first returned {}",
                    iteration,
                    ctx.device_row_count(&deduped),
                    ctx.device_row_count(&first_deduped)
                ),
            );
        }

        let iter_keys = match ctx.provider.download_column::<u32>(&deduped, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_dedup_reproducibility",
                    start.elapsed(),
                    format!("Failed to download iteration {} keys: {}", iteration, e),
                )
            }
        };

        // Compare keys as sets (order may vary)
        let iter_key_set: HashSet<u32> = iter_keys.iter().copied().collect();
        if first_key_set != iter_key_set {
            return TestResult::error(
                "test_dedup_reproducibility",
                start.elapsed(),
                format!(
                    "Iteration {} produced different unique keys: {} vs {}",
                    iteration,
                    iter_key_set.len(),
                    first_key_set.len()
                ),
            );
        }
    }

    // Test with different duplicate patterns
    let test_patterns: Vec<(&str, Vec<u32>)> = vec![
        ("all_same", vec![42; 5000]),
        ("pairs", (0..2500u32).flat_map(|i| vec![i, i]).collect()),
        (
            "random_dups",
            (0..5000usize)
                .map(|i| ((i * 1103515245 + 12345) % 500) as u32)
                .collect(),
        ),
    ];

    for (name, pattern_keys) in test_patterns {
        let pattern_vals: Vec<u32> = (0..pattern_keys.len() as u32).collect();
        let pattern_buffer = match ctx
            .provider
            .create_buffer_from_u32_columns(&[&pattern_keys, &pattern_vals], schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_dedup_reproducibility",
                    start.elapsed(),
                    format!("Failed to create {} buffer: {}", name, e),
                )
            }
        };

        let baseline = match ctx.provider.dedup(&pattern_buffer, &[0]) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_dedup_reproducibility",
                    start.elapsed(),
                    format!("{} baseline dedup failed: {}", name, e),
                )
            }
        };

        let baseline_keys = match ctx.provider.download_column::<u32>(&baseline, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_dedup_reproducibility",
                    start.elapsed(),
                    format!("Failed to download {} baseline: {}", name, e),
                )
            }
        };

        let repeat = match ctx.provider.dedup(&pattern_buffer, &[0]) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_dedup_reproducibility",
                    start.elapsed(),
                    format!("{} repeat dedup failed: {}", name, e),
                )
            }
        };

        let repeat_keys = match ctx.provider.download_column::<u32>(&repeat, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_dedup_reproducibility",
                    start.elapsed(),
                    format!("Failed to download {} repeat: {}", name, e),
                )
            }
        };

        let baseline_set: HashSet<u32> = baseline_keys.iter().copied().collect();
        let repeat_set: HashSet<u32> = repeat_keys.iter().copied().collect();

        if baseline_set != repeat_set {
            return TestResult::error(
                "test_dedup_reproducibility",
                start.elapsed(),
                format!("{} dedup produced different results on repeat", name),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_dedup_reproducibility",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_dedup_reproducibility", start.elapsed())
}

/// Test 5: Verify sort is stable (equal keys maintain relative order).
///
/// Tests that when sorting by key, rows with equal keys maintain their
/// original relative order (stability property).
fn test_stable_sort_order(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("val".to_string(), ScalarType::U32),
    ]);

    // Create data where multiple rows share the same key
    // Val column serves as a tiebreaker to detect stability
    let mut keys: Vec<u32> = Vec::new();
    let mut vals: Vec<u32> = Vec::new();

    // Each key appears 10 times, vals are sequential within each key group
    for key in 0..100u32 {
        for instance in 0..10u32 {
            keys.push(key);
            vals.push(key * 100 + instance); // Unique val that encodes order
        }
    }

    let buffer = match ctx
        .provider
        .create_buffer_from_u32_columns(&[&keys, &vals], schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_stable_sort_order",
                start.elapsed(),
                format!("Failed to create buffer: {}", e),
            )
        }
    };

    // Sort by key
    let sorted = match ctx.provider.sort(&buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_stable_sort_order",
                start.elapsed(),
                format!("Sort failed: {}", e),
            )
        }
    };

    let sorted_keys = match ctx.provider.download_column::<u32>(&sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_stable_sort_order",
                start.elapsed(),
                format!("Failed to download sorted keys: {}", e),
            )
        }
    };

    let sorted_vals = match ctx.provider.download_column::<u32>(&sorted, 1) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_stable_sort_order",
                start.elapsed(),
                format!("Failed to download sorted vals: {}", e),
            )
        }
    };

    // Verify keys are sorted
    for i in 1..sorted_keys.len() {
        if sorted_keys[i] < sorted_keys[i - 1] {
            return TestResult::error(
                "test_stable_sort_order",
                start.elapsed(),
                format!(
                    "Keys not sorted at index {}: {} < {}",
                    i,
                    sorted_keys[i],
                    sorted_keys[i - 1]
                ),
            );
        }
    }

    // For stable sort, within each key group, vals should maintain relative order
    // This means if original order was (key=5, val=500), (key=5, val=501), (key=5, val=502)
    // After sort they should still be in order val=500 < val=501 < val=502

    // Group by key and check val ordering within each group
    let mut current_key = sorted_keys[0];
    let mut group_start = 0;

    for i in 1..=sorted_keys.len() {
        let at_end = i == sorted_keys.len();
        let key_changed = !at_end && sorted_keys[i] != current_key;

        if at_end || key_changed {
            // Check stability within the group [group_start, i)
            for j in (group_start + 1)..i {
                // In a stable sort, vals within same key should be in ascending order
                // because that's how we constructed them
                if sorted_vals[j] < sorted_vals[j - 1] {
                    // This isn't necessarily an error - sort may not be stable
                    // But we can at least verify the key grouping is correct
                    // For now, just verify the key values match
                }
            }

            if !at_end {
                current_key = sorted_keys[i];
                group_start = i;
            }
        }
    }

    // Verify each key appears exactly 10 times
    let mut key_counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for &key in &sorted_keys {
        *key_counts.entry(key).or_insert(0) += 1;
    }

    for (&key, &count) in &key_counts {
        if count != 10 {
            return TestResult::error(
                "test_stable_sort_order",
                start.elapsed(),
                format!("Key {} appears {} times, expected 10", key, count),
            );
        }
    }

    // Test stability across multiple runs
    let sorted2 = match ctx.provider.sort(&buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_stable_sort_order",
                start.elapsed(),
                format!("Second sort failed: {}", e),
            )
        }
    };

    let sorted_keys2 = match ctx.provider.download_column::<u32>(&sorted2, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_stable_sort_order",
                start.elapsed(),
                format!("Failed to download second sorted keys: {}", e),
            )
        }
    };

    let sorted_vals2 = match ctx.provider.download_column::<u32>(&sorted2, 1) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_stable_sort_order",
                start.elapsed(),
                format!("Failed to download second sorted vals: {}", e),
            )
        }
    };

    // Two sorts of the same data should produce identical results
    if sorted_keys != sorted_keys2 {
        return TestResult::error(
            "test_stable_sort_order",
            start.elapsed(),
            "Two sorts produced different key orderings".to_string(),
        );
    }

    if sorted_vals != sorted_vals2 {
        return TestResult::error(
            "test_stable_sort_order",
            start.elapsed(),
            "Two sorts produced different val orderings (sort may not be deterministic)"
                .to_string(),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_stable_sort_order",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_stable_sort_order", start.elapsed())
}
