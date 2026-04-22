//! Category G08: Device count integrity tests.
//!
//! Validates device-resident row counts for compaction and groupby outputs.

use crate::harness::{CategoryResult, TestContext, TestResult};
use std::time::Instant;
use xlog_core::{AggOp, ScalarType, Schema};
use xlog_cuda::CudaBuffer;

/// Run all tests in this category.
pub fn run_all(ctx: &TestContext) -> CategoryResult {
    let start = Instant::now();
    let mut results = CategoryResult::new("g08_device_counts");

    results.add_result(test_device_compact_count(ctx));
    results.add_result(test_groupby_device_count_sum(ctx));

    results.set_duration(start.elapsed());
    results
}

fn read_device_count(ctx: &TestContext, buffer: &CudaBuffer) -> Result<u32, String> {
    let mut host_count = [0u32];
    ctx.device
        .inner()
        .dtoh_sync_copy_into(buffer.num_rows_device(), &mut host_count)
        .map_err(|e| format!("Failed to read device row count: {}", e))?;
    Ok(host_count[0])
}

fn test_device_compact_count(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let schema = Schema::new(vec![("id".to_string(), ScalarType::U32)]);
    let data: Vec<u32> = (0..32u32).collect();
    let buffer = match ctx.provider.create_buffer_from_slice::<u32>(&data, schema) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_device_compact_count",
                start.elapsed(),
                format!("Failed to create buffer: {}", e),
            )
        }
    };

    // Keep every 3rd element.
    let mask: Vec<u8> = (0..data.len())
        .map(|i| if i % 3 == 0 { 1 } else { 0 })
        .collect();
    let expected_count = mask.iter().map(|&m| m as u32).sum::<u32>();
    let expected_values: Vec<u32> = data
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if i % 3 == 0 { Some(v) } else { None })
        .collect();

    let d_mask = match ctx.device.inner().htod_sync_copy(&mask) {
        Ok(m) => m,
        Err(e) => {
            return TestResult::error(
                "test_device_compact_count",
                start.elapsed(),
                format!("Failed to upload mask: {}", e),
            )
        }
    };

    let filtered = match ctx.provider.filter_by_device_mask(&buffer, &d_mask) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_device_compact_count",
                start.elapsed(),
                format!("filter_by_device_mask failed: {}", e),
            )
        }
    };

    let device_count = match read_device_count(ctx, &filtered) {
        Ok(count) => count,
        Err(e) => return TestResult::error("test_device_compact_count", start.elapsed(), e),
    };

    if device_count != expected_count {
        return TestResult::error(
            "test_device_compact_count",
            start.elapsed(),
            format!(
                "Device count mismatch: got {}, expected {}",
                device_count, expected_count
            ),
        );
    }

    let filtered_values = match ctx.provider.download_column::<u32>(&filtered, 0) {
        Ok(values) => values,
        Err(e) => {
            return TestResult::error(
                "test_device_compact_count",
                start.elapsed(),
                format!("Failed to download filtered data: {}", e),
            )
        }
    };

    if filtered_values != expected_values {
        return TestResult::error(
            "test_device_compact_count",
            start.elapsed(),
            format!(
                "Filtered values mismatch: got {:?}, expected {:?}",
                filtered_values, expected_values
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_device_compact_count",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_device_compact_count", start.elapsed())
}

fn test_groupby_device_count_sum(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("val".to_string(), ScalarType::U32),
    ]);
    let keys: Vec<u32> = vec![1, 1, 2, 2, 2, 3];
    let vals: Vec<u32> = vec![10, 20, 1, 1, 1, 5];

    let buffer = match ctx
        .provider
        .create_buffer_from_u32_columns(&[&keys, &vals], schema)
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_groupby_device_count_sum",
                start.elapsed(),
                format!("Failed to create groupby buffer: {}", e),
            )
        }
    };

    let grouped = match ctx
        .provider
        .groupby_multi_agg(&buffer, &[0], &[(1, AggOp::Sum)])
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_groupby_device_count_sum",
                start.elapsed(),
                format!("groupby_multi_agg failed: {}", e),
            )
        }
    };

    let device_count = match read_device_count(ctx, &grouped) {
        Ok(count) => count,
        Err(e) => return TestResult::error("test_groupby_device_count_sum", start.elapsed(), e),
    };

    if device_count != 3 {
        return TestResult::error(
            "test_groupby_device_count_sum",
            start.elapsed(),
            format!(
                "Device group count mismatch: got {}, expected 3",
                device_count
            ),
        );
    }

    let out_keys = match ctx.provider.download_column::<u32>(&grouped, 0) {
        Ok(values) => values,
        Err(e) => {
            return TestResult::error(
                "test_groupby_device_count_sum",
                start.elapsed(),
                format!("Failed to download group keys: {}", e),
            )
        }
    };

    let out_sums = match ctx.provider.download_column::<u64>(&grouped, 1) {
        Ok(values) => values,
        Err(e) => {
            return TestResult::error(
                "test_groupby_device_count_sum",
                start.elapsed(),
                format!("Failed to download group sums: {}", e),
            )
        }
    };

    if out_keys.len() != out_sums.len() {
        return TestResult::error(
            "test_groupby_device_count_sum",
            start.elapsed(),
            format!(
                "Group output length mismatch: keys={}, sums={}",
                out_keys.len(),
                out_sums.len()
            ),
        );
    }

    let mut pairs: Vec<(u32, u64)> = out_keys.into_iter().zip(out_sums.into_iter()).collect();
    pairs.sort_by_key(|(k, _)| *k);

    let expected = vec![(1u32, 30u64), (2u32, 3u64), (3u32, 5u64)];
    if pairs != expected {
        return TestResult::error(
            "test_groupby_device_count_sum",
            start.elapsed(),
            format!(
                "Group sums mismatch: got {:?}, expected {:?}",
                pairs, expected
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_groupby_device_count_sum",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_groupby_device_count_sum", start.elapsed())
}
