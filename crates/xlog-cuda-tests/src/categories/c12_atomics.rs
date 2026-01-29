//! Category 12: Atomic Operations
//!
//! Tests atomic operation correctness including hash join atomics, dedup atomics,
//! high contention scenarios, atomic counting, and concurrent atomic updates.

use crate::harness::{CategoryResult, TestContext, TestResult};
use std::collections::HashSet;
use std::time::Instant;
use xlog_core::{ScalarType, Schema};

/// Run all tests in this category.
pub fn run_all(ctx: &TestContext) -> CategoryResult {
    let mut results = CategoryResult::new("c12_atomics");
    let start = Instant::now();

    results.add_result(test_hash_join_atomic_correctness(ctx));
    results.add_result(test_dedup_atomic_correctness(ctx));
    results.add_result(test_high_contention_join(ctx));
    results.add_result(test_atomic_counting(ctx));
    results.add_result(test_concurrent_atomic_updates(ctx));

    results.set_duration(start.elapsed());
    results
}

/// Test 1: Hash join uses atomic hash table operations - verify correctness.
///
/// Hash join builds a hash table using atomic operations to handle collisions.
/// This test verifies that concurrent atomic insertions produce correct results.
fn test_hash_join_atomic_correctness(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let left_schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("lval".to_string(), ScalarType::U32),
    ]);
    let right_schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("rval".to_string(), ScalarType::U32),
    ]);

    // Test with various sizes to exercise atomic contention
    let test_cases: Vec<(usize, usize, f64)> = vec![
        (1000, 500, 0.5),   // 50% match rate
        (5000, 2500, 0.5),  // 50% match rate, larger
        (10000, 1000, 1.0), // All right keys match
        (10000, 5000, 0.8), // 80% match rate
    ];

    for (left_size, right_size, match_rate) in test_cases {
        // Create left table with sequential keys
        let left_keys: Vec<u32> = (0..left_size as u32).collect();
        let left_vals: Vec<u32> = left_keys.iter().map(|&k| k * 10).collect();

        // Create right table with keys that have 'match_rate' overlap
        let matching_right = (right_size as f64 * match_rate) as usize;
        let mut right_keys: Vec<u32> = Vec::with_capacity(right_size);
        let mut right_vals: Vec<u32> = Vec::with_capacity(right_size);

        // First portion: matching keys
        for i in 0..matching_right {
            let key = (i * left_size / matching_right.max(1)) as u32;
            right_keys.push(key);
            right_vals.push(key * 100);
        }
        // Remaining: non-matching keys
        for i in matching_right..right_size {
            let key = (left_size as u32) + (i as u32);
            right_keys.push(key);
            right_vals.push(key * 100);
        }

        let left_buffer = match ctx
            .provider
            .create_buffer_from_u32_columns(&[&left_keys, &left_vals], left_schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_hash_join_atomic_correctness",
                    start.elapsed(),
                    format!("Left {}: failed to create buffer: {}", left_size, e),
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
                    "test_hash_join_atomic_correctness",
                    start.elapsed(),
                    format!("Right {}: failed to create buffer: {}", right_size, e),
                )
            }
        };

        // Perform hash join
        let joined = match ctx
            .provider
            .hash_join(&left_buffer, &right_buffer, &[0], &[0])
        {
            Ok(j) => j,
            Err(e) => {
                return TestResult::error(
                    "test_hash_join_atomic_correctness",
                    start.elapsed(),
                    format!("Hash join failed ({}x{}): {}", left_size, right_size, e),
                )
            }
        };

        // Verify join count matches expected
        if ctx.device_row_count(&joined) != matching_right as u64 {
            return TestResult::error(
                "test_hash_join_atomic_correctness",
                start.elapsed(),
                format!(
                    "Join ({}x{}) returned {} rows, expected {}",
                    left_size, right_size, ctx.device_row_count(&joined), matching_right
                ),
            );
        }

        // Download and verify join results
        let joined_keys = match ctx.provider.download_column_u32(&joined, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_hash_join_atomic_correctness",
                    start.elapsed(),
                    format!("Failed to download joined keys: {}", e),
                )
            }
        };

        let joined_lvals = match ctx.provider.download_column_u32(&joined, 1) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_hash_join_atomic_correctness",
                    start.elapsed(),
                    format!("Failed to download joined lvals: {}", e),
                )
            }
        };

        let joined_rvals = match ctx.provider.download_column_u32(&joined, 2) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_hash_join_atomic_correctness",
                    start.elapsed(),
                    format!("Failed to download joined rvals: {}", e),
                )
            }
        };

        // Verify each joined row
        for i in 0..ctx.device_row_count(&joined) as usize {
            let key = joined_keys[i];
            let lval = joined_lvals[i];
            let rval = joined_rvals[i];

            // Key must be in left table range
            if key >= left_size as u32 {
                return TestResult::error(
                    "test_hash_join_atomic_correctness",
                    start.elapsed(),
                    format!(
                        "Row {}: key {} is outside left table range [0, {})",
                        i, key, left_size
                    ),
                );
            }

            // lval should be key * 10
            if lval != key * 10 {
                return TestResult::error(
                    "test_hash_join_atomic_correctness",
                    start.elapsed(),
                    format!(
                        "Row {}: lval {} doesn't match expected {} for key {}",
                        i,
                        lval,
                        key * 10,
                        key
                    ),
                );
            }

            // rval should be key * 100
            if rval != key * 100 {
                return TestResult::error(
                    "test_hash_join_atomic_correctness",
                    start.elapsed(),
                    format!(
                        "Row {}: rval {} doesn't match expected {} for key {}",
                        i,
                        rval,
                        key * 100,
                        key
                    ),
                );
            }
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_hash_join_atomic_correctness",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_hash_join_atomic_correctness", start.elapsed())
}

/// Test 2: Dedup uses atomic marking - verify correctness.
///
/// Dedup operations use atomic operations to mark duplicates. This test
/// verifies that concurrent atomic marking produces correct deduplication.
fn test_dedup_atomic_correctness(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("val".to_string(), ScalarType::U32),
    ]);

    // Test various duplicate patterns
    let test_cases: Vec<(&str, Vec<u32>, Vec<u32>)> = vec![
        // (name, keys, vals)
        ("all_same", vec![42; 10000], (0..10000u32).collect()),
        (
            "pairs",
            (0..5000u32).flat_map(|i| vec![i, i]).collect(),
            (0..10000u32).collect(),
        ),
        (
            "random_dups",
            (0..10000usize)
                .map(|i| ((i * 1103515245 + 12345) % 1000) as u32)
                .collect(),
            (0..10000u32).collect(),
        ),
        (
            "clustered_dups",
            (0..10000usize).map(|i| (i / 10) as u32).collect(),
            (0..10000u32).collect(),
        ),
        (
            "sparse_dups",
            (0..10000usize)
                .map(|i| if i % 100 == 0 { 0 } else { i as u32 })
                .collect(),
            (0..10000u32).collect(),
        ),
    ];

    for (name, keys, vals) in test_cases {
        let buffer = match ctx
            .provider
            .create_buffer_from_u32_columns(&[&keys, &vals], schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_dedup_atomic_correctness",
                    start.elapsed(),
                    format!("Pattern {}: failed to create buffer: {}", name, e),
                )
            }
        };

        // Dedup by key column
        let deduped = match ctx.provider.dedup(&buffer, &[0]) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_dedup_atomic_correctness",
                    start.elapsed(),
                    format!("Pattern {}: dedup failed: {}", name, e),
                )
            }
        };

        // Calculate expected unique count
        let unique_keys: HashSet<u32> = keys.iter().cloned().collect();
        let expected_unique = unique_keys.len();

        if ctx.device_row_count(&deduped) != expected_unique as u64 {
            return TestResult::error(
                "test_dedup_atomic_correctness",
                start.elapsed(),
                format!(
                    "Pattern {}: dedup returned {} rows, expected {}",
                    name, ctx.device_row_count(&deduped), expected_unique
                ),
            );
        }

        // Download and verify uniqueness
        let deduped_keys = match ctx.provider.download_column_u32(&deduped, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_dedup_atomic_correctness",
                    start.elapsed(),
                    format!("Pattern {}: failed to download deduped keys: {}", name, e),
                )
            }
        };

        // Verify all keys are unique
        let mut seen_keys: HashSet<u32> = HashSet::new();
        for &k in &deduped_keys {
            if !seen_keys.insert(k) {
                return TestResult::error(
                    "test_dedup_atomic_correctness",
                    start.elapsed(),
                    format!("Pattern {}: duplicate key {} in dedup result", name, k),
                );
            }
        }

        // Verify all original unique keys are present
        if seen_keys != unique_keys {
            return TestResult::error(
                "test_dedup_atomic_correctness",
                start.elapsed(),
                format!("Pattern {}: dedup result missing some unique keys", name),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_dedup_atomic_correctness",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_dedup_atomic_correctness", start.elapsed())
}

/// Test 3: Join with many hash collisions (adversarial keys).
///
/// Adversarial key patterns that cause hash collisions stress atomic
/// contention in the hash table. This test uses keys designed to create
/// maximum collision.
fn test_high_contention_join(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let left_schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("lval".to_string(), ScalarType::U32),
    ]);
    let right_schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("rval".to_string(), ScalarType::U32),
    ]);

    // Pattern 1: Keys that are multiples of power of 2 (common hash table size)
    // These will likely collide if the hash table uses power-of-2 sizing
    const SIZE: usize = 5000;

    let collision_keys: Vec<u32> = (0..SIZE).map(|i| (i * 256) as u32).collect();
    let left_vals: Vec<u32> = collision_keys.iter().map(|&k| k + 1).collect();
    let right_vals: Vec<u32> = collision_keys.iter().map(|&k| k + 2).collect();

    let left_buffer = match ctx
        .provider
        .create_buffer_from_u32_columns(&[&collision_keys, &left_vals], left_schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_high_contention_join",
                start.elapsed(),
                format!("Failed to create left buffer: {}", e),
            )
        }
    };

    let right_buffer = match ctx
        .provider
        .create_buffer_from_u32_columns(&[&collision_keys, &right_vals], right_schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_high_contention_join",
                start.elapsed(),
                format!("Failed to create right buffer: {}", e),
            )
        }
    };

    // Perform hash join with collision-prone keys
    let joined = match ctx
        .provider
        .hash_join(&left_buffer, &right_buffer, &[0], &[0])
    {
        Ok(j) => j,
        Err(e) => {
            return TestResult::error(
                "test_high_contention_join",
                start.elapsed(),
                format!("Hash join with collision keys failed: {}", e),
            )
        }
    };

    // All keys should match
    if ctx.device_row_count(&joined) != SIZE as u64 {
        return TestResult::error(
            "test_high_contention_join",
            start.elapsed(),
            format!(
                "Collision join returned {} rows, expected {}",
                ctx.device_row_count(&joined), SIZE
            ),
        );
    }

    // Download and verify
    let joined_keys = match ctx.provider.download_column_u32(&joined, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_high_contention_join",
                start.elapsed(),
                format!("Failed to download collision join keys: {}", e),
            )
        }
    };

    let joined_lvals = match ctx.provider.download_column_u32(&joined, 1) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_high_contention_join",
                start.elapsed(),
                format!("Failed to download collision join lvals: {}", e),
            )
        }
    };

    let joined_rvals = match ctx.provider.download_column_u32(&joined, 2) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_high_contention_join",
                start.elapsed(),
                format!("Failed to download collision join rvals: {}", e),
            )
        }
    };

    // Verify all key-value relationships
    for i in 0..ctx.device_row_count(&joined) as usize {
        let key = joined_keys[i];
        let lval = joined_lvals[i];
        let rval = joined_rvals[i];

        if lval != key + 1 {
            return TestResult::error(
                "test_high_contention_join",
                start.elapsed(),
                format!(
                    "Collision row {}: lval {} != key + 1 = {}",
                    i,
                    lval,
                    key + 1
                ),
            );
        }

        if rval != key + 2 {
            return TestResult::error(
                "test_high_contention_join",
                start.elapsed(),
                format!(
                    "Collision row {}: rval {} != key + 2 = {}",
                    i,
                    rval,
                    key + 2
                ),
            );
        }
    }

    // Verify all keys are present (no lost due to collisions)
    let seen_keys: HashSet<u32> = joined_keys.iter().cloned().collect();
    let expected_keys: HashSet<u32> = collision_keys.iter().cloned().collect();
    if seen_keys != expected_keys {
        return TestResult::error(
            "test_high_contention_join",
            start.elapsed(),
            format!(
                "Missing keys in collision join: expected {}, got {}",
                expected_keys.len(),
                seen_keys.len()
            ),
        );
    }

    // Pattern 2: All same key (maximum contention)
    let same_key_left: Vec<u32> = vec![12345; 1000];
    let same_key_left_vals: Vec<u32> = (0..1000u32).collect();
    let same_key_right: Vec<u32> = vec![12345; 100];
    let same_key_right_vals: Vec<u32> = (0..100u32).map(|i| i * 1000).collect();

    let left_same = match ctx
        .provider
        .create_buffer_from_u32_columns(&[&same_key_left, &same_key_left_vals], left_schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_high_contention_join",
                start.elapsed(),
                format!("Failed to create same-key left buffer: {}", e),
            )
        }
    };

    let right_same = match ctx.provider.create_buffer_from_u32_columns(
        &[&same_key_right, &same_key_right_vals],
        right_schema.clone(),
    ) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_high_contention_join",
                start.elapsed(),
                format!("Failed to create same-key right buffer: {}", e),
            )
        }
    };

    // Join with same key - should produce 1000 * 100 = 100,000 rows (Cartesian for that key)
    let same_joined = match ctx.provider.hash_join(&left_same, &right_same, &[0], &[0]) {
        Ok(j) => j,
        Err(e) => {
            return TestResult::error(
                "test_high_contention_join",
                start.elapsed(),
                format!("Same-key hash join failed: {}", e),
            )
        }
    };

    let expected_same = 1000 * 100;
    if ctx.device_row_count(&same_joined) != expected_same as u64 {
        return TestResult::error(
            "test_high_contention_join",
            start.elapsed(),
            format!(
                "Same-key join returned {} rows, expected {} (Cartesian)",
                ctx.device_row_count(&same_joined), expected_same
            ),
        );
    }

    // Verify all joined keys are the same key
    let same_joined_keys = match ctx.provider.download_column_u32(&same_joined, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_high_contention_join",
                start.elapsed(),
                format!("Failed to download same-key join keys: {}", e),
            )
        }
    };

    for (i, &key) in same_joined_keys.iter().enumerate() {
        if key != 12345 {
            return TestResult::error(
                "test_high_contention_join",
                start.elapsed(),
                format!("Same-key row {}: key {} != 12345", i, key),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_high_contention_join",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_high_contention_join", start.elapsed())
}

/// Test 4: Operations that rely on atomic counting (filter scan).
///
/// Filter operations use atomic prefix sum / scan to compute output positions.
/// This test verifies that atomic counting produces correct output offsets.
fn test_atomic_counting(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Test prefix_sum_mask directly (if available) and verify through filter
    let test_cases: Vec<(usize, Box<dyn Fn(usize) -> bool>)> = vec![
        // (size, predicate)
        (1000, Box::new(|i| i % 2 == 0)),   // 50% selectivity
        (10000, Box::new(|i| i % 10 == 0)), // 10% selectivity
        (50000, Box::new(|i| i < 10000)),   // First 20%
        (100000, Box::new(|i| i % 7 == 0)), // ~14% selectivity
    ];

    for (size, predicate) in test_cases {
        // Create mask
        let mask: Vec<u8> = (0..size)
            .map(|i| if predicate(i) { 1 } else { 0 })
            .collect();
        let expected_count: usize = mask.iter().map(|&m| m as usize).sum();

        // Test prefix_sum_mask
        let (prefix_sums, total) = match ctx.provider.prefix_sum_mask(&mask) {
            Ok(result) => result,
            Err(e) => {
                return TestResult::error(
                    "test_atomic_counting",
                    start.elapsed(),
                    format!("Size {}: prefix_sum_mask failed: {}", size, e),
                )
            }
        };

        // Verify total count
        if total != expected_count as u32 {
            return TestResult::error(
                "test_atomic_counting",
                start.elapsed(),
                format!(
                    "Size {}: prefix_sum total {} != expected {}",
                    size, total, expected_count
                ),
            );
        }

        // Verify prefix sum is monotonic and correct
        let mut running_sum = 0u32;
        for (i, &m) in mask.iter().enumerate() {
            if prefix_sums[i] != running_sum {
                return TestResult::error(
                    "test_atomic_counting",
                    start.elapsed(),
                    format!(
                        "Size {}: prefix_sums[{}] = {}, expected {}",
                        size, i, prefix_sums[i], running_sum
                    ),
                );
            }
            running_sum += m as u32;
        }

        // Verify final prefix sum + last mask equals total
        if running_sum != total {
            return TestResult::error(
                "test_atomic_counting",
                start.elapsed(),
                format!(
                    "Size {}: final sum {} != total {}",
                    size, running_sum, total
                ),
            );
        }

        // Also verify through filter operation
        let data: Vec<u32> = (0..size as u32).collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_u32_slice(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_atomic_counting",
                    start.elapsed(),
                    format!("Size {}: failed to create buffer: {}", size, e),
                )
            }
        };

        let filtered = match ctx.provider.filter_by_mask(&buffer, &mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_atomic_counting",
                    start.elapsed(),
                    format!("Size {}: filter failed: {}", size, e),
                )
            }
        };

        if ctx.device_row_count(&filtered) != expected_count as u64 {
            return TestResult::error(
                "test_atomic_counting",
                start.elapsed(),
                format!(
                    "Size {}: filter returned {} rows, expected {}",
                    size, ctx.device_row_count(&filtered), expected_count
                ),
            );
        }

        // Verify filtered positions match prefix sums
        let filtered_data = match ctx.provider.download_column_u32(&filtered, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_atomic_counting",
                    start.elapsed(),
                    format!("Size {}: failed to download filtered: {}", size, e),
                )
            }
        };

        let mut expected_idx = 0;
        for (i, &m) in mask.iter().enumerate() {
            if m == 1 {
                if expected_idx >= filtered_data.len() {
                    return TestResult::error(
                        "test_atomic_counting",
                        start.elapsed(),
                        format!(
                            "Size {}: filtered data too short at expected index {}",
                            size, expected_idx
                        ),
                    );
                }
                if filtered_data[expected_idx] != i as u32 {
                    return TestResult::error(
                        "test_atomic_counting",
                        start.elapsed(),
                        format!(
                            "Size {}: filtered[{}] = {}, expected {}",
                            size, expected_idx, filtered_data[expected_idx], i
                        ),
                    );
                }
                expected_idx += 1;
            }
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_atomic_counting",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_atomic_counting", start.elapsed())
}

/// Test 5: Multiple operations using atomics in sequence.
///
/// Tests that multiple operations using atomics can run in sequence without
/// interference, verifying proper atomic operation cleanup between operations.
fn test_concurrent_atomic_updates(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("val".to_string(), ScalarType::U32),
    ]);

    // Run multiple atomic-using operations in sequence
    const NUM_ITERATIONS: usize = 5;
    const SIZE: usize = 10000;

    for iteration in 0..NUM_ITERATIONS {
        // Create data unique to this iteration
        let keys: Vec<u32> = (0..SIZE).map(|i| (i + iteration * 1000) as u32).collect();
        let vals: Vec<u32> = keys.iter().map(|&k| k * 10).collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_u32_columns(&[&keys, &vals], schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_concurrent_atomic_updates",
                    start.elapsed(),
                    format!("Iteration {}: failed to create buffer: {}", iteration, e),
                )
            }
        };

        // Operation 1: Dedup (uses atomic marking)
        let deduped = match ctx.provider.dedup(&buffer, &[0]) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_concurrent_atomic_updates",
                    start.elapsed(),
                    format!("Iteration {}: dedup failed: {}", iteration, e),
                )
            }
        };

        // All keys are unique, so dedup should return same count
        if ctx.device_row_count(&deduped) != SIZE as u64 {
            return TestResult::error(
                "test_concurrent_atomic_updates",
                start.elapsed(),
                format!(
                    "Iteration {}: dedup returned {} rows, expected {}",
                    iteration, ctx.device_row_count(&deduped), SIZE
                ),
            );
        }

        // Operation 2: Filter (uses atomic scan)
        let mask: Vec<u8> = (0..SIZE).map(|i| if i % 3 == 0 { 1 } else { 0 }).collect();
        let expected_filter = (SIZE + 2) / 3;

        let filtered = match ctx.provider.filter_by_mask(&buffer, &mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_concurrent_atomic_updates",
                    start.elapsed(),
                    format!("Iteration {}: filter failed: {}", iteration, e),
                )
            }
        };

        if ctx.device_row_count(&filtered) != expected_filter as u64 {
            return TestResult::error(
                "test_concurrent_atomic_updates",
                start.elapsed(),
                format!(
                    "Iteration {}: filter returned {} rows, expected {}",
                    iteration, ctx.device_row_count(&filtered), expected_filter
                ),
            );
        }

        // Operation 3: Self-join (uses atomic hash table)
        // Join buffer with itself on key
        let joined = match ctx.provider.hash_join(&buffer, &buffer, &[0], &[0]) {
            Ok(j) => j,
            Err(e) => {
                return TestResult::error(
                    "test_concurrent_atomic_updates",
                    start.elapsed(),
                    format!("Iteration {}: self-join failed: {}", iteration, e),
                )
            }
        };

        // Self-join should return SIZE rows (each key matches exactly once)
        if ctx.device_row_count(&joined) != SIZE as u64 {
            return TestResult::error(
                "test_concurrent_atomic_updates",
                start.elapsed(),
                format!(
                    "Iteration {}: self-join returned {} rows, expected {}",
                    iteration, ctx.device_row_count(&joined), SIZE
                ),
            );
        }

        // Verify joined results
        let joined_keys = match ctx.provider.download_column_u32(&joined, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_concurrent_atomic_updates",
                    start.elapsed(),
                    format!(
                        "Iteration {}: failed to download join keys: {}",
                        iteration, e
                    ),
                )
            }
        };

        // Verify all original keys appear in join result
        let joined_key_set: HashSet<u32> = joined_keys.iter().cloned().collect();
        let original_key_set: HashSet<u32> = keys.iter().cloned().collect();

        if joined_key_set != original_key_set {
            return TestResult::error(
                "test_concurrent_atomic_updates",
                start.elapsed(),
                format!(
                    "Iteration {}: self-join keys don't match original ({} vs {})",
                    iteration,
                    joined_key_set.len(),
                    original_key_set.len()
                ),
            );
        }

        // Operation 4: Another dedup to verify atomics reset properly
        let deduped2 = match ctx.provider.dedup(&buffer, &[0]) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_concurrent_atomic_updates",
                    start.elapsed(),
                    format!("Iteration {}: second dedup failed: {}", iteration, e),
                )
            }
        };

        if ctx.device_row_count(&deduped2) != SIZE as u64 {
            return TestResult::error(
                "test_concurrent_atomic_updates",
                start.elapsed(),
                format!(
                    "Iteration {}: second dedup returned {} rows, expected {} (atomics may not have reset)",
                    iteration, ctx.device_row_count(&deduped2), SIZE
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_concurrent_atomic_updates",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_concurrent_atomic_updates", start.elapsed())
}
