//! Category 14: Integer edge cases
//!
//! Tests integer boundary conditions including overflow boundaries,
//! full range coverage, signed comparison, and wraparound keys.

use crate::harness::{CategoryResult, TestContext, TestResult};
use std::collections::HashSet;
use std::time::Instant;
use xlog_core::{ScalarType, Schema};

/// Run all tests in this category.
pub fn run_all(ctx: &TestContext) -> CategoryResult {
    let mut results = CategoryResult::new("c14_integer");
    let start = Instant::now();

    results.add_result(test_i64_overflow_boundaries(ctx));
    results.add_result(test_u64_overflow_boundaries(ctx));
    results.add_result(test_u32_full_range(ctx));
    results.add_result(test_i64_signed_comparison(ctx));
    results.add_result(test_integer_wraparound_keys(ctx));

    results.set_duration(start.elapsed());
    results
}

/// Test 1: Test i64::MIN, i64::MAX, 0, -1, 1.
///
/// Verifies that signed 64-bit integer boundary values are handled correctly
/// in sorting and filtering operations.
fn test_i64_overflow_boundaries(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::I64)]);

    // Create data with i64 boundary values
    let data: Vec<i64> = vec![
        i64::MIN,
        i64::MAX,
        0,
        -1,
        1,
        i64::MIN + 1,
        i64::MAX - 1,
        -2,
        2,
        i64::MIN / 2,
        i64::MAX / 2,
        -i64::MAX, // i64::MIN + 1
        100,
        -100,
        1000000,
        -1000000,
    ];

    let buffer = match ctx
        .provider
        .create_buffer_from_slice::<i64>(&data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_i64_overflow_boundaries",
                start.elapsed(),
                format!("Failed to create buffer: {}", e),
            )
        }
    };

    // Sort the buffer
    let sorted = match ctx.provider.sort(&buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_i64_overflow_boundaries",
                start.elapsed(),
                format!("Sort failed: {}", e),
            )
        }
    };

    // Download sorted data
    let sorted_data = match ctx.provider.download_column::<i64>(&sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_i64_overflow_boundaries",
                start.elapsed(),
                format!("Failed to download sorted column: {}", e),
            )
        }
    };

    // Verify row count preserved
    if sorted_data.len() != data.len() {
        return TestResult::error(
            "test_i64_overflow_boundaries",
            start.elapsed(),
            format!(
                "Sort returned {} rows, expected {}",
                sorted_data.len(),
                data.len()
            ),
        );
    }

    // Verify sort order
    for i in 1..sorted_data.len() {
        if sorted_data[i] < sorted_data[i - 1] {
            return TestResult::error(
                "test_i64_overflow_boundaries",
                start.elapsed(),
                format!(
                    "Sort order incorrect at index {}: {} should be >= {}",
                    i,
                    sorted_data[i],
                    sorted_data[i - 1]
                ),
            );
        }
    }

    // Verify i64::MIN is first
    if sorted_data[0] != i64::MIN {
        return TestResult::error(
            "test_i64_overflow_boundaries",
            start.elapsed(),
            format!(
                "First element should be i64::MIN ({}), got {}",
                i64::MIN,
                sorted_data[0]
            ),
        );
    }

    // Verify i64::MAX is last
    if sorted_data[sorted_data.len() - 1] != i64::MAX {
        return TestResult::error(
            "test_i64_overflow_boundaries",
            start.elapsed(),
            format!(
                "Last element should be i64::MAX ({}), got {}",
                i64::MAX,
                sorted_data[sorted_data.len() - 1]
            ),
        );
    }

    // Verify all original values are present
    let original_set: HashSet<i64> = data.iter().copied().collect();
    let sorted_set: HashSet<i64> = sorted_data.iter().copied().collect();
    if original_set != sorted_set {
        return TestResult::error(
            "test_i64_overflow_boundaries",
            start.elapsed(),
            "Some values were lost or changed during sort".to_string(),
        );
    }

    // Test filter: keep only negative values
    let mask: Vec<u8> = data.iter().map(|&v| if v < 0 { 1 } else { 0 }).collect();
    let filtered = match ctx.provider.filter_by_mask(&buffer, &mask) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_i64_overflow_boundaries",
                start.elapsed(),
                format!("Filter failed: {}", e),
            )
        }
    };

    let filtered_data = match ctx.provider.download_column::<i64>(&filtered, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_i64_overflow_boundaries",
                start.elapsed(),
                format!("Failed to download filtered column: {}", e),
            )
        }
    };

    // Verify filter result count
    let expected_negative_count = data.iter().filter(|&&v| v < 0).count();
    if filtered_data.len() != expected_negative_count {
        return TestResult::error(
            "test_i64_overflow_boundaries",
            start.elapsed(),
            format!(
                "Filter returned {} rows, expected {}",
                filtered_data.len(),
                expected_negative_count
            ),
        );
    }

    // Verify i64::MIN is in the filtered results
    if !filtered_data.contains(&i64::MIN) {
        return TestResult::error(
            "test_i64_overflow_boundaries",
            start.elapsed(),
            "Filtered negative values should include i64::MIN".to_string(),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_i64_overflow_boundaries",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_i64_overflow_boundaries", start.elapsed())
}

/// Test 2: Test u64::MIN (0), u64::MAX.
///
/// Verifies that unsigned 64-bit integer boundary values are handled correctly.
fn test_u64_overflow_boundaries(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U64)]);

    // Create data with u64 boundary values
    let data: Vec<u64> = vec![
        u64::MIN, // 0
        u64::MAX,
        1,
        u64::MAX - 1,
        u64::MAX / 2,
        u64::MAX / 4,
        u64::MAX / 4 * 3,
        100,
        1000,
        1_000_000,
        1_000_000_000,
        1_000_000_000_000,
        // High bit patterns
        0x8000_0000_0000_0000, // Only high bit set
        0xFFFF_FFFF_0000_0000, // Upper 32 bits
        0x0000_0000_FFFF_FFFF, // Lower 32 bits
        0xAAAA_AAAA_AAAA_AAAA, // Alternating bits
    ];

    let buffer = match ctx
        .provider
        .create_buffer_from_slice::<u64>(&data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_u64_overflow_boundaries",
                start.elapsed(),
                format!("Failed to create buffer: {}", e),
            )
        }
    };

    // Sort the buffer
    let sorted = match ctx.provider.sort(&buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_u64_overflow_boundaries",
                start.elapsed(),
                format!("Sort failed: {}", e),
            )
        }
    };

    // Download sorted data
    let sorted_data = match ctx.provider.download_column::<u64>(&sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_u64_overflow_boundaries",
                start.elapsed(),
                format!("Failed to download sorted column: {}", e),
            )
        }
    };

    // Verify row count preserved
    if sorted_data.len() != data.len() {
        return TestResult::error(
            "test_u64_overflow_boundaries",
            start.elapsed(),
            format!(
                "Sort returned {} rows, expected {}",
                sorted_data.len(),
                data.len()
            ),
        );
    }

    // Verify sort order
    for i in 1..sorted_data.len() {
        if sorted_data[i] < sorted_data[i - 1] {
            return TestResult::error(
                "test_u64_overflow_boundaries",
                start.elapsed(),
                format!(
                    "Sort order incorrect at index {}: {} should be >= {}",
                    i,
                    sorted_data[i],
                    sorted_data[i - 1]
                ),
            );
        }
    }

    // Verify u64::MIN (0) is first
    if sorted_data[0] != u64::MIN {
        return TestResult::error(
            "test_u64_overflow_boundaries",
            start.elapsed(),
            format!("First element should be 0, got {}", sorted_data[0]),
        );
    }

    // Verify u64::MAX is last
    if sorted_data[sorted_data.len() - 1] != u64::MAX {
        return TestResult::error(
            "test_u64_overflow_boundaries",
            start.elapsed(),
            format!(
                "Last element should be u64::MAX ({}), got {}",
                u64::MAX,
                sorted_data[sorted_data.len() - 1]
            ),
        );
    }

    // Verify all original values are present
    let original_set: HashSet<u64> = data.iter().copied().collect();
    let sorted_set: HashSet<u64> = sorted_data.iter().copied().collect();
    if original_set != sorted_set {
        return TestResult::error(
            "test_u64_overflow_boundaries",
            start.elapsed(),
            "Some values were lost or changed during sort".to_string(),
        );
    }

    // Verify high bit value is correctly ordered (not treated as negative)
    let high_bit_value = 0x8000_0000_0000_0000u64;
    let high_bit_pos = sorted_data.iter().position(|&v| v == high_bit_value);
    if let Some(pos) = high_bit_pos {
        // All values before should be less, all values after should be greater
        for i in 0..pos {
            if sorted_data[i] >= high_bit_value {
                return TestResult::error(
                    "test_u64_overflow_boundaries",
                    start.elapsed(),
                    format!(
                        "Value {} at position {} should be less than {} at position {}",
                        sorted_data[i], i, high_bit_value, pos
                    ),
                );
            }
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_u64_overflow_boundaries",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_u64_overflow_boundaries", start.elapsed())
}

/// Test 3: Test across full u32 range.
///
/// Verifies that operations work correctly across the full range of u32 values,
/// including boundary values and values with specific bit patterns.
fn test_u32_full_range(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Create data spanning the full u32 range
    let mut data: Vec<u32> = vec![
        u32::MIN, // 0
        u32::MAX,
        1,
        u32::MAX - 1,
        u32::MAX / 2,
        u32::MAX / 4,
        u32::MAX / 4 * 3,
        // Byte boundary values
        0xFF,
        0xFF00,
        0xFF_0000,
        0xFF00_0000,
        // Bit patterns
        0x8000_0000, // Only high bit set
        0xFFFF_0000, // Upper 16 bits
        0x0000_FFFF, // Lower 16 bits
        0xAAAA_AAAA, // Alternating bits
        0x5555_5555, // Inverted alternating bits
    ];

    // Add evenly distributed values across the range
    for i in 0..16 {
        data.push((u32::MAX as u64 * i / 16) as u32);
    }

    let buffer = match ctx
        .provider
        .create_buffer_from_slice::<u32>(&data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_u32_full_range",
                start.elapsed(),
                format!("Failed to create buffer: {}", e),
            )
        }
    };

    // Sort the buffer
    let sorted = match ctx.provider.sort(&buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_u32_full_range",
                start.elapsed(),
                format!("Sort failed: {}", e),
            )
        }
    };

    // Download sorted data
    let sorted_data = match ctx.provider.download_column::<u32>(&sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_u32_full_range",
                start.elapsed(),
                format!("Failed to download sorted column: {}", e),
            )
        }
    };

    // Verify row count preserved
    if sorted_data.len() != data.len() {
        return TestResult::error(
            "test_u32_full_range",
            start.elapsed(),
            format!(
                "Sort returned {} rows, expected {}",
                sorted_data.len(),
                data.len()
            ),
        );
    }

    // Verify sort order
    for i in 1..sorted_data.len() {
        if sorted_data[i] < sorted_data[i - 1] {
            return TestResult::error(
                "test_u32_full_range",
                start.elapsed(),
                format!(
                    "Sort order incorrect at index {}: {} should be >= {}",
                    i,
                    sorted_data[i],
                    sorted_data[i - 1]
                ),
            );
        }
    }

    // Verify 0 is first
    if sorted_data[0] != 0 {
        return TestResult::error(
            "test_u32_full_range",
            start.elapsed(),
            format!("First element should be 0, got {}", sorted_data[0]),
        );
    }

    // Verify u32::MAX is last
    if sorted_data[sorted_data.len() - 1] != u32::MAX {
        return TestResult::error(
            "test_u32_full_range",
            start.elapsed(),
            format!(
                "Last element should be u32::MAX ({}), got {}",
                u32::MAX,
                sorted_data[sorted_data.len() - 1]
            ),
        );
    }

    // Test filter: keep only values in upper half of range
    let threshold = u32::MAX / 2;
    let mask: Vec<u8> = data
        .iter()
        .map(|&v| if v >= threshold { 1 } else { 0 })
        .collect();
    let filtered = match ctx.provider.filter_by_mask(&buffer, &mask) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_u32_full_range",
                start.elapsed(),
                format!("Filter failed: {}", e),
            )
        }
    };

    let filtered_data = match ctx.provider.download_column::<u32>(&filtered, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_u32_full_range",
                start.elapsed(),
                format!("Failed to download filtered column: {}", e),
            )
        }
    };

    // Verify all filtered values are >= threshold
    for &val in &filtered_data {
        if val < threshold {
            return TestResult::error(
                "test_u32_full_range",
                start.elapsed(),
                format!(
                    "Filtered value {} should be >= threshold {}",
                    val, threshold
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_u32_full_range",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_u32_full_range", start.elapsed())
}

/// Test 4: Verify signed comparison works correctly.
///
/// Tests that i64 values are compared as signed integers (negative < positive),
/// not as unsigned bit patterns.
fn test_i64_signed_comparison(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::I64)]);

    // Create data that would sort differently if treated as unsigned
    // In unsigned interpretation, negative numbers have high bit set and would sort last
    let data: Vec<i64> = vec![
        -1, // 0xFFFF_FFFF_FFFF_FFFF as unsigned
        -2, // 0xFFFF_FFFF_FFFF_FFFE as unsigned
        -1000,
        -1_000_000,
        0,
        1,
        2,
        1000,
        1_000_000,
        i64::MIN, // 0x8000_0000_0000_0000 - would sort middle if unsigned
        i64::MAX, // 0x7FFF_FFFF_FFFF_FFFF - would sort before -1 if unsigned
    ];

    let buffer = match ctx
        .provider
        .create_buffer_from_slice::<i64>(&data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_i64_signed_comparison",
                start.elapsed(),
                format!("Failed to create buffer: {}", e),
            )
        }
    };

    // Sort the buffer
    let sorted = match ctx.provider.sort(&buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_i64_signed_comparison",
                start.elapsed(),
                format!("Sort failed: {}", e),
            )
        }
    };

    // Download sorted data
    let sorted_data = match ctx.provider.download_column::<i64>(&sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_i64_signed_comparison",
                start.elapsed(),
                format!("Failed to download sorted column: {}", e),
            )
        }
    };

    // Verify signed sort order
    for i in 1..sorted_data.len() {
        if sorted_data[i] < sorted_data[i - 1] {
            return TestResult::error(
                "test_i64_signed_comparison",
                start.elapsed(),
                format!(
                    "Signed sort order incorrect at index {}: {} should be >= {}",
                    i,
                    sorted_data[i],
                    sorted_data[i - 1]
                ),
            );
        }
    }

    // Verify expected order: i64::MIN should come before negative values
    // Expected order: i64::MIN, -1_000_000, -1000, -2, -1, 0, 1, 2, 1000, 1_000_000, i64::MAX
    let mut expected = data.clone();
    expected.sort();

    if sorted_data != expected {
        return TestResult::error(
            "test_i64_signed_comparison",
            start.elapsed(),
            format!(
                "Sort order doesn't match expected signed order.\nGot: {:?}\nExpected: {:?}",
                sorted_data, expected
            ),
        );
    }

    // Specifically verify that negative values come before positive values
    let first_non_negative_idx = sorted_data.iter().position(|&v| v >= 0);
    if let Some(idx) = first_non_negative_idx {
        // All values before idx should be negative
        for i in 0..idx {
            if sorted_data[i] >= 0 {
                return TestResult::error(
                    "test_i64_signed_comparison",
                    start.elapsed(),
                    format!(
                        "Negative values should come before positive: found {} at index {}",
                        sorted_data[i], i
                    ),
                );
            }
        }
    }

    // Verify i64::MIN (most negative) is first
    if sorted_data[0] != i64::MIN {
        return TestResult::error(
            "test_i64_signed_comparison",
            start.elapsed(),
            format!(
                "First element should be i64::MIN ({}), got {} - may be treated as unsigned",
                i64::MIN,
                sorted_data[0]
            ),
        );
    }

    // Verify i64::MAX (most positive) is last
    if sorted_data[sorted_data.len() - 1] != i64::MAX {
        return TestResult::error(
            "test_i64_signed_comparison",
            start.elapsed(),
            format!(
                "Last element should be i64::MAX ({}), got {}",
                i64::MAX,
                sorted_data[sorted_data.len() - 1]
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_i64_signed_comparison",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_i64_signed_comparison", start.elapsed())
}

/// Test 5: Test keys near wraparound boundaries in hash tables.
///
/// Verifies that hash join operations work correctly with keys near integer
/// boundaries that might cause hash collisions or wraparound issues.
fn test_integer_wraparound_keys(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let left_schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("lval".to_string(), ScalarType::U32),
    ]);
    let right_schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("rval".to_string(), ScalarType::U32),
    ]);

    // Keys near wraparound boundary (u32::MAX and near it)
    let left_keys: Vec<u32> = vec![
        u32::MAX,
        u32::MAX - 1,
        u32::MAX - 2,
        0,
        1,
        2,
        u32::MAX / 2,
        u32::MAX / 2 + 1,
        // Keys that might cause hash collisions
        0x8000_0000,
        0x8000_0001,
        0xFFFF_FFFF,
        0xFFFF_FFFE,
    ];
    let left_vals: Vec<u32> = left_keys.iter().map(|&k| k.wrapping_add(100)).collect();

    // Right keys - subset of left keys
    let right_keys: Vec<u32> = vec![u32::MAX, u32::MAX - 1, 0, 1, u32::MAX / 2, 0x8000_0000];
    let right_vals: Vec<u32> = right_keys.iter().map(|&k| k.wrapping_mul(10)).collect();

    let left_buffer = match ctx
        .provider
        .create_buffer_from_u32_columns(&[&left_keys, &left_vals], left_schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_integer_wraparound_keys",
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
                "test_integer_wraparound_keys",
                start.elapsed(),
                format!("Failed to create right buffer: {}", e),
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
                "test_integer_wraparound_keys",
                start.elapsed(),
                format!("Hash join failed: {}", e),
            )
        }
    };

    // Calculate expected join count
    let right_key_set: HashSet<u32> = right_keys.iter().copied().collect();
    let expected_matches = left_keys
        .iter()
        .filter(|k| right_key_set.contains(k))
        .count();

    if ctx.device_row_count(&joined) != expected_matches as u64 {
        return TestResult::error(
            "test_integer_wraparound_keys",
            start.elapsed(),
            format!(
                "Join returned {} rows, expected {}",
                ctx.device_row_count(&joined),
                expected_matches
            ),
        );
    }

    // Download and verify join results
    let joined_keys = match ctx.provider.download_column::<u32>(&joined, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_integer_wraparound_keys",
                start.elapsed(),
                format!("Failed to download joined keys: {}", e),
            )
        }
    };

    let joined_lvals = match ctx.provider.download_column::<u32>(&joined, 1) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_integer_wraparound_keys",
                start.elapsed(),
                format!("Failed to download joined lvals: {}", e),
            )
        }
    };

    let joined_rvals = match ctx.provider.download_column::<u32>(&joined, 2) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_integer_wraparound_keys",
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

        // Verify lval matches expected pattern
        let expected_lval = key.wrapping_add(100);
        if lval != expected_lval {
            return TestResult::error(
                "test_integer_wraparound_keys",
                start.elapsed(),
                format!(
                    "Row {}: lval {} doesn't match expected {} for key {}",
                    i, lval, expected_lval, key
                ),
            );
        }

        // Verify rval matches expected pattern
        let expected_rval = key.wrapping_mul(10);
        if rval != expected_rval {
            return TestResult::error(
                "test_integer_wraparound_keys",
                start.elapsed(),
                format!(
                    "Row {}: rval {} doesn't match expected {} for key {}",
                    i, rval, expected_rval, key
                ),
            );
        }

        // Verify key is in right table
        if !right_key_set.contains(&key) {
            return TestResult::error(
                "test_integer_wraparound_keys",
                start.elapsed(),
                format!("Row {}: key {} is not in right table", i, key),
            );
        }
    }

    // Verify all right keys that should match are present in results
    let joined_key_set: HashSet<u32> = joined_keys.iter().copied().collect();
    let left_key_set: HashSet<u32> = left_keys.iter().copied().collect();
    for &rkey in &right_keys {
        if left_key_set.contains(&rkey) && !joined_key_set.contains(&rkey) {
            return TestResult::error(
                "test_integer_wraparound_keys",
                start.elapsed(),
                format!("Key {} should appear in join result but doesn't", rkey),
            );
        }
    }

    // Specifically verify wraparound keys are handled correctly
    let critical_keys = [u32::MAX, 0, 0x8000_0000];
    for &ckey in &critical_keys {
        if left_key_set.contains(&ckey) && right_key_set.contains(&ckey) {
            if !joined_key_set.contains(&ckey) {
                return TestResult::error(
                    "test_integer_wraparound_keys",
                    start.elapsed(),
                    format!(
                        "Critical key {} (0x{:08X}) missing from join result",
                        ckey, ckey
                    ),
                );
            }
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_integer_wraparound_keys",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_integer_wraparound_keys", start.elapsed())
}
