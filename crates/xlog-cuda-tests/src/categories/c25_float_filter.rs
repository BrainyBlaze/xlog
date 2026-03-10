//! Category 25: Float filter predicate edge cases
//!
//! Tests floating-point comparison semantics in filter operations:
//! - Eq/Ne use IEEE 754 semantics (NaN == NaN is false, NaN != NaN is true)
//! - Lt/Le/Gt/Ge use total ordering (NaN > Inf is true, -0.0 < +0.0 is true)
//!
//! Total ordering: -Inf < negative numbers < -0.0 < +0.0 < positive numbers < +Inf < NaN

use crate::harness::{CategoryResult, TestContext, TestResult};
use std::time::Instant;
use xlog_core::{ScalarType, Schema};
use xlog_cuda::CompareOp;

/// Run all tests in this category.
pub fn run_all(ctx: &TestContext) -> CategoryResult {
    let mut results = CategoryResult::new("c25_float_filter");
    let start = Instant::now();

    // f64 tests
    results.add_result(test_f64_nan_greater_than_inf(ctx));
    results.add_result(test_f64_negative_zero_less_than_positive_zero(ctx));
    results.add_result(test_f64_nan_equality_ieee(ctx));
    results.add_result(test_f64_nan_ordering_total(ctx));
    results.add_result(test_f64_total_ordering_comprehensive(ctx));

    // f32 tests
    results.add_result(test_f32_nan_greater_than_inf(ctx));
    results.add_result(test_f32_negative_zero_less_than_positive_zero(ctx));
    results.add_result(test_f32_nan_equality_ieee(ctx));
    results.add_result(test_f32_nan_ordering_total(ctx));
    results.add_result(test_f32_total_ordering_comprehensive(ctx));

    results.set_duration(start.elapsed());
    results
}

/// Test 1: f64 NaN > Infinity should return TRUE (total ordering).
///
/// Under total ordering, NaN is greater than all other values including Infinity.
fn test_f64_nan_greater_than_inf(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::F64)]);

    // Data with NaN and Infinity values
    let data: Vec<f64> = vec![
        f64::NAN,          // index 0: should pass (NaN > Inf)
        f64::INFINITY,     // index 1: should NOT pass (Inf > Inf is false)
        f64::NEG_INFINITY, // index 2: should NOT pass
        f64::NAN,          // index 3: should pass (NaN > Inf)
        1.0,               // index 4: should NOT pass
        f64::INFINITY,     // index 5: should NOT pass
    ];

    let buffer = match ctx
        .provider
        .create_buffer_from_slice::<f64>(&data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_f64_nan_greater_than_inf",
                start.elapsed(),
                format!("Failed to create buffer: {}", e),
            )
        }
    };

    // Filter for values > INFINITY using CompareOp::Gt
    // Under total ordering, NaN > INFINITY is TRUE
    // So 2 NaN values should pass this filter
    let filtered = match ctx
        .provider
        .filter_f64(&buffer, 0, f64::INFINITY, CompareOp::Gt)
    {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_f64_nan_greater_than_inf",
                start.elapsed(),
                format!("Filter failed: {}", e),
            )
        }
    };

    // 2 NaN values should be greater than INFINITY under total ordering
    if ctx.device_row_count(&filtered) != 2 {
        return TestResult::error(
            "test_f64_nan_greater_than_inf",
            start.elapsed(),
            format!(
                "Expected 2 rows where val > INFINITY (the NaN values), got {} (NaN > Inf should be true per total ordering)",
                ctx.device_row_count(&filtered)
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_f64_nan_greater_than_inf",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_f64_nan_greater_than_inf", start.elapsed())
}

/// Test 2: f64 -0.0 < +0.0 should return TRUE (total ordering).
///
/// Under total ordering, -0.0 is less than +0.0 (unlike IEEE 754 where they're equal).
fn test_f64_negative_zero_less_than_positive_zero(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::F64)]);

    // Data with negative and positive zeros
    let data: Vec<f64> = vec![
        -0.0, // index 0: should pass (-0.0 < +0.0)
        0.0,  // index 1: should NOT pass (+0.0 < +0.0 is false)
        -0.0, // index 2: should pass (-0.0 < +0.0)
        0.0,  // index 3: should NOT pass
        1.0,  // index 4: should NOT pass
        -1.0, // index 5: should pass (-1.0 < +0.0)
    ];

    let buffer = match ctx
        .provider
        .create_buffer_from_slice::<f64>(&data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_f64_negative_zero_less_than_positive_zero",
                start.elapsed(),
                format!("Failed to create buffer: {}", e),
            )
        }
    };

    // Filter for values < 0.0 (which is +0.0)
    // Under total ordering, -0.0 < +0.0 is TRUE
    // So -0.0 (x2) and -1.0 should pass = 3 values
    let filtered = match ctx.provider.filter_f64(&buffer, 0, 0.0, CompareOp::Lt) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_f64_negative_zero_less_than_positive_zero",
                start.elapsed(),
                format!("Filter failed: {}", e),
            )
        }
    };

    // -0.0 (x2) and -1.0 should pass = 3 values
    let expected_count = 3;
    if ctx.device_row_count(&filtered) as usize != expected_count {
        return TestResult::error(
            "test_f64_negative_zero_less_than_positive_zero",
            start.elapsed(),
            format!(
                "Expected {} rows where val < 0.0, got {} (-0.0 < +0.0 should be true per total ordering)",
                expected_count,
                ctx.device_row_count(&filtered)
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_f64_negative_zero_less_than_positive_zero",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed(
        "test_f64_negative_zero_less_than_positive_zero",
        start.elapsed(),
    )
}

/// Test 3: f64 NaN == NaN should return FALSE, NaN != NaN should return TRUE (IEEE 754).
///
/// Equality comparisons use IEEE 754 semantics where NaN is not equal to anything.
fn test_f64_nan_equality_ieee(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::F64)]);

    // Data with NaN values
    let data: Vec<f64> = vec![f64::NAN, 1.0, f64::NAN, 2.0, f64::NAN, 3.0];

    let buffer = match ctx
        .provider
        .create_buffer_from_slice::<f64>(&data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_f64_nan_equality_ieee",
                start.elapsed(),
                format!("Failed to create buffer: {}", e),
            )
        }
    };

    // Test NaN == NaN (should be FALSE per IEEE 754)
    let eq_result = match ctx.provider.filter_f64(&buffer, 0, f64::NAN, CompareOp::Eq) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_f64_nan_equality_ieee",
                start.elapsed(),
                format!("Filter Eq failed: {}", e),
            )
        }
    };

    if ctx.device_row_count(&eq_result) != 0 {
        return TestResult::error(
            "test_f64_nan_equality_ieee",
            start.elapsed(),
            format!(
                "Expected 0 rows where val == NaN, got {} (NaN == NaN should be false per IEEE 754)",
                ctx.device_row_count(&eq_result)
            ),
        );
    }

    // Test NaN != NaN (should be TRUE per IEEE 754) - all 6 values should pass
    let ne_result = match ctx.provider.filter_f64(&buffer, 0, f64::NAN, CompareOp::Ne) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_f64_nan_equality_ieee",
                start.elapsed(),
                format!("Filter Ne failed: {}", e),
            )
        }
    };

    if ctx.device_row_count(&ne_result) != 6 {
        return TestResult::error(
            "test_f64_nan_equality_ieee",
            start.elapsed(),
            format!(
                "Expected 6 rows where val != NaN (all values), got {} (x != NaN should be true for all x per IEEE 754)",
                ctx.device_row_count(&ne_result)
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_f64_nan_equality_ieee",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_f64_nan_equality_ieee", start.elapsed())
}

/// Test 4: f64 NaN ordering comparisons use total ordering (NaN > everything).
///
/// Lt/Le/Gt/Ge use total ordering where NaN is greater than all other values.
fn test_f64_nan_ordering_total(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::F64)]);

    // Data: just NaN values
    let data: Vec<f64> = vec![f64::NAN, f64::NAN, f64::NAN];

    let buffer = match ctx
        .provider
        .create_buffer_from_slice::<f64>(&data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_f64_nan_ordering_total",
                start.elapsed(),
                format!("Failed to create buffer: {}", e),
            )
        }
    };

    // Test NaN < 0.0 (should be FALSE - NaN is greater than 0.0 in total ordering)
    let lt_result = match ctx.provider.filter_f64(&buffer, 0, 0.0, CompareOp::Lt) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_f64_nan_ordering_total",
                start.elapsed(),
                format!("Filter Lt failed: {}", e),
            )
        }
    };

    if ctx.device_row_count(&lt_result) != 0 {
        return TestResult::error(
            "test_f64_nan_ordering_total",
            start.elapsed(),
            format!("NaN < 0.0 should be false (NaN > everything in total ordering), but got {} matches", ctx.device_row_count(&lt_result)),
        );
    }

    // Test NaN > 0.0 (should be TRUE - NaN is greater than 0.0 in total ordering)
    let gt_result = match ctx.provider.filter_f64(&buffer, 0, 0.0, CompareOp::Gt) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_f64_nan_ordering_total",
                start.elapsed(),
                format!("Filter Gt failed: {}", e),
            )
        }
    };

    if ctx.device_row_count(&gt_result) != 3 {
        return TestResult::error(
            "test_f64_nan_ordering_total",
            start.elapsed(),
            format!("NaN > 0.0 should be true for all 3 NaN values (total ordering), but got {} matches", ctx.device_row_count(&gt_result)),
        );
    }

    // Test NaN <= 0.0 (should be FALSE - NaN is greater than 0.0 in total ordering)
    let le_result = match ctx.provider.filter_f64(&buffer, 0, 0.0, CompareOp::Le) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_f64_nan_ordering_total",
                start.elapsed(),
                format!("Filter Le failed: {}", e),
            )
        }
    };

    if ctx.device_row_count(&le_result) != 0 {
        return TestResult::error(
            "test_f64_nan_ordering_total",
            start.elapsed(),
            format!("NaN <= 0.0 should be false (NaN > everything in total ordering), but got {} matches", ctx.device_row_count(&le_result)),
        );
    }

    // Test NaN >= 0.0 (should be TRUE - NaN is greater than 0.0 in total ordering)
    let ge_result = match ctx.provider.filter_f64(&buffer, 0, 0.0, CompareOp::Ge) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_f64_nan_ordering_total",
                start.elapsed(),
                format!("Filter Ge failed: {}", e),
            )
        }
    };

    if ctx.device_row_count(&ge_result) != 3 {
        return TestResult::error(
            "test_f64_nan_ordering_total",
            start.elapsed(),
            format!("NaN >= 0.0 should be true for all 3 NaN values (total ordering), but got {} matches", ctx.device_row_count(&ge_result)),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_f64_nan_ordering_total",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_f64_nan_ordering_total", start.elapsed())
}

/// Test 5: f64 comprehensive total ordering test.
///
/// Verifies the complete total ordering: -Inf < -1.0 < -0.0 < +0.0 < 1.0 < +Inf < NaN
fn test_f64_total_ordering_comprehensive(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::F64)]);

    // Data covering the full ordering spectrum
    let data: Vec<f64> = vec![
        f64::NEG_INFINITY, // index 0
        -1.0,              // index 1
        -0.0,              // index 2
        0.0,               // index 3 (+0.0)
        1.0,               // index 4
        f64::INFINITY,     // index 5
        f64::NAN,          // index 6
    ];

    let buffer = match ctx
        .provider
        .create_buffer_from_slice::<f64>(&data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_f64_total_ordering_comprehensive",
                start.elapsed(),
                format!("Failed to create buffer: {}", e),
            )
        }
    };

    // Test: values < +0.0 should include -Inf, -1.0, -0.0 = 3 values
    let lt_zero = match ctx.provider.filter_f64(&buffer, 0, 0.0, CompareOp::Lt) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_f64_total_ordering_comprehensive",
                start.elapsed(),
                format!("Filter Lt failed: {}", e),
            )
        }
    };

    if ctx.device_row_count(&lt_zero) != 3 {
        return TestResult::error(
            "test_f64_total_ordering_comprehensive",
            start.elapsed(),
            format!(
                "Expected 3 values < +0.0 (-Inf, -1.0, -0.0), got {}",
                ctx.device_row_count(&lt_zero)
            ),
        );
    }

    // Test: values > +Inf should include only NaN = 1 value
    let gt_inf = match ctx
        .provider
        .filter_f64(&buffer, 0, f64::INFINITY, CompareOp::Gt)
    {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_f64_total_ordering_comprehensive",
                start.elapsed(),
                format!("Filter Gt Inf failed: {}", e),
            )
        }
    };

    if ctx.device_row_count(&gt_inf) != 1 {
        return TestResult::error(
            "test_f64_total_ordering_comprehensive",
            start.elapsed(),
            format!(
                "Expected 1 value > +Inf (NaN), got {}",
                ctx.device_row_count(&gt_inf)
            ),
        );
    }

    // Test: values >= NaN should include only NaN = 1 value
    let ge_nan = match ctx.provider.filter_f64(&buffer, 0, f64::NAN, CompareOp::Ge) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_f64_total_ordering_comprehensive",
                start.elapsed(),
                format!("Filter Ge NaN failed: {}", e),
            )
        }
    };

    if ctx.device_row_count(&ge_nan) != 1 {
        return TestResult::error(
            "test_f64_total_ordering_comprehensive",
            start.elapsed(),
            format!(
                "Expected 1 value >= NaN (only NaN itself), got {}",
                ctx.device_row_count(&ge_nan)
            ),
        );
    }

    // Test: values <= -Inf should include only -Inf = 1 value
    let le_neg_inf = match ctx
        .provider
        .filter_f64(&buffer, 0, f64::NEG_INFINITY, CompareOp::Le)
    {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_f64_total_ordering_comprehensive",
                start.elapsed(),
                format!("Filter Le -Inf failed: {}", e),
            )
        }
    };

    if ctx.device_row_count(&le_neg_inf) != 1 {
        return TestResult::error(
            "test_f64_total_ordering_comprehensive",
            start.elapsed(),
            format!(
                "Expected 1 value <= -Inf (only -Inf itself), got {}",
                ctx.device_row_count(&le_neg_inf)
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_f64_total_ordering_comprehensive",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_f64_total_ordering_comprehensive", start.elapsed())
}

/// Test 6: f32 NaN > Infinity should return TRUE (total ordering).
///
/// Under total ordering, NaN is greater than all other values including Infinity.
fn test_f32_nan_greater_than_inf(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::F32)]);

    // Data with NaN and Infinity values
    let data: Vec<f32> = vec![
        f32::NAN,          // index 0: should pass (NaN > Inf)
        f32::INFINITY,     // index 1: should NOT pass (Inf > Inf is false)
        f32::NEG_INFINITY, // index 2: should NOT pass
        f32::NAN,          // index 3: should pass (NaN > Inf)
        1.0f32,            // index 4: should NOT pass
        f32::INFINITY,     // index 5: should NOT pass
    ];

    let buffer = match ctx
        .provider
        .create_buffer_from_slice::<f32>(&data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_f32_nan_greater_than_inf",
                start.elapsed(),
                format!("Failed to create buffer: {}", e),
            )
        }
    };

    // Filter for values > INFINITY using CompareOp::Gt
    // Under total ordering, NaN > INFINITY is TRUE
    // So 2 NaN values should pass this filter
    let filtered = match ctx
        .provider
        .filter_f32(&buffer, 0, f32::INFINITY, CompareOp::Gt)
    {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_f32_nan_greater_than_inf",
                start.elapsed(),
                format!("Filter failed: {}", e),
            )
        }
    };

    // 2 NaN values should be greater than INFINITY under total ordering
    if ctx.device_row_count(&filtered) != 2 {
        return TestResult::error(
            "test_f32_nan_greater_than_inf",
            start.elapsed(),
            format!(
                "Expected 2 rows where val > INFINITY (the NaN values), got {} (NaN > Inf should be true per total ordering)",
                ctx.device_row_count(&filtered)
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_f32_nan_greater_than_inf",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_f32_nan_greater_than_inf", start.elapsed())
}

/// Test 7: f32 -0.0 < +0.0 should return TRUE (total ordering).
///
/// Under total ordering, -0.0 is less than +0.0 (unlike IEEE 754 where they're equal).
fn test_f32_negative_zero_less_than_positive_zero(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::F32)]);

    // Data with negative and positive zeros
    let data: Vec<f32> = vec![
        -0.0f32, // index 0: should pass (-0.0 < +0.0)
        0.0f32,  // index 1: should NOT pass (+0.0 < +0.0 is false)
        -0.0f32, // index 2: should pass (-0.0 < +0.0)
        0.0f32,  // index 3: should NOT pass
        1.0f32,  // index 4: should NOT pass
        -1.0f32, // index 5: should pass (-1.0 < +0.0)
    ];

    let buffer = match ctx
        .provider
        .create_buffer_from_slice::<f32>(&data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_f32_negative_zero_less_than_positive_zero",
                start.elapsed(),
                format!("Failed to create buffer: {}", e),
            )
        }
    };

    // Filter for values < 0.0 (which is +0.0)
    // Under total ordering, -0.0 < +0.0 is TRUE
    // So -0.0 (x2) and -1.0 should pass = 3 values
    let filtered = match ctx.provider.filter_f32(&buffer, 0, 0.0f32, CompareOp::Lt) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_f32_negative_zero_less_than_positive_zero",
                start.elapsed(),
                format!("Filter failed: {}", e),
            )
        }
    };

    // -0.0 (x2) and -1.0 should pass = 3 values
    let expected_count = 3;
    if ctx.device_row_count(&filtered) as usize != expected_count {
        return TestResult::error(
            "test_f32_negative_zero_less_than_positive_zero",
            start.elapsed(),
            format!(
                "Expected {} rows where val < 0.0, got {} (-0.0 < +0.0 should be true per total ordering)",
                expected_count,
                ctx.device_row_count(&filtered)
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_f32_negative_zero_less_than_positive_zero",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed(
        "test_f32_negative_zero_less_than_positive_zero",
        start.elapsed(),
    )
}

/// Test 8: f32 NaN == NaN should return FALSE, NaN != NaN should return TRUE (IEEE 754).
///
/// Equality comparisons use IEEE 754 semantics where NaN is not equal to anything.
fn test_f32_nan_equality_ieee(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::F32)]);

    // Data with NaN values
    let data: Vec<f32> = vec![f32::NAN, 1.0f32, f32::NAN, 2.0f32, f32::NAN, 3.0f32];

    let buffer = match ctx
        .provider
        .create_buffer_from_slice::<f32>(&data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_f32_nan_equality_ieee",
                start.elapsed(),
                format!("Failed to create buffer: {}", e),
            )
        }
    };

    // Test NaN == NaN (should be FALSE per IEEE 754)
    let eq_result = match ctx.provider.filter_f32(&buffer, 0, f32::NAN, CompareOp::Eq) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_f32_nan_equality_ieee",
                start.elapsed(),
                format!("Filter Eq failed: {}", e),
            )
        }
    };

    if ctx.device_row_count(&eq_result) != 0 {
        return TestResult::error(
            "test_f32_nan_equality_ieee",
            start.elapsed(),
            format!(
                "Expected 0 rows where val == NaN, got {} (NaN == NaN should be false per IEEE 754)",
                ctx.device_row_count(&eq_result)
            ),
        );
    }

    // Test NaN != NaN (should be TRUE per IEEE 754) - all 6 values should pass
    let ne_result = match ctx.provider.filter_f32(&buffer, 0, f32::NAN, CompareOp::Ne) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_f32_nan_equality_ieee",
                start.elapsed(),
                format!("Filter Ne failed: {}", e),
            )
        }
    };

    if ctx.device_row_count(&ne_result) != 6 {
        return TestResult::error(
            "test_f32_nan_equality_ieee",
            start.elapsed(),
            format!(
                "Expected 6 rows where val != NaN (all values), got {} (x != NaN should be true for all x per IEEE 754)",
                ctx.device_row_count(&ne_result)
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_f32_nan_equality_ieee",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_f32_nan_equality_ieee", start.elapsed())
}

/// Test 9: f32 NaN ordering comparisons use total ordering (NaN > everything).
///
/// Lt/Le/Gt/Ge use total ordering where NaN is greater than all other values.
fn test_f32_nan_ordering_total(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::F32)]);

    // Data: just NaN values
    let data: Vec<f32> = vec![f32::NAN, f32::NAN, f32::NAN];

    let buffer = match ctx
        .provider
        .create_buffer_from_slice::<f32>(&data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_f32_nan_ordering_total",
                start.elapsed(),
                format!("Failed to create buffer: {}", e),
            )
        }
    };

    // Test NaN < 0.0 (should be FALSE - NaN is greater than 0.0 in total ordering)
    let lt_result = match ctx.provider.filter_f32(&buffer, 0, 0.0f32, CompareOp::Lt) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_f32_nan_ordering_total",
                start.elapsed(),
                format!("Filter Lt failed: {}", e),
            )
        }
    };

    if ctx.device_row_count(&lt_result) != 0 {
        return TestResult::error(
            "test_f32_nan_ordering_total",
            start.elapsed(),
            format!("NaN < 0.0 should be false (NaN > everything in total ordering), but got {} matches", ctx.device_row_count(&lt_result)),
        );
    }

    // Test NaN > 0.0 (should be TRUE - NaN is greater than 0.0 in total ordering)
    let gt_result = match ctx.provider.filter_f32(&buffer, 0, 0.0f32, CompareOp::Gt) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_f32_nan_ordering_total",
                start.elapsed(),
                format!("Filter Gt failed: {}", e),
            )
        }
    };

    if ctx.device_row_count(&gt_result) != 3 {
        return TestResult::error(
            "test_f32_nan_ordering_total",
            start.elapsed(),
            format!("NaN > 0.0 should be true for all 3 NaN values (total ordering), but got {} matches", ctx.device_row_count(&gt_result)),
        );
    }

    // Test NaN <= 0.0 (should be FALSE - NaN is greater than 0.0 in total ordering)
    let le_result = match ctx.provider.filter_f32(&buffer, 0, 0.0f32, CompareOp::Le) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_f32_nan_ordering_total",
                start.elapsed(),
                format!("Filter Le failed: {}", e),
            )
        }
    };

    if ctx.device_row_count(&le_result) != 0 {
        return TestResult::error(
            "test_f32_nan_ordering_total",
            start.elapsed(),
            format!("NaN <= 0.0 should be false (NaN > everything in total ordering), but got {} matches", ctx.device_row_count(&le_result)),
        );
    }

    // Test NaN >= 0.0 (should be TRUE - NaN is greater than 0.0 in total ordering)
    let ge_result = match ctx.provider.filter_f32(&buffer, 0, 0.0f32, CompareOp::Ge) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_f32_nan_ordering_total",
                start.elapsed(),
                format!("Filter Ge failed: {}", e),
            )
        }
    };

    if ctx.device_row_count(&ge_result) != 3 {
        return TestResult::error(
            "test_f32_nan_ordering_total",
            start.elapsed(),
            format!("NaN >= 0.0 should be true for all 3 NaN values (total ordering), but got {} matches", ctx.device_row_count(&ge_result)),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_f32_nan_ordering_total",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_f32_nan_ordering_total", start.elapsed())
}

/// Test 10: f32 comprehensive total ordering test.
///
/// Verifies the complete total ordering: -Inf < -1.0 < -0.0 < +0.0 < 1.0 < +Inf < NaN
fn test_f32_total_ordering_comprehensive(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::F32)]);

    // Data covering the full ordering spectrum
    let data: Vec<f32> = vec![
        f32::NEG_INFINITY, // index 0
        -1.0f32,           // index 1
        -0.0f32,           // index 2
        0.0f32,            // index 3 (+0.0)
        1.0f32,            // index 4
        f32::INFINITY,     // index 5
        f32::NAN,          // index 6
    ];

    let buffer = match ctx
        .provider
        .create_buffer_from_slice::<f32>(&data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_f32_total_ordering_comprehensive",
                start.elapsed(),
                format!("Failed to create buffer: {}", e),
            )
        }
    };

    // Test: values < +0.0 should include -Inf, -1.0, -0.0 = 3 values
    let lt_zero = match ctx.provider.filter_f32(&buffer, 0, 0.0f32, CompareOp::Lt) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_f32_total_ordering_comprehensive",
                start.elapsed(),
                format!("Filter Lt failed: {}", e),
            )
        }
    };

    if ctx.device_row_count(&lt_zero) != 3 {
        return TestResult::error(
            "test_f32_total_ordering_comprehensive",
            start.elapsed(),
            format!(
                "Expected 3 values < +0.0 (-Inf, -1.0, -0.0), got {}",
                ctx.device_row_count(&lt_zero)
            ),
        );
    }

    // Test: values > +Inf should include only NaN = 1 value
    let gt_inf = match ctx
        .provider
        .filter_f32(&buffer, 0, f32::INFINITY, CompareOp::Gt)
    {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_f32_total_ordering_comprehensive",
                start.elapsed(),
                format!("Filter Gt Inf failed: {}", e),
            )
        }
    };

    if ctx.device_row_count(&gt_inf) != 1 {
        return TestResult::error(
            "test_f32_total_ordering_comprehensive",
            start.elapsed(),
            format!(
                "Expected 1 value > +Inf (NaN), got {}",
                ctx.device_row_count(&gt_inf)
            ),
        );
    }

    // Test: values >= NaN should include only NaN = 1 value
    let ge_nan = match ctx.provider.filter_f32(&buffer, 0, f32::NAN, CompareOp::Ge) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_f32_total_ordering_comprehensive",
                start.elapsed(),
                format!("Filter Ge NaN failed: {}", e),
            )
        }
    };

    if ctx.device_row_count(&ge_nan) != 1 {
        return TestResult::error(
            "test_f32_total_ordering_comprehensive",
            start.elapsed(),
            format!(
                "Expected 1 value >= NaN (only NaN itself), got {}",
                ctx.device_row_count(&ge_nan)
            ),
        );
    }

    // Test: values <= -Inf should include only -Inf = 1 value
    let le_neg_inf = match ctx
        .provider
        .filter_f32(&buffer, 0, f32::NEG_INFINITY, CompareOp::Le)
    {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_f32_total_ordering_comprehensive",
                start.elapsed(),
                format!("Filter Le -Inf failed: {}", e),
            )
        }
    };

    if ctx.device_row_count(&le_neg_inf) != 1 {
        return TestResult::error(
            "test_f32_total_ordering_comprehensive",
            start.elapsed(),
            format!(
                "Expected 1 value <= -Inf (only -Inf itself), got {}",
                ctx.device_row_count(&le_neg_inf)
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_f32_total_ordering_comprehensive",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_f32_total_ordering_comprehensive", start.elapsed())
}
