mod common;

use common::setup_provider;
use xlog_core::{f32_total_order_key, f64_total_order_key, ScalarType, Schema};
use xlog_cuda::{CompareOp, CudaKernelProvider};

fn comparison_matches<K: Ord>(left: K, right: K, op: CompareOp) -> bool {
    match op {
        CompareOp::Eq => left == right,
        CompareOp::Ne => left != right,
        CompareOp::Lt => left < right,
        CompareOp::Le => left <= right,
        CompareOp::Gt => left > right,
        CompareOp::Ge => left >= right,
    }
}

fn f64_columns(
    provider: &CudaKernelProvider,
    left: &[f64],
    right: &[f64],
) -> xlog_core::Result<xlog_cuda::CudaBuffer> {
    let left_bytes: Vec<u8> = left.iter().flat_map(|value| value.to_le_bytes()).collect();
    let right_bytes: Vec<u8> = right.iter().flat_map(|value| value.to_le_bytes()).collect();
    provider.create_buffer_from_slices(
        &[&left_bytes, &right_bytes],
        Schema::new(vec![
            ("left".into(), ScalarType::F64),
            ("right".into(), ScalarType::F64),
        ]),
    )
}

fn f32_columns(
    provider: &CudaKernelProvider,
    left: &[f32],
    right: &[f32],
) -> xlog_core::Result<xlog_cuda::CudaBuffer> {
    let left_bytes: Vec<u8> = left.iter().flat_map(|value| value.to_le_bytes()).collect();
    let right_bytes: Vec<u8> = right.iter().flat_map(|value| value.to_le_bytes()).collect();
    provider.create_buffer_from_slices(
        &[&left_bytes, &right_bytes],
        Schema::new(vec![
            ("left".into(), ScalarType::F32),
            ("right".into(), ScalarType::F32),
        ]),
    )
}

fn f64_cases() -> (Vec<f64>, Vec<f64>) {
    let edges = [
        f64::from_bits(0xfff8_0000_0000_0002),
        f64::NEG_INFINITY,
        -1.0,
        -0.0,
        0.0,
        1.0,
        f64::INFINITY,
        f64::from_bits(0x7ff8_0000_0000_0001),
        f64::from_bits(0x7ff8_0000_0000_0002),
    ];
    let mut left = Vec::new();
    let mut right = Vec::new();
    for &left_value in &edges {
        for &right_value in &edges {
            left.push(left_value);
            right.push(right_value);
        }
    }

    let mut left_bits = 0x243f_6a88_85a3_08d3_u64;
    let mut right_bits = 0x1319_8a2e_0370_7344_u64;
    for _ in 0..512 {
        left_bits ^= left_bits << 13;
        left_bits ^= left_bits >> 7;
        left_bits ^= left_bits << 17;
        right_bits = right_bits
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        left.push(f64::from_bits(left_bits));
        right.push(f64::from_bits(right_bits));
    }
    (left, right)
}

fn f32_cases() -> (Vec<f32>, Vec<f32>) {
    let edges = [
        f32::from_bits(0xffc0_0002),
        f32::NEG_INFINITY,
        -1.0,
        -0.0,
        0.0,
        1.0,
        f32::INFINITY,
        f32::from_bits(0x7fc0_0001),
        f32::from_bits(0x7fc0_0002),
    ];
    let mut left = Vec::new();
    let mut right = Vec::new();
    for &left_value in &edges {
        for &right_value in &edges {
            left.push(left_value);
            right.push(right_value);
        }
    }

    let mut left_bits = 0x243f_6a88_u32;
    let mut right_bits = 0x85a3_08d3_u32;
    for _ in 0..512 {
        left_bits ^= left_bits << 13;
        left_bits ^= left_bits >> 17;
        left_bits ^= left_bits << 5;
        right_bits = right_bits
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        left.push(f32::from_bits(left_bits));
        right.push(f32::from_bits(right_bits));
    }
    (left, right)
}

#[test]
fn f64_comparisons_and_selections_match_host_total_order() {
    let Some(provider) = setup_provider() else {
        return;
    };
    let (left, right) = f64_cases();
    let input = f64_columns(provider.as_ref(), &left, &right).expect("f64 input");

    for op in [
        CompareOp::Eq,
        CompareOp::Ne,
        CompareOp::Lt,
        CompareOp::Le,
        CompareOp::Gt,
        CompareOp::Ge,
    ] {
        let mask = provider
            .compare_columns::<f64>(&input, 0, 1, op)
            .expect("f64 comparison");
        let filtered = provider
            .filter_by_device_mask(&input, &mask)
            .expect("f64 comparison compaction");
        let actual_left = provider
            .download_column::<f64>(&filtered, 0)
            .expect("f64 left result");
        let actual_right = provider
            .download_column::<f64>(&filtered, 1)
            .expect("f64 right result");
        let expected: Vec<(u64, u64)> = left
            .iter()
            .zip(&right)
            .filter(|(left, right)| {
                comparison_matches(
                    f64_total_order_key(**left),
                    f64_total_order_key(**right),
                    op,
                )
            })
            .map(|(left, right)| (left.to_bits(), right.to_bits()))
            .collect();
        let actual: Vec<(u64, u64)> = actual_left
            .iter()
            .zip(&actual_right)
            .map(|(left, right)| (left.to_bits(), right.to_bits()))
            .collect();
        assert_eq!(actual, expected, "f64 comparison mismatch for {op:?}");
    }

    let left_buffer = provider
        .create_buffer_from_slice::<f64>(
            &left,
            Schema::new(vec![("value".into(), ScalarType::F64)]),
        )
        .expect("f64 left buffer");
    let right_buffer = provider
        .create_buffer_from_slice::<f64>(
            &right,
            Schema::new(vec![("value".into(), ScalarType::F64)]),
        )
        .expect("f64 right buffer");
    for (actual, select_minimum) in [
        (provider.min_columns(&left_buffer, &right_buffer), true),
        (provider.max_columns(&left_buffer, &right_buffer), false),
    ] {
        let actual = provider
            .download_column::<f64>(&actual.expect("f64 selection"), 0)
            .expect("f64 selection result");
        let expected: Vec<u64> = left
            .iter()
            .zip(&right)
            .map(|(&left, &right)| {
                let left_key = f64_total_order_key(left);
                let right_key = f64_total_order_key(right);
                if (select_minimum && left_key <= right_key)
                    || (!select_minimum && left_key >= right_key)
                {
                    left.to_bits()
                } else {
                    right.to_bits()
                }
            })
            .collect();
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
        );
    }
}

#[test]
fn f32_comparisons_and_selections_match_host_total_order() {
    let Some(provider) = setup_provider() else {
        return;
    };
    let (left, right) = f32_cases();
    let input = f32_columns(provider.as_ref(), &left, &right).expect("f32 input");

    for op in [
        CompareOp::Eq,
        CompareOp::Ne,
        CompareOp::Lt,
        CompareOp::Le,
        CompareOp::Gt,
        CompareOp::Ge,
    ] {
        let mask = provider
            .compare_columns::<f32>(&input, 0, 1, op)
            .expect("f32 comparison");
        let filtered = provider
            .filter_by_device_mask(&input, &mask)
            .expect("f32 comparison compaction");
        let actual_left = provider
            .download_column::<f32>(&filtered, 0)
            .expect("f32 left result");
        let actual_right = provider
            .download_column::<f32>(&filtered, 1)
            .expect("f32 right result");
        let expected: Vec<(u32, u32)> = left
            .iter()
            .zip(&right)
            .filter(|(left, right)| {
                comparison_matches(
                    f32_total_order_key(**left),
                    f32_total_order_key(**right),
                    op,
                )
            })
            .map(|(left, right)| (left.to_bits(), right.to_bits()))
            .collect();
        let actual: Vec<(u32, u32)> = actual_left
            .iter()
            .zip(&actual_right)
            .map(|(left, right)| (left.to_bits(), right.to_bits()))
            .collect();
        assert_eq!(actual, expected, "f32 comparison mismatch for {op:?}");
    }

    let left_buffer = provider
        .create_buffer_from_slice::<f32>(
            &left,
            Schema::new(vec![("value".into(), ScalarType::F32)]),
        )
        .expect("f32 left buffer");
    let right_buffer = provider
        .create_buffer_from_slice::<f32>(
            &right,
            Schema::new(vec![("value".into(), ScalarType::F32)]),
        )
        .expect("f32 right buffer");
    for (actual, select_minimum) in [
        (provider.min_columns(&left_buffer, &right_buffer), true),
        (provider.max_columns(&left_buffer, &right_buffer), false),
    ] {
        let actual = provider
            .download_column::<f32>(&actual.expect("f32 selection"), 0)
            .expect("f32 selection result");
        let expected: Vec<u32> = left
            .iter()
            .zip(&right)
            .map(|(&left, &right)| {
                let left_key = f32_total_order_key(left);
                let right_key = f32_total_order_key(right);
                if (select_minimum && left_key <= right_key)
                    || (!select_minimum && left_key >= right_key)
                {
                    left.to_bits()
                } else {
                    right.to_bits()
                }
            })
            .collect();
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
        );
    }
}
