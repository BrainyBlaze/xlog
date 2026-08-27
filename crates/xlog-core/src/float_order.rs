//! Canonical IEEE 754 total-order keys for relational floating-point values.

/// Convert raw `f32` bits into the unsigned key used by XLOG's total order.
///
/// The mapping is bijective over all bit patterns and orders values exactly as
/// [`f32::total_cmp`]. Consequently, negative and positive zero are distinct,
/// as are NaN sign, signaling state, and payload.
pub const fn f32_total_order_key_from_bits(bits: u32) -> u32 {
    let mask = if bits >> 31 == 0 {
        0x8000_0000
    } else {
        u32::MAX
    };
    bits ^ mask
}

/// Convert an `f32` value into the unsigned key used by XLOG's total order.
pub const fn f32_total_order_key(value: f32) -> u32 {
    f32_total_order_key_from_bits(value.to_bits())
}

/// Convert raw `f64` bits into the unsigned key used by XLOG's total order.
///
/// The mapping is bijective over all bit patterns and orders values exactly as
/// [`f64::total_cmp`]. Consequently, negative and positive zero are distinct,
/// as are NaN sign, signaling state, and payload.
pub const fn f64_total_order_key_from_bits(bits: u64) -> u64 {
    let mask = if bits >> 63 == 0 {
        0x8000_0000_0000_0000
    } else {
        u64::MAX
    };
    bits ^ mask
}

/// Convert an `f64` value into the unsigned key used by XLOG's total order.
pub const fn f64_total_order_key(value: f64) -> u64 {
    f64_total_order_key_from_bits(value.to_bits())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_edge_cases_match_standard_total_order() {
        let f32_values = [
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
        for &left in &f32_values {
            for &right in &f32_values {
                assert_eq!(
                    f32_total_order_key(left).cmp(&f32_total_order_key(right)),
                    left.total_cmp(&right)
                );
            }
        }

        let f64_values = [
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
        for &left in &f64_values {
            for &right in &f64_values {
                assert_eq!(
                    f64_total_order_key(left).cmp(&f64_total_order_key(right)),
                    left.total_cmp(&right)
                );
            }
        }
    }

    #[test]
    fn seeded_bit_patterns_match_standard_total_order() {
        let mut left = 0x243f_6a88_85a3_08d3_u64;
        let mut right = 0x1319_8a2e_0370_7344_u64;
        for _ in 0..10_000 {
            left ^= left << 13;
            left ^= left >> 7;
            left ^= left << 17;
            right = right
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);

            let left_f64 = f64::from_bits(left);
            let right_f64 = f64::from_bits(right);
            assert_eq!(
                f64_total_order_key(left_f64).cmp(&f64_total_order_key(right_f64)),
                left_f64.total_cmp(&right_f64)
            );

            let left_f32 = f32::from_bits(left as u32);
            let right_f32 = f32::from_bits(right as u32);
            assert_eq!(
                f32_total_order_key(left_f32).cmp(&f32_total_order_key(right_f32)),
                left_f32.total_cmp(&right_f32)
            );
        }
    }
}
