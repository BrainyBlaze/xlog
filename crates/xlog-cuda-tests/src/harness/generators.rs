//! Property-based data generators for CUDA certification tests.

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

/// Size generator for edge case testing.
pub(crate) struct SizeGen;

impl SizeGen {
    /// Standard edge case sizes covering block boundaries and common edge cases.
    pub(crate) fn edge_cases() -> Vec<usize> {
        vec![
            0, 1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511,
            512, 513, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097, 8191, 8192, 8193,
            16383, 16384, 16385, 32767, 32768, 32769, 65535, 65536, 65537,
        ]
    }

    /// Sizes near 32-bit overflow boundary.
    pub(crate) fn near_i32_max() -> Vec<usize> {
        vec![
            (i32::MAX as usize) - 2,
            (i32::MAX as usize) - 1,
            i32::MAX as usize,
        ]
    }

    /// Sizes that are prime numbers (worst case for some algorithms).
    pub(crate) fn primes() -> Vec<usize> {
        vec![
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
            89, 97, 101, 103, 107, 127, 131, 251, 257, 509, 521, 1021, 1031, 2039, 2053, 4093,
            4099, 8191, 8209, 16381, 16411, 32749, 32771, 65521, 65537,
        ]
    }

    /// Random sizes within a range using deterministic RNG.
    pub(crate) fn random(count: usize, min: usize, max: usize, seed: u64) -> Vec<usize> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..count).map(|_| rng.gen_range(min..=max)).collect()
    }

    /// Warp-related sizes (multiples of 32, and off-by-one).
    pub(crate) fn warp_related() -> Vec<usize> {
        vec![
            31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128, 129, 159, 160, 161, 191, 192, 193, 223,
            224, 225, 255, 256, 257, 287, 288, 289,
        ]
    }

    /// Block-related sizes (multiples of 256, and off-by-one).
    pub(crate) fn block_related() -> Vec<usize> {
        vec![
            255, 256, 257, 511, 512, 513, 767, 768, 769, 1023, 1024, 1025, 1279, 1280, 1281, 1535,
            1536, 1537,
        ]
    }
}

/// Value distribution patterns for test data generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Distribution {
    AllEqual,
    AllUnique,
    Sorted,
    ReverseSorted,
    Alternating,
    AdversarialHash,
    Random,
    HalfAndHalf,
    Sparse,
    Dense,
}

impl Distribution {
    /// Get all distribution variants.
    pub(crate) fn all() -> Vec<Distribution> {
        vec![
            Distribution::AllEqual,
            Distribution::AllUnique,
            Distribution::Sorted,
            Distribution::ReverseSorted,
            Distribution::Alternating,
            Distribution::AdversarialHash,
            Distribution::Random,
        ]
    }

    /// Generate u32 values with this distribution.
    pub(crate) fn generate_u32(&self, count: usize, seed: u64) -> Vec<u32> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        match self {
            Distribution::AllEqual => vec![42u32; count],
            Distribution::AllUnique => (0..count as u32).collect(),
            Distribution::Sorted => (0..count as u32).collect(),
            Distribution::ReverseSorted => (0..count as u32).rev().collect(),
            Distribution::Alternating => (0..count)
                .map(|i| if i % 2 == 0 { 0 } else { u32::MAX })
                .collect(),
            Distribution::AdversarialHash => (0..count).map(|i| (i as u32) * 256).collect(),
            Distribution::Random => (0..count).map(|_| rng.gen()).collect(),
            Distribution::HalfAndHalf => (0..count)
                .map(|i| if i < count / 2 { 0 } else { 1 })
                .collect(),
            Distribution::Sparse => (0..count)
                .map(|i| if i % 10 == 0 { 1 } else { 0 })
                .collect(),
            Distribution::Dense => (0..count)
                .map(|i| if i % 10 == 0 { 0 } else { 1 })
                .collect(),
        }
    }

    /// Generate i64 values with this distribution.
    pub(crate) fn generate_i64(&self, count: usize, seed: u64) -> Vec<i64> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        match self {
            Distribution::AllEqual => vec![42i64; count],
            Distribution::AllUnique => (0..count as i64).collect(),
            Distribution::Sorted => (0..count as i64).collect(),
            Distribution::ReverseSorted => (0..count as i64).rev().collect(),
            Distribution::Alternating => (0..count)
                .map(|i| if i % 2 == 0 { i64::MIN } else { i64::MAX })
                .collect(),
            Distribution::AdversarialHash => (0..count).map(|i| (i as i64) * 256).collect(),
            Distribution::Random => (0..count).map(|_| rng.gen()).collect(),
            Distribution::HalfAndHalf => (0..count)
                .map(|i| if i < count / 2 { -1 } else { 1 })
                .collect(),
            Distribution::Sparse => (0..count)
                .map(|i| if i % 10 == 0 { 1 } else { 0 })
                .collect(),
            Distribution::Dense => (0..count)
                .map(|i| if i % 10 == 0 { 0 } else { 1 })
                .collect(),
        }
    }

    /// Generate f64 values with this distribution.
    pub(crate) fn generate_f64(&self, count: usize, seed: u64) -> Vec<f64> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        match self {
            Distribution::AllEqual => vec![42.0f64; count],
            Distribution::AllUnique => (0..count).map(|i| i as f64).collect(),
            Distribution::Sorted => (0..count).map(|i| i as f64).collect(),
            Distribution::ReverseSorted => (0..count).map(|i| (count - 1 - i) as f64).collect(),
            Distribution::Alternating => (0..count)
                .map(|i| if i % 2 == 0 { f64::MIN } else { f64::MAX })
                .collect(),
            Distribution::AdversarialHash => (0..count).map(|i| (i as f64) * 256.0).collect(),
            Distribution::Random => (0..count).map(|_| rng.gen_range(-1e10..1e10)).collect(),
            Distribution::HalfAndHalf => (0..count)
                .map(|i| if i < count / 2 { -1.0 } else { 1.0 })
                .collect(),
            Distribution::Sparse => (0..count)
                .map(|i| if i % 10 == 0 { 1.0 } else { 0.0 })
                .collect(),
            Distribution::Dense => (0..count)
                .map(|i| if i % 10 == 0 { 0.0 } else { 1.0 })
                .collect(),
        }
    }

    /// Generate u8 mask values with this distribution.
    pub(crate) fn generate_mask(&self, count: usize, seed: u64) -> Vec<u8> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        match self {
            Distribution::AllEqual => vec![1u8; count],
            Distribution::AllUnique => (0..count).map(|i| (i % 2) as u8).collect(),
            Distribution::Sorted => (0..count)
                .map(|i| if i < count / 2 { 0 } else { 1 })
                .collect(),
            Distribution::ReverseSorted => (0..count)
                .map(|i| if i < count / 2 { 1 } else { 0 })
                .collect(),
            Distribution::Alternating => (0..count).map(|i| (i % 2) as u8).collect(),
            Distribution::AdversarialHash => (0..count).map(|i| (i % 2) as u8).collect(),
            Distribution::Random => (0..count)
                .map(|_| if rng.gen_bool(0.5) { 1 } else { 0 })
                .collect(),
            Distribution::HalfAndHalf => (0..count)
                .map(|i| if i < count / 2 { 0 } else { 1 })
                .collect(),
            Distribution::Sparse => (0..count)
                .map(|i| if i % 10 == 0 { 1 } else { 0 })
                .collect(),
            Distribution::Dense => (0..count)
                .map(|i| if i % 10 == 0 { 0 } else { 1 })
                .collect(),
        }
    }
}

/// Numeric edge value generators for specific types.
pub(crate) struct NumericEdges;

impl NumericEdges {
    pub(crate) fn u32_edges() -> Vec<u32> {
        vec![0, 1, 2, u32::MAX - 1, u32::MAX]
    }

    pub(crate) fn i32_edges() -> Vec<i32> {
        vec![i32::MIN, i32::MIN + 1, -1, 0, 1, i32::MAX - 1, i32::MAX]
    }

    pub(crate) fn u64_edges() -> Vec<u64> {
        vec![0, 1, 2, u64::MAX - 1, u64::MAX]
    }

    pub(crate) fn i64_edges() -> Vec<i64> {
        vec![i64::MIN, i64::MIN + 1, -1, 0, 1, i64::MAX - 1, i64::MAX]
    }

    pub(crate) fn f32_edges() -> Vec<f32> {
        vec![
            f32::NEG_INFINITY,
            f32::MIN,
            -1.0,
            -f32::MIN_POSITIVE,
            -0.0,
            0.0,
            f32::MIN_POSITIVE,
            1.0,
            f32::MAX,
            f32::INFINITY,
            f32::NAN,
        ]
    }

    pub(crate) fn f64_edges() -> Vec<f64> {
        vec![
            f64::NEG_INFINITY,
            f64::MIN,
            -1.0,
            -f64::MIN_POSITIVE,
            -0.0,
            0.0,
            f64::MIN_POSITIVE,
            1.0,
            f64::MAX,
            f64::INFINITY,
            f64::NAN,
        ]
    }

    pub(crate) fn f64_subnormals() -> Vec<f64> {
        vec![
            5e-324,
            1e-323,
            1e-320,
            1e-310,
            f64::MIN_POSITIVE / 2.0,
            -5e-324,
            -1e-323,
            -f64::MIN_POSITIVE / 2.0,
        ]
    }

    pub(crate) fn f64_fma_stress() -> Vec<(f64, f64, f64)> {
        vec![
            (1.0 + 1e-15, 1.0 - 1e-15, 1e-30),
            (1e100, 1e100, 1e-100),
            (1e-100, 1e-100, 1e-200),
            (f64::MAX / 2.0, 2.0, -f64::MAX),
        ]
    }
}

/// Alignment generator for memory access testing.
pub(crate) struct AlignmentGen;

impl AlignmentGen {
    pub(crate) fn offsets() -> Vec<(usize, &'static str)> {
        vec![
            (0, "aligned"),
            (1, "off-by-1"),
            (2, "off-by-2"),
            (3, "off-by-3"),
            (4, "off-by-4"),
            (5, "off-by-5"),
            (6, "off-by-6"),
            (7, "off-by-7"),
        ]
    }

    pub(crate) fn violating_offsets(required_alignment: usize) -> Vec<usize> {
        (1..required_alignment).collect()
    }
}

/// Key distribution generator for hash table and join testing.
pub(crate) struct KeyDistribution;

impl KeyDistribution {
    /// Generate key pairs for join testing. Returns (left_keys, right_keys, expected_match_count).
    pub(crate) fn join_test_data(
        size: usize,
        overlap_ratio: f64,
        seed: u64,
    ) -> (Vec<u32>, Vec<u32>, usize) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let overlap_size = (size as f64 * overlap_ratio) as usize;
        let common: Vec<u32> = (0..overlap_size as u32).collect();
        let left_only: Vec<u32> = (overlap_size as u32..(size as u32)).collect();
        let right_only: Vec<u32> = ((size + overlap_size) as u32..((2 * size) as u32)).collect();

        let mut left = common.clone();
        left.extend(left_only);
        left.shuffle(&mut rng);

        let mut right = common.clone();
        right.extend(right_only);
        right.shuffle(&mut rng);

        (left, right, overlap_size)
    }

    /// Generate keys with many hash collisions.
    pub(crate) fn collision_keys(count: usize, collision_factor: usize) -> Vec<u32> {
        (0..count).map(|i| (i * collision_factor) as u32).collect()
    }

    /// Generate keys for groupby testing with specified group count.
    pub(crate) fn groupby_keys(total_rows: usize, num_groups: usize, seed: u64) -> Vec<u32> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..total_rows)
            .map(|_| rng.gen_range(0..num_groups as u32))
            .collect()
    }
}
