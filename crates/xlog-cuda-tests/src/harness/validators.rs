//! CPU reference implementations for validating GPU results.

/// CPU reference implementations for all kernel operations.
pub mod reference {
    use std::collections::HashMap;

    /// Hash join returning (left_idx, right_idx) pairs.
    pub fn hash_join_u32(left: &[u32], right: &[u32]) -> Vec<(usize, usize)> {
        let mut result = Vec::new();
        let right_map: HashMap<u32, Vec<usize>> = right.iter()
            .enumerate()
            .fold(HashMap::new(), |mut acc, (idx, val)| {
                acc.entry(*val).or_default().push(idx);
                acc
            });

        for (left_idx, left_val) in left.iter().enumerate() {
            if let Some(right_indices) = right_map.get(left_val) {
                for &right_idx in right_indices {
                    result.push((left_idx, right_idx));
                }
            }
        }
        result
    }

    /// Multi-column hash join.
    pub fn hash_join_multi(left_cols: &[&[u32]], right_cols: &[&[u32]]) -> Vec<(usize, usize)> {
        if left_cols.is_empty() || right_cols.is_empty() {
            return Vec::new();
        }

        let left_len = left_cols[0].len();
        let right_len = right_cols[0].len();

        // Build hash map on right side
        let mut right_map: HashMap<Vec<u32>, Vec<usize>> = HashMap::new();
        for i in 0..right_len {
            let key: Vec<u32> = right_cols.iter().map(|col| col[i]).collect();
            right_map.entry(key).or_default().push(i);
        }

        // Probe with left side
        let mut result = Vec::new();
        for i in 0..left_len {
            let key: Vec<u32> = left_cols.iter().map(|col| col[i]).collect();
            if let Some(right_indices) = right_map.get(&key) {
                for &right_idx in right_indices {
                    result.push((i, right_idx));
                }
            }
        }
        result
    }

    /// Semi join - returns left indices that have a match in right.
    pub fn semi_join_u32(left: &[u32], right: &[u32]) -> Vec<usize> {
        let right_set: std::collections::HashSet<u32> = right.iter().copied().collect();
        left.iter()
            .enumerate()
            .filter(|(_, val)| right_set.contains(val))
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Anti join - returns left indices that have NO match in right.
    pub fn anti_join_u32(left: &[u32], right: &[u32]) -> Vec<usize> {
        let right_set: std::collections::HashSet<u32> = right.iter().copied().collect();
        left.iter()
            .enumerate()
            .filter(|(_, val)| !right_set.contains(val))
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Filter by comparison operation.
    pub fn filter_compare_u32(data: &[u32], op: CompareOp, val: u32) -> Vec<usize> {
        data.iter()
            .enumerate()
            .filter(|(_, d)| op.apply(**d, val))
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Filter by comparison operation for i64.
    pub fn filter_compare_i64(data: &[i64], op: CompareOp, val: i64) -> Vec<usize> {
        data.iter()
            .enumerate()
            .filter(|(_, d)| op.apply(**d, val))
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Filter by comparison operation for f64.
    pub fn filter_compare_f64(data: &[f64], op: CompareOp, val: f64) -> Vec<f64> {
        data.iter()
            .filter(|d| op.apply_f64(**d, val))
            .copied()
            .collect()
    }

    /// Compact by mask.
    pub fn compact_by_mask<T: Copy>(data: &[T], mask: &[u8]) -> Vec<T> {
        data.iter()
            .zip(mask.iter())
            .filter(|(_, m)| **m != 0)
            .map(|(d, _)| *d)
            .collect()
    }

    /// Stable radix sort returning (sorted_data, permutation).
    pub fn radix_sort_u32(keys: &[u32]) -> (Vec<u32>, Vec<u32>) {
        let mut indexed: Vec<(u32, u32)> = keys.iter()
            .enumerate()
            .map(|(i, &k)| (k, i as u32))
            .collect();

        // Stable sort
        indexed.sort_by_key(|(k, _)| *k);

        let sorted: Vec<u32> = indexed.iter().map(|(k, _)| *k).collect();
        let perm: Vec<u32> = indexed.iter().map(|(_, i)| *i).collect();

        (sorted, perm)
    }

    /// Apply permutation to data.
    pub fn apply_permutation<T: Copy>(data: &[T], perm: &[u32]) -> Vec<T> {
        perm.iter().map(|&i| data[i as usize]).collect()
    }

    /// Inclusive prefix sum.
    pub fn inclusive_scan(data: &[u32]) -> Vec<u32> {
        let mut result = Vec::with_capacity(data.len());
        let mut sum = 0u32;
        for &val in data {
            sum = sum.wrapping_add(val);
            result.push(sum);
        }
        result
    }

    /// Exclusive prefix sum.
    pub fn exclusive_scan(data: &[u32]) -> Vec<u32> {
        let mut result = Vec::with_capacity(data.len());
        let mut sum = 0u32;
        for &val in data {
            result.push(sum);
            sum = sum.wrapping_add(val);
        }
        result
    }

    /// Group by count (assumes sorted input).
    pub fn groupby_count_sorted(keys: &[u32]) -> Vec<(u32, u64)> {
        if keys.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::new();
        let mut current_key = keys[0];
        let mut count = 1u64;

        for &key in &keys[1..] {
            if key == current_key {
                count += 1;
            } else {
                result.push((current_key, count));
                current_key = key;
                count = 1;
            }
        }
        result.push((current_key, count));
        result
    }

    /// Group by sum (assumes sorted input).
    pub fn groupby_sum_sorted(keys: &[u32], vals: &[u32]) -> Vec<(u32, u64)> {
        if keys.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::new();
        let mut current_key = keys[0];
        let mut sum = vals[0] as u64;

        for i in 1..keys.len() {
            if keys[i] == current_key {
                sum += vals[i] as u64;
            } else {
                result.push((current_key, sum));
                current_key = keys[i];
                sum = vals[i] as u64;
            }
        }
        result.push((current_key, sum));
        result
    }

    /// Group by min (assumes sorted input).
    pub fn groupby_min_sorted(keys: &[u32], vals: &[u32]) -> Vec<(u32, u32)> {
        if keys.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::new();
        let mut current_key = keys[0];
        let mut min_val = vals[0];

        for i in 1..keys.len() {
            if keys[i] == current_key {
                min_val = min_val.min(vals[i]);
            } else {
                result.push((current_key, min_val));
                current_key = keys[i];
                min_val = vals[i];
            }
        }
        result.push((current_key, min_val));
        result
    }

    /// Group by max (assumes sorted input).
    pub fn groupby_max_sorted(keys: &[u32], vals: &[u32]) -> Vec<(u32, u32)> {
        if keys.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::new();
        let mut current_key = keys[0];
        let mut max_val = vals[0];

        for i in 1..keys.len() {
            if keys[i] == current_key {
                max_val = max_val.max(vals[i]);
            } else {
                result.push((current_key, max_val));
                current_key = keys[i];
                max_val = vals[i];
            }
        }
        result.push((current_key, max_val));
        result
    }

    /// Dedup sorted data.
    pub fn dedup_sorted<T: Eq + Copy>(data: &[T]) -> Vec<T> {
        if data.is_empty() {
            return Vec::new();
        }

        let mut result = vec![data[0]];
        for &val in &data[1..] {
            if val != *result.last().unwrap() {
                result.push(val);
            }
        }
        result
    }

    /// Mark duplicates in sorted data (true = unique, false = duplicate).
    pub fn mark_duplicates<T: Eq>(sorted: &[T]) -> Vec<bool> {
        if sorted.is_empty() {
            return Vec::new();
        }

        let mut result = vec![true]; // First element is always unique
        for i in 1..sorted.len() {
            result.push(sorted[i] != sorted[i - 1]);
        }
        result
    }

    /// Sorted set union.
    pub fn sorted_union<T: Ord + Copy>(a: &[T], b: &[T]) -> Vec<T> {
        let mut result = Vec::new();
        let mut i = 0;
        let mut j = 0;

        while i < a.len() && j < b.len() {
            if a[i] < b[j] {
                if result.last() != Some(&a[i]) {
                    result.push(a[i]);
                }
                i += 1;
            } else if a[i] > b[j] {
                if result.last() != Some(&b[j]) {
                    result.push(b[j]);
                }
                j += 1;
            } else {
                if result.last() != Some(&a[i]) {
                    result.push(a[i]);
                }
                i += 1;
                j += 1;
            }
        }

        while i < a.len() {
            if result.last() != Some(&a[i]) {
                result.push(a[i]);
            }
            i += 1;
        }

        while j < b.len() {
            if result.last() != Some(&b[j]) {
                result.push(b[j]);
            }
            j += 1;
        }

        result
    }

    /// Sorted set difference (a - b).
    pub fn sorted_diff<T: Ord + Copy>(a: &[T], b: &[T]) -> Vec<T> {
        let mut result = Vec::new();
        let mut i = 0;
        let mut j = 0;

        while i < a.len() && j < b.len() {
            if a[i] < b[j] {
                result.push(a[i]);
                i += 1;
            } else if a[i] > b[j] {
                j += 1;
            } else {
                i += 1;
                j += 1;
            }
        }

        while i < a.len() {
            result.push(a[i]);
            i += 1;
        }

        result
    }

    /// Pack columns into row-major byte array.
    pub fn pack_keys(cols: &[&[u8]], col_sizes: &[usize], num_rows: usize) -> Vec<u8> {
        let row_size: usize = col_sizes.iter().sum();
        let mut result = vec![0u8; row_size * num_rows];

        for row in 0..num_rows {
            let mut offset = 0;
            for (col_idx, &col_size) in col_sizes.iter().enumerate() {
                let src_start = row * col_size;
                let src_end = src_start + col_size;
                let dst_start = row * row_size + offset;
                let dst_end = dst_start + col_size;
                result[dst_start..dst_end].copy_from_slice(&cols[col_idx][src_start..src_end]);
                offset += col_size;
            }
        }
        result
    }

    /// FNV-1a hash.
    pub fn hash_fnv1a(data: &[u8]) -> u32 {
        const FNV_PRIME: u32 = 16777619;
        const FNV_OFFSET: u32 = 2166136261;

        let mut hash = FNV_OFFSET;
        for &byte in data {
            hash ^= byte as u32;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    /// Unpack a column from packed rows.
    pub fn unpack_column(packed: &[u8], row_size: usize, col_offset: usize, col_size: usize, num_rows: usize) -> Vec<u8> {
        let mut result = vec![0u8; col_size * num_rows];

        for row in 0..num_rows {
            let src_start = row * row_size + col_offset;
            let src_end = src_start + col_size;
            let dst_start = row * col_size;
            let dst_end = dst_start + col_size;
            result[dst_start..dst_end].copy_from_slice(&packed[src_start..src_end]);
        }
        result
    }

    /// Comparison operation enum.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum CompareOp {
        Eq,
        Ne,
        Lt,
        Le,
        Gt,
        Ge,
    }

    impl CompareOp {
        pub fn apply<T: PartialOrd>(&self, a: T, b: T) -> bool {
            match self {
                CompareOp::Eq => a == b,
                CompareOp::Ne => a != b,
                CompareOp::Lt => a < b,
                CompareOp::Le => a <= b,
                CompareOp::Gt => a > b,
                CompareOp::Ge => a >= b,
            }
        }

        pub fn apply_f64(&self, a: f64, b: f64) -> bool {
            match self {
                CompareOp::Eq => a == b,
                CompareOp::Ne => a != b,
                CompareOp::Lt => a < b,
                CompareOp::Le => a <= b,
                CompareOp::Gt => a > b,
                CompareOp::Ge => a >= b,
            }
        }
    }
}

/// Comparison utilities for GPU vs CPU result validation.
pub mod compare {
    /// Assert u32 slices are equal with detailed diff on failure.
    pub fn assert_eq_u32(gpu: &[u32], cpu: &[u32], context: &str) {
        if gpu.len() != cpu.len() {
            panic!("{}: length mismatch: GPU={}, CPU={}", context, gpu.len(), cpu.len());
        }

        let mut first_diff = None;
        let mut diff_count = 0;

        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            if g != c {
                if first_diff.is_none() {
                    first_diff = Some((i, *g, *c));
                }
                diff_count += 1;
            }
        }

        if let Some((idx, gpu_val, cpu_val)) = first_diff {
            panic!(
                "{}: {} differences found. First at index {}: GPU={}, CPU={}",
                context, diff_count, idx, gpu_val, cpu_val
            );
        }
    }

    /// Assert i64 slices are equal with detailed diff on failure.
    pub fn assert_eq_i64(gpu: &[i64], cpu: &[i64], context: &str) {
        if gpu.len() != cpu.len() {
            panic!("{}: length mismatch: GPU={}, CPU={}", context, gpu.len(), cpu.len());
        }

        let mut first_diff = None;
        let mut diff_count = 0;

        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            if g != c {
                if first_diff.is_none() {
                    first_diff = Some((i, *g, *c));
                }
                diff_count += 1;
            }
        }

        if let Some((idx, gpu_val, cpu_val)) = first_diff {
            panic!(
                "{}: {} differences found. First at index {}: GPU={}, CPU={}",
                context, diff_count, idx, gpu_val, cpu_val
            );
        }
    }

    /// Assert u64 slices are equal with detailed diff on failure.
    pub fn assert_eq_u64(gpu: &[u64], cpu: &[u64], context: &str) {
        if gpu.len() != cpu.len() {
            panic!("{}: length mismatch: GPU={}, CPU={}", context, gpu.len(), cpu.len());
        }

        let mut first_diff = None;
        let mut diff_count = 0;

        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            if g != c {
                if first_diff.is_none() {
                    first_diff = Some((i, *g, *c));
                }
                diff_count += 1;
            }
        }

        if let Some((idx, gpu_val, cpu_val)) = first_diff {
            panic!(
                "{}: {} differences found. First at index {}: GPU={}, CPU={}",
                context, diff_count, idx, gpu_val, cpu_val
            );
        }
    }

    /// Assert f64 slices are equal within ULP tolerance.
    pub fn assert_eq_f64_ulp(gpu: &[f64], cpu: &[f64], max_ulp: u64, context: &str) {
        if gpu.len() != cpu.len() {
            panic!("{}: length mismatch: GPU={}, CPU={}", context, gpu.len(), cpu.len());
        }

        let mut first_diff = None;
        let mut diff_count = 0;

        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            let ulp_diff = ulp_distance(*g, *c);
            if ulp_diff > max_ulp {
                if first_diff.is_none() {
                    first_diff = Some((i, *g, *c, ulp_diff));
                }
                diff_count += 1;
            }
        }

        if let Some((idx, gpu_val, cpu_val, ulp)) = first_diff {
            panic!(
                "{}: {} ULP violations (max={}). First at index {}: GPU={}, CPU={}, ULP={}",
                context, diff_count, max_ulp, idx, gpu_val, cpu_val, ulp
            );
        }
    }

    /// Assert f64 slices are equal within relative tolerance.
    pub fn assert_eq_f64_rel(gpu: &[f64], cpu: &[f64], rel_tol: f64, context: &str) {
        if gpu.len() != cpu.len() {
            panic!("{}: length mismatch: GPU={}, CPU={}", context, gpu.len(), cpu.len());
        }

        let mut first_diff = None;
        let mut diff_count = 0;

        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            let rel_diff = if *c == 0.0 {
                g.abs()
            } else {
                ((g - c) / c).abs()
            };

            if rel_diff > rel_tol && !g.is_nan() && !c.is_nan() {
                if first_diff.is_none() {
                    first_diff = Some((i, *g, *c, rel_diff));
                }
                diff_count += 1;
            }
        }

        if let Some((idx, gpu_val, cpu_val, rel)) = first_diff {
            panic!(
                "{}: {} relative tolerance violations (max={}). First at index {}: GPU={}, CPU={}, rel_diff={}",
                context, diff_count, rel_tol, idx, gpu_val, cpu_val, rel
            );
        }
    }

    /// Compute ULP distance between two f64 values.
    fn ulp_distance(a: f64, b: f64) -> u64 {
        if a.is_nan() || b.is_nan() {
            return u64::MAX;
        }
        if a == b {
            return 0;
        }
        if a.is_infinite() || b.is_infinite() {
            return u64::MAX;
        }

        let a_bits = a.to_bits() as i64;
        let b_bits = b.to_bits() as i64;

        (a_bits - b_bits).unsigned_abs()
    }

    /// Assert sets are equal (order-independent).
    pub fn assert_set_eq_u32(gpu: &[u32], cpu: &[u32], context: &str) {
        let mut gpu_sorted = gpu.to_vec();
        let mut cpu_sorted = cpu.to_vec();
        gpu_sorted.sort();
        cpu_sorted.sort();

        if gpu_sorted != cpu_sorted {
            panic!(
                "{}: set mismatch. GPU (sorted): {:?}, CPU (sorted): {:?}",
                context,
                &gpu_sorted[..gpu_sorted.len().min(10)],
                &cpu_sorted[..cpu_sorted.len().min(10)]
            );
        }
    }

    /// Assert permutation is valid.
    pub fn assert_valid_permutation(perm: &[u32], len: usize, context: &str) {
        if perm.len() != len {
            panic!("{}: permutation length {} != expected {}", context, perm.len(), len);
        }

        let mut seen = vec![false; len];
        for (i, &idx) in perm.iter().enumerate() {
            if idx as usize >= len {
                panic!("{}: permutation index {} out of bounds at position {}", context, idx, i);
            }
            if seen[idx as usize] {
                panic!("{}: duplicate permutation index {} at position {}", context, idx, i);
            }
            seen[idx as usize] = true;
        }
    }

    /// Assert sort is stable (equal keys maintain relative order).
    pub fn assert_stable_sort(original_keys: &[u32], original_vals: &[u32], sorted_keys: &[u32], sorted_vals: &[u32], context: &str) {
        // Group by key in original order
        let mut key_to_vals: std::collections::HashMap<u32, Vec<u32>> = std::collections::HashMap::new();
        for (&k, &v) in original_keys.iter().zip(original_vals.iter()) {
            key_to_vals.entry(k).or_default().push(v);
        }

        // Check sorted result maintains relative order within each key group
        let mut key_to_idx: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        for (&k, &v) in sorted_keys.iter().zip(sorted_vals.iter()) {
            let idx = key_to_idx.entry(k).or_insert(0);
            let expected_vals = key_to_vals.get(&k).unwrap();
            if *idx >= expected_vals.len() {
                panic!("{}: too many values for key {}", context, k);
            }
            if v != expected_vals[*idx] {
                panic!(
                    "{}: stability violation for key {}. Expected value {} but got {}",
                    context, k, expected_vals[*idx], v
                );
            }
            *idx += 1;
        }
    }

    /// Check if result is sorted.
    pub fn is_sorted_u32(data: &[u32]) -> bool {
        data.windows(2).all(|w| w[0] <= w[1])
    }

    /// Check if result is sorted descending.
    pub fn is_sorted_desc_u32(data: &[u32]) -> bool {
        data.windows(2).all(|w| w[0] >= w[1])
    }
}
