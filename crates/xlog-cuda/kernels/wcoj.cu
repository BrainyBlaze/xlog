// kernels/wcoj.cu
//
// v0.6.2 GPU 3-way Worst-Case Optimal Join — triangle slice
// (u32 / Symbol via u32 path; u64 via parallel kernels).
//
// Implements
//
//   tri(X, Y, Z) :- e_xy(X, Y), e_yz(Y, Z), e_xz(X, Z)
//
// for already-sorted, already-deduped binary u32 relations:
//   * e_xy: lex-sorted by (X, Y)
//   * e_yz: lex-sorted by (Y, Z)
//   * e_xz: lex-sorted by (X, Z)
//
// Algorithm shape:
//
//   * The u32 / Symbol entry uses the histogram-guided block-slice
//     kernels below. A block owns a contiguous slice of the
//     prefix-summed root-key work space from SRDatalog Algorithm 2.
//   * Paper §5 Algorithm 2 lines 1,3,4,5,7,9,10,12 preserved; lines 6 + per-warp narrowing dropped per Phase-1 §2.2 A5 hardware constraint.
//   * The u64 entry uses row-wise count and materialize kernels:
//     thread `i` is bound to (X, Y) = (e_xy.col0[i], e_xy.col1[i])
//     and emits every (X, Y, Z) such that (Y, Z) ∈ e_yz and
//     (X, Z) ∈ e_xz.
//   * Z is found by a sorted-merge intersection of
//     e_yz.col1 over the Y range and e_xz.col1 over the X range,
//     each range located by a per-thread binary search on the
//     respective col0.
//   * Two-pass count → materialize: pass 1 fills `out_counts[i]`
//     with the number of triangles thread `i` will emit. The
//     recorded scan helper prefix-sums `out_counts` to get write
//     offsets and the total row count; pass 2 re-runs the
//     intersection, writing each emitted (X, Y, Z) at
//     `offsets[i] + j` for the j-th match.
//   * Output is naturally lex-sorted by (X, Y, Z): e_xy is
//     (X, Y)-sorted, sorted-merge enumerates Z in ascending
//     order, so the per-thread emission stride is monotone.
//
// Determinism: thread `i` always writes the same (count, rows) for
// the same input. Output positions come from a deterministic
// prefix-sum, not from runtime atomics.

#include <cstdint>

namespace {

// Lower-bound binary search for the first index in
// `col[0..n)` whose value is >= `target`. Returns `n` if no such
// index exists. Inputs are u32; the search is purely arithmetic
// so it is fully deterministic across threads.
__device__ __forceinline__ uint32_t lower_bound_u32(
    const uint32_t* __restrict__ col, uint32_t n, uint32_t target) {
    uint32_t lo = 0;
    uint32_t hi = n;
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        if (col[mid] < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// Upper-bound binary search for the first index whose value is
// > `target`. Returns `n` if no such index exists. Used together
// with `lower_bound_u32(target)` to obtain the equal-key range.
__device__ __forceinline__ uint32_t upper_bound_u32(
    const uint32_t* __restrict__ col, uint32_t n, uint32_t target) {
    uint32_t lo = 0;
    uint32_t hi = n;
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        if (col[mid] <= target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// Count the number of common values between two sorted u32
// ranges `[a, a+na)` and `[b, b+nb)`. Both ranges are assumed
// strictly ascending (deduped by caller), so a single sorted
// merge pass with no equal-element bookkeeping is sufficient.
__device__ __forceinline__ uint32_t intersect_count(
    const uint32_t* __restrict__ a, uint32_t na,
    const uint32_t* __restrict__ b, uint32_t nb) {
    uint32_t ia = 0, ib = 0;
    uint32_t cnt = 0;
    while (ia < na && ib < nb) {
        uint32_t va = a[ia];
        uint32_t vb = b[ib];
        if (va == vb) {
            cnt += 1;
            ia += 1;
            ib += 1;
        } else if (va < vb) {
            ia += 1;
        } else {
            ib += 1;
        }
    }
    return cnt;
}

// Same as `intersect_count` but emits each match into the three
// output columns at positions `out_base + 0..k-1`, where `k` is
// the returned count. Caller has reserved exactly that many
// slots via the count-pass + prefix-sum.
__device__ __forceinline__ uint32_t intersect_emit_xyz(
    uint32_t x, uint32_t y,
    const uint32_t* __restrict__ a, uint32_t na,
    const uint32_t* __restrict__ b, uint32_t nb,
    uint32_t out_base,
    uint32_t* __restrict__ out_x,
    uint32_t* __restrict__ out_y,
    uint32_t* __restrict__ out_z) {
    uint32_t ia = 0, ib = 0;
    uint32_t emitted = 0;
    while (ia < na && ib < nb) {
        uint32_t va = a[ia];
        uint32_t vb = b[ib];
        if (va == vb) {
            uint32_t pos = out_base + emitted;
            out_x[pos] = x;
            out_y[pos] = y;
            out_z[pos] = va;
            emitted += 1;
            ia += 1;
            ib += 1;
        } else if (va < vb) {
            ia += 1;
        } else {
            ib += 1;
        }
    }
    return emitted;
}

// ===============================================================
// U64 helpers — parallel to the u32 set above. Counters / counts /
// offsets / row_count stay u32 because they're bounded by
// `u32::MAX` (the host-side row-count guard upstream of every
// WCOJ entry), so only the join-key signatures change.
// ===============================================================

__device__ __forceinline__ uint32_t lower_bound_u64(
    const uint64_t* __restrict__ col, uint32_t n, uint64_t target) {
    uint32_t lo = 0;
    uint32_t hi = n;
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        if (col[mid] < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

__device__ __forceinline__ uint32_t upper_bound_u64(
    const uint64_t* __restrict__ col, uint32_t n, uint64_t target) {
    uint32_t lo = 0;
    uint32_t hi = n;
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        if (col[mid] <= target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

}  // anonymous namespace

extern "C" __global__ void wcoj_build_metadata_mark_boundaries_u32(
    const uint32_t* __restrict__ keys,
    uint32_t n,
    uint32_t* __restrict__ boundary_mask,
    uint32_t* __restrict__ boundary_prefix) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    uint32_t is_boundary = (i == 0 || keys[i] != keys[i - 1]) ? 1u : 0u;
    boundary_mask[i] = is_boundary;
    boundary_prefix[i] = is_boundary;
}

extern "C" __global__ void wcoj_build_metadata_mark_boundaries_u64(
    const uint64_t* __restrict__ keys,
    uint32_t n,
    uint32_t* __restrict__ boundary_mask,
    uint32_t* __restrict__ boundary_prefix) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    uint32_t is_boundary = (i == 0 || keys[i] != keys[i - 1]) ? 1u : 0u;
    boundary_mask[i] = is_boundary;
    boundary_prefix[i] = is_boundary;
}

extern "C" __global__ void wcoj_build_metadata_scatter_u32(
    const uint32_t* __restrict__ keys,
    uint32_t n,
    const uint32_t* __restrict__ boundary_mask,
    const uint32_t* __restrict__ boundary_prefix,
    uint32_t* __restrict__ unique_keys,
    uint32_t* __restrict__ fan_out,
    uint32_t* __restrict__ prefix_sum) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || boundary_mask[i] == 0u) {
        return;
    }
    uint32_t out = boundary_prefix[i];
    uint32_t key = keys[i];
    uint32_t end = upper_bound_u32(keys, n, key);
    uint32_t degree = end - i;
    unique_keys[out] = key;
    fan_out[out] = degree;
    prefix_sum[out] = i;
}

extern "C" __global__ void wcoj_build_metadata_scatter_u64(
    const uint64_t* __restrict__ keys,
    uint32_t n,
    const uint32_t* __restrict__ boundary_mask,
    const uint32_t* __restrict__ boundary_prefix,
    uint64_t* __restrict__ unique_keys,
    uint32_t* __restrict__ fan_out,
    uint32_t* __restrict__ prefix_sum) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || boundary_mask[i] == 0u) {
        return;
    }
    uint32_t out = boundary_prefix[i];
    uint64_t key = keys[i];
    uint32_t end = upper_bound_u64(keys, n, key);
    uint32_t degree = end - i;
    unique_keys[out] = key;
    fan_out[out] = degree;
    prefix_sum[out] = i;
}

extern "C" __global__ void wcoj_triangle_build_hg_work_plan_u32(
    const uint32_t* __restrict__ xy_col0,
    const uint32_t* __restrict__ xy_col1,
    uint32_t n_xy,
    const uint32_t* __restrict__ yz_col0,
    uint32_t n_yz,
    const uint32_t* __restrict__ xz_col0,
    uint32_t n_xz,
    uint32_t* __restrict__ xy_work_prefix,
    uint32_t* __restrict__ xy_yz_start,
    uint32_t* __restrict__ xy_yz_end,
    uint32_t* __restrict__ xy_xz_start,
    uint32_t* __restrict__ xy_xz_end) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        xy_work_prefix[n_xy] = 0;
    }
    if (i >= n_xy) {
        return;
    }
    uint32_t x = xy_col0[i];
    uint32_t y = xy_col1[i];
    uint32_t yz_lo = lower_bound_u32(yz_col0, n_yz, y);
    uint32_t yz_hi = upper_bound_u32(yz_col0, n_yz, y);
    uint32_t xz_lo = lower_bound_u32(xz_col0, n_xz, x);
    uint32_t xz_hi = upper_bound_u32(xz_col0, n_xz, x);
    uint32_t yz_len = yz_hi - yz_lo;
    uint32_t xz_len = xz_hi - xz_lo;
    xy_work_prefix[i] = (yz_len < xz_len) ? yz_len : xz_len;
    xy_yz_start[i] = yz_lo;
    xy_yz_end[i] = yz_hi;
    xy_xz_start[i] = xz_lo;
    xy_xz_end[i] = xz_hi;
}

extern "C" __global__ void wcoj_triangle_count_hg_u32(
    const uint32_t* __restrict__ yz_col1,
    uint32_t n_yz,
    const uint32_t* __restrict__ xz_col1,
    uint32_t n_xz,
    const uint32_t* __restrict__ xy_work_prefix,
    const uint32_t* __restrict__ xy_yz_start,
    const uint32_t* __restrict__ xy_yz_end,
    const uint32_t* __restrict__ xy_xz_start,
    const uint32_t* __restrict__ xy_xz_end,
    uint32_t n_xy,
    uint32_t total_work,
    uint32_t block_work_unit,
    uint32_t* __restrict__ out_block_counts) {
    // T4 Alg.2 line 3: block b owns a slice [bs, be) of the flattened work space.
    uint32_t block_start = blockIdx.x * block_work_unit;
    if (block_start >= total_work) {
        if (threadIdx.x == 0) {
            out_block_counts[blockIdx.x] = 0;
        }
        return;
    }
    uint32_t block_end = block_start + block_work_unit;
    if (block_end < block_start || block_end > total_work) {
        block_end = total_work;
    }

    uint32_t local_count = 0;
    for (uint32_t work_idx = block_start + threadIdx.x;
         work_idx < block_end;
         work_idx += blockDim.x) {
        // T4 Alg.2 line 4: binary-search C to recover the root row κ.
        uint32_t root_pos = upper_bound_u32(xy_work_prefix, n_xy + 1, work_idx);
        if (root_pos == 0) {
            continue;
        }
        uint32_t xy_idx = root_pos - 1;
        if (xy_idx >= n_xy) {
            continue;
        }
        uint32_t row_start = xy_work_prefix[xy_idx];
        uint32_t yz_lo = xy_yz_start[xy_idx];
        uint32_t yz_hi = xy_yz_end[xy_idx];
        uint32_t xz_lo = xy_xz_start[xy_idx];
        uint32_t xz_hi = xy_xz_end[xy_idx];
        if (yz_hi > n_yz || xz_hi > n_xz || yz_lo >= yz_hi || xz_lo >= xz_hi) {
            continue;
        }
        uint32_t yz_len = yz_hi - yz_lo;
        uint32_t xz_len = xz_hi - xz_lo;
        uint32_t probe_offset = work_idx - row_start;
        // T4 Alg.2 line 6: narrow the first handle to this lane's share.
        if (yz_len <= xz_len) {
            if (probe_offset >= yz_len) {
                continue;
            }
            uint32_t z = yz_col1[yz_lo + probe_offset];
            // T4 Alg.2 line 10: leaf intersection.
            uint32_t found = lower_bound_u32(xz_col1 + xz_lo, xz_len, z);
            if (found < xz_len && xz_col1[xz_lo + found] == z) {
                local_count += 1;
            }
        } else {
            if (probe_offset >= xz_len) {
                continue;
            }
            uint32_t z = xz_col1[xz_lo + probe_offset];
            // T4 Alg.2 line 10: leaf intersection.
            uint32_t found = lower_bound_u32(yz_col1 + yz_lo, yz_len, z);
            if (found < yz_len && yz_col1[yz_lo + found] == z) {
                local_count += 1;
            }
        }
    }

    __shared__ uint32_t partial[256];
    partial[threadIdx.x] = local_count;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        // T3 Alg.1 line 9: per-block output counts feed the deterministic scan.
        out_block_counts[blockIdx.x] = partial[0];
    }
}

// D1 aggregate-fused variant of wcoj_triangle_count_hg_u32: instead of
// reducing matches to one total, each match increments the counter of its
// e_xy root row (`out_row_counts[xy_idx]`, length n_xy, zero-initialized by
// the host). e_xy is lex-sorted by (X, Y), so per-X group counts are a
// segmented sum over contiguous row ranges downstream. Integer atomicAdd is
// order-insensitive, so the final counter values are deterministic
// (precedent: groupby_sum). The triangle rows are never materialized.
extern "C" __global__ void wcoj_triangle_groupby_root_count_hg_u32(
    const uint32_t* __restrict__ yz_col1,
    uint32_t n_yz,
    const uint32_t* __restrict__ xz_col1,
    uint32_t n_xz,
    const uint32_t* __restrict__ xy_work_prefix,
    const uint32_t* __restrict__ xy_yz_start,
    const uint32_t* __restrict__ xy_yz_end,
    const uint32_t* __restrict__ xy_xz_start,
    const uint32_t* __restrict__ xy_xz_end,
    uint32_t n_xy,
    uint32_t total_work,
    uint32_t block_work_unit,
    uint32_t* __restrict__ out_row_counts) {
    uint32_t block_start = blockIdx.x * block_work_unit;
    if (block_start >= total_work) {
        return;
    }
    uint32_t block_end = block_start + block_work_unit;
    if (block_end < block_start || block_end > total_work) {
        block_end = total_work;
    }

    for (uint32_t work_idx = block_start + threadIdx.x;
         work_idx < block_end;
         work_idx += blockDim.x) {
        uint32_t root_pos = upper_bound_u32(xy_work_prefix, n_xy + 1, work_idx);
        if (root_pos == 0) {
            continue;
        }
        uint32_t xy_idx = root_pos - 1;
        if (xy_idx >= n_xy) {
            continue;
        }
        uint32_t row_start = xy_work_prefix[xy_idx];
        uint32_t yz_lo = xy_yz_start[xy_idx];
        uint32_t yz_hi = xy_yz_end[xy_idx];
        uint32_t xz_lo = xy_xz_start[xy_idx];
        uint32_t xz_hi = xy_xz_end[xy_idx];
        if (yz_hi > n_yz || xz_hi > n_xz || yz_lo >= yz_hi || xz_lo >= xz_hi) {
            continue;
        }
        uint32_t yz_len = yz_hi - yz_lo;
        uint32_t xz_len = xz_hi - xz_lo;
        uint32_t probe_offset = work_idx - row_start;
        bool matched = false;
        if (yz_len <= xz_len) {
            if (probe_offset >= yz_len) {
                continue;
            }
            uint32_t z = yz_col1[yz_lo + probe_offset];
            uint32_t found = lower_bound_u32(xz_col1 + xz_lo, xz_len, z);
            matched = found < xz_len && xz_col1[xz_lo + found] == z;
        } else {
            if (probe_offset >= xz_len) {
                continue;
            }
            uint32_t z = xz_col1[xz_lo + probe_offset];
            uint32_t found = lower_bound_u32(yz_col1 + yz_lo, yz_len, z);
            matched = found < yz_len && yz_col1[yz_lo + found] == z;
        }
        if (matched) {
            atomicAdd(&out_row_counts[xy_idx], 1u);
        }
    }
}

namespace {

// D1 widening helper: resolve one flattened work index to its e_xy root
// row and, on a completed triangle, the aggregate value (Y when
// `value_from_z == 0`, the matched Z otherwise). Traversal is identical to
// `wcoj_triangle_groupby_root_count_hg_u32`. Returns false for padding /
// out-of-range / unmatched work items.
__device__ __forceinline__ bool groupby_root_match_value_u32(
    const uint32_t* __restrict__ yz_col1, uint32_t n_yz,
    const uint32_t* __restrict__ xz_col1, uint32_t n_xz,
    const uint32_t* __restrict__ xy_col1,
    uint32_t value_from_z,
    const uint32_t* __restrict__ xy_work_prefix,
    const uint32_t* __restrict__ xy_yz_start,
    const uint32_t* __restrict__ xy_yz_end,
    const uint32_t* __restrict__ xy_xz_start,
    const uint32_t* __restrict__ xy_xz_end,
    uint32_t n_xy,
    uint32_t work_idx,
    uint32_t* __restrict__ out_xy_idx,
    uint32_t* __restrict__ out_value) {
    uint32_t root_pos = upper_bound_u32(xy_work_prefix, n_xy + 1, work_idx);
    if (root_pos == 0) {
        return false;
    }
    uint32_t xy_idx = root_pos - 1;
    if (xy_idx >= n_xy) {
        return false;
    }
    uint32_t row_start = xy_work_prefix[xy_idx];
    uint32_t yz_lo = xy_yz_start[xy_idx];
    uint32_t yz_hi = xy_yz_end[xy_idx];
    uint32_t xz_lo = xy_xz_start[xy_idx];
    uint32_t xz_hi = xy_xz_end[xy_idx];
    if (yz_hi > n_yz || xz_hi > n_xz || yz_lo >= yz_hi || xz_lo >= xz_hi) {
        return false;
    }
    uint32_t yz_len = yz_hi - yz_lo;
    uint32_t xz_len = xz_hi - xz_lo;
    uint32_t probe_offset = work_idx - row_start;
    bool matched = false;
    uint32_t z = 0;
    if (yz_len <= xz_len) {
        if (probe_offset >= yz_len) {
            return false;
        }
        z = yz_col1[yz_lo + probe_offset];
        uint32_t found = lower_bound_u32(xz_col1 + xz_lo, xz_len, z);
        matched = found < xz_len && xz_col1[xz_lo + found] == z;
    } else {
        if (probe_offset >= xz_len) {
            return false;
        }
        z = xz_col1[xz_lo + probe_offset];
        uint32_t found = lower_bound_u32(yz_col1 + yz_lo, yz_len, z);
        matched = found < yz_len && yz_col1[yz_lo + found] == z;
    }
    if (!matched) {
        return false;
    }
    *out_xy_idx = xy_idx;
    *out_value = (value_from_z != 0u) ? z : xy_col1[xy_idx];
    return true;
}

}  // anonymous namespace

// D1 widening: sum/min/max variants of the aggregate-fused root kernel.
// Each match contributes its value (Y or Z, both triangle output
// variables) to the e_xy root row's accumulator, alongside a match
// counter (`out_row_counts`) that downstream compaction uses to drop
// roots with no completion. Integer atomics are order-insensitive, so
// final accumulator values are deterministic. Host pre-initializes
// accumulators: 0 for sum/max, 0xFFFFFFFF for min.
extern "C" __global__ void wcoj_triangle_groupby_root_sum_hg_u32(
    const uint32_t* __restrict__ yz_col1,
    uint32_t n_yz,
    const uint32_t* __restrict__ xz_col1,
    uint32_t n_xz,
    const uint32_t* __restrict__ xy_col1,
    uint32_t value_from_z,
    const uint32_t* __restrict__ xy_work_prefix,
    const uint32_t* __restrict__ xy_yz_start,
    const uint32_t* __restrict__ xy_yz_end,
    const uint32_t* __restrict__ xy_xz_start,
    const uint32_t* __restrict__ xy_xz_end,
    uint32_t n_xy,
    uint32_t total_work,
    uint32_t block_work_unit,
    uint32_t* __restrict__ out_row_counts,
    unsigned long long* __restrict__ out_row_sums) {
    uint32_t block_start = blockIdx.x * block_work_unit;
    if (block_start >= total_work) {
        return;
    }
    uint32_t block_end = block_start + block_work_unit;
    if (block_end < block_start || block_end > total_work) {
        block_end = total_work;
    }
    for (uint32_t work_idx = block_start + threadIdx.x;
         work_idx < block_end;
         work_idx += blockDim.x) {
        uint32_t xy_idx = 0;
        uint32_t value = 0;
        if (groupby_root_match_value_u32(
                yz_col1, n_yz, xz_col1, n_xz, xy_col1, value_from_z,
                xy_work_prefix, xy_yz_start, xy_yz_end, xy_xz_start,
                xy_xz_end, n_xy, work_idx, &xy_idx, &value)) {
            atomicAdd(&out_row_counts[xy_idx], 1u);
            atomicAdd(&out_row_sums[xy_idx], (unsigned long long)value);
        }
    }
}

extern "C" __global__ void wcoj_triangle_groupby_root_min_hg_u32(
    const uint32_t* __restrict__ yz_col1,
    uint32_t n_yz,
    const uint32_t* __restrict__ xz_col1,
    uint32_t n_xz,
    const uint32_t* __restrict__ xy_col1,
    uint32_t value_from_z,
    const uint32_t* __restrict__ xy_work_prefix,
    const uint32_t* __restrict__ xy_yz_start,
    const uint32_t* __restrict__ xy_yz_end,
    const uint32_t* __restrict__ xy_xz_start,
    const uint32_t* __restrict__ xy_xz_end,
    uint32_t n_xy,
    uint32_t total_work,
    uint32_t block_work_unit,
    uint32_t* __restrict__ out_row_counts,
    uint32_t* __restrict__ out_row_mins) {
    uint32_t block_start = blockIdx.x * block_work_unit;
    if (block_start >= total_work) {
        return;
    }
    uint32_t block_end = block_start + block_work_unit;
    if (block_end < block_start || block_end > total_work) {
        block_end = total_work;
    }
    for (uint32_t work_idx = block_start + threadIdx.x;
         work_idx < block_end;
         work_idx += blockDim.x) {
        uint32_t xy_idx = 0;
        uint32_t value = 0;
        if (groupby_root_match_value_u32(
                yz_col1, n_yz, xz_col1, n_xz, xy_col1, value_from_z,
                xy_work_prefix, xy_yz_start, xy_yz_end, xy_xz_start,
                xy_xz_end, n_xy, work_idx, &xy_idx, &value)) {
            atomicAdd(&out_row_counts[xy_idx], 1u);
            atomicMin(&out_row_mins[xy_idx], value);
        }
    }
}

extern "C" __global__ void wcoj_triangle_groupby_root_max_hg_u32(
    const uint32_t* __restrict__ yz_col1,
    uint32_t n_yz,
    const uint32_t* __restrict__ xz_col1,
    uint32_t n_xz,
    const uint32_t* __restrict__ xy_col1,
    uint32_t value_from_z,
    const uint32_t* __restrict__ xy_work_prefix,
    const uint32_t* __restrict__ xy_yz_start,
    const uint32_t* __restrict__ xy_yz_end,
    const uint32_t* __restrict__ xy_xz_start,
    const uint32_t* __restrict__ xy_xz_end,
    uint32_t n_xy,
    uint32_t total_work,
    uint32_t block_work_unit,
    uint32_t* __restrict__ out_row_counts,
    uint32_t* __restrict__ out_row_maxs) {
    uint32_t block_start = blockIdx.x * block_work_unit;
    if (block_start >= total_work) {
        return;
    }
    uint32_t block_end = block_start + block_work_unit;
    if (block_end < block_start || block_end > total_work) {
        block_end = total_work;
    }
    for (uint32_t work_idx = block_start + threadIdx.x;
         work_idx < block_end;
         work_idx += blockDim.x) {
        uint32_t xy_idx = 0;
        uint32_t value = 0;
        if (groupby_root_match_value_u32(
                yz_col1, n_yz, xz_col1, n_xz, xy_col1, value_from_z,
                xy_work_prefix, xy_yz_start, xy_yz_end, xy_xz_start,
                xy_xz_end, n_xy, work_idx, &xy_idx, &value)) {
            atomicAdd(&out_row_counts[xy_idx], 1u);
            atomicMax(&out_row_maxs[xy_idx], value);
        }
    }
}

extern "C" __global__ void wcoj_triangle_materialize_hg_u32(
    const uint32_t* __restrict__ xy_col0,
    const uint32_t* __restrict__ xy_col1,
    const uint32_t* __restrict__ yz_col1,
    uint32_t n_yz,
    const uint32_t* __restrict__ xz_col1,
    uint32_t n_xz,
    const uint32_t* __restrict__ xy_work_prefix,
    const uint32_t* __restrict__ xy_yz_start,
    const uint32_t* __restrict__ xy_yz_end,
    const uint32_t* __restrict__ xy_xz_start,
    const uint32_t* __restrict__ xy_xz_end,
    uint32_t n_xy,
    uint32_t total_work,
    uint32_t block_work_unit,
    const uint32_t* __restrict__ block_offsets,
    uint32_t total_rows,
    uint32_t* __restrict__ out_x,
    uint32_t* __restrict__ out_y,
    uint32_t* __restrict__ out_z) {
    uint32_t block_start = blockIdx.x * block_work_unit;
    if (block_start >= total_work) {
        return;
    }
    uint32_t block_end = block_start + block_work_unit;
    if (block_end < block_start || block_end > total_work) {
        block_end = total_work;
    }

    uint32_t local_count = 0;
    for (uint32_t work_idx = block_start + threadIdx.x;
         work_idx < block_end;
         work_idx += blockDim.x) {
        uint32_t root_pos = upper_bound_u32(xy_work_prefix, n_xy + 1, work_idx);
        if (root_pos == 0) {
            continue;
        }
        uint32_t xy_idx = root_pos - 1;
        if (xy_idx >= n_xy) {
            continue;
        }
        uint32_t row_start = xy_work_prefix[xy_idx];
        uint32_t yz_lo = xy_yz_start[xy_idx];
        uint32_t yz_hi = xy_yz_end[xy_idx];
        uint32_t xz_lo = xy_xz_start[xy_idx];
        uint32_t xz_hi = xy_xz_end[xy_idx];
        if (yz_hi > n_yz || xz_hi > n_xz || yz_lo >= yz_hi || xz_lo >= xz_hi) {
            continue;
        }
        uint32_t yz_len = yz_hi - yz_lo;
        uint32_t xz_len = xz_hi - xz_lo;
        uint32_t probe_offset = work_idx - row_start;
        if (yz_len <= xz_len) {
            if (probe_offset >= yz_len) {
                continue;
            }
            uint32_t z = yz_col1[yz_lo + probe_offset];
            uint32_t found = lower_bound_u32(xz_col1 + xz_lo, xz_len, z);
            if (found < xz_len && xz_col1[xz_lo + found] == z) {
                local_count += 1;
            }
        } else {
            if (probe_offset >= xz_len) {
                continue;
            }
            uint32_t z = xz_col1[xz_lo + probe_offset];
            uint32_t found = lower_bound_u32(yz_col1 + yz_lo, yz_len, z);
            if (found < yz_len && yz_col1[yz_lo + found] == z) {
                local_count += 1;
            }
        }
    }

    __shared__ uint32_t thread_prefix[256];
    thread_prefix[threadIdx.x] = local_count;
    __syncthreads();
    for (uint32_t stride = 1; stride < blockDim.x; stride <<= 1) {
        uint32_t add = 0;
        if (threadIdx.x >= stride) {
            add = thread_prefix[threadIdx.x - stride];
        }
        __syncthreads();
        thread_prefix[threadIdx.x] += add;
        __syncthreads();
    }
    uint32_t thread_base = block_offsets[blockIdx.x];
    if (threadIdx.x > 0) {
        thread_base += thread_prefix[threadIdx.x - 1];
    }

    uint32_t local_emit = 0;
    for (uint32_t work_idx = block_start + threadIdx.x;
         work_idx < block_end;
         work_idx += blockDim.x) {
        uint32_t root_pos = upper_bound_u32(xy_work_prefix, n_xy + 1, work_idx);
        if (root_pos == 0) {
            continue;
        }
        uint32_t xy_idx = root_pos - 1;
        if (xy_idx >= n_xy) {
            continue;
        }
        uint32_t row_start = xy_work_prefix[xy_idx];
        uint32_t yz_lo = xy_yz_start[xy_idx];
        uint32_t yz_hi = xy_yz_end[xy_idx];
        uint32_t xz_lo = xy_xz_start[xy_idx];
        uint32_t xz_hi = xy_xz_end[xy_idx];
        if (yz_hi > n_yz || xz_hi > n_xz || yz_lo >= yz_hi || xz_lo >= xz_hi) {
            continue;
        }
        uint32_t yz_len = yz_hi - yz_lo;
        uint32_t xz_len = xz_hi - xz_lo;
        uint32_t probe_offset = work_idx - row_start;
        uint32_t matched = 0;
        uint32_t z = 0;
        if (yz_len <= xz_len) {
            if (probe_offset >= yz_len) {
                continue;
            }
            z = yz_col1[yz_lo + probe_offset];
            uint32_t found = lower_bound_u32(xz_col1 + xz_lo, xz_len, z);
            matched = (found < xz_len && xz_col1[xz_lo + found] == z) ? 1u : 0u;
        } else {
            if (probe_offset >= xz_len) {
                continue;
            }
            z = xz_col1[xz_lo + probe_offset];
            uint32_t found = lower_bound_u32(yz_col1 + yz_lo, yz_len, z);
            matched = (found < yz_len && yz_col1[yz_lo + found] == z) ? 1u : 0u;
        }
        if (matched != 0u) {
            uint32_t out = thread_base + local_emit;
            if (out < total_rows) {
                out_x[out] = xy_col0[xy_idx];
                out_y[out] = xy_col1[xy_idx];
                out_z[out] = z;
            }
            local_emit += 1;
        }
    }
}

extern "C" __global__ void wcoj_triangle_build_hg_work_plan_u64(
    const uint64_t* __restrict__ xy_col0,
    const uint64_t* __restrict__ xy_col1,
    uint32_t n_xy,
    const uint64_t* __restrict__ yz_col0,
    uint32_t n_yz,
    const uint64_t* __restrict__ xz_col0,
    uint32_t n_xz,
    uint32_t* __restrict__ xy_work_prefix,
    uint32_t* __restrict__ xy_yz_start,
    uint32_t* __restrict__ xy_yz_end,
    uint32_t* __restrict__ xy_xz_start,
    uint32_t* __restrict__ xy_xz_end) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        xy_work_prefix[n_xy] = 0;
    }
    if (i >= n_xy) {
        return;
    }
    uint64_t x = xy_col0[i];
    uint64_t y = xy_col1[i];
    uint32_t yz_lo = lower_bound_u64(yz_col0, n_yz, y);
    uint32_t yz_hi = upper_bound_u64(yz_col0, n_yz, y);
    uint32_t xz_lo = lower_bound_u64(xz_col0, n_xz, x);
    uint32_t xz_hi = upper_bound_u64(xz_col0, n_xz, x);
    uint32_t yz_len = yz_hi - yz_lo;
    uint32_t xz_len = xz_hi - xz_lo;
    xy_work_prefix[i] = (yz_len < xz_len) ? yz_len : xz_len;
    xy_yz_start[i] = yz_lo;
    xy_yz_end[i] = yz_hi;
    xy_xz_start[i] = xz_lo;
    xy_xz_end[i] = xz_hi;
}

extern "C" __global__ void wcoj_triangle_count_hg_u64(
    const uint64_t* __restrict__ yz_col1,
    uint32_t n_yz,
    const uint64_t* __restrict__ xz_col1,
    uint32_t n_xz,
    const uint32_t* __restrict__ xy_work_prefix,
    const uint32_t* __restrict__ xy_yz_start,
    const uint32_t* __restrict__ xy_yz_end,
    const uint32_t* __restrict__ xy_xz_start,
    const uint32_t* __restrict__ xy_xz_end,
    uint32_t n_xy,
    uint32_t total_work,
    uint32_t block_work_unit,
    uint32_t* __restrict__ out_block_counts) {
    uint32_t block_start = blockIdx.x * block_work_unit;
    if (block_start >= total_work) {
        if (threadIdx.x == 0) {
            out_block_counts[blockIdx.x] = 0;
        }
        return;
    }
    uint32_t block_end = block_start + block_work_unit;
    if (block_end < block_start || block_end > total_work) {
        block_end = total_work;
    }

    uint32_t local_count = 0;
    for (uint32_t work_idx = block_start + threadIdx.x;
         work_idx < block_end;
         work_idx += blockDim.x) {
        uint32_t root_pos = upper_bound_u32(xy_work_prefix, n_xy + 1, work_idx);
        if (root_pos == 0) {
            continue;
        }
        uint32_t xy_idx = root_pos - 1;
        if (xy_idx >= n_xy) {
            continue;
        }
        uint32_t row_start = xy_work_prefix[xy_idx];
        uint32_t yz_lo = xy_yz_start[xy_idx];
        uint32_t yz_hi = xy_yz_end[xy_idx];
        uint32_t xz_lo = xy_xz_start[xy_idx];
        uint32_t xz_hi = xy_xz_end[xy_idx];
        if (yz_hi > n_yz || xz_hi > n_xz || yz_lo >= yz_hi || xz_lo >= xz_hi) {
            continue;
        }
        uint32_t yz_len = yz_hi - yz_lo;
        uint32_t xz_len = xz_hi - xz_lo;
        uint32_t probe_offset = work_idx - row_start;
        if (yz_len <= xz_len) {
            if (probe_offset >= yz_len) {
                continue;
            }
            uint64_t z = yz_col1[yz_lo + probe_offset];
            uint32_t found = lower_bound_u64(xz_col1 + xz_lo, xz_len, z);
            if (found < xz_len && xz_col1[xz_lo + found] == z) {
                local_count += 1;
            }
        } else {
            if (probe_offset >= xz_len) {
                continue;
            }
            uint64_t z = xz_col1[xz_lo + probe_offset];
            uint32_t found = lower_bound_u64(yz_col1 + yz_lo, yz_len, z);
            if (found < yz_len && yz_col1[yz_lo + found] == z) {
                local_count += 1;
            }
        }
    }

    __shared__ uint32_t partial[256];
    partial[threadIdx.x] = local_count;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out_block_counts[blockIdx.x] = partial[0];
    }
}

// D1 widening — u64-key sibling of
// `wcoj_triangle_groupby_root_count_hg_u32`: each match increments the
// counter of its e_xy root row. Traversal mirrors
// `wcoj_triangle_count_hg_u64`; counters stay u32 (row counts bounded by
// the host-side u32 row guard upstream of every WCOJ entry).
extern "C" __global__ void wcoj_triangle_groupby_root_count_hg_u64(
    const uint64_t* __restrict__ yz_col1,
    uint32_t n_yz,
    const uint64_t* __restrict__ xz_col1,
    uint32_t n_xz,
    const uint32_t* __restrict__ xy_work_prefix,
    const uint32_t* __restrict__ xy_yz_start,
    const uint32_t* __restrict__ xy_yz_end,
    const uint32_t* __restrict__ xy_xz_start,
    const uint32_t* __restrict__ xy_xz_end,
    uint32_t n_xy,
    uint32_t total_work,
    uint32_t block_work_unit,
    uint32_t* __restrict__ out_row_counts) {
    uint32_t block_start = blockIdx.x * block_work_unit;
    if (block_start >= total_work) {
        return;
    }
    uint32_t block_end = block_start + block_work_unit;
    if (block_end < block_start || block_end > total_work) {
        block_end = total_work;
    }

    for (uint32_t work_idx = block_start + threadIdx.x;
         work_idx < block_end;
         work_idx += blockDim.x) {
        uint32_t root_pos = upper_bound_u32(xy_work_prefix, n_xy + 1, work_idx);
        if (root_pos == 0) {
            continue;
        }
        uint32_t xy_idx = root_pos - 1;
        if (xy_idx >= n_xy) {
            continue;
        }
        uint32_t row_start = xy_work_prefix[xy_idx];
        uint32_t yz_lo = xy_yz_start[xy_idx];
        uint32_t yz_hi = xy_yz_end[xy_idx];
        uint32_t xz_lo = xy_xz_start[xy_idx];
        uint32_t xz_hi = xy_xz_end[xy_idx];
        if (yz_hi > n_yz || xz_hi > n_xz || yz_lo >= yz_hi || xz_lo >= xz_hi) {
            continue;
        }
        uint32_t yz_len = yz_hi - yz_lo;
        uint32_t xz_len = xz_hi - xz_lo;
        uint32_t probe_offset = work_idx - row_start;
        bool matched = false;
        if (yz_len <= xz_len) {
            if (probe_offset >= yz_len) {
                continue;
            }
            uint64_t z = yz_col1[yz_lo + probe_offset];
            uint32_t found = lower_bound_u64(xz_col1 + xz_lo, xz_len, z);
            matched = found < xz_len && xz_col1[xz_lo + found] == z;
        } else {
            if (probe_offset >= xz_len) {
                continue;
            }
            uint64_t z = xz_col1[xz_lo + probe_offset];
            uint32_t found = lower_bound_u64(yz_col1 + yz_lo, yz_len, z);
            matched = found < yz_len && yz_col1[yz_lo + found] == z;
        }
        if (matched) {
            atomicAdd(&out_row_counts[xy_idx], 1u);
        }
    }
}

// D1 widening — reduce per-row match counts into per-unique-root u64
// totals. `group_start[g]` is the first e_xy row of unique root g
// (ascending; the WCOJ metadata builder's prefix_sum), so a row's group is
// found by binary search. Key-width agnostic: only row indices are read.
// Integer atomicAdd keeps the totals deterministic.
extern "C" __global__ void wcoj_groupby_root_segment_sum_counts_u32(
    const uint32_t* __restrict__ row_counts,
    uint32_t n_rows,
    const uint32_t* __restrict__ group_start,
    uint32_t key_count,
    unsigned long long* __restrict__ out_sums) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows) {
        return;
    }
    uint32_t c = row_counts[i];
    if (c == 0) {
        return;
    }
    uint32_t pos = upper_bound_u32(group_start, key_count, i);
    if (pos == 0) {
        return;
    }
    atomicAdd(&out_sums[pos - 1], (unsigned long long)c);
}

namespace {

// S1c widening helper: u64 sibling of `groupby_root_match_value_u32`.
// Resolves one flattened work index to its e_xy root row and, on a
// completed triangle, the aggregate value (Y when `value_from_z == 0`,
// the matched Z otherwise). Traversal is identical to
// `wcoj_triangle_groupby_root_count_hg_u64`.
__device__ __forceinline__ bool groupby_root_match_value_u64(
    const uint64_t* __restrict__ yz_col1, uint32_t n_yz,
    const uint64_t* __restrict__ xz_col1, uint32_t n_xz,
    const uint64_t* __restrict__ xy_col1,
    uint32_t value_from_z,
    const uint32_t* __restrict__ xy_work_prefix,
    const uint32_t* __restrict__ xy_yz_start,
    const uint32_t* __restrict__ xy_yz_end,
    const uint32_t* __restrict__ xy_xz_start,
    const uint32_t* __restrict__ xy_xz_end,
    uint32_t n_xy,
    uint32_t work_idx,
    uint32_t* __restrict__ out_xy_idx,
    uint64_t* __restrict__ out_value) {
    uint32_t root_pos = upper_bound_u32(xy_work_prefix, n_xy + 1, work_idx);
    if (root_pos == 0) {
        return false;
    }
    uint32_t xy_idx = root_pos - 1;
    if (xy_idx >= n_xy) {
        return false;
    }
    uint32_t row_start = xy_work_prefix[xy_idx];
    uint32_t yz_lo = xy_yz_start[xy_idx];
    uint32_t yz_hi = xy_yz_end[xy_idx];
    uint32_t xz_lo = xy_xz_start[xy_idx];
    uint32_t xz_hi = xy_xz_end[xy_idx];
    if (yz_hi > n_yz || xz_hi > n_xz || yz_lo >= yz_hi || xz_lo >= xz_hi) {
        return false;
    }
    uint32_t yz_len = yz_hi - yz_lo;
    uint32_t xz_len = xz_hi - xz_lo;
    uint32_t probe_offset = work_idx - row_start;
    bool matched = false;
    uint64_t z = 0;
    if (yz_len <= xz_len) {
        if (probe_offset >= yz_len) {
            return false;
        }
        z = yz_col1[yz_lo + probe_offset];
        uint32_t found = lower_bound_u64(xz_col1 + xz_lo, xz_len, z);
        matched = found < xz_len && xz_col1[xz_lo + found] == z;
    } else {
        if (probe_offset >= xz_len) {
            return false;
        }
        z = xz_col1[xz_lo + probe_offset];
        uint32_t found = lower_bound_u64(yz_col1 + yz_lo, yz_len, z);
        matched = found < yz_len && yz_col1[yz_lo + found] == z;
    }
    if (!matched) {
        return false;
    }
    *out_xy_idx = xy_idx;
    *out_value = (value_from_z != 0u) ? z : xy_col1[xy_idx];
    return true;
}

}  // anonymous namespace

// S1c widening: u64-key sum/min/max variants of the aggregate-fused root
// kernel. Each match contributes its u64 value (Y or Z) to the e_xy root
// row's accumulator, alongside a match counter that downstream compaction
// uses to drop roots with no completion. Integer atomics are
// order-insensitive, so final accumulator values are deterministic. Host
// pre-initializes accumulators: 0 for sum/max, u64::MAX for min. Sum
// accumulates u64 (wrapping on overflow, exactly like groupby_sum_u64).
extern "C" __global__ void wcoj_triangle_groupby_root_sum_hg_u64(
    const uint64_t* __restrict__ yz_col1,
    uint32_t n_yz,
    const uint64_t* __restrict__ xz_col1,
    uint32_t n_xz,
    const uint64_t* __restrict__ xy_col1,
    uint32_t value_from_z,
    const uint32_t* __restrict__ xy_work_prefix,
    const uint32_t* __restrict__ xy_yz_start,
    const uint32_t* __restrict__ xy_yz_end,
    const uint32_t* __restrict__ xy_xz_start,
    const uint32_t* __restrict__ xy_xz_end,
    uint32_t n_xy,
    uint32_t total_work,
    uint32_t block_work_unit,
    uint32_t* __restrict__ out_row_counts,
    unsigned long long* __restrict__ out_row_sums) {
    uint32_t block_start = blockIdx.x * block_work_unit;
    if (block_start >= total_work) {
        return;
    }
    uint32_t block_end = block_start + block_work_unit;
    if (block_end < block_start || block_end > total_work) {
        block_end = total_work;
    }
    for (uint32_t work_idx = block_start + threadIdx.x;
         work_idx < block_end;
         work_idx += blockDim.x) {
        uint32_t xy_idx = 0;
        uint64_t value = 0;
        if (groupby_root_match_value_u64(
                yz_col1, n_yz, xz_col1, n_xz, xy_col1, value_from_z,
                xy_work_prefix, xy_yz_start, xy_yz_end, xy_xz_start,
                xy_xz_end, n_xy, work_idx, &xy_idx, &value)) {
            atomicAdd(&out_row_counts[xy_idx], 1u);
            atomicAdd(&out_row_sums[xy_idx], (unsigned long long)value);
        }
    }
}

extern "C" __global__ void wcoj_triangle_groupby_root_min_hg_u64(
    const uint64_t* __restrict__ yz_col1,
    uint32_t n_yz,
    const uint64_t* __restrict__ xz_col1,
    uint32_t n_xz,
    const uint64_t* __restrict__ xy_col1,
    uint32_t value_from_z,
    const uint32_t* __restrict__ xy_work_prefix,
    const uint32_t* __restrict__ xy_yz_start,
    const uint32_t* __restrict__ xy_yz_end,
    const uint32_t* __restrict__ xy_xz_start,
    const uint32_t* __restrict__ xy_xz_end,
    uint32_t n_xy,
    uint32_t total_work,
    uint32_t block_work_unit,
    uint32_t* __restrict__ out_row_counts,
    unsigned long long* __restrict__ out_row_mins) {
    uint32_t block_start = blockIdx.x * block_work_unit;
    if (block_start >= total_work) {
        return;
    }
    uint32_t block_end = block_start + block_work_unit;
    if (block_end < block_start || block_end > total_work) {
        block_end = total_work;
    }
    for (uint32_t work_idx = block_start + threadIdx.x;
         work_idx < block_end;
         work_idx += blockDim.x) {
        uint32_t xy_idx = 0;
        uint64_t value = 0;
        if (groupby_root_match_value_u64(
                yz_col1, n_yz, xz_col1, n_xz, xy_col1, value_from_z,
                xy_work_prefix, xy_yz_start, xy_yz_end, xy_xz_start,
                xy_xz_end, n_xy, work_idx, &xy_idx, &value)) {
            atomicAdd(&out_row_counts[xy_idx], 1u);
            atomicMin(&out_row_mins[xy_idx], (unsigned long long)value);
        }
    }
}

extern "C" __global__ void wcoj_triangle_groupby_root_max_hg_u64(
    const uint64_t* __restrict__ yz_col1,
    uint32_t n_yz,
    const uint64_t* __restrict__ xz_col1,
    uint32_t n_xz,
    const uint64_t* __restrict__ xy_col1,
    uint32_t value_from_z,
    const uint32_t* __restrict__ xy_work_prefix,
    const uint32_t* __restrict__ xy_yz_start,
    const uint32_t* __restrict__ xy_yz_end,
    const uint32_t* __restrict__ xy_xz_start,
    const uint32_t* __restrict__ xy_xz_end,
    uint32_t n_xy,
    uint32_t total_work,
    uint32_t block_work_unit,
    uint32_t* __restrict__ out_row_counts,
    unsigned long long* __restrict__ out_row_maxs) {
    uint32_t block_start = blockIdx.x * block_work_unit;
    if (block_start >= total_work) {
        return;
    }
    uint32_t block_end = block_start + block_work_unit;
    if (block_end < block_start || block_end > total_work) {
        block_end = total_work;
    }
    for (uint32_t work_idx = block_start + threadIdx.x;
         work_idx < block_end;
         work_idx += blockDim.x) {
        uint32_t xy_idx = 0;
        uint64_t value = 0;
        if (groupby_root_match_value_u64(
                yz_col1, n_yz, xz_col1, n_xz, xy_col1, value_from_z,
                xy_work_prefix, xy_yz_start, xy_yz_end, xy_xz_start,
                xy_xz_end, n_xy, work_idx, &xy_idx, &value)) {
            atomicAdd(&out_row_counts[xy_idx], 1u);
            atomicMax(&out_row_maxs[xy_idx], (unsigned long long)value);
        }
    }
}

// S1c widening — reduce per-row u64 aggregate partials into per-unique-root
// totals, sibling of `wcoj_groupby_root_segment_sum_counts_u32`. Rows with
// zero matches are skipped (their partials are identities anyway); roots
// whose entire segment is empty are compacted away downstream via the
// count totals. `group_start[g]` is the first e_xy row of unique root g.
extern "C" __global__ void wcoj_groupby_root_segment_sum_values_u64(
    const uint32_t* __restrict__ row_counts,
    const unsigned long long* __restrict__ row_values,
    uint32_t n_rows,
    const uint32_t* __restrict__ group_start,
    uint32_t key_count,
    unsigned long long* __restrict__ out_sums) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows || row_counts[i] == 0) {
        return;
    }
    uint32_t pos = upper_bound_u32(group_start, key_count, i);
    if (pos == 0) {
        return;
    }
    atomicAdd(&out_sums[pos - 1], row_values[i]);
}

extern "C" __global__ void wcoj_groupby_root_segment_min_values_u64(
    const uint32_t* __restrict__ row_counts,
    const unsigned long long* __restrict__ row_values,
    uint32_t n_rows,
    const uint32_t* __restrict__ group_start,
    uint32_t key_count,
    unsigned long long* __restrict__ out_mins) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows || row_counts[i] == 0) {
        return;
    }
    uint32_t pos = upper_bound_u32(group_start, key_count, i);
    if (pos == 0) {
        return;
    }
    atomicMin(&out_mins[pos - 1], row_values[i]);
}

extern "C" __global__ void wcoj_groupby_root_segment_max_values_u64(
    const uint32_t* __restrict__ row_counts,
    const unsigned long long* __restrict__ row_values,
    uint32_t n_rows,
    const uint32_t* __restrict__ group_start,
    uint32_t key_count,
    unsigned long long* __restrict__ out_maxs) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows || row_counts[i] == 0) {
        return;
    }
    uint32_t pos = upper_bound_u32(group_start, key_count, i);
    if (pos == 0) {
        return;
    }
    atomicMax(&out_maxs[pos - 1], row_values[i]);
}

extern "C" __global__ void wcoj_triangle_materialize_hg_u64(
    const uint64_t* __restrict__ xy_col0,
    const uint64_t* __restrict__ xy_col1,
    const uint64_t* __restrict__ yz_col1,
    uint32_t n_yz,
    const uint64_t* __restrict__ xz_col1,
    uint32_t n_xz,
    const uint32_t* __restrict__ xy_work_prefix,
    const uint32_t* __restrict__ xy_yz_start,
    const uint32_t* __restrict__ xy_yz_end,
    const uint32_t* __restrict__ xy_xz_start,
    const uint32_t* __restrict__ xy_xz_end,
    uint32_t n_xy,
    uint32_t total_work,
    uint32_t block_work_unit,
    const uint32_t* __restrict__ block_offsets,
    uint32_t total_rows,
    uint64_t* __restrict__ out_x,
    uint64_t* __restrict__ out_y,
    uint64_t* __restrict__ out_z) {
    uint32_t block_start = blockIdx.x * block_work_unit;
    if (block_start >= total_work) {
        return;
    }
    uint32_t block_end = block_start + block_work_unit;
    if (block_end < block_start || block_end > total_work) {
        block_end = total_work;
    }

    uint32_t local_count = 0;
    for (uint32_t work_idx = block_start + threadIdx.x;
         work_idx < block_end;
         work_idx += blockDim.x) {
        uint32_t root_pos = upper_bound_u32(xy_work_prefix, n_xy + 1, work_idx);
        if (root_pos == 0) {
            continue;
        }
        uint32_t xy_idx = root_pos - 1;
        if (xy_idx >= n_xy) {
            continue;
        }
        uint32_t row_start = xy_work_prefix[xy_idx];
        uint32_t yz_lo = xy_yz_start[xy_idx];
        uint32_t yz_hi = xy_yz_end[xy_idx];
        uint32_t xz_lo = xy_xz_start[xy_idx];
        uint32_t xz_hi = xy_xz_end[xy_idx];
        if (yz_hi > n_yz || xz_hi > n_xz || yz_lo >= yz_hi || xz_lo >= xz_hi) {
            continue;
        }
        uint32_t yz_len = yz_hi - yz_lo;
        uint32_t xz_len = xz_hi - xz_lo;
        uint32_t probe_offset = work_idx - row_start;
        if (yz_len <= xz_len) {
            if (probe_offset >= yz_len) {
                continue;
            }
            uint64_t z = yz_col1[yz_lo + probe_offset];
            uint32_t found = lower_bound_u64(xz_col1 + xz_lo, xz_len, z);
            if (found < xz_len && xz_col1[xz_lo + found] == z) {
                local_count += 1;
            }
        } else {
            if (probe_offset >= xz_len) {
                continue;
            }
            uint64_t z = xz_col1[xz_lo + probe_offset];
            uint32_t found = lower_bound_u64(yz_col1 + yz_lo, yz_len, z);
            if (found < yz_len && yz_col1[yz_lo + found] == z) {
                local_count += 1;
            }
        }
    }

    __shared__ uint32_t thread_prefix[256];
    thread_prefix[threadIdx.x] = local_count;
    __syncthreads();
    for (uint32_t stride = 1; stride < blockDim.x; stride <<= 1) {
        uint32_t add = 0;
        if (threadIdx.x >= stride) {
            add = thread_prefix[threadIdx.x - stride];
        }
        __syncthreads();
        thread_prefix[threadIdx.x] += add;
        __syncthreads();
    }
    uint32_t thread_base = block_offsets[blockIdx.x];
    if (threadIdx.x > 0) {
        thread_base += thread_prefix[threadIdx.x - 1];
    }

    uint32_t local_emit = 0;
    for (uint32_t work_idx = block_start + threadIdx.x;
         work_idx < block_end;
         work_idx += blockDim.x) {
        uint32_t root_pos = upper_bound_u32(xy_work_prefix, n_xy + 1, work_idx);
        if (root_pos == 0) {
            continue;
        }
        uint32_t xy_idx = root_pos - 1;
        if (xy_idx >= n_xy) {
            continue;
        }
        uint32_t row_start = xy_work_prefix[xy_idx];
        uint32_t yz_lo = xy_yz_start[xy_idx];
        uint32_t yz_hi = xy_yz_end[xy_idx];
        uint32_t xz_lo = xy_xz_start[xy_idx];
        uint32_t xz_hi = xy_xz_end[xy_idx];
        if (yz_hi > n_yz || xz_hi > n_xz || yz_lo >= yz_hi || xz_lo >= xz_hi) {
            continue;
        }
        uint32_t yz_len = yz_hi - yz_lo;
        uint32_t xz_len = xz_hi - xz_lo;
        uint32_t probe_offset = work_idx - row_start;
        uint32_t matched = 0;
        uint64_t z = 0;
        if (yz_len <= xz_len) {
            if (probe_offset >= yz_len) {
                continue;
            }
            z = yz_col1[yz_lo + probe_offset];
            uint32_t found = lower_bound_u64(xz_col1 + xz_lo, xz_len, z);
            matched = (found < xz_len && xz_col1[xz_lo + found] == z) ? 1u : 0u;
        } else {
            if (probe_offset >= xz_len) {
                continue;
            }
            z = xz_col1[xz_lo + probe_offset];
            uint32_t found = lower_bound_u64(yz_col1 + yz_lo, yz_len, z);
            matched = (found < yz_len && yz_col1[yz_lo + found] == z) ? 1u : 0u;
        }
        if (matched != 0u) {
            uint32_t out = thread_base + local_emit;
            if (out < total_rows) {
                out_x[out] = xy_col0[xy_idx];
                out_y[out] = xy_col1[xy_idx];
                out_z[out] = z;
            }
            local_emit += 1;
        }
    }
}

extern "C" __global__ void wcoj_triangle_count_hg_cached_u32(
    const uint32_t* __restrict__ xy_col0,
    const uint32_t* __restrict__ xy_col1,
    const uint32_t* __restrict__ yz_col1,
    uint32_t n_yz,
    const uint32_t* __restrict__ xz_col1,
    uint32_t n_xz,
    const uint32_t* __restrict__ xy_work_prefix,
    const uint32_t* __restrict__ xy_yz_start,
    const uint32_t* __restrict__ xy_yz_end,
    const uint32_t* __restrict__ xy_xz_start,
    const uint32_t* __restrict__ xy_xz_end,
    uint32_t n_xy,
    uint32_t total_work,
    uint32_t block_work_unit,
    uint32_t* __restrict__ out_block_counts,
    uint32_t* __restrict__ scratch_x,
    uint32_t* __restrict__ scratch_y,
    uint32_t* __restrict__ scratch_z) {
    uint32_t block_start = blockIdx.x * block_work_unit;
    if (block_start >= total_work) {
        if (threadIdx.x == 0) {
            out_block_counts[blockIdx.x] = 0;
        }
        return;
    }
    uint32_t block_end = block_start + block_work_unit;
    if (block_end < block_start || block_end > total_work) {
        block_end = total_work;
    }

    uint32_t local_count = 0;
    uint32_t local_x[16];
    uint32_t local_y[16];
    uint32_t local_z[16];
    for (uint32_t work_idx = block_start + threadIdx.x;
         work_idx < block_end;
         work_idx += blockDim.x) {
        uint32_t root_pos = upper_bound_u32(xy_work_prefix, n_xy + 1, work_idx);
        if (root_pos == 0) {
            continue;
        }
        uint32_t xy_idx = root_pos - 1;
        if (xy_idx >= n_xy) {
            continue;
        }
        uint32_t row_start = xy_work_prefix[xy_idx];
        uint32_t yz_lo = xy_yz_start[xy_idx];
        uint32_t yz_hi = xy_yz_end[xy_idx];
        uint32_t xz_lo = xy_xz_start[xy_idx];
        uint32_t xz_hi = xy_xz_end[xy_idx];
        if (yz_hi > n_yz || xz_hi > n_xz || yz_lo >= yz_hi || xz_lo >= xz_hi) {
            continue;
        }
        uint32_t yz_len = yz_hi - yz_lo;
        uint32_t xz_len = xz_hi - xz_lo;
        uint32_t probe_offset = work_idx - row_start;
        uint32_t matched = 0;
        if (yz_len <= xz_len) {
            if (probe_offset >= yz_len) {
                continue;
            }
            uint32_t z = yz_col1[yz_lo + probe_offset];
            uint32_t found = lower_bound_u32(xz_col1 + xz_lo, xz_len, z);
            matched = (found < xz_len && xz_col1[xz_lo + found] == z) ? 1u : 0u;
            if (matched != 0u && local_count < 16u) {
                local_x[local_count] = xy_col0[xy_idx];
                local_y[local_count] = xy_col1[xy_idx];
                local_z[local_count] = z;
            }
        } else {
            if (probe_offset >= xz_len) {
                continue;
            }
            uint32_t z = xz_col1[xz_lo + probe_offset];
            uint32_t found = lower_bound_u32(yz_col1 + yz_lo, yz_len, z);
            matched = (found < yz_len && yz_col1[yz_lo + found] == z) ? 1u : 0u;
            if (matched != 0u && local_count < 16u) {
                local_x[local_count] = xy_col0[xy_idx];
                local_y[local_count] = xy_col1[xy_idx];
                local_z[local_count] = z;
            }
        }
        if (matched != 0u) {
            local_count += 1;
        }
    }

    __shared__ uint32_t thread_prefix[1024];
    thread_prefix[threadIdx.x] = local_count;
    __syncthreads();
    for (uint32_t stride = 1; stride < blockDim.x; stride <<= 1) {
        uint32_t add = 0;
        if (threadIdx.x >= stride) {
            add = thread_prefix[threadIdx.x - stride];
        }
        __syncthreads();
        thread_prefix[threadIdx.x] += add;
        __syncthreads();
    }
    uint32_t thread_base = threadIdx.x == 0 ? 0 : thread_prefix[threadIdx.x - 1];
    if (threadIdx.x == blockDim.x - 1) {
        out_block_counts[blockIdx.x] = thread_prefix[threadIdx.x];
    }

    uint32_t scratch_base = blockIdx.x * block_work_unit + thread_base;
    for (uint32_t i = 0; i < local_count && i < 16u; ++i) {
        uint32_t out = scratch_base + i;
        scratch_x[out] = local_x[i];
        scratch_y[out] = local_y[i];
        scratch_z[out] = local_z[i];
    }
}

extern "C" __global__ void wcoj_triangle_materialize_hg_cached_u32(
    const uint32_t* __restrict__ block_counts,
    const uint32_t* __restrict__ block_offsets,
    uint32_t block_work_unit,
    uint32_t total_rows,
    const uint32_t* __restrict__ scratch_x,
    const uint32_t* __restrict__ scratch_y,
    const uint32_t* __restrict__ scratch_z,
    uint32_t* __restrict__ out_x,
    uint32_t* __restrict__ out_y,
    uint32_t* __restrict__ out_z) {
    uint32_t count = block_counts[blockIdx.x];
    uint32_t out_base = block_offsets[blockIdx.x];
    uint32_t scratch_base = blockIdx.x * block_work_unit;
    for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
        uint32_t out = out_base + i;
        if (out < total_rows) {
            uint32_t scratch = scratch_base + i;
            out_x[out] = scratch_x[scratch];
            out_y[out] = scratch_y[scratch];
            out_z[out] = scratch_z[scratch];
        }
    }
}

extern "C" __global__ void wcoj_scan_hg_block_counts_u32(
    const uint32_t* __restrict__ counts,
    uint32_t n,
    uint32_t* __restrict__ offsets,
    uint32_t* __restrict__ total_rows) {
    __shared__ uint32_t scan[1024];
    uint32_t tid = threadIdx.x;
    uint32_t value = (tid < n) ? counts[tid] : 0u;
    scan[tid] = value;
    __syncthreads();
    for (uint32_t stride = 1; stride < blockDim.x; stride <<= 1) {
        uint32_t add = 0;
        if (tid >= stride) {
            add = scan[tid - stride];
        }
        __syncthreads();
        scan[tid] += add;
        __syncthreads();
    }
    if (tid < n) {
        offsets[tid] = scan[tid] - value;
    }
    if (tid == 0) {
        *total_rows = (n == 0) ? 0u : scan[n - 1];
    }
}

// Single-thread reducer: total = counts[n-1] + offsets[n-1].
//
// `offsets` is produced by an in-place exclusive prefix-sum over
// the per-row counts (the device-side scan helper). The inclusive
// total — i.e. the number of triangles to materialize — is
// `offsets[n-1] + counts[n-1]`. Writing the scalar to device
// memory means the host can use the sanctioned
// `dtoh_scalar_untracked` chokepoint to read just the total.
extern "C" __global__ void wcoj_compute_total(
    const uint32_t* __restrict__ counts,
    const uint32_t* __restrict__ offsets,
    uint32_t n,
    uint32_t* __restrict__ total) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    if (n == 0) {
        *total = 0;
        return;
    }
    *total = counts[n - 1] + offsets[n - 1];
}

// ===============================================================
// 4-cycle HG WCOJ kernels (u32).
//
//   cycle4(W, X, Y, Z) :- e1(W, X), e2(X, Y), e3(Y, Z), e4(Z, W)
// ===============================================================

namespace {

// Test whether the pair (a, b) is present in a relation sorted lex
// by (col0, col1). First narrows to the col0 == a range via
// lower/upper bound, then binary-searches col1 within that range.
// Returns true iff some row equals (a, b).
__device__ __forceinline__ bool contains_pair_u32(
    const uint32_t* __restrict__ col0,
    const uint32_t* __restrict__ col1,
    uint32_t n,
    uint32_t a,
    uint32_t b) {
    uint32_t lo = lower_bound_u32(col0, n, a);
    uint32_t hi = upper_bound_u32(col0, n, a);
    if (lo == hi) {
        return false;
    }
    uint32_t inner_offset = lower_bound_u32(col1 + lo, hi - lo, b);
    uint32_t inner_idx = lo + inner_offset;
    return inner_idx < hi && col1[inner_idx] == b;
}

__device__ __forceinline__ bool resolve_4cycle_e2_work_u32(
    const uint32_t* __restrict__ e2_work_prefix,
    const uint32_t* __restrict__ e2_col1,
    const uint32_t* __restrict__ e3_col0,
    const uint32_t* __restrict__ e3_col1,
    uint32_t n_e3,
    uint32_t e2_lo,
    uint32_t e2_hi,
    uint32_t local,
    uint32_t* out_y,
    uint32_t* out_z) {
    if (e2_lo >= e2_hi) {
        return false;
    }
    uint32_t target = e2_work_prefix[e2_lo] + local;
    uint32_t span = e2_hi - e2_lo + 1;
    uint32_t pos = upper_bound_u32(e2_work_prefix + e2_lo, span, target);
    if (pos == 0) {
        return false;
    }
    uint32_t j = e2_lo + pos - 1;
    if (j >= e2_hi) {
        return false;
    }
    uint32_t e3_offset = target - e2_work_prefix[j];
    uint32_t y = e2_col1[j];
    uint32_t e3_lo = lower_bound_u32(e3_col0, n_e3, y);
    uint32_t e3_hi = upper_bound_u32(e3_col0, n_e3, y);
    if (e3_offset >= e3_hi - e3_lo) {
        return false;
    }
    *out_y = y;
    *out_z = e3_col1[e3_lo + e3_offset];
    return true;
}

}  // anonymous namespace

extern "C" __global__ void wcoj_4cycle_build_e2_work_prefix_u32(
    const uint32_t* __restrict__ e2_col1,
    uint32_t n_e2,
    const uint32_t* __restrict__ e3_col0,
    uint32_t n_e3,
    uint32_t* __restrict__ e2_work_prefix) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        e2_work_prefix[n_e2] = 0;
    }
    if (i >= n_e2) {
        return;
    }
    uint32_t y = e2_col1[i];
    uint32_t e3_lo = lower_bound_u32(e3_col0, n_e3, y);
    uint32_t e3_hi = upper_bound_u32(e3_col0, n_e3, y);
    e2_work_prefix[i] = e3_hi - e3_lo;
}

extern "C" __global__ void wcoj_4cycle_build_hg_work_plan_u32(
    const uint32_t* __restrict__ e1_col1,
    uint32_t n_e1,
    const uint32_t* __restrict__ e2_col0,
    uint32_t n_e2,
    const uint32_t* __restrict__ e2_work_prefix,
    uint32_t* __restrict__ e1_work_prefix,
    uint32_t* __restrict__ e1_e2_start,
    uint32_t* __restrict__ e1_e2_end) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        e1_work_prefix[n_e1] = 0;
    }
    if (i >= n_e1) {
        return;
    }
    uint32_t x = e1_col1[i];
    uint32_t e2_lo = lower_bound_u32(e2_col0, n_e2, x);
    uint32_t e2_hi = upper_bound_u32(e2_col0, n_e2, x);
    uint32_t work = e2_work_prefix[e2_hi] - e2_work_prefix[e2_lo];
    e1_work_prefix[i] = work;
    e1_e2_start[i] = e2_lo;
    e1_e2_end[i] = e2_hi;
}

extern "C" __global__ void wcoj_4cycle_count_hg_u32(
    const uint32_t* __restrict__ e1_col0,
    const uint32_t* __restrict__ e1_col1,
    uint32_t n_e1,
    const uint32_t* __restrict__ e2_col1,
    const uint32_t* __restrict__ e3_col0,
    const uint32_t* __restrict__ e3_col1,
    uint32_t n_e3,
    const uint32_t* __restrict__ e4_col0,
    const uint32_t* __restrict__ e4_col1,
    uint32_t n_e4,
    const uint32_t* __restrict__ e1_work_prefix,
    const uint32_t* __restrict__ e2_work_prefix,
    const uint32_t* __restrict__ e1_e2_start,
    const uint32_t* __restrict__ e1_e2_end,
    uint32_t total_work,
    uint32_t block_work_unit,
    uint32_t* __restrict__ out_block_counts) {
    uint32_t block_start = blockIdx.x * block_work_unit;
    if (block_start >= total_work) {
        if (threadIdx.x == 0) {
            out_block_counts[blockIdx.x] = 0;
        }
        return;
    }
    uint32_t block_end = block_start + block_work_unit;
    if (block_end < block_start || block_end > total_work) {
        block_end = total_work;
    }

    uint32_t local_count = 0;
    for (uint32_t work_idx = block_start + threadIdx.x;
         work_idx < block_end;
         work_idx += blockDim.x) {
        uint32_t root_pos = upper_bound_u32(e1_work_prefix, n_e1 + 1, work_idx);
        if (root_pos == 0) {
            continue;
        }
        uint32_t e1_idx = root_pos - 1;
        if (e1_idx >= n_e1) {
            continue;
        }
        uint32_t local = work_idx - e1_work_prefix[e1_idx];
        uint32_t w = e1_col0[e1_idx];
        uint32_t e2_lo = e1_e2_start[e1_idx];
        uint32_t e2_hi = e1_e2_end[e1_idx];
        uint32_t y = 0;
        uint32_t z = 0;
        if (resolve_4cycle_e2_work_u32(
                e2_work_prefix, e2_col1, e3_col0, e3_col1, n_e3, e2_lo, e2_hi, local, &y, &z)
            && contains_pair_u32(e4_col0, e4_col1, n_e4, z, w)) {
            local_count += 1;
        }
    }

    __shared__ uint32_t partial[256];
    partial[threadIdx.x] = local_count;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out_block_counts[blockIdx.x] = partial[0];
    }
}

// S1c aggregate-fused variant of wcoj_4cycle_count_hg_u32: instead of
// reducing matches to per-block totals, each completed 4-cycle increments
// the counter of its e1 root row (`out_row_counts[e1_idx]`, length n_e1,
// zero-initialized by the host). e1 is lex-sorted by (W, X), so per-W group
// counts are a contiguous-range reduction downstream. Integer atomicAdd is
// order-insensitive, so the final counter values are deterministic
// (precedent: wcoj_triangle_groupby_root_count_hg_u32). The 4-cycle rows
// are never materialized. `e1_col1` is kept for ABI symmetry with the
// count/materialize siblings.
extern "C" __global__ void wcoj_4cycle_groupby_root_count_hg_u32(
    const uint32_t* __restrict__ e1_col0,
    const uint32_t* __restrict__ e1_col1,
    uint32_t n_e1,
    const uint32_t* __restrict__ e2_col1,
    const uint32_t* __restrict__ e3_col0,
    const uint32_t* __restrict__ e3_col1,
    uint32_t n_e3,
    const uint32_t* __restrict__ e4_col0,
    const uint32_t* __restrict__ e4_col1,
    uint32_t n_e4,
    const uint32_t* __restrict__ e1_work_prefix,
    const uint32_t* __restrict__ e2_work_prefix,
    const uint32_t* __restrict__ e1_e2_start,
    const uint32_t* __restrict__ e1_e2_end,
    uint32_t total_work,
    uint32_t block_work_unit,
    uint32_t* __restrict__ out_row_counts) {
    (void)e1_col1;
    uint32_t block_start = blockIdx.x * block_work_unit;
    if (block_start >= total_work) {
        return;
    }
    uint32_t block_end = block_start + block_work_unit;
    if (block_end < block_start || block_end > total_work) {
        block_end = total_work;
    }

    for (uint32_t work_idx = block_start + threadIdx.x;
         work_idx < block_end;
         work_idx += blockDim.x) {
        uint32_t root_pos = upper_bound_u32(e1_work_prefix, n_e1 + 1, work_idx);
        if (root_pos == 0) {
            continue;
        }
        uint32_t e1_idx = root_pos - 1;
        if (e1_idx >= n_e1) {
            continue;
        }
        uint32_t local = work_idx - e1_work_prefix[e1_idx];
        uint32_t w = e1_col0[e1_idx];
        uint32_t e2_lo = e1_e2_start[e1_idx];
        uint32_t e2_hi = e1_e2_end[e1_idx];
        uint32_t y = 0;
        uint32_t z = 0;
        if (resolve_4cycle_e2_work_u32(
                e2_work_prefix, e2_col1, e3_col0, e3_col1, n_e3, e2_lo, e2_hi, local, &y, &z)
            && contains_pair_u32(e4_col0, e4_col1, n_e4, z, w)) {
            atomicAdd(&out_row_counts[e1_idx], 1u);
        }
    }
}

namespace {

// S1d helper: resolve one flattened work index to its e1 root row and, on
// a completed 4-cycle, the aggregate value (X = e1.col1 when
// `value_sel == 0`, the resolved Y when 1, the matched Z when 2).
// Traversal is identical to `wcoj_4cycle_groupby_root_count_hg_u32`.
// Returns false for padding / out-of-range / unmatched work items.
__device__ __forceinline__ bool groupby_root_4cycle_match_value_u32(
    const uint32_t* __restrict__ e1_col0,
    const uint32_t* __restrict__ e1_col1,
    uint32_t n_e1,
    const uint32_t* __restrict__ e2_col1,
    const uint32_t* __restrict__ e3_col0,
    const uint32_t* __restrict__ e3_col1,
    uint32_t n_e3,
    const uint32_t* __restrict__ e4_col0,
    const uint32_t* __restrict__ e4_col1,
    uint32_t n_e4,
    uint32_t value_sel,
    const uint32_t* __restrict__ e1_work_prefix,
    const uint32_t* __restrict__ e2_work_prefix,
    const uint32_t* __restrict__ e1_e2_start,
    const uint32_t* __restrict__ e1_e2_end,
    uint32_t work_idx,
    uint32_t* __restrict__ out_e1_idx,
    uint32_t* __restrict__ out_value) {
    uint32_t root_pos = upper_bound_u32(e1_work_prefix, n_e1 + 1, work_idx);
    if (root_pos == 0) {
        return false;
    }
    uint32_t e1_idx = root_pos - 1;
    if (e1_idx >= n_e1) {
        return false;
    }
    uint32_t local = work_idx - e1_work_prefix[e1_idx];
    uint32_t w = e1_col0[e1_idx];
    uint32_t e2_lo = e1_e2_start[e1_idx];
    uint32_t e2_hi = e1_e2_end[e1_idx];
    uint32_t y = 0;
    uint32_t z = 0;
    if (!resolve_4cycle_e2_work_u32(
            e2_work_prefix, e2_col1, e3_col0, e3_col1, n_e3, e2_lo, e2_hi, local, &y, &z)
        || !contains_pair_u32(e4_col0, e4_col1, n_e4, z, w)) {
        return false;
    }
    *out_e1_idx = e1_idx;
    *out_value = (value_sel == 0u) ? e1_col1[e1_idx] : (value_sel == 1u ? y : z);
    return true;
}

}  // anonymous namespace

// S1d: sum/min/max variants of the aggregate-fused 4-cycle root kernel.
// Each completed 4-cycle contributes its value (X, Y or Z — all 4-cycle
// output variables) to the e1 root row's accumulator, alongside a match
// counter (`out_row_counts`) that downstream compaction uses to drop roots
// with no completion. Integer atomics are order-insensitive, so final
// accumulator values are deterministic. Host pre-initializes accumulators:
// 0 for sum/max, 0xFFFFFFFF for min.
extern "C" __global__ void wcoj_4cycle_groupby_root_sum_hg_u32(
    const uint32_t* __restrict__ e1_col0,
    const uint32_t* __restrict__ e1_col1,
    uint32_t n_e1,
    const uint32_t* __restrict__ e2_col1,
    const uint32_t* __restrict__ e3_col0,
    const uint32_t* __restrict__ e3_col1,
    uint32_t n_e3,
    const uint32_t* __restrict__ e4_col0,
    const uint32_t* __restrict__ e4_col1,
    uint32_t n_e4,
    uint32_t value_sel,
    const uint32_t* __restrict__ e1_work_prefix,
    const uint32_t* __restrict__ e2_work_prefix,
    const uint32_t* __restrict__ e1_e2_start,
    const uint32_t* __restrict__ e1_e2_end,
    uint32_t total_work,
    uint32_t block_work_unit,
    uint32_t* __restrict__ out_row_counts,
    unsigned long long* __restrict__ out_row_sums) {
    uint32_t block_start = blockIdx.x * block_work_unit;
    if (block_start >= total_work) {
        return;
    }
    uint32_t block_end = block_start + block_work_unit;
    if (block_end < block_start || block_end > total_work) {
        block_end = total_work;
    }
    for (uint32_t work_idx = block_start + threadIdx.x;
         work_idx < block_end;
         work_idx += blockDim.x) {
        uint32_t e1_idx = 0;
        uint32_t value = 0;
        if (groupby_root_4cycle_match_value_u32(
                e1_col0, e1_col1, n_e1, e2_col1, e3_col0, e3_col1, n_e3,
                e4_col0, e4_col1, n_e4, value_sel, e1_work_prefix,
                e2_work_prefix, e1_e2_start, e1_e2_end, work_idx, &e1_idx,
                &value)) {
            atomicAdd(&out_row_counts[e1_idx], 1u);
            atomicAdd(&out_row_sums[e1_idx], (unsigned long long)value);
        }
    }
}

extern "C" __global__ void wcoj_4cycle_groupby_root_min_hg_u32(
    const uint32_t* __restrict__ e1_col0,
    const uint32_t* __restrict__ e1_col1,
    uint32_t n_e1,
    const uint32_t* __restrict__ e2_col1,
    const uint32_t* __restrict__ e3_col0,
    const uint32_t* __restrict__ e3_col1,
    uint32_t n_e3,
    const uint32_t* __restrict__ e4_col0,
    const uint32_t* __restrict__ e4_col1,
    uint32_t n_e4,
    uint32_t value_sel,
    const uint32_t* __restrict__ e1_work_prefix,
    const uint32_t* __restrict__ e2_work_prefix,
    const uint32_t* __restrict__ e1_e2_start,
    const uint32_t* __restrict__ e1_e2_end,
    uint32_t total_work,
    uint32_t block_work_unit,
    uint32_t* __restrict__ out_row_counts,
    uint32_t* __restrict__ out_row_mins) {
    uint32_t block_start = blockIdx.x * block_work_unit;
    if (block_start >= total_work) {
        return;
    }
    uint32_t block_end = block_start + block_work_unit;
    if (block_end < block_start || block_end > total_work) {
        block_end = total_work;
    }
    for (uint32_t work_idx = block_start + threadIdx.x;
         work_idx < block_end;
         work_idx += blockDim.x) {
        uint32_t e1_idx = 0;
        uint32_t value = 0;
        if (groupby_root_4cycle_match_value_u32(
                e1_col0, e1_col1, n_e1, e2_col1, e3_col0, e3_col1, n_e3,
                e4_col0, e4_col1, n_e4, value_sel, e1_work_prefix,
                e2_work_prefix, e1_e2_start, e1_e2_end, work_idx, &e1_idx,
                &value)) {
            atomicAdd(&out_row_counts[e1_idx], 1u);
            atomicMin(&out_row_mins[e1_idx], value);
        }
    }
}

extern "C" __global__ void wcoj_4cycle_groupby_root_max_hg_u32(
    const uint32_t* __restrict__ e1_col0,
    const uint32_t* __restrict__ e1_col1,
    uint32_t n_e1,
    const uint32_t* __restrict__ e2_col1,
    const uint32_t* __restrict__ e3_col0,
    const uint32_t* __restrict__ e3_col1,
    uint32_t n_e3,
    const uint32_t* __restrict__ e4_col0,
    const uint32_t* __restrict__ e4_col1,
    uint32_t n_e4,
    uint32_t value_sel,
    const uint32_t* __restrict__ e1_work_prefix,
    const uint32_t* __restrict__ e2_work_prefix,
    const uint32_t* __restrict__ e1_e2_start,
    const uint32_t* __restrict__ e1_e2_end,
    uint32_t total_work,
    uint32_t block_work_unit,
    uint32_t* __restrict__ out_row_counts,
    uint32_t* __restrict__ out_row_maxs) {
    uint32_t block_start = blockIdx.x * block_work_unit;
    if (block_start >= total_work) {
        return;
    }
    uint32_t block_end = block_start + block_work_unit;
    if (block_end < block_start || block_end > total_work) {
        block_end = total_work;
    }
    for (uint32_t work_idx = block_start + threadIdx.x;
         work_idx < block_end;
         work_idx += blockDim.x) {
        uint32_t e1_idx = 0;
        uint32_t value = 0;
        if (groupby_root_4cycle_match_value_u32(
                e1_col0, e1_col1, n_e1, e2_col1, e3_col0, e3_col1, n_e3,
                e4_col0, e4_col1, n_e4, value_sel, e1_work_prefix,
                e2_work_prefix, e1_e2_start, e1_e2_end, work_idx, &e1_idx,
                &value)) {
            atomicAdd(&out_row_counts[e1_idx], 1u);
            atomicMax(&out_row_maxs[e1_idx], value);
        }
    }
}

extern "C" __global__ void wcoj_4cycle_materialize_hg_u32(
    const uint32_t* __restrict__ e1_col0,
    const uint32_t* __restrict__ e1_col1,
    uint32_t n_e1,
    const uint32_t* __restrict__ e2_col1,
    const uint32_t* __restrict__ e3_col0,
    const uint32_t* __restrict__ e3_col1,
    uint32_t n_e3,
    const uint32_t* __restrict__ e4_col0,
    const uint32_t* __restrict__ e4_col1,
    uint32_t n_e4,
    const uint32_t* __restrict__ e1_work_prefix,
    const uint32_t* __restrict__ e2_work_prefix,
    const uint32_t* __restrict__ e1_e2_start,
    const uint32_t* __restrict__ e1_e2_end,
    uint32_t total_work,
    uint32_t block_work_unit,
    const uint32_t* __restrict__ block_offsets,
    uint32_t total_rows,
    uint32_t* __restrict__ out_w,
    uint32_t* __restrict__ out_x,
    uint32_t* __restrict__ out_y,
    uint32_t* __restrict__ out_z) {
    uint32_t block_start = blockIdx.x * block_work_unit;
    if (block_start >= total_work) {
        return;
    }
    uint32_t block_end = block_start + block_work_unit;
    if (block_end < block_start || block_end > total_work) {
        block_end = total_work;
    }

    uint32_t local_count = 0;
    for (uint32_t work_idx = block_start + threadIdx.x;
         work_idx < block_end;
         work_idx += blockDim.x) {
        uint32_t root_pos = upper_bound_u32(e1_work_prefix, n_e1 + 1, work_idx);
        if (root_pos == 0) {
            continue;
        }
        uint32_t e1_idx = root_pos - 1;
        if (e1_idx >= n_e1) {
            continue;
        }
        uint32_t local = work_idx - e1_work_prefix[e1_idx];
        uint32_t w = e1_col0[e1_idx];
        uint32_t e2_lo = e1_e2_start[e1_idx];
        uint32_t e2_hi = e1_e2_end[e1_idx];
        uint32_t y = 0;
        uint32_t z = 0;
        if (resolve_4cycle_e2_work_u32(
                e2_work_prefix, e2_col1, e3_col0, e3_col1, n_e3, e2_lo, e2_hi, local, &y, &z)
            && contains_pair_u32(e4_col0, e4_col1, n_e4, z, w)) {
            local_count += 1;
        }
    }

    __shared__ uint32_t thread_prefix[256];
    thread_prefix[threadIdx.x] = local_count;
    __syncthreads();
    for (uint32_t stride = 1; stride < blockDim.x; stride <<= 1) {
        uint32_t add = 0;
        if (threadIdx.x >= stride) {
            add = thread_prefix[threadIdx.x - stride];
        }
        __syncthreads();
        thread_prefix[threadIdx.x] += add;
        __syncthreads();
    }
    uint32_t thread_base = block_offsets[blockIdx.x];
    if (threadIdx.x > 0) {
        thread_base += thread_prefix[threadIdx.x - 1];
    }

    uint32_t local_emit = 0;
    for (uint32_t work_idx = block_start + threadIdx.x;
         work_idx < block_end;
         work_idx += blockDim.x) {
        uint32_t root_pos = upper_bound_u32(e1_work_prefix, n_e1 + 1, work_idx);
        if (root_pos == 0) {
            continue;
        }
        uint32_t e1_idx = root_pos - 1;
        if (e1_idx >= n_e1) {
            continue;
        }
        uint32_t local = work_idx - e1_work_prefix[e1_idx];
        uint32_t w = e1_col0[e1_idx];
        uint32_t x = e1_col1[e1_idx];
        uint32_t e2_lo = e1_e2_start[e1_idx];
        uint32_t e2_hi = e1_e2_end[e1_idx];
        uint32_t y = 0;
        uint32_t z = 0;
        if (resolve_4cycle_e2_work_u32(
                e2_work_prefix, e2_col1, e3_col0, e3_col1, n_e3, e2_lo, e2_hi, local, &y, &z)
            && contains_pair_u32(e4_col0, e4_col1, n_e4, z, w)) {
            uint32_t out = thread_base + local_emit;
            if (out < total_rows) {
                out_w[out] = w;
                out_x[out] = x;
                out_y[out] = y;
                out_z[out] = z;
            }
            local_emit += 1;
        }
    }
}

// ===============================================================
// 4-cycle HG WCOJ kernels (u64).
// Same block-slice shape as the u32 pair, with 64-bit join-key buffers.
// Counters / counts / offsets / row_count stay u32 (bounded by
// the host-side row-count guard upstream).
// ===============================================================

namespace {

__device__ __forceinline__ bool contains_pair_u64(
    const uint64_t* __restrict__ col0,
    const uint64_t* __restrict__ col1,
    uint32_t n,
    uint64_t a,
    uint64_t b) {
    uint32_t lo = lower_bound_u64(col0, n, a);
    uint32_t hi = upper_bound_u64(col0, n, a);
    if (lo == hi) {
        return false;
    }
    uint32_t inner_offset = lower_bound_u64(col1 + lo, hi - lo, b);
    uint32_t inner_idx = lo + inner_offset;
    return inner_idx < hi && col1[inner_idx] == b;
}

__device__ __forceinline__ bool resolve_4cycle_e2_work_u64(
    const uint32_t* __restrict__ e2_work_prefix,
    const uint64_t* __restrict__ e2_col1,
    const uint64_t* __restrict__ e3_col0,
    const uint64_t* __restrict__ e3_col1,
    uint32_t n_e3,
    uint32_t e2_lo,
    uint32_t e2_hi,
    uint32_t local,
    uint64_t* out_y,
    uint64_t* out_z) {
    if (e2_lo >= e2_hi) {
        return false;
    }
    uint32_t target = e2_work_prefix[e2_lo] + local;
    uint32_t span = e2_hi - e2_lo + 1;
    uint32_t pos = upper_bound_u32(e2_work_prefix + e2_lo, span, target);
    if (pos == 0) {
        return false;
    }
    uint32_t j = e2_lo + pos - 1;
    if (j >= e2_hi) {
        return false;
    }
    uint32_t e3_offset = target - e2_work_prefix[j];
    uint64_t y = e2_col1[j];
    uint32_t e3_lo = lower_bound_u64(e3_col0, n_e3, y);
    uint32_t e3_hi = upper_bound_u64(e3_col0, n_e3, y);
    if (e3_offset >= e3_hi - e3_lo) {
        return false;
    }
    *out_y = y;
    *out_z = e3_col1[e3_lo + e3_offset];
    return true;
}

}  // anonymous namespace

extern "C" __global__ void wcoj_4cycle_build_e2_work_prefix_u64(
    const uint64_t* __restrict__ e2_col1,
    uint32_t n_e2,
    const uint64_t* __restrict__ e3_col0,
    uint32_t n_e3,
    uint32_t* __restrict__ e2_work_prefix) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        e2_work_prefix[n_e2] = 0;
    }
    if (i >= n_e2) {
        return;
    }
    uint64_t y = e2_col1[i];
    uint32_t e3_lo = lower_bound_u64(e3_col0, n_e3, y);
    uint32_t e3_hi = upper_bound_u64(e3_col0, n_e3, y);
    e2_work_prefix[i] = e3_hi - e3_lo;
}

extern "C" __global__ void wcoj_4cycle_build_hg_work_plan_u64(
    const uint64_t* __restrict__ e1_col1,
    uint32_t n_e1,
    const uint64_t* __restrict__ e2_col0,
    uint32_t n_e2,
    const uint32_t* __restrict__ e2_work_prefix,
    uint32_t* __restrict__ e1_work_prefix,
    uint32_t* __restrict__ e1_e2_start,
    uint32_t* __restrict__ e1_e2_end) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        e1_work_prefix[n_e1] = 0;
    }
    if (i >= n_e1) {
        return;
    }
    uint64_t x = e1_col1[i];
    uint32_t e2_lo = lower_bound_u64(e2_col0, n_e2, x);
    uint32_t e2_hi = upper_bound_u64(e2_col0, n_e2, x);
    uint32_t work = e2_work_prefix[e2_hi] - e2_work_prefix[e2_lo];
    e1_work_prefix[i] = work;
    e1_e2_start[i] = e2_lo;
    e1_e2_end[i] = e2_hi;
}

extern "C" __global__ void wcoj_4cycle_count_hg_u64(
    const uint64_t* __restrict__ e1_col0,
    const uint64_t* __restrict__ e1_col1,
    uint32_t n_e1,
    const uint64_t* __restrict__ e2_col1,
    const uint64_t* __restrict__ e3_col0,
    const uint64_t* __restrict__ e3_col1,
    uint32_t n_e3,
    const uint64_t* __restrict__ e4_col0,
    const uint64_t* __restrict__ e4_col1,
    uint32_t n_e4,
    const uint32_t* __restrict__ e1_work_prefix,
    const uint32_t* __restrict__ e2_work_prefix,
    const uint32_t* __restrict__ e1_e2_start,
    const uint32_t* __restrict__ e1_e2_end,
    uint32_t total_work,
    uint32_t block_work_unit,
    uint32_t* __restrict__ out_block_counts) {
    uint32_t block_start = blockIdx.x * block_work_unit;
    if (block_start >= total_work) {
        if (threadIdx.x == 0) {
            out_block_counts[blockIdx.x] = 0;
        }
        return;
    }
    uint32_t block_end = block_start + block_work_unit;
    if (block_end < block_start || block_end > total_work) {
        block_end = total_work;
    }

    uint32_t local_count = 0;
    for (uint32_t work_idx = block_start + threadIdx.x;
         work_idx < block_end;
         work_idx += blockDim.x) {
        uint32_t root_pos = upper_bound_u32(e1_work_prefix, n_e1 + 1, work_idx);
        if (root_pos == 0) {
            continue;
        }
        uint32_t e1_idx = root_pos - 1;
        if (e1_idx >= n_e1) {
            continue;
        }
        uint32_t local = work_idx - e1_work_prefix[e1_idx];
        uint64_t w = e1_col0[e1_idx];
        uint32_t e2_lo = e1_e2_start[e1_idx];
        uint32_t e2_hi = e1_e2_end[e1_idx];
        uint64_t y = 0;
        uint64_t z = 0;
        if (resolve_4cycle_e2_work_u64(
                e2_work_prefix, e2_col1, e3_col0, e3_col1, n_e3, e2_lo, e2_hi, local, &y, &z)
            && contains_pair_u64(e4_col0, e4_col1, n_e4, z, w)) {
            local_count += 1;
        }
    }

    __shared__ uint32_t partial[256];
    partial[threadIdx.x] = local_count;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out_block_counts[blockIdx.x] = partial[0];
    }
}

// S1d slice 2: u64-key aggregate-fused variant of wcoj_4cycle_count_hg_u64.
// Instead of reducing matches to per-block totals, each completed 4-cycle
// increments the counter of its e1 root row (`out_row_counts[e1_idx]`,
// length n_e1, zero-initialized by the host). e1 is lex-sorted by (W, X),
// so per-W group counts are a contiguous-range reduction downstream
// (relation-metadata segment sum — the recorded groupby is U32/Symbol-key
// only). Integer atomicAdd is order-insensitive, so the final counter
// values are deterministic (precedent:
// wcoj_triangle_groupby_root_count_hg_u64). The 4-cycle rows are never
// materialized. `e1_col1` is kept for ABI symmetry with the
// count/materialize siblings.
extern "C" __global__ void wcoj_4cycle_groupby_root_count_hg_u64(
    const uint64_t* __restrict__ e1_col0,
    const uint64_t* __restrict__ e1_col1,
    uint32_t n_e1,
    const uint64_t* __restrict__ e2_col1,
    const uint64_t* __restrict__ e3_col0,
    const uint64_t* __restrict__ e3_col1,
    uint32_t n_e3,
    const uint64_t* __restrict__ e4_col0,
    const uint64_t* __restrict__ e4_col1,
    uint32_t n_e4,
    const uint32_t* __restrict__ e1_work_prefix,
    const uint32_t* __restrict__ e2_work_prefix,
    const uint32_t* __restrict__ e1_e2_start,
    const uint32_t* __restrict__ e1_e2_end,
    uint32_t total_work,
    uint32_t block_work_unit,
    uint32_t* __restrict__ out_row_counts) {
    (void)e1_col1;
    uint32_t block_start = blockIdx.x * block_work_unit;
    if (block_start >= total_work) {
        return;
    }
    uint32_t block_end = block_start + block_work_unit;
    if (block_end < block_start || block_end > total_work) {
        block_end = total_work;
    }

    for (uint32_t work_idx = block_start + threadIdx.x;
         work_idx < block_end;
         work_idx += blockDim.x) {
        uint32_t root_pos = upper_bound_u32(e1_work_prefix, n_e1 + 1, work_idx);
        if (root_pos == 0) {
            continue;
        }
        uint32_t e1_idx = root_pos - 1;
        if (e1_idx >= n_e1) {
            continue;
        }
        uint32_t local = work_idx - e1_work_prefix[e1_idx];
        uint64_t w = e1_col0[e1_idx];
        uint32_t e2_lo = e1_e2_start[e1_idx];
        uint32_t e2_hi = e1_e2_end[e1_idx];
        uint64_t y = 0;
        uint64_t z = 0;
        if (resolve_4cycle_e2_work_u64(
                e2_work_prefix, e2_col1, e3_col0, e3_col1, n_e3, e2_lo, e2_hi, local, &y, &z)
            && contains_pair_u64(e4_col0, e4_col1, n_e4, z, w)) {
            atomicAdd(&out_row_counts[e1_idx], 1u);
        }
    }
}

extern "C" __global__ void wcoj_4cycle_materialize_hg_u64(
    const uint64_t* __restrict__ e1_col0,
    const uint64_t* __restrict__ e1_col1,
    uint32_t n_e1,
    const uint64_t* __restrict__ e2_col1,
    const uint64_t* __restrict__ e3_col0,
    const uint64_t* __restrict__ e3_col1,
    uint32_t n_e3,
    const uint64_t* __restrict__ e4_col0,
    const uint64_t* __restrict__ e4_col1,
    uint32_t n_e4,
    const uint32_t* __restrict__ e1_work_prefix,
    const uint32_t* __restrict__ e2_work_prefix,
    const uint32_t* __restrict__ e1_e2_start,
    const uint32_t* __restrict__ e1_e2_end,
    uint32_t total_work,
    uint32_t block_work_unit,
    const uint32_t* __restrict__ block_offsets,
    uint32_t total_rows,
    uint64_t* __restrict__ out_w,
    uint64_t* __restrict__ out_x,
    uint64_t* __restrict__ out_y,
    uint64_t* __restrict__ out_z) {
    uint32_t block_start = blockIdx.x * block_work_unit;
    if (block_start >= total_work) {
        return;
    }
    uint32_t block_end = block_start + block_work_unit;
    if (block_end < block_start || block_end > total_work) {
        block_end = total_work;
    }

    uint32_t local_count = 0;
    for (uint32_t work_idx = block_start + threadIdx.x;
         work_idx < block_end;
         work_idx += blockDim.x) {
        uint32_t root_pos = upper_bound_u32(e1_work_prefix, n_e1 + 1, work_idx);
        if (root_pos == 0) {
            continue;
        }
        uint32_t e1_idx = root_pos - 1;
        if (e1_idx >= n_e1) {
            continue;
        }
        uint32_t local = work_idx - e1_work_prefix[e1_idx];
        uint64_t w = e1_col0[e1_idx];
        uint32_t e2_lo = e1_e2_start[e1_idx];
        uint32_t e2_hi = e1_e2_end[e1_idx];
        uint64_t y = 0;
        uint64_t z = 0;
        if (resolve_4cycle_e2_work_u64(
                e2_work_prefix, e2_col1, e3_col0, e3_col1, n_e3, e2_lo, e2_hi, local, &y, &z)
            && contains_pair_u64(e4_col0, e4_col1, n_e4, z, w)) {
            local_count += 1;
        }
    }

    __shared__ uint32_t thread_prefix[256];
    thread_prefix[threadIdx.x] = local_count;
    __syncthreads();
    for (uint32_t stride = 1; stride < blockDim.x; stride <<= 1) {
        uint32_t add = 0;
        if (threadIdx.x >= stride) {
            add = thread_prefix[threadIdx.x - stride];
        }
        __syncthreads();
        thread_prefix[threadIdx.x] += add;
        __syncthreads();
    }
    uint32_t thread_base = block_offsets[blockIdx.x];
    if (threadIdx.x > 0) {
        thread_base += thread_prefix[threadIdx.x - 1];
    }

    uint32_t local_emit = 0;
    for (uint32_t work_idx = block_start + threadIdx.x;
         work_idx < block_end;
         work_idx += blockDim.x) {
        uint32_t root_pos = upper_bound_u32(e1_work_prefix, n_e1 + 1, work_idx);
        if (root_pos == 0) {
            continue;
        }
        uint32_t e1_idx = root_pos - 1;
        if (e1_idx >= n_e1) {
            continue;
        }
        uint32_t local = work_idx - e1_work_prefix[e1_idx];
        uint64_t w = e1_col0[e1_idx];
        uint64_t x = e1_col1[e1_idx];
        uint32_t e2_lo = e1_e2_start[e1_idx];
        uint32_t e2_hi = e1_e2_end[e1_idx];
        uint64_t y = 0;
        uint64_t z = 0;
        if (resolve_4cycle_e2_work_u64(
                e2_work_prefix, e2_col1, e3_col0, e3_col1, n_e3, e2_lo, e2_hi, local, &y, &z)
            && contains_pair_u64(e4_col0, e4_col1, n_e4, z, w)) {
            uint32_t out = thread_base + local_emit;
            if (out < total_rows) {
                out_w[out] = w;
                out_x[out] = x;
                out_y[out] = y;
                out_z[out] = z;
            }
            local_emit += 1;
        }
    }
}

// ===============================================================
// v0.6.2 — WCOJ layout fast-path device checker.
//
// Returns flag = 1 iff the 2-column input is strictly lex-sorted
// AND full-row unique. Caller initializes flag to 1; each thread
// owns one adjacent pair (col0[i-1], col1[i-1]) vs (col0[i], col1[i]).
// On any non-strict-increase, atomicExch(flag, 0) clears it.
// `n` is the LOGICAL row count (not row_cap); caller resolves it
// upstream via `logical_row_count_u32`.
//
// Strict-increase is the right check because
// dedup_full_row_recorded's contract is sorted+unique; we want
// to bypass it iff we'd produce the same output.
//
// Threads with i == 0 or i >= n early-out (the "previous row"
// is undefined). Empty / single-row inputs are handled by the
// host (see provider/wcoj.rs).
// ===============================================================

extern "C" __global__ void wcoj_layout_check_sorted_unique_u32(
    const uint32_t* __restrict__ col0,
    const uint32_t* __restrict__ col1,
    uint32_t n,
    uint32_t* __restrict__ flag) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0 || i >= n) {
        return;
    }
    uint32_t a0 = col0[i - 1];
    uint32_t a1 = col1[i - 1];
    uint32_t b0 = col0[i];
    uint32_t b1 = col1[i];
    // Strict lex-increase: (a0, a1) < (b0, b1).
    bool strictly_less = (a0 < b0) || (a0 == b0 && a1 < b1);
    if (!strictly_less) {
        atomicExch(flag, 0u);
    }
}

extern "C" __global__ void wcoj_layout_check_sorted_unique_u64(
    const uint64_t* __restrict__ col0,
    const uint64_t* __restrict__ col1,
    uint32_t n,
    uint32_t* __restrict__ flag) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0 || i >= n) {
        return;
    }
    uint64_t a0 = col0[i - 1];
    uint64_t a1 = col1[i - 1];
    uint64_t b0 = col0[i];
    uint64_t b1 = col1[i];
    bool strictly_less = (a0 < b0) || (a0 == b0 && a1 < b1);
    if (!strictly_less) {
        atomicExch(flag, 0u);
    }
}

// ===============================================================
// W3.2 — General-arity WCOJ clique kernel (K = 5, 6).
// Single C++ template instantiated at K=5 and K=6 from the same
// source. The ABI wrappers are template-call-only; no hand-written
// K-specific algorithm body lives behind any wrapper.
//
// Paper §5: each thread fixes one plan-selected leader-edge row.
// The first two binding positions are read from the selected row.
// Recursively bind the remaining positions via level-keyed template
// recursion. At each level L, the candidate set is the L-way
// intersection of plan-ordered edges whose col0 is an already-bound
// position and whose col1 is the current position. Runtime dispatch
// orients and layouts each edge according to KCliqueVariableOrder,
// then supplies leader_edge_idx, edge_order, and iteration_order.
// ===============================================================

// Edge index for an unordered pair with 0 <= i < j < K_VAL.
template <int K_VAL>
__device__ __forceinline__ int clique_edge_idx_t(int i, int j) {
    return i * (K_VAL - 1) - i * (i - 1) / 2 + (j - i - 1);
}

template <int K_VAL>
__device__ __forceinline__ int clique_edge_idx_unordered_t(int a, int b) {
    int lo = a < b ? a : b;
    int hi = a < b ? b : a;
    return clique_edge_idx_t<K_VAL>(lo, hi);
}

__device__ __forceinline__ int plan_or_identity_slot(
    const uint8_t* order,
    int fallback) {
    return order == nullptr ? fallback : static_cast<int>(order[fallback]);
}

// Binary-search membership: does sorted [arr+lo, arr+hi) contain target?
template <typename T>
__device__ __forceinline__ bool contains_in_range_t(
    const T* arr, uint32_t lo, uint32_t hi, T target) {
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        T v = arr[mid];
        if (v == target) return true;
        if (v < target) lo = mid + 1;
        else hi = mid;
    }
    return false;
}

// Lower/upper bound generic over T (delegates to existing typed
// helpers). Compile-time selection via T width.
template <typename T>
__device__ __forceinline__ uint32_t lower_bound_t(const T* arr, uint32_t n, T target);

template <>
__device__ __forceinline__ uint32_t lower_bound_t<uint32_t>(
    const uint32_t* arr, uint32_t n, uint32_t target) {
    return lower_bound_u32(arr, n, target);
}

template <>
__device__ __forceinline__ uint32_t lower_bound_t<uint64_t>(
    const uint64_t* arr, uint32_t n, uint64_t target) {
    return lower_bound_u64(arr, n, target);
}

template <typename T>
__device__ __forceinline__ uint32_t upper_bound_t(const T* arr, uint32_t n, T target);

template <>
__device__ __forceinline__ uint32_t upper_bound_t<uint32_t>(
    const uint32_t* arr, uint32_t n, uint32_t target) {
    return upper_bound_u32(arr, n, target);
}

template <>
__device__ __forceinline__ uint32_t upper_bound_t<uint64_t>(
    const uint64_t* arr, uint32_t n, uint64_t target) {
    return upper_bound_u64(arr, n, target);
}

// Level-keyed template recursion. Bind vertex `Level` via L-way
// intersection of L back-edges from already-bound v_0..v_{Level-1}.
// Terminates when Level == K_VAL (count++ or materialize).
//
// `Out` is a callable that consumes the fully-bound binding[].
// For count: incrementing a counter. For materialize: writing to
// output columns.
template <int K_VAL, int Level, typename T, typename Out>
__device__ __forceinline__ void clique_recurse_t(
    const T* const* edge_col0,
    const T* const* edge_col1,
    const uint32_t* edge_n,
    const uint8_t* edge_order,
    const uint8_t* iteration_order,
    T* binding,
    Out& out) {
    if constexpr (Level >= K_VAL) {
        out(binding);
        return;
    } else {
        int current = plan_or_identity_slot(iteration_order, Level);
        int first = plan_or_identity_slot(iteration_order, 0);
        int e0L = static_cast<int>(
            plan_or_identity_slot(edge_order, clique_edge_idx_unordered_t<K_VAL>(first, current)));
        T v0 = binding[0];
        uint32_t lo0 = lower_bound_t<T>(edge_col0[e0L], edge_n[e0L], v0);
        uint32_t hi0 = upper_bound_t<T>(edge_col0[e0L], edge_n[e0L], v0);
        for (uint32_t k = lo0; k < hi0; ++k) {
            T candidate = edge_col1[e0L][k];
            bool all_match = true;
            for (int j = 1; j < Level; ++j) {
                int prior = plan_or_identity_slot(iteration_order, j);
                int eJL = static_cast<int>(
                    plan_or_identity_slot(edge_order, clique_edge_idx_unordered_t<K_VAL>(prior, current)));
                T vj = binding[j];
                uint32_t loJ = lower_bound_t<T>(edge_col0[eJL], edge_n[eJL], vj);
                uint32_t hiJ = upper_bound_t<T>(edge_col0[eJL], edge_n[eJL], vj);
                if (!contains_in_range_t<T>(edge_col1[eJL], loJ, hiJ, candidate)) {
                    all_match = false;
                    break;
                }
            }
            if (all_match) {
                binding[Level] = candidate;
                clique_recurse_t<K_VAL, Level + 1, T, Out>(
                    edge_col0, edge_col1, edge_n, edge_order, iteration_order, binding, out);
            }
        }
    }
}

// Per-thread count: returns the number of K_VAL-cliques extending
// the leader row.
template <int K_VAL, typename T>
__device__ __forceinline__ uint32_t wcoj_clique_template_count_t(
    const T* const* edge_col0,
    const T* const* edge_col1,
    const uint32_t* edge_n,
    uint32_t leader_edge_idx,
    const uint8_t* edge_order,
    const uint8_t* iteration_order,
    uint32_t leader_idx) {
    T binding[K_VAL];
    binding[0] = edge_col0[leader_edge_idx][leader_idx];
    binding[1] = edge_col1[leader_edge_idx][leader_idx];
    uint32_t count = 0;
    auto counter = [&count](T*) { count++; };
    clique_recurse_t<K_VAL, 2, T, decltype(counter)>(
        edge_col0, edge_col1, edge_n, edge_order, iteration_order, binding, counter);
    return count;
}

// Per-thread emit: writes one row per K_VAL-clique extension into
// the output column array starting at `base`. Returns total
// emitted (caller is responsible for budget).
template <int K_VAL, typename T>
__device__ __forceinline__ uint32_t wcoj_clique_template_emit_t(
    const T* const* edge_col0,
    const T* const* edge_col1,
    const uint32_t* edge_n,
    uint32_t leader_edge_idx,
    const uint8_t* edge_order,
    const uint8_t* iteration_order,
    uint32_t leader_idx,
    T* const* out_cols,
    uint32_t base) {
    T binding[K_VAL];
    binding[0] = edge_col0[leader_edge_idx][leader_idx];
    binding[1] = edge_col1[leader_edge_idx][leader_idx];
    uint32_t emitted = 0;
    auto emitter = [&](T* b) {
        uint32_t pos = base + emitted;
        for (int c = 0; c < K_VAL; ++c) {
            out_cols[c][pos] = b[c];
        }
        emitted++;
    };
    clique_recurse_t<K_VAL, 2, T, decltype(emitter)>(
        edge_col0, edge_col1, edge_n, edge_order, iteration_order, binding, emitter);
    return emitted;
}

// Paper §5 Algorithm 1 Phase 1: Histograms maintained alongside data; refreshed during Merge per Authorization 5 (2026-05-17)
template <typename T>
__device__ __forceinline__ uint32_t wcoj_clique_metadata_leader_work_total(
    const T* unique_keys,
    const uint32_t* fan_out,
    const uint32_t* prefix_sum,
    uint32_t total) {
    (void)unique_keys;
    if (total == 0u || fan_out == nullptr || prefix_sum == nullptr) {
        return 0u;
    }
    uint32_t last = total - 1u;
    return prefix_sum[last] + fan_out[last];
}

__device__ __forceinline__ uint32_t wcoj_clique_metadata_leader_idx(
    const uint32_t* fan_out,
    const uint32_t* prefix_sum,
    uint32_t total,
    uint32_t work_idx) {
    if (total == 0u || fan_out == nullptr || prefix_sum == nullptr) {
        return work_idx;
    }
    uint32_t upper = upper_bound_u32(prefix_sum, total, work_idx);
    uint32_t key_idx = (upper == 0u) ? 0u : upper - 1u;
    uint32_t group_start = prefix_sum[key_idx];
    uint32_t group_degree = fan_out[key_idx];
    uint32_t in_group = work_idx - group_start;
    if (in_group >= group_degree) {
        return work_idx;
    }
    return group_start + in_group;
}

// ---------------------------------------------------------------
// Grid-level templates — absorb thread-idx + bound checks so the
// `extern "C"` ABI wrappers below are single-statement
// template-calls, satisfying the locked Tier-1 source-audit
// contract (W3.2 plan §345: "exactly one template-call statement
// with no conditionals").
// ---------------------------------------------------------------

template <int K_VAL, typename T>
__device__ __forceinline__ void wcoj_clique_template_count_hg_grid_t(
    const T* const* edge_col0,
    const T* const* edge_col1,
    const uint32_t* edge_n,
    uint32_t leader_edge_idx,
    const uint8_t* edge_order,
    const uint8_t* iteration_order,
    uint32_t leader_count,
    const T* unique_keys,
    const uint32_t* fan_out,
    const uint32_t* prefix_sum,
    uint32_t total,
    uint32_t block_work_unit,
    uint32_t* out_block_counts,
    uint32_t* out_thread_counts) {
    uint32_t leader_work_total = wcoj_clique_metadata_leader_work_total<T>(
        unique_keys, fan_out, prefix_sum, total);
    if (leader_work_total == 0u || leader_work_total > leader_count) {
        leader_work_total = leader_count;
    }
    uint32_t block_start = blockIdx.x * block_work_unit;
    if (block_start >= leader_work_total) {
        out_thread_counts[blockIdx.x * blockDim.x + threadIdx.x] = 0;
        if (threadIdx.x == 0) {
            out_block_counts[blockIdx.x] = 0;
        }
        return;
    }
    uint32_t block_end = block_start + block_work_unit;
    if (block_end < block_start || block_end > leader_work_total) {
        block_end = leader_work_total;
    }

    uint32_t local_count = 0;
    for (uint32_t work_idx = block_start + threadIdx.x;
         work_idx < block_end;
         work_idx += blockDim.x) {
        uint32_t i = wcoj_clique_metadata_leader_idx(fan_out, prefix_sum, total, work_idx);
        local_count += wcoj_clique_template_count_t<K_VAL, T>(
            edge_col0, edge_col1, edge_n, leader_edge_idx, edge_order, iteration_order, i);
    }
    out_thread_counts[blockIdx.x * blockDim.x + threadIdx.x] = local_count;

    __shared__ uint32_t partial[256];
    partial[threadIdx.x] = local_count;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out_block_counts[blockIdx.x] = partial[0];
    }
}

template <int K_VAL, typename T>
__device__ __forceinline__ void wcoj_clique_template_materialize_hg_grid_t(
    const T* const* edge_col0,
    const T* const* edge_col1,
    const uint32_t* edge_n,
    uint32_t leader_edge_idx,
    const uint8_t* edge_order,
    const uint8_t* iteration_order,
    uint32_t leader_count,
    const T* unique_keys,
    const uint32_t* fan_out,
    const uint32_t* prefix_sum,
    uint32_t total,
    uint32_t block_work_unit,
    const uint32_t* __restrict__ thread_counts,
    const uint32_t* __restrict__ block_offsets,
    uint32_t total_rows,
    T* const* out_cols) {
    uint32_t leader_work_total = wcoj_clique_metadata_leader_work_total<T>(
        unique_keys, fan_out, prefix_sum, total);
    if (leader_work_total == 0u || leader_work_total > leader_count) {
        leader_work_total = leader_count;
    }
    uint32_t block_start = blockIdx.x * block_work_unit;
    if (block_start >= leader_work_total) {
        return;
    }
    uint32_t block_end = block_start + block_work_unit;
    if (block_end < block_start || block_end > leader_work_total) {
        block_end = leader_work_total;
    }

    uint32_t local_count = thread_counts[blockIdx.x * blockDim.x + threadIdx.x];

    __shared__ uint32_t thread_prefix[256];
    thread_prefix[threadIdx.x] = local_count;
    __syncthreads();
    for (uint32_t stride = 1; stride < blockDim.x; stride <<= 1) {
        uint32_t add = 0;
        if (threadIdx.x >= stride) {
            add = thread_prefix[threadIdx.x - stride];
        }
        __syncthreads();
        thread_prefix[threadIdx.x] += add;
        __syncthreads();
    }

    uint32_t thread_base = block_offsets[blockIdx.x];
    if (threadIdx.x > 0) {
        thread_base += thread_prefix[threadIdx.x - 1];
    }

    uint32_t local_emit = 0;
    for (uint32_t work_idx = block_start + threadIdx.x;
         work_idx < block_end;
         work_idx += blockDim.x) {
        uint32_t i = wcoj_clique_metadata_leader_idx(fan_out, prefix_sum, total, work_idx);
        uint32_t emitted = wcoj_clique_template_emit_t<K_VAL, T>(
            edge_col0, edge_col1, edge_n, leader_edge_idx, edge_order, iteration_order, i, out_cols, thread_base + local_emit);
        local_emit += emitted;
    }
    (void)total_rows;
}

// ---------------------------------------------------------------
// ABI wrappers — template calls only. Each wrapper delegates to the
// grid-level template without local control flow.
// ---------------------------------------------------------------

extern "C" __global__ void wcoj_clique5_count_hg_u32(
    const uint32_t* const* edge_col0,
    const uint32_t* const* edge_col1,
    const uint32_t* edge_n,
    uint32_t leader_edge_idx,
    const uint8_t* edge_order,
    const uint8_t* iteration_order,
    uint32_t leader_count,
    const uint32_t* unique_keys,
    const uint32_t* fan_out,
    const uint32_t* prefix_sum,
    uint32_t total,
    uint32_t block_work_unit,
    uint32_t* out_block_counts,
    uint32_t* out_thread_counts) {
    wcoj_clique_template_count_hg_grid_t<5, uint32_t>(edge_col0, edge_col1, edge_n, leader_edge_idx, edge_order, iteration_order, leader_count, unique_keys, fan_out, prefix_sum, total, block_work_unit, out_block_counts, out_thread_counts);
}

extern "C" __global__ void wcoj_clique5_materialize_hg_u32(
    const uint32_t* const* edge_col0,
    const uint32_t* const* edge_col1,
    const uint32_t* edge_n,
    uint32_t leader_edge_idx,
    const uint8_t* edge_order,
    const uint8_t* iteration_order,
    uint32_t leader_count,
    const uint32_t* unique_keys,
    const uint32_t* fan_out,
    const uint32_t* prefix_sum,
    uint32_t total,
    uint32_t block_work_unit,
    const uint32_t* __restrict__ thread_counts,
    const uint32_t* __restrict__ block_offsets,
    uint32_t total_rows,
    uint32_t* const* out_cols) {
    wcoj_clique_template_materialize_hg_grid_t<5, uint32_t>(edge_col0, edge_col1, edge_n, leader_edge_idx, edge_order, iteration_order, leader_count, unique_keys, fan_out, prefix_sum, total, block_work_unit, thread_counts, block_offsets, total_rows, out_cols);
}

extern "C" __global__ void wcoj_clique5_count_hg_u64(
    const uint64_t* const* edge_col0,
    const uint64_t* const* edge_col1,
    const uint32_t* edge_n,
    uint32_t leader_edge_idx,
    const uint8_t* edge_order,
    const uint8_t* iteration_order,
    uint32_t leader_count,
    const uint64_t* unique_keys,
    const uint32_t* fan_out,
    const uint32_t* prefix_sum,
    uint32_t total,
    uint32_t block_work_unit,
    uint32_t* out_block_counts,
    uint32_t* out_thread_counts) {
    wcoj_clique_template_count_hg_grid_t<5, uint64_t>(edge_col0, edge_col1, edge_n, leader_edge_idx, edge_order, iteration_order, leader_count, unique_keys, fan_out, prefix_sum, total, block_work_unit, out_block_counts, out_thread_counts);
}

extern "C" __global__ void wcoj_clique5_materialize_hg_u64(
    const uint64_t* const* edge_col0,
    const uint64_t* const* edge_col1,
    const uint32_t* edge_n,
    uint32_t leader_edge_idx,
    const uint8_t* edge_order,
    const uint8_t* iteration_order,
    uint32_t leader_count,
    const uint64_t* unique_keys,
    const uint32_t* fan_out,
    const uint32_t* prefix_sum,
    uint32_t total,
    uint32_t block_work_unit,
    const uint32_t* __restrict__ thread_counts,
    const uint32_t* __restrict__ block_offsets,
    uint32_t total_rows,
    uint64_t* const* out_cols) {
    wcoj_clique_template_materialize_hg_grid_t<5, uint64_t>(edge_col0, edge_col1, edge_n, leader_edge_idx, edge_order, iteration_order, leader_count, unique_keys, fan_out, prefix_sum, total, block_work_unit, thread_counts, block_offsets, total_rows, out_cols);
}

extern "C" __global__ void wcoj_clique6_count_hg_u32(
    const uint32_t* const* edge_col0,
    const uint32_t* const* edge_col1,
    const uint32_t* edge_n,
    uint32_t leader_edge_idx,
    const uint8_t* edge_order,
    const uint8_t* iteration_order,
    uint32_t leader_count,
    const uint32_t* unique_keys,
    const uint32_t* fan_out,
    const uint32_t* prefix_sum,
    uint32_t total,
    uint32_t block_work_unit,
    uint32_t* out_block_counts,
    uint32_t* out_thread_counts) {
    wcoj_clique_template_count_hg_grid_t<6, uint32_t>(edge_col0, edge_col1, edge_n, leader_edge_idx, edge_order, iteration_order, leader_count, unique_keys, fan_out, prefix_sum, total, block_work_unit, out_block_counts, out_thread_counts);
}

extern "C" __global__ void wcoj_clique6_materialize_hg_u32(
    const uint32_t* const* edge_col0,
    const uint32_t* const* edge_col1,
    const uint32_t* edge_n,
    uint32_t leader_edge_idx,
    const uint8_t* edge_order,
    const uint8_t* iteration_order,
    uint32_t leader_count,
    const uint32_t* unique_keys,
    const uint32_t* fan_out,
    const uint32_t* prefix_sum,
    uint32_t total,
    uint32_t block_work_unit,
    const uint32_t* __restrict__ thread_counts,
    const uint32_t* __restrict__ block_offsets,
    uint32_t total_rows,
    uint32_t* const* out_cols) {
    wcoj_clique_template_materialize_hg_grid_t<6, uint32_t>(edge_col0, edge_col1, edge_n, leader_edge_idx, edge_order, iteration_order, leader_count, unique_keys, fan_out, prefix_sum, total, block_work_unit, thread_counts, block_offsets, total_rows, out_cols);
}

extern "C" __global__ void wcoj_clique6_count_hg_u64(
    const uint64_t* const* edge_col0,
    const uint64_t* const* edge_col1,
    const uint32_t* edge_n,
    uint32_t leader_edge_idx,
    const uint8_t* edge_order,
    const uint8_t* iteration_order,
    uint32_t leader_count,
    const uint64_t* unique_keys,
    const uint32_t* fan_out,
    const uint32_t* prefix_sum,
    uint32_t total,
    uint32_t block_work_unit,
    uint32_t* out_block_counts,
    uint32_t* out_thread_counts) {
    wcoj_clique_template_count_hg_grid_t<6, uint64_t>(edge_col0, edge_col1, edge_n, leader_edge_idx, edge_order, iteration_order, leader_count, unique_keys, fan_out, prefix_sum, total, block_work_unit, out_block_counts, out_thread_counts);
}

extern "C" __global__ void wcoj_clique6_materialize_hg_u64(
    const uint64_t* const* edge_col0,
    const uint64_t* const* edge_col1,
    const uint32_t* edge_n,
    uint32_t leader_edge_idx,
    const uint8_t* edge_order,
    const uint8_t* iteration_order,
    uint32_t leader_count,
    const uint64_t* unique_keys,
    const uint32_t* fan_out,
    const uint32_t* prefix_sum,
    uint32_t total,
    uint32_t block_work_unit,
    const uint32_t* __restrict__ thread_counts,
    const uint32_t* __restrict__ block_offsets,
    uint32_t total_rows,
    uint64_t* const* out_cols) {
    wcoj_clique_template_materialize_hg_grid_t<6, uint64_t>(edge_col0, edge_col1, edge_n, leader_edge_idx, edge_order, iteration_order, leader_count, unique_keys, fan_out, prefix_sum, total, block_work_unit, thread_counts, block_offsets, total_rows, out_cols);
}

extern "C" __global__ void wcoj_clique7_count_hg_u32(
    const uint32_t* const* edge_col0,
    const uint32_t* const* edge_col1,
    const uint32_t* edge_n,
    uint32_t leader_edge_idx,
    const uint8_t* edge_order,
    const uint8_t* iteration_order,
    uint32_t leader_count,
    const uint32_t* unique_keys,
    const uint32_t* fan_out,
    const uint32_t* prefix_sum,
    uint32_t total,
    uint32_t block_work_unit,
    uint32_t* out_block_counts,
    uint32_t* out_thread_counts) {
    wcoj_clique_template_count_hg_grid_t<7, uint32_t>(edge_col0, edge_col1, edge_n, leader_edge_idx, edge_order, iteration_order, leader_count, unique_keys, fan_out, prefix_sum, total, block_work_unit, out_block_counts, out_thread_counts);
}

extern "C" __global__ void wcoj_clique7_materialize_hg_u32(
    const uint32_t* const* edge_col0,
    const uint32_t* const* edge_col1,
    const uint32_t* edge_n,
    uint32_t leader_edge_idx,
    const uint8_t* edge_order,
    const uint8_t* iteration_order,
    uint32_t leader_count,
    const uint32_t* unique_keys,
    const uint32_t* fan_out,
    const uint32_t* prefix_sum,
    uint32_t total,
    uint32_t block_work_unit,
    const uint32_t* __restrict__ thread_counts,
    const uint32_t* __restrict__ block_offsets,
    uint32_t total_rows,
    uint32_t* const* out_cols) {
    wcoj_clique_template_materialize_hg_grid_t<7, uint32_t>(edge_col0, edge_col1, edge_n, leader_edge_idx, edge_order, iteration_order, leader_count, unique_keys, fan_out, prefix_sum, total, block_work_unit, thread_counts, block_offsets, total_rows, out_cols);
}

extern "C" __global__ void wcoj_clique7_count_hg_u64(
    const uint64_t* const* edge_col0,
    const uint64_t* const* edge_col1,
    const uint32_t* edge_n,
    uint32_t leader_edge_idx,
    const uint8_t* edge_order,
    const uint8_t* iteration_order,
    uint32_t leader_count,
    const uint64_t* unique_keys,
    const uint32_t* fan_out,
    const uint32_t* prefix_sum,
    uint32_t total,
    uint32_t block_work_unit,
    uint32_t* out_block_counts,
    uint32_t* out_thread_counts) {
    wcoj_clique_template_count_hg_grid_t<7, uint64_t>(edge_col0, edge_col1, edge_n, leader_edge_idx, edge_order, iteration_order, leader_count, unique_keys, fan_out, prefix_sum, total, block_work_unit, out_block_counts, out_thread_counts);
}

extern "C" __global__ void wcoj_clique7_materialize_hg_u64(
    const uint64_t* const* edge_col0,
    const uint64_t* const* edge_col1,
    const uint32_t* edge_n,
    uint32_t leader_edge_idx,
    const uint8_t* edge_order,
    const uint8_t* iteration_order,
    uint32_t leader_count,
    const uint64_t* unique_keys,
    const uint32_t* fan_out,
    const uint32_t* prefix_sum,
    uint32_t total,
    uint32_t block_work_unit,
    const uint32_t* __restrict__ thread_counts,
    const uint32_t* __restrict__ block_offsets,
    uint32_t total_rows,
    uint64_t* const* out_cols) {
    wcoj_clique_template_materialize_hg_grid_t<7, uint64_t>(edge_col0, edge_col1, edge_n, leader_edge_idx, edge_order, iteration_order, leader_count, unique_keys, fan_out, prefix_sum, total, block_work_unit, thread_counts, block_offsets, total_rows, out_cols);
}

extern "C" __global__ void wcoj_clique8_count_hg_u32(
    const uint32_t* const* edge_col0,
    const uint32_t* const* edge_col1,
    const uint32_t* edge_n,
    uint32_t leader_edge_idx,
    const uint8_t* edge_order,
    const uint8_t* iteration_order,
    uint32_t leader_count,
    const uint32_t* unique_keys,
    const uint32_t* fan_out,
    const uint32_t* prefix_sum,
    uint32_t total,
    uint32_t block_work_unit,
    uint32_t* out_block_counts,
    uint32_t* out_thread_counts) {
    wcoj_clique_template_count_hg_grid_t<8, uint32_t>(edge_col0, edge_col1, edge_n, leader_edge_idx, edge_order, iteration_order, leader_count, unique_keys, fan_out, prefix_sum, total, block_work_unit, out_block_counts, out_thread_counts);
}

extern "C" __global__ void wcoj_clique8_materialize_hg_u32(
    const uint32_t* const* edge_col0,
    const uint32_t* const* edge_col1,
    const uint32_t* edge_n,
    uint32_t leader_edge_idx,
    const uint8_t* edge_order,
    const uint8_t* iteration_order,
    uint32_t leader_count,
    const uint32_t* unique_keys,
    const uint32_t* fan_out,
    const uint32_t* prefix_sum,
    uint32_t total,
    uint32_t block_work_unit,
    const uint32_t* __restrict__ thread_counts,
    const uint32_t* __restrict__ block_offsets,
    uint32_t total_rows,
    uint32_t* const* out_cols) {
    wcoj_clique_template_materialize_hg_grid_t<8, uint32_t>(edge_col0, edge_col1, edge_n, leader_edge_idx, edge_order, iteration_order, leader_count, unique_keys, fan_out, prefix_sum, total, block_work_unit, thread_counts, block_offsets, total_rows, out_cols);
}

extern "C" __global__ void wcoj_clique8_count_hg_u64(
    const uint64_t* const* edge_col0,
    const uint64_t* const* edge_col1,
    const uint32_t* edge_n,
    uint32_t leader_edge_idx,
    const uint8_t* edge_order,
    const uint8_t* iteration_order,
    uint32_t leader_count,
    const uint64_t* unique_keys,
    const uint32_t* fan_out,
    const uint32_t* prefix_sum,
    uint32_t total,
    uint32_t block_work_unit,
    uint32_t* out_block_counts,
    uint32_t* out_thread_counts) {
    wcoj_clique_template_count_hg_grid_t<8, uint64_t>(edge_col0, edge_col1, edge_n, leader_edge_idx, edge_order, iteration_order, leader_count, unique_keys, fan_out, prefix_sum, total, block_work_unit, out_block_counts, out_thread_counts);
}

extern "C" __global__ void wcoj_clique8_materialize_hg_u64(
    const uint64_t* const* edge_col0,
    const uint64_t* const* edge_col1,
    const uint32_t* edge_n,
    uint32_t leader_edge_idx,
    const uint8_t* edge_order,
    const uint8_t* iteration_order,
    uint32_t leader_count,
    const uint64_t* unique_keys,
    const uint32_t* fan_out,
    const uint32_t* prefix_sum,
    uint32_t total,
    uint32_t block_work_unit,
    const uint32_t* __restrict__ thread_counts,
    const uint32_t* __restrict__ block_offsets,
    uint32_t total_rows,
    uint64_t* const* out_cols) {
    wcoj_clique_template_materialize_hg_grid_t<8, uint64_t>(edge_col0, edge_col1, edge_n, leader_edge_idx, edge_order, iteration_order, leader_count, unique_keys, fan_out, prefix_sum, total, block_work_unit, thread_counts, block_offsets, total_rows, out_cols);
}

// ---------------------------------------------------------------
// S1e — aggregate-fused variant of wcoj_clique{K}_count_hg_*:
// instead of reducing matches into block/thread totals, each
// leader-edge row's K-clique completion count is accumulated into
// `out_row_counts[row]` (length leader_count, zero-initialized by
// the host). Leader-edge row identity is the same `i` the count
// template resolves via `wcoj_clique_metadata_leader_idx`; the
// row's group key is the leader edge's col0 (= binding[0], the
// plan's position-0 root variable). The leader edge is lex-sorted
// by (root, col1), so per-root group counts are a segmented sum
// over contiguous row ranges downstream. Integer atomicAdd is
// order-insensitive, so the final counter values are deterministic
// (precedent: wcoj_triangle_groupby_root_count_hg_u32). The clique
// rows are never materialized.
// ---------------------------------------------------------------

template <int K_VAL, typename T>
__device__ __forceinline__ void wcoj_clique_template_groupby_root_count_hg_grid_t(
    const T* const* edge_col0,
    const T* const* edge_col1,
    const uint32_t* edge_n,
    uint32_t leader_edge_idx,
    const uint8_t* edge_order,
    const uint8_t* iteration_order,
    uint32_t leader_count,
    const T* unique_keys,
    const uint32_t* fan_out,
    const uint32_t* prefix_sum,
    uint32_t total,
    uint32_t block_work_unit,
    uint32_t* out_row_counts) {
    uint32_t leader_work_total = wcoj_clique_metadata_leader_work_total<T>(
        unique_keys, fan_out, prefix_sum, total);
    if (leader_work_total == 0u || leader_work_total > leader_count) {
        leader_work_total = leader_count;
    }
    uint32_t block_start = blockIdx.x * block_work_unit;
    if (block_start >= leader_work_total) {
        return;
    }
    uint32_t block_end = block_start + block_work_unit;
    if (block_end < block_start || block_end > leader_work_total) {
        block_end = leader_work_total;
    }

    for (uint32_t work_idx = block_start + threadIdx.x;
         work_idx < block_end;
         work_idx += blockDim.x) {
        uint32_t i = wcoj_clique_metadata_leader_idx(fan_out, prefix_sum, total, work_idx);
        uint32_t c = wcoj_clique_template_count_t<K_VAL, T>(
            edge_col0, edge_col1, edge_n, leader_edge_idx, edge_order, iteration_order, i);
        if (c != 0u) {
            atomicAdd(&out_row_counts[i], c);
        }
    }
}

extern "C" __global__ void wcoj_clique5_groupby_root_count_hg_u32(
    const uint32_t* const* edge_col0,
    const uint32_t* const* edge_col1,
    const uint32_t* edge_n,
    uint32_t leader_edge_idx,
    const uint8_t* edge_order,
    const uint8_t* iteration_order,
    uint32_t leader_count,
    const uint32_t* unique_keys,
    const uint32_t* fan_out,
    const uint32_t* prefix_sum,
    uint32_t total,
    uint32_t block_work_unit,
    uint32_t* out_row_counts) {
    wcoj_clique_template_groupby_root_count_hg_grid_t<5, uint32_t>(edge_col0, edge_col1, edge_n, leader_edge_idx, edge_order, iteration_order, leader_count, unique_keys, fan_out, prefix_sum, total, block_work_unit, out_row_counts);
}

extern "C" __global__ void wcoj_clique6_groupby_root_count_hg_u32(
    const uint32_t* const* edge_col0,
    const uint32_t* const* edge_col1,
    const uint32_t* edge_n,
    uint32_t leader_edge_idx,
    const uint8_t* edge_order,
    const uint8_t* iteration_order,
    uint32_t leader_count,
    const uint32_t* unique_keys,
    const uint32_t* fan_out,
    const uint32_t* prefix_sum,
    uint32_t total,
    uint32_t block_work_unit,
    uint32_t* out_row_counts) {
    wcoj_clique_template_groupby_root_count_hg_grid_t<6, uint32_t>(edge_col0, edge_col1, edge_n, leader_edge_idx, edge_order, iteration_order, leader_count, unique_keys, fan_out, prefix_sum, total, block_work_unit, out_row_counts);
}

// =====================================================================
// D2 Free Join (S2 spike) — level-synchronous frontier engine
// primitives (docs/plans/2026-06-12-d2-free-join-design.md §2).
//
// A frontier is a SoA set of u32 columns; each row is one partial
// binding plus, per not-yet-exhausted atom, a (lo, hi) trie range
// into that atom's lex-sorted dedup'd buffer (flat sorted-range
// tries: a trie node IS a contiguous row range; get(key) is a
// binary-search refinement). Column sets are passed as device
// pointer tables so the kernels stay arity-generic.
//
// Determinism contract (§2.3): two-phase count → scan → emit, no
// atomics anywhere; output order is parent-row order × lex order of
// the cover range, so results are canonical without re-sorting.
//
// Work-index mapping: when the cover atom has per-row parent ranges
// (has_parent_range == 1) work item w maps to frontier row r via the
// exclusive work prefix (same upper_bound discipline as the triangle
// HG kernels). When the cover atom is untouched every row shares the
// constant range [const_lo, const_hi), so the mapping is a uniform
// divide/modulo and no work prefix is materialized at all.

// Per-frontier-row cover range length, written into work_prefix[i];
// work_prefix[n_frontier] zeroed for the exclusive scan that follows
// (mirrors wcoj_triangle_build_hg_work_plan_u32).
extern "C" __global__ void fj_expand_work_prefix_u32(
    const uint32_t* __restrict__ parent_lo,
    const uint32_t* __restrict__ parent_hi,
    uint32_t n_frontier,
    uint32_t* __restrict__ work_prefix) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        work_prefix[n_frontier] = 0;
    }
    if (i >= n_frontier) {
        return;
    }
    work_prefix[i] = parent_hi[i] - parent_lo[i];
}

// Phase 1 of EXPAND: mark distinct cover-prefix group starts. A
// candidate position starts a group iff it is the first position of
// its frontier row's range or any of the cover columns changes value
// vs the previous position (ranges are lex-sorted, so groups are
// contiguous runs). group_marks has length total_work + 1; the
// trailing slot is zeroed for the exclusive scan.
//
// identity_groups == 1: the cover consumes through the atom's last
// column, so every position is its own group and the start predicate
// is skipped (the pass then only exists to evaluate fused probes).
//
// Fused probe filters: range-exhausting probe subatoms whose key
// variables are all bound by THIS node's cover are folded into the
// mark (group survives only if every fused probe's binary-search
// refinement is non-empty). Keys are constant within a group (cover
// values), so per-position evaluation is exact. This removes the
// separate probe kernel + mask compaction for those subatoms — the
// emit pass then materializes exactly the surviving children.
// probe_desc packs one descriptor per fused probe, sequentially:
//   [n_cols, has_range, in_lo_ptr, in_hi_ptr, n_atom_rows,
//    data_col_ptr * n_cols, cover_var_idx * n_cols]
extern "C" __global__ void fj_expand_count_u32(
    const uint32_t* const* __restrict__ cover_cols,
    uint32_t n_cover_cols,
    const uint32_t* __restrict__ parent_lo,
    const uint32_t* __restrict__ work_prefix,
    uint32_t has_parent_range,
    uint32_t const_lo,
    uint32_t const_hi,
    uint32_t n_frontier,
    uint32_t total_work,
    uint32_t identity_groups,
    const unsigned long long* __restrict__ probe_desc,
    uint32_t n_fused_probes,
    uint32_t* __restrict__ group_marks) {
    uint32_t w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w == 0) {
        group_marks[total_work] = 0;
    }
    if (w >= total_work) {
        return;
    }
    uint32_t r = 0;
    uint32_t lo;
    uint32_t pos;
    if (has_parent_range != 0u) {
        r = upper_bound_u32(work_prefix, n_frontier + 1, w) - 1u;
        lo = parent_lo[r];
        pos = lo + (w - work_prefix[r]);
    } else {
        uint32_t len = const_hi - const_lo;
        r = w / len;
        lo = const_lo;
        pos = const_lo + (w % len);
    }
    if (identity_groups == 0u) {
        uint32_t start = (pos == lo) ? 1u : 0u;
        if (start == 0u) {
            for (uint32_t j = 0; j < n_cover_cols; ++j) {
                const uint32_t* col = cover_cols[j];
                if (col[pos] != col[pos - 1u]) {
                    start = 1u;
                    break;
                }
            }
        }
        if (start == 0u) {
            group_marks[w] = 0;
            return;
        }
    }
    uint32_t ok = 1;
    const unsigned long long* d = probe_desc;
    for (uint32_t q = 0; q < n_fused_probes; ++q) {
        uint32_t p_cols = (uint32_t)d[0];
        uint32_t p_has_range = (uint32_t)d[1];
        uint32_t p_lo;
        uint32_t p_hi;
        if (p_has_range != 0u) {
            p_lo = ((const uint32_t*)d[2])[r];
            p_hi = ((const uint32_t*)d[3])[r];
        } else {
            p_lo = 0;
            p_hi = (uint32_t)d[4];
        }
        for (uint32_t k = 0; k < p_cols && p_lo < p_hi; ++k) {
            const uint32_t* col = (const uint32_t*)d[5 + k];
            uint32_t cover_idx = (uint32_t)d[5 + p_cols + k];
            uint32_t v = cover_cols[cover_idx][pos];
            uint32_t new_lo = p_lo + lower_bound_u32(col + p_lo, p_hi - p_lo, v);
            uint32_t new_hi = p_lo + upper_bound_u32(col + p_lo, p_hi - p_lo, v);
            p_lo = new_lo;
            p_hi = new_hi;
        }
        if (p_lo >= p_hi) {
            ok = 0;
            break;
        }
        d += 5 + 2 * (size_t)p_cols;
    }
    group_marks[w] = ok;
}

// Phase 2 of EXPAND: emit one child frontier row per distinct cover
// group. Output index = group_offsets[w] (exclusive scan of the
// marks), so emission is atomics-free and deterministic. The group's
// refined subrange [pos, g_hi) is recovered by successive
// upper_bound refinement on the cover columns (each column is sorted
// within the prefix-equal range of the previous columns).
//
// group_offsets == null: the host skipped the count pass entirely
// (identity-group cover with no fused probes — the cover consumes
// through its atom's LAST column, so rows in a trie range share all
// columns before the range's depth and, inputs being full-row
// deduped, the remaining column suffix is distinct within every
// range: every candidate position is its own group) and out == w.
//
// group_offsets != null: survival is read from the scanned marks
// (group_offsets[w + 1] > group_offsets[w]), NOT recomputed from the
// group-start predicate — fused probe filters in the count pass can
// zero a mark that the predicate alone would set.
// Copied parent columns are split into two groups so the same launch
// shape serves every width class: VAR columns hold bound data values
// (uint32_t here, uint64_t in the _u64 twin) while RANGE columns hold
// u32 row indices in every width class.
extern "C" __global__ void fj_expand_emit_u32(
    const uint32_t* const* __restrict__ cover_cols,
    uint32_t n_cover_cols,
    const uint32_t* __restrict__ parent_lo,
    const uint32_t* __restrict__ parent_hi,
    const uint32_t* __restrict__ work_prefix,
    uint32_t has_parent_range,
    uint32_t const_lo,
    uint32_t const_hi,
    uint32_t n_frontier,
    uint32_t total_work,
    const uint32_t* __restrict__ group_offsets,
    const uint32_t* const* __restrict__ parent_copy_var_cols,
    uint32_t* const* __restrict__ child_copy_var_cols,
    uint32_t n_copy_var_cols,
    const uint32_t* const* __restrict__ parent_copy_range_cols,
    uint32_t* const* __restrict__ child_copy_range_cols,
    uint32_t n_copy_range_cols,
    uint32_t* const* __restrict__ child_var_cols,
    uint32_t keep_cover_range,
    uint32_t* __restrict__ child_cover_lo,
    uint32_t* __restrict__ child_cover_hi) {
    uint32_t w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= total_work) {
        return;
    }
    uint32_t r;
    uint32_t lo;
    uint32_t pos;
    uint32_t row_hi;
    if (has_parent_range != 0u) {
        r = upper_bound_u32(work_prefix, n_frontier + 1, w) - 1u;
        lo = parent_lo[r];
        pos = lo + (w - work_prefix[r]);
        row_hi = parent_hi[r];
    } else {
        uint32_t len = const_hi - const_lo;
        r = w / len;
        lo = const_lo;
        pos = const_lo + (w % len);
        row_hi = const_hi;
    }
    uint32_t out;
    if (group_offsets == nullptr) {
        out = w;
    } else {
        out = group_offsets[w];
        if (group_offsets[w + 1u] == out) {
            return; // not a surviving group start
        }
    }
    for (uint32_t j = 0; j < n_copy_var_cols; ++j) {
        child_copy_var_cols[j][out] = parent_copy_var_cols[j][r];
    }
    for (uint32_t j = 0; j < n_copy_range_cols; ++j) {
        child_copy_range_cols[j][out] = parent_copy_range_cols[j][r];
    }
    for (uint32_t j = 0; j < n_cover_cols; ++j) {
        child_var_cols[j][out] = cover_cols[j][pos];
    }
    if (keep_cover_range != 0u) {
        uint32_t g_hi = row_hi;
        for (uint32_t j = 0; j < n_cover_cols; ++j) {
            const uint32_t* col = cover_cols[j];
            g_hi = pos + upper_bound_u32(col + pos, g_hi - pos, col[pos]);
        }
        child_cover_lo[out] = pos;
        child_cover_hi[out] = g_hi;
    }
}

// PROBE: per frontier row, refine the probe atom's current trie
// range by binary search on each probe key column (all keys already
// bound — they are frontier columns, so reads coalesce). Writes the
// refined range (when the atom still has unconsumed columns) and the
// survival mask consumed by the existing mask+scan+gather
// compaction. combine_mask == 1 ANDs into an earlier probe's mask.
extern "C" __global__ void fj_probe_refine_u32(
    const uint32_t* const* __restrict__ probe_cols,
    uint32_t n_probe_cols,
    const uint32_t* const* __restrict__ key_cols,
    const uint32_t* __restrict__ in_lo,
    const uint32_t* __restrict__ in_hi,
    uint32_t has_range,
    uint32_t const_lo,
    uint32_t const_hi,
    uint32_t n_frontier,
    uint32_t keep_range,
    uint32_t* __restrict__ out_lo,
    uint32_t* __restrict__ out_hi,
    uint8_t* __restrict__ mask,
    uint32_t combine_mask) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_frontier) {
        return;
    }
    uint32_t lo = (has_range != 0u) ? in_lo[i] : const_lo;
    uint32_t hi = (has_range != 0u) ? in_hi[i] : const_hi;
    for (uint32_t j = 0; j < n_probe_cols && lo < hi; ++j) {
        const uint32_t* col = probe_cols[j];
        uint32_t v = key_cols[j][i];
        uint32_t new_lo = lo + lower_bound_u32(col + lo, hi - lo, v);
        uint32_t new_hi = lo + upper_bound_u32(col + lo, hi - lo, v);
        lo = new_lo;
        hi = new_hi;
    }
    uint8_t hit = (lo < hi) ? 1u : 0u;
    if (combine_mask != 0u) {
        hit = (mask[i] != 0u && hit != 0u) ? 1u : 0u;
    }
    mask[i] = hit;
    if (keep_range != 0u) {
        out_lo[i] = lo;
        out_hi[i] = hi;
    }
}

// ---------------------------------------------------------------
// D2 Free Join — u64 width-class twins. Row indices, ranges, work
// prefixes, and group marks stay u32 (the frontier budget bounds
// total_work below 2^32); only DATA columns (cover, probe, var) and
// their value comparisons widen to uint64_t. The fused-probe
// descriptor layout is unchanged — data_col_ptr entries simply point
// at uint64_t columns.
// ---------------------------------------------------------------

extern "C" __global__ void fj_expand_count_u64(
    const uint64_t* const* __restrict__ cover_cols,
    uint32_t n_cover_cols,
    const uint32_t* __restrict__ parent_lo,
    const uint32_t* __restrict__ work_prefix,
    uint32_t has_parent_range,
    uint32_t const_lo,
    uint32_t const_hi,
    uint32_t n_frontier,
    uint32_t total_work,
    uint32_t identity_groups,
    const unsigned long long* __restrict__ probe_desc,
    uint32_t n_fused_probes,
    uint32_t* __restrict__ group_marks) {
    uint32_t w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w == 0) {
        group_marks[total_work] = 0;
    }
    if (w >= total_work) {
        return;
    }
    uint32_t r = 0;
    uint32_t lo;
    uint32_t pos;
    if (has_parent_range != 0u) {
        r = upper_bound_u32(work_prefix, n_frontier + 1, w) - 1u;
        lo = parent_lo[r];
        pos = lo + (w - work_prefix[r]);
    } else {
        uint32_t len = const_hi - const_lo;
        r = w / len;
        lo = const_lo;
        pos = const_lo + (w % len);
    }
    if (identity_groups == 0u) {
        uint32_t start = (pos == lo) ? 1u : 0u;
        if (start == 0u) {
            for (uint32_t j = 0; j < n_cover_cols; ++j) {
                const uint64_t* col = cover_cols[j];
                if (col[pos] != col[pos - 1u]) {
                    start = 1u;
                    break;
                }
            }
        }
        if (start == 0u) {
            group_marks[w] = 0;
            return;
        }
    }
    uint32_t ok = 1;
    const unsigned long long* d = probe_desc;
    for (uint32_t q = 0; q < n_fused_probes; ++q) {
        uint32_t p_cols = (uint32_t)d[0];
        uint32_t p_has_range = (uint32_t)d[1];
        uint32_t p_lo;
        uint32_t p_hi;
        if (p_has_range != 0u) {
            p_lo = ((const uint32_t*)d[2])[r];
            p_hi = ((const uint32_t*)d[3])[r];
        } else {
            p_lo = 0;
            p_hi = (uint32_t)d[4];
        }
        for (uint32_t k = 0; k < p_cols && p_lo < p_hi; ++k) {
            const uint64_t* col = (const uint64_t*)d[5 + k];
            uint32_t cover_idx = (uint32_t)d[5 + p_cols + k];
            uint64_t v = cover_cols[cover_idx][pos];
            uint32_t new_lo = p_lo + lower_bound_u64(col + p_lo, p_hi - p_lo, v);
            uint32_t new_hi = p_lo + upper_bound_u64(col + p_lo, p_hi - p_lo, v);
            p_lo = new_lo;
            p_hi = new_hi;
        }
        if (p_lo >= p_hi) {
            ok = 0;
            break;
        }
        d += 5 + 2 * (size_t)p_cols;
    }
    group_marks[w] = ok;
}

extern "C" __global__ void fj_expand_emit_u64(
    const uint64_t* const* __restrict__ cover_cols,
    uint32_t n_cover_cols,
    const uint32_t* __restrict__ parent_lo,
    const uint32_t* __restrict__ parent_hi,
    const uint32_t* __restrict__ work_prefix,
    uint32_t has_parent_range,
    uint32_t const_lo,
    uint32_t const_hi,
    uint32_t n_frontier,
    uint32_t total_work,
    const uint32_t* __restrict__ group_offsets,
    const uint64_t* const* __restrict__ parent_copy_var_cols,
    uint64_t* const* __restrict__ child_copy_var_cols,
    uint32_t n_copy_var_cols,
    const uint32_t* const* __restrict__ parent_copy_range_cols,
    uint32_t* const* __restrict__ child_copy_range_cols,
    uint32_t n_copy_range_cols,
    uint64_t* const* __restrict__ child_var_cols,
    uint32_t keep_cover_range,
    uint32_t* __restrict__ child_cover_lo,
    uint32_t* __restrict__ child_cover_hi) {
    uint32_t w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= total_work) {
        return;
    }
    uint32_t r;
    uint32_t lo;
    uint32_t pos;
    uint32_t row_hi;
    if (has_parent_range != 0u) {
        r = upper_bound_u32(work_prefix, n_frontier + 1, w) - 1u;
        lo = parent_lo[r];
        pos = lo + (w - work_prefix[r]);
        row_hi = parent_hi[r];
    } else {
        uint32_t len = const_hi - const_lo;
        r = w / len;
        lo = const_lo;
        pos = const_lo + (w % len);
        row_hi = const_hi;
    }
    uint32_t out;
    if (group_offsets == nullptr) {
        out = w;
    } else {
        out = group_offsets[w];
        if (group_offsets[w + 1u] == out) {
            return; // not a surviving group start
        }
    }
    for (uint32_t j = 0; j < n_copy_var_cols; ++j) {
        child_copy_var_cols[j][out] = parent_copy_var_cols[j][r];
    }
    for (uint32_t j = 0; j < n_copy_range_cols; ++j) {
        child_copy_range_cols[j][out] = parent_copy_range_cols[j][r];
    }
    for (uint32_t j = 0; j < n_cover_cols; ++j) {
        child_var_cols[j][out] = cover_cols[j][pos];
    }
    if (keep_cover_range != 0u) {
        uint32_t g_hi = row_hi;
        for (uint32_t j = 0; j < n_cover_cols; ++j) {
            const uint64_t* col = cover_cols[j];
            g_hi = pos + upper_bound_u64(col + pos, g_hi - pos, col[pos]);
        }
        child_cover_lo[out] = pos;
        child_cover_hi[out] = g_hi;
    }
}

extern "C" __global__ void fj_probe_refine_u64(
    const uint64_t* const* __restrict__ probe_cols,
    uint32_t n_probe_cols,
    const uint64_t* const* __restrict__ key_cols,
    const uint32_t* __restrict__ in_lo,
    const uint32_t* __restrict__ in_hi,
    uint32_t has_range,
    uint32_t const_lo,
    uint32_t const_hi,
    uint32_t n_frontier,
    uint32_t keep_range,
    uint32_t* __restrict__ out_lo,
    uint32_t* __restrict__ out_hi,
    uint8_t* __restrict__ mask,
    uint32_t combine_mask) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_frontier) {
        return;
    }
    uint32_t lo = (has_range != 0u) ? in_lo[i] : const_lo;
    uint32_t hi = (has_range != 0u) ? in_hi[i] : const_hi;
    for (uint32_t j = 0; j < n_probe_cols && lo < hi; ++j) {
        const uint64_t* col = probe_cols[j];
        uint64_t v = key_cols[j][i];
        uint32_t new_lo = lo + lower_bound_u64(col + lo, hi - lo, v);
        uint32_t new_hi = lo + upper_bound_u64(col + lo, hi - lo, v);
        lo = new_lo;
        hi = new_hi;
    }
    uint8_t hit = (lo < hi) ? 1u : 0u;
    if (combine_mask != 0u) {
        hit = (mask[i] != 0u && hit != 0u) ? 1u : 0u;
    }
    mask[i] = hit;
    if (keep_range != 0u) {
        out_lo[i] = lo;
        out_hi[i] = hi;
    }
}

// COUNT epilogue (§2.4 factorized counting): per surviving frontier
// row, the count multiplicity is the product of the remaining live
// trie-range lengths (unconsumed trailing columns of partially
// consumed atoms — the d-representation count). Width-agnostic:
// ranges are u32 row indices in every width class. n_ranges == 0
// (fully-consumed plans) degenerates to multiplicity 1.
extern "C" __global__ void fj_count_multiplicity(
    const uint32_t* const* __restrict__ range_lo_cols,
    const uint32_t* const* __restrict__ range_hi_cols,
    uint32_t n_ranges,
    uint32_t n_frontier,
    unsigned long long* __restrict__ mult) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_frontier) {
        return;
    }
    unsigned long long m = 1ull;
    for (uint32_t k = 0; k < n_ranges; ++k) {
        m *= (unsigned long long)(range_hi_cols[k][i] - range_lo_cols[k][i]);
    }
    mult[i] = m;
}

// ---------------------------------------------------------------------------
// D3 S3 spike — factorized recursive delta (TC novel-set pipeline).
//
// The per-source novel set of a semi-naive TC iteration is
//   novel[x] = (∪_{y ∈ delta[x]} edge[y]) \ R[x]
// — a union of sorted trie ranges minus the stable relation's rows.
// These kernels evaluate that union–diff over a dense-domain
// characteristic bitvector (one row of `words_per_row` u32 words per
// source x) instead of materializing the witness-multiplied flat
// join: rediscoveries and duplicate witnesses collapse in the bitmap
// and only surviving novel tuples are ever written out. The emit
// order (word index = (x, z) order) makes the output lex-sorted and
// full-row-deduped by construction.
//
// Out-of-domain ids fail closed: kernels set *error_flag and skip the
// write; the host surfaces the flag as a typed error after sync.

// Per delta row: locate the edge trie range for y (edge is lex-sorted
// (y, z), full-row deduped) and record its start + length. Lengths go
// into work_prefix for the exclusive scan that drives the mark pass;
// the trailing slot is zeroed for the scan.
extern "C" __global__ void fj_delta_range_u32(
    const uint32_t* __restrict__ delta_y,
    uint32_t n_delta,
    const uint32_t* __restrict__ edge_y,
    uint32_t n_edge,
    uint32_t* __restrict__ range_lo,
    uint32_t* __restrict__ work_prefix) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        work_prefix[n_delta] = 0;
    }
    if (i >= n_delta) {
        return;
    }
    uint32_t y = delta_y[i];
    uint32_t lo = lower_bound_u32(edge_y, n_edge, y);
    uint32_t hi = upper_bound_u32(edge_y, n_edge, y);
    range_lo[i] = lo;
    work_prefix[i] = hi - lo;
}

// Mark candidates: one work item per (delta row, range offset). Sets
// bit (x, z) for every candidate successor; atomicOr collapses
// duplicate witnesses for free.
extern "C" __global__ void fj_delta_mark_u32(
    const uint32_t* __restrict__ delta_x,
    uint32_t n_delta,
    const uint32_t* __restrict__ range_lo,
    const uint32_t* __restrict__ work_prefix,
    uint32_t total_work,
    const uint32_t* __restrict__ edge_z,
    uint32_t* __restrict__ bitmap,
    uint32_t words_per_row,
    uint32_t domain,
    uint32_t* __restrict__ error_flag) {
    uint32_t w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= total_work) {
        return;
    }
    uint32_t r = upper_bound_u32(work_prefix, n_delta + 1, w) - 1u;
    uint32_t k = w - work_prefix[r];
    uint32_t x = delta_x[r];
    uint32_t z = edge_z[range_lo[r] + k];
    if (x >= domain || z >= domain) {
        atomicExch(error_flag, 1u);
        return;
    }
    atomicOr(&bitmap[(size_t)x * words_per_row + (z >> 5)], 1u << (z & 31u));
}

// Subtract the stable relation: clear bit (x, z) for every row of R.
// Clearing a bit that was never marked is a no-op, so R needs no
// pre-filtering against the candidate set.
extern "C" __global__ void fj_delta_subtract_u32(
    const uint32_t* __restrict__ r_x,
    const uint32_t* __restrict__ r_z,
    uint32_t n_r,
    uint32_t* __restrict__ bitmap,
    uint32_t words_per_row,
    uint32_t domain,
    uint32_t* __restrict__ error_flag) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_r) {
        return;
    }
    uint32_t x = r_x[j];
    uint32_t z = r_z[j];
    if (x >= domain || z >= domain) {
        atomicExch(error_flag, 1u);
        return;
    }
    atomicAnd(&bitmap[(size_t)x * words_per_row + (z >> 5)], ~(1u << (z & 31u)));
}

// Per-word survivor counts for the exclusive scan that sizes and
// places the emit. The trailing slot is zeroed for the scan.
extern "C" __global__ void fj_delta_popcount(
    const uint32_t* __restrict__ bitmap,
    uint32_t n_words,
    uint32_t* __restrict__ counts) {
    uint32_t w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w == 0) {
        counts[n_words] = 0;
    }
    if (w >= n_words) {
        return;
    }
    counts[w] = __popc(bitmap[w]);
}

// Emit surviving novel pairs at their scanned offsets. Word order is
// (x, z) order, and bits are drained ascending, so the output is
// lex-sorted and deduped by construction — it is simultaneously the
// next iteration's delta and the union input.
extern "C" __global__ void fj_delta_emit_u32(
    const uint32_t* __restrict__ bitmap,
    uint32_t words_per_row,
    uint32_t n_words,
    const uint32_t* __restrict__ offsets,
    uint32_t* __restrict__ out_x,
    uint32_t* __restrict__ out_z) {
    uint32_t w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= n_words) {
        return;
    }
    uint32_t bits = bitmap[w];
    if (bits == 0u) {
        return;
    }
    uint32_t x = w / words_per_row;
    uint32_t base = (w % words_per_row) << 5;
    uint32_t o = offsets[w];
    while (bits != 0u) {
        uint32_t b = __ffs(bits) - 1u;
        out_x[o] = x;
        out_z[o] = base + b;
        o += 1u;
        bits &= bits - 1u;
    }
}
