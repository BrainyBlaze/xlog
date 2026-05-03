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
// Algorithm shape (mirrors SRDatalog Algorithm 2 specialised to a
// triangle, with simplifications for v1):
//
//   * Dispatch one thread per row of e_xy. Thread `i` is bound
//     to (X, Y) = (e_xy.col0[i], e_xy.col1[i]) and emits every
//     (X, Y, Z) such that (Y, Z) ∈ e_yz and (X, Z) ∈ e_xz.
//   * Z is found by a sorted-merge intersection of
//     e_yz.col1 over the Y range and e_xz.col1 over the X range,
//     each range located by a per-thread binary search on the
//     respective col0.
//   * Two-pass count → materialize: pass 1 fills `out_counts[i]`
//     with the number of triangles thread `i` will emit. The host
//     prefix-sums `out_counts` to get write offsets and the total
//     row count; pass 2 re-runs the intersection, writing each
//     emitted (X, Y, Z) at `offsets[i] + j` for the j-th match.
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

__device__ __forceinline__ uint32_t intersect_count_u64(
    const uint64_t* __restrict__ a, uint32_t na,
    const uint64_t* __restrict__ b, uint32_t nb) {
    uint32_t ia = 0, ib = 0;
    uint32_t cnt = 0;
    while (ia < na && ib < nb) {
        uint64_t va = a[ia];
        uint64_t vb = b[ib];
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

__device__ __forceinline__ uint32_t intersect_emit_xyz_u64(
    uint64_t x, uint64_t y,
    const uint64_t* __restrict__ a, uint32_t na,
    const uint64_t* __restrict__ b, uint32_t nb,
    uint32_t out_base,
    uint64_t* __restrict__ out_x,
    uint64_t* __restrict__ out_y,
    uint64_t* __restrict__ out_z) {
    uint32_t ia = 0, ib = 0;
    uint32_t emitted = 0;
    while (ia < na && ib < nb) {
        uint64_t va = a[ia];
        uint64_t vb = b[ib];
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

}  // anonymous namespace

// Per-thread count kernel: one thread per row of e_xy.
//
// For thread `i` with row (X, Y) = (xy_col0[i], xy_col1[i]):
//   * Locate the Y range in e_yz via [lower_bound, upper_bound)
//     on yz_col0 → range of Z values e_yz contributes.
//   * Locate the X range in e_xz via [lower_bound, upper_bound)
//     on xz_col0 → range of Z values e_xz contributes.
//   * Count intersection size → out_counts[i].
//
// Threads with `i >= n_xy` do nothing. The kernel writes
// out_counts[i] = 0 in that case if its slot is in range; the
// host pre-zeroes the buffer so no write is needed for OOB.
extern "C" __global__ void wcoj_triangle_count(
    const uint32_t* __restrict__ xy_col0,
    const uint32_t* __restrict__ xy_col1,
    uint32_t n_xy,
    const uint32_t* __restrict__ yz_col0,
    const uint32_t* __restrict__ yz_col1,
    uint32_t n_yz,
    const uint32_t* __restrict__ xz_col0,
    const uint32_t* __restrict__ xz_col1,
    uint32_t n_xz,
    uint32_t* __restrict__ out_counts) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_xy) {
        return;
    }
    uint32_t x = xy_col0[i];
    uint32_t y = xy_col1[i];

    uint32_t yz_lo = lower_bound_u32(yz_col0, n_yz, y);
    uint32_t yz_hi = upper_bound_u32(yz_col0, n_yz, y);
    uint32_t xz_lo = lower_bound_u32(xz_col0, n_xz, x);
    uint32_t xz_hi = upper_bound_u32(xz_col0, n_xz, x);

    uint32_t cnt = 0;
    if (yz_hi > yz_lo && xz_hi > xz_lo) {
        cnt = intersect_count(
            yz_col1 + yz_lo, yz_hi - yz_lo,
            xz_col1 + xz_lo, xz_hi - xz_lo);
    }
    out_counts[i] = cnt;
}

// Single-thread reducer: total = counts[n-1] + offsets[n-1].
//
// `offsets` is produced by an in-place exclusive prefix-sum over
// the per-row counts (the device-side scan helper). The inclusive
// total — i.e. the number of triangles to materialize — is
// `offsets[n-1] + counts[n-1]`. Writing the scalar to device
// memory means the host can use the sanctioned
// `dtoh_scalar_untracked` chokepoint to read just the total,
// avoiding the v1 count-vector D2H.
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

// Per-thread materialize kernel: same lookups + intersection as
// the count kernel, but emits (X, Y, Z) into the output columns
// at positions derived from the device-side prefix-sum of the
// count array (no host scan).
//
// Caller passes:
//   * `out_offsets[i]` — exclusive-scan of the per-row counts,
//     so thread `i` writes its `j`-th triangle at
//     `out_offsets[i] + j`.
//   * `total_rows`     — sum of all counts; the output column
//     allocations must be at least this many u32s. Threads with
//     `out_offsets[i] == total_rows` (i.e., no work to emit)
//     simply return.
extern "C" __global__ void wcoj_triangle_materialize(
    const uint32_t* __restrict__ xy_col0,
    const uint32_t* __restrict__ xy_col1,
    uint32_t n_xy,
    const uint32_t* __restrict__ yz_col0,
    const uint32_t* __restrict__ yz_col1,
    uint32_t n_yz,
    const uint32_t* __restrict__ xz_col0,
    const uint32_t* __restrict__ xz_col1,
    uint32_t n_xz,
    const uint32_t* __restrict__ out_offsets,
    uint32_t total_rows,
    uint32_t* __restrict__ out_x,
    uint32_t* __restrict__ out_y,
    uint32_t* __restrict__ out_z) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_xy) {
        return;
    }
    uint32_t x = xy_col0[i];
    uint32_t y = xy_col1[i];
    uint32_t base = out_offsets[i];
    if (base >= total_rows) {
        return;
    }

    uint32_t yz_lo = lower_bound_u32(yz_col0, n_yz, y);
    uint32_t yz_hi = upper_bound_u32(yz_col0, n_yz, y);
    uint32_t xz_lo = lower_bound_u32(xz_col0, n_xz, x);
    uint32_t xz_hi = upper_bound_u32(xz_col0, n_xz, x);

    if (yz_hi <= yz_lo || xz_hi <= xz_lo) {
        return;
    }
    intersect_emit_xyz(
        x, y,
        yz_col1 + yz_lo, yz_hi - yz_lo,
        xz_col1 + xz_lo, xz_hi - xz_lo,
        base, out_x, out_y, out_z);
}

// ===============================================================
// v0.6.5 slice 2 — 4-cycle WCOJ kernels (u32).
//
//   cycle4(W, X, Y, Z) :- e1(W, X), e2(X, Y), e3(Y, Z), e4(Z, W)
//
// Inputs are already-sorted, already-deduped binary u32 relations:
//   * e1: lex-sorted by (W, X)
//   * e2: lex-sorted by (X, Y)
//   * e3: lex-sorted by (Y, Z)
//   * e4: lex-sorted by (Z, W)
//
// Algorithm: dispatch one thread per row of e1. Thread `i` is bound
// to (W, X) = (e1.col0[i], e1.col1[i]). For each `y` in the X-range
// of e2, then each `z` in the Y-range of e3, the thread checks
// whether (z, w) ∈ e4 via a two-level binary search (z on e4.col0,
// then w on e4.col1 within the Z-equal range). Each closing match
// emits one (W, X, Y, Z) quad.
//
// Two-pass count → materialize, mirrors triangle. Output positions
// come from a deterministic prefix-sum, no atomics.
//
// Heavy-row note: the inner chain is sequential per-thread; the
// outer parallelism is over slot-0 rows. Histogram-guided
// scheduling for heavy slot-0 rows is deferred.
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

}  // anonymous namespace

// Per-thread count kernel: one thread per row of e1.
//
// Thread `i` walks the chain e1 → e2 → e3 → e4 and accumulates the
// number of (W, X, Y, Z) quads it will emit.
extern "C" __global__ void wcoj_4cycle_count(
    const uint32_t* __restrict__ e1_col0,
    const uint32_t* __restrict__ e1_col1,
    uint32_t n_e1,
    const uint32_t* __restrict__ e2_col0,
    const uint32_t* __restrict__ e2_col1,
    uint32_t n_e2,
    const uint32_t* __restrict__ e3_col0,
    const uint32_t* __restrict__ e3_col1,
    uint32_t n_e3,
    const uint32_t* __restrict__ e4_col0,
    const uint32_t* __restrict__ e4_col1,
    uint32_t n_e4,
    uint32_t* __restrict__ out_counts) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_e1) {
        return;
    }
    uint32_t w = e1_col0[i];
    uint32_t x = e1_col1[i];

    uint32_t e2_lo = lower_bound_u32(e2_col0, n_e2, x);
    uint32_t e2_hi = upper_bound_u32(e2_col0, n_e2, x);

    uint32_t cnt = 0;
    for (uint32_t j = e2_lo; j < e2_hi; ++j) {
        uint32_t y = e2_col1[j];
        uint32_t e3_lo = lower_bound_u32(e3_col0, n_e3, y);
        uint32_t e3_hi = upper_bound_u32(e3_col0, n_e3, y);
        for (uint32_t k = e3_lo; k < e3_hi; ++k) {
            uint32_t z = e3_col1[k];
            if (contains_pair_u32(e4_col0, e4_col1, n_e4, z, w)) {
                cnt += 1;
            }
        }
    }
    out_counts[i] = cnt;
}

// Per-thread materialize kernel: same chain walk as the count
// kernel, emits (W, X, Y, Z) into the output columns at
// out_offsets[i] + j for the j-th match.
extern "C" __global__ void wcoj_4cycle_materialize(
    const uint32_t* __restrict__ e1_col0,
    const uint32_t* __restrict__ e1_col1,
    uint32_t n_e1,
    const uint32_t* __restrict__ e2_col0,
    const uint32_t* __restrict__ e2_col1,
    uint32_t n_e2,
    const uint32_t* __restrict__ e3_col0,
    const uint32_t* __restrict__ e3_col1,
    uint32_t n_e3,
    const uint32_t* __restrict__ e4_col0,
    const uint32_t* __restrict__ e4_col1,
    uint32_t n_e4,
    const uint32_t* __restrict__ out_offsets,
    uint32_t total_rows,
    uint32_t* __restrict__ out_w,
    uint32_t* __restrict__ out_x,
    uint32_t* __restrict__ out_y,
    uint32_t* __restrict__ out_z) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_e1) {
        return;
    }
    uint32_t w = e1_col0[i];
    uint32_t x = e1_col1[i];
    uint32_t base = out_offsets[i];
    if (base >= total_rows) {
        return;
    }

    uint32_t e2_lo = lower_bound_u32(e2_col0, n_e2, x);
    uint32_t e2_hi = upper_bound_u32(e2_col0, n_e2, x);

    uint32_t emitted = 0;
    for (uint32_t j = e2_lo; j < e2_hi; ++j) {
        uint32_t y = e2_col1[j];
        uint32_t e3_lo = lower_bound_u32(e3_col0, n_e3, y);
        uint32_t e3_hi = upper_bound_u32(e3_col0, n_e3, y);
        for (uint32_t k = e3_lo; k < e3_hi; ++k) {
            uint32_t z = e3_col1[k];
            if (contains_pair_u32(e4_col0, e4_col1, n_e4, z, w)) {
                uint32_t pos = base + emitted;
                out_w[pos] = w;
                out_x[pos] = x;
                out_y[pos] = y;
                out_z[pos] = z;
                emitted += 1;
            }
        }
    }
}

// ===============================================================
// U64 entry kernels — same shape as the u32 pair above, with
// 64-bit join-key buffers. `wcoj_compute_total` is reused as-is
// (counters stay u32).
// ===============================================================

extern "C" __global__ void wcoj_triangle_count_u64(
    const uint64_t* __restrict__ xy_col0,
    const uint64_t* __restrict__ xy_col1,
    uint32_t n_xy,
    const uint64_t* __restrict__ yz_col0,
    const uint64_t* __restrict__ yz_col1,
    uint32_t n_yz,
    const uint64_t* __restrict__ xz_col0,
    const uint64_t* __restrict__ xz_col1,
    uint32_t n_xz,
    uint32_t* __restrict__ out_counts) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_xy) {
        return;
    }
    uint64_t x = xy_col0[i];
    uint64_t y = xy_col1[i];

    uint32_t yz_lo = lower_bound_u64(yz_col0, n_yz, y);
    uint32_t yz_hi = upper_bound_u64(yz_col0, n_yz, y);
    uint32_t xz_lo = lower_bound_u64(xz_col0, n_xz, x);
    uint32_t xz_hi = upper_bound_u64(xz_col0, n_xz, x);

    uint32_t cnt = 0;
    if (yz_hi > yz_lo && xz_hi > xz_lo) {
        cnt = intersect_count_u64(
            yz_col1 + yz_lo, yz_hi - yz_lo,
            xz_col1 + xz_lo, xz_hi - xz_lo);
    }
    out_counts[i] = cnt;
}

extern "C" __global__ void wcoj_triangle_materialize_u64(
    const uint64_t* __restrict__ xy_col0,
    const uint64_t* __restrict__ xy_col1,
    uint32_t n_xy,
    const uint64_t* __restrict__ yz_col0,
    const uint64_t* __restrict__ yz_col1,
    uint32_t n_yz,
    const uint64_t* __restrict__ xz_col0,
    const uint64_t* __restrict__ xz_col1,
    uint32_t n_xz,
    const uint32_t* __restrict__ out_offsets,
    uint32_t total_rows,
    uint64_t* __restrict__ out_x,
    uint64_t* __restrict__ out_y,
    uint64_t* __restrict__ out_z) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_xy) {
        return;
    }
    uint64_t x = xy_col0[i];
    uint64_t y = xy_col1[i];
    uint32_t base = out_offsets[i];
    if (base >= total_rows) {
        return;
    }

    uint32_t yz_lo = lower_bound_u64(yz_col0, n_yz, y);
    uint32_t yz_hi = upper_bound_u64(yz_col0, n_yz, y);
    uint32_t xz_lo = lower_bound_u64(xz_col0, n_xz, x);
    uint32_t xz_hi = upper_bound_u64(xz_col0, n_xz, x);

    if (yz_hi <= yz_lo || xz_hi <= xz_lo) {
        return;
    }
    intersect_emit_xyz_u64(
        x, y,
        yz_col1 + yz_lo, yz_hi - yz_lo,
        xz_col1 + xz_lo, xz_hi - xz_lo,
        base, out_x, out_y, out_z);
}

// ===============================================================
// v0.6.5 slice 2 — 4-cycle WCOJ kernels (u64).
// Same shape as the u32 pair, with 64-bit join-key buffers.
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

}  // anonymous namespace

extern "C" __global__ void wcoj_4cycle_count_u64(
    const uint64_t* __restrict__ e1_col0,
    const uint64_t* __restrict__ e1_col1,
    uint32_t n_e1,
    const uint64_t* __restrict__ e2_col0,
    const uint64_t* __restrict__ e2_col1,
    uint32_t n_e2,
    const uint64_t* __restrict__ e3_col0,
    const uint64_t* __restrict__ e3_col1,
    uint32_t n_e3,
    const uint64_t* __restrict__ e4_col0,
    const uint64_t* __restrict__ e4_col1,
    uint32_t n_e4,
    uint32_t* __restrict__ out_counts) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_e1) {
        return;
    }
    uint64_t w = e1_col0[i];
    uint64_t x = e1_col1[i];

    uint32_t e2_lo = lower_bound_u64(e2_col0, n_e2, x);
    uint32_t e2_hi = upper_bound_u64(e2_col0, n_e2, x);

    uint32_t cnt = 0;
    for (uint32_t j = e2_lo; j < e2_hi; ++j) {
        uint64_t y = e2_col1[j];
        uint32_t e3_lo = lower_bound_u64(e3_col0, n_e3, y);
        uint32_t e3_hi = upper_bound_u64(e3_col0, n_e3, y);
        for (uint32_t k = e3_lo; k < e3_hi; ++k) {
            uint64_t z = e3_col1[k];
            if (contains_pair_u64(e4_col0, e4_col1, n_e4, z, w)) {
                cnt += 1;
            }
        }
    }
    out_counts[i] = cnt;
}

extern "C" __global__ void wcoj_4cycle_materialize_u64(
    const uint64_t* __restrict__ e1_col0,
    const uint64_t* __restrict__ e1_col1,
    uint32_t n_e1,
    const uint64_t* __restrict__ e2_col0,
    const uint64_t* __restrict__ e2_col1,
    uint32_t n_e2,
    const uint64_t* __restrict__ e3_col0,
    const uint64_t* __restrict__ e3_col1,
    uint32_t n_e3,
    const uint64_t* __restrict__ e4_col0,
    const uint64_t* __restrict__ e4_col1,
    uint32_t n_e4,
    const uint32_t* __restrict__ out_offsets,
    uint32_t total_rows,
    uint64_t* __restrict__ out_w,
    uint64_t* __restrict__ out_x,
    uint64_t* __restrict__ out_y,
    uint64_t* __restrict__ out_z) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_e1) {
        return;
    }
    uint64_t w = e1_col0[i];
    uint64_t x = e1_col1[i];
    uint32_t base = out_offsets[i];
    if (base >= total_rows) {
        return;
    }

    uint32_t e2_lo = lower_bound_u64(e2_col0, n_e2, x);
    uint32_t e2_hi = upper_bound_u64(e2_col0, n_e2, x);

    uint32_t emitted = 0;
    for (uint32_t j = e2_lo; j < e2_hi; ++j) {
        uint64_t y = e2_col1[j];
        uint32_t e3_lo = lower_bound_u64(e3_col0, n_e3, y);
        uint32_t e3_hi = upper_bound_u64(e3_col0, n_e3, y);
        for (uint32_t k = e3_lo; k < e3_hi; ++k) {
            uint64_t z = e3_col1[k];
            if (contains_pair_u64(e4_col0, e4_col1, n_e4, z, w)) {
                uint32_t pos = base + emitted;
                out_w[pos] = w;
                out_x[pos] = x;
                out_y[pos] = y;
                out_z[pos] = z;
                emitted += 1;
            }
        }
    }
}

// ===============================================================
// v0.6.2 A2-lite — adaptive-dispatch skew classifier.
//
// Combined kernel that histograms the three triangle join-key
// columns (e1.col1 = Y, e2.col0 = Y, e3.col0 = X) into 3 × 64
// buckets in a single launch. Caller D2Hs the 768-byte output
// and computes max(max_bucket / row_count) per column, then
// thresholds.
//
// Bucket function: high 6 bits of a multiplicative-hash mixer
// over the key. Critical for distinguishing structured-but-
// uniform inputs (e.g. all keys multiples of 64) from genuinely
// skewed inputs — raw low-bit bucketing (`key & 63`) collapses
// such inputs into bucket 0 and false-positives catastrophically.
// Confirmed by the host probe in
// `docs/evidence/2026-05-01-wcoj-bench-baseline/`: with the
// mixer at 64 buckets the super-hub vs uniform/empty margin is
// ≥ 9× across both 10K and 50K bench fixture sizes.
//
// The kernel does NOT zero its output. Caller must zero on the
// same launch_stream BEFORE invoking the kernel — CUDA does not
// provide a global barrier inside one regular kernel, so an
// in-kernel "zero pass + atomic merge" can race. The caller's
// recorder pattern is:
//
//   1. cuMemsetD8Async(histogram_buf, 0, 192*4)
//   2. wcoj_triangle_skew_histogram_{u32,u64}(...)
//   3. cu_stream.synchronize()
//   4. dtoh_small_metadata_untracked(histogram_buf, 192)
//
// All four steps run under one LaunchRecorder window in the
// provider entry.

#define WCOJ_SKEW_NUM_BUCKETS 64

namespace {

__device__ __forceinline__ uint32_t wcoj_skew_bucket_u32(uint32_t key) {
    // Knuth's golden-ratio constant; high 6 bits have the
    // strongest avalanche characteristics.
    return (key * 2654435761u) >> 26;
}

__device__ __forceinline__ uint32_t wcoj_skew_bucket_u64(uint64_t key) {
    // SplitMix64 mixer; high 6 bits.
    uint64_t z = key + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return (uint32_t)(z >> 58);
}

}  // anonymous namespace

// Combined three-column histogram. Host launches a 1D grid of
// `3 * blocks_per_col` blocks; the kernel splits by
// `blockIdx.x / blocks_per_col` so the first third processes
// `xy_col1`, the second `yz_col0`, the third `xz_col0`.
// Each block builds a shared-mem 64-bucket histogram and
// atomically merges into the global `histograms[col_idx*64..]`.
extern "C" __global__ void wcoj_triangle_skew_histogram_u32(
    const uint32_t* __restrict__ xy_col1,
    uint32_t n_xy,
    const uint32_t* __restrict__ yz_col0,
    uint32_t n_yz,
    const uint32_t* __restrict__ xz_col0,
    uint32_t n_xz,
    uint32_t blocks_per_col,
    uint32_t* __restrict__ histograms) {
    __shared__ uint32_t shared_hist[WCOJ_SKEW_NUM_BUCKETS];
    if (threadIdx.x < WCOJ_SKEW_NUM_BUCKETS) {
        shared_hist[threadIdx.x] = 0;
    }
    __syncthreads();

    uint32_t col_block = blockIdx.x % blocks_per_col;
    uint32_t col_idx = blockIdx.x / blocks_per_col;
    if (col_idx >= 3) {
        return;
    }
    uint32_t i = col_block * blockDim.x + threadIdx.x;

    const uint32_t* col;
    uint32_t n;
    if (col_idx == 0) {
        col = xy_col1; n = n_xy;
    } else if (col_idx == 1) {
        col = yz_col0; n = n_yz;
    } else {
        col = xz_col0; n = n_xz;
    }
    if (i < n) {
        uint32_t b = wcoj_skew_bucket_u32(col[i]);
        atomicAdd(&shared_hist[b], 1u);
    }
    __syncthreads();

    if (threadIdx.x < WCOJ_SKEW_NUM_BUCKETS) {
        uint32_t v = shared_hist[threadIdx.x];
        if (v != 0) {
            atomicAdd(&histograms[col_idx * WCOJ_SKEW_NUM_BUCKETS + threadIdx.x], v);
        }
    }
}

extern "C" __global__ void wcoj_triangle_skew_histogram_u64(
    const uint64_t* __restrict__ xy_col1,
    uint32_t n_xy,
    const uint64_t* __restrict__ yz_col0,
    uint32_t n_yz,
    const uint64_t* __restrict__ xz_col0,
    uint32_t n_xz,
    uint32_t blocks_per_col,
    uint32_t* __restrict__ histograms) {
    __shared__ uint32_t shared_hist[WCOJ_SKEW_NUM_BUCKETS];
    if (threadIdx.x < WCOJ_SKEW_NUM_BUCKETS) {
        shared_hist[threadIdx.x] = 0;
    }
    __syncthreads();

    uint32_t col_block = blockIdx.x % blocks_per_col;
    uint32_t col_idx = blockIdx.x / blocks_per_col;
    if (col_idx >= 3) {
        return;
    }
    uint32_t i = col_block * blockDim.x + threadIdx.x;

    const uint64_t* col;
    uint32_t n;
    if (col_idx == 0) {
        col = xy_col1; n = n_xy;
    } else if (col_idx == 1) {
        col = yz_col0; n = n_yz;
    } else {
        col = xz_col0; n = n_xz;
    }
    if (i < n) {
        uint32_t b = wcoj_skew_bucket_u64(col[i]);
        atomicAdd(&shared_hist[b], 1u);
    }
    __syncthreads();

    if (threadIdx.x < WCOJ_SKEW_NUM_BUCKETS) {
        uint32_t v = shared_hist[threadIdx.x];
        if (v != 0) {
            atomicAdd(&histograms[col_idx * WCOJ_SKEW_NUM_BUCKETS + threadIdx.x], v);
        }
    }
}

// ===============================================================
// v0.6.5 slice 2 — 4-cycle adaptive-dispatch skew classifier.
//
// Combined kernel that histograms the four 4-cycle **lookup-key**
// columns into 4 × 64 buckets in a single launch:
//   * e1.col0 — W (iteration grid partition + W-validation in e4)
//   * e2.col0 — X (lookup key for the e1 → e2 join)
//   * e3.col0 — Y (lookup key for the e2 → e3 join)
//   * e4.col0 — Z (lookup key for the e3 → e4 join)
//
// These are the columns the count kernel binary-searches on;
// concentration on any one of them produces heavy lookup ranges
// for the threads that hit them. Histogramming col1 instead would
// miss skew that exists ONLY on the lookup-key side (e.g. all e2
// rows share the same X but X is uniform across e1 — the kernel
// would still hammer the same e2 range).
//
// Output is a 4 × 64 = 256-bucket histogram (1024 bytes).
// Caller D2Hs the histogram and reduces to per-position skew
// scores; the provider takes max(score_per_position) for the
// dispatch decision (slice 2 plan amendment 7 — keeps score in
// [0, 1] so triangle's 0.10 threshold transfers directly).
//
// Caller zeros the output buffer on the launch_stream before
// invoking the kernel — the same pattern as the triangle
// classifier. Bucket function (high 6 bits of the multiplicative
// hash) is shared with the triangle classifier.
// ===============================================================

extern "C" __global__ void wcoj_4cycle_skew_histogram_u32(
    const uint32_t* __restrict__ e1_col0,
    uint32_t n_e1,
    const uint32_t* __restrict__ e2_col0,
    uint32_t n_e2,
    const uint32_t* __restrict__ e3_col0,
    uint32_t n_e3,
    const uint32_t* __restrict__ e4_col0,
    uint32_t n_e4,
    uint32_t blocks_per_col,
    uint32_t* __restrict__ histograms) {
    __shared__ uint32_t shared_hist[WCOJ_SKEW_NUM_BUCKETS];
    if (threadIdx.x < WCOJ_SKEW_NUM_BUCKETS) {
        shared_hist[threadIdx.x] = 0;
    }
    __syncthreads();

    uint32_t col_block = blockIdx.x % blocks_per_col;
    uint32_t col_idx = blockIdx.x / blocks_per_col;
    if (col_idx >= 4) {
        return;
    }
    uint32_t i = col_block * blockDim.x + threadIdx.x;

    const uint32_t* col;
    uint32_t n;
    if (col_idx == 0) {
        col = e1_col0; n = n_e1;
    } else if (col_idx == 1) {
        col = e2_col0; n = n_e2;
    } else if (col_idx == 2) {
        col = e3_col0; n = n_e3;
    } else {
        col = e4_col0; n = n_e4;
    }
    if (i < n) {
        uint32_t b = wcoj_skew_bucket_u32(col[i]);
        atomicAdd(&shared_hist[b], 1u);
    }
    __syncthreads();

    if (threadIdx.x < WCOJ_SKEW_NUM_BUCKETS) {
        uint32_t v = shared_hist[threadIdx.x];
        if (v != 0) {
            atomicAdd(&histograms[col_idx * WCOJ_SKEW_NUM_BUCKETS + threadIdx.x], v);
        }
    }
}

extern "C" __global__ void wcoj_4cycle_skew_histogram_u64(
    const uint64_t* __restrict__ e1_col0,
    uint32_t n_e1,
    const uint64_t* __restrict__ e2_col0,
    uint32_t n_e2,
    const uint64_t* __restrict__ e3_col0,
    uint32_t n_e3,
    const uint64_t* __restrict__ e4_col0,
    uint32_t n_e4,
    uint32_t blocks_per_col,
    uint32_t* __restrict__ histograms) {
    __shared__ uint32_t shared_hist[WCOJ_SKEW_NUM_BUCKETS];
    if (threadIdx.x < WCOJ_SKEW_NUM_BUCKETS) {
        shared_hist[threadIdx.x] = 0;
    }
    __syncthreads();

    uint32_t col_block = blockIdx.x % blocks_per_col;
    uint32_t col_idx = blockIdx.x / blocks_per_col;
    if (col_idx >= 4) {
        return;
    }
    uint32_t i = col_block * blockDim.x + threadIdx.x;

    const uint64_t* col;
    uint32_t n;
    if (col_idx == 0) {
        col = e1_col0; n = n_e1;
    } else if (col_idx == 1) {
        col = e2_col0; n = n_e2;
    } else if (col_idx == 2) {
        col = e3_col0; n = n_e3;
    } else {
        col = e4_col0; n = n_e4;
    }
    if (i < n) {
        uint32_t b = wcoj_skew_bucket_u64(col[i]);
        atomicAdd(&shared_hist[b], 1u);
    }
    __syncthreads();

    if (threadIdx.x < WCOJ_SKEW_NUM_BUCKETS) {
        uint32_t v = shared_hist[threadIdx.x];
        if (v != 0) {
            atomicAdd(&histograms[col_idx * WCOJ_SKEW_NUM_BUCKETS + threadIdx.x], v);
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
