// kernels/wcoj.cu
//
// v0.6.2 GPU 3-way Worst-Case Optimal Join — triangle slice (u32).
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
