// kernels/totalorder.cuh
//
// IEEE-754 totalOrder bit normalization for f32 / f64. Single source of
// truth for the host-side and device-side mapping used by:
//   * sort.cu            — multi-column radix-sort key gather
//   * dedup.cu           — deterministic full-row dedup / diff probe
//
// The host-side equivalents live in `xlog_core::float_order`. Every
// relational comparison, selection, deduplication, difference, and sort path
// must use this same bijection.
//
// Mapping: take the raw bit pattern; if the sign bit is 0 (non-negative),
// flip the sign bit; if the sign bit is 1 (negative), flip every bit.
// The result is a bijection over the 32- / 64-bit space, so distinct
// raw bit patterns map to distinct ordered keys (and vice versa).

#pragma once

#include <cstdint>

__device__ __forceinline__ uint32_t xlog_f32_to_ordered_u32(uint32_t bits) {
    uint32_t sign = bits >> 31;
    uint32_t mask = sign ? 0xFFFFFFFFu : 0x80000000u;
    return bits ^ mask;
}

__device__ __forceinline__ uint64_t xlog_f64_to_ordered_u64(uint64_t bits) {
    uint64_t sign = bits >> 63;
    uint64_t mask = sign ? 0xFFFFFFFFFFFFFFFFull : 0x8000000000000000ull;
    return bits ^ mask;
}

__device__ __forceinline__ uint32_t xlog_f32_total_order_key(float value) {
    return xlog_f32_to_ordered_u32(__float_as_uint(value));
}

__device__ __forceinline__ uint64_t xlog_f64_total_order_key(double value) {
    return xlog_f64_to_ordered_u64((uint64_t)__double_as_longlong(value));
}

template <typename Key>
__device__ __forceinline__ bool xlog_total_order_compare_keys(
    Key left,
    Key right,
    uint8_t op
) {
    switch (op) {
        case 0: return left == right;
        case 1: return left != right;
        case 2: return left < right;
        case 3: return left <= right;
        case 4: return left > right;
        case 5: return left >= right;
        default: return false;
    }
}

__device__ __forceinline__ bool xlog_f32_total_order_compare(
    float left,
    float right,
    uint8_t op
) {
    return xlog_total_order_compare_keys(
        xlog_f32_total_order_key(left),
        xlog_f32_total_order_key(right),
        op
    );
}

__device__ __forceinline__ bool xlog_f64_total_order_compare(
    double left,
    double right,
    uint8_t op
) {
    return xlog_total_order_compare_keys(
        xlog_f64_total_order_key(left),
        xlog_f64_total_order_key(right),
        op
    );
}

__device__ __forceinline__ float xlog_f32_total_min(float left, float right) {
    return xlog_f32_total_order_key(left) <= xlog_f32_total_order_key(right)
        ? left
        : right;
}

__device__ __forceinline__ float xlog_f32_total_max(float left, float right) {
    return xlog_f32_total_order_key(left) >= xlog_f32_total_order_key(right)
        ? left
        : right;
}

__device__ __forceinline__ double xlog_f64_total_min(double left, double right) {
    return xlog_f64_total_order_key(left) <= xlog_f64_total_order_key(right)
        ? left
        : right;
}

__device__ __forceinline__ double xlog_f64_total_max(double left, double right) {
    return xlog_f64_total_order_key(left) >= xlog_f64_total_order_key(right)
        ? left
        : right;
}
