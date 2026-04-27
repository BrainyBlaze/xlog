// kernels/totalorder.cuh
//
// IEEE-754 totalOrder bit normalization for f32 / f64. Single source of
// truth for the host-side and device-side mapping used by:
//   * sort.cu            — multi-column radix-sort key gather
//   * dedup.cu           — deterministic full-row dedup / diff probe
//
// The host-side equivalents live in Rust as `f{32,64}::total_cmp`. All
// three definitions must agree, otherwise the deterministic
// dedup/diff/sort pipeline can sort by one ordering and probe by
// another.
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
