#pragma once
#include <stdint.h>

// Philox4x32 arithmetic shared by token selection and semantic action sampling.
// Keeping one round separate lets callers retain their existing work accounting.
static __device__ inline void philox4x32_round(uint32_t value[4], uint32_t key[2]) {
    const uint64_t a = uint64_t(0xd2511f53U) * value[0];
    const uint64_t b = uint64_t(0xcd9e8d57U) * value[2];
    const uint32_t next[4] = {
        uint32_t(b >> 32) ^ value[1] ^ key[0], uint32_t(b),
        uint32_t(a >> 32) ^ value[3] ^ key[1], uint32_t(a)
    };
    for (int i = 0; i < 4; ++i) value[i] = next[i];
    key[0] += 0x9e3779b9U;
    key[1] += 0xbb67ae85U;
}

static __device__ inline void philox4x32_10(
    uint32_t output[4], const uint32_t counter[4], const uint32_t key[2]
) {
    for (int i = 0; i < 4; ++i) output[i] = counter[i];
    uint32_t round_key[2] = {key[0], key[1]};
    for (int round = 0; round < 10; ++round) philox4x32_round(output, round_key);
}
