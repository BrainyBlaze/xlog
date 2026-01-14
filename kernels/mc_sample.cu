// Monte Carlo Bernoulli sampler for XLOG.
//
// This kernel fills `out[i]` with 0/1 samples for independent Bernoulli
// variables, laid out as a row-major matrix:
//   i = sample_idx * num_vars + var_idx
//
// The random source is a counter-based SplitMix64 mixer keyed by (seed, i).
// This provides reproducible, statistically decent 32-bit uniforms for MC.

#include <stdint.h>

static __device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

extern "C" __global__ void mc_sample_bernoulli(
    uint8_t* __restrict__ out,
    const float* __restrict__ probs,
    uint32_t num_vars,
    uint32_t num_samples,
    uint64_t seed
) {
    const uint64_t tid = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    const uint64_t total = (uint64_t)num_vars * (uint64_t)num_samples;
    if (tid >= total) {
        return;
    }

    const uint32_t var_idx = (uint32_t)(tid % (uint64_t)num_vars);
    const float p = probs[var_idx];

    // Clamp p defensively; callers validate probabilities but this avoids NaNs
    // propagating into unpredictable behavior.
    const float p_clamped = (p <= 0.0f) ? 0.0f : ((p >= 1.0f) ? 1.0f : p);

    // Generate a 32-bit uniform from SplitMix64, then convert to (0,1).
    const uint64_t x = splitmix64(seed ^ (tid * 0x9e3779b97f4a7c15ULL));
    const uint32_t r = (uint32_t)(x >> 32);
    const float u = ((float)r + 0.5f) * (1.0f / 4294967296.0f); // 2^-32

    out[tid] = (u < p_clamped) ? 1u : 0u;
}

