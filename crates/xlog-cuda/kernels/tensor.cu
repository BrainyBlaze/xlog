// Dense numerical operations. Matrix products are dispatched through cuBLAS.
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

static __device__ __forceinline__ float from_bf16(uint16_t value) {
    return __uint_as_float(static_cast<uint32_t>(value) << 16);
}

static __device__ __forceinline__ uint16_t to_bf16(float value) {
    uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7fffffffU) > 0x7f800000U) return static_cast<uint16_t>((bits >> 16) | 0x40U);
    return static_cast<uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}

static __device__ float block_sum(float value) {
    __shared__ float warps[8];
    const unsigned lane = threadIdx.x & 31;
    const unsigned warp = threadIdx.x >> 5;
    for (unsigned offset = 16; offset; offset >>= 1) value += __shfl_down_sync(0xffffffffU, value, offset);
    if (lane == 0) warps[warp] = value;
    __syncthreads();
    value = threadIdx.x < 8 ? warps[lane] : 0.0f;
    if (warp == 0) {
        for (unsigned offset = 16; offset; offset >>= 1) value += __shfl_down_sync(0xffffffffU, value, offset);
        if (lane == 0) warps[0] = value;
    }
    __syncthreads();
    return warps[0];
}

static __device__ float block_max(float value) {
    __shared__ float warps[8];
    const unsigned lane = threadIdx.x & 31;
    const unsigned warp = threadIdx.x >> 5;
    for (unsigned offset = 16; offset; offset >>= 1) value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, offset));
    if (lane == 0) warps[warp] = value;
    __syncthreads();
    value = threadIdx.x < 8 ? warps[lane] : -INFINITY;
    if (warp == 0) {
        for (unsigned offset = 16; offset; offset >>= 1) value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, offset));
        if (lane == 0) warps[0] = value;
    }
    __syncthreads();
    return warps[0];
}

extern "C" __global__ void tensor_embedding_bf16(
    const uint32_t* tokens, const uint16_t* weight, uint16_t* output,
    uint64_t elements, uint32_t width
) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < elements) output[i] = weight[static_cast<uint64_t>(tokens[i / width]) * width + i % width];
}

extern "C" __global__ void tensor_rms_norm_bf16(
    const uint16_t* input, const uint16_t* weight, uint16_t* output,
    uint32_t width, float epsilon
) {
    const uint64_t start = static_cast<uint64_t>(blockIdx.x) * width;
    float sum = 0.0f;
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        const float x = from_bf16(input[start + column]);
        sum += x * x;
    }
    const float inverse = rsqrtf(block_sum(sum) / static_cast<float>(width) + epsilon);
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        // The reference casts the normalized activation before multiplying its BF16 scale.
        const uint16_t normalized = to_bf16(from_bf16(input[start + column]) * inverse);
        output[start + column] = to_bf16(from_bf16(normalized) * from_bf16(weight[column]));
    }
}

extern "C" __global__ void tensor_rotary_bf16(
    const uint16_t* input, uint16_t* output, uint64_t elements,
    uint32_t heads, uint32_t head_dim, float theta
) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elements) return;
    const uint32_t column = i % head_dim;
    const uint32_t half = head_dim / 2;
    const uint64_t position = i / (static_cast<uint64_t>(heads) * head_dim);
    const float inverse = 1.0f / powf(theta, static_cast<float>(2 * (column % half)) / head_dim);
    const float angle = static_cast<float>(position) * inverse;
    const float cosine = from_bf16(to_bf16(cosf(angle)));
    const float sine = from_bf16(to_bf16(sinf(angle)));
    const uint64_t partner = column < half ? i + half : i - half;
    const float rotated = (column < half ? -1.0f : 1.0f) * from_bf16(input[partner]);
    const float left = from_bf16(to_bf16(from_bf16(input[i]) * cosine));
    const float right = from_bf16(to_bf16(rotated * sine));
    output[i] = to_bf16(left + right);
}

extern "C" __global__ void tensor_add_bf16(
    const uint16_t* left, const uint16_t* right, uint16_t* output, uint64_t elements
) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < elements) output[i] = to_bf16(from_bf16(left[i]) + from_bf16(right[i]));
}

extern "C" __global__ void tensor_silu_gate_bf16(
    const uint16_t* gate, const uint16_t* value, uint16_t* output, uint64_t elements
) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < elements) {
        const float x = from_bf16(gate[i]);
        const float activated = from_bf16(to_bf16(x / (1.0f + expf(-x))));
        output[i] = to_bf16(activated * from_bf16(value[i]));
    }
}

extern "C" __global__ void tensor_attention_pack_bf16(
    const uint16_t* input, uint16_t* output, uint64_t elements,
    uint32_t sequence, uint32_t heads, uint32_t source_heads, uint32_t head_dim
) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elements) return;
    const uint32_t column = i % head_dim;
    const uint64_t row = i / head_dim;
    const uint32_t position = row % sequence;
    const uint32_t head = row / sequence;
    const uint32_t source_head = head / (heads / source_heads);
    output[i] = input[(static_cast<uint64_t>(position) * source_heads + source_head) * head_dim + column];
}

extern "C" __global__ void tensor_attention_unpack_bf16(
    const uint16_t* input, uint16_t* output, uint64_t elements,
    uint32_t sequence, uint32_t heads, uint32_t head_dim
) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elements) return;
    const uint32_t column = i % head_dim;
    const uint64_t row = i / head_dim;
    const uint32_t head = row % heads;
    const uint32_t position = row / heads;
    output[i] = input[(static_cast<uint64_t>(head) * sequence + position) * head_dim + column];
}

extern "C" __global__ void tensor_causal_softmax_bf16(
    const uint16_t* scores, uint16_t* probabilities, uint32_t sequence, float scale
) {
    const uint32_t query = blockIdx.x % sequence;
    const uint64_t start = static_cast<uint64_t>(blockIdx.x) * sequence;
    float maximum = -INFINITY;
    for (uint32_t key = threadIdx.x; key <= query; key += blockDim.x) {
        const float score = from_bf16(to_bf16(from_bf16(scores[start + key]) * scale));
        maximum = fmaxf(maximum, score);
    }
    maximum = block_max(maximum);
    float denominator = 0.0f;
    for (uint32_t key = threadIdx.x; key <= query; key += blockDim.x) {
        const float score = from_bf16(to_bf16(from_bf16(scores[start + key]) * scale));
        denominator += expf(score - maximum);
    }
    denominator = block_sum(denominator);
    for (uint32_t key = threadIdx.x; key < sequence; key += blockDim.x) {
        float probability = 0.0f;
        if (key <= query) {
            const float score = from_bf16(to_bf16(from_bf16(scores[start + key]) * scale));
            probability = expf(score - maximum) / denominator;
        }
        probabilities[start + key] = to_bf16(probability);
    }
}

extern "C" __global__ void tensor_prediction_row_bf16(
    const uint16_t* input, uint16_t* output, uint32_t width, uint32_t position
) {
    const uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column < width) output[column] = input[static_cast<uint64_t>(position) * width + column];
}

extern "C" __global__ void tensor_bf16_to_f32(
    const uint16_t* input, float* output, uint64_t elements
) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < elements) output[i] = from_bf16(input[i]);
}

extern "C" __global__ void tensor_select_f32(
    const float* input, const uint32_t* indices, float* output, uint32_t count
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = input[indices[i]];
}

extern "C" __global__ void tensor_check_finite_f32(
    const float* input, uint32_t* invalid, uint64_t elements
) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < elements && !isfinite(input[i])) atomicOr(invalid, 1U);
}
