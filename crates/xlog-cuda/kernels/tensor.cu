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

// Cotangents and optimizer state stay FP32. Casts in the BF16 forward use the
// declared straight-through surrogate; no derivative is assigned to token IDs.
extern "C" __global__ void tensor_fill_f32(float* output, uint64_t elements) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < elements) output[i] = 0.0f;
}

extern "C" __global__ void tensor_accumulate_f32(
    const float* left, const float* right, float* output, uint64_t elements
) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < elements) output[i] = left[i] + right[i];
}

extern "C" __global__ void tensor_scale_f32(
    const float* input, float* output, uint64_t elements, float scale
) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < elements) output[i] = input[i] * scale;
}

extern "C" __global__ void tensor_rms_norm_vjp_f32(
    const uint16_t* input, const uint16_t* weight, const float* cotangent,
    float* output, uint32_t width, float epsilon
) {
    const uint64_t start = static_cast<uint64_t>(blockIdx.x) * width;
    float square = 0.0f, product = 0.0f;
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        const float x = from_bf16(input[start + column]);
        square += x * x;
        product += x * from_bf16(weight[column]) * cotangent[start + column];
    }
    const float inverse = rsqrtf(block_sum(square) / static_cast<float>(width) + epsilon);
    __syncthreads();
    const float correction = block_sum(product) * inverse * inverse * inverse / static_cast<float>(width);
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        output[start + column] = inverse * from_bf16(weight[column]) * cotangent[start + column]
            - from_bf16(input[start + column]) * correction;
    }
}

extern "C" __global__ void tensor_rotary_vjp_f32(
    const float* cotangent, float* output, uint64_t elements,
    uint32_t heads, uint32_t head_dim, float theta
) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elements) return;
    const uint32_t column = i % head_dim, half = head_dim / 2;
    const uint64_t position = i / (static_cast<uint64_t>(heads) * head_dim);
    const float inverse = 1.0f / powf(theta, static_cast<float>(2 * (column % half)) / head_dim);
    const float angle = static_cast<float>(position) * inverse;
    const float cosine = from_bf16(to_bf16(cosf(angle)));
    const float sine = from_bf16(to_bf16(sinf(angle)));
    const uint64_t partner = column < half ? i + half : i - half;
    output[i] = cosine * cotangent[i] + (column < half ? sine : -sine) * cotangent[partner];
}

extern "C" __global__ void tensor_silu_gate_vjp_f32(
    const uint16_t* gate, const uint16_t* value, const float* cotangent,
    float* gate_gradient, float* value_gradient, uint64_t elements
) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elements) return;
    const float x = from_bf16(gate[i]);
    const float sigmoid = 1.0f / (1.0f + expf(-x));
    gate_gradient[i] = cotangent[i] * from_bf16(value[i]) * sigmoid * (1.0f + x * (1.0f - sigmoid));
    value_gradient[i] = cotangent[i] * from_bf16(to_bf16(x / (1.0f + expf(-x))));
}

extern "C" __global__ void tensor_prediction_row_vjp_f32(
    const float* cotangent, float* output, uint64_t elements, uint32_t width, uint32_t position
) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < elements) output[i] = i / width == position ? cotangent[i % width] : 0.0f;
}

extern "C" __global__ void tensor_attention_pack_f32(
    const float* input, float* output, uint64_t elements,
    uint32_t sequence, uint32_t heads, uint32_t source_heads, uint32_t head_dim
) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elements) return;
    const uint32_t column = i % head_dim;
    const uint64_t row = i / head_dim;
    const uint32_t position = row % sequence, head = row / sequence;
    const uint32_t source_head = head / (heads / source_heads);
    output[i] = input[(static_cast<uint64_t>(position) * source_heads + source_head) * head_dim + column];
}

extern "C" __global__ void tensor_attention_reduce_f32(
    const float* input, float* output, uint64_t elements,
    uint32_t sequence, uint32_t heads, uint32_t source_heads, uint32_t head_dim
) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elements) return;
    const uint32_t column = i % head_dim;
    const uint64_t row = i / head_dim;
    const uint32_t source_head = row % source_heads, position = row / source_heads;
    const uint32_t group = heads / source_heads;
    float sum = 0.0f;
    for (uint32_t offset = 0; offset < group; ++offset) {
        const uint32_t head = source_head * group + offset;
        sum += input[(static_cast<uint64_t>(head) * sequence + position) * head_dim + column];
    }
    output[i] = sum;
}

extern "C" __global__ void tensor_causal_softmax_vjp_f32(
    const uint16_t* scores, const float* cotangent, float* output, uint32_t sequence, float scale
) {
    const uint32_t query = blockIdx.x % sequence;
    const uint64_t start = static_cast<uint64_t>(blockIdx.x) * sequence;
    float maximum = -INFINITY;
    for (uint32_t key = threadIdx.x; key <= query; key += blockDim.x) {
        maximum = fmaxf(maximum, from_bf16(to_bf16(from_bf16(scores[start + key]) * scale)));
    }
    maximum = block_max(maximum);
    float denominator = 0.0f;
    for (uint32_t key = threadIdx.x; key <= query; key += blockDim.x) {
        denominator += expf(from_bf16(to_bf16(from_bf16(scores[start + key]) * scale)) - maximum);
    }
    denominator = block_sum(denominator);
    __syncthreads();
    float product = 0.0f;
    for (uint32_t key = threadIdx.x; key <= query; key += blockDim.x) {
        const float p = expf(from_bf16(to_bf16(from_bf16(scores[start + key]) * scale)) - maximum) / denominator;
        product += p * cotangent[start + key];
    }
    product = block_sum(product);
    for (uint32_t key = threadIdx.x; key < sequence; key += blockDim.x) {
        // Recompute the probability before its BF16 cast, rather than treating
        // the rounded probabilities as an exactly normalized distribution.
        const float p = key <= query
            ? expf(from_bf16(to_bf16(from_bf16(scores[start + key]) * scale)) - maximum) / denominator
            : 0.0f;
        output[start + key] = key <= query ? scale * p * (cotangent[start + key] - product) : 0.0f;
    }
}

// kind=0: a full-vocabulary next-token target; kind=1: a binary logit difference.
extern "C" __global__ void tensor_loss_f32(
    const float* logits, float* cotangent, float* loss,
    uint32_t vocabulary, uint32_t kind, uint32_t target, uint32_t negative, uint32_t positive
) {
    if (kind == 1) {
        const float a = logits[positive] - logits[negative];
        const float e = expf(-fabsf(a));
        // Evaluate the target residual directly: computing p - 1 first would
        // erase a small positive-target gradient when FP32 rounds p to one.
        const float small = e / (1.0f + e);
        const float large = 1.0f / (1.0f + e);
        const float derivative = target == 1 ? (a >= 0.0f ? -small : -large)
                                             : (a >= 0.0f ? large : small);
        for (uint32_t i = threadIdx.x; i < vocabulary; i += blockDim.x) {
            cotangent[i] = i == positive ? derivative : (i == negative ? -derivative : 0.0f);
        }
        const float signed_margin = target == 1 ? -a : a;
        if (threadIdx.x == 0) loss[0] = fmaxf(signed_margin, 0.0f) + log1pf(e);
    } else {
        float maximum = -INFINITY;
        for (uint32_t i = threadIdx.x; i < vocabulary; i += blockDim.x) maximum = fmaxf(maximum, logits[i]);
        maximum = block_max(maximum);
        float denominator = 0.0f;
        for (uint32_t i = threadIdx.x; i < vocabulary; i += blockDim.x) denominator += expf(logits[i] - maximum);
        denominator = block_sum(denominator);
        for (uint32_t i = threadIdx.x; i < vocabulary; i += blockDim.x) {
            cotangent[i] = expf(logits[i] - maximum) / denominator - (i == target ? 1.0f : 0.0f);
        }
        if (threadIdx.x == 0) loss[0] = maximum - logits[target] + logf(denominator);
    }
}

extern "C" __global__ void tensor_sum_squares_f32(
    const float* input, float* partials, uint64_t elements
) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const float value = i < elements ? input[i] : 0.0f;
    const float sum = block_sum(value * value);
    if (threadIdx.x == 0) partials[blockIdx.x] = sum;
}

extern "C" __global__ void tensor_norm_f32(const float* partials, float* output, uint64_t elements) {
    float sum = 0.0f;
    for (uint64_t i = threadIdx.x; i < elements; i += blockDim.x) sum += partials[i];
    sum = block_sum(sum);
    if (threadIdx.x == 0) output[0] = sqrtf(sum);
}

extern "C" __global__ void tensor_adam_f32(
    const float* master, const float* first, const float* second, const float* gradient,
    float* next_master, float* next_first, float* next_second, uint16_t* forward,
    uint64_t elements, float rate, float clip, float first_correction, float second_correction
) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elements) return;
    const float g = gradient[i] * clip;
    const float m = 0.9f * first[i] + (1.0f - 0.9f) * g;
    const float v = 0.999f * second[i] + (1.0f - 0.999f) * g * g;
    const float value = master[i] - rate * (m / first_correction) / (sqrtf(v / second_correction) + 1.0e-8f);
    next_master[i] = value;
    next_first[i] = m;
    next_second[i] = v;
    forward[i] = to_bf16(value);
}
