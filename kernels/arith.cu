#include <cstdint>
#include <cmath>

#define ARITH_OP_ADD 0
#define ARITH_OP_SUB 1
#define ARITH_OP_MUL 2
#define ARITH_OP_DIV 3
#define ARITH_OP_MOD 4
#define ARITH_OP_MIN 5
#define ARITH_OP_MAX 6

extern "C" __global__ void arith_binary_i64(
    const int64_t* __restrict__ a,
    const int64_t* __restrict__ b,
    uint32_t n,
    uint8_t op,
    int64_t* __restrict__ out
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    int64_t x = a[gid];
    int64_t y = b[gid];
    int64_t v = 0;
    uint64_t ux = static_cast<uint64_t>(x);
    uint64_t uy = static_cast<uint64_t>(y);
    switch (op) {
        case ARITH_OP_ADD: v = static_cast<int64_t>(ux + uy); break;
        case ARITH_OP_SUB: v = static_cast<int64_t>(ux - uy); break;
        case ARITH_OP_MUL: v = static_cast<int64_t>(ux * uy); break;
        case ARITH_OP_DIV: v = (y == 0) ? INT64_MAX : (x / y); break;
        case ARITH_OP_MOD: v = (y == 0) ? 0 : (x % y); break;
        case ARITH_OP_MIN: v = (x < y) ? x : y; break;
        case ARITH_OP_MAX: v = (x > y) ? x : y; break;
        default: v = 0;
    }
    out[gid] = v;
}

extern "C" __global__ void arith_binary_i32(
    const int32_t* __restrict__ a,
    const int32_t* __restrict__ b,
    uint32_t n,
    uint8_t op,
    int32_t* __restrict__ out
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    int32_t x = a[gid];
    int32_t y = b[gid];
    int32_t v = 0;
    uint32_t ux = static_cast<uint32_t>(x);
    uint32_t uy = static_cast<uint32_t>(y);
    switch (op) {
        case ARITH_OP_ADD: v = static_cast<int32_t>(ux + uy); break;
        case ARITH_OP_SUB: v = static_cast<int32_t>(ux - uy); break;
        case ARITH_OP_MUL: v = static_cast<int32_t>(ux * uy); break;
        case ARITH_OP_DIV: v = (y == 0) ? INT32_MAX : (x / y); break;
        case ARITH_OP_MOD: v = (y == 0) ? 0 : (x % y); break;
        case ARITH_OP_MIN: v = (x < y) ? x : y; break;
        case ARITH_OP_MAX: v = (x > y) ? x : y; break;
        default: v = 0;
    }
    out[gid] = v;
}

extern "C" __global__ void arith_binary_u64(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint32_t n,
    uint8_t op,
    uint64_t* __restrict__ out
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    uint64_t x = a[gid];
    uint64_t y = b[gid];
    uint64_t v = 0;
    switch (op) {
        case ARITH_OP_ADD: v = x + y; break;
        case ARITH_OP_SUB: v = x - y; break;
        case ARITH_OP_MUL: v = x * y; break;
        case ARITH_OP_DIV: v = (y == 0) ? UINT64_MAX : (x / y); break;
        case ARITH_OP_MOD: v = (y == 0) ? 0 : (x % y); break;
        case ARITH_OP_MIN: v = (x < y) ? x : y; break;
        case ARITH_OP_MAX: v = (x > y) ? x : y; break;
        default: v = 0;
    }
    out[gid] = v;
}

extern "C" __global__ void arith_binary_u32(
    const uint32_t* __restrict__ a,
    const uint32_t* __restrict__ b,
    uint32_t n,
    uint8_t op,
    uint32_t* __restrict__ out
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    uint32_t x = a[gid];
    uint32_t y = b[gid];
    uint32_t v = 0;
    switch (op) {
        case ARITH_OP_ADD: v = x + y; break;
        case ARITH_OP_SUB: v = x - y; break;
        case ARITH_OP_MUL: v = x * y; break;
        case ARITH_OP_DIV: v = (y == 0) ? UINT32_MAX : (x / y); break;
        case ARITH_OP_MOD: v = (y == 0) ? 0 : (x % y); break;
        case ARITH_OP_MIN: v = (x < y) ? x : y; break;
        case ARITH_OP_MAX: v = (x > y) ? x : y; break;
        default: v = 0;
    }
    out[gid] = v;
}

extern "C" __global__ void arith_binary_f64(
    const double* __restrict__ a,
    const double* __restrict__ b,
    uint32_t n,
    uint8_t op,
    double* __restrict__ out
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    double x = a[gid];
    double y = b[gid];
    double v = 0.0;
    switch (op) {
        case ARITH_OP_ADD: v = x + y; break;
        case ARITH_OP_SUB: v = x - y; break;
        case ARITH_OP_MUL: v = x * y; break;
        case ARITH_OP_DIV: v = x / y; break;
        case ARITH_OP_MOD: v = fmod(x, y); break;
        case ARITH_OP_MIN: v = (x < y) ? x : y; break;
        case ARITH_OP_MAX: v = (x > y) ? x : y; break;
        default: v = 0.0;
    }
    out[gid] = v;
}

extern "C" __global__ void arith_binary_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    uint32_t n,
    uint8_t op,
    float* __restrict__ out
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    float x = a[gid];
    float y = b[gid];
    float v = 0.0f;
    switch (op) {
        case ARITH_OP_ADD: v = x + y; break;
        case ARITH_OP_SUB: v = x - y; break;
        case ARITH_OP_MUL: v = x * y; break;
        case ARITH_OP_DIV: v = x / y; break;
        case ARITH_OP_MOD: v = fmodf(x, y); break;
        case ARITH_OP_MIN: v = (x < y) ? x : y; break;
        case ARITH_OP_MAX: v = (x > y) ? x : y; break;
        default: v = 0.0f;
    }
    out[gid] = v;
}

extern "C" __global__ void arith_abs_i64(
    const int64_t* __restrict__ a,
    uint32_t n,
    int64_t* __restrict__ out
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    int64_t v = a[gid];
    uint64_t uv = static_cast<uint64_t>(v);
    if (v < 0) {
        uv = (~uv) + 1;
    }
    out[gid] = static_cast<int64_t>(uv);
}

extern "C" __global__ void arith_abs_i32(
    const int32_t* __restrict__ a,
    uint32_t n,
    int32_t* __restrict__ out
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    int32_t v = a[gid];
    uint32_t uv = static_cast<uint32_t>(v);
    if (v < 0) {
        uv = (~uv) + 1;
    }
    out[gid] = static_cast<int32_t>(uv);
}

extern "C" __global__ void arith_abs_f64(
    const double* __restrict__ a,
    uint32_t n,
    double* __restrict__ out
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    out[gid] = fabs(a[gid]);
}

extern "C" __global__ void arith_abs_f32(
    const float* __restrict__ a,
    uint32_t n,
    float* __restrict__ out
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    out[gid] = fabsf(a[gid]);
}

extern "C" __global__ void arith_pow_f64(
    const double* __restrict__ base,
    const double* __restrict__ exp,
    uint32_t n,
    double* __restrict__ out
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    out[gid] = pow(base[gid], exp[gid]);
}

extern "C" __global__ void arith_fill_const_u32(
    uint32_t value,
    uint32_t n,
    uint32_t* __restrict__ out
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    out[gid] = value;
}

extern "C" __global__ void arith_fill_const_u64(
    uint64_t value,
    uint32_t n,
    uint64_t* __restrict__ out
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    out[gid] = value;
}

extern "C" __global__ void arith_fill_const_i64(
    int64_t value,
    uint32_t n,
    int64_t* __restrict__ out
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    out[gid] = value;
}

extern "C" __global__ void arith_fill_const_i32(
    int32_t value,
    uint32_t n,
    int32_t* __restrict__ out
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    out[gid] = value;
}

extern "C" __global__ void arith_fill_const_f64(
    double value,
    uint32_t n,
    double* __restrict__ out
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    out[gid] = value;
}

extern "C" __global__ void arith_fill_const_f32(
    float value,
    uint32_t n,
    float* __restrict__ out
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    out[gid] = value;
}

extern "C" __global__ void arith_fill_const_u8(
    uint8_t value,
    uint32_t n,
    uint8_t* __restrict__ out
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    out[gid] = value;
}

__device__ inline uint32_t type_size(uint8_t code) {
    switch (code) {
        case 0: return 4;  // U32
        case 1: return 8;  // U64
        case 2: return 4;  // I32
        case 3: return 8;  // I64
        case 4: return 4;  // F32
        case 5: return 8;  // F64
        case 6: return 1;  // Bool
        case 7: return 4;  // Symbol
        default: return 4;
    }
}

__device__ inline double load_as_f64(const uint8_t* p, uint8_t code) {
    switch (code) {
        case 0: return (double)(*(const uint32_t*)p);
        case 1: return (double)(*(const uint64_t*)p);
        case 2: return (double)(*(const int32_t*)p);
        case 3: return (double)(*(const int64_t*)p);
        case 4: return (double)(*(const float*)p);
        case 5: return *(const double*)p;
        case 6: return (double)(*(const uint8_t*)p);
        case 7: return (double)(*(const uint32_t*)p);
        default: return 0.0;
    }
}

__device__ inline int64_t load_as_i64(const uint8_t* p, uint8_t code) {
    switch (code) {
        case 0: return (int64_t)(*(const uint32_t*)p);
        case 1: return (int64_t)(*(const uint64_t*)p);
        case 2: return (int64_t)(*(const int32_t*)p);
        case 3: return *(const int64_t*)p;
        case 4: return (int64_t)(*(const float*)p);
        case 5: return (int64_t)(*(const double*)p);
        case 6: return (int64_t)(*(const uint8_t*)p);
        case 7: return (int64_t)(*(const uint32_t*)p);
        default: return 0;
    }
}

extern "C" __global__ void arith_cast(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    uint32_t n,
    uint8_t src_type,
    uint8_t dst_type
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    uint32_t src_sz = type_size(src_type);
    uint32_t dst_sz = type_size(dst_type);
    const uint8_t* in = input + (uint64_t)gid * src_sz;
    uint8_t* out = output + (uint64_t)gid * dst_sz;

    if (dst_type == 4 || dst_type == 5) {
        double v = load_as_f64(in, src_type);
        if (dst_type == 4) {
            float f = (float)v;
            *(float*)out = f;
        } else {
            *(double*)out = v;
        }
        return;
    }

    int64_t v = load_as_i64(in, src_type);
    switch (dst_type) {
        case 0: *(uint32_t*)out = (uint32_t)v; break;
        case 1: *(uint64_t*)out = (uint64_t)v; break;
        case 2: *(int32_t*)out = (int32_t)v; break;
        case 3: *(int64_t*)out = v; break;
        case 6: *(uint8_t*)out = (uint8_t)(v != 0); break;
        case 7: *(uint32_t*)out = (uint32_t)v; break;
        default: break;
    }
}
