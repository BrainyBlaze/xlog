# GPU-Resident Execution + CLI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove CPU round-trips in filter/groupby/arithmetic and ship a production `xlog` CLI with deterministic + probabilistic support.

**Architecture:** Build GPU predicate evaluation with typed kernels + mask DAG composition, move groupby IDs/keys fully on-device, and add a new `xlog` CLI crate that uses Arrow IPC for I/O. Keep deterministic semantics, explicit errors, and release-quality tests.

**Tech Stack:** Rust, CUDA PTX, cudarc, Arrow IPC, clap, assert_cmd

---

### Task 1: Define scalar type codes for device kernels

**Files:**
- Modify: `crates/xlog-core/src/types.rs`
- Test: `crates/xlog-core/src/types.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn test_scalar_type_code_roundtrip() {
    for ty in [
        ScalarType::U32,
        ScalarType::U64,
        ScalarType::I32,
        ScalarType::I64,
        ScalarType::F32,
        ScalarType::F64,
        ScalarType::Bool,
        ScalarType::Symbol,
    ] {
        let code = ty.to_code();
        let back = ScalarType::from_code(code).unwrap();
        assert_eq!(ty, back);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-core --lib test_scalar_type_code_roundtrip`
Expected: FAIL (missing methods `to_code`/`from_code`).

**Step 3: Implement code mapping**

Add to `crates/xlog-core/src/types.rs`:

```rust
impl ScalarType {
    pub fn to_code(&self) -> u8 {
        match self {
            ScalarType::U32 => 0,
            ScalarType::U64 => 1,
            ScalarType::I32 => 2,
            ScalarType::I64 => 3,
            ScalarType::F32 => 4,
            ScalarType::F64 => 5,
            ScalarType::Bool => 6,
            ScalarType::Symbol => 7,
        }
    }

    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(ScalarType::U32),
            1 => Some(ScalarType::U64),
            2 => Some(ScalarType::I32),
            3 => Some(ScalarType::I64),
            4 => Some(ScalarType::F32),
            5 => Some(ScalarType::F64),
            6 => Some(ScalarType::Bool),
            7 => Some(ScalarType::Symbol),
            _ => None,
        }
    }
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-core --lib test_scalar_type_code_roundtrip`
Expected: PASS.

**Step 5: Commit**

```bash
git add crates/xlog-core/src/types.rs
git commit -m "feat(core): add scalar type codes for kernels"
```

---

### Task 2: Add CUDA arithmetic kernels and load ARITH module

**Files:**
- Create: `kernels/arith.cu`
- Modify: `crates/xlog-cuda/build.rs`
- Modify: `crates/xlog-cuda/src/provider/mod.rs`
- Test: `crates/xlog-cuda/tests/ptx_validation.rs`

**Step 1: Write failing test for ARITH PTX presence**

Append to `crates/xlog-cuda/tests/ptx_validation.rs`:

```rust
#[test]
fn validate_arith_ptx_contains_expected_kernels() {
    let ptx = std::fs::read_to_string("kernels/arith.ptx")
        .expect("arith.ptx should exist after build");
    assert!(ptx.contains("arith_binary_i64"));
    assert!(ptx.contains("arith_binary_f64"));
    assert!(ptx.contains("arith_abs_i64"));
    assert!(ptx.contains("arith_cast"));
    assert!(ptx.contains("arith_fill_const_u32"));
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-cuda-tests --test ptx_validation validate_arith_ptx_contains_expected_kernels --release`
Expected: FAIL (missing `kernels/arith.ptx`).

**Step 3: Add `kernels/arith.cu`**

Create `kernels/arith.cu`:

```cuda
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
    switch (op) {
        case ARITH_OP_ADD: v = x + y; break;
        case ARITH_OP_SUB: v = x - y; break;
        case ARITH_OP_MUL: v = x * y; break;
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
    switch (op) {
        case ARITH_OP_ADD: v = x + y; break;
        case ARITH_OP_SUB: v = x - y; break;
        case ARITH_OP_MUL: v = x * y; break;
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
    out[gid] = (v < 0) ? -v : v;
}

extern "C" __global__ void arith_abs_i32(
    const int32_t* __restrict__ a,
    uint32_t n,
    int32_t* __restrict__ out
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    int32_t v = a[gid];
    out[gid] = (v < 0) ? -v : v;
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
```

**Step 4: Compile new kernel in build script**

Update `crates/xlog-cuda/build.rs` kernel list:

```rust
let kernels = [
    "join", "dedup", "groupby", "scan", "sort", "filter", "pack", "set_ops", "circuit",
    "mc_sample", "arith",
];
```

**Step 5: Load ARITH module in provider**

Add to `crates/xlog-cuda/src/provider/mod.rs`:

```rust
const ARITH_PTX: &str = include_str!("../../../kernels/arith.ptx");
pub const ARITH_MODULE: &str = "xlog_arith";

pub mod arith_kernels {
    pub const ARITH_BINARY_I64: &str = "arith_binary_i64";
    pub const ARITH_BINARY_I32: &str = "arith_binary_i32";
    pub const ARITH_BINARY_U64: &str = "arith_binary_u64";
    pub const ARITH_BINARY_U32: &str = "arith_binary_u32";
    pub const ARITH_BINARY_F64: &str = "arith_binary_f64";
    pub const ARITH_BINARY_F32: &str = "arith_binary_f32";
    pub const ARITH_ABS_I64: &str = "arith_abs_i64";
    pub const ARITH_ABS_I32: &str = "arith_abs_i32";
    pub const ARITH_ABS_F64: &str = "arith_abs_f64";
    pub const ARITH_ABS_F32: &str = "arith_abs_f32";
    pub const ARITH_POW_F64: &str = "arith_pow_f64";
    pub const ARITH_CAST: &str = "arith_cast";
    pub const ARITH_FILL_CONST_U32: &str = "arith_fill_const_u32";
    pub const ARITH_FILL_CONST_U64: &str = "arith_fill_const_u64";
    pub const ARITH_FILL_CONST_I64: &str = "arith_fill_const_i64";
    pub const ARITH_FILL_CONST_I32: &str = "arith_fill_const_i32";
    pub const ARITH_FILL_CONST_F64: &str = "arith_fill_const_f64";
    pub const ARITH_FILL_CONST_F32: &str = "arith_fill_const_f32";
    pub const ARITH_FILL_CONST_U8: &str = "arith_fill_const_u8";
}
```

Add ARITH module load inside `CudaKernelProvider::new`:

```rust
let arith_module = Ptx::from_src(ARITH_PTX);
let _arith = device
    .inner()
    .load_ptx(arith_module, ARITH_MODULE, &[
        arith_kernels::ARITH_BINARY_I64,
        arith_kernels::ARITH_BINARY_I32,
        arith_kernels::ARITH_BINARY_U64,
        arith_kernels::ARITH_BINARY_U32,
        arith_kernels::ARITH_BINARY_F64,
        arith_kernels::ARITH_BINARY_F32,
        arith_kernels::ARITH_ABS_I64,
        arith_kernels::ARITH_ABS_I32,
        arith_kernels::ARITH_ABS_F64,
        arith_kernels::ARITH_ABS_F32,
        arith_kernels::ARITH_POW_F64,
        arith_kernels::ARITH_CAST,
        arith_kernels::ARITH_FILL_CONST_U32,
        arith_kernels::ARITH_FILL_CONST_U64,
        arith_kernels::ARITH_FILL_CONST_I64,
        arith_kernels::ARITH_FILL_CONST_I32,
        arith_kernels::ARITH_FILL_CONST_F64,
        arith_kernels::ARITH_FILL_CONST_F32,
        arith_kernels::ARITH_FILL_CONST_U8,
    ])
    .map_err(|e| XlogError::Kernel(format!("Failed to load arith PTX: {}", e)))?;
```

**Step 6: Run ARITH PTX validation test**

Run: `cargo test -p xlog-cuda-tests --test ptx_validation validate_arith_ptx_contains_expected_kernels --release`
Expected: PASS.

**Step 7: Commit**

```bash
git add kernels/arith.cu crates/xlog-cuda/build.rs crates/xlog-cuda/src/provider/mod.rs crates/xlog-cuda-tests/tests/ptx_validation.rs
git commit -m "feat(cuda): add arithmetic PTX module"
```

---

### Task 3: Wire GPU arithmetic ops in provider

**Files:**
- Modify: `crates/xlog-cuda/src/provider/mod.rs`
- Test: `crates/xlog-cuda/tests/type_coverage_tests.rs`

**Step 1: Write failing tests for numeric types**

Add to `crates/xlog-cuda/tests/type_coverage_tests.rs`:

```rust
#[test]
fn test_arith_u32_i32_u64_f32() {
    let Some(provider) = create_test_provider() else {
        return;
    };

    let schema_u32 = Schema::new(vec![("v".to_string(), ScalarType::U32)]);
    let a_u32 = provider.create_buffer_from_u32_slice(&[1, 2, 3], schema_u32.clone()).unwrap();
    let b_u32 = provider.create_buffer_from_u32_slice(&[4, 5, 6], schema_u32.clone()).unwrap();
    let sum_u32 = provider.add_columns(&a_u32, &b_u32).unwrap();
    let vals_u32 = provider.download_column_u32(&sum_u32, 0).unwrap();
    assert_eq!(vals_u32, vec![5, 7, 9]);

    let schema_i32 = Schema::new(vec![("v".to_string(), ScalarType::I32)]);
    let a_i32 = provider.create_buffer_from_i32_slice(&[-3, 4, -5], schema_i32.clone()).unwrap();
    let abs_i32 = provider.abs_column(&a_i32).unwrap();
    let vals_i32 = provider.download_column_i32(&abs_i32, 0).unwrap();
    assert_eq!(vals_i32, vec![3, 4, 5]);

    let schema_u64 = Schema::new(vec![("v".to_string(), ScalarType::U64)]);
    let a_u64 = provider.create_buffer_from_u64_slice(&[10, 20, 30], schema_u64.clone()).unwrap();
    let b_u64 = provider.create_buffer_from_u64_slice(&[1, 2, 3], schema_u64.clone()).unwrap();
    let diff_u64 = provider.sub_columns(&a_u64, &b_u64).unwrap();
    let vals_u64 = provider.download_column_u64(&diff_u64, 0).unwrap();
    assert_eq!(vals_u64, vec![9, 18, 27]);

    let schema_f32 = Schema::new(vec![("v".to_string(), ScalarType::F32)]);
    let a_f32 = provider.create_buffer_from_f32_slice(&[1.5, -2.0, 3.0], schema_f32.clone()).unwrap();
    let b_f32 = provider.create_buffer_from_f32_slice(&[2.0, 2.0, 0.5], schema_f32.clone()).unwrap();
    let prod_f32 = provider.mul_columns(&a_f32, &b_f32).unwrap();
    let vals_f32 = provider.download_column_f32(&prod_f32, 0).unwrap();
    assert!((vals_f32[0] - 3.0).abs() < 1e-6);
    assert!((vals_f32[1] + 4.0).abs() < 1e-6);
    assert!((vals_f32[2] - 1.5).abs() < 1e-6);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-cuda --test type_coverage_tests test_arith_u32_i32_u64_f32 --release`
Expected: FAIL (new kernels not wired, or missing download helpers for i32/f32 if not used).

**Step 3: Implement GPU arithmetic dispatch**

In `crates/xlog-cuda/src/provider/mod.rs`, replace `binary_arith_op`, `abs_column`, `pow_columns`, and `cast_column` to use ARITH kernels. Use op codes matching `arith.cu`.

Add helper to launch binary ops:

```rust
fn binary_arith_op_device<T: DeviceRepr>(
    &self,
    a: &CudaBuffer,
    b: &CudaBuffer,
    op: u8,
    kernel: &str,
) -> Result<CudaBuffer> {
    if a.num_rows() != b.num_rows() || a.arity() != 1 || b.arity() != 1 {
        return Err(XlogError::Kernel("Arithmetic requires matching single-column buffers".into()));
    }
    let n = a.num_rows() as u32;
    let col_a = a.column(0).ok_or_else(|| XlogError::Kernel("Missing column 0".into()))?;
    let col_b = b.column(0).ok_or_else(|| XlogError::Kernel("Missing column 0".into()))?;

    let mut out = self.memory.alloc::<T>(n as usize)?;
    let func = self.device.inner().get_func(ARITH_MODULE, kernel)
        .ok_or_else(|| XlogError::Kernel("arith kernel not found".into()))?;
    let config = LaunchConfig::for_num_elems(n);

    unsafe {
        func.clone().launch(config, (col_a, col_b, n, op, &mut out))
    }
    .map_err(|e| XlogError::Kernel(format!("arith binary failed: {}", e)))?;

    self.device.synchronize()?;
    Ok(CudaBuffer::from_columns(vec![out.into()], a.num_rows(), a.schema.clone()))
}
```

Update public arithmetic methods to call this helper with the correct kernel:

```rust
pub fn add_columns(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
    match a.schema().column_type(0) {
        Some(ScalarType::I64) => self.binary_arith_op_device::<i64>(a, b, 0, arith_kernels::ARITH_BINARY_I64),
        Some(ScalarType::I32) => self.binary_arith_op_device::<i32>(a, b, 0, arith_kernels::ARITH_BINARY_I32),
        Some(ScalarType::U64) => self.binary_arith_op_device::<u64>(a, b, 0, arith_kernels::ARITH_BINARY_U64),
        Some(ScalarType::U32 | ScalarType::Symbol) => self.binary_arith_op_device::<u32>(a, b, 0, arith_kernels::ARITH_BINARY_U32),
        Some(ScalarType::F64) => self.binary_arith_op_device::<f64>(a, b, 0, arith_kernels::ARITH_BINARY_F64),
        Some(ScalarType::F32) => self.binary_arith_op_device::<f32>(a, b, 0, arith_kernels::ARITH_BINARY_F32),
        other => Err(XlogError::Kernel(format!("Arithmetic not supported for {:?}", other))),
    }
}
```

Repeat for `sub_columns`, `mul_columns`, `div_columns`, `mod_columns`, `min_columns`, `max_columns` with op codes 1..6.

Implement GPU abs:

```rust
pub fn abs_column(&self, a: &CudaBuffer) -> Result<CudaBuffer> {
    if a.arity() != 1 {
        return Err(XlogError::Kernel("Arithmetic requires single-column buffers".into()));
    }
    let n = a.num_rows() as u32;
    let col = a.column(0).ok_or_else(|| XlogError::Kernel("Missing column 0".into()))?;
    let config = LaunchConfig::for_num_elems(n);

    match a.schema().column_type(0) {
        Some(ScalarType::I64) => {
            let mut out = self.memory.alloc::<i64>(n as usize)?;
            let func = self.device.inner().get_func(ARITH_MODULE, arith_kernels::ARITH_ABS_I64)
                .ok_or_else(|| XlogError::Kernel("arith_abs_i64 not found".into()))?;
            unsafe { func.clone().launch(config, (col, n, &mut out)) }
                .map_err(|e| XlogError::Kernel(format!("abs_i64 failed: {}", e)))?;
            self.device.synchronize()?;
            Ok(CudaBuffer::from_columns(vec![out.into()], a.num_rows(), a.schema.clone()))
        }
        Some(ScalarType::I32) => {
            let mut out = self.memory.alloc::<i32>(n as usize)?;
            let func = self.device.inner().get_func(ARITH_MODULE, arith_kernels::ARITH_ABS_I32)
                .ok_or_else(|| XlogError::Kernel("arith_abs_i32 not found".into()))?;
            unsafe { func.clone().launch(config, (col, n, &mut out)) }
                .map_err(|e| XlogError::Kernel(format!("abs_i32 failed: {}", e)))?;
            self.device.synchronize()?;
            Ok(CudaBuffer::from_columns(vec![out.into()], a.num_rows(), a.schema.clone()))
        }
        Some(ScalarType::F64) => {
            let mut out = self.memory.alloc::<f64>(n as usize)?;
            let func = self.device.inner().get_func(ARITH_MODULE, arith_kernels::ARITH_ABS_F64)
                .ok_or_else(|| XlogError::Kernel("arith_abs_f64 not found".into()))?;
            unsafe { func.clone().launch(config, (col, n, &mut out)) }
                .map_err(|e| XlogError::Kernel(format!("abs_f64 failed: {}", e)))?;
            self.device.synchronize()?;
            Ok(CudaBuffer::from_columns(vec![out.into()], a.num_rows(), a.schema.clone()))
        }
        Some(ScalarType::F32) => {
            let mut out = self.memory.alloc::<f32>(n as usize)?;
            let func = self.device.inner().get_func(ARITH_MODULE, arith_kernels::ARITH_ABS_F32)
                .ok_or_else(|| XlogError::Kernel("arith_abs_f32 not found".into()))?;
            unsafe { func.clone().launch(config, (col, n, &mut out)) }
                .map_err(|e| XlogError::Kernel(format!("abs_f32 failed: {}", e)))?;
            self.device.synchronize()?;
            Ok(CudaBuffer::from_columns(vec![out.into()], a.num_rows(), a.schema.clone()))
        }
        Some(ScalarType::U32 | ScalarType::U64 | ScalarType::Bool | ScalarType::Symbol) => {
            self.clone_buffer(a)
        }
        other => Err(XlogError::Kernel(format!("Abs not supported for {:?}", other))),
    }
}
```

Implement `pow_columns` using f64 conversion + `arith_pow_f64` (cast to f64 first using `cast_column` then call pow kernel).

Implement `cast_column` using `arith_cast` kernel with `ScalarType::to_code()` for src/dst.

**Step 4: Run type coverage test**

Run: `cargo test -p xlog-cuda --test type_coverage_tests test_arith_u32_i32_u64_f32 --release`
Expected: PASS.

**Step 5: Commit**

```bash
git add crates/xlog-cuda/src/provider/mod.rs crates/xlog-cuda/tests/type_coverage_tests.rs
 git commit -m "feat(cuda): run arithmetic ops on GPU"
```

---

### Task 4: Add typed compare kernels and column-column comparisons

**Files:**
- Modify: `kernels/filter.cu`
- Modify: `crates/xlog-cuda/src/provider/mod.rs`
- Test: `crates/xlog-cuda/tests/filter_tests.rs`

**Step 1: Add failing tests for new types and column-column compare**

Append to `crates/xlog-cuda/tests/filter_tests.rs`:

```rust
#[test]
fn test_filter_i32_u64_f32_bool_and_column_column_compare() {
    let Some(provider) = create_test_provider() else {
        return;
    };

    let schema_i32 = Schema::new(vec![("v".to_string(), ScalarType::I32)]);
    let buf_i32 = provider.create_buffer_from_i32_slice(&[-2, 0, 3], schema_i32).unwrap();
    let filtered_i32 = provider.filter_i32(&buf_i32, 0, -1, CompareOp::Gt).unwrap();
    let vals_i32 = provider.download_column_i32(&filtered_i32, 0).unwrap();
    assert_eq!(vals_i32, vec![0, 3]);

    let schema_u64 = Schema::new(vec![("v".to_string(), ScalarType::U64)]);
    let buf_u64 = provider.create_buffer_from_u64_slice(&[1, 5, 9], schema_u64).unwrap();
    let filtered_u64 = provider.filter_u64(&buf_u64, 0, 5, CompareOp::Ge).unwrap();
    let vals_u64 = provider.download_column_u64(&filtered_u64, 0).unwrap();
    assert_eq!(vals_u64, vec![5, 9]);

    let schema_f32 = Schema::new(vec![("v".to_string(), ScalarType::F32)]);
    let buf_f32 = provider.create_buffer_from_f32_slice(&[1.0, -1.5, 2.5], schema_f32).unwrap();
    let filtered_f32 = provider.filter_f32(&buf_f32, 0, 0.0, CompareOp::Gt).unwrap();
    let vals_f32 = provider.download_column_f32(&filtered_f32, 0).unwrap();
    assert_eq!(vals_f32.len(), 2);

    let schema_bool = Schema::new(vec![("v".to_string(), ScalarType::Bool)]);
    let buf_bool = provider.create_buffer_from_u8_slice(&[0, 1, 1, 0], schema_bool).unwrap();
    let filtered_bool = provider.filter_bool(&buf_bool, 0, true, CompareOp::Eq).unwrap();
    let vals_bool = provider.download_column_u8(&filtered_bool, 0).unwrap();
    assert_eq!(vals_bool, vec![1, 1]);

    // Column-column compare (u32)
    let schema_u32 = Schema::new(vec![
        ("a".to_string(), ScalarType::U32),
        ("b".to_string(), ScalarType::U32),
    ]);
    let buf_u32 = provider
        .create_buffer_from_u32_columns(&[&[1, 2, 3], &[1, 9, 3]], schema_u32)
        .unwrap();
    let mask = provider.compare_columns_u32(&buf_u32, 0, 1, CompareOp::Eq).unwrap();
    let filtered = provider.filter_by_device_mask(&buf_u32, &mask).unwrap();
    let vals = provider.download_column_u32(&filtered, 0).unwrap();
    assert_eq!(vals, vec![1, 3]);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-cuda --test filter_tests test_filter_i32_u64_f32_bool_and_column_column_compare --release`
Expected: FAIL (missing kernels and provider methods).

**Step 3: Add kernels in `kernels/filter.cu`**

Add typed compare kernels for i32/u64/f32/bool and column-column compares:

```cuda
extern "C" __global__ void filter_compare_i32(
    const int32_t* __restrict__ column,
    int32_t constant,
    uint32_t num_rows,
    uint8_t op,
    uint8_t* __restrict__ mask
) { /* same pattern as i64 */ }

extern "C" __global__ void filter_compare_u64(
    const uint64_t* __restrict__ column,
    uint64_t constant,
    uint32_t num_rows,
    uint8_t op,
    uint8_t* __restrict__ mask
) { /* same pattern as u32 */ }

extern "C" __global__ void filter_compare_f32(
    const float* __restrict__ column,
    float constant,
    uint32_t num_rows,
    uint8_t op,
    uint8_t* __restrict__ mask
) { /* same pattern as f64 */ }

extern "C" __global__ void filter_compare_u8(
    const uint8_t* __restrict__ column,
    uint8_t constant,
    uint32_t num_rows,
    uint8_t op,
    uint8_t* __restrict__ mask
) { /* compare u8 */ }

extern "C" __global__ void filter_compare_u32_col(
    const uint32_t* __restrict__ left,
    const uint32_t* __restrict__ right,
    uint32_t num_rows,
    uint8_t op,
    uint8_t* __restrict__ mask
) { /* column-column compare */ }

// Add i32/i64/u64/f32/f64/u8 column-column variants with same pattern.
```

**Step 4: Wire provider methods**

In `crates/xlog-cuda/src/provider/mod.rs`:
- Add new kernel names in `filter_kernels` for the added kernels.
- Load them in module init.
- Add methods:

```rust
pub fn filter_i32(&self, input: &CudaBuffer, col: usize, value: i32, op: CompareOp) -> Result<CudaBuffer> { /* launch filter_compare_i32 + prefix scan + compact */ }
pub fn filter_u64(&self, input: &CudaBuffer, col: usize, value: u64, op: CompareOp) -> Result<CudaBuffer> { /* launch filter_compare_u64 */ }
pub fn filter_f32(&self, input: &CudaBuffer, col: usize, value: f32, op: CompareOp) -> Result<CudaBuffer> { /* launch filter_compare_f32 */ }
pub fn filter_bool(&self, input: &CudaBuffer, col: usize, value: bool, op: CompareOp) -> Result<CudaBuffer> { /* launch filter_compare_u8 */ }

pub fn compare_columns_u32(&self, input: &CudaBuffer, left: usize, right: usize, op: CompareOp) -> Result<cudarc::driver::CudaSlice<u8>> { /* launch filter_compare_u32_col */ }
// add compare_columns for i32/i64/u64/f32/f64/u8
```

Ensure `filter_by_device_mask` is public to reuse in runtime.

**Step 5: Run filter test**

Run: `cargo test -p xlog-cuda --test filter_tests test_filter_i32_u64_f32_bool_and_column_column_compare --release`
Expected: PASS.

**Step 6: Commit**

```bash
git add kernels/filter.cu crates/xlog-cuda/src/provider/mod.rs crates/xlog-cuda/tests/filter_tests.rs
git commit -m "feat(cuda): add typed filter compares and column-column masks"
```

---

### Task 5: GPU predicate evaluation in executor

**Files:**
- Modify: `crates/xlog-runtime/src/executor.rs`
- Test: `crates/xlog-runtime/tests/executor_config_tests.rs`

**Step 1: Add failing test for GPU predicate evaluation**

Append to `crates/xlog-runtime/tests/executor_config_tests.rs`:

```rust
#[test]
fn test_executor_filter_with_column_column_compare_and_symbol() {
    if cudarc::driver::CudaDevice::count().unwrap_or(0) == 0 {
        return;
    }

    let device = Arc::new(CudaDevice::new(0).unwrap());
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), MemoryBudget::with_limit(1 << 28)));
    let provider = Arc::new(CudaKernelProvider::new(device, memory).unwrap());
    let mut executor = Executor::new(provider.clone());

    let schema = Schema::new(vec![
        ("a".to_string(), ScalarType::U32),
        ("b".to_string(), ScalarType::U32),
        ("s".to_string(), ScalarType::Symbol),
    ]);

    let buf = provider
        .create_buffer_from_u32_columns(&[&[1, 2, 3], &[1, 9, 3], &[42, 7, 42]], schema)
        .unwrap();

    let predicate = Expr::And(vec![
        Expr::Compare {
            left: Box::new(Expr::Column(0)),
            op: CompareOp::Eq,
            right: Box::new(Expr::Column(1)),
        },
        Expr::Compare {
            left: Box::new(Expr::Column(2)),
            op: CompareOp::Eq,
            right: Box::new(Expr::Const(ConstValue::Symbol("sym".to_string()))),
        },
    ]);

    let filtered = executor.execute_filter(&buf, &predicate).unwrap();
    let vals = provider.download_column_u32(&filtered, 0).unwrap();
    assert_eq!(vals, vec![1, 3]);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-runtime --test executor_config_tests test_executor_filter_with_column_column_compare_and_symbol --release`
Expected: FAIL (GPU predicate path not implemented; symbol const comparison not supported on device).

**Step 3: Implement GPU predicate evaluation**

In `crates/xlog-runtime/src/executor.rs`, add helpers:

```rust
fn wrap_single_column(buffer: &CudaBuffer, col_idx: usize) -> Result<CudaBuffer> {
    let col = buffer.column(col_idx).ok_or_else(|| XlogError::Execution("Column not found".into()))?;
    let ty = buffer.schema().column_type(col_idx).ok_or_else(|| XlogError::Execution("Missing type".into()))?;
    let schema = Schema::new(vec![("expr".to_string(), ty)]);
    Ok(CudaBuffer::from_columns(vec![col.clone()], buffer.num_rows(), schema))
}

fn eval_expr_gpu(&self, expr: &Expr, input: &CudaBuffer) -> Result<CudaBuffer> { /* recursive; use provider.add/sub/mul/div/mod/min/max/abs/pow/cast */ }

fn eval_predicate_mask_gpu(&self, expr: &Expr, input: &CudaBuffer) -> Result<cudarc::driver::CudaSlice<u8>> { /* Compare uses provider.compare_columns_* or filter_* const, And/Or/Not use mask ops */ }
```

Then update `execute_filter` to:

```rust
let mask = self.eval_predicate_mask_gpu(predicate, input)?;
self.provider.filter_by_device_mask(input, &mask)
```

Use `ScalarType::Symbol` comparisons as u32 in provider (const symbol hashed via `xlog_core::hash_symbol_to_u32`).

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-runtime --test executor_config_tests test_executor_filter_with_column_column_compare_and_symbol --release`
Expected: PASS.

**Step 5: Commit**

```bash
git add crates/xlog-runtime/src/executor.rs crates/xlog-runtime/tests/executor_config_tests.rs
git commit -m "feat(runtime): execute filters on GPU via mask DAG"
```

---

### Task 6: GPU groupby IDs and key extraction

**Files:**
- Modify: `crates/xlog-cuda/src/provider/mod.rs`
- Modify: `kernels/groupby.cu`
- Modify: `kernels/pack.cu` (if needed)
- Test: `crates/xlog-cuda/tests/groupby_tests.rs`

**Step 1: Add failing test for GPU key extraction**

Append to `crates/xlog-cuda/tests/groupby_tests.rs`:

```rust
#[test]
fn test_groupby_multi_key_device_keys() {
    let Some(provider) = create_test_provider() else {
        return;
    };

    let schema = Schema::new(vec![
        ("k1".to_string(), ScalarType::U32),
        ("k2".to_string(), ScalarType::U32),
        ("v".to_string(), ScalarType::U32),
    ]);

    let buf = provider
        .create_buffer_from_u32_columns(&[&[1, 1, 2, 2], &[10, 10, 20, 20], &[5, 7, 11, 13]], schema)
        .unwrap();

    let result = provider.groupby_multi_agg(&buf, &[0, 1], &[(2, AggOp::Sum)]).unwrap();
    let k1 = provider.download_column_u32(&result, 0).unwrap();
    let k2 = provider.download_column_u32(&result, 1).unwrap();
    let sums = provider.download_column_u64(&result, 2).unwrap();

    assert_eq!(k1, vec![1, 2]);
    assert_eq!(k2, vec![10, 20]);
    assert_eq!(sums, vec![12, 24]);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-cuda --test groupby_tests test_groupby_multi_key_device_keys --release`
Expected: FAIL (device key extraction not implemented).

**Step 3: Implement GPU group IDs and key extraction**

In `crates/xlog-cuda/src/provider/mod.rs`:
- Replace CPU boundary downloads with device scans using `multiblock_scan_phase1` + `multiblock_scan_phase3`.
- Use `detect_group_boundaries` to produce device boundary mask.
- Compute group IDs on device using prefix sum of boundaries.
- Add a device kernel to write group start indices from `boundaries` and `prefix_sum`.
- Use `pack_keys` + `gather_packed_rows` + `unpack_column` to extract key columns on device.

Add kernel in `kernels/groupby.cu`:

```cuda
extern "C" __global__ void group_start_indices(
    const uint8_t* __restrict__ boundaries,
    const uint32_t* __restrict__ boundary_pos,
    uint32_t n,
    uint32_t* __restrict__ group_first_idx
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    if (boundaries[gid]) {
        uint32_t group = boundary_pos[gid];
        group_first_idx[group] = gid;
    }
}
```

Use `pack_kernels::PACK_KEYS` to pack key columns for the original input, then `gather_packed_rows` to gather packed group keys at `group_first_idx`, and `unpack_column` to materialize each key column in the result buffer.

**Step 4: Run groupby test**

Run: `cargo test -p xlog-cuda --test groupby_tests test_groupby_multi_key_device_keys --release`
Expected: PASS.

**Step 5: Commit**

```bash
git add crates/xlog-cuda/src/provider/mod.rs kernels/groupby.cu kernels/pack.cu crates/xlog-cuda/tests/groupby_tests.rs
git commit -m "feat(cuda): compute groupby ids and keys on device"
```

---

### Task 7: Create `xlog` CLI crate with deterministic execution

**Files:**
- Create: `crates/xlog-cli/Cargo.toml`
- Create: `crates/xlog-cli/src/main.rs`
- Modify: `Cargo.toml`
- Test: `crates/xlog-cli/tests/run_cli_tests.rs`

**Step 1: Write failing CLI test**

Create `crates/xlog-cli/tests/run_cli_tests.rs`:

```rust
use assert_cmd::Command;

#[test]
fn test_xlog_run_basic() {
    if cudarc::driver::CudaDevice::count().unwrap_or(0) == 0 {
        return;
    }

    let mut cmd = Command::cargo_bin("xlog").unwrap();
    cmd.args(["run", "examples/xlog/00-basics/01_tc_reachability.xlog"]);
    cmd.assert().success();
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-cli --test run_cli_tests --release`
Expected: FAIL (crate/binary not found).

**Step 3: Add crate + dependencies**

Create `crates/xlog-cli/Cargo.toml`:

```toml
[package]
name = "xlog-cli"
version.workspace = true
edition.workspace = true

[[bin]]
name = "xlog"
path = "src/main.rs"

[dependencies]
clap = { version = "4.5", features = ["derive"] }
arrow = { version = "53", default-features = false, features = ["ffi", "ipc", "csv", "prettyprint"] }
thiserror.workspace = true
xlog-core = { path = "../xlog-core" }
xlog-cuda = { path = "../xlog-cuda" }
xlog-logic = { path = "../xlog-logic" }
xlog-gpu = { path = "../xlog-gpu" }

[dev-dependencies]
assert_cmd = "2.0"
```

Add to workspace `Cargo.toml` members.

**Step 4: Implement deterministic CLI**

Create `crates/xlog-cli/src/main.rs`:

```rust
use clap::{Parser, Subcommand, ValueEnum};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::csv::WriterBuilder;
use arrow::util::pretty::pretty_format_batches;
use xlog_core::{MemoryBudget, Result, XlogError};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_gpu::logic::LogicProgram;

#[derive(Parser)]
#[command(author, version, about = "XLOG CLI")]
pub struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Run(RunArgs),
    Prob(ProbArgs),
}

#[derive(Parser)]
struct RunArgs {
    source: PathBuf,
    #[arg(long, default_value = "0")]
    device: usize,
    #[arg(long, default_value = "1024")]
    memory_mb: u64,
    #[arg(long)]
    input: Vec<String>,
    #[arg(long, value_enum, default_value = "pretty")]
    output: OutputFormat,
    #[arg(long)]
    output_dir: Option<PathBuf>,
}

#[derive(Parser)]
struct ProbArgs {
    source: PathBuf,
    #[arg(long, default_value = "0")]
    device: usize,
    #[arg(long, default_value = "1024")]
    memory_mb: u64,
    #[arg(long, value_enum, default_value = "exact_ddnnf")]
    prob_engine: ProbEngineCli,
    #[arg(long, default_value = "10000")]
    samples: usize,
    #[arg(long, default_value = "0")]
    seed: u64,
    #[arg(long, default_value = "0.95")]
    confidence: f64,
    #[arg(long, value_enum, default_value = "pretty")]
    output: OutputFormat,
    #[arg(long)]
    output_dir: Option<PathBuf>,
}

#[derive(Copy, Clone, ValueEnum)]
enum OutputFormat {
    Pretty,
    Csv,
    Arrow,
}

#[derive(Copy, Clone, ValueEnum)]
enum ProbEngineCli {
    ExactDdnnf,
    Mc,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => run_deterministic(args),
        Command::Prob(args) => run_probabilistic(args),
    }
}

fn make_provider(device: usize, memory_mb: u64) -> Result<Arc<CudaKernelProvider>> {
    let device = Arc::new(CudaDevice::new(device)?);
    let memory = Arc::new(GpuMemoryManager::new(
        device.clone(),
        MemoryBudget::with_limit(memory_mb * 1024 * 1024),
    ));
    Ok(Arc::new(CudaKernelProvider::new(device, memory)?))
}

fn parse_inputs(inputs: &[String]) -> Result<HashMap<String, PathBuf>> {
    let mut out = HashMap::new();
    for entry in inputs {
        let (name, path) = entry
            .split_once('=')
            .ok_or_else(|| XlogError::Execution(format!("Invalid --input '{}', expected rel=path", entry)))?;
        out.insert(name.to_string(), PathBuf::from(path));
    }
    Ok(out)
}

fn run_deterministic(args: RunArgs) -> Result<()> {
    let provider = make_provider(args.device, args.memory_mb)?;
    let source = std::fs::read_to_string(&args.source)
        .map_err(|e| XlogError::Execution(format!("Failed to read {}: {}", args.source.display(), e)))?;

    let program = LogicProgram::compile(&source)?;
    let mut inputs = HashMap::new();
    for (name, path) in parse_inputs(&args.input)? {
        let buf = provider.read_arrow_ipc_stream_file(&path)?;
        inputs.insert(name, buf);
    }

    let result = program.evaluate(provider.clone(), inputs)?;
    emit_logic_results(provider.as_ref(), &result.queries, args.output, args.output_dir.as_deref())
}

fn emit_logic_results(
    provider: &CudaKernelProvider,
    queries: &[xlog_gpu::logic::LogicQueryResult],
    format: OutputFormat,
    output_dir: Option<&Path>,
) -> Result<()> {
    for (i, q) in queries.iter().enumerate() {
        let batch = provider.to_arrow_record_batch(&q.buffer)?;
        match format {
            OutputFormat::Pretty => {
                let formatted = pretty_format_batches(&[batch])
                    .map_err(|e| XlogError::Execution(format!("Pretty print failed: {}", e)))?;
                println!("{}\n{}", q.relation_name, formatted);
            }
            OutputFormat::Csv => {
                let mut out = Vec::new();
                let mut writer = WriterBuilder::new().build(&mut out);
                writer.write(&batch).map_err(|e| XlogError::Execution(format!("CSV write failed: {}", e)))?;
                writer.finish().map_err(|e| XlogError::Execution(format!("CSV finish failed: {}", e)))?;
                println!("{}\n{}", q.relation_name, String::from_utf8_lossy(&out));
            }
            OutputFormat::Arrow => {
                let dir = output_dir.unwrap_or_else(|| Path::new("."));
                let path = dir.join(format!("query_{}.arrow", i));
                provider.write_arrow_ipc_stream_file(&q.buffer, &path)?;
                println!("wrote {}", path.display());
            }
        }
    }
    Ok(())
}
```

**Step 5: Run CLI test**

Run: `cargo test -p xlog-cli --test run_cli_tests --release`
Expected: PASS.

**Step 6: Commit**

```bash
git add Cargo.toml crates/xlog-cli/Cargo.toml crates/xlog-cli/src/main.rs crates/xlog-cli/tests/run_cli_tests.rs
git commit -m "feat(cli): add xlog run command"
```

---

### Task 8: Add probabilistic CLI support + tests

**Files:**
- Modify: `crates/xlog-cli/src/main.rs`
- Test: `crates/xlog-cli/tests/prob_cli_tests.rs`

**Step 1: Add failing test**

Create `crates/xlog-cli/tests/prob_cli_tests.rs`:

```rust
use assert_cmd::Command;

#[test]
fn test_xlog_prob_exact_and_mc() {
    if cudarc::driver::CudaDevice::count().unwrap_or(0) == 0 {
        return;
    }

    let mut cmd = Command::cargo_bin("xlog").unwrap();
    cmd.args(["prob", "examples/prob/01-wet-conditioning.xlog", "--prob-engine", "exact_ddnnf"]);
    cmd.assert().success();

    let mut cmd = Command::cargo_bin("xlog").unwrap();
    cmd.args([
        "prob",
        "examples/prob/04-nonmonotone-mc.xlog",
        "--prob-engine",
        "mc",
        "--samples",
        "1000",
        "--seed",
        "42",
    ]);
    cmd.assert().success();
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-cli --test prob_cli_tests --release`
Expected: FAIL (prob mode not implemented).

**Step 3: Implement `run_probabilistic` in CLI**

Add to `crates/xlog-cli/src/main.rs`:

```rust
use xlog_logic::parse_program;
use xlog_prob::exact::{ExactDdnnfProgram, GpuConfig};
use xlog_prob::mc::{McEvalConfig, McProgram};

fn run_probabilistic(args: ProbArgs) -> Result<()> {
    let source = std::fs::read_to_string(&args.source)
        .map_err(|e| XlogError::Execution(format!("Failed to read {}: {}", args.source.display(), e)))?;

    let config = GpuConfig {
        device_ordinal: args.device,
        memory_bytes: args.memory_mb * 1024 * 1024,
    };

    match args.prob_engine {
        ProbEngineCli::ExactDdnnf => {
            let prog = ExactDdnnfProgram::compile_source_with_gpu(&source, config)?;
            let result = prog.evaluate()?;
            emit_prob_exact(result, args.output, args.output_dir.as_deref())
        }
        ProbEngineCli::Mc => {
            let prog = McProgram::compile_source_with_gpu(&source, config)?;
            let cfg = McEvalConfig {
                samples: args.samples,
                seed: args.seed,
                confidence: args.confidence,
                ..Default::default()
            };
            let result = prog.evaluate(cfg)?;
            emit_prob_mc(result, args.output, args.output_dir.as_deref())
        }
    }
}
```

Add output helpers that build Arrow batches and print/emit:

```rust
fn emit_prob_exact(
    result: xlog_prob::exact::ExactResult,
    format: OutputFormat,
    output_dir: Option<&Path>,
) -> Result<()> {
    let mut atoms = Vec::new();
    let mut probs = Vec::new();
    let mut log_probs = Vec::new();
    for q in result.query_probs {
        atoms.push(q.atom.to_string());
        probs.push(q.prob);
        log_probs.push(q.log_prob);
    }
    let batch = arrow::record_batch::RecordBatch::try_from_iter(vec![
        ("atom", Arc::new(arrow::array::StringArray::from(atoms)) as Arc<dyn arrow::array::Array>),
        ("prob", Arc::new(arrow::array::Float64Array::from(probs)) as Arc<dyn arrow::array::Array>),
        ("log_prob", Arc::new(arrow::array::Float64Array::from(log_probs)) as Arc<dyn arrow::array::Array>),
    ]).map_err(|e| XlogError::Execution(format!("Failed to build prob batch: {}", e)))?;

    emit_batch("prob", &batch, format, output_dir)
}

fn emit_prob_mc(
    result: xlog_prob::mc::McResult,
    format: OutputFormat,
    output_dir: Option<&Path>,
) -> Result<()> {
    let mut atoms = Vec::new();
    let mut probs = Vec::new();
    let mut log_probs = Vec::new();
    let mut stderr = Vec::new();
    let mut ci_low = Vec::new();
    let mut ci_high = Vec::new();
    for q in result.query_estimates {
        atoms.push(q.atom.to_string());
        probs.push(q.prob);
        log_probs.push(q.log_prob);
        stderr.push(q.stderr);
        ci_low.push(q.ci_low);
        ci_high.push(q.ci_high);
    }
    let batch = arrow::record_batch::RecordBatch::try_from_iter(vec![
        ("atom", Arc::new(arrow::array::StringArray::from(atoms)) as Arc<dyn arrow::array::Array>),
        ("prob", Arc::new(arrow::array::Float64Array::from(probs)) as Arc<dyn arrow::array::Array>),
        ("log_prob", Arc::new(arrow::array::Float64Array::from(log_probs)) as Arc<dyn arrow::array::Array>),
        ("stderr", Arc::new(arrow::array::Float64Array::from(stderr)) as Arc<dyn arrow::array::Array>),
        ("ci_low", Arc::new(arrow::array::Float64Array::from(ci_low)) as Arc<dyn arrow::array::Array>),
        ("ci_high", Arc::new(arrow::array::Float64Array::from(ci_high)) as Arc<dyn arrow::array::Array>),
    ]).map_err(|e| XlogError::Execution(format!("Failed to build mc batch: {}", e)))?;

    emit_batch("prob", &batch, format, output_dir)
}

fn emit_batch(name: &str, batch: &arrow::record_batch::RecordBatch, format: OutputFormat, output_dir: Option<&Path>) -> Result<()> {
    match format {
        OutputFormat::Pretty => {
            let formatted = pretty_format_batches(&[batch.clone()])
                .map_err(|e| XlogError::Execution(format!("Pretty print failed: {}", e)))?;
            println!("{}\n{}", name, formatted);
        }
        OutputFormat::Csv => {
            let mut out = Vec::new();
            let mut writer = WriterBuilder::new().build(&mut out);
            writer.write(batch).map_err(|e| XlogError::Execution(format!("CSV write failed: {}", e)))?;
            writer.finish().map_err(|e| XlogError::Execution(format!("CSV finish failed: {}", e)))?;
            println!("{}\n{}", name, String::from_utf8_lossy(&out));
        }
        OutputFormat::Arrow => {
            let dir = output_dir.unwrap_or_else(|| Path::new("."));
            let path = dir.join(format!("{}_prob.arrow", name));
            let mut out = Vec::new();
            let mut writer = arrow::ipc::writer::StreamWriter::try_new(&mut out, &batch.schema())
                .map_err(|e| XlogError::Execution(format!("Arrow writer failed: {}", e)))?;
            writer.write(batch).map_err(|e| XlogError::Execution(format!("Arrow write failed: {}", e)))?;
            writer.finish().map_err(|e| XlogError::Execution(format!("Arrow finish failed: {}", e)))?;
            std::fs::write(&path, out).map_err(|e| XlogError::Execution(format!("Arrow write file failed: {}", e)))?;
            println!("wrote {}", path.display());
        }
    }
    Ok(())
}
```

**Step 4: Run prob CLI test**

Run: `cargo test -p xlog-cli --test prob_cli_tests --release`
Expected: PASS.

**Step 5: Commit**

```bash
git add crates/xlog-cli/src/main.rs crates/xlog-cli/tests/prob_cli_tests.rs
git commit -m "feat(cli): add probabilistic execution"
```

---

### Task 9: Update docs for CLI and GPU-resident execution

**Files:**
- Modify: `docs/ROADMAP.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `examples/README.md`

**Step 1: Update roadmap items**

In `docs/ROADMAP.md`, mark CLI/REPL as implemented for v0.3.x (or add `xlog` CLI as implemented in v0.2.x if aligning with release), and move filter/groupby GPU-resident work to implemented.

**Step 2: Update architecture notes**

In `docs/ARCHITECTURE.md`, update:
- Execution pipeline for Filter to use GPU mask DAG.
- Groupby to compute group IDs and keys on-device.
- Add CLI invocation section and Arrow IPC I/O.

**Step 3: Update examples README**

Add CLI run instructions to `examples/README.md`:

```markdown
xlog run examples/xlog/00-basics/01_tc_reachability.xlog
xlog run --input edge=data.arrow examples/xlog/00-basics/01_tc_reachability.xlog
```

**Step 4: Commit**

```bash
git add docs/ROADMAP.md docs/ARCHITECTURE.md examples/README.md
git commit -m "docs: document gpu-resident execution and cli"
```

---

## Final Verification

Run:

```bash
cargo test --workspace --all-targets --exclude pyxlog --release
cargo test -p xlog-cuda-tests --test certification_suite --release -- --nocapture
```

Expected: PASS.

---

## Execution Handoff

Plan complete and saved to `docs/plans/2026-01-15-gpu-exec-cli-plan.md`. Two execution options:

1. Subagent-Driven (this session) - I dispatch fresh subagent per task, review between tasks, fast iteration
2. Parallel Session (separate) - Open new session with executing-plans, batch execution with checkpoints

Which approach?
