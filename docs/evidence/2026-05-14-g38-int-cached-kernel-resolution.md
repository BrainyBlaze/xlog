# G38 G_INT M_INT.6 Cached-Kernel Resolution

**Goal document:** `docs/plans/2026-05-14-supervisor-goal-038.md`
**Sub-goal:** G_INT
**Metric:** M_INT.6 cached-kernel resolution
**Branch:** `feat/w3-bundle-integration`
**Status:** PASS. The cached HG u32 triangle kernels are used by exactly one
production launch path.

## Gate Text

Goal-038 requires:

```text
M_INT.6 Cached-kernel resolution
wcoj_triangle_count_hg_cached_u32 either deleted OR used by exactly one production path
1 path OR deleted
```

S_INT.4 adds:

```text
grep production call sites; if zero -> delete kernel + provider entry + manifest
entry + tests; if non-zero -> identify single production path; if both G1 and
another use it -> unify.
```

## Search

Command:

```text
rg -n "wcoj_triangle_count_hg_cached_u32|count_hg_cached|hg_cached" \
  crates/xlog-cuda/src crates/xlog-cuda/kernels crates/xlog-cuda/tests \
  crates/xlog-cuda-tests crates/xlog-integration
```

Result:

```text
crates/xlog-cuda/kernels/wcoj.cu:780:extern "C" __global__ void wcoj_triangle_count_hg_cached_u32(
crates/xlog-cuda/kernels/wcoj.cu:894:extern "C" __global__ void wcoj_triangle_materialize_hg_cached_u32(
crates/xlog-cuda/src/provider/mod.rs:277:    pub const WCOJ_TRIANGLE_COUNT_HG_CACHED_U32: &str = "wcoj_triangle_count_hg_cached_u32";
crates/xlog-cuda/src/provider/mod.rs:278:    pub const WCOJ_TRIANGLE_MATERIALIZE_HG_CACHED_U32: &str =
crates/xlog-cuda/src/provider/mod.rs:279:        "wcoj_triangle_materialize_hg_cached_u32";
crates/xlog-cuda/src/provider/wcoj_metadata.rs:756:                .get_func(WCOJ_MODULE, wcoj_kernels::WCOJ_TRIANGLE_COUNT_HG_CACHED_U32)
crates/xlog-cuda/src/provider/wcoj_metadata.rs:759:                        "wcoj_triangle_count_hg_cached_u32 kernel not found".to_string(),
crates/xlog-cuda/src/provider/wcoj_metadata.rs:927:                    wcoj_kernels::WCOJ_TRIANGLE_MATERIALIZE_HG_CACHED_U32,
crates/xlog-cuda/src/provider/wcoj_metadata.rs:931:                        "wcoj_triangle_materialize_hg_cached_u32 kernel not found".to_string(),
crates/xlog-cuda/tests/test_w33_hg_source_audit.rs:82:    let materialize = extract_extern_c_global_body(&src, "wcoj_triangle_materialize_hg_cached_u32")
crates/xlog-cuda/tests/test_w33_hg_source_audit.rs:83:        .expect("wcoj_triangle_materialize_hg_cached_u32 must exist");
crates/xlog-cuda/src/kernel_manifest_data.rs:467:            "wcoj_triangle_count_hg_cached_u32",
crates/xlog-cuda/src/kernel_manifest_data.rs:468:            "wcoj_triangle_materialize_hg_cached_u32",
```

## Production Path

The single production launch path is:

```text
crates/xlog-cuda/src/provider/wcoj_metadata.rs:653
pub fn wcoj_triangle_hg_u32_with_plan_recorded(...)
```

Inside that function:

```text
crates/xlog-cuda/src/provider/wcoj_metadata.rs:756
.get_func(WCOJ_MODULE, wcoj_kernels::WCOJ_TRIANGLE_COUNT_HG_CACHED_U32)

crates/xlog-cuda/src/provider/wcoj_metadata.rs:927
wcoj_kernels::WCOJ_TRIANGLE_MATERIALIZE_HG_CACHED_U32
```

Higher-level callers route through the same provider path rather than launching
the cached kernels directly:

```text
crates/xlog-cuda/src/provider/wcoj_metadata.rs:342
pub fn wcoj_triangle_hg_u32_recorded(...)

crates/xlog-cuda/src/provider/wcoj.rs:307
pub fn wcoj_triangle_u32_recorded(...)

crates/xlog-runtime/src/executor/wcoj_dispatch.rs:1037
crates/xlog-runtime/src/executor/wcoj_dispatch.rs:1151
provider.wcoj_triangle_hg_u32_recorded(...)
```

The integration benches also call the provider-level triangle entry points; they
do not introduce another cached-kernel launch surface.

## Verdict

M_INT.6 is green by the `1 path` branch of the gate.

The cached HG u32 triangle kernels are not deleted because they are reachable
from exactly one production launch path:
`CudaKernelProvider::wcoj_triangle_hg_u32_with_plan_recorded`.
