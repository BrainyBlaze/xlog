// crates/xlog-cuda/tests/test_csm_env_dispatch.rs
//! Env-dispatch routing tests for the recorded hash-join CSM
//! (count-scan-materialize) sub-strategy.
//!
//! Asserts that the recorded hash-join dispatch:
//!   * routes through the CSM methods for `JoinType::Inner` and
//!     `JoinType::LeftOuter` (indexed and non-indexed) when CSM is
//!     enabled via env (`XLOG_USE_RECORDED_CSM` or umbrella
//!     `XLOG_USE_RECORDED_OPS`),
//!   * does NOT route through CSM for `JoinType::Semi` /
//!     `JoinType::Anti` (no CSM implementation),
//!   * does NOT route through CSM when CSM env is off (legacy
//!     recorded path or legacy non-recorded path), and
//!   * short-circuits before CSM when the upstream eligibility
//!     check (≤4 keys) fails.
//!
//! Each test mutates process-wide env vars; the file MUST run with
//! `--test-threads=1`. The full xlog-cuda gate command
//! (`cargo test -p xlog-cuda --tests --release -- --test-threads=1`)
//! already enforces this.

use std::sync::Arc;

use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::{CudaBuffer, CudaDevice, CudaKernelProvider, GpuMemoryManager, JoinType};

const ENV_OPS: &str = "XLOG_USE_RECORDED_OPS";
const ENV_HJ: &str = "XLOG_USE_RECORDED_HASH_JOIN";
const ENV_CSM: &str = "XLOG_USE_RECORDED_CSM";

/// RAII guard that clears the three env vars on construction
/// and on Drop, so each test starts and ends from a known state
/// regardless of the previous test's flow.
struct EnvGuard;
impl EnvGuard {
    fn new() -> Self {
        clear_env();
        Self
    }
}
impl Drop for EnvGuard {
    fn drop(&mut self) {
        clear_env();
    }
}

fn clear_env() {
    // SAFETY: tests run with `--test-threads=1`; no other thread
    // is concurrently reading the process environment.
    unsafe {
        std::env::remove_var(ENV_OPS);
        std::env::remove_var(ENV_HJ);
        std::env::remove_var(ENV_CSM);
    }
}

fn set_env(name: &str, value: &str) {
    // SAFETY: tests run with `--test-threads=1`; no other thread
    // is concurrently reading the process environment.
    unsafe {
        std::env::set_var(name, value);
    }
}

struct DiscardSink;
impl LoggingSink for DiscardSink {
    fn emit(&self, _record: LogRecord) -> Result<(), SinkError> {
        Ok(())
    }
}

struct TestCtx {
    device: Arc<CudaDevice>,
    memory: Arc<GpuMemoryManager>,
    provider: CudaKernelProvider,
}

fn build_ctx() -> Option<TestCtx> {
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
    let async_resource: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(
        AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool)),
    );
    let logging: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(LoggingResource::new(
        async_resource,
        Arc::new(DiscardSink) as Arc<dyn LoggingSink>,
    ));
    let budget: Box<dyn DeviceMemoryResource + Send + Sync> =
        Box::new(GlobalDeviceBudget::new(logging, 64 * 1024 * 1024));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(64 * 1024 * 1024),
        Arc::clone(&runtime),
    ));
    let provider = CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory))
        .expect("provider with_runtime");
    Some(TestCtx {
        device,
        memory,
        provider,
    })
}

/// Build a deterministic Left[U32, U32] (LROWS rows, key in [0, LKEYS))
/// and Right[U32, U32] (RROWS rows, key in [0, RKEYS)). Overlapping key
/// spaces guarantee at least some matched and some unmatched rows for
/// both Inner and LeftOuter join types.
fn build_overlap_buffers(ctx: &TestCtx) -> (CudaBuffer, CudaBuffer) {
    const LROWS: usize = 64;
    const RROWS: usize = 32;
    const LKEYS: u32 = 16;
    const RKEYS: u32 = 8;
    let schema = Schema::new(vec![
        ("k".to_string(), ScalarType::U32),
        ("v".to_string(), ScalarType::U32),
    ]);
    let device = ctx.device.inner();

    let mut lk = Vec::with_capacity(LROWS * 4);
    let mut lv = Vec::with_capacity(LROWS * 4);
    for i in 0..LROWS {
        lk.extend_from_slice(&((i as u32) % LKEYS).to_le_bytes());
        lv.extend_from_slice(&((i as u32) + 100_000).to_le_bytes());
    }
    let mut lk_b = ctx.memory.alloc::<u8>(LROWS * 4).expect("alloc lk");
    let mut lv_b = ctx.memory.alloc::<u8>(LROWS * 4).expect("alloc lv");
    device.htod_sync_copy_into(&lk, &mut lk_b).expect("htod lk");
    device.htod_sync_copy_into(&lv, &mut lv_b).expect("htod lv");
    let mut l_rows = ctx.memory.alloc::<u32>(1).expect("alloc l_rows");
    device
        .htod_sync_copy_into(&[LROWS as u32], &mut l_rows)
        .expect("htod l_rows");
    let left = CudaBuffer::from_columns(
        vec![lk_b.into(), lv_b.into()],
        LROWS as u64,
        l_rows,
        schema.clone(),
    );

    let mut rk = Vec::with_capacity(RROWS * 4);
    let mut rv = Vec::with_capacity(RROWS * 4);
    for j in 0..RROWS {
        rk.extend_from_slice(&((j as u32) % RKEYS).to_le_bytes());
        rv.extend_from_slice(&((j as u32) + 200_000).to_le_bytes());
    }
    let mut rk_b = ctx.memory.alloc::<u8>(RROWS * 4).expect("alloc rk");
    let mut rv_b = ctx.memory.alloc::<u8>(RROWS * 4).expect("alloc rv");
    device.htod_sync_copy_into(&rk, &mut rk_b).expect("htod rk");
    device.htod_sync_copy_into(&rv, &mut rv_b).expect("htod rv");
    let mut r_rows = ctx.memory.alloc::<u32>(1).expect("alloc r_rows");
    device
        .htod_sync_copy_into(&[RROWS as u32], &mut r_rows)
        .expect("htod r_rows");
    let right =
        CudaBuffer::from_columns(vec![rk_b.into(), rv_b.into()], RROWS as u64, r_rows, schema);

    (left, right)
}

#[test]
fn dispatch_routes_to_csm_for_inner_non_indexed_with_umbrella_env() {
    let _g = EnvGuard::new();
    let Some(ctx) = build_ctx() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    set_env(ENV_OPS, "1");
    let (left, right) = build_overlap_buffers(&ctx);
    let before = ctx.provider.csm_invocations();
    let result = ctx
        .provider
        .hash_join_v2_with_limit(&left, &right, &[0], &[0], JoinType::Inner, None)
        .expect("inner join via CSM dispatch");
    assert_eq!(
        ctx.provider.csm_invocations() - before,
        1,
        "CSM must be invoked exactly once for eligible Inner non-indexed"
    );
    assert!(
        result.num_rows() > 0,
        "Inner join with overlap must produce some rows"
    );
}

#[test]
fn dispatch_routes_to_csm_for_inner_indexed_with_umbrella_env() {
    let _g = EnvGuard::new();
    let Some(ctx) = build_ctx() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    set_env(ENV_OPS, "1");
    let (left, right) = build_overlap_buffers(&ctx);
    let index = ctx
        .provider
        .build_join_index_v2(&right, &[0])
        .expect("build_join_index_v2");
    let before = ctx.provider.csm_invocations();
    let result = ctx
        .provider
        .hash_join_v2_with_index(&left, &right, &[0], &[0], JoinType::Inner, &index, None)
        .expect("inner indexed join via CSM dispatch");
    assert_eq!(
        ctx.provider.csm_invocations() - before,
        1,
        "CSM must be invoked exactly once for eligible Inner indexed"
    );
    assert!(result.num_rows() > 0);
}

#[test]
fn dispatch_routes_to_csm_for_left_outer_non_indexed_with_umbrella_env() {
    let _g = EnvGuard::new();
    let Some(ctx) = build_ctx() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    set_env(ENV_OPS, "1");
    let (left, right) = build_overlap_buffers(&ctx);
    let left_rows = left.num_rows();
    let before = ctx.provider.csm_invocations();
    let result = ctx
        .provider
        .hash_join_v2_with_limit(&left, &right, &[0], &[0], JoinType::LeftOuter, None)
        .expect("left_outer join via CSM dispatch");
    assert_eq!(
        ctx.provider.csm_invocations() - before,
        1,
        "CSM must be invoked exactly once for eligible LeftOuter non-indexed"
    );
    assert!(
        result.num_rows() >= left_rows,
        "LeftOuter must keep every left row (matched or zero-padded)"
    );
}

#[test]
fn dispatch_routes_to_csm_for_left_outer_indexed_with_umbrella_env() {
    let _g = EnvGuard::new();
    let Some(ctx) = build_ctx() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    set_env(ENV_OPS, "1");
    let (left, right) = build_overlap_buffers(&ctx);
    let left_rows = left.num_rows();
    let index = ctx
        .provider
        .build_join_index_v2(&right, &[0])
        .expect("build_join_index_v2");
    let before = ctx.provider.csm_invocations();
    let result = ctx
        .provider
        .hash_join_v2_with_index(&left, &right, &[0], &[0], JoinType::LeftOuter, &index, None)
        .expect("left_outer indexed join via CSM dispatch");
    assert_eq!(
        ctx.provider.csm_invocations() - before,
        1,
        "CSM must be invoked exactly once for eligible LeftOuter indexed"
    );
    assert!(result.num_rows() >= left_rows);
}

#[test]
fn dispatch_does_not_route_to_csm_for_semi_or_anti_under_csm_env() {
    let _g = EnvGuard::new();
    let Some(ctx) = build_ctx() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    set_env(ENV_OPS, "1");
    let (left, right) = build_overlap_buffers(&ctx);
    let before = ctx.provider.csm_invocations();
    ctx.provider
        .hash_join_v2_with_limit(&left, &right, &[0], &[0], JoinType::Semi, None)
        .expect("semi join");
    ctx.provider
        .hash_join_v2_with_limit(&left, &right, &[0], &[0], JoinType::Anti, None)
        .expect("anti join");
    assert_eq!(
        ctx.provider.csm_invocations() - before,
        0,
        "Semi/Anti must never route through CSM"
    );

    let index = ctx
        .provider
        .build_join_index_v2(&right, &[0])
        .expect("build_join_index_v2");
    ctx.provider
        .hash_join_v2_with_index(&left, &right, &[0], &[0], JoinType::Semi, &index, None)
        .expect("indexed semi join");
    ctx.provider
        .hash_join_v2_with_index(&left, &right, &[0], &[0], JoinType::Anti, &index, None)
        .expect("indexed anti join");
    assert_eq!(
        ctx.provider.csm_invocations() - before,
        0,
        "Indexed Semi/Anti must never route through CSM"
    );
}

#[test]
fn dispatch_does_not_route_to_csm_when_only_hash_join_env_is_set() {
    let _g = EnvGuard::new();
    let Some(ctx) = build_ctx() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    // Recorded hash-join is on, but CSM specifically is not — the
    // dispatch must use the legacy recorded methods.
    set_env(ENV_HJ, "1");
    let (left, right) = build_overlap_buffers(&ctx);
    let before = ctx.provider.csm_invocations();
    let result = ctx
        .provider
        .hash_join_v2_with_limit(&left, &right, &[0], &[0], JoinType::Inner, None)
        .expect("inner join via legacy recorded path");
    assert_eq!(
        ctx.provider.csm_invocations() - before,
        0,
        "without CSM env, dispatch must use the legacy recorded path"
    );
    assert!(result.num_rows() > 0);
}

#[test]
fn dispatch_does_not_route_to_csm_when_no_recorded_env_is_set() {
    let _g = EnvGuard::new();
    let Some(ctx) = build_ctx() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let (left, right) = build_overlap_buffers(&ctx);
    let before = ctx.provider.csm_invocations();
    let result = ctx
        .provider
        .hash_join_v2_with_limit(&left, &right, &[0], &[0], JoinType::Inner, None)
        .expect("inner join via legacy non-recorded path");
    assert_eq!(
        ctx.provider.csm_invocations() - before,
        0,
        "with no recorded env, dispatch must use the legacy non-recorded path"
    );
    assert!(result.num_rows() > 0);
}

#[test]
fn dispatch_short_circuits_before_csm_for_more_than_four_keys() {
    // Eligibility constraint inherited from `pack_keys`: the recorded
    // path supports at most 4 key columns. With 5 keys, the public API
    // short-circuits to the legacy non-recorded path BEFORE the
    // recorded dispatch is reached, so CSM never runs even when its
    // env var is set.
    let _g = EnvGuard::new();
    let Some(ctx) = build_ctx() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    set_env(ENV_OPS, "1");
    let device = ctx.device.inner();

    const ROWS: usize = 16;
    let schema = Schema::new(vec![
        ("a".to_string(), ScalarType::U32),
        ("b".to_string(), ScalarType::U32),
        ("c".to_string(), ScalarType::U32),
        ("d".to_string(), ScalarType::U32),
        ("e".to_string(), ScalarType::U32),
    ]);

    let mut cols_a = Vec::new();
    let mut cols_b = Vec::new();
    for col in 0..5u32 {
        let mut data = Vec::with_capacity(ROWS * 4);
        for i in 0..ROWS {
            data.extend_from_slice(&((i as u32) + col * 100).to_le_bytes());
        }
        let mut buf_a = ctx.memory.alloc::<u8>(ROWS * 4).expect("alloc col left");
        let mut buf_b = ctx.memory.alloc::<u8>(ROWS * 4).expect("alloc col right");
        device
            .htod_sync_copy_into(&data, &mut buf_a)
            .expect("htod a");
        device
            .htod_sync_copy_into(&data, &mut buf_b)
            .expect("htod b");
        cols_a.push(buf_a.into());
        cols_b.push(buf_b.into());
    }
    let mut l_rows = ctx.memory.alloc::<u32>(1).expect("alloc l_rows");
    let mut r_rows = ctx.memory.alloc::<u32>(1).expect("alloc r_rows");
    device
        .htod_sync_copy_into(&[ROWS as u32], &mut l_rows)
        .expect("htod l_rows");
    device
        .htod_sync_copy_into(&[ROWS as u32], &mut r_rows)
        .expect("htod r_rows");
    let left = CudaBuffer::from_columns(cols_a, ROWS as u64, l_rows, schema.clone());
    let right = CudaBuffer::from_columns(cols_b, ROWS as u64, r_rows, schema);

    let before = ctx.provider.csm_invocations();
    let result = ctx
        .provider
        .hash_join_v2_with_limit(
            &left,
            &right,
            &[0, 1, 2, 3, 4],
            &[0, 1, 2, 3, 4],
            JoinType::Inner,
            None,
        )
        .expect("inner join with 5 keys via legacy path");
    assert_eq!(
        ctx.provider.csm_invocations() - before,
        0,
        ">4 keys must short-circuit before CSM"
    );
    assert!(result.num_rows() > 0);
}
