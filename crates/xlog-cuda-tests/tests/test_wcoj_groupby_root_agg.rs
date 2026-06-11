//! D1 widening — aggregate-fused WCOJ: group-by-root sum/min/max over the
//! triangle shape (u32 value columns).
//!
//! Contract under test: `wcoj_triangle_groupby_root_agg_u32_recorded`
//! computes, for `q(X, agg(V)) :- e_xy(X,Y), e_yz(Y,Z), e_xz(X,Z)` with
//! `agg ∈ {sum, min, max}` and `V ∈ {Y, Z}`, the same (X, agg) row set as
//! the unfused production path (materialize triangles, then groupby agg) —
//! WITHOUT materializing the triangle rows. Both paths are checked against
//! a host brute-force oracle. Bag semantics: every (Y, Z) completion
//! contributes its value, exactly like the materialized projection.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use cudarc::driver::sys;

use xlog_core::{AggOp, MemoryBudget, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::{CudaBuffer, CudaColumn};
use xlog_cuda::wcoj_metadata::{WcojRootAggValue, WCOJ_HG_BLOCK_WORK_UNIT_DEFAULT};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

struct DiscardSink;
impl LoggingSink for DiscardSink {
    fn emit(&self, _record: LogRecord) -> Result<(), SinkError> {
        Ok(())
    }
}

struct Fixture {
    memory: Arc<GpuMemoryManager>,
    provider: Arc<CudaKernelProvider>,
    pool: Arc<StreamPool>,
}

fn make_fixture() -> Option<Fixture> {
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
        Box::new(GlobalDeviceBudget::new(logging, 512 * 1024 * 1024));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(512 * 1024 * 1024),
        runtime,
    ));
    let provider =
        Arc::new(CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory)).ok()?);
    Some(Fixture {
        memory,
        provider,
        pool,
    })
}

fn upload_binary_u32(memory: &Arc<GpuMemoryManager>, rows: &[(u32, u32)]) -> CudaBuffer {
    let n = rows.len() as u32;
    let mut col0 = memory
        .alloc::<u8>(rows.len() * std::mem::size_of::<u32>())
        .expect("alloc col0");
    let mut col1 = memory
        .alloc::<u8>(rows.len() * std::mem::size_of::<u32>())
        .expect("alloc col1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let col0_bytes: Vec<u8> = rows.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
    let col1_bytes: Vec<u8> = rows.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
    let device = memory.device().inner();
    device
        .htod_sync_copy_into(&col0_bytes, &mut col0)
        .expect("htod col0");
    device
        .htod_sync_copy_into(&col1_bytes, &mut col1)
        .expect("htod col1");
    device
        .htod_sync_copy_into(&[n], &mut d_num_rows)
        .expect("htod d_num_rows");
    let schema = Schema::new(vec![
        ("col0".to_string(), ScalarType::U32),
        ("col1".to_string(), ScalarType::U32),
    ]);
    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into()],
        n as u64,
        d_num_rows,
        schema,
        n,
    )
}

fn buffer_rows(memory: &Arc<GpuMemoryManager>, buffer: &CudaBuffer) -> usize {
    if let Some(n) = buffer.cached_row_count() {
        return n as usize;
    }
    let mut host = [0u32; 1];
    memory
        .device()
        .inner()
        .dtoh_sync_copy_into(buffer.num_rows_device(), &mut host)
        .expect("dtoh num_rows");
    host[0] as usize
}

/// Raw partial-column download: groupby outputs allocate columns at
/// capacity, so copies must use the logical byte length, not the
/// allocation length.
fn download_column_bytes(
    memory: &Arc<GpuMemoryManager>,
    buffer: &CudaBuffer,
    col: usize,
    elem_size: usize,
) -> Vec<u8> {
    let n = buffer_rows(memory, buffer);
    let mut bytes = vec![0u8; n * elem_size];
    if n == 0 {
        return bytes;
    }
    let CudaColumn::Owned(c) = buffer.column(col).expect("column") else {
        panic!("column must be owned");
    };
    unsafe {
        let res = sys::cuMemcpyDtoH_v2(bytes.as_mut_ptr() as *mut _, *c.device_ptr(), bytes.len());
        assert_eq!(res, sys::cudaError_enum::CUDA_SUCCESS, "dtoh column copy");
    }
    bytes
}

fn download_u32_column(memory: &Arc<GpuMemoryManager>, buffer: &CudaBuffer, col: usize) -> Vec<u32> {
    download_column_bytes(memory, buffer, col, 4)
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn download_u64_column(memory: &Arc<GpuMemoryManager>, buffer: &CudaBuffer, col: usize) -> Vec<u64> {
    download_column_bytes(memory, buffer, col, 8)
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Sorted (X, agg) pairs from a 2-column (U32 key, U64 agg) buffer.
fn download_groups_u64(memory: &Arc<GpuMemoryManager>, buffer: &CudaBuffer) -> Vec<(u32, u64)> {
    let keys = download_u32_column(memory, buffer, 0);
    let aggs = download_u64_column(memory, buffer, 1);
    assert_eq!(keys.len(), aggs.len());
    let mut out: Vec<(u32, u64)> = keys.into_iter().zip(aggs).collect();
    out.sort();
    out
}

/// Sorted (X, agg) pairs from a 2-column (U32 key, U32 agg) buffer.
fn download_groups_u32(memory: &Arc<GpuMemoryManager>, buffer: &CudaBuffer) -> Vec<(u32, u32)> {
    let keys = download_u32_column(memory, buffer, 0);
    let aggs = download_u32_column(memory, buffer, 1);
    assert_eq!(keys.len(), aggs.len());
    let mut out: Vec<(u32, u32)> = keys.into_iter().zip(aggs).collect();
    out.sort();
    out
}

/// All (x, y, z) triangle completions (bag of join rows; inputs deduped).
fn oracle_triangles(
    e_xy: &[(u32, u32)],
    e_yz: &[(u32, u32)],
    e_xz: &[(u32, u32)],
) -> Vec<(u32, u32, u32)> {
    let xz_set: BTreeSet<(u32, u32)> = e_xz.iter().copied().collect();
    let mut yz_by_y: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for (y, z) in e_yz {
        yz_by_y.entry(*y).or_default().push(*z);
    }
    let mut out = Vec::new();
    for (x, y) in e_xy {
        if let Some(zs) = yz_by_y.get(y) {
            for z in zs {
                if xz_set.contains(&(*x, *z)) {
                    out.push((*x, *y, *z));
                }
            }
        }
    }
    out
}

fn oracle_value(triple: (u32, u32, u32), value: WcojRootAggValue) -> u32 {
    match value {
        WcojRootAggValue::Y => triple.1,
        WcojRootAggValue::Z => triple.2,
    }
}

fn oracle_sums(
    e_xy: &[(u32, u32)],
    e_yz: &[(u32, u32)],
    e_xz: &[(u32, u32)],
    value: WcojRootAggValue,
) -> Vec<(u32, u64)> {
    let mut sums: BTreeMap<u32, u64> = BTreeMap::new();
    for t in oracle_triangles(e_xy, e_yz, e_xz) {
        *sums.entry(t.0).or_default() += u64::from(oracle_value(t, value));
    }
    sums.into_iter().collect()
}

fn oracle_mins(
    e_xy: &[(u32, u32)],
    e_yz: &[(u32, u32)],
    e_xz: &[(u32, u32)],
    value: WcojRootAggValue,
) -> Vec<(u32, u32)> {
    let mut mins: BTreeMap<u32, u32> = BTreeMap::new();
    for t in oracle_triangles(e_xy, e_yz, e_xz) {
        let v = oracle_value(t, value);
        mins.entry(t.0)
            .and_modify(|m| *m = (*m).min(v))
            .or_insert(v);
    }
    mins.into_iter().collect()
}

fn oracle_maxs(
    e_xy: &[(u32, u32)],
    e_yz: &[(u32, u32)],
    e_xz: &[(u32, u32)],
    value: WcojRootAggValue,
) -> Vec<(u32, u32)> {
    let mut maxs: BTreeMap<u32, u32> = BTreeMap::new();
    for t in oracle_triangles(e_xy, e_yz, e_xz) {
        let v = oracle_value(t, value);
        maxs.entry(t.0)
            .and_modify(|m| *m = (*m).max(v))
            .or_insert(v);
    }
    maxs.into_iter().collect()
}

fn sorted_unique(rows: impl IntoIterator<Item = (u32, u32)>) -> Vec<(u32, u32)> {
    let set: BTreeSet<(u32, u32)> = rows.into_iter().collect();
    set.into_iter().collect()
}

/// Triangle output column carrying the aggregate value: (X, Y, Z) = (0, 1, 2).
fn value_col(value: WcojRootAggValue) -> usize {
    match value {
        WcojRootAggValue::Y => 1,
        WcojRootAggValue::Z => 2,
    }
}

/// Unfused production baseline: materialize triangles, then groupby agg.
fn baseline_buffer(
    fix: &Fixture,
    e_xy: &CudaBuffer,
    e_yz: &CudaBuffer,
    e_xz: &CudaBuffer,
    agg: AggOp,
    value: WcojRootAggValue,
) -> CudaBuffer {
    let stream = fix.pool.acquire().expect("stream");
    let tri = fix
        .provider
        .wcoj_triangle_hg_u32_recorded(e_xy, e_yz, e_xz, WCOJ_HG_BLOCK_WORK_UNIT_DEFAULT, stream)
        .expect("baseline triangle materialize");
    fix.provider
        .groupby_multi_agg(&tri, &[0], &[(value_col(value), agg)])
        .expect("baseline groupby agg")
}

fn fused_buffer(
    fix: &Fixture,
    e_xy: &CudaBuffer,
    e_yz: &CudaBuffer,
    e_xz: &CudaBuffer,
    agg: AggOp,
    value: WcojRootAggValue,
) -> CudaBuffer {
    let stream = fix.pool.acquire().expect("stream");
    fix.provider
        .wcoj_triangle_groupby_root_agg_u32_recorded(
            e_xy,
            e_yz,
            e_xz,
            agg,
            value,
            WCOJ_HG_BLOCK_WORK_UNIT_DEFAULT,
            stream,
        )
        .expect("fused groupby-root agg")
}

fn run_case(
    name: &str,
    e_xy_rows: &[(u32, u32)],
    e_yz_rows: &[(u32, u32)],
    e_xz_rows: &[(u32, u32)],
) {
    let Some(fix) = make_fixture() else {
        eprintln!("skipping {name}: no CUDA device");
        return;
    };
    assert!(
        !oracle_triangles(e_xy_rows, e_yz_rows, e_xz_rows).is_empty(),
        "{name}: fixture must contain at least one triangle"
    );
    let e_xy = upload_binary_u32(&fix.memory, e_xy_rows);
    let e_yz = upload_binary_u32(&fix.memory, e_yz_rows);
    let e_xz = upload_binary_u32(&fix.memory, e_xz_rows);

    for value in [WcojRootAggValue::Y, WcojRootAggValue::Z] {
        // Sum (U64 output).
        let expected = oracle_sums(e_xy_rows, e_yz_rows, e_xz_rows, value);
        let baseline = baseline_buffer(&fix, &e_xy, &e_yz, &e_xz, AggOp::Sum, value);
        assert_eq!(
            download_groups_u64(&fix.memory, &baseline),
            expected,
            "{name}/{value:?}: unfused sum baseline vs oracle"
        );
        let fused = fused_buffer(&fix, &e_xy, &e_yz, &e_xz, AggOp::Sum, value);
        assert_eq!(
            download_groups_u64(&fix.memory, &fused),
            expected,
            "{name}/{value:?}: fused sum vs oracle"
        );

        // Min (U32 output).
        let expected = oracle_mins(e_xy_rows, e_yz_rows, e_xz_rows, value);
        let baseline = baseline_buffer(&fix, &e_xy, &e_yz, &e_xz, AggOp::Min, value);
        assert_eq!(
            download_groups_u32(&fix.memory, &baseline),
            expected,
            "{name}/{value:?}: unfused min baseline vs oracle"
        );
        let fused = fused_buffer(&fix, &e_xy, &e_yz, &e_xz, AggOp::Min, value);
        assert_eq!(
            download_groups_u32(&fix.memory, &fused),
            expected,
            "{name}/{value:?}: fused min vs oracle"
        );

        // Max (U32 output).
        let expected = oracle_maxs(e_xy_rows, e_yz_rows, e_xz_rows, value);
        let baseline = baseline_buffer(&fix, &e_xy, &e_yz, &e_xz, AggOp::Max, value);
        assert_eq!(
            download_groups_u32(&fix.memory, &baseline),
            expected,
            "{name}/{value:?}: unfused max baseline vs oracle"
        );
        let fused = fused_buffer(&fix, &e_xy, &e_yz, &e_xz, AggOp::Max, value);
        assert_eq!(
            download_groups_u32(&fix.memory, &fused),
            expected,
            "{name}/{value:?}: fused max vs oracle"
        );
    }
}

#[test]
fn groupby_root_agg_matches_oracle_small() {
    // K4 on {1..4} plus a disjoint triangle {5,6,7}; multiple completions
    // per root so sum != min != max.
    let e_xy = sorted_unique([
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
        (5, 6),
        (5, 7),
        (6, 7),
    ]);
    let e_yz = sorted_unique([(2, 3), (2, 4), (3, 4), (6, 7)]);
    let e_xz = sorted_unique([(1, 3), (1, 4), (2, 4), (3, 4), (5, 7)]);
    run_case("small", &e_xy, &e_yz, &e_xz);
}

#[test]
fn groupby_root_agg_matches_oracle_skewed_hub() {
    // Super-hub: X=0 connects to 512 Y values; Y values chain to a shared
    // Z band; plus a uniform background. Exercises heavy per-root fanout
    // (many work units feeding the same root accumulator).
    let mut e_xy: Vec<(u32, u32)> = Vec::new();
    let mut e_yz: Vec<(u32, u32)> = Vec::new();
    let mut e_xz: Vec<(u32, u32)> = Vec::new();
    for y in 1..=512u32 {
        e_xy.push((0, y));
        for z in 1000..1016u32 {
            e_yz.push((y, z));
        }
    }
    for z in 1000..1016u32 {
        e_xz.push((0, z));
    }
    for i in 0..200u32 {
        let (a, b, c) = (2000 + i, 3000 + i, 4000 + i);
        e_xy.push((a, b));
        e_yz.push((b, c));
        e_xz.push((a, c));
    }
    let e_xy = sorted_unique(e_xy);
    let e_yz = sorted_unique(e_yz);
    let e_xz = sorted_unique(e_xz);
    run_case("skewed_hub", &e_xy, &e_yz, &e_xz);
}

#[test]
fn groupby_root_agg_bag_semantics_duplicate_projected_values() {
    // X=1 completes triangles (1,2,9) and (1,3,9): the projected (X, Z)
    // bag is {(1,9), (1,9)} — sum(Z) must count Z=9 twice (bag, not set).
    let e_xy = sorted_unique([(1, 2), (1, 3)]);
    let e_yz = sorted_unique([(2, 9), (3, 9)]);
    let e_xz = sorted_unique([(1, 9)]);

    let Some(fix) = make_fixture() else {
        eprintln!("skipping bag_semantics: no CUDA device");
        return;
    };
    let e_xy_b = upload_binary_u32(&fix.memory, &e_xy);
    let e_yz_b = upload_binary_u32(&fix.memory, &e_yz);
    let e_xz_b = upload_binary_u32(&fix.memory, &e_xz);
    let fused = fused_buffer(
        &fix,
        &e_xy_b,
        &e_yz_b,
        &e_xz_b,
        AggOp::Sum,
        WcojRootAggValue::Z,
    );
    assert_eq!(
        download_groups_u64(&fix.memory, &fused),
        vec![(1u32, 18u64)],
        "sum(Z) over the bag of completions: 9 + 9"
    );
}

#[test]
fn groupby_root_agg_empty_intersection_roots_are_absent() {
    // X=9 has e_xy edges but no completing (Y,Z): it must NOT appear in
    // the fused output, even though min-init (u32::MAX) / sum-init (0)
    // rows exist in the per-row accumulator.
    let e_xy = sorted_unique([(1, 2), (9, 2), (9, 3)]);
    let e_yz = sorted_unique([(2, 3)]);
    let e_xz = sorted_unique([(1, 3)]);

    let Some(fix) = make_fixture() else {
        eprintln!("skipping empty_intersection: no CUDA device");
        return;
    };
    let e_xy_b = upload_binary_u32(&fix.memory, &e_xy);
    let e_yz_b = upload_binary_u32(&fix.memory, &e_yz);
    let e_xz_b = upload_binary_u32(&fix.memory, &e_xz);
    let fused = fused_buffer(
        &fix,
        &e_xy_b,
        &e_yz_b,
        &e_xz_b,
        AggOp::Min,
        WcojRootAggValue::Z,
    );
    assert_eq!(
        download_groups_u32(&fix.memory, &fused),
        vec![(1u32, 3u32)],
        "only X=1 completes a triangle"
    );
}

#[test]
fn groupby_root_agg_sum_zero_valued_groups_are_present() {
    // V=Z with Z=0: the group exists with sum 0 — "no completion" (absent)
    // must stay distinguishable from "sum of zeros" (present, 0).
    let e_xy = sorted_unique([(1, 2)]);
    let e_yz = sorted_unique([(2, 0)]);
    let e_xz = sorted_unique([(1, 0)]);

    let Some(fix) = make_fixture() else {
        eprintln!("skipping zero_valued: no CUDA device");
        return;
    };
    let e_xy_b = upload_binary_u32(&fix.memory, &e_xy);
    let e_yz_b = upload_binary_u32(&fix.memory, &e_yz);
    let e_xz_b = upload_binary_u32(&fix.memory, &e_xz);
    let fused = fused_buffer(
        &fix,
        &e_xy_b,
        &e_yz_b,
        &e_xz_b,
        AggOp::Sum,
        WcojRootAggValue::Z,
    );
    assert_eq!(
        download_groups_u64(&fix.memory, &fused),
        vec![(1u32, 0u64)],
        "X=1 has one completion with value 0"
    );
}

/// S1b measurement (agg-widening gate: fused >= 3x vs unfused on skewed
/// fixtures). Run explicitly:
/// `cargo test -p xlog-cuda-tests --test test_wcoj_groupby_root_agg \
///    --release -- --ignored --nocapture`
/// Asserts parity; timing ratios are PRINTED and recorded as evidence, not
/// asserted (wall-clock assertions are machine-dependent).
#[test]
#[ignore = "S1b measurement: run explicitly with --ignored --nocapture"]
fn s1b_measurement_agg_fused_vs_unfused() {
    let Some(fix) = make_fixture() else {
        eprintln!("skipping s1b_measurement: no CUDA device");
        return;
    };

    let hub = |n_y: u32, n_z: u32| {
        let mut e_xy = Vec::new();
        let mut e_yz = Vec::new();
        let mut e_xz = Vec::new();
        for y in 1..=n_y {
            e_xy.push((0u32, y));
            for z in 0..n_z {
                e_yz.push((y, 1_000_000 + z));
            }
        }
        for z in 0..n_z {
            e_xz.push((0u32, 1_000_000 + z));
        }
        // Uniform background so the group column is not a single value.
        for i in 0..1000u32 {
            let (a, b, c) = (2_000_000 + i, 3_000_000 + i, 4_000_000 + i);
            e_xy.push((a, b));
            e_yz.push((b, c));
            e_xz.push((a, c));
        }
        (sorted_unique(e_xy), sorted_unique(e_yz), sorted_unique(e_xz))
    };

    let cases: Vec<(&str, AggOp, (Vec<(u32, u32)>, Vec<(u32, u32)>, Vec<(u32, u32)>))> = vec![
        ("sum_z_hub_10k_z16", AggOp::Sum, hub(10_000, 16)),
        ("sum_z_hub_50k_z16", AggOp::Sum, hub(50_000, 16)),
        ("min_z_hub_10k_z16", AggOp::Min, hub(10_000, 16)),
        ("min_z_hub_50k_z16", AggOp::Min, hub(50_000, 16)),
        ("max_z_hub_10k_z16", AggOp::Max, hub(10_000, 16)),
        ("max_z_hub_50k_z16", AggOp::Max, hub(50_000, 16)),
    ];

    const REPS: usize = 5;
    let value = WcojRootAggValue::Z;
    for (name, agg, (e_xy_rows, e_yz_rows, e_xz_rows)) in &cases {
        let e_xy = upload_binary_u32(&fix.memory, e_xy_rows);
        let e_yz = upload_binary_u32(&fix.memory, e_yz_rows);
        let e_xz = upload_binary_u32(&fix.memory, e_xz_rows);

        // One stream per case, reused across reps (StreamPool is grow-only).
        let stream = fix.pool.acquire().expect("stream");
        // Warmup both paths once and assert parity vs the host oracle.
        let _ = baseline_buffer(&fix, &e_xy, &e_yz, &e_xz, *agg, value);
        let warm = fix
            .provider
            .wcoj_triangle_groupby_root_agg_u32_recorded(
                &e_xy,
                &e_yz,
                &e_xz,
                *agg,
                value,
                WCOJ_HG_BLOCK_WORK_UNIT_DEFAULT,
                stream,
            )
            .expect("fused warmup");
        match agg {
            AggOp::Sum => assert_eq!(
                download_groups_u64(&fix.memory, &warm),
                oracle_sums(e_xy_rows, e_yz_rows, e_xz_rows, value),
                "{name}: fused parity"
            ),
            AggOp::Min => assert_eq!(
                download_groups_u32(&fix.memory, &warm),
                oracle_mins(e_xy_rows, e_yz_rows, e_xz_rows, value),
                "{name}: fused parity"
            ),
            AggOp::Max => assert_eq!(
                download_groups_u32(&fix.memory, &warm),
                oracle_maxs(e_xy_rows, e_yz_rows, e_xz_rows, value),
                "{name}: fused parity"
            ),
            _ => unreachable!(),
        }
        drop(warm);

        let mut unfused_ms = Vec::with_capacity(REPS);
        let mut fused_ms = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let t = std::time::Instant::now();
            let tri = fix
                .provider
                .wcoj_triangle_hg_u32_recorded(
                    &e_xy,
                    &e_yz,
                    &e_xz,
                    WCOJ_HG_BLOCK_WORK_UNIT_DEFAULT,
                    stream,
                )
                .expect("baseline triangle");
            let grouped = fix
                .provider
                .groupby_multi_agg(&tri, &[0], &[(value_col(value), *agg)])
                .expect("baseline groupby");
            fix.provider.device().inner().synchronize().expect("sync");
            unfused_ms.push(t.elapsed().as_secs_f64() * 1e3);
            drop(grouped);

            let t = std::time::Instant::now();
            let fused = fix
                .provider
                .wcoj_triangle_groupby_root_agg_u32_recorded(
                    &e_xy,
                    &e_yz,
                    &e_xz,
                    *agg,
                    value,
                    WCOJ_HG_BLOCK_WORK_UNIT_DEFAULT,
                    stream,
                )
                .expect("fused");
            fix.provider.device().inner().synchronize().expect("sync");
            fused_ms.push(t.elapsed().as_secs_f64() * 1e3);
            drop(fused);
        }
        unfused_ms.sort_by(|a, b| a.total_cmp(b));
        fused_ms.sort_by(|a, b| a.total_cmp(b));
        let med_unfused = unfused_ms[REPS / 2];
        let med_fused = fused_ms[REPS / 2];
        println!(
            "S1b {name}: unfused median {med_unfused:.3} ms, fused median {med_fused:.3} ms, \
             speedup {:.2}x (n_xy={}, n_yz={}, n_xz={})",
            med_unfused / med_fused,
            e_xy_rows.len(),
            e_yz_rows.len(),
            e_xz_rows.len()
        );
    }
}

/// The recorded groupby Sum extension consumed by the fused sum path:
/// U64 value columns must reduce with u64 accumulation.
#[test]
fn groupby_recorded_sum_u64_values_matches_host() {
    let Some(fix) = make_fixture() else {
        eprintln!("skipping recorded_sum_u64: no CUDA device");
        return;
    };
    // (key u32, value u64) with values above u32::MAX to prove 64-bit reads.
    let rows: Vec<(u32, u64)> = vec![
        (1, 5_000_000_000),
        (1, 7),
        (2, 1),
        (2, u32::MAX as u64 + 1),
        (3, 0),
    ];
    let n = rows.len() as u32;
    let mut col0 = fix.memory.alloc::<u8>(rows.len() * 4).expect("alloc col0");
    let mut col1 = fix.memory.alloc::<u8>(rows.len() * 8).expect("alloc col1");
    let mut d_num_rows = fix.memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let col0_bytes: Vec<u8> = rows.iter().flat_map(|(k, _)| k.to_le_bytes()).collect();
    let col1_bytes: Vec<u8> = rows.iter().flat_map(|(_, v)| v.to_le_bytes()).collect();
    let device = fix.memory.device().inner();
    device
        .htod_sync_copy_into(&col0_bytes, &mut col0)
        .expect("htod col0");
    device
        .htod_sync_copy_into(&col1_bytes, &mut col1)
        .expect("htod col1");
    device
        .htod_sync_copy_into(&[n], &mut d_num_rows)
        .expect("htod d_num_rows");
    let schema = Schema::new(vec![
        ("k".to_string(), ScalarType::U32),
        ("v".to_string(), ScalarType::U64),
    ]);
    let buffer = CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into()],
        n as u64,
        d_num_rows,
        schema,
        n,
    );
    let stream = fix.pool.acquire().expect("stream");
    let grouped = fix
        .provider
        .groupby_multi_agg_recorded(&buffer, &[0], &[(1, AggOp::Sum)], stream)
        .expect("recorded groupby sum over u64 values");
    assert_eq!(
        download_groups_u64(&fix.memory, &grouped),
        vec![
            (1u32, 5_000_000_007u64),
            (2, u32::MAX as u64 + 2),
            (3, 0)
        ],
    );
}

// =====================================================================
// S1c widening — u64-key sum/min/max.
// =====================================================================

fn upload_binary_u64(memory: &Arc<GpuMemoryManager>, rows: &[(u64, u64)]) -> CudaBuffer {
    let n = rows.len() as u32;
    let mut col0 = memory
        .alloc::<u8>(rows.len() * std::mem::size_of::<u64>())
        .expect("alloc col0");
    let mut col1 = memory
        .alloc::<u8>(rows.len() * std::mem::size_of::<u64>())
        .expect("alloc col1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let col0_bytes: Vec<u8> = rows.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
    let col1_bytes: Vec<u8> = rows.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
    let device = memory.device().inner();
    device
        .htod_sync_copy_into(&col0_bytes, &mut col0)
        .expect("htod col0");
    device
        .htod_sync_copy_into(&col1_bytes, &mut col1)
        .expect("htod col1");
    device
        .htod_sync_copy_into(&[n], &mut d_num_rows)
        .expect("htod d_num_rows");
    let schema = Schema::new(vec![
        ("col0".to_string(), ScalarType::U64),
        ("col1".to_string(), ScalarType::U64),
    ]);
    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into()],
        n as u64,
        d_num_rows,
        schema,
        n,
    )
}

/// Sorted (X, agg) pairs from a 2-column (U64 key, U64 agg) buffer.
fn download_groups_u64_u64(
    memory: &Arc<GpuMemoryManager>,
    buffer: &CudaBuffer,
) -> Vec<(u64, u64)> {
    let keys = download_u64_column(memory, buffer, 0);
    let aggs = download_u64_column(memory, buffer, 1);
    assert_eq!(keys.len(), aggs.len());
    let mut out: Vec<(u64, u64)> = keys.into_iter().zip(aggs).collect();
    out.sort();
    out
}

/// The unfused u64-key baseline (materialize + legacy groupby) must reduce
/// U64 value columns: the legacy groupby is the path u64-key buffers take
/// (the recorded groupby is U32/Symbol-key only).
#[test]
fn groupby_legacy_agg_u64_values_matches_host() {
    let Some(fix) = make_fixture() else {
        eprintln!("skipping legacy_agg_u64: no CUDA device");
        return;
    };
    // (key u64 above 2^40, value u64 above u32::MAX) so any width
    // truncation visibly fails.
    const B: u64 = 1 << 40;
    let rows: Vec<(u64, u64)> = vec![
        (B + 1, 5_000_000_000),
        (B + 1, 7),
        (B + 2, 1),
        (B + 2, u32::MAX as u64 + 1),
        (B + 3, 0),
    ];
    let buffer = upload_binary_u64(&fix.memory, &rows);
    let grouped = fix
        .provider
        .groupby_multi_agg(&buffer, &[0], &[(1, AggOp::Sum)])
        .expect("legacy groupby sum over u64 values");
    assert_eq!(
        download_groups_u64_u64(&fix.memory, &grouped),
        vec![
            (B + 1, 5_000_000_007u64),
            (B + 2, u32::MAX as u64 + 2),
            (B + 3, 0)
        ],
        "legacy u64 sum"
    );
    let buffer = upload_binary_u64(&fix.memory, &rows);
    let grouped = fix
        .provider
        .groupby_multi_agg(&buffer, &[0], &[(1, AggOp::Min)])
        .expect("legacy groupby min over u64 values");
    assert_eq!(
        download_groups_u64_u64(&fix.memory, &grouped),
        vec![(B + 1, 7u64), (B + 2, 1), (B + 3, 0)],
        "legacy u64 min"
    );
    let buffer = upload_binary_u64(&fix.memory, &rows);
    let grouped = fix
        .provider
        .groupby_multi_agg(&buffer, &[0], &[(1, AggOp::Max)])
        .expect("legacy groupby max over u64 values");
    assert_eq!(
        download_groups_u64_u64(&fix.memory, &grouped),
        vec![
            (B + 1, 5_000_000_000u64),
            (B + 2, u32::MAX as u64 + 1),
            (B + 3, 0)
        ],
        "legacy u64 max"
    );
}

/// All (x, y, z) triangle completions over u64 relations.
fn oracle_triangles_u64(
    e_xy: &[(u64, u64)],
    e_yz: &[(u64, u64)],
    e_xz: &[(u64, u64)],
) -> Vec<(u64, u64, u64)> {
    let xz_set: BTreeSet<(u64, u64)> = e_xz.iter().copied().collect();
    let mut yz_by_y: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    for (y, z) in e_yz {
        yz_by_y.entry(*y).or_default().push(*z);
    }
    let mut out = Vec::new();
    for (x, y) in e_xy {
        if let Some(zs) = yz_by_y.get(y) {
            for z in zs {
                if xz_set.contains(&(*x, *z)) {
                    out.push((*x, *y, *z));
                }
            }
        }
    }
    out
}

fn oracle_value_u64(triple: (u64, u64, u64), value: WcojRootAggValue) -> u64 {
    match value {
        WcojRootAggValue::Y => triple.1,
        WcojRootAggValue::Z => triple.2,
    }
}

/// Host oracle for one u64 aggregate (sum wraps like the u64 accumulator).
fn oracle_agg_u64(
    e_xy: &[(u64, u64)],
    e_yz: &[(u64, u64)],
    e_xz: &[(u64, u64)],
    agg: AggOp,
    value: WcojRootAggValue,
) -> Vec<(u64, u64)> {
    let mut out: BTreeMap<u64, u64> = BTreeMap::new();
    for t in oracle_triangles_u64(e_xy, e_yz, e_xz) {
        let v = oracle_value_u64(t, value);
        match agg {
            AggOp::Sum => {
                *out.entry(t.0).or_default() = out.get(&t.0).copied().unwrap_or(0).wrapping_add(v)
            }
            AggOp::Min => {
                out.entry(t.0)
                    .and_modify(|m| *m = (*m).min(v))
                    .or_insert(v);
            }
            AggOp::Max => {
                out.entry(t.0)
                    .and_modify(|m| *m = (*m).max(v))
                    .or_insert(v);
            }
            other => panic!("unsupported oracle agg {other:?}"),
        }
    }
    out.into_iter().collect()
}

fn sorted_unique_u64(rows: impl IntoIterator<Item = (u64, u64)>) -> Vec<(u64, u64)> {
    let set: BTreeSet<(u64, u64)> = rows.into_iter().collect();
    set.into_iter().collect()
}

fn run_case_u64(
    name: &str,
    e_xy_rows: &[(u64, u64)],
    e_yz_rows: &[(u64, u64)],
    e_xz_rows: &[(u64, u64)],
) {
    let Some(fix) = make_fixture() else {
        eprintln!("skipping {name}: no CUDA device");
        return;
    };
    let e_xy = upload_binary_u64(&fix.memory, e_xy_rows);
    let e_yz = upload_binary_u64(&fix.memory, e_yz_rows);
    let e_xz = upload_binary_u64(&fix.memory, e_xz_rows);
    let stream = fix.pool.acquire().expect("stream");

    for agg in [AggOp::Sum, AggOp::Min, AggOp::Max] {
        for value in [WcojRootAggValue::Y, WcojRootAggValue::Z] {
            let case = format!("{name}/{agg:?}/{value:?}");
            let expected = oracle_agg_u64(e_xy_rows, e_yz_rows, e_xz_rows, agg, value);
            assert!(
                !expected.is_empty(),
                "{case}: fixture must contain at least one triangle"
            );

            // Unfused production baseline: materialize u64 triangles, then
            // legacy groupby with the same AggOp over the value column.
            let tri = fix
                .provider
                .wcoj_triangle_hg_u64_recorded(
                    &e_xy,
                    &e_yz,
                    &e_xz,
                    WCOJ_HG_BLOCK_WORK_UNIT_DEFAULT,
                    stream,
                )
                .expect("baseline u64 triangle materialize");
            let vcol = match value {
                WcojRootAggValue::Y => 1,
                WcojRootAggValue::Z => 2,
            };
            let grouped = fix
                .provider
                .groupby_multi_agg(&tri, &[0], &[(vcol, agg)])
                .expect("baseline u64 groupby agg");
            let baseline = download_groups_u64_u64(&fix.memory, &grouped);
            assert_eq!(baseline, expected, "{case}: unfused u64 baseline vs oracle");

            let fused = fix
                .provider
                .wcoj_triangle_groupby_root_agg_u64_recorded(
                    &e_xy,
                    &e_yz,
                    &e_xz,
                    agg,
                    value,
                    WCOJ_HG_BLOCK_WORK_UNIT_DEFAULT,
                    stream,
                )
                .expect("fused u64 groupby-root agg");
            let fused = download_groups_u64_u64(&fix.memory, &fused);
            assert_eq!(fused, expected, "{case}: fused u64 vs oracle");
        }
    }
}

#[test]
fn groupby_root_agg_u64_matches_oracle_small() {
    // K4 + disjoint triangle, keys above 2^33 so width truncation visibly
    // fails; values are the keys themselves (also above 2^33).
    const B: u64 = 1 << 33;
    let map = |rows: &[(u32, u32)]| -> Vec<(u64, u64)> {
        sorted_unique_u64(rows.iter().map(|&(a, b)| (B + a as u64, B + b as u64)))
    };
    let e_xy = map(&[
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
        (5, 6),
        (5, 7),
        (6, 7),
    ]);
    let e_yz = map(&[(2, 3), (2, 4), (3, 4), (6, 7)]);
    let e_xz = map(&[(1, 3), (1, 4), (2, 4), (3, 4), (5, 7)]);
    run_case_u64("u64_small", &e_xy, &e_yz, &e_xz);
}

#[test]
fn groupby_root_agg_u64_matches_oracle_skewed_hub() {
    const B: u64 = 1 << 40;
    let mut e_xy: Vec<(u64, u64)> = Vec::new();
    let mut e_yz: Vec<(u64, u64)> = Vec::new();
    let mut e_xz: Vec<(u64, u64)> = Vec::new();
    for y in 1..=512u64 {
        e_xy.push((B, B + y));
        for z in 1000..1016u64 {
            e_yz.push((B + y, B + z));
        }
    }
    for z in 1000..1016u64 {
        e_xz.push((B, B + z));
    }
    for i in 0..200u64 {
        let (a, b, c) = (B + 2000 + i, B + 3000 + i, B + 4000 + i);
        e_xy.push((a, b));
        e_yz.push((b, c));
        e_xz.push((a, c));
    }
    let e_xy = sorted_unique_u64(e_xy);
    let e_yz = sorted_unique_u64(e_yz);
    let e_xz = sorted_unique_u64(e_xz);
    run_case_u64("u64_skewed_hub", &e_xy, &e_yz, &e_xz);
}

#[test]
fn groupby_root_agg_u64_empty_intersection_roots_are_absent() {
    const B: u64 = 1 << 33;
    let e_xy = sorted_unique_u64([(B + 1, B + 2), (B + 9, B + 2), (B + 9, B + 3)]);
    let e_yz = sorted_unique_u64([(B + 2, B + 3)]);
    let e_xz = sorted_unique_u64([(B + 1, B + 3)]);

    let Some(fix) = make_fixture() else {
        eprintln!("skipping u64_agg_empty_intersection: no CUDA device");
        return;
    };
    let e_xy_b = upload_binary_u64(&fix.memory, &e_xy);
    let e_yz_b = upload_binary_u64(&fix.memory, &e_yz);
    let e_xz_b = upload_binary_u64(&fix.memory, &e_xz);
    let stream = fix.pool.acquire().expect("stream");
    for agg in [AggOp::Sum, AggOp::Min, AggOp::Max] {
        let fused = fix
            .provider
            .wcoj_triangle_groupby_root_agg_u64_recorded(
                &e_xy_b,
                &e_yz_b,
                &e_xz_b,
                agg,
                WcojRootAggValue::Z,
                WCOJ_HG_BLOCK_WORK_UNIT_DEFAULT,
                stream,
            )
            .expect("fused u64 groupby-root agg");
        let fused = download_groups_u64_u64(&fix.memory, &fused);
        assert_eq!(
            fused,
            vec![(B + 1, B + 3)],
            "{agg:?}: only X=B+1 completes a triangle; X=B+9 must be absent"
        );
    }
}

/// S1c measurement, u64-key sum/min/max (gate: fused >= 3x vs unfused on a
/// skewed fixture). Run explicitly:
/// `cargo test -p xlog-cuda-tests --test test_wcoj_groupby_root_agg \
///    --release -- --ignored --nocapture`
#[test]
#[ignore = "S1c measurement: run explicitly with --ignored --nocapture"]
fn s1c_measurement_u64_agg_fused_vs_unfused() {
    let Some(fix) = make_fixture() else {
        eprintln!("skipping s1c_measurement_u64_agg: no CUDA device");
        return;
    };

    const B: u64 = 1 << 40;
    let hub = |n_y: u64, n_z: u64| {
        let mut e_xy = Vec::new();
        let mut e_yz = Vec::new();
        let mut e_xz = Vec::new();
        for y in 1..=n_y {
            e_xy.push((B, B + y));
            for z in 0..n_z {
                e_yz.push((B + y, B + 1_000_000 + z));
            }
        }
        for z in 0..n_z {
            e_xz.push((B, B + 1_000_000 + z));
        }
        for i in 0..1000u64 {
            let (a, b, c) = (B + 2_000_000 + i, B + 3_000_000 + i, B + 4_000_000 + i);
            e_xy.push((a, b));
            e_yz.push((b, c));
            e_xz.push((a, c));
        }
        (
            sorted_unique_u64(e_xy),
            sorted_unique_u64(e_yz),
            sorted_unique_u64(e_xz),
        )
    };

    let (e_xy_rows, e_yz_rows, e_xz_rows) = hub(10_000, 16);
    let e_xy = upload_binary_u64(&fix.memory, &e_xy_rows);
    let e_yz = upload_binary_u64(&fix.memory, &e_yz_rows);
    let e_xz = upload_binary_u64(&fix.memory, &e_xz_rows);
    let stream = fix.pool.acquire().expect("stream");

    const REPS: usize = 5;
    for agg in [AggOp::Sum, AggOp::Min, AggOp::Max] {
        let value = WcojRootAggValue::Z;
        let expected = oracle_agg_u64(&e_xy_rows, &e_yz_rows, &e_xz_rows, agg, value);

        // Warmup both paths once; assert fused parity vs the host oracle.
        let tri = fix
            .provider
            .wcoj_triangle_hg_u64_recorded(
                &e_xy,
                &e_yz,
                &e_xz,
                WCOJ_HG_BLOCK_WORK_UNIT_DEFAULT,
                stream,
            )
            .expect("baseline warmup");
        drop(tri);
        let warm = fix
            .provider
            .wcoj_triangle_groupby_root_agg_u64_recorded(
                &e_xy,
                &e_yz,
                &e_xz,
                agg,
                value,
                WCOJ_HG_BLOCK_WORK_UNIT_DEFAULT,
                stream,
            )
            .expect("fused warmup");
        assert_eq!(
            download_groups_u64_u64(&fix.memory, &warm),
            expected,
            "u64_hub_10k_z16/{agg:?}: fused parity"
        );
        drop(warm);

        let mut unfused_ms = Vec::with_capacity(REPS);
        let mut fused_ms = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let t = std::time::Instant::now();
            let tri = fix
                .provider
                .wcoj_triangle_hg_u64_recorded(
                    &e_xy,
                    &e_yz,
                    &e_xz,
                    WCOJ_HG_BLOCK_WORK_UNIT_DEFAULT,
                    stream,
                )
                .expect("baseline triangle");
            let grouped = fix
                .provider
                .groupby_multi_agg(&tri, &[0], &[(2, agg)])
                .expect("baseline groupby");
            fix.provider.device().inner().synchronize().expect("sync");
            unfused_ms.push(t.elapsed().as_secs_f64() * 1e3);
            drop(grouped);

            let t = std::time::Instant::now();
            let fused = fix
                .provider
                .wcoj_triangle_groupby_root_agg_u64_recorded(
                    &e_xy,
                    &e_yz,
                    &e_xz,
                    agg,
                    value,
                    WCOJ_HG_BLOCK_WORK_UNIT_DEFAULT,
                    stream,
                )
                .expect("fused");
            fix.provider.device().inner().synchronize().expect("sync");
            fused_ms.push(t.elapsed().as_secs_f64() * 1e3);
            drop(fused);
        }
        unfused_ms.sort_by(|a, b| a.total_cmp(b));
        fused_ms.sort_by(|a, b| a.total_cmp(b));
        let med_unfused = unfused_ms[REPS / 2];
        let med_fused = fused_ms[REPS / 2];
        println!(
            "S1c u64_hub_10k_z16 {agg:?}(Z): unfused median {med_unfused:.3} ms, fused median \
             {med_fused:.3} ms, speedup {:.2}x (n_xy={}, n_yz={}, n_xz={})",
            med_unfused / med_fused,
            e_xy_rows.len(),
            e_yz_rows.len(),
            e_xz_rows.len()
        );
    }
}

/// Provider-level lock: the u32 fused agg entry must reject Symbol value
/// columns (symbol ids are not summable/orderable data; the unfused
/// groupby rejects them too, so a silent fused result would diverge).
#[test]
fn groupby_root_agg_rejects_symbol_value_columns() {
    let Some(fix) = make_fixture() else {
        eprintln!("skipping symbol_value_reject: no CUDA device");
        return;
    };
    let upload_symbol = |rows: &[(u32, u32)]| -> CudaBuffer {
        let n = rows.len() as u32;
        let mut col0 = fix.memory.alloc::<u8>(rows.len() * 4).expect("alloc col0");
        let mut col1 = fix.memory.alloc::<u8>(rows.len() * 4).expect("alloc col1");
        let mut d_num_rows = fix.memory.alloc::<u32>(1).expect("alloc d_num_rows");
        let col0_bytes: Vec<u8> = rows.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
        let col1_bytes: Vec<u8> = rows.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
        let device = fix.memory.device().inner();
        device
            .htod_sync_copy_into(&col0_bytes, &mut col0)
            .expect("htod col0");
        device
            .htod_sync_copy_into(&col1_bytes, &mut col1)
            .expect("htod col1");
        device
            .htod_sync_copy_into(&[n], &mut d_num_rows)
            .expect("htod d_num_rows");
        let schema = Schema::new(vec![
            ("col0".to_string(), ScalarType::Symbol),
            ("col1".to_string(), ScalarType::Symbol),
        ]);
        CudaBuffer::from_columns_with_host_count(
            vec![col0.into(), col1.into()],
            n as u64,
            d_num_rows,
            schema,
            n,
        )
    };
    let e_xy = upload_symbol(&[(1, 2)]);
    let e_yz = upload_symbol(&[(2, 3)]);
    let e_xz = upload_symbol(&[(1, 3)]);
    let stream = fix.pool.acquire().expect("stream");
    for agg in [AggOp::Sum, AggOp::Min, AggOp::Max] {
        let err = match fix.provider.wcoj_triangle_groupby_root_agg_u32_recorded(
            &e_xy,
            &e_yz,
            &e_xz,
            agg,
            WcojRootAggValue::Z,
            WCOJ_HG_BLOCK_WORK_UNIT_DEFAULT,
            stream,
        ) {
            Err(err) => err,
            Ok(_) => panic!("{agg:?}: Symbol value columns must be rejected"),
        };
        let msg = format!("{err}");
        assert!(
            msg.contains("must be U32"),
            "{agg:?}: rejection must name the value-type gate, got: {msg}"
        );
    }
}
