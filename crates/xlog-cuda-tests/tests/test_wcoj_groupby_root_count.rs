//! D1 aggregate-fused WCOJ: group-by-root count over the triangle shape.
//!
//! Contract under test: `wcoj_triangle_groupby_root_count_u32_recorded`
//! computes, for `q(X, count) :- e_xy(X,Y), e_yz(Y,Z), e_xz(X,Z)` grouped by
//! the variable-order root X, the same (X, count) row set as the unfused
//! production path (materialize triangles, then groupby count) — WITHOUT
//! materializing the triangle rows. Both paths are checked against a host
//! brute-force oracle.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use cudarc::driver::sys;
use cudarc::driver::DevicePtr;

use xlog_core::{AggOp, MemoryBudget, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::{CudaBuffer, CudaColumn};
use xlog_cuda::wcoj_metadata::WCOJ_HG_BLOCK_WORK_UNIT_DEFAULT;
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
        let res = sys::cuMemcpyDtoH_v2(
            bytes.as_mut_ptr() as *mut _,
            *c.device_ptr(),
            bytes.len(),
        );
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

/// Sorted (X, count) pairs from a 2-column (U32 key, U64 count) buffer.
fn download_group_counts(memory: &Arc<GpuMemoryManager>, buffer: &CudaBuffer) -> Vec<(u32, u64)> {
    let keys = download_u32_column(memory, buffer, 0);
    let counts = download_u64_column(memory, buffer, 1);
    assert_eq!(keys.len(), counts.len());
    let mut out: Vec<(u32, u64)> = keys.into_iter().zip(counts).collect();
    out.sort();
    out
}

/// Host brute-force oracle: per-X count of distinct (Y, Z) triangle
/// completions.
fn oracle_group_counts(
    e_xy: &[(u32, u32)],
    e_yz: &[(u32, u32)],
    e_xz: &[(u32, u32)],
) -> Vec<(u32, u64)> {
    let xz_set: BTreeSet<(u32, u32)> = e_xz.iter().copied().collect();
    let mut yz_by_y: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for (y, z) in e_yz {
        yz_by_y.entry(*y).or_default().push(*z);
    }
    let mut counts: BTreeMap<u32, u64> = BTreeMap::new();
    for (x, y) in e_xy {
        if let Some(zs) = yz_by_y.get(y) {
            for z in zs {
                if xz_set.contains(&(*x, *z)) {
                    *counts.entry(*x).or_default() += 1;
                }
            }
        }
    }
    counts.into_iter().collect()
}

fn sorted_unique(rows: impl IntoIterator<Item = (u32, u32)>) -> Vec<(u32, u32)> {
    let set: BTreeSet<(u32, u32)> = rows.into_iter().collect();
    set.into_iter().collect()
}

/// Unfused production baseline: materialize triangles, then groupby count.
fn baseline_group_counts(fix: &Fixture, e_xy: &CudaBuffer, e_yz: &CudaBuffer, e_xz: &CudaBuffer) -> Vec<(u32, u64)> {
    let stream = fix.pool.acquire().expect("stream");
    let tri = fix
        .provider
        .wcoj_triangle_hg_u32_recorded(e_xy, e_yz, e_xz, WCOJ_HG_BLOCK_WORK_UNIT_DEFAULT, stream)
        .expect("baseline triangle materialize");
    let grouped = fix
        .provider
        .groupby_multi_agg(&tri, &[0], &[(1, AggOp::Count)])
        .expect("baseline groupby count");
    download_group_counts(&fix.memory, &grouped)
}

fn run_case(name: &str, e_xy_rows: &[(u32, u32)], e_yz_rows: &[(u32, u32)], e_xz_rows: &[(u32, u32)]) {
    let Some(fix) = make_fixture() else {
        eprintln!("skipping {name}: no CUDA device");
        return;
    };
    let expected = oracle_group_counts(e_xy_rows, e_yz_rows, e_xz_rows);
    assert!(
        !expected.is_empty(),
        "{name}: fixture must contain at least one triangle"
    );

    let e_xy = upload_binary_u32(&fix.memory, e_xy_rows);
    let e_yz = upload_binary_u32(&fix.memory, e_yz_rows);
    let e_xz = upload_binary_u32(&fix.memory, e_xz_rows);

    let baseline = baseline_group_counts(&fix, &e_xy, &e_yz, &e_xz);
    assert_eq!(baseline, expected, "{name}: unfused baseline vs oracle");

    let stream = fix.pool.acquire().expect("stream");
    let fused = fix
        .provider
        .wcoj_triangle_groupby_root_count_u32_recorded(
            &e_xy,
            &e_yz,
            &e_xz,
            WCOJ_HG_BLOCK_WORK_UNIT_DEFAULT,
            stream,
        )
        .expect("fused groupby-root count");
    let fused = download_group_counts(&fix.memory, &fused);
    assert_eq!(fused, expected, "{name}: fused vs oracle");
}

#[test]
fn groupby_root_count_matches_oracle_small() {
    // K4 on {1..4} plus a disjoint triangle {5,6,7}; X=1 completes 3
    // triangles, X=2 completes 1, X=5 completes 1.
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
fn groupby_root_count_matches_oracle_skewed_hub() {
    // Super-hub: X=0 connects to 512 Y values; Y values chain to a shared
    // Z band; plus a uniform background. Exercises heavy per-root fanout
    // (many work units feeding the same root counter).
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
    // Uniform background away from the hub.
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
fn groupby_root_count_empty_intersection_roots_are_absent() {
    // X=9 has e_xy edges but no completing (Y,Z): it must NOT appear in the
    // fused output (group-by over the join result, not over e_xy).
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
    let stream = fix.pool.acquire().expect("stream");
    let fused = fix
        .provider
        .wcoj_triangle_groupby_root_count_u32_recorded(
            &e_xy_b,
            &e_yz_b,
            &e_xz_b,
            WCOJ_HG_BLOCK_WORK_UNIT_DEFAULT,
            stream,
        )
        .expect("fused groupby-root count");
    let fused = download_group_counts(&fix.memory, &fused);
    assert_eq!(fused, vec![(1u32, 1u64)], "only X=1 completes a triangle");
}
