//! S1d — aggregate-fused WCOJ: group-by-root sum/min/max over the 4-cycle
//! shape.
//!
//! Contract under test: `wcoj_4cycle_groupby_root_agg_u32_recorded`
//! computes, for `q(W, agg(V)) :- e1(W,X), e2(X,Y), e3(Y,Z), e4(Z,W)` with
//! `agg ∈ {Sum, Min, Max}` and `V ∈ {X, Y, Z}` grouped by the
//! variable-order root W, the same (W, agg) row set as the unfused
//! production path (materialize 4-cycles, then groupby) — WITHOUT
//! materializing the 4-cycle rows. Both paths are checked against a host
//! brute-force oracle.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use cudarc::driver::sys;

use xlog_core::{AggOp, MemoryBudget, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::{CudaBuffer, CudaColumn};
use xlog_cuda::wcoj_metadata::{Wcoj4CycleRootAggValue, WCOJ_HG_BLOCK_WORK_UNIT_DEFAULT};
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

/// Sorted (W, agg) pairs from a 2-column (U32 key, U64 agg) buffer.
fn download_groups_u64(memory: &Arc<GpuMemoryManager>, buffer: &CudaBuffer) -> Vec<(u32, u64)> {
    let keys = download_u32_column(memory, buffer, 0);
    let aggs = download_u64_column(memory, buffer, 1);
    assert_eq!(keys.len(), aggs.len());
    let mut out: Vec<(u32, u64)> = keys.into_iter().zip(aggs).collect();
    out.sort();
    out
}

/// Sorted (W, agg) pairs from a 2-column (U32 key, U32 agg) buffer.
fn download_groups_u32(memory: &Arc<GpuMemoryManager>, buffer: &CudaBuffer) -> Vec<(u32, u32)> {
    let keys = download_u32_column(memory, buffer, 0);
    let aggs = download_u32_column(memory, buffer, 1);
    assert_eq!(keys.len(), aggs.len());
    let mut out: Vec<(u32, u32)> = keys.into_iter().zip(aggs).collect();
    out.sort();
    out
}

/// Host brute-force oracle: the bag of (W, X, Y, Z) 4-cycle completions.
fn oracle_quads(
    e1: &[(u32, u32)],
    e2: &[(u32, u32)],
    e3: &[(u32, u32)],
    e4: &[(u32, u32)],
) -> Vec<(u32, u32, u32, u32)> {
    let e4_set: BTreeSet<(u32, u32)> = e4.iter().copied().collect();
    let mut e2_by_x: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for (x, y) in e2 {
        e2_by_x.entry(*x).or_default().push(*y);
    }
    let mut e3_by_y: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for (y, z) in e3 {
        e3_by_y.entry(*y).or_default().push(*z);
    }
    let mut quads = Vec::new();
    for (w, x) in e1 {
        if let Some(ys) = e2_by_x.get(x) {
            for y in ys {
                if let Some(zs) = e3_by_y.get(y) {
                    for z in zs {
                        if e4_set.contains(&(*z, *w)) {
                            quads.push((*w, *x, *y, *z));
                        }
                    }
                }
            }
        }
    }
    quads
}

fn oracle_value(quad: (u32, u32, u32, u32), value: Wcoj4CycleRootAggValue) -> u32 {
    match value {
        Wcoj4CycleRootAggValue::X => quad.1,
        Wcoj4CycleRootAggValue::Y => quad.2,
        Wcoj4CycleRootAggValue::Z => quad.3,
    }
}

fn oracle_sums(
    e1: &[(u32, u32)],
    e2: &[(u32, u32)],
    e3: &[(u32, u32)],
    e4: &[(u32, u32)],
    value: Wcoj4CycleRootAggValue,
) -> Vec<(u32, u64)> {
    let mut sums: BTreeMap<u32, u64> = BTreeMap::new();
    for quad in oracle_quads(e1, e2, e3, e4) {
        *sums.entry(quad.0).or_default() += u64::from(oracle_value(quad, value));
    }
    sums.into_iter().collect()
}

fn oracle_mins(
    e1: &[(u32, u32)],
    e2: &[(u32, u32)],
    e3: &[(u32, u32)],
    e4: &[(u32, u32)],
    value: Wcoj4CycleRootAggValue,
) -> Vec<(u32, u32)> {
    let mut mins: BTreeMap<u32, u32> = BTreeMap::new();
    for quad in oracle_quads(e1, e2, e3, e4) {
        let v = oracle_value(quad, value);
        mins.entry(quad.0)
            .and_modify(|m| *m = (*m).min(v))
            .or_insert(v);
    }
    mins.into_iter().collect()
}

fn oracle_maxs(
    e1: &[(u32, u32)],
    e2: &[(u32, u32)],
    e3: &[(u32, u32)],
    e4: &[(u32, u32)],
    value: Wcoj4CycleRootAggValue,
) -> Vec<(u32, u32)> {
    let mut maxs: BTreeMap<u32, u32> = BTreeMap::new();
    for quad in oracle_quads(e1, e2, e3, e4) {
        let v = oracle_value(quad, value);
        maxs.entry(quad.0)
            .and_modify(|m| *m = (*m).max(v))
            .or_insert(v);
    }
    maxs.into_iter().collect()
}

fn sorted_unique(rows: impl IntoIterator<Item = (u32, u32)>) -> Vec<(u32, u32)> {
    let set: BTreeSet<(u32, u32)> = rows.into_iter().collect();
    set.into_iter().collect()
}

/// (W, X, Y, Z) materialized output column carrying the aggregate value.
fn value_col(value: Wcoj4CycleRootAggValue) -> usize {
    match value {
        Wcoj4CycleRootAggValue::X => 1,
        Wcoj4CycleRootAggValue::Y => 2,
        Wcoj4CycleRootAggValue::Z => 3,
    }
}

/// Unfused production baseline: materialize 4-cycles, then groupby agg.
fn baseline_buffer(
    fix: &Fixture,
    stream: xlog_cuda::device_runtime::StreamId,
    e1: &CudaBuffer,
    e2: &CudaBuffer,
    e3: &CudaBuffer,
    e4: &CudaBuffer,
    agg_op: AggOp,
    value: Wcoj4CycleRootAggValue,
) -> CudaBuffer {
    let quads = fix
        .provider
        .wcoj_4cycle_hg_u32_recorded(e1, e2, e3, e4, WCOJ_HG_BLOCK_WORK_UNIT_DEFAULT, stream)
        .expect("baseline 4-cycle materialize");
    fix.provider
        .groupby_multi_agg(&quads, &[0], &[(value_col(value), agg_op)])
        .expect("baseline groupby agg")
}

fn fused_buffer(
    fix: &Fixture,
    stream: xlog_cuda::device_runtime::StreamId,
    e1: &CudaBuffer,
    e2: &CudaBuffer,
    e3: &CudaBuffer,
    e4: &CudaBuffer,
    agg_op: AggOp,
    value: Wcoj4CycleRootAggValue,
) -> CudaBuffer {
    fix.provider
        .wcoj_4cycle_groupby_root_agg_u32_recorded(
            e1,
            e2,
            e3,
            e4,
            agg_op,
            value,
            WCOJ_HG_BLOCK_WORK_UNIT_DEFAULT,
            stream,
        )
        .expect("fused 4-cycle groupby-root agg")
}

fn run_case(
    name: &str,
    e1_rows: &[(u32, u32)],
    e2_rows: &[(u32, u32)],
    e3_rows: &[(u32, u32)],
    e4_rows: &[(u32, u32)],
) {
    let Some(fix) = make_fixture() else {
        eprintln!("skipping {name}: no CUDA device");
        return;
    };
    assert!(
        !oracle_quads(e1_rows, e2_rows, e3_rows, e4_rows).is_empty(),
        "{name}: fixture must contain at least one 4-cycle"
    );
    let e1 = upload_binary_u32(&fix.memory, e1_rows);
    let e2 = upload_binary_u32(&fix.memory, e2_rows);
    let e3 = upload_binary_u32(&fix.memory, e3_rows);
    let e4 = upload_binary_u32(&fix.memory, e4_rows);
    // One stream per case (grow-only StreamPool, capacity 16).
    let stream = fix.pool.acquire().expect("stream");

    for value in [
        Wcoj4CycleRootAggValue::X,
        Wcoj4CycleRootAggValue::Y,
        Wcoj4CycleRootAggValue::Z,
    ] {
        // Sum (U64 output).
        let expected = oracle_sums(e1_rows, e2_rows, e3_rows, e4_rows, value);
        let baseline = baseline_buffer(&fix, stream, &e1, &e2, &e3, &e4, AggOp::Sum, value);
        assert_eq!(
            download_groups_u64(&fix.memory, &baseline),
            expected,
            "{name}/{value:?}: unfused sum baseline vs oracle"
        );
        let fused = fused_buffer(&fix, stream, &e1, &e2, &e3, &e4, AggOp::Sum, value);
        assert_eq!(
            download_groups_u64(&fix.memory, &fused),
            expected,
            "{name}/{value:?}: fused sum vs oracle"
        );

        // Min (U32 output).
        let expected = oracle_mins(e1_rows, e2_rows, e3_rows, e4_rows, value);
        let baseline = baseline_buffer(&fix, stream, &e1, &e2, &e3, &e4, AggOp::Min, value);
        assert_eq!(
            download_groups_u32(&fix.memory, &baseline),
            expected,
            "{name}/{value:?}: unfused min baseline vs oracle"
        );
        let fused = fused_buffer(&fix, stream, &e1, &e2, &e3, &e4, AggOp::Min, value);
        assert_eq!(
            download_groups_u32(&fix.memory, &fused),
            expected,
            "{name}/{value:?}: fused min vs oracle"
        );

        // Max (U32 output).
        let expected = oracle_maxs(e1_rows, e2_rows, e3_rows, e4_rows, value);
        let baseline = baseline_buffer(&fix, stream, &e1, &e2, &e3, &e4, AggOp::Max, value);
        assert_eq!(
            download_groups_u32(&fix.memory, &baseline),
            expected,
            "{name}/{value:?}: unfused max baseline vs oracle"
        );
        let fused = fused_buffer(&fix, stream, &e1, &e2, &e3, &e4, AggOp::Max, value);
        assert_eq!(
            download_groups_u32(&fix.memory, &fused),
            expected,
            "{name}/{value:?}: fused max vs oracle"
        );
    }
}

#[test]
fn cycle4_groupby_root_agg_matches_oracle_small() {
    // W=1 closes 3 cycles, W=2 closes 2; W=3 has e1/e2/e3 paths but no
    // closing e4 edge back to itself. Distinct X/Y/Z values per root so
    // min != max.
    let e1 = sorted_unique([(1, 10), (1, 11), (2, 10), (3, 12)]);
    let e2 = sorted_unique([(10, 20), (10, 21), (11, 20), (12, 22)]);
    let e3 = sorted_unique([(20, 30), (21, 31), (22, 32)]);
    let e4 = sorted_unique([(30, 1), (30, 2), (31, 1), (31, 2), (32, 9)]);
    run_case("cycle4_agg_small", &e1, &e2, &e3, &e4);
}

#[test]
fn cycle4_groupby_root_agg_matches_oracle_skewed_hub() {
    // Super-hub: W=0 connects to 256 X values; each X fans to a shared
    // Y band; each Y fans to a shared Z band; every Z closes back to 0.
    // Heavy per-root fanout: many work units fold into the same root
    // accumulator.
    let mut e1: Vec<(u32, u32)> = Vec::new();
    let mut e2: Vec<(u32, u32)> = Vec::new();
    let mut e3: Vec<(u32, u32)> = Vec::new();
    let mut e4: Vec<(u32, u32)> = Vec::new();
    for x in 1..=256u32 {
        e1.push((0, x));
        for y in 1000..1008u32 {
            e2.push((x, y));
        }
    }
    for y in 1000..1008u32 {
        for z in 2000..2008u32 {
            e3.push((y, z));
        }
    }
    for z in 2000..2008u32 {
        e4.push((z, 0));
    }
    // Uniform background away from the hub.
    for i in 0..200u32 {
        let (a, b, c, d) = (10_000 + i, 20_000 + i, 30_000 + i, 40_000 + i);
        e1.push((a, b));
        e2.push((b, c));
        e3.push((c, d));
        e4.push((d, a));
    }
    let e1 = sorted_unique(e1);
    let e2 = sorted_unique(e2);
    let e3 = sorted_unique(e3);
    let e4 = sorted_unique(e4);
    run_case("cycle4_agg_skewed_hub", &e1, &e2, &e3, &e4);
}

/// Bag semantics: distinct (X, Y, Z) completions sharing the same
/// projected value must each contribute (sum counts duplicates).
#[test]
fn cycle4_groupby_root_agg_bag_semantics_duplicate_projected_values() {
    // W=1 has two completions (X=10,Y=20,Z=30) and (X=11,Y=21,Z=30):
    // sum(Z) = 60 (Z=30 counted twice), min(Z) = max(Z) = 30.
    let e1 = sorted_unique([(1, 10), (1, 11)]);
    let e2 = sorted_unique([(10, 20), (11, 21)]);
    let e3 = sorted_unique([(20, 30), (21, 30)]);
    let e4 = sorted_unique([(30, 1)]);

    let Some(fix) = make_fixture() else {
        eprintln!("skipping cycle4_bag_semantics: no CUDA device");
        return;
    };
    let e1_b = upload_binary_u32(&fix.memory, &e1);
    let e2_b = upload_binary_u32(&fix.memory, &e2);
    let e3_b = upload_binary_u32(&fix.memory, &e3);
    let e4_b = upload_binary_u32(&fix.memory, &e4);
    let stream = fix.pool.acquire().expect("stream");
    let fused = fused_buffer(
        &fix,
        stream,
        &e1_b,
        &e2_b,
        &e3_b,
        &e4_b,
        AggOp::Sum,
        Wcoj4CycleRootAggValue::Z,
    );
    assert_eq!(
        download_groups_u64(&fix.memory, &fused),
        vec![(1u32, 60u64)],
        "sum must count the duplicate projected Z once per completion"
    );
}

/// Roots whose e1 rows never complete a 4-cycle must be ABSENT from the
/// output (group-by over the join result, not over e1).
#[test]
fn cycle4_groupby_root_agg_empty_intersection_roots_are_absent() {
    let e1 = sorted_unique([(1, 10), (9, 10), (9, 11)]);
    let e2 = sorted_unique([(10, 20)]);
    let e3 = sorted_unique([(20, 30)]);
    let e4 = sorted_unique([(30, 1)]);

    let Some(fix) = make_fixture() else {
        eprintln!("skipping cycle4_empty_intersection: no CUDA device");
        return;
    };
    let e1_b = upload_binary_u32(&fix.memory, &e1);
    let e2_b = upload_binary_u32(&fix.memory, &e2);
    let e3_b = upload_binary_u32(&fix.memory, &e3);
    let e4_b = upload_binary_u32(&fix.memory, &e4);
    let stream = fix.pool.acquire().expect("stream");
    for (agg, expected) in [
        (AggOp::Sum, vec![(1u32, 30u64)]),
        (AggOp::Min, vec![(1u32, 30u64)]),
        (AggOp::Max, vec![(1u32, 30u64)]),
    ] {
        let fused = fused_buffer(
            &fix,
            stream,
            &e1_b,
            &e2_b,
            &e3_b,
            &e4_b,
            agg,
            Wcoj4CycleRootAggValue::Z,
        );
        let rows: Vec<(u32, u64)> = match agg {
            AggOp::Sum => download_groups_u64(&fix.memory, &fused),
            _ => download_groups_u32(&fix.memory, &fused)
                .into_iter()
                .map(|(k, v)| (k, u64::from(v)))
                .collect(),
        };
        assert_eq!(rows, expected, "{agg:?}: only W=1 closes a 4-cycle");
    }
}

/// Sum over a value that is 0 for every completion must still emit the
/// group (count>0 gates presence, not the aggregate value).
#[test]
fn cycle4_groupby_root_agg_sum_zero_valued_groups_are_present() {
    // X value 0 for the only completing root: sum(X) = 0, group present.
    let e1 = sorted_unique([(1, 0)]);
    let e2 = sorted_unique([(0, 20)]);
    let e3 = sorted_unique([(20, 30)]);
    let e4 = sorted_unique([(30, 1)]);

    let Some(fix) = make_fixture() else {
        eprintln!("skipping cycle4_sum_zero: no CUDA device");
        return;
    };
    let e1_b = upload_binary_u32(&fix.memory, &e1);
    let e2_b = upload_binary_u32(&fix.memory, &e2);
    let e3_b = upload_binary_u32(&fix.memory, &e3);
    let e4_b = upload_binary_u32(&fix.memory, &e4);
    let stream = fix.pool.acquire().expect("stream");
    let fused = fused_buffer(
        &fix,
        stream,
        &e1_b,
        &e2_b,
        &e3_b,
        &e4_b,
        AggOp::Sum,
        Wcoj4CycleRootAggValue::X,
    );
    assert_eq!(
        download_groups_u64(&fix.memory, &fused),
        vec![(1u32, 0u64)],
        "zero-valued sum group must be present"
    );
}

/// Symbol value columns must be rejected by the provider entry (the
/// executor declines them before dispatch; this locks the provider-level
/// gate for direct callers).
#[test]
fn cycle4_groupby_root_agg_rejects_symbol_value_columns() {
    let Some(fix) = make_fixture() else {
        eprintln!("skipping cycle4_symbol_value_reject: no CUDA device");
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
    let e1 = upload_symbol(&[(1, 10)]);
    let e2 = upload_symbol(&[(10, 20)]);
    let e3 = upload_symbol(&[(20, 30)]);
    let e4 = upload_symbol(&[(30, 1)]);
    let stream = fix.pool.acquire().expect("stream");
    for agg in [AggOp::Sum, AggOp::Min, AggOp::Max] {
        for value in [
            Wcoj4CycleRootAggValue::X,
            Wcoj4CycleRootAggValue::Y,
            Wcoj4CycleRootAggValue::Z,
        ] {
            let err = match fix.provider.wcoj_4cycle_groupby_root_agg_u32_recorded(
                &e1,
                &e2,
                &e3,
                &e4,
                agg,
                value,
                WCOJ_HG_BLOCK_WORK_UNIT_DEFAULT,
                stream,
            ) {
                Err(err) => err,
                Ok(_) => panic!("{agg:?}/{value:?}: Symbol value columns must be rejected"),
            };
            let msg = format!("{err}");
            assert!(
                msg.contains("must be U32"),
                "{agg:?}/{value:?}: rejection must name the value-type gate, got: {msg}"
            );
        }
    }
}

/// S1d measurement (gate: fused >= 3x vs unfused on the skewed 4-cycle
/// fixtures, per aggregate). Run explicitly:
/// `cargo test -p xlog-cuda-tests --test test_wcoj_4cycle_groupby_root_agg \
///    --release -- --ignored --nocapture`
/// Asserts parity; timing ratios are PRINTED and recorded as evidence, not
/// asserted (wall-clock assertions are machine-dependent).
#[test]
#[ignore = "S1d measurement: run explicitly with --ignored --nocapture"]
fn s1d_measurement_4cycle_agg_fused_vs_unfused() {
    let Some(fix) = make_fixture() else {
        eprintln!("skipping s1d_measurement: no CUDA device");
        return;
    };

    // Hub W=0 with n_x X-fanout; X values share a 16-wide Y band; Y values
    // share a 16-wide Z band; every Z closes back to W=0. Completions per
    // root row: 256 — the materialized 4-cycle row count is n_x * 256.
    let hub = |n_x: u32| {
        let mut e1 = Vec::new();
        let mut e2 = Vec::new();
        let mut e3 = Vec::new();
        let mut e4 = Vec::new();
        for x in 1..=n_x {
            e1.push((0u32, x));
            for y in 0..16u32 {
                e2.push((x, 1_000_000 + y));
            }
        }
        for y in 0..16u32 {
            for z in 0..16u32 {
                e3.push((1_000_000 + y, 2_000_000 + z));
            }
        }
        for z in 0..16u32 {
            e4.push((2_000_000 + z, 0u32));
        }
        // Uniform background so the group column is not a single value.
        for i in 0..1000u32 {
            let (a, b, c, d) = (
                3_000_000 + i,
                4_000_000 + i,
                5_000_000 + i,
                6_000_000 + i,
            );
            e1.push((a, b));
            e2.push((b, c));
            e3.push((c, d));
            e4.push((d, a));
        }
        (
            sorted_unique(e1),
            sorted_unique(e2),
            sorted_unique(e3),
            sorted_unique(e4),
        )
    };

    let (e1_rows, e2_rows, e3_rows, e4_rows) = hub(10_000);
    let e1 = upload_binary_u32(&fix.memory, &e1_rows);
    let e2 = upload_binary_u32(&fix.memory, &e2_rows);
    let e3 = upload_binary_u32(&fix.memory, &e3_rows);
    let e4 = upload_binary_u32(&fix.memory, &e4_rows);
    // One stream reused across all reps (grow-only StreamPool).
    let stream = fix.pool.acquire().expect("stream");

    const REPS: usize = 5;
    let value = Wcoj4CycleRootAggValue::Z;
    for agg in [AggOp::Sum, AggOp::Min, AggOp::Max] {
        // Parity check (also acts as warmup of both paths).
        match agg {
            AggOp::Sum => {
                let expected = oracle_sums(&e1_rows, &e2_rows, &e3_rows, &e4_rows, value);
                let warm = fused_buffer(&fix, stream, &e1, &e2, &e3, &e4, agg, value);
                assert_eq!(
                    download_groups_u64(&fix.memory, &warm),
                    expected,
                    "{agg:?}: fused parity"
                );
                let base = baseline_buffer(&fix, stream, &e1, &e2, &e3, &e4, agg, value);
                assert_eq!(
                    download_groups_u64(&fix.memory, &base),
                    expected,
                    "{agg:?}: unfused parity"
                );
            }
            _ => {
                let expected = match agg {
                    AggOp::Min => oracle_mins(&e1_rows, &e2_rows, &e3_rows, &e4_rows, value),
                    _ => oracle_maxs(&e1_rows, &e2_rows, &e3_rows, &e4_rows, value),
                };
                let warm = fused_buffer(&fix, stream, &e1, &e2, &e3, &e4, agg, value);
                assert_eq!(
                    download_groups_u32(&fix.memory, &warm),
                    expected,
                    "{agg:?}: fused parity"
                );
                let base = baseline_buffer(&fix, stream, &e1, &e2, &e3, &e4, agg, value);
                assert_eq!(
                    download_groups_u32(&fix.memory, &base),
                    expected,
                    "{agg:?}: unfused parity"
                );
            }
        }

        let mut unfused_ms = Vec::with_capacity(REPS);
        let mut fused_ms = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let t = std::time::Instant::now();
            let base = baseline_buffer(&fix, stream, &e1, &e2, &e3, &e4, agg, value);
            fix.provider.device().inner().synchronize().expect("sync");
            unfused_ms.push(t.elapsed().as_secs_f64() * 1e3);
            drop(base);

            let t = std::time::Instant::now();
            let fused = fused_buffer(&fix, stream, &e1, &e2, &e3, &e4, agg, value);
            fix.provider.device().inner().synchronize().expect("sync");
            fused_ms.push(t.elapsed().as_secs_f64() * 1e3);
            drop(fused);
        }
        unfused_ms.sort_by(|a, b| a.total_cmp(b));
        fused_ms.sort_by(|a, b| a.total_cmp(b));
        let med_unfused = unfused_ms[REPS / 2];
        let med_fused = fused_ms[REPS / 2];
        println!(
            "S1d cycle4_hub_10k {agg:?}(Z): unfused median {med_unfused:.3} ms, fused median \
             {med_fused:.3} ms, speedup {:.2}x (n_e1={}, n_e2={}, n_e3={}, n_e4={})",
            med_unfused / med_fused,
            e1_rows.len(),
            e2_rows.len(),
            e3_rows.len(),
            e4_rows.len()
        );
    }
}
