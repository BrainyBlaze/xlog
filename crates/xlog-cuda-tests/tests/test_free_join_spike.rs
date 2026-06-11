//! D2 Phase A — S2 spike: provider-level GPU Free Join frontier engine.
//!
//! Contract under test: `free_join_execute_u32_recorded` executes a
//! hand-built Free Join plan (design doc
//! `docs/plans/2026-06-12-d2-free-join-design.md` §2-§3) over
//! layout-normalized u32 relations via level-synchronous frontier
//! execution (two-phase EXPAND count→scan→emit per cover subatom,
//! per-subatom PROBE refine + mask compaction) and produces exactly the
//! join's projected row set, checked against host brute-force oracles
//! on every fixture.
//!
//! Measurement tests (`#[ignore]`) implement the S2 gates:
//!   * blowup chain — >= 2x vs the production binary hash-join path;
//!   * hub triangle — <= 1.2x vs the dedicated triangle WCOJ kernel.

use std::collections::BTreeSet;
use std::sync::Arc;

use cudarc::driver::sys;

use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::{CudaBuffer, CudaColumn};
use xlog_cuda::wcoj_metadata::WCOJ_HG_BLOCK_WORK_UNIT_DEFAULT;
use xlog_cuda::{
    CudaDevice, CudaKernelProvider, FjNode, FjPlan, FjSubAtom, GpuMemoryManager, JoinType,
};

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

fn make_fixture_with_budget(budget_bytes: u64) -> Option<Fixture> {
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
        Box::new(GlobalDeviceBudget::new(logging, budget_bytes as usize));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(budget_bytes),
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

fn make_fixture() -> Option<Fixture> {
    make_fixture_with_budget(512 * 1024 * 1024)
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

/// Raw partial-column download: engine outputs may allocate columns at
/// capacity, so copies must use the logical byte length.
fn download_u32_column(
    memory: &Arc<GpuMemoryManager>,
    buffer: &CudaBuffer,
    col: usize,
) -> Vec<u32> {
    let n = buffer_rows(memory, buffer);
    let mut bytes = vec![0u8; n * 4];
    if n == 0 {
        return Vec::new();
    }
    let CudaColumn::Owned(c) = buffer.column(col).expect("column") else {
        panic!("column must be owned");
    };
    unsafe {
        let res = sys::cuMemcpyDtoH_v2(bytes.as_mut_ptr() as *mut _, *c.device_ptr(), bytes.len());
        assert_eq!(res, sys::cudaError_enum::CUDA_SUCCESS, "dtoh column copy");
    }
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Download all columns of a result buffer as a sorted set of rows.
fn download_rows_sorted(
    memory: &Arc<GpuMemoryManager>,
    buffer: &CudaBuffer,
) -> Vec<Vec<u32>> {
    let arity = buffer.arity();
    let cols: Vec<Vec<u32>> = (0..arity)
        .map(|c| download_u32_column(memory, buffer, c))
        .collect();
    let n = cols.first().map(|c| c.len()).unwrap_or(0);
    let mut rows: Vec<Vec<u32>> = (0..n)
        .map(|i| cols.iter().map(|c| c[i]).collect())
        .collect();
    rows.sort();
    rows
}

fn sorted_unique(rows: impl IntoIterator<Item = (u32, u32)>) -> Vec<(u32, u32)> {
    let set: BTreeSet<(u32, u32)> = rows.into_iter().collect();
    set.into_iter().collect()
}

// =====================================================================
// Host oracles
// =====================================================================

/// Chain oracle: Q(a,x,y,z,b) :- R(a,x), S(x,y), T(y,z), U(z,b).
fn oracle_chain(
    r: &[(u32, u32)],
    s: &[(u32, u32)],
    t: &[(u32, u32)],
    u: &[(u32, u32)],
) -> Vec<Vec<u32>> {
    let mut s_by_x: std::collections::BTreeMap<u32, Vec<u32>> = Default::default();
    for &(x, y) in s {
        s_by_x.entry(x).or_default().push(y);
    }
    let mut t_by_y: std::collections::BTreeMap<u32, Vec<u32>> = Default::default();
    for &(y, z) in t {
        t_by_y.entry(y).or_default().push(z);
    }
    let mut u_by_z: std::collections::BTreeMap<u32, Vec<u32>> = Default::default();
    for &(z, b) in u {
        u_by_z.entry(z).or_default().push(b);
    }
    let mut out: BTreeSet<Vec<u32>> = BTreeSet::new();
    for &(a, x) in r {
        let Some(ys) = s_by_x.get(&x) else { continue };
        for &y in ys {
            let Some(zs) = t_by_y.get(&y) else { continue };
            for &z in zs {
                let Some(bs) = u_by_z.get(&z) else { continue };
                for &b in bs {
                    out.insert(vec![a, x, y, z, b]);
                }
            }
        }
    }
    out.into_iter().collect()
}

/// Star oracle: Q(x,a,b,c) :- R(x,a), S(x,b), T(x,c).
fn oracle_star(r: &[(u32, u32)], s: &[(u32, u32)], t: &[(u32, u32)]) -> Vec<Vec<u32>> {
    let mut s_by_x: std::collections::BTreeMap<u32, Vec<u32>> = Default::default();
    for &(x, b) in s {
        s_by_x.entry(x).or_default().push(b);
    }
    let mut t_by_x: std::collections::BTreeMap<u32, Vec<u32>> = Default::default();
    for &(x, c) in t {
        t_by_x.entry(x).or_default().push(c);
    }
    let mut out: BTreeSet<Vec<u32>> = BTreeSet::new();
    for &(x, a) in r {
        let (Some(bs), Some(cs)) = (s_by_x.get(&x), t_by_x.get(&x)) else {
            continue;
        };
        for &b in bs {
            for &c in cs {
                out.insert(vec![x, a, b, c]);
            }
        }
    }
    out.into_iter().collect()
}

/// Triangle oracle: Q(x,y,z) :- R(x,y), S(y,z), T(x,z).
fn oracle_triangle(r: &[(u32, u32)], s: &[(u32, u32)], t: &[(u32, u32)]) -> Vec<Vec<u32>> {
    let t_set: BTreeSet<(u32, u32)> = t.iter().copied().collect();
    let mut s_by_y: std::collections::BTreeMap<u32, Vec<u32>> = Default::default();
    for &(y, z) in s {
        s_by_y.entry(y).or_default().push(z);
    }
    let mut out: BTreeSet<Vec<u32>> = BTreeSet::new();
    for &(x, y) in r {
        let Some(zs) = s_by_y.get(&y) else { continue };
        for &z in zs {
            if t_set.contains(&(x, z)) {
                out.insert(vec![x, y, z]);
            }
        }
    }
    out.into_iter().collect()
}

// =====================================================================
// Hand-built Free Join plans (design doc §3)
// =====================================================================

fn sub(input_idx: usize, vars: &[usize]) -> FjSubAtom {
    FjSubAtom {
        input_idx,
        var_positions: vars.to_vec(),
    }
}

/// Chain, natural binary2fj-shaped plan. Inputs [R, S, T, U]; vars
/// a=0, x=1, y=2, z=3, b=4.
fn chain_plan_natural() -> FjPlan {
    FjPlan {
        num_vars: 5,
        nodes: vec![
            FjNode {
                cover: sub(0, &[0, 1]),
                probes: vec![sub(1, &[1])],
            },
            FjNode {
                cover: sub(1, &[2]),
                probes: vec![sub(2, &[2])],
            },
            FjNode {
                cover: sub(2, &[3]),
                probes: vec![sub(3, &[3])],
            },
            FjNode {
                cover: sub(3, &[4]),
                probes: vec![],
            },
        ],
        output_vars: vec![0, 1, 2, 3, 4],
    }
}

/// Chain, U-as-cover plan: node 3 iterates U's distinct z values
/// (small) and probes T(z), so the z expansion never inflates the
/// frontier with T's full fanout. T is consumed entirely by probes.
fn chain_plan_u_cover() -> FjPlan {
    FjPlan {
        num_vars: 5,
        nodes: vec![
            FjNode {
                cover: sub(0, &[0, 1]),
                probes: vec![sub(1, &[1])],
            },
            FjNode {
                cover: sub(1, &[2]),
                probes: vec![sub(2, &[2])],
            },
            FjNode {
                cover: sub(3, &[3]),
                probes: vec![sub(2, &[3])],
            },
            FjNode {
                cover: sub(3, &[4]),
                probes: vec![],
            },
        ],
        output_vars: vec![0, 1, 2, 3, 4],
    }
}

/// Star/clover plan. Inputs [R, S, T]; vars x=0, a=1, b=2, c=3.
fn star_plan() -> FjPlan {
    FjPlan {
        num_vars: 4,
        nodes: vec![
            FjNode {
                cover: sub(0, &[0, 1]),
                probes: vec![sub(1, &[0]), sub(2, &[0])],
            },
            FjNode {
                cover: sub(1, &[2]),
                probes: vec![],
            },
            FjNode {
                cover: sub(2, &[3]),
                probes: vec![],
            },
        ],
        output_vars: vec![0, 1, 2, 3],
    }
}

/// Triangle plan. Inputs [R, S, T]; vars x=0, y=1, z=2.
fn triangle_plan() -> FjPlan {
    FjPlan {
        num_vars: 3,
        nodes: vec![
            FjNode {
                cover: sub(0, &[0, 1]),
                probes: vec![sub(1, &[1]), sub(2, &[0])],
            },
            FjNode {
                cover: sub(1, &[2]),
                probes: vec![sub(2, &[2])],
            },
        ],
        output_vars: vec![0, 1, 2],
    }
}

// =====================================================================
// Parity tests
// =====================================================================

#[test]
fn fj_chain_matches_oracle() {
    let Some(fix) = make_fixture() else {
        eprintln!("skipping fj_chain: no CUDA device");
        return;
    };
    // Small chain with shared joins, dead ends, and multi-fanout.
    let r = sorted_unique([(0, 1), (0, 2), (7, 2), (9, 99)]);
    let s = sorted_unique([(1, 10), (1, 11), (2, 10), (2, 12), (50, 50)]);
    let t = sorted_unique([(10, 20), (10, 21), (11, 22), (12, 23), (60, 60)]);
    let u = sorted_unique([(20, 5), (21, 5), (21, 6), (23, 7), (70, 70)]);
    let expected = oracle_chain(&r, &s, &t, &u);
    assert!(!expected.is_empty(), "fixture must produce rows");

    let bufs = [
        upload_binary_u32(&fix.memory, &r),
        upload_binary_u32(&fix.memory, &s),
        upload_binary_u32(&fix.memory, &t),
        upload_binary_u32(&fix.memory, &u),
    ];
    let inputs: Vec<&CudaBuffer> = bufs.iter().collect();
    let stream = fix.pool.acquire().expect("stream");

    for (label, plan) in [
        ("natural", chain_plan_natural()),
        ("u_cover", chain_plan_u_cover()),
    ] {
        let out = fix
            .provider
            .free_join_execute_u32_recorded(&inputs, &plan, stream)
            .expect("free join chain");
        let got = download_rows_sorted(&fix.memory, &out);
        assert_eq!(got, expected, "chain ({label}) vs oracle");
    }
}

#[test]
fn fj_star_matches_oracle() {
    let Some(fix) = make_fixture() else {
        eprintln!("skipping fj_star: no CUDA device");
        return;
    };
    let r = sorted_unique([(1, 100), (1, 101), (2, 200), (3, 300)]);
    let s = sorted_unique([(1, 7), (1, 8), (2, 9), (4, 4)]);
    let t = sorted_unique([(1, 70), (2, 90), (2, 91), (5, 5)]);
    let expected = oracle_star(&r, &s, &t);
    assert!(!expected.is_empty(), "fixture must produce rows");

    let bufs = [
        upload_binary_u32(&fix.memory, &r),
        upload_binary_u32(&fix.memory, &s),
        upload_binary_u32(&fix.memory, &t),
    ];
    let inputs: Vec<&CudaBuffer> = bufs.iter().collect();
    let stream = fix.pool.acquire().expect("stream");
    let out = fix
        .provider
        .free_join_execute_u32_recorded(&inputs, &star_plan(), stream)
        .expect("free join star");
    let got = download_rows_sorted(&fix.memory, &out);
    assert_eq!(got, expected, "star vs oracle");
}

#[test]
fn fj_triangle_matches_oracle_and_dedicated_kernel() {
    let Some(fix) = make_fixture() else {
        eprintln!("skipping fj_triangle: no CUDA device");
        return;
    };
    // K4 on {1..4} plus a disjoint triangle {5,6,7} (same shape as the
    // dedicated-kernel cert fixtures).
    let r = sorted_unique([
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
    let s = sorted_unique([(2, 3), (2, 4), (3, 4), (6, 7)]);
    let t = sorted_unique([(1, 3), (1, 4), (2, 4), (3, 4), (5, 7)]);
    let expected = oracle_triangle(&r, &s, &t);
    assert!(!expected.is_empty(), "fixture must contain triangles");

    let bufs = [
        upload_binary_u32(&fix.memory, &r),
        upload_binary_u32(&fix.memory, &s),
        upload_binary_u32(&fix.memory, &t),
    ];
    let inputs: Vec<&CudaBuffer> = bufs.iter().collect();
    let stream = fix.pool.acquire().expect("stream");
    let out = fix
        .provider
        .free_join_execute_u32_recorded(&inputs, &triangle_plan(), stream)
        .expect("free join triangle");
    let got = download_rows_sorted(&fix.memory, &out);
    assert_eq!(got, expected, "triangle vs oracle");

    // Row-set parity vs the dedicated triangle WCOJ kernel.
    let tri = fix
        .provider
        .wcoj_triangle_hg_u32_recorded(
            &bufs[0],
            &bufs[1],
            &bufs[2],
            WCOJ_HG_BLOCK_WORK_UNIT_DEFAULT,
            stream,
        )
        .expect("dedicated triangle");
    let dedicated = download_rows_sorted(&fix.memory, &tri);
    assert_eq!(got, dedicated, "free join vs dedicated triangle kernel");
}

#[test]
fn fj_layout_normalizes_unsorted_inputs() {
    // §2.3: all inputs layout-normalized per dispatch (31b0ccf0
    // contract). Unsorted, duplicated uploads must produce the same
    // row set as their sorted+deduped form.
    let Some(fix) = make_fixture() else {
        eprintln!("skipping fj_layout_normalize: no CUDA device");
        return;
    };
    let r_raw = vec![(7u32, 2u32), (0, 2), (0, 1), (0, 2), (9, 99)];
    let s_raw = vec![(2u32, 12u32), (1, 10), (2, 10), (1, 11), (1, 10)];
    let t_raw = vec![(12u32, 23u32), (10, 20), (11, 22), (10, 21), (10, 20)];
    let u_raw = vec![(23u32, 7u32), (20, 5), (21, 6), (21, 5), (20, 5)];
    let expected = oracle_chain(
        &sorted_unique(r_raw.clone()),
        &sorted_unique(s_raw.clone()),
        &sorted_unique(t_raw.clone()),
        &sorted_unique(u_raw.clone()),
    );
    assert!(!expected.is_empty(), "fixture must produce rows");

    let bufs = [
        upload_binary_u32(&fix.memory, &r_raw),
        upload_binary_u32(&fix.memory, &s_raw),
        upload_binary_u32(&fix.memory, &t_raw),
        upload_binary_u32(&fix.memory, &u_raw),
    ];
    let inputs: Vec<&CudaBuffer> = bufs.iter().collect();
    let stream = fix.pool.acquire().expect("stream");
    let out = fix
        .provider
        .free_join_execute_u32_recorded(&inputs, &chain_plan_natural(), stream)
        .expect("free join unsorted chain");
    let got = download_rows_sorted(&fix.memory, &out);
    assert_eq!(got, expected, "unsorted+duplicated inputs normalized");
}

#[test]
fn fj_empty_result_is_empty_buffer() {
    let Some(fix) = make_fixture() else {
        eprintln!("skipping fj_empty: no CUDA device");
        return;
    };
    // Triangle with no closing T edge: probes kill every row.
    let r = sorted_unique([(1, 2), (4, 5)]);
    let s = sorted_unique([(2, 3), (5, 6)]);
    let t = sorted_unique([(8, 9)]);
    let bufs = [
        upload_binary_u32(&fix.memory, &r),
        upload_binary_u32(&fix.memory, &s),
        upload_binary_u32(&fix.memory, &t),
    ];
    let inputs: Vec<&CudaBuffer> = bufs.iter().collect();
    let stream = fix.pool.acquire().expect("stream");
    let out = fix
        .provider
        .free_join_execute_u32_recorded(&inputs, &triangle_plan(), stream)
        .expect("free join empty triangle");
    assert_eq!(out.arity(), 3, "projected arity preserved");
    let got = download_rows_sorted(&fix.memory, &out);
    assert!(got.is_empty(), "no triangles must yield an empty buffer");
}

#[test]
fn fj_rejects_unbound_probe_vars() {
    let Some(fix) = make_fixture() else {
        eprintln!("skipping fj_rejects: no CUDA device");
        return;
    };
    // Probe on z (var 2) at node 1, before z is bound: invalid plan.
    let bad = FjPlan {
        num_vars: 3,
        nodes: vec![FjNode {
            cover: sub(0, &[0, 1]),
            probes: vec![sub(1, &[2])],
        }],
        output_vars: vec![0, 1],
    };
    let r = sorted_unique([(1, 2)]);
    let s = sorted_unique([(2, 3)]);
    let bufs = [
        upload_binary_u32(&fix.memory, &r),
        upload_binary_u32(&fix.memory, &s),
    ];
    let inputs: Vec<&CudaBuffer> = bufs.iter().collect();
    let stream = fix.pool.acquire().expect("stream");
    let err = fix
        .provider
        .free_join_execute_u32_recorded(&inputs, &bad, stream);
    assert!(err.is_err(), "unbound probe var must be rejected");
}

// =====================================================================
// S2 measurement (gates from design doc §4). Run explicitly:
// `cargo test -p xlog-cuda-tests --test test_free_join_spike --release \
//    -- --ignored --nocapture`
// =====================================================================

/// Blowup chain fixture (cardinalities documented in
/// docs/evidence/2026-06-12-s2-free-join-spike/README.md):
///   R: hub a=0 fanning into 2000 x values            (|R| = 2000)
///   S: complete bipartite x -> 50 y values           (|S| = 100_000)
///   T: each y -> 50 distinct z values (z = y*64+i)   (|T| = 2500)
///   U: 3 b values for the single z* = 7*64 + 3       (|U| = 3)
/// Left-deep binary intermediates: |R join S| = 100_000,
/// |R join S join T| = 5_000_000 >> |Q| = 6000.
fn blowup_chain_fixture() -> (
    Vec<(u32, u32)>,
    Vec<(u32, u32)>,
    Vec<(u32, u32)>,
    Vec<(u32, u32)>,
) {
    let mut r = Vec::new();
    for x in 0..2000u32 {
        r.push((0u32, x));
    }
    let mut s = Vec::new();
    for x in 0..2000u32 {
        for y in 0..50u32 {
            s.push((x, y));
        }
    }
    let mut t = Vec::new();
    for y in 0..50u32 {
        for i in 0..50u32 {
            t.push((y, y * 64 + i));
        }
    }
    let z_star = 7 * 64 + 3;
    let u: Vec<(u32, u32)> = (0..3u32).map(|b| (z_star, b)).collect();
    (
        sorted_unique(r),
        sorted_unique(s),
        sorted_unique(t),
        sorted_unique(u),
    )
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.total_cmp(b));
    samples[samples.len() / 2]
}

#[test]
#[ignore = "S2 measurement: run explicitly with --ignored --nocapture"]
fn s2_measurement_blowup_chain_vs_binary() {
    let Some(fix) = make_fixture_with_budget(2 * 1024 * 1024 * 1024) else {
        eprintln!("skipping s2 blowup: no CUDA device");
        return;
    };
    let (r, s, t, u) = blowup_chain_fixture();
    let expected = oracle_chain(&r, &s, &t, &u);
    println!(
        "S2 blowup fixture: |R|={} |S|={} |T|={} |U|={} |Q|={}",
        r.len(),
        s.len(),
        t.len(),
        u.len(),
        expected.len()
    );

    let bufs = [
        upload_binary_u32(&fix.memory, &r),
        upload_binary_u32(&fix.memory, &s),
        upload_binary_u32(&fix.memory, &t),
        upload_binary_u32(&fix.memory, &u),
    ];
    let inputs: Vec<&CudaBuffer> = bufs.iter().collect();
    let stream = fix.pool.acquire().expect("stream");

    // Production binary baseline: left-deep hash_join_v2 in the
    // lowerer's syntactic body order R, S, T, U (joined columns:
    // R.x=S.x, S.y=T.y, T.z=U.z). hash_join_v2 is the production
    // provider entry the executor's ChainJoin path drives; output
    // keeps all columns of both sides (combine_schemas), which only
    // helps the baseline's comparability since no projection cost is
    // charged to it.
    let run_binary = |sync: bool| -> CudaBuffer {
        let j1 = fix
            .provider
            .hash_join_v2(&bufs[0], &bufs[1], &[1], &[0], JoinType::Inner)
            .expect("R join S");
        // j1 cols: (a, x, x, y)
        let j2 = fix
            .provider
            .hash_join_v2(&j1, &bufs[2], &[3], &[0], JoinType::Inner)
            .expect("RS join T");
        // j2 cols: (a, x, x, y, y, z)
        let j3 = fix
            .provider
            .hash_join_v2(&j2, &bufs[3], &[5], &[0], JoinType::Inner)
            .expect("RST join U");
        if sync {
            fix.provider.device().inner().synchronize().expect("sync");
        }
        j3
    };
    let run_fj = |plan: &FjPlan| -> CudaBuffer {
        let out = fix
            .provider
            .free_join_execute_u32_recorded(&inputs, plan, stream)
            .expect("free join");
        fix.provider.device().inner().synchronize().expect("sync");
        out
    };

    // Warmup + parity.
    let warm_binary = run_binary(true);
    println!(
        "binary baseline intermediate check: |J3| = {}",
        buffer_rows(&fix.memory, &warm_binary)
    );
    {
        // Binary baseline parity: project (a,x,y,z,b) = cols 0,1,3,5,7.
        let cols: Vec<Vec<u32>> = [0usize, 1, 3, 5, 7]
            .iter()
            .map(|&c| download_u32_column(&fix.memory, &warm_binary, c))
            .collect();
        let mut rows: Vec<Vec<u32>> = (0..cols[0].len())
            .map(|i| cols.iter().map(|c| c[i]).collect())
            .collect();
        rows.sort();
        rows.dedup();
        assert_eq!(rows, expected, "binary baseline vs oracle");
    }
    drop(warm_binary);
    let plan_u_cover = chain_plan_u_cover();
    let plan_natural = chain_plan_natural();
    let warm_fj = run_fj(&plan_u_cover);
    assert_eq!(
        download_rows_sorted(&fix.memory, &warm_fj),
        expected,
        "free join (u_cover) vs oracle"
    );
    drop(warm_fj);
    let warm_fj_nat = run_fj(&plan_natural);
    assert_eq!(
        download_rows_sorted(&fix.memory, &warm_fj_nat),
        expected,
        "free join (natural) vs oracle"
    );
    drop(warm_fj_nat);

    const RUNS: usize = 3;
    let mut binary_ms = Vec::with_capacity(RUNS);
    let mut fj_ms = Vec::with_capacity(RUNS);
    let mut fj_nat_ms = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let t0 = std::time::Instant::now();
        let out = run_binary(true);
        binary_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        drop(out);

        let t0 = std::time::Instant::now();
        let out = run_fj(&plan_u_cover);
        fj_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        drop(out);

        let t0 = std::time::Instant::now();
        let out = run_fj(&plan_natural);
        fj_nat_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        drop(out);
    }
    let med_binary = median(&mut binary_ms);
    let med_fj = median(&mut fj_ms);
    let med_fj_nat = median(&mut fj_nat_ms);
    println!(
        "S2 blowup chain: binary median {med_binary:.3} ms, free-join (u_cover) median \
         {med_fj:.3} ms, free-join (natural) median {med_fj_nat:.3} ms, speedup (u_cover) \
         {:.2}x, speedup (natural) {:.2}x [gate: >= 2x]",
        med_binary / med_fj,
        med_binary / med_fj_nat,
    );
}

#[test]
#[ignore = "S2 measurement: run explicitly with --ignored --nocapture"]
fn s2_measurement_triangle_vs_dedicated_kernel() {
    let Some(fix) = make_fixture() else {
        eprintln!("skipping s2 triangle: no CUDA device");
        return;
    };
    // W5.2-style hub triangle: x=0 fans into 10k y values, each y
    // reaches 16 z values, hub closes through 16 x-z edges; plus a
    // 1000-row uniform background.
    let mut e_xy = Vec::new();
    let mut e_yz = Vec::new();
    let mut e_xz = Vec::new();
    for y in 1..=10_000u32 {
        e_xy.push((0u32, y));
        for z in 0..16u32 {
            e_yz.push((y, 1_000_000 + z));
        }
    }
    for z in 0..16u32 {
        e_xz.push((0u32, 1_000_000 + z));
    }
    for i in 0..1000u32 {
        let (a, b, c) = (2_000_000 + i, 3_000_000 + i, 4_000_000 + i);
        e_xy.push((a, b));
        e_yz.push((b, c));
        e_xz.push((a, c));
    }
    let e_xy = sorted_unique(e_xy);
    let e_yz = sorted_unique(e_yz);
    let e_xz = sorted_unique(e_xz);
    let expected = oracle_triangle(&e_xy, &e_yz, &e_xz);

    let bufs = [
        upload_binary_u32(&fix.memory, &e_xy),
        upload_binary_u32(&fix.memory, &e_yz),
        upload_binary_u32(&fix.memory, &e_xz),
    ];
    let inputs: Vec<&CudaBuffer> = bufs.iter().collect();
    let stream = fix.pool.acquire().expect("stream");

    // Dedicated production path: per-dispatch layout normalization
    // (31b0ccf0 contract) + the histogram-guided triangle kernel —
    // the same sequence the executor's triangle dispatch performs.
    let run_dedicated = || -> CudaBuffer {
        let l_xy = fix
            .provider
            .wcoj_layout_u32_recorded(&bufs[0], stream)
            .expect("layout xy");
        let l_yz = fix
            .provider
            .wcoj_layout_u32_recorded(&bufs[1], stream)
            .expect("layout yz");
        let l_xz = fix
            .provider
            .wcoj_layout_u32_recorded(&bufs[2], stream)
            .expect("layout xz");
        let tri = fix
            .provider
            .wcoj_triangle_hg_u32_recorded(
                &l_xy,
                &l_yz,
                &l_xz,
                WCOJ_HG_BLOCK_WORK_UNIT_DEFAULT,
                stream,
            )
            .expect("dedicated triangle");
        fix.provider.device().inner().synchronize().expect("sync");
        tri
    };
    let plan = triangle_plan();
    let run_fj = || -> CudaBuffer {
        let out = fix
            .provider
            .free_join_execute_u32_recorded(&inputs, &plan, stream)
            .expect("free join triangle");
        fix.provider.device().inner().synchronize().expect("sync");
        out
    };

    // Warmup + parity.
    let warm_dedicated = run_dedicated();
    let dedicated_rows = download_rows_sorted(&fix.memory, &warm_dedicated);
    assert_eq!(dedicated_rows, expected, "dedicated vs oracle");
    drop(warm_dedicated);
    let warm_fj = run_fj();
    assert_eq!(
        download_rows_sorted(&fix.memory, &warm_fj),
        expected,
        "free join vs oracle"
    );
    drop(warm_fj);

    const RUNS: usize = 3;
    let mut dedicated_ms = Vec::with_capacity(RUNS);
    let mut fj_ms = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let t0 = std::time::Instant::now();
        let out = run_dedicated();
        dedicated_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        drop(out);

        let t0 = std::time::Instant::now();
        let out = run_fj();
        fj_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        drop(out);
    }
    let med_dedicated = median(&mut dedicated_ms);
    let med_fj = median(&mut fj_ms);
    println!(
        "S2 hub triangle: dedicated median {med_dedicated:.3} ms, free-join median \
         {med_fj:.3} ms, ratio {:.2}x of dedicated [gate: <= 1.2x] \
         (n_xy={}, n_yz={}, n_xz={}, |Q|={})",
        med_fj / med_dedicated,
        e_xy.len(),
        e_yz.len(),
        e_xz.len(),
        expected.len()
    );
}

/// 10x-scale companion to the gate fixture: quantifies how much of the
/// small-fixture ratio is fixed per-node launch/sync overhead (which
/// amortizes with work size) vs algorithmic cost (which does not).
/// Production routing keeps triangles on the dedicated kernel either
/// way (design §3); this measurement informs the gate interpretation,
/// it does not replace the gate.
#[test]
#[ignore = "S2 scale measurement: run explicitly with --ignored --nocapture"]
fn s2_measurement_triangle_at_scale() {
    let Some(fix) = make_fixture() else {
        eprintln!("skipping s2 triangle scale: no CUDA device");
        return;
    };
    let mut e_xy = Vec::new();
    let mut e_yz = Vec::new();
    let mut e_xz = Vec::new();
    for y in 1..=100_000u32 {
        e_xy.push((0u32, y));
        for z in 0..32u32 {
            e_yz.push((y, 1_000_000 + z));
        }
    }
    for z in 0..32u32 {
        e_xz.push((0u32, 1_000_000 + z));
    }
    for i in 0..1000u32 {
        let (a, b, c) = (2_000_000 + i, 3_000_000 + i, 4_000_000 + i);
        e_xy.push((a, b));
        e_yz.push((b, c));
        e_xz.push((a, c));
    }
    let e_xy = sorted_unique(e_xy);
    let e_yz = sorted_unique(e_yz);
    let e_xz = sorted_unique(e_xz);

    let bufs = [
        upload_binary_u32(&fix.memory, &e_xy),
        upload_binary_u32(&fix.memory, &e_yz),
        upload_binary_u32(&fix.memory, &e_xz),
    ];
    let inputs: Vec<&CudaBuffer> = bufs.iter().collect();
    let stream = fix.pool.acquire().expect("stream");

    let run_dedicated = || -> CudaBuffer {
        let l_xy = fix
            .provider
            .wcoj_layout_u32_recorded(&bufs[0], stream)
            .expect("layout xy");
        let l_yz = fix
            .provider
            .wcoj_layout_u32_recorded(&bufs[1], stream)
            .expect("layout yz");
        let l_xz = fix
            .provider
            .wcoj_layout_u32_recorded(&bufs[2], stream)
            .expect("layout xz");
        let tri = fix
            .provider
            .wcoj_triangle_hg_u32_recorded(
                &l_xy,
                &l_yz,
                &l_xz,
                WCOJ_HG_BLOCK_WORK_UNIT_DEFAULT,
                stream,
            )
            .expect("dedicated triangle");
        fix.provider.device().inner().synchronize().expect("sync");
        tri
    };
    let plan = triangle_plan();
    let run_fj = || -> CudaBuffer {
        let out = fix
            .provider
            .free_join_execute_u32_recorded(&inputs, &plan, stream)
            .expect("free join triangle");
        fix.provider.device().inner().synchronize().expect("sync");
        out
    };

    // Warmup + cheap parity (row counts; full row-set parity is locked
    // by the gate fixture's oracle comparison).
    let warm_dedicated = run_dedicated();
    let n_dedicated = buffer_rows(&fix.memory, &warm_dedicated);
    drop(warm_dedicated);
    let warm_fj = run_fj();
    let n_fj = buffer_rows(&fix.memory, &warm_fj);
    drop(warm_fj);
    assert_eq!(n_fj, n_dedicated, "row-count parity at scale");

    const RUNS: usize = 3;
    let mut dedicated_ms = Vec::with_capacity(RUNS);
    let mut fj_ms = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let t0 = std::time::Instant::now();
        let out = run_dedicated();
        dedicated_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        drop(out);

        let t0 = std::time::Instant::now();
        let out = run_fj();
        fj_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        drop(out);
    }
    let med_dedicated = median(&mut dedicated_ms);
    let med_fj = median(&mut fj_ms);
    println!(
        "S2 hub triangle AT SCALE: dedicated median {med_dedicated:.3} ms, free-join \
         median {med_fj:.3} ms, ratio {:.2}x of dedicated [informative] \
         (n_xy={}, n_yz={}, n_xz={}, |Q|={})",
        med_fj / med_dedicated,
        e_xy.len(),
        e_yz.len(),
        e_xz.len(),
        n_fj
    );
}
