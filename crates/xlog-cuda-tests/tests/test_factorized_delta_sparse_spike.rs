//! D3 sparse-domain spike — parity + bench gate.
//!
//! Design: `docs/plans/2026-06-14-d3-sparse-domain-spike.md`. The
//! hash-set novel step must match the CPU oracle on irregular fixtures
//! (incl. ids far over the dense 2^16 cap), and the `#[ignore]` gate
//! measures it against the dense bitvector path on a large-domain
//! high-multiplicity fixture. Bench is RunPod-only (no local perf).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::sys;
use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamId, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::{CudaBuffer, CudaColumn};
use xlog_cuda::provider::FjDeltaCols;
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
    Some(Fixture { memory, provider })
}

fn make_fixture() -> Option<Fixture> {
    make_fixture_with_budget(1024 * 1024 * 1024)
}

fn upload(memory: &Arc<GpuMemoryManager>, rows: &[(u32, u32)]) -> CudaBuffer {
    let n = rows.len() as u32;
    let mut col0 = memory.alloc::<u8>(rows.len() * 4).expect("col0");
    let mut col1 = memory.alloc::<u8>(rows.len() * 4).expect("col1");
    let mut d_nr = memory.alloc::<u32>(1).expect("nr");
    let b0: Vec<u8> = rows.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
    let b1: Vec<u8> = rows.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
    let dev = memory.device().inner();
    dev.htod_sync_copy_into(&b0, &mut col0).expect("h0");
    dev.htod_sync_copy_into(&b1, &mut col1).expect("h1");
    dev.htod_sync_copy_into(&[n], &mut d_nr).expect("hnr");
    let schema = Schema::new(vec![
        ("c0".to_string(), ScalarType::U32),
        ("c1".to_string(), ScalarType::U32),
    ]);
    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into()],
        n as u64,
        d_nr,
        schema,
        n,
    )
}

fn rows_of(memory: &Arc<GpuMemoryManager>, buf: &CudaBuffer) -> usize {
    buf.cached_row_count()
        .map(|n| n as usize)
        .unwrap_or_else(|| {
            let mut h = [0u32; 1];
            memory
                .device()
                .inner()
                .dtoh_sync_copy_into(buf.num_rows_device(), &mut h)
                .expect("nr");
            h[0] as usize
        })
}

fn download_set(memory: &Arc<GpuMemoryManager>, buf: &CudaBuffer) -> Vec<(u32, u32)> {
    let n = rows_of(memory, buf);
    if n == 0 {
        return Vec::new();
    }
    let col = |i: usize| -> Vec<u32> {
        let mut bytes = vec![0u8; n * 4];
        let CudaColumn::Owned(c) = buf.column(i).expect("col") else {
            panic!("owned");
        };
        unsafe {
            let r =
                sys::cuMemcpyDtoH_v2(bytes.as_mut_ptr() as *mut _, *c.device_ptr(), bytes.len());
            assert_eq!(r, sys::cudaError_enum::CUDA_SUCCESS);
        }
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let (c0, c1) = (col(0), col(1));
    let mut v: Vec<(u32, u32)> = c0.into_iter().zip(c1).collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// Layout-normalize edge key-first (the sparse entry requires it).
fn normalize(fix: &Fixture, edge: &CudaBuffer) -> CudaBuffer {
    fix.provider
        .wcoj_layout_u32_recorded(edge, StreamId::DEFAULT)
        .expect("normalize edge")
}

fn oracle_novel(delta: &[(u32, u32)], edge: &[(u32, u32)], r: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let rs: BTreeSet<(u32, u32)> = r.iter().copied().collect();
    let mut succ: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();
    for &(y, z) in edge {
        succ.entry(y).or_default().insert(z);
    }
    let mut novel = BTreeSet::new();
    for &(x, y) in delta {
        if let Some(zs) = succ.get(&y) {
            for &z in zs {
                if !rs.contains(&(x, z)) {
                    novel.insert((x, z));
                }
            }
        }
    }
    novel.into_iter().collect()
}

/// Large-id irregular fixture — ids span far over the dense 2^16 cap,
/// so this case can only be handled by the sparse path.
fn large_id_edges() -> Vec<(u32, u32)> {
    const B: u32 = 1 << 20;
    vec![
        (B, B + 1),
        (B + 1, B + 2),
        (B + 2, B),
        (3 * B, 4 * B),
        (3 * B, 5 * B),
        (4 * B, 6 * B),
        (5 * B, 6 * B),
        (6 * B, 7 * B),
        (9 * B, 9 * B),
    ]
}

#[test]
fn sparse_novel_step_matches_oracle_large_ids() {
    let Some(fix) = make_fixture() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    const B: u32 = 1 << 20;
    let edge = large_id_edges();
    let delta = vec![
        (B, B + 1),
        (B, B + 1),
        (B + 2, B),
        (3 * B, 5 * B),
        (7 * B, 7 * B),
    ];
    let full_r = vec![(B, B + 2), (3 * B, 6 * B)];
    let expected = oracle_novel(&delta, &edge, &full_r);
    assert!(!expected.is_empty(), "fixture must yield novel rows");

    let edge_norm = normalize(&fix, &upload(&fix.memory, &edge));
    let novel = fix
        .provider
        .fj_delta_sparse_novel_u32_recorded(
            &upload(&fix.memory, &delta),
            &edge_norm,
            &upload(&fix.memory, &full_r),
            FjDeltaCols::CANONICAL,
            0,
            StreamId::DEFAULT,
        )
        .expect("sparse novel step")
        .expect("not budget-declined");
    assert_eq!(download_set(&fix.memory, &novel), expected);
}

#[test]
fn sparse_novel_empty_and_saturated() {
    let Some(fix) = make_fixture() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let edge_norm = normalize(&fix, &upload(&fix.memory, &[(1, 2)]));
    let n0 = fix
        .provider
        .fj_delta_sparse_novel_u32_recorded(
            &upload(&fix.memory, &[]),
            &edge_norm,
            &upload(&fix.memory, &[(1, 2)]),
            FjDeltaCols::CANONICAL,
            0,
            StreamId::DEFAULT,
        )
        .expect("empty delta")
        .expect("not budget-declined");
    assert_eq!(rows_of(&fix.memory, &n0), 0);

    let n1 = fix
        .provider
        .fj_delta_sparse_novel_u32_recorded(
            &upload(&fix.memory, &[(0, 1)]),
            &edge_norm,
            &upload(&fix.memory, &[(0, 2)]),
            FjDeltaCols::CANONICAL,
            0,
            StreamId::DEFAULT,
        )
        .expect("saturated")
        .expect("not budget-declined");
    assert_eq!(rows_of(&fix.memory, &n1), 0);
}

/// Cross-check the sparse hash-set path against the dense bitvector
/// path on a small dense fixture: both must produce the same novel set.
#[test]
fn sparse_matches_dense_on_small_fixture() {
    let Some(fix) = make_fixture() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let edge: Vec<(u32, u32)> = vec![(0, 1), (1, 2), (2, 0), (1, 3), (3, 4), (0, 4)];
    let delta = vec![(5, 0), (5, 1), (6, 2), (5, 0)];
    let full_r = vec![(5, 2), (6, 0)];
    let edge_norm = normalize(&fix, &upload(&fix.memory, &edge));

    let sparse = fix
        .provider
        .fj_delta_sparse_novel_u32_recorded(
            &upload(&fix.memory, &delta),
            &edge_norm,
            &upload(&fix.memory, &full_r),
            FjDeltaCols::CANONICAL,
            0,
            StreamId::DEFAULT,
        )
        .expect("sparse")
        .expect("not budget-declined");
    let dense = fix
        .provider
        .fj_delta_novel_u32_recorded(
            &upload(&fix.memory, &delta),
            &edge_norm,
            &upload(&fix.memory, &full_r),
            FjDeltaCols::CANONICAL,
            64,
            StreamId::DEFAULT,
        )
        .expect("dense");
    assert_eq!(
        download_set(&fix.memory, &sparse),
        download_set(&fix.memory, &dense),
        "sparse and dense novel sets must agree"
    );
}

// ---------------------------------------------------------------------------
// Bench gate (#[ignore], RunPod only): high-multiplicity large-domain
// fixture where the dense bitvector is infeasible. The sparse hash-set
// novel step (one iteration) vs a legacy materialize+sort+dedup+diff
// reference built from the same provider primitives.

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Hub fixture: `h` hub nodes each with `f` out-edges to a shared
/// large-id sink block; `x` source nodes each pointing (in delta) at
/// every hub. One semi-naive step then produces |x|*|sink| distinct
/// novel pairs with multiplicity = #hubs — a witness blowup over a
/// domain far above 2^16.
fn hub_fixture(n_x: u32, n_hub: u32, n_sink: u32) -> (Vec<(u32, u32)>, Vec<(u32, u32)>) {
    const HUB_BASE: u32 = 1 << 20;
    const SINK_BASE: u32 = 1 << 22;
    let mut edge = Vec::with_capacity((n_hub * n_sink) as usize);
    for h in 0..n_hub {
        for s in 0..n_sink {
            edge.push((HUB_BASE + h, SINK_BASE + s));
        }
    }
    let mut delta = Vec::with_capacity((n_x * n_hub) as usize);
    for x in 0..n_x {
        for h in 0..n_hub {
            delta.push((x, HUB_BASE + h));
        }
    }
    (delta, edge)
}

#[test]
#[ignore = "S(sparse) gate — run on RunPod, never locally"]
fn sparse_gate_hub_blowup() {
    // n_x=512 sources × n_hub=64 hubs × n_sink=512 sink =>
    // candidates(witnesses) = 512*64*512 = 16.78M, distinct novel =
    // 512*512 = 262,144 (multiplicity 64). Domain ~2^22 (dense
    // bitvector would need 2^44/8 bytes — infeasible).
    let (delta_v, edge_v) = hub_fixture(512, 64, 512);
    let distinct_novel = 512usize * 512;
    let witnesses = 512usize * 64 * 512;
    eprintln!(
        "sparse gate hub: |delta|={} |edge|={} witnesses={witnesses} distinct_novel={distinct_novel}",
        delta_v.len(),
        edge_v.len()
    );

    let budget = 12u64 * 1024 * 1024 * 1024;
    let Some(fix) = make_fixture_with_budget(budget) else {
        eprintln!("skipping sparse gate: no CUDA device");
        return;
    };
    let edge_norm = normalize(&fix, &upload(&fix.memory, &edge_v));
    let empty_r: Vec<(u32, u32)> = Vec::new();

    const REPS: usize = 3;
    let mut sparse_ms = Vec::new();
    let mut sparse_peak = Vec::new();
    let mut legacy_ms = Vec::new();
    let mut legacy_peak = Vec::new();
    let mut sparse_rows = 0usize;
    let mut legacy_rows = 0usize;

    // Warm-up.
    let _ = fix.provider.fj_delta_sparse_novel_u32_recorded(
        &upload(&fix.memory, &delta_v),
        &edge_norm,
        &upload(&fix.memory, &empty_r),
        FjDeltaCols::CANONICAL,
        0,
        StreamId::DEFAULT,
    );

    for rep in 0..REPS {
        // Sparse factorized novel step.
        let delta_buf = upload(&fix.memory, &delta_v);
        let r_buf = upload(&fix.memory, &empty_r);
        fix.memory.reset_peak();
        let t0 = Instant::now();
        let novel = fix
            .provider
            .fj_delta_sparse_novel_u32_recorded(
                &delta_buf,
                &edge_norm,
                &r_buf,
                FjDeltaCols::CANONICAL,
                0,
                StreamId::DEFAULT,
            )
            .expect("sparse novel")
            .expect("not budget-declined");
        let sdt = t0.elapsed().as_secs_f64() * 1000.0;
        let sp = fix.memory.peak_bytes();
        sparse_rows = rows_of(&fix.memory, &novel);
        drop(novel);

        // Legacy reference: materialize the join (hash_join_v2) then
        // dedup → distinct (x,z). R is empty, so novel == distinct
        // candidates; this is the witness-materializing baseline.
        let delta_buf2 = upload(&fix.memory, &delta_v);
        fix.memory.reset_peak();
        let t1 = Instant::now();
        let joined = fix
            .provider
            .hash_join_v2(
                &delta_buf2,
                &edge_norm,
                &[1],
                &[0],
                xlog_cuda::JoinType::Inner,
            )
            .expect("hash join");
        // project (x, z) = columns (0, 3) then dedup.
        let proj = fix
            .provider
            .wcoj_project_output_columns_recorded(
                &joined,
                &[0, 3],
                Schema::new(vec![
                    ("x".to_string(), ScalarType::U32),
                    ("z".to_string(), ScalarType::U32),
                ]),
                StreamId::DEFAULT,
            )
            .expect("project");
        let deduped = fix.provider.dedup_full_row(&proj).expect("dedup");
        let ldt = t1.elapsed().as_secs_f64() * 1000.0;
        let lp = fix.memory.peak_bytes();
        legacy_rows = rows_of(&fix.memory, &deduped);

        eprintln!(
            "sparse gate rep {rep}: sparse {sdt:.1} ms / {:.1} MiB ; legacy {ldt:.1} ms / {:.1} MiB",
            sp as f64 / (1024.0 * 1024.0),
            lp as f64 / (1024.0 * 1024.0)
        );
        sparse_ms.push(sdt);
        sparse_peak.push(sp as f64);
        legacy_ms.push(ldt);
        legacy_peak.push(lp as f64);
    }

    assert_eq!(sparse_rows, distinct_novel, "sparse novel count");
    assert_eq!(legacy_rows, distinct_novel, "legacy novel count (parity)");

    let sm = median(&mut sparse_ms);
    let sp = median(&mut sparse_peak);
    let lm = median(&mut legacy_ms);
    let lp = median(&mut legacy_peak);
    eprintln!(
        "SPARSE gate hub: distinct_novel={distinct_novel} witnesses={witnesses} | \
         sparse {sm:.1} ms / {:.1} MiB ; legacy {lm:.1} ms / {:.1} MiB | \
         peak {:.2}x  wall-clock {:.3}x  (gate: peak<1.0x at wall<=1.2x)",
        sp / (1024.0 * 1024.0),
        lp / (1024.0 * 1024.0),
        lp / sp.max(1.0),
        sm / lm.max(1.0),
    );
}
