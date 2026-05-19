//! v0.6.5 slice 2 — WCOJ 4-cycle benchmark baseline.
//!
//! Bench-only. Compares the gate-controlled GPU 4-cycle WCOJ
//! dispatch against the existing binary-join chain on identical
//! fixtures, across u32 and u64.
//!
//! # Default matrix
//!
//! widths × fixtures × sizes × modes =
//!   {u32, u64} × {uniform, superhub} × {2K} × {Off, Force, Adaptive}
//! = 12 cells.
//!
//! Compact relative to triangle's 37-cell matrix because
//! 4-cycle's binary-join fallback is a 4-input cross-product
//! that scales poorly with row count — 2K rows per relation is
//! the sweet spot for a tractable bench while still exercising
//! the kernel path under non-trivial work.
//!
//! Modes:
//!   * **Off**:      force-off. Binary-join chain only.
//!   * **Force**:    force-on. WCOJ pipeline; classifier bypassed.
//!   * **Adaptive**: adaptive opt-in. Classifier dispatches when
//!                   score ≥ 0.10. uniform → binary; superhub → WCOJ.
//!
//! Each cell pre-runs a correctness check outside the timed
//! region: gate-off vs gate-on row sets must agree, and the
//! 4-cycle dispatch counter must be 0/1 in the right cells.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use cudarc::driver::sys;
use xlog_core::{MemoryBudget, RuntimeConfig, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::CudaBuffer;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_logic::Compiler;
use xlog_runtime::Executor;

const FOUR_CYCLE_SOURCE: &str = "cycle4(W, X, Y, Z) :- e1(W, X), e2(X, Y), e3(Y, Z), e4(Z, W).";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Width {
    U32,
    U64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fixture {
    Uniform,
    Superhub,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Off,
    Force,
    Adaptive,
}

#[inline]
fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn uniform_pairs(seed: u64, rows: u32) -> Vec<(u32, u32)> {
    let mut state = seed;
    let mut out: Vec<(u32, u32)> = Vec::with_capacity(rows as usize);
    let key_range = (rows / 2).max(8);
    for _ in 0..rows {
        let a = (lcg_next(&mut state) as u32) % key_range;
        let b = (lcg_next(&mut state) as u32) % key_range;
        out.push((a, b));
    }
    out.sort();
    out.dedup();
    out
}

fn superhub_pairs(seed: u64, rows: u32) -> Vec<(u32, u32)> {
    let mut state = seed;
    let mut out: Vec<(u32, u32)> = Vec::with_capacity(rows as usize);
    let hub: u32 = 1;
    let tail = (rows / 4).max(8);
    // 75% rows touch hub on col0; remainder is uniform tail.
    let hub_rows = (rows * 3) / 4;
    for _ in 0..hub_rows {
        let v = (lcg_next(&mut state) as u32) % rows;
        out.push((hub, v));
    }
    for _ in 0..tail {
        let a = (lcg_next(&mut state) as u32) % rows + 100;
        let b = (lcg_next(&mut state) as u32) % rows + 100;
        out.push((a, b));
    }
    out.sort();
    out.dedup();
    out
}

fn make_fixture(kind: Fixture, rows: u32) -> [Vec<(u32, u32)>; 4] {
    match kind {
        Fixture::Uniform => [
            uniform_pairs(0xa1, rows),
            uniform_pairs(0xa2, rows),
            uniform_pairs(0xa3, rows),
            uniform_pairs(0xa4, rows),
        ],
        Fixture::Superhub => [
            superhub_pairs(0xb1, rows),
            superhub_pairs(0xb2, rows),
            superhub_pairs(0xb3, rows),
            superhub_pairs(0xb4, rows),
        ],
    }
}

struct DiscardSink;
impl LoggingSink for DiscardSink {
    fn emit(&self, _record: LogRecord) -> Result<(), SinkError> {
        Ok(())
    }
}

struct Provider {
    _device: Arc<CudaDevice>,
    _runtime: Arc<XlogDeviceRuntime>,
    memory: Arc<GpuMemoryManager>,
    provider: Arc<CudaKernelProvider>,
    _pool: Arc<StreamPool>,
}

fn make_provider() -> Option<Provider> {
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let pool = Arc::new(StreamPool::new(Arc::clone(&device), 1024));
    let async_resource: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(
        AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool)),
    );
    let logging: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(LoggingResource::new(
        async_resource,
        Arc::new(DiscardSink) as Arc<dyn LoggingSink>,
    ));
    let budget: Box<dyn DeviceMemoryResource + Send + Sync> =
        Box::new(GlobalDeviceBudget::new(logging, 8 * 1024 * 1024 * 1024));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(8 * 1024 * 1024 * 1024),
        Arc::clone(&runtime),
    ));
    let provider =
        Arc::new(CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory)).ok()?);
    Some(Provider {
        _device: device,
        _runtime: runtime,
        memory,
        provider,
        _pool: pool,
    })
}

fn upload_pairs(memory: &Arc<GpuMemoryManager>, rows: &[(u32, u32)], width: Width) -> CudaBuffer {
    let n = rows.len() as u32;
    let scalar = match width {
        Width::U32 => ScalarType::U32,
        Width::U64 => ScalarType::U64,
    };
    let elem_bytes = match width {
        Width::U32 => 4,
        Width::U64 => 8,
    };
    let bytes_per_col = (n as usize) * elem_bytes;
    let mut col0 = memory.alloc::<u8>(bytes_per_col).expect("alloc col0");
    let mut col1 = memory.alloc::<u8>(bytes_per_col).expect("alloc col1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let dev = memory.device().inner();
    if n > 0 {
        let (c0, c1): (Vec<u8>, Vec<u8>) = match width {
            Width::U32 => (
                rows.iter().flat_map(|&(a, _)| a.to_le_bytes()).collect(),
                rows.iter().flat_map(|&(_, b)| b.to_le_bytes()).collect(),
            ),
            Width::U64 => (
                rows.iter()
                    .flat_map(|&(a, _)| (a as u64).to_le_bytes())
                    .collect(),
                rows.iter()
                    .flat_map(|&(_, b)| (b as u64).to_le_bytes())
                    .collect(),
            ),
        };
        dev.htod_sync_copy_into(&c0, &mut col0).expect("htod c0");
        dev.htod_sync_copy_into(&c1, &mut col1).expect("htod c1");
    }
    dev.htod_sync_copy_into(&[n], &mut d_num_rows)
        .expect("htod n");
    let schema = Schema::new(vec![
        ("col0".to_string(), scalar),
        ("col1".to_string(), scalar),
    ]);
    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into()],
        n as u64,
        d_num_rows,
        schema,
        n,
    )
}

fn build_executor(
    prov: &Provider,
    config: RuntimeConfig,
    plan: &xlog_ir::ExecutionPlan,
    rel_ids: &BTreeMap<String, xlog_core::RelId>,
    inputs: &[Vec<(u32, u32)>; 4],
    width: Width,
) -> Executor {
    let mut executor = Executor::new_with_config(Arc::clone(&prov.provider), config);
    for (name, rel_id) in rel_ids {
        executor.register_relation(*rel_id, name);
    }
    for (idx, name) in ["e1", "e2", "e3", "e4"].iter().enumerate() {
        let buf = upload_pairs(&prov.memory, &inputs[idx], width);
        executor.put_relation(name, buf);
    }
    let _ = plan; // executor reads plan in execute_plan
    executor
}

fn rows_at(buf: &CudaBuffer) -> usize {
    match buf.cached_row_count() {
        Some(c) => c as usize,
        None => {
            let mut n = [0u32; 1];
            unsafe {
                sys::cuMemcpyDtoH_v2(
                    n.as_mut_ptr() as *mut _,
                    *buf.num_rows_device().device_ptr(),
                    std::mem::size_of::<u32>(),
                );
            }
            n[0] as usize
        }
    }
}

fn download_quads(buf: &CudaBuffer, width: Width) -> BTreeSet<(u64, u64, u64, u64)> {
    let n = rows_at(buf);
    if n == 0 {
        return BTreeSet::new();
    }
    let elem_bytes = match width {
        Width::U32 => 4,
        Width::U64 => 8,
    };
    let mut bytes = [
        vec![0u8; n * elem_bytes],
        vec![0u8; n * elem_bytes],
        vec![0u8; n * elem_bytes],
        vec![0u8; n * elem_bytes],
    ];
    for c in 0..4 {
        unsafe {
            sys::cuMemcpyDtoH_v2(
                bytes[c].as_mut_ptr() as *mut _,
                *buf.column(c).unwrap().device_ptr(),
                bytes[c].len(),
            );
        }
    }
    let mut out: BTreeSet<(u64, u64, u64, u64)> = BTreeSet::new();
    for i in 0..n {
        let read = |c: usize| -> u64 {
            match width {
                Width::U32 => {
                    u32::from_le_bytes(bytes[c][i * 4..i * 4 + 4].try_into().unwrap()) as u64
                }
                Width::U64 => u64::from_le_bytes(bytes[c][i * 8..i * 8 + 8].try_into().unwrap()),
            }
        };
        out.insert((read(0), read(1), read(2), read(3)));
    }
    out
}

fn correctness_check(
    prov: &Provider,
    plan: &xlog_ir::ExecutionPlan,
    rel_ids: &BTreeMap<String, xlog_core::RelId>,
    inputs: &[Vec<(u32, u32)>; 4],
    width: Width,
) {
    let off_config = RuntimeConfig::default().with_wcoj_4cycle_dispatch(Some(false));
    let mut exec_off = build_executor(prov, off_config, plan, rel_ids, inputs, width);
    exec_off.execute_plan(plan).expect("execute_plan off");
    assert_eq!(exec_off.wcoj_4cycle_dispatch_count(), 0);
    let off_rows = exec_off
        .store()
        .get("cycle4")
        .map(|b| download_quads(b, width))
        .unwrap_or_default();

    let on_config = RuntimeConfig::default().with_wcoj_4cycle_dispatch(Some(true));
    let mut exec_on = build_executor(prov, on_config, plan, rel_ids, inputs, width);
    exec_on.execute_plan(plan).expect("execute_plan on");
    assert_eq!(exec_on.wcoj_4cycle_dispatch_count(), 1);
    let on_rows = exec_on
        .store()
        .get("cycle4")
        .map(|b| download_quads(b, width))
        .unwrap_or_default();

    assert_eq!(
        off_rows, on_rows,
        "WCOJ 4-cycle bench correctness: gate-on row set must equal gate-off"
    );
}

fn config_for(mode: Mode) -> RuntimeConfig {
    match mode {
        Mode::Off => RuntimeConfig::default().with_wcoj_4cycle_dispatch(Some(false)),
        Mode::Force => RuntimeConfig::default().with_wcoj_4cycle_dispatch(Some(true)),
        Mode::Adaptive => RuntimeConfig::default().with_wcoj_4cycle_dispatch_adaptive(Some(true)),
    }
}

fn bench_cell(
    c: &mut Criterion,
    prov: &Provider,
    width: Width,
    fixture: Fixture,
    mode: Mode,
    rows: u32,
) {
    let inputs = make_fixture(fixture, rows);
    let total_rows: usize = inputs.iter().map(|v| v.len()).sum();

    let mut compiler = Compiler::new();
    let plan = compiler.compile(FOUR_CYCLE_SOURCE).expect("compile");
    let rel_ids: BTreeMap<String, xlog_core::RelId> = compiler
        .rel_ids()
        .iter()
        .map(|(k, v)| (k.clone(), *v))
        .collect();

    correctness_check(prov, &plan, &rel_ids, &inputs, width);

    let label = format!("{:?}/{:?}/{:?}/{}", width, fixture, mode, rows);
    let mut group = c.benchmark_group("wcoj_4cycle");
    group.throughput(Throughput::Elements(total_rows as u64));
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    group.bench_with_input(BenchmarkId::from_parameter(&label), &(), |b, _| {
        let mut executor = build_executor(prov, config_for(mode), &plan, &rel_ids, &inputs, width);
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                executor.store_mut().remove("cycle4");
                let start = Instant::now();
                executor.execute_plan(&plan).expect("execute_plan iter");
                total += start.elapsed();
            }
            total
        });
    });
    group.finish();
}

fn bench_all(c: &mut Criterion) {
    let Some(prov) = make_provider() else {
        eprintln!("Skipping wcoj_4cycle bench: CUDA unavailable");
        return;
    };
    const ROWS: u32 = 2000;
    for &width in &[Width::U32, Width::U64] {
        for &fixture in &[Fixture::Uniform, Fixture::Superhub] {
            for &mode in &[Mode::Off, Mode::Force, Mode::Adaptive] {
                bench_cell(c, &prov, width, fixture, mode, ROWS);
            }
        }
    }
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
