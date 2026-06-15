//! `wcoj_phase_report` — measurement-only report binary.
//!
//! Runs the WCOJ adaptive dispatch on the bench's super-hub
//! 10K/50K × u32/u64 cells, captures per-phase timings (CUDA
//! events for the 4 GPU phases inside the triangle entry,
//! `Instant` for classifier + 3 layouts + wall-clock), prints a
//! markdown table, then applies the locked decision rule:
//!
//!   * triangle_materialize_ms / wall ≥ 0.50 → heavy-row materialization
//!     offload with count threshold ≥ 4
//!   * triangle_count_ms + scan + total ≥ 0.50 * wall → schedule
//!     count/materialize work
//!   * layout_total + classifier + residual ≥ 0.50 * wall →
//!     pipeline overhead first
//!
//! Run with:
//!
//!   cargo run -p xlog-integration --bin wcoj_phase_report \
//!     --features wcoj-phase-timing --release
//!
//! No CI integration. Diagnostic-only. Production builds (no
//! `wcoj-phase-timing` feature) cannot build this binary.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use xlog_core::{MemoryBudget, RelId, RuntimeConfig, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::CudaBuffer;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_logic::Compiler;
use xlog_runtime::executor::wcoj_phase_timing::WcojDispatchPhaseTiming;
use xlog_runtime::Executor;
use xlog_stats::{
    ColumnStats, JoinSelectivity, KeyHeatStats, PrefixDegreeStats, RelationStats, StatsSnapshot,
};

const SOURCE: &str = "tri(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z).";
const RECURSIVE_K5_HISTOGRAM_SOURCE: &str = r#"
    pred seed01(u32, u32).
    pred path01(u32, u32).
    pred e02(u32, u32). pred e03(u32, u32). pred e04(u32, u32).
    pred e12(u32, u32). pred e13(u32, u32). pred e14(u32, u32).
    pred e23(u32, u32). pred e24(u32, u32).
    pred e34(u32, u32).
    pred clique5(u32, u32, u32, u32, u32).

    path01(A, B) :- seed01(A, B).
    clique5(A, B, C, D, E) :-
        path01(A, B), e02(A, C), e03(A, D), e04(A, E),
        e12(B, C), e13(B, D), e14(B, E),
        e23(C, D), e24(C, E),
        e34(D, E).
    path01(A, C) :- clique5(A, B, C, D, E).
"#;
/// Per-cell sample count. Each iteration uploads inputs, runs
/// `execute_plan`, reads timing. We discard the first N as
/// warmup, take median of the rest.
const ITERS: usize = 30;
const WARMUP: usize = 5;

#[derive(Clone, Copy, Debug)]
enum Width {
    U32,
    U64,
}
impl Width {
    fn label(self) -> &'static str {
        match self {
            Width::U32 => "u32",
            Width::U64 => "u64",
        }
    }
    fn scalar(self) -> ScalarType {
        match self {
            Width::U32 => ScalarType::U32,
            Width::U64 => ScalarType::U64,
        }
    }
    fn bytes(self) -> usize {
        match self {
            Width::U32 => 4,
            Width::U64 => 8,
        }
    }
}

#[inline]
fn lcg_next(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
    *s
}

fn dedup(mut v: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    v.sort();
    v.dedup();
    v
}

fn superhub_xy(seed: u64, rows: u32, kr: u32, hub_y: u32) -> Vec<(u32, u32)> {
    let mut s = seed;
    (0..rows)
        .map(|i| {
            let a = (lcg_next(&mut s) % kr as u64) as u32;
            let b = if i % 2 == 0 {
                hub_y
            } else {
                (lcg_next(&mut s) % kr as u64) as u32
            };
            (a, b)
        })
        .collect()
}

fn superhub_first(seed: u64, rows: u32, kr: u32, hub_first: u32) -> Vec<(u32, u32)> {
    let mut s = seed;
    (0..rows)
        .map(|i| {
            let a = if i % 2 == 0 {
                hub_first
            } else {
                (lcg_next(&mut s) % kr as u64) as u32
            };
            let b = (lcg_next(&mut s) % kr as u64) as u32;
            (a, b)
        })
        .collect()
}

fn build_superhub_inputs(rows: u32) -> [Vec<(u32, u32)>; 3] {
    let kr = (rows / 10).max(1000);
    [
        dedup(superhub_xy(101, rows, kr, 7)),
        dedup(superhub_first(202, rows, kr, 7)),
        dedup(superhub_first(303, rows, kr, 13)),
    ]
}

fn upload(memory: &Arc<GpuMemoryManager>, rows: &[(u32, u32)], width: Width) -> CudaBuffer {
    let n = rows.len() as u32;
    let bpc = (n as usize) * width.bytes();
    let mut col0 = memory.alloc::<u8>(bpc).expect("alloc c0");
    let mut col1 = memory.alloc::<u8>(bpc).expect("alloc c1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc nr");
    let device = memory.device().inner();
    if n > 0 {
        match width {
            Width::U32 => {
                let c0: Vec<u8> = rows.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
                let c1: Vec<u8> = rows.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
                device.htod_sync_copy_into(&c0, &mut col0).expect("htod c0");
                device.htod_sync_copy_into(&c1, &mut col1).expect("htod c1");
            }
            Width::U64 => {
                let c0: Vec<u8> = rows
                    .iter()
                    .flat_map(|(a, _)| (*a as u64).to_le_bytes())
                    .collect();
                let c1: Vec<u8> = rows
                    .iter()
                    .flat_map(|(_, b)| (*b as u64).to_le_bytes())
                    .collect();
                device.htod_sync_copy_into(&c0, &mut col0).expect("htod c0");
                device.htod_sync_copy_into(&c1, &mut col1).expect("htod c1");
            }
        }
    }
    device
        .htod_sync_copy_into(&[n], &mut d_num_rows)
        .expect("htod nr");
    let schema = Schema::new(vec![
        ("col0".to_string(), width.scalar()),
        ("col1".to_string(), width.scalar()),
    ]);
    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into()],
        n as u64,
        d_num_rows,
        schema,
        n,
    )
}

struct DiscardSink;
impl LoggingSink for DiscardSink {
    fn emit(&self, _r: LogRecord) -> Result<(), SinkError> {
        Ok(())
    }
}

struct Fix {
    memory: Arc<GpuMemoryManager>,
    provider: Arc<CudaKernelProvider>,
}

fn make_fix() -> Option<Fix> {
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let pool = Arc::new(StreamPool::new(Arc::clone(&device), 1024));
    let limit_bytes: usize = 8 * 1024 * 1024 * 1024;
    let async_resource: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(
        AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool)),
    );
    let logging: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(LoggingResource::new(
        async_resource,
        Arc::new(DiscardSink) as Arc<dyn LoggingSink>,
    ));
    let budget: Box<dyn DeviceMemoryResource + Send + Sync> =
        Box::new(GlobalDeviceBudget::new(logging, limit_bytes));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(limit_bytes as u64),
        Arc::clone(&runtime),
    ));
    let provider =
        Arc::new(CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory)).ok()?);
    Some(Fix { memory, provider })
}

fn median_of(values: &[f32]) -> f32 {
    let mut v: Vec<f32> = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

#[derive(Default)]
struct CellMedians {
    classifier_ms: f32,
    layout_xy_ms: f32,
    layout_yz_ms: f32,
    layout_xz_ms: f32,
    layout_total_ms: f32,
    triangle_count_ms: f32,
    triangle_scan_ms: f32,
    triangle_total_ms: f32,
    triangle_materialize_ms: f32,
    triangle_gpu_total_ms: f32,
    execute_plan_wall_ms: f32,
    residual_overhead_ms: f32,
    samples: usize,
}

#[derive(Default)]
struct KCliqueHistogramReport {
    dispatch_count: u64,
    merge_refresh_count: u64,
    merge_refresh_nanos: u128,
    metadata_build_count: u64,
    metadata_build_nanos: u64,
    execute_plan_wall_nanos: u128,
    output_rows: u64,
}

fn measure_cell(fix: &Fix, rows: u32, width: Width) -> CellMedians {
    let inputs_host = build_superhub_inputs(rows);

    let mut compiler = Compiler::new();
    let plan = compiler.compile(SOURCE).expect("compile");
    // Adaptive (default-on) — same path the production runtime
    // uses. Force-on bypasses the classifier; we want the full
    // measured path including classifier overhead.
    let config = RuntimeConfig::default();
    let mut executor = Executor::new_with_config(Arc::clone(&fix.provider), config);
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }

    let mut samples: Vec<WcojDispatchPhaseTiming> = Vec::with_capacity(ITERS);
    for i in 0..ITERS {
        executor.put_relation("e1", upload(&fix.memory, &inputs_host[0], width));
        executor.put_relation("e2", upload(&fix.memory, &inputs_host[1], width));
        executor.put_relation("e3", upload(&fix.memory, &inputs_host[2], width));
        executor.execute_plan(&plan).expect("execute_plan");
        if let Some(t) = executor.take_wcoj_phase_timing() {
            if i >= WARMUP {
                samples.push(t);
            }
        } else {
            eprintln!(
                "WARN: iter {i} on {}-{}K produced no phase timing — classifier rejected the fixture or dispatch fell back. Recheck fixture: super-hub should always trigger adaptive dispatch.",
                width.label(),
                rows / 1000
            );
        }
        let _ = executor.store_mut().remove("tri");
    }

    let pull = |get: fn(&WcojDispatchPhaseTiming) -> f32| -> f32 {
        let v: Vec<f32> = samples.iter().map(get).collect();
        median_of(&v)
    };

    CellMedians {
        classifier_ms: pull(|t| t.classifier_ms),
        layout_xy_ms: pull(|t| t.layout_xy_ms),
        layout_yz_ms: pull(|t| t.layout_yz_ms),
        layout_xz_ms: pull(|t| t.layout_xz_ms),
        layout_total_ms: pull(|t| t.layout_total_ms),
        triangle_count_ms: pull(|t| t.triangle_count_ms),
        triangle_scan_ms: pull(|t| t.triangle_scan_ms),
        triangle_total_ms: pull(|t| t.triangle_total_ms),
        triangle_materialize_ms: pull(|t| t.triangle_materialize_ms),
        triangle_gpu_total_ms: pull(|t| t.triangle_gpu_total_ms),
        execute_plan_wall_ms: pull(|t| t.execute_plan_wall_ms),
        residual_overhead_ms: pull(|t| t.residual_overhead_ms),
        samples: samples.len(),
    }
}

fn recursive_k5_inputs() -> BTreeMap<&'static str, Vec<(u32, u32)>> {
    BTreeMap::from([
        ("seed01", vec![(1, 2)]),
        ("e02", vec![(1, 3), (1, 4), (1, 5)]),
        ("e03", vec![(1, 4), (1, 5), (1, 6)]),
        ("e04", vec![(1, 5), (1, 6), (1, 7)]),
        ("e12", vec![(2, 3), (3, 4), (4, 5)]),
        ("e13", vec![(2, 4), (3, 5), (4, 6)]),
        ("e14", vec![(2, 5), (3, 6), (4, 7)]),
        ("e23", vec![(3, 4), (4, 5), (5, 6)]),
        ("e24", vec![(3, 5), (4, 6), (5, 7)]),
        ("e34", vec![(4, 5), (5, 6), (6, 7)]),
    ])
}

fn recursive_k5_stats(rel_ids: &BTreeMap<String, RelId>) -> StatsSnapshot {
    let mut snapshot = StatsSnapshot::default();
    let canonical_edges = [
        ("path01", 0u8, 1u8),
        ("e02", 0, 2),
        ("e03", 0, 3),
        ("e04", 0, 4),
        ("e12", 1, 2),
        ("e13", 1, 3),
        ("e14", 1, 4),
        ("e23", 2, 3),
        ("e24", 2, 4),
        ("e34", 3, 4),
    ];
    for name in [
        "seed01", "path01", "e02", "e03", "e04", "e12", "e13", "e14", "e23", "e24", "e34",
    ] {
        let rel = *rel_ids.get(name).expect("relation id");
        snapshot.rel_names.push((rel, name.to_string()));
        let mut stats = RelationStats::new(rel);
        stats.update_cardinality(if name == "seed01" { 1 } else { 2_005 });
        for col_idx in [0usize, 1usize] {
            let mut col = ColumnStats::new(col_idx, ScalarType::U32);
            col.update_distinct(1_005);
            stats.add_column(col);
            stats.add_prefix_degree(PrefixDegreeStats::new(col_idx, 2.0, 2.5));
            stats.add_key_heat(KeyHeatStats::new(col_idx, 0.75, 0.75));
        }
        snapshot.relations.push(stats);
    }
    for (left_idx, (left_name, left_i, left_j)) in canonical_edges.iter().enumerate() {
        let left_rel = *rel_ids.get(*left_name).expect("left relation id");
        for (right_name, right_i, right_j) in canonical_edges.iter().skip(left_idx + 1) {
            if left_i == right_i || left_i == right_j || left_j == right_i || left_j == right_j {
                let right_rel = *rel_ids.get(*right_name).expect("right relation id");
                let mut sel = JoinSelectivity::new(left_rel, right_rel);
                sel.set_keys(vec![0], vec![0]);
                sel.set_selectivity(0.001);
                snapshot.join_selectivities.push(sel);
            }
        }
    }
    snapshot
}

fn measure_recursive_k5_histogram_refresh(fix: &Fix) -> KCliqueHistogramReport {
    let inputs = recursive_k5_inputs();
    let mut id_compiler = Compiler::new();
    let _ = id_compiler
        .compile(RECURSIVE_K5_HISTOGRAM_SOURCE)
        .expect("compile recursive k5 ids");
    let rel_ids: BTreeMap<String, RelId> = id_compiler
        .rel_ids()
        .iter()
        .map(|(name, rel)| (name.clone(), *rel))
        .collect();
    let snapshot = recursive_k5_stats(&rel_ids);

    let mut compiler = Compiler::new();
    let plan = compiler
        .compile_with_stats_snapshot(RECURSIVE_K5_HISTOGRAM_SOURCE, Some(&snapshot))
        .expect("compile recursive k5");
    let mut executor =
        Executor::new_with_config(Arc::clone(&fix.provider), RuntimeConfig::default());
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    for (name, rows) in inputs {
        executor.put_relation(name, upload(&fix.memory, &rows, Width::U32));
    }

    fix.provider.reset_kclique_metadata_build_metrics();
    let wall = Instant::now();
    executor.execute_plan(&plan).expect("execute recursive k5");
    let execute_plan_wall_nanos = wall.elapsed().as_nanos();
    let output_rows = executor
        .store()
        .get("path01")
        .and_then(|buf| buf.cached_row_count())
        .unwrap_or(0) as u64;

    KCliqueHistogramReport {
        dispatch_count: executor.wcoj_clique5_dispatch_count(),
        merge_refresh_count: executor.kclique_histogram_refresh_count(),
        merge_refresh_nanos: executor.kclique_histogram_refresh_nanos(),
        metadata_build_count: fix.provider.kclique_metadata_build_count(),
        metadata_build_nanos: fix.provider.kclique_metadata_build_nanos(),
        execute_plan_wall_nanos,
        output_rows,
    }
}

#[derive(Debug, Clone, Copy)]
enum Verdict {
    /// triangle_materialize_ms / wall ≥ 0.50 → heavy-row materialization offload.
    HeavyRowMaterializationWarranted,
    /// count + scan + total ≥ 0.50 * wall → scheduler.
    SchedulerWarranted,
    /// layout + classifier + residual ≥ 0.50 * wall → pipeline overhead.
    PipelineOverheadWarranted,
    /// None of the above clear 0.50 → no clear winner.
    Inconclusive,
}

fn classify(c: &CellMedians) -> Verdict {
    let wall = c.execute_plan_wall_ms;
    if wall <= 0.0 {
        return Verdict::Inconclusive;
    }
    let mat_share = c.triangle_materialize_ms / wall;
    let csm_share = (c.triangle_count_ms + c.triangle_scan_ms + c.triangle_total_ms) / wall;
    let pipe_share = (c.layout_total_ms + c.classifier_ms + c.residual_overhead_ms) / wall;

    if mat_share >= 0.50 {
        Verdict::HeavyRowMaterializationWarranted
    } else if csm_share >= 0.50 {
        Verdict::SchedulerWarranted
    } else if pipe_share >= 0.50 {
        Verdict::PipelineOverheadWarranted
    } else {
        Verdict::Inconclusive
    }
}

fn print_cell(label: &str, c: &CellMedians) {
    let v = classify(c);
    let wall = c.execute_plan_wall_ms.max(1e-6);
    let pct = |x: f32| (x / wall) * 100.0;
    println!("### `{label}` (median over {} samples)", c.samples);
    println!();
    println!("| bucket | ms | % wall |");
    println!("|---|---:|---:|");
    println!(
        "| classifier | {:.3} | {:.1}% |",
        c.classifier_ms,
        pct(c.classifier_ms)
    );
    println!(
        "| layout_xy | {:.3} | {:.1}% |",
        c.layout_xy_ms,
        pct(c.layout_xy_ms)
    );
    println!(
        "| layout_yz | {:.3} | {:.1}% |",
        c.layout_yz_ms,
        pct(c.layout_yz_ms)
    );
    println!(
        "| layout_xz | {:.3} | {:.1}% |",
        c.layout_xz_ms,
        pct(c.layout_xz_ms)
    );
    println!(
        "| **layout_total** | **{:.3}** | **{:.1}%** |",
        c.layout_total_ms,
        pct(c.layout_total_ms)
    );
    println!(
        "| triangle_count | {:.3} | {:.1}% |",
        c.triangle_count_ms,
        pct(c.triangle_count_ms)
    );
    println!(
        "| triangle_scan | {:.3} | {:.1}% |",
        c.triangle_scan_ms,
        pct(c.triangle_scan_ms)
    );
    println!(
        "| triangle_total | {:.3} | {:.1}% |",
        c.triangle_total_ms,
        pct(c.triangle_total_ms)
    );
    println!(
        "| triangle_materialize | {:.3} | {:.1}% |",
        c.triangle_materialize_ms,
        pct(c.triangle_materialize_ms)
    );
    println!(
        "| **triangle_gpu_total** | **{:.3}** | **{:.1}%** |",
        c.triangle_gpu_total_ms,
        pct(c.triangle_gpu_total_ms)
    );
    println!(
        "| residual_overhead | {:.3} | {:.1}% |",
        c.residual_overhead_ms,
        pct(c.residual_overhead_ms)
    );
    println!(
        "| **wall** | **{:.3}** | **100.0%** |",
        c.execute_plan_wall_ms
    );
    println!();
    println!("Verdict: **{v:?}**");
    println!();
}

fn print_kclique_histogram_refresh_report(c: &KCliqueHistogramReport) {
    let wall = c.execute_plan_wall_nanos.max(1);
    let pct = |nanos: u128| (nanos as f64 / wall as f64) * 100.0;
    let avg_merge_refresh = if c.merge_refresh_count == 0 {
        0.0
    } else {
        c.merge_refresh_nanos as f64 / c.merge_refresh_count as f64
    };
    let avg_metadata_build = if c.metadata_build_count == 0 {
        0.0
    } else {
        c.metadata_build_nanos as f64 / c.metadata_build_count as f64
    };

    println!("## K-clique Histogram Refresh");
    println!();
    println!("Synthetic recursive K=5 fixture measuring runtime histogram refresh overhead.");
    println!();
    println!("| bucket | raw | ns | ns/call | % wall |");
    println!("|---|---:|---:|---:|---:|");
    println!(
        "| recursive_k5_dispatch | {} | 0 | 0.0 | 0.0% |",
        c.dispatch_count
    );
    println!(
        "| merge_histogram_refresh | {} | {} | {:.1} | {:.3}% |",
        c.merge_refresh_count,
        c.merge_refresh_nanos,
        avg_merge_refresh,
        pct(c.merge_refresh_nanos)
    );
    println!(
        "| leader_metadata_build | {} | {} | {:.1} | {:.3}% |",
        c.metadata_build_count,
        c.metadata_build_nanos,
        avg_metadata_build,
        pct(c.metadata_build_nanos as u128)
    );
    println!(
        "| execute_plan_wall | 1 | {} | {}.0 | 100.000% |",
        c.execute_plan_wall_nanos, c.execute_plan_wall_nanos
    );
    println!();
    println!("Output rows: {}", c.output_rows);
    println!();
}

fn main() {
    println!("# WCOJ Phase Timing Report\n");
    println!(
        "Decision rule (locked): if any cell shows ≥50% wall in **materialize**, heavy-row materialization offload is warranted (count threshold ≥ 4, dual-grid). If ≥50% in count+scan+total, design count/materialize scheduling instead. If ≥50% in layout+classifier+residual, optimize pipeline overhead before kernel scheduling.\n"
    );
    println!(
        "Each cell runs {ITERS} iterations on a single Executor (default-on adaptive); first {WARMUP} discarded as warmup. Median is reported per bucket.\n"
    );

    let Some(fix) = make_fix() else {
        eprintln!("FATAL: no CUDA device — cannot run phase report");
        std::process::exit(1);
    };

    let cells: Vec<(&str, u32, Width)> = vec![
        ("superhub-u32-10K", 10_000, Width::U32),
        ("superhub-u32-50K", 50_000, Width::U32),
        ("superhub-u64-10K", 10_000, Width::U64),
        ("superhub-u64-50K", 50_000, Width::U64),
    ];

    let mut verdicts: BTreeMap<&'static str, Verdict> = BTreeMap::new();
    for (label, rows, width) in &cells {
        let c = measure_cell(&fix, *rows, *width);
        print_cell(label, &c);
        verdicts.insert(label, classify(&c));
    }

    let kclique_hist = measure_recursive_k5_histogram_refresh(&fix);
    print_kclique_histogram_refresh_report(&kclique_hist);

    println!("---\n");
    println!("## Cross-cell verdict\n");
    let mut counts = BTreeMap::<String, u32>::new();
    for (_, v) in verdicts.iter() {
        *counts.entry(format!("{v:?}")).or_insert(0) += 1;
    }
    for (k, v) in &counts {
        println!("- {k}: {v} cells");
    }
    println!();
    println!(
        "If majority verdict is `HeavyRowMaterializationWarranted`, proceed with heavy-row materialization offload (count threshold ≥ 4, dual-grid deterministic materialization). If `SchedulerWarranted`, design count/materialize work scheduling. If `PipelineOverheadWarranted`, optimize launch/alloc/dispatch overhead. `Inconclusive` means no single bucket clears 50% — split investment or measure more sizes."
    );
}
