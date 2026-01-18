#![allow(clippy::arc_with_non_send_sync)]
//! Performance benchmarks for XLOG GPU-accelerated logic evaluation.
//!
//! Run with: `cargo bench -p xlog-gpu`
//!
//! These benchmarks measure the performance of:
//! - Transitive closure (reachability) at various edge counts
//! - Hash join throughput at various cardinalities
//!
//! Note: These benchmarks require a CUDA-capable GPU.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::collections::HashMap;
use std::sync::Arc;

use xlog_core::{MemoryBudget, Schema, ScalarType};
use xlog_cuda::{CudaBuffer, CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_gpu::logic::LogicProgram;

// =============================================================================
// Helper Functions
// =============================================================================

/// Creates a CUDA kernel provider with sufficient memory for benchmarks.
fn make_provider(memory_mb: u64) -> Option<Arc<CudaKernelProvider>> {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(_) => return None,
    };
    let memory = Arc::new(GpuMemoryManager::new(
        device.clone(),
        MemoryBudget::with_limit(memory_mb * 1024 * 1024),
    ));
    match CudaKernelProvider::new(device, memory) {
        Ok(p) => Some(Arc::new(p)),
        Err(_) => None,
    }
}

/// Generates a chain graph: 0->1->2->...->n-1.
/// This is a sparse graph that stresses transitive closure iteration count.
fn generate_chain_graph(n: u32) -> Vec<(u32, u32)> {
    (0..n.saturating_sub(1)).map(|i| (i, i + 1)).collect()
}

/// Generates a random graph with specified number of edges.
/// Uses a simple LCG for reproducibility.
fn generate_random_graph(num_nodes: u32, num_edges: u32, seed: u64) -> Vec<(u32, u32)> {
    let mut edges = Vec::with_capacity(num_edges as usize);
    let mut rng_state = seed;
    let mut next_rand = || {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        rng_state
    };

    for _ in 0..num_edges {
        let src = (next_rand() % num_nodes as u64) as u32;
        let dst = (next_rand() % num_nodes as u64) as u32;
        edges.push((src, dst));
    }
    edges
}

/// Generates a complete bipartite graph K_{n,n}.
/// Total edges = n * n.
fn generate_complete_bipartite(n: u32) -> Vec<(u32, u32)> {
    let mut edges = Vec::with_capacity((n * n) as usize);
    for i in 0..n {
        for j in 0..n {
            edges.push((i, n + j));
        }
    }
    edges
}

/// Creates a CudaBuffer containing edge data from a list of (src, dst) pairs.
fn create_edge_buffer(
    provider: &CudaKernelProvider,
    edges: &[(u32, u32)],
) -> xlog_core::Result<CudaBuffer> {
    let schema = Schema::new(vec![
        ("col_0".to_string(), ScalarType::U32),
        ("col_1".to_string(), ScalarType::U32),
    ]);

    let mut col0: Vec<u8> = Vec::with_capacity(edges.len() * 4);
    let mut col1: Vec<u8> = Vec::with_capacity(edges.len() * 4);

    for (src, dst) in edges {
        col0.extend_from_slice(&src.to_le_bytes());
        col1.extend_from_slice(&dst.to_le_bytes());
    }

    provider.create_buffer_from_slices(&[&col0, &col1], schema)
}

// =============================================================================
// Transitive Closure Benchmarks
// =============================================================================

/// Transitive closure program source.
const TC_PROGRAM: &str = r#"
pred edge(u32, u32).
pred reach(u32, u32).

reach(X, Y) :- edge(X, Y).
reach(X, Z) :- reach(X, Y), edge(Y, Z).

?- reach(X, Y).
"#;

fn bench_transitive_closure_chain(c: &mut Criterion) {
    let Some(provider) = make_provider(4096) else {
        eprintln!("Skipping bench_transitive_closure_chain: No CUDA device");
        return;
    };

    let mut group = c.benchmark_group("tc_chain");
    group.sample_size(10);

    // Chain graphs: measure iteration depth handling
    for chain_len in [100, 500, 1000, 2000].iter() {
        let edges = generate_chain_graph(*chain_len);
        let program = LogicProgram::compile(TC_PROGRAM).expect("compile TC program");

        group.throughput(Throughput::Elements(*chain_len as u64));
        group.bench_with_input(
            BenchmarkId::new("depth", chain_len),
            &edges,
            |b, edges| {
                b.iter_batched(
                    || {
                        let edge_buf = create_edge_buffer(&provider, edges).expect("create edge buffer");
                        let mut inputs = HashMap::new();
                        inputs.insert("edge".to_string(), edge_buf);
                        inputs
                    },
                    |inputs| {
                        program.evaluate(black_box(provider.clone()), black_box(inputs))
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn bench_transitive_closure_random(c: &mut Criterion) {
    let Some(provider) = make_provider(8192) else {
        eprintln!("Skipping bench_transitive_closure_random: No CUDA device");
        return;
    };

    let mut group = c.benchmark_group("tc_random");
    group.sample_size(10);

    // Random sparse graphs at varying scales
    // Using edge counts: 10K, 100K, 1M (targeting different GPU utilization levels)
    for (label, num_nodes, num_edges) in [
        ("10K_edges", 1000, 10_000),
        ("100K_edges", 5000, 100_000),
        ("1M_edges", 20000, 1_000_000),
    ]
    .iter()
    {
        let edges = generate_random_graph(*num_nodes, *num_edges, 42);
        let program = LogicProgram::compile(TC_PROGRAM).expect("compile TC program");

        group.throughput(Throughput::Elements(*num_edges as u64));
        group.bench_with_input(BenchmarkId::new("edges", label), &edges, |b, edges| {
            b.iter_batched(
                || {
                    match create_edge_buffer(&provider, edges) {
                        Ok(buf) => {
                            let mut inputs = HashMap::new();
                            inputs.insert("edge".to_string(), buf);
                            Some(inputs)
                        }
                        Err(_) => None,
                    }
                },
                |inputs| {
                    if let Some(inputs) = inputs {
                        program.evaluate(black_box(provider.clone()), black_box(inputs))
                    } else {
                        Ok(xlog_gpu::logic::LogicEvalResult {
                            queries: Vec::new(),
                            stats: None,
                        })
                    }
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn bench_transitive_closure_dense(c: &mut Criterion) {
    let Some(provider) = make_provider(8192) else {
        eprintln!("Skipping bench_transitive_closure_dense: No CUDA device");
        return;
    };

    let mut group = c.benchmark_group("tc_dense");
    group.sample_size(10);

    // Dense bipartite graphs (worst case for join output size)
    for n in [100, 200, 500].iter() {
        let edges = generate_complete_bipartite(*n);
        let num_edges = edges.len() as u64;
        let program = LogicProgram::compile(TC_PROGRAM).expect("compile TC program");

        group.throughput(Throughput::Elements(num_edges));
        group.bench_with_input(BenchmarkId::new("bipartite", n), &edges, |b, edges| {
            b.iter_batched(
                || {
                    match create_edge_buffer(&provider, edges) {
                        Ok(buf) => {
                            let mut inputs = HashMap::new();
                            inputs.insert("edge".to_string(), buf);
                            Some(inputs)
                        }
                        Err(_) => None,
                    }
                },
                |inputs| {
                    if let Some(inputs) = inputs {
                        program.evaluate(black_box(provider.clone()), black_box(inputs))
                    } else {
                        Ok(xlog_gpu::logic::LogicEvalResult {
                            queries: Vec::new(),
                            stats: None,
                        })
                    }
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// =============================================================================
// Hash Join Throughput Benchmarks
// =============================================================================

/// Simple join program for measuring raw join throughput.
const JOIN_PROGRAM: &str = r#"
pred left(u32, u32).
pred right(u32, u32).
pred result(u32, u32, u32).

result(A, B, C) :- left(A, B), right(B, C).

?- result(A, B, C).
"#;

fn bench_hash_join_throughput(c: &mut Criterion) {
    let Some(provider) = make_provider(8192) else {
        eprintln!("Skipping bench_hash_join_throughput: No CUDA device");
        return;
    };

    let mut group = c.benchmark_group("join_throughput");
    group.sample_size(10);

    // Test various cardinality combinations
    // Format: (left_rows, right_rows, join_selectivity_factor)
    for (label, left_rows, right_rows) in [
        ("10Kx10K", 10_000u32, 10_000u32),
        ("100Kx10K", 100_000, 10_000),
        ("10Kx100K", 10_000, 100_000),
        ("100Kx100K", 100_000, 100_000),
        ("1Mx10K", 1_000_000, 10_000),
        ("1Mx100K", 1_000_000, 100_000),
    ]
    .iter()
    {
        // Generate left relation: (key, value) where key is uniformly distributed
        let key_range = 10_000u32; // Controls selectivity
        let left_edges: Vec<(u32, u32)> = {
            let mut rng_state = 123u64;
            (0..*left_rows)
                .map(|i| {
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((rng_state % key_range as u64) as u32, i)
                })
                .collect()
        };

        let right_edges: Vec<(u32, u32)> = {
            let mut rng_state = 456u64;
            (0..*right_rows)
                .map(|i| {
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((rng_state % key_range as u64) as u32, i)
                })
                .collect()
        };

        let program = LogicProgram::compile(JOIN_PROGRAM).expect("compile join program");
        let total_input_rows = (*left_rows + *right_rows) as u64;

        group.throughput(Throughput::Elements(total_input_rows));
        group.bench_with_input(
            BenchmarkId::new("cardinality", label),
            &(left_edges, right_edges),
            |b, (left_edges, right_edges)| {
                b.iter_batched(
                    || {
                        let left_buf = create_edge_buffer(&provider, left_edges).ok()?;
                        let right_buf = create_edge_buffer(&provider, right_edges).ok()?;
                        let mut inputs = HashMap::new();
                        inputs.insert("left".to_string(), left_buf);
                        inputs.insert("right".to_string(), right_buf);
                        Some(inputs)
                    },
                    |inputs| {
                        if let Some(inputs) = inputs {
                            program.evaluate(black_box(provider.clone()), black_box(inputs))
                        } else {
                            Ok(xlog_gpu::logic::LogicEvalResult {
                                queries: Vec::new(),
                                stats: None,
                            })
                        }
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn bench_hash_join_selectivity(c: &mut Criterion) {
    let Some(provider) = make_provider(8192) else {
        eprintln!("Skipping bench_hash_join_selectivity: No CUDA device");
        return;
    };

    let mut group = c.benchmark_group("join_selectivity");
    group.sample_size(10);

    let base_rows = 100_000u32;

    // Test different join selectivities by varying key range
    // Smaller key range = more matches per key = higher selectivity
    for (label, key_range) in [
        ("high_sel_100", 100u32),      // Very high selectivity
        ("med_sel_1K", 1_000),         // Medium selectivity
        ("low_sel_10K", 10_000),       // Low selectivity
        ("very_low_100K", 100_000),    // Very low selectivity
    ]
    .iter()
    {
        let left_edges: Vec<(u32, u32)> = {
            let mut rng_state = 789u64;
            (0..base_rows)
                .map(|i| {
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((rng_state % *key_range as u64) as u32, i)
                })
                .collect()
        };

        let right_edges: Vec<(u32, u32)> = {
            let mut rng_state = 101112u64;
            (0..base_rows)
                .map(|i| {
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((rng_state % *key_range as u64) as u32, i)
                })
                .collect()
        };

        let program = LogicProgram::compile(JOIN_PROGRAM).expect("compile join program");

        group.throughput(Throughput::Elements((base_rows * 2) as u64));
        group.bench_with_input(
            BenchmarkId::new("key_range", label),
            &(left_edges, right_edges),
            |b, (left_edges, right_edges)| {
                b.iter_batched(
                    || {
                        let left_buf = create_edge_buffer(&provider, left_edges).ok()?;
                        let right_buf = create_edge_buffer(&provider, right_edges).ok()?;
                        let mut inputs = HashMap::new();
                        inputs.insert("left".to_string(), left_buf);
                        inputs.insert("right".to_string(), right_buf);
                        Some(inputs)
                    },
                    |inputs| {
                        if let Some(inputs) = inputs {
                            program.evaluate(black_box(provider.clone()), black_box(inputs))
                        } else {
                            Ok(xlog_gpu::logic::LogicEvalResult {
                                queries: Vec::new(),
                                stats: None,
                            })
                        }
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// =============================================================================
// Multi-way Join Benchmarks
// =============================================================================

/// Three-way join program.
const THREE_WAY_JOIN_PROGRAM: &str = r#"
pred r1(u32, u32).
pred r2(u32, u32).
pred r3(u32, u32).
pred result(u32, u32, u32, u32).

result(A, B, C, D) :- r1(A, B), r2(B, C), r3(C, D).

?- result(A, B, C, D).
"#;

fn bench_multiway_join(c: &mut Criterion) {
    let Some(provider) = make_provider(8192) else {
        eprintln!("Skipping bench_multiway_join: No CUDA device");
        return;
    };

    let mut group = c.benchmark_group("multiway_join");
    group.sample_size(10);

    for (label, rows_per_rel) in [
        ("10K_each", 10_000u32),
        ("50K_each", 50_000),
        ("100K_each", 100_000),
    ]
    .iter()
    {
        let key_range = 1000u32;

        let make_relation = |seed: u64| -> Vec<(u32, u32)> {
            let mut rng_state = seed;
            (0..*rows_per_rel)
                .map(|_| {
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let a = (rng_state % key_range as u64) as u32;
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let b = (rng_state % key_range as u64) as u32;
                    (a, b)
                })
                .collect()
        };

        let r1_data = make_relation(111);
        let r2_data = make_relation(222);
        let r3_data = make_relation(333);

        let program = LogicProgram::compile(THREE_WAY_JOIN_PROGRAM).expect("compile 3-way join");
        let total_rows = (*rows_per_rel * 3) as u64;

        group.throughput(Throughput::Elements(total_rows));
        group.bench_with_input(
            BenchmarkId::new("rows", label),
            &(r1_data, r2_data, r3_data),
            |b, (r1_data, r2_data, r3_data)| {
                b.iter_batched(
                    || {
                        let r1_buf = create_edge_buffer(&provider, r1_data).ok()?;
                        let r2_buf = create_edge_buffer(&provider, r2_data).ok()?;
                        let r3_buf = create_edge_buffer(&provider, r3_data).ok()?;
                        let mut inputs = HashMap::new();
                        inputs.insert("r1".to_string(), r1_buf);
                        inputs.insert("r2".to_string(), r2_buf);
                        inputs.insert("r3".to_string(), r3_buf);
                        Some(inputs)
                    },
                    |inputs| {
                        if let Some(inputs) = inputs {
                            program.evaluate(black_box(provider.clone()), black_box(inputs))
                        } else {
                            Ok(xlog_gpu::logic::LogicEvalResult {
                                queries: Vec::new(),
                                stats: None,
                            })
                        }
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// =============================================================================
// Aggregation Benchmarks
// =============================================================================

/// Aggregation program for counting.
const AGGREGATE_PROGRAM: &str = r#"
pred edge(u32, u32).
pred degree(u32, i64).

degree(X, count(Y)) :- edge(X, Y).

?- degree(X, C).
"#;

fn bench_aggregation(c: &mut Criterion) {
    let Some(provider) = make_provider(4096) else {
        eprintln!("Skipping bench_aggregation: No CUDA device");
        return;
    };

    let mut group = c.benchmark_group("aggregation");
    group.sample_size(10);

    for (label, num_edges, num_groups) in [
        ("100K_1K_groups", 100_000u32, 1_000u32),
        ("100K_10K_groups", 100_000, 10_000),
        ("1M_10K_groups", 1_000_000, 10_000),
        ("1M_100K_groups", 1_000_000, 100_000),
    ]
    .iter()
    {
        let edges: Vec<(u32, u32)> = {
            let mut rng_state = 42u64;
            (0..*num_edges)
                .map(|i| {
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((rng_state % *num_groups as u64) as u32, i)
                })
                .collect()
        };

        let program = LogicProgram::compile(AGGREGATE_PROGRAM).expect("compile aggregate program");

        group.throughput(Throughput::Elements(*num_edges as u64));
        group.bench_with_input(BenchmarkId::new("config", label), &edges, |b, edges| {
            b.iter_batched(
                || {
                    let edge_buf = create_edge_buffer(&provider, edges).ok()?;
                    let mut inputs = HashMap::new();
                    inputs.insert("edge".to_string(), edge_buf);
                    Some(inputs)
                },
                |inputs| {
                    if let Some(inputs) = inputs {
                        program.evaluate(black_box(provider.clone()), black_box(inputs))
                    } else {
                        Ok(xlog_gpu::logic::LogicEvalResult {
                            queries: Vec::new(),
                            stats: None,
                        })
                    }
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// =============================================================================
// Criterion Groups
// =============================================================================

criterion_group!(
    name = tc_benches;
    config = Criterion::default().significance_level(0.1).noise_threshold(0.05);
    targets = bench_transitive_closure_chain, bench_transitive_closure_random, bench_transitive_closure_dense
);

criterion_group!(
    name = join_benches;
    config = Criterion::default().significance_level(0.1).noise_threshold(0.05);
    targets = bench_hash_join_throughput, bench_hash_join_selectivity, bench_multiway_join
);

criterion_group!(
    name = agg_benches;
    config = Criterion::default().significance_level(0.1).noise_threshold(0.05);
    targets = bench_aggregation
);

criterion_main!(tc_benches, join_benches, agg_benches);
