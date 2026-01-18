//! Performance benchmarks for XLOG probabilistic inference.
//!
//! Run with: `cargo bench -p xlog-prob`
//!
//! These benchmarks measure the performance of:
//! - Exact inference via Decision-DNNF (circuits/sec, variables/circuit)
//! - Monte Carlo inference (samples/sec, worlds/sec)
//!
//! Note: These benchmarks require a CUDA-capable GPU and the D4 compiler for exact inference.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use xlog_prob::exact::{ExactDdnnfProgram, GpuConfig};
use xlog_prob::mc::{McEvalConfig, McProgram};

// =============================================================================
// Helper Functions
// =============================================================================

/// Creates GPU config for benchmarks.
fn make_gpu_config(memory_mb: u64) -> GpuConfig {
    GpuConfig {
        device_ordinal: 0,
        memory_bytes: memory_mb * 1024 * 1024,
    }
}

/// Simple LCG random number generator state
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    fn next_prob(&mut self) -> f64 {
        let r = (self.next() % 100) as f64 / 100.0;
        // Clamp to valid probability range
        r.max(0.01).min(0.99)
    }
}

/// Generates a probabilistic path reachability program.
/// Creates a chain of probabilistic edges: e(0,1), e(1,2), ..., e(n-2,n-1)
/// Each edge has probability p.
fn generate_prob_path_program(num_nodes: usize, edge_prob: f64) -> String {
    let mut source = String::new();

    // Probabilistic edges
    for i in 0..num_nodes.saturating_sub(1) {
        source.push_str(&format!("{}::edge({}, {}).\n", edge_prob, i, i + 1));
    }

    // Reachability rules
    source.push_str("\nreach(X, Y) :- edge(X, Y).\n");
    source.push_str("reach(X, Z) :- reach(X, Y), edge(Y, Z).\n");

    // Query reachability from first to last
    source.push_str(&format!("\nquery(reach(0, {})).\n", num_nodes - 1));

    source
}

/// Generates a probabilistic grid program.
/// Creates an n x n grid where each cell has probability p of being active.
/// Queries connectivity from (0,0) to (n-1,n-1) via 4-connectivity.
fn generate_prob_grid_program(grid_size: usize, cell_prob: f64) -> String {
    let mut source = String::new();

    // Probabilistic cells
    for i in 0..grid_size {
        for j in 0..grid_size {
            source.push_str(&format!("{}::active({}, {}).\n", cell_prob, i, j));
        }
    }

    // Connectivity: can move between adjacent active cells
    source.push_str("\nconnected(X1, Y1, X2, Y2) :- active(X1, Y1), active(X2, Y2), X2 is X1 + 1, Y1 = Y2.\n");
    source.push_str("connected(X1, Y1, X2, Y2) :- active(X1, Y1), active(X2, Y2), X1 = X2, Y2 is Y1 + 1.\n");

    // Reachability
    source.push_str("\nreachable(X, Y) :- X = 0, Y = 0, active(0, 0).\n");
    source.push_str("reachable(X2, Y2) :- reachable(X1, Y1), connected(X1, Y1, X2, Y2).\n");

    // Query
    let target = grid_size - 1;
    source.push_str(&format!("\nquery(reachable({}, {})).\n", target, target));

    source
}

/// Generates a Bayesian network style program with specified structure.
fn generate_bayesian_network_program(num_vars: usize, num_edges: usize, seed: u64) -> String {
    let mut source = String::new();
    let mut rng = Lcg::new(seed);

    // Root nodes (first quarter are unconditional)
    let num_roots = (num_vars / 4).max(1);
    for i in 0..num_roots {
        source.push_str(&format!("{:.2}::node({}).\n", rng.next_prob(), i));
    }

    // Intermediate nodes depend on earlier nodes
    for i in num_roots..num_vars {
        let parent = (rng.next() % i as u64) as usize;
        let prob = rng.next_prob();
        source.push_str(&format!("{:.2}::node({}) :- node({}).\n", prob, i, parent));
    }

    // Add some extra edges for complexity
    for _ in 0..(num_edges.saturating_sub(num_vars - num_roots)) {
        let parent = (rng.next() % (num_vars - 1) as u64) as usize;
        let child = ((rng.next() % (num_vars - parent - 1).max(1) as u64) as usize) + parent + 1;
        if child < num_vars {
            let prob = rng.next_prob();
            source.push_str(&format!(
                "{:.2}::node({}) :- node({}).\n",
                prob,
                child,
                parent
            ));
        }
    }

    // Query the last node
    source.push_str(&format!("\nquery(node({})).\n", num_vars - 1));

    source
}

/// Generates a simple probabilistic program with annotated disjunctions.
fn generate_annotated_disjunction_program(num_ads: usize) -> String {
    let mut source = String::new();

    // Create annotated disjunctions
    for i in 0..num_ads {
        source.push_str(&format!(
            "0.3::choice({}, a); 0.3::choice({}, b); 0.4::choice({}, c).\n",
            i, i, i
        ));
    }

    // Derive a result based on choices
    source.push_str("\nresult(I) :- choice(I, a).\n");
    source.push_str("result(I) :- choice(I, b).\n");

    // Count results
    source.push_str("\nhas_result :- result(_).\n");

    source.push_str("\nquery(has_result).\n");

    source
}

// =============================================================================
// Exact Inference Benchmarks
// =============================================================================

fn bench_exact_path(c: &mut Criterion) {
    let config = make_gpu_config(4096);

    let mut group = c.benchmark_group("exact_path");
    group.sample_size(10);

    // Path lengths with varying complexity
    for path_len in [5, 10, 15, 20, 25].iter() {
        let source = generate_prob_path_program(*path_len, 0.8);

        let program = match ExactDdnnfProgram::compile_source_with_gpu(&source, config) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Skipping exact_path {}: {}", path_len, e);
                continue;
            }
        };

        let num_vars = program.num_vars();

        group.throughput(Throughput::Elements(num_vars as u64));
        group.bench_with_input(
            BenchmarkId::new("vars", num_vars),
            &path_len,
            |b, _| {
                b.iter(|| program.evaluate());
            },
        );
    }

    group.finish();
}

fn bench_exact_grid(c: &mut Criterion) {
    let config = make_gpu_config(4096);

    let mut group = c.benchmark_group("exact_grid");
    group.sample_size(10);

    // Small grids (complexity grows quickly)
    for grid_size in [3, 4, 5, 6].iter() {
        let source = generate_prob_grid_program(*grid_size, 0.7);
        let num_cells = grid_size * grid_size;

        let program = match ExactDdnnfProgram::compile_source_with_gpu(&source, config) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Skipping exact_grid {}x{}: {}", grid_size, grid_size, e);
                continue;
            }
        };

        let num_vars = program.num_vars();

        group.throughput(Throughput::Elements(num_cells as u64));
        group.bench_with_input(
            BenchmarkId::new("cells", num_cells),
            &grid_size,
            |b, _| {
                b.iter(|| program.evaluate());
            },
        );

        // Log circuit complexity
        eprintln!("exact_grid {}x{}: {} vars in circuit", grid_size, grid_size, num_vars);
    }

    group.finish();
}

fn bench_exact_bayesian(c: &mut Criterion) {
    let config = make_gpu_config(4096);

    let mut group = c.benchmark_group("exact_bayesian");
    group.sample_size(10);

    // Bayesian networks with varying sizes
    for (label, num_vars, num_edges) in [
        ("small_10v", 10, 15),
        ("medium_20v", 20, 30),
        ("large_30v", 30, 50),
        ("xlarge_50v", 50, 80),
    ]
    .iter()
    {
        let source = generate_bayesian_network_program(*num_vars, *num_edges, 42);

        let program = match ExactDdnnfProgram::compile_source_with_gpu(&source, config) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Skipping exact_bayesian {}: {}", label, e);
                continue;
            }
        };

        let circuit_vars = program.num_vars();

        group.throughput(Throughput::Elements(*num_vars as u64));
        group.bench_with_input(BenchmarkId::new("config", label), &label, |b, _| {
            b.iter(|| program.evaluate());
        });

        // Log circuit size for analysis
        eprintln!("exact_bayesian {}: {} circuit vars", label, circuit_vars);
    }

    group.finish();
}

// =============================================================================
// Monte Carlo Inference Benchmarks
// =============================================================================

fn bench_mc_samples_scaling(c: &mut Criterion) {
    let config = make_gpu_config(4096);

    let mut group = c.benchmark_group("mc_samples");
    group.sample_size(10);

    // Fixed program, varying sample counts
    let source = generate_prob_path_program(20, 0.8);
    let program = match McProgram::compile_source_with_gpu(&source, config) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Skipping mc_samples: {}", e);
            return;
        }
    };

    for num_samples in [1000, 5000, 10000, 50000, 100000].iter() {
        let mc_config = McEvalConfig {
            samples: *num_samples,
            seed: 42,
            confidence: 0.95,
            ..Default::default()
        };

        group.throughput(Throughput::Elements(*num_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", num_samples),
            &num_samples,
            |b, _| {
                b.iter(|| program.evaluate(black_box(mc_config.clone())));
            },
        );
    }

    group.finish();
}

fn bench_mc_vars_scaling(c: &mut Criterion) {
    let config = make_gpu_config(4096);

    let mut group = c.benchmark_group("mc_vars");
    group.sample_size(10);

    // Fixed sample count, varying program complexity
    let mc_config = McEvalConfig {
        samples: 10000,
        seed: 42,
        confidence: 0.95,
        ..Default::default()
    };

    for num_ads in [10, 50, 100, 500, 1000].iter() {
        let source = generate_annotated_disjunction_program(*num_ads);

        let program = match McProgram::compile_source_with_gpu(&source, config) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Skipping mc_vars {}: {}", num_ads, e);
                continue;
            }
        };

        let num_vars = program.num_vars();

        group.throughput(Throughput::Elements(num_vars as u64));
        group.bench_with_input(BenchmarkId::new("vars", num_vars), &num_ads, |b, _| {
            b.iter(|| program.evaluate(black_box(mc_config.clone())));
        });
    }

    group.finish();
}

fn bench_mc_path(c: &mut Criterion) {
    let config = make_gpu_config(4096);

    let mut group = c.benchmark_group("mc_path");
    group.sample_size(10);

    let mc_config = McEvalConfig {
        samples: 10000,
        seed: 42,
        confidence: 0.95,
        ..Default::default()
    };

    for path_len in [10, 25, 50, 100, 200].iter() {
        let source = generate_prob_path_program(*path_len, 0.9);

        let program = match McProgram::compile_source_with_gpu(&source, config) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Skipping mc_path {}: {}", path_len, e);
                continue;
            }
        };

        let num_vars = program.num_vars();

        group.throughput(Throughput::Elements((mc_config.samples * num_vars) as u64));
        group.bench_with_input(BenchmarkId::new("path_len", path_len), &path_len, |b, _| {
            b.iter(|| program.evaluate(black_box(mc_config.clone())));
        });
    }

    group.finish();
}

fn bench_mc_grid(c: &mut Criterion) {
    let config = make_gpu_config(4096);

    let mut group = c.benchmark_group("mc_grid");
    group.sample_size(10);

    let mc_config = McEvalConfig {
        samples: 5000,
        seed: 42,
        confidence: 0.95,
        ..Default::default()
    };

    for grid_size in [5, 10, 15, 20].iter() {
        let source = generate_prob_grid_program(*grid_size, 0.7);

        let program = match McProgram::compile_source_with_gpu(&source, config) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Skipping mc_grid {}x{}: {}", grid_size, grid_size, e);
                continue;
            }
        };

        let num_vars = program.num_vars();
        let num_cells = grid_size * grid_size;

        group.throughput(Throughput::Elements((mc_config.samples * num_cells) as u64));
        group.bench_with_input(BenchmarkId::new("grid", grid_size), &grid_size, |b, _| {
            b.iter(|| program.evaluate(black_box(mc_config.clone())));
        });

        // Log stats
        eprintln!(
            "mc_grid {}x{}: {} vars, {} worlds evaluated",
            grid_size,
            grid_size,
            num_vars,
            mc_config.samples
        );
    }

    group.finish();
}

fn bench_mc_bayesian(c: &mut Criterion) {
    let config = make_gpu_config(4096);

    let mut group = c.benchmark_group("mc_bayesian");
    group.sample_size(10);

    let mc_config = McEvalConfig {
        samples: 10000,
        seed: 42,
        confidence: 0.95,
        ..Default::default()
    };

    for (label, num_vars, num_edges) in [
        ("small_50v", 50, 80),
        ("medium_100v", 100, 160),
        ("large_200v", 200, 350),
        ("xlarge_500v", 500, 800),
    ]
    .iter()
    {
        let source = generate_bayesian_network_program(*num_vars, *num_edges, 42);

        let program = match McProgram::compile_source_with_gpu(&source, config) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Skipping mc_bayesian {}: {}", label, e);
                continue;
            }
        };

        let actual_vars = program.num_vars();

        group.throughput(Throughput::Elements((mc_config.samples * actual_vars) as u64));
        group.bench_with_input(BenchmarkId::new("config", label), &label, |b, _| {
            b.iter(|| program.evaluate(black_box(mc_config.clone())));
        });

        // Log actual complexity
        eprintln!(
            "mc_bayesian {}: {} vars, {} samples = {} world evaluations",
            label,
            actual_vars,
            mc_config.samples,
            mc_config.samples * actual_vars
        );
    }

    group.finish();
}

// =============================================================================
// GPU Gradient Benchmarks (Exact with Gradients)
// =============================================================================

fn bench_exact_gradients(c: &mut Criterion) {
    let config = make_gpu_config(4096);

    let mut group = c.benchmark_group("exact_gradients");
    group.sample_size(10);

    for path_len in [5, 10, 15, 20].iter() {
        let source = generate_prob_path_program(*path_len, 0.8);

        let program = match ExactDdnnfProgram::compile_source_with_gpu(&source, config) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Skipping exact_gradients {}: {}", path_len, e);
                continue;
            }
        };

        let num_vars = program.num_vars();

        group.throughput(Throughput::Elements(num_vars as u64));
        group.bench_with_input(
            BenchmarkId::new("vars", num_vars),
            &path_len,
            |b, _| {
                b.iter(|| program.evaluate_gpu_with_grads());
            },
        );
    }

    group.finish();
}

// =============================================================================
// Criterion Groups
// =============================================================================

criterion_group!(
    name = exact_benches;
    config = Criterion::default().significance_level(0.1).noise_threshold(0.05);
    targets = bench_exact_path, bench_exact_grid, bench_exact_bayesian, bench_exact_gradients
);

criterion_group!(
    name = mc_benches;
    config = Criterion::default().significance_level(0.1).noise_threshold(0.05);
    targets = bench_mc_samples_scaling, bench_mc_vars_scaling, bench_mc_path, bench_mc_grid, bench_mc_bayesian
);

criterion_main!(exact_benches, mc_benches);
