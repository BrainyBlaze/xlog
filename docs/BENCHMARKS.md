# XLOG Performance Benchmarks

This document describes XLOG's performance benchmarking suite, methodology, and baseline metrics.

## Table of Contents

1. [Running Benchmarks](#running-benchmarks)
2. [Benchmark Categories](#benchmark-categories)
3. [Methodology](#methodology)
4. [Baseline Metrics](#baseline-metrics)
5. [Interpreting Results](#interpreting-results)
6. [CI Integration](#ci-integration)
7. [Contributing Benchmarks](#contributing-benchmarks)

---

## Running Benchmarks

### Prerequisites

- CUDA-capable NVIDIA GPU (compute capability 7.0+; development device: RTX PRO 3000 Blackwell, SM120)
- CUDA Toolkit 13.x
- D4 knowledge compiler (for exact inference benchmarks)
- Sufficient GPU memory (4GB minimum, 12GB recommended for neural-symbolic training)

### Quick Start

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark suite
cargo bench -p xlog-gpu    # Transitive closure, joins, aggregation
cargo bench -p xlog-prob   # Exact inference, Monte Carlo
cargo bench -p xlog-stats  # Statistics manager
cargo bench -p xlog-solve  # SAT solver

# Run specific benchmark group
cargo bench -p xlog-gpu -- tc_benches
cargo bench -p xlog-gpu -- join_benches

# Generate HTML report
cargo bench -- --save-baseline baseline_name
cargo bench -- --baseline baseline_name
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CUDA_VISIBLE_DEVICES` | GPU device ordinal | `0` |
| `XLOG_BENCH_MEMORY_MB` | GPU memory budget | `4096` |

---

## Benchmark Categories

### GPU Logic Benchmarks (`xlog-gpu`)

**Location:** `crates/xlog-gpu/benches/logic_bench.rs`

#### Transitive Closure

Tests recursive query evaluation (semi-naive fixpoint iteration).

| Benchmark | Description | Metric |
|-----------|-------------|--------|
| `tc_chain` | Chain graph 0→1→2→...→n | Iteration depth |
| `tc_random` | Random sparse graph | Rows/sec |
| `tc_dense` | Complete bipartite K_{n,n} | Output explosion |

**Parameters:**
- `tc_chain`: depth 100, 500, 1000, 2000
- `tc_random`: 10K, 100K, 1M edges
- `tc_dense`: K_{100,100}, K_{200,200}, K_{500,500}

#### Hash Join Throughput

Tests GPU hash join kernel performance.

| Benchmark | Description | Metric |
|-----------|-------------|--------|
| `join_throughput` | Varying cardinality | Rows/sec |
| `join_selectivity` | Varying key range | Output rows/input |
| `multiway_join` | 3-way join | Intermediate explosion |

**Parameters:**
- Cardinalities: 10Kx10K to 1Mx100K
- Key ranges: 100 (high selectivity) to 100K (low selectivity)
- Multi-way: 10K, 50K, 100K rows per relation

#### Aggregation

Tests GROUP BY with COUNT aggregate.

| Benchmark | Description | Metric |
|-----------|-------------|--------|
| `aggregation` | COUNT by group | Groups/sec |

**Parameters:**
- 100K rows with 1K groups
- 100K rows with 10K groups
- 1M rows with 10K groups
- 1M rows with 100K groups

### Probabilistic Benchmarks (`xlog-prob`)

**Location:** `crates/xlog-prob/benches/prob_bench.rs`

#### Exact Inference (Decision-DNNF)

Tests knowledge compilation and weighted model counting.

| Benchmark | Description | Metric |
|-----------|-------------|--------|
| `exact_path` | Probabilistic path | Circuits/sec |
| `exact_grid` | Probabilistic grid | Cells/sec |
| `exact_bayesian` | Bayesian network | Variables/circuit |
| `exact_gradients` | With gradient computation | Grads/sec |

**Parameters:**
- Path lengths: 5, 10, 15, 20, 25 nodes
- Grid sizes: 3x3, 4x4, 5x5, 6x6
- Bayesian: 10, 20, 30, 50 variables

#### Monte Carlo Inference

Tests GPU-accelerated random sampling.

| Benchmark | Description | Metric |
|-----------|-------------|--------|
| `mc_samples` | Sample count scaling | Samples/sec |
| `mc_vars` | Variable count scaling | Worlds/sec |
| `mc_path` | Probabilistic path | (samples × vars)/sec |
| `mc_grid` | Probabilistic grid | (samples × cells)/sec |
| `mc_bayesian` | Bayesian network | (samples × vars)/sec |

**Parameters:**
- Sample counts: 1K, 5K, 10K, 50K, 100K
- AD counts: 10, 50, 100, 500, 1000
- Path lengths: 10, 25, 50, 100, 200
- Grid sizes: 5x5, 10x10, 15x15, 20x20
- Bayesian: 50, 100, 200, 500 variables

### Statistics Manager Benchmarks (`xlog-stats`)

**Location:** `crates/xlog-stats/benches/stats_bench.rs`

Tests relation registration, cardinality tracking, join estimation.

### Solver Benchmarks (`xlog-solve`)

**Location:** `crates/xlog-solve/benches/solver_bench.rs`

Tests SAT solving, gradient computation, state management.

---

## Methodology

### Measurement Approach

XLOG benchmarks use [Criterion.rs](https://github.com/bheisler/criterion.rs) for statistically rigorous performance measurement.

| Setting | Value | Rationale |
|---------|-------|-----------|
| Sample size | 10-100 | GPU warmup + noise reduction |
| Warm-up | 3 iterations | JIT compilation, caching |
| Significance level | 0.1 | Detect 10% regressions |
| Noise threshold | 0.05 | Ignore <5% variance |

### Throughput Calculation

```
Throughput = Elements / Time

Elements vary by benchmark type:
- Transitive closure: input edges
- Joins: total input rows (left + right)
- Aggregation: input rows
- Exact inference: circuit variables
- MC inference: samples × variables
```

### Warm-up Protocol

GPU benchmarks include warm-up to ensure:
1. PTX modules are compiled and cached
2. Memory pools are initialized
3. CUDA context is established

### Reproducibility

All random data generation uses deterministic seeding:
- LCG with fixed seed (no system entropy)
- Same seed produces identical graphs

---

## Baseline Metrics

Development hardware: **NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU** (12 GB, SM120, compute capability 12.0, driver 591.59).

All baseline targets below are measured on this device. Throughput numbers on desktop-class GPUs (e.g. RTX 4090, RTX 5090) will differ due to higher memory bandwidth and SM count.

### Transitive Closure

| Configuration | Target | Notes |
|---------------|--------|-------|
| 100K random edges | >1M rows/sec | Sparse graph |
| 1M random edges | >5M rows/sec | Medium graph |
| K_{500,500} bipartite | >10M rows/sec | Dense output |

### Hash Join

| Configuration | Target | Notes |
|---------------|--------|-------|
| 100K × 100K | >50M rows/sec | Medium cardinality |
| 1M × 100K | >100M rows/sec | Large left relation |
| High selectivity | >20M rows/sec | Many output rows |

### Exact Inference

| Configuration | Target | Notes |
|---------------|--------|-------|
| 20-variable path | <100ms | Small circuit |
| 50-variable Bayesian | <500ms | Medium complexity |
| With gradients | <2× base | Backward pass overhead |

### Monte Carlo

| Configuration | Target | Notes |
|---------------|--------|-------|
| 100K samples, 100 vars | >10M worlds/sec | Throughput mode |
| 10K samples, 500 vars | >5M worlds/sec | Complexity mode |

### Neural-Symbolic Training (v0.4.0-alpha)

Measured on development hardware with `01_minimal` (MNIST addition, 512 images, 5 epochs, batch_size=64).

| Metric | Value | Notes |
|--------|-------|-------|
| `PTX JIT \(cold\)` | 0.02 s | Cubin loading (1750x speedup from ~35s) |
| `first_epoch_sec` | ~75 s | Cold-start (d4 compile + verify), warm-starts drop to 0.26s |
| `steady_epoch_sec_mean` | ~0.25 s | Epochs 2-5 after warmup (Batched evaluation) |
| `per_query_ms` | ~1.0 ms | Per-query forward+backward through circuit |
| Cache speedup | 2.74x | Circuit caching vs no caching (95% CI: [2.29, 3.18]) |

Evidence: `examples/neural/results/evidence/cache_ablation_20260218.json`

---

## Interpreting Results

### Criterion Output

```
tc_random/edges/100K_edges
                        time:   [12.345 ms 12.456 ms 12.567 ms]
                        thrpt:  [7.9567 Melem/s 8.0283 Melem/s 8.1003 Melem/s]
                 change: [-2.5% -1.2% +0.1%] (p = 0.12 > 0.10)
                        No change in performance detected.
```

| Field | Meaning |
|-------|---------|
| `time` | [lower bound, estimate, upper bound] at 95% CI |
| `thrpt` | Throughput in million elements per second |
| `change` | Comparison vs baseline |
| `p` | Statistical significance |

### Performance Regression Detection

A benchmark is flagged as a regression if:
1. `change` lower bound > +5%
2. `p` < 0.10

### Common Issues

| Symptom | Cause | Solution |
|---------|-------|----------|
| High variance | GPU thermal throttling | Cool-down period |
| First run slow | JIT compilation | Ignore first sample |
| OOM errors | Large input | Reduce memory budget |
| Missing benchmarks | No CUDA device | Check GPU availability |

---

## CI Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/bench.yml
name: Benchmarks

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4

      - name: Run benchmarks
        run: cargo bench --no-fail-fast -- --save-baseline pr_${{ github.sha }}

      - name: Compare to main
        if: github.event_name == 'pull_request'
        run: |
          git fetch origin main
          git checkout origin/main
          cargo bench --no-fail-fast -- --save-baseline main_baseline
          git checkout -
          cargo bench --no-fail-fast -- --baseline main_baseline --load-baseline pr_${{ github.sha }}

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: target/criterion/
```

### Regression Alerts

CI fails if any benchmark shows:
- >10% regression vs main branch
- Statistical significance p < 0.05

### Benchmark History

Historical results are stored in:
- `target/criterion/` (local)
- GitHub Actions artifacts (CI)

---

## Contributing Benchmarks

### Adding a New Benchmark

1. Create benchmark function:

```rust
fn bench_new_feature(c: &mut Criterion) {
    let mut group = c.benchmark_group("new_feature");
    group.sample_size(10);

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("size", size),
            size,
            |b, &size| {
                b.iter(|| {
                    // Benchmark code here
                    black_box(operation(size))
                });
            },
        );
    }

    group.finish();
}
```

2. Add to criterion group:

```rust
criterion_group!(
    name = my_benches;
    config = Criterion::default();
    targets = bench_new_feature
);

criterion_main!(my_benches);
```

3. Add to Cargo.toml:

```toml
[[bench]]
name = "my_bench"
harness = false

[dev-dependencies]
criterion = "0.5"
```

### Benchmark Guidelines

| Guideline | Rationale |
|-----------|-----------|
| Use `black_box()` | Prevent dead code elimination |
| Handle GPU errors gracefully | CI may lack GPU |
| Use deterministic data | Reproducibility |
| Document expected performance | Regression detection |
| Keep sample size reasonable | CI time budget |

### Review Checklist

- [ ] Benchmark measures meaningful operation
- [ ] Throughput metric is appropriate
- [ ] Parameters cover realistic range
- [ ] Handles missing GPU gracefully
- [ ] Documentation updated

---

## See Also

- [Architecture](ARCHITECTURE.md) - System design
- [Roadmap](ROADMAP.md) - Development plans
- [v0.3.x Scope](plans/v0.3.x-scope.md) - Current release scope
