# XLOG

[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![CUDA Tests](https://img.shields.io/badge/CUDA%20tests-140%2F140-brightgreen.svg)](docs/certification/2026-01-14-cuda-certification-results.md)

**XLOG** is a GPU-accelerated Datalog query engine. It compiles declarative logic programs into optimized relational plans and executes them on NVIDIA GPUs, achieving high throughput for recursive queries, graph analytics, and probabilistic inference.

---

## Features

| Category | Capabilities |
|----------|--------------|
| **Datalog** | Rules, facts, recursion (semi-naive), stratified negation, aggregation |
| **Arithmetic** | Comparisons, `is` expressions, builtins (`abs`, `min`, `max`, `pow`, `cast`) |
| **GPU Operators** | Hash joins, radix sort, filter, dedup, union, difference, groupby |
| **Float Predicates** | IEEE 754 total ordering for `f32`/`f64` (`NaN > Inf > nums > +0 > -0 > -Inf`) |
| **Probabilistic** | Exact inference (knowledge compilation), Monte Carlo sampling |
| **Interop** | Arrow IPC, DLPack (zero-copy), Python bindings |
| **Profiling** | `--stats` flag for per-stratum/per-operation timing, memory tracking |

---

## Installation

### Requirements

- Linux (x86_64)
- NVIDIA GPU with compute capability **sm_70+** (Volta or newer)
- CUDA Toolkit 12.x
- Rust 1.75+ (for building from source)

### Build from Source

```bash
git clone https://github.com/anthropics/xlog.git
cd xlog
cargo build --release
```

The `xlog` CLI binary will be at `target/release/xlog`.

### Python Package

```bash
cd crates/xlog-gpu-py
pip install maturin
maturin develop --release
```

---

## Quick Start

### Example: Transitive Closure

Create a file `reachability.xlog`:

```prolog
% Declare predicates with types
pred edge(u32, u32).
pred reach(u32, u32).

% Facts: a small graph
edge(1, 2).
edge(2, 3).
edge(3, 4).
edge(4, 5).

% Rules: transitive closure
reach(X, Y) :- edge(X, Y).
reach(X, Z) :- reach(X, Y), edge(Y, Z).

% Query: what can node 1 reach?
?- reach(1, N).
```

### Run with CLI

```bash
# Run the program
xlog run reachability.xlog

# Output:
# reach(1, N):
# | N |
# |---|
# | 2 |
# | 3 |
# | 4 |
# | 5 |
```

### Run with Rust API

```rust
use xlog_gpu::LogicProgram;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let source = r#"
        pred edge(u32, u32).
        pred reach(u32, u32).

        edge(1, 2). edge(2, 3). edge(3, 4).

        reach(X, Y) :- edge(X, Y).
        reach(X, Z) :- reach(X, Y), edge(Y, Z).

        ?- reach(1, N).
    "#;

    let program = LogicProgram::compile(source)?;
    let results = program.run()?;

    for (name, buffer) in results {
        println!("{}: {} rows", name, buffer.num_rows());
    }
    Ok(())
}
```

### Run with Python

```python
import xlog_gpu

source = """
pred edge(u32, u32).
pred reach(u32, u32).

edge(1, 2). edge(2, 3). edge(3, 4).

reach(X, Y) :- edge(X, Y).
reach(X, Z) :- reach(X, Y), edge(Y, Z).

?- reach(1, N).
"""

program = xlog_gpu.LogicProgram.compile(source)
results = program.evaluate()

# Results are DLPack tensors (zero-copy GPU data)
for name, capsule in results.items():
    # Convert to PyTorch, CuPy, or any DLPack-compatible library
    import torch
    tensor = torch.from_dlpack(capsule)
    print(f"{name}: {tensor}")
```

---

## Probabilistic Inference

XLOG supports probabilistic Datalog with two inference engines:

### Exact Inference (Knowledge Compilation)

```prolog
% Probabilistic facts (Bernoulli random variables)
0.3::rain.
0.7::sprinkler.

% Deterministic rules
wet :- rain.
wet :- sprinkler.

% Evidence and query
evidence(sprinkler, false).
query(wet).
```

```bash
xlog prob weather.xlog --prob-engine exact_ddnnf
# P(wet | not sprinkler) = 0.3
```

### Monte Carlo Sampling

```bash
xlog prob weather.xlog --prob-engine mc --samples 10000
# P(wet) ≈ 0.301 ± 0.009 (95% CI)
```

---

## Language Overview

### Facts and Rules

```prolog
% Facts: ground atoms
parent(alice, bob).
parent(bob, charlie).

% Rules: head :- body
grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
```

### Negation (Stratified)

```prolog
has_child(X) :- parent(X, _).
childless(X) :- person(X), not has_child(X).
```

### Arithmetic

```prolog
% Comparisons
adult(X) :- person(X, Age), Age >= 18.

% Computed values with 'is'
double(X, Y) :- number(X), Y is X * 2.

% Builtins: abs, min, max, pow, cast
distance(X, Y, D) :- point(X, A), point(Y, B), D is abs(A - B).
```

### Aggregation

```prolog
% Count, sum, min, max
degree(X, COUNT(Y)) :- edge(X, Y).
total_weight(SUM(W)) :- edge(_, _, W).
```

### Types

```prolog
pred node(u32).
pred edge(u32, u32).
pred weight(u32, u32, f64).
pred label(u32, symbol).  % symbol = interned string (reversible)
```

Supported types: `u32`, `u64`, `i32`, `i64`, `f32`, `f64`, `bool`, `symbol`

### Modules and Functions (v0.3.2)

Organize code into reusable modules with `use` imports and define custom functions with arithmetic, conditionals, and recursion. Symbol values display as readable strings for improved debugging.

```prolog
% Module example (math.xlog)
func abs(X) = if X < 0 then 0 - X else X.

% Main program
use math::{abs}.

pred task(symbol, f64).
task(temperature, -5.0).

pred result(symbol, f64).
result(Label, AbsVal) :- task(Label, Val), abs(Val) is AbsVal.

?- result(X, Y).
% Output: result(temperature, 5.0).
```

**Features:**

- **Modules**: Define functions in `.xlog` files and import them with `use module::{func1, func2}`
- **User-Defined Functions**: Create reusable functions with `func name(args) = expression`
  - Arithmetic operations: `+`, `-`, `*`, `/`, `%`
  - Conditionals: `if condition then expr1 else expr2`
  - Recursion: Functions can call themselves and other functions
- **Reversible Symbols**: Symbol values round-trip through the string table, displaying as readable strings in query output

---

## CLI Reference

```bash
# Deterministic execution
xlog run program.xlog
xlog run program.xlog --output csv
xlog run program.xlog --output arrow --output-dir ./results

# With external data (Arrow IPC files)
xlog run program.xlog --input edge=graph.arrow

# Probabilistic execution
xlog prob program.xlog --prob-engine exact_ddnnf
xlog prob program.xlog --prob-engine mc --samples 10000 --seed 42

# Performance profiling
xlog run program.xlog --stats          # Human-readable timing
xlog run program.xlog --stats --json   # JSON format

# Options
xlog run --help
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/ARCHITECTURE.md) | System design, crate structure, algorithms |
| [Roadmap](docs/ROADMAP.md) | Feature status and development plans |
| [Benchmarks](docs/BENCHMARKS.md) | Performance methodology and baseline metrics |
| [Probabilistic Tier](docs/architecture/xlog-prob.md) | Exact and Monte Carlo inference |
| [Data Interop](docs/architecture/cudf-interop.md) | Arrow and DLPack integration |
| [Examples](examples/) | Annotated example programs |
| [CUDA Certification](docs/certification/2026-01-14-cuda-certification-results.md) | Test coverage (140/140 passing) |

---

## Project Structure

```
xlog/
├── crates/
│   ├── xlog-core/       # Foundation types and traits
│   ├── xlog-ir/         # Intermediate representations (RIR nodes)
│   ├── xlog-logic/      # Parser, compiler, optimizer
│   ├── xlog-runtime/    # Query executor
│   ├── xlog-cuda/       # GPU kernels and memory management
│   ├── xlog-stats/      # Runtime statistics and optimizer feedback
│   ├── xlog-prob/       # Probabilistic inference
│   ├── xlog-solve/      # Solver services (SAT/MaxSAT)
│   ├── xlog-gpu/        # High-level Rust API
│   ├── xlog-gpu-py/     # Python bindings
│   ├── xlog-cli/        # Command-line interface
│   └── xlog-cuda-tests/ # CUDA certification suite
├── kernels/             # CUDA kernel sources (.cu)
├── examples/            # Example .xlog programs
└── docs/                # Documentation
```

---

## Development

### Run Tests

```bash
# Full test suite (release mode recommended for GPU tests)
cargo test --workspace --all-targets --release

# CUDA certification suite only
cargo test -p xlog-cuda-tests --test certification_suite --release
```

### Run Examples

```bash
# Using the CLI
cargo run -p xlog-cli --release -- run examples/xlog/00-basics/01_tc_reachability.xlog

# Using the example runner (with more options)
cargo run -p xlog-logic --release --example xlog_run -- \
    examples/xlog/00-basics/01_tc_reachability.xlog \
    --device 0 --memory-mb 1024 --limit 100
```

---

## Contributing

Contributions are welcome! Please see:

- [Architecture Guide](docs/ARCHITECTURE.md) for system design
- [Roadmap](docs/ROADMAP.md) for planned features
- Run `cargo fmt` and `cargo clippy` before submitting

---

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

---

## Acknowledgments

XLOG builds on research in GPU-accelerated Datalog and probabilistic logic programming:

- [GPUlog](https://dl.acm.org/doi/10.1145/3183713.3183727) — HISA indexing, parallel fixpoint
- [VFLog](https://dl.acm.org/doi/10.1145/3639310) — Columnar GPU Datalog
- [ProbLog](https://dtai.cs.kuleuven.be/problog/) — Knowledge compilation for probabilistic logic
- [D4](https://github.com/crillab/d4) — Decision-DNNF compiler (vendored)
