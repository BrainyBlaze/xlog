# XLOG

[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![CUDA Tests](https://img.shields.io/badge/CUDA%20tests-206%2F206-brightgreen.svg)](docs/architecture/cuda-certification.md)
[![Version](https://img.shields.io/badge/version-v0.5.0-blue.svg)](CHANGELOG.md)

> **Release status:** `v0.5.0` — GPU-resident ILP credit/loss path (zero D2H), P2a term embeddings
> (`register_embedding` / `forward_embedding` with device-aware autograd), P2b extended training
> controls (gradient clipping, early stopping, lr management), P3 incremental verifier
> (`GpuCdclWorkspace` arena reuse). See `CHANGELOG.md`.

**XLOG is a GPU-native logic programming language for unified symbolic reasoning.** Neural-symbolic systems today keep symbolic reasoning on the CPU while neural computation runs on the GPU; every training iteration pays a PCIe round-trip that dominates wall-clock time at scale. XLOG closes that gap: its compiler and runtime span four reasoning paradigms — deterministic Datalog evaluation, probabilistic inference via knowledge compilation (PIR → CNF → D4 → XGCF), SAT/MaxSAT verification, and differentiable neural-symbolic training — on a single CUDA runtime with zero host–device transfers in production paths. Implemented in Rust with 21 custom CUDA kernel files (14.2K lines of device code), XLOG caches compiled circuits across training iterations, yielding a measured **2.74× end-to-end speedup** (95% CI `[2.29, 3.18]`) on the MNIST addition benchmark, and exposes GPU-resident results via DLPack and Arrow for zero-copy interop with PyTorch, JAX, and cuDF.

See [`docs/whitepaper/main.pdf`](docs/whitepaper/main.pdf) for the full v0.5.0 technical whitepaper.

---

## Why XLOG

XLOG is not a DSL bolted onto a tensor framework. It is a full typed logic programming language:

- **Typed predicates** over a closed scalar set (`u32`, `u64`, `i32`, `i64`, `f32`, `f64`, `bool`, `symbol`) with single-pass type inference that rejects programs before any GPU kernel runs.
- **User-defined functions, modules and imports, stratified aggregation, integrity constraints** — the language surface supports real program decomposition, not just rule lists.
- **One syntax for four paradigms:** probabilistic facts (`p::f.`), annotated disjunctions, neural predicate declarations (`nn/k`), and SAT constraints share the same syntactic core with deterministic Datalog. A single parser, type checker, stratifier, and lowerer produces every downstream IR.
- **GPU-resident semantics.** Relational operators (hash join, radix sort, filter, dedup, set difference, grouped aggregation) run as custom CUDA kernels. Knowledge-compilation circuits execute level-parallel on the device. SAT/MaxSAT CDCL uses on-GPU model and proof validation.
- **Runtime inside your training loop, not a service.** DLPack capsules and Arrow IPC expose GPU-resident query results and gradient tensors directly — no copies, no synchronization barriers.

---

## When to use XLOG

- **Neural-symbolic training** where logic structure depends on the program, not on network weights. Compiled circuits are cached across iterations; only weights change, not the DAG. This is where the 2.74× speedup comes from.
- **Probabilistic reasoning at scale** where exact inference benefits from a compile-once, evaluate-many discipline — training loops, sensitivity analyses, batched queries over weighted model counting.
- **Graph analytics, program analysis, recursive queries** over device-resident data, with semi-naive fixpoint evaluation and zero host round-trips.
- **Learned-rule induction (dILP)** with GPU-resident credit assignment, sparse candidate masks, and transactional promotion gates.

---

## Features

| Category | Capabilities |
|---|---|
| **Datalog core** | Rules, facts, recursion (semi-naive), stratified negation and aggregation |
| **Language** | Typed predicates, UDFs, modules (`use` imports), `private` visibility, reversible symbols |
| **Arithmetic** | `is` expressions, `+ - * / %`, built-ins (`abs`, `min`, `max`, `pow`, `cast`), `if/then/else` |
| **Aggregation** | Head-positional `count`, `sum`, `min`, `max`, `logsumexp` |
| **GPU operators** | Hash joins, radix sort, filter, dedup, union, difference, group-by — all custom CUDA |
| **Float semantics** | IEEE 754 total ordering for `f32`/`f64` (`NaN > Inf > nums > +0 > -0 > -Inf`) |
| **Probabilistic** | Exact inference via knowledge compilation (D4 → XGCF), Monte Carlo sampling, WFS negation |
| **Neural-symbolic** | Neural predicates (`nn/k`), PyTorch autograd integration, circuit caching, term embeddings |
| **dILP training** | Sparse GPU mask, deterministic mode, six-gate promotion pipeline, holdout validation |
| **SAT/MaxSAT** | GPU CDCL verifier with on-device model/proof validation, continuous local search |
| **Interop** | DLPack capsules (zero-copy), Arrow IPC/C Data, PyTorch/JAX/cuDF, Python bindings |
| **Profiling** | `--stats` flag: per-stratum timing, memory accounting, host-transfer counters |

---

## Core concepts

XLOG programs are **stratified logic programs with typed predicates**. The compilation pipeline is: parse (PEG) → stratify via SCC analysis of the predicate dependency graph → lower to a relational IR (`RIR`) → cost-aware optimize → dispatch to a GPU backend. Four backends share this frontend: deterministic evaluation via semi-naive fixpoint (`xlog-runtime`), probabilistic inference via knowledge compilation to device-resident arithmetic circuits (`xlog-prob`), SAT/MaxSAT verification (`xlog-solve`), and differentiable training via PyTorch autograd integration (`xlog-neural` + `pyxlog`).

Key language features worth naming: **reversible symbols** (bidirectional string–ID mapping, so query output stays human-readable without losing GPU-friendly dense integer identifiers); **stratified aggregation** (head-positional, compiled to `GroupBy` RIR nodes executed as radix-sort-and-reduce kernels); **integrity constraints** (headless rules desugared into auxiliary rules whose output must be empty at evaluation completion); **pragma directives** that influence compiler behavior from within a program.

For the full language surface see [`docs/language-reference.md`](docs/language-reference.md). For the design philosophy and the case for each design decision see **Section 3 of the whitepaper** ([`docs/whitepaper/main.pdf`](docs/whitepaper/main.pdf)).

---

## Installation

### Requirements

- Linux (x86_64)
- NVIDIA GPU with compute capability **sm_70+** (Volta or newer)
- CUDA Toolkit 12.x
- Rust 1.75+ (for building from source)

### Build from source

```bash
git clone https://github.com/BrainyBlaze/xlog.git
cd xlog
cargo build --release
```

The `xlog` CLI binary will be at `target/release/xlog`.

### Python package

```bash
cd crates/pyxlog
pip install maturin
maturin develop --release
```

---

## Quick start

Create `reachability.xlog`:

```xlog
pred edge(u32, u32).
pred reach(u32, u32).

edge(1, 2). edge(2, 3). edge(3, 4).

reach(X, Y) :- edge(X, Y).
reach(X, Z) :- reach(X, Y), edge(Y, Z).

?- reach(1, N).
```

Run it:

```bash
xlog run reachability.xlog
```

Expected output: `N = 2, 3, 4`.

For the full language reference and worked examples, see [`docs/language-reference.md`](docs/language-reference.md). For runnable programs covering arithmetic, aggregations, probabilistic inference, and neural-symbolic training, see the [`examples/`](examples/) tree. For Rust and Python API usage, see [`examples/python/`](examples/python/) and [`docs/architecture/python-bindings.md`](docs/architecture/python-bindings.md).

---

## CLI reference

```bash
# Deterministic execution
xlog run program.xlog
xlog run program.xlog --output csv
xlog run program.xlog --output arrow --output-dir ./results

# External data (Arrow IPC)
xlog run program.xlog --input edge=graph.arrow

# Probabilistic execution
xlog prob program.xlog --prob-engine exact_ddnnf
xlog prob program.xlog --prob-engine mc --samples 10000 --seed 42

# Profiling
xlog run program.xlog --stats
xlog run program.xlog --stats --stats-format json

xlog run --help
```

See [`docs/architecture/cli-reference.md`](docs/architecture/cli-reference.md) for the complete flag reference.

---

## Documentation

| Document | Scope |
|---|---|
| [Whitepaper (PDF)](docs/whitepaper/main.pdf) | **Primary reference.** v0.5.0 technical whitepaper covering language, architecture, probabilistic inference, neural-symbolic bridge, evaluation, and related work. |
| [Language reference](docs/language-reference.md) | Full language surface: types, predicates, rules, modules, UDFs, aggregations, pragmas |
| [Architecture](docs/ARCHITECTURE.md) | System design, crate structure, IR layers, GPU execution model |
| [Probabilistic tier](docs/architecture/xlog-prob.md) | Exact knowledge compilation and Monte Carlo inference |
| [Solver services](docs/architecture/solver-services.md) | GPU CDCL verifier, SAT/MaxSAT services, workspace arena reuse |
| [dILP training](docs/architecture/dilp-training.md) | Differentiable ILP trainer architecture and GPU hot-loop contract |
| [dILP showcase report](docs/architecture/dilp-showcase-report.md) | End-to-end dILP training results |
| [CLI reference](docs/architecture/cli-reference.md) | Full flag and subcommand reference |
| [Arrow / DLPack interop](docs/architecture/cudf-interop.md) | Zero-copy interop with cuDF, PyTorch, JAX |
| [Python bindings](docs/architecture/python-bindings.md) | `pyxlog` API surface |
| [GPU execution](docs/architecture/gpu-execution.md) | Semi-naive fixpoint on GPU, kernel dispatch |
| [Query optimizer](docs/architecture/query-optimizer.md) | Cost-aware join ordering, predicate pushdown |
| [CUDA certification](docs/architecture/cuda-certification.md) | Kernel certification suite coverage |
| [Examples](examples/) | Annotated programs: basics, arithmetic, graphs, aggregations, probabilistic, neural |
| [v0.3.2 showcase](examples/xlog/80-v032-showcase/) | Multi-module production-grade examples |

---

## Project structure

```
xlog/
├── crates/
│   ├── xlog-core/         # Foundation types and traits
│   ├── xlog-ir/           # Intermediate representations (RIR nodes)
│   ├── xlog-logic/        # Language frontend: parser, stratifier, lowerer, optimizer
│   ├── xlog-runtime/      # Deterministic query executor
│   ├── xlog-cuda/         # GPU kernels and memory management
│   ├── xlog-stats/        # Runtime statistics and optimizer feedback
│   ├── xlog-prob/         # Probabilistic inference (exact + MC)
│   ├── xlog-neural/       # Neural-symbolic integration
│   ├── xlog-solve/        # SAT/MaxSAT solver services
│   ├── xlog-gpu/          # High-level Rust API
│   ├── pyxlog/            # Python bindings + training API
│   ├── xlog-cli/          # Command-line interface
│   └── xlog-cuda-tests/   # CUDA certification suite
├── kernels/               # CUDA kernel sources (.cu)
├── examples/              # Example .xlog programs + Python neural-symbolic demos
└── docs/                  # Documentation + whitepaper
```

---

## Development

```bash
# Full test suite (release mode recommended for GPU tests)
cargo test --workspace --all-targets --exclude pyxlog --release

# CUDA certification suite only
cargo test -p xlog-cuda-tests --test certification_suite --release

# Run an example program
cargo run -p xlog-cli --release -- run examples/xlog/00-basics/01_tc_reachability.xlog
```

---

## Contributing

Contributions are welcome. Please read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) first for the crate layout and layering rules. Run `cargo fmt` and `cargo clippy --all-targets -- -D warnings` before submitting.

---

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

---

## Acknowledgments

XLOG builds on research in logic programming languages, GPU-accelerated Datalog, probabilistic logic programming, and neural-symbolic AI. Primary influences:

- [Prolog (SWI-Prolog)](https://www.swi-prolog.org/) and [Mercury](https://mercurylang.org/) — typed logic programming traditions
- [Soufflé](https://souffle-lang.github.io/) — typed Datalog with ahead-of-time compilation
- [GPUlog](https://dl.acm.org/doi/10.1145/3183713.3183727) — HISA indexing, parallel fixpoint on GPU
- [VFLog](https://dl.acm.org/doi/10.1145/3639310) — columnar GPU Datalog
- [ProbLog](https://dtai.cs.kuleuven.be/problog/) and [DeepProbLog](https://github.com/ML-KULeuven/deepproblog) — probabilistic logic programming and neural-symbolic integration
- [D4](https://github.com/crillab/d4) — decision-DNNF compilation reference
