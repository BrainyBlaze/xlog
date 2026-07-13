# XLOG

**A GPU-native logic programming language for unified symbolic reasoning.**

[![CI](https://github.com/BrainyBlaze/xlog/actions/workflows/ci.yml/badge.svg)](https://github.com/BrainyBlaze/xlog/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)
[![crates.io](https://img.shields.io/crates/v/xlog-cli.svg?label=xlog-cli&color=blue)](https://crates.io/crates/xlog-cli)
[![PyPI](https://img.shields.io/pypi/v/pyxlog.svg?label=pyxlog&color=blue)](https://pypi.org/project/pyxlog/)
[![Docs](https://img.shields.io/badge/docs-xlog.md-8A2BE2.svg)](https://xlog.md)

**Documentation: [xlog.md](https://xlog.md)** · [Whitepaper](paper/) · [Language reference](docs/language-reference.md) · [Examples](examples/)

Neural-symbolic systems today keep symbolic reasoning on the CPU while neural computation runs
on the GPU, and every training iteration pays a PCIe round-trip that dominates wall-clock time
at scale. XLOG closes that gap: one compiler and one CUDA runtime span four reasoning
paradigms — deterministic Datalog, exact and approximate probabilistic inference, SAT/MaxSAT
verification, and differentiable neural-symbolic training — with zero tracked host–device
transfers in production data planes.

Implemented in Rust with custom CUDA kernels, XLOG caches compiled circuits across training
iterations and exposes GPU-resident results through DLPack and Arrow for zero-copy interop
with PyTorch, JAX, and cuDF. On the MNIST-addition neural-symbolic benchmark this yields a
measured **2.74× end-to-end speedup** (95% CI `[2.29, 3.18]`) over a CPU-resident baseline.

---

## Why XLOG

XLOG is not a DSL bolted onto a tensor framework. It is a full typed logic programming language:

- **Typed predicates** over a closed scalar set (`u32`, `u64`, `i32`, `i64`, `f32`, `f64`, `bool`, `symbol`) with single-pass type inference that rejects ill-typed programs before any GPU kernel runs.
- **User-defined functions, modules and imports, stratified aggregation, and integrity constraints**, so programs decompose cleanly instead of collapsing into flat rule lists.
- **One syntax for four paradigms:** probabilistic facts (`p::f.`), annotated disjunctions, neural predicate declarations (`nn/k`), and SAT constraints share the same syntactic core with deterministic Datalog.
- **GPU-resident semantics:** relational operators, circuit evaluation, and verification paths run on the device instead of bouncing through the host.
- **A runtime inside your training loop, not a service:** DLPack capsules and Arrow IPC expose GPU-resident query results and gradient tensors directly.

## When to use XLOG

- **Neural-symbolic training** where the logic structure depends on the program, not on network weights: compiled circuits are cached across iterations, so only weights change, never the DAG.
- **Probabilistic reasoning** where exact inference benefits from a compile-once, evaluate-many discipline across repeated queries or training loops.
- **Graph analytics, program analysis, and recursive queries** over device-resident data, with semi-naive fixpoint evaluation and minimal host round-trips.
- **Learned-rule induction (differentiable ILP)** with GPU-resident credit assignment, sparse candidate masks, and transactional promotion gates.

---

## Quick start

Create `reachability.xlog`:

```prolog
pred edge(u32, u32).
pred reach(u32, u32).

edge(1, 2).
edge(2, 3).
edge(3, 4).

reach(X, Y) :- edge(X, Y).
reach(X, Z) :- reach(X, Y), edge(Y, Z).

?- reach(1, N).
```

Run it:

```bash
./target/release/xlog run reachability.xlog
```

Expected output:

```text
__xlog_query_0
+-------+
| col_0 |
+-------+
| 2     |
| 3     |
| 4     |
+-------+
```

The [language reference](docs/language-reference.md) covers the full surface, and the
[examples](examples/) directory contains annotated programs for lists and meta-predicates,
magic sets, probabilistic aggregates, approximate inference, epistemic reasoning
([`examples/epistemic/`](examples/epistemic/)), and Python neural-symbolic training
([`examples/python/`](examples/python/)).

---

## Features

| Category | Capabilities |
|---|---|
| **Datalog core** | Rules, facts, recursion (semi-naive fixpoint), stratified negation, aggregation, magic sets, incremental parsing |
| **Type system & modules** | Typed predicates, user-defined functions, modules with `use` imports, `private` visibility, reversible symbols |
| **Arithmetic** | `is` expressions, `+ - * / %`, builtins (`abs`, `min`, `max`, `pow`, `cast`), `if/then/else` |
| **Lists & meta-predicates** | Finite `list<T>` and `term` values, safe `findall` / `maplist` / term-inspection predicates, deterministic negation-as-failure |
| **Aggregation** | Head-positional `count`, `sum`, `min`, `max`, `logsumexp`, aggregate lifting |
| **Probabilistic inference** | Exact inference via knowledge compilation (decision-DNNF → GPU arithmetic circuits), Monte Carlo sampling, well-founded-semantics negation, approximate-inference pragmas |
| **Epistemic reasoning** | Epistemic operators with finite nested modal chains, recursive epistemic execution over the GPU-backed WFS contract, epistemic splitting, probabilistic epistemic evidence |
| **SAT / MaxSAT** | GPU CDCL verifier with on-device model and proof validation, MaxSAT, portfolio solving, continuous local search |
| **Neural-symbolic training** | Neural predicates (`nn/k`), PyTorch autograd integration, circuit caching across iterations, term embeddings, joint neural + symbolic rule-weight training |
| **Rule induction** | Differentiable ILP with sparse GPU masks, deterministic mode, promotion pipeline, holdout validation, and bounded exact induction (`xlog-induce`) with top-K CUDA scoring |
| **GPU execution** | Custom CUDA kernels for hash joins, radix sort, filter, dedup, union, difference, group-by; worst-case-optimal joins; delta coalescing; runtime CSE; adaptive re-optimization; persistent hash-index reuse |
| **Float semantics** | IEEE 754 total ordering for `f32`/`f64` (`NaN > Inf > nums > +0 > -0 > -Inf`) |
| **Diagnostics & provenance** | Rule and fact provenance, proof traces, planner telemetry, host-transfer audits, module-boundary diagnostics, `--stats` per-stratum timing and memory accounting |
| **Interop** | DLPack capsules (zero-copy), Arrow IPC / C Data Interface, PyTorch, JAX, cuDF, Python bindings (`pyxlog`) |

---

## How it works

XLOG programs are **stratified logic programs with typed predicates**. The compilation pipeline
is: parse (PEG) → stratify via SCC analysis of the predicate dependency graph → lower to a
relational IR (`RIR`) → cost-aware optimization → dispatch to a GPU backend. Four backends share
this frontend:

- deterministic evaluation via semi-naive fixpoint (`xlog-runtime`),
- probabilistic inference via knowledge compilation to device-resident arithmetic circuits (`xlog-prob`),
- SAT/MaxSAT verification (`xlog-solve`),
- differentiable training via PyTorch autograd integration (`xlog-neural` + `pyxlog`).

Language features worth naming:

- **Reversible symbols** provide bidirectional string↔ID mapping, so query output stays human-readable without giving up GPU-friendly dense integer identifiers.
- **Stratified aggregation** compiles to `GroupBy` RIR nodes executed as radix-sort-and-reduce kernels.
- **Integrity constraints** are headless rules desugared into auxiliary rules whose output must be empty at evaluation completion.
- **Pragma directives** influence compiler behavior from within a program.

For the design rationale, see the [technical whitepaper](paper/).

---

## Supported platform

Public releases of XLOG are supported on Linux `x86_64` with an NVIDIA GPU and CUDA Toolkit 13.x:

- Linux `x86_64`
- `nvidia-smi` sees the GPU
- `nvcc --version` works
- Rust `rustc` and `cargo` are available
- Python 3.8 or newer
- `xlog prob` host-readable output requires `xlog-cli` built with `host-io`

Run the doctor first after cloning:

```bash
python scripts/xlog_doctor.py
```

---

## Installation

### Source install

```bash
git clone https://github.com/BrainyBlaze/xlog.git
cd xlog
python scripts/xlog_doctor.py
cargo build --release

# If you need host-readable probabilistic CLI output (`xlog prob`),
# build the CLI with host I/O enabled.
cargo build --release -p xlog-cli --features host-io
```

The release binary is `./target/release/xlog`.

Published artifacts follow tagged releases and may lag the current `main` branch.

### GitHub release binary install

Download the Linux `x86_64` archive from the GitHub Releases page, unpack it, and run the bundled
`xlog` binary from the extracted directory. Public release archives are built with `host-io`, so
`xlog prob` has host-readable output without a rebuild.

### PyPI install

Install the latest published `pyxlog` wheel from PyPI:

```bash
pip install pyxlog
```

`pyxlog` auto-configures `XLOG_CUBIN_DIR` from its packaged `pyxlog/kernels/` directory when the
wheel includes staged CUDA artifacts. If you are running probe scripts, artifact replays, or
source-tree experiments outside that packaged layout, export `XLOG_CUBIN_DIR` yourself before
importing `pyxlog`:

```bash
export XLOG_CUBIN_DIR=/path/to/xlog/crates/pyxlog/python/pyxlog/kernels
```

For unreleased `main` branch features, use the local development install below instead of
expecting PyPI to match the current `main` branch.

### crates.io install

Install the latest published CLI crate from crates.io:

```bash
cargo install xlog-cli --features host-io
```

As with the GitHub and PyPI artifacts, published crate versions follow tagged releases and may lag
the current `main` branch. The Cargo-installed binary embeds portable PTX for all runtime
kernels, so it can run without a sidecar `kernels/` directory. If a staged `kernels/` directory or
`XLOG_CUBIN_DIR` is present, xlog still prefers those filesystem artifacts so release archives and
local builds can use architecture-specific cubins first.

### CUDA kernel artifact model

XLOG does not track generated `.ptx` or `.cubin` files in git. Kernel artifacts are produced from
`kernels/*.cu` by the Rust build and are resolved at runtime in this order:

1. `XLOG_CUBIN_DIR`
2. a package- or binary-adjacent `kernels/` directory
3. Cargo build output for source-tree builds
4. embedded portable PTX compiled into the Cargo-installed binary

This means `cargo install xlog-cli --features host-io` works without a sidecar `kernels/` directory,
while GitHub release archives and PyPI wheels still ship staged kernel artifacts for faster,
architecture-specific startup when available.

### Local Python development install

Install into the exact Python interpreter used by your downstream project. Do not rely on bare
`maturin develop` from the xlog checkout: if this repository has its own `.venv`, maturin can
install into that environment while your project imports a different Python.

```bash
python scripts/install_pyxlog_for_python.py --python /usr/local/bin/python --user
```

The helper stages CUDA kernels, builds a local wheel for the requested interpreter, installs that
wheel with the same interpreter's `pip`, and verifies that the installed `pyxlog` package contains
`pyxlog/kernels/`.

---

## CLI reference

```bash
# Deterministic execution
./target/release/xlog run program.xlog
./target/release/xlog run program.xlog --output csv
./target/release/xlog run program.xlog --output arrow --output-dir ./results

# External data (Arrow IPC)
./target/release/xlog run program.xlog --input edge=graph.arrow

# Probabilistic execution
./target/release/xlog prob program.xlog --prob-engine exact_ddnnf
./target/release/xlog prob program.xlog --prob-engine mc --samples 10000 --seed 42

# Profiling
./target/release/xlog run program.xlog --stats
./target/release/xlog run program.xlog --stats --stats-format json

# Explain diagnostics
./target/release/xlog explain program.xlog
./target/release/xlog explain --format json program.xlog

./target/release/xlog run --help
```

See the [CLI reference](docs/architecture/cli-reference.md) for the complete flag reference.

---

## Documentation

The documentation website is **[xlog.md](https://xlog.md)**. Key references in this repository:

| Document | Scope |
|---|---|
| [Whitepaper](paper/) | Primary technical reference: language, architecture, probabilistic inference, epistemic reasoning, neural-symbolic bridge, evaluation, and related work |
| [Language reference](docs/language-reference.md) | Full language surface: types, predicates, rules, modules, UDFs, aggregations, pragmas |
| [Architecture](docs/ARCHITECTURE.md) | System design, crate structure, IR layers, GPU execution model |
| [Benchmarks](docs/BENCHMARKS.md) | Performance methodology and benchmark artifacts |
| [Probabilistic tier](docs/architecture/xlog-prob.md) | Exact knowledge compilation and Monte Carlo inference |
| [Solver services](docs/architecture/solver-services.md) | GPU CDCL verifier, SAT/MaxSAT services, workspace arena reuse |
| [Differentiable ILP training](docs/architecture/dilp-training.md) | Trainer architecture and the GPU hot-loop contract |
| [ILP showcase report](docs/architecture/dilp-showcase-report.md) | End-to-end rule-induction training results |
| [WCOJ architecture guide](docs/wcoj-architecture-guide.md) | Worst-case-optimal joins: RIR, promoter, dispatch, cost model, recursive integration |
| [WCOJ user guide](docs/wcoj-user-guide.md) | Eligibility, fallback behavior, performance tuning, troubleshooting |
| [CLI reference](docs/architecture/cli-reference.md) | Full flag and subcommand reference |
| [Arrow / DLPack interop](docs/architecture/cudf-interop.md) | Zero-copy interop with cuDF, PyTorch, and JAX |
| [Python bindings](docs/architecture/python-bindings.md) | `pyxlog` API surface |
| [CUDA certification](docs/architecture/cuda-certification.md) | Kernel certification suite coverage |
| [Roadmap](ROADMAP.md) | Feature status and planned work |
| [Examples](examples/) | Annotated programs: basics, arithmetic, graphs, aggregations, probabilistic, epistemic, and neural |

---

## Project structure

```text
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
│   ├── xlog-induce/       # Native exact-induction engine and scorer
│   ├── pyxlog/            # Python bindings + training API
│   ├── xlog-cli/          # Command-line interface
│   └── xlog-cuda-tests/   # CUDA certification suite
├── kernels/               # CUDA kernel sources (.cu)
├── examples/              # Example .xlog programs + Python neural-symbolic demos
├── docs/                  # Documentation site source (published to xlog.md)
└── paper/                 # Technical whitepaper (LaTeX source)
```

---

## Development

```bash
# Full test suite (release mode recommended for GPU tests)
cargo test --workspace --all-targets --exclude pyxlog --release

# CUDA certification suite only
cargo test -p xlog-cuda-tests --test certification_suite --release

# Canonical manual GPU release validation
bash scripts/validate_release_gpu.sh --mode release

# Run an example program
cargo run -p xlog-cli --release -- run examples/xlog/00-basics/01_tc_reachability.xlog
```

---

## Contributing

Contributions are welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) and
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) first for the crate layout and layering rules,
and run `cargo fmt` plus `cargo clippy --all-targets -- -D warnings` before submitting.

---

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

---

## Acknowledgments

XLOG builds on research in logic programming languages, GPU-accelerated Datalog, probabilistic
logic programming, and neural-symbolic AI. Primary influences:

- [Prolog (SWI-Prolog)](https://www.swi-prolog.org/) and [Mercury](https://mercurylang.org/) — typed logic programming traditions
- [Souffle](https://souffle-lang.github.io/) — typed Datalog with ahead-of-time compilation
- [GPUlog](https://dl.acm.org/doi/10.1145/3183713.3183727) — HISA indexing and parallel fixpoint on GPU
- [VFLog](https://dl.acm.org/doi/10.1145/3639310) — columnar GPU Datalog
- [ProbLog](https://dtai.cs.kuleuven.be/problog/) and [DeepProbLog](https://github.com/ML-KULeuven/deepproblog) — probabilistic logic programming and neural-symbolic integration
- [d4](https://github.com/crillab/d4) — decision-DNNF compilation reference
