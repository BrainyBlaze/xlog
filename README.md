# XLOG

[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)
[![CUDA Tests](https://img.shields.io/badge/CUDA%20tests-206%2F206-brightgreen.svg)](docs/architecture/cuda-certification.md)
[![Version](https://img.shields.io/badge/version-v0.7.0-blue.svg)](CHANGELOG.md)

> **Release status:** `v0.7.0` - General WCOJ Architecture and Runtime Expansion.
> The release expands the v0.6.2 triangle accelerator into a production WCOJ
> subsystem: first-class multiway RIR, variable-order and cardinality cost models,
> recursive/SCC integration, K5-K8 clique coverage, runtime histogram block
> slicing, helper splitting for buried skew, bounded CUDA Graph replay for the
> DTS-DLM hot loop, and end-to-end DTS-DLM validation. See `ROADMAP.md`,
> `CHANGELOG.md`, `docs/wcoj-architecture-guide.md`, and
> `docs/wcoj-user-guide.md`.

**XLOG is a GPU-native logic programming language for unified symbolic reasoning.**
Neural-symbolic systems today keep symbolic reasoning on the CPU while neural computation runs on
the GPU; every training iteration pays a PCIe round-trip that dominates wall-clock time at scale.
XLOG closes that gap: its compiler and runtime span four reasoning paradigms - deterministic
Datalog evaluation, probabilistic inference via knowledge compilation (PIR -> CNF -> D4 -> XGCF),
SAT/MaxSAT verification, and differentiable neural-symbolic training - on a single CUDA runtime
with zero host-device transfers in production paths. Implemented in Rust with custom CUDA kernels,
XLOG caches compiled circuits across training iterations, yielding a measured **2.74x end-to-end
speedup** (95% CI `[2.29, 3.18]`) on the MNIST addition benchmark, and exposes GPU-resident
results via DLPack and Arrow for zero-copy interop with PyTorch, JAX, and cuDF.

See [`docs/whitepaper/main.pdf`](docs/whitepaper/main.pdf) for the v0.5.0 technical whitepaper.
The whitepaper is a stable research reference; the installation, packaging, and release contract
below tracks the current public release process on `main`.

---

## Why XLOG

XLOG is not a DSL bolted onto a tensor framework. It is a full typed logic programming language:

- **Typed predicates** over a closed scalar set (`u32`, `u64`, `i32`, `i64`, `f32`, `f64`, `bool`, `symbol`) with single-pass type inference that rejects programs before any GPU kernel runs.
- **User-defined functions, modules and imports, stratified aggregation, integrity constraints** let programs decompose cleanly instead of collapsing into flat rule lists.
- **One syntax for four paradigms:** probabilistic facts (`p::f.`), annotated disjunctions, neural predicate declarations (`nn/k`), and SAT constraints share the same syntactic core with deterministic Datalog.
- **GPU-resident semantics:** relational operators, circuit evaluation, and verification paths run on the device instead of bouncing through the host.
- **Runtime inside your training loop, not a service:** DLPack capsules and Arrow IPC expose GPU-resident query results and gradient tensors directly.

---

## When to use XLOG

- **Neural-symbolic training** where logic structure depends on the program, not on network weights. Compiled circuits are cached across iterations; only weights change, not the DAG.
- **Probabilistic reasoning at scale** where exact inference benefits from a compile-once, evaluate-many discipline across repeated queries or training loops.
- **Graph analytics, program analysis, and recursive queries** over device-resident data, with semi-naive fixpoint evaluation and minimal host round-trips.
- **Learned-rule induction (dILP)** with GPU-resident credit assignment, sparse candidate masks, and transactional promotion gates.

---

## Features

| Category | Capabilities |
|---|---|
| **Datalog core** | Rules, facts, recursion (semi-naive), stratified negation, aggregation |
| **Language** | Typed predicates, UDFs, modules (`use` imports), `private` visibility, reversible symbols |
| **Arithmetic** | `is` expressions, `+ - * / %`, builtins (`abs`, `min`, `max`, `pow`, `cast`), `if/then/else` |
| **Aggregation** | Head-positional `count`, `sum`, `min`, `max`, `logsumexp` |
| **GPU operators** | Hash joins, radix sort, filter, dedup, union, difference, group-by - all custom CUDA |
| **Float semantics** | IEEE 754 total ordering for `f32`/`f64` (`NaN > Inf > nums > +0 > -0 > -Inf`) |
| **Probabilistic** | Exact inference via knowledge compilation (D4 -> XGCF), Monte Carlo sampling, WFS negation |
| **Neural-symbolic** | Neural predicates (`nn/k`), PyTorch autograd integration, circuit caching, term embeddings |
| **dILP training** | Sparse GPU mask, deterministic mode, promotion pipeline, holdout validation, artifact save/load |
| **Bounded exact induction** | `xlog-induce` plus `ilp_exact` CUDA scoring with top-K per topology and fixed-size D2H summaries |
| **SAT/MaxSAT** | GPU CDCL verifier with on-device model/proof validation, continuous local search |
| **Interop** | DLPack capsules (zero-copy), Arrow IPC/C Data, PyTorch/JAX/cuDF, Python bindings |
| **Profiling** | `--stats` flag for per-stratum timing, memory accounting, and host-transfer counters |

---

## Core concepts

XLOG programs are **stratified logic programs with typed predicates**. The compilation pipeline is:
parse (PEG) -> stratify via SCC analysis of the predicate dependency graph -> lower to a
relational IR (`RIR`) -> cost-aware optimize -> dispatch to a GPU backend. Four backends share
this frontend: deterministic evaluation via semi-naive fixpoint (`xlog-runtime`), probabilistic
inference via knowledge compilation to device-resident arithmetic circuits (`xlog-prob`),
SAT/MaxSAT verification (`xlog-solve`), and differentiable training via PyTorch autograd
integration (`xlog-neural` + `pyxlog`).

Key language features worth naming:

- **Reversible symbols** provide bidirectional string-ID mapping, so query output stays human-readable without giving up GPU-friendly dense integer identifiers.
- **Stratified aggregation** is compiled to `GroupBy` RIR nodes executed as radix-sort-and-reduce kernels.
- **Integrity constraints** are headless rules desugared into auxiliary rules whose output must be empty at evaluation completion.
- **Pragma directives** influence compiler behavior from within a program.

For the full language surface, see [`docs/language-reference.md`](docs/language-reference.md).
For the design rationale behind the language framing, see the whitepaper in
[`docs/whitepaper/main.pdf`](docs/whitepaper/main.pdf).

---

## Supported platform contract

Public releases of XLOG are supported on Linux `x86_64` with an NVIDIA GPU and CUDA Toolkit 13.x.
The public setup assumes:

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

Published artifacts follow tagged releases and may lag the current `main` branch workspace version
shown at the top of this README.

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
wheel includes staged CUDA artifacts. If you are running probe scripts, v3 artifact replays, or
source-tree experiments outside that packaged layout, export `XLOG_CUBIN_DIR` yourself before
importing `pyxlog`:

```bash
export XLOG_CUBIN_DIR=/home/dev/projects/xlog/crates/pyxlog/python/pyxlog/kernels
```

For unreleased `main` branch features, use the local development install below instead of
expecting PyPI to match the current workspace version.

### crates.io install

Install the latest published CLI crate from crates.io:

```bash
cargo install xlog-cli --features host-io
```

As with the GitHub and PyPI artifacts, published crate versions follow tagged releases and may lag
the current `main` branch workspace version. The Cargo-installed binary embeds portable PTX for all
runtime kernels, so it can run without a sidecar `kernels/` directory. If a staged `kernels/`
directory or `XLOG_CUBIN_DIR` is present, xlog still prefers those filesystem artifacts so release
archives and local builds can use architecture-specific cubins first.

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

For the full language reference and worked examples, see
[`docs/language-reference.md`](docs/language-reference.md). For runnable programs covering
arithmetic, aggregation, probabilistic inference, and neural-symbolic training, see the
[`examples/`](examples/) tree. For Rust and Python API usage, see [`examples/python/`](examples/python/)
and [`docs/architecture/python-bindings.md`](docs/architecture/python-bindings.md).

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

./target/release/xlog run --help
```

See [`docs/architecture/cli-reference.md`](docs/architecture/cli-reference.md) for the complete
flag reference.

---

## Documentation

| Document | Scope |
|---|---|
| [Whitepaper (PDF)](docs/whitepaper/main.pdf) | Primary reference. v0.5.0 technical whitepaper covering language, architecture, probabilistic inference, neural-symbolic bridge, evaluation, and related work |
| [Language reference](docs/language-reference.md) | Full language surface: types, predicates, rules, modules, UDFs, aggregations, pragmas |
| [Architecture](docs/ARCHITECTURE.md) | System design, crate structure, IR layers, GPU execution model |
| [Roadmap](ROADMAP.md) | Feature status, shipped milestones, and planned work |
| [Benchmarks](docs/BENCHMARKS.md) | Performance methodology and benchmark artifacts |
| [WCOJ architecture guide](docs/wcoj-architecture-guide.md) | RIR, promoter, dispatch, cost model, recursive integration, and Phase-2 WCOJ mechanisms |
| [WCOJ user guide](docs/wcoj-user-guide.md) | Eligibility, fallback behavior, performance tuning, env vars, troubleshooting, and DTS-DLM guidance |
| [Probabilistic tier](docs/architecture/xlog-prob.md) | Exact knowledge compilation and Monte Carlo inference |
| [Solver services](docs/architecture/solver-services.md) | GPU CDCL verifier, SAT/MaxSAT services, workspace arena reuse |
| [dILP training](docs/architecture/dilp-training.md) | Differentiable ILP trainer architecture and GPU hot-loop contract |
| [dILP showcase report](docs/architecture/dilp-showcase-report.md) | End-to-end dILP training results |
| [CLI reference](docs/architecture/cli-reference.md) | Full flag and subcommand reference |
| [Arrow / DLPack interop](docs/architecture/cudf-interop.md) | Zero-copy interop with cuDF, PyTorch, and JAX |
| [Python bindings](docs/architecture/python-bindings.md) | `pyxlog` API surface |
| [CUDA certification](docs/architecture/cuda-certification.md) | Kernel certification suite coverage |
| [Examples](examples/) | Annotated programs: basics, arithmetic, graphs, aggregations, probabilistic, and neural |
| [v0.3.2 showcase](examples/xlog/80-v032-showcase/) | Multi-module production-grade examples |

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
└── docs/                  # Documentation + whitepaper
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

Contributions are welcome. Please read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) first for
the crate layout and layering rules. See [`ROADMAP.md`](ROADMAP.md) for planned work, and run
`cargo fmt` plus `cargo clippy --all-targets -- -D warnings` before submitting.

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

- [Prolog (SWI-Prolog)](https://www.swi-prolog.org/) and [Mercury](https://mercurylang.org/) - typed logic programming traditions
- [Souffle](https://souffle-lang.github.io/) - typed Datalog with ahead-of-time compilation
- [GPUlog](https://dl.acm.org/doi/10.1145/3183713.3183727) - HISA indexing and parallel fixpoint on GPU
- [VFLog](https://dl.acm.org/doi/10.1145/3639310) - columnar GPU Datalog
- [ProbLog](https://dtai.cs.kuleuven.be/problog/) and [DeepProbLog](https://github.com/ML-KULeuven/deepproblog) - probabilistic logic programming and neural-symbolic integration
- [D4](https://github.com/crillab/d4) - decision-DNNF compilation reference
