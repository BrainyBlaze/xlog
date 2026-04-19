# XLOG

[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-v0.5.0-blue.svg)](CHANGELOG.md)

> **Release status:** `v0.5.0` — GPU-resident ILP credit/loss path (zero D2H), P2a term embeddings
> (`register_embedding` / `forward_embedding` with device-aware autograd), P2b extended training
> controls (gradient clipping, early stopping, lr management), P3 incremental verifier
> (`GpuCdclWorkspace` arena reuse). See `docs/ROADMAP.md` and `CHANGELOG.md`.

**XLOG** is a GPU-accelerated Datalog query engine with neural-symbolic integration. It compiles declarative logic programs into optimized relational plans and executes them on NVIDIA GPUs, achieving high throughput for recursive queries, graph analytics, probabilistic inference, and neural-symbolic training.

---

## Features

| Category | Capabilities |
|----------|--------------|
| **Datalog** | Rules, facts, recursion (semi-naive), stratified negation, aggregation |
| **Arithmetic** | Comparisons, `is` expressions, builtins (`abs`, `min`, `max`, `pow`, `cast`) |
| **Modules** | `use` imports, `private` predicates, nested module paths, circular import detection |
| **User-Defined Functions** | `func` syntax, `if-then-else` conditionals, recursive functions with base-case validation |
| **Reversible Symbols** | Bidirectional string-to-ID mapping, readable query output, Arrow dictionary encoding |
| **GPU Operators** | Hash joins, radix sort, filter, dedup, union, difference, groupby |
| **Float Predicates** | IEEE 754 total ordering for `f32`/`f64` (`NaN > Inf > nums > +0 > -0 > -Inf`) |
| **Probabilistic** | Exact inference (knowledge compilation), Monte Carlo sampling, negation (stratified + WFS) |
| **Neural-Symbolic** | Neural predicates (`nn/4`), PyTorch integration, differentiable training, circuit caching, term embeddings |
| **dILP Training** | Differentiable ILP: sparse GPU mask, deterministic mode, promotion gates, holdout validation, artifact save/load |
| **Bounded Exact Induction** | `xlog-induce` + `ilp_exact` CUDA kernel: score all `(L, R)` pairs across four topologies (chain / star / fanout / fanin) in one batched pass, top-K per topology, constant-size D2H budget — see [docs/architecture/bounded-exact-induction.md](docs/architecture/bounded-exact-induction.md) |
| **Interop** | Arrow IPC, DLPack (zero-copy), Python bindings, PyTorch autograd |
| **Profiling** | `--stats` flag for per-stratum/per-operation timing, memory tracking |

---

## Supported Platform Contract

Public releases of XLOG are supported on Linux x86_64 with an NVIDIA GPU and CUDA Toolkit 13.x.
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

### Planned GitHub release binary install

GitHub release archives are not published yet. When they are available, download the Linux x86_64
archive, unpack it, and run the bundled `xlog` binary from the extracted directory. Public release
archives are built with `host-io`, so `xlog prob` has host-readable output without a rebuild.

### Planned PyPI install

The `pyxlog` PyPI package is not published yet. For now, use the local development install below.
When it is published, the install flow will be:

```bash
pip install pyxlog
```

### Planned crates.io install

The CLI crate is not published on crates.io yet. When it is, the install flow will be:

```bash
cargo install xlog-cli --features host-io
```

### Local Python development install

For editable local development from source:

```bash
cd crates/pyxlog
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
./target/release/xlog run reachability.xlog

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
import pyxlog

source = """
pred edge(u32, u32).
pred reach(u32, u32).

edge(1, 2). edge(2, 3). edge(3, 4).

reach(X, Y) :- edge(X, Y).
reach(X, Z) :- reach(X, Y), edge(Y, Z).

?- reach(1, N).
"""

program = pyxlog.LogicProgram.compile(source)
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

> **CLI build note:** The `xlog prob` examples below assume `xlog-cli` was built with
> `--features host-io`. Deterministic `xlog run` works with the default build.

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
# Requires xlog-cli built with `--features host-io`
./target/release/xlog prob weather.xlog --prob-engine exact_ddnnf
# P(wet | not sprinkler) = 0.3
```

### Monte Carlo Sampling

```bash
# Requires xlog-cli built with `--features host-io`
./target/release/xlog prob weather.xlog --prob-engine mc --samples 10000
# P(wet) ≈ 0.301 ± 0.009 (95% CI)
```

### Negation in Probabilistic Programs

Exact inference supports negation with automatic stratification and Well-Founded Semantics:

```prolog
% Stratified negation (layered evaluation)
0.3::rain.
dry :- not rain.
query(dry).
% P(dry) = 0.7
```

```prolog
% Non-monotone negation (cycles through negation)
% Handled via Well-Founded Semantics
0.5::bias.
p :- bias, not q.
q :- not p.
query(p).
% WFS: atoms in cycle may be undefined (probability 0)
```

Gradients flow correctly through negated literals for neural-symbolic training.

---

## Neural-Symbolic Training (Introduced In v0.4.0-alpha, Available In v0.5.0)

XLOG supports neural-symbolic integration where neural network outputs become probabilistic facts in logic programs.
This infrastructure was introduced during the `v0.4.0-alpha` milestone and remains
available in the current `v0.5.0` release line.

Current required neural example set:
- `examples/neural/01_minimal`
- `examples/neural/02_coins`
- `examples/neural/03_mnist_multidigit`
- `examples/neural/04_hwf`
- `examples/neural/05_poker`
- `examples/neural/06_clutrr`

### Neural Predicates

Define neural networks as probabilistic fact generators:

```prolog
% Neural predicate: network outputs become probabilities for digit classification
nn(mnist_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).

% Logic rule using neural outputs
addition(X, Y, Z) :- digit(X, D1), digit(Y, D2), Z is D1 + D2.
```

### Training from Logic Supervision

Train neural networks using only logical constraints — no direct labels required:

```python
import pyxlog
import torch
import torch.nn as nn

# Define a CNN for MNIST
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

# Compile the neural-symbolic program
program = pyxlog.Program.compile("""
    nn(mnist_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).
    addition(X, Y, Z) :- digit(X, D1), digit(Y, D2), Z is D1 + D2.
""")

# Register PyTorch network with optimizer
net = MNISTNet().cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
program.register_network("mnist_net", net, optimizer)

# Load MNIST images as tensor source
program.add_tensor_source("train", train_images)  # [60000, 1, 28, 28]

# Generate training queries from addition labels
# The network learns digit classification from sum supervision only!
queries = []
for i in range(0, len(train_labels), 2):
    expected_sum = train_labels[i] + train_labels[i+1]
    queries.append(f"addition({i}, {i+1}, {expected_sum})")

# Train the model
history = pyxlog.train_model(
    program,
    queries,
    epochs=50,
    batch_size=32,
    log_iter=10
)

print(f"Initial loss: {history.epoch_losses[0]:.4f}")
print(f"Final loss: {history.epoch_losses[-1]:.4f}")
```

### How It Works

1. **Neural predicates** declare that a network provides probability distributions
2. **Forward pass**: Network outputs become weights in annotated disjunctions
3. **Knowledge compilation**: Logic program compiled to d-DNNF circuit
4. **Weighted model counting**: Circuit evaluated for query probability
5. **Backward pass**: Gradients flow from loss through circuit back to network
6. **Circuit caching**: Compiled circuits reused across training iterations (100x+ speedup)

### Term Embeddings (v0.5.0)

Register embedding modules for explicit PyTorch-side training. Embedding predicates use the
label-free `nn/3` form:

```python
program = pyxlog.Program.compile("""
    nn(entity_embed, [X], E) :: embed(X, E).
""")

# Trainable nn.Embedding — autograd intact, user-managed optimizer
embedding = torch.nn.Embedding(100, 64).cuda()
optimizer = torch.optim.Adam(embedding.parameters())
program.register_embedding("entity_embed", embedding, trainable=True)

# Batched lookup — returns [n, dim] tensor on same device as embedding
vectors = program.forward_embedding("entity_embed", [0, 5, 42])
loss = my_loss_fn(vectors, targets)
loss.backward()
optimizer.step()
```

Frozen lookup with raw tensors (no gradient flow):

```python
weights = torch.randn(100, 64).cuda()
program.register_embedding("entity_embed", weights, trainable=False)
vectors = program.forward_embedding("entity_embed", [0, 5, 42])
assert not vectors.requires_grad
```

Cross-registration validation prevents mixing embedding and classification declarations.
Compile-time rejection catches same network name used as both forms.

### Training API

```python
# Single query forward-backward (convenience; reads one scalar loss back to host)
loss = program.forward_backward("addition(0, 1, 7)")

# Strict GPU-native forward-backward (returns CUDA tensor loss; no host reads required)
loss_t = program.forward_backward_tensor("addition(0, 1, 7)")

# Batch training with optimizer step
stats = program.train_epoch(queries, batch_size=32)
avg_loss = stats.avg_loss

# Full training loop with logging
history = pyxlog.train_model(
    program,
    queries,
    epochs=50,
    batch_size=32,
    log_iter=10,  # Log every 10 batches
)
```

---

## Differentiable ILP (dILP) Training (Beta)

XLOG includes a differentiable Inductive Logic Programming trainer that learns Datalog rules from positive/negative examples using gradient descent on a GPU.

### Training a Rule

```python
from pyxlog.ilp import train_only, TrainConfig

source = """
    edge(1, 2). edge(2, 3). edge(3, 4). edge(4, 5).
    learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""
pos = [("reach", [1, 3]), ("reach", [2, 4]), ("reach", [1, 4])]
neg = [("reach", [1, 1]), ("reach", [3, 2])]

config = TrainConfig(
    step_budget_per_attempt=150,
    max_attempts=5,
    tau_start=2.0,
    tau_floor=0.05,
    seed=42,
)

result = train_only(source, "W", pos, neg, config)
if result.converged:
    print(f"Discovered: {result.discovered_rule}")
    result.artifact.save("learned.json")
```

### Promotion Pipeline

```python
from pyxlog.ilp import train_and_promote, TrainConfig

config = TrainConfig(
    check_ambiguity=True,
    holdout_threshold=0.95,
    typed_schema_required=True,
)
promotion = train_and_promote(source, "W", pos, neg, config)

print(f"Status: {promotion.status}")
for gate in promotion.gates:
    print(f"  {gate.name}: {'PASS' if gate.passed else 'FAIL'} — {gate.detail}")
```

### Key Features

- **Sparse GPU mask**: Candidate soft-probs sent via `set_rule_mask_sparse`; no Python-side N³ mask materialization
- **Deterministic mode**: Seeded training path with reproducible candidate ranking and persisted `selected_hard`
- **Multi-start optimizer**: Adaptive temperature, entropy regularization, plateau detection
- **Promotion gates**: Convergence, novel rate audit, regression check, holdout F1 threshold, ambiguity scan, typed schema gate
- **Holdout strategies**: LOO for small sets (`<=20` positives), k-fold for larger sets
- **Telemetry**: `forward_p95_us`, allocation summaries, and host-transfer accounting (`host_transfer_stats`)
- **Artifact persistence**: JSON save/load with SHA-256 hash verification
- **Recursive candidates**: Optional body-references-head rules via `allow_recursive_candidates=True`

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

### Modules (v0.3.2)

Organize large programs into reusable, encapsulated modules:

```prolog
% finance/compensation.xlog
pred base_salary(symbol, u32).
func bonus_multiplier(Tier) = if Tier = cast(1, u32) then cast(20, u32) else cast(10, u32).

% main.xlog
use finance/compensation.

pred employee_bonus(symbol, u32).
employee_bonus(EmpId, Bonus) :-
    base_salary(EmpId, Salary),
    Mult is bonus_multiplier(cast(1, u32)),
    Bonus is Salary * Mult / cast(100, u32).
```

**Module features:**
- `use module.` — import all public predicates and functions
- `use module::{pred1, func1}.` — selective imports
- `use nested/path/module.` — nested module paths
- `private pred` / `private func` — hide implementation details
- Circular import detection with clear error messages

### User-Defined Functions (v0.3.2)

Create reusable calculation functions with arithmetic and conditionals:

```prolog
% Simple arithmetic
func square(X) = X * X.
func cube(X) = X * X * X.

% Conditionals with if-then-else
func rating_tier(Score) =
    if Score >= cast(90, u32) then cast(1, u32)
    else if Score >= cast(75, u32) then cast(2, u32)
    else if Score >= cast(60, u32) then cast(3, u32)
    else cast(4, u32).

% Recursive functions (base case required)
func factorial(N) = if N <= cast(1, u32) then cast(1, u32) else N * factorial(N - cast(1, u32)).

% Usage in rules
pred result(u32, u32).
result(X, Y) :- input(X), Y is square(X).
```

**Function features:**
- Arithmetic: `+`, `-`, `*`, `/`, `%`
- Comparisons: `<`, `<=`, `>`, `>=`, `=`, `!=`
- Conditionals: `if cond then expr1 else expr2`
- Type casting: `cast(value, type)`
- Recursion with mandatory base-case validation
- Runtime depth limiting (`#pragma max_recursion_depth = 1000`)

### Reversible Symbols (v0.3.2)

Symbol values display as human-readable strings in query output:

```prolog
pred employee(symbol, symbol, symbol).
employee(e001, "Alice Chen", eng).
employee(e002, "Bob Smith", sales).

pred department(symbol, symbol).
department(eng, "Engineering").
department(sales, "Sales").

?- employee(Id, Name, Dept).
% Output:
%   Id=e001, Name=Alice Chen, Dept=eng
%   Id=e002, Name=Bob Smith, Dept=sales
```

**Symbol features:**
- Bidirectional string-to-ID mapping via global intern table
- Sequential ID allocation (0, 1, 2...) for compact storage
- Thread-safe concurrent access
- Arrow dictionary encoding for efficient serialization

---

### Showcase Examples (v0.3.2)

Four production-grade multi-module applications demonstrating all v0.3.2 features:

| Domain | Description | Features |
|--------|-------------|----------|
| [01-enterprise](examples/xlog/80-v032-showcase/01-enterprise/) | HR, finance, org hierarchy | Recursive management chains, compensation UDFs |
| [02-knowledge-graph](examples/xlog/80-v032-showcase/02-knowledge-graph/) | Movie database with semantic reasoning | Type hierarchies, ROI calculations, decade analytics |
| [03-game-analytics](examples/xlog/80-v032-showcase/03-game-analytics/) | Gaming platform with ELO rankings | Achievement prerequisites, friend-of-friend, tier calculations |
| [04-supply-chain](examples/xlog/80-v032-showcase/04-supply-chain/) | Logistics with BOM explosion | Shipping reachability, cost optimization, inventory alerts |

**Run a showcase example:**
```bash
cargo run -p xlog-cli --release --features host-io -- \
    run examples/xlog/80-v032-showcase/01-enterprise/main.xlog
```

---

## CLI Reference

```bash
# Deterministic execution
./target/release/xlog run program.xlog
./target/release/xlog run program.xlog --output csv
./target/release/xlog run program.xlog --output arrow --output-dir ./results

# With external data (Arrow IPC files)
./target/release/xlog run program.xlog --input edge=graph.arrow

# Probabilistic execution
# Requires `xlog-cli` built with `--features host-io`
./target/release/xlog prob program.xlog --prob-engine exact_ddnnf
./target/release/xlog prob program.xlog --prob-engine mc --samples 10000 --seed 42

# Performance profiling
./target/release/xlog run program.xlog --stats          # Human-readable timing
./target/release/xlog run program.xlog --stats --json   # JSON format

# Options
./target/release/xlog run --help
```

If `xlog` is installed on your `PATH` in a later packaging workflow, you can drop the
`./target/release/` prefix.

---

## Documentation

GPU-native compilation status: the GPU-native exact path is implemented end-to-end:
PIR → GPU CNF (`encode_cnf_gpu`) → GPU D4 compile → GPU CDCL equivalence verification → XGCF + GPU cache-aware eval.
The legacy CPU D4 vendor pipeline is removed.

| Document | Description |
|----------|-------------|
| [Language Reference](docs/language-reference.md) | Complete syntax guide: types, predicates, rules, modules, UDFs |
| [Architecture](docs/ARCHITECTURE.md) | System design, crate structure, algorithms |
| [Roadmap](docs/ROADMAP.md) | Feature status and development plans |
| [Benchmarks](docs/BENCHMARKS.md) | Performance methodology and baseline metrics |
| [Probabilistic Tier](docs/architecture/xlog-prob.md) | Exact and Monte Carlo inference |
| [Solver Services](docs/architecture/solver-services.md) | GPU CDCL verifier (zero host reads) + workspace arena reuse + SAT/MaxSAT services |
| [Neural-Symbolic Design](docs/plans/2026-01-20-v0.4.0-neural-symbolic-design.md) | v0.4.0 neural-symbolic integration design |
| [GPU-Native Compilation Design](docs/design/2026-01-22-gpu-native-compilation-design.md) | v0.5.0 design for GPU D4 + GPU CDCL verifier |
| [Data Interop](docs/architecture/cudf-interop.md) | Arrow and DLPack integration |
| [Examples](examples/) | Annotated example programs |
| [Neural Examples](examples/neural/) | Neural-symbolic training examples |
| [dILP Beta Design](docs/plans/2026-02-26-dilp-hardening-design.md) | dILP trainer hardening design |
| [dILP Beta Plan](docs/plans/2026-02-26-dilp-beta-impl.md) | dILP beta implementation plan (9 tasks) |
| [dILP Architecture](docs/architecture/dilp-training.md) | Runtime/trainer architecture and GPU hot-loop contract |
| [Term Embeddings Design](docs/plans/2026-03-08-p2a-term-embeddings-design.md) | P2a embedding registration, forward API, cross-registration validation |
| [Provenance Primitives Design](docs/plans/2026-03-08-provenance-primitives-design.md) | Retained provenance metadata for external Rust consumers (leaf atoms, choice sources, formula iterator) |
| [GPU Hot-loop Transfer Elimination](docs/plans/2026-03-01-gpu-hotloop-transfer-elimination.md) | Transfer-reduction design |
| [Sparse Executor Transfer Fix](docs/plans/2026-03-01-sparse-executor-transfer-fix.md) | Sparse-mask executor alignment and implementation |
| [v0.3.2 Showcase](examples/xlog/80-v032-showcase/) | Production-grade multi-module examples |
| [CUDA Certification](docs/architecture/cuda-certification.md) | Certification suite coverage (current HEAD) |

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
│   ├── xlog-prob/       # Probabilistic inference (exact + MC)
│   ├── xlog-neural/     # Neural-symbolic integration (v0.4.0)
│   ├── xlog-solve/      # Solver services (SAT/MaxSAT)
│   ├── xlog-gpu/        # High-level Rust API
│   ├── pyxlog/          # Python bindings + training API
│   ├── xlog-cli/        # Command-line interface
│   └── xlog-cuda-tests/ # CUDA certification suite
├── kernels/             # CUDA kernel sources (.cu)
├── examples/            # Example .xlog programs
│   ├── xlog/            # Deterministic Datalog examples
│   ├── prob/            # Probabilistic examples
│   ├── python/          # Python API examples
│   └── neural/          # Neural-symbolic training examples
└── docs/                # Documentation
```

---

## Development

### Run Tests

```bash
# Full test suite (release mode recommended for GPU tests)
cargo test --workspace --all-targets --exclude pyxlog --release

# CUDA certification suite only
cargo test -p xlog-cuda-tests --test certification_suite --release
```

### Run Examples

```bash
# Using the CLI
./target/release/xlog run examples/xlog/00-basics/01_tc_reachability.xlog

# Build and run the public CLI from source
cargo run -p xlog-cli --release --features host-io -- \
    run examples/xlog/00-basics/01_tc_reachability.xlog

# Full example harness (ci/dev/release modes)
python scripts/validate_examples.py --mode ci
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

XLOG builds on research in GPU-accelerated Datalog, probabilistic logic programming, and neural-symbolic AI:

- [GPUlog](https://dl.acm.org/doi/10.1145/3183713.3183727) — HISA indexing, parallel fixpoint
- [VFLog](https://dl.acm.org/doi/10.1145/3639310) — Columnar GPU Datalog
- [ProbLog](https://dtai.cs.kuleuven.be/problog/) — Knowledge compilation for probabilistic logic
- [D4](https://github.com/crillab/d4) — Decision-DNNF compilation reference
