# `xlog-prob` Architecture (Phase 4)

`xlog-prob` is XLOG’s probabilistic reasoning tier. It consumes a probabilistic `.xlog` program (probabilistic facts, annotated disjunctions, evidence, and probabilistic queries) and evaluates query probabilities either:

- **Exactly** via knowledge compilation (`prob_engine=exact_ddnnf`): PIR → CNF → D4 → Decision-DNNF → GPU weighted model counting + gradients.
- **Approximately** via Monte Carlo sampling (`prob_engine=mc`): GPU sampling of probabilistic leaves + deterministic evaluation per sampled world, with uncertainty reporting.

This document explains the implementation as it exists on `main` and points to concrete entry points in the codebase.

---

## Key Entry Points

### Core crate
- `crates/xlog-prob/src/exact.rs`: exact inference API (`ExactDdnnfProgram`)
- `crates/xlog-prob/src/mc.rs`: Monte Carlo engine (`McProgram`) + non-monotone SCC semantics (`NONMONOTONE_SEMANTICS`)
- `crates/xlog-prob/src/provenance.rs`: provenance extraction + grounding + probabilistic lowering
- `crates/xlog-prob/src/pir.rs`: PIR graph data model
- `crates/xlog-prob/src/cnf.rs`: Tseitin encoding + CNF var mapping
- `crates/xlog-prob/src/kc/d4.rs`: D4 compiler wrapper (runs vendored `d4`)
- `crates/xlog-prob/src/kc/ddnnf.rs`: Decision-DNNF parser
- `crates/xlog-prob/src/xgcf.rs`: XGCF (GPU circuit format) construction
- `crates/xlog-prob/src/gpu.rs`: GPU upload + evaluation glue (`GpuXgcf`)

### CUDA kernels
- `kernels/circuit.cu` / `kernels/circuit.ptx`: forward + backward kernels for XGCF circuits
- `kernels/mc_sample.cu` / `kernels/mc_sample.ptx`: Bernoulli sampling kernel used by `mc`

### Python bindings (DLPack-first)
- `crates/xlog-gpu-py/src/lib.rs`: `xlog_gpu` module (PyO3)
  - Probabilistic API: `Program.compile(..., prob_engine="exact_ddnnf"|"mc")`
  - Deterministic API: `LogicProgram.compile(...)`

---

## Language Surface (Probabilistic Profile)

The probabilistic surface is parsed by `xlog-logic` (see `crates/xlog-logic/src/grammar.pest` and `crates/xlog-logic/src/parser.rs`) and represented in the AST (`xlog_logic::ast::Program`).

Supported constructs:

- **Probabilistic facts**: `p::atom(...).` (Bernoulli)
- **Annotated disjunctions (AD)**: `p1::a1(...); p2::a2(...).` (categorical; optional “none” outcome if probabilities sum < 1)
- **Evidence**: `evidence(atom(...), true|false).`
- **Queries**: `query(atom(...)).`
- **Engine selection**: `#pragma prob_engine = exact_ddnnf|mc` (API overrides take precedence in Python)

---

## Engine Selection and Contracts

### `prob_engine=exact_ddnnf` (exact)

**Primary goal:** compute exact conditional probabilities `P(Q|E)` and (optionally) gradients w.r.t. probabilistic leaf log-weights on GPU.

**Current semantic constraints (enforced by implementation):**
- **Negation is fully supported** via NNF transformation and Well-Founded Semantics:
  - Stratified negation: automatic layer detection and two-valued evaluation
  - Non-monotone (cyclic) negation: WFS three-valued semantics (True/False/Undefined)
  - Gradients flow through negated literals with correct sign flip
- Recursive programs are supported; provenance is constructed as an acyclic PIR by semi-naive, iteration-indexed unrolling.
- **Aggregation is not supported** in exact inference rule bodies.

If a program uses aggregation, use `prob_engine=mc`.

### `prob_engine=mc` (approximate / P3)

**Primary goal:** provide a robust, explicit escape hatch for:
- non-monotone recursion (cycles through `not` and/or aggregates), and
- probabilistic programs that are not supported by the exact provenance compiler.

**Outputs are marked approximate** and include uncertainty metadata (standard error + confidence interval, sample counts, and seed).

**Non-monotone SCC semantics:** `xlog_prob::mc::NONMONOTONE_SEMANTICS` (also surfaced to Python results).

---

## Exact Path (`ExactDdnnfProgram`)

### Pipeline Overview

1. **Parse + stratify** (`xlog_logic::parse_program`, then `crates/xlog-prob/src/provenance.rs`).
2. **Provenance extraction** → PIR graph:
   - probabilistic facts become PIR leaf literals with probabilities (`leaf_probs`)
   - annotated disjunctions become a chain of Bernoulli decision variables (`choice_probs`)
   - derived tuples map to PIR formulas (`tuple_formulas`)
3. **Tseitin encoding** (PIR → CNF) with a stable var map (`crates/xlog-prob/src/cnf.rs`).
4. **Knowledge compilation (D4)**: CNF → Decision-DNNF (`crates/xlog-prob/src/kc/d4.rs`).
5. **Lower to XGCF**: Decision-DNNF → GPU circuit format (`crates/xlog-prob/src/xgcf.rs`).
6. **GPU evaluation** (`crates/xlog-prob/src/gpu.rs` + `kernels/circuit.ptx`):
   - forward pass computes `log WMC(...)` in log-space
   - backward pass computes gradients w.r.t. leaf log-weights

### Conditional probability

For each query variable `Q` and evidence `E`:

- `log Z_E  = log WMC(E)`
- `log Z_EQ = log WMC(E ∧ Q)`
- `log P(Q|E) = log Z_EQ − log Z_E`

This is implemented in `crates/xlog-prob/src/exact.rs` (`ExactDdnnfProgram::evaluate` and `ExactDdnnfProgram::evaluate_gpu_with_grads`).

### GPU state and caching

`ExactDdnnfProgram` stores the compiled `Xgcf` in memory and lazily initializes GPU state in a `OnceLock` (`GpuExactState`). The first evaluation uploads the circuit to the configured CUDA device (`GpuConfig { device_ordinal, memory_bytes }`), and subsequent evaluations reuse it.

### Vendored D4 build and invocation

D4 is vendored under `vendor/d4` and built automatically by `crates/xlog-prob/build.rs` during `cargo build` / `cargo test`:

- Output binary is staged into Cargo `OUT_DIR` and exported as `XLOG_PROB_D4_PATH`.
- `D4Compiler::detect()` reads `XLOG_PROB_D4_PATH` and runs that binary for compilation.

`ExactDdnnfProgram` writes CNF and reads the Decision-DNNF output via a temporary working directory.

---

## Monte Carlo Path (`McProgram`)

### Sampling plan

`McProgram` compiles probabilistic leaves into a flat Bernoulli vector (`bernoulli_probs: Vec<f32>`):

- each probabilistic fact is a direct Bernoulli variable
- each annotated disjunction is encoded as a **chain of conditional Bernoulli decisions** (matching the exact/provenance lowering):
  - for categorical probabilities `(p0, p1, …, pk)` the chain samples `k-1` Bernoullis with conditional probabilities `p_i / remaining`
  - if the probabilities sum to `< 1`, an explicit “none” outcome is represented by the remaining mass

This compilation is implemented in `crates/xlog-prob/src/mc.rs` (`compile_sampling_plan`).

### GPU sampling

Sampling uses `CudaKernelProvider::sample_bernoulli_matrix(...)`, which calls `kernels/mc_sample.ptx` to generate a row-major `[samples][num_vars]` matrix of 0/1 bytes on the GPU and copies it back to host memory for evaluation.

Sampling is deterministic given `seed`.

### Deterministic evaluation per world

For each sample:

1. Clone the base EDB store (deterministic facts), apply sampled probabilistic facts and AD outcomes.
2. Evaluate the program SCC-by-SCC using a CPU relation store (`HashMap<String, Relation>` where `Relation` is a set of tuples).

SCC evaluation strategy:

- **Monotone non-recursive**: single forward pass over rules.
- **Monotone recursive**: semi-naive fixpoint with per-rule delta selection.
- **Non-monotone SCCs** (cycles through `not` and/or aggregates): synchronous iteration with explicit cycle detection:
  - if a fixpoint is reached, use it
  - if a cycle is detected, use the intersection of all states in the cycle (skeptical tuples only)
  - if the iteration budget is exceeded, use the intersection across all visited states (conservative)

This is implemented in `crates/xlog-prob/src/mc.rs` (`evaluate_program_inplace`, `eval_monotone_recursive_scc`, `eval_nonmonotone_scc`).

### Evidence conditioning and uncertainty reporting

`mc` uses rejection sampling for evidence:

- Only samples satisfying `evidence(...)` are counted (`evidence_samples`).
- If evidence is present and never satisfied, evaluation fails with a deterministic error.

For each query, `mc` estimates `P(Q|E)` as a binomial proportion and reports:

- `prob`, `log_prob`
- standard error (`stderr`)
- two-sided confidence interval (`ci_low`, `ci_high`) for the configured `confidence`
- `samples`, `evidence_samples`, `seed`
- non-monotone SCC diagnostics (`nonmonotone_sccs`, `nonmonotone_cycles`, `nonmonotone_iteration_limit_hits`)

---

## Python API (`xlog_gpu`)

The PyO3 extension `crates/xlog-gpu-py` exposes two entry points:

- `Program.compile(source, device=0, memory_mb=1024, prob_engine=None) -> CompiledProgram`
  - `CompiledProgram.evaluate(return_grads=False, ...) -> EvalResult`
  - outputs probabilities as DLPack tensors (`prob`, `log_prob`)
  - exact engine optionally returns per-query gradients (`grad_true`, `grad_false`)
  - MC engine returns uncertainty metadata and sets `approx=True`
- `LogicProgram.compile(source, device=0, memory_mb=1024) -> CompiledLogicProgram`
  - `CompiledLogicProgram.evaluate(dlpack_inputs={...}) -> LogicEvalResult`

All GPU table interchange is via DLPack capsules (framework-agnostic). See `examples/python/` for end-to-end scripts.

---

## Reproducibility

### Rust (workspace)

```bash
cargo test --workspace --all-targets --release
```

### Rust (`xlog-prob` focused)

```bash
cargo test -p xlog-prob --all-targets --release
```

### CUDA certification suite (release)

```bash
cargo test -p xlog-cuda-tests --test certification_suite --release -- --nocapture
```

### Python examples (local wheel)

```bash
cd crates/xlog-gpu-py
python -m pip install --upgrade pip maturin
maturin develop --release
python ../../examples/python/03_prob_mc_nonmonotone_torch.py
```

