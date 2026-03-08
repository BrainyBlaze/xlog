# `xlog-prob` Architecture (Phase 4)

`xlog-prob` is XLOG’s probabilistic reasoning tier. It consumes a probabilistic `.xlog` program (probabilistic facts, annotated disjunctions, evidence, and probabilistic queries) and evaluates query probabilities either:

- **Exactly** via GPU-native knowledge compilation (`prob_engine=exact_ddnnf`): PIR → GPU CNF → GPU D4 compiler → XGCF circuit → GPU weighted model counting + gradients.
- **Approximately** via Monte Carlo sampling (`prob_engine=mc`): GPU sampling of probabilistic leaves + deterministic evaluation per sampled world, with uncertainty reporting.

This document explains the implementation as it exists in the repository and points to concrete entry points in the codebase.

---

## Key Entry Points

### Core crate
- `crates/xlog-prob/src/exact.rs`: exact inference API (`ExactDdnnfProgram`)
- `crates/xlog-prob/src/mc.rs`: Monte Carlo engine (`McProgram`) + non-monotone SCC semantics (`NONMONOTONE_SEMANTICS`)
- `crates/xlog-prob/src/provenance.rs`: provenance extraction + grounding + probabilistic lowering
- `crates/xlog-prob/src/pir.rs`: PIR graph data model
- `crates/xlog-prob/src/cnf.rs`: Tseitin encoding + CNF var mapping
- `crates/xlog-prob/src/compilation/gpu_pir.rs`: GPU PIR graph layout (device-resident SoA)
- `crates/xlog-prob/src/compilation/gpu_pir_intern.rs`: GPU PIR interner (deterministic, memory-bounded)
- `crates/xlog-prob/src/compilation/gpu_cnf.rs`: GPU PIR→CNF encoder (`encode_cnf_gpu`)
- `crates/xlog-prob/src/kc/ddnnf.rs`: Decision-DNNF parser (tests/fixtures only; no CPU D4 compilation)
- `crates/xlog-prob/src/xgcf.rs`: XGCF (GPU circuit format) construction
- `crates/xlog-prob/src/gpu.rs`: GPU upload + evaluation glue (`GpuXgcf`)
- `crates/xlog-prob/src/neural_fast_path.rs`: GPU neural fast-path slot mapping + AD-chain glue
- `crates/xlog-prob/src/compilation/validation.rs`: GPU-native equivalence verifier (`φ ≡ C`) using GPU CDCL (zero host reads)

### CUDA kernels
- `kernels/circuit.cu` / `kernels/circuit.ptx`: forward + backward kernels for XGCF circuits
- `kernels/cache.cu` / `kernels/cache.ptx`: CNF hashing + cache lookup/insert + cache store helpers
- `kernels/cnf.cu` / `kernels/cnf.ptx`: GPU PIR→CNF encoding kernels (`xlog_cnf`)
- `kernels/d4.cu` / `kernels/d4.ptx`: GPU D4 compilation kernels (frontier expansion, smoothing, build)
- `kernels/mc_sample.cu` / `kernels/mc_sample.ptx`: Bernoulli sampling kernel used by `mc`
- `kernels/sat.cu` / `kernels/sat.ptx`: GPU CDCL verifier + GPU-native equivalence query construction helpers
- `kernels/neural.cu` / `kernels/neural.ptx`: neural fast-path AD weight fill + chain-rule gradient scatter (`xlog_neural`)
- `kernels/weights.cu` / `kernels/weights.ptx`: GPU-native weight/evidence builders for exact inference

### Python bindings (DLPack-first)
- `crates/pyxlog/src/lib.rs`: `pyxlog` module (PyO3)
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
3. **GPU PIR → CNF** (`encode_cnf_gpu`, `crates/xlog-prob/src/compilation/gpu_cnf.rs`) with a device-resident var map.
4. **GPU D4 compile + verify**: CNF → device-resident XGCF with cache storage
   (`compile_gpu_d4_and_verify_cached`, `crates/xlog-prob/src/compilation/` + `kernels/d4.ptx` + `kernels/sat.ptx`).
5. **GPU evaluation** via cache-aware kernels (`crates/xlog-prob/src/compilation/gpu_cache.rs` + `kernels/circuit.ptx`):
   - forward pass computes `log WMC(...)` in log-space
   - backward pass computes gradients w.r.t. leaf log-weights

### Conditional probability

For each query variable `Q` and evidence `E`:

- `log Z_E  = log WMC(E)`
- `log Z_EQ = log WMC(E ∧ Q)`
- `log P(Q|E) = log Z_EQ − log Z_E`

This is implemented in `crates/xlog-prob/src/exact.rs` (`ExactDdnnfProgram::evaluate` and `ExactDdnnfProgram::evaluate_gpu_with_grads`).

### GPU state and caching

`ExactDdnnfProgram` compiles CNF on the GPU, invokes GPU D4 + GPU CDCL verification, and stores the resulting circuit in a
device-resident `GpuCircuitCache`. The program holds a cache handle and CUDA provider in `GpuExactState`; evaluations reuse
the cached slot and run cache-aware XGCF kernels with no CPU D4 invocation and no CNF/DDNNF host materialization.

### CPU D4/DDNNF compilation (removed)

The repository no longer includes a CPU D4/DDNNF compilation pipeline:
- No vendored D4 snapshot.
- No shell-out to an external `d4` binary.

Decision-DNNF parsing (`crates/xlog-prob/src/kc/ddnnf.rs`) remains available for tests/fixtures only; production exact
inference uses the GPU-native compiler + verifier and device-resident circuits.

The GPU-native encoder (`encode_cnf_gpu`) in `crates/xlog-prob/src/compilation/gpu_cnf.rs` produces a device-resident
`GpuCnf` for the GPU D4/CDCL pipeline and is now wired into `ExactDdnnfProgram` via
`compile_gpu_d4_and_verify_cached` with a device-resident `GpuCircuitCache`.

---

## GPU-Native Compilation + Verification (v0.5.0 foundation)

XLOG’s target architecture is a **100% GPU-native** compilation + verification path (GPU D4 + GPU CDCL verifier) with
**zero data-plane host transfers**. This path is now integrated into `ExactDdnnfProgram` (see
`docs/design/2026-01-22-gpu-native-compilation-design.md`):

- `xlog_prob::compilation::validate_equivalence_gpu` proves `φ ≡ C` by solving two UNSAT queries on GPU:
  - `UNSAT(φ ∧ ¬C)`
  - `UNSAT(C ∧ ¬φ)`

**Verifier contract:**
- **Zero device→host reads** in the production verifier path (the host never downloads SAT/UNSAT status).
- **Fail-fast on mismatch**: GPU-side assertion/validation kernels trap; the host observes only CUDA success/failure.
- **Capacity-safe CNF handling**: CNF buffers may be allocated with capacity > exact size; all index math uses
  device-resident `GpuCnf::{num_vars,num_clauses,num_lits}`.

This verifier module is used by GPU-native compilation utilities in `crates/xlog-prob/src/compilation/` and now powers
the default `ExactDdnnfProgram` pipeline with a device-resident `GpuCircuitCache`.

**Phase 3 status:** GPU PIR→CNF encoding is implemented and tested via `encode_cnf_gpu` + `kernels/cnf.cu` with
device-resident counts and CSR emission; equivalence tests live in `crates/xlog-prob/tests/gpu_cnf.rs`.

**Phase 4 status:** Cache + integration is implemented: GPU-resident cache (`gpu_cache.rs` + `kernels/cache.cu`),
cache-aware XGCF evaluation, GPU-only exact compilation (`compile_gpu_d4_and_verify_cached`), and guardrails enforcing
no device→host reads in the cache path.

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

Sampling uses `CudaKernelProvider::sample_bernoulli_matrix_device(...)`, which calls `kernels/mc_sample.ptx` to generate a row-major `[samples][num_vars]` matrix of 0/1 bytes on the GPU. Sampling is deterministic given `seed`.

### Device-only evaluation (GPU-native counts)

The primary GPU-native API is:

- `McProgram::evaluate_gpu_device(...) -> McDeviceResult`

It evaluates each sampled world on the GPU (host control-plane launches only) and returns device-resident count buffers:

- `query_counts: DeviceSlice<u32>` (one per query atom)
- `evidence_count: DeviceSlice<u32>` (samples satisfying all evidence)

Internally (`crates/xlog-prob/src/mc.rs`), MC evaluation:

1. Keeps the sample matrix on device and builds per-sample probabilistic fact buffers on the GPU.
2. Evaluates the deterministic core using `xlog-runtime::Executor` (GPU relation store + GPU operators).
3. Computes query/evidence truth flags and accumulates counts on-device (`kernels/mc_eval.cu`).

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

### Host outputs and CPU validation (feature-gated)

Host-readable probability outputs are behind `--features host-io`:

- `McProgram::evaluate(...) -> McResult` downloads the device counts and computes probabilities + confidence intervals.
- `McProgram::evaluate_cpu(...) -> McResult` is a validation/debug path that downloads the sample matrix and evaluates via
  a CPU relation store.

---

## Python API (`pyxlog`)

The PyO3 extension `crates/pyxlog` exposes two entry points:

- `Program.compile(source, device=0, memory_mb=1024, prob_engine=None) -> CompiledProgram`
  - `CompiledProgram.evaluate(return_grads=False, ...) -> EvalResult`
  - outputs probabilities as DLPack tensors (`prob`, `log_prob`)
  - exact engine optionally returns per-query gradients (`grad_true`, `grad_false`)
  - MC engine returns uncertainty metadata and sets `approx=True`
- `LogicProgram.compile(source, device=0, memory_mb=1024) -> CompiledLogicProgram`
  - `CompiledLogicProgram.evaluate(dlpack_inputs={...}) -> LogicEvalResult`

All GPU table interchange is via DLPack capsules (framework-agnostic). See `examples/python/` for end-to-end scripts.

For training workloads with neural predicates, `pyxlog` uses the **GPU neural fast-path** described in
`docs/design/2026-01-22-gpu-native-compilation-design.md` §5.3:
- neural outputs are imported as CUDA tensors via DLPack (no `.tolist()`),
- AD-chain weights and probability gradients are computed on GPU (`kernels/neural.cu`),
- Torch receives device-resident gradients via `output.backward(grad)`.
- The strict GPU-native entrypoint is `CompiledProgram.forward_backward_tensor(query) -> torch.Tensor` which returns a
  CUDA scalar loss (no device->host reads required). The legacy `forward_backward(query) -> f64` helper reads back a
  single scalar for convenience.

---

## Reproducibility

### Rust (workspace)

```bash
cargo test --workspace --all-targets --exclude pyxlog --release
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
cd crates/pyxlog
python -m pip install --upgrade pip maturin
maturin develop --release
python ../../examples/python/03_prob_mc_nonmonotone_torch.py
```

## Provenance Primitives (Rust-only)

`xlog-prob` exposes retained provenance metadata so external Rust consumers can resolve PIR nodes
back to their source atoms and annotated-disjunction metadata. All data is retained inline during
extraction — no new passes or post-hoc reconstruction.

### New type

- `ChoiceSource` — captures explicit annotated-disjunction heads (with probabilities), the
  Bernoulli decision stage index (`choice_index`), and an optional AD identity (`source_id`,
  `None` in v1).

### New fields on `Provenance`

- `leaf_atoms: BTreeMap<LeafId, GroundAtom>` — one entry per probabilistic fact leaf
- `choice_sources: BTreeMap<ChoiceVarId, ChoiceSource>` — one entry per Bernoulli decision variable

### Accessors

- `leaf_atom(LeafId) -> Option<&GroundAtom>` — resolve a PIR leaf to its source atom
- `choice_source(ChoiceVarId) -> Option<&ChoiceSource>` — resolve a decision node to AD metadata
- `atoms_with_formulas() -> impl Iterator<Item = (&GroundAtom, PirNodeId)>` — iterate all atoms
  with provenance formulas (exposes `tuple_formulas` read-only)

### Re-exports

`xlog-prob` lib.rs re-exports key types at crate root:

```rust
pub use pir::{ChoiceVarId, LeafId, PirGraph, PirNode, PirNodeId};
pub use provenance::{ChoiceSource, GroundAtom, Provenance, Value};
```

### Design documents

- `docs/plans/2026-03-08-provenance-primitives-design.md`
- `docs/plans/2026-03-08-provenance-primitives-plan.md`
- `docs/xlog-change-request-provenance.md` (original change request)

---

## See Also

- [dILP Training Architecture](dilp-training.md) — Differentiable ILP (shares XGCF/provenance infrastructure)
- [GPU Execution](gpu-execution.md) — Mask DAG evaluation, stream compaction
- [Python Bindings](python-bindings.md) — User-facing API reference
