# xlog v0.5.0 Technical Whitepaper — Design Spec

**Format:** Technical report (arxiv-style, ~15-20 pages, 7000-8000 words)
**Primary audience:** ML/NeSy researchers
**Secondary audiences:** Systems/PL researchers, potential adopters
**Central narrative:** (a) First GPU-native unified logic platform + (c) Bridging logic engines and deep learning frameworks

---

## Decisions Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Structure | Architecture-first (Approach A) | Natural flow: foundation then applications; suits the unified-platform narrative |
| Benchmark style | Qualitative comparison + absolute numbers | No head-to-head benchmarks available; honest framing |
| Unshipped features | Extensibility in architecture (c) + Future Work section (b) | Shows design foresight without overpromising |
| Benchmark data source | Existing reports/docs (verified on GPU machine) | Not from this session's runs |
| Attribution style | No Co-Authored-By signatures | User preference |

---

## Section Plan

### 1. Introduction (~600 words)

**Opening problem:** Symbolic AI (Datalog, ProbLog, ILP) and neural AI (PyTorch, deep learning) operate in separate worlds with different runtimes, data formats, and execution models. Existing bridges (DeepProbLog, NeurASP) work but run symbolic inference on CPU, creating a bottleneck when training neural-symbolic models at scale.

**The gap:** No existing system performs deterministic evaluation, probabilistic inference, knowledge compilation, SAT verification, and neural-symbolic training entirely on GPU with zero host transfers in production paths.

**xlog's answer:** A unified GPU-native Datalog engine spanning four reasoning paradigms — deterministic, probabilistic, neural-symbolic, and SAT/MaxSAT — with a layered architecture designed for interoperability with the ML ecosystem (DLPack, Arrow, PyTorch autograd).

**Contributions (5 bullets):**
- GPU-resident semi-naive Datalog evaluation with custom CUDA kernels
- GPU-native knowledge compilation pipeline (PIR -> CNF -> D4 -> XGCF) with compile-once/evaluate-many semantics
- End-to-end differentiable neural-symbolic training with circuit caching
- Zero-copy interop with ML frameworks via DLPack and Arrow
- Differentiable ILP with sparse GPU masks and a 6-gate promotion pipeline

**Close:** Paper roadmap paragraph.

---

### 2. Architecture (~1200 words)

**2.1 System Overview**
15-crate layered architecture:
- Tier 0: `xlog-core` (types, errors, symbol interning)
- Tier 1: Domain IRs + providers (`xlog-ir`, `xlog-cuda`, `xlog-stats`, `xlog-neural`)
- Tier 2: Subsystems (`xlog-logic`, `xlog-runtime`, `xlog-solve`)
- Tier 3: Integrated reasoning (`xlog-gpu`, `xlog-prob`)
- Tier 4: User interfaces (`pyxlog`, `xlog-cli`)

Include ASCII or mermaid dependency diagram.

**2.2 Compilation Pipeline**
```
Source -> PEG Parser -> Stratifier (SCC) -> Lowerer (AST->RIR, DP join planning) -> Optimizer -> ExecutionPlan
```
Emphasize cost-aware join planning with cardinality hints.

**2.3 GPU Residency Model**
- Hard guarantee: all runtime semantic state fits in GPU memory or deterministic error
- No silent OOC, no CPU fallback
- Zero D2H in production query paths, verified via byte-level accounting (`host_transfer_stats()`)
- Why it matters: eliminates PCIe round-trips that dominate latency in hybrid approaches

**2.4 IR Stack**
- AST -> RIR -> PIR -> XGCF (each with metadata: cardinality hints, memory estimates)
- Architecture accommodates future IRs (EIR for epistemic, SIR for solver) — extensibility by design, not yet implemented

---

### 3. GPU-Native Datalog Execution (~1000 words)

**3.1 Semi-Naive Evaluation on GPU**
- Stratum-ordered execution, SCC-aware scheduling
- Delta relations on-device, fixpoint iteration
- Custom CUDA kernels (not cuBLAS/cuSPARSE wrappers)

**3.2 Kernel Design (8 core relational kernels)**
- Hash join (inner/semi/anti/left-outer)
- Radix sort (4-bit, multi-column keys, GPU prefix sums)
- Filter (typed comparisons, IEEE 754 total ordering)
- GroupBy (count, sum, min, max, logsumexp)
- Dedup, set ops, scan, pack
- Brief description of each, not deep-dive. Highlight IEEE 754 total ordering as a correctness design choice.

**3.3 Adaptive Join Planning**
- DP join planner uses runtime statistics from `xlog-stats`
- Feedback loop: execution stats inform next compilation's cost model

**3.4 Reversible Symbols**
- Bidirectional string-to-u32 mapping, sequential allocation
- Efficient GPU storage with human-readable output
- Arrow dictionary encoding for export

---

### 4. Probabilistic Inference on GPU (~1500 words)

Heaviest section — carries the most technical weight.

**4.1 Approach**
- ProbLog's compile-once/evaluate-many model, but entire pipeline on GPU
- Contrast: ProbLog compiles and evaluates on CPU; xlog does both on device

**4.2 GPU Knowledge Compilation Pipeline**
Step-by-step with pipeline diagram:
1. Provenance extraction (PIR) — probabilistic fact labels propagated through derivations on-device (`pir.cu`)
2. CNF encoding — Tseitin transformation on GPU, zero host reads (`cnf.cu`)
3. D4 compilation — Decision-DNNF compiler on GPU (`d4.cu`, most technically novel kernel)
4. CDCL verification — GPU SAT solver proves equivalence (phi equiv C) via two UNSAT checks (`sat.cu`). Complete verifier, not heuristic.
5. XGCF evaluation — Forward: weighted model count in log-space. Backward: gradients for training (`circuit.cu`)

**4.3 Circuit Caching**
- D4 compilation expensive (cold start ~75s on MNIST addition)
- Circuit structure depends on program, not weights
- Cache compiled XGCF; subsequent iterations only update leaf weights + re-evaluate
- Measured: 2.74x speedup (95% CI: [2.29, 3.18]), steady-state epoch: 75s -> 0.25s

**4.4 Monte Carlo Sampling**
- GPU-parallel world sampling for programs too large for exact inference
- Two methods: rejection sampling, evidence clamping (auto-selected on Bernoulli evidence)
- Confidence intervals reported
- MC optimization: 8.6% wall-clock improvement (14.11s -> 12.90s, 1K-sample benchmark)

**4.5 Well-Founded Semantics**
- Support for non-monotone negation through WFS
- Enables programs with cyclic negation that stratification can't handle

---

### 5. Neural-Symbolic Bridge (~1200 words)

Primary audience section.

**5.1 Neural Predicates**
- `nn/4` Datalog syntax: a neural network as a predicate
- PyTorch networks registered via Python API, called during evaluation
- Forward: network outputs become probabilistic facts -> knowledge compilation pipeline
- Backward: gradients from circuit evaluation -> PyTorch autograd -> network
- GIL released during GPU work

**5.2 End-to-End Training Loop**
Concrete walkthrough of MNIST addition (01_minimal):
1. Register PyTorch digit classifiers as neural predicates
2. Compile probabilistic addition program (once) -> XGCF circuit
3. Per epoch: forward through network -> inject as leaf weights -> evaluate circuit -> loss -> backward through circuit -> backward through network -> optimizer step
4. Circuit reused across epochs — only leaf weights change

This subsection gets the most space — it's the narrative core for ML readers.

**5.3 Term Embeddings (v0.5.0)**
- `register_embedding()`: attach nn.Embedding or frozen tensors to Datalog symbols
- `forward_embedding()`: retrieve during evaluation
- Entity representation learning within the logic framework

**5.4 Differentiable ILP (Beta)**
- Sparse GPU mask API — avoid N^3 Python-side materialization
- 6-gate promotion pipeline: convergence, novelty, regression, holdout F1, ambiguity, typed schema
- Hard-negative mining
- Artifact save/load with SHA-256 verification
- Clearly labeled as beta

---

### 6. Interoperability (~600 words)

The "bridging" narrative.

**6.1 DLPack** — Zero-copy GPU tensor sharing, bidirectional with PyTorch/JAX/NumPy

**6.2 Arrow IPC** — Columnar export, dictionary encoding for symbols, ecosystem integration (Pandas, Polars, DuckDB)

**6.3 Python Bindings** — PyO3, .pyi type stubs, GIL release during GPU ops

**6.4 PyTorch Autograd** — Circuit evaluation participates in autograd graph, gradients flow seamlessly, no custom autograd reimplementation

---

### 7. Evaluation (~1000 words)

**7.1 Methodology**
- Hardware: NVIDIA RTX PRO 3000 Blackwell (12GB, SM120)
- Release mode, median of N runs, confidence intervals where applicable
- Numbers sourced from verified test reports (run on GPU machine), not ad-hoc

**7.2 Absolute Performance**

*Deterministic:*
- Transitive closure throughput (100K, 1M edges)
- Hash join throughput (100K x 100K, 1M x 100K)

*Probabilistic:*
- Exact inference latency (20-var, 50-var)
- MC sampling throughput (worlds/sec)

*Neural-symbolic (MNIST addition, 01_minimal):*
- Cold start: ~75s (D4 + CDCL)
- Steady-state epoch: ~0.25s (cached)
- Cache speedup: 2.74x (95% CI: [2.29, 3.18])
- Final accuracy: 99.07% held-out (20 epochs)
- Per-query: ~1.0ms forward+backward

**7.3 Qualitative Comparison Table**

| | DeepProbLog | ProbLog2 | GPUlog | xlog |
|---|---|---|---|---|
| Symbolic execution | CPU | CPU | GPU | GPU |
| Probabilistic inference | CPU | CPU | -- | GPU |
| Knowledge compilation | CPU | CPU | -- | GPU |
| Neural integration | Yes | No | No | Yes |
| Zero-copy ML interop | No | No | No | Yes |
| Differentiable ILP | No | No | No | Yes (beta) |

Cite published numbers from GPUlog (45x), VFLog (200x), DeepProbLog. Frame: "each solves a piece; xlog unifies on GPU."

**7.4 CUDA Certification**
- 206 tests, 25 categories, 100% passing — engineering quality signal

---

### 8. Related Work (~800 words)

**GPU Datalog:** GPUlog, VFLog, mnmgDatalog — deterministic only, no probabilistic/neural

**Probabilistic Logic Programming:** ProbLog2, DeepProbLog, NeurASP — CPU-bound symbolic inference

**GPU SAT:** ParaFROST (complete, proof generation), FastFourierSAT (incomplete, continuous local search) — standalone solvers, not embedded in logic programming

**Differentiable ILP:** dILP (Evans & Grefenstette) — CPU-based, dense rule materialization vs xlog's sparse GPU masks

**Positioning:** Each system excels in its niche. xlog unifies these capabilities on a single GPU-resident platform with zero-copy ML interop — not superiority on any single axis, but the integrated stack.

---

### 9. Limitations & Future Work (~500 words)

**9.1 Current Limitations**
- NVIDIA GPU required (no AMD/Intel/Apple Silicon)
- All data must fit in GPU memory (no out-of-core)
- Python batch query coerces to u32 entity IDs
- dILP is beta
- No formal head-to-head benchmarks against DeepProbLog

**9.2 Future Work** (clearly labeled as not implemented)
- Epistemic logic (xlog-elp): world views, modal K/M operators, EIR slot reserved
- Out-of-core execution: streaming for programs exceeding GPU memory
- Magic sets: query-directed optimization
- Multi-GPU: partitioned evaluation (inspired by mnmgDatalog)
- Incremental parsing: re-compile only changed rules

---

## Diagrams Needed

1. **Crate dependency diagram** (Section 2.1) — tiered layout showing all 15 crates
2. **Compilation pipeline** (Section 2.2) — Source to ExecutionPlan
3. **Knowledge compilation pipeline** (Section 4.2) — PIR -> CNF -> D4 -> CDCL -> XGCF -> Eval
4. **Training loop** (Section 5.2) — MNIST addition end-to-end flow
5. **Qualitative comparison table** (Section 7.3)

---

## Constraints

- Every performance claim must cite a source (benchmark report, CHANGELOG entry, test output)
- Every feature claim must be verified against the audited codebase
- No features described that don't exist or don't work
- Code examples must be copy-pasteable from actual examples
- Unshipped features only in Section 9.2, clearly labeled
