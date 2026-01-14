# XLOG Design Document

**Project:** XLOG (GPU-Native Logic / Probabilistic / Epistemic Programming)
**Subsystems:** `xlog-logic`, `xlog-prob`, `xlog-elp`, `xlog-solve`
**Primary constraint:** End-to-end **GPU-resident** execution for *semantic evaluation* (facts, intermediate relations, inference state, and solver state remain on GPU). Host involvement is limited to orchestration, I/O, and compilation.

> **Implementation status (2026-01-14):** Phase 3 (`xlog-logic`) and Phase 4 (`xlog-prob` exact `exact_ddnnf` + approximate `mc`, plus Python `xlog_gpu`) are implemented on `main`. See `docs/VALIDATION_REPORT.md` and `docs/architecture/xlog-prob.md`.

---

## 1. Executive Summary

XLOG is a unified, GPU-native declarative programming stack that targets four closely-related reasoning paradigms:

1. **`xlog-logic`**: Datalog/Prolog-like *deterministic* rule evaluation (recursive fixpoint computation).
2. **`xlog-prob`**: ProbLog/DeepProbLog-like *probabilistic* and differentiable reasoning via semiring provenance and circuit-style evaluation (Weighted / Algebraic Model Counting).
3. **`xlog-elp`**: ASP-style *epistemic logic programming* (ELP) with world views and modal operators **K** (“known”) and **M** (“possible”), plus an epistemic negation operator interpreted over sets of answer sets/worlds. This area has multiple semantics; XLOG must be explicit and robust about which semantics it implements. ([Texas Tech University][1])
4. **`xlog-solve`**: GPU-native *search/solving* services (SAT/MaxSAT-like core + ASP/ELP encodings + optimization), designed to be the shared inner engine used by `xlog-prob` and `xlog-elp`.

The core novelty is **not** “running Datalog on the GPU” (that exists), but a **single, coherent GPU-first platform** where:

* deterministic recursion (`xlog-logic`) uses high-performance GPU relational algebra kernels, building on modern GPU Datalog insights like HISA/range-indexing and column-oriented layouts; ([Thomas Gilray's Research][2])
* probabilistic inference (`xlog-prob`) uses GPU-evaluable provenance graphs/circuits similar in spirit to ProbLog’s compilation + evaluation (but redesigned so evaluation and learning are GPU-native); ([Departement Computerwetenschappen][3])
* epistemic reasoning (`xlog-elp`) is supported with a **GPU-parallel world-view engine** based on **generate-and-test with propagation** plus **epistemic splitting** to decompose programs, minimizing the number of candidate world views that require full checks; 
* the solver layer (`xlog-solve`) is engineered around *GPU-suitable search primitives* (massively-parallel local search / continuous local search, GPU-accelerated inprocessing, and GPU-friendly verification strategies), informed by recent GPU SAT research. ([arXiv][4])

---

## 2. Design Goals and Non-Goals

### 2.1 Goals

**G1 — GPU-resident semantic evaluation.**
All semantic evaluation data structures (facts, derived relations, solver state, circuit values, sampling state) remain GPU-resident during execution.

**G2 — CuDF-first, custom-kernel-where-necessary.**
XLOG uses cuDF/libcudf for baseline tabular operations (joins, groupby, aggregations) where feasible. ([RAPIDS Docs][5])
For recursion, dedup/incremental maintenance, and solver kernels, XLOG provides custom CUDA kernels when cuDF is insufficient or non-optimal.

**G3 — Formal semantics with explicit tiers.**
XLOG must provide *explicit semantics choices* and “tiers” (exact vs approximate) per subsystem, especially for ELP.

**G4 — Practical implementability.**
A staged implementation plan delivers value early (GPU Datalog + neural facts) and grows toward full ELP support.

**G5 — Robustness and verifiability.**
Where “exactness” is claimed, XLOG includes proof/certificate artifacts or cross-check capability, inspired by “certified” GPU inprocessing ideas in SAT. ([Springer][6])

### 2.2 Non-Goals (initial releases)

* **NG1:** Full Prolog with unrestricted backtracking, cuts, and arbitrary term structures (XLOG is relational/columnar-first; it can later add constrained backtracking as a specialized feature).
* **NG2:** Full general-purpose ELP under *every* proposed semantics; XLOG must pick a default semantics and support a small set of well-motivated alternatives.
* **NG3:** A fully general-purpose GPU CDCL solver competitive with best CPU solvers on all SAT benchmarks in v1 (XLOG targets workloads with high parallelism and repeated solving patterns where GPU shines).

---

## 3. Background and Key Constraints from Prior Work

### 3.1 GPU Datalog reality check

Modern GPU Datalog engines demonstrate that recursion + joins can be GPU-fast with specialized data structures:

* **GPUlog** uses a **hash-indexed sorted array (HISA)** supporting efficient range queries (joins), lock-free deduplication, and parallel iteration, reporting up to **45×** speedups vs a leading CPU engine on certain workloads. ([Thomas Gilray's Research][2])
* **VFLog** shows that a **column-oriented** GPU Datalog runtime can deliver **200×** gains over CPU column engines and outperform prior GPU Datalog in workloads including KRR. ([arXiv][7])
* **mnmgDatalog** demonstrates feasibility and key challenges of **multi-node, multi-GPU Datalog**, highlighting workload partitioning, GPU-to-GPU communication, and kernel design. ([hpcrl.github.io][8])

XLOG must adopt these lessons: recursion is feasible, but only if the runtime is built around incremental relational operations and GPU-suitable storage/indexes.

### 3.2 Probabilistic logic reality check

ProbLog’s common operational approach is:

1. ground relevant parts of the program,
2. break cycles,
3. compile to a logical form (e.g., d-DNNF/SDD/BDD),
4. evaluate via weighted model counting. ([Departement Computerwetenschappen][3])

DeepProbLog extends this by introducing **neural predicates** and adapting inference/learning to integrate deep nets and ProbLog-style reasoning end-to-end. ([NeurIPS Proceedings][9])
NeurASP treats neural network outputs as probability distributions over facts in an ASP program. ([IJCAI][10])

XLOG’s probabilistic subsystem must preserve the core idea (a quantitative inference backend) but redesign compilation/evaluation so the *hot path* is GPU-native.

### 3.3 Epistemic logic programming reality check

ELPs extend ASP with epistemic operators and world views:

* Gelfond-style epistemic specifications distinguish **objective** vs **subjective** literals (K/M), define satisfaction over sets of belief sets, and define world views via reducts. ([Texas Tech University][1])
* Multiple semantics exist; original approaches can yield **self-supported** world views. This motivated “foundedness” criteria and new semantics such as **Founded Autoepistemic Equilibrium Logic (FAEEL)**. 
* **eclingo** is a practical solver for Gelfond 1991 semantics built on clingo and uses a **guess-and-check** strategy; the eclingo project also references both Gelfond 1991 and founded world views work. ([arXiv][11])
* Existence of a world view is computationally high in the polynomial hierarchy (Σ_P^3-complete in general, as discussed in ELP literature), highlighting the need for decomposition/propagation and parallel strategies. ([IJCAI][12])
* Recent work proposes **generate-and-test with propagation** to cut the number of candidates needing full testing, potentially exponentially reducing candidates with only linear overhead. 
* **Epistemic splitting** provides a modular decomposition principle and (for epistemically stratified programs) implies uniqueness of world view under semantics satisfying the property; it also helps reduce solving scope. 

For XLOG, ELP is the “hardest” component. The correct strategy is not to claim universal exactness immediately, but to implement:

* a default semantics designed for robustness (avoid unintended/self-supported world views),
* a GPU-parallel solver architecture, and
* exactness guarantees on well-defined fragments first.

---

## 4. System Overview

### 4.1 High-level architecture

XLOG has one front-end and four backends:

* **Front-end:** XLOG compiler + static analyzer + optimizer (CPU-resident compilation is acceptable; runtime evaluation stays GPU-resident).
* **Shared GPU runtime substrate:**

  * GPU relation store (columnar base + optional indexes)
  * kernel library (joins, dedup, aggregation, fixpoint scheduling)
  * memory manager + spill policy (GPU-first; optional UVM spill as last resort)

Backends:

1. `xlog-logic` runtime: GPU fixedpoint engine (semi-naïve evaluation).
2. `xlog-prob` runtime: provenance/circuit builder + GPU evaluator + autodiff hooks.
3. `xlog-elp` runtime: world-view engine built around splitting + generate/test/propagate.
4. `xlog-solve` runtime: GPU solving services for SAT/MaxSAT-style cores, plus encodings used by `xlog-prob` and `xlog-elp`.

### 4.2 Core intermediate representations

To keep the platform coherent, XLOG uses layered IRs:

* **RIR (Relational IR):** joins/projections/unions/dedup + fixpoint loops.
* **PIR (Provenance IR):** semiring expressions, probabilistic facts, annotated derivations, circuit graph.
* **EIR (Epistemic IR):** epistemic atoms, modal constraints, world-view requirements, splitting structure.
* **SIR (Solve IR):** boolean constraints, weighted constraints, optimization objectives, model enumeration requests.

All IR nodes must have:

* GPU cost model metadata (estimated row counts, skew, memory),
* determinism/reproducibility flags,
* incremental update semantics (delta rules).

---

## 5. XLOG Language Surface

XLOG is a *family* of languages sharing syntax where possible. You will expose them as distinct “profiles”:

* **xlog-logic:** deterministic rules (Datalog-style)
* **xlog-prob:** probabilistic facts + semiring annotations + neural predicates
* **xlog-elp:** epistemic operators + world-view queries/constraints
* **xlog-solve:** constraints, optimization, and solver directives

### 5.1 Common syntax (illustrative)

```prolog
// Deterministic rule
reach(X,Y) :- edge(X,Y).
reach(X,Y) :- edge(X,Z), reach(Z,Y).

// Constraint (ASP-like)
:- reach(X,X).     // forbid cycles

// Probabilistic fact (xlog-prob profile)
0.7::edge(a,b).

// Neural predicate (xlog-prob profile; DeepProbLog-like concept)
nn(mnist, Img, Digit) :: digit(Img, Digit).

// Epistemic constraint (xlog-elp profile)
interview(X) :- not K eligible(X), not K not eligible(X).
```

Notes:

* `:-` is rule implication; `:-` with empty head is a constraint.
* `K` and `M` (or `not`-epistemic) are reserved for `xlog-elp`.
* Probabilistic annotations and neural declarations are reserved for `xlog-prob`.

---

## 6. Semantics and Tiering

XLOG must be explicit about what is guaranteed.

### 6.1 `xlog-logic` semantics

* Base: Datalog with stratified negation (initially).
* Evaluation: semi-naïve fixpoint iteration; relational algebra kernels executed on GPU.

Exactness: **Exact** for supported fragment.

### 6.2 `xlog-prob` semantics

* Base: ProbLog-like distribution semantics; evaluation via compilation to an evaluatable form and model counting. ProbLog explicitly uses knowledge compilation and supports evaluatables like d-DNNF/SDD/BDD. ([Departement Computerwetenschappen][3])
* Generalization: semiring-based evaluation (aProbLog-style). ([AAAI Open Access Journal][13])

Tiering:

* **P1 (Exact, circuit-evaluable):** programs compiled into decomposable circuits evaluated exactly on GPU.
* **P2 (Exact, restricted structure):** acyclic probabilistic dependencies or bounded-treewidth fragments.
* **P3 (Approximate):** sampling / variational approximations with GPU-parallel estimators and calibrated uncertainty.

### 6.3 `xlog-elp` semantics

**Key design decision:** choose a default semantics that is robust to “unintended world views.”
The literature explicitly notes self-supported world views under original G91 semantics and proposes foundedness and semantics like FAEEL to address this. 

#### Supported semantics modes (XLOG v1 target)

* **ELP-S1 (Default): FAEEL-style founded world views**

  * Rationale: addresses self-supportedness via foundedness/unfounded-set ideas. 
* **ELP-S2 (Compatibility): Gelfond 1991 (G91)**

  * Rationale: broad historical reference and supported in tools like eclingo. ([arXiv][11])

#### Evaluation strategy (algorithmic core)

XLOG adopts a **generate-and-test** approach with two key accelerators:

1. **Propagation of epistemic consequences** to reduce candidates, following modern proposals that can drastically reduce the number of guesses needing testing. 
2. **Epistemic splitting** to decompose programs into bottom/top components where applicable, yielding modular evaluation and (for epistemically stratified programs under splitting-satisfying semantics) uniqueness results. 

Tiering:

* **E1 (Exact):** epistemically stratified programs, or programs admitting splitting decomposition into small components, with bounded epistemic atom count per component.
* **E2 (Exact but bounded):** general programs with bounded epistemic atoms; brute-force candidates are parallelized on GPU with pruning/propagation.
* **E3 (Approximate):** sampling-based world-view approximation (clearly labeled), for programs beyond feasible bounds.

### 6.4 `xlog-solve` semantics

`xlog-solve` provides a set of solver services rather than a single semantics:

* **S-SAT:** boolean satisfiability / model enumeration (GPU-oriented).
* **S-MaxSAT:** weighted optimization.
* **S-ASP-core:** answer-set solving for supported ASP fragments via compilation/encodings.
* **S-ELP-check:** inner checks needed by `xlog-elp`.

Implementation strategy is hybrid:

* Massive parallel search primitives informed by GPU SAT work like FastFourierSAT’s GPU-accelerated continuous local search approach. ([arXiv][4])
* GPU-accelerated simplification/inprocessing and proof/certificate strategies inspired by ParaFROST’s “certified” GPU inprocessing line of work. ([Springer][6])

---

## 7. Runtime Substrate

### 7.1 GPU relation store

**Baseline:** cuDF/libcudf for standard joins, groupby, and aggregations. ([RAPIDS Docs][5])

**Augmentation (required for recursion + incremental maintenance):**

* delta/full relation maintenance (semi-naïve evaluation)
* lock-free or low-contention deduplication
* optional range index / hash index

**Design:** “Columnar base + optional indexes”

* Base storage: columnar buffers (compatible with cuDF and VFLog design assumptions). ([arXiv][7])
* Optional index: HISA-like structure for relations where range queries and dedup dominate, based on GPUlog’s findings. ([Thomas Gilray's Research][2])

XLOG chooses storage per relation via profiling:

* small/medium relations: cuDF join + sort/dedup
* large recursive workloads: build HISA-like index to accelerate repeated joins and incremental updates.

### 7.2 Fixpoint scheduler (xlog-logic)

* Rule compilation to relational algebra kernels (joins/projections/unions)
* Semi-naïve delta iteration
* Heuristic rule ordering and kernel fusion
* Deterministic tie-breaking for reproducibility

This aligns with the modern “compile Datalog to iterative relational algebra kernels” architecture. ([hpcrl.github.io][8])

### 7.3 Multi-GPU scaling

Adopt a design compatible with mnmgDatalog:

* hash-based partitioning on join keys
* GPU-aware all-to-all exchange
* overlap communication and compute
* local aggregation where possible ([hpcrl.github.io][8])

XLOG’s multi-GPU support is initially:

* single-node multi-GPU (NVLink/NVSwitch)
  then
* multi-node MPI + CUDA-aware communication (advanced).

---

## 8. Subsystem Designs

## 8.1 `xlog-logic`: Deterministic GPU Logic Engine

### Responsibilities

* Parse and type-check deterministic rules
* Compile to RIR
* Execute recursive evaluation to fixpoint
* Provide materialized relations to other subsystems

### Key design choices

1. **Semi-naïve evaluation as the default** for recursion.
2. **Index selection:** build HISA-like or columnar indexes based on observed workload.
3. **Dedup strategy:** lock-free or segmented dedup; GPUlog identifies tuple merge/dedup as a dominant bottleneck and designs around it. ([Thomas Gilray's Research][2])

### Deliverables (MVP)

* binary and n-ary relations
* transitive closure, reachability, simple static analyses
* incremental updates (EDB changes) supported via delta recomputation

---

## 8.2 `xlog-prob`: Probabilistic & Differentiable Engine

### Responsibilities

* Support probabilistic facts and semiring annotations
* Build provenance (PIR) for queried outputs
* Compile provenance to GPU-evaluable form
* Provide gradients for learning (DeepProbLog-like end-to-end training behavior)

### Design anchor: “compile-once, evaluate-many”

ProbLog highlights the practical importance of compiling once and re-evaluating with different evidence; it also exposes multiple evaluatables (d-DNNF/SDD/BDD). ([Departement Computerwetenschappen][3])

XLOG adopts this pattern but targets GPU-native representation:

* compile PIR → **GPU circuit graph** (DAG)
* support repeated evidence updates and re-evaluation without rebuilding everything

### Circuit evaluation on GPU

* represent circuit as a topologically sorted DAG
* level-by-level parallel evaluation (warp-coherent)
* semiring ops are customizable (aProbLog-style). ([AAAI Open Access Journal][13])

### Neural integration

DeepProbLog integrates neural predicates into probabilistic logic programming and adapts inference/learning. ([NeurIPS Proceedings][9])
NeurASP treats neural outputs as distributions over facts consumed by ASP rules. ([IJCAI][10])

XLOG design:

* “neural facts” are represented as a *batched probabilistic fact table* on GPU:

  * key columns for grounding identifiers
  * value columns for probabilities/logits
* `xlog-prob` consumes these tables directly (no CPU marshaling per example)

### Exactness and practicality

* exact probabilistic inference is feasible when provenance compiles to tractable circuits; otherwise fall back to approximate sampling with explicit uncertainty.

---

## 8.3 `xlog-elp`: Epistemic Logic Programming Engine

### Responsibilities

* Provide epistemic operators K/M (and/or epistemic negation forms) over collections of answer sets (“world views”).
* Compute world views under a selected semantics (default: founded/FAEEL; optional: G91).
* Support epistemic constraints and queries.

### Semantics choice and robustness

The literature points out self-supported world views under G91 and introduces foundedness and new semantics based on autoepistemic + equilibrium logic, aiming to capture founded world views. 
XLOG’s default is therefore a **founded-world-view** semantics mode (ELP-S1), with G91 as a compatibility mode.

### Core algorithm: GPU-parallel Generate–Propagate–Test

**Why this algorithm?**

* eclingo uses a guess-and-check strategy based on generating truth values for subjective literals and then checking via cautious/brave consequences. ([arXiv][11])
* Newer work proposes generate-and-test frameworks with **propagation** that can exponentially reduce the number of candidates. 

**XLOG implementation plan:**

1. **Extract epistemic atoms** (subjective literals).
2. **Construct candidate generator**:

   * represent guesses as bitvectors (GPU-friendly)
   * use propagation constraints to prune
3. **Apply epistemic splitting** when possible:

   * split into bottom (objective) and top (subjective references only) components.
   * bottom computed once; top simplified per bottom world view. 
4. **Test candidates** using `xlog-solve`:

   * check existence/consistency of stable models of the epistemic reduct(s)
   * compute brave/cautious consequences needed to validate K/M assignments
5. **Assemble world views** (and optionally return them; also support existence-only queries).

### GPU parallelism strategy

ELP solving involves many candidate checks. This maps naturally to GPU if we:

* batch candidate guesses into large warps/blocks
* run the inner solver checks in parallel across candidates
* share intermediate compiled encodings across candidates

### Exactness constraints (initially)

Because world-view reasoning can be very hard (high complexity class, per ELP literature), XLOG provides exactness guarantees initially on:

* epistemically stratified programs (benefits from splitting and uniqueness results) 
* bounded epistemic atoms per component (enables GPU-parallel brute force with pruning)

For larger programs:

* provide approximate world-view sampling with explicit confidence bounds.

### Alternative/secondary strategy: Single-shot translations

Single-shot approaches translate ELPs into ASP encodings solvable “in one shot,” rather than multiple solver calls. ([IJCAI][12])
XLOG can adopt this as an optimization path:

* if translation yields a compact SIR instance, it can be faster than iterative guess-check.

### Longer-term strategy: QBF/ASP(Q) route (research track)

There are proposals to rewrite ELPs into ASP with quantifiers, solved via QBF solvers. ([CEUR-WS][14])
This is relevant as a *verification back-end* or long-term exactness extension, but is not the best v1 path for full GPU-only execution due to the current state of GPU QBF technology.

---

## 8.4 `xlog-solve`: GPU Solver Services

### Responsibilities

* Provide satisfiability, optimization, enumeration, and checking services to other subsystems.
* Offer reusable solver kernels with GPU-friendly designs.

### Solver kernel strategy (GPU-first)

A purely GPU CDCL solver is not the only viable approach; GPU SAT progress suggests that GPU-friendly methods include:

* massively-parallel local/continuous local search, where gradient-like computations can be accelerated on GPU (FastFourierSAT). ([arXiv][4])
* GPU-accelerated inprocessing/simplification and proof generation approaches (ParaFROST line). ([Springer][6])

**XLOG v1 design choice:**

* `xlog-solve` provides two complementary engines:

  1. **Solve-LS:** GPU local/continuous local search engine for rapid candidate generation and optimization.
  2. **Solve-Exact (bounded):** exact SAT/ASP checking for bounded-size instances where GPU-parallel reasoning is feasible, with optional certificates.

### Certification / robustness

ParaFROST highlights “certified” GPU-accelerated simplifications and proof generation for credibility. ([Springer][6])
XLOG adopts an analogous philosophy:

* when claiming “exact,” generate a checkable artifact:

  * DRAT-like proofs for CNF reductions (where applicable),
  * or re-check via an independent kernel (GPU-based checker) to avoid CPU fallback.

---

## 9. Cross-Subsystem Integration

### 9.1 How `xlog-elp` and `xlog-prob` coexist

XLOG supports mixed pipelines but keeps semantics explicit:

* Deterministic rules (`xlog-logic`) produce objective facts.
* Probabilistic rules (`xlog-prob`) produce distributions over facts or numeric outputs.
* Epistemic reasoning (`xlog-elp`) reasons about *world views of answer sets*.

**Recommended integration pattern (robust and implementable):**

* Convert probabilistic outputs into objective atoms via:

  * thresholding constraints (e.g., `p(a) :- prob(a) > 0.9.`)
  * top-k selection
  * sampling: generate sampled worlds, then reason epistemically over the sampled set (approximate and labeled as such)

This avoids underspecified “probabilistic epistemic logic” semantics while still enabling practical combined workflows.

### 9.2 Deep learning integration (training loops)

* `xlog-prob` provides differentiable objectives (DeepProbLog-like). ([NeurIPS Proceedings][9])
* `xlog-solve` provides constraint satisfaction / structured loss signals (NeurASP-like). ([IJCAI][10])

XLOG exposes:

* a PyTorch extension interface where:

  * neural outputs are written into GPU fact tables,
  * XLOG runs inference/constraints on GPU,
  * gradients flow back.

---

## 10. Performance Engineering Plan

### 10.1 Core bottlenecks to engineer for

1. **Dedup and delta/full merge** (known major bottleneck in GPU Datalog engines). ([Thomas Gilray's Research][2])
2. **Skewed joins** (hot keys cause load imbalance).
3. **Memory pressure** from intermediate relations and provenance graphs.
4. **Candidate explosion** in ELP world view solving.

### 10.2 Techniques

* Adaptive index building (HISA-like) for repeated range queries. ([Thomas Gilray's Research][2])
* Columnar kernels for joins and aggregations (VFLog direction). ([arXiv][7])
* Workload partitioning and GPU-aware all-to-all for multi-GPU. ([hpcrl.github.io][8])
* Epistemic splitting + propagation to reduce ELP candidate checks. 
* GPU circuit evaluation: levelized DAG evaluation with fused semiring ops.
* Solver batching: run many similar checks in parallel with shared memory layouts.

---

## 11. Validation, Correctness, and Robustness

### 11.1 Correctness strategy per module

* `xlog-logic`: cross-check against a reference CPU Datalog engine on random small instances; property-based testing.
* `xlog-prob`: validate against ProbLog on benchmark programs where compilation is tractable; ProbLog’s compilation/evaluation workflow is documented and provides stable reference behavior. ([Departement Computerwetenschappen][3])
* `xlog-elp`: validate against eclingo/selp on ground benchmarks for supported semantics modes (G91 compatibility mode especially). eclingo explicitly implements G91 via guess-check. ([arXiv][11])
* `xlog-solve`: for exact mode, require proof/certificate checking; follow the spirit of certified GPU inprocessing work. ([Springer][6])

### 11.2 Determinism & reproducibility

* deterministic sorting and hash seeds
* fixed floating-point mode options for probabilistic semiring evaluation
* seeded sampling for approximate tiers

---

## 12. Benchmarking and Evaluation Plan

### 12.1 Deterministic logic

* transitive closure, same-generation, points-to analysis (mirroring GPU Datalog literature). ([Thomas Gilray's Research][2])

### 12.2 Probabilistic/neural

* DeepProbLog-style tasks (MNIST addition, etc.) and program induction examples. ([NeurIPS Proceedings][9])
* NeurASP-style constraints over neural outputs. ([IJCAI][10])

### 12.3 Epistemic

* standard ELP benchmark suites used by solvers (planning, introspection examples)
* measure candidate pruning effectiveness (propagation impact) per generate-and-test with propagation insights. 

### 12.4 Solver kernels

* SAT/MaxSAT microbenchmarks focusing on throughput for batched instances
* compare GPU LS vs exact bounded mode on structured encodings

---

## 13. Implementation Roadmap

### Phase 0 — Foundations

* GPU relation store abstraction: cuDF tables + custom kernels interface
* RIR and RIR→GPU kernel compiler
* GPU memory manager + profiling hooks

### Phase 1 — `xlog-logic` MVP

* stratified Datalog (no disjunction)
* semi-naïve fixpoint engine
* dedup/merge kernel library

### Phase 2 — `xlog-prob` MVP

* probabilistic facts + deterministic rules
* provenance capture
* GPU circuit evaluation for a restricted tractable subset
* neural fact table ingestion (PyTorch integration)

### Phase 3 — `xlog-solve` MVP

* GPU local/continuous local search engine (Solve-LS)
* batched constraint evaluation kernels
* bounded exact checker for small encodings

### Phase 4 — `xlog-elp` MVP

* implement ELP-S2 (G91 compatibility) first:

  * generate/test candidate subjective literals and validate via inner checks (eclingo-like architecture). ([arXiv][11])
* add epistemic splitting detection and decomposition. 
* add propagation-based pruning (Fandinno & Lillo line). 

### Phase 5 — Robust default semantics and scaling

* implement ELP-S1 (founded/FAEEL-style default) 
* multi-GPU scaling for `xlog-logic` (single-node first, then multi-node inspired by mnmgDatalog). ([hpcrl.github.io][8])
* expand exactness envelopes and improve proof/certification pipeline in `xlog-solve`

---

## 14. Key Open Decisions and “Hidden Gaps” (Now Explicit)

These are the items you must decide early because they determine architecture and feasibility:

1. **ELP default semantics:** FAEEL-style founded world views vs other modern semantics. XLOG chooses founded/FAEEL as default because of the explicit “foundedness” robustness motivation in the literature. 
2. **ASP/ELP inner solving strategy:**

   * SAT-based encodings vs native stable model computation kernels
   * single-shot translation vs multi-shot guess-check
     XLOG uses guess-propagate-test with optional single-shot fast paths.
3. **Probabilistic compilation location:** full GPU vs CPU-assisted compilation.
   XLOG v1: compilation can be CPU (front-end) but the repeated evaluation + learning loops are GPU-resident; later, introduce GPU-assisted compilation for dynamic programs.
4. **Data structure choice:** pure cuDF vs cuDF + HISA/VFLog-style custom storage.
   XLOG chooses hybrid: cuDF baseline + custom indexes/kernels for recursion hot paths. ([Thomas Gilray's Research][2])
5. **Exactness envelope disclosure:** must be explicit in the API (program-class detection + tier selection + warnings).
6. **Multi-GPU strategy:** begin with single-node multi-GPU, then adopt distributed strategies inspired by mnmgDatalog (hash-based distribution, CUDA-aware comm). ([hpcrl.github.io][8])

---

## 15. Summary of the “Most Robust, Novel, Efficient, Implementable” Solution

**Robustness**

* Default ELP semantics: founded world views (FAEEL-style) to avoid self-supported world views. 
* Explicit tiers and correctness artifacts for exact modes (inspired by certified GPU SAT work). ([Springer][6])

**Efficiency**

* GPU-native semi-naïve recursion with hybrid storage/indexing inspired by GPUlog/VFLog. ([Thomas Gilray's Research][2])
* GPU-parallel ELP candidate evaluation with splitting + propagation to reduce candidate explosion. 
* Batched solver services leveraging GPU-friendly approaches (CLS/local search, GPU simplifications). ([arXiv][4])

**Novelty**

* A *single* GPU-resident platform that unifies:

  * Datalog-style recursion,
  * ProbLog/DeepProbLog-style differentiable probabilistic reasoning,
  * Epistemic ASP world-view reasoning,
  * and a shared GPU solve layer,
    all compiled through a coherent IR stack and executed without CPU fallback in the semantic hot path.

**Implementability**

* The roadmap deliberately sequences components so each phase produces a usable system:

  * `xlog-logic` first (usable immediately)
  * `xlog-prob` next (DeepProbLog-like workloads)
  * `xlog-solve` then (shared infrastructure)
  * `xlog-elp` once inner solving is mature

---

The next steps to be concrete engineering output rather than architecture:

* a formal **language reference** for the four profiles (including grammar fragments and static typing rules),
* a detailed **IR specification** (RIR/PIR/EIR/SIR schemas),
* and a first-cut **kernel plan** (join/dedup/fixpoint primitives, circuit-eval primitives, and ELP candidate-check primitives).

[1]: https://www.depts.ttu.edu/cs/research/documents/32.pdf "Title"
[2]: https://thomas.gilray.org/pdf/datalog-gpu.pdf "Optimizing Datalog for the GPU"
[3]: https://dtai.cs.kuleuven.be/problog/tutorial/python/01-compile-once.html "Recipe: compile-once evaluate-many — ProbLog: Probabilistic Programming"
[4]: https://arxiv.org/abs/2308.15020?utm_source=chatgpt.com "Massively Parallel Continuous Local Search for Hybrid SAT Solving on GPUs"
[5]: https://docs.rapids.ai/api/cudf/latest/user_guide/groupby/?utm_source=chatgpt.com "GroupBy — cudf 25.12.00 documentation"
[6]: https://link.springer.com/article/10.1007/s10703-023-00432-z?utm_source=chatgpt.com "Certified SAT solving with GPU accelerated inprocessing"
[7]: https://arxiv.org/pdf/2501.13051?utm_source=chatgpt.com "arXiv:2501.13051v1 [cs.DB] 22 Jan 2025"
[8]: https://hpcrl.github.io/ICS2025-webpage/program/Proceedings_ICS25/ics25-71.pdf "Multi-Node Multi-GPU Datalog"
[9]: https://papers.nips.cc/paper/7632-deepproblog-neural-probabilistic-logic-programming?utm_source=chatgpt.com "DeepProbLog: Neural Probabilistic Logic Programming"
[10]: https://www.ijcai.org/proceedings/2020/0243.pdf?utm_source=chatgpt.com "NeurASP: Embracing Neural Networks into Answer Set ..."
[11]: https://arxiv.org/abs/2008.02018 "[2008.02018] eclingo: A solver for Epistemic Logic Programs"
[12]: https://www.ijcai.org/proceedings/2018/0237.pdf "Single-Shot Epistemic Logic Program Solving"
[13]: https://ojs.aaai.org/index.php/AAAI/article/view/7852?utm_source=chatgpt.com "An Algebraic Prolog for Reasoning about Possible Worlds"
[14]: https://ceur-ws.org/Vol-3281/paper7.pdf "Evaluating Epistemic Logic Programs via Answer Set Programming with Quantifiers"
