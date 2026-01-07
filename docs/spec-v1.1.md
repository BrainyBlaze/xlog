# XLOG Design Document v1.1

**GPU-Native Declarative Programming for Deterministic, Probabilistic, Epistemic, and Solving Workloads**

**Date:** January 7, 2026
**Status:** Revised design after review feedback (addressing identified gaps and risks)

---

## 0. Revision summary

This revision directly resolves the gaps and concerns you enumerated:

* **ELP complexity / “bounded” ambiguity:** introduces *precise*, enforceable feasibility bounds (k, candidate budget, structural decomposition triggers) and a deterministic tier-selection policy grounded in known complexity results. World-view existence is Σ_P^3-complete and is treated as such.
* **PIR → GPU circuit format:** defines a concrete **XLOG GPU Circuit Format (XGCF)** and an implementable pipeline that integrates a proven Decision-DNNF compiler (**D4**) plus optional certification. ([Departement Computerwetenschappen][1])
* **CPU compilation “hand-wave”:** clarified as “compile-once, evaluate-many” and engineered so *training/evidence updates do not trigger recompilation* (ProbLog’s standard operational pattern). ([Departement Computerwetenschappen][2])
* **Solver strategy uncertainty:** provides a concrete **solve policy** (LS vs Exact vs Portfolio) tied to whether the caller needs *witness*, *counterexample*, *UNSAT/optimality proof*, or *universal property* (needed for cautious reasoning in ELP).
* **Memory spill policy:** specifies an explicit **memory budget contract**, deterministic failure modes, and an opt-in out-of-core mode (without silently violating GPU-residency guarantees).
* **Neural predicate gradients:** specifies forward/backward math and a PyTorch integration strategy (custom autograd over XGCF).
* **ELP propagation specifics:** makes propagation operational: what is propagated, how it is computed, and how it is batched on GPU (Generate–Propagate–Test per Fandinno & Lillo).
* **Single-shot vs multi-shot decision:** defines an explicit planner heuristic and when each dominates; single-shot supported based on known “single-shot ELP solving” literature.
* **“No full Prolog” vs Prolog-like syntax:** clarified: Prolog-like *notation* is allowed; Prolog’s *execution model* (general unification/backtracking/cut) is not.
* **Probabilistic–epistemic integration:** replaces ad-hoc thresholding as the default with a **principled semantics option**: probability over epistemic queries via Monte Carlo over probabilistic facts + batched ELP evaluation, consistent with quantitative directions in modern ELP complexity work. ([Departement Computerwetenschappen][1])
* **Portability:** CUDA-first (due to cuDF/libcudf), but kernel interfaces are explicitly abstracted to allow future HIP/SYCL backends without rewriting the compiler.

---

## 1. Overview

XLOG is a GPU-native declarative programming platform unifying four reasoning paradigms:

| Subsystem      | Purpose                                                                 | Primary Inspirations                                                                         |
| -------------- | ----------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **xlog-logic** | Deterministic Datalog-style recursion and stratified negation on GPU    | GPUlog (HISA), VFLog (columnar GPU Datalog) ([arXiv][3])                                     |
| **xlog-prob**  | Probabilistic + differentiable reasoning (ProbLog/DeepProbLog-like)     | ProbLog knowledge compilation (d-DNNF/SDD/BDD), WMC ([Departement Computerwetenschappen][1]) |
| **xlog-elp**   | Epistemic Logic Programming (world views) with robust default semantics | eclingo (G91 guess-check), FAEEL founded world views, epistemic splitting, propagation       |
| **xlog-solve** | SAT/MaxSAT/ASP/ELP solving services for GPU                             | ParaFROST certified GPU inprocessing; FastFourierSAT GPU CLS                                 |

---

## 2. Core architectural constraints and decisions

### 2.1 GPU-residency contract (G1) — now formalized

**Default execution contract (Strict GPU-Resident Mode):**

* All runtime semantic state (relations, deltas, circuit values, solver state, candidate batches) must fit in **device memory** and remain device-resident during evaluation.
* If the plan cannot be executed within the configured GPU memory budget, XLOG returns a **deterministic `RESOURCE_EXHAUSTED`** error with diagnostics and remediation guidance.

**Optional execution contract (Out-of-Core Mode, explicit opt-in):**

* XLOG may spill **immutable** intermediates to host-pinned memory or UVM-managed pages; evaluation remains on GPU but may incur page migration.
* Out-of-core mode is explicitly surfaced in the API (`--allow-ooc`) and is never used silently.

This resolves the “UVM as last resort” ambiguity by making spill behavior a deliberate, user-visible choice.

### 2.2 Hybrid storage strategy (kept; now made operational)

* **Baseline:** cuDF/libcudf relations for standard joins/groupbys early in the roadmap.
* **Hot-path acceleration:** GPUlog-style HISA indexes and VFLog-style columnar layouts are added per predicate based on observed recursion and join patterns. GPUlog explicitly motivates HISA for efficient range queries, lock-free dedup, and parallel iteration. ([arXiv][3])
* XLOG selects a physical layout per predicate:

  * `CUDF_TABLE` (default, interoperability),
  * `HISA_INDEXED` (recursion-heavy joins/dedup),
  * `VFLOG_COLUMNAR` (bandwidth-dominated columnar workloads). ([arXiv][4])

### 2.3 Layered IR stack (kept; now specified)

* **RIR:** Relational IR (joins, anti-joins, groupby aggregates, union/dedup, delta iteration).
* **PIR:** Provenance IR (weighted Boolean formula / circuit terms; differentiable semiring ops).
* **EIR:** Epistemic IR (subjective literals, modal constraints, world-view objectives).
* **SIR:** Solve IR (CNF + cardinality + weights; proof/cert hooks).

Each IR node includes:

* estimated cardinality ranges,
* memory peak estimates,
* skew hints (for join partitioning),
* incremental update semantics (delta/full),
* tier requirements (exact vs approximate).

---

## 3. Clarification: Prolog-like syntax vs Datalog execution model

**XLOG supports Prolog-like surface notation** (`:-`, variables, predicates) for familiarity.

**XLOG does not implement full Prolog semantics** (unrestricted unification over nested terms, depth-first backtracking search, cut, etc.). Execution is *bottom-up*, relational, set-based (Datalog-style) with optional constrained solving through `xlog-solve`.

This addresses the “NG1 contradiction”: syntax is a UI choice; execution model is explicitly relational/GPU-parallel.

---

## 4. Memory management and spill policy (new detailed design)

### 4.1 Memory budget model

For each run, XLOG defines:

* **Device budget** `B_dev` (default: `0.80 * free_device_mem`).
* **Operator budget** per SCC / per query stage (planner-controlled).
* **Peak estimation** using:

  * input sizes,
  * join selectivity stats (sampled on GPU),
  * worst-case caps for recursion.

### 4.2 Deterministic behavior under memory pressure

If estimated peak > `B_dev` in strict mode:

* compiler returns `PLAN_REJECTED_MEMORY`,
* includes:

  * which SCC/rule causes the peak,
  * estimated intermediate sizes,
  * suggested mitigations:

    * enable multi-GPU partitioning,
    * strengthen domain guards,
    * enable out-of-core mode,
    * apply projection pushdown / aggregate early.

If runtime peak unexpectedly exceeds due to skew:

* runtime triggers **deterministic abort** at a checkpoint boundary (never half-corrupt results),
* dumps the skew signature and hot-key histogram.

### 4.3 Out-of-core mode (explicit)

If `--allow-ooc`:

* XLOG may spill *materialized immutable intermediates* (e.g., final `R` of an SCC, or a frozen circuit DAG).
* XLOG uses explicit prefetching and eviction policies (LRU by buffer class), but does not promise performance.

---

## 5. xlog-logic: deterministic GPU logic engine (refined)

### 5.1 Evaluation model

* Semi-naive evaluation per SCC.
* Delta relations maintained explicitly.
* Dedup/merge is treated as a first-class kernel (because GPUlog shows it is central and requires GPU-specific design). ([arXiv][3])

### 5.2 Multi-GPU recursion and skew handling (expanded)

XLOG adopts partitioning strategies aligned with multi-node/multi-GPU Datalog work, including radix-hash partitioning and GPU-aware all-to-all for iterative computation. ([ACM Digital Library][5])

**Skew plan (explicit):**

* detect hot keys (top-K frequency) on GPU during profiling,
* replicate “hot partitions” across GPUs (controlled replication factor),
* use skew-aware repartitioning between iterations if delta skew changes.

This is a necessary engineering step beyond “hash partitioning works.”

---

## 6. xlog-prob: probabilistic + differentiable reasoning (major revision)

The most important clarification: **PIR → GPU circuit is now fully specified** and the compilation pipeline is implementable with existing compilers.

### 6.1 Compilation strategy: compile-once, evaluate-many (no hand-wave)

ProbLog explicitly promotes *compile once* then evaluate repeatedly under varying evidence/queries, and supports multiple “evaluatables” including d-DNNF/SDD/BDD. ([Departement Computerwetenschappen][2])

**XLOG adopts the same operational pattern:**

* Rule structure is compiled once into a circuit form.
* Training iterations only update leaf weights (neural outputs / fact probabilities) and evidence indicators.
* Therefore CPU compilation is not on the hot path for typical DeepProbLog-like workloads.

### 6.2 CNF + Decision-DNNF backbone (D4 integration)

**Why:** Weighted Model Counting (WMC) and knowledge compilation are established reductions for probabilistic logic inference. ([starai.cs.ucla.edu][6])
ProbLog’s tutorial explains that knowledge compilation techniques like d-DNNF/SDD/OBDD address disjoint sum and enable efficient probability evaluation. ([Departement Computerwetenschappen][1])

**Compiler pipeline (implementable):**

1. **Grounding & query slicing (GPU):** `xlog-logic` produces the relevant grounded subset for the query/evidence.
2. **Weighted CNF emission (CPU or GPU):** emit a WCNF (or CNF + weight map).
3. **Optional GPU inprocessing (GPU):** use ParaFROST-style simplification kernels to reduce CNF size before compilation; ParaFROST is explicitly designed for GPU-accelerated inprocessing and certified simplifications.
4. **Decision-DNNF compilation (CPU):** compile CNF to Decision-DNNF using **D4**, a state-of-the-art top-down Decision-DNNF compiler.
5. **Optional certification (debug/assurance mode):**

   * Use “certifying decision-DNNF compilers” approaches where feasible, or store checkable proof artifacts.
6. **Lower to XGCF (GPU format):** convert Decision-DNNF to a GPU-friendly linearized circuit representation.
7. **Evaluate and differentiate on GPU** (see below).

### 6.3 XLOG GPU Circuit Format (XGCF) — new specification

**Design goal:** GPU-evaluable, batchable, and differentiable representation.

**Representation (SoA arrays on GPU):**

* `node_type[i]` ∈ {CONST0, CONST1, LIT, AND, OR, DECISION}
* `a[i], b[i]` child indices (meaning depends on node_type)
* `var[i]` for DECISION nodes (variable id)
* `lit[i]` for LIT nodes (literal id; mapped to leaf weight)
* `level_offsets[]` for topological “levels” enabling level-by-level evaluation
* `value[i]` in log-space (`float32` or `float64` configurable)
* `adj[i]` reverse-mode adjoint buffer (same dtype as `value`)

**Batching:**

* Circuits for multiple queries can be concatenated with `circuit_offsets`.
* Neural batch dimension is expressed as multiple leaf-weight tables; evaluation uses fused kernels for leaf-gather.

### 6.4 Exactness tiers (now operational)

* **P1 (Exact):** the compiled circuit guarantees determinism/decomposability conditions needed for correct sum/product evaluation (Decision-DNNF/d-DNNF gives this by construction for WMC). ([Departement Computerwetenschappen][1])
* **P2 (Exact restricted):** acyclic / bounded-treewidth fragments directly compiled without full CNF→DNNF (future optimization).
* **P3 (Approximate):** Monte Carlo / variational when compilation is infeasible.

### 6.5 Neural predicate gradients (new: explicit math + integration plan)

XLOG evaluates probabilistic circuits in **log-space** for stability:

* **LIT node:** `v = log(w_lit)` with evidence masking.
* **AND node:** `v = v_a + v_b + …`
* **OR node:** `v = logsumexp(v_a, v_b, …)`
* **DECISION node:** `v = logsumexp(log(p(var=1)) + v_true, log(p(var=0)) + v_false)`

**Reverse-mode autodiff on XGCF:**

* Initialize `adj[root] = dL/dv_root` from the loss.
* For AND: distribute `adj` unchanged to children.
* For OR:
  `adj[child_j] += adj[parent] * exp(v_child_j - v_parent)`
  (i.e., softmax weights derived from logsumexp; numerically stable).
* For LIT: accumulate `dL/dlog(w_lit)` and convert to `dL/dw_lit` as needed.

**PyTorch integration (implementable):**

* Expose `XLOGCircuitFunction` as a custom autograd Function:

  * `forward`: uploads leaf weights (from neural outputs) and runs circuit forward kernel(s), returning query log-probabilities.
  * `backward`: runs reverse kernels on GPU to produce gradients w.r.t. leaf weights; PyTorch then backpropagates into neural nets normally.

This is the missing “how gradients flow” detail.

---

## 7. xlog-solve: solver services (major clarification)

### 7.1 Why two engines (LS + Exact) and how XLOG chooses

Recent GPU SAT work argues that CDCL is sequential and hard to GPU-parallelize; FastFourierSAT instead uses massively parallel continuous local search (CLS) and shows strong GPU acceleration (notably >100× faster gradient computation than CPU prototypes).
ParaFROST demonstrates GPU-accelerated inprocessing with certification/proof generation to validate simplifications.

**Therefore XLOG includes:**

* **Solve-LS:** fast GPU CLS/local search for producing satisfying assignments or high-quality solutions quickly.
* **Solve-Exact:** exact solving mode that can produce UNSAT/optimality evidence where required; includes certified simplification hooks.

### 7.2 Solve policy (now explicit)

`xlog-solve` selects an engine per request based on a **Solve Contract**:

| Request type                | Needs witness? | Needs proof (UNSAT/optimal)? | Needs universal property (“for all models”)? | Default engine                                             |
| --------------------------- | -------------: | ---------------------------: | -------------------------------------------: | ---------------------------------------------------------- |
| SAT witness                 |            Yes |                           No |                                           No | **Solve-LS**, verify                                       |
| MaxSAT approximate          |            Yes |                           No |                                           No | **Solve-LS**                                               |
| UNSAT proof required        |             No |                          Yes |                                           No | **Solve-Exact**                                            |
| Optimality proof required   |            Yes |                          Yes |                                           No | **Solve-Exact**                                            |
| Cautious consequence checks |      Sometimes |                        Often |                                      **Yes** | **Hybrid:** LS for counterexamples + Exact for final proof |

This is critical for ELP: eclingo’s checker relies on cautious/brave consequence reasoning (universal/existential checks).
XLOG uses LS to find counterexamples quickly, but exact mode (or explicit `UNKNOWN`) is required to conclude universal facts.

### 7.3 Verification (robustness guarantee)

Regardless of engine:

* any candidate model is verified on GPU by constraint kernels before being returned.
* if verification fails, the candidate is discarded and the solver continues.

---

## 8. xlog-elp: epistemic logic programming (major expansion)

### 8.1 Complexity and tiering (now precise)

World view existence is known to be **Σ_P^3-complete**, and ELP reasoning is generally one level higher in the polynomial hierarchy than ASP tasks.

XLOG therefore defines **hard feasibility bounds** for exact tiers:

#### Definitions

* Let **k** = number of distinct epistemic atoms / subjective literals after normalization and splitting (per component).
* Let **C_max** = maximum candidate guesses allowed for exact enumeration in a component.
* Let **T_check** = estimated mean cost of one candidate “tester” check (measured online).

#### Default thresholds (configurable; chosen for implementability)

* **E1 (Exact, structural):**

  * epistemically stratified or successfully split into components where *each component* has `k ≤ 24` and at least one component is purely objective/stratified (so `xlog-logic` handles it).
  * splitting uses Cabalar et al.’s epistemic splitting notion; for FAEEL, splitting property holds.
* **E2 (Exact, bounded enumeration):**

  * for each component: `k ≤ 16` **or** predicted `|candidates_after_propagation| ≤ 50,000`.
  * if either bound is exceeded, planner escalates to E3.
* **E3 (Approximate):**

  * sampling-based or bounded-search world-view approximation with explicit `UNKNOWN/approx` labeling.

These numbers are defaults, not claims of universality. They exist to make the tier boundaries machine-checkable and operational.

### 8.2 Default semantics choice (kept; strengthened)

**Default:** founded world views via FAEEL-style semantics to avoid self-supported world views that can arise under original G91 semantics.

**Compatibility mode:** G91 semantics (eclingo-style) for interoperability and regression validation. eclingo explicitly targets G91 and uses guess-and-check with cautious/brave consequences.

### 8.3 Core algorithm: Generate–Propagate–Test (now concrete)

XLOG’s ELP engine is built to match the best-known practical approach: generate-and-test with propagation, which can exponentially reduce tested candidates with only linear overhead (as reported by Fandinno & Lillo).

#### Step A — Normalize and split

1. Normalize subjective literals (canonicalize `K l`, `M l`, epistemic negations).
2. Apply epistemic splitting:

   * find bottom component `B` and top component `T` where `T` refers to `B` only via subjective literals.
3. Evaluate `B` once when possible (stratified → `xlog-logic`; otherwise `xlog-solve`).

#### Step B — Generator (candidates)

* Represent guesses as bitvectors over epistemic atoms (GPU-friendly).
* Initial candidate set:

  * E1/E2: full candidate space bounded by k, then pruned by propagation.
  * E3: stochastic generator (importance sampling over guesses).

#### Step C — Propagation (what is propagated, explicitly)

Propagation operates at two levels:

**(1) Structural propagation (zero solver calls; always on)**

* Modal consistency constraints:

  * `K l ⇒ M l`
  * `K l ⇒ ¬K ¬l` (where consistent)
  * `¬M l ⇒ ¬K l`
* Dependency monotonicity checks:

  * if an epistemic atom does not occur in the reduct-relevant portion for the query, it can be dropped (dead epistemic literal elimination).

**(2) Semantic propagation (batched solver calls; GPU parallel)**
For a *partial* guess `g` and an undecided epistemic atom `e` about objective literal `l`:

* **Brave test:** is there a stable model satisfying `l` under `g`?
* **Counterexample test (for cautious):** is there a stable model satisfying `¬l` under `g`?

These are SAT-style checks; XLOG batches them:

* many partial guesses × many pending tests → one GPU batched solve request.
* Solve-LS is used first to quickly find counterexamples; Solve-Exact is invoked when E1/E2 exactness requires concluding “no counterexample exists.”

This mirrors the operational meaning of brave/cautious reasoning used in eclingo-style checking while making the propagation mechanics explicit.

#### Step D — Tester

A candidate is accepted as a world view iff:

* it is internally consistent,
* it matches the brave/cautious consequences required by the chosen semantics (FAEEL default or G91 compatibility),
* all world-view constraints are satisfied.

### 8.4 Single-shot vs multi-shot solving decision (now explicit)

Single-shot ELP solving exists and is motivated by Σ_P^3 complexity; it encodes epistemic reasoning into a single solve instance (at the cost of a potentially large encoding).

**Planner decision rule (default):**

* Use **multi-shot Generate–Propagate–Test** when:

  * repeated evaluations are expected (training loops, multiple queries),
  * splitting succeeds and reduces k per component,
  * propagation is effective (candidate budget shrinks quickly).
* Use **single-shot translation** when:

  * splitting fails,
  * `k > 16` and the estimated single-shot encoding size is ≤ 3× the ground program size,
  * the request is one-off and exactness is required.

This makes the choice predictable and tied to measurable quantities.

---

## 9. Probabilistic–epistemic integration (now principled)

Your review correctly called thresholding/sampling “workaround-like.” XLOG now defines explicit integration modes so users know exactly what they are getting.

### 9.1 Mode PE-MC (default integration; principled and implementable)

**Semantics (informal but precise):**

* probabilistic facts define a distribution over objective programs (as in ProbLog’s distribution semantics).
* for each sampled probabilistic world, evaluate the ELP and compute whether an epistemic query holds.
* estimate `P(query)` as the Monte Carlo average.

This is principled (probability over worlds), fully GPU-parallelizable (massively batched samples), and aligns with modern ELP research directions that discuss quantitative reasoning on probabilities of epistemic literals. ([Departement Computerwetenschappen][1])

**Implementation:**

* sample N worlds on GPU (bitpacked assignments of probabilistic facts),
* batch N ELP evaluations via `xlog-elp` + `xlog-solve`,
* aggregate results into probability estimates + confidence intervals.

### 9.2 Mode PE-Exact (restricted; roadmap)

For small programs where the epistemic layer can be compiled into a circuit-like form, XLOG can attempt exact integration via weighted counting over epistemic constraints. This is a research track, not a v1 guarantee.

---

## 10. Portability stance (addressing vendor lock concern)

XLOG is CUDA-first because:

* cuDF/libcudf is NVIDIA RAPIDS and CUDA-based.

However, XLOG now explicitly defines a **Kernel Provider Interface**:

* logical operator API: join, dedup, groupby, circuit eval kernels, SAT primitives
* backend implementations:

  * `cuda_provider` (v1)
  * `hip_provider` / `sycl_provider` (roadmap)

The compiler and IR do not embed CUDA-specific assumptions beyond the provider boundary.

---

## 11. Updated phased roadmap (reordered per recommendation)

Your recommendation to prototype solver early is accepted. `xlog-solve` is shared infrastructure and should not be deferred.

### Phase 0 — Foundations (unchanged)

* IRs + profiler + GPU memory budget enforcement (strict mode first)

### Phase 1 — xlog-logic MVP

* semi-naive recursion, dedup kernels, stratified negation

### Phase 2 — xlog-solve MVP (moved earlier)

* Solve-LS (FastFourierSAT-style CLS) baseline and GPU verification
* Solve-Exact skeleton + certified inprocessing hooks (ParaFROST-inspired)

### Phase 3 — xlog-prob MVP

* PIR → WCNF
* D4 Decision-DNNF compilation + XGCF lowering ([Departement Computerwetenschappen][1])
* GPU circuit eval + autograd

### Phase 4 — xlog-elp MVP

* G91 compatibility mode first (eclingo-style guess-check semantics for regression)
* Generate–Propagate–Test with semantic propagation
* Splitting and stratified fast path

### Phase 5 — Robust default semantics and scaling

* FAEEL default semantics
* multi-GPU scaling for recursion and batched ELP checks (mnmgDatalog-inspired partitioning strategies) ([ACM Digital Library][5])

---

## 12. “Most robust, novel, efficient, implementable” — why this version qualifies

### Robust

* Explicit *exactness* and *boundedness* contracts for ELP (no vague “escape hatches”).
* Proof/certification philosophy for “exact” claims in solving and compilation.
* Deterministic memory failure modes and explicit out-of-core opt-in.

### Novel

* Unified GPU-resident platform spanning:

  * recursive Datalog engine (GPUlog/VFLog class), ([arXiv][3])
  * probabilistic inference via Decision-DNNF lowering into a GPU circuit format (XGCF), ([Departement Computerwetenschappen][1])
  * epistemic world view computation using propagation and splitting, engineered for GPU batching.

### Efficient

* Recursion and dedup engineered around known GPU Datalog bottlenecks and data structures (HISA/columnar). ([arXiv][3])
* Solver layer uses GPU-friendly CLS where it fits and exact mode where universal/proof requirements demand it.
* ELP propagation reduces candidate explosion substantially in practice (per reported results).

### Implementable

* Every “hard” component has a concrete, staged plan:

  * circuits via D4 + XGCF rather than speculative GPU compilation,
  * ELP via established guess-check pattern + modern propagation + splitting,
  * solver via published GPU approaches and an explicit decision policy.

---

## 13. Next concrete deliverables (aligned with your verdict)

To convert this design into build-ready specifications, the next three documents should be produced:

1. **XLOG Language Reference (Profiles: logic/prob/elp/solve)**

   * grammar, typing/modes, domain safety rules, semantics tier annotations.
2. **IR Specifications (RIR/PIR/EIR/SIR) + Cost Model**

   * schemas, lowering rules, memory-estimation and tier-selection logic.
3. **Kernel & Runtime Plan**

   * required kernels (join/dedup/agg, circuit eval, solve primitives),
   * memory budget enforcement implementation,
   * batching interfaces for ELP propagation checks and probabilistic Monte Carlo.


[1]: https://dtai.cs.kuleuven.be/problog/tutorial/advanced/00_inference.html?utm_source=chatgpt.com "Inference in ProbLog - Probabilistic Programming - DTAI"
[2]: https://dtai.cs.kuleuven.be/problog/tutorial/python/01-compile-once.html?utm_source=chatgpt.com "Recipe: compile-once evaluate-many — ProbLog - DTAI"
[3]: https://arxiv.org/html/2311.02206v5?utm_source=chatgpt.com "Optimizing Datalog for the GPU"
[4]: https://arxiv.org/abs/2501.13051?utm_source=chatgpt.com "Column-Oriented Datalog on the GPU"
[5]: https://dl.acm.org/doi/10.1145/3721145.3730431?utm_source=chatgpt.com "Multi-Node Multi-GPU Datalog | Proceedings of the 39th ..."
[6]: https://starai.cs.ucla.edu/papers/FierensTPLP15.pdf?utm_source=chatgpt.com "Inference and Learning in Probabilistic Logic Programs ..."
