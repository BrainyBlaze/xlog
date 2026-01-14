Below is an **Implementation-Planning Addendum** that resolves your remaining minor observations, followed by the next three build-ready specifications you recommended: **Language Reference**, **IR Specification**, and **Kernel & Runtime Plan**.

Where I reference external systems/papers, I now provide concrete sources (eclingo, FAEEL, generate-and-test with propagation, single-shot solving, ProbLog compile-once, D4, ParaFROST, FastFourierSAT, GPUlog, VFLog, mnmgDatalog). ([arXiv][1])

---

# XLOG Implementation-Planning Addendum v1.2

> **Implementation status (2026-01-14):** Phase 3 (`xlog-logic`) and Phase 4 (`xlog-prob` exact `exact_ddnnf` + approximate `mc`, plus Python `xlog_gpu`) are implemented on `main`. See `docs/VALIDATION_REPORT.md` and `docs/architecture/xlog-prob.md`.

## A. Addressing remaining minor observations

### A1) D4 as an external CPU dependency (and dynamic programs)

**What remains true:** the v1.1/v1.2 probabilistic exact path uses a **CPU knowledge compilation step** (CNF → Decision-DNNF), with **D4** as the initial backend. ([IJCAI][2])
**Why it is acceptable:** ProbLog’s normal workflow is “compile once, evaluate many,” and the system supports multiple evaluatables (d-DNNF/SDD/BDD). ([Departement Computerwetenschappen][3])
XLOG follows the same amortization strategy: **training and evidence updates do not recompile circuits**—only leaf weights and evidence masks change.

#### v1.2 design refinement: KC backend abstraction + caching + incremental structure reuse

**Introduce `KCBackend` plugin interface:**

* `compile(CNF) -> CircuitIR`
* `supports_certificates() -> bool`
* `compile_with_certificate(CNF) -> (CircuitIR, Cert)`
* `normalize(CNF) -> CNF` (optional canonicalization for hashing)

**Backends (planned):**

* `KCBackendD4` (default exact compiler) ([IJCAI][2])
* future `KCBackendSDD` and `KCBackendBDD` (for different tradeoffs), aligned with ProbLog’s evaluatable concept. ([Departement Computerwetenschappen][3])

**Cache design (critical for dynamic programs):**

* Cache key: `hash(CNF_normal_form, var_order_hint, compilation_options)`.
* Cache value: compiled XGCF circuit + metadata (var map, literal map, node levels).
* Storage tiers:

  * GPU memory (hot),
  * host pinned memory (warm),
  * disk (cold, optional).

**Incremental structure reuse for “dynamic programs”:**
Dynamic structure changes generally arise from:

* changes in *rule structure* (program text changes), or
* changes in *domain size* causing different grounding topology.

XLOG’s mitigation is **modular compilation**:

* partition provenance/CNF into **components** using D4’s decomposition-friendly perspective (it is explicitly based on decomposition and component caching). ([IJCAI][2])
* compile and cache components independently, then assemble a top-level circuit that references component outputs.

Practically: small edits often affect only a subset of components; most cached subcircuits remain reusable.

**GPU-side preprocessing to shrink CNF before D4:**

* Use ParaFROST-style GPU inprocessing/simplification as an optional step (safe even if compilation is CPU) to reduce CNF size and compilation time. ([gears.win.tue.nl][4])

---

### A2) E2 “≤ 50K candidates after propagation” needs empirical validation

Agreed. v1.2 makes the E2 boundary **adaptive and measurable** rather than static.

#### v1.2: Adaptive candidate-budget controller

Define:

* `k`: epistemic atoms in a component
* `N0 = 2^k`: naive candidate space
* `ρ_struct`: reduction factor from structural propagation (measured)
* `ρ_sem`: reduction factor from semantic propagation (estimated online from early batches)

**Estimator workflow:**

1. Run structural propagation on the full candidate bitset representation (GPU bitwise filters).
2. If `N_struct ≤ N_hard_cap`, continue.
3. Otherwise, run *pilot* semantic propagation on a small random sample of candidates (e.g., a few thousand) to estimate `ρ_sem`.
4. Predict `N_final ≈ N_struct * ρ_sem`.
5. Tier decision:

   * if `N_final ≤ C_max` → E2 exact enumeration,
   * else → E3 approximate (or single-shot encoding if predicted compact; see A4).

This gives you:

* a **machine-checkable** tier boundary,
* an empirical feedback loop,
* a clear place to tune thresholds based on real benchmarks (which is what you want).

---

### A3) PE-MC confidence intervals (reproducible specification)

In PE-MC we estimate probabilities of a Boolean query `Q` (“epistemic query holds”) under Monte Carlo samples.

Let:

* `n` = number of sampled probabilistic worlds,
* `s` = number of samples where `Q` holds,
* `p̂ = s/n`.

#### Default interval: Wilson score interval (95% by default)

Wilson is stable even for small n and extreme probabilities.

For confidence level `1−α`, let `z = Φ^{-1}(1−α/2)`.

Compute:

* `den = 1 + z^2/n`
* `center = (p̂ + z^2/(2n)) / den`
* `radius = (z * sqrt( (p̂(1−p̂) + z^2/(4n)) / n )) / den`

Return `[center − radius, center + radius]`.

#### Optional exact interval: Clopper–Pearson

Expose as `#pragma pe_ci=clopper_pearson` for users who want exact binomial intervals (more conservative).

#### Non-Boolean epistemic metrics

If the user requests an expectation of a bounded scalar `X ∈ [0,1]` (e.g., plausibility score), default to **empirical Bernstein** bounds; optional bootstrap for arbitrary bounded metrics with deterministic seeding.

**Reproducibility contract:**

* deterministic PRNG seeds per run,
* record `n`, `α`, method, and seed in the output metadata.

---

### A4) FAEEL implementation details (no longer deferred)

This is now the key: **FAEEL can be implemented as “founded G91 world views.”**

The FAEEL paper explicitly states that its world views “precisely coincide with the set of founded G91 world views” and develops foundedness via an unfounded-set definition. 

#### v1.2 Implementation strategy for `xlog-elp` default semantics (FAEEL)

**Step 1 — Compute candidate G91 world views (reuse eclingo architecture):**

* eclingo uses a guess-and-check strategy: guess truth values for subjective literals, then check using cautious and brave consequences. ([arXiv][1])
  XLOG already implements this as Generate–Propagate–Test.

**Step 2 — Foundedness filter**
Given a candidate world view `W` (set of stable models), test whether `W` is **founded**.

The paper defines:

* an **unfounded set** `S` as a set of pairs ⟨X, I⟩ (X atoms, I interpretation) satisfying a “no justifying rule” condition, including a specific epistemic condition preventing circular justification through positive `K a` references. 
* a world view `W` is *unfounded* if such an `S` exists meeting certain criteria; otherwise it is founded. 

**Engineering this in XLOG (implementable approach):**

* Reduce foundedness checking to a **bounded existential search** in `xlog-solve`:

  * variables encode membership of atoms in each X component for each model I,
  * constraints encode the four “no justifying rule” conditions (including the epistemic condition 4 involving positive subjective literals). 
* If `xlog-solve` finds an `S`, reject `W` as unfounded.
* Otherwise accept `W` as founded (thus FAEEL-valid, per the paper’s equivalence result). 

**Why this is practical:** you avoid building a separate bespoke “FAEEL solver” from scratch; you build:

* a G91 engine (already required for compatibility),
* plus a foundedness checker (a solver-backed filter).

**Splitting support:** FAEEL also satisfies epistemic splitting (proved by Fandinno), so splitting-based decomposition remains sound for the default semantics. ([arXiv][5])

---

### A5) Single-shot ELP translation (selp) as a first-class planner option

Single-shot solving is not hypothetical: there is an established translation from ELPs to non-ground ASP with bounded arity, enabling a single ASP solver call (“selp”). ([arXiv][6])
The IJCAI single-shot paper also ties the complexity to Σ_3^P for bounded-arity non-ground ASP evaluation, aligning with ELP complexity. ([IJCAI][7])

XLOG v1.2 keeps the prior rule (“multi-shot by default; single-shot if favorable”) but now justifies it with this concrete translation literature and uses it as an optimization lever. ([IJCAI][8])

---

# Deliverable 1: XLOG Language Reference v0.1

**Profiles:** `xlog-logic`, `xlog-prob`, `xlog-elp`, `xlog-solve`

## 1) Design principles

* **Declarative surface, relational execution.** Prolog-like syntax, Datalog-style bottom-up execution (no general backtracking).
* **Finite grounding by construction.** Variables must be domain-safe.
* **Explicit semantics and tier selection.** Especially for probabilistic and epistemic features.
* **GPU-first ergonomics.** Data model is columnar relations; types/domains are explicit.

## 2) Lexical conventions

* **Predicate names / constants:** `lower_snake_case`
* **Variables:** `UpperCamel` or leading uppercase `X`, `ImgId`
* **Modal operators (ELP):** `K`, `M`
* **Negation as failure:** `not`
* **Classical negation (optional):** `~p(a)` (profile-dependent)

## 3) Types and domains

### 3.1 Built-in scalar types

* `u32`, `u64`, `i32`, `i64`
* `f32`, `f64`
* `bool`
* `symbol` (dictionary-encoded)
* `enum{...}` (finite)

### 3.2 Domain declarations (mandatory for safety)

Examples:

```prolog
domain node_id : u32 in [0..1000000).
domain digit   : enum{0,1,2,3,4,5,6,7,8,9}.
```

## 4) Predicate declarations

```prolog
pred edge(node_id, node_id).
pred reach(node_id, node_id) @derived.
pred out_degree(node_id, u32) @derived @key(1).
```

Optional annotations:

* `@key(cols...)`: indicates key columns for dedup/indexing
* `@layout(cudf|hisa|vflog)`: hint physical layout
* `@materialize(always|auto|never)`

## 5) Facts and rules (xlog-logic)

### 5.1 Facts

```prolog
edge(1,2).
edge(2,3).
```

### 5.2 Rules

```prolog
reach(X,Y) :- edge(X,Y).
reach(X,Z) :- reach(X,Y), edge(Y,Z).
```

### 5.3 Constraints

```prolog
:- reach(X,X).
```

### 5.4 Stratified negation (only if stratifiable)

```prolog
isolated(X) :- node(X), not edge(X,_), not edge(_,X).
```

## 6) Aggregates (xlog-logic)

Canonical aggregate syntax (GPU-friendly):

```prolog
out_degree(X, count(Y)) :- edge(X,Y).
max_w(X, max(W)) :- weight(X,W).
```

Notes:

* Aggregates must be **stratified** (no recursion through aggregates in v0.1).
* Supported in v0.1: `count`, `sum`, `min`, `max`, `logsumexp`.

## 7) Probabilistic constructs (xlog-prob)

### 7.1 Probabilistic facts

```prolog
0.7::rain().
0.2::edge(1,2).
```

### 7.2 Annotated disjunction

```prolog
0.6::coin(heads); 0.4::coin(tails).
```

### 7.3 Evidence and queries

```prolog
evidence(rain(), true).
query(reach(1,3)).
```

### 7.4 Compilation/evaluation directives

```prolog
#pragma prob_engine = exact_ddnnf   // default exact
#pragma prob_engine = mc            // sampling
#pragma prob_cache = on
```

ProbLog’s compile-once pattern motivates these directives. ([Departement Computerwetenschappen][3])

## 8) Neural predicates (xlog-prob)

Neural declaration:

```prolog
nn digit_classifier(img_id) -> digit @backend(torch) @model("mnist_cnn").
```

Usage pattern:

* Neural outputs materialize into a **GPU fact table**:

  * `nn_digit(img_id, digit, prob)` (implicit)
* The compiler rewrites:

  * `digit(Img, D)` into `nn_digit(Img, D, P)` leaf weights for `xlog-prob`.

DeepProbLog motivates the neural predicate concept and end-to-end learning objective. ([arXiv][9])

## 9) Epistemic constructs (xlog-elp)

### 9.1 Syntax

Subjective literals:

* `K l` — l is true in **all** answer sets in the world view
* `M l` — l is true in **some** answer set in the world view

Example:

```prolog
interview(X) :- not K eligible(X), not K ~eligible(X).
```

### 9.2 Semantics directives

```prolog
#pragma epistemic_semantics = faeel   // default
#pragma epistemic_semantics = g91     // compatibility mode
```

FAEEL is designed to satisfy foundedness and epistemic splitting. 
G91 is the semantics implemented by eclingo via guess-check with cautious/brave checks. ([arXiv][1])

### 9.3 World-view queries

```prolog
wv_query( K goal(a) ).
wv_query( M goal(a) ).
```

## 10) Solve blocks (xlog-solve)

A Solve block expresses SAT/MaxSAT/ASP-like constraints and optimization.

```prolog
solve @need(witness) {
  1 { pick(X) : item(X) } 1.
  :- pick(X), pick(Y), conflict(X,Y).
  #maximize { W,X : pick(X), weight(X,W) }.
}
```

### 10.1 Solve contracts

* `@need(witness)`
* `@need(unsat_proof)`
* `@need(optimality_proof)`
* `@need(universal_property)` (for cautious consequences)

Solver policy ties directly to these needs. FastFourierSAT motivates GPU CLS for witnesses; ParaFROST motivates certified inprocessing for proof-oriented exactness. ([arXiv][10])

---

# Deliverable 2: IR Specifications v0.1

**RIR / PIR / EIR / SIR**

## 1) Compilation pipeline (end-to-end)

1. **Parse + type/domain check** (reject unsafe grounding).
2. **Profile selection** (`logic`, `prob`, `elp`, `solve`).
3. **Dependency analysis**

   * stratification (negation/aggregates)
   * SCC decomposition
4. **Lower to RIR** for deterministic parts.
5. **If prob:** build PIR → emit CNF → KC compile → XGCF
6. **If elp:** build EIR:

   * extract epistemic atoms
   * attempt epistemic splitting
   * choose multi-shot vs single-shot (selp-like) plan ([IJCAI][7])
7. **If solve:** build SIR and dispatch.

## 2) RIR (Relational IR)

### 2.1 Core node types

* `RIR_Scan(RelId)`
* `RIR_Filter(RelId, PredicateExpr)`
* `RIR_Project(RelId, Columns)`
* `RIR_Join(left, right, left_keys, right_keys, join_type)`

  * `join_type ∈ {INNER, LEFT, SEMI, ANTI}`
* `RIR_GroupBy(RelId, key_cols, agg_specs)`
* `RIR_Union(RelId a, RelId b)`
* `RIR_Distinct(RelId)`
* `RIR_Diff(RelId a, RelId b)` (set difference)
* `RIR_Fixpoint(SCCId, rules[], termination=DELTA_EMPTY)`

### 2.2 Metadata (required on all nodes)

* `schema`
* `est_rows_range`
* `est_bytes_range`
* `skew_signature` (top-k keys, entropy)
* `determinism_mode`
* `layout_hint` (cudf/hisa/vflog)
* `delta_semantics` (if node supports incremental eval)

## 3) PIR (Provenance IR)

### 3.1 Purpose

Represent the derivation structure of probabilistic queries as a weighted Boolean form suitable for:

* knowledge compilation (Decision-DNNF) and WMC, or
* Monte Carlo approximation.

ProbLog’s inference documentation explicitly motivates compilation targets like d-DNNF/SDD/BDD to solve the disjoint-sum problem. ([Departement Computerwetenschappen][11])

### 3.2 Nodes

* `PIR_Lit(literal_id, weight_source)`
* `PIR_And(children[])`
* `PIR_Or(children[])`
* `PIR_Decision(var_id, child_false, child_true)` (matches Decision-DNNF)

### 3.3 Lowering paths

* `PIR -> CNF` (WCNF or CNF+weights)
* `CNF -> Decision-DNNF` via D4 backend ([IJCAI][2])
* `Decision-DNNF -> XGCF` (GPU circuit)

## 4) XGCF (XLOG GPU Circuit Format)

As specified in v1.1/v1.2:

* `node_type[i] ∈ {CONST0, CONST1, LIT, AND, OR, DECISION}`
* `a[i], b[i]` or `child_offsets` for variadic nodes
* `level_offsets[]` for topological levels
* `value[i]`, `adj[i]` buffers

## 5) EIR (Epistemic IR)

### 5.1 Core objects

* `EAtom(id, objective_literal, kind=K|M, polarity)`
* `SplitPlan(bottom_program, top_program, interface_atoms)` (optional)
* `GuessSpace(k, representation=bitset|sparse)`
* `PropagationRules(structural_rules[], semantic_rules[])`
* `TestTask(type=BRAVE|CAUTIOUS, program_slice_id, literal)`
* `WorldViewCandidate(guess, test_results, models_ref)`

### 5.2 Planner outputs

* Multi-shot plan:

  * generator program
  * propagation schedule
  * tester calls (batched)
* Single-shot plan:

  * ELP → ASP encoding (selp-like) ([arXiv][6])
  * single solve call
  * extract world view representation

## 6) SIR (Solve IR)

### 6.1 Core constraints

* `SIR_CNF(clauses_csr, num_vars)`
* `SIR_Cardinality(vars[], lo, hi)`
* `SIR_Weights(lit_weights)`
* `SIR_Objective(type=MAX|MIN, weighted_literals[])`
* `SIR_ProofPolicy(require_proof=true|false)`
* `SIR_Mode(mode=SAT|MAXSAT|ENUM|CHECK)`

### 6.2 Certificates

If proof required:

* store DRAT/FRAT-like proof artifacts when supported by exact backend; ParaFROST demonstrates proof generation and successful verification for GPU-accelerated SAT solving with inprocessing. ([cca.informatik.uni-freiburg.de][12])

---

# Deliverable 3: Kernel & Runtime Plan v0.1

**GPU operator library and execution scheduling**

## 1) Runtime layers

1. **Memory Manager**

   * RMM-backed pools
   * budget enforcement
   * deterministic OOM checkpoints
2. **Relation Store**

   * `CUDF_TABLE` baseline
   * optional `HISA_INDEXED` for recursion-heavy joins/dedup (GPUlog) ([arXiv][13])
   * optional `VFLOG_COLUMNAR` storage strategy (VFLog) ([arXiv][14])
3. **Kernel Scheduler**

   * stream-aware execution
   * batched operator fusion where possible
4. **Profiler**

   * row counts, skew, memory peaks
   * candidate pruning ratios for ELP

## 2) Relational kernels (xlog-logic)

### 2.1 Join kernels

* `hash_join_build(keys, payload)`
* `hash_join_probe(keys, payload)`
* `sort_merge_join(keys)` (for deterministic mode and range-friendly keys)
* `semi_join`, `anti_join`

**Baseline:** libcudf join operators; later replace hot paths with custom kernels based on profiling.

### 2.2 Dedup / distinct kernels

* `sort_unique` (deterministic)
* `lock_free_dedup` (fast set-content determinism)

GPUlog explicitly identifies lock-free deduplication as a core requirement of HISA. ([arXiv][15])

### 2.3 Groupby kernels

* `groupby_count`, `groupby_sum`, `groupby_min/max`
* `groupby_logsumexp` (for probabilistic semirings)

### 2.4 Fixpoint loop kernels

* `delta_apply`
* `delta_diff` (Δnew = Δcand \ R)
* termination detection (Δ empty)

## 3) Circuit kernels (xlog-prob)

### 3.1 Forward evaluation (log-space)

* `gather_leaf_weights(lit_ids → logw)` with evidence masks
* `eval_level_AND`
* `eval_level_OR` (logsumexp)
* `eval_level_DECISION` (logsumexp with var probabilities)

ProbLog’s documentation and tutorials emphasize compilation targets such as smoothed deterministic decomposable NNF (d-DNNF) to enable correct disjoint-sum handling. ([Departement Computerwetenschappen][11])

### 3.2 Backward (reverse-mode) kernels

* `backprop_AND`
* `backprop_OR` (softmax-style contributions `exp(v_child - v_parent)`)
* `backprop_DECISION` (split gradients to var probabilities and child nodes)
* `scatter_leaf_grads`

### 3.3 Autograd interface

* a single fused “forward+backward” API exposed to PyTorch as a custom op.

DeepProbLog motivates differentiable neural predicates and learning through probabilistic logic programs. ([arXiv][9])

## 4) Solver kernels (xlog-solve)

### 4.1 Solve-LS (GPU local/continuous local search)

* `clause_eval(assignments_batch → satisfied_mask)`
* `score_update` / `gradient_compute` (CLS-specific)
* `variable_flip` / `restart`

FastFourierSAT provides a concrete GPU CLS design and reports large speedups in gradient computation and promising solving performance. ([arXiv][10])

### 4.2 Solve-Exact (proof-oriented)

* GPU inprocessing kernels:

  * subsumption checks
  * variable elimination candidates
  * clause compaction/GC
* proof emission + verification hooks

ParaFROST demonstrates GPU accelerated inprocessing and a certified variant with proofs that can be verified. ([gears.win.tue.nl][4])

## 5) ELP kernels (xlog-elp)

### 5.1 Candidate representation kernels

* bitset-based candidate storage: `cand_bits[candidate_id][word]`
* structural propagation:

  * apply implications (bitwise AND/OR masks)
  * dead epistemic literal elimination

### 5.2 Semantic propagation batching

* build a `TestBatch`:

  * (candidate_guess_id, test_type, objective_literal)
* dispatch to `xlog-solve`:

  * LS first for counterexamples
  * Exact when proof/universal property required

This corresponds to the guess-check pattern used in eclingo (cautious/brave checks) and the propagation-driven pruning idea in generate-and-test with propagation. ([arXiv][1])

### 5.3 Foundedness checker kernels (FAEEL default)

* build foundedness-check SIR instance per world view candidate
* dispatch to `xlog-solve` for existence of an unfounded set (if found → reject)

FAEEL’s paper defines foundedness via unfounded sets and states the equivalence with founded G91 world views. 

## 6) Multi-GPU execution support (kernel plan hooks)

* partition relations via radix-hash on keys
* all-to-all delta exchange
* skew-aware replication of hot keys

mnmgDatalog provides evidence that multi-node, multi-GPU Datalog with radix-hash partitioning and GPU-aware all-to-all communication is feasible and beneficial. ([dl.acm.org][16])

---

# What you can do immediately with these deliverables

1. **Start implementing `xlog-logic`** using libcudf joins/groupby, with the RIR pipeline above, while simultaneously building the memory-budget enforcement.
2. **Prototype `xlog-solve` early** as planned, starting with a batched clause-eval kernel + a simple GPU local search baseline, then iterating toward FastFourierSAT-style CLS integration. ([arXiv][10])
3. **Implement `xlog-prob` exact path** using D4 as the first KC backend and XGCF as the stable GPU representation. ([IJCAI][2])
4. **Implement `xlog-elp` in two passes:**

   * G91 via eclingo-style guess-check structure (but GPU batched), ([arXiv][1])
   * then add the foundedness filter to obtain the default FAEEL semantics. 

---

If you want the next step to be even closer to code, I can turn these deliverables into:

* concrete C++/CUDA interface headers for the kernel provider boundary,
* a JSON/FlatBuffers schema for serialized RIR/PIR/EIR/SIR graphs,
* and detailed pseudocode for the ELP Generate–Propagate–Test loop (including batching layouts and solver call contracts).

[1]: https://arxiv.org/abs/2008.02018?utm_source=chatgpt.com "eclingo: A solver for Epistemic Logic Programs"
[2]: https://www.ijcai.org/proceedings/2017/0093.pdf?utm_source=chatgpt.com "An Improved Decision-DNNF Compiler"
[3]: https://dtai.cs.kuleuven.be/problog/tutorial/python/01-compile-once.html?utm_source=chatgpt.com "Recipe: compile-once evaluate-many — ProbLog - DTAI"
[4]: https://gears.win.tue.nl/papers/parafrost_gpu.pdf?utm_source=chatgpt.com "SAT Solving with GPU Accelerated Inprocessing"
[5]: https://arxiv.org/abs/1907.09247?utm_source=chatgpt.com "Founded (Auto)Epistemic Equilibrium Logic Satisfies Epistemic Splitting"
[6]: https://arxiv.org/abs/2001.01089?utm_source=chatgpt.com "selp: A Single-Shot Epistemic Logic Program Solver"
[7]: https://www.ijcai.org/proceedings/2018/0237.pdf?utm_source=chatgpt.com "Single-Shot Epistemic Logic Program Solving"
[8]: https://www.ijcai.org/proceedings/2018/237?utm_source=chatgpt.com "Single-Shot Epistemic Logic Program Solving"
[9]: https://arxiv.org/abs/1805.10872?utm_source=chatgpt.com "[1805.10872] DeepProbLog: Neural Probabilistic Logic Programming"
[10]: https://arxiv.org/abs/2308.15020?utm_source=chatgpt.com "Massively Parallel Continuous Local Search for Hybrid SAT Solving on GPUs"
[11]: https://dtai.cs.kuleuven.be/problog/tutorial/advanced/00_inference.html?utm_source=chatgpt.com "Inference in ProbLog - Probabilistic Programming - DTAI"
[12]: https://cca.informatik.uni-freiburg.de/papers/OsamaWijsBiere-FMSD23.pdf?utm_source=chatgpt.com "Certified SAT solving with GPU accelerated inprocessing"
[13]: https://arxiv.org/abs/2311.02206?utm_source=chatgpt.com "Optimizing Datalog for the GPU"
[14]: https://arxiv.org/abs/2501.13051?utm_source=chatgpt.com "Column-Oriented Datalog on the GPU"
[15]: https://arxiv.org/html/2311.02206v5?utm_source=chatgpt.com "Optimizing Datalog for the GPU"
[16]: https://dl.acm.org/doi/10.1145/3721145.3730431?utm_source=chatgpt.com "Multi-Node Multi-GPU Datalog | Proceedings of the 39th ..."
