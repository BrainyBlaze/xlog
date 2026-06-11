# Factorized Hypergraph Execution — Research Report

Status: RESEARCH (design-phase groundwork; no implementation in this document's scope)
Date: 2026-06-11
Origin: F2 of the 2026-06-10 GPU-paths audit — "factorized deterministic
execution" was confirmed absent from the codebase (zero matches for any
factorized representation). This report establishes (a) what the algorithm
landscape offers, (b) what XLOG already has that a factorized engine could
reuse, and (c) ranked integration directions, each gated by a bench spike
per the established perf discipline (spike first; spike branches stay
unmerged as evidence).

Evidence basis: a 108-agent multi-source research sweep with 3-vote
adversarial verification (24 claims confirmed unanimously, 1 refuted) for
the theory sections, four codebase exploration passes for §2, and a
targeted single-pass primary-source sweep (Free Join PDF, F-IVM, Galley,
VFLog) for the systems sections. Sections are labeled **[verified]** or
**[single-pass]** accordingly.

---

## 1. Problem Statement

XLOG's deterministic engine materializes every intermediate: binary joins
emit ALL left+right columns (`relational.rs:4331-4339`), WCOJ kernels emit
full output rows, aggregates always consume a fully materialized join
output (`RirNode::GroupBy` has no pushdown form, `rir.rs:394-400`), and
recursive deltas are full-width columnar copies diffed by full-row equality
(`recursive.rs:673-762`). The WCOJ kernels bound *output* size at the AGM
bound for the supported cyclic shapes, but for acyclic-but-wide queries,
aggregate-over-join queries, and recursive deltas, materialization is the
asymptotic cost.

Factorized execution (f-/d-representations, FAQ-style aggregate pushdown,
Free Join-style lazy tries) replaces materialized relations with nested,
shared representations whose size is bounded by fractional hypertree-width
measures, on which counting/aggregation/enumeration remain efficient
without flattening. The question this report answers: which of these
techniques survive contact with a GPU-resident, recorded-launch,
zero-tracked-transfer engine — and in what order to try them.

## 2. Codebase Ground Truth

(Four exploration passes, 2026-06-11; spot-corrections by hand where noted.)

### 2.1 Assets a factorized engine can reuse

| Asset | Where | Why it matters |
|---|---|---|
| Lex-sorted, deduplicated 2-col edge arrays (u32/u64) | WCOJ layout, `provider/wcoj.rs` | A sorted column pair IS a 2-level trie in flat form; level-1 factorization already exists |
| `WcojRelationMetadata { unique_keys, fan_out, prefix_sum }` | `xlog-cuda/src/wcoj_metadata.rs:37-104` | Exactly a CSR/first-trie-level: distinct keys + offsets into sorted payloads. A d-representation's union/product nodes can be encoded the same way |
| Histogram-guided work plans (`xy_work_prefix`, block counts/offsets) | `wcoj_metadata.rs:48-62` | The deterministic two-phase (count→materialize) discipline + load balancing carries to factorized construction unchanged |
| Binary-search + sorted-merge intersection primitives | `kernels/wcoj.cu:51-141` | The core ops of trie-join (LFTJ-style) already exist as atomics-free device functions |
| Hypergraph IR (`HypergraphRule`, `Vertex`, `Hyperedge`) | `xlog-logic/src/hypergraph/ir.rs:12-104` | Total construction from AST; the input a decomposition planner needs |
| Greedy stats-driven variable-order planner (`FullVariableOrder`: cost prediction, variable shares, helper splits) | `hypergraph/var_order.rs:334-364` | A variable order IS the skeleton of an f-representation; planner + `StatsSource` (cardinality, NDV, prefix degree, key heat) are directly reusable |
| K-clique planned dispatch K=2..8 incl. K7/K8 (W6.4) | `provider/wcoj.rs:2063,2141`; `rir.rs:147-187` | Plan plumbing (edge permutation, column swaps, sorted-layout specs, stream groups) shows how to carry a richer plan from RIR to kernels |
| Stats: NDV (HLL), prefix degrees (avg/max), key heat/skew | `xlog-stats/src/stats.rs` | Inputs for fhtw-aware planning and factorization-benefit estimation |
| `count_lift_gpu` aggregate lifting | `xlog-prob/src/exact.rs:141-227`, `provenance.rs:1286-1615` | Existing in-house proof that structure-exploiting shortcuts (DP over count outcomes instead of 2^n enumeration) work in this engine |
| Device-resident fixpoint precedent (MC resident engine) | `mc/resident.rs`, `kernels/mc_resident.cu` | Sparse slot arenas + per-world counters + device convergence flags show the allocation/contract style any factorized device structure must follow |

(Exploration-agent correction: one pass claimed K=7/8 kernels are absent;
they exist — `wcoj_clique7/8_u32_recorded_planned`, `provider/wcoj.rs:2063,2141`.)

### 2.2 Gaps (confirmed absent)

- No trie / prefix-sharing structure; no d-representation node vocabulary;
  intermediates are always flat `CudaBuffer` SoA.
- No (fractional) hypertree decomposition planner — only fixed shapes
  (triangle, 4-cycle, K-clique 2..8) promote to WCOJ; everything else takes
  the binary-join tree.
- No aggregate pushdown: `GroupBy` consumes materialized join output only;
  it sorts the FULL buffer by key columns first (`provider/groupby.rs:170`).
- No factorized deltas: recursion stores full-width delta buffers, diffs by
  full-row sorted binary search (`relational.rs:1039-1156`).
- No cross-rule or cross-iteration sharing of intermediates.

### 2.3 Hard constraints any design must satisfy

1. **Zero tracked transfers in the data plane**; bounded
   `dtoh_scalar_untracked` metadata reads only. Factorized structures must
   be built and consumed device-side; size discovery via the established
   count→scan→materialize two-phase pattern.
2. **Recorded-launch discipline**: launch-sequence shapes known at record
   time; data-dependent depth handled by the host-orchestrated loop or by
   device-side fixpoints (MC engine precedent).
3. **Deterministic set semantics**: full-row dedup/diff with totalOrder
   float comparators. Sorted tries are *canonical*, which makes
   factorized dedup/diff well-defined — an advantage over hash tries.
4. **Fail-closed typing**: unsupported shapes decline (counted via
   `wcoj_error_decline_count`) or reject typed — never silently wrong.
5. **Memory budgets** (`GlobalDeviceBudget`, `MemoryBudget`): preallocate;
   no host-driven mid-kernel resizing.
6. **API edges are flat**: DLPack/Arrow consumers and the relation store
   expect flat columnar buffers — factorized intermediates need explicit
   flatten boundaries (or aggregate consumption) at the edges.

### 2.4 Materialization choke points (asymptotic targets)

| # | Site | Cost today | Factorized counterpart |
|---|---|---|---|
| C1 | Binary join output = ALL columns both sides (`relational.rs:4331`) | O(out × (aL+aR)) | f-rep: size bounded by fhtw; or at minimum column pruning |
| C2 | GroupBy full-buffer sort after materialized join (`groupby.rs:170`) | O(rows × arity) sort | FAQ/LMFAO: aggregate pushed through the variable order; the join is never materialized |
| C3 | Recursive full/delta full-width buffers + per-iteration union/diff (`recursive.rs:673-840`) | 3× materialization per iteration | factorized deltas; or at minimum delta-key compaction |
| C4 | WCOJ kernels always emit flat (X,Y,Z,…) rows | O(matches × K) | enumerate-on-demand from layout metadata; emit aggregates directly from the kernel walk |
| C5 | Provenance: non-count aggregates enumerate 2^k outcome masks, k≤16 (`provenance.rs:1417-1451`); D4 frontier blowup on unstructured CNF | exponential pockets | factorized provenance circuits: PIR arrives decomposable; D4 work shrinks (count-lift precedent) |

## 3. Algorithm Landscape

### 3.1 Factorized databases: f-/d-representations **[verified, 3-0 votes]**

Core results (Olteanu & Závodný, TODS 2015; Bakibayev et al., PVLDB 2012;
SIGMOD Record 2016):

- **Representations.** f-representations are algebraic expressions over
  singletons, union, and product, with nesting governed by an f-tree;
  compression comes from distributivity of product over union.
  d-representations add *named, reusable subexpressions* (definitions) —
  structurally, circuits. ["Factorised representations of relations are
  algebraic expressions constructed using singleton relations and the
  relational operators union and product."]
- **Size bounds.** Q(D) admits an f-representation of size O(|D|^s(Q)) and
  a d-representation of size O(|D|^s↑(Q)), with **s↑(Q) = fhtw(Q)** for
  equi-joins (two-way translation d-trees ↔ fractional hypertree
  decompositions). Bounds are asymptotically optimal within
  f-tree/d-tree-structured representations.
- **Hierarchy.** 1 ≤ s↑(Q) ≤ s(Q) ≤ ρ*(Q) ≤ |Q|; the gap between the AGM
  flat exponent ρ* and the factorized exponent can be as large as |Q| —
  i.e., **factorized results can be exponentially more succinct than what
  WCOJ materializes flat**, with s(Q)=1 for arbitrarily large queries.
- **Tightness (ICDT 2026, Berkholz & Vinall-Smeeth, arXiv:2503.20438).**
  d-representations are structured deterministic {∪,×}-circuits;
  unconditional N^Ω(tw) circuit lower bound for bounded arity (tight up to
  a constant in the exponent), and submodular-width bounds
  O(N^{(1+δ)subw}) vs N^Ω(subw^{1/4}) for unbounded arity. Width governs
  size; no cleverer circuit construction escapes it.
- **Operations without materialization.** Constant-delay tuple
  enumeration; one-pass semiring aggregates (count/sum/min/max, model
  counting, ML gradient aggregates); aggregates-on-top-of-joins computed
  by propagating distributive aggregates up the recursion **without
  materializing the factorized join**; computable in worst-case-optimal
  time modulo log factors (FDB f-plan operators are quasilinear in
  input+output representation size). Caveats that matter for us:
  group-by/order-by constant-delay enumeration holds **iff** the variable
  order satisfies structural conditions (group-by variables at roots /
  children of group-by variables), and assumes sorted unions.

Sources: oz-tods15.pdf, boz-vldb12.pdf, SIGMOD Rec. 10.1145/3003665.3003667,
o-beyondnp16.pdf, arXiv:2503.20438, fdbresearch.github.io.

### 3.2 FAQ / InsideOut **[verified, 3-0]**

InsideOut (Abo Khamis/Ngo/Rudra, PODS 2016) solves Functional Aggregate
Queries by variable elimination over a variable order with WCOJ-grade
analysis: runtime Õ(N^faqw + output), where faqw generalizes fhtw. The
main technical contribution is characterizing when a variable ordering is
semantically equivalent to the input's, plus an approximation algorithm
minimizing fractional FAQ-width. **Unification result [verified]:** the
worst-case-optimal factorization algorithm *degenerates to LeapFrog
TrieJoin* exactly when the variable order is a branchless path with no
sharing — tries/d-trees over variable orders are the single data structure
linking WCOJ, factorized execution, and FAQ; FAQ is the bottom-up DP,
factorization the top-down memoized variant, with equal runtime complexity.

Honesty note: the popular claim that FAQ "subsumes CSP, databases, matrix
ops, PGM inference, and logic" was **refuted 0-3** in adversarial
verification — do not repeat the subsumption rhetoric; only the
algorithmic/width results above are verified. PANDA and AJAR produced no
verified claims in this run.

### 3.3 Free Join (Wang/Willsey/Suciu, SIGMOD 2023) **[single-pass, primary PDF]**

Extracted directly from arXiv:2301.10841 (PDF):

- **COLT (Column-Oriented Lazy Trie):** "builds the inner subtries lazily,
  by creating each subtrie on demand … adapts the lazy trie … to use a
  column-oriented layout. And unlike the original lazy trie which builds
  at least one trie level per table, COLT completely eliminates the cost
  of trie building for left tables."
- **Plan space:** Free Join plans are sequences of nodes over *subatoms*
  (an atom restricted to a variable subset) with *partitionings*; a
  `binary2fj` conversion turns any binary plan into an equivalent Free
  Join plan, then optimization moves it "anywhere between a left-deep plan
  or a Generic Join plan."
- **Vectorized execution** is integral ("collect multiple data values
  before entering the next iteration level").
- **Explicit factorization bridge:** "we can view the trie data structure
  as a factorized representation of a relation … we can use this
  factorized representation to compress large outputs, saving time and
  space during materialization."
- **Results:** implemented in Rust; "matches or outperforms both
  traditional binary joins and Generic Join on standard query benchmarks."

Relevance: Free Join is the practical recipe for *incremental adoption* —
it shows hash-join plans and WCOJ are endpoints of one spectrum, with
laziness recovering binary-join performance on acyclic queries. XLOG
already lives at both endpoints (hash_join_v2 + fixed-shape WCOJ) with no
bridge between them.

### 3.4 LMFAO / F-IVM **[single-pass]**

F-IVM (Nikolic, Olteanu et al.; VLDB J 2023, arXiv:2303.08583): unified
incremental maintenance via (1) higher-order IVM, (2) factorized
computation, (3) a **ring abstraction** — views map keys to payloads from a
ring; one view hierarchy maintains group-by aggregates, linear-regression
covariance matrices, Chow-Liu trees, and matrix chains, outperforming
first-order/recursive IVM "by orders of magnitude while using less
memory." LMFAO (SIGMOD 2020 lineage, same group) batches many aggregates
over one factorized join tree. Neither has verified claims in this run —
treat quantitative numbers as unverified; the architectural pattern
(payload rings over a shared view tree) is what matters for XLOG's
neural/dILP aggregate workloads.

### 3.5 Factorized provenance & knowledge compilation **[verified, 3-0]**

Factorized query results are formally a knowledge compilation language —
deterministic Decomposable Ordered Multi-valued Decision Diagrams
(d-DOMDDs), "a special instance" of **d-DNNFs over vtrees**;
d-representations are deterministic structured {∪,×}-circuits, where
determinism gives linear-time model counting and constant-delay model
enumeration (Beyond-NP 2016; corroborated by Amarilli et al. ICALP 2017
and arXiv:2503.20438).

This is the deepest XLOG-specific synergy: **XLOG's exact engine already
evaluates d-DNNF-class circuits on GPU (XGCF) and pays its exponential
cost in GPU-D4 compilation of unstructured CNF.** A factorized join
evaluation *constructs* a deterministic decomposable circuit as a
byproduct — for the join-structure part of provenance, the D4 step's work
could be partially pre-done by construction, the same way `count_lift_gpu`
already bypasses D4 for count aggregates. The codebase pass (§2, C5)
identified the matching hooks: PIR is already a shared DAG with interning
(`provenance.rs:186-322`), so the change is in *what* gets built, not the
substrate.

### 3.6 GPU / data-parallel state of the art **[verified gap + single-pass]**

The adversarial verification produced **zero surviving claims** of any
published GPU/data-parallel implementation of factorized joins, FAQ
solvers, or WCOJ-with-factorized-intermediates — the verified literature
is sequential/RAM-model. Single-pass findings on the adjacent systems:

- **VFLog** (arXiv:2501.13051, 2025): CUDA Datalog runtime, column-oriented
  GPU datastructure, "over 200x" vs CPU column-oriented Datalog and ~2.5x
  over GPU Datalog engines — but no WCOJ, no factorization. GDlog /
  "Optimizing Datalog for the GPU" (ASPLOS'24 lineage) similar: hash-trie
  flat layouts, binary joins.
- **Galley** (arXiv:2408.14706, Deeds/Ahrens/Balazinska/Suciu): cost-based
  lowering of sparse tensor algebra using "a novel extension of the FAQ
  framework"; 1-300× on ML-over-joins; CPU sparse-tensor compilers, no GPU
  evidence.
- The project's canonical SRDatalog reference (arXiv:2604.20073, P1-P5)
  was independently fetched by the sweep under the WCOJ angle — designs
  here must stay aligned with its claims as before.

**Conclusion: GPU-resident factorized Datalog execution is unoccupied
territory.** Nobody has published flat, level-synchronous, columnar
encodings of d-representations with recorded-launch construction. That is
both the risk (no recipe to copy) and the opportunity (novel contribution
if the bench spikes pan out).

### 3.7 Factorized representations in recursive Datalog **[single-pass gap]**

Targeted search found **no published work** combining factorized
representations with semi-naive recursive evaluation (nearest neighbors:
semiring Datalog convergence theory — Khamis et al., "Convergence of
Datalog over (Pre-)Semirings"; differential/incremental Datalog —
FlowLog, DDlog lineage; F-IVM for non-recursive views). Whether
d-representations compose under fixpoint with maintained width bounds is
an open question (also flagged as an open question by the verified sweep).
For XLOG this means recursion-facing factorization (D3 below) is research,
not engineering.

## 4. Integration Directions (ranked)

Ranking criteria: theory confidence × reuse of §2.1 assets × respect for
§2.3 constraints × expected asymptotic win at §2.4 choke points.

### D1 — FAQ-style factorized aggregates in the WCOJ walk (RECOMMENDED FIRST)

Push distributive aggregates (count/sum/min/max; semiring-generalized
later) through the existing variable-order walk instead of
materialize-then-GroupBy. Targets **C2 + C4**.

- Theory: verified — one-pass aggregate propagation up the variable order,
  never materializing the join (§3.1); InsideOut equivalence (§3.2).
- Mechanics: the triangle/clique kernels already walk prefixes in variable
  order; instead of `intersect_emit_xyz` writing rows, an aggregate
  variant accumulates per-prefix partial aggregates (block-local then
  global scan — same two-phase discipline). RIR gains a fused
  `GroupBy{input: MultiWayJoin}` recognition (planner rewrite), fail-closed
  to the existing path for unsupported shapes.
- Reuse: kernels, work plans, metadata, cost model, dispatch counters.
  No new persistent data structure. No flatten-boundary problem (output is
  already small).
- Synergy: `count_lift` (prob aggregates) and dILP credit aggregation are
  downstream consumers of the same pattern.

### D2 — COLT-style flat lazy tries + Free Join plan space (general-arity bridge)

Generalize beyond the fixed shapes by adopting Free Join's plan space
(subatoms/partitionings) over a **level-synchronous, flat columnar trie**:
eager two-phase construction per level (count→scan→materialize — the
recorded-launch-compatible replacement for COLT's on-demand laziness),
`WcojRelationMetadata` as the level-1 encoding. Targets **C1** and the
general-arity WCOJ gap (W3.2's general-arity template ambition).

- Theory: Free Join shows binary and WCOJ plans are one spectrum
  [single-pass]; tries are factorized representations [verified bridge].
- Risk: COLT's laziness is what makes acyclic queries cheap; a fully eager
  per-level build may pay trie-construction cost binary joins avoid. The
  spike must measure exactly this. Mitigation: build levels only for
  subatoms the plan actually probes (plan-time laziness instead of
  run-time laziness).
- Reuse: planner (`FullVariableOrder` generalizes), intersection
  primitives, histogram work plans, stats.

### D3 — d-representation intermediates + factorized recursive deltas (research)

Keep join intermediates as d-representations (flat circuit encoding:
per-level union arrays + product offsets, definitions as shared level
segments) and extend to semi-naive deltas. Targets **C1 + C3**. Highest
asymptotic ceiling (fhtw vs ρ*: exponentially smaller intermediates),
highest novelty (no published recursive-factorized work, §3.7), highest
risk (canonical-form maintenance under union/diff; width planning;
flatten boundaries). Sorted canonical tries make factorized dedup/diff
*definable*; whether they're efficient on GPU is the open question. Do not
attempt before D1/D2 establish the flat-trie substrate.

### D4 — Factorized provenance into PIR/XGCF (probabilistic synergy)

Construct provenance as deterministic decomposable circuits during
(factorized) join evaluation, so the exact engine's D4 compilation
inherits structure instead of rediscovering it from flat CNF. Targets
**C5**. Verified theory bridge (§3.5: d-reps ⊂ d-DNNF), in-house precedent
(`count_lift_gpu`), and the codebase hooks are mapped
(aggregate-outcome folding, rule proof-path sharing, D4 decision-order
hints — `provenance.rs:1417-1451`, `:687-829`, `gpu_d4/mod.rs:54-76`).
Uniquely valuable to XLOG (few engines own a GPU d-DNNF stack), but
sequenced after D1 since the cheap wins there (D4 hints, aggregate-outcome
folding) don't require the factorized deterministic engine at all.

### Cross-cutting: width-aware planning

D2-D4 eventually need an fhtw-aware decomposition planner.
`HypergraphRule` + `StatsSource` are the inputs; the greedy variable-order
scorer is the starting point. Exact fhtw is NP-hard — use the existing
greedy + stats heuristics with cost-model gating (the W2.5 cardinality
cost-model discipline), not exact decomposition.

## 5. Bench-Spike Gates (per perf discipline: spike before any plan)

Each spike: minimum-viable, on a `bench-spike/*` branch, preserved
unmerged as evidence; production GPU box; LP-MULTI-RUN protocol where
applicable; fail the gate → direction parked with evidence.

| Spike | Direction | Workload | Gate |
|---|---|---|---|
| S1 aggregate-fused WCOJ | D1 | `q(X, count(*)) :- e1(X,Y), e2(Y,Z), e3(X,Z)` on the W5.2 hub/skew fixtures, vs materialize+sort+groupby | ≥5× wall-clock on skewed fixtures AND ≤1.1× regression on small uniform; zero tracked transfers |
| S2 flat trie + Free Join plan | D2 | 4-atom acyclic chain/star with large intermediate (binary plan blowup case) + triangle (parity case) | beat binary join ≥2× on the blowup fixture; ≤1.2× of current WCOJ on triangle |
| S3 factorized delta | D3 | transitive closure on dense block graph (delta blowup) | ≥5× peak-memory reduction at wall-clock ≤1.2×; deterministic row-set parity with current engine |
| S4 provenance structure hints | D4 | exact inference on a join-heavy probabilistic program | measurable D4 frontier/compile-time reduction (target ≥30% per the codebase analysis) with identical probabilities (1e-9) |

S1 first: it is the only spike with verified theory, kernel-local change
surface, and no new persistent data structure.

## 6. Risks and Open Questions

1. **GPU encoding of definitions/sharing** — cached definitions
   (d-rep sharing) reintroduce pointer-chasing and load imbalance; the
   verified literature is RAM-model. Mitigation: level-synchronous flat
   segments + the existing histogram/block-slice balancing; the W33
   persistent-threads work-stealing line is directly relevant to irregular
   fanout. (Open question also flagged by the verified sweep.)
2. **Laziness vs recorded launches** — COLT's on-demand subtries imply
   data-dependent allocation. Two-phase eager-per-level is the compliant
   substitute; its overhead is exactly what S2 measures.
3. **When factorization loses** — fhtw = ρ* for the already-supported
   clique shapes (no succinctness win there; D1's aggregate win still
   applies). Shape/stats gating must route flat execution when widths
   coincide; extend the W2.5 cost model rather than adding a new oracle.
4. **Flatten boundaries** — store/DLPack/Arrow edges are flat; factorized
   intermediates must be consumed (aggregates) or flattened before the
   edge. Enumeration-to-flat is output-linear [verified constant-delay],
   so the boundary is safe but must be explicit in any design.
5. **Recursion theory gap** — no published factorized-delta semantics;
   width bounds under fixpoint composition unproven. D3 is research with
   publication potential, not a scheduled feature.
6. **Group-by order constraints** — one-pass aggregates require compatible
   variable orders [verified iff-condition]; the planner must check the
   condition and fall back fail-closed.
7. **Claim hygiene** — the FAQ "unifies everything" claim is refuted; keep
   XLOG's documentation to the verified statements (size bounds, one-pass
   aggregates, LFTJ degeneration). Per the audit discipline: no capability
   claims until a spike's gate evidence exists.
8. **SRDatalog alignment** — designs must not contradict the P1-P5 claims
   of arXiv:2604.20073 (the project's canonical WCOJ/recursive reference).

## 7. Recommendation

Proceed with **S1 (aggregate-fused WCOJ)** as the first authorized spike:
verified theory, kernel-local blast radius, reuses every §2.1 asset,
respects every §2.3 constraint, and its win condition (aggregate queries
over skewed joins) is a real workload class for the DTS/dILP consumers.
D2 follows if S1's substrate proves out; D4's cheap subset (D4 hints +
aggregate-outcome folding in provenance) can proceed in parallel since it
is independent of the deterministic engine; D3 stays parked until D1+D2
evidence exists.

---

## Appendix: Primary sources

Verified (3-0 adversarial votes): Olteanu & Závodný TODS 2015
(cs.ox.ac.uk/dan.olteanu/papers/oz-tods15.pdf); Bakibayev/Olteanu/Závodný
PVLDB 2012 (boz-vldb12.pdf); Olteanu & Schleich SIGMOD Record 2016
(doi 10.1145/3003665.3003667); Olteanu Beyond-NP 2016 (o-beyondnp16.pdf);
Berkholz & Vinall-Smeeth ICDT 2026 (arXiv:2503.20438); Abo
Khamis/Ngo/Rudra PODS 2016 (arXiv:1504.04044); Amarilli et al. ICALP 2017
(arXiv:1702.05589). Refuted (0-3): FAQ-subsumption rhetoric.

Single-pass (not adversarially verified): Free Join, Wang/Willsey/Suciu
SIGMOD 2023 (arXiv:2301.10841, PDF extraction); F-IVM VLDB J 2023
(arXiv:2303.08583); Galley (arXiv:2408.14706); VFLog (arXiv:2501.13051);
GDlog/“Optimizing Datalog for the GPU” (arXiv:2311.02206); FlowLog
(arXiv:2511.00865); Khamis et al. “Convergence of Datalog over
(Pre-)Semirings” (arXiv:2105.14435).
