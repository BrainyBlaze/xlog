# D2 — GPU Free Join: Level-Synchronous Factorized Join Execution

Status: DESIGN + phased implementation plan (S2 spike gate before
production wiring; no slice ships without its gate evidence).
Origin: `docs/plans/2026-06-11-factorized-hypergraph-research.md` §4 D2.
Primary source: Wang/Willsey/Suciu, "Free Join: Unifying Worst-Case
Optimal and Traditional Joins", SIGMOD 2023 (arXiv:2301.10841) — verbatim
extracts below are from the PDF.

## 1. What Free Join is (exact, from the paper)

- **Plan**: a list of *nodes* `[φ1..φm]`; each node is a list of
  *subatoms* `R(y)` (atom restricted to a variable subset); across the
  whole plan, each atom's subatoms partition its variables. Variables
  available to node k are those of all preceding nodes.
- **Execution** (`fn join(all_tries, plan, tuple)`): node k iterates its
  FIRST subatom (the *cover*), probes the remaining subatoms with the
  already-bound variables, and recurses into subtries. A left-deep binary
  plan corresponds to alternating [iterate-full-atom, probe] nodes
  (`binary2fj`); Generic Join is the degenerate plan with one variable
  per node. Validity: every probed subatom's variables must be available
  (covered by preceding nodes + the node's cover).
- **COLT** (Column-Oriented Lazy Trie): trie levels built lazily on
  demand; the cover ("left") tables are never trie-built at all.
- Reported results: up to 19.36× over binary join (acyclic) and 31.6×
  over Generic Join; vectorized execution is integral.

## 2. The GPU adaptation (the novel part)

The paper's depth-first recursion and on-demand hash-trie construction
are CPU idioms. The XLOG adaptation replaces both:

### 2.1 Flat sorted-range tries (replaces COLT hash tries)

For a lex-sorted, deduplicated k-column relation, **a trie node is a
contiguous row range** `[lo, hi)` and `get(key)` is a binary-search
refinement of that range on the next column. Therefore:

- The trie *is* the existing WCOJ layout (`wcoj_layout_u32_recorded`:
  sort + dedup, with a sorted-fast-path check). Zero additional build
  cost, zero pointers, fully recorded-launch compatible, canonical (so
  determinism/dedup contracts hold by construction).
- COLT's laziness is subsumed: no per-level structure is ever built —
  levels materialize as *ranges held by the consumer*, the strictest
  possible laziness. The cover atom is consumed as plain columns.
- This generalizes `WcojRelationMetadata{unique_keys, fan_out,
  prefix_sum}` (a *materialized* level-1) — which remains useful as an
  optional accelerator for first-level expansion of high-reuse atoms,
  but is not required for correctness.

### 2.2 Level-synchronous frontier execution (replaces DFS recursion)

Maintain a **bindings frontier**: a columnar device buffer where each row
is one partial binding. Columns: the bound variables (u32 each) plus, per
not-yet-exhausted atom, a `(lo, hi)` u32 range pair into that atom's
sorted buffer (its current subtrie). Plan nodes execute as bulk
operations over the whole frontier:

For node φk with cover subatom C(y_c) and probe subatoms P1(y_1)..Pj(y_j):

1. **EXPAND (cover)**: each frontier row's range for C contains
   `hi - lo` candidate extensions grouped by the cover variables. Run the
   established two-phase discipline: count kernel (per-row distinct
   cover-prefix group count via the sorted layout) → exclusive scan →
   emit kernel writing the expanded frontier (parent columns copied,
   new variables bound, C's range refined to the group's subrange).
   This is exactly the histogram-guided work-plan pattern the
   triangle/4-cycle kernels already use, generalized.
2. **PROBE (each Pi)**: one kernel per probe subatom: for each expanded
   row, binary-search Pi's current range on the probe key columns
   (variables of y_i, all bound); write the refined `(lo, hi)` or a
   0-mask on miss. Then one compaction (existing mask + scan + gather
   kernels) removes dead rows. Probes after expansion = Free Join's
   "probe into other tries"; batching across the frontier = the paper's
   vectorized execution taken to its limit.
3. After the last node, the frontier rows ARE the join result for the
   projected variables (plus untouched trailing ranges, see §2.4):
   materialize via the head projection, or feed the aggregate path.

Memory: the frontier after node k has exactly one row per *distinct
binding of the variables bound so far that survives all probes so far* —
i.e. it is the factorized prefix set of the f-representation induced by
the plan's variable order. Peak memory = max frontier × row width, which
the planner bounds with the existing cardinality stats; an over-budget
estimate declines to the binary path (fail-safe, counted).

### 2.3 Correctness invariants (non-negotiable)

- All inputs layout-normalized per dispatch (31b0ccf0 contract).
- Two-phase count→scan→emit for every expansion (no atomics in emit
  paths; deterministic output order = lex order of the plan's variable
  sequence, making results canonical without re-sorting).
- Set semantics: inputs deduped; expansion by distinct cover groups;
  per-node probe refinement cannot duplicate rows ⇒ frontier stays
  duplicate-free by construction; a final dedup is therefore NOT needed
  when the head projects all bound variables — when the head projects a
  subset, reuse the existing projection+dedup path (same as binary).
- Zero tracked transfers; row counts via the cached/untracked metadata
  discipline; recorded launches throughout.

### 2.4 Factorized payoffs kept (not deferred)

- **Aggregate pushdown composes**: count-by-root terminates at the last
  node by summing, per root group, the *product of remaining range
  sizes* for independent trailing atoms — the d-representation count.
  The existing fused group-by-root reduction (staging → compact →
  recorded groupby) is reused on the frontier.
- **Output compression**: when the final node leaves trailing cover-only
  variables (paper's "factorized representation to compress large
  outputs"), materialization defers their cross-product to the
  enumeration kernel — output size bounded by the factorized size until
  the consumer demands flat rows.

## 3. Plan IR, planner, dispatch

- **RIR**: new `RirNode::FreeJoin { inputs: Vec<RirNode /*Scan*/>,
  nodes: Vec<FjNode>, var_classes, output_columns, fallback:
  Box<RirNode> }` where `FjNode { cover: SubAtom, probes: Vec<SubAtom> }`
  and `SubAtom { input_idx, var_positions }`. Mirrors MultiWayJoin's
  contract (fallback embedded; store never partially mutated).
- **Planner** (xlog-logic): `binary2fj` from the existing bushy/left-deep
  plan (paper Fig. 9), then the paper's optimizations: (a) push probe
  subatoms into the earliest node where their key variables are
  available; (b) split nodes (toward GJ) only when the cardinality model
  predicts intermediate blowup (stats: NDV/prefix degrees already
  available). Triangle/4-cycle/K-clique shapes keep their dedicated
  promoters and kernels — Free Join handles every OTHER ≥3-atom
  conjunctive body (and 2-atom bodies stay on ChainJoin). Promotion is
  shape-gated + cost-gated, default ON with `XLOG_DISABLE_FREE_JOIN`
  kill switch and `free_join_dispatch_count` counter; every structural
  mismatch declines silently to the embedded binary fallback; pipeline
  errors via `wcoj_decline_on_error` ("free-join" stage).
- **Executor**: `try_dispatch_free_join` in wcoj_dispatch.rs following
  the established prologue (names → buffers → width classify → runtime →
  stream) then the frontier loop driving the new provider entries.
- **Provider/kernels** (xlog-cuda): `fj_expand_count_u32`,
  `fj_expand_emit_u32`, `fj_probe_refine_u32` (+ `_u64` mirrors),
  reusing `lower_bound/upper_bound`, the multiblock scan, and
  mask-compaction kernels. Frontier buffer = standard `CudaBuffer`
  (SoA u32 columns), allocated per node via the two-phase counts.

## 4. Phase plan with gates (all phases in scope; nothing deferred)

| Phase | Deliverable | Gate |
|---|---|---|
| A — S2 spike | Provider-level frontier engine (u32) executing hand-built plans for: 4-atom chain `Q(a,x,y,z,b): R(a,x),S(x,y),T(y,z),U(z,b)` with a blowup fixture; star/clover; triangle | ≥2× vs the production binary-join path on the blowup fixture; ≤1.2× vs the dedicated triangle WCOJ kernel on the W5.2 triangle fixture; row-set parity vs binary on ALL fixtures |
| B — production wiring | RIR node + binary2fj planner + optimizations + executor dispatch + fallback + counters/kill switch + e2e tests through real Datalog source (incl. kill-switch parity, decline cases, 5-atom bodies) | full 4-crate suites green; e2e parity on ≥6 shapes |
| C — completion | u64 keys; recursive-SCC integration (FreeJoin node inside `execute_wcoj_or_fallback_node` with delta-rewritten scans); aggregate fusion over the final frontier (count-by-root incl. factorized trailing-range counting); docs/evidence/changelog | parity vs unfused in recursion (TC-style fixtures); fused count gate ≥3× on a skewed ≥4-atom fixture; full regression 0 failures |

Phase A failing its gate parks D2 with evidence (branch preserved,
research doc updated) — that is the only acceptable "deferral", and it
is a measured negative result, not debt.

## 5. Risks pinned during design

1. **Frontier row width** grows with atoms (2 range cols/atom): bounded
   by dropping ranges of exhausted atoms (subatoms fully consumed) —
   the planner computes each node's live-range set statically.
2. **Skewed expansion** (one row → 10⁶ children): two-phase sizing makes
   it correct; load balance reuses the block-work-unit slicing pattern;
   the W33 lesson (fine-grained per-block slicing) applies if measured.
3. **Probe-key gather cost**: probe keys are frontier columns — coalesced
   reads; binary search per row mirrors the proven triangle inner loop.
4. **Planner regressions**: Free Join only dispatches where the dedicated
   shape kernels don't, and only ≥3 atoms — existing paths untouched;
   the embedded fallback guarantees behavioral equivalence on decline.
