# Phase 4 Integrated Design (xlog-prob + P4.1–P4.4 + Python `xlog-gpu`)

**Date:** 2026-01-13  
**Status:** Approved (interactive requirements lock)  
**Targets:** Linux x86_64 + CUDA-only  

This document captures the **integrated Phase 4** design: deliver `xlog-prob` (probabilistic + differentiable reasoning) while completing/solidifying the Phase 4 substrate items in `docs/ROADMAP.md` (CuDF/Arrow/DLPack interop, optimizer, incremental maintenance, adaptive indexing) and shipping a user-visible Python package.

---

## 1) Locked Decisions

### 1.1 Packaging / Platform
- **Python package name:** `xlog-gpu` (PyPI `xlog` is already taken).
- **Packaging:** PyO3 + `maturin` wheels.
- **Torch:** optional dependency; primary interop uses **DLPack capsules** (framework-agnostic).
- **Platform:** Linux x86_64 + CUDA-only (Phase 4).

### 1.2 Knowledge compilation backend
- **D4 is vendored** in-repo and built as part of the workspace/tooling.
- No external system dependency on a preinstalled `d4` binary.

### 1.3 Exactness / Tier contracts (xlog-prob)
- **P1/P2 exact** for programs that can be compiled into tractable decomposable circuits (Decision-DNNF) and evaluated on GPU.
- **Full recursion is supported**, including probabilistic heads in recursive SCCs, via a fixpoint-aware provenance construction (see §4).
- **Non-monotone recursion** (recursion through `not` and/or aggregates) is permitted syntactically, but:
  - **Default:** compilation error with a clear diagnostic and remediation.
  - **Only if user explicitly requests P3** (via `#pragma prob_engine = mc` or CLI `--prob-engine mc`, with CLI overriding pragma): approximate execution is allowed.

### 1.4 Controls / UX
- **P3 selection:** both `#pragma` in source and CLI flag exist; **CLI overrides pragma**.

---

## 2) Phase 4 Outcomes (User-Visible)

### 2.1 Rust API
- `xlog_logic` remains the deterministic compiler for the Datalog fragment and is used for grounding/slicing.
- New `xlog_prob` API provides:
  - compile probabilistic programs (or probabilistic slices)
  - maintain caches for “compile once, evaluate many”
  - evaluate queries under evidence, return probabilities/log-probabilities
  - compute gradients w.r.t. leaf weights (prob facts + neural tables)

### 2.2 Python API (package: `xlog-gpu`)
MVP surface (subject to naming refinement):
- `xlog_gpu.Program.compile(source: str, *, device=0, memory_mb=..., prob_engine="exact|mc", ...) -> CompiledProgram`
- `CompiledProgram.evaluate(queries=[...], evidence=[...], *, return_grads=False, dlpack_inputs={...}) -> Result`
- All GPU data interchange is via **DLPack**:
  - inputs accepted as DLPack capsule objects (or objects exposing `__dlpack__`)
  - outputs returned as DLPack capsules

---

## 3) Architecture Overview (Integrated)

```
    .xlog source (logic + prob)               Python (optional)
               │                                   │
               ▼                                   │
    xlog-logic parse/stratify/lower/optimize       │
               │                                   │
               ├─ deterministic execution (GPU) ───┘  (DLPack GPU tables in/out)
               │
               ▼
        grounding + query slicing
               │
               ▼
           xlog-prob
    (PIR → (W)CNF → D4 → XGCF)
               │
               ▼
     GPU circuit eval + autodiff
               │
               ▼
        query probs + gradients
```

**Key integration point:** P4.1–P4.4 are not “side quests”; they are the substrate that keeps Phase 4 feasible:
- stats + optimizer reduce slice size and memory peaks
- incremental maintenance supports repeated evaluation with changing evidence/weights
- adaptive indexing helps repeated scans/joins during grounding
- DLPack/Arrow interop makes Python viable without host roundtrips

---

## 4) Semantics: Distribution Semantics + Fixpoint

### 4.1 World semantics
Probabilistic facts/choices induce a distribution over “worlds”. In each world:
- the deterministic rules are evaluated to a **least fixpoint** (recursive Datalog semantics)
- the query is true/false based on whether the derived fact is in the fixpoint result

### 4.2 Recursion inside xlog-prob exact path
To keep knowledge compilation feasible, Phase 4 uses a **fixpoint-aware provenance construction**:
- Evaluate the relevant grounded slice semi-naively (SCC-by-SCC).
- For each derived tuple, maintain a **formula** describing when it is derived.
- Ensure formulas form a DAG by constructing them in iteration order (semi-naive unrolling):
  - iteration `k` formulas only depend on tuples from iterations `< k`
  - stop when no new tuples appear (fixpoint)

This yields **acyclic PIR** even for recursive programs, at the cost of potentially large provenance (budgeted and cached).

---

## 5) IRs: PIR + XGCF

### 5.1 PIR (Provenance IR)
Represent provenance as a semiring-aware Boolean circuit skeleton:
- `PIR_Lit(literal_id, weight_source)`
- `PIR_And(children[])`
- `PIR_Or(children[])`
- `PIR_Decision(var_id, child_false, child_true)` (Decision-DNNF shape)

PIR must support:
- deterministic rule OR-of-bodies and AND-of-literals
- probabilistic facts and annotated disjunction choices
- evidence variables (as leaf indicators)
- recursion unrolling (iteration-indexed nodes)

### 5.2 CNF/WCNF emission
Use Tseitin encoding:
- introduce CNF variables for PIR nodes
- add equivalence constraints
- attach weights only to probabilistic choice variables (and evidence indicators if modeled as variables)

### 5.3 D4 output ingestion
D4 produces Decision-DNNF. Phase 4 requires:
- a stable parser/adapter to map D4 nodes/vars into XLOG ids
- a deterministic mapping from PIR/CNF vars → D4 vars → XGCF leaf ids

### 5.4 XGCF (XLOG GPU Circuit Format)
GPU-evaluable SoA representation, evaluated in log-space:
- `node_type[i] ∈ {CONST0, CONST1, LIT, AND, OR, DECISION}`
- children via `a[i], b[i]` and/or `child_offsets` + `child_indices` for variadic nodes
- `level_offsets[]` for topological “levels”
- `value[i]` (log-space), `adj[i]` (reverse-mode)

Batching:
- multiple queries can share a single compiled circuit via multiple roots
- or multiple circuits can be concatenated with `circuit_offsets`

---

## 6) Evaluation & Autodiff

### 6.1 Forward evaluation (log-space)
- LIT: `v = log(w_lit)` with evidence mask
- AND: `v = Σ v_child`
- OR: `v = logsumexp(v_children)`
- DECISION: `v = logsumexp(log(p(var=1)) + v_true, log(p(var=0)) + v_false)`

### 6.2 Backward (reverse-mode)
Accumulate gradients:
- AND: propagate `adj[parent]` to all children
- OR: softmax-weighted propagation
- DECISION: split gradients to children and to decision probability parameters
- LIT: accumulate `dL/dlog(w_lit)` and export leaf grads

### 6.3 Query probabilities under evidence
MVP approach for conditional probabilities:
- compute `log Z_E = log WMC(E)`
- compute `log Z_EQ = log WMC(E ∧ Q)`
- return `log P(Q|E) = log Z_EQ − log Z_E`

---

## 7) Approximate Tier (P3) — explicit opt-in

### 7.1 Trigger conditions
P3 is allowed only when explicitly requested:
- `#pragma prob_engine = mc` (source), or
- CLI `--prob-engine mc` (CLI overrides pragma)

### 7.2 Scope
P3 is the escape hatch for:
- non-monotone recursion (recursion through `not`/aggregates)
- circuit compilation infeasible under the configured budgets

### 7.3 Semantics & reporting (robustness requirement)
P3 results must be **explicitly labeled approximate** and include uncertainty:
- report `(estimate, stderr/confidence interval, samples, seed)`
- for non-monotone programs, the engine must be explicit about the semantics used for the inner evaluation
  (Phase 4 begins with an honest, bounded/unknown-aware approach; exact stable-model probability semantics is deferred).

---

## 8) Interop (P4.1) + Python Binding Strategy

### 8.1 Rust interop status (already present)
`crates/xlog-cuda` already provides:
- Arrow RecordBatch + Arrow IPC helpers (copying)
- DLPack table import/export (zero-copy per column)

### 8.2 Python boundary (Phase 4)
- The Python extension accepts/returns DLPack capsules and performs:
  - schema validation
  - lifetime management
  - safe error conversion (Rust errors → Python exceptions)

---

## 9) Caching, Incrementality, and Stats (P4.2–P4.4)

Phase 4 requires a coherent caching story:
- cache compiled artifacts by content hash (program slice + settings):
  - grounded slice / relevant relations
  - PIR graph
  - CNF/WCNF + var maps
  - D4 Decision-DNNF output (optional)
  - XGCF GPU buffers (primary)
- evidence/weight updates should not invalidate XGCF unless the slice changes

P4.2–P4.4 are leveraged to keep slice and GPU costs bounded:
- optimizer: predicate pushdown + join ordering reduces grounding work
- stats snapshot feedback: use runtime observations to guide lowering/plans
- adaptive indexing: reuse join indexes for hot relations during repeated evaluation

---

## 10) Non-Goals (Phase 4)

- Cross-platform packaging (macOS/Windows) — deferred.
- A full internal Torch runtime — deferred (external neural via DLPack ships first).
- Exact stable-model probability semantics for non-monotone recursion — deferred (requires deeper solver integration).

