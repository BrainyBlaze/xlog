# Phase 4 Integrated Implementation Plan (xlog-prob + P4.1–P4.4 + Python `xlog-gpu`)

**Date:** 2026-01-13  
**Status:** Completed on `phase4-integrated`  
**Targets:** Linux x86_64 + CUDA-only  

This plan executes the design in `docs/plans/2026-01-13-phase4-integrated-design.md` and updates the Phase 4 roadmap items (P4.1–P4.4) as substrate, not as separate projects.

## Execution Summary (2026-01-14)

All tasks in this plan are implemented on branch `phase4-integrated`, including the P3 Monte Carlo engine (Task 7) and Python API wiring.

**Key artifacts:**
- MC engine: `crates/xlog-prob/src/mc.rs` + sampler kernel `kernels/mc_sample.ptx`
- MC tests: `crates/xlog-prob/tests/mc.rs` and CUDA certification coverage under `crates/xlog-cuda-tests/`
- Python API: `crates/xlog-gpu-py/src/lib.rs` (`prob_engine="exact_ddnnf"|"mc"`)
- Python examples: `examples/python/03_prob_mc_nonmonotone_torch.py`, probabilistic `.xlog` examples under `examples/prob/`

---

## 0) Acceptance Criteria (Phase 4 “Done”)

### 0.1 Deterministic logic (regression)
- `cargo test --workspace --all-targets` passes in `debug` and `release`.
- CUDA certification suite remains green (no kernel regressions).

### 0.2 xlog-prob (exact)
- Can parse + run minimal probabilistic programs:
  - probabilistic facts (`0.7::rain().`)
  - annotated disjunctions (`0.6::coin(heads); 0.4::coin(tails).`)
  - evidence + queries (`evidence/2`, `query/1`)
- Exact path: `P(Q|E)` via `log Z(E∧Q) - log Z(E)` using Decision-DNNF + GPU WMC.
- Supports recursion by provenance unrolling to an acyclic PIR (semi-naive iteration order).
- Supports gradients w.r.t. leaf log-weights (prob facts + neural tables) via reverse-mode on GPU circuit.

### 0.3 xlog-prob (approx / P3)
- Non-monotone recursion is:
  - **error by default**, with a diagnostic that says “requires P3 (`prob_engine=mc`)”
  - allowed only when explicitly requested via `#pragma prob_engine = mc` or an explicit API override (e.g., Python `Program.compile(..., prob_engine="mc")`)
- Approx results include uncertainty metadata (samples/seed + stderr/CI).

### 0.4 Python distribution (`xlog-gpu`)
- Built wheels via `maturin` containing a `xlog_gpu` importable module.
- Torch remains optional; primary API uses DLPack:
  - accepts DLPack capsules / `__dlpack__` producers
  - returns DLPack capsules (framework-agnostic)
- Ships a minimal end-to-end Python example that:
  - loads a `.xlog` program
  - ingests GPU tables via DLPack
  - evaluates queries and returns result tables as DLPack

---

## 1) Repo/Build Scaffolding

### Task 1.1: Add new crates + top-level layout
**Create:**
- `crates/xlog-prob/` (core probabilistic engine)
- `crates/xlog-gpu-py/` (PyO3 extension for `xlog-gpu`)

**Modify:**
- `Cargo.toml` (workspace members, workspace deps)

**Notes:**
- Keep CUDA kernels for circuits in `kernels/` alongside existing modules; load via `crates/xlog-cuda/src/provider.rs`.

### Task 1.2: Vendor D4 in-repo (build as part of workspace)
**Create (example layout):**
- `vendor/d4/` (pinned upstream snapshot)
- `crates/xlog-prob/build.rs` to build `vendor/d4` into `target/` (or a deterministic `target/d4/` subdir)

**Deliverable:**
- `xlog-prob` can invoke a known-path `d4` binary without requiring a system install.

**Validation:**
- `cargo build -p xlog-prob` builds D4 as needed.

---

## 2) Language Surface (Parser/AST) for Probabilistic Profile

### Task 2.1: Extend grammar for probabilistic facts / annotated disjunction
**Modify:**
- `crates/xlog-logic/src/grammar.pest`
- `crates/xlog-logic/src/parser.rs`
- `crates/xlog-logic/src/ast.rs`

**Add parse coverage for:**
- `0.7::rain().`
- `0.2::edge(1,2).`
- `0.6::coin(heads); 0.4::coin(tails).`
- `evidence(rain(), true).`
- `query(reach(1,3)).`
- `#pragma prob_engine = exact_ddnnf|mc` (and optional cache directives)

**Tests:**
- Add focused parser tests under `crates/xlog-logic/src/parser.rs` (and/or a new `crates/xlog-logic/tests/` module) for each construct.

### Task 2.2: Surface engine selection + gating
**Modify:**
- `crates/xlog-logic/src/stratify.rs` (and any validation pass)

**Behavior:**
- Detect recursion through negation/aggregates (“non-monotone recursion”).
- If `prob_engine != mc`: return an error message that explicitly says P3 is required.
- If `prob_engine == mc`: allow compilation to proceed (execution semantics handled in `xlog-prob`).

---

## 3) Provenance Extraction → PIR

### Task 3.1: Define PIR data model
**Create:**
- `crates/xlog-prob/src/pir.rs`

**Requirements:**
- Node types: `Lit`, `And`, `Or`, `Decision` (+ consts)
- Stable id mapping for:
  - probabilistic fact variables
  - AD choice variables
  - evidence indicators (or evidence as leaf masks)
- Topological order / levelization support for later XGCF lowering.

### Task 3.2: Build provenance during semi-naive evaluation
**Modify / Create (one viable factoring):**
- `crates/xlog-runtime/` to expose a provenance-capable execution mode, or
- implement a parallel “provenance executor” inside `crates/xlog-prob/` that reuses `xlog-logic` lowering.

**Behavior:**
- Evaluate SCC-by-SCC semi-naively.
- For each derived tuple, store a PIR expression describing derivation.
- Ensure acyclicity for recursive programs by iteration-indexed construction (unrolling).

**Validation:**
- Unit tests for simple recursive provenance (e.g., transitive closure) that check PIR DAG invariants.

---

## 4) PIR → (W)CNF + Var Mapping

### Task 4.1: Tseitin encoding
**Create:**
- `crates/xlog-prob/src/cnf.rs`

**Deliverables:**
- Emit CNF/WCNF in a D4-compatible format.
- Persist a var-map (`pir_node_id` → `cnf_var_id`, plus leaf weight map).

**Tests:**
- Golden tests for tiny PIR graphs that validate:
  - satisfiable assignments match expected derivations
  - weight map matches leaf ids

---

## 5) Knowledge Compilation (D4) → Circuit Ingestion

### Task 5.1: Run D4 and parse Decision-DNNF output
**Create:**
- `crates/xlog-prob/src/kc/d4.rs` (backend wrapper)
- `crates/xlog-prob/src/kc/ddnnf.rs` (parser + in-memory circuit)

**Requirements:**
- Deterministic mapping from CNF vars → D4 vars → circuit literals.
- Durable cache keys: content hash of (slice + cnf + settings) → compiled ddnnf artifact.

---

## 6) Circuit Lowering → XGCF + GPU Eval/Autodiff

### Task 6.1: Define XGCF format + host-side builder
**Create:**
- `crates/xlog-prob/src/xgcf.rs`

**Deliverables:**
- SoA arrays for node types and child indices
- `level_offsets[]` for level-by-level GPU evaluation
- support multiple roots (queries) over a shared circuit

### Task 6.2: Add CUDA kernels for circuit forward/backward
**Create/Modify:**
- `kernels/circuit.cu` (new PTX module)
- `crates/xlog-cuda/src/provider.rs` (load module, expose entrypoints)
- `crates/xlog-prob/src/gpu.rs` (safe wrappers around provider calls)

**Kernels:**
- forward: per-level evaluation (log-space) for `AND`, `OR` (logsumexp), `DECISION`, `LIT`
- backward: reverse-mode accumulation, leaf gradient export

**Validation:**
- CPU reference evaluator for small circuits + GPU parity tests (small N).

---

## 7) P3 Monte Carlo Engine (Explicit Opt-In)

### Task 7.1: Implement MC driver with uncertainty reporting
**Create:**
- `crates/xlog-prob/src/mc.rs`

**Behavior:**
- Sample probabilistic facts/choices on GPU.
- For each sample, evaluate deterministic core (and/or a semantics-restricted probabilistic evaluation).
- Return `(estimate, stderr/CI, samples, seed)` and mark results “approx”.

**Notes:**
- For non-monotone recursion, the engine must explicitly document the semantics used (bounded/unknown-aware is acceptable for Phase 4, but must be honest).

---

## 8) Python Package: `xlog-gpu` (PyO3 + maturin, DLPack-first)

### Task 8.1: Create PyO3 extension crate and pyproject
**Create (suggested):**
- `crates/xlog-gpu-py/Cargo.toml` (`crate-type = ["cdylib"]`)
- `crates/xlog-gpu-py/pyproject.toml` (maturin)
- `crates/xlog-gpu-py/src/lib.rs` (PyO3 module `xlog_gpu`)

**Expose:**
- `Program.compile(source: str, ..., prob_engine=...)`
- `CompiledProgram.evaluate(..., dlpack_inputs=..., return_grads=...)`

### Task 8.2: DLPack boundary
**Use:**
- `crates/xlog-cuda/src/dlpack.rs` for import/export.

**Requirements:**
- Accept either:
  - a raw DLPack capsule (`PyCapsule`), or
  - an object with `__dlpack__` (call it to obtain a capsule)
- Return DLPack capsules for output columns (and optional gradient tables).

### Task 8.3: Optional torch helpers (non-default)
**Approach:**
- Provide examples (not hard dependency) using `torch.utils.dlpack.{to_dlpack, from_dlpack}`.

---

## 9) Docs + Examples + CI

### Task 9.1: Update docs to match locked decisions
**Modify:**
- `docs/ROADMAP.md` (Phase 4 integrated plan pointers; `xlog-gpu`; vendored D4; P3 gating)
- `docs/architecture/cudf-interop.md` (Python package name + DLPack-first guidance)

### Task 9.2: Add end-to-end examples
**Create:**
- `examples/prob/` (tiny programs + expected results)
- `examples/python/` (`xlog-gpu` DLPack demo: table in/out)

### Task 9.3: CI hooks (Linux + CUDA)
**Add:**
- a CI job that builds the Python wheel (at least `maturin build --release`)
- smoke-test import + a tiny evaluate call (if CI runner has GPU; otherwise run CPU-only import tests)

---

## 10) Execution Order (Recommended)

1. Tasks 1–2 (scaffolding + parsing + gating)
2. Tasks 3–6 (exact pipeline end-to-end for tiny programs)
3. Task 8 (Python layer wired to exact path)
4. Task 7 (P3 MC engine)
5. Task 9 (docs/examples/CI hardening)
