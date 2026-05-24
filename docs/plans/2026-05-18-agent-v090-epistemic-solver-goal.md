# Agent Goal 041 - v0.9.0 Epistemic And Solver Semantics

**Agent:** Agent B, v0.9.0 forward worker.
**Branch:** `feat/v090-epistemic-solver-semantics`.
**Worktree:** `.worktrees/v090-epistemic`.
**Base:** `main` at or after `656a8c62` (`docs(roadmap): focus v080 on dts ml python productization`).
**Integration order:** v0.9.0 work may proceed in parallel, but it must not merge before v0.8.0. After v0.8.0 lands, rebase or merge `main` and rerun all compatibility gates.
**Status:** Dispatch-ready goal document with GPU-native production-path scope amendment. Implementation begins only after the worktree is created and baseline status is recorded. CPU-only or fixture-only epistemic execution is not closeable for v0.9.0.

## 0. Process Model

This goal uses the requested Goal-Driven Software Development Process and GQM framing.

Sources read:

- Goal-Driven Software Development Process, Wikipedia, accessed 2026-05-18:
  https://en.wikipedia.org/wiki/Goal-Driven_Software_Development_Process
- GQM, Wikipedia, accessed 2026-05-18:
  https://en.wikipedia.org/wiki/GQM

Method adaptation:

- GDSP starts from goals, then lets technical feasibility refine the goal. This agent must first establish semantic goals and only then select parser, IR, solver, and runtime mechanisms.
- GDSP emphasizes top-down goals plus bottom-up implementation convergence. This agent must preserve a trace from epistemic business goals to xlog-logic/xlog-ir/xlog-solve design choices.
- GDSP favors small goal sets. v0.9.0 is a forward branch; if the semantic scope expands beyond this document, split the work instead of widening it silently.
- GQM supplies the measurement structure: conceptual goal, operational questions, quantitative metrics, then interpretation.

## 1. Business Goal

**BG090.** Ship xlog v0.9.0 as the epistemic and solver-semantics release train, adding EIR, epistemic reasoning modes, Generate-Propagate-Test execution, and solver-service integration without regressing the v0.8.0 DTS-DLM ML/Python substrate.

The release is successful only if:

- epistemic semantics are represented explicitly rather than hidden in ad hoc rewrites;
- world views are the semantic boundary object for epistemic evaluation;
- compatibility and default semantics are separately testable;
- self-supported epistemic conclusions are allowed only in the explicit G91 compatibility mode, not in the default founded mode;
- accepted epistemic execution is fully GPU-native after parsing/planning;
- epistemic programs route through production lowering/runtime dispatch and the WCOJ/GPU path where eligible;
- nonzero-arity epistemic model-membership checks run on GPU over existing relation layouts and tuple buffers, not row-count-only or CPU tuple scans;
- solver services integrate through existing GPU-native solver production paths for assumptions, learned clauses, MaxSAT, and portfolio execution;
- probabilistic integration remains coherent on the existing GPU-native exact/provenance path;
- v0.9.0 reuses the existing WCOJ, solver, and probabilistic engines instead of creating parallel epistemic-specific engines;
- v0.8.0 pyxlog/DTS-DLM compatibility remains green after rebasing.

## 1.1 Semantic Contract

Epistemic logic programming extends ordinary logic programming with modal reasoning over sets of stable models. Ordinary rules reason about truth in one candidate model; epistemic rules reason about whether a predicate is known across all accepted models, possible in at least one accepted model, or not known across the current world view.

The v0.9.0 semantic boundary is:

```
world view W = stable models of the program reduced using epistemic facts derived from W
```

This self-reference is the reason v0.9.0 must make the phases explicit:

1. Generate candidate epistemic assumptions.
2. Propagate those assumptions into a reduced ordinary program.
3. Test the stable models of the reduced program against the original epistemic assumptions.
4. Accept only candidates whose world-view checks hold.

The implementation principle is:

```
represent epistemic semantics explicitly first;
lower to ordinary runtime, probabilistic, or solver machinery only after the semantic boundary is proven.
```

Agent B must preserve two semantics families:

- **G91 compatibility mode:** classic Gelfond 1991-style compatibility semantics, useful for legacy behavior and fixtures that intentionally allow compatibility-style self-support.
- **FAEEL default mode:** founded autoepistemic equilibrium-style semantics, rejecting circular epistemic support unless it is independently founded.

The solver-level contract is incremental and status-aware. Each epistemic candidate may push assumptions, solve, check world-view consistency, retract assumptions, and reuse learned clauses only when reuse is semantically valid. Solver outcomes must distinguish SAT, UNSAT, UNKNOWN, and TIMEOUT.

## 1.2 GPU-Native Requirement

v0.9.0 must be a fully GPU-native epistemic release. Bounded CPU fixtures are allowed only as development scaffolding and semantic oracle tests; they do not satisfy closure gates.

The accepted v0.9.0 target is:

- EIR parses and represents epistemic constructs explicitly.
- Accepted epistemic programs lower into production executable IR, not a fixture-only side path.
- Runtime dispatch launches GPU kernels for epistemic candidate generation, propagation, world-view validation, and accepted result materialization.
- WCOJ planner eligibility, layout construction, scheduling, and helper-splitting decisions apply to epistemic reductions where the reduced ordinary program is WCOJ-eligible.
- Candidate assumptions, world-view bitsets, model-membership checks, and rejection reasons are represented in GPU-resident buffers during the hot path.
- Nonzero-arity model-membership checks compare stable-model tuple keys on GPU using the existing relation column/layout machinery, sorted labels, and device row buffers. Zero-arity row-count checks alone are not sufficient for closure.
- SAT/MaxSAT assumptions, learned-clause transfer, and portfolio solving run through existing GPU-native solver services or documented adapters into those services.
- Probabilistic evidence from accepted world views flows into the existing GPU-native exact/provenance path without CPU-only recomputation in the accepted execution path.
- Solver and probabilistic fixture modules may remain only as semantic-oracle scaffolding. Accepted release paths must call the existing production solver/probability cores.

Allowed CPU responsibilities are parsing, static planning, launch orchestration, diagnostics formatting, and final result transfer. CPU fallback for candidate enumeration, nonzero-arity tuple membership, world-view validation, SAT/MaxSAT search, or probabilistic recomputation is a blocker unless the fallback is limited to a negative test or a semantic oracle that is not used by accepted release paths.

Existing non-epistemic programs must continue to use the normal parser, stratifier, RIR lowering, runtime, probabilistic, and WCOJ paths where eligible. Agent B must extend that production path for epistemic execution without weakening the non-epistemic path.

Current-branch correction: an implementation that only has EIR, semantic fixtures, CPU-side solver enumeration, and probabilistic evidence fixtures is not complete. It may be preserved as semantic-oracle evidence, but it cannot close G090_GPU, G090_SOLVER, G090_PROB, G090_CERT, or G090_CLOSE until the GPU-native path is implemented and measured.

### 1.3 Production-Path Reuse Locks

The following reuse locks are part of the v0.9.0 acceptance contract:

- **Runtime/WCOJ reuse.** Epistemic reductions must compile into existing RIR and dispatch through existing runtime, WCOJ, K-clique, helper-split, runtime-histogram, and cost-gated routing machinery where eligible. A separate epistemic WCOJ planner, relation store, or dispatch engine is out of scope.
- **Solver reuse.** Accepted SAT, MaxSAT, assumptions, learned-clause, and portfolio execution must be implemented by wiring epistemic candidates into existing `xlog-solve` GPU CNF/CDCL/solver production paths or thin adapters over those paths. CPU exhaustive assignment enumeration is semantic-oracle evidence only.
- **Probabilistic reuse.** Accepted world-view evidence must flow into existing `xlog-prob` GPU exact/provenance/PIR/knowledge-compilation production paths. A bounded epistemic probability circuit may be kept only as a fixture oracle and must not be the accepted execution path.
- **Tuple-membership reuse.** Nonzero-arity stable-model membership must be checked on GPU against existing relation layouts, device columns, tuple-key encodings, sorted labels, and materialized row buffers. Row-count-only membership is valid only for zero-arity predicates or negative fixtures.
- **Evidence reuse.** Certification must include counters or traces proving the above production paths executed, and a source audit proving no parallel epistemic-only solver, probability, WCOJ, or tuple-store engine was introduced.

## 2. Scope Boundaries

### In Scope

- Epistemic Intermediate Representation (EIR).
- G91 compatibility mode.
- FAEEL default mode.
- Generate-Propagate-Test execution.
- Epistemic splitting.
- Full GPU-native epistemic runtime execution.
- WCOJ-backed epistemic reductions where the reduced ordinary program is eligible.
- GPU-native nonzero-arity stable-model tuple membership using existing relation layouts and device buffers.
- GPU-resident world-view, candidate, and rejection buffers.
- Integration of epistemic reasoning with probabilistic inference.
- Solver-service integration with xlog-logic constraints.
- Incremental SAT semantics, assumptions, learned-clause transfer.
- MaxSAT with soft constraints.
- GPU portfolio SAT/MaxSAT dispatch.
- Incremental circuit updates and alternative knowledge compiler adapters.
- Epistemic semantics guide and solver-semantics certification tests.

### Out Of Scope

- v0.8.0 DTS-DLM ML/Python implementation work.
- pyxlog public API changes unless explicitly coordinated with Agent A.
- relation delta session APIs, M37-A+B bridge helpers, native exact-induction consumer integration.
- WCOJ kernel rewrites or CUDA Graph changes unless a v0.9 semantic test proves a correctness blocker.
- Reimplementation of WCOJ/K-clique planning, helper splitting, runtime histograms, relation storage, tuple membership storage, solver search, or probabilistic inference as epistemic-only parallel engines.
- CPU-only accepted execution for epistemic candidate generation, world-view validation, SAT/MaxSAT search, or probabilistic recomputation.
- Row-count-only model-membership checks for nonzero-arity predicates in accepted execution.
- Fixture-only epistemic semantics as a release substitute.
- Any push, tag, release-board update, or merge to `main` without explicit coordinator authorization.

### Coordination Locks

- Agent B may prototype in parallel but must not land before Agent A's v0.8.0 branch.
- Do not edit files owned by active v0.8.0 work unless the coordinator approves a shared interface change.
- Keep compatibility tests runnable after v0.8.0 rebase.
- Any pyxlog-facing changes must be additive and default-off until v0.8.0 compatibility has been revalidated.
- Treat semantic changes as correctness-sensitive. A green compile is not enough.

## 3. Roadmap Mapping

| ROADMAP area | v0.9.0 goal node | Agent responsibility |
|---|---|---|
| xlog-logic | G090_EIR, G090_G91, G090_FAEEL, G090_GPT, G090_SPLIT | Parser/logic semantics and IR mapping |
| Runtime/WCOJ/GPU | G090_GPU | production lowering, GPU-resident world-view execution, WCOJ-backed reductions, nonzero-arity tuple membership over existing layouts |
| Solver Services | G090_SOLVER | GPU-native SAT assumptions, incremental solving, learned clauses, MaxSAT, portfolio dispatch through existing solver core |
| Probabilistic Reasoning | G090_PROB | accepted world-view evidence on existing GPU-native exact/provenance path |
| Documentation and Tests | G090_CERT, G090_DOC | Golden semantic-oracle fixtures, GPU certs, solver certs, guide |

## 4. Goal Hierarchy

```
BG090 - xlog v0.9.0 epistemic and solver semantics
 |
 +-- G090_PRE       Baseline inventory and semantic fixture selection
 +-- G090_EIR       Epistemic Intermediate Representation
 +-- G090_G91       G91 compatibility mode
 +-- G090_FAEEL     FAEEL default semantics
 +-- G090_GPT       Generate-Propagate-Test execution
 +-- G090_SPLIT     Epistemic splitting
 +-- G090_GPU       GPU-native runtime and WCOJ execution
 +-- G090_SOLVER    Solver-service integration
 +-- G090_PROB      Probabilistic and circuit integration
 +-- G090_CERT      Certification and regression gates
 +-- G090_DOC       User and architecture documentation
 +-- G090_CLOSE     Closure proposal after v0.8.0 rebase
```

## 5. GQM Decomposition

### G090_PRE - Baseline Inventory And Semantic Fixture Selection

**Goal.** Establish a clean v0.9.0 forward worktree plus semantic-oracle and GPU-certification fixtures for the purpose of implementing epistemic/solver semantics without disrupting v0.8.0.

**Questions.**

- Q090_PRE.1: Is the worktree based on the intended commit?
- Q090_PRE.2: Which crates own parsing, logic semantics, IR, probabilistic inference, and solver services?
- Q090_PRE.3: What golden examples define G91, FAEEL, splitting, and Generate-Propagate-Test behavior?
- Q090_PRE.4: Which v0.8.0 compatibility checks must be rerun after rebase?
- Q090_PRE.5: Which examples require multiple stable models and therefore exercise world-view semantics rather than ordinary model semantics?

**Metrics.**

| Metric | Target |
|---|---|
| M090_PRE.1 branch base | base recorded and clean before implementation |
| M090_PRE.2 ownership map | crate/file ownership table committed in evidence |
| M090_PRE.3 fixture inventory | at least one positive and one negative fixture for each semantic mode |
| M090_PRE.4 compatibility list | v0.8-owned tests to rerun after rebase listed |
| M090_PRE.5 world-view fixtures | at least two fixtures with multiple stable models and explicit know/possible/not-known expectations |

**Evidence.** `docs/evidence/<date>-v090-pre/README.md`.

### G090_EIR - Epistemic Intermediate Representation

**Goal.** Add EIR for the purpose of representing epistemic constructs explicitly, with respect to parser output, semantic analysis, and GPU-native lowering boundaries.

**Questions.**

- Q090_EIR.1: What syntax and AST nodes represent epistemic operators?
- Q090_EIR.2: How does EIR relate to existing RIR and probabilistic IR?
- Q090_EIR.3: Can unsupported constructs fail with typed diagnostics instead of silent fallback?
- Q090_EIR.4: Where is the proven boundary for lowering EIR into GPU-native runtime, probabilistic, or solver machinery?

**Metrics.**

| Metric | Target |
|---|---|
| M090_EIR.1 AST/EIR nodes | explicit representation committed |
| M090_EIR.2 parser tests | positive and negative syntax fixtures pass |
| M090_EIR.3 lowering boundary | EIR-to-GPU-executable boundary documented |
| M090_EIR.4 diagnostics | unsupported constructs return typed errors |
| M090_EIR.5 explicit operators | `know`, `possible`, and `not know` equivalents represented without ad hoc string rewrites |
| M090_EIR.6 production route | accepted epistemic forms have a production lowering route; rejected forms are explicit |
| M090_EIR.7 no DTS regression | v0.8 pyxlog compatibility tests still pass or are not touched |

**Expected targets.**

- `crates/xlog-logic/`
- `crates/xlog-ir/`
- semantic docs under `docs/architecture/`
- tests under `crates/xlog-logic/tests/` or equivalent

### G090_G91 - G91 Compatibility Mode

**Goal.** Implement G91 as a compatibility mode for the purpose of supporting classic epistemic logic behavior while keeping default semantics separate.

**Questions.**

- Q090_G91.1: How is G91 selected?
- Q090_G91.2: Which fixtures distinguish G91 from FAEEL?
- Q090_G91.3: Does compatibility mode avoid changing default behavior?
- Q090_G91.4: Which fixtures document compatibility-style self-support that is intentionally not accepted by FAEEL?

**Metrics.**

| Metric | Target |
|---|---|
| M090_G91.1 mode selection | explicit config, flag, or source annotation |
| M090_G91.2 golden fixtures | 100 percent pass |
| M090_G91.3 mode isolation | default mode output unchanged on non-epistemic fixtures |
| M090_G91.4 docs | compatibility behavior documented |
| M090_G91.5 G91-only cases | at least one fixture accepted by G91 and rejected by FAEEL |

### G090_FAEEL - FAEEL Default Semantics

**Goal.** Implement FAEEL as the default epistemic semantics for the purpose of giving xlog a founded autoepistemic equilibrium mode with testable behavior.

**Questions.**

- Q090_FAEEL.1: What is the minimal executable core for FAEEL in v0.9.0?
- Q090_FAEEL.2: What examples prove foundedness and equilibrium behavior?
- Q090_FAEEL.3: How are contradictions or no-model cases represented?
- Q090_FAEEL.4: How does the implementation reject circular epistemic support?

**Metrics.**

| Metric | Target |
|---|---|
| M090_FAEEL.1 core semantics | minimal core implemented and documented |
| M090_FAEEL.2 golden fixtures | 100 percent pass |
| M090_FAEEL.3 no-model behavior | typed result or diagnostic, not panic |
| M090_FAEEL.4 G91 distinction | fixtures show at least one intentional G91/FAEEL difference |
| M090_FAEEL.5 foundedness guard | self-supported epistemic fixture rejected with documented reason |

### G090_GPT - Generate-Propagate-Test Execution

**Goal.** Add Generate-Propagate-Test execution for the purpose of supporting epistemic candidate generation and constraint filtering in a controlled pipeline.

**Questions.**

- Q090_GPT.1: What is generated, what is propagated, and what is tested?
- Q090_GPT.2: Which phases can reuse existing runtime/probabilistic/solver machinery?
- Q090_GPT.3: How does the execution trace remain inspectable?
- Q090_GPT.4: Does the test phase verify guessed `know` facts against all stable models and guessed `possible` facts against at least one stable model?

**Metrics.**

| Metric | Target |
|---|---|
| M090_GPT.1 phase separation | generate, propagate, test boundaries visible in code |
| M090_GPT.2 trace output | debug/trace mode reports phase counts and GPU launch counters |
| M090_GPT.3 correctness fixtures | accepted/rejected candidate fixtures pass |
| M090_GPT.4 bounded behavior | candidate explosion guard implemented or explicitly scoped |
| M090_GPT.5 world-view validation | trace records guess count, reduced-program model count, accepted world-view count, and rejection reasons |
| M090_GPT.6 GPU residency | candidate generation, propagation, and world-view validation hot path uses GPU-resident buffers |

### G090_SPLIT - Epistemic Splitting

**Goal.** Add epistemic splitting for the purpose of decomposing programs where semantics permit, with respect to correctness and performance isolation.

**Questions.**

- Q090_SPLIT.1: Which dependency graph defines a valid split?
- Q090_SPLIT.2: Can independent components be solved separately and recombined?
- Q090_SPLIT.3: What invalid split cases must be rejected?
- Q090_SPLIT.4: How are epistemic dependencies represented so buried modal coupling cannot be split unsafely?

**Metrics.**

| Metric | Target |
|---|---|
| M090_SPLIT.1 graph construction | deterministic dependency graph |
| M090_SPLIT.2 valid split fixtures | 100 percent pass |
| M090_SPLIT.3 invalid split fixtures | typed rejection |
| M090_SPLIT.4 recomposition | recomposed output equals unsplit output on fixtures |
| M090_SPLIT.5 modal coupling guard | fixture with cross-component epistemic dependency is not split |
| M090_SPLIT.6 GPU split execution | valid split components execute through GPU-native subplans, not CPU-only recomposition |

### G090_GPU - GPU-Native Runtime And WCOJ Execution

**Goal.** Implement full GPU-native epistemic execution for the purpose of making v0.9.0 a production runtime feature, with respect to RIR/runtime dispatch, WCOJ eligibility, GPU-resident world-view state, and measurable GPU execution.

**Questions.**

- Q090_GPU.1: How do accepted EIR programs lower into production executable plans?
- Q090_GPU.2: Which epistemic reductions are WCOJ-eligible, and how is eligibility proven?
- Q090_GPU.3: Which GPU buffers represent candidate assumptions, model membership, world views, and rejection reasons?
- Q090_GPU.4: Which kernels perform Generate-Propagate-Test, world-view validation, and materialization?
- Q090_GPU.5: How are host transfers bounded to input loading, launch parameters, diagnostics, and final results?
- Q090_GPU.6: How is GPU execution measured in tests and certification evidence?
- Q090_GPU.7: How are nonzero-arity stable-model tuple keys matched on GPU using existing relation layouts and device columns?
- Q090_GPU.8: Which evidence proves the implementation reused existing WCOJ/K-clique/helper-split/runtime-histogram paths instead of a parallel epistemic route?

**Metrics.**

| Metric | Target |
|---|---|
| M090_GPU.1 production lowering | accepted epistemic fixture runs through production runtime dispatch |
| M090_GPU.2 WCOJ eligibility | at least one epistemic reduction uses the WCOJ planner/path where eligible |
| M090_GPU.3 GPU buffers | candidate, world-view, and rejection state have GPU-resident representations |
| M090_GPU.4 kernel coverage | GPU kernels cover candidate generation, propagation, validation, and materialization hot paths |
| M090_GPU.5 CPU fallback ban | accepted execution trace records zero CPU candidate enumeration/world-view validation fallbacks |
| M090_GPU.6 launch evidence | certification logs include nonzero GPU launch counts and kernel timing for epistemic execution |
| M090_GPU.7 parity | GPU output matches semantic oracle on all G91, FAEEL, GPT, and splitting fixtures |
| M090_GPU.8 transfer budget | host-device transfers are bounded and reported; no per-candidate host round trip in hot path |
| M090_GPU.9 nonzero-arity membership | at least two fixtures with arity >= 1 check stable-model tuple membership on GPU over existing relation layouts |
| M090_GPU.10 row-count guard | nonzero-arity membership fails closed if only row-count metadata is available |
| M090_GPU.11 production path reuse | source audit and runtime counters prove existing RIR/runtime/WCOJ/K-clique/helper-split paths executed where eligible |

**Expected targets.**

- `crates/xlog-ir/`
- `crates/xlog-runtime/`
- `crates/xlog-cuda/`
- `crates/xlog-logic/`
- runtime and CUDA integration tests under the relevant crates

### G090_SOLVER - Solver-Service Integration

**Goal.** Integrate GPU-native solver services with xlog-logic constraints for the purpose of enabling incremental SAT, assumptions, learned-clause transfer, MaxSAT, and portfolio solving under a clear interface, by reusing the existing solver production core rather than creating an epistemic-only search engine.

**Questions.**

- Q090_SOLVER.1: What is the minimal solver interface between xlog-logic and solver services?
- Q090_SOLVER.2: How are assumptions represented and retracted?
- Q090_SOLVER.3: How are learned clauses transferred across incremental calls?
- Q090_SOLVER.4: How are soft constraints represented for MaxSAT?
- Q090_SOLVER.5: How does GPU portfolio solving execute candidate SAT/MaxSAT work without CPU search in the accepted path?
- Q090_SOLVER.6: How are SAT, UNSAT, UNKNOWN, and TIMEOUT propagated to the epistemic candidate state machine?
- Q090_SOLVER.7: Which existing `xlog-solve` GPU CNF/CDCL/solver paths are used for accepted epistemic candidates?
- Q090_SOLVER.8: Which fixture-only CPU solver code remains, and how is it prevented from entering accepted release paths?

**Metrics.**

| Metric | Target |
|---|---|
| M090_SOLVER.1 interface | trait/API documented and tested |
| M090_SOLVER.2 incremental SAT | add/retract assumption fixtures pass on GPU-native path |
| M090_SOLVER.3 learned clauses | transfer observable in GPU trace or test double |
| M090_SOLVER.4 MaxSAT | soft-constraint fixture returns expected optimum on GPU-native path |
| M090_SOLVER.5 GPU portfolio | portfolio dispatch executes on GPU or GPU-backed adapter with measured launch evidence |
| M090_SOLVER.6 failure modes | UNSAT/UNKNOWN/TIMEOUT represented distinctly |
| M090_SOLVER.7 assumption lifecycle | push, solve, retract, and reuse trace proves no assumption leak between candidates |
| M090_SOLVER.8 CPU search ban | accepted solver path records zero CPU exhaustive assignment enumeration |
| M090_SOLVER.9 production solver reuse | accepted SAT/MaxSAT fixtures execute through existing GPU CNF/CDCL/solver production APIs or thin adapters over them |
| M090_SOLVER.10 fixture isolation | CPU semantic-oracle solver facade is gated so it cannot satisfy closure metrics |

**Expected targets.**

- `crates/xlog-solve/`
- `crates/xlog-logic/`
- `docs/architecture/`
- solver tests under the relevant crates

### G090_PROB - Probabilistic And Circuit Integration

**Goal.** Integrate epistemic reasoning with probabilistic inference for the purpose of preserving coherent query semantics across deterministic, epistemic, and probabilistic layers, by feeding accepted world-view evidence into the existing GPU-native exact/provenance production path.

**Questions.**

- Q090_PROB.1: How do epistemic choices affect probabilistic evidence and query compilation?
- Q090_PROB.2: Can circuits be updated incrementally when epistemic assumptions change?
- Q090_PROB.3: Which external compiler adapters are necessary for v0.9.0?
- Q090_PROB.4: How is probabilistic evidence conditioned on accepted world views without bypassing epistemic validation?
- Q090_PROB.5: Which existing `xlog-prob` exact/provenance/PIR/knowledge-compilation APIs consume accepted world-view evidence?
- Q090_PROB.6: How is fixture-only epistemic probability code prevented from replacing the production exact path?

**Metrics.**

| Metric | Target |
|---|---|
| M090_PROB.1 semantic contract | documented interaction between epistemic and probabilistic layers |
| M090_PROB.2 incremental circuit fixture | changed assumption updates circuit without full rebuild where supported |
| M090_PROB.3 compiler adapter | at least one alternative compiler adapter design or implementation |
| M090_PROB.4 numerical stability | deterministic fixture within documented tolerance |
| M090_PROB.5 evidence conditioning | probabilistic integration consumes accepted world views, not raw unvalidated guesses |
| M090_PROB.6 GPU exact integration | accepted world-view evidence updates the GPU-native exact/provenance path |
| M090_PROB.7 CPU recompute ban | accepted probabilistic epistemic path records zero CPU-only probability recomputation |
| M090_PROB.8 production prob reuse | accepted probabilistic fixtures execute through existing GPU exact/provenance/PIR/knowledge-compilation APIs |
| M090_PROB.9 fixture isolation | bounded epistemic probability fixtures are marked oracle-only and cannot satisfy closure metrics |

### G090_CERT - Certification And Regression Gates

**Goal.** Certify v0.9.0 GPU-native semantics for the purpose of making merge decisions objective after v0.8.0 lands.

**Metrics.**

| Metric | Target |
|---|---|
| M090_CERT.1 semantic golden tests | 100 percent pass |
| M090_CERT.2 solver tests | 100 percent pass for GPU-native solver scope |
| M090_CERT.3 parser diagnostics | positive and negative syntax fixtures pass |
| M090_CERT.4 v0.8 compatibility | v0.8 pyxlog/DTS cert subset rerun after rebase |
| M090_CERT.5 formatting | `cargo fmt --check` pass |
| M090_CERT.6 workspace health | agreed cargo test subset pass |
| M090_CERT.7 semantic trace fixtures | Generate-Propagate-Test traces include generated, accepted, and rejected candidate counts |
| M090_CERT.8 GPU-native evidence | certification evidence includes GPU launch counts, kernel timings, and zero CPU fallback counters |
| M090_CERT.9 WCOJ evidence | at least one WCOJ-eligible epistemic reduction proves WCOJ planner/runtime dispatch |
| M090_CERT.10 nonzero-arity membership | certification includes GPU tuple-key membership evidence for arity >= 1 predicates |
| M090_CERT.11 solver production reuse | certification includes traces proving accepted SAT/MaxSAT work used existing GPU solver production paths |
| M090_CERT.12 prob production reuse | certification includes traces proving accepted probabilistic evidence used existing GPU exact/provenance paths |
| M090_CERT.13 no parallel engines | source audit reports zero new epistemic-only WCOJ, solver-search, probability-inference, or tuple-store engines in accepted paths |

### G090_DOC - Documentation

**Goal.** Document epistemic and solver semantics for the purpose of making the release usable and auditable.

**Metrics.**

| Metric | Target |
|---|---|
| M090_DOC.1 epistemic guide | guide explains EIR, G91, FAEEL, GPT, splitting |
| M090_DOC.2 solver guide | guide explains GPU-native assumptions, incremental SAT, MaxSAT, portfolio dispatch, and failure states |
| M090_DOC.3 examples | at least one runnable example per implemented major semantic mode |
| M090_DOC.4 roadmap sync | ROADMAP v0.9.0 rows updated only at closure, not prematurely marked done |
| M090_DOC.5 GPU/WCOJ execution | guide documents the production GPU-native and WCOJ-backed epistemic execution path |

### G090_CLOSE - Closure Proposal After v0.8.0 Rebase

**Goal.** Produce a v0.9.0 closure proposal after rebasing onto v0.8.0 for the purpose of deciding whether the branch can merge.

**Metrics.**

| Metric | Target |
|---|---|
| M090_CLOSE.1 rebase | branch rebased or merged on top of v0.8.0 integration commit |
| M090_CLOSE.2 conflict report | all conflicts and resolutions documented |
| M090_CLOSE.3 metric table | every G090 node has PASS/FAIL/BLOCKED status |
| M090_CLOSE.4 release decision | recommendation is MERGE_READY, HOLD_FOR_FIXES, or SCOPE_AMENDMENT_REQUIRED |
| M090_CLOSE.5 no implicit release | no push, tag, board update, or merge without coordinator authorization |

## 6. KPIs

| KPI | Gate |
|---|---|
| KPI090.1 EIR explicitness | all epistemic constructs represented explicitly or rejected with typed diagnostics |
| KPI090.2 World-view boundary | accepted results are world views whose stable models satisfy the original epistemic guesses |
| KPI090.3 G91 compatibility | 100 percent G91 golden fixtures pass, including documented G91-only compatibility cases |
| KPI090.4 FAEEL default correctness | 100 percent FAEEL golden fixtures pass, including foundedness rejection cases |
| KPI090.5 GPT correctness | accepted/rejected candidate fixtures pass with phase-count trace evidence |
| KPI090.6 Splitting correctness | split and unsplit outputs match on valid fixtures; invalid modal coupling is rejected |
| KPI090.7 GPU-native execution | accepted epistemic execution uses GPU kernels with zero CPU fallback counters |
| KPI090.8 WCOJ integration | WCOJ-eligible epistemic reductions dispatch through the WCOJ/GPU runtime path |
| KPI090.9 Solver semantics | incremental SAT and MaxSAT fixtures pass on GPU-native path with status distinction |
| KPI090.10 Probabilistic coherence | epistemic/probabilistic fixtures pass on GPU-native exact path within documented tolerance |
| KPI090.11 v0.8 compatibility after rebase | v0.8 pyxlog/DTS compatibility subset remains green |
| KPI090.12 Release hygiene | no unapproved merge, push, tag, or v0.8-owned API drift |
| KPI090.13 Nonzero-arity membership | arity >= 1 epistemic membership fixtures pass on GPU over existing relation layouts |
| KPI090.14 Production-path reuse | certification proves existing runtime/WCOJ, solver, and probabilistic production paths were reused |
| KPI090.15 Fixture containment | semantic-oracle fixtures remain useful but cannot satisfy release closure metrics |

Concurrency hardening note: the v0.6.0 A3 same-process multi-executor CUDA
primary-context drift item remains a runtime-concurrency backlog item, not part
of the v0.9.0 epistemic/solver closure KPI surface above. It must not be counted
as implemented by this goal unless a future runtime hardening change certifies
zero A3 drift in the `per_thread` and `shared` fixture modes.

## 7. Worktree Setup

Coordinator command:

```bash
cd /home/dev/projects/xlog
git worktree add .worktrees/v090-epistemic -b feat/v090-epistemic-solver-semantics
cd .worktrees/v090-epistemic
git status --short --branch
```

Agent first actions:

1. Record `git rev-parse HEAD` and `git status --short --branch`.
2. Read this document, `ROADMAP.md` v0.9.0, relevant xlog-logic/xlog-ir/xlog-solve docs, and the v0.8.0 agent goal document.
3. Create `docs/evidence/<date>-v090-pre/README.md` with baseline state and ownership map.
4. Execute G090_PRE before any implementation.

## 8. Reporting Protocol

After each sub-goal, report:

- branch and commit SHA;
- files changed;
- metrics table with PASS/FAIL/BLOCKED;
- exact validation commands and raw numbers;
- whether any v0.8-owned surface was touched;
- next sub-goal or halt condition.

## 9. Halt Conditions

Halt and ask the coordinator if:

- a semantic design requires changing pyxlog public APIs owned by v0.8.0;
- a solver integration requires a new external dependency not already accepted by the project;
- an implementation attempts to close with CPU-only or fixture-only epistemic execution;
- an implementation introduces a parallel epistemic-only WCOJ, relation-store, solver-search, probability-inference, or tuple-membership engine instead of reusing existing production paths;
- nonzero-arity tuple membership cannot be proven on GPU over existing relation layouts;
- accepted solver or probabilistic execution cannot be traced through existing production APIs;
- GPU-native execution cannot be proven with launch counters, kernel timings, and zero CPU fallback counters;
- a correctness fixture conflicts with existing documented semantics;
- a metric needs to be weakened or redefined;
- the branch cannot rebase cleanly after v0.8.0 lands;
- any action would require push, tag, merge, or release-board mutation.

## 10. Final Deliverable

The final output is a v0.9.0 closure proposal with:

- complete GQM metric table;
- evidence links;
- all sub-goal commit SHAs;
- rebase/conflict report against v0.8.0 integration commit;
- remaining risk summary;
- merge recommendation.

v0.9.0 is not release-ready until v0.8.0 has landed and every KPI090 gate is green.
