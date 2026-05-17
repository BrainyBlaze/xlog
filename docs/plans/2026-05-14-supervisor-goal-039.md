# Supervisor Goal 039 — Phase 2: v0.6.5 Final Closure + DTS-DLM Hot-Loop Completion

**Supervisor:** Claude Code.
**Implementer:** Codex CLI on tmux session `codex-xlog`.
**Predecessor:** `docs/plans/2026-05-14-supervisor-goal-038.md` (Phase 1: W3 axis closure).
**Predecessor state at Phase-2 cut:** Phase 1 DONE — `feat/w3-bundle-integration` HEAD `<SET_AT_PHASE1_CLOSE>`; W3.3, W3.5, W3.6, W3.7, W3.8, W3.9 all DONE; W3.4 re-validated; W4.1/W5.1/W5.2/W2.5 regression-free; peak VRAM ≤ 38 GB.
**Phase-2 launch precondition:** Phase 1 closed AND user has explicitly authorized Phase 2 dispatch.
**Date:** 2026-05-14.
**Methodology:** Basili–Caldiera–Rombach GQM + GQM+Strategies. References: https://en.wikipedia.org/wiki/GQM.
**Paper:** SRDatalog arXiv:[2604.20073](https://arxiv.org/abs/2604.20073).
**Consumer:** DTS-DLM (Dialetheic Tensor-Symbolic Diffusion Language Model) at `/home/dev/projects/dts-dlm`. xlog is the GPU-resident hot-loop substrate for DTS-DLM Theses 7–11 per `docs/dts-dlm-system-design.md`.
**Closure board:** `docs/v065-closure-board.md`. Phase 2 advances the **remaining 5 OPEN items** (W5.3, W5.4, W6.1, W6.2 existing + new W6.3, W6.4, W6.5, W6.6 added by Phase 2) AND clears the path for W7.1 user-authorized release tag.

---

## 0. Process locks (durable, inherited + DTS-DLM extensions)

Goal-037 process locks 1–10 + goal-038 Phase-1 locks 11–14 inherited where applicable. Phase 2 extends with DTS-DLM-consumer-specific locks 15–22:

The inherited 10 locks (see goal-037 §0):

1. No simplification clauses
2. No back-compat shims
3. No `Ok(None)` decline for paper-aligned shapes
4. No bench-gate substitution
5. No dead-code preservation
6. No comment rot
7. No `Co-Authored-By` trailers
8. No `v0.6.6` references
9. Bench-spike-first
10. GQM+Strategies dispatch shape

Phase-2 extensions:

11. **DTS-DLM consumer-contract preservation.** Every pyxlog API DTS-DLM consumes is **frozen**. New optional parameters with defaults are permitted iff they do not change pre-existing call sites. No signature changes, no behavioral changes outside the documented `deterministic=True` + seed-pin contract. The frozen surface spans two contract groups:
    - **Group A — Stage 4/5 active surface** (DTS-DLM xlog usage dump Scenarios 1+2+3, expanded R6 per supervisor amendment 2026-05-17 for M37-F readiness): `LogicProgram.compile`, `program.session`, `session.put_relation`, `session.evaluate`, `session.export_relation`, `IlpProgramFactory.compile`, `train_on_compiled_relations`, `TrainConfig`, `StrictTrainResult`, `LogicSessionAdapter` (Scenario 3 persistent session wrapper), **`train_and_promote`** (xlog-induce meta-API on top of `train_on_compiled_relations`; consumes the same `TrainConfig`/produces same `StrictTrainResult`; adds promotion-gate evaluation: convergence + novel-rate + regression + holdout-F1 gates per FINAL-REPORT M37-F candidate scope; M37-F dILP rule discovery consumer queued post-M37-A).
    - **Group B — M18 / M37-A neural-symbolic training surface** (DTS-DLM xlog usage dump Scenario 4 + M37-A queued per `dts-dlm/docs/plans/2026-05-11-m37c-prime-narrow-recalibration-plan-freeze.md` §8; expanded R7 per supervisor amendment 2026-05-17 to cover v0.4.0-beta/ga/v0.5.x surfaces per FINAL-REPORT Stone #9): `nn/4` neural predicate Datalog syntax (`nn(network, [inputs], output, [labels]) :: predicate(args)`), `register_network(name, nn_module, optimizer)`, `forward_backward_tensor(query)` (returns CUDA tensor loss with zero host reads — strict GPU-native contract), `train_epoch(queries, batch_size=N)`, XGCF GPU forward+backward weighted model counting, circuit caching (100× speedup on repeated queries — performance contract quantified per R8 + KPI-9), probabilistic queries `P(Query|Evidence)` with per-query gradient output, **`register_embedding(name, embedding_module)`** (term embeddings per FINAL-REPORT Stone #9 v0.4.0-beta/ga shipped surface), **training controls**: gradient-clipping parameters (`max_grad_norm`, clip strategy), early-stopping criteria (patience, min-delta), scheduler config (StepLR/CosineLR/Plateau), learning-rate override on `register_network`, **Bounded Exact Induction** (`xlog-induce` crate public API: `bounded_exact_induce(program, examples, budget)` + result types) per v0.4.0-beta/ga shipped surface. Group B is **proven by M18 phases A–D** (`xlog_alpha_source.py` source emitter); v0.5.x extensions proven via xlog `examples/neural/` reference suite (MNIST-Add 99.07% with addition-only supervision). M37-A and M37-F are the two queued consumers of the expanded Group B surface.
12. **Determinism contract preservation** (extended R9 per supervisor amendment 2026-05-17 for M37-A dynamic rule injection scenario). WP2.6 memoization-safety contract: `deterministic=True` + seed-pin produces bit-exact relation output under three regimes:
    - (a) **Stateless evaluation** across `evaluate()` calls on an unchanged program — bit-exact across 100 replays (G_W53 M_W53.3).
    - (b) **Repeated `register_network`** calls in the same session with the same `nn.Module` and `torch.optim.Optimizer` — bit-exact gradient output across 100 invocations (G_W53 new sub-test per R9).
    - (c) **Program mutation between `evaluate()` calls** — when Stage 5 ILP induction injects a new rule via `train_and_promote` or M37-A injects a learned rule mid-session, subsequent `evaluate()` calls with the same seed produce bit-exact output across runs (G_W53 new sub-test per R9 — dynamic rule injection scenario). xlog internal random state (hash-seed for relation indexing, search-tie-breaking, optimizer initialization) MUST be deterministic under seed-pin or expose configuration to be made deterministic.
13. **Possibilistic-vs-probabilistic separation.** xlog kernels are agnostic to (pro, contra) confidence channels. xlog provides structural derivation; xlog MUST NOT introduce probabilistic-sum aggregation into Stage 4 kernels (DTS-DLM plan-freeze §2.2(e)). Stage 5 ILP induction sandbox (`pyxlog.ilp`) is the only locus where probabilistic-sum is permitted, and it is OUT OF SCOPE for Phase 2.
14. **Sort-label metadata is consumer-visible.** Every output relation MUST emit per-column sort labels. The `Sort enrichment: N sort-map misses` diagnostic in DTS-DLM `xlog_executor.py:157` becomes an error condition for Phase 2 (G_W65), not a tolerated warning.
15. **m37c-prime trace replay is the canonical DTS-DLM fixture.** Pilot trace at `/home/dev/projects/dts-dlm/out/m37c-prime/` is source of truth for Stage 4 rule-shape distribution. Synthetic DTS-DLM-analog fixtures reproduce this distribution; final G_E2E validation runs on m37c-prime directly.
16. **No DTS-DLM repo mutations** except (a) read-only profiler-trace instrumentation in G_PRE (reverted after trace), (b) maturin-built xlog wheel reinstall for G_E2E. No DTS-DLM source code changes outside these two precisely-scoped exceptions.
17. **No changes to `xlog-induce` crate.** Stage 5 ILP path is untouched. W6.x scope is xlog-cuda / xlog-runtime / xlog-logic only.
18. **No changes to `xlog-prob`, `xlog-neural`, `xlog-solve`, `xlog-gpu`, `xlog-stats`, `xlog-cli`, `xlog-core`, `xlog-ir`, `pyxlog` crates** beyond what G_W65 (sort labels), G_W66 (CUDA Graphs in xlog-cuda), G_E2E (pyxlog rebuild), and G_M37A_SURFACE (M37-A surface regression cert — NO code edits; cert-only) strictly require AND lock 11 permits. Critically, `xlog-prob` (XGCF + circuit caching) and `xlog-neural` (`nn/4` + `register_network` + `train_epoch`) must be preserved verbatim: their existing behavior is M37-A's substrate. G_PURGE2 MUST NOT remove any Group B symbol even if no Phase-2 call site references it — Group B has a queued downstream consumer (M37-A, queued unconditionally per m37c-prime plan-freeze §8). Resolution if static analysis flags Group B symbols as dead: add a smoke test that exercises the symbol in `crates/xlog-integration/tests/test_m37a_surface_preservation.rs`, do NOT remove the symbol.
19. **No re-opening of Phase-1 W3 axis.** Phase-2 work that reveals a Phase-1 regression escalates; does NOT silently revert Phase-1 commits.
20. **No re-validation of W3.4 inside Phase 2.** Phase 2 regression-checks W3.4 as part of G_INT2. Phase-2-introduced regression is root-caused in Phase-2 scope.
21. **No xlog tag pushes inside Phase 2.** v0.6.5 tag remains W7.1-gated, user-authorized. Phase 2 produces tag-ready HEAD only.
22. **No behavioral drift on frozen API.** Same-signature same-name pyxlog methods must produce semantically-equivalent output across xlog versions. Cert: G_E2E captures reference output set on `main @ f62188b7` via DTS-DLM `pyxlog`-roundtrip test; replays on Phase-2 HEAD; asserts bit-exact.

---

## 1. Strategic context (GQM+Strategies)

### 1.1 Business goal

> **BG0.** Ship DTS-DLM v3 serving artifact with v0.6.5-tagged xlog as the GPU substrate, such that DTS-DLM v3 satisfies its frontier-promotion criteria on the m37c-prime pilot fixture set, with no regression vs the v2 serving baseline (`/home/dev/projects/dts-dlm/data/serving/v2/`, promoted 2026-04-13). v0.6.5 tag (W7.1) fires only after every closure-board item is DONE AND user explicitly authorizes.

### 1.2 Assumptions

| ID | Assumption | Source |
|---|---|---|
| A1 | DTS-DLM Stage 4 hot-loop wall-time is dominated by `xlog.evaluate()` when rule shapes are WCOJ-eligible. | Hypothesis — measured under G_PRE |
| A2 | DTS-DLM Stage 4 rule-shape distribution is mixed: chain joins (2-atom bodies) are most frequent; triangle/cycle/clique shapes appear in dILP-induced rules. | DTS-DLM xlog usage dump §1 Scenario 1 (`src/dts_dlm/propagate/xlog_executor.py:1301-1450`) |
| A3 | DTS-DLM relies on `deterministic=True` + seed-pin contract for memoization safety (WP2.6). | DTS-DLM xlog usage dump §6 + project memory |
| A4 | DTS-DLM confidence math (Belnap pro/contra) runs in PyTorch on top of xlog structural output, NOT in xlog kernels. | DTS-DLM xlog usage dump §3 + §6 |
| A5 | Paper SRDatalog (arXiv:2604.20073) is canonical theoretical reference for xlog WCOJ + recursive-Datalog design. | Project memory `reference_srdatalog_paper.md` |
| A6 | Phase-1 `feat/w3-bundle-integration` HEAD is durable Phase-2 base. Triangle/cycle/clique covered there; chain joins are NOT and require G_W63. | Phase-1 hand-off contract goal-038 §10 |
| A7 | Witness-chain recoverability (Belnap accountability per DTS-SPEC-01 Law of Metastable Contradiction) must survive any optimization including chain-join routing. | DTS-DLM system-design Thesis 8 + DTS-SPEC-01 |
| A8 | DLPack zero-copy is load-bearing for Stage 4 throughput. Any host-transfer introduced by optimization regresses Stage 4 even if kernel speedup increases. | DTS-DLM xlog usage dump §1 Memory profile |

### 1.3 Strategy

> Measure xlog.evaluate() share of Stage 4 (G_PRE) to set G_W63 priority; close remaining 4 OPEN W3-adjacent items (W5.3 determinism harness, W5.4 replay gate, W6.1 architecture guide doc, W6.2 user guide doc); add 4 new W6.x items for DTS-DLM gaps (W6.3 chain-join promoter, W6.4 K=7/K=8 templates, W6.5 sort-label propagation, W6.6 CUDA Graphs); extend Phase-1 G_W39 harness with DTS-DLM-analog fixture; validate composed bundle end-to-end against m37c-prime KPIs (G_E2E); produce tag-ready HEAD and await user-authorized W7.1 release tag.

### 1.4 Phase-2 KPI

> **KPI-1:** m37c-prime trace-replay end-to-end wall-time speedup ≥ **2.0×** (gate) / ≥ **3.0×** (stretch) vs `main @ f62188b7`.
> **KPI-2:** Determinism: bit-exact relation output across **100** successive m37c-prime replays under fixed seed.
> **KPI-3:** Zero DTS-DLM API surface regressions: every pyxlog call site in DTS-DLM `src/dts_dlm/integrations/pyxlog/` functions with unchanged signatures AND semantically-equivalent output.
> **KPI-4:** Closure board reaches 0 OPEN (W3.x via Phase 1 + W5.3/W5.4/W6.1/W6.2/W6.3/W6.4/W6.5/W6.6 via Phase 2) excluding W7.1 release tag; W7.1 cleared for user-authorized firing.
> **KPI-5:** Peak VRAM on m37c-prime full replay ≤ 80% of 48 GB (≤ 38 GB).
> **KPI-6:** Witness-chain recoverability: every fact in WMIR-derived relation has a recoverable witness chain via `choice_sources` / `leaf_atoms` accessors (100% on 50 docs).
> **KPI-7** (extended R10 per supervisor amendment 2026-05-17): DLPack zero-copy preserved on Stage 4 hot path AND Stage 5 witness-chain traversal path. Zero `cudaMemcpyDtoH` / `cudaMemcpyHtoD` outside `enrich_support_sorts` on Stage 4 hot path. Zero `cudaMemcpyDtoH` / `HtoD` on `choice_sources` / `leaf_atoms` accessor traversal during Stage 5 governance for m37c-prime 50-doc replay. Cert: G_E2E adds CUDA-event trace assertion for Stage-5 witness traversal path.
> **KPI-8:** M18 / M37-A neural-symbolic training surface preserved: `nn/4` + `register_network` + `forward_backward_tensor` + `train_epoch` + XGCF + circuit caching + `register_embedding` + training controls + Bounded Exact Induction all functional on Phase-2 HEAD; M18-D verified config reproduces AUROC gains 0.387 / 0.387 / 0.463 with ≥ 99.8% coverage on the slice referenced in `dts-dlm/docs/research/2026-05-08-pre-m37/04-FINAL-REPORT.md` Stone #9.
> **KPI-9** (NEW R8 per supervisor amendment 2026-05-17): Circuit caching performance contract quantified. Cache scope = per `program.session()`. Cache survives across `forward_backward_tensor()` invocations within a session. Cache-hit speedup ≥ 50× on repeated identical query (gate; paper claims 100× as headline). Cache-hit rate ≥ 95% on M18-D verified config replay over 10 epochs (since M18-D rule program is constant across epochs, all repeat-queries should hit cache). M37-F `train_and_promote` per-attempt repeated queries also hit cache. Cert: G_M37A_SURFACE M_M37A.5 extended to gate cache-hit rate explicitly.

---

## 2. Goal hierarchy (Phase-2 GQM tree)

```
BG0 — DTS-DLM v3 ship with v0.6.5 xlog
 │
 ├── G_PRE — Profiler trace prerequisite (sets G_W63 priority HIGH/MEDIUM/LOW)
 │
 ├── Closure-board OPEN items (existing):
 │   ├── G_W53 — W5.3 single determinism test harness (WCOJ + binary-join + recursive)
 │   ├── G_W54 — W5.4 downstream widened-frontier stress replay gate
 │   ├── G_W61_DOC — W6.1 WCOJ architecture guide (docs/wcoj-architecture-guide.md)
 │   └── G_W62_DOC — W6.2 WCOJ user guide (docs/wcoj-user-guide.md)
 │
 ├── Closure-board OPEN items (newly added by Phase 2):
 │   ├── G_W63_CHAIN — W6.3 chain-join WCOJ-shape promoter
 │   ├── G_W64_K78 — W6.4 K=7/K=8 clique templates
 │   ├── G_W65_SORT — W6.5 sort-label propagation
 │   └── G_W66_CUDAGRAPH — W6.6 CUDA Graph capture for Stage 4 hot loop
 │
 ├── G_W39_DTSDLM — Phase-1 G_W39 harness extension (DTS-DLM-analog fixture)
 │
 ├── G_M37A_SURFACE — M18 / M37-A neural-symbolic training surface preservation cert (KPI-8)
 │
 ├── G_INT2 — Phase-2 integration gate (W3.4/W4.1/W5.1/W5.2/W2.5 regression-free)
 │
 ├── G_PURGE2 — Phase-2 cross-cutting refactor + dead-code/comment purge
 │
 ├── G_CLOSE2 — Phase-2 closure proposal + user approval + board update for W5.3, W5.4, W6.1–W6.6
 │
 └── G_E2E — DTS-DLM end-to-end validation (KPI-1..KPI-7) → W7.1 tag-ready handoff
```

16 sub-goals (1 prerequisite + 4 existing OPEN + 4 new W6.x + 1 harness extension + 1 M37-A surface cert + 3 closure + 1 E2E + 1 final). Dependency DAG at §4.

---

## 3. Per-goal GQM decomposition

### 3.1 G_PRE — Quantify xlog.evaluate() share of Stage 4 wall-time

**Goal.** Analyze the m37c-prime Stage 4 execution trace for the purpose of characterizing `xlog.evaluate()` wall-time share with respect to G_W63 ROI ceiling from the viewpoint of optimization prioritization in the context of Phase-2 dispatch sequencing.

**Why prerequisite.** A4 says confidence math runs in PyTorch on top of xlog. If `xlog.evaluate() % of Stage 4 < 30%`, G_W63 chain-join optimization has bounded ROI (Amdahl). G_PRE decides G_W63 HIGH/MEDIUM/LOW.

**Questions.**
- **Q_PRE.1** What fraction of Stage 4 wall-time on m37c-prime is `xlog.evaluate()` vs `enrich_support_sorts()` vs PyTorch confidence propagation?
- **Q_PRE.2** Rule-shape distribution within `evaluate()` calls? (chain-2 / triangle / cycle-4 / clique-K / recursive / mixed)
- **Q_PRE.3** Dominant cost: `put_relation` (DLPack handle), `evaluate` (kernel), or `export_relation` (DLPack out)?

**Metrics.**

| Metric | Definition | Target |
|---|---|---|
| **M_PRE.1** `evaluate_pct` | `time(session.evaluate()) / time(stage_4_total)` avg over 50 m37c-prime docs | Reported; no gate (informational + decision-rule input) |
| **M_PRE.2** `rule_shape_histogram` | Count of `evaluate()` invocations per shape | JSON histogram |
| **M_PRE.3** `phase_breakdown` | (put_relation, evaluate, export_relation, enrich_support_sorts) fractions of Stage 4 | Decimal tuple |

**Strategies.**
- **S_PRE.1** Instrument DTS-DLM `src/dts_dlm/propagate/xlog_executor.py` (XlogStage4Executor) with `time.perf_counter_ns()` around each `put_relation` / `evaluate` / `export_relation` / `enrich_support_sorts` call. JSONL to `/tmp/g39-pre-trace/<doc_id>.jsonl`.
- **S_PRE.2** Run m37c-prime arm-C path on 50 docs (`M37C_DOC_LIMIT=50`).
- **S_PRE.3** Aggregate; emit `docs/evidence/2026-05-14-g39-pre-profiler-trace/report.md` with M_PRE.* + decision-rule outcome + raw JSONL preserved.
- **S_PRE.4** **Decision rule** encoded in report:
  - `M_PRE.1 ≥ 0.60` → G_W63 priority **HIGH** (sequences before G_E2E)
  - `0.30 ≤ M_PRE.1 < 0.60` → G_W63 priority **MEDIUM** (parallel with G_W64–W66)
  - `M_PRE.1 < 0.30` → G_W63 priority **LOW** (defer to v0.7 OR add to v0.6.5 only if user authorizes; KPI-1 reassessed)
- **S_PRE.5** Revert DTS-DLM instrumentation per lock 16.

**Acceptance.** Report committed; decision rule outcome stated; instrumentation reverted.

---

### 3.2 G_W53 — W5.3 single determinism test harness (board entry)

**Goal.** Analyze cross-mode determinism (WCOJ + binary-join + recursive + dynamic-rule-injection) for the purpose of certifying bit-exact relation output across all execution modes with respect to KPI-2 + WP2.6 memoization safety from the viewpoint of DTS-DLM consumer-contract preservation in the context of closure board W5.3.

**Board entry text.** *"Single test harness for WCOJ + binary-join + recursive determinism. Today the parts exist separately."*

**Questions.**
- **Q_W53.1** Does a single test harness exercise all three execution modes (WCOJ / binary-join / recursive) on the same fixture with bit-exact equality assertion?
- **Q_W53.2** Does the harness cover dynamic rule injection (Stage 5 → Stage 4 path) per A3 and lock 12?
- **Q_W53.3** Are 100 successive runs bit-exact under fixed seed?

**Metrics.**

| Metric | Target |
|---|---|
| **M_W53.1** Test file `crates/xlog-integration/tests/test_cross_mode_determinism.rs` exists and runs all three modes on the same fixture | 1/1 |
| **M_W53.2** Bit-exact row equality WCOJ ≡ binary-join ≡ recursive output on shared fixture | 3-way bit-exact |
| **M_W53.3** 100 successive `evaluate()` runs under fixed seed produce bit-exact output | 100/100 |
| **M_W53.4** Dynamic-rule-injection scenario: compile R1, evaluate, inject R2, re-evaluate; 100 runs same seed; bit-exact | 100/100 |
| **M_W53.5** Stage 5 → Stage 4 arm-D rollback fixture: simulate `train_on_compiled_relations` returning `StrictTrainResult.discovered_rule`; inject; re-evaluate; 100 runs; bit-exact | 100/100 |
| **M_W53.6** If non-determinism source identified, RCA documented in `docs/evidence/2026-05-14-g39-w53-determinism-rca.md` | doc present iff RCA needed |

**Strategies.**
- **S_W53.1** Cut `feat/w53-determinism-harness-g39` from `feat/w3-bundle-integration` HEAD.
- **S_W53.2** Create `test_cross_mode_determinism.rs`. Fixture: shared 3-relation triangle + chain rule + recursive TC.
- **S_W53.3** Section 1: WCOJ path. Section 2: binary-join path (W4.2 nested-loop or hash). Section 3: recursive path (W4.1).
- **S_W53.4** Section 4: dynamic-rule-injection — `program.add_rule(...)` (verify API exists; fall back to `program.recompile_with(extra_rule=R2)` if not).
- **S_W53.5** Section 5: Stage 5 simulator emitting `StrictTrainResult` mock; inject into session.
- **S_W53.6** Run 100× under `XLOG_DETERMINISTIC=1` + fixed seed; assert bit-exact across all 100.
- **S_W53.7** If non-determinism: identify source (kernel-launch ordering, atomicAdd contention, scratch reuse); fix with deterministic alternative; re-cert.

**Acceptance.** All M_W53.* green.

---

### 3.3 G_W54 — W5.4 downstream widened-frontier stress replay gate (board entry)

**Goal.** Analyze the downstream widened-frontier stress replay path for the purpose of building a clean-gate replay harness with respect to integration-cert determinism gates from the viewpoint of DTS-DLM downstream consumer in the context of closure board W5.4.

**Board entry text.** *"Downstream widened-frontier stress replay clean gate. Replay harness must be built — none exists today."*

**Questions.**
- **Q_W54.1** What is the "widened-frontier stress replay" downstream consumer? (DTS-DLM frontier-promotion replay on m37c-prime per `data/serving/v2/`)
- **Q_W54.2** Does the replay harness produce a clean run logged in evidence?
- **Q_W54.3** Does the integration cert gate on replay determinism?

**Metrics.**

| Metric | Target |
|---|---|
| **M_W54.1** Replay harness committed at `crates/xlog-integration/tests/test_widened_frontier_replay.rs` (or equivalent path) | 1/1 |
| **M_W54.2** Clean replay run on canonical fixture | PASS |
| **M_W54.3** Replay evidence committed at `docs/evidence/2026-05-14-g39-w54-widened-frontier-replay/run.log` | log present, PASS marker |
| **M_W54.4** Integration cert gates on M_W54.2 | cert exists in CUDA cert suite |

**Strategies.**
- **S_W54.1** Coordinate with user to confirm canonical widened-frontier fixture (likely m37c-prime arm-D rollback trace).
- **S_W54.2** Build replay harness that consumes DTS-DLM frontier-promotion artifacts (`data/serving/v2/`) AND runs xlog through the replay.
- **S_W54.3** Capture run log + replay state; assert determinism + no errors.
- **S_W54.4** Add cert to CUDA cert suite that exercises a minimal replay representative.

**Acceptance.** All M_W54.* green.

---

### 3.4 G_W61_DOC — W6.1 WCOJ architecture guide (board entry)

**Goal.** Analyze the v0.6.5 WCOJ surface for the purpose of authoring a dedicated architecture guide with respect to comprehensive coverage of RIR/promoter/dispatch/cost-model/recursive-integration from the viewpoint of internal maintainers and external integrators in the context of closure board W6.1.

**Board entry text.** *"Dedicated WCOJ architecture guide. Covers RIR, promoter, dispatch, cost model, recursive integration."*

**Questions.**
- **Q_W61D.1** Does `docs/wcoj-architecture-guide.md` cover RIR / promoter / dispatch / cost model / recursive integration / Phase-1 HG-WCOJ + helper-split + stream-mux / Phase-2 chain-promoter + K=7/8 templates?
- **Q_W61D.2** Is it cross-linked from ROADMAP.md?

**Metrics.**

| Metric | Target |
|---|---|
| **M_W61D.1** `docs/wcoj-architecture-guide.md` exists with non-trivial coverage | file present, ≥ 1500 lines or appropriate depth |
| **M_W61D.2** Sections cover: RIR (`MultiWayJoin`/`ChainJoin`), promoter (`try_promote_triangle/4cycle/clique_k/chain`), dispatch (`wcoj_dispatch.rs`), cost model (`WcojVariableOrderingModel`/W2.1), recursive integration (W4.1 P1/P4) | 5/5 sections present |
| **M_W61D.3** Phase-1 + Phase-2 architecture covered: HG block-slice + helper-split + stream-mux + chain-promoter + K=7/8 templates + CUDA Graphs | 6/6 sub-sections present |
| **M_W61D.4** Cross-linked from `ROADMAP.md` | reference present |
| **M_W61D.5** Paper-§ citations correct: §5 Algorithm 1/2 cited where HG-WCOJ described; §5 helper-splitting cited where W3.7 described; §6 cited where stream-mux described | citation audit PASS |

**Strategies.**
- **S_W61D.1** Cut `feat/w61d-arch-guide-g39` from `feat/w3-bundle-integration` HEAD.
- **S_W61D.2** Author `docs/wcoj-architecture-guide.md` covering all sections in M_W61D.2 + M_W61D.3.
- **S_W61D.3** Cross-link from ROADMAP.md.
- **S_W61D.4** Citation audit: verify every paper reference (§4/§5/§6) is accurate vs arXiv:2604.20073.

**Acceptance.** All M_W61D.* green.

---

### 3.5 G_W62_DOC — W6.2 WCOJ user guide (board entry)

**Goal.** Analyze the v0.6.5 WCOJ user-facing surface for the purpose of authoring a user guide covering eligibility/fallback/performance-tuning with respect to consumer-actionable guidance from the viewpoint of downstream library users (DTS-DLM + external) in the context of closure board W6.2.

**Board entry text.** *"User-facing WCOJ eligibility / fallback / performance tuning guide. NOT just code-doc rehash."*

**Questions.**
- **Q_W62D.1** Does `docs/wcoj-user-guide.md` cover eligibility (shapes that route through WCOJ vs fall back to hash-join), env vars, config builders, when to opt into Cardinality vs Skew cost model, threshold tuning?
- **Q_W62D.2** Is it actionable (decision flowcharts, performance-tuning recipes) not descriptive (code-doc rehash)?

**Metrics.**

| Metric | Target |
|---|---|
| **M_W62D.1** `docs/wcoj-user-guide.md` exists | file present |
| **M_W62D.2** Sections cover: WCOJ eligibility decision flow; fallback paths; env vars (XLOG_WCOJ_*); config builders (CompilerConfig); cost-model trade-offs (Cardinality vs Skew); threshold tuning per workload | 6/6 sections |
| **M_W62D.3** Decision flowchart for "should I use WCOJ?" given workload shape | present |
| **M_W62D.4** Performance-tuning recipes for: skewed graphs, deep join trees, multi-rule strata, recursive workloads | 4/4 recipes |
| **M_W62D.5** Cross-linked from ROADMAP.md | reference present |

**Strategies.**
- **S_W62D.1** Cut `feat/w62d-user-guide-g39` from `feat/w3-bundle-integration` HEAD.
- **S_W62D.2** Author `docs/wcoj-user-guide.md` per M_W62D.2 + M_W62D.4.
- **S_W62D.3** Include decision flowchart (mermaid or ASCII).
- **S_W62D.4** Cross-link from ROADMAP.md.

**Acceptance.** All M_W62D.* green.

---

### 3.6 G_W63_CHAIN — W6.3 chain-join WCOJ-shape promoter (NEW board entry)

**Goal.** Analyze the 2-atom chain-join rule shape (DTS-DLM Stage 4's dominant shape per A2) for the purpose of routing through an optimized join path with respect to gate ≥ 2× / stretch ≥ 4× speedup on m37c-prime chain-shaped rules from the viewpoint of DTS-DLM hot-loop acceleration in the context of a NEW W6.3 closure-board entry added by Phase 2.

**Disclaimer (M-2 audit amendment):** G_W63_CHAIN is xlog-original engineering work, NOT directly paper-derived. Paper §3 names chain joins only as a problem case for binary plans; it does not prescribe a chain-specific WCOJ optimization. xlog's chain-join routing through W4.2/W4.3/WCOJ-degenerate-triangle is xlog's design choice based on G_PRE measurement + W2.1 cost-model integration.

**Priority gating per G_PRE.4 outcome:**
- HIGH (M_PRE.1 ≥ 0.60): G_W63_CHAIN blocks G_E2E.
- MEDIUM (0.30 ≤ M_PRE.1 < 0.60): parallel with G_W64–W66; all must merge before G_E2E.
- LOW (M_PRE.1 < 0.30): deferred to v0.7 with explicit user authorization OR Phase 2 ships without G_W63; KPI-1 reassessed.

**Questions.**
- **Q_W63.1** Optimal join path for 2-atom chain `head(X,Y) :- a(X,Z), b(Z,Y)` under W2.1 cost model?
- **Q_W63.2** Can the chain promoter detect shape at AOT compile time?
- **Q_W63.3** Cost-model threshold mapping for chain shapes?
- **Q_W63.4** Does promoter compose with Phase-1 G4 helper-split (helper-split output occasionally produces chain shapes)?
- **Q_W63.5** Does paper P4 (delta-outermost) cert PASS on chain-routed recursive bodies?

**Metrics.**

| Metric | Definition | Gate | Stretch |
|---|---|---|---|
| **M_W63.1** | speedup on m37c-prime chain-shaped rules (trace replay subset, ≥ 100 chain invocations) | ≥ **2.0×** | ≥ **4.0×** |
| **M_W63.2** | speedup on synthetic chain fixture at scale 977K | ≥ **1.5×** | ≥ **3.0×** |
| **M_W63.3** | row equality on every chain fixture | bit-exact | — |
| **M_W63.4** | no regression on triangle/cycle/clique (Phase-1 W3 bundle) | within **±3%** | — |
| **M_W63.5** | chain-shape detection precision: zero misroutes on non-chain rules | 0 misroutes (cert suite) | — |
| **M_W63.6** | helper-split + chain-promoter composition: helper-split output containing chain shapes correctly re-routed | composition cert PASS | — |
| **M_W63.7** | P4 delta-outermost cert PASS on chain-routed recursive bodies | cert PASS | — |
| **M_W63.8** | Peak VRAM | ≤ 38 GB | — |

**Strategies.**
- **S_W63.1** Cut `bench-spike/w63-chain-promoter-g39` from `feat/w3-bundle-integration` HEAD.
- **S_W63.2** Add `chain_join_promoter` pass in `crates/xlog-logic/src/optimizer/`:
  - After `selectivity_pass`, before `helper_split_pass`.
  - Detects: 2-atom inner-join body with exactly one shared variable position; emits `ChainJoin` RIR node.
- **S_W63.3** Runtime dispatch on `ChainJoin`:
  - `|a| × |b| ≤ NESTED_LOOP_TOTAL_THRESHOLD` → W4.2 nested-loop
  - Both pre-sorted on join key → W4.3 sort-merge operator (W4.3 is operator-only post-closure; dispatch is new)
  - Heavy fanout per W3.3 metadata → WCOJ HG-as-degenerate-triangle path with synthetic identity `xz` relation
  - Else → `hash_join_v2`
- **S_W63.4** Spike on m37c-prime trace replay subset; gate M_W63.1.
- **S_W63.5** Production branch `feat/w63-chain-promoter-prod-g39` with full W2.1 cost-model integration.
- **S_W63.6** Composition cert: fixture that exercises helper-split AND produces chain shapes; assert correct re-route.
- **S_W63.7** P4 cert: chain-routed recursive body re-picks leader to Δ relation.

**Acceptance.** All M_W63.* gates met OR graceful-defer per priority gating.

---

### 3.7 G_W64_K78 — W6.4 K=7 / K=8 clique templates (NEW board entry)

**Goal.** Analyze the `wcoj_clique_template_count_t<K, T>` template surface for the purpose of extending arity coverage to K=7 and K=8 with respect to paper §3 "6-to-8-way join rules" claim from the viewpoint of paper-aligned arity completeness in the context of DTS-DLM dILP-induced high-arity rules + a NEW W6.4 closure-board entry.

**Questions.**
- **Q_W64.1** Does W3.2 template generalize to K=7/K=8 without algorithmic changes (Tier-1 wrapper-call-only)?
- **Q_W64.2** Per-thread register / shared-memory footprint at K=8 fits launch parameters?
- **Q_W64.3** Promoter `try_promote_clique_k` extends to K=7/K=8 with constraint changes only?

**Metrics.**

| Metric | Target |
|---|---|
| **M_W64.1** 4 new provider entries: `wcoj_clique7_u32_recorded`, `wcoj_clique7_u64_recorded`, `wcoj_clique8_u32_recorded`, `wcoj_clique8_u64_recorded` | 4/4 |
| **M_W64.2** Tier-1 source-audit: each ABI wrapper template-call-only (no hand-written algorithm) | 8/8 (4 entries × Tier-1 + Tier-2) |
| **M_W64.3** Promoter accepts K=7/K=8; rejects K=9 | 4/4 promoter certs |
| **M_W64.4** Runtime dispatch counter advances on K=7 + K=8 | 2/2 counter certs |
| **M_W64.5** Row equality on K=7/K=8 fixtures vs hash-join reference | bit-exact |
| **M_W64.6** Per-thread register footprint at K=8 (via `--ptxas-options="-v"`) | ≤ 64 registers |
| **M_W64.7** Peak VRAM | ≤ 38 GB |

**Strategies.**
- **S_W64.1** Spike `bench-spike/w64-k78-template-g39` from `feat/w3-bundle-integration` HEAD.
- **S_W64.2** Extend `kernel_manifest_data.rs` + `CudaKernelProvider` with 4 new methods delegating to generic `wcoj_clique_recorded_inner` (mirrors W3.2 design).
- **S_W64.3** Extend `try_promote_clique_k` for k∈{7,8} (reuse `recursive_scan_count == 0` gate).
- **S_W64.4** Add 8 source-audit certs + 4 promoter certs + 2 dispatch counter certs + 2 row-equality certs.
- **S_W64.5** PTXas verbose check; gate M_W64.6.

**Acceptance.** All M_W64.* green.

---

### 3.8 G_W65_SORT — W6.5 sort-label propagation (NEW board entry)

**Goal.** Analyze the output relation schema metadata for the purpose of eliminating `Sort enrichment: N sort-map misses` diagnostic flood with respect to per-column sort-label authoritativeness from the viewpoint of DTS-DLM consumer log readability + downstream sort-aware operation correctness in the context of process lock 14 + NEW W6.5 closure-board entry.

**Questions.**
- **Q_W65.1** Root cause of un-inferred sort labels on padding columns?
- **Q_W65.2** Does fixing schema layer eliminate diagnostic, or does DTS-DLM `enrich_support_sorts` also need updating? (Per lock 16, DTS-DLM source is frozen except for G_PRE; xlog-side fix only.)
- **Q_W65.3** Does fix preserve existing schema call sites?

**Metrics.**

| Metric | Target |
|---|---|
| **M_W65.1** `Sort enrichment: N sort-map misses` diagnostic count on m37c-prime 50-doc replay | **0** |
| **M_W65.2** Schema-API regression test: all existing relation-schema call sites pass | 100% PASS |
| **M_W65.3** Sort label coverage: every output relation column has non-default sort label | 100% coverage cert |
| **M_W65.4** DTS-DLM `xlog_executor.py:157` unchanged | grep verification: file unchanged |
| **M_W65.5** RCA documented | `docs/evidence/2026-05-14-g39-w65-sort-label-rca.md` present |

**Strategies.**
- **S_W65.1** Locate code path emitting padding columns without labels: grep `padding`+`sort` in `crates/xlog-runtime/src/executor/` + `crates/xlog-cuda/src/provider/`.
- **S_W65.2** Document RCA in evidence doc.
- **S_W65.3** Implement xlog-runtime fix:
  - (a) propagate sort labels from input schema through executor to output, OR
  - (b) elide padding columns entirely if no downstream consumer
- **S_W65.4** Run small subset (5 docs) to confirm M_W65.1 = 0.
- **S_W65.5** Full m37c-prime 50-doc replay; confirm M_W65.1 = 0.

**Acceptance.** All M_W65.* green.

---

### 3.9 G_W66_CUDAGRAPH — W6.6 CUDA Graph capture (NEW board entry)

**Goal.** Analyze the Stage 4 hot-loop kernel launch sequence for the purpose of capturing into CUDA Graphs with respect to gate ≥ 1.2× / stretch ≥ 1.5× speedup on m37c-prime via launch-overhead amortization from the viewpoint of Thesis 9 "GPU-Resident Hot Loop" paper-explicit claim in the context of NEW W6.6 closure-board entry.

**Paper anchor:** Thesis 9 explicitly names "CUDA Graphs" — *"Tensorized joins, CUDA Graphs, zero CPU round-trips in core"*. Without CUDA Graphs, kernel launch overhead caps Stage 4 throughput at ~10K launches/sec; m37c-prime pilot dispatches ~24K kernel launches per arm, adding 0.5–1s pure launch overhead.

**Questions.**
- **Q_W66.1** Which kernel sequence in Stage 4 is suitable for CUDA Graph capture? (Count + Scan + Resize + Materialize per rule per stratum is the natural unit.)
- **Q_W66.2** Does Graph capture preserve determinism (lock 12) and DLPack zero-copy (KPI-7)?
- **Q_W66.3** Does Graph re-capture cost amortize over enough invocations to net positive?

**Metrics.**

| Metric | Definition | Gate | Stretch |
|---|---|---|---|
| **M_W66.1** | speedup on m37c-prime Stage 4 wall-time | ≥ **1.2×** | ≥ **1.5×** |
| **M_W66.2** | kernel-launch overhead reduction (measured via `cudaEventElapsedTime` per kernel + total wall) | ≥ **50%** | ≥ **80%** |
| **M_W66.3** | determinism preserved under Graph capture | 100/100 bit-exact (subset cert) | — |
| **M_W66.4** | DLPack zero-copy preserved | zero host transfers on hot path (cert) | — |
| **M_W66.5** | Peak VRAM | ≤ 38 GB | — |
| **M_W66.6** | Re-capture cost amortization: Graph capture launched at most 1× per fixpoint iteration, not per evaluate() call | cert PASS |

**Strategies.**
- **S_W66.1** Spike `bench-spike/w66-cuda-graph-g39` from `feat/w3-bundle-integration` HEAD.
- **S_W66.2** Identify capture unit: per-rule (Count → Scan → Resize → Materialize); per-stratum (multiple rules); per-fixpoint-iteration (multiple strata). Start with per-rule.
- **S_W66.3** Use `cudaStreamBeginCapture` / `cudaStreamEndCapture` / `cudaGraphInstantiate` / `cudaGraphLaunch`.
- **S_W66.4** Cache instantiated graphs keyed by (rule_id, schema_signature); re-capture only on schema change.
- **S_W66.5** Bench m37c-prime Stage 4 wall-time; gate M_W66.1.
- **S_W66.6** Determinism cert: 100 runs with Graph capture enabled; bit-exact.

**Acceptance.** All M_W66.* gates met.

---

### 3.10 G_W39_DTSDLM — Phase-1 G_W39 harness extension

**Goal.** Analyze the Phase-1 G_W39 paper-class harness for the purpose of adding a DTS-DLM-analog fixture module with respect to faithful reproduction of m37c-prime rule-shape distribution from the viewpoint of bundle composed-performance validation in the context of bridging paper-class to consumer-class.

**Questions.**
- **Q_W39D.1** Does m37c-prime rule-shape distribution (from G_PRE.2) translate into a synthetic fixture exercising the bundle?
- **Q_W39D.2** Does the DTS-DLM-analog achieve ≥ 5× speedup vs baseline?
- **Q_W39D.3** Is harness extension via Phase-1 M_W39.7 pluggable API surface-clean?

**Metrics.**

| Metric | Definition | Gate | Stretch |
|---|---|---|---|
| **M_W39D.1** DTS-DLM-analog fixture committed: `dts_dlm_analog(scale)` | 1/1 | — |
| **M_W39D.2** Fixture exercises bundle paths: G1 metadata, G2 branch (or graceful-flag), G3 branch (or graceful-flag), G4 helper-split, G5 multi-stream, G_W63 chain promoter (if HIGH/MEDIUM), G_W66 CUDA Graphs | 7/7 paths or graceful-flag | — |
| **M_W39D.3** DTS-DLM-analog speedup vs `main @ f62188b7` | ≥ **5.0×** | ≥ **10.0×** |
| **M_W39D.4** Geo-mean across 4 fixtures (3 paper-class + 1 DTS-DLM-analog) | ≥ **5.0×** | ≥ **10.0×** |
| **M_W39D.5** Determinism: row-set-equal vs `main` reference | bit-exact | — |
| **M_W39D.6** Reproducibility: CV across 10 runs | ≤ 5% | — |
| **M_W39D.7** No harness-driver changes (extension via pluggable API only) | 0 driver edits | — |
| **M_W39D.8** Peak VRAM | ≤ 38 GB | — |

**Strategies.**
- **S_W39D.1** New module `crates/xlog-integration/benches/fixtures/dts_dlm_analog.rs`:
  - Parameters from G_PRE.2 rule-shape histogram
  - u64 IDs from Zipfian α=2.5 over 1024-entry codebook (DTS-DLM VQ codebook size)
  - Rule mix synthesized to match G_PRE.2 proportions (chain-2 / triangle / cycle-4 / clique-K / recursive)
  - 4-column relations `(id1, id2, pro f32, contra f32)` matching DTS-DLM WMIR (only `id1`,`id2` joined; pro/contra projected payload)
- **S_W39D.2** Add via Phase-1 M_W39.7 pluggable-module API. NO driver edits.
- **S_W39D.3** Bench against pre-bundle baseline saved during Phase 1.

**Acceptance.** All M_W39D.* gates met.

---

### 3.11 G_M37A_SURFACE — M18 / M37-A neural-symbolic training surface preservation cert

**Goal.** Analyze the xlog v0.4.0-alpha neural-symbolic training surface (`nn/4`, `register_network`, `forward_backward_tensor`, `train_epoch`, XGCF, circuit caching) for the purpose of certifying functional + behavioral + performance preservation on Phase-2 HEAD with respect to KPI-8 + process lock 11 Group B from the viewpoint of M37-A queued-consumer readiness in the context of post-v0.6.5 DTS-DLM milestone progression.

**Scope discipline.** G_M37A_SURFACE is **cert-only**, NOT new development. M37-A itself is DTS-DLM-side work executed POST v0.6.5 tag (per `dts-dlm/docs/plans/2026-05-11-m37c-prime-narrow-recalibration-plan-freeze.md` §8 unconditional queue). Phase 2's responsibility is preserving the surface, not consuming it. Any cert finding that requires xlog code changes escalates to supervisor and may invoke a separate post-Phase-2 fix bundle rather than expanding goal-039 scope.

**Anchor.** M18 phases A–D verified configurations:
- M18-B: 3 `nn/4` predicates (predicate, arg0, arg1) composed via `fact_compiled(X,P,A0,A1)` — PASS
- M18-C: AUROC gains 0.626 / 0.161 / 0.703 on 3 slots; attribution-cleanliness held — PASS
- M18-D: AUROC gains 0.387 / 0.387 / 0.463 with ≥ 99.8% coverage on expanded label space (15 preds / 44 entities) via warmup pre-pass — PASS

**Questions.**
- **Q_M37A.1** Does the M18-D verified `xlog_alpha_source.py` α-shape source emitter still compile + execute under Phase-2 xlog wheel?
- **Q_M37A.2** Do the M18-D PASS metrics (AUROC 0.387/0.387/0.463, coverage ≥99.8%) reproduce within ±5% on Phase-2 HEAD?
- **Q_M37A.3** Is the strict-GPU-native contract of `forward_backward_tensor` preserved (zero host reads during loss computation)?
- **Q_M37A.4** Is XGCF circuit caching preserved (100× speedup on repeated query — performance contract)?
- **Q_M37A.5** Do the 6 reference examples shipped with v0.4.0-alpha still pass (MNIST-Add 99.07% accuracy with addition-only supervision as canonical canary)?
- **Q_M37A.6** Are `nn/4` Datalog syntax, `register_network` PyTorch integration, and `train_epoch` batch API functional?

**Metrics.**

| Metric | Definition | Target |
|---|---|---|
| **M_M37A.1** | `xlog_alpha_source.py` compiles + executes against Phase-2 wheel without API errors | EXIT 0 |
| **M_M37A.2** | M18-D verified config reproduces AUROC gains within ±5% of baseline (0.387/0.387/0.463) | 3/3 within ±5% |
| **M_M37A.3** | M18-D coverage reproduces ≥ 99.8% | ≥ 99.8% |
| **M_M37A.4** | `forward_backward_tensor` zero-host-reads cert: CUDA event trace shows no `cudaMemcpyDtoH` during loss computation | 0 host reads |
| **M_M37A.5** (extended R8 per supervisor amendment 2026-05-17 + KPI-9 quantification) | XGCF circuit caching cert: (a) second call to same query returns ≥ 50× faster than first call (100× is paper-claim; 50× gate floor accounting for system variance); (b) cache scope verified per `program.session()` (cache misses across sessions even with same query — cert: 2 sessions, same query, both first-call latencies > 50× their respective second-call latencies); (c) cache survives `forward_backward_tensor()` invocations (cert: forward_backward followed by repeat query returns cache-hit); (d) cache-hit rate ≥ 95% on M18-D verified config 10-epoch replay (cert: counter on `Program::cache_hit_count` / `Program::cache_miss_count` ratio); (e) M37-F `train_and_promote` per-attempt query repeat hits cache (cert: same `train_and_promote` invocation's internal repeat queries show ≥ 50× speedup on second hit) | (a) ratio ≥ 50× per-query; (b) cross-session miss cert PASS; (c) post-forward-backward cache-hit cert PASS; (d) ≥ 95% hit rate; (e) `train_and_promote` internal repeat ≥ 50× |
| **M_M37A.6** | MNIST-Add reference example reproduces ≥ 99.0% accuracy (verbatim 99.07% claim from v0.4.0-alpha; allow 99.0% gate) | ≥ 99.0% |
| **M_M37A.7** | `nn/4` Datalog parser accepts `nn(name, [inputs], output, [labels]) :: predicate(args)` syntax | parse PASS |
| **M_M37A.8** | `register_network(name, nn_module, optimizer)` accepts a `torch.nn.Module` + `torch.optim.Optimizer` and registers the network for forward + backward | registration PASS + retrieval PASS |
| **M_M37A.9** | `train_epoch(queries, batch_size=32)` runs an epoch and produces a loss trajectory | trajectory non-empty + decreasing on canonical MNIST-Add fixture |
| **M_M37A.10** (extended R11 per supervisor amendment 2026-05-17 — full Group B symbol enumeration) | Symbol-preservation cert (smoke test): if no Phase-2 production call site references a Group B symbol, the smoke test in `crates/xlog-integration/tests/test_m37a_surface_preservation.rs` exercises it; G_PURGE2 cannot remove the symbol. **Smoke test MUST instantiate each of the following Group B symbols at least once** (per lock 11 Group B expanded under R7): `nn/4` Datalog syntax parse + compile; `register_network(name, nn::Linear, optim::Adam)` roundtrip; `forward_backward_tensor(query)` returning CUDA tensor; `train_epoch(queries, batch_size=8)` running one epoch; XGCF query producing gradient; circuit caching ≥ 50× per-query speedup; P(Query|Evidence) probabilistic query returning per-query gradient; `register_embedding(name, nn::Embedding)` roundtrip; training-controls APIs (gradient-clipping via `register_network` with `max_grad_norm`; early-stopping via `train_epoch` patience param; scheduler via `register_network` with `StepLR`; LR override via `register_network` keyword arg); `bounded_exact_induce(program, examples, budget)` returning at least one rule. **11 explicit instantiations**, one per Group B symbol family. Source-audit cert: every Group B symbol in lock 11 list appears at least once in the smoke test source file. | 11/11 instantiations PASS; source-audit cert PASS |

**Strategies.**
- **S_M37A.1** Cut `feat/m37a-surface-cert-g39` from `feat/w6-bundle-integration` HEAD (after G_W39_DTSDLM merge, before G_INT2 final gate).
- **S_M37A.2** Build Phase-2 xlog wheel via `maturin develop --release`.
- **S_M37A.3** Run `xlog_alpha_source.py` (M18 source emitter on DTS-DLM side, read-only) against Phase-2 wheel; capture pass/fail + any API drift.
- **S_M37A.4** Reproduce M18-D config on the canonical M18-D fixture; measure AUROC + coverage; compare against `dts-dlm/docs/plans/2026-05-*-m18d-*.md` baseline.
- **S_M37A.5** Add xlog-side cert: `crates/xlog-integration/tests/test_m37a_surface_preservation.rs` exercises the 10 metrics above.
- **S_M37A.6** Run 6 reference examples from `.worktrees/v0.4.0-alpha-integrated` (or equivalent path; verify worktree exists per project memory).
- **S_M37A.7** XGCF circuit caching: time first vs second invocation of identical query; assert ratio ≥ 50×.
- **S_M37A.8** Document M37-A readiness in `docs/evidence/2026-05-14-g39-m37a-surface-preservation/report.md` with reproduced metrics + any deviations from M18 baseline.

**Acceptance.** All M_M37A.* gates met. G_M37A_SURFACE GREEN gates G_INT2's final regression sweep.

**Out-of-scope (M37-A consumer-side work, NOT in goal-039):**
- M37-A plan-freeze authoring (DTS-DLM side; references `04-FINAL-REPORT.md` lines 176-186 and §8 of m37c-prime plan-freeze)
- M37-A bridge architecture (LearnedBridge from M21-0 or new design)
- M37-A training-loop wire-up (`target_committed_fact(F)` supervision query)
- M22 100-doc training corpus + M25 corpus_n500 evaluation runs
- 5-label verdict assignment (RECOVERS / PARTIAL / NO_EFFECT / REGRESSES / STRUCTURAL_NULL)
- Belnap-aware reward extension (M37-B)

---

### 3.12 G_INT2 — Phase-2 integration gate

**Goal.** Analyze the Phase-2-integrated bundle for the purpose of verifying composition-time correctness with respect to ALL prior closure metrics (W3.4, W4.1, W5.1, W5.2, W2.5) regression-free + workspace cleanliness + VRAM safety + memory pool sizing from the viewpoint of Phase-2 DoD gate in the context of pre-closure-proposal validation.

**Questions, Metrics, Strategies** parallel goal-038 §5.4 G_INT, applied to `feat/w6-bundle-integration` HEAD with Phase-2 sub-goals merged in.

**Metrics.**

| Metric | Target |
|---|---|
| **M_INT2.1** W3.4 successor re-validation post-Phase-2 (via `wcoj_w33_superhub` bench on the W3.4-canonical superhub-50K fixture; W3.4 original `wcoj_fusion_bench.rs` retired by Phase-1 G1 S1.4 per process lock 2) | ratio ≥ 1.51× (1.590× × 0.95) |
| **M_INT2.2** W4.1 cert regression | 3/3 PASS |
| **M_INT2.3** W5.1 cert trio EXACT counter/row-set match | 3/3 EXACT |
| **M_INT2.4** W5.2 bench corpus within ±10% | 36/36 |
| **M_INT2.5** W2.5 default-flip cert PASS | PASS |
| **M_INT2.6** Workspace fmt | EXIT 0 |
| **M_INT2.7** Workspace build `-D warnings` | EXIT 0 |
| **M_INT2.8** Workspace test | 0 fail |
| **M_INT2.9** CUDA cert suite (W5.3, W5.4 additions, full W6.x kernel surface) | 1/1 |
| **M_INT2.10** Peak VRAM on full bench + cert | ≤ 38 GB |
| **M_INT2.11** Per-stream pool sizing cert: `XLOG_WCOJ_POOL_MB_PER_STREAM` env honored; default 256 MB per stream; 4-stream × 4-arm 3.2 GB headroom validated | PASS |
| **M_INT2.12** DLPack zero-copy preserved cert: zero host transfers on hot path (Phase-2 changes don't introduce DtoH/HtoD) | PASS |
| **M_INT2.13** Witness-chain integrity cert: every Phase-2 sub-goal preserves `choice_sources` / `leaf_atoms` accessors functioning | PASS |

**Strategies.**
- **S_INT2.1** Create branch `feat/w6-bundle-integration` from `feat/w3-bundle-integration` HEAD.
- **S_INT2.2** Merge sub-goals in order: G_W53 → G_W54 → G_W61_DOC → G_W62_DOC → G_W63 (priority-gated) → G_W64 → G_W65 → G_W66 → G_W39_DTSDLM.
- **S_INT2.3** Run M_INT2.1–13 sequentially; stop on first failure; root-cause; fix on integration branch.
- **S_INT2.4** Per-stream pool sizing: new env `XLOG_WCOJ_POOL_MB_PER_STREAM` (default 256), enumerated 4-arm × 4-stream worst-case 3.2 GB headroom in `docs/wcoj-architecture-guide.md` (G_W61_DOC dep).

**Acceptance.** All M_INT2.* green.

---

### 3.13 G_PURGE2 — Phase-2 cross-cutting refactor + purge

**Goal.** Analyze the Phase-2-integrated codebase for the purpose of removing all dead code/comments/env vars/deps introduced by Phase 2 from the viewpoint of process locks 5 + 6 + Karpathy 3 in the context of pre-closure cleanup.

Inherits goal-038 §5.5 G_PURGE; applied on `feat/w6-bundle-integration` HEAD post G_INT2.

**Metrics.** M_PURGE2.1–9 parallel to goal-038 M_PURGE.1–9, scoped to Phase-2 touched files.

---

### 3.14 G_CLOSE2 — Phase-2 closure proposal + user approval + board update

**Goal.** Analyze Phase-2-integrated bundle for the purpose of obtaining user approval to (a) mark W5.3, W5.4, W6.1, W6.2 DONE on existing board entries AND (b) ADD new W6.3, W6.4, W6.5, W6.6 entries to the board as DONE from the viewpoint of process rule 1 in the context of Phase-2 closure.

**Questions, Metrics, Strategies.**

| Metric | Target |
|---|---|
| **M_CLOSE2.1** Closure proposal at `docs/plans/2026-05-XX-phase2-closure-proposal.md` | committed |
| **M_CLOSE2.2** User approval in thread | explicit "mark DONE + ADD new entries" |
| **M_CLOSE2.3** Board update commit: 4 OPEN→DONE flips (W5.3/W5.4/W6.1/W6.2) + 4 new DONE entries (W6.3/W6.4/W6.5/W6.6) | committed |
| **M_CLOSE2.4** Closure board reaches 0 OPEN excluding W7.1 release tag | board verified |
| **M_CLOSE2.5** W7.1 ready-state: every other board item DONE AND user authorization required before tag fires | hand-off explicit |
| **M_CLOSE2.6** Documented divergences carried forward from Phase 1 + Phase 2 | section in closure proposal |

---

### 3.15 G_E2E — DTS-DLM end-to-end validation (BG0 satisfaction)

**Goal.** Analyze m37c-prime pilot end-to-end on the Phase-2-integrated bundle for the purpose of confirming BG0 + KPI-1..KPI-7 from the viewpoint of organizational milestone closure in the context of v0.6.5 tag-readiness.

**Questions.**
- **Q_E2E.1** Does m37c-prime e2e wall-time achieve ≥ 2× speedup (gate) / ≥ 3× (stretch) vs `main @ f62188b7`?
- **Q_E2E.2** Does m37c-prime replay produce bit-exact output across 100 runs (KPI-2)?
- **Q_E2E.3** Do all pyxlog call sites in DTS-DLM function unchanged-and-semantically-equivalent (KPI-3 + lock 22)?
- **Q_E2E.4** Does DTS-DLM v2 serving regression test pass (lock 22)?
- **Q_E2E.5** Is peak VRAM within budget on full m37c-prime replay (KPI-5)?
- **Q_E2E.6** Is witness-chain recoverable for every WMIR-derived fact (KPI-6)?
- **Q_E2E.7** Is DLPack zero-copy preserved on Stage 4 hot path (KPI-7)?
- **Q_E2E.8** Is per-doc `LogicProgram.compile()` time ≤ 50 ms (A-13)?
- **Q_E2E.9** Is cross-doc session cleanup leak-free (A-12)?
- **Q_E2E.10** Is no-behavioral-drift cert PASS (lock 22)?

**Metrics.**

| Metric | Definition | Gate | Stretch |
|---|---|---|---|
| **M_E2E.1** | m37c-prime e2e wall-time speedup vs `main @ f62188b7` | ≥ **2.0×** (KPI-1) | ≥ **3.0×** |
| **M_E2E.2** | Determinism: bit-exact across **100** replays under fixed seed | 100/100 (KPI-2) | — |
| **M_E2E.3** | DTS-DLM API regression: `grep -r 'pyxlog\.' /home/dev/projects/dts-dlm/src/dts_dlm/` every call site unchanged + functional | 0 regressions (KPI-3) | — |
| **M_E2E.4** | DTS-DLM v2 serving regression: `pytest /home/dev/projects/dts-dlm/tests/test_serving_v2_baseline.py` (or equivalent) | 0 fail | — |
| **M_E2E.5** | xlog wheel install via maturin | EXIT 0 | — |
| **M_E2E.6** | Peak VRAM on full m37c-prime replay | ≤ 38 GB (KPI-5) | — |
| **M_E2E.7** | Witness-chain recoverability: for 50 m37c-prime docs, every WMIR-derived fact has non-empty `choice_sources` + `leaf_atoms` chain | 100% recoverable (KPI-6) | — |
| **M_E2E.8** | DLPack zero-copy: zero `cudaMemcpyDtoH` / `cudaMemcpyHtoD` outside `enrich_support_sorts` on Stage 4 hot path | 0 host transfers (KPI-7) | — |
| **M_E2E.9** | Per-doc `LogicProgram.compile()` median time | ≤ 50 ms | ≤ 25 ms |
| **M_E2E.10** | Cross-doc cleanup: `torch.cuda.memory_allocated` cycles per-doc with no accumulation across 50 docs | 0 cumulative growth | — |
| **M_E2E.11** | Behavioral-drift cert: capture reference output set on `main @ f62188b7` via pyxlog roundtrip; replay on Phase-2 HEAD; bit-exact | PASS | — |
| **M_E2E.12** | W4.1 explicit cert on Phase-2 HEAD (not just implicit regression) | 3/3 PASS | — |

**Strategies.**
- **S_E2E.1** Build xlog from `feat/w6-bundle-integration` HEAD; install into DTS-DLM venv via `maturin develop --release`.
- **S_E2E.2** Run m37c-prime arm-C pilot e2e on 50 docs; capture wall-time. Compare against baseline.
- **S_E2E.3** Run 100× determinism check with `XLOG_DETERMINISTIC=1` + fixed seed.
- **S_E2E.4** API regression: grep + DTS-DLM test suite.
- **S_E2E.5** DTS-DLM v2 serving baseline regression test.
- **S_E2E.6** VRAM snapshot via `cudaMemGetInfo` + `nvidia-smi --query-gpu=memory.used` during full replay.
- **S_E2E.7** Witness-chain cert: iterate `pipeline_run` output relations; for each non-base fact, assert `choice_sources` returns non-empty AND `leaf_atoms` returns non-empty.
- **S_E2E.8** DLPack cert: CUDA event tracing on Stage 4 hot path; assert no `cudaMemcpyDtoH`/`HtoD` calls except inside `enrich_support_sorts`.
- **S_E2E.9** Compile-time cert: `time.perf_counter_ns()` around `LogicProgram.compile()`; median across 50 docs.
- **S_E2E.10** Cross-doc cleanup: `torch.cuda.memory_allocated()` before + after each doc; assert no cumulative trend across 50 docs.
- **S_E2E.11** Behavioral-drift cert: pre-capture reference outputs on `main @ f62188b7` (one-time); replay on Phase-2 HEAD; BTreeSet bit-exact assertion.
- **S_E2E.12** Explicit W4.1 cert run on integration HEAD (independent of M_INT2.2 regression check).

**Acceptance.** All M_E2E.* gates met; KPI-1..KPI-7 satisfied.

---

### 3.16 G_TAG — W7.1 release tag handoff (NOT executed in Phase 2)

**Goal.** Analyze Phase-2-DONE state for the purpose of preparing W7.1 release-tag handoff with respect to user-authorization-required gate from the viewpoint of process rule 1 in the context of v0.6.5 final ship.

**Note.** Per process lock 21, Phase 2 does NOT execute the tag. G_TAG is a HANDOFF preparation step — produces tag-ready HEAD and the tag command — user then explicitly authorizes "tag v0.6.5" and tag fires in a separate user-driven action.

**Metrics.**

| Metric | Target |
|---|---|
| **M_TAG.1** Tag-ready HEAD: `feat/w6-bundle-integration` HEAD (or merged to main) with all certs green | HEAD SHA recorded |
| **M_TAG.2** Tag command artifact: `git tag -a v0.6.5 -m "<message>" <SHA>` written to handoff doc | command present |
| **M_TAG.3** Final closure-board snapshot: 0 OPEN, 0 IN-PROGRESS, all DONE | board verified |
| **M_TAG.4** User authorization in thread (user types literal "tag v0.6.5" or equivalent) | explicit | 

**Strategies.**
- **S_TAG.1** Confirm Phase-2 DONE state (§6 DoD met).
- **S_TAG.2** Write handoff doc `docs/plans/2026-05-XX-v065-tag-handoff.md` with tag command + HEAD SHA + final closure-board snapshot.
- **S_TAG.3** Submit to user; await explicit "tag v0.6.5"; tag fires user-side only.

**Acceptance.** Handoff doc present; user authorizes tag; tag pushed by user (not by agent).

---

## 4. Dependency DAG (Phase-2 execution order)

```
                            ┌──────────────────────────────────────┐
                            │ G_PRE (prerequisite — first)         │
                            └──────────────────────────────────────┘
                                            │
            ┌─────────┬─────────┬──────────┴──────────┬───────────┬──────────┬─────────┐
            ▼         ▼         ▼                     ▼           ▼          ▼         ▼
         G_W53     G_W54    G_W61_DOC             G_W62_DOC    G_W64      G_W65    G_W66
                                                                G_W63 (priority-gated by G_PRE)
                                            │
                                            ▼
                                      G_W39_DTSDLM
                                            │
                                            ▼
                                    G_M37A_SURFACE (Group B cert)
                                            │
                                            ▼
                              feat/w6-bundle-integration
                                            │
                                            ▼
                                         G_INT2
                                            │
                                            ▼
                                         G_PURGE2
                                            │
                                            ▼
                                         G_CLOSE2
                                            │
                                            ▼
                                         G_E2E
                                            │
                                            ▼
                                         G_TAG → user-authorized v0.6.5 tag
```

**Parallelization:** All sub-goals can proceed in parallel after G_PRE except W6.3 priority-gated, G_W39_DTSDLM (needs G_W63 outcome), and G_INT2 (requires all merges).

---

## 5. Definition of Done (Phase 2)

Phase 2 is DONE when ALL hold simultaneously:

1. **G_PRE complete:** report committed, decision rule outcome stated.
2. **Per-sub-goal metrics green or graceful-deferred:**
   - G_W53: 1/1, 3-way bit-exact, 100/100, 100/100, 100/100 (M_W53.1–6)
   - G_W54: M_W54.1–4 green
   - G_W61_DOC: M_W61D.1–5 green
   - G_W62_DOC: M_W62D.1–5 green
   - G_W63: M_W63.1–8 green OR graceful-deferred per G_PRE LOW outcome (user-authorized)
   - G_W64: M_W64.1–7 green
   - G_W65: M_W65.1–5 green
   - G_W66: M_W66.1–6 green
   - G_W39_DTSDLM: M_W39D.1–8 green
   - G_M37A_SURFACE: M_M37A.1–10 green (KPI-8 satisfied)
   - G_INT2: M_INT2.1–13 green
   - G_PURGE2: M_PURGE2.1–9 green
   - G_CLOSE2: M_CLOSE2.1–6 green
   - G_E2E: M_E2E.1–12 green (KPI-1..KPI-7 all satisfied)
   - G_TAG: M_TAG.1–3 green; M_TAG.4 awaits user
3. **Closure board state:** 4 OPEN→DONE flips (W5.3, W5.4, W6.1, W6.2) + 4 new DONE entries (W6.3, W6.4, W6.5, W6.6); 0 OPEN remaining except W7.1.
4. **User explicit DONE approval** in thread per process rule 1.
5. **W7.1 release tag** awaits user authorization (tag-ready, not tag-fired).
6. **DTS-DLM v3 ship-readiness:** BG0 satisfied per KPI-1..KPI-7.

---

## 6. Out-of-bounds (Phase 2)

Goal-037 §13 items 1–10 + goal-038 §7 items 11–14 in force. Phase-2 additions:

15. **No re-opening of Phase-1 W3 axis closure.** Phase-2 work revealing Phase-1 regression escalates; never silently reverts.
16. **No xlog tag pushes inside Phase 2.** v0.6.5 tag (W7.1) is user-gated, last action.
17. **No DTS-DLM repo mutations** except G_PRE instrumentation (reverted) + G_E2E wheel reinstall.
18. **No xlog-induce changes.** Stage 5 ILP frozen.
19. **No pyxlog API signature OR behavioral changes** beyond `deterministic=True` + seed-pin contract (lock 11 + 22).
20. **No new env vars beyond:** `XLOG_DETERMINISTIC` (if not present, add for seed-pin contract), `XLOG_WCOJ_POOL_MB_PER_STREAM` (per-stream pool), `XLOG_WCOJ_W63_CHAIN_ENABLE` (chain promoter kill switch ONLY IF Phase 2 needs it; default ON; remove before close if unused per lock 5). NO `_LEGACY`, NO `_FALLBACK`, NO `_DISABLE_*`.

---

## 7. Iteration protocol

### 7.1 Per-sub-goal loop

For each G-node:
1. Read; understand G, Q, M, S.
2. Bench-spike-first (where applicable): cut `bench-spike/<descriptor>-g39`; minimum-viable; gate; 3-redesign budget.
3. Production: cut `feat/<descriptor>-prod-g39` (or directly fold into integration for non-perf goals); apply G_PURGE2 incrementally.
4. Integration: merge into `feat/w6-bundle-integration`; run G_INT2 gates; fix regressions.

### 7.2 Phase-2 stop conditions

COMPLETE when §5 DoD 1–6 hold.

STUCK (escalate) when:
- Sub-goal spike fails ≥ 3 consecutive redesigns + no graceful-defer applicable.
- G_E2E M_E2E.1 < 2.0× after integration.
- KPI-3 violation (DTS-DLM API regression).
- KPI-6/KPI-7 violation (witness-chain or DLPack regression).
- W3.4/W4.1/W5.1/W5.2 regress on Phase-2 integration.

### 7.3 Self-evaluation checklist

```
[ ] Spike branch passed gate M_n.1 (where applicable)
[ ] Production branch implements ALL variants (u32, u64 where applicable)
[ ] Zero TODO/FIXME/XXX/HACK on touched files
[ ] Zero new Ok(None) decline for paper-aligned shapes
[ ] Zero new env vars beyond doc-allowed
[ ] Zero new cfg(test) gates on production
[ ] Pre-existing superseded paths removed
[ ] Tier-1 paper-citation comments on every Alg.2 line (G_W64 only)
[ ] Workspace gates green: fmt, build -D warnings, test
[ ] CUDA cert suite 1/1
[ ] M_n.* all green
[ ] W3.4 re-validation ≥ 1.51× (Phase-2 regression)
[ ] W4.1 3/3 PASS (Phase-2 regression)
[ ] W5.1 cert trio EXACT (Phase-2 regression)
[ ] W5.2 corpus within ±10% (Phase-2 regression)
[ ] DTS-DLM API unchanged (lock 11)
[ ] No behavioral drift (lock 22; G_E2E M_E2E.11 PASS)
[ ] Determinism 100/100 under seed-pin (lock 12; G_W53 explicit)
[ ] Sort labels on all output columns (lock 14; G_W65 explicit)
[ ] Possibilistic-vs-probabilistic preserved (lock 13)
[ ] DTS-DLM repo unchanged except authorized exceptions (lock 16)
[ ] Witness-chain recoverable (KPI-6; G_E2E M_E2E.7)
[ ] DLPack zero-copy preserved (KPI-7; G_E2E M_E2E.8)
[ ] M37-A Group B surface preserved (KPI-8; G_M37A_SURFACE M_M37A.1–10)
[ ] Peak VRAM ≤ 38 GB (KPI-5)
[ ] No co-authored-by trailers
[ ] No v0.6.6 references (except authorized W6.3 LOW-priority defer note if invoked)
[ ] G_PURGE2 applied to touched files
```

---

## 8. Dispatch instructions

```
/goal @docs/plans/2026-05-14-supervisor-goal-039.md
```

Tab → Enter to confirm. Never `C-c` on idle codex. `codex resume <UUID>` on death.

**Phase-2 launch precondition (CHECK BEFORE DISPATCH):**
- Phase 1 (goal-038) DONE: closure board W3 axis = 9/9 DONE
- Phase 1 closure proposal user-approved in thread
- `feat/w3-bundle-integration` HEAD SHA recorded in §0 predecessor state
- User has explicitly authorized Phase 2 dispatch

**First dispatch action:** G_PRE profiler trace. Implementer instruments DTS-DLM `xlog_executor.py`, runs 50-doc m37c-prime arm-C trace, emits `docs/evidence/2026-05-14-g39-pre-profiler-trace/report.md` with M_PRE.1/.2/.3 + decision-rule outcome (G_W63 HIGH/MEDIUM/LOW), reverts instrumentation.

---

## 9. Audit-findings mapping (validation amendments applied)

| Amendment | Severity | Section addressed |
|---|---|---|
| A-2: W5.1 + W5.2 regression gates | HIGH | M_INT2.3 + M_INT2.4 |
| **M37-A surface preservation** (post-M37-C′ informational addendum) | **HIGH** | Lock 11 Group B + Lock 18 explicit + KPI-8 + G_M37A_SURFACE §3.11 + M_M37A.1–10 |
| A-3: Explicit M_E2E.W4.1 (not just implicit) | MEDIUM | M_E2E.12 |
| A-4: M_E2E.DLPACK zero-host-transfer cert | HIGH | M_E2E.8 + KPI-7 |
| A-5: M_E2E.WITNESS witness-chain recoverability | HIGH | M_E2E.7 + KPI-6 |
| A-6: Process lock 22 No behavioral drift | MEDIUM | Lock 22 + M_E2E.11 |
| A-7: Stretch targets alongside gate floors | MEDIUM | G_W63 M_W63.1/2, G_W66 M_W66.1, M_W39D.3/4, M_E2E.1/9 |
| A-8: 100-run determinism (was 10) | MEDIUM | M_W53.3/4/5 + M_E2E.2 + KPI-2 |
| A-9: VRAM peak + growth metrics | HIGH | M_W63.8, M_W64.7, M_W65 (implicit via M_INT2.10), M_W66.5, M_W39D.8, M_INT2.10, M_E2E.6 + KPI-5 |
| A-10: Per-stream pool sizing | MEDIUM | M_INT2.11 + lock 20 env var |
| A-11: New W6.6 CUDA Graph capture | MEDIUM | Entire §3.9 G_W66 |
| A-12: M_E2E.CLEANUP cross-doc cleanup cert | LOW | M_E2E.10 |
| A-13: M_E2E.COMPILE per-doc compile time | LOW | M_E2E.9 |
| M-2: "G_W63_CHAIN is xlog-original, not paper-derived" disclaimer | LOW | §3.6 disclaimer block |

---

## 10. References

- **GQM paradigm:** https://en.wikipedia.org/wiki/GQM; Basili–Caldiera–Rombach (1994); Basili et al. (2007).
- **Paper:** arXiv:[2604.20073](https://arxiv.org/abs/2604.20073).
- **Predecessor (Phase 1):** `docs/plans/2026-05-14-supervisor-goal-038.md`.
- **Predecessor (W3 paper-alignment):** `docs/plans/2026-05-13-supervisor-goal-037.md`.
- **Closure board:** `docs/v065-closure-board.md`.
- **DTS-DLM:**
  - System design: `/home/dev/projects/dts-dlm/docs/dts-dlm-system-design.md`
  - ROADMAP: `/home/dev/projects/dts-dlm/ROADMAP.md`
  - Stage 4 hot loop: `src/dts_dlm/propagate/xlog_executor.py:1301-1450`
  - Stage 5 ILP: `src/dts_dlm/integrations/pyxlog/tensorized_ilp.py:452-728` + `ilp.py:115-200`
  - Persistent session: `src/dts_dlm/integrations/pyxlog/session.py`
  - Sort enrichment: `src/dts_dlm/propagate/xlog_executor.py:157`
  - DTS-DLM xlog usage dump: prior conversation message (Scenarios 1–4, contracts, math)
- **M37-A queued milestone (DTS-DLM next):**
  - `dts-dlm/docs/research/2026-05-08-pre-m37/04-FINAL-REPORT.md` — Stone #9 motivation (lines 99–119), M37-A proposal (lines 176–186), cost/risk/novelty (Medium / Medium / High)
  - `dts-dlm/docs/plans/2026-05-11-m37c-prime-narrow-recalibration-plan-freeze.md` §8 — M37-A queued unconditionally
  - M18 phases A–D plans + α-shape source emitter `dts-dlm/src/dts_dlm/learn/xlog_alpha_source.py`
- **Validation audit (goal-038/039):** prior conversation message (13 amendments A-1..A-13 + M-1/M-2) + M37-A surface-preservation amendment (this update)
- **Karpathy guidelines:** https://x.com/karpathy/status/2015883857489522876.

---

**End of Phase-2 goal document.** Implementer begins with G_PRE profiler trace. Supervisor awaits G_PRE report + decision rule outcome before authorizing G_W63/G_W64/G_W65/G_W66/G_W53/G_W54/G_W61_DOC/G_W62_DOC.
