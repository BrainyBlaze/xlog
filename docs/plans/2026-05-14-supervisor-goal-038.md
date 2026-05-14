# Supervisor Goal 038 — Phase 1: W3 Axis Closure (v0.6.5)

**Supervisor:** Claude Code.
**Implementer:** Codex CLI on tmux session `codex-xlog`.
**Predecessor:** `docs/plans/2026-05-13-supervisor-goal-037.md` (W3 paper-alignment bundle; partially closed at goal-038 cut).
**Successor:** `docs/plans/2026-05-14-supervisor-goal-039.md` (Phase 2: v0.6.5 final closure + DTS-DLM hot-loop completion).
**Date:** 2026-05-14.
**Methodology:** Basili–Caldiera–Rombach Goal/Question/Metric paradigm (GQM); GQM+Strategies extension. References: https://en.wikipedia.org/wiki/GQM; Basili V., Caldiera G., Rombach H.D. (1994) "The Goal Question Metric Approach"; Basili et al. (2007) "GQM+Strategies".
**Paper:** SRDatalog arXiv:[2604.20073](https://arxiv.org/abs/2604.20073).
**Closure board:** `docs/v065-closure-board.md`. Phase 1 advances **6 of 11 OPEN** items: W3.3, W3.5, W3.6, W3.7, W3.8, W3.9 from OPEN to DONE. Phase 2 (goal-039) closes the remaining 5 (W5.3, W5.4, W6.1, W6.2, plus newly-opened DTS-DLM items, plus W7.1 release-tag gate).

---

## 0. Predecessor state at Phase-1 cut

### 0.1 Goal-037 state (Codex audit `docs/evidence/2026-05-14-g37-stop-condition-audit/`)

| Goal-037 sub-goal | State | Branch / commit |
|---|---|---|
| G1 W3.3 HG block-slice | ✅ per-branch GREEN | `feat/w33-hg-block-slice-prod @ 035b0713` |
| G4 W3.7 helper-split AOT | ✅ per-branch GREEN | `feat/w37-helper-split-aot-g37 @ bfd80d67` |
| G5 W3.8 stream-mux AOT | ✅ per-branch GREEN | `feat/w38-stream-mux-aot-g37 @ 792cea72` |
| G2 W3.5 shmem narrow | ❌ §14.2 STUCK (3 RED redesigns on small cell: 0.501×, 0.521×, 0.498× on inherited `triangle-small-inner-4K` fixture; large cell: 0.554×, 0.510×, 0.526×) | `bench-spike/w35-*-g37` preserved unmerged |
| G3 W3.6 warp prims | ❌ RED (2 redesigns: 0.597×, 0.595×) | `bench-spike/w36-*-g37` preserved unmerged |
| G6 W3.9 paper-class harness | ❌ MISSING | — |
| G7 cross-cutting refactor | ❌ MISSING | — |
| W3.4 closure re-validation | ❌ NOT RUN | — |
| W4.1 final-bundle regression | ❌ NOT RUN | — |
| `feat/w3-bundle-integration` | ❌ DOES NOT EXIST | — |
| Closure proposal | ❌ NOT WRITTEN | — |
| Board update | ❌ NOT STARTED (correctly, per process rule 1) | — |

### 0.2 Closure board state (current)

11 OPEN items: W3.3, W3.5, W3.6, W3.7, W3.8, W3.9, W5.3, W5.4, W6.1, W6.2, W7.1. 1 IN-PROGRESS: W1.1 (the board itself). 15 DONE. **Phase 1 closes 6 of 11 OPEN (the W3 axis).**

### 0.3 Validation findings applied to this document

This document incorporates all 5 HIGH/MEDIUM amendments identified in the goal-038/goal-039 validation pass (audit findings A-1, A-2, A-7, A-9, M-1; see §11 for full mapping).

---

## 1. Process locks (durable, inherited from goal-037)

Goal-037 process locks 1–10 remain in force verbatim. Phase 1 adds no new locks; DTS-DLM-specific extensions (locks 11–22) live in Phase 2 (goal-039 §0).

The 10 inherited locks summarized (see goal-037 §0 for full text):

1. No simplification clauses
2. No back-compat shims
3. No `Ok(None)` decline for paper-aligned shapes
4. No bench-gate substitution
5. No dead-code preservation (extends to bundle-touched files only; non-touched files flagged in followup doc, not removed)
6. No comment rot
7. No `Co-Authored-By` trailers
8. No `v0.6.6` references
9. Bench-spike-first
10. GQM+Strategies dispatch shape

---

## 2. Strategic context (GQM+Strategies organizational frame)

### 2.1 Phase-1 business goal

> **BG1.** Close the W3 axis of the v0.6.5 closure board by advancing W3.3, W3.5, W3.6, W3.7, W3.8, W3.9 from OPEN to DONE, producing an integrated `feat/w3-bundle-integration` branch HEAD that satisfies Phase-1 Definition of Done (§6 below), preserves W3.4/W4.1/W5.1/W5.2/W2.5 closure metrics regression-free, and is ready for hand-off to Phase 2 (goal-039).

### 2.2 Phase-1 assumptions

| ID | Assumption | Source |
|---|---|---|
| A1 | Goal-037 per-branch GREEN sub-goals (G1, G4, G5) remain GREEN after merge into integration branch. | Codex audit `docs/evidence/2026-05-14-g37-stop-condition-audit/` |
| A2 | Shared-memory narrowing (G_W35) and warp-cooperative prefix (G_W36) interventions per paper §5 Algorithm 2 lines 6–7 are **fan-out-dependent**; the inherited `triangle-small-inner-4K` fixture (p50 = 2 probes/row) cannot test the paper's claim. | Codex RCA in stop-condition audit; cross-walk against paper §5 |
| A3 | The 2× regression in goal-037 G2 spikes reflects kernel-wide `__syncthreads()` and branch-divergence cost from per-key-dispatch-within-kernel design, not pure fixture mismatch. | Codex resource diagnostic (`c53dce32`) rules out register/stack/local/big-shared-mem |
| A4 | The cached HG count kernel (`wcoj_triangle_count_hg_cached_u32`) is either dead code or a parallel production path; integration must resolve which (process lock 5 / 2). | G5 evidence README explicitly notes "production phase intentionally does not use" the cached variant |
| A5 | Paper §3.5 imperative (3) — launch-time load-balancer, no runtime coordination — implies graceful close is paper-aligned when per-key dispatch within a single kernel cannot satisfy it on current hardware. The HG kernel **without** per-key dispatch is still paper §5 Algorithm 2 outer block-slicing (lines 1, 3, 4, 5, 7, 9, 10, 12); only lines 6 and the per-warp narrowing are dropped. | Paper §5 reading; goal-037 §6 Q2.4 |

### 2.3 Strategy

> Close the W3 axis by (a) reopening G_W35/G_W36 with paper-class fixtures and parity guards, (b) building the production-scale harness G_W39 (paper-class fixtures + memory metrics + bundle-path assertion + extensibility for Phase 2 DTS-DLM-analog fixture), (c) running the integration gate G_INT that validates W3.4 + W4.1 + W5.1 + W5.2 regression-free composition + VRAM safety, (d) applying the cross-cutting G_PURGE that removes all bundle-introduced dead code, (e) writing the closure proposal G_CLOSE for the W3 axis. Each sub-goal has explicit graceful-close paths so the bundle cannot perma-stall on hardware-Amdahl limits.

### 2.4 Phase-1 KPI

> **KPI-P1.1:** Closure board W3 axis = 9/9 DONE (W3.1+W3.2+W3.3+W3.4+W3.5+W3.6+W3.7+W3.8+W3.9), board commit applied with user approval.
> **KPI-P1.2:** `feat/w3-bundle-integration` HEAD satisfies §6 DoD items 1–7.
> **KPI-P1.3:** W3.4 closure metric re-validated within ±5% of original 1.590× (≥ 1.51×).
> **KPI-P1.4:** W4.1 dispatch certs (multirec_triangle, multirec_4cycle, selfrec_triangle) PASS on integrated bundle.
> **KPI-P1.5:** W5.1 cert trio (Same Generation 4-cycle, skewed multiway triangle, deep-recursive triangle) PASS on integrated bundle with counters and row-sets matching pre-bundle exact values.
> **KPI-P1.6:** W5.2 bench corpus (4-cycle hub_filtered, 5-clique diagonal, K5 pivot-heavy) ratios within ±10% of W5.2 closure baseline cells.
> **KPI-P1.7:** Peak VRAM ≤ 80% of 48 GB (RTX 6000 Ada) on production-scale fixtures; cumulative per-iteration growth ≤ 1% for recursive fixtures.

---

## 3. Goal hierarchy (Phase-1 GQM tree)

```
BG1 — Close W3 axis (organizational)
 │
 ├── G_W35 — W3.5 shared-mem narrowing (reopened, paper-class fixture, parity guard, graceful-close)
 │   └─► G_W36 — W3.6 warp-level primitives (sequenced after G_W35)
 │
 ├── G_W39 — W3.9 production-scale harness (paper-class + VRAM metrics + pluggable for Phase 2)
 │
 ├── G_INT — Integration: W3.4 + W4.1 + W5.1 + W5.2 regression gates + workspace gates + VRAM gate
 │
 ├── G_PURGE — G7 cross-cutting refactor + dead-code/comment purge
 │
 └── G_CLOSE — W3-axis closure proposal + user approval + board update
```

6 sub-goals. Dependency DAG (§4 below).

---

## 4. Dependency DAG (Phase-1 execution order)

```
G_W35 ─────► G_W36 ─────┐
                         │
G_W39 ───────────────────┤
                         ▼
                     G_INT
                         │
                         ▼
                     G_PURGE
                         │
                         ▼
                     G_CLOSE
                         │
                         ▼
              Phase 1 DONE → hand-off to Phase 2 (goal-039)
```

**Parallelization:** G_W35 must precede G_W36 (G_W36 builds on G_W35's HG-shared-mem kernel as its baseline). G_W39 can run in parallel with the G_W35→G_W36 chain because G_W39 is harness work.

**Critical path:** G_W35 → G_W36 → G_INT → G_PURGE → G_CLOSE.

---

## 5. Per-goal GQM decomposition

> Each G-node follows Basili template: **Analyze** *object* **for the purpose of** *purpose* **with respect to** *quality* **from the viewpoint of** *viewpoint* **in the context of** *context*.

---

### 5.1 G_W35 — W3.5 shared-memory narrowing (reopened)

**Goal.** Analyze Algorithm 2 line-6 cooperative shared-memory narrowing for the purpose of evaluating speedup achievability with respect to gate ≥ 1.5× / stretch ≥ 3× on a paper-aligned fixture from the viewpoint of paper §5 fidelity in the context of Phase-1 stuck-state recovery.

**Predecessor state.** Goal-037 G2 RED (3 redesigns) on inherited fixture. Reopened with Modified Response 1 fixture amendment (Codex `docs/evidence/2026-05-14-g37-stop-condition-audit/response1_readiness.md`).

**Questions.**
- **Q_W35.1** Does Algorithm 2 line 6 yield speedup on a paper-aligned fixture where κ has non-trivial share (p50 ≥ 128 probes/row)?
- **Q_W35.2** Does the shared-memory path regress non-trivially on low-fanout regimes (p50 ≤ 8)?
- **Q_W35.3** Root cause of goal-037 G2 2× regression — kernel-wide `__syncthreads()`, branch divergence, or shared-memory bank conflicts?
- **Q_W35.4** Is graceful-close (revert to single global-path HG kernel) paper-aligned per §5 reading in A5?

**Metrics.**

| Metric | Definition | Gate | Stretch |
|---|---|---|---|
| **M_W35.1** | speedup on `triangle-line6-fanout-512` (p50 = 512, 262K rows, ≤ 12 KB per-key bracket) | ≥ **1.5×** | ≥ **3×** |
| **M_W35.2-PARITY** | speedup on `triangle-small-inner-4K` geometry (p50 = 2) as parity sub-cell | ≥ **0.95×** (paired Wilcoxon p < 0.05 across 10 trials) | — |
| **M_W35.3** | row-set equality vs reference hash-join output | bit-exact | — |
| **M_W35.4** | shared-memory occupancy per block | ≤ 32 KB | — |
| **M_W35.5** | `__syncthreads()` count per block per row | ≤ 3 (diagnostic) | — |
| **M_W35.6** | RCA artifact identifying root cause of goal-037 G2 regression | text in evidence README | — |
| **M_W35.7** | Peak VRAM during bench | report value; gate ≤ 80% of 48 GB (38 GB) | — |

**Strategies.**
- **S_W35.1** Cut `bench-spike/w35-line6-fanout-g38` from `feat/w33-hg-block-slice-prod @ 035b0713`. Worktree `.worktrees/w35-line6-fanout-g38`.
- **S_W35.2** Implement minimum-viable line-6 shared-memory narrowing per Codex readiness contract.
- **S_W35.3** Generate `triangle-line6-fanout-512` fixture: 512 root keys, 1 xy row per root, 512 z fanout per root, expected 262,144 output rows.
- **S_W35.4** Bench both cells (paper-class + parity); both must pass for spike GREEN. Statistical significance via paired Wilcoxon p < 0.05 across 10 trials.
- **S_W35.5** **Graceful-close path (paper-aligned per A5).** If spike fails M_W35.2-PARITY (< 0.95×) after one redesign: revert kernel to single global-path (no per-key dispatch within kernel); document M_W35.1 as **unreachable under current hardware-Amdahl constraint** with paper-citation `// Paper §5 Algorithm 2 lines 1,3,4,5,7,9,10,12 preserved; lines 6 + per-warp narrowing dropped per Phase-1 §2.2 A5 hardware constraint` in kernel header; mark G_W35 closed-as-graceful. **Closed-as-graceful is paper-aligned, not a simplification clause.**
- **S_W35.6** Spike-redesign budget: 3 redesigns on the new fixture (fresh count; the failure mode is fixture-driven per Codex RCA).
- **S_W35.7** If spike GREEN, cut production branch `feat/w35-line6-fanout-prod` from spike HEAD. Apply G_PURGE incrementally.

**Acceptance.** All M_W35.* gates met OR graceful-close per S_W35.5 with paper-citation justification.

---

### 5.2 G_W36 — W3.6 warp-level primitives (reopened)

**Goal.** Analyze Algorithm 2 line-7 cooperative warp `Prefix(x_j)` for the purpose of evaluating speedup achievability with respect to gate ≥ 1.3× / stretch ≥ 2× on a paper-aligned fixture from the viewpoint of paper §5 fidelity in the context of Phase-1 stuck-state recovery, building on G_W35.

**Predecessor state.** Goal-037 G3 RED (2 redesigns). Same recovery pattern as G_W35.

**Questions.**
- **Q_W36.1** Does cooperative warp `Prefix(x_j)` via `__shfl_sync` yield speedup vs G_W35 shared-mem path on a fixture with tunable per-warp fan-out?
- **Q_W36.2** Does warp coordination compose with G_W35's per-key tile, or interfere?

**Metrics.**

| Metric | Definition | Gate | Stretch |
|---|---|---|---|
| **M_W36.1** | speedup on `triangle-line7-warp-prefix-512` vs G_W35-only path | ≥ **1.3×** | ≥ **2×** |
| **M_W36.2-PARITY** | speedup on inherited fixture geometry (parity sub-cell) | ≥ **0.95×** (Wilcoxon p < 0.05) | — |
| **M_W36.3** | row-set equality | bit-exact | — |
| **M_W36.4** | warp coordination overhead per row | ≤ 5 cycles (diagnostic) | — |
| **M_W36.5** | Peak VRAM | ≤ 38 GB | — |

**Strategies.**
- **S_W36.1** Cut `bench-spike/w36-line7-fanout-g38` from passing G_W35 HEAD.
- **S_W36.2** Implement minimum-viable warp-cooperative `Prefix(x_j)` via `__shfl_sync(mask, val, src_lane)`.
- **S_W36.3** Same graceful-close as G_W35 with paper-citation: line 7 dropped under hardware-Amdahl; HG kernel without warp-cooperation still paper §5 Algorithm 2 outer.
- **S_W36.4** Spike budget: 3 redesigns.

**Acceptance.** All M_W36.* gates met OR graceful-close per S_W36.3.

---

### 5.3 G_W39 — W3.9 production-scale harness

**Goal.** Analyze the W3 bundle's composed performance (G1+G_W35+G_W36+G4+G5) for the purpose of evaluating production-scale viability with respect to gate ≥ 5× / stretch ≥ 10× geometric-mean speedup from the viewpoint of paper §7 evaluation methodology in the context of Phase-1 closure and Phase-2 extensibility.

**Predecessor state.** Goal-037 G6 MISSING.

**Questions.**
- **Q_W39.1** Does the composed bundle achieve geo-mean ≥ 5× speedup on 3 paper-class fixtures vs `main @ f62188b7`?
- **Q_W39.2** Is the speedup reproducible (CV ≤ 5% across 10 runs)?
- **Q_W39.3** Is row equality preserved on every fixture?
- **Q_W39.4** Does the harness accept pluggable fixture modules for Phase 2 DTS-DLM-analog addition?
- **Q_W39.5** Is peak VRAM within budget on production-scale fixtures?

**Metrics.**

| Metric | Definition | Gate | Stretch |
|---|---|---|---|
| **M_W39.1** | 3 paper-class fixtures committed: CallGraphEdge-analog, Andersen-analog, ddisasm-analog | 3/3 | — |
| **M_W39.2** | Bundle paths invoked per fixture: G1 metadata, G_W35 branch (or graceful-flag), G_W36 branch (or graceful-flag), G4 helper-split, G5 multi-stream | 5/5 per fixture | — |
| **M_W39.3** | Each fixture reports ratio vs `main @ f62188b7` | ratio reported | — |
| **M_W39.4** | Geometric mean speedup across 3 paper-class fixtures | ≥ **5.0×** | ≥ **10.0×** |
| **M_W39.5** | Determinism: row-set-equal on every fixture vs `main` reference | bit-exact | — |
| **M_W39.6** | Reproducibility: CV across 10 runs per fixture | ≤ 5% | — |
| **M_W39.7** | Harness accepts pluggable fixture modules | `add_fixture_module(module_path)` API exists | — |
| **M_W39.8** | Peak VRAM per fixture | ≤ 38 GB (80% of 48 GB) | — |
| **M_W39.9** | Cumulative per-iteration VRAM growth (recursive fixtures only) | ≤ 1% per iteration | — |

**Strategies.**
- **S_W39.1** New file `crates/xlog-integration/benches/wcoj_paper_class.rs`.
- **S_W39.2** New module `crates/xlog-integration/benches/fixtures/paper_class.rs` with three generators:
  - `call_graph_edge_analog(scale)` — power-law α≈2.5, hub degree ≈ 0.1·scale
  - `andersen_analog(scale)` — bipartite alloc/load/store/assign + field-sensitive granularity
  - `ddisasm_analog(scale)` — bidirectional dataflow + mutual recursion (W4.1 coverage)
- **S_W39.3** Each fixture asserts bundle paths via `bench_function` setup phase.
- **S_W39.4** Baseline: `git checkout f62188b7 && cargo bench --bench wcoj_paper_class -- --save-baseline pre-bundle`; bundle HEAD: `cargo bench --bench wcoj_paper_class -- --baseline pre-bundle`.
- **S_W39.5** Architectural extensibility: structure `fixtures/` as module directory; Phase 2 drops `dts_dlm_analog.rs` alongside without changing harness driver. Cert: `M_W39.7` API existence test.
- **S_W39.6** VRAM measurement via `cudaMemGetInfo` snapshot before/after each iteration; emit JSONL trace.

**Acceptance.** All M_W39.* gates met simultaneously.

---

### 5.4 G_INT — Integration + W3.4/W4.1/W5.1/W5.2 regression + workspace gates + VRAM safety

**Goal.** Analyze the integrated W3 bundle for the purpose of verifying composition-time correctness with respect to ALL prior closure metrics (W3.4, W4.1, W5.1, W5.2, W2.5) regression-free + workspace cleanliness + VRAM safety from the viewpoint of the Phase-1 DoD gate in the context of pre-closure-proposal validation.

**Predecessor state.** No `feat/w3-bundle-integration` branch exists.

**Questions.**
- **Q_INT.1** Does integration HEAD preserve W3.4 closure metric (≥ 1.51× on `wcoj_w34_kernel_fusion`)?
- **Q_INT.2** Do W4.1's 3 dispatch certs PASS?
- **Q_INT.3** Does the W5.1 cert trio PASS with pre-bundle counters and row-sets?
- **Q_INT.4** Does the W5.2 bench corpus stay within ±10% of closure baseline?
- **Q_INT.5** Does cached-kernel question resolve (`wcoj_triangle_count_hg_cached_u32` used / unused / unified)?
- **Q_INT.6** Does workspace gate pass on integration branch?
- **Q_INT.7** Is peak VRAM within budget across all certification fixtures?

**Metrics.**

| Metric | Definition | Target |
|---|---|---|
| **M_INT.1** W3.4 re-validation | `cargo bench --bench wcoj_w34_kernel_fusion` on integration HEAD | ratio ≥ **1.51×** (1.590× ± 5%) |
| **M_INT.2** W4.1 cert regression | `cargo test -p xlog-integration --test test_wcoj_recursive_dispatch multirec_triangle multirec_4cycle selfrec_triangle` | 3/3 PASS |
| **M_INT.3** W5.1 cert trio regression | `cargo test -p xlog-cuda-tests --test certification_suite --release` includes `same_gen_4cycle`, `skewed_multiway_triangle`, `deep_recursive_triangle` scenarios with EXACT counter values (1, 1, 6) and row-set sizes (14, 4, 4) from W5.1 closure | 3/3 EXACT match |
| **M_INT.4** W5.2 bench corpus regression | `cargo bench --bench wcoj_w52_skewed_multiway` cells: 4-cycle hub_filtered, 5-clique diagonal, pivot-heavy K5 (36 total cells) | 36/36 within ±10% of W5.2 closure baseline |
| **M_INT.5** W2.5 default-flip cert | `cargo test -p xlog-runtime test_w25_default_flip` | PASS (Cardinality is default; env skew opt-out preserved) |
| **M_INT.6** Cached-kernel resolution | `wcoj_triangle_count_hg_cached_u32` either deleted OR used by exactly one production path | 1 path OR deleted |
| **M_INT.7** Workspace fmt | `cargo fmt --check --all` | EXIT 0 |
| **M_INT.8** Workspace build with `-D warnings` | `RUSTFLAGS="-D warnings" cargo build --release --workspace --exclude pyxlog` | EXIT 0 |
| **M_INT.9** Workspace test | `cargo test --workspace --release --exclude pyxlog --exclude xlog-cuda-tests` | 0 fail |
| **M_INT.10** CUDA cert suite | `cargo test -p xlog-cuda-tests --test certification_suite --release` | 1/1 (206 internal scenarios) |
| **M_INT.11** Peak VRAM on cert + bench | `cudaMemGetInfo` snapshot during longest cert + each bench fixture | ≤ 38 GB |

**Strategies.**
- **S_INT.1** Create branch `feat/w3-bundle-integration` from `main @ f62188b7`.
- **S_INT.2** Merge in order: `feat/w33-hg-block-slice-prod` → `feat/w37-helper-split-aot-g37` → `feat/w38-stream-mux-aot-g37` → `feat/w35-line6-fanout-prod` (or graceful-close artifact) → `feat/w36-line7-fanout-prod` (or graceful-close artifact) → `feat/w39-paper-class-bench`.
- **S_INT.3** Run M_INT.1 → M_INT.11 sequentially; stop on first failure; root-cause; fix on integration branch; re-run from failed metric.
- **S_INT.4** Cached-kernel resolution per process locks 2 + 5: grep production call sites; if zero → delete kernel + provider entry + manifest entry + tests; if non-zero → identify single production path; if both G1 and another use it → unify. Document in `docs/evidence/2026-05-14-g38-int-cached-kernel-resolution.md`.

**Acceptance.** All M_INT.* green on `feat/w3-bundle-integration`.

---

### 5.5 G_PURGE — G7 cross-cutting refactor + dead-code/comment purge

**Goal.** Analyze the integrated bundle codebase for the purpose of removing all dead code/comments/env vars/deps introduced by the bundle from the viewpoint of process locks 5 + 6 + Karpathy guideline 3 in the context of pre-closure cleanup.

Inherited from goal-037 §10. Re-applied because goal-037 G7 never ran (no integration branch).

**Questions.** Q_PURGE.1–8 verbatim from goal-037 §10.

**Metrics.**

| Metric | Target |
|---|---|
| **M_PURGE.1** `// TODO\|FIXME\|XXX\|HACK` count on bundle-touched files | 0 |
| **M_PURGE.2** `// removed\|added for\|see PR\|deferred` count on bundle-touched files | 0 |
| **M_PURGE.3** `cargo +nightly udeps --workspace --all-targets` unused deps | 0 |
| **M_PURGE.4** `RUSTFLAGS="-D dead_code -D unused_imports -D unused_variables" cargo build --workspace --all-targets --release` | EXIT 0 |
| **M_PURGE.5** `Co-Authored-By` in bundle commits | 0 |
| **M_PURGE.6** `v0.6.6` references in bundle files/commits | 0 |
| **M_PURGE.7** Tier-1 paper-citation comments coverage on Alg.2-mapped kernel lines | 100% |
| **M_PURGE.8** `docs/evidence/2026-05-14-g38-dead-code-followup.md` flagging pre-existing dead code on non-touched files | exists ≥ 0 entries |
| **M_PURGE.9** Graceful-close paper-citation present in kernel headers if S_W35.5 / S_W36.3 invoked | citation pattern `// Paper §5 ... per Phase-1 A5` present |

**Strategies.** Inherit goal-037 §10 S7.1–S7.4 verbatim on `feat/w3-bundle-integration` HEAD after G_INT closes.

**Acceptance.** All M_PURGE.* green.

---

### 5.6 G_CLOSE — W3-axis closure proposal + user approval + board update

**Goal.** Analyze the integrated, purged bundle for the purpose of obtaining user approval to mark W3.3, W3.5, W3.6, W3.7, W3.8, W3.9 DONE on the closure board from the viewpoint of process rule 1 in the context of Phase-1 closure.

**Questions.**
- **Q_CLOSE.1** Does the closure proposal accurately enumerate every sub-goal's status (DONE / graceful-closed / metric-by-metric)?
- **Q_CLOSE.2** Does the user explicitly approve DONE marking in thread?
- **Q_CLOSE.3** Is the Phase-1 → Phase-2 hand-off documented?
- **Q_CLOSE.4** Is the documented §4 head/body merge divergence (paper-§4 head/body partition + Green-2012 single-pass path merge NOT implemented in Phase 1; pointer-to-future-work, not Phase-2 scope) explicitly recorded?

**Metrics.**

| Metric | Target |
|---|---|
| **M_CLOSE.1** Closure proposal exists | `docs/plans/2026-05-XX-w3-bundle-closure-proposal.md` committed |
| **M_CLOSE.2** User approval in thread | explicit "mark DONE" or equivalent |
| **M_CLOSE.3** Board update commit applies OPEN→DONE for W3.3, W3.5, W3.6, W3.7, W3.8, W3.9 | committed |
| **M_CLOSE.4** Phase-2 hand-off note | hand-off section in closure proposal naming integration HEAD SHA + W3.4 invariants Phase 2 must preserve + ready-for-goal-039 statement |
| **M_CLOSE.5** Documented divergences section | §4 head/body merge + any graceful-close path invocations documented under "Documented divergences — not blocking v0.6.5, flagged for v0.7+" |

**Strategies.**
- **S_CLOSE.1** Write closure proposal using goal-037 §14.4 template with Phase-1 final SHAs + metric outcomes.
- **S_CLOSE.2** Update `docs/evidence/2026-05-07-w3-paper-alignment-audit/README.md` with "BUNDLE CLOSED (PHASE 1)" final-commit note.
- **S_CLOSE.3** Submit to user; await explicit DONE approval; on approval, board-update commit.
- **S_CLOSE.4** Confirm integration HEAD SHA is referenced in goal-039 §0 predecessor state before authorizing Phase-2 dispatch.

**Acceptance.** All M_CLOSE.* green; board flipped; Phase 2 cleared.

---

## 6. Definition of Done (Phase 1)

Phase 1 is DONE when ALL hold simultaneously:

1. **Per-sub-goal metrics green or graceful-closed:**
   - G_W35: M_W35.1–7 gates met OR graceful-close per S_W35.5
   - G_W36: M_W36.1–5 gates met OR graceful-close per S_W36.3
   - G_W39: M_W39.1–9 gates met
   - G_INT: M_INT.1–11 green
   - G_PURGE: M_PURGE.1–9 green
   - G_CLOSE: M_CLOSE.1–5 green
2. **KPI satisfaction:** KPI-P1.1 through KPI-P1.7 all hold.
3. **User explicit DONE approval** in thread per process rule 1.
4. **W7.1 release tag remains user-gated** (Phase 2 closes W5.3/W5.4/W6.1/W6.2 + new W6.3–W6.6; W7.1 fires only after both phases done).
5. **Phase 2 ready-state:** `feat/w3-bundle-integration` HEAD SHA durable, referenced in goal-039 §0.

---

## 7. Out-of-bounds (Phase 1)

Goal-037 §13 items 1–10 in force. Phase-1 additions:

11. **Phase 1 does NOT introduce W6.x closure board entries.** W6.3–W6.6 (new DTS-DLM items) are Phase 2 scope.
12. **Phase 1 does NOT modify DTS-DLM source.** Phase 2 G_PRE owns instrumented profiler trace.
13. **Phase 1 does NOT introduce DTS-DLM-analog fixture.** Phase 2 owns extension via M_W39.7 pluggable API.
14. **Phase 1 does NOT touch W5.3/W5.4/W6.1/W6.2 board entries.** Those are Phase 2 scope.

---

## 8. Iteration protocol

### 8.1 Per-sub-goal loop

For each G-node ∈ {G_W35, G_W36, G_W39, G_INT, G_PURGE, G_CLOSE}:
1. Read G-node's section; understand G, Q, M, S.
2. Bench-spike-first (where applicable): cut `bench-spike/<descriptor>-g38`; minimum-viable; gate; 3-redesign budget; failed spikes preserved.
3. Production phase: cut `feat/<descriptor>-prod`; full implementation; G_PURGE incrementally.
4. Integration: merge into `feat/w3-bundle-integration`; run G_INT gates; fix regressions.

### 8.2 Phase-1 stop conditions

COMPLETE when §6 DoD items 1–5 hold.

STUCK (escalate) when:
- G_W35 or G_W36 fail gate AND parity guard AND graceful-close rejected by supervisor.
- G_INT M_INT.1 fails W3.4 (> 5% regression).
- G_INT M_INT.2 regresses W4.1.
- G_INT M_INT.3 regresses W5.1 cert counters/row-sets.
- G_INT M_INT.4 regresses W5.2 corpus > 10%.
- G_INT M_INT.11 exceeds 38 GB VRAM ceiling.

### 8.3 Self-evaluation checklist (per sub-goal)

```
[ ] Spike branch passed gate M_n.1 (where applicable)
[ ] Production branch implements ALL variants (u32, u64, triangle, 4-cycle, K5, K6 where applicable)
[ ] Zero TODO/FIXME/XXX/HACK on touched files
[ ] Zero new Ok(None) decline for paper-aligned shapes
[ ] Zero new env vars beyond doc-allowed
[ ] Zero new cfg(test) gates on production code
[ ] Pre-existing superseded code paths removed
[ ] Tier-1 paper-citation comments on every Alg.2 line
[ ] Graceful-close paper-citation present if invoked (paper §5 alignment justified)
[ ] Workspace gates green: fmt, build -D warnings, test
[ ] CUDA cert suite 1/1
[ ] W5.1 cert trio EXACT counter/row-set match
[ ] W5.2 bench corpus within ±10%
[ ] W3.4 re-validation ≥ 1.51×
[ ] W4.1 3/3 PASS
[ ] Peak VRAM ≤ 38 GB
[ ] M_n.* all green OR graceful-closed
[ ] No co-authored-by trailers
[ ] No v0.6.6 references
[ ] G_PURGE applied to touched files
```

---

## 9. Dispatch instructions

```
/goal @docs/plans/2026-05-14-supervisor-goal-038.md
```

Tab → Enter to confirm. Never `C-c` on idle codex. `codex resume <UUID>` if codex dies.

**First dispatch action:** G_W35 spike. Implementer cuts `bench-spike/w35-line6-fanout-g38` from `feat/w33-hg-block-slice-prod @ 035b0713`, implements minimum-viable line-6 narrowing, generates `triangle-line6-fanout-512` fixture, runs paired bench, reports M_W35.1 / .2-PARITY outcome.

---

## 10. Phase-1 → Phase-2 hand-off contract

| Artifact | Phase-2 consumer |
|---|---|
| `feat/w3-bundle-integration` HEAD SHA | All goal-039 G-nodes base |
| `docs/plans/2026-05-XX-w3-bundle-closure-proposal.md` | goal-039 §0 predecessor state |
| `crates/xlog-integration/benches/wcoj_paper_class.rs` + `fixtures/paper_class.rs` | goal-039 G_W39_DTSDLM fixture-extension |
| `docs/evidence/2026-05-14-g38-dead-code-followup.md` | goal-039 G_PURGE2 inputs |
| Closure board with W3 axis 9/9 DONE | goal-039 closes W5.3/W5.4/W6.1/W6.2 + adds W6.3–W6.6 |
| W3.4 / W4.1 / W5.1 / W5.2 regression-cert artifacts | goal-039 integration-gate inputs |

Phase 2 dispatches only after Phase-1 DoD met AND user authorizes goal-039 launch.

---

## 11. Audit-findings mapping (validation amendments applied)

| Amendment | Severity | Section addressed |
|---|---|---|
| A-1: Document §4 head/body merge as known divergence | LOW | M_CLOSE.5 + §5.6 |
| A-2: W5.1 + W5.2 regression gates added to integration | HIGH | M_INT.3 + M_INT.4 |
| A-7: Stretch targets alongside gate floors | MEDIUM | M_W35.1, M_W36.1, M_W39.4 (gate/stretch columns) |
| A-9: VRAM peak + growth metrics | HIGH | M_W35.7, M_W36.5, M_W39.8, M_W39.9, M_INT.11 |
| M-1: Paper-citation justification for graceful-close | LOW | A5 in §2.2; S_W35.5; S_W36.3; M_PURGE.9 |

---

## 12. References

- **GQM paradigm:** https://en.wikipedia.org/wiki/GQM; Basili–Caldiera–Rombach (1994); Basili et al. (2007) GQM+Strategies.
- **Paper:** arXiv:[2604.20073](https://arxiv.org/abs/2604.20073).
- **Predecessor goal:** `docs/plans/2026-05-13-supervisor-goal-037.md`.
- **Successor goal (Phase 2):** `docs/plans/2026-05-14-supervisor-goal-039.md`.
- **Closure board:** `docs/v065-closure-board.md`.
- **W3 paper-alignment audit:** `docs/evidence/2026-05-07-w3-paper-alignment-audit/README.md` on `feat/w3-paper-alignment-audit` (`134884fc`).
- **Goal-037 stop-condition audit (Codex):** `docs/evidence/2026-05-14-g37-stop-condition-audit/`.
- **Karpathy guidelines:** https://x.com/karpathy/status/2015883857489522876.

---

**End of Phase-1 goal document.** Implementer begins with G_W35 spike. Supervisor awaits G_W35 spike report.
