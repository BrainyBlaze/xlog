# Supervisor Goal 037 — W3 Paper-Alignment Bundle: Full SRDatalog Architectural Reconstruction (W3.3 + W3.5 + W3.6 + W3.7 + W3.8 + W3.9, anchored on W3.4)

**Supervisor:** Claude Code.
**Implementer:** Codex CLI on tmux session `codex-xlog`.
**Predecessor:** G36 W3.5 third re-spike (`feat/w35-css-tree-yz`) and prior W3.3 chain G11–G27 (preserved unmerged). All prior W3.3 attempts (G11–G27) and W3.5 spikes (G34/G35/G36) are **superseded** by this bundle goal. The bundle treats them as evidence of what local-optimization-on-the-wrong-kernel-shape cannot deliver; the bundle replaces the kernel shape.
**Date:** 2026-05-13.
**Paper:** "Scaling Worst-Case Optimal Datalog to GPUs" — Sun, Qi, Gilray, Kumar, Micinski. arXiv:[2604.20073](https://arxiv.org/abs/2604.20073). Audit: `docs/evidence/2026-05-07-w3-paper-alignment-audit/README.md` on branch `feat/w3-paper-alignment-audit` (`134884fc`).
**Closure board:** `docs/v065-closure-board.md` (HEAD `f62188b7`). This goal advances W3.3, W3.5, W3.6, W3.7, W3.8, W3.9 from `OPEN → DONE`. W3.4 stays `DONE` and is the preserved integration anchor.
**Goal-Driven Development framework:** Basili–Caldiera–Rombach Goal/Question/Metric paradigm + GQM+Strategies extension. See https://en.wikipedia.org/wiki/GQM.

---

## 0. Process locks (DURABLE — not subject to slice-internal revision)

These rules override any tactical decision made during implementation. They exist because prior W3.3 attempts (G11–G27) drifted toward local optimization of the wrong kernel shape. They will not drift again.

1. **No simplification clauses.** Phrases like *"specialised to triangle for v1"*, *"general-arity deferred"*, *"histogram deferred to v0.6.6"*, *"fallback kernel preserved"* are FORBIDDEN in goal-related code, comments, plans, evidence, or commits. The bundle either implements the paper's architecture in full at the bundle's stated arities (triangle + 4-cycle + 5-clique + 6-clique), or the bundle is not done.
2. **No back-compat shims.** No deprecated env vars, no `_v1`/`_legacy` symbols, no parallel kernel paths kept "in case the new one regresses". The new architecture replaces the old; the old code is removed from the codebase in the same series of commits that introduces the new code.
3. **No `Ok(None)` decline paths for paper-aligned dispatch shapes.** A WCOJ dispatch site that the paper says should match MUST match. Decline paths remain only for shapes the paper does not cover (i.e., shapes outside triangle/4-cycle/K5/K6/helper-split rules).
4. **No bench gate substitution.** Every sub-item's acceptance metric is an **absolute wall-time speedup ratio on a paper-class fixture**. Variance proxies, cycle counters, kernel-internal timers, and SM-utilization heuristics are diagnostic only, never gate-substituting (G29 amendment rejection precedent).
5. **No dead-code preservation.** Every code path, comment, env var, helper, struct field, and test fixture that the bundle's restructuring renders unused MUST be deleted in the same commit chain. Pre-existing dead code on bundle-touched files is in-scope for removal; pre-existing dead code on non-touched files is flagged in a `dead-code-followup.md` evidence note (not removed in this bundle).
6. **No comment rot.** Comments that explain WHAT the code does (vs. the non-obvious WHY), comments that reference removed code, comments that cite task IDs / PR numbers / "added for X", comments that describe deferred work — ALL removed from bundle-touched files. Surviving comments cite paper section/line or encode load-bearing constraints.
7. **No co-authored-by trailers.** Per `feedback_no_co_authored_by.md`.
8. **No `v0.6.6` references.** Per process rule 5 of `docs/v065-closure-board.md`.
9. **Bench-spike-first.** Every sub-item begins with a minimum-viable spike branch (`bench-spike/w3X-<descriptor>`) that measures the gate metric on the paper-class fixture using the proposed algorithm at minimum-viable scale. If the spike fails the gate, the implementer redesigns until a spike passes; failed spike branches stay unmerged as evidence. Only after spike passes does the implementer cut the production branch.
10. **GQM+Strategies dispatch shape.** All implementer-bound work is dispatched as S/Q/M codes into `codex-xlog` via `/goal @docs/plans/2026-05-13-supervisor-goal-037.md`. Implementer must reference S/Q/M codes from this file in every commit.

---

## 1. Goal hierarchy (Basili–Caldiera–Rombach GQM template)

### G0 — Bundle root goal

> **Purpose:** Restructure xlog's WCOJ runtime so that its production kernel pipeline is an instance of SRDatalog's Algorithm 1 + Algorithm 2 + helper-relation splitting + stream-aligned multiplexing — at full theoretical fidelity, on production-scale fixtures, in the canonical recursive Datalog path.
> **Issue:** The current production kernel (`wcoj_triangle_count` at `crates/xlog-cuda/kernels/wcoj.cu:240`) is a thread-per-driver-row scheme that implements only Algorithm 2 line 10 (leaf-variable intersect) faithfully; lines 1, 3, 4, 6, 7, 8 are absent or collapsed. The `wcoj_triangle_skew_histogram_u32` kernel at `wcoj.cu:793` is a per-call dispatch classifier, not Algorithm 1 Phase 1's persistent fan-out array. Helper-relation splitting (paper §5 Figure 3+5) is absent. Stream multiplexing (paper §6) is absent. The audit at `134884fc` already named these gaps; this bundle closes them.
> **Object of study:** The WCOJ kernel pipeline at `crates/xlog-cuda/{kernels,src/provider}/` and the executor dispatch at `crates/xlog-runtime/src/executor/`, the optimizer at `crates/xlog-logic/src/optimizer/`, and the recursive evaluator at `crates/xlog-runtime/src/executor/recursive.rs`.
> **Viewpoint:** A reader who has the SRDatalog paper open in another window MUST be able to read any WCOJ kernel, dispatch path, or optimizer pass in xlog and recognize it as a faithful instance of a specific paper algorithm, with the paper's section number and pseudocode line range cited in the source.

### G1 — W3.3 paper-aligned histogram-guided block scheduling
Anchor: paper §5 Algorithm 1 Phase 1 (lines 2–7) + Algorithm 2 lines 1, 3, 4, 6.

### G2 — W3.5 shared-memory optimization for small relations
Anchor: paper §5 Algorithm 2 line 6 "Narrow first-source handle to warp's share of key κ" + paper §4 storage SoA constraint.

### G3 — W3.6 warp-level cooperative primitives
Anchor: paper §5 Algorithm 2 line 7 "Prefix(x₁=U[κ]) on all sources ⊳ Cooperative".

### G4 — W3.7 helper-relation splitting (AOT rule-rewriter)
Anchor: paper §5 helper-relation splitting; Figure 3 (CallGraphEdge → HelpNT); Figure 5 (HeapAllocHelper 35.8× ablation).

### G5 — W3.8 stream-aligned rule multiplexing
Anchor: paper §6 in entirety (CALM theorem, phase-aligned schedule Count→Scan→Resize→Materialize across independent CUDA streams).

### G6 — W3.9 production-scale benchmark suite
Anchor: paper §7 evaluation methodology; §7.1 (≥977K-tuple DOOP-class fixtures); §7.3 (ablation methodology).

### G7 — Cross-cutting structural refactor + dead-code/comment purge
Anchor: this goal's process locks 5 + 6 + Karpathy guideline 3 (surgical changes; remove orphans created by your changes; mention pre-existing dead code).

---

## 2. Theoretical foundation — paper claims that constrain implementation

The following are the **load-bearing claims** from the paper. Any commit that contradicts one of them is a violation and is rejected on review.

### T1 — Paper §3.5 architectural imperatives

> (1) two-phase count-and-materialize with deterministic offsets, no atomic serialization;
> (2) flat sorted-column storage with cheap iterative delta merges, no auxiliary index rebuild;
> (3) launch-time load-balancer, no runtime coordination.

Imperative (1) is satisfied by current code (`wcoj_triangle_count` → `wcoj_compute_total` → `wcoj_triangle_materialize`). Imperative (2) is satisfied flatly but lacks the paper §4 head/body partition + Green-2012 single-pass path merge. Imperative (3) is violated — `wcoj_triangle_skew_histogram_*` is per-call.

### T2 — Paper §4 architecture

> "It first employs a Histogram phase to load-balance the outermost join (§5), followed by deterministic two-phase WCOJ rule evaluation (Count and Materialize)."

> "SRDatalog represents all relations strictly as flat, sorted arrays utilizing a Structure-of-Arrays (SoA) layout. Each relation is sorted lexicographically in radix order according to the column access sequence dictated by the query plan."

> "SRDatalog physically partitions each relation into a lightweight head buffer and a massive fully-sorted body. During the Merge phase, incoming deltas are cheaply integrated strictly into the small head array. SRDatalog only triggers a heavy, single-pass GPU path merge (Green et al., 2012) to flush the head into the body when a capacity threshold is reached."

### T3 — Paper §5 Algorithm 1: Work-Balanced GPU WCOJ

```
1: Relations R₁,…,Rₗ as sorted column arrays; variable order (x₁,…,xₘ); p blocks, w warps/block
2: Phase 1: Histogram (one warp per root key)
3:   U[0..K) ← unique values of x₁ across sources
4:   for all i ∈ [0,K) in parallel do
5:     W[i] ← Degree(U[i]) ⊳ Fan-out under root key i
6:   end for
7:   C ← PrefixSum(W); T ← C[K−1]
8: Phase 2: Count kernel (Alg. 2 with emit → count++)
9:   tc[…] ← per-thread output counts
10:  offsets ← PrefixSum(tc); allocate output to ∑tc
11: Phase 3: Materialize kernel
```

### T4 — Paper §5 Algorithm 2: HG-WCOJ Kernel (per block b)

```
1: Prefix sums C[0..K), total T, root keys U
2: Variable order (x₁,…,xₘ), sources Sⱼ for each variable xⱼ
3: [bs,be) ← block b's slice of [0,T) ⊳ ⌈T/p⌉
4: κ ← BinarySearch(C, bs) ⊳ Starting root key
5: for each key κ overlapping [bs,be) do
6:   Narrow first-source handle to warp's share of key κ
7:   Prefix(x₁=U[κ]) on all sources ⊳ Cooperative
8:   for each x₂ in Intersect(S[x₁→x₂]) across sources do
9:     Prefix(x₂); narrow child handles
10:    for each x₃ in Intersect(S[x₁,x₂→x₃]) do
11:      ⋯ ⊳ Recurse through variable order
12:      At leaf: emit(x₁,…,xₘ)
13:    end for
14:  end for
15: end for
```

### T5 — Paper §5 helper-relation splitting

> "By factoring only these specific clauses into an independent HelpNT relation, the previously buried skew keys (sn, dsc, h) are exposed as top-level columns in the newly generated rule."

Figure 5 reports HeapAllocHelper at 35.8× from this splitting alone, when histogram alignment was previously buried at an inner variable.

### T6 — Paper §6 stream-aligned rule multiplexing

> "SRDatalog's AOT compiler orchestrates a phase-aligned schedule that interleaves execution at the granularity of individual pipeline phases (Count → Scan → Resize → Materialize) across independent CUDA streams."

CALM-justified. Bites on rule-rich strata where individual rules don't saturate SMs.

### T7 — Paper §2 semi-naive (P1) — already satisfied by W4.1

> "ΔTC_{i+1} = π(ΔTC_i ⋈ Edge) ∖ TC_full"

W4.1 closure entry already records P1 alignment via `rewrite_scan_nth` occurrence-identity preservation. The bundle MUST NOT regress this.

### T8 — Paper P4 delta-outermost — documented divergence to RESOLVE

W4.1 closure note: *"Correctness preserved via row-set-parity certs; perf alignment partial; out of scope for W4.1 per D8 process locks; named at point of reference for any subsequent perf-focused W3.x or W4.x work."*

This bundle is exactly that perf-focused W3.x work. **G1 (W3.3) MUST resolve the divergence** by making the cost-model leader re-pick after Δ rewrite. The leader for a recursive rule must be the relation that minimizes `Degree(U)` over the actual Δ contents at iteration `i`, not the static stats from compile time.

---

## 3. Bundle architecture and dependency DAG

```
                       ┌──────────────────────────────────────┐
                       │  G7 — Cross-cutting refactor / purge │
                       │  (continuous; touches every sub-goal)│
                       └──────────────────────────────────────┘

   G1 (W3.3) ─────────┬──────────────► G2 (W3.5) ────► G3 (W3.6)
       │              │                                    │
       │              │                                    ▼
       │              └────────────────► G4 (W3.7) ──► G5 (W3.8)
       │                                     │            │
       │                                     ▼            ▼
       └───────────────────────────────► G6 (W3.9) ◄──────┘
                                              │
                                              ▼
                                  Bundle DONE (closure proposal → user)
```

**Critical path:** G1 → G6. G1 is the architectural pivot: it replaces the kernel shape from "thread-per-driver-row" to "block-slice over prefix-sum flattened root space". Every subsequent sub-goal builds on G1's new kernel shape:

- G2 (W3.5) shared-memory caches the warp-share narrowed in Algorithm 2 line 6. Cannot exist meaningfully without G1's block-slice.
- G3 (W3.6) implements cooperative warp `Prefix(x_j)` from Algorithm 2 line 7. Requires G1's per-block root-key context to know what `x₁` it is computing under.
- G4 (W3.7) elevates buried skew to the outermost variable so G1's launch-time slicer has skew to slice. Without G4, paper Figure 5's 35.8× is unreachable.
- G5 (W3.8) overlaps phase-aligned schedules across rules. Independent of G1's correctness but composes with G1's launch-param plumbing.
- G6 (W3.9) is the bench harness that exercises G1+G2+G3+G4+G5 on paper-class fixtures.

**W3.4 (DONE) preservation:** W3.4's layout+count fusion at threshold `XLOG_WCOJ_W34_THRESHOLD=4096` must compose with G1's HG kernel. The fusion targets the layout pre-pass; G1 changes the count kernel. The two are orthogonal **only if the W3.4 fused kernel is also restructured to consume Algorithm 1 Phase 1's `C` and `U` arrays**. G1's scope therefore EXTENDS to refactoring `wcoj_triangle_fused_lc_count` to the HG block-slice shape; W3.4's bench gate (superhub-50K ≥ 1.3×) must still hold post-refactor.

---

## 4. G1 — W3.3 paper-aligned histogram-guided block scheduling

### G1.G — Goal (GQM template)

> **Purpose:** Replace the thread-per-driver-row scheme with Algorithm 2's block-slice scheme, fed by Algorithm 1 Phase 1's persistent histogram metadata.
> **Issue:** Current per-call hash-bucket classifier produces a single skew score (`WCOJ_ADAPTIVE_SKEW_THRESHOLD=0.10`) that decides *whether* to route to WCOJ, not *how* to slice it. This is structurally insufficient; the skew score throws away the per-key fan-out distribution that Algorithm 2 line 4 needs.
> **Object:** `wcoj_triangle_count`, `wcoj_triangle_materialize`, `wcoj_triangle_fused_lc_count`, `wcoj_4cycle_count`, `wcoj_4cycle_materialize`, `wcoj_clique{5,6}_count`, `wcoj_clique{5,6}_materialize` (all variants — u32 and u64). The persistent histogram metadata lives on `CudaBuffer` (or a sibling `WcojRelationMetadata` struct on the per-relation handle in `crates/xlog-cuda/src/buffer.rs`).
> **Viewpoint:** A reader holding paper §5 Algorithm 2 sees `wcoj_triangle_count`'s body and matches it line-for-line to lines 1–15, with each pseudocode line annotated with `// Alg.2 line N` in the kernel source.

### G1.Q — Questions

- **Q1.1** Where does the histogram metadata live? On `CudaBuffer` directly, or on a parallel `WcojRelationMetadata` keyed by `RelId`?
- **Q1.2** How is the histogram updated during Merge? In-place incremental update of `W[]`, or full rebuild from `head ∪ body`?
- **Q1.3** What is the storage cost of `U[]`, `W[]`, `C[]` for a relation of size N? Paper does not bound this; xlog must measure.
- **Q1.4** For recursive rules, does the leader's `C` change every iteration? Yes — Δ changes the fan-out. The cost-model leader must re-pick post-rewrite (resolves T8).
- **Q1.5** What is the block-slice scheme: fixed `⌈T/p⌉` per block (paper), or work-stealing fallback for tail skew? Paper says fixed; xlog MUST follow paper.
- **Q1.6** How does Algorithm 2 line 6 ("Narrow first-source handle to warp's share of key κ") interact with the `lower_bound_u32` / `upper_bound_u32` device helpers? The warp narrowing produces a per-warp `(lo, hi)` range; current helpers do per-thread.

### G1.M — Metrics

| Metric | Target | Measurement |
|---|---|---|
| **M1.1** Speedup vs current main on canonical super-hub triangle fixture (50K driver, 1 hub key fanning to 4K) | **≥ 2.0×** wall-time | `cargo bench --bench wcoj_w33_superhub` (new harness; see G6) |
| **M1.2** No regression on uniform-50K triangle fixture | within **±5%** | same harness, uniform variant |
| **M1.3** Determinism: row-set-equal output vs current main on every fixture | bit-exact | `download_triples` BTreeSet equality cert |
| **M1.4** Algorithm 2 line correspondence: every line of `wcoj_triangle_count` body maps to an Alg.2 pseudocode line | 15/15 lines covered | source-audit cert (Tier-1 wrapper contract) |
| **M1.5** Histogram persistence: zero per-call histogram launches in the steady-state recursive loop | 0 launches per fixpoint iteration after iteration 1 | `wcoj_phase_report` feature counter; assert `histogram_launches == 1 + n_relations_with_delta` |
| **M1.6** Storage overhead of metadata per relation | **< 1%** of relation byte size for relations ≥ 977K rows | direct `CudaBuffer::byte_size` comparison |
| **M1.7** P4 delta-outermost: leader re-picks after Δ rewrite | leader_relid post-rewrite ∈ {Δ_relations} for every recursive rule | new cert `delta_outermost_leader_selection` |

### G1.S — Strategies

* **S1.1** Cut `bench-spike/w33-hg-block-slice` from `main @ f62188b7`. Spike implements the minimum-viable HG kernel: hard-coded triangle, hard-coded leader=XY, hard-coded `U[]`/`W[]`/`C[]` constructed by host. Measure superhub-50K under V3 paired-batching. **Gate: M1.1 ≥ 2.0×.** If spike fails, redesign — do not proceed to production.
* **S1.2** Production branch `feat/w33-hg-block-slice-prod` from `main`. Implement persistent metadata:
  - New struct `WcojRelationMetadata { pub unique_keys: CudaBuffer, pub fan_out: CudaBuffer, pub prefix_sum: CudaBuffer, pub total: u64 }` in `crates/xlog-cuda/src/wcoj_metadata.rs` (new file).
  - `CudaKernelProvider` gains `wcoj_build_metadata_recorded(&self, sorted_col: &CudaColumn, stream: &Stream) -> Result<WcojRelationMetadata>` — runs `wcoj_build_metadata_u32` and `_u64` kernels (new in `kernels/wcoj.cu`).
  - `wcoj_build_metadata_u32` kernel: one warp per key (paper Alg.1 line 2), computes unique values via segmented reduction on sorted column, writes `W[i]` via degree count, then `multiblock_scan_u32_inplace_on_stream` produces `C[]`.
* **S1.3** New kernel `wcoj_triangle_count_hg_u32`:
  - Block-grid sized `p = ceil(T / block_work_unit)`; `block_work_unit = 1024` initially, tunable via `XLOG_WCOJ_BLOCK_WORK_UNIT` (≤ 8192 ceiling enforced).
  - Line-for-line implementation of Algorithm 2 lines 1–15, with comments `// T4 Alg.2 line N: <verbatim pseudocode line>` on the kernel body.
  - The intersect at line 10 is delegated to existing `intersect_count`; this is the one pre-existing helper that already matched paper line 10.
* **S1.4** Replace `wcoj_triangle_count` call sites in `crates/xlog-cuda/src/provider/wcoj.rs` with `wcoj_triangle_count_hg_u32`. **DELETE** `wcoj_triangle_count` and `wcoj_triangle_fused_lc_count` (W3.4's fused variant is replaced by `wcoj_triangle_count_hg_u32` consuming `WcojRelationMetadata` produced lazily — single fused path, no threshold dispatch needed, env var `XLOG_WCOJ_W34_THRESHOLD` removed). The W3.4 closure metric (superhub-50K 1.590×) MUST be re-validated against the new fused HG path; bundle does not ship until that re-validation passes.
* **S1.5** Same restructure for `wcoj_triangle_materialize`, `wcoj_4cycle_count`/`_materialize`, `wcoj_clique{5,6}_count`/`_materialize` (u32 and u64 variants). No "triangle-only v1" — all kernels in the WCOJ family adopt the HG block-slice shape in this bundle. The clique template `wcoj_clique_template_count_t<K, T>` at `wcoj.cu:1187` is restructured to take `WcojRelationMetadata` for its outermost variable and dispatch block-slice over `C[]` instead of thread-per-row.
* **S1.6** Recursive integration: in `executor/recursive.rs`, after Δ is computed and merged, `provider.wcoj_build_metadata_recorded` is called on the Δ relation. The cost-model leader (W2.1's `WcojVariableOrderingModel`) re-picks using the post-merge metadata; the rewritten variant binds the chosen leader to the Δ relation per P4. New cert `delta_outermost_leader_selection` pins this.
* **S1.7** Remove `WCOJ_ADAPTIVE_SKEW_THRESHOLD`, `wcoj_triangle_skew_histogram_u32/u64`, `wcoj_triangle_skew_score_u32/u64`, `wcoj_4cycle_skew_histogram_u32/u64`, `wcoj_4cycle_skew_score_u32/u64`, `wcoj_skew_bucket_u32/u64`, the `SkewScoreSource` trait, the `TriangleScorer`/`Cycle4Scorer` impls, and the `ENV_USE_WCOJ_TRIANGLE_ADAPTIVE` / `ENV_DISABLE_WCOJ_TRIANGLE` env vars. These are the per-call dispatch classifier replaced by Algorithm 1 Phase 1. **All test files referencing these symbols are restructured or deleted.**

---

## 5. G2 — W3.5 shared-memory optimization for small relations

### G2.G — Goal

> Implement Algorithm 2 line 6's "Narrow first-source handle to warp's share of key κ" as a shared-memory cooperative load when the per-key fan-out fits within a configurable shared-memory tile.

### G2.Q — Questions

- **Q2.1** What is the threshold below which a relation's per-key bracket fits in `__shared__`? Conservative bound: 12 KB per warp share (out of 48 KB per block, 4 warps/block).
- **Q2.2** Does the threshold apply to the whole relation or per-key? Per-key, derived from `W[k]` from G1's metadata. A relation may have a mixed regime (some keys' brackets fit, others don't); the kernel dispatches per-key at line 6.
- **Q2.3** Should small-relation caching also cover the leader's `U[]`? Yes — when `|U|` ≤ 4096 entries × 8 bytes = 32 KB.

### G2.M — Metrics

| Metric | Target | Measurement |
|---|---|---|
| **M2.1** Speedup on small-relation triangle fixture (`triangle-small-inner-4K` from G36 baseline) | **≥ 1.5×** | `cargo bench --bench wcoj_w35_smallrel` |
| **M2.2** No regression above shared-mem threshold | within **±5%** on `triangle-large-yz-200K` | same harness |
| **M2.3** Determinism: row-set-equal | bit-exact | cert |
| **M2.4** Shared-memory occupancy: ≤ 32 KB per block at threshold | ≤ 32 KB | `cuOccupancyMaxActiveBlocksPerMultiprocessor` query in cert |

### G2.S — Strategies

* **S2.1** Spike `bench-spike/w35-shmem-narrow` from G1's production branch. Implement per-key shared-mem narrowing for `wcoj_triangle_count_hg_u32` only. Gate M2.1.
* **S2.2** If spike passes, fold into production branch `feat/w35-shmem-prod`. Extend to all kernel variants (4-cycle, K5, K6, u32+u64). NO "triangle-only v1".
* **S2.3** Threshold: compile-time const `WCOJ_SHMEM_NARROW_BYTES = 12288` (12 KB per warp share). Tunable via `XLOG_WCOJ_SHMEM_NARROW_BYTES` with ≤ 16384 ceiling.
* **S2.4** Per-key dispatch within kernel: `if (W[κ] * sizeof_slot ≤ WCOJ_SHMEM_NARROW_BYTES) shared_path else global_path`. NOT a separate kernel; same kernel, branch within. The cost of branch divergence at the per-key boundary is bench-measured (G6); if divergence cost exceeds shared-mem gain, the design returns to a single global-path kernel and the goal is closed with metric M2.1 marked **unreachable on current hardware**, NOT silently dropped.

---

## 6. G3 — W3.6 warp-level cooperative primitives

### G3.G — Goal

> Implement Algorithm 2 line 7 ("Prefix(x₁=U[κ]) on all sources ⊳ Cooperative") and line 9 ("Prefix(x₂); narrow child handles") as cooperative warp operations using `__shfl_*` intrinsics.

### G3.Q — Questions

- **Q3.1** Which warp intrinsic family is needed? `__shfl_sync` for value broadcast within warp; `__ballot_sync` for predicate-mask reductions; `__match_any_sync` for key-equality grouping.
- **Q3.2** Below what fan-out is warp-cooperative cheaper than global memory? Bench-measure under G6.
- **Q3.3** How does warp cooperation interact with G2's shared-memory narrowing? Composable: shared-mem tile loaded cooperatively by warp; subsequent warp lookups use shfl over the tile.

### G3.M — Metrics

| Metric | Target | Measurement |
|---|---|---|
| **M3.1** Speedup on small-relation triangle fixture (post-G2) | **≥ 1.3×** vs G2-only path | `cargo bench --bench wcoj_w36_warp` |
| **M3.2** No regression above warp-coop threshold | within **±5%** | same harness |
| **M3.3** Determinism: row-set-equal | bit-exact | cert |

### G3.S — Strategies

* **S3.1** Spike `bench-spike/w36-warp-coop` from G2's production branch. Implement warp-cooperative `Prefix(x_j)` for the y-level lookup (intermediate variable) where fan-out per warp is small. Gate M3.1.
* **S3.2** If spike passes, fold into production branch. Extend across all WCOJ kernels.
* **S3.3** No fallback to non-cooperative path inside the kernel; the kernel is the cooperative one. (A warp with `W[κ]` exceeding the cooperative threshold falls through to G2's shared-mem path; G2's path falls through to global. Single kernel, three nested regimes, all in-line.)

---

## 7. G4 — W3.7 helper-relation splitting (AOT rule-rewriter)

### G4.G — Goal

> Implement an AOT rule-rewriter pass in `xlog-logic::optimizer/` that detects rules where the skew-eligible variable is buried at an inner join level, splits the inner sub-pattern into an independent "helper relation", and rewrites the outer rule to consume the helper at the outermost variable position. Paper §5 Figure 3 + Figure 5; HeapAllocHelper 35.8× attribution.

### G4.Q — Questions

- **Q4.1** How does the rewriter detect "buried skew"? Compose: (a) variable-graph analysis from W2.2's `selectivity_pass` (`crates/xlog-logic/src/optimizer/selectivity_pass.rs`); (b) per-relation `Degree` statistics from G1's `WcojRelationMetadata`. A variable at depth ≥ 2 in the join tree with `max(Degree(U[k]))/avg(Degree(U[k])) ≥ HEAVY_SKEW_RATIO` is buried-skew.
- **Q4.2** What is `HEAVY_SKEW_RATIO`? Paper does not name a threshold. xlog measures across DOOP-class fixtures and chooses the minimum ratio at which Figure-5-style speedup activates. Conservative initial value: 10×.
- **Q4.3** Helper relation lifecycle: materialized once at AOT, or per-iteration in recursive context? Paper Figure 3 suggests AOT for non-recursive bodies; for recursive bodies, helper-relation materialization composes with semi-naive Δ.
- **Q4.4** How does helper-splitting interact with W4.1 (P1 occurrence semantics)? The split helper introduces a new predicate; if the original rule was self-recursive (paper P1 same-predicate occurrence), the split must preserve P1. Verified via selfrec triangle cert from W4.1.

### G4.M — Metrics

| Metric | Target | Measurement |
|---|---|---|
| **M4.1** Speedup on CallGraphEdge-analog fixture (6-variable deep-join with inner skew) | **≥ 2×** vs no-rewrite baseline | `cargo bench --bench wcoj_w37_helper_split` |
| **M4.2** Reproduce paper Figure 5 HeapAllocHelper attribution within an order of magnitude | **≥ 10×** on HeapAllocHelper-analog fixture | same harness |
| **M4.3** Determinism: row-set-equal between split and unsplit plans | bit-exact | cert |
| **M4.4** No regression on rules with no buried skew | within **±3%** | same harness, uniform fixture |
| **M4.5** Recursive composition: split rules in SCC-recursive bodies preserve P1 semantics | W4.1 cert suite still 3/3 PASS | regression cert |

### G4.S — Strategies

* **S4.1** Spike `bench-spike/w37-helper-split-hand` from G3's production branch. **Hand-write** the split rule for a single fixture (CallGraphEdge-analog). Measure M4.1 with hand-split plan vs unsplit. Gate ≥ 2×. **No automatic rewriter yet — the spike validates the algorithmic claim before xlog commits to the optimizer pass.**
* **S4.2** If hand-split spike passes, cut production branch `feat/w37-helper-split-aot`. Implement the optimizer pass:
  - New file `crates/xlog-logic/src/optimizer/helper_split_pass.rs`.
  - Public entry `helper_split_pass(plan: &mut LogicalPlan, stats: &StatsSnapshot) -> Vec<HelperRelationSpec>`.
  - Detection algorithm: walk the join tree bottom-up; at each join level, query `stats` (via G1's metadata) for skew ratio; if `ratio ≥ HEAVY_SKEW_RATIO`, mark the sub-tree at that level for extraction.
  - Extraction: replace marked sub-tree with `Scan(Helper_<n>)` in outer rule; emit a separate rule `Helper_<n>(vars) :- <marked sub-tree>` to be evaluated before the outer rule in the same stratum.
* **S4.3** Recursive composition: helper relations in recursive SCCs are evaluated with their own delta tracking; the outer recursive rule's Δ rewrite (W4.1's `rewrite_scan_nth`) treats the helper's predicate as just another body atom. Selfrec triangle cert from W4.1 re-runs against split plans.
* **S4.4** Bench fixtures for M4.1/M4.2 are NEW deliverables; see G6 for the corpus.

---

## 8. G5 — W3.8 stream-aligned rule multiplexing

### G5.G — Goal

> Implement paper §6's phase-aligned schedule. AOT compiler pass groups Count and Materialize kernels across independent rules within a stratum by phase, dispatching onto separate CUDA streams. CALM-justified.

### G5.Q — Questions

- **Q5.1** What counts as "independent rules"? Rules in the same stratum whose body atoms share no IDB write-target. (Inter-stratum is already serialized; intra-stratum independence is the CALM bite.)
- **Q5.2** How many streams? Paper does not name a number. Hardware-derived: `min(SM_count / 4, max_independent_rules_in_stratum)`. Initial value: 4.
- **Q5.3** Phase alignment: must Count kernels of rules A and B complete before either's Materialize starts? No — Materialize of A depends only on A's Count via prefix-sum; can interleave with B's Count. Paper schedule is Count(A) → Scan(A) ‖ Count(B) → Scan(B) ‖ Materialize(A) → Materialize(B), where ‖ denotes stream-parallel.
- **Q5.4** Memory allocation overlap: paper says "by actively overlapping independent compute and memory allocation stalls". Allocation is on the host via `Memory::alloc`; multi-stream allocation requires per-stream memory pools.

### G5.M — Metrics

| Metric | Target | Measurement |
|---|---|---|
| **M5.1** Speedup on rule-rich stratum fixture (3+ independent rules) | **≥ 1.27×** vs sequential dispatch | `cargo bench --bench wcoj_w38_stream_mux` |
| **M5.2** No regression on single-rule strata | within **±3%** | same harness |
| **M5.3** Determinism: row-set-equal across all rules vs sequential reference | bit-exact for every rule | cert |
| **M5.4** Hardware concurrency factor on rule-rich fixture | **≥ 1.2** (wall-clock / sum-of-kernel-times) | `cuEventElapsedTime` per kernel + total wall |

### G5.S — Strategies

* **S5.1** Spike `bench-spike/w38-stream-mux-hand` from G4's production branch. **Hand-schedule** 3 independent rules onto 3 streams for a single fixture; measure M5.1.
* **S5.2** If spike passes, cut production branch `feat/w38-stream-mux-aot`. Implement scheduler:
  - New file `crates/xlog-logic/src/optimizer/stream_schedule_pass.rs`.
  - Public entry `schedule_streams(stratum: &Stratum, hw: &HardwareCapabilities) -> StreamSchedule`.
  - Algorithm: dependency-graph rules by head-predicate writes; topologically sort with phase nodes (Count_r, Scan_r, Resize_r, Materialize_r per rule); assign streams via greedy bin-packing maximizing per-stream phase throughput.
* **S5.3** Provider stream API: `CudaKernelProvider` gains `wcoj_*_recorded_on_stream(&self, ..., stream: &Stream)` variants for every WCOJ entry. Existing `wcoj_*_recorded` keep their default-stream behavior; they call `wcoj_*_recorded_on_stream` with the provider's default stream. NOT a parallel code path — the default-stream variant is a single-line wrapper, no duplicated logic.
* **S5.4** Memory pools: per-stream `Arc<Memory>` allocator pool; pool size sized to stratum's worst-case rule allocation. Removed when bundle closes if pool overhead exceeds 5% of stream-mux gain.

---

## 9. G6 — W3.9 production-scale benchmark suite

### G6.G — Goal

> Build a production-scale benchmark harness exercising the full G1+G2+G3+G4+G5 bundle on paper-class fixtures (≥ 977K tuples, DOOP-class graph structure with natural heavy keys).

### G6.Q — Questions

- **Q6.1** Which paper benchmarks are in scope? Paper §7 mentions doop, ddisasm, polonius, andersen, CSPA, LSQB Q9, Same Generation. xlog implements analogs (not full reproductions — license / data ownership concerns).
- **Q6.2** Fixture generation: synthesize at scale, or import real graph data? Synthesize, with parameters tuned to match published statistics (vertex count, edge count, degree distribution skewness exponent).
- **Q6.3** Baseline for ratio: the bundle measures vs `main @ f62188b7` (pre-bundle). NOT vs CPU baselines (paper's 21-47× claim is not the bundle's claim).

### G6.M — Metrics

| Metric | Target | Measurement |
|---|---|---|
| **M6.1** Harness committed with ≥ 3 paper-class fixtures: CallGraphEdge-analog, Andersen-analog, ddisasm-analog | 3/3 fixtures present | `git ls-files crates/xlog-integration/benches/wcoj_paper_class_*` |
| **M6.2** Each fixture exercises full bundle (G1+G2+G3+G4+G5) | 5/5 sub-bundles invoked per fixture | bench output asserts paths taken |
| **M6.3** Each fixture reports ratio vs `main @ f62188b7` baseline | ratio reported on every run | bench harness output |
| **M6.4** Geometric mean speedup across 3 fixtures vs baseline | **≥ 5×** | derived from bench output |
| **M6.5** Determinism: row-set-equal on every fixture vs `main` reference run | bit-exact | cert |
| **M6.6** Reproducibility: re-running the suite yields ratios within ±5% | within ±5% across 10 runs | bench harness CV statistic |

### G6.S — Strategies

* **S6.1** New file `crates/xlog-integration/benches/wcoj_paper_class.rs` with criterion harness.
* **S6.2** New module `crates/xlog-integration/benches/fixtures/paper_class.rs` for fixture generators:
  - `call_graph_edge_analog(scale: usize) -> Vec<(Tuple, Predicate)>` — power-law degree distribution with α ≈ 2.5, hub degree ≈ 0.1 × scale.
  - `andersen_analog(scale)` — bipartite alloc/load/store/assign rules with field-sensitive granularity.
  - `ddisasm_analog(scale)` — bidirectional dataflow with mutual recursion (W4.1 coverage).
* **S6.3** Each fixture asserts (via `bench_function`'s setup phase) that bundle paths are taken: G1 metadata built, G2 shared-mem branch hit for at least one key, G3 warp-coop branch hit for at least one key, G4 helper-split active in plan, G5 multi-stream dispatch active.
* **S6.4** Baseline run: `git checkout f62188b7 && cargo bench --bench wcoj_paper_class -- --save-baseline pre-bundle && git checkout feat/w3-bundle-final && cargo bench --bench wcoj_paper_class -- --baseline pre-bundle`.

---

## 10. G7 — Cross-cutting structural refactor + dead-code/comment purge

### G7.G — Goal

> Every file touched by G1–G6 is left in a **paper-aligned, comment-purged, dead-code-free** state. No legacy parallel paths. No "compatibility" env vars. No comments that describe WHAT (only WHY). No co-authored-by trailers.

### G7.Q — Questions

- **Q7.1** What qualifies as "touched"? Any file with non-trivial edit in G1–G6's commit chain. Whitespace-only or rename-only touches are not in-scope for G7.
- **Q7.2** What qualifies as "dead code"? Per cargo: `cargo +nightly udeps` for unused deps; `cargo machete` for unused workspace deps; `RUSTFLAGS="-W dead_code" cargo build` for unused symbols. PLUS: env vars never read at runtime; trait impls with no live callers; test fixtures never referenced.
- **Q7.3** What qualifies as a "WHAT comment" to purge? Any comment that restates the code in English. Any comment of form `// removed X`, `// added for Y`, `// see PR #Z`, `// TODO`, `// XXX`, `// FIXME`. Any comment that references a deferred follow-up.
- **Q7.4** What WHY comments survive? Comments citing paper section/line (`// T4 Alg.2 line 6: Narrow first-source handle to warp's share of key κ`); comments encoding a non-obvious invariant (`// SAFETY: caller guarantees ...`); comments naming a specific historical bug that the code prevents (`// guards against pre-W3.3 per-call histogram cost`).

### G7.M — Metrics

| Metric | Target | Measurement |
|---|---|---|
| **M7.1** `cargo +nightly udeps --workspace --all-targets` reports zero unused dependencies | 0 | CI gate |
| **M7.2** `RUSTFLAGS="-D dead_code -D unused_imports -D unused_variables" cargo build --workspace --all-targets --release` exits 0 | exit 0 | CI gate |
| **M7.3** No `// TODO`, `// XXX`, `// FIXME`, `// HACK` on bundle-touched files | 0 occurrences | `grep -rE '// (TODO\|XXX\|FIXME\|HACK)' <touched-files>` |
| **M7.4** No `Co-Authored-By:` trailers in bundle commits | 0 | `git log <bundle-range> --grep 'Co-Authored-By'` |
| **M7.5** Every paper-aligned kernel line has either no comment or a `T<n> <claim>` paper-citation comment | 100% on bundle-touched kernel files | source-audit cert (Tier-1 wrapper contract) |
| **M7.6** Pre-existing dead code on non-touched files: flagged in `docs/evidence/2026-05-bundle-dead-code-followup.md`, not removed in this bundle | follow-up doc present, ≥ 0 entries | `git show <bundle-final>:docs/evidence/2026-05-bundle-dead-code-followup.md` |

### G7.S — Strategies

* **S7.1** After each sub-goal's production branch is merged into the bundle integration branch `feat/w3-bundle-integration`, run G7 enforcement: `cargo +nightly udeps`, `cargo machete`, `RUSTFLAGS="-D dead_code" cargo build`. Fix until green.
* **S7.2** After all sub-goals merge, scan bundle-touched files: `git diff --name-only main..feat/w3-bundle-integration | xargs grep -nE '// (TODO|XXX|FIXME|HACK)'`. Remove every hit or refactor the underlying code so the comment is no longer needed.
* **S7.3** Manual review pass on every bundle-touched kernel file: every comment is reviewed against the WHAT/WHY criterion in Q7.3/Q7.4. Removed unless WHY.
* **S7.4** Source-audit certs for Tier-1 paper-alignment: new test file `crates/xlog-cuda-tests/tests/paper_alignment_source_audit.rs` parses kernel source, asserts every Alg.2-referencing function has the expected `T4 Alg.2 line N:` comment skeleton on its body lines.

---

## 11. Karpathy guideline enforcement matrix

| Karpathy guideline | Bundle interpretation | Enforcement |
|---|---|---|
| **(1) Think Before Coding** — surface assumptions, push back on simpler approaches | Every sub-goal has explicit Q-section listing open questions; spike phase forces answers before production. If implementer believes a simpler design satisfies the paper, implementer surfaces the simplification to supervisor BEFORE writing code; supervisor evaluates against T1–T8 paper claims. | Spike-first protocol (process lock 9); supervisor review of every spike's branch HEAD before production branch is cut |
| **(2) Simplicity First** — minimum code, no speculative features | Every line in every commit traces to a specific G-Q-M-S code in this document. No "while I'm here, let me also …" edits. No abstractions for single-use code (e.g., no `WcojKernelLauncher` trait if there's only one launcher). | Commit-message convention: every commit references the G-Q-M-S code(s) it advances; commits failing this convention are rejected |
| **(3) Surgical Changes** — touch only what you must; clean up YOUR mess | Pre-existing dead code outside bundle-touched files is FLAGGED, not removed (M7.6). Bundle-created orphans are removed in the same commit (M7.1, M7.2). | G7 cross-cutting goal |
| **(4) Goal-Driven Execution** — define success criteria, loop until verified | Every G has Q + M; every M has a target value and a measurement command. Implementer loops the sub-goal's spike → production cycle until M values are green. | Sub-goal-level stop conditions in §13 |

---

## 12. Definition of Done (bundle-level)

The bundle is DONE when ALL of the following hold simultaneously:

1. **Sub-goal metrics green:**
   - M1.1 ≥ 2.0×, M1.2 within ±5%, M1.3 bit-exact, M1.4 15/15, M1.5 minimal, M1.6 < 1%, M1.7 cert passes
   - M2.1 ≥ 1.5×, M2.2 within ±5%, M2.3 bit-exact, M2.4 ≤ 32 KB
   - M3.1 ≥ 1.3×, M3.2 within ±5%, M3.3 bit-exact
   - M4.1 ≥ 2×, M4.2 ≥ 10×, M4.3 bit-exact, M4.4 within ±3%, M4.5 W4.1 certs still 3/3
   - M5.1 ≥ 1.27×, M5.2 within ±3%, M5.3 bit-exact, M5.4 ≥ 1.2
   - M6.1 3/3 fixtures, M6.2 5/5 paths, M6.3 ratios reported, M6.4 ≥ 5×, M6.5 bit-exact, M6.6 within ±5% CV
   - M7.1 0, M7.2 exit 0, M7.3 0, M7.4 0, M7.5 100%, M7.6 followup doc present

2. **W3.4 closure metric re-validated:** superhub-50K ≥ 1.3× post-G1 refactor (W3.4's original closure bound).

3. **W4.1 closure regression:** W4.1's 3 dispatch certs PASS, including selfrec triangle. T8 (P4 delta-outermost) divergence is RESOLVED (M1.7 cert).

4. **Workspace test count delta:** workspace pass count post-bundle ≥ pass count pre-bundle minus deletions for removed paper-misaligned tests. Net new tests: ≥ 30 (the bundle's certs). Net deletions: deletions for removed dispatch classifier tests, removed per-call histogram tests, removed fallback-kernel tests.

5. **All gates green:** `cargo fmt --check --all` exit 0; `RUSTFLAGS="-D warnings" cargo build --release --workspace --exclude pyxlog` exit 0; `cargo test --workspace --release --exclude pyxlog --exclude xlog-cuda-tests` 0 fail; `cargo test -p xlog-cuda-tests --test certification_suite --release` 1/1; `cargo bench --bench wcoj_paper_class -- --baseline pre-bundle` reports ratios meeting M6.4.

6. **Closure proposal:** `docs/plans/2026-05-XX-w3-bundle-closure-proposal.md` written and pushed; user explicitly approves DONE in thread; board-update commit applies `OPEN → DONE` for W3.3, W3.5, W3.6, W3.7, W3.8, W3.9 (W3.4 remains DONE, possibly with metric re-statement).

7. **Audit alignment:** `docs/evidence/2026-05-07-w3-paper-alignment-audit/README.md` (on `feat/w3-paper-alignment-audit`) is updated with a final "BUNDLE CLOSED" note pinning the bundle's final commit hash and the paper-section-to-code mapping for every Tier-1 kernel.

---

## 13. Out-of-bounds (bundle-level constraints)

The agent MUST NOT do any of the following. Doing any of these is a process violation and the supervisor rejects the work.

1. **Modify any of the following crates/files except where strictly required by paper-alignment:** `crates/xlog-prob/`, `crates/xlog-neural/`, `crates/xlog-solve/`, `crates/xlog-gpu/`, `crates/xlog-induce/`, `crates/pyxlog/`, `crates/xlog-stats/`, `crates/xlog-cli/`. The bundle is a WCOJ runtime restructure; it does not touch probabilistic, neural, solver, or Python paths.
2. **Modify W4.1 promoter or rewriter semantics** beyond what M1.7 (delta-outermost cert) strictly requires. The `rewrite_scan_nth` fix at `rewrite.rs:303-321 + :488-526` is W4.1's load-bearing P1 invariant; the bundle MUST NOT regress it.
3. **Add `cfg(test)` gating for production logic.** Per W2.3 closure precedent, test-only logic uses Cargo features (`recursive-stats-trace`, etc.). Production paths are unconditional.
4. **Add new env vars** beyond those strictly named in S1.4 (`XLOG_WCOJ_BLOCK_WORK_UNIT`), S2.3 (`XLOG_WCOJ_SHMEM_NARROW_BYTES`), S4.2 (`HEAVY_SKEW_RATIO` if exposed), S5.2 (`XLOG_WCOJ_STREAM_COUNT` if exposed). NO `_FALLBACK`, NO `_LEGACY`, NO `_DISABLE_<feature>` env vars. The new architecture is the architecture.
5. **Touch any file under `crates/xlog-cuda/src/provider/` other than the WCOJ-related ones** (`wcoj.rs`, `wcoj_dispatch.rs`'s sibling provider entries, `wcoj_metadata.rs` new). `provider/relational.rs`, `provider/ilp.rs`, `provider/mc.rs`, etc., are out of bounds.
6. **Bundle work into v0.6.6 milestone.** Per process lock 8: no `v0.6.6` references in any new file, comment, plan, evidence README, or commit message.
7. **Remove the `cargo bench --bench wcoj_phase_report` feature gate** (`wcoj-phase-timing`). Phase timing is diagnostic; production path is unconditional already. Leave the feature gate alone.
8. **Touch `crates/xlog-cuda-tests/tests/certification_suite/`** beyond adding new certs for the bundle. The 206-internal-scenario suite is the authoritative gate; modifying its existing scenarios is out of bounds.
9. **Use AI-assisted commit message generation that produces `Co-Authored-By: Claude` or similar trailers.** Per `feedback_no_co_authored_by.md`.
10. **Mark any sub-item DONE without explicit user approval in the thread.** Per process rule 1 of `docs/v065-closure-board.md`. The agent proposes; the user disposes.

---

## 14. Iteration protocol (agent self-evaluation rubric)

### 14.1 Per-sub-goal loop

For each sub-goal G_n ∈ {G1, G2, G3, G4, G5, G6, G7}:

1. **Read** G_n's section in this document. Understand G, Q, M, S.
2. **Spike phase:**
   - Cut `bench-spike/<descriptor>` branch from current bundle integration HEAD.
   - Implement minimum-viable version of the algorithm.
   - Run the M_n.1 (primary metric) measurement.
   - If green: proceed to production. If red: redesign; iterate spike. Failed spike branches stay unmerged as evidence.
3. **Production phase:**
   - Cut `feat/<descriptor>-prod` branch from current bundle integration HEAD.
   - Implement full version (all variants — u32, u64, triangle, 4-cycle, K5, K6 as applicable).
   - Apply G7 cross-cutting purge on touched files.
   - Run ALL M_n.* metrics.
   - If ALL green: propose merge to `feat/w3-bundle-integration`. If any red: redesign; iterate; failed production branches stay unmerged as evidence.
4. **Bundle integration:**
   - Merge production branch into `feat/w3-bundle-integration`.
   - Run W3.4 closure re-validation (superhub-50K ≥ 1.3×).
   - Run W4.1 cert regression (3/3 PASS).
   - Run workspace gate (`cargo fmt`, `cargo build -D warnings`, `cargo test`).
   - If all green: sub-goal is provisionally DONE. If any red: investigate; if root cause is in current sub-goal's code, fix in current sub-goal; if root cause is in prior sub-goal, escalate to supervisor.

### 14.2 Bundle-level stop condition

Bundle is COMPLETE when:

- All 7 sub-goals are provisionally DONE
- §12 Definition of Done conditions 1–7 ALL hold
- Closure proposal written and user has explicitly approved DONE in thread

Bundle is STUCK (escalate to supervisor) when:

- A sub-goal's spike fails ≥ 3 consecutive redesigns
- A sub-goal's production phase regresses a prior sub-goal's metric by > 10%
- The W4.1 cert suite regresses
- The W3.4 closure metric fails post-G1 refactor

### 14.3 Self-evaluation checklist (run before declaring sub-goal DONE)

```
[ ] Spike branch passed gate metric M_n.1 with documented measurement
[ ] Production branch implements ALL variants (u32, u64, triangle, 4-cycle, K5, K6 where applicable)
[ ] Production branch has zero `// TODO`, `// FIXME`, `// XXX`, `// HACK` on touched files
[ ] Production branch has zero new `Ok(None)` decline paths for paper-aligned shapes
[ ] Production branch has zero new env vars beyond those listed in this document
[ ] Production branch has zero new `cfg(test)` gates on production code
[ ] Production branch removes ALL pre-existing code paths that the new architecture supersedes
[ ] Every Tier-1 kernel function body has a `// T<n> Alg.<x> line N:` comment skeleton on every paper-mapped line
[ ] Workspace gates green: fmt, build -D warnings, test
[ ] CUDA cert suite 1/1
[ ] M_n.* metrics: all green with documented measurements
[ ] W3.4 closure re-validation: superhub-50K ≥ 1.3×
[ ] W4.1 cert regression: 3/3 PASS
[ ] No co-authored-by trailers in any sub-goal commit
[ ] No `v0.6.6` references in any sub-goal commit / file / plan
[ ] G7 cross-cutting purge applied to touched files in the same sub-goal commit chain
```

### 14.4 Closure-proposal format (for user review at bundle DONE)

```markdown
# W3 Bundle Closure Proposal

**Bundle:** W3.3 + W3.5 + W3.6 + W3.7 + W3.8 + W3.9 paper-alignment bundle.
**Integration branch:** `feat/w3-bundle-integration` HEAD `<sha>`.
**Goal document:** `docs/plans/2026-05-13-supervisor-goal-037.md`.

## Sub-goal status

| Sub-goal | Production branch | Final commit | Spike (preserved) | Metric status |
|---|---|---|---|---|
| G1 W3.3 | feat/w33-hg-block-slice-prod | <sha> | bench-spike/w33-hg-block-slice @ <sha> | M1.1=<x>×, M1.2=<x>%, M1.7 cert PASS |
| G2 W3.5 | feat/w35-shmem-prod | <sha> | bench-spike/w35-shmem-narrow @ <sha> | M2.1=<x>×, M2.4=<x> KB |
| G3 W3.6 | feat/w36-warp-prod | <sha> | bench-spike/w36-warp-coop @ <sha> | M3.1=<x>× |
| G4 W3.7 | feat/w37-helper-split-aot | <sha> | bench-spike/w37-helper-split-hand @ <sha> | M4.1=<x>×, M4.2=<x>× |
| G5 W3.8 | feat/w38-stream-mux-aot | <sha> | bench-spike/w38-stream-mux-hand @ <sha> | M5.1=<x>×, M5.4=<x> |
| G6 W3.9 | feat/w39-paper-class-bench | <sha> | (no spike — harness work) | M6.4=<x>×, M6.6=<x>% CV |
| G7 cross-cutting | (per-sub-goal commits) | — | — | M7.1=<x>, M7.4=<x>, M7.5=<x>% |

## W3.4 closure re-validation
- Superhub-50K: <x>× (target ≥ 1.3×)

## W4.1 cert regression
- multirec_triangle, multirec_4cycle, selfrec_triangle: 3/3 PASS

## Workspace gates
- `cargo fmt --check --all`: exit 0
- `RUSTFLAGS="-D warnings" cargo build --release --workspace --exclude pyxlog`: exit 0
- `cargo test --workspace --release --exclude pyxlog --exclude xlog-cuda-tests`: <pass>/0/<ignored>
- `cargo test -p xlog-cuda-tests --test certification_suite --release`: 1/1 (206 internal)
- `cargo bench --bench wcoj_paper_class -- --baseline pre-bundle`: geo-mean speedup <x>×

## Dead-code follow-up
- `docs/evidence/2026-05-bundle-dead-code-followup.md`: <n> entries flagged for separate cleanup

## Audit alignment
- `docs/evidence/2026-05-07-w3-paper-alignment-audit/README.md` updated with "BUNDLE CLOSED" note pinning HEAD `<sha>`

## Request
User approval to mark W3.3, W3.5, W3.6, W3.7, W3.8, W3.9 as DONE on `docs/v065-closure-board.md`.
```

---

## 15. Dispatch instruction (for codex-xlog)

The implementer agent (Codex CLI on tmux session `codex-xlog`) is dispatched via:

```
/goal @docs/plans/2026-05-13-supervisor-goal-037.md
```

Per `feedback_codex_goal_dispatch_flow.md`: after pasting `/goal @<file>.md`, Tab opens the "Replace goal?" popup; Enter confirms. Plain Enter alone does NOT submit.

Per `feedback_codex_goal_command_syntax.md`: the `/goal` command takes `@filename` reference, NOT inline text. This file content is what activates the experimental Goal tracking.

Per `feedback_gqm_strategies_dispatch.md`: all implementer→supervisor reports use S/Q/M codes from this document. Commit messages reference G/S codes. Plan iterations reference Q codes when surfacing new questions.

Per `feedback_codex_tmux_safe_input_clearing.md`: NEVER use C-c to clear codex input. Use C-u or Escape.

Per `feedback_codex_session_recovery.md`: if codex dies, ALWAYS `codex resume <UUID>`; UUIDs live in `~/.codex/sessions/YYYY/MM/DD/rollout-*-<UUID>.jsonl`.

---

## 16. References

- Paper: arXiv:[2604.20073](https://arxiv.org/abs/2604.20073) — Sun, Qi, Gilray, Kumar, Micinski. "Scaling Worst-Case Optimal Datalog to GPUs."
- Audit: `docs/evidence/2026-05-07-w3-paper-alignment-audit/README.md` on `feat/w3-paper-alignment-audit` (`134884fc`).
- Closure board: `docs/v065-closure-board.md` (HEAD `f62188b7`).
- Path C bundle expansion: `docs/plans/2026-05-13-path-c-paper-grounded-findings.md`.
- W4.1 closure entry on closure board (includes T8 documented divergence).
- W3.4 closure: superhub-50K 1.590× preservation baseline.
- Karpathy guidelines: https://x.com/karpathy/status/2015883857489522876.
- GQM paradigm: Basili, V.R., Caldiera, G., Rombach, H.D. (1994) "The Goal Question Metric Approach". https://en.wikipedia.org/wiki/GQM.
- GQM+Strategies extension: Basili, V., Heidrich, J., Lindvall, M., Münch, J., Regardie, M., Trendowicz, A. (2007).
- Green-2012 path merge: Green, O., McColl, R., Bader, D.A. (2012) "GPU Merge Path: A GPU Merging Algorithm".
- CALM theorem: Hellerstein, J.M., Alvaro, P. (2020) "Keeping CALM: When Distributed Consistency is Easy".

---

**End of goal document.** Implementer agent begins with G1 spike. Supervisor awaits spike-phase report.
