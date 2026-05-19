# Supervisor Goal 034 — W3.5 Shared-Memory Blocking Bench Spike

**Supervisor:** Claude Code.
**Implementer:** Codex CLI on tmux session `codex-xlog`.
**Predecessor:** G33 W3.4 closure FF-merged to `main @ f62188b7`. Tally on main 14 DONE / 1 IN-PROGRESS / 0 BLOCKED / 11 OPEN / 26 Total. W3.5 is next in Path C roadmap order.
**Date:** 2026-05-13.

---

## Context

W3.5 row from `docs/v065-closure-board.md @ f62188b7`:

> | W3.5 | ROADMAP item #12 | OPEN | — | Shared-memory blocking. Per-block tiles of the inner relation cached in `__shared__`. Threshold-gated on relation size. | Bench: shared-mem path shows **≥ 1.5× speedup** vs. baseline kernel on a fixture small enough that the inner fits per-block; deterministic. |

Per `feedback_perf_bench_spike_first.md` discipline: spike first, full impl only if spike proves ≥ 1.5× achievable.

### Paper grounding

Paper §6: *"raw GPU memory bandwidth"* is mechanism #1 of the five-mechanism synergy. Shared-memory blocking is the canonical optimization for converting global memory traffic to fast on-chip access. Paper Algorithm 2 (HG-WCOJ kernel) implicitly relies on the inner relation being small enough to fit in registers/shared memory for the per-block intersection scan.

### Triangle WCOJ "inner relation" identification

For the canonical triangle join `T(x,y,z) :- E(x,y), E(y,z), E(x,z)` with W2.1 leader-first ordering:
- **Outer (driver)**: the column being scanned in the first slot — typically `rel_xy.x` sorted.
- **Middle**: `rel_yz` indexed on `y` for binary search lookups.
- **Inner**: `rel_xz` indexed on `x` for the final intersection check.

The "inner relation" for shared-memory caching is the SMALLEST of the three, OR the one whose access pattern is most random (poorest cache locality under global memory). In the canonical W3.4 fused path (`wcoj_triangle_fused_lc_count`), all three relations are accessed via binary search; the W3.5 candidate is **the smallest relation, loaded into `__shared__` per block as a tile**.

### Hardware envelope (NVIDIA RTX PRO 3000 Blackwell)

- Shared memory per SM: 100 KB (carveout up to 96 KB usable per block).
- Default block-level shared-mem limit without `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)`: 48 KB per block.
- Per-row footprint for u32 column triple (x, y, z): 12 bytes. → 48 KB / 12 ≈ 4,096 rows fit per block at default budget. → 96 KB / 12 ≈ 8,192 rows fit per block with dynamic carveout.

W3.5 spike targets a fixture where the smallest relation is ≤ 4,000 rows (fits the default 48 KB tile).

---

## G34 — W3.5 bench spike (shared-memory blocking)

### Goal

Cut `bench-spike/w35-shared-memory` from `main @ f62188b7`. Implement a minimum-viable shared-memory-blocked triangle count kernel. Construct a small-inner fixture. Measure under V3 paired-batching at the canonical fixture. Report ≥ 1.5× verdict.

If spike clears ≥ 1.5× at any tested scale → recommend G35 full implementation. If spike fails → branch stays unmerged as evidence; G35 = empirical-gap finding to user.

### Strategies (GQM+Strategies)

* **S34.1** Cut `bench-spike/w35-shared-memory` from `main @ f62188b7`. Worktree at `.worktrees/w35-shared-memory`.

* **S34.2 — Fixture construction.** Construct a NEW bench fixture `triangle-small-inner-4K` for this spike (new fixtures ARE allowed in spike per G34 — not under G31's tighter rules):
  * Two large relations + one small relation. Recommended sizes:
    * `rel_xy`: 50,000 rows (the "outer" driver, fixed scale matching superhub-50K).
    * `rel_yz`: 50,000 rows (the "middle", binary-searched).
    * `rel_xz`: **4,000 rows** (the "inner", small enough to fit in 48 KB `__shared__` per block).
  * Hub-pattern key distribution NOT required (no skew needed for this candidate); use uniform random keys in `[0, 2000)` for x, `[0, 2000)` for z so that rel_xz density is naturally ~1/1000 → ~4K rows.
  * Fixture builder lives in `crates/xlog-integration/benches/wcoj_shared_mem_bench.rs` (new file). Deterministic via fixed seed.
  * Document fixture characteristics (per-relation row count, key cardinality, expected triangle count) in the bench file header comment.

* **S34.3 — Spike kernel implementation.** In `crates/xlog-cuda/kernels/wcoj.cu`:
  * New kernel `wcoj_triangle_shared_inner_count` (count-phase analog of G31's `wcoj_triangle_fused_lc_count`).
  * Design pattern:
    * Each block cooperatively loads ALL of `rel_xz` into `extern __shared__` memory at kernel entry (one-time per block).
    * Block-level `__syncthreads()` after load.
    * Threads scan their assigned `rel_xy` rows; for each row, binary-search `rel_yz` for the `y` match (global memory, same as baseline), then check intersection against the per-block `rel_xz` cache (SHARED memory, NOT global).
    * Output count emitted to global memory.
  * Gate behind `#[cfg(feature = "w35-spike-profiling")]` cargo feature (G34 spike is feature-isolated; production dispatch untouched).
  * Add kernel manifest entry for the new symbol.

* **S34.4 — Provider entry.** In `crates/xlog-cuda/src/provider/wcoj.rs`:
  * New entry `wcoj_triangle_shared_inner_u32_recorded` behind `#[cfg(feature = "w35-spike-profiling")]`. Signature matches G31's spike entry contract (caller provides sorted/deduped inputs OR the entry handles sort+dedup internally — pick whichever matches the existing live call convention).
  * Use `cudaFuncSetAttribute(MaxDynamicSharedMemorySize, 49152)` on the new kernel to authorize the 48 KB block budget.
  * NO change to live dispatch routing.

* **S34.5 — V3 bench harness.** In `crates/xlog-integration/benches/wcoj_shared_mem_bench.rs`:
  * Two Criterion benches: `wcoj_baseline_count` (existing unfused count path on the new fixture) + `wcoj_shared_inner_count` (new shared-mem path on the new fixture).
  * V3 protocol: `sample_size(200)`, `iters=1`, paired-batching (alternate baseline+spike per iteration).
  * Row-equality assertion BEFORE first timing: panic if `download_counts` from shared-mem ≠ baseline. (Note: this is a COUNT-phase spike, not materialize, so we compare counts not triple rows.)

* **S34.6 — Measurement protocol.** Run `cargo bench --bench wcoj_shared_mem_bench --features w35-spike-profiling`. Capture:
  * Per-fixture: baseline median + 95% CI, shared-inner median + 95% CI, paired delta µs + 95% CI, paired delta %, speedup ratio (`baseline / shared`).
  * ≥ 1.5× gate verdict at the canonical fixture.
  * If feasible, ALSO measure at a "larger inner" fixture (e.g., `triangle-small-inner-8K` with 8,192 inner rows) to test the threshold boundary at the 96 KB dynamic carveout. This is OPTIONAL — only add if S34.5 implementation is straightforward to parameterize.

* **S34.7 — Evidence README.** `docs/evidence/2026-05-13-w35-shared-memory-spike/README.md` MUST contain:
  * Parent SHA `f62188b7` explicit.
  * Branch HEAD SHA reported in REVIEW REQUEST.
  * Fixture spec: per-relation row counts, key cardinality, expected triangle count, deterministic seed.
  * Kernel design summary: shared-mem tile size, block dimensions, syncthreads boundary, threshold gating logic.
  * Row-equality PASS confirmation (count value match).
  * Per-fixture measurement table.
  * ≥ 1.5× verdict per fixture.
  * Closure-readiness verdict:
    * If ANY fixture clears ≥ 1.5× → "W3.5 closure-ready: recommend G35 full implementation".
    * If NO fixture clears → "W3.5 spike empirically insufficient: recommend re-scope (try different inner-relation selection, different fixture shape, or threshold value) OR defer".

* **S34.8** Branch UNMERGED to `main` + G31 + G32 + G33 ancestor branches. No FF-merge, no push, no tag.

* **S34.9 — Final gates BEFORE the G34 commit.**
  * `cargo fmt --check --all` EXIT 0.
  * `RUSTFLAGS="-D warnings" cargo build --release --workspace --exclude pyxlog --features w35-spike-profiling` EXIT 0.
  * `cargo test -p xlog-cuda-tests --test certification_suite --release` 1/1.
  * `cargo bench --no-run --bench wcoj_shared_mem_bench --features w35-spike-profiling` EXIT 0.

* **S34.10 — Forbidden behaviors.**
  * No `git push`, no `git tag`, no `--force`, no `--no-verify`, no `--dangerously-bypass`.
  * No FF-merge into main.
  * No `docs/v065-closure-board.md` edit.
  * No `v0.6.6` references.
  * **No production dispatch routing change.** The spike kernel is only callable from the spike bench harness via the feature gate.
  * No closure proposal in this goal.
  * No W3.5 mark-DONE.
  * No relaxing the ≥ 1.5× gate.
  * No materialize-phase shared-mem variant (this spike is count-only; materialize is a future extension).

* **S34.11** Single bundled commit subject `spike(w35): shared-memory blocking spike (inner-relation tile) on triangle-small-inner-4K`.

### Questions

* **Q34.1** Branch HEAD SHA?
* **Q34.2** Fixture spec (per-relation row counts + key cardinality + expected triangle count + seed)?
* **Q34.3** Spike kernel design summary (tile size, block dims, syncthreads boundary)?
* **Q34.4** Row-equality PASS at fixture (count value match between baseline + shared-mem)?
* **Q34.5** Per-fixture: baseline median + 95% CI + shared median + 95% CI + paired delta + ratio + ≥ 1.5× verdict?
* **Q34.6** Closure-readiness verdict + recommendation (G35-impl or re-scope)?
* **Q34.7** Branch unmerged from main + G31/G32/G33 ancestors?
* **Q34.8** Final gates all EXIT 0?

### Metrics

* **M34.1** `bench-spike/w35-shared-memory` exists; HEAD reachable from no parent branches.
* **M34.2** Evidence README exists with all sections.
* **M34.3** Row-equality PASS at fixture.
* **M34.4** Per-fixture measurement table populated.
* **M34.5** Closure-readiness verdict explicit + grounded in data.
* **M34.6** `git diff main..HEAD -- crates/xlog-cuda/src/provider/wcoj.rs` shows new entry only behind `#[cfg(feature = "w35-spike-profiling")]`; no live-dispatch reroute.
* **M34.7** Final gates all EXIT 0.
* **M34.8** No tag; no origin push.
* **M34.9** Branch unmerged from main.

### Supervisor validation per locked protocol

* Read evidence README end-to-end.
* `git rev-parse <branch>` ≠ main + ancestors.
* `git diff main..HEAD -- crates/xlog-cuda/src/provider/wcoj.rs` only-adds + feature-gated.
* Verify per-fixture measurements + ≥ 1.5× verdicts.
* Run final gates from supervisor session.
* Verify branch unmerged + no tag + no origin push.

### Decision branching after G34

* **If spike clears ≥ 1.5×:** G35 = W3.5 production implementation (promote kernel; add threshold dispatch on smallest-relation row count; cert grid; bench evidence). G36 = W3.5 closure proposal.
* **If spike fails at small-inner-4K:** present empirical-gap to user — options to re-scope (try smaller inner, different access pattern, or hash-based shared cache instead of linear tile), or to defer W3.5 pending W3.9 production-scale workloads where shared-mem benefit may differ.

Proceed: cut spike branch from main, construct fixture, implement shared-mem-blocked count kernel behind feature gate, wire V3 bench, measure, write evidence README with closure-readiness verdict, single bundled commit. Emit REVIEW REQUEST with HEAD SHA + ratio + verdict.
