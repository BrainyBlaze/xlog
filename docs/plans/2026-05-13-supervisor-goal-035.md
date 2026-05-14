# Supervisor Goal 035 — W3.5 Re-Spike: Shared-Memory Bitmap Cache (Bloom-filter style)

**Supervisor:** Claude Code.
**Implementer:** Codex CLI on tmux session `codex-xlog`.
**Predecessor:** G34 binary-search-tile spike on `bench-spike/w35-shared-memory @ 816fca4b` FAILED ≥1.5× gate (0.804× ratio on `triangle-small-inner-4K`). Spike branch preserved unmerged as evidence. G34's own closure verdict recommended re-spike with bitmap/hash cache.
**Date:** 2026-05-13.

---

## Context

G34 evidence preserved at `.worktrees/w35-shared-memory/docs/evidence/2026-05-13-w35-shared-memory-spike/README.md`:

> "shared-memory hash/bitmap cache instead of binary-searching a copied sorted tile."

### Root-cause analysis of G34's 0.804× failure

The binary-search-tile design loaded sorted `rel_xz` (4K rows × 8 bytes = 32KB) into `extern __shared__` and ran per-thread `std::lower_bound`-style binary search against the tile. Hypothesized failure modes:

1. **Bank conflicts**: binary search produces random-ish access patterns in shared memory. With 32 banks of 4 bytes each, 4K-row tile means consecutive threads accessing wildly different bank-aligned offsets → 8-32× serialization on conflict.
2. **Thread divergence**: per-thread binary search has data-dependent early termination. Threads in a warp may complete at different log2(N) ± stragglers, causing the warp to wait on the slowest thread.
3. **Loop overhead vs payload imbalance**: ~12 iterations of binary search per lookup (log2(4096) = 12) with branch-and-bounds-check, against a global-memory baseline whose binary-search overhead is *also* ~12 iterations but with better latency hiding via warp scheduler.

The bitmap design eliminates (1) and (2) entirely; for (3) it converts the inner loop from O(log n) per-lookup to O(1) probe + occasional verification.

### Bitmap design principles

- **Storage layout**: 32KB `__shared__` bitmap = 32 × 1024 × 8 = **262,144 bits**. For 4,000 set bits (one per row), density ≈ 1.5%. Expected false-positive rate ≈ 1.5%.
- **Hash function**: `h(x, z) = ((x * P1) ^ (z * P2)) & (BITMAP_BITS - 1)` where `P1`, `P2` are odd 32-bit primes (e.g., `0x85ebca6b`, `0xc2b2ae35` from MurmurHash3 fmix32 constants). Bit-and with `BITMAP_BITS - 1` requires `BITMAP_BITS` to be a power of two; 262,144 = 2^18 ✓.
- **Cooperative load**: at kernel entry, every thread in the block zeros a stripe of the bitmap (`memset` pattern). Then each thread reads a stripe of `rel_xz` rows, hashes each row's `(x, z)`, and atomically sets the corresponding bit via `atomicOr` on a `uint32_t` slot. One `__syncthreads()` after construction.
- **Lookup**: per-thread per-row: hash `(rel_xy.x, rel_yz_match.z)`, read bitmap word, check bit. If 0 → no match, skip; if 1 → fall back to global-memory binary search on `rel_xz` to verify (the canonical fast unfused W3.1 sort-accessor path).
- **Bank conflicts in bitmap**: bitmap stored as `__shared__ uint32_t bitmap[8192]` (8 KB) gives 4 banks per warp; or split as 4 staggered tables to fully avoid conflicts. **Decision**: start with single-table layout (simpler); if conflicts dominate measured time, split into 32 stripes (one per bank).

### Acceptance gate (unchanged from G34)

W3.5 row gate: ≥ 1.5× speedup vs baseline kernel on a fixture small enough that the inner fits per-block. Same fixture (`triangle-small-inner-4K`) since the W3.5 row text is fixture-shape-independent and we want apples-to-apples comparison with G34's 0.804× number.

---

## G35 — W3.5 re-spike (bitmap/Bloom-filter shared-memory cache)

### Goal

Cut `bench-spike/w35-bitmap-cache` from `main @ f62188b7` (NOT from G34's branch — G35 is an independent design exploration; both stay unmerged in parallel as evidence). Implement minimum-viable bitmap-cached count kernel. Reuse G34's fixture and bench scaffolding. Measure under V3 paired-batching. Report ≥1.5× verdict.

If spike clears ≥1.5× → G36 = full W3.5 production implementation with bitmap cache.
If spike fails → present G35 result + G34 result jointly to user with options: (a) try a third design (e.g., perfect-hash cuckoo, multi-level cache), (b) re-scope to defer W3.5 pending W3.9 production-scale workloads, (c) amend W3.5 gate (requires user approval per W1.1).

### Strategies (GQM+Strategies)

* **S35.1** Cut `bench-spike/w35-bitmap-cache` from `main @ f62188b7`. Worktree at `.worktrees/w35-bitmap-cache`. **Do NOT branch from G34's `bench-spike/w35-shared-memory`** — G35 is a clean re-spike; both designs evaluated against the same `main` baseline.

* **S35.2 — Fixture reuse.** Reuse G34's `triangle-small-inner-4K` fixture spec EXACTLY:
  * `rel_xy`: 50,000 rows, key cardinality 2,000, seed 35,101.
  * `rel_yz`: 50,000 rows, key cardinality 2,000, seed 35,202.
  * `rel_xz`: 4,000 rows, key cardinality 2,000, seed 35,303.
  * Expected total triangle count: 1,219.
  * The fixture builder lives in `crates/xlog-integration/benches/wcoj_shared_mem_bench.rs` on G34's branch. Re-create an equivalent file on G35's branch (NEW: `wcoj_bitmap_cache_bench.rs`) — copy-port the fixture builder, do NOT cross-import from G34's branch.

* **S35.3 — Bitmap kernel implementation.** In `crates/xlog-cuda/kernels/wcoj.cu` (new function, NOT replacing G31/G32 production fused kernel):
  * Function `wcoj_triangle_bitmap_cache_count` with signature analogous to G34's `wcoj_triangle_shared_inner_count`.
  * Bitmap layout: `__shared__ uint32_t bitmap[8192]` (32 KB, 262,144 bits).
  * Cooperative bitmap construction:
    ```cuda
    // Zero the bitmap (each thread zeros 32 words)
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int words_per_thread = 8192 / block_size;
    for (int i = 0; i < words_per_thread; i++) {
        bitmap[tid * words_per_thread + i] = 0u;
    }
    __syncthreads();

    // Each thread sets bits for its slice of rel_xz rows
    int xz_rows = ...;
    for (int row = tid; row < xz_rows; row += block_size) {
        uint32_t x = rel_xz_col0[row];
        uint32_t z = rel_xz_col1[row];
        uint32_t h = ((x * 0x85ebca6bu) ^ (z * 0xc2b2ae35u)) & 0x3FFFFu;  // 18-bit mask
        atomicOr(&bitmap[h >> 5], 1u << (h & 31u));
    }
    __syncthreads();
    ```
  * Per-thread per-`rel_xy` row lookup:
    ```cuda
    // (after binary-search match against rel_yz for the y key)
    uint32_t h = ((x_val * 0x85ebca6bu) ^ (z_val * 0xc2b2ae35u)) & 0x3FFFFu;
    bool maybe_present = (bitmap[h >> 5] >> (h & 31u)) & 1u;
    if (maybe_present) {
        // Verify via global-memory binary search on sorted rel_xz
        if (rel_xz_global_binary_search(x_val, z_val)) {
            count++;
        }
    }
    ```
  * Block size: 256 threads (matches G34); bitmap stripe-zeroing requires `8192 / 256 = 32` words per thread.
  * Dynamic shared-memory cap: `cudaFuncSetAttribute(MaxDynamicSharedMemorySize, 32 * 1024)`. Static `__shared__ uint32_t bitmap[8192]` is acceptable; dynamic is overkill for fixed 32KB.

* **S35.4 — Provider entry.** In `crates/xlog-cuda/src/provider/wcoj.rs`:
  * Two new spike entries behind `#[cfg(feature = "w35-bitmap-spike")]` (DIFFERENT cargo feature name from G34's `w35-spike-profiling` so the two spikes don't interfere when both branches are checked out):
    * `wcoj_triangle_baseline_count_u32_for_w35_bitmap_recorded` (same baseline as G34, re-implemented on this branch).
    * `wcoj_triangle_bitmap_cache_count_u32_recorded` (the new bitmap variant).
  * Both feature-gated; production dispatch untouched.

* **S35.5 — V3 bench harness.** New file `crates/xlog-integration/benches/wcoj_bitmap_cache_bench.rs`:
  * Two Criterion benches: `wcoj_baseline_count_w35_bitmap` + `wcoj_bitmap_cache_count`.
  * V3 protocol: `sample_size(200)`, `iters=1`, paired-batching.
  * Row-equality assertion BEFORE first timing: panic if `download_counts` from bitmap ≠ baseline.
  * Also report **bitmap false-positive rate** as a diagnostic metric (count of "maybe_present" hits divided by total lookups). Document expected ~1.5%.

* **S35.6 — Measurement protocol.** Run `cargo bench --bench wcoj_bitmap_cache_bench --features w35-bitmap-spike` and capture:
  * Per-fixture: baseline median + 95% CI, bitmap median + 95% CI, paired delta µs + 95% CI, paired delta %, speedup ratio.
  * ≥ 1.5× gate verdict.
  * Diagnostic: measured bitmap false-positive rate vs theoretical ~1.5%.
  * Diagnostic: bitmap construction time as % of total kernel time (helps decide if construction-amortization across multiple `rel_xy` queries is worth pursuing).

* **S35.7 — Evidence README.** `docs/evidence/2026-05-13-w35-bitmap-cache-spike/README.md` MUST contain:
  * Parent SHA `f62188b7` (= main); G34 cross-reference (`bench-spike/w35-shared-memory @ 816fca4b` as the design-comparison baseline at 0.804×).
  * Branch HEAD SHA reported in REVIEW REQUEST.
  * Fixture spec (identical to G34).
  * Kernel design summary: bitmap layout, hash function, bank-conflict mitigation strategy, sync boundaries.
  * Row-equality PASS confirmation (count match).
  * Per-fixture measurement table.
  * Diagnostics: measured false-positive rate, construction-phase time share.
  * ≥ 1.5× verdict.
  * Comparison table: G34 binary-search-tile 0.804× vs G35 bitmap-cache ratio.
  * Closure-readiness verdict + recommendation (G36-impl or further re-scope).

* **S35.8** Branch UNMERGED to `main` + G34 (`bench-spike/w35-shared-memory`). Both spike branches preserved in parallel as evidence.

* **S35.9 — Final gates BEFORE the G35 commit.**
  * `cargo fmt --check --all` EXIT 0.
  * `RUSTFLAGS="-D warnings" cargo build --release --workspace --exclude pyxlog --features w35-bitmap-spike` EXIT 0.
  * `cargo test -p xlog-cuda-tests --test certification_suite --release` 1/1.
  * `cargo bench --no-run --bench wcoj_bitmap_cache_bench --features w35-bitmap-spike` EXIT 0.

* **S35.10 — Forbidden behaviors.**
  * No `git push`, no `git tag`, no `--force`, no `--no-verify`.
  * No FF-merge into main.
  * No `docs/v065-closure-board.md` edit.
  * No `v0.6.6` references.
  * **No production dispatch routing change.** Both new entries feature-gated behind `w35-bitmap-spike` (NOTE: different feature name from G34's `w35-spike-profiling`).
  * No closure proposal in this goal.
  * No W3.5 mark-DONE.
  * No relaxing the ≥ 1.5× gate.
  * No fixture change (must use `triangle-small-inner-4K` with identical seeds for apples-to-apples comparison).
  * No deletion/modification of G34's branch or worktree.

* **S35.11** Single bundled commit subject `spike(w35): bitmap shared-memory cache re-spike on triangle-small-inner-4K (G34 follow-up)`.

### Questions

* **Q35.1** Branch HEAD SHA?
* **Q35.2** Fixture spec matches G34 (identical seeds + sizes + expected count 1,219)?
* **Q35.3** Bitmap kernel design summary (bitmap size, hash function constants, sync boundary, bank-conflict strategy)?
* **Q35.4** Row-equality PASS (count match 1,219)?
* **Q35.5** Per-fixture: baseline median + 95% CI + bitmap median + 95% CI + paired delta + ratio + ≥ 1.5× verdict?
* **Q35.6** Diagnostics: measured false-positive rate vs theoretical ~1.5%? Bitmap construction-phase time share?
* **Q35.7** Comparison table G34 binary-search-tile 0.804× vs G35 bitmap-cache ratio?
* **Q35.8** Closure-readiness verdict + recommendation (G36-impl or further re-scope)?
* **Q35.9** Branch unmerged from main + G34?
* **Q35.10** Final gates all EXIT 0?

### Metrics

* **M35.1** `bench-spike/w35-bitmap-cache` exists; HEAD reachable from neither `main` nor G34.
* **M35.2** Evidence README exists with all sections + G34 comparison table.
* **M35.3** Row-equality PASS (count = 1,219).
* **M35.4** Per-fixture measurement table populated.
* **M35.5** Diagnostic measurements (FP rate + construction-share) present.
* **M35.6** `git diff main..HEAD -- crates/xlog-cuda/src/provider/wcoj.rs | grep -E '^\\+\\s*pub fn '` shows new entries; all preceded by `#[cfg(feature = "w35-bitmap-spike")]`.
* **M35.7** Final gates all EXIT 0.
* **M35.8** No tag; no origin push.
* **M35.9** Branch unmerged from main + G34.

### Supervisor validation per locked protocol

* Read evidence README end-to-end.
* `git rev-parse <branch>` ≠ main + G34.
* Verify M35.6 feature-gating on bitmap-spike entries.
* Verify diagnostic measurements present + within expected ranges.
* Run final gates from supervisor session.
* Verify branch unmerged + no tag + no origin push.

### Decision branching after G35

* **If G35 spike clears ≥ 1.5×:**
  * G36 = full W3.5 production impl with bitmap cache (promote kernel, add threshold dispatch on smallest-relation row count + bitmap construction time amortization, cert grid pinning routing + correctness + false-positive bound + W3.x prior preservation, bench evidence).
  * G37 = W3.5 closure proposal (staged OPEN→DONE for user approval per W1.1).

* **If G35 spike also fails (< 1.5×):**
  * Present joint G34 + G35 evidence to user with options:
    * (a) Try a third design (perfect-hash cuckoo, multi-level cache, warp-cooperative search, etc.).
    * (b) Defer W3.5 pending W3.9 production-scale workloads — at production scale the inner-relation latency contribution may differ, possibly making shared-mem caching more impactful.
    * (c) Amend W3.5 gate with user approval per W1.1 (e.g., lower the speedup threshold based on measured ceiling, or scope to a different metric like memory-bandwidth-utilization).
    * (d) Remove W3.5 from the v0.6.5 board with paper-grounded rationale (paper §6 lists shared-mem as a mechanism that pays off mostly at production-DOOP scale; v0.6.5 closure-shape may not need W3.5).

Proceed: cut spike branch from main, port fixture builder + baseline path, implement bitmap-cached count kernel behind `w35-bitmap-spike` feature gate, wire V3 bench with FP-rate + construction-share diagnostics, measure, write evidence README with G34 comparison table + closure-readiness verdict, single bundled commit. Emit REVIEW REQUEST with HEAD SHA + ratio + comparison vs G34 + recommendation.
