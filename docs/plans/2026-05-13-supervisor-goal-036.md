# Supervisor Goal 036 — W3.5 Third Re-Spike: CSS-Tree Cached `rel_yz` BST Top (dominant-cost relation)

**Supervisor:** Claude Code.
**Implementer:** Codex CLI on tmux session `codex-xlog`.
**Predecessor:** G34 binary-search-tile spike FAILED (0.804×). G35 bitmap-cache spike FAILED (0.407×). Joint G34+G35 evidence shows the right relation to cache is the *dominant-cost relation* `rel_yz` (binary search of which consumes most kernel time), not `rel_xz` (the smallest). G35 evidence diagnostic confirmed: "the larger cost is scanning every `rel_yz` candidate for each `rel_xy` row."
**Date:** 2026-05-13.

---

## Context

### Root-cause from G34 + G35

The baseline kernel's per-`rel_xy`-row cost breakdown is approximately:
- `rel_yz` binary search: `log2(50000) ≈ 16` iters in global memory
- Per `rel_yz` match (avg 25 per row for uniform fixture): `rel_xz` binary search `log2(4000) ≈ 12` iters

Total per row ≈ `16 + 25×12 = 316` iters. The `rel_yz` step contributes ~5% of inner-loop iterations BUT runs for every row (50K times). The `rel_xz` step runs only after rel_yz match — and at this fixture scale, `rel_xz` is small enough that L1 cache already serves it efficiently in baseline.

**Caching `rel_xz` (G34, G35) targets the secondary lookup that L1 already handles well. Caching `rel_yz` targets the per-row primary lookup that runs 50K times and is the actual bottleneck.**

### Design choice: CSS-tree (cache-sensitive search tree) BST top

Direct shared-mem caching of `rel_yz` is impossible — 50K rows × 8 bytes = 400 KB > 48 KB block budget. The CSS-tree pattern caches the **top of the binary search tree** over sorted `rel_yz`, not the rows themselves.

CSS-tree top structure (4,096 entries × 8 bytes = 32 KB, fits in `__shared__`):
- Each entry: `(pivot_y: u32, leaf_start_idx: u32)`.
- Entries 0..4095 form a complete binary tree of depth 12 over sorted `rel_yz`.
- Each leaf bracket covers `50000 / 4096 ≈ 12` `rel_yz` rows in global memory.

Per `rel_xy.y` lookup:
1. Walk the 12-level BST top in shared memory: 12 comparisons, each a single shared-mem read + branch. ~30 cycles/level × 12 = ~360 cycles. No bank conflicts (BST traversal is read-only and threads in a warp may take divergent paths, but only the active subtree of each thread accesses memory).
2. At leaf, do a linear scan / 4-iter binary search over the 12-row bracket in global memory. ~4 × 400 cycles = ~1,600 cycles.
3. Total: ~2,000 cycles vs baseline's `16 × 400 = 6,400` cycles. Expected speedup ~3.2×.

This is paper-§4-aligned (paper's Algorithm 1 step "Build Index" is essentially building exactly this kind of structure for the merge phase).

### Why this should beat baseline (unlike G34/G35)

- **Targets the dominant-cost relation** (`rel_yz`), not the L1-served secondary (`rel_xz`).
- **No false-positive overhead** (BST is exact, unlike Bloom filter).
- **Reduces global-memory access count by ~4×** (16 iters → 4 iters in global).
- **Construction cost is amortized**: built once per block, used 50K times per block. Construction is ~10K cycles; per-row savings are ~4,400 cycles each. Break-even at ~3 rows; we have 50K rows.

### Fixture strategy

Two fixtures this time (NEW per S36 — G34/G35 only tested 1 fixture each):
1. **`triangle-small-inner-4K` (identical to G34/G35)** — for A/B/C comparison across all three spikes on the same fixture.
2. **`triangle-medium-yz-200K`** — `rel_yz` scaled to 200K rows. CSS-tree top still 4096 entries (BST depth grows to 14, leaf brackets ~48 rows each). Tests whether CSS-tree advantage *scales* with `rel_yz` size — paper §6 predicts mechanism #1 (memory bandwidth) pays off more at scale.

Acceptance: ≥ 1.5× on AT LEAST ONE fixture. If 4K fails but 200K passes, that's still a closure-worthy signal (the gate is fixture-shape dependent and W3.9 production-scale fixtures will be the real test).

---

## G36 — W3.5 third re-spike (CSS-tree cached `rel_yz` BST top)

### Goal

Cut `bench-spike/w35-css-tree-yz` from `main @ f62188b7`. Implement minimum-viable CSS-tree-cached count kernel. Run on two fixtures (`triangle-small-inner-4K` for cross-spike comparison + `triangle-medium-yz-200K` for scale-emergence test). Measure under V3 paired-batching. Report ≥ 1.5× verdict on each.

### Strategies (GQM+Strategies)

* **S36.1** Cut `bench-spike/w35-css-tree-yz` from `main @ f62188b7`. Worktree at `.worktrees/w35-css-tree-yz`. **Independent branch from G34 and G35** (parallel evidence shape).

* **S36.2 — Fixture builder.** New bench file `crates/xlog-integration/benches/wcoj_css_tree_bench.rs`:
  * **Fixture A: `triangle-small-inner-4K`** — IDENTICAL spec to G34/G35 (xy 50K seed 35101; yz 50K seed 35202; xz 4K seed 35303; key cardinality 2000; expected total count 1219).
  * **Fixture B: `triangle-medium-yz-200K`** — xy 50K seed 36101; yz **200K** seed 36202; xz 4K seed 36303; key cardinality 2000 (so yz density goes up); expected total count documented from CPU oracle.
  * Both fixtures generate sorted/unique `(u32, u32)` binary relations deterministically.

* **S36.3 — CSS-tree construction kernel.** In `crates/xlog-cuda/kernels/wcoj.cu`:
  * New device function `build_css_tree_top` that constructs the 4096-entry BST top from sorted `rel_yz`:
    ```cuda
    // 4096 entries; cooperative across block. Each thread builds a stripe.
    // Entry i = (pivot_y, leaf_start_idx) where i is in BFS level-order.
    // Leaf bracket for entry i (where i >= 2048) covers global rows
    //   [leaf_start_idx, leaf_start_idx + bracket_size).
    __shared__ struct { uint32_t pivot_y; uint32_t leaf_start_idx; } css_tree[4096];

    int yz_rows = ...;
    int bracket_size = (yz_rows + 4095) / 4096;

    // Build leaf entries (bottom level, indices 2048..4095)
    for (int leaf = tid; leaf < 2048; leaf += blockDim.x) {
        int idx = 2048 + leaf;
        int gstart = leaf * bracket_size;
        css_tree[idx].leaf_start_idx = gstart;
        css_tree[idx].pivot_y = (gstart < yz_rows) ? rel_yz_col0[gstart] : 0xFFFFFFFFu;
    }
    __syncthreads();

    // Build internal levels bottom-up
    for (int level_size = 1024; level_size >= 1; level_size /= 2) {
        for (int i = tid; i < level_size; i += blockDim.x) {
            int idx = level_size + i;  // BFS index for this internal node
            int left_child = 2 * idx;
            // Internal node's pivot = pivot of its RIGHTMOST descendant in left subtree
            // = pivot of left_child's rightmost descendant
            // For simplicity, internal[idx].pivot = right_child[idx].pivot (i.e., min of right subtree)
            // and leaf_start_idx = 0 (unused for internal nodes)
            css_tree[idx].pivot_y = css_tree[2 * idx + 1].pivot_y;
            css_tree[idx].leaf_start_idx = 0;
        }
        __syncthreads();
    }
    ```
  * The exact CSS-tree layout (BFS vs Eytzinger) is an implementation detail — pick whichever is simpler. Eytzinger layout has better cache locality for BST traversal but is slightly more complex to construct; BFS is simpler. Start with BFS; if measurement shows BST traversal is the bottleneck, refine.

* **S36.4 — CSS-tree lookup + count kernel.** New kernel `wcoj_triangle_css_tree_count`:
  * Build CSS-tree (S36.3).
  * Per `rel_xy.y` lookup:
    ```cuda
    uint32_t y_val = rel_xy_col1[row];

    // Walk BFS BST: start at root (index 1), descend
    int node = 1;
    while (node < 2048) {  // internal nodes
        if (y_val < css_tree[node].pivot_y) {
            node = 2 * node;       // go left
        } else {
            node = 2 * node + 1;   // go right
        }
    }
    // At leaf (node in 2048..4095)
    int leaf_idx = node - 2048;
    uint32_t bracket_start = css_tree[node].leaf_start_idx;
    uint32_t bracket_end = (leaf_idx < 2047) ? css_tree[node + 1].leaf_start_idx : yz_rows;

    // Linear/binary scan in bracket [bracket_start, bracket_end) of global rel_yz_col0
    // for matching y_val. For each match, do rel_xz binary search (global) and count.
    ```
  * Block size: 256 threads. Threads in a warp may diverge during BST walk; serialization is bounded by warp size × BST depth = 32 × 12 = ~384 cycles worst case.
  * Gate behind `#[cfg(feature = "w35-css-tree-spike")]` (third distinct feature name — coexists with G34's `w35-spike-profiling` and G35's `w35-bitmap-spike`).

* **S36.5 — Provider entry.** In `crates/xlog-cuda/src/provider/wcoj.rs`:
  * Three new spike entries behind `#[cfg(feature = "w35-css-tree-spike")]`:
    * `wcoj_triangle_baseline_count_u32_for_w35_css_tree_recorded` (re-implemented baseline for clean comparison).
    * `wcoj_triangle_css_tree_count_u32_recorded` (the new CSS-tree variant).
    * `wcoj_triangle_css_tree_count_u32_with_diagnostics_recorded` (diagnostic variant emitting BST traversal cycles, leaf-bracket access count, etc.).
  * All feature-gated; production dispatch route untouched.

* **S36.6 — V3 bench harness.** In `wcoj_css_tree_bench.rs`:
  * Four Criterion benches: baseline + CSS-tree on each of two fixtures = 4 cells.
  * V3 protocol: `sample_size(200)`, `iters=1`, paired-batching.
  * Row-equality assertion BEFORE timing (per fixture).
  * Diagnostic emit before timing: BST traversal cycles, leaf-bracket linear-scan cycles, construction cycles.

* **S36.7 — Measurement protocol.** Run `cargo bench --bench wcoj_css_tree_bench --features w35-css-tree-spike`. Capture:
  * Per-fixture: baseline median + 95% CI, CSS-tree median + 95% CI, paired delta, ratio, ≥ 1.5× verdict.
  * Diagnostics: BST-traversal time share, leaf-scan time share, construction time share.

* **S36.8 — Evidence README.** `docs/evidence/2026-05-13-w35-css-tree-yz-spike/README.md` MUST contain:
  * Parent SHA `f62188b7` (= main); G34/G35 cross-references with their commit SHAs + ratios.
  * Branch HEAD SHA reported in REVIEW REQUEST.
  * Both fixture specs.
  * CSS-tree design summary: layout (BFS/Eytzinger), construction cost, traversal depth, leaf-bracket sizes per fixture.
  * Row-equality PASS confirmation per fixture.
  * Per-fixture measurement table (4 cells).
  * Diagnostic breakdown (BST traversal % / leaf scan % / construction %).
  * ≥ 1.5× verdict per fixture.
  * **Cross-spike comparison table**: G34 0.804× vs G35 0.407× vs G36 ratios on `triangle-small-inner-4K`. The first apples-to-apples three-way comparison.
  * Closure-readiness verdict per fixture:
    * If small-4K passes: "W3.5 closure-ready at any scale; recommend G37 full implementation".
    * If medium-200K passes but small-4K fails: "W3.5 closure-ready at scale ≥ 200K rel_yz; recommend G37 with scale-threshold dispatch".
    * If both fail: "W3.5 three-design empirical insufficiency; present joint findings to user with options for further re-scope or deferral."

* **S36.9** Branch UNMERGED to `main` + G34 + G35. All three spike branches preserved in parallel as evidence.

* **S36.10 — Final gates.**
  * `cargo fmt --check --all` EXIT 0.
  * `RUSTFLAGS="-D warnings" cargo build --release --workspace --exclude pyxlog --features w35-css-tree-spike` EXIT 0.
  * `cargo test -p xlog-cuda-tests --test certification_suite --release` 1/1.
  * `cargo bench --no-run --bench wcoj_css_tree_bench --features w35-css-tree-spike` EXIT 0.

* **S36.11 — Forbidden behaviors.**
  * No `git push`, no `git tag`, no `--force`, no `--no-verify`.
  * No FF-merge into main.
  * No `docs/v065-closure-board.md` edit.
  * No `v0.6.6` references.
  * **No production dispatch routing change.** All entries feature-gated.
  * No closure proposal in this goal.
  * No W3.5 mark-DONE.
  * No relaxing the ≥ 1.5× gate.
  * No deletion/modification of G34 or G35 branches/worktrees.
  * No fixture change for `triangle-small-inner-4K` (must use identical seeds to G34/G35 for valid cross-spike comparison).

* **S36.12** Single bundled commit subject `spike(w35): CSS-tree cached rel_yz BST top re-spike on triangle-small-inner-4K + triangle-medium-yz-200K`.

### Questions

* **Q36.1** Branch HEAD SHA?
* **Q36.2** Both fixture specs (seeds + sizes + expected counts)?
* **Q36.3** CSS-tree design summary (BFS/Eytzinger, construction strategy, traversal depth per fixture)?
* **Q36.4** Row-equality PASS at both fixtures?
* **Q36.5** Per-fixture (×2): baseline + CSS-tree medians + paired delta + ratio + ≥ 1.5× verdict?
* **Q36.6** Diagnostic breakdown: BST traversal % / leaf scan % / construction %?
* **Q36.7** Three-way cross-spike comparison table on triangle-small-inner-4K (G34 / G35 / G36 ratios)?
* **Q36.8** Closure-readiness verdict per fixture + overall recommendation?
* **Q36.9** Branch unmerged from main + G34 + G35?
* **Q36.10** Final gates all EXIT 0?

### Metrics

* **M36.1** `bench-spike/w35-css-tree-yz` exists; HEAD reachable from NONE of {main, G34, G35}.
* **M36.2** Evidence README exists with all sections + 3-way comparison table.
* **M36.3** Row-equality PASS at both fixtures.
* **M36.4** Per-fixture measurement table (4 cells) populated.
* **M36.5** Diagnostic measurements present per fixture.
* **M36.6** All new provider entries feature-gated behind `w35-css-tree-spike`.
* **M36.7** Final gates all EXIT 0.
* **M36.8** No tag; no origin push.
* **M36.9** Branch unmerged from main + G34 + G35.

### Decision branching after G36

* **If ≥ 1.5× on small-4K:** G37 = full W3.5 production impl with CSS-tree. G38 = closure proposal.
* **If ≥ 1.5× ONLY on medium-200K:** G37 = full W3.5 impl with scale-threshold dispatch (CSS-tree above some rel_yz size, fall through to baseline below). G38 = closure proposal. This is the "scale-emergence" closure path.
* **If both fail:** present joint G34 + G35 + G36 evidence (3 designs, all failed) to user with options:
  * (a) Defer W3.5 to W3.9 production-scale (this becomes the strongest empirical argument for paper §6's "production-DOOP-only" mechanism characterization).
  * (b) Amend W3.5 gate per W1.1 (lower threshold based on measured ceiling).
  * (c) Remove W3.5 from v0.6.5 board.

Proceed: cut spike branch from main, implement two fixtures + CSS-tree kernel + diagnostics behind feature gate, measure under V3 at both scales, write evidence README with 3-way comparison + closure-readiness verdict per fixture, single bundled commit. Emit REVIEW REQUEST with HEAD SHA + per-fixture ratios + recommendation.
