# S2 — GPU Free Join Spike (Phase A, interim)

Gates (design doc §4): blowup chain >= 2x vs production binary path;
triangle <= 1.2x of the dedicated wcoj_triangle_hg_u32_recorded kernel.
3 runs x median-of-reps, idle GPU, 1942 MHz sustained / 59 C end-state.
Repro: `cargo test -p xlog-cuda-tests --release --test test_free_join_spike
-- --ignored --nocapture` (x3).

## Baseline engine (commit 63a6526a, no identity path) — measured first
- chain (u_cover plan): 1.17x / 1.62x / 1.73x — FAIL
- triangle: 3.43x / 2.03x / 2.30x of dedicated — FAIL

## With identity-group fast path (this commit)
The expand count+scan+host-sync passes are skipped whenever the cover
consumes through its atom's last column (full-row dedup makes every
candidate position its own group; emit takes the null-offsets out==w
branch).
- chain (u_cover): **2.03x / 2.90x / 2.43x — GATE PASS (all runs)**
- chain (natural plan): 1.09x / 1.41x / 1.21x — recorded for comparison;
  plan choice matters (Phase B planner picks u_cover-style plans)
- triangle: 3.75x(cold) / 1.61x / 1.43x of dedicated — **GATE FAIL**,
  improved from baseline; the remaining gap is the separate probe
  kernels + mask compactions vs the dedicated kernel's fused
  expand+intersect loop.

## Verdict and remedy
Chain gate: PASS. Triangle gate: FAIL (1.43-1.61x sustained vs 1.2x).
The identified remedy is fused probe filters in the expand-count pass
(kernel side ALREADY IMPLEMENTED in this commit:
`fj_expand_count_u32`'s `probe_desc`/`n_fused_probes` parameters — the
host launches currently pass the benign null configuration; the Rust
fusion analysis + descriptor packing is the remaining work, preserved
in-branch as the next step; a prior in-flight host-side attempt is in
the git stash "fj fused-probe optimization v2" and contains a
param-lifetime bug — rebuild from the kernel contract, do not pop it
blindly). Phase B does not proceed until the triangle gate passes.

## Update: host-side probe fusion implemented (manual session 2)

Fused probes (key vars ⊆ node cover's new vars AND probe exhausts its
atom) now fold into the expand-count pass as existence filters — no
separate probe kernel, no mask compaction for those subatoms. All 6
parity tests green. Isolated, serial measurements (--test-threads=1,
idle GPU; earlier combined runs were self-contended and are superseded):

- **Chain gate (>= 2x vs binary)**: u_cover 1.69x / 2.76x / 2.59x —
  **PASS on median (2.59x)**, run-to-run spread disclosed. Natural plan
  ~1.0x: plan choice is decisive (Phase B planner requirement).
- **Triangle gate (<= 1.2x of dedicated)**: 1.73x at the gate fixture;
  **2.04x at 10x scale** (n_yz = 3.2M) — **FAIL, and the gap is
  structural, not amortizing overhead**: the generic engine writes the
  frontier twice (node-0 cover copy + final output) ≈ 2x the memory
  traffic of the fused single-pass dedicated kernel.

## Phase A verdict (for the gate decision)

Chain: PASS. Triangle: FAIL at both scales with a quantified structural
cause. Production routing (design §3) keeps triangle/4-cycle/k-clique on
their dedicated kernels — Free Join only handles shapes with NO dedicated
kernel, where the binary tree is the only alternative and the chain gate
is the relevant one. Whether Phase B proceeds under that routing argument
(triangle gate reinterpreted as "informative, shape never routed") is a
program-criteria decision recorded for the maintainer — NOT silently
re-gated here.

## Gate decision (maintainer, 2026-06-12)

Option A accepted: Phase A stands under the production-routing argument —
dedicated shapes (triangle/4-cycle/k-clique) never route to Free Join;
the triangle measurement is retained as the recorded bound on the cost
of generality (1.73x / 2.04x). Phase B authorized.

## Phase B — production integration (manual session 3)

RIR-level integration landed on `feat/d2-free-join`:

- **General multiway promoter** (`try_promote_general_multiway` +
  arity-aware `walk_general_node` in `xlog-logic/src/promote.rs`): any
  Project(inner-join tree) body with >= 3 Scan atoms that every dedicated
  promoter declined becomes a generic `MultiWayJoin` with dense
  first-occurrence variable classes. Scan widths come from the new
  `ExecutionPlan.rel_arities` (populated by the lowerer's AST pre-pass);
  Cartesian (keyless) joins stay on the W4.2 nested-loop routing.
- **Executor dispatch** (`try_dispatch_free_join` in
  `xlog-runtime/src/executor/wcoj_dispatch.rs`): binary2fj over slot
  order with earliest-node probe pushing; declines silently (prefix
  violation, non-u32/Symbol inputs, repeated cover vars) to the embedded
  binary fallback. Wired into BOTH executor paths — recursive
  (`execute_wcoj_or_fallback_node`) and the non-recursive inline chain in
  `execute_stratum_impl` (gap found by the e2e test: the inline chain
  never called the shared helper). Fresh installs dedup explicitly (Free
  Join emits one row per derivation path, unlike the dedicated kernels).
  Counter `free_join_dispatch_count`; kill switch
  `XLOG_DISABLE_FREE_JOIN=1`.
- **Provenance bug found by regression and fixed.** All dedicated
  promoters reorder `inputs` canonically while `output_columns` stays in
  the fallback's column space (documented MultiWayJoin convention).
  Interpreting those nodes positionally permuted head columns whenever a
  stats-reordered triangle declined dedicated dispatch and fell through
  to Free Join (5 row-set parity failures:
  test_selectivity_pass_reordering, test_w26_heat_selectivity x3,
  test_wcoj_4cycle_rir_shape_cert). A leaf-sequence check would be
  unsound for self-join bodies, so the fix is provenance: new
  `MultiwayPlan::FreeJoin` variant set ONLY by the general promoter
  (whose inputs are the fallback's leaves in traversal order, making the
  two column spaces coincide); the dispatcher accepts exactly that
  variant, which also subsumes the dedicated-shape carve-out for rotated
  nodes the structural matchers miss.
- **Epistemic certification awareness**: generic Free Join routes are a
  separate preflight bucket (`free_join_route_count`) excluded from the
  hard dedicated-WCOJ obligation (opportunistic-by-contract), with
  `free_join_dispatch_count` recorded in the runtime counter trace and
  CLI evidence JSON.

Verification (local, functional only — perf measurements are
RunPod-only by standing rule):
- e2e (`xlog-integration/tests/test_free_join_e2e.rs`): 4-atom chain
  fires (counter == 1) with kill-switch row parity; dedicated triangle
  declines (counter == 0, rows correct); non-prefix body declines
  (counter == 0, rows correct via fallback).
- Promoter pipeline shape test
  (`xlog-logic/tests/test_promote_multiway.rs`): chain lands as generic
  MultiWayJoin, FreeJoin-marked, dense vars, fallback preserved.
- Full workspace regression (`cargo test --workspace --all-targets
  --exclude pyxlog --release --no-fail-fast`): 263 targets ok; one
  failure (`g38_mint11_vram` -> g04_transfer_efficiency 7/8) attributed
  to GPU contention from concurrent test binaries — passes 207/207 in
  isolation; D2 touches no transfer path.
