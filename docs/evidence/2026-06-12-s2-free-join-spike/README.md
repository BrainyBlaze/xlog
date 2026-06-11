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
