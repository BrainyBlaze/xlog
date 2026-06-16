# ST-TRC Engine Super-Graph — Feasibility Read (fork A)

Date: 2026-06-16. Engine-side feasibility for the scoped ask in the consumer
scoping doc (`/home/dev/projects/dts-dlm/docs/plans/2026-06-16-st-trc-frontier-consumer-scoping.md`,
rev `8b0c8cb`) over the preserved candidate surface landed consumer-side
(`tensorized_ilp.py`, commit `3464e40`). Status: feasibility — **FEASIBLE,
fork A, no capacity blocker**. Implementation is a downstream gate (a scoped
engine slice pending the operator's go); this doc is the decision artifact.

## What "rule induction = gradient update" asks the engine for

The consumer enumerates a bounded candidate surface (now preserved, not
rank-0-collapsed): top-K clause candidates per topology, bounded by
`max_active_rules`. Each candidate already fixes its join binding — body
relations `(i, j)`, `left_keys` / `right_keys`, `head_bindings`, head relation
`k`, plus the existing discrete viability score. The net-new is to make the
**scoring differentiable** over that surface and execute the selection on GPU:
swap the discrete viability sort for a learnable score + a hard mask, keep the
candidate surface and the two-phase shadow-validation promotion gate.

Three-way seam (locked in #xlog):
- (i) learnable score over the candidates + Straight-Through Gumbel-Softmax
  (temperature) → hard mask — torch-side, shared boundary.
- (ii) **fused candidate-join super-step** firing the masked candidate join
  templates over the relation pool — **this engine lane** (the net-new).
- (iii) mask-gated valuations routed to the score via the XGCF log-space
  circuit autodiff + DLPack — the training-surface seam (generalizes the
  already-merged, finite-difference-verified guard surface).
- temperature schedule / curriculum / candidate pool / parity contract —
  consumer.

## Feasibility

1. **The dILP `[n, w = |constants|^v]` substitution wall is avoided by
   construction.** That wall is a dense-tensor artifact — dILP materializes
   every ground substitution. XLOG is relational: the candidate join templates
   (Free Join frontier engine, aggregate-fused WCOJ, factorized routes already
   on main) execute over the actual tuples present and produce a data-sized
   intermediate bounded by the WCOJ cost model. No `|constants|^v` axis exists
   to blow up. The core de-risk is delivered by the engine already on main.

2. **Score-parameter memory is a non-issue.** The learnable score is one value
   per preserved candidate (bounded by `max_active_rules`, tens). Even a dense
   `#rel^3` upper bound over a tens-sized pool is KB–low-MB fp32. Trivial.

3. **Binding is resolved consumer-side, so fork-A/B is moot for the engine.**
   The candidate surface carries `left_keys` / `right_keys` / `head_bindings`
   per candidate (`_infer_topology`), so the engine scores over candidates that
   already fix the join pattern. The super-graph need not derive bindings.

4. **Compute frontier (the real cost, all tractable for tens-sized).** Forward
   fires the masked candidate join templates, bounded by the number of active
   candidates × `T` forward steps. A soft, temperature-annealed forward (needed
   early for exploration, see §risk) fires the full top-K weighted; those
   candidate joins **batch/fuse through the Free Join / WCOJ machinery** into a
   single fused super-step rather than N separate launches — the net-new engine
   engineering, and where the factorized lane pays off. Pool/top-K size (tens)
   is the load-bearing lever.

5. **Autodiff composes via the proven pattern.** The mask-gated candidate
   valuations route to the score through the same custom autograd path the
   merged mixed-trainable-rule guard surface exercises (a single-candidate soft
   mask is the special case the multi-candidate ST-Gumbel mask generalizes);
   DLPack zero-copy at the store export boundary.

## The one genuine risk: gradient/exploration signal (not capacity)

Straight-through Gumbel is biased, and a hard forward only fires selected
templates — an off-template produces no output and so gets no gradient to turn
it on. Mitigations, all agreed: the preserved multi-candidate surface (a hard
mask needs something to mask — rank-0 collapse left nothing); a
temperature-annealed **soft-first** forward so every candidate gets gradient
early, annealing to the hard executable mask only after the soft phase
separates the correct `(i, j) → k`; and a Phase-1a soft pre-check on the
existing guard surface (no new engine work) as a cheap signal-convergence test
before the fused super-step lands. Phase-1 rediscover-a-known-rule is the gate
that retires this risk.

## Net-new engine work (this lane, when greenlit)

The fused candidate-join super-step: given the preserved candidate surface + a
(soft or hard) per-candidate weight, fire the candidate join templates as one
fused batched operation over the relation pool (reusing Free Join / WCOJ
batching), emitting per-candidate head valuations for the mask gate. Everything
else is reuse: candidate enumeration + binding + two-phase validation
(consumer), the autodiff core + DLPack (training-surface seam), the join
kernels (already on main).

## Cost model / where it grows

- Score memory: `O(#candidates)` (trivial).
- Forward: `O(#fired candidates × per-join cost) × T` steps; soft regime fires
  `O(#candidates)` joins/step, collapsed by the fused super-step.
- Pool size is the lever; keep it tens. Fork B (extra binding axis) is not
  needed — bindings are enumerated consumer-side.

## Verdict

Fork A is feasible with no capacity blocker; the memory gate is passed and the
`|constants|^v` wall is structurally avoided. The remaining risk is the
ST-Gumbel exploration signal, retired empirically by Phase-1 rediscovery on a
soft-first curriculum. The net-new engine work is the bounded, well-scoped
fused candidate-join super-step. Sequence: Phase-1a soft pre-check (existing
surface, no engine work) → Phase-1b ST-Gumbel + fused super-step (this lane) →
Phase-2 discover-at-impasse. Engine implementation is a scoped slice pending
the operator's go.
