# Supervisor Goal 015 — W3.3 Second-Pass RCA On G14 Isolated-Bench Residual

**Supervisor:** Claude Code.
**Implementer:** Codex CLI on tmux session `codex-xlog`.
**Predecessor:** G14 partial-validation result. Spike commit `24c51bda` on `bench-spike/w33-merge-resident-histogram-isolated`. Tighter-bench harness removed ~75.5% of the G12 gap (paired delta +56.575 µs / +43.91%, baseline ~129 µs, merge-resident ~185 µs). Row-equality PASS. D7b FAIL by ~8.8× over the ±5% budget. superhub-50K not measured per S14.5 stop. Result partially contradicts G13: ~56.3 µs structural-or-implementation residual remains unattributed by G13's three phase buckets (refresh 0.149 µs + launch-slice 0.027 µs + metadata 0.095 µs ≈ 0.27 µs).
**Date:** 2026-05-12.

---

## Context

G14 collapsed the G12 +229.6 µs measurement to +56.6 µs, proving G13's bench-noise hypothesis was directionally correct but incomplete. The remaining ~56 µs sits in a region G13's phase-timing did not probe — Codex's S14.6 recommendation explicitly names three candidate sources:

1. **CUDA completion-sync overhead** (`cudaDeviceSynchronize` / `cudaStreamSynchronize` between paired baseline and merge-resident launches).
2. **Output-allocation per-iteration residual** (even with priming, allocator/page-fault behavior across iterations).
3. **Provider-call dispatch / FFI boundary overhead** (Rust→CUDA function-call cost, parameter marshaling).

G15 instruments those three sources and re-runs the G14 paired-iter bench under feature `wcoj-phase-timing` to attribute the +56.3 µs gap. The goal mirrors G13's discipline: forensic measurement, no design change, branch unmerged.

### Why this matters for v0.6.5 closure

If a single source dominates (e.g., 50 µs of completion-sync), the structural floor needs revision in W3.3 plan iteration 2 and the production design may need restructuring (e.g., batched launches that amortize sync cost). If the residual spreads thinly across many sub-µs sources, the merge-resident design may be **fundamentally bounded above the +5% budget** under current CUDA semantics — at which point the user faces the gate-amendment-vs-defer-W3.3 decision with empirical evidence rather than guess.

### Measurement-of-measurement caveat (must be addressed in README)

`cudaDeviceSynchronize` after each call serializes execution and adds ~3–10 µs cost on WSL2. The "phase-time the residual" probe IS measurement and may itself contribute to the residual. The forensic README MUST report both `phases-without-sync-probes` AND `phases-with-sync-probes` deltas to separate the measurement-of-measurement effect from the actual operation cost. This was implicit in G13 but is explicit in G15 because the residual is small enough that probe overhead matters.

---

## G15 — Second-pass RCA ONLY

### Goal

Produce a forensic record at `docs/evidence/2026-05-12-w33-isolated-residual-phase-attribution/README.md` on branch `forensic/w33-isolated-residual-phase-attribution` (cut from `bench-spike/w33-merge-resident-histogram-isolated @ 24c51bda`) decomposing G14's +56.6 µs uniform-u32-10K paired delta into per-phase buckets covering completion-sync, output-allocation residual, and provider-call dispatch boundary. Branch stays unmerged. No design change.

### Strategies (GQM+Strategies)

* **S15.1** Cut `forensic/w33-isolated-residual-phase-attribution` from `bench-spike/w33-merge-resident-histogram-isolated @ 24c51bda`. Worktree at `.worktrees/w33-residual-forensic`.
* **S15.2** Extend phase-timing instrumentation with three new bucket families. Honor the G13 zero-overhead-when-feature-off contract:
  * `WcojPhaseTiming::CompletionSync` — bracket cudaDeviceSynchronize / cudaStreamSynchronize calls between paired launches.
  * `WcojPhaseTiming::OutputAllocationResidual` — bracket any output buffer allocation/touch per iteration beyond first-priming.
  * `WcojPhaseTiming::ProviderCallDispatch` — bracket the Rust→CUDA FFI boundary cost: parameter setup, dispatch, return-value handling, excluding the kernel-launch itself.
* **S15.3** Re-run uniform-u32-10K AND superhub-50K with phase timing under both `--mode=sched-phases` AND a new `--mode=sched-phases-no-completion-sync` variant (or equivalent) that omits the completion-sync probe. The diff between the two modes attributes the measurement-of-measurement cost.
* **S15.4** Decomposition tables in README:

  **Table A: per-phase buckets, uniform-u32-10K and superhub-50K, with-sync-probe variant.**

  | Phase | uniform-u32-10K µs | superhub-50K µs | Classification |
  |---|---|---|---|
  | (G13 buckets retained) Refresh / LaunchSlice / Metadata | … | … | Structural P3/P5 |
  | CompletionSync (new) | … | … | Structural CUDA / measurement-of-measurement |
  | OutputAllocationResidual (new) | … | … | Implementation |
  | ProviderCallDispatch (new) | … | … | Structural FFI |
  | triangle_count / scan / total / materialize | … | … | Noise / kernel-internal |
  | Residual / unattributed | … | … | Investigation continues |

  **Table B: measurement-of-measurement effect.**

  | Mode | uniform-u32-10K paired delta µs | Difference |
  |---|---|---|
  | with-sync-probe (G15 default) | … | — |
  | without-sync-probe | … | (probe-induced overhead) |

* **S15.5** Reconciliation: the buckets in Table A (uniform column) MUST sum to within Criterion CI of the G14 paired delta +56.575 µs. The README must state the arithmetic explicitly with any unattributed remainder called out as `Residual / unattributed` with magnitude. If unattributed > 10 µs, recommend G16-RCA-deeper.
* **S15.6** Verdict section in README:
  * **Dominant source identification.** Name the bucket carrying the largest µs of the +56.3 µs gap.
  * **Reducibility verdict.** For each bucket > 5 µs: is it reducible (implementation overhead, can be fixed in design 3) OR structural (can't be fixed without amending D7b or changing CUDA semantics)? Cite paper P3/P5 OR CUDA documentation OR WSL2-specific behavior for each classification.
  * **Plan-iteration-2 input update.** Replace G13's stale plan-iter-2 line (which assumed ~0.27 µs structural floor) with a corrected floor estimate based on G15 data. Name design directions consistent with the corrected floor.
  * **Gate-amendment recommendation.** If reducible buckets sum below the +5% budget once fixed, recommend G16 = plan iter 2 (design 3, with fix list). If reducible buckets sum above the +5% budget even after fixes, recommend user-decision-required (gate amendment vs defer W3.3 vs alternative-design exploration).
* **S15.7** Branch UNMERGED to main, plan, G12-spike, G13-forensic, AND G14-respike. No FF-merge, no push, no tag.
* **S15.8** Single bundled commit subject `forensic(w33): isolated-bench residual phase attribution (completion-sync + alloc + dispatch buckets)`. Final commit must be the forensic README.

### Questions

* **Q15.1** Forensic branch HEAD SHA?
* **Q15.2** uniform-u32-10K Table A per-bucket µs values; reconciliation sum vs +56.575 µs (within Criterion CI).
* **Q15.3** superhub-50K Table A per-bucket µs values (now measured under phase-timing — same as G13's reasoning for measuring both cells in RCA).
* **Q15.4** Table B measurement-of-measurement µs difference: how much of the residual is probe-induced rather than operation-induced?
* **Q15.5** Dominant source identified by name + µs.
* **Q15.6** For each reducible-classified bucket > 5 µs: estimated fix path (code-level, not theoretical).
* **Q15.7** Corrected structural-floor estimate (replacing G13's ~0.27 µs claim).
* **Q15.8** Gate-amendment recommendation: design-3-plausible / user-decision-required.
* **Q15.9** Branch unmerged from all five parents (main / plan / G12-spike / G13-forensic / G14-respike)?

### Metrics

* **M15.1** `forensic/w33-isolated-residual-phase-attribution` exists; HEAD reachable from neither main, plan, G12-spike, G13-forensic, nor G14-respike.
* **M15.2** `docs/evidence/2026-05-12-w33-isolated-residual-phase-attribution/README.md` exists.
* **M15.3** `cargo build --release --features wcoj-phase-timing -p xlog-cuda -p xlog-runtime -p xlog-integration` EXIT 0.
* **M15.4** `wcoj_phase_report` binary runs successfully for both `--mode=sched-phases` AND `--mode=sched-phases-no-completion-sync` (or equivalent) on both cells; raw output captured to evidence directory.
* **M15.5** Table A sums (uniform column) reconcile to +56.575 µs within Criterion CI; arithmetic in README.
* **M15.6** Table A populated for both cells.
* **M15.7** Table B populated with both modes' deltas.
* **M15.8** Verdict section names dominant source + reducibility classification for every >5µs bucket + corrected floor + gate-amendment recommendation.
* **M15.9** `cargo fmt --check --all` EXIT 0.
* **M15.10** `git tag --points-at HEAD` empty; `git ls-remote --heads origin "forensic/w33*"` empty.
* **M15.11** Branch unmerged from all five parents.

### Supervisor validation per locked protocol

* Read evidence README end-to-end.
* `git rev-parse forensic/w33-isolated-residual-phase-attribution` ≠ main / plan / G12-spike / G13-forensic / G14-respike.
* `cargo build --release --features wcoj-phase-timing` EXIT 0 from supervisor session.
* Verify Table A sums reconcile to +56.575 µs within stated CI.
* Verify Table B explicitly separates probe-induced from operation-induced overhead.
* Verify verdict section names a dominant source + classifies every bucket > 5 µs + provides updated structural-floor estimate + recommends design-3 path OR user-decision.
* Verify branch unmerged + no tag + no origin push.

If verdict recommends "design-3-plausible": supervisor confirms G15 complete and writes G16 (plan iter 2 with explicit fix list grounded in G15 numbers).

If verdict recommends "user-decision-required": supervisor presents the empirical case to the user with three options (gate amendment with measured rationale / defer W3.3 to v0.6.6 / alternative-design exploration).

### Forbidden behaviors

* No `git push`, no `git tag`, no `--force`, no `--no-verify`, no `--dangerously-bypass`.
* No merge of `forensic/w33-isolated-residual-phase-attribution` into ANY other branch.
* No `docs/v065-closure-board.md` edit.
* No `v0.6.6` references in code or design — the *recommendation* may mention deferral as one option, but no code/board change implements it in this goal.
* **No design proposal.** RCA only. Plan iteration 2 (if recommended) is a future goal.
* No production-code change beyond feature-gated timing probes. The probes MUST compile to zero overhead with `wcoj-phase-timing` off. Mandatory verification.
* No removal of G13 phase buckets; the existing instrumentation stays.
* No "fix attempts" in this goal — reducible-bucket identification is informational; fixes go in G16/G17.
* No D7 amendment (per plan D6 LOCK). The recommendation may name amendment as an option for the user but does not enact it.
* No re-running the G14 Criterion in production mode — phase-timing run is the measurement; Criterion stays as the validation surface for G16/G17.

### Why this is scoped tight

G13 produced one structural-floor verdict (~0.27 µs); G14 partially contradicted it (~56 µs residual unexplained). G15's job is to attribute that residual precisely so plan iteration 2 can be grounded in numbers, not guesses. Without G15, every subsequent goal in the W3.3 chain would be theory-driven against incomplete data — the exact failure mode that produced the R6 iteration 8 collapse. The measurement-of-measurement caveat is the safety check that prevents the probe from becoming the new noise source.

Proceed: cut forensic branch from `24c51bda`, extend phase-timing with completion-sync + output-alloc-residual + provider-call-dispatch buckets, run wcoj_phase_report on both cells under both probe-on and probe-off modes, decompose, write README with Tables A+B + verdict + corrected floor + recommendation, single bundled commit. No merge, no push, no tag.
