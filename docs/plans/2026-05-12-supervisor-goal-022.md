# Supervisor Goal 022 — W3.3 Design-Behavior RCA-4 (Why The Merge-Resident Design Doesn't Deliver Speedup)

**Supervisor:** Claude Code.
**Implementer:** Codex CLI on tmux session `codex-xlog`.
**Predecessor:** G21 stop at CI-width gate. Commit `19d322fc` on `bench-spike/w33-superhub-scale-sweep-v3-stable`. Despite stop firing, **3-run statistical pattern at superhub-50K is empirically clear**: G18 +10.96 µs / 0.989×, G19 +36.66 µs / 0.964×, G21 +18.02 µs / 0.982× — mean **+21.88 µs / ratio 0.978×**. The merge-resident design is consistently **1–4% SLOWER than baseline**, NOT the ≥ 2.0× speedup paper-P3 was supposed to enable. This is no longer a measurement-noise question.
**Date:** 2026-05-12.

---

## Context

User directive recorded 2026-05-12 (verbatim): *"NO ANY FUCKING DEFERS, TOYSHIT, SIMPLIFICATION. SETUP A CLEAR GOAL TO FIX THE ROOT CAUSE AND DELIVER FULL AND ROBUST IMPLEMENTATION!!!!"*

The W3.3 forensic chain has spent G13 + G15 + G17 + G20 attributing measurement noise. Across all four RCAs, the GPU-kernel + W3.3 design-probe variance has been **sub-µs**. Yet across G18 + G19 + G21 measurements, the merge-resident path is **consistently ~22 µs slower** than baseline at superhub-50K. The chain has answered "is the noise in our way?" — no, it isn't — and now must answer the actual paper-claim question: **does the W3.3 design DO what P3 says it should do?**

G22 PIVOTS the chain. No more measurement methodology iterations. No more sample-size tweaks. No more Criterion-vs-parity audits. The question is: *what is the W3.3 design code actually doing at runtime, and why doesn't it produce the speedup that P3-aligned launch-time work-unit slicing should produce on a heavy-row fixture?*

### Paper P3 expectation (memory `reference_srdatalog_paper.md`)

> *Histograms maintained alongside data; computed incrementally during Merge; consumed at kernel launch-time to assign balanced thread-block work-unit slices over the outermost-join's search space.*

Translated to runtime behavior:

1. **Slicing matters.** The kernel launch should partition the outermost-search-space into balanced slices, not run a uniform grid.
2. **Slices should differ across blocks.** Different blocks should receive different slice boundaries based on row-count distribution.
3. **The work each block does should be more balanced under merge-resident than under baseline.** That's the entire point of the optimization.

### Four candidate root causes (concrete)

* **RC1 — Slicing computed but not applied.** Launch-time code reads the histogram and computes slice boundaries, but the kernel launch grid uses the SAME uniform block shape as baseline. The "slicing" is host-side bookkeeping that the GPU never sees.
* **RC2 — Kernel body ignores slice boundaries.** The kernel receives slice metadata as parameters (or via constant memory / device pointer) but its loop body doesn't actually use the slice boundaries to bound iteration. Every block still iterates over the same uniform row range, with slicing data as dead input.
* **RC3 — Histogram bins misaligned with actual hub distribution.** Histogram is computed over wrong feature axis (e.g., row count instead of edge count per hub) or wrong scale (bins too coarse or too fine), so slicing decisions don't reflect actual heavy-row clusters.
* **RC4 — Hardware warp scheduler already balances work.** Modern GPUs (post-Volta) auto-distribute warp-level work via scheduler. If the triangle kernel's per-block work is already balanced by hardware below the host-slicing granularity, the merge-resident path adds bookkeeping overhead with no work-balancing benefit.

G22 distinguishes between these four with code-level + runtime-level instrumentation.

---

## G22 — Design-behavior RCA-4

### Goal

Produce a forensic record at `docs/evidence/2026-05-12-w33-design-behavior-rca4/README.md` on branch `forensic/w33-design-behavior-rca4` (cut from `bench-spike/w33-superhub-scale-sweep-v3-stable @ 19d322fc`) that:

1. Inspects the actual W3.3 merge-resident code paths (provider/runtime/kernel) and reports what they DO with the histogram data.
2. Instruments runtime probes to capture per-block work distribution, slice-assignment ledger, and histogram bin contents at launch time.
3. Compares baseline vs merge-resident execution behavior on superhub-50K.
4. Identifies which root cause (RC1 / RC2 / RC3 / RC4) explains the consistent ~22 µs slowdown.
5. Proposes a concrete code-level fix recipe for G23 to implement.

Branch stays unmerged. Diagnostic probes are allowed in `crates/xlog-cuda/src/` and `crates/xlog-runtime/src/` ONLY behind the existing `wcoj-phase-timing` feature gate (zero overhead when off). NO change to production code paths.

### Strategies (GQM+Strategies)

* **S22.1** Cut `forensic/w33-design-behavior-rca4` from `bench-spike/w33-superhub-scale-sweep-v3-stable @ 19d322fc`. Worktree at `.worktrees/w33-design-rca4`.
* **S22.2** **Code-inspection phase.** Read and document in the README:
  * The exact location and behavior of the merge-resident histogram refresh in `crates/xlog-runtime/src/executor/recursive.rs`.
  * The exact location and behavior of the launch-time slice-read in `crates/xlog-cuda/src/provider/wcoj.rs`.
  * The kernel signature(s) the merge-resident path launches, and whether they differ from the baseline triangle kernel.
  * The kernel body source (in `.cu` files referenced by the kernels): does it consume slice metadata or ignore it?
  * Any explicit assertion that "the merge-resident path launches a DIFFERENT kernel than baseline" — or the inverse, "they launch the same kernel with the same grid shape."
* **S22.3** **Runtime instrumentation phase.** Add feature-gated probes (under `wcoj-phase-timing` extension) to record:
  * **Slice ledger:** for each launch on the merge-resident path, dump the slice boundaries that were computed (per-block start/end row indices, or per-block expected row counts).
  * **Per-block work counter:** add a CUDA-kernel-side per-block counter that records how many input rows each block actually processed and how many output rows each block produced. Read back to host post-launch.
  * **Histogram dump:** dump the histogram bin contents at launch time (bin populations, min/max/mean across bins) for each merge-resident launch.
  * **Baseline-vs-merge kernel diff:** explicit assertion in instrumentation output of whether the two paths invoked the SAME kernel symbol or DIFFERENT kernel symbols.
* **S22.4** **Diagnostic-run phase.** Run ONE launch of baseline + ONE launch of merge-resident on superhub-50K with the new probes enabled. Capture output to `docs/evidence/2026-05-12-w33-design-behavior-rca4/design_behavior_dump.txt`. ALSO dump per-block work distribution to `per_block_work.csv` (columns: `path, block_idx, input_rows, output_rows, time_ns`).
* **S22.5** **Analysis phase.** Build tables in README:
  * **Table A: code-path inspection.** For each W3.3 code surface (refresh / slice-read / kernel launch / kernel body), state what the code DOES vs what P3 expects.
  * **Table B: histogram contents.** Bin populations + distribution stats from one merge-resident launch. Does the histogram reflect heavy-row variance?
  * **Table C: per-block work distribution.** Min / Q1 / median / Q3 / max of input rows per block + output rows per block, for baseline AND merge-resident. Does merge-resident produce more balanced per-block work than baseline?
  * **Table D: kernel-symbol diff.** Did merge-resident launch a different kernel symbol than baseline? If same symbol, what differed in parameters?
* **S22.6** **Verdict phase.** README must contain:
  * Confirmed root cause: RC1 / RC2 / RC3 / RC4 / a hybrid / something else identified during inspection.
  * Evidence quote (specific file:line citation in `crates/` or `.cu` source).
  * Concrete fix recipe: which file changes, which kernel changes, expected behavioral effect. The fix recipe must be specific enough that G23 can execute it without further investigation.
* **S22.7** **Forbidden modifications in G22.** No PRODUCTION-PATH code change. Only diagnostic probes behind `wcoj-phase-timing` feature gate. The probes MUST compile to zero overhead when the feature is off. Verify via `git diff 19d322fc..HEAD -- 'crates/xlog-cuda/src/' 'crates/xlog-runtime/src/' | grep -v '^[+]' | head` and explicit feature-gate check.
* **S22.8** Branch UNMERGED to all twelve parents (main, plan, G12-spike, G13-forensic, G14-respike, G15-forensic, G16-parity, G17-audit, G18-respike-fixed, G19-scale-sweep, G20-stability-rca3, G21-scale-sweep-v3-stable). No FF-merge, no push, no tag.
* **S22.9** Single bundled commit subject `forensic(w33): design-behavior RCA-4 (slice ledger + per-block work + histogram dump + root cause)`. Final commit = forensic README.

### Questions

* **Q22.1** Branch HEAD SHA?
* **Q22.2** Code-inspection (Table A): does the merge-resident path launch a DIFFERENT kernel than baseline, or the SAME kernel with possibly-different parameters?
* **Q22.3** Histogram contents (Table B): does the histogram show actual heavy-row variance at superhub-50K? What's the min/max/mean bin population?
* **Q22.4** Per-block work distribution (Table C): does merge-resident produce more balanced per-block work than baseline? Quantify the difference (variance reduction in input_rows-per-block; output_rows-per-block).
* **Q22.5** Kernel-symbol diff (Table D): same or different kernel symbols? If same, what's the parameter delta?
* **Q22.6** Confirmed root cause from RC1/RC2/RC3/RC4 (or other) with file:line citation?
* **Q22.7** Concrete fix recipe: which files/lines/kernels must G23 modify?
* **Q22.8** Branch unmerged from all 12 parents? Production-path code byte-identical to G21 HEAD (`19d322fc`)?

### Metrics

* **M22.1** `forensic/w33-design-behavior-rca4` exists; HEAD reachable from none of the 12 parents.
* **M22.2** `docs/evidence/2026-05-12-w33-design-behavior-rca4/README.md` exists.
* **M22.3** `design_behavior_dump.txt` and `per_block_work.csv` exist.
* **M22.4** Tables A / B / C / D populated.
* **M22.5** Verdict section names confirmed root cause + specific file:line citation + concrete fix recipe.
* **M22.6** `cargo build --release --features wcoj-phase-timing` EXIT 0.
* **M22.7** `cargo build --release` (no wcoj-phase-timing) EXIT 0 — probes compile out cleanly.
* **M22.8** Strict scientific control: no NON-feature-gated code change in `crates/xlog-cuda/src/` or `crates/xlog-runtime/src/`. Verify each diff line is wrapped in `#[cfg(feature = "wcoj-phase-timing")]` OR analogous gate, OR is the feature-flagged probe-adding code itself.
* **M22.9** `cargo fmt --check --all` EXIT 0.
* **M22.10** `git tag --points-at HEAD` empty; `git ls-remote --heads origin "forensic/w33*"` empty.
* **M22.11** Branch unmerged from all 12 parents.

### Supervisor validation per locked protocol

* Read evidence README end-to-end including all four tables.
* `git rev-parse forensic/w33-design-behavior-rca4` ≠ all 12 parent SHAs.
* Verify M22.6 + M22.7 (both feature-on AND feature-off builds compile).
* Verify M22.8 strict feature-gating: spot-check the diff and confirm all changes are inside `#[cfg(feature = "wcoj-phase-timing")]` blocks or analogous gating.
* Verify Table A names specific file:line citations for each W3.3 code surface inspected.
* Verify Table C shows per-block work distribution variance comparison (baseline vs merge-resident, with quantified delta).
* Verify verdict names ONE root cause (or explicit hybrid attribution) with file:line citation.
* Verify the fix recipe is concrete enough that G23 can execute without re-investigation.
* Verify branch unmerged + no tag + no origin push.

If verdict identifies root cause + fix recipe: G23 = implement the fix (production-path change ALLOWED, scoped to the file/kernel/parameter changes named in G22's fix recipe). G24 = re-run scale sweep under V3 protocol with the fix to validate ≥ 2.0× speedup at superhub-50K (and possibly 200K/1M). G25 = closure proposal.

If verdict identifies "no root cause in W3.3 code — RC4 hardware-already-balances applies": G23 = redesign-or-defer decision presented to user with empirical evidence that the merge-resident design layer adds nothing to what the hardware already does. The user's "no defers" directive applies; redesign is the path, not deferral.

### Forbidden behaviors

* No `git push`, no `git tag`, no `--force`, no `--no-verify`, no `--dangerously-bypass`.
* No merge of `forensic/w33-design-behavior-rca4` into ANY other branch.
* No `docs/v065-closure-board.md` edit.
* No `v0.6.6` references in code or board (verdict text may mention v0.6.6 only as a possible future scope for non-W3.3 work, not as a deferral plan).
* **No PRODUCTION-PATH code change.** All probes MUST compile to zero overhead with `wcoj-phase-timing` off (M22.7 mandatory).
* No measurement-noise attribution work in this goal — that ground is covered by G13/G15/G17/G20.
* No bench harness changes — the Criterion bench is irrelevant for G22; use direct-launch instrumented runs.
* No D7 amendment.
* No design-change PROPOSAL beyond the fix recipe — the proposal lives in the recipe; the implementation lives in G23.
* No closure proposal in this goal.

### Why this is scoped tight

11 goals into the chain, the symptom is identified (~22 µs slowdown, ratio 0.978× at superhub-50K) and measurement-noise has been exhaustively ruled out. The disease must be in the design itself — either in how slicing is computed, how it's passed to the kernel, how the kernel uses it, or in whether the kernel needs it at all. G22 inspects ALL FOUR by reading the code AND running probed measurements ONCE on one launch per path. The output is a verdict + a fix recipe specific enough for G23 to execute.

This is not measurement work. This is engineering RCA at the level of "what does the code do."

Proceed: cut design-behavior-rca4 branch from `19d322fc`, code-inspect the merge-resident path (S22.2), add feature-gated probes (S22.3), run one baseline + one merge-resident launch with probes on (S22.4), build Tables A/B/C/D (S22.5), name root cause + file:line + fix recipe (S22.6), single bundled commit. No merge, no push, no tag. NO production-path code change.
