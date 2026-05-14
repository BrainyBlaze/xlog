# Supervisor Goal 016 — W3.3 Harness-Parity Diagnostic (Criterion vs. wcoj_phase_report)

**Supervisor:** Claude Code.
**Implementer:** Codex CLI on tmux session `codex-xlog`.
**Predecessor:** G15 second-pass RCA APPROVED. Forensic commit `775902ed` on `forensic/w33-isolated-residual-phase-attribution`. Three new probes (CompletionSync, OutputAllocationResidual, ProviderCallDispatch) found NO fixable overhead. Cross-harness +73.116 µs residual identified as dominant bucket: G14 Criterion `iter_custom` paired records `merge − baseline = +56.575 µs` (SLOWER), while G15 `wcoj_phase_report` records `merge − baseline = −16.541 µs with sync probe / −2.618 µs without sync probe` (FASTER). Same implementation code, same fixtures, two contradictory verdicts. Verdict: `user-decision-required`.
**Date:** 2026-05-12.

---

## Context

User decision recorded 2026-05-12: "Harness-parity diagnostic" (Recommended option).

Three measurement surfaces (R6 phase-attribution, G13 phase-attribution, G15 expanded phase-attribution) agree the merge-resident design has **sub-µs intrinsic cost**. ONE surface (G14 Criterion `iter_custom`) disagrees. Before any implementation OR deferral decision, the +73.116 µs cross-harness residual must be attributed.

### G16 hypothesis space

The +73.116 µs disagreement between Criterion and `wcoj_phase_report` measuring the same provider calls must come from one of three sources:

1. **Timing-window bracket mismatch.** Criterion's `iter_custom` and `wcoj_phase_report`'s probe-based timing bracket different code regions around the inner provider call. Per-iteration raw deltas would already differ.
2. **Aggregation-pipeline mismatch.** Both harnesses see the same per-iteration raw deltas, but Criterion's median/sample/deviation pipeline produces a different reported aggregate than `wcoj_phase_report`'s simpler median. Per-iteration deltas would align but reported medians diverge.
3. **Process-context mismatch.** Different binary invocation context (Criterion's bench harness vs. `wcoj_phase_report` binary) produces different cache state, allocator state, thread/scheduling state. Per-iteration deltas would *be* different even with aligned brackets, because the underlying execution is different.

G16 designs ONE experiment that distinguishes between (1), (2), and (3).

### Experimental construct: parity binary

A single binary that on each iteration runs BOTH:

* A **Criterion-style timing window**: a `let t0 = Instant::now(); <inner_call>; let dt_criterion = t0.elapsed();` block mirroring exactly what Criterion's `iter_custom` brackets, with the same `black_box` discipline and the same paired baseline/merge order.
* A **phase-report-style scope window**: feature-gated `WcojPhaseTiming::*` probes wrapping exactly the same inner call.

Both measurements record per-iteration raw µs values into a CSV row. The diff between the two columns IS the attribution.

If `dt_criterion == dt_phase_report` per iteration but reported medians differ: aggregation issue (path 2). Codex's next step is to inspect Criterion's median computation.

If `dt_criterion != dt_phase_report` per iteration: bracket-mismatch (path 1) OR process-context mismatch (path 3). The per-iteration delta-of-deltas (`dt_criterion - dt_phase_report`) is itself the new bucket to attribute. Codex's next step is to inspect what code each harness wraps that the other doesn't.

This is a *one-experiment* design that produces interpretable results in all three branches of the hypothesis space.

---

## G16 — Harness-parity diagnostic ONLY

### Goal

Produce a forensic record at `docs/evidence/2026-05-12-w33-harness-parity-diagnostic/README.md` on branch `forensic/w33-harness-parity-diagnostic` (cut from `forensic/w33-isolated-residual-phase-attribution @ 775902ed`) attributing the +73.116 µs cross-harness residual via single-process per-iteration parity timing. Branch stays unmerged. No design change.

### Strategies (GQM+Strategies)

* **S16.1** Cut `forensic/w33-harness-parity-diagnostic` from `forensic/w33-isolated-residual-phase-attribution @ 775902ed`. Worktree at `.worktrees/w33-harness-parity`.
* **S16.2** Create a new bench-harness binary `crates/xlog-integration/src/bin/wcoj_harness_parity.rs` (or equivalent location matching `wcoj_phase_report.rs` precedent). The binary MUST:
  * Run `uniform-u32-10K` AND `superhub-50K` cells (same as G15).
  * Pre-upload fixtures and pre-allocate output buffers (same setup discipline as G14 bench harness).
  * Run N iterations (≥ 50; configurable via `--iterations`).
  * On each iteration: paired baseline launch + paired merge-resident launch, alternating order by iteration parity (same as G14 S14.3e).
  * On each launch: open BOTH timing windows simultaneously:
    * `let t0 = std::time::Instant::now(); <inner_provider_call>; let dt_inst = t0.elapsed().as_nanos() as f64 / 1000.0;` — the Criterion-equivalent surface.
    * Feature-gated `WcojPhaseTiming::*` probes wrapping `<inner_provider_call>` — the phase-report-equivalent surface.
  * Record per-iteration: `cell, iter_idx, path, dt_instant_us, dt_phase_report_us, dt_delta_us` (where `dt_delta_us = dt_instant_us - dt_phase_report_us`).
  * Output to CSV at `docs/evidence/2026-05-12-w33-harness-parity-diagnostic/parity_raw.csv`.
* **S16.3** Aggregation in README:
  * For each cell × path (baseline, merge-resident): report median + 95% CI of `dt_instant_us`, `dt_phase_report_us`, AND `dt_delta_us`.
  * Compute paired delta-of-deltas: `(merge_dt_instant_us - baseline_dt_instant_us) - (merge_dt_phase_report_us - baseline_dt_phase_report_us)`. THIS IS THE ATTRIBUTION OF THE +73 µs.
  * If `dt_delta_us` median is ≈ 0 per iteration: aggregation-pipeline mismatch confirmed; report Criterion's sample/median config and recommend G17 = Criterion-config audit.
  * If `dt_delta_us` median is large and consistent: bracket-mismatch confirmed; report the per-iteration `dt_delta_us` distribution and recommend G17 = code-inspection of Criterion's `iter_custom` source vs. `wcoj_phase_report`'s probe scope.
  * If `dt_delta_us` median is large and inconsistent (high variance): process-context mismatch suspected; recommend G17 = isolation experiment (run both harnesses in the same binary AND in separate binaries, diff results).
* **S16.4** Run a Criterion bench using THE SAME inner-call code path that `wcoj_harness_parity.rs` uses, recorded in a separate but parallel artifact for cross-check. This is the *control*: if Criterion's reported aggregate matches `dt_instant_us` from `wcoj_harness_parity.rs`, then the parity binary correctly reproduces Criterion's behavior. If Criterion's reported aggregate differs from `dt_instant_us`, then there's a Criterion-internal step that the parity binary missed and a deeper RCA is required.
* **S16.5** Evidence README at `docs/evidence/2026-05-12-w33-harness-parity-diagnostic/README.md` MUST contain:
  * Branch + base + G15-forensic + G14-respike + G13-forensic + G12-spike + G11-plan + main SHAs explicit.
  * Parity binary command line.
  * Tables: per-cell, per-path, per-surface medians + CIs.
  * Per-iteration `dt_delta_us` distribution table (e.g., min / Q1 / median / Q3 / max).
  * Cross-check vs. true-Criterion run on the same inner code (S16.4).
  * Attribution verdict: aggregation / bracket / process-context.
  * Code-level next-step recommendation grounded in which hypothesis was confirmed.
* **S16.6** Branch UNMERGED to main, plan, G12-spike, G13-forensic, G14-respike, AND G15-forensic. No FF-merge, no push, no tag.
* **S16.7** Single bundled commit subject `forensic(w33): harness-parity diagnostic (Criterion-vs-phase-report per-iteration deltas)`. Final commit = forensic README.

### Questions

* **Q16.1** Harness-parity branch HEAD SHA?
* **Q16.2** Per-iteration `dt_delta_us` median for uniform-u32-10K paired (merge − baseline): is it ≈ 0 (aggregation issue) OR significantly non-zero (bracket or process-context issue)?
* **Q16.3** Per-iteration `dt_delta_us` distribution: min / Q1 / median / Q3 / max for both cells.
* **Q16.4** True-Criterion cross-check (S16.4): does Criterion's reported aggregate match the parity binary's `dt_instant_us` median? If not, what's the residual?
* **Q16.5** Confirmed hypothesis: aggregation (path 2) / bracket-mismatch (path 1) / process-context (path 3)?
* **Q16.6** Code-level next-step recommendation grounded in confirmed hypothesis.
* **Q16.7** Branch unmerged from all six parents?

### Metrics

* **M16.1** `forensic/w33-harness-parity-diagnostic` exists; HEAD reachable from neither main, plan, G12-spike, G13-forensic, G14-respike, nor G15-forensic.
* **M16.2** Parity binary `crates/xlog-integration/src/bin/wcoj_harness_parity.rs` (or analogous location) exists and compiles under `--features wcoj-phase-timing`.
* **M16.3** `cargo run -p xlog-integration --bin wcoj_harness_parity --release --features wcoj-phase-timing -- --cells=uniform-u32-10K,superhub-50K --iterations=50` (or analogous) completes successfully.
* **M16.4** CSV at `docs/evidence/2026-05-12-w33-harness-parity-diagnostic/parity_raw.csv` exists with at least 50 iterations × 2 cells × 2 paths = 200 rows.
* **M16.5** README contains per-cell × per-path medians + CIs for `dt_instant_us`, `dt_phase_report_us`, AND `dt_delta_us`.
* **M16.6** Per-iteration `dt_delta_us` distribution (min/Q1/median/Q3/max) reported.
* **M16.7** True-Criterion cross-check completed; comparison reported.
* **M16.8** Attribution verdict named: aggregation / bracket / process-context.
* **M16.9** `docs/evidence/2026-05-12-w33-harness-parity-diagnostic/README.md` exists.
* **M16.10** `cargo fmt --check --all` EXIT 0.
* **M16.11** `git tag --points-at HEAD` empty; `git ls-remote --heads origin "forensic/w33*"` empty.
* **M16.12** Branch unmerged from all six parents.

### Supervisor validation per locked protocol

* Read evidence README end-to-end.
* `git rev-parse forensic/w33-harness-parity-diagnostic` ≠ main / plan / G12-spike / G13-forensic / G14-respike / G15-forensic.
* `cargo run --bin wcoj_harness_parity --release --features wcoj-phase-timing -- --cells=uniform-u32-10K --iterations=50` exits 0 from supervisor session.
* Verify CSV has ≥ 200 rows; spot-check one row's `dt_delta_us` is `dt_instant_us - dt_phase_report_us` to within float precision.
* Verify median tables + distribution stats populated.
* Verify true-Criterion cross-check ran and is reported.
* Verify attribution verdict names one of the three hypotheses.
* Verify branch unmerged + no tag + no origin push.

If verdict = **aggregation (path 2)**: G17 = Criterion-config audit; likely fix in benchmark code (Criterion sample size, deviation handling). W3.3 may pass D7b with corrected aggregation.

If verdict = **bracket-mismatch (path 1)**: G17 = code-inspection of Criterion's `iter_custom` source vs. `wcoj_phase_report`'s probe scope to identify what code is wrapped differently. Fix in bench harness OR in probe instrumentation.

If verdict = **process-context (path 3)**: G17 = isolation experiment (run both harnesses in same binary AND separate binaries, diff results). Possibly intractable without OS-level instrumentation.

### Forbidden behaviors

* No `git push`, no `git tag`, no `--force`, no `--no-verify`, no `--dangerously-bypass`.
* No merge of `forensic/w33-harness-parity-diagnostic` into ANY other branch.
* No `docs/v065-closure-board.md` edit.
* No `v0.6.6` references in code or board.
* **No design proposal.** RCA only.
* No production-code change beyond the new parity binary and feature-gated probes. Provider/runtime/memory/kernel code under `crates/xlog-cuda/src/` and `crates/xlog-runtime/src/` MUST stay byte-identical to G15 HEAD `775902ed`. Mandatory verification.
* No "fix attempts" — attribution is informational; fixes are G17.
* No D7 amendment.
* No removal of G15 phase buckets; the existing instrumentation stays.

### Why this is scoped tight

G11 → G15 produced four W3.3 forensics. Each contradicted some part of the prior verdict. G15 closed with `user-decision-required` because the cross-harness residual is the dominant bucket and no probe attributed it. G16's experimental construct (single binary, both timing surfaces per iteration) is the smallest experiment that distinguishes between the three hypothesis branches and produces a code-level next step under each branch. Without G16, every subsequent W3.3 goal would either guess at the +73 µs source OR re-litigate the gate-vs-defer-vs-amend decision without data. The discipline is the same as G13 vs G12: forensic measurement BEFORE design decisions.

Proceed: cut harness-parity branch from `775902ed`, build the parity binary with simultaneous `Instant::now()` + phase-probe timing per iteration, run both cells with ≥ 50 iterations, capture CSV + run true-Criterion cross-check, write README with attribution verdict + code-level next step, single bundled commit. No merge, no push, no tag.
