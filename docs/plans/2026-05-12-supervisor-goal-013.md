# Supervisor Goal 013 — W3.3 Forensic Phase Attribution On G12 Spike (RCA, No Redesign)

**Supervisor:** Claude Code.
**Implementer:** Codex CLI.
**Predecessor:** G12 measured failure. Spike commit `3490fd09 spike(w33): merge-resident histogram cell measurements (uniform-u32-10K + superhub-50K)` on `bench-spike/w33-merge-resident-histogram` (base `feat/w33-paper-aligned-plan-it1` @ `a4c299fd`). uniform-u32-10K row-equality PASS but D7b FAIL by +229.62 µs / +21.14% / 1.21×, ~4.2× over the ±5% budget. superhub-50K not measured per S12.4 stop. Spike branch unmerged.
**Date:** 2026-05-12.

---

## Context

User decision recorded 2026-05-12: "third design attempt with deep root cause investigation and fixes." Sequence: **G13 RCA-only** → G14 plan iteration 2 (design 3, grounded in RCA findings) → G15 spike for design 3 → G16 implementation (only if G15 passes) → G17 closure.

This is the same forensic discipline applied to R6 at `f1142b3e`: phase-attribution measurement BEFORE redesign, not theory-driven design. The R6 forensic decomposed +478 µs on uniform-u32-50K into 304 µs structural + 130 µs implementation + 44 µs noise — that decomposition is what made R6's "measured-rejected" verdict load-bearing. G13 produces the equivalent decomposition for G12's +229.6 µs.

### Existing infrastructure (do NOT rebuild)

* `crates/xlog-cuda/src/wcoj_phase_timing.rs` — phase-timing probes, Cargo feature `wcoj-phase-timing`.
* `crates/xlog-runtime/src/executor/wcoj_phase_timing.rs` — runtime-side instrumentation.
* `crates/xlog-integration/src/bin/wcoj_phase_report.rs` — phase-decomposition runner with `--mode=sched-phases`.

### What's MISSING for G13

Existing probes were instrumented for the R6 front-end architecture (`mask_histogram` / `classify_heavy_rows` / `partition_scan`). The G12 spike has different overhead sites:

1. **Merge-phase histogram refresh** (`refresh_wcoj_merge_resident_histogram_for_spike` invocation in `crates/xlog-runtime/src/executor/recursive.rs:726`).
2. **Launch-time slice-assignment read** (in `crates/xlog-cuda/src/provider/wcoj.rs`).
3. **Metadata allocation / lifetime** on `CudaBuffer` (`crates/xlog-cuda/src/memory.rs`).

G13 must extend the existing instrumentation to capture these three sites WITHOUT changing G12 spike semantics, then re-run `uniform-u32-10K` and `superhub-50K` with phase timing on.

---

## G13 — Forensic phase-attribution ONLY

### Goal

Produce a forensic evidence record at `docs/evidence/2026-05-12-w33-merge-resident-phase-attribution/README.md` on branch `forensic/w33-merge-resident-phase-attribution` (cut from `bench-spike/w33-merge-resident-histogram` @ `3490fd09`) decomposing G12's +229.62 µs into per-phase µs buckets, with explicit identification of (a) structural floor (irreducible per any P3-aligned design), (b) implementation overhead (reducible in design 3). Branch stays unmerged regardless of outcome. No new design, no production-code change beyond timing probes.

### Strategies (GQM+Strategies)

* **S13.1** Cut `forensic/w33-merge-resident-phase-attribution` from `bench-spike/w33-merge-resident-histogram` @ `3490fd09`. Worktree at `.worktrees/w33-forensic`.
* **S13.2** Extend phase-timing instrumentation to cover the three G12 overhead sites:
  * `WcojPhaseTiming` enum gains 3 new variants: `MergeResidentHistogramRefresh`, `LaunchTimeSliceRead`, `MetadataMaintenance` (or analogous names matching existing naming convention).
  * Probes inserted at the three sites listed above. Probes MUST be gated by `wcoj-phase-timing` feature and MUST compile to zero overhead when the feature is off (verify via diff against spike branch's release-mode codegen if needed).
  * `wcoj_phase_report` binary extended to emit the 3 new buckets in `--mode=sched-phases` output.
* **S13.3** Re-run uniform-u32-10K and superhub-50K with phase timing enabled:
  * `cargo run -p xlog-integration --bin wcoj_phase_report --release --features wcoj-phase-timing -- --mode=sched-phases --cells=uniform-u32-10K,superhub-50K`
  * Or equivalent invocation matching existing CLI shape — preserve the R6 forensic's invocation pattern.
  * Capture raw output to evidence directory.
* **S13.4** Decompose +229.6 µs (uniform-u32-10K) and the corresponding superhub-50K delta into per-phase buckets. Required output table:

  | Phase | uniform-u32-10K µs | superhub-50K µs | Classification |
  |---|---|---|---|
  | (existing R6 probe buckets, where applicable) | … | … | structural/implementation |
  | MergeResidentHistogramRefresh | … | … | structural/implementation |
  | LaunchTimeSliceRead | … | … | structural/implementation |
  | MetadataMaintenance | … | … | structural/implementation |
  | Noise | … | … | (within Criterion CI) |

* **S13.5** Comparison table vs R6 forensic at `f1142b3e`:

  | Metric | R6 (uniform-u32-50K) | G12 (uniform-u32-10K) | Notes |
  |---|---|---|---|
  | Total delta | +478 µs | +229.62 µs | … |
  | Structural | 304 µs (64%) | … | … |
  | Implementation | 130 µs (27%) | … | … |
  | Noise | 44 µs (9%) | … | … |
  | Dominant site | mask_histogram (128 µs) | … | … |

* **S13.6** Forensic README at `docs/evidence/2026-05-12-w33-merge-resident-phase-attribution/README.md` MUST contain:
  * Branch SHA + base SHA + G12 commit SHA explicit.
  * The two decomposition tables (S13.4 + S13.5).
  * Explicit "structural floor" verdict: which µs is irreducible for any P3-aligned design with launch-time slice consumption.
  * Explicit "implementation overhead" candidates: which µs could be reduced by code changes (B1-B7 style enumeration if multiple).
  * Design-3 implication statement: what design DIRECTION (not design itself) is consistent with the structural floor (e.g., "additive-zero on uniform path requires X").
  * Plan-iteration-2 input statement: which findings goal G14's plan iteration 2 MUST address.
* **S13.7** Branch stays UNMERGED to `main`, to `feat/w33-paper-aligned-plan-it1`, AND to `bench-spike/w33-merge-resident-histogram`. No FF-merge, no push, no tag. Per `feedback_perf_bench_spike_first.md` precedent: forensic branches are durable evidence, not delivery vehicles. Mirrors the R6 forensic discipline at `f1142b3e`.
* **S13.8** Single bundled commit on the forensic branch with subject `forensic(w33): merge-resident histogram phase attribution (uniform-u32-10K + superhub-50K)`. If multiple commits are needed for the instrumentation extension, final commit must be the forensic README.

### Questions

* **Q13.1** What is the forensic branch HEAD SHA?
* **Q13.2** uniform-u32-10K per-phase decomposition: report each bucket's µs estimate, with classification (structural / implementation / noise). Sum must reconcile against +229.62 µs ± Criterion CI.
* **Q13.3** superhub-50K per-phase decomposition: report each bucket's µs estimate, with classification.
* **Q13.4** Which phase is the dominant overhead source for uniform-u32-10K?
* **Q13.5** What is the structural floor for any P3-aligned design with launch-time slice consumption? State the µs floor with paper-claim justification.
* **Q13.6** What implementation-overhead candidates exist? List each with estimated µs reduction.
* **Q13.7** Given S13.5 comparison, is the G12 spike's overhead profile qualitatively similar to R6 (different cost source, same structural-vs-implementation ratio) or qualitatively different (e.g., entirely structural with no implementation overhead to reduce)?
* **Q13.8** Branch unmerged at completion? `git branch --merged main | grep forensic/w33-merge-resident-phase-attribution` empty AND `git branch --merged bench-spike/w33-merge-resident-histogram | grep forensic/w33-merge-resident-phase-attribution` empty?

### Metrics

* **M13.1** Forensic branch exists; HEAD reachable from neither `main`, nor `feat/w33-paper-aligned-plan-it1`, nor `bench-spike/w33-merge-resident-histogram`.
* **M13.2** `docs/evidence/2026-05-12-w33-merge-resident-phase-attribution/README.md` exists.
* **M13.3** Phase-timing instrumentation extension compiles: `cargo build --release --features wcoj-phase-timing -p xlog-cuda -p xlog-runtime -p xlog-integration` EXIT 0.
* **M13.4** `wcoj_phase_report` binary executes the decomposition run successfully and writes raw output to evidence directory.
* **M13.5** Decomposition table (S13.4) sums reconcile against +229.62 µs (uniform-u32-10K) within Criterion CI. State the reconciliation arithmetic in the README.
* **M13.6** Comparison table (S13.5) populated with non-empty values for every cell.
* **M13.7** "Structural floor" verdict line present in README with explicit µs value AND paper-claim justification (P3 / P5 reference).
* **M13.8** Plan-iteration-2 input statement present in README, naming at least one design direction consistent with the structural floor.
* **M13.9** `cargo fmt --check --all` EXIT 0 on forensic branch.
* **M13.10** `git tag --points-at HEAD` empty on forensic branch; `git ls-remote --heads origin "forensic/w33*"` empty.
* **M13.11** Forensic branch is unmerged (Q13.8 empty).

### Supervisor validation per locked protocol

* Read evidence README end-to-end.
* `git rev-parse forensic/w33-merge-resident-phase-attribution` ≠ `main` HEAD ≠ `feat/w33-paper-aligned-plan-it1` HEAD ≠ `bench-spike/w33-merge-resident-histogram` HEAD.
* `cargo build --release --features wcoj-phase-timing` EXIT 0 from supervisor session.
* Verify decomposition table sums reconcile to +229.62 µs within stated CI.
* Verify comparison table vs R6 forensic is populated.
* Verify "structural floor" line is unambiguous (number + verdict + justification).
* Verify "plan-iteration-2 input" line names at least one design direction.
* Verify branch unmerged + no tag + no origin push.

If all green: supervisor confirms G13 RCA complete and writes G14 covering W3.3 plan iteration 2 (design 3), grounded explicitly in G13's structural-floor verdict and implementation-overhead candidates.

### Forbidden behaviors

* No `git push`, no `git tag`, no `--force`, no `--no-verify`, no `--dangerously-bypass`.
* No merge of `forensic/w33-merge-resident-phase-attribution` into ANY other branch.
* No `docs/v065-closure-board.md` edit (W3.3 stays OPEN per plan D8).
* No `v0.6.6` references.
* **No new design proposal in this goal.** RCA only. Design-3 direction may be stated as a one-line direction in the README's "plan-iteration-2 input" line, but no plan iteration 2 in this commit.
* No production-code change to the G12 spike's actual histogram refresh / launch / metadata code paths beyond adding feature-gated timing probes. The probes themselves MUST compile to zero overhead when `wcoj-phase-timing` is off.
* No "fix attempts" in this goal — implementation overhead identification is informational only; fixes go in G14/G15.
* No D7 amendment (per plan D6 LOCK).
* No re-running of the G12 spike Criterion in production mode in this goal — the phase-timing run is sufficient.

### Why this is scoped tight

R6 produced a forensic record (`f1142b3e`) BEFORE the W3.3 design space was reopened. That forensic is what made the redesign discipline load-bearing — every later W3.3 plan iteration could refer to its specific µs decomposition. G13 is the equivalent for G12. Without it, plan iteration 2 (design 3) would be theory-driven again. The user explicitly asked for "deep root cause investigation and fixes" — G13 produces the investigation; G14 produces the fix proposal grounded in G13's data.

Proceed: cut forensic branch, extend phase-timing instrumentation, re-run uniform-u32-10K + superhub-50K with phase timing, decompose, produce README with two tables + structural-floor verdict + plan-iter-2 input line, single bundled commit. No merge, no push, no tag.
