# Supervisor Goal 014 — W3.3 Tighter-Bench Re-Spike (Measurement Isolation, NOT Redesign)

**Supervisor:** Claude Code.
**Implementer:** Codex CLI on tmux session `codex-xlog`.
**Predecessor:** G13 RCA APPROVED. Forensic commit `d2a2fca5b83c6e95ce6787eecf0204ebefc136ed` on `forensic/w33-merge-resident-phase-attribution`. Phase decomposition proved G12's +229.62 µs Criterion delta = ~0.27 µs P3/P5 design floor (0.02%) + ~229.59 µs Criterion-harness wall residual / upload-allocation jitter (99.98%). Plan-iteration-2 input recorded verbatim in `docs/evidence/2026-05-12-w33-merge-resident-phase-attribution/README.md`: *"Keep the P3/P5 merge-resident direction in scope, but require a bench-spike protocol that separates launch-time slice assignment from upload/allocation noise and uses paired baseline/merge timing for the D7b cell."*
**Date:** 2026-05-12.

---

## Context

G14 is the explicit answer to the user's "third design attempt with deep root cause investigation and fixes" directive. The G13 RCA inverted the strategic picture: G12 was NOT a design failure equivalent to R6. R6 was structural rejection at 16× over budget; G12's P3/P5 design has a measured ~0.27 µs floor. The "fix" the user asked for is therefore **measurement isolation**, NOT a third design.

This goal preserves the G12 implementation code unchanged and tightens ONLY the bench harness surface. If G14's tighter-bench measurements pass D7a + D7b, the P3/P5 merge-resident design is validated and a future goal authorizes production implementation. If G14 still misses D7b under tighter measurement, the RCA's structural-floor verdict is contradicted and a new RCA cycle is required — but per G13 evidence this outcome is unlikely.

### Why a new branch instead of patching G12

G12's `bench-spike/w33-merge-resident-histogram` @ `3490fd09` is **durable evidence of a noisy-bench failure mode**. Per `feedback_perf_bench_spike_first.md`, failed spike branches stay unmerged as part of the historical record. G14 cuts a sibling branch carrying G12's implementation forward plus the new bench harness. Both branches remain unmerged; the diff between them documents the measurement-isolation change in isolation.

---

## G14 — Tighter-bench re-spike ONLY

### Goal

Produce a Criterion measurement record at `docs/evidence/2026-05-12-w33-isolated-bench-respike/README.md` on branch `bench-spike/w33-merge-resident-histogram-isolated` (cut from `bench-spike/w33-merge-resident-histogram @ 3490fd09`) decomposing whether the P3/P5 merge-resident histogram design passes D7a (`superhub-50K ≥ 2.0×`) and D7b (`uniform-u32-10K within ±5%`) under a measurement-isolated bench harness that brackets only launch-time work. Branch stays unmerged regardless of outcome. **No implementation-code change beyond the bench harness file.**

### Strategies (GQM+Strategies)

* **S14.1** Cut `bench-spike/w33-merge-resident-histogram-isolated` from `bench-spike/w33-merge-resident-histogram @ 3490fd09`. Worktree at `.worktrees/w33-respike-isolated`.
* **S14.2** Modify ONLY `crates/xlog-integration/benches/wcoj_triangle_bench.rs` (the bench harness file). All four other G12-allowed files (`crates/xlog-cuda/src/memory.rs`, `crates/xlog-cuda/src/provider/wcoj.rs`, `crates/xlog-runtime/src/executor/recursive.rs`, plus the G13 phase-timing instrumentation files) MUST be byte-identical to their state at `3490fd09`. Verify via `git diff 3490fd09..HEAD -- crates/ ':!crates/xlog-integration/benches/'` being empty.
* **S14.3** Bench harness improvements, each justified against G13's RCA finding (~229.59 µs residual / upload-allocation jitter):
  * **S14.3a Pre-upload fixtures outside timed region.** Fixture upload to GPU happens in Criterion setup, not inside the timed iteration body. Eliminates upload jitter from the timing window.
  * **S14.3b Pre-allocate output buffers outside timed region.** `download_triples` output destination allocated once in setup. Eliminates allocation jitter from the timing window.
  * **S14.3c Use `Criterion::iter_custom` for paired baseline/merge timing.** Each Criterion iteration measures BOTH the baseline launch and the merge-resident launch back-to-back within the same iteration, recording the *difference*. This eliminates scheduler/cache interference from cross-iteration variance.
  * **S14.3d Timing window brackets ONLY launch-time slice-assignment + kernel invocation + completion sync.** No upload, no allocation, no row-equality assertion inside the timed bracket.
  * **S14.3e Row-equality assertion stays — but OUTSIDE the timed bracket.** Per S12.4 stop-condition discipline: PASS row-equality is still mandatory before any timing data is accepted; the assertion just happens before the timing region opens.
  * **S14.3f Iteration count tuned for sub-µs resolution.** Criterion's default warmup + measurement budgets may be insufficient when the actual signal is sub-µs. Increase samples or use `WarmUpTime` / `MeasurementTime` overrides as needed.
* **S14.4** Measure cells in plan-prescribed order:
  1. `uniform-u32-10K` FIRST with row-equality assertion PASS before timing.
  2. `superhub-50K` SECOND with row-equality assertion PASS before timing.
  Report paired delta (merge − baseline) Criterion median + 95% CI for both cells.
* **S14.5** Honor original W3.3 stop conditions VERBATIM from G12 S12.4:
  * If `uniform-u32-10K` paired delta exceeds the ±5% budget (relative to baseline median), STOP. Do NOT measure superhub-50K. Report failure.
  * If `uniform-u32-10K` passes but `superhub-50K` paired ratio is below 2.0×, report both results — do NOT broaden the spike implementation.
* **S14.6** Evidence README at `docs/evidence/2026-05-12-w33-isolated-bench-respike/README.md` MUST contain:
  * Branch SHA + base SHA + G12 spike SHA + G13 forensic SHA explicit.
  * Bench-harness change diff summary (file `wcoj_triangle_bench.rs` only).
  * Per-cell paired delta + 95% CI + pass/fail verdict.
  * Comparison against G12's Criterion delta (showing whether tighter measurement converges toward the G13 ~0.27 µs design floor).
  * D7a + D7b pass/fail verdicts unambiguous.
  * Stop-condition outcome line explicit.
  * Next-step recommendation line consistent with verdict (validate-and-impl / structural-recheck / etc.).
* **S14.7** Branch stays UNMERGED to main, to plan branch, to G12 spike, AND to G13 forensic. No FF-merge, no push, no tag.
* **S14.8** Single bundled commit subject `spike(w33): isolated-bench re-spike (paired-iter timing, pre-uploaded fixtures)`. If multiple commits needed, final commit must be the evidence README.

### Questions

* **Q14.1** Re-spike branch HEAD SHA?
* **Q14.2** Diff confirmation: `git diff 3490fd09..HEAD -- crates/ ':!crates/xlog-integration/benches/'` empty? (Q14.2 PASS proves S14.2 honored.)
* **Q14.3** uniform-u32-10K paired delta: row-equality PASS? Criterion median delta? Within ±5%? Quote raw Criterion estimates.
* **Q14.4** superhub-50K paired ratio: row-equality PASS? Criterion median ratio? Within ≥ 2.0×? Quote raw estimates. Only answer if Q14.3 passed; otherwise state "not measured per S14.5 stop condition".
* **Q14.5** Comparison vs G12 +229.62 µs delta: how much closer did the tighter bench get to the G13 ~0.27 µs structural floor?
* **Q14.6** Which stop-condition branch was taken? (stopped at uniform / both cells measured / passed uniform, failed superhub)
* **Q14.7** Branch unmerged from all four parents (main / plan / spike / forensic)?

### Metrics

* **M14.1** `bench-spike/w33-merge-resident-histogram-isolated` exists; HEAD reachable from neither main, nor `feat/w33-paper-aligned-plan-it1`, nor `bench-spike/w33-merge-resident-histogram`, nor `forensic/w33-merge-resident-phase-attribution`.
* **M14.2** `docs/evidence/2026-05-12-w33-isolated-bench-respike/README.md` exists on re-spike branch.
* **M14.3** `cargo bench -p xlog-integration --bench wcoj_triangle_bench --no-run` EXIT 0.
* **M14.4** `git diff 3490fd09..HEAD -- crates/ ':!crates/xlog-integration/benches/'` is byte-empty.
* **M14.5** Row-equality PASS for uniform-u32-10K stated verbatim in README; same for superhub-50K if measured.
* **M14.6** uniform-u32-10K paired Criterion median delta stated with absolute µs + ±% vs baseline; pass/fail vs ±5% stated.
* **M14.7** superhub-50K either measured with PASS row-equality + paired ratio, OR explicitly skipped per S14.5.
* **M14.8** Comparison line (Q14.5) present in README with numeric delta-from-G13-floor.
* **M14.9** `cargo fmt --check --all` EXIT 0 on re-spike branch.
* **M14.10** `git tag --points-at HEAD` empty on re-spike branch; `git ls-remote --heads origin "bench-spike/w33*"` empty.
* **M14.11** Re-spike branch is unmerged from all four parents (Q14.7 empty).

### Supervisor validation per locked protocol

* Read evidence README end-to-end.
* `git rev-parse bench-spike/w33-merge-resident-histogram-isolated` ≠ main / plan / spike / forensic.
* `git diff 3490fd09..HEAD -- crates/ ':!crates/xlog-integration/benches/'` empty (proves no impl-code change).
* `cargo bench -p xlog-integration --bench wcoj_triangle_bench --no-run` EXIT 0 from supervisor session.
* Verify uniform-u32-10K paired delta is reported FIRST in README; superhub-50K is SECOND or marked skipped.
* Verify D7b ±5% pass/fail verdict is unambiguous.
* Verify comparison-vs-G13-floor line shows the measurement-isolation effect.
* Verify branch unmerged + no tag + no origin push.

If D7a + D7b both PASS: supervisor confirms G14 validates the P3/P5 merge-resident design empirically; G15 = implementation plan iteration 2 grounded in G14 numbers; G16 = production implementation; G17 = closure proposal.

If D7b fails: supervisor reports a measurement-isolation result that contradicts the G13 structural-floor verdict — the user decides whether to extend G13 with deeper RCA, amend D7 with explicit approval, or defer W3.3 from v0.6.5.

If D7b passes but D7a fails: supervisor reports D7a-only design gap — G15 may either propose a design tweak targeting the superhub case OR recommend partial closure with D7b-only acceptance (user decision).

### Forbidden behaviors

* No `git push`, no `git tag`, no `--force`, no `--no-verify`, no `--dangerously-bypass`.
* No FF-merge of `bench-spike/w33-merge-resident-histogram-isolated` into any other branch.
* No `docs/v065-closure-board.md` edit (W3.3 stays OPEN per plan D8).
* No `v0.6.6` references.
* **No implementation-code change beyond `crates/xlog-integration/benches/wcoj_triangle_bench.rs`.** Mandatory verification per M14.4.
* No re-introduction of any R6 anti-pattern in the bench harness (per-call histogram launch / heavy-light split / classify_heavy_rows / front-end mask_histogram+classify+partition_scan).
* No D7 amendment (per plan D6 LOCK).
* No "rescue" attempts in this goal if uniform-u32-10K fails — STOP per S14.5 and report.
* No removal of row-equality assertions; they remain mandatory per S14.3e (just moved outside the timed region).

### Why this is scoped tight

G13 told us where to look (measurement, not design). G14 looks there with one variable changed (bench harness). If the P3/P5 design passes D7a+D7b under the tighter measurement, we have empirical validation for production. If it fails, we have a clean contradiction of the G13 RCA that demands re-investigation. Either way, **only one thing changed between G12 and G14: the bench**. That's the scientific control.

Proceed: cut re-spike branch from `3490fd09`, modify only `wcoj_triangle_bench.rs`, run paired measurements in plan-prescribed order, honor stop conditions, write evidence README with comparison-vs-G13-floor line, single bundled commit. No merge, no push, no tag.
