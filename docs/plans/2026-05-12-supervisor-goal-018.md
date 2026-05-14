# Supervisor Goal 018 — W3.3 Clean Respike Under Fixed Bench Harness (D7b PASS Expected)

**Supervisor:** Claude Code.
**Implementer:** Codex CLI on tmux session `codex-xlog`.
**Predecessor:** G17 audit APPROVED. Forensic commit `38dcc7fa` on `forensic/w33-criterion-aggregation-audit`. Per-batch amortization confirmed as dominant aggregation phantom source; bootstrap-resampling ruled out (`+0.000 µs` contribution). Code-level fix identified verbatim from S17.6: *"Force single-iteration batches (`iters=1` equivalent) or use an explicit batch-size-1 paired measurement that records baseline and merge-resident in the same sample before computing the paired delta."* Fix scope: purely in `crates/xlog-integration/benches/wcoj_triangle_bench.rs`. No impl-code change.
**Date:** 2026-05-12.

---

## Context

User decision recorded 2026-05-12: "Apply fix + clean respike on a NEW branch" (Recommended option).

Seven W3.3 branches now exist forming a forensic chain:

| Branch | HEAD | Verdict |
|---|---|---|
| `feat/w33-paper-aligned-plan-it1` | `a4c299fd` | Plan APPROVED |
| `bench-spike/w33-merge-resident-histogram` | `3490fd09` | Apparent D7b FAIL (noisy harness) |
| `forensic/w33-merge-resident-phase-attribution` | `d2a2fca5` | RCA-1: 99.98% noise claim |
| `bench-spike/w33-merge-resident-histogram-isolated` | `24c51bda` | D7b FAIL +56.6 µs (tighter harness, still mismatched) |
| `forensic/w33-isolated-residual-phase-attribution` | `775902ed` | RCA-2: cross-harness +73 µs unattributed |
| `forensic/w33-harness-parity-diagnostic` | `4a8031ef` | Aggregation-pipeline mismatch |
| `forensic/w33-criterion-aggregation-audit` | `38dcc7fa` | Per-batch amortization + iters=1 fix recipe |

G18 is the first goal expected to produce a clean D7b PASS, validating the design under a corrected Criterion harness. The fix is small and localized; the experiment is identical to G14 except for the bench harness shape; the predicted outcome is `+6.12 µs paired delta < +6.52 µs ±5% budget = PASS`.

If G18 produces the predicted PASS, the W3.3 chain is closure-ready. G19 = closure proposal + board OPEN→DONE edit + memory file + MEMORY.md update + FF-merge of `feat/w33-paper-aligned-plan-it1` to main.

If G18 does NOT produce the predicted PASS (i.e., the iters=1 fix doesn't fully eliminate the phantom), the chain reverts to user-decision-required and the alternative-closure paths (skip-respike-direct-close OR amend-D7b-with-evidence OR defer-W3.3) come back on the table.

### Why a new branch instead of patching G14

G14 (`bench-spike/w33-merge-resident-histogram-isolated @ 24c51bda`) is durable evidence of the buggy-bench-harness behavior. Per `feedback_perf_bench_spike_first.md` and the W2.5/W4.2/W4.3/W5.2 precedents, failed/superseded spike branches stay unmerged as historical record. G18 cuts a sibling respike branch carrying G14's bench scaffolding plus the iters=1 fix. Both branches remain unmerged; the diff documents the harness fix in isolation.

---

## G18 — Apply iters=1 fix + clean respike

### Goal

Produce a Criterion measurement record at `docs/evidence/2026-05-12-w33-respike-fixed-harness/README.md` on branch `bench-spike/w33-merge-resident-histogram-respike-fixed` (cut from `bench-spike/w33-merge-resident-histogram-isolated @ 24c51bda`) showing whether the merge-resident P3/P5 design PASSES D7a and D7b under a Criterion harness that uses single-iteration paired batches (`iters=1` equivalent) per G17's identified fix. Branch stays unmerged regardless of outcome. **No implementation-code change beyond the bench harness file.**

### Strategies (GQM+Strategies)

* **S18.1** Cut `bench-spike/w33-merge-resident-histogram-respike-fixed` from `bench-spike/w33-merge-resident-histogram-isolated @ 24c51bda`. Worktree at `.worktrees/w33-respike-fixed`.
* **S18.2** Modify ONLY `crates/xlog-integration/benches/wcoj_triangle_bench.rs`. All other files under `crates/` MUST be byte-identical to G14 HEAD `24c51bda`. Verify via `git diff 24c51bda..HEAD -- crates/ ':!crates/xlog-integration/benches/'` byte-empty.
* **S18.3** Bench-harness fix per G17 S17.6 verdict. Use Criterion's `iter_custom` closure pattern where the closure runs **exactly one paired iteration per `iters` invocation, regardless of the `iters` argument value**. Concretely:
  * Inside `iter_custom(|iters: u64| -> Duration { ... })`, accumulate Duration across `iters` iterations but each iteration runs ONE paired baseline + ONE paired merge-resident call (not a batch).
  * Each iteration records its own `Instant::now()` start/stop, returning the cumulative Duration of `iters` independent paired measurements.
  * This forces Criterion to compute `mean = total / iters` over a sum of single-paired measurements, which matches the parity binary's per-launch reality.
  * Alternative if `iter_custom`'s API is awkward: use `iter_batched_ref` with `BatchSize::PerIteration` (Criterion's idiom for "one inner call per sample").
* **S18.4** Measure cells in plan-prescribed order (same as G14 S14.4):
  1. `uniform-u32-10K` FIRST with row-equality assertion PASS before timing.
  2. `superhub-50K` SECOND with row-equality assertion PASS before timing.
* **S18.5** Honor original W3.3 stop conditions VERBATIM from G12 S12.4 / G14 S14.5:
  * If `uniform-u32-10K` paired delta exceeds ±5% budget, STOP. Do NOT measure superhub-50K. Report failure.
  * If `uniform-u32-10K` passes but `superhub-50K` paired ratio is below 2.0×, report both — do NOT broaden the spike.
* **S18.6** Evidence README at `docs/evidence/2026-05-12-w33-respike-fixed-harness/README.md` MUST contain:
  * Branch + all parent SHAs (G11/G12/G13/G14/G15/G16/G17/main) explicit.
  * Bench-harness change diff summary (the iters=1 fix applied to the relevant Criterion call).
  * Per-cell paired delta + 95% CI + pass/fail verdict.
  * Comparison against G14 paired delta (+56.575 µs) AND G16 parity Instant paired delta (+6.120 µs) showing whether the fixed harness converges to the parity Instant reality.
  * D7a + D7b pass/fail verdicts unambiguous.
  * Stop-condition outcome line explicit.
  * Next-step recommendation: if both gates PASS, recommend G19 = closure proposal; if any gate FAILs, recommend user-decision-required with the new data.
* **S18.7** Branch UNMERGED to main, plan, G12-spike, G13-forensic, G14-respike, G15-forensic, G16-parity, G17-audit. No FF-merge, no push, no tag.
* **S18.8** Single bundled commit subject `spike(w33): respike under fixed harness (iters=1 paired batches)`. If multiple commits needed, final commit must be the evidence README.

### Questions

* **Q18.1** Respike-fixed branch HEAD SHA?
* **Q18.2** Diff confirmation: `git diff 24c51bda..HEAD -- crates/ ':!crates/xlog-integration/benches/'` empty? PASS proves S18.2 honored.
* **Q18.3** uniform-u32-10K paired delta: row-equality PASS? Criterion median delta? Within ±5%? Quote raw Criterion estimates.
* **Q18.4** superhub-50K paired ratio (or paired delta, depending on whether speedup target applies to this fixture): row-equality PASS? Criterion median? D7a verdict?
* **Q18.5** Comparison vs G14 +56.575 µs delta AND G16 parity Instant +6.120 µs: how close did the fixed harness converge to parity Instant reality?
* **Q18.6** Which stop-condition branch was taken? (both PASS / stopped at uniform FAIL / passed uniform but superhub FAIL)
* **Q18.7** Branch unmerged from all eight parents?

### Metrics

* **M18.1** `bench-spike/w33-merge-resident-histogram-respike-fixed` exists; HEAD reachable from neither main, plan, G12-spike, G13-forensic, G14-respike, G15-forensic, G16-parity, nor G17-audit.
* **M18.2** `docs/evidence/2026-05-12-w33-respike-fixed-harness/README.md` exists on respike-fixed branch.
* **M18.3** `cargo bench -p xlog-integration --bench wcoj_triangle_bench --no-run` EXIT 0.
* **M18.4** `git diff 24c51bda..HEAD -- crates/ ':!crates/xlog-integration/benches/'` byte-empty.
* **M18.5** Row-equality PASS for uniform-u32-10K stated verbatim in README; same for superhub-50K if measured.
* **M18.6** uniform-u32-10K paired Criterion median delta stated with absolute µs + ±% vs baseline; pass/fail vs ±5% stated.
* **M18.7** superhub-50K either measured with PASS row-equality + paired ratio/delta + D7a verdict, OR explicitly skipped per S18.5.
* **M18.8** Convergence comparison line in README: parity Instant +6.120 µs (G16) vs G14 +56.575 µs vs G18 fixed-harness paired delta — explicit numeric.
* **M18.9** `cargo fmt --check --all` EXIT 0 on respike-fixed branch.
* **M18.10** `git tag --points-at HEAD` empty; `git ls-remote --heads origin "bench-spike/w33*"` empty.
* **M18.11** Branch unmerged from all eight parents.

### Supervisor validation per locked protocol

* Read evidence README end-to-end.
* `git rev-parse bench-spike/w33-merge-resident-histogram-respike-fixed` ≠ all 8 parent SHAs.
* `git diff 24c51bda..HEAD -- crates/ ':!crates/xlog-integration/benches/'` byte-empty (proves no impl-code change).
* `cargo bench -p xlog-integration --bench wcoj_triangle_bench --no-run` EXIT 0 from supervisor session.
* Verify uniform-u32-10K paired delta is reported FIRST; superhub-50K is SECOND or marked skipped.
* Verify D7b ±5% pass/fail verdict is unambiguous.
* Verify convergence comparison line shows the fix's effect numerically.
* Verify branch unmerged + no tag + no origin push.

If D7a + D7b both PASS: supervisor confirms G18 validates the design+harness pair; G19 = W3.3 closure proposal grounded in the full G11–G18 evidence chain; G19 also handles board OPEN→DONE edit, memory file, MEMORY.md update, and FF-merge.

If D7b PASSes but D7a fails: supervisor reports partial PASS; user decides whether D7a's ≥ 2.0× target is reachable at the actual fixture size or whether scope adjustment is needed.

If D7b fails again: chain reverts to user-decision-required with G14/G16/G17/G18 evidence — implies the iters=1 fix did not fully eliminate the phantom, or there's a second-order effect not captured by G17.

### Forbidden behaviors

* No `git push`, no `git tag`, no `--force`, no `--no-verify`, no `--dangerously-bypass`.
* No FF-merge of `bench-spike/w33-merge-resident-histogram-respike-fixed` into any other branch.
* No `docs/v065-closure-board.md` edit. (Board edit is G19's job, only if G18 PASSes.)
* No `v0.6.6` references.
* **No implementation-code change beyond `crates/xlog-integration/benches/wcoj_triangle_bench.rs`.** Mandatory M18.4 verification.
* No R6 anti-pattern in the bench harness.
* No D7 amendment.
* No "rescue" attempts if uniform-u32-10K fails — STOP per S18.5 and report.
* No removal of row-equality assertions; mandatory per S18.4.
* No closure proposal in this goal — that's G19's job, dependent on G18 outcome.

### Why this is scoped tight

G17 named the fix precisely. G18 applies exactly that fix and measures whether the predicted PASS materializes. The scientific control is identical to G14 (only bench harness file changes from the G14 baseline) — the *only* variable is the iters=1 / single-iteration paired batching. If the design PASSes, the seven-iteration forensic chain has been a coherent investigation producing a verifiable conclusion. If it doesn't, the chain remains coherent but the conclusion is more nuanced and a different closure path becomes necessary.

Proceed: cut respike-fixed branch from `24c51bda`, apply iters=1 paired-batching fix to `wcoj_triangle_bench.rs`, run cells in plan-prescribed order, honor stop conditions, write evidence README with convergence comparison line, single bundled commit. No merge, no push, no tag.
