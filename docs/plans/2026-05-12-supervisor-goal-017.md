# Supervisor Goal 017 — W3.3 Criterion Aggregation Audit (Batch-Compatible Parity Mode)

**Supervisor:** Claude Code.
**Implementer:** Codex CLI on tmux session `codex-xlog`.
**Predecessor:** G16 harness-parity diagnostic APPROVED. Forensic commit `4a8031ef` on `forensic/w33-harness-parity-diagnostic`. Aggregation-pipeline mismatch confirmed (path 2 of three-hypothesis space). Parity `Instant` paired delta `+6.12 µs` (UNDER ±5% budget `+6.52 µs`) on uniform-u32-10K. Criterion's reported aggregate adds `~+32 µs` phantom overhead. Both timing surfaces inside parity binary agree to ~0.02 µs per launch. W3.3 design empirically validated as sub-µs; Criterion validation surface is the lone disagreeing measurement.
**Date:** 2026-05-12.

---

## Context

User decision recorded 2026-05-12: "G17 = audit Criterion aggregation, then close W3.3 cleanly" (Recommended option).

The G16 verdict identified Criterion's `iter_custom` aggregation pipeline as the source of the +56–73 µs cross-harness residual. G17's job is to attribute the ~+32 µs aggregation phantom precisely enough that:

1. A code-level fix (bench harness adjustment) is identifiable, OR
2. The discrepancy is documented as a Criterion-internal artifact that doesn't reflect production behavior.

Either outcome unblocks G18 = W3.3 closure proposal.

### Criterion `iter_custom` mechanics (background)

`Bencher::iter_custom<F: FnMut(u64) -> Duration>(routine)` calls the closure with `iters: u64`. The closure is responsible for running the inner work `iters` times and returning the total duration. Criterion then:

* Per-sample: computes `mean = total / iters` and records this as one sample.
* Per-run: collects ~100 samples (default sample size).
* Median estimate: bootstrap-resampled median over the sample set.

**Two places where the parity binary's `Instant` median can diverge from Criterion's reported median:**

1. **Per-batch amortization.** A single `Instant::now(); for _ in 0..iters { inner(); } Instant::now()` may not produce a total that equals `iters * (per-iteration cost in isolation)`. Warm caches, allocator stability, branch prediction, JIT-style runtime effects can make the per-iteration cost INSIDE a batch differ from the per-iteration cost of a standalone iteration. This is the most likely source.

2. **Bootstrap-resampled median vs. order-statistic median.** Criterion uses bootstrap resampling to estimate the population median; the parity binary computes the order-statistic median directly. For 50 samples both should agree closely, but with large variance the bootstrap can shift the estimate.

G17 distinguishes between these two by running a third mode in the parity binary.

---

## G17 — Criterion aggregation audit ONLY

### Goal

Produce a forensic record at `docs/evidence/2026-05-12-w33-criterion-aggregation-audit/README.md` on branch `forensic/w33-criterion-aggregation-audit` (cut from `forensic/w33-harness-parity-diagnostic @ 4a8031ef`) attributing the ~+32 µs aggregation phantom in Criterion's reported aggregate. The record must close one of two paths: (A) bench-harness fix identifiable, OR (B) documented Criterion-internal artifact. Branch stays unmerged. No design change.

### Strategies (GQM+Strategies)

* **S17.1** Cut `forensic/w33-criterion-aggregation-audit` from `forensic/w33-harness-parity-diagnostic @ 4a8031ef`. Worktree at `.worktrees/w33-criterion-audit`.
* **S17.2** Extend the existing `wcoj_harness_parity` binary (do NOT create a new binary) with a third timing mode `--mode=batch-amortized`:
  * Run inner work N times in a tight loop with a SINGLE `Instant::now()` start and SINGLE `Instant::now()` stop.
  * Vary `N ∈ {1, 5, 10, 25, 50, 100}` to expose any nonlinear amortization.
  * Record per-N: `total_us`, `mean_per_iter_us = total_us / N`, paired baseline + merge-resident.
  * Output a separate CSV at `docs/evidence/2026-05-12-w33-criterion-aggregation-audit/batch_amortization.csv` with columns: `cell, N, path, total_us, mean_per_iter_us`.
* **S17.3** Mirror Criterion's exact `iter_custom` pattern in the parity binary:
  * `--mode=criterion-mirror` invokes a closure of signature `|iters: u64| -> Duration` that exactly replicates how Criterion calls the bench harness.
  * Use Criterion's own default sample size (~100 samples).
  * Record per-sample raw `total / iters` mean.
  * Output CSV at `docs/evidence/2026-05-12-w33-criterion-aggregation-audit/criterion_mirror.csv`.
  * Compute bootstrap-resampled median (1000 resamples) over the 100 samples.
* **S17.4** Run a fresh true-Criterion bench using `wcoj_triangle_bench` with `--features wcoj-phase-timing`. Capture Criterion's own `sample.json` artifact at `target/criterion/.../new/sample.json` for both uniform-u32-10K and superhub-50K paired cells.
* **S17.5** Comparison tables in README:
  * **Table A: batch-amortization sweep.** For each `N`, show baseline + merge-resident `mean_per_iter_us` and paired delta. Identifies whether the per-iteration cost is constant-in-N (no amortization) or decreases with N (warm-cache amortization).
  * **Table B: criterion-mirror vs raw parity.** 100 samples each; compare order-statistic median, bootstrap median, mean, standard deviation. Identifies whether the bootstrap resampling shifts the median estimate beyond what raw order statistics show.
  * **Table C: criterion-mirror vs true-Criterion.** Parity-binary criterion-mirror median vs. Criterion's reported median from `sample.json`. If they match, parity-binary correctly reproduces Criterion's aggregation; if not, there's an additional Criterion-internal step.
* **S17.6** Verdict section in README:
  * Identify which source explains the ~+32 µs aggregation phantom (per-batch amortization / bootstrap-resampling / Criterion-internal-other).
  * Quantify the contribution of each identified source.
  * Recommend code-level next step:
    * If batch-amortization dominates: bench-harness fix = use `iter_custom` with `iters=1` (forces single-iteration batches, prevents amortization).
    * If bootstrap-resampling dominates: bench-harness fix = use order-statistic median estimator via Criterion config OR use `iter_batched` with explicit batch size = 1.
    * If Criterion-internal-other: G18 = document the discrepancy and close W3.3 on the G16 parity-Instant evidence.
* **S17.7** Branch UNMERGED to main, plan, G12-spike, G13-forensic, G14-respike, G15-forensic, AND G16-parity. No FF-merge, no push, no tag.
* **S17.8** Single bundled commit subject `forensic(w33): Criterion aggregation audit (batch amortization + bootstrap mirror)`. Final commit = forensic README.

### Questions

* **Q17.1** Branch HEAD SHA?
* **Q17.2** Batch-amortization Table A: does `mean_per_iter_us` decrease with N? If yes, by how much?
* **Q17.3** Criterion-mirror Table C: does parity-binary criterion-mirror median match Criterion's `sample.json`-derived median? If not, residual?
* **Q17.4** Identified dominant source of ~+32 µs phantom (per-batch amortization / bootstrap / Criterion-internal-other)?
* **Q17.5** Quantified contribution of each source.
* **Q17.6** Code-level next step recommendation grounded in confirmed source.
* **Q17.7** Branch unmerged from all seven parents?

### Metrics

* **M17.1** `forensic/w33-criterion-aggregation-audit` exists; HEAD reachable from neither main, plan, G12-spike, G13-forensic, G14-respike, G15-forensic, nor G16-parity.
* **M17.2** Parity binary has `--mode=batch-amortized` AND `--mode=criterion-mirror` AND retains existing modes.
* **M17.3** `cargo run -p xlog-integration --bin wcoj_harness_parity --release --features wcoj-phase-timing -- --mode=batch-amortized --cells=uniform-u32-10K,superhub-50K --batch-sizes=1,5,10,25,50,100` completes successfully.
* **M17.4** `cargo run -p xlog-integration --bin wcoj_harness_parity --release --features wcoj-phase-timing -- --mode=criterion-mirror --cells=uniform-u32-10K,superhub-50K --samples=100` completes successfully.
* **M17.5** `batch_amortization.csv` exists with all combinations of `cell × N × path`.
* **M17.6** `criterion_mirror.csv` exists with ≥ 100 samples per cell per path.
* **M17.7** Fresh true-Criterion run captured `sample.json` artifacts for both cells.
* **M17.8** README Tables A, B, C populated.
* **M17.9** Verdict names dominant source AND code-level next step.
* **M17.10** Strict scientific-control: `git diff 4a8031ef..HEAD -- 'crates/xlog-cuda/src/' 'crates/xlog-runtime/src/'` byte-empty.
* **M17.11** `cargo fmt --check --all` EXIT 0.
* **M17.12** `git tag --points-at HEAD` empty; `git ls-remote --heads origin "forensic/w33*"` empty.
* **M17.13** Branch unmerged from all seven parents.

### Supervisor validation per locked protocol

* Read evidence README end-to-end.
* `git rev-parse forensic/w33-criterion-aggregation-audit` ≠ all seven parent SHAs.
* Verify Table A shows per-N values for ≥ 6 batch sizes; if `mean_per_iter_us` decreases by > 5 µs across the N range, batch-amortization is the prime suspect.
* Verify Table C compares parity-criterion-mirror against `sample.json`-derived median.
* Verify verdict names a specific source AND a concrete code-level fix or documentation path.
* Verify M17.10 strict scientific control: impl paths untouched.
* Verify branch unmerged + no tag + no origin push.

If verdict = **batch-amortization** AND fix is identifiable: G18 = update `wcoj_triangle_bench.rs` to use `iters=1` batching, re-run G14 with clean PASS, then G19 = W3.3 closure proposal citing full G11–G18 chain.

If verdict = **bootstrap-resampling** AND fix is identifiable: G18 = update bench config to use order-statistic median or `iter_batched` with batch_size=1, re-run G14 with clean PASS, then G19 = closure.

If verdict = **Criterion-internal-other / not reducible**: G18 = W3.3 closure proposal citing G11–G17 evidence chain and documenting Criterion as a known validation-surface artifact rather than a real perf gate. W3.3 closes on parity-Instant evidence.

### Forbidden behaviors

* No `git push`, no `git tag`, no `--force`, no `--no-verify`, no `--dangerously-bypass`.
* No merge of `forensic/w33-criterion-aggregation-audit` into ANY other branch.
* No `docs/v065-closure-board.md` edit.
* No `v0.6.6` references.
* **No design proposal.** Audit only.
* No production-code change to `crates/xlog-cuda/src/` or `crates/xlog-runtime/src/`. Mandatory M17.10 verification.
* No new R6 anti-pattern.
* No fix attempts in this goal — verdict is informational; fixes (if identifiable) go in G18.
* No D7 amendment.
* Reuse the EXISTING `wcoj_harness_parity` binary; do NOT create a third binary. Add new modes via `--mode=*` flag dispatch.

### Why this is scoped tight

G16 attributed the residual to Criterion's aggregation pipeline. G17 attributes WHICH PART of the aggregation pipeline. Without this attribution, G18 either guesses at a bench-harness fix (likely wrong) or skips the fix entirely and relies on the parity-Instant evidence alone (acceptable but weaker closure). The two-mode addition (batch-amortized + criterion-mirror) plus the sample.json cross-check is the minimum experiment that distinguishes the candidates. The verdict directly determines G18's shape: bench-fix-then-close, or document-then-close.

Proceed: cut audit branch from `4a8031ef`, extend `wcoj_harness_parity` with batch-amortized and criterion-mirror modes, run both modes on both cells, capture sample.json from fresh true-Criterion run, build Tables A/B/C, identify dominant source + recommend code-level fix, single bundled commit. No merge, no push, no tag.
