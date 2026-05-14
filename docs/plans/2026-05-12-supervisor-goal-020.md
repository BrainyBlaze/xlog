# Supervisor Goal 020 — W3.3 50K Reference Stability Diagnostic (Variance Attribution RCA-3)

**Supervisor:** Claude Code.
**Implementer:** Codex CLI on tmux session `codex-xlog`.
**Predecessor:** G19 scale sweep stopped per S19.5(a). Commit `822aeb99` on `bench-spike/w33-superhub-scale-sweep`. 50K reference re-run paired delta `+36.655 µs / +3.719% / ratio 0.964×` vs G18's `+10.962 µs / +1.064% / 0.989×` — Δ `+25.693 µs`, outside ±20 µs band. 200K and 1M cells SKIPPED. The merge-resident path on superhub-50K has inherent run-to-run paired-delta variance ≥ 25 µs that exceeds the resolution needed for the ≥ 2.0× gate.
**Date:** 2026-05-12.

---

## Context

User decision recorded 2026-05-12: "G20 = stability diagnostic on 50K reference (RCA-3)" (option 2 of four).

The W3.3 chain has produced two paired-delta measurements at superhub-50K under the same fixed iters=1 harness:

| Run | Paired delta µs | Speedup ratio |
|---|---:|---:|
| G18 (run 1) | +10.962 | 0.989× |
| G19 (run 2) | +36.655 | 0.964× |
| Difference | +25.693 | — |

The chain cannot proceed to closure or to scale-emergence testing until this variance is attributed to a specific source AND ideally a methodology fix is identified. G20 produces that attribution.

### Four candidate variance sources (per G19 verdict + standard Criterion benchmark hygiene)

1. **System / scheduler noise.** Background CPU processes, CUDA driver state changes, WSL2 hypervisor scheduling. Fix candidates: fixed CPU/GPU clocks (`nvidia-smi -lgc`), CPU pinning (`taskset`), running multiple consecutive trials and averaging.

2. **Per-launch kernel-launch variance.** GPU launch latency has its own jitter even when the kernel itself is deterministic. The merge-resident path includes additional launch-time slicing logic that may have higher launch jitter than baseline. Fix candidates: larger warm-up loop before timing, paired-difference rather than absolute timing per sample, accepting GPU launch-floor variance as a hard limit.

3. **Criterion sampling artifact at iters=1.** Single-iteration samples have inherently higher variance than multi-iteration samples. The G18 iters=1 fix avoided per-batch amortization but may have introduced sample-size noise. Fix candidates: `sample_size(200)` instead of default 100, median-of-medians across multiple Criterion runs, switch to `iter_batched` with `BatchSize::PerIteration` (Criterion idiom that's semantically iters=1 but with internal stability heuristics).

4. **Warmup effects.** First iteration of each sample may differ from subsequent (cold cache, allocator first-touch). Fix candidates: explicit warmup loop before Criterion-timed region, or paired-iter ordering that ensures both paths see the same warmup state.

### Why this is the correct discipline

G19's outcome wasn't "design fails at scale" — it was "we can't reliably measure at this scale". Skipping G20 and jumping to closure would mean committing to either gate-amendment or deferral based on a noisy measurement. The user explicitly chose the disciplined attribution path. Once G20 identifies the variance source AND a fix, G21 = scale sweep with the fix applied, and a clean decision becomes possible.

---

## G20 — Stability diagnostic ONLY

### Goal

Produce a forensic record at `docs/evidence/2026-05-12-w33-50K-stability-rca3/README.md` on branch `forensic/w33-50K-stability-rca3` (cut from `bench-spike/w33-superhub-scale-sweep @ 822aeb99`) attributing the ~25 µs run-to-run paired-delta variance on superhub-50K to one or more of the four candidate sources, and identifying the smallest methodological change that produces reproducible measurements (defined as: ≤ 5 µs run-to-run paired-delta variance across M=10 consecutive runs). Branch stays unmerged. No design change. No production-impl change.

### Strategies (GQM+Strategies)

* **S20.1** Cut `forensic/w33-50K-stability-rca3` from `bench-spike/w33-superhub-scale-sweep @ 822aeb99`. Worktree at `.worktrees/w33-50K-stability`.
* **S20.2** Extend the existing `wcoj_harness_parity` binary (the one used in G16/G17) with a new `--mode=stability-rca` mode. **Do NOT create another binary.** All other files under `crates/` MUST be byte-identical to G19 HEAD `822aeb99` except `wcoj_harness_parity.rs`. Verify via `git diff 822aeb99..HEAD -- crates/ ':!crates/xlog-integration/src/bin/wcoj_harness_parity.rs'` byte-empty.
* **S20.3** `--mode=stability-rca` measurement protocol:
  * Run M = 10 consecutive 50K paired-iter cells under each of 5 methodology variants:
    * **V1 (baseline / control):** Exact G18 iters=1 paired-batching shape. Expected to reproduce G18-style ±25 µs variance.
    * **V2 (warmup-extended):** V1 with explicit warmup loop of 100 paired launches before Criterion-timed region.
    * **V3 (sample-size-200):** V1 with `sample_size(200)` instead of default 100.
    * **V4 (multi-run-median):** V1 run 5 times consecutively; report median-of-medians instead of single-run median.
    * **V5 (combined):** V2 + V3 + V4 combined.
  * For each variant, record per-run baseline median, merge-resident median, paired delta, paired delta %, speedup ratio.
  * Output CSV at `docs/evidence/2026-05-12-w33-50K-stability-rca3/stability_runs.csv` with columns: `variant, run_idx, baseline_us, merge_us, paired_delta_us, paired_delta_pct, ratio`.
* **S20.4** Phase-time each run using existing `wcoj-phase-timing` feature gates to capture: completion-sync µs, output-allocation µs, provider-call-dispatch µs, merge-resident-histogram-refresh µs, launch-time-slice-read µs, metadata-maintenance µs. Per-run phase breakdown attributes WHICH phase contributes most to per-run variance.
* **S20.5** Variance analysis tables in README:
  * **Table A: per-variant paired-delta variance.** For each variant V1-V5: M=10 runs, report min/Q1/median/Q3/max/stddev of paired delta. Identifies which variant produces ≤ 5 µs run-to-run variance.
  * **Table B: per-phase variance.** For V1 (control), per-phase µs across 10 runs: min/median/max/stddev for each phase. Identifies WHICH phase has the highest variance contribution.
  * **Table C: variance attribution.** Map total paired-delta variance to phase contributions: e.g., "of the ±25 µs total variance, X µs comes from completion-sync, Y µs from output-allocation, etc."
* **S20.6** Verdict section in README:
  * Identify dominant variance source from Table C.
  * Identify smallest-fix variant from Table A that meets ≤ 5 µs run-to-run reproducibility.
  * Recommend G21 scope: rerun scale sweep with the identified fix variant. State the expected resolution boundary (e.g., "≤ 5 µs run-to-run after fix; ≥ 2.0× gate is now resolvable").
  * If NO variant reaches ≤ 5 µs reproducibility: declare "variance is inherent to hardware/CUDA at this fixture size" and recommend user-decision-required between scale-up / gate-amendment / defer.
* **S20.7** Branch UNMERGED to all ten parents (main, plan, G12-spike, G13-forensic, G14-respike, G15-forensic, G16-parity, G17-audit, G18-respike-fixed, G19-scale-sweep). No FF-merge, no push, no tag.
* **S20.8** Single bundled commit subject `forensic(w33): 50K reference stability RCA-3 (variance attribution + fix-candidate sweep)`. Final commit = forensic README.

### Questions

* **Q20.1** Forensic branch HEAD SHA?
* **Q20.2** V1 (control) paired-delta variance across M=10 runs: min/Q1/median/Q3/max/stddev? Does it reproduce G18/G19's ~±25 µs variance?
* **Q20.3** Per-phase variance attribution: which phase carries the largest variance contribution?
* **Q20.4** V2 (warmup), V3 (sample-size-200), V4 (multi-run-median), V5 (combined): which variants achieve ≤ 5 µs run-to-run reproducibility on paired delta?
* **Q20.5** If a fix variant works: what's its variance + median? If multiple work, which is the smallest methodological change?
* **Q20.6** If no variant works: what's the noise floor? What user-decision is recommended?
* **Q20.7** Branch unmerged from all ten parents?

### Metrics

* **M20.1** `forensic/w33-50K-stability-rca3` exists; HEAD reachable from neither main, plan, nor any of the 9 prior W3.3 branches.
* **M20.2** `docs/evidence/2026-05-12-w33-50K-stability-rca3/README.md` exists.
* **M20.3** `cargo run -p xlog-integration --bin wcoj_harness_parity --release --features wcoj-phase-timing -- --mode=stability-rca --runs=10 --variants=v1,v2,v3,v4,v5` completes successfully (or equivalent invocation matching the parity binary's CLI conventions).
* **M20.4** `stability_runs.csv` exists with `10 runs × 5 variants × 2 paths = 100 rows minimum` (more if V4 multi-run-median adds rows).
* **M20.5** Tables A, B, C populated in README with explicit µs numbers.
* **M20.6** Verdict names dominant variance source AND recommends a specific G21 scope (with fix variant) OR a specific user-decision.
* **M20.7** Strict scientific control: `git diff 822aeb99..HEAD -- crates/ ':!crates/xlog-integration/src/bin/wcoj_harness_parity.rs'` byte-empty.
* **M20.8** `cargo fmt --check --all` EXIT 0.
* **M20.9** `git tag --points-at HEAD` empty; `git ls-remote --heads origin "forensic/w33*"` empty.
* **M20.10** Branch unmerged from all 10 parents.

### Supervisor validation per locked protocol

* Read evidence README end-to-end.
* `git rev-parse forensic/w33-50K-stability-rca3` ≠ all 10 parent SHAs.
* Verify M20.7 strict scientific control (only `wcoj_harness_parity.rs` changed under crates/).
* Spot-check stability_runs.csv: at least 100 rows; each row has baseline + merge µs + paired delta.
* Verify Table A shows per-variant variance with stddev or min/max numbers.
* Verify Table B / C populate per-phase µs numbers from the wcoj-phase-timing feature.
* Verify verdict names dominant source AND a concrete G21 recommendation.
* Verify branch unmerged + no tag + no origin push.

If verdict identifies a working fix variant: G21 = rerun the G19 scale sweep with that fix, expecting reproducible measurements.
If no fix variant works: G21 = closure-decision (defer / amend D7a / accept D7b-only) on the user's call.

### Forbidden behaviors

* No `git push`, no `git tag`, no `--force`, no `--no-verify`, no `--dangerously-bypass`.
* No merge of `forensic/w33-50K-stability-rca3` into ANY other branch.
* No `docs/v065-closure-board.md` edit.
* No `v0.6.6` references in code (the verdict text may mention deferral as a recommendation option).
* **No production-code change** to `crates/xlog-cuda/src/` or `crates/xlog-runtime/src/`. Mandatory M20.7 verification.
* **No new binary.** Reuse existing `wcoj_harness_parity` with new `--mode=stability-rca` flag.
* No R6 anti-pattern.
* No D7 amendment.
* No closure proposal in this goal — G21 or later, dependent on G20 verdict.
* No fix attempts that change provider/runtime code — fixes are bench/harness methodology only.

### Why this is scoped tight

G19 produced "can't measure" rather than "design failed". G20 attributes WHY we can't measure and identifies the smallest fix. Five variants is enough to distinguish between the four candidate sources (V1 baseline + one variant per fix-class). 10 runs per variant is the minimum statistically meaningful sample for variance estimation. Phase-timing each run isolates which operation contributes the noise. The verdict directly enables either (a) G21 = re-run scale sweep with fix, or (b) clean user-decision on the chain's terminal state.

Proceed: cut stability-rca-3 branch from `822aeb99`, extend `wcoj_harness_parity` with `--mode=stability-rca`, run 5 variants × 10 runs at superhub-50K with phase-timing on, build Tables A/B/C, identify dominant variance source + smallest-fix variant + recommend G21 OR user-decision, single bundled commit. No merge, no push, no tag.
