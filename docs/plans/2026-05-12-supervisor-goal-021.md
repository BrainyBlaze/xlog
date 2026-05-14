# Supervisor Goal 021 — W3.3 Scale Sweep Under V3 Stability Protocol (Clean Closure Decision)

**Supervisor:** Claude Code.
**Implementer:** Codex CLI on tmux session `codex-xlog`.
**Predecessor:** G20 stability RCA-3 APPROVED. Forensic commit `43dc0b4a` on `forensic/w33-50K-stability-rca3`. **V3 sample_size(200) identified as smallest passing fix:** paired-delta stddev 4.215 µs (under ≤ 5 µs reproducibility target) vs V1 control 6.011 µs (FAIL) and V2 warmup-extended 10.603 µs (FAIL — worse). 88% of variance attributed to host-side: output_allocation_residual 7.07 µs stddev (48% share) + provider_call_dispatch 6.02 µs stddev (41% share). GPU-kernel + W3.3 design probes: all sub-µs. **The W3.3 design is NOT the noise source; the Criterion default sample size at iters=1 is.**
**Date:** 2026-05-12.

---

## Context

User decision recorded 2026-05-12: "G21 = rerun G19 scale sweep with V3 protocol" (Recommended option).

The 10-branch W3.3 chain has produced an end-to-end forensic story:
- G11: paper-aligned plan APPROVED
- G12 → G16: spike + harness-attribution chain
- G17: per-batch amortization confirmed as Criterion phantom source
- G18: iters=1 fix → D7b PASS clean, D7a apparent FAIL at superhub-50K
- G19: scale sweep BLOCKED by 50K reference instability (paired delta ±25 µs swing)
- G20: variance attributed to host allocator/dispatch (not design); V3 sample_size(200) fix identified

G21 cashes the V3 fix into the Criterion bench and reruns the scale sweep. If the 50K reference reproduces within ±5 µs AND scale-emergence resolves (either ≥ 2.0× at 200K/1M or measurably flat), the W3.3 closure decision is finally grounded in clean data.

### Why G21 base is G19, not G20

G19 HEAD `822aeb99` already added the `superhub-200K` and `superhub-1M` Criterion cells to `wcoj_triangle_bench.rs`. G20 only modified `wcoj_harness_parity.rs` (the diagnostic binary, not the Criterion bench). So G19 carries the necessary bench-cell scaffolding; G21 just needs to add `sample_size(200)` to the existing scale-sweep Criterion groups.

### What V3 fix means in code terms

The V3 protocol in G20 was: same G18 iters=1 paired-batching, but with Criterion's `sample_size(200)` instead of the default 100. In the actual Criterion bench (`wcoj_triangle_bench.rs`), this is a one-call addition to the `Criterion::default().sample_size(200)` configuration or `group.sample_size(200)` on the relevant benchmark group. No methodology shift, no design change.

---

## G21 — Scale sweep under V3 protocol

### Goal

Produce a Criterion measurement record at `docs/evidence/2026-05-12-w33-scale-sweep-v3-stable/README.md` on branch `bench-spike/w33-superhub-scale-sweep-v3-stable` (cut from `bench-spike/w33-superhub-scale-sweep @ 822aeb99`) reporting paired delta + 95% CI + speedup ratio for superhub-50K reference + superhub-200K + superhub-1M cells under the V3 sample_size(200) protocol. The 50K reference MUST reproduce within ±5 µs paired-delta stddev (or at least ±15 µs if absolute stddev is not directly measurable from a single Criterion run — defer to Criterion's reported 95% CI as the reproducibility proxy). If 50K reproduces and 200K/1M proceed, report speedup curve and threshold-crossing analysis. Branch stays unmerged. No design change. No production-impl change.

### Strategies (GQM+Strategies)

* **S21.1** Cut `bench-spike/w33-superhub-scale-sweep-v3-stable` from `bench-spike/w33-superhub-scale-sweep @ 822aeb99`. Worktree at `.worktrees/w33-scale-sweep-v3`.
* **S21.2** Modify ONLY `crates/xlog-integration/benches/wcoj_triangle_bench.rs`. Apply Criterion `sample_size(200)` to the superhub-50K/200K/1M benchmark group(s). Verify via `git diff 822aeb99..HEAD -- crates/ ':!crates/xlog-integration/benches/'` byte-empty.
* **S21.3** Use the EXACT G18 iters=1 paired-batching pattern inside the bench (already present at G19 base); only change is the `sample_size(200)` config. No methodology shift.
* **S21.4** Measure cells in scale-ascending order:
  1. `superhub-50K` reference rerun under V3 protocol. Use the G18+G19 paired-delta numbers (+10.962 µs and +36.655 µs) as the bounding "reproducibility envelope" — if G21's 50K paired delta falls between or near these values within Criterion's 95% CI, reproducibility is acceptable.
  2. `superhub-200K` (new cell, second).
  3. `superhub-1M` (new cell, third).
  Row-equality assertion PASS before timing at each scale.
* **S21.5** Stop conditions:
  * If `superhub-50K` paired-delta Criterion 95% CI is wildly outside G18 + G19 bracket (e.g., median outside [+0 µs, +50 µs] envelope OR CI width > 20 µs), STOP and report: V3 fix did not stabilize as predicted in production bench surface; recommend deeper RCA or fallback to V4/V5.
  * If `superhub-200K` shows row-equality FAIL, STOP and report (correctness violation at scale would be the BIG finding).
  * Otherwise measure all three; report speedup curve.
* **S21.6** Evidence README MUST contain:
  * Branch + all 10 parent SHAs explicit.
  * Diff summary: `sample_size(200)` change applied at named lines.
  * Per-scale: baseline median + 95% CI, merge-resident median + 95% CI, paired delta µs + 95% CI, paired delta %, speedup ratio, D7a ≥ 2.0× verdict, D7b ±5% verdict (if applicable).
  * Speedup curve table showing ratio at each scale + ratio trend direction.
  * Reproducibility verdict: did V3 stabilize the 50K reference?
  * Scale-emergence verdict: does ≥ 2.0× emerge at any tested scale?
  * Recommendation: close-with-threshold (if ratio ≥ 2.0× at some scale) / user-decision-required (if no scale shows ≥ 2.0× under stable measurement) / extend-sweep-larger (if ratio trends positive with scale but doesn't reach 2.0× by 1M).
* **S21.7** Branch UNMERGED to all eleven parents (main, plan, G12-spike, G13-forensic, G14-respike, G15-forensic, G16-parity, G17-audit, G18-respike-fixed, G19-scale-sweep, G20-stability-rca3). No FF-merge, no push, no tag.
* **S21.8** Single bundled commit subject `spike(w33): scale sweep under V3 stability protocol (sample_size=200)`. Final commit = evidence README.

### Questions

* **Q21.1** Branch HEAD SHA?
* **Q21.2** `git diff 822aeb99..HEAD -- crates/ ':!crates/xlog-integration/benches/'` byte-empty?
* **Q21.3** superhub-50K reference under V3: paired delta + 95% CI + verdict on reproducibility vs G18 (+10.96 µs) / G19 (+36.66 µs) envelope?
* **Q21.4** superhub-200K: paired delta + 95% CI + speedup ratio + D7a verdict?
* **Q21.5** superhub-1M: paired delta + 95% CI + speedup ratio + D7a verdict?
* **Q21.6** Speedup curve: ratio trend with scale? Threshold-crossing scale (if any)?
* **Q21.7** Reproducibility verdict: V3 stabilizes 50K (yes/no)?
* **Q21.8** Scale-emergence verdict + recommendation (close-with-threshold / user-decision-required / extend-sweep)?
* **Q21.9** Branch unmerged from all eleven parents?

### Metrics

* **M21.1** `bench-spike/w33-superhub-scale-sweep-v3-stable` exists; HEAD reachable from none of the 11 prior branches.
* **M21.2** `docs/evidence/2026-05-12-w33-scale-sweep-v3-stable/README.md` exists.
* **M21.3** `cargo bench -p xlog-integration --bench wcoj_triangle_bench --no-run` EXIT 0.
* **M21.4** `git diff 822aeb99..HEAD -- crates/ ':!crates/xlog-integration/benches/'` byte-empty.
* **M21.5** Row-equality PASS at all measured scales.
* **M21.6** Per-scale measurements with absolute µs + paired delta + 95% CI + speedup ratio + D7a verdict.
* **M21.7** Speedup curve table populated.
* **M21.8** Reproducibility and scale-emergence verdicts explicit in README.
* **M21.9** `cargo fmt --check --all` EXIT 0.
* **M21.10** `git tag --points-at HEAD` empty; `git ls-remote --heads origin "bench-spike/w33*"` empty.
* **M21.11** Branch unmerged from all 11 parents.

### Supervisor validation per locked protocol

* Read evidence README end-to-end.
* `git rev-parse bench-spike/w33-superhub-scale-sweep-v3-stable` ≠ all 11 parent SHAs.
* `cargo bench -p xlog-integration --bench wcoj_triangle_bench --no-run` EXIT 0 from supervisor session.
* Verify M21.4 strict scientific control (only `wcoj_triangle_bench.rs` changed).
* Verify the `sample_size(200)` change is present at a named line.
* Verify all three scales measured in ascending order.
* Verify reproducibility + scale-emergence verdicts unambiguous.
* Verify branch unmerged + no tag + no origin push.

If reproducibility VERDICT = YES and scale-emergence shows ≥ 2.0× at 1M: G22 = W3.3 closure proposal grounded in 11-iteration evidence chain. Board OPEN → DONE edit. Memory file. MEMORY.md update. FF-merge of `feat/w33-paper-aligned-plan-it1` to main.

If reproducibility = YES and ≥ 2.0× emerges at 200K: G22 = closure proposal documenting scale threshold "≥ 200K rows". Same closure ceremony.

If reproducibility = YES but no scale shows ≥ 2.0×: G22 = user-decision-required with finally-clean data — three sub-options (extend to 5M / amend D7a / defer to v0.6.6).

If reproducibility = NO: G22 = revert to V4 or V5 protocol; deeper stability work needed.

### Forbidden behaviors

* No `git push`, no `git tag`, no `--force`, no `--no-verify`, no `--dangerously-bypass`.
* No FF-merge of `bench-spike/w33-superhub-scale-sweep-v3-stable` into any other branch.
* No `docs/v065-closure-board.md` edit (board edit is G22's job, conditional on G21 outcome).
* No `v0.6.6` references in code.
* **No production-impl change** to `crates/xlog-cuda/src/` or `crates/xlog-runtime/src/`. Mandatory M21.4 verification.
* No methodology change beyond V3 sample_size(200) — keep iters=1 paired-batching exactly as G18 + G19 established.
* No D7 amendment.
* No "rescue" attempts if 50K reproducibility fails — STOP per S21.5 and report.
* No removal of row-equality assertions.
* No closure proposal in this goal.
* No measurement beyond superhub-50K/200K/1M — if 5M is needed, separate goal.

### Why this is scoped tight

G20 named the fix. G21 applies it to the production validation surface (the actual Criterion bench, not the parity diagnostic binary) and produces the scale-emergence verdict under stable measurement. One file changes (`wcoj_triangle_bench.rs`), one config call adds (`sample_size(200)`), three cells measure (50K reference + 200K + 1M). Either the chain closes (clean speedup at scale → W3.3 DONE) or we have empirically-grounded data for the final closure decision (no scale shows speedup under stable measurement → amend or defer). 11 iterations of forensic discipline finally cash out into actionable closure data.

Proceed: cut scale-sweep-v3-stable branch from `822aeb99`, add `sample_size(200)` to the superhub bench group in `wcoj_triangle_bench.rs`, run 50K reference + 200K + 1M cells in scale-ascending order with row-equality discipline, write evidence README with speedup curve table + reproducibility + scale-emergence verdicts + recommendation, single bundled commit. No merge, no push, no tag.
