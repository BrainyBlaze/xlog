# Supervisor Goal 019 — W3.3 Superhub Scale Sweep (200K + 1M cells under fixed harness)

**Supervisor:** Claude Code.
**Implementer:** Codex CLI on tmux session `codex-xlog`.
**Predecessor:** G18 fixed-harness respike APPROVED. Commit `d217a9c5` on `bench-spike/w33-merge-resident-histogram-respike-fixed`. D7b uniform-u32-10K **PASS clean** (paired delta −0.418 µs / −0.279%, merge-resident slightly FASTER than baseline, budget +7.48 µs unused). D7a superhub-50K **FAIL** (paired delta +10.96 µs / +1.064%, speedup ratio 0.989× vs ≥ 2.0× target — no heavy-row acceleration at this scale). G17 phantom-vs-real diagnosis fully validated by 56.99 µs convergence between G14 buggy and G18 fixed harness.
**Date:** 2026-05-12.

---

## Context

User decision recorded 2026-05-12: "Test on larger superhub fixture (e.g., 200K, 1M) before deciding" (Recommended option).

G18 proved the merge-resident P3/P5 design is correctness-preserving and harness-validated, but the ≥ 2.0× heavy-row speedup at superhub-50K isn't delivered. Three plausible causes per G18 README:

1. **Fixture too small.** Superhub-50K may have insufficient skew for histogram-guided launch slicing to matter. Production workloads may have millions of rows.
2. **Slicing overhead == benefit.** Sub-µs slicing cost exactly offsets work-balancing benefit at this scale.
3. **Histogram-bin misalignment.** Bins not calibrated to the actual hub distribution shape.

G19 distinguishes (1) from (2)+(3) empirically: if speedup emerges at 200K or 1M scale, (1) is confirmed and W3.3 closes with documented scale threshold. If still no speedup at 1M, the design needs deeper investigation OR the gate needs amendment.

### Why this is the disciplined next step

The W3.3 plan's D7a clause specifies *"super-hub fixture's heavy-row case shows ≥ 2.0× speedup vs. uniform block dispatch on the canonical fixture"* without fixing a specific row count. The "canonical fixture" `tests/...adaptive_dispatch::superhub_fixture` was 50K rows in G18. The plan does not prohibit larger fixtures from being canonical; it constrains the bench shape (super-hub heavy-row case), not the size.

If the design delivers ≥ 2.0× at 1M but not 50K, that's a *scale-dependent* result that warrants documented threshold + acceptance, not a design failure. This is consistent with how W3.5 ("threshold below which the kernel reads sorted slot from `__shared__`") and W3.6 (warp-coop "below the W3.5 threshold") are board-encoded — perf optimizations naturally have scale thresholds.

---

## G19 — Scale sweep ONLY

### Goal

Produce a Criterion measurement record at `docs/evidence/2026-05-12-w33-superhub-scale-sweep/README.md` on branch `bench-spike/w33-superhub-scale-sweep` (cut from `bench-spike/w33-merge-resident-histogram-respike-fixed @ d217a9c5`) reporting paired delta + speedup ratio for superhub at three scales (50K reference, 200K, 1M). The record must enable a clean decision between "speedup emerges at scale → close W3.3 with threshold" and "no speedup at any tested scale → user-decision-required". Branch stays unmerged. No design change.

### Strategies (GQM+Strategies)

* **S19.1** Cut `bench-spike/w33-superhub-scale-sweep` from `bench-spike/w33-merge-resident-histogram-respike-fixed @ d217a9c5`. Worktree at `.worktrees/w33-scale-sweep`.
* **S19.2** Modify ONLY:
  * `crates/xlog-integration/benches/wcoj_triangle_bench.rs` — add `superhub-200K` and `superhub-1M` cells under the same fixed iters=1 paired-batching harness G18 introduced.
  * The fixture-generation code (likely in `crates/xlog-integration/tests/` or wherever `superhub_fixture` is defined) — extend to parameterize size. **MAY** modify the fixture-generator file IF it's currently 50K-only; verify what file the existing superhub_fixture lives in and extend it with the same shape at the new sizes.
  * All other files under `crates/` MUST be byte-identical to G18 HEAD `d217a9c5`. Verify via `git diff d217a9c5..HEAD -- crates/ ':!crates/xlog-integration/'` byte-empty (allow `crates/xlog-integration/**` since both bench file and possibly fixture file live there).
* **S19.3** Use the EXACT iters=1 paired-batching pattern from G18's `wcoj_triangle_bench.rs` for the new cells. No deviation from G18's measurement methodology — only the fixture size changes.
* **S19.4** Measure cells in scale-ascending order:
  1. `superhub-50K` (reference re-run on G19 branch to confirm baseline reproducibility; expected paired delta ≈ +10.96 µs / 0.989× from G18).
  2. `superhub-200K` (first new cell).
  3. `superhub-1M` (second new cell).
  Row-equality assertion PASS before timing at each scale.
* **S19.5** Stop-condition handling:
  * If `superhub-50K` reference re-run produces a wildly different paired delta from G18 (e.g., outside ±20 µs), STOP and report harness instability — do not proceed to larger fixtures.
  * If `superhub-200K` shows row-equality FAIL, STOP and report (correctness violation at scale would be the BIG finding).
  * Otherwise measure all three scales; the verdict is purely informational.
* **S19.6** Evidence README MUST contain:
  * Branch + all parent SHAs (G11/G12/G13/G14/G15/G16/G17/G18/main) explicit.
  * Diff summary (bench harness + fixture generator if extended).
  * Per-scale: baseline median, merge-resident median, paired delta µs, paired delta %, speedup ratio, D7a ≥ 2.0× pass/fail verdict.
  * Speedup curve table showing ratio trend with scale.
  * Verdict section: "speedup emerges at scale X" / "no speedup at any tested scale".
  * Recommendation: if speedup emerges, G20 = closure proposal documenting scale threshold; if not, G20 = user-decision (amend D7a / deeper RCA / defer W3.3).
* **S19.7** Branch UNMERGED to main, plan, G12-spike, G13-forensic, G14-respike, G15-forensic, G16-parity, G17-audit, G18-respike-fixed. No FF-merge, no push, no tag.
* **S19.8** Single bundled commit subject `spike(w33): superhub scale sweep (50K reference + 200K + 1M cells)`. Final commit = evidence README.

### Questions

* **Q19.1** Scale-sweep branch HEAD SHA?
* **Q19.2** `git diff d217a9c5..HEAD -- crates/ ':!crates/xlog-integration/'` empty? (proves no W3.3-impl changes outside xlog-integration).
* **Q19.3** `superhub-50K` reference re-run paired delta: matches G18's +10.96 µs within Criterion CI? If not, report instability.
* **Q19.4** `superhub-200K` paired delta + speedup ratio. D7a verdict?
* **Q19.5** `superhub-1M` paired delta + speedup ratio. D7a verdict?
* **Q19.6** Speedup curve: does ratio increase with scale? At what scale (if any) does ≥ 2.0× threshold cross?
* **Q19.7** Recommendation: close-with-threshold OR user-decision-required?
* **Q19.8** Branch unmerged from all nine parents?

### Metrics

* **M19.1** `bench-spike/w33-superhub-scale-sweep` exists; HEAD reachable from neither main, plan, G12-spike, G13-forensic, G14-respike, G15-forensic, G16-parity, G17-audit, nor G18-respike-fixed.
* **M19.2** `docs/evidence/2026-05-12-w33-superhub-scale-sweep/README.md` exists.
* **M19.3** `cargo bench -p xlog-integration --bench wcoj_triangle_bench --no-run` EXIT 0.
* **M19.4** `git diff d217a9c5..HEAD -- crates/ ':!crates/xlog-integration/'` byte-empty.
* **M19.5** Row-equality PASS at all three scales (50K reference, 200K, 1M).
* **M19.6** Per-scale measurements reported with absolute µs medians + paired delta + speedup ratio + D7a verdict.
* **M19.7** Speedup curve table populated showing ratio at each scale.
* **M19.8** Verdict: "speedup emerges at scale X" or "no speedup at any tested scale".
* **M19.9** `cargo fmt --check --all` EXIT 0.
* **M19.10** `git tag --points-at HEAD` empty; `git ls-remote --heads origin "bench-spike/w33*"` empty.
* **M19.11** Branch unmerged from all nine parents.

### Supervisor validation per locked protocol

* Read evidence README end-to-end.
* `git rev-parse bench-spike/w33-superhub-scale-sweep` ≠ all 9 parent SHAs.
* `cargo bench -p xlog-integration --bench wcoj_triangle_bench --no-run` EXIT 0 from supervisor session.
* Verify M19.4 strict scientific control (no W3.3 impl code changed outside xlog-integration).
* Verify all three scales measured in ascending order.
* Verify speedup curve shows ratio trend; verify D7a verdict at each scale.
* Verify branch unmerged + no tag + no origin push.

If `superhub-1M` shows ratio ≥ 2.0×: supervisor confirms scale-emergent speedup; G20 = closure proposal documenting scale threshold (e.g., "≥ 2.0× holds at 1M+ rows"), board OPEN → DONE edit, memory file, MEMORY.md update, FF-merge.

If no scale shows ratio ≥ 2.0× but speedup ratio trends positive with scale: supervisor reports trend evidence; G20 = user-decision-required with three sub-options (extend sweep to 5M / amend D7a with measured rationale / defer W3.3).

If speedup ratio is flat or negative across scales: supervisor reports no-scale-emergence; G20 = user-decision-required with the case for amend-D7a-or-defer being stronger.

### Forbidden behaviors

* No `git push`, no `git tag`, no `--force`, no `--no-verify`, no `--dangerously-bypass`.
* No FF-merge of `bench-spike/w33-superhub-scale-sweep` into any other branch.
* No `docs/v065-closure-board.md` edit.
* No `v0.6.6` references.
* **No implementation-code change to `crates/xlog-cuda/src/` or `crates/xlog-runtime/src/`.** Mandatory M19.4 verification.
* No R6 anti-pattern.
* No D7 amendment.
* No "rescue" attempts on superhub-50K reference re-run instability — STOP per S19.5 and report.
* No removal of row-equality assertions.
* No closure proposal in this goal — that's G20's job, dependent on G19 outcome.
* No measurement of fixtures beyond superhub-50K/200K/1M in this goal. If user wants 5M, that's a separate goal.

### Why this is scoped tight

G18 produced a partial result (D7b PASS / D7a FAIL at 50K). The cleanest empirical question is "does the speedup emerge at scale?" — a binary check with three measurements. Two new cells, one reference cell, single commit. If the answer is yes, W3.3 closes; if no, the user decides between amend-or-defer with evidence. Either outcome unblocks the rest of v0.6.5 within one or two more goals.

Proceed: cut scale-sweep branch from `d217a9c5`, add superhub-200K and superhub-1M cells using the EXACT G18 iters=1 harness pattern, run all three scales in ascending order with row-equality discipline, write evidence README with speedup curve table + verdict + recommendation, single bundled commit. No merge, no push, no tag.
