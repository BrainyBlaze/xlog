# Supervisor Goal 024 — W3.3 V3 Scale Sweep On Slice-Aware Implementation (Final D7a Measurement)

**Supervisor:** Claude Code.
**Implementer:** Codex CLI on tmux session `codex-xlog`.
**Predecessor:** G23 production fix APPROVED. Commits `dcb556db` ← `cf7b8f0b` on `feat/w33-slice-aware-implementation`. All 15 M23 metrics green. Per-block-output stddev reduced 49.85% on superhub-50K (459.998 → 230.670). D7b uniform-u32-10K PASSES under V3 (+4.317 µs / +1.347% vs ±5% budget). Adaptive skew detection routes uniform paths to baseline kernels; weight-based slicing computes balanced slice boundaries from column degree maps. Zero R6 anti-patterns. Row-equality PASS across all tests + CUDA cert 1/1 + both builds compile.
**Date:** 2026-05-12.

---

## Context

User decision recorded 2026-05-12: "Yes, dispatch G24 V3 scale sweep now" (Recommended).

G23 delivered the slice-aware production implementation with measurable work-balancing (stddev −49.85%). The remaining empirical question for W3.3 closure is **end-to-end Criterion D7a speedup** at multiple fixture scales. The stddev reduction is necessary but not sufficient evidence: 4× more grid blocks means more kernel launch overhead, while better work-balancing reduces idle time waiting for slow blocks. Whether the net effect crosses the ≥ 2.0× D7a gate is what G24 measures.

### V3 protocol already on the bench

G23's evidence README confirms `crates/xlog-integration/benches/wcoj_triangle_bench.rs` already uses `group.sample_size(200)` (V3) via the G21 lineage carried forward. **G24 does NOT need to modify the bench harness.** G24 simply runs the existing bench against the slice-aware path at three scales.

### Closure path

If D7a ≥ 2.0× emerges at any tested scale (50K, 200K, or 1M):
- **G25 = closure proposal** documenting the scale threshold, plus board OPEN→DONE edit, memory file, MEMORY.md update, and FF-merge of `feat/w33-paper-aligned-plan-it1` to main (W2.5/W4.2/W4.3/W5.2 closure precedent).

If D7a ≥ 1.5× but < 2.0× at any scale (partial win):
- User-decision: extend to 5M / amend D7a with measured 1.X× rationale / deeper RCA on remaining overhead. The chain has produced enough data that the user can make an informed call without further forensic work.

If D7a < 1.5× at all scales:
- G25 = focused RCA-5 on why the work-balancing doesn't translate to wall-time speedup. Candidate causes: launch-overhead-eats-savings, slicing-pessimization-on-200K+, slice-bin-heuristic-suboptimal-at-scale.

---

## G24 — V3 scale sweep on slice-aware implementation

### Goal

Produce a Criterion measurement record at `docs/evidence/2026-05-12-w33-slice-aware-scale-validation/README.md` on branch `bench-spike/w33-slice-aware-scale-validation` (cut from `feat/w33-slice-aware-implementation @ dcb556db`) reporting baseline + merge-resident-sliced paired delta + 95% CI + speedup ratio for superhub at three scales (50K, 200K, 1M) under V3 sample_size(200) protocol. Branch stays unmerged. No production-impl change.

### Strategies (GQM+Strategies)

* **S24.1** Cut `bench-spike/w33-slice-aware-scale-validation` from `feat/w33-slice-aware-implementation @ dcb556db`. Worktree at `.worktrees/w33-slice-aware-scale`.

* **S24.2** Modify ONLY `crates/xlog-integration/benches/wcoj_triangle_bench.rs` IF and ONLY IF the existing G19+G21 cells (superhub-50K + superhub-200K + superhub-1M) are not already present. Verify via `git grep 'superhub-200K\|superhub-1M' feat/w33-slice-aware-implementation -- crates/xlog-integration/benches/`. If both cells are present, NO bench modification is needed and `git diff dcb556db..HEAD -- crates/` should be byte-empty. If missing, add them following the EXACT G19/G21 pattern.

* **S24.3** Use the EXACT G23 slice-aware path under measurement. No production-code change. The slice-aware kernels + adaptive skew detection + weight-based slicing from G23 are what's being measured.

* **S24.4** Measure cells in scale-ascending order:
  1. `superhub-50K` reference. Expected: paired delta near G18/G21 envelope BUT with the slice-aware path active; speedup ratio expected to improve significantly from G21's 0.982× toward ≥ 1.0× (parity) or higher.
  2. `superhub-200K` (new measurement under slice-aware path).
  3. `superhub-1M` (new measurement under slice-aware path).
  Row-equality assertion PASS before timing at each scale.

* **S24.5** Stop conditions:
  * If `superhub-50K` row-equality FAILs, STOP (correctness regression — critical bug).
  * If `superhub-50K` paired-delta Criterion 95% CI width > 40 µs (looser than G21's 20 µs because we're not gating on reproducibility now — we're measuring), report instability but proceed; if width > 80 µs, STOP and report.
  * Otherwise measure all three scales.

* **S24.6** Evidence README MUST contain:
  * Branch + all 13 parent SHAs explicit.
  * Confirmation that no production-impl files changed: `git diff dcb556db..HEAD -- crates/xlog-cuda/ crates/xlog-runtime/ --shortstat` should show 0 changed.
  * Per-scale: baseline median + 95% CI, merge-resident median + 95% CI, paired delta µs + 95% CI, paired delta %, speedup ratio (`baseline / merge`), D7a ≥ 2.0× verdict.
  * Speedup curve table with explicit ratio trend.
  * Cumulative comparison vs G21 (the pre-slice-aware buggy implementation at the same scales).
  * D7a verdict at each scale.
  * Recommendation: close-with-threshold (if ratio ≥ 2.0× at some scale) / user-decision-required-with-partial-win / RCA-5-required (if ratio < 1.5× everywhere).

* **S24.7** Branch UNMERGED to all 14 parents (main, plan, G12-spike, G13-forensic, G14-respike, G15-forensic, G16-parity, G17-audit, G18-respike-fixed, G19-scale-sweep, G20-stability-rca3, G21-scale-sweep-v3-stable, G22-design-behavior-rca4, G23-slice-aware-implementation). No FF-merge, no push, no tag.

* **S24.8** Single bundled commit subject `spike(w33): slice-aware scale validation (50K + 200K + 1M under V3)`. Final commit = evidence README.

### Questions

* **Q24.1** Branch HEAD SHA?
* **Q24.2** No production-impl change confirmed?
* **Q24.3** superhub-50K under slice-aware: baseline + merge medians + paired delta + 95% CI + speedup ratio + D7a verdict?
* **Q24.4** superhub-200K: same fields + D7a verdict?
* **Q24.5** superhub-1M: same fields + D7a verdict?
* **Q24.6** Speedup curve trend across scales? Threshold-crossing scale (if any)?
* **Q24.7** Comparison vs G21 buggy implementation at same scales: how much did slice-aware improve over scaffolding-stub?
* **Q24.8** Recommendation: close-with-threshold / user-decision-required / RCA-5-required?
* **Q24.9** Branch unmerged from all 14 parents?

### Metrics

* **M24.1** `bench-spike/w33-slice-aware-scale-validation` exists; HEAD reachable from none of 14 parents.
* **M24.2** `docs/evidence/2026-05-12-w33-slice-aware-scale-validation/README.md` exists.
* **M24.3** `cargo bench -p xlog-integration --bench wcoj_triangle_bench --no-run` EXIT 0.
* **M24.4** No production-impl change: `git diff dcb556db..HEAD -- crates/xlog-cuda/ crates/xlog-runtime/` byte-empty.
* **M24.5** Row-equality PASS at all measured scales.
* **M24.6** Per-scale measurements with median + 95% CI + paired delta + ratio + D7a verdict.
* **M24.7** Speedup curve table populated.
* **M24.8** Comparison-vs-G21 table populated.
* **M24.9** `cargo fmt --check --all` EXIT 0.
* **M24.10** `git tag --points-at HEAD` empty; `git ls-remote --heads origin "bench-spike/w33*"` empty.
* **M24.11** Branch unmerged from all 14 parents.

### Supervisor validation per locked protocol

* Read evidence README end-to-end.
* `git rev-parse bench-spike/w33-slice-aware-scale-validation` ≠ all 14 parent SHAs.
* Verify M24.4 strict no-impl-change (production code byte-identical to G23 HEAD).
* `cargo bench -p xlog-integration --bench wcoj_triangle_bench --no-run` EXIT 0 from supervisor session.
* Verify per-scale measurements present with row-equality verdicts.
* Verify speedup curve table populated; verify D7a verdict at each scale.
* Verify branch unmerged + no tag + no origin push.

If D7a ≥ 2.0× at any scale: supervisor confirms W3.3 closure is empirically grounded. Goal-025 = closure proposal grounded in G11-G24 evidence chain.

If D7a 1.5× ≤ ratio < 2.0× at any scale: supervisor presents partial-win options to user (extend sweep / amend gate / deeper RCA).

If D7a < 1.5× at all scales: supervisor reports unexpected null result; goal-025 = RCA-5 on launch-overhead vs work-balancing tradeoff.

### Forbidden behaviors

* No `git push`, no `git tag`, no `--force`, no `--no-verify`, no `--dangerously-bypass`.
* No FF-merge of `bench-spike/w33-slice-aware-scale-validation` into any other branch.
* No `docs/v065-closure-board.md` edit (G25's conditional job).
* No `v0.6.6` references in code.
* **No production-impl change.** `crates/xlog-cuda/src/`, `crates/xlog-cuda/kernels/`, `crates/xlog-runtime/src/` MUST stay byte-identical to G23 HEAD `dcb556db`. Mandatory M24.4 verification.
* No methodology change beyond what G18/G21/G23 already established (iters=1 + V3 sample_size(200) + paired-batching).
* No D7 amendment.
* No "rescue" attempts if row-equality fails at scale — STOP per S24.5 and report (correctness regression is critical).
* No closure proposal in this goal — G25's job, conditional on G24 result.
* No measurement beyond superhub-50K/200K/1M.

### Why this is the chain's empirical capstone

23 supervisor goals have produced: paper-aligned plan, identified scaffolding-stub root cause via code inspection + runtime probes, shipped a slice-aware production implementation with adaptive skew detection + weight-based slicing + measurable −49.85% stddev reduction, validated D7b preservation. The remaining question — does ≥ 2.0× speedup actually materialize at production scale — is what G24 answers. One Criterion bench, three scales, one verdict. Either W3.3 closes within 1 more goal, or the partial-win/null-result paths give the user grounded options.

Proceed: cut slice-aware-scale-validation branch from `dcb556db`, run scale sweep (50K + 200K + 1M) under V3 + iters=1 paired-batching against the slice-aware path, write evidence README with per-scale paired delta + speedup ratio + D7a verdicts + speedup curve + comparison-vs-G21 + recommendation, single bundled commit. No merge, no push, no tag.
