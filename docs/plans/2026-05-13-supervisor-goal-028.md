# Supervisor Goal 028 — W3.3 Persistent-Threads Scale Validation (superhub-200K + 1M)

**Supervisor:** Claude Code.
**Implementer:** Codex CLI on tmux session `codex-xlog`.
**Predecessor:** G27 NOT-CLOSURE-READY at superhub-50K despite persistent-threads + heavy-slice splitting + atomic dispatch counter all engineering-sound (correctness PASS, routing PASS, CUDA cert 1/1, both builds compile). The 50K fixture is too small for the work-balancing payoff to overcome the GPU kernel-launch latency floor.
**Date:** 2026-05-13.

---

## Context

User directive recorded 2026-05-13 (Path 1 selected over Path 2): scale the canonical fixture upward before considering D7a amendment.

**The chain's empirical narrative** going into G28:

| Architecture | Cell | Ratio | Failure mode |
|---|---|---:|---|
| G23 static 468 blocks | 50K | 0.555× (G24) | Launch overhead 4× baseline |
| G26 Phase 2 static 117 blocks (grid-stride) | 50K | 0.407× | Work-balancing collapsed to 0.03% |
| G27 persistent threads + heavy-slice splitting | 50K | **STILL FAILS** | Per Codex: "D7a failed despite correctness and routing being sound" |

The G27 architecture is the "have your cake and eat it" pattern — bounded slice size + dynamic distribution + single launch. If it can't deliver ≥ 2.0× at 50K, the gate is **likely structurally unreachable at 50K** because the GPU launch-latency floor (~30-50µs even for the cleanest kernel) dominates any algorithmic improvement when total work is only 50K rows.

### Hypothesis for scale-emergence

Heavy-row variance grows with fixture size:
- 50K: max-output-row per block ~5041 (24× median 211)
- 200K: expected max ~20K+ (heavy-row concentration scales)
- 1M: expected max ~100K+ (very pronounced imbalance)

The kernel-launch latency floor is roughly constant (~30-50µs); but **the work each kernel performs scales linearly with fixture size**. So the payoff ratio of work-balancing benefit / launch overhead **grows favorably with scale**. At 200K, the ratio should be 4× more favorable than 50K; at 1M, 20× more favorable.

**G28 tests this hypothesis empirically.** If D7a ≥ 2.0× emerges at 200K or 1M, W3.3 closes with documented scale threshold. If it doesn't emerge at any tested scale, the gate is empirically proven unreachable on this architecture and Path 2 (D7a amendment) becomes data-grounded.

---

## G28 — Scale validation on persistent-threads implementation

### Goal

Cut `bench-spike/w33-persistent-threads-scale-validation` from `feat/w33-persistent-threads-work-stealing @ <G27 HEAD>` (the head of G27 after its 5-commit cascade lands). Run V3 sample_size(200) Criterion measurement at:
- `superhub-50K` reference (confirm G27 result reproduces under fresh measurement)
- `superhub-200K`
- `superhub-1M`

Report paired delta + 95% CI + speedup ratio + D7a verdict at each scale. Branch stays unmerged. No production-impl change.

### Strategies (GQM+Strategies)

* **S28.1** Cut `bench-spike/w33-persistent-threads-scale-validation` from G27 HEAD (whatever SHA results from the 5-commit cascade). Worktree at `.worktrees/w33-persistent-scale-val`.

* **S28.2 — Scientific control.** No production-impl change. Modify ONLY `crates/xlog-integration/benches/wcoj_triangle_bench.rs` IF AND ONLY IF the existing G19/G21/G24 cells (superhub-50K + 200K + 1M) are not already present. Verify via `git grep 'superhub-200K\|superhub-1M' crates/xlog-integration/benches/`. If present, NO bench modification needed.

* **S28.3 — V3 protocol preserved.** Use the exact V3 sample_size(200) + iters=1 paired-batching pattern. No methodology change. The implementation under test is G27's persistent-threads + heavy-slice splitting code path.

* **S28.4 — Scale-ascending measurement.** Measure in order:
  1. `superhub-50K` reference. Expected: reproduce G27's measured D7a ratio within ±20% (sanity check that the G27 result is stable across runs).
  2. `superhub-200K` new measurement under persistent-threads path.
  3. `superhub-1M` new measurement under persistent-threads path.
  Row-equality PASS at each scale BEFORE timing.

* **S28.5 — Stop conditions.**
  * If `superhub-50K` row-equality FAILs OR ratio diverges wildly from G27 (e.g., outside [0.2×, 1.0×]), STOP and report harness-instability.
  * If `superhub-200K` row-equality FAILs, STOP (correctness regression at scale would be the big finding).
  * Otherwise measure all three.

* **S28.6 — Evidence README.** `docs/evidence/2026-05-13-w33-persistent-threads-scale-validation/README.md` MUST contain:
  * All 18 parent SHAs explicit (main + G11–G26 + 6595b969 + G27).
  * Confirmation: `git diff <G27_HEAD>..HEAD -- crates/xlog-cuda/ crates/xlog-runtime/` byte-empty.
  * Per-scale: baseline median + 95% CI, merge-resident median + 95% CI, paired delta µs + 95% CI, paired delta %, speedup ratio (`baseline / merge`), D7a ≥ 2.0× verdict.
  * Speedup curve table + ratio trend direction.
  * Comparison vs G27's 50K result (was it reproduced?).
  * Comparison vs G24's 50K result (G23 architecture) and G26 Phase 2's 50K result (grid-stride architecture).
  * Closure-readiness verdict: "W3.3 closure-ready at scale X" / "no scale clears D7a — Path 2 amendment empirically grounded".

* **S28.7** Branch UNMERGED to all 18 parents. No FF-merge, no push, no tag.

* **S28.8** Single bundled commit subject `spike(w33): persistent-threads scale validation (50K + 200K + 1M under V3)`. Final commit = evidence README.

### Questions

* **Q28.1** Branch HEAD SHA?
* **Q28.2** No production-impl change confirmed?
* **Q28.3** superhub-50K reference under persistent threads: baseline + merge medians + 95% CI + paired delta + ratio + D7a verdict?
* **Q28.4** superhub-200K: same fields + D7a verdict?
* **Q28.5** superhub-1M: same fields + D7a verdict?
* **Q28.6** Speedup curve trend: does ratio increase with scale? Threshold-crossing scale (if any)?
* **Q28.7** Comparison vs G24 (G23 architecture) + G26 Phase 2 (grid-stride) + G27 (persistent threads) at each scale where comparable data exists?
* **Q28.8** Closure-readiness verdict + recommendation?
* **Q28.9** Branch unmerged from all 18 parents?

### Metrics

* **M28.1** Branch exists; HEAD reachable from none of 18 parents.
* **M28.2** Evidence README exists with measurement tables.
* **M28.3** `cargo bench --no-run` EXIT 0.
* **M28.4** `git diff <G27_HEAD>..HEAD -- crates/xlog-cuda/ crates/xlog-runtime/` byte-empty.
* **M28.5** Row-equality PASS at all measured scales.
* **M28.6** Per-scale: medians + CI + paired delta + ratio + D7a verdict.
* **M28.7** Speedup curve populated with explicit trend direction.
* **M28.8** Closure-readiness verdict explicit.
* **M28.9** `cargo fmt --check --all` EXIT 0.
* **M28.10** No tag; no origin push.
* **M28.11** Branch unmerged from all 18 parents.

### Supervisor validation per locked protocol

* Read evidence README end-to-end.
* `git rev-parse <branch>` ≠ all 18 parent SHAs.
* `cargo bench --no-run` EXIT 0 from supervisor session.
* Verify M28.4 strict no-impl-change.
* Verify per-scale measurements + D7a verdicts present.
* Verify closure-readiness recommendation grounded in data.
* Verify branch unmerged + no tag + no origin push.

If D7a ≥ 2.0× at any scale: G29 = W3.3 closure proposal grounded in G11–G28 evidence chain documenting scale threshold + FF-merge of `feat/w33-paper-aligned-plan-it1` to main.

If D7a < 2.0× at all scales: G29 = present empirically-grounded D7a amendment proposal to user (Path 2 fallback) with full chain data tables.

### Forbidden behaviors

* No `git push`, no `git tag`, no `--force`, no `--no-verify`, no `--dangerously-bypass`.
* No FF-merge into main.
* No `docs/v065-closure-board.md` edit (G29's conditional job).
* No `v0.6.6` references.
* **No production-impl change.** `crates/xlog-cuda/` and `crates/xlog-runtime/` byte-identical to G27 HEAD. M28.4 mandatory.
* No methodology change beyond V3 + iters=1 + paired-batching.
* No D7 amendment in this goal.
* No closure proposal in this goal.
* No measurement beyond superhub-50K/200K/1M.

### Why this is the empirical decision point

After 18 W3.3 iteration branches and 3 architectural attempts (G23 static-468, G26 static-117 grid-stride, G27 persistent-threads), G28 is the final empirical test: does the work-balancing benefit scale-emerge to deliver ≥ 2.0× at production-realistic fixture sizes? If yes → W3.3 closes cleanly. If no → the data justifies D7a amendment with the entire G11–G28 chain as supporting evidence.

Proceed: cut scale-validation branch from G27 HEAD, run V3 Criterion at all three scales in ascending order with row-equality discipline, write evidence README with speedup curve + D7a verdicts + closure-readiness recommendation, single bundled commit. Emit REVIEW REQUEST with HEAD SHA + per-scale ratios + closure-readiness recommendation.
