# Supervisor Goal 012 — W3.3 Bench Spike (Merge-Resident Histogram, Cell Measurements Only)

**Supervisor:** Claude Code.
**Implementer:** Codex CLI.
**Predecessor:** G11 APPROVED. Plan iteration 1 durable at `a4c299fd9a2e9b2ca295464e13f815ba9c35d90a` on `feat/w33-paper-aligned-plan-it1`, unmerged. User explicitly approved plan content via supervisor goal-011's Plan-Approval Gate.
**Date:** 2026-05-11.

---

## Context

W3.3 plan iteration 1 (`docs/plans/2026-05-12-w33-paper-aligned-plan.md` at `a4c299fd`) defines the Bench-Spike-First Protocol (§ Bench-Spike-First Protocol, lines 72–88) and Step Sequence (§ Step 2, lines 108–137). G12 implements ONLY the spike. The plan's D1 LOCK says: *"The next goal after this plan is only the spike on `bench-spike/w33-merge-resident-histogram`. Production implementation starts only after spike evidence is reviewed and explicitly approved."*

The spike exists to bound D7b's tightest cell (`uniform-u32-10K`, +11 µs budget) before any production-code expansion. Per `feedback_perf_bench_spike_first.md`, the spike branch stays unmerged regardless of outcome — it is measurement evidence, not a delivery vehicle.

---

## G12 — Bench spike measurement ONLY

### Goal

Produce a measurement record at `docs/evidence/2026-05-12-w33-merge-resident-histogram-spike/README.md` on branch `bench-spike/w33-merge-resident-histogram` cut from `feat/w33-paper-aligned-plan-it1` at `a4c299fd`. The record contains row-equality-asserted Criterion medians + ratios for `uniform-u32-10K` (D7b binding cell) and `superhub-50K` (D7a binding cell). Implementation must be the **minimum-viable** relation-resident histogram surface needed to time these two cells — nothing more.

### Strategies (GQM+Strategies)

* **S12.1** Cut `bench-spike/w33-merge-resident-histogram` from `feat/w33-paper-aligned-plan-it1` at `a4c299fd`. Use a worktree at `.worktrees/w33-spike` to keep main's working tree clean.
* **S12.2** Implement the minimum-viable relation-resident histogram surface respecting plan locks D2/D3/D4: histogram lives on `CudaBuffer` or named sibling metadata struct (D2); refresh point is Merge phase (D3); consumption point is kernel launch-time work-unit slice assignment (D4). Allowed files per plan Step 2 enumeration:
  * `crates/xlog-cuda/src/memory.rs` (histogram storage)
  * `crates/xlog-cuda/src/provider/wcoj.rs` (launch-time slicing)
  * `crates/xlog-runtime/src/executor/recursive.rs` (Merge-phase refresh hook)
  * `crates/xlog-integration/benches/wcoj_triangle_bench.rs` (cell harness)
  * `docs/evidence/2026-05-12-w33-merge-resident-histogram-spike/README.md` (evidence)
* **S12.3** Measure cells in the order specified by plan Step 2.3:
  1. `uniform-u32-10K` FIRST (D7b tightest, +11 µs budget). Assert row-equality via `download_triples` before timing.
  2. `superhub-50K` SECOND (D7a ≥ 2.0× gate). Assert row-equality via `download_triples` before timing.
  Report Criterion median + delta + ratio vs. the existing baseline launch path for both cells.
* **S12.4** Honor the plan's stop conditions verbatim:
  * If `uniform-u32-10K` median exceeds the ±5% budget, STOP. Report the failure and do NOT measure `superhub-50K`. Do NOT amend the implementation to "rescue" the cell — that decision is for a future plan-iteration goal.
  * If `uniform-u32-10K` passes but `superhub-50K` does not, report both results and recommend a plan-iteration goal — do NOT broaden the spike implementation in this goal.
* **S12.5** Evidence README at `docs/evidence/2026-05-12-w33-merge-resident-histogram-spike/README.md` MUST contain:
  * Branch SHA + base SHA explicitly stated
  * Plan locks D1–D8 referenced as the design contract
  * Each cell: row-equality assertion result, raw Criterion median, ±5%/×2.0 budget check, pass/fail verdict
  * Stop-condition outcome stated explicitly (`stopped at uniform-u32-10K` / `both cells measured` / `passed uniform-u32-10K, failed superhub-50K`)
  * Forensic phase numbers from `f1142b3e` cited as the baseline rejection evidence to contrast against
* **S12.6** Spike branch stays UNMERGED. No FF-merge to plan branch, no FF-merge to main, no push, no tag. Per `feedback_perf_bench_spike_first.md`, the unmerged branch IS the durable evidence.
* **S12.7** Single bundled commit on the spike branch after measurements are recorded, subject `spike(w33): merge-resident histogram cell measurements (uniform-u32-10K + superhub-50K)`. If the implementation requires multiple logical commits (e.g., scaffolding before bench harness), each commit must be on the spike branch and the final commit must be the evidence README.

### Questions (per cell)

* **Q12.1** What is the spike branch HEAD SHA at completion?
* **Q12.2** `uniform-u32-10K` (D7b cell): row-equality PASS? Criterion median delta vs. baseline? Within ±5%? Quote the raw Criterion estimates.
* **Q12.3** `superhub-50K` (D7a cell): row-equality PASS? Criterion median ratio vs. baseline? Within ≥ 2.0×? Quote the raw Criterion estimates. **Only answer this if Q12.2 passed; otherwise state "not measured per S12.4 stop condition".**
* **Q12.4** Which stop-condition branch was taken? (`stopped at uniform-u32-10K` / `both cells measured` / `passed uniform-u32-10K, failed superhub-50K`)
* **Q12.5** Did the spike implementation introduce any production-style breadth beyond the minimum needed for these two cells? List concrete files touched outside the S12.2 allowed list.
* **Q12.6** Is the branch unmerged at completion? `git branch --merged main | grep bench-spike/w33-merge-resident-histogram` should be empty.

### Metrics

* **M12.1** Spike branch `bench-spike/w33-merge-resident-histogram` exists, HEAD reachable from neither `main` nor `feat/w33-paper-aligned-plan-it1`.
* **M12.2** Evidence README exists at `docs/evidence/2026-05-12-w33-merge-resident-histogram-spike/README.md` on the spike branch.
* **M12.3** Criterion bench compiles: `cargo bench -p xlog-integration --bench wcoj_triangle_bench --no-run` EXIT 0.
* **M12.4** Row-equality assertion for `uniform-u32-10K`: PASS. Stated verbatim in README.
* **M12.5** `uniform-u32-10K` Criterion median delta: stated verbatim with absolute µs value AND ±% vs. baseline. Pass/fail vs. ±5% budget stated.
* **M12.6** `superhub-50K` measurement either taken with PASS row-equality + median ratio stated, OR explicitly skipped per S12.4 stop condition with reason cited.
* **M12.7** `cargo fmt --check --all` EXIT 0 on spike branch.
* **M12.8** No `crates/` file outside the S12.2 enumerated list is modified between the spike branch HEAD and `feat/w33-paper-aligned-plan-it1` (modulo the bench-harness expansion which is allowed).
* **M12.9** `git tag --points-at HEAD` empty on spike branch; `git ls-remote --heads origin "bench-spike/w33*"` empty.
* **M12.10** Spike branch is unmerged: `git branch --merged main | grep bench-spike/w33-merge-resident-histogram` empty.

### Supervisor validation per locked protocol (after spike COMPLETE)

* Read the evidence README end-to-end.
* `git rev-parse bench-spike/w33-merge-resident-histogram` ≠ `git rev-parse main` ≠ `git rev-parse feat/w33-paper-aligned-plan-it1`.
* `cargo bench -p xlog-integration --bench wcoj_triangle_bench --no-run` EXIT 0 from supervisor session on the spike branch.
* Verify `uniform-u32-10K` measurement is FIRST in the README and superhub-50K is SECOND or marked "not measured".
* Verify D7b ±5% pass/fail verdict is stated unambiguously (a number + comparison + verdict).
* Verify forensic `f1142b3e` phase numbers (304 µs / 130 µs / +19 µs / 16×) appear in the README as the baseline contrast.
* Verify branch is unmerged.
* Verify `git tag --points-at HEAD` empty and no origin push of spike branch.

If all green and the spike PASSES both cells: supervisor confirms G12 complete and writes goal-013 (production implementation plan iteration 2, NOT the implementation itself).
If `uniform-u32-10K` fails: supervisor confirms G12 produced rejection evidence, requests user decision (rewrite plan with different design vs. defer W3.3 vs. amend D7 with explicit approval).
If `uniform-u32-10K` passes but `superhub-50K` fails: supervisor confirms G12 produced mixed evidence, requests user decision (proceed with D7b-only implementation plan vs. amend design).

### Forbidden behaviors

* No `git push`, no `git tag`, no `--force`, no `--no-verify`, no `--dangerously-bypass`.
* No FF-merge or merge of `bench-spike/w33-merge-resident-histogram` into `main` or `feat/w33-paper-aligned-plan-it1`.
* No `docs/v065-closure-board.md` edit (W3.3 stays OPEN per D8).
* No `v0.6.6` references.
* No production-style breadth beyond minimum-viable spike (per S12.2 file list; bench harness is the only file that may grow).
* No attempt to "fix" a failing `uniform-u32-10K` cell by amending the implementation in the same goal — STOP per S12.4 and report.
* No D7 amendment (per plan D6 LOCK).
* No new R6 anti-pattern in the spike implementation: the four anti-patterns listed in the plan's Forbidden Directions section (per-call histogram launch / heavy-light split / per-call `classify_heavy_rows` / front-end `mask_histogram`+`classify`+`partition_scan`) are forbidden in the spike code, not just in the production design.

### Why this is scoped tight

The R6 failure happened because the implementation was authorized without a bounding measurement of the binding cell. G12 enforces the bounding measurement as the gate before any production-code authorization. The spike's value is in the answer it produces — not in the code it ships — which is why the branch stays unmerged.

Proceed: cut spike branch, implement minimum-viable surface, measure cells in order, honor stop conditions, write evidence README, single bundled commit. No board edit, no merge, no push.
