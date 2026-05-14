# Supervisor Goal 011 — W3.3 Plan Iteration 1 Draft (Paper-Aligned Redesign, Bench-Spike-First)

**Supervisor:** Claude Code.
**Implementer:** Codex CLI.
**Predecessor:** G10 W2.5 closure complete on local `main` at `3f8e5d4c` (board edit) / `1d1e1fc5` (closure proposal). All 5 supervisor gates green (fmt, warnings build, targeted W2.5 sweep 9/9 cells, CUDA cert suite 1/1, canonical workspace EXIT 0 with **no F-W43-12/15 exception consumed**). Closure-board tally: DONE 13, IN-PROGRESS 1 (W1.1), BLOCKED 0, OPEN 9 (W3.3, W3.4, W3.5, W3.6, W5.3, W5.4, W6.1, W6.2, W7.1), Total 23.
**Date:** 2026-05-11.

---

## Context

W3.3 (`Histogram-guided block scheduling / heavy-row offload`, board row at `docs/v065-closure-board.md:87`) is OPEN with **one prior failed attempt** that this redesign MUST NOT repeat.

### Prior failure on record

* **Attempt:** iteration-8 R6 on branch `feat/w33-histogram-block-scheduling` (HEAD post-`f1142b3e`, preserved unmerged).
* **Outcome:** 12/12 correctness cells green; **0/6 D7 perf gates green**. D7a: u32 1.337×, u64 1.155× (target ≥ 2.0×). D7b: all 4 cells FAIL with +110% to +153% overhead.
* **Phase-attribution forensic (commit `f1142b3e`):** uniform-u32-50K +478 µs delta = 304 µs structural (mask_histogram 128 + classify 96 + partition_scan 80) + 130 µs implementation overhead + ~44 µs noise. Structural cost is **16× the +19 µs D7b budget**.
* **Verdict in memory `project_w33_r6_failure.md`:** "**measured-and-rejected**, not merely 'bench failed.'"

### Paper grounding (memory `reference_srdatalog_paper.md`)

* **Paper:** "Scaling Worst-Case Optimal Datalog to GPUs" (Sun, Qi, Gilray, Kumar, Micinski), arXiv:[2604.20073](https://arxiv.org/abs/2604.20073), §5 lines ~382–413.
* **P3 verbatim from memory:** *"Histograms maintained alongside data; computed incrementally during Merge; consumed at kernel launch-time to assign balanced thread-block work-unit slices over the outermost-join's search space. NOT per-call histograms (that's the wrong-paper interpretation that sank W3.3's failed implementation)."*

### Audit brief (paper-alignment authority)

* **Branch:** `feat/w3-paper-alignment-audit` HEAD `134884fc` (preserved unmerged).
* **Audit-derived redesign brief:** histogram lives on `CudaBuffer` or sibling metadata; update during Merge phase; consume at kernel launch to balance thread-block work-unit slices; **no per-call histogram launch, no heavy/light kernel split, no per-call classify_heavy_rows kernel**.

---

## G11 — Plan iteration 1 ONLY

### Goal

Produce a single planning artifact at `docs/plans/2026-05-12-w33-paper-aligned-plan.md` (the W3.3 plan iteration 1) on a fresh branch `feat/w33-paper-aligned-plan-it1` cut from local `main` at `3f8e5d4c`. **No production-code change in this goal.** A subsequent goal will authorize the bench spike; a later goal still will authorize implementation.

This goal exists because W3.3 redesign has unusually high paper-alignment risk, and supervisor precedent (W4.2, W4.3, W5.2) shows bench-spike-first discipline must be authored into the plan BEFORE the spike branch is cut.

### Strategies (GQM+Strategies)

* **S11.1**: Plan iteration 1 file at `docs/plans/2026-05-12-w33-paper-aligned-plan.md`. Subject `docs(plan): W3.3 iteration 1 — paper-aligned histogram-on-CudaBuffer redesign`.
* **S11.2**: Plan MUST cite P3 verbatim from `reference_srdatalog_paper.md` between `<!-- BEGIN VERBATIM P3 -->` / `<!-- END VERBATIM P3 -->` machine-checkable wrappers (per W5.2/W2.5 precedent for verbatim alignment).
* **S11.3**: Plan MUST cite the W3.3 board acceptance line verbatim (≥ 2.0× speedup on superhub fixture, deterministic `download_triples` row-equality, no uniform-fixture regression within ±5%) between `<!-- BEGIN VERBATIM W33-ACCEPTANCE -->` / `<!-- END VERBATIM W33-ACCEPTANCE -->` wrappers anchored to `docs/v065-closure-board.md` at HEAD `3f8e5d4c`.
* **S11.4**: Plan MUST enumerate explicit FORBIDDEN directions derived from the R6 failure, with the inline phrase "**measured-rejected per `f1142b3e`**" attached to each: no per-call histogram launch; no heavy/light kernel split; no per-call `classify_heavy_rows` kernel; no front-end `mask_histogram` / `classify` / `partition_scan` pass.
* **S11.5**: Plan MUST enumerate LOCKED design directions D1–D7+ per W2.5/W5.2 precedent. At minimum, locks for: (D1) bench-spike-first sequencing; (D2) histogram storage location = `CudaBuffer` (or explicitly named sibling metadata struct), justified against P5 (flat columnar storage); (D3) histogram update site = Merge phase, justified against P3; (D4) histogram consumption site = kernel launch-time work-unit slice assignment, justified against P3; (D5) D7a + D7b paper gates remain unchanged (≥ 2.0× speedup; ±5% uniform); (D6) no D7 amendment without explicit user approval; (D7) no changes to existing `wcoj_triangle_*_recorded` kernel signatures except as required by the launch-time slice-assignment surface.
* **S11.6**: Plan MUST define the bench-spike-first protocol: spike branch name `bench-spike/w33-merge-resident-histogram`; minimum-viable bench surface = run uniform-u32-10K (the tightest D7b cell at +11 µs budget) + superhub-50K (the D7a cell) with the `CudaBuffer`-resident histogram in place; spike must bound D7b *before* any production-code change; spike branch stays unmerged regardless of outcome (per memory `feedback_perf_bench_spike_first.md`).
* **S11.7**: Plan MUST include a Source-Of-Truth References section listing: `reference_srdatalog_paper.md`, `project_w33_r6_failure.md`, audit brief at `feat/w3-paper-alignment-audit` HEAD `134884fc`, failed branch at `feat/w33-histogram-block-scheduling` HEAD post-`f1142b3e`, forensic evidence at `docs/evidence/2026-05-07-w33-phase-attribution/README.md` (on failed branch), board row at `docs/v065-closure-board.md:87`.
* **S11.8**: Plan MUST include a Paper-Alignment Note section explicitly contrasting per-call (R6, rejected) vs. Merge-resident (P3-aligned, proposed) histogram lifetimes, with the four phase-attribution numbers cited (304 µs structural / 130 µs impl / +19 µs D7b budget / 16× ratio).
* **S11.9**: Plan MUST include a Risk Register section enumerating named risks F-W33-1 … F-W33-N, including at least: (F-W33-1) Merge-phase update cost might still exceed D7b budget; (F-W33-2) `CudaBuffer` API surface change blast radius; (F-W33-3) launch-time slice-assignment kernel launch overhead; (F-W33-4) interaction with W3.1's flat-columnar contract (P5); (F-W33-5) interaction with W3.2's deterministic offsets (P5); (F-W33-6) layout fast-path test (`test_wcoj_layout_u32.rs::wcoj_layout_u32_already_sorted_deduped_round_trips`) flakiness under F-W43-12/15 exception accounting.
* **S11.10**: Plan MUST end with an Acceptance Grid table mapping each W3.3 board sub-clause to (a) planned spike-stage evidence, (b) planned implementation-stage evidence, (c) planned gate command — the same 3-column shape used in the W2.5 plan at `56685fa3`.

### Questions (per step)

* **Q11.1**: Does the plan's Paper-Alignment Note quote P3 verbatim from `reference_srdatalog_paper.md`? (Supervisor will `diff -u` against the memory file.)
* **Q11.2**: Does the plan's Forbidden Directions list use the literal phrase "**measured-rejected per `f1142b3e`**" for each R6 anti-pattern?
* **Q11.3**: Does the plan's Acceptance Grid quote the W3.3 board sub-clauses verbatim from `docs/v065-closure-board.md` at HEAD `3f8e5d4c`?
* **Q11.4**: Does the bench-spike-first protocol specify uniform-u32-10K (D7b worst-case cell) as a required spike measurement, and superhub-50K (D7a) as a required spike measurement, before any production change?
* **Q11.5**: Does the LOCKED directions section have ≥ 7 named locks (D1–D7+), each with paper-claim or evidence justification?
* **Q11.6**: Is the plan free of `v0.6.6` references and free of any `--no-verify` / `--force` / `git push` / `git tag` commands?

### Metrics

* **M11.1**: Plan file lands at `docs/plans/2026-05-12-w33-paper-aligned-plan.md`. Subject: `docs(plan): W3.3 iteration 1 — paper-aligned histogram-on-CudaBuffer redesign`.
* **M11.2**: Plan branch: `feat/w33-paper-aligned-plan-it1` cut from `main` at `3f8e5d4c`.
* **M11.3**: Anchored commit count `git rev-list --count main..<plan-commit> = 1` (single commit, no code change).
* **M11.4**: Plan contains: Status / Paper-Alignment Note / Verbatim P3 Excerpt / Verbatim W3.3 Board Acceptance / Locked Directions D1–D7+ / Forbidden Directions / Bench-Spike-First Protocol / Acceptance Grid / Source-Of-Truth References / Risk Register / Plan-Approval Gate.
* **M11.5**: Verbatim diff supervisor-side: `diff -u <(extract from plan between BEGIN/END VERBATIM P3 markers) <(awk extract from reference_srdatalog_paper.md) → DIFF_EXIT=0`. Same for VERBATIM W33-ACCEPTANCE against `docs/v065-closure-board.md` at `3f8e5d4c`.
* **M11.6**: Final gates run BEFORE the plan commit and captured inline in the plan's Verification section: `cargo fmt --check --all` exit 0; canonical workspace test invocation **not required** for a plan-only commit, but the plan must state that no production-code change is included.
* **M11.7**: `git diff main..feat/w33-paper-aligned-plan-it1 -- crates/` is empty (no crate change).
* **M11.8**: `git diff main..feat/w33-paper-aligned-plan-it1 -- 'docs/v065-closure-board.md'` is empty (no board change; board change is a later goal).

### Supervisor validation per locked protocol

* Read the plan end-to-end.
* `diff -u` VERBATIM P3 between plan and `reference_srdatalog_paper.md`. EXIT=0.
* `diff -u` VERBATIM W33-ACCEPTANCE between plan and `docs/v065-closure-board.md` at `3f8e5d4c`. EXIT=0.
* `grep -c "measured-rejected per .f1142b3e."` on the plan ≥ 4 (one per R6 anti-pattern).
* `git diff --stat main..feat/w33-paper-aligned-plan-it1` shows exactly one file modified: the new plan file.
* `cargo fmt --check --all` from supervisor session: EXIT 0.
* Verify no `v0.6.6` / `--no-verify` / `--force` / `git push` / `git tag` references in the plan.
* Verify the Bench-Spike-First Protocol section names `bench-spike/w33-merge-resident-histogram` and requires uniform-u32-10K + superhub-50K spike measurements before any production-code change.
* Verify the Risk Register names ≥ 6 F-W33-* risks.

If all green: supervisor confirms plan iteration 1 is closure-ready for user review, and writes goal-012 covering the bench-spike sequence (NOT implementation — the spike is its own gate). If the user explicitly approves the plan AND the spike result, only then does a future goal authorize production implementation.

### Forbidden behaviors

* No production-code change. No `crates/` edits. No CUDA-kernel edits. No new kernel files.
* No `git push`, no `git tag`, no `--force`, no `--no-verify`, no `--dangerously-bypass`.
* No `docs/v065-closure-board.md` edit (no W3.3 row state change in this goal).
* No `v0.6.6` references.
* No re-tuning of D7a / D7b gates without explicit user approval (per memory `project_w33_r6_failure.md`).
* No proposal of any of the four R6 anti-patterns (per-call histogram launch, heavy/light kernel split, per-call `classify_heavy_rows`, front-end `mask_histogram`/`classify`/`partition_scan`). Each must appear in the Forbidden Directions list, never in the proposed design.
* No claim that the bench spike is optional, can be deferred, or can be replaced by analysis-only justification.

### Why this is scoped tight

W3.3 has already burned one full iteration (8 sub-iterations) and a forensic investigation because the design landed without paper alignment and without a bench-spike gate. The W5.2 closure precedent demonstrated that bench-spike-first prevents this exact failure mode (W5.2's spike at `eacd3815` stayed unmerged; the production decision used its evidence). G11 enforces the same gate sequencing for W3.3 BEFORE any line of GPU code is written.

Proceed: single plan commit on `feat/w33-paper-aligned-plan-it1`. No spike, no implementation, no board edit. The plan IS the deliverable.
