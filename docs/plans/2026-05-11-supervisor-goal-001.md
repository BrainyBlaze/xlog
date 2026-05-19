# Supervisor Goal 001 — Tally Fix + W5.1 Plan Iteration 1

**Supervisor:** Claude Code (in `~/projects/xlog/.worktrees/w43-sort-merge-join`, just FF-merged W4.3 to `main` at HEAD `66f69dbe`).
**Implementer:** Codex CLI (in `~/projects/xlog` on `main`, this tmux session).
**Date:** 2026-05-11.
**Framing discipline:** Goal-Driven Software Development Process + GQM (Goal-Question-Metric) + GQM+Strategies. Every goal below is structured as {Goal (outcome), Questions (what to inspect), Metrics (auditable values)}; nothing ships without metric validation.

---

## Final outcome (load-bearing across all goals)

Full production-grade v0.6.5 with all 21 closure board items DONE per `docs/v065-closure-board.md` process rules. SRDatalog paper (arXiv:2604.20073) P1-P5 claims are load-bearing for any W3.x/W4.x kernel/runtime change.

## Non-negotiable process rules (from `docs/v065-closure-board.md` §"Process Rules" + project memory)

1. **No DONE marking on any board item until supervisor approves in thread.** Codex never self-marks DONE.
2. **No `git push`. No `git tag`.** v0.6.5 tag is W7.1, fires only when board reaches 0 OPEN AND supervisor explicitly authorizes.
3. **No FF-merge until supervisor authorizes per item.** Each closure follows the W4.x precedent: plan-iteration commits → impl commits → bench commits → closure proposal → supervisor approves DONE → board edit → memory entry → MEMORY.md update → FF-merge.
4. **No `v0.6.6` reference in NEW files / comments / plans / commits.** Existing references in shipped slices stay; new ones forbidden.
5. **Bench-spike-first for any perf-claim work** (`feedback_perf_bench_spike_first.md`): minimum-viable bench spike on a `bench-spike/wXY-*` branch measures the gate target BEFORE the full plan; failed spike branches stay unmerged as evidence.
6. **Plan-iteration discipline.** Every amendment to a plan lands as a new iteration commit with F-WXY-N findings, before/after table, and process observation. No silent edits to canonical D-tables / Acceptance Grids / Steps.
7. **Paper alignment.** SRDatalog paper (arXiv:2604.20073) P1-P5 constrain W3.x/W4.x. Cite paper §section + page when invoking a claim. Memory pointer: `reference_srdatalog_paper.md`.
8. **Worktree per closure item.** Mirror W4.1/W4.2/W4.3 precedent: create `feat/wXY-*` worktree under `.worktrees/`, work there, FF-merge to `main` only on supervisor approval.
9. **F-W43-12 + F-W43-15 workspace-test gate exception** applies to all subsequent gates: three enumerated `test_wcoj_layout_*` flake files (`fast_path`, `u32`, `u64`) are exempt; siblings (`sort_roundtrip`, `sort_u32`, `sort_u64`) and all other files must pass. Cert suite remains 1/1 (authoritative).
10. **F-W43-13 / F-W43-16 / F-W43-18 drift lessons.** When amending contracts, grep canonical D-table + Acceptance Grid + Step prose. When quoting prior findings in derived docs, quote verbatim from canonical source. Anchor commit counts in closure proposals to a named commit, not live HEAD.

## Communication protocol

* Supervisor writes goal-spec files at `docs/plans/2026-05-11-supervisor-goal-NNN.md` (this file is #001).
* Supervisor sends `/goal` via tmux pointing at the goal-spec file.
* Codex acknowledges, reads spec, executes goals in order, posts **"GOAL G_N COMPLETE — REVIEW REQUEST"** in TUI when each sub-goal is done with metric values measured.
* Supervisor monitors via `tmux capture-pane -t codex -p | tail -N` polling.
* Supervisor validates against metrics, replies with approval / findings / refinement requests via tmux send-keys.
* Iteration continues until all goals in the spec are approved by supervisor.

---

## G0 — Tally drift fix (cheap precondition before any new closure)

**Goal**: Repair the closure board's status tally row so its arithmetic is internally consistent.

**Background**: The tally row at `docs/v065-closure-board.md:51-54` currently states `DONE=10 + IN-PROGRESS=1 + BLOCKED=1 + OPEN=9 → Total=21`. But the OPEN-rows in the actual table are 11 (W3.3, W3.4, W3.5, W3.6, W5.1, W5.2, W5.3, W5.4, W6.1, W6.2, W7.1). Sum 10+1+1+11 = 23, exceeding the stated Total=21. This is drift from previous closures (likely the W4.3 closure-board commit mis-decremented OPEN).

### Questions

* **Q0.1**: Audit each row's `Status` column. List the actual state-of-each-item.
* **Q0.2**: Which slice of items contributes the OPEN-count off-by-2? (Hypothesis: W7.1 may not have been in the original total — but the table includes it. Or W2.5 BLOCKED was originally counted as OPEN.)
* **Q0.3**: Is the stated `Total=21` correct, or does the table actually contain a different number of rows? Count rows directly.

### Metrics

* **M0.1**: `DONE + IN-PROGRESS + BLOCKED + OPEN == Total` exactly.
* **M0.2**: The state-by-state list in each tally row matches the actual `Status` cells in the table.
* **M0.3**: One commit, subject `docs(closure-board): fix tally drift after W4.3 close`. No other changes in the same commit.

### Process

1. Audit table; produce a state-by-state list.
2. Decide whether Total stays 21 or needs revision.
3. Propose the corrected tally row to supervisor BEFORE committing.
4. Supervisor reviews, approves or asks for refinement.
5. On approval: commit the fix.

---

## G1 — W5.1 plan iteration 1 (cert trio scoping; NO code yet)

**Goal**: Produce iteration-1 plan for W5.1 (three new cert tests). Plan-only commit; no implementation yet.

**Background**: W5.1's board entry says *"Three new test files in `xlog-integration/tests/`; each asserts row-set parity vs. CPU oracle and dispatch counter > 0."* The three certs are: GPU Same Generation cert, skewed multiway GPU cert, deep-recursive WCOJ cert. W5.1 closure unblocks half of W2.5's dependency set.

### Questions

* **Q1.1**: What CPU oracle exists today for each cert?
  * Same Generation: is there an existing CPU/host-side same-gen evaluator (likely `xlog-runtime` CPU backend or a host-Datalog walk)? Where?
  * Skewed multiway: per W2.6 evidence, dispatch is GPU-only; CPU oracle = a host-side equivalent? Or compare to binary-join chain via existing `hash_join_v2`?
  * Deep recursive: likely a host-Datalog fixpoint walker. Existing? If not, what's the minimal-viable oracle?
* **Q1.2**: What dispatch counter applies to each? (`wcoj_triangle_dispatch_count`, `wcoj_4cycle_dispatch_count`, `wcoj_clique5_dispatch_count`, `wcoj_clique6_dispatch_count`, or `wcoj_recursive_dispatch_count`?)
* **Q1.3**: Counter-assertion form: per F-W43-13 exact-equality discipline, prefer `== N` over `>= 1`. Identify the deterministic N per cert.
* **Q1.4**: Existing-cert templates: which `crates/xlog-integration/tests/test_wcoj_*.rs` file is the closest template for each of the three? (Likely `test_wcoj_recursive_dispatch.rs`, `test_wcoj_4cycle_skew.rs`, `test_wcoj_recursive_dispatch.rs`.)
* **Q1.5**: Paper P1-P5 alignment:
  * P1 (semi-naïve over body-clause OCCURRENCES, not predicate names) — relevant to deep-recursive cert: must admit same-predicate self-recursion per W4.1.
  * P2/P3/P4/P5 — re-read paper memory `reference_srdatalog_paper.md` and cite which Px each cert validates.
* **Q1.6**: Fixture design: minimal-rows-but-non-trivial fixtures that hit the dispatch path AND have non-empty output (parity check is meaningless on empty output sets). Same Gen: depth N where N matches dispatch.
* **Q1.7**: F-W43-12 + F-W43-15 workspace-test gate compatibility: do any of the three new certs share device-context state with the three exempt files? If yes, document the isolation strategy.

### Metrics

* **M1.1**: Plan file committed at `docs/plans/2026-05-11-w51-cert-trio-plan.md` with header `iteration 1 canonical`.
* **M1.2**: Plan contains: Acceptance Line (from board), Paper-alignment note, Process Rule Compliance, Read-Only Surface (with file:line citations for the existing CPU oracle / dispatch counter / cert templates), Direction table D1-DN (locks), Step-by-Step Execution Plan (≥ 6 steps including spec → plan → execute → verify → closure proposal → supervisor approval gate), Acceptance Grid (3 certs × per-cert criteria + bench/cert-suite gates), Source-of-Truth References, Risk Register, Plan-Approval Gate.
* **M1.3**: Each of D1-DN has explicit lock wording (cf. W4.3 plan's D1/D2/D3/D4/D5/D6/D7/D8).
* **M1.4**: Each of the 3 certs has an Acceptance Grid row with: cert name, counter assertion (exact-equality form), parity oracle source, fixture description, expected row-set size, deterministic seed/order.
* **M1.5**: NO code commits. Plan-iteration commit only with subject `docs(plan): W5.1 iteration 1 — cert trio (Same Gen + Skewed Multiway + Deep Recursive)`.
* **M1.6**: Worktree pre-created at `.worktrees/w51-cert-trio` on branch `feat/w51-cert-trio` (off `main` HEAD `66f69dbe`), and the plan-iteration commit lives there (NOT on main).

### Process

1. Create worktree `.worktrees/w51-cert-trio` on branch `feat/w51-cert-trio` (off `main` HEAD `66f69dbe`).
2. Read-only recon: search for existing CPU oracle code paths, existing dispatch counters, existing cert templates, paper §§ relevant to each cert.
3. Re-read `reference_srdatalog_paper.md` (project memory pointer) for P1-P5; cite which Px each cert validates.
4. Draft plan iteration 1 with all sections per M1.2.
5. Commit plan-iteration commit per M1.5 on the worktree branch.
6. Post **"GOAL G1 COMPLETE — REVIEW REQUEST"** with measured M1.1-M1.6 values.

---

## Communication / monitoring expectations

* Codex: after G0 audit (Q0.1-Q0.3), STOP and request supervisor review BEFORE committing G0's fix.
* Codex: after G1 plan draft (M1.1-M1.6), commit plan-iteration commit and post review request.
* Supervisor: polls codex pane every ~30s during active execution.
* Refinement loop: if supervisor finds drift (process-rule violation, paper-misalignment, metric not auditable), codex amends per supervisor's findings and re-requests review. Cf. W4.3 iteration 5/6 amendment pattern.

When G0 + G1 are both supervisor-approved, supervisor writes goal-spec 002 covering W5.1 execution (Steps 2-N from G1's plan) + W5.2 plan iteration 1.

Proceed with G0 first.
