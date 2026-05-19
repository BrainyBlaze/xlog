# Supervisor Goal 010 — W2.5 Stage 3 + Closure (Steps 6-7 + Four Follow-Ups)

**Supervisor:** Claude Code.
**Implementer:** Codex CLI.
**Predecessor:** G9 Steps 2-5 complete on `feat/w25-cost-model-default-flip` at `36e8e46d`. 5 W2.5 commits total: plan + default-flip + safety-floor + slice-4 + W2.6 refresh. Supervisor-approved with all regression suites pass-verified independently (W2.1 11/0, W2.2 6/0, W2.4 3/0, W2.6 7/0, slice-4 1/0; W2.3 codex's gate).
**Date:** 2026-05-11.

---

## G9 supervisor approval record

G9 APPROVED. Independent locked-protocol audit covered each of Steps 2 / 3 / 4+5:

* **Step 2** (`0f5b30d2`): default-flip surgical diff (resolver branch + doc updates + 1 test renamed + 1 env-opt-out test added); D2 LOCKED satisfied (no new field/env-var); 2/0 unit test pass; fmt EXIT=0.
* **Step 3** (`37133ca0`): safety-floor integration cert via TDD RED-via-production-mutation; D5 LOCKED satisfied (missing-stats fallback at `wcoj_cost_model.rs:373` unchanged); 7/0 file-level pass; fmt EXIT=0.
* **Step 4** (`d7e69101`): slice-4 bare-default branch added; D6 LOCKED satisfied (counter == 1 holds); supervisor 1/0 pass.
* **Step 5** (`36e8e46d`): regression sweep surfaced one W2.6 stale default-skew assumption; fixed via legacy vs bare-default scenario split with row-set parity preserved for both. Supervisor re-runs: W2.1 11/0, W2.2 6/0, W2.4 3/0, W2.6 7/0. Zero runtime/kernel/provider changes.

W2.5 default-flip is now production-correct: bare `RuntimeConfig::default()` returns `CostModelKind::Cardinality`; env `XLOG_WCOJ_COST_MODEL=skew` restores legacy; missing-stats safety floor delegates correctly; slice-4 stable-triangle counter still 1; W2.1/W2.2/W2.3/W2.4/W2.6 regression sweep clean.

Code is complete. Remaining work: documentation + closure proposal + final gates + four follow-up actions.

---

## G10 — Steps 6-7 + four follow-ups

### Goal

Land W2.5 closure end-to-end: (Step 6) closure proposal at `docs/plans/2026-05-12-w25-closure-proposal.md`, (Step 7) final gates run + captured in proposal, plus the four follow-up actions per W4.3/W5.1/W5.2 precedent (board OPEN→DONE, memory file, MEMORY.md update, FF-merge to main). Single goal because W2.5 is smaller scope than W5.2 (5 implementation commits vs W5.2's 6).

### Strategies (GQM+Strategies)

* **S10.1**: Closure proposal at `docs/plans/2026-05-12-w25-closure-proposal.md`. Anchor commit count to the closure-proposal commit (per F-W43-18). Quote D2/D5/D6/D7 + Acceptance Grid VERBATIM from plan `56685fa3` (per F-W43-13/16). Use HTML comment wrappers (per W5.2 precedent) for machine-checkable verbatim diff.
* **S10.2**: Final gates run BEFORE the closure-proposal commit so exit codes can be captured inline. Standard suite: fmt + warnings-as-errors build + CUDA cert suite + canonical workspace test + targeted W2.5 cert sweep.
* **S10.3**: Three closure-board response options enumerated (Accept / Reject / Defer); Response 1 recommended given all sub-clauses satisfied.
* **S10.4**: Four follow-up actions in defined order (board edit first, FF-merge last, memory updates in parallel).
* **S10.5**: Board edit handles ONE state change (W2.5 OPEN → DONE) — unlike W5.2 (which cascade-unblocked W2.5), W2.5 unblocks nothing new. Tally update: DONE 12 → 13, OPEN 10 → 9, Total stays 23.

### Questions (per step)

* **Q10.1 (Step 6)**: For each of W2.5's four acceptance sub-clauses, what's the concrete delivered evidence (commit hash + file path + measured outcome)?
* **Q10.2 (Step 6)**: How does the closure proposal cite W5.2's per-workload landscape as the bench-spike-first input? Quote specific W5.2 README lines.
* **Q10.3 (Step 7)**: All final-gate exit codes captured. Any F-W43-12/15 + g04_transfer_efficiency exception consumed during gate runs? Report exact victim file/test if so.
* **Q10.4 (Follow-Ups)**: After the board edit + FF-merge, what's the new main HEAD hash? Verify `git tag --points-at HEAD` empty and `git ls-remote --heads origin "feat/w25*"` empty.

### Metrics

* **M10.1**: Step 6 commit lands closure proposal at `docs/plans/2026-05-12-w25-closure-proposal.md`. Subject: `docs(w25): add closure proposal`.
* **M10.2**: Closure proposal includes: Status / Commit Anchor / Verbatim Plan Excerpts (D2/D5/D6/D7 + Acceptance Grid) / Evidence Summary (per sub-clause) / Verification (gate exit codes) / Scope & Holds / Closure-Board Response Options.
* **M10.3**: Verbatim diff supervisor-side: `diff -u <(git show 56685fa3:...) <(awk ... <closure-proposal>) → DIFF_EXIT=0` for D2 + D5 + D6 + D7 + Acceptance Grid.
* **M10.4**: Final gates all green (fmt, warnings build, cert suite 1/1, canonical workspace under F-W43 exception, targeted W2.5 sweep).
* **M10.5**: Anchored commit count `git rev-list --count main..<closure-proposal-commit> = N` cited in proposal.
* **M10.6**: Three response options enumerated; Response 1 recommended.
* **M10.7**: Board edit commit lands on `feat/w25-cost-model-default-flip` BEFORE FF-merge; subject `docs(w25): mark W2.5 DONE on closure board`. Tally update: DONE 12→13, OPEN 10→9, BLOCKED 0 (unchanged), Total 23 (unchanged).
* **M10.8**: Memory file at `~/.claude/projects/-home-dev-projects-xlog/memory/project_w25_closed.md` per W4.1/W4.2/W4.3/W5.1/W5.2 precedent.
* **M10.9**: MEMORY.md updated with W2.5 closure pointer after W5.2 line.
* **M10.10**: FF-merge succeeds; main HEAD advances to board-edit commit; NO push, NO tag.

### Supervisor validation per locked protocol (after FOLLOW-UPS COMPLETE)

* Re-run fmt + warnings build + cert suite + canonical workspace + targeted W2.5 sweep from supervisor session.
* Read closure proposal end-to-end.
* `diff -u` D2 + D5 + D6 + D7 + Acceptance Grid between proposal and plan `56685fa3`. Each diff EXIT=0.
* Verify anchored commit count matches `git rev-list --count main..<closure-proposal-commit>`.
* Verify board edit content: W2.5 row DONE with full closure record; tally arithmetic 13+1+0+9=23.
* Read memory file; verify W4.1/W4.2/W4.3/W5.1/W5.2 format alignment.
* Verify MEMORY.md has W2.5 pointer.
* Verify `git -C ~/projects/xlog log --oneline -3 main` shows W2.5 chain at HEAD.
* Verify `git tag --points-at HEAD` empty and `git ls-remote --heads origin "feat/w25*"` empty.

If all green: supervisor confirms W2.5 closure complete and writes goal-011 covering the next OPEN item (likely W3.3 histogram-guided block scheduling, the largest-scope remaining perf item; or smaller W5.3/W5.4 cert/harness items first for tempo).

### Forbidden behaviors

* No `git push`, no `git tag`, no `--force`, no `--no-verify`, no `--dangerously-bypass`.
* No new code paths beyond closure proposal text + board edit + memory file + MEMORY.md.
* No production kernel/provider/executor/runtime changes.
* No `v0.6.6` references.
* No retroactive amendment of G9 Steps 2-5 commits.

Proceed: Step 6 closure proposal first, then Step 7 final gates run with exit codes captured into the proposal (commit may be amended once if needed), then the four follow-up actions in order.
