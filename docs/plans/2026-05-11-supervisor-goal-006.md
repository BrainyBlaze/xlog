# Supervisor Goal 006 — W5.2 Stage 3 Wrap-Up (Plan Steps 6 + 7)

**Supervisor:** Claude Code.
**Implementer:** Codex CLI.
**Predecessor:** Supervisor goal 005 → G5 Steps 2–5. Branch `feat/w52-skewed-multiway-bench` at `05fe9a0c`. 5 commits: plan + skeleton + 4-cycle + 5-clique + pivot-heavy K5. All sub-steps supervisor-approved under the locked protocol.
**Date:** 2026-05-11.

---

## G5 supervisor approval record

G5 COMPLETE. All 4 sub-steps (2 + 3 + 4 + 5) approved with independent locked-protocol audits:

| Step | Commit | Workload | Direction | Ratio range | TSV ↔ README match |
|------|--------|----------|-----------|-------------|---------------------|
| 2 | `4e4f6d15` | (skeleton) | — | — | — |
| 3 | `1090a7af` | 4-cycle hub_filtered | GPU 3/3 | 2.12×–7.02× | byte-for-byte ✓ |
| 4 | `12ed487c` | 5-clique diagonal | HASH 3/3 | 0.49×–0.59× | byte-for-byte ✓ |
| 5 | `05fe9a0c` | Pivot-heavy K5 | HASH 3/3 | 0.55×–0.87× (rising) | byte-for-byte ✓ |

36 total measurements (3 workloads × 4 cells × 3 runs); zero direction flips; all parity-before-timing assertions present; D3/D6/F-W52-7 locks all verified in source; no `crates/xlog-cuda/` changes; no push, no tag, no forbidden tokens.

---

## G6 — W5.2 plan Steps 6 + 7 (aggregated evidence + closure proposal + final gates)

### Goal

Produce the aggregated W5.2 evidence section (cross-workload summary), the closure proposal, and run all final gates. Final REVIEW REQUEST after the closure-proposal commit lands.

### Strategies (GQM+Strategies)

* **S6.1**: Aggregate the 36 measurements across 3 workloads into a single per-workload summary table within the existing evidence README — NOT a new file. Each workload row: min/median/max ratio, direction-stability (e.g., "GPU 3/3"), and an observed-regime note (e.g., "trends toward parity at N=40").
* **S6.2**: Closure proposal at `docs/plans/2026-05-12-w52-closure-proposal.md` (NOT under `docs/evidence/`) per W4.3/W5.1 precedent. Anchor commit count to closure-proposal commit (per F-W43-18). Quote D7 + Acceptance Grid VERBATIM from plan `c2e7aaf4` (per F-W43-13/16).
* **S6.3**: Final gates run sequentially BEFORE the closure-proposal commit: targeted bench compile, fmt, warnings-as-errors build, CUDA cert suite 1/1, canonical workspace test under F-W43-12/15 + g04_transfer_efficiency exception. Numbers captured in the closure proposal with exit codes.
* **S6.4**: Three closure-board response options enumerated in proposal (Accept / Reject / Defer); Response 1 recommended with reasoning if all gates green.

### Questions

* **Q6.1**: For the cross-workload summary, what's the load-bearing per-workload claim? (Per plan D8, the answer is one of: "stable GPU 3/3 in tested range" OR "stable HASH 3/3 in tested range" OR "GPU/HASH mixed → unstable cell flagged".)
* **Q6.2**: Does W5.2 satisfy the closure-board cert criterion *"Bench harness committed; evidence file with crossover thresholds vs. binary-join"* given the 3 workloads' per-shape findings?
* **Q6.3**: How should the closure proposal frame the "no crossover observed in tested range" outcomes for 5-clique + pivot-heavy K5 vs the GPU-dominant 4-cycle? Plan D8 allows this explicitly; proposal must cite D8 in the response.
* **Q6.4**: What does W5.2 closure unblock? (Cite W2.5 closure-board row directly.)
* **Q6.5**: F-W43-12/15 + g04_transfer_efficiency flake exception accounting: was any exception consumed during Step 7's gate runs? Report exact victim file/test if so.

### Metrics

* **M6.1**: Evidence README extended with a cross-workload summary section (after existing per-step sections). New commit subject: `docs(w52): aggregate cross-workload evidence`.
* **M6.2**: Closure proposal committed at `docs/plans/2026-05-12-w52-closure-proposal.md`. Anchored commit count cited explicitly (e.g., `git rev-list --count main..<closure-proposal-commit-hash> = N`).
* **M6.3**: D7 + Acceptance Grid quoted byte-for-byte from plan commit `c2e7aaf4` in the closure proposal. Verified by `diff -u <(git show c2e7aaf4:...) <(awk ... <closure-proposal.md>)`.
* **M6.4**: All gates green: `cargo fmt --check --all` exit 0; `RUSTFLAGS="-D warnings" cargo build --release --workspace --exclude pyxlog` exit 0; `cargo bench -p xlog-integration --bench w52_skewed_multiway_bench --no-run` exit 0; `cargo test -p xlog-cuda-tests --test certification_suite --release` 1/1; `cargo test --workspace --release --exclude pyxlog --exclude xlog-cuda-tests` exit 0 (or exception consumed per F-W43-12/15 + g04_transfer_efficiency narrowed enumeration).
* **M6.5**: Three closure-board response options enumerated with Response 1 (Accept as DONE) recommended in proposal text.
* **M6.6**: NO board edit, NO DONE marking, NO FF-merge, NO push, NO tag.
* **M6.7**: NO `crates/xlog-cuda/` changes in this goal.
* **M6.8**: NO `v0.6.6` references; no forbidden tokens.

### Process

1. **Cross-workload summary commit** (Step 6 first half): append summary table + per-workload narrative to existing evidence README. Single commit, README-only.
2. **Closure proposal commit** (Step 6 second half): new file `docs/plans/2026-05-12-w52-closure-proposal.md`. Single commit, closure-proposal file only.
3. **Final gates** (Step 7): run all M6.4 commands in sequence; capture exit codes. If gate fails (especially the cert suite or canonical workspace), retry once after brief settle to disambiguate flake vs regression.
4. **Post**: `GOAL G6 COMPLETE — REVIEW REQUEST` with all M6.x measured values + closure-proposal commit hash.

### Supervisor validation per locked protocol (when codex posts G6 REVIEW REQUEST)

* Re-run fmt, warnings build, bench compile, cert suite, canonical workspace test from supervisor session. Capture exit codes + result lines.
* Read closure proposal end-to-end.
* `diff -u` D7 + Acceptance Grid between closure proposal and plan `c2e7aaf4`. Expect zero output.
* Verify anchored commit count matches `git rev-list --count main..<closure-proposal-commit> = N`.
* Verify three response options enumerated with Response 1 recommended.
* Verify NO board edit, NO push, NO tag, NO FF-merge (`git diff main..feat/w52-skewed-multiway-bench --name-only` shows no `docs/v065-closure-board.md` and no `memory/` updates).
* Verify `git ls-remote --heads origin "feat/w52*"` empty.
* Verify `git tag --points-at HEAD` empty.
* Read cross-workload summary in README; verify per-workload claims match the per-step TSV evidence.

If all green: supervisor approves G6 + writes goal-007 for the four follow-up actions (board OPEN→DONE for W5.2; memory file; MEMORY.md update; FF-merge to main).

If findings: supervisor surfaces them; codex remediates BEFORE approval.

### Forbidden behaviors (locked, carry over)

* No `git push`, no `git tag`, no `--force`, no `--no-verify`, no `--dangerously-bypass`.
* No production kernel / provider / executor changes (W5.2 is bench-only per plan D1).
* No board edit, no DONE marking — those land in supervisor-goal-007 after closure-proposal supervisor approval.
* No FF-merge until goal-007.
* No claim of P1/P3/P4 alignment. P2/P5 only.
* No `v0.6.6` references.
* No retroactive changes to Step 2/3/4/5 commits — closure proposal must anchor to those commits as-is.

Proceed with Step 6 (cross-workload summary first, then closure proposal).
