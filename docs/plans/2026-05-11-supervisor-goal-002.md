# Supervisor Goal 002 — W5.1 Implementation (G1 plan execution)

**Supervisor:** Claude Code.
**Implementer:** Codex CLI.
**Date:** 2026-05-11.
**Predecessor:** Supervisor goal 001 (G0 committed at `ffe27f4d`, G1 plan committed at `85d62e84` on `feat/w51-cert-trio`).
**Framing:** GQM (Goal-Question-Metric) + GQM+Strategies. Continues the supervisor pattern.

---

## Approval record

G1 SUPERVISOR-APPROVED. The W5.1 iteration-1 plan at `.worktrees/w51-cert-trio/docs/plans/2026-05-11-w51-cert-trio-plan.md` (commit `85d62e84`) is the canonical execution spec. D1-D8 LOCKED. Acceptance Grid + Verification Gates + Paper-Alignment + Risk Register validated.

Supervisor validation summary (auditable):

* **M1.1**: Header `# W5.1 Cert Trio Plan (iteration 1 canonical)` present.
* **M1.2**: Required sections present (Acceptance Line, Paper-Alignment Note, Process Rule Compliance, Read-Only Surface, Direction Table, Step-by-Step Execution Plan, Acceptance Grid, Source-of-Truth References, Risk Register, Plan-Approval Gate).
* **M1.3**: D1-D8 all LOCKED.
* **M1.4**: Acceptance Grid has 3 cert rows × {counter assertion, parity oracle, fixture, row-set size, deterministic order}.
* **M1.5**: Plan-only commit, 258 insertions, no Rust / test / evidence / board changes.
* **M1.6**: Worktree branch `feat/w51-cert-trio` off main HEAD `ffe27f4d`, not on main.
* **Paper-alignment**: P1 (Same Gen) + P2/P5 (skewed multiway) + P1/P2/P4/P5 (deep recursive); P3 explicitly excluded (W3.3-owned).
* **Risk register**: 5 entries with explicit mitigation paths.

Code may now be written.

---

## G2 — W5.1 implementation (Steps 2-7 from the plan)

**Goal**: Execute Steps 2-7 of the W5.1 iteration-1 plan. Each cert is TDD red/green per the plan; per-step supervisor review checkpoints are below.

### Process per cert

For each cert (Steps 2, 3, 4), follow the canonical TDD red/green cycle from `superpowers:test-driven-development`:

1. **RED**: write the failing test first; run; confirm it fails for the EXPECTED reason (not a compile error in unrelated code).
2. **GREEN**: write minimum code to make it pass; run; confirm it passes.
3. **Counter measurement**: run the cert under `--nocapture` and record the actual dispatch-counter value. If it differs from the plan's locked value (`== 1` for Same Gen + Skewed Multiway, `== 6` for Deep Recursive), STOP and request supervisor approval for a plan iteration 2 amendment per D2 lock. Do NOT silently adjust the plan or the assertion.
4. **Parity oracle**: assert non-empty CPU oracle output BEFORE the GPU comparison (D7 LOCKED).
5. **Commit**: one cert per commit. Subjects per the plan's Step 2-4 templates.

### Per-step REVIEW REQUEST checkpoints

After EACH cert (Steps 2, 3, 4 individually), post **"GOAL G2 STEP_N COMPLETE — REVIEW REQUEST"** with the measured counter, the cert's row-set size, and the cert's commit hash. Supervisor will validate and approve before the next step proceeds. This mirrors the W4.3 Step 6-10 cycle.

After all three certs land, execute Step 5 (Aggregate W5.1 targeted cert gate), Step 6 (Repository verification under the F-W43-12 + F-W43-15 exception), then Step 7 (Evidence + closure proposal). Post **"GOAL G2 COMPLETE — REVIEW REQUEST"** with full metric values after Step 7.

### Questions (Q2.1 - Q2.6)

* **Q2.1**: For each cert, does the dispatch-counter measured value match the plan's D2-locked value? (`== 1` / `== 1` / `== 6`).
* **Q2.2**: For each cert, does CPU oracle output match GPU output as `BTreeSet<Tuple>` parity AND is the CPU oracle output non-empty (D7)?
* **Q2.3**: Do all three new tests pass under the targeted Cargo command in G1 plan §"Global Verification Gates"?
* **Q2.4**: Does `cargo fmt --check --all` exit 0?
* **Q2.5**: Does `cargo test -p xlog-cuda-tests --test certification_suite --release` remain 1/1?
* **Q2.6**: Does the canonical workspace command exit 0 outside the three enumerated F-W43-12 + F-W43-15 exception files? If any non-exempt file fails, STOP and request supervisor review BEFORE adjusting scope.

### Metrics (M2.1 - M2.7)

* **M2.1**: 3 cert test files committed individually with commit subjects matching plan §Step 2/3/4. Each commit has a single test file added; no other source changes in the same commit.
* **M2.2**: 3/3 W5.1 certs PASS under the targeted command.
* **M2.3**: Workspace pass-count delta = +3 (one test per cert; if any cert opens multiple test fns inside one file, delta = +N as reported).
* **M2.4**: `cargo fmt --check --all` exit 0.
* **M2.5**: `RUSTFLAGS="-D warnings" cargo build --release --workspace --exclude pyxlog` exit 0.
* **M2.6**: CUDA cert suite 1/1.
* **M2.7**: Canonical workspace command per gate exception passes (only the three enumerated `test_wcoj_layout_*` files may flake; siblings + everything else must pass).

### Closure-proposal deliverable

Step 7 must produce `docs/plans/2026-05-11-w51-closure-proposal.md` (NOT under `docs/evidence/`; closure proposals live under `docs/plans/` per the W4.3 precedent at `docs/plans/2026-05-11-w43-closure-proposal.md`).

The closure proposal MUST:

* Anchor commit counts to a named commit (the closure-proposal commit itself), NOT live HEAD (per F-W43-18 lesson).
* Quote D7 + Acceptance Grid VERBATIM from the iteration-1 plan (per F-W43-13/F-W43-16 quote-from-canonical-source lesson).
* Enumerate measured metric values: per-cert counter values, per-cert row-set sizes, commit hashes, gate exit codes.
* Map the board's stated cert criterion (`Three new test files in xlog-integration/tests/; each asserts row-set parity vs. CPU oracle and dispatch counter > 0`) to the delivered work; explicitly note that W5.1 tightened `> 0` to exact-equality per F-W43-13.
* Raise three closure-board response options (Accept as DONE / Reject / Defer). Recommend Response 1 with reasoning if metrics all green.
* DO NOT mark DONE on the board. DO NOT FF-merge. DO NOT push or tag. Supervisor authorizes those after reviewing the closure proposal.

### Discipline reminders (re-iterated from goal 001)

* No `v0.6.6` in new files / comments / commit messages.
* No `git push`. No `git tag`.
* No FF-merge until supervisor authorizes per item.
* No board edit on `docs/v065-closure-board.md` until supervisor approves the closure proposal.
* No `--no-verify`, no `--dangerously-bypass-approvals-and-sandbox`, no force pushes.
* Bench-spike-first does NOT apply to W5.1 (certification work; no perf claim).
* Cert suite is the authoritative gate per MEMORY.md.
* `feedback_perf_bench_spike_first.md`: failed branches stay unmerged as evidence (relevant for W5.2 onward, not W5.1).
* F-W43-12 + F-W43-15 workspace-test exception is enumerated-files-only (three specific files); siblings must pass.

### Drift watch (supervisor concerns)

Per `superpowers:verification-before-completion`, codex must NOT claim success without running the actual gate commands. Each REVIEW REQUEST should cite the exit-status output AND the test-result line counts, not paraphrase them.

Per the W4.3 iteration-6 lessons (F-W43-13/16/18), watch for:

* Paraphrase drift when quoting the plan in the closure proposal — quote verbatim.
* Contract drift between the closure proposal and the actual cert assertions — grep the cert file for the assertion text and verify it matches the proposal text byte-for-byte.
* Live-HEAD-based commit counts that become invalid on the closure-proposal commit itself — anchor to a named commit.

Proceed with Step 2 first.
