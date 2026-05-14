# Supervisor Goal 009 — W2.5 Stage 2 (Plan Steps 2-5)

**Supervisor:** Claude Code.
**Implementer:** Codex CLI.
**Predecessor:** G8 plan iter-1 at `56685fa3` on `feat/w25-cost-model-default-flip`. Supervisor-approved under locked protocol: all M8.x metrics + spot-checked line citations + D1-D9 LOCKED + 8 F-W25-N risks + 4 sub-clauses mapped + W5.2 evidence cited.
**Date:** 2026-05-11.

---

## G8 supervisor approval record

G8 APPROVED. Independent locked-protocol audit:

* 1 plan-only commit (+383 lines) on `feat/w25-cost-model-default-flip`.
* Plan structure: Acceptance Line + Paper-Alignment + Process Rule Compliance + Read-Only Surface + D-table D1-D9 LOCKED + 7-Step Execution Plan + Acceptance Grid + Source-of-Truth + 8 F-W25 Risk Register + Plan-Approval Gate.
* Cited line numbers spot-checked: `crates/xlog-core/src/config.rs:131-135` matches `pub wcoj_cost_model: Option<CostModelKind>`; slice-4 cert `:359-365` matches `assert_eq!(dispatched.wcoj_triangle_dispatch_count(), 1, ...)`.
* W5.2 evidence cited as bench-spike-first input; no new spike branch.
* No code commits; no push; no tag; no `v0.6.6` references.

Code may now be written.

---

## G9 — W2.5 implementation Steps 2-5

### Goal

Execute Steps 2-5 of the W2.5 plan: default-flip implementation + safety-floor cert + env-opt-out cert + slice-4 regression cert + regression sweep. Per-step REVIEW REQUEST checkpoints after Steps 2, 3, 4 (Step 5 is a regression sweep — bundled with Step 4's REVIEW REQUEST). Steps 6-7 (closure proposal + final gates) covered by supervisor goal 010.

### Strategies (GQM+Strategies)

* **S9.1**: **Step 2 is a single-symbol code change** in `RuntimeConfig::resolved_wcoj_cost_model()`. The default branch (no field override + no env override) returns `CostModelKind::Cardinality` instead of `CostModelKind::SkewClassifier`. Per D2 LOCKED: no new `RuntimeConfig` field, no new env var.
* **S9.2**: **TDD red/green for the default-flip**: write a cert asserting `RuntimeConfig::default().resolved_wcoj_cost_model() == CostModelKind::Cardinality`; run cert (RED on current code); flip the resolver; run cert (GREEN).
* **S9.3**: **Safety-floor cert (Step 3)**: re-confirm the existing `test_wcoj_cardinality_cost_model.rs:469-526` "missing-stats delegates to skew" cert still passes under the new default. If not, the safety floor needs refinement.
* **S9.4**: **Env-opt-out cert (Step 3)**: cert that `XLOG_WCOJ_COST_MODEL=skew` env var resolves to `CostModelKind::SkewClassifier` AND results in skew-classifier dispatch behavior matching pre-flip.
* **S9.5**: **Slice-4 stable-triangle regression cert (Step 4)**: run existing `test_wcoj_recursive_dispatch::stable_triangle_in_recursive_scc_dispatches_wcoj_on_seeding` under the new default; counter MUST still be `== 1` per D6 LOCKED.
* **S9.6**: **Regression sweep (Step 5)**: run W2.1/W2.2/W2.3/W2.4/W2.6 cert suites; verify no test breaks.
* **S9.7**: **F-W43-12/15 exception** inherited: workspace test gate exempts only the three enumerated `test_wcoj_layout_*` files; siblings + everything else must pass.

### Questions (per step checkpoint)

* **Q9.1 (Step 2)**: Does `RuntimeConfig::default().resolved_wcoj_cost_model() == CostModelKind::Cardinality` hold after the flip? What's the exact diff for the resolver change?
* **Q9.2 (Step 3)**: Does the existing missing-stats safety-floor cert at `test_wcoj_cardinality_cost_model.rs:469-526` still pass with the new default? If yes, is the cert exercising the bare-default code path (not the explicit `with_wcoj_cost_model(Some(Cardinality))` override)?
* **Q9.3 (Step 3)**: Does `XLOG_WCOJ_COST_MODEL=skew` env cert fire correctly under the new default? Add an integration cert if one doesn't exist.
* **Q9.4 (Step 4)**: Does `stable_triangle_in_recursive_scc_dispatches_wcoj_on_seeding` still assert `wcoj_triangle_dispatch_count() == 1` under the new default? Run the test and capture the exact counter value.
* **Q9.5 (Step 5)**: Do W2.1/W2.2/W2.3/W2.4/W2.6 acceptance suites all pass under the new default? Capture per-suite pass/fail counts.
* **Q9.6**: Did any test require modification because it explicitly hard-coded `SkewClassifier` as a default expectation? If yes: was the modification minimal (just the default expectation) or invasive (test logic changed)?

### Metrics

* **M9.1**: Step 2 commit subject: `feat(w25): default-flip wcoj_cost_model resolver to Cardinality`. Single code change in `crates/xlog-core/src/config.rs` resolver + 1 unit test added/updated in same file. NO new `RuntimeConfig` field, NO new env var.
* **M9.2**: Step 3 commit subject: `test(w25): cert safety-floor + env-opt-out under new default`. New or updated tests covering: (a) missing-stats safety floor delegates correctly; (b) `XLOG_WCOJ_COST_MODEL=skew` restores legacy behavior.
* **M9.3**: Step 4 commit subject: `test(w25): assert slice-4 stable-triangle counter == 1 under new default`. Test may be a new integration cert OR an update to the existing slice-4 cert that runs it under `RuntimeConfig::default()` (no explicit cost-model override).
* **M9.4**: Step 5 (regression sweep): no commit (verification only). Run W2.1+W2.2+W2.3+W2.4+W2.6 cert suites; capture pass/fail counts. Posted as part of Step 4's REVIEW REQUEST.
* **M9.5**: NO production kernel/provider changes (`crates/xlog-cuda/kernels/`, `crates/xlog-cuda/src/provider/`, runtime executor/dispatch logic untouched except the cost-model factory which is configured via RuntimeConfig).
* **M9.6**: NO board edit, NO DONE marking, NO FF-merge, NO push, NO tag in any G9 commit.
* **M9.7**: All gates green: fmt, warnings-as-errors workspace build, CUDA cert suite, targeted W2.5 cert run, F-W43-12/15-narrowed canonical workspace test.

### Supervisor validation per locked protocol (per step checkpoint)

For EACH of Steps 2, 3, 4 (after codex posts REVIEW REQUEST), supervisor will:

1. Re-run `cargo fmt --check --all` from supervisor session.
2. Re-run `RUSTFLAGS="-D warnings" cargo build --release --workspace --exclude pyxlog`.
3. Run the targeted test(s) from supervisor session; capture exit code + counter assertion line.
4. Read the committed source change; verify minimal scope (just the default-flip in resolver; no new fields/env-vars).
5. Verify `git diff main..feat/w25-cost-model-default-flip --name-only`: only `crates/xlog-core/src/` and `crates/xlog-integration/tests/` should appear.
6. Verify no kernel/provider changes via `git diff --name-only main..feat/w25-cost-model-default-flip | grep -E "kernels/|provider/" || echo NO_CHANGES_OK`.
7. Verify no push: `git ls-remote --heads origin "feat/w25*"` empty.
8. Verify no tag: `git tag --points-at HEAD` empty.

After Step 4 + Step 5 (regression sweep), supervisor will additionally:
* Run W2.1/W2.2/W2.3/W2.4/W2.6 cert suites from supervisor session; compare pass/fail counts to codex's report.
* Run CUDA cert suite (authoritative gate).
* Run canonical workspace test under F-W43-12/15 enumerated exception.

### Forbidden behaviors

* No `git push`, no `git tag`, no `--force`, no `--no-verify`, no `--dangerously-bypass`.
* No new `RuntimeConfig` field. No new env var. No new cost-model variant.
* No board edit, no DONE marking — those land in supervisor-goal-010/011 after closure-proposal approval.
* No FF-merge until goal-011.
* No paper claim P1/P3/P4. W2.5 inherits W2.1's cost-model basis + W5.2's P2/P5 evidence.
* No `v0.6.6` references.
* No silent assertion adjustment: if a regression test fails because it hard-coded `SkewClassifier` as the default, the fix is to update the expectation (minimal change), NOT to retain the old default.

### Drift watch (supervisor concerns)

* **Missing-stats safety floor**: per plan D5 LOCKED, `CardinalityAwareCostModel` must call the skew fallback when stats are missing/zero, AND classifier `Err`/`Ok(None)` must never be overridden by cardinality. Codex must NOT silently change this fallback behavior while making the default-flip.
* **W2.4 / W2.6 interaction**: per plan D9 LOCKED, the regression sweep must include W2.4 (record_join_result feedback) and W2.6 (heat-aware leader) certs. If those fail under the new default, codex must surface them; do not silently amend them.
* **W2.1 / W2.3 interaction**: W2.1's variable-ordering cost-model trait and W2.3's per-iteration recursive cardinality update share `StatsManager` with the W2.5 cost model. Any test that depends on the default cost-model behavior must be updated explicitly, not silently.

Proceed with Step 2 first (default-flip implementation + initial cert).
