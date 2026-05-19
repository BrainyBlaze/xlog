# Supervisor Goal 008 — W2.5 Stage 1 (Plan Iter 1, Plan-Only)

**Supervisor:** Claude Code.
**Implementer:** Codex CLI.
**Predecessor:** Supervisor goal 007 → W5.2 closure complete on `main` HEAD `8941c487`. W2.5 transitioned BLOCKED → OPEN as part of that closure. All four W2.5 blockers (W3.2 / W4.1 / W5.1 / W5.2) are now DONE.
**Date:** 2026-05-11.
**Framing:** GQM + GQM+Strategies under the **locked supervisor protocol**: every checkpoint triggers supervisor-side gate runs + verbatim diffs + file content audits BEFORE approval. No paraphrase acceptance.

---

## Acceptance Line (from `docs/v065-closure-board.md` W2.5 row, just updated by G7)

> W2.5 | Internal | OPEN | — | Default-flip `RuntimeConfig::wcoj_cost_model` from `SkewClassifier` to `Cardinality`. Foundation + kernel + runtime + cert + benchmark evidence is now in hand: W2.1, W2.2, W2.3, W2.4, W3.2, W4.1, W5.1, and W5.2 are DONE. W5.2 supplies per-workload LP-MULTI-RUN direction-stability evidence for the default-flip decision: 4-cycle hub-filtered is GPU-favored, while 5-clique diagonal and pivot-heavy K5 are hash-favored in the tested ranges. | New default ships; slice 4 stable-triangle counter still 1 (cardinality + missing-stats safety floor delegates correctly); explicit env opt-out (`XLOG_WCOJ_COST_MODEL=skew`) restores legacy behavior; bench evidence from W5.2 documents the parity / improvement.

The closure criterion has four sub-clauses; the plan must satisfy each:

1. **New default ships**: `RuntimeConfig::wcoj_cost_model` default = `Cardinality`.
2. **Slice 4 stable-triangle counter still == 1**: cardinality + missing-stats safety floor delegates correctly.
3. **Explicit env opt-out**: `XLOG_WCOJ_COST_MODEL=skew` restores legacy behavior.
4. **Bench evidence**: W5.2's per-workload corpus documents the parity / improvement.

## Bench-spike-first satisfaction

W2.5 is a perf-claim closure (the default-flip implicitly claims cardinality is at least at parity on representative workloads). **Per `feedback_perf_bench_spike_first.md`, bench-spike-first applies — but the spike is already done.** W5.2 IS the spike-equivalent for W2.5: it provides 36 LP-MULTI-RUN measurements across 3 workloads with per-shape direction-stability findings. The W2.5 plan cites W5.2 evidence directly; no separate `bench-spike/w25-*` branch is created.

The W5.2 evidence supports the default-flip as follows:
* 4-cycle hub-filtered: GPU 12/12 (2.12×–7.02×) → cardinality model should prefer GPU dispatch for 4-cycle.
* 5-clique diagonal: HASH 12/12 (0.49×–0.59×) → cardinality model should NOT mandate GPU dispatch for 5-clique diagonal (binary-hash wins).
* Pivot-heavy K5: HASH 12/12 (0.55×–0.91× rising) → similar to 5-clique; near-parity at high N suggests a fixture-dependent threshold.

The default-flip ships a model that admits all three findings. Implementation does NOT need to encode all per-workload thresholds; the model's existing cardinality logic (from W2.1) is what's being made default. W2.5 is purely about the default value of the existing logic.

---

## G8 — W2.5 plan iteration 1 (plan-only commit)

### Goal

Produce iteration-1 plan for the W2.5 default-flip closure. Plan-only commit; no implementation. The plan must enumerate the four acceptance sub-clauses + locked process rules + safety-floor strategy.

### Strategies (GQM+Strategies)

* **S8.1**: New worktree `.worktrees/w25-cost-model-default-flip` on `feat/w25-cost-model-default-flip` off `main` HEAD `8941c487`.
* **S8.2**: Default-flip is a **single-symbol code change** in `xlog-core` (or wherever `RuntimeConfig` lives). The plan must locate the precise definition site + cite line numbers.
* **S8.3**: **Safety floor for missing-stats**: when stats are empty (e.g., first dispatch, no `record_join_result` feedback yet), cardinality model must fall back to a safe default that doesn't crash the runtime. Plan must specify what "safe default" means; cite the existing W2.4 missing-stats handling or define a new fallback policy.
* **S8.4**: **Env-opt-out preservation**: `XLOG_WCOJ_COST_MODEL=skew` must still parse + apply correctly after the flip. Plan must cite where the env-knob parsing lives.
* **S8.5**: **Slice-4 regression safety**: the slice-4 stable-triangle counter test (`crates/xlog-integration/tests/...?`) must still pass with the new default. Plan must cite the test file + the counter assertion line.
* **S8.6**: **Test coverage additions**: (a) safety-floor cert (missing-stats case under new default), (b) env-opt-out cert (XLOG_WCOJ_COST_MODEL=skew restores prior behavior), (c) parity cert (existing slice-4 + W2.1 + W2.6 tests still pass).
* **S8.7**: **Paper alignment**: W2.5 does NOT claim new paper alignment. The cardinality model itself is W2.1-owned (paper-aligned with P2 count+materialize choice driven by cardinality). W2.5's default-flip simply makes the existing paper-aligned path default. No P1/P3/P4 claim; P5 implicit through the W2.1 cost-model trait.

### Questions

* **Q8.1**: Where is `RuntimeConfig::wcoj_cost_model` defined? Cite the file + line + current default value.
* **Q8.2**: Where is the env-knob `XLOG_WCOJ_COST_MODEL` parsed? Cite the file + line + the existing parse logic (likely a `match` on string values).
* **Q8.3**: Where is the slice-4 stable-triangle counter test? Cite the file + line + the assertion `== 1`.
* **Q8.4**: What does the cardinality model do when stats are missing? Audit the existing W2.1 / W2.3 / W2.6 code paths for missing-stats handling. Is there already a safety floor? If yes, cite it; if no, propose one.
* **Q8.5**: Does W2.4's `record_join_result` feedback path interact with the default-flip? (W2.4 wires GPU dispatch counters back to `StatsManager`; cardinality model reads from `StatsManager`; cycle.)
* **Q8.6**: Does W2.6's `HeatAwareLeaderModel` interact with the default-flip? (W2.6 adds heat/selectivity to variable ordering; same cost model surface.)
* **Q8.7**: What's the regression-test cell matrix for the new default? (Slice-4 stable-triangle + W2.1 acceptance grid + W2.2 selectivity-pass + W2.4 record-feedback + W2.6 heat-selectivity certs.)

### Metrics

* **M8.1**: Plan file committed at `.worktrees/w25-cost-model-default-flip/docs/plans/2026-05-11-w25-default-flip-plan.md` with header `iteration 1 canonical`.
* **M8.2**: Plan contains: Acceptance Line, Paper-Alignment Note (cite W2.1's cost-model paper alignment + W5.2 bench evidence as input), Process Rule Compliance, Read-Only Surface (with file:line citations for RuntimeConfig + env knob + slice-4 test + missing-stats handling), Direction Table D1-DN (≥ 6 locks), Step-by-Step Execution Plan (≥ 5 steps including spec → plan → impl-default-flip → impl-safety-floor-cert → impl-env-opt-out-cert → regression-test-sweep → closure proposal → final gates), Acceptance Grid (per-sub-clause), Source-of-Truth References (incl. W5.2 evidence at `docs/evidence/2026-05-12-w52-skewed-multiway-bench/README.md`), Risk Register (≥ 5 entries), Plan-Approval Gate.
* **M8.3**: D-table includes the four locked sub-clauses (default-ships, slice-4-counter-still-1, env-opt-out-works, W5.2-evidence-cited).
* **M8.4**: NO code commits. Plan-only commit. Subject: `docs(plan): W2.5 iteration 1 — cost-model default-flip (Skew → Cardinality)`.
* **M8.5**: Worktree created at `.worktrees/w25-cost-model-default-flip` on branch `feat/w25-cost-model-default-flip` off `main` HEAD `8941c487`.
* **M8.6**: W5.2 evidence cited as bench-spike-first satisfaction reference (NOT a new bench-spike branch).
* **M8.7**: Risk Register includes ≥ 5 F-W25-N entries: (a) safety-floor missing-stats crash; (b) env-opt-out regression; (c) slice-4 counter regression; (d) W2.4/W2.6 cost-model-trait interaction; (e) closure-board "parity / improvement" interpretation if W5.2 evidence shows mixed results.

### Supervisor validation per locked protocol (when codex posts G8 REVIEW REQUEST)

* Re-run `cargo fmt --check --all` from supervisor session.
* Re-run `RUSTFLAGS="-D warnings" cargo build --release --workspace --exclude pyxlog` (plan-only commit shouldn't break anything).
* Read the plan file end-to-end.
* Verify each of the four W2.5 acceptance sub-clauses has an explicit plan-section mapping (D-table lock OR Acceptance Grid row).
* Verify W5.2 evidence is cited in Paper-Alignment + Read-Only Surface + Source-of-Truth References.
* Verify Risk Register has ≥ 5 F-W25-N entries.
* Verify no code commits via `git diff main..feat/w25-cost-model-default-flip --stat` (only the plan file).
* Verify no push (`git ls-remote --heads origin "feat/w25*"` empty).
* Verify no tag (`git tag --points-at HEAD` empty).
* Verify cited line numbers exist by spot-checking 2-3 of them.

### Process for codex

1. Create worktree `.worktrees/w25-cost-model-default-flip` on `feat/w25-cost-model-default-flip` off `main` HEAD `8941c487`.
2. Read-only recon: grep / read for `RuntimeConfig`, `wcoj_cost_model`, `XLOG_WCOJ_COST_MODEL`, `SkewClassifier`, `Cardinality`, slice-4 stable-triangle counter assertions, missing-stats safety paths.
3. Decide on safety-floor strategy + cite existing W2.4 missing-stats handling (or propose new).
4. Draft plan iteration-1 with all sections per M8.2.
5. Commit plan-iteration commit per M8.4 on the worktree branch.
6. Post **"GOAL G8 COMPLETE — REVIEW REQUEST"** with measured M8.1-M8.7 values.

### Forbidden behaviors

* No DONE marking, no FF-merge, no push, no tag.
* No `v0.6.6` references in new files / commits.
* No production kernel/provider/executor changes in G8 (G8 is plan-only).
* No `--force`, `--no-verify`, `--dangerously-bypass`.
* No new bench-spike branch (W5.2 satisfies bench-spike-first).
* No paper claim P1/P3/P4 by W2.5 (W2.5 is config flip, not new paper alignment).

Proceed with G8 (worktree + recon + plan iter-1).
