# W2.5 Cost-Model Default-Flip Plan - iteration 1 canonical

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close W2.5 by flipping `RuntimeConfig::wcoj_cost_model` from the legacy `SkewClassifier` default to `Cardinality`, while preserving the slice-4 stable-triangle counter, the explicit `XLOG_WCOJ_COST_MODEL=skew` opt-out, and the W5.2 evidence link.

**Architecture:** W2.5 is a narrow runtime-configuration flip over the existing slice-5 `CardinalityAwareCostModel`; it must not add kernels, providers, executor routes, benchmark harnesses, or paper claims. The implementation changes the default-resolution branch, then adds regression certs that prove the missing-stats safety floor delegates to the legacy skew classifier and that env opt-out restores legacy behavior.

**Tech Stack:** Rust config in `xlog-core`, runtime cost-model factory and WCOJ dispatch in `xlog-runtime`, CUDA-backed integration tests in `xlog-integration`, W5.2 evidence under `docs/evidence/`.

---

## Acceptance Line

From `docs/v065-closure-board.md:78`:

`W2.5 | Internal | OPEN | — | Default-flip RuntimeConfig::wcoj_cost_model from SkewClassifier to Cardinality. Foundation + kernel + runtime + cert + benchmark evidence is now in hand: W2.1, W2.2, W2.3, W2.4, W3.2, W4.1, W5.1, and W5.2 are DONE. W5.2 supplies per-workload LP-MULTI-RUN direction-stability evidence for the default-flip decision: 4-cycle hub-filtered is GPU-favored, while 5-clique diagonal and pivot-heavy K5 are hash-favored in the tested ranges. | New default ships; slice 4 stable-triangle counter still 1 (cardinality + missing-stats safety floor delegates correctly); explicit env opt-out (XLOG_WCOJ_COST_MODEL=skew) restores legacy behavior; bench evidence from W5.2 documents the parity / improvement.`

The closure criterion has four sub-clauses:

1. New default ships: `RuntimeConfig::wcoj_cost_model` resolves to `CostModelKind::Cardinality` when no field or env override is set.
2. Slice-4 stable-triangle counter still equals 1: cardinality + missing-stats safety floor delegates to the skew model.
3. Env opt-out: `XLOG_WCOJ_COST_MODEL=skew` restores the legacy `SkewClassifier` behavior.
4. Bench evidence: W5.2's LP-MULTI-RUN corpus documents parity / improvement and is cited as the bench-spike-first input.

## Paper-Alignment Note

W2.5 makes **no new paper claim**. The paper-alignment basis is the already-shipped W2.1 cost-model surface:

- `crates/xlog-logic/src/wcoj_var_ordering.rs:1-14` defines W2.1's variable-ordering cost-model purpose and the `LeaderCardinalityModel` default-leader fallback semantics.
- `crates/xlog-logic/src/wcoj_var_ordering.rs:49-80` defines the `WcojVariableOrderingModel` trait.
- `crates/xlog-logic/src/wcoj_var_ordering.rs:82-107` defines the cardinality safety floor for compile-time leader selection.
- `crates/xlog-logic/src/compiler_config.rs:24-44` defines `WcojVarOrderingKind::{Disabled, LeaderCardinality, HeatAware}` and keeps W2.1 opt-in at compile time.

W2.5 only makes the existing runtime dispatch cost-model selector default to the existing `CardinalityAwareCostModel`. It must not claim P1, P3, or P4. P3 histogram-guided launch balancing remains W3.3-owned. W5.2 evidence remains P2/P5-only per `docs/evidence/2026-05-12-w52-skewed-multiway-bench/README.md:117-120` and `:360-362`.

## Process Rule Compliance

- Worktree: `.worktrees/w25-cost-model-default-flip`.
- Branch: `feat/w25-cost-model-default-flip`.
- Base: local `main` HEAD `8941c487`.
- Plan-only commit for G8. No Rust code, no tests, no evidence README, no board edit, no DONE marking.
- Bench-spike-first is satisfied by W5.2; do not create a new `bench-spike/w25-*` branch.
- No production kernel/provider/executor route changes in W2.5; the implementation is config/default-selection plus tests.
- No push, no tag, no FF-merge until separately authorized.
- Commit subject for this plan iteration: `docs(plan): W2.5 iteration 1 — cost-model default-flip (Skew → Cardinality)`.

## Read-Only Surface

### RuntimeConfig And Env Knob

- `RuntimeConfig` lives at `crates/xlog-core/src/config.rs:59-136`; `wcoj_cost_model: Option<CostModelKind>` is defined at `:131-135`.
- `CostModelKind::{SkewClassifier, Cardinality}` is defined at `crates/xlog-core/src/config.rs:138-154`.
- Current default sets `wcoj_cost_model: None` at `crates/xlog-core/src/config.rs:156-171`.
- Current precedence docs and builder are at `crates/xlog-core/src/config.rs:249-265`.
- Current resolver parses `XLOG_WCOJ_COST_MODEL` at `crates/xlog-core/src/config.rs:274-285`: field override wins, env value `cardinality` selects `Cardinality`, and every other value currently falls back to `SkewClassifier`.
- Existing config tests for env parsing and override precedence are at `crates/xlog-core/src/config.rs:362-435`.

### Runtime Cost-Model Factory And Dispatch Sites

- `CardinalityAwareCostModel` thresholds and decision rule are at `crates/xlog-runtime/src/executor/wcoj_cost_model.rs:266-310`.
- The missing-stats safety floor is implemented by `populated_cards` at `crates/xlog-runtime/src/executor/wcoj_cost_model.rs:329-344`.
- Triangle dispatch delegates to the skew fallback when stats are missing at `crates/xlog-runtime/src/executor/wcoj_cost_model.rs:362-386`.
- 4-cycle dispatch delegates the same way at `crates/xlog-runtime/src/executor/wcoj_cost_model.rs:388-404`.
- The factory that maps `RuntimeConfig` to the active cost model is at `crates/xlog-runtime/src/executor/wcoj_cost_model.rs:420-424`.
- Triangle adaptive dispatch calls the factory at `crates/xlog-runtime/src/executor/wcoj_dispatch.rs:916-945`.
- 4-cycle adaptive dispatch calls the same factory at `crates/xlog-runtime/src/executor/wcoj_dispatch.rs:1395-1423`.

### Missing-Stats Safety And Stats Feedback

- Unit tests already pin the safety-floor semantics at `crates/xlog-runtime/src/executor/wcoj_cost_model.rs:739-794`: missing stats and zero-cardinality inputs delegate, classifier `Err` / `Ok(None)` fall back.
- `StatsManager::estimate_join_cardinality` has a default estimate path at `crates/xlog-stats/src/manager.rs:238-296`, but W2.5's safety floor must keep using `CardinalityAwareCostModel::populated_cards` first so unknown stats do not silently become dispatch evidence.
- W2.4 feedback writes successful WCOJ results back into `StatsManager` at `crates/xlog-runtime/src/executor/wcoj_dispatch.rs:685-735` and `:752-794`; it skips unknown output rows and missing/zero input cards.
- Existing W2.4 integration coverage proves `record_join_result` is called when dispatch has seeded cards at `crates/xlog-integration/tests/test_wcoj_record_join_result_feedback.rs:345-374`, and proves missing input cards skip selectivity records at `:494-515`.
- W2.6 interaction: `record_wcoj_feedback` documents rotated feedback pairs for `var_order = Some(_)` at `crates/xlog-runtime/src/executor/wcoj_dispatch.rs:697-725`; W2.6 integration tests seed cards explicitly through `build_executor` at `crates/xlog-integration/tests/test_w26_heat_selectivity.rs:330-355`.

### Slice-4 Stable-Triangle Counter

- Primary slice-4 cert: `crates/xlog-integration/tests/test_wcoj_recursive_dispatch.rs:322-370`; the WCOJ-on run asserts `wcoj_triangle_dispatch_count() == 1` at `:359-365` and row-set parity at `:366-370`.
- Existing cardinality opt-in counterpart: `crates/xlog-integration/tests/test_wcoj_cardinality_cost_model.rs:376-401` currently proves legacy default preserves the same counter under the old default.
- Existing missing-stats integration cert: `crates/xlog-integration/tests/test_wcoj_cardinality_cost_model.rs:469-526` proves cardinality opt-in with no seeded stats delegates to the skew baseline with counter + row-set parity.

### W5.2 Evidence Input

- W5.2 cross-workload summary is at `docs/evidence/2026-05-12-w52-skewed-multiway-bench/README.md:339-350`: 36 measurements, 4-cycle GPU 12/12 with 2.1156x-7.0174x, 5-clique HASH 12/12 with 0.4945x-0.5945x, pivot-heavy K5 HASH 12/12 with 0.5365x-0.9098x.
- W5.2 acceptance status is at `docs/evidence/2026-05-12-w52-skewed-multiway-bench/README.md:352-358`: parity before timing, exact/non-empty output, LP-MULTI-RUN evidence, and stable threshold/no-crossover findings.
- W5.2 paper scope is at `docs/evidence/2026-05-12-w52-skewed-multiway-bench/README.md:117-120` and `:360-362`: P2/P5 only, no P1/P3/P4 claim.

## Direction Table

| ID | Lock | Direction |
|----|------|-----------|
| **D1** | **LOCKED: plan-only G8.** | G8 creates only this plan file. No implementation, no tests, no evidence README, no board edit, no DONE marking. |
| **D2** | **LOCKED: default ships by resolver change.** | W2.5 implementation changes `RuntimeConfig::resolved_wcoj_cost_model()` so unset field + unset env resolves to `CostModelKind::Cardinality`; do not add a new `RuntimeConfig` field or env var. |
| **D3** | **LOCKED: explicit config override still wins.** | `with_wcoj_cost_model(Some(...))` remains highest precedence. Existing config-field override tests must be updated only as needed to reflect the new default, not removed. |
| **D4** | **LOCKED: env opt-out is `skew`.** | `XLOG_WCOJ_COST_MODEL=skew` must resolve to `CostModelKind::SkewClassifier` and restore legacy dispatch behavior. Unrecognized env values remain conservative and resolve to `SkewClassifier`. |
| **D5** | **LOCKED: missing-stats safety floor delegates.** | `CardinalityAwareCostModel` must continue to call the skew fallback whenever any slot relation has missing or zero stats, and classifier `Err` / `Ok(None)` must never be overridden by cardinality. |
| **D6** | **LOCKED: slice-4 stable triangle stays exact.** | Under the new default, the stable-triangle recursive SCC cert must still assert `wcoj_triangle_dispatch_count() == 1` and row-set parity against the binary reference. |
| **D7** | **LOCKED: W5.2 is the bench-spike-first input.** | Do not create a W2.5 spike branch. Cite W5.2's 36-measurement corpus as the benchmark evidence for default-flip parity / improvement. |
| **D8** | **LOCKED: no new paper alignment.** | W2.5 inherits W2.1's cost-model basis and W5.2's P2/P5 evidence. No P1/P3/P4 claim. |
| **D9** | **LOCKED: W2.4/W2.6 interactions stay covered.** | The test sweep must include W2.4 feedback and W2.6 heat/selectivity certs because they share `StatsManager` and `var_order` feedback paths with the cardinality model. |
| **D10** | **LOCKED: closure remains proposal-gated.** | After implementation and evidence, write a closure proposal; do not edit the closure board or mark DONE until supervisor explicitly approves. |

## Step-By-Step Execution Plan

### Step 1: Plan Iteration 1

**Files:**
- Create: `docs/plans/2026-05-11-w25-default-flip-plan.md`

- [ ] **Step 1.1: Commit this plan-only artifact**

Run:

```bash
git add docs/plans/2026-05-11-w25-default-flip-plan.md
git commit -m "docs(plan): W2.5 iteration 1 — cost-model default-flip (Skew → Cardinality)"
```

Expected: one docs-only commit; `git diff main..HEAD --stat` shows only this plan.

### Step 2: Default-Flip Implementation

**Files:**
- Modify: `crates/xlog-core/src/config.rs`

- [ ] **Step 2.1: Write/adjust config unit tests first**

Update tests under `crates/xlog-core/src/config.rs:362-435`:

- Rename/update `cost_model_default_is_skew_classifier_when_unset` to assert `RuntimeConfig::default().resolved_wcoj_cost_model() == CostModelKind::Cardinality`.
- Add or adjust a test that sets `XLOG_WCOJ_COST_MODEL=skew` and asserts `CostModelKind::SkewClassifier`.
- Keep config-field override tests: explicit `Some(Cardinality)` and `Some(SkewClassifier)` still win over env.
- Keep garbage/unrecognized env resolving to `SkewClassifier`.

Run:

```bash
cargo test -p xlog-core --lib --release wcoj_cost_model
```

Expected before implementation: default test fails because current resolver returns `SkewClassifier`.

- [ ] **Step 2.2: Change the resolver default**

Change only `RuntimeConfig::resolved_wcoj_cost_model()` at `crates/xlog-core/src/config.rs:274-285`:

```rust
match normalized.as_deref() {
    Some("cardinality") => CostModelKind::Cardinality,
    Some("skew") | Some("skewclassifier") => CostModelKind::SkewClassifier,
    Some(_) => CostModelKind::SkewClassifier,
    None => CostModelKind::Cardinality,
}
```

Also update stale comments in `crates/xlog-core/src/config.rs:131-147` and `:249-261` so the docs say the production default is `Cardinality`, while explicit `skew` and unrecognized env values are conservative legacy opt-outs.

- [ ] **Step 2.3: Run the config tests green**

Run:

```bash
cargo test -p xlog-core --lib --release wcoj_cost_model
```

Expected: config tests pass with the new default and explicit `skew` opt-out.

Commit:

```bash
git add crates/xlog-core/src/config.rs
git commit -m "feat(w25): flip WCOJ cost-model default to cardinality"
```

### Step 3: Safety-Floor And Env-Opt-Out Certs

**Files:**
- Modify: `crates/xlog-integration/tests/test_wcoj_cardinality_cost_model.rs`

- [ ] **Step 3.1: Add a missing-stats default cert**

Adapt the existing opt-in test at `crates/xlog-integration/tests/test_wcoj_cardinality_cost_model.rs:469-526` so a new test runs:

1. Baseline with `RuntimeConfig::default().with_wcoj_cost_model(Some(CostModelKind::SkewClassifier))`.
2. New default with bare `RuntimeConfig::default()`.
3. No seeded stats in either run.
4. Assert the default run has the same `wcoj_triangle_dispatch_count()` and same `tri` rows as the explicit skew baseline.

This proves the new default's missing-stats safety floor delegates correctly without relying on force-mode.

- [ ] **Step 3.2: Add an env-opt-out integration cert**

Using the existing env-lock helper at `crates/xlog-integration/tests/test_wcoj_cardinality_cost_model.rs:332-360`, add a test that:

1. Sets `XLOG_WCOJ_COST_MODEL=skew`.
2. Runs bare `RuntimeConfig::default()` under no seeded stats.
3. Runs explicit `with_wcoj_cost_model(Some(CostModelKind::SkewClassifier))`.
4. Asserts counter + row-set parity.

This proves the process-global env opt-out restores legacy behavior after the default flip.

- [ ] **Step 3.3: Update existing default assertions**

Update `cardinality_default_off_keeps_slice4_dispatch_counts` at `crates/xlog-integration/tests/test_wcoj_cardinality_cost_model.rs:376-401` because the name and comment will be stale after the default flip. Preserve the `== 1` assertion for the force-gated stable triangle, or replace it with the stronger default-missing-stats test above if duplication becomes unclear.

- [ ] **Step 3.4: Run targeted integration certs**

Run:

```bash
cargo test -p xlog-integration --release --test test_wcoj_cardinality_cost_model -- --nocapture
```

Expected: all `test_wcoj_cardinality_cost_model` tests pass. If CUDA is unavailable, stop; W2.5 cannot close without real runtime evidence.

Commit:

```bash
git add crates/xlog-integration/tests/test_wcoj_cardinality_cost_model.rs
git commit -m "test(w25): certify cardinality default safety floor and skew opt-out"
```

### Step 4: Slice-4 Stable-Triangle Regression Cert

**Files:**
- Modify: `crates/xlog-integration/tests/test_wcoj_recursive_dispatch.rs`

- [ ] **Step 4.1: Add/adjust bare-default stable-triangle assertion**

The existing slice-4 cert at `crates/xlog-integration/tests/test_wcoj_recursive_dispatch.rs:322-370` uses explicit force-on for the WCOJ run. Add a W2.5-specific default-config run, or extend the existing cert carefully, so bare `RuntimeConfig::default()` under the new default still dispatches exactly once on the stable triangle seeding pass when stats are missing.

Expected assertion:

```rust
assert_eq!(default_exec.wcoj_triangle_dispatch_count(), 1);
assert_eq!(default_rows, reference_rows);
```

Use the same fixture and binary reference already present in the test. Do not weaken the existing force-on assertion.

- [ ] **Step 4.2: Run the slice-4 cert**

Run:

```bash
cargo test -p xlog-integration --release --test test_wcoj_recursive_dispatch stable_triangle_in_recursive_scc_dispatches_wcoj_on_seeding -- --nocapture
```

Expected: the test passes with the new bare-default branch and preserves `== 1`.

Commit:

```bash
git add crates/xlog-integration/tests/test_wcoj_recursive_dispatch.rs
git commit -m "test(w25): lock slice-4 stable-triangle default counter"
```

### Step 5: Regression Sweep

**Files:**
- No new files unless a failing prior requires a narrowly scoped test patch.

- [ ] **Step 5.1: Run W2.1 / W2.4 / W2.6 targeted priors**

Run:

```bash
cargo test -p xlog-integration --release --test test_w21_variable_ordering -- --nocapture
cargo test -p xlog-integration --release --test test_wcoj_record_join_result_feedback -- --nocapture
cargo test -p xlog-integration --release --test test_w26_heat_selectivity -- --nocapture
```

Expected: all targeted prior suites pass. These cover W2.1 row-set/leader parity, W2.4 feedback into `StatsManager`, and W2.6 heat/selectivity interaction with feedback pairs.

- [ ] **Step 5.2: Run runtime/core cost-model unit tests**

Run:

```bash
cargo test -p xlog-core --lib --release wcoj_cost_model
cargo test -p xlog-runtime --lib --release wcoj_cost_model
```

Expected: both pass.

- [ ] **Step 5.3: Commit evidence README if needed**

If supervisor requires implementation evidence before closure proposal, create `docs/evidence/2026-05-12-w25-default-flip/README.md` with exact command outputs and scope notes. If all evidence can live in the closure proposal, skip this commit and do not create a redundant evidence file.

### Step 6: Closure Proposal

**Files:**
- Create: `docs/plans/2026-05-12-w25-closure-proposal.md`

- [ ] **Step 6.1: Draft closure proposal**

Include:

- Commit list anchored to the closure-proposal commit per F-W43-18 precedent.
- The four W2.5 acceptance sub-clauses and their evidence.
- W5.2 evidence summary and citation as bench-spike-first satisfaction.
- Paper-alignment note: no new claim; W2.1 cost-model trait is the basis; W5.2 evidence is P2/P5-only.
- Final gates with exit codes.
- Response options Accept / Reject / Defer, Response 1 recommended if all gates pass.

Commit:

```bash
git add docs/plans/2026-05-12-w25-closure-proposal.md
git commit -m "docs(w25): add default-flip closure proposal"
```

### Step 7: Final Gates

Run sequentially before posting closure review:

```bash
cargo fmt --check --all
RUSTFLAGS="-D warnings" cargo build --release --workspace --exclude pyxlog
cargo test -p xlog-core --lib --release wcoj_cost_model
cargo test -p xlog-runtime --lib --release wcoj_cost_model
cargo test -p xlog-integration --release --test test_wcoj_cardinality_cost_model -- --nocapture
cargo test -p xlog-integration --release --test test_wcoj_recursive_dispatch stable_triangle_in_recursive_scc_dispatches_wcoj_on_seeding -- --nocapture
cargo test -p xlog-integration --release --test test_w21_variable_ordering -- --nocapture
cargo test -p xlog-integration --release --test test_wcoj_record_join_result_feedback -- --nocapture
cargo test -p xlog-integration --release --test test_w26_heat_selectivity -- --nocapture
cargo test -p xlog-cuda-tests --test certification_suite --release
cargo test --workspace --release --exclude pyxlog --exclude xlog-cuda-tests
```

Expected:

- All targeted tests pass.
- CUDA cert suite passes 1/1.
- Canonical workspace command exits 0 or consumes only the previously enumerated F-W43-12/F-W43-15 layout-file exception, with a targeted non-exempt sweep if any exception is consumed.

## Acceptance Grid

| Sub-clause | Planned evidence | Gate |
|---|---|---|
| New default ships | `RuntimeConfig::default().resolved_wcoj_cost_model() == CostModelKind::Cardinality`; factory selects `CardinalityAwareCostModel` for bare default. | `cargo test -p xlog-core --lib --release wcoj_cost_model`; runtime integration certs using bare `RuntimeConfig::default()`. |
| Slice-4 stable-triangle counter still `== 1` | W2.5 branch of `stable_triangle_in_recursive_scc_dispatches_wcoj_on_seeding` runs bare default and asserts `wcoj_triangle_dispatch_count() == 1` plus row-set parity. | `cargo test -p xlog-integration --release --test test_wcoj_recursive_dispatch stable_triangle_in_recursive_scc_dispatches_wcoj_on_seeding -- --nocapture`. |
| Env opt-out restores legacy | `XLOG_WCOJ_COST_MODEL=skew` resolves to `SkewClassifier` and an integration cert matches explicit skew baseline counter + row set. | `cargo test -p xlog-core --lib --release wcoj_cost_model`; `cargo test -p xlog-integration --release --test test_wcoj_cardinality_cost_model -- --nocapture`. |
| W5.2 bench evidence documents parity / improvement | Closure proposal cites W5.2 36-measurement corpus: 4-cycle GPU 12/12, 5-clique HASH 12/12, pivot-heavy K5 HASH 12/12, zero direction flips. | Source audit of `docs/evidence/2026-05-12-w52-skewed-multiway-bench/README.md:339-358` and closure proposal review. |

## Source-Of-Truth References

- Closure board W2.5 row: `docs/v065-closure-board.md:78`.
- Runtime config selector: `crates/xlog-core/src/config.rs:59-136`, `:138-154`, `:156-171`, `:249-285`.
- Runtime cost-model implementation: `crates/xlog-runtime/src/executor/wcoj_cost_model.rs:266-310`, `:329-344`, `:362-404`, `:420-424`.
- Runtime dispatch factory call sites: `crates/xlog-runtime/src/executor/wcoj_dispatch.rs:916-945`, `:1395-1423`.
- Slice-4 stable-triangle cert: `crates/xlog-integration/tests/test_wcoj_recursive_dispatch.rs:322-370`.
- Existing cardinality safety certs: `crates/xlog-runtime/src/executor/wcoj_cost_model.rs:739-794`; `crates/xlog-integration/tests/test_wcoj_cardinality_cost_model.rs:469-526`.
- W2.4 feedback interaction: `crates/xlog-runtime/src/executor/wcoj_dispatch.rs:685-735`, `:752-794`; `crates/xlog-integration/tests/test_wcoj_record_join_result_feedback.rs:345-374`, `:494-515`.
- W2.6 heat/selectivity interaction: `crates/xlog-integration/tests/test_w26_heat_selectivity.rs:330-355`, `:1060-1115`, `:1140-1160`.
- W5.2 evidence: `docs/evidence/2026-05-12-w52-skewed-multiway-bench/README.md:117-120`, `:339-358`, `:360-362`.

## Risk Register

| ID | Risk | Mitigation |
|----|------|------------|
| F-W25-1 | Missing-stats default flip could crash or over-dispatch because `StatsManager::estimate_join_cardinality` has default cardinalities. | Keep `CardinalityAwareCostModel::populated_cards` as the first gate; add a bare-default missing-stats integration cert proving delegation to explicit skew baseline. |
| F-W25-2 | `XLOG_WCOJ_COST_MODEL=skew` could stop restoring legacy behavior after default flip. | Add core resolver test and integration env-opt-out cert with env lock; explicit `Some(CostModelKind::SkewClassifier)` remains highest precedence. |
| F-W25-3 | Slice-4 stable triangle could change dispatch count under bare default. | Extend `test_wcoj_recursive_dispatch` to assert bare-default `wcoj_triangle_dispatch_count() == 1` plus row-set parity. |
| F-W25-4 | W2.4 feedback or W2.6 heat/selectivity paths could interact with the new default through shared `StatsManager` state. | Run W2.4 and W2.6 targeted suites; keep feedback writer skip conditions and rotated-pair semantics unchanged. |
| F-W25-5 | Closure-board "parity / improvement" could be overstated because W5.2 has mixed per-workload directions. | State the mixed W5.2 findings honestly: 4-cycle GPU-favored, 5-clique and pivot-heavy hash-favored in tested ranges. Default flip uses the existing cardinality model, not a per-workload threshold claim. |
| F-W25-6 | Existing tests that assume "default is skew" could become stale and pass for the wrong reason if only comments are changed. | Search for default-skew assumptions; update assertions and test names, and run `rg "default.*SkewClassifier|default.*skew|default cost model"` before final gates. |
| F-W25-7 | Env-var tests may race because `XLOG_WCOJ_COST_MODEL` is process-global. | Use the existing env-lock/snapshot helpers in `xlog-core` and `test_wcoj_cardinality_cost_model`. |
| F-W25-8 | Plan or implementation could accidentally add new paper claims. | Closure proposal must include the no-new-claim note and cite W2.1 + W5.2 P2/P5-only evidence. |

## Plan-Approval Gate

G8 stops after this plan-only commit. Implementation requires supervisor approval of:

- Worktree and branch match S8.1.
- Plan file path and header match M8.1.
- This plan contains all required sections from M8.2.
- Direction table includes the four locked acceptance sub-clauses.
- Risk register has at least five `F-W25-*` entries.
- W5.2 evidence is cited as bench-spike-first satisfaction.
- `git diff main..feat/w25-cost-model-default-flip --stat` shows only `docs/plans/2026-05-11-w25-default-flip-plan.md`.
- No push, no tag, no FF-merge, no board edit, no DONE marking.
