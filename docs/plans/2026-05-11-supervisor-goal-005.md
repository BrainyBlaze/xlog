# Supervisor Goal 005 — W5.2 Stage 3 Implementation (Plan Steps 2–5)

**Supervisor:** Claude Code.
**Implementer:** Codex CLI.
**Predecessor:** Supervisor goal 004 → G4 plan iter-1. Plan at `.worktrees/w52-skewed-multiway-bench/docs/plans/2026-05-11-w52-bench-plan.md` (commit `c2e7aaf4` on `feat/w52-skewed-multiway-bench`). Supervisor-approved under the locked protocol: 7/7 M4.x metrics + paper alignment + 7-entry Risk Register + fmt + warnings build.
**Date:** 2026-05-11.

---

## G4 supervisor approval record

G4 APPROVED. Independent gate audit results:

* Plan file 234 lines, plan-only commit, no `crates/` changes.
* D7 LP-MULTI-RUN locked verbatim: *"all crossover/ratio claims require ≥ 3 runs + min/median/max + win-direction-stability per cell"*.
* Acceptance Grid covers 4-cycle + 5-clique + pivot-heavy K5 with per-workload acceptance criteria + cell minimum + LP-MULTI-RUN evidence column.
* Risk Register: **7 F-W52-N entries** including F-W52-4 (incorporating my supervisor-side `g04_transfer_efficiency` cert-flake finding) and F-W52-6 (criterion `new/estimates.json` overwrite between runs — operational detail for LP-MULTI-RUN execution).
* Paper alignment correct: P2 + P5 only; P3 explicitly remains W3.3-owned even for pivot-heavy fixtures (D9).
* No remote push of `feat/w52-skewed-multiway-bench`; no tag.

Code may now be written.

---

## G5 — W5.2 implementation Steps 2–5 (bench skeleton + 3 workloads)

### Goal

Execute Steps 2–5 of the plan: bench skeleton + 4-cycle workload + 5-clique workload + pivot-heavy K5 workload. Per-step REVIEW REQUEST checkpoints after each workload (Steps 3, 4, 5). Steps 6–7 (aggregated evidence + closure proposal + final gates) covered by supervisor goal 006.

### Per-step process (every workload)

1. **Bench skeleton arrives first** (Step 2): provider + criterion setup + extraction tooling that snapshots `new/estimates.json` medians per run into a stable TSV (per F-W52-6).
2. **Per-workload TDD-like cycle** (Steps 3, 4, 5):
   1. Add workload-specific fixture + GPU WCOJ path + binary hash baseline as separate bench groups.
   2. Pre-cell parity check (BTreeSet row-set equality) BEFORE any timing.
   3. Run 3 sequential bench invocations; snapshot medians per run.
   4. Compute per-cell ratio min/median/max; classify win-direction stability.
   5. Commit per workload with subjects from plan §Steps 3/4/5.
   6. Post **"GOAL G5 STEP_N COMPLETE — REVIEW REQUEST"** with per-cell ratio table + direction stability per cell + commit hash.

### Questions (per workload, must be answerable from the REVIEW REQUEST)

* **Q5.1**: Does each cell have row-set parity (GPU vs binary hash) BEFORE timing? (D3 lock.)
* **Q5.2**: How many independent runs were captured? Must be ≥ 3 per cell (D7 LP-MULTI-RUN).
* **Q5.3**: Per cell, what's the ratio min/median/max?
* **Q5.4**: Per cell, what's the win-direction stability (e.g., "GPU 3/3", "mixed 2/3", "unstable")?
* **Q5.5**: Are there any unstable cells? If yes, are they reported as unstable per D7?
* **Q5.6**: Does the per-workload paper claim stay within P2/P5? (No P1/P3/P4.)
* **Q5.7**: For pivot-heavy K5, do the binary hash baseline's first-join steps reflect "pivot-incident joins first" per D-table lock?

### Metrics

* **M5.1**: Step 2 bench skeleton committed; subject from plan §Step 2.
* **M5.2**: Step 3 (4-cycle) committed; subject `bench(w52): measure 4-cycle crossover`; ≥ 4 cells × 3 runs in evidence captured.
* **M5.3**: Step 4 (5-clique) committed; subject `bench(w52): measure 5-clique crossover`; ≥ 4 cells × 3 runs.
* **M5.4**: Step 5 (pivot-heavy K5) committed; subject `bench(w52): measure pivot-heavy multiway crossover`; ≥ 4 cells × 3 runs.
* **M5.5**: NO production kernel/provider changes (`crates/xlog-cuda/kernels/` and `crates/xlog-cuda/src/provider/` untouched).
* **M5.6**: NO `RuntimeConfig` field additions, NO env knobs, NO new public APIs beyond what's needed for the bench.
* **M5.7**: All 3 workloads' raw extraction files (per-run TSVs) preserved at `/tmp/w52_*` for supervisor sanity-check.
* **M5.8**: NO forbidden tokens (v0.6.6, deferred, future-slice, ABANDONED) in any committed file.
* **M5.9**: NO push to `origin/feat/w52-skewed-multiway-bench`.
* **M5.10**: NO tag.

### Supervisor validation per locked protocol (per REVIEW REQUEST checkpoint)

For EACH of Steps 3, 4, 5 (after codex posts the REVIEW REQUEST), supervisor will:

1. Re-run `cargo bench -p xlog-integration --bench <bench-name> --no-run` from supervisor session.
2. Re-run the bench ≥ 1 time and extract medians via `target/criterion/.../new/estimates.json`.
3. Compare supervisor's run-1 medians to codex's reported run-3 medians; flag any cell where the ratio direction differs across the supervisor + codex 4 total measurements.
4. Read the committed bench file source; verify fixture matches plan-locked parameters, parity check is BEFORE timing, only provider-direct calls are used.
5. Diff `git diff main..feat/w52-skewed-multiway-bench --stat`; verify ONLY plan + bench file + Cargo.toml lines.
6. Verify no `crates/xlog-cuda/kernels/` or `crates/xlog-cuda/src/provider/` changes via `git diff --name-only main..feat/w52-skewed-multiway-bench | grep -E "kernels/|provider/" || echo NO_CHANGES_OK`.
7. Verify no remote push: `git ls-remote --heads origin "feat/w52*"` empty.
8. Verify no tag: `git tag --points-at <commit>` empty.

If any check fails: supervisor surfaces finding; codex remediates; only then approval.

### Forbidden behaviors (locked)

* No `git push`, no `git tag`, no `--force`, no `--no-verify`, no `--dangerously-bypass`.
* No production kernel / provider / executor changes. W5.2 is bench-only per plan D1.
* No board edit, no DONE marking — those land in supervisor-goal-007 after Step 7 closure-proposal review.
* No FF-merge until full G5 + G6 + supervisor closure-proposal review complete.
* No claim of P1/P3/P4 alignment in any committed bench/evidence file. P2/P5 only.
* No silent assertion adjustment if a measured ratio contradicts the plan's hypothesis. Per F-W43-2 PROVISIONAL pattern: any contradiction triggers plan iteration-2 amendment BEFORE assertion change.
* No `v0.6.6` references.

### Drift watch (supervisor concerns from W4.3/W5.1/W5.2-spike experience)

* **GPU thermal/contention skew**: if codex's run 1 is dramatically slower than runs 2/3 (as we saw in the G3 refinement), this is GPU warm-up not a measurement issue. Plan accordingly.
* **F-W52-6 criterion estimates.json overwrite**: snapshot per run; if a run's medians are not captured before the next run starts, the run is LOST. Codex must snapshot immediately.
* **F-W52-5 no-crossover finding is valid**: if all 3 runs show GPU 3/3 across all cells, "no crossover observed in tested range" IS a valid closure finding per D8. Do not invent a phantom threshold.
* **Per-cell stability MATTERS more than the headline ratio**: a cell with min=0.95×, median=1.2×, max=3.0× is unstable; cannot support a closure claim without stabilization.

Proceed with Step 2 first (bench skeleton). Then Steps 3, 4, 5 sequentially, posting REVIEW REQUEST after each.
