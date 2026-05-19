# Supervisor Goal 004 — W5.2 Stage 2 (Plan Iter 1) + Locked Multi-Run Discipline

**Supervisor:** Claude Code.
**Implementer:** Codex CLI.
**Predecessor:** Supervisor goal 003 → G3 spike + REFINEMENT. Spike branch `bench-spike/w52-skewed-multiway` HEAD `eacd3815`. Spike preserved unmerged per `feedback_perf_bench_spike_first.md`.
**Date:** 2026-05-11.

---

## G3 supervisor approval record

G3 + refinement APPROVED. Locked-protocol audit results:

* Branch isolation verified (spike on its own branch; main unchanged at `af5c85f4`).
* Scope verified (only 3 files: Cargo.toml + bench + README; zero kernel/provider changes).
* Multi-run reproducibility verified (12 cells × 3 runs = 36 measurements, all GPU 3/3, no direction flips).
* Raw TSV extraction matches README tables byte-for-byte.
* Counter-finding retracted explicitly with methodological lesson.
* Paper alignment correct (P2 + P5; P3 excluded as W3.3-owned).
* No push, no tag, no kernel changes, no v0.6.6, no DONE marking.
* Independent supervisor bench re-run produced hub_filtered N=1000 = 3.12× (within codex's run-1 3.11× — confirming single-run 0.61× was an anomaly).

The G3 refinement also delivered a **new process lesson** that locks into iteration-1 of the W5.2 plan: single bench invocations are NOT sufficient for crossover claims. All W5.2 plan acceptance gates must require multi-run min/median/max + win-direction-stability evidence.

---

## New locked process lesson (carries forward for all remaining perf items: W5.2 / W3.3-W3.6)

**LP-MULTI-RUN**: Any closure that makes a perf claim (speedup ratio, crossover threshold, monotonicity, win-direction) MUST be backed by ≥ 3 independent bench runs with:

1. Per-cell ratio captured for each run (raw criterion `estimates.json` extraction is the source of truth).
2. Per-cell `min / median / max` ratio across runs reported in the evidence README.
3. Per-cell win-direction stability documented (e.g., "GPU 3/3" or "mixed: 2/3 GPU, 1/3 hash").
4. Any cell that flips direction across runs is flagged as unstable; closure-grade claims cannot be made on unstable cells without a separate stabilization step (longer measurement_time, more samples, isolated GPU run).
5. Single-run claims that contradict the multi-run summary trigger plan-amendment, not silent revision.

This lesson generalizes the F-W43-2 PROVISIONAL pattern: empirical evidence may falsify a single measurement; iteration-N+ amends.

LP-MULTI-RUN applies to W5.2, W3.3, W3.4, W3.5, W3.6 closure work. W5.1 (cert-only, no perf claim) was exempt; W4.2 / W4.3 production benches used single-run criterion samples but had directionally consistent results across many cells — the new lock makes the multi-run guard explicit going forward.

---

## G4 — W5.2 Stage 2: plan iteration 1 (plan-only commit)

### Goal

Produce iteration-1 plan for the full W5.2 closure. Plan-only commit; no implementation. Spike data + retraction lesson + LP-MULTI-RUN feed forward as canonical inputs.

### Strategies (GQM+Strategies)

* **Strategy S4.1**: Expand from the spike's 1 workload (4-cycle) to all three W5.2-required shapes: 4-cycle + 5-clique (W3.2-eligible) + pivot-heavy multi-way (new shape; recon must identify a concrete representative).
* **Strategy S4.2**: Worktree `.worktrees/w52-skewed-multiway-bench` on `feat/w52-skewed-multiway-bench` off `main` HEAD `af5c85f4`.
* **Strategy S4.3**: LP-MULTI-RUN locked in plan D7 (acceptance gates): all crossover-threshold / ratio claims require ≥ 3 runs + min/median/max + win-direction-stability per cell.
* **Strategy S4.4**: Provider-direct envelope-parity methodology continues. No production kernel changes.
* **Strategy S4.5**: Closure-board cert criterion "Bench harness committed; evidence file with crossover thresholds vs. binary-join" interpreted as: 3 workloads × ≥ 3 runs × ≥ 4 cells per workload = ≥ 36 measurements minimum, with per-workload crossover threshold (or "no crossover observed in tested range" finding) recorded in evidence.

### Questions

* **Q4.1**: What does "pivot-heavy multi-way pattern" mean concretely? Board entry doesn't define it. Candidates: star-shape join (one center node many spokes), many-to-one chain, high-cardinality center column. Recon must pick one + cite paper §section if applicable.
* **Q4.2**: Does the 5-clique workload use W3.2's `wcoj_clique5_*` provider entries? Cite the exact provider fn + line numbers.
* **Q4.3**: For each workload, what's a deterministic fixture that produces non-trivial output without hitting the F-W43 production threshold (4M Cartesian)?
* **Q4.4**: Spike showed hub_filtered crosses the 2.0× threshold at the smallest cell (N=50). Does the full plan need to search BELOW N=50 to find a stable binary-win regime, or accept "GPU dominates the tested range" as the closure finding?
* **Q4.5**: What unblocks W2.5 specifically? Board entry §W2.5 says "remaining blockers are W3.2, W4.1, W5.1, W5.2"; W3.2/W4.1/W5.1 done. After W5.2, W2.5 should unblock — verify that mapping holds.
* **Q4.6**: Paper alignment for the three workloads: P2 + P5 for all three. NO P1 (non-recursive), NO P3 (W3.3-owned histogram), NO P4 (delta-outermost = recursive).

### Metrics

* **M4.1**: Plan file committed at `.worktrees/w52-skewed-multiway-bench/docs/plans/2026-05-11-w52-bench-plan.md` (or similar dated path) with header `iteration 1 canonical`.
* **M4.2**: Plan contains: Acceptance Line, Paper-Alignment Note, Process Rule Compliance (including LP-MULTI-RUN), Read-Only Surface (with provider/kernel line citations for 4cycle + clique5 + pivot-heavy targets), Direction Table D1-DN (≥ 7 locks including LP-MULTI-RUN), Step-by-Step Execution Plan (≥ 6 steps including spec → plan → spike-carry-forward → bench harness → 3-run measurement → evidence → closure proposal), Acceptance Grid (3 workloads × per-shape acceptance criteria), Source-of-Truth References (incl. spike branch `bench-spike/w52-skewed-multiway` HEAD `eacd3815`), Risk Register, Plan-Approval Gate.
* **M4.3**: D7 explicitly cites LP-MULTI-RUN with locked language: "all crossover/ratio claims require ≥ 3 runs + min/median/max + win-direction-stability per cell".
* **M4.4**: NO code commits. Plan-iteration commit only. Subject: `docs(plan): W5.2 iteration 1 — skewed multiway bench (4cycle + 5clique + pivot-heavy)`.
* **M4.5**: Worktree pre-created at `.worktrees/w52-skewed-multiway-bench` on branch `feat/w52-skewed-multiway-bench` off main HEAD `af5c85f4`.
* **M4.6**: Spike branch `bench-spike/w52-skewed-multiway` HEAD `eacd3815` cited in plan as canonical falsifiability-evidence reference.
* **M4.7**: Risk Register includes ≥ 5 entries with explicit mitigations, including: (a) pivot-heavy fixture definition risk; (b) GPU thermal/contention skew of single runs (mitigated by LP-MULTI-RUN); (c) 5-clique kernel cost dominates and changes crossover shape; (d) F-W43-12/15 + F-W52-1-equivalent flake exception accounting; (e) closure-board "crossover thresholds" requirement may need to be amended if no stable crossover exists in tested range (F-W43-2 PROVISIONAL pattern).

### Supervisor validation per locked protocol (when codex posts G4 REVIEW REQUEST)

* Re-run `cargo fmt --check --all`.
* Re-run `RUSTFLAGS="-D warnings" cargo build --release --workspace --exclude pyxlog` (plan-only commit shouldn't break anything).
* Read the plan file end-to-end.
* Verify D7 contains LP-MULTI-RUN locked language verbatim.
* Verify each of the 3 workloads has an acceptance grid row + paper-claim citation (P2 + P5 only).
* Verify the spike branch reference cites `eacd3815`.
* Verify no code commits via `git diff main..feat/w52-skewed-multiway-bench --stat` (only the plan file).
* Verify no push (`git ls-remote --heads origin "feat/w52*"` returns empty).
* Verify no tag (`git tag --points-at HEAD` returns empty).

### Process for codex

1. Create worktree `.worktrees/w52-skewed-multiway-bench` on `feat/w52-skewed-multiway-bench` off `main` HEAD `af5c85f4`.
2. Read-only recon: cite line numbers for `wcoj_4cycle_*`, `wcoj_clique5_*`, and any existing pivot-heavy bench / cert references.
3. Decide on a concrete "pivot-heavy multi-way pattern" — propose a definition with paper-§ citation if possible, OR cite the absence of paper guidance and propose a synthetic shape.
4. Draft plan iteration-1 with all sections per M4.2.
5. Commit plan-iteration commit per M4.4 on the worktree branch.
6. Post **"GOAL G4 COMPLETE — REVIEW REQUEST"** with measured M4.1-M4.7 values.

### Discipline reminders (locked, carry over)

* No DONE marking on board.
* No FF-merge, no push, no tag.
* No `v0.6.6` references.
* No production kernel/provider changes.
* No `--no-verify`, no force, no `--dangerously-bypass`.
* LP-MULTI-RUN applies to all subsequent G_N goals under W5.2 (G5 = implementation will require the 3-run pattern).
* F-W43-12 + F-W43-15 + observed `g04_transfer_efficiency` cert-suite flake (1-failure-then-pass pattern observed during W5.2 G3 supervisor validation) are inherited flake-exception classes; siblings + non-flake tests must pass.

Proceed with G4.
