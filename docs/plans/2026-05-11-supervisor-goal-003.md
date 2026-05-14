# Supervisor Goal 003 — W5.2 Stage 1 (Bench Spike) + Stage 2 (Plan Iter 1)

**Supervisor:** Claude Code.
**Implementer:** Codex CLI (tmux session `codex`, currently at `~/projects/xlog` on `main` HEAD `af5c85f4`).
**Predecessor:** Supervisor goals 001 + 002 (G0 tally fix + G1 W5.1 plan + G2 W5.1 implementation + closure follow-ups). W5.1 closed DONE on 2026-05-11 with 20/20 supervisor back-validation.
**Date:** 2026-05-11.
**Framing:** Goal-Driven SDP + GQM (Goal-Question-Metric) + GQM+Strategies. **New locked supervisor protocol** (post W5.1 back-validation): every step's REVIEW REQUEST triggers independent supervisor gate-runs + verbatim-quote diffs + file content audits BEFORE approval. No paraphrase acceptance.

---

## Acknowledgement: new locked supervisor protocol

W5.1 closed under a loose supervisor protocol — I (supervisor) approved codex's REVIEW REQUEST metrics without independently re-running the cited gate commands. The W5.1 work turned out correct on 20/20 back-validation, but the prior protocol violated `superpowers:verification-before-completion` ("NO COMPLETION CLAIMS WITHOUT FRESH VERIFICATION EVIDENCE"). The locked protocol is:

**Supervisor approval workflow per checkpoint:**

1. Codex posts `GOAL G_N STEP_M COMPLETE — REVIEW REQUEST` with self-reported metrics.
2. **Supervisor re-runs every cited gate command** in its own session, captures exit codes + result lines.
3. **Supervisor reads every cited test/bench/doc file** to verify assertion forms, fixture parameters, oracle references.
4. **Supervisor diffs verbatim quotes** against canonical sources (`diff -u` with `DIFF_EXIT=0` required).
5. **Supervisor cross-checks paper claims** P1-P5 mapping if invoked.
6. **Only then** does supervisor send approval. Findings (if any) trigger codex remediation BEFORE next step.

This adds ~5-15 minutes per checkpoint but it is non-negotiable. The previous error mode (trust without verification) is closed.

---

## Final outcome (load-bearing across all W5.2 goals)

W5.2 closed DONE on the v0.6.5 closure board with:

* Bench harness committed at `crates/xlog-integration/benches/w52_skewed_multiway_bench.rs`.
* Evidence README at `docs/evidence/2026-05-12-w52-skewed-multiway-bench/README.md` with **crossover thresholds vs binary-join** for at least three workload shapes (4-cycle, 5-clique, pivot-heavy multi-way).
* Spike branch `bench-spike/w52-skewed-multiway` preserved unmerged per `feedback_perf_bench_spike_first.md` (independent of acceptance / rejection at spike).
* W5.2 OPEN → DONE; tally DONE 11 → 12, OPEN 10 → 9.
* W2.5 unblocked (W5.2 was its remaining blocker after W3.2/W4.1 closed).
* FF-merge to local main on supervisor authorization; no push, no tag.

---

## Non-negotiable process rules (from `docs/v065-closure-board.md` §"Process Rules" + project memory; carried over from goals 001/002)

1. No DONE marking on any board item until supervisor approves in thread.
2. No `git push`. No `git tag`. W7.1 release is the only path to v0.6.5 tag.
3. No FF-merge until supervisor authorizes per item.
4. No `v0.6.6` reference in NEW files / comments / plans / commits.
5. **Bench-spike-first for any perf-claim work** (`feedback_perf_bench_spike_first.md`): minimum-viable spike measures gate target BEFORE full plan; failed spike branches stay unmerged as evidence.
6. Plan-iteration discipline: every amendment lands as a new iteration commit with F-WXY-N findings, before/after table, and process observation.
7. **Paper alignment**: P1-P5 from arXiv:2604.20073 constrain W3.x/W4.x. **W5.2 is bench-only**, does not change kernels — paper-alignment note should explicitly enumerate which Px each measured workload exercises (P2 count+materialize + P5 flat columnar are the likely candidates; P3 histogram-guided remains W3.3-owned and must NOT be claimed by W5.2).
8. Worktree per closure: `feat/w52-skewed-multiway-bench` under `.worktrees/` off main HEAD.
9. F-W43-12 + F-W43-15 workspace-test gate exception inherited: three enumerated `test_wcoj_layout_*` flake files (`fast_path`, `u32`, `u64`) exempt; siblings + everything else must pass.
10. F-W43-13/16/18 lessons: contract changes are file-wide concerns; quote canonical sources verbatim in derived docs; anchor commit counts in closure proposals to a named commit (not live HEAD).

---

## G3 — Stage 1: bench spike (proves gate target is achievable BEFORE full plan)

### Goal

Validate empirically that a bench harness CAN measure crossover thresholds (GPU WCOJ vs binary-join hash) for at least ONE of W5.2's three workload shapes. The spike's job is **falsifiability**, not closure — it produces evidence that either (a) supports proceeding to a full plan, or (b) surfaces a counter-finding that requires plan-amendment (cf. F-W43-2 PROVISIONAL pattern that closed W4.3).

### Strategies (GQM+Strategies "Strategy" tier)

* **Strategy S3.1**: One workload first, not three. Minimum-viable spike measures 4-cycle (W3.2 4cycle kernel already production-ready). 5-clique and pivot-heavy deferred to Stage 3 production-bench expansion.
* **Strategy S3.2**: Spike branch `bench-spike/w52-skewed-multiway` on a separate worktree `.worktrees/w52-bench-spike`. Branch stays unmerged regardless of outcome (per `feedback_perf_bench_spike_first.md`).
* **Strategy S3.3**: Provider-direct envelope-parity methodology (per W4.2/W4.3 spike precedent). NO executor-path timing for the spike — spike isolates kernel-level cost. Production bench at Stage 3 will follow W4.3's lesson: provider-direct + explicit detection cost where applicable.
* **Strategy S3.4**: Synthetic fixture varies skew + size to find crossover regime where GPU WCOJ wins vs binary-join hash AND where binary-join wins. The crossover IS the measurement; "GPU always wins" or "GPU never wins" both falsify the gate.

### Questions

* **Q3.1**: For the 4-cycle workload, at what `(L_count, R_count)` Cartesian-equivalent fixture size does GPU WCOJ beat binary-join hash by ≥ 1× (any positive ratio)? At what size does GPU win ≥ 2×?
* **Q3.2**: Is there a regime where binary-join wins (small fixtures, asymmetric shapes, high-skew distributions)? Define the crossover boundary.
* **Q3.3**: What's the binary-join's baseline cost for the same fixture (via existing `hash_join_v2`)? Provider-direct calls only.
* **Q3.4**: Does the spike kernel use existing W3.2 `wcoj_4cycle_*` paths, or does it construct a new code path? (Locked to existing — no new kernels in W5.2 per Strategy S3.3.)
* **Q3.5**: F-W43-2 PROVISIONAL pattern: what would constitute a counter-finding that requires plan amendment at Stage 2 vs proceeding straight to Stage 3?
* **Q3.6**: Paper-alignment: which paper claims (P2 count+materialize, P5 flat columnar) does the spike exercise? P3 (histogram) must NOT be claimed.

### Metrics

* **M3.1**: Bench-spike file at `.worktrees/w52-bench-spike/crates/xlog-integration/benches/w52_spike_4cycle_bench.rs` exists, compiles (`cargo bench --bench w52_spike_4cycle_bench --no-run` exit 0).
* **M3.2**: Spike runs to completion (`cargo bench --bench w52_spike_4cycle_bench` exit 0); evidence captured in `.worktrees/w52-bench-spike/docs/evidence/2026-05-11-w52-bench-spike/README.md`.
* **M3.3**: README contains: cell matrix (≥ 4 cells covering 50×50 to 2000×2000 or comparable range), median GPU WCOJ latency per cell, median binary-join hash latency per cell, speedup ratio per cell, crossover threshold (the `(L, R)` where ratio crosses 1.0× and 2.0×).
* **M3.4**: Spike branch HEAD anchored in evidence README; branch NEVER FF-merged to main regardless of outcome.
* **M3.5**: NO production kernel changes (`crates/xlog-cuda/kernels/` untouched, `crates/xlog-cuda/src/provider/` untouched). Spike is bench-only.
* **M3.6**: No `v0.6.6` references; no `git push`; no `git tag`. `git tag --points-at HEAD` returns 0.
* **M3.7**: Workspace gates clean per F-W43-12/15 exception: fmt exit 0; `-D warnings` workspace build exit 0; CUDA cert suite 1/1; targeted bench compile exit 0.

### Process

1. **Create spike worktree**: `git -C ~/projects/xlog worktree add .worktrees/w52-bench-spike -b bench-spike/w52-skewed-multiway af5c85f4`.
2. **Recon** (read-only): identify existing `wcoj_4cycle_*` provider entry points; cite line numbers. Identify existing bench precedents at `crates/xlog-integration/benches/wcoj_4cycle_bench.rs` (if it exists) and `w42_production_nested_loop_bench.rs`/`w43_production_sort_merge_bench.rs`.
3. **Draft spike bench file**: minimum-viable bench measuring 4-cycle GPU WCOJ vs binary-join hash on 4–6 cells. Criterion `sample_size=50`, `measurement_time=8s`, `warm_up_time=1s` (W4.3 precedent).
4. **Build + run spike**: compile + execute; capture median timings + ratios.
5. **Write evidence README**: cells, timings, ratios, crossover threshold, observed regime where binary-join wins (if any), observed regime where GPU wins (if any). NO benchmark-ratio CLAIM yet — the spike's role is empirical evidence, not closure-grade certification.
6. **Commit spike**: one or two commits on `bench-spike/w52-skewed-multiway`. NO commit on main. NO FF-merge.
7. **Post**: `GOAL G3 COMPLETE — REVIEW REQUEST` with measured cells, ratios, and recommended next action (proceed to Stage 2 plan iter 1 vs amend gate vs reject).

### Supervisor validation per locked protocol

When codex posts G3 REVIEW REQUEST, supervisor will independently:

* Re-run `cargo bench --bench w52_spike_4cycle_bench --no-run` from supervisor session; capture exit code.
* Re-run `cargo bench --bench w52_spike_4cycle_bench` from supervisor session; capture exit code + median timings.
* Read the bench file and verify: cell matrix matches codex's report, provider-direct calls only, no executor scaffolding.
* Read the evidence README and verify: numbers in tables match the actual bench output.
* Run `git -C ~/projects/xlog log --oneline main..bench-spike/w52-skewed-multiway`; verify NO commit on main.
* Run `git -C ~/projects/xlog tag --points-at bench-spike/w52-skewed-multiway`; verify no tag.
* `RUSTFLAGS="-D warnings" cargo build --release --workspace --exclude pyxlog` exit 0.
* Confirm spike branch NOT pushed (origin/main still at its pre-W5.1 commit + nothing under `origin/bench-spike/*`).

---

## G4 — Stage 2: W5.2 plan iteration 1 (full plan based on spike data)

Gated on G3 supervisor approval. Goal-spec for G4 will be written by supervisor AFTER reviewing G3 spike evidence; G4 details depend on what the spike measures.

Outline (preview, not committed):

* Worktree `.worktrees/w52-skewed-multiway-bench` on `feat/w52-skewed-multiway-bench` off `main` HEAD.
* Plan covers full bench harness expansion: 4-cycle (validated by spike) + 5-clique (W3.2-eligible) + pivot-heavy multi-way (new shape).
* Acceptance Grid per-shape with measured crossover thresholds.
* Paper P2/P5 cited per workload; P3 explicitly excluded (W3.3-owned).
* No production kernel changes; bench-only.
* Closure-board mapping: W5.2's stated gate ("crossover thresholds vs binary-join") satisfied iff evidence README documents thresholds for all three shapes.

---

## Forbidden behaviors (re-iterated; codex must reject any drift)

* No DONE marking on `docs/v065-closure-board.md` for W5.2.
* No `git push`, no `git tag`, no `--force`, no `--no-verify`, no `--dangerously-bypass-approvals-and-sandbox`.
* No `v0.6.6` references in new files / comments / commits.
* No FF-merge of `bench-spike/w52-skewed-multiway` to main. Spike stays unmerged per memory.
* No production kernel / provider changes during spike. Spike is bench-only.
* No paper claim P1 (W5.2 is non-recursive bench), no P3 (histogram-owned by W3.3), no P4 (delta-outermost is recursive-specific).
* No assertion of "W5.2 unblocks W2.5" inside the spike commit messages — that's a closure-proposal-level claim, not a spike-level claim.

---

## Communication protocol (carried over)

* Codex `/goal` pointed at this file.
* Codex posts `GOAL G3 COMPLETE — REVIEW REQUEST` after spike + evidence commit.
* Supervisor runs validation gates independently per the locked protocol above.
* If green: supervisor approves + writes goal-spec 004 (G4 = Stage 2 plan iter 1).
* If findings: supervisor surfaces them; codex remediates BEFORE supervisor sends approval; iterate.

Proceed with G3. Start with worktree creation and recon.
