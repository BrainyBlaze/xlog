# Supervisor Goal 032 — W3.4 Full Production Implementation (layout+count fusion + threshold auto-disable + acceptance cert grid)

**Supervisor:** Claude Code.
**Implementer:** Codex CLI on tmux session `codex-xlog`.
**Predecessor:** G31 bench spike on `bench-spike/w34-kernel-fusion @ 0276fd8d` PASSED at 1.491× on superhub-50K (≥1.3× gate). Candidate A (layout+count) selected. Spike branch stays unmerged as evidence per `feedback_perf_bench_spike_first.md`.
**Date:** 2026-05-13.

---

## Context

G31 proved:
- Layout+count fusion delivers 1.491× speedup on superhub-50K (29,539 rows row-equality PASS).
- Per-phase profile: layout 29.4% + count 19.8% = 49.2% wall (the fusion target).
- The fused kernel is correct (deterministic row-for-row output match) and feature-gate-compilable.
- The spike intentionally did not reroute production dispatch — that's G32's job.

W3.4 row from `docs/v065-closure-board.md @ 7bb56e4d`:

> | W3.4 | OPEN | — | Kernel fusion where benchmarks show materialization overhead dominates. Targeted candidates: layout + count, or count + materialize. | Bench: fused kernel shows **≥ 1.3× speedup** vs. 2-kernel sequence on a fixture where materialize is the long pole; deterministic. No regression on a small fixture where fusion penalty exceeds savings (must auto-disable below threshold). |

W3.4 acceptance has TWO sub-gates:
1. **Performance gate** — fused ≥ 1.3× on a long-pole-materialize fixture. G31 satisfied this at 1.491× on superhub-50K (note: superhub-50K is NOT a materialize-long-pole fixture — materialize is 24.4%, not dominant; layout+count is the long pole here, which is still a valid kernel-fusion target).
2. **Regression gate** — no regression on a small fixture where fusion penalty exceeds savings; **MUST auto-disable below threshold.**

G32 builds the production path that satisfies BOTH gates: live dispatch routes through fused kernel above a threshold, falls back to the existing 4-phase pipeline below it.

---

## G32 — Full W3.4 production implementation

### Goal

Cut `feat/w34-kernel-fusion-impl` from `main @ 7bb56e4d`. Promote the G31 spike fused kernel to a live production path with:

1. **Production dispatch entry** — fused path callable from executor without feature gate.
2. **Threshold-based dispatch policy** — heuristic decides fused-vs-unfused per call based on fixture characteristics (row count, slot widths, dedup heuristics). Threshold value is benchmark-grounded.
3. **Auto-disable below threshold** — small fixtures route to existing 4-phase pipeline; no regression.
4. **Acceptance cert grid** — tests pin: (a) above-threshold fixture routes to fused + row-equality holds, (b) below-threshold fixture routes to unfused + row-equality holds, (c) threshold sensitivity proof (one fixture above threshold + one below, both correct).
5. **Bench evidence** — V3 paired-batching at superhub-50K (above threshold) shows ≥1.3× ratio preserved; small fixture (below threshold) shows ratio within ±5% of unfused baseline (no regression).
6. **Branch stays unmerged** pending G33 closure proposal + user approval.

### Strategies (GQM+Strategies)

* **S32.1** Cut `feat/w34-kernel-fusion-impl` from `main @ 7bb56e4d`. Worktree at `.worktrees/w34-fusion-impl`.

* **S32.2 — Promote fused kernel to production path.**
  * In `crates/xlog-cuda/kernels/wcoj.cu`, the spike kernel `wcoj_triangle_fused_lc_count` is already a clean reusable artifact. Keep its name and signature; remove its `w34-spike-profiling` gate.
  * In `crates/xlog-cuda/src/provider/wcoj.rs`, rename or augment the G31 entry from `wcoj_triangle_fused_lc_u32_recorded` (under `#[cfg(feature = "w34-spike-profiling")]`) to a production entry `wcoj_triangle_fused_lc_u32_recorded` (no cfg gate). The signature contract is identical to G31 — caller provides pre-sorted/deduped inputs OR the entry sorts+dedups internally before invoking the fused count kernel. Pick whichever shape matches the existing live `wcoj_triangle_u32_recorded` call convention so the executor can route without convention change.
  * Add kernel manifest entry (already in G31 manifest; leave as-is since manifest is feature-agnostic).

* **S32.3 — Threshold-based dispatch policy.**
  * In `crates/xlog-runtime/src/executor/*` (locate the WCOJ triangle dispatch site — likely under a wcoj/triangle handler), add a threshold-based fork:
    ```
    if input_row_count >= W34_FUSION_THRESHOLD {
        provider.wcoj_triangle_fused_lc_u32_recorded(...)
    } else {
        provider.wcoj_triangle_u32_recorded(...)
    }
    ```
  * `W34_FUSION_THRESHOLD` is a const in `crates/xlog-runtime/src/config.rs` (or wherever `RuntimeConfig` lives). Initial value: grounded by S32.4 measurement; place a TODO-rationale comment pointing to evidence README.
  * Threshold input metric: total input row count across the three slots (count proxy for total work). NOT a max-row metric, NOT a dedup-rate-aware metric (those add complexity without G31-validated benefit). Document this choice in the threshold doc-comment.
  * Make threshold env-overridable: `XLOG_WCOJ_W34_THRESHOLD` env var overrides the const if set. Validation: parse as u32, fall back to const on parse failure with a `tracing::warn!`.
  * NO change to `CompilerConfig`, NO change to `CostModel`, NO change to any other dispatch policy. The W3.4 fork is the only new branching.

* **S32.4 — Threshold calibration measurement.**
  * Re-use the G31 bench harness (`crates/xlog-integration/benches/wcoj_fusion_bench.rs`) WITHOUT feature-gate. Drop the `--features w34-spike-profiling` requirement; the bench now exercises live dispatch.
  * Add ONE new "small fixture" Criterion case: `superhub-1K` (1,024 input rows — order of magnitude below 50K). NOTE: this is a NEW BENCH FIXTURE, which is allowed under G32 (not under G31). Generate via the same hub-pattern fixture generator that produced `superhub-50K`.
  * Sweep threshold candidates: 4096, 8192, 16384, 32768, 65536. At each candidate, measure baseline (unfused) vs fused-routed (with threshold forced to dispatch fused regardless) at both superhub-1K and superhub-50K. Pick the threshold where:
    * superhub-50K speedup stays ≥ 1.3× (preserves G31 perf gate).
    * superhub-1K speedup is ≥ 0.95× (no >5% regression at small fixture).
  * Document the chosen threshold + sweep data in evidence README.

* **S32.5 — Acceptance cert grid.**
  * New cert file: `crates/xlog-cuda-tests/tests/test_wcoj_w34_fusion.rs`.
  * Cert (A): "above-threshold routes to fused + correct" — fixture with input_row_count >= W34_FUSION_THRESHOLD invokes fused path (verified by a counter on provider, NOT by inspecting kernel name string), `download_triples` row-set equality vs unfused reference (compute reference by env-forcing threshold to u32::MAX).
  * Cert (B): "below-threshold routes to unfused + correct" — fixture with input_row_count < W34_FUSION_THRESHOLD invokes unfused path (counter verifies), `download_triples` row-set equality.
  * Cert (C): "env override works" — `XLOG_WCOJ_W34_THRESHOLD=u32::MAX` env on a 50K fixture forces unfused path; `XLOG_WCOJ_W34_THRESHOLD=0` env on a 1K fixture forces fused path; both row-equality.
  * Cert (D): "default threshold value" — assert the const equals the value picked by S32.4 measurement.
  * Cert (E): "no regression on existing W3.x acceptance" — slice-4 stable-triangle counter still 1; W3.1 sort-accessor + W3.2 K-clique priors unchanged; W2.5 default `RuntimeConfig::default().resolved_wcoj_cost_model() == Cardinality` unchanged; missing-stats safety floor still delegates to skew.
  * Existing CUDA cert suite (`cargo test -p xlog-cuda-tests --test certification_suite --release`) MUST stay 1/1 PASS.

* **S32.6 — Add provider counter for routing verification.**
  * On `CudaKernelProvider`, add `wcoj_triangle_fused_dispatch_count: AtomicU64` and `wcoj_triangle_unfused_dispatch_count: AtomicU64` with corresponding getter methods.
  * Increment in the runtime dispatch fork (S32.3). Used by certs (S32.5) for routing verification without string-matching kernel names.
  * Initialize both counters at provider construction; no `reset` method needed (test setup constructs fresh provider per test).

* **S32.7 — Bench evidence.**
  * `docs/evidence/2026-05-13-w34-kernel-fusion-impl/README.md` MUST contain:
    * Parent SHA `7bb56e4d`; branch HEAD SHA reported by REVIEW REQUEST.
    * S32.2 implementation diff summary (which files modified + LOC counts).
    * S32.4 threshold sweep table (per-candidate-threshold ratios at 50K + 1K) + chosen threshold + rationale.
    * S32.4 final V3 measurement at chosen threshold: superhub-50K ratio (must be ≥1.3×) + superhub-1K ratio (must be ≥0.95×).
    * S32.5 cert grid pass/fail (A through E + canonical W3.x priors).
    * Final gates (S32.8) all EXIT 0.

* **S32.8 — Final gates BEFORE the G32 commit.**
  * `cargo fmt --check --all` EXIT 0.
  * `RUSTFLAGS="-D warnings" cargo build --release --workspace --exclude pyxlog` EXIT 0 (NO feature flag — live dispatch must compile clean).
  * `cargo test -p xlog-cuda-tests --test certification_suite --release` 1/1.
  * `cargo test -p xlog-cuda-tests --test test_wcoj_w34_fusion --release` 5/5 (A through E).
  * Targeted W2.5/W3.1/W3.2/slice-4 regression: `cargo test --release -p xlog-cuda-tests --test test_wcoj_cardinality_cost_model --test test_wcoj_layout_u32 --test test_wcoj_clique5 --test test_wcoj_clique6 --test test_wcoj_record_join_result_feedback` all EXIT 0.
  * `cargo bench --no-run --bench wcoj_fusion_bench` EXIT 0 (no feature flag now).

* **S32.9** Branch UNMERGED to main + G30 + G31 ancestor branches. No FF-merge, no push, no tag.

* **S32.10 — Forbidden behaviors.**
  * No `git push`, no `git tag`, no `--force`, no `--no-verify`, no `--dangerously-bypass`.
  * No FF-merge into main.
  * No `docs/v065-closure-board.md` edit (board edits only via G33 closure proposal).
  * No `v0.6.6` references (per W1.1 rule #5).
  * No production code-path change outside the W3.4 fork. The triangle dispatch site adds exactly one if-else branch; all other WCOJ dispatch sites (slice 2 / 4-cycle, K5, K6, slice 4 recursive, sort accessors, layout fast-path) bit-identical to main.
  * No W3.4 marked DONE (G33's job, gated on user approval).
  * No threshold value picked without S32.4 sweep data.
  * No feature gate on the production dispatch fork (G32 promotes; feature gates are spike-only).
  * No CompilerConfig/CostModel touched.
  * No new bench fixtures beyond superhub-1K.

* **S32.11** Single bundled commit subject `feat(w34): production kernel fusion (layout+count) with threshold dispatch + auto-disable + cert grid`.

### Questions

* **Q32.1** Branch HEAD SHA?
* **Q32.2** Production fused entry promoted (file path + cfg-gate removed)?
* **Q32.3** Threshold dispatch fork added (file path + threshold metric + env override behavior)?
* **Q32.4** Threshold sweep table at 4K/8K/16K/32K/64K candidates with per-fixture ratios? Chosen threshold + rationale?
* **Q32.5** Final V3 measurement: superhub-50K ratio (must be ≥1.3×)? superhub-1K ratio (must be ≥0.95×)?
* **Q32.6** Cert grid A through E + canonical W3.x priors all PASS?
* **Q32.7** Provider counters `wcoj_triangle_fused_dispatch_count` + `wcoj_triangle_unfused_dispatch_count` exposed?
* **Q32.8** Final gates (fmt, warnings build, CUDA cert 1/1, W3.4 certs 5/5, priors green, bench compile) all EXIT 0?
* **Q32.9** No board/proposal diff? No `v0.6.6` strings?
* **Q32.10** Branch unmerged from main + G30 + G31 ancestors?

### Metrics

* **M32.1** `feat/w34-kernel-fusion-impl` exists; HEAD reachable from neither `main` nor G30/G31 ancestor branches.
* **M32.2** Evidence README exists with all sections + threshold sweep table + final V3 measurement.
* **M32.3** Per-fixture ratios: superhub-50K ≥ 1.3×; superhub-1K ≥ 0.95×.
* **M32.4** Cert grid 5/5 PASS (A through E).
* **M32.5** Canonical W3.x prior certs all PASS unchanged.
* **M32.6** `git diff main..HEAD -- docs/v065-closure-board.md docs/plans/` byte-empty.
* **M32.7** `git diff main..HEAD -- . | grep -c "v0.6.6\|R6"` = 0.
* **M32.8** No tag at HEAD; no `feat/w34*` remote ref.
* **M32.9** Branch unmerged from main.
* **M32.10** Final gates all EXIT 0 captured in evidence README.

### Supervisor validation per locked protocol

* Read evidence README end-to-end.
* `git rev-parse <branch>` ≠ `main` + ancestor branches.
* Run final gates from supervisor session: fmt + warnings build + CUDA cert + W3.4 certs + priors + bench compile.
* Inspect dispatch fork in executor to verify it's a single if-else + env-override; no broader rerouting.
* Inspect threshold sweep table for monotone-where-expected pattern (small threshold → small-fixture regression risk; large threshold → 50K perf loss).
* Verify branch unmerged + no tag + no origin push + no board diff.

### Decision branching after G32

* **If all M32 gates green:** dispatch G33 = W3.4 closure proposal. G33 stages a `docs/plans/2026-05-13-w34-closure-proposal.md` with: G31 spike evidence + G32 impl evidence + cert grid + bench evidence + W3.4 row mark OPEN→DONE staged for user approval per W1.1 rule #1. Then continue Path C: G34+ W3.5/W3.6/W3.7/W3.8/W3.9.
* **If S32.4 sweep shows NO threshold satisfies both gates simultaneously:** stop and emit empirical-gap finding to user — superhub-1K may be too aggressive a small-fixture proxy; option to define a different "small fixture" or amend W3.4 acceptance (per W1.1 rule #1, only with user approval).
* **If cert grid fails:** debug; do NOT commit until all 5 certs pass. The cert grid IS the W3.4 contract.

### Why this is structurally complete

The W3.4 acceptance text has two gates (perf + auto-disable-regression). G31 satisfied perf in isolation; G32 satisfies BOTH by routing through threshold logic that auto-disables small fixtures. The cert grid pins the routing behavior so future refactors can't accidentally remove the auto-disable branch. The env override gives operators a safety hatch.

Proceed: cut impl branch from main, promote fused kernel to production, add threshold dispatch fork with env override, sweep threshold candidates, pick threshold, build cert grid, write evidence README, run final gates, single bundled commit. Emit REVIEW REQUEST with HEAD SHA + threshold value + per-fixture ratios + cert grid pass count + final gates results.
