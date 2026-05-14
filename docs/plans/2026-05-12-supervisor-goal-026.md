# Supervisor Goal 026 — W3.3 D7a Measurement on `6595b969` + Phase B Grid-Amortization Fix (Resume Path)

**Supervisor:** Claude Code.
**Implementer:** Codex CLI on tmux session `codex-xlog`.
**Predecessor:** G25 dispatched (combined RCA-5 + grid-amortization fix) but only Phase A probes landed at `7eb94bc2` before computer restart. User manually committed `6595b969 feat(w33): compute slice prefix on device during refresh` on `feat/w33-slice-aware-implementation` between sessions — substantive optimization moving slice-prefix computation from host to device. **G25's base (`dcb556db`) is now stale; the partial G25 branch (`7eb94bc2`) is superseded.** G26 resumes the path from the new best `6595b969`.
**Date:** 2026-05-12.

---

## Context

User directive (verbatim, recorded 2026-05-12): *"NO TOY SHIT! NO SIMPLIFICATIONS! NO WORKAROUND! NO SILENT BYPASSES! ONLY PRODUCTION GRADE CODE! ONLY DEEP ROOT CAUSES INVESTIGATIONS! ONLY PROPER AND HIGHEST QUALITY ALGORITHMIC AND STRUCTURAL FIXES!"*

Meta-goal: deliver FULL v0.6.5 per `docs/v065-closure-board.md` + `ROADMAP.md`. W3.3 is the first OPEN item; G26 closes it via D7a ≥ 2.0× empirical proof.

### State at `6595b969` (from `design_behavior_dump.txt` on `feat/w33-slice-aware-implementation`)

| Metric | Value |
|---|---|
| `baseline_grid_blocks` | 117 (`ceil(n_xy/256)`) |
| `merge_grid_blocks` | **468** (`= slice_count`) — STILL 4× more than baseline |
| `applied_grid_same` | `false` |
| Kernel symbols | `wcoj_triangle_count_sliced` + `wcoj_triangle_materialize_sliced` |
| `stored_histogram_contents` | `device_slice_starts+device_slice_ends` (real, device-resident) |
| Histogram weights | `min=2 max=4970 mean=833.6` (heavy-row variance real) |
| Per-block output stddev (baseline) | 459.998 |
| Per-block output stddev (merge) | 230.473 |
| Variance reduction | **49.897%** ✓ |

### What 6595b969 changed vs G23 `dcb556db`

| File | Change |
|---|---|
| `crates/xlog-cuda/kernels/wcoj.cu` | +38 lines — new device-side slice-prefix computation kernel |
| `crates/xlog-cuda/src/kernel_manifest_data.rs` | +2 lines — register new kernel |
| `crates/xlog-cuda/src/memory.rs` | +145 lines / -refresh body — refresh now launches device-side prefix kernel instead of CPU compute |
| `crates/xlog-cuda/src/provider/mod.rs` | +2 lines — provider wiring |
| design_behavior_dump.txt | updated to reflect device-side path |
| per_block_work.csv | updated 932 row delta from re-run |
| README.md | text patch (not updated to reflect device-side path — stale) |

This reduces refresh-pipeline overhead (no more column download → CPU compute → upload) but **does NOT change the kernel launch grid sizing**. The 468-block launch count is still in place. Per G24 measurement at `dcb556db`, that launch overhead consumed +1000 µs and resulted in 1.8× slowdown.

### Hypothesis space for G26

* **H1 — device-side prefix alone makes D7a pass.** If the +1000 µs at G24 was dominantly refresh-pipeline cost (column download + CPU compute + upload), then moving prefix to device should shave most of it. **Measure first.**
* **H2 — launch overhead still dominates.** If +1000 µs at G24 was mostly 4× kernel-launch driver overhead (~250 µs per launch × 4), then device-side prefix helps marginally and the grid-amortization fix is still needed.

G26 measures FIRST, then applies the Phase B fix only if H2 is confirmed. No speculative fix; data drives the decision.

---

## G26 — Measurement-first then Phase B grid-amortization fix if needed

### Goal

Cut `feat/w33-grid-amortized-v2` from `feat/w33-slice-aware-implementation @ 6595b969`. Phase 1: run V3 sample_size(200) Criterion measurement at superhub-50K on the CURRENT code (`6595b969` AS-IS). Phase 2: if D7a < 2.0×, apply grid-stride loop fix (kernel `_sliced` bodies + launch surface) and re-measure at superhub-50K (and 200K spot-check if 50K passes). Final commit must include evidence README with measurement results + fix-applied verdict + D7a verdict. Branch stays unmerged.

### Strategies (GQM+Strategies)

* **S26.1** Cut `feat/w33-grid-amortized-v2` from `feat/w33-slice-aware-implementation @ 6595b969`. Worktree at `.worktrees/w33-grid-amortized-v2`. The prior G25 branch `feat/w33-slice-aware-launch-amortized @ 7eb94bc2` stays preserved as historical evidence of the Phase A probe scaffolding.

* **S26.2 — Phase 1: Baseline measurement on `6595b969`.** Run V3 sample_size(200) Criterion at superhub-50K under the existing `wcoj_triangle_bench.rs` configuration. Capture baseline + merge-resident-sliced medians + 95% CI + paired delta + speedup ratio. Row-equality MUST PASS. Output Phase 1 measurement to `docs/evidence/2026-05-12-w33-grid-amortized-v2/phase1_baseline_measurement.tsv`.

* **S26.3 — Phase 1 decision gate.**
  * **If speedup ratio ≥ 2.0× at superhub-50K:** H1 confirmed. Mark Phase 1 PASS, skip Phase 2 (no fix needed), proceed to S26.6 evidence + commit. Optional spot-check 200K under same protocol.
  * **If speedup ratio < 2.0×:** H2 confirmed; Phase 2 required. Continue to S26.4.

* **S26.4 — Phase 2: Grid-amortization fix.** Apply grid-stride-over-slices loop to BOTH `_sliced` kernel bodies in `crates/xlog-cuda/kernels/wcoj.cu`:
  ```cuda
  for (int s = blockIdx.x; s < slice_count; s += gridDim.x) {
      int start = slice_starts[s];
      int end = slice_ends[s];
      for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
          // body identical to current _sliced body
      }
  }
  ```
  Modify launch surface in `crates/xlog-cuda/src/provider/wcoj.rs` (count + materialize sites) to set `grid_dim.x = ceil(n_xy / 256)` (or analogous baseline-equivalent), NOT `slice_count`. Slice plan still passed as device pointer + scalar.

* **S26.5 — Phase 2 re-measurement.** After fix applied, run identical V3 Criterion at superhub-50K. If ratio ≥ 1.5× and trending positive, ALSO run superhub-200K. Capture all to `docs/evidence/2026-05-12-w33-grid-amortized-v2/phase2_post_fix_measurement.tsv`.

* **S26.6 — Evidence README.** `docs/evidence/2026-05-12-w33-grid-amortized-v2/README.md` MUST contain:
  * All 16 parent SHAs explicit (main + G11–G25 + user's `6595b969`).
  * Diff summary: what changed in Phase 2 (if applied) with file:line citations.
  * Phase 1 measurement table: per-cell median + CI + paired delta + ratio + D7a verdict.
  * Phase 2 measurement table (if applied): same fields + comparison vs Phase 1.
  * Per-block-output stddev verification: post-fix stddev still < baseline 459.998 (work-balancing preserved).
  * Post-fix grid block count on superhub-50K (expect ≈ 117 if fix applied; ≈ 468 if no fix applied).
  * D7b spot-check uniform-u32-10K under V3: paired delta + verdict (expect unchanged from G23 because adaptive skew detection routes uniform → baseline kernel).
  * Closure-readiness verdict: "W3.3 closure-ready (D7a PASS at scale X)" / "needs further work (specific next-step)".

* **S26.7 — Existing test gates green:**
  * `cargo test --workspace --release --exclude pyxlog --exclude xlog-cuda-tests` EXIT 0.
  * `cargo test -p xlog-cuda-tests --test certification_suite --release` 1/1.
  * `cargo bench --no-run` EXIT 0.
  * `cargo build --release --features wcoj-phase-timing` EXIT 0.
  * `cargo build --release` (no feature) EXIT 0.
  * `cargo fmt --check --all` EXIT 0.

* **S26.8** Branch UNMERGED to all 17 parents (main + G11–G25 + `6595b969` + the partial G25 branch `7eb94bc2`). No FF-merge, no push, no tag.

* **S26.9** Commit structure (multi-commit allowed):
  1. (Phase 1 only) `feat(w33): phase 1 baseline measurement on 6595b969`
  2. (Phase 2 if needed) `feat(w33): phase 2 grid-stride loop + baseline grid_dim fix`
  3. (Phase 2 if needed) `feat(w33): phase 2 post-fix re-measurement`
  4. Final commit: evidence README + tables A/B.
  
  If Phase 1 PASSES, commits 2+3 are skipped; evidence README documents the H1-confirmed path.

* **S26.10 — R6 anti-pattern check.** Zero R6 anti-patterns: `git grep classify_heavy_rows` + `git grep mask_histogram` in new W3.3 code BOTH empty post-G26.

### Questions

* **Q26.1** Branch HEAD SHA?
* **Q26.2** Phase 1 superhub-50K under V3: baseline + merge medians + 95% CI + paired delta + speedup ratio + D7a verdict?
* **Q26.3** Phase 1 decision: pass (skip Phase 2) or fail (apply Phase 2)?
* **Q26.4** (If Phase 2 applied) Kernel modification: grid-stride loop confirmed at `kernels/wcoj.cu` file:line?
* **Q26.5** (If Phase 2 applied) Launch surface: `grid_dim.x = ceil(n_xy/256)` confirmed at `provider/wcoj.rs` file:line?
* **Q26.6** (If Phase 2 applied) Phase 2 superhub-50K measurement: same fields + D7a verdict + comparison vs Phase 1?
* **Q26.7** (If Phase 2 applied + 50K passes) Phase 2 superhub-200K: paired delta + ratio + D7a verdict?
* **Q26.8** Row-equality PASS on all measured scales?
* **Q26.9** Per-block-output stddev STILL strictly less than baseline 459.998 post-fix (work-balancing preserved)?
* **Q26.10** Grid block count post-fix on superhub-50K: ≈ 117 (baseline) if fix applied?
* **Q26.11** D7b spot-check uniform-u32-10K paired delta + verdict (should remain ≤ ±5% via adaptive skew detection)?
* **Q26.12** All existing test gates green?
* **Q26.13** Closure-readiness verdict?
* **Q26.14** Branch unmerged from all 17 parents?
* **Q26.15** Zero R6 anti-patterns?

### Metrics

* **M26.1** `feat/w33-grid-amortized-v2` exists; HEAD reachable from none of 17 parents.
* **M26.2** `docs/evidence/2026-05-12-w33-grid-amortized-v2/README.md` exists with measurement tables.
* **M26.3** `cargo bench --no-run` EXIT 0.
* **M26.4** Strict scientific control on Phase 2 (if applied): only `kernels/wcoj.cu` + `provider/wcoj.rs` modified; `memory.rs` + `recursive.rs` byte-identical to `6595b969`.
* **M26.5** Row-equality PASS at all measured scales.
* **M26.6** Phase 1 + Phase 2 (if applied) tables populated.
* **M26.7** Per-block-output stddev under fixed impl STILL < baseline 459.998.
* **M26.8** Grid block count post-fix on superhub-50K ≈ 117 (if Phase 2 applied).
* **M26.9** D7b uniform-u32-10K paired delta ≤ ±5%.
* **M26.10** All existing tests EXIT 0; CUDA cert 1/1.
* **M26.11** `cargo build --release --features wcoj-phase-timing` EXIT 0.
* **M26.12** `cargo build --release` EXIT 0.
* **M26.13** `cargo fmt --check --all` EXIT 0.
* **M26.14** `git tag --points-at HEAD` empty; `git ls-remote --heads origin "feat/w33*"` empty.
* **M26.15** Branch unmerged from all 17 parents.
* **M26.16** Zero R6 anti-patterns: `git grep classify_heavy_rows` + `git grep mask_histogram` in new W3.3 code BOTH empty.

### Supervisor validation per locked protocol

* Read evidence README (Phase 1 + Phase 2 if applied).
* `git rev-parse feat/w33-grid-amortized-v2` ≠ all 17 parent SHAs.
* Verify M26.4 strict scientific control on production-code change.
* Run `cargo build --release --features wcoj-phase-timing` AND `cargo build --release` from supervisor session.
* Run CUDA cert suite from supervisor session.
* Verify D7a verdict at the highest passing scale.
* Verify branch unmerged + no tag + no origin push.

If D7a ≥ 2.0× at any measured scale: G27 = closure proposal grounded in G11–G26 evidence chain + optional 1M sweep for completeness.

If D7a < 2.0× at all measured scales after Phase 2: G27 = deeper RCA (kernel-level profiling with Nsight, persistent-threads redesign, or alternative kernel architecture).

### Forbidden behaviors

* No `git push`, no `git tag`, no `--force`, no `--no-verify`, no `--dangerously-bypass`.
* No FF-merge into main.
* No `docs/v065-closure-board.md` edit (G27's conditional job).
* No `v0.6.6` references.
* **No R6 anti-patterns** (per-call histogram launch / heavy-light kernel split / per-call classify_heavy_rows / front-end mask_histogram+classify+partition_scan). Each measured-rejected per `f1142b3e`.
* No modification of existing `wcoj_triangle_count` or `wcoj_triangle_materialize` kernels (baseline uniform path). Modify only `_sliced` variants.
* No modification of `6595b969`'s device-side slice-prefix computation in `memory.rs` (it's the user's improvement; keep intact).
* No modification of the existing adaptive skew detection (uniform paths must remain on baseline kernel).
* No removal of row-equality assertions.
* No D7 amendment.
* No closure proposal in this goal.
* **No simplification or toy implementation.** The grid-stride loop is the standard CUDA idiom; apply it correctly with proper iteration semantics. If Phase 1 already passes, no Phase 2 needed (this is empirical efficiency, not simplification).
* No methodology change in the Criterion bench (V3 + iters=1 + paired-batching stays).

### Why this goal closes W3.3

The W3.3 chain has produced: paper-aligned plan, RC1+RC3 root cause attribution, slice-aware production implementation, work-balancing validation (49.85-49.90% stddev reduction), bench methodology (V3 sample_size 200 + iters=1 paired-batching), and a user-committed device-side prefix optimization. The remaining gap is the kernel-launch grid amortization. G26 either confirms the device-side prefix alone closes the gap (H1) or applies the textbook grid-stride loop fix (H2). One Criterion measurement decides which path.

Proceed: read goal-026, cut `feat/w33-grid-amortized-v2` from `6595b969`, run Phase 1 V3 measurement at superhub-50K, apply Phase 2 grid-amortization fix ONLY if Phase 1 D7a < 2.0×, capture all in evidence README with closure-readiness verdict, single or multi-commit (final = README), emit REVIEW REQUEST with HEAD SHA + measurement tables + D7a verdict + closure-readiness recommendation.
